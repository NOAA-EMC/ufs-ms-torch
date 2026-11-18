# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""CCPP ``mp_tempo`` / ``mp_tempo_post`` wrapper semantics over the torch-native TEMPO runtime.

This follows ``MP/TEMPO/mp_tempo.F90:mp_tempo_run`` and
``mp_tempo_post.F90:mp_tempo_post_run``
around the unchanged TEMPO v3 core, for the configuration the oracle GFS v17
suite uses: no aerosol-aware and no hail-aware fields (the optional
``nc/nwfa/nifa/ng/volg`` arrays are absent) and no ``land_input``.

Per call (``mp_tempo.F90:430-575``):

* ``ndt = max(nint(dtp/dt_inner), 1)``, ``dt = dtp/ndt``, raised to ``dt_inner``
  when ``dt <= dt_inner``;
* ``qv = q/(1-q)`` from specific humidity; with ``convert_dry_rho`` every
  hydrometeor mass/number ratio is divided by ``1-q``;
* ``dz`` from the interface geopotential ``phii``;
* every inner step: ``rho`` from ``prsl``, the current temperature and ``qv``
  (TEMPO ``roverrv``/``rdry``); ``w = -omega/(rho*g)``; ``tempo_run``;
  precipitation accumulated in m (``/1000``) with ``snow`` including ice and
  ``prcp`` the sum of all four;
* ``sr`` from the last inner step, ``refl_10cm`` copied only when
  ``do_radar_ref``;
* back to specific humidity / moist ratios (``/(1+qv)``) and tendencies
  ``(new-old)/dtp``; ``ten_u = ten_v = 0``.

The TEMPO import-time configuration must be what ``mp_tempo_init`` sets up: the
CCPP host constants (``TEMPO_TORCH_HOST_CONSTANTS=ufs_physcons``) and
``tempo_init(hailaware_flag=.false.)`` (``TEMPO_TORCH_HAIL_AWARE_INIT=0``), see
``segments/_host_constants.py``; the entry raises otherwise. Semi-Lagrangian
sedimentation follows the Fortran operation order.

The sedimentation loops use the static substep bound from ``min_layer_depth``,
the host grid's thinnest layer in m (a grid property known when the caller
prepares the mesh), and fall-speed ceilings (``max_sedimentation_fall_speed`` for rain
and graupel, ``max_snow_sedimentation_fall_speed`` for snow; default the worst case
``torch_native.DEFAULT_MAX_SEDIMENTATION_FALL_SPEED``). Any column whose Fortran
substep count would exceed the loop is flagged in
``diagnostics["sedimentation_substep_overflow"]`` without a host read, and the
bound's overflow region (``torch_native.SEDIMENTATION_BOUND_*``) is set, so a driver
reruns the step with a higher ceiling instead of keeping a truncated result.

The compaction and substep-region settings of ``torch_native`` are module state, so a process runs one
TEMPO configuration at a time; ``mp_tempo_run`` sets them for its call and restores them afterwards.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from mpas_pytorch import _torch_compat as torch_compat
from ufs_physics_common import compaction
from ufs_physics_common.compile_policy import stored

from tempo_torch import torch_native
from tempo_torch import host_constants as _host_constants
from tempo_torch.coefficients import constants
from tempo_torch.parameters import TABLE_SPECS


CON_G = 9.80665  # physcons.F90 con_g
# MPAS tracer table of the oracle lane: qv qc qr qi qs qg ni nr sgs_tke o3mr.
MPAS_NTRAC = 10
MPAS_TRACER_INDEX: Mapping[str, int] = {
    "ntqv": 1,
    "ntcw": 2,
    "ntrw": 3,
    "ntiw": 4,
    "ntsw": 5,
    "ntgl": 6,
    "ntinc": 7,
    "ntrnc": 8,
}
SPECIES = ("qc", "qr", "qi", "qs", "qg", "ni", "nr")
TENDENCY_SLOTS: Mapping[str, str] = {
    "dspechum": "ntqv",
    "dqc": "ntcw",
    "dqr": "ntrw",
    "dqi": "ntiw",
    "dqs": "ntsw",
    "dqg": "ntgl",
    "dni": "ntinc",
    "dnr": "ntrnc",
}


@dataclass(frozen=True)
class MpTempoResult:
    tendencies: dict[str, torch.Tensor]
    ten_q: torch.Tensor
    ten_t: torch.Tensor
    ten_u: torch.Tensor
    ten_v: torch.Tensor
    prcp: torch.Tensor
    rain: torch.Tensor
    graupel: torch.Tensor
    ice: torch.Tensor
    snow: torch.Tensor
    sr: torch.Tensor
    refl_10cm: torch.Tensor
    inner_steps: int
    inner_dt: float
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def compaction_overflow(self) -> torch.Tensor:
        """Device bool: a TEMPO compaction buffer (substep tiers, lane groups, source window or substep bound;
        regions ``tempo.*``) overflowed since the last ``ufs_physics_common.compaction.reset_overflow``. Check it when
        compaction is set: an overflowed call's results are not exact, and the call runs again with larger buffers.
        Reads no host value."""
        mask = sum(1 << bit for name, bit in compaction._bits.items() if name.startswith("tempo."))
        return (compaction.overflow_flag(self.ten_t.device) & mask) != 0


def fortran_nint(value: float) -> int:
    """Fortran ``nint``: round half away from zero."""
    magnitude = int(math.floor(abs(value) + 0.5))
    return magnitude if value >= 0.0 else -magnitude


def inner_timestep(dtp: float, dt_inner: float) -> tuple[int, float]:
    """``mp_tempo.F90:430-432`` (``dt`` is raised to ``dt_inner`` when ``dtp/ndt <= dt_inner``)."""
    ndt = max(fortran_nint(float(dtp) / float(dt_inner)), 1)
    dt = float(dtp) / ndt
    if dt <= float(dt_inner):
        dt = float(dt_inner)
    return ndt, dt


def require_host_constants(is_hail_aware: bool = False) -> None:
    """The TEMPO import-time configuration must be what ``mp_tempo_init`` sets up."""
    if _host_constants.ACTIVE_PROFILE != _host_constants.UFS_PHYSCONS:
        raise RuntimeError(
            "mp_tempo semantics need the CCPP host constants: set "
            f"{_host_constants.HOST_CONSTANTS_ENV}={_host_constants.UFS_PHYSCONS} before importing tempo_torch "
            f"(active profile: {_host_constants.ACTIVE_PROFILE})"
        )
    if _host_constants.HAIL_AWARE_INIT != bool(is_hail_aware):
        raise RuntimeError(
            f"mp_tempo_init passes hailaware_flag={bool(is_hail_aware)} to tempo_init: set "
            f"{_host_constants.HAIL_AWARE_INIT_ENV}={int(bool(is_hail_aware))} before importing tempo_torch"
        )


@dataclass(frozen=True)
class MpTempoInitialization:
    spechum: torch.Tensor
    tables: torch_native.TorchFrozenTables
    semi_sedi: bool
    convert_dry_rho: bool
    is_initialized: bool


def mp_tempo_init(*, spechum: torch.Tensor, tables: torch_native.TorchFrozenTables,
    semi_sedi: bool, convert_dry_rho: bool = True, restart: bool = False,
    is_initialized: bool = False, do_sat_adj: bool = False) -> MpTempoInitialization:
    """Selected non-aerosol R4 initialization using prepared frozen coefficients.

    Host constants and their derived coefficients use the existing import-time
    UFS profile. Load and authenticate tables outside numerics, then pass their
    device tensors here. The non-aerosol source only changes specific humidity;
    its local density/height scratch has no copyback. MPAS reprojects Statein
    after physics_init, so this transient result must not modify native tracers.
    """
    for name,value in (("semi_sedi",semi_sedi),("convert_dry_rho",convert_dry_rho),
        ("restart",restart),("is_initialized",is_initialized),("do_sat_adj",do_sat_adj)):
        if type(value) is not bool:
            raise TypeError(f"TEMPO initialization requires static {name}")
    if do_sat_adj:
        raise NotImplementedError("R4 MPAS TEMPO must condense internally")
    require_host_constants(False)
    if spechum.ndim != 2 or min(spechum.shape) < 1 or spechum.dtype not in torch_native.WORKING_DTYPES:
        raise ValueError("TEMPO initialization requires column-first float32/float64 humidity")
    torch_compat.assert_async(torch.isfinite(spechum).all(),"nonfinite TEMPO initial humidity")
    if not isinstance(tables,torch_native.TorchFrozenTables):
        raise TypeError("TEMPO initialization requires prepared tensor tables")
    if set(tables.arrays) != {spec.name for spec in TABLE_SPECS}:
        raise ValueError("TEMPO initialization requires the complete frozen table inventory")
    for spec in TABLE_SPECS:
        table = tables.arrays[spec.name]
        if (not isinstance(table,torch.Tensor) or table.shape != spec.shape
            or table.dtype != spechum.dtype or table.device != spechum.device):
            raise ValueError(f"invalid prepared TEMPO table {spec.name}")
    humidity = spechum if restart or is_initialized else torch.where(spechum < 0.,1.e-10,spechum)
    return MpTempoInitialization(humidity,tables,semi_sedi,convert_dry_rho,True)


def _zeros_like_column(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros(reference.shape[:1], dtype=reference.dtype, device=reference.device)


def mp_tempo_run(
    *,
    spechum: torch.Tensor,
    qc: torch.Tensor,
    qr: torch.Tensor,
    qi: torch.Tensor,
    qs: torch.Tensor,
    qg: torch.Tensor,
    ni: torch.Tensor,
    nr: torch.Tensor,
    tgrs: torch.Tensor,
    prsl: torch.Tensor,
    phii: torch.Tensor,
    omega: torch.Tensor,
    dtp: float,
    min_layer_depth: float,
    tables: Any,
    semi_sedi: bool,
    dt_inner: float | None = None,
    convert_dry_rho: bool = True,
    do_radar_ref: bool = False,
    con_g: float = CON_G,
    prcp: torch.Tensor | None = None,
    rain: torch.Tensor | None = None,
    graupel: torch.Tensor | None = None,
    ice: torch.Tensor | None = None,
    snow: torch.Tensor | None = None,
    refl_10cm: torch.Tensor | None = None,
    itimestep: int = 1,
    ntrac: int = MPAS_NTRAC,
    tracer_index: Mapping[str, int] = MPAS_TRACER_INDEX,
    is_aerosol_aware: bool = False,
    is_hail_aware: bool = False,
    sedimentation_compaction: Mapping[str, tuple[tuple[int, float], ...]] | None = None,
    semi_lagrangian_window_sources: int | None = None,
    lane_compaction: Mapping[str, float] | None = None,
    max_sedimentation_fall_speed: float = torch_native.DEFAULT_MAX_SEDIMENTATION_FALL_SPEED,
    max_snow_sedimentation_fall_speed: float | None = None,
    return_tendencies: bool = True,
    sedimentation_substep_region: bool = False,
    parameter_scales: Mapping[str, torch.Tensor] | None = None,
    spechum_entry: torch.Tensor | None = None,
    tgrs_entry: torch.Tensor | None = None,
) -> MpTempoResult:
    """Run ``mp_tempo_run`` for one ``[ncol, nlev]`` block (``phii`` is ``[ncol, nlev+1]``).

    ``spechum_entry`` fixes the pressure-mass basis of phase heat before convection.
    ``tgrs_entry`` anchors native vapor/liquid calorics at the same group entry.
    Without a host group, use this wrapper's entry humidity and temperature.
    ``dt_inner`` defaults to ``dtp`` (``GFS_typedefs``). Precipitation accumulators and
    ``refl_10cm`` default to the zero values the CCPP interstitial reset provides.
    The compaction arguments hold for the call only (restored afterwards); ``None`` means the dense or unbounded
    default, never a process-wide setting. ``sedimentation_compaction`` sets the substep compaction tiers
    (``torch_native.set_sedimentation_compaction``; ``None``: every substep on every column).
    ``semi_lagrangian_window_sources`` bounds the sources a semi-Lagrangian destination spans (a wider one sets
    overflow region ``tempo.window``; ``None``: unbounded). ``lane_compaction`` sets the lane buffers
    {group: fraction} of the masked FP64 groups (``torch_native.set_lane_compaction``; a buffer too small sets overflow
    region ``tempo.lanes.<group>``; ``None``: every lane). With any of them set, check
    ``MpTempoResult.compaction_overflow``. ``return_tendencies=False`` leaves
    ``MpTempoResult.tendencies`` empty (the per-species tendencies are in ``ten_q``); a
    compiled caller then does not write them as graph outputs. ``sedimentation_substep_region``
    makes a compiled call run each rain/graupel semi-Lagrangian substep in one separately compiled
    region reused across substeps (``torch_native.set_sedimentation_substep_region``); eager calls
    are unaffected. ``parameter_scales`` holds trainable scales by name, 0-d or 1-element
    (``torch_native.run_tempo_native``: ``nt_c``, ``demott_nuc_ssati``, ``av_s``, ``av_g``); None: no op.
    """

    if is_aerosol_aware or is_hail_aware:
        raise NotImplementedError("mp_tempo port covers the oracle suite: is_aerosol_aware=F, is_hail_aware=F")
    require_host_constants(is_hail_aware)
    with (_sedimentation_compaction(sedimentation_compaction),
          _semi_lagrangian_window(semi_lagrangian_window_sources), _lane_compaction(lane_compaction),
          _sedimentation_substep_region(sedimentation_substep_region)):
        return _mp_tempo_run(
            spechum_entry=spechum_entry,
            tgrs_entry=tgrs_entry,
            spechum=spechum, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, ni=ni, nr=nr, tgrs=tgrs, prsl=prsl,
            phii=phii, omega=omega, dtp=dtp, min_layer_depth=min_layer_depth, tables=tables,
            semi_sedi=semi_sedi, dt_inner=dt_inner, convert_dry_rho=convert_dry_rho,
            do_radar_ref=do_radar_ref, con_g=con_g, prcp=prcp, rain=rain, graupel=graupel, ice=ice,
            snow=snow, refl_10cm=refl_10cm, itimestep=itimestep, ntrac=ntrac, tracer_index=tracer_index,
            max_sedimentation_fall_speed=max_sedimentation_fall_speed,
            max_snow_sedimentation_fall_speed=max_snow_sedimentation_fall_speed,
            return_tendencies=return_tendencies,
            **({} if parameter_scales is None else {"parameter_scales": parameter_scales}),
        )


@contextmanager
def _sedimentation_substep_region(enabled: bool) -> Iterator[None]:
    previous = torch_native.set_sedimentation_substep_region(enabled)
    try:
        yield
    finally:
        torch_native.set_sedimentation_substep_region(previous)


@contextmanager
def _semi_lagrangian_window(sources: int | None) -> Iterator[None]:
    previous = torch_native.set_semi_lagrangian_window_sources(sources)
    try:
        yield
    finally:
        torch_native.set_semi_lagrangian_window_sources(previous)


@contextmanager
def _lane_compaction(fractions: Mapping[str, float] | None) -> Iterator[None]:
    previous = torch_native.set_lane_compaction({} if fractions is None else fractions)
    try:
        yield
    finally:
        torch_native.set_lane_compaction(previous)


@contextmanager
def _sedimentation_compaction(tiers: Mapping[str, tuple[tuple[int, float], ...]] | None) -> Iterator[None]:
    previous = torch_native.set_sedimentation_compaction({} if tiers is None else tiers)
    try:
        yield
    finally:
        torch_native.set_sedimentation_compaction(previous)


def _mp_tempo_run(
    *,
    spechum: torch.Tensor,
    qc: torch.Tensor,
    qr: torch.Tensor,
    qi: torch.Tensor,
    qs: torch.Tensor,
    qg: torch.Tensor,
    ni: torch.Tensor,
    nr: torch.Tensor,
    tgrs: torch.Tensor,
    prsl: torch.Tensor,
    phii: torch.Tensor,
    omega: torch.Tensor,
    dtp: float,
    min_layer_depth: float,
    tables: Any,
    semi_sedi: bool,
    dt_inner: float | None,
    convert_dry_rho: bool,
    do_radar_ref: bool,
    con_g: float,
    prcp: torch.Tensor | None,
    rain: torch.Tensor | None,
    graupel: torch.Tensor | None,
    ice: torch.Tensor | None,
    snow: torch.Tensor | None,
    refl_10cm: torch.Tensor | None,
    itimestep: int,
    ntrac: int,
    tracer_index: Mapping[str, int],
    max_sedimentation_fall_speed: float,
    max_snow_sedimentation_fall_speed: float | None,
    return_tendencies: bool,
    parameter_scales: Mapping[str, torch.Tensor] | None = None,
    spechum_entry: torch.Tensor | None = None,
    tgrs_entry: torch.Tensor | None = None,
) -> MpTempoResult:
    if spechum.ndim != 2 or phii.shape != (spechum.shape[0], spechum.shape[1] + 1):
        raise ValueError("mp_tempo_run expects [ncol, nlev] fields and phii [ncol, nlev+1]")
    if spechum_entry is not None and (spechum_entry.shape != spechum.shape
            or spechum_entry.dtype != spechum.dtype or spechum_entry.device != spechum.device):
        raise ValueError("mp_tempo_run entry humidity must match the column fields")
    if tgrs_entry is not None and (tgrs_entry.shape != tgrs.shape
            or tgrs_entry.dtype != tgrs.dtype or tgrs_entry.device != tgrs.device):
        raise ValueError("mp_tempo_run entry temperature must match the column fields")
    ndt, dt = inner_timestep(dtp, dtp if dt_inner is None else dt_inner)

    accum = {
        "prcp": prcp,
        "rain": rain,
        "graupel": graupel,
        "ice": ice,
        "snow": snow,
    }
    accum = {name: _zeros_like_column(spechum) if value is None else value for name, value in accum.items()}
    refl_out = torch.zeros_like(spechum) if refl_10cm is None else refl_10cm

    one = 1.0
    # The host passes the tracers as views of one [column, level, tracer] array; about 45 kernels of the
    # compiled TEMPO step read these ratios. Stored, they read contiguous fields instead of re-forming each
    # ratio from strided tracer reads (same values: ``stored`` is an identity view).
    qv = stored(spechum / (one - spechum))
    entry_specific = spechum if spechum_entry is None else spechum_entry
    terminal_heat_mass_scale = one / (one - entry_specific)
    phase_heat_reference_temperature = tgrs if tgrs_entry is None else tgrs_entry
    species_in = {"qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg, "ni": ni, "nr": nr}
    if convert_dry_rho:
        state_species = {name: stored(value / (one - spechum)) for name, value in species_in.items()}
    else:
        state_species = dict(species_in)
    dz = (phii[:, 1:] - phii[:, :-1]) / con_g
    roverrv = float(constants.ROVERRV)
    rdry = float(constants.RDRY)
    prepared_tables = torch_native.prepare_torch_tables(tables, spechum) if tables is not None else None

    temperature = tgrs
    overflow: torch.Tensor | None = None
    frozen_fraction = _zeros_like_column(spechum)
    refl_core: torch.Tensor | None = None
    for _ in range(ndt):
        rho = roverrv * prsl / (rdry * temperature * (qv + roverrv))
        w = -omega / (rho * con_g)
        state = {"t": temperature, "p": prsl, "w": w, "dz": dz, "qv": qv,
                 "terminal_heat_mass_scale": terminal_heat_mass_scale,
                 "phase_heat_reference_temperature": phase_heat_reference_temperature, **state_species}
        result = torch_native.run_tempo_native(
            state,
            dt=dt,
            is_first_step=itimestep == 1,
            use_temperature=True,
            tables=prepared_tables,
            semi_sedi_flag=bool(semi_sedi),
            refl10cm_from_melting=False,
            # MpTempoResult carries no hail field; skipping the unread diagnostic
            # removes its torch.nonzero host sync and graph-capture break.
            max_hail_diameter_flag=False,
            min_layer_depth=float(min_layer_depth),
            max_sedimentation_fall_speed=float(max_sedimentation_fall_speed),
            max_snow_sedimentation_fall_speed=(None if max_snow_sedimentation_fall_speed is None
                                               else float(max_snow_sedimentation_fall_speed)),
            # Effective radii are never read here and reflectivity only with do_radar_ref.
            radar_and_effective_radii=bool(do_radar_ref),
            **({} if parameter_scales is None else {"parameter_scales": parameter_scales}),
        )
        out = result.state
        temperature = out["t"]
        qv = out["qv"]
        state_species = {name: out[name] for name in SPECIES}
        diags = result.workspace.tempo_diags
        ice_p = diags["ice_liquid_equiv_precip"].reshape(-1)
        snow_p = diags["snow_liquid_equiv_precip"].reshape(-1)
        graupel_p = diags["graupel_liquid_equiv_precip"].reshape(-1)
        rain_p = diags["rain_precip"].reshape(-1)
        accum["ice"] = accum["ice"] + torch.clamp_min(ice_p / 1000.0, 0.0)
        accum["snow"] = accum["snow"] + (torch.clamp_min(ice_p, 0.0) + torch.clamp_min(snow_p, 0.0)) / 1000.0
        accum["graupel"] = accum["graupel"] + torch.clamp_min(graupel_p / 1000.0, 0.0)
        accum["rain"] = accum["rain"] + torch.clamp_min(rain_p / 1000.0, 0.0)
        accum["prcp"] = accum["prcp"] + (
            torch.clamp_min(ice_p, 0.0)
            + torch.clamp_min(snow_p, 0.0)
            + torch.clamp_min(graupel_p, 0.0)
            + torch.clamp_min(rain_p, 0.0)
        ) / 1000.0
        frozen_fraction = diags["frozen_fraction"].reshape(-1)
        # Plain lookups, not dict.get: compile-reachable code keeps no .get calls (structural ratchet).
        refl_core = diags["refl10cm"] if "refl10cm" in diags else None
        step_diagnostics = result.workspace.diagnostics
        if torch_native.SEDIMENTATION_SUBSTEP_OVERFLOW in step_diagnostics:
            step_overflow = step_diagnostics[torch_native.SEDIMENTATION_SUBSTEP_OVERFLOW]
            overflow = step_overflow if overflow is None else overflow | step_overflow
        # The wrapper consumes only these extracted outputs, not the diagnostic workspace.
        result = out = state = diags = step_diagnostics = None
        del ice_p, snow_p, graupel_p, rain_p

    if do_radar_ref:
        if refl_core is None:
            raise RuntimeError("TEMPO runtime returned no refl10cm diagnostic")
        refl_out = refl_core.reshape(spechum.shape)

    new_spechum = qv / (one + qv)
    if convert_dry_rho:
        state_species = {name: value / (one + qv) for name, value in state_species.items()}
    tendencies = {"dspechum": (new_spechum - spechum) / dtp}
    for name in SPECIES:
        tendencies[f"d{name}"] = (state_species[name] - species_in[name]) / dtp
    ten_t = (temperature - tgrs) / dtp
    zero = torch.zeros_like(spechum)
    slot_of = {int(tracer_index[TENDENCY_SLOTS[name]]) - 1: name for name in TENDENCY_SLOTS}
    ten_q = torch.stack([tendencies[slot_of[k]] if k in slot_of else zero for k in range(ntrac)], dim=-1)
    diagnostics = {} if overflow is None else {torch_native.SEDIMENTATION_SUBSTEP_OVERFLOW: overflow}
    return MpTempoResult(
        tendencies=tendencies if return_tendencies else {},
        ten_q=ten_q,
        ten_t=ten_t,
        ten_u=zero,
        ten_v=zero,
        prcp=accum["prcp"],
        rain=accum["rain"],
        graupel=accum["graupel"],
        ice=accum["ice"],
        snow=accum["snow"],
        sr=frozen_fraction,
        refl_10cm=refl_out,
        inner_steps=ndt,
        inner_dt=dt,
        diagnostics=diagnostics,
    )


def mp_tempo_post_run(ten_t: torch.Tensor, prslk: torch.Tensor, ttendlim: float) -> torch.Tensor:
    """``mp_tempo_post.F90``: limit the potential-temperature tendency when ``ttendlim >= 0``."""
    if float(ttendlim) < 0.0:
        return ten_t
    mp_tend = ten_t / prslk
    mp_tend = torch.clamp(mp_tend, min=-float(ttendlim), max=float(ttendlim))
    return mp_tend * prslk
