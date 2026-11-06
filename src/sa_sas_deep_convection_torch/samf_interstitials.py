# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Host-side CCPP interstitial rules shared by the SA-SAS deep and shallow ports.

* ``ConvectionTracerLayout``: the convective-tracer bookkeeping of
  ``CCPP_typedefs.F90::gfs_interstitial_setup_tracers`` (``otsptflag``,
  ``tracers_total``, ``ntk``, ``nn``, ``nsamftrac``, ``nscav``) for a tracer table.
* ``build_clw``: the ``clw`` construction of ``GFS_suite_interstitial_3_run`` (and the
  identical refresh in ``GFS_DCNV_generic_post_run``): transported tracers in
  ``otsptflag`` order from slot 3, ice/water in slots 1/2 by microphysics code.
* ``samf_water_reconciliation``: the MPAS SAMF water reconciliation of the oracle
  ``GFS_DCNV_generic_post_run`` / ``GFS_SCNV_generic_post_run`` (active when
  ``MPAS_UFS_SAMF_WATER_RECONCILIATION=1``). Column sums run level by level in the
  Fortran order; there is no host synchronisation and its fail-closed guards are
  returned as an ``errflg`` tensor.

All tracer indices below are 1-based CCPP indices; 0 means "absent".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

# CCPP GFS_typedefs.F90 microphysics codes.
IMP_PHYSICS_CODES = {
    "wsm6": 6,
    "thompson": 8,
    "mg": 10,
    "gfdl": 11,
    "fer_hires": 15,
    "nssl": 17,
    "nssl2mccn": 18,
    "tempo": 88,
}

INDEX_NAMES = (
    "ntqv", "ntcw", "ntiw", "ntrw", "ntsw", "ntgl", "ntclamt", "ntrnc", "ntsnc", "ntgnc",
    "ntlnc", "ntinc", "ntoz", "ntke", "nthl", "nthnc", "ntgv", "nthv", "ntccn", "ntccna",
    "ntrz", "ntgz", "nthz", "ntsigma", "ntomega",
)
# CCPP_typedefs.F90: tracers that are never convectively transported.
OTSPT_EXCLUDED = (
    "ntcw", "ntiw", "ntclamt", "ntrw", "ntsw", "ntrnc", "ntlnc", "ntinc", "ntsnc", "ntgl", "ntgnc",
    "nthl", "nthnc", "ntgv", "nthv", "ntccn", "ntccna", "ntrz", "ntgz", "nthz", "ntsigma", "ntomega",
)
# Oracle MPAS lane tracer table (qv qc qr qi qs qg ni nr sgs_tke o3mr).
MPAS_TEMPO_TRACERS = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "sgs_tke", "o3mr")
MPAS_TEMPO_INDICES = {
    "ntqv": 1, "ntcw": 2, "ntrw": 3, "ntiw": 4, "ntsw": 5, "ntgl": 6,
    "ntinc": 7, "ntrnc": 8, "ntke": 9, "ntoz": 10,
}
MPAS_NWAT = 6  # MPAS atmos_model.F90 Cfg%nwat


@dataclass(frozen=True)
class ConvectionTracerLayout:
    """Convective tracer bookkeeping for one tracer table (all indices 1-based)."""

    name: str
    ntrac: int
    nn: int
    ntr: int
    ntk: int
    itc: int
    ntc: int
    otsptflag: tuple[bool, ...]
    tracers_total: int
    indices: tuple[tuple[str, int], ...]

    def index(self, name: str) -> int:
        return dict(self.indices).get(name, 0)

    def as_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ntrac": self.ntrac,
            "nn": self.nn,
            "ntr": self.ntr,
            "ntk": self.ntk,
            "itc": self.itc,
            "ntc": self.ntc,
            "otsptflag": list(self.otsptflag),
            "tracers_total": self.tracers_total,
            "indices": dict(self.indices),
        }


def ccpp_convective_tracers(
    ntrac: int, indices: Mapping[str, int], *, satmedmf: bool, trans_trac: bool, ncnd: int
) -> dict[str, Any]:
    """``CCPP_typedefs.F90`` rule (cscnv=ras=do_shoc=.false., trans_aero=.false.)."""

    otsptflag = [True] * ntrac
    tracers_total = 0
    ntk = 0
    if satmedmf or trans_trac:
        excluded = {int(indices.get(name, 0)) for name in OTSPT_EXCLUDED} - {0}
        tracers = 2
        for n in range(2, ntrac + 1):
            transported = n not in excluded
            otsptflag[n - 1] = transported
            if transported:
                tracers += 1
                if int(indices.get("ntke", 0)) == n:
                    ntk = tracers
        tracers_total = tracers - 2
    if int(indices.get("ntiw", 0)) > 0:
        absent = sum(1 for name in ("ntclamt", "ntsigma", "ntomega") if int(indices.get(name, 0)) > 0)
        nn = ntrac - 1 - absent
    elif int(indices.get("ntcw", 0)) > 0:
        nn = ntrac
    else:
        nn = ntrac + 1
    return {
        "otsptflag": tuple(otsptflag),
        "tracers_total": tracers_total,
        "nsamftrac": tracers_total if (satmedmf or trans_trac) else 0,
        "ntk": ntk,
        "nn": nn,
        "nscav": ntrac - ncnd + 2,
        "itc": 0,
    }


def mpas_tempo_layout(*, ntc: int | None = None) -> ConvectionTracerLayout:
    """MPAS/TEMPO transport layout; pass the actual cap's ntchm as ntc.

    Omission preserves the historical G3.2 nscav-sized chemistry fixture.
    """

    rule = ccpp_convective_tracers(
        len(MPAS_TEMPO_TRACERS), MPAS_TEMPO_INDICES, satmedmf=True, trans_trac=True, ncnd=MPAS_NWAT - 1
    )
    # nscav sizes a different host work array. The real SAMF cap passes ntchm,
    # which is zero in R4; the G3.2 chemistry fixture passes nscav instead.
    if ntc is not None and (type(ntc) is not int or ntc < 0):
        raise ValueError("MPAS convection chemistry extent must be a static nonnegative integer")
    return ConvectionTracerLayout(
        name="mpas_tempo",
        ntrac=len(MPAS_TEMPO_TRACERS),
        nn=rule["nn"],
        ntr=rule["nsamftrac"],
        ntk=rule["ntk"],
        itc=rule["itc"],
        ntc=rule["nscav"] if ntc is None else ntc,
        otsptflag=rule["otsptflag"],
        tracers_total=rule["tracers_total"],
        indices=tuple(sorted(MPAS_TEMPO_INDICES.items())),
    )


def legacy_layout(ntrac: int, *, indices: Mapping[str, int], ntk: int, itc: int, ntc: int) -> ConvectionTracerLayout:
    """The frozen legacy harness table: nn=ntrac+2, all tracers but qv transported."""

    return ConvectionTracerLayout(
        name="legacy",
        ntrac=ntrac,
        nn=ntrac + 2,
        ntr=ntrac,
        ntk=ntk,
        itc=itc,
        ntc=ntc,
        otsptflag=(False, *([True] * (ntrac - 1))),
        tracers_total=ntrac,
        indices=tuple(sorted((k, int(v)) for k, v in indices.items() if int(v) > 0)),
    )


def layout_from_config(value: Any) -> ConvectionTracerLayout | None:
    """Decode ``config['tracer_layout']`` (a layout, a mapping, or ``'mpas_tempo'``)."""

    if value is None:
        return None
    if isinstance(value, ConvectionTracerLayout):
        return value
    if isinstance(value, str):
        if value == "mpas_tempo":
            return mpas_tempo_layout()
        raise ValueError(f"unknown tracer layout {value!r}")
    if isinstance(value, Mapping):
        indices = {str(k): int(v) for k, v in dict(value["indices"]).items() if int(v) > 0}
        return ConvectionTracerLayout(
            name=str(value.get("name", "custom")),
            ntrac=int(value["ntrac"]),
            nn=int(value["nn"]),
            ntr=int(value["ntr"]),
            ntk=int(value["ntk"]),
            itc=int(value["itc"]),
            ntc=int(value["ntc"]),
            otsptflag=tuple(bool(v) for v in value["otsptflag"]),
            tracers_total=int(value["tracers_total"]),
            indices=tuple(sorted(indices.items())),
        )
    raise TypeError(f"tracer_layout must be a layout, mapping or name, got {type(value).__name__}")


def transported_slots(layout: ConvectionTracerLayout) -> tuple[tuple[int, int], ...]:
    """(gq0 0-based index, clw 0-based slot) pairs of the otsptflag loop (n=2..ntrac)."""

    pairs = []
    tracers = 2
    for n in range(2, layout.ntrac + 1):
        if layout.otsptflag[n - 1]:
            tracers += 1
            pairs.append((n - 1, tracers - 1))
    return tuple(pairs)


def refreshes_ice_water(imp_physics: int | None) -> bool:
    if imp_physics is None:
        return True
    return imp_physics in {IMP_PHYSICS_CODES[name] for name in ("thompson", "tempo", "nssl", "wsm6", "mg", "fer_hires")}


def build_clw(
    gq0: torch.Tensor,
    clw: torch.Tensor,
    layout: ConvectionTracerLayout,
    *,
    imp_physics: int | None,
    transport: bool,
) -> torch.Tensor:
    """``GFS_suite_interstitial_3`` / ``GFS_DCNV_generic_post`` clw refresh (pure tensor ops)."""

    clw = clw.clone()
    if transport:
        for source, slot in transported_slots(layout):
            clw[:, :, slot] = gq0[:, :, source]
    ntcw, ntiw = layout.index("ntcw"), layout.index("ntiw")
    if imp_physics == IMP_PHYSICS_CODES["gfdl"]:
        clw[:, :, 0] = gq0[:, :, ntcw - 1]
    elif refreshes_ice_water(imp_physics):
        clw[:, :, 0] = gq0[:, :, ntiw - 1]
        clw[:, :, 1] = gq0[:, :, ntcw - 1]
    return clw


def tendencies_from_dclw(
    ten_q: torch.Tensor, dclw: torch.Tensor, layout: ConvectionTracerLayout, *, is_gfdl: bool
) -> torch.Tensor:
    """The post-scheme ``ten_q`` fill from ``dclw`` (identical in DCNV and SCNV post)."""

    ten_q = ten_q.clone()
    if layout.tracers_total > 0:
        for target, slot in transported_slots(layout):
            ten_q[:, :, target] = dclw[:, :, slot]
    ntcw, ntiw = layout.index("ntcw"), layout.index("ntiw")
    if ntcw > 0:
        if is_gfdl:
            ten_q[:, :, ntcw - 1] = dclw[:, :, 0] + dclw[:, :, 1]
        elif ntiw > 0:
            ten_q[:, :, ntiw - 1] = dclw[:, :, 0]
            ten_q[:, :, ntcw - 1] = dclw[:, :, 1]
        else:
            ten_q[:, :, ntcw - 1] = dclw[:, :, 0] + dclw[:, :, 1]
    return ten_q


def interface_pressure(psp: torch.Tensor, delp: torch.Tensor) -> torch.Tensor:
    """prsi(:,1)=ps, prsi(:,k+1)=prsi(:,k)-delp(:,k), level by level as in the harness."""

    levels = [psp]
    for k in range(delp.shape[1]):
        levels.append(levels[-1] - delp[:, k])
    return torch.stack(levels, dim=1)


def _container_water_levels(values: torch.Tensor, km: int, water_slots: tuple[int, ...]) -> torch.Tensor:
    # Same slot order as _container_water, for all levels at once. The sum starts at +0.0 (a -0.0
    # first slot becomes +0.0, as in _container_water) without a zero-filled temporary.
    if not water_slots:
        return torch.zeros_like(values[:, :km, 0])
    total = values[:, :km, water_slots[0]] + 0.0
    for slot in water_slots[1:]:
        total = total + values[:, :km, slot]
    return total


def _container_water(values: torch.Tensor, k: int, water_slots: tuple[int, ...]) -> torch.Tensor:
    total = torch.zeros_like(values[:, k, 0])
    for slot in water_slots:
        total = total + values[:, k, slot]
    return total


def samf_water_reconciliation(
    gq0: torch.Tensor,
    gt0: torch.Tensor,
    save_q: torch.Tensor,
    rain1: torch.Tensor,
    area: torch.Tensor,
    prsi: torch.Tensor,
    frain: torch.Tensor,
    con_g: torch.Tensor,
    con_cp: torch.Tensor,
    con_hvap: torch.Tensor,
    layout: ConvectionTracerLayout,
    *,
    delt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle MPAS SAMF water reconciliation; returns (gq0, gt0, errflg).

    The column water residual (atmospheric container-water change plus requested
    precipitation) is removed by rescaling vapour, ``gt0 -= (hvap/cp) * dqv`` keeps
    ``cp*T + Lv*qv``. The Fortran error returns become ``errflg = 1`` (no host read).
    """

    ncol, km, ntrac = gq0.shape
    qv = layout.index("ntqv")
    if not 1 <= qv <= ntrac:
        raise ValueError("water reconciliation requires a valid ntqv")
    water_slots = tuple(
        index - 1
        for index in (layout.index(name) for name in ("ntqv", "ntcw", "ntiw", "ntrw", "ntsw", "ntgl"))
        if 1 <= index <= ntrac
    )
    zero = torch.zeros_like(rain1)
    positive_geometry = (area > 0.0) & (con_g > 0.0)
    # Level terms for all levels at once; the column sums keep the source level order.
    layer_mass = torch.where(positive_geometry[:, None], area[:, None] * torch.abs(prsi[:, :km] - prsi[:, 1:km + 1]) / con_g, 0.0)
    requested = area * 1000.0 * frain * rain1
    qv_all = gq0[:, :km, qv - 1]
    # Column sums in level order, read level by level from the [ncol, km] terms (no level-major copy).
    water_change = layer_mass * (_container_water_levels(gq0, km, water_slots) - _container_water_levels(save_q, km, water_slots))
    vapour_mass = layer_mass * torch.clamp_min(qv_all, 0.0)
    column_delta, column_qv_mass = requested, zero
    for k in range(km):
        column_delta = column_delta + water_change[:, k]
        column_qv_mass = column_qv_mass + vapour_mass[:, k]
    # The relative bounds below count float64 rounding steps (5e-13 = 2252 ulp); in float32 the
    # same count is scaled by the epsilon ratio (``relative`` is 5e-13 exactly in float64).
    relative = 5.0e-13 * (torch.finfo(gq0.dtype).eps / 2.0 ** -52)
    column_tol = torch.clamp_min(relative * (torch.abs(column_delta) + requested), 1.0)
    has_vapour = column_qv_mass > 0.0
    fraction = torch.where(has_vapour, -column_delta / torch.where(has_vapour, column_qv_mass, 1.0), zero)
    error = (~has_vapour) & (torch.abs(column_delta) > column_tol)
    # The residual of a call grows with the call: 1% of column vapour at dt 300 s, the same bound per second at any step.
    error = error | (torch.abs(fraction) > 1.0e-2 * delt / 300.0) | (fraction < -1.0)

    eps64 = torch.finfo(gq0.dtype).eps
    latent_over_cp = con_hvap / con_cp
    requested_k = fraction[:, None] * torch.clamp_min(qv_all, 0.0)
    reconciled = qv_all + requested_k
    roundoff_tol = 64.0 * eps64 * torch.maximum(torch.maximum(torch.ones_like(qv_all), torch.abs(qv_all)), torch.abs(requested_k))
    error = error | (reconciled < -roundoff_tol).any(dim=1)
    floored = reconciled < 0.0
    correction = torch.where(floored, -qv_all, requested_k)
    reconciled = torch.where(floored, 0.0, reconciled)
    temperature_correction = -latent_over_cp * correction
    # The running minimum only feeds the fail-closed min_qv < 0 guard; min is exact in any order.
    min_qv = torch.minimum(torch.full_like(rain1, torch.finfo(gq0.dtype).max), reconciled.amin(dim=1))
    terms = (
        layer_mass * correction,
        torch.where(floored, layer_mass * (correction - requested_k), 0.0),
        layer_mass * con_cp * temperature_correction,
        layer_mass * con_hvap * correction,
    )
    sums = [zero] * len(terms)  # +0.0 starts, as the zero-filled sums before
    for k in range(km):
        sums = [total + term[:, k] for total, term in zip(sums, terms)]
    applied, floor_water, thermal, latent = sums
    qv_new = reconciled
    t_new = gt0[:, :km] + temperature_correction
    # Aggregate fail-closed guards (column totals; reduction order is not Fortran's,
    # these only decide errflg and never change state).
    raw_total = column_delta.sum()
    applied_total = applied.sum()
    thermal_total = thermal.sum()
    latent_total = latent.sum()
    # The totals are sums of cancelling column values, so their rounding follows the size of those values and not of
    # the totals: a chunk total near zero in FP32 failed by rounding alone (column residuals at most 0.22%).
    water_scale = torch.abs(column_delta).sum() + torch.abs(applied).sum()
    energy_scale = torch.abs(thermal).sum() + torch.abs(latent).sum()
    water_tol = torch.clamp_min(relative * water_scale, 1.0)
    energy_tol = torch.clamp_min(relative * energy_scale, 1.0)
    aggregate_error = (
        (torch.abs(raw_total + applied_total) > water_tol)
        | (torch.abs(floor_water.sum()) > water_tol)
        | (torch.abs(thermal_total + latent_total) > energy_tol)
        | (min_qv.min() < 0.0)
    )
    # Only the vapour slot changes: one select instead of a full copy plus a slot write.
    slots = torch.arange(ntrac, device=gq0.device)
    gq0 = torch.where(slots == qv - 1, qv_new[:, :, None], gq0)
    gt0 = t_new
    errflg = (error.any() | aggregate_error).to(gq0.dtype)
    return gq0, gt0, errflg
