# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Live PBL bindings for the selected R4 TEMPO/SATMEDMF tracer layout."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping

import torch
from sa_tke_edmf_torch.segments.segment_a import run as pack_tracers
from sa_tke_edmf_torch.state import SaTkeEdmfState
from sa_tke_edmf_torch.tracers import MPAS_TEMPO_TRACER_LAYOUT
from sa_tke_edmf_torch.segments import (
    segment_b, segment_c, segment_d, segment_e, segment_f, segment_g,
    segment_h, segment_i, segment_j, segment_k, segment_l, segment_m,
    segment_n, segment_o, segment_p, segment_q, segment_r,
    segment_s, segment_t,
)

from .physics_prefix import PhysicsPrefixResult
from .containers import Statein
from .grid import Grid
from .precision import WORKING_DTYPES


def _pbl_pre_tensors(q, temperature, u, v):
    packed = pack_tracers(SaTkeEdmfState(
        fields={"q1": q, "qgrs": q, "gq0": q, "gt0": temperature, "gu0": u, "gv0": v},
        config={"imp_physics": 88, "tracer_layout": MPAS_TEMPO_TRACER_LAYOUT},
    ))
    return packed.require("vdftra"), packed.require("rtg_ozone_index").to(torch.int32)


def apply_pbl_pre(prefix: PhysicsPrefixResult) -> PhysicsPrefixResult:
    """Bind GFS_PBL_generic_pre to live Stateout, copying only its two outputs.

    This adapter selects the R4 nvdiff=ntrac=10 SATMEDMF/TEMPO contract.
    The caller validates physical input values before entering the physics group.
    """
    stateout = prefix.stateout
    q = stateout.gq0
    if q.ndim != 3 or q.shape[2] != 10 or min(q.shape[:2]) < 1:
        raise ValueError("R4 PBL pre requires nonempty [ncol,nlev,10] tracers")
    for name in ("gq0", "gt0", "gu0", "gv0"):
        value = getattr(stateout, name)
        shape = q.shape if name == "gq0" else q.shape[:2]
        if value.dtype != q.dtype or q.dtype not in WORKING_DTYPES or value.shape != shape or value.device != q.device:
            raise ValueError(f"invalid live R4 PBL pre field {name}")
    from sa_tke_edmf_torch.compile import run_compile_island

    vdftra, ozone = run_compile_island("interstitial", "pre", _pbl_pre_tensors,
                                     q, stateout.gt0, stateout.gu0, stateout.gv0)
    # Segment A also makes private TKE workspace; the source pre does not write it.
    work = {
        **prefix.interstitial,
        "vdftra": vdftra,
        "rtg_ozone_index": ozone,
    }
    return replace(prefix, interstitial=work)


@dataclass(frozen=True)
class PBLMixingResult:
    prefix: PhysicsPrefixResult
    hpbl: torch.Tensor
    dku3d_h: torch.Tensor
    dku3d_e: torch.Tensor


def run_pbl_sequence(
    statein: Statein, grid: Grid, prefix: PhysicsPrefixResult, *,
    dtp: torch.Tensor, dtf: torch.Tensor, dtidx: torch.Tensor,
    control: Mapping[str, object],
) -> PBLMixingResult:
    """Run the selected cap's pre, mixing, canopy and post exactly once.

    Surface processing must finish before this call. The returned hpbl and
    prefix feed GWD; the native commit owns the later tendency finalizer.
    """
    mixed = apply_pbl_mixing(statein, grid, apply_pbl_pre(prefix),
        delt=dtp, control=control, dtidx=dtidx)
    canopy = apply_pbl_canopy(mixed.prefix, control=control)
    post = apply_pbl_post(canopy, dtp=dtp, dtf=dtf, dtidx=dtidx, control=control)
    return replace(mixed, prefix=post)


def _run_pbl_regions(*args, **kwargs):
    from sa_tke_edmf_torch.compile import regional_compile

    with regional_compile():
        return run_pbl_sequence(*args, **kwargs)


# The same numerical driver, with reusable exact-policy tensor regions. Carried
# prefix/configuration containers never cross a compiled recurrence boundary.
from ufs_physics_common import compile_policy

_run_pbl_regions.__wrapped__ = run_pbl_sequence
_run_pbl_regions.warm = lambda *args, **kwargs: None
compile_policy._compiled[run_pbl_sequence] = _run_pbl_regions


def apply_pbl_mixing(
    statein: Statein, grid: Grid, prefix: PhysicsPrefixResult, *,
    delt: torch.Tensor, control: Mapping[str, object], dtidx: torch.Tensor,
) -> PBLMixingResult:
    """Bind source-owned R4 core outputs without applying the PBL post-step.

    The caller must first run apply_pbl_pre. This function does not perform
    canopy chemistry, unpack tendencies, or commit the PBL post-step.
    """
    work = prefix.interstitial
    surface = prefix.sfcprop
    stateout = prefix.stateout
    ncol, nlev = stateout.gt0.shape
    if delt.ndim != 0 or delt.dtype not in WORKING_DTYPES:
        raise ValueError("PBL timestep must be a float32/float64 scalar tensor")
    # The core indexes vdftra after the TEMPO pre-step packs ice before rain.
    config = {**control, "imp_physics": 88, "tracer_layout": MPAS_TEMPO_TRACER_LAYOUT}
    fields = {
        "q1": work["vdftra"], "rtg": work["dvdftra"],
        "t1": stateout.gt0, "u1": stateout.gu0, "v1": stateout.gv0,
        "gt0": stateout.gt0, "gq0": stateout.gq0,
        "usfco": surface.usfco, "vsfco": surface.vsfco,
        "swh": prefix.carry.radtend["htrsw"],
        "hlw": prefix.carry.radtend["htrlw"], "xmu": work["xmu"],
        "garea": grid.area, "zvfun": work["zvfun"], "sigmaf": work["sigmaf"],
        "psk": statein.prsik[:, 0], "rbsoil": work["rb"],
        "zorl": surface.zorl, "tsea": surface.tsfc,
        "fm": surface.ffmm, "fh": surface.ffhh,
        "heat": work["hflxq"], "evap": surface.evap,
        "stress": work["stress"], "spd1": work["wind"],
        "del": work["del"], "kinver": work["kinver"],
        "delt": delt.expand(ncol), "tkeh": work["tkeh"][:, :nlev],
        "dtend": prefix.carry.intdiag["dtend"],
        "dtidx": dtidx,
        "u10m": prefix.carry.intdiag["u10m"],
        "v10m": prefix.carry.intdiag["v10m"],
        **{n: getattr(statein, n) for n in (
            "def_1", "def_2", "def_3", "prsi", "prsl", "prslk", "phii", "phil",
        )},
        **{n: work[n] for n in ("ten_t", "ten_u", "ten_v")},
    }
    current = SaTkeEdmfState(fields, config)
    for segment in (
        segment_b, segment_c, segment_d, segment_e, segment_f, segment_g,
        segment_h, segment_i, segment_j, segment_k, segment_l, segment_m,
        segment_n, segment_o, segment_p, segment_q, segment_r,
    ):
        current = segment.run(current)
    updates = {
        "dvdftra": current.require("rtg"),
        "tkeh": torch.cat((current.require("tkeh"), work["tkeh"][:, nlev:]), dim=1),
        "kpbl": current.require("kpbl").to(torch.int32),
        **{n: current.require(n) for n in ("ten_t", "ten_u", "ten_v")},
        **{n + "1": current.require(n) for n in ("dusfc", "dvsfc", "dtsfc", "dqsfc")},
    }
    carry = replace(prefix.carry, intdiag={
        **prefix.carry.intdiag,
        **{n: current.require(n) for n in ("dtend", "dkt", "dku")},
    })
    result = replace(prefix, interstitial={**work, **updates}, carry=carry)
    return PBLMixingResult(result, current.require("hpbl"),
                           current.require("dku3d_h"), current.require("dku3d_e"))


def apply_pbl_canopy(prefix: PhysicsPrefixResult, *, control: Mapping[str, object]) -> PhysicsPrefixResult:
    """Retain canopy_driver's exact early return for the selected R4 flags."""
    if control["do_canopy"] and control["cplaqm"]:
        raise NotImplementedError("live active canopy_driver binding is not qualified")
    return prefix


def _pbl_post_tensors(fields, config):
    current = segment_t.run(segment_s.run(SaTkeEdmfState(fields, config)))
    return {name: current.require(name) for name in (
        "gt0", "gu0", "gv0", "gq0", "dtend", "ten_q", "ten_t_pbl", "ten_q_pbl",
        "dtdt", "dudt", "dvdt", "dqdt",
    )}


def apply_pbl_post(
    prefix: PhysicsPrefixResult, *, dtp: torch.Tensor, dtf: torch.Tensor,
    dtidx: torch.Tensor, control: Mapping[str, object],
) -> PhysicsPrefixResult:
    """Unpack and commit R4 PBL tendencies and source-owned flux diagnostics.

    The frozen target disables external flux/chemistry coupling. Active external
    coupling must not silently use this selected noncoupled binding.
    """
    if any(control[n] for n in ("cplflx", "cplaqm", "cplchm", "cplcat")):
        raise NotImplementedError("live external PBL flux coupling is not qualified")
    for name, value in (("dtp", dtp), ("dtf", dtf)):
        if value.ndim != 0 or value.dtype not in WORKING_DTYPES or value.device != prefix.stateout.gt0.device:
            raise ValueError(f"PBL post {name} requires a device-local float32/float64 scalar")
    work = prefix.interstitial
    out = prefix.stateout
    ncol = out.gt0.shape[0]
    fields = {
        "q1": work["vdftra"], "rtg": work["dvdftra"],
        "gt0": out.gt0, "gu0": out.gu0, "gv0": out.gv0, "gq0": out.gq0,
        "delt": dtp.expand(ncol), "dtf": dtf,
        "dtend": prefix.carry.intdiag["dtend"], "dtidx": dtidx,
        **{n: work[n] for n in ("ten_t", "ten_u", "ten_v", "dtdt", "dudt", "dvdt", "dqdt")},
    }
    config = {name: control[name] for name in (
        "tend_opt_pbl", "ldiag3d", "lsidea", "index_of_process_pbl",
        "index_of_temperature", "index_of_x_wind", "index_of_y_wind",
    ) if name in control}
    config.update(imp_physics=88, tracer_layout=MPAS_TEMPO_TRACER_LAYOUT)
    from sa_tke_edmf_torch.compile import run_compile_island

    current = run_compile_island("interstitial", "post", _pbl_post_tensors, fields, config)
    diag = {**prefix.carry.intdiag, "dtend": current["dtend"]}
    if control["lssav"]:
        fluxes = {"dusfc": work["dusfc1"], "dvsfc": work["dvsfc1"],
                  "dtsfc": work["dtsfc1"] * work["hffac"], "dqsfc": work["dqsfc1"]}
        for name, value in fluxes.items():
            diag[name] = diag[name] + value * dtf
            diag[name + "i"] = value
    stateout = replace(out, **{n: current[n] for n in ("gt0", "gu0", "gv0", "gq0")})
    return replace(prefix, stateout=stateout,
        interstitial={**work, **{n: current[n] for n in
            ("ten_q", "ten_t_pbl", "ten_q_pbl", "dtdt", "dudt", "dvdt", "dqdt")}},
        carry=replace(prefix.carry, intdiag=diag))
