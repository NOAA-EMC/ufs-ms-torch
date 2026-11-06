# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment U: final cloud, mass-flux, and tendency outputs."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch.compile import run_compile_island
from sa_sas_deep_convection_torch.constants import GRAV, KM, NN, NTR
from sa_sas_deep_convection_torch.state import tracer_layout, runtime_dimensions, SaSasDeepConvectionState, require_real

CHECKPOINT_FIELDS = (
    "ten_t",
    "ten_u",
    "ten_v",
    "ten_q",
    "dqtr",
    "cldwrk",
    "rn",
    "kbot",
    "ktop",
    "kcnv",
    "ud_mf",
    "dd_mf",
    "dt_mf",
    "cnvw",
    "cnvc",
)


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 3503-3711 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    cnvflg = state.require("cnvflg")
    rn = state.require("rn")
    kmax = state.require("kmax")
    kb = state.require("kb")
    kbcon = state.require("kbcon")
    ktcon = state.require("ktcon")
    ktop = state.require("ktop")
    jmin = state.require("jmin")
    eta = state.require("eta")
    etad = state.require("etad")
    edto = state.require("edto")
    xmb = state.require("xmb")
    cnvwt = state.require("cnvwt")
    dellal = state.require("dellal")
    t1 = state.require("t1")
    u1 = state.require("u1")
    v1 = state.require("v1")
    q1 = state.require("q1")
    qtr = state.require("qtr")
    new_t1 = state.require("new_t1").clone()
    new_u1 = state.require("new_u1").clone()
    new_v1 = state.require("new_v1").clone()
    new_q1 = state.require("new_q1").clone()
    new_qtr = state.require("new_qtr").clone()
    cnvw = state.require("cnvw").clone()
    cnvc = state.require("cnvc").clone()
    ud_mf = state.require("ud_mf").clone()
    dd_mf = state.require("dd_mf").clone()
    dt_mf = state.require("dt_mf").clone()
    wet_dep = state.require("wet_dep").clone()
    pfld = state.require("pfld")
    sigmagfm = state.require("sigmagfm")
    ncol = cnvflg.shape[0]
    for name, tensor in {
        "cnvflg": cnvflg,
        "rn": rn,
        "kmax": kmax,
        "kb": kb,
        "kbcon": kbcon,
        "ktcon": ktcon,
        "ktop": ktop,
        "jmin": jmin,
        "edto": edto,
        "xmb": xmb,
        "sigmagfm": sigmagfm,
    }.items():
        require_real(name, tensor, (ncol,))
    for name, tensor in {
        "eta": eta,
        "etad": etad,
        "cnvwt": cnvwt,
        "dellal": dellal,
        "t1": t1,
        "u1": u1,
        "v1": v1,
        "q1": q1,
        "new_t1": new_t1,
        "new_u1": new_u1,
        "new_v1": new_v1,
        "new_q1": new_q1,
        "cnvw": cnvw,
        "cnvc": cnvc,
        "ud_mf": ud_mf,
        "dd_mf": dd_mf,
        "dt_mf": dt_mf,
        "pfld": pfld,
    }.items():
        require_real(name, tensor, (ncol, KM))
    for name, tensor in {"qtr": qtr, "new_qtr": new_qtr}.items():
        require_real(name, tensor, (ncol, KM, NN))

    (
        new_t1,
        new_u1,
        new_v1,
        new_q1,
        new_qtr,
        wet_dep,
        cnvw,
        cnvc,
        ud_mf,
        dd_mf,
        dt_mf,
        ten_t,
        ten_u,
        ten_v,
        ten_q,
        dqtr,
    ) = run_compile_island(
        "u_outputs",
        "final_outputs",
        _final_outputs_block,
        cnvflg,
        rn,
        kmax,
        kb,
        kbcon,
        ktcon,
        ktop,
        jmin,
        eta,
        etad,
        edto,
        xmb,
        cnvwt,
        dellal,
        t1,
        u1,
        v1,
        q1,
        qtr,
        state.require("to"),
        state.require("qo"),
        state.require("uo"),
        state.require("vo"),
        state.require("ctro"),
        new_t1,
        new_u1,
        new_v1,
        new_q1,
        new_qtr,
        cnvw,
        cnvc,
        ud_mf,
        dd_mf,
        dt_mf,
        wet_dep,
        pfld,
        sigmagfm,
        state.require("delt"),
        state.require("cscale"),
        state.require("rd"),
        bool(state.config.get("hwrf_samfdeep", False)),
        bool(state.config.get("progsigma", False)),
        int(state.config.get("ncloud", 0)),
        int(state.config.get("ntc", 0)),
        int(state.config.get("ntk", 0)),
    )

    ntrac = tracer_layout(state).ntrac
    if ten_q.shape[2] != ntrac:
        # samfdeepcnv sets only ten_q(:,:,1) of the host-tracer-sized array.
        vapour = ten_q[:, :, 0]
        ten_q = torch.zeros((ncol, KM, ntrac), dtype=ten_q.dtype, device=ten_q.device)
        ten_q[:, :, 0] = vapour
    return state.with_updates(
        {
            "new_t1": new_t1,
            "new_u1": new_u1,
            "new_v1": new_v1,
            "new_q1": new_q1,
            "new_qtr": new_qtr,
            "wet_dep": wet_dep,
            "cnvw": cnvw,
            "cnvc": cnvc,
            "ud_mf": ud_mf,
            "dd_mf": dd_mf,
            "dt_mf": dt_mf,
            "ten_t": ten_t,
            "ten_u": ten_u,
            "ten_v": ten_v,
            "ten_q": ten_q,
            "dqtr": dqtr,
        }
    )


def _final_outputs_block(
    cnvflg: torch.Tensor,
    rn: torch.Tensor,
    kmax: torch.Tensor,
    kb: torch.Tensor,
    kbcon: torch.Tensor,
    ktcon: torch.Tensor,
    ktop: torch.Tensor,
    jmin: torch.Tensor,
    eta: torch.Tensor,
    etad: torch.Tensor,
    edto: torch.Tensor,
    xmb: torch.Tensor,
    cnvwt: torch.Tensor,
    dellal: torch.Tensor,
    t1: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    q1: torch.Tensor,
    qtr: torch.Tensor,
    to: torch.Tensor,
    qo: torch.Tensor,
    uo: torch.Tensor,
    vo: torch.Tensor,
    ctro: torch.Tensor,
    new_t1: torch.Tensor,
    new_u1: torch.Tensor,
    new_v1: torch.Tensor,
    new_q1: torch.Tensor,
    new_qtr: torch.Tensor,
    cnvw: torch.Tensor,
    cnvc: torch.Tensor,
    ud_mf: torch.Tensor,
    dd_mf: torch.Tensor,
    dt_mf: torch.Tensor,
    wet_dep: torch.Tensor,
    pfld: torch.Tensor,
    sigmagfm: torch.Tensor,
    delt: torch.Tensor,
    cscale: torch.Tensor,
    rd: torch.Tensor,
    hwrf_samfdeep: bool,
    progsigma: bool,
    ncloud: int,
    ntc: int,
    ntk: int,
) -> tuple[torch.Tensor, ...]:
    KM = q1.shape[1]
    NTR = ctro.shape[2]
    ncol = cnvflg.shape[0]
    levels = torch.arange(1, KM + 1, dtype=cnvflg.dtype, device=cnvflg.device).reshape(1, KM)
    active_rain = (cnvflg != 0.0) & (rn > 0.0)
    active_rain_levels = active_rain[:, None]

    cloud = active_rain_levels & (levels >= kbcon[:, None]) & (levels < ktcon[:, None])
    cnvw = torch.where(cloud, cnvwt * xmb[:, None] * delt * cscale, cnvw)
    cover = 0.04 * torch.log1p(675.0 * eta * xmb[:, None])
    cnvc = torch.where(cloud, torch.clamp(cover, min=0.0, max=0.6), cnvc)

    if ncloud > 0:
        cloud_liquid_layer = active_rain_levels & (levels >= kbcon[:, None]) & (levels <= ktcon[:, None])
        tem = dellal * xmb[:, None] * delt
        ice_fraction = torch.clamp((263.16 - new_t1) * (1.0 / (263.16 - 233.16)), min=0.0, max=1.0)
        has_water_slot = new_qtr[:, :, 1] > -999.0
        qtr0 = new_qtr[:, :, 0]
        qtr1 = new_qtr[:, :, 1]
        qtr0 = torch.where(cloud_liquid_layer & has_water_slot, qtr0 + tem * ice_fraction, qtr0)
        qtr1 = torch.where(cloud_liquid_layer & has_water_slot, qtr1 + tem * (1.0 - ice_fraction), qtr1)
        qtr0 = torch.where(cloud_liquid_layer & ~has_water_slot, qtr0 + tem, qtr0)
        new_qtr = new_qtr.clone()
        new_qtr[:, :, 0] = qtr0
        new_qtr[:, :, 1] = qtr1

    dry_active = (cnvflg != 0.0) & (rn <= 0.0)
    dry_layer = dry_active[:, None] & (levels <= kmax[:, None])
    new_t1 = torch.where(dry_layer, to, new_t1)
    new_q1 = torch.where(dry_layer, qo, new_q1)
    new_u1 = torch.where(dry_layer, uo, new_u1)
    new_v1 = torch.where(dry_layer, vo, new_v1)
    if not hwrf_samfdeep:
        new_qtr = new_qtr.clone()
        new_qtr[:, :, 2 : 2 + NTR] = torch.where(dry_layer[:, :, None], ctro, new_qtr[:, :, 2 : 2 + NTR])
    if not hwrf_samfdeep and ntc > 0:
        wet_levels = (levels > 1.0) & (levels < float(KM))
        wet_layer = dry_active[:, None, None] & wet_levels[:, :, None] & (levels[:, :, None] <= ktcon[:, None, None])
        wet_dep = wet_dep.clone()
        wet_dep[:, :, :ntc] = torch.where(wet_layer, torch.zeros_like(wet_dep[:, :, :ntc]), wet_dep[:, :, :ntc])

    up = active_rain_levels & (levels >= kb[:, None]) & (levels < ktop[:, None])
    ud_mf = torch.where(up, eta * xmb[:, None] * delt, ud_mf)
    down = active_rain_levels & (levels <= jmin[:, None])
    dd_mf = torch.where(down, edto[:, None] * etad * xmb[:, None] * delt, dd_mf)
    top_minus_one = active_rain_levels & (ktop[:, None] == levels + 1.0)
    dt_mf = torch.where(top_minus_one, ud_mf, dt_mf)

    if (not hwrf_samfdeep) and ntk > 0:
        ntk_index = ntk - 1
        mid_levels = torch.arange(2, KM, dtype=cnvflg.dtype, device=cnvflg.device).reshape(1, KM - 2)
        sigma = sigmagfm if progsigma else torch.maximum(sigmagfm, torch.full_like(sigmagfm, 0.03))
        qtr_ntk = new_qtr[:, :, ntk_index]
        up_tke = active_rain[:, None] & (mid_levels > kb[:, None]) & (mid_levels < ktop[:, None])
        tem = 0.5 * (eta[:, : KM - 2] + eta[:, 1 : KM - 1]) * xmb[:, None]
        rho = pfld[:, 1 : KM - 1] * 100.0 / (rd * new_t1[:, 1 : KM - 1])
        # Rejected columns may have sigma=0. Fortran enters this division only
        # on the TKE update branch; unselected 0/0 poisons masked reverse mode.
        ptem = tem / torch.where(up_tke, sigma[:, None] * rho, 1.0)
        qtr_mid = torch.where(up_tke, qtr_ntk[:, 1 : KM - 1] + 0.5 * sigma[:, None] * ptem * ptem, qtr_ntk[:, 1 : KM - 1])
        down_tke = active_rain[:, None] & (mid_levels <= jmin[:, None])
        tem = 0.5 * edto[:, None] * (etad[:, : KM - 2] + etad[:, 1 : KM - 1]) * xmb[:, None]
        rho = pfld[:, 1 : KM - 1] * 100.0 / (rd * new_t1[:, 1 : KM - 1])
        ptem = tem / torch.where(down_tke, sigma[:, None] * rho, 1.0)
        qtr_mid = torch.where(down_tke, qtr_mid + 0.5 * sigma[:, None] * ptem * ptem, qtr_mid)
        new_qtr = new_qtr.clone()
        new_qtr[:, 1 : KM - 1, ntk_index] = qtr_mid

    ten_t = (new_t1 - t1) / delt
    ten_u = (new_u1 - u1) / delt
    ten_v = (new_v1 - v1) / delt
    ten_q = torch.zeros((ncol, KM, NTR), dtype=cnvflg.dtype, device=cnvflg.device)
    ten_q[:, :, 0] = (new_q1 - q1) / delt
    dqtr = (new_qtr - qtr) / delt
    return new_t1, new_u1, new_v1, new_q1, new_qtr, wet_dep, cnvw, cnvc, ud_mf, dd_mf, dt_mf, ten_t, ten_u, ten_v, ten_q, dqtr


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
