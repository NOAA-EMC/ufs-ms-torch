# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment S: feedback update, positive borrowing, and wet deposition."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch.compile import run_compile_island
from sa_sas_deep_convection_torch.constants import EPS, EPSM1, GRAV, KM, NN, NTC, NTR
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real
from sa_sas_deep_convection_torch.thermo import NXPVS, PVS_XMAX, PVS_XMIN, _fpvs_table_for_device

CHECKPOINT_FIELDS = ("new_t1", "new_u1", "new_v1", "new_q1", "new_qtr", "wet_dep", "cnvwt")


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 3122-3371 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    cnvflg = state.require("cnvflg")
    kmax = state.require("kmax")
    ktcon = state.require("ktcon")
    kb = state.require("kb")
    del_ = state.require("del")
    delp = state.require("delp")
    zi = state.require("zi")
    t1 = state.require("t1")
    q1 = state.require("q1")
    u1 = state.require("u1")
    v1 = state.require("v1")
    qtr = state.require("qtr")
    pfld = state.require("pfld")
    xmb = state.require("xmb")
    dellah = state.require("dellah")
    dellaq = state.require("dellaq")
    dellau = state.require("dellau")
    dellav = state.require("dellav")
    dellae = state.require("dellae")
    chem_pw = state.require("chem_pw")
    ctro = state.require("ctro").clone()
    ctr = state.require("ctr").clone()
    to = state.require("to").clone()
    qo = state.require("qo").clone()
    uo = state.require("uo").clone()
    vo = state.require("vo").clone()
    qeso = state.require("qeso").clone()
    new_t1 = state.require("new_t1").clone()
    new_q1 = state.require("new_q1").clone()
    new_u1 = state.require("new_u1").clone()
    new_v1 = state.require("new_v1").clone()
    new_qtr = state.require("new_qtr").clone()
    wet_dep = state.require("wet_dep").clone()
    ncol = cnvflg.shape[0]
    for name, tensor in {"cnvflg": cnvflg, "kmax": kmax, "ktcon": ktcon, "kb": kb, "xmb": xmb}.items():
        require_real(name, tensor, (ncol,))
    for name, tensor in {
        "del": del_,
        "delp": delp,
        "zi": zi,
        "t1": t1,
        "q1": q1,
        "u1": u1,
        "v1": v1,
        "pfld": pfld,
        "dellah": dellah,
        "dellaq": dellaq,
        "dellau": dellau,
        "dellav": dellav,
        "to": to,
        "qo": qo,
        "uo": uo,
        "vo": vo,
        "qeso": qeso,
        "new_t1": new_t1,
        "new_q1": new_q1,
        "new_u1": new_u1,
        "new_v1": new_v1,
    }.items():
        require_real(name, tensor, (ncol, KM))
    for name, tensor in {"qtr": qtr, "new_qtr": new_qtr}.items():
        require_real(name, tensor, (ncol, KM, NN))
    for name, tensor in {"ctr": ctr, "ctro": ctro, "dellae": dellae}.items():
        require_real(name, tensor, (ncol, KM, NTR))
    for name, tensor in {"chem_pw": chem_pw, "wet_dep": wet_dep}.items():
        require_real(name, tensor, (ncol, KM, NTC))

    (
        to,
        qo,
        uo,
        vo,
        qeso,
        ctro,
        ctr,
        new_t1,
        new_q1,
        new_u1,
        new_v1,
        new_qtr,
        wet_dep,
        delhbar,
        delqbar,
        deltbar,
        delubar,
        delvbar,
        qcond,
        delebar,
    ) = run_compile_island(
        "s_feedback",
        "feedback_borrow_wetdep",
        _feedback_borrow_wetdep_block,
        cnvflg,
        kmax,
        ktcon,
        kb,
        del_,
        delp,
        zi,
        t1,
        q1,
        u1,
        v1,
        qtr,
        pfld,
        xmb,
        dellah,
        dellaq,
        dellau,
        dellav,
        dellae,
        chem_pw,
        ctro,
        ctr,
        to,
        qo,
        uo,
        vo,
        qeso,
        new_t1,
        new_q1,
        new_u1,
        new_v1,
        new_qtr,
        wet_dep,
        _fpvs_table_for_device(str(t1.device), t1.dtype),
        state.require("delt"),
        state.require("hvap"),
        state.require("cp"),
        bool(state.config.get("hwrf_samfdeep", False)),
        int(state.config.get("ntk", 0)),
        int(state.config.get("itc", 0)),
        int(state.config.get("ntc", 0)),
        int(state.config.get("ntr", 0)),
    )

    return state.with_updates(
        {
            "to": to,
            "qo": qo,
            "uo": uo,
            "vo": vo,
            "qeso": qeso,
            "ctro": ctro,
            "ctr": ctr,
            "new_t1": new_t1,
            "new_q1": new_q1,
            "new_u1": new_u1,
            "new_v1": new_v1,
            "new_qtr": new_qtr,
            "wet_dep": wet_dep,
            "delhbar": delhbar,
            "delqbar": delqbar,
            "deltbar": deltbar,
            "delubar": delubar,
            "delvbar": delvbar,
            "qcond": qcond,
            "delebar": delebar,
        }
    )


def _ordered_sum(values: torch.Tensor) -> torch.Tensor:
    """Level sum in the source's loop order: torch.sum groups terms differently, which changes the last bit."""
    return values.cumsum(dim=1)[:, -1].clone()


def _feedback_borrow_wetdep_block(
    cnvflg: torch.Tensor,
    kmax: torch.Tensor,
    ktcon: torch.Tensor,
    kb: torch.Tensor,
    del_: torch.Tensor,
    delp: torch.Tensor,
    zi: torch.Tensor,
    t1: torch.Tensor,
    q1: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    qtr: torch.Tensor,
    pfld: torch.Tensor,
    xmb: torch.Tensor,
    dellah: torch.Tensor,
    dellaq: torch.Tensor,
    dellau: torch.Tensor,
    dellav: torch.Tensor,
    dellae: torch.Tensor,
    chem_pw: torch.Tensor,
    ctro: torch.Tensor,
    ctr: torch.Tensor,
    to: torch.Tensor,
    qo: torch.Tensor,
    uo: torch.Tensor,
    vo: torch.Tensor,
    qeso: torch.Tensor,
    new_t1: torch.Tensor,
    new_q1: torch.Tensor,
    new_u1: torch.Tensor,
    new_v1: torch.Tensor,
    new_qtr: torch.Tensor,
    wet_dep: torch.Tensor,
    fpvs_table: torch.Tensor,
    delt: torch.Tensor,
    hvap: torch.Tensor,
    cp: torch.Tensor,
    hwrf_samfdeep: bool,
    ntk: int,
    itc: int,
    ntc: int,
    ntr: int,
) -> tuple[torch.Tensor, ...]:
    KM = q1.shape[1]
    NTR = ntr
    NTC = ntc
    ncol = cnvflg.shape[0]
    active = cnvflg != 0.0
    levels = torch.arange(1, KM + 1, dtype=cnvflg.dtype, device=cnvflg.device).reshape(1, KM)
    active_to_kmax = active[:, None] & (levels <= kmax[:, None])
    active_to_ktcon = active[:, None] & (levels <= ktcon[:, None])
    feedback_layer = active_to_kmax & active_to_ktcon
    to = torch.where(active_to_kmax, t1, to)
    qo = torch.where(active_to_kmax, q1, qo)
    uo = torch.where(active_to_kmax, u1, uo)
    vo = torch.where(active_to_kmax, v1, vo)
    qeso_candidate = 0.01 * _fpvs_from_table(t1, fpvs_table)
    qeso_candidate = EPS * qeso_candidate / (pfld + EPSM1 * qeso_candidate)
    qeso = torch.where(active_to_kmax, torch.clamp_min(qeso_candidate, 1.0e-8), qeso)
    if not hwrf_samfdeep:
        ctro = torch.where(active_to_kmax[:, :, None], qtr[:, :, 2 : 2 + NTR], ctro)

    zero_col = torch.zeros_like(cnvflg)
    delhbar = zero_col.clone()
    delqbar = zero_col.clone()
    deltbar = zero_col.clone()
    delubar = zero_col.clone()
    delvbar = zero_col.clone()
    qcond = zero_col.clone()
    delebar = torch.zeros((ncol, NTR), dtype=cnvflg.dtype, device=cnvflg.device)

    tem2 = xmb[:, None] * delt
    dellat_all = (dellah - hvap * dellaq) / cp
    new_t1 = torch.where(feedback_layer, t1 + tem2 * dellat_all, new_t1)
    new_q1 = torch.where(feedback_layer, q1 + tem2 * dellaq, new_q1)
    new_u1 = torch.where(feedback_layer, u1 + tem2 * dellau, new_u1)
    new_v1 = torch.where(feedback_layer, v1 + tem2 * dellav, new_v1)

    tem_mass = xmb[:, None] * (1000.0 * del_) / GRAV
    delhbar = _ordered_sum(torch.where(feedback_layer, tem_mass * dellah, torch.zeros_like(dellah)))
    delqbar = _ordered_sum(torch.where(feedback_layer, tem_mass * dellaq, torch.zeros_like(dellaq)))
    deltbar = _ordered_sum(torch.where(feedback_layer, tem_mass * dellat_all, torch.zeros_like(dellat_all)))
    delubar = _ordered_sum(torch.where(feedback_layer, tem_mass * dellau, torch.zeros_like(dellau)))
    delvbar = _ordered_sum(torch.where(feedback_layer, tem_mass * dellav, torch.zeros_like(dellav)))

    prefix_levels = levels[:, : KM - 1]
    prefix_layer = active[:, None] & (prefix_levels <= ktcon[:, None])
    q1_prefix = new_q1[:, : KM - 1]
    q1_mass = q1_prefix * delp[:, : KM - 1] / GRAV
    tsumn = _ordered_sum(torch.where(prefix_layer & (q1_prefix < 0.0), q1_mass, torch.zeros_like(q1_mass)))
    tsump = _ordered_sum(torch.where(prefix_layer & (q1_prefix > 0.0), q1_mass, torch.zeros_like(q1_mass)))
    rtnp = torch.ones_like(cnvflg)
    need_borrow = active & (tsump > 0.0) & (tsumn < 0.0)
    pos_dominates = tsump > torch.abs(tsumn)
    safe_tsump = torch.where(need_borrow & pos_dominates, tsump, torch.ones_like(tsump))
    safe_tsumn = torch.where(need_borrow & ~pos_dominates, tsumn, -torch.ones_like(tsumn))
    rtnp = torch.where(need_borrow & pos_dominates, tsumn / safe_tsump, rtnp)
    rtnp = torch.where(need_borrow & ~pos_dominates, tsump / safe_tsumn, rtnp)
    borrow_layer = active[:, None] & (levels < float(KM)) & (levels <= ktcon[:, None]) & (rtnp[:, None] < 0.0)
    pos_matrix = pos_dominates[:, None]
    rtnp_matrix = rtnp[:, None]
    adjusted_q1 = torch.where(pos_matrix & (new_q1 < 0.0), torch.zeros_like(new_q1), new_q1)
    adjusted_q1 = torch.where(pos_matrix & (new_q1 > 0.0), (1.0 + rtnp_matrix) * new_q1, adjusted_q1)
    adjusted_q1 = torch.where((~pos_matrix) & (new_q1 < 0.0), (1.0 + rtnp_matrix) * new_q1, adjusted_q1)
    adjusted_q1 = torch.where((~pos_matrix) & (new_q1 > 0.0), torch.zeros_like(new_q1), adjusted_q1)
    new_q1 = torch.where(borrow_layer, adjusted_q1, new_q1)

    if not hwrf_samfdeep:
        tke_tracer_one_based = ntk - 2
        tracer_feedback = feedback_layer[:, :, None]
        ctr = torch.where(tracer_feedback, ctr + dellae * xmb[:, None, None] * delt, ctr)
        delebar = _ordered_sum(torch.where(tracer_feedback, dellae * xmb[:, None, None] * (1000.0 * del_)[:, :, None] / GRAV, torch.zeros_like(dellae)))

        dz_prefix = torch.zeros_like(delp[:, : KM - 1])
        dz_prefix[:, 0] = zi[:, 0]
        dz_prefix[:, 1:] = zi[:, 1 : KM - 1] - zi[:, : KM - 2]
        tracer_ids = torch.arange(1, NTR + 1, dtype=cnvflg.dtype, device=cnvflg.device).reshape(1, 1, NTR)
        ctr_prefix = ctr[:, : KM - 1, :]
        ctr_mass = torch.where(
            tracer_ids == float(tke_tracer_one_based),
            ctr_prefix * dz_prefix[:, :, None],
            ctr_prefix * delp[:, : KM - 1, None] / GRAV,
        )
        tracer_prefix_layer = prefix_layer[:, :, None]
        tsumn_tr = _ordered_sum(torch.where(tracer_prefix_layer & (ctr_prefix < 0.0), ctr_mass, torch.zeros_like(ctr_mass)))
        tsump_tr = _ordered_sum(torch.where(tracer_prefix_layer & (ctr_prefix > 0.0), ctr_mass, torch.zeros_like(ctr_mass)))
        rtnp_tr = torch.ones_like(tsump_tr)
        need_borrow_tr = active[:, None] & (tsump_tr > 0.0) & (tsumn_tr < 0.0)
        pos_tr = tsump_tr > torch.abs(tsumn_tr)
        safe_tsump_tr = torch.where(need_borrow_tr & pos_tr, tsump_tr, torch.ones_like(tsump_tr))
        safe_tsumn_tr = torch.where(need_borrow_tr & ~pos_tr, tsumn_tr, -torch.ones_like(tsumn_tr))
        rtnp_tr = torch.where(need_borrow_tr & pos_tr, tsumn_tr / safe_tsump_tr, rtnp_tr)
        rtnp_tr = torch.where(need_borrow_tr & ~pos_tr, tsump_tr / safe_tsumn_tr, rtnp_tr)
        borrow_tracer_layer = active[:, None, None] & (levels[:, :, None] < float(KM)) & (levels[:, :, None] <= ktcon[:, None, None]) & (rtnp_tr[:, None, :] < 0.0)
        pos_tr_matrix = pos_tr[:, None, :]
        rtnp_tr_matrix = rtnp_tr[:, None, :]
        adjusted_ctr = torch.where(pos_tr_matrix & (ctr < 0.0), torch.zeros_like(ctr), ctr)
        adjusted_ctr = torch.where(pos_tr_matrix & (ctr > 0.0), (1.0 + rtnp_tr_matrix) * ctr, adjusted_ctr)
        adjusted_ctr = torch.where((~pos_tr_matrix) & (ctr < 0.0), (1.0 + rtnp_tr_matrix) * ctr, adjusted_ctr)
        adjusted_ctr = torch.where((~pos_tr_matrix) & (ctr > 0.0), torch.zeros_like(ctr), adjusted_ctr)
        ctr = torch.where(borrow_tracer_layer, adjusted_ctr, ctr)
        new_qtr_tracers = torch.where(active_to_ktcon[:, :, None], ctr, new_qtr[:, :, 2 : 2 + NTR])
        new_qtr = new_qtr.clone()
        new_qtr[:, :, 2 : 2 + NTR] = new_qtr_tracers

        do_aerosols = itc > 0 and ntc > 0 and ntr >= itc + ntc - 3
        if do_aerosols:
            wet_layer = active[:, None, None] & (levels[:, :, None] > kb[:, None, None]) & (levels[:, :, None] < ktcon[:, None, None])
            dp_aerosol = (1000.0 * del_)[:, :, None]
            wet_val = chem_pw * GRAV / dp_aerosol
            wet_val = wet_val * xmb[:, None, None] * delt * dp_aerosol
            wet_dep = torch.where(wet_layer, wet_val, wet_dep)
            qtr_start = itc - 1
            qtr_stop = qtr_start + NTC
            qtr_aerosols = new_qtr[:, :, qtr_start:qtr_stop]
            layer = wet_layer & (qtr_aerosols < 0.0)
            needed = -qtr_aerosols * dp_aerosol
            enough = wet_dep >= needed
            wet_after_enough = torch.where(layer & enough, wet_dep - needed, wet_dep)
            qtr_aerosols = torch.where(layer & enough, torch.zeros_like(qtr_aerosols), qtr_aerosols)
            wet_dep = torch.where(layer & ~enough, torch.zeros_like(wet_after_enough), wet_after_enough)
            new_qtr = new_qtr.clone()
            new_qtr[:, :, qtr_start:qtr_stop] = qtr_aerosols

    return (
        to,
        qo,
        uo,
        vo,
        qeso,
        ctro,
        ctr,
        new_t1,
        new_q1,
        new_u1,
        new_v1,
        new_qtr,
        wet_dep,
        delhbar,
        delqbar,
        deltbar,
        delubar,
        delvbar,
        qcond,
        delebar,
    )


def _fpvs_from_table(temperature: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    xinc = (PVS_XMAX - PVS_XMIN) / (NXPVS - 1)
    c2 = 1.0 / xinc
    c1 = 1.0 - PVS_XMIN * c2
    xj = torch.clamp(c1 + c2 * temperature, 1.0, float(NXPVS))
    jx = torch.minimum(xj.to(torch.int64), torch.full_like(xj, NXPVS - 1, dtype=torch.int64))
    return table[jx - 1] + (xj - jx.to(xj.dtype)) * (table[jx] - table[jx - 1])


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
