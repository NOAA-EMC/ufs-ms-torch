# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment T: aerosol wet deposition carry-through and precipitation evaporation."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch import recurrence_chunks as chunks
from sa_sas_deep_convection_torch.compile import masked_sum_chunk_size, run_compile_island, run_recurrence_chunk, recurrence_chunk_size, ordered_chunks_enabled
from sa_sas_deep_convection_torch.constants import EPS, EPSM1, KM
from sa_sas_deep_convection_torch.segments.segment_d import _fpvs_from_table
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real
from sa_sas_deep_convection_torch.thermo import _fpvs_table_for_device

CHECKPOINT_FIELDS = ("wet_dep", "chem_c", "chem_pw", "qevap", "delqev", "rainevap", "rn")


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 3373-3500 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    cnvflg = state.require("cnvflg")
    kmax = state.require("kmax")
    ktcon = state.require("ktcon")
    kb = state.require("kb")
    kbcon = state.require("kbcon")
    jmin = state.require("jmin")
    del_ = state.require("del")
    pfld = state.require("pfld")
    pwo = state.require("pwo")
    pwdo = state.require("pwdo")
    edto = state.require("edto")
    xmb = state.require("xmb")
    qeso = state.require("qeso").clone()
    new_t1 = state.require("new_t1").clone()
    new_q1 = state.require("new_q1").clone()
    rn = state.require("rn").clone()
    delqbar = state.require("delqbar").clone()
    deltbar = state.require("deltbar").clone()
    ktop = state.require("ktop").clone()
    kbot = state.require("kbot").clone()
    kcnv = state.require("kcnv").clone()
    cldwrk = state.require("cldwrk").clone()
    ncol = cnvflg.shape[0]
    for name, tensor in {
        "cnvflg": cnvflg,
        "kmax": kmax,
        "ktcon": ktcon,
        "kb": kb,
        "kbcon": kbcon,
        "jmin": jmin,
        "edto": edto,
        "xmb": xmb,
        "rn": rn,
        "delqbar": delqbar,
        "deltbar": deltbar,
        "ktop": ktop,
        "kbot": kbot,
        "kcnv": kcnv,
        "cldwrk": cldwrk,
    }.items():
        require_real(name, tensor, (ncol,))
    for name, tensor in {"del": del_, "pfld": pfld, "pwo": pwo, "pwdo": pwdo, "qeso": qeso, "new_t1": new_t1, "new_q1": new_q1}.items():
        require_real(name, tensor, (ncol, KM))

    qcond = state.require("qcond").clone()
    rainevap = state.require("rainevap").clone()
    (
        qeso,
        new_t1,
        new_q1,
        rn,
        rntot,
        qevap,
        delqev,
        delqbar,
        deltbar,
        qcond,
        rainevap,
        ktop,
        kbot,
        kcnv,
        cldwrk,
    ) = _recurrence_coordinator(
        "t_evap_wetdep",
        "evaporation_wetdep",
        _evaporation_wetdep_block,
        cnvflg,
        kmax,
        ktcon,
        kb,
        kbcon,
        jmin,
        del_,
        pfld,
        pwo,
        pwdo,
        edto,
        xmb,
        qeso,
        new_t1,
        new_q1,
        rn,
        delqbar,
        deltbar,
        ktop,
        kbot,
        kcnv,
        cldwrk,
        qcond,
        rainevap,
        state.require("aa1"),
        _fpvs_table_for_device(str(cnvflg.device), cnvflg.dtype),
        state.require("delt"),
        state.require("evef"),
        state.require("el2orc"),
        state.require("grav"),
        state.require("elocp"),
        bool(state.config.get("do_ca", False)),
    )

    return state.with_updates(
        {
            "qeso": qeso,
            "new_t1": new_t1,
            "new_q1": new_q1,
            "rn": rn,
            "rntot": rntot,
            "qevap": qevap,
            "delqev": delqev,
            "delqbar": delqbar,
            "deltbar": deltbar,
            "qcond": qcond,
            "rainevap": rainevap,
            "ktop": ktop,
            "kbot": kbot,
            "kcnv": kcnv,
            "cldwrk": cldwrk,
        }
    )


def _evaporation_wetdep_block(
    cnvflg: torch.Tensor,
    kmax: torch.Tensor,
    ktcon: torch.Tensor,
    kb: torch.Tensor,
    kbcon: torch.Tensor,
    jmin: torch.Tensor,
    del_: torch.Tensor,
    pfld: torch.Tensor,
    pwo: torch.Tensor,
    pwdo: torch.Tensor,
    edto: torch.Tensor,
    xmb: torch.Tensor,
    qeso: torch.Tensor,
    new_t1: torch.Tensor,
    new_q1: torch.Tensor,
    rn: torch.Tensor,
    delqbar: torch.Tensor,
    deltbar: torch.Tensor,
    ktop: torch.Tensor,
    kbot: torch.Tensor,
    kcnv: torch.Tensor,
    cldwrk: torch.Tensor,
    qcond: torch.Tensor,
    rainevap: torch.Tensor,
    aa1: torch.Tensor,
    fpvs_table: torch.Tensor,
    delt: torch.Tensor,
    evef: torch.Tensor,
    el2orc: torch.Tensor,
    grav: torch.Tensor,
    elocp: torch.Tensor,
    do_ca: bool,
) -> tuple[torch.Tensor, ...]:
    KM = qeso.shape[1]
    active = cnvflg != 0.0
    level_values = torch.arange(1, KM + 1, dtype=qeso.dtype, device=cnvflg.device).reshape(1, KM)
    qeso_layer = active[:, None] & (level_values <= kmax[:, None]) & (level_values <= ktcon[:, None])
    qeso_candidate = 0.01 * _fpvs_from_table(new_t1, fpvs_table)
    qeso_candidate = EPS * qeso_candidate / (pfld + EPSM1 * qeso_candidate)
    qeso = torch.where(qeso_layer, torch.clamp_min(qeso_candidate, 1.0e-8), qeso)

    rntot = torch.zeros_like(cnvflg)
    delqev = torch.zeros_like(cnvflg)
    delq2 = torch.zeros_like(cnvflg)
    flg = active.clone()
    levels = level_values.reshape(KM)
    size = recurrence_chunk_size()
    sum_size = masked_sum_chunk_size()
    rain_rows = chunks.pack_profiles(pwo, pwdo)
    for stop in range(KM, 0, -sum_size):
        start = max(0, stop - sum_size)
        rntot = run_recurrence_chunk("t_ordered_chunks", "rain_total", chunks.rain_total,
            active, kmax, ktcon, kb, jmin, levels[start], rain_rows[start:stop],
            edto, xmb, delt, rntot)
    del rain_rows

    qevap = torch.zeros_like(cnvflg)
    q1_pieces, t1_pieces = [], []
    inputs = chunks.pack_profiles(del_, new_q1, new_t1, qeso, pwo, pwdo)
    carry = (rn, qevap, delqev, delq2, qcond, delqbar, deltbar, flg)
    for stop in range(KM, 0, -size):
        start = max(0, stop - size)
        result, carry = run_recurrence_chunk("t_ordered_chunks", "evaporation", chunks.evaporation,
            active, kmax, ktcon, kb, jmin, levels[start],
            inputs[start:stop], carry, rntot,
            edto, xmb, delt, evef, el2orc, grav, elocp)
        q1_pieces.append(result[0])
        t1_pieces.append(result[1])
    rn, qevap, delqev, delq2, qcond, delqbar, deltbar, flg = carry
    new_q1 = chunks.join_levels(new_q1, q1_pieces[::-1], 0)
    new_t1 = chunks.join_levels(new_t1, t1_pieces[::-1], 0)

    if do_ca:
        rainevap = torch.where(active, delqev, rainevap)

    reset_negative = active & (rn < 0.0) & (~flg)
    rn = torch.where(reset_negative, torch.zeros_like(rn), rn)
    raining = active & (rn > 0.0)
    rn = torch.where(active & (rn <= 0.0), torch.zeros_like(rn), rn)
    ktop = torch.where(raining, ktcon, ktop)
    kbot = torch.where(raining, kbcon, kbot)
    kcnv = torch.where(raining, torch.ones_like(kcnv), kcnv)
    cldwrk = torch.where(raining, aa1, cldwrk)
    return qeso, new_t1, new_q1, rn, rntot, qevap, delqev, delqbar, deltbar, qcond, rainevap, ktop, kbot, kcnv, cldwrk


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}


def _recurrence_coordinator(category, name, fn, *args):
    if ordered_chunks_enabled("t_ordered_chunks"):
        return fn(*args)
    return run_compile_island(category, name, fn, *args)
