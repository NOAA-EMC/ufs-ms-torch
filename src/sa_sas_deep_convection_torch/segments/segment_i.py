# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment I: updraft moisture, precipitation, and cloud work."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch import recurrence_chunks as chunks
from sa_sas_deep_convection_torch.compile import masked_sum_chunk_size, run_compile_island, run_recurrence_chunk, recurrence_chunk_size, ordered_chunks_enabled
from sa_sas_deep_convection_torch.constants import FV, GRAV, KM
from sa_sas_deep_convection_torch.ops import sqrt_zero_grad
from sa_sas_deep_convection_torch.segments.segment_d import _gather_column
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real

CHECKPOINT_FIELDS = ("pwo", "pwdo", "qrcko", "aa1", "aa2", "cldwrk", "cnvflg", "return_stage")


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 1520-1704 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    cnvflg = state.require("cnvflg")
    kb = state.require("kb")
    kbcon = state.require("kbcon")
    kbcon1 = state.require("kbcon1")
    kbmax = state.require("kbmax")
    ktcon = state.require("ktcon")
    heo = state.require("heo")
    heso = state.require("heso")
    hcko = state.require("hcko")
    qo = state.require("qo")
    qeso = state.require("qeso")
    to = state.require("to")
    xlamue = state.require("xlamue")
    eta = state.require("eta")
    zi = state.require("zi")
    zo = state.require("zo")
    del_ = state.require("del")
    c0t = state.require("c0t")
    qcko = state.require("qcko")
    qrcko = state.require("qrcko")
    pwo = state.require("pwo")
    pwdo = state.require("pwdo")
    buo = state.require("buo")
    dellal = state.require("dellal")
    pwavo = state.require("pwavo")
    cnvwt = state.require("cnvwt")
    zdqca = state.require("zdqca")
    dbyo = state.require("dbyo")
    dbyo1 = state.require("dbyo1")
    drag = state.require("drag")
    wush = state.require("wush")
    uo = state.require("uo")
    vo = state.require("vo")
    ncol = cnvflg.shape[0]
    require_real("cnvflg", cnvflg, (ncol,))
    require_real("kb", kb, (ncol,))
    require_real("kbcon", kbcon, (ncol,))
    require_real("kbcon1", kbcon1, (ncol,))
    require_real("kbmax", kbmax, (ncol,))
    require_real("ktcon", ktcon, (ncol,))
    for name, tensor in {
        "heo": heo,
        "heso": heso,
        "hcko": hcko,
        "qo": qo,
        "qeso": qeso,
        "to": to,
        "xlamue": xlamue,
        "eta": eta,
        "zi": zi,
        "zo": zo,
        "del": del_,
        "c0t": c0t,
        "qcko": qcko,
        "qrcko": qrcko,
        "pwo": pwo,
        "pwdo": pwdo,
        "buo": buo,
        "dellal": dellal,
        "cnvwt": cnvwt,
        "zdqca": zdqca,
        "dbyo": dbyo,
        "dbyo1": dbyo1,
        "drag": drag,
        "wush": wush,
        "uo": uo,
        "vo": vo,
    }.items():
        require_real(name, tensor, (ncol, KM))

    (
        hmin,
        lmin,
        jmin,
        xmbmax,
        qcko,
        qrcko,
        pwo,
        pwdo,
        buo,
        dellal,
        pwavo,
        cnvwt,
        zdqca,
        drag,
        wush,
        aa1,
        dbyo1,
        cnvflg,
    ) = _recurrence_coordinator(
        "i_updraft_precip",
        "updraft_precip_cloud_work",
        _updraft_precip_cloud_work_block,
        cnvflg,
        kb,
        kbcon,
        kbcon1,
        kbmax,
        ktcon,
        heo,
        heso,
        hcko,
        qo,
        qeso,
        to,
        xlamue,
        eta,
        zi,
        zo,
        del_,
        c0t,
        qcko,
        qrcko,
        pwo,
        pwdo,
        buo,
        dellal,
        pwavo,
        cnvwt,
        zdqca,
        dbyo,
        dbyo1,
        drag,
        wush,
        uo,
        vo,
        state.require("lmin"),
        state.require("jmin"),
        state.require("aa1"),
        state.require("delt"),
        state.require("el2orc"),
        state.require("hvap"),
        state.require("c1"),
        state.require("cp"),
        state.require("xlamud"),
        int(state.config.get("ncloud", 0)),
    )

    return state.with_updates(
        {
            "hmin": hmin,
            "lmin": lmin,
            "jmin": jmin,
            "xmbmax": xmbmax,
            "qcko": qcko,
            "qrcko": qrcko,
            "pwo": pwo,
            "pwdo": pwdo,
            "buo": buo,
            "dellal": dellal,
            "pwavo": pwavo,
            "cnvwt": cnvwt,
            "zdqca": zdqca,
            "drag": drag,
            "wush": wush,
            "aa1": aa1,
            "dbyo1": dbyo1,
            "cnvflg": cnvflg,
            "return_stage": torch.zeros_like(cnvflg),
        }
    )


def _updraft_precip_cloud_work_block(
    cnvflg: torch.Tensor,
    kb: torch.Tensor,
    kbcon: torch.Tensor,
    kbcon1: torch.Tensor,
    kbmax: torch.Tensor,
    ktcon: torch.Tensor,
    heo: torch.Tensor,
    heso: torch.Tensor,
    hcko: torch.Tensor,
    qo: torch.Tensor,
    qeso: torch.Tensor,
    to: torch.Tensor,
    xlamue: torch.Tensor,
    eta: torch.Tensor,
    zi: torch.Tensor,
    zo: torch.Tensor,
    del_: torch.Tensor,
    c0t: torch.Tensor,
    qcko: torch.Tensor,
    qrcko: torch.Tensor,
    pwo: torch.Tensor,
    pwdo: torch.Tensor,
    buo: torch.Tensor,
    dellal: torch.Tensor,
    pwavo: torch.Tensor,
    cnvwt: torch.Tensor,
    zdqca: torch.Tensor,
    dbyo: torch.Tensor,
    dbyo1: torch.Tensor,
    drag: torch.Tensor,
    wush: torch.Tensor,
    uo: torch.Tensor,
    vo: torch.Tensor,
    lmin_input: torch.Tensor,
    jmin_input: torch.Tensor,
    aa1_input: torch.Tensor,
    delt: torch.Tensor,
    el2orc: torch.Tensor,
    hvap: torch.Tensor,
    c1: torch.Tensor,
    cp: torch.Tensor,
    xlamud: torch.Tensor,
    ncloud: int,
) -> tuple[torch.Tensor, ...]:
    KM = zo.shape[1]
    active = cnvflg != 0.0
    hmin = torch.where(active, _gather_column(heo, kbcon1), torch.zeros_like(cnvflg))
    lmin = torch.where(active, kbmax, lmin_input)
    jmin = torch.where(active, kbmax, jmin_input)
    level_indices = torch.arange(1, KM + 1, dtype=cnvflg.dtype, device=cnvflg.device)
    size = recurrence_chunk_size()
    search_rows = chunks.pack_profiles(heo)
    for start in range(1, KM - 1, size):
        stop = min(start + size, KM - 1)
        hmin, lmin = run_recurrence_chunk("i_ordered_chunks", "updraft_minimum", chunks.updraft_minimum,
            active, kbmax, kbcon1, level_indices[start], search_rows[start:stop], hmin, lmin)
    del search_rows
    jmin = torch.where(active, torch.minimum(lmin, ktcon - 1.0), jmin)
    jmin = torch.where(active, torch.maximum(jmin, kbcon1 + 1.0), jmin)
    cnvflg = torch.where(active & (jmin >= ktcon), torch.zeros_like(cnvflg), cnvflg)

    active = cnvflg != 0.0
    xmbmax = torch.zeros_like(cnvflg)
    xmbmax_candidate = 1000.0 * _gather_column(del_, kbcon) / (GRAV * delt)
    xmbmax = torch.where(active, xmbmax_candidate, xmbmax)

    level_values = torch.arange(1, KM + 1, dtype=cnvflg.dtype, device=cnvflg.device).reshape(1, KM)
    base_mask = active[:, None] & (kb[:, None] == level_values)
    qcko = torch.where(base_mask, qo, qcko)
    qrcko = torch.where(base_mask, qo, qrcko)

    original = (qcko, qrcko, pwo, buo, dellal, cnvwt, zdqca, drag, wush)
    pieces = [[] for _ in original]
    inputs = chunks.pack_profiles(zi, qeso, to, dbyo, xlamue, qo, eta, del_, c0t, qcko, qrcko, pwo,
              buo, dellal, cnvwt, zdqca, drag, wush, xlamud, uo, vo)
    qcko_previous = qcko[:, 0].contiguous()
    for start in range(1, KM - 1, size):
        stop = min(start + size, KM - 1)
        result, pwavo = run_recurrence_chunk("i_ordered_chunks", "precip_updraft", chunks.precip_updraft,
            active, kb, ktcon, kbcon, jmin, level_indices[start],
            inputs[start - 1:stop + 1],
            qcko_previous, pwavo, el2orc, hvap, c1, cp, ncloud)
        for values, value in zip(pieces, result, strict=True):
            values.append(value)
        qcko_previous = result[0][:, -1]
    qcko, qrcko, pwo, buo, dellal, cnvwt, zdqca, drag, wush = (
        chunks.join_levels(base, values, 1) for base, values in zip(original, pieces, strict=True))
    del inputs

    aa1 = torch.where(cnvflg != 0.0, torch.zeros_like(aa1_input), aa1_input)
    work_rows = chunks.pack_profiles(zo, buo)
    size = masked_sum_chunk_size()
    for start in range(1, KM - 1, size):
        stop = min(start + size, KM - 1)
        aa1 = run_recurrence_chunk("i_ordered_chunks", "updraft_work", chunks.updraft_work,
            active, kbcon, ktcon, level_indices[start], work_rows[start:stop + 1], aa1)
    work_mask = (cnvflg != 0.0)[:, None] & (level_values >= kbcon[:, None]) & (level_values < ktcon[:, None])
    work_mask = work_mask & (level_values > 1.0) & (level_values < float(KM))
    dbyo1 = torch.where(work_mask, hcko - heso, dbyo1)
    cnvflg = torch.where((cnvflg != 0.0) & (aa1 <= 0.0), torch.zeros_like(cnvflg), cnvflg)
    return hmin, lmin, jmin, xmbmax, qcko, qrcko, pwo, pwdo, buo, dellal, pwavo, cnvwt, zdqca, drag, wush, aa1, dbyo1, cnvflg


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}


def _recurrence_coordinator(category, name, fn, *args):
    if ordered_chunks_enabled("i_ordered_chunks"):
        return fn(*args)
    return run_compile_island(category, name, fn, *args)
