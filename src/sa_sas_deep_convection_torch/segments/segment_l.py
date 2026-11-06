# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment L: downdraft shear, moisture, and cloud work."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch import recurrence_chunks as chunks

from sa_sas_deep_convection_torch.compile import masked_sum_chunk_size, run_compile_island, run_recurrence_chunk, recurrence_chunk_size, ordered_chunks_enabled
from sa_sas_deep_convection_torch.constants import KM, NTR
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real

CHECKPOINT_FIELDS = ("hcdo", "qcdo", "ucdo", "vcdo", "ecdo", "edt", "edto", "cnvflg", "return_stage")


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 2088-2236 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    cnvflg = state.require("cnvflg")
    jmin = state.require("jmin")
    kd94 = state.require("kd94")
    zi = state.require("zi")
    zo = state.require("zo")
    heo = state.require("heo")
    heso = state.require("heso")
    qo = state.require("qo")
    qeso = state.require("qeso")
    to = state.require("to")
    uo = state.require("uo")
    vo = state.require("vo")
    ctro = state.require("ctro")
    etad = state.require("etad")
    xlamdet = state.require("xlamdet")
    xlamddt = state.require("xlamddt")
    xlamd = state.require("xlamd")
    hcdo = state.require("hcdo")
    qcdo = state.require("qcdo")
    qrcdo = state.require("qrcdo")
    ucdo = state.require("ucdo")
    vcdo = state.require("vcdo")
    ecdo = state.require("ecdo")
    dbyo = state.require("dbyo")
    pwdo = state.require("pwdo")
    pwevo = state.require("pwevo")
    edto = state.require("edto")
    aa1 = state.require("aa1")
    ncol = cnvflg.shape[0]
    for name, tensor in {
        "cnvflg": cnvflg,
        "jmin": jmin,
        "kd94": kd94,
        "xlamdet": xlamdet,
        "xlamddt": xlamddt,
        "xlamd": xlamd,
    }.items():
        require_real(name, tensor, (ncol,))
    for name, tensor in {
        "zi": zi,
        "zo": zo,
        "heo": heo,
        "heso": heso,
        "qo": qo,
        "qeso": qeso,
        "to": to,
        "uo": uo,
        "vo": vo,
        "etad": etad,
        "hcdo": hcdo,
        "qcdo": qcdo,
        "qrcdo": qrcdo,
        "ucdo": ucdo,
        "vcdo": vcdo,
        "dbyo": dbyo,
        "pwdo": pwdo,
    }.items():
        require_real(name, tensor, (ncol, KM))
    require_real("ctro", ctro, (ncol, KM, NTR))
    require_real("ecdo", ecdo, (ncol, KM, NTR))

    (
        hcdo,
        qcdo,
        qrcdo,
        ucdo,
        vcdo,
        ecdo,
        dbyo,
        pwdo,
        pwevo,
        edto,
        aa1,
        cnvflg,
    ) = _recurrence_coordinator(
        "l_downdraft",
        "downdraft_recurrence",
        _downdraft_recurrence_block,
        cnvflg,
        jmin,
        kd94,
        zi,
        zo,
        heo,
        heso,
        qo,
        qeso,
        to,
        uo,
        vo,
        ctro,
        etad,
        xlamdet,
        xlamddt,
        xlamd,
        hcdo,
        qcdo,
        qrcdo,
        ucdo,
        vcdo,
        ecdo,
        dbyo,
        pwdo,
        pwevo,
        edto,
        aa1,
        state.require("pgcon"),
        state.require("el2orc"),
        state.require("hvap"),
        state.require("pwavo"),
        state.require("grav"),
        state.require("cp"),
        state.require("fv"),
        bool(state.config.get("hwrf_samfdeep", False)),
    )

    return state.with_updates(
        {
            "hcdo": hcdo,
            "qcdo": qcdo,
            "qrcdo": qrcdo,
            "ucdo": ucdo,
            "vcdo": vcdo,
            "ecdo": ecdo,
            "dbyo": dbyo,
            "pwdo": pwdo,
            "pwevo": pwevo,
            "edto": edto,
            "aa1": aa1,
            "cnvflg": cnvflg,
            "return_stage": torch.zeros_like(cnvflg),
        }
    )


def _downdraft_recurrence_block(
    cnvflg: torch.Tensor,
    jmin: torch.Tensor,
    kd94: torch.Tensor,
    zi: torch.Tensor,
    zo: torch.Tensor,
    heo: torch.Tensor,
    heso: torch.Tensor,
    qo: torch.Tensor,
    qeso: torch.Tensor,
    to: torch.Tensor,
    uo: torch.Tensor,
    vo: torch.Tensor,
    ctro: torch.Tensor,
    etad: torch.Tensor,
    xlamdet: torch.Tensor,
    xlamddt: torch.Tensor,
    xlamd: torch.Tensor,
    hcdo: torch.Tensor,
    qcdo: torch.Tensor,
    qrcdo: torch.Tensor,
    ucdo: torch.Tensor,
    vcdo: torch.Tensor,
    ecdo: torch.Tensor,
    dbyo: torch.Tensor,
    pwdo: torch.Tensor,
    pwevo: torch.Tensor,
    edto: torch.Tensor,
    aa1: torch.Tensor,
    pgcon: torch.Tensor,
    el2orc: torch.Tensor,
    hvap: torch.Tensor,
    pwavo: torch.Tensor,
    grav: torch.Tensor,
    cp: torch.Tensor,
    fv: torch.Tensor,
    hwrf_samfdeep: bool,
) -> tuple[torch.Tensor, ...]:
    KM = zo.shape[1]
    NTR = ctro.shape[2]
    active = cnvflg != 0.0
    pwevo = torch.where(active, torch.zeros_like(pwevo), pwevo)
    size = recurrence_chunk_size()
    levels = torch.arange(1, KM + 1, dtype=cnvflg.dtype, device=cnvflg.device)
    seed = active[:, None] & (jmin[:, None] == levels[None, :])
    hcdo, qcdo, qrcdo = (torch.where(seed, environment, original)
                         for environment, original in ((heo, hcdo), (qo, qcdo), (qo, qrcdo)))
    ucdo, vcdo = torch.where(seed, uo, ucdo), torch.where(seed, vo, vcdo)
    if not hwrf_samfdeep:
        ecdo = torch.where(seed[:, :, None], ctro, ecdo)
    original = (hcdo, ucdo, vcdo, dbyo)
    pieces = [[] for _ in original]
    carry = tuple(value[:, -1].contiguous() for value in (hcdo, ucdo, vcdo))
    profile_rows = chunks.pack_profiles(zi, heo, heso, uo, vo, hcdo, ucdo, vcdo, dbyo)
    for stop in range(KM - 1, 0, -size):
        start = max(0, stop - size)
        result = run_recurrence_chunk("l_ordered_chunks", "downdraft_heat", chunks.downdraft_heat,
            active, jmin, kd94, levels[start], profile_rows[start:stop + 1],
            carry, xlamdet, xlamddt, xlamd, pgcon)
        for values, value in zip(pieces, result, strict=True):
            values.append(value)
        carry = tuple(value[:, 0] for value in result[:3])
    hcdo, ucdo, vcdo, dbyo = (
        chunks.join_levels(base, values[::-1], 0) for base, values in zip(original, pieces, strict=True))
    del profile_rows

    if not hwrf_samfdeep:
        pieces = []
        carry = ecdo[:, -1].contiguous()
        profile_rows = chunks.pack_profiles(zi)
        tracer_rows = chunks.pack_profiles(ctro, ecdo)
        for stop in range(KM - 1, 0, -size):
            start = max(0, stop - size)
            result = run_recurrence_chunk("l_ordered_chunks", "tracer_down", chunks.tracer_down,
                active, jmin, levels[start], profile_rows[start:stop + 1], tracer_rows[start:stop + 1], carry, xlamdet)
            pieces.append(result)
            carry = result[:, 0]
        ecdo = chunks.join_levels(ecdo, pieces[::-1], 0)
        del profile_rows, tracer_rows

    original = (qrcdo, qcdo, pwdo)
    pieces = [[] for _ in original]
    carry = qrcdo[:, -1].contiguous()
    profile_rows = chunks.pack_profiles(zi, qeso, to, dbyo, qo, etad, qrcdo, qcdo, pwdo)
    for stop in range(KM - 1, 0, -size):
        start = max(0, stop - size)
        result, pwevo = run_recurrence_chunk("l_ordered_chunks", "downdraft_moisture", chunks.downdraft_moisture,
            active, jmin, levels[start], profile_rows[start:stop + 1], carry, pwevo,
            xlamdet, el2orc, hvap)
        for values, value in zip(pieces, result, strict=True):
            values.append(value)
        carry = result[0][:, 0]
    qrcdo, qcdo, pwdo = (
        chunks.join_levels(base, values[::-1], 0) for base, values in zip(original, pieces, strict=True))
    del profile_rows

    edtmax = torch.full_like(edto, 0.3)
    edto_candidate = torch.where(pwevo < 0.0, torch.minimum(-edto * pwavo / torch.where(active & (pwevo < 0.0), pwevo, torch.ones_like(pwevo)), edtmax), torch.zeros_like(edto))
    edto = torch.where(active, edto_candidate, edto)

    profile_rows = chunks.pack_profiles(zo, qeso, to, hcdo, heso, qo)
    size = masked_sum_chunk_size()
    for stop in range(KM - 1, 0, -size):
        start = max(0, stop - size)
        aa1 = run_recurrence_chunk("l_ordered_chunks", "downdraft_work", chunks.downdraft_work,
            active, jmin, levels[start], profile_rows[start:stop + 1], edto, aa1, el2orc, grav, cp, fv, hvap)
    cnvflg = torch.where(active & (aa1 <= 0.0), torch.zeros_like(cnvflg), cnvflg)
    return hcdo, qcdo, qrcdo, ucdo, vcdo, ecdo, dbyo, pwdo, pwevo, edto, aa1, cnvflg


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}


def _recurrence_coordinator(category, name, fn, *args):
    if ordered_chunks_enabled("l_ordered_chunks"):
        return fn(*args)
    return run_compile_island(category, name, fn, *args)
