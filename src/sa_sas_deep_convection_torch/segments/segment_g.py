# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment G: mass-flux and updraft tracer recurrence."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch import recurrence_chunks as chunks

from sa_sas_deep_convection_torch.compile import run_compile_island, run_recurrence_chunk, recurrence_chunk_size, ordered_chunks_enabled
from sa_sas_deep_convection_torch.constants import KM, NTC, NTR
from sa_sas_deep_convection_torch.segments.segment_d import _gather_column
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real

CHECKPOINT_FIELDS = ("eta", "hcko", "qcko", "ecko", "ercko", "chem_c", "chem_pw", "xlamue")


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 1140-1343 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    cnvflg = state.require("cnvflg")
    kb = state.require("kb")
    kbcon = state.require("kbcon")
    kmax = state.require("kmax")
    ktconn = state.require("ktconn")
    qeso = state.require("qeso")
    frh = state.require("frh")
    zi = state.require("zi")
    eta = state.require("eta")
    xlamue = state.require("xlamue")
    xlamud = state.require("xlamud")
    heo = state.require("heo")
    heso = state.require("heso")
    hcko = state.require("hcko")
    qcko = state.require("qcko")
    ucko = state.require("ucko")
    vcko = state.require("vcko")
    uo = state.require("uo")
    vo = state.require("vo")
    dbyo = state.require("dbyo")
    ctro = state.require("ctro")
    ecko = state.require("ecko")
    ercko = state.require("ercko")
    chem_c = state.require("chem_c")
    chem_pw = state.require("chem_pw")
    c0t = state.require("c0t")
    fscav = state.require("fscav")
    ncol = cnvflg.shape[0]
    require_real("cnvflg", cnvflg, (ncol,))
    require_real("kb", kb, (ncol,))
    require_real("kbcon", kbcon, (ncol,))
    require_real("kmax", kmax, (ncol,))
    require_real("ktconn", ktconn, (ncol,))
    require_real("qeso", qeso, (ncol, KM))
    require_real("frh", frh, (ncol, KM))
    require_real("zi", zi, (ncol, KM))
    require_real("eta", eta, (ncol, KM))
    require_real("xlamue", xlamue, (ncol, KM))
    require_real("xlamud", xlamud, (ncol, KM))
    require_real("heo", heo, (ncol, KM))
    require_real("heso", heso, (ncol, KM))
    require_real("hcko", hcko, (ncol, KM))
    require_real("qcko", qcko, (ncol, KM))
    require_real("ucko", ucko, (ncol, KM))
    require_real("vcko", vcko, (ncol, KM))
    require_real("uo", uo, (ncol, KM))
    require_real("vo", vo, (ncol, KM))
    require_real("dbyo", dbyo, (ncol, KM))
    require_real("ctro", ctro, (ncol, KM, NTR))
    require_real("ecko", ecko, (ncol, KM, NTR))
    require_real("ercko", ercko, (ncol, KM, NTR))
    require_real("chem_c", chem_c, (ncol, KM, NTC))
    require_real("chem_pw", chem_pw, (ncol, KM, NTC))
    require_real("c0t", c0t, (ncol, KM))
    require_real("fscav", fscav, (NTC,))

    active = cnvflg != 0.0
    hwrf_samfdeep = bool(state.config.get("hwrf_samfdeep", False))
    itc = int(state.config.get("itc", 0))
    ntc = int(state.config.get("ntc", 0))
    ntk = int(state.config.get("ntk", 0))
    ntr = int(state.config.get("ntr", NTR))
    do_aerosols = False if hwrf_samfdeep else (itc > 0 and ntc > 0 and ntr > 0 and ntr >= itc + ntc - 3)

    fent1 = state.require("fent1")
    fent2 = state.require("fent2")
    fent1, fent2, xlamue, xlamud = run_compile_island(
        "g_entrainment_formula",
        "entrainment_formula",
        _entrainment_formula_block,
        active,
        kbcon,
        kmax,
        qeso,
        frh,
        fent1,
        fent2,
        xlamue,
        xlamud,
        state.require("cxlamet"),
        state.require("cxlamdt"),
        hwrf_samfdeep,
    )

    (
        eta,
        kmax,
        ktconn,
        hcko,
        ucko,
        vcko,
        pwavo,
        ecko,
        ercko,
        dbyo,
        chem_c,
        chem_pw,
    ) = _recurrence_coordinator(
        "g_mass_flux_recurrence",
        "mass_flux_recurrence",
        _mass_flux_recurrence_block,
        active,
        kb,
        kbcon,
        kmax,
        ktconn,
        zi,
        eta,
        xlamue,
        xlamud,
        heo,
        heso,
        hcko,
        ucko,
        vcko,
        uo,
        vo,
        dbyo,
        ctro,
        ecko,
        ercko,
        chem_c,
        chem_pw,
        state.require("pwavo"),
        c0t,
        fscav,
        state.require("pgcon"),
        hwrf_samfdeep,
        do_aerosols,
        itc,
        ntc,
        ntk,
        ntr,
    )

    return state.with_updates(
        {
            "fent1": fent1,
            "fent2": fent2,
            "xlamue": xlamue,
            "xlamud": xlamud,
            "eta": eta,
            "kmax": kmax,
            "ktconn": ktconn,
            "hcko": hcko,
            "qcko": qcko,
            "ucko": ucko,
            "vcko": vcko,
            "pwavo": pwavo,
            "ecko": ecko,
            "ercko": ercko,
            "dbyo": dbyo,
            "chem_c": chem_c,
            "chem_pw": chem_pw,
        }
    )


def _entrainment_formula_block(
    active: torch.Tensor,
    kbcon: torch.Tensor,
    kmax: torch.Tensor,
    qeso: torch.Tensor,
    frh: torch.Tensor,
    fent1: torch.Tensor,
    fent2: torch.Tensor,
    xlamue: torch.Tensor,
    xlamud: torch.Tensor,
    cxlamet: torch.Tensor,
    cxlamdt: torch.Tensor,
    hwrf_samfdeep: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    KM = qeso.shape[1]
    level_values = torch.arange(1, KM + 1, dtype=qeso.dtype, device=qeso.device).reshape(1, KM)
    interior_layer = active[:, None] & (level_values >= 2.0) & (level_values < float(KM)) & (level_values > kbcon[:, None]) & (level_values < kmax[:, None])
    qeso_base = _gather_column(qeso, kbcon)
    tem = qeso / qeso_base[:, None]
    fent1 = torch.where(interior_layer, torch.minimum(tem * tem, torch.full_like(tem, 3.0)), fent1)
    fent2 = torch.where(interior_layer, torch.minimum(tem * tem * tem, torch.full_like(tem, 5.2)), fent2)
    xlamue = torch.where(interior_layer, xlamue * fent1 + cxlamet[:, None] * frh * fent2, xlamue)
    if not hwrf_samfdeep:
        xlamud = torch.where(interior_layer, xlamud + cxlamdt[:, None] * frh, xlamud)
    return fent1, fent2, xlamue, xlamud


def _mass_flux_recurrence_block(
    active: torch.Tensor,
    kb: torch.Tensor,
    kbcon: torch.Tensor,
    kmax: torch.Tensor,
    ktconn: torch.Tensor,
    zi: torch.Tensor,
    eta: torch.Tensor,
    xlamue: torch.Tensor,
    xlamud: torch.Tensor,
    heo: torch.Tensor,
    heso: torch.Tensor,
    hcko: torch.Tensor,
    ucko: torch.Tensor,
    vcko: torch.Tensor,
    uo: torch.Tensor,
    vo: torch.Tensor,
    dbyo: torch.Tensor,
    ctro: torch.Tensor,
    ecko: torch.Tensor,
    ercko: torch.Tensor,
    chem_c: torch.Tensor,
    chem_pw: torch.Tensor,
    pwavo_input: torch.Tensor,
    c0t: torch.Tensor,
    fscav: torch.Tensor,
    pgcon: torch.Tensor,
    hwrf_samfdeep: bool,
    do_aerosols: bool,
    itc: int,
    ntc: int,
    ntk: int,
    ntr: int,
    *,
    updraft_microblock: int = 0,
    updraft_compensated: bool = False,
) -> tuple[torch.Tensor, ...]:
    KM = zi.shape[1]
    size = recurrence_chunk_size()
    levels = torch.arange(1, KM + 1, dtype=kb.dtype, device=kb.device)
    pieces = []
    carry = eta[:, -1].contiguous()
    profile_rows = chunks.pack_profiles(zi, xlamue, xlamud, eta)
    for stop in range(KM - 1, 0, -size):
        start = max(0, stop - size)
        result = run_recurrence_chunk("g_ordered_chunks", "eta_lower", chunks.eta_lower,
            active, kb, kbcon, levels[start], profile_rows[start:stop + 1], carry)
        pieces.append(result)
        carry = result[:, 0]
    eta = chunks.join_levels(eta, pieces[::-1], 0)
    del profile_rows
    flg = active.clone()
    carry = eta[:, 0].contiguous()
    pieces = []
    profile_rows = chunks.pack_profiles(zi, xlamue, xlamud, eta)
    for start in range(1, KM - 1, size):
        stop = min(start + size, KM - 1)
        result, kmax, ktconn, flg = run_recurrence_chunk("g_ordered_chunks", "eta_upper", chunks.eta_upper,
            flg, kbcon, kmax, ktconn, levels[start], profile_rows[start - 1:stop], carry)
        pieces.append(result)
        carry = result[:, -1]
    eta = chunks.join_levels(eta, pieces, 1)
    del profile_rows

    hcko, ucko, vcko, ecko, ercko, pwavo = run_compile_island(
        "g_entrainment_formula", "seed_updraft", chunks.seed_updraft,
        active, kb, levels, heo, uo, vo, ctro, hcko, ucko, vcko, ecko, ercko, pwavo_input,
        hwrf_samfdeep)

    original = (hcko, ucko, vcko, dbyo)
    pieces = [[] for _ in original]
    carry = tuple(value[:, 0].contiguous() for value in (hcko, ucko, vcko))
    profile_rows = chunks.pack_profiles(zi, xlamue, xlamud, heo, heso, uo, vo, hcko, ucko, vcko, dbyo)
    # A live anchor must span the whole updraft, not restart at each ordered chunk.
    updraft_size = KM if updraft_microblock else size
    for start in range(1, KM - 1, updraft_size):
        stop = min(start + updraft_size, KM - 1)
        args = active, kb, kmax, levels[start], profile_rows[start - 1:stop], carry, pgcon
        if updraft_microblock:
            result = chunks.updraft(*args, microblock=updraft_microblock, compensated=updraft_compensated)
        else:
            result = run_recurrence_chunk("g_ordered_chunks", "updraft", chunks.updraft, *args)
        for values, value in zip(pieces, result, strict=True):
            values.append(value)
        carry = tuple(value[:, -1] for value in result[:3])
    hcko, ucko, vcko, dbyo = (
        chunks.join_levels(base, values, 1) for base, values in zip(original, pieces, strict=True))
    del profile_rows

    if not hwrf_samfdeep:
        tracer_count = itc - 3 if do_aerosols else ntr
        ecko, ercko = _advance_tracers(ecko, ercko, ctro, xlamue, zi, active, kb, kmax, 0, tracer_count, levels)
        if do_aerosols:
            # Chemistry slots are independent within this pass. The source's
            # subsequent TKE pass can overlap one slot and must remain after it.
            ecko, ercko, chem_c, chem_pw = _advance_aerosols(
                ecko, ercko, chem_c, chem_pw, ctro, xlamue, zi, eta,
                c0t, fscav, active, kb, kmax, itc - 3, ntc, levels)
            if ntk > 2:
                ecko, ercko = _advance_tracer(ecko, ercko, ctro, xlamue, zi, active, kb, kmax, ntk - 3, levels)

    return eta, kmax, ktconn, hcko, ucko, vcko, pwavo, ecko, ercko, dbyo, chem_c, chem_pw


def _replace_tracer_block(original, updated, start, stop):
    # Stack the existing slot axis; preserve its frozen structural contract.
    return torch.stack([original[:, :, index] if index < start or index >= stop
                        else updated[:, :, index - start]
                        for index in range(original.shape[2])], dim=2)


def _advance_tracer(ecko, ercko, ctro, xlamue, zi, active, kb, kmax, tracer_index, levels):
    return _advance_tracers(ecko, ercko, ctro, xlamue, zi, active, kb, kmax,
                            tracer_index, tracer_index + 1, levels)


def _advance_tracers(ecko, ercko, ctro, xlamue, zi, active, kb, kmax, begin, end, levels):
    if begin == end:
        return ecko, ercko
    km = zi.shape[1]
    selected = ecko[:, :, begin:end]
    original = ercko[:, :, begin:end]
    pieces, originals = [], []
    carry = selected[:, 0].contiguous()
    profile_rows = chunks.pack_profiles(zi, xlamue)
    tracer_rows = chunks.pack_profiles(ctro[:, :, begin:end], selected, original)
    for start in range(1, km - 1, recurrence_chunk_size()):
        stop = min(start + recurrence_chunk_size(), km - 1)
        result, result_original = run_recurrence_chunk("g_ordered_chunks", "tracer_up", chunks.tracer_up,
            active, kb, kmax, levels[start], profile_rows[start - 1:stop], tracer_rows[start - 1:stop], carry)
        pieces.append(result)
        originals.append(result_original)
        carry = result[:, -1]
    placed = chunks.join_levels(selected, pieces, 1)
    placed_original = chunks.join_levels(original, originals, 1)
    return (_replace_tracer_block(ecko, placed, begin, end),
            _replace_tracer_block(ercko, placed_original, begin, end))


def _advance_aerosols(ecko, ercko, chem_c, chem_pw, ctro, xlamue, zi, eta,
                       c0t, fscav, active, kb, kmax, begin, count, levels):
    km = zi.shape[1]
    end = begin + count
    selected = (ecko[:, :, begin:end], ercko[:, :, begin:end], chem_c[:, :, :count], chem_pw[:, :, :count])
    pieces = [[] for _ in selected]
    carry = selected[0][:, 0].contiguous()
    profile_rows = chunks.pack_profiles(zi, xlamue, eta, c0t)
    tracer_rows = chunks.pack_profiles(ctro[:, :, begin:end], *selected)
    for start in range(1, km - 1, recurrence_chunk_size()):
        stop = min(start + recurrence_chunk_size(), km - 1)
        result = run_recurrence_chunk("g_ordered_chunks", "aerosol_up", chunks.aerosol_up,
            active, kb, kmax, levels[start], profile_rows[start - 1:stop], tracer_rows[start - 1:stop], fscav[:count], carry)
        for values, value in zip(pieces, result, strict=True):
            values.append(value)
        carry = result[0][:, -1]
    results = tuple(chunks.join_levels(base, values, 1)
                    for base, values in zip(selected, pieces, strict=True))
    return (_replace_tracer_block(ecko, results[0], begin, end),
            _replace_tracer_block(ercko, results[1], begin, end),
            _replace_tracer_block(chem_c, results[2], 0, count),
            _replace_tracer_block(chem_pw, results[3], 0, count))


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}


def _recurrence_coordinator(category, name, fn, *args):
    if ordered_chunks_enabled("g_ordered_chunks"):
        return fn(*args)
    return run_compile_island(category, name, fn, *args)
