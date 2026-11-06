# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment J: overshoot and updraft velocity."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch import recurrence_chunks as chunks
from sa_sas_deep_convection_torch.compile import masked_sum_chunk_size, run_compile_island, run_recurrence_chunk, recurrence_chunk_size, ordered_chunks_enabled
from sa_sas_deep_convection_torch.constants import KM
from sa_sas_deep_convection_torch.ops import sqrt_zero_grad
from sa_sas_deep_convection_torch.segments.segment_d import _gather_column
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real

CHECKPOINT_FIELDS = ("wu2", "buo", "drag", "wush", "wc", "omegaout", "omega_u", "cnvflg")


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 1712-1899 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    cnvflg = state.require("cnvflg")
    kbcon = state.require("kbcon")
    kbcon1 = state.require("kbcon1")
    ktcon = state.require("ktcon")
    kmax = state.require("kmax")
    zi = state.require("zi")
    zo = state.require("zo")
    qeso = state.require("qeso")
    qo = state.require("qo")
    to = state.require("to")
    dbyo = state.require("dbyo")
    qcko = state.require("qcko")
    qrcko = state.require("qrcko")
    xlamue = state.require("xlamue")
    eta = state.require("eta")
    del_ = state.require("del")
    c0t = state.require("c0t")
    dellal = state.require("dellal")
    pwo = state.require("pwo")
    pwavo = state.require("pwavo")
    cnvwt = state.require("cnvwt")
    zdqca = state.require("zdqca")
    wu2 = state.require("wu2")
    buo = state.require("buo")
    drag = state.require("drag")
    wush = state.require("wush")
    omegaout = state.require("omegaout")
    omega_u = state.require("omega_u")
    po = state.require("po")
    dot = state.require("dot")
    ncol = cnvflg.shape[0]
    for name, tensor in {
        "cnvflg": cnvflg,
        "kbcon": kbcon,
        "kbcon1": kbcon1,
        "ktcon": ktcon,
        "kmax": kmax,
    }.items():
        require_real(name, tensor, (ncol,))
    for name, tensor in {
        "zi": zi,
        "zo": zo,
        "qeso": qeso,
        "qo": qo,
        "to": to,
        "dbyo": dbyo,
        "qcko": qcko,
        "qrcko": qrcko,
        "xlamue": xlamue,
        "eta": eta,
        "del": del_,
        "c0t": c0t,
        "dellal": dellal,
        "pwo": pwo,
        "cnvwt": cnvwt,
        "zdqca": zdqca,
        "wu2": wu2,
        "buo": buo,
        "drag": drag,
        "wush": wush,
        "omegaout": omegaout,
        "omega_u": omega_u,
        "po": po,
        "dot": dot,
    }.items():
        require_real(name, tensor, (ncol, KM))

    (
        cnvflg,
        aa2,
        ktcon1,
        qcko,
        qrcko,
        dellal,
        pwo,
        pwavo,
        cnvwt,
        zdqca,
        wu2,
        omegaout,
        omega_u,
        wc,
        sumx,
    ) = _recurrence_coordinator(
        "j_diagnostics",
        "overshoot_updraft",
        _overshoot_updraft_block,
        cnvflg,
        kbcon,
        kbcon1,
        ktcon,
        kmax,
        zi,
        zo,
        qeso,
        qo,
        to,
        dbyo,
        qcko,
        qrcko,
        xlamue,
        eta,
        del_,
        c0t,
        dellal,
        pwo,
        pwavo,
        cnvwt,
        zdqca,
        wu2,
        buo,
        drag,
        wush,
        omegaout,
        omega_u,
        po,
        dot,
        state.require("aa1"),
        state.require("aa2"),
        state.require("el2orc"),
        state.require("fv"),
        state.require("cp"),
        state.require("hvap"),
        state.require("grav"),
        state.require("c1"),
        state.require("rd"),
        state.require("delt"),
        state.require("omegain"),
        int(state.config.get("ncloud", 0)) > 0,
        bool(state.config.get("hwrf_samfdeep", False)),
        bool(state.config.get("progomega", False)),
        bool(state.config.get("first_time_step", False)),
        bool(state.config.get("restart", False)),
    )

    return state.with_updates(
        {
            "aa2": aa2,
            "ktcon1": ktcon1,
            "qcko": qcko,
            "qrcko": qrcko,
            "dellal": dellal,
            "pwo": pwo,
            "pwavo": pwavo,
            "cnvwt": cnvwt,
            "zdqca": zdqca,
            "wu2": wu2,
            "buo": buo,
            "drag": drag,
            "wush": wush,
            "omegaout": omegaout,
            "omega_u": omega_u,
            "wc": wc,
            "sumx": sumx,
            "cnvflg": cnvflg,
        }
    )


def _overshoot_updraft_block(
    cnvflg: torch.Tensor,
    kbcon: torch.Tensor,
    kbcon1: torch.Tensor,
    ktcon: torch.Tensor,
    kmax: torch.Tensor,
    zi: torch.Tensor,
    zo: torch.Tensor,
    qeso: torch.Tensor,
    qo: torch.Tensor,
    to: torch.Tensor,
    dbyo: torch.Tensor,
    qcko: torch.Tensor,
    qrcko: torch.Tensor,
    xlamue: torch.Tensor,
    eta: torch.Tensor,
    del_: torch.Tensor,
    c0t: torch.Tensor,
    dellal: torch.Tensor,
    pwo: torch.Tensor,
    pwavo: torch.Tensor,
    cnvwt: torch.Tensor,
    zdqca: torch.Tensor,
    wu2: torch.Tensor,
    buo: torch.Tensor,
    drag: torch.Tensor,
    wush: torch.Tensor,
    omegaout: torch.Tensor,
    omega_u: torch.Tensor,
    po: torch.Tensor,
    dot: torch.Tensor,
    aa1: torch.Tensor,
    aa2_base: torch.Tensor,
    el2orc: torch.Tensor,
    fv: torch.Tensor,
    cp: torch.Tensor,
    hvap: torch.Tensor,
    grav: torch.Tensor,
    c1: torch.Tensor,
    rd: torch.Tensor,
    delt: torch.Tensor,
    omegain: torch.Tensor,
    ncloud_positive: bool,
    hwrf_samfdeep: bool,
    progomega: bool,
    first_time_step: bool,
    restart: bool,
) -> tuple[torch.Tensor, ...]:
    KM = zo.shape[1]
    active = cnvflg != 0.0
    aa2 = torch.where(active, 0.1 * aa1, aa2_base)
    flg = active
    ktcon1 = ktcon
    cloud_depth_half = 0.5 * (_gather_column(zi, ktcon) - _gather_column(zi, kbcon))
    cloud_top_height = _gather_column(zi, ktcon)
    level_indices = torch.arange(1, KM + 1, dtype=cnvflg.dtype, device=cnvflg.device)
    size = recurrence_chunk_size()
    top_rows = chunks.pack_profiles(zo, qeso, to, dbyo, zi)
    for start in range(1, KM - 1, size):
        stop = min(start + size, KM - 1)
        aa2, ktcon1, flg = run_recurrence_chunk("j_ordered_chunks", "overshoot_top", chunks.overshoot_top,
            flg, ktcon, kmax, level_indices[start],
            top_rows[start - 1:stop + 1],
            aa2, ktcon1, cloud_top_height, cloud_depth_half, el2orc, fv, cp, hvap, grav)
    del top_rows

    original = (qcko, qrcko, dellal, pwo, cnvwt, zdqca)
    pieces = [[] for _ in original]
    inputs = chunks.pack_profiles(zi, qeso, to, dbyo, xlamue, qo, eta, del_, c0t, qcko, qrcko, dellal, pwo, cnvwt, zdqca)
    qcko_previous = qcko[:, 0].contiguous()
    for start in range(1, KM - 1, size):
        stop = min(start + size, KM - 1)
        result, pwavo = run_recurrence_chunk("j_ordered_chunks", "overshoot_moisture", chunks.overshoot_moisture,
            active, ktcon, ktcon1, level_indices[start],
            inputs[start - 1:stop + 1],
            qcko_previous, pwavo, el2orc, hvap, c1, grav, ncloud_positive)
        for values, value in zip(pieces, result, strict=True):
            values.append(value)
        qcko_previous = result[0][:, -1]
    qcko, qrcko, dellal, pwo, cnvwt, zdqca = (
        chunks.join_levels(base, values, 1) for base, values in zip(original, pieces, strict=True))
    del inputs

    if hwrf_samfdeep:
        level_values = torch.arange(1, KM + 1, dtype=cnvflg.dtype, device=cnvflg.device).reshape(1, KM)
        layer = active[:, None] & (level_values == kbcon1[:, None])
        rho_base = torch.where(layer, po, torch.ones_like(po)) / (rd * torch.where(layer, to, torch.ones_like(to)))
        wucb = -0.01 * dot / (rho_base * grav)
        wu2 = torch.where(layer, torch.where(wucb > 0.0, wucb * wucb, torch.zeros_like(wucb)), wu2)

    if progomega:
        # This branch still writes slices; the functional recurrence paths borrow inputs.
        wu2, omegaout, omega_u = wu2.clone(), omegaout.clone(), omega_u.clone()
        omega = omegain
        omega = torch.where(active[:, None] & (omega < 1.0e-5), torch.zeros_like(omega), omega)
        if first_time_step and not restart:
            omega = torch.where(active[:, None], torch.full_like(omega, -1.2), omega)
        mid_levels = torch.arange(2, KM, dtype=cnvflg.dtype, device=cnvflg.device).reshape(1, KM - 2)
        mid = slice(1, KM - 1)
        layer = active[:, None] & (mid_levels >= kbcon1[:, None]) & (mid_levels < ktcon[:, None])
        dp = 1000.0 * del_[:, mid]
        dz = zi[:, 2:KM] - zi[:, 1 : KM - 1]
        dp = torch.where(layer, dp, torch.ones_like(dp))
        dz = torch.where(layer, dz, torch.ones_like(dz))
        dp_dz = dp / dz
        term_a = delt * (1.5 * drag[:, mid] * dp_dz)
        term_b = 1.0 - delt * 1.2 * wush[:, mid] * dp_dz
        omega_level = omega[:, mid]
        term_c = omega_level - delt * 0.6 * buo[:, mid] * dp_dz - delt * omega_level * (omega[:, : KM - 2] - omega_level) / dp
        discr = term_b * term_b - 4.0 * term_a * term_c
        quadratic = (-term_b - sqrt_zero_grad(torch.clamp_min(discr, 0.0))) / (2.0 * torch.where(layer & (discr >= 0.0), term_a, torch.ones_like(term_a)))
        omegaout_val = torch.where(discr >= 0.0, quadratic, omega_level)
        omegaout_val = torch.maximum(torch.minimum(omegaout_val, torch.full_like(omegaout_val, -1.2)), torch.full_like(omegaout_val, -80.0))
        omegaout[:, mid] = torch.where(layer, omegaout_val, omegaout[:, mid])
        rho = torch.where(layer, po[:, mid], torch.ones_like(po[:, mid])) * 100.0 / (rd * torch.where(layer, to[:, mid], torch.ones_like(to[:, mid])))
        omega_layer = torch.maximum(omegaout[:, mid], torch.full_like(omegaout_val, -80.0))
        omega_u[:, mid] = torch.where(layer, omega_layer, omega_u[:, mid])
        velocity = (-omega_layer) / (rho * grav)
        wu2[:, mid] = torch.where(layer, torch.maximum(velocity * velocity, torch.zeros_like(velocity)), wu2[:, mid])
    else:
        pieces = []
        wu2_previous = wu2[:, 0].contiguous()
        velocity_rows = chunks.pack_profiles(zi, drag, buo, wush, wu2)
        for start in range(1, KM - 1, size):
            stop = min(start + size, KM - 1)
            result = run_recurrence_chunk("j_ordered_chunks", "updraft_velocity", chunks.updraft_velocity,
                active, kbcon1, ktcon, level_indices[start],
                velocity_rows[start - 1:stop + 1], wu2_previous)
            pieces.append(result)
            wu2_previous = result[:, -1]
        wu2 = chunks.join_levels(wu2, pieces, 1)
        del velocity_rows
        level_values = level_indices.reshape(1, KM)
        layer = active[:, None] & (level_values > kbcon1[:, None]) & (level_values < ktcon[:, None])
        rho = po * 100.0 / (rd * to)
        omega_val = -sqrt_zero_grad(wu2) * rho * grav
        omega_u = torch.where(layer, torch.maximum(omega_val, torch.full_like(omega_val, -80.0)), omega_u)

    average_rows = chunks.pack_profiles(zi, wu2)
    wc, sumx = torch.zeros_like(cnvflg), torch.zeros_like(cnvflg)
    size = masked_sum_chunk_size()
    for start in range(1, KM - 1, size):
        stop = min(start + size, KM - 1)
        wc, sumx = run_recurrence_chunk("j_ordered_chunks", "velocity_average", chunks.velocity_average,
            active, kbcon1, ktcon, level_indices[start], average_rows[start - 1:stop], wc, sumx)
    cnvflg = torch.where(active & (sumx == 0.0), torch.zeros_like(cnvflg), cnvflg)
    wc = torch.where((cnvflg != 0.0) & (sumx != 0.0), wc / torch.where((cnvflg != 0.0) & (sumx != 0.0), sumx, torch.ones_like(sumx)), wc)
    cnvflg = torch.where((cnvflg != 0.0) & (wc < 1.0e-4), torch.zeros_like(cnvflg), cnvflg)
    return cnvflg, aa2, ktcon1, qcko, qrcko, dellal, pwo, pwavo, cnvwt, zdqca, wu2, omegaout, omega_u, wc, sumx


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}


def _recurrence_coordinator(category, name, fn, *args):
    if ordered_chunks_enabled("j_ordered_chunks"):
        return fn(*args)
    return run_compile_island(category, name, fn, *args)
