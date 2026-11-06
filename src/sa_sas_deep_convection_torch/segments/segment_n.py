# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment N: TVD subsidence and tracer limiter increments."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch.compile import run_compile_island, run_recurrence_chunk, ordered_chunks_enabled
from sa_sas_deep_convection_torch.constants import KM, NTR
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real

CHECKPOINT_FIELDS = ("q_diff", "e_diff", "flxtvd", "new_t1", "new_q1", "new_qtr", "asqecflg")


def _limiter_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    valid = torch.abs(denominator) > 1.0e-22
    safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
    rrkp = torch.where(valid, numerator / safe_denominator, torch.zeros_like(denominator))
    return (rrkp + torch.abs(rrkp)) / (1.0 + torch.abs(rrkp))


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 2405-2567 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    cnvflg = state.require("cnvflg")
    ktcon = state.require("ktcon")
    kb = state.require("kb")
    jmin = state.require("jmin")
    eta = state.require("eta")
    etad = state.require("etad")
    edto = state.require("edto")
    del_ = state.require("del")
    q1 = state.require("q1")
    qo = state.require("qo")
    ctr = state.require("ctr")
    ctro = state.require("ctro")
    dellaq = state.require("dellaq").clone()
    dellae = state.require("dellae").clone()
    ncol = cnvflg.shape[0]
    for name, tensor in {"cnvflg": cnvflg, "ktcon": ktcon, "kb": kb, "jmin": jmin, "edto": edto}.items():
        require_real(name, tensor, (ncol,))
    for name, tensor in {"eta": eta, "etad": etad, "del": del_, "q1": q1, "qo": qo, "dellaq": dellaq}.items():
        require_real(name, tensor, (ncol, KM))
    for name, tensor in {"ctr": ctr, "ctro": ctro, "dellae": dellae}.items():
        require_real(name, tensor, (ncol, KM, NTR))

    q_diff, e_diff, flxtvd, dellaq, dellae = _limiter_coordinator(
        "n_tvd_limiter",
        "tvd_limiter",
        _tvd_limiter_block,
        cnvflg,
        ktcon,
        kb,
        jmin,
        eta,
        etad,
        edto,
        del_,
        q1,
        qo,
        ctr,
        ctro,
        dellaq,
        dellae,
        state.require("grav"),
        bool(state.config.get("hwrf_samfdeep", False)),
    )

    return state.with_updates(
        {
            "q_diff": q_diff,
            "e_diff": e_diff,
            "flxtvd": flxtvd,
            "dellaq": dellaq,
            "dellae": dellae,
            "asqecflg": torch.zeros_like(cnvflg),
        }
    )


def _tvd_limiter_block(
    cnvflg: torch.Tensor,
    ktcon: torch.Tensor,
    kb: torch.Tensor,
    jmin: torch.Tensor,
    eta: torch.Tensor,
    etad: torch.Tensor,
    edto: torch.Tensor,
    del_: torch.Tensor,
    q1: torch.Tensor,
    qo: torch.Tensor,
    ctr: torch.Tensor,
    ctro: torch.Tensor,
    dellaq: torch.Tensor,
    dellae: torch.Tensor,
    grav: torch.Tensor,
    hwrf_samfdeep: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_diff, flxtvd, dellaq = run_recurrence_chunk("n_tensor_backward", "tvd_moisture", _tvd_moisture_block,
        cnvflg, ktcon, kb, jmin, eta, etad, edto, del_, q1, qo, dellaq, grav)
    if hwrf_samfdeep:
        e_diff = torch.zeros_like(ctr)
    else:
        e_diff, flxtvd, dellae = run_recurrence_chunk("n_tensor_backward", "tvd_tracer", _tvd_tracer_block,
            cnvflg, ktcon, kb, jmin, eta, etad, edto, del_, ctr, ctro, dellae, grav)
    return q_diff, e_diff, flxtvd, dellaq, dellae


def _tvd_geometry(cnvflg, ktcon, kb, jmin, eta, etad, edto):
    km = eta.shape[1]
    active = cnvflg != 0.0
    levels = torch.arange(1, km, dtype=cnvflg.dtype, device=cnvflg.device).reshape(1, km - 1)
    level_layer_le = active[:, None] & (levels <= ktcon[:, None])
    level_layer_lt = active[:, None] & (levels < ktcon[:, None])
    tem = torch.zeros((cnvflg.shape[0], km - 1), dtype=cnvflg.dtype, device=cnvflg.device)
    tem = torch.where(levels >= kb[:, None], eta[:, :km - 1], tem)
    tem = torch.where(levels <= jmin[:, None], tem - edto[:, None] * etad[:, :km - 1], tem)
    top_double = level_layer_lt & (levels == jmin[:, None])
    base_double = level_layer_lt & (levels == kb[:, None])
    return active, level_layer_le, level_layer_lt, top_double, base_double, tem


def _tvd_moisture_block(cnvflg, ktcon, kb, jmin, eta, etad, edto, del_, q1, qo, dellaq, grav):
    KM = q1.shape[1]
    ncol = cnvflg.shape[0]
    dtype, device = cnvflg.dtype, cnvflg.device
    active, level_layer_le, level_layer_lt, top_double, base_double, tem = _tvd_geometry(cnvflg, ktcon, kb, jmin, eta, etad, edto)
    q_diff = torch.zeros_like(q1)
    q_diff[:, 1:] = torch.where(level_layer_le, q1[:, : KM - 1] - q1[:, 1:], q_diff[:, 1:])
    q_boundary_positive = torch.maximum(torch.zeros_like(cnvflg), 2.0 * q1[:, 0] - q1[:, 1]) - q1[:, 0]
    q_boundary_negative = torch.minimum(torch.zeros_like(cnvflg), 2.0 * q1[:, 0] - q1[:, 1]) - q1[:, 0]
    q_diff[:, 0] = torch.where(active, torch.where(q1[:, 0] >= 0.0, q_boundary_positive, q_boundary_negative), q_diff[:, 0])

    flxtvd = torch.zeros_like(q1)
    q_diff_next = torch.zeros((ncol, KM - 1), dtype=dtype, device=device)
    q_diff_next[:, : KM - 2] = q_diff[:, 2:]
    ph_pos = _limiter_ratio(q_diff_next, q_diff[:, 1:])
    tem1_pos = q1[:, 1:] + ph_pos * (qo[:, : KM - 1] - q1[:, 1:])
    ph_neg = _limiter_ratio(q_diff[:, : KM - 1], q_diff[:, 1:])
    tem1_neg = q1[:, : KM - 1] + ph_neg * (qo[:, : KM - 1] - q1[:, : KM - 1])
    tem1 = torch.where(tem > 0.0, tem1_pos, torch.where(tem < 0.0, tem1_neg, qo[:, : KM - 1]))
    flux = torch.where(tem > 0.0, tem * tem1_pos, torch.where(tem < 0.0, tem * tem1_neg, torch.zeros_like(tem)))
    flxtvd[:, 1:] = torch.where(level_layer_lt, flux, flxtvd[:, 1:])
    dellaq[:, 1:] = torch.where(
        top_double,
        dellaq[:, 1:] - edto[:, None] * etad[:, : KM - 1] * tem1 * grav / (1000.0 * del_[:, 1:]),
        dellaq[:, 1:],
    )
    dellaq[:, : KM - 1] = torch.where(
        base_double,
        dellaq[:, : KM - 1] - eta[:, : KM - 1] * tem1 * grav / (1000.0 * del_[:, : KM - 1]),
        dellaq[:, : KM - 1],
    )
    delta_q = (flxtvd[:, 1:] - flxtvd[:, : KM - 1]) * grav / (1000.0 * del_[:, : KM - 1])
    dellaq[:, : KM - 1] = torch.where(level_layer_le, dellaq[:, : KM - 1] + delta_q, dellaq[:, : KM - 1])

    return q_diff, flxtvd, dellaq


def _tvd_tracer_block(cnvflg, ktcon, kb, jmin, eta, etad, edto, del_, ctr, ctro, dellae, grav):
    KM, NTR = ctr.shape[1:]
    ncol = cnvflg.shape[0]
    dtype, device = cnvflg.dtype, cnvflg.device
    active, level_layer_le, level_layer_lt, top_double, base_double, tem = _tvd_geometry(cnvflg, ktcon, kb, jmin, eta, etad, edto)
    e_diff = torch.zeros_like(ctr)
    e_diff[:, 1:, :] = torch.where(
        level_layer_le[:, :, None],
        ctr[:, : KM - 1, :] - ctr[:, 1:, :],
        e_diff[:, 1:, :],
    )
    e_boundary_positive = torch.maximum(torch.zeros((ncol, NTR), dtype=dtype, device=device), 2.0 * ctr[:, 0, :] - ctr[:, 1, :]) - ctr[:, 0, :]
    e_boundary_negative = torch.minimum(torch.zeros((ncol, NTR), dtype=dtype, device=device), 2.0 * ctr[:, 0, :] - ctr[:, 1, :]) - ctr[:, 0, :]
    e_diff[:, 0, :] = torch.where(active[:, None], torch.where(ctr[:, 0, :] >= 0.0, e_boundary_positive, e_boundary_negative), e_diff[:, 0, :])

    e_diff_next = torch.zeros((ncol, KM - 1, NTR), dtype=dtype, device=device)
    e_diff_next[:, : KM - 2, :] = e_diff[:, 2:, :]
    ph_pos_e = _limiter_ratio(e_diff_next, e_diff[:, 1:, :])
    tem1_pos_e = ctr[:, 1:, :] + ph_pos_e * (ctro[:, : KM - 1, :] - ctr[:, 1:, :])
    ph_neg_e = _limiter_ratio(e_diff[:, : KM - 1, :], e_diff[:, 1:, :])
    tem1_neg_e = ctr[:, : KM - 1, :] + ph_neg_e * (ctro[:, : KM - 1, :] - ctr[:, : KM - 1, :])
    tem_e = tem[:, :, None]
    tem1_e = torch.where(tem_e > 0.0, tem1_pos_e, torch.where(tem_e < 0.0, tem1_neg_e, ctro[:, : KM - 1, :]))
    flux_e = torch.where(tem_e > 0.0, tem_e * tem1_pos_e, torch.where(tem_e < 0.0, tem_e * tem1_neg_e, torch.zeros_like(tem1_e)))
    flxtvd_e = torch.zeros_like(ctr)
    flxtvd_e[:, 1:, :] = torch.where(level_layer_lt[:, :, None], flux_e, flxtvd_e[:, 1:, :])
    dellae[:, 1:, :] = torch.where(
        top_double[:, :, None],
        dellae[:, 1:, :] - edto[:, None, None] * etad[:, : KM - 1, None] * tem1_e * grav / (1000.0 * del_[:, 1:, None]),
        dellae[:, 1:, :],
    )
    dellae[:, : KM - 1, :] = torch.where(
        base_double[:, :, None],
        dellae[:, : KM - 1, :] - eta[:, : KM - 1, None] * tem1_e * grav / (1000.0 * del_[:, : KM - 1, None]),
        dellae[:, : KM - 1, :],
    )
    delta_e = (flxtvd_e[:, 1:, :] - flxtvd_e[:, : KM - 1, :]) * grav / (1000.0 * del_[:, : KM - 1, None])
    dellae[:, : KM - 1, :] = torch.where(level_layer_le[:, :, None], dellae[:, : KM - 1, :] + delta_e, dellae[:, : KM - 1, :])
    return e_diff, flxtvd_e[:, :, NTR - 1], dellae


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}


def _limiter_coordinator(category, name, fn, *args):
    if ordered_chunks_enabled("n_tensor_backward"):
        return fn(*args)
    return run_compile_island(category, name, fn, *args)
