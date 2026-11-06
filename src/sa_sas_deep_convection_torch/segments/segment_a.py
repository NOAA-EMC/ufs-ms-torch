# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment A: entry initialization and workspace seed."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch.compile import run_compile_island
from sa_sas_deep_convection_torch.constants import KM, NN, NTR
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real, tracer_layout

CHECKPOINT_FIELDS = (
    "ten_t",
    "ten_u",
    "ten_v",
    "ten_q",
    "dqtr",
    "new_t1",
    "new_q1",
    "new_qtr",
    "errflg",
)


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 320-344 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    qtr = state.require("qtr")
    q1 = state.require("q1")
    t1 = state.require("t1")
    u1 = state.require("u1")
    v1 = state.require("v1")
    ncol = q1.shape[0]
    require_real("q1", q1, (ncol, KM))
    require_real("t1", t1, (ncol, KM))
    require_real("u1", u1, (ncol, KM))
    require_real("v1", v1, (ncol, KM))
    require_real("qtr", qtr, (ncol, KM, NN))

    (
        ten_t,
        ten_u,
        ten_v,
        ten_q,
        dqtr,
        new_t1,
        new_u1,
        new_v1,
        new_q1,
        new_qtr,
        errmsg_code,
        errflg,
        gravinv,
        invdelt,
        elocp,
        el2orc,
        fact1,
        fact2,
    ) = run_compile_island(
        "a_entry_init",
        "entry_initialization",
        _entry_initialization_block,
        qtr,
        q1,
        t1,
        u1,
        v1,
        state.require("grav"),
        state.require("delt"),
        state.require("hvap"),
        state.require("cp"),
        state.require("rv"),
        state.require("cvap"),
        state.require("cliq"),
        state.require("t0c"),
    )
    # samfdeepcnv's ten_q spans the host tracer array (ntrac), not the convective
    # transport array; only its vapour slot is ever set by the scheme.
    ten_q = torch.zeros((ncol, KM, tracer_layout(state).ntrac), dtype=q1.dtype, device=q1.device)
    return state.with_updates(
        {
            "ten_t": ten_t,
            "ten_u": ten_u,
            "ten_v": ten_v,
            "ten_q": ten_q,
            "dqtr": dqtr,
            "new_t1": new_t1,
            "new_u1": new_u1,
            "new_v1": new_v1,
            "new_q1": new_q1,
            "new_qtr": new_qtr,
            "errmsg_code": errmsg_code,
            "errflg": errflg,
            "gravinv": gravinv,
            "invdelt": invdelt,
            "elocp": elocp,
            "el2orc": el2orc,
            "fact1": fact1,
            "fact2": fact2,
        }
    )


def _entry_initialization_block(
    qtr: torch.Tensor,
    q1: torch.Tensor,
    t1: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    grav: torch.Tensor,
    delt: torch.Tensor,
    hvap: torch.Tensor,
    cp: torch.Tensor,
    rv: torch.Tensor,
    cvap: torch.Tensor,
    cliq: torch.Tensor,
    t0c: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    KM = q1.shape[1]
    NTR = qtr.shape[2] - 2
    ncol = q1.shape[0]
    ten_t = torch.zeros_like(t1)
    ten_u = torch.zeros_like(u1)
    ten_v = torch.zeros_like(v1)
    ten_q = torch.zeros((ncol, KM, NTR), dtype=q1.dtype, device=q1.device)
    dqtr = torch.zeros_like(qtr)
    scalar = torch.zeros((), dtype=q1.dtype, device=q1.device)
    return (
        ten_t,
        ten_u,
        ten_v,
        ten_q,
        dqtr,
        t1.clone(),
        u1.clone(),
        v1.clone(),
        q1.clone(),
        qtr.clone(),
        scalar.clone(),
        scalar.clone(),
        1.0 / grav,
        1.0 / delt,
        hvap / cp,
        hvap * hvap / (rv * cp),
        (cvap - cliq) / rv,
        hvap / rv - ((cvap - cliq) / rv) * t0c,
    )


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
