# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment A: entry initialization and workspace seed."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch.compile import run_compile_island
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real, tracer_layout


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 320-344 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    qtr = state.require("qtr")
    # The host binds q1 to the vapour slot of gq0 (a stride-ntrac view). Stored once as its own array for every
    # later reader: the islands of A and C each copied the view at entry (0.41 ms per 60 km step on the RTX PRO
    # 6000, round 8 wave 4). Values unchanged.
    q1 = state.require("q1").contiguous()
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
    ten_q = q1.new_zeros(()).expand(ncol, KM, tracer_layout(state).ntrac)
    # The updated state starts as copies of the state, with or without gradients (one form).
    new_t1, new_u1, new_v1, new_q1, new_qtr = (x.clone() for x in (t1, u1, v1, q1, qtr))
    return state.with_updates(
        {
            "q1": q1,
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
    # The tendencies are zero until segment U forms them: stride-0 zeros (no fills; compaction takes their rows as
    # views, round 8), a distinct tensor object each.
    ten_t = t1.new_zeros(()).expand(t1.shape)
    ten_u = u1.new_zeros(()).expand(u1.shape)
    ten_v = v1.new_zeros(()).expand(v1.shape)
    ten_q = q1.new_zeros(()).expand(ncol, KM, NTR)
    dqtr = qtr.new_zeros(()).expand(qtr.shape)
    scalar = torch.zeros((), dtype=q1.dtype, device=q1.device)
    return (
        ten_t,
        ten_u,
        ten_v,
        ten_q,
        dqtr,
        scalar.clone(),
        scalar.clone(),
        1.0 / grav,
        1.0 / delt,
        hvap / cp,
        hvap * hvap / (rv * cp),
        (cvap - cliq) / rv,
        hvap / rv - ((cvap - cliq) / rv) * t0c,
    )


