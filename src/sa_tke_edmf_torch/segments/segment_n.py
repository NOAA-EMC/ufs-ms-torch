# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment N: TVD flux-limiter preparation for TKE/TTE and tracers."""

from __future__ import annotations

import torch

from sa_tke_edmf_torch.autodiff import safe_signed_denominator
from sa_tke_edmf_torch.compile import run_compile_island
from sa_tke_edmf_torch.constants import IM, KM, NTRAC
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64

CHECKPOINT_FIELDS = ("e_half", "e_diff", "q_half", "q_diff")


def _tvd_limiter_block(
    q1: torch.Tensor,
    te: torch.Tensor,
    xmf: torch.Tensor,
    xmfd: torch.Tensor,
    kpbl: torch.Tensor,
    krad: torch.Tensor,
    mrad: torch.Tensor,
    pcnvflg: torch.Tensor,
    scuflg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    KM = q1.shape[1]
    NTRAC = q1.shape[2]
    dtype = te.dtype
    device = te.device
    ncol = q1.shape[0]
    kps = KM // 2
    ntrac1 = NTRAC - 1
    qh = torch.zeros((ncol, KM - 1, ntrac1), dtype=dtype, device=device)
    q_half = torch.zeros((ncol, KM - 1, ntrac1), dtype=dtype, device=device)
    q_diff = torch.zeros((ncol, KM, ntrac1), dtype=dtype, device=device)
    e_half = torch.zeros((ncol, KM - 1), dtype=dtype, device=device)
    e_diff = torch.zeros((ncol, KM), dtype=dtype, device=device)
    eps = torch.full((), 1.000000031374395e-22, dtype=dtype, device=device)

    qh[:, :kps, :] = 0.5 * (q1[:, :kps, :ntrac1] + q1[:, 1 : kps + 1, :ntrac1])
    q_diff[:, 1 : kps + 1, :] = q1[:, :kps, :ntrac1] - q1[:, 1 : kps + 1, :ntrac1]
    q_edge = 2.0 * q1[:, 0, :ntrac1] - q1[:, 1, :ntrac1]
    q_diff[:, 0, :] = torch.where(
        q1[:, 0, :ntrac1] >= 0.0,
        torch.maximum(torch.zeros_like(q_edge), q_edge) - q1[:, 0, :ntrac1],
        torch.minimum(torch.zeros_like(q_edge), q_edge) - q1[:, 0, :ntrac1],
    )

    kmx = torch.maximum(kpbl, krad)
    levels = torch.arange(1, kps + 1, dtype=torch.int64, device=device)
    level_mask = levels[None, :]
    q_active = ((pcnvflg | scuflg)[:, None, None]) & (level_mask[:, :, None] < kmx[:, None, None])
    tem = torch.zeros((ncol, kps), dtype=dtype, device=device)
    tem = torch.where(pcnvflg[:, None] & (level_mask < kpbl[:, None]), xmf[:, :kps], tem)
    tem = torch.where(scuflg[:, None] & (level_mask >= mrad[:, None]) & (level_mask < krad[:, None]), tem - xmfd[:, :kps], tem)
    tem_q = tem[:, :, None]

    q_center = q_diff[:, 1 : kps + 1, :]
    q_active_ratio = torch.abs(q_center) > eps
    q_denominator = safe_signed_denominator(q_center)
    # Float64 ratios rounded once (no-op in float64): in float32 the division backward (a/b)/b
    # overflows where the face is not selected (incoming gradient 0) and 0 * inf is NaN.
    q_rrkp_pos = torch.where(q_active_ratio, (q_diff[:, 2 : kps + 2, :].double() / q_denominator.double()).to(dtype), tem_q)
    q_rrkp_pos = torch.where(q_active_ratio, q_rrkp_pos, torch.zeros_like(q_rrkp_pos))
    q_phkp_pos = (q_rrkp_pos + torch.abs(q_rrkp_pos)) / (1.0 + torch.abs(q_rrkp_pos))
    q_pos_value = q1[:, 1 : kps + 1, :ntrac1] + q_phkp_pos * (qh[:, :kps, :] - q1[:, 1 : kps + 1, :ntrac1])

    q_rrkp_neg = torch.where(q_active_ratio, (q_diff[:, :kps, :].double() / q_denominator.double()).to(dtype), tem_q)
    q_rrkp_neg = torch.where(q_active_ratio, q_rrkp_neg, torch.zeros_like(q_rrkp_neg))
    q_phkp_neg = (q_rrkp_neg + torch.abs(q_rrkp_neg)) / (1.0 + torch.abs(q_rrkp_neg))
    q_neg_value = q1[:, :kps, :ntrac1] + q_phkp_neg * (qh[:, :kps, :] - q1[:, :kps, :ntrac1])
    q_half[:, :kps, :] = torch.where(
        q_active & (tem_q > 0.0),
        q_pos_value,
        torch.where(q_active & (tem_q < 0.0), q_neg_value, qh[:, :kps, :]),
    )

    tei = torch.zeros((ncol, KM - 1), dtype=dtype, device=device)
    tei[:, :kps] = 0.5 * (te[:, :kps] + te[:, 1 : kps + 1])
    e_diff[:, 1 : kps + 1] = te[:, :kps] - te[:, 1 : kps + 1]
    e_edge = 2.0 * te[:, 0] - te[:, 1]
    e_diff[:, 0] = torch.where(
        te[:, 0] >= 0.0,
        torch.maximum(torch.zeros_like(e_edge), e_edge) - te[:, 0],
        torch.minimum(torch.zeros_like(e_edge), e_edge) - te[:, 0],
    )
    e_active = ((pcnvflg | scuflg)[:, None]) & (level_mask < kmx[:, None])
    e_center = e_diff[:, 1 : kps + 1]
    e_active_ratio = torch.abs(e_center) > eps
    e_denominator = safe_signed_denominator(e_center)
    e_rrkp_pos = torch.where(e_active_ratio, (e_diff[:, 2 : kps + 2].double() / e_denominator.double()).to(dtype),
                             torch.zeros_like(e_center))
    e_phkp_pos = (e_rrkp_pos + torch.abs(e_rrkp_pos)) / (1.0 + torch.abs(e_rrkp_pos))
    e_pos_value = te[:, 1 : kps + 1] + e_phkp_pos * (tei[:, :kps] - te[:, 1 : kps + 1])

    e_rrkp_neg = torch.where(e_active_ratio, (e_diff[:, :kps].double() / e_denominator.double()).to(dtype),
                             torch.zeros_like(e_center))
    e_phkp_neg = (e_rrkp_neg + torch.abs(e_rrkp_neg)) / (1.0 + torch.abs(e_rrkp_neg))
    e_neg_value = te[:, :kps] + e_phkp_neg * (tei[:, :kps] - te[:, :kps])
    e_half[:, :kps] = torch.where(
        e_active & (tem > 0.0),
        e_pos_value,
        torch.where(e_active & (tem < 0.0), e_neg_value, tei[:, :kps]),
    )
    return e_half, e_diff, q_half, q_diff


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    KM = state.require("q1").shape[1]
    NTRAC = state.require("q1").shape[2]
    q1 = state.require("q1")
    te = state.require("te")
    xmf = state.require("xmf")
    xmfd = state.require("xmfd")
    kpbl = state.require("kpbl").to(torch.int64)
    krad = state.require("krad").to(torch.int64)
    mrad = state.require("mrad").to(torch.int64)
    pcnvflg = state.require("pcnvflg") > 0.5
    scuflg = state.require("scuflg") > 0.5
    IM = q1.shape[0]
    require_float64("q1", q1, (IM, KM, NTRAC))
    require_float64("te", te, (IM, KM))
    require_float64("xmf", xmf, (IM, KM))
    require_float64("xmfd", xmfd, (IM, KM))

    e_half, e_diff, q_half, q_diff = run_compile_island(
        "n_tvd_limiter",
        "limiters",
        _tvd_limiter_block,
        q1,
        te,
        xmf,
        xmfd,
        kpbl,
        krad,
        mrad,
        pcnvflg,
        scuflg,
    )

    return state.with_updates(
        {
            "e_half": e_half,
            "e_diff": e_diff,
            "q_half": q_half,
            "q_diff": q_diff,
        }
    )


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
