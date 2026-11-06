# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment O: TKE/TTE tridiagonal solve and recovered NTKE tendency."""

from __future__ import annotations

import torch
from ufs_physics_common.quotient import quotient

from sa_tke_edmf_torch.compile import run_compile_island
from sa_tke_edmf_torch.constants import GRAV, IM, KM, NDTEND, NTKE, NTRAC
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64
from sa_tke_edmf_torch.diagnostics import diagnostic_slots, add_diagnostic_

CHECKPOINT_FIELDS = ("tkeh", "te", "tke", "ten_t")


def _tridit_tke(
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Match tridi.f:tridit for one right-hand side.

    The sweep reads level-major rows (one contiguous row per level) and returns ``[ncol, KM]``.
    """

    KM = diagonal.shape[1]
    lower, diagonal, upper, rhs = (x.t().contiguous() for x in (lower, diagonal, upper, rhs))
    fk = 1.0 / diagonal[0]
    upper_cols = [fk * upper[0]]
    rhs_cols = [fk * rhs[0]]

    for k in range(1, KM - 1):
        factor = 1.0 / (diagonal[k] - lower[k - 1] * upper_cols[-1])
        upper_cols.append(factor * upper[k])
        rhs_cols.append(factor * (rhs[k] - lower[k - 1] * rhs_cols[-1]))

    fk = 1.0 / (diagonal[KM - 1] - lower[KM - 2] * upper_cols[-1])
    rhs_cols.append(fk * (rhs[KM - 1] - lower[KM - 2] * rhs_cols[-1]))
    solution_cols = [rhs_cols[-1]]
    for k in range(KM - 2, -1, -1):
        solution_cols.append(rhs_cols[k] - upper_cols[k] * solution_cols[-1])

    return torch.stack(list(reversed(solution_cols)), dim=0).t().contiguous()


def _borrow_negative_energy(
    f1: torch.Tensor,
    delp: torch.Tensor,
    *,
    active: torch.Tensor | None = None,
    lower_bound: torch.Tensor | None = None,
    upper_bound: torch.Tensor | None = None,
    k_count: int,
) -> torch.Tensor:
    dtype = f1.dtype
    device = f1.device
    IM = f1.shape[0]
    gravi = 1.0 / GRAV
    head = f1[:, :k_count]
    delp_head = delp[:, :k_count]
    levels = torch.arange(1, k_count + 1, dtype=torch.int64, device=device)
    in_range = torch.ones((IM, k_count), dtype=torch.bool, device=device)
    if active is not None:
        in_range = in_range & active[:, None]
    if lower_bound is not None:
        in_range = in_range & (levels[None, :] >= lower_bound[:, None])
    if upper_bound is not None:
        in_range = in_range & (levels[None, :] <= upper_bound[:, None])

    weighted = head * delp_head * gravi
    tsumn = torch.where(in_range & (head < 0.0), weighted, torch.zeros_like(weighted)).cumsum(dim=1)[:, -1].clone()  # source order
    tsump = torch.where(in_range & (head > 0.0), weighted, torch.zeros_like(weighted)).cumsum(dim=1)[:, -1].clone()

    rtnp = torch.ones((IM,), dtype=dtype, device=device)
    has_both = (tsump > 0.0) & (tsumn < 0.0)
    # Each divisor is benign wherever its branch is unselected: a tiny nonzero sum there would give
    # sum**2 -> 0 in the division backward and 0 * inf = NaN. Selected sums can be subnormal
    # (tracer round-off), so the quotients take range-safe derivatives (ufs_physics_common.quotient).
    safe_tsump = torch.where(has_both & (tsump > torch.abs(tsumn)), tsump, torch.ones_like(tsump))
    safe_tsumn = torch.where(has_both & (tsump <= torch.abs(tsumn)), tsumn, -torch.ones_like(tsumn))
    rtnp = torch.where(has_both & (tsump > torch.abs(tsumn)), quotient(tsumn, safe_tsump), rtnp)
    rtnp = torch.where(has_both & (tsump <= torch.abs(tsumn)), quotient(tsump, safe_tsumn), rtnp)

    active_adjust = (rtnp < 0.0)[:, None] & in_range
    positive_surplus = (tsump > torch.abs(tsumn))[:, None]
    negative = head < 0.0
    positive = head > 0.0
    scale = (1.0 + rtnp)[:, None]
    adjusted_head = torch.where(active_adjust & positive_surplus & negative, torch.zeros_like(head), head)
    adjusted_head = torch.where(active_adjust & positive_surplus & positive, scale * adjusted_head, adjusted_head)
    adjusted_head = torch.where(active_adjust & (~positive_surplus) & negative, scale * adjusted_head, adjusted_head)
    adjusted_head = torch.where(active_adjust & (~positive_surplus) & positive, torch.zeros_like(adjusted_head), adjusted_head)

    if k_count == f1.shape[1]:
        return adjusted_head
    return torch.cat((adjusted_head, f1[:, k_count:]), dim=1)


def _tke_system_rhs(
    te: torch.Tensor,
    delp: torch.Tensor,
    prsl: torch.Tensor,
    rdzt: torch.Tensor,
    dkq: torch.Tensor,
    e_half: torch.Tensor,
    xmf: torch.Tensor,
    xmfd: torch.Tensor,
    qcko_tke: torch.Tensor,
    qcdo_tke: torch.Tensor,
    kpbl: torch.Tensor,
    mrad: torch.Tensor,
    krad: torch.Tensor,
    pcnvflg: torch.Tensor,
    scuflg: torch.Tensor,
    dt2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    KM = te.shape[1]
    dtype = te.dtype
    device = te.device
    ncol = te.shape[0]
    km1 = KM - 1

    diagonal = torch.zeros((ncol, KM), dtype=dtype, device=device)
    dtodsd = dt2 / delp[:, :km1]
    dtodsu = dt2 / delp[:, 1:KM]
    dsig = prsl[:, :km1] - prsl[:, 1:KM]
    rdz = rdzt[:, :km1]
    dsdz2 = dsig * dkq[:, :km1] * rdz * rdz
    upper = -dtodsd * dsdz2
    lower = -dtodsu * dsdz2
    diagonal[:, 0] = 1.0 - upper[:, 0]
    diagonal[:, 1:km1] = 1.0 - lower[:, : km1 - 1] - upper[:, 1:km1]
    diagonal[:, KM - 1] = 1.0 - lower[:, km1 - 1]
    tem2 = dsig * rdz

    levels = torch.arange(1, KM, dtype=torch.int64, device=device)
    updraft_active = pcnvflg[:, None] & (levels[None, :] < kpbl[:, None])
    ptem = 0.5 * tem2 * xmf[:, :km1]
    ptem1 = dtodsd * ptem
    ptem2 = dtodsu * ptem
    ptem_q = qcko_tke[:, :km1] + qcko_tke[:, 1:KM]
    zeros_2d = torch.zeros((ncol, km1), dtype=dtype, device=device)
    rhs_left_delta = torch.where(updraft_active, -ptem_q * ptem1, zeros_2d)
    rhs_right_delta = torch.where(updraft_active, ptem_q * ptem2, zeros_2d)

    downdraft_active = scuflg[:, None] & (levels[None, :] >= mrad[:, None]) & (levels[None, :] < krad[:, None])
    ptem = 0.5 * tem2 * xmfd[:, :km1]
    ptem1 = dtodsd * ptem
    ptem2 = dtodsu * ptem
    ptem_q = qcdo_tke[:, :km1] + qcdo_tke[:, 1:KM]
    down_left = torch.where(downdraft_active, ptem_q * ptem1, zeros_2d)
    down_right = torch.where(downdraft_active, -ptem_q * ptem2, zeros_2d)

    kmx = torch.maximum(kpbl, krad)
    tvd_active = ((pcnvflg | scuflg)[:, None]) & (levels[None, :] < kmx[:, None])
    ptem = tem2 * (xmf[:, :km1] - xmfd[:, :km1])
    ptem1 = dtodsd * ptem
    ptem2 = dtodsu * ptem
    transport_left = torch.where(tvd_active, e_half[:, :km1] * ptem1, zeros_2d)
    transport_right = torch.where(tvd_active, -e_half[:, :km1] * ptem2, zeros_2d)
    # Each source row receives the interface below before the interface above (no zero padding:
    # the first row takes only the terms above, the last only the terms below).
    below = torch.cat((te[:, :1], te[:, 1:KM] + rhs_right_delta + down_right + transport_right), dim=1)
    rhs = torch.cat((below[:, :km1] + rhs_left_delta + down_left + transport_left, below[:, km1:]), dim=1)
    return lower, diagonal, upper, rhs


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    NTKE = state.require("q1").shape[2]
    KM = state.require("q1").shape[1]
    NTRAC = state.require("q1").shape[2]
    te = state.require("te")
    tke = state.require("tke")
    tkeh = state.require("tkeh")
    ten_t = state.require("ten_t")
    delp = state.require("del")
    prsl = state.require("prsl")
    rdzt = state.require("rdzt")
    dkq = state.require("dkq")
    q1 = state.require("q1")
    rtg = state.require("rtg")
    dtend = state.require("dtend").clone()
    e_half = state.require("e_half")
    xmf = state.require("xmf")
    xmfd = state.require("xmfd")
    qcko = state.require("qcko")
    qcdo = state.require("qcdo")
    delt = state.require("delt")
    kpbl = state.require("kpbl").to(torch.int64)
    krad = state.require("krad").to(torch.int64)
    mrad = state.require("mrad").to(torch.int64)
    pcnvflg = state.require("pcnvflg") > 0.5
    scuflg = state.require("scuflg") > 0.5
    IM = te.shape[0]
    require_float64("te", te, (IM, KM))
    require_float64("tke", tke, (IM, KM))
    require_float64("tkeh", tkeh, (IM, KM))
    require_float64("ten_t", ten_t, (IM, KM))
    require_float64("del", delp, (IM, KM))
    require_float64("prsl", prsl, (IM, KM))
    require_float64("rdzt", rdzt, (IM, KM - 1))
    require_float64("dkq", dkq, (IM, KM - 1))
    require_float64("q1", q1, (IM, KM, NTRAC))
    require_float64("rtg", rtg, (IM, KM, NTRAC))
    slots = diagnostic_slots(state)
    require_float64("e_half", e_half, (IM, KM - 1))
    require_float64("xmf", xmf, (IM, KM))
    require_float64("xmfd", xmfd, (IM, KM))
    require_float64("qcko", qcko, (IM, KM, NTRAC))
    require_float64("qcdo", qcdo, (IM, KM, NTRAC))

    dtype = te.dtype
    device = te.device
    dt2 = delt[0]
    rdt = 1.0 / dt2
    km1 = KM - 1

    lower, diagonal, upper, rhs = run_compile_island(
        "tridiagonal",
        "tke_system_rhs",
        _tke_system_rhs,
        te,
        delp,
        prsl,
        rdzt,
        dkq,
        e_half,
        xmf,
        xmfd,
        qcko[:, :, NTKE - 1],
        qcdo[:, :, NTKE - 1],
        kpbl,
        mrad,
        krad,
        pcnvflg,
        scuflg,
        dt2,
    )

    f1 = run_compile_island("tridiagonal", "tke_thomas", _tridit_tke, lower, diagonal, upper, rhs)

    active = pcnvflg | scuflg
    kbx = torch.ones((IM,), dtype=torch.int64, device=device)
    kmx = torch.maximum(kpbl, krad)
    p_only = pcnvflg & (~scuflg)
    s_only = (~pcnvflg) & scuflg
    kmx = torch.where(p_only, kpbl, kmx)
    kbx = torch.where(s_only, mrad, kbx)
    kmx = torch.where(s_only, krad, kmx)

    f1 = run_compile_island(
        "tridiagonal",
        "tke_borrow_active",
        _borrow_negative_energy,
        f1,
        delp,
        active=active,
        lower_bound=kbx,
        upper_bound=kmx,
        k_count=KM // 2,
    )
    f1 = run_compile_island(
        "tridiagonal",
        "tke_borrow_full",
        _borrow_negative_energy,
        f1,
        delp,
        k_count=KM,
    )

    qtend = (f1 - q1[:, :, NTKE - 1]) * rdt
    rtg = torch.cat((rtg[:, :, : NTKE - 1], (rtg[:, :, NTKE - 1] + qtend)[:, :, None], rtg[:, :, NTKE:]), dim=2)
    if bool(state.config.get("ldiag3d", False)):
        dtend = add_diagnostic_(dtend.clone(), qtend, slots[3 + NTKE - 1])

    return state.with_updates(
        {
            "rtg": rtg,
            "dtend": dtend,
            "tke_tridiagonal_solution": f1,
        }
    )


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
