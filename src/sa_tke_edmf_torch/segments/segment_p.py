# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment P: heat, moisture, and tracer tridiagonal recovery."""
from __future__ import annotations
import torch
from ufs_physics_common.quotient import quotient
from sa_tke_edmf_torch.compile import run_compile_island
from sa_tke_edmf_torch.constants import CP, ELOCP, FV, GRAV, HVAP, IM, KM, NDTEND, NTCW, NTIW, NTKE, NTQV, NTRAC, NTRW, QLMIN, QMIN, RD
from sa_tke_edmf_torch.diagnostics import diagnostic_slots, add_diagnostic_
from sa_tke_edmf_torch.segments.segment_o import _borrow_negative_energy
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64
from sa_tke_edmf_torch.tracers import satmedmf_hydrometeor_indices
CHECKPOINT_FIELDS = ('rtg', 'ten_t', 'dtend', 'gt0', 'gq0')

def _tridin_heat_tracers(lower: torch.Tensor, diagonal: torch.Tensor, upper: torch.Tensor, rhs_heat: torch.Tensor, rhs_tracers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    KM = diagonal.shape[1]
    # Level-major rows: each sweep step reads one contiguous row ([ncol] or [ncol, ntrac]) per operand.
    lower, diagonal, upper, rhs_heat = (x.t().contiguous() for x in (lower, diagonal, upper, rhs_heat))
    rhs_tracers = rhs_tracers.movedim(1, 0).contiguous()
    fk = 1.0 / diagonal[0]
    previous = (fk * upper[0], fk * rhs_heat[0], fk[:, None] * rhs_tracers[0])
    pieces = [[value[None]] for value in previous]
    for first in range(1, KM - 1, 4):
        stop = min(first + 4, KM - 1)
        result = run_compile_island(
            "tridiagonal", "heat_forward_levels", _heat_forward_levels,
            lower[first - 1:stop - 1], diagonal[first:stop], upper[first:stop],
            rhs_heat[first:stop], rhs_tracers[first:stop], *previous,
        )
        for rows, value in zip(pieces, result, strict=True):
            rows.append(value)
        previous = tuple(value[-1] for value in result)
    upper_rows, heat_rows, tracer_rows = (torch.cat(rows, dim=0) for rows in pieces)
    fk = 1.0 / (diagonal[KM - 1] - lower[KM - 2] * previous[0])
    below_heat = fk * (rhs_heat[KM - 1] - lower[KM - 2] * previous[1])
    below_tracers = fk[:, None] * (rhs_tracers[KM - 1] - lower[KM - 2][:, None] * previous[2])
    heat_pieces, tracer_pieces = [below_heat[None]], [below_tracers[None]]
    for stop in range(KM - 1, 0, -4):
        first = max(0, stop - 4)
        heat, tracers = run_compile_island(
            "tridiagonal", "heat_backward_levels", _heat_backward_levels,
            upper_rows[first:stop], heat_rows[first:stop], tracer_rows[first:stop],
            below_heat, below_tracers,
        )
        heat_pieces.append(heat)
        tracer_pieces.append(tracers)
        below_heat, below_tracers = heat[0], tracers[0]
    return (torch.cat(list(reversed(heat_pieces)), dim=0).t().contiguous(),
            torch.cat(list(reversed(tracer_pieces)), dim=0).movedim(0, 1).contiguous())


def _heat_forward_levels(lower, diagonal, upper, rhs_heat, rhs_tracers,
                          previous_upper, previous_heat, previous_tracers):
    upper_rows, heat_rows, tracer_rows = [], [], []
    for k in range(diagonal.shape[0]):
        factor = 1.0 / (diagonal[k] - lower[k] * previous_upper)
        previous_upper = factor * upper[k]
        previous_heat = factor * (rhs_heat[k] - lower[k] * previous_heat)
        previous_tracers = factor[:, None] * (rhs_tracers[k] - lower[k][:, None] * previous_tracers)
        upper_rows.append(previous_upper)
        heat_rows.append(previous_heat)
        tracer_rows.append(previous_tracers)
    return tuple(torch.stack(rows, dim=0) for rows in (upper_rows, heat_rows, tracer_rows))


def _heat_backward_levels(upper, heat, tracers, below_heat, below_tracers):
    # Bounded outputs prevent a 54-level/9-tracer fused kernel's 12-16 KiB thread stack.
    heat_rows, tracer_rows = [], []
    for k in range(upper.shape[0] - 1, -1, -1):
        below_heat = heat[k] - upper[k] * below_heat
        below_tracers = tracers[k] - upper[k][:, None] * below_tracers
        heat_rows.append(below_heat)
        tracer_rows.append(below_tracers)
    return (torch.stack(list(reversed(heat_rows)), dim=0),
            torch.stack(list(reversed(tracer_rows)), dim=0))

def _mass_flux_bounds(pcnvflg: torch.Tensor, scuflg: torch.Tensor, kpbl: torch.Tensor, mrad: torch.Tensor, krad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = kpbl.device
    IM = kpbl.shape[0]
    active = pcnvflg | scuflg
    lower_bound = torch.ones((IM,), dtype=torch.int64, device=device)
    upper_bound = torch.maximum(kpbl, krad)
    p_only = pcnvflg & ~scuflg
    s_only = ~pcnvflg & scuflg
    upper_bound = torch.where(p_only, kpbl, upper_bound)
    lower_bound = torch.where(s_only, mrad, lower_bound)
    upper_bound = torch.where(s_only, krad, upper_bound)
    return (active, lower_bound, upper_bound)

def _borrow_negative_energy_many(f1: torch.Tensor, delp: torch.Tensor, *, active: torch.Tensor | None=None, lower_bound: torch.Tensor | None=None, upper_bound: torch.Tensor | None=None, k_count: int) -> torch.Tensor:
    dtype = f1.dtype
    device = f1.device
    ncol = f1.shape[0]
    nfield = f1.shape[2]
    gravi = 1.0 / GRAV
    head = f1[:, :k_count, :]
    delp_head = delp[:, :k_count, None]
    levels = torch.arange(1, k_count + 1, dtype=torch.int64, device=device)
    in_range = torch.ones((ncol, k_count, 1), dtype=torch.bool, device=device)
    if active is not None:
        in_range = in_range & active[:, None, None]
    if lower_bound is not None:
        in_range = in_range & (levels[None, :, None] >= lower_bound[:, None, None])
    if upper_bound is not None:
        in_range = in_range & (levels[None, :, None] <= upper_bound[:, None, None])
    weighted = head * delp_head * gravi
    tsumn = torch.where(in_range & (head < 0.0), weighted, 0.0).cumsum(dim=1)[:, -1].clone()  # source order
    tsump = torch.where(in_range & (head > 0.0), weighted, 0.0).cumsum(dim=1)[:, -1].clone()
    rtnp = torch.ones((ncol, nfield), dtype=dtype, device=device)
    has_both = (tsump > 0.0) & (tsumn < 0.0)
    # Each divisor is benign wherever its branch is unselected: a tiny nonzero sum there would give
    # sum**2 -> 0 in the division backward and 0 * inf = NaN. Selected sums can be subnormal
    # (tracer round-off), so the quotients take range-safe derivatives (ufs_physics_common.quotient).
    safe_tsump = torch.where(has_both & (tsump > torch.abs(tsumn)), tsump, torch.ones_like(tsump))
    safe_tsumn = torch.where(has_both & (tsump <= torch.abs(tsumn)), tsumn, -torch.ones_like(tsumn))
    rtnp = torch.where(has_both & (tsump > torch.abs(tsumn)), quotient(tsumn, safe_tsump), rtnp)
    rtnp = torch.where(has_both & (tsump <= torch.abs(tsumn)), quotient(tsump, safe_tsumn), rtnp)
    active_adjust = (rtnp[:, None, :] < 0.0) & in_range
    positive_surplus = (tsump > torch.abs(tsumn))[:, None, :]
    negative = head < 0.0
    positive = head > 0.0
    scale = (1.0 + rtnp)[:, None, :]
    adjusted_head = torch.where(active_adjust & positive_surplus & negative, 0.0, head)
    adjusted_head = torch.where(active_adjust & positive_surplus & positive, scale * adjusted_head, adjusted_head)
    adjusted_head = torch.where(active_adjust & ~positive_surplus & negative, scale * adjusted_head, adjusted_head)
    adjusted_head = torch.where(active_adjust & ~positive_surplus & positive, 0.0, adjusted_head)
    if k_count == f1.shape[1]:
        return adjusted_head
    return torch.cat((adjusted_head, f1[:, k_count:, :]), dim=1)

def _borrow_negative_hydrometeor_from_vapor(heat_state: torch.Tensor, tracer_state: torch.Tensor, tracer_index: int, active: torch.Tensor, lower_bound: torch.Tensor, upper_bound: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    KM = heat_state.shape[1]
    device = heat_state.device
    k_count = KM // 2
    # The source loop covers levels 1..KM/2; above them every mask is false and the state passes through.
    levels = torch.arange(1, KM + 1, dtype=torch.int64, device=device)
    in_range = active[:, None] & (levels[None, :] >= lower_bound[:, None]) & (levels[None, :] <= upper_bound[:, None]) & (levels[None, :] <= k_count)
    hyd = tracer_state[:, :, tracer_index]
    vapor = tracer_state[:, :, NTQV - 1]
    negative_hyd = in_range & (hyd < 0.0)
    combined = vapor + hyd
    vapor_covers = negative_hyd & (combined >= 0.0)
    vapor_positive = negative_hyd & (combined < 0.0) & (vapor > 0.0)
    heat_adjusted = torch.where(vapor_covers, heat_state - ELOCP * hyd, heat_state)
    vapor_adjusted = torch.where(vapor_covers, combined, vapor)
    hyd_adjusted = torch.where(vapor_covers, 0.0, hyd)
    heat_adjusted = torch.where(vapor_positive, heat_adjusted + ELOCP * vapor, heat_adjusted)
    hyd_adjusted = torch.where(vapor_positive, combined, hyd_adjusted)
    vapor_adjusted = torch.where(vapor_positive, 0.0, vapor_adjusted)
    slot = torch.arange(tracer_state.shape[2], device=device)
    tracer_result = torch.where(slot == tracer_index, hyd_adjusted[:, :, None],
                                torch.where(slot == NTQV - 1, vapor_adjusted[:, :, None], tracer_state))
    return (heat_adjusted, tracer_result)

def _heat_tracer_system_rhs(q1: torch.Tensor, t1: torch.Tensor, delp: torch.Tensor, prsl: torch.Tensor, zi: torch.Tensor, rdzt: torch.Tensor, dkt: torch.Tensor, q_half: torch.Tensor, tcko: torch.Tensor, tcdo: torch.Tensor, qcko: torch.Tensor, qcdo: torch.Tensor, xmf: torch.Tensor, xmfd: torch.Tensor, heat: torch.Tensor, evap: torch.Tensor, kpbl: torch.Tensor, mrad: torch.Tensor, krad: torch.Tensor, pcnvflg: torch.Tensor, scuflg: torch.Tensor, dt2: torch.Tensor, sa3dtke: bool, use_lpt: bool, ntcw: int = NTCW, ntiw: int = NTIW) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    KM = q1.shape[1]
    NTRAC = q1.shape[2]
    dtype = q1.dtype
    device = q1.device
    ncol = q1.shape[0]
    km1 = KM - 1
    ntrac1 = NTRAC - 1
    qliq = torch.clamp_min(q1[:, :, ntcw - 1], QLMIN)
    qice_source = q1[:, :, ntiw - 1]
    if sa3dtke:
        # satmedmfvdifq.F:651 sums q1(ntiw) + q1(5) + q1(6) for SA-3D-TKE.
        qice_source = qice_source + q1[:, :, 4] + q1[:, :, 5]
    qice = torch.clamp_min(qice_source, QLMIN)
    dtdz1 = dt2 / (zi[:, 1] - zi[:, 0])
    dtodsd = dt2 / delp[:, :km1]
    dtodsu = dt2 / delp[:, 1:KM]
    dsig = prsl[:, :km1] - prsl[:, 1:KM]
    rdz = rdzt[:, :km1]
    tem1 = dsig * dkt[:, :km1] * rdz
    dsdzt = tem1 * (GRAV / CP)
    if use_lpt:
        dsdzt = dsdzt - tem1 * ELOCP * (qliq[:, 1:KM] - qliq[:, :km1]) * rdz
        dsdzt = dsdzt - (1.0 + 0.33 / 2.5) * tem1 * ELOCP * (qice[:, 1:KM] - qice[:, :km1]) * rdz
    dsdz2 = tem1 * rdz
    upper = -dtodsd * dsdz2
    lower = -dtodsu * dsdz2
    diagonal = torch.zeros((ncol, KM), dtype=dtype, device=device)
    diagonal[:, 0] = 1.0 - upper[:, 0]
    diagonal[:, 1:km1] = 1.0 - lower[:, :km1 - 1] - upper[:, 1:km1]
    diagonal[:, KM - 1] = 1.0 - lower[:, km1 - 1]
    rhs_heat = t1.clone()
    rhs_heat[:, 0] = rhs_heat[:, 0] + dtdz1 * heat
    # The source initializes each row from the interface below before adding
    # the interface above. Retain that order, including both plume terms.
    rhs_heat[:, 1:KM] = rhs_heat[:, 1:KM] - dtodsu * dsdzt
    rhs_tracers = q1[:, :, :ntrac1].clone()
    rhs_tracers[:, 0, NTQV - 1] = rhs_tracers[:, 0, NTQV - 1] + dtdz1 * evap
    levels = torch.arange(1, KM, dtype=torch.int64, device=device)
    tem2 = dsig * rdz
    zeros_2d = torch.zeros((ncol, km1), dtype=dtype, device=device)
    zeros_3d = torch.zeros((ncol, km1, ntrac1), dtype=dtype, device=device)
    updraft_active = pcnvflg[:, None] & (levels[None, :] < kpbl[:, None])
    ptem = 0.5 * tem2 * xmf[:, :km1]
    ptem1 = dtodsd * ptem
    ptem2 = dtodsu * ptem
    heat_delta = tcko[:, :km1] + tcko[:, 1:KM] - (t1[:, :km1] + t1[:, 1:KM])
    up_heat = torch.where(updraft_active, -heat_delta * ptem1, zeros_2d)
    rhs_heat[:, 1:KM] = rhs_heat[:, 1:KM] + torch.where(updraft_active, heat_delta * ptem2, zeros_2d)
    tracer_sum = qcko[:, :km1, :ntrac1] + qcko[:, 1:KM, :ntrac1]
    up_tracers = torch.where(updraft_active[:, :, None], -tracer_sum * ptem1[:, :, None], zeros_3d)
    rhs_tracers[:, 1:KM, :] = rhs_tracers[:, 1:KM, :] + torch.where(updraft_active[:, :, None], tracer_sum * ptem2[:, :, None], zeros_3d)
    downdraft_active = scuflg[:, None] & (levels[None, :] >= mrad[:, None]) & (levels[None, :] < krad[:, None])
    ptem = 0.5 * tem2 * xmfd[:, :km1]
    ptem1 = dtodsd * ptem
    ptem2 = dtodsu * ptem
    heat_delta = tcdo[:, :km1] + tcdo[:, 1:KM] - (t1[:, :km1] + t1[:, 1:KM])
    down_heat = torch.where(downdraft_active, heat_delta * ptem1, zeros_2d)
    rhs_heat[:, 1:KM] = rhs_heat[:, 1:KM] + torch.where(downdraft_active, -heat_delta * ptem2, zeros_2d)
    tracer_sum = qcdo[:, :km1, :ntrac1] + qcdo[:, 1:KM, :ntrac1]
    down_tracers = torch.where(downdraft_active[:, :, None], tracer_sum * ptem1[:, :, None], zeros_3d)
    rhs_tracers[:, 1:KM, :] = rhs_tracers[:, 1:KM, :] + torch.where(downdraft_active[:, :, None], -tracer_sum * ptem2[:, :, None], zeros_3d)
    kmx = torch.maximum(kpbl, krad)
    tvd_active = (pcnvflg | scuflg)[:, None] & (levels[None, :] < kmx[:, None])
    ptem = tem2 * (xmf[:, :km1] - xmfd[:, :km1])
    ptem1 = dtodsd * ptem
    ptem2 = dtodsu * ptem
    transport_tracers = torch.where(tvd_active[:, :, None], q_half[:, :km1, :] * ptem1[:, :, None], zeros_3d)
    rhs_tracers[:, 1:KM, :] = rhs_tracers[:, 1:KM, :] + torch.where(tvd_active[:, :, None], -q_half[:, :km1, :] * ptem2[:, :, None], zeros_3d)
    rhs_heat[:, :km1] = rhs_heat[:, :km1] + dtodsd * dsdzt
    rhs_heat[:, :km1] = rhs_heat[:, :km1] + up_heat
    rhs_heat[:, :km1] = rhs_heat[:, :km1] + down_heat
    rhs_tracers[:, :km1, :] = rhs_tracers[:, :km1, :] + up_tracers
    rhs_tracers[:, :km1, :] = rhs_tracers[:, :km1, :] + down_tracers
    rhs_tracers[:, :km1, :] = rhs_tracers[:, :km1, :] + transport_tracers
    return (lower, diagonal, upper, rhs_heat, rhs_tracers)

def _heat_tracer_recovery(q1, t1, delp, prsl, zi, rdzt, dkt, q_half, tcko, tcdo, qcko, qcdo, xmf, xmfd, heat, evap, dt2, rtg, dtend, kpbl, krad, mrad, pcnvflg, scuflg, use_lpt, sa3dtke, ldiag3d, gen_tend, ntcw=NTCW, ntiw=NTIW, ntrw=NTRW, slots=None):
    KM = q1.shape[1]
    NTRAC = q1.shape[2]
    dtend = dtend.clone()
    dtype = q1.dtype
    device = q1.device
    rdt = 1.0 / dt2
    km1 = KM - 1
    ntrac1 = NTRAC - 1
    active, lower_bound, upper_bound = _mass_flux_bounds(pcnvflg, scuflg, kpbl, mrad, krad)
    def region(name, fn, *args, **kwargs):
        return run_compile_island("tridiagonal", name, fn, *args, **kwargs)

    lower, diagonal, upper, rhs_heat, rhs_tracers = region("heat_tracer_rhs", _heat_tracer_system_rhs, q1, t1, delp, prsl, zi, rdzt, dkt, q_half, tcko, tcdo, qcko, qcdo, xmf, xmfd, heat, evap, kpbl, mrad, krad, pcnvflg, scuflg, dt2, sa3dtke, use_lpt, ntcw, ntiw)
    solved_heat, solved_tracers = region("heat_tracer_thomas", _tridin_heat_tracers, lower, diagonal, upper, rhs_heat, rhs_tracers)
    qv_solution = region("heat_vapor_borrow", _borrow_negative_energy, solved_tracers[:, :, NTQV - 1], delp, active=active, lower_bound=lower_bound, upper_bound=upper_bound, k_count=KM // 2)
    qv_solution = region("heat_vapor_borrow_all", _borrow_negative_energy, qv_solution, delp, k_count=KM)
    slot = torch.arange(ntrac1, device=device)
    solved_tracers = torch.where(slot == NTQV - 1, qv_solution[:, :, None], solved_tracers)
    solved_heat, solved_tracers = region("heat_hydrometeor_borrow", _borrow_negative_hydrometeor_from_vapor, solved_heat, solved_tracers, ntcw - 1, active, lower_bound, upper_bound)
    if ntrw > 0:
        solved_heat, solved_tracers = region("heat_hydrometeor_borrow", _borrow_negative_hydrometeor_from_vapor, solved_heat, solved_tracers, ntrw - 1, active, lower_bound, upper_bound)
    nonvapor_tracers = region("heat_tracer_borrow", _borrow_negative_energy_many, solved_tracers[:, :, 1:ntrac1], delp, active=active, lower_bound=lower_bound, upper_bound=upper_bound, k_count=KM // 2)
    nonvapor_tracers = region("heat_tracer_borrow_all", _borrow_negative_energy_many, nonvapor_tracers, delp, k_count=KM)
    solved_tracers = torch.cat((solved_tracers[:, :, :1], nonvapor_tracers), dim=2)
    ten_t = (solved_heat - t1) * rdt
    rtg_qv = rtg[:, :, NTQV - 1] + (solved_tracers[:, :, NTQV - 1] - q1[:, :, NTQV - 1]) * rdt
    if ldiag3d and (not gen_tend):
        dtend = dtend.clone()
        add_diagnostic_(dtend, ten_t * dt2, slots[0])
        add_diagnostic_(dtend, (solved_tracers[:, :, NTQV - 1] - q1[:, :, NTQV - 1]), slots[3 + NTQV - 1])
    qtend_nonvapor = (solved_tracers[:, :, 1:ntrac1] - q1[:, :, 1:ntrac1]) * rdt
    # Vapor (slot NTQV = 1) and the other diffused tracers are updated; the TKE slots pass through.
    rtg = torch.cat((rtg_qv[:, :, None], rtg[:, :, 1:ntrac1] + qtend_nonvapor, rtg[:, :, ntrac1:]), dim=2)
    if ldiag3d and (not gen_tend):
        for n in range(1, ntrac1):
            add_diagnostic_(dtend, qtend_nonvapor[:, :, n - 1] * dt2, slots[3 + n])
    rho_a = prsl[:, 0] / (RD * t1[:, 0] * (1.0 + FV * torch.clamp_min(q1[:, 0, NTQV - 1], QMIN)))
    dtsfc = rho_a * CP * heat
    dqsfc = rho_a * HVAP * evap
    return (rtg, ten_t, dtend, dtsfc, dqsfc, solved_heat, solved_tracers)

def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    KM = state.require('q1').shape[1]
    NTRAC = state.require('q1').shape[2]
    q1 = state.require('q1')
    t1 = state.require('t1')
    gt0 = state.require('gt0')
    gq0 = state.require('gq0')
    delp = state.require('del')
    prsl = state.require('prsl')
    zi = state.require('zi')
    rdzt = state.require('rdzt')
    dkt = state.require('dkt')
    q_half = state.require('q_half')
    tcko = state.require('tcko')
    tcdo = state.require('tcdo')
    qcko = state.require('qcko')
    qcdo = state.require('qcdo')
    xmf = state.require('xmf')
    xmfd = state.require('xmfd')
    heat = state.require('heat')
    evap = state.require('evap')
    delt = state.require('delt')
    ten_t = state.require('ten_t')
    rtg = state.require('rtg')
    dtend = state.require('dtend')
    kpbl = state.require('kpbl').to(torch.int64)
    krad = state.require('krad').to(torch.int64)
    mrad = state.require('mrad').to(torch.int64)
    pcnvflg = state.require('pcnvflg') > 0.5
    scuflg = state.require('scuflg') > 0.5
    IM = q1.shape[0]
    require_float64('q1', q1, (IM, KM, NTRAC))
    require_float64('t1', t1, (IM, KM))
    require_float64('gt0', gt0, (IM, KM))
    require_float64('gq0', gq0, (IM, KM, NTRAC))
    require_float64('del', delp, (IM, KM))
    require_float64('prsl', prsl, (IM, KM))
    require_float64('zi', zi, (IM, KM + 1))
    require_float64('rdzt', rdzt, (IM, KM - 1))
    require_float64('dkt', dkt, (IM, KM))
    require_float64('q_half', q_half, (IM, KM - 1, NTRAC - 1))
    require_float64('tcko', tcko, (IM, KM))
    require_float64('tcdo', tcdo, (IM, KM))
    require_float64('qcko', qcko, (IM, KM, NTRAC))
    require_float64('qcdo', qcdo, (IM, KM, NTRAC))
    require_float64('xmf', xmf, (IM, KM))
    require_float64('xmfd', xmfd, (IM, KM))
    require_float64('ten_t', ten_t, (IM, KM))
    require_float64('rtg', rtg, (IM, KM, NTRAC))
    slots = diagnostic_slots(state)
    rtg, ten_t, dtend, dtsfc, dqsfc, solved_heat, solved_tracers = run_compile_island('tridiagonal', 'heat_tracer_recovery', _heat_tracer_recovery, q1, t1, delp, prsl, zi, rdzt, dkt, q_half, tcko, tcdo, qcko, qcdo, xmf, xmfd, heat, evap, delt[0], rtg, dtend, kpbl, krad, mrad, pcnvflg, scuflg, bool(state.config.get('use_lpt', False)), bool(state.config.get('sa3dtke', False)), bool(state.config.get('ldiag3d', False)), bool(state.config.get('gen_tend', False)), *satmedmf_hydrometeor_indices(state.config, NTRAC), slots)
    return state.with_updates({'rtg': rtg, 'ten_t': ten_t, 'dtend': dtend, 'dtsfc': dtsfc, 'dqsfc': dqsfc, 'heat_moisture_solution': solved_heat, 'tracer_solution': solved_tracers})

def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
