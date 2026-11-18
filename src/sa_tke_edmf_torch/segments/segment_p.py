# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment P: heat, moisture, and tracer tridiagonal recovery."""
from __future__ import annotations
import torch
from ufs_physics_common.ordered_sum import level_sum
from ufs_physics_common.quotient import quotient
from sa_tke_edmf_torch.compile import run_compile_island
from sa_tke_edmf_torch.constants import CP, ELOCP, FV, GRAV, HVAP, NTCW, NTIW, NTQV, NTRW, QLMIN, QMIN, RD
from sa_tke_edmf_torch.diagnostics import diagnostic_slots, add_diagnostic_
from sa_tke_edmf_torch.segments.segment_o import _borrow_negative_energy
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64
from sa_tke_edmf_torch.tracers import satmedmf_hydrometeor_indices

def _tridin_heat_tracers(lower: torch.Tensor, diagonal: torch.Tensor, upper: torch.Tensor, rhs_heat: torch.Tensor, rhs_tracers: torch.Tensor, *previous: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    KM = diagonal.shape[0]
    # Level-major rows (``_heat_tracer_system_rhs`` writes them, with the first sweep row in ``previous``): each
    # sweep step reads one contiguous row ([ncol] or [ncol, ntrac]) per operand. The joins and end rows are
    # compiled regions around the compiled level blocks (eager, they are strided copies and cats).
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
    upper_rows, heat_rows, tracer_rows, below_heat, below_tracers = run_compile_island(
        "tridiagonal", "heat_forward_join", _heat_forward_join, tuple(map(tuple, pieces)), diagonal[KM - 1],
        lower[KM - 2], rhs_heat[KM - 1], rhs_tracers[KM - 1])
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
    return run_compile_island("tridiagonal", "heat_backward_join", _heat_backward_join,
                              tuple(reversed(heat_pieces)), tuple(reversed(tracer_pieces)))


def _heat_forward_join(pieces, diagonal_last, lower_last, rhs_heat_last, rhs_tracers_last):
    upper_rows, heat_rows, tracer_rows = (torch.cat(rows, dim=0) for rows in pieces)
    previous = (upper_rows[-1], heat_rows[-1], tracer_rows[-1])
    fk = 1.0 / (diagonal_last - lower_last * previous[0])
    below_heat = fk * (rhs_heat_last - lower_last * previous[1])
    below_tracers = fk[:, None] * (rhs_tracers_last - lower_last[:, None] * previous[2])
    return upper_rows, heat_rows, tracer_rows, below_heat, below_tracers


def _heat_backward_join(heat_pieces, tracer_pieces):
    return (torch.cat(heat_pieces, dim=0).t().contiguous(),
            torch.cat(tracer_pieces, dim=0).movedim(0, 1).contiguous())


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
    # Bounded outputs keep a fused kernel over many levels and tracers from needing a large thread stack.
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
    tsumn = level_sum(torch.where(in_range & (head < 0.0), weighted, 0.0), 1)  # source order
    tsump = level_sum(torch.where(in_range & (head > 0.0), weighted, 0.0), 1)
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

def _borrow_negative_hydrometeor_from_vapor(heat_state: torch.Tensor, tracer_state: torch.Tensor, tracer_index: int, vapor: torch.Tensor, active: torch.Tensor, lower_bound: torch.Tensor, upper_bound: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Heat, the ``tracer_index`` hydrometeor and ``vapor`` after the hydrometeor borrows from vapor.

    The hydrometeor is read from ``tracer_state``; the updated slots are returned on their own (the caller joins
    the tracers once), not as a rewritten copy of every tracer."""
    KM = heat_state.shape[1]
    device = heat_state.device
    k_count = KM // 2
    # The source loop covers levels 1..KM/2; above them every mask is false and the state passes through.
    levels = torch.arange(1, KM + 1, dtype=torch.int64, device=device)
    in_range = active[:, None] & (levels[None, :] >= lower_bound[:, None]) & (levels[None, :] <= upper_bound[:, None]) & (levels[None, :] <= k_count)
    hyd = tracer_state[:, :, tracer_index]
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
    return (heat_adjusted, hyd_adjusted, vapor_adjusted)


def _borrow_tracer_negative_energy(tracers, slot, delp, **kwargs):
    # The slot is selected inside the region: a slot view passed in is copied to a contiguous field first.
    return _borrow_negative_energy(tracers[:, :, slot], delp, **kwargs)


def _borrow_nonvapor_negative_energy(tracers, hydrometeors, slots, delp, **kwargs):
    # The non-vapor tracers with the borrowed hydrometeor slots, joined inside the region (no tracer copy).
    values = tracers[:, :, 1:]
    slot = torch.arange(1, tracers.shape[2], device=tracers.device)
    for index, value in zip(slots, hydrometeors, strict=True):
        values = torch.where(slot == index, value[:, :, None], values)
    return _borrow_negative_energy_many(values, delp, **kwargs)

def _heat_tracer_system_rhs(q1: torch.Tensor, t1: torch.Tensor, delp: torch.Tensor, prsl: torch.Tensor, zi: torch.Tensor, rdzt: torch.Tensor, dkt: torch.Tensor, q_half: torch.Tensor, tcko: torch.Tensor, tcdo: torch.Tensor, qcko: torch.Tensor, qcdo: torch.Tensor, xmf: torch.Tensor, xmfd: torch.Tensor, heat: torch.Tensor, evap: torch.Tensor, kpbl: torch.Tensor, mrad: torch.Tensor, krad: torch.Tensor, pcnvflg: torch.Tensor, scuflg: torch.Tensor, dt2: torch.Tensor, sa3dtke: bool, use_lpt: bool, ntcw: int = NTCW, ntiw: int = NTIW) -> tuple[torch.Tensor, ...]:
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
    # Level-major outputs [level, column(, tracer)] for the Thomas sweep, each written once: no zero-filled
    # matrix or row-by-row slice assignments. Every element keeps the source's terms in the source's order.
    diagonal = torch.cat(((1.0 - upper[:, 0])[None], (1.0 - lower[:, :km1 - 1] - upper[:, 1:km1]).t(),
                          (1.0 - lower[:, km1 - 1])[None]), dim=0)
    heat_from_below = t1[:, 1:KM] - dtodsu * dsdzt
    vapor = NTQV - 1
    tracers_bottom = q1[:, 0, :ntrac1]
    tracers_bottom = torch.cat((tracers_bottom[:, :vapor], (tracers_bottom[:, vapor] + dtdz1 * evap)[:, None],
                                tracers_bottom[:, vapor + 1:]), dim=1)
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
    heat_from_below = heat_from_below + torch.where(updraft_active, heat_delta * ptem2, zeros_2d)
    up_sum = qcko[:, :km1, :ntrac1] + qcko[:, 1:KM, :ntrac1]
    up_tracers = torch.where(updraft_active[:, :, None], -up_sum * ptem1[:, :, None], zeros_3d)
    up_ptem2 = ptem2
    downdraft_active = scuflg[:, None] & (levels[None, :] >= mrad[:, None]) & (levels[None, :] < krad[:, None])
    ptem = 0.5 * tem2 * xmfd[:, :km1]
    ptem1 = dtodsd * ptem
    ptem2 = dtodsu * ptem
    heat_delta = tcdo[:, :km1] + tcdo[:, 1:KM] - (t1[:, :km1] + t1[:, 1:KM])
    down_heat = torch.where(downdraft_active, heat_delta * ptem1, zeros_2d)
    heat_from_below = heat_from_below + torch.where(downdraft_active, -heat_delta * ptem2, zeros_2d)
    down_sum = qcdo[:, :km1, :ntrac1] + qcdo[:, 1:KM, :ntrac1]
    down_tracers = torch.where(downdraft_active[:, :, None], down_sum * ptem1[:, :, None], zeros_3d)
    down_ptem2 = ptem2
    kmx = torch.maximum(kpbl, krad)
    tvd_active = (pcnvflg | scuflg)[:, None] & (levels[None, :] < kmx[:, None])
    ptem = tem2 * (xmf[:, :km1] - xmfd[:, :km1])
    ptem1 = dtodsd * ptem
    ptem2 = dtodsu * ptem
    transport_tracers = torch.where(tvd_active[:, :, None], q_half[:, :km1, :] * ptem1[:, :, None], zeros_3d)
    tvd_ptem2 = ptem2

    def tracers_from_below(rows):
        # Interface-below terms of the RHS rows ``rows`` + 1 (row-local: the same expression for any row range).
        value = q1[:, 1:KM, :ntrac1][:, rows] + torch.where(updraft_active[:, rows, None],
            up_sum[:, rows] * up_ptem2[:, rows, None], zeros_3d[:, rows])
        value = value + torch.where(downdraft_active[:, rows, None], -down_sum[:, rows] * down_ptem2[:, rows, None],
                                    zeros_3d[:, rows])
        return value + torch.where(tvd_active[:, rows, None], -q_half[:, :km1, :][:, rows] * tvd_ptem2[:, rows, None],
                                   zeros_3d[:, rows])
    # The source initializes each row from the interface below (levels 1..KM-1, done above) before adding the
    # interface above (levels 0..KM-2), both plume terms included; the bottom row starts from the surface flux.
    heat_from_above = dtodsd * dsdzt
    heat_bottom = (((t1[:, 0] + dtdz1 * heat) + heat_from_above[:, 0]) + up_heat[:, 0]) + down_heat[:, 0]
    heat_middle = ((heat_from_below[:, :km1 - 1] + heat_from_above[:, 1:]) + up_heat[:, 1:]) + down_heat[:, 1:]
    rhs_heat = torch.cat((heat_bottom[None], heat_middle.t(), heat_from_below[:, km1 - 1][None]), dim=0)
    # All KM rows as one expression, so each plume term is read once and the level-major result is written once:
    # compiled on CUDA, joining the rows 0..KM-2 with the top row stores both parts [ncol, levels, ntrac] and copies
    # them again into the join. Row r > 0 starts from the
    # interface-below terms of row r - 1; the plume terms (rows 0..KM-2) are padded with a row that the top row,
    # which takes the interface-below terms alone, never selects.
    row = torch.arange(KM, device=device)[None, :, None]
    first = torch.where(row == 0, tracers_bottom[:, None],
                        torch.nn.functional.pad(tracers_from_below(slice(0, km1)), (0, 0, 1, 0)))
    plume = [torch.nn.functional.pad(term, (0, 0, 0, 1)) for term in (up_tracers, down_tracers, transport_tracers)]
    rhs_tracers = torch.where(row < km1, ((first + plume[0]) + plume[1]) + plume[2], first)
    rhs_tracers = rhs_tracers.movedim(1, 0).contiguous()
    lower, upper = lower.t().contiguous(), upper.t().contiguous()
    # The Thomas sweep's first row.
    fk = 1.0 / diagonal[0]
    return (lower, diagonal, upper, rhs_heat, rhs_tracers,
            fk * upper[0], fk * rhs_heat[0], fk[:, None] * rhs_tracers[0])

def _heat_tracer_recovery(q1, t1, delp, prsl, zi, rdzt, dkt, q_half, tcko, tcdo, qcko, qcdo, xmf, xmfd, heat, evap, dt2, rtg, dtend, kpbl, krad, mrad, pcnvflg, scuflg, use_lpt, sa3dtke, ldiag3d, gen_tend, ntcw=NTCW, ntiw=NTIW, ntrw=NTRW, slots=None):
    KM = q1.shape[1]
    NTRAC = q1.shape[2]
    dtend = dtend.clone()
    ntrac1 = NTRAC - 1
    active, lower_bound, upper_bound = _mass_flux_bounds(pcnvflg, scuflg, kpbl, mrad, krad)
    def region(name, fn, *args, **kwargs):
        return run_compile_island("tridiagonal", name, fn, *args, **kwargs)

    system = region("heat_tracer_rhs", _heat_tracer_system_rhs, q1, t1, delp, prsl, zi, rdzt, dkt, q_half, tcko, tcdo, qcko, qcdo, xmf, xmfd, heat, evap, kpbl, mrad, krad, pcnvflg, scuflg, dt2, sa3dtke, use_lpt, ntcw, ntiw)
    solved_heat, solved_tracers = region("heat_tracer_thomas", _tridin_heat_tracers, *system)
    # Vapor and the borrowing hydrometeors are carried as their own fields and joined with the other tracers once
    # (in _heat_tracer_tendencies), so a borrow does not rewrite all tracers.
    vapor = region("heat_vapor_borrow", _borrow_tracer_negative_energy, solved_tracers, NTQV - 1, delp, active=active, lower_bound=lower_bound, upper_bound=upper_bound, k_count=KM // 2)
    vapor = region("heat_vapor_borrow_all", _borrow_negative_energy, vapor, delp, k_count=KM)
    hydrometeors, hydrometeor_slots = [], [ntcw - 1] + ([ntrw - 1] if ntrw > 0 else [])
    for index in hydrometeor_slots:
        solved_heat, hydrometeor, vapor = region("heat_hydrometeor_borrow", _borrow_negative_hydrometeor_from_vapor, solved_heat, solved_tracers, index, vapor, active, lower_bound, upper_bound)
        hydrometeors.append(hydrometeor)
    nonvapor_tracers = region("heat_tracer_borrow", _borrow_nonvapor_negative_energy, solved_tracers, tuple(hydrometeors), tuple(hydrometeor_slots), delp, active=active, lower_bound=lower_bound, upper_bound=upper_bound, k_count=KM // 2)
    nonvapor_tracers = region("heat_tracer_borrow_all", _borrow_negative_energy_many, nonvapor_tracers, delp, k_count=KM)
    solved_tracers, ten_t, rtg, qtend_nonvapor, dtsfc, dqsfc = region(
        "heat_tracer_tendencies", _heat_tracer_tendencies, solved_heat, vapor, nonvapor_tracers, t1, q1,
        rtg, prsl, heat, evap, dt2, ldiag3d and not gen_tend)
    if ldiag3d and (not gen_tend):
        dtend = dtend.clone()
        add_diagnostic_(dtend, ten_t * dt2, slots[0])
        add_diagnostic_(dtend, (solved_tracers[:, :, NTQV - 1] - q1[:, :, NTQV - 1]), slots[3 + NTQV - 1])
        for n in range(1, ntrac1):
            add_diagnostic_(dtend, qtend_nonvapor[:, :, n - 1] * dt2, slots[3 + n])
    return (rtg, ten_t, dtend, dtsfc, dqsfc, solved_heat, solved_tracers)


def _heat_tracer_tendencies(solved_heat, vapor, nonvapor_tracers, t1, q1, rtg, prsl, heat, evap, dt2, diagnostics):
    ntrac1 = q1.shape[2] - 1
    rdt = 1.0 / dt2
    solved_tracers = torch.cat((vapor[:, :, None], nonvapor_tracers), dim=2)
    ten_t = (solved_heat - t1) * rdt
    rtg_qv = rtg[:, :, NTQV - 1] + (solved_tracers[:, :, NTQV - 1] - q1[:, :, NTQV - 1]) * rdt
    qtend_nonvapor = (solved_tracers[:, :, 1:ntrac1] - q1[:, :, 1:ntrac1]) * rdt
    # Vapor (slot NTQV = 1) and the other diffused tracers are updated; the TKE slots pass through.
    rtg = torch.cat((rtg_qv[:, :, None], rtg[:, :, 1:ntrac1] + qtend_nonvapor, rtg[:, :, ntrac1:]), dim=2)
    rho_a = prsl[:, 0] / (RD * t1[:, 0] * (1.0 + FV * torch.clamp_min(q1[:, 0, NTQV - 1], QMIN)))
    dtsfc = rho_a * CP * heat
    dqsfc = rho_a * HVAP * evap
    # The non-vapor tendencies are an output only for the dtend diagnostics: as a region output they are stored and
    # read back by the rtg join; otherwise the join forms them.
    return solved_tracers, ten_t, rtg, qtend_nonvapor if diagnostics else None, dtsfc, dqsfc

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

