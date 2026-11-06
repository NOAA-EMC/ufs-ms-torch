# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment G: thermal mass-flux updraft plume from ``mfpbltq``."""

from __future__ import annotations

import torch

from sa_tke_edmf_torch.compile import run_compile_island
from sa_tke_edmf_torch.constants import (
    EL2ORC,
    ELOCP,
    EPS,
    EPSM1,
    F32_0P01,
    F32_0P001,
    F32_0P2,
    F32_0P4,
    F32_0P55,
    F32_0P999,
    F32_3P14,
    FV,
    GRAV,
    IM,
    KM,
    NTCW,
    NTRAC,
    QMIN,
)
from sa_tke_edmf_torch.autodiff import safe_signed_denominator, safe_sqrt_nonnegative
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64
from sa_tke_edmf_torch.thermo import fpvs_from_table, fpvs_table_like
from sa_tke_edmf_torch.constants import source_real

if NTCW != 2:
    raise ImportError(f"_plume_block concatenates the qv, ql, tracer and TKE slots; it needs NTCW == 2, got {NTCW}")

CHECKPOINT_FIELDS = ("xmf", "xlamue", "tcko", "qcko", "buou", "wush")


def levels_over(base: torch.Tensor, first: int, levels: list[torch.Tensor]) -> torch.Tensor:
    """``base`` with levels ``first, first + 1, ...`` (dim 1) replaced by ``levels``, in one slice.

    Stacking the untouched levels as per-level views ``base[:, j]`` made Inductor emit one copy
    kernel per level on CUDA (offline sm_80 codegen, x1.10242); values are unchanged.
    The levels stack level-major (each level's row contiguous) and scatter back transposed.
    """
    return run_compile_island("g_plume_recurrence", "assemble_levels", _levels_over,
                              base, first, tuple(levels))


def _levels_over(base, first, levels):
    return torch.slice_scatter(base, torch.stack(levels, dim=0).movedim(0, 1), dim=1, start=first,
                               end=first + len(levels))


def level_rows(values: torch.Tensor, count: int) -> torch.Tensor:
    """Levels ``0..count-1`` of ``[ncol, KM, ...]`` as contiguous level-major rows ``[count, ncol, ...]``:
    a recurrence reads one level per step, and a row is contiguous where a column ``[:, k]`` is strided."""
    return values[:, :count].movedim(1, 0).contiguous()


def _first_plume_thermo_recurrence(
    cnvflg: torch.Tensor,
    initial_thlu: torch.Tensor,
    initial_qtu: torch.Tensor,
    initial_buo: torch.Tensor,
    xlamue: torch.Tensor,
    xlamueq: torch.Tensor,
    zl: torch.Tensor,
    thlx: torch.Tensor,
    qtx: torch.Tensor,
    pix: torch.Tensor,
    plyr: torch.Tensor,
    thvx: torch.Tensor,
    thlu_base: torch.Tensor,
    qtu_base: torch.Tensor,
    buo_base: torch.Tensor,
    qmin: torch.Tensor,
    fpvs_table: torch.Tensor,
    c01: float = F32_0P01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    KM = zl.shape[1]
    kmpbl = KM // 2
    zl_rows, xlamue_rows, xlamueq_rows, thlx_rows, qtx_rows, pix_rows, plyr_rows, thvx_rows, thlu_base_rows, qtu_base_rows, buo_base_rows = (
        level_rows(x, kmpbl) for x in (zl, xlamue, xlamueq, thlx, qtx, pix, plyr, thvx, thlu_base, qtu_base, buo_base))
    thlu_cols = [initial_thlu]
    qtu_cols = [initial_qtu]
    buo_cols = [initial_buo]
    for j in range(1, kmpbl):
        dz = zl_rows[j] - zl_rows[j - 1]
        tem = 0.5 * xlamue_rows[j - 1] * dz
        factor = 1.0 + tem
        thlu_num_left = (1.0 - tem) * thlu_cols[-1]
        thlu_num_right = tem * (thlx_rows[j - 1] + thlx_rows[j])
        thlu_j = torch.where(
            cnvflg,
            (thlu_num_left + thlu_num_right) / factor,
            thlu_base_rows[j],
        )
        tem = 0.5 * xlamueq_rows[j - 1] * dz
        factor = 1.0 + tem
        qtu_num_left = (1.0 - tem) * qtu_cols[-1]
        qtu_num_right = tem * (qtx_rows[j - 1] + qtx_rows[j])
        qtu_j = torch.where(
            cnvflg,
            (qtu_num_left + qtu_num_right) / factor,
            qtu_base_rows[j],
        )
        tlu = thlu_j / pix_rows[j]
        es = c01 * fpvs_from_table(tlu, fpvs_table)
        qs = torch.maximum(qmin, EPS * es / (plyr_rows[j] + EPSM1 * es))
        dq = qtu_j - qs
        saturated = cnvflg & (dq > 0.0)
        # Float64 quotient rounded once (no-op in float64): the floored tlu**2 underflows float32.
        gamma = (EL2ORC * qs.double() / safe_signed_denominator(tlu).double() ** 2).to(qs.dtype)
        qlu = dq / (1.0 + gamma)
        qtu_j = torch.where(saturated, qs + qlu, qtu_j)
        tem1 = torch.where(saturated, 1.0 + FV * qs - qlu, 1.0 + FV * qtu_j)
        thup = thlu_j + pix_rows[j] * ELOCP * qlu
        thvu = torch.where(saturated, thup * tem1, thlu_j * tem1)
        thlu_cols.append(thlu_j)
        qtu_cols.append(qtu_j)
        buo_cols.append(torch.where(cnvflg, GRAV * (thvu / thvx_rows[j] - 1.0), buo_base_rows[j]))
    return levels_over(thlu_base, 0, thlu_cols), levels_over(qtu_base, 0, qtu_cols), levels_over(buo_base, 0, buo_cols)


def _wu2_recurrence(
    cnvflg: torch.Tensor,
    xlamue: torch.Tensor,
    zm: torch.Tensor,
    buo: torch.Tensor,
    wush: torch.Tensor,
    wu2_base: torch.Tensor,
) -> torch.Tensor:
    KM = zm.shape[1]
    kmpbl = KM // 2
    zm_rows, xlamue_rows, buo_rows, wush_rows, wu2_base_rows = (
        level_rows(x, kmpbl) for x in (zm, xlamue, buo, wush, wu2_base))
    bb1 = 2.0
    bb2 = 4.0
    dz0 = zm_rows[0]
    tem = 0.5 * bb1 * xlamue_rows[0] * dz0
    tem1 = bb2 * buo_rows[0] * dz0
    ptem1 = 1.0 + tem
    wu2_cols = [torch.where(cnvflg, tem1 / ptem1, wu2_base_rows[0])]
    for j in range(1, kmpbl):
        dz = zm_rows[j] - zm_rows[j - 1]
        tem = 0.25 * bb1 * (xlamue_rows[j - 1] + xlamue_rows[j]) * dz
        positive_prev_wu2 = wu2_cols[-1] > 0.0
        sqrt_prev_wu2 = torch.where(positive_prev_wu2, safe_sqrt_nonnegative(wu2_cols[-1]), torch.zeros_like(wu2_cols[-1]))
        tem1 = bb2 * buo_rows[j] - wush_rows[j] * sqrt_prev_wu2
        tem2 = tem1 * dz
        ptem = (1.0 - tem) * wu2_cols[-1]
        ptem1 = 1.0 + tem
        wu2_cols.append(torch.where(cnvflg, (ptem + tem2) / ptem1, wu2_base_rows[j]))
    return levels_over(wu2_base, 0, wu2_cols)


def _cloud_plume_recurrence(
    cnvflg: torch.Tensor,
    kpbl: torch.Tensor,
    xlamue: torch.Tensor,
    xlamueq: torch.Tensor,
    zl: torch.Tensor,
    thlx: torch.Tensor,
    qtx: torch.Tensor,
    pix: torch.Tensor,
    plyr: torch.Tensor,
    thlu_base: torch.Tensor,
    qtu_base: torch.Tensor,
    tcko_base: torch.Tensor,
    qcko_qv_base: torch.Tensor,
    qcko_ql_base: torch.Tensor,
    qmin: torch.Tensor,
    fpvs_table: torch.Tensor,
    c01: float = F32_0P01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    KM = zl.shape[1]
    kmpbl = KM // 2
    zl_rows, xlamue_rows, xlamueq_rows, thlx_rows, qtx_rows, pix_rows, plyr_rows, thlu_base_rows, qtu_base_rows, tcko_base_rows, qcko_qv_base_rows, qcko_ql_base_rows = (
        level_rows(x, kmpbl) for x in (zl, xlamue, xlamueq, thlx, qtx, pix, plyr, thlu_base, qtu_base, tcko_base, qcko_qv_base, qcko_ql_base))
    thlu_cols = [torch.where(cnvflg, thlx_rows[0], thlu_base_rows[0])]
    # Level 0 of the other fields keeps its base value (outputs below start at level 1).
    qtu_cols = [qtu_base_rows[0]]
    tcko_cols = []
    qcko_qv_cols = []
    qcko_ql_cols = []
    for j in range(1, kmpbl):
        kk = j + 1
        active = cnvflg & (kk <= kpbl)
        dz = zl_rows[j] - zl_rows[j - 1]
        tem = 0.5 * xlamue_rows[j - 1] * dz
        factor = 1.0 + tem
        thlu_j = torch.where(
            active,
            ((1.0 - tem) * thlu_cols[-1] + tem * (thlx_rows[j - 1] + thlx_rows[j])) / factor,
            thlu_base_rows[j],
        )
        tem = 0.5 * xlamueq_rows[j - 1] * dz
        factor = 1.0 + tem
        qtu_j = torch.where(
            active,
            ((1.0 - tem) * qtu_cols[-1] + tem * (qtx_rows[j - 1] + qtx_rows[j])) / factor,
            qtu_base_rows[j],
        )
        tlu = thlu_j / pix_rows[j]
        es = c01 * fpvs_from_table(tlu, fpvs_table)
        qs = torch.maximum(qmin, EPS * es / (plyr_rows[j] + EPSM1 * es))
        dq = qtu_j - qs
        saturated = active & (dq > 0.0)
        # Float64 quotient rounded once (no-op in float64): the floored tlu**2 underflows float32.
        gamma = (EL2ORC * qs.double() / safe_signed_denominator(tlu).double() ** 2).to(qs.dtype)
        qlu = dq / (1.0 + gamma)
        qtu_j = torch.where(saturated, qs + qlu, qtu_j)
        thlu_cols.append(thlu_j)
        qtu_cols.append(qtu_j)
        qcko_qv_cols.append(torch.where(saturated, qs, torch.where(active, qtu_j, qcko_qv_base_rows[j])))
        qcko_ql_cols.append(
            torch.where(saturated, qlu, torch.where(active, torch.zeros_like(qlu), qcko_ql_base_rows[j]))
        )
        tcko_cols.append(torch.where(saturated, tlu + ELOCP * qlu, torch.where(active, tlu, tcko_base_rows[j])))
    return (
        levels_over(thlu_base, 0, thlu_cols),
        levels_over(qtu_base, 1, qtu_cols[1:]),
        levels_over(tcko_base, 1, tcko_cols),
        levels_over(qcko_qv_base, 1, qcko_qv_cols),
        levels_over(qcko_ql_base, 1, qcko_ql_cols),
    )


def _momentum_plume_recurrence(
    cnvflg: torch.Tensor,
    kpbl: torch.Tensor,
    xlamuem: torch.Tensor,
    zl: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    ucko_base: torch.Tensor,
    vcko_base: torch.Tensor,
    pgcon: float = F32_0P55,
) -> tuple[torch.Tensor, torch.Tensor]:
    KM = zl.shape[1]
    kmpbl = KM // 2
    zl_rows, xlamuem_rows, u1_rows, v1_rows, ucko_base_rows, vcko_base_rows = (
        level_rows(x, kmpbl) for x in (zl, xlamuem, u1, v1, ucko_base, vcko_base))
    ucko_cols = [ucko_base_rows[0]]
    vcko_cols = [vcko_base_rows[0]]
    for j in range(1, kmpbl):
        kk = j + 1
        active = cnvflg & (kk <= kpbl)
        dz = zl_rows[j] - zl_rows[j - 1]
        tem = 0.5 * xlamuem_rows[j - 1] * dz
        factor = 1.0 + tem
        ptem = tem + pgcon
        ptem1 = tem - pgcon
        ucko_cols.append(
            torch.where(
                active,
                ((1.0 - tem) * ucko_cols[-1] + ptem * u1_rows[j] + ptem1 * u1_rows[j - 1]) / factor,
                ucko_base_rows[j],
            )
        )
        vcko_cols.append(
            torch.where(
                active,
                ((1.0 - tem) * vcko_cols[-1] + ptem * v1_rows[j] + ptem1 * v1_rows[j - 1]) / factor,
                vcko_base_rows[j],
            )
        )
    return levels_over(ucko_base, 1, ucko_cols[1:]), levels_over(vcko_base, 1, vcko_cols[1:])


def _tracer_plume_recurrence(
    cnvflg: torch.Tensor,
    kpbl: torch.Tensor,
    xlamueq: torch.Tensor,
    zl: torch.Tensor,
    q1_tracers: torch.Tensor,
    qcko_tracers_base: torch.Tensor,
) -> torch.Tensor:
    KM = zl.shape[1]
    kmpbl = KM // 2
    zl_rows, xlamueq_rows, q1_tracers_rows, qcko_tracers_base_rows = (
        level_rows(x, kmpbl) for x in (zl, xlamueq, q1_tracers, qcko_tracers_base))
    previous = qcko_tracers_base_rows[0]
    pieces = []
    for first in range(1, kmpbl, 4):
        stop = min(first + 4, kmpbl)
        piece = run_compile_island(
            "g_plume_recurrence", "tracer_levels", _tracer_plume_levels,
            cnvflg, kpbl, previous, xlamueq_rows[first - 1:stop - 1],
            zl_rows[first - 1:stop], q1_tracers_rows[first - 1:stop],
            qcko_tracers_base_rows[first:stop],
            torch.arange(first + 1, stop + 1, device=kpbl.device)[:, None],
        )
        pieces.append(piece)
        previous = piece[-1]
    if not pieces:
        return qcko_tracers_base
    return torch.slice_scatter(qcko_tracers_base, torch.cat(pieces, dim=0).movedim(0, 1),
                               dim=1, start=1, end=kmpbl)


def _tracer_plume_levels(cnvflg, kpbl, previous, xlamueq_rows, zl_rows,
                         q1_tracers_rows, base_rows, levels):
    """Bound live recurrence outputs; a 26-level fused FP64 kernel exhausted H100 driver memory."""
    tracer_cols = []
    for j in range(base_rows.shape[0]):
        kk = levels[j]
        active = cnvflg & (kk <= kpbl)
        dz = zl_rows[j + 1] - zl_rows[j]
        tem = 0.5 * xlamueq_rows[j] * dz
        factor = 1.0 + tem
        tracer_cols.append(
            torch.where(
                active[:, None],
                ((1.0 - tem[:, None]) * previous + tem[:, None] * (q1_tracers_rows[j + 1] + q1_tracers_rows[j]))
                / factor[:, None],
                base_rows[j],
            )
        )
        previous = tracer_cols[-1]
    return torch.stack(tracer_cols, dim=0)


def _plume_prepare(q1, zl, zm, thlx, thvx, hpbl, kpbl, vpert, buo,
                   tkemean, vez0fun, xlamue, cnvflg, ce_scale):
    """Tensor-only plume initialization; ordered recurrence bodies stay separate."""
    IM, KM, NTRAC = q1.shape
    dtype, device = q1.dtype, q1.device
    kmpbl = KM // 2
    wu2 = torch.zeros((IM, KM), dtype=dtype, device=device)
    thlu = torch.zeros((IM, KM), dtype=dtype, device=device)
    qtx = q1[:, :, 0] + q1[:, :, NTCW - 1]
    qtu = torch.zeros((IM, KM), dtype=dtype, device=device)
    ptem = torch.minimum(1.5 * vpert, torch.full((IM,), 3.0, dtype=dtype, device=device))
    initial_thlu = torch.where(cnvflg, thlx[:, 0] + ptem, thlu[:, 0])
    initial_qtu = torch.where(cnvflg, qtx[:, 0], qtu[:, 0])
    initial_buo = torch.where(cnvflg, GRAV * ptem / thvx[:, 0], buo[:, 0])
    ce0t = ce_scale * vez0fun
    tke_boost = torch.minimum(safe_sqrt_nonnegative(tkemean / 2.0), torch.full((IM,), 5.0, dtype=dtype, device=device)) * ce_scale
    ce0t = torch.where((tkemean > 2.0) & cnvflg, torch.maximum(ce0t, tke_boost), ce0t)
    delz = _pbl_half_delz(zl, kpbl)
    xlamax = ce0t / delz
    plume_levels = torch.arange(1, kmpbl + 1, dtype=torch.int64, device=device)
    tem = torch.maximum(hpbl[:, None] - zm[:, :kmpbl] + delz[:, None], delz[:, None])
    entrainment = ce0t[:, None] * (1.0 / (zm[:, :kmpbl] + delz[:, None]) + 1.0 / tem)
    candidate = torch.where(plume_levels[None, :] < kpbl[:, None], entrainment, xlamax[:, None])
    xlamue_next = xlamue.clone()
    xlamue_next[:, :kmpbl] = torch.where(cnvflg[:, None], candidate, xlamue[:, :kmpbl])
    return (wu2, thlu, qtu, qtx, initial_thlu, initial_qtu, initial_buo,
            ce0t, xlamue_next, xlamue_next.clone(), xlamue_next.clone())


def _plume_block(
    q1: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    zl: torch.Tensor,
    zm: torch.Tensor,
    plyr: torch.Tensor,
    pix: torch.Tensor,
    thlx: torch.Tensor,
    thvx: torch.Tensor,
    gdx: torch.Tensor,
    hpbl: torch.Tensor,
    kpbl: torch.Tensor,
    vpert: torch.Tensor,
    buo: torch.Tensor,
    wush: torch.Tensor,
    tkemean: torch.Tensor,
    vez0fun: torch.Tensor,
    xmf: torch.Tensor,
    tcko: torch.Tensor,
    qcko: torch.Tensor,
    ucko: torch.Tensor,
    vcko: torch.Tensor,
    xlamue: torch.Tensor,
    cnvflg: torch.Tensor,
    dt2: torch.Tensor,
    a1: torch.Tensor,
    qmin: torch.Tensor,
    fpvs_table: torch.Tensor,
    real_coefficients: tuple[float, ...],
) -> tuple[torch.Tensor, ...]:
    """Compose the existing ordered plume routines in one pure tensor region."""
    c01, c001, c2, F32_0P4, c55, c999, cpi = real_coefficients
    KM = q1.shape[1]
    NTRAC = q1.shape[2]
    IM = q1.shape[0]
    dtype = q1.dtype
    device = q1.device
    def region(name, fn, *args):
        return run_compile_island("g_plume_recurrence", name, fn, *args)

    (wu2, thlu, qtu, qtx, initial_thlu, initial_qtu, initial_buo,
     ce0t, xlamue, xlamueq, xlamuem) = region(
        "plume_prepare", _plume_prepare, q1, zl, zm, thlx, thvx, hpbl, kpbl,
        vpert, buo, tkemean, vez0fun, xlamue, cnvflg, F32_0P4)
    (thlu, qtu, buo) = region("first_thermo", _first_plume_thermo_recurrence, cnvflg, initial_thlu, initial_qtu, initial_buo, xlamue, xlamueq, zl, thlx, qtx, pix, plyr, thvx, thlu, qtu, buo, qmin, fpvs_table, c01)
    wu2 = region("wu2", _wu2_recurrence, cnvflg, xlamue, zm, buo, wush, wu2)
    (hpbl, kpbl, cnvflg, kpblx, kpbly) = region("update_pbl", _update_pbl_from_wu2, zm, hpbl, kpbl, cnvflg, wu2)
    (xlamue, xmf) = region("mass_flux", _post_wu2_mass_flux_block, cnvflg, kpbl, kpblx, kpbly, hpbl, zm, zl, ce0t, xlamue, xmf, wu2, gdx, dt2, a1, (c001, c2, c999, cpi))
    xlamueq = xlamue
    xlamuem = xlamue
    thlu_base = thlu
    qtu_base = qtu
    (thlu, qtu, tcko, qcko_qv, qcko_ql) = region("cloud", _cloud_plume_recurrence, cnvflg, kpbl, xlamue, xlamueq, zl, thlx, qtx, pix, plyr, thlu_base, qtu_base, tcko, qcko[:, :, 0], qcko[:, :, NTCW - 1], qmin, fpvs_table, c01)
    (ucko, vcko) = region("momentum", _momentum_plume_recurrence, cnvflg, kpbl, xlamuem, zl, u1, v1, ucko, vcko, c55)
    tracer_slice = slice(NTCW, NTRAC - 1)
    qcko_tracers = region("tracer_recurrence", _tracer_plume_recurrence, cnvflg, kpbl, xlamueq, zl, q1[:, :, tracer_slice], qcko[:, :, tracer_slice])
    # Slots qv (0), ql (NTCW - 1 == 1), tracers NTCW..NTRAC-2 and TKE (NTRAC - 1) partition the
    # tracer axis, so one concatenation assembles qcko. Two scatters gave the backward an index
    # test on a constant slot, which Inductor 2.8 CPU vector codegen emits as ``bool ? vector :
    # scalar`` (CppCompileError). Values are unchanged: this only moves data.
    qcko = torch.cat((qcko_qv[:, :, None], qcko_ql[:, :, None], qcko_tracers, qcko[:, :, NTRAC - 1 :]), dim=2)
    return (
        xmf,
        xlamue,
        tcko,
        qcko,
        ucko,
        vcko,
        buo,
        wush,
        hpbl,
        kpbl.to(dtype),
        cnvflg.to(dtype),
    )


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    KM = state.require("q1").shape[1]
    NTRAC = state.require("q1").shape[2]
    q1 = state.require("q1")
    t1 = state.require("t1")
    u1 = state.require("u1")
    v1 = state.require("v1")
    zl = state.require("zl")
    zm = state.require("zm")
    plyr = state.require("plyr")
    pix = state.require("pix")
    thlx = state.require("thlx")
    thvx = state.require("thvx")
    gdx = state.require("gdx")
    hpbl = state.require("hpbl")
    kpbl = state.require("kpbl").to(torch.int64)
    vpert = state.require("vpert")
    buo = state.require("buou").clone()
    wush = state.require("wush")
    tkemean = state.require("tkemean")
    vez0fun = state.require("vez0fun")
    xmf = state.require("xmf").clone()
    tcko = state.require("tcko").clone()
    qcko = state.require("qcko").clone()
    ucko = state.require("ucko").clone()
    vcko = state.require("vcko").clone()
    xlamue = state.require("xlamue").clone()
    cnvflg = state.require("pcnvflg") > 0.5
    IM = q1.shape[0]
    require_float64("q1", q1, (IM, KM, NTRAC))
    require_float64("t1", t1, (IM, KM))
    require_float64("u1", u1, (IM, KM))
    require_float64("v1", v1, (IM, KM))
    require_float64("zl", zl, (IM, KM))
    require_float64("zm", zm, (IM, KM))
    require_float64("plyr", plyr, (IM, KM))
    require_float64("pix", pix, (IM, KM))
    require_float64("thlx", thlx, (IM, KM))
    require_float64("thvx", thvx, (IM, KM))

    dtype = q1.dtype
    device = q1.device
    kmpbl = KM // 2
    dt2 = state.require("delt")[0]
    a1 = torch.full((), state.config.get("bl_upfr", 0.08), dtype=dtype, device=device)
    qmin = torch.full((), source_real(1e-8, state.config), dtype=dtype, device=device)
    fpvs_table = fpvs_table_like(q1)

    _out_0, _out_1, _out_2, _out_3, _out_4, _out_5, _out_6, _out_7, _out_8, _out_9, _out_10 = run_compile_island(
        "g_plume_recurrence", "plume_block", _plume_block,
        q1,
        u1,
        v1,
        zl,
        zm,
        plyr,
        pix,
        thlx,
        thvx,
        gdx,
        hpbl,
        kpbl,
        vpert,
        buo,
        wush,
        tkemean,
        vez0fun,
        xmf,
        tcko,
        qcko,
        ucko,
        vcko,
        xlamue,
        cnvflg,
        dt2,
        a1,
        qmin,
        fpvs_table,
        tuple(source_real(v, state.config) for v in (0.01, 0.001, 0.2, 0.4, 0.55, 0.999, 3.14)),
    )
    return state.with_updates({
        'xmf': _out_0,
        'xlamue': _out_1,
        'tcko': _out_2,
        'qcko': _out_3,
        'ucko': _out_4,
        'vcko': _out_5,
        'buou': _out_6,
        'wush': _out_7,
        'hpbl': _out_8,
        'kpbl': _out_9,
        'pcnvflg': _out_10,
    })


def _take_level(values: torch.Tensor, one_based_index: torch.Tensor) -> torch.Tensor:
    index = torch.clamp(one_based_index.to(torch.int64) - 1, min=0, max=values.shape[1] - 1)
    return values.gather(1, index[:, None]).squeeze(1)


def _pbl_half_delz(zl: torch.Tensor, kpbl: torch.Tensor) -> torch.Tensor:
    half = torch.maximum(torch.div(kpbl, 2, rounding_mode="trunc"), torch.ones_like(kpbl))
    return _take_level(zl, half + 1) - _take_level(zl, half)


def _post_wu2_mass_flux_block(
    cnvflg: torch.Tensor,
    kpbl: torch.Tensor,
    kpblx: torch.Tensor,
    kpbly: torch.Tensor,
    hpbl: torch.Tensor,
    zm: torch.Tensor,
    zl: torch.Tensor,
    ce0t: torch.Tensor,
    xlamue: torch.Tensor,
    xmf: torch.Tensor,
    wu2: torch.Tensor,
    gdx: torch.Tensor,
    dt2: torch.Tensor,
    a1: torch.Tensor,
    real_coefficients: tuple[float, ...] = (F32_0P001, F32_0P2, F32_0P999, F32_3P14),
) -> tuple[torch.Tensor, torch.Tensor]:
    F32_0P001, F32_0P2, F32_0P999, F32_3P14 = real_coefficients
    KM = zl.shape[1]
    dtype = xmf.dtype
    device = xmf.device
    ncol = xmf.shape[0]
    kmpbl = KM // 2
    plume_levels = torch.arange(1, kmpbl + 1, dtype=torch.int64, device=device)
    delz = _pbl_half_delz(zl, kpbl)
    xlamax = ce0t / delz
    lowered = cnvflg & (kpblx < kpbly)
    tem = torch.maximum(hpbl[:, None] - zm[:, :kmpbl] + delz[:, None], delz[:, None])
    entrainment = ce0t[:, None] * (1.0 / (zm[:, :kmpbl] + delz[:, None]) + 1.0 / tem)
    candidate = torch.where(plume_levels[None, :] < kpbl[:, None], entrainment, xlamax[:, None])
    xlamue_next = xlamue.clone()
    xlamue_next[:, :kmpbl] = torch.where(lowered[:, None], candidate, xlamue[:, :kmpbl])
    xlamue = xlamue_next

    plume_active = cnvflg[:, None] & (plume_levels[None, :] < kpbl[:, None])
    dz_plume = zl[:, 1 : kmpbl + 1] - zl[:, :kmpbl]
    zeros = torch.zeros((ncol, kmpbl), dtype=dtype, device=device)
    xlamavg = torch.where(plume_active, xlamue[:, :kmpbl] * dz_plume, zeros).cumsum(dim=1)[:, -1].clone()  # source order
    sumx = torch.where(plume_active, dz_plume, zeros).cumsum(dim=1)[:, -1].clone()
    sumx_safe = torch.where(cnvflg & (sumx != 0.0), sumx, torch.ones_like(sumx))
    xlamavg = torch.where(cnvflg, xlamavg / sumx_safe, xlamavg)

    positive_wu2 = wu2[:, :kmpbl] > 0.0
    sqrt_wu2 = torch.where(positive_wu2, safe_sqrt_nonnegative(wu2[:, :kmpbl]), zeros)
    xmf_next = xmf.clone()
    xmf_next[:, :kmpbl] = torch.where(plume_active, a1 * sqrt_wu2, xmf[:, :kmpbl])
    xmf = xmf_next

    xlamavg_safe = torch.where(cnvflg & (xlamavg != 0.0), xlamavg, torch.ones_like(xlamavg))
    sigma = F32_3P14 * (F32_0P2 / xlamavg_safe) * (F32_0P2 / xlamavg_safe) / (gdx * gdx)
    sigma = torch.clamp(sigma, min=F32_0P001, max=F32_0P999)
    scaldfunc = torch.where(sigma > a1, torch.clamp((1.0 - sigma) * (1.0 - sigma), min=0.0, max=1.0), torch.ones_like(sigma))
    scaled = torch.where(sigma[:, None] > a1, sigma[:, None] * xmf[:, :kmpbl] / a1, xmf[:, :kmpbl])
    xmf_next = xmf.clone()
    xmf_next[:, :kmpbl] = torch.where(plume_active, torch.minimum(scaldfunc[:, None] * scaled, dz_plume / dt2), xmf[:, :kmpbl])
    return xlamue, xmf_next


def _update_pbl_from_wu2(
    zm: torch.Tensor,
    hpbl: torch.Tensor,
    kpbl: torch.Tensor,
    cnvflg: torch.Tensor,
    wu2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    KM = zm.shape[1]
    device = zm.device
    dtype = zm.dtype
    IM = zm.shape[0]
    kmpbl = KM // 2
    kpbly = kpbl.clone()
    crossings = wu2[:, 1:kmpbl] <= 0.0
    has_crossing = crossings.any(dim=1)
    first_crossing_offset = torch.argmax(crossings.to(torch.int64), dim=1)
    default_offset = torch.full((IM,), kmpbl - 2, dtype=torch.int64, device=device)
    crossing_offset = torch.where(has_crossing, first_crossing_offset, default_offset)
    crossing_level = crossing_offset + 1
    kpblx = torch.where(cnvflg, crossing_level + 1, torch.ones((IM,), dtype=torch.int64, device=device))
    rbup = torch.where(cnvflg, wu2.gather(1, crossing_level[:, None]).squeeze(1), torch.zeros((IM,), dtype=dtype, device=device))
    rbdn = torch.where(cnvflg, wu2.gather(1, (crossing_level - 1)[:, None]).squeeze(1), torch.zeros((IM,), dtype=dtype, device=device))
    crossing = (rbdn > 0.0) & (rbup < 0.0)
    denominator = torch.where(crossing, rbdn - rbup, torch.ones_like(rbdn))
    rbint = torch.where(
        rbdn <= 0.0,
        torch.zeros_like(rbdn),
        torch.where(rbup >= 0.0, torch.ones_like(rbdn), rbdn / denominator),
    )
    hpblx = _take_level(zm, kpblx - 1) + rbint * (_take_level(zm, kpblx) - _take_level(zm, kpblx - 1))
    lowered = cnvflg & (kpblx < kpbl)
    kpbl = torch.where(lowered, kpblx, kpbl)
    hpbl = torch.where(lowered, hpblx, hpbl)
    cnvflg = torch.where(cnvflg & (kpbl <= 1), torch.zeros_like(cnvflg), cnvflg)
    return hpbl, kpbl, cnvflg, kpblx, kpbly


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
