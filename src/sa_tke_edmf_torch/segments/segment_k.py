# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment K: mass-flux merge and mixing-length seed fields."""

from __future__ import annotations
import torch
from ufs_physics_common.precision import working_power
from sa_tke_edmf_torch.autodiff import safe_positive_denominator, safe_sqrt_nonnegative
from sa_tke_edmf_torch.compile import run_compile_island
from sa_tke_edmf_torch.constants import (
    F32_0P001,
    F32_0P1,
    F32_0P15,
    F32_0P16,
    F32_0P2,
    F32_0P3,
    F32_0P4,
    F32_0P55,
    F32_1E_M4,
    F32_2P7,
    F32_3P7,
    FV,
    GRAV,
    IM,
    KM,
    NTRAC,
    NTQV,
    QMIN,
    source_real,
)
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64

CHECKPOINT_FIELDS = ("prn", "elm", "ele", "ckz", "chz")


def _tc_mass_flux_policy(
    xmf: torch.Tensor,
    xmfd: torch.Tensor,
    zol: torch.Tensor,
    u10m: torch.Tensor,
    v10m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source policy after mfscuq and before the K mixing-length calculation."""
    xmf = torch.where((zol > -0.5)[:, None], torch.zeros_like(xmf), xmf)
    wind = safe_sqrt_nonnegative(u10m * u10m + v10m * v10m)
    factor = 1.0 - torch.clamp(wind - 20.0, min=0.0, max=10.0) / 10.0
    return (xmf * factor[:, None], xmfd * factor[:, None])


def _mixing_length_background(
    prsi,
    zi,
    hpbl,
    kpbl,
    rbsoil,
    phih,
    phim,
    gdx,
    tvx,
    pcnvflg,
    ck0,
    ch0,
    ck1,
    ch1,
    prmax,
    kinver=None,
    xkzm_h=0.1,
    xkzm_m=0.1,
    xkzm_s=F32_0P1,
    sfcfrac=F32_0P1,
    background_cap=F32_0P15,
    zvfun=None,
    surface_cap=F32_0P3,
):
    KM = zi.shape[1] - 1
    km1 = KM - 1
    kmpbl = KM // 2
    ncol = prsi.shape[0]
    levels = torch.arange(1, KM, dtype=torch.int64, device=prsi.device)
    tx1 = 1.0 / prsi[:, 0]
    ratio = gdx / 1000.0
    xkzm_hx = torch.where(gdx >= 1000.0, torch.full_like(gdx, xkzm_h), xkzm_h * ratio)
    xkzm_mx = torch.where(gdx >= 1000.0, torch.full_like(gdx, xkzm_m), xkzm_m * ratio)
    ptem = prsi[:, 1:KM] * tx1[:, None]
    tem1 = 1.0 - ptem
    tem2 = torch.minimum(torch.ones_like(ptem), torch.exp(-(tem1 * tem1 * 2.5)))
    rlmnz = torch.maximum(30.0 * tem2, torch.full_like(tem2, 5.0))
    tem2 = torch.minimum(torch.ones_like(ptem), torch.exp(-(tem1 * tem1 * 10.0)))
    xkzo = xkzm_hx[:, None] * tem2
    above_s = ptem >= xkzm_s
    previous_above = torch.cat(
        (torch.zeros((ncol, 1), dtype=torch.bool, device=prsi.device), above_s[:, :-1]),
        dim=1,
    )
    reset_tx2 = (~above_s) & previous_above
    reset_indices = torch.where(
        reset_tx2, (levels - 1)[None, :], torch.zeros_like(levels)[None, :]
    )
    latest_reset = torch.cummax(reset_indices, dim=1).values
    tx2 = 1.0 / prsi.gather(1, latest_reset)
    tem1_m = 1.0 - prsi[:, 1:KM] * tx2
    tem1_m = tem1_m * tem1_m * 5.0
    xkzmo = torch.where(
        above_s,
        xkzm_mx[:, None],
        xkzm_mx[:, None] * torch.minimum(torch.ones_like(ptem), torch.exp(-tem1_m)),
    )
    if kinver is not None:
        below = levels[None, :] < kinver[:, None]
        xkzo = torch.where(below, xkzo, torch.zeros_like(xkzo))
        xkzmo = torch.where(below, xkzmo, torch.zeros_like(xkzmo))
        rlmnz = torch.where(below, rlmnz, torch.full_like(rlmnz, 5.0))
    sfcfrac_hpbl = sfcfrac * hpbl
    tem = torch.where(
        torch.abs(phim) > 1.0e-30,
        phih / safe_positive_denominator(phim),
        torch.ones_like(phim),
    )
    tem1 = torch.maximum(zi[:, 1:KM] - sfcfrac_hpbl[:, None], torch.zeros_like(ptem))
    tem2 = tem1 / safe_positive_denominator(hpbl - sfcfrac_hpbl)[:, None]
    tem_capped = torch.nan_to_num(
        torch.minimum(tem, torch.ones_like(tem)), nan=1.0, posinf=1.0, neginf=1.0
    )
    tem_stable = torch.nan_to_num(
        torch.maximum(tem, torch.ones_like(tem)), nan=1.0, posinf=1.0, neginf=1.0
    )
    prn_candidate = torch.where(
        pcnvflg[:, None],
        tem_capped[:, None] + (1.0 - tem_capped[:, None]) * tem2,
        tem_stable[:, None],
    )
    prn_candidate = torch.maximum(
        torch.minimum(prn_candidate, prmax.expand_as(prn_candidate)),
        torch.full_like(prn_candidate, 0.25),
    )
    active = (levels[None, :] < kpbl[:, None]) & (levels[None, :] <= kmpbl)
    prn = torch.where(active, prn_candidate, torch.ones_like(ptem))
    ck_candidate = ck0 + (ck1 - ck0) * tem2
    ck_candidate = torch.maximum(
        torch.minimum(ck_candidate, ck0.expand_as(ck_candidate)),
        ck1.expand_as(ck_candidate),
    )
    ch_candidate = ch0 + (ch1 - ch0) * tem2
    ch_candidate = torch.maximum(
        torch.minimum(ch_candidate, ch0.expand_as(ch_candidate)),
        ch1.expand_as(ch_candidate),
    )
    ckz = torch.cat(
        (torch.where(active, ck_candidate, ck1.expand_as(ptem)), ck1.expand(ncol, 1)),
        dim=1,
    )
    chz = torch.cat(
        (torch.where(active, ch_candidate, ch1.expand_as(ptem)), ch1.expand(ncol, 1)),
        dim=1,
    )
    cap = (zi[:, 1:KM] > 250.0) & ((tvx[:, 1:KM] - tvx[:, :km1]) >= 0.0)
    xkzo = torch.where(cap, torch.minimum(xkzo, torch.full_like(xkzo, background_cap)), xkzo)
    xkzmo = torch.where(
        cap, torch.minimum(xkzmo, torch.full_like(xkzmo, background_cap)), xkzmo
    )
    rlmnz = torch.where(cap, torch.minimum(rlmnz, torch.full_like(rlmnz, 5.0)), rlmnz)
    if zvfun is not None:
        low_inversion = (zi[:, 1:KM] <= 250.0) & ((tvx[:, 1:KM] - tvx[:, :km1]) > 0.0)
        reduced = torch.minimum(torch.clamp_min(xkzo * zvfun[:, None], surface_cap), xkzo)
        xkzo = torch.where(low_inversion, reduced, xkzo)
        reduced = torch.minimum(torch.clamp_min(xkzmo * zvfun[:, None], surface_cap), xkzmo)
        xkzmo = torch.where(low_inversion, reduced, xkzmo)
        reduced = torch.minimum(torch.clamp_min(rlmnz * zvfun[:, None], 10.0), rlmnz)
        rlmnz = torch.where(low_inversion, reduced, rlmnz)
    # This averaging is ordered and remains a recurrence.
    smoothed = [rlmnz[:, 0]]
    for k in range(1, km1):
        smoothed.append(0.5 * (smoothed[-1] + rlmnz[:, k]))
    smoothed.append(torch.full((ncol,), 5.0, dtype=prsi.dtype, device=prsi.device))
    return prn, ckz, chz, xkzo, xkzmo, torch.stack(smoothed, dim=1)


def _parcel_levels(energy, length, active, thvx_start, dz_rows, gotvx_rows,
                   thvx_rows, shear_rows, start_rows, coefficient, *, upward):
    """At most eight ordered levels, reused for every interior parcel block."""
    for index in range(dz_rows.shape[0]):
        k = index if upward else dz_rows.shape[0] - 1 - index
        enabled = active & start_rows[k]
        tem1 = gotvx_rows[k] * ((thvx_start - thvx_rows[k]) if upward else
                                  (thvx_rows[k] - thvx_start))
        dz = dz_rows[k]
        next_e = energy + (tem1 - coefficient * safe_sqrt_nonnegative(energy) * shear_rows[k]) * dz
        length = torch.where(enabled, length + dz, length)
        stopped = enabled & (next_e < 0.0)
        delta_e = next_e - energy
        denominator = torch.where(stopped & (torch.abs(delta_e) > 1e-30), delta_e, 1.0)
        ptem = torch.where(stopped, next_e, 0.0) / denominator
        length = torch.where(stopped, torch.maximum(length - ptem * dz, torch.zeros_like(energy)), length)
        energy = torch.where(enabled, next_e, energy)
        active = active & ~stopped
    return energy, length, active


def _parcel_prepare(zl, thvx, tke, gotvx, shr2, starts, minimum_energy):
    KM = zl.shape[1]
    indices = starts[0].clamp(max=KM - 2)
    thvx_start = thvx.index_select(1, indices)
    seed = torch.maximum(
        2.0 * tke.index_select(1, indices), torch.full_like(thvx_start, minimum_energy)
    )
    # Carry-independent level terms, level-major; only the energy recurrences loop.
    zero = torch.zeros_like(seed)
    dz_all = (zl[:, 1:] - zl[:, :-1]).t().contiguous()[:, :, None]
    shear_all = safe_sqrt_nonnegative(shr2[:, : KM - 1]).t().contiguous()[:, :, None]
    # Level-major [KM, ncol, 1] rows; each level's tem1 is formed inside the loop so
    # memory stays O(ncol * starts) rather than O(ncol * starts * KM).
    gotvx2_rows = (2.0 * gotvx).t().contiguous()[:, :, None]
    thvx_rows = thvx.t().contiguous()[:, :, None]
    level_numbers = torch.arange(1, KM, device=starts.device)[:, None, None]
    start_up = level_numbers >= starts + 1
    start_dn = level_numbers <= starts + 1

    # Seed and recurrent masks share one layout; both sweeps reuse the immutable seed.
    initial_active = (starts < KM - 1).expand_as(seed).contiguous()
    return (indices, thvx_start, seed, zero, dz_all, shear_all, gotvx2_rows,
            thvx_rows, start_up, start_dn, initial_active)


def _mixing_length_parcel_chunk(
    zi, zl, thvx, tke, gotvx, shr2, tsea, stress, qv, phims, rlmnz, starts,
    real_coefficients=(F32_0P001, F32_0P4),
):
    F32_0P001, F32_0P4 = real_coefficients
    KM = zl.shape[1]
    (indices, thvx_start, seed, zero, dz_all, shear_all, gotvx2_rows,
     thvx_rows, start_up, start_dn, initial_active) = run_compile_island(
        "k_mixing_length", "parcel_prepare", _parcel_prepare,
        zl, thvx, tke, gotvx, shr2, starts, F32_0P001)
    energy, zlup = seed, zero
    active_up = initial_active
    for first in range(0, KM - 1, 8):
        stop = min(first + 8, KM - 1)
        energy, zlup, active_up = run_compile_island(
            "k_mixing_length", "parcel_levels", _parcel_levels,
            energy, zlup, active_up, thvx_start, dz_all[first:stop],
            gotvx2_rows[first + 1:stop + 1], thvx_rows[first + 1:stop + 1],
            shear_all[first:stop], start_up[first:stop], F32_0P4, upward=True,
        )
    energy = seed
    zldn = zero
    active_dn = initial_active
    for stop in range(KM - 2, 0, -8):
        first = max(0, stop - 8)
        energy, zldn, active_dn = run_compile_island(
            "k_mixing_length", "parcel_levels", _parcel_levels,
            energy, zldn, active_dn, thvx_start, dz_all[first:stop],
            gotvx2_rows[first:stop], thvx_rows[first:stop],
            shear_all[first:stop], start_dn[first + 1:stop + 1], F32_0P4, upward=False,
        )
    return run_compile_island(
        "k_mixing_length", "parcel_finish", _parcel_finish,
        zi, zl, gotvx, tsea, stress, qv, phims, rlmnz, indices, thvx_start,
        energy, zlup, zldn, active_dn, start_dn[0], zero, F32_0P4)


def _parcel_finish(zi, zl, gotvx, tsea, stress, qv, phims, rlmnz, indices,
                   thvx_start, energy, zlup, zldn, active_dn, surface_mask, zero, coefficient):
    F32_0P4 = coefficient
    if zl.shape[1] > 1:
        active = active_dn & surface_mask
        dz = zl[:, 0, None]
        surface_tv = (tsea * (1.0 + FV * qv[:, 0]))[:, None]
        tem1 = 2.0 * gotvx[:, 0, None] * (surface_tv - thvx_start)
        tem2 = (safe_sqrt_nonnegative(stress) * phims)[
            :, None
        ] / safe_positive_denominator(F32_0P4 * dz)
        tem2 = F32_0P4 * safe_sqrt_nonnegative(energy) * tem2
        next_e = energy + (tem1 - tem2) * dz
        zldn = torch.where(active, zldn + dz, zldn)
        stopped = active & (next_e < 0.0)
        delta_e = next_e - energy
        denominator = torch.where(stopped & (torch.abs(delta_e) > 1e-30), delta_e, 1.0)
        ptem = torch.where(stopped, next_e, 0.0) / denominator
        zldn = torch.where(stopped, torch.maximum(zldn - ptem * dz, zero), zldn)
        energy = torch.where(active, next_e, energy)
    tem = 0.5 * (zi.index_select(1, indices + 1) - zi.index_select(1, indices))
    tem1 = torch.minimum(tem, rlmnz.index_select(1, indices))
    ptem2 = torch.minimum(zlup, zldn)
    rlam = torch.minimum(torch.maximum(ptem2, tem1), torch.full_like(ptem2, 300.0))
    ptem2 = safe_sqrt_nonnegative(zlup * zldn)
    ele = torch.minimum(torch.maximum(ptem2, tem1), torch.full_like(ptem2, 300.0))
    elmh = torch.minimum(1.5 * ele, torch.full_like(ptem2, 1000.0))
    return (rlam, ele, elmh)


def _mixing_length_caps(
    zi, zl, zol, hpbl, gdx, rbsoil, rlam, ele_seed, elmh_seed, tc_pbl, sfc_rlm,
    real_coefficients=(F32_0P4, F32_0P2, F32_3P7, F32_2P7),
):
    F32_0P4, F32_0P2, F32_3P7, F32_2P7 = real_coefficients
    KM = zl.shape[1]
    tem = F32_0P4 * zl[:, : KM - 1]
    unstable_base = torch.where(zol < 0.0, 1.0 - 100.0 * zol, torch.ones_like(zol))
    zk_unstable = (
        tem * working_power(unstable_base, F32_0P2)[:, None]
    )
    zk_stable = tem / F32_3P7
    zk_neutral = tem / (1.0 + F32_2P7 * zol)[:, None]
    zk = torch.where(
        zol[:, None] < 0.0,
        zk_unstable,
        torch.where(zol[:, None] < 1.0, zk_neutral, zk_stable),
    )
    if tc_pbl:
        elm_candidate = torch.sqrt(1.0 / (1.0 / (zk**2) + 1.0 / (rlam**2)))
    else:
        elm_candidate = zk * rlam / (rlam + zk)
        if sfc_rlm:
            surface_layer = (rbsoil[:, None] <= 0.0) & (
                zl[:, : KM - 1]
                < torch.minimum(torch.full_like(hpbl, 100.0), hpbl * 0.05)[:, None]
            )
            elm_candidate = torch.where(surface_layer, zk, elm_candidate)
    elm_candidate = torch.cat((zk[:, :1], elm_candidate[:, 1:]), dim=1)
    dz = zi[:, 1:KM] - zi[:, : KM - 1]
    cap = torch.maximum(gdx[:, None], dz)
    elm = torch.minimum(elm_candidate, cap)
    ele = elm if tc_pbl else torch.minimum(ele_seed, cap)
    return (
        torch.cat((elm, elm[:, -1:]), dim=1),
        torch.cat((ele, ele[:, -1:]), dim=1),
        torch.cat((elmh_seed, elmh_seed[:, -1:]), dim=1),
    )


def _mixing_length_seed_block(
    prsi: torch.Tensor,
    zi: torch.Tensor,
    zl: torch.Tensor,
    thvx: torch.Tensor,
    tke: torch.Tensor,
    hpbl: torch.Tensor,
    kpbl: torch.Tensor,
    zol: torch.Tensor,
    tsea: torch.Tensor,
    stress: torch.Tensor,
    rbsoil: torch.Tensor,
    phih: torch.Tensor,
    phim: torch.Tensor,
    qv: torch.Tensor,
    tvx: torch.Tensor,
    gotvx: torch.Tensor,
    gdx: torch.Tensor,
    shr2: torch.Tensor,
    pcnvflg: torch.Tensor,
    phims: torch.Tensor,
    ck0: torch.Tensor,
    ch0: torch.Tensor,
    ck1: torch.Tensor,
    ch1: torch.Tensor,
    prmax: torch.Tensor,
    tc_pbl: bool,
    sfc_rlm: bool,
    kinver=None,
    xkzm_h=0.1,
    xkzm_m=0.1,
    xkzm_s=F32_0P1,
    sfcfrac=F32_0P1,
    background_cap=F32_0P15,
    zvfun=None,
    surface_cap=F32_0P3,
    parcel_coefficients=(F32_0P001, F32_0P4),
    caps_coefficients=(F32_0P4, F32_0P2, F32_3P7, F32_2P7),
) -> tuple[torch.Tensor, ...]:
    KM = zl.shape[1]
    km1 = KM - 1
    prn, ckz, chz, xkzo, xkzmo, rlmnz = run_compile_island(
        "k_mixing_length",
        "background",
        _mixing_length_background,
        prsi,
        zi,
        hpbl,
        kpbl,
        rbsoil,
        phih,
        phim,
        gdx,
        tvx,
        pcnvflg,
        ck0,
        ch0,
        ck1,
        ch1,
        prmax,
        kinver,
        xkzm_h,
        xkzm_m,
        xkzm_s,
        sfcfrac,
        background_cap,
        zvfun,
        surface_cap,
    )
    # Each (column, start level) parcel is independent, so all starts run in one
    # call; the per-element arithmetic and level order are unchanged.
    starts = torch.arange(km1, dtype=torch.int64, device=zl.device)[None, :]
    rlam, ele_seed, elmh_seed = run_compile_island(
        "k_mixing_length",
        "parcel_chunk",
        _mixing_length_parcel_chunk,
        zi,
        zl,
        thvx,
        tke,
        gotvx,
        shr2,
        tsea,
        stress,
        qv,
        phims,
        rlmnz,
        starts,
        parcel_coefficients,
    )
    elm, ele, elmh = run_compile_island(
        "k_mixing_length",
        "caps",
        _mixing_length_caps,
        zi,
        zl,
        zol,
        hpbl,
        gdx,
        rbsoil,
        rlam,
        ele_seed,
        elmh_seed,
        tc_pbl,
        sfc_rlm,
        caps_coefficients,
    )
    return (prn, ckz, chz, xkzo, xkzmo, rlmnz, rlam, elm, ele, elmh)


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    KM = state.require("q1").shape[1]
    NTRAC = state.require("q1").shape[2]
    q1 = state.require("q1")
    t1 = state.require("t1")
    u1 = state.require("u1")
    v1 = state.require("v1")
    prsi = state.require("prsi")
    zi = state.require("zi")
    zl = state.require("zl")
    thvx = state.require("thvx")
    qlx = state.require("qlx")
    tke = state.require("tke")
    hpbl = state.require("hpbl")
    kpbl = state.require("kpbl").to(torch.int64)
    zol = state.require("zol")
    garea = state.require("garea")
    tsea = state.require("tsea")
    stress = state.require("stress")
    rbsoil = state.require("rbsoil")
    phih = state.require("phih")
    phim = state.require("phim")
    IM = q1.shape[0]
    require_float64("q1", q1, (IM, KM, NTRAC))
    require_float64("t1", t1, (IM, KM))
    require_float64("u1", u1, (IM, KM))
    require_float64("v1", v1, (IM, KM))
    require_float64("prsi", prsi, (IM, KM + 1))
    require_float64("zi", zi, (IM, KM + 1))
    require_float64("zl", zl, (IM, KM))
    require_float64("thvx", thvx, (IM, KM))
    require_float64("qlx", qlx, (IM, KM))
    require_float64("tke", tke, (IM, KM))
    dtype = q1.dtype
    device = q1.device
    km1 = KM - 1
    kmpbl = KM // 2
    qmin = torch.full((), source_real(1e-8, state.config), dtype=dtype, device=device)
    tte_edmf = bool(state.config.get("tte_edmf", False))
    tc_pbl = bool(state.config.get("tc_pbl", False))
    xmf = state.require("xmf")
    xmfd = state.require("xmfd")
    if tc_pbl:
        xmf, xmfd = _tc_mass_flux_policy(xmf, xmfd, zol, u1[:, 0], v1[:, 0])
    sfc_rlm = bool(state.config.get("sfc_rlm", False))
    ck0 = torch.full((), source_real(0.55 if tc_pbl else 0.4, state.config), dtype=dtype, device=device)
    ch0 = ck0
    ck1 = torch.full((), source_real(0.16 if tte_edmf else 0.15, state.config), dtype=dtype, device=device)
    ch1 = ck1
    prmax = torch.full((), 6.0 if tte_edmf else 4.0, dtype=dtype, device=device)
    qv = torch.maximum(q1[:, :, NTQV - 1], qmin)
    tvx = t1 * (1.0 + FV * qv - qlx)
    gotvx = torch.div(GRAV, thvx)
    gdx = safe_sqrt_nonnegative(garea)
    rdzt = 1.0 / (zl[:, 1:] - zl[:, :-1])
    dw2 = (u1[:, :-1] - u1[:, 1:]) ** 2 + (v1[:, :-1] - v1[:, 1:]) ** 2
    shr2 = torch.maximum(dw2, torch.full_like(dw2, source_real(1e-4, state.config))) * rdzt * rdzt
    pcnvflg = state.require("pcnvflg") > 0.5
    # Bound the compiler graph at large depths; L16 can fuse the same helpers.
    seed_runner = _mixing_length_seed_block
    if KM <= 16:
        seed_runner = lambda *args: run_compile_island(
            "k_mixing_length", "seed_block", _mixing_length_seed_block, *args
        )
    prn, ckz, chz, xkzo, xkzmo, rlmnz, rlam, elm, ele, elmh = seed_runner(
        prsi,
        zi,
        zl,
        thvx,
        tke,
        hpbl,
        kpbl,
        zol,
        tsea,
        stress,
        rbsoil,
        phih,
        phim,
        qv,
        tvx,
        gotvx,
        gdx,
        shr2,
        pcnvflg,
        state.require("phims"),
        ck0,
        ch0,
        ck1,
        ch1,
        prmax,
        tc_pbl,
        sfc_rlm,
        state.fields.get("kinver"),
        state.config.get("xkzm_h", 0.1),
        state.config.get("xkzm_m", 0.1),
        state.config.get("xkzm_s", F32_0P1),
        source_real(0.1, state.config),
        source_real(0.15, state.config),
        state.fields.get("zvfun"),
        source_real(0.3, state.config),
        tuple(source_real(v, state.config) for v in (0.001, 0.4)),
        tuple(source_real(v, state.config) for v in (0.4, 0.2, 3.7, 2.7)),
    )
    return state.with_updates(
        {
            "xmf": xmf,
            "xmfd": xmfd,
            "prn": prn,
            "elm": elm,
            "ele": ele,
            "ckz": ckz,
            "chz": chz,
            "xkzo": xkzo,
            "xkzmo": xkzmo,
            "rlmnz": rlmnz,
            "rlam": rlam,
            "elmh": elmh,
            "shr2": shr2,
            "gotvx": gotvx,
        }
    )


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
