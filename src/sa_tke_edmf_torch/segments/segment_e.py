# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment E: mean TKE and Monin-Obukhov surface similarity outputs."""

from __future__ import annotations

import torch
from ufs_physics_common.precision import working_power

from sa_tke_edmf_torch.autodiff import safe_positive_denominator, safe_sqrt_nonnegative
from sa_tke_edmf_torch.compile import run_compile_island
from sa_tke_edmf_torch.constants import F32_0P01, F32_0P1, F32_0P4, F32_ONE_THIRD, FV, GRAV, IM, KM, QMIN, source_real
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64

CHECKPOINT_FIELDS = ("tkemean", "zol", "phims", "phihs", "scuflg")


def _mean_tke(tke: torch.Tensor, zi: torch.Tensor, kpbl: torch.Tensor) -> torch.Tensor:
    total = torch.zeros_like(kpbl, dtype=tke.dtype)
    depth = torch.zeros_like(total)
    for k in range(tke.shape[1] // 2):
        active = k + 1 < kpbl
        dz = zi[:, k + 1] - zi[:, k]
        total = total + torch.where(active, tke[:, k] * dz, torch.zeros_like(total))
        depth = depth + torch.where(active, dz, torch.zeros_like(depth))
    has_mean = (total > 0.0) & (depth > 0.0)
    denominator = torch.where(has_mean, depth, torch.ones_like(depth))
    return torch.where(has_mean, total / denominator, total)


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    KM = state.require("q1").shape[1]
    q1 = state.require("q1")
    theta = state.require("theta")
    zi = state.require("zi")
    zl = state.require("zl")
    thlvx = state.require("thlvx")
    thvx = state.require("thvx")
    u1 = state.require("u1")
    v1 = state.require("v1")
    te = state.require("te")
    tke = state.require("tke")
    hpbl = state.require("hpbl")
    kpbl = state.require("kpbl")
    pblflg = state.require("pblflg") > 0.5
    rbsoil = state.require("rbsoil")
    heat = state.require("heat")
    evap = state.require("evap")
    stress = state.require("stress")
    zorl = state.require("zorl")
    garea = state.require("garea")
    IM = q1.shape[0]
    require_float64("q1", q1, (IM, KM, q1.shape[2]))
    require_float64("theta", theta, (IM, KM))
    require_float64("zi", zi, (IM, KM + 1))
    require_float64("zl", zl, (IM, KM))
    require_float64("thlvx", thlvx, (IM, KM))
    require_float64("thvx", thvx, (IM, KM))
    require_float64("u1", u1, (IM, KM))
    require_float64("v1", v1, (IM, KM))
    require_float64("te", te, (IM, KM))
    require_float64("tke", tke, (IM, KM))
    if ("fm" in state.fields) != ("fh" in state.fields):
        raise ValueError("surface similarity requires both fm and fh")
    fm = state.fields.get("fm", torch.ones_like(rbsoil))
    fh = state.fields.get("fh", torch.ones_like(rbsoil))
    require_float64("fm", fm, (IM,))
    require_float64("fh", fh, (IM,))
    sigmaf = state.fields.get("sigmaf", torch.zeros_like(rbsoil))
    require_float64("sigmaf", sigmaf, (IM,))

    dtype = te.dtype
    tc_pbl = bool(state.config.get("tc_pbl", False))
    tte_edmf = bool(state.config.get("tte_edmf", False))
    (
        tkemean,
        zol,
        phims,
        phihs,
        phim,
        phih,
        scuflg,
        pcnvflg,
        vpert,
        vez0fun,
        gdx,
        hpbl,
        kpbl,
        rbup,
        rbdn,
        sflux,
        ustar,
        tke,
        pblflg,
    ) = run_compile_island(
        "e_surface_similarity",
        "surface_similarity",
        _surface_similarity_block,
        q1,
        theta,
        zi,
        zl,
        thlvx,
        thvx,
        u1,
        v1,
        te,
        tke,
        hpbl,
        kpbl,
        pblflg,
        rbsoil,
        heat,
        evap,
        stress,
        zorl,
        garea,
        state.require("rbup"),
        state.require("rbdn"),
        state.require("bf"),
        state.require("shr2"),
        state.require("thvs"),
        state.require("kb1"),
        state.require("thermal"),
        state.require("crb"),
        tte_edmf,
        tc_pbl,
        fm,
        fh,
        sigmaf,
        tuple(source_real(v, state.config) for v in (0.01, 0.1, 0.4, 0.33333333)),
    )
    return state.with_updates(
        {
            "tkemean": tkemean,
            "zol": zol,
            "phims": phims,
            "phihs": phihs,
            "phim": phim,
            "phih": phih,
            "scuflg": scuflg,
            "pcnvflg": pcnvflg.to(dtype),
            "vpert": vpert,
            "vez0fun": vez0fun,
            "gdx": gdx,
            "hpbl": hpbl,
            "kpbl": kpbl,
            "rbup": rbup,
            "rbdn": rbdn,
            "sflux": sflux,
            "ustar": ustar,
            "tke": tke,
            "pblflg": pblflg.to(dtype),
        }
    )


def _surface_similarity_block(
    q1: torch.Tensor,
    theta: torch.Tensor,
    zi: torch.Tensor,
    zl: torch.Tensor,
    thlvx: torch.Tensor,
    thvx: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    te: torch.Tensor,
    tke: torch.Tensor,
    hpbl: torch.Tensor,
    kpbl: torch.Tensor,
    pblflg: torch.Tensor,
    rbsoil: torch.Tensor,
    heat: torch.Tensor,
    evap: torch.Tensor,
    stress: torch.Tensor,
    zorl: torch.Tensor,
    garea: torch.Tensor,
    rbup_state: torch.Tensor,
    rbdn_state: torch.Tensor,
    bf: torch.Tensor,
    shr2: torch.Tensor,
    thvs_state: torch.Tensor,
    kb1: torch.Tensor,
    thermal_base: torch.Tensor,
    crb: torch.Tensor,
    tte_edmf: bool,
    tc_pbl: bool,
    fm: torch.Tensor,
    fh: torch.Tensor,
    sigmaf: torch.Tensor,
    real_coefficients: tuple[float, ...],
) -> tuple[torch.Tensor, ...]:
    F32_0P01, F32_0P1, F32_0P4, F32_ONE_THIRD = real_coefficients
    KM = q1.shape[1]
    dtype = te.dtype
    device = te.device
    ncol = te.shape[0]
    true_float = torch.ones((ncol,), dtype=dtype, device=device)
    one = torch.ones((), dtype=dtype, device=device)
    gravity = torch.full((), GRAV, dtype=dtype, device=device)

    # TKE uses the incoming PBL height; TTE recomputes after its energy split.
    if not tte_edmf:
        tkemean = _mean_tke(tke, zi, kpbl)

    sfcflg = rbsoil <= 0.0
    sflux = heat + evap * FV * theta[:, 0]
    zol = torch.maximum(rbsoil * fm * fm / fh, torch.full((ncol,), -100.0, dtype=dtype, device=device))
    zol = torch.where(sfcflg, torch.minimum(zol, torch.full_like(zol, -1.0e-8)), torch.maximum(zol, torch.full_like(zol, 1.0e-8)))
    phihs = torch.where(
        sfcflg,
        # Float64 reciprocal rounded once (no-op in float64): its backward 1/x**2 at the floor overflows float32.
        safe_sqrt_nonnegative((1.0 / safe_positive_denominator(1.0 - 16.0 * zol).double()).to(dtype)),
        1.0 + 5.0 * zol,
    )
    phims = torch.where(sfcflg, safe_sqrt_nonnegative(phihs), phihs)
    sfcfrac_hpbl = F32_0P1 * hpbl
    zol1 = zol * F32_0P1 * hpbl / safe_positive_denominator(zl[:, 0])
    phih = torch.where(
        sfcflg,
        safe_sqrt_nonnegative((1.0 / safe_positive_denominator(1.0 - 16.0 * zol1).double()).to(dtype)),
        1.0 + 5.0 * zol1,
    )
    phim = torch.where(sfcflg, safe_sqrt_nonnegative(phih), phih)

    pcnvflg = pblflg & (zol < -0.02)
    ustar = safe_sqrt_nonnegative(stress)
    wst3 = gravity / thvx[:, 0] * sflux * hpbl
    # Source ust3 uses a real power; scalar exponent 3.0 reduces to multiplies.
    ust3 = torch.pow(ustar, torch.full((), 3.0, dtype=dtype, device=device))
    scale_base = torch.where(pblflg,
        ust3 + 7.0 * F32_0P4 * wst3 * F32_0P1,
        torch.ones_like(wst3))
    wscale = working_power(scale_base, F32_ONE_THIRD)
    wscale = torch.maximum(wscale, ustar / 5.0)
    vpert = torch.where(
        pcnvflg,
        torch.maximum(
            heat / safe_positive_denominator(wscale)
            + evap / safe_positive_denominator(wscale) * FV * theta[:, 0],
            torch.zeros((ncol,), dtype=dtype, device=device),
        ),
        torch.zeros((ncol,), dtype=dtype, device=device),
    )
    thermal = torch.where(pcnvflg, thermal_base + torch.minimum(4.5 * vpert, torch.full_like(vpert, 3.0)), thermal_base)
    if tte_edmf:
        thermal = torch.where(pcnvflg, thermal_base + torch.minimum(3.0 * vpert, torch.full_like(vpert, 3.0)), thermal)

    kpbl_i = kpbl.to(torch.int64)
    rbup = torch.where(pcnvflg, rbsoil, rbup_state)
    rbdn = rbdn_state
    kpblx = kpbl_i.clone()
    flg = torch.where(pcnvflg, torch.zeros((ncol,), dtype=torch.bool, device=device), torch.ones((ncol,), dtype=torch.bool, device=device))
    for k in range(1, KM // 2):
        active = (~flg) & (k + 1 > kb1)
        rbdn = torch.where(active, rbup, rbdn)
        if not tc_pbl:
            spdk2 = torch.maximum(u1[:, k] * u1[:, k] + v1[:, k] * v1[:, k], one)
            candidate = (thlvx[:, k] - thermal) * (GRAV * zl[:, k] / thlvx[:, 0]) / spdk2
        else:
            spdk2 = (
                torch.maximum((u1[:, k] - u1[:, 0]) ** 2 + (v1[:, k] - v1[:, 0]) ** 2, one)
                + 100.0 * ustar * ustar
            )
            candidate = (thlvx[:, k] - thermal) * (GRAV * (zl[:, k] - zl[:, 0]) / thlvx[:, 0]) / spdk2
        rbup = torch.where(active, candidate, rbup)
        kpblx = torch.where(active, torch.full_like(kpblx, k + 1), kpblx)
        flg = torch.where(active, rbup > crb, flg)

    hpbl_recomputed = _interpolated_hpbl(zl, zi, kpblx, rbdn, rbup, crb)
    kpbl_recomputed = torch.where(hpbl_recomputed < _take_level(zi, kpblx), kpblx - 1, kpblx)
    hpbl = torch.where(pcnvflg, hpbl_recomputed, hpbl)
    kpbl = torch.where(pcnvflg, kpbl_recomputed.to(dtype), kpbl)
    pcnvflg = torch.where((kpbl <= 1.0) & pcnvflg, torch.zeros_like(pcnvflg), pcnvflg)
    pblflg = torch.where((kpbl <= 1.0) & pblflg, torch.zeros_like(pblflg), pblflg)

    if tte_edmf:
        gotvx = gravity / thvx
        ris = torch.maximum(zol * phihs / (phims * phims), torch.full_like(zol, -100.0))
        ri = torch.empty((ncol, KM - 1), dtype=dtype, device=device)
        tem1_k1 = gotvx[:, 0] * (thlvx[:, 0] - thvs_state) / zl[:, 0]
        tem1_k1 = 0.5 * (tem1_k1 + bf[:, 0])
        ptem1_k1 = torch.maximum(u1[:, 0] * u1[:, 0] + v1[:, 0] * v1[:, 0], one) / (zl[:, 0] * zl[:, 0])
        ptem1_k1 = 0.5 * (ptem1_k1 + shr2[:, 0])
        ri[:, 0] = torch.maximum(tem1_k1 / ptem1_k1, torch.full_like(tem1_k1, -100.0))
        if KM - 1 > 1:
            ri[:, 1:] = torch.maximum(
                0.5 * (bf[:, : KM - 2] + bf[:, 1 : KM - 1]) / (0.5 * (shr2[:, : KM - 2] + shr2[:, 1 : KM - 1])),
                torch.full((ncol, KM - 2), -100.0, dtype=dtype, device=device),
            )
        ri = torch.where(zl[:, : KM - 1] <= (F32_0P1 * hpbl)[:, None], ris[:, None], ri)
        # Each branch sees only its own sign of ri, so the unselected pole (ri = 0.5 or -1/3) has a finite backward.
        ri_neg, ri_pos = torch.where(ri < 0.0, ri, -1.0), torch.where(ri < 0.0, 1.0, ri)
        epotte = torch.where(ri < 0.0, ri_neg / (2.0 * ri_neg - 1.0), ri_pos / (1.0 + 3.0 * ri_pos))
        tke = tke.clone()
        tke[:, : KM - 1] = te[:, : KM - 1] * (1.0 - epotte)
        tke[:, KM - 1] = tke[:, KM - 2]
        tkemean = _mean_tke(tke, zi, kpbl)

    scuflg = true_float
    z0 = F32_0P01 * zorl
    vegetation = torch.clamp((sigmaf - F32_0P1) / (1.0 - F32_0P1), min=0.0, max=1.0)
    roughness = torch.clamp((z0 - F32_0P1) / (1.0 - F32_0P1), min=0.0, max=1.0)
    vez0fun = (1.0 + safe_sqrt_nonnegative(vegetation)) * (1.0 + roughness)
    gdx = safe_sqrt_nonnegative(garea)
    return (
        tkemean,
        zol,
        phims,
        phihs,
        phim,
        phih,
        scuflg,
        pcnvflg,
        vpert,
        vez0fun,
        gdx,
        hpbl,
        kpbl,
        rbup,
        rbdn,
        sflux,
        ustar,
        tke,
        pblflg,
    )


def _take_level(values: torch.Tensor, one_based_index: torch.Tensor) -> torch.Tensor:
    index = torch.clamp(one_based_index.to(torch.int64) - 1, min=0, max=values.shape[1] - 1)
    return values.gather(1, index[:, None]).squeeze(1)


def _interpolated_hpbl(
    zl: torch.Tensor,
    zi: torch.Tensor,
    kpblx: torch.Tensor,
    rbdn: torch.Tensor,
    rbup: torch.Tensor,
    crb: torch.Tensor,
) -> torch.Tensor:
    z_lower = _take_level(zl, kpblx - 1)
    z_upper = _take_level(zl, kpblx)
    crossing = (rbdn < crb) & (rbup > crb)
    denominator = torch.where(crossing, rbup - rbdn, torch.ones_like(rbup))
    rbint = torch.where(
        rbdn >= crb,
        torch.zeros_like(crb),
        torch.where(rbup <= crb, torch.ones_like(crb), (crb - rbdn) / denominator),
    )
    return torch.where(kpblx > 1, z_lower + rbint * (z_upper - z_lower), zi[:, 0])


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
