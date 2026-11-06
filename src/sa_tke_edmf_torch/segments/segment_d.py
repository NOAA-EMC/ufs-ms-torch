# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment D: bulk Richardson PBL-height scan."""

from __future__ import annotations

import torch
from ufs_physics_common.precision import working_power

from sa_tke_edmf_torch.autodiff import safe_positive_denominator, safe_sqrt_nonnegative
from sa_tke_edmf_torch.compile import run_compile_island
from sa_tke_edmf_torch.constants import FV, GRAV, IM, KM, NTRAC, NTQV, QMIN, source_real
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64

CHECKPOINT_FIELDS = ("hpbl", "kpbl", "rbup", "rbdn", "pblflg")


def _take_level(values: torch.Tensor, one_based_index: torch.Tensor) -> torch.Tensor:
    index = torch.clamp(one_based_index - 1, min=0, max=values.shape[1] - 1)
    return values.gather(1, index[:, None]).squeeze(1)


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    KM = state.require("q1").shape[1]
    NTRAC = state.require("q1").shape[2]
    q1 = state.require("q1")
    u1 = state.require("u1")
    v1 = state.require("v1")
    zl = state.require("zl")
    zi = state.require("zi")
    thlvx = state.require("thlvx")
    tsea = state.require("tsea")
    heat = state.require("heat")
    evap = state.require("evap")
    rbsoil = state.require("rbsoil")
    stress = state.require("stress")
    zorl = state.require("zorl")
    IM = q1.shape[0]
    require_float64("q1", q1, (IM, KM, NTRAC))
    require_float64("u1", u1, (IM, KM))
    require_float64("v1", v1, (IM, KM))
    require_float64("zl", zl, (IM, KM))
    require_float64("zi", zi, (IM, KM + 1))
    require_float64("thlvx", thlvx, (IM, KM))

    dtype = q1.dtype
    tc_pbl = bool(state.config.get("tc_pbl", False))
    hpbl, kpbl, rbup, rbdn, pblflg, thvs, shr2, rdzt, kb1, thermal, crb = run_compile_island(
        "d_pbl_scan",
        "bulk_richardson_scan",
        _pbl_scan_block,
        q1,
        u1,
        v1,
        zl,
        zi,
        thlvx,
        tsea,
        heat,
        evap,
        rbsoil,
        stress,
        zorl,
        state.require("theta"),
        tc_pbl,
        state.fields.get("u10m", u1[:, 0]),
        state.fields.get("v10m", v1[:, 0]),
        tuple(source_real(v, state.config) for v in
              (0.01, 1e-4, 1e-7, 0.16, -0.18, 0.15, 0.35, 0.1)),
    )
    return state.with_updates(
        {
            "hpbl": hpbl,
            "kpbl": kpbl.to(dtype),
            "hpblx": hpbl,
            "kpblx": kpbl.to(dtype),
            "rbup": rbup,
            "rbdn": rbdn,
            "pblflg": pblflg.to(dtype),
            "thvs": thvs,
            "shr2": shr2,
            "rdzt": rdzt,
            "kb1": kb1,
            "thermal": thermal,
            "crb": crb,
        }
    )


def _pbl_scan_block(
    q1: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    zl: torch.Tensor,
    zi: torch.Tensor,
    thlvx: torch.Tensor,
    tsea: torch.Tensor,
    heat: torch.Tensor,
    evap: torch.Tensor,
    rbsoil: torch.Tensor,
    stress: torch.Tensor,
    zorl: torch.Tensor,
    theta: torch.Tensor,
    tc_pbl: bool,
    u10m: torch.Tensor,
    v10m: torch.Tensor,
    real_coefficients: tuple[float, ...],
) -> tuple[torch.Tensor, ...]:
    c01, f0, c1em7, c16, exponent, crbmin, crbmax, slfac = real_coefficients
    KM = q1.shape[1]
    dtype = q1.dtype
    device = q1.device
    ncol = q1.shape[0]
    true = torch.ones((ncol,), dtype=torch.bool, device=device)
    false = torch.zeros((ncol,), dtype=torch.bool, device=device)
    one_i = torch.ones((ncol,), dtype=torch.int64, device=device)
    qmin = torch.full((), QMIN, dtype=dtype, device=device)

    z0 = c01 * zorl
    sfcflg = torch.where(rbsoil > 0.0, false, true)
    sflux = heat + evap * FV * theta[:, 0]
    pblflg = sfcflg & (sflux > 0.0)
    thvs = tsea * (1.0 + FV * torch.maximum(q1[:, 0, NTQV - 1], qmin))
    rdzt = 1.0 / safe_positive_denominator(zl[:, 1:] - zl[:, :-1])
    dw2 = (u1[:, :-1] - u1[:, 1:]) ** 2 + (v1[:, :-1] - v1[:, 1:]) ** 2
    shr2 = torch.maximum(dw2, torch.full_like(dw2, f0)) * rdzt * rdzt

    thermal = torch.where(pblflg, thlvx[:, 0], thvs)
    wind10 = torch.maximum(safe_sqrt_nonnegative(u10m * u10m + v10m * v10m), torch.ones((), dtype=dtype, device=device))
    robn = wind10 / safe_positive_denominator(f0 * z0)
    stable_crb = torch.clamp(c16 * working_power(c1em7 * robn, exponent), min=crbmin, max=crbmax)
    crb = torch.where(pblflg, torch.full((ncol,), 0.25, dtype=dtype, device=device), stable_crb)
    ustar = safe_sqrt_nonnegative(stress)

    kmpbl = KM // 2
    flg = false.clone()
    rbup = rbsoil.clone()
    rbdn = torch.zeros((ncol,), dtype=dtype, device=device)
    kpblx = one_i.clone()
    for k in range(kmpbl):
        active = ~flg
        rbdn = torch.where(active, rbup, rbdn)
        if not tc_pbl:
            spdk2 = torch.maximum(u1[:, k] * u1[:, k] + v1[:, k] * v1[:, k], torch.ones((), dtype=dtype, device=device))
            candidate = (thlvx[:, k] - thermal) * (GRAV * zl[:, k] / safe_positive_denominator(thlvx[:, 0])) / spdk2
        else:
            spdk2 = (
                torch.maximum(
                    (u1[:, k] - u1[:, 0]) ** 2 + (v1[:, k] - v1[:, 0]) ** 2,
                    torch.ones((), dtype=dtype, device=device),
                )
                + 100.0 * ustar * ustar
            )
            candidate = (thlvx[:, k] - thermal) * (GRAV * (zl[:, k] - zl[:, 0]) / safe_positive_denominator(thlvx[:, 0])) / spdk2
        rbup = torch.where(active, candidate, rbup)
        kpblx = torch.where(active, torch.full_like(kpblx, k + 1), kpblx)
        flg = torch.where(active, rbup > crb, flg)

    found_above_first = kpblx > 1
    hpblx = torch.where(
        found_above_first,
        _interpolated_hpbl(zl, zi, kpblx, rbdn, rbup, crb),
        zl[:, 0],
    )
    kpblx = torch.where(found_above_first & (hpblx < _take_level(zi, kpblx)), kpblx - 1, kpblx)
    kpblx = torch.where(found_above_first, kpblx, one_i)
    hpbl = hpblx
    kpbl = kpblx
    pblflg = torch.where(kpbl <= 1, false, pblflg)

    sfcpbl = slfac * hpbl
    kb1 = one_i.clone()
    flg = pblflg.clone()
    for k in range(1, kmpbl):
        below = flg & (zl[:, k] <= sfcpbl)
        kb1 = torch.where(below, torch.full_like(kb1, k + 1), kb1)
        flg = torch.where(below, flg, false)
    kb1 = torch.where(pblflg, torch.minimum(kb1, kpbl), kb1)

    flg = true.clone()
    recompute = pblflg & (kb1 > 1)
    flg = torch.where(recompute, false, flg)
    rbup = torch.where(recompute, rbsoil, rbup)
    thermal = torch.where(recompute, _take_level(thlvx, kb1), thermal)
    kpblx = torch.where(recompute, kb1, kpblx)
    hpblx = torch.where(recompute, _take_level(zl, kb1), hpblx)
    for k in range(1, kmpbl):
        active = (~flg) & (k + 1 > kb1)
        rbdn = torch.where(active, rbup, rbdn)
        if not tc_pbl:
            spdk2 = torch.maximum(u1[:, k] * u1[:, k] + v1[:, k] * v1[:, k], torch.ones((), dtype=dtype, device=device))
            candidate = (thlvx[:, k] - thermal) * (GRAV * zl[:, k] / safe_positive_denominator(thlvx[:, 0])) / spdk2
        else:
            spdk2 = (
                torch.maximum(
                    (u1[:, k] - u1[:, 0]) ** 2 + (v1[:, k] - v1[:, 0]) ** 2,
                    torch.ones((), dtype=dtype, device=device),
                )
                + 100.0 * ustar * ustar
            )
            candidate = (thlvx[:, k] - thermal) * (GRAV * (zl[:, k] - zl[:, 0]) / safe_positive_denominator(thlvx[:, 0])) / spdk2
        rbup = torch.where(active, candidate, rbup)
        kpblx = torch.where(active, torch.full_like(kpblx, k + 1), kpblx)
        flg = torch.where(active, rbup > crb, flg)

    hpblx_recomputed = _interpolated_hpbl(zl, zi, kpblx, rbdn, rbup, crb)
    kpblx_recomputed = torch.where(
        hpblx_recomputed < _take_level(zi, kpblx),
        kpblx - 1,
        kpblx,
    )
    hpbl = torch.where(recompute, hpblx_recomputed, hpbl)
    kpbl = torch.where(recompute, kpblx_recomputed, kpbl)
    return hpbl, kpbl, rbup, rbdn, pblflg, thvs, shr2, rdzt, kb1, thermal, crb


def _interpolated_hpbl(
    zl: torch.Tensor,
    zi: torch.Tensor,
    kpblx: torch.Tensor,
    rbdn: torch.Tensor,
    rbup: torch.Tensor,
    crb: torch.Tensor,
) -> torch.Tensor:
    del zi
    z_lower = _take_level(zl, kpblx - 1)
    z_upper = _take_level(zl, kpblx)
    rbint = torch.where(
        rbdn >= crb,
        torch.zeros_like(crb),
        torch.where(
            rbup <= crb,
            torch.ones_like(crb),
            # Float64 quotient rounded once (no-op in float64): float32 (a/b)/b overflows at the floor.
            ((crb - rbdn).double() / safe_positive_denominator(rbup - rbdn).double()).to(crb.dtype),
        ),
    )
    return torch.where(kpblx > 1, z_lower + rbint * (z_upper - z_lower), zl[:, 0])


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
