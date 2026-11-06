# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment F: pre-mass-flux cloud search and seed fields."""

from __future__ import annotations

import torch

from sa_tke_edmf_torch.autodiff import safe_positive_denominator, safe_sqrt_nonnegative
from sa_tke_edmf_torch.compile import run_compile_island
from sa_tke_edmf_torch.segments.segment_g import levels_over
from sa_tke_edmf_torch.constants import IM, KM, NTRAC
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64

CHECKPOINT_FIELDS = (
    "scuflg",
    "kcld",
    "krad",
    "radmin",
    "tcko",
    "qcko",
    "tcdo",
    "qcdo",
    "xmf",
    "xlamue",
    "buou",
    "wush",
)


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    KM = state.require("q1").shape[1]
    NTRAC = state.require("q1").shape[2]
    q1 = state.require("q1")
    t1 = state.require("t1")
    u1 = state.require("u1")
    v1 = state.require("v1")
    zl = state.require("zl")
    zi = state.require("zi")
    qlx = state.require("qlx")
    pcnvflg = state.require("pcnvflg") > 0.5
    IM = q1.shape[0]
    require_float64("q1", q1, (IM, KM, NTRAC))
    require_float64("t1", t1, (IM, KM))
    require_float64("u1", u1, (IM, KM))
    require_float64("v1", v1, (IM, KM))
    require_float64("zl", zl, (IM, KM))
    require_float64("zi", zi, (IM, KM + 1))
    require_float64("qlx", qlx, (IM, KM))

    radiation_fields = ("swh", "hlw", "xmu")
    if any(n in state.fields for n in radiation_fields):
        swh, hlw, xmu = (state.require(n) for n in radiation_fields)
        require_float64("swh", swh, (IM, KM))
        require_float64("hlw", hlw, (IM, KM))
        require_float64("xmu", xmu, (IM,))
        radx = (zi[:, 1:KM] - zi[:, :KM - 1]) * (swh[:, :KM - 1] * xmu[:, None] + hlw[:, :KM - 1])
    else:
        radx = torch.zeros_like(qlx[:, :KM - 1])

    dtype = q1.dtype
    (
        scuflg,
        kcld,
        krad,
        radmin,
        tcko,
        ucko,
        vcko,
        qcko,
        tcdo,
        qcdo,
        xmf,
        xlamue,
        buou,
        wush,
    ) = run_compile_island(
        "f_cloud_seed",
        "cloud_search_seed",
        _cloud_search_seed_block,
        q1,
        t1,
        u1,
        v1,
        zl,
        zi,
        qlx,
        pcnvflg,
        radx,
    )

    return state.with_updates(
        {
            "scuflg": scuflg.to(dtype),
            "kcld": kcld.to(dtype),
            "krad": krad.to(dtype),
            "radmin": radmin,
            "tcko": tcko,
            "ucko": ucko,
            "vcko": vcko,
            "qcko": qcko,
            "tcdo": tcdo,
            "qcdo": qcdo,
            "xmf": xmf,
            "xlamue": xlamue,
            "buou": buou,
            "wush": wush,
            "radx": radx,
            "ucdo": torch.where(scuflg[:, None], u1, torch.zeros_like(u1)),
            "vcdo": torch.where(scuflg[:, None], v1, torch.zeros_like(v1)),
        }
    )


def _cloud_search_seed_block(
    q1: torch.Tensor,
    t1: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    zl: torch.Tensor,
    zi: torch.Tensor,
    qlx: torch.Tensor,
    pcnvflg: torch.Tensor,
    radx: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    KM = q1.shape[1]
    NTRAC = q1.shape[2]
    dtype = q1.dtype
    device = q1.device
    ncol = q1.shape[0]
    km1 = KM - 1
    kmscu = KM // 2
    zeros_2 = torch.zeros((ncol, KM), dtype=dtype, device=device)
    zeros_3 = torch.zeros((ncol, KM, NTRAC), dtype=dtype, device=device)
    scuflg = torch.ones((ncol,), dtype=torch.bool, device=device)
    flg = scuflg.clone()
    lcld = torch.full((ncol,), km1, dtype=torch.int64, device=device)
    kcld = torch.full((ncol,), km1, dtype=torch.int64, device=device)
    krad = torch.ones((ncol,), dtype=torch.int64, device=device)
    radmin = torch.zeros((ncol,), dtype=dtype, device=device)

    # lcld: first level at or above 2500 m; kcld: first cloudy level scanning down
    # from kmscu. argmax returns the first hit (exact integers).
    high = zl[:, :km1] >= 2500.0
    lcld = torch.where(high.any(dim=1), high.to(torch.int8).argmax(dim=1) + 1, lcld)
    scan_levels = torch.arange(1, kmscu + 1, dtype=torch.int64, device=device)
    cloud = ((scan_levels <= lcld[:, None]) & (qlx[:, :kmscu] >= 3.5e-5)).flip(1)
    kcld = torch.where(cloud.any(dim=1), kmscu - cloud.to(torch.int8).argmax(dim=1), kcld)

    scuflg = torch.where(scuflg & (kcld == km1), torch.zeros_like(scuflg), scuflg)
    # Downward scan from kcld through the contiguous cloudy layer: a level is scanned while
    # no clear level has been met above it (exact integer cumsum, top first).
    within = (scan_levels <= kcld[:, None]).flip(1)
    cloudy = (qlx[:, :kmscu] >= 3.5e-5).flip(1)
    clear_seen = torch.cumsum((within & ~cloudy).to(torch.int32), dim=1) > 0
    scanned = scuflg[:, None] & within & cloudy & ~clear_seen
    rad_down = radx[:, :kmscu].flip(1)
    masked = torch.where(scanned & ~torch.isnan(rad_down), rad_down, float("inf"))
    first_min = masked.argmin(dim=1)
    lowest = masked.gather(1, first_min[:, None]).squeeze(1)
    minimum = lowest < radmin
    radmin = torch.where(minimum, lowest, radmin)
    krad = torch.where(minimum, kmscu - first_min, krad)
    scuflg = torch.where(scuflg & (krad <= 1), torch.zeros_like(scuflg), scuflg)
    scuflg = torch.where(scuflg & (radmin >= 0.0), torch.zeros_like(scuflg), scuflg)

    tcko = torch.where(pcnvflg[:, None], t1, zeros_2)
    ucko = torch.where(pcnvflg[:, None], u1, zeros_2)
    vcko = torch.where(pcnvflg[:, None], v1, zeros_2)
    qcko = zeros_3.clone()
    qcko[:, :, : NTRAC - 1] = torch.where(pcnvflg[:, None, None], q1[:, :, : NTRAC - 1], qcko[:, :, : NTRAC - 1])
    tcdo = torch.where(scuflg[:, None], t1, zeros_2)
    qcdo = zeros_3.clone()
    qcdo[:, :, : NTRAC - 1] = torch.where(scuflg[:, None, None], q1[:, :, : NTRAC - 1], qcdo[:, :, : NTRAC - 1])

    wush_cols = []
    for k in range(1, KM // 2):
        dz = zi[:, k + 1] - zi[:, k]
        du = 0.5 * (u1[:, k - 1] - u1[:, k + 1]) / safe_positive_denominator(dz)
        dv = 0.5 * (v1[:, k - 1] - v1[:, k + 1]) / safe_positive_denominator(dz)
        wush_cols.append(0.5 * safe_sqrt_nonnegative(du * du + dv * dv))
    wush = levels_over(zeros_2, 1, wush_cols)
    return (
        scuflg,
        kcld,
        krad,
        radmin,
        tcko,
        ucko,
        vcko,
        qcko,
        tcdo,
        qcdo,
        zeros_2,
        torch.zeros((ncol, KM - 1), dtype=dtype, device=device),
        zeros_2.clone(),
        wush,
    )


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
