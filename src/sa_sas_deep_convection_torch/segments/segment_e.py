# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment E: pressure-depth, CIN, and RH trigger filters."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch.compile import run_compile_island
from sa_sas_deep_convection_torch.constants import KM
from sa_sas_deep_convection_torch.segments.segment_d import _gather_column
from ufs_physics_common.ordered_sum import masked_sum
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real

CHECKPOINT_FIELDS = (
    "pdot",
    "cina",
    "rhbar",
    "kbcon",
    "kbcon1",
    "cnvflg",
    "return_stage",
    "boundary_reached",
)


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 817-973 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    cnvflg = state.require("cnvflg").clone()
    kbcon = state.require("kbcon").clone()
    kb = state.require("kb").clone()
    kbm = state.require("kbm")
    kbmax = state.require("kbmax")
    kmax = state.require("kmax")
    dot = state.require("dot")
    islimsk = state.require("islimsk")
    pfld = state.require("pfld")
    zi = state.require("zi")
    hpbl = state.require("hpbl")
    heo = state.require("heo")
    heso = state.require("heso")
    rh = state.require("rh")
    zo = state.require("zo")
    ncol = cnvflg.shape[0]
    require_real("cnvflg", cnvflg, (ncol,))
    require_real("kbcon", kbcon, (ncol,))
    require_real("kb", kb, (ncol,))
    require_real("dot", dot, (ncol, KM))
    require_real("islimsk", islimsk, (ncol,))
    require_real("pfld", pfld, (ncol, KM))
    require_real("zi", zi, (ncol, KM))
    require_real("hpbl", hpbl, (ncol,))
    require_real("heo", heo, (ncol, KM))
    require_real("heso", heso, (ncol, KM))
    require_real("rh", rh, (ncol, KM))
    require_real("zo", zo, (ncol, KM))

    pdot = state.require("pdot").clone()
    pdot, cnvflg, kb, hmax, kbcon, rhbar, sumx = run_compile_island(
        "e_trigger_filters",
        "trigger_filters",
        _trigger_filters_block,
        cnvflg,
        kbcon,
        kb,
        kbm,
        kbmax,
        kmax,
        dot,
        islimsk,
        pfld,
        zi,
        hpbl,
        heo,
        heso,
        rh,
        zo,
        pdot,
        state.require("hmax"),
        state.require("w3l"),
        state.require("w3s"),
        state.require("w4l"),
        state.require("w4s"),
        state.require("ca_deep"),
        bool(state.config.get("do_ca", False)),
        bool(state.config.get("ca_trigger", False)),
        float(state.config.get("nthresh", 0.0)),
    )

    return state.with_updates(
        {
            "pdot": pdot,
            "cnvflg": cnvflg,
            "kb": kb,
            "hmax": hmax,
            "kbcon": kbcon,
            "rhbar": rhbar,
            "sumx": sumx,
            "return_stage": torch.zeros_like(cnvflg),
            "boundary_reached": torch.ones_like(cnvflg),
        }
    )


def _trigger_filters_block(
    cnvflg: torch.Tensor,
    kbcon: torch.Tensor,
    kb: torch.Tensor,
    kbm: torch.Tensor,
    kbmax: torch.Tensor,
    kmax: torch.Tensor,
    dot: torch.Tensor,
    islimsk: torch.Tensor,
    pfld: torch.Tensor,
    zi: torch.Tensor,
    hpbl: torch.Tensor,
    heo: torch.Tensor,
    heso: torch.Tensor,
    rh: torch.Tensor,
    zo: torch.Tensor,
    pdot: torch.Tensor,
    hmax_input: torch.Tensor,
    w3l: torch.Tensor,
    w3s: torch.Tensor,
    w4l: torch.Tensor,
    w4s: torch.Tensor,
    ca_deep: torch.Tensor,
    do_ca: bool,
    ca_trigger: bool,
    nthresh: float,
) -> tuple[torch.Tensor, ...]:
    KM = zo.shape[1]
    active = cnvflg != 0.0
    # Fortran returns the whole batch when no LFC survived the preceding
    # search. CA may revive individual columns only while that call lives.
    batch_continues = torch.any(active)
    pdot_candidate = 0.01 * _gather_column(dot, kbcon)
    pdot = torch.where(active, pdot_candidate, pdot)

    land = islimsk == 1.0
    w3 = torch.where(land, w3l.expand_as(pdot), w3s.expand_as(pdot))
    w4 = torch.where(land, w4l.expand_as(pdot), w4s.expand_as(pdot))
    tem_low = (pdot - w4) / (w3 - w4)
    tem_high = -(pdot + w4) / (w4 - w3)
    tem = torch.where(pdot <= w4, tem_low, torch.where(pdot >= -w4, tem_high, torch.zeros_like(pdot)))
    tem = torch.clamp(tem, -1.0, 1.0)
    ptem = 1.0 - tem
    cinpcr = 180.0 - ptem * (0.5 * (180.0 - 120.0))
    tem1 = _gather_column(pfld, kb) - _gather_column(pfld, kbcon)
    cin_reject = active & (tem1 > cinpcr) & (_gather_column(zi, kbcon) > hpbl)
    cnvflg = torch.where(cin_reject, torch.zeros_like(cnvflg), cnvflg)

    if do_ca and ca_trigger:
        cnvflg = torch.where(batch_continues & (ca_deep > nthresh), torch.ones_like(cnvflg), cnvflg)
        cnvflg = torch.where(kbcon == kmax, torch.zeros_like(cnvflg), cnvflg)
    batch_continues = batch_continues & torch.any(cnvflg != 0.0)

    active = cnvflg != 0.0
    hmax = torch.where(active, heo[:, 0], hmax_input)
    kb = torch.where(active, torch.ones_like(kb), kb)
    # Running strict maximum: the first occurrence of the window maximum wins; NaN never wins.
    levels = torch.arange(2, KM + 1, dtype=heo.dtype, device=heo.device)
    values = heo[:, 1:KM]
    window = active[:, None] & (levels <= kbm[:, None])
    masked = torch.where(window & ~torch.isnan(values), values, float("-inf"))
    first_peak = masked.argmax(dim=1)
    peak = masked.gather(1, first_peak[:, None]).squeeze(1)
    update = peak > hmax
    kb = torch.where(update, levels[first_peak], kb)
    hmax = torch.where(update, peak, hmax)

    kbcon = torch.where(active, kmax, kbcon)
    hkb = _gather_column(heo, kb)
    # First qualifying level; argmax returns the first hit.
    lfc_levels = torch.arange(1, KM, dtype=heo.dtype, device=heo.device)
    found = active[:, None] & (lfc_levels <= kbmax[:, None]) & (lfc_levels > kb[:, None]) & (hkb[:, None] > heso[:, :KM - 1])
    kbcon = torch.where(found.any(dim=1), lfc_levels[found.to(torch.int8).argmax(dim=1)], kbcon)
    cnvflg = torch.where((cnvflg != 0.0) & (kbcon == kmax), torch.zeros_like(cnvflg), cnvflg)

    if do_ca and ca_trigger:
        cnvflg = torch.where(batch_continues & (ca_deep > nthresh), torch.ones_like(cnvflg), cnvflg)
        cnvflg = torch.where(kbcon == kmax, torch.zeros_like(cnvflg), cnvflg)
    batch_continues = batch_continues & torch.any(cnvflg != 0.0)

    active = cnvflg != 0.0
    pdot = torch.where(active, 0.01 * _gather_column(dot, kbcon), pdot)
    rhbar = torch.zeros_like(cnvflg)
    sumx = torch.zeros_like(cnvflg)
    # Terms for all levels at once; the ordered sums stay sequential.
    lfc_layer = (active[:, None] & (lfc_levels >= kb[:, None]) & (lfc_levels < kbcon[:, None])).t().contiguous()
    dz_all = zo[:, 1:KM] - zo[:, :KM - 1]
    rh_terms = (rh[:, :KM - 1] * dz_all).t().contiguous()
    dz_all = dz_all.t().contiguous()
    rhbar = masked_sum(rhbar, lfc_layer, rh_terms)
    sumx = masked_sum(sumx, lfc_layer, dz_all)
    rhbar = torch.where(active, rhbar / torch.where(active, sumx, torch.ones_like(sumx)), rhbar)
    cnvflg = torch.where(active & (rhbar < 0.75), torch.zeros_like(cnvflg), cnvflg)

    if do_ca and ca_trigger:
        cnvflg = torch.where(batch_continues & (ca_deep > nthresh), torch.ones_like(cnvflg), cnvflg)
        cnvflg = torch.where(kbcon == kmax, torch.zeros_like(cnvflg), cnvflg)
    return pdot, cnvflg, kb, hmax, kbcon, rhbar, sumx


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
