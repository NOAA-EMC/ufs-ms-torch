# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment F: TKE and convective-available entrainment rates."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch import recurrence_chunks as chunks
from sa_sas_deep_convection_torch.compile import (
    masked_sum_chunk_size, run_compile_island, run_recurrence_chunk,
    ordered_chunks_enabled,
)
from sa_sas_deep_convection_torch.constants import KM, NTK
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real

CHECKPOINT_FIELDS = ("tkemean", "xlamue", "xlamud", "xlamd", "xlamdd", "cnvflg", "ca_deep")


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 979-1134 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    cnvflg = state.require("cnvflg")
    kb = state.require("kb")
    kbcon = state.require("kbcon")
    kmax = state.require("kmax")
    zo = state.require("zo")
    zi = state.require("zi")
    tkeh = state.require("tkeh")
    ncol = cnvflg.shape[0]
    require_real("cnvflg", cnvflg, (ncol,))
    require_real("kb", kb, (ncol,))
    require_real("kbcon", kbcon, (ncol,))
    require_real("kmax", kmax, (ncol,))
    require_real("zo", zo, (ncol, KM))
    require_real("zi", zi, (ncol, KM))
    require_real("tkeh", tkeh, (ncol, KM))

    hwrf_samfdeep = bool(state.config.get("hwrf_samfdeep", False))
    ntk = int(state.config.get("ntk", NTK))
    (
        sumx,
        tkemean,
        clamt,
        xlamdet,
        xlamddt,
        cxlamet,
        cxlamdt,
        xlamue,
        xlamud,
    ) = _recurrence_coordinator(
        "f_entrainment_rates",
        "entrainment_rates",
        _entrainment_rates_block,
        cnvflg,
        kb,
        kbcon,
        kmax,
        zo,
        zi,
        tkeh,
        state.require("xlamue").clone(),
        state.require("xlamud").clone(),
        state.require("clam"),
        state.require("xlamde"),
        state.require("xlamdd"),
        state.require("cxlame"),
        state.require("cxlamd"),
        state.require("crtlame"),
        state.require("ca_deep"),
        hwrf_samfdeep,
        ntk,
        bool(state.config.get("do_ca", False)),
        bool(state.config.get("ca_entr", False)),
        float(state.config.get("nthresh", 0.0)),
    )

    return state.with_updates(
        {
            "sumx": sumx,
            "tkemean": tkemean,
            "clamt": clamt,
            "xlamdet": xlamdet,
            "xlamddt": xlamddt,
            "cxlamet": cxlamet,
            "cxlamdt": cxlamdt,
            "xlamue": xlamue,
            "xlamud": xlamud,
            "xlamd": torch.zeros_like(cnvflg),
            "xlamdd": state.require("xlamdd").expand_as(cnvflg),
        }
    )


def _recurrence_coordinator(category, name, fn, *args):
    if ordered_chunks_enabled("f_ordered_chunks"):
        return fn(*args)
    return run_compile_island(category, name, fn, *args)


def _entrainment_rates_block(
    cnvflg: torch.Tensor,
    kb: torch.Tensor,
    kbcon: torch.Tensor,
    kmax: torch.Tensor,
    zo: torch.Tensor,
    zi: torch.Tensor,
    tkeh: torch.Tensor,
    xlamue: torch.Tensor,
    xlamud: torch.Tensor,
    clam: torch.Tensor,
    xlamde: torch.Tensor,
    xlamdd: torch.Tensor,
    cxlame: torch.Tensor,
    cxlamd: torch.Tensor,
    crtlame: torch.Tensor,
    ca_deep: torch.Tensor,
    hwrf_samfdeep: bool,
    ntk: int,
    do_ca: bool,
    ca_entr: bool,
    nthresh: float,
) -> tuple[torch.Tensor, ...]:
    KM = zo.shape[1]
    active = cnvflg != 0.0
    tkemean = torch.zeros_like(cnvflg)
    sumx = torch.zeros_like(cnvflg)
    if (not hwrf_samfdeep) and ntk > 0:
        size = masked_sum_chunk_size()
        levels = torch.arange(1, KM, dtype=kb.dtype, device=kb.device)
        profile_rows = chunks.pack_profiles(zo, tkeh)
        for start in range(0, KM - 1, size):
            end = min(start + size, KM - 1)
            tkemean, sumx = run_recurrence_chunk(
                "f_ordered_chunks", "tke_mean", chunks.tke_mean,
                active, kb, kbcon, levels[start], profile_rows[start:end + 1],
                tkemean, sumx,
            )
    # CA trigger/closure controls do not change F unless CA entrainment is on.
    # Canonicalize only the unused literals, preserving the physical predicate.
    ca_override = do_ca and ca_entr
    return run_recurrence_chunk(
        "f_ordered_chunks", "entrainment_closure", _entrainment_rates_from_mean,
        cnvflg, kbcon, kmax, zo, zi, xlamue, xlamud, clam, xlamde, xlamdd,
        cxlame, cxlamd, crtlame, ca_deep, hwrf_samfdeep, ntk > 0,
        ca_override, nthresh if ca_override else 0.0, tkemean, sumx,
    )


def _entrainment_rates_from_mean(
    cnvflg, kbcon, kmax, zo, zi, xlamue, xlamud, clam, xlamde, xlamdd,
    cxlame, cxlamd, crtlame, ca_deep, hwrf_samfdeep, has_tke,
    ca_override, nthresh, tkemean, sumx,
):
    KM = zo.shape[1]
    active = cnvflg != 0.0
    clamt = torch.zeros_like(cnvflg)
    if (not hwrf_samfdeep) and has_tke:
        tkemean = torch.where(active, tkemean / torch.where(active, sumx, torch.ones_like(sumx)), tkemean)
        clamd = 0.03
        tkemx = 0.65
        tkemn = 0.05
        dtke = tkemx - tkemn
        mid = clam + clamd * (1.0 - 2.0 * (tkemx - tkemean) / dtke)
        clamt = torch.where(tkemean > tkemx, clam + clamd, torch.where(tkemean < tkemn, clam - clamd, mid))
        if ca_override:
            clamt = torch.where(ca_deep > nthresh, clam - 0.03, clam)
        xlamdet = torch.where(active, xlamde.expand_as(cnvflg), torch.zeros_like(cnvflg))
        xlamddt = torch.where(active, xlamdd.expand_as(cnvflg), torch.zeros_like(cnvflg))
        cxlamet = torch.where(active, cxlame.expand_as(cnvflg), torch.zeros_like(cnvflg))
        cxlamdt = torch.where(active, cxlamd.expand_as(cnvflg), torch.zeros_like(cnvflg))
        scale = torch.minimum(1.0 + tkemean / 2.0, torch.full_like(tkemean, 15.0))
        high_tke = active & (tkemean > 2.0)
        clamt = torch.where(high_tke, scale * clam, clamt)
        xlamdet = torch.where(high_tke, scale * xlamdet, xlamdet)
        xlamddt = torch.where(high_tke, scale * xlamddt, xlamddt)
        cxlamet = torch.where(high_tke, scale * cxlamet, cxlamet)
        cxlamdt = torch.where(high_tke, scale * cxlamdt, cxlamdt)
    else:
        if ca_override:
            clamt = torch.where(ca_deep > nthresh, clam - 0.03, clam)
        else:
            clamt = torch.where(active, clam.expand_as(cnvflg), clamt)
        xlamdet = torch.where(active, xlamde.expand_as(cnvflg), torch.zeros_like(cnvflg))
        xlamddt = torch.where(active, xlamdd.expand_as(cnvflg), torch.zeros_like(cnvflg))
        cxlamet = torch.where(active, cxlame.expand_as(cnvflg), torch.zeros_like(cnvflg))
        cxlamdt = torch.where(active, cxlamd.expand_as(cnvflg), torch.zeros_like(cnvflg))

    # Every entrainment level is independent after the source-ordered TKE sum.
    levels = torch.arange(1, KM + 1, dtype=kbcon.dtype, device=kbcon.device)[None, :]
    interior = levels < float(KM)
    layer = active[:, None] & interior
    # The wrapped final neighbour is never selected. Preserve that boundary
    # directly and use a finite operand where the source does no division.
    dz = torch.roll(zo, shifts=-1, dims=1) - zo
    denominator = zi + dz
    ratio = clamt[:, None] / torch.where(layer, denominator, torch.ones_like(denominator))
    entrainment = torch.maximum(ratio, crtlame.expand_as(clamt)[:, None])
    xlamue = torch.where(layer, entrainment, xlamue)
    if hwrf_samfdeep:
        valid_base = active & (kbcon >= 1.0) & (kbcon < float(KM))
        index = torch.where(valid_base, kbcon - 1.0, torch.zeros_like(kbcon)).to(torch.int64)
        xlamx = torch.where(valid_base, xlamue.gather(1, index[:, None])[:, 0], torch.zeros_like(cnvflg))
        layer = active[:, None] & interior & (levels > 1.0) & (levels > kbcon[:, None]) & (levels < kmax[:, None])
        xlamue = torch.where(layer, xlamx[:, None], xlamue)
        down = xlamx[:, None]
    else:
        down = (0.001 * clamt)[:, None]
    layer = active[:, None] & interior & (levels < kmax[:, None])
    xlamud = torch.where(layer, down, xlamud)
    return sumx, tkemean, clamt, xlamdet, xlamddt, cxlamet, cxlamdt, xlamue, xlamud


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
