# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment H: entrained LFC, CIN, and first cloud-top origin."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch.compile import run_compile_island
from sa_sas_deep_convection_torch.constants import FV, GRAV, KM
from sa_sas_deep_convection_torch.segments.segment_d import _gather_column
from ufs_physics_common.ordered_sum import masked_sum
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real

CHECKPOINT_FIELDS = (
    "kbcon",
    "kbcon1",
    "ktcon",
    "ktcon1",
    "ktconn",
    "jmin",
    "cnvflg",
    "return_stage",
)


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 1349-1513 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    cnvflg = state.require("cnvflg").clone()
    kb = state.require("kb")
    kbcon = state.require("kbcon")
    kmax = state.require("kmax")
    ktconn = state.require("ktconn")
    dbyo = state.require("dbyo")
    pfld = state.require("pfld")
    qeso = state.require("qeso")
    qo = state.require("qo")
    to = state.require("to")
    zo = state.require("zo")
    pdot = state.require("pdot")
    islimsk = state.require("islimsk")
    ncol = cnvflg.shape[0]
    require_real("cnvflg", cnvflg, (ncol,))
    require_real("kb", kb, (ncol,))
    require_real("kbcon", kbcon, (ncol,))
    require_real("kmax", kmax, (ncol,))
    require_real("ktconn", ktconn, (ncol,))
    require_real("dbyo", dbyo, (ncol, KM))
    require_real("pfld", pfld, (ncol, KM))
    require_real("qeso", qeso, (ncol, KM))
    require_real("qo", qo, (ncol, KM))
    require_real("to", to, (ncol, KM))
    require_real("zo", zo, (ncol, KM))
    require_real("pdot", pdot, (ncol,))
    require_real("islimsk", islimsk, (ncol,))

    kbcon1, cina, ktcon, cnvflg = run_compile_island(
        "h_lfc_cloudtop",
        "lfc_cin_cloudtop",
        _lfc_cin_cloudtop_block,
        cnvflg,
        kb,
        kbcon,
        kmax,
        ktconn,
        dbyo,
        pfld,
        qeso,
        qo,
        to,
        zo,
        pdot,
        islimsk,
        state.require("cina").clone(),
        state.require("ca_deep"),
        state.require("el2orc"),
        state.require("cp"),
        state.require("hvap"),
        state.require("w3l"),
        state.require("w3s"),
        state.require("w4l"),
        state.require("w4s"),
        bool(state.config.get("do_ca", False)),
        bool(state.config.get("ca_trigger", False)),
        bool(state.config.get("hwrf_samfdeep", False)),
        float(state.config.get("nthresh", 0.0)),
    )

    return state.with_updates(
        {
            "kbcon1": kbcon1,
            "cina": cina,
            "ktcon": ktcon,
            "ktcon1": torch.zeros_like(ktcon),
            "cnvflg": cnvflg,
            "return_stage": torch.zeros_like(cnvflg),
        }
    )


def _apply_ca_trigger(
    cnvflg: torch.Tensor,
    kbcon: torch.Tensor,
    kmax: torch.Tensor,
    ca_deep: torch.Tensor,
    do_ca: bool,
    ca_trigger: bool,
    nthresh: float,
    batch_continues: torch.Tensor,
) -> torch.Tensor:
    if do_ca and ca_trigger:
        cnvflg = torch.where(batch_continues & (ca_deep > nthresh), torch.ones_like(cnvflg), cnvflg)
        cnvflg = torch.where(kbcon == kmax, torch.zeros_like(cnvflg), cnvflg)
    return cnvflg


def _lfc_cin_cloudtop_block(
    cnvflg: torch.Tensor,
    kb: torch.Tensor,
    kbcon: torch.Tensor,
    kmax: torch.Tensor,
    ktconn: torch.Tensor,
    dbyo: torch.Tensor,
    pfld: torch.Tensor,
    qeso: torch.Tensor,
    qo: torch.Tensor,
    to: torch.Tensor,
    zo: torch.Tensor,
    pdot: torch.Tensor,
    islimsk: torch.Tensor,
    cina: torch.Tensor,
    ca_deep: torch.Tensor,
    el2orc: torch.Tensor,
    cp: torch.Tensor,
    hvap: torch.Tensor,
    w3l: torch.Tensor,
    w3s: torch.Tensor,
    w4l: torch.Tensor,
    w4s: torch.Tensor,
    do_ca: bool,
    ca_trigger: bool,
    hwrf_samfdeep: bool,
    nthresh: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    KM = zo.shape[1]
    active = cnvflg != 0.0
    batch_continues = torch.any(active)
    # First qualifying level (exact small-integer levels); argmax returns the first hit.
    levels = torch.arange(2, KM, dtype=kmax.dtype, device=kmax.device)
    hit = active[:, None] & (levels < kmax[:, None]) & (levels >= kbcon[:, None]) & (dbyo[:, 1:KM - 1] > 0.0)
    kbcon1 = torch.where(hit.any(dim=1), levels[hit.to(torch.int8).argmax(dim=1)], kmax)
    cnvflg = torch.where((cnvflg != 0.0) & (kbcon1 == kmax), torch.zeros_like(cnvflg), cnvflg)
    active = cnvflg != 0.0
    pressure_depth = _gather_column(pfld, kbcon) - _gather_column(pfld, kbcon1)
    cnvflg = torch.where(active & (pressure_depth > 25.0), torch.zeros_like(cnvflg), cnvflg)

    cnvflg = _apply_ca_trigger(cnvflg, kbcon, kmax, ca_deep, do_ca, ca_trigger, nthresh, batch_continues)
    batch_continues = batch_continues & torch.any(cnvflg != 0.0)

    active = cnvflg != 0.0
    # Terms for all levels at once; the ordered CIN sum keeps the source association.
    cur = slice(1, KM - 1)
    inside = active[:, None] & (levels > kb[:, None]) & (levels < kbcon1[:, None])
    layer = inside.t().contiguous()
    dz1 = zo[:, 2:KM] - zo[:, cur]
    # Above kmax to is 0. Those terms never enter cina; a unit temperature keeps their derivatives finite for a
    # compiled backward, which runs every lane with zero-filled incoming gradients.
    t_cur = torch.where(inside, to[:, cur], 1.0)
    gamma = el2orc * qeso[:, cur] / (t_cur * t_cur)
    rfact = 1.0 + FV * cp * gamma * t_cur / hvap
    term1 = (dz1 * (GRAV / (cp * t_cur)) * dbyo[:, cur] / (1.0 + gamma) * rfact).t().contiguous()
    term2 = (dz1 * GRAV * FV * torch.maximum(torch.zeros_like(cina)[:, None], qeso[:, cur] - qo[:, cur])).t().contiguous()
    cina = masked_sum(cina, layer, term1, term2)

    if hwrf_samfdeep:
        cnvflg = torch.where((cnvflg != 0.0) & (cina < -120.0), torch.zeros_like(cnvflg), cnvflg)
    else:
        land = islimsk == 1.0
        w3 = torch.where(land, w3l.expand_as(cina), w3s.expand_as(cina))
        w4 = torch.where(land, w4l.expand_as(cina), w4s.expand_as(cina))
        tem_low = (pdot - w4) / (w3 - w4)
        tem_high = -(pdot + w4) / (w4 - w3)
        tem = torch.where(pdot <= w4, tem_low, torch.where(pdot >= -w4, tem_high, torch.zeros_like(pdot)))
        tem = 1.0 - torch.clamp(tem, -1.0, 1.0)
        cinacr = -120.0 - tem * (0.5 * (-120.0 - (-80.0)))
        cnvflg = torch.where((cnvflg != 0.0) & (cina < cinacr), torch.zeros_like(cnvflg), cnvflg)

    cnvflg = _apply_ca_trigger(cnvflg, kbcon, kmax, ca_deep, do_ca, ca_trigger, nthresh, batch_continues)
    batch_continues = batch_continues & torch.any(cnvflg != 0.0)

    active = cnvflg != 0.0
    hit = active[:, None] & (levels < kmax[:, None]) & (levels > kbcon1[:, None]) & (dbyo[:, 1:KM - 1] < 0.0)
    ktcon = torch.where(hit.any(dim=1), levels[hit.to(torch.int8).argmax(dim=1)], torch.ones_like(cnvflg))
    ktcon = torch.where(active & (ktcon == 1.0) & (ktconn > 1.0), ktconn, ktcon)
    depth = _gather_column(pfld, kbcon) - _gather_column(pfld, ktcon)
    cnvflg = torch.where(active & (depth < 200.0), torch.zeros_like(cnvflg), cnvflg)

    cnvflg = _apply_ca_trigger(cnvflg, kbcon, kmax, ca_deep, do_ca, ca_trigger, nthresh, batch_continues)
    return kbcon1, cina, ktcon, cnvflg


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
