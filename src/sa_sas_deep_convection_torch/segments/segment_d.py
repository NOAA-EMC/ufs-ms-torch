# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment D: source-level and first LFC search."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch.compile import run_compile_island
from sa_sas_deep_convection_torch.ops import stored
from sa_sas_deep_convection_torch.constants import EPS, EPSM1, GRAV, KM, NTR
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real
from sa_sas_deep_convection_torch.thermo import NXPVS, PVS_XMAX, PVS_XMIN, _fpvs_table_for_device

CHECKPOINT_FIELDS = (
    "kb",
    "kb1",
    "kbcon",
    "kbcon1",
    "cnvflg",
    "kcnv",
    "return_stage",
    "boundary_reached",
)


def _gather_column(field: torch.Tensor, one_based_index: torch.Tensor) -> torch.Tensor:
    return field.gather(1, (one_based_index.to(torch.int64) - 1).reshape(-1, 1)).reshape(-1)


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 691-814 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    zo = state.require("zo")
    sfcpbl = state.require("sfcpbl")
    heo = state.require("heo").clone()
    heso = state.require("heso").clone()
    kbm = state.require("kbm")
    kbmax = state.require("kbmax")
    kmax = state.require("kmax")
    pfld = state.require("pfld")
    to_old = state.require("to")
    qo_old = state.require("qo")
    uo_old = state.require("uo")
    vo_old = state.require("vo")
    qeso_old = state.require("qeso")
    ctro_old = state.require("ctro")
    cnvflg = state.require("cnvflg")
    ncol = zo.shape[0]
    require_real("zo", zo, (ncol, KM))
    require_real("sfcpbl", sfcpbl, (ncol,))
    require_real("heo", heo, (ncol, KM))
    require_real("heso", heso, (ncol, KM))
    require_real("pfld", pfld, (ncol, KM))
    require_real("to", to_old, (ncol, KM))
    require_real("qo", qo_old, (ncol, KM))
    require_real("uo", uo_old, (ncol, KM))
    require_real("vo", vo_old, (ncol, KM))
    require_real("qeso", qeso_old, (ncol, KM))
    require_real("ctro", ctro_old, (ncol, KM, NTR))
    require_real("cnvflg", cnvflg, (ncol,))

    (
        kb1,
        kb,
        hmax,
        to,
        qo,
        uo,
        vo,
        po,
        qeso,
        rh,
        frh,
        heo,
        heso,
        ctro,
        kbcon,
        cnvflg,
    ) = run_compile_island(
        "d_source_lfc",
        "source_lfc",
        _source_lfc_block,
        zo,
        sfcpbl,
        heo,
        heso,
        kbm,
        kbmax,
        kmax,
        pfld,
        to_old,
        qo_old,
        uo_old,
        vo_old,
        qeso_old,
        ctro_old,
        state.require("rh"),
        state.require("frh"),
        cnvflg,
        _fpvs_table_for_device(str(zo.device), zo.dtype),
        state.require("fact1"),
        state.require("fact2"),
        state.require("el2orc"),
        state.require("hvap"),
        state.require("cp"),
        bool(state.config.get("hwrf_samfdeep", False)),
    )

    return state.with_updates(
        {
            "kb1": kb1,
            "kb": kb,
            "hmax": hmax,
            "to": to,
            "qo": qo,
            "uo": uo,
            "vo": vo,
            "po": po,
            "qeso": qeso,
            "rh": rh,
            "frh": frh,
            "heo": heo,
            "heso": heso,
            "ctro": ctro,
            "kbcon": kbcon,
            "kbcon1": torch.zeros_like(kbcon),
            "cnvflg": cnvflg,
            "return_stage": torch.zeros_like(kbcon),
            "boundary_reached": torch.ones_like(kbcon),
        }
    )


def _source_lfc_block(
    zo: torch.Tensor,
    sfcpbl: torch.Tensor,
    heo: torch.Tensor,
    heso: torch.Tensor,
    kbm: torch.Tensor,
    kbmax: torch.Tensor,
    kmax: torch.Tensor,
    pfld: torch.Tensor,
    to_old: torch.Tensor,
    qo_old: torch.Tensor,
    uo_old: torch.Tensor,
    vo_old: torch.Tensor,
    qeso_old: torch.Tensor,
    ctro_old: torch.Tensor,
    rh_base: torch.Tensor,
    frh_base: torch.Tensor,
    cnvflg: torch.Tensor,
    fpvs_table: torch.Tensor,
    fact1: torch.Tensor,
    fact2: torch.Tensor,
    el2orc: torch.Tensor,
    hvap: torch.Tensor,
    cp: torch.Tensor,
    hwrf_samfdeep: bool,
) -> tuple[torch.Tensor, ...]:
    KM = zo.shape[1]
    ncol = zo.shape[0]
    # kb1 is the last level of the leading run of PBL levels (from level 2): exact levels.
    levels = torch.arange(2, KM + 1, dtype=zo.dtype, device=zo.device)
    outside = ~(zo[:, 1:KM - 1] <= sfcpbl[:, None])
    first_out = outside.to(torch.int8).argmax(dim=1)
    kb1 = torch.where(outside.any(dim=1), torch.where(first_out > 0, levels[first_out - 1], 1.0),
                      torch.full((ncol,), float(KM - 1), dtype=zo.dtype, device=zo.device))
    kb1 = torch.minimum(kb1, kbm)

    hmax = _gather_column(heo, kb1)
    # Running strict maximum: the first occurrence of the window maximum wins; NaN never wins.
    window = (levels > kb1[:, None]) & (levels <= kbm[:, None])
    values = heo[:, 1:KM]
    masked = torch.where(window & ~torch.isnan(values), values, float("-inf"))
    first_peak = masked.argmax(dim=1)
    # gather, not amax: the derivative goes to the first maximal level only, as in the loop.
    peak = masked.gather(1, first_peak[:, None]).squeeze(1)
    update = peak > hmax
    kb = torch.where(update, levels[first_peak], kb1)
    hmax = torch.where(update, peak, hmax)

    level_ids = torch.arange(1, KM, dtype=zo.dtype, device=zo.device).reshape(1, KM - 1)
    active_interface = level_ids <= (kmax[:, None] - 1.0)
    # Interface updates act on levels 1..KM-1 (level KM is never an interface): one select per
    # field over all KM levels, with the KM-1 interface values padded by one level. Fields read
    # by several kernels (here, segments E-H and the cloud model's column gather) are stored
    # once (``stored``); Inductor otherwise recomputes each select inside every consumer. rh,
    # frh and po have one consumer each and stay fused into it.
    all_levels = torch.arange(1, KM + 1, dtype=zo.dtype, device=zo.device).reshape(1, KM)
    active = (all_levels <= (kmax[:, None] - 1.0)) & (all_levels < float(KM))

    def pad(interface: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(interface, (0, 1))

    dz = 0.5 * (zo[:, 1:KM] - zo[:, : KM - 1])
    dp = 0.5 * (pfld[:, 1:KM] - pfld[:, : KM - 1])
    es = 0.01 * _fpvs_from_table(to_old[:, 1:KM], fpvs_table)
    pprime = pfld[:, 1:KM] + EPSM1 * es
    qs = EPS * es / pprime
    dqsdp = -qs / pprime
    temperature = torch.where(active_interface, to_old[:, 1:KM], 1.0)
    desdt = es * (fact1 / temperature + fact2 / (temperature * temperature))
    dqsdt = qs * pfld[:, 1:KM] * desdt / (es * pprime)
    gamma = el2orc * qeso_old[:, 1:KM] / (temperature * temperature)
    dt = (GRAV * dz + hvap * dqsdp * dp) / (cp * (1.0 + gamma))
    dq = dqsdt * dt + dqsdp * dp
    to = stored(torch.where(active, pad(to_old[:, 1:KM] + dt), to_old))
    qo = torch.where(active, pad(qo_old[:, 1:KM] + dq), qo_old)
    po = torch.where(active, pad(0.5 * (pfld[:, : KM - 1] + pfld[:, 1:KM])), torch.zeros_like(to_old))

    qeso_interface = 0.01 * _fpvs_from_table(to[:, : KM - 1], fpvs_table)
    qeso_interface = EPS * qeso_interface / (po[:, : KM - 1] + EPSM1 * qeso_interface)
    qeso_interface = torch.clamp_min(qeso_interface, 1.0e-8)
    qeso = stored(torch.where(active, pad(qeso_interface), qeso_old))
    qo = stored(torch.where(active, torch.clamp_min(qo, 1.0e-10), qo))

    rh_interface = torch.minimum(qo[:, : KM - 1] / torch.where(active_interface, qeso[:, : KM - 1], 1.0), torch.ones_like(qeso[:, : KM - 1]))
    rh = torch.where(active, pad(rh_interface), rh_base)
    frh = torch.where(active, pad(1.0 - rh_interface), frh_base)
    heo_interface = 0.5 * GRAV * (zo[:, : KM - 1] + zo[:, 1:KM]) + cp * to[:, : KM - 1] + hvap * qo[:, : KM - 1]
    heso_interface = 0.5 * GRAV * (zo[:, : KM - 1] + zo[:, 1:KM]) + cp * to[:, : KM - 1] + hvap * qeso[:, : KM - 1]
    heo = stored(torch.where(active, pad(heo_interface), heo))
    heso = stored(torch.where(active, pad(heso_interface), heso))
    uo = stored(torch.where(active, pad(0.5 * (uo_old[:, : KM - 1] + uo_old[:, 1:KM])), uo_old))
    vo = stored(torch.where(active, pad(0.5 * (vo_old[:, : KM - 1] + vo_old[:, 1:KM])), vo_old))
    if not hwrf_samfdeep:
        ctro_candidate = 0.5 * (ctro_old[:, : KM - 1, :] + ctro_old[:, 1:KM, :])
        ctro = stored(torch.where(active[:, :, None], torch.nn.functional.pad(ctro_candidate, (0, 0, 0, 1)), ctro_old))
    else:
        ctro = ctro_old.clone()

    hkb = _gather_column(heo, kb)
    # First qualifying level; argmax returns the first hit.
    lfc_levels = torch.arange(1, KM, dtype=zo.dtype, device=zo.device)
    found = (lfc_levels <= kbmax[:, None]) & (lfc_levels > kb[:, None]) & (hkb[:, None] > heso[:, :KM - 1])
    kbcon = torch.where(found.any(dim=1), lfc_levels[found.to(torch.int8).argmax(dim=1)], kmax)
    cnvflg = torch.where(kbcon == kmax, torch.zeros_like(cnvflg), cnvflg)
    return kb1, kb, hmax, to, qo, uo, vo, po, qeso, rh, frh, heo, heso, ctro, kbcon, cnvflg


def _fpvs_from_table(temperature: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    xinc = (PVS_XMAX - PVS_XMIN) / (NXPVS - 1)
    c2 = 1.0 / xinc
    c1 = 1.0 - PVS_XMIN * c2
    xj = torch.clamp(c1 + c2 * temperature, 1.0, float(NXPVS))
    jx = torch.minimum(xj.to(torch.int64), torch.full_like(xj, NXPVS - 1, dtype=torch.int64))
    return table[jx - 1] + (xj - jx.to(xj.dtype)) * (table[jx] - table[jx - 1])


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
