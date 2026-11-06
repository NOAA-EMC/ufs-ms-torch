# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment A: GFS PBL generic pre-interstitial tracer pack."""

from __future__ import annotations

import torch
from sa_tke_edmf_torch.autodiff import static_tensor

from sa_tke_edmf_torch.constants import IM, KM, NTRAC, NTKEV, NTQV
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64
from sa_tke_edmf_torch.tracers import ozone_index, pack_indices, uses_thompson_tempo_pack

CHECKPOINT_FIELDS = ("vdftra", "tracer_index_policy", "gt0", "gq0", "gu0", "gv0")


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    NTKEV = state.require("q1").shape[2]
    """Run the satmedmf ``nvdiff == ntrac`` branches of ``GFS_PBL_generic_pre_run``.

    Thompson-or-TEMPO with TKE not last (``ntkev /= ntke``, e.g. the MPAS table):
    vdftra is packed qv qc qi qr qs qg ni nr o3 with TKE at ``ntkev = nvdiff``, and
    ``rtg_ozone_index = 9``; satmedmfvdifq then sees the packed array as ``q1``.
    Otherwise vdftra is an exact copy of qgrs and ``rtg_ozone_index = ntoz``.
    """

    KM = state.require("q1").shape[1]
    NTRAC = state.require("q1").shape[2]
    qgrs = state.require("qgrs")
    gt0 = state.require("gt0")
    gq0 = state.require("gq0")
    gu0 = state.require("gu0")
    gv0 = state.require("gv0")
    IM = qgrs.shape[0]
    require_float64("qgrs", qgrs, (IM, KM, NTRAC))
    require_float64("gt0", gt0, (IM, KM))
    require_float64("gq0", gq0, (IM, KM, NTRAC))
    require_float64("gu0", gu0, (IM, KM))
    require_float64("gv0", gv0, (IM, KM))

    updates: dict[str, torch.Tensor] = {}
    if uses_thompson_tempo_pack(state.config, NTRAC):
        source, ntke = pack_indices(state.config, NTRAC)
        vdftra = torch.zeros_like(qgrs)
        vdftra[:, :, : len(source)] = qgrs[:, :, static_tensor(source, dtype=torch.int64, device=qgrs.device)]
        vdftra[:, :, NTKEV - 1] = qgrs[:, :, ntke]
        rtg_ozone_index = float(len(source))
        updates["q1"] = vdftra
    else:
        vdftra = qgrs.clone()
        rtg_ozone_index = float(ozone_index(state.config))
    tracer_index_policy = torch.full((IM,), float(NTQV), dtype=qgrs.dtype, device=qgrs.device)
    updates.update(
        {
            "vdftra": vdftra,
            "rtg_ozone_index": torch.full((), rtg_ozone_index, dtype=qgrs.dtype, device=qgrs.device),
            "tracer_index_policy": tracer_index_policy,
            "tkeh": torch.clamp_min(vdftra[:, :, NTKEV - 1], 1.0e-9),
            "gt0": gt0,
            "gq0": gq0,
            "gu0": gu0,
            "gv0": gv0,
        }
    )
    return state.with_updates(updates)


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
