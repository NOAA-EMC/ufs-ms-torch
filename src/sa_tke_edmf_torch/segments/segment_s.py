# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment S: post-interstitial tracer tendency pack."""

from __future__ import annotations

import torch
from sa_tke_edmf_torch.autodiff import static_tensor

from sa_tke_edmf_torch.constants import IM, KM, NTRAC
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64
from sa_tke_edmf_torch.tracers import pack_indices, uses_thompson_tempo_pack

CHECKPOINT_FIELDS = ("ten_q", "dvdftra")


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    """``GFS_PBL_generic_post_run`` ten_q from dvdftra (= rtg) for the satmedmf nvdiff == ntrac branches."""

    KM = state.require("q1").shape[1]
    NTRAC = state.require("q1").shape[2]
    rtg = state.require("rtg")
    IM = rtg.shape[0]
    require_float64("rtg", rtg, (IM, KM, NTRAC))
    if uses_thompson_tempo_pack(state.config, NTRAC):
        # Thompson-or-TEMPO unpack: dvdftra(1..9) -> global species slots, dvdftra(ntkev) -> ntke.
        destination, ntke = pack_indices(state.config, NTRAC)
        source = {slot: n for n, slot in enumerate(destination)} | {ntke: NTRAC - 1}
        if len(source) == NTRAC:
            # Every global slot is unpacked: one gather through the static inverse map.
            inverse = static_tensor(tuple(source[slot] for slot in range(NTRAC)), dtype=torch.int64, device=rtg.device)
            ten_q = rtg.index_select(2, inverse)
        else:
            ten_q = torch.zeros_like(rtg)
            ten_q[:, :, static_tensor(destination, dtype=torch.int64, device=rtg.device)] = rtg[:, :, : len(destination)]
            ten_q[:, :, ntke] = rtg[:, :, NTRAC - 1]
    else:
        ten_q = rtg.clone()
    dvdftra = rtg.clone()
    return state.with_updates({"ten_q": ten_q, "dvdftra": dvdftra})


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
