# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment Q: surface flux diagnostics and optional dissipative heating."""

from __future__ import annotations

import torch

from sa_tke_edmf_torch.constants import CP, IM, KM, NDTEND, NTRAC
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64
from sa_tke_edmf_torch.diagnostics import diagnostic_slots, add_diagnostic_

CHECKPOINT_FIELDS = ("rtg", "ten_t", "dtsfc", "dqsfc", "dtend")


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    KM = state.require("q1").shape[1]
    NTRAC = state.require("q1").shape[2]
    rtg = state.require("rtg")
    ten_t = state.require("ten_t").clone()
    dtsfc = state.require("dtsfc")
    dqsfc = state.require("dqsfc")
    dtend = state.require("dtend").clone()
    diss = state.require("diss")
    delt = state.require("delt")
    IM = rtg.shape[0]
    require_float64("rtg", rtg, (IM, KM, NTRAC))
    require_float64("ten_t", ten_t, (IM, KM))
    require_float64("dtsfc", dtsfc, (IM,))
    require_float64("dqsfc", dqsfc, (IM,))
    slots = diagnostic_slots(state)
    require_float64("diss", diss, (IM, KM - 1))

    if bool(state.config.get("dspheat", False)):
        increment = diss / CP
        ten_t[:, : KM - 1] = ten_t[:, : KM - 1] + increment
        if bool(state.config.get("ldiag3d", False)) and not bool(state.config.get("gen_tend", False)):
            padding = torch.cat((increment * delt[0], torch.zeros_like(ten_t[:, -1:])), dim=1)
            dtend = add_diagnostic_(dtend.clone(), padding, slots[0])

    return state.with_updates({"ten_t": ten_t, "dtend": dtend})


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
