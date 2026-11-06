# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment H: pre-``mfscuq`` downdraft entry state."""

from __future__ import annotations

import torch

from sa_tke_edmf_torch.constants import IM, KM, NTRAC
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64

CHECKPOINT_FIELDS = ("xmfd", "xlamde", "tcdo", "qcdo", "buod")


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    KM = state.require("q1").shape[1]
    NTRAC = state.require("q1").shape[2]
    tcdo = state.require("tcdo")
    qcdo = state.require("qcdo")
    IM = tcdo.shape[0]
    require_float64("tcdo", tcdo, (IM, KM))
    require_float64("qcdo", qcdo, (IM, KM, NTRAC))

    dtype = tcdo.dtype
    device = tcdo.device
    return state.with_updates(
        {
            "xmfd": torch.zeros((IM, KM), dtype=dtype, device=device),
            "xlamde": torch.zeros((IM, KM - 1), dtype=dtype, device=device),
            "tcdo": tcdo,
            "qcdo": qcdo,
            "buod": torch.zeros((IM, KM), dtype=dtype, device=device),
        }
    )


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
