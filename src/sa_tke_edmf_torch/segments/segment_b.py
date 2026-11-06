# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment B: satmedmfvdifq entry workspace and TKE half-level seed."""

from __future__ import annotations

import torch

from sa_tke_edmf_torch.constants import IM, KM, NTRAC, NTKE
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64

CHECKPOINT_FIELDS = ("rtg", "tkeh", "errflg")


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    NTKE = state.require("q1").shape[2]
    """Run the B boundary initialization from ``satmedmfvdifq_run``."""

    KM = state.require("q1").shape[1]
    NTRAC = state.require("q1").shape[2]
    q1 = state.require("q1")
    IM = q1.shape[0]
    require_float64("rtg", state.require("rtg"), (IM, KM, NTRAC))
    # satmedmfvdifq_run zeroes rtg on entry (oracle satmedmfvdifq.F:420); an incoming seed is discarded.
    rtg = torch.zeros((IM, KM, NTRAC), dtype=q1.dtype, device=q1.device)
    require_float64("q1", q1, (IM, KM, NTRAC))

    te = torch.clamp_min(q1[:, :, NTKE - 1], 1.0e-9)
    tke = te.clone()
    tkeh = torch.zeros((IM, KM), dtype=rtg.dtype, device=rtg.device)
    tteh = torch.zeros((IM, KM), dtype=rtg.dtype, device=rtg.device)
    if bool(state.config.get("tte_edmf", False)):
        tteh[:, : KM - 1] = 0.5 * (te[:, : KM - 1] + te[:, 1:KM])
    else:
        tkeh[:, : KM - 1] = 0.5 * (te[:, : KM - 1] + te[:, 1:KM])

    return state.with_updates(
        {
            "rtg": rtg,
            "te": te,
            "tke": tke,
            "tkeh": tkeh,
            "tteh": tteh,
            "errflg": torch.zeros((), dtype=rtg.dtype, device=rtg.device),
        }
    )


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
