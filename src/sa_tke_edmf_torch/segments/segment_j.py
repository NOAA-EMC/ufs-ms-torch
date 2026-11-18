# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment J: ``mfscuq`` exit checkpoint state, before diffusivity setup."""

from __future__ import annotations

from sa_tke_edmf_torch.state import SaTkeEdmfState


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    return state


