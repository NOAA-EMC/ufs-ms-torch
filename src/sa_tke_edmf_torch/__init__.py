# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""PyTorch-native SA-TKE-EDMF PBL / vertical mixing port."""

from sa_tke_edmf_torch.driver import (
    IMPLEMENTED_SEGMENTS,
    SEGMENT_IDS,
    SaTkeEdmfState,
    initial_state_from_materialized_profile,
    run_sa_tke_edmf,
    run_sa_tke_edmf_production,
)

__all__ = [
    "IMPLEMENTED_SEGMENTS",
    "SEGMENT_IDS",
    "SaTkeEdmfState",
    "initial_state_from_materialized_profile",
    "run_sa_tke_edmf",
    "run_sa_tke_edmf_production",
]
