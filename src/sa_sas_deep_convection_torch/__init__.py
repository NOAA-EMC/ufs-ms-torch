# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""PyTorch-native SA-SAS deep convection port."""

from sa_sas_deep_convection_torch.constants import IMPLEMENTED_SEGMENTS, SEGMENT_IDS
from sa_sas_deep_convection_torch.driver import (
    SaSasDeepConvectionState,
    checkpoint_fields,
    initial_state_case01_active_maritime_deep,
    initial_state_from_scenario_case,
    initial_state_oracle_mpas_case,
    run_sa_sas_deep_convection,
    run_sa_sas_deep_convection_production,
)

__all__ = [
    "IMPLEMENTED_SEGMENTS",
    "SEGMENT_IDS",
    "SaSasDeepConvectionState",
    "checkpoint_fields",
    "initial_state_case01_active_maritime_deep",
    "initial_state_from_scenario_case",
    "initial_state_oracle_mpas_case",
    "run_sa_sas_deep_convection",
    "run_sa_sas_deep_convection_production",
]
