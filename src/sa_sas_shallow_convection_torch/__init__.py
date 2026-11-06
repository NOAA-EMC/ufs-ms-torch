# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""PyTorch-native SA-SAS shallow-convection target API."""

from sa_sas_shallow_convection_torch.compile import (
    COMPILE_CATEGORIES,
    DEFAULT_COMPILE_CATEGORIES,
    clear_compile_cache,
    compile_categories,
    compile_categories_enabled,
    compile_categories_explicitly_set,
    compile_island_status,
    normalize_compile_categories,
    restore_compile_categories,
    set_compile_categories,
)
from sa_sas_shallow_convection_torch.constants import IMPLEMENTED_SEGMENTS, SEGMENT_IDS
from sa_sas_shallow_convection_torch.driver import (
    CASE_CONFIGS,
    PRODUCTION_FIELDS,
    checkpoint_fields,
    initial_state_case08_active_maritime_default,
    initial_state_for_case,
    initial_state_oracle_mpas_case,
    run_sa_sas_shallow_convection,
    run_sa_sas_shallow_convection_production,
)
from sa_sas_shallow_convection_torch.state import SaSasShallowConvectionState

__all__ = [
    "IMPLEMENTED_SEGMENTS",
    "SEGMENT_IDS",
    "CASE_CONFIGS",
    "PRODUCTION_FIELDS",
    "COMPILE_CATEGORIES",
    "DEFAULT_COMPILE_CATEGORIES",
    "SaSasShallowConvectionState",
    "checkpoint_fields",
    "clear_compile_cache",
    "compile_categories",
    "compile_categories_enabled",
    "compile_categories_explicitly_set",
    "compile_island_status",
    "initial_state_case08_active_maritime_default",
    "initial_state_for_case",
    "initial_state_oracle_mpas_case",
    "normalize_compile_categories",
    "restore_compile_categories",
    "run_sa_sas_shallow_convection",
    "run_sa_sas_shallow_convection_production",
    "set_compile_categories",
]
