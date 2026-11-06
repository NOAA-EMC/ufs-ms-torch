# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Noah-MP land-surface PyTorch profile-SCM target."""

from .compile import (
    DEFAULT_COMPILE_CATEGORIES,
    EXPERIMENTAL_FULL_COMPILE_CATEGORIES,
    MECHANISM_CERTIFIED_COMPILE_CATEGORIES,
    RATCHET_BLOCKED_COMPILE_CATEGORIES,
    clear_compile_cache,
    compile_categories,
    compile_categories_enabled,
    compile_island_status,
    normalize_compile_categories,
    set_compile_categories,
)
from .driver import (
    NoahMpFinalOInputs,
    NoahMpPreparedProductionCase,
    NoahMpPreparedCall,
    PRODUCTION_OUTPUT_FIELDS,
    PREPARED_TENSOR_COLUMNS_KEY,
    NoahMpProductionOutput,
    SMOOTH_TENSOR_OVERRIDE_KEY,
    TARGET_STATUS,
    prepare_noahmp_land_surface_case,
    prepare_noahmp_land_surface_production_case,
    prepare_noahmp_land_surface_call,
    run_noahmp_land_surface,
    run_noahmp_land_surface_driver,
    run_noahmp_land_surface_prepared_production_tuple,
    run_noahmp_land_surface_call_tuple,
    run_noahmp_land_surface_production,
    run_noahmp_land_surface_production_tuple,
    target_status,
)

SCHEME_NAME = "NOAHMP_LAND_SURFACE_MODEL"

from .runtime import PreparedNoahRuntime


__all__ = [
    "SCHEME_NAME",
    "PreparedNoahRuntime",
    "DEFAULT_COMPILE_CATEGORIES",
    "EXPERIMENTAL_FULL_COMPILE_CATEGORIES",
    "MECHANISM_CERTIFIED_COMPILE_CATEGORIES",
    "RATCHET_BLOCKED_COMPILE_CATEGORIES",
    "NoahMpFinalOInputs",
    "NoahMpPreparedProductionCase",
    "NoahMpPreparedCall",
    "NoahMpProductionOutput",
    "PRODUCTION_OUTPUT_FIELDS",
    "PREPARED_TENSOR_COLUMNS_KEY",
    "SMOOTH_TENSOR_OVERRIDE_KEY",
    "TARGET_STATUS",
    "clear_compile_cache",
    "compile_categories",
    "compile_categories_enabled",
    "compile_island_status",
    "normalize_compile_categories",
    "prepare_noahmp_land_surface_case",
    "prepare_noahmp_land_surface_production_case",
    "prepare_noahmp_land_surface_call",
    "run_noahmp_land_surface",
    "run_noahmp_land_surface_driver",
    "run_noahmp_land_surface_prepared_production_tuple",
    "run_noahmp_land_surface_call_tuple",
    "run_noahmp_land_surface_production",
    "run_noahmp_land_surface_production_tuple",
    "set_compile_categories",
    "target_status",
]
