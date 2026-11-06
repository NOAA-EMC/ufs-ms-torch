# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Torch-native GFS surface-layer target kernels."""

from .cases import (
    BOOL_INPUT_FIELDS,
    CHECKPOINT_OUTPUT_FIELDS,
    INT_INPUT_FIELDS,
    REAL_INPUT_FIELDS,
    SURFACE_DIAG_OUTPUT_FIELDS,
    SURFACE_DIFF_OUTPUT_FIELDS,
    SurfaceCaseConfig,
)
from .driver import (
    config_as_dict,
    fpvs,
    fpvs_table_like,
    run_gfs_surface_layer,
    run_gfs_surface_layer_compiled,
    run_gfs_surface_layer_efficient,
    run_gfs_surface_layer_regional_compiled,
    run_sfc_diff_two_passes,
    sfc_diag_run,
    sfc_diff_run,
    stability,
    znot_m_v6,
    znot_m_v7,
    znot_t_v6,
    znot_t_v7,
)
from .runtime import PreparedSurfaceRuntime

__all__ = [
    "BOOL_INPUT_FIELDS",
    "CHECKPOINT_OUTPUT_FIELDS",
    "INT_INPUT_FIELDS",
    "REAL_INPUT_FIELDS",
    "SURFACE_DIAG_OUTPUT_FIELDS",
    "SURFACE_DIFF_OUTPUT_FIELDS",
    "SurfaceCaseConfig",
    "PreparedSurfaceRuntime",
    "config_as_dict",
    "fpvs",
    "fpvs_table_like",
    "run_gfs_surface_layer",
    "run_gfs_surface_layer_compiled",
    "run_gfs_surface_layer_efficient",
    "run_gfs_surface_layer_regional_compiled",
    "run_sfc_diff_two_passes",
    "sfc_diag_run",
    "sfc_diff_run",
    "stability",
    "znot_m_v6",
    "znot_m_v7",
    "znot_t_v6",
    "znot_t_v7",
]
