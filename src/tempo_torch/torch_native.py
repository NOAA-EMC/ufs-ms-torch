# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Torch-native TEMPO backend under incremental conversion.

This module is intentionally separate from the NumPy reference segments.  The
reference path remains the accuracy oracle, while this backend converts stable
prefix segments to tensor operations that can run on GPU and participate in
autograd.
"""

from __future__ import annotations

import cmath
import math
import time
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass, replace
from hashlib import sha256
from typing import Any

import numpy as np
import torch
import torch._dynamo.config as torch_dynamo_config
from ufs_physics_common.compaction import note_overflow
from ufs_physics_common.ordered_sum import masked_sum, parallel_masked_sums
from ufs_physics_common.precision import working_power

from tempo_torch.coefficients import constants
from tempo_torch.coefficients import CHECKPOINT_FIELDS


def _complex_beta_constant_python(m1: complex, m2: complex) -> complex:
    return 2.0 * m1 / (m2 - m1) * (m2 / (m2 - m1) * cmath.log(m2 / m1) - 1.0)


_TORCH_TABLE_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_TORCH_STATIC_NDARRAY_FAST_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
# Working dtypes of the native backend: float64 (the parity dtype) and float32 (profiling).
WORKING_DTYPES = (torch.float32, torch.float64)
_TORCH_SCALAR_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_TORCH_INT_SCALAR_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_TORCH_INT_ARANGE_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_TORCH_STATIC_TENSOR_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_CompileCacheKey = tuple[Any, ...]
_RADAR_COMPILED_REDUCTION_CACHE: dict[_CompileCacheKey, Any] = {}
_RADAR_COMPILED_REDUCTION_DISABLED: set[_CompileCacheKey] = set()
_SEDIMENTATION_RECURRENCE_COMPILED_CACHE: dict[_CompileCacheKey, Any] = {}
_SEDIMENTATION_RECURRENCE_COMPILE_DISABLED: set[_CompileCacheKey] = set()
_TENSOR_COMPILE_ISLAND_CACHE: dict[_CompileCacheKey, Any] = {}
_TENSOR_COMPILE_ISLAND_DISABLED: set[_CompileCacheKey] = set()
_TENSOR_COMPILE_ISLAND_STATUS: dict[str, dict[str, Any]] = {}
_TENSOR_COMPILE_ISLAND_TIMINGS: list[dict[str, Any]] = []
_SPARSE_DIAGNOSTIC_COMPILED_CACHE: dict[_CompileCacheKey, Any] = {}
_SPARSE_DIAGNOSTIC_COMPILE_DISABLED: set[_CompileCacheKey] = set()


@dataclass
class _PreparedCompileDispatch:
    """Instance-owned direct dispatch after one signature-validated compile lookup."""

    callables: dict[tuple[Any, ...], Any]
    fast_hits: int = 0
    misses: int = 0


_ACTIVE_PREPARED_COMPILE_DISPATCH: ContextVar[_PreparedCompileDispatch | None] = ContextVar(
    "tempo_prepared_compile_dispatch",
    default=None,
)

# The frozen corpus exercises more than Dynamo's default eight guards for a
# few shared recurrence code objects.  Keep the override scoped to TEMPO's
# individual compile/prime call: process-global compiler policy remains
# untouched, while genuinely unbounded specialization still fails at 128.
TEMPO_COMPILE_RECOMPILE_LIMIT = 128

# Four bounded N-limiter mappings.  These are execution schedules over one
# algebraic implementation: the split modes change only Inductor fusion
# boundaries, while the option dictionaries are passed to the individual
# torch.compile call instead of mutating process-global Inductor state.
N_LIMITER_COMPILE_SCHEDULES: dict[str, dict[str, Any]] = {
    "baseline": {"pressure_split": False, "options": {}},
    "coordinate_descent": {
        "pressure_split": False,
        "options": {
            "coordinate_descent_tuning": True,
            "coordinate_descent_check_all_directions": True,
        },
    },
    "pressure_split": {"pressure_split": True, "options": {}},
    "pressure_split_autotune": {
        "pressure_split": True,
        "options": {
            "coordinate_descent_tuning": True,
            "coordinate_descent_check_all_directions": True,
            "max_autotune_pointwise": True,
            "triton.autotune_pointwise": True,
        },
    },
}
# Q1-selected production schedule.  The options remain scoped to the N
# limit-and-sum compile only; they do not mutate process-global Inductor state.
N_LIMITER_COMPILE_SCHEDULE = "coordinate_descent"
N_TENDENCY_LAYOUT_MODES = ("separate", "producer_packed")
# Keep producer packing opt-in until its complete CUDA row, resource, and AD gates pass.
N_TENDENCY_LAYOUT_MODE = "separate"

DIAGNOSTIC_COMPUTE_MODES = ("baseline", "compact_active")
# The Fortran predicate is sparse in the approved corpus (hail: 8.13% mean;
# melting snow/graupel: 11.02%/5.08% in the exercising case).  Keep the exact
# device-side gate in the production policy so requested diagnostics do not run
# their ordered 50/60-bin scans over inactive cells.
DIAGNOSTIC_COMPUTE_MODE = "compact_active"

# Bounded M4A regional mappings.  Each non-baseline mode changes only tensor
# scheduling/lifetime; the scientific kernels and their operation order remain
# shared with the baseline path.  Keeping the candidates independently
# selectable lets the CUDA qualification run accept or reject them atomically.
REGIONAL_EXECUTION_MODES: dict[str, frozenset[str]] = {
    "baseline": frozenset(),
    "g_shared_zeros": frozenset({"g_shared_zeros"}),
    "o_fused_updates": frozenset({"o_fused_updates"}),
    "q_combined_checks": frozenset({"q_combined_checks"}),
    "r1_fused_copyback": frozenset({"r1_fused_copyback"}),
    "g_shared_zeros_r1_fused_copyback": frozenset({"g_shared_zeros", "r1_fused_copyback"}),
    "all_candidates": frozenset({"g_shared_zeros", "o_fused_updates", "q_combined_checks", "r1_fused_copyback"}),
}
# Q1-selected production mapping.  The rejected O/Q candidates remain
# independently selectable for diagnostics but are not part of the default.
REGIONAL_EXECUTION_MODE = "g_shared_zeros"

DEFAULT_EULERIAN_SEDIMENTATION_SUBSTEPS = 4
DEFAULT_LONG_DT_EULERIAN_SEDIMENTATION_SUBSTEPS = 24
DEFAULT_LONG_DT_EULERIAN_THRESHOLD_SECONDS = 120.0
DEFAULT_SEMI_SEDIMENTATION_SUBSTEPS = 4
DEFAULT_PRODUCTION_SEMI_SEDIMENTATION_SUBSTEPS = 1
DEFAULT_SNOW_SEDIMENTATION_SUBSTEPS = 4
# The caps above were accuracy-gated only on the approved corpus (dt <= 120 s,
# nz=59). Fortran substeps each column int(dt*vt/dz + 1) times
# (module_mp_tempo_main.F90 rain/graupel/snow_fallspeed), divided by 10 on the
# semi-Lagrangian path (main.F90:601,642). A caller that knows its grid passes
# ``min_layer_depth``; the loop count then becomes max(cap, bound), where the
# bound uses this fall-speed ceiling. TEMPO limits rain size (d0r_max=2.5 mm,
# module_mp_tempo_params.F90:76, enforced at main.F90:1015-1019; 7.09 m/s at
# rhof=1) and two-moment graupel size (25.4 mm, main.F90:1186-1189), but not
# snow or one-moment graupel, and every fall speed carries the unbounded factor
# rhof=sqrt(rho_not/rho) (main.F90:1282). The ceiling is therefore a host
# assumption, not a source guarantee: the runtime always reports
# ``diagnostics[SEDIMENTATION_SUBSTEP_OVERFLOW]`` (per column, on device, no
# host sync) so a violated assumption is detectable instead of silent.
DEFAULT_MAX_SEDIMENTATION_FALL_SPEED = 20.0
SEDIMENTATION_SUBSTEP_OVERFLOW = "sedimentation_substep_overflow"
# Overflow regions of the static loop bounds: rain and graupel share one bound, snow has its own.
SEDIMENTATION_BOUND_RAIN_GRAUPEL = "tempo.bound.rain_graupel"
SEDIMENTATION_BOUND_SNOW = "tempo.bound.snow"
SEDIMENTATION_RECURRENCE_EXECUTION_MODES = (
    "preparation_update_fused",
    "fixed_step",
)
# The package default keeps the P1/P3/P5 recurrence wrappers eager. Whole-recurrence
# fusion stays an explicit option until it is measured on CUDA.
SEDIMENTATION_RECURRENCE_EXECUTION_MODE = "fixed_step"
# CUDA compile islands are opt-in. Accuracy gates and eager timing should not
# enter Dynamo/Inductor unless the caller explicitly requests compile evidence.
RADAR_FUSED_REDUCTION_COMPILE_CUDA = False
RADAR_FUSED_REDUCTION_COMPILE_DEVICES = ("cpu", "cuda")
SEDIMENTATION_RECURRENCE_COMPILE_CUDA = False
EULERIAN_SEDIMENTATION_RECURRENCE_COMPILE_CUDA = False
SEMI_SEDIMENTATION_RECURRENCE_COMPILE_CUDA = False
SEDIMENTATION_RECURRENCE_COMPILE_DEVICES = ("cpu", "cuda")
SEMI_LAGRANGIAN_REDUCTION_COMPILE_CUDA = False
SEMI_LAGRANGIAN_REDUCTION_COMPILE_DEVICES = ("cpu", "cuda")
SEMI_LAGRANGIAN_REDUCTION_COMPILE_POLICY = "explicit_opt_in_chunk_compile_outer_eager"
SEMI_LAGRANGIAN_REDUCTION_OUTER_EAGER_NAMES = frozenset({"single", "species"})
TENSOR_COMPILE_ISLAND_CUDA = False
TENSOR_COMPILE_ISLAND_DEVICES = ("cpu", "cuda")
STRICT_CUDA_COMPILE_ISLANDS = False
SEMI_LAGRANGIAN_STREAMING_REDUCTION_ENABLED = True
SEMI_LAGRANGIAN_COMPUTE_MODES = ("baseline_chunked", "ordered_compiled_scan", "ordered_geometry_bound", "fortran_ordered")
# "fortran_ordered" (opt-in, G3.3b) evaluates every semi-Lagrangian step in the Fortran
# operation order: sequential zi, top-down wi limiter recurrence, (wi*dt)*(1/steps) arrival
# points, same-cell formula, increasing-source sums and the ordered precip scan.
# The ordered scan is an opt-in A1 candidate until its CUDA speed and AD gates pass.
SEMI_LAGRANGIAN_OVERLAP_MODE = "baseline_chunked"
SEMI_LAGRANGIAN_ORDERED_SCAN_CHUNK_SOURCES = 8
# Unbounded by default: every supported vertical depth retains its full source range.
# Tests may set a smaller window to verify the explicit overflow contract.
SEMI_LAGRANGIAN_ORDERED_WINDOW_SOURCES: int | None = None
# Explicit opt-in domain, checked on the live remapped geometry. At most two
# partial cells plus ceil(destination width / minimum full-source width) cells
# can intersect a destination. These are compressed widths, not velocity guesses.
SEMI_LAGRANGIAN_GEOMETRY_MIN_SOURCE_WIDTH = 40.0
SEMI_LAGRANGIAN_GEOMETRY_MAX_DESTINATION_WIDTH = 865.0
SEMI_SEARCHSORTED_FULL_FUSION_MAX_ELEMENTS = 20_000_000
SEMI_LAGRANGIAN_OVERLAP_CHUNK_MAX_ELEMENTS = 20_000_000
HAIL_FULL_FUSION_MAX_ELEMENTS = 20_000_000

from tempo_torch.coefficients import (
    _HOMOGENEOUS_FREEZING_TEMPERATURE,
    _RADAR_BETA_WATER_ICE,
    _RADAR_COMPLEX_ICE_0,
    _RADAR_COMPLEX_ICE_SQ,
    _RADAR_COMPLEX_WATER_0,
    _RADAR_COMPLEX_WATER_SQ,
    _RADAR_K_W,
    _RADAR_REFLECTIVITY_FACTOR,
    _SCALAR_CIE,
    _SCALAR_CLOUD_WATER_THRESHOLDS,
    _SCALAR_CRE,
    _SCALAR_CRG,
    _SCALAR_CSE,
    _SCALAR_CSG,
    _SCALAR_GRAUPEL_AM_G,
    _SCALAR_GRAUPEL_AV_G,
    _SCALAR_GRAUPEL_BV_G,
    _SCALAR_GRAUPEL_CGE,
    _SCALAR_GRAUPEL_CGG,
    _SCALAR_GRAUPEL_RHO_G,
    _SCALAR_GRAUPEL_SINGLE_MOMENT_LAMG_SCALE,
    _SCALAR_GRAUPEL_WATER_THRESHOLDS,
    _SCALAR_RADAR_GRAUPEL_BINS,
    _SCALAR_RADAR_GRAUPEL_QUADRATURE_WEIGHTS,
    _SCALAR_RADAR_SNOW_BINS,
    _SCALAR_RADAR_SNOW_QUADRATURE_WEIGHTS,
    _SCALAR_RAIN_WATER_THRESHOLDS,
    _SCALAR_SNOW_WATER_THRESHOLDS,
    _SNOW_MOMENT_COMPILE_FIELDS,
    _STATIC_CLOUD_ACTIVATION_NA_AXIS,
    _STATIC_CLOUD_ACTIVATION_T_AXIS,
    _STATIC_CLOUD_ACTIVATION_W_AXIS,
    _STATIC_CLOUD_CCE,
    _STATIC_CLOUD_CCG,
    _STATIC_CLOUD_NUMBER_BINS,
    _STATIC_CLOUD_OCG1,
    _STATIC_CLOUD_OCG2,
    _STATIC_D0I,
    _STATIC_DROP_EVAP_NUMBER_TABLE,
    _STATIC_GRAUPEL_AM_G,
    _STATIC_GRAUPEL_AV_G,
    _STATIC_GRAUPEL_BV_G,
    _STATIC_GRAUPEL_CGE,
    _STATIC_GRAUPEL_CGG,
    _STATIC_GRAUPEL_RHO_G,
    _STATIC_HAIL_BINS,
    _STATIC_ICE_AUTOCONVERSION_TABLES,
    _STATIC_NDARRAY_IDS,
    _STATIC_RADAR_BINS,
    _STATIC_RAIN_DIAMETER_BINS,
    _STATIC_SIMPSON_WEIGHTS,
    _STATIC_SNOW_A,
    _STATIC_SNOW_B,
    _STATIC_SNOW_CLOUD_WATER_EFFICIENCY,
    _STATIC_SNOW_CSE,
    _STATIC_SNOW_CSG,
    _STATIC_SNOW_DIAMETER_BINS,
    _STATIC_WARM_RAIN_EFFICIENCY,
)

# Frozen TEMPO REAL64 initialization (mu_i=0, bm_i=3, bv_i=1).
# Host math.gamma differs at Gamma(3.5), which affects small number tendencies.
_SCALAR_CIG = (1.0, 6.0, 24.0, 1.0, 1.0, 3.32335097044784256, 1.32934038817913702)
# Rain number fall speed uses the same frozen Gamma(3.5) coefficient.
_SCALAR_CRG = (*_SCALAR_CRG[:6], 3.32335097044784256, *_SCALAR_CRG[7:])


def clear_tensor_compile_island_cache() -> None:
    _TENSOR_COMPILE_ISLAND_CACHE.clear()
    _TENSOR_COMPILE_ISLAND_DISABLED.clear()
    _TENSOR_COMPILE_ISLAND_STATUS.clear()
    _TENSOR_COMPILE_ISLAND_TIMINGS.clear()
    _RADAR_COMPILED_REDUCTION_CACHE.clear()
    _RADAR_COMPILED_REDUCTION_DISABLED.clear()
    _SEDIMENTATION_RECURRENCE_COMPILED_CACHE.clear()
    _SEDIMENTATION_RECURRENCE_COMPILE_DISABLED.clear()
    _SPARSE_DIAGNOSTIC_COMPILED_CACHE.clear()
    _SPARSE_DIAGNOSTIC_COMPILE_DISABLED.clear()


def set_n_limiter_compile_schedule(schedule: str) -> str:
    """Select one bounded N mapping without changing global compiler config."""
    global N_LIMITER_COMPILE_SCHEDULE
    if schedule not in N_LIMITER_COMPILE_SCHEDULES:
        raise ValueError(
            f"unsupported N-limiter compile schedule {schedule!r}; "
            f"expected one of {tuple(N_LIMITER_COMPILE_SCHEDULES)}"
        )
    previous = N_LIMITER_COMPILE_SCHEDULE
    N_LIMITER_COMPILE_SCHEDULE = schedule
    return previous


def n_limiter_compile_schedule() -> dict[str, Any]:
    selected = N_LIMITER_COMPILE_SCHEDULES[N_LIMITER_COMPILE_SCHEDULE]
    return {
        "name": N_LIMITER_COMPILE_SCHEDULE,
        "pressure_split": bool(selected["pressure_split"]),
        "options": dict(selected["options"]),
        "scope": "N limit-and-sum only",
    }


def set_n_tendency_layout_mode(mode: str) -> str:
    global N_TENDENCY_LAYOUT_MODE
    if mode not in N_TENDENCY_LAYOUT_MODES:
        raise ValueError(f"unsupported N tendency layout mode {mode!r}")
    previous = N_TENDENCY_LAYOUT_MODE
    N_TENDENCY_LAYOUT_MODE = mode
    return previous


def n_tendency_layout_mode() -> str:
    return N_TENDENCY_LAYOUT_MODE


def set_diagnostic_compute_mode(mode: str) -> str:
    """Select bounded hail/radar work mapping without changing formulas."""
    global DIAGNOSTIC_COMPUTE_MODE
    if mode not in DIAGNOSTIC_COMPUTE_MODES:
        raise ValueError(
            f"unsupported diagnostic compute mode {mode!r}; expected one of {DIAGNOSTIC_COMPUTE_MODES}"
        )
    previous = DIAGNOSTIC_COMPUTE_MODE
    DIAGNOSTIC_COMPUTE_MODE = mode
    return previous


def diagnostic_compute_mode() -> str:
    return DIAGNOSTIC_COMPUTE_MODE


def set_regional_execution_mode(mode: str) -> str:
    """Select one bounded B-O/Q/S execution mapping."""
    global REGIONAL_EXECUTION_MODE
    if mode not in REGIONAL_EXECUTION_MODES:
        raise ValueError(
            f"unsupported regional execution mode {mode!r}; "
            f"expected one of {tuple(REGIONAL_EXECUTION_MODES)}"
        )
    previous = REGIONAL_EXECUTION_MODE
    REGIONAL_EXECUTION_MODE = mode
    return previous


def regional_execution_mode() -> str:
    return REGIONAL_EXECUTION_MODE


def set_sedimentation_recurrence_execution_mode(mode: str) -> str:
    """Select one order-preserving compile boundary for P1/P3/P5 recurrences."""
    global SEDIMENTATION_RECURRENCE_EXECUTION_MODE
    if mode not in SEDIMENTATION_RECURRENCE_EXECUTION_MODES:
        raise ValueError(
            f"unsupported sedimentation recurrence execution mode {mode!r}; "
            f"expected one of {SEDIMENTATION_RECURRENCE_EXECUTION_MODES}"
        )
    previous = SEDIMENTATION_RECURRENCE_EXECUTION_MODE
    SEDIMENTATION_RECURRENCE_EXECUTION_MODE = mode
    return previous


def sedimentation_recurrence_execution_mode() -> str:
    return SEDIMENTATION_RECURRENCE_EXECUTION_MODE


# Sedimentation substep compaction. Each column runs exactly its own (Fortran) substep count;
# the static loop bound only caps it. With tiers {kind: ((first, fraction), ...)} for kind
# "rain_graupel" (rain and graupel semi-Lagrangian, one shared loop), "rain" or "graupel" (one of
# them alone, as in the P2/P4 segment functions) or "snow" (Eulerian), substeps from ``first`` on
# run on a buffer of ceil(fraction * ncol) columns gathered from those that still need them, instead
# of on every column under a mask. Tiers nest: a tier's buffer is gathered from the previous tier's
# buffer and written back into it, so only the outermost buffer copies to and from the full
# arrays. Column-local arithmetic is unchanged; a tier whose buffer is too small sets its overflow
# bit and the driver reruns the step. A kind without tiers runs every substep on every column; a
# tier of at least the enclosing buffer's (or MIN) columns runs there uncompacted.
SEDIMENTATION_COMPACTION_KINDS = ("rain_graupel", "rain", "graupel", "snow")
SEDIMENTATION_COMPACTION: dict[str, tuple[tuple[int, float], ...]] = {}
SEDIMENTATION_COMPACTION_MIN_COLUMNS = 512


def set_sedimentation_compaction(
    tiers: Mapping[str, tuple[tuple[int, float], ...]],
) -> dict[str, tuple[tuple[int, float], ...]]:
    """Set the substep compaction tiers {kind: ((first substep, column fraction), ...)}; return the previous."""
    global SEDIMENTATION_COMPACTION
    checked = {}
    for kind, kind_tiers in tiers.items():
        kind_tiers = tuple((int(first), float(fraction)) for first, fraction in kind_tiers)
        firsts = [first for first, _ in kind_tiers]
        if (kind not in SEDIMENTATION_COMPACTION_KINDS or any(first < 0 for first in firsts)
                or firsts != sorted(set(firsts)) or any(not 0.0 < f <= 1.0 for _, f in kind_tiers)):
            raise ValueError("compaction tiers need a known kind, increasing first substeps >= 0 "
                             "and fractions in (0, 1]")
        checked[kind] = kind_tiers
    previous, SEDIMENTATION_COMPACTION = SEDIMENTATION_COMPACTION, checked
    return previous


def sedimentation_compaction() -> dict[str, tuple[tuple[int, float], ...]]:
    return dict(SEDIMENTATION_COMPACTION)


def _sedimentation_substep_tiers(
    run: Callable[[dict[str, torch.Tensor], int, int], dict[str, torch.Tensor]],
    columns: dict[str, torch.Tensor],
    *,
    kind: str,
    need: Callable[[dict[str, torch.Tensor], int], torch.Tensor],
    bound: int,
    axes: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Substeps [0, bound) of a column-local recurrence, later tiers on nested compacted buffers.

    ``run(columns, first, stop)`` runs substeps [first, stop) and returns the entries it
    updates; ``columns`` holds every per-column input (column axis 0, or ``axes[name]``).
    ``need(columns, first)`` is the [ncol] mask of the columns that need substep ``first``; a
    column needing a substep needs every earlier one, so each tier's columns are a subset of the
    previous tier's. A buffer takes those columns first (stable order), then columns that need
    nothing, which the recurrence's own masks leave unchanged; gather and write-back copy values
    exactly. A tier whose buffer is too small sets overflow region ``tempo.<kind>.<first>``; that
    step's results are not exact and the driver reruns it with a larger buffer.
    """
    # Plain lookups, not dict.get: compile-reachable code keeps no .get calls (structural ratchet).
    axes = {name: axes[name] if axes is not None and name in axes else 0 for name in columns}
    kind_tiers = SEDIMENTATION_COMPACTION[kind] if kind in SEDIMENTATION_COMPACTION else ()
    tiers = [(first, fraction) for first, fraction in kind_tiers if first < bound]
    edges = [0, *(first for first, _ in tiers), bound]
    segments = [(first, stop, fraction)
                for (first, stop), fraction in zip(zip(edges[:-1], edges[1:]), (1.0, *(f for _, f in tiers)))
                if first < stop]  # a tier from substep 0 leaves no uncompacted range
    name = next(iter(columns))
    ncol = columns[name].shape[axes[name]]
    return _sedimentation_nested_tiers(run, columns, segments, kind=kind, need=need, axes=axes, ncol=ncol)[0]


def _sedimentation_nested_tiers(
    run: Callable[[dict[str, torch.Tensor], int, int], dict[str, torch.Tensor]],
    columns: dict[str, torch.Tensor],
    segments: list[tuple[int, int, float]],
    *,
    kind: str,
    need: Callable[[dict[str, torch.Tensor], int], torch.Tensor],
    axes: Mapping[str, int],
    ncol: int,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Run ``segments`` on ``columns`` (a buffer of the full arrays); return them and the updated names."""
    if not segments:
        return columns, ()
    (first, stop, fraction), inner = segments[0], segments[1:]
    name = next(iter(columns))
    size = columns[name].shape[axes[name]]
    capacity = max(math.ceil(fraction * ncol), SEDIMENTATION_COMPACTION_MIN_COLUMNS)
    if capacity >= size:
        updated = run(columns, first, stop)
        columns, names = _sedimentation_nested_tiers(run, {**columns, **updated}, inner, kind=kind, need=need,
                                                     axes=axes, ncol=ncol)
        return columns, tuple(dict.fromkeys((*updated, *names)))
    needed = need(columns, first)
    # An undersized buffer does not stop the run: the tier sets its overflow bit and the
    # driver reruns the step with a larger buffer (ufs_physics_common.compaction).
    note_overflow(f"tempo.{kind}.{first}", needed.sum() > capacity)
    order = torch.argsort(needed.to(torch.int8), descending=True, stable=True)[:capacity]
    part = {name: value.index_select(axes[name], order) for name, value in columns.items()}
    updated = run(part, first, stop)
    part, names = _sedimentation_nested_tiers(run, {**part, **updated}, inner, kind=kind, need=need,
                                              axes=axes, ncol=ncol)
    names = tuple(dict.fromkeys((*updated, *names)))
    # contiguous(): an expanded (stride-0) input, e.g. a zero precipitation accumulator, keeps its
    # stride under compiled index_copy, so every column would write one element.
    return {**columns, **{name: columns[name].contiguous().index_copy(axes[name], order, part[name])
                          for name in names}}, names


def set_semi_lagrangian_compute_mode(mode: str) -> str:
    """Select the baseline or the bounded, Fortran-ordered A1 scan candidate."""
    global SEMI_LAGRANGIAN_OVERLAP_MODE
    if mode not in SEMI_LAGRANGIAN_COMPUTE_MODES:
        raise ValueError(f"unsupported semi-Lagrangian overlap mode {mode!r}")
    previous = SEMI_LAGRANGIAN_OVERLAP_MODE
    SEMI_LAGRANGIAN_OVERLAP_MODE = mode
    return previous


def semi_lagrangian_compute_mode() -> str:
    return SEMI_LAGRANGIAN_OVERLAP_MODE


def _regional_execution_enabled(candidate: str) -> bool:
    return candidate in REGIONAL_EXECUTION_MODES[REGIONAL_EXECUTION_MODE]


def _tensor_compile_options(name: str) -> dict[str, Any]:
    if not name.startswith("limit_and_sum_tendencies") and not name.startswith("n_sum_"):
        return {}
    return dict(N_LIMITER_COMPILE_SCHEDULES[N_LIMITER_COMPILE_SCHEDULE]["options"])


def cuda_compile_island_flags() -> dict[str, bool]:
    return {
        "RADAR_FUSED_REDUCTION_COMPILE_CUDA": RADAR_FUSED_REDUCTION_COMPILE_CUDA,
        "SEDIMENTATION_RECURRENCE_COMPILE_CUDA": SEDIMENTATION_RECURRENCE_COMPILE_CUDA,
        "EULERIAN_SEDIMENTATION_RECURRENCE_COMPILE_CUDA": EULERIAN_SEDIMENTATION_RECURRENCE_COMPILE_CUDA,
        "SEMI_SEDIMENTATION_RECURRENCE_COMPILE_CUDA": SEMI_SEDIMENTATION_RECURRENCE_COMPILE_CUDA,
        "SEMI_LAGRANGIAN_REDUCTION_COMPILE_CUDA": SEMI_LAGRANGIAN_REDUCTION_COMPILE_CUDA,
        "TENSOR_COMPILE_ISLAND_CUDA": TENSOR_COMPILE_ISLAND_CUDA,
    }


def set_strict_cuda_compile_islands(enabled: bool) -> bool:
    global STRICT_CUDA_COMPILE_ISLANDS
    previous = STRICT_CUDA_COMPILE_ISLANDS
    STRICT_CUDA_COMPILE_ISLANDS = bool(enabled)
    return previous


def set_cuda_compile_categories(
    *,
    elementwise: bool | None = None,
    radar: bool | None = None,
    eulerian_sedimentation: bool | None = None,
    semi_sedimentation: bool | None = None,
    semi_lagrangian_reduction: bool | None = None,
    legacy_sedimentation: bool | None = None,
) -> dict[str, bool]:
    global RADAR_FUSED_REDUCTION_COMPILE_CUDA
    global SEDIMENTATION_RECURRENCE_COMPILE_CUDA
    global EULERIAN_SEDIMENTATION_RECURRENCE_COMPILE_CUDA
    global SEMI_SEDIMENTATION_RECURRENCE_COMPILE_CUDA
    global SEMI_LAGRANGIAN_REDUCTION_COMPILE_CUDA
    global TENSOR_COMPILE_ISLAND_CUDA
    previous = cuda_compile_island_flags()
    if radar is not None:
        RADAR_FUSED_REDUCTION_COMPILE_CUDA = bool(radar)
    if legacy_sedimentation is not None:
        SEDIMENTATION_RECURRENCE_COMPILE_CUDA = bool(legacy_sedimentation)
    if eulerian_sedimentation is not None:
        EULERIAN_SEDIMENTATION_RECURRENCE_COMPILE_CUDA = bool(eulerian_sedimentation)
    if semi_sedimentation is not None:
        SEMI_SEDIMENTATION_RECURRENCE_COMPILE_CUDA = bool(semi_sedimentation)
    if semi_lagrangian_reduction is not None:
        SEMI_LAGRANGIAN_REDUCTION_COMPILE_CUDA = bool(semi_lagrangian_reduction)
    if elementwise is not None:
        TENSOR_COMPILE_ISLAND_CUDA = bool(elementwise)
    return previous


def set_cuda_compile_islands_enabled(enabled: bool) -> dict[str, bool]:
    # Compatibility helper for older smoke/golden callers. New benchmark code uses
    # set_cuda_compile_categories so compile evidence is attributable by category.
    return set_cuda_compile_categories(
        elementwise=enabled,
        radar=enabled,
        eulerian_sedimentation=enabled,
        semi_sedimentation=enabled,
        semi_lagrangian_reduction=enabled,
        legacy_sedimentation=False,
    )


def restore_cuda_compile_islands_enabled(previous: Mapping[str, bool]) -> None:
    global RADAR_FUSED_REDUCTION_COMPILE_CUDA
    global SEDIMENTATION_RECURRENCE_COMPILE_CUDA
    global EULERIAN_SEDIMENTATION_RECURRENCE_COMPILE_CUDA
    global SEMI_SEDIMENTATION_RECURRENCE_COMPILE_CUDA
    global SEMI_LAGRANGIAN_REDUCTION_COMPILE_CUDA
    global TENSOR_COMPILE_ISLAND_CUDA
    RADAR_FUSED_REDUCTION_COMPILE_CUDA = bool(previous["RADAR_FUSED_REDUCTION_COMPILE_CUDA"])
    SEDIMENTATION_RECURRENCE_COMPILE_CUDA = bool(previous.get("SEDIMENTATION_RECURRENCE_COMPILE_CUDA", False))
    EULERIAN_SEDIMENTATION_RECURRENCE_COMPILE_CUDA = bool(
        previous.get(
            "EULERIAN_SEDIMENTATION_RECURRENCE_COMPILE_CUDA",
            previous.get("SEDIMENTATION_RECURRENCE_COMPILE_CUDA", False),
        )
    )
    SEMI_SEDIMENTATION_RECURRENCE_COMPILE_CUDA = bool(
        previous.get(
            "SEMI_SEDIMENTATION_RECURRENCE_COMPILE_CUDA",
            previous.get("SEDIMENTATION_RECURRENCE_COMPILE_CUDA", False),
        )
    )
    SEMI_LAGRANGIAN_REDUCTION_COMPILE_CUDA = bool(previous["SEMI_LAGRANGIAN_REDUCTION_COMPILE_CUDA"])
    TENSOR_COMPILE_ISLAND_CUDA = bool(previous["TENSOR_COMPILE_ISLAND_CUDA"])


def _compile_signature_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return (
            "tensor",
            str(value.device),
            str(value.dtype),
            tuple(int(dim) for dim in value.shape),
            tuple(int(stride) for stride in value.stride()),
            bool(value.requires_grad),
        )
    if isinstance(value, np.ndarray):
        digest = sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
        return ("ndarray", str(value.dtype), tuple(int(dim) for dim in value.shape), digest)
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.bool_):
            value = bool(value)
        elif np.issubdtype(value.dtype, np.integer):
            value = int(value)
        elif np.issubdtype(value.dtype, np.floating):
            value = float(value)
        elif np.issubdtype(value.dtype, np.complexfloating):
            value = complex(value)
        else:
            value = str(value)
    if isinstance(value, (bool, int, float, str, type(None))):
        return ("const", type(value).__name__, value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_compile_signature_value(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_compile_signature_value(item) for item in value))
    if isinstance(value, Mapping):
        return (
            "mapping",
            type(value).__qualname__,
            tuple(
                (str(key), _compile_signature_value(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return (
            "dataclass",
            type(value).__qualname__,
            tuple(
                (field.name, _compile_signature_value(getattr(value, field.name)))
                for field in dataclass_fields(value)
            ),
        )
    return ("object", type(value).__module__, type(value).__qualname__, repr(value))


def _compile_signature_values(values: tuple[Any, ...]) -> tuple[Any, ...]:
    """Build common flat tensor/literal signatures without recursive dispatch."""
    signatures: list[Any] = []
    append = signatures.append
    for value in values:
        if torch.is_tensor(value):
            append(
                (
                    "tensor",
                    str(value.device),
                    str(value.dtype),
                    tuple(value.shape),
                    tuple(value.stride()),
                    bool(value.requires_grad),
                )
            )
        elif isinstance(value, bool):
            append(("const", "bool", value))
        elif isinstance(value, int):
            append(("const", "int", value))
        elif isinstance(value, float):
            append(("const", "float", value))
        elif value is None:
            append(("const", "NoneType", None))
        else:
            append(_compile_signature_value(value))
    return tuple(signatures)


def _compile_runtime_signature_value(value: Any) -> Any:
    """Build a hashable specialization key without formatting tensor metadata."""
    if torch.is_tensor(value):
        device = value.device
        return (
            "tensor",
            device.type,
            device.index,
            value.dtype,
            tuple(value.shape),
            tuple(value.stride()),
            value.requires_grad,
        )
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.bool_):
            value = bool(value)
        elif np.issubdtype(value.dtype, np.integer):
            value = int(value)
        elif np.issubdtype(value.dtype, np.floating):
            value = float(value)
        elif np.issubdtype(value.dtype, np.complexfloating):
            value = complex(value)
    if isinstance(value, (bool, int, float, complex, str, type(None))):
        return ("const", type(value), value)
    return _compile_signature_value(value)


def _compile_cache_key(
    name: str,
    function: Any,
    reference: torch.Tensor,
    *,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
    compile_options: Mapping[str, Any] | None = None,
) -> _CompileCacheKey:
    return (
        name,
        getattr(function, "__module__", ""),
        getattr(function, "__qualname__", repr(function)),
        _compile_runtime_signature_value(reference),
        tuple(_compile_runtime_signature_value(value) for value in args),
        tuple(
            (key, _compile_runtime_signature_value(value))
            for key, value in sorted((kwargs or {}).items(), key=lambda pair: pair[0])
        ),
        tuple(sorted((compile_options or {}).items())),
    )


def _compile_signature_label(key: _CompileCacheKey) -> str:
    return sha256(repr(key).encode("utf-8")).hexdigest()[:16]


def _prepared_compile_key(
    status_name: str,
    function: Any,
    runtime_signature: Any,
) -> tuple[Any, ...]:
    """Never reuse a prepared callable across a different guarded signature."""
    return (status_name, function, runtime_signature)


def _prepared_runtime_signature(
    reference: torch.Tensor,
    args: tuple[Any, ...],
    *,
    compile_options: Mapping[str, Any] | None = None,
) -> tuple[Any, ...]:
    """Build the minimum safe hot-path key without the full cache-key machinery."""
    return (
        _compile_runtime_signature_value(reference),
        tuple(_compile_runtime_signature_value(value) for value in args),
        tuple(sorted((compile_options or {}).items())),
    )


def _compile_exception_reason(exc: Exception) -> str:
    message = str(exc).splitlines()[0] if str(exc) else ""
    reason = type(exc).__name__
    if message:
        reason = f"{reason}: {message}"
    return reason[:500]


def _record_tensor_compile_island_status(
    name: str,
    status: str,
    reason: str | None = None,
    *,
    cache_key: _CompileCacheKey | None = None,
) -> None:
    record: dict[str, Any] = {"status": status}
    if reason:
        record["reason"] = reason
    if cache_key is not None:
        record["signature"] = _compile_signature_label(cache_key)
    _TENSOR_COMPILE_ISLAND_STATUS[name] = record


def tensor_compile_island_status() -> dict[str, dict[str, Any]]:
    return {name: dict(record) for name, record in sorted(_TENSOR_COMPILE_ISLAND_STATUS.items())}


def tensor_compile_island_timings() -> tuple[dict[str, Any], ...]:
    return tuple(dict(record) for record in _TENSOR_COMPILE_ISLAND_TIMINGS)


def _record_compile_island_timing(
    name: str,
    cache_key: _CompileCacheKey,
    timing: Mapping[str, float],
) -> None:
    _TENSOR_COMPILE_ISLAND_TIMINGS.append(
        {
            "name": name,
            "signature": _compile_signature_label(cache_key),
            **{key: float(value) for key, value in timing.items()},
        }
    )


def _raise_compile_island_failure_if_strict(
    status_name: str,
    status: str,
    reason: str,
    exc: Exception,
) -> None:
    if STRICT_CUDA_COMPILE_ISLANDS:
        raise RuntimeError(f"{status_name} {status}: {reason}") from exc


def _compile_and_prime_with_scoped_scalar_capture(
    function: Any,
    args: tuple[Any, ...],
    compile_kwargs: Mapping[str, Any],
) -> tuple[Any | None, Any | None, str | None, Exception | None, dict[str, float]]:
    """Create and first-call one lazy compiler wrapper under scoped Dynamo state."""
    compiled = None
    result = None
    stage = None
    failure = None
    wrapper_seconds = 0.0
    first_call_seconds = 0.0
    with torch_dynamo_config.patch(
        capture_scalar_outputs=True,
        recompile_limit=TEMPO_COMPILE_RECOMPILE_LIMIT,
    ):
        wrapper_start = time.perf_counter()
        try:
            compiled = torch.compile(function, **compile_kwargs)
        except Exception as exc:
            stage = "compile_failed"
            failure = exc
        wrapper_seconds = time.perf_counter() - wrapper_start
        if failure is None:
            first_call_start = time.perf_counter()
            try:
                result = compiled(*args)
            except Exception as exc:
                stage = "runtime_failed"
                failure = exc
            first_call_seconds = time.perf_counter() - first_call_start
    return compiled, result, stage, failure, {
        "wrapper_seconds": wrapper_seconds,
        "first_call_seconds": first_call_seconds,
        "total_seconds": wrapper_seconds + first_call_seconds,
    }


def _run_tensor_compile_island(
    name: str,
    function: Any,
    reference: torch.Tensor,
    *args: Any,
) -> Any:
    if (
        not TENSOR_COMPILE_ISLAND_CUDA
        or reference.device.type not in TENSOR_COMPILE_ISLAND_DEVICES
        or torch.compiler.is_compiling()
    ):
        return function(*args)
    compile_options = _tensor_compile_options(name)
    prepared_dispatch = _ACTIVE_PREPARED_COMPILE_DISPATCH.get()
    prepared_key = _prepared_compile_key(
        name,
        function,
        _prepared_runtime_signature(
            reference,
            args,
            compile_options=compile_options,
        ),
    )
    if prepared_dispatch is not None:
        prepared_callable = prepared_dispatch.callables.get(prepared_key)
        if prepared_callable is not None:
            prepared_dispatch.fast_hits += 1
            try:
                return prepared_callable(*args)
            except Exception as exc:
                reason = _compile_exception_reason(exc)
                raise RuntimeError(
                    f"prepared compile dispatch {name} runtime_failed: {reason}"
                ) from exc
        prepared_dispatch.misses += 1
    cache_key = _compile_cache_key(
        name,
        function,
        reference,
        args=args,
        compile_options=compile_options,
    )
    if cache_key in _TENSOR_COMPILE_ISLAND_DISABLED:
        if prepared_dispatch is not None:
            raise RuntimeError(f"prepared compile dispatch {name} is disabled after an earlier failure")
        return function(*args)
    compiled = _TENSOR_COMPILE_ISLAND_CACHE.get(cache_key)
    if compiled is None:
        compile_kwargs: dict[str, Any] = {
            "fullgraph": True,
            "dynamic": False,
            "mode": None,
        }
        if compile_options:
            compile_kwargs["options"] = compile_options
        compiled, result, failure_stage, failure, compile_timing = _compile_and_prime_with_scoped_scalar_capture(
            function,
            args,
            compile_kwargs,
        )
        _record_compile_island_timing(name, cache_key, compile_timing)
        if failure is not None:
            reason = _compile_exception_reason(failure)
            _TENSOR_COMPILE_ISLAND_DISABLED.add(cache_key)
            _record_tensor_compile_island_status(
                name,
                failure_stage or "compile_failed",
                reason,
                cache_key=cache_key,
            )
            _raise_compile_island_failure_if_strict(
                name,
                failure_stage or "compile_failed",
                reason,
                failure,
            )
            if prepared_dispatch is not None:
                raise RuntimeError(
                    f"prepared compile dispatch {name} {failure_stage or 'compile_failed'}: {reason}"
                ) from failure
            return function(*args)
        assert compiled is not None
        _TENSOR_COMPILE_ISLAND_CACHE[cache_key] = compiled
        _record_tensor_compile_island_status(name, "compiled", cache_key=cache_key)
        if prepared_dispatch is not None:
            prepared_dispatch.callables[prepared_key] = compiled
        return result
    try:
        result = compiled(*args)
    except Exception as exc:
        reason = _compile_exception_reason(exc)
        _TENSOR_COMPILE_ISLAND_CACHE.pop(cache_key, None)
        _TENSOR_COMPILE_ISLAND_DISABLED.add(cache_key)
        _record_tensor_compile_island_status(
            name,
            "runtime_failed",
            reason,
            cache_key=cache_key,
        )
        _raise_compile_island_failure_if_strict(name, "runtime_failed", reason, exc)
        if prepared_dispatch is not None:
            raise RuntimeError(
                f"prepared compile dispatch {name} runtime_failed: {reason}"
            ) from exc
        return function(*args)
    if prepared_dispatch is not None:
        prepared_dispatch.callables[prepared_key] = compiled
    return result


def _run_sparse_diagnostic_compile_island(
    name: str,
    function: Any,
    reference: torch.Tensor,
    *args: Any,
    enabled: bool,
    devices: tuple[str, ...],
) -> Any:
    """Compile compact diagnostic science once across active-count shapes."""
    if (
        not enabled
        or reference.device.type not in devices
        or torch.compiler.is_compiling()
    ):
        return function(*args)
    status_name = f"sparse_diagnostic:{name}"
    def dynamic_signature(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return (
                "dynamic_tensor",
                value.device.type,
                value.device.index,
                value.dtype,
                value.ndim,
                value.requires_grad,
            )
        return _compile_runtime_signature_value(value)

    cache_key: _CompileCacheKey = (
        name,
        getattr(function, "__module__", ""),
        getattr(function, "__qualname__", repr(function)),
        dynamic_signature(reference),
        tuple(dynamic_signature(value) for value in args),
    )
    prepared_dispatch = _ACTIVE_PREPARED_COMPILE_DISPATCH.get()
    prepared_key = _prepared_compile_key(status_name, function, cache_key)
    if prepared_dispatch is not None:
        prepared_callable = prepared_dispatch.callables.get(prepared_key)
        if prepared_callable is not None:
            prepared_dispatch.fast_hits += 1
            try:
                return prepared_callable(*args)
            except Exception as exc:
                reason = _compile_exception_reason(exc)
                raise RuntimeError(
                    f"prepared compile dispatch {status_name} runtime_failed: {reason}"
                ) from exc
        prepared_dispatch.misses += 1
    if cache_key in _SPARSE_DIAGNOSTIC_COMPILE_DISABLED:
        if prepared_dispatch is not None:
            raise RuntimeError(f"prepared compile dispatch {status_name} is disabled")
        return function(*args)
    compiled = _SPARSE_DIAGNOSTIC_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        compiled, result, failure_stage, failure, compile_timing = _compile_and_prime_with_scoped_scalar_capture(
            function,
            args,
            {"fullgraph": True, "dynamic": True, "mode": None},
        )
        _record_compile_island_timing(status_name, cache_key, compile_timing)
        if failure is not None:
            reason = _compile_exception_reason(failure)
            _SPARSE_DIAGNOSTIC_COMPILE_DISABLED.add(cache_key)
            _record_tensor_compile_island_status(
                status_name,
                failure_stage or "compile_failed",
                reason,
                cache_key=cache_key,
            )
            _raise_compile_island_failure_if_strict(
                status_name,
                failure_stage or "compile_failed",
                reason,
                failure,
            )
            if prepared_dispatch is not None:
                raise RuntimeError(
                    f"prepared compile dispatch {status_name} {failure_stage or 'compile_failed'}: {reason}"
                ) from failure
            return function(*args)
        assert compiled is not None
        _SPARSE_DIAGNOSTIC_COMPILED_CACHE[cache_key] = compiled
        _record_tensor_compile_island_status(status_name, "compiled_dynamic", cache_key=cache_key)
        if prepared_dispatch is not None:
            prepared_dispatch.callables[prepared_key] = compiled
        return result
    try:
        result = compiled(*args)
    except Exception as exc:
        reason = _compile_exception_reason(exc)
        _SPARSE_DIAGNOSTIC_COMPILED_CACHE.pop(cache_key, None)
        _SPARSE_DIAGNOSTIC_COMPILE_DISABLED.add(cache_key)
        _record_tensor_compile_island_status(
            status_name,
            "runtime_failed",
            reason,
            cache_key=cache_key,
        )
        _raise_compile_island_failure_if_strict(status_name, "runtime_failed", reason, exc)
        if prepared_dispatch is not None:
            raise RuntimeError(
                f"prepared compile dispatch {status_name} runtime_failed: {reason}"
            ) from exc
        return function(*args)
    if prepared_dispatch is not None:
        prepared_dispatch.callables[prepared_key] = compiled
    return result


def sedimentation_substep_bound(
    *,
    dt: float,
    min_layer_depth: float,
    semi_sedi_flag: bool,
    max_fall_speed: float = DEFAULT_MAX_SEDIMENTATION_FALL_SPEED,
) -> int:
    """Static upper bound on TEMPO's per-column sedimentation substep count.

    Mirrors Fortran ``int(dt/(dz/vt) + 1)`` with ``vt <= max_fall_speed`` and
    ``dz >= min_layer_depth``, plus one guard step for floating-point rounding
    of that quotient; the semi-Lagrangian path then applies ``int(n/10) + 1``.
    Inputs are Python numbers known at preparation time, so no tensor is read.
    """
    if not min_layer_depth > 0.0:
        raise ValueError("min_layer_depth must be positive")
    if not max_fall_speed > 0.0:
        raise ValueError("max_fall_speed must be positive")
    steps = int(abs(float(dt)) * float(max_fall_speed) / float(min_layer_depth) + 1.0) + 1
    if semi_sedi_flag:
        steps = max(int(steps / 10.0) + 1, 1)
    return steps


def _record_sedimentation_substep_overflow(
    diagnostics: dict[str, torch.Tensor],
    required_steps: torch.Tensor,
    loop_count: int,
    branch_active: torch.Tensor,
    *,
    region: str,
) -> None:
    """Flag columns whose Fortran substep count exceeds the static loop bound.

    ORs the per-column flag into ``diagnostics[SEDIMENTATION_SUBSTEP_OVERFLOW]`` and sets the
    bound's overflow region (``tempo.bound.<kind>``, ``ufs_physics_common.compaction``), so a
    driver reruns the step with a larger bound instead of keeping a truncated result. The flag
    is device state, not ``torch._assert_async`` (which compiled code drops).
    """
    overflow = (required_steps > loop_count) & branch_active.reshape(-1)
    note_overflow(region, overflow.any())
    if SEDIMENTATION_SUBSTEP_OVERFLOW in diagnostics:
        overflow = diagnostics[SEDIMENTATION_SUBSTEP_OVERFLOW] | overflow
    diagnostics[SEDIMENTATION_SUBSTEP_OVERFLOW] = overflow


def _select_sedimentation_substeps(
    *,
    dt: float,
    semi_sedi_flag: bool,
    override: int | None,
    eulerian_cap: int,
    long_dt_eulerian_cap: int,
    long_dt_eulerian_threshold: float,
    semi_cap: int,
    min_layer_depth: float | None = None,
    max_fall_speed: float = DEFAULT_MAX_SEDIMENTATION_FALL_SPEED,
) -> int:
    selected = _select_sedimentation_cap(
        dt=dt,
        semi_sedi_flag=semi_sedi_flag,
        override=override,
        eulerian_cap=eulerian_cap,
        long_dt_eulerian_cap=long_dt_eulerian_cap,
        long_dt_eulerian_threshold=long_dt_eulerian_threshold,
        semi_cap=semi_cap,
    )
    if min_layer_depth is None:
        return selected
    return max(
        selected,
        sedimentation_substep_bound(
            dt=dt,
            min_layer_depth=min_layer_depth,
            semi_sedi_flag=semi_sedi_flag,
            max_fall_speed=max_fall_speed,
        ),
    )


def _select_sedimentation_cap(
    *,
    dt: float,
    semi_sedi_flag: bool,
    override: int | None,
    eulerian_cap: int,
    long_dt_eulerian_cap: int,
    long_dt_eulerian_threshold: float,
    semi_cap: int,
) -> int:
    if eulerian_cap < 1:
        raise ValueError("max_eulerian_sedimentation_substeps must be positive")
    if long_dt_eulerian_cap < 1:
        raise ValueError("max_long_dt_eulerian_sedimentation_substeps must be positive")
    if long_dt_eulerian_threshold <= 0.0:
        raise ValueError("long_dt_eulerian_threshold must be positive")
    if semi_cap < 1:
        raise ValueError("max_semi_sedimentation_substeps must be positive")
    if override is not None:
        if override < 1:
            raise ValueError("max_sedimentation_substeps must be positive")
        return override
    if semi_sedi_flag:
        return semi_cap
    if abs(float(dt)) >= long_dt_eulerian_threshold:
        return long_dt_eulerian_cap
    return eulerian_cap


def _default_sedimentation_substeps(
    *,
    dt: float,
    semi_sedi_flag: bool,
    override: int | None,
) -> int:
    return _select_sedimentation_substeps(
        dt=dt,
        semi_sedi_flag=semi_sedi_flag,
        override=override,
        eulerian_cap=DEFAULT_EULERIAN_SEDIMENTATION_SUBSTEPS,
        long_dt_eulerian_cap=DEFAULT_LONG_DT_EULERIAN_SEDIMENTATION_SUBSTEPS,
        long_dt_eulerian_threshold=DEFAULT_LONG_DT_EULERIAN_THRESHOLD_SECONDS,
        semi_cap=DEFAULT_PRODUCTION_SEMI_SEDIMENTATION_SUBSTEPS,
    )


def _default_snow_sedimentation_substeps(
    *,
    semi_sedi_flag: bool,
    branch_cap: int,
    override: int | None,
    snow_cap: int | None,
) -> int:
    if snow_cap is not None and snow_cap < 1:
        raise ValueError("max_snow_sedimentation_substeps must be positive")
    if snow_cap is not None:
        return snow_cap
    if override is not None:
        return branch_cap
    if semi_sedi_flag:
        return DEFAULT_SNOW_SEDIMENTATION_SUBSTEPS
    return branch_cap


def _production_snow_sedimentation_substeps(
    *,
    dt: float,
    semi_sedi_flag: bool,
    branch_substeps: int,
    override: int | None,
    snow_cap: int,
    min_layer_depth: float | None,
    max_fall_speed: float,
) -> int:
    if snow_cap < 1:
        raise ValueError("max_snow_sedimentation_substeps must be positive")
    selected = (
        branch_substeps
        if override is not None
        else snow_cap if semi_sedi_flag else branch_substeps
    )
    if min_layer_depth is None:
        return selected
    # Snow always sediments with the Eulerian scheme (main.F90:696-703), so its
    # bound has no semi-Lagrangian reduction even when semi_sedi_flag is set.
    return max(
        selected,
        sedimentation_substep_bound(
            dt=dt,
            min_layer_depth=min_layer_depth,
            semi_sedi_flag=False,
            max_fall_speed=max_fall_speed,
        ),
    )


@dataclass(frozen=True)
class TorchSegmentBResult:
    state: Mapping[str, torch.Tensor]
    ncol: int
    nz: int
    use_temperature: bool
    is_first_step: bool


@dataclass(frozen=True)
class TorchSegmentCWorkspace:
    tend_work: torch.Tensor
    tend: Mapping[str, torch.Tensor]
    tendencies: Mapping[str, torch.Tensor]
    snow_moments: Mapping[str, torch.Tensor]
    fallspeeds: Mapping[str, torch.Tensor]
    vtboost: torch.Tensor
    temp: torch.Tensor
    pres: torch.Tensor
    qv: torch.Tensor
    rho: torch.Tensor
    diagnostics: Mapping[str, torch.Tensor]
    do_micro: torch.Tensor
    supersaturated: torch.Tensor
    local_first_step: bool
    dt: float
    odt: float


@dataclass(frozen=True)
class TorchSegmentCResult:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentCWorkspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentDWorkspace:
    base: TorchSegmentCWorkspace
    nwfa: torch.Tensor
    nifa: torch.Tensor
    nwfaten: torch.Tensor
    nifaten: torch.Tensor
    nwfa1d: torch.Tensor | None
    nifa1d: torch.Tensor | None
    nwfa_present: bool
    nifa_present: bool
    local_first_step: bool
    dt: float


@dataclass(frozen=True)
class TorchSegmentDResult:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentDWorkspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentEWorkspace:
    base: TorchSegmentDWorkspace
    l_qr: torch.Tensor
    l_qi: torch.Tensor
    l_qs: torch.Tensor
    rr: torch.Tensor
    ri: torch.Tensor
    rs: torch.Tensor
    nr: torch.Tensor
    ni: torch.Tensor
    ilamr: torch.Tensor
    ilami: torch.Tensor
    mvd_r: torch.Tensor
    tendencies: Mapping[str, torch.Tensor]
    snow_moments: Mapping[str, torch.Tensor]
    dt: float
    odt: float


@dataclass(frozen=True)
class TorchSegmentEResult:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentEWorkspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentFWorkspace:
    base: TorchSegmentEWorkspace
    l_qc: torch.Tensor
    rc: torch.Tensor
    nc: torch.Tensor
    ncsave: torch.Tensor | None
    nc1d: torch.Tensor | None
    ilamc: torch.Tensor
    mvd_c: torch.Tensor
    tendencies: Mapping[str, torch.Tensor]
    dt: float
    odt: float


@dataclass(frozen=True)
class TorchSegmentFResult:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentFWorkspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentGWorkspace:
    base: TorchSegmentFWorkspace
    l_qg: torch.Tensor
    rg: torch.Tensor
    ng: torch.Tensor
    rb: torch.Tensor
    idx_bg: torch.Tensor
    ilamg: torch.Tensor
    mvd_g: torch.Tensor
    ng1d: torch.Tensor | None
    qb1d: torch.Tensor | None
    tendencies: Mapping[str, torch.Tensor]
    thermo: Mapping[str, torch.Tensor]
    do_micro: torch.Tensor
    supersaturated: torch.Tensor
    dt: float
    odt: float


@dataclass(frozen=True)
class TorchSegmentGResult:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentGWorkspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentHWorkspace:
    base: TorchSegmentGWorkspace
    tend: Mapping[str, torch.Tensor]
    tendencies: Mapping[str, torch.Tensor]
    tables: Any
    dt: float
    odt: float
    # (idx_r, idx_r1) of G's rain (rr, ilamr): H's rain-snow/graupel and I's rain-freezing tables
    # share them. None when H ran no microphysics.
    rain_table_indices: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass(frozen=True)
class TorchSegmentHResult:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentHWorkspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentIWorkspace:
    base: TorchSegmentHWorkspace
    tend: Mapping[str, torch.Tensor]
    tendencies: Mapping[str, torch.Tensor]
    tables: Any
    idx_in: torch.Tensor
    idx_tc: torch.Tensor
    nwfa: torch.Tensor
    nifa: torch.Tensor
    dt: float
    odt: float


@dataclass(frozen=True)
class TorchSegmentIResult:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentIWorkspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentJWorkspace:
    base: TorchSegmentIWorkspace
    tend: Mapping[str, torch.Tensor]
    tendencies: Mapping[str, torch.Tensor]
    tables: Any
    t1_subl: torch.Tensor
    idx_i: torch.Tensor
    idx_i1: torch.Tensor
    dt: float
    odt: float
    packed_ice_tendencies: torch.Tensor | None = None


@dataclass(frozen=True)
class TorchSegmentJResult:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentJWorkspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentKWorkspace:
    base: TorchSegmentJWorkspace
    tend: Mapping[str, torch.Tensor]
    tendencies: Mapping[str, torch.Tensor]
    tables: Any
    vtboost: torch.Tensor
    dt: float
    odt: float
    packed_riming_tendencies: torch.Tensor | None = None


@dataclass(frozen=True)
class TorchSegmentKResult:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentKWorkspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentLWorkspace:
    base: TorchSegmentKWorkspace
    tend: Mapping[str, torch.Tensor]
    tendencies: Mapping[str, torch.Tensor]
    tables: Any
    dt: float
    odt: float
    packed_melting_tendencies: torch.Tensor | None = None


@dataclass(frozen=True)
class TorchSegmentLResult:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentLWorkspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentMWorkspace:
    base: TorchSegmentLWorkspace
    tend: Mapping[str, torch.Tensor]
    tendencies: Mapping[str, torch.Tensor]
    tables: Any
    dt: float
    odt: float


@dataclass(frozen=True)
class TorchSegmentMResult:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentMWorkspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentNWorkspace:
    base: TorchSegmentMWorkspace
    tend_work: torch.Tensor
    tend: Mapping[str, torch.Tensor]
    tendencies: Mapping[str, torch.Tensor]
    tables: Any
    temp: torch.Tensor
    qv: torch.Tensor
    rho: torch.Tensor
    nwfa: torch.Tensor
    nifa: torch.Tensor
    nwfaten: torch.Tensor
    nifaten: torch.Tensor
    l_qc: torch.Tensor
    rc: torch.Tensor
    nc: torch.Tensor
    nc1d: torch.Tensor | None
    ncsave: torch.Tensor | None
    ilamc: torch.Tensor
    mvd_c: torch.Tensor
    l_qr: torch.Tensor
    rr: torch.Tensor
    nr: torch.Tensor
    nr1d: torch.Tensor
    ilamr: torch.Tensor
    mvd_r: torch.Tensor
    l_qi: torch.Tensor
    ri: torch.Tensor
    ni: torch.Tensor
    ni1d: torch.Tensor
    ilami: torch.Tensor
    l_qs: torch.Tensor
    rs: torch.Tensor
    snow_moments: Mapping[str, torch.Tensor]
    l_qg: torch.Tensor
    rg: torch.Tensor
    ng: torch.Tensor
    rb: torch.Tensor
    idx_bg: torch.Tensor
    ilamg: torch.Tensor
    mvd_g: torch.Tensor
    ng1d: torch.Tensor | None
    qb1d: torch.Tensor | None
    thermo: Mapping[str, torch.Tensor]
    do_micro: torch.Tensor
    supersaturated: torch.Tensor
    dt: float
    odt: float


@dataclass(frozen=True)
class TorchSegmentNResult:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentNWorkspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentOWorkspace:
    base: TorchSegmentNWorkspace | None
    tend_work: torch.Tensor
    tend: Mapping[str, torch.Tensor]
    tendencies: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, torch.Tensor]
    fallspeeds: Mapping[str, torch.Tensor]
    tables: Any
    vtboost: torch.Tensor
    temp: torch.Tensor
    qv: torch.Tensor
    rho: torch.Tensor
    nwfa: torch.Tensor
    nifa: torch.Tensor
    nwfaten: torch.Tensor
    nifaten: torch.Tensor
    nwfa1d: torch.Tensor | None
    nifa1d: torch.Tensor | None
    l_qc: torch.Tensor
    rc: torch.Tensor
    nc: torch.Tensor
    nc1d: torch.Tensor | None
    ncsave: torch.Tensor | None
    ilamc: torch.Tensor
    mvd_c: torch.Tensor
    l_qr: torch.Tensor
    rr: torch.Tensor
    nr: torch.Tensor
    nr1d: torch.Tensor
    ilamr: torch.Tensor
    mvd_r: torch.Tensor
    l_qi: torch.Tensor
    ri: torch.Tensor
    ni: torch.Tensor
    ni1d: torch.Tensor
    ilami: torch.Tensor
    l_qs: torch.Tensor
    rs: torch.Tensor
    snow_moments: Mapping[str, torch.Tensor]
    l_qg: torch.Tensor
    rg: torch.Tensor
    ng: torch.Tensor
    rb: torch.Tensor
    idx_bg: torch.Tensor
    ilamg: torch.Tensor
    mvd_g: torch.Tensor
    ng1d: torch.Tensor | None
    qb1d: torch.Tensor | None
    thermo: Mapping[str, torch.Tensor]
    idx_d: torch.Tensor
    idx_c: torch.Tensor
    idx_n: torch.Tensor
    active_columns: torch.Tensor
    do_micro: torch.Tensor
    supersaturated: torch.Tensor
    dt: float
    odt: float


@dataclass(frozen=True)
class TorchSegmentOResult:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentOWorkspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentP1Workspace:
    base: TorchSegmentOWorkspace
    tend_work: torch.Tensor
    tend: Mapping[str, torch.Tensor]
    tendencies: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, torch.Tensor]
    fallspeeds: Mapping[str, torch.Tensor]
    rr: torch.Tensor
    nr: torch.Tensor
    vtrr: torch.Tensor
    vtnr: torch.Tensor
    substeps_sedi: torch.Tensor
    ktop_sedi: torch.Tensor
    rain_precip: torch.Tensor
    semi_sedi_flag: bool
    branch_active: torch.Tensor
    dt: float
    odt: float


@dataclass(frozen=True)
class TorchSegmentP1Result:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentP1Workspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentP2Workspace:
    base: TorchSegmentOWorkspace
    tend_work: torch.Tensor
    tend: Mapping[str, torch.Tensor]
    tendencies: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, torch.Tensor]
    fallspeeds: Mapping[str, torch.Tensor]
    l_qr: torch.Tensor
    rr: torch.Tensor
    nr: torch.Tensor
    nr1d: torch.Tensor
    ilamr: torch.Tensor
    mvd_r: torch.Tensor
    vtrr: torch.Tensor
    vtnr: torch.Tensor
    substeps_sedi: torch.Tensor
    ktop_sedi: torch.Tensor
    rain_precip: torch.Tensor
    semi_sedi_flag: bool
    branch_active: torch.Tensor
    dt: float
    odt: float


@dataclass(frozen=True)
class TorchSegmentP2Result:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentP2Workspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentP3Workspace:
    base: TorchSegmentP1Workspace
    tend_work: torch.Tensor
    tend: Mapping[str, torch.Tensor]
    tendencies: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, torch.Tensor]
    fallspeeds: Mapping[str, torch.Tensor]
    rg: torch.Tensor
    ng: torch.Tensor
    rb: torch.Tensor
    vtrg: torch.Tensor
    vtng: torch.Tensor
    substeps_sedi: torch.Tensor
    ktop_sedi: torch.Tensor
    graupel_liquid_equiv_precip: torch.Tensor
    semi_sedi_flag: bool
    branch_active: torch.Tensor
    dt: float
    odt: float


@dataclass(frozen=True)
class TorchSegmentP3Result:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentP3Workspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentP4Workspace:
    base: TorchSegmentP2Workspace
    tend_work: torch.Tensor
    tend: Mapping[str, torch.Tensor]
    tendencies: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, torch.Tensor]
    fallspeeds: Mapping[str, torch.Tensor]
    l_qg: torch.Tensor
    rg: torch.Tensor
    ng: torch.Tensor
    rb: torch.Tensor
    idx_bg: torch.Tensor
    ilamg: torch.Tensor
    mvd_g: torch.Tensor
    ng1d: torch.Tensor | None
    qb1d: torch.Tensor | None
    vtrg: torch.Tensor
    vtng: torch.Tensor
    substeps_sedi: torch.Tensor
    ktop_sedi: torch.Tensor
    graupel_liquid_equiv_precip: torch.Tensor
    semi_sedi_flag: bool
    branch_active: torch.Tensor
    dt: float
    odt: float


@dataclass(frozen=True)
class TorchSegmentP4Result:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentP4Workspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentP5Workspace:
    base: TorchSegmentP3Workspace | TorchSegmentP4Workspace | None
    tend_work: torch.Tensor
    tend: Mapping[str, torch.Tensor]
    tendencies: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, torch.Tensor]
    fallspeeds: Mapping[str, torch.Tensor]
    l_qs: torch.Tensor
    rs: torch.Tensor
    l_qi: torch.Tensor
    ri: torch.Tensor
    ni: torch.Tensor
    ilami: torch.Tensor
    l_qc: torch.Tensor
    rc: torch.Tensor
    nc: torch.Tensor
    ilamc: torch.Tensor
    vtrs: torch.Tensor
    vtri: torch.Tensor
    vtni: torch.Tensor
    vtrc: torch.Tensor
    vtnc: torch.Tensor
    snow_liquid_equiv_precip: torch.Tensor
    ice_liquid_equiv_precip: torch.Tensor
    cloud_precip: torch.Tensor
    path_id: str
    semi_sedi_flag: bool
    dt: float
    odt: float
    q_temp: torch.Tensor | None = None
    q_qv: torch.Tensor | None = None
    q_rho: torch.Tensor | None = None
    q_thermo: Mapping[str, torch.Tensor] | None = None
    q_nc1d: torch.Tensor | None = None
    q_ncsave: torch.Tensor | None = None
    q_snow_moments: Mapping[str, torch.Tensor] | None = None
    q_ng1d: torch.Tensor | None = None
    q_qb1d: torch.Tensor | None = None
    q_nwfa1d: torch.Tensor | None = None
    q_nifa1d: torch.Tensor | None = None
    q_active_columns: torch.Tensor | None = None


@dataclass(frozen=True)
class TorchSegmentP5Result:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentP5Workspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentQWorkspace:
    base: TorchSegmentP5Workspace | None
    tend_work: torch.Tensor
    tend: Mapping[str, torch.Tensor]
    tendencies: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, torch.Tensor]
    fallspeeds: Mapping[str, torch.Tensor]
    temp: torch.Tensor
    qv: torch.Tensor
    rho: torch.Tensor
    t1d: torch.Tensor
    qv1d: torch.Tensor
    l_qc: torch.Tensor
    qc1d: torch.Tensor
    rc: torch.Tensor
    nc: torch.Tensor
    nc1d: torch.Tensor | None
    ncsave: torch.Tensor | None
    ilamc: torch.Tensor
    mvd_c: torch.Tensor
    l_qr: torch.Tensor
    qr1d: torch.Tensor
    rr: torch.Tensor
    nr: torch.Tensor
    nr1d: torch.Tensor
    ilamr: torch.Tensor
    mvd_r: torch.Tensor
    l_qi: torch.Tensor
    qi1d: torch.Tensor
    ni: torch.Tensor
    ni1d: torch.Tensor
    ri: torch.Tensor
    ilami: torch.Tensor
    l_qs: torch.Tensor
    qs1d: torch.Tensor
    rs: torch.Tensor
    snow_moments: Mapping[str, torch.Tensor]
    l_qg: torch.Tensor
    qg1d: torch.Tensor
    rg: torch.Tensor
    ng: torch.Tensor
    ng1d: torch.Tensor | None
    qb1d: torch.Tensor | None
    rb: torch.Tensor
    idx_bg: torch.Tensor
    ilamg: torch.Tensor
    mvd_g: torch.Tensor
    nwfa1d: torch.Tensor | None
    nifa1d: torch.Tensor | None
    frozen_fraction: torch.Tensor
    active_columns: torch.Tensor
    path_id: str
    semi_sedi_flag: bool
    dt: float
    odt: float


@dataclass(frozen=True)
class TorchSegmentQResult:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentQWorkspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentR1Workspace:
    base: TorchSegmentQWorkspace
    diagnostics: Mapping[str, torch.Tensor]
    rain_precip: torch.Tensor
    cloud_precip: torch.Tensor
    ice_liquid_equiv_precip: torch.Tensor
    snow_liquid_equiv_precip: torch.Tensor
    graupel_liquid_equiv_precip: torch.Tensor
    frozen_fraction: torch.Tensor
    frz_rain_precip: torch.Tensor
    cloud_number_mixing_ratio: torch.Tensor
    rain_med_vol_diam: torch.Tensor
    graupel_med_vol_diam: torch.Tensor
    max_hail_diameter: torch.Tensor
    refl10cm: torch.Tensor
    path_id: str
    semi_sedi_flag: bool
    dt: float
    odt: float


@dataclass(frozen=True)
class TorchSegmentR1Result:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentR1Workspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentR2Workspace:
    base: TorchSegmentR1Workspace
    diagnostics: Mapping[str, torch.Tensor]
    melting_level_index: torch.Tensor | None  # None: reflectivity skipped (run_r2 compute_reflectivity=False)
    melting_snow_reflectivity: torch.Tensor | None
    melting_graupel_reflectivity: torch.Tensor | None
    complex_water_0: complex
    complex_ice_0: complex
    refl10cm: torch.Tensor | None
    refl10cm_from_melting_flag: bool | torch.Tensor
    path_id: str
    semi_sedi_flag: bool
    dt: float
    odt: float


@dataclass(frozen=True)
class TorchSegmentR2Result:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentR2Workspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class TorchSegmentSWorkspace:
    base: TorchSegmentR2Workspace | None
    diagnostics: Mapping[str, torch.Tensor]
    tempo_diags: Mapping[str, torch.Tensor]
    re_cloud: torch.Tensor | None  # None: radii skipped (run_s compute_effective_radii=False)
    re_ice: torch.Tensor | None
    re_snow: torch.Tensor | None
    path_id: str
    semi_sedi_flag: bool
    dt: float
    odt: float


@dataclass(frozen=True)
class TorchSegmentSResult:
    state: Mapping[str, torch.Tensor]
    workspace: TorchSegmentSWorkspace
    ncol: int
    nz: int


@dataclass(frozen=True)
class SemiLagrangianSedimentationGeometry:
    steps_tensor: torch.Tensor
    steps_float: torch.Tensor
    step_factor: torch.Tensor
    zi: torch.Tensor
    limiter_prefix: torch.Tensor


@dataclass(frozen=True)
class TorchPrefixSnapshot:
    segment_id: str
    result: Any


@dataclass(frozen=True)
class TorchPrefixResult:
    final: Any
    trace: tuple[TorchPrefixSnapshot, ...]


@dataclass(frozen=True)
class TorchFrozenTables:
    arrays: Mapping[str, torch.Tensor]
    file_records: tuple[Any, ...] = ()
    mmap_enabled: bool = False

    @property
    def resident_payload_bytes(self) -> int:
        return int(sum(array.numel() * array.element_size() for array in self.arrays.values()))

    @property
    def logical_payload_bytes(self) -> int:
        return self.resident_payload_bytes


@dataclass(frozen=True)
class PreparedTempoPolicy:
    """Static science and specialization policy for one prepared TEMPO instance."""

    dt: float = 1.0
    is_first_step: bool = True
    use_temperature: bool = True
    turn_off_micro: bool = False
    cloud_condensation: bool = True
    semi_sedi_flag: bool = True
    refl10cm_from_melting: bool | torch.Tensor = False
    max_hail_diameter_flag: bool = True
    max_sedimentation_substeps: int | None = None
    max_eulerian_sedimentation_substeps: int = DEFAULT_EULERIAN_SEDIMENTATION_SUBSTEPS
    max_long_dt_eulerian_sedimentation_substeps: int = DEFAULT_LONG_DT_EULERIAN_SEDIMENTATION_SUBSTEPS
    long_dt_eulerian_threshold: float = DEFAULT_LONG_DT_EULERIAN_THRESHOLD_SECONDS
    max_semi_sedimentation_substeps: int = DEFAULT_PRODUCTION_SEMI_SEDIMENTATION_SUBSTEPS
    max_snow_sedimentation_substeps: int = DEFAULT_SNOW_SEDIMENTATION_SUBSTEPS
    min_layer_depth: float | None = None
    max_sedimentation_fall_speed: float = DEFAULT_MAX_SEDIMENTATION_FALL_SPEED
    dynamic_sparse_diagnostics: bool = False


@dataclass(frozen=True)
class PreparedTempoRuntimeStatus:
    phase: str
    captured: bool
    graph_scope: str | None
    uncaptured_calls: int
    replay_calls: int
    compile_dispatch_fast_hits: int
    compile_dispatch_misses: int
    failure_stage: str | None = None
    failure_reason: str | None = None


def _prepared_state_signature(state: Mapping[str, torch.Tensor]) -> tuple[Any, ...]:
    if not state:
        raise ValueError("prepared TEMPO state is empty")
    signature: list[Any] = []
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"prepared TEMPO field {name!r} is not a torch.Tensor")
        signature.append(
            (
                name,
                tuple(value.shape),
                tuple(value.stride()),
                str(value.dtype),
                str(value.device),
            )
        )
    return tuple(signature)


def _prepared_result_tensors(result: TorchSegmentSResult) -> dict[str, torch.Tensor]:
    tensors = {f"state.{name}": value for name, value in sorted(result.state.items())}
    workspace = result.workspace
    tensors.update(
        {
            f"diagnostics.{name}": value
            for name, value in sorted(workspace.diagnostics.items())
        }
    )
    tensors.update(
        {
            f"tempo_diags.{name}": value
            for name, value in sorted(workspace.tempo_diags.items())
        }
    )
    tensors.update(
        {
            "re_cloud": workspace.re_cloud,
            "re_ice": workspace.re_ice,
            "re_snow": workspace.re_snow,
        }
    )
    return tensors


def _prepared_execution_policy_signature() -> tuple[Any, ...]:
    compile_flags = cuda_compile_island_flags()
    return (
        N_LIMITER_COMPILE_SCHEDULE,
        N_TENDENCY_LAYOUT_MODE,
        DIAGNOSTIC_COMPUTE_MODE,
        REGIONAL_EXECUTION_MODE,
        SEDIMENTATION_RECURRENCE_EXECUTION_MODE,
        SEMI_LAGRANGIAN_OVERLAP_MODE,
        SEMI_LAGRANGIAN_STREAMING_REDUCTION_ENABLED,
        SEMI_LAGRANGIAN_ORDERED_SCAN_CHUNK_SOURCES,
        SEMI_LAGRANGIAN_ORDERED_WINDOW_SOURCES,
        SEMI_LAGRANGIAN_GEOMETRY_MIN_SOURCE_WIDTH,
        SEMI_LAGRANGIAN_GEOMETRY_MAX_DESTINATION_WIDTH,
        STRICT_CUDA_COMPILE_ISLANDS,
        tuple(sorted(compile_flags.items())),
    )


class PreparedTempoRuntime:
    """Instance-scoped static TEMPO carrier with explicit CUDA Graph lifecycle.

    The class owns stable input buffers and prepared lookup tables.  Capture and
    replay never fall back to eager execution: a capture failure is recorded and
    raised.  Autodiff stays on the ordinary torch-native science path through
    :meth:`run_differentiable`; CUDA Graph replay is intentionally forward-only.
    """

    _BOUNDARIES = frozenset({"resident", "copy_in", "copy_in_out"})

    def __init__(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        tables: Any | None = None,
        policy: PreparedTempoPolicy | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.policy = policy or PreparedTempoPolicy()
        self._execution_policy_signature = _prepared_execution_policy_signature()
        self._input_signature = _prepared_state_signature(state)
        devices = {value.device for value in state.values()}
        if len(devices) != 1:
            raise ValueError("prepared TEMPO fields must share one device")
        self.device = next(iter(devices))
        self._source_requires_grad = any(value.requires_grad for value in state.values())
        self._inputs = {
            name: value.clone(memory_format=torch.contiguous_format)
            for name, value in state.items()
        }
        self._input_signature = _prepared_state_signature(self._inputs)
        self.tables = prepare_torch_tables(tables, self._inputs) if tables is not None else None
        selector = self.policy.refl10cm_from_melting
        if isinstance(selector, torch.Tensor):
            selector = selector.to(device=self.device).clone()
        self._refl10cm_from_melting = selector
        self._stream = stream
        if stream is not None and self.device.type != "cuda":
            raise ValueError("a CUDA stream requires CUDA-resident prepared inputs")
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_scope: str | None = None
        self._captured_q_result: TorchSegmentQResult | None = None
        self._captured_result: TorchSegmentSResult | None = None
        self._captured_outputs: dict[str, torch.Tensor] = {}
        self._host_outputs: dict[str, torch.Tensor] = {}
        self._phase = "prepared"
        self._failure_stage: str | None = None
        self._failure_reason: str | None = None
        self._uncaptured_calls = 0
        self._replay_calls = 0
        self._compile_dispatch = _PreparedCompileDispatch(callables={})

    @property
    def inputs(self) -> Mapping[str, torch.Tensor]:
        return self._inputs

    @property
    def captured_result(self) -> TorchSegmentSResult:
        if self._captured_result is None:
            raise RuntimeError("prepared TEMPO instance has no captured result")
        return self._captured_result

    @property
    def status(self) -> PreparedTempoRuntimeStatus:
        return PreparedTempoRuntimeStatus(
            phase=self._phase,
            captured=self._graph is not None,
            graph_scope=self._graph_scope,
            uncaptured_calls=self._uncaptured_calls,
            replay_calls=self._replay_calls,
            compile_dispatch_fast_hits=self._compile_dispatch.fast_hits,
            compile_dispatch_misses=self._compile_dispatch.misses,
            failure_stage=self._failure_stage,
            failure_reason=self._failure_reason,
        )

    @property
    def ownership_accounting(self) -> Mapping[str, int | str | None]:
        input_bytes = sum(value.numel() * value.element_size() for value in self._inputs.values())
        table_bytes = (
            self.tables.resident_payload_bytes
            if isinstance(self.tables, TorchFrozenTables)
            else 0
        )
        output_bytes = sum(
            value.numel() * value.element_size()
            for value in self._captured_outputs.values()
        )
        host_output_bytes = sum(
            value.numel() * value.element_size()
            for value in self._host_outputs.values()
        )
        return {
            "graph_scope": self._graph_scope,
            "input_field_count": len(self._inputs),
            "resident_input_bytes": int(input_bytes),
            "resident_timed_copy_in_event_count": 0,
            "resident_timed_copy_in_bytes": 0,
            "copy_in_event_count_per_call": len(self._inputs),
            "copy_in_bytes_per_call": int(input_bytes),
            "resident_table_bytes": int(table_bytes),
            "captured_output_tensor_count": len(self._captured_outputs),
            "captured_output_bytes": int(output_bytes),
            "copy_out_event_count_per_call": len(self._host_outputs),
            "copy_out_bytes_per_call": int(host_output_bytes),
            "compile_dispatch_callable_count": len(self._compile_dispatch.callables),
        }

    def _record_failure(self, stage: str, exc: BaseException) -> None:
        self._phase = "failed"
        self._failure_stage = stage
        message = str(exc).splitlines()[0] if str(exc) else ""
        self._failure_reason = f"{type(exc).__name__}: {message}"[:500]

    def _validate_execution_policy(self) -> None:
        current = _prepared_execution_policy_signature()
        if current != self._execution_policy_signature:
            raise RuntimeError(
                "prepared TEMPO execution policy changed after instance construction"
            )

    def _validate_source(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        allow_host_source: bool,
    ) -> None:
        if set(state) != set(self._inputs):
            missing = sorted(set(self._inputs) - set(state))
            extra = sorted(set(state) - set(self._inputs))
            raise ValueError(f"prepared TEMPO field mismatch: missing={missing}, extra={extra}")
        for name, destination in self._inputs.items():
            source = state[name]
            if not isinstance(source, torch.Tensor):
                raise TypeError(f"prepared TEMPO field {name!r} is not a torch.Tensor")
            if source.shape != destination.shape:
                raise ValueError(
                    f"prepared TEMPO shape mismatch for {name}: {tuple(source.shape)} != {tuple(destination.shape)}"
                )
            if source.dtype != destination.dtype:
                raise ValueError(
                    f"prepared TEMPO dtype mismatch for {name}: {source.dtype} != {destination.dtype}"
                )
            if source.device != destination.device:
                allowed = allow_host_source and source.device.type == "cpu" and destination.device.type == "cuda"
                if not allowed:
                    raise ValueError(
                        f"prepared TEMPO device mismatch for {name}: {source.device} != {destination.device}"
                    )

    def load_inputs(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        allow_host_source: bool = False,
        non_blocking: bool = False,
    ) -> None:
        self._validate_source(state, allow_host_source=allow_host_source)
        with torch.no_grad():
            for name, destination in self._inputs.items():
                destination.copy_(state[name], non_blocking=non_blocking)

    def _execute(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        use_prepared_dispatch: bool = True,
    ) -> TorchSegmentSResult:
        self._validate_execution_policy()
        policy = self.policy
        token = None
        if use_prepared_dispatch:
            token = _ACTIVE_PREPARED_COMPILE_DISPATCH.set(self._compile_dispatch)
        try:
            result = run_tempo_native(
                state,
                dt=policy.dt,
                is_first_step=policy.is_first_step,
                use_temperature=policy.use_temperature,
                tables=self.tables,
                turn_off_micro=policy.turn_off_micro,
                cloud_condensation=policy.cloud_condensation,
                semi_sedi_flag=policy.semi_sedi_flag,
                refl10cm_from_melting=self._refl10cm_from_melting,
                max_hail_diameter_flag=policy.max_hail_diameter_flag,
                max_sedimentation_substeps=policy.max_sedimentation_substeps,
                max_eulerian_sedimentation_substeps=policy.max_eulerian_sedimentation_substeps,
                max_long_dt_eulerian_sedimentation_substeps=policy.max_long_dt_eulerian_sedimentation_substeps,
                long_dt_eulerian_threshold=policy.long_dt_eulerian_threshold,
                max_semi_sedimentation_substeps=policy.max_semi_sedimentation_substeps,
                max_snow_sedimentation_substeps=policy.max_snow_sedimentation_substeps,
                min_layer_depth=policy.min_layer_depth,
                max_sedimentation_fall_speed=policy.max_sedimentation_fall_speed,
            )
        finally:
            if token is not None:
                _ACTIVE_PREPARED_COMPILE_DISPATCH.reset(token)
        if not isinstance(result, TorchSegmentSResult):
            raise RuntimeError("prepared TEMPO full execution did not return Segment S")
        return result

    def _execute_graph_core(self) -> TorchSegmentQResult:
        self._validate_execution_policy()
        policy = self.policy
        token = _ACTIVE_PREPARED_COMPILE_DISPATCH.set(self._compile_dispatch)
        try:
            result = run_tempo_native(
                self._inputs,
                dt=policy.dt,
                is_first_step=policy.is_first_step,
                use_temperature=policy.use_temperature,
                tables=self.tables,
                turn_off_micro=policy.turn_off_micro,
                cloud_condensation=policy.cloud_condensation,
                semi_sedi_flag=policy.semi_sedi_flag,
                refl10cm_from_melting=self._refl10cm_from_melting,
                max_hail_diameter_flag=policy.max_hail_diameter_flag,
                max_sedimentation_substeps=policy.max_sedimentation_substeps,
                max_eulerian_sedimentation_substeps=policy.max_eulerian_sedimentation_substeps,
                max_long_dt_eulerian_sedimentation_substeps=policy.max_long_dt_eulerian_sedimentation_substeps,
                long_dt_eulerian_threshold=policy.long_dt_eulerian_threshold,
                max_semi_sedimentation_substeps=policy.max_semi_sedimentation_substeps,
                max_snow_sedimentation_substeps=policy.max_snow_sedimentation_substeps,
                min_layer_depth=policy.min_layer_depth,
                max_sedimentation_fall_speed=policy.max_sedimentation_fall_speed,
                _prepared_stop_after_q=True,
            )
        finally:
            _ACTIVE_PREPARED_COMPILE_DISPATCH.reset(token)
        if not isinstance(result, TorchSegmentQResult):
            raise RuntimeError("prepared TEMPO graph core did not return Segment Q")
        return result

    def _finish_graph_core(self, q_result: TorchSegmentQResult) -> TorchSegmentSResult:
        self._validate_execution_policy()
        policy = self.policy
        sedimentation_substeps = _select_sedimentation_substeps(
            dt=policy.dt,
            semi_sedi_flag=policy.semi_sedi_flag,
            override=policy.max_sedimentation_substeps,
            eulerian_cap=policy.max_eulerian_sedimentation_substeps,
            long_dt_eulerian_cap=policy.max_long_dt_eulerian_sedimentation_substeps,
            long_dt_eulerian_threshold=policy.long_dt_eulerian_threshold,
            semi_cap=policy.max_semi_sedimentation_substeps,
            min_layer_depth=policy.min_layer_depth,
            max_fall_speed=policy.max_sedimentation_fall_speed,
        )
        token = _ACTIVE_PREPARED_COMPILE_DISPATCH.set(self._compile_dispatch)
        try:
            return _finish_tempo_native_from_q(
                q_result,
                dt=policy.dt,
                is_first_step=policy.is_first_step,
                tables=self.tables,
                turn_off_micro=policy.turn_off_micro,
                cloud_condensation=policy.cloud_condensation,
                semi_sedi_flag=policy.semi_sedi_flag,
                refl10cm_from_melting=self._refl10cm_from_melting,
                max_hail_diameter_flag=policy.max_hail_diameter_flag,
                sedimentation_substeps=sedimentation_substeps,
                copy=False,
            )
        finally:
            _ACTIVE_PREPARED_COMPILE_DISPATCH.reset(token)

    def run_uncaptured(
        self,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        allow_host_source: bool = False,
        non_blocking: bool = False,
    ) -> TorchSegmentSResult:
        if state is not None:
            self.load_inputs(
                state,
                allow_host_source=allow_host_source,
                non_blocking=non_blocking,
            )
        self._uncaptured_calls += 1
        try:
            return self._execute(self._inputs)
        except Exception as exc:
            self._record_failure("execute", exc)
            raise

    def run_differentiable(self, state: Mapping[str, torch.Tensor]) -> TorchSegmentSResult:
        """Run the same science path without copying leaves or capturing its graph."""
        self._validate_source(state, allow_host_source=False)
        self._uncaptured_calls += 1
        try:
            return self._execute(state, use_prepared_dispatch=False)
        except Exception as exc:
            self._record_failure("differentiable_execute", exc)
            raise

    def capture(self, *, warmup_calls: int = 3) -> TorchSegmentSResult:
        precondition_error: Exception | None = None
        if self.device.type != "cuda":
            precondition_error = RuntimeError("CUDA Graph capture requires CUDA-resident prepared inputs")
        elif warmup_calls < 1:
            precondition_error = ValueError("warmup_calls must be positive")
        elif self._source_requires_grad or any(
            value.requires_grad for value in self._inputs.values()
        ):
            precondition_error = RuntimeError("prepared CUDA Graph capture is forward-only")
        elif self._graph is not None:
            precondition_error = RuntimeError("prepared TEMPO instance is already captured")
        if precondition_error is not None:
            self._record_failure("capture_precondition", precondition_error)
            raise precondition_error
        stream = self._stream or torch.cuda.Stream(device=self.device)
        self._stream = stream
        partial_graph = self.policy.dynamic_sparse_diagnostics or DIAGNOSTIC_COMPUTE_MODE == "compact_active"
        try:
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(warmup_calls):
                    if partial_graph:
                        self._finish_graph_core(self._execute_graph_core())
                    else:
                        self._execute(self._inputs)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                if partial_graph:
                    q_result = self._execute_graph_core()
                else:
                    result = self._execute(self._inputs)
            torch.cuda.synchronize(self.device)
            if partial_graph:
                result = self._finish_graph_core(q_result)
        except Exception as exc:
            self._record_failure("capture", exc)
            raise RuntimeError(f"prepared TEMPO CUDA Graph capture failed: {self._failure_reason}") from exc
        self._graph = graph
        self._graph_scope = "B-Q" if partial_graph else "B-S"
        self._captured_q_result = q_result if partial_graph else None
        self._captured_result = result
        self._captured_outputs = _prepared_result_tensors(result)
        self._phase = "captured"
        return result

    def replay(
        self,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        boundary: str = "resident",
    ) -> TorchSegmentSResult | Mapping[str, torch.Tensor]:
        if boundary not in self._BOUNDARIES:
            raise ValueError(f"unknown prepared TEMPO ownership boundary: {boundary}")
        if state is not None and any(value.requires_grad for value in state.values()):
            if boundary != "resident":
                raise RuntimeError(
                    "differentiable TEMPO execution requires resident CUDA tensors"
                )
            return self.run_differentiable(state)
        if self._graph is None or self._captured_result is None:
            raise RuntimeError("prepared TEMPO instance must be captured before replay")
        try:
            self._validate_execution_policy()
            caller_stream = torch.cuda.current_stream(self.device)
            replay_stream = self._stream or caller_stream
            replay_stream.wait_stream(caller_stream)
            with torch.cuda.stream(replay_stream):
                if state is not None:
                    self.load_inputs(
                        state,
                        allow_host_source=boundary != "resident",
                        non_blocking=boundary != "resident",
                    )
                self._graph.replay()
                self._replay_calls += 1
                if self._graph_scope == "B-Q":
                    if self._captured_q_result is None:
                        raise RuntimeError("prepared TEMPO B-Q graph is missing its captured Q result")
                    self._captured_result = self._finish_graph_core(self._captured_q_result)
                replay_result: TorchSegmentSResult | Mapping[str, torch.Tensor]
                if boundary == "copy_in_out":
                    replay_result = self.copy_outputs_to_host(
                        self._captured_result,
                        non_blocking=True,
                    )
                else:
                    replay_result = self._captured_result
            caller_stream.wait_stream(replay_stream)
            return replay_result
        except Exception as exc:
            self._record_failure("replay", exc)
            raise RuntimeError(f"prepared TEMPO CUDA Graph replay failed: {self._failure_reason}") from exc

    def copy_outputs_to_host(
        self,
        result: TorchSegmentSResult,
        *,
        non_blocking: bool = True,
    ) -> Mapping[str, torch.Tensor]:
        outputs = _prepared_result_tensors(result)
        if not self._host_outputs:
            self._host_outputs = {
                name: torch.empty_like(
                    value,
                    device="cpu",
                    pin_memory=value.device.type == "cuda",
                )
                for name, value in outputs.items()
            }
        if set(outputs) != set(self._host_outputs):
            raise RuntimeError("prepared TEMPO output surface changed after preparation")
        for name, destination in self._host_outputs.items():
            source = outputs[name]
            if destination.shape != source.shape or destination.dtype != source.dtype:
                raise RuntimeError(f"prepared TEMPO output signature changed for {name}")
            destination.copy_(source, non_blocking=non_blocking)
        return self._host_outputs


def compact_s_prefix_result(result: TorchPrefixResult) -> TorchPrefixResult:
    """Drop retained B-S trace/base workspaces when only final S outputs are needed."""
    return TorchPrefixResult(
        final=compact_s_result(result.final),
        trace=(),
    )


def compact_s_result(final: TorchSegmentSResult) -> TorchSegmentSResult:
    """Return final Segment S outputs without the upstream workspace chain."""
    workspace = final.workspace
    if not isinstance(workspace, TorchSegmentSWorkspace):
        raise TypeError("compact_s_result requires a final Segment S result")
    compact_workspace = TorchSegmentSWorkspace(
        base=None,
        diagnostics=workspace.diagnostics,
        tempo_diags=workspace.tempo_diags,
        re_cloud=workspace.re_cloud,
        re_ice=workspace.re_ice,
        re_snow=workspace.re_snow,
        path_id=workspace.path_id,
        semi_sedi_flag=workspace.semi_sedi_flag,
        dt=workspace.dt,
        odt=workspace.odt,
    )
    return TorchSegmentSResult(
        state=final.state,
        workspace=compact_workspace,
        ncol=final.ncol,
        nz=final.nz,
    )


def compact_q_result(result: TorchSegmentQResult) -> TorchSegmentQResult:
    """Drop Q data that downstream R/S no longer consume."""
    workspace = result.workspace
    if not isinstance(workspace, TorchSegmentQWorkspace):
        raise TypeError("compact_q_result requires a Segment Q result")
    empty = torch.empty((0,), **_factory(workspace.temp))
    return TorchSegmentQResult(
        state=result.state,
        workspace=replace(
            workspace,
            base=None,
            tend_work=empty,
            tend={},
            tendencies={},
            fallspeeds={},
        ),
        ncol=result.ncol,
        nz=result.nz,
    )


def compact_o_result(result: TorchSegmentOResult) -> TorchSegmentOResult:
    """Drop the B-N workspace chain once O has denormalized P/Q inputs."""
    workspace = result.workspace
    if not isinstance(workspace, TorchSegmentOWorkspace):
        raise TypeError("compact_o_result requires a Segment O result")
    return TorchSegmentOResult(
        state=result.state,
        workspace=replace(workspace, base=None),
        ncol=result.ncol,
        nz=result.nz,
    )


def run_tempo_native(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = True,
    refl10cm_from_melting: bool | torch.Tensor = False,
    max_hail_diameter_flag: bool = True,
    max_sedimentation_substeps: int | None = None,
    max_eulerian_sedimentation_substeps: int = DEFAULT_EULERIAN_SEDIMENTATION_SUBSTEPS,
    max_long_dt_eulerian_sedimentation_substeps: int = DEFAULT_LONG_DT_EULERIAN_SEDIMENTATION_SUBSTEPS,
    long_dt_eulerian_threshold: float = DEFAULT_LONG_DT_EULERIAN_THRESHOLD_SECONDS,
    max_semi_sedimentation_substeps: int = DEFAULT_PRODUCTION_SEMI_SEDIMENTATION_SUBSTEPS,
    max_snow_sedimentation_substeps: int = DEFAULT_SNOW_SEDIMENTATION_SUBSTEPS,
    min_layer_depth: float | None = None,
    max_sedimentation_fall_speed: float = DEFAULT_MAX_SEDIMENTATION_FALL_SPEED,
    max_snow_sedimentation_fall_speed: float | None = None,
    radar_and_effective_radii: bool = True,
    copy: bool = False,
    _prepared_stop_after_q: bool = False,
) -> TorchSegmentSResult | TorchSegmentQResult:
    """Run the full torch-native TEMPO step as one production operation.

    Boundary-sized prefix functions remain available for the L1-L8 gate stack.
    This entry point is the benchmark/runtime path: it follows the natural
    B-S scientific flow and returns only final S outputs.

    ``min_layer_depth`` (m, the host grid's thinnest layer) raises each
    sedimentation loop to ``sedimentation_substep_bound`` when that exceeds
    the corpus cap, with fall-speed ceiling ``max_sedimentation_fall_speed``
    for rain and graupel and ``max_snow_sedimentation_fall_speed`` (default:
    the same) for snow. ``diagnostics[SEDIMENTATION_SUBSTEP_OVERFLOW]`` flags any
    column whose Fortran substep count exceeded the loop actually run, and the
    bound's overflow region (``SEDIMENTATION_BOUND_RAIN_GRAUPEL`` or
    ``SEDIMENTATION_BOUND_SNOW``) is set so a driver can rerun with a larger bound.
    ``radar_and_effective_radii=False`` skips the R2 reflectivity and the S effective
    radii (their diagnostics are then absent) for a caller that reads neither.
    """

    sedimentation_substeps = _select_sedimentation_substeps(
        dt=dt,
        semi_sedi_flag=semi_sedi_flag,
        override=max_sedimentation_substeps,
        eulerian_cap=max_eulerian_sedimentation_substeps,
        long_dt_eulerian_cap=max_long_dt_eulerian_sedimentation_substeps,
        long_dt_eulerian_threshold=long_dt_eulerian_threshold,
        semi_cap=max_semi_sedimentation_substeps,
        min_layer_depth=min_layer_depth,
        max_fall_speed=max_sedimentation_fall_speed,
    )
    snow_sedimentation_substeps = _production_snow_sedimentation_substeps(
        dt=dt,
        semi_sedi_flag=semi_sedi_flag,
        branch_substeps=sedimentation_substeps,
        override=max_sedimentation_substeps,
        snow_cap=max_snow_sedimentation_substeps,
        min_layer_depth=min_layer_depth,
        max_fall_speed=(max_sedimentation_fall_speed if max_snow_sedimentation_fall_speed is None
                        else max_snow_sedimentation_fall_speed),
    )

    b_result = run_b(state, use_temperature=use_temperature, is_first_step=is_first_step, copy=copy)
    c_result = run_c(b_result, dt=dt, is_first_step=is_first_step, copy=copy)
    del b_result
    d_result = run_d(c_result, dt=dt, is_first_step=is_first_step, copy=copy)
    del c_result
    e_result = run_e(d_result, dt=dt, is_first_step=is_first_step, copy=copy)
    del d_result
    f_result = run_f(e_result, dt=dt, is_first_step=is_first_step, copy=copy)
    del e_result
    g_result = run_g(f_result, dt=dt, is_first_step=is_first_step, copy=copy)
    del f_result
    h_result = run_h(
        g_result,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    del g_result
    i_result = run_i(
        h_result,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    del h_result
    j_result = run_j(
        i_result,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    del i_result
    k_result = run_k(
        j_result,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    del j_result
    l_result = run_l(
        k_result,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    del k_result
    m_result = run_m(
        l_result,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    del l_result
    n_result = run_n(
        m_result,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        production_sedimentation_moments=True,
        copy=copy,
    )
    del m_result
    o_result = run_o(
        n_result,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        copy=copy,
    )
    del n_result
    o_result = compact_o_result(o_result)
    if semi_sedi_flag:
        p4_result = run_p2_p4(
            o_result,
            dt=dt,
            is_first_step=is_first_step,
            tables=tables,
            turn_off_micro=turn_off_micro,
            cloud_condensation=cloud_condensation,
            max_sedimentation_substeps=sedimentation_substeps,
            copy=copy,
        )
        p5_input: TorchSegmentP3Result | TorchSegmentP4Result = p4_result
        del p4_result
    else:
        p1_result = run_p1(
            o_result,
            dt=dt,
            is_first_step=is_first_step,
            tables=tables,
            turn_off_micro=turn_off_micro,
            cloud_condensation=cloud_condensation,
            semi_sedi_flag=False,
            max_sedimentation_substeps=sedimentation_substeps,
            copy=copy,
        )
        p3_result = run_p3(
            p1_result,
            dt=dt,
            is_first_step=is_first_step,
            tables=tables,
            turn_off_micro=turn_off_micro,
            cloud_condensation=cloud_condensation,
            semi_sedi_flag=False,
            max_sedimentation_substeps=sedimentation_substeps,
            copy=copy,
        )
        del p1_result
        p5_input = p3_result
        del p3_result
    del o_result
    p5_result = run_p5(
        p5_input,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=sedimentation_substeps,
        max_snow_sedimentation_substeps=snow_sedimentation_substeps,
        compact_base=True,
        copy=copy,
    )
    del p5_input
    q_result = run_q(
        p5_result,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=sedimentation_substeps,
        compact_base=True,
        copy=copy,
    )
    del p5_result
    q_result = compact_q_result(q_result)
    if _prepared_stop_after_q:
        return q_result
    return _finish_tempo_native_from_q(
        q_result,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        refl10cm_from_melting=refl10cm_from_melting,
        max_hail_diameter_flag=max_hail_diameter_flag,
        sedimentation_substeps=sedimentation_substeps,
        copy=copy,
        radar_and_effective_radii=radar_and_effective_radii,
    )


def _finish_tempo_native_from_q(
    q_result: TorchSegmentQResult,
    *,
    dt: float,
    is_first_step: bool,
    tables: Any | None,
    turn_off_micro: bool,
    cloud_condensation: bool,
    semi_sedi_flag: bool,
    refl10cm_from_melting: bool | torch.Tensor,
    max_hail_diameter_flag: bool,
    sedimentation_substeps: int,
    copy: bool,
    radar_and_effective_radii: bool = True,
) -> TorchSegmentSResult:
    """Finish the single production science path after the graph-stable Q boundary."""
    r1_result = run_r1(
        q_result,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        refl10cm_from_melting=False,
        max_hail_diameter_flag=max_hail_diameter_flag,
        max_sedimentation_substeps=sedimentation_substeps,
        defer_reflectivity_to_r2=True,
        copy=copy,
    )
    del q_result
    r2_result = run_r2(
        r1_result,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        refl10cm_from_melting=refl10cm_from_melting,
        max_hail_diameter_flag=max_hail_diameter_flag,
        max_sedimentation_substeps=sedimentation_substeps,
        compute_reflectivity=radar_and_effective_radii,
        copy=copy,
    )
    del r1_result
    s_result = run_s(
        r2_result,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        refl10cm_from_melting=refl10cm_from_melting,
        max_hail_diameter_flag=max_hail_diameter_flag,
        max_sedimentation_substeps=sedimentation_substeps,
        compute_effective_radii=radar_and_effective_radii,
        copy=copy,
    )
    del r2_result
    compact = compact_s_result(s_result)
    del s_result
    return compact


def to_torch_state(
    state: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
    requires_grad_fields: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for name, values in state.items():
        if isinstance(values, torch.Tensor):
            tensor = values.to(device=device) if device is not None else values
        else:
            array = np.asarray(values)
            dtype = torch.float64 if array.dtype.kind == "f" else None
            tensor = torch.as_tensor(array, dtype=dtype, device=device)
        if name in requires_grad_fields:
            tensor = tensor.clone().requires_grad_(True)
        converted[name] = tensor
    return converted


def run_prefix_b_to_g(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    copy: bool = False,
) -> TorchPrefixResult:
    trace: list[TorchPrefixSnapshot] = []

    b_result = run_b(state, use_temperature=use_temperature, is_first_step=is_first_step, copy=copy)
    trace.append(TorchPrefixSnapshot("B", b_result))
    c_result = run_c(b_result, dt=dt, is_first_step=is_first_step, copy=copy)
    trace.append(TorchPrefixSnapshot("C", c_result))
    d_result = run_d(c_result, dt=dt, is_first_step=is_first_step, copy=copy)
    trace.append(TorchPrefixSnapshot("D", d_result))
    e_result = run_e(d_result, dt=dt, is_first_step=is_first_step, copy=copy)
    trace.append(TorchPrefixSnapshot("E", e_result))
    f_result = run_f(e_result, dt=dt, is_first_step=is_first_step, copy=copy)
    trace.append(TorchPrefixSnapshot("F", f_result))
    g_result = run_g(f_result, dt=dt, is_first_step=is_first_step, copy=copy)
    trace.append(TorchPrefixSnapshot("G", g_result))
    return TorchPrefixResult(final=g_result, trace=tuple(trace))


def run_prefix_b_to_h(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = run_prefix_b_to_g(
        state,
        dt=dt,
        is_first_step=is_first_step,
        use_temperature=use_temperature,
        copy=copy,
    )
    h_result = run_h(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    return TorchPrefixResult(
        final=h_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("H", h_result)),
    )


def run_prefix_b_to_i(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = run_prefix_b_to_h(
        state,
        dt=dt,
        is_first_step=is_first_step,
        use_temperature=use_temperature,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    i_result = run_i(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    return TorchPrefixResult(
        final=i_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("I", i_result)),
    )


def run_prefix_b_to_j(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = run_prefix_b_to_i(
        state,
        dt=dt,
        is_first_step=is_first_step,
        use_temperature=use_temperature,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    j_result = run_j(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    return TorchPrefixResult(
        final=j_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("J", j_result)),
    )


def run_prefix_b_to_k(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = run_prefix_b_to_j(
        state,
        dt=dt,
        is_first_step=is_first_step,
        use_temperature=use_temperature,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    k_result = run_k(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    return TorchPrefixResult(
        final=k_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("K", k_result)),
    )


def run_prefix_b_to_l(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = run_prefix_b_to_k(
        state,
        dt=dt,
        is_first_step=is_first_step,
        use_temperature=use_temperature,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    l_result = run_l(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    return TorchPrefixResult(
        final=l_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("L", l_result)),
    )


def run_prefix_b_to_m(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = run_prefix_b_to_l(
        state,
        dt=dt,
        is_first_step=is_first_step,
        use_temperature=use_temperature,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    m_result = run_m(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    return TorchPrefixResult(
        final=m_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("M", m_result)),
    )


def run_prefix_b_to_n(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = run_prefix_b_to_m(
        state,
        dt=dt,
        is_first_step=is_first_step,
        use_temperature=use_temperature,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    n_result = run_n(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    return TorchPrefixResult(
        final=n_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("N", n_result)),
    )


def run_prefix_b_to_o(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = run_prefix_b_to_n(
        state,
        dt=dt,
        is_first_step=is_first_step,
        use_temperature=use_temperature,
        tables=tables,
        turn_off_micro=turn_off_micro,
        copy=copy,
    )
    o_result = run_o(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        copy=copy,
    )
    return TorchPrefixResult(
        final=o_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("O", o_result)),
    )


def run_prefix_b_to_p1(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = False,
    max_sedimentation_substeps: int | None = None,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = run_prefix_b_to_o(
        state,
        dt=dt,
        is_first_step=is_first_step,
        use_temperature=use_temperature,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        copy=copy,
    )
    p1_result = run_p1(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
        copy=copy,
    )
    return TorchPrefixResult(
        final=p1_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("P1", p1_result)),
    )


def run_prefix_b_to_p3(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = False,
    max_sedimentation_substeps: int | None = None,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = run_prefix_b_to_p1(
        state,
        dt=dt,
        is_first_step=is_first_step,
        use_temperature=use_temperature,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
        copy=copy,
    )
    p3_result = run_p3(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
        copy=copy,
    )
    return TorchPrefixResult(
        final=p3_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("P3", p3_result)),
    )


def run_prefix_b_to_p2(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = True,
    max_sedimentation_substeps: int | None = None,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = run_prefix_b_to_o(
        state,
        dt=dt,
        is_first_step=is_first_step,
        use_temperature=use_temperature,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        copy=copy,
    )
    p2_result = run_p2(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
        copy=copy,
    )
    return TorchPrefixResult(
        final=p2_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("P2", p2_result)),
    )


def run_prefix_b_to_p4(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = True,
    max_sedimentation_substeps: int | None = None,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = run_prefix_b_to_p2(
        state,
        dt=dt,
        is_first_step=is_first_step,
        use_temperature=use_temperature,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
        copy=copy,
    )
    p4_result = run_p4(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
        copy=copy,
    )
    return TorchPrefixResult(
        final=p4_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("P4", p4_result)),
    )


def run_prefix_b_to_p5(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = False,
    max_sedimentation_substeps: int | None = None,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = (
        run_prefix_b_to_p4(
            state,
            dt=dt,
            is_first_step=is_first_step,
            use_temperature=use_temperature,
            tables=tables,
            turn_off_micro=turn_off_micro,
            cloud_condensation=cloud_condensation,
            semi_sedi_flag=True,
            max_sedimentation_substeps=max_sedimentation_substeps,
            copy=copy,
        )
        if semi_sedi_flag
        else run_prefix_b_to_p3(
            state,
            dt=dt,
            is_first_step=is_first_step,
            use_temperature=use_temperature,
            tables=tables,
            turn_off_micro=turn_off_micro,
            cloud_condensation=cloud_condensation,
            semi_sedi_flag=False,
            max_sedimentation_substeps=max_sedimentation_substeps,
            copy=copy,
        )
    )
    p5_result = run_p5(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
        copy=copy,
    )
    return TorchPrefixResult(
        final=p5_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("P5", p5_result)),
    )


def run_prefix_b_to_q(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = True,
    max_sedimentation_substeps: int | None = None,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = run_prefix_b_to_p5(
        state,
        dt=dt,
        is_first_step=is_first_step,
        use_temperature=use_temperature,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
        copy=copy,
    )
    q_result = run_q(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
        copy=copy,
    )
    return TorchPrefixResult(
        final=q_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("Q", q_result)),
    )


def run_prefix_b_to_r1(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = True,
    refl10cm_from_melting: bool = False,
    max_hail_diameter_flag: bool = True,
    max_sedimentation_substeps: int | None = None,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = run_prefix_b_to_q(
        state,
        dt=dt,
        is_first_step=is_first_step,
        use_temperature=use_temperature,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
        copy=copy,
    )
    r1_result = run_r1(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        refl10cm_from_melting=refl10cm_from_melting,
        max_hail_diameter_flag=max_hail_diameter_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
        copy=copy,
    )
    return TorchPrefixResult(
        final=r1_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("R1", r1_result)),
    )


def run_prefix_b_to_r2(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = True,
    refl10cm_from_melting: bool | torch.Tensor = False,
    max_hail_diameter_flag: bool = True,
    max_sedimentation_substeps: int | None = None,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = run_prefix_b_to_r1(
        state,
        dt=dt,
        is_first_step=is_first_step,
        use_temperature=use_temperature,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        refl10cm_from_melting=False,
        max_hail_diameter_flag=max_hail_diameter_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
        copy=copy,
    )
    r2_result = run_r2(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        refl10cm_from_melting=refl10cm_from_melting,
        max_hail_diameter_flag=max_hail_diameter_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
        copy=copy,
    )
    return TorchPrefixResult(
        final=r2_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("R2", r2_result)),
    )


def run_prefix_b_to_s(
    state: Mapping[str, torch.Tensor],
    *,
    dt: float = 1.0,
    is_first_step: bool = True,
    use_temperature: bool = True,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = True,
    refl10cm_from_melting: bool | torch.Tensor = False,
    max_hail_diameter_flag: bool = True,
    max_sedimentation_substeps: int | None = None,
    copy: bool = False,
) -> TorchPrefixResult:
    prefix = run_prefix_b_to_r2(
        state,
        dt=dt,
        is_first_step=is_first_step,
        use_temperature=use_temperature,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        refl10cm_from_melting=refl10cm_from_melting,
        max_hail_diameter_flag=max_hail_diameter_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
        copy=copy,
    )
    s_result = run_s(
        prefix.final,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        refl10cm_from_melting=refl10cm_from_melting,
        max_hail_diameter_flag=max_hail_diameter_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
        copy=copy,
    )
    return TorchPrefixResult(
        final=s_result,
        trace=(*prefix.trace, TorchPrefixSnapshot("S", s_result)),
    )


def run_b(
    state: Mapping[str, torch.Tensor],
    *,
    use_temperature: bool = True,
    is_first_step: bool = True,
    copy: bool = False,
) -> TorchSegmentBResult:
    staged = dict(state)
    _require_checkpoint_fields(staged, "B")
    ncol, nz = _validate_state(staged, CHECKPOINT_FIELDS, "B")
    if not use_temperature:
        if "th" not in staged or "pii" not in staged:
            raise KeyError("Segment B theta mode requires th and pii fields")
        staged["t"] = staged["th"] * staged["pii"]
    if copy:
        staged = {name: _copy_tensor(values) for name, values in staged.items()}
    return TorchSegmentBResult(
        state=staged,
        ncol=ncol,
        nz=nz,
        use_temperature=use_temperature,
        is_first_step=is_first_step,
    )


def run_c(
    state: Mapping[str, torch.Tensor] | TorchSegmentBResult,
    *,
    dt: float = 1.0,
    is_first_step: bool | None = None,
    copy: bool = False,
) -> TorchSegmentCResult:
    input_state = _state_mapping(state)
    _require_checkpoint_fields(input_state, "C")
    ncol, nz = _validate_state(input_state, CHECKPOINT_FIELDS, "C")
    if dt == 0.0:
        raise ZeroDivisionError("Segment C dt must be nonzero")
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    profile_shape = (ncol, nz)
    reference = input_state["t"]
    tend = _zero_tensor_mapping(profile_shape, reference, constants.TEND_WORK_FIELDS)
    tend_work, tend = _contiguous_tend_work_tensor(tend)
    tendencies = _zero_tensor_mapping(profile_shape, reference, constants.TENDENCY_FIELDS)
    snow_moments = _zero_tensor_mapping(profile_shape, reference, constants.SNOW_MOMENT_FIELDS)
    zero_fallspeed = torch.zeros((1, 1), **_factory(reference)).expand(ncol, nz + 1)
    fallspeeds = {name: zero_fallspeed for name in constants.FALLSPEED_FIELDS}
    zero_diagnostic = torch.zeros((1,), **_factory(reference)).expand(ncol)
    diagnostics = {name: zero_diagnostic for name in constants.PRECIP_DIAG_FIELDS}

    temp = input_state["t"]
    pres = input_state["p"]
    qv = torch.clamp_min(input_state["qv"], _scalar(constants.MIN_QV, reference))
    rho = (
        _scalar(constants.ROVERRV, reference)
        * pres
        / (_scalar(constants.RDRY, reference) * temp * (qv + _scalar(constants.ROVERRV, reference)))
    )
    false_tensor = torch.zeros(profile_shape, dtype=torch.bool, device=reference.device)
    workspace = TorchSegmentCWorkspace(
        tend_work=tend_work,
        tend=tend,
        tendencies=tendencies,
        snow_moments=snow_moments,
        fallspeeds=fallspeeds,
        vtboost=torch.ones(profile_shape, **_factory(reference)),
        temp=temp,
        pres=pres,
        qv=qv,
        rho=rho,
        diagnostics=diagnostics,
        do_micro=false_tensor,
        supersaturated=false_tensor,
        local_first_step=bool(is_first_step) if is_first_step is not None else False,
        dt=float(dt),
        odt=1.0 / float(dt),
    )
    return TorchSegmentCResult(result_state, workspace, ncol, nz)


def run_d(
    state: Mapping[str, torch.Tensor] | TorchSegmentCResult,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    nwfa1d: torch.Tensor | None = None,
    nifa1d: torch.Tensor | None = None,
    copy: bool = False,
) -> TorchSegmentDResult:
    c_result = _segment_c_result(state, dt=dt, is_first_step=is_first_step)
    ncol, nz = c_result.ncol, c_result.nz
    input_state = c_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = c_result.workspace.dt if dt is None else float(dt)
    local_first_step = c_result.workspace.local_first_step if is_first_step is None else bool(is_first_step)
    water_1d = _optional_profile("D", "nwfa1d", nwfa1d, input_state, ("nwfa1d", "nwfa"), (ncol, nz))
    ice_1d = _optional_profile("D", "nifa1d", nifa1d, input_state, ("nifa1d", "nifa"), (ncol, nz))
    dz = input_state["dz"]
    base = c_result.workspace
    nwfaten = base.tendencies["nwfaten"]
    nifaten = base.tendencies["nifaten"]
    # The analytic profiles are only observable when the host omits aerosol
    # fields, or on the first call when an all-zero host column must be seeded.
    # Avoid their cumsum/exp corridor on ordinary host-provided subsequent calls.
    nwfa_init = (
        init_water_friendly_aerosols(dz)
        if water_1d is None or local_first_step
        else None
    )
    nifa_init = (
        init_ice_friendly_aerosols(dz)
        if ice_1d is None or local_first_step
        else None
    )
    nwfa = nwfa_init if nwfa_init is not None and water_1d is None else _scalar(0.0, base.rho)
    nifa = nifa_init if nifa_init is not None and ice_1d is None else _scalar(0.0, base.rho)
    if local_first_step and water_1d is not None:
        assert nwfa_init is not None
        empty_water_columns = torch.sum(water_1d, dim=1, keepdim=True) < _scalar(constants.EPS, base.rho)
        nwfa = torch.where(empty_water_columns, nwfa_init, nwfa)
    if local_first_step and ice_1d is not None:
        assert nifa_init is not None
        empty_ice_columns = torch.sum(ice_1d, dim=1, keepdim=True) < _scalar(constants.EPS, base.rho)
        nifa = torch.where(empty_ice_columns, nifa_init, nifa)
    nwfa, nifa = aerosol_check_and_update(
        rho=base.rho,
        nwfa=nwfa,
        nifa=nifa,
        nwfaten=nwfaten,
        nifaten=nifaten,
        dt=resolved_dt,
        nwfa1d=water_1d,
        nifa1d=ice_1d,
    )
    workspace = TorchSegmentDWorkspace(
        base=base,
        nwfa=nwfa,
        nifa=nifa,
        nwfaten=nwfaten,
        nifaten=nifaten,
        nwfa1d=water_1d,
        nifa1d=ice_1d,
        nwfa_present=water_1d is not None,
        nifa_present=ice_1d is not None,
        local_first_step=local_first_step,
        dt=resolved_dt,
    )
    return TorchSegmentDResult(result_state, workspace, ncol, nz)


def init_water_friendly_aerosols(dz1d: torch.Tensor) -> torch.Tensor:
    return _init_aerosol_profile(dz1d, n0=constants.NACCN0, n1=constants.NACCN1)


def init_ice_friendly_aerosols(dz1d: torch.Tensor) -> torch.Tensor:
    return _init_aerosol_profile(dz1d, n0=constants.NAIN0, n1=constants.NAIN1)


def aerosol_check_and_update(
    *,
    rho: torch.Tensor,
    nwfa: torch.Tensor,
    nifa: torch.Tensor,
    nwfaten: torch.Tensor,
    nifaten: torch.Tensor,
    dt: float,
    nwfa1d: torch.Tensor | None = None,
    nifa1d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if TENSOR_COMPILE_ISLAND_CUDA and rho.device.type in TENSOR_COMPILE_ISLAND_DEVICES:
        return _run_tensor_compile_island(
            "aerosol_check_and_update",
            _aerosol_check_and_update_kernel,
            rho,
            rho,
            nwfa,
            nifa,
            nwfaten,
            nifaten,
            nwfa if nwfa1d is None else nwfa1d,
            nifa if nifa1d is None else nifa1d,
            float(dt),
            nwfa1d is not None,
            nifa1d is not None,
        )
    water = nwfa
    ice = nifa
    if nwfa1d is not None:
        water = (nwfa1d + nwfaten * dt) * rho
    if nifa1d is not None:
        ice = (nifa1d + nifaten * dt) * rho
    water = _max_first(
        _scalar(constants.NWFA_DEFAULT, rho) * rho,
        _min_first(_scalar(constants.AERO_MAX, rho) * rho, water),
    )
    ice = _max_first(
        _scalar(constants.NIFA_DEFAULT, rho) * rho,
        _min_first(_scalar(constants.AERO_MAX, rho) * rho, ice),
    )
    return water, ice


def _aerosol_check_and_update_kernel(
    rho: torch.Tensor,
    nwfa: torch.Tensor,
    nifa: torch.Tensor,
    nwfaten: torch.Tensor,
    nifaten: torch.Tensor,
    nwfa1d: torch.Tensor,
    nifa1d: torch.Tensor,
    dt: float,
    update_water: bool,
    update_ice: bool,
    _nwfa_default: float = float(constants.NWFA_DEFAULT),
    _nifa_default: float = float(constants.NIFA_DEFAULT),
    _aero_max: float = float(constants.AERO_MAX),
) -> tuple[torch.Tensor, torch.Tensor]:
    water = (nwfa1d + nwfaten * dt) * rho if update_water else nwfa
    ice = (nifa1d + nifaten * dt) * rho if update_ice else nifa
    water = torch.maximum(_nwfa_default * rho, torch.minimum(_aero_max * rho, water))
    ice = torch.maximum(_nifa_default * rho, torch.minimum(_aero_max * rho, ice))
    return water, ice


def run_e(
    state: Mapping[str, torch.Tensor] | TorchSegmentDResult,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    copy: bool = False,
) -> TorchSegmentEResult:
    d_result = _segment_d_result(state, dt=dt, is_first_step=is_first_step)
    input_state = d_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = d_result.workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment E dt must be nonzero")
    odt = 1.0 / resolved_dt
    base = d_result.workspace.base
    tendencies = _tensor_mapping(base.tendencies, copy=copy)
    rain = rain_check_and_update(
        rho=base.rho,
        qr1d=input_state["qr"],
        nr1d=input_state["nr"],
        qrten=tendencies["qrten"],
        nrten=tendencies["nrten"],
        dt=resolved_dt,
        odt=odt,
    )
    ice = ice_check_and_update(
        rho=base.rho,
        qi1d=input_state["qi"],
        ni1d=input_state["ni"],
        qiten=tendencies["qiten"],
        niten=tendencies["niten"],
        dt=resolved_dt,
        odt=odt,
    )
    snow = snow_check_and_update(
        rho=base.rho,
        qs1d=input_state["qs"],
        qsten=tendencies["qsten"],
        dt=resolved_dt,
        odt=odt,
    )
    snow_moment_values = compute_snow_moments(
        rs=snow["rs"],
        temp=base.temp,
        l_qs=snow["l_qs"],
        template=base.snow_moments,
    )
    result_state.update(
        {
            "qr": rain["qr1d"],
            "nr": rain["nr1d"],
            "qi": ice["qi1d"],
            "ni": ice["ni1d"],
            "qs": snow["qs1d"],
        }
    )
    tendencies.update(
        {
            "qrten": rain["qrten"],
            "nrten": rain["nrten"],
            "qiten": ice["qiten"],
            "niten": ice["niten"],
            "qsten": snow["qsten"],
        }
    )
    workspace = TorchSegmentEWorkspace(
        base=d_result.workspace,
        l_qr=rain["l_qr"],
        l_qi=ice["l_qi"],
        l_qs=snow["l_qs"],
        rr=rain["rr"],
        ri=ice["ri"],
        rs=snow["rs"],
        nr=rain["nr"],
        ni=ice["ni"],
        ilamr=rain["ilamr"],
        ilami=ice["ilami"],
        mvd_r=rain["mvd_r"],
        tendencies=tendencies,
        snow_moments=snow_moment_values,
        dt=resolved_dt,
        odt=odt,
    )
    return TorchSegmentEResult(result_state, workspace, d_result.ncol, d_result.nz)


def rain_check_and_update(
    *,
    rho: torch.Tensor,
    qr1d: torch.Tensor,
    nr1d: torch.Tensor,
    qrten: torch.Tensor,
    nrten: torch.Tensor,
    dt: float,
    odt: float,
) -> dict[str, torch.Tensor]:
    values = _run_tensor_compile_island(
        "rain_check_and_update",
        _rain_check_and_update_kernel,
        rho,
        rho,
        qr1d,
        nr1d,
        qrten,
        nrten,
        float(dt),
        float(odt),
        _tensor(constants.BM_R, rho),
    )
    names = ("l_qr", "qr1d", "nr1d", "rr", "nr", "qrten", "nrten", "ilamr", "mvd_r")
    return dict(zip(names, values, strict=True))


def _rain_check_and_update_kernel(rho: torch.Tensor, qr_old: torch.Tensor, nr_old: torch.Tensor, qrt_old: torch.Tensor, nrt_old: torch.Tensor, dt: float, odt: float, rain_power: torch.Tensor, _const_segment_e_R1=constants.R1, _const_segment_e_R2=constants.R2, _const_segment_e_MU_R=constants.MU_R, _const_SCALAR_CRG_1=_SCALAR_CRG[1], _const_segment_e_ORG3=constants.ORG3, _const_segment_e_AM_R=constants.AM_R, _const_SCALAR_CRG_2=_SCALAR_CRG[2], _const_segment_e_ORG2=constants.ORG2, _const_segment_e_OBMR=constants.OBMR, _const_segment_e_D0R_MAX=constants.D0R_MAX, _const_segment_e_D0R=constants.D0R) -> tuple[torch.Tensor, ...]:
    active = qr_old + qrt_old * dt > float(_const_segment_e_R1)
    qr_active = qr_old + qrt_old * dt
    rr_active = qr_active * rho
    raw_nr = (nr_old + nrt_old * dt) * rho
    nr0 = torch.clamp_min(raw_nr, float(_const_segment_e_R2))
    hit0 = nr0 <= float(_const_segment_e_R2)
    lam_hit0 = float(3.0 + _const_segment_e_MU_R + 0.672) / 0.001
    # Retain the source division after the products; number-limit branches amplify reassociation.
    nr_hit0 = float(_const_SCALAR_CRG_1 * _const_segment_e_ORG3) * rr_active * (lam_hit0 * lam_hit0 * lam_hit0) / float(_const_segment_e_AM_R)
    nr1 = torch.where(hit0, nr_hit0, nr0)
    raw_ratio = float(_const_segment_e_AM_R * _const_SCALAR_CRG_2 * _const_segment_e_ORG2) * nr1 / torch.where(
        active, rr_active, torch.ones_like(rr_active)
    )
    lam1 = working_power(torch.where(active, raw_ratio, torch.ones_like(raw_ratio)), float(_const_segment_e_OBMR))
    mvd1 = torch.div(float(3.0 + _const_segment_e_MU_R + 0.672), lam1)
    high = mvd1 > float(_const_segment_e_D0R_MAX)
    low = ~high & (mvd1 < float(_const_segment_e_D0R * 0.75))
    mvd_limit = torch.clamp(mvd1, min=float(_const_segment_e_D0R * 0.75), max=float(_const_segment_e_D0R_MAX))
    lam_limit = torch.div(float(3.0 + _const_segment_e_MU_R + 0.672), mvd_limit)
    nr_limit = float(_const_SCALAR_CRG_1 * _const_segment_e_ORG3) * rr_active * torch.pow(lam_limit, rain_power) / float(_const_segment_e_AM_R)
    hit_limit = hit0 | high | low
    nr_active = torch.where(high | low, nr_limit, nr1)
    lam_active = torch.where(high | low, lam_limit, lam1)
    mvd_active = torch.where(high | low, mvd_limit, mvd1)
    nrt_active = torch.where(hit_limit, (nr_active / rho - nr_old) * odt, nrt_old)
    # BM_R is a REAL source operand, not an integer multiplication shortcut.
    nr1_active = float(_const_SCALAR_CRG_1 * _const_segment_e_ORG3) * qr_active * torch.pow(lam_active, rain_power) / float(_const_segment_e_AM_R)
    return (active, torch.where(active, qr_active, 0.0), torch.where(active, nr1_active, 0.0), torch.where(active, rr_active, float(_const_segment_e_R1)), torch.where(active, nr_active, float(_const_segment_e_R2)), torch.where(active, qrt_old, -qr_old * odt), torch.where(active, nrt_active, -nr_old * odt), torch.where(active, 1.0 / lam_active, 0.0), torch.where(active, mvd_active, float(_const_segment_e_D0R)))


def ice_check_and_update(
    *,
    rho: torch.Tensor,
    qi1d: torch.Tensor,
    ni1d: torch.Tensor,
    qiten: torch.Tensor,
    niten: torch.Tensor,
    dt: float,
    odt: float,
) -> dict[str, torch.Tensor]:
    values = _run_tensor_compile_island(
        "ice_check_and_update",
        _ice_check_and_update_kernel,
        rho,
        rho,
        qi1d,
        ni1d,
        qiten,
        niten,
        float(dt),
        float(odt),
        _tensor(constants.BM_I, rho),
    )
    names = ("l_qi", "qi1d", "ni1d", "ri", "ni", "qiten", "niten", "ilami")
    return dict(zip(names, values, strict=True))


def _ice_check_and_update_kernel(rho: torch.Tensor, qi_old: torch.Tensor, ni_old: torch.Tensor, qit_old: torch.Tensor, nit_old: torch.Tensor, dt: float, odt: float, ice_power: torch.Tensor, _const_segment_e_R1=constants.R1, _const_segment_e_R2=constants.R2, _const_SCALAR_CIE_1=_SCALAR_CIE[1], _const_segment_e_MAX_NI=constants.MAX_NI, _const_SCALAR_CIG_0=_SCALAR_CIG[0], _const_segment_e_OIG2=constants.OIG2, _const_segment_e_AM_I=constants.AM_I, _const_SCALAR_CIG_1=_SCALAR_CIG[1], _const_segment_e_OIG1=constants.OIG1, _const_segment_e_OBMI=constants.OBMI, _const_segment_e_BM_I=constants.BM_I, _const_segment_e_MU_I=constants.MU_I, _const_segment_e_D0S=constants.D0S) -> tuple[torch.Tensor, ...]:
    active = qi_old + qit_old * dt > float(_const_segment_e_R1)
    qi_active = qi_old + qit_old * dt
    ri_active = qi_active * rho
    # Fortran ice_check_and_update (module_mp_tempo_main.F90:1066-1067, 1080, 1092-1093) uses the unfloored
    # ri = (qi+qiten*dt)*rho and qi on active points, even when ri < R1 (rho < 1). The
    # guard only replaces inactive elements, keeping masked branches finite for autograd.
    ri_safe = torch.where(active, ri_active, float(_const_segment_e_R1))
    qi_safe = torch.where(active, qi_active, float(_const_segment_e_R1))
    raw_ni = (ni_old + nit_old * dt) * rho
    ni0 = torch.clamp_min(raw_ni, float(_const_segment_e_R2))
    hit0 = ni0 <= float(_const_segment_e_R2)
    lami_min = float(_const_SCALAR_CIE_1 / 5e-06)
    ni_hit0 = torch.clamp_max(float(_const_SCALAR_CIG_0 * _const_segment_e_OIG2) * ri_safe / float(_const_segment_e_AM_I) * torch.pow(torch.full_like(ri_safe, lami_min), ice_power), float(_const_segment_e_MAX_NI))
    ni1 = torch.where(hit0, ni_hit0, ni0)
    lami1 = working_power(float(_const_segment_e_AM_I * _const_SCALAR_CIG_1 * _const_segment_e_OIG1) * ni1 / ri_safe, float(_const_segment_e_OBMI))
    xdi = float(_const_segment_e_BM_I + _const_segment_e_MU_I + 1.0) / lami1
    small = xdi < 5e-06
    large = ~small & (xdi > float(_const_segment_e_D0S))
    lami_small = float(_const_SCALAR_CIE_1 / 5e-06)
    ni_small = ni_hit0
    lami_large = float(_const_SCALAR_CIE_1 / _const_segment_e_D0S)
    ni_large = float(_const_SCALAR_CIG_0 * _const_segment_e_OIG2) * ri_safe / float(_const_segment_e_AM_I) * torch.pow(torch.full_like(ri_safe, lami_large), ice_power)
    lami_active = torch.where(large, lami_large, lami1)
    ni_active = torch.where(large, ni_large, ni1)
    lami_active = torch.where(small, lami_small, lami_active)
    ni_active = torch.where(small, ni_small, ni_active)
    hit_limit = hit0 | small | large
    nit_active = torch.where(hit_limit, (ni_active / rho - ni_old) * odt, nit_old)
    # BM_I is real in Fortran. A scalar 3.0 selects Torch's integer-cube
    # shortcut, whose one-ULP error survives the later tendency subtraction.
    ni1_candidate = float(_const_SCALAR_CIG_0 * _const_segment_e_OIG2) * qi_safe / float(_const_segment_e_AM_I) * torch.pow(lami_active, ice_power)
    # Python scalar / Tensor lowers to reciprocal multiplication in eager mode.
    ni1_active = torch.maximum(torch.div(float(_const_segment_e_R2), rho), torch.minimum(ni1_candidate, torch.div(float(_const_segment_e_MAX_NI), rho)))
    return (active, torch.where(active, qi_active, 0.0), torch.where(active, ni1_active, 0.0), torch.where(active, ri_active, float(_const_segment_e_R1)), torch.where(active, ni_active, float(_const_segment_e_R2)), torch.where(active, qit_old, -qi_old * odt), torch.where(active, nit_active, -ni_old * odt), torch.where(active, 1.0 / lami_active, 0.0))


def snow_check_and_update(
    *,
    rho: torch.Tensor,
    qs1d: torch.Tensor,
    qsten: torch.Tensor,
    dt: float,
    odt: float,
) -> dict[str, torch.Tensor]:
    values = _run_tensor_compile_island(
        "snow_check_and_update",
        _snow_check_and_update_kernel,
        rho,
        rho,
        qs1d,
        qsten,
        float(dt),
        float(odt),
    )
    names = ("l_qs", "qs1d", "rs", "qsten")
    return dict(zip(names, values, strict=True))


def _snow_check_and_update_kernel(rho: torch.Tensor, qs_old: torch.Tensor, qst_old: torch.Tensor, dt: float, odt: float, _const_segment_e_R1=constants.R1) -> tuple[torch.Tensor, ...]:
    active = qs_old + qst_old * dt > float(_const_segment_e_R1)
    qs_active = qs_old + qst_old * dt
    rs_active = qs_active * rho
    return (active, torch.where(active, qs_active, 0.0), torch.where(active, rs_active, float(_const_segment_e_R1)), torch.where(active, qst_old, -qs_old * odt))


def _production_precipitation_checks_tensor(
    *,
    rho: torch.Tensor,
    qr1d: torch.Tensor,
    nr1d: torch.Tensor,
    qrten: torch.Tensor,
    nrten: torch.Tensor,
    qi1d: torch.Tensor,
    ni1d: torch.Tensor,
    qiten: torch.Tensor,
    niten: torch.Tensor,
    qs1d: torch.Tensor,
    qsten: torch.Tensor,
    dt: float,
    odt: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    values = _run_tensor_compile_island(
        "production_precipitation_checks",
        _production_precipitation_checks_kernel,
        rho,
        rho,
        qr1d,
        nr1d,
        qrten,
        nrten,
        qi1d,
        ni1d,
        qiten,
        niten,
        qs1d,
        qsten,
        float(dt),
        float(odt),
        _tensor(constants.BM_I, rho),
        _tensor(constants.BM_R, rho),
    )
    rain_names = ("l_qr", "qr1d", "nr1d", "rr", "nr", "qrten", "nrten", "ilamr", "mvd_r")
    ice_names = ("l_qi", "qi1d", "ni1d", "ri", "ni", "qiten", "niten", "ilami")
    snow_names = ("l_qs", "qs1d", "rs", "qsten")
    return (
        dict(zip(rain_names, values[:9], strict=True)),
        dict(zip(ice_names, values[9:17], strict=True)),
        dict(zip(snow_names, values[17:], strict=True)),
    )


def _production_precipitation_checks_kernel(
    rho: torch.Tensor,
    qr1d: torch.Tensor,
    nr1d: torch.Tensor,
    qrten: torch.Tensor,
    nrten: torch.Tensor,
    qi1d: torch.Tensor,
    ni1d: torch.Tensor,
    qiten: torch.Tensor,
    niten: torch.Tensor,
    qs1d: torch.Tensor,
    qsten: torch.Tensor,
    dt: float,
    odt: float,
    ice_power: torch.Tensor,
    rain_power: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    rain = _rain_check_and_update_kernel(rho, qr1d, nr1d, qrten, nrten, dt, odt, rain_power)
    ice = _ice_check_and_update_kernel(rho, qi1d, ni1d, qiten, niten, dt, odt, ice_power)
    snow = _snow_check_and_update_kernel(rho, qs1d, qsten, dt, odt)
    return (*rain, *ice, *snow)


def compute_snow_moments(
    *,
    rs: torch.Tensor,
    temp: torch.Tensor,
    l_qs: torch.Tensor,
    template: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    template_values = tuple(template[name] for name in _SNOW_MOMENT_COMPILE_FIELDS)
    values = _run_tensor_compile_island(
        "compute_snow_moments",
        _compute_snow_moments_kernel,
        rs,
        rs,
        temp,
        l_qs,
        *template_values,
        _tensor(_STATIC_SNOW_A, rs),
        _tensor(_STATIC_SNOW_B, rs),
    )
    return dict(zip(_SNOW_MOMENT_COMPILE_FIELDS, values, strict=True))


def _compute_sedimentation_snow_moments(
    *,
    rs: torch.Tensor,
    temp: torch.Tensor,
    l_qs: torch.Tensor,
    template: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    smob, smoc = _run_tensor_compile_island(
        "compute_sedimentation_snow_moments",
        _compute_sedimentation_snow_moments_kernel,
        rs,
        rs,
        temp,
        l_qs,
        template["smob"],
        template["smoc"],
        _tensor(_STATIC_SNOW_A, rs),
        _tensor(_STATIC_SNOW_B, rs),
    )
    values = dict(template)
    values["smob"] = smob
    values["smoc"] = smoc
    return values


def _compute_sedimentation_snow_moments_kernel(
    rs: torch.Tensor,
    temp: torch.Tensor,
    l_qs: torch.Tensor,
    template_smob: torch.Tensor,
    template_smoc: torch.Tensor,
    snow_a: torch.Tensor,
    snow_b: torch.Tensor,
    _t0: float = float(constants.T0),
    _oams: float = float(constants.OAMS),
    _bm_s: float = float(constants.BM_S),
    _cse0: float = float(_SCALAR_CSE[0]),
) -> tuple[torch.Tensor, torch.Tensor]:
    # Masked columns (no snow) compute on a placeholder mixing ratio: their results are discarded by the final where,
    # but 0 ** b there would send 0 * inf = NaN into the reverse pass. Kept columns are unchanged bitwise.
    rs = torch.where(l_qs, rs, torch.ones_like(rs))
    tc = torch.minimum(temp.new_full((), -0.1), temp - _t0)
    smob = rs * _oams
    if abs(_bm_s - 2.0) < 0.001:
        loga = _snow_poly(snow_a, tc, _bm_s)
        a_value = working_power(10.0, loga)
        b_value = _snow_poly(snow_b, tc, _bm_s)
        smo2 = working_power(smob / a_value, 1.0 / b_value)
    else:
        smo2 = smob
    smoc = working_power(10.0, _snow_poly(snow_a, tc, _cse0)) * working_power(
        smo2,
        _snow_poly(snow_b, tc, _cse0),
    )
    return torch.where(l_qs, smob, template_smob), torch.where(l_qs, smoc, template_smoc)


def _compute_snow_moments_kernel(rs: torch.Tensor, temp: torch.Tensor, l_qs: torch.Tensor, template_smo0: torch.Tensor, template_smo1: torch.Tensor, template_smo2: torch.Tensor, template_smob: torch.Tensor, template_smoc: torch.Tensor, template_smoe: torch.Tensor, template_smof: torch.Tensor, template_smog: torch.Tensor, template_smoz: torch.Tensor, template_ns: torch.Tensor, snow_a: torch.Tensor, snow_b: torch.Tensor, _const_segment_e_T0=constants.T0, _const_segment_e_OAMS=constants.OAMS, _const_segment_e_BM_S=constants.BM_S, _const_SCALAR_CSE_0=_SCALAR_CSE[0], _const_segment_e_LAM0=constants.LAM0, _const_segment_e_LAM1=constants.LAM1, _const_segment_e_KAP0=constants.KAP0, _const_segment_e_KAP1=constants.KAP1, _const_segment_e_MU_S=constants.MU_S, _const_SCALAR_CSG_14=_SCALAR_CSG[14], _const_SCALAR_CSE_14=_SCALAR_CSE[14], _const_SCALAR_CSE_12=_SCALAR_CSE[12], _const_SCALAR_CSE_15=_SCALAR_CSE[15], _const_SCALAR_CSE_16=_SCALAR_CSE[16], _const_SCALAR_CSE_2=_SCALAR_CSE[2]) -> tuple[torch.Tensor, ...]:
    # Masked columns (no snow) compute on a placeholder mixing ratio: their results are discarded by the final where,
    # but 0 ** b there would send 0 * inf = NaN into the reverse pass. Kept columns are unchanged bitwise.
    rs = torch.where(l_qs, rs, torch.ones_like(rs))
    tc = torch.minimum(temp.new_full((), -0.1), temp - float(_const_segment_e_T0))
    smob = rs * float(_const_segment_e_OAMS)
    if abs(float(_const_segment_e_BM_S) - 2.0) < 0.001:
        loga = _snow_poly(snow_a, tc, float(_const_segment_e_BM_S))
        a_value = working_power(10.0, loga)
        b_value = _snow_poly(snow_b, tc, float(_const_segment_e_BM_S))
        smo2 = working_power(smob / a_value, 1.0 / b_value)
    else:
        smo2 = smob
    smoc = working_power(10.0, _snow_poly(snow_a, tc, _const_SCALAR_CSE_0)) * working_power(smo2, _snow_poly(snow_b, tc, _const_SCALAR_CSE_0))
    smo0 = working_power(10.0, _snow_poly_no_shape(snow_a, tc)) * working_power(smo2, _snow_poly_no_shape(snow_b, tc))
    smo1 = working_power(10.0, _snow_poly(snow_a, tc, 1.0)) * working_power(smo2, _snow_poly(snow_b, tc, 1.0))
    m0 = smob / smoc
    mrat = smob * m0 * m0 * m0
    slam1 = m0 * float(_const_segment_e_LAM0)
    slam2 = m0 * float(_const_segment_e_LAM1)
    ns = mrat * float(_const_segment_e_KAP0) / slam1 + mrat * float(_const_segment_e_KAP1) * working_power(m0, float(_const_segment_e_MU_S)) * _const_SCALAR_CSG_14 / working_power(slam2, float(_const_SCALAR_CSE_14))
    smoe = working_power(10.0, _snow_poly(snow_a, tc, _const_SCALAR_CSE_12)) * working_power(smo2, _snow_poly(snow_b, tc, _const_SCALAR_CSE_12))
    smof = working_power(10.0, _snow_poly(snow_a, tc, _const_SCALAR_CSE_15)) * working_power(smo2, _snow_poly(snow_b, tc, _const_SCALAR_CSE_15))
    smog = working_power(10.0, _snow_poly(snow_a, tc, _const_SCALAR_CSE_16)) * working_power(smo2, _snow_poly(snow_b, tc, _const_SCALAR_CSE_16))
    smoz = working_power(10.0, _snow_poly(snow_a, tc, _const_SCALAR_CSE_2)) * working_power(smo2, _snow_poly(snow_b, tc, _const_SCALAR_CSE_2))
    return (torch.where(l_qs, smo0, template_smo0), torch.where(l_qs, smo1, template_smo1), torch.where(l_qs, smo2, template_smo2), torch.where(l_qs, smob, template_smob), torch.where(l_qs, smoc, template_smoc), torch.where(l_qs, smoe, template_smoe), torch.where(l_qs, smof, template_smof), torch.where(l_qs, smog, template_smog), torch.where(l_qs, smoz, template_smoz), torch.where(l_qs, ns, template_ns))


def run_f(
    state: Mapping[str, torch.Tensor] | TorchSegmentEResult,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    nc1d: torch.Tensor | None = None,
    land: torch.Tensor | None = None,
    copy: bool = False,
) -> TorchSegmentFResult:
    e_result = _segment_e_result(state, dt=dt, is_first_step=is_first_step)
    input_state = e_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = e_result.workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment F dt must be nonzero")
    odt = 1.0 / resolved_dt
    shape = (e_result.ncol, e_result.nz)
    nc_values, nc_key = _optional_profile_with_key("F", "nc1d", nc1d, input_state, ("nc1d", "nc"), shape)
    land_values = land if land is not None else input_state.get("land")
    base = e_result.workspace.base.base
    tendencies = _tensor_mapping(e_result.workspace.tendencies, copy=copy)
    initial_nc, ncsave = _initial_cloud_number(
        rho=base.rho,
        nc1d=nc_values,
        land=land_values,
        shape=shape,
    )
    cloud = cloud_check_and_update(
        rho=base.rho,
        qc1d=input_state["qc"],
        nc=initial_nc,
        qcten=tendencies["qcten"],
        ncten=tendencies["ncten"],
        dt=resolved_dt,
        odt=odt,
        nc1d=nc_values,
        ncsave=ncsave,
    )
    result_state["qc"] = cloud["qc1d"]
    if nc_key is not None and cloud["nc1d"] is not None:
        result_state[nc_key] = cloud["nc1d"]
    tendencies["qcten"] = cloud["qcten"]
    tendencies["ncten"] = cloud["ncten"]
    workspace = TorchSegmentFWorkspace(
        base=e_result.workspace,
        l_qc=cloud["l_qc"],
        rc=cloud["rc"],
        nc=cloud["nc"],
        ncsave=ncsave,
        nc1d=cloud["nc1d"],
        ilamc=cloud["ilamc"],
        mvd_c=cloud["mvd_c"],
        tendencies=tendencies,
        dt=resolved_dt,
        odt=odt,
    )
    return TorchSegmentFResult(result_state, workspace, e_result.ncol, e_result.nz)


def get_constant_cloud_number(*, shape: tuple[int, int], reference: torch.Tensor, land: torch.Tensor | None = None) -> torch.Tensor:
    if land is None:
        return torch.full(shape, float(constants.NT_C_L), **_factory(reference))
    land_values = land.to(device=reference.device)
    if land_values.ndim == 0:
        value = torch.where(
            land_values == 1,
            _scalar(constants.NT_C_L, reference),
            _scalar(constants.NT_C_O, reference),
        )
        return torch.full(shape, 1.0, **_factory(reference)) * value
    if tuple(land_values.shape) != (shape[0],):
        raise ValueError(f"Segment F land shape {tuple(land_values.shape)} differs from [{shape[0]}]")
    column_values = torch.where(
        land_values[:, None] == 1,
        _scalar(constants.NT_C_L, reference),
        _scalar(constants.NT_C_O, reference),
    )
    return column_values.expand(shape)


def cloud_check_and_update(
    *,
    rho: torch.Tensor,
    qc1d: torch.Tensor,
    nc: torch.Tensor,
    qcten: torch.Tensor,
    ncten: torch.Tensor,
    dt: float,
    odt: float,
    nc1d: torch.Tensor | None = None,
    ncsave: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    if nc1d is not None:
        values = _run_tensor_compile_island(
            "cloud_check_and_update_optional",
            _cloud_check_and_update_optional_kernel,
            rho,
            rho,
            qc1d,
            nc,
            qcten,
            ncten,
            nc1d,
            _tensor(_STATIC_CLOUD_CCE[1], rho),
            _tensor(_STATIC_CLOUD_CCG[0], rho),
            _tensor(_STATIC_CLOUD_OCG2, rho),
            _tensor(_STATIC_CLOUD_CCG[1], rho),
            _tensor(_STATIC_CLOUD_OCG1, rho),
            float(dt),
            float(odt),
        )
        names = ("l_qc", "qc1d", "nc1d", "rc", "nc", "qcten", "ncten", "ilamc", "mvd_c")
        return dict(zip(names, values, strict=True))
    nc_base = ncsave if ncsave is not None else _scalar(float(constants.NT_C_L), qc1d)
    values = _run_tensor_compile_island(
        "cloud_check_and_update_base",
        _cloud_check_and_update_base_kernel,
        rho,
        rho,
        qc1d,
        nc_base,
        qcten,
        ncten,
        _tensor(_STATIC_CLOUD_CCG[1], rho),
        _tensor(_STATIC_CLOUD_OCG1, rho),
        float(dt),
        float(odt),
    )
    l_qc, qc1d_out, rc, nc_out, qcten_out, ncten_out, ilamc, mvd_c = values
    return {
        "l_qc": l_qc,
        "qc1d": qc1d_out,
        "nc1d": None,
        "rc": rc,
        "nc": nc_out,
        "qcten": qcten_out,
        "ncten": ncten_out,
        "ilamc": ilamc,
        "mvd_c": mvd_c,
    }


def _cloud_nuc_no_cache(nc: torch.Tensor, _const_segment_f_NU_C_SCALE=constants.NU_C_SCALE) -> torch.Tensor:
    ratio = float(_const_segment_f_NU_C_SCALE) / nc
    return torch.where(ratio >= 12.5, 15, torch.clamp(torch.floor(ratio + 0.5).to(torch.int64) + 2, max=15))


def _cloud_lam_no_cache(nc_work: torch.Tensor, rc: torch.Tensor, nu_c: torch.Tensor, ccg1_table: torch.Tensor, ocg1_table: torch.Tensor, _const_segment_e_AM_R=constants.AM_R, _const_segment_e_OBMR=constants.OBMR) -> torch.Tensor:
    index = nu_c.to(torch.int64) - 1
    # Evaluated in float64 and rounded once (the working_power policy): with rc near 1e-12 and nc at its
    # maximum, the division's backward factor (x/rc)/rc exceeds the float32 range although the derivative
    # of the result does not. The casts are no-ops in float64.
    ratio = (nc_work.double() * float(_const_segment_e_AM_R) * ccg1_table[index].double() * ocg1_table[index].double()
             / rc.double())
    return working_power(ratio, float(_const_segment_e_OBMR)).to(nc_work.dtype)


def _cloud_check_common_merge(active: torch.Tensor, qc_old: torch.Tensor, qc_active: torch.Tensor, rc_active: torch.Tensor, nc_work_active: torch.Tensor, qcten: torch.Tensor, nct_current: torch.Tensor, nct_active: torch.Tensor, ilamc_active: torch.Tensor, mvd_active: torch.Tensor, odt: float, _const_segment_e_R1=constants.R1, _const_segment_f_NT_C_MIN=constants.NT_C_MIN, _const_segment_f_D0C=constants.D0C) -> tuple[torch.Tensor, ...]:
    return (active, torch.where(active, qc_active, 0.0), torch.where(active, rc_active, float(_const_segment_e_R1)), torch.where(active, nc_work_active, float(_const_segment_f_NT_C_MIN)), torch.where(active, qcten, -qc_old * odt), torch.where(active, nct_active, nct_current), torch.where(active, ilamc_active, 0.0), torch.where(active, mvd_active, float(_const_segment_f_D0C)))


def _cloud_check_and_update_optional_kernel(rho: torch.Tensor, qc_old: torch.Tensor, nc: torch.Tensor, qcten: torch.Tensor, ncten: torch.Tensor, nc1d: torch.Tensor, cce1_table: torch.Tensor, ccg0_table: torch.Tensor, ocg2_table: torch.Tensor, ccg1_table: torch.Tensor, ocg1_table: torch.Tensor, dt: float, odt: float, _const_segment_e_R1=constants.R1, _const_segment_f_NT_C_MIN=constants.NT_C_MIN, _const_segment_f_NT_C_MAX=constants.NT_C_MAX, _const_segment_e_BM_R=constants.BM_R, _const_segment_f_D0C=constants.D0C, _const_segment_e_D0R=constants.D0R, _const_segment_e_AM_R=constants.AM_R) -> tuple[torch.Tensor, ...]:
    del nc
    active = qc_old + qcten * dt > float(_const_segment_e_R1)
    qc_active = qc_old + qcten * dt
    rc_active = qc_active * rho
    # Inactive lanes use 1, not R1: every output is masked by ``active``, and (x/R1)/R1 in the backward of
    # x/rc overflows float32 (0 * inf = NaN).
    rc_for_lam = torch.where(active, rc_active, 1.0)
    raw_nc = (nc1d + ncten * dt) * rho
    nc0 = torch.clamp(raw_nc, min=float(_const_segment_f_NT_C_MIN), max=float(_const_segment_f_NT_C_MAX))
    hit0 = (raw_nc <= float(_const_segment_f_NT_C_MIN)) | (raw_nc > float(_const_segment_f_NT_C_MAX))
    nu0 = _cloud_nuc_no_cache(nc0)
    lam0 = _cloud_lam_no_cache(nc0, rc_for_lam, nu0, ccg1_table, ocg1_table)
    xdc = (float(_const_segment_e_BM_R) + nu0.to(dtype=rho.dtype) + 1.0) / lam0
    small = xdc < float(_const_segment_f_D0C)
    large = ~small & (xdc > float(_const_segment_e_D0R * 2.0))
    table_index = nu0.to(torch.int64) - 1
    cce1 = cce1_table[table_index]
    ccg0_ocg2 = ccg0_table[table_index] * ocg2_table[table_index]
    lam_limit = torch.where(small, cce1 / float(_const_segment_f_D0C), torch.where(large, cce1 / float(_const_segment_e_D0R * 2.0), lam0))
    hit_limit = hit0 | small | large
    nc_work_active = ccg0_ocg2 * rc_for_lam / float(_const_segment_e_AM_R) * _libm_power(lam_limit, 3.0)
    nct_active = torch.where(hit_limit, (nc_work_active / rho - nc1d) * odt, ncten)
    nc_optional_candidate = ccg0_ocg2 * qc_active / float(_const_segment_e_AM_R) * _libm_power(lam_limit, 3.0)
    nc_optional_active = torch.maximum(float(_const_segment_f_NT_C_MIN) / rho, torch.minimum(nc_optional_candidate, float(_const_segment_f_NT_C_MAX) / rho))
    nu = _cloud_nuc_no_cache(nc_work_active)
    lam = _cloud_lam_no_cache(nc_work_active, rc_for_lam, nu, ccg1_table, ocg1_table)
    ilamc_active = 1.0 / lam
    mvd_active = torch.clamp((3.0 + nu.to(dtype=rho.dtype) + 0.672) * ilamc_active, min=float(_const_segment_f_D0C), max=float(_const_segment_e_D0R))
    l_qc, qc1d_out, rc, nc_out, qcten_out, ncten_out, ilamc, mvd_c = _cloud_check_common_merge(active, qc_old, qc_active, rc_active, nc_work_active, qcten, -nc1d * odt, nct_active, ilamc_active, mvd_active, odt)
    return (l_qc, qc1d_out, torch.where(active, nc_optional_active, 0.0), rc, nc_out, qcten_out, ncten_out, ilamc, mvd_c)


def _cloud_check_and_update_base_kernel(rho: torch.Tensor, qc_old: torch.Tensor, nc_base: torch.Tensor, qcten: torch.Tensor, ncten: torch.Tensor, ccg1_table: torch.Tensor, ocg1_table: torch.Tensor, dt: float, odt: float, _const_segment_e_R1=constants.R1, _const_segment_e_D0R=constants.D0R, _const_segment_f_D0C=constants.D0C) -> tuple[torch.Tensor, ...]:
    active = qc_old + qcten * dt > float(_const_segment_e_R1)
    qc_active = qc_old + qcten * dt
    rc_active = qc_active * rho
    # Inactive lanes use 1, not R1: every output is masked by ``active``, and (x/R1)/R1 in the backward of
    # x/rc overflows float32 (0 * inf = NaN).
    rc_for_lam = torch.where(active, rc_active, 1.0)
    nu = _cloud_nuc_no_cache(nc_base)
    lam = _cloud_lam_no_cache(nc_base, rc_for_lam, nu, ccg1_table, ocg1_table)
    ilamc_active = 1.0 / lam
    mvd_active = torch.clamp((3.0 + nu.to(dtype=rho.dtype) + 0.672) * ilamc_active, min=float(_const_segment_f_D0C), max=float(_const_segment_e_D0R))
    return _cloud_check_common_merge(active, qc_old, qc_active, rc_active, nc_base, qcten, ncten, ncten, ilamc_active, mvd_active, odt)


def run_g(
    state: Mapping[str, torch.Tensor] | TorchSegmentFResult,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    copy: bool = False,
) -> TorchSegmentGResult:
    f_result = _segment_f_result(state, dt=dt, is_first_step=is_first_step)
    input_state = f_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = f_result.workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment G dt must be nonzero")
    odt = 1.0 / resolved_dt
    c_base = f_result.workspace.base.base.base
    shape = (f_result.ncol, f_result.nz)
    ng_values, ng_key = _optional_profile_with_key(
        "G",
        "ng1d",
        None,
        input_state,
        ("ng1d", "ng"),
        shape,
    )
    qb_values, qb_key = _optional_profile_with_key(
        "G",
        "qb1d",
        None,
        input_state,
        ("qb1d", "qb"),
        shape,
    )
    local_first_step = (
        f_result.workspace.base.base.local_first_step
        if is_first_step is None
        else bool(is_first_step)
    )
    if local_first_step and ng_values is not None and qb_values is not None:
        ng_values, qb_values = initialize_graupel_columns_tensor(
            rho=c_base.rho,
            qg1d=input_state["qg"],
            ng1d=ng_values,
            qb1d=qb_values,
        )
    tendencies = _tensor_mapping(f_result.workspace.tendencies, copy=copy)
    graupel = graupel_check_and_update(
        rho=c_base.rho,
        qg1d=input_state["qg"],
        qgten=tendencies["qgten"],
        ngten=tendencies["ngten"],
        qbten=tendencies["qbten"],
        dt=resolved_dt,
        odt=odt,
        ng1d=ng_values,
        qb1d=qb_values,
    )
    result_state["qg"] = graupel["qg1d"]
    if ng_key is not None and graupel["ng1d"] is not None:
        result_state[ng_key] = graupel["ng1d"]
    if qb_key is not None and graupel["qb1d"] is not None:
        result_state[qb_key] = graupel["qb1d"]
    tendency_reset_names = ("qcten", "qiten", "qrten", "qsten", "qgten", "ngten", "qbten", "niten", "nrten", "ncten")
    if _regional_execution_enabled("g_shared_zeros"):
        # These tensors are immutable seeds: downstream stages replace mapping
        # entries and never mutate the shared storage.  One fill therefore has
        # the same value/gradient contract as ten independent fills.
        shared_zero = torch.zeros_like(tendencies[tendency_reset_names[0]])
        for name in tendency_reset_names:
            tendencies[name] = shared_zero
    else:
        for name in tendency_reset_names:
            tendencies[name] = torch.zeros_like(tendencies[name])
    thermo, supersaturated = thermo_vars(qv=c_base.qv, temp=c_base.temp, pres=c_base.pres, rho=c_base.rho)
    do_micro = (
        f_result.workspace.l_qc
        | f_result.workspace.base.l_qr
        | f_result.workspace.base.l_qi
        | f_result.workspace.base.l_qs
        | graupel["l_qg"]
    )
    workspace = TorchSegmentGWorkspace(
        base=f_result.workspace,
        l_qg=graupel["l_qg"],
        rg=graupel["rg"],
        ng=graupel["ng"],
        rb=graupel["rb"],
        idx_bg=graupel["idx"],
        ilamg=graupel["ilamg"],
        mvd_g=graupel["mvd_g"],
        ng1d=graupel["ng1d"],
        qb1d=graupel["qb1d"],
        tendencies=tendencies,
        thermo=thermo,
        do_micro=do_micro,
        supersaturated=supersaturated,
        dt=resolved_dt,
        odt=odt,
    )
    return TorchSegmentGResult(result_state, workspace, f_result.ncol, f_result.nz)


def run_h(
    state: Mapping[str, torch.Tensor] | TorchSegmentGResult,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    copy: bool = False,
) -> TorchSegmentHResult:
    g_result = _segment_g_result(state, dt=dt, is_first_step=is_first_step)
    input_state = g_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = g_result.workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment H dt must be nonzero")
    odt = 1.0 / resolved_dt
    c_base = g_result.workspace.base.base.base.base
    tend = _tensor_mapping(c_base.tend, copy=copy)
    tendencies = _tensor_mapping(g_result.workspace.tendencies, copy=copy)
    resolved_tables = tables

    if not turn_off_micro:
        if resolved_tables is None:
            raise ValueError("Segment H requires prepared TEMPO coefficient tables")
        active_micro = g_result.workspace.do_micro | g_result.workspace.supersaturated
        rain_table_indices = _rain_table_indices(g_result.workspace.base.base.rr, g_result.workspace.base.base.ilamr)
        tend = warm_rain_tendencies(
            rhof=g_result.workspace.thermo["rhof"],
            l_qc=g_result.workspace.base.l_qc,
            rc=g_result.workspace.base.rc,
            nc=g_result.workspace.base.nc,
            ilamc=g_result.workspace.base.ilamc,
            mvd_c=g_result.workspace.base.mvd_c,
            l_qr=g_result.workspace.base.base.l_qr,
            rr=g_result.workspace.base.base.rr,
            nr=g_result.workspace.base.base.nr,
            mvd_r=g_result.workspace.base.base.mvd_r,
            tend=tend,
            odt=odt,
            active_micro=active_micro,
        )
        tend = rain_snow_rain_graupel_tendencies(
            temp=c_base.temp,
            l_qr=g_result.workspace.base.base.l_qr,
            rr=g_result.workspace.base.base.rr,
            nr=g_result.workspace.base.base.nr,
            ilamr=g_result.workspace.base.base.ilamr,
            l_qs=g_result.workspace.base.base.l_qs,
            rs=g_result.workspace.base.base.rs,
            l_qg=g_result.workspace.l_qg,
            rg=g_result.workspace.rg,
            ng=g_result.workspace.ng,
            ilamg=g_result.workspace.ilamg,
            idx=g_result.workspace.idx_bg,
            tend=tend,
            tables=resolved_tables,
            odt=odt,
            active_micro=active_micro,
            rain_table_indices=rain_table_indices,
        )

    workspace = TorchSegmentHWorkspace(
        base=g_result.workspace,
        tend=tend,
        tendencies=tendencies,
        tables=resolved_tables,
        dt=resolved_dt,
        odt=odt,
        rain_table_indices=None if turn_off_micro else rain_table_indices,
    )
    return TorchSegmentHResult(result_state, workspace, g_result.ncol, g_result.nz)


def warm_rain_tendencies(
    *,
    rhof: torch.Tensor,
    l_qc: torch.Tensor,
    rc: torch.Tensor,
    nc: torch.Tensor,
    ilamc: torch.Tensor,
    mvd_c: torch.Tensor,
    l_qr: torch.Tensor,
    rr: torch.Tensor,
    nr: torch.Tensor,
    mvd_r: torch.Tensor,
    tend: Mapping[str, torch.Tensor],
    odt: float,
    active_micro: torch.Tensor,
) -> dict[str, torch.Tensor]:
    names = _WARM_RAIN_TENDENCY_FIELDS
    updated = dict(tend)
    values = _run_tensor_compile_island(
        "warm_rain_tendencies",
        _warm_rain_tendencies_kernel,
        rc,
        rhof,
        l_qc,
        rc,
        nc,
        ilamc,
        mvd_c,
        l_qr,
        rr,
        nr,
        mvd_r,
        active_micro,
        *tuple(updated[name] for name in names),
        _tensor(_STATIC_CLOUD_CCG[2], rc),
        _tensor(_STATIC_CLOUD_OCG2, rc),
        _tensor(_STATIC_RAIN_DIAMETER_BINS, rc),
        _tensor(_STATIC_WARM_RAIN_EFFICIENCY, rc),
        float(odt),
    )
    for name, value in zip(names, values, strict=True):
        updated[name] = value
    return updated


_WARM_RAIN_TENDENCY_FIELDS = (
    "pnr_rcr",
    "prr_wau",
    "pnr_wau",
    "pnc_wau",
    "prr_rcw",
    "pnc_rcw",
)


def _warm_rain_tendencies_kernel(rhof: torch.Tensor, l_qc: torch.Tensor, rc: torch.Tensor, nc: torch.Tensor, ilamc: torch.Tensor, mvd_c: torch.Tensor, l_qr: torch.Tensor, rr: torch.Tensor, nr: torch.Tensor, mvd_r: torch.Tensor, active_micro: torch.Tensor, pnr_rcr: torch.Tensor, prr_wau: torch.Tensor, pnr_wau: torch.Tensor, pnc_wau: torch.Tensor, prr_rcw: torch.Tensor, pnc_rcw: torch.Tensor, cloud_ccg2_table: torch.Tensor, cloud_ocg2_table: torch.Tensor, rain_diameter_bins: torch.Tensor, warm_rain_efficiency: torch.Tensor, odt: float, _const_segment_e_D0R=constants.D0R, _const_segment_e_R1=constants.R1, _const_segment_e_R2=constants.R2, _const_segment_f_D0C=constants.D0C, _const_segment_e_AM_R=constants.AM_R, _const_segment_e_OBMR=constants.OBMR, _const_segment_h_AUTOCON_NR_FACTOR=constants.AUTOCON_NR_FACTOR, _const_segment_h_NBR=constants.NBR, _const_segment_h_NBC=constants.NBC, _const_segment_e_MU_R=constants.MU_R, _const_segment_e_ORG2=constants.ORG2, _const_segment_h_T1_QR_QC=constants.T1_QR_QC, _const_segment_h_FV_R=constants.FV_R, _const_SCALAR_CRE_8=_SCALAR_CRE[8]) -> tuple[torch.Tensor, ...]:
    active = active_micro
    rain_collision = active & l_qr & (mvd_r > float(_const_segment_e_D0R))
    ef_rr = torch.clamp_min(1.0 - torch.exp(2300.0 * (mvd_r - 0.00195)), -0.1)
    pnr_rcr = torch.where(rain_collision, ef_rr * 2.0 * nr * rr, pnr_rcr)
    cloud_auto = active & l_qc & (rc > 1e-05)
    rc_safe = torch.clamp_min(rc, float(_const_segment_e_R1))
    nc_safe = torch.clamp_min(nc, float(_const_segment_e_R2))
    ilamc_safe = torch.where(ilamc > 0.0, ilamc, 1.0)
    mvd_c_safe = torch.clamp_min(mvd_c, float(_const_segment_f_D0C))
    nu_c = _cloud_nuc_no_cache(nc_safe)
    lamc = 1.0 / ilamc_safe
    xdc = torch.clamp_min(working_power(rc_safe / (float(_const_segment_e_AM_R) * nc_safe), float(_const_segment_e_OBMR)) * 1000000.0, float(_const_segment_f_D0C * 1000000.0))
    table_index = nu_c - 1
    ccg2 = cloud_ccg2_table[table_index]
    ocg2 = cloud_ocg2_table[table_index]
    dc_g = working_power(ccg2 * ocg2, float(_const_segment_e_OBMR)) / lamc * 1000000.0
    # Preserve the source product order before the sixth-root cancellation.
    dc_b_argument = xdc * xdc * xdc * dc_g * dc_g * dc_g - xdc * xdc * xdc * xdc * xdc * xdc
    dc_b = torch.pow(torch.where(cloud_auto, dc_b_argument, 1.0), 1.0 / 6.0)
    zeta1_arg = 6.25e-06 * xdc * dc_b * dc_b * dc_b - 0.4
    zeta1 = 0.5 * (zeta1_arg + torch.abs(zeta1_arg))
    zeta = 0.027 * rc_safe * zeta1
    taud_arg = 0.5 * dc_b - 7.5
    taud = 0.5 * (taud_arg + torch.abs(taud_arg)) + float(_const_segment_e_R1)
    # Masked lanes divide by 1: their 3.72/(rc*taud)**2 backward factor overflows float32 (0 * inf = NaN).
    tau = torch.div(3.72, torch.where(cloud_auto, rc_safe * taud, 1.0))
    prr_wau_candidate = _min_first(rc_safe * odt, zeta / tau)
    pnr_wau_candidate = prr_wau_candidate / (float(_const_segment_e_AM_R * _const_segment_h_AUTOCON_NR_FACTOR * _const_segment_e_D0R ** 3) * nu_c.to(dtype=rc.dtype))
    pnc_wau_candidate = _min_first(nc_safe * odt, prr_wau_candidate / (float(_const_segment_e_AM_R) * mvd_c_safe * mvd_c_safe * mvd_c_safe))
    prr_wau = torch.where(cloud_auto, prr_wau_candidate, prr_wau)
    pnr_wau = torch.where(cloud_auto, pnr_wau_candidate, pnr_wau)
    pnc_wau = torch.where(cloud_auto, pnc_wau_candidate, pnc_wau)
    rain_cloud = active & l_qr & l_qc & (mvd_r > float(_const_segment_e_D0R)) & (mvd_c > float(_const_segment_f_D0C))
    rain_ratio = _clamp_tiny(mvd_r / rain_diameter_bins[0])
    rain_bin = (1 + torch.floor(float(_const_segment_h_NBR) * torch.log(rain_ratio) / torch.log(rain_diameter_bins[-1] / rain_diameter_bins[0])).to(torch.int64)).clamp(1, _const_segment_h_NBR)
    cloud_bin = (torch.trunc(mvd_c * 1000000.0).to(torch.int64) - 1).clamp(0, _const_segment_h_NBC - 1)
    ef_rw = warm_rain_efficiency[rain_bin - 1, cloud_bin]
    lamr = torch.div(float(3.0 + _const_segment_e_MU_R + 0.672), torch.clamp_min(mvd_r, float(_const_segment_e_D0R)))
    n0_r = nr * float(_const_segment_e_ORG2) * lamr
    # The source multiplies cloud mass/number before the rain moment and power.
    collection_prefix = rhof * float(_const_segment_h_T1_QR_QC) * ef_rw
    collection_power = (lamr + float(_const_segment_h_FV_R)) ** (-float(_const_SCALAR_CRE_8))
    prr_rcw_candidate = _min_first(rc_safe * odt, collection_prefix * rc_safe * n0_r * collection_power)
    pnc_rcw_candidate = _min_first(nc_safe * odt, collection_prefix * nc_safe * n0_r * collection_power)
    prr_rcw = torch.where(rain_cloud, prr_rcw_candidate, prr_rcw)
    pnc_rcw = torch.where(rain_cloud, pnc_rcw_candidate, pnc_rcw)
    return (pnr_rcr, prr_wau, pnr_wau, pnc_wau, prr_rcw, pnc_rcw)


def rain_snow_rain_graupel_tendencies(
    *,
    temp: torch.Tensor,
    l_qr: torch.Tensor,
    rr: torch.Tensor,
    nr: torch.Tensor,
    ilamr: torch.Tensor,
    l_qs: torch.Tensor,
    rs: torch.Tensor,
    l_qg: torch.Tensor,
    rg: torch.Tensor,
    ng: torch.Tensor,
    ilamg: torch.Tensor,
    idx: torch.Tensor,
    tend: Mapping[str, torch.Tensor],
    tables: Any,
    odt: float,
    active_micro: torch.Tensor,
    rain_table_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    if not (
        TENSOR_COMPILE_ISLAND_CUDA
        and temp.device.type in TENSOR_COMPILE_ISLAND_DEVICES
        and not torch.compiler.is_compiling()
    ):
        return _rain_snow_rain_graupel_tendencies_eager(
            temp=temp,
            l_qr=l_qr,
            rr=rr,
            nr=nr,
            ilamr=ilamr,
            l_qs=l_qs,
            rs=rs,
            l_qg=l_qg,
            rg=rg,
            ng=ng,
            ilamg=ilamg,
            idx=idx,
            tend=tend,
            tables=tables,
            odt=odt,
            active_micro=active_micro,
            rain_table_indices=rain_table_indices,
        )
    names = (
        "prr_rcs",
        "prs_rcs",
        "prg_rcs",
        "pnr_rcs",
        "png_rcs",
        "pbg_rcs",
        "prg_rcg",
        "prr_rcg",
        "pnr_rcg",
        "pbg_rcg",
        "png_rcg",
    )
    arrays = tables.arrays
    values = _run_tensor_compile_island(
        "rain_snow_rain_graupel_tendencies",
        _rain_snow_rain_graupel_tendencies_kernel,
        temp,
        temp,
        l_qr,
        rr,
        nr,
        ilamr,
        l_qs,
        rs,
        l_qg,
        rg,
        ng,
        ilamg,
        idx,
        active_micro,
        *tuple(tend[name] for name in names),
        *tuple(
            _tensor(arrays[name], temp)
            for name in (
                "tmr_racs2",
                "tcr_sacr2",
                "tmr_racs1",
                "tcr_sacr1",
                "tcs_racs1",
                "tms_sacr1",
                "tnr_racs1",
                "tnr_racs2",
                "tnr_sacr1",
                "tnr_sacr2",
                "tmr_racg",
                "tcr_gacr",
                "tnr_racg",
                "tnr_gacr",
                "tcg_racg",
            )
        ),
        _tensor(_STATIC_GRAUPEL_AM_G, temp),
        _tensor(_STATIC_GRAUPEL_RHO_G, temp),
        float(odt),
    )
    updated = dict(tend)
    for name, value in zip(names, values, strict=True):
        updated[name] = value
    return updated


def _rain_snow_rain_graupel_tendencies_eager(
    *,
    temp: torch.Tensor,
    l_qr: torch.Tensor,
    rr: torch.Tensor,
    nr: torch.Tensor,
    ilamr: torch.Tensor,
    l_qs: torch.Tensor,
    rs: torch.Tensor,
    l_qg: torch.Tensor,
    rg: torch.Tensor,
    ng: torch.Tensor,
    ilamg: torch.Tensor,
    idx: torch.Tensor,
    tend: Mapping[str, torch.Tensor],
    tables: Any,
    odt: float,
    active_micro: torch.Tensor,
    rain_table_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    updated = dict(tend)
    arrays = tables.arrays
    rain_snow = active_micro & l_qr & l_qs & (rr >= _scalar(_SCALAR_RAIN_WATER_THRESHOLDS[0], rr)) & (rs >= _scalar(_SCALAR_SNOW_WATER_THRESHOLDS[0], rs))
    cold_rs = rain_snow & (temp < _scalar(constants.T0, temp))
    warm_rs = rain_snow & ~cold_rs
    idx_t = _temperature_table_index(temp)
    idx_r, idx_r1 = _rain_table_indices(rr, ilamr) if rain_table_indices is None else rain_table_indices
    idx_s = _mass_table_index(rs, exponent_offset=constants.NIS2, upper=constants.NTB_S)

    tmr_racs2 = _table_gather_array(arrays["tmr_racs2"], temp, idx_s, idx_t, idx_r1, idx_r)
    tcr_sacr2 = _table_gather_array(arrays["tcr_sacr2"], temp, idx_s, idx_t, idx_r1, idx_r)
    tmr_racs1 = _table_gather_array(arrays["tmr_racs1"], temp, idx_s, idx_t, idx_r1, idx_r)
    tcr_sacr1 = _table_gather_array(arrays["tcr_sacr1"], temp, idx_s, idx_t, idx_r1, idx_r)
    tcs_racs1 = _table_gather_array(arrays["tcs_racs1"], temp, idx_s, idx_t, idx_r1, idx_r)
    tms_sacr1 = _table_gather_array(arrays["tms_sacr1"], temp, idx_s, idx_t, idx_r1, idx_r)

    prr_rcs_cold = _max_first(-rr * _scalar(odt, rr), -(tmr_racs2 + tcr_sacr2 + tmr_racs1 + tcr_sacr1))
    prs_rcs_cold = _max_first(-rs * _scalar(odt, rs), tmr_racs2 + tcr_sacr2 - tcs_racs1 - tms_sacr1)
    prg_rcs_cold = _min_first((rr + rs) * _scalar(odt, rr), tmr_racs1 + tcr_sacr1 + tcs_racs1 + tms_sacr1)
    pnr_rcs_cold = _min_first(
        nr * _scalar(odt, nr),
        _table_gather_array(arrays["tnr_racs1"], temp, idx_s, idx_t, idx_r1, idx_r)
        + _table_gather_array(arrays["tnr_racs2"], temp, idx_s, idx_t, idx_r1, idx_r)
        + _table_gather_array(arrays["tnr_sacr1"], temp, idx_s, idx_t, idx_r1, idx_r)
        + _table_gather_array(arrays["tnr_sacr2"], temp, idx_s, idx_t, idx_r1, idx_r),
    )
    prs_rcs_warm = _max_first(-rs * _scalar(odt, rs), -tcs_racs1 - tms_sacr1 + tmr_racs2 + tcr_sacr2)
    _masked_update_mapping(
        updated,
        cold_rs,
        {
            "prr_rcs": prr_rcs_cold,
            "prs_rcs": prs_rcs_cold,
            "prg_rcs": prg_rcs_cold,
            "pnr_rcs": pnr_rcs_cold,
            "png_rcs": pnr_rcs_cold,
            "pbg_rcs": _scalar(constants.METERS3_TO_LITERS / constants.RHO_I, prg_rcs_cold) * prg_rcs_cold,
        },
    )
    _masked_update_mapping(updated, warm_rs, {"prr_rcs": -prs_rcs_warm, "prs_rcs": prs_rcs_warm})

    rain_graupel = active_micro & l_qr & l_qg & (rr >= _scalar(_SCALAR_RAIN_WATER_THRESHOLDS[0], rr)) & (rg >= _scalar(_SCALAR_GRAUPEL_WATER_THRESHOLDS[0], rg))
    cold_rg = rain_graupel & (temp < _scalar(constants.T0, temp))
    warm_rg = rain_graupel & ~cold_rg
    density_idx = idx.to(torch.int64).clamp(1, constants.NRHG)
    idx_g, idx_g1 = _graupel_table_indices(rg, ilamg, density_idx)
    tmr_racg = _table_gather_array(arrays["tmr_racg"], temp, idx_g1, idx_g, density_idx, idx_r1, idx_r)
    tcr_gacr = _table_gather_array(arrays["tcr_gacr"], temp, idx_g1, idx_g, density_idx, idx_r1, idx_r)
    tnr_racg = _table_gather_array(arrays["tnr_racg"], temp, idx_g1, idx_g, density_idx, idx_r1, idx_r)
    tnr_gacr = _table_gather_array(arrays["tnr_gacr"], temp, idx_g1, idx_g, density_idx, idx_r1, idx_r)

    prg_rcg_cold = _min_first(rr * _scalar(odt, rr), tmr_racg + tcr_gacr)
    pnr_rcg_cold = _min_first(nr * _scalar(odt, nr), tnr_racg + tnr_gacr)
    prr_rcg_warm = _min_first(rg * _scalar(odt, rg), _table_gather_array(arrays["tcg_racg"], temp, idx_g1, idx_g, density_idx, idx_r1, idx_r))
    png_rcg_warm = _min_first(ng * _scalar(odt, ng), tnr_racg)
    prg_rcg_warm = -prr_rcg_warm
    rho_g = _take(_STATIC_GRAUPEL_RHO_G, density_idx, temp)

    _masked_update_mapping(
        updated,
        cold_rg,
        {
            "prg_rcg": prg_rcg_cold,
            "prr_rcg": -prg_rcg_cold,
            "pnr_rcg": pnr_rcg_cold,
            "pbg_rcg": _scalar(constants.METERS3_TO_LITERS / constants.RHO_I, prg_rcg_cold) * prg_rcg_cold,
        },
    )
    _masked_update_mapping(
        updated,
        warm_rg,
        {
            "prg_rcg": prg_rcg_warm,
            "prr_rcg": prr_rcg_warm,
            "pnr_rcg": -_scalar(1.5, temp) * tnr_gacr,
            "pbg_rcg": _scalar(constants.METERS3_TO_LITERS, prg_rcg_warm) * prg_rcg_warm / rho_g,
            "png_rcg": png_rcg_warm,
        },
    )
    return updated


def _table_gather_tensor_no_cache(table: torch.Tensor, *indices_1based: torch.Tensor) -> torch.Tensor:
    return table[tuple(index.to(torch.int64) - 1 for index in indices_1based)]


def _rain_table_indices_no_cache(
    rr: torch.Tensor,
    ilamr: torch.Tensor,
    _const_segment_h_NIR2=constants.NIR2,
    _const_segment_h_NTB_R=constants.NTB_R,
    _const_segment_h_NIR3=constants.NIR3,
    _const_segment_h_NTB_R1=constants.NTB_R1,
    _const_segment_e_CRG_2=constants.CRG[2],
    _const_segment_e_ORG2=constants.ORG2,
    _const_segment_e_CRG_0=constants.CRG[0],
    _const_segment_e_BM_R=constants.BM_R,
    _const_segment_e_AM_R=constants.AM_R,
    _const_SCALAR_CRE_0=_SCALAR_CRE[0],
) -> tuple[torch.Tensor, torch.Tensor]:
    idx_r = _mass_table_index_no_cache(rr, _const_segment_h_NIR2, _const_segment_h_NTB_R)
    ilamr_safe = torch.where(ilamr > 0.0, ilamr, 1.0)
    lamr = 1.0 / ilamr_safe
    lam_exp = lamr * float(
        (_const_segment_e_CRG_2 * _const_segment_e_ORG2 * (1.0 / _const_segment_e_CRG_0))
        ** _const_segment_e_BM_R
    )
    n0_exp = float(1.0 / _const_segment_e_CRG_0 / _const_segment_e_AM_R) * rr * torch.pow(
        lam_exp,
        float(_const_SCALAR_CRE_0),
    )
    idx_r1 = _mass_table_index_no_cache(n0_exp, _const_segment_h_NIR3, _const_segment_h_NTB_R1)
    return idx_r, idx_r1


def _graupel_table_indices_no_cache(
    rg: torch.Tensor,
    ilamg: torch.Tensor,
    density_idx: torch.Tensor,
    am_g_table: torch.Tensor,
    _const_segment_h_NIG2=constants.NIG2,
    _const_segment_h_NTB_G=constants.NTB_G,
    _const_segment_h_NIG3=constants.NIG3,
    _const_segment_h_NTB_G1=constants.NTB_G1,
    _const_segment_g_CGG_2_0=constants.CGG[2][0],
    _const_segment_g_OGG2=constants.OGG2,
    _const_segment_g_OGG1=constants.OGG1,
    _const_segment_g_BM_G=constants.BM_G,
    _const_SCALAR_GRAUPEL_CGE_0_0=_SCALAR_GRAUPEL_CGE[0][0],
) -> tuple[torch.Tensor, torch.Tensor]:
    idx_g = _mass_table_index_no_cache(rg, _const_segment_h_NIG2, _const_segment_h_NTB_G)
    ilamg_safe = torch.where(ilamg > 0.0, ilamg, 1.0)
    lamg = 1.0 / ilamg_safe
    lam_exp = lamg * float(
        (_const_segment_g_CGG_2_0 * _const_segment_g_OGG2 * _const_segment_g_OGG1)
        ** _const_segment_g_BM_G
    )
    am_g = am_g_table[density_idx - 1]
    n0_exp = float(_const_segment_g_OGG1) * rg / am_g * torch.pow(
        lam_exp,
        float(_const_SCALAR_GRAUPEL_CGE_0_0),
    )
    idx_g1 = _mass_table_index_no_cache(n0_exp, _const_segment_h_NIG3, _const_segment_h_NTB_G1)
    return idx_g, idx_g1


def _rain_snow_rain_graupel_tendencies_kernel(
    temp: torch.Tensor,
    l_qr: torch.Tensor,
    rr: torch.Tensor,
    nr: torch.Tensor,
    ilamr: torch.Tensor,
    l_qs: torch.Tensor,
    rs: torch.Tensor,
    l_qg: torch.Tensor,
    rg: torch.Tensor,
    ng: torch.Tensor,
    ilamg: torch.Tensor,
    idx: torch.Tensor,
    active_micro: torch.Tensor,
    prr_rcs: torch.Tensor,
    prs_rcs: torch.Tensor,
    prg_rcs: torch.Tensor,
    pnr_rcs: torch.Tensor,
    png_rcs: torch.Tensor,
    pbg_rcs: torch.Tensor,
    prg_rcg: torch.Tensor,
    prr_rcg: torch.Tensor,
    pnr_rcg: torch.Tensor,
    pbg_rcg: torch.Tensor,
    png_rcg: torch.Tensor,
    tmr_racs2_table: torch.Tensor,
    tcr_sacr2_table: torch.Tensor,
    tmr_racs1_table: torch.Tensor,
    tcr_sacr1_table: torch.Tensor,
    tcs_racs1_table: torch.Tensor,
    tms_sacr1_table: torch.Tensor,
    tnr_racs1_table: torch.Tensor,
    tnr_racs2_table: torch.Tensor,
    tnr_sacr1_table: torch.Tensor,
    tnr_sacr2_table: torch.Tensor,
    tmr_racg_table: torch.Tensor,
    tcr_gacr_table: torch.Tensor,
    tnr_racg_table: torch.Tensor,
    tnr_gacr_table: torch.Tensor,
    tcg_racg_table: torch.Tensor,
    am_g_table: torch.Tensor,
    rho_g_table: torch.Tensor,
    odt: float,
    _const_RAIN_THRESHOLD=_SCALAR_RAIN_WATER_THRESHOLDS[0],
    _const_SNOW_THRESHOLD=_SCALAR_SNOW_WATER_THRESHOLDS[0],
    _const_GRAUPEL_THRESHOLD=_SCALAR_GRAUPEL_WATER_THRESHOLDS[0],
    _const_segment_e_T0=constants.T0,
    _const_segment_h_NTB_T=constants.NTB_T,
    _const_segment_h_NIS2=constants.NIS2,
    _const_segment_h_NTB_S=constants.NTB_S,
    _const_segment_g_NRHG=constants.NRHG,
    _const_segment_g_METERS3_TO_LITERS=constants.METERS3_TO_LITERS,
    _const_segment_e_RHO_I=constants.RHO_I,
) -> tuple[torch.Tensor, ...]:
    rain_snow = active_micro & l_qr & l_qs & (rr >= float(_const_RAIN_THRESHOLD)) & (rs >= float(_const_SNOW_THRESHOLD))
    cold_rs = rain_snow & (temp < float(_const_segment_e_T0))
    warm_rs = rain_snow & ~cold_rs
    tempc = temp - float(_const_segment_e_T0)
    idx_t_raw = torch.trunc((tempc - 2.5) / 5.0).to(torch.int64) - 1
    idx_t = torch.clamp(-idx_t_raw, 1, _const_segment_h_NTB_T)
    idx_r, idx_r1 = _rain_table_indices_no_cache(rr, ilamr)
    idx_s = _mass_table_index_no_cache(rs, _const_segment_h_NIS2, _const_segment_h_NTB_S)

    tmr_racs2 = _table_gather_tensor_no_cache(tmr_racs2_table, idx_s, idx_t, idx_r1, idx_r)
    tcr_sacr2 = _table_gather_tensor_no_cache(tcr_sacr2_table, idx_s, idx_t, idx_r1, idx_r)
    tmr_racs1 = _table_gather_tensor_no_cache(tmr_racs1_table, idx_s, idx_t, idx_r1, idx_r)
    tcr_sacr1 = _table_gather_tensor_no_cache(tcr_sacr1_table, idx_s, idx_t, idx_r1, idx_r)
    tcs_racs1 = _table_gather_tensor_no_cache(tcs_racs1_table, idx_s, idx_t, idx_r1, idx_r)
    tms_sacr1 = _table_gather_tensor_no_cache(tms_sacr1_table, idx_s, idx_t, idx_r1, idx_r)

    prr_rcs_cold = torch.maximum(-rr * odt, -(tmr_racs2 + tcr_sacr2 + tmr_racs1 + tcr_sacr1))
    prs_rcs_cold = torch.maximum(-rs * odt, tmr_racs2 + tcr_sacr2 - tcs_racs1 - tms_sacr1)
    prg_rcs_cold = torch.minimum((rr + rs) * odt, tmr_racs1 + tcr_sacr1 + tcs_racs1 + tms_sacr1)
    pnr_rcs_cold = torch.minimum(
        nr * odt,
        _table_gather_tensor_no_cache(tnr_racs1_table, idx_s, idx_t, idx_r1, idx_r)
        + _table_gather_tensor_no_cache(tnr_racs2_table, idx_s, idx_t, idx_r1, idx_r)
        + _table_gather_tensor_no_cache(tnr_sacr1_table, idx_s, idx_t, idx_r1, idx_r)
        + _table_gather_tensor_no_cache(tnr_sacr2_table, idx_s, idx_t, idx_r1, idx_r),
    )
    prs_rcs_warm = torch.maximum(-rs * odt, -tcs_racs1 - tms_sacr1 + tmr_racs2 + tcr_sacr2)
    prr_rcs = torch.where(cold_rs, prr_rcs_cold, torch.where(warm_rs, -prs_rcs_warm, prr_rcs))
    prs_rcs = torch.where(cold_rs, prs_rcs_cold, torch.where(warm_rs, prs_rcs_warm, prs_rcs))
    prg_rcs = torch.where(cold_rs, prg_rcs_cold, prg_rcs)
    pnr_rcs = torch.where(cold_rs, pnr_rcs_cold, pnr_rcs)
    png_rcs = torch.where(cold_rs, pnr_rcs_cold, png_rcs)
    pbg_rcs = torch.where(cold_rs, float(_const_segment_g_METERS3_TO_LITERS / _const_segment_e_RHO_I) * prg_rcs_cold, pbg_rcs)

    rain_graupel = active_micro & l_qr & l_qg & (rr >= float(_const_RAIN_THRESHOLD)) & (rg >= float(_const_GRAUPEL_THRESHOLD))
    cold_rg = rain_graupel & (temp < float(_const_segment_e_T0))
    warm_rg = rain_graupel & ~cold_rg
    density_idx = idx.to(torch.int64).clamp(1, _const_segment_g_NRHG)
    idx_g, idx_g1 = _graupel_table_indices_no_cache(rg, ilamg, density_idx, am_g_table)
    tmr_racg = _table_gather_tensor_no_cache(tmr_racg_table, idx_g1, idx_g, density_idx, idx_r1, idx_r)
    tcr_gacr = _table_gather_tensor_no_cache(tcr_gacr_table, idx_g1, idx_g, density_idx, idx_r1, idx_r)
    tnr_racg = _table_gather_tensor_no_cache(tnr_racg_table, idx_g1, idx_g, density_idx, idx_r1, idx_r)
    tnr_gacr = _table_gather_tensor_no_cache(tnr_gacr_table, idx_g1, idx_g, density_idx, idx_r1, idx_r)

    prg_rcg_cold = torch.minimum(rr * odt, tmr_racg + tcr_gacr)
    pnr_rcg_cold = torch.minimum(nr * odt, tnr_racg + tnr_gacr)
    prr_rcg_warm = torch.minimum(
        rg * odt,
        _table_gather_tensor_no_cache(tcg_racg_table, idx_g1, idx_g, density_idx, idx_r1, idx_r),
    )
    png_rcg_warm = torch.minimum(ng * odt, tnr_racg)
    prg_rcg_warm = -prr_rcg_warm
    rho_g = rho_g_table[density_idx - 1]
    prg_rcg = torch.where(cold_rg, prg_rcg_cold, torch.where(warm_rg, prg_rcg_warm, prg_rcg))
    prr_rcg = torch.where(cold_rg, -prg_rcg_cold, torch.where(warm_rg, prr_rcg_warm, prr_rcg))
    pnr_rcg = torch.where(cold_rg, pnr_rcg_cold, torch.where(warm_rg, -1.5 * tnr_gacr, pnr_rcg))
    pbg_rcg = torch.where(
        cold_rg,
        float(_const_segment_g_METERS3_TO_LITERS / _const_segment_e_RHO_I) * prg_rcg_cold,
        torch.where(warm_rg, float(_const_segment_g_METERS3_TO_LITERS) * prg_rcg_warm / rho_g, pbg_rcg),
    )
    png_rcg = torch.where(warm_rg, png_rcg_warm, png_rcg)
    return (
        prr_rcs,
        prs_rcs,
        prg_rcs,
        pnr_rcs,
        png_rcs,
        pbg_rcs,
        prg_rcg,
        prr_rcg,
        pnr_rcg,
        pbg_rcg,
        png_rcg,
    )


def run_i(
    state: Mapping[str, torch.Tensor] | TorchSegmentHResult,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    copy: bool = False,
) -> TorchSegmentIResult:
    h_result = _segment_h_result(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
    )
    input_state = h_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = h_result.workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment I dt must be nonzero")
    odt = 1.0 / resolved_dt
    resolved_tables = tables if tables is not None else h_result.workspace.tables
    g_workspace = h_result.workspace.base
    d_workspace = g_workspace.base.base.base
    e_workspace = g_workspace.base.base
    c_base = g_workspace.base.base.base.base
    tend = _tensor_mapping(h_result.workspace.tend, copy=copy)
    tendencies = _tensor_mapping(h_result.workspace.tendencies, copy=copy)
    idx_in = _int_scalar(0, c_base.temp, dtype=torch.int32)
    idx_tc = _int_scalar(0, c_base.temp, dtype=torch.int32)

    if not turn_off_micro:
        if resolved_tables is None:
            raise ValueError("Segment I requires prepared TEMPO coefficient tables")
        active_micro = g_workspace.do_micro | g_workspace.supersaturated
        tend, idx_in, idx_tc = ice_nucleation_tendencies(
            temp=c_base.temp,
            rho=c_base.rho,
            w1d=input_state["w"],
            qv=c_base.qv,
            qvsi=g_workspace.thermo["qvsi"],
            ssati=g_workspace.thermo["ssati"],
            ssatw=g_workspace.thermo["ssatw"],
            nwfa=d_workspace.nwfa,
            nifa=d_workspace.nifa,
            ni=e_workspace.ni,
            smo0=e_workspace.snow_moments["smo0"],
            rc=g_workspace.base.rc,
            nc=g_workspace.base.nc,
            rr=g_workspace.base.base.rr,
            nr=g_workspace.base.base.nr,
            ilamr=g_workspace.base.base.ilamr,
            tend=tend,
            tables=resolved_tables,
            dt=resolved_dt,
            odt=odt,
            nwfa_present=d_workspace.nwfa_present,
            nifa_present=d_workspace.nifa_present,
            active_micro=active_micro,
            rain_table_indices=h_result.workspace.rain_table_indices,
        )

    workspace = TorchSegmentIWorkspace(
        base=h_result.workspace,
        tend=tend,
        tendencies=tendencies,
        tables=resolved_tables,
        idx_in=idx_in,
        idx_tc=idx_tc,
        nwfa=d_workspace.nwfa,
        nifa=d_workspace.nifa,
        dt=resolved_dt,
        odt=odt,
    )
    return TorchSegmentIResult(result_state, workspace, h_result.ncol, h_result.nz)


def ice_nucleation_tendencies(
    *,
    temp: torch.Tensor,
    rho: torch.Tensor,
    w1d: torch.Tensor,
    qv: torch.Tensor,
    qvsi: torch.Tensor,
    ssati: torch.Tensor,
    ssatw: torch.Tensor,
    nwfa: torch.Tensor,
    nifa: torch.Tensor,
    ni: torch.Tensor,
    smo0: torch.Tensor,
    rc: torch.Tensor,
    nc: torch.Tensor,
    rr: torch.Tensor,
    nr: torch.Tensor,
    ilamr: torch.Tensor,
    tend: Mapping[str, torch.Tensor],
    tables: Any,
    dt: float,
    odt: float,
    nwfa_present: bool,
    nifa_present: bool,
    active_micro: torch.Tensor,
    rain_table_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    if not (
        TENSOR_COMPILE_ISLAND_CUDA
        and temp.device.type in TENSOR_COMPILE_ISLAND_DEVICES
        and not torch.compiler.is_compiling()
    ):
        return _ice_nucleation_tendencies_eager(
            temp=temp,
            rho=rho,
            w1d=w1d,
            qv=qv,
            qvsi=qvsi,
            ssati=ssati,
            ssatw=ssatw,
            nwfa=nwfa,
            nifa=nifa,
            ni=ni,
            smo0=smo0,
            rc=rc,
            nc=nc,
            rr=rr,
            nr=nr,
            ilamr=ilamr,
            tend=tend,
            tables=tables,
            dt=dt,
            odt=odt,
            nwfa_present=nwfa_present,
            nifa_present=nifa_present,
            active_micro=active_micro,
            rain_table_indices=rain_table_indices,
        )
    names = (
        "prg_rfz",
        "pri_rfz",
        "pni_rfz",
        "pnr_rfz",
        "png_rfz",
        "pbg_rfz",
        "pri_wfz",
        "pni_wfz",
        "pri_inu",
        "pni_inu",
        "pni_iha",
        "pri_iha",
    )
    arrays = tables.arrays
    values = _run_tensor_compile_island(
        "ice_nucleation_tendencies",
        _ice_nucleation_tendencies_kernel,
        temp,
        temp,
        rho,
        w1d,
        qv,
        qvsi,
        ssati,
        ssatw,
        nwfa,
        nifa,
        ni,
        smo0,
        rc,
        nc,
        rr,
        nr,
        ilamr,
        active_micro,
        *tuple(tend[name] for name in names),
        *tuple(_tensor(arrays[name], temp) for name in ("tpg_qrfz", "tpi_qrfz", "tni_qrfz", "tnr_qrfz", "tpi_qcfz", "tni_qcfz")),
        _tensor(_STATIC_CLOUD_NUMBER_BINS, temp),
        float(dt),
        float(odt),
        bool(nwfa_present),
        bool(nifa_present),
    )
    updated = dict(tend)
    for name, value in zip(names, values[: len(names)], strict=True):
        updated[name] = value
    return updated, values[-2], values[-1]


def _ice_nucleation_tendencies_eager(
    *,
    temp: torch.Tensor,
    rho: torch.Tensor,
    w1d: torch.Tensor,
    qv: torch.Tensor,
    qvsi: torch.Tensor,
    ssati: torch.Tensor,
    ssatw: torch.Tensor,
    nwfa: torch.Tensor,
    nifa: torch.Tensor,
    ni: torch.Tensor,
    smo0: torch.Tensor,
    rc: torch.Tensor,
    nc: torch.Tensor,
    rr: torch.Tensor,
    nr: torch.Tensor,
    ilamr: torch.Tensor,
    tend: Mapping[str, torch.Tensor],
    tables: Any,
    dt: float,
    odt: float,
    nwfa_present: bool,
    nifa_present: bool,
    active_micro: torch.Tensor,
    rain_table_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    updated = dict(tend)
    arrays = tables.arrays
    cold = active_micro & (temp < _scalar(constants.T0, temp))
    tempc = temp - _scalar(constants.T0, temp)
    idx_tc = torch.where(
        cold,
        _freezing_temperature_index(temp).to(torch.int32),
        _int_scalar(0, temp, dtype=torch.int32),
    )
    xni_seed = _demott_nucleation(tempc, rho, nifa) if nifa_present else _scalar(1000.0, temp)
    idx_in = torch.where(cold, _in_table_index(xni_seed).to(torch.int32), _int_scalar(0, temp, dtype=torch.int32))
    idx_in_i64 = idx_in.to(torch.int64).clamp(1, constants.NTB_IN)
    idx_tc_i64 = idx_tc.to(torch.int64).clamp(1, constants.NTB_T1)
    rate_max = (qv - qvsi) * rho * odt * 0.999

    rain_table = cold & (rr > _scalar(_SCALAR_RAIN_WATER_THRESHOLDS[0], rr))
    idx_r, idx_r1 = _rain_table_indices(rr, ilamr) if rain_table_indices is None else rain_table_indices
    prg_rfz = _table_gather_array(arrays["tpg_qrfz"], temp, idx_r, idx_r1, idx_tc_i64, idx_in_i64) * _scalar(odt, temp)
    pri_rfz = _table_gather_array(arrays["tpi_qrfz"], temp, idx_r, idx_r1, idx_tc_i64, idx_in_i64) * _scalar(odt, temp)
    pni_rfz = _table_gather_array(arrays["tni_qrfz"], temp, idx_r, idx_r1, idx_tc_i64, idx_in_i64) * _scalar(odt, temp)
    pnr_rfz = _table_gather_array(arrays["tnr_qrfz"], temp, idx_r, idx_r1, idx_tc_i64, idx_in_i64) * _scalar(odt, temp)
    prg_rfz = _min_first(rr * _scalar(odt, rr), prg_rfz)
    pnr_rfz = _min_first(nr * _scalar(odt, nr), pnr_rfz)
    png_rfz = pnr_rfz * torch.clamp(
        working_power(_scalar(10.0, w1d), -_scalar(0.1, w1d) * w1d) + _scalar(0.1, w1d),
        min=0.1,
        max=1.0,
    )
    rain_bulk = cold & ~rain_table & (rr > _scalar(constants.R1, rr)) & (temp < _scalar(constants.HGFRZ, temp))
    updated["prg_rfz"] = torch.where(rain_table, prg_rfz, updated["prg_rfz"])
    updated["pri_rfz"] = torch.where(rain_table, pri_rfz, torch.where(rain_bulk, rr * _scalar(odt, rr), updated["pri_rfz"]))
    updated["pni_rfz"] = torch.where(rain_table, pni_rfz, torch.where(rain_bulk, nr * _scalar(odt, nr), updated["pni_rfz"]))
    updated["pnr_rfz"] = torch.where(rain_table, pnr_rfz, updated["pnr_rfz"])
    updated["png_rfz"] = torch.where(rain_table, png_rfz, updated["png_rfz"])
    updated["pbg_rfz"] = torch.where(
        cold,
        _scalar(constants.METERS3_TO_LITERS / constants.RHO_I, temp) * updated["prg_rfz"],
        updated["pbg_rfz"],
    )

    cloud_table = cold & (rc > _scalar(_SCALAR_CLOUD_WATER_THRESHOLDS[0], rc))
    idx_c, idx_n = _cloud_table_indices(rc, nc)
    pri_wfz = _table_gather_array(arrays["tpi_qcfz"], temp, idx_c, idx_n, idx_tc_i64, idx_in_i64) * _scalar(odt, temp)
    pri_wfz = _min_first(rc * _scalar(odt, rc), pri_wfz)
    pni_wfz = _table_gather_array(arrays["tni_qcfz"], temp, idx_c, idx_n, idx_tc_i64, idx_in_i64) * _scalar(odt, temp)
    pni_wfz = _min_first(
        _min_first(nc * _scalar(odt, nc), pri_wfz / _scalar(2.0 * constants.XM0I, pri_wfz)),
        pni_wfz,
    )
    cloud_bulk = cold & ~cloud_table & (rc > _scalar(constants.R1, rc)) & (temp < _scalar(constants.HGFRZ, temp))
    updated["pri_wfz"] = torch.where(cloud_table, pri_wfz, torch.where(cloud_bulk, rc * _scalar(odt, rc), updated["pri_wfz"]))
    updated["pni_wfz"] = torch.where(cloud_table, pni_wfz, torch.where(cloud_bulk, nc * _scalar(odt, nc), updated["pni_wfz"]))

    demott_mask = cold & (
        (ssati >= _scalar(constants.DEMOTT_NUC_SSATI, ssati))
        | ((ssatw > _scalar(constants.EPS, ssatw)) & (tempc < _scalar(constants.DEMOTT_NUC_TEMPC, tempc)))
    )
    if nifa_present:
        xnc = _demott_nucleation(tempc, rho, nifa)
    else:
        xnc = torch.clamp_max(
            _scalar(constants.TNO, temp) * torch.exp(_scalar(constants.ATO, temp) * (_scalar(constants.T0, temp) - temp)),
            float(constants.ICENUC_MAX),
        )
    xni_after_freezing = ni + (updated["pni_rfz"] + updated["pni_wfz"]) * _scalar(dt, ni)
    pni_inu = _scalar(0.5, temp) * (xnc - xni_after_freezing + torch.abs(xnc - xni_after_freezing)) * _scalar(odt, temp)
    pri_inu = _min_first(rate_max, _scalar(constants.XM0I, temp) * pni_inu)
    pni_inu = pri_inu / _scalar(constants.XM0I, temp)
    updated["pri_inu"] = torch.where(demott_mask, pri_inu, updated["pri_inu"])
    updated["pni_inu"] = torch.where(demott_mask, pni_inu, updated["pni_inu"])

    if nwfa_present:
        xni_total = smo0 + ni + (updated["pni_rfz"] + updated["pni_wfz"] + updated["pni_inu"]) * _scalar(dt, ni)
        homogeneous = cold & (xni_total <= _scalar(constants.MAX_NI, xni_total)) & (temp < _scalar(238.0, temp)) & (ssati >= _scalar(0.4, ssati))
        xnc_h = _koop_nucleation(temp, ssatw, nwfa, dt)
        pni_iha = xnc_h * _scalar(odt, xnc_h)
        pri_iha = _min_first(rate_max, _scalar(constants.XM0I * 0.1, temp) * pni_iha)
        pni_iha = pri_iha / _scalar(constants.XM0I * 0.1, temp)
        updated["pni_iha"] = torch.where(homogeneous, pni_iha, updated["pni_iha"])
        updated["pri_iha"] = torch.where(homogeneous, pri_iha, updated["pri_iha"])

    return updated, idx_in, idx_tc


def _demott_nucleation_no_cache(
    tempc: torch.Tensor,
    rho: torch.Tensor,
    nifa: torch.Tensor,
    _const_segment_i_RHO_NOT0=constants.RHO_NOT0,
) -> torch.Tensor:
    nifa_cc = torch.clamp_min(nifa * float(_const_segment_i_RHO_NOT0 * 1.0e-6) / rho, 0.5)
    cold_magnitude = torch.clamp_min(-tempc, 0.0)
    xni = 5.94e-5 * torch.pow(cold_magnitude, 3.33) * torch.pow(
        nifa_cc,
        -0.0264 * tempc + 0.0033,
    )
    xni = xni * rho / float(_const_segment_i_RHO_NOT0) * 1000.0
    return torch.clamp_min(xni, 0.0)


def _koop_nucleation_no_cache(
    temp: torch.Tensor,
    satw: torch.Tensor,
    naero: torch.Tensor,
    dt: float,
    _const_segment_i_R_UNI=constants.R_UNI,
    _const_segment_i_AR_VOLUME=constants.AR_VOLUME,
) -> torch.Tensor:
    mu_diff = 210368.0 + 131.438 * temp - 3.32373e6 / temp - 41729.1 * torch.log(temp)
    a_w_i = torch.exp(mu_diff / (float(_const_segment_i_R_UNI) * temp))
    delta_aw = satw - a_w_i
    log_j_rate = -906.7 + 8502.0 * delta_aw - 26924.0 * delta_aw * delta_aw + 29180.0 * delta_aw * delta_aw * delta_aw
    log_j_rate = torch.clamp_max(log_j_rate, 20.0)
    j_rate = torch.pow(10.0, log_j_rate)
    prob_h = torch.clamp_max(1.0 - torch.exp(-j_rate * float(_const_segment_i_AR_VOLUME * dt)), 1.0)
    xni = torch.where(prob_h > 0.0, torch.clamp_max(prob_h * naero, 1000.0e3), 0.0)
    return torch.clamp_min(xni, 0.0)


def _ice_nucleation_tendencies_kernel(
    temp: torch.Tensor,
    rho: torch.Tensor,
    w1d: torch.Tensor,
    qv: torch.Tensor,
    qvsi: torch.Tensor,
    ssati: torch.Tensor,
    ssatw: torch.Tensor,
    nwfa: torch.Tensor,
    nifa: torch.Tensor,
    ni: torch.Tensor,
    smo0: torch.Tensor,
    rc: torch.Tensor,
    nc: torch.Tensor,
    rr: torch.Tensor,
    nr: torch.Tensor,
    ilamr: torch.Tensor,
    active_micro: torch.Tensor,
    prg_rfz: torch.Tensor,
    pri_rfz: torch.Tensor,
    pni_rfz: torch.Tensor,
    pnr_rfz: torch.Tensor,
    png_rfz: torch.Tensor,
    pbg_rfz: torch.Tensor,
    pri_wfz: torch.Tensor,
    pni_wfz: torch.Tensor,
    pri_inu: torch.Tensor,
    pni_inu: torch.Tensor,
    pni_iha: torch.Tensor,
    pri_iha: torch.Tensor,
    tpg_qrfz_table: torch.Tensor,
    tpi_qrfz_table: torch.Tensor,
    tni_qrfz_table: torch.Tensor,
    tnr_qrfz_table: torch.Tensor,
    tpi_qcfz_table: torch.Tensor,
    tni_qcfz_table: torch.Tensor,
    cloud_number_bins: torch.Tensor,
    dt: float,
    odt: float,
    nwfa_present: bool,
    nifa_present: bool,
    _const_segment_e_T0=constants.T0,
    _const_segment_i_NTB_T1=constants.NTB_T1,
    _const_segment_i_NTB_IN=constants.NTB_IN,
    _const_segment_i_NIIN2=constants.NIIN2,
    _const_segment_i_HGFRZ=constants.HGFRZ,
    _const_segment_g_METERS3_TO_LITERS=constants.METERS3_TO_LITERS,
    _const_segment_e_RHO_I=constants.RHO_I,
    _const_CLOUD_THRESHOLD=_SCALAR_CLOUD_WATER_THRESHOLDS[0],
    _const_RAIN_THRESHOLD=_SCALAR_RAIN_WATER_THRESHOLDS[0],
    _const_segment_e_R1=constants.R1,
    _const_segment_i_XM0I=constants.XM0I,
    _const_segment_i_DEMOTT_NUC_SSATI=constants.DEMOTT_NUC_SSATI,
    _const_segment_g_EPS=constants.EPS,
    _const_segment_i_DEMOTT_NUC_TEMPC=constants.DEMOTT_NUC_TEMPC,
    _const_segment_i_TNO=constants.TNO,
    _const_segment_i_ATO=constants.ATO,
    _const_segment_i_ICENUC_MAX=constants.ICENUC_MAX,
    _const_segment_i_MAX_NI=constants.MAX_NI,
) -> tuple[torch.Tensor, ...]:
    cold = active_micro & (temp < float(_const_segment_e_T0))
    tempc = temp - float(_const_segment_e_T0)
    nint_tc = torch.where(-tempc >= 0.0, torch.floor(-tempc + 0.5), torch.ceil(-tempc - 0.5)).to(torch.int64)
    idx_tc_active = torch.clamp(nint_tc, 1, _const_segment_i_NTB_T1).to(torch.int32)
    idx_tc = torch.where(cold, idx_tc_active, torch.zeros_like(idx_tc_active))
    xni_seed = _demott_nucleation_no_cache(tempc, rho, nifa) if nifa_present else torch.full_like(temp, 1000.0)
    idx_in_active = torch.where(
        xni_seed > 1.0,
        _mass_table_index_no_cache(xni_seed, _const_segment_i_NIIN2, _const_segment_i_NTB_IN),
        torch.ones_like(xni_seed, dtype=torch.int64),
    ).to(torch.int32)
    idx_in = torch.where(cold, idx_in_active, torch.zeros_like(idx_in_active))
    idx_in_i64 = idx_in.to(torch.int64).clamp(1, _const_segment_i_NTB_IN)
    idx_tc_i64 = idx_tc.to(torch.int64).clamp(1, _const_segment_i_NTB_T1)
    rate_max = (qv - qvsi) * rho * odt * 0.999

    rain_table = cold & (rr > float(_const_RAIN_THRESHOLD))
    idx_r, idx_r1 = _rain_table_indices_no_cache(rr, ilamr)
    prg_rfz_candidate = torch.minimum(rr * odt, _table_gather_tensor_no_cache(tpg_qrfz_table, idx_r, idx_r1, idx_tc_i64, idx_in_i64) * odt)
    pri_rfz_candidate = _table_gather_tensor_no_cache(tpi_qrfz_table, idx_r, idx_r1, idx_tc_i64, idx_in_i64) * odt
    pni_rfz_candidate = _table_gather_tensor_no_cache(tni_qrfz_table, idx_r, idx_r1, idx_tc_i64, idx_in_i64) * odt
    pnr_rfz_candidate = torch.minimum(nr * odt, _table_gather_tensor_no_cache(tnr_qrfz_table, idx_r, idx_r1, idx_tc_i64, idx_in_i64) * odt)
    png_rfz_candidate = pnr_rfz_candidate * torch.clamp(torch.pow(10.0, -0.1 * w1d) + 0.1, min=0.1, max=1.0)
    rain_bulk = cold & ~rain_table & (rr > float(_const_segment_e_R1)) & (temp < float(_const_segment_i_HGFRZ))
    prg_rfz = torch.where(rain_table, prg_rfz_candidate, prg_rfz)
    pri_rfz = torch.where(rain_table, pri_rfz_candidate, torch.where(rain_bulk, rr * odt, pri_rfz))
    pni_rfz = torch.where(rain_table, pni_rfz_candidate, torch.where(rain_bulk, nr * odt, pni_rfz))
    pnr_rfz = torch.where(rain_table, pnr_rfz_candidate, pnr_rfz)
    png_rfz = torch.where(rain_table, png_rfz_candidate, png_rfz)
    pbg_rfz = torch.where(cold, float(_const_segment_g_METERS3_TO_LITERS / _const_segment_e_RHO_I) * prg_rfz, pbg_rfz)

    cloud_table = cold & (rc > float(_const_CLOUD_THRESHOLD))
    idx_c, idx_n = _cloud_table_indices_no_cache(rc, nc, cloud_number_bins)
    pri_wfz_candidate = torch.minimum(rc * odt, _table_gather_tensor_no_cache(tpi_qcfz_table, idx_c, idx_n, idx_tc_i64, idx_in_i64) * odt)
    pni_wfz_candidate = _table_gather_tensor_no_cache(tni_qcfz_table, idx_c, idx_n, idx_tc_i64, idx_in_i64) * odt
    pni_wfz_candidate = torch.minimum(torch.minimum(nc * odt, pri_wfz_candidate / float(2.0 * _const_segment_i_XM0I)), pni_wfz_candidate)
    cloud_bulk = cold & ~cloud_table & (rc > float(_const_segment_e_R1)) & (temp < float(_const_segment_i_HGFRZ))
    pri_wfz = torch.where(cloud_table, pri_wfz_candidate, torch.where(cloud_bulk, rc * odt, pri_wfz))
    pni_wfz = torch.where(cloud_table, pni_wfz_candidate, torch.where(cloud_bulk, nc * odt, pni_wfz))

    demott_mask = cold & ((ssati >= float(_const_segment_i_DEMOTT_NUC_SSATI)) | ((ssatw > float(_const_segment_g_EPS)) & (tempc < float(_const_segment_i_DEMOTT_NUC_TEMPC))))
    xnc = _demott_nucleation_no_cache(tempc, rho, nifa) if nifa_present else torch.clamp_max(float(_const_segment_i_TNO) * torch.exp(float(_const_segment_i_ATO) * (float(_const_segment_e_T0) - temp)), float(_const_segment_i_ICENUC_MAX))
    xni_after_freezing = ni + (pni_rfz + pni_wfz) * dt
    pni_inu_candidate = 0.5 * (xnc - xni_after_freezing + torch.abs(xnc - xni_after_freezing)) * odt
    pri_inu_candidate = torch.minimum(rate_max, float(_const_segment_i_XM0I) * pni_inu_candidate)
    pni_inu_candidate = pri_inu_candidate / float(_const_segment_i_XM0I)
    pri_inu = torch.where(demott_mask, pri_inu_candidate, pri_inu)
    pni_inu = torch.where(demott_mask, pni_inu_candidate, pni_inu)

    if nwfa_present:
        xni_total = smo0 + ni + (pni_rfz + pni_wfz + pni_inu) * dt
        homogeneous = cold & (xni_total <= float(_const_segment_i_MAX_NI)) & (temp < 238.0) & (ssati >= 0.4)
        xnc_h = _koop_nucleation_no_cache(temp, ssatw, nwfa, dt)
        pni_iha_candidate = xnc_h * odt
        pri_iha_candidate = torch.minimum(rate_max, float(_const_segment_i_XM0I * 0.1) * pni_iha_candidate)
        pni_iha_candidate = pri_iha_candidate / float(_const_segment_i_XM0I * 0.1)
        pni_iha = torch.where(homogeneous, pni_iha_candidate, pni_iha)
        pri_iha = torch.where(homogeneous, pri_iha_candidate, pri_iha)
    return (
        prg_rfz,
        pri_rfz,
        pni_rfz,
        pnr_rfz,
        png_rfz,
        pbg_rfz,
        pri_wfz,
        pni_wfz,
        pri_inu,
        pni_inu,
        pni_iha,
        pri_iha,
        idx_in,
        idx_tc,
    )


def run_j(
    state: Mapping[str, torch.Tensor] | TorchSegmentIResult,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    copy: bool = False,
) -> TorchSegmentJResult:
    i_result = _segment_i_result(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
    )
    input_state = i_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = i_result.workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment J dt must be nonzero")
    odt = 1.0 / resolved_dt
    h_workspace = i_result.workspace.base
    g_workspace = h_workspace.base
    f_workspace = g_workspace.base
    e_workspace = f_workspace.base
    c_base = e_workspace.base.base
    tend = _tensor_mapping(i_result.workspace.tend, copy=copy)
    tendencies = _tensor_mapping(i_result.workspace.tendencies, copy=copy)
    t1_subl = _scalar(0.0, c_base.temp)
    idx_i = _int_scalar(0, c_base.temp, dtype=torch.int32)
    idx_i1 = _int_scalar(0, c_base.temp, dtype=torch.int32)
    packed_ice_tendencies: torch.Tensor | None = None

    if not turn_off_micro:
        active_micro = g_workspace.do_micro | g_workspace.supersaturated
        t1_subl = sublimation_factor(
            rho=c_base.rho,
            temp=c_base.temp,
            qvsi=g_workspace.thermo["qvsi"],
            tcond=g_workspace.thermo["tcond"],
            diffu=g_workspace.thermo["diffu"],
            ssati=g_workspace.thermo["ssati"],
        )
        ice_result = ice_process_tendencies(
            rhof=g_workspace.thermo["rhof"],
            rhof2=g_workspace.thermo["rhof2"],
            rho=c_base.rho,
            w1d=input_state["w"],
            temp=c_base.temp,
            qv=c_base.qv,
            qvsi=g_workspace.thermo["qvsi"],
            diffu=g_workspace.thermo["diffu"],
            ssati=g_workspace.thermo["ssati"],
            vsc2=g_workspace.thermo["vsc2"],
            l_qi=e_workspace.l_qi,
            ri=e_workspace.ri,
            ni=e_workspace.ni,
            ilami=e_workspace.ilami,
            l_qs=e_workspace.l_qs,
            rs=e_workspace.rs,
            smoe=e_workspace.snow_moments["smoe"],
            smof=e_workspace.snow_moments["smof"],
            smo1=e_workspace.snow_moments["smo1"],
            rr=e_workspace.rr,
            nr=e_workspace.nr,
            ilamr=e_workspace.ilamr,
            mvd_r=e_workspace.mvd_r,
            l_qg=g_workspace.l_qg,
            rg=g_workspace.rg,
            ng=g_workspace.ng,
            ilamg=g_workspace.ilamg,
            idx=g_workspace.idx_bg,
            tend=tend,
            t1_subl=t1_subl,
            odt=odt,
            active_micro=active_micro,
            return_pack=N_TENDENCY_LAYOUT_MODE == "producer_packed" and TENSOR_COMPILE_ISLAND_CUDA and not copy,
        )
        if len(ice_result) == 4:
            tend, idx_i, idx_i1, packed_ice_tendencies = ice_result
        else:
            tend, idx_i, idx_i1 = ice_result

    workspace = TorchSegmentJWorkspace(
        base=i_result.workspace,
        tend=tend,
        tendencies=tendencies,
        tables=i_result.workspace.tables,
        t1_subl=t1_subl,
        idx_i=idx_i,
        idx_i1=idx_i1,
        dt=resolved_dt,
        odt=odt,
        packed_ice_tendencies=packed_ice_tendencies,
    )
    return TorchSegmentJResult(result_state, workspace, i_result.ncol, i_result.nz)


def sublimation_factor(
    *,
    rho: torch.Tensor,
    temp: torch.Tensor,
    qvsi: torch.Tensor,
    tcond: torch.Tensor,
    diffu: torch.Tensor,
    ssati: torch.Tensor,
) -> torch.Tensor:
    if TENSOR_COMPILE_ISLAND_CUDA and rho.device.type in TENSOR_COMPILE_ISLAND_DEVICES:
        return _run_tensor_compile_island(
            "sublimation_factor",
            _sublimation_factor_no_cache,
            rho,
            rho,
            temp,
            qvsi,
            tcond,
            diffu,
            ssati,
        )
    otemp = 1.0 / temp
    rvs = rho * qvsi
    lsub = _scalar(constants.LSUB, temp)
    rvs_p = rvs * otemp * (lsub * otemp * _scalar(constants.ORV, temp) - 1.0)
    rvs_pp = rvs * (
        otemp * (lsub * otemp * _scalar(constants.ORV, temp) - 1.0) * otemp * (lsub * otemp * _scalar(constants.ORV, temp) - 1.0)
        + (-2.0 * lsub * otemp * otemp * otemp * _scalar(constants.ORV, temp))
        + otemp * otemp
    )
    gamsc = lsub * diffu / tcond * rvs_p
    gamsc_ratio = gamsc / (1.0 + gamsc)
    alphsc = 0.5 * _square(gamsc_ratio) * rvs_pp / rvs_p * rvs / rvs_p
    alphsc = torch.clamp_min(alphsc, _scalar(1.0e-9, alphsc))
    xsat = torch.where(torch.abs(ssati) < _scalar(1.0e-9, ssati), 0.0, ssati)
    return (
        _scalar(4.0 * constants.PI, temp)
        * (1.0 - alphsc * xsat + 2.0 * alphsc * alphsc * xsat * xsat - 5.0 * alphsc * alphsc * alphsc * xsat * xsat * xsat)
        / (1.0 + gamsc)
    )


def _sublimation_factor_no_cache(rho: torch.Tensor, temp: torch.Tensor, qvsi: torch.Tensor, tcond: torch.Tensor, diffu: torch.Tensor, ssati: torch.Tensor, _const_segment_j_LSUB=constants.LSUB, _const_segment_g_ORV=constants.ORV, _const_segment_e_PI=constants.PI) -> torch.Tensor:
    otemp = 1.0 / temp
    rvs = rho * qvsi
    lsub = float(_const_segment_j_LSUB)
    thermal_slope = lsub * otemp * float(_const_segment_g_ORV) - 1.0
    rvs_p = rvs * otemp * thermal_slope
    rvs_pp = rvs * (otemp * thermal_slope * otemp * thermal_slope + -2.0 * lsub * otemp * otemp * otemp * float(_const_segment_g_ORV) + otemp * otemp)
    gamsc = lsub * diffu / tcond * rvs_p
    gamsc_ratio = gamsc / (1.0 + gamsc)
    alphsc = 0.5 * _square(gamsc_ratio) * rvs_pp / rvs_p * rvs / rvs_p
    alphsc = torch.clamp_min(alphsc, 1e-09)
    xsat = torch.where(torch.abs(ssati) < 1e-09, 0.0, ssati)
    return float(4.0 * _const_segment_e_PI) * (1.0 - alphsc * xsat + 2.0 * alphsc * alphsc * xsat * xsat - 5.0 * alphsc * alphsc * alphsc * xsat * xsat * xsat) / (1.0 + gamsc)


def ice_process_tendencies(
    *,
    rhof: torch.Tensor,
    rhof2: torch.Tensor,
    rho: torch.Tensor,
    w1d: torch.Tensor,
    temp: torch.Tensor,
    qv: torch.Tensor,
    qvsi: torch.Tensor,
    diffu: torch.Tensor,
    ssati: torch.Tensor,
    vsc2: torch.Tensor,
    l_qi: torch.Tensor,
    ri: torch.Tensor,
    ni: torch.Tensor,
    ilami: torch.Tensor,
    l_qs: torch.Tensor,
    rs: torch.Tensor,
    smoe: torch.Tensor,
    smof: torch.Tensor,
    smo1: torch.Tensor,
    rr: torch.Tensor,
    nr: torch.Tensor,
    ilamr: torch.Tensor,
    mvd_r: torch.Tensor,
    l_qg: torch.Tensor,
    rg: torch.Tensor,
    ng: torch.Tensor,
    ilamg: torch.Tensor,
    idx: torch.Tensor,
    tend: Mapping[str, torch.Tensor],
    t1_subl: torch.Tensor,
    odt: float,
    active_micro: torch.Tensor,
    return_pack: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor] | tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    names = _ICE_PROCESS_TENDENCY_FIELDS
    updated = dict(tend)
    compile_args = (
        temp,
        rhof,
        rhof2,
        rho,
        w1d,
        temp,
        qv,
        qvsi,
        diffu,
        ssati,
        vsc2,
        l_qi,
        ri,
        ni,
        ilami,
        l_qs,
        rs,
        smoe,
        smof,
        smo1,
        rr,
        nr,
        ilamr,
        mvd_r,
        l_qg,
        rg,
        ng,
        ilamg,
        idx,
        t1_subl,
        active_micro,
        *tuple(updated[name] for name in names),
        *_ice_autoconversion_tensors(temp),
        _tensor(_STATIC_GRAUPEL_AV_G, temp),
        _tensor(_STATIC_GRAUPEL_CGG[10], temp),
        _tensor(_STATIC_GRAUPEL_CGE[10], temp),
        float(odt),
    )
    if return_pack:
        values = _run_tensor_compile_island(
            "ice_process_tendencies_packed", _ice_process_tendencies_packed_kernel, *compile_args
        )
    else:
        values = _run_tensor_compile_island(
            "ice_process_tendencies", _ice_process_tendencies_kernel, *compile_args
        )
    if return_pack:
        packed, idx_i, idx_i1 = values
        field_values = packed.unbind(0)
    else:
        field_values = values[: len(names)]
        idx_i = values[len(names)]
        idx_i1 = values[len(names) + 1]
    for name, value in zip(names, field_values, strict=True):
        updated[name] = value
    if return_pack:
        return updated, idx_i, idx_i1, packed
    return updated, idx_i, idx_i1


_ICE_PROCESS_TENDENCY_FIELDS = (
    "pri_ide",
    "pni_ide",
    "prs_ide",
    "prs_iau",
    "pni_iau",
    "prs_sci",
    "pni_sci",
    "pri_rci",
    "pnr_rci",
    "png_rci",
    "pni_rci",
    "prr_rci",
    "prg_rci",
    "pbg_rci",
    "prs_sde",
    "prg_gde",
    "png_gde",
)


def _ice_process_tendencies_packed_kernel(*args: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = _ice_process_tendencies_kernel(*args)
    return torch.stack(values[: len(_ICE_PROCESS_TENDENCY_FIELDS)], dim=0), values[-2], values[-1]


def _ice_process_tendencies_kernel(rhof: torch.Tensor, rhof2: torch.Tensor, rho: torch.Tensor, w1d: torch.Tensor, temp: torch.Tensor, qv: torch.Tensor, qvsi: torch.Tensor, diffu: torch.Tensor, ssati: torch.Tensor, vsc2: torch.Tensor, l_qi: torch.Tensor, ri: torch.Tensor, ni: torch.Tensor, ilami: torch.Tensor, l_qs: torch.Tensor, rs: torch.Tensor, smoe: torch.Tensor, smof: torch.Tensor, smo1: torch.Tensor, rr: torch.Tensor, nr: torch.Tensor, ilamr: torch.Tensor, mvd_r: torch.Tensor, l_qg: torch.Tensor, rg: torch.Tensor, ng: torch.Tensor, ilamg: torch.Tensor, idx: torch.Tensor, t1_subl: torch.Tensor, active_micro: torch.Tensor, pri_ide: torch.Tensor, pni_ide: torch.Tensor, prs_ide: torch.Tensor, prs_iau: torch.Tensor, pni_iau: torch.Tensor, prs_sci: torch.Tensor, pni_sci: torch.Tensor, pri_rci: torch.Tensor, pnr_rci: torch.Tensor, png_rci: torch.Tensor, pni_rci: torch.Tensor, prr_rci: torch.Tensor, prg_rci: torch.Tensor, pbg_rci: torch.Tensor, prs_sde: torch.Tensor, prg_gde: torch.Tensor, png_gde: torch.Tensor, tpi_ide_table: torch.Tensor, tps_iaus_table: torch.Tensor, tni_iaus_table: torch.Tensor, graupel_av_g_table: torch.Tensor, graupel_cgg10_table: torch.Tensor, graupel_cge10_table: torch.Tensor, odt: float, _const_segment_e_T0=constants.T0, _const_segment_j_NII2=constants.NII2, _const_segment_j_NTB_I=constants.NTB_I, _const_segment_j_NII3=constants.NII3, _const_segment_j_NTB_I1=constants.NTB_I1, _const_STATIC_D0I=_STATIC_D0I, _const_segment_e_BM_I=constants.BM_I, _const_segment_e_MU_I=constants.MU_I, _const_segment_e_AM_I=constants.AM_I, _const_segment_j_C_CUBE=constants.C_CUBE, _const_segment_e_OIG1=constants.OIG1, _const_SCALAR_CIG_4=_SCALAR_CIG[4], _const_segment_e_D0S=constants.D0S, _const_SCALAR_SNOW_WATER_THRESHOLDS_0=_SCALAR_SNOW_WATER_THRESHOLDS[0], _const_segment_j_T1_QS_QI=constants.T1_QS_QI, _const_segment_j_EF_SI=constants.EF_SI, _const_SCALAR_RAIN_WATER_THRESHOLDS_0=_SCALAR_RAIN_WATER_THRESHOLDS[0], _const_segment_e_ORG2=constants.ORG2, _const_segment_j_T1_QR_QI=constants.T1_QR_QI, _const_segment_j_EF_RI=constants.EF_RI, _const_segment_h_FV_R=constants.FV_R, _const_SCALAR_CRE_8=_SCALAR_CRE[8], _const_segment_j_T2_QR_QI=constants.T2_QR_QI, _const_SCALAR_CRE_7=_SCALAR_CRE[7], _const_segment_e_RHO_I=constants.RHO_I, _const_segment_j_C_SQRD=constants.C_SQRD, _const_segment_j_T1_QS_SD=constants.T1_QS_SD, _const_segment_j_T2_QS_SD=constants.T2_QS_SD, _const_segment_g_EPS=constants.EPS, _const_segment_g_NRHG=constants.NRHG, _const_segment_g_OGG2=constants.OGG2, _const_segment_j_SC3=constants.SC3, _const_segment_j_T1_QG_SD=constants.T1_QG_SD) -> tuple[torch.Tensor, ...]:
    cold = active_micro & (temp < float(_const_segment_e_T0))
    rvs = rho * qvsi
    rate_max = (qv - qvsi) * rho * odt * 0.999
    idx_i_raw = torch.where(ri > 0.0, _mass_table_index_no_cache(ri, _const_segment_j_NII2, _const_segment_j_NTB_I), 1)
    idx_i1_raw = torch.where(ni > 0.0, _mass_table_index_no_cache(ni, _const_segment_j_NII3, _const_segment_j_NTB_I1), 1)
    ice = cold & l_qi
    idx_i = torch.where(ice, idx_i_raw.to(torch.int32), 0)
    idx_i1 = torch.where(ice, idx_i1_raw.to(torch.int32), 0)
    xdi = torch.clamp_min(float(_const_segment_e_BM_I + _const_segment_e_MU_I + 1.0) * ilami, float(_const_STATIC_D0I))
    # BM_I is real in the source; preserve its real-power operation.
    xmi = float(_const_segment_e_AM_I) * torch.pow(xdi, torch.full_like(xdi, float(_const_segment_e_BM_I)))
    oxmi = 1.0 / xmi
    tpi = tpi_ide_table[idx_i_raw - 1, idx_i1_raw - 1]
    tps = tps_iaus_table[idx_i_raw - 1, idx_i1_raw - 1]
    tni = tni_iaus_table[idx_i_raw - 1, idx_i1_raw - 1]
    pri_ide_raw = float(_const_segment_j_C_CUBE) * t1_subl * diffu * ssati * rvs * float(_const_segment_e_OIG1 * _const_SCALAR_CIG_4) * ni * ilami
    pri_ide_neg = _max_first(_max_first(-ri * odt, pri_ide_raw), rate_max)
    pni_ide_neg = _max_first(-ni * odt, pri_ide_neg * oxmi)
    pri_ide_pos = _min_first(pri_ide_raw, rate_max)
    neg_ice = ice & (pri_ide_raw < 0.0)
    pos_ice = ice & ~neg_ice
    pri_ide = torch.where(neg_ice, pri_ide_neg, pri_ide)
    pni_ide = torch.where(neg_ice, pni_ide_neg, pni_ide)
    pri_ide = torch.where(pos_ice, tpi * pri_ide_pos, pri_ide)
    prs_ide = torch.where(pos_ice, (1.0 - tpi) * pri_ide_pos, prs_ide)
    large_ice = ice & ((idx_i_raw == _const_segment_j_NTB_I) | (xdi > float(5.0 * _const_segment_e_D0S)))
    small_ice = ice & ~large_ice & (xdi < float(0.1 * _const_segment_e_D0S))
    mid_ice = ice & ~large_ice & ~small_ice
    prs_iau_large = ri * 0.99 * odt
    pni_iau_large = ni * 0.95 * odt
    large_or_mid = large_ice | mid_ice
    prs_iau_candidate = torch.where(large_ice, prs_iau_large, _min_first(prs_iau_large, tps * odt))
    pni_iau_candidate = torch.where(large_ice, pni_iau_large, _min_first(pni_iau_large, tni * odt))
    prs_iau = torch.where(large_or_mid, prs_iau_candidate, prs_iau)
    pni_iau = torch.where(large_or_mid, pni_iau_candidate, pni_iau)
    prs_iau = torch.where(small_ice, 0.0, prs_iau)
    pni_iau = torch.where(small_ice, 0.0, pni_iau)
    snow_collect = ice & (rs >= float(_const_SCALAR_SNOW_WATER_THRESHOLDS_0))
    prs_sci_candidate = float(_const_segment_j_T1_QS_QI) * rhof * float(_const_segment_j_EF_SI) * ri * smoe
    prs_sci = torch.where(snow_collect, prs_sci_candidate, prs_sci)
    pni_sci = torch.where(snow_collect, prs_sci_candidate * oxmi, pni_sci)
    rain_collect = ice & (rr >= float(_const_SCALAR_RAIN_WATER_THRESHOLDS_0)) & (mvd_r > 4.0 * xdi)
    ilamr_safe = torch.where(ilamr > 0.0, ilamr, 1.0)
    lamr = 1.0 / ilamr_safe
    n0_r = nr * float(_const_segment_e_ORG2) * lamr
    # The source multiplies ice mass/number before n0_r and the real power.
    rain_power_8 = (lamr + float(_const_segment_h_FV_R)) ** (-float(_const_SCALAR_CRE_8))
    rain_power_7 = (lamr + float(_const_segment_h_FV_R)) ** (-float(_const_SCALAR_CRE_7))
    rain_factor_8 = rhof * float(_const_segment_j_T1_QR_QI) * float(_const_segment_j_EF_RI)
    rain_factor_7 = rhof * float(_const_segment_j_T2_QR_QI) * float(_const_segment_j_EF_RI)
    pri_rci_candidate = rain_factor_8 * ri * n0_r * rain_power_8
    pnr_rci_candidate = _min_first(nr * odt, rain_factor_8 * ni * n0_r * rain_power_8)
    png_rci_candidate = pnr_rci_candidate * torch.clamp(working_power(10.0, -0.1 * w1d) + 0.1, min=0.1, max=1.0)
    prr_rci_candidate = _min_first(rr * odt, rain_factor_7 * ni * n0_r * rain_power_7)
    prg_rci_candidate = pri_rci_candidate + prr_rci_candidate
    pri_rci = torch.where(rain_collect, pri_rci_candidate, pri_rci)
    pnr_rci = torch.where(rain_collect, pnr_rci_candidate, pnr_rci)
    png_rci = torch.where(rain_collect, png_rci_candidate, png_rci)
    pni_rci = torch.where(rain_collect, pri_rci_candidate * oxmi, pni_rci)
    prr_rci = torch.where(rain_collect, prr_rci_candidate, prr_rci)
    prg_rci = torch.where(rain_collect, prg_rci_candidate, prg_rci)
    pbg_rci = torch.where(rain_collect, prg_rci_candidate / float(_const_segment_e_RHO_I), pbg_rci)
    snow = cold & l_qs
    c_snow = float(_const_segment_j_C_SQRD) + (temp - float(_const_segment_e_T0) + 1.5) * float(_const_segment_j_C_CUBE - _const_segment_j_C_SQRD) / (-30.0 + 1.5)
    c_snow = torch.clamp(c_snow, min=float(_const_segment_j_C_SQRD), max=float(_const_segment_j_C_CUBE))
    prs_sde_raw = c_snow * t1_subl * diffu * ssati * rvs * (float(_const_segment_j_T1_QS_SD) * smo1 + float(_const_segment_j_T2_QS_SD) * rhof2 * vsc2 * smof)
    prs_sde_neg = _max_first(_max_first(-rs * odt, prs_sde_raw), rate_max)
    prs_sde_pos = _min_first(prs_sde_raw, rate_max)
    prs_sde = torch.where(snow & (prs_sde_raw < 0.0), prs_sde_neg, torch.where(snow, prs_sde_pos, prs_sde))
    graupel = cold & l_qg & (ssati < -float(_const_segment_g_EPS))
    graupel_idx = idx.to(torch.int64).clamp(1, _const_segment_g_NRHG)
    ilamg_safe = torch.where(ilamg > 0.0, ilamg, 1.0)
    n0_g = ng * float(_const_segment_g_OGG2) * (1.0 / ilamg_safe)
    table_index = graupel_idx - 1
    cge10 = graupel_cge10_table[table_index]
    t2_qg_sd = float(0.28 * _const_segment_j_SC3) * torch.sqrt(graupel_av_g_table[table_index]) * graupel_cgg10_table[table_index]
    prg_gde_raw = float(_const_segment_j_C_CUBE) * t1_subl * diffu * ssati * rvs * n0_g * (float(_const_segment_j_T1_QG_SD) * _libm_power(ilamg_safe, 2.0) + t2_qg_sd * vsc2 * rhof2 * ilamg_safe ** cge10)
    prg_gde_neg = _max_first(_max_first(-rg * odt, prg_gde_raw), rate_max)
    prg_gde_pos = _min_first(prg_gde_raw, rate_max)
    graupel_neg = graupel & (prg_gde_raw < 0.0)
    prg_gde = torch.where(graupel & ~graupel_neg, prg_gde_pos, prg_gde)
    prg_gde = torch.where(graupel_neg, prg_gde_neg, prg_gde)
    png_gde = torch.where(graupel_neg, prg_gde_neg * ng / rg, png_gde)
    return (pri_ide, pni_ide, prs_ide, prs_iau, pni_iau, prs_sci, pni_sci, pri_rci, pnr_rci, png_rci, pni_rci, prr_rci, prg_rci, pbg_rci, prs_sde, prg_gde, png_gde, idx_i, idx_i1)


def run_k(
    state: Mapping[str, torch.Tensor] | TorchSegmentJResult,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    copy: bool = False,
) -> TorchSegmentKResult:
    j_result = _segment_j_result(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
    )
    input_state = j_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = j_result.workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment K dt must be nonzero")
    odt = 1.0 / resolved_dt
    i_workspace = j_result.workspace.base
    h_workspace = i_workspace.base
    g_workspace = h_workspace.base
    f_workspace = g_workspace.base
    e_workspace = f_workspace.base
    c_base = e_workspace.base.base
    tend = _tensor_mapping(j_result.workspace.tend, copy=copy)
    tendencies = _tensor_mapping(j_result.workspace.tendencies, copy=copy)
    vtboost = torch.ones_like(c_base.vtboost)
    packed_riming_tendencies: torch.Tensor | None = None

    if not turn_off_micro:
        active_micro = g_workspace.do_micro | g_workspace.supersaturated
        riming_result = riming_tendencies(
            temp=c_base.temp,
            rhof=g_workspace.thermo["rhof"],
            visco=g_workspace.thermo["visco"],
            l_qc=f_workspace.l_qc,
            rc=f_workspace.rc,
            nc=f_workspace.nc,
            mvd_c=f_workspace.mvd_c,
            l_qs=e_workspace.l_qs,
            rs=e_workspace.rs,
            smo0=e_workspace.snow_moments["smo0"],
            smob=e_workspace.snow_moments["smob"],
            smoc=e_workspace.snow_moments["smoc"],
            smoe=e_workspace.snow_moments["smoe"],
            vtboost=vtboost,
            l_qg=g_workspace.l_qg,
            rg=g_workspace.rg,
            ng=g_workspace.ng,
            ilamg=g_workspace.ilamg,
            idx=g_workspace.idx_bg,
            tend=tend,
            odt=odt,
            active_micro=active_micro,
            return_pack=N_TENDENCY_LAYOUT_MODE == "producer_packed" and TENSOR_COMPILE_ISLAND_CUDA and not copy,
        )
        if len(riming_result) == 3:
            tend, vtboost, packed_riming_tendencies = riming_result
        else:
            tend, vtboost = riming_result

    workspace = TorchSegmentKWorkspace(
        base=j_result.workspace,
        tend=tend,
        tendencies=tendencies,
        tables=j_result.workspace.tables,
        vtboost=vtboost,
        dt=resolved_dt,
        odt=odt,
        packed_riming_tendencies=packed_riming_tendencies,
    )
    return TorchSegmentKResult(result_state, workspace, j_result.ncol, j_result.nz)


def riming_tendencies(
    *,
    temp: torch.Tensor,
    rhof: torch.Tensor,
    visco: torch.Tensor,
    l_qc: torch.Tensor,
    rc: torch.Tensor,
    nc: torch.Tensor,
    mvd_c: torch.Tensor,
    l_qs: torch.Tensor,
    rs: torch.Tensor,
    smo0: torch.Tensor,
    smob: torch.Tensor,
    smoc: torch.Tensor,
    smoe: torch.Tensor,
    vtboost: torch.Tensor,
    l_qg: torch.Tensor,
    rg: torch.Tensor,
    ng: torch.Tensor,
    ilamg: torch.Tensor,
    idx: torch.Tensor,
    tend: Mapping[str, torch.Tensor],
    odt: float,
    active_micro: torch.Tensor,
    return_pack: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor] | tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    names = _RIMING_TENDENCY_FIELDS
    updated = dict(tend)
    compile_args = (
        temp,
        temp,
        rhof,
        visco,
        rc,
        nc,
        mvd_c,
        l_qc,
        rs,
        smo0,
        smob,
        smoc,
        smoe,
        l_qs,
        rg,
        ng,
        ilamg,
        idx,
        l_qg,
        vtboost,
        active_micro,
        *tuple(updated[name] for name in names),
        updated["prs_sde"],
        _tensor(_STATIC_SNOW_CLOUD_WATER_EFFICIENCY, temp),
        _tensor(_STATIC_SNOW_DIAMETER_BINS, temp),
        _tensor(_STATIC_GRAUPEL_AV_G, temp),
        _tensor(_STATIC_GRAUPEL_BV_G, temp),
        _tensor(_STATIC_GRAUPEL_CGG[5], temp),
        _tensor(_STATIC_GRAUPEL_CGG[8], temp),
        _tensor(_STATIC_GRAUPEL_CGE[8], temp),
        float(odt),
    )
    if return_pack:
        values = _run_tensor_compile_island(
            "riming_tendencies_packed", _riming_tendencies_packed_kernel, *compile_args
        )
    else:
        values = _run_tensor_compile_island(
            "riming_tendencies", _riming_tendencies_kernel, *compile_args
        )
    if return_pack:
        packed, vtboost_out = values
        field_values = packed.unbind(0)
    else:
        field_values = values[: len(names)]
        vtboost_out = values[len(names)]
    for name, value in zip(names, field_values, strict=True):
        updated[name] = value
    if return_pack:
        return updated, vtboost_out, packed
    return updated, vtboost_out


_RIMING_TENDENCY_FIELDS = (
    "prs_scw",
    "pnc_scw",
    "prg_scw",
    "png_scw",
    "pbg_scw",
    "prg_gcw",
    "pnc_gcw",
    "pbg_gcw",
    "pni_ihm",
    "pri_ihm",
    "prs_ihm",
    "prg_ihm",
)


def _riming_tendencies_packed_kernel(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
    values = _riming_tendencies_kernel(*args)
    return torch.stack(values[: len(_RIMING_TENDENCY_FIELDS)], dim=0), values[-1]


def _riming_tendencies_kernel(temp: torch.Tensor, rhof: torch.Tensor, visco: torch.Tensor, rc: torch.Tensor, nc: torch.Tensor, mvd_c: torch.Tensor, l_qc: torch.Tensor, rs: torch.Tensor, smo0: torch.Tensor, smob: torch.Tensor, smoc: torch.Tensor, smoe: torch.Tensor, l_qs: torch.Tensor, rg: torch.Tensor, ng: torch.Tensor, ilamg: torch.Tensor, idx: torch.Tensor, l_qg: torch.Tensor, vtboost: torch.Tensor, active_micro: torch.Tensor, prs_scw: torch.Tensor, pnc_scw: torch.Tensor, prg_scw: torch.Tensor, png_scw: torch.Tensor, pbg_scw: torch.Tensor, prg_gcw: torch.Tensor, pnc_gcw: torch.Tensor, pbg_gcw: torch.Tensor, pni_ihm: torch.Tensor, pri_ihm: torch.Tensor, prs_ihm: torch.Tensor, prg_ihm: torch.Tensor, prs_sde: torch.Tensor, snow_cloud_water_efficiency: torch.Tensor, snow_diameter_bins: torch.Tensor, graupel_av_g_table: torch.Tensor, graupel_bv_g_table: torch.Tensor, graupel_cgg5_table: torch.Tensor, graupel_cgg8_table: torch.Tensor, graupel_cge8_table: torch.Tensor, odt: float, _const_segment_e_T0=constants.T0, _const_segment_f_D0C=constants.D0C, _const_segment_e_D0S=constants.D0S, _const_segment_k_NBS=constants.NBS, _const_segment_k_NBC=constants.NBC, _const_segment_k_T1_QS_QC=constants.T1_QS_QC, _const_segment_k_RIME_THRESHOLD=constants.RIME_THRESHOLD, _const_segment_g_EPS=constants.EPS, _const_segment_k_RIME_CONVERSION=constants.RIME_CONVERSION, _const_segment_k_AV_S=constants.AV_S, _const_segment_k_BV_S=constants.BV_S, _const_segment_k_FV_S=constants.FV_S, _const_segment_g_METERS3_TO_LITERS=constants.METERS3_TO_LITERS, _const_segment_k_RHO_S=constants.RHO_S, _const_SCALAR_GRAUPEL_WATER_THRESHOLDS_0=_SCALAR_GRAUPEL_WATER_THRESHOLDS[0], _const_segment_g_NRHG=constants.NRHG, _const_segment_g_BM_G=constants.BM_G, _const_segment_g_MU_G=constants.MU_G, _const_segment_g_OGG3=constants.OGG3, _const_segment_e_RHO_W=constants.RHO_W, _const_segment_g_D0G=constants.D0G, _const_segment_e_PI=constants.PI, _const_segment_g_OGG2=constants.OGG2, _const_segment_i_XM0I=constants.XM0I) -> tuple[torch.Tensor, ...]:
    tempc = temp - float(_const_segment_e_T0)
    xds = smoc / _clamp_tiny(smob)
    snow_cloud = active_micro & l_qc & l_qs & (mvd_c > float(_const_segment_f_D0C)) & (xds > float(_const_segment_e_D0S))
    snow_bin = (1 + torch.floor(float(_const_segment_k_NBS) * torch.log(_clamp_tiny(xds / snow_diameter_bins[0])) / torch.log(snow_diameter_bins[-1] / snow_diameter_bins[0])).to(torch.int64)).clamp(1, _const_segment_k_NBS)
    cloud_bin = (torch.trunc(mvd_c * 1000000.0).to(torch.int64) - 1).clamp(0, _const_segment_k_NBC - 1)
    ef_sw = snow_cloud_water_efficiency[snow_bin - 1, cloud_bin]
    prs_scw_candidate = _min_first(rc * odt, rhof * float(_const_segment_k_T1_QS_QC) * ef_sw * rc * smoe)
    pnc_scw_candidate = _min_first(nc * odt, rhof * float(_const_segment_k_T1_QS_QC) * ef_sw * nc * smoe)
    pnc_scw = torch.where(snow_cloud, pnc_scw_candidate, pnc_scw)
    prs_scw = torch.where(snow_cloud, prs_scw_candidate, prs_scw)
    conversion = snow_cloud & (temp < float(_const_segment_e_T0)) & (prs_scw > float(_const_segment_k_RIME_THRESHOLD) * prs_sde) & (prs_sde > float(_const_segment_g_EPS))
    # r_frac is read only where conversion holds (prs_sde > EPS); a benign divisor elsewhere keeps the
    # backward of the unselected division finite (forward unchanged where selected).
    r_frac = torch.clamp_max(prs_scw / torch.where(conversion, prs_sde, 1.0), 30.0)
    g_frac = torch.clamp_max(0.15 + (r_frac - 2.0) * 0.028, float(_const_segment_k_RIME_CONVERSION))
    vtboost_candidate = torch.clamp_max(1.1 + (r_frac - 2.0) * 0.014, 1.5)
    prg_scw_candidate = g_frac * prs_scw
    png_scw_candidate = prg_scw_candidate * smo0 / _clamp_tiny(rs)
    xds_rime = torch.where(conversion, xds, 1.0)
    vts = float(_const_segment_k_AV_S) * xds_rime ** float(_const_segment_k_BV_S) * torch.exp(-float(_const_segment_k_FV_S) * xds_rime)
    const_ri = -1.0 * (mvd_c * 500000.0) * vts / torch.clamp_max(tempc, -0.1)
    const_ri = torch.clamp(const_ri, min=0.1, max=10.0)
    rime_dens = (0.051 + 0.114 * const_ri - 0.0055 * const_ri * const_ri) * 1000.0
    dense_enough = rime_dens >= 150.0
    final_g_frac = torch.where(dense_enough, g_frac, 0.0)
    final_prg_scw = torch.where(dense_enough, prg_scw_candidate, 0.0)
    final_png_scw = torch.where(dense_enough, png_scw_candidate, 0.0)
    snow_dens_frac = torch.clamp(rs * odt / _clamp_tiny(rs * odt + final_prg_scw), min=0.0, max=1.0)
    pbg_scw_candidate = float(_const_segment_g_METERS3_TO_LITERS) * final_prg_scw / (float(_const_segment_k_RHO_S) * snow_dens_frac + rime_dens * (1.0 - snow_dens_frac))
    prg_scw = torch.where(conversion, final_prg_scw, prg_scw)
    png_scw = torch.where(conversion, final_png_scw, png_scw)
    pbg_scw = torch.where(conversion, pbg_scw_candidate, pbg_scw)
    prs_scw = torch.where(conversion, (1.0 - final_g_frac) * prs_scw, prs_scw)
    vtboost = torch.where(conversion, vtboost_candidate, vtboost)
    graupel_cloud = active_micro & l_qc & l_qg & (rg >= float(_const_SCALAR_GRAUPEL_WATER_THRESHOLDS_0)) & (mvd_c > float(_const_segment_f_D0C))
    graupel_idx = idx.to(torch.int64).clamp(1, _const_segment_g_NRHG)
    table_index = graupel_idx - 1
    ilamg_safe = torch.where(ilamg > 0.0, ilamg, 1.0)
    xdg = float(_const_segment_g_BM_G + _const_segment_g_MU_G + 1.0) * ilamg_safe
    av_g = graupel_av_g_table[table_index]
    bv_g = graupel_bv_g_table[table_index]
    cgg5 = graupel_cgg5_table[table_index]
    cgg8 = graupel_cgg8_table[table_index]
    cge8 = graupel_cge8_table[table_index]
    vtg = rhof * av_g * cgg5 * float(_const_segment_g_OGG3) * ilamg_safe ** bv_g
    stoke_g = mvd_c * mvd_c * vtg * float(_const_segment_e_RHO_W) / (9.0 * visco * xdg)
    const_ri_g = -1.0 * (mvd_c * 500000.0) * vtg / torch.clamp_max(tempc, -0.1)
    const_ri_g = torch.clamp(const_ri_g, min=0.1, max=10.0)
    rime_dens_g = (0.051 + 0.114 * const_ri_g - 0.0055 * const_ri_g * const_ri_g) * 1000.0
    graupel_large = graupel_cloud & (xdg > float(_const_segment_g_D0G))
    stoke_log = (stoke_g >= 0.4) & (stoke_g <= 10.0)
    ef_gw = torch.where(stoke_log, 0.55 * torch.log10(torch.where(stoke_log, 2.51 * stoke_g, 1.0)), torch.where(stoke_g < 0.4, 0.0, 0.77))
    ef_gw = torch.where(temp > float(_const_segment_e_T0), ef_gw * 0.1, ef_gw)
    t1_qg_qc = float(_const_segment_e_PI * 0.25) * av_g * cgg8
    # Preserve the original reciprocal product before the collection products.
    n0_g = ng * float(_const_segment_g_OGG2) * (1.0 / ilamg_safe)
    ilamg_cge8 = ilamg_safe ** cge8
    prg_gcw_candidate = rhof * t1_qg_qc * ef_gw * rc * n0_g * ilamg_cge8
    pnc_gcw_candidate = _min_first(nc * odt, rhof * t1_qg_qc * ef_gw * nc * n0_g * ilamg_cge8)
    prg_gcw = torch.where(graupel_large, prg_gcw_candidate, prg_gcw)
    pnc_gcw = torch.where(graupel_large, pnc_gcw_candidate, pnc_gcw)
    cold_graupel = graupel_large & (temp < float(_const_segment_e_T0))
    pbg_gcw = torch.where(cold_graupel, float(_const_segment_g_METERS3_TO_LITERS) * prg_gcw_candidate / rime_dens_g, pbg_gcw)
    hallett = cold_graupel & (prg_gcw_candidate > float(_const_segment_g_EPS)) & (tempc > -8.0)
    tf = torch.clamp_min(torch.minimum(0.5 * (-3.0 - tempc), 0.33333333 * (8.0 + tempc)), 0.0)
    pni_ihm_candidate = 350000000.0 * tf * prg_gcw_candidate
    pri_ihm_candidate = float(_const_segment_i_XM0I) * pni_ihm_candidate
    ihm_denom = torch.where(hallett, prs_scw + prg_gcw_candidate, 1.0)  # >= EPS where selected
    pni_ihm = torch.where(hallett, pni_ihm_candidate, pni_ihm)
    pri_ihm = torch.where(hallett, pri_ihm_candidate, pri_ihm)
    prs_ihm = torch.where(hallett, prs_scw / ihm_denom * pri_ihm_candidate, prs_ihm)
    prg_ihm = torch.where(hallett, prg_gcw_candidate / ihm_denom * pri_ihm_candidate, prg_ihm)
    return (prs_scw, pnc_scw, prg_scw, png_scw, pbg_scw, prg_gcw, pnc_gcw, pbg_gcw, pni_ihm, pri_ihm, prs_ihm, prg_ihm, vtboost)


def run_l(
    state: Mapping[str, torch.Tensor] | TorchSegmentKResult,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    copy: bool = False,
) -> TorchSegmentLResult:
    k_result = _segment_k_result(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
    )
    input_state = k_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = k_result.workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment L dt must be nonzero")
    odt = 1.0 / resolved_dt
    j_workspace = k_result.workspace.base
    i_workspace = j_workspace.base
    h_workspace = i_workspace.base
    g_workspace = h_workspace.base
    f_workspace = g_workspace.base
    e_workspace = f_workspace.base
    c_base = e_workspace.base.base
    tend = _tensor_mapping(k_result.workspace.tend, copy=copy)
    tendencies = _tensor_mapping(k_result.workspace.tendencies, copy=copy)
    packed_melting_tendencies: torch.Tensor | None = None

    if not turn_off_micro:
        active_micro = g_workspace.do_micro | g_workspace.supersaturated
        melting_result = melting_tendencies(
            rhof2=g_workspace.thermo["rhof2"],
            rho=c_base.rho,
            temp=c_base.temp,
            qvsi=g_workspace.thermo["qvsi"],
            tcond=g_workspace.thermo["tcond"],
            diffu=g_workspace.thermo["diffu"],
            vsc2=g_workspace.thermo["vsc2"],
            ssati=g_workspace.thermo["ssati"],
            delqvs=g_workspace.thermo["delqvs"],
            l_qs=e_workspace.l_qs,
            rs=e_workspace.rs,
            smof=e_workspace.snow_moments["smof"],
            smo0=e_workspace.snow_moments["smo0"],
            smo1=e_workspace.snow_moments["smo1"],
            l_qg=g_workspace.l_qg,
            rg=g_workspace.rg,
            ng=g_workspace.ng,
            ilamg=g_workspace.ilamg,
            idx=g_workspace.idx_bg,
            tend=tend,
            dt=resolved_dt,
            odt=odt,
            active_micro=active_micro,
            return_pack=N_TENDENCY_LAYOUT_MODE == "producer_packed" and TENSOR_COMPILE_ISLAND_CUDA and not copy,
        )
        if isinstance(melting_result, tuple):
            tend, packed_melting_tendencies = melting_result
        else:
            tend = melting_result

    workspace = TorchSegmentLWorkspace(
        base=k_result.workspace,
        tend=tend,
        tendencies=tendencies,
        tables=k_result.workspace.tables,
        dt=resolved_dt,
        odt=odt,
        packed_melting_tendencies=packed_melting_tendencies,
    )
    return TorchSegmentLResult(result_state, workspace, k_result.ncol, k_result.nz)


def melting_tendencies(
    *,
    rhof2: torch.Tensor,
    rho: torch.Tensor,
    temp: torch.Tensor,
    qvsi: torch.Tensor,
    tcond: torch.Tensor,
    diffu: torch.Tensor,
    vsc2: torch.Tensor,
    ssati: torch.Tensor,
    delqvs: torch.Tensor,
    l_qs: torch.Tensor,
    rs: torch.Tensor,
    smof: torch.Tensor,
    smo0: torch.Tensor,
    smo1: torch.Tensor,
    l_qg: torch.Tensor,
    rg: torch.Tensor,
    ng: torch.Tensor,
    ilamg: torch.Tensor,
    idx: torch.Tensor,
    tend: Mapping[str, torch.Tensor],
    dt: float,
    odt: float,
    active_micro: torch.Tensor,
    return_pack: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], torch.Tensor]:
    names = _MELTING_TENDENCY_FIELDS
    updated = dict(tend)
    compile_args = (
        temp,
        rhof2,
        rho,
        temp,
        qvsi,
        tcond,
        diffu,
        vsc2,
        ssati,
        delqvs,
        l_qs,
        rs,
        smof,
        smo0,
        smo1,
        l_qg,
        rg,
        ng,
        ilamg,
        idx,
        active_micro,
        *tuple(updated[name] for name in names),
        updated["prr_rcs"],
        _tensor(_STATIC_GRAUPEL_AV_G, temp),
        _tensor(_STATIC_GRAUPEL_CGG[10], temp),
        _tensor(_STATIC_GRAUPEL_CGE[10], temp),
        _tensor(_STATIC_GRAUPEL_RHO_G, temp),
        float(dt),
        float(odt),
        bool(dt > float(constants.TIMESTEP_CONVERSION_RIME_TO_RAIN)),
    )
    if return_pack:
        values = _run_tensor_compile_island(
            "melting_tendencies_packed", _melting_tendencies_packed_kernel, *compile_args
        )
    else:
        values = _run_tensor_compile_island(
            "melting_tendencies", _melting_tendencies_kernel, *compile_args
        )
    packed = values if return_pack else None
    field_values = packed.unbind(0) if packed is not None else values
    for name, value in zip(names, field_values, strict=True):
        updated[name] = value
    if packed is not None:
        return updated, packed
    return updated


_MELTING_TENDENCY_FIELDS = (
    "prr_sml",
    "pnr_sml",
    "prs_sde",
    "prr_gml",
    "pnr_gml",
    "pbg_gml",
    "prg_gde",
    "png_gde",
    "prr_rcw",
    "prs_scw",
    "prg_gcw",
)


def _melting_tendencies_packed_kernel(*args: Any) -> torch.Tensor:
    return torch.stack(_melting_tendencies_kernel(*args), dim=0)


def _melting_tendencies_kernel(rhof2: torch.Tensor, rho: torch.Tensor, temp: torch.Tensor, qvsi: torch.Tensor, tcond: torch.Tensor, diffu: torch.Tensor, vsc2: torch.Tensor, ssati: torch.Tensor, delqvs: torch.Tensor, l_qs: torch.Tensor, rs: torch.Tensor, smof: torch.Tensor, smo0: torch.Tensor, smo1: torch.Tensor, l_qg: torch.Tensor, rg: torch.Tensor, ng: torch.Tensor, ilamg: torch.Tensor, idx: torch.Tensor, active_micro: torch.Tensor, prr_sml: torch.Tensor, pnr_sml: torch.Tensor, prs_sde: torch.Tensor, prr_gml: torch.Tensor, pnr_gml: torch.Tensor, pbg_gml: torch.Tensor, prg_gde: torch.Tensor, png_gde: torch.Tensor, prr_rcw: torch.Tensor, prs_scw: torch.Tensor, prg_gcw: torch.Tensor, prr_rcs: torch.Tensor, graupel_av_g_table: torch.Tensor, graupel_cgg10_table: torch.Tensor, graupel_cge10_table: torch.Tensor, graupel_rho_g_table: torch.Tensor, dt: float, odt: float, convert_rime_to_rain: bool, _const_segment_e_T0=constants.T0, _const_segment_l_LVAP0=constants.LVAP0, _const_segment_l_T1_QS_ME=constants.T1_QS_ME, _const_segment_l_T2_QS_ME=constants.T2_QS_ME, _const_segment_l_OLFUS=constants.OLFUS, _const_segment_j_C_CUBE=constants.C_CUBE, _const_segment_j_T1_QS_SD=constants.T1_QS_SD, _const_segment_j_T2_QS_SD=constants.T2_QS_SD, _const_segment_g_NRHG=constants.NRHG, _const_segment_g_OGG2=constants.OGG2, _const_segment_e_PI=constants.PI, _const_segment_j_SC3=constants.SC3, _const_segment_l_T1_QG_ME=constants.T1_QG_ME, _const_segment_e_RHO_W=constants.RHO_W, _const_segment_g_METERS3_TO_LITERS=constants.METERS3_TO_LITERS, _const_segment_j_T1_QG_SD=constants.T1_QG_SD) -> tuple[torch.Tensor, ...]:
    warm = active_micro & (temp > float(_const_segment_e_T0))
    tempc = temp - float(_const_segment_e_T0)
    rvs = rho * qvsi
    t1_subl = _sublimation_factor_no_cache(rho, temp, qvsi, tcond, diffu, ssati)
    snow = warm & l_qs
    prr_sml_raw = (tempc * tcond - float(_const_segment_l_LVAP0) * diffu * delqvs) * (float(_const_segment_l_T1_QS_ME) * smo1 + float(_const_segment_l_T2_QS_ME) * rhof2 * vsc2 * smof)
    snow_melt = snow & (prr_sml_raw > 0.0)
    prr_sml_candidate = prr_sml_raw + float(4218.0 * _const_segment_l_OLFUS) * tempc * (prr_rcs + prs_scw)
    prr_sml_candidate = _min_first(rs * odt, torch.clamp_min(prr_sml_candidate, 0.0))
    pnr_sml_candidate = _min_first(smo0 * odt, smo0 / _clamp_tiny(rs) * prr_sml_candidate * working_power(10.0, -0.25 * tempc))
    prr_sml = torch.where(snow, 0.0, prr_sml)
    pnr_sml = torch.where(snow, 0.0, pnr_sml)
    prr_sml = torch.where(snow_melt, prr_sml_candidate, prr_sml)
    pnr_sml = torch.where(snow_melt, pnr_sml_candidate, pnr_sml)
    snow_subl = snow & ~snow_melt & (ssati < 0.0)
    prs_sde_candidate = float(_const_segment_j_C_CUBE) * t1_subl * diffu * ssati * rvs * (float(_const_segment_j_T1_QS_SD) * smo1 + float(_const_segment_j_T2_QS_SD) * rhof2 * vsc2 * smof)
    prs_sde = torch.where(snow_subl, _max_first(-rs * odt, prs_sde_candidate), prs_sde)
    graupel = warm & l_qg
    graupel_idx = idx.to(torch.int64).clamp(1, _const_segment_g_NRHG)
    table_index = graupel_idx - 1
    ilamg_safe = torch.where(ilamg > 0.0, ilamg, 1.0)
    n0_g = ng * float(_const_segment_g_OGG2) * (1.0 / ilamg_safe)
    n0_melt_default = n0_g
    lamg = 1.0 / ilamg_safe
    n0_melt_limited = torch.div(0.0001, _clamp_tiny(rg)) * float(_const_segment_g_OGG2) * lamg
    n0_melt = torch.where(rg * ng < 0.0001, n0_melt_limited, n0_melt_default)
    cge10 = graupel_cge10_table[table_index]
    rho_g = graupel_rho_g_table[table_index]
    t2_qg_me = float(_const_segment_e_PI * 4.0 * _const_segment_j_C_CUBE * _const_segment_l_OLFUS * 0.28 * _const_segment_j_SC3) * torch.sqrt(graupel_av_g_table[table_index]) * graupel_cgg10_table[table_index]
    prr_gml_raw = (tempc * tcond - float(_const_segment_l_LVAP0) * diffu * delqvs) * n0_melt * (float(_const_segment_l_T1_QG_ME) * _libm_power(ilamg_safe, 2.0) + t2_qg_me * rhof2 * vsc2 * ilamg_safe ** cge10)
    prr_gml_candidate = _min_first(rg * odt, torch.clamp_min(prr_gml_raw, 0.0))
    graupel_melt = graupel & (prr_gml_candidate > 0.0)
    melt_f = torch.clamp(prr_gml_candidate * dt / _clamp_tiny(rg), min=0.05, max=1.0)
    melt_density = torch.clamp(melt_f * rho_g, min=50.0, max=float(_const_segment_e_RHO_W))
    pbg_gml_candidate = float(_const_segment_g_METERS3_TO_LITERS) * prr_gml_candidate / melt_density
    pnr_gml_candidate = prr_gml_candidate * ng / _clamp_tiny(rg) * torch.pow(10.0, -0.33 * tempc)
    prr_gml = torch.where(graupel, 0.0, prr_gml)
    pnr_gml = torch.where(graupel, 0.0, pnr_gml)
    pbg_gml = torch.where(graupel, 0.0, pbg_gml)
    prr_gml = torch.where(graupel_melt, prr_gml_candidate, prr_gml)
    pnr_gml = torch.where(graupel_melt, pnr_gml_candidate, pnr_gml)
    pbg_gml = torch.where(graupel_melt, pbg_gml_candidate, pbg_gml)
    graupel_subl = graupel & ~graupel_melt & (ssati < 0.0)
    t2_qg_sd = float(0.28 * _const_segment_j_SC3) * torch.sqrt(graupel_av_g_table[table_index]) * graupel_cgg10_table[table_index]
    prg_gde_raw = float(_const_segment_j_C_CUBE) * t1_subl * diffu * ssati * rvs * n0_g * (float(_const_segment_j_T1_QG_SD) * _libm_power(ilamg_safe, 2.0) + t2_qg_sd * vsc2 * rhof2 * ilamg_safe ** cge10)
    prg_gde_candidate = _max_first(-rg * odt, prg_gde_raw)
    prg_gde = torch.where(graupel_subl, prg_gde_candidate, prg_gde)
    png_gde = torch.where(graupel_subl, prg_gde_candidate * ng / _clamp_tiny(rg), png_gde)
    if convert_rime_to_rain:
        prr_rcw = torch.where(warm, prr_rcw + prs_scw + prg_gcw, prr_rcw)
        prs_scw = torch.where(warm, 0.0, prs_scw)
        prg_gcw = torch.where(warm, 0.0, prg_gcw)
    return (prr_sml, pnr_sml, prs_sde, prr_gml, pnr_gml, pbg_gml, prg_gde, png_gde, prr_rcw, prs_scw, prg_gcw)


def run_m(
    state: Mapping[str, torch.Tensor] | TorchSegmentLResult,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    copy: bool = False,
) -> TorchSegmentMResult:
    l_result = _segment_l_result(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
    )
    input_state = l_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = l_result.workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment M dt must be nonzero")
    odt = 1.0 / resolved_dt
    k_workspace = l_result.workspace.base
    j_workspace = k_workspace.base
    i_workspace = j_workspace.base
    h_workspace = i_workspace.base
    g_workspace = h_workspace.base
    f_workspace = g_workspace.base
    e_workspace = f_workspace.base
    d_workspace = e_workspace.base
    c_base = d_workspace.base
    tend = _tensor_mapping(l_result.workspace.tend, copy=copy)
    tendencies = _tensor_mapping(l_result.workspace.tendencies, copy=copy)

    if not turn_off_micro and (d_workspace.nwfa_present or d_workspace.nifa_present):
        tend = aerosol_scavenging_tendencies(
            temp=c_base.temp,
            rho=c_base.rho,
            rhof=g_workspace.thermo["rhof"],
            visco=g_workspace.thermo["visco"],
            nwfa=d_workspace.nwfa,
            nifa=d_workspace.nifa,
            l_qr=e_workspace.l_qr,
            nr=e_workspace.nr,
            ilamr=e_workspace.ilamr,
            mvd_r=e_workspace.mvd_r,
            l_qs=e_workspace.l_qs,
            rs=e_workspace.rs,
            smob=e_workspace.snow_moments["smob"],
            smoc=e_workspace.snow_moments["smoc"],
            smoe=e_workspace.snow_moments["smoe"],
            l_qg=g_workspace.l_qg,
            rg=g_workspace.rg,
            ng=g_workspace.ng,
            ilamg=g_workspace.ilamg,
            idx=g_workspace.idx_bg,
            tend=tend,
            odt=odt,
        )

    workspace = TorchSegmentMWorkspace(
        base=l_result.workspace,
        tend=tend,
        tendencies=tendencies,
        tables=l_result.workspace.tables,
        dt=resolved_dt,
        odt=odt,
    )
    return TorchSegmentMResult(result_state, workspace, l_result.ncol, l_result.nz)


def aerosol_scavenging_tendencies(
    *,
    temp: torch.Tensor,
    rho: torch.Tensor,
    rhof: torch.Tensor,
    visco: torch.Tensor,
    nwfa: torch.Tensor,
    nifa: torch.Tensor,
    l_qr: torch.Tensor,
    nr: torch.Tensor,
    ilamr: torch.Tensor,
    mvd_r: torch.Tensor,
    l_qs: torch.Tensor,
    rs: torch.Tensor,
    smob: torch.Tensor,
    smoc: torch.Tensor,
    smoe: torch.Tensor,
    l_qg: torch.Tensor,
    rg: torch.Tensor,
    ng: torch.Tensor,
    ilamg: torch.Tensor,
    idx: torch.Tensor,
    tend: Mapping[str, torch.Tensor],
    odt: float,
) -> dict[str, torch.Tensor]:
    names = _AEROSOL_SCAVENGING_TENDENCY_FIELDS
    updated = dict(tend)
    values = _run_tensor_compile_island(
        "aerosol_scavenging_tendencies",
        _aerosol_scavenging_tendencies_kernel,
        temp,
        temp,
        rho,
        rhof,
        visco,
        nwfa,
        nifa,
        l_qr,
        nr,
        ilamr,
        mvd_r,
        l_qs,
        rs,
        smob,
        smoc,
        smoe,
        l_qg,
        rg,
        ng,
        ilamg,
        idx,
        *tuple(updated[name] for name in names),
        _scalar(float(constants.WF_AEROSOL_SIZE), temp),
        _scalar(float(constants.IF_AEROSOL_SIZE), temp),
        _tensor(_STATIC_GRAUPEL_AV_G, temp),
        _tensor(_STATIC_GRAUPEL_CGG[8], temp),
        _tensor(_STATIC_GRAUPEL_CGE[8], temp),
        float(odt),
    )
    for name, value in zip(names, values, strict=True):
        updated[name] = value
    return updated


_AEROSOL_SCAVENGING_TENDENCY_FIELDS = (
    "pna_rca",
    "pnd_rcd",
    "pna_sca",
    "pnd_scd",
    "pna_gca",
    "pnd_gcd",
)


def _aerosol_scavenging_tendencies_kernel(temp: torch.Tensor, rho: torch.Tensor, rhof: torch.Tensor, visco: torch.Tensor, nwfa: torch.Tensor, nifa: torch.Tensor, l_qr: torch.Tensor, nr: torch.Tensor, ilamr: torch.Tensor, mvd_r: torch.Tensor, l_qs: torch.Tensor, rs: torch.Tensor, smob: torch.Tensor, smoc: torch.Tensor, smoe: torch.Tensor, l_qg: torch.Tensor, rg: torch.Tensor, ng: torch.Tensor, ilamg: torch.Tensor, idx: torch.Tensor, pna_rca: torch.Tensor, pnd_rcd: torch.Tensor, pna_sca: torch.Tensor, pnd_scd: torch.Tensor, pna_gca: torch.Tensor, pnd_gcd: torch.Tensor, wf_aerosol_size: torch.Tensor, if_aerosol_size: torch.Tensor, graupel_av_g_table: torch.Tensor, graupel_cgg8_table: torch.Tensor, graupel_cge8_table: torch.Tensor, odt: float, _const_segment_e_D0R=constants.D0R, _const_segment_e_ORG2=constants.ORG2, _const_segment_h_T1_QR_QC=constants.T1_QR_QC, _const_segment_h_FV_R=constants.FV_R, _const_SCALAR_CRE_8=_SCALAR_CRE[8], _const_SCALAR_SNOW_WATER_THRESHOLDS_0=_SCALAR_SNOW_WATER_THRESHOLDS[0], _const_segment_k_T1_QS_QC=constants.T1_QS_QC, _const_SCALAR_GRAUPEL_WATER_THRESHOLDS_0=_SCALAR_GRAUPEL_WATER_THRESHOLDS[0], _const_segment_g_NRHG=constants.NRHG, _const_segment_g_BM_G=constants.BM_G, _const_segment_g_MU_G=constants.MU_G, _const_segment_e_PI=constants.PI, _const_segment_g_OGG2=constants.OGG2) -> tuple[torch.Tensor, ...]:
    rain = l_qr & (mvd_r > float(_const_segment_e_D0R))
    ilamr_safe = torch.where(ilamr > 0.0, ilamr, 1.0)
    lamr = 1.0 / ilamr_safe
    n0_r = nr * float(_const_segment_e_ORG2) * lamr
    rain_factor = rhof * float(_const_segment_h_T1_QR_QC) * n0_r * (lamr + float(_const_segment_h_FV_R)) ** (-float(_const_SCALAR_CRE_8))
    ef_ra_w = _aerosol_collection_efficiency_rain(mvd_r, wf_aerosol_size, visco, rho, temp)
    ef_ra_i = _aerosol_collection_efficiency_rain(mvd_r, if_aerosol_size, visco, rho, temp)
    pna_rca = torch.where(rain, _min_first(nwfa * odt, rain_factor * ef_ra_w * nwfa), pna_rca)
    pnd_rcd = torch.where(rain, _min_first(nifa * odt, rain_factor * ef_ra_i * nifa), pnd_rcd)
    snow = l_qs & (rs > float(_const_SCALAR_SNOW_WATER_THRESHOLDS_0))
    xds = smoc / _clamp_tiny(smob)
    ef_sa_w = _aerosol_collection_efficiency_snow(xds, wf_aerosol_size, visco, rho, temp)
    ef_sa_i = _aerosol_collection_efficiency_snow(xds, if_aerosol_size, visco, rho, temp)
    snow_factor = rhof * float(_const_segment_k_T1_QS_QC) * smoe
    pna_sca = torch.where(snow, _min_first(nwfa * odt, snow_factor * ef_sa_w * nwfa), pna_sca)
    pnd_scd = torch.where(snow, _min_first(nifa * odt, snow_factor * ef_sa_i * nifa), pnd_scd)
    graupel = l_qg & (rg > float(_const_SCALAR_GRAUPEL_WATER_THRESHOLDS_0))
    graupel_idx = idx.to(torch.int64).clamp(1, _const_segment_g_NRHG)
    table_index = graupel_idx - 1
    ilamg_safe = torch.where(ilamg > 0.0, ilamg, 1.0)
    xdg = float(_const_segment_g_BM_G + _const_segment_g_MU_G + 1.0) * ilamg_safe
    ef_ga_w = _aerosol_collection_efficiency_graupel(xdg, wf_aerosol_size, visco, rho, temp)
    ef_ga_i = _aerosol_collection_efficiency_graupel(xdg, if_aerosol_size, visco, rho, temp)
    av_g = graupel_av_g_table[table_index]
    cgg8 = graupel_cgg8_table[table_index]
    cge8 = graupel_cge8_table[table_index]
    t1_qg_qc = float(_const_segment_e_PI * 0.25) * av_g * cgg8
    n0_g = ng * float(_const_segment_g_OGG2) / ilamg_safe
    graupel_factor = rhof * t1_qg_qc * n0_g * ilamg_safe ** cge8
    pna_gca = torch.where(graupel, _min_first(nwfa * odt, graupel_factor * ef_ga_w * nwfa), pna_gca)
    pnd_gcd = torch.where(graupel, _min_first(nifa * odt, graupel_factor * ef_ga_i * nifa), pnd_gcd)
    return (pna_rca, pnd_rcd, pna_sca, pnd_scd, pna_gca, pnd_gcd)


def _aerosol_collection_efficiency_common(d: torch.Tensor, da: torch.Tensor, visc: torch.Tensor, rhoa: torch.Tensor, temp: torch.Tensor, vt: torch.Tensor, rho_p: float, _const_segment_m_MEANPATH=constants.MEANPATH, _const_segment_m_BOLTZMAN=constants.BOLTZMAN, _const_segment_e_PI=constants.PI) -> torch.Tensor:
    cc = 1.0 + float(2.0 * _const_segment_m_MEANPATH) / da * (1.257 + 0.4 * torch.exp(float(-0.55 / _const_segment_m_MEANPATH) * da))
    diff = float(_const_segment_m_BOLTZMAN) * temp * cc / (float(3.0 * _const_segment_e_PI) * visc * da)
    re = 0.5 * rhoa * d * vt / visc
    sc = visc / (rhoa * diff)
    st = (float(rho_p) - rhoa) * da * da * vt * cc / (9.0 * visc * d)
    re_safe = _clamp_tiny(re)
    sc_safe = _clamp_tiny(sc)
    aval = torch.log1p(re_safe)
    st2 = (1.2 + 1.0 / 12.0 * aval) / (1.0 + aval)
    eff = 4.0 / (re_safe * sc_safe) * (1.0 + 0.4 * torch.sqrt(re_safe) * sc_safe ** 0.3333 + 0.16 * torch.sqrt(re_safe) * torch.sqrt(sc_safe)) + 4.0 * da / d * (0.02 + da / d * (1.0 + 2.0 * torch.sqrt(re_safe)))
    inertial_base = torch.clamp_min((st - st2) / (st - st2 + 0.666667), 0.0)
    inertial = inertial_base ** 1.5
    eff = torch.where(st > st2, eff + inertial, eff)
    return torch.clamp(eff, min=1e-05, max=1.0)


def _aerosol_collection_efficiency_rain(d: torch.Tensor, da: torch.Tensor, visc: torch.Tensor, rhoa: torch.Tensor, temp: torch.Tensor, _const_segment_e_RHO_W=constants.RHO_W) -> torch.Tensor:
    d = torch.where(d > 0.0, d, torch.ones_like(d))
    vt = -0.1021 + 4932.0 * d - 955100.0 * d * d + 79340000.0 * d * d * d - 2362000000.0 * d * d * d * d
    return _aerosol_collection_efficiency_common(d, da, visc, rhoa, temp, vt, float(_const_segment_e_RHO_W))


def _aerosol_collection_efficiency_snow(d: torch.Tensor, da: torch.Tensor, visc: torch.Tensor, rhoa: torch.Tensor, temp: torch.Tensor, _const_segment_k_AV_S=constants.AV_S, _const_segment_k_BV_S=constants.BV_S, _const_segment_k_RHO_S=constants.RHO_S) -> torch.Tensor:
    d = torch.where(d > 0.0, d, torch.ones_like(d))
    vt = float(_const_segment_k_AV_S) * d ** float(_const_segment_k_BV_S)
    return _aerosol_collection_efficiency_common(d, da, visc, rhoa, temp, vt, float(_const_segment_k_RHO_S))


def _aerosol_collection_efficiency_graupel(d: torch.Tensor, da: torch.Tensor, visc: torch.Tensor, rhoa: torch.Tensor, temp: torch.Tensor, _const_SCALAR_GRAUPEL_AV_G_segment_g_IDX_BG1_1=_SCALAR_GRAUPEL_AV_G[constants.IDX_BG1 - 1], _const_SCALAR_GRAUPEL_BV_G_segment_g_IDX_BG1_1=_SCALAR_GRAUPEL_BV_G[constants.IDX_BG1 - 1], _const_SCALAR_GRAUPEL_RHO_G_segment_g_IDX_BG1_1=_SCALAR_GRAUPEL_RHO_G[constants.IDX_BG1 - 1]) -> torch.Tensor:
    d = torch.where(d > 0.0, d, torch.ones_like(d))
    vt = float(_const_SCALAR_GRAUPEL_AV_G_segment_g_IDX_BG1_1) * d ** float(_const_SCALAR_GRAUPEL_BV_G_segment_g_IDX_BG1_1)
    return _aerosol_collection_efficiency_common(d, da, visc, rhoa, temp, vt, float(_const_SCALAR_GRAUPEL_RHO_G_segment_g_IDX_BG1_1))


def aerosol_collection_efficiency_tensor(
    d: torch.Tensor,
    da: torch.Tensor,
    visc: torch.Tensor,
    rhoa: torch.Tensor,
    temp: torch.Tensor,
    species: str,
) -> torch.Tensor:
    if species == "r":
        vt = (
            -_scalar(0.1021, d)
            + _scalar(4.932e3, d) * d
            - _scalar(0.9551e6, d) * d * d
            + _scalar(0.07934e9, d) * d * d * d
            - _scalar(0.002362e12, d) * d * d * d * d
        )
        rho_p = _scalar(constants.RHO_W, d)
    elif species == "s":
        vt = _scalar(constants.AV_S, d) * d ** _scalar(constants.BV_S, d)
        rho_p = _scalar(constants.RHO_S, d)
    elif species == "g":
        vt = _scalar(_SCALAR_GRAUPEL_AV_G[constants.IDX_BG1 - 1], d) * d ** _scalar(_SCALAR_GRAUPEL_BV_G[constants.IDX_BG1 - 1], d)
        rho_p = _scalar(_SCALAR_GRAUPEL_RHO_G[constants.IDX_BG1 - 1], d)
    else:
        raise ValueError(f"unknown aerosol collection species {species!r}")

    cc = 1.0 + _scalar(2.0 * constants.MEANPATH, da) / da * (
        _scalar(1.257, da) + _scalar(0.4, da) * torch.exp(-_scalar(0.55 / constants.MEANPATH, da) * da)
    )
    diff = _scalar(constants.BOLTZMAN, temp) * temp * cc / (_scalar(3.0 * constants.PI, temp) * visc * da)
    re = _scalar(0.5, d) * rhoa * d * vt / visc
    sc = visc / (rhoa * diff)
    st = (rho_p - rhoa) * da * da * vt * cc / (_scalar(9.0, d) * visc * d)
    re_safe = _clamp_tiny(re)
    sc_safe = _clamp_tiny(sc)
    aval = torch.log1p(re_safe)
    st2 = (_scalar(1.2, d) + _scalar(1.0 / 12.0, d) * aval) / (1.0 + aval)
    eff = (
        _scalar(4.0, d)
        / (re_safe * sc_safe)
        * (
            1.0
            + _scalar(0.4, d) * torch.sqrt(re_safe) * sc_safe ** _scalar(0.3333, d)
            + _scalar(0.16, d) * torch.sqrt(re_safe) * torch.sqrt(sc_safe)
        )
        + _scalar(4.0, d)
        * da
        / d
        * (_scalar(0.02, d) + da / d * (1.0 + _scalar(2.0, d) * torch.sqrt(re_safe)))
    )
    inertial_base = torch.clamp_min((st - st2) / (st - st2 + _scalar(0.666667, st)), _scalar(0.0, st))
    inertial = inertial_base ** _scalar(1.5, st)
    eff = torch.where(st > st2, eff + inertial, eff)
    return torch.clamp(eff, min=1.0e-5, max=1.0)


def run_n(
    state: Mapping[str, torch.Tensor] | TorchSegmentMResult,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    production_sedimentation_moments: bool = False,
    copy: bool = False,
) -> TorchSegmentNResult:
    m_result = _segment_m_result(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
    )
    input_state = m_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = m_result.workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment N dt must be nonzero")
    odt = 1.0 / resolved_dt
    l_workspace = m_result.workspace.base
    k_workspace = l_workspace.base
    j_workspace = k_workspace.base
    i_workspace = j_workspace.base
    h_workspace = i_workspace.base
    g_workspace = h_workspace.base
    f_workspace = g_workspace.base
    e_workspace = f_workspace.base
    d_workspace = e_workspace.base
    c_base = d_workspace.base
    tend = _tensor_mapping(m_result.workspace.tend, copy=copy)
    tendencies = _tensor_mapping(m_result.workspace.tendencies, copy=copy)

    tend, tendencies = limit_and_sum_tendencies_tensor(
        rho=c_base.rho,
        temp=c_base.temp,
        idx=g_workspace.idx_bg,
        lvap=g_workspace.thermo["lvap"],
        ocp=g_workspace.thermo["ocp"],
        qvsi=g_workspace.thermo["qvsi"],
        qv=c_base.qv,
        l_qc=f_workspace.l_qc,
        rc=f_workspace.rc,
        l_qi=e_workspace.l_qi,
        ri=e_workspace.ri,
        l_qr=e_workspace.l_qr,
        rr=e_workspace.rr,
        l_qs=e_workspace.l_qs,
        rs=e_workspace.rs,
        l_qg=g_workspace.l_qg,
        rg=g_workspace.rg,
        tend=tend,
        tendencies=tendencies,
        odt=odt,
        producer_packs=(
            j_workspace.packed_ice_tendencies,
            k_workspace.packed_riming_tendencies,
            l_workspace.packed_melting_tendencies,
        ),
    )

    temp = input_state["t"] + tendencies["tten"] * resolved_dt
    qv = torch.clamp_min(input_state["qv"] + tendencies["qvten"] * resolved_dt, _scalar(constants.MIN_QV, temp))
    rho = _scalar(constants.ROVERRV, temp) * c_base.pres / (_scalar(constants.RDRY, temp) * temp * (qv + _scalar(constants.ROVERRV, temp)))

    nwfaten = tendencies["nwfaten"]
    nifaten = tendencies["nifaten"]
    nwfaten = nwfaten - (tend["pna_rca"] + tend["pna_sca"] + tend["pna_gca"] + tend["pni_iha"]) / rho
    nifaten = nifaten - (tend["pnd_rcd"] + tend["pnd_scd"] + tend["pnd_gcd"]) / rho
    tendencies["nwfaten"] = nwfaten
    tendencies["nifaten"] = nifaten
    nwfa, nifa = aerosol_check_and_update(
        rho=rho,
        nwfa=d_workspace.nwfa,
        nifa=d_workspace.nifa,
        nwfaten=nwfaten,
        nifaten=nifaten,
        dt=resolved_dt,
        nwfa1d=d_workspace.nwfa1d,
        nifa1d=d_workspace.nifa1d,
    )

    cloud, graupel = cloud_graupel_check_and_update(
        rho=rho,
        qc1d=input_state["qc"],
        nc=f_workspace.nc,
        qcten=tendencies["qcten"],
        ncten=tendencies["ncten"],
        nc1d=f_workspace.nc1d,
        ncsave=f_workspace.ncsave,
        qg1d=input_state["qg"],
        qgten=tendencies["qgten"],
        ngten=tendencies["ngten"],
        qbten=tendencies["qbten"],
        ng1d=g_workspace.ng1d,
        qb1d=g_workspace.qb1d,
        dt=resolved_dt,
        odt=odt,
    )
    for name in ("qcten", "ncten"):
        tendencies[name] = cloud[name]
    for name in ("qgten", "ngten", "qbten"):
        tendencies[name] = graupel[name]

    if production_sedimentation_moments:
        rain, ice, snow = _production_precipitation_checks_tensor(
            rho=rho,
            qr1d=input_state["qr"],
            nr1d=input_state["nr"],
            qrten=tendencies["qrten"],
            nrten=tendencies["nrten"],
            qi1d=input_state["qi"],
            ni1d=input_state["ni"],
            qiten=tendencies["qiten"],
            niten=tendencies["niten"],
            qs1d=input_state["qs"],
            qsten=tendencies["qsten"],
            dt=resolved_dt,
            odt=odt,
        )
    else:
        rain = rain_check_and_update(
            rho=rho,
            qr1d=input_state["qr"],
            nr1d=input_state["nr"],
            qrten=tendencies["qrten"],
            nrten=tendencies["nrten"],
            dt=resolved_dt,
            odt=odt,
        )
        ice = ice_check_and_update(
            rho=rho,
            qi1d=input_state["qi"],
            ni1d=input_state["ni"],
            qiten=tendencies["qiten"],
            niten=tendencies["niten"],
            dt=resolved_dt,
            odt=odt,
        )
        snow = snow_check_and_update(
            rho=rho,
            qs1d=input_state["qs"],
            qsten=tendencies["qsten"],
            dt=resolved_dt,
            odt=odt,
        )
    tendencies["qrten"] = rain["qrten"]
    tendencies["nrten"] = rain["nrten"]
    tendencies["qiten"] = ice["qiten"]
    tendencies["niten"] = ice["niten"]
    tendencies["qsten"] = snow["qsten"]
    snow_moment_function = _compute_sedimentation_snow_moments if production_sedimentation_moments else compute_snow_moments
    snow_moments = snow_moment_function(rs=snow["rs"], temp=temp, l_qs=snow["l_qs"], template=e_workspace.snow_moments)

    thermo, supersaturated = thermo_vars(qv=qv, temp=temp, pres=c_base.pres, rho=rho)
    tend_work, tend_views = _contiguous_tend_work_tensor(tend)

    workspace = TorchSegmentNWorkspace(
        base=m_result.workspace,
        tend_work=tend_work,
        tend=tend_views,
        tendencies=tendencies,
        tables=m_result.workspace.tables,
        temp=temp,
        qv=qv,
        rho=rho,
        nwfa=nwfa,
        nifa=nifa,
        nwfaten=nwfaten,
        nifaten=nifaten,
        l_qc=cloud["l_qc"],
        rc=cloud["rc"],
        nc=cloud["nc"],
        nc1d=f_workspace.nc1d,
        ncsave=f_workspace.ncsave,
        ilamc=cloud["ilamc"],
        mvd_c=cloud["mvd_c"],
        l_qr=rain["l_qr"],
        rr=rain["rr"],
        nr=rain["nr"],
        nr1d=input_state["nr"],
        ilamr=rain["ilamr"],
        mvd_r=rain["mvd_r"],
        l_qi=ice["l_qi"],
        ri=ice["ri"],
        ni=ice["ni"],
        ni1d=input_state["ni"],
        ilami=ice["ilami"],
        l_qs=snow["l_qs"],
        rs=snow["rs"],
        snow_moments=snow_moments,
        l_qg=graupel["l_qg"],
        rg=graupel["rg"],
        ng=graupel["ng"],
        rb=graupel["rb"],
        idx_bg=graupel["idx"],
        ilamg=graupel["ilamg"],
        mvd_g=graupel["mvd_g"],
        ng1d=g_workspace.ng1d,
        qb1d=g_workspace.qb1d,
        thermo=thermo,
        do_micro=g_workspace.do_micro,
        supersaturated=supersaturated,
        dt=resolved_dt,
        odt=odt,
    )
    return TorchSegmentNResult(result_state, workspace, m_result.ncol, m_result.nz)


def _o_cloud_update_and_check(
    *,
    rho: torch.Tensor,
    qc1d: torch.Tensor,
    nc: torch.Tensor,
    nc1d: torch.Tensor | None,
    ncsave: torch.Tensor | None,
    prw_vcd: torch.Tensor,
    pnc_wcd: torch.Tensor,
    qvten: torch.Tensor,
    qcten: torch.Tensor,
    ncten: torch.Tensor,
    nwfaten: torch.Tensor,
    tten: torch.Tensor,
    lvap: torch.Tensor,
    ocp: torch.Tensor,
    dt: float,
    odt: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
    """Fuse O's adjacent tendency update and cloud consistency check."""
    if nc1d is not None:
        values = _run_tensor_compile_island(
            "o_cloud_update_and_check_optional",
            _o_cloud_update_and_check_optional_kernel,
            rho,
            rho,
            qc1d,
            nc,
            nc1d,
            prw_vcd,
            pnc_wcd,
            qvten,
            qcten,
            ncten,
            nwfaten,
            tten,
            lvap,
            ocp,
            _tensor(_STATIC_CLOUD_CCE[1], rho),
            _tensor(_STATIC_CLOUD_CCG[0], rho),
            _tensor(_STATIC_CLOUD_OCG2, rho),
            _tensor(_STATIC_CLOUD_CCG[1], rho),
            _tensor(_STATIC_CLOUD_OCG1, rho),
            float(dt),
            float(odt),
        )
        tendency_names = ("qvten", "qcten", "ncten", "nwfaten", "tten")
        cloud_names = ("l_qc", "qc1d", "nc1d", "rc", "nc", "qcten", "ncten", "ilamc", "mvd_c")
        return (
            dict(zip(tendency_names, values[:5], strict=True)),
            dict(zip(cloud_names, values[5:], strict=True)),
        )
    cloud_number_base = ncsave if ncsave is not None else _scalar(float(constants.NT_C_L), rho)
    values = _run_tensor_compile_island(
        "o_cloud_update_and_check_base",
        _o_cloud_update_and_check_base_kernel,
        rho,
        rho,
        qc1d,
        nc,
        cloud_number_base,
        prw_vcd,
        pnc_wcd,
        qvten,
        qcten,
        ncten,
        nwfaten,
        tten,
        lvap,
        ocp,
        _tensor(_STATIC_CLOUD_CCG[1], rho),
        _tensor(_STATIC_CLOUD_OCG1, rho),
        float(dt),
        float(odt),
    )
    tendency_names = ("qvten", "qcten", "ncten", "nwfaten", "tten")
    cloud_names = ("l_qc", "qc1d", "rc", "nc", "qcten", "ncten", "ilamc", "mvd_c")
    cloud = dict(zip(cloud_names, values[5:], strict=True))
    cloud["nc1d"] = None
    return dict(zip(tendency_names, values[:5], strict=True)), cloud


def _o_cloud_tendency_updates(
    prw_vcd: torch.Tensor,
    pnc_wcd: torch.Tensor,
    qvten: torch.Tensor,
    qcten: torch.Tensor,
    ncten: torch.Tensor,
    nwfaten: torch.Tensor,
    tten: torch.Tensor,
    lvap: torch.Tensor,
    ocp: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return (
        qvten - prw_vcd,
        qcten + prw_vcd,
        ncten + pnc_wcd,
        nwfaten - pnc_wcd,
        tten + lvap * ocp * prw_vcd,
    )


def _o_cloud_update_and_check_optional_kernel(
    rho: torch.Tensor,
    qc1d: torch.Tensor,
    nc: torch.Tensor,
    nc1d: torch.Tensor,
    prw_vcd: torch.Tensor,
    pnc_wcd: torch.Tensor,
    qvten: torch.Tensor,
    qcten: torch.Tensor,
    ncten: torch.Tensor,
    nwfaten: torch.Tensor,
    tten: torch.Tensor,
    lvap: torch.Tensor,
    ocp: torch.Tensor,
    cce1_table: torch.Tensor,
    ccg0_table: torch.Tensor,
    ocg2_table: torch.Tensor,
    ccg1_table: torch.Tensor,
    ocg1_table: torch.Tensor,
    dt: float,
    odt: float,
) -> tuple[torch.Tensor, ...]:
    updated = _o_cloud_tendency_updates(
        prw_vcd, pnc_wcd, qvten, qcten, ncten, nwfaten, tten, lvap, ocp
    )
    cloud = _cloud_check_and_update_optional_kernel(
        rho,
        qc1d,
        nc,
        updated[1],
        updated[2],
        nc1d,
        cce1_table,
        ccg0_table,
        ocg2_table,
        ccg1_table,
        ocg1_table,
        dt,
        odt,
    )
    return (updated[0], cloud[5], cloud[6], updated[3], updated[4], *cloud)


def _o_cloud_update_and_check_base_kernel(
    rho: torch.Tensor,
    qc1d: torch.Tensor,
    nc: torch.Tensor,
    cloud_number_base: torch.Tensor,
    prw_vcd: torch.Tensor,
    pnc_wcd: torch.Tensor,
    qvten: torch.Tensor,
    qcten: torch.Tensor,
    ncten: torch.Tensor,
    nwfaten: torch.Tensor,
    tten: torch.Tensor,
    lvap: torch.Tensor,
    ocp: torch.Tensor,
    ccg1_table: torch.Tensor,
    ocg1_table: torch.Tensor,
    dt: float,
    odt: float,
) -> tuple[torch.Tensor, ...]:
    del nc
    updated = _o_cloud_tendency_updates(
        prw_vcd, pnc_wcd, qvten, qcten, ncten, nwfaten, tten, lvap, ocp
    )
    cloud = _cloud_check_and_update_base_kernel(
        rho,
        qc1d,
        cloud_number_base,
        updated[1],
        updated[2],
        ccg1_table,
        ocg1_table,
        dt,
        odt,
    )
    return (updated[0], cloud[4], cloud[5], updated[3], updated[4], *cloud)


def _o_rain_update_and_check(
    *,
    rho: torch.Tensor,
    qr1d: torch.Tensor,
    nr1d: torch.Tensor,
    prv_rev: torch.Tensor,
    pnr_rev: torch.Tensor,
    qrten: torch.Tensor,
    qvten: torch.Tensor,
    nrten: torch.Tensor,
    nwfaten: torch.Tensor,
    tten: torch.Tensor,
    lvap: torch.Tensor,
    ocp: torch.Tensor,
    dt: float,
    odt: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    values = _run_tensor_compile_island(
        "o_rain_update_and_check",
        _o_rain_update_and_check_kernel,
        rho,
        rho,
        qr1d,
        nr1d,
        prv_rev,
        pnr_rev,
        qrten,
        qvten,
        nrten,
        nwfaten,
        tten,
        lvap,
        ocp,
        float(dt),
        float(odt),
        _tensor(constants.BM_R, rho),
    )
    tendency_names = ("qrten", "qvten", "nrten", "nwfaten", "tten")
    rain_names = ("l_qr", "qr1d", "nr1d", "rr", "nr", "qrten", "nrten", "ilamr", "mvd_r")
    return (
        dict(zip(tendency_names, values[:5], strict=True)),
        dict(zip(rain_names, values[5:], strict=True)),
    )


def _o_rain_update_and_check_kernel(
    rho: torch.Tensor,
    qr1d: torch.Tensor,
    nr1d: torch.Tensor,
    prv_rev: torch.Tensor,
    pnr_rev: torch.Tensor,
    qrten: torch.Tensor,
    qvten: torch.Tensor,
    nrten: torch.Tensor,
    nwfaten: torch.Tensor,
    tten: torch.Tensor,
    lvap: torch.Tensor,
    ocp: torch.Tensor,
    dt: float,
    odt: float,
    rain_power: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    qrten_updated = qrten - prv_rev
    qvten_updated = qvten + prv_rev
    nrten_updated = nrten - pnr_rev
    nwfaten_updated = nwfaten + pnr_rev
    tten_updated = tten - lvap * ocp * prv_rev
    rain = _rain_check_and_update_kernel(
        rho,
        qr1d,
        nr1d,
        qrten_updated,
        nrten_updated,
        dt,
        odt,
        rain_power,
    )
    return (rain[5], qvten_updated, rain[6], nwfaten_updated, tten_updated, *rain)


def run_o(
    state: Mapping[str, torch.Tensor] | TorchSegmentNResult,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    copy: bool = False,
) -> TorchSegmentOResult:
    n_result = _segment_n_result(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
    )
    input_state = n_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = n_result.workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment O dt must be nonzero")
    odt = 1.0 / resolved_dt
    m_workspace = n_result.workspace.base
    l_workspace = m_workspace.base
    k_workspace = l_workspace.base
    j_workspace = k_workspace.base
    i_workspace = j_workspace.base
    h_workspace = i_workspace.base
    g_workspace = h_workspace.base
    f_workspace = g_workspace.base
    e_workspace = f_workspace.base
    d_workspace = e_workspace.base
    c_base = d_workspace.base

    tend = _tensor_mapping(n_result.workspace.tend, copy=copy)
    tendencies = _tensor_mapping(n_result.workspace.tendencies, copy=copy)
    temp = n_result.workspace.temp
    qv = n_result.workspace.qv
    rho = n_result.workspace.rho
    thermo = dict(n_result.workspace.thermo)
    idx_d = _int_scalar(0, temp)
    idx_c = _int_scalar(0, temp)
    idx_n = _int_scalar(0, temp)

    cloud = {
        "l_qc": n_result.workspace.l_qc,
        "rc": n_result.workspace.rc,
        "nc": n_result.workspace.nc,
        "nc1d": n_result.workspace.nc1d,
        "ilamc": n_result.workspace.ilamc,
        "mvd_c": n_result.workspace.mvd_c,
    }
    if not turn_off_micro and cloud_condensation:
        tend, idx_d, idx_c, idx_n = cloud_condensation_step_tensor(
            rho=rho,
            temp=temp,
            w1d=input_state["w"],
            ssatw=thermo["ssatw"],
            lvap=thermo["lvap"],
            tcond=thermo["tcond"],
            diffu=thermo["diffu"],
            lvt2=thermo["lvt2"],
            nwfa1d=d_workspace.nwfa1d,
            nwfa=n_result.workspace.nwfa,
            ncsave=n_result.workspace.ncsave,
            qv=qv,
            qvs=thermo["qvs"],
            l_qc=n_result.workspace.l_qc,
            rc=n_result.workspace.rc,
            nc=n_result.workspace.nc,
            tend=tend,
            tables=n_result.workspace.tables,
            dt=resolved_dt,
            odt=odt,
            idx_d=None,
            idx_c=None,
            idx_n=None,
        )
        if _regional_execution_enabled("o_fused_updates"):
            cloud_tendencies, cloud = _o_cloud_update_and_check(
                rho=rho,
                qc1d=input_state["qc"],
                nc=n_result.workspace.nc,
                nc1d=n_result.workspace.nc1d,
                ncsave=n_result.workspace.ncsave,
                prw_vcd=tend["prw_vcd"],
                pnc_wcd=tend["pnc_wcd"],
                qvten=tendencies["qvten"],
                qcten=tendencies["qcten"],
                ncten=tendencies["ncten"],
                nwfaten=tendencies["nwfaten"],
                tten=tendencies["tten"],
                lvap=thermo["lvap"],
                ocp=thermo["ocp"],
                dt=resolved_dt,
                odt=odt,
            )
            tendencies.update(cloud_tendencies)
        else:
            tendencies["qvten"] = tendencies["qvten"] - tend["prw_vcd"]
            tendencies["qcten"] = tendencies["qcten"] + tend["prw_vcd"]
            tendencies["ncten"] = tendencies["ncten"] + tend["pnc_wcd"]
            tendencies["nwfaten"] = tendencies["nwfaten"] - tend["pnc_wcd"]
            tendencies["tten"] = tendencies["tten"] + thermo["lvap"] * thermo["ocp"] * tend["prw_vcd"]

            cloud = cloud_check_and_update(
                rho=rho,
                qc1d=input_state["qc"],
                nc=n_result.workspace.nc,
                qcten=tendencies["qcten"],
                ncten=tendencies["ncten"],
                dt=resolved_dt,
                odt=odt,
                nc1d=n_result.workspace.nc1d,
                ncsave=n_result.workspace.ncsave,
            )
            tendencies["qcten"] = cloud["qcten"]
            tendencies["ncten"] = cloud["ncten"]

        qv, temp, rho, thermo = _refresh_thermo_tensor(
            input_state=input_state,
            tendencies=tendencies,
            pres=c_base.pres,
            dt=resolved_dt,
        )

    rain = {
        "l_qr": n_result.workspace.l_qr,
        "rr": n_result.workspace.rr,
        "nr": n_result.workspace.nr,
        "nr1d": n_result.workspace.nr1d,
        "ilamr": n_result.workspace.ilamr,
        "mvd_r": n_result.workspace.mvd_r,
    }
    if not turn_off_micro:
        tend = rain_evaporation_tensor(
            rho=rho,
            temp=temp,
            ssatw=thermo["ssatw"],
            lvap=thermo["lvap"],
            tcond=thermo["tcond"],
            diffu=thermo["diffu"],
            vsc2=thermo["vsc2"],
            rhof2=thermo["rhof2"],
            qv=qv,
            qvs=thermo["qvs"],
            l_qr=rain["l_qr"],
            rr=rain["rr"],
            nr=rain["nr"],
            ilamr=rain["ilamr"],
            tend=tend,
            odt=odt,
        )
        if _regional_execution_enabled("o_fused_updates"):
            rain_tendencies, rain = _o_rain_update_and_check(
                rho=rho,
                qr1d=input_state["qr"],
                nr1d=input_state["nr"],
                prv_rev=tend["prv_rev"],
                pnr_rev=tend["pnr_rev"],
                qrten=tendencies["qrten"],
                qvten=tendencies["qvten"],
                nrten=tendencies["nrten"],
                nwfaten=tendencies["nwfaten"],
                tten=tendencies["tten"],
                lvap=thermo["lvap"],
                ocp=thermo["ocp"],
                dt=resolved_dt,
                odt=odt,
            )
            tendencies.update(rain_tendencies)
        else:
            tendencies["qrten"] = tendencies["qrten"] - tend["prv_rev"]
            tendencies["qvten"] = tendencies["qvten"] + tend["prv_rev"]
            tendencies["nrten"] = tendencies["nrten"] - tend["pnr_rev"]
            tendencies["nwfaten"] = tendencies["nwfaten"] + tend["pnr_rev"]
            tendencies["tten"] = tendencies["tten"] - thermo["lvap"] * thermo["ocp"] * tend["prv_rev"]

            rain = rain_check_and_update(
                rho=rho,
                qr1d=input_state["qr"],
                nr1d=input_state["nr"],
                qrten=tendencies["qrten"],
                nrten=tendencies["nrten"],
                dt=resolved_dt,
                odt=odt,
            )
            tendencies["qrten"] = rain["qrten"]
            tendencies["nrten"] = rain["nrten"]

        qv, temp, rho, thermo = _refresh_thermo_tensor(
            input_state=input_state,
            tendencies=tendencies,
            pres=c_base.pres,
            dt=resolved_dt,
        )

    tend_work, tend_views = _contiguous_tend_work_tensor(tend)
    workspace = TorchSegmentOWorkspace(
        base=n_result.workspace,
        tend_work=tend_work,
        tend=tend_views,
        tendencies=tendencies,
        diagnostics=c_base.diagnostics,
        fallspeeds=c_base.fallspeeds,
        tables=n_result.workspace.tables,
        vtboost=k_workspace.vtboost,
        temp=temp,
        qv=qv,
        rho=rho,
        nwfa=n_result.workspace.nwfa,
        nifa=n_result.workspace.nifa,
        nwfaten=tendencies["nwfaten"],
        nifaten=tendencies["nifaten"],
        nwfa1d=d_workspace.nwfa1d,
        nifa1d=d_workspace.nifa1d,
        l_qc=cloud["l_qc"],
        rc=cloud["rc"],
        nc=cloud["nc"],
        nc1d=n_result.workspace.nc1d,
        ncsave=n_result.workspace.ncsave,
        ilamc=cloud["ilamc"],
        mvd_c=cloud["mvd_c"],
        l_qr=rain["l_qr"],
        rr=rain["rr"],
        nr=rain["nr"],
        nr1d=n_result.workspace.nr1d,
        ilamr=rain["ilamr"],
        mvd_r=rain["mvd_r"],
        l_qi=n_result.workspace.l_qi,
        ri=n_result.workspace.ri,
        ni=n_result.workspace.ni,
        ni1d=n_result.workspace.ni1d,
        ilami=n_result.workspace.ilami,
        l_qs=n_result.workspace.l_qs,
        rs=n_result.workspace.rs,
        snow_moments=n_result.workspace.snow_moments,
        l_qg=n_result.workspace.l_qg,
        rg=n_result.workspace.rg,
        ng=n_result.workspace.ng,
        rb=n_result.workspace.rb,
        idx_bg=n_result.workspace.idx_bg,
        ilamg=n_result.workspace.ilamg,
        mvd_g=n_result.workspace.mvd_g,
        ng1d=n_result.workspace.ng1d,
        qb1d=n_result.workspace.qb1d,
        thermo=thermo,
        idx_d=idx_d,
        idx_c=idx_c,
        idx_n=idx_n,
        active_columns=_active_columns_from_g_workspace(g_workspace),
        do_micro=n_result.workspace.do_micro,
        supersaturated=n_result.workspace.supersaturated,
        dt=resolved_dt,
        odt=odt,
    )
    return TorchSegmentOResult(result_state, workspace, n_result.ncol, n_result.nz)


def initialize_graupel_columns_tensor(
    *,
    rho: torch.Tensor,
    qg1d: torch.Tensor,
    ng1d: torch.Tensor,
    qb1d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_tensor_compile_island(
        "initialize_graupel_columns",
        _initialize_graupel_columns_kernel,
        qg1d,
        rho,
        qg1d,
        ng1d,
        qb1d,
        _tensor(_STATIC_GRAUPEL_AM_G, qg1d),
    )


def _initialize_graupel_columns_kernel(
    rho: torch.Tensor,
    qg1d: torch.Tensor,
    ng1d: torch.Tensor,
    qb1d: torch.Tensor,
    am_g_table: torch.Tensor,
    _mass_limit: float = float(constants.R1),
    _eps: float = float(constants.EPS),
    _volume_scale: float = float(
        constants.METERS3_TO_LITERS / _SCALAR_GRAUPEL_RHO_G[constants.IDX_BG1 - 1]
    ),
    _lambda_g: float = float((3.0 + constants.MU_G + 0.672) / 5.0e-3),
    _number_scale: float = float(_SCALAR_GRAUPEL_CGG[1][0] * constants.OGG3),
    _density_scale: float = 10.0,
    _density_offset: float = 0.5,
    _density_count: int = int(constants.NRHG),
) -> tuple[torch.Tensor, torch.Tensor]:
    initialize_column = (
        (torch.sum(qg1d, dim=1) > _mass_limit)
        & (torch.sum(ng1d, dim=1) < _eps)
        & (torch.sum(qb1d, dim=1) < _eps)
    )[:, None]
    active = initialize_column & (qg1d > _mass_limit)
    rg = qg1d * rho
    rb = rg * _volume_scale
    density = _density_scale * _clamp_tiny(rg) / _clamp_tiny(rb)
    idx = torch.clamp(
        torch.floor(density + _density_offset).to(torch.int64) + 1,
        1,
        _density_count,
    )
    ng_value = _number_scale * rg * (_lambda_g * _lambda_g * _lambda_g) / am_g_table[idx - 1] / rho
    qb_value = rb / rho
    return torch.where(active, ng_value, ng1d), torch.where(active, qb_value, qb1d)


def run_p1(
    state: Mapping[str, torch.Tensor] | TorchSegmentOResult,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = False,
    max_sedimentation_substeps: int | None = None,
    copy: bool = False,
) -> TorchSegmentP1Result:
    if max_sedimentation_substeps is not None and max_sedimentation_substeps < 1:
        raise ValueError("Segment P1 max_sedimentation_substeps must be positive")
    o_result = _segment_o_result(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
    )
    input_state = o_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = o_result.workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment P1 dt must be nonzero")
    sedimentation_substeps = _default_sedimentation_substeps(
        dt=resolved_dt,
        semi_sedi_flag=semi_sedi_flag,
        override=max_sedimentation_substeps,
    )
    odt = 1.0 / resolved_dt

    tend = _tensor_mapping(o_result.workspace.tend, copy=copy)
    tendencies = _tensor_mapping(o_result.workspace.tendencies, copy=copy)
    diagnostics = _tensor_mapping(o_result.workspace.diagnostics, copy=copy)
    fallspeeds = _tensor_mapping(o_result.workspace.fallspeeds, copy=copy)
    rr = o_result.workspace.rr
    nr = o_result.workspace.nr
    vtrr = fallspeeds["vtrr"]
    vtnr = fallspeeds["vtnr"]
    substeps_sedi = torch.ones((o_result.ncol,), dtype=torch.int64, device=rr.device)
    ktop_sedi = torch.ones((o_result.ncol,), dtype=torch.int64, device=rr.device)
    branch_active = _column_activity(o_result.workspace.l_qr)

    if not semi_sedi_flag:
        vtrr, vtnr, substeps_sedi, ktop_sedi = rain_fallspeed_tensor(
            rhof=o_result.workspace.thermo["rhof"],
            rr=rr,
            ilamr=o_result.workspace.ilamr,
            dz1d=input_state["dz"],
            vt=vtrr,
            vtn=vtnr,
            dt=resolved_dt,
        )
        rain_xr = _species_tensor(rr, nr)
        rain_vt = _species_tensor(vtrr[:, : rr.shape[1]], vtnr[:, : nr.shape[1]])
        rain_xten = _species_tensor(tendencies["qrten"], tendencies["nrten"])
        rain_precip = diagnostics["rain_precip"]
        rain_xr, rain_xten, rain_precip = _run_sedimentation_recurrence(
            "eulerian_species_recurrence",
            _eulerian_sedimentation_species_recurrence,
            rain_xr,
            rain_xr,
            rain_vt,
            input_state["dz"],
            o_result.workspace.rho,
            rain_xten,
            _species_limit_tensor((constants.R1, constants.R2), reference=rain_xr),
            _sedimentation_steps_tensor(substeps_sedi, o_result.ncol, rain_xr),
            _sedimentation_ktop_tensor(ktop_sedi, o_result.ncol, rr.shape[1], rain_xr),
            rain_precip,
            sedimentation_substeps,
            resolved_dt,
            0,
        )
        rr, nr = rain_xr.unbind(dim=0)
        tendencies["qrten"], tendencies["nrten"] = rain_xten.unbind(dim=0)
        diagnostics["rain_precip"] = rain_precip
        _record_sedimentation_substep_overflow(diagnostics, substeps_sedi, sedimentation_substeps, branch_active,
                                               region=SEDIMENTATION_BOUND_RAIN_GRAUPEL)

    fallspeeds = {**fallspeeds, "vtrr": vtrr, "vtnr": vtnr}
    tend_work, tend_views = _contiguous_tend_work_tensor(tend)
    workspace = TorchSegmentP1Workspace(
        base=o_result.workspace,
        tend_work=tend_work,
        tend=tend_views,
        tendencies=tendencies,
        diagnostics=diagnostics,
        fallspeeds=fallspeeds,
        rr=rr,
        nr=nr,
        vtrr=vtrr,
        vtnr=vtnr,
        substeps_sedi=substeps_sedi,
        ktop_sedi=ktop_sedi,
        rain_precip=diagnostics["rain_precip"],
        semi_sedi_flag=bool(semi_sedi_flag),
        branch_active=branch_active,
        dt=resolved_dt,
        odt=odt,
    )
    return TorchSegmentP1Result(result_state, workspace, o_result.ncol, o_result.nz)


def _run_sedimentation_recurrence(
    name: str,
    function: Any,
    reference: torch.Tensor,
    *args: Any,
) -> Any:
    status_name = f"sedimentation_recurrence:{name}"
    if (
        not _sedimentation_recurrence_compile_enabled(name)
        or reference.device.type not in SEDIMENTATION_RECURRENCE_COMPILE_DEVICES
        or torch.compiler.is_compiling()
    ):
        return function(*args)
    prepared_dispatch = _ACTIVE_PREPARED_COMPILE_DISPATCH.get()
    prepared_key = _prepared_compile_key(
        status_name,
        function,
        _prepared_runtime_signature(reference, args),
    )
    if prepared_dispatch is not None:
        prepared_callable = prepared_dispatch.callables.get(prepared_key)
        if prepared_callable is not None:
            prepared_dispatch.fast_hits += 1
            try:
                return prepared_callable(*args)
            except Exception as exc:
                reason = _compile_exception_reason(exc)
                raise RuntimeError(
                    f"prepared compile dispatch {status_name} runtime_failed: {reason}"
                ) from exc
        prepared_dispatch.misses += 1
    cache_key = _compile_cache_key(name, function, reference, args=args)
    if cache_key in _SEDIMENTATION_RECURRENCE_COMPILE_DISABLED:
        if prepared_dispatch is not None:
            raise RuntimeError(f"prepared compile dispatch {status_name} is disabled")
        return function(*args)
    compiled = _SEDIMENTATION_RECURRENCE_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        compiled, result, failure_stage, failure, compile_timing = _compile_and_prime_with_scoped_scalar_capture(
            function,
            args,
            {
                "fullgraph": True,
                "dynamic": False,
                "options": {"triton.cudagraphs": False},
            },
        )
        _record_compile_island_timing(status_name, cache_key, compile_timing)
        if failure is not None:
            reason = _compile_exception_reason(failure)
            _SEDIMENTATION_RECURRENCE_COMPILE_DISABLED.add(cache_key)
            _record_tensor_compile_island_status(
                status_name,
                failure_stage or "compile_failed",
                reason,
                cache_key=cache_key,
            )
            _raise_compile_island_failure_if_strict(
                status_name,
                failure_stage or "compile_failed",
                reason,
                failure,
            )
            if prepared_dispatch is not None:
                raise RuntimeError(
                    f"prepared compile dispatch {status_name} {failure_stage or 'compile_failed'}: {reason}"
                ) from failure
            return function(*args)
        assert compiled is not None
        _SEDIMENTATION_RECURRENCE_COMPILED_CACHE[cache_key] = compiled
        _record_tensor_compile_island_status(status_name, "compiled", cache_key=cache_key)
        if prepared_dispatch is not None:
            prepared_dispatch.callables[prepared_key] = compiled
        return result
    try:
        result = compiled(*args)
    except Exception as exc:
        reason = _compile_exception_reason(exc)
        _SEDIMENTATION_RECURRENCE_COMPILED_CACHE.pop(cache_key, None)
        _SEDIMENTATION_RECURRENCE_COMPILE_DISABLED.add(cache_key)
        _record_tensor_compile_island_status(
            status_name,
            "runtime_failed",
            reason,
            cache_key=cache_key,
        )
        _raise_compile_island_failure_if_strict(status_name, "runtime_failed", reason, exc)
        if prepared_dispatch is not None:
            raise RuntimeError(
                f"prepared compile dispatch {status_name} runtime_failed: {reason}"
            ) from exc
        return function(*args)
    if prepared_dispatch is not None:
        prepared_dispatch.callables[prepared_key] = compiled
    return result


def _sedimentation_recurrence_compile_enabled(name: str) -> bool:
    lowered = name.lower()
    if lowered in {"eulerian_species_recurrence", "snow_sedimentation_recurrence"}:
        if SEDIMENTATION_RECURRENCE_EXECUTION_MODE == "fixed_step":
            return False
        return EULERIAN_SEDIMENTATION_RECURRENCE_COMPILE_CUDA or SEMI_SEDIMENTATION_RECURRENCE_COMPILE_CUDA
    if lowered in {
        "eulerian_species_recurrence_step",
        "snow_sedimentation_recurrence_step",
    }:
        if SEDIMENTATION_RECURRENCE_EXECUTION_MODE != "fixed_step":
            return False
        return EULERIAN_SEDIMENTATION_RECURRENCE_COMPILE_CUDA or SEMI_SEDIMENTATION_RECURRENCE_COMPILE_CUDA
    if lowered.endswith("_recurrence") and "_step" not in lowered:
        return False
    if SEDIMENTATION_RECURRENCE_COMPILE_CUDA:
        return True
    if "semilagrangian" in lowered or "semi_lagrangian" in lowered:
        return SEMI_SEDIMENTATION_RECURRENCE_COMPILE_CUDA
    if "eulerian" in lowered:
        return EULERIAN_SEDIMENTATION_RECURRENCE_COMPILE_CUDA
    return (
        EULERIAN_SEDIMENTATION_RECURRENCE_COMPILE_CUDA
        or SEMI_SEDIMENTATION_RECURRENCE_COMPILE_CUDA
    )


def _eulerian_sedimentation_species_recurrence_step_kernel(
    xr: torch.Tensor,
    vt: torch.Tensor,
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    xten: torch.Tensor,
    limit_tensor: torch.Tensor,
    steps_tensor: torch.Tensor,
    ktop_tensor: torch.Tensor,
    active_step: torch.Tensor,
    precip: torch.Tensor,
    dt: float,
    precip_species_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xr, xten, precip_next = _sedimentation_species_tensor_active_step(
        xr,
        vt,
        dz1d,
        rho,
        xten,
        limit_tensor,
        steps_tensor,
        ktop_tensor,
        active_step,
        precip,
        dt,
        precip_species_index,
    )
    return xr, xten, precip_next if precip_next is not None else precip


def _eulerian_sedimentation_species_recurrence(
    xr: torch.Tensor,
    vt: torch.Tensor,
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    xten: torch.Tensor,
    limit_tensor: torch.Tensor,
    steps_tensor: torch.Tensor,
    ktop_tensor: torch.Tensor,
    precip: torch.Tensor,
    sedimentation_substeps: int,
    dt: float,
    precip_species_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for step_index in range(sedimentation_substeps):
        active_step = steps_tensor > step_index
        step_args = (
            xr, vt, dz1d, rho, xten, limit_tensor, steps_tensor, ktop_tensor,
            active_step, precip, dt, precip_species_index,
        )
        if SEDIMENTATION_RECURRENCE_EXECUTION_MODE == "fixed_step":
            xr, xten, precip = _run_sedimentation_recurrence(
                "eulerian_species_recurrence_step",
                _eulerian_sedimentation_species_recurrence_step_kernel,
                xr,
                *step_args,
            )
        else:
            xr, xten, precip = _eulerian_sedimentation_species_recurrence_step_kernel(
                *step_args
            )
    return xr, xten, precip


def _rain_semilagrangian_recurrence_update_step_kernel(
    dz1d: torch.Tensor,
    qr1d: torch.Tensor,
    nr1d_initial: torch.Tensor,
    rho: torch.Tensor,
    rhof: torch.Tensor,
    l_qr: torch.Tensor,
    rain_xr: torch.Tensor,
    nr1d: torch.Tensor,
    ilamr: torch.Tensor,
    mvd_r: torch.Tensor,
    rain_xten: torch.Tensor,
    rain_precip: torch.Tensor,
    rain_vt: torch.Tensor,
    sedimented: torch.Tensor,
    sedimented_tendencies: torch.Tensor,
    sedimented_precip: torch.Tensor,
    rain_zero_based_levels: torch.Tensor,
    rain_one_based_levels: torch.Tensor,
    step_active: torch.Tensor,
    branch_active_1d: torch.Tensor,
    dt: float,
    odt: float,
    rain_power: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rr, nr = rain_xr.unbind(dim=0)
    qrten, nrten = rain_xten.unbind(dim=0)
    vtrr, vtnr = rain_vt.unbind(dim=0)
    step_active_1d = step_active[:, 0] & branch_active_1d
    rr_next, nr_next = sedimented.unbind(dim=0)
    qrten_next, nrten_next = sedimented_tendencies.unbind(dim=0)
    rain_precip_next = sedimented_precip
    rr, nr = _masked_merge_components(step_active, (rr, nr), (rr_next, nr_next))
    qrten, nrten = _masked_merge_components(step_active, (qrten, nrten), (qrten_next, nrten_next))
    rain_precip = torch.where(step_active_1d, rain_precip_next, rain_precip)
    l_qr_next, _, nr1d_next, rr_check, nr_check, qrten_check, nrten_check, ilamr_check, mvd_r_check = _rain_check_and_update_kernel(
        rho,
        qr1d,
        nr1d_initial,
        qrten,
        nrten,
        dt,
        odt,
        rain_power,
    )
    l_qr = torch.where(step_active, l_qr_next, l_qr)
    rr, nr, nr1d, ilamr, mvd_r, qrten, nrten = _masked_merge_components(step_active, (rr, nr, nr1d, ilamr, mvd_r, qrten, nrten), (rr_check, nr_check, nr1d_next, ilamr_check, mvd_r_check, qrten_check, nrten_check))
    vtrr_next, vtnr_next, _, _ = _rain_fallspeed_zero_tail_kernel(
        rhof,
        rr,
        ilamr,
        dz1d,
        rain_zero_based_levels,
        rain_one_based_levels,
        dt,
        False,
    )
    vtrr, vtnr = _masked_merge_components(step_active, (vtrr, vtnr), (vtrr_next, vtnr_next))
    return (l_qr, _species_tensor(rr, nr), nr1d, ilamr, mvd_r, _species_tensor(qrten, nrten), rain_precip, _species_tensor(vtrr, vtnr))


def _rain_semilagrangian_recurrence(dz1d: torch.Tensor, qr1d: torch.Tensor, nr1d_initial: torch.Tensor, rho: torch.Tensor, rhof: torch.Tensor, l_qr: torch.Tensor, rr: torch.Tensor, nr: torch.Tensor, nr1d: torch.Tensor, ilamr: torch.Tensor, mvd_r: torch.Tensor, qrten: torch.Tensor, nrten: torch.Tensor, rain_precip: torch.Tensor, vtrr: torch.Tensor, vtnr: torch.Tensor, rain_limit_tensor: torch.Tensor, rain_zero_based_levels: torch.Tensor, rain_one_based_levels: torch.Tensor, substeps_sedi: torch.Tensor, branch_active: torch.Tensor, sedimentation_substeps: int, dt: float, odt: float, rain_power: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rain alone through the shared semi-Lagrangian recurrence (segment P2)."""
    rain = _rain_semilagrangian_member(qr1d, nr1d_initial, l_qr, rr, nr, nr1d, ilamr, mvd_r, qrten, nrten,
                                       rain_precip, vtrr, vtnr, rain_limit_tensor, rain_zero_based_levels,
                                       rain_one_based_levels, substeps_sedi, branch_active, rain_power)
    out = _semilagrangian_rain_graupel_recurrence(rain, None, dz1d=dz1d, rho=rho, rhof=rhof,
                                                  bound=sedimentation_substeps, dt=dt, odt=odt)
    return tuple(out[n] for n in _RAIN_SEMILAGRANGIAN_OUTPUTS)


_RAIN_SEMILAGRANGIAN_OUTPUTS = ("l_qr", "rr", "nr", "nr1d", "ilamr", "mvd_r", "qrten", "nrten", "rain_precip",
                                "vtrr", "vtnr")
_GRAUPEL_SEMILAGRANGIAN_OUTPUTS = ("l_qg", "rg", "ng", "rb", "idx_bg", "ilamg", "ng1d", "qb1d", "mvd_g", "qgten",
                                   "ngten", "qbten", "graupel_precip", "vtrg", "vtng")


def _rain_semilagrangian_member(qr1d: torch.Tensor, nr1d_initial: torch.Tensor, l_qr: torch.Tensor, rr: torch.Tensor, nr: torch.Tensor, nr1d: torch.Tensor, ilamr: torch.Tensor, mvd_r: torch.Tensor, qrten: torch.Tensor, nrten: torch.Tensor, rain_precip: torch.Tensor, vtrr: torch.Tensor, vtnr: torch.Tensor, rain_limit_tensor: torch.Tensor, rain_zero_based_levels: torch.Tensor, rain_one_based_levels: torch.Tensor, substeps_sedi: torch.Tensor, branch_active: torch.Tensor, rain_power: torch.Tensor) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    """Rain's per-column fields and its static operands for ``_semilagrangian_rain_graupel_recurrence``."""
    columns = dict(qr1d=qr1d, nr1d_initial=nr1d_initial, l_qr=l_qr, rr=rr, nr=nr, nr1d=nr1d, ilamr=ilamr,
                   mvd_r=mvd_r, qrten=qrten, nrten=nrten, rain_precip=rain_precip, vtrr=vtrr, vtnr=vtnr,
                   rain_limit=rain_limit_tensor, rain_steps=substeps_sedi, rain_active=branch_active)
    return columns, (rain_zero_based_levels, rain_one_based_levels, rain_power)


def _graupel_semilagrangian_member(qg1d: torch.Tensor, visco: torch.Tensor, l_qg: torch.Tensor, rg: torch.Tensor, ng: torch.Tensor, rb: torch.Tensor, idx_bg: torch.Tensor, ilamg: torch.Tensor, mvd_g: torch.Tensor, qgten: torch.Tensor, ngten: torch.Tensor, qbten: torch.Tensor, graupel_precip: torch.Tensor, vtrg: torch.Tensor, vtng: torch.Tensor, ng1d: torch.Tensor | None, qb1d: torch.Tensor | None, ng_template: torch.Tensor | None, qb_template: torch.Tensor | None, graupel_limit_tensor: torch.Tensor, graupel_am_g_table: torch.Tensor, graupel_cgg_table: torch.Tensor, graupel_zero_based_levels: torch.Tensor, graupel_one_based_levels: torch.Tensor, substeps_sedi: torch.Tensor, branch_active: torch.Tensor) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    """Graupel's per-column fields and its static operands for ``_semilagrangian_rain_graupel_recurrence``."""
    if ng_template is not None and qb_template is not None and (ng1d is None or qb1d is None):
        raise ValueError("Graupel semi-Lagrangian optional recurrence requires ng1d and qb1d")
    columns = dict(qg1d=qg1d, visco=visco, l_qg=l_qg, rg=rg, ng=ng, rb=rb, idx_bg=idx_bg, ilamg=ilamg,
                   mvd_g=mvd_g, qgten=qgten, ngten=ngten, qbten=qbten, graupel_precip=graupel_precip,
                   vtrg=vtrg, vtng=vtng, graupel_limit=graupel_limit_tensor, graupel_steps=substeps_sedi,
                   graupel_active=branch_active,
                   **{n: v for n, v in (("ng1d", ng1d), ("qb1d", qb1d), ("ng_template", ng_template),
                                        ("qb_template", qb_template)) if v is not None})
    return columns, (graupel_am_g_table, graupel_cgg_table, graupel_zero_based_levels, graupel_one_based_levels)


def _pack_species(*packed: torch.Tensor) -> torch.Tensor:
    """Species tensors of several kinds as one [species, ...] tensor (one kind: itself)."""
    if len(packed) == 1:
        return packed[0]
    return _species_tensor(*(component for kind in packed for component in kind.unbind(dim=0)))


def _semilagrangian_rain_graupel_recurrence(
    rain: tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]] | None,
    graupel: tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]] | None,
    *,
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    rhof: torch.Tensor,
    bound: int,
    dt: float,
    odt: float,
) -> dict[str, torch.Tensor]:
    """Rain and graupel semi-Lagrangian sedimentation (main.F90:592-690) in one substep loop.

    The source runs rain's substeps, then graupel's; the two loops share no state, so substep
    ``s`` of both runs here together. In the Fortran-ordered mode one
    ``semilagrangian_sedimentation`` evaluation per substep covers rain's two and graupel's three
    species, each with its own kind's substep count: every element sees the arithmetic of its
    own kind's call. ``zi`` (sequential sums of ``dz``) is formed once per call. Substeps run on
    one set of nested compaction tiers (kind ``rain_graupel``, or ``rain``/``graupel`` for one
    of them alone); a column needs substep ``s`` if either kind does, and a kind's own masks leave
    a column it does not need unchanged.

    Traced by ``torch.compile`` with ``SEDIMENTATION_SUBSTEP_REGION`` on (Fortran-ordered mode),
    each tier's substep loop leaves the traced graph (a graph break that reads no tensor value)
    and each substep runs in one compiled region (``_semilagrangian_rain_graupel_substep``),
    compiled once per input shape and reused by every substep of that shape; the tiers' gathers
    and write-backs stay in the traced graphs. The substep's arithmetic is the same in both forms.
    """
    kinds = tuple(name for name, member in (("rain", rain), ("graupel", graupel)) if member is not None)
    if not kinds:
        raise ValueError("semi-Lagrangian recurrence needs rain or graupel")
    fortran = SEMI_LAGRANGIAN_OVERLAP_MODE == "fortran_ordered"
    columns = dict(dz1d=dz1d, rho=rho, rhof=rhof)
    if fortran:
        columns["zi"] = _semilagrangian_ordered_zi(dz1d)
    for member in (rain, graupel):
        if member is not None:
            columns.update(member[0])
    # Static operands, the same on every tier and substep.
    operands = {}
    if rain is not None:
        operands["rain_levels0"], operands["rain_levels1"], operands["rain_power"] = rain[1]
    if graupel is not None:
        (operands["graupel_am_g_table"], operands["graupel_cgg_table"], operands["graupel_levels0"],
         operands["graupel_levels1"]) = graupel[1]
    return _semilagrangian_rain_graupel_tiers(columns, operands, kinds=kinds, fortran=fortran, bound=bound, dt=dt,
                                              odt=odt)


def _semilagrangian_rain_graupel_tiers(
    columns: dict[str, torch.Tensor],
    operands: dict[str, torch.Tensor],
    *,
    kinds: tuple[str, ...],
    fortran: bool,
    bound: int,
    dt: float,
    odt: float,
) -> dict[str, torch.Tensor]:
    """The substeps of ``_semilagrangian_rain_graupel_recurrence`` on its compaction tiers."""
    species_counts = tuple(2 if name == "rain" else 3 for name in kinds)

    def need(c: dict[str, torch.Tensor], first: int) -> torch.Tensor:
        needed = None
        for name in kinds:
            kind_needs = c[f"{name}_active"][:, 0] & (c[f"{name}_steps"] > first)
            needed = kind_needs if needed is None else needed | kind_needs
        return needed

    def run(c: dict[str, torch.Tensor], first: int, stop: int) -> dict[str, torch.Tensor]:
        # carry: what a substep updates; fixed: what it reads only (per tier).
        carry = {}
        fixed = dict(operands, dz1d=c["dz1d"], rho=c["rho"], rhof=c["rhof"])
        if "rain" in kinds:
            carry.update(rain_xr=_species_tensor(c["rr"], c["nr"]), rain_xten=_species_tensor(c["qrten"], c["nrten"]),
                         rain_vt=_species_tensor(c["vtrr"], c["vtnr"]), rain_precip=c["rain_precip"],
                         **{n: c[n] for n in ("l_qr", "nr1d", "ilamr", "mvd_r")})
            fixed.update(qr1d=c["qr1d"], nr1d_initial=c["nr1d_initial"], rain_steps=c["rain_steps"],
                         rain_active_1d=c["rain_active"][:, 0])
        if "graupel" in kinds:
            carry.update(graupel_xr=_species_tensor(c["rg"], c["ng"], c["rb"]),
                         graupel_xten=_species_tensor(c["qgten"], c["ngten"], c["qbten"]),
                         graupel_vt=_species_tensor(c["vtrg"], c["vtng"], c["vtrg"]),
                         graupel_precip=c["graupel_precip"],
                         **{n: c[n] for n in ("l_qg", "idx_bg", "ilamg", "mvd_g", "ng1d", "qb1d") if n in c})
            fixed.update(qg1d=c["qg1d"], visco=c["visco"], graupel_steps=c["graupel_steps"],
                         graupel_active_1d=c["graupel_active"][:, 0],
                         **{n: c[n] for n in ("ng_template", "qb_template") if n in c})
        if fortran:
            # Per-species substep counts and limits, fixed over the call.
            fixed["zi"] = c["zi"]
            fixed["steps_all"] = (c[f"{kinds[0]}_steps"] if len(kinds) == 1 else
                                  _species_tensor(*(c[f"{name}_steps"] for name, count in zip(kinds, species_counts)
                                                    for _ in range(count))))
            fixed["limit_all"] = _pack_species(*(c[f"{name}_limit"] for name in kinds))
        else:
            fixed["geometry"] = {name: semilagrangian_geometry_tensor(dz1d=c["dz1d"], steps=c[f"{name}_steps"],
                                                                      dt=dt, odt=odt)
                                 for name in kinds}
            fixed.update({f"{name}_limit": c[f"{name}_limit"] for name in kinds})
        if fortran and SEDIMENTATION_SUBSTEP_REGION and torch.compiler.is_compiling():
            carry = _semilagrangian_rain_graupel_substeps_in_region(carry, fixed, first, stop, kinds=kinds, dt=dt,
                                                                    odt=odt)
        else:
            for step_index in range(first, stop):
                carry = _semilagrangian_rain_graupel_substep(carry, fixed, step_index, kinds=kinds, fortran=fortran,
                                                             dt=dt, odt=odt)
        updated = {}
        if "rain" in kinds:
            rr, nr = carry["rain_xr"].unbind(dim=0)
            qrten, nrten = carry["rain_xten"].unbind(dim=0)
            vtrr, vtnr = carry["rain_vt"].unbind(dim=0)
            updated.update(l_qr=carry["l_qr"], rr=rr, nr=nr, nr1d=carry["nr1d"], ilamr=carry["ilamr"],
                           mvd_r=carry["mvd_r"], qrten=qrten, nrten=nrten, rain_precip=carry["rain_precip"],
                           vtrr=vtrr, vtnr=vtnr)
        if "graupel" in kinds:
            rg, ng, rb = carry["graupel_xr"].unbind(dim=0)
            qgten, ngten, qbten = carry["graupel_xten"].unbind(dim=0)
            vtrg, vtng, _ = carry["graupel_vt"].unbind(dim=0)
            updated.update(l_qg=carry["l_qg"], rg=rg, ng=ng, rb=rb, idx_bg=carry["idx_bg"], ilamg=carry["ilamg"],
                           mvd_g=carry["mvd_g"], qgten=qgten, ngten=ngten, qbten=qbten,
                           graupel_precip=carry["graupel_precip"], vtrg=vtrg, vtng=vtng)
            updated.update({n: carry[n] for n in ("ng1d", "qb1d") if n in carry})
        return updated

    return _sedimentation_substep_tiers(run, columns, kind="_".join(kinds), need=need, bound=bound,
                                        axes={"rain_limit": 1, "graupel_limit": 1})


def _semilagrangian_rain_graupel_substep(
    carry: dict[str, torch.Tensor],
    fixed: dict[str, Any],
    step_index: int | torch.Tensor,
    *,
    kinds: tuple[str, ...],
    fortran: bool,
    dt: float,
    odt: float,
    scalars: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Substep ``step_index`` of rain's and graupel's semi-Lagrangian loops (main.F90:592-690).

    ``carry`` holds the fields a substep updates, ``fixed`` those it only reads (see
    ``_semilagrangian_rain_graupel_tiers``). ``step_index`` enters only through comparisons with
    the substep counts, so a 0-d integer tensor gives the same masks as the int. ``scalars`` names
    0-d entries of ``fixed`` passed with shape [1] (a compiled region guards a 0-d float input by
    reading its value); they are viewed back to 0-d here.
    """
    if scalars:
        fixed = {name: value.reshape(()) if name in scalars else value for name, value in fixed.items()}
    species_counts = tuple(2 if name == "rain" else 3 for name in kinds)
    dz1d, rho, rhof = fixed["dz1d"], fixed["rho"], fixed["rhof"]
    if fortran:
        offsets = tuple(sum(species_counts[:index]) for index in range(len(kinds)))
        xr_all = _pack_species(*(carry[f"{name}_xr"] for name in kinds))
        xr_new, xten_new, precip_new = _semilagrangian_fortran_ordered_species_tensor(
            dz1d=dz1d,
            rho=rho,
            xr=xr_all,
            xten=_pack_species(*(carry[f"{name}_xten"] for name in kinds)),
            vt=_pack_species(*(carry[f"{name}_vt"] for name in kinds)),
            limit_tensor=_species_limit_tensor(fixed["limit_all"], reference=xr_all),
            precip=tuple(carry[f"{name}_precip"] for name in kinds),
            precip_species_index=offsets,
            steps_tensor=fixed["steps_all"],
            dt=float(dt),
            odt=float(odt),
            step_index=step_index,
            zi=fixed["zi"],
        )
        sedimented = {name: (xr_new[offset:offset + count], xten_new[offset:offset + count], precip)
                      for name, offset, count, precip in zip(kinds, offsets, species_counts, precip_new)}
    else:
        sedimented = {
            name: semilagrangian_sedimentation_species_tensor(
                dz1d=dz1d,
                rho=rho,
                xr=carry[f"{name}_xr"],
                xten=carry[f"{name}_xten"],
                vt=carry[f"{name}_vt"],
                steps=fixed[f"{name}_steps"],
                limits=fixed[f"{name}_limit"],
                precip=carry[f"{name}_precip"],
                dt=dt,
                odt=odt,
                step_index=step_index,
                precip_species_index=0,
                geometry=fixed["geometry"][name],
            )
            for name in kinds
        }
    carry = dict(carry)
    if "rain" in kinds:
        rain_xr, rain_xten, rain_vt, rain_precip = (carry[f"rain_{n}"] for n in ("xr", "xten", "vt", "precip"))
        rain_sedimented, rain_sedimented_tendencies, rain_sedimented_precip = sedimented["rain"]
        rain_active_1d = fixed["rain_active_1d"]
        step_active = (rain_active_1d & (fixed["rain_steps"] > step_index))[:, None]
        l_qr, rain_xr, nr1d, ilamr, mvd_r, rain_xten, rain_precip, rain_vt = _run_sedimentation_recurrence(
            "rain_semilagrangian_recurrence_update_step",
            _rain_semilagrangian_recurrence_update_step_kernel,
            rain_xr,
            dz1d,
            fixed["qr1d"],
            fixed["nr1d_initial"],
            rho,
            rhof,
            carry["l_qr"],
            rain_xr,
            carry["nr1d"],
            carry["ilamr"],
            carry["mvd_r"],
            rain_xten,
            rain_precip,
            rain_vt,
            rain_sedimented,
            rain_sedimented_tendencies,
            rain_sedimented_precip if rain_sedimented_precip is not None else rain_precip,
            fixed["rain_levels0"],
            fixed["rain_levels1"],
            step_active,
            rain_active_1d,
            dt,
            odt,
            fixed["rain_power"],
        )
        carry.update(rain_xr=rain_xr, rain_xten=rain_xten, rain_vt=rain_vt, rain_precip=rain_precip, l_qr=l_qr,
                     nr1d=nr1d, ilamr=ilamr, mvd_r=mvd_r)
    if "graupel" in kinds:
        graupel_xr, graupel_xten, graupel_vt, graupel_precip = (
            carry[f"graupel_{n}"] for n in ("xr", "xten", "vt", "precip"))
        step_active, graupel_xr, graupel_xten, graupel_precip, _ = _graupel_semilagrangian_sediment_step(
            graupel_xr,
            graupel_xten,
            graupel_precip,
            *sedimented["graupel"],
            fixed["graupel_steps"],
            fixed["graupel_active_1d"],
            step_index,
        )
        if "ng_template" in fixed and "qb_template" in fixed:
            (l_qg, graupel_xr, idx_bg, ilamg, ng1d, qb1d, mvd_g, graupel_xten, graupel_precip,
             graupel_vt) = _run_sedimentation_recurrence(
                "graupel_semilagrangian_recurrence_optional_update_step",
                _graupel_semilagrangian_recurrence_step_optional_kernel,
                graupel_xr,
                dz1d,
                fixed["qg1d"],
                rho,
                rhof,
                fixed["visco"],
                carry["l_qg"],
                graupel_xr,
                carry["idx_bg"],
                carry["ilamg"],
                carry["ng1d"],
                carry["qb1d"],
                carry["mvd_g"],
                graupel_xten,
                graupel_precip,
                graupel_vt,
                fixed["ng_template"],
                fixed["qb_template"],
                fixed["graupel_am_g_table"],
                fixed["graupel_cgg_table"],
                fixed["graupel_levels0"],
                fixed["graupel_levels1"],
                step_active,
                dt,
                odt,
            )
            carry.update(ng1d=ng1d, qb1d=qb1d)
        else:
            l_qg, graupel_xr, idx_bg, ilamg, mvd_g, graupel_xten, graupel_precip, graupel_vt = (
                _run_sedimentation_recurrence(
                    "graupel_semilagrangian_recurrence_single_update_step",
                    _graupel_semilagrangian_recurrence_step_single_kernel,
                    graupel_xr,
                    dz1d,
                    fixed["qg1d"],
                    rho,
                    rhof,
                    carry["l_qg"],
                    graupel_xr,
                    carry["idx_bg"],
                    carry["ilamg"],
                    carry["mvd_g"],
                    graupel_xten,
                    graupel_precip,
                    graupel_vt,
                    fixed["graupel_cgg_table"],
                    fixed["graupel_levels0"],
                    fixed["graupel_levels1"],
                    step_active,
                    dt,
                    odt,
                )
            )
        carry.update(graupel_xr=graupel_xr, graupel_xten=graupel_xten, graupel_vt=graupel_vt,
                     graupel_precip=graupel_precip, l_qg=l_qg, idx_bg=idx_bg, ilamg=ilamg, mvd_g=mvd_g)
    return carry


# Rain/graupel substeps as one separately compiled region (``set_sedimentation_substep_region``).
# Off by default; apply_tempo turns it on when it compiles TEMPO.
SEDIMENTATION_SUBSTEP_REGION = False
# Input variants the region may compile (tier buffer shapes, grad mode, kinds): three tiers, two
# column-chunk widths, ladder rungs and both grad modes stay well below it.
SEDIMENTATION_SUBSTEP_REGION_RECOMPILE_LIMIT = 64
_SEDIMENTATION_SUBSTEP_REGION: dict[str, Callable[..., dict[str, torch.Tensor]]] = {}


def set_sedimentation_substep_region(enabled: bool) -> bool:
    """Run each traced rain/graupel semi-Lagrangian substep in one reused compiled region; return the previous.

    While ``torch.compile`` traces the Fortran-ordered rain/graupel recurrence, each substep becomes
    one opaque operator (``tempo_torch::semilagrangian_substep``) whose implementation runs
    ``_semilagrangian_rain_graupel_substep`` compiled once per input shape under the exact-numerics
    policy: the enclosing graph stays one graph and holds the substep body once per shape instead of
    once per substep. Eager calls, and calls traced while it is off, keep the substeps in line.
    """
    global SEDIMENTATION_SUBSTEP_REGION
    if type(enabled) is not bool:
        raise TypeError("sedimentation substep region switch requires a bool")
    previous, SEDIMENTATION_SUBSTEP_REGION = SEDIMENTATION_SUBSTEP_REGION, enabled
    return previous


def _semilagrangian_rain_graupel_substeps_in_region(
    carry: dict[str, torch.Tensor],
    fixed: dict[str, Any],
    first: int,
    stop: int,
    *,
    kinds: tuple[str, ...],
    dt: float,
    odt: float,
) -> dict[str, torch.Tensor]:
    """Substeps [first, stop) of a tier, each one ``tempo_torch::semilagrangian_substep`` call."""
    carry_names, fixed_names = tuple(carry), tuple(fixed)
    statics = (",".join(kinds), ",".join(carry_names), ",".join(fixed_names), float(dt), float(odt))
    for step_index in range(first, stop):
        out = torch.ops.tempo_torch.semilagrangian_substep(
            [carry[name] for name in carry_names], [fixed[name] for name in fixed_names], step_index, *statics)
        carry = dict(zip(carry_names, out))
    return carry


def _substep_region_call(
    carry: list[torch.Tensor],
    fixed: list[torch.Tensor],
    step_index: int,
    kinds: str,
    carry_names: str,
    fixed_names: str,
    dt: float,
    odt: float,
) -> list[torch.Tensor]:
    """The compiled substep on named inputs; outputs in ``carry_names`` order. The substep index
    enters as a 0-d tensor, so one compiled variant per input shape serves every substep."""
    if "region" not in _SEDIMENTATION_SUBSTEP_REGION:
        from ufs_physics_common import compile_policy

        compile_policy.install_exact_numerics()
        _SEDIMENTATION_SUBSTEP_REGION["region"] = torch.compile(
            _semilagrangian_rain_graupel_substep, backend=compile_policy._inductor_backend, fullgraph=True,
            dynamic=False)
    names = carry_names.split(",")
    fixed_map = dict(zip(fixed_names.split(","), fixed, strict=True))
    # Dynamo guards a 0-d float input of a compiled region by value (Tensor.item() on every call,
    # a host read; torch 2.8): such entries enter with shape [1] and the substep views them back.
    scalars = tuple(name for name, value in fixed_map.items() if value.dim() == 0)
    fixed_map = {name: value.reshape(1) if name in scalars else value for name, value in fixed_map.items()}
    steps = fixed_map["steps_all"]
    step = torch.full((), step_index, dtype=steps.dtype, device=steps.device)
    with torch_dynamo_config.patch(recompile_limit=SEDIMENTATION_SUBSTEP_REGION_RECOMPILE_LIMIT):
        out = _SEDIMENTATION_SUBSTEP_REGION["region"](dict(zip(names, carry, strict=True)), fixed_map, step,
                                                      kinds=tuple(kinds.split(",")), fortran=True, dt=dt, odt=odt,
                                                      scalars=scalars)
    return [out[name] for name in names]


def _fresh_contiguous(outputs: list[torch.Tensor], inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Operator outputs as new contiguous tensors: an output sharing storage with an input or an
    earlier output is copied (an operator's outputs may not alias), others are made contiguous (the
    fake implementation promises contiguous layouts). Values are unchanged."""
    taken = {t.untyped_storage().data_ptr() for t in inputs if t.numel()}
    fresh = []
    for value in outputs:
        pointer = value.untyped_storage().data_ptr()
        value = value.clone(memory_format=torch.contiguous_format) if pointer in taken else value.contiguous()
        taken.add(value.untyped_storage().data_ptr())
        fresh.append(value)
    return fresh


def _semilagrangian_substep_vjp(carry, fixed, grads, step, *, kinds, dt, odt, scalars):
    """Transform the tensor body before compilation; saved inputs keep their AD history."""
    inputs = (*carry.values(), *fixed.values())
    wanted = tuple(value for value in inputs if value.is_floating_point())

    def body(*floating):
        replacements = iter(floating)
        leaves = [next(replacements) if value.is_floating_point() else value for value in inputs]
        out = _semilagrangian_rain_graupel_substep(
            dict(zip(carry, leaves[:len(carry)], strict=True)),
            dict(zip(fixed, leaves[len(carry):], strict=True)), step,
            kinds=kinds, fortran=True, dt=dt, odt=odt, scalars=scalars)
        return tuple(value for value in out.values() if value.is_floating_point())

    _, pullback = torch.func.vjp(body, *wanted)
    found = iter(pullback(tuple(grad for value, grad in zip(carry.values(), grads, strict=True)
                                if value.is_floating_point())))
    return [next(found) if value.is_floating_point() else value.new_empty((0,)) for value in inputs]


def _substep_region_vjp_call(carry, fixed, grads, step_index, kinds, carry_names, fixed_names, dt, odt):
    from ufs_physics_common import compile_policy

    if "vjp" not in _SEDIMENTATION_SUBSTEP_REGION:
        compile_policy.install_exact_numerics()
        _SEDIMENTATION_SUBSTEP_REGION["vjp"] = torch.compile(
            _semilagrangian_substep_vjp, backend=compile_policy._inductor_vjp_backend, fullgraph=True, dynamic=False)
    names = carry_names.split(",")
    fixed_map = dict(zip(fixed_names.split(","), fixed, strict=True))
    scalars = tuple(name for name, value in fixed_map.items() if value.dim() == 0)
    fixed_map = {name: value.reshape(1) if name in scalars else value for name, value in fixed_map.items()}
    step = torch.full((), step_index, dtype=fixed_map["steps_all"].dtype, device=fixed_map["steps_all"].device)
    with torch_dynamo_config.patch(recompile_limit=SEDIMENTATION_SUBSTEP_REGION_RECOMPILE_LIMIT):
        found = _SEDIMENTATION_SUBSTEP_REGION["vjp"](
            dict(zip(names, carry, strict=True)), fixed_map, grads, step,
            kinds=tuple(kinds.split(",")), dt=dt, odt=odt, scalars=scalars)
    # Scalar normalization changes only the compiled signature, not the operator's gradient layouts.
    return [grad.reshape_as(value) if value.is_floating_point() else grad
            for value, grad in zip((*carry, *fixed), found, strict=True)]


@torch.library.custom_op("tempo_torch::semilagrangian_substep", mutates_args=())
def _semilagrangian_substep_op(
    carry: list[torch.Tensor],
    fixed: list[torch.Tensor],
    step_index: int,
    kinds: str,
    carry_names: str,
    fixed_names: str,
    dt: float,
    odt: float,
) -> list[torch.Tensor]:
    """One rain/graupel semi-Lagrangian substep in its compiled region (``set_sedimentation_substep_region``).

    No autograd graph is recorded here (gradients come from the backward operator), so the region
    runs under no_grad and compiles an inference graph."""
    with torch.no_grad():
        out = _substep_region_call(carry, fixed, step_index, kinds, carry_names, fixed_names, dt, odt)
    for value, reference in zip(out, carry, strict=True):
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise RuntimeError("semi-Lagrangian substep changed a carried field's shape or dtype")
    return _fresh_contiguous(out, [*carry, *fixed])


@_semilagrangian_substep_op.register_fake
def _semilagrangian_substep_fake(carry, fixed, step_index, kinds, carry_names, fixed_names, dt, odt):
    return [torch.empty(value.shape, dtype=value.dtype, device=value.device) for value in carry]


def _autograd_dispatch() -> Any:
    """Autograd dispatch inside an operator's implementation.

    A custom operator's implementation runs with the autograd dispatch keys excluded, so
    ``torch.enable_grad()`` alone records nothing there (every gradient came back zero). As
    torch's higher-order operators do (``_ForceDispatchKeyGuard``), this restores those keys for
    the backward operator's recompute and VJP; other excluded keys stay excluded.
    """
    keys = torch._C.DispatchKey
    exclude = torch._C._dispatch_tls_local_exclude_set()
    for key in (keys.AutogradFunctionality, keys.AutogradOther, keys.AutogradNestedTensor):
        exclude = exclude.remove(key)
    return torch._C._ForceDispatchKeyGuard(torch._C._dispatch_tls_local_include_set(), exclude)


@torch.library.custom_op("tempo_torch::semilagrangian_substep_backward", mutates_args=())
def _semilagrangian_substep_backward_op(
    carry: list[torch.Tensor],
    fixed: list[torch.Tensor],
    grads: list[torch.Tensor],
    step_index: int,
    kinds: str,
    carry_names: str,
    fixed_names: str,
    dt: float,
    odt: float,
) -> list[torch.Tensor]:
    """Input gradients of one substep: its region recomputed under autograd (a checkpoint), then the
    vector-Jacobian product. Floating inputs get a gradient (zeros where none flows); others an empty
    tensor."""
    with _autograd_dispatch(), torch.no_grad():
        result = _substep_region_vjp_call(carry, fixed, grads, step_index, kinds, carry_names, fixed_names, dt, odt)
    return _fresh_contiguous(result, [*carry, *fixed, *grads])


@_semilagrangian_substep_backward_op.register_fake
def _semilagrangian_substep_backward_fake(carry, fixed, grads, step_index, kinds, carry_names, fixed_names, dt,
                                          odt):
    return [torch.empty(value.shape if value.is_floating_point() else (0,), dtype=value.dtype, device=value.device)
            for value in (*carry, *fixed)]


def _semilagrangian_substep_setup_context(ctx, inputs, output):
    carry, fixed, step_index, *statics = inputs
    ctx.counts = (len(carry), len(fixed))
    ctx.statics = (step_index, *statics)
    ctx.save_for_backward(*carry, *fixed)


def _semilagrangian_substep_backward(ctx, grads):
    ncarry, nfixed = ctx.counts
    saved = ctx.saved_tensors
    carry, fixed = list(saved[:ncarry]), list(saved[ncarry:])
    grads = [torch.zeros_like(value, memory_format=torch.contiguous_format) if grad is None else grad
             for value, grad in zip(carry, grads, strict=True)]
    found = torch.ops.tempo_torch.semilagrangian_substep_backward(carry, fixed, grads, *ctx.statics)
    found = [grad if value.is_floating_point() else None for value, grad in zip((*carry, *fixed), found, strict=True)]
    return found[:ncarry], found[ncarry:], None, None, None, None, None, None


_semilagrangian_substep_op.register_autograd(_semilagrangian_substep_backward,
                                             setup_context=_semilagrangian_substep_setup_context)


def _graupel_semilagrangian_sediment_step(
    graupel_xr: torch.Tensor,
    graupel_xten: torch.Tensor,
    graupel_precip: torch.Tensor,
    sedimented: torch.Tensor,
    sedimented_tendencies: torch.Tensor,
    sedimented_precip: torch.Tensor | None,
    substeps_sedi: torch.Tensor,
    branch_active_1d: torch.Tensor,
    step_index: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Graupel's sedimented species where this substep is active (the semi-Lagrangian call is shared)."""
    step_active_1d = branch_active_1d & (substeps_sedi > step_index)
    step_active = step_active_1d[:, None]
    rg, ng, rb = graupel_xr.unbind(dim=0)
    qgten, ngten, qbten = graupel_xten.unbind(dim=0)
    rg_next, ng_next, rb_next = sedimented.unbind(dim=0)
    qgten_next, ngten_next, qbten_next = sedimented_tendencies.unbind(dim=0)
    graupel_precip_next = sedimented_precip if sedimented_precip is not None else graupel_precip
    rg, ng, rb = _masked_merge_components(step_active, (rg, ng, rb), (rg_next, ng_next, rb_next))
    qgten, ngten, qbten = _masked_merge_components(step_active, (qgten, ngten, qbten), (qgten_next, ngten_next, qbten_next))
    graupel_precip = torch.where(step_active_1d, graupel_precip_next, graupel_precip)
    return step_active, _species_tensor(rg, ng, rb), _species_tensor(qgten, ngten, qbten), graupel_precip, step_active_1d


def _graupel_semilagrangian_recurrence_step_single_kernel(
    dz1d: torch.Tensor,
    qg1d: torch.Tensor,
    rho: torch.Tensor,
    rhof: torch.Tensor,
    l_qg: torch.Tensor,
    graupel_xr: torch.Tensor,
    idx_bg: torch.Tensor,
    ilamg: torch.Tensor,
    mvd_g: torch.Tensor,
    graupel_xten: torch.Tensor,
    graupel_precip: torch.Tensor,
    graupel_vt: torch.Tensor,
    graupel_cgg_table: torch.Tensor,
    graupel_zero_based_levels: torch.Tensor,
    graupel_one_based_levels: torch.Tensor,
    step_active: torch.Tensor,
    dt: float,
    odt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vtrg, vtng, _ = graupel_vt.unbind(dim=0)
    rg, ng, rb = graupel_xr.unbind(dim=0)
    qgten, ngten, qbten = graupel_xten.unbind(dim=0)
    graupel_values = _graupel_check_and_update_single_kernel(
        rho,
        qg1d,
        qgten,
        ngten,
        qbten,
        dt,
        odt,
    )
    (
        l_qg_next,
        _,
        rg_check,
        ng_check,
        rb_check,
        idx_check,
        qgten_check,
        ngten_check,
        qbten_check,
        ilamg_check,
        mvd_g_check,
    ) = graupel_values
    l_qg = torch.where(step_active, l_qg_next, l_qg)
    idx_bg = torch.where(step_active, idx_check, idx_bg)
    rg, ng, rb, ilamg, mvd_g, qgten, ngten, qbten = _masked_merge_components(step_active, (rg, ng, rb, ilamg, mvd_g, qgten, ngten, qbten), (rg_check, ng_check, rb_check, ilamg_check, mvd_g_check, qgten_check, ngten_check, qbten_check))
    vtrg_next, vtng_next, _, _ = _graupel_fallspeed_single_zero_tail_kernel(
        rhof,
        rg,
        rb,
        idx_bg,
        ilamg,
        dz1d,
        graupel_cgg_table,
        graupel_zero_based_levels,
        graupel_one_based_levels,
        dt,
        False,
    )
    vtrg, vtng = _masked_merge_components(step_active, (vtrg, vtng), (vtrg_next, vtng_next))
    return (l_qg, _species_tensor(rg, ng, rb), idx_bg, ilamg, mvd_g, _species_tensor(qgten, ngten, qbten), graupel_precip, _species_tensor(vtrg, vtng, vtrg))


def _graupel_semilagrangian_recurrence_step_optional_kernel(
    dz1d: torch.Tensor,
    qg1d: torch.Tensor,
    rho: torch.Tensor,
    rhof: torch.Tensor,
    visco: torch.Tensor,
    l_qg: torch.Tensor,
    graupel_xr: torch.Tensor,
    idx_bg: torch.Tensor,
    ilamg: torch.Tensor,
    ng1d: torch.Tensor,
    qb1d: torch.Tensor,
    mvd_g: torch.Tensor,
    graupel_xten: torch.Tensor,
    graupel_precip: torch.Tensor,
    graupel_vt: torch.Tensor,
    ng_template: torch.Tensor,
    qb_template: torch.Tensor,
    graupel_am_g_table: torch.Tensor,
    graupel_cgg_table: torch.Tensor,
    graupel_zero_based_levels: torch.Tensor,
    graupel_one_based_levels: torch.Tensor,
    step_active: torch.Tensor,
    dt: float,
    odt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vtrg, vtng, _ = graupel_vt.unbind(dim=0)
    rg, ng, rb = graupel_xr.unbind(dim=0)
    qgten, ngten, qbten = graupel_xten.unbind(dim=0)
    graupel_values = _graupel_check_and_update_optional_kernel(
        rho,
        qg1d,
        qgten,
        ngten,
        qbten,
        ng_template,
        qb_template,
        graupel_am_g_table,
        dt,
        odt,
    )
    (
        l_qg_next,
        _,
        ng1d_next,
        qb1d_next,
        rg_check,
        ng_check,
        rb_check,
        idx_check,
        qgten_check,
        ngten_check,
        qbten_check,
        ilamg_check,
        mvd_g_check,
    ) = graupel_values
    l_qg = torch.where(step_active, l_qg_next, l_qg)
    idx_bg = torch.where(step_active, idx_check, idx_bg)
    rg, ng, rb, ilamg, mvd_g, qgten, ngten, qbten = _masked_merge_components(step_active, (rg, ng, rb, ilamg, mvd_g, qgten, ngten, qbten), (rg_check, ng_check, rb_check, ilamg_check, mvd_g_check, qgten_check, ngten_check, qbten_check))
    ng1d = _masked_merge_components(step_active, (ng1d,), (ng1d_next,))[0]
    qb1d = _masked_merge_components(step_active, (qb1d,), (qb1d_next,))[0]
    vtrg_next, vtng_next, _, _ = _graupel_fallspeed_optional_zero_tail_kernel(
        rho,
        visco,
        qb_template,
        rhof,
        rg,
        rb,
        idx_bg,
        ilamg,
        dz1d,
        graupel_cgg_table,
        graupel_zero_based_levels,
        graupel_one_based_levels,
        dt,
        False,
    )
    vtrg, vtng = _masked_merge_components(step_active, (vtrg, vtng), (vtrg_next, vtng_next))
    return (l_qg, _species_tensor(rg, ng, rb), idx_bg, ilamg, ng1d, qb1d, mvd_g, _species_tensor(qgten, ngten, qbten), graupel_precip, _species_tensor(vtrg, vtng, vtrg))


def _graupel_semilagrangian_recurrence(dz1d: torch.Tensor, qg1d: torch.Tensor, rho: torch.Tensor, rhof: torch.Tensor, visco: torch.Tensor, l_qg: torch.Tensor, rg: torch.Tensor, ng: torch.Tensor, rb: torch.Tensor, idx_bg: torch.Tensor, ilamg: torch.Tensor, mvd_g: torch.Tensor, qgten: torch.Tensor, ngten: torch.Tensor, qbten: torch.Tensor, graupel_precip: torch.Tensor, vtrg: torch.Tensor, vtng: torch.Tensor, ng1d: torch.Tensor | None, qb1d: torch.Tensor | None, ng_template: torch.Tensor | None, qb_template: torch.Tensor | None, graupel_limit_tensor: torch.Tensor, graupel_am_g_table: torch.Tensor, graupel_cgg_table: torch.Tensor, graupel_zero_based_levels: torch.Tensor, graupel_one_based_levels: torch.Tensor, substeps_sedi: torch.Tensor, branch_active: torch.Tensor, sedimentation_substeps: int, dt: float, odt: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Graupel alone through the shared semi-Lagrangian recurrence (segment P4)."""
    graupel = _graupel_semilagrangian_member(qg1d, visco, l_qg, rg, ng, rb, idx_bg, ilamg, mvd_g, qgten, ngten,
                                             qbten, graupel_precip, vtrg, vtng, ng1d, qb1d, ng_template,
                                             qb_template, graupel_limit_tensor, graupel_am_g_table,
                                             graupel_cgg_table, graupel_zero_based_levels,
                                             graupel_one_based_levels, substeps_sedi, branch_active)
    out = _semilagrangian_rain_graupel_recurrence(None, graupel, dz1d=dz1d, rho=rho, rhof=rhof,
                                                  bound=sedimentation_substeps, dt=dt, odt=odt)
    return tuple(out[n] if n in out else None for n in _GRAUPEL_SEMILAGRANGIAN_OUTPUTS)


def _snow_sedimentation_recurrence_step_kernel(rs: torch.Tensor, vtrs: torch.Tensor, dz1d: torch.Tensor, rho: torch.Tensor, qsten: torch.Tensor, snow_liquid_equiv_precip: torch.Tensor, snow_limit_tensor: torch.Tensor, snow_steps: torch.Tensor, snow_ktop: torch.Tensor, active_step: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rs, qsten, precip_next = _sedimentation_tensor_active_step(rs, vtrs, dz1d, rho, qsten, snow_limit_tensor, snow_steps, snow_ktop, active_step, snow_liquid_equiv_precip, dt, None)
    return rs, qsten, precip_next if precip_next is not None else snow_liquid_equiv_precip


def _snow_sedimentation_recurrence(rs: torch.Tensor, vtrs: torch.Tensor, dz1d: torch.Tensor, rho: torch.Tensor, qsten: torch.Tensor, snow_liquid_equiv_precip: torch.Tensor, snow_limit_tensor: torch.Tensor, snow_steps: torch.Tensor, snow_ktop: torch.Tensor, snow_sedimentation_substeps: int, dt: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    columns = dict(rs=rs, vtrs=vtrs, dz1d=dz1d, rho=rho, qsten=qsten, precip=snow_liquid_equiv_precip,
                   steps=snow_steps, ktop=snow_ktop)

    def run(c: dict[str, torch.Tensor], first: int, stop: int) -> dict[str, torch.Tensor]:
        rs, qsten, snow_liquid_equiv_precip = c["rs"], c["qsten"], c["precip"]
        snow_steps = c["steps"]
        for step_index in range(first, stop):
            active_step = snow_steps > step_index
            step_args = (
                rs, c["vtrs"], c["dz1d"], c["rho"], qsten, snow_liquid_equiv_precip,
                snow_limit_tensor, snow_steps, c["ktop"], active_step, dt,
            )
            if SEDIMENTATION_RECURRENCE_EXECUTION_MODE == "fixed_step":
                rs, qsten, snow_liquid_equiv_precip = _run_sedimentation_recurrence(
                    "snow_sedimentation_recurrence_step",
                    _snow_sedimentation_recurrence_step_kernel,
                    rs,
                    *step_args,
                )
            else:
                rs, qsten, snow_liquid_equiv_precip = _snow_sedimentation_recurrence_step_kernel(
                    *step_args
                )
        return dict(rs=rs, qsten=qsten, precip=snow_liquid_equiv_precip)

    out = _sedimentation_substep_tiers(run, columns, kind="snow", need=lambda c, first: c["steps"] > first,
                                       bound=snow_sedimentation_substeps)
    return (out["rs"], out["qsten"], out["precip"])


def run_p2(
    state: Mapping[str, torch.Tensor] | TorchSegmentOResult,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = True,
    max_sedimentation_substeps: int | None = None,
    copy: bool = False,
) -> TorchSegmentP2Result:
    if max_sedimentation_substeps is not None and max_sedimentation_substeps < 1:
        raise ValueError("Segment P2 max_sedimentation_substeps must be positive")
    o_result = _segment_o_result(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
    )
    input_state = o_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = o_result.workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment P2 dt must be nonzero")
    sedimentation_substeps = _default_sedimentation_substeps(
        dt=resolved_dt,
        semi_sedi_flag=semi_sedi_flag,
        override=max_sedimentation_substeps,
    )
    odt = 1.0 / resolved_dt
    p2 = _p2_rain_setup(o_result, input_state, copy=copy, semi_sedi_flag=semi_sedi_flag, dt=resolved_dt)
    out = (
        _semilagrangian_rain_graupel_recurrence(
            p2["member"], None, dz1d=input_state["dz"], rho=o_result.workspace.rho,
            rhof=o_result.workspace.thermo["rhof"], bound=sedimentation_substeps, dt=resolved_dt, odt=odt)
        if semi_sedi_flag else None
    )
    return _p2_rain_finish(o_result, result_state, p2, out, sedimentation_substeps=sedimentation_substeps,
                           semi_sedi_flag=semi_sedi_flag, dt=resolved_dt, odt=odt)


def _p2_rain_setup(
    o_result: TorchSegmentOResult,
    input_state: Mapping[str, torch.Tensor],
    *,
    copy: bool,
    semi_sedi_flag: bool,
    dt: float,
) -> dict[str, Any]:
    """Segment P2 before its recurrence: rain fall speeds, substep counts and recurrence inputs."""
    tend = _tensor_mapping(o_result.workspace.tend, copy=copy)
    tendencies = _tensor_mapping(o_result.workspace.tendencies, copy=copy)
    diagnostics = _tensor_mapping(o_result.workspace.diagnostics, copy=copy)
    fallspeeds = _tensor_mapping(o_result.workspace.fallspeeds, copy=copy)
    l_qr = o_result.workspace.l_qr
    rr = o_result.workspace.rr
    nr = o_result.workspace.nr
    nr1d = o_result.workspace.nr1d
    ilamr = o_result.workspace.ilamr
    mvd_r = o_result.workspace.mvd_r
    vtrr = fallspeeds["vtrr"]
    vtnr = fallspeeds["vtnr"]
    substeps_sedi = torch.ones((o_result.ncol,), dtype=torch.int64, device=rr.device)
    ktop_sedi = torch.ones((o_result.ncol,), dtype=torch.int64, device=rr.device)
    branch_active = _column_activity(l_qr)
    member = None

    if semi_sedi_flag:
        vtrr, vtnr, substeps_initial, ktop_sedi = rain_fallspeed_tensor(
            rhof=o_result.workspace.thermo["rhof"],
            rr=rr,
            ilamr=ilamr,
            dz1d=input_state["dz"],
            vt=vtrr,
            vtn=vtnr,
            dt=dt,
        )
        substeps_sedi = torch.clamp_min(
            torch.trunc(substeps_initial.to(dtype=rr.dtype) / _scalar(constants.SEMI_SEDI_FACTOR, rr)).to(torch.int64)
            + 1,
            1,
        )
        rain_xr_reference = _species_tensor(rr, nr)
        member = _rain_semilagrangian_member(
            input_state["qr"],
            input_state["nr"],
            l_qr,
            rr,
            nr,
            nr1d,
            ilamr,
            mvd_r,
            tendencies["qrten"],
            tendencies["nrten"],
            diagnostics["rain_precip"],
            vtrr,
            vtnr,
            _species_limit_tensor((constants.R1, constants.R2), reference=rain_xr_reference),
            _int_arange(0, rr.shape[1], rr)[None, :],
            _int_arange(1, rr.shape[1] + 1, rr)[None, :],
            substeps_sedi,
            branch_active,
            _tensor(constants.BM_R, rr),
        )
    return dict(tend=tend, tendencies=tendencies, diagnostics=diagnostics, fallspeeds=fallspeeds, l_qr=l_qr, rr=rr,
                nr=nr, nr1d=nr1d, ilamr=ilamr, mvd_r=mvd_r, vtrr=vtrr, vtnr=vtnr, substeps_sedi=substeps_sedi,
                ktop_sedi=ktop_sedi, branch_active=branch_active, member=member)


def _p2_rain_finish(
    o_result: TorchSegmentOResult,
    result_state: dict[str, torch.Tensor],
    p2: dict[str, Any],
    out: Mapping[str, torch.Tensor] | None,
    *,
    sedimentation_substeps: int,
    semi_sedi_flag: bool,
    dt: float,
    odt: float,
) -> TorchSegmentP2Result:
    """Segment P2 after its recurrence (``out``: the recurrence's rain outputs, None without it)."""
    tendencies, diagnostics = p2["tendencies"], p2["diagnostics"]
    l_qr, rr, nr, nr1d, ilamr, mvd_r, vtrr, vtnr = (
        p2[n] for n in ("l_qr", "rr", "nr", "nr1d", "ilamr", "mvd_r", "vtrr", "vtnr"))
    if out is not None:
        (l_qr, rr, nr, nr1d, ilamr, mvd_r, tendencies["qrten"], tendencies["nrten"], diagnostics["rain_precip"],
         vtrr, vtnr) = (out[n] for n in _RAIN_SEMILAGRANGIAN_OUTPUTS)
        _record_sedimentation_substep_overflow(diagnostics, p2["substeps_sedi"], sedimentation_substeps,
                                               p2["branch_active"], region=SEDIMENTATION_BOUND_RAIN_GRAUPEL)

    fallspeeds = {**p2["fallspeeds"], "vtrr": vtrr, "vtnr": vtnr}
    tend_work, tend_views = _contiguous_tend_work_tensor(p2["tend"])
    workspace = TorchSegmentP2Workspace(
        base=o_result.workspace,
        tend_work=tend_work,
        tend=tend_views,
        tendencies=tendencies,
        diagnostics=diagnostics,
        fallspeeds=fallspeeds,
        l_qr=l_qr,
        rr=rr,
        nr=nr,
        nr1d=nr1d,
        ilamr=ilamr,
        mvd_r=mvd_r,
        vtrr=vtrr,
        vtnr=vtnr,
        substeps_sedi=p2["substeps_sedi"],
        ktop_sedi=p2["ktop_sedi"],
        rain_precip=diagnostics["rain_precip"],
        semi_sedi_flag=bool(semi_sedi_flag),
        branch_active=p2["branch_active"],
        dt=dt,
        odt=odt,
    )
    return TorchSegmentP2Result(result_state, workspace, o_result.ncol, o_result.nz)


def run_p3(
    state: Mapping[str, torch.Tensor] | TorchSegmentP1Result,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = False,
    max_sedimentation_substeps: int | None = None,
    copy: bool = False,
) -> TorchSegmentP3Result:
    if max_sedimentation_substeps is not None and max_sedimentation_substeps < 1:
        raise ValueError("Segment P3 max_sedimentation_substeps must be positive")
    p1_result = _segment_p1_result(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
    )
    input_state = p1_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = p1_result.workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment P3 dt must be nonzero")
    sedimentation_substeps = _default_sedimentation_substeps(
        dt=resolved_dt,
        semi_sedi_flag=semi_sedi_flag,
        override=max_sedimentation_substeps,
    )
    odt = 1.0 / resolved_dt

    o_workspace = p1_result.workspace.base
    tend = _tensor_mapping(p1_result.workspace.tend, copy=copy)
    tendencies = _tensor_mapping(p1_result.workspace.tendencies, copy=copy)
    diagnostics = _tensor_mapping(p1_result.workspace.diagnostics, copy=copy)
    fallspeeds = _tensor_mapping(p1_result.workspace.fallspeeds, copy=copy)
    rg = o_workspace.rg
    ng = o_workspace.ng
    rb = o_workspace.rb
    vtrg = fallspeeds["vtrg"]
    vtng = fallspeeds["vtng"]
    substeps_sedi = torch.ones((p1_result.ncol,), dtype=torch.int64, device=rg.device)
    ktop_sedi = torch.ones((p1_result.ncol,), dtype=torch.int64, device=rg.device)
    branch_active = _column_activity(o_workspace.l_qg)

    if not semi_sedi_flag:
        vtrg, vtng, substeps_sedi, ktop_sedi = graupel_fallspeed_tensor(
            rhof=o_workspace.thermo["rhof"],
            rho=o_workspace.rho,
            visco=o_workspace.thermo["visco"],
            rg=rg,
            rb=rb,
            qb1d=o_workspace.qb1d,
            idx=o_workspace.idx_bg,
            ilamg=o_workspace.ilamg,
            dz1d=input_state["dz"],
            vt=vtrg,
            vtn=vtng,
            dt=resolved_dt,
        )
        volume_limit = constants.METERS3_TO_LITERS * constants.R1 / constants.RHO_G[constants.NRHG - 1]
        graupel_xr = _species_tensor(rg, ng, rb)
        graupel_vt = _species_tensor(vtrg[:, : rg.shape[1]], vtng[:, : ng.shape[1]], vtrg[:, : rb.shape[1]])
        graupel_xten = _species_tensor(tendencies["qgten"], tendencies["ngten"], tendencies["qbten"])
        graupel_precip = diagnostics["graupel_liquid_equiv_precip"]
        graupel_xr, graupel_xten, graupel_precip = _run_sedimentation_recurrence(
            "eulerian_species_recurrence",
            _eulerian_sedimentation_species_recurrence,
            graupel_xr,
            graupel_xr,
            graupel_vt,
            input_state["dz"],
            o_workspace.rho,
            graupel_xten,
            _species_limit_tensor((constants.R1, constants.R2, volume_limit), reference=graupel_xr),
            _sedimentation_steps_tensor(substeps_sedi, p1_result.ncol, graupel_xr),
            _sedimentation_ktop_tensor(ktop_sedi, p1_result.ncol, rg.shape[1], graupel_xr),
            graupel_precip,
            sedimentation_substeps,
            resolved_dt,
            0,
        )
        rg, ng, rb = graupel_xr.unbind(dim=0)
        tendencies["qgten"], tendencies["ngten"], tendencies["qbten"] = graupel_xten.unbind(dim=0)
        diagnostics["graupel_liquid_equiv_precip"] = graupel_precip
        _record_sedimentation_substep_overflow(diagnostics, substeps_sedi, sedimentation_substeps, branch_active,
                                               region=SEDIMENTATION_BOUND_RAIN_GRAUPEL)

    fallspeeds = {**fallspeeds, "vtrg": vtrg, "vtng": vtng}
    tend_work, tend_views = _contiguous_tend_work_tensor(tend)
    workspace = TorchSegmentP3Workspace(
        base=p1_result.workspace,
        tend_work=tend_work,
        tend=tend_views,
        tendencies=tendencies,
        diagnostics=diagnostics,
        fallspeeds=fallspeeds,
        rg=rg,
        ng=ng,
        rb=rb,
        vtrg=vtrg,
        vtng=vtng,
        substeps_sedi=substeps_sedi,
        ktop_sedi=ktop_sedi,
        graupel_liquid_equiv_precip=diagnostics["graupel_liquid_equiv_precip"],
        semi_sedi_flag=bool(semi_sedi_flag),
        branch_active=branch_active,
        dt=resolved_dt,
        odt=odt,
    )
    return TorchSegmentP3Result(result_state, workspace, p1_result.ncol, p1_result.nz)


def run_p4(
    state: Mapping[str, torch.Tensor] | TorchSegmentP2Result,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = True,
    max_sedimentation_substeps: int | None = None,
    copy: bool = False,
) -> TorchSegmentP4Result:
    if max_sedimentation_substeps is not None and max_sedimentation_substeps < 1:
        raise ValueError("Segment P4 max_sedimentation_substeps must be positive")
    p2_result = _segment_p2_result(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
    )
    input_state = p2_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = p2_result.workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment P4 dt must be nonzero")
    sedimentation_substeps = _default_sedimentation_substeps(
        dt=resolved_dt,
        semi_sedi_flag=semi_sedi_flag,
        override=max_sedimentation_substeps,
    )
    odt = 1.0 / resolved_dt
    o_workspace = p2_result.workspace.base
    p4 = _p4_graupel_setup(o_workspace, _p_mappings(p2_result.workspace), input_state, copy=copy,
                           semi_sedi_flag=semi_sedi_flag, dt=resolved_dt, ncol=p2_result.ncol)
    out = (
        _semilagrangian_rain_graupel_recurrence(
            None, p4["member"], dz1d=input_state["dz"], rho=o_workspace.rho, rhof=o_workspace.thermo["rhof"],
            bound=sedimentation_substeps, dt=resolved_dt, odt=odt)
        if semi_sedi_flag else None
    )
    return _p4_graupel_finish(p2_result, result_state, p4, out, sedimentation_substeps=sedimentation_substeps,
                              semi_sedi_flag=semi_sedi_flag, dt=resolved_dt, odt=odt)


def run_p2_p4(
    state: Mapping[str, torch.Tensor] | TorchSegmentOResult,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    max_sedimentation_substeps: int | None = None,
    copy: bool = False,
) -> TorchSegmentP4Result:
    """Segments P2 and P4 (semi-Lagrangian rain, then graupel) with one shared substep loop.

    The production path's form of ``run_p4(run_p2(...))``: the same P4 result, with rain and
    graupel sedimenting in one ``_semilagrangian_rain_graupel_recurrence`` (the source's two
    loops share no state, main.F90:592-690).
    """
    if max_sedimentation_substeps is not None and max_sedimentation_substeps < 1:
        raise ValueError("Segment P2/P4 max_sedimentation_substeps must be positive")
    o_result = _segment_o_result(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
    )
    input_state = o_result.state
    p2_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = o_result.workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment P2/P4 dt must be nonzero")
    sedimentation_substeps = _default_sedimentation_substeps(
        dt=resolved_dt,
        semi_sedi_flag=True,
        override=max_sedimentation_substeps,
    )
    odt = 1.0 / resolved_dt
    p2 = _p2_rain_setup(o_result, input_state, copy=copy, semi_sedi_flag=True, dt=resolved_dt)
    # Graupel reads none of P2's rain updates: its setup reads the P2 mappings before them.
    p4 = _p4_graupel_setup(o_result.workspace, {name: p2[name] for name in _P_MAPPINGS}, input_state, copy=copy,
                           semi_sedi_flag=True, dt=resolved_dt, ncol=o_result.ncol)
    out = _semilagrangian_rain_graupel_recurrence(
        p2["member"], p4["member"], dz1d=input_state["dz"], rho=o_result.workspace.rho,
        rhof=o_result.workspace.thermo["rhof"], bound=sedimentation_substeps, dt=resolved_dt, odt=odt)
    p2_result = _p2_rain_finish(o_result, p2_state, p2, out, sedimentation_substeps=sedimentation_substeps,
                                semi_sedi_flag=True, dt=resolved_dt, odt=odt)
    # P4's mappings start from P2's (with its rain updates), as run_p4 builds them; rain updated
    # none of the graupel entries its setup read.
    p4 = {**p4, **{name: _tensor_mapping(value, copy=copy)
                   for name, value in _p_mappings(p2_result.workspace).items()}}
    p4_state = {name: _copy_tensor(values) for name, values in p2_result.state.items()} if copy else dict(p2_result.state)
    return _p4_graupel_finish(p2_result, p4_state, p4, out, sedimentation_substeps=sedimentation_substeps,
                              semi_sedi_flag=True, dt=resolved_dt, odt=odt)


_P_MAPPINGS = ("tend", "tendencies", "diagnostics", "fallspeeds")


def _p_mappings(workspace: Any) -> dict[str, Mapping[str, torch.Tensor]]:
    return {name: getattr(workspace, name) for name in _P_MAPPINGS}


def _p4_graupel_setup(
    o_workspace: TorchSegmentOWorkspace,
    p2_mappings: Mapping[str, Mapping[str, torch.Tensor]],
    input_state: Mapping[str, torch.Tensor],
    *,
    copy: bool,
    semi_sedi_flag: bool,
    dt: float,
    ncol: int,
) -> dict[str, Any]:
    """Segment P4 before its recurrence: graupel fall speeds, substep counts and recurrence inputs."""
    tend = _tensor_mapping(p2_mappings["tend"], copy=copy)
    tendencies = _tensor_mapping(p2_mappings["tendencies"], copy=copy)
    diagnostics = _tensor_mapping(p2_mappings["diagnostics"], copy=copy)
    fallspeeds = _tensor_mapping(p2_mappings["fallspeeds"], copy=copy)
    l_qg = o_workspace.l_qg
    rg = o_workspace.rg
    ng = o_workspace.ng
    rb = o_workspace.rb
    idx_bg = o_workspace.idx_bg
    ilamg = o_workspace.ilamg
    mvd_g = o_workspace.mvd_g
    ng_template = _optional_input_profile_tensor(input_state, ("ng1d", "ng"), o_workspace.ng1d)
    qb_template = _optional_input_profile_tensor(input_state, ("qb1d", "qb"), o_workspace.qb1d)
    ng1d = ng_template
    qb1d = qb_template
    vtrg = fallspeeds["vtrg"]
    vtng = fallspeeds["vtng"]
    substeps_sedi = torch.ones((ncol,), dtype=torch.int64, device=rg.device)
    ktop_sedi = torch.ones((ncol,), dtype=torch.int64, device=rg.device)
    branch_active = _column_activity(l_qg)
    member = None

    if semi_sedi_flag:
        vtrg, vtng, substeps_initial, ktop_sedi = graupel_fallspeed_tensor(
            rhof=o_workspace.thermo["rhof"],
            rho=o_workspace.rho,
            visco=o_workspace.thermo["visco"],
            rg=rg,
            rb=rb,
            qb1d=qb_template,
            idx=idx_bg,
            ilamg=ilamg,
            dz1d=input_state["dz"],
            vt=vtrg,
            vtn=vtng,
            dt=dt,
        )
        substeps_sedi = torch.clamp_min(
            torch.trunc(substeps_initial.to(dtype=rg.dtype) / _scalar(constants.SEMI_SEDI_FACTOR, rg)).to(torch.int64)
            + 1,
            1,
        )
        volume_limit = constants.METERS3_TO_LITERS * constants.R1 / constants.RHO_G[constants.NRHG - 1]
        graupel_xr_reference = _species_tensor(rg, ng, rb)
        member = _graupel_semilagrangian_member(
            input_state["qg"],
            o_workspace.thermo["visco"],
            l_qg,
            rg,
            ng,
            rb,
            idx_bg,
            ilamg,
            mvd_g,
            tendencies["qgten"],
            tendencies["ngten"],
            tendencies["qbten"],
            diagnostics["graupel_liquid_equiv_precip"],
            vtrg,
            vtng,
            ng1d,
            qb1d,
            ng_template,
            qb_template,
            _species_limit_tensor((constants.R1, constants.R2, volume_limit), reference=graupel_xr_reference),
            _tensor(_STATIC_GRAUPEL_AM_G, rg),
            _tensor(_STATIC_GRAUPEL_CGG, rg),
            _int_arange(0, rg.shape[1], rg)[None, :],
            _int_arange(1, rg.shape[1] + 1, rg)[None, :],
            substeps_sedi,
            branch_active,
        )
    return dict(tend=tend, tendencies=tendencies, diagnostics=diagnostics, fallspeeds=fallspeeds, l_qg=l_qg, rg=rg,
                ng=ng, rb=rb, idx_bg=idx_bg, ilamg=ilamg, mvd_g=mvd_g, ng1d=ng1d, qb1d=qb1d, vtrg=vtrg, vtng=vtng,
                substeps_sedi=substeps_sedi, ktop_sedi=ktop_sedi, branch_active=branch_active, member=member)


def _p4_graupel_finish(
    p2_result: TorchSegmentP2Result,
    result_state: dict[str, torch.Tensor],
    p4: dict[str, Any],
    out: Mapping[str, torch.Tensor] | None,
    *,
    sedimentation_substeps: int,
    semi_sedi_flag: bool,
    dt: float,
    odt: float,
) -> TorchSegmentP4Result:
    """Segment P4 after its recurrence (``out``: the recurrence's graupel outputs, None without it)."""
    tendencies, diagnostics = p4["tendencies"], p4["diagnostics"]
    l_qg, rg, ng, rb, idx_bg, ilamg, ng1d, qb1d, mvd_g, vtrg, vtng = (
        p4[n] for n in ("l_qg", "rg", "ng", "rb", "idx_bg", "ilamg", "ng1d", "qb1d", "mvd_g", "vtrg", "vtng"))
    if out is not None:
        (l_qg, rg, ng, rb, idx_bg, ilamg, ng1d, qb1d, mvd_g, tendencies["qgten"], tendencies["ngten"],
         tendencies["qbten"], diagnostics["graupel_liquid_equiv_precip"], vtrg, vtng) = (
            out[n] if n in out else None for n in _GRAUPEL_SEMILAGRANGIAN_OUTPUTS)
        _record_sedimentation_substep_overflow(diagnostics, p4["substeps_sedi"], sedimentation_substeps,
                                               p4["branch_active"], region=SEDIMENTATION_BOUND_RAIN_GRAUPEL)

    fallspeeds = {**p4["fallspeeds"], "vtrg": vtrg, "vtng": vtng}
    tend_work, tend_views = _contiguous_tend_work_tensor(p4["tend"])
    workspace = TorchSegmentP4Workspace(
        base=p2_result.workspace,
        tend_work=tend_work,
        tend=tend_views,
        tendencies=tendencies,
        diagnostics=diagnostics,
        fallspeeds=fallspeeds,
        l_qg=l_qg,
        rg=rg,
        ng=ng,
        rb=rb,
        idx_bg=idx_bg,
        ilamg=ilamg,
        mvd_g=mvd_g,
        ng1d=ng1d,
        qb1d=qb1d,
        vtrg=vtrg,
        vtng=vtng,
        substeps_sedi=p4["substeps_sedi"],
        ktop_sedi=p4["ktop_sedi"],
        graupel_liquid_equiv_precip=diagnostics["graupel_liquid_equiv_precip"],
        semi_sedi_flag=bool(semi_sedi_flag),
        branch_active=p4["branch_active"],
        dt=dt,
        odt=odt,
    )
    return TorchSegmentP4Result(result_state, workspace, p2_result.ncol, p2_result.nz)


def run_p5(
    state: Mapping[str, torch.Tensor] | TorchSegmentP3Result | TorchSegmentP4Result,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = False,
    max_sedimentation_substeps: int | None = None,
    max_snow_sedimentation_substeps: int | None = None,
    compact_base: bool = False,
    copy: bool = False,
) -> TorchSegmentP5Result:
    if max_sedimentation_substeps is not None and max_sedimentation_substeps < 1:
        raise ValueError("Segment P5 max_sedimentation_substeps must be positive")
    if max_snow_sedimentation_substeps is not None and max_snow_sedimentation_substeps < 1:
        raise ValueError("Segment P5 max_snow_sedimentation_substeps must be positive")
    upstream_result = (
        _segment_p4_result(
            state,
            dt=dt,
            is_first_step=is_first_step,
            tables=tables,
            turn_off_micro=turn_off_micro,
            cloud_condensation=cloud_condensation,
            semi_sedi_flag=True,
            max_sedimentation_substeps=max_sedimentation_substeps,
        )
        if semi_sedi_flag
        else _segment_p3_result(
            state,
            dt=dt,
            is_first_step=is_first_step,
            tables=tables,
            turn_off_micro=turn_off_micro,
            cloud_condensation=cloud_condensation,
            semi_sedi_flag=False,
            max_sedimentation_substeps=max_sedimentation_substeps,
        )
    )
    input_state = upstream_result.state
    ncol = upstream_result.ncol
    nz = upstream_result.nz
    upstream_workspace = upstream_result.workspace
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    resolved_dt = upstream_workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment P5 dt must be nonzero")
    sedimentation_substeps = _default_sedimentation_substeps(
        dt=resolved_dt,
        semi_sedi_flag=semi_sedi_flag,
        override=max_sedimentation_substeps,
    )
    snow_sedimentation_substeps = _default_snow_sedimentation_substeps(
        semi_sedi_flag=semi_sedi_flag,
        branch_cap=sedimentation_substeps,
        override=max_sedimentation_substeps,
        snow_cap=max_snow_sedimentation_substeps,
    )
    odt = 1.0 / resolved_dt

    if semi_sedi_flag:
        rain_workspace = upstream_workspace.base
        o_workspace = rain_workspace.base
    else:
        rain_workspace = upstream_workspace.base
        o_workspace = rain_workspace.base
    q_nwfa1d = None
    q_nifa1d = None
    q_active_columns = None
    if compact_base:
        q_nwfa1d = o_workspace.nwfa1d
        q_nifa1d = o_workspace.nifa1d
        q_active_columns = _active_columns_from_o_workspace(o_workspace)

    tend = _tensor_mapping(upstream_workspace.tend, copy=copy)
    tendencies = _tensor_mapping(upstream_workspace.tendencies, copy=copy)
    diagnostics = _tensor_mapping(upstream_workspace.diagnostics, copy=copy)
    fallspeeds = _tensor_mapping(upstream_workspace.fallspeeds, copy=copy)

    l_qs = o_workspace.l_qs
    rs = o_workspace.rs
    l_qi = o_workspace.l_qi
    ri = o_workspace.ri
    ni = o_workspace.ni
    ilami = o_workspace.ilami
    l_qc = o_workspace.l_qc
    rc = o_workspace.rc
    nc = o_workspace.nc
    ilamc = o_workspace.ilamc
    o_temp = o_workspace.temp
    o_qv = o_workspace.qv
    o_rho = o_workspace.rho
    o_thermo = o_workspace.thermo
    o_snow_moments = o_workspace.snow_moments
    o_nc1d = o_workspace.nc1d
    o_ncsave = o_workspace.ncsave
    o_ng1d = o_workspace.ng1d
    o_qb1d = o_workspace.qb1d
    rain_rr = rain_workspace.rr
    rain_vtrr = rain_workspace.vtrr
    vtboost = o_workspace.vtboost
    base_workspace = None if compact_base else upstream_workspace
    if compact_base:
        del upstream_result, upstream_workspace, rain_workspace, o_workspace
    vtrs = fallspeeds["vtrs"]
    vtri = fallspeeds["vtri"]
    vtni = fallspeeds["vtni"]
    vtrc = fallspeeds["vtrc"]
    vtnc = fallspeeds["vtnc"]

    vtrs, snow_steps, snow_ktop = snow_fallspeed_tensor(
        rhof=o_thermo["rhof"],
        rs=rs,
        prr_sml=tend["prr_sml"],
        smob=o_snow_moments["smob"],
        smoc=o_snow_moments["smoc"],
        rr=rain_rr,
        vtrr=rain_vtrr,
        dz1d=input_state["dz"],
        vt=vtrs,
        vtboost=vtboost,
        dt=resolved_dt,
    )
    rs, tendencies["qsten"], diagnostics["snow_liquid_equiv_precip"] = _run_sedimentation_recurrence(
        "snow_sedimentation_recurrence",
        _snow_sedimentation_recurrence,
        rs,
        rs,
        vtrs,
        input_state["dz"],
        o_rho,
        tendencies["qsten"],
        diagnostics["snow_liquid_equiv_precip"],
        _limit_like(constants.R1, rs),
        snow_steps,
        snow_ktop,
        snow_sedimentation_substeps,
        resolved_dt,
    )
    _record_sedimentation_substep_overflow(diagnostics, snow_steps, snow_sedimentation_substeps, _column_activity(l_qs),
                                           region=SEDIMENTATION_BOUND_SNOW)

    vtri, vtni, ice_steps, ice_ktop = ice_fallspeed_tensor(
        rhof=o_thermo["rhof"],
        ri=ri,
        ilami=ilami,
        dz1d=input_state["dz"],
        vt=vtri,
        vtn=vtni,
        dt=resolved_dt,
    )
    vtrc, vtnc, cloud_ktop = cloud_fallspeed_tensor(
        rhof=o_thermo["rhof"],
        w1d=input_state["w"],
        rc=rc,
        nc=nc,
        ilamc=ilamc,
        dz1d=input_state["dz"],
        vt=vtrc,
        vtn=vtnc,
    )
    (
        ri,
        ni,
        tendencies["qiten"],
        tendencies["niten"],
        diagnostics["ice_liquid_equiv_precip"],
        rc,
        nc,
        tendencies["qcten"],
        tendencies["ncten"],
        diagnostics["cloud_precip"],
    ) = _run_tensor_compile_island(
        "p5_ice_cloud_sedimentation",
        _p5_ice_cloud_sedimentation_kernel,
        ri,
        ri,
        ni,
        vtri[:, : ri.shape[1]],
        vtni[:, : ni.shape[1]],
        tendencies["qiten"],
        tendencies["niten"],
        ice_steps,
        ice_ktop,
        diagnostics["ice_liquid_equiv_precip"],
        rc,
        nc,
        vtrc[:, : rc.shape[1]],
        vtnc[:, : nc.shape[1]],
        tendencies["qcten"],
        tendencies["ncten"],
        cloud_ktop,
        diagnostics["cloud_precip"],
        input_state["dz"],
        o_rho,
        resolved_dt,
    )

    fallspeeds = {**fallspeeds, "vtrs": vtrs, "vtri": vtri, "vtni": vtni, "vtrc": vtrc, "vtnc": vtnc}
    tend_work, tend_views = _contiguous_tend_work_tensor(tend)
    workspace = TorchSegmentP5Workspace(
        base=base_workspace,
        tend_work=tend_work,
        tend=tend_views,
        tendencies=tendencies,
        diagnostics=diagnostics,
        fallspeeds=fallspeeds,
        l_qs=l_qs,
        rs=rs,
        l_qi=l_qi,
        ri=ri,
        ni=ni,
        ilami=ilami,
        l_qc=l_qc,
        rc=rc,
        nc=nc,
        ilamc=ilamc,
        vtrs=vtrs,
        vtri=vtri,
        vtni=vtni,
        vtrc=vtrc,
        vtnc=vtnc,
        snow_liquid_equiv_precip=diagnostics["snow_liquid_equiv_precip"],
        ice_liquid_equiv_precip=diagnostics["ice_liquid_equiv_precip"],
        cloud_precip=diagnostics["cloud_precip"],
        path_id="semi_lagrangian_sedimentation" if semi_sedi_flag else "eulerian_sedimentation",
        semi_sedi_flag=bool(semi_sedi_flag),
        dt=resolved_dt,
        odt=odt,
        q_temp=o_temp if compact_base else None,
        q_qv=o_qv if compact_base else None,
        q_rho=o_rho if compact_base else None,
        q_thermo=o_thermo if compact_base else None,
        q_nc1d=o_nc1d if compact_base else None,
        q_ncsave=o_ncsave if compact_base else None,
        q_snow_moments=o_snow_moments if compact_base else None,
        q_ng1d=o_ng1d if compact_base else None,
        q_qb1d=o_qb1d if compact_base else None,
        q_nwfa1d=q_nwfa1d,
        q_nifa1d=q_nifa1d,
        q_active_columns=q_active_columns,
    )
    return TorchSegmentP5Result(result_state, workspace, ncol, nz)


def run_q(
    state: Mapping[str, torch.Tensor] | TorchSegmentP5Result,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = True,
    max_sedimentation_substeps: int | None = None,
    compact_base: bool = False,
    copy: bool = False,
) -> TorchSegmentQResult:
    p5_result = _segment_p5_result(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
    )
    input_state = p5_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    ncol = p5_result.ncol
    nz = p5_result.nz
    p5_workspace = p5_result.workspace
    active_columns = _active_columns_from_p5_workspace(p5_workspace)
    resolved_dt = p5_workspace.dt if dt is None else float(dt)
    if resolved_dt == 0.0:
        raise ZeroDivisionError("Segment Q dt must be nonzero")
    odt = 1.0 / resolved_dt

    p5_tend = p5_workspace.tend
    p5_tendencies = p5_workspace.tendencies
    p5_diagnostics = p5_workspace.diagnostics
    p5_fallspeeds = p5_workspace.fallspeeds
    p5_nc = p5_workspace.nc
    path_id = p5_workspace.path_id
    p5_semi_sedi_flag = p5_workspace.semi_sedi_flag
    if p5_workspace.base is None:
        if (
            p5_workspace.q_temp is None
            or p5_workspace.q_qv is None
            or p5_workspace.q_rho is None
            or p5_workspace.q_thermo is None
            or p5_workspace.q_snow_moments is None
        ):
            raise ValueError("Compact Segment P5 workspace is missing Q dependencies")
        temp = p5_workspace.q_temp
        qv = p5_workspace.q_qv
        o_rho = p5_workspace.q_rho
        o_thermo = p5_workspace.q_thermo
        o_nc1d = p5_workspace.q_nc1d
        o_ncsave = p5_workspace.q_ncsave
        o_snow_moments = p5_workspace.q_snow_moments
        o_ng1d = p5_workspace.q_ng1d
        o_qb1d = p5_workspace.q_qb1d
        d_nwfa1d = p5_workspace.q_nwfa1d
        d_nifa1d = p5_workspace.q_nifa1d
    else:
        o_workspace = _segment_o_workspace_from_p5(p5_workspace)
        d_workspace = _segment_d_workspace_from_o(o_workspace)
        temp = o_workspace.temp
        qv = o_workspace.qv
        o_rho = o_workspace.rho
        o_thermo = o_workspace.thermo
        o_nc1d = o_workspace.nc1d
        o_ncsave = o_workspace.ncsave
        o_snow_moments = o_workspace.snow_moments
        o_ng1d = o_workspace.ng1d
        o_qb1d = o_workspace.qb1d
        d_nwfa1d = d_workspace.nwfa1d
        d_nifa1d = d_workspace.nifa1d
    base_workspace = None if compact_base or p5_workspace.base is None else p5_workspace
    if base_workspace is None:
        p5_result = None
        p5_workspace = None

    tend = _tensor_mapping(p5_tend, copy=copy)
    tendencies = _tensor_mapping(p5_tendencies, copy=copy)
    diagnostics = _tensor_mapping(p5_diagnostics, copy=copy)
    fallspeeds = _tensor_mapping(p5_fallspeeds, copy=copy)
    if not turn_off_micro:
        tendencies = {
            **tendencies,
            **freeze_cloud_melt_ice_tensor(
            temp=temp,
            rho=o_rho,
            ocp=o_thermo["ocp"],
            lvap=o_thermo["lvap"],
            qi1d=input_state["qi"],
            ni1d=input_state["ni"],
            qiten=tendencies["qiten"],
            niten=tendencies["niten"],
            qc1d=input_state["qc"],
            qcten=tendencies["qcten"],
            ncten=tendencies["ncten"],
            tten=tendencies["tten"],
            dt=resolved_dt,
            odt=odt,
            nc1d=o_nc1d,
            ncsave=o_ncsave,
            ),
        }

    t1d = input_state["t"] + tendencies["tten"] * _scalar(resolved_dt, temp)
    qv1d = torch.clamp_min(input_state["qv"] + tendencies["qvten"] * _scalar(resolved_dt, temp), _scalar(constants.MIN_QV, temp))
    rho = _scalar(constants.ROVERRV, temp) * input_state["p"] / (
        _scalar(constants.RDRY, temp) * temp * (qv + _scalar(constants.ROVERRV, temp))
    )
    nwfa1d = _final_optional_aerosol_tensor(
        state=input_state,
        aliases=("nwfa1d", "nwfa"),
        base=d_nwfa1d,
        tendency=tendencies["nwfaten"],
        default=constants.NWFA_DEFAULT,
        dt=resolved_dt,
    )
    nifa1d = _final_optional_aerosol_tensor(
        state=input_state,
        aliases=("nifa1d", "nifa"),
        base=d_nifa1d,
        tendency=tendencies["nifaten"],
        default=constants.NIFA_DEFAULT,
        dt=resolved_dt,
    )

    cloud, graupel = cloud_graupel_check_and_update(
        rho=rho,
        qc1d=input_state["qc"],
        nc=p5_nc,
        qcten=tendencies["qcten"],
        ncten=tendencies["ncten"],
        nc1d=o_nc1d,
        ncsave=o_ncsave,
        qg1d=input_state["qg"],
        qgten=tendencies["qgten"],
        ngten=tendencies["ngten"],
        qbten=tendencies["qbten"],
        ng1d=o_ng1d,
        qb1d=o_qb1d,
        dt=resolved_dt,
        odt=odt,
    )
    for name in ("qcten", "ncten"):
        tendencies[name] = cloud[name]
    for name in ("qgten", "ngten", "qbten"):
        tendencies[name] = graupel[name]

    if _regional_execution_enabled("q_combined_checks"):
        rain, ice, snow = _production_precipitation_checks_tensor(
            rho=rho,
            qr1d=input_state["qr"],
            nr1d=input_state["nr"],
            qrten=tendencies["qrten"],
            nrten=tendencies["nrten"],
            qi1d=input_state["qi"],
            ni1d=input_state["ni"],
            qiten=tendencies["qiten"],
            niten=tendencies["niten"],
            qs1d=input_state["qs"],
            qsten=tendencies["qsten"],
            dt=resolved_dt,
            odt=odt,
        )
    else:
        rain = rain_check_and_update(
            rho=rho,
            qr1d=input_state["qr"],
            nr1d=input_state["nr"],
            qrten=tendencies["qrten"],
            nrten=tendencies["nrten"],
            dt=resolved_dt,
            odt=odt,
        )
        ice = ice_check_and_update(
            rho=rho,
            qi1d=input_state["qi"],
            ni1d=input_state["ni"],
            qiten=tendencies["qiten"],
            niten=tendencies["niten"],
            dt=resolved_dt,
            odt=odt,
        )
        snow = snow_check_and_update(
            rho=rho,
            qs1d=input_state["qs"],
            qsten=tendencies["qsten"],
            dt=resolved_dt,
            odt=odt,
        )
    tendencies["qrten"] = rain["qrten"]
    tendencies["nrten"] = rain["nrten"]
    tendencies["qiten"] = ice["qiten"]
    tendencies["niten"] = ice["niten"]
    tendencies["qsten"] = snow["qsten"]
    snow_moments = _final_snow_moments_tensor(
        rs=snow["rs"],
        temp=temp,
        l_qs=snow["l_qs"],
        template=o_snow_moments,
    )

    frozen_numerator = (
        diagnostics["ice_liquid_equiv_precip"]
        + diagnostics["snow_liquid_equiv_precip"]
        + diagnostics["graupel_liquid_equiv_precip"]
    )
    frozen_fraction = frozen_numerator / (frozen_numerator + diagnostics["rain_precip"] + _scalar(constants.R1, temp))
    diagnostics["frozen_fraction"] = frozen_fraction

    tend_work, tend_views = _contiguous_tend_work_tensor(tend)
    workspace = TorchSegmentQWorkspace(
        base=base_workspace,
        tend_work=tend_work,
        tend=tend_views,
        tendencies=tendencies,
        diagnostics=diagnostics,
        fallspeeds=fallspeeds,
        temp=temp,
        qv=qv,
        rho=rho,
        t1d=t1d,
        qv1d=qv1d,
        l_qc=cloud["l_qc"],
        qc1d=cloud["qc1d"],
        rc=cloud["rc"],
        nc=cloud["nc"],
        nc1d=cloud["nc1d"],
        ncsave=o_ncsave,
        ilamc=cloud["ilamc"],
        mvd_c=cloud["mvd_c"],
        l_qr=rain["l_qr"],
        qr1d=rain["qr1d"],
        rr=rain["rr"],
        nr=rain["nr"],
        nr1d=rain["nr1d"],
        ilamr=rain["ilamr"],
        mvd_r=rain["mvd_r"],
        l_qi=ice["l_qi"],
        qi1d=ice["qi1d"],
        ni=ice["ni"],
        ni1d=ice["ni1d"],
        ri=ice["ri"],
        ilami=ice["ilami"],
        l_qs=snow["l_qs"],
        qs1d=snow["qs1d"],
        rs=snow["rs"],
        snow_moments=snow_moments,
        l_qg=graupel["l_qg"],
        qg1d=graupel["qg1d"],
        rg=graupel["rg"],
        ng=graupel["ng"],
        ng1d=graupel["ng1d"],
        qb1d=graupel["qb1d"],
        rb=graupel["rb"],
        idx_bg=graupel["idx"],
        ilamg=graupel["ilamg"],
        mvd_g=graupel["mvd_g"],
        nwfa1d=nwfa1d,
        nifa1d=nifa1d,
        frozen_fraction=frozen_fraction,
        active_columns=active_columns,
        path_id=path_id,
        semi_sedi_flag=p5_semi_sedi_flag,
        dt=resolved_dt,
        odt=odt,
    )
    return TorchSegmentQResult(result_state, workspace, ncol, nz)


def _r1_masked_copyback_kernel(
    active_columns: torch.Tensor,
    *fields: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Write the nine independent masked state outputs in one compile island."""
    if len(fields) != 18:
        raise ValueError("R1 copyback requires nine new and nine original fields")
    return tuple(
        torch.where(active_columns, fields[index], fields[index + 9])
        for index in range(9)
    )


def run_r1(
    state: Mapping[str, torch.Tensor] | TorchSegmentQResult,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = True,
    refl10cm_from_melting: bool = False,
    max_hail_diameter_flag: bool = True,
    max_sedimentation_substeps: int | None = None,
    defer_reflectivity_to_r2: bool = False,
    copy: bool = False,
) -> TorchSegmentR1Result:
    q_result = _segment_q_result(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
    )
    input_state = q_result.state
    result_state = {name: _copy_tensor(values) for name, values in input_state.items()} if copy else dict(input_state)
    q_workspace = q_result.workspace
    copyback_fields = {
        "t": q_workspace.t1d,
        "qv": q_workspace.qv1d,
        "qc": q_workspace.qc1d,
        "qr": q_workspace.qr1d,
        "qi": q_workspace.qi1d,
        "qs": q_workspace.qs1d,
        "qg": q_workspace.qg1d,
        "ni": q_workspace.ni1d,
        "nr": q_workspace.nr1d,
    }
    active_columns = _active_columns_from_q(q_workspace)[:, None]
    if _regional_execution_enabled("r1_fused_copyback"):
        names = tuple(copyback_fields)
        outputs = _run_tensor_compile_island(
            "r1_masked_copyback",
            _r1_masked_copyback_kernel,
            q_workspace.temp,
            active_columns,
            *(copyback_fields[name] for name in names),
            *(input_state[name] for name in names),
        )
        result_state.update(zip(names, outputs, strict=True))
    else:
        for name, values in copyback_fields.items():
            result_state[name] = torch.where(active_columns, values, input_state[name])

    # The model-facing Fortran driver returns optional number/aerosol/hail
    # fields as well as the nine mandatory fields. Inactive columns return
    # their pre-microphysics checked state, already held in input_state.
    for aliases, values in (
        (("nc1d", "nc"), q_workspace.nc1d),
        (("nwfa1d", "nwfa"), q_workspace.nwfa1d),
        (("nifa1d", "nifa"), q_workspace.nifa1d),
        (("ng1d", "ng"), q_workspace.ng1d),
        (("qb1d", "qb"), q_workspace.qb1d),
    ):
        if values is not None:
            for name in aliases:
                if name in input_state:
                    result_state[name] = torch.where(active_columns, values, input_state[name])

    diagnostics = _tensor_mapping(q_workspace.diagnostics, copy=copy)
    diagnostics["frozen_fraction"] = q_workspace.frozen_fraction
    diagnostics["frz_rain_precip"] = freezing_rain_tensor(
        temp=q_workspace.temp[:, 0],
        rain_precip=diagnostics["rain_precip"],
        cloud_precip=diagnostics["cloud_precip"],
    )
    # These main diagnostics are allocated only after the Fortran early-return
    # guard. The driver initializes their public buffers to zero on each call.
    cloud_number_mixing_ratio = torch.where(active_columns, q_workspace.nc * q_workspace.rho, 0.0)
    rain_med_vol_diam = torch.where(active_columns, q_workspace.mvd_r, 0.0)
    graupel_med_vol_diam = torch.where(active_columns, q_workspace.mvd_g, 0.0)
    max_hail_diameter = (
        max_hail_diam_tensor(
            rho=q_workspace.rho,
            rg=q_workspace.rg,
            ng=q_workspace.ng,
            ilamg=q_workspace.ilamg,
            idx=q_workspace.idx_bg,
        )
        if max_hail_diameter_flag
        else torch.zeros_like(q_workspace.rho)
    )
    if defer_reflectivity_to_r2:
        # R2 always recomputes ordinary components before optionally applying
        # melting corrections. The compact B-S path has no R1 observer, so do
        # not execute the same diagnostic twice there.
        refl10cm = q_workspace.rho.new_empty((0,))
    else:
        refl10cm = reflectivity_10cm_tensor(
            refl10cm_from_melting_flag=refl10cm_from_melting,
            temp=q_workspace.temp,
            l_qr=q_workspace.l_qr,
            rr=q_workspace.rr,
            nr=q_workspace.nr,
            ilamr=q_workspace.ilamr,
            l_qs=q_workspace.l_qs,
            rs=q_workspace.rs,
            smoc=q_workspace.snow_moments["smoc"],
            smob=q_workspace.snow_moments["smob"],
            smoz=q_workspace.snow_moments["smoz"],
            l_qg=q_workspace.l_qg,
            rg=q_workspace.rg,
            ng=q_workspace.ng,
            idx=q_workspace.idx_bg,
            ilamg=q_workspace.ilamg,
        )
    workspace = TorchSegmentR1Workspace(
        base=q_workspace,
        diagnostics=diagnostics,
        rain_precip=diagnostics["rain_precip"],
        cloud_precip=diagnostics["cloud_precip"],
        ice_liquid_equiv_precip=diagnostics["ice_liquid_equiv_precip"],
        snow_liquid_equiv_precip=diagnostics["snow_liquid_equiv_precip"],
        graupel_liquid_equiv_precip=diagnostics["graupel_liquid_equiv_precip"],
        frozen_fraction=diagnostics["frozen_fraction"],
        frz_rain_precip=diagnostics["frz_rain_precip"],
        cloud_number_mixing_ratio=cloud_number_mixing_ratio,
        rain_med_vol_diam=rain_med_vol_diam,
        graupel_med_vol_diam=graupel_med_vol_diam,
        max_hail_diameter=max_hail_diameter,
        refl10cm=refl10cm,
        path_id=q_workspace.path_id,
        semi_sedi_flag=q_workspace.semi_sedi_flag,
        dt=q_workspace.dt,
        odt=q_workspace.odt,
    )
    return TorchSegmentR1Result(result_state, workspace, q_result.ncol, q_result.nz)


def run_r2(
    state: Mapping[str, torch.Tensor] | TorchSegmentR1Result,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = True,
    refl10cm_from_melting: bool | torch.Tensor = False,
    max_hail_diameter_flag: bool = True,
    max_sedimentation_substeps: int | None = None,
    compute_reflectivity: bool = True,
    copy: bool = False,
) -> TorchSegmentR2Result:
    r1_result = _segment_r1_result(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_hail_diameter_flag=max_hail_diameter_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
    )
    result_state = {name: _copy_tensor(values) for name, values in r1_result.state.items()} if copy else dict(r1_result.state)
    q_workspace = r1_result.workspace.base
    diagnostics = _tensor_mapping(r1_result.workspace.diagnostics, copy=copy)
    if not compute_reflectivity:
        # A caller that reads no reflectivity skips it (a static choice; the S boundary keeps it).
        workspace = TorchSegmentR2Workspace(
            base=r1_result.workspace,
            diagnostics=diagnostics,
            melting_level_index=None,
            melting_snow_reflectivity=None,
            melting_graupel_reflectivity=None,
            complex_water_0=_RADAR_COMPLEX_WATER_0,
            complex_ice_0=_RADAR_COMPLEX_ICE_0,
            refl10cm=None,
            refl10cm_from_melting_flag=refl10cm_from_melting,
            path_id=r1_result.workspace.path_id,
            semi_sedi_flag=r1_result.workspace.semi_sedi_flag,
            dt=r1_result.workspace.dt,
            odt=r1_result.workspace.odt,
        )
        return TorchSegmentR2Result(result_state, workspace, r1_result.ncol, r1_result.nz)
    refl10cm, melting_level_index, ze_snow, ze_graupel = reflectivity_10cm_components_tensor(
        refl10cm_from_melting_flag=refl10cm_from_melting,
        temp=q_workspace.temp,
        l_qr=q_workspace.l_qr,
        rr=q_workspace.rr,
        nr=q_workspace.nr,
        ilamr=q_workspace.ilamr,
        l_qs=q_workspace.l_qs,
        rs=q_workspace.rs,
        smoc=q_workspace.snow_moments["smoc"],
        smob=q_workspace.snow_moments["smob"],
        smoz=q_workspace.snow_moments["smoz"],
        l_qg=q_workspace.l_qg,
        rg=q_workspace.rg,
        ng=q_workspace.ng,
        idx=q_workspace.idx_bg,
        ilamg=q_workspace.ilamg,
    )
    diagnostics["refl10cm"] = refl10cm
    diagnostics["melting_level_index"] = melting_level_index
    workspace = TorchSegmentR2Workspace(
        base=r1_result.workspace,
        diagnostics=diagnostics,
        melting_level_index=melting_level_index,
        melting_snow_reflectivity=ze_snow,
        melting_graupel_reflectivity=ze_graupel,
        complex_water_0=_RADAR_COMPLEX_WATER_0,
        complex_ice_0=_RADAR_COMPLEX_ICE_0,
        refl10cm=refl10cm,
        refl10cm_from_melting_flag=refl10cm_from_melting,
        path_id=r1_result.workspace.path_id,
        semi_sedi_flag=r1_result.workspace.semi_sedi_flag,
        dt=r1_result.workspace.dt,
        odt=r1_result.workspace.odt,
    )
    return TorchSegmentR2Result(result_state, workspace, r1_result.ncol, r1_result.nz)


def run_s(
    state: Mapping[str, torch.Tensor] | TorchSegmentR2Result,
    *,
    dt: float | None = None,
    is_first_step: bool | None = None,
    tables: Any | None = None,
    turn_off_micro: bool = False,
    cloud_condensation: bool = True,
    semi_sedi_flag: bool = True,
    refl10cm_from_melting: bool | torch.Tensor = False,
    max_hail_diameter_flag: bool = True,
    max_sedimentation_substeps: int | None = None,
    compute_effective_radii: bool = True,
    copy: bool = False,
) -> TorchSegmentSResult:
    r2_result = _segment_r2_result(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        refl10cm_from_melting=refl10cm_from_melting,
        max_hail_diameter_flag=max_hail_diameter_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
    )
    result_state = {name: _copy_tensor(values) for name, values in r2_result.state.items()} if copy else dict(r2_result.state)
    q_workspace = r2_result.workspace.base.base
    diagnostics = _tensor_mapping(r2_result.workspace.diagnostics, copy=copy)
    re_cloud = re_ice = re_snow = None
    if compute_effective_radii:
        re_cloud, re_ice, re_snow = effective_radius_tensor(
            temp=q_workspace.temp,
            l_qc=q_workspace.l_qc,
            nc=q_workspace.nc,
            ilamc=q_workspace.ilamc,
            l_qi=q_workspace.l_qi,
            ilami=q_workspace.ilami,
            l_qs=q_workspace.l_qs,
            rs=q_workspace.rs,
            snow_moments=q_workspace.snow_moments,
        )
        diagnostics["re_cloud"] = re_cloud
        diagnostics["re_ice"] = re_ice
        diagnostics["re_snow"] = re_snow
    tempo_diags = _driver_diagnostic_copyback_tensor(r2_result.workspace, re_cloud, re_ice, re_snow)
    workspace = TorchSegmentSWorkspace(
        base=r2_result.workspace,
        diagnostics=diagnostics,
        tempo_diags=tempo_diags,
        re_cloud=re_cloud,
        re_ice=re_ice,
        re_snow=re_snow,
        path_id=r2_result.workspace.path_id,
        semi_sedi_flag=r2_result.workspace.semi_sedi_flag,
        dt=r2_result.workspace.dt,
        odt=r2_result.workspace.odt,
    )
    return TorchSegmentSResult(result_state, workspace, r2_result.ncol, r2_result.nz)


_OVER_DEPLETION_INPUT_FIELDS = (
    "pri_inu", "pri_ide", "pni_ide", "prs_ide", "prs_sde", "prg_gde", "pri_iha",
    "prr_wau", "pri_wfz", "prr_rcw", "prs_scw", "prg_scw", "prg_gcw",
    "prs_iau", "prs_sci", "pri_rci", "prg_rfz", "pbg_rfz", "pri_rfz", "prr_rci",
    "prr_rcs", "prr_rcg", "prs_ihm", "prr_sml", "prs_rcs", "prg_ihm", "prr_gml",
    "prg_rcg", "pbg_rcg",
)
_OVER_DEPLETION_OUTPUT_FIELDS = (*_OVER_DEPLETION_INPUT_FIELDS, "pri_ihm")


def check_over_depletion_tendencies(
    *,
    rho: torch.Tensor,
    temp: torch.Tensor,
    qvsi: torch.Tensor,
    qv: torch.Tensor,
    l_qc: torch.Tensor,
    rc: torch.Tensor,
    l_qi: torch.Tensor,
    ri: torch.Tensor,
    l_qr: torch.Tensor,
    rr: torch.Tensor,
    l_qs: torch.Tensor,
    rs: torch.Tensor,
    l_qg: torch.Tensor,
    rg: torch.Tensor,
    tend: Mapping[str, torch.Tensor],
    odt: float,
) -> dict[str, torch.Tensor]:
    updated = dict(tend)
    if TENSOR_COMPILE_ISLAND_CUDA and rho.device.type in TENSOR_COMPILE_ISLAND_DEVICES:
        values = _run_tensor_compile_island(
            "check_over_depletion_tendencies",
            _check_over_depletion_tendencies_kernel,
            rho,
            rho,
            temp,
            qvsi,
            qv,
            l_qc,
            rc,
            l_qi,
            ri,
            l_qr,
            rr,
            l_qs,
            rs,
            l_qg,
            rg,
            float(odt),
            *(updated[name] for name in _OVER_DEPLETION_INPUT_FIELDS),
        )
        updated.update(zip(_OVER_DEPLETION_OUTPUT_FIELDS, values, strict=True))
        return updated
    sump = updated["pri_inu"] + updated["pri_ide"] + updated["prs_ide"] + updated["prs_sde"] + updated["prg_gde"] + updated["pri_iha"]
    rate_max = (qv - qvsi) * rho * odt * 0.999
    mask = ((sump > _scalar(constants.EPS, sump)) & (sump > rate_max)) | ((sump < -_scalar(constants.EPS, sump)) & (sump < rate_max))
    ratio_denominator = torch.where(mask, sump, torch.ones_like(sump))
    ratio = torch.where(mask, rate_max / ratio_denominator, torch.ones_like(sump))
    for name in ("pri_inu", "pri_ide", "pni_ide", "prs_ide", "prs_sde", "prg_gde", "pri_iha"):
        updated[name] = updated[name] * ratio

    cloud_sump = -updated["prr_wau"] - updated["pri_wfz"] - updated["prr_rcw"] - updated["prs_scw"] - updated["prg_scw"] - updated["prg_gcw"]
    cloud_mask = l_qc & (cloud_sump < -rc * _scalar(odt, rc))
    cloud_denominator = torch.where(cloud_mask, cloud_sump, torch.ones_like(cloud_sump))
    updated = _scale_tendencies_if(updated, ("prr_wau", "pri_wfz", "prr_rcw", "prs_scw", "prg_scw", "prg_gcw"), cloud_mask, (-rc * _scalar(odt, rc)) / cloud_denominator)

    ice_sump = updated["pri_ide"] - updated["prs_iau"] - updated["prs_sci"] - updated["pri_rci"]
    ice_mask = l_qi & (ice_sump < -ri * _scalar(odt, ri))
    ice_denominator = torch.where(ice_mask, ice_sump, torch.ones_like(ice_sump))
    updated = _scale_tendencies_if(updated, ("pri_ide", "prs_iau", "prs_sci", "pri_rci"), ice_mask, (-ri * _scalar(odt, ri)) / ice_denominator)

    rain_sump = -updated["prg_rfz"] - updated["pri_rfz"] - updated["prr_rci"] + updated["prr_rcs"] + updated["prr_rcg"]
    rain_mask = l_qr & (rain_sump < -rr * _scalar(odt, rr))
    rain_denominator = torch.where(rain_mask, rain_sump, torch.ones_like(rain_sump))
    updated = _scale_tendencies_if(updated, ("prg_rfz", "pbg_rfz", "pri_rfz", "prr_rci", "prr_rcs", "prr_rcg"), rain_mask, (-rr * _scalar(odt, rr)) / rain_denominator)

    snow_sump = updated["prs_sde"] - updated["prs_ihm"] - updated["prr_sml"] + updated["prs_rcs"]
    snow_mask = l_qs & (snow_sump < -rs * _scalar(odt, rs))
    snow_denominator = torch.where(snow_mask, snow_sump, torch.ones_like(snow_sump))
    updated = _scale_tendencies_if(updated, ("prs_sde", "prs_ihm", "prr_sml", "prs_rcs"), snow_mask, (-rs * _scalar(odt, rs)) / snow_denominator)

    graupel_sump = updated["prg_gde"] - updated["prg_ihm"] - updated["prr_gml"] + updated["prg_rcg"]
    graupel_mask = l_qg & (graupel_sump < -rg * _scalar(odt, rg))
    graupel_denominator = torch.where(graupel_mask, graupel_sump, torch.ones_like(graupel_sump))
    updated = _scale_tendencies_if(updated, ("prg_gde", "prg_ihm", "prr_gml", "prg_rcg", "pbg_rcg"), graupel_mask, (-rg * _scalar(odt, rg)) / graupel_denominator)

    updated["pri_ihm"] = updated["prs_ihm"] + updated["prg_ihm"]
    rcg_abs = torch.minimum(torch.abs(updated["prr_rcg"]), torch.abs(updated["prg_rcg"]))
    updated["prr_rcg"] = rcg_abs * _sign_one(updated["prr_rcg"])
    updated["prg_rcg"] = -updated["prr_rcg"]
    updated["pbg_rcg"] = _scalar(constants.METERS3_TO_LITERS / constants.RHO_I, temp) * updated["prg_rcg"]

    warm = temp > _scalar(constants.T0, temp)
    rcs_abs = torch.minimum(torch.abs(updated["prr_rcs"]), torch.abs(updated["prs_rcs"]))
    prr_rcs = rcs_abs * _sign_one(updated["prr_rcs"])
    updated["prr_rcs"] = torch.where(warm, prr_rcs, updated["prr_rcs"])
    updated["prs_rcs"] = torch.where(warm, -prr_rcs, updated["prs_rcs"])
    return updated


def _check_over_depletion_tendencies_kernel(
    rho: torch.Tensor,
    temp: torch.Tensor,
    qvsi: torch.Tensor,
    qv: torch.Tensor,
    l_qc: torch.Tensor,
    rc: torch.Tensor,
    l_qi: torch.Tensor,
    ri: torch.Tensor,
    l_qr: torch.Tensor,
    rr: torch.Tensor,
    l_qs: torch.Tensor,
    rs: torch.Tensor,
    l_qg: torch.Tensor,
    rg: torch.Tensor,
    odt: float,
    pri_inu: torch.Tensor,
    pri_ide: torch.Tensor,
    pni_ide: torch.Tensor,
    prs_ide: torch.Tensor,
    prs_sde: torch.Tensor,
    prg_gde: torch.Tensor,
    pri_iha: torch.Tensor,
    prr_wau: torch.Tensor,
    pri_wfz: torch.Tensor,
    prr_rcw: torch.Tensor,
    prs_scw: torch.Tensor,
    prg_scw: torch.Tensor,
    prg_gcw: torch.Tensor,
    prs_iau: torch.Tensor,
    prs_sci: torch.Tensor,
    pri_rci: torch.Tensor,
    prg_rfz: torch.Tensor,
    pbg_rfz: torch.Tensor,
    pri_rfz: torch.Tensor,
    prr_rci: torch.Tensor,
    prr_rcs: torch.Tensor,
    prr_rcg: torch.Tensor,
    prs_ihm: torch.Tensor,
    prr_sml: torch.Tensor,
    prs_rcs: torch.Tensor,
    prg_ihm: torch.Tensor,
    prr_gml: torch.Tensor,
    prg_rcg: torch.Tensor,
    pbg_rcg: torch.Tensor,
    _eps: float = float(constants.EPS),
    _t0: float = float(constants.T0),
    _graupel_volume_scale: float = float(constants.METERS3_TO_LITERS / constants.RHO_I),
) -> tuple[torch.Tensor, ...]:
    sump = pri_inu + pri_ide + prs_ide + prs_sde + prg_gde + pri_iha
    rate_max = (qv - qvsi) * rho * odt * 0.999
    mask = ((sump > _eps) & (sump > rate_max)) | ((sump < -_eps) & (sump < rate_max))
    ratio = torch.where(mask, rate_max / torch.where(mask, sump, torch.ones_like(sump)), 1.0)
    pri_inu = pri_inu * ratio
    pri_ide = pri_ide * ratio
    pni_ide = pni_ide * ratio
    prs_ide = prs_ide * ratio
    prs_sde = prs_sde * ratio
    prg_gde = prg_gde * ratio
    pri_iha = pri_iha * ratio

    cloud_sump = -prr_wau - pri_wfz - prr_rcw - prs_scw - prg_scw - prg_gcw
    cloud_mask = l_qc & (cloud_sump < -rc * odt)
    ratio = torch.where(cloud_mask, (-rc * odt) / torch.where(cloud_mask, cloud_sump, torch.ones_like(cloud_sump)), 1.0)
    prr_wau = prr_wau * ratio
    pri_wfz = pri_wfz * ratio
    prr_rcw = prr_rcw * ratio
    prs_scw = prs_scw * ratio
    prg_scw = prg_scw * ratio
    prg_gcw = prg_gcw * ratio

    ice_sump = pri_ide - prs_iau - prs_sci - pri_rci
    ice_mask = l_qi & (ice_sump < -ri * odt)
    ratio = torch.where(ice_mask, (-ri * odt) / torch.where(ice_mask, ice_sump, torch.ones_like(ice_sump)), 1.0)
    pri_ide = pri_ide * ratio
    prs_iau = prs_iau * ratio
    prs_sci = prs_sci * ratio
    pri_rci = pri_rci * ratio

    rain_sump = -prg_rfz - pri_rfz - prr_rci + prr_rcs + prr_rcg
    rain_mask = l_qr & (rain_sump < -rr * odt)
    ratio = torch.where(rain_mask, (-rr * odt) / torch.where(rain_mask, rain_sump, torch.ones_like(rain_sump)), 1.0)
    prg_rfz = prg_rfz * ratio
    pbg_rfz = pbg_rfz * ratio
    pri_rfz = pri_rfz * ratio
    prr_rci = prr_rci * ratio
    prr_rcs = prr_rcs * ratio
    prr_rcg = prr_rcg * ratio

    snow_sump = prs_sde - prs_ihm - prr_sml + prs_rcs
    snow_mask = l_qs & (snow_sump < -rs * odt)
    ratio = torch.where(snow_mask, (-rs * odt) / torch.where(snow_mask, snow_sump, torch.ones_like(snow_sump)), 1.0)
    prs_sde = prs_sde * ratio
    prs_ihm = prs_ihm * ratio
    prr_sml = prr_sml * ratio
    prs_rcs = prs_rcs * ratio

    graupel_sump = prg_gde - prg_ihm - prr_gml + prg_rcg
    graupel_mask = l_qg & (graupel_sump < -rg * odt)
    ratio = torch.where(graupel_mask, (-rg * odt) / torch.where(graupel_mask, graupel_sump, torch.ones_like(graupel_sump)), 1.0)
    prg_gde = prg_gde * ratio
    prg_ihm = prg_ihm * ratio
    prr_gml = prr_gml * ratio
    prg_rcg = prg_rcg * ratio
    pbg_rcg = pbg_rcg * ratio

    pri_ihm = prs_ihm + prg_ihm
    rcg_abs = torch.minimum(torch.abs(prr_rcg), torch.abs(prg_rcg))
    prr_rcg = rcg_abs * torch.where(prr_rcg < 0.0, -1.0, 1.0)
    prg_rcg = -prr_rcg
    pbg_rcg = _graupel_volume_scale * prg_rcg
    warm = temp > _t0
    rcs_abs = torch.minimum(torch.abs(prr_rcs), torch.abs(prs_rcs))
    prr_rcs_warm = rcs_abs * torch.where(prr_rcs < 0.0, -1.0, 1.0)
    prr_rcs = torch.where(warm, prr_rcs_warm, prr_rcs)
    prs_rcs = torch.where(warm, -prr_rcs_warm, prs_rcs)
    return (
        pri_inu, pri_ide, pni_ide, prs_ide, prs_sde, prg_gde, pri_iha,
        prr_wau, pri_wfz, prr_rcw, prs_scw, prg_scw, prg_gcw,
        prs_iau, prs_sci, pri_rci, prg_rfz, pbg_rfz, pri_rfz, prr_rci,
        prr_rcs, prr_rcg, prs_ihm, prr_sml, prs_rcs, prg_ihm, prr_gml,
        prg_rcg, pbg_rcg, pri_ihm,
    )


_SUM_TENDENCY_FIELDS = (
    "prr_wau", "pnr_wau", "pnc_wau", "prr_rcw", "pnc_rcw", "pnr_rcr",
    "prs_scw", "pnc_scw", "png_scw", "pbg_scw", "prg_gcw", "pnc_gcw", "pbg_gcw",
    "pri_ihm", "pni_ihm", "prs_ihm", "prg_ihm", "prg_scw",
    "prr_rcs", "pnr_rcs", "prg_rcs", "png_rcs", "prs_rcs", "pbg_rcs",
    "prr_rcg", "pnr_rcg", "prg_rcg", "png_rcg", "pbg_rcg",
    "pri_inu", "pni_inu", "pri_iha", "pni_iha", "pri_wfz", "pni_wfz",
    "prg_rfz", "png_rfz", "pnr_rfz", "pri_rfz", "pni_rfz", "pbg_rfz",
    "prs_sde", "pri_ide", "pni_ide", "prs_ide", "prg_gde", "png_gde",
    "pni_iau", "prs_iau", "prr_sml", "prr_gml", "pbg_sml", "pbg_gml",
    "pnr_sml", "pnr_gml", "prr_rci", "pnr_rci", "pri_rci", "pni_rci",
    "prg_rci", "png_rci", "pbg_rci", "pni_sci", "prs_sci",
)
_SUM_TENDENCY_OUTPUT_FIELDS = (
    "qvten", "qcten", "ncten", "qiten", "niten", "qrten",
    "nrten", "qsten", "qgten", "ngten", "qbten", "tten",
)
# L overwrites a few K/J fields and K overwrites a few J fields. N receives
# only the latest producer's packed storage for each named tendency.
_N_PACKED_OTHER_FIELDS = tuple(
    name for name in _SUM_TENDENCY_FIELDS
    if name not in _ICE_PROCESS_TENDENCY_FIELDS
    and name not in _RIMING_TENDENCY_FIELDS
    and name not in _MELTING_TENDENCY_FIELDS
)
_N_PACKED_OTHER_INDICES = {name: index for index, name in enumerate(_N_PACKED_OTHER_FIELDS)}
_N_PACKED_TENDENCY_SELECTORS = tuple(
    (2, _MELTING_TENDENCY_FIELDS.index(name)) if name in _MELTING_TENDENCY_FIELDS
    else (1, _RIMING_TENDENCY_FIELDS.index(name)) if name in _RIMING_TENDENCY_FIELDS
    else (0, _ICE_PROCESS_TENDENCY_FIELDS.index(name)) if name in _ICE_PROCESS_TENDENCY_FIELDS
    else (3, _N_PACKED_OTHER_INDICES[name])
    for name in _SUM_TENDENCY_FIELDS
)
_SUM_MASS_INPUT_FIELDS = (
    "pri_inu", "pri_iha", "pri_ide", "prs_ide", "prs_sde", "prg_gde",
    "prr_wau", "pri_wfz", "prr_rcw", "prs_scw", "prg_scw", "prg_gcw",
    "pri_ihm", "pri_rfz", "prs_iau", "prs_sci", "pri_rci",
    "prr_sml", "prr_gml", "prr_rcs", "prr_rcg", "prg_rfz", "prr_rci",
    "prs_rcs", "prs_ihm", "prg_rcg", "prg_rci", "prg_rcs", "prg_ihm",
)
_SUM_MASS_OUTPUT_FIELDS = ("qvten", "qcten", "qiten", "qrten", "qsten", "qgten")
_SUM_NUMBER_INPUT_FIELDS = (
    "pnc_wau", "pnc_rcw", "pni_wfz", "pnc_scw", "pnc_gcw",
    "pni_inu", "pni_iha", "pni_ihm", "pni_rfz", "pni_ide", "pni_iau", "pni_sci", "pni_rci",
    "pnr_wau", "pnr_sml", "pnr_gml", "pnr_rfz", "pnr_rcr", "pnr_rcg", "pnr_rcs", "pnr_rci",
    "png_scw", "png_rfz", "png_rcg", "png_rci", "png_rcs", "png_gde",
    "pbg_scw", "pbg_rfz", "pbg_gcw", "pbg_rci", "pbg_rcs", "pbg_rcg", "pbg_sml", "pbg_gml",
    "prg_gde", "prg_ihm",
)
_SUM_NUMBER_OUTPUT_FIELDS = ("ncten", "niten", "nrten", "ngten", "qbten")
_SUM_THERMAL_INPUT_FIELDS = (
    "pri_inu", "pri_ide", "prs_ide", "prs_sde", "prg_gde", "pri_iha",
    "pri_wfz", "pri_rfz", "prg_rfz", "prs_scw", "prg_scw", "prg_gcw",
    "prg_rcs", "prs_rcs", "prr_rci", "prg_rcg",
    "prr_sml", "prr_gml", "prr_rcg", "prr_rcs",
)
_SUM_MASS_INPUT_INDICES = tuple(_SUM_TENDENCY_FIELDS.index(name) for name in _SUM_MASS_INPUT_FIELDS)
_SUM_MASS_OUTPUT_INDICES = tuple(_SUM_TENDENCY_OUTPUT_FIELDS.index(name) for name in _SUM_MASS_OUTPUT_FIELDS)
_SUM_NUMBER_INPUT_INDICES = tuple(_SUM_TENDENCY_FIELDS.index(name) for name in _SUM_NUMBER_INPUT_FIELDS)
_SUM_NUMBER_OUTPUT_INDICES = tuple(_SUM_TENDENCY_OUTPUT_FIELDS.index(name) for name in _SUM_NUMBER_OUTPUT_FIELDS)
_SUM_THERMAL_INPUT_INDICES = tuple(_SUM_TENDENCY_FIELDS.index(name) for name in _SUM_THERMAL_INPUT_FIELDS)
_SUM_THERMAL_OUTPUT_INDEX = _SUM_TENDENCY_OUTPUT_FIELDS.index("tten")


def sum_tendencies_tensor(
    *,
    rho: torch.Tensor,
    temp: torch.Tensor,
    idx: torch.Tensor,
    lvap: torch.Tensor,
    ocp: torch.Tensor,
    tend: Mapping[str, torch.Tensor],
    tendencies: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    updated = dict(tendencies)
    if TENSOR_COMPILE_ISLAND_CUDA and rho.device.type in TENSOR_COMPILE_ISLAND_DEVICES:
        values = _run_tensor_compile_island(
            "sum_tendencies",
            _sum_tendencies_kernel,
            rho,
            rho,
            temp,
            idx,
            lvap,
            ocp,
            _tensor(_STATIC_GRAUPEL_RHO_G, temp),
            *(tend[name] for name in _SUM_TENDENCY_FIELDS),
            *(updated[name] for name in _SUM_TENDENCY_OUTPUT_FIELDS),
        )
        updated.update(zip(_SUM_TENDENCY_OUTPUT_FIELDS, values, strict=True))
        return updated
    orho = 1.0 / rho
    lfus2 = _scalar(constants.LSUB, temp) - lvap
    updated["qvten"] = updated["qvten"] + (
        -tend["pri_inu"] - tend["pri_iha"] - tend["pri_ide"] - tend["prs_ide"] - tend["prs_sde"] - tend["prg_gde"]
    ) * orho
    updated["qcten"] = updated["qcten"] + (
        -tend["prr_wau"] - tend["pri_wfz"] - tend["prr_rcw"] - tend["prs_scw"] - tend["prg_scw"] - tend["prg_gcw"]
    ) * orho
    updated["ncten"] = updated["ncten"] + (
        -tend["pnc_wau"] - tend["pnc_rcw"] - tend["pni_wfz"] - tend["pnc_scw"] - tend["pnc_gcw"]
    ) * orho
    updated["qiten"] = updated["qiten"] + (
        tend["pri_inu"] + tend["pri_iha"] + tend["pri_ihm"] + tend["pri_wfz"] + tend["pri_rfz"] + tend["pri_ide"]
        - tend["prs_iau"] - tend["prs_sci"] - tend["pri_rci"]
    ) * orho
    updated["niten"] = updated["niten"] + (
        tend["pni_inu"] + tend["pni_iha"] + tend["pni_ihm"] + tend["pni_wfz"] + tend["pni_rfz"] + tend["pni_ide"]
        - tend["pni_iau"] - tend["pni_sci"] - tend["pni_rci"]
    ) * orho
    updated["qrten"] = updated["qrten"] + (
        tend["prr_wau"] + tend["prr_rcw"] + tend["prr_sml"] + tend["prr_gml"] + tend["prr_rcs"] + tend["prr_rcg"]
        - tend["prg_rfz"] - tend["pri_rfz"] - tend["prr_rci"]
    ) * orho
    updated["nrten"] = updated["nrten"] + (
        tend["pnr_wau"] + tend["pnr_sml"] + tend["pnr_gml"]
        - (tend["pnr_rfz"] + tend["pnr_rcr"] + tend["pnr_rcg"] + tend["pnr_rcs"] + tend["pnr_rci"] + tend["pni_rfz"])
    ) * orho
    updated["qsten"] = updated["qsten"] + (
        tend["prs_iau"] + tend["prs_sde"] + tend["prs_sci"] + tend["prs_scw"] + tend["prs_rcs"] + tend["prs_ide"]
        - tend["prs_ihm"] - tend["prr_sml"]
    ) * orho
    updated["qgten"] = updated["qgten"] + (
        tend["prg_scw"] + tend["prg_rfz"] + tend["prg_gde"] + tend["prg_rcg"] + tend["prg_gcw"] + tend["prg_rci"] + tend["prg_rcs"]
        - tend["prg_ihm"] - tend["prr_gml"]
    ) * orho
    updated["ngten"] = updated["ngten"] + (
        tend["png_scw"] + tend["png_rfz"] - tend["png_rcg"] + tend["png_rci"] + tend["png_rcs"] + tend["png_gde"] - tend["pnr_gml"]
    ) * orho
    graupel_idx = idx.to(torch.int64).clamp(1, constants.NRHG)
    updated["qbten"] = updated["qbten"] + (
        tend["pbg_scw"] + tend["pbg_rfz"] + tend["pbg_gcw"] + tend["pbg_rci"] + tend["pbg_rcs"] + tend["pbg_rcg"] + tend["pbg_sml"] - tend["pbg_gml"]
        + _scalar(constants.METERS3_TO_LITERS, temp) * (tend["prg_gde"] - tend["prg_ihm"]) / _take(_STATIC_GRAUPEL_RHO_G, graupel_idx, temp)
    ) * orho
    cold_tten = (
        _scalar(constants.LSUB, temp) * ocp * (tend["pri_inu"] + tend["pri_ide"] + tend["prs_ide"] + tend["prs_sde"] + tend["prg_gde"] + tend["pri_iha"])
        + lfus2 * ocp * (
            tend["pri_wfz"] + tend["pri_rfz"] + tend["prg_rfz"] + tend["prs_scw"] + tend["prg_scw"] + tend["prg_gcw"]
            + tend["prg_rcs"] + tend["prs_rcs"] + tend["prr_rci"] + tend["prg_rcg"]
        )
    ) * orho
    warm_tten = (
        _scalar(constants.LFUS, temp) * ocp * (-tend["prr_sml"] - tend["prr_gml"] - tend["prr_rcg"] - tend["prr_rcs"])
        + _scalar(constants.LSUB, temp) * ocp * (tend["prs_sde"] + tend["prg_gde"])
    ) * orho
    updated["tten"] = updated["tten"] + torch.where(temp < _scalar(constants.T0, temp), cold_tten, warm_tten)
    return updated


def _split_sum_tendencies_tensor(
    *,
    rho: torch.Tensor,
    temp: torch.Tensor,
    idx: torch.Tensor,
    lvap: torch.Tensor,
    ocp: torch.Tensor,
    tend: Mapping[str, torch.Tensor],
    tendencies: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Evaluate the existing N sums across three bounded fusion islands."""
    updated = dict(tendencies)
    mass = _run_tensor_compile_island(
        "n_sum_mass",
        _sum_mass_tendencies_kernel,
        rho,
        rho,
        *(tend[name] for name in _SUM_MASS_INPUT_FIELDS),
        *(updated[name] for name in _SUM_MASS_OUTPUT_FIELDS),
    )
    number = _run_tensor_compile_island(
        "n_sum_number_volume",
        _sum_number_tendencies_kernel,
        rho,
        rho,
        idx,
        _tensor(_STATIC_GRAUPEL_RHO_G, temp),
        *(tend[name] for name in _SUM_NUMBER_INPUT_FIELDS),
        *(updated[name] for name in _SUM_NUMBER_OUTPUT_FIELDS),
    )
    thermal = _run_tensor_compile_island(
        "n_sum_thermal",
        _sum_thermal_tendency_kernel,
        rho,
        rho,
        temp,
        lvap,
        ocp,
        *(tend[name] for name in _SUM_THERMAL_INPUT_FIELDS),
        updated["tten"],
    )
    updated.update(zip(_SUM_MASS_OUTPUT_FIELDS, mass, strict=True))
    updated.update(zip(_SUM_NUMBER_OUTPUT_FIELDS, number, strict=True))
    updated["tten"] = thermal
    return updated


def _sum_mass_tendencies_kernel(
    rho: torch.Tensor,
    *values: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    (
        pri_inu, pri_iha, pri_ide, prs_ide, prs_sde, prg_gde,
        prr_wau, pri_wfz, prr_rcw, prs_scw, prg_scw, prg_gcw,
        pri_ihm, pri_rfz, prs_iau, prs_sci, pri_rci,
        prr_sml, prr_gml, prr_rcs, prr_rcg, prg_rfz, prr_rci,
        prs_rcs, prs_ihm, prg_rcg, prg_rci, prg_rcs, prg_ihm,
        qvten, qcten, qiten, qrten, qsten, qgten,
    ) = values
    orho = 1.0 / rho
    qvten = qvten + (-pri_inu - pri_iha - pri_ide - prs_ide - prs_sde - prg_gde) * orho
    qcten = qcten + (-prr_wau - pri_wfz - prr_rcw - prs_scw - prg_scw - prg_gcw) * orho
    qiten = qiten + (pri_inu + pri_iha + pri_ihm + pri_wfz + pri_rfz + pri_ide - prs_iau - prs_sci - pri_rci) * orho
    qrten = qrten + (prr_wau + prr_rcw + prr_sml + prr_gml + prr_rcs + prr_rcg - prg_rfz - pri_rfz - prr_rci) * orho
    qsten = qsten + (prs_iau + prs_sde + prs_sci + prs_scw + prs_rcs + prs_ide - prs_ihm - prr_sml) * orho
    qgten = qgten + (prg_scw + prg_rfz + prg_gde + prg_rcg + prg_gcw + prg_rci + prg_rcs - prg_ihm - prr_gml) * orho
    return qvten, qcten, qiten, qrten, qsten, qgten


def _sum_number_tendencies_kernel(
    rho: torch.Tensor,
    idx: torch.Tensor,
    graupel_density: torch.Tensor,
    *values: torch.Tensor,
    _meters3_to_liters: float = float(constants.METERS3_TO_LITERS),
) -> tuple[torch.Tensor, ...]:
    (
        pnc_wau, pnc_rcw, pni_wfz, pnc_scw, pnc_gcw,
        pni_inu, pni_iha, pni_ihm, pni_rfz, pni_ide, pni_iau, pni_sci, pni_rci,
        pnr_wau, pnr_sml, pnr_gml, pnr_rfz, pnr_rcr, pnr_rcg, pnr_rcs, pnr_rci,
        png_scw, png_rfz, png_rcg, png_rci, png_rcs, png_gde,
        pbg_scw, pbg_rfz, pbg_gcw, pbg_rci, pbg_rcs, pbg_rcg, pbg_sml, pbg_gml,
        prg_gde, prg_ihm,
        ncten, niten, nrten, ngten, qbten,
    ) = values
    orho = 1.0 / rho
    ncten = ncten + (-pnc_wau - pnc_rcw - pni_wfz - pnc_scw - pnc_gcw) * orho
    niten = niten + (pni_inu + pni_iha + pni_ihm + pni_wfz + pni_rfz + pni_ide - pni_iau - pni_sci - pni_rci) * orho
    nrten = nrten + (pnr_wau + pnr_sml + pnr_gml - (pnr_rfz + pnr_rcr + pnr_rcg + pnr_rcs + pnr_rci + pni_rfz)) * orho
    ngten = ngten + (png_scw + png_rfz - png_rcg + png_rci + png_rcs + png_gde - pnr_gml) * orho
    graupel_idx = idx.to(torch.int64).clamp(1, graupel_density.shape[0])
    qbten = qbten + (
        pbg_scw + pbg_rfz + pbg_gcw + pbg_rci + pbg_rcs + pbg_rcg + pbg_sml - pbg_gml
        + _meters3_to_liters * (prg_gde - prg_ihm) / graupel_density[graupel_idx - 1]
    ) * orho
    return ncten, niten, nrten, ngten, qbten


def _sum_thermal_tendency_kernel(
    rho: torch.Tensor,
    temp: torch.Tensor,
    lvap: torch.Tensor,
    ocp: torch.Tensor,
    *values: torch.Tensor,
    _lsub: float = float(constants.LSUB),
    _lfus: float = float(constants.LFUS),
    _t0: float = float(constants.T0),
) -> torch.Tensor:
    (
        pri_inu, pri_ide, prs_ide, prs_sde, prg_gde, pri_iha,
        pri_wfz, pri_rfz, prg_rfz, prs_scw, prg_scw, prg_gcw,
        prg_rcs, prs_rcs, prr_rci, prg_rcg,
        prr_sml, prr_gml, prr_rcg, prr_rcs,
        tten,
    ) = values
    orho = 1.0 / rho
    lfus2 = _lsub - lvap
    cold_tten = (
        _lsub * ocp * (pri_inu + pri_ide + prs_ide + prs_sde + prg_gde + pri_iha)
        + lfus2 * ocp * (pri_wfz + pri_rfz + prg_rfz + prs_scw + prg_scw + prg_gcw + prg_rcs + prs_rcs + prr_rci + prg_rcg)
    ) * orho
    warm_tten = (
        _lfus * ocp * (-prr_sml - prr_gml - prr_rcg - prr_rcs)
        + _lsub * ocp * (prs_sde + prg_gde)
    ) * orho
    return tten + torch.where(temp < _t0, cold_tten, warm_tten)


def _sum_tendencies_kernel(
    rho: torch.Tensor,
    temp: torch.Tensor,
    idx: torch.Tensor,
    lvap: torch.Tensor,
    ocp: torch.Tensor,
    graupel_density: torch.Tensor,
    *values: torch.Tensor,
    _lsub: float = float(constants.LSUB),
    _meters3_to_liters: float = float(constants.METERS3_TO_LITERS),
    _lfus: float = float(constants.LFUS),
    _t0: float = float(constants.T0),
) -> tuple[torch.Tensor, ...]:
    tend_values = values[: len(_SUM_TENDENCY_FIELDS)]
    tendency_values = values[len(_SUM_TENDENCY_FIELDS) :]
    mass = _sum_mass_tendencies_kernel(
        rho,
        *(tend_values[index] for index in _SUM_MASS_INPUT_INDICES),
        *(tendency_values[index] for index in _SUM_MASS_OUTPUT_INDICES),
    )
    number = _sum_number_tendencies_kernel(
        rho,
        idx,
        graupel_density,
        *(tend_values[index] for index in _SUM_NUMBER_INPUT_INDICES),
        *(tendency_values[index] for index in _SUM_NUMBER_OUTPUT_INDICES),
        _meters3_to_liters=_meters3_to_liters,
    )
    thermal = _sum_thermal_tendency_kernel(
        rho,
        temp,
        lvap,
        ocp,
        *(tend_values[index] for index in _SUM_THERMAL_INPUT_INDICES),
        tendency_values[_SUM_THERMAL_OUTPUT_INDEX],
        _lsub=_lsub,
        _lfus=_lfus,
        _t0=_t0,
    )
    return (
        mass[0], mass[1], number[0], mass[2], number[1], mass[3],
        number[2], mass[4], mass[5], number[3], number[4], thermal,
    )


_OVER_DEPLETION_FROM_SUM_INDICES = tuple(
    _SUM_TENDENCY_FIELDS.index(name) for name in _OVER_DEPLETION_INPUT_FIELDS
)
_SUM_LIMITED_OUTPUT_INDICES = tuple(
    _OVER_DEPLETION_OUTPUT_FIELDS.index(name) if name in _OVER_DEPLETION_OUTPUT_FIELDS else -1
    for name in _SUM_TENDENCY_FIELDS
)


def limit_and_sum_tendencies_tensor(
    *,
    rho: torch.Tensor,
    temp: torch.Tensor,
    idx: torch.Tensor,
    lvap: torch.Tensor,
    ocp: torch.Tensor,
    qvsi: torch.Tensor,
    qv: torch.Tensor,
    l_qc: torch.Tensor,
    rc: torch.Tensor,
    l_qi: torch.Tensor,
    ri: torch.Tensor,
    l_qr: torch.Tensor,
    rr: torch.Tensor,
    l_qs: torch.Tensor,
    rs: torch.Tensor,
    l_qg: torch.Tensor,
    rg: torch.Tensor,
    tend: Mapping[str, torch.Tensor],
    tendencies: Mapping[str, torch.Tensor],
    odt: float,
    producer_packs: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if N_LIMITER_COMPILE_SCHEDULES[N_LIMITER_COMPILE_SCHEDULE]["pressure_split"]:
        limited = check_over_depletion_tendencies(
            rho=rho,
            temp=temp,
            qvsi=qvsi,
            qv=qv,
            l_qc=l_qc,
            rc=rc,
            l_qi=l_qi,
            ri=ri,
            l_qr=l_qr,
            rr=rr,
            l_qs=l_qs,
            rs=rs,
            l_qg=l_qg,
            rg=rg,
            tend=tend,
            odt=odt,
        )
        return limited, _split_sum_tendencies_tensor(
            rho=rho,
            temp=temp,
            idx=idx,
            lvap=lvap,
            ocp=ocp,
            tend=limited,
            tendencies=tendencies,
        )
    if not (TENSOR_COMPILE_ISLAND_CUDA and rho.device.type in TENSOR_COMPILE_ISLAND_DEVICES):
        limited = check_over_depletion_tendencies(
            rho=rho,
            temp=temp,
            qvsi=qvsi,
            qv=qv,
            l_qc=l_qc,
            rc=rc,
            l_qi=l_qi,
            ri=ri,
            l_qr=l_qr,
            rr=rr,
            l_qs=l_qs,
            rs=rs,
            l_qg=l_qg,
            rg=rg,
            tend=tend,
            odt=odt,
        )
        return limited, sum_tendencies_tensor(
            rho=rho,
            temp=temp,
            idx=idx,
            lvap=lvap,
            ocp=ocp,
            tend=limited,
            tendencies=tendencies,
        )
    packed_ready = (
        N_TENDENCY_LAYOUT_MODE == "producer_packed"
        and producer_packs is not None
        and all(torch.is_tensor(pack) for pack in producer_packs)
    )
    tendency_inputs = (
        (*producer_packs, *(tend[name] for name in _N_PACKED_OTHER_FIELDS))
        if packed_ready
        else tuple(tend[name] for name in _SUM_TENDENCY_FIELDS)
    )
    compile_args = (
        rho,
        rho,
        temp,
        idx,
        lvap,
        ocp,
        qvsi,
        qv,
        l_qc,
        rc,
        l_qi,
        ri,
        l_qr,
        rr,
        l_qs,
        rs,
        l_qg,
        rg,
        _tensor(_STATIC_GRAUPEL_RHO_G, temp),
        float(odt),
        *tendency_inputs,
        *(tendencies[name] for name in _SUM_TENDENCY_OUTPUT_FIELDS),
    )
    if packed_ready:
        values = _run_tensor_compile_island(
            "limit_and_sum_tendencies_packed", _limit_and_sum_tendencies_packed_kernel, *compile_args
        )
    else:
        values = _run_tensor_compile_island(
            "limit_and_sum_tendencies", _limit_and_sum_tendencies_kernel, *compile_args
        )
    split = len(_OVER_DEPLETION_OUTPUT_FIELDS)
    limited = dict(tend)
    limited.update(zip(_OVER_DEPLETION_OUTPUT_FIELDS, values[:split], strict=True))
    summed = dict(tendencies)
    summed.update(zip(_SUM_TENDENCY_OUTPUT_FIELDS, values[split:], strict=True))
    return limited, summed


def _limit_and_sum_tendencies_packed_kernel(*args: Any) -> tuple[torch.Tensor, ...]:
    prefix = args[:19]
    ice_pack, riming_pack, melting_pack = args[19:22]
    remaining = args[22:]
    sources = (ice_pack, riming_pack, melting_pack, remaining)
    tend_values = tuple(sources[group][index] for group, index in _N_PACKED_TENDENCY_SELECTORS)
    return _limit_and_sum_tendencies_kernel(
        *prefix,
        *tend_values,
        *remaining[len(_N_PACKED_OTHER_FIELDS):],
    )


def _limit_and_sum_tendencies_kernel(
    rho: torch.Tensor,
    temp: torch.Tensor,
    idx: torch.Tensor,
    lvap: torch.Tensor,
    ocp: torch.Tensor,
    qvsi: torch.Tensor,
    qv: torch.Tensor,
    l_qc: torch.Tensor,
    rc: torch.Tensor,
    l_qi: torch.Tensor,
    ri: torch.Tensor,
    l_qr: torch.Tensor,
    rr: torch.Tensor,
    l_qs: torch.Tensor,
    rs: torch.Tensor,
    l_qg: torch.Tensor,
    rg: torch.Tensor,
    graupel_density: torch.Tensor,
    odt: float,
    *values: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    tend_values = values[: len(_SUM_TENDENCY_FIELDS)]
    tendency_values = values[len(_SUM_TENDENCY_FIELDS) :]
    limited_values = _check_over_depletion_tendencies_kernel(
        rho,
        temp,
        qvsi,
        qv,
        l_qc,
        rc,
        l_qi,
        ri,
        l_qr,
        rr,
        l_qs,
        rs,
        l_qg,
        rg,
        odt,
        *(tend_values[index] for index in _OVER_DEPLETION_FROM_SUM_INDICES),
    )
    sum_values = tuple(
        limited_values[limited_index] if limited_index >= 0 else tend_values[index]
        for index, limited_index in enumerate(_SUM_LIMITED_OUTPUT_INDICES)
    )
    summed_values = _sum_tendencies_kernel(
        rho,
        temp,
        idx,
        lvap,
        ocp,
        graupel_density,
        *sum_values,
        *tendency_values,
    )
    return (*limited_values, *summed_values)


def graupel_check_and_update(*, rho: torch.Tensor, qg1d: torch.Tensor, qgten: torch.Tensor, ngten: torch.Tensor, qbten: torch.Tensor, dt: float, odt: float, ng1d: torch.Tensor | None=None, qb1d: torch.Tensor | None=None, _const_STATIC_GRAUPEL_AM_G=_STATIC_GRAUPEL_AM_G) -> dict[str, torch.Tensor]:
    if ng1d is not None and qb1d is not None:
        values = _run_tensor_compile_island('graupel_check_and_update_optional', _graupel_check_and_update_optional_kernel, rho, rho, qg1d, qgten, ngten, qbten, ng1d, qb1d, _tensor(_const_STATIC_GRAUPEL_AM_G, rho), float(dt), float(odt))
        names = ('l_qg', 'qg1d', 'ng1d', 'qb1d', 'rg', 'ng', 'rb', 'idx', 'qgten', 'ngten', 'qbten', 'ilamg', 'mvd_g')
        return dict(zip(names, values, strict=True))
    values = _run_tensor_compile_island('graupel_check_and_update_single', _graupel_check_and_update_single_kernel, rho, rho, qg1d, qgten, ngten, qbten, float(dt), float(odt))
    l_qg, qg1d_out, rg, ng, rb, idx, qgten_out, ngten_out, qbten_out, ilamg, mvd_g = values
    return {'l_qg': l_qg, 'qg1d': qg1d_out, 'ng1d': None, 'qb1d': None, 'rg': rg, 'ng': ng, 'rb': rb, 'idx': idx, 'qgten': qgten_out, 'ngten': ngten_out, 'qbten': qbten_out, 'ilamg': ilamg, 'mvd_g': mvd_g}


def _graupel_check_common_merge(active: torch.Tensor, qg_old: torch.Tensor, qg_active: torch.Tensor, rg_active: torch.Tensor, ng_active: torch.Tensor, rb_active: torch.Tensor, idx: torch.Tensor, qgten: torch.Tensor, ilamg_active: torch.Tensor, mvd_g_active: torch.Tensor, odt: float, _const_segment_g_IDX_BG1=constants.IDX_BG1, _const_segment_e_R1=constants.R1, _const_segment_e_R2=constants.R2, _const_segment_g_METERS3_TO_LITERS=constants.METERS3_TO_LITERS, _const_SCALAR_GRAUPEL_RHO_G_segment_g_IDX_BG1_1=_SCALAR_GRAUPEL_RHO_G[constants.IDX_BG1 - 1], _const_segment_g_D0G=constants.D0G) -> tuple[torch.Tensor, ...]:
    return (active, torch.where(active, qg_active, 0.0), torch.where(active, rg_active, float(_const_segment_e_R1)), torch.where(active, ng_active, float(_const_segment_e_R2)), torch.where(active, rb_active, float(_const_segment_g_METERS3_TO_LITERS * _const_segment_e_R1 / _const_SCALAR_GRAUPEL_RHO_G_segment_g_IDX_BG1_1)), torch.where(active, idx, int(_const_segment_g_IDX_BG1)), torch.where(active, qgten, -qg_old * odt), torch.where(active, ilamg_active, 0.0), torch.where(active, mvd_g_active, float(_const_segment_g_D0G)))


def _density_index_no_cache(value: torch.Tensor, _const_segment_g_NRHG=constants.NRHG) -> torch.Tensor:
    return torch.clamp(torch.floor(value + 0.5).to(torch.int64) + 1, 1, _const_segment_g_NRHG)


def _graupel_check_and_update_optional_kernel(rho: torch.Tensor, qg_old: torch.Tensor, qgten: torch.Tensor, ngten: torch.Tensor, qbten: torch.Tensor, ng_old: torch.Tensor, qb_old: torch.Tensor, am_g: torch.Tensor, dt: float, odt: float, _const_segment_e_R1=constants.R1, _const_segment_e_R2=constants.R2, _const_segment_g_METERS3_TO_LITERS=constants.METERS3_TO_LITERS, _const_SCALAR_GRAUPEL_RHO_G_segment_g_NRHG_1=_SCALAR_GRAUPEL_RHO_G[constants.NRHG - 1], _const_SCALAR_GRAUPEL_RHO_G_0=_SCALAR_GRAUPEL_RHO_G[0], _const_segment_g_MU_G=constants.MU_G, _const_SCALAR_GRAUPEL_CGG_1_0=_SCALAR_GRAUPEL_CGG[1][0], _const_segment_g_OGG3=constants.OGG3, _const_SCALAR_GRAUPEL_CGG_2_0=_SCALAR_GRAUPEL_CGG[2][0], _const_segment_g_OGG2=constants.OGG2, _const_segment_g_OBMG=constants.OBMG, _const_segment_e_D0R=constants.D0R) -> tuple[torch.Tensor, ...]:
    active = qg_old + qgten * dt > float(_const_segment_e_R1)
    qg_active = qg_old + qgten * dt
    rg_active = qg_active * rho
    rg_safe = _clamp_tiny(rg_active)
    ng0 = torch.clamp_min((ng_old + ngten * dt) * rho, float(_const_segment_e_R2))
    rb_lower = rg_safe * float(_const_segment_g_METERS3_TO_LITERS / _const_SCALAR_GRAUPEL_RHO_G_segment_g_NRHG_1)
    rb_upper = rg_safe * float(_const_segment_g_METERS3_TO_LITERS / _const_SCALAR_GRAUPEL_RHO_G_0)
    rb_active = torch.minimum(torch.maximum((qb_old + qbten * dt) * rho, rb_lower), rb_upper)
    idx_initial = _density_index_no_cache(10.0 * rg_safe / _clamp_tiny(rb_active))
    am_initial = am_g[idx_initial - 1]
    hit0 = ng0 <= float(_const_segment_e_R2)
    lam_hit0 = float(3.0 + _const_segment_g_MU_G + 0.672) / 0.0015
    ng_hit0 = float(_const_SCALAR_GRAUPEL_CGG_1_0 * _const_segment_g_OGG3) * rg_safe * lam_hit0 ** 3.0 / am_initial
    ng1 = torch.where(hit0, ng_hit0, ng0)
    lam1 = (am_initial * float(_const_SCALAR_GRAUPEL_CGG_2_0 * _const_segment_g_OGG2) * ng1 / rg_safe) ** float(_const_segment_g_OBMG)
    mvd1 = float(3.0 + _const_segment_g_MU_G + 0.672) / lam1
    high = mvd1 > 0.0254
    low = ~high & (mvd1 < float(_const_segment_e_D0R))
    mvd_limit = torch.clamp(mvd1, min=float(_const_segment_e_D0R), max=0.0254)
    lam_limit = float(3.0 + _const_segment_g_MU_G + 0.672) / mvd_limit
    ng_limit = float(_const_SCALAR_GRAUPEL_CGG_1_0 * _const_segment_g_OGG3) * rg_safe * _libm_power(lam_limit, 3.0) / am_initial
    hit_limit = hit0 | high | low
    ng_active = torch.where(hit_limit, ng_limit, ng1)
    lamg = torch.where(hit_limit, lam_limit, lam1)
    mvd_g_active = torch.where(hit_limit, mvd_limit, mvd1)
    ngten_active = torch.where(hit_limit, (ng_limit / rho - ng_old) * odt, ngten)
    ng1d_active = float(_const_SCALAR_GRAUPEL_CGG_1_0 * _const_segment_g_OGG3) * qg_active * _libm_power(lamg, 3.0) / am_initial
    qb1d_active = torch.minimum(torch.maximum(qb_old + qbten * dt, qg_active * float(_const_segment_g_METERS3_TO_LITERS / _const_SCALAR_GRAUPEL_RHO_G_segment_g_NRHG_1)), qg_active * float(_const_segment_g_METERS3_TO_LITERS / _const_SCALAR_GRAUPEL_RHO_G_0))
    idx = _density_index_no_cache(10.0 * _clamp_tiny(qg_active) / _clamp_tiny(qb1d_active))
    ilamg_active = 1.0 / lamg
    l_qg, qg1d_out, rg, ng, rb, idx_out, qgten_out, ilamg, mvd_g = _graupel_check_common_merge(active, qg_old, qg_active, rg_active, ng_active, rb_active, idx, qgten, ilamg_active, mvd_g_active, odt)
    return (l_qg, qg1d_out, torch.where(active, ng1d_active, 0.0), torch.where(active, qb1d_active, 0.0), rg, ng, rb, idx_out, qgten_out, torch.where(active, ngten_active, -ng_old * odt), torch.where(active, qbten, -qb_old * odt), ilamg, mvd_g)


def _graupel_check_and_update_single_kernel(rho: torch.Tensor, qg_old: torch.Tensor, qgten: torch.Tensor, ngten: torch.Tensor, qbten: torch.Tensor, dt: float, odt: float, _const_segment_e_R1=constants.R1, _const_segment_g_IDX_BG1=constants.IDX_BG1, _const_segment_g_GONV_MIN=constants.GONV_MIN, _const_segment_g_GONV_MAX=constants.GONV_MAX, _const_SCALAR_GRAUPEL_AM_G_segment_g_IDX_BG1_1=_SCALAR_GRAUPEL_AM_G[constants.IDX_BG1 - 1], _const_SCALAR_GRAUPEL_CGG_0_0=_SCALAR_GRAUPEL_CGG[0][0], _const_segment_g_OGE1=constants.OGE1, _const_SCALAR_GRAUPEL_SINGLE_MOMENT_LAMG_SCALE=_SCALAR_GRAUPEL_SINGLE_MOMENT_LAMG_SCALE, _const_SCALAR_GRAUPEL_CGG_1_0=_SCALAR_GRAUPEL_CGG[1][0], _const_segment_g_OGG3=constants.OGG3, _const_segment_g_METERS3_TO_LITERS=constants.METERS3_TO_LITERS, _const_SCALAR_GRAUPEL_RHO_G_segment_g_IDX_BG1_1=_SCALAR_GRAUPEL_RHO_G[constants.IDX_BG1 - 1], _const_segment_g_MU_G=constants.MU_G) -> tuple[torch.Tensor, ...]:
    active = qg_old + qgten * dt > float(_const_segment_e_R1)
    qg_active = qg_old + qgten * dt
    rg_active = qg_active * rho
    rg_safe = torch.where(active, rg_active, 1.0)
    idx = torch.full(qg_old.shape, _const_segment_g_IDX_BG1, dtype=torch.int64, device=qg_old.device)
    ygra1 = torch.log10(torch.clamp_min(rg_active, 1e-09))
    zans1 = 3.4 + 2.0 / 7.0 * (ygra1 + 8.0)
    n0_exp = torch.clamp(torch.pow(10.0, zans1), min=float(_const_segment_g_GONV_MIN), max=float(_const_segment_g_GONV_MAX))
    am = float(_const_SCALAR_GRAUPEL_AM_G_segment_g_IDX_BG1_1)
    lam_exp = (n0_exp * am * float(_const_SCALAR_GRAUPEL_CGG_0_0) / rg_safe) ** float(_const_segment_g_OGE1)
    lamg = lam_exp * float(_const_SCALAR_GRAUPEL_SINGLE_MOMENT_LAMG_SCALE)
    # BM_G is a real source exponent; an integer cube changes the number moment.
    ng_active = float(_const_SCALAR_GRAUPEL_CGG_1_0 * _const_segment_g_OGG3) * rg_active * torch.pow(lamg, torch.full_like(lamg, 3.0)) / am
    rb_active = float(_const_segment_g_METERS3_TO_LITERS / _const_SCALAR_GRAUPEL_RHO_G_segment_g_IDX_BG1_1) * rg_active
    ilamg_active = 1.0 / lamg
    mvd_g_active = float(3.0 + _const_segment_g_MU_G + 0.672) * ilamg_active
    l_qg, qg1d_out, rg, ng, rb, idx_out, qgten_out, ilamg, mvd_g = _graupel_check_common_merge(active, qg_old, qg_active, rg_active, ng_active, rb_active, idx, qgten, ilamg_active, mvd_g_active, odt)
    return (l_qg, qg1d_out, rg, ng, rb, idx_out, qgten_out, ngten, qbten, ilamg, mvd_g)


def _cloud_graupel_check_and_update_optional_kernel(
    rho: torch.Tensor,
    qc1d: torch.Tensor,
    nc: torch.Tensor,
    qcten: torch.Tensor,
    ncten: torch.Tensor,
    nc1d: torch.Tensor,
    qg1d: torch.Tensor,
    qgten: torch.Tensor,
    ngten: torch.Tensor,
    qbten: torch.Tensor,
    ng1d: torch.Tensor,
    qb1d: torch.Tensor,
    cce1_table: torch.Tensor,
    ccg0_table: torch.Tensor,
    ocg2_table: torch.Tensor,
    ccg1_table: torch.Tensor,
    ocg1_table: torch.Tensor,
    am_g: torch.Tensor,
    dt: float,
    odt: float,
) -> tuple[torch.Tensor, ...]:
    cloud = _cloud_check_and_update_optional_kernel(
        rho,
        qc1d,
        nc,
        qcten,
        ncten,
        nc1d,
        cce1_table,
        ccg0_table,
        ocg2_table,
        ccg1_table,
        ocg1_table,
        dt,
        odt,
    )
    graupel = _graupel_check_and_update_optional_kernel(
        rho,
        qg1d,
        qgten,
        ngten,
        qbten,
        ng1d,
        qb1d,
        am_g,
        dt,
        odt,
    )
    return (*cloud, *graupel)


def cloud_graupel_check_and_update(
    *,
    rho: torch.Tensor,
    qc1d: torch.Tensor,
    nc: torch.Tensor,
    qcten: torch.Tensor,
    ncten: torch.Tensor,
    nc1d: torch.Tensor | None,
    ncsave: torch.Tensor | None,
    qg1d: torch.Tensor,
    qgten: torch.Tensor,
    ngten: torch.Tensor,
    qbten: torch.Tensor,
    ng1d: torch.Tensor | None,
    qb1d: torch.Tensor | None,
    dt: float,
    odt: float,
) -> tuple[dict[str, torch.Tensor | None], dict[str, torch.Tensor | None]]:
    if nc1d is None or ng1d is None or qb1d is None:
        return (
            cloud_check_and_update(
                rho=rho,
                qc1d=qc1d,
                nc=nc,
                qcten=qcten,
                ncten=ncten,
                nc1d=nc1d,
                ncsave=ncsave,
                dt=dt,
                odt=odt,
            ),
            graupel_check_and_update(
                rho=rho,
                qg1d=qg1d,
                qgten=qgten,
                ngten=ngten,
                qbten=qbten,
                ng1d=ng1d,
                qb1d=qb1d,
                dt=dt,
                odt=odt,
            ),
        )
    values = _run_tensor_compile_island(
        "cloud_graupel_check_and_update_optional",
        _cloud_graupel_check_and_update_optional_kernel,
        rho,
        rho,
        qc1d,
        nc,
        qcten,
        ncten,
        nc1d,
        qg1d,
        qgten,
        ngten,
        qbten,
        ng1d,
        qb1d,
        _tensor(_STATIC_CLOUD_CCE[1], rho),
        _tensor(_STATIC_CLOUD_CCG[0], rho),
        _tensor(_STATIC_CLOUD_OCG2, rho),
        _tensor(_STATIC_CLOUD_CCG[1], rho),
        _tensor(_STATIC_CLOUD_OCG1, rho),
        _tensor(_STATIC_GRAUPEL_AM_G, rho),
        float(dt),
        float(odt),
    )
    cloud_names = ("l_qc", "qc1d", "nc1d", "rc", "nc", "qcten", "ncten", "ilamc", "mvd_c")
    graupel_names = ("l_qg", "qg1d", "ng1d", "qb1d", "rg", "ng", "rb", "idx", "qgten", "ngten", "qbten", "ilamg", "mvd_g")
    cloud_count = len(cloud_names)
    return (
        dict(zip(cloud_names, values[:cloud_count], strict=True)),
        dict(zip(graupel_names, values[cloud_count:], strict=True)),
    )


def _cloud_condensation_newton_update(
    clap: torch.Tensor,
    *,
    qv: torch.Tensor,
    qvs: torch.Tensor,
    lvt2: torch.Tensor,
) -> torch.Tensor:
    exp_value = torch.exp(lvt2 * clap)
    fcd = qvs * exp_value - qv + clap
    dfcd = qvs * lvt2 * exp_value + 1.0
    return clap - fcd / dfcd


def cloud_condensation_step_tensor(
    *,
    rho: torch.Tensor,
    temp: torch.Tensor,
    w1d: torch.Tensor,
    ssatw: torch.Tensor,
    lvap: torch.Tensor,
    tcond: torch.Tensor,
    diffu: torch.Tensor,
    lvt2: torch.Tensor,
    nwfa1d: torch.Tensor | None,
    nwfa: torch.Tensor,
    ncsave: torch.Tensor | None,
    qv: torch.Tensor,
    qvs: torch.Tensor,
    l_qc: torch.Tensor,
    rc: torch.Tensor,
    nc: torch.Tensor,
    tend: Mapping[str, torch.Tensor],
    tables: Any,
    dt: float,
    odt: float,
    idx_d: torch.Tensor | None = None,
    idx_c: torch.Tensor | None = None,
    idx_n: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    updated = dict(tend)
    idx_d_values = _int_scalar(0, temp) if idx_d is None else idx_d
    idx_c_values = _int_scalar(0, temp) if idx_c is None else idx_c
    idx_n_values = _int_scalar(0, temp) if idx_n is None else idx_n
    compile_activation = (
        nwfa1d is not None
        and TENSOR_COMPILE_ISLAND_CUDA
        and temp.device.type in TENSOR_COMPILE_ISLAND_DEVICES
        and not torch.compiler.is_compiling()
    )
    leading_args = (
        rho,
        temp,
        ssatw,
        lvap,
        tcond,
        diffu,
        lvt2,
        qv,
        qvs,
        l_qc,
        rc,
        nc,
        updated["prw_vcd"],
        updated["pnc_wcd"],
    )
    trailing_args = (
        idx_d_values,
        idx_c_values,
        idx_n_values,
        _tensor(_STATIC_DROP_EVAP_NUMBER_TABLE, temp),
        _tensor(_STATIC_CLOUD_NUMBER_BINS, temp),
        float(dt),
        float(odt),
    )
    if compile_activation:
        values = _run_tensor_compile_island(
            "cloud_activation_condensation_step",
            _cloud_activation_condensation_step_kernel,
            rho,
            *leading_args,
            w1d,
            nwfa,
            _tensor(_STATIC_CLOUD_ACTIVATION_NA_AXIS, temp),
            _tensor(_STATIC_CLOUD_ACTIVATION_W_AXIS, temp),
            _tensor(_STATIC_CLOUD_ACTIVATION_T_AXIS, temp),
            _tensor(_activation_table_array(tables), temp),
            *trailing_args,
        )
    else:
        if nwfa1d is not None:
            xnc = torch.clamp_min(
                activate_cloud_number_tensor(temp, w1d, nwfa, tables),
                float(constants.NT_C_MIN),
            )
        elif ncsave is not None:
            xnc = ncsave
        else:
            xnc = _scalar(float(constants.NT_C_L), nc)
        values = _run_tensor_compile_island(
            "cloud_condensation_step",
            _cloud_condensation_step_kernel,
            rho,
            *leading_args,
            xnc,
            *trailing_args,
        )
    prw_vcd, pnc_wcd, idx_d_values, idx_c_values, idx_n_values = values
    updated["prw_vcd"] = prw_vcd
    updated["pnc_wcd"] = pnc_wcd
    return updated, idx_d_values, idx_c_values, idx_n_values


def _activate_cloud_number_no_cache(
    temp: torch.Tensor,
    w1d: torch.Tensor,
    nwfa: torch.Tensor,
    ta_na: torch.Tensor,
    ta_ww: torch.Tensor,
    ta_tk: torch.Tensor,
    activation_table: torch.Tensor,
) -> torch.Tensor:
    n_local = nwfa * 1.0e-6
    n_local = torch.where(
        n_local >= ta_na[-1],
        ta_na[-1] - 1.0,
        torch.where(n_local <= ta_na[0], ta_na[0] + 1.0, n_local),
    )
    i = torch.clamp(
        _bucketize_right_static_axis_tensor(n_local, ta_na).to(torch.int64) + 1,
        2,
        ta_na.shape[0],
    )
    x1 = torch.log(ta_na[i - 2])
    x2 = torch.log(ta_na[i - 1])

    w_local = torch.where(
        w1d >= ta_ww[-1],
        ta_ww[-1] - 1.0,
        torch.where(w1d <= ta_ww[0], ta_ww[0] + 0.001, w1d),
    )
    j = torch.clamp(
        _bucketize_right_static_axis_tensor(w_local, ta_ww).to(torch.int64) + 1,
        2,
        ta_ww.shape[0],
    )
    y1 = torch.log(ta_ww[j - 2])
    y2 = torch.log(ta_ww[j - 1])

    temperature_index = (temp - ta_tk[0]) * 0.1
    temperature_index = torch.where(
        temperature_index >= 0.0,
        torch.floor(temperature_index + 0.5),
        torch.ceil(temperature_index - 0.5),
    ).to(torch.int64)
    k = torch.clamp(temperature_index + 1, 1, ta_tk.shape[0])
    a = activation_table[i - 2, j - 2, k - 1, 2, 1]
    b = activation_table[i - 1, j - 2, k - 1, 2, 1]
    c = activation_table[i - 1, j - 1, k - 1, 2, 1]
    d = activation_table[i - 2, j - 1, k - 1, 2, 1]
    nx = torch.log(n_local)
    wy = torch.log(w_local)
    tx = (nx - x1) / (x2 - x1)
    uy = (wy - y1) / (y2 - y1)
    fraction = (
        (1.0 - tx) * (1.0 - uy) * a
        + tx * (1.0 - uy) * b
        + tx * uy * c
        + (1.0 - tx) * uy * d
    )
    return nwfa * fraction


def _cloud_activation_condensation_step_kernel(
    rho: torch.Tensor,
    temp: torch.Tensor,
    ssatw: torch.Tensor,
    lvap: torch.Tensor,
    tcond: torch.Tensor,
    diffu: torch.Tensor,
    lvt2: torch.Tensor,
    qv: torch.Tensor,
    qvs: torch.Tensor,
    l_qc: torch.Tensor,
    rc: torch.Tensor,
    nc: torch.Tensor,
    prw_vcd: torch.Tensor,
    pnc_wcd: torch.Tensor,
    w1d: torch.Tensor,
    nwfa: torch.Tensor,
    ta_na: torch.Tensor,
    ta_ww: torch.Tensor,
    ta_tk: torch.Tensor,
    activation_table: torch.Tensor,
    idx_d_values: torch.Tensor,
    idx_c_values: torch.Tensor,
    idx_n_values: torch.Tensor,
    drop_table: torch.Tensor,
    cloud_number_bins: torch.Tensor,
    dt: float,
    odt: float,
    _nt_c_min: float = float(constants.NT_C_MIN),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xnc = torch.clamp_min(
        _activate_cloud_number_no_cache(temp, w1d, nwfa, ta_na, ta_ww, ta_tk, activation_table),
        _nt_c_min,
    )
    return _cloud_condensation_step_kernel(
        rho,
        temp,
        ssatw,
        lvap,
        tcond,
        diffu,
        lvt2,
        qv,
        qvs,
        l_qc,
        rc,
        nc,
        prw_vcd,
        pnc_wcd,
        xnc,
        idx_d_values,
        idx_c_values,
        idx_n_values,
        drop_table,
        cloud_number_bins,
        dt,
        odt,
    )


def _mass_table_index_no_cache(mass: torch.Tensor, exponent_offset: int, upper: int) -> torch.Tensor:
    safe = _clamp_tiny(mass)
    exponent = torch.floor(torch.log10(safe))
    exponent_i = exponent.to(torch.int64)
    mantissa = safe / torch.pow(10.0, exponent)
    digit = torch.floor(mantissa).to(torch.int64)
    index = digit + 9 * (exponent_i - int(exponent_offset))
    return torch.clamp(index, 1, int(upper))


def _cloud_table_indices_no_cache(rc: torch.Tensor, nc: torch.Tensor, cloud_number_bins: torch.Tensor, _const_segment_i_NIC2=constants.NIC2, _const_segment_i_NTB_C=constants.NTB_C, _const_segment_i_NBC=constants.NBC) -> tuple[torch.Tensor, torch.Tensor]:
    idx_c = _mass_table_index_no_cache(rc, _const_segment_i_NIC2, _const_segment_i_NTB_C)
    nc_safe = torch.clamp_min(nc, cloud_number_bins[0])
    n_scaled = 1.0 + float(_const_segment_i_NBC) * torch.log(nc_safe / cloud_number_bins[0]) / torch.log(cloud_number_bins[-1] / cloud_number_bins[0])
    idx_n = torch.where(n_scaled >= 0.0, torch.floor(n_scaled + 0.5), torch.ceil(n_scaled - 0.5)).to(torch.int64)
    return (idx_c, torch.clamp(idx_n, 1, _const_segment_i_NBC))


def _cloud_condensation_step_kernel(rho: torch.Tensor, temp: torch.Tensor, ssatw: torch.Tensor, lvap: torch.Tensor, tcond: torch.Tensor, diffu: torch.Tensor, lvt2: torch.Tensor, qv: torch.Tensor, qvs: torch.Tensor, l_qc: torch.Tensor, rc: torch.Tensor, nc: torch.Tensor, prw_vcd: torch.Tensor, pnc_wcd: torch.Tensor, xnc: torch.Tensor, idx_d_values: torch.Tensor, idx_c_values: torch.Tensor, idx_n_values: torch.Tensor, drop_table: torch.Tensor, cloud_number_bins: torch.Tensor, dt: float, odt: float, _const_segment_g_EPS=constants.EPS, _const_segment_e_R1=constants.R1, _const_segment_g_ORV=constants.ORV, _const_segment_e_PI=constants.PI, _const_segment_e_RHO_W=constants.RHO_W, _const_segment_h_NBC=constants.NBC) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    orho = 1.0 / rho
    active = torch.abs(ssatw) >= float(_const_segment_g_EPS)
    clap = (qv - qvs) / (1.0 + lvt2 * qvs)
    clap = _cloud_condensation_newton_update(clap, qv=qv, qvs=qvs, lvt2=lvt2)
    clap = _cloud_condensation_newton_update(clap, qv=qv, qvs=qvs, lvt2=lvt2)
    clap = _cloud_condensation_newton_update(clap, qv=qv, qvs=qvs, lvt2=lvt2)
    xrc = rc + clap * rho
    has_cloud_mass = active & (xrc > float(_const_segment_e_R1))
    lost_cloud_mass = active & ~has_cloud_mass
    prw_vcd = torch.where(has_cloud_mass, clap * odt, prw_vcd)
    activating = has_cloud_mass & (clap > float(_const_segment_g_EPS))
    pnc_activation = 0.5 * (xnc - nc + torch.abs(xnc - nc)) * odt * orho
    pnc_wcd = torch.where(activating, pnc_activation, pnc_wcd)
    evaporating = has_cloud_mass & ~activating & l_qc & (ssatw < -1e-06) & (clap < -float(_const_segment_g_EPS))
    otemp = 1.0 / temp
    rvs = rho * qvs
    thermal_slope = lvap * otemp * float(_const_segment_g_ORV) - 1.0
    rvs_p = rvs * otemp * thermal_slope
    rvs_pp = rvs * (otemp * thermal_slope * otemp * thermal_slope + -(2.0 * float(_const_segment_g_ORV)) * lvap * otemp * otemp * otemp + otemp * otemp)
    gamsc = lvap * diffu / tcond * rvs_p
    alphsc = 0.5 * (gamsc / (1.0 + gamsc)) * (gamsc / (1.0 + gamsc)) * rvs_pp / rvs_p * rvs / rvs_p
    alphsc = torch.clamp_min(alphsc, 1e-09)
    xsat = torch.where(torch.abs(ssatw) < 1e-09, 0.0, ssatw)
    t1_evap = float(2.0 * _const_segment_e_PI) * (1.0 - alphsc * xsat + 2.0 * alphsc * alphsc * xsat * xsat - 5.0 * alphsc * alphsc * alphsc * xsat * xsat * xsat) / (1.0 + gamsc)
    dc_star_arg = -2.0 * dt * t1_evap / float(2.0 * _const_segment_e_PI) * 4.0 * diffu * ssatw * rvs / float(_const_segment_e_RHO_W)
    dc_star = torch.sqrt(torch.clamp_min(dc_star_arg, 0.0))
    idx_d_evap = torch.clamp(torch.floor(dc_star * 1000000.0).to(torch.int64), 1, _const_segment_h_NBC)
    idx_c_evap, idx_n_evap = _cloud_table_indices_no_cache(rc, nc, cloud_number_bins)
    idx_d_values = torch.where(evaporating, idx_d_evap, idx_d_values)
    idx_c_values = torch.where(evaporating, idx_c_evap, idx_c_values)
    idx_n_values = torch.where(evaporating, idx_n_evap, idx_n_values)
    tnc_wev = drop_table[idx_d_evap - 1, idx_c_evap - 1, idx_n_evap - 1]
    prw_evap = torch.maximum(-rc * 0.99 * orho * odt, prw_vcd)
    pnc_evap = torch.maximum(-nc * 0.99 * orho * odt, -tnc_wev * odt * orho)
    prw_vcd = torch.where(evaporating, prw_evap, prw_vcd)
    pnc_wcd = torch.where(evaporating, pnc_evap, pnc_wcd)
    prw_vcd = torch.where(lost_cloud_mass, -rc * orho * odt, prw_vcd)
    pnc_wcd = torch.where(lost_cloud_mass, -nc * orho * odt, pnc_wcd)
    return (prw_vcd, pnc_wcd, idx_d_values, idx_c_values, idx_n_values)


def activate_cloud_number_tensor(
    temp: torch.Tensor,
    w1d: torch.Tensor,
    nwfa: torch.Tensor,
    tables: Any,
    land: torch.Tensor | None = None,
) -> torch.Tensor:
    n_local = nwfa * _scalar(1.0e-6, nwfa)
    ta_na = _tensor(_STATIC_CLOUD_ACTIVATION_NA_AXIS, nwfa)
    n_local = torch.where(
        n_local >= ta_na[-1],
        ta_na[-1] - _scalar(1.0, nwfa),
        torch.where(n_local <= ta_na[0], ta_na[0] + _scalar(1.0, nwfa), n_local),
    )
    i = _bucketize_right_static_axis_tensor(n_local, ta_na) + 1
    i = torch.clamp(i.to(torch.int64), 2, int(ta_na.numel()))
    x1 = torch.log(ta_na[i - 2])
    x2 = torch.log(ta_na[i - 1])

    w_local = w1d
    ta_ww = _tensor(_STATIC_CLOUD_ACTIVATION_W_AXIS, nwfa)
    w_local = torch.where(
        w_local >= ta_ww[-1],
        ta_ww[-1] - _scalar(1.0, nwfa),
        torch.where(w_local <= ta_ww[0], ta_ww[0] + _scalar(0.001, nwfa), w_local),
    )
    j = _bucketize_right_static_axis_tensor(w_local, ta_ww) + 1
    j = torch.clamp(j.to(torch.int64), 2, int(ta_ww.numel()))
    y1 = torch.log(ta_ww[j - 2])
    y2 = torch.log(ta_ww[j - 1])

    ta_tk = _tensor(_STATIC_CLOUD_ACTIVATION_T_AXIS, nwfa)
    k = torch.clamp(_fortran_nint_tensor((temp - ta_tk[0]) * _scalar(0.1, temp)) + 1, 1, int(ta_tk.numel()))
    l = _int_scalar(3, temp)
    m = _int_scalar(2, temp)
    table = _activation_table_array(tables)
    a = _table_gather_array(table, temp, i - 1, j - 1, k, l, m)
    b = _table_gather_array(table, temp, i, j - 1, k, l, m)
    c = _table_gather_array(table, temp, i, j, k, l, m)
    d = _table_gather_array(table, temp, i - 1, j, k, l, m)
    nx = torch.log(n_local)
    wy = torch.log(w_local)
    tx = (nx - x1) / (x2 - x1)
    uy = (wy - y1) / (y2 - y1)
    fraction = (1.0 - tx) * (1.0 - uy) * a + tx * (1.0 - uy) * b + tx * uy * c + (1.0 - tx) * uy * d
    if land is not None:
        lower = torch.where(land != 1, _scalar(0.15, fraction), _scalar(0.0, fraction))
        fraction = torch.maximum(fraction, lower)
    return nwfa * fraction


def rain_evaporation_tensor(
    *,
    rho: torch.Tensor,
    temp: torch.Tensor,
    ssatw: torch.Tensor,
    lvap: torch.Tensor,
    tcond: torch.Tensor,
    diffu: torch.Tensor,
    vsc2: torch.Tensor,
    rhof2: torch.Tensor,
    qv: torch.Tensor,
    qvs: torch.Tensor,
    l_qr: torch.Tensor,
    rr: torch.Tensor,
    nr: torch.Tensor,
    ilamr: torch.Tensor,
    tend: Mapping[str, torch.Tensor],
    odt: float,
) -> dict[str, torch.Tensor]:
    updated = dict(tend)
    if TENSOR_COMPILE_ISLAND_CUDA and rho.device.type in TENSOR_COMPILE_ISLAND_DEVICES:
        prv_rev, pnr_rev = _run_tensor_compile_island(
            "rain_evaporation",
            _rain_evaporation_kernel,
            rho,
            rho,
            temp,
            ssatw,
            lvap,
            tcond,
            diffu,
            vsc2,
            rhof2,
            qv,
            qvs,
            l_qr,
            rr,
            nr,
            ilamr,
            updated["prw_vcd"],
            updated["prr_gml"],
            updated["prv_rev"],
            updated["pnr_rev"],
            float(odt),
        )
        updated["prv_rev"] = prv_rev
        updated["pnr_rev"] = pnr_rev
        return updated
    active = l_qr & (ssatw < _scalar(-constants.EPS, ssatw)) & (updated["prw_vcd"] <= 0.0)
    orho = 1.0 / rho
    tempc = temp - _scalar(constants.T0, temp)
    otemp = 1.0 / temp
    rvs = rho * qvs
    thermal_slope = lvap * otemp * _scalar(constants.ORV, temp) - 1.0
    rvs_p = rvs * otemp * thermal_slope
    rvs_pp = rvs * (
        otemp * thermal_slope * otemp * thermal_slope
        + (-2.0 * lvap * otemp * otemp * otemp * _scalar(constants.ORV, temp))
        + otemp * otemp
    )
    gamsc = lvap * diffu / tcond * rvs_p
    alphsc = 0.5 * (gamsc / (1.0 + gamsc)) * (gamsc / (1.0 + gamsc)) * rvs_pp / rvs_p * rvs / rvs_p
    alphsc = torch.clamp_min(alphsc, _scalar(1.0e-9, alphsc))
    xsat = _min_first(_scalar(-1.0e-9, ssatw), ssatw)
    t1_evap = (
        _scalar(2.0 * constants.PI, temp)
        * (
            1.0
            - alphsc * xsat
            + _scalar(2.0, temp) * alphsc * alphsc * xsat * xsat
            - _scalar(5.0, temp) * alphsc * alphsc * alphsc * xsat * xsat * xsat
        )
        / (1.0 + gamsc)
    )

    low_humidity = (qv / qvs < _scalar(0.95, qv)) & (rr * orho <= _scalar(1.0e-8, rr))
    prv_low = rr * orho * _scalar(odt, rr)
    ilamr_safe = torch.where(ilamr > 0.0, ilamr, _scalar(1.0, ilamr))
    lamr = 1.0 / ilamr_safe
    n0_r = nr * _scalar(constants.ORG2, nr) * lamr
    prv_general = (
        _scalar(constants.T1_QR_EV, temp)
        * torch.pow(ilamr_safe, torch.full_like(ilamr_safe, float(_SCALAR_CRE[9])))
        + _scalar(constants.T2_QR_EV, temp)
        * vsc2
        * rhof2
        * ((lamr + _scalar(0.5 * constants.FV_R, temp)) ** _scalar(-_SCALAR_CRE[10], temp))
    )
    prv_general = t1_evap * diffu * (-ssatw) * n0_r * rvs * prv_general
    rate_max = torch.minimum(rr * orho * _scalar(odt, rr), (qvs - qv) * _scalar(odt, qv))
    prv_general = torch.minimum(rate_max, prv_general * orho)
    eva_factor = _min_first(
        _scalar(1.0, temp),
        _scalar(0.01, temp) + _scalar(0.99 - 0.01, temp) * (tempc / _scalar(20.0, temp)),
    )
    prv_general = torch.where(updated["prr_gml"] > 0.0, prv_general * eva_factor, prv_general)
    prv_rev = torch.where(low_humidity, prv_low, prv_general)
    prv_rev = torch.where(active, prv_rev, updated["prv_rev"])
    rr_safe = _clamp_tiny(rr)
    pnr_rev = torch.minimum(nr * 0.99 * orho * odt, prv_rev * nr / rr_safe)
    updated["prv_rev"] = prv_rev
    updated["pnr_rev"] = torch.where(active, pnr_rev, updated["pnr_rev"])
    return updated


def _rain_evaporation_kernel(
    rho: torch.Tensor,
    temp: torch.Tensor,
    ssatw: torch.Tensor,
    lvap: torch.Tensor,
    tcond: torch.Tensor,
    diffu: torch.Tensor,
    vsc2: torch.Tensor,
    rhof2: torch.Tensor,
    qv: torch.Tensor,
    qvs: torch.Tensor,
    l_qr: torch.Tensor,
    rr: torch.Tensor,
    nr: torch.Tensor,
    ilamr: torch.Tensor,
    prw_vcd: torch.Tensor,
    prr_gml: torch.Tensor,
    prv_rev_old: torch.Tensor,
    pnr_rev_old: torch.Tensor,
    odt: float,
    _const_segment_g_EPS=constants.EPS,
    _const_segment_e_T0=constants.T0,
    _const_segment_g_ORV=constants.ORV,
    _const_segment_e_PI=constants.PI,
    _const_segment_e_ORG2=constants.ORG2,
    _const_segment_o_T1_QR_EV=constants.T1_QR_EV,
    _const_segment_o_T2_QR_EV=constants.T2_QR_EV,
    _const_segment_h_FV_R=constants.FV_R,
    _const_SCALAR_CRE_9=_SCALAR_CRE[9],
    _const_SCALAR_CRE_10=_SCALAR_CRE[10],
) -> tuple[torch.Tensor, torch.Tensor]:
    active = l_qr & (ssatw < -float(_const_segment_g_EPS)) & (prw_vcd <= 0.0)
    orho = 1.0 / rho
    tempc = temp - float(_const_segment_e_T0)
    otemp = 1.0 / temp
    rvs = rho * qvs
    thermal_slope = lvap * otemp * float(_const_segment_g_ORV) - 1.0
    rvs_p = rvs * otemp * thermal_slope
    rvs_pp = rvs * (
        otemp * thermal_slope * otemp * thermal_slope
        + (-2.0 * lvap * otemp * otemp * otemp * float(_const_segment_g_ORV))
        + otemp * otemp
    )
    gamsc = lvap * diffu / tcond * rvs_p
    gamsc_ratio = gamsc / (1.0 + gamsc)
    alphsc = torch.clamp_min(0.5 * gamsc_ratio * gamsc_ratio * rvs_pp / rvs_p * rvs / rvs_p, 1.0e-9)
    xsat = torch.minimum(torch.full_like(ssatw, -1.0e-9), ssatw)
    t1_evap = (
        float(2.0 * _const_segment_e_PI)
        * (1.0 - alphsc * xsat + 2.0 * alphsc * alphsc * xsat * xsat
           - 5.0 * alphsc * alphsc * alphsc * xsat * xsat * xsat)
        / (1.0 + gamsc)
    )

    low_humidity = (qv / qvs < 0.95) & (rr * orho <= 1.0e-8)
    prv_low = rr * orho * odt
    ilamr_safe = torch.where(ilamr > 0.0, ilamr, 1.0)
    lamr = 1.0 / ilamr_safe
    n0_r = nr * float(_const_segment_e_ORG2) * lamr
    # Source CRE(10) is REAL; multiplication changes active evaporation by one ULP.
    prv_general = (
        float(_const_segment_o_T1_QR_EV)
        * torch.pow(ilamr_safe, torch.full_like(ilamr_safe, float(_const_SCALAR_CRE_9)))
        + float(_const_segment_o_T2_QR_EV)
        * vsc2
        * rhof2
        * ((lamr + float(0.5 * _const_segment_h_FV_R)) ** -float(_const_SCALAR_CRE_10))
    )
    prv_general = t1_evap * diffu * (-ssatw) * n0_r * rvs * prv_general
    rate_max = torch.minimum(rr * orho * odt, (qvs - qv) * odt)
    prv_general = torch.minimum(rate_max, prv_general * orho)
    eva_factor = torch.minimum(torch.ones_like(temp), 0.01 + 0.98 * (tempc / 20.0))
    prv_general = torch.where(prr_gml > 0.0, prv_general * eva_factor, prv_general)
    prv_rev = torch.where(low_humidity, prv_low, prv_general)
    prv_rev = torch.where(active, prv_rev, prv_rev_old)
    rr_safe = _clamp_tiny(rr)
    pnr_rev = torch.minimum(nr * 0.99 * orho * odt, prv_rev * nr / rr_safe)
    return prv_rev, torch.where(active, pnr_rev, pnr_rev_old)


def _sedimentation_motion_bounds(
    *,
    moving: torch.Tensor,
    speed: torch.Tensor,
    dz1d: torch.Tensor,
    dt: float,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ncol, nz = speed.shape
    moving_i64 = moving.to(torch.int64)
    levels = _int_arange(1, nz + 1, reference)[None, :]
    ktop = torch.clamp_min(torch.max(levels.expand(ncol, nz) * moving_i64, dim=1).values, _int_scalar(1, reference))
    speed_safe = _clamp_tiny(speed)
    substep_candidates = torch.trunc(_scalar(dt, reference) / (dz1d / speed_safe) + _scalar(1.0, reference)).to(torch.int64)
    substeps = torch.clamp_min(torch.max(substep_candidates * moving_i64, dim=1).values, _int_scalar(1, reference))
    ktop = torch.clamp_max(ktop, max(nz - 1, 0))
    return substeps, ktop


def _append_column_no_check(body: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    tail_column = tail[:, None] if tail.ndim == 1 else tail
    result = body.new_empty((body.shape[0], body.shape[1] + 1))
    result[:, :-1] = body
    result[:, -1:] = tail_column
    return result


def _append_zero_column_no_check(body: torch.Tensor) -> torch.Tensor:
    result = body.new_empty((body.shape[0], body.shape[1] + 1))
    result[:, :-1] = body
    result[:, -1] = 0.0
    return result


def _prepend_zero_column_no_check(body: torch.Tensor) -> torch.Tensor:
    result = body.new_empty((body.shape[0], body.shape[1] + 1))
    result[:, 0] = 0.0
    result[:, 1:] = body
    return result


def _suffix_fill_from_active_no_cache(
    formula: torch.Tensor,
    tail: torch.Tensor,
    active: torch.Tensor,
    zero_based_levels: torch.Tensor,
) -> torch.Tensor:
    ncol, nz = formula.shape
    levels = zero_based_levels.expand(ncol, nz)
    source_index = torch.where(active, levels, int(nz))
    fill_index = torch.flip(torch.cummin(torch.flip(source_index, dims=(1,)), dim=1).values, dims=(1,))
    extended = _append_column_no_check(formula, tail)
    return torch.gather(extended, 1, fill_index)


def _suffix_fill_from_active_zero_tail_no_cache(
    formula: torch.Tensor,
    active: torch.Tensor,
    zero_based_levels: torch.Tensor,
) -> torch.Tensor:
    ncol, nz = formula.shape
    levels = zero_based_levels.expand(ncol, nz)
    source_index = torch.where(active, levels, int(nz))
    fill_index = torch.flip(torch.cummin(torch.flip(source_index, dims=(1,)), dim=1).values, dims=(1,))
    extended = _append_zero_column_no_check(formula)
    return torch.gather(extended, 1, fill_index)


def _sedimentation_motion_bounds_no_cache(
    *,
    moving: torch.Tensor,
    speed: torch.Tensor,
    dz1d: torch.Tensor,
    dt: float,
    one_based_levels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ncol, nz = speed.shape
    moving_i64 = moving.to(torch.int64)
    ktop = torch.clamp_min(torch.max(one_based_levels.expand(ncol, nz) * moving_i64, dim=1).values, torch.ones((ncol,), dtype=torch.int64, device=speed.device))
    speed_safe = _clamp_tiny(speed)
    substep_candidates = torch.trunc(dt / (dz1d / speed_safe) + 1.0).to(torch.int64)
    substeps = torch.clamp_min(torch.max(substep_candidates * moving_i64, dim=1).values, torch.ones((ncol,), dtype=torch.int64, device=speed.device))
    ktop = torch.clamp_max(ktop, max(nz - 1, 0))
    return substeps, ktop


def rain_fallspeed_tensor(
    *,
    rhof: torch.Tensor,
    rr: torch.Tensor,
    ilamr: torch.Tensor,
    dz1d: torch.Tensor,
    vt: torch.Tensor,
    vtn: torch.Tensor,
    dt: float,
    compute_substeps: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nz = rr.shape[1]
    return _run_tensor_compile_island(
        "rain_fallspeed",
        _rain_fallspeed_kernel,
        rr,
        rhof,
        rr,
        ilamr,
        dz1d,
        vt,
        vtn,
        _int_arange(0, nz, rr)[None, :],
        _int_arange(1, nz + 1, rr)[None, :],
        float(dt),
        bool(compute_substeps),
    )


def _rain_fallspeed_kernel(rhof: torch.Tensor, rr: torch.Tensor, ilamr: torch.Tensor, dz1d: torch.Tensor, vt: torch.Tensor, vtn: torch.Tensor, zero_based_levels: torch.Tensor, one_based_levels: torch.Tensor, dt: float, compute_substeps: bool, _const_segment_e_R1=constants.R1, _const_segment_h_AV_R=constants.AV_R, _const_SCALAR_CRG_5=_SCALAR_CRG[5], _const_segment_e_ORG3=constants.ORG3, _const_SCALAR_CRE_2=_SCALAR_CRE[2], _const_segment_h_FV_R=constants.FV_R, _const_SCALAR_CRE_5=_SCALAR_CRE[5], _const_SCALAR_CRG_6=_SCALAR_CRG[6], _const_SCALAR_CRG_11=_SCALAR_CRG[11], _const_SCALAR_CRE_11=_SCALAR_CRE[11], _const_SCALAR_CRE_6=_SCALAR_CRE[6]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ncol, nz = rr.shape
    vt_tail = vt[:, nz]
    vtn_tail = vtn[:, nz]
    active = rr > float(_const_segment_e_R1)
    lamr = 1.0 / torch.where(ilamr > 0.0, ilamr, 1.0)
    vt_formula = rhof * float(_const_segment_h_AV_R) * _const_SCALAR_CRG_5 * float(_const_segment_e_ORG3) * lamr ** _const_SCALAR_CRE_2 * (lamr + float(_const_segment_h_FV_R)) ** (-_const_SCALAR_CRE_5)
    vtn_formula = rhof * float(_const_segment_h_AV_R) * _const_SCALAR_CRG_6 / _const_SCALAR_CRG_11 * lamr ** _const_SCALAR_CRE_11 * (lamr + float(_const_segment_h_FV_R)) ** (-_const_SCALAR_CRE_6)
    vt_body = _suffix_fill_from_active_no_cache(vt_formula, vt_tail, active, zero_based_levels)
    vtn_body = _suffix_fill_from_active_no_cache(vtn_formula, vtn_tail, active, zero_based_levels)
    vt_updated = _append_column_no_check(vt_body, vt_tail)
    vtn_updated = _append_column_no_check(vtn_body, vtn_tail)
    if not compute_substeps:
        empty = torch.empty((0,), dtype=torch.int64, device=rr.device)
        return (vt_updated, vtn_updated, empty, empty)
    max_vt = torch.maximum(vt_body, vtn_body)
    moving = max_vt > 0.001
    substeps, ktop = _sedimentation_motion_bounds_no_cache(moving=moving, speed=max_vt, dz1d=dz1d, dt=dt, one_based_levels=one_based_levels)
    return (vt_updated, vtn_updated, substeps, ktop)


def _rain_fallspeed_zero_tail_kernel(rhof: torch.Tensor, rr: torch.Tensor, ilamr: torch.Tensor, dz1d: torch.Tensor, zero_based_levels: torch.Tensor, one_based_levels: torch.Tensor, dt: float, compute_substeps: bool, _const_segment_e_R1=constants.R1, _const_segment_h_AV_R=constants.AV_R, _const_SCALAR_CRG_5=_SCALAR_CRG[5], _const_segment_e_ORG3=constants.ORG3, _const_SCALAR_CRE_2=_SCALAR_CRE[2], _const_segment_h_FV_R=constants.FV_R, _const_SCALAR_CRE_5=_SCALAR_CRE[5], _const_SCALAR_CRG_6=_SCALAR_CRG[6], _const_SCALAR_CRG_11=_SCALAR_CRG[11], _const_SCALAR_CRE_11=_SCALAR_CRE[11], _const_SCALAR_CRE_6=_SCALAR_CRE[6]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active = rr > float(_const_segment_e_R1)
    lamr = 1.0 / torch.where(ilamr > 0.0, ilamr, 1.0)
    vt_formula = rhof * float(_const_segment_h_AV_R) * _const_SCALAR_CRG_5 * float(_const_segment_e_ORG3) * lamr ** _const_SCALAR_CRE_2 * (lamr + float(_const_segment_h_FV_R)) ** (-_const_SCALAR_CRE_5)
    vtn_formula = rhof * float(_const_segment_h_AV_R) * _const_SCALAR_CRG_6 / _const_SCALAR_CRG_11 * lamr ** _const_SCALAR_CRE_11 * (lamr + float(_const_segment_h_FV_R)) ** (-_const_SCALAR_CRE_6)
    vt_body = _suffix_fill_from_active_zero_tail_no_cache(vt_formula, active, zero_based_levels)
    vtn_body = _suffix_fill_from_active_zero_tail_no_cache(vtn_formula, active, zero_based_levels)
    vt_updated = _append_zero_column_no_check(vt_body)
    vtn_updated = _append_zero_column_no_check(vtn_body)
    if not compute_substeps:
        empty = torch.empty((0,), dtype=torch.int64, device=rr.device)
        return (vt_updated, vtn_updated, empty, empty)
    max_vt = torch.maximum(vt_body, vtn_body)
    moving = max_vt > 0.001
    substeps, ktop = _sedimentation_motion_bounds_no_cache(moving=moving, speed=max_vt, dz1d=dz1d, dt=dt, one_based_levels=one_based_levels)
    return (vt_updated, vtn_updated, substeps, ktop)


def graupel_fallspeed_tensor(*, rhof: torch.Tensor, rho: torch.Tensor, visco: torch.Tensor, rg: torch.Tensor, rb: torch.Tensor, qb1d: torch.Tensor | None, idx: torch.Tensor, ilamg: torch.Tensor, dz1d: torch.Tensor, vt: torch.Tensor, vtn: torch.Tensor, dt: float, compute_substeps: bool=True, _const_STATIC_GRAUPEL_CGG=_STATIC_GRAUPEL_CGG) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nz = rg.shape[1]
    common_args = (rhof, rg, rb, idx, ilamg, dz1d, vt, vtn, _tensor(_const_STATIC_GRAUPEL_CGG, rg), _int_arange(0, nz, rg)[None, :], _int_arange(1, nz + 1, rg)[None, :], float(dt), bool(compute_substeps))
    if qb1d is not None:
        return _run_tensor_compile_island('graupel_fallspeed_optional', _graupel_fallspeed_optional_kernel, rg, rho, visco, qb1d, *common_args)
    return _run_tensor_compile_island('graupel_fallspeed_single', _graupel_fallspeed_single_kernel, rg, *common_args)


def _graupel_fallspeed_finish(rhof: torch.Tensor, rg: torch.Tensor, ilamg: torch.Tensor, dz1d: torch.Tensor, vt: torch.Tensor, vtn: torch.Tensor, cgg: torch.Tensor, idx: torch.Tensor, afall: torch.Tensor | float, bfall: torch.Tensor | float, zero_based_levels: torch.Tensor, one_based_levels: torch.Tensor, dt: float, compute_substeps: bool, _const_segment_e_R1=constants.R1, _const_segment_g_NRHG=constants.NRHG, _const_segment_g_OGG3=constants.OGG3, _const_segment_g_MU_G=constants.MU_G, _const_segment_g_OGG2=constants.OGG2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ncol, nz = rg.shape
    vt_tail = vt[:, nz]
    vtn_tail = vtn[:, nz]
    active = rg > float(_const_segment_e_R1)
    idx0 = torch.clamp(idx.to(torch.int64), 1, _const_segment_g_NRHG) - 1
    ilamg_safe = torch.where(ilamg > 0.0, ilamg, 1.0)
    vt_formula = rhof * afall * cgg[5, idx0] * float(_const_segment_g_OGG3) * ilamg_safe ** bfall
    if _const_segment_g_MU_G == 0.0:
        vtn_formula = rhof * afall * cgg[6, idx0] / cgg[11, idx0] * ilamg_safe ** bfall
    else:
        vtn_formula = rhof * afall * cgg[7, idx0] * float(_const_segment_g_OGG2) * ilamg_safe ** bfall
    vt_body = _suffix_fill_from_active_no_cache(vt_formula, vt_tail, active, zero_based_levels)
    vtn_body = _suffix_fill_from_active_no_cache(vtn_formula, vtn_tail, active, zero_based_levels)
    vt_updated = _append_column_no_check(vt_body, vt_tail)
    vtn_updated = _append_column_no_check(vtn_body, vtn_tail)
    if not compute_substeps:
        empty = torch.empty((0,), dtype=torch.int64, device=rg.device)
        return (vt_updated, vtn_updated, empty, empty)
    moving = vt_body > 0.001
    substeps, ktop = _sedimentation_motion_bounds_no_cache(moving=moving, speed=vt_body, dz1d=dz1d, dt=dt, one_based_levels=one_based_levels)
    return (vt_updated, vtn_updated, substeps, ktop)


def _graupel_fallspeed_finish_zero_tail(rhof: torch.Tensor, rg: torch.Tensor, ilamg: torch.Tensor, dz1d: torch.Tensor, cgg: torch.Tensor, idx: torch.Tensor, afall: torch.Tensor | float, bfall: torch.Tensor | float, zero_based_levels: torch.Tensor, one_based_levels: torch.Tensor, dt: float, compute_substeps: bool, _const_segment_e_R1=constants.R1, _const_segment_g_NRHG=constants.NRHG, _const_segment_g_OGG3=constants.OGG3, _const_segment_g_MU_G=constants.MU_G, _const_segment_g_OGG2=constants.OGG2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active = rg > float(_const_segment_e_R1)
    idx0 = torch.clamp(idx.to(torch.int64), 1, _const_segment_g_NRHG) - 1
    ilamg_safe = torch.where(ilamg > 0.0, ilamg, 1.0)
    vt_formula = rhof * afall * cgg[5, idx0] * float(_const_segment_g_OGG3) * ilamg_safe ** bfall
    if _const_segment_g_MU_G == 0.0:
        vtn_formula = rhof * afall * cgg[6, idx0] / cgg[11, idx0] * ilamg_safe ** bfall
    else:
        vtn_formula = rhof * afall * cgg[7, idx0] * float(_const_segment_g_OGG2) * ilamg_safe ** bfall
    vt_body = _suffix_fill_from_active_zero_tail_no_cache(vt_formula, active, zero_based_levels)
    vtn_body = _suffix_fill_from_active_zero_tail_no_cache(vtn_formula, active, zero_based_levels)
    vt_updated = _append_zero_column_no_check(vt_body)
    vtn_updated = _append_zero_column_no_check(vtn_body)
    if not compute_substeps:
        empty = torch.empty((0,), dtype=torch.int64, device=rg.device)
        return (vt_updated, vtn_updated, empty, empty)
    moving = vt_body > 0.001
    substeps, ktop = _sedimentation_motion_bounds_no_cache(moving=moving, speed=vt_body, dz1d=dz1d, dt=dt, one_based_levels=one_based_levels)
    return (vt_updated, vtn_updated, substeps, ktop)


def _graupel_fallspeed_optional_kernel(rho: torch.Tensor, visco: torch.Tensor, qb1d: torch.Tensor, rhof: torch.Tensor, rg: torch.Tensor, rb: torch.Tensor, idx: torch.Tensor, ilamg: torch.Tensor, dz1d: torch.Tensor, vt: torch.Tensor, vtn: torch.Tensor, cgg: torch.Tensor, zero_based_levels: torch.Tensor, one_based_levels: torch.Tensor, dt: float, compute_substeps: bool, _const_segment_g_METERS3_TO_LITERS=constants.METERS3_TO_LITERS, _const_segment_g_RHO_G_0=constants.RHO_G[0], _const_segment_g_RHO_G_segment_g_NRHG_1=constants.RHO_G[constants.NRHG - 1], _const_segment_p3_A_COEFF=constants.A_COEFF, _const_segment_p3_EARTH_GRAVITY=constants.EARTH_GRAVITY, _const_segment_p3_B_COEFF=constants.B_COEFF) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del qb1d
    dens_g = torch.clamp(float(_const_segment_g_METERS3_TO_LITERS) * rg / _clamp_tiny(rb), min=float(_const_segment_g_RHO_G_0), max=float(_const_segment_g_RHO_G_segment_g_NRHG_1))
    afall = float(_const_segment_p3_A_COEFF) * (float(4.0 * _const_segment_p3_EARTH_GRAVITY / 3.0) * dens_g / rho) ** float(_const_segment_p3_B_COEFF)
    afall = afall * visco ** float(1.0 - 2.0 * _const_segment_p3_B_COEFF)
    bfall = float(3.0 * _const_segment_p3_B_COEFF - 1.0)
    return _graupel_fallspeed_finish(rhof, rg, ilamg, dz1d, vt, vtn, cgg, idx, afall, bfall, zero_based_levels, one_based_levels, dt, compute_substeps)


def _graupel_fallspeed_single_kernel(rhof: torch.Tensor, rg: torch.Tensor, rb: torch.Tensor, idx: torch.Tensor, ilamg: torch.Tensor, dz1d: torch.Tensor, vt: torch.Tensor, vtn: torch.Tensor, cgg: torch.Tensor, zero_based_levels: torch.Tensor, one_based_levels: torch.Tensor, dt: float, compute_substeps: bool, _const_segment_p3_AV_G_OLD=constants.AV_G_OLD, _const_segment_p3_BV_G_OLD=constants.BV_G_OLD) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del rb
    return _graupel_fallspeed_finish(rhof, rg, ilamg, dz1d, vt, vtn, cgg, idx, float(_const_segment_p3_AV_G_OLD), float(_const_segment_p3_BV_G_OLD), zero_based_levels, one_based_levels, dt, compute_substeps)


def _graupel_fallspeed_optional_zero_tail_kernel(rho: torch.Tensor, visco: torch.Tensor, qb1d: torch.Tensor, rhof: torch.Tensor, rg: torch.Tensor, rb: torch.Tensor, idx: torch.Tensor, ilamg: torch.Tensor, dz1d: torch.Tensor, cgg: torch.Tensor, zero_based_levels: torch.Tensor, one_based_levels: torch.Tensor, dt: float, compute_substeps: bool, _const_segment_g_METERS3_TO_LITERS=constants.METERS3_TO_LITERS, _const_segment_g_RHO_G_0=constants.RHO_G[0], _const_segment_g_RHO_G_segment_g_NRHG_1=constants.RHO_G[constants.NRHG - 1], _const_segment_p3_A_COEFF=constants.A_COEFF, _const_segment_p3_EARTH_GRAVITY=constants.EARTH_GRAVITY, _const_segment_p3_B_COEFF=constants.B_COEFF) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del qb1d
    dens_g = torch.clamp(float(_const_segment_g_METERS3_TO_LITERS) * rg / _clamp_tiny(rb), min=float(_const_segment_g_RHO_G_0), max=float(_const_segment_g_RHO_G_segment_g_NRHG_1))
    afall = float(_const_segment_p3_A_COEFF) * (float(4.0 * _const_segment_p3_EARTH_GRAVITY / 3.0) * dens_g / rho) ** float(_const_segment_p3_B_COEFF)
    afall = afall * visco ** float(1.0 - 2.0 * _const_segment_p3_B_COEFF)
    bfall = float(3.0 * _const_segment_p3_B_COEFF - 1.0)
    return _graupel_fallspeed_finish_zero_tail(rhof, rg, ilamg, dz1d, cgg, idx, afall, bfall, zero_based_levels, one_based_levels, dt, compute_substeps)


def _graupel_fallspeed_single_zero_tail_kernel(rhof: torch.Tensor, rg: torch.Tensor, rb: torch.Tensor, idx: torch.Tensor, ilamg: torch.Tensor, dz1d: torch.Tensor, cgg: torch.Tensor, zero_based_levels: torch.Tensor, one_based_levels: torch.Tensor, dt: float, compute_substeps: bool, _const_segment_p3_AV_G_OLD=constants.AV_G_OLD, _const_segment_p3_BV_G_OLD=constants.BV_G_OLD) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del rb
    return _graupel_fallspeed_finish_zero_tail(rhof, rg, ilamg, dz1d, cgg, idx, float(_const_segment_p3_AV_G_OLD), float(_const_segment_p3_BV_G_OLD), zero_based_levels, one_based_levels, dt, compute_substeps)


def snow_fallspeed_tensor(
    *,
    rhof: torch.Tensor,
    rs: torch.Tensor,
    prr_sml: torch.Tensor,
    smob: torch.Tensor,
    smoc: torch.Tensor,
    rr: torch.Tensor,
    vtrr: torch.Tensor,
    dz1d: torch.Tensor,
    vt: torch.Tensor,
    vtboost: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nz = rs.shape[1]
    return _run_tensor_compile_island(
        "snow_fallspeed",
        _snow_fallspeed_kernel,
        rs,
        rhof,
        rs,
        prr_sml,
        smob,
        smoc,
        rr,
        vtrr,
        dz1d,
        vt,
        vtboost,
        _tensor(_STATIC_SNOW_CSG, rs),
        _tensor(_STATIC_SNOW_CSE, rs),
        _int_arange(0, nz, rs)[None, :],
        _int_arange(1, nz + 1, rs)[None, :],
        float(dt),
    )


def _snow_fallspeed_kernel(rhof: torch.Tensor, rs: torch.Tensor, prr_sml: torch.Tensor, smob: torch.Tensor, smoc: torch.Tensor, rr: torch.Tensor, vtrr: torch.Tensor, dz1d: torch.Tensor, vt: torch.Tensor, vtboost: torch.Tensor, csg: torch.Tensor, cse: torch.Tensor, zero_based_levels: torch.Tensor, one_based_levels: torch.Tensor, dt: float, _const_segment_e_R1=constants.R1, _const_segment_e_LAM0=constants.LAM0, _const_segment_p5_FV_S=constants.FV_S, _const_segment_e_LAM1=constants.LAM1, _const_segment_e_KAP0=constants.KAP0, _const_segment_e_KAP1=constants.KAP1, _const_segment_e_MU_S=constants.MU_S, _const_segment_p5_AV_S=constants.AV_S) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ncol, nz = rs.shape
    vt_tail = vt[:, nz]
    active = rs > float(_const_segment_e_R1)
    # The source skips the formula below its snow-mass threshold. Safe unused
    # operands prevent overflowing powers from poisoning the masked VJP.
    smob_safe = torch.where(active, smob, 1.0)
    smoc_safe = torch.where(active, smoc, 1.0)
    # Float32 runs evaluate the fall-speed moments in float64 and round vts once (float64 runs: unchanged). With
    # little snow mrat is large: mrat ** MU_S is huge and ils ** cse tiny, and the float32 backward of their product
    # overflows (NaN in a 24 h float32 window adjoint). This changes float32 vts at round-off level.
    work = torch.float64 if smob.dtype == torch.float32 else smob.dtype
    csg_w, cse_w = csg.to(work), cse.to(work)
    mrat = 1.0 / (smoc_safe.to(work) / smob_safe.to(work))
    ils1 = 1.0 / (mrat * float(_const_segment_e_LAM0) + float(_const_segment_p5_FV_S))
    ils2 = 1.0 / (mrat * float(_const_segment_e_LAM1) + float(_const_segment_p5_FV_S))
    t1_vts = float(_const_segment_e_KAP0) * csg_w[3] * ils1 ** cse_w[3]
    t2_vts = float(_const_segment_e_KAP1) * mrat ** float(_const_segment_e_MU_S) * csg_w[9] * ils2 ** cse_w[9]
    ils1 = 1.0 / (mrat * float(_const_segment_e_LAM0))
    ils2 = 1.0 / (mrat * float(_const_segment_e_LAM1))
    t3_vts = float(_const_segment_e_KAP0) * csg_w[0] * ils1 ** cse_w[0]
    t4_vts = float(_const_segment_e_KAP1) * mrat ** float(_const_segment_e_MU_S) * csg_w[6] * ils2 ** cse_w[6]
    vts = (rhof.to(work) * float(_const_segment_p5_AV_S) * (t1_vts + t2_vts) / _clamp_tiny(t3_vts + t4_vts)).to(smob.dtype)
    sr = rs / _clamp_tiny(rs + rr)
    vt_melting = vts * sr + (1.0 - sr) * vtrr[:, :nz]
    vt_formula = torch.where(prr_sml > 0.0, vt_melting, vts * vtboost)
    vt_body = _suffix_fill_from_active_no_cache(vt_formula, vt_tail, active, zero_based_levels)
    moving = vt_body > 0.001
    substeps, ktop = _sedimentation_motion_bounds_no_cache(moving=moving, speed=vt_body, dz1d=dz1d, dt=dt, one_based_levels=one_based_levels)
    vt_updated = _append_column_no_check(vt_body, vt_tail)
    return (vt_updated, substeps, ktop)


def ice_fallspeed_tensor(
    *,
    rhof: torch.Tensor,
    ri: torch.Tensor,
    ilami: torch.Tensor,
    dz1d: torch.Tensor,
    vt: torch.Tensor,
    vtn: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nz = ri.shape[1]
    return _run_tensor_compile_island(
        "ice_fallspeed",
        _ice_fallspeed_kernel,
        ri,
        rhof,
        ri,
        ilami,
        dz1d,
        vt,
        vtn,
        _int_arange(0, nz, ri)[None, :],
        _int_arange(1, nz + 1, ri)[None, :],
        float(dt),
    )


def _ice_fallspeed_kernel(rhof: torch.Tensor, ri: torch.Tensor, ilami: torch.Tensor, dz1d: torch.Tensor, vt: torch.Tensor, vtn: torch.Tensor, zero_based_levels: torch.Tensor, one_based_levels: torch.Tensor, dt: float, _const_segment_e_R1=constants.R1, _const_segment_p5_AV_I=constants.AV_I, _const_SCALAR_CIG_2=_SCALAR_CIG[2], _const_segment_e_OIG2=constants.OIG2, _const_SCALAR_CIG_5=_SCALAR_CIG[5], _const_SCALAR_CIG_6=_SCALAR_CIG[6]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ncol, nz = ri.shape
    vt_tail = vt[:, nz]
    vtn_tail = vtn[:, nz]
    active = ri > float(_const_segment_e_R1)
    ilami_safe = torch.where(ilami > 0.0, ilami, 0.0)
    # Keep the source's coefficient operations separate before sedimentation.
    vt_formula = rhof * float(_const_segment_p5_AV_I) * float(_const_SCALAR_CIG_2) * float(_const_segment_e_OIG2) * ilami_safe
    vtn_formula = rhof * float(_const_segment_p5_AV_I) * float(_const_SCALAR_CIG_5) / float(_const_SCALAR_CIG_6) * ilami_safe
    vt_body = _suffix_fill_from_active_no_cache(vt_formula, vt_tail, active, zero_based_levels)
    vtn_body = _suffix_fill_from_active_no_cache(vtn_formula, vtn_tail, active, zero_based_levels)
    moving = vt_body > 0.001
    substeps, ktop = _sedimentation_motion_bounds_no_cache(moving=moving, speed=vt_body, dz1d=dz1d, dt=dt, one_based_levels=one_based_levels)
    vt_updated = _append_column_no_check(vt_body, vt_tail)
    vtn_updated = _append_column_no_check(vtn_body, vtn_tail)
    return (vt_updated, vtn_updated, substeps, ktop)


def cloud_fallspeed_tensor(
    *,
    rhof: torch.Tensor,
    w1d: torch.Tensor,
    rc: torch.Tensor,
    nc: torch.Tensor,
    ilamc: torch.Tensor,
    dz1d: torch.Tensor,
    vt: torch.Tensor,
    vtn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nz = rc.shape[1]
    if nz <= 1:
        return _run_tensor_compile_island(
            "cloud_fallspeed_nz1",
            _cloud_fallspeed_nz1_kernel,
            rc,
            rhof,
            w1d,
            rc,
            nc,
            ilamc,
            vt,
            vtn,
            _tensor(_STATIC_CLOUD_CCG[4], rc),
            _tensor(_STATIC_CLOUD_OCG2, rc),
            _tensor(_STATIC_CLOUD_CCG[3], rc),
            _tensor(_STATIC_CLOUD_OCG1, rc),
        )
    return _run_tensor_compile_island(
        "cloud_fallspeed",
        _cloud_fallspeed_kernel,
        rc,
        rhof,
        w1d,
        rc,
        nc,
        ilamc,
        dz1d,
        vt,
        vtn,
        _tensor(_STATIC_CLOUD_CCG[4], rc),
        _tensor(_STATIC_CLOUD_OCG2, rc),
        _tensor(_STATIC_CLOUD_CCG[3], rc),
        _tensor(_STATIC_CLOUD_OCG1, rc),
        _int_arange(1, nz, rc)[None, :],
        _int_arange(1, nz + 1, rc)[None, :],
    )


def _cloud_fallspeed_kernel(rhof: torch.Tensor, w1d: torch.Tensor, rc: torch.Tensor, nc: torch.Tensor, ilamc: torch.Tensor, dz1d: torch.Tensor, vt: torch.Tensor, vtn: torch.Tensor, ccg4_table: torch.Tensor, ocg2_table: torch.Tensor, ccg3_table: torch.Tensor, ocg1_table: torch.Tensor, scan_levels: torch.Tensor, body_levels: torch.Tensor, _const_segment_e_R2=constants.R2, _const_segment_f_NT_C_MIN=constants.NT_C_MIN, _const_segment_p5_AV_C=constants.AV_C, _const_segment_e_R1=constants.R1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ncol, nz = rc.shape
    hgt_before = _prepend_zero_column_no_check(torch.cumsum(dz1d[:, :nz - 2], dim=1))
    scan_active = hgt_before <= 500.0
    cloud_candidates = scan_levels.expand(ncol, nz - 1) * (scan_active & (rc[:, :nz - 1] > float(_const_segment_e_R2))).to(torch.int64)
    ktop = torch.max(cloud_candidates, dim=1).values
    ktop = torch.clamp_min(ktop, torch.ones((ncol,), dtype=torch.int64, device=rc.device))
    nu_c = _cloud_nuc_no_cache(torch.clamp_min(nc, float(_const_segment_f_NT_C_MIN)))
    table_index = nu_c.to(torch.int64) - 1
    ccg4 = ccg4_table[table_index]
    ocg2 = ocg2_table[table_index]
    ccg3 = ccg3_table[table_index]
    ocg1 = ocg1_table[table_index]
    av_c = float(_const_segment_p5_AV_C)
    ilamc2 = _square(ilamc)
    vt_formula = rhof * av_c * ccg4 * ocg2 * ilamc2
    vtn_formula = rhof * av_c * ccg3 * ocg1 * ilamc2
    level_active = (body_levels <= ktop[:, None]) & (rc > float(_const_segment_e_R1)) & (w1d < 0.1)
    vt_body = torch.where(level_active, vt_formula, vt[:, :nz])
    vtn_body = torch.where(level_active, vtn_formula, vtn[:, :nz])
    vt_updated = _append_column_no_check(vt_body, vt[:, nz])
    vtn_updated = _append_column_no_check(vtn_body, vtn[:, nz])
    return (vt_updated, vtn_updated, ktop)


def _cloud_fallspeed_nz1_kernel(rhof: torch.Tensor, w1d: torch.Tensor, rc: torch.Tensor, nc: torch.Tensor, ilamc: torch.Tensor, vt: torch.Tensor, vtn: torch.Tensor, ccg4_table: torch.Tensor, ocg2_table: torch.Tensor, ccg3_table: torch.Tensor, ocg1_table: torch.Tensor, _const_segment_f_NT_C_MIN=constants.NT_C_MIN, _const_segment_p5_AV_C=constants.AV_C, _const_segment_e_R1=constants.R1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ncol, nz = rc.shape
    ktop = torch.ones((ncol,), dtype=torch.int64, device=rc.device)
    nu_c = _cloud_nuc_no_cache(torch.clamp_min(nc, float(_const_segment_f_NT_C_MIN)))
    table_index = nu_c.to(torch.int64) - 1
    av_c = float(_const_segment_p5_AV_C)
    ilamc2 = _square(ilamc)
    vt_formula = rhof * av_c * ccg4_table[table_index] * ocg2_table[table_index] * ilamc2
    vtn_formula = rhof * av_c * ccg3_table[table_index] * ocg1_table[table_index] * ilamc2
    level_active = (rc > float(_const_segment_e_R1)) & (w1d < 0.1)
    vt_body = torch.where(level_active, vt_formula, vt[:, :nz])
    vtn_body = torch.where(level_active, vtn_formula, vtn[:, :nz])
    return (_append_column_no_check(vt_body, vt[:, nz]), _append_column_no_check(vtn_body, vtn[:, nz]), ktop)


def _species_tensor(*components: torch.Tensor) -> torch.Tensor:
    if not components:
        raise ValueError("at least one species component is required")
    reference = components[0]
    shape = reference.shape
    for component in components[1:]:
        if component.shape != shape:
            raise ValueError("species components must have matching shapes")
        if component.dtype != reference.dtype or component.device != reference.device:
            raise ValueError("species components must have matching dtype and device")
    return torch.stack(components, dim=0)


def _masked_merge_components(
    mask: torch.Tensor,
    current: tuple[torch.Tensor, ...],
    replacement: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    if len(current) != len(replacement):
        raise ValueError("current and replacement component counts must match")
    if not current:
        return ()
    if len(current) == 1:
        return (torch.where(mask, replacement[0], current[0]),)
    return tuple(
        torch.where(mask, replacement_component, current_component)
        for current_component, replacement_component in zip(current, replacement)
    )


def _masked_update_mapping(
    updated: dict[str, torch.Tensor],
    mask: torch.Tensor,
    replacements: Mapping[str, torch.Tensor],
) -> None:
    names = tuple(replacements)
    merged = _masked_merge_components(
        mask,
        tuple(updated[name] for name in names),
        tuple(replacements[name] for name in names),
    )
    for name, values in zip(names, merged, strict=True):
        updated[name] = values


def _sedimentation_steps_tensor(steps: torch.Tensor | int, ncol: int, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(steps, torch.Tensor):
        return steps.to(dtype=torch.int64, device=reference.device)
    return torch.full((ncol,), int(steps), dtype=torch.int64, device=reference.device)


def _sedimentation_ktop_tensor(
    ktop_sedi: torch.Tensor | int | None,
    ncol: int,
    nz: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if ktop_sedi is None:
        return torch.full((ncol,), nz - 1, dtype=torch.int64, device=reference.device)
    if isinstance(ktop_sedi, torch.Tensor):
        return ktop_sedi.to(dtype=torch.int64, device=reference.device)
    return torch.full((ncol,), int(ktop_sedi), dtype=torch.int64, device=reference.device)


def _species_limit_tensor(limits: torch.Tensor | tuple[float | torch.Tensor, ...], *, reference: torch.Tensor) -> torch.Tensor:
    species_count, ncol, nz = reference.shape
    if isinstance(limits, torch.Tensor):
        tensor = limits.to(dtype=reference.dtype, device=reference.device)
        if tensor.shape == reference.shape:
            return tensor
        if tensor.shape == (species_count, ncol, 1):
            return tensor.expand_as(reference)
        if tensor.ndim == 1 and tensor.shape[0] == species_count:
            return tensor[:, None, None].expand_as(reference)
        raise ValueError("packed species limit tensor shape must match [species,ncol,nz], [species,ncol,1], or [species]")
    if len(limits) != species_count:
        raise ValueError("limit count must match packed species count")
    packed = torch.empty((species_count, ncol, 1), **_factory(reference))
    for index, limit in enumerate(limits):
        if isinstance(limit, torch.Tensor):
            tensor = limit.to(dtype=reference.dtype, device=reference.device)
            if tensor.ndim == 0:
                packed[index] = tensor
            elif tensor.ndim == 1 and tensor.shape[0] == ncol:
                packed[index] = tensor[:, None]
            elif tensor.ndim == 2 and tensor.shape == (ncol, 1):
                packed[index] = tensor
            elif tensor.ndim == 2 and tensor.shape == (ncol, nz):
                full = torch.empty_like(reference)
                for full_index, full_limit in enumerate(limits):
                    full[full_index] = _limit_like(full_limit, reference[full_index])
                return full
            else:
                raise ValueError("species sedimentation limits must be scalar, [ncol], [ncol,1], or [ncol,nz]")
        else:
            packed[index] = float(limit)
    return packed


def _sedimentation_species_tensor_active_step(
    xr: torch.Tensor,
    vt: torch.Tensor,
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    xten: torch.Tensor,
    limit_tensor: torch.Tensor,
    steps_tensor: torch.Tensor,
    ktop_tensor: torch.Tensor,
    active_step: torch.Tensor,
    precip: torch.Tensor | None,
    dt: float,
    precip_species_index: int,
    low_limit_mass_for_precip: float = constants.LOW_LIMIT_MASS_FOR_PRECIP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    species_count, ncol, nz = xr.shape
    inv_steps = 1.0 / steps_tensor.to(dtype=xr.dtype)
    sed_r = vt[:, :, :nz] * xr
    flux = torch.empty_like(xr)
    flux[:, :, :-1] = sed_r[:, :, 1:] - sed_r[:, :, :-1]
    flux[:, :, -1] = -sed_r[:, :, -1]
    levels = torch.arange(1, nz + 1, dtype=torch.int64, device=xr.device)[None, None, :]
    level_active = active_step[None, :, None] & ((levels == nz) | (ktop_tensor[None, :, None] >= levels))
    odz = 1.0 / dz1d
    orho = 1.0 / rho
    inv_steps_3d = inv_steps[None, :, None]
    xten_candidate = xten + flux * odz[None, :, :] * inv_steps_3d * orho[None, :, :]
    xr_candidate = torch.maximum(limit_tensor, xr + flux * odz[None, :, :] * dt * inv_steps_3d)
    xten_updated = torch.where(level_active, xten_candidate, xten)
    xr_updated = torch.where(level_active, xr_candidate, xr)
    precip_updated = precip
    if precip is not None:
        precip_candidate = precip + sed_r[precip_species_index, :, 0] * dt * inv_steps
        precip_active = active_step & (xr_updated[precip_species_index, :, 0] > low_limit_mass_for_precip)
        precip_updated = torch.where(precip_active, precip_candidate, precip)
    return xr_updated, xten_updated, precip_updated


def _sedimentation_pair_components_active_step(
    xr0: torch.Tensor,
    xr1: torch.Tensor,
    vt0: torch.Tensor,
    vt1: torch.Tensor,
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    xten0: torch.Tensor,
    xten1: torch.Tensor,
    steps_tensor: torch.Tensor,
    ktop_tensor: torch.Tensor,
    precip: torch.Tensor,
    dt: float,
    limit0: float,
    limit1: float,
    low_limit_mass_for_precip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ncol, nz = xr0.shape
    inv_steps = 1.0 / steps_tensor.to(dtype=xr0.dtype)
    sed0 = vt0[:, :nz] * xr0
    sed1 = vt1[:, :nz] * xr1
    flux0 = torch.empty_like(xr0)
    flux1 = torch.empty_like(xr1)
    flux0[:, :-1] = sed0[:, 1:] - sed0[:, :-1]
    flux1[:, :-1] = sed1[:, 1:] - sed1[:, :-1]
    flux0[:, -1] = -sed0[:, -1]
    flux1[:, -1] = -sed1[:, -1]
    levels = torch.arange(1, nz + 1, dtype=torch.int64, device=xr0.device)[None, :]
    active_step = steps_tensor > 0
    level_active = active_step[:, None] & ((levels == nz) | (ktop_tensor[:, None] >= levels))
    odz = 1.0 / dz1d
    orho = 1.0 / rho
    inv_steps_2d = inv_steps[:, None]
    xten0_candidate = xten0 + flux0 * odz * inv_steps_2d * orho
    xten1_candidate = xten1 + flux1 * odz * inv_steps_2d * orho
    xr0_candidate = torch.clamp_min(xr0 + flux0 * odz * dt * inv_steps_2d, limit0)
    xr1_candidate = torch.clamp_min(xr1 + flux1 * odz * dt * inv_steps_2d, limit1)
    xr0_updated = torch.where(level_active, xr0_candidate, xr0)
    xr1_updated = torch.where(level_active, xr1_candidate, xr1)
    xten0_updated = torch.where(level_active, xten0_candidate, xten0)
    xten1_updated = torch.where(level_active, xten1_candidate, xten1)
    precip_candidate = precip + sed0[:, 0] * dt * inv_steps
    precip_active = active_step & (xr0_updated[:, 0] > low_limit_mass_for_precip)
    precip_updated = torch.where(precip_active, precip_candidate, precip)
    return xr0_updated, xr1_updated, xten0_updated, xten1_updated, precip_updated


def _p5_ice_cloud_sedimentation_kernel(
    ri: torch.Tensor,
    ni: torch.Tensor,
    vtri: torch.Tensor,
    vtni: torch.Tensor,
    qiten: torch.Tensor,
    niten: torch.Tensor,
    ice_steps: torch.Tensor,
    ice_ktop: torch.Tensor,
    ice_precip: torch.Tensor,
    rc: torch.Tensor,
    nc: torch.Tensor,
    vtrc: torch.Tensor,
    vtnc: torch.Tensor,
    qcten: torch.Tensor,
    ncten: torch.Tensor,
    cloud_ktop: torch.Tensor,
    cloud_precip: torch.Tensor,
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    dt: float,
    _mass_limit: float = float(constants.R1),
    _number_limit: float = float(constants.R2),
    _precip_limit: float = float(constants.LOW_LIMIT_MASS_FOR_PRECIP),
) -> tuple[torch.Tensor, ...]:
    ice = _sedimentation_pair_components_active_step(
        ri,
        ni,
        vtri,
        vtni,
        dz1d,
        rho,
        qiten,
        niten,
        ice_steps,
        ice_ktop,
        ice_precip,
        dt,
        float(_mass_limit),
        float(_number_limit),
        float(_precip_limit),
    )
    cloud = _sedimentation_pair_components_active_step(
        rc,
        nc,
        vtrc,
        vtnc,
        dz1d,
        rho,
        qcten,
        ncten,
        torch.ones_like(ice_steps),
        cloud_ktop,
        cloud_precip,
        dt,
        float(_mass_limit),
        float(_number_limit),
        float(_precip_limit),
    )
    return (*ice, *cloud)


def _sedimentation_species_tensor_step(
    xr: torch.Tensor,
    vt: torch.Tensor,
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    xten: torch.Tensor,
    limit_tensor: torch.Tensor,
    steps_tensor: torch.Tensor,
    ktop_tensor: torch.Tensor,
    precip: torch.Tensor | None,
    dt: float,
    step_index: int,
    precip_species_index: int,
    low_limit_mass_for_precip: float = constants.LOW_LIMIT_MASS_FOR_PRECIP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return _sedimentation_species_tensor_active_step(
        xr,
        vt,
        dz1d,
        rho,
        xten,
        limit_tensor,
        steps_tensor,
        ktop_tensor,
        steps_tensor > step_index,
        precip,
        dt,
        precip_species_index,
        low_limit_mass_for_precip,
    )


def sedimentation_species_tensor(
    *,
    xr: torch.Tensor,
    vt: torch.Tensor,
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    xten: torch.Tensor,
    limits: torch.Tensor | tuple[float | torch.Tensor, ...],
    steps: torch.Tensor | int,
    ktop_sedi: torch.Tensor | int | None,
    precip: torch.Tensor | None,
    dt: float,
    step_index: int = 0,
    precip_species_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if xr.ndim != 3:
        raise ValueError("species sedimentation expects xr shape [nspecies,ncol,nz]")
    species_count, ncol, nz = xr.shape
    if vt.shape[:2] != (species_count, ncol) or vt.shape[2] < nz:
        raise ValueError("species sedimentation vt shape must match xr with at least nz vertical entries")
    if xten.shape != xr.shape:
        raise ValueError("species sedimentation xten shape must match xr")
    if dz1d.shape != (ncol, nz) or rho.shape != (ncol, nz):
        raise ValueError("species sedimentation dz1d/rho shape must be [ncol,nz]")
    if not 0 <= precip_species_index < species_count:
        raise ValueError("precip_species_index is out of range")

    steps_tensor = _sedimentation_steps_tensor(steps, ncol, xr)
    ktop_tensor = _sedimentation_ktop_tensor(ktop_sedi, ncol, nz, xr)
    limit_tensor = _species_limit_tensor(limits, reference=xr)
    return _sedimentation_species_tensor_step(
        xr,
        vt,
        dz1d,
        rho,
        xten,
        limit_tensor,
        steps_tensor,
        ktop_tensor,
        precip,
        float(dt),
        int(step_index),
        int(precip_species_index),
    )


def _sedimentation_tensor_active_step(
    xr: torch.Tensor,
    vt: torch.Tensor,
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    xten: torch.Tensor,
    limit_tensor: torch.Tensor,
    steps_tensor: torch.Tensor,
    ktop_tensor: torch.Tensor,
    active_step: torch.Tensor,
    precip: torch.Tensor | None,
    dt: float,
    precip_count: int | None,
    low_limit_mass_for_precip: float = constants.LOW_LIMIT_MASS_FOR_PRECIP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    ncol, nz = xr.shape
    inv_steps = 1.0 / steps_tensor.to(dtype=xr.dtype)
    sed_r = vt[:, :nz] * xr
    flux = torch.empty_like(xr)
    flux[:, :-1] = sed_r[:, 1:] - sed_r[:, :-1]
    flux[:, -1:] = -sed_r[:, -1:]
    levels = torch.arange(1, nz + 1, dtype=torch.int64, device=xr.device)[None, :]
    level_active = active_step[:, None] & ((levels == nz) | (ktop_tensor[:, None] >= levels))
    odz = 1.0 / dz1d
    orho = 1.0 / rho
    xten_candidate = xten + flux * odz * inv_steps[:, None] * orho
    xr_candidate = torch.maximum(limit_tensor, xr + flux * odz * dt * inv_steps[:, None])
    xten_updated = torch.where(level_active, xten_candidate, xten)
    xr_updated = torch.where(level_active, xr_candidate, xr)
    precip_updated = precip
    if precip is not None:
        precip_rows = ncol if precip_count is None else int(precip_count)
        precip_candidate = precip + sed_r[:precip_rows, 0] * dt * inv_steps[:precip_rows]
        precip_active = active_step[:precip_rows] & (xr_updated[:precip_rows, 0] > low_limit_mass_for_precip)
        precip_updated = torch.where(precip_active, precip_candidate, precip)
    return xr_updated, xten_updated, precip_updated


def _sedimentation_tensor_step(
    xr: torch.Tensor,
    vt: torch.Tensor,
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    xten: torch.Tensor,
    limit_tensor: torch.Tensor,
    steps_tensor: torch.Tensor,
    ktop_tensor: torch.Tensor,
    precip: torch.Tensor | None,
    dt: float,
    step_index: int,
    precip_count: int | None,
    low_limit_mass_for_precip: float = constants.LOW_LIMIT_MASS_FOR_PRECIP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return _sedimentation_tensor_active_step(
        xr,
        vt,
        dz1d,
        rho,
        xten,
        limit_tensor,
        steps_tensor,
        ktop_tensor,
        steps_tensor > step_index,
        precip,
        dt,
        precip_count,
        low_limit_mass_for_precip,
    )


def sedimentation_tensor(
    *,
    xr: torch.Tensor,
    vt: torch.Tensor,
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    xten: torch.Tensor,
    limit: float | torch.Tensor,
    steps: torch.Tensor | int,
    ktop_sedi: torch.Tensor | int | None,
    precip: torch.Tensor | None,
    dt: float,
    step_index: int = 0,
    precip_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    ncol, nz = xr.shape
    steps_tensor = _sedimentation_steps_tensor(steps, ncol, xr)
    ktop_tensor = _sedimentation_ktop_tensor(ktop_sedi, ncol, nz, xr)
    return _sedimentation_tensor_step(
        xr,
        vt,
        dz1d,
        rho,
        xten,
        _limit_like(limit, xr),
        steps_tensor,
        ktop_tensor,
        precip,
        float(dt),
        int(step_index),
        precip_count,
    )


def semilagrangian_geometry_tensor(
    *,
    dz1d: torch.Tensor,
    steps: torch.Tensor | int,
    dt: float,
    odt: float,
) -> SemiLagrangianSedimentationGeometry:
    ncol, _ = dz1d.shape
    if isinstance(steps, torch.Tensor):
        steps_tensor = steps.to(torch.int64)
    else:
        steps_tensor = torch.full((ncol,), int(steps), dtype=torch.int64, device=dz1d.device)
    steps_float = torch.clamp_min(steps_tensor.to(dtype=dz1d.dtype), 1.0)
    step_factor = float(dt) / steps_float
    zi = _prepend_zero_column(torch.cumsum(dz1d, dim=1))
    limiter_delta = float(0.05 * odt) * dz1d * steps_float[:, None]
    limiter_prefix = _prepend_zero_column(torch.cumsum(limiter_delta, dim=1))
    return SemiLagrangianSedimentationGeometry(
        steps_tensor=steps_tensor,
        steps_float=steps_float,
        step_factor=step_factor,
        zi=zi,
        limiter_prefix=limiter_prefix,
    )


def _lower_bound_lastdim_tensor(sorted_values: torch.Tensor, query_values: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.sum((sorted_values.unsqueeze(-2) < query_values.unsqueeze(-1)).to(torch.int64), dim=-1)
    expanded_elements = int(query_values.numel()) * int(sorted_values.shape[-1])
    if expanded_elements <= SEMI_SEARCHSORTED_FULL_FUSION_MAX_ELEMENTS:
        return torch.sum((sorted_values.unsqueeze(-2) < query_values.unsqueeze(-1)).to(torch.int64), dim=-1)
    if query_values.shape[:-1] != sorted_values.shape[:-1]:
        query_values = query_values.expand((*sorted_values.shape[:-1], query_values.shape[-1]))
    query_width = int(query_values.shape[-1])
    sorted_width = int(sorted_values.shape[-1])
    prefix_elements = max(1, int(query_values.numel()) // max(1, query_width))
    chunk_width = max(1, min(query_width, SEMI_SEARCHSORTED_FULL_FUSION_MAX_ELEMENTS // max(1, prefix_elements * sorted_width)))
    result = torch.empty(query_values.shape, dtype=torch.int64, device=query_values.device)
    sorted_expanded = sorted_values.unsqueeze(-2)
    for start in range(0, query_width, chunk_width):
        stop = min(query_width, start + chunk_width)
        query_chunk = query_values[..., start:stop]
        result[..., start:stop] = torch.sum(
            (sorted_expanded < query_chunk.unsqueeze(-1)).to(torch.int64),
            dim=-1,
        )
    return result


def _bucketize_right_static_axis_tensor(values: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    axis_shape = (1,) * values.ndim + (axis.shape[0],)
    return torch.sum((values.unsqueeze(-1) >= axis.reshape(axis_shape)).to(torch.int64), dim=-1)


@torch.compiler.disable
def _run_semilagrangian_reduction(
    name: str,
    function: Any,
    reference: torch.Tensor,
    *args: Any,
) -> Any:
    status_name = f"semi_lagrangian_reduction:{name}"
    if (
        not SEMI_LAGRANGIAN_REDUCTION_COMPILE_CUDA
        or reference.device.type not in SEMI_LAGRANGIAN_REDUCTION_COMPILE_DEVICES
        or torch.compiler.is_compiling()
    ):
        return function(*args)
    if name in SEMI_LAGRANGIAN_REDUCTION_OUTER_EAGER_NAMES:
        if status_name not in _TENSOR_COMPILE_ISLAND_STATUS:
            cache_key = _compile_cache_key(name, function, reference, args=args)
            _record_tensor_compile_island_status(
                status_name,
                "outer_eager_by_policy",
                SEMI_LAGRANGIAN_REDUCTION_COMPILE_POLICY,
                cache_key=cache_key,
            )
        return function(*args)
    prepared_dispatch = _ACTIVE_PREPARED_COMPILE_DISPATCH.get()
    prepared_key = _prepared_compile_key(
        status_name,
        function,
        _prepared_runtime_signature(reference, args),
    )
    if prepared_dispatch is not None:
        prepared_callable = prepared_dispatch.callables.get(prepared_key)
        if prepared_callable is not None:
            prepared_dispatch.fast_hits += 1
            try:
                return prepared_callable(*args)
            except Exception as exc:
                reason = _compile_exception_reason(exc)
                raise RuntimeError(
                    f"prepared compile dispatch {status_name} runtime_failed: {reason}"
                ) from exc
        prepared_dispatch.misses += 1
    cache_key = _compile_cache_key(name, function, reference, args=args)
    if cache_key in _TENSOR_COMPILE_ISLAND_DISABLED:
        if prepared_dispatch is not None:
            raise RuntimeError(f"prepared compile dispatch {status_name} is disabled")
        return function(*args)
    compiled = _TENSOR_COMPILE_ISLAND_CACHE.get(cache_key)
    if compiled is None:
        compiled, result, failure_stage, failure, compile_timing = _compile_and_prime_with_scoped_scalar_capture(
            function,
            args,
            {"fullgraph": True, "dynamic": False, "mode": None},
        )
        _record_compile_island_timing(status_name, cache_key, compile_timing)
        if failure is not None:
            reason = _compile_exception_reason(failure)
            _TENSOR_COMPILE_ISLAND_DISABLED.add(cache_key)
            _record_tensor_compile_island_status(
                status_name,
                failure_stage or "compile_failed",
                reason,
                cache_key=cache_key,
            )
            _raise_compile_island_failure_if_strict(
                status_name,
                failure_stage or "compile_failed",
                reason,
                failure,
            )
            if prepared_dispatch is not None:
                raise RuntimeError(
                    f"prepared compile dispatch {status_name} {failure_stage or 'compile_failed'}: {reason}"
                ) from failure
            return function(*args)
        assert compiled is not None
        _TENSOR_COMPILE_ISLAND_CACHE[cache_key] = compiled
        _record_tensor_compile_island_status(status_name, "compiled", cache_key=cache_key)
        if prepared_dispatch is not None:
            prepared_dispatch.callables[prepared_key] = compiled
        return result
    try:
        result = compiled(*args)
    except Exception as exc:
        reason = _compile_exception_reason(exc)
        _TENSOR_COMPILE_ISLAND_CACHE.pop(cache_key, None)
        _TENSOR_COMPILE_ISLAND_DISABLED.add(cache_key)
        _record_tensor_compile_island_status(
            status_name,
            "runtime_failed",
            reason,
            cache_key=cache_key,
        )
        _raise_compile_island_failure_if_strict(status_name, "runtime_failed", reason, exc)
        if prepared_dispatch is not None:
            raise RuntimeError(
                f"prepared compile dispatch {status_name} runtime_failed: {reason}"
            ) from exc
        return function(*args)
    if prepared_dispatch is not None:
        prepared_dispatch.callables[prepared_key] = compiled
    return result


def _semi_overlap_chunk_width(*, prefix_elements: int, source_count: int) -> int:
    if source_count < 1:
        return 1
    if prefix_elements < 1:
        return source_count
    return max(
        1,
        min(
            source_count,
            SEMI_LAGRANGIAN_OVERLAP_CHUNK_MAX_ELEMENTS // max(1, int(prefix_elements)),
        ),
    )


def _gather_lastdim_valid(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, -1, torch.clamp(index, 0, values.shape[-1] - 1).to(torch.int64))


def _semilagrangian_geometry_source_bound(
    zi_low: torch.Tensor, zi_high: torch.Tensor, dza: torch.Tensor,
) -> int:
    """Prove the selected ordered window on live geometry, without a host read.

    Complete source cells inside the destination each occupy at least the
    declared minimum width. The first and last cells may be partial. The extra
    top source can only be the final cell; its width may be zero. Asynchronous
    device assertions reject an unsupported domain before truncated results can
    be consumed. The unbounded ordered policy retains the general input scope.
    """
    nz = zi_low.shape[-1]
    width = zi_high - zi_low
    torch._assert_async(torch.all((width > 0.0) & (width <= SEMI_LAGRANGIAN_GEOMETRY_MAX_DESTINATION_WIDTH)),
                        "ordered_geometry_bound: destination widths exceed the supported domain")
    torch._assert_async(torch.all(torch.isfinite(dza)) & torch.all(dza[..., :nz] >= SEMI_LAGRANGIAN_GEOMETRY_MIN_SOURCE_WIDTH)
                        & torch.all(dza[..., nz:] >= 0.0),
                        "ordered_geometry_bound: compressed source widths violate the supported domain")
    return min(nz + 1, math.ceil(SEMI_LAGRANGIAN_GEOMETRY_MAX_DESTINATION_WIDTH /
                               SEMI_LAGRANGIAN_GEOMETRY_MIN_SOURCE_WIDTH) + 2)


def _semilagrangian_ordered_scan_chunk_kernel(
    qsum: torch.Tensor,
    zsum: torch.Tensor,
    kb: torch.Tensor,
    kt: torch.Tensor,
    qa: torch.Tensor,
    dza: torch.Tensor,
    offset: torch.Tensor,
    max_sources: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eight increasing Fortran source offsets in one reusable compile island."""
    greater = kt > kb
    for local in range(SEMI_LAGRANGIAN_ORDERED_SCAN_CHUNK_SOURCES):
        source_offset = offset + local
        source_index = kb - 1 + source_offset
        inner = greater & (kb + source_offset < kt) & (source_offset < max_sources - 1)
        qa_source = _gather_lastdim_valid(qa, source_index)
        dza_source = _gather_lastdim_valid(dza, source_index)
        qsum = qsum + torch.where(inner, qa_source * dza_source, 0.0)
        zsum = zsum + torch.where(inner, dza_source, 0.0)
    return qsum, zsum


def _semilagrangian_ordered_overlap_scan(
    zi_low: torch.Tensor,
    zi_high: torch.Tensor,
    za: torch.Tensor,
    dza: torch.Tensor,
    qa: torch.Tensor,
    qpi: torch.Tensor,
    qmi: torch.Tensor,
    process: torch.Tensor,
    xr: torch.Tensor,
    xten: torch.Tensor,
    rho: torch.Tensor,
    limit_tensor: torch.Tensor,
    odt: float,
    max_sources: int,
    *,
    zi_interfaces: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Insertion indices are discrete branch selectors, not differentiable physics.
    # searchsorted avoids the dense [target,source] comparison used by the old probe.
    if zi_interfaces is None:
        kb_index = torch.searchsorted(za.contiguous(), zi_low.contiguous(), right=False)
        kt_index = torch.searchsorted(za.contiguous(), zi_high.contiguous(), right=False)
    else:
        # Every shared destination interface has the same insertion index for
        # its upper and lower neighbors. Search the full contiguous query once.
        # Independent primitive intervals retain the separate-search path.
        interface_indices = torch.searchsorted(
            za.contiguous(), zi_interfaces.contiguous(), right=False,
        )
        kb_index = interface_indices[..., :-1]
        kt_index = interface_indices[..., 1:]
    nz = xr.shape[-1]
    kb = torch.where(kb_index <= nz, torch.clamp_min(kb_index, 1), 1)
    kt = torch.where(kt_index <= nz + 1, kt_index, 0)
    span = torch.where(process & (kt >= kb), kt - kb + 1, 0)
    overflow = process & (span > max_sources)
    same = process & (kt == kb) & (~overflow)
    greater = process & (kt > kb) & (~overflow)
    kb0 = kb - 1
    kt0 = kt - 1

    za_kb = _gather_lastdim_valid(za, kb0)
    dza_kb = _gather_lastdim_valid(dza, kb0)
    qpi_kb = _gather_lastdim_valid(qpi, kb0)
    qmi_kb = _gather_lastdim_valid(qmi, kb0)
    tl = (zi_low - za_kb) / torch.where(same | greater, dza_kb, 1.0)
    th_same = (zi_high - za_kb) / torch.where(same, dza_kb, 1.0)
    qqd_kb = 0.5 * (qpi_kb - qmi_kb)
    qql = qqd_kb * tl * tl + qmi_kb * tl
    qqh_same = qqd_kb * th_same * th_same + qmi_kb * th_same
    same_width = th_same - tl
    xr_same = (qqh_same - qql) / torch.where(same, same_width, 1.0)
    qa_kb = _gather_lastdim_valid(qa, kb0)
    qsum = (qa_kb - qql) * dza_kb
    zsum = (1.0 - tl) * dza_kb
    for source_offset in range(1, max_sources - 1, SEMI_LAGRANGIAN_ORDERED_SCAN_CHUNK_SOURCES):
        # A rank-zero tensor keeps the offset out of the compile cache key. Its
        # value changes between launches without changing the compiled graph.
        offset = _int_scalar(source_offset, xr)
        qsum, zsum = _run_semilagrangian_reduction(
            "ordered_scan_chunk",
            _semilagrangian_ordered_scan_chunk_kernel,
            xr,
            qsum, zsum, kb, kt, qa, dza, offset, max_sources,
        )

    za_kt = _gather_lastdim_valid(za, kt0)
    dza_kt = _gather_lastdim_valid(dza, kt0)
    qpi_kt = _gather_lastdim_valid(qpi, kt0)
    qmi_kt = _gather_lastdim_valid(qmi, kt0)
    th = (zi_high - za_kt) / torch.where(greater, dza_kt, 1.0)
    qqd_kt = 0.5 * (qpi_kt - qmi_kt)
    dqh = qqd_kt * th * th + qmi_kt * th
    qsum = qsum + torch.where(greater, dqh * dza_kt, 0.0)
    zsum = zsum + torch.where(greater, th * dza_kt, 0.0)
    xr_gt = qsum / torch.where(greater, zsum, 1.0)
    xr_candidate = torch.maximum(limit_tensor, torch.where(same, xr_same, torch.where(greater, xr_gt, xr)))
    xten_candidate = xten + (xr_candidate - xr) * (1.0 / rho) * odt
    update = same | greater
    xr_updated = torch.where(update, xr_candidate, xr)
    xten_updated = torch.where(update, xten_candidate, xten)
    nan = torch.full_like(xr_updated, float("nan"))
    return torch.where(overflow, nan, xr_updated), torch.where(overflow, nan, xten_updated), torch.any(overflow)


def _semilagrangian_ordered_precip_chunk_kernel(
    updated: torch.Tensor,
    open_column: torch.Tensor,
    za: torch.Tensor,
    dza: torch.Tensor,
    qa: torch.Tensor,
    qpi: torch.Tensor,
    qmi: torch.Tensor,
    offset: torch.Tensor,
    nz: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    for local in range(SEMI_LAGRANGIAN_ORDERED_SCAN_CHUNK_SOURCES):
        source = offset + local
        index = torch.clamp(source, max=nz - 1).reshape(1).expand(za.shape[0], 1)
        za_low = torch.gather(za, 1, index).squeeze(1)
        za_high = torch.gather(za, 1, index + 1).squeeze(1)
        dza_source = torch.gather(dza, 1, index).squeeze(1)
        qa_source = torch.gather(qa, 1, index).squeeze(1)
        qpi_source = torch.gather(qpi, 1, index).squeeze(1)
        qmi_source = torch.gather(qmi, 1, index).squeeze(1)
        valid = source < nz
        fully = open_column & valid & (za_low < 0.0) & (za_high <= 0.0)
        partial = open_column & valid & (za_low < 0.0) & (za_high > 0.0)
        th = (0.0 - za_low) / torch.where(partial, dza_source, 1.0)
        qqd = 0.5 * (qpi_source - qmi_source)
        partial_mass = (qqd * th * th + qmi_source * th) * dza_source
        delta = torch.where(fully, qa_source * dza_source, torch.where(partial, partial_mass, 0.0))
        updated = torch.where(fully | partial, updated + delta, updated)
        open_column = open_column & (~partial)
    return updated, open_column


def _semilagrangian_ordered_precip_scan(
    active_step: torch.Tensor,
    za: torch.Tensor,
    dza: torch.Tensor,
    qa: torch.Tensor,
    qpi: torch.Tensor,
    qmi: torch.Tensor,
    precip: torch.Tensor,
) -> torch.Tensor:
    nz = qa.shape[-1] - 1
    updated = precip
    open_column = active_step
    for source in range(0, nz, SEMI_LAGRANGIAN_ORDERED_SCAN_CHUNK_SOURCES):
        offset = _int_scalar(source, precip)
        updated, open_column = _run_semilagrangian_reduction(
            "ordered_precip_chunk",
            _semilagrangian_ordered_precip_chunk_kernel,
            precip,
            updated, open_column, za, dza, qa, qpi, qmi, offset, nz,
        )
    return updated


def _semilagrangian_overlap_chunk_kernel(
    zi_low_expanded: torch.Tensor,
    zi_high_expanded: torch.Tensor,
    za_source: torch.Tensor,
    dza_source: torch.Tensor,
    qpi_source: torch.Tensor,
    qmi_source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    qqd_source = 0.5 * (qpi_source - qmi_source)
    dza_denominator = torch.where(dza_source != 0.0, dza_source, torch.ones_like(dza_source))
    tl = torch.clamp((zi_low_expanded - za_source) / dza_denominator, min=0.0, max=1.0)
    th = torch.clamp((zi_high_expanded - za_source) / dza_denominator, min=0.0, max=1.0)
    overlap = th > tl
    primitive_low = qqd_source * tl * tl + qmi_source * tl
    primitive_high = qqd_source * th * th + qmi_source * th
    mass = (primitive_high - primitive_low) * dza_source
    depth = (th - tl) * dza_source
    return (
        torch.sum(torch.where(overlap, mass, 0.0), dim=-1),
        torch.sum(torch.where(overlap, depth, 0.0), dim=-1),
    )


def _semilagrangian_precip_chunk_kernel(
    active_step: torch.Tensor,
    za_source: torch.Tensor,
    dza_source: torch.Tensor,
    qpi_source: torch.Tensor,
    qmi_source: torch.Tensor,
) -> torch.Tensor:
    qqd_source = 0.5 * (qpi_source - qmi_source)
    th = torch.clamp((0.0 - za_source) / _safe_denominator_no_cache(dza_source), min=0.0, max=1.0)
    primitive_high = qqd_source * th * th + qmi_source * th
    mass = primitive_high * dza_source
    active = active_step & (th > 0.0)
    return torch.sum(torch.where(active, mass, 0.0), dim=-1)


def _semilagrangian_single_overlap_precip_chunk_kernel(
    zi_low_expanded: torch.Tensor,
    zi_high_expanded: torch.Tensor,
    za_source: torch.Tensor,
    dza_source: torch.Tensor,
    qpi_source: torch.Tensor,
    qmi_source: torch.Tensor,
    active_step: torch.Tensor,
    za_precip_source: torch.Tensor,
    dza_precip_source: torch.Tensor,
    qpi_precip_source: torch.Tensor,
    qmi_precip_source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qdelta, zdelta = _semilagrangian_overlap_chunk_kernel(
        zi_low_expanded,
        zi_high_expanded,
        za_source,
        dza_source,
        qpi_source,
        qmi_source,
    )
    precip_delta = _semilagrangian_precip_chunk_kernel(
        active_step,
        za_precip_source,
        dza_precip_source,
        qpi_precip_source,
        qmi_precip_source,
    )
    return qdelta, zdelta, precip_delta


def _semilagrangian_species_overlap_precip_chunk_kernel(
    zi_low_expanded: torch.Tensor,
    zi_high_expanded: torch.Tensor,
    za_source: torch.Tensor,
    dza_source: torch.Tensor,
    qpi_source: torch.Tensor,
    qmi_source: torch.Tensor,
    active_step: torch.Tensor,
    za_precip_source: torch.Tensor,
    dza_precip_source: torch.Tensor,
    qpi_precip_source: torch.Tensor,
    qmi_precip_source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qdelta, zdelta = _semilagrangian_overlap_chunk_kernel(
        zi_low_expanded,
        zi_high_expanded,
        za_source,
        dza_source,
        qpi_source,
        qmi_source,
    )
    precip_delta = _semilagrangian_precip_chunk_kernel(
        active_step,
        za_precip_source,
        dza_precip_source,
        qpi_precip_source,
        qmi_precip_source,
    )
    return qdelta, zdelta, precip_delta


def _semilagrangian_sedimentation_tensor_eager(
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    xr: torch.Tensor,
    xten: torch.Tensor,
    vt: torch.Tensor,
    limit_tensor: torch.Tensor,
    precip: torch.Tensor | None,
    odt: float,
    step_index: int,
    precip_count: int | None,
    steps_tensor: torch.Tensor,
    steps_float: torch.Tensor,
    step_factor: torch.Tensor,
    zi: torch.Tensor,
    limiter_prefix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    ncol, nz = xr.shape
    active_step = steps_tensor > step_index

    wi = torch.empty((ncol, nz + 1), **_factory(xr))
    wi[:, 0:1] = vt[:, 0:1]
    wi[:, 1:2] = 0.5 * (vt[:, 1:2] + vt[:, 0:1])
    if nz > 3:
        wi[:, 2 : nz - 1] = (
            (9.0 / 16.0) * (vt[:, 2 : nz - 1] + vt[:, 1 : nz - 2])
            - (1.0 / 16.0) * (vt[:, 3:nz] + vt[:, : nz - 3])
        )
    wi[:, nz - 1 : nz] = 0.5 * (vt[:, nz - 1 : nz] + vt[:, nz - 2 : nz - 1])
    wi[:, nz : nz + 1] = vt[:, nz : nz + 1]
    wi_corrected = torch.empty_like(wi)
    wi_corrected[:, :1] = wi[:, :1]
    wi_corrected[:, 1:nz] = torch.where(vt[:, 1:nz] == 0.0, vt[:, : nz - 1], wi[:, 1:nz])
    wi_corrected[:, nz : nz + 1] = wi[:, nz : nz + 1]
    wi = wi_corrected
    limiter_base = wi - limiter_prefix
    wi = torch.flip(torch.cummax(torch.flip(limiter_base, dims=(1,)), dim=1).values, dims=(1,)) + limiter_prefix

    za_body = zi - wi * step_factor[:, None]
    za = _append_column(za_body, zi[:, nz])
    dza = za[:, 1:] - za[:, :-1]
    qa = _append_zero_column_no_check(xr * dz1d / _safe_denominator_no_cache(dza[:, :nz]))

    dip = (qa[:, 2 : nz + 1] - qa[:, 1:nz]) / _safe_denominator_no_cache(dza[:, 2 : nz + 1] + dza[:, 1:nz])
    dim = (qa[:, 1:nz] - qa[:, : nz - 1]) / _safe_denominator_no_cache(dza[:, : nz - 1] + dza[:, 1:nz])
    qpi_formula = qa[:, 1:nz] + 0.5 * (dip + dim) * dza[:, 1:nz]
    qmi_formula = 2.0 * qa[:, 1:nz] - qpi_formula
    reset = (dip * dim <= 0.0) | (qpi_formula > 0.0) | (qmi_formula < 0.0)
    qpi = torch.empty_like(qa)
    qmi = torch.empty_like(qa)
    qpi[:, :1] = qa[:, :1]
    qmi[:, :1] = qa[:, :1]
    qpi[:, 1:nz] = torch.where(reset, qa[:, 1:nz], qpi_formula)
    qmi[:, 1:nz] = torch.where(reset, qa[:, 1:nz], qmi_formula)
    qpi[:, nz : nz + 1] = qa[:, nz : nz + 1]
    qmi[:, nz : nz + 1] = qa[:, nz : nz + 1]

    rr_save = xr
    mass_prefix = _prepend_zero_column(torch.cumsum(qa * dza, dim=1))
    depth_prefix = _prepend_zero_column(torch.cumsum(dza, dim=1))
    process = active_step[:, None] & (zi[:, :nz] < za[:, nz : nz + 1])

    kb_index = _lower_bound_lastdim_tensor(za, zi[:, :nz])
    kb = torch.where(kb_index <= nz, torch.clamp_min(kb_index, 1), _int_scalar(1, xr))

    kt = _lower_bound_lastdim_tensor(za, zi[:, 1 : nz + 1])
    kt = torch.where(kt <= nz + 1, kt, _int_scalar(0, xr))

    kb0 = torch.clamp(kb - 1, 0, nz)
    kt0 = torch.clamp(kt - 1, 0, nz)
    za_kb = _gather_columns_valid(za, kb0)
    dza_kb = _gather_columns_valid(dza, kb0)
    qpi_kb = _gather_columns_valid(qpi, kb0)
    qmi_kb = _gather_columns_valid(qmi, kb0)
    zi_low = zi[:, :nz]
    zi_high = zi[:, 1 : nz + 1]
    tl = (zi_low - za_kb) / _safe_denominator_no_cache(dza_kb)
    th_same = (zi_high - za_kb) / _safe_denominator_no_cache(dza_kb)
    qqd_kb = 0.5 * (qpi_kb - qmi_kb)
    qql = qqd_kb * tl * tl + qmi_kb * tl
    qqh_same = qqd_kb * th_same * th_same + qmi_kb * th_same
    xr_same = (qqh_same - qql) / _safe_denominator_no_cache(th_same - tl)

    dql = _gather_columns_valid(qa, kb0) - qql
    zsum = (1.0 - tl) * dza_kb
    qsum = dql * dza_kb
    inner_start = torch.clamp(kb, 0, nz + 1)
    inner_stop = torch.clamp(kt - 1, 0, nz + 1)
    inner_active = process & (kt > kb)
    qsum = qsum + torch.where(
        inner_active,
        _gather_columns_valid(mass_prefix, inner_stop) - _gather_columns_valid(mass_prefix, inner_start),
        0.0,
    )
    zsum = zsum + torch.where(
        inner_active,
        _gather_columns_valid(depth_prefix, inner_stop) - _gather_columns_valid(depth_prefix, inner_start),
        0.0,
    )
    za_kt = _gather_columns_valid(za, kt0)
    dza_kt = _gather_columns_valid(dza, kt0)
    qpi_kt = _gather_columns_valid(qpi, kt0)
    qmi_kt = _gather_columns_valid(qmi, kt0)
    th = (zi_high - za_kt) / _safe_denominator_no_cache(dza_kt)
    qqd_kt = 0.5 * (qpi_kt - qmi_kt)
    dqh = qqd_kt * th * th + qmi_kt * th
    zsum = zsum + th * dza_kt
    qsum = qsum + dqh * dza_kt
    same = process & (kt == kb)
    greater = process & (kt > kb)
    xr_gt = qsum / torch.where(greater, zsum, torch.ones_like(zsum))

    xr_candidate = torch.where(same, xr_same, torch.where(greater, xr_gt, xr))
    xr_candidate = torch.maximum(limit_tensor, xr_candidate)
    xten_candidate = xten + (xr_candidate - rr_save) * (1.0 / rho) * odt
    update = process & (same | greater)
    xr_updated = torch.where(update, xr_candidate, xr)
    xten_updated = torch.where(update, xten_candidate, xten)
    precip_updated = precip
    if precip is not None:
        precip_rows = ncol if precip_count is None else int(precip_count)
        fully = active_step[:precip_rows, None] & (za[:precip_rows, :nz] < 0.0) & (za[:precip_rows, 1 : nz + 1] <= 0.0)
        partial = active_step[:precip_rows, None] & (~fully) & (za[:precip_rows, :nz] < 0.0) & (za[:precip_rows, 1 : nz + 1] > 0.0)
        partial_index = torch.argmax(partial.to(torch.int64), dim=1)
        has_partial = torch.any(partial, dim=1)
        first_partial = torch.where(has_partial, partial_index, int(nz))
        levels0 = _int_arange(0, nz, xr)[None, :]
        full_mask = fully & (levels0 < first_partial[:, None])
        th_precip = (0.0 - za[:precip_rows, :nz]) / _safe_denominator_no_cache(dza[:precip_rows, :nz])
        qqd_precip = 0.5 * (qpi[:precip_rows, :nz] - qmi[:precip_rows, :nz])
        qqh_precip = qqd_precip * th_precip * th_precip + qmi[:precip_rows, :nz] * th_precip
        partial_mask = partial & (levels0 == first_partial[:, None])
        precip_updated = (
            precip
            + torch.sum(
                torch.where(
                    full_mask,
                    qa[:precip_rows, :nz] * dza[:precip_rows, :nz],
                    0.0,
                ),
                dim=1,
            )
            + torch.sum(
                torch.where(
                    partial_mask,
                    qqh_precip * dza[:precip_rows, :nz],
                    0.0,
                ),
                dim=1,
            )
        )
    return xr_updated, xten_updated, precip_updated


def _semilagrangian_sedimentation_tensor_streaming(
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    xr: torch.Tensor,
    xten: torch.Tensor,
    vt: torch.Tensor,
    limit_tensor: torch.Tensor,
    precip: torch.Tensor | None,
    odt: float,
    step_index: int,
    precip_count: int | None,
    steps_tensor: torch.Tensor,
    steps_float: torch.Tensor,
    step_factor: torch.Tensor,
    zi: torch.Tensor,
    limiter_prefix: torch.Tensor,
    return_overflow: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    ncol, nz = xr.shape
    active_step = steps_tensor > step_index

    wi = torch.empty((ncol, nz + 1), **_factory(xr))
    wi[:, 0:1] = vt[:, 0:1]
    wi[:, 1:2] = 0.5 * (vt[:, 1:2] + vt[:, 0:1])
    if nz > 3:
        wi[:, 2 : nz - 1] = (
            (9.0 / 16.0) * (vt[:, 2 : nz - 1] + vt[:, 1 : nz - 2])
            - (1.0 / 16.0) * (vt[:, 3:nz] + vt[:, : nz - 3])
        )
    wi[:, nz - 1 : nz] = 0.5 * (vt[:, nz - 1 : nz] + vt[:, nz - 2 : nz - 1])
    wi[:, nz : nz + 1] = vt[:, nz : nz + 1]
    wi_corrected = torch.empty_like(wi)
    wi_corrected[:, :1] = wi[:, :1]
    wi_corrected[:, 1:nz] = torch.where(vt[:, 1:nz] == 0.0, vt[:, : nz - 1], wi[:, 1:nz])
    wi_corrected[:, nz : nz + 1] = wi[:, nz : nz + 1]
    wi = wi_corrected
    limiter_base = wi - limiter_prefix
    wi = torch.flip(torch.cummax(torch.flip(limiter_base, dims=(1,)), dim=1).values, dims=(1,)) + limiter_prefix

    za_body = zi - wi * step_factor[:, None]
    za = _append_column(za_body, zi[:, nz])
    dza = za[:, 1:] - za[:, :-1]
    qa = _append_zero_column_no_check(xr * dz1d / _safe_denominator_no_cache(dza[:, :nz]))

    dip = (qa[:, 2 : nz + 1] - qa[:, 1:nz]) / _safe_denominator_no_cache(dza[:, 2 : nz + 1] + dza[:, 1:nz])
    dim = (qa[:, 1:nz] - qa[:, : nz - 1]) / _safe_denominator_no_cache(dza[:, : nz - 1] + dza[:, 1:nz])
    qpi_formula = qa[:, 1:nz] + 0.5 * (dip + dim) * dza[:, 1:nz]
    qmi_formula = 2.0 * qa[:, 1:nz] - qpi_formula
    reset = (dip * dim <= 0.0) | (qpi_formula > 0.0) | (qmi_formula < 0.0)
    qpi = torch.empty_like(qa)
    qmi = torch.empty_like(qa)
    qpi[:, :1] = qa[:, :1]
    qmi[:, :1] = qa[:, :1]
    qpi[:, 1:nz] = torch.where(reset, qa[:, 1:nz], qpi_formula)
    qmi[:, 1:nz] = torch.where(reset, qa[:, 1:nz], qmi_formula)
    qpi[:, nz : nz + 1] = qa[:, nz : nz + 1]
    qmi[:, nz : nz + 1] = qa[:, nz : nz + 1]

    if SEMI_LAGRANGIAN_OVERLAP_MODE in ("ordered_compiled_scan", "ordered_geometry_bound"):
        zi_low = zi[:, :nz]
        zi_high = zi[:, 1 : nz + 1]
        process = active_step[:, None] & (zi_low < za[:, nz : nz + 1])
        xr_updated, xten_updated, overflow = _semilagrangian_ordered_overlap_scan(
            zi_low, zi_high, za, dza, qa, qpi, qmi, process,
            xr, xten, rho, limit_tensor, odt,
            _semilagrangian_geometry_source_bound(zi_low, zi_high, dza)
            if SEMI_LAGRANGIAN_OVERLAP_MODE == "ordered_geometry_bound" else
            nz + 1 if SEMI_LAGRANGIAN_ORDERED_WINDOW_SOURCES is None
            else min(nz + 1, SEMI_LAGRANGIAN_ORDERED_WINDOW_SOURCES),
            zi_interfaces=zi,
        )
        precip_updated = precip
        if precip is not None:
            precip_rows = ncol if precip_count is None else int(precip_count)
            precip_updated = _semilagrangian_ordered_precip_scan(
                active_step[:precip_rows],
                za[:precip_rows], dza[:precip_rows], qa[:precip_rows],
                qpi[:precip_rows], qmi[:precip_rows], precip,
            )
        if return_overflow:
            return xr_updated, xten_updated, precip_updated, overflow
        return xr_updated, xten_updated, precip_updated

    zi_low = zi[:, :nz]
    zi_high = zi[:, 1 : nz + 1]
    qsum: torch.Tensor | None = None
    zsum: torch.Tensor | None = None
    precip_delta: torch.Tensor | None = None
    source_count = nz + 1
    source_chunk = _semi_overlap_chunk_width(prefix_elements=ncol * nz, source_count=source_count)
    zi_low_expanded = zi_low[:, :, None]
    zi_high_expanded = zi_high[:, :, None]
    for source_start in range(0, source_count, source_chunk):
        source_stop = min(source_count, source_start + source_chunk)
        za_source = za[:, None, source_start:source_stop]
        dza_source = dza[:, None, source_start:source_stop]
        qmi_source = qmi[:, None, source_start:source_stop]
        if precip is not None and source_start < nz:
            precip_rows = ncol if precip_count is None else int(precip_count)
            precip_stop = min(nz, source_stop)
            qdelta, zdelta, precip_chunk_delta = _run_semilagrangian_reduction(
                "single_overlap_precip_chunk",
                _semilagrangian_single_overlap_precip_chunk_kernel,
                xr,
                zi_low_expanded,
                zi_high_expanded,
                za_source,
                dza_source,
                qpi[:, None, source_start:source_stop],
                qmi_source,
                active_step[:precip_rows, None],
                za[:precip_rows, source_start:precip_stop],
                dza[:precip_rows, source_start:precip_stop],
                qpi[:precip_rows, source_start:precip_stop],
                qmi[:precip_rows, source_start:precip_stop],
            )
            precip_delta = precip_chunk_delta if precip_delta is None else precip_delta + precip_chunk_delta
        else:
            qdelta, zdelta = _run_semilagrangian_reduction(
                "single_overlap_chunk",
                _semilagrangian_overlap_chunk_kernel,
                xr,
                zi_low_expanded,
                zi_high_expanded,
                za_source,
                dza_source,
                qpi[:, None, source_start:source_stop],
                qmi_source,
            )
        qsum = qdelta if qsum is None else qsum + qdelta
        zsum = zdelta if zsum is None else zsum + zdelta
    if qsum is None or zsum is None:
        qsum = xr * 0.0
        zsum = xr * 0.0

    process = active_step[:, None] & (zi_low < za[:, nz : nz + 1])
    update = process & (zsum > 0.0)
    xr_candidate = qsum / torch.where(update, zsum, torch.ones_like(zsum))
    xr_candidate = torch.maximum(limit_tensor, xr_candidate)
    xten_candidate = xten + (xr_candidate - xr) * (1.0 / rho) * odt
    xr_updated = torch.where(update, xr_candidate, xr)
    xten_updated = torch.where(update, xten_candidate, xten)

    precip_updated = precip
    if precip is not None:
        if precip_delta is None:
            precip_delta = precip * 0.0
        precip_updated = precip + precip_delta
    return xr_updated, xten_updated, precip_updated


def semilagrangian_sedimentation_tensor(
    *,
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    xr: torch.Tensor,
    xten: torch.Tensor,
    vt: torch.Tensor,
    steps: torch.Tensor | int,
    limit: float | torch.Tensor,
    precip: torch.Tensor | None,
    dt: float,
    odt: float,
    step_index: int = 0,
    precip_count: int | None = None,
    geometry: SemiLagrangianSedimentationGeometry | None = None,
    return_overflow: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    ncol, nz = xr.shape
    if nz < 2:
        raise ValueError("Semi-Lagrangian sedimentation requires at least two vertical levels")
    if geometry is None:
        geometry = semilagrangian_geometry_tensor(dz1d=dz1d, steps=steps, dt=dt, odt=odt)
    if geometry.steps_tensor.shape[0] != ncol:
        raise ValueError("Semi-Lagrangian geometry batch size does not match xr")
    if geometry.zi.shape != (ncol, nz + 1):
        raise ValueError("Semi-Lagrangian geometry zi shape does not match xr")
    if geometry.limiter_prefix.shape != (ncol, nz + 1):
        raise ValueError("Semi-Lagrangian geometry limiter shape does not match xr")
    if geometry.steps_float.shape[0] != ncol or geometry.step_factor.shape[0] != ncol:
        raise ValueError("Semi-Lagrangian geometry step vectors do not match xr")
    kernel = (
        _semilagrangian_sedimentation_tensor_streaming
        if SEMI_LAGRANGIAN_STREAMING_REDUCTION_ENABLED
        else _semilagrangian_sedimentation_tensor_eager
    )
    result = _run_semilagrangian_reduction(
        "single",
        kernel,
        xr,
        dz1d,
        rho,
        xr,
        xten,
        vt,
        _limit_like(limit, xr),
        precip,
        float(odt),
        int(step_index),
        precip_count,
        geometry.steps_tensor,
        geometry.steps_float,
        geometry.step_factor,
        geometry.zi,
        geometry.limiter_prefix,
        *((True,) if return_overflow and kernel is _semilagrangian_sedimentation_tensor_streaming
          and SEMI_LAGRANGIAN_OVERLAP_MODE in ("ordered_compiled_scan", "ordered_geometry_bound") else ()),
    )
    if return_overflow and len(result) == 3:
        return (*result, torch.zeros((), dtype=torch.bool, device=xr.device))
    return result


def _gather_species_levels_valid(values: torch.Tensor, index_0based: torch.Tensor) -> torch.Tensor:
    safe_index = torch.clamp(index_0based, 0, values.shape[-1] - 1).to(dtype=torch.int64, device=values.device)
    return torch.gather(values, dim=-1, index=safe_index)


def _semilagrangian_sedimentation_species_tensor_eager(
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    xr: torch.Tensor,
    xten: torch.Tensor,
    vt: torch.Tensor,
    limit_tensor: torch.Tensor,
    precip: torch.Tensor | None,
    odt: float,
    step_index: int,
    precip_species_index: int,
    steps_tensor: torch.Tensor,
    steps_float: torch.Tensor,
    step_factor: torch.Tensor,
    zi: torch.Tensor,
    limiter_prefix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    species_count, ncol, nz = xr.shape
    active_step = steps_tensor > step_index

    wi = torch.empty((species_count, ncol, nz + 1), **_factory(xr))
    wi[:, :, 0:1] = vt[:, :, 0:1]
    wi[:, :, 1:2] = 0.5 * (vt[:, :, 1:2] + vt[:, :, 0:1])
    if nz > 3:
        wi[:, :, 2 : nz - 1] = (
            (9.0 / 16.0) * (vt[:, :, 2 : nz - 1] + vt[:, :, 1 : nz - 2])
            - (1.0 / 16.0) * (vt[:, :, 3:nz] + vt[:, :, : nz - 3])
        )
    wi[:, :, nz - 1 : nz] = 0.5 * (vt[:, :, nz - 1 : nz] + vt[:, :, nz - 2 : nz - 1])
    wi[:, :, nz : nz + 1] = vt[:, :, nz : nz + 1]
    wi_corrected = torch.empty_like(wi)
    wi_corrected[:, :, :1] = wi[:, :, :1]
    wi_corrected[:, :, 1:nz] = torch.where(vt[:, :, 1:nz] == 0.0, vt[:, :, : nz - 1], wi[:, :, 1:nz])
    wi_corrected[:, :, nz : nz + 1] = wi[:, :, nz : nz + 1]
    wi = wi_corrected
    limiter = limiter_prefix[None, :, :]
    limiter_base = wi - limiter
    wi = torch.flip(torch.cummax(torch.flip(limiter_base, dims=(2,)), dim=2).values, dims=(2,)) + limiter

    za_body = zi[None, :, :] - wi * step_factor[None, :, None]
    za = torch.empty((species_count, ncol, nz + 2), **_factory(xr))
    za[:, :, : nz + 1] = za_body
    za[:, :, nz + 1 : nz + 2] = zi[None, :, nz : nz + 1]
    dza = za[:, :, 1:] - za[:, :, :-1]
    qa = torch.empty((species_count, ncol, nz + 1), **_factory(xr))
    qa[:, :, :nz] = xr * dz1d[None, :, :] / _safe_denominator_no_cache(dza[:, :, :nz])
    qa[:, :, nz : nz + 1] = 0.0

    dip = (qa[:, :, 2 : nz + 1] - qa[:, :, 1:nz]) / _safe_denominator_no_cache(dza[:, :, 2 : nz + 1] + dza[:, :, 1:nz])
    dim = (qa[:, :, 1:nz] - qa[:, :, : nz - 1]) / _safe_denominator_no_cache(dza[:, :, : nz - 1] + dza[:, :, 1:nz])
    qpi_formula = qa[:, :, 1:nz] + 0.5 * (dip + dim) * dza[:, :, 1:nz]
    qmi_formula = 2.0 * qa[:, :, 1:nz] - qpi_formula
    reset = (dip * dim <= 0.0) | (qpi_formula > 0.0) | (qmi_formula < 0.0)
    qpi = torch.empty_like(qa)
    qmi = torch.empty_like(qa)
    qpi[:, :, :1] = qa[:, :, :1]
    qmi[:, :, :1] = qa[:, :, :1]
    qpi[:, :, 1:nz] = torch.where(reset, qa[:, :, 1:nz], qpi_formula)
    qmi[:, :, 1:nz] = torch.where(reset, qa[:, :, 1:nz], qmi_formula)
    qpi[:, :, nz : nz + 1] = qa[:, :, nz : nz + 1]
    qmi[:, :, nz : nz + 1] = qa[:, :, nz : nz + 1]

    rr_save = xr
    mass_prefix = torch.empty((species_count, ncol, nz + 2), **_factory(xr))
    depth_prefix = torch.empty((species_count, ncol, nz + 2), **_factory(xr))
    mass_prefix[:, :, :1] = 0.0
    depth_prefix[:, :, :1] = 0.0
    mass_prefix[:, :, 1:] = torch.cumsum(qa * dza, dim=2)
    depth_prefix[:, :, 1:] = torch.cumsum(dza, dim=2)
    zi_low = zi[None, :, :nz]
    zi_high = zi[None, :, 1 : nz + 1]
    process = active_step[None, :, None] & (zi_low < za[:, :, nz : nz + 1])

    kb_index = _lower_bound_lastdim_tensor(za, zi_low)
    kb = torch.where(kb_index <= nz, torch.clamp_min(kb_index, 1), _int_scalar(1, xr))

    kt = _lower_bound_lastdim_tensor(za, zi_high)
    kt = torch.where(kt <= nz + 1, kt, _int_scalar(0, xr))

    kb0 = torch.clamp(kb - 1, 0, nz)
    kt0 = torch.clamp(kt - 1, 0, nz)
    za_kb = _gather_species_levels_valid(za, kb0)
    dza_kb = _gather_species_levels_valid(dza, kb0)
    qpi_kb = _gather_species_levels_valid(qpi, kb0)
    qmi_kb = _gather_species_levels_valid(qmi, kb0)
    tl = (zi_low - za_kb) / _safe_denominator_no_cache(dza_kb)
    th_same = (zi_high - za_kb) / _safe_denominator_no_cache(dza_kb)
    qqd_kb = 0.5 * (qpi_kb - qmi_kb)
    qql = qqd_kb * tl * tl + qmi_kb * tl
    qqh_same = qqd_kb * th_same * th_same + qmi_kb * th_same
    xr_same = (qqh_same - qql) / _safe_denominator_no_cache(th_same - tl)

    dql = _gather_species_levels_valid(qa, kb0) - qql
    zsum = (1.0 - tl) * dza_kb
    qsum = dql * dza_kb
    inner_start = torch.clamp(kb, 0, nz + 1)
    inner_stop = torch.clamp(kt - 1, 0, nz + 1)
    inner_active = process & (kt > kb)
    qsum = qsum + torch.where(
        inner_active,
        _gather_species_levels_valid(mass_prefix, inner_stop) - _gather_species_levels_valid(mass_prefix, inner_start),
        0.0,
    )
    zsum = zsum + torch.where(
        inner_active,
        _gather_species_levels_valid(depth_prefix, inner_stop) - _gather_species_levels_valid(depth_prefix, inner_start),
        0.0,
    )
    za_kt = _gather_species_levels_valid(za, kt0)
    dza_kt = _gather_species_levels_valid(dza, kt0)
    qpi_kt = _gather_species_levels_valid(qpi, kt0)
    qmi_kt = _gather_species_levels_valid(qmi, kt0)
    th = (zi_high - za_kt) / _safe_denominator_no_cache(dza_kt)
    qqd_kt = 0.5 * (qpi_kt - qmi_kt)
    dqh = qqd_kt * th * th + qmi_kt * th
    zsum = zsum + th * dza_kt
    qsum = qsum + dqh * dza_kt
    same = process & (kt == kb)
    greater = process & (kt > kb)
    xr_gt = qsum / torch.where(greater, zsum, torch.ones_like(zsum))

    xr_candidate = torch.where(same, xr_same, torch.where(greater, xr_gt, xr))
    xr_candidate = torch.maximum(limit_tensor, xr_candidate)
    xten_candidate = xten + (xr_candidate - rr_save) * (1.0 / rho[None, :, :]) * odt
    update = process & (same | greater)
    xr_updated = torch.where(update, xr_candidate, xr)
    xten_updated = torch.where(update, xten_candidate, xten)
    precip_updated = precip
    if precip is not None:
        za_p = za[precip_species_index]
        dza_p = dza[precip_species_index]
        qa_p = qa[precip_species_index]
        qpi_p = qpi[precip_species_index]
        qmi_p = qmi[precip_species_index]
        fully = active_step[:, None] & (za_p[:, :nz] < 0.0) & (za_p[:, 1 : nz + 1] <= 0.0)
        partial = active_step[:, None] & (~fully) & (za_p[:, :nz] < 0.0) & (za_p[:, 1 : nz + 1] > 0.0)
        partial_index = torch.argmax(partial.to(torch.int64), dim=1)
        has_partial = torch.any(partial, dim=1)
        first_partial = torch.where(has_partial, partial_index, int(nz))
        levels0 = _int_arange(0, nz, xr)[None, :]
        full_mask = fully & (levels0 < first_partial[:, None])
        th_precip = (0.0 - za_p[:, :nz]) / _safe_denominator_no_cache(dza_p[:, :nz])
        qqd_precip = 0.5 * (qpi_p[:, :nz] - qmi_p[:, :nz])
        qqh_precip = qqd_precip * th_precip * th_precip + qmi_p[:, :nz] * th_precip
        partial_mask = partial & (levels0 == first_partial[:, None])
        precip_updated = (
            precip
            + torch.sum(torch.where(full_mask, qa_p[:, :nz] * dza_p[:, :nz], 0.0), dim=1)
            + torch.sum(torch.where(partial_mask, qqh_precip * dza_p[:, :nz], 0.0), dim=1)
        )
    return xr_updated, xten_updated, precip_updated


def _semilagrangian_sedimentation_species_tensor_streaming(
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    xr: torch.Tensor,
    xten: torch.Tensor,
    vt: torch.Tensor,
    limit_tensor: torch.Tensor,
    precip: torch.Tensor | None,
    odt: float,
    step_index: int,
    precip_species_index: int,
    steps_tensor: torch.Tensor,
    steps_float: torch.Tensor,
    step_factor: torch.Tensor,
    zi: torch.Tensor,
    limiter_prefix: torch.Tensor,
    return_overflow: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    species_count, ncol, nz = xr.shape
    active_step = steps_tensor > step_index

    wi = torch.empty((species_count, ncol, nz + 1), **_factory(xr))
    wi[:, :, 0:1] = vt[:, :, 0:1]
    wi[:, :, 1:2] = 0.5 * (vt[:, :, 1:2] + vt[:, :, 0:1])
    if nz > 3:
        wi[:, :, 2 : nz - 1] = (
            (9.0 / 16.0) * (vt[:, :, 2 : nz - 1] + vt[:, :, 1 : nz - 2])
            - (1.0 / 16.0) * (vt[:, :, 3:nz] + vt[:, :, : nz - 3])
        )
    wi[:, :, nz - 1 : nz] = 0.5 * (vt[:, :, nz - 1 : nz] + vt[:, :, nz - 2 : nz - 1])
    wi[:, :, nz : nz + 1] = vt[:, :, nz : nz + 1]
    wi_corrected = torch.empty_like(wi)
    wi_corrected[:, :, :1] = wi[:, :, :1]
    wi_corrected[:, :, 1:nz] = torch.where(vt[:, :, 1:nz] == 0.0, vt[:, :, : nz - 1], wi[:, :, 1:nz])
    wi_corrected[:, :, nz : nz + 1] = wi[:, :, nz : nz + 1]
    wi = wi_corrected
    limiter = limiter_prefix[None, :, :]
    limiter_base = wi - limiter
    wi = torch.flip(torch.cummax(torch.flip(limiter_base, dims=(2,)), dim=2).values, dims=(2,)) + limiter

    za_body = zi[None, :, :] - wi * step_factor[None, :, None]
    za = torch.empty((species_count, ncol, nz + 2), **_factory(xr))
    za[:, :, : nz + 1] = za_body
    za[:, :, nz + 1 : nz + 2] = zi[None, :, nz : nz + 1]
    dza = za[:, :, 1:] - za[:, :, :-1]
    qa = torch.empty((species_count, ncol, nz + 1), **_factory(xr))
    qa[:, :, :nz] = xr * dz1d[None, :, :] / _safe_denominator_no_cache(dza[:, :, :nz])
    qa[:, :, nz : nz + 1] = 0.0

    dip = (qa[:, :, 2 : nz + 1] - qa[:, :, 1:nz]) / _safe_denominator_no_cache(dza[:, :, 2 : nz + 1] + dza[:, :, 1:nz])
    dim = (qa[:, :, 1:nz] - qa[:, :, : nz - 1]) / _safe_denominator_no_cache(dza[:, :, : nz - 1] + dza[:, :, 1:nz])
    qpi_formula = qa[:, :, 1:nz] + 0.5 * (dip + dim) * dza[:, :, 1:nz]
    qmi_formula = 2.0 * qa[:, :, 1:nz] - qpi_formula
    reset = (dip * dim <= 0.0) | (qpi_formula > 0.0) | (qmi_formula < 0.0)
    qpi = torch.empty_like(qa)
    qmi = torch.empty_like(qa)
    qpi[:, :, :1] = qa[:, :, :1]
    qmi[:, :, :1] = qa[:, :, :1]
    qpi[:, :, 1:nz] = torch.where(reset, qa[:, :, 1:nz], qpi_formula)
    qmi[:, :, 1:nz] = torch.where(reset, qa[:, :, 1:nz], qmi_formula)
    qpi[:, :, nz : nz + 1] = qa[:, :, nz : nz + 1]
    qmi[:, :, nz : nz + 1] = qa[:, :, nz : nz + 1]

    if SEMI_LAGRANGIAN_OVERLAP_MODE in ("ordered_compiled_scan", "ordered_geometry_bound"):
        zi_low = zi[None, :, :nz].expand(species_count, -1, -1)
        zi_high = zi[None, :, 1 : nz + 1].expand(species_count, -1, -1)
        process = active_step[None, :, None] & (zi_low < za[:, :, nz : nz + 1])
        xr_updated, xten_updated, overflow = _semilagrangian_ordered_overlap_scan(
            zi_low, zi_high, za, dza, qa, qpi, qmi, process,
            xr, xten, rho[None, :, :], limit_tensor, odt,
            _semilagrangian_geometry_source_bound(zi_low, zi_high, dza)
            if SEMI_LAGRANGIAN_OVERLAP_MODE == "ordered_geometry_bound" else
            nz + 1 if SEMI_LAGRANGIAN_ORDERED_WINDOW_SOURCES is None
            else min(nz + 1, SEMI_LAGRANGIAN_ORDERED_WINDOW_SOURCES),
            zi_interfaces=zi[None, :, :].expand(species_count, -1, -1),
        )
        precip_updated = precip
        if precip is not None:
            precip_updated = _semilagrangian_ordered_precip_scan(
                active_step,
                za[precip_species_index], dza[precip_species_index],
                qa[precip_species_index], qpi[precip_species_index],
                qmi[precip_species_index], precip,
            )
        if return_overflow:
            return xr_updated, xten_updated, precip_updated, overflow
        return xr_updated, xten_updated, precip_updated

    zi_low = zi[None, :, :nz]
    zi_high = zi[None, :, 1 : nz + 1]
    qsum: torch.Tensor | None = None
    zsum: torch.Tensor | None = None
    precip_delta: torch.Tensor | None = None
    source_count = nz + 1
    source_chunk = _semi_overlap_chunk_width(
        prefix_elements=species_count * ncol * nz,
        source_count=source_count,
    )
    zi_low_expanded = zi_low[:, :, :, None]
    zi_high_expanded = zi_high[:, :, :, None]
    for source_start in range(0, source_count, source_chunk):
        source_stop = min(source_count, source_start + source_chunk)
        za_source = za[:, :, None, source_start:source_stop]
        dza_source = dza[:, :, None, source_start:source_stop]
        qmi_source = qmi[:, :, None, source_start:source_stop]
        if precip is not None and source_start < nz:
            precip_stop = min(nz, source_stop)
            za_p = za[precip_species_index]
            dza_p = dza[precip_species_index]
            qpi_p = qpi[precip_species_index]
            qmi_p = qmi[precip_species_index]
            qdelta, zdelta, precip_chunk_delta = _run_semilagrangian_reduction(
                "species_overlap_precip_chunk",
                _semilagrangian_species_overlap_precip_chunk_kernel,
                xr,
                zi_low_expanded,
                zi_high_expanded,
                za_source,
                dza_source,
                qpi[:, :, None, source_start:source_stop],
                qmi_source,
                active_step[:, None],
                za_p[:, source_start:precip_stop],
                dza_p[:, source_start:precip_stop],
                qpi_p[:, source_start:precip_stop],
                qmi_p[:, source_start:precip_stop],
            )
            precip_delta = precip_chunk_delta if precip_delta is None else precip_delta + precip_chunk_delta
        else:
            qdelta, zdelta = _run_semilagrangian_reduction(
                "species_overlap_chunk",
                _semilagrangian_overlap_chunk_kernel,
                xr,
                zi_low_expanded,
                zi_high_expanded,
                za_source,
                dza_source,
                qpi[:, :, None, source_start:source_stop],
                qmi_source,
            )
        qsum = qdelta if qsum is None else qsum + qdelta
        zsum = zdelta if zsum is None else zsum + zdelta
    if qsum is None or zsum is None:
        qsum = xr * 0.0
        zsum = xr * 0.0

    process = active_step[None, :, None] & (zi_low < za[:, :, nz : nz + 1])
    update = process & (zsum > 0.0)
    xr_candidate = qsum / torch.where(update, zsum, torch.ones_like(zsum))
    xr_candidate = torch.maximum(limit_tensor, xr_candidate)
    xten_candidate = xten + (xr_candidate - xr) * (1.0 / rho[None, :, :]) * odt
    xr_updated = torch.where(update, xr_candidate, xr)
    xten_updated = torch.where(update, xten_candidate, xten)

    precip_updated = precip
    if precip is not None:
        if precip_delta is None:
            precip_delta = precip * 0.0
        precip_updated = precip + precip_delta
    return xr_updated, xten_updated, precip_updated


def _semilagrangian_fortran_ordered_species_tensor(
    *,
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    xr: torch.Tensor,
    xten: torch.Tensor,
    vt: torch.Tensor,
    limit_tensor: torch.Tensor,
    precip: torch.Tensor | tuple[torch.Tensor, ...] | None,
    precip_species_index: int | tuple[int, ...],
    steps_tensor: torch.Tensor,
    dt: float,
    odt: float,
    step_index: int | torch.Tensor,
    zi: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | tuple[torch.Tensor, ...] | None]:
    """One ``semilagrangian_sedimentation`` call per species in the Fortran operation order.

    ``module_mp_tempo_main.F90:1562-1749``: ``zi`` by sequential sums, the ``wi``
    diffusivity limiter as the top-down recurrence on ``decfl``, arrival points
    ``zi - wi*dt*(1/steps)``, the same-cell quadratic formula, source sums in
    increasing source order (masked adds of exact zeros keep that order), and
    ``max(xr, limit)`` with the tendency update on every processed level. Discrete
    choices are masks; loops run over static level counts, never tensor values.
    Only the ordered accumulations and the ``decfl`` recurrence loop over levels;
    level-independent operands are formed once with the same per-element arithmetic.

    Species of several calls share one evaluation: ``steps_tensor`` is [ncol] (one count for
    every species) or [species, ncol] (each species its own call's count), and ``precip`` with
    ``precip_species_index`` may be tuples (one accumulator per call, read from that call's
    species). ``zi`` may be passed in from ``_semilagrangian_ordered_zi(dz1d)``.
    """
    species_count, ncol, nz = xr.shape
    if nz < 4:
        raise ValueError("Fortran-ordered semi-Lagrangian sedimentation requires at least four levels")
    active_step = steps_tensor > step_index
    steps_float = torch.clamp_min(steps_tensor.to(dtype=xr.dtype), 1.0)
    inv_steps = 1.0 / steps_float
    con1 = 0.05
    dz_columns = dz1d.unbind(dim=1)
    if zi is None:
        zi = _semilagrangian_ordered_zi(dz1d)
    v = vt[..., : nz + 1]
    mid = (9.0 / 16.0) * (v[..., 2 : nz - 1] + v[..., 1 : nz - 2]) - (1.0 / 16.0) * (v[..., 3:nz] + v[..., : nz - 3])
    interior = torch.cat((0.5 * (v[..., 1:2] + v[..., 0:1]), mid,
                          0.5 * (v[..., nz - 1:nz] + v[..., nz - 2:nz - 1])), dim=-1)
    # The v==0 fix-up reads v[j-1], never the updated wi, so it is elementwise.
    interior = torch.where(v[..., 1:nz] == 0.0, v[..., 0 : nz - 1], interior)
    wi_columns = [v[..., 0], *interior.unbind(dim=-1), v[..., nz]]
    # Limiter term ((con1*dz)*odt)*steps does not depend on the recurrence.
    limiter = (con1 * dz1d * odt * steps_float[..., None]).unbind(dim=-1)
    for j in range(nz - 1, -1, -1):
        decfl = (wi_columns[j + 1] - wi_columns[j]) * dt * inv_steps / dz_columns[j]
        limited = wi_columns[j + 1] - limiter[j]
        wi_columns[j] = torch.where(decfl > con1, limited, wi_columns[j])
    # Level-major assembly keeps the recurrence rows contiguous. Stacking on
    # the last axis makes generated per-level copy kernels write with stride nz.
    wi = torch.stack(wi_columns, dim=0).movedim(0, -1)
    za = torch.cat((zi - wi * dt * inv_steps[..., None],
                    zi[None, :, nz:nz + 1].expand(species_count, -1, -1)), dim=-1)
    dza = za[..., 1:] - za[..., :-1]
    qa = torch.cat((xr * dz1d / _safe_denominator_no_cache(dza[..., :nz]),
                    torch.zeros_like(xr[..., :1])), dim=-1)
    dip = (qa[..., 2 : nz + 1] - qa[..., 1:nz]) / _safe_denominator_no_cache(dza[..., 2 : nz + 1] + dza[..., 1:nz])
    dim = (qa[..., 1:nz] - qa[..., : nz - 1]) / _safe_denominator_no_cache(dza[..., : nz - 1] + dza[..., 1:nz])
    qpi_formula = qa[..., 1:nz] + 0.5 * (dip + dim) * dza[..., 1:nz]
    qmi_formula = 2.0 * qa[..., 1:nz] - qpi_formula
    reset = (dip * dim <= 0.0) | (qpi_formula > 0.0) | (qmi_formula < 0.0)
    qpi = torch.cat((qa[..., :1], torch.where(reset, qa[..., 1:nz], qpi_formula), qa[..., nz:]), dim=-1)
    qmi = torch.cat((qa[..., :1], torch.where(reset, qa[..., 1:nz], qmi_formula), qa[..., nz:]), dim=-1)

    zi_low = zi[:, :nz].expand(species_count, ncol, nz)
    zi_high = zi[:, 1:].expand(species_count, ncol, nz)
    step_mask = active_step.expand(species_count, ncol)
    process = step_mask[..., None] & (zi_low < za[..., nz : nz + 1])
    # Neighboring destinations share interfaces; retain the same left-insertion rule.
    interfaces = zi.expand(species_count, ncol, nz + 1)
    interface_indices = torch.searchsorted(za.contiguous(), interfaces.contiguous(), right=False)
    kb_index = interface_indices[..., :-1]
    kt = interface_indices[..., 1:]
    kb = torch.clamp(kb_index, 1, nz)
    same = process & (kt == kb)
    greater = process & (kt > kb)
    kb0 = kb - 1
    kt0 = kt - 1
    za_kb = _gather_lastdim_valid(za, kb0)
    dza_kb = _gather_lastdim_valid(dza, kb0)
    qa_kb = _gather_lastdim_valid(qa, kb0)
    qpi_kb = _gather_lastdim_valid(qpi, kb0)
    qmi_kb = _gather_lastdim_valid(qmi, kb0)
    either = same | greater
    tl = (zi_low - za_kb) / torch.where(either, dza_kb, 1.0)
    th_same = (zi_high - za_kb) / torch.where(same, dza_kb, 1.0)
    qqd_kb = 0.5 * (qpi_kb - qmi_kb)
    qqh_same = qqd_kb * (th_same * th_same) + qmi_kb * th_same
    qql = qqd_kb * (tl * tl) + qmi_kb * tl
    xr_same = (qqh_same - qql) / torch.where(same, th_same - tl, 1.0)
    zsum = (1.0 - tl) * dza_kb
    qsum = (qa_kb - qql) * dza_kb
    # Inner arrival cells m (kb0 < m < kt0) of destination k. kb0 and kt0 never decrease with k,
    # so each cell is inner to at most one destination: the last k with kb0 < m. One pass over
    # m in increasing order forms each destination's sum as the Fortran loop does (base, then
    # cells kb+1..kt-1), in O(nz) per column; the sum is read back at the last inner cell.
    sources = torch.arange(nz, device=xr.device).expand(species_count, ncol, nz).contiguous()
    owner = torch.searchsorted(kb0.contiguous(), sources, right=False) - 1
    owner_index = torch.clamp_min(owner, 0)
    inner = ((owner >= 0) & torch.gather(greater, -1, owner_index)
             & (sources < torch.gather(kt0, -1, owner_index)))
    first_inner = inner & (sources == torch.gather(kb0, -1, owner_index) + 1)
    z_base = torch.gather(zsum, -1, owner_index).unbind(dim=-1)
    q_base = torch.gather(qsum, -1, owner_index).unbind(dim=-1)
    z_terms = dza[..., :nz].unbind(dim=-1)
    q_terms = (qa[..., :nz] * dza[..., :nz]).unbind(dim=-1)
    starts = first_inner.unbind(dim=-1)
    z_running, q_running = z_base[0], q_base[0]
    z_columns, q_columns = [], []
    for m in range(nz):
        # Values off a destination's inner run are never read back.
        z_running = torch.where(starts[m], z_base[m] + z_terms[m], z_running + z_terms[m])
        q_running = torch.where(starts[m], q_base[m] + q_terms[m], q_running + q_terms[m])
        z_columns.append(z_running)
        q_columns.append(q_running)
    z_partial = torch.stack(z_columns, dim=0).movedim(0, -1)
    q_partial = torch.stack(q_columns, dim=0).movedim(0, -1)
    has_inner = greater & (kt0 - kb0 >= 2)
    last_inner = torch.clamp(kt0 - 1, 0, nz - 1)
    zsum = torch.where(has_inner, torch.gather(z_partial, -1, last_inner), zsum)
    qsum = torch.where(has_inner, torch.gather(q_partial, -1, last_inner), qsum)
    za_kt = _gather_lastdim_valid(za, kt0)
    dza_kt = _gather_lastdim_valid(dza, kt0)
    qpi_kt = _gather_lastdim_valid(qpi, kt0)
    qmi_kt = _gather_lastdim_valid(qmi, kt0)
    th = (zi_high - za_kt) / torch.where(greater, dza_kt, 1.0)
    qqd_kt = 0.5 * (qpi_kt - qmi_kt)
    dqh = qqd_kt * (th * th) + qmi_kt * th
    zsum = zsum + th * dza_kt
    qsum = qsum + dqh * dza_kt
    xr_greater = qsum / torch.where(greater, zsum, 1.0)
    candidate = torch.where(same, xr_same, torch.where(greater, xr_greater, xr))
    candidate = torch.maximum(candidate, limit_tensor)
    xten_candidate = xten + (candidate - xr) * (1.0 / rho[None, :, :]) * odt
    xr_updated = torch.where(process, candidate, xr)
    xten_updated = torch.where(process, xten_candidate, xten)

    if precip is None:
        return xr_updated, xten_updated, None
    if isinstance(precip, torch.Tensor):
        return xr_updated, xten_updated, _semilagrangian_ordered_precip(
            precip, int(precip_species_index), step_mask, za, dza, qa, qpi, qmi)
    return xr_updated, xten_updated, tuple(
        _semilagrangian_ordered_precip(value, int(index), step_mask, za, dza, qa, qpi, qmi)
        for value, index in zip(precip, precip_species_index, strict=True))


def _semilagrangian_ordered_zi(dz1d: torch.Tensor) -> torch.Tensor:
    """Interface heights ``zi`` of ``semilagrangian_sedimentation`` by sequential sums of ``dz``."""
    ncol, nz = dz1d.shape
    dz_columns = dz1d.unbind(dim=1)
    zi_columns = [torch.zeros_like(dz_columns[0])]
    for k in range(nz):
        zi_columns.append(zi_columns[-1] + dz_columns[k])
    return torch.stack(zi_columns, dim=1)


def _semilagrangian_ordered_precip(
    precip: torch.Tensor,
    precip_species_index: int,
    step_mask: torch.Tensor,
    za: torch.Tensor,
    dza: torch.Tensor,
    qa: torch.Tensor,
    qpi: torch.Tensor,
    qmi: torch.Tensor,
) -> torch.Tensor:
    """The surface accumulator of one call, from its precipitating species (Fortran level order)."""
    nz = qa.shape[-1] - 1
    precip_updated = precip
    rows = precip.shape[0]
    za_low = za[precip_species_index, :rows, :nz]
    za_high = za[precip_species_index, :rows, 1 : nz + 1]
    dza_p = dza[precip_species_index, :rows, :nz]
    qa_p = qa[precip_species_index, :rows, :nz]
    qpi_p = qpi[precip_species_index, :rows, :nz]
    qmi_p = qmi[precip_species_index, :rows, :nz]
    below = za_low < 0.0
    raw_fully = below & (za_high <= 0.0)
    raw_partial = (~raw_fully) & below & (za_high > 0.0)
    # A column stays open until its first partial level: exclusive count of
    # earlier partial levels is zero (integer arithmetic, exact).
    earlier = torch.cumsum(raw_partial.to(torch.int32), dim=-1) - raw_partial.to(torch.int32)
    open_column = step_mask[precip_species_index, :rows, None] & (earlier == 0)
    fully = open_column & raw_fully
    partial = open_column & raw_partial
    th_p = (0.0 - za_low) / torch.where(partial, dza_p, 1.0)
    qqd_p = 0.5 * (qpi_p - qmi_p)
    qqh_p = qqd_p * (th_p * th_p) + qmi_p * th_p
    if parallel_masked_sums():
        return masked_sum(precip, (fully | partial).t(), torch.where(fully, qa_p * dza_p, qqh_p * dza_p).t())
    # Level-major copies make each ordered precip term a contiguous slice.
    contribution = torch.where(fully, qa_p * dza_p, qqh_p * dza_p).t().contiguous().unbind(dim=0)
    selected = (fully | partial).t().contiguous().unbind(dim=0)
    for k in range(nz):
        precip_updated = torch.where(selected[k], precip_updated + contribution[k], precip_updated)
    return precip_updated


def semilagrangian_sedimentation_species_tensor(
    *,
    dz1d: torch.Tensor,
    rho: torch.Tensor,
    xr: torch.Tensor,
    xten: torch.Tensor,
    vt: torch.Tensor,
    steps: torch.Tensor | int,
    limits: tuple[float | torch.Tensor, ...],
    precip: torch.Tensor | None,
    dt: float,
    odt: float,
    step_index: int = 0,
    precip_species_index: int = 0,
    geometry: SemiLagrangianSedimentationGeometry | None = None,
    return_overflow: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if xr.ndim != 3:
        raise ValueError("species semi-Lagrangian sedimentation expects xr shape [nspecies,ncol,nz]")
    species_count, ncol, nz = xr.shape
    if nz < 2:
        raise ValueError("Semi-Lagrangian sedimentation requires at least two vertical levels")
    if xten.shape != xr.shape:
        raise ValueError("species semi-Lagrangian xten shape must match xr")
    if vt.shape[:2] != (species_count, ncol) or vt.shape[2] < nz + 1:
        raise ValueError("species semi-Lagrangian vt shape must match xr with nz+1 vertical entries")
    if dz1d.shape != (ncol, nz) or rho.shape != (ncol, nz):
        raise ValueError("species semi-Lagrangian dz1d/rho shape must be [ncol,nz]")
    if not 0 <= precip_species_index < species_count:
        raise ValueError("precip_species_index is out of range")
    if SEMI_LAGRANGIAN_OVERLAP_MODE == "fortran_ordered":
        # The Fortran-ordered evaluation reads only the substep counts, not the geometry.
        if geometry is not None:
            steps_tensor = geometry.steps_tensor
        elif isinstance(steps, torch.Tensor):
            steps_tensor = steps.to(torch.int64)
        else:
            steps_tensor = torch.full((ncol,), int(steps), dtype=torch.int64, device=dz1d.device)
        if steps_tensor.shape[0] != ncol:
            raise ValueError("Semi-Lagrangian geometry batch size does not match xr")
        ordered = _semilagrangian_fortran_ordered_species_tensor(
            dz1d=dz1d,
            rho=rho,
            xr=xr,
            xten=xten,
            vt=vt,
            limit_tensor=_species_limit_tensor(limits, reference=xr),
            precip=precip,
            precip_species_index=int(precip_species_index),
            steps_tensor=steps_tensor,
            dt=float(dt),
            odt=float(odt),
            step_index=int(step_index),
        )
        if return_overflow:
            return (*ordered, torch.zeros((), dtype=torch.bool, device=xr.device))
        return ordered
    if geometry is None:
        geometry = semilagrangian_geometry_tensor(dz1d=dz1d, steps=steps, dt=dt, odt=odt)
    if geometry.steps_tensor.shape[0] != ncol:
        raise ValueError("Semi-Lagrangian geometry batch size does not match xr")
    if geometry.zi.shape != (ncol, nz + 1):
        raise ValueError("Semi-Lagrangian geometry zi shape does not match xr")
    if geometry.limiter_prefix.shape != (ncol, nz + 1):
        raise ValueError("Semi-Lagrangian geometry limiter shape does not match xr")
    if geometry.steps_float.shape[0] != ncol or geometry.step_factor.shape[0] != ncol:
        raise ValueError("Semi-Lagrangian geometry step vectors do not match xr")
    kernel = (
        _semilagrangian_sedimentation_species_tensor_streaming
        if SEMI_LAGRANGIAN_STREAMING_REDUCTION_ENABLED
        else _semilagrangian_sedimentation_species_tensor_eager
    )
    result = _run_semilagrangian_reduction(
        "species",
        kernel,
        xr,
        dz1d,
        rho,
        xr,
        xten,
        vt,
        _species_limit_tensor(limits, reference=xr),
        precip,
        float(odt),
        int(step_index),
        int(precip_species_index),
        geometry.steps_tensor,
        geometry.steps_float,
        geometry.step_factor,
        geometry.zi,
        geometry.limiter_prefix,
        *((True,) if return_overflow and kernel is _semilagrangian_sedimentation_species_tensor_streaming
          and SEMI_LAGRANGIAN_OVERLAP_MODE in ("ordered_compiled_scan", "ordered_geometry_bound") else ()),
    )
    if return_overflow and len(result) == 3:
        return (*result, torch.zeros((), dtype=torch.bool, device=xr.device))
    return result


def freeze_cloud_melt_ice_tensor(
    *,
    temp: torch.Tensor,
    rho: torch.Tensor,
    ocp: torch.Tensor,
    lvap: torch.Tensor,
    qi1d: torch.Tensor,
    ni1d: torch.Tensor,
    qiten: torch.Tensor,
    niten: torch.Tensor,
    qc1d: torch.Tensor,
    qcten: torch.Tensor,
    ncten: torch.Tensor,
    tten: torch.Tensor,
    dt: float,
    odt: float,
    nc1d: torch.Tensor | None = None,
    ncsave: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if TENSOR_COMPILE_ISLAND_CUDA and temp.device.type in TENSOR_COMPILE_ISLAND_DEVICES:
        if nc1d is not None:
            cloud_number_base = nc1d
        elif ncsave is not None:
            cloud_number_base = ncsave / rho
        else:
            cloud_number_base = float(constants.NT_C_L) / rho
        values = _run_tensor_compile_island(
            "freeze_cloud_melt_ice",
            _freeze_cloud_melt_ice_kernel,
            temp,
            temp,
            rho,
            ocp,
            lvap,
            qi1d,
            ni1d,
            qiten,
            niten,
            qc1d,
            qcten,
            ncten,
            tten,
            cloud_number_base,
            float(dt),
            float(odt),
        )
        return dict(zip(("qiten", "niten", "qcten", "ncten", "tten"), values, strict=True))
    qiten_work = qiten
    niten_work = niten
    qcten_work = qcten
    ncten_work = ncten
    tten_work = tten
    dt_value = _scalar(dt, temp)
    odt_value = _scalar(odt, temp)

    xri = torch.clamp_min(qi1d + qiten_work * dt_value, _scalar(0.0, temp))
    # Match cloud/ice presence checks; roundoff remnants cannot transfer moments.
    melt = (temp > _scalar(constants.T0, temp)) & (xri > _scalar(constants.R1, temp))
    qcten_work = torch.where(melt, qcten_work + xri * odt_value, qcten_work)
    ncten_work = torch.where(melt, ncten_work + ni1d * odt_value, ncten_work)
    qiten_work = torch.where(melt, qiten_work - xri * odt_value, qiten_work)
    niten_work = torch.where(melt, -ni1d * odt_value, niten_work)
    tten_work = torch.where(melt, tten_work - _scalar(constants.LFUS, temp) * ocp * xri * odt_value, tten_work)

    xrc = torch.clamp_min(qc1d + qcten_work * dt_value, _scalar(0.0, temp))
    freeze = (temp < _scalar(_HOMOGENEOUS_FREEZING_TEMPERATURE, temp)) & (xrc > _scalar(constants.R1, temp))
    if nc1d is not None:
        xnc = nc1d + ncten_work * dt_value
    elif ncsave is not None:
        xnc = ncsave / rho + ncten_work * dt_value
    else:
        xnc = _scalar(constants.NT_C_L, temp) / rho + ncten_work * dt_value
    lfus2 = _scalar(constants.LSUB, temp) - lvap
    qiten_work = torch.where(freeze, qiten_work + xrc * odt_value, qiten_work)
    niten_work = torch.where(freeze, niten_work + xnc * odt_value, niten_work)
    qcten_work = torch.where(freeze, qcten_work - xrc * odt_value, qcten_work)
    ncten_work = torch.where(freeze, ncten_work - xnc * odt_value, ncten_work)
    tten_work = torch.where(freeze, tten_work + lfus2 * ocp * xrc * odt_value, tten_work)
    return {
        "qiten": qiten_work,
        "niten": niten_work,
        "qcten": qcten_work,
        "ncten": ncten_work,
        "tten": tten_work,
    }


def _freeze_cloud_melt_ice_kernel(
    temp: torch.Tensor,
    rho: torch.Tensor,
    ocp: torch.Tensor,
    lvap: torch.Tensor,
    qi1d: torch.Tensor,
    ni1d: torch.Tensor,
    qiten: torch.Tensor,
    niten: torch.Tensor,
    qc1d: torch.Tensor,
    qcten: torch.Tensor,
    ncten: torch.Tensor,
    tten: torch.Tensor,
    cloud_number_base: torch.Tensor,
    dt: float,
    odt: float,
    _const_segment_e_T0=constants.T0,
    _const_segment_n_LFUS=constants.LFUS,
    _const_segment_j_LSUB=constants.LSUB,
    _const_homogeneous_freezing_temperature=_HOMOGENEOUS_FREEZING_TEMPERATURE,
    _const_segment_e_R1=constants.R1,
) -> tuple[torch.Tensor, ...]:
    xri = torch.clamp_min(qi1d + qiten * dt, 0.0)
    melt = (temp > float(_const_segment_e_T0)) & (xri > float(_const_segment_e_R1))
    qcten = torch.where(melt, qcten + xri * odt, qcten)
    ncten = torch.where(melt, ncten + ni1d * odt, ncten)
    qiten = torch.where(melt, qiten - xri * odt, qiten)
    niten = torch.where(melt, -ni1d * odt, niten)
    tten = torch.where(melt, tten - float(_const_segment_n_LFUS) * ocp * xri * odt, tten)

    xrc = torch.clamp_min(qc1d + qcten * dt, 0.0)
    freeze = (temp < float(_const_homogeneous_freezing_temperature)) & (xrc > float(_const_segment_e_R1))
    xnc = cloud_number_base + ncten * dt
    lfus2 = float(_const_segment_j_LSUB) - lvap
    qiten = torch.where(freeze, qiten + xrc * odt, qiten)
    niten = torch.where(freeze, niten + xnc * odt, niten)
    qcten = torch.where(freeze, qcten - xrc * odt, qcten)
    ncten = torch.where(freeze, ncten - xnc * odt, ncten)
    tten = torch.where(freeze, tten + lfus2 * ocp * xrc * odt, tten)
    return qiten, niten, qcten, ncten, tten


def freezing_rain_tensor(
    *,
    temp: torch.Tensor,
    rain_precip: torch.Tensor,
    cloud_precip: torch.Tensor | None = None,
) -> torch.Tensor:
    surface_temp = temp[:, 0] if temp.ndim == 2 else temp
    mask = surface_temp - _scalar(constants.T0, surface_temp) < _scalar(-0.5, surface_temp)
    values = rain_precip
    if cloud_precip is not None:
        values = values + cloud_precip
    return torch.where(mask, values, 0.0)


def reflectivity_10cm_tensor(
    *,
    refl10cm_from_melting_flag: bool,
    temp: torch.Tensor,
    l_qr: torch.Tensor,
    rr: torch.Tensor,
    nr: torch.Tensor,
    ilamr: torch.Tensor,
    l_qs: torch.Tensor,
    rs: torch.Tensor,
    smoc: torch.Tensor,
    smob: torch.Tensor,
    smoz: torch.Tensor,
    l_qg: torch.Tensor,
    rg: torch.Tensor,
    ng: torch.Tensor,
    idx: torch.Tensor,
    ilamg: torch.Tensor,
) -> torch.Tensor:
    del temp, rs, smoc, smob
    if refl10cm_from_melting_flag:
        raise NotImplementedError("melting reflectivity is implemented by run_r2")
    ze_rain, ze_snow, ze_graupel = reflectivity_10cm_ordinary_components_tensor(
        l_qr=l_qr,
        rr=rr,
        nr=nr,
        ilamr=ilamr,
        l_qs=l_qs,
        smoz=smoz,
        l_qg=l_qg,
        rg=rg,
        ng=ng,
        idx=idx,
        ilamg=ilamg,
    )
    dbz = _scalar(10.0, rr) * torch.log10((ze_rain + ze_snow + ze_graupel) * _scalar(1.0e18, rr))
    return torch.clamp_min(dbz, _scalar(-35.0, rr))


def reflectivity_10cm_components_tensor(
    *,
    refl10cm_from_melting_flag: bool | torch.Tensor,
    temp: torch.Tensor,
    l_qr: torch.Tensor,
    rr: torch.Tensor,
    nr: torch.Tensor,
    ilamr: torch.Tensor,
    l_qs: torch.Tensor,
    rs: torch.Tensor,
    smoc: torch.Tensor,
    smob: torch.Tensor,
    smoz: torch.Tensor,
    l_qg: torch.Tensor,
    rg: torch.Tensor,
    ng: torch.Tensor,
    idx: torch.Tensor,
    ilamg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ze_rain, ze_snow, ze_graupel = reflectivity_10cm_ordinary_components_tensor(
        l_qr=l_qr,
        rr=rr,
        nr=nr,
        ilamr=ilamr,
        l_qs=l_qs,
        smoz=smoz,
        l_qg=l_qg,
        rg=rg,
        ng=ng,
        idx=idx,
        ilamg=ilamg,
    )
    if isinstance(refl10cm_from_melting_flag, torch.Tensor):
        selector = refl10cm_from_melting_flag.to(device=rr.device)
        if selector.dtype == torch.bool:
            if selector.ndim != 1 or selector.shape[0] != rr.shape[0]:
                raise ValueError("refl10cm_from_melting tensor mask must have shape [ncol]")
            selected_melting_level, selected_ze_snow, selected_ze_graupel = _reflectivity_melting_components_tensor(
                temp=temp,
                l_qr=l_qr,
                l_qs=l_qs,
                l_qg=l_qg,
                rr=rr,
                rs=rs,
                smob=smob,
                smoc=smoc,
                rg=rg,
                ng=ng,
                ilamg=ilamg,
                idx=idx,
                ze_snow=ze_snow,
                ze_graupel=ze_graupel,
            )
            selector_2d = selector[:, None]
            melting_level = torch.where(selector, selected_melting_level, 1)
            ze_snow = torch.where(selector_2d, selected_ze_snow, ze_snow)
            ze_graupel = torch.where(selector_2d, selected_ze_graupel, ze_graupel)
        else:
            column_indices = selector.to(dtype=torch.int64)
            melting_level, ze_snow, ze_graupel = _reflectivity_melting_components_selected_tensor(
                column_indices=column_indices,
                temp=temp,
                l_qr=l_qr,
                l_qs=l_qs,
                l_qg=l_qg,
                rr=rr,
                rs=rs,
                smob=smob,
                smoc=smoc,
                rg=rg,
                ng=ng,
                ilamg=ilamg,
                idx=idx,
                ze_snow=ze_snow,
                ze_graupel=ze_graupel,
            )
    elif refl10cm_from_melting_flag:
        melting_level, ze_snow, ze_graupel = _reflectivity_melting_components_tensor(
            temp=temp,
            l_qr=l_qr,
            l_qs=l_qs,
            l_qg=l_qg,
            rr=rr,
            rs=rs,
            smob=smob,
            smoc=smoc,
            rg=rg,
            ng=ng,
            ilamg=ilamg,
            idx=idx,
            ze_snow=ze_snow,
            ze_graupel=ze_graupel,
        )
    else:
        melting_level = torch.ones((rr.shape[0],), dtype=torch.int64, device=rr.device)
    dbz = _scalar(10.0, rr) * torch.log10((ze_rain + ze_snow + ze_graupel) * _scalar(1.0e18, rr))
    return torch.clamp_min(dbz, _scalar(-35.0, rr)), melting_level, ze_snow, ze_graupel


def _reflectivity_melting_components_tensor(
    *,
    temp: torch.Tensor,
    l_qr: torch.Tensor,
    l_qs: torch.Tensor,
    l_qg: torch.Tensor,
    rr: torch.Tensor,
    rs: torch.Tensor,
    smob: torch.Tensor,
    smoc: torch.Tensor,
    rg: torch.Tensor,
    ng: torch.Tensor,
    ilamg: torch.Tensor,
    idx: torch.Tensor,
    ze_snow: torch.Tensor,
    ze_graupel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    melting_level = find_melting_level_tensor(temp=temp, l_qr=l_qr, l_qs=l_qs, l_qg=l_qg)
    levels = torch.arange(1, rr.shape[1] + 1, dtype=torch.int64, device=rr.device)[None, :]
    melting_mask = (melting_level[:, None] > 2) & (levels < melting_level[:, None] - 1)
    snow_melting = l_qs & melting_mask
    graupel_melting = l_qg & melting_mask
    ze_snow = torch.where(
        snow_melting,
        reflectivity_from_melting_snow_tensor(
            rs=rs,
            smob=smob,
            smoc=smoc,
            rr=rr,
            active_mask=snow_melting,
        ),
        ze_snow,
    )
    ze_graupel = torch.where(
        graupel_melting,
        reflectivity_from_melting_graupel_tensor(
            rg=rg,
            ng=ng,
            ilamg=ilamg,
            idx=idx,
            rr=rr,
            active_mask=graupel_melting,
        ),
        ze_graupel,
    )
    return melting_level, ze_snow, ze_graupel


def _reflectivity_melting_components_selected_tensor(
    *,
    column_indices: torch.Tensor,
    temp: torch.Tensor,
    l_qr: torch.Tensor,
    l_qs: torch.Tensor,
    l_qg: torch.Tensor,
    rr: torch.Tensor,
    rs: torch.Tensor,
    smob: torch.Tensor,
    smoc: torch.Tensor,
    rg: torch.Tensor,
    ng: torch.Tensor,
    ilamg: torch.Tensor,
    idx: torch.Tensor,
    ze_snow: torch.Tensor,
    ze_graupel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if column_indices.ndim != 1:
        raise ValueError("refl10cm_from_melting index tensor must have shape [nselected]")
    melting_level = torch.ones((rr.shape[0],), dtype=torch.int64, device=rr.device)
    if column_indices.numel() == 0:
        return melting_level, ze_snow, ze_graupel
    selected_melting_level, selected_ze_snow, selected_ze_graupel = _reflectivity_melting_components_tensor(
        temp=temp.index_select(0, column_indices),
        l_qr=l_qr.index_select(0, column_indices),
        l_qs=l_qs.index_select(0, column_indices),
        l_qg=l_qg.index_select(0, column_indices),
        rr=rr.index_select(0, column_indices),
        rs=rs.index_select(0, column_indices),
        smob=smob.index_select(0, column_indices),
        smoc=smoc.index_select(0, column_indices),
        rg=rg.index_select(0, column_indices),
        ng=ng.index_select(0, column_indices),
        ilamg=ilamg.index_select(0, column_indices),
        idx=idx.index_select(0, column_indices),
        ze_snow=ze_snow.index_select(0, column_indices),
        ze_graupel=ze_graupel.index_select(0, column_indices),
    )
    melting_level = melting_level.index_copy(0, column_indices, selected_melting_level)
    ze_snow = ze_snow.index_copy(0, column_indices, selected_ze_snow)
    ze_graupel = ze_graupel.index_copy(0, column_indices, selected_ze_graupel)
    return melting_level, ze_snow, ze_graupel


def reflectivity_10cm_ordinary_components_tensor(
    *,
    l_qr: torch.Tensor,
    rr: torch.Tensor,
    nr: torch.Tensor,
    ilamr: torch.Tensor,
    l_qs: torch.Tensor,
    smoz: torch.Tensor,
    l_qg: torch.Tensor,
    rg: torch.Tensor,
    ng: torch.Tensor,
    idx: torch.Tensor,
    ilamg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if TENSOR_COMPILE_ISLAND_CUDA and rr.device.type in TENSOR_COMPILE_ISLAND_DEVICES:
        return _run_tensor_compile_island(
            "reflectivity_ordinary",
            _reflectivity_10cm_ordinary_components_kernel,
            rr,
            l_qr,
            rr,
            nr,
            ilamr,
            l_qs,
            smoz,
            l_qg,
            rg,
            ng,
            idx,
            ilamg,
            _tensor(_STATIC_GRAUPEL_AM_G, rr),
        )
    lamr = 1.0 / torch.where(ilamr > 0.0, ilamr, _scalar(1.0, ilamr))
    n0_r = nr * _scalar(constants.ORG2, rr) * lamr
    ze_rain_value = n0_r * _scalar(_SCALAR_CRG[3], rr) * ilamr ** _scalar(_SCALAR_CRE[3], rr)
    ze_rain = torch.where(l_qr, ze_rain_value, _scalar(1.0e-22, rr))

    ze_snow_value = (
        _scalar(0.176 / 0.93, rr)
        * _scalar(6.0 / constants.PI, rr)
        * _scalar(6.0 / constants.PI, rr)
        * _scalar(constants.AM_S / 900.0, rr)
        * _scalar(constants.AM_S / 900.0, rr)
        * smoz
    )
    ze_snow = torch.where(l_qs, ze_snow_value, _scalar(1.0e-22, rr))

    idx_safe = torch.clamp(idx.to(torch.int64), 1, constants.NRHG)
    am_g = _take(_STATIC_GRAUPEL_AM_G, idx_safe, rr)
    lamg = 1.0 / torch.where(ilamg > 0.0, ilamg, _scalar(1.0, ilamg))
    n0_g = ng * _scalar(constants.OGG2, rr) * lamg
    ze_graupel_value = (
        _scalar(0.176 / 0.93, rr)
        * _scalar(6.0 / constants.PI, rr)
        * _scalar(6.0 / constants.PI, rr)
        * (am_g / _scalar(900.0, rr))
        * (am_g / _scalar(900.0, rr))
        * n0_g
        * _scalar(_SCALAR_GRAUPEL_CGG[3][0], rr)
        * ilamg ** _scalar(_SCALAR_GRAUPEL_CGE[3][0], rr)
    )
    ze_graupel = torch.where(l_qg, ze_graupel_value, _scalar(1.0e-22, rr))
    return ze_rain, ze_snow, ze_graupel


def _reflectivity_10cm_ordinary_components_kernel(
    l_qr: torch.Tensor,
    rr: torch.Tensor,
    nr: torch.Tensor,
    ilamr: torch.Tensor,
    l_qs: torch.Tensor,
    smoz: torch.Tensor,
    l_qg: torch.Tensor,
    rg: torch.Tensor,
    ng: torch.Tensor,
    idx: torch.Tensor,
    ilamg: torch.Tensor,
    graupel_am: torch.Tensor,
    _org2: float = float(constants.ORG2),
    _crg3: float = float(_SCALAR_CRG[3]),
    _cre3: float = float(_SCALAR_CRE[3]),
    _snow_dielectric: float = float(0.176 / 0.93),
    _six_over_pi: float = float(6.0 / constants.PI),
    _snow_mass_scale: float = float(constants.AM_S / 900.0),
    _nrhg: int = int(constants.NRHG),
    _ogg2: float = float(constants.OGG2),
    _cgg3: float = float(_SCALAR_GRAUPEL_CGG[3][0]),
    _cge3: float = float(_SCALAR_GRAUPEL_CGE[3][0]),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lamr = 1.0 / torch.where(ilamr > 0.0, ilamr, 1.0)
    n0_r = nr * _org2 * lamr
    ze_rain_value = n0_r * _crg3 * torch.pow(ilamr, _cre3)
    ze_rain = torch.where(l_qr, ze_rain_value, 1.0e-22)

    ze_snow_value = _snow_dielectric * _six_over_pi * _six_over_pi * _snow_mass_scale * _snow_mass_scale * smoz
    ze_snow = torch.where(l_qs, ze_snow_value, 1.0e-22)

    idx_safe = torch.clamp(idx.to(torch.int64), 1, _nrhg)
    am_g = graupel_am[idx_safe - 1]
    lamg = 1.0 / torch.where(ilamg > 0.0, ilamg, 1.0)
    n0_g = ng * _ogg2 * lamg
    am_scaled = am_g / 900.0
    ze_graupel_value = _snow_dielectric * _six_over_pi * _six_over_pi * am_scaled * am_scaled * n0_g * _cgg3 * torch.pow(ilamg, _cge3)
    ze_graupel = torch.where(l_qg, ze_graupel_value, 1.0e-22)
    return ze_rain, ze_snow, ze_graupel


def find_melting_level_tensor(
    *,
    temp: torch.Tensor,
    l_qr: torch.Tensor,
    l_qs: torch.Tensor,
    l_qg: torch.Tensor,
) -> torch.Tensor:
    ncol, nz = temp.shape
    if nz < 2:
        return torch.ones((ncol,), dtype=torch.int64, device=temp.device)
    cond = (
        (temp[:, : nz - 1] > _scalar(constants.T0, temp))
        & l_qr[:, : nz - 1]
        & (l_qs[:, 1:] | l_qg[:, 1:])
    )
    levels = torch.arange(2, nz + 1, dtype=torch.int64, device=temp.device)[None, :]
    return torch.max(torch.where(cond, levels.expand(ncol, nz - 1), _int_scalar(1, temp)), dim=1).values


def _run_radar_reduction(
    name: str,
    function: Any,
    reference: torch.Tensor,
    *args: Any,
) -> torch.Tensor:
    status_name = f"radar_reduction:{name}"
    if (
        not RADAR_FUSED_REDUCTION_COMPILE_CUDA
        or reference.device.type not in RADAR_FUSED_REDUCTION_COMPILE_DEVICES
        or torch.compiler.is_compiling()
    ):
        return function(*args)
    prepared_dispatch = _ACTIVE_PREPARED_COMPILE_DISPATCH.get()
    prepared_key = _prepared_compile_key(
        status_name,
        function,
        _prepared_runtime_signature(reference, args),
    )
    if prepared_dispatch is not None:
        prepared_callable = prepared_dispatch.callables.get(prepared_key)
        if prepared_callable is not None:
            prepared_dispatch.fast_hits += 1
            try:
                return prepared_callable(*args)
            except Exception as exc:
                reason = _compile_exception_reason(exc)
                raise RuntimeError(
                    f"prepared compile dispatch {status_name} runtime_failed: {reason}"
                ) from exc
        prepared_dispatch.misses += 1
    cache_key = _compile_cache_key(name, function, reference, args=args)
    if cache_key in _RADAR_COMPILED_REDUCTION_DISABLED:
        if prepared_dispatch is not None:
            raise RuntimeError(f"prepared compile dispatch {status_name} is disabled")
        return function(*args)
    compiled = _RADAR_COMPILED_REDUCTION_CACHE.get(cache_key)
    if compiled is None:
        compiled, result, failure_stage, failure, compile_timing = _compile_and_prime_with_scoped_scalar_capture(
            function,
            args,
            {"fullgraph": True, "dynamic": False, "mode": None},
        )
        _record_compile_island_timing(status_name, cache_key, compile_timing)
        if failure is not None:
            reason = _compile_exception_reason(failure)
            _RADAR_COMPILED_REDUCTION_DISABLED.add(cache_key)
            _record_tensor_compile_island_status(
                status_name,
                failure_stage or "compile_failed",
                reason,
                cache_key=cache_key,
            )
            _raise_compile_island_failure_if_strict(
                status_name,
                failure_stage or "compile_failed",
                reason,
                failure,
            )
            if prepared_dispatch is not None:
                raise RuntimeError(
                    f"prepared compile dispatch {status_name} {failure_stage or 'compile_failed'}: {reason}"
                ) from failure
            return function(*args)
        assert compiled is not None
        _RADAR_COMPILED_REDUCTION_CACHE[cache_key] = compiled
        _record_tensor_compile_island_status(status_name, "compiled", cache_key=cache_key)
        if prepared_dispatch is not None:
            prepared_dispatch.callables[prepared_key] = compiled
        return result
    try:
        result = compiled(*args)
    except Exception as exc:
        reason = _compile_exception_reason(exc)
        _RADAR_COMPILED_REDUCTION_CACHE.pop(cache_key, None)
        _RADAR_COMPILED_REDUCTION_DISABLED.add(cache_key)
        _record_tensor_compile_island_status(
            status_name,
            "runtime_failed",
            reason,
            cache_key=cache_key,
        )
        _raise_compile_island_failure_if_strict(status_name, "runtime_failed", reason, exc)
        if prepared_dispatch is not None:
            raise RuntimeError(
                f"prepared compile dispatch {status_name} runtime_failed: {reason}"
            ) from exc
        return function(*args)
    if prepared_dispatch is not None:
        prepared_dispatch.callables[prepared_key] = compiled
    return result


def _reflectivity_melting_graupel_reduction_tensor(rg: torch.Tensor, ng: torch.Tensor, ilamg: torch.Tensor, idx: torch.Tensor, rr: torch.Tensor, graupel_am_g_table: torch.Tensor, bin_values: torch.Tensor, quadrature_weights: torch.Tensor, _const_segment_e_R1=constants.R1, _const_segment_g_OGG2=constants.OGG2, _const_segment_g_NRHG=constants.NRHG, _const_segment_g_OBMG=constants.OBMG, _const_segment_r2_RADAR_BINS=constants.RADAR_BINS, _const_segment_r2_LAMBDA_RADAR=constants.LAMBDA_RADAR, _const_segment_r2_MELT_OUTSIDE=constants.MELT_OUTSIDE, _const_segment_g_MU_G=constants.MU_G) -> torch.Tensor:
    sr = torch.clamp(1.0 - rg / torch.clamp_min(rg + rr, float(_const_segment_e_R1)), min=0.01, max=0.99)
    fmelt = sr * sr
    lamg = 1.0 / torch.where(ilamg > 0.0, ilamg, 1.0)
    n0_g = ng * float(_const_segment_g_OGG2) * lamg
    idx_safe = torch.clamp(idx.to(torch.int64), 1, _const_segment_g_NRHG)
    am_g = graupel_am_g_table[idx_safe - 1]
    ocmg = torch.pow(1.0 / am_g, float(_const_segment_g_OBMG))
    eta: torch.Tensor | None = None
    for bin_index in range(_const_segment_r2_RADAR_BINS):
        bin_value = bin_values[bin_index]
        mass = am_g * bin_value * bin_value * bin_value
        backscatter = _rayleigh_soak_wetgraupel_no_cache(x_g=mass, a_geo=ocmg, b_geo=_const_segment_g_OBMG, fmelt=fmelt, lambda_radar=_const_segment_r2_LAMBDA_RADAR, meltratio_outside=_const_segment_r2_MELT_OUTSIDE, m_w=_RADAR_COMPLEX_WATER_0, m_i=_RADAR_COMPLEX_ICE_0)
        f_d = n0_g * bin_value ** _const_segment_g_MU_G * torch.exp(-lamg * bin_value)
        term = f_d * backscatter * quadrature_weights[bin_index]
        eta = term if eta is None else eta + term
    if eta is None:
        eta = rg * 0.0
    return float(_RADAR_REFLECTIVITY_FACTOR) * eta


def _reflectivity_melting_snow_reduction_tensor(rs: torch.Tensor, rr: torch.Tensor, fmelt: torch.Tensor, m0: torch.Tensor, mrat: torch.Tensor, slam1: torch.Tensor, slam2: torch.Tensor, bin_values: torch.Tensor, quadrature_weights: torch.Tensor, _const_segment_r2_RADAR_BINS=constants.RADAR_BINS, _const_segment_e_AM_S=constants.AM_S, _const_segment_r2__OCMS=constants._OCMS, _const_segment_r2__OBMS=constants._OBMS, _const_segment_r2_LAMBDA_RADAR=constants.LAMBDA_RADAR, _const_segment_r2_MELT_OUTSIDE=constants.MELT_OUTSIDE, _const_segment_e_KAP0=constants.KAP0, _const_segment_e_KAP1=constants.KAP1, _const_segment_e_MU_S=constants.MU_S) -> torch.Tensor:
    eta: torch.Tensor | None = None
    for bin_index in range(_const_segment_r2_RADAR_BINS):
        bin_value = bin_values[bin_index]
        mass = float(_const_segment_e_AM_S) * bin_value * bin_value
        backscatter = _rayleigh_soak_wetgraupel_no_cache(x_g=mass, a_geo=_const_segment_r2__OCMS, b_geo=_const_segment_r2__OBMS, fmelt=fmelt, lambda_radar=_const_segment_r2_LAMBDA_RADAR, meltratio_outside=_const_segment_r2_MELT_OUTSIDE, m_w=_RADAR_COMPLEX_WATER_0, m_i=_RADAR_COMPLEX_ICE_0)
        f_d = mrat * (_const_segment_e_KAP0 * torch.exp(-slam1 * bin_value) + _const_segment_e_KAP1 * (m0 * bin_value) ** _const_segment_e_MU_S * torch.exp(-slam2 * bin_value))
        term = f_d * backscatter * quadrature_weights[bin_index]
        eta = term if eta is None else eta + term
    if eta is None:
        eta = rs * 0.0
    return float(_RADAR_REFLECTIVITY_FACTOR) * eta


def _melting_snow_bin_chunk_tensor(
    eta: torch.Tensor | None,
    fmelt: torch.Tensor,
    m0: torch.Tensor,
    mrat: torch.Tensor,
    slam1: torch.Tensor,
    slam2: torch.Tensor,
    bin_values: tuple[float, ...],
    quadrature_weights: tuple[float, ...],
    _am_s: float = float(constants.AM_S),
    _ocms: float = float(constants._OCMS),
    _obms: float = float(constants._OBMS),
    _lambda_radar: float = float(constants.LAMBDA_RADAR),
    _melt_outside: float = float(constants.MELT_OUTSIDE),
    _kap0: float = float(constants.KAP0),
    _kap1: float = float(constants.KAP1),
    _mu_s: float = float(constants.MU_S),
) -> torch.Tensor:
    """Continue the original ordered quadrature, with at most four live bins."""
    for bin_value, weight in zip(bin_values, quadrature_weights, strict=True):
        mass = _am_s * bin_value * bin_value
        backscatter = _rayleigh_soak_wetgraupel_no_cache(
            x_g=mass, a_geo=_ocms, b_geo=_obms,
            fmelt=fmelt, lambda_radar=_lambda_radar,
            meltratio_outside=_melt_outside,
            m_w=_RADAR_COMPLEX_WATER_0, m_i=_RADAR_COMPLEX_ICE_0,
        )
        f_d = mrat * (
            _kap0 * torch.exp(-slam1 * bin_value)
            + _kap1 * (m0 * bin_value) ** _mu_s
            * torch.exp(-slam2 * bin_value)
        )
        term = f_d * backscatter * weight
        eta = term if eta is None else eta + term
    assert eta is not None
    return eta


def _melting_graupel_bin_chunk_tensor(
    eta: torch.Tensor | None,
    fmelt: torch.Tensor,
    lamg: torch.Tensor,
    n0_g: torch.Tensor,
    am_g: torch.Tensor,
    ocmg: torch.Tensor,
    bin_values: tuple[float, ...],
    quadrature_weights: tuple[float, ...],
    _obmg: float = float(constants.OBMG),
    _lambda_radar: float = float(constants.LAMBDA_RADAR),
    _melt_outside: float = float(constants.MELT_OUTSIDE),
    _mu_g: float = float(constants.MU_G),
) -> torch.Tensor:
    for bin_value, weight in zip(bin_values, quadrature_weights, strict=True):
        mass = am_g * bin_value * bin_value * bin_value
        backscatter = _rayleigh_soak_wetgraupel_no_cache(
            x_g=mass, a_geo=ocmg, b_geo=_obmg,
            fmelt=fmelt, lambda_radar=_lambda_radar,
            meltratio_outside=_melt_outside,
            m_w=_RADAR_COMPLEX_WATER_0, m_i=_RADAR_COMPLEX_ICE_0,
        )
        f_d = n0_g * bin_value ** _mu_g * torch.exp(-lamg * bin_value)
        term = f_d * backscatter * weight
        eta = term if eta is None else eta + term
    assert eta is not None
    return eta


def _run_melting_bin_chunks(
    name: str,
    function: Any,
    reference: torch.Tensor,
    spatial_args: tuple[torch.Tensor, ...],
    bin_values: tuple[float, ...],
    quadrature_weights: tuple[float, ...],
) -> torch.Tensor:
    """Keep dispatch outside science kernels and carry the ordered accumulator."""
    eta = None
    for start in range(0, len(bin_values), 4):
        args = (
            eta, *spatial_args,
            bin_values[start:start + 4], quadrature_weights[start:start + 4],
        )
        if DIAGNOSTIC_COMPUTE_MODE == "compact_active" and reference.ndim == 1:
            eta = _run_sparse_diagnostic_compile_island(
                name, function, reference, *args,
                enabled=RADAR_FUSED_REDUCTION_COMPILE_CUDA,
                devices=RADAR_FUSED_REDUCTION_COMPILE_DEVICES,
            )
        else:
            eta = _run_radar_reduction(name, function, reference, *args)
    assert eta is not None
    return float(_RADAR_REFLECTIVITY_FACTOR) * eta


def _reflectivity_melting_snow_ordered_chunks_tensor(
    rs: torch.Tensor, rr: torch.Tensor, fmelt: torch.Tensor,
    m0: torch.Tensor, mrat: torch.Tensor, slam1: torch.Tensor,
    slam2: torch.Tensor, bin_values: tuple[float, ...],
    quadrature_weights: tuple[float, ...],
) -> torch.Tensor:
    return _run_melting_bin_chunks(
        "melting_snow_bin_chunk", _melting_snow_bin_chunk_tensor, rs,
        (fmelt, m0, mrat, slam1, slam2), bin_values, quadrature_weights,
    )


def _reflectivity_melting_graupel_ordered_chunks_tensor(
    rg: torch.Tensor, ng: torch.Tensor, ilamg: torch.Tensor, idx: torch.Tensor,
    rr: torch.Tensor, graupel_am_g_table: torch.Tensor,
    bin_values: tuple[float, ...], quadrature_weights: tuple[float, ...],
) -> torch.Tensor:
    sr = torch.clamp(
        1.0 - rg / torch.clamp_min(rg + rr, float(constants.R1)),
        min=0.01, max=0.99,
    )
    fmelt = sr * sr
    lamg = 1.0 / torch.where(ilamg > 0.0, ilamg, 1.0)
    n0_g = ng * float(constants.OGG2) * lamg
    idx_safe = torch.clamp(idx.to(torch.int64), 1, constants.NRHG)
    am_g = graupel_am_g_table[idx_safe - 1]
    ocmg = torch.pow(1.0 / am_g, float(constants.OBMG))
    return _run_melting_bin_chunks(
        "melting_graupel_bin_chunk", _melting_graupel_bin_chunk_tensor, rg,
        (fmelt, lamg, n0_g, am_g, ocmg), bin_values, quadrature_weights,
    )


def _compact_diagnostic_reduction(
    name: str,
    function: Any,
    active_mask: torch.Tensor,
    spatial_args: tuple[torch.Tensor, ...],
    static_args: tuple[Any, ...],
    *,
    compile_reduction: bool = True,
) -> torch.Tensor:
    """Gather active cells, run science on device, and scatter without a host count."""
    active_indices = torch.nonzero(active_mask.reshape(-1), as_tuple=False)[:, 0]
    selected_args = tuple(
        value.reshape(-1).index_select(0, active_indices) for value in spatial_args
    )
    reference = selected_args[0]
    selected = _run_sparse_diagnostic_compile_island(
        name,
        function,
        reference,
        *selected_args,
        *static_args,
        enabled=RADAR_FUSED_REDUCTION_COMPILE_CUDA,
        devices=RADAR_FUSED_REDUCTION_COMPILE_DEVICES,
    ) if compile_reduction else function(*selected_args, *static_args)
    return (
        torch.zeros_like(spatial_args[0].reshape(-1))
        .index_copy(0, active_indices, selected)
        .reshape(spatial_args[0].shape)
    )


def reflectivity_from_melting_graupel_tensor(
    *,
    rg: torch.Tensor,
    ng: torch.Tensor,
    ilamg: torch.Tensor,
    idx: torch.Tensor,
    rr: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    compiled_bins = RADAR_FUSED_REDUCTION_COMPILE_CUDA and rg.device.type in RADAR_FUSED_REDUCTION_COMPILE_DEVICES
    static_args = (
        _tensor(_STATIC_GRAUPEL_AM_G, rg),
        _SCALAR_RADAR_GRAUPEL_BINS if compiled_bins else _tensor(_SCALAR_RADAR_GRAUPEL_BINS, rg),
        _SCALAR_RADAR_GRAUPEL_QUADRATURE_WEIGHTS if compiled_bins else _tensor(_SCALAR_RADAR_GRAUPEL_QUADRATURE_WEIGHTS, rg),
    )
    if DIAGNOSTIC_COMPUTE_MODE == "compact_active" and active_mask is not None:
        return _compact_diagnostic_reduction(
            "melting_graupel_compact_active",
            _reflectivity_melting_graupel_ordered_chunks_tensor if compiled_bins else _reflectivity_melting_graupel_reduction_tensor,
            active_mask,
            (rg, ng, ilamg, idx, rr),
            static_args,
            compile_reduction=not compiled_bins,
        )
    if compiled_bins:
        return _reflectivity_melting_graupel_ordered_chunks_tensor(
            rg, ng, ilamg, idx, rr, *static_args,
        )
    return _run_radar_reduction(
        "melting_graupel",
        _reflectivity_melting_graupel_reduction_tensor,
        rg,
        rg,
        ng,
        ilamg,
        idx,
        rr,
        *static_args,
    )


def reflectivity_from_melting_snow_tensor(
    *,
    rs: torch.Tensor,
    smob: torch.Tensor,
    smoc: torch.Tensor,
    rr: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    sr = torch.clamp(
        1.0 - rs / _safe_denominator_no_cache(rs + rr),
        min=0.01,
        max=0.99,
    )
    fmelt = sr * sr
    m0 = smob / _clamp_tiny(smoc)
    mrat = smob * m0 * m0 * m0
    slam1 = m0 * float(constants.LAM0)
    slam2 = m0 * float(constants.LAM1)
    compiled_bins = RADAR_FUSED_REDUCTION_COMPILE_CUDA and rs.device.type in RADAR_FUSED_REDUCTION_COMPILE_DEVICES
    static_args = (
        _SCALAR_RADAR_SNOW_BINS if compiled_bins else _tensor(_SCALAR_RADAR_SNOW_BINS, rs),
        _SCALAR_RADAR_SNOW_QUADRATURE_WEIGHTS if compiled_bins else _tensor(_SCALAR_RADAR_SNOW_QUADRATURE_WEIGHTS, rs),
    )
    if DIAGNOSTIC_COMPUTE_MODE == "compact_active" and active_mask is not None:
        return _compact_diagnostic_reduction(
            "melting_snow_compact_active",
            _reflectivity_melting_snow_ordered_chunks_tensor if compiled_bins else _reflectivity_melting_snow_reduction_tensor,
            active_mask,
            (rs, rr, fmelt, m0, mrat, slam1, slam2),
            static_args,
            compile_reduction=not compiled_bins,
        )
    if compiled_bins:
        return _reflectivity_melting_snow_ordered_chunks_tensor(
            rs, rr, fmelt, m0, mrat, slam1, slam2, *static_args,
        )
    return _run_radar_reduction(
        "melting_snow",
        _reflectivity_melting_snow_reduction_tensor,
        rs,
        rs,
        rr,
        fmelt,
        m0,
        mrat,
        slam1,
        slam2,
        *static_args,
    )


def rayleigh_soak_wetgraupel_tensor(
    *,
    x_g: torch.Tensor,
    a_geo: torch.Tensor,
    b_geo: float,
    fmelt: torch.Tensor,
    lambda_radar: float,
    meltratio_outside: float,
    m_w: complex,
    m_i: complex,
) -> torch.Tensor:
    fm = torch.clamp(fmelt, min=0.0, max=1.0)
    mra = _scalar(min(max(float(meltratio_outside), 0.0), 1.0), fm)
    mra = mra + (1.0 - mra) * fm
    x_w = x_g * fm
    d_g = a_geo * x_g ** _scalar(b_geo, fm)
    vg_sphere = _scalar(constants.PI / 6.0, fm) * d_g * d_g * d_g
    rhog = torch.clamp(x_g / _clamp_tiny(vg_sphere), min=10.0, max=900.0)
    vg = x_g / rhog
    outside_limit = 1.0 - rhog / _scalar(1000.0, fm)
    fmgrenz = (_scalar(900.0, fm) - rhog) / _safe_denominator(mra * _scalar(900.0, fm) - rhog + _scalar(0.9, fm) * rhog)
    volg_core = torch.where(
        fm <= fmgrenz,
        (1.0 - mra * fm) * vg,
        (x_g - x_w) / _scalar(900.0, fm) + x_w / _scalar(1000.0, fm),
    )
    volg = torch.where(mra <= outside_limit, vg * (1.0 - mra * fm), volg_core)
    d_large = (_scalar(6.0 / constants.PI, fm) * _clamp_tiny(volg)) ** _scalar(1.0 / 3.0, fm)
    volice = (x_g - x_w) / _clamp_tiny(volg * _scalar(900.0, fm))
    volwater = x_w / _clamp_tiny(_scalar(1000.0, fm) * volg)
    volair = 1.0 - volice - volwater
    volmix1 = volice / torch.clamp_min(volice + volwater, _scalar(1.0e-10, fm))

    m1t = m_w * m_w
    m3t = m_i * m_i
    beta3 = (
        _RADAR_BETA_WATER_ICE
        if m1t == _RADAR_COMPLEX_WATER_SQ and m3t == _RADAR_COMPLEX_ICE_SQ
        else _complex_beta_constant_python(m1t, m3t)
    )
    vol3 = volmix1
    m_tmp_sq = _complex_div_pair(
        ((1.0 - vol3) * m1t.real + vol3 * beta3.real * m3t.real - vol3 * beta3.imag * m3t.imag,
         (1.0 - vol3) * m1t.imag + vol3 * beta3.real * m3t.imag + vol3 * beta3.imag * m3t.real),
        (1.0 - vol3 + vol3 * beta3.real, vol3 * beta3.imag),
    )
    m_tmp = _complex_sqrt_pair(m_tmp_sq)

    m1t = _complex_mul_pair(m_tmp, m_tmp)
    m2t = (1.0, 0.0)
    m3t = (4.0, 0.0)
    vol2 = volair
    beta2 = _complex_beta_pair(m1t, m2t)
    beta2_m2 = beta2
    numerator = (
        (1.0 - vol2) * m1t[0] + vol2 * beta2_m2[0],
        (1.0 - vol2) * m1t[1] + vol2 * beta2_m2[1],
    )
    denominator = (1.0 - vol2 + vol2 * beta2[0], vol2 * beta2[1])
    m_core = _complex_sqrt_pair(_complex_div_pair(numerator, denominator))
    m_core_sq = _complex_mul_pair(m_core, m_core)
    ratio = _complex_div_pair((m_core_sq[0] - 1.0, m_core_sq[1]), (m_core_sq[0] + 2.0, m_core_sq[1]))
    backscatter = _complex_abs_square_pair(ratio)
    backscatter = backscatter * _scalar(constants.PI**5 / lambda_radar**4, fm) * _pow6(d_large)
    return torch.where(d_g < _scalar(constants.R1, fm), 0.0, backscatter)


def _rayleigh_soak_wetgraupel_no_cache(*, x_g: torch.Tensor | float, a_geo: torch.Tensor | float, b_geo: float, fmelt: torch.Tensor, lambda_radar: float, meltratio_outside: float, m_w: complex, m_i: complex, _const_segment_e_PI=constants.PI, _const_segment_e_R1=constants.R1) -> torch.Tensor:
    fm = torch.clamp(fmelt, min=0.0, max=1.0)
    mra = float(min(max(float(meltratio_outside), 0.0), 1.0))
    mra = mra + (1.0 - mra) * fm
    x_w = x_g * fm
    d_g = a_geo * x_g ** float(b_geo)
    vg_sphere = float(_const_segment_e_PI / 6.0) * d_g * d_g * d_g
    if isinstance(x_g, torch.Tensor):
        rhog = torch.clamp(x_g / _clamp_tiny(vg_sphere), min=10.0, max=900.0)
    else:
        # Literal snow-bin geometry is immutable. Keep the same scalar order
        # while allowing the compiler to fold it before tracing live physics.
        rhog = min(max(x_g / max(vg_sphere, 1e-300), 10.0), 900.0)
    vg = x_g / rhog
    outside_limit = 1.0 - rhog / 1000.0
    fmgrenz = (900.0 - rhog) / _safe_denominator_no_cache(mra * 900.0 - rhog + 0.9 * rhog)
    volg_core = torch.where(fm <= fmgrenz, (1.0 - mra * fm) * vg, (x_g - x_w) / 900.0 + x_w / 1000.0)
    volg = torch.where(mra <= outside_limit, vg * (1.0 - mra * fm), volg_core)
    d_large = (float(6.0 / _const_segment_e_PI) * _clamp_tiny(volg)) ** (1.0 / 3.0)
    volice = (x_g - x_w) / _clamp_tiny(volg * 900.0)
    volwater = x_w / _clamp_tiny(1000.0 * volg)
    volair = 1.0 - volice - volwater
    volmix1 = volice / torch.clamp_min(volice + volwater, 1e-10)
    m1t = m_w * m_w
    m3t = m_i * m_i
    beta3 = _RADAR_BETA_WATER_ICE if m1t == _RADAR_COMPLEX_WATER_SQ and m3t == _RADAR_COMPLEX_ICE_SQ else _complex_beta_constant_python(m1t, m3t)
    vol3 = volmix1
    m_tmp_sq = _complex_div_pair(((1.0 - vol3) * m1t.real + vol3 * beta3.real * m3t.real - vol3 * beta3.imag * m3t.imag, (1.0 - vol3) * m1t.imag + vol3 * beta3.real * m3t.imag + vol3 * beta3.imag * m3t.real), (1.0 - vol3 + vol3 * beta3.real, vol3 * beta3.imag))
    m_tmp = _complex_sqrt_pair_no_cache(m_tmp_sq)
    m1t_pair = _complex_mul_pair(m_tmp, m_tmp)
    m2t = (1.0, 0.0)
    vol2 = volair
    beta2 = _complex_beta_pair(m1t_pair, m2t)
    numerator = ((1.0 - vol2) * m1t_pair[0] + vol2 * beta2[0], (1.0 - vol2) * m1t_pair[1] + vol2 * beta2[1])
    denominator = (1.0 - vol2 + vol2 * beta2[0], vol2 * beta2[1])
    m_core = _complex_sqrt_pair_no_cache(_complex_div_pair(numerator, denominator))
    m_core_sq = _complex_mul_pair(m_core, m_core)
    ratio = _complex_div_pair((m_core_sq[0] - 1.0, m_core_sq[1]), (m_core_sq[0] + 2.0, m_core_sq[1]))
    backscatter = _complex_abs_square_pair(ratio)
    backscatter = backscatter * float(_const_segment_e_PI ** 5 / lambda_radar ** 4) * _pow6(d_large)
    if isinstance(d_g, torch.Tensor):
        return torch.where(d_g < float(_const_segment_e_R1), 0.0, backscatter)
    return torch.zeros_like(backscatter) if d_g < float(_const_segment_e_R1) else backscatter


def effective_radius_tensor(
    *,
    temp: torch.Tensor,
    l_qc: torch.Tensor,
    nc: torch.Tensor,
    ilamc: torch.Tensor,
    l_qi: torch.Tensor,
    ilami: torch.Tensor,
    l_qs: torch.Tensor,
    rs: torch.Tensor,
    snow_moments: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if snow_moments is None:
        zero_template = {name: torch.zeros_like(rs) for name in constants.SNOW_MOMENT_FIELDS}
        snow_moments = compute_snow_moments(rs=rs, temp=temp, l_qs=l_qs, template=zero_template)
    return _run_tensor_compile_island(
        "effective_radius",
        _effective_radius_kernel,
        rs,
        l_qc,
        nc,
        ilamc,
        l_qi,
        ilami,
        l_qs,
        snow_moments["smob"],
        snow_moments["smoc"],
    )


def _effective_radius_kernel(
    l_qc: torch.Tensor,
    nc: torch.Tensor,
    ilamc: torch.Tensor,
    l_qi: torch.Tensor,
    ilami: torch.Tensor,
    l_qs: torch.Tensor,
    smob: torch.Tensor,
    smoc: torch.Tensor,
    _const_segment_f_NT_C_MIN=constants.NT_C_MIN,
    _const_segment_f_NU_C_SCALE=constants.NU_C_SCALE,
    _const_segment_e_MU_I=constants.MU_I,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nc_safe = torch.clamp_min(nc, float(_const_segment_f_NT_C_MIN))
    ratio = float(_const_segment_f_NU_C_SCALE) / nc_safe
    nu_c_int = torch.where(
        ratio >= 12.5,
        15,
        torch.clamp(torch.floor(ratio + 0.5).to(torch.int64) + 2, max=15),
    )
    nu_c = nu_c_int.to(dtype=nc.dtype)
    re_qc_value = torch.clamp(
        0.5 * (3.0 + nu_c) * ilamc,
        min=2.51e-6,
        max=50.0e-6,
    )
    re_qi_value = torch.clamp(
        float(0.5 * (3.0 + _const_segment_e_MU_I)) * ilami,
        min=2.51e-6,
        max=125.0e-6,
    )
    re_qs_value = torch.clamp(
        0.5 * smoc / _clamp_tiny(smob),
        min=5.01e-6,
        max=999.0e-6,
    )
    return (
        torch.where(l_qc, re_qc_value, 0.0),
        torch.where(l_qi, re_qi_value, 0.0),
        torch.where(l_qs, re_qs_value, 0.0),
    )


def max_hail_diam_tensor(
    *,
    rho: torch.Tensor,
    rg: torch.Tensor,
    ng: torch.Tensor,
    ilamg: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    hbins, dhbins = _hail_bins_tensor(rho)
    density_table = _tensor(_STATIC_GRAUPEL_RHO_G, rho)
    if DIAGNOSTIC_COMPUTE_MODE == "compact_active":
        return _max_hail_diam_compact_active_tensor(
            rho=rho,
            rg=rg,
            ng=ng,
            ilamg=ilamg,
            idx=idx,
            hbins=hbins,
            dhbins=dhbins,
            density_table=density_table,
        )
    if int(rho.numel()) * int(constants.NHBINS) <= HAIL_FULL_FUSION_MAX_ELEMENTS:
        return _run_tensor_compile_island(
            "max_hail_diam",
            _max_hail_diam_full_kernel,
            rho,
            rho,
            rg,
            ng,
            ilamg,
            idx,
            hbins,
            dhbins,
            density_table,
        )
    return _run_tensor_compile_island(
        "max_hail_diam_streaming",
        _max_hail_diam_streaming_kernel,
        rho,
        rho,
        rg,
        ng,
        ilamg,
        idx,
        hbins,
        dhbins,
        density_table,
    )


def _max_hail_diam_compact_active_tensor(
    *,
    rho: torch.Tensor,
    rg: torch.Tensor,
    ng: torch.Tensor,
    ilamg: torch.Tensor,
    idx: torch.Tensor,
    hbins: torch.Tensor,
    dhbins: torch.Tensor,
    density_table: torch.Tensor,
) -> torch.Tensor:
    """Apply the Fortran eligibility gate before the ordered 50-bin scan."""
    active, _, _ = _max_hail_terms(rho, rg, ng, ilamg, idx, density_table)
    active_indices = torch.nonzero(active.reshape(-1), as_tuple=False)[:, 0]
    rho_flat = rho.reshape(-1)
    selected_rho = rho_flat.index_select(0, active_indices)
    selected = _run_sparse_diagnostic_compile_island(
        "max_hail_diam_compact_active",
        _max_hail_diam_full_kernel,
        selected_rho,
        selected_rho,
        rg.reshape(-1).index_select(0, active_indices),
        ng.reshape(-1).index_select(0, active_indices),
        ilamg.reshape(-1).index_select(0, active_indices),
        idx.reshape(-1).index_select(0, active_indices),
        hbins,
        dhbins,
        density_table,
        enabled=TENSOR_COMPILE_ISLAND_CUDA,
        devices=TENSOR_COMPILE_ISLAND_DEVICES,
    )
    return torch.zeros_like(rho_flat).index_copy(0, active_indices, selected).reshape(rho.shape)


def _max_hail_terms(rho: torch.Tensor, rg: torch.Tensor, ng: torch.Tensor, ilamg: torch.Tensor, idx: torch.Tensor, density_table: torch.Tensor, _const_segment_g_NRHG=constants.NRHG, _const_segment_g_OGG2=constants.OGG2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx_safe = torch.clamp(idx.to(torch.int64), 1, _const_segment_g_NRHG)
    density = density_table[idx_safe - 1]
    active = (rg / rho >= 1e-06) & (density >= 350.0)
    lamg = 1.0 / torch.where(ilamg > 0.0, ilamg, 1.0)
    n0_g = ng * float(_const_segment_g_OGG2) * lamg
    return (active, lamg, n0_g)


def _max_hail_diam_full_kernel(rho: torch.Tensor, rg: torch.Tensor, ng: torch.Tensor, ilamg: torch.Tensor, idx: torch.Tensor, hbins: torch.Tensor, dhbins: torch.Tensor, density_table: torch.Tensor, _const_segment_r1_NHBINS=constants.NHBINS, _const_segment_g_MU_G=constants.MU_G, _const_segment_r1_HAIL_CONC_THRESHOLD=constants.HAIL_CONC_THRESHOLD) -> torch.Tensor:
    active, lamg, n0_g = _max_hail_terms(rho, rg, ng, ilamg, idx, density_table)
    bin_shape = (1,) * rho.ndim + (_const_segment_r1_NHBINS,)
    descending_bins = torch.flip(hbins, dims=(0,))
    descending_widths = torch.flip(dhbins, dims=(0,))
    bin_values = descending_bins.reshape(bin_shape)
    bin_weight = 1.0 if float(_const_segment_g_MU_G) == 0.0 else torch.pow(bin_values, float(_const_segment_g_MU_G))
    increments = n0_g.unsqueeze(-1) * bin_weight * torch.exp(-lamg.unsqueeze(-1) * bin_values) * descending_widths.reshape(bin_shape)
    cumulative = torch.cumsum(increments, dim=-1)
    crossed = active.unsqueeze(-1) & (cumulative > float(_const_segment_r1_HAIL_CONC_THRESHOLD))
    has_crossing = torch.any(crossed, dim=-1)
    first_crossing = torch.argmax(crossed.to(torch.int64), dim=-1)
    gather_index = first_crossing.unsqueeze(-1)
    sum_nh = torch.gather(cumulative, dim=-1, index=gather_index).squeeze(-1)
    previous_index = torch.clamp(first_crossing - 1, 0, _const_segment_r1_NHBINS - 1).unsqueeze(-1)
    sum_t = torch.where(first_crossing > 0, torch.gather(cumulative, dim=-1, index=previous_index).squeeze(-1), 0.0)
    bin_value = descending_bins[first_crossing]
    next_bin_value = descending_bins[torch.clamp(first_crossing - 1, 0, _const_segment_r1_NHBINS - 1)]
    interpolated = bin_value - (sum_nh - float(_const_segment_r1_HAIL_CONC_THRESHOLD)) / _safe_denominator_no_cache(sum_nh - sum_t) * (bin_value - next_bin_value)
    candidate_hail = torch.where(first_crossing == 0, descending_bins[0], torch.where(next_bin_value > 0.001, interpolated, rho.new_full((), 0.0001)))
    hail_max = torch.where(has_crossing, candidate_hail, 0.0)
    return 1000.0 * hail_max


def _max_hail_diam_streaming_kernel(rho: torch.Tensor, rg: torch.Tensor, ng: torch.Tensor, ilamg: torch.Tensor, idx: torch.Tensor, hbins: torch.Tensor, dhbins: torch.Tensor, density_table: torch.Tensor, _const_segment_r1_HAIL_CONC_THRESHOLD=constants.HAIL_CONC_THRESHOLD, _const_segment_r1_NHBINS=constants.NHBINS, _const_segment_g_MU_G=constants.MU_G) -> torch.Tensor:
    active, lamg, n0_g = _max_hail_terms(rho, rg, ng, ilamg, idx, density_table)
    cumulative = torch.zeros_like(rho)
    hail_max = torch.zeros_like(rho)
    has_crossing = torch.zeros(rho.shape, dtype=torch.bool, device=rho.device)
    threshold = float(_const_segment_r1_HAIL_CONC_THRESHOLD)
    for bin_index in range(_const_segment_r1_NHBINS - 1, -1, -1):
        previous_sum = cumulative
        bin_value = hbins[bin_index]
        bin_weight = 1.0 if float(_const_segment_g_MU_G) == 0.0 else torch.pow(bin_value, float(_const_segment_g_MU_G))
        increment = n0_g * bin_weight * torch.exp(-lamg * bin_value) * dhbins[bin_index]
        cumulative = cumulative + increment
        crossing_now = active & ~has_crossing & (cumulative > threshold)
        if bin_index >= _const_segment_r1_NHBINS - 1:
            candidate_hail = hbins[-1]
        else:
            next_bin_value = hbins[bin_index + 1]
            interpolated = bin_value - (cumulative - threshold) / _safe_denominator_no_cache(cumulative - previous_sum) * (bin_value - next_bin_value)
            candidate_hail = torch.where(next_bin_value > 0.001, interpolated, 0.0001)
        hail_max = torch.where(crossing_now, candidate_hail, hail_max)
        has_crossing = has_crossing | crossing_now
    return 1000.0 * hail_max


def _refresh_thermo_tensor(
    *,
    input_state: Mapping[str, torch.Tensor],
    tendencies: Mapping[str, torch.Tensor],
    pres: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Mapping[str, torch.Tensor]]:
    if (
        TENSOR_COMPILE_ISLAND_CUDA
        and pres.device.type in TENSOR_COMPILE_ISLAND_DEVICES
        and not torch.compiler.is_compiling()
    ):
        values = _run_tensor_compile_island(
            "refresh_thermo",
            _refresh_thermo_kernel,
            pres,
            input_state["qv"],
            input_state["t"],
            tendencies["qvten"],
            tendencies["tten"],
            pres,
            float(dt),
        )
        qv, temp, rho = values[:3]
        return qv, temp, rho, dict(zip(_THERMO_VAR_FIELD_NAMES, values[3:], strict=True))
    qv = torch.clamp_min(input_state["qv"] + tendencies["qvten"] * _scalar(dt, pres), _scalar(constants.MIN_QV, pres))
    temp = input_state["t"] + tendencies["tten"] * _scalar(dt, pres)
    rho = _scalar(constants.ROVERRV, temp) * pres / (_scalar(constants.RDRY, temp) * temp * (qv + _scalar(constants.ROVERRV, temp)))
    thermo, _ = thermo_vars(qv=qv, temp=temp, pres=pres, rho=rho)
    return qv, temp, rho, thermo


_THERMO_VAR_FIELD_NAMES = (
    "rhof",
    "rhof2",
    "qvs",
    "delqvs",
    "qvsi",
    "satw",
    "sati",
    "ssatw",
    "ssati",
    "diffu",
    "visco",
    "vsc2",
    "ocp",
    "lvap",
    "tcond",
    "lvt2",
)


def _refresh_thermo_kernel(
    input_qv: torch.Tensor,
    input_temp: torch.Tensor,
    qvten: torch.Tensor,
    tten: torch.Tensor,
    pres: torch.Tensor,
    dt: float,
    _min_qv: float = float(constants.MIN_QV),
    _roverrv: float = float(constants.ROVERRV),
    _rdry: float = float(constants.RDRY),
) -> tuple[torch.Tensor, ...]:
    qv = torch.clamp_min(input_qv + qvten * dt, _min_qv)
    temp = input_temp + tten * dt
    rho = _roverrv * pres / (_rdry * temp * (qv + _roverrv))
    thermo = _thermo_vars_kernel(qv, temp, pres, rho)
    return (qv, temp, rho, *thermo[:-1])


def thermo_vars(
    *,
    qv: torch.Tensor,
    temp: torch.Tensor,
    pres: torch.Tensor,
    rho: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if not (
        TENSOR_COMPILE_ISLAND_CUDA
        and temp.device.type in TENSOR_COMPILE_ISLAND_DEVICES
        and not torch.compiler.is_compiling()
    ):
        return _thermo_vars_eager(qv=qv, temp=temp, pres=pres, rho=rho)
    values = _run_tensor_compile_island(
        "thermo_vars",
        _thermo_vars_kernel,
        temp,
        qv,
        temp,
        pres,
        rho,
    )
    return dict(zip(_THERMO_VAR_FIELD_NAMES, values[:-1], strict=True)), values[-1]


def _thermo_vars_eager(
    *,
    qv: torch.Tensor,
    temp: torch.Tensor,
    pres: torch.Tensor,
    rho: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    tempc = temp - _scalar(constants.T0, temp)
    rhof = torch.sqrt(_scalar(constants.RHO_NOT, temp) / rho)
    rhof2 = torch.sqrt(rhof)
    qvs = calc_rslf(pres, temp)
    delqvs = torch.clamp_min(calc_rslf(pres, _scalar(float(constants.T0), temp)) - qv, 0.0)
    qvsi = torch.where(tempc <= 0.0, calc_rsif(pres, temp), qvs)
    satw = qv / qvs
    sati = qv / qvsi
    ssatw = satw - 1.0
    ssati = sati - 1.0
    ssatw = torch.where(torch.abs(ssatw) < _scalar(constants.EPS, temp), 0.0, ssatw)
    ssati = torch.where(torch.abs(ssati) < _scalar(constants.EPS, temp), 0.0, ssati)
    diffu = _scalar(2.11e-5, temp) * (temp / _scalar(constants.T0, temp)) ** _scalar(1.94, temp) * (_scalar(101325.0, temp) / pres)
    visco = torch.where(
        tempc >= 0.0,
        (_scalar(1.718, temp) + _scalar(0.0049, temp) * tempc) * _scalar(1.0e-5, temp),
        (_scalar(1.718, temp) + _scalar(0.0049, temp) * tempc - _scalar(1.2e-5, temp) * tempc * tempc) * _scalar(1.0e-5, temp),
    )
    ocp = 1.0 / (_scalar(constants.CP, temp) * (1.0 + _scalar(0.887, temp) * qv))
    vsc2 = torch.sqrt(rho / visco)
    lvap = _scalar(constants.LVAP0, temp) + _scalar(2106.0 - 4218.0, temp) * tempc
    # Conductivity roundoff propagates through near-total ice depletion.
    tcond = (_scalar(5.69, temp) + _scalar(0.0168, temp) * tempc) * _scalar(1.0e-5, temp) * _scalar(418.936, temp)
    otemp = 1.0 / temp
    lvt2 = lvap * lvap * ocp * _scalar(constants.ORV, temp) * otemp * otemp
    return (
        {
            "rhof": rhof,
            "rhof2": rhof2,
            "qvs": qvs,
            "delqvs": delqvs,
            "qvsi": qvsi,
            "satw": satw,
            "sati": sati,
            "ssatw": ssatw,
            "ssati": ssati,
            "diffu": diffu,
            "visco": visco,
            "vsc2": vsc2,
            "ocp": ocp,
            "lvap": lvap,
            "tcond": tcond,
            "lvt2": lvt2,
        },
        ssati > 0.0,
    )


def _calc_rslf_no_cache(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    x = torch.clamp_min(t - 273.16, -80.0)
    esl = 0.611583699e03 + x * (
        0.444606896e02
        + x
        * (
            0.143177157e01
            + x
            * (
                0.264224321e-1
                + x
                * (
                    0.299291081e-3
                    + x * (0.203154182e-5 + x * (0.702620698e-8 + x * (0.379534310e-11 + x * -0.321582393e-13)))
                )
            )
        )
    )
    esl = torch.minimum(esl, p * 0.15)
    return 0.622 * esl / (p - esl)


def _calc_rsif_no_cache(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    x = torch.clamp_min(t - 273.16, -80.0)
    esi = 0.609868993e03 + x * (
        0.499320233e02
        + x
        * (
            0.184672631e01
            + x
            * (
                0.402737184e-1
                + x
                * (
                    0.565392987e-3
                    + x * (0.521693933e-5 + x * (0.307839583e-7 + x * (0.105785160e-9 + x * 0.161444444e-12)))
                )
            )
        )
    )
    esi = torch.minimum(esi, p * 0.15)
    return 0.622 * esi / torch.clamp_min(p - esi, 1.0e-4)


def _thermo_vars_kernel(
    qv: torch.Tensor,
    temp: torch.Tensor,
    pres: torch.Tensor,
    rho: torch.Tensor,
    _t0: float = float(constants.T0),
    _rho_not: float = float(constants.RHO_NOT),
    _eps: float = float(constants.EPS),
    _cp: float = float(constants.CP),
    _orv: float = float(constants.ORV),
    _lvap0: float = float(constants.LVAP0),
) -> tuple[torch.Tensor, ...]:
    tempc = temp - _t0
    rhof = torch.sqrt(_rho_not / rho)
    rhof2 = torch.sqrt(rhof)
    qvs = _calc_rslf_no_cache(pres, temp)
    delqvs = torch.clamp_min(_calc_rslf_no_cache(pres, torch.full_like(temp, _t0)) - qv, 0.0)
    qvsi = torch.where(tempc <= 0.0, _calc_rsif_no_cache(pres, temp), qvs)
    satw = qv / qvs
    sati = qv / qvsi
    ssatw = satw - 1.0
    ssati = sati - 1.0
    ssatw = torch.where(torch.abs(ssatw) < _eps, 0.0, ssatw)
    ssati = torch.where(torch.abs(ssati) < _eps, 0.0, ssati)
    diffu = 2.11e-5 * (temp / _t0) ** 1.94 * (101325.0 / pres)
    visco = torch.where(
        tempc >= 0.0,
        (1.718 + 0.0049 * tempc) * 1.0e-5,
        (1.718 + 0.0049 * tempc - 1.2e-5 * tempc * tempc) * 1.0e-5,
    )
    ocp = 1.0 / (_cp * (1.0 + 0.887 * qv))
    vsc2 = torch.sqrt(rho / visco)
    lvap = _lvap0 + (2106.0 - 4218.0) * tempc
    tcond = (5.69 + 0.0168 * tempc) * 1.0e-5 * 418.936
    otemp = 1.0 / temp
    lvt2 = lvap * lvap * ocp * _orv * otemp * otemp
    return (
        rhof,
        rhof2,
        qvs,
        delqvs,
        qvsi,
        satw,
        sati,
        ssatw,
        ssati,
        diffu,
        visco,
        vsc2,
        ocp,
        lvap,
        tcond,
        lvt2,
        ssati > 0.0,
    )


def calc_rslf(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    x = _max_first(_scalar(-80.0, t), t - _scalar(273.16, t))
    esl = _scalar(0.611583699e03, t) + x * (
        _scalar(0.444606896e02, t)
        + x
        * (
            _scalar(0.143177157e01, t)
            + x
            * (
                _scalar(0.264224321e-1, t)
                + x
                * (
                    _scalar(0.299291081e-3, t)
                    + x
                    * (_scalar(0.203154182e-5, t) + x * (_scalar(0.702620698e-8, t) + x * (_scalar(0.379534310e-11, t) + x * _scalar(-0.321582393e-13, t))))
                )
            )
        )
    )
    esl = torch.minimum(esl, p * _scalar(0.15, t))
    return _scalar(0.622, t) * esl / (p - esl)


def calc_rsif(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    x = _max_first(_scalar(-80.0, t), t - _scalar(273.16, t))
    esi = _scalar(0.609868993e03, t) + x * (
        _scalar(0.499320233e02, t)
        + x
        * (
            _scalar(0.184672631e01, t)
            + x
            * (
                _scalar(0.402737184e-1, t)
                + x
                * (
                    _scalar(0.565392987e-3, t)
                    + x
                    * (_scalar(0.521693933e-5, t) + x * (_scalar(0.307839583e-7, t) + x * (_scalar(0.105785160e-9, t) + x * _scalar(0.161444444e-12, t))))
                )
            )
        )
    )
    esi = torch.minimum(esi, p * _scalar(0.15, t))
    return _scalar(0.622, t) * esi / _max_first(_scalar(1.0e-4, t), p - esi)


def _temperature_table_index(temp: torch.Tensor) -> torch.Tensor:
    tempc = temp - _scalar(constants.T0, temp)
    idx_t = torch.trunc((tempc - _scalar(2.5, temp)) / _scalar(5.0, temp)).to(torch.int64) - 1
    return torch.clamp(-idx_t, 1, constants.NTB_T)


def _mass_table_index(mass: torch.Tensor, *, exponent_offset: int, upper: int) -> torch.Tensor:
    safe = _clamp_tiny(mass)
    exponent = torch.floor(torch.log10(safe))
    exponent_i = exponent.to(torch.int64)
    mantissa = safe / torch.pow(_scalar(10.0, mass), exponent)
    digit = torch.floor(mantissa).to(torch.int64)
    index = digit + 9 * (exponent_i - int(exponent_offset))
    return torch.clamp(index, 1, int(upper))


def _rain_table_indices(rr: torch.Tensor, ilamr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    idx_r = _mass_table_index(rr, exponent_offset=constants.NIR2, upper=constants.NTB_R)
    ilamr_safe = torch.where(ilamr > 0.0, ilamr, _scalar(1.0, ilamr))
    lamr = 1.0 / ilamr_safe
    lam_exp = lamr * _scalar(
        (constants.CRG[2] * constants.ORG2 * (1.0 / constants.CRG[0])) ** constants.BM_R,
        rr,
    )
    n0_exp = _scalar(1.0 / _SCALAR_CRG[0] / constants.AM_R, rr) * rr * lam_exp ** _scalar(_SCALAR_CRE[0], rr)
    idx_r1 = _mass_table_index(n0_exp, exponent_offset=constants.NIR3, upper=constants.NTB_R1)
    return idx_r, idx_r1


def _graupel_table_indices(
    rg: torch.Tensor,
    ilamg: torch.Tensor,
    density_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx_g = _mass_table_index(rg, exponent_offset=constants.NIG2, upper=constants.NTB_G)
    ilamg_safe = torch.where(ilamg > 0.0, ilamg, _scalar(1.0, ilamg))
    lamg = 1.0 / ilamg_safe
    lam_exp = lamg * _scalar(
        (constants.CGG[2][0] * constants.OGG2 * constants.OGG1) ** constants.BM_G,
        rg,
    )
    am_g = _take(_STATIC_GRAUPEL_AM_G, density_idx, rg)
    n0_exp = _scalar(constants.OGG1, rg) * rg / am_g * lam_exp ** _scalar(_SCALAR_GRAUPEL_CGE[0][0], rg)
    idx_g1 = _mass_table_index(n0_exp, exponent_offset=constants.NIG3, upper=constants.NTB_G1)
    return idx_g, idx_g1


def _table_gather(
    arrays: Mapping[str, Any],
    name: str,
    reference: torch.Tensor,
    *indices_1based: torch.Tensor,
) -> torch.Tensor:
    return _table_gather_array(arrays[name], reference, *indices_1based)


def _table_gather_array(
    array: Any,
    reference: torch.Tensor,
    *indices_1based: torch.Tensor,
) -> torch.Tensor:
    indices_0based = tuple(_table_index_0based(index, reference) for index in indices_1based)
    torch_gather = getattr(array, "torch_gather", None)
    if torch_gather is not None:
        return torch_gather(indices_0based, reference)
    table = _table_tensor(array, reference)
    return table[indices_0based]


def _table_index_0based(index_1based: torch.Tensor | int | np.integer, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(index_1based, (int, np.integer)):
        return torch.full(reference.shape, int(index_1based) - 1, dtype=torch.int64, device=reference.device)
    index = index_1based.to(torch.int64) - 1
    if index.ndim == 0 and reference.ndim > 0:
        return index.expand(reference.shape)
    return index


def prepare_torch_tables(tables: Any, reference: torch.Tensor | Mapping[str, torch.Tensor]) -> Any:
    """Materialize TEMPO lookup tables as torch tensors before compiled/timed runs."""
    arrays = getattr(tables, "arrays", None)
    if not isinstance(arrays, Mapping):
        return tables
    if isinstance(reference, Mapping):
        try:
            reference_tensor = next(iter(reference.values()))
        except StopIteration as exc:
            raise ValueError("reference state for table preparation is empty") from exc
    else:
        reference_tensor = reference
    torch_arrays = {
        name: array if getattr(array, "torch_gather", None) is not None else _table_tensor(array, reference_tensor)
        for name, array in arrays.items()
    }
    file_records = tuple(getattr(tables, "file_records", ()))
    return TorchFrozenTables(arrays=torch_arrays, file_records=file_records, mmap_enabled=False)


def _table_tensor(array: Any, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return _static_tensor(array, reference)
    if torch.compiler.is_compiling():
        # Traced as a graph constant. Dynamo cannot read dtype.byteorder; a
        # non-native table fails loudly in from_numpy instead of converting.
        return torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
    if _is_fake_or_meta_tensor(reference):
        native_array = array
        dtype = getattr(array, "dtype", None)
        if dtype is not None and getattr(dtype, "byteorder", "=") not in ("=", "|"):
            native_array = array.astype(dtype.newbyteorder("="))
        return torch.as_tensor(native_array, dtype=reference.dtype, device=reference.device)
    key = _table_tensor_cache_key(array, reference)
    cached = _TORCH_TABLE_CACHE.get(key)
    if cached is None:
        native_array = array
        dtype = getattr(array, "dtype", None)
        if dtype is not None and getattr(dtype, "byteorder", "=") not in ("=", "|"):
            native_array = array.astype(dtype.newbyteorder("="))
        cached = torch.as_tensor(native_array, dtype=reference.dtype, device=reference.device)
        _TORCH_TABLE_CACHE[key] = cached
    return cached


def _static_tensor(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == reference.dtype and tensor.device == reference.device:
        return tensor
    if _is_fake_or_meta_tensor(reference):
        return tensor.to(dtype=reference.dtype, device=reference.device)
    try:
        source_identity: tuple[Any, ...] = (
            "data",
            int(tensor.data_ptr()),
            int(tensor.storage_offset()),
        )
    except RuntimeError:
        source_identity = ("object", id(tensor))
    key = (
        source_identity,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        reference.dtype,
        str(reference.device),
    )
    cached = _TORCH_STATIC_TENSOR_CACHE.get(key)
    if cached is None:
        cached = tensor.to(dtype=reference.dtype, device=reference.device)
        _TORCH_STATIC_TENSOR_CACHE[key] = cached
    return cached


def _table_tensor_cache_key(array: Any, reference: torch.Tensor) -> tuple[Any, ...]:
    return (
        _array_cache_identity(array),
        tuple(getattr(array, "shape", ())),
        tuple(getattr(array, "strides", ())),
        str(getattr(array, "dtype", "")),
        str(reference.device),
        reference.dtype,
        torch.is_inference_mode_enabled(),
    )


def _array_cache_identity(array: Any) -> tuple[str, int] | tuple[str, int, int]:
    interface = getattr(array, "__array_interface__", None)
    if isinstance(interface, Mapping):
        data = interface.get("data")
        if isinstance(data, tuple) and data:
            base = getattr(array, "base", None)
            owner_id = id(base) if base is not None else id(array)
            return ("data", int(data[0]), owner_id)
    return ("object", id(array))


def _activation_table_array(tables: Any) -> Any:
    arrays = getattr(tables, "arrays", None)
    if isinstance(arrays, Mapping) and "tnccn_act" in arrays:
        return arrays["tnccn_act"]
    native_array = getattr(tables, "native_array", None)
    if native_array is not None:
        return native_array("tnccn_act")
    raise KeyError("TEMPO activation table tnccn_act is required for Segment O")


def _libm_power(values: torch.Tensor, exponent: float) -> torch.Tensor:
    """The source's real power: a tensor exponent reaches libm pow; a Python 2.0 or 3.0 takes torch's multiplication."""
    return working_power(values, values.new_tensor(exponent))


def _square(values: torch.Tensor) -> torch.Tensor:
    return values * values


def _cube(values: torch.Tensor) -> torch.Tensor:
    return values * values * values


def _pow4(values: torch.Tensor) -> torch.Tensor:
    squared = values * values
    return squared * squared


def _pow6(values: torch.Tensor) -> torch.Tensor:
    squared = values * values
    return squared * squared * squared


def _min_first(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    # Preserve Fortran first-operand tie behavior and prevent rejected-branch
    # tangents from leaking through torch.minimum's forward-AD rule.
    return torch.where(first <= second, first, second)


def _max_first(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.where(first >= second, first, second)


def _in_table_index(xni: torch.Tensor) -> torch.Tensor:
    return torch.where(
        xni > _scalar(1.0, xni),
        _mass_table_index(xni, exponent_offset=constants.NIIN2, upper=constants.NTB_IN),
        _int_scalar(1, xni),
    )


def _freezing_temperature_index(temp: torch.Tensor) -> torch.Tensor:
    tempc = temp - _scalar(constants.T0, temp)
    return torch.clamp(_fortran_nint_tensor(-tempc), 1, constants.NTB_T1)


def _cloud_table_indices(rc: torch.Tensor, nc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    idx_c = _mass_table_index(rc, exponent_offset=constants.NIC2, upper=constants.NTB_C)
    t_nc = _tensor(_STATIC_CLOUD_NUMBER_BINS, nc)
    nc_safe = torch.clamp_min(nc, t_nc[0])
    idx_n = _fortran_nint_tensor(
        _scalar(1.0, nc)
        + _scalar(constants.NBC, nc)
        * torch.log(nc_safe / t_nc[0])
        / torch.log(t_nc[-1] / t_nc[0])
    )
    return idx_c, torch.clamp(idx_n, 1, constants.NBC)


def _bounded_density_index_tensor(value: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.floor(value + _scalar(0.5, value)).to(torch.int64) + 1, 1, constants.NRHG)


def _demott_nucleation(tempc: torch.Tensor, rho: torch.Tensor, nifa: torch.Tensor) -> torch.Tensor:
    nifa_cc = _max_first(
        _scalar(0.5, nifa),
        nifa * _scalar(constants.RHO_NOT0 * 1.0e-6, nifa) / rho,
    )
    cold_magnitude = torch.clamp_min(-tempc, _scalar(0.0, tempc))
    xni = (
        _scalar(5.94e-5, tempc)
        * cold_magnitude ** _scalar(3.33, tempc)
        * nifa_cc ** ((-_scalar(0.0264, tempc) * tempc) + _scalar(0.0033, tempc))
    )
    xni = xni * rho / _scalar(constants.RHO_NOT0, rho) * _scalar(1000.0, rho)
    return torch.clamp_min(xni, _scalar(0.0, xni))


def _koop_nucleation(temp: torch.Tensor, satw: torch.Tensor, naero: torch.Tensor, dt: float) -> torch.Tensor:
    mu_diff = (
        _scalar(210368.0, temp)
        + _scalar(131.438, temp) * temp
        - _scalar(3.32373e6, temp) / temp
        - _scalar(41729.1, temp) * torch.log(temp)
    )
    a_w_i = torch.exp(mu_diff / (_scalar(constants.R_UNI, temp) * temp))
    delta_aw = satw - a_w_i
    log_j_rate = (
        -_scalar(906.7, temp)
        + _scalar(8502.0, temp) * delta_aw
        - _scalar(26924.0, temp) * delta_aw * delta_aw
        + _scalar(29180.0, temp) * delta_aw * delta_aw * delta_aw
    )
    log_j_rate = _min_first(_scalar(20.0, log_j_rate), log_j_rate)
    j_rate = torch.pow(_scalar(10.0, temp), log_j_rate)
    prob_h = _min_first(
        _scalar(1.0, j_rate),
        1.0 - torch.exp(-j_rate * _scalar(constants.AR_VOLUME * dt, temp)),
    )
    xni = torch.where(prob_h > 0.0, _min_first(_scalar(1000.0e3, prob_h), prob_h * naero), 0.0)
    return torch.clamp_min(xni, _scalar(0.0, xni))


def _fortran_nint_tensor(value: torch.Tensor) -> torch.Tensor:
    return torch.where(
        value >= 0.0,
        torch.floor(value + _scalar(0.5, value)),
        torch.ceil(value - _scalar(0.5, value)),
    ).to(torch.int64)


def _ice_table_indices(ri: torch.Tensor, ni: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    idx_i = torch.where(
        ri > 0.0,
        _mass_table_index(ri, exponent_offset=constants.NII2, upper=constants.NTB_I),
        _int_scalar(1, ri),
    )
    idx_i1 = torch.where(
        ni > 0.0,
        _mass_table_index(ni, exponent_offset=constants.NII3, upper=constants.NTB_I1),
        _int_scalar(1, ni),
    )
    return idx_i, idx_i1


def _ice_autoconversion_tensors(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tpi_ide, tps_iaus, tni_iaus = _STATIC_ICE_AUTOCONVERSION_TABLES
    return _tensor(tpi_ide, reference), _tensor(tps_iaus, reference), _tensor(tni_iaus, reference)


def _scale_tendencies_if(
    tend: dict[str, torch.Tensor],
    names: tuple[str, ...],
    mask: torch.Tensor,
    ratio: torch.Tensor,
) -> dict[str, torch.Tensor]:
    safe_ratio = torch.where(mask, ratio, _scalar(1.0, ratio))
    for name in names:
        tend[name] = tend[name] * safe_ratio
    return tend



def _tiny_value(reference: torch.Tensor) -> float:
    """The divide/log guard: 1e-300 in float64 (unchanged), the smallest normal number otherwise.

    1e-300 rounds to 0 in float32, so the guard vanished and masked lanes computed 0/0, whose backward
    turned every parameter gradient into NaN in reverse-mode checks.
    """
    return 1.0e-300 if reference.dtype == torch.float64 else float(torch.finfo(reference.dtype).tiny)


def _clamp_tiny(values: torch.Tensor) -> torch.Tensor:
    return torch.clamp_min(values, _tiny_value(values))

def _sign_one(values: torch.Tensor) -> torch.Tensor:
    return torch.where(values < 0.0, _scalar(-1.0, values), _scalar(1.0, values))


def _safe_denominator(values: torch.Tensor) -> torch.Tensor:
    tiny = _scalar(_tiny_value(values), values)
    tiny_with_sign = torch.where(values < 0.0, -tiny, tiny)
    return torch.where(torch.abs(values) > tiny, values, tiny_with_sign)


def _safe_denominator_no_cache(values: torch.Tensor) -> torch.Tensor:
    tiny = _tiny_value(values)
    tiny_with_sign = torch.where(values < 0.0, -tiny, tiny)
    return torch.where(torch.abs(values) > tiny, values, tiny_with_sign)


def _complex_mul_pair(
    first: tuple[Any, Any],
    second: tuple[Any, Any],
) -> tuple[Any, Any]:
    return (
        first[0] * second[0] - first[1] * second[1],
        first[0] * second[1] + first[1] * second[0],
    )


def _complex_div_pair(
    numerator: tuple[Any, Any],
    denominator: tuple[Any, Any],
) -> tuple[Any, Any]:
    denom = denominator[0] * denominator[0] + denominator[1] * denominator[1]
    return (
        (numerator[0] * denominator[0] + numerator[1] * denominator[1]) / denom,
        (numerator[1] * denominator[0] - numerator[0] * denominator[1]) / denom,
    )


def _complex_log_pair(values: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        0.5 * torch.log(values[0] * values[0] + values[1] * values[1]),
        torch.atan2(values[1], values[0]),
    )


def _complex_sqrt_pair(values: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    magnitude = torch.hypot(values[0], values[1])
    real = torch.sqrt(torch.clamp_min(0.5 * (magnitude + values[0]), _scalar(0.0, values[0])))
    imag_abs = torch.sqrt(torch.clamp_min(0.5 * (magnitude - values[0]), _scalar(0.0, values[0])))
    imag = torch.where(values[1] < 0.0, -imag_abs, imag_abs)
    return real, imag


def _complex_sqrt_pair_no_cache(values: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    magnitude = torch.hypot(values[0], values[1])
    real = torch.sqrt(torch.clamp_min(0.5 * (magnitude + values[0]), 0.0))
    imag_abs = torch.sqrt(torch.clamp_min(0.5 * (magnitude - values[0]), 0.0))
    imag = torch.where(values[1] < 0.0, -imag_abs, imag_abs)
    return real, imag


def _complex_abs_square_pair(values: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    return values[0] * values[0] + values[1] * values[1]


def _complex_beta_pair(
    m1: tuple[torch.Tensor, torch.Tensor],
    m2: tuple[Any, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    m2_minus_m1 = (m2[0] - m1[0], m2[1] - m1[1])
    m2_over_delta = _complex_div_pair(m2, m2_minus_m1)
    m2_over_m1 = _complex_div_pair(m2, m1)
    log_term = _complex_log_pair(m2_over_m1)
    inner = _complex_mul_pair(m2_over_delta, log_term)
    inner = (inner[0] - 1.0, inner[1])
    return _complex_mul_pair(_complex_div_pair((2.0 * m1[0], 2.0 * m1[1]), m2_minus_m1), inner)


def _column_activity(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 0:
        return mask.expand(1, 1)
    if mask.ndim == 1:
        return mask[:, None]
    return torch.any(mask, dim=1, keepdim=True)


def _contiguous_tend_work_tensor(tend: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    reference = next(iter(tend.values()))
    tend_work = torch.empty((0,), **_factory(reference))
    return tend_work, dict(tend)


def _zero_tensor_mapping(
    shape: tuple[int, int],
    reference: torch.Tensor,
    names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    zero = torch.zeros((1, 1), **_factory(reference)).expand(shape)
    return {name: zero for name in names}


def get_nuc(nc: torch.Tensor) -> torch.Tensor:
    ratio = _scalar(constants.NU_C_SCALE, nc) / nc
    return torch.where(
        ratio >= _scalar(12.5, nc),
        _int_scalar(15, nc),
        torch.clamp(torch.floor(ratio + 0.5).to(torch.int64) + 2, max=15),
    )


def _cloud_lam(nc_work: torch.Tensor, rc: torch.Tensor, nu_c: torch.Tensor) -> torch.Tensor:
    ccg1, ocg1 = _cloud_lambda_table_bundle(nu_c, rc)
    return (nc_work * _scalar(constants.AM_R, rc) * ccg1 * ocg1 / rc) ** _scalar(constants.OBMR, rc)


def _initial_cloud_number(
    *,
    rho: torch.Tensor,
    nc1d: torch.Tensor | None,
    land: torch.Tensor | None,
    shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if nc1d is not None:
        return torch.zeros(shape, **_factory(rho)), None
    nc = get_constant_cloud_number(shape=shape, reference=rho, land=land)
    return nc, nc


def _segment_c_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentCResult,
    *,
    dt: float | None,
    is_first_step: bool | None,
) -> TorchSegmentCResult:
    if isinstance(state, TorchSegmentCResult):
        return state
    return run_c(state, dt=1.0 if dt is None else dt, is_first_step=is_first_step)


def _segment_d_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentDResult,
    *,
    dt: float | None,
    is_first_step: bool | None,
) -> TorchSegmentDResult:
    if isinstance(state, TorchSegmentDResult):
        return state
    return run_d(state, dt=dt, is_first_step=is_first_step)


def _segment_e_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentEResult,
    *,
    dt: float | None,
    is_first_step: bool | None,
) -> TorchSegmentEResult:
    if isinstance(state, TorchSegmentEResult):
        return state
    return run_e(state, dt=dt, is_first_step=is_first_step)


def _segment_f_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentFResult,
    *,
    dt: float | None,
    is_first_step: bool | None,
) -> TorchSegmentFResult:
    if isinstance(state, TorchSegmentFResult):
        return state
    return run_f(state, dt=dt, is_first_step=is_first_step)


def _segment_g_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentGResult,
    *,
    dt: float | None,
    is_first_step: bool | None,
) -> TorchSegmentGResult:
    if isinstance(state, TorchSegmentGResult):
        return state
    return run_g(state, dt=dt, is_first_step=is_first_step)


def _segment_h_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentHResult,
    *,
    dt: float | None,
    is_first_step: bool | None,
    tables: Any | None,
    turn_off_micro: bool,
) -> TorchSegmentHResult:
    if isinstance(state, TorchSegmentHResult):
        return state
    return run_h(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
    )


def _segment_i_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentIResult,
    *,
    dt: float | None,
    is_first_step: bool | None,
    tables: Any | None,
    turn_off_micro: bool,
) -> TorchSegmentIResult:
    if isinstance(state, TorchSegmentIResult):
        return state
    return run_i(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
    )


def _segment_j_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentJResult,
    *,
    dt: float | None,
    is_first_step: bool | None,
    tables: Any | None,
    turn_off_micro: bool,
) -> TorchSegmentJResult:
    if isinstance(state, TorchSegmentJResult):
        return state
    return run_j(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
    )


def _segment_k_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentKResult,
    *,
    dt: float | None,
    is_first_step: bool | None,
    tables: Any | None,
    turn_off_micro: bool,
) -> TorchSegmentKResult:
    if isinstance(state, TorchSegmentKResult):
        return state
    return run_k(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
    )


def _segment_l_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentLResult,
    *,
    dt: float | None,
    is_first_step: bool | None,
    tables: Any | None,
    turn_off_micro: bool,
) -> TorchSegmentLResult:
    if isinstance(state, TorchSegmentLResult):
        return state
    return run_l(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
    )


def _segment_m_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentMResult,
    *,
    dt: float | None,
    is_first_step: bool | None,
    tables: Any | None,
    turn_off_micro: bool,
) -> TorchSegmentMResult:
    if isinstance(state, TorchSegmentMResult):
        return state
    return run_m(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
    )


def _segment_n_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentNResult,
    *,
    dt: float | None,
    is_first_step: bool | None,
    tables: Any | None,
    turn_off_micro: bool,
) -> TorchSegmentNResult:
    if isinstance(state, TorchSegmentNResult):
        return state
    return run_n(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
    )


def _segment_o_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentOResult,
    *,
    dt: float | None,
    is_first_step: bool | None,
    tables: Any | None,
    turn_off_micro: bool,
    cloud_condensation: bool,
) -> TorchSegmentOResult:
    if isinstance(state, TorchSegmentOResult):
        return state
    return run_o(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
    )


def _segment_p1_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentP1Result,
    *,
    dt: float | None,
    is_first_step: bool | None,
    tables: Any | None,
    turn_off_micro: bool,
    cloud_condensation: bool,
    semi_sedi_flag: bool,
    max_sedimentation_substeps: int | None,
) -> TorchSegmentP1Result:
    if isinstance(state, TorchSegmentP1Result):
        return state
    return run_p1(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
    )


def _segment_p2_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentP2Result,
    *,
    dt: float | None,
    is_first_step: bool | None,
    tables: Any | None,
    turn_off_micro: bool,
    cloud_condensation: bool,
    semi_sedi_flag: bool,
    max_sedimentation_substeps: int | None,
) -> TorchSegmentP2Result:
    if isinstance(state, TorchSegmentP2Result):
        return state
    return run_p2(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
    )


def _segment_p3_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentP3Result,
    *,
    dt: float | None,
    is_first_step: bool | None,
    tables: Any | None,
    turn_off_micro: bool,
    cloud_condensation: bool,
    semi_sedi_flag: bool,
    max_sedimentation_substeps: int | None,
) -> TorchSegmentP3Result:
    if isinstance(state, TorchSegmentP3Result):
        return state
    return run_p3(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
    )


def _segment_p4_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentP4Result,
    *,
    dt: float | None,
    is_first_step: bool | None,
    tables: Any | None,
    turn_off_micro: bool,
    cloud_condensation: bool,
    semi_sedi_flag: bool,
    max_sedimentation_substeps: int | None,
) -> TorchSegmentP4Result:
    if isinstance(state, TorchSegmentP4Result):
        return state
    return run_p4(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
    )


def _segment_p5_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentP5Result,
    *,
    dt: float | None,
    is_first_step: bool | None,
    tables: Any | None,
    turn_off_micro: bool,
    cloud_condensation: bool,
    semi_sedi_flag: bool,
    max_sedimentation_substeps: int | None,
) -> TorchSegmentP5Result:
    if isinstance(state, TorchSegmentP5Result):
        return state
    return run_p5(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
    )


def _segment_q_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentQResult,
    *,
    dt: float | None,
    is_first_step: bool | None,
    tables: Any | None,
    turn_off_micro: bool,
    cloud_condensation: bool,
    semi_sedi_flag: bool,
    max_sedimentation_substeps: int | None,
    compact_base: bool = False,
) -> TorchSegmentQResult:
    if isinstance(state, TorchSegmentQResult):
        return state
    return run_q(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
        compact_base=compact_base,
    )


def _segment_r1_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentR1Result,
    *,
    dt: float | None,
    is_first_step: bool | None,
    tables: Any | None,
    turn_off_micro: bool,
    cloud_condensation: bool,
    semi_sedi_flag: bool,
    max_sedimentation_substeps: int | None,
    max_hail_diameter_flag: bool = True,
) -> TorchSegmentR1Result:
    if isinstance(state, TorchSegmentR1Result):
        return state
    return run_r1(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        refl10cm_from_melting=False,
        max_hail_diameter_flag=max_hail_diameter_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
    )


def _segment_r2_result(
    state: Mapping[str, torch.Tensor] | TorchSegmentR2Result,
    *,
    dt: float | None,
    is_first_step: bool | None,
    tables: Any | None,
    turn_off_micro: bool,
    cloud_condensation: bool,
    semi_sedi_flag: bool,
    refl10cm_from_melting: bool | torch.Tensor,
    max_sedimentation_substeps: int | None,
    max_hail_diameter_flag: bool = True,
) -> TorchSegmentR2Result:
    if isinstance(state, TorchSegmentR2Result):
        return state
    return run_r2(
        state,
        dt=dt,
        is_first_step=is_first_step,
        tables=tables,
        turn_off_micro=turn_off_micro,
        cloud_condensation=cloud_condensation,
        semi_sedi_flag=semi_sedi_flag,
        refl10cm_from_melting=refl10cm_from_melting,
        max_hail_diameter_flag=max_hail_diameter_flag,
        max_sedimentation_substeps=max_sedimentation_substeps,
    )


def _segment_c_workspace_from_o(workspace: TorchSegmentOWorkspace) -> TorchSegmentCWorkspace:
    n_workspace = workspace.base
    if n_workspace is None:
        raise ValueError("Segment O upstream workspace chain has been compacted")
    m_workspace = n_workspace.base
    l_workspace = m_workspace.base
    k_workspace = l_workspace.base
    j_workspace = k_workspace.base
    i_workspace = j_workspace.base
    h_workspace = i_workspace.base
    g_workspace = h_workspace.base
    f_workspace = g_workspace.base
    e_workspace = f_workspace.base
    d_workspace = e_workspace.base
    return d_workspace.base


def _segment_o_workspace_from_p5(workspace: TorchSegmentP5Workspace) -> TorchSegmentOWorkspace:
    if workspace.base is None:
        raise ValueError("Segment P5 upstream workspace chain has been compacted")
    if isinstance(workspace.base, TorchSegmentP4Workspace):
        return workspace.base.base.base
    return workspace.base.base.base


def _segment_d_workspace_from_o(workspace: TorchSegmentOWorkspace) -> TorchSegmentDWorkspace:
    n_workspace = workspace.base
    if n_workspace is None:
        raise ValueError("Segment O upstream workspace chain has been compacted")
    m_workspace = n_workspace.base
    l_workspace = m_workspace.base
    k_workspace = l_workspace.base
    j_workspace = k_workspace.base
    i_workspace = j_workspace.base
    h_workspace = i_workspace.base
    g_workspace = h_workspace.base
    f_workspace = g_workspace.base
    e_workspace = f_workspace.base
    return e_workspace.base


def _segment_g_workspace_from_p5_workspace(workspace: TorchSegmentP5Workspace) -> TorchSegmentGWorkspace:
    o_workspace = _segment_o_workspace_from_p5(workspace)
    return _segment_g_workspace_from_o_workspace(o_workspace)


def _segment_g_workspace_from_o_workspace(workspace: TorchSegmentOWorkspace) -> TorchSegmentGWorkspace:
    n_workspace = workspace.base
    if n_workspace is None:
        raise ValueError("Segment O upstream workspace chain has been compacted")
    m_workspace = n_workspace.base
    l_workspace = m_workspace.base
    k_workspace = l_workspace.base
    j_workspace = k_workspace.base
    i_workspace = j_workspace.base
    h_workspace = i_workspace.base
    return h_workspace.base


def _segment_g_workspace_from_q(workspace: TorchSegmentQWorkspace) -> TorchSegmentGWorkspace:
    if workspace.base is None:
        raise ValueError("Segment Q upstream workspace chain has been compacted")
    return _segment_g_workspace_from_p5_workspace(workspace.base)


def _active_columns_from_p5_workspace(workspace: TorchSegmentP5Workspace) -> torch.Tensor:
    if workspace.base is None:
        if workspace.q_active_columns is None:
            raise ValueError("Compact Segment P5 workspace is missing q_active_columns")
        return workspace.q_active_columns
    g_workspace = _segment_g_workspace_from_p5_workspace(workspace)
    return _active_columns_from_g_workspace(g_workspace)


def _active_columns_from_g_workspace(g_workspace: TorchSegmentGWorkspace) -> torch.Tensor:
    f_workspace = g_workspace.base
    e_workspace = f_workspace.base
    hydrometeor_active = (
        f_workspace.l_qc
        | e_workspace.l_qr
        | e_workspace.l_qi
        | e_workspace.l_qs
        | g_workspace.l_qg
    )
    supersaturated = g_workspace.thermo["ssati"] > 0.0
    return torch.any(hydrometeor_active, dim=1) | torch.any(supersaturated, dim=1)


def _active_columns_from_o_workspace(workspace: TorchSegmentOWorkspace) -> torch.Tensor:
    return workspace.active_columns


def _active_columns_from_q(workspace: TorchSegmentQWorkspace) -> torch.Tensor:
    return workspace.active_columns


def _state_mapping(state: Mapping[str, torch.Tensor] | Any) -> Mapping[str, torch.Tensor]:
    if isinstance(state, Mapping):
        return state
    result_state = getattr(state, "state", None)
    if isinstance(result_state, Mapping):
        return result_state
    raise TypeError("Torch-native TEMPO segment requires a state mapping or prior result")


def _require_checkpoint_fields(state: Mapping[str, torch.Tensor], segment_id: str) -> None:
    missing = [name for name in CHECKPOINT_FIELDS if name not in state]
    if missing:
        raise KeyError(f"Segment {segment_id} missing required fields: {missing}")


def _validate_state(state: Mapping[str, torch.Tensor], fields: tuple[str, ...], segment_id: str) -> tuple[int, int]:
    shape: tuple[int, int] | None = None
    dtype: torch.dtype | None = None
    for name in fields:
        values = state[name]
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"Segment {segment_id} field {name} must be a torch.Tensor")
        if values.ndim != 2:
            raise ValueError(f"Segment {segment_id} field {name} must have shape [ncol,nz]")
        dtype = values.dtype if dtype is None else dtype
        if values.dtype not in WORKING_DTYPES or values.dtype != dtype:
            raise TypeError(f"Segment {segment_id} field {name} must be {dtype} (float32 or float64), got {values.dtype}")
        current_shape = (values.shape[0], values.shape[1])
        if shape is None:
            shape = current_shape
        elif current_shape != shape:
            raise ValueError(f"Segment {segment_id} field {name} shape {current_shape} differs from {shape}")
    if shape is None:
        raise ValueError(f"Segment {segment_id} requires at least one field")
    return int(shape[0]), int(shape[1])


def _optional_profile(
    segment_id: str,
    label: str,
    explicit: torch.Tensor | None,
    state: Mapping[str, torch.Tensor],
    aliases: tuple[str, ...],
    shape: tuple[int, int],
) -> torch.Tensor | None:
    values, _ = _optional_profile_with_key(segment_id, label, explicit, state, aliases, shape)
    return values


def _optional_profile_with_key(
    segment_id: str,
    label: str,
    explicit: torch.Tensor | None,
    state: Mapping[str, torch.Tensor],
    aliases: tuple[str, ...],
    shape: tuple[int, int],
) -> tuple[torch.Tensor | None, str | None]:
    values = explicit
    key = None
    if values is None:
        for alias in aliases:
            if alias in state:
                values = state[alias]
                key = alias
                break
    if values is None:
        return None, None
    if values.ndim != 2:
        raise ValueError(f"Segment {segment_id} {label} must have shape [ncol,nz]")
    if values.dtype not in WORKING_DTYPES:
        raise TypeError(f"Segment {segment_id} {label} must be torch.float32 or torch.float64, got {values.dtype}")
    if tuple(values.shape) != shape:
        raise ValueError(f"Segment {segment_id} {label} shape {tuple(values.shape)} differs from {shape}")
    return values, key


def _optional_input_profile_tensor(
    state: Mapping[str, torch.Tensor],
    aliases: tuple[str, ...],
    fallback: torch.Tensor | None,
) -> torch.Tensor | None:
    for alias in aliases:
        if alias in state:
            return state[alias]
    return fallback


def _final_optional_aerosol_tensor(
    *,
    state: Mapping[str, torch.Tensor],
    aliases: tuple[str, ...],
    base: torch.Tensor | None,
    tendency: torch.Tensor,
    default: np.float64,
    dt: float,
) -> torch.Tensor | None:
    values: torch.Tensor | None = None
    for alias in aliases:
        if alias in state:
            values = state[alias]
            break
    if values is None:
        values = base
    if values is None:
        return None
    return torch.clamp(values + tendency * _scalar(dt, tendency), min=float(default), max=float(constants.AERO_MAX))


def _final_snow_moments_tensor(
    *,
    rs: torch.Tensor,
    temp: torch.Tensor,
    l_qs: torch.Tensor,
    template: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    _ = template
    smo2, smob, smoc, smoz = _run_tensor_compile_island(
        "compute_final_snow_moments",
        _compute_final_snow_moments_kernel,
        rs,
        rs,
        temp,
        l_qs,
        _tensor(_STATIC_SNOW_A, rs),
        _tensor(_STATIC_SNOW_B, rs),
    )
    zero = torch.zeros_like(rs)
    return {
        "smo0": zero,
        "smo1": zero,
        "smo2": smo2,
        "smob": smob,
        "smoc": smoc,
        "smoe": zero,
        "smof": zero,
        "smog": zero,
        "smoz": smoz,
        "ns": zero,
    }


def _compute_final_snow_moments_kernel(
    rs: torch.Tensor,
    temp: torch.Tensor,
    l_qs: torch.Tensor,
    snow_a: torch.Tensor,
    snow_b: torch.Tensor,
    _t0: float = float(constants.T0),
    _oams: float = float(constants.OAMS),
    _bm_s: float = float(constants.BM_S),
    _cse0: float = float(_SCALAR_CSE[0]),
    _cse2: float = float(_SCALAR_CSE[2]),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Masked columns (no snow) compute on a placeholder mixing ratio: their results are discarded by the final where,
    # but 0 ** b there would send 0 * inf = NaN into the reverse pass. Kept columns are unchanged bitwise.
    rs = torch.where(l_qs, rs, torch.ones_like(rs))
    tc = torch.minimum(temp.new_full((), -0.1), temp - _t0)
    smob = rs * _oams
    if abs(_bm_s - 2.0) < 0.001:
        a_value = working_power(10.0, _snow_poly(snow_a, tc, _bm_s))
        b_value = _snow_poly(snow_b, tc, _bm_s)
        smo2 = working_power(smob / a_value, 1.0 / b_value)
    else:
        smo2 = smob
    smoc = working_power(10.0, _snow_poly(snow_a, tc, _cse0)) * working_power(
        smo2, _snow_poly(snow_b, tc, _cse0)
    )
    smoz = working_power(10.0, _snow_poly(snow_a, tc, _cse2)) * working_power(
        smo2, _snow_poly(snow_b, tc, _cse2)
    )
    zero = torch.zeros_like(rs)
    return (
        torch.where(l_qs, smo2, zero),
        torch.where(l_qs, smob, zero),
        torch.where(l_qs, smoc, zero),
        torch.where(l_qs, smoz, zero),
    )


def _init_aerosol_profile(dz: torch.Tensor, *, n0: np.float64, n1: np.float64) -> torch.Tensor:
    if dz.ndim != 2:
        raise ValueError("Segment D dz1d must have shape [ncol,nz]")
    if dz.dtype not in WORKING_DTYPES:
        raise TypeError(f"Segment D dz1d must be float32 or float64, got {dz.dtype}")
    if dz.shape[1] < 2:
        raise ValueError("Segment D aerosol initialization requires at least two vertical levels")
    n0_value = float(n0)
    n1_value = float(n1)
    return _run_tensor_compile_island(
        "init_aerosol_profile",
        _init_aerosol_profile_kernel,
        dz,
        dz,
        n0_value,
        n1_value,
        -math.log(n1_value / n0_value),
    )


def _init_aerosol_profile_kernel(
    dz: torch.Tensor,
    n0: float,
    n1: float,
    log_ratio: float,
) -> torch.Tensor:
    hgt = _prepend_zero_column_no_check(torch.cumsum(dz[:, 1:], dim=1))
    h_01 = torch.where(
        hgt[:, 0] <= 1000.0,
        0.8,
        torch.where(
            hgt[:, 0] >= 2500.0,
            0.01,
            0.8 * torch.cos(hgt[:, 0] * 0.001 - 1.0),
        ),
    )
    decay = log_ratio / h_01
    profile = n1 + n0 * torch.exp(-((hgt - hgt[:, :1]) / 1000.0) * decay[:, None])
    first = n1 + n0 * torch.exp(-((hgt[:, 1] - hgt[:, 0]) / 1000.0) * decay)
    result = profile.new_empty(profile.shape)
    result[:, 0] = first
    result[:, 1:] = profile[:, 1:]
    return result


def _snow_poly(coefficients: torch.Tensor, tc: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    return (
        coefficients[0]
        + coefficients[1] * tc
        + coefficients[2] * shape
        + coefficients[3] * tc * shape
        + coefficients[4] * tc * tc
        + coefficients[5] * shape * shape
        + coefficients[6] * tc * tc * shape
        + coefficients[7] * tc * shape * shape
        + coefficients[8] * tc * tc * tc
        + coefficients[9] * shape * shape * shape
    )


def _snow_poly_no_shape(coefficients: torch.Tensor, tc: torch.Tensor) -> torch.Tensor:
    return coefficients[0] + coefficients[1] * tc + coefficients[4] * tc * tc + coefficients[8] * tc * tc * tc


def _take(values: np.ndarray, index_1based: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    table = _table_tensor(values, reference)
    return table[index_1based.to(torch.int64) - 1]


def _cloud_limit_table_bundle(nu_c: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cce1 = _take(_STATIC_CLOUD_CCE[1], nu_c, reference)
    ccg0 = _take(_STATIC_CLOUD_CCG[0], nu_c, reference)
    ocg2 = _take(_STATIC_CLOUD_OCG2, nu_c, reference)
    return cce1, ccg0, ocg2


def _cloud_diameter_table_bundle(nu_c: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ccg2 = _take(_STATIC_CLOUD_CCG[2], nu_c, reference)
    ocg2 = _take(_STATIC_CLOUD_OCG2, nu_c, reference)
    return ccg2, ocg2


def _cloud_fallspeed_table_bundle(
    nu_c: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ccg4 = _take(_STATIC_CLOUD_CCG[4], nu_c, reference)
    ocg2 = _take(_STATIC_CLOUD_OCG2, nu_c, reference)
    ccg3 = _take(_STATIC_CLOUD_CCG[3], nu_c, reference)
    ocg1 = _take(_STATIC_CLOUD_OCG1, nu_c, reference)
    return ccg4, ocg2, ccg3, ocg1


def _cloud_lambda_table_bundle(nu_c: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ccg1 = _take(_STATIC_CLOUD_CCG[1], nu_c, reference)
    ocg1 = _take(_STATIC_CLOUD_OCG1, nu_c, reference)
    return ccg1, ocg1


def _graupel_sd_table_bundle(index_1based: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_av_cgg10 = torch.sqrt(_take(_STATIC_GRAUPEL_AV_G, index_1based, reference)) * _take(
        _STATIC_GRAUPEL_CGG[10],
        index_1based,
        reference,
    )
    cge10 = _take(_STATIC_GRAUPEL_CGE[10], index_1based, reference)
    return sqrt_av_cgg10, cge10


def _graupel_sd_density_table_bundle(
    index_1based: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sqrt_av_cgg10, cge10 = _graupel_sd_table_bundle(index_1based, reference)
    rho_g = _take(_STATIC_GRAUPEL_RHO_G, index_1based, reference)
    return sqrt_av_cgg10, cge10, rho_g


def _graupel_riming_table_bundle(
    index_1based: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    av_g = _take(_STATIC_GRAUPEL_AV_G, index_1based, reference)
    bv_g = _take(_STATIC_GRAUPEL_BV_G, index_1based, reference)
    cgg5 = _take(_STATIC_GRAUPEL_CGG[5], index_1based, reference)
    cgg8 = _take(_STATIC_GRAUPEL_CGG[8], index_1based, reference)
    cge8 = _take(_STATIC_GRAUPEL_CGE[8], index_1based, reference)
    return av_g, bv_g, cgg5, cgg8, cge8


def _graupel_aerosol_table_bundle(
    index_1based: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    av_g = _take(_STATIC_GRAUPEL_AV_G, index_1based, reference)
    cgg8 = _take(_STATIC_GRAUPEL_CGG[8], index_1based, reference)
    cge8 = _take(_STATIC_GRAUPEL_CGE[8], index_1based, reference)
    return av_g, cgg8, cge8


def _gather_level(values: torch.Tensor, index_0based: torch.Tensor) -> torch.Tensor:
    clamped = torch.clamp(index_0based.to(torch.int64), 0, values.shape[1] - 1)
    return values.gather(1, clamped[:, None]).squeeze(1)


def _gather_columns(values: torch.Tensor, index_0based: torch.Tensor) -> torch.Tensor:
    clamped = torch.clamp(index_0based.to(torch.int64), 0, values.shape[1] - 1)
    return torch.gather(values, 1, clamped)


def _gather_columns_valid(values: torch.Tensor, index_0based: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, 1, index_0based)


def _append_column(body: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    tail_column = tail[:, None] if tail.ndim == 1 else tail
    if tail_column.shape != (body.shape[0], 1):
        raise ValueError("tail column must have shape [ncol] or [ncol,1]")
    result = body.new_empty((body.shape[0], body.shape[1] + 1))
    result[:, :-1] = body
    result[:, -1:] = tail_column
    return result


def _prepend_column(head: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
    head_column = head[:, None] if head.ndim == 1 else head
    if head_column.shape != (body.shape[0], 1):
        raise ValueError("head column must have shape [ncol] or [ncol,1]")
    result = body.new_empty((body.shape[0], body.shape[1] + 1))
    result[:, :1] = head_column
    result[:, 1:] = body
    return result


def _prepend_zero_column(body: torch.Tensor) -> torch.Tensor:
    return _prepend_column(torch.zeros((body.shape[0], 1), **_factory(body)), body)


def _suffix_fill_from_active(formula: torch.Tensor, tail: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    ncol, nz = formula.shape
    levels = _int_arange(0, nz, formula)[None, :].expand(ncol, nz)
    source_index = torch.where(active, levels, nz)
    fill_index = torch.flip(torch.cummin(torch.flip(source_index, dims=(1,)), dim=1).values, dims=(1,))
    extended = _append_column(formula, tail)
    return torch.gather(extended, 1, fill_index)


def _hail_bins_tensor(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bins, widths = _STATIC_HAIL_BINS
    return _tensor(bins, reference), _tensor(widths, reference)


def _radar_bins_tensor(numbins: int, lowbin: float, highbin: float, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    key = (int(numbins), float(lowbin), float(highbin))
    bins, widths = _STATIC_RADAR_BINS[key]
    return _tensor(bins, reference), _tensor(widths, reference)


def _simpson_weights_tensor(numbins: int, reference: torch.Tensor) -> torch.Tensor:
    weights = _STATIC_SIMPSON_WEIGHTS[int(numbins)]
    return _tensor(weights, reference)


def _driver_diagnostic_copyback_tensor(
    workspace: TorchSegmentR2Workspace,
    re_cloud: torch.Tensor | None,
    re_ice: torch.Tensor | None,
    re_snow: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    r1_workspace = workspace.base
    return {
        "rain_precip": r1_workspace.rain_precip,
        "ice_liquid_equiv_precip": r1_workspace.ice_liquid_equiv_precip,
        "snow_liquid_equiv_precip": r1_workspace.snow_liquid_equiv_precip,
        "graupel_liquid_equiv_precip": r1_workspace.graupel_liquid_equiv_precip,
        "frozen_fraction": r1_workspace.frozen_fraction,
        "frz_rain_precip": r1_workspace.frz_rain_precip,
        "cloud_number_mixing_ratio": r1_workspace.cloud_number_mixing_ratio,
        "rain_med_vol_diam": r1_workspace.rain_med_vol_diam,
        "graupel_med_vol_diam": r1_workspace.graupel_med_vol_diam,
        **({} if re_cloud is None else {"re_cloud": re_cloud, "re_ice": re_ice, "re_snow": re_snow}),
        **({} if workspace.refl10cm is None else {"refl10cm": workspace.refl10cm}),
        "max_hail_diameter_sfc": r1_workspace.max_hail_diameter[:, 0],
        "max_hail_diameter_column": torch.max(r1_workspace.max_hail_diameter, dim=1).values,
    }


def _copy_tensor(values: torch.Tensor) -> torch.Tensor:
    return values.clone()


def _tensor_mapping(values: Mapping[str, torch.Tensor], *, copy: bool) -> dict[str, torch.Tensor]:
    if copy:
        return {name: _copy_tensor(tensor) for name, tensor in values.items()}
    return dict(values)


def _factory(reference: torch.Tensor) -> dict[str, Any]:
    return {"dtype": reference.dtype, "device": reference.device}


def _is_fake_or_meta_tensor(reference: torch.Tensor) -> bool:
    return reference.device.type == "meta" or type(reference).__name__ == "FakeTensor"


def _int_arange(start: int, stop: int, reference: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling() or _is_fake_or_meta_tensor(reference):
        return torch.arange(start, stop, dtype=torch.int64, device=reference.device)
    key = (
        int(start),
        int(stop),
        str(reference.device),
        torch.is_inference_mode_enabled(),
    )
    cached = _TORCH_INT_ARANGE_CACHE.get(key)
    if cached is None:
        cached = torch.arange(start, stop, dtype=torch.int64, device=reference.device)
        _TORCH_INT_ARANGE_CACHE[key] = cached
    return cached


def _scalar(value: Any, reference: torch.Tensor) -> torch.Tensor | float:
    scalar_value = float(value)
    if torch.compiler.is_compiling() or _is_fake_or_meta_tensor(reference):
        return torch.scalar_tensor(scalar_value, dtype=reference.dtype, device=reference.device)
    key = (
        scalar_value,
        str(reference.device),
        reference.dtype,
        torch.is_inference_mode_enabled(),
    )
    cached = _TORCH_SCALAR_CACHE.get(key)
    if cached is None:
        cached = torch.as_tensor(scalar_value, **_factory(reference))
        _TORCH_SCALAR_CACHE[key] = cached
    return cached


def _int_scalar(value: int, reference: torch.Tensor, *, dtype: torch.dtype = torch.int64) -> torch.Tensor | int:
    scalar_value = int(value)
    if torch.compiler.is_compiling() or _is_fake_or_meta_tensor(reference):
        return torch.scalar_tensor(scalar_value, dtype=dtype, device=reference.device)
    key = (
        scalar_value,
        str(reference.device),
        dtype,
        torch.is_inference_mode_enabled(),
    )
    cached = _TORCH_INT_SCALAR_CACHE.get(key)
    if cached is None:
        cached = torch.as_tensor(scalar_value, dtype=dtype, device=reference.device)
        _TORCH_INT_SCALAR_CACHE[key] = cached
    return cached


def _limit_like(value: Any, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(dtype=reference.dtype, device=reference.device)
        if tensor.ndim == 1:
            return tensor[:, None]
        return tensor
    return _scalar(value, reference)


def _tensor(values: Any, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return _static_tensor(values, reference)
    if isinstance(values, np.ndarray):
        # Dynamo cannot trace id() of an untracked array; a traced table is a graph constant.
        if torch.compiler.is_compiling():
            return _table_tensor(values, reference)
        if id(values) in _STATIC_NDARRAY_IDS and not _is_fake_or_meta_tensor(reference):
            fast_key = (
                id(values),
                str(reference.device),
                reference.dtype,
                torch.is_inference_mode_enabled(),
            )
            cached = _TORCH_STATIC_NDARRAY_FAST_CACHE.get(fast_key)
            if cached is None:
                cached = _table_tensor(values, reference)
                _TORCH_STATIC_NDARRAY_FAST_CACHE[fast_key] = cached
            return cached
        return _table_tensor(values, reference)
    if isinstance(values, (float, np.floating)):
        # Scalar constants reuse the cached 0-dim tensor: no per-call host copy.
        return _scalar(values, reference)
    return torch.as_tensor(values, **_factory(reference))


def _normalize_compile_literal_defaults(value: Any) -> Any:
    scalar = value
    if isinstance(value, np.ndarray) and value.shape == ():
        scalar = value[()]
    if isinstance(scalar, np.bool_):
        return bool(scalar)
    if isinstance(scalar, np.integer):
        return int(scalar)
    if isinstance(scalar, np.floating):
        return float(scalar)
    if isinstance(scalar, np.complexfloating):
        return complex(scalar)
    if isinstance(value, np.generic):
        return scalar
    if isinstance(value, tuple):
        return tuple(_normalize_compile_literal_defaults(item) for item in value)
    return value


def _normalize_numpy_scalar_function_defaults() -> None:
    for value in tuple(globals().values()):
        if getattr(value, "__module__", None) != __name__:
            continue
        defaults = getattr(value, "__defaults__", None)
        if defaults:
            value.__defaults__ = tuple(_normalize_compile_literal_defaults(item) for item in defaults)
        kwdefaults = getattr(value, "__kwdefaults__", None)
        if kwdefaults:
            value.__kwdefaults__ = {
                key: _normalize_compile_literal_defaults(item)
                for key, item in kwdefaults.items()
            }


_normalize_numpy_scalar_function_defaults()


__all__ = [
    "TorchPrefixResult",
    "TorchPrefixSnapshot",
    "TorchSegmentBResult",
    "TorchSegmentCResult",
    "TorchSegmentCWorkspace",
    "TorchSegmentDResult",
    "TorchSegmentDWorkspace",
    "TorchSegmentEResult",
    "TorchSegmentEWorkspace",
    "TorchSegmentFResult",
    "TorchSegmentFWorkspace",
    "TorchSegmentGResult",
    "TorchSegmentGWorkspace",
    "TorchSegmentHResult",
    "TorchSegmentHWorkspace",
    "TorchSegmentIResult",
    "TorchSegmentIWorkspace",
    "TorchSegmentJResult",
    "TorchSegmentJWorkspace",
    "TorchSegmentKResult",
    "TorchSegmentKWorkspace",
    "TorchSegmentLResult",
    "TorchSegmentLWorkspace",
    "TorchSegmentMResult",
    "TorchSegmentMWorkspace",
    "TorchSegmentNResult",
    "TorchSegmentNWorkspace",
    "TorchSegmentOResult",
    "TorchSegmentOWorkspace",
    "TorchSegmentP1Result",
    "TorchSegmentP1Workspace",
    "TorchSegmentP2Result",
    "TorchSegmentP2Workspace",
    "TorchSegmentP3Result",
    "TorchSegmentP3Workspace",
    "TorchSegmentP4Result",
    "TorchSegmentP4Workspace",
    "TorchSegmentP5Result",
    "TorchSegmentP5Workspace",
    "TorchSegmentQResult",
    "TorchSegmentQWorkspace",
    "TorchSegmentR1Result",
    "TorchSegmentR1Workspace",
    "TorchSegmentR2Result",
    "TorchSegmentR2Workspace",
    "TorchSegmentSResult",
    "TorchSegmentSWorkspace",
    "DEFAULT_PRODUCTION_SEMI_SEDIMENTATION_SUBSTEPS",
    "calc_rsif",
    "calc_rslf",
    "compact_s_result",
    "compact_s_prefix_result",
    "run_b",
    "run_c",
    "run_d",
    "run_e",
    "run_f",
    "run_g",
    "run_h",
    "run_i",
    "run_j",
    "run_k",
    "run_l",
    "run_m",
    "run_n",
    "run_o",
    "run_p1",
    "run_p2",
    "run_p3",
    "run_p4",
    "run_p5",
    "run_q",
    "run_r1",
    "run_r2",
    "run_s",
    "run_prefix_b_to_g",
    "run_prefix_b_to_h",
    "run_prefix_b_to_i",
    "run_prefix_b_to_j",
    "run_prefix_b_to_k",
    "run_prefix_b_to_l",
    "run_prefix_b_to_m",
    "run_prefix_b_to_n",
    "run_prefix_b_to_o",
    "run_prefix_b_to_p1",
    "run_prefix_b_to_p2",
    "run_prefix_b_to_p3",
    "run_prefix_b_to_p4",
    "run_prefix_b_to_p5",
    "run_prefix_b_to_q",
    "run_prefix_b_to_r1",
    "run_prefix_b_to_r2",
    "run_prefix_b_to_s",
    "run_tempo_native",
    "thermo_vars",
    "to_torch_state",
]
