# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Regional ``torch.compile`` dispatch for the SA-TKE-EDMF tensor islands; outside a region they run eagerly."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

import torch
from ufs_physics_common.compile_policy import dispatcher, island, orchestrating

TensorResult = TypeVar("TensorResult")

COMPILE_CATEGORIES = frozenset(
    {
        "d_pbl_scan",
        "e_surface_similarity",
        "f_cloud_seed",
        "g_plume_recurrence",
        "i_downdraft",
        "k_mixing_length",
        "l_diffusivity",
        "m_tendency_block",
        "m_tke_recurrence",
        "m_plume_recurrence",
        "n_tvd_limiter",
        "tridiagonal",
        "interstitial",
    }
)
_regional = False


@contextmanager
def regional_compile():
    """Keep the coupled PBL driver eager and compile its tensor blocks exactly."""
    global _regional
    previous, _regional = _regional, True
    try:
        yield
    finally:
        _regional = previous


_BORROWED_ISLANDS = frozenset({"cloud", "advance_te_substeps_base"})


@dispatcher
def run_compile_island(
    category: str,
    name: str,
    fn: Callable[..., TensorResult],
    *args: Any,
    **kwargs: Any,
) -> TensorResult:
    if category not in COMPILE_CATEGORIES:
        raise ValueError(f"unknown SA-TKE compile category {category!r}")
    # Nested islands compose their pure functions when an outer region traces; an orchestration region calls the
    # island's own region (``compile_policy.island``).
    if _regional and (orchestrating() or not torch.compiler.is_compiling()):
        # These are orchestration over separately compiled blocks, not tensor regions.
        if category == "g_plume_recurrence" and name in {"plume_block", "tracer_recurrence"}:
            return fn(*args, **kwargs)
        if category == "tridiagonal" and name in {"heat_tracer_recovery", "heat_tracer_thomas"}:
            return fn(*args, **kwargs)
        if category == "k_mixing_length" and name in {"seed_block", "parcel_chunk"}:
            return fn(*args, **kwargs)
        # These islands write no input, so without gradients they read their strided inputs in place ("cloud" reads
        # the qv and ql slots of qcko, the TKE substeps read te[:, :km-1]) instead of copying each whole view at the
        # region entry; with gradients the copy keeps the layout the derivative graph expects.
        borrowed = name in _BORROWED_ISLANDS and not torch.is_grad_enabled()
        return island(fn, args, kwargs, normalize_layouts=name != "assemble_levels" and not borrowed)
    return fn(*args, **kwargs)
