# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Small tensor helpers for device-pure, autodiff-safe arithmetic."""

from __future__ import annotations

import torch


def static_tensor(values, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """Device copy of a static host constant, built once per values, dtype and device.

    A per-call host constant costs a pageable copy and a stream sync on CUDA and
    blocks graph capture. Callers must not modify the result in place.
    """
    if torch.compiler.is_compiling():  # traced as a graph constant; no cache lookup
        return torch.tensor(values, dtype=dtype, device=device)
    return _static_tensor(_freeze(values), dtype, torch.device(device), torch.is_inference_mode_enabled())


def _freeze(values):
    return tuple(_freeze(v) for v in values) if isinstance(values, (list, tuple)) else values


def _exact(values):
    """Cache key that keeps -0.0 apart from 0.0 and matches NaN to itself."""
    if isinstance(values, tuple):
        return tuple(_exact(v) for v in values)
    return values.hex() if isinstance(values, float) else values


_STATIC_TENSORS: dict = {}


def _static_tensor(values, dtype, device, inference):
    key = (_exact(values), dtype, device, inference)
    cached = _STATIC_TENSORS.get(key)
    if cached is None:
        cached = _STATIC_TENSORS[key] = torch.tensor(values, dtype=dtype, device=device)
    return cached


def safe_sqrt_nonnegative(values: torch.Tensor, *, floor: float = 1.0e-30) -> torch.Tensor:
    """Square root with a zero-gradient inactive side for masked formulas."""

    active = values > floor
    safe_values = torch.where(active, values, 1.0)
    return torch.where(active, torch.sqrt(safe_values), 0.0)


def safe_positive_denominator(values: torch.Tensor, *, floor: float = 1.0e-30) -> torch.Tensor:
    """Keep masked divisions finite without changing positive physical inputs."""

    return torch.where(values > floor, values, floor)


def safe_signed_denominator(values: torch.Tensor, *, floor: float = 1.0e-30) -> torch.Tensor:
    """Keep divisions finite while preserving denominator sign away from zero."""

    return torch.where(
        torch.abs(values) > floor,
        values,
        torch.where(values < 0.0, values.new_full((), -floor), values.new_full((), floor)),
    )
