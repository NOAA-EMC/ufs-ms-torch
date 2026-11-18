# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The Hogan cloud decorrelation length of ``GFS_rrtmgp_cloud_overlap``."""

from __future__ import annotations

import torch


TensorOrString = torch.Tensor | str

DEFAULT_DECORRELATION_LENGTH_KM = 1.911111111111111


def _to_real(value: TensorOrString, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64 (the working dtype), got {value.dtype}")
    return value


def compute_hogan_decorrelation_length(xlat: torch.Tensor, con_pi: torch.Tensor) -> torch.Tensor:
    """Match Hogan/Shonk latitude-dependent cloud decorrelation length."""

    lat = _to_real(xlat, name="xlat")
    pi = _to_real(con_pi, name="con_pi").reshape(-1)[0].to(device=lat.device)
    return torch.maximum(
        torch.full((), 0.6, dtype=lat.dtype, device=lat.device),
        2.78 - 4.6 * torch.abs(lat / pi),
    )


__all__ = [
    "DEFAULT_DECORRELATION_LENGTH_KM",
    "compute_hogan_decorrelation_length",
]
