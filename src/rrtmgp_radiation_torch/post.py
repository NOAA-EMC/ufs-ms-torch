# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Heating rates from the radiative fluxes (``mo_heating_rates``)."""

from __future__ import annotations

import torch


GRAV = 9.80665
CP_DRY = 1004.64


TensorOrString = torch.Tensor | str


def _to_real(value: TensorOrString, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64 (the working dtype), got {value.dtype}")
    return value


def compute_heating_rate(
    flux_up: torch.Tensor,
    flux_down: torch.Tensor,
    p_lev: torch.Tensor,
    *,
    grav: float = GRAV,
    cp_dry: float = CP_DRY,
) -> torch.Tensor:
    """Match RTE/RRTMGP `mo_heating_rates::compute_heating_rate_general`.

    Fortran `(ncol,nlev)` arrays are stored as `(nlev,ncol)`, so the vertical
    interface axis is axis 0.
    """

    up = _to_real(flux_up, name="flux_up")
    down = _to_real(flux_down, name="flux_down")
    pressure = _to_real(p_lev, name="p_lev")
    if up.shape != down.shape or up.shape != pressure.shape:
        raise ValueError(
            "flux_up, flux_down, and p_lev must have identical checkpoint-shaped "
            f"interfaces, got {up.shape}, {down.shape}, {pressure.shape}"
        )
    if up.ndim != 2 or up.shape[0] < 2:
        raise ValueError(f"interface fields must have shape [nlev, ncol], got {up.shape}")
    return (
        (up[1:, :] - up[:-1, :] - down[1:, :] + down[:-1, :])
        * grav
        / (cp_dry * (pressure[1:, :] - pressure[:-1, :]))
    )


