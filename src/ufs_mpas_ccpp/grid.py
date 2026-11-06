# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Selected R4 grid allocation and MPAS-to-CCPP geometry mapping."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
from mpas_pytorch.constants import UFS_CONSTANTS

from .precision import WORKING_DTYPES


@dataclass(frozen=True)
class Grid:
    """GFS_grid_type pointers allocated in R4, including ozone lifecycle carries."""

    xlon: torch.Tensor
    xlat: torch.Tensor
    xlat_d: torch.Tensor
    xlon_d: torch.Tensor
    sinlat: torch.Tensor
    coslat: torch.Tensor
    area: torch.Tensor
    dx: torch.Tensor
    ddy_o3: torch.Tensor
    jindx1_o3: torch.Tensor
    jindx2_o3: torch.Tensor

    @classmethod
    def allocate_r4(cls, *, ncol: int, device: torch.device | str, dtype: torch.dtype = torch.float64) -> Grid:
        if type(ncol) is not int or ncol <= 0:
            raise ValueError("Grid allocation requires a positive integer ncol")
        return cls(
            **{
                field.name: torch.zeros(
                    ncol,
                    dtype=torch.int32
                    if field.name in ("jindx1_o3", "jindx2_o3")
                    else dtype,
                    device=device,
                )
                for field in fields(cls)
            }
        )


@dataclass(frozen=True)
class GridProjection:
    grid: Grid
    length_scale: torch.Tensor


def project_grid(
    target: Grid,
    *,
    n_cells_solve: int,
    lat_cell: torch.Tensor,
    lon_cell: torch.Tensor,
    area_cell: torch.Tensor,
    mesh_density: torch.Tensor,
    nominal_min_dc: torch.Tensor,
    config_len_disp: float,
) -> GridProjection:
    """Map the eight source-written fields; retain interpolation state and tails."""
    n = n_cells_solve
    if type(n) is not int or n <= 0:
        raise ValueError("grid owned extent must be a positive integer")
    if type(config_len_disp) not in (int, float):
        raise TypeError("configured dispersion length must be static")
    device, real = lat_cell.device, lat_cell.dtype
    for value in (lat_cell, lon_cell, area_cell, mesh_density):
        if (
            value.ndim != 1
            or value.shape[0] < n
            or real not in WORKING_DTYPES
            or value.dtype != real
            or value.device != device
        ):
            raise ValueError("invalid native grid input shape/dtype/device")
        torch._assert_async(
            torch.isfinite(value[:n]).all(), "nonfinite native grid input"
        )
    if (
        nominal_min_dc.numel() != 1
        or nominal_min_dc.dtype != real
        or nominal_min_dc.device != device
    ):
        raise ValueError("invalid nominal minimum cell spacing")
    for field in fields(target):
        value = getattr(target, field.name)
        dtype = torch.int32 if field.name.startswith("jindx") else real
        if (
            value.ndim != 1
            or value.shape[0] < n
            or value.dtype != dtype
            or value.device != device
        ):
            raise ValueError(f"invalid allocated grid member: {field.name}")
    length = (
        torch.full((), config_len_disp, dtype=real, device=device)
        if config_len_disp > 0.0
        else nominal_min_dc.reshape(())
    )
    torch._assert_async(
        torch.isfinite(length) & (length > 0.0), "invalid grid length scale"
    )
    torch._assert_async((mesh_density[:n] > 0.0).all(), "nonpositive mesh density")
    lat, lon = lat_cell[:n], lon_cell[:n]
    sine = torch.sin(lat)
    updates = {
        "xlat": lat,
        "xlon": lon,
        "xlat_d": lat * (180.0 / UFS_CONSTANTS.PI),
        "xlon_d": lon * (180.0 / UFS_CONSTANTS.PI),
        "sinlat": sine,
        "coslat": torch.sqrt(1.0 - sine * sine),
        "area": area_cell[:n],
        "dx": length / mesh_density[:n] ** 0.25,
    }
    return GridProjection(
        replace(
            target,
            **{
                name: torch.cat((value, getattr(target, name)[n:]))
                for name, value in updates.items()
            },
        ),
        length,
    )
