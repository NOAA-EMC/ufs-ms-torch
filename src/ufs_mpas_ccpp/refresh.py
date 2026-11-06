# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Source host output refresh after the native commit halo exchange."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from mpas_pytorch.constants import UFS_CONSTANTS, ConstantsProfile
from mpas_pytorch.diagnostics import native_output_diagnostics
from mpas_pytorch.dynamics.vector_reconstruction import mpas_reconstruct_2d

from .containers import NativeAtmosphere, ProjectionMesh
from .projection import _working_on_device


@dataclass(frozen=True)
class ReconstructionMesh:
    """Initialized MPAS geometry, not the stale coefficients stored in the IC."""

    n_cells: int
    n_edges: int
    edges_on_cell: torch.Tensor
    n_edges_on_cell: torch.Tensor
    coeffs_reconstruct: torch.Tensor
    lat_cell: torch.Tensor
    lon_cell: torch.Tensor
    on_a_sphere: bool = True


@dataclass(frozen=True)
class OutputDiagnostics:
    """Native diagnostic buffers; unwritten halo/padding rows retain their values."""

    rho: torch.Tensor
    pressure: torch.Tensor
    u_reconstruct_x: torch.Tensor
    u_reconstruct_y: torch.Tensor
    u_reconstruct_z: torch.Tensor


@dataclass(frozen=True)
class DiagnosticRefresh:
    atmosphere: NativeAtmosphere
    diagnostics: OutputDiagnostics


def refresh_host_diagnostics(
    native: NativeAtmosphere,
    projection_mesh: ProjectionMesh,
    reconstruction_mesh: ReconstructionMesh,
    carried: OutputDiagnostics,
    *,
    exchanged_u: torch.Tensor,
    constants: ConstantsProfile = UFS_CONSTANTS,
) -> DiagnosticRefresh:
    """Match ufs_mpas_refresh_diagnostics using already halo-exchanged primaries.

    Output thermodynamics cover nCells. Wind reconstruction covers nCellsSolve:
    the source does not supply includeHalos to mpas_reconstruct. This function
    does not perform transport or replace halo primaries with captured outputs.
    The selected R4 native water-vapor index is zero, as in the canonical helper.
    """
    mesh = reconstruction_mesh
    nc, ne, owned = mesh.n_cells, mesh.n_edges, native.n_cells_solve
    if (
        any(type(value) is not int for value in (nc, ne, owned))
        or not 0 < owned <= nc
        or ne <= 0
        or type(mesh.on_a_sphere) is not bool
        or native.rho_zz.ndim != 2
    ):
        raise ValueError("invalid native refresh extents/configuration")
    levels = native.rho_zz.shape[1]
    device = native.rho_zz.device
    dtype = native.rho_zz.dtype
    _working_on_device("rho_zz", native.rho_zz, device)
    if levels < 1 or exchanged_u.ndim != 2 or exchanged_u.shape[0] < ne:
        raise ValueError("invalid native refresh edge extent")

    def cell_field(name: str, value: torch.Tensor, count: int) -> None:
        _working_on_device(name, value, device, dtype)
        if value.ndim != 2 or value.shape[0] < count or value.shape[1] != levels:
            raise ValueError(f"invalid native refresh field extent: {name}")

    for name in ("rho_zz", "theta_m", "theta", "pressure_base", "pressure_p"):
        cell_field(name, getattr(native, name), nc)
    cell_field("zz", projection_mesh.zz, nc)
    for name in ("u_reconstruct_zonal", "u_reconstruct_meridional"):
        cell_field(name, getattr(native, name), owned)
    for name in ("rho", "pressure"):
        cell_field(name, getattr(carried, name), nc)
    for name in ("u_reconstruct_x", "u_reconstruct_y", "u_reconstruct_z"):
        cell_field(name, getattr(carried, name), owned)
    _working_on_device("scalars", native.scalars, device, dtype)
    if (
        native.scalars.ndim != 3
        or native.scalars.shape[0] < nc
        or native.scalars.shape[1] != levels
        or native.scalars.shape[2] < 1
    ):
        raise ValueError("invalid native refresh scalar extent")
    cell_field("exchanged_u", exchanged_u, ne)
    torch._assert_async(
        torch.isfinite(exchanged_u[:ne]).all(), "nonfinite refresh edge velocity"
    )

    edges, counts = mesh.edges_on_cell, mesh.n_edges_on_cell
    if (
        edges.dtype != torch.int64
        or counts.dtype != torch.int64
        or edges.device != device
        or counts.device != device
        or edges.ndim != 2
        or edges.shape[0] < owned
        or edges.shape[1] < 1
        or counts.ndim != 1
        or counts.shape[0] < owned
    ):
        raise ValueError("invalid native refresh connectivity")
    slots = edges.shape[1]
    coeffs = mesh.coeffs_reconstruct
    _working_on_device("coeffs_reconstruct", coeffs, device, dtype)
    if coeffs.ndim != 3 or coeffs.shape[0] < owned or coeffs.shape[1:] != (slots, 3):
        raise ValueError("invalid native refresh coefficient extent")
    for name in ("lat_cell", "lon_cell"):
        value = getattr(mesh, name)
        _working_on_device(name, value, device, dtype)
        if value.ndim != 1 or value.shape[0] < owned:
            raise ValueError(f"invalid native refresh coordinate extent: {name}")
    active = torch.arange(slots, device=device)[None, :] < counts[:owned, None]
    donors = edges[:owned]
    torch._assert_async(
        ((counts[:owned] > 0) & (counts[:owned] <= slots)).all(),
        "invalid native refresh stencil count",
    )
    torch._assert_async(
        (~active | ((donors >= 1) & (donors <= ne))).all(),
        "invalid native refresh edge donor",
    )
    # Mask coefficients before multiplication, including unused NaN IC slots.
    safe_coeffs = torch.where(active[..., None], coeffs[:owned], 0.0)
    torch._assert_async(
        torch.isfinite(safe_coeffs).all(), "nonfinite refresh coefficient"
    )
    payload = {
        "edge_rows_u": torch.where(active, donors - 1, 0),
        "valid": active,
        "coeff_xyz": safe_coeffs,
        "row_start": 0,
        "row_end": owned,
        "cell_padding_offset": 0,
        "clat": torch.cos(mesh.lat_cell[:owned])[:, None],
        "slat": torch.sin(mesh.lat_cell[:owned])[:, None],
        "clon": torch.cos(mesh.lon_cell[:owned])[:, None],
        "slon": torch.sin(mesh.lon_cell[:owned])[:, None],
    }
    winds = mpas_reconstruct_2d(
        owned,
        ne,
        levels,
        exchanged_u,
        coeffs,
        edges,
        counts,
        mesh.lat_cell,
        mesh.lon_cell,
        on_a_sphere=mesh.on_a_sphere,
        cell_padding_offset=0,
        _connectivity_payload=payload,
    )
    thermo = native_output_diagnostics(
        theta_m=native.theta_m[:nc],
        rho_zz=native.rho_zz[:nc],
        scalars=native.scalars[:nc],
        zz=projection_mesh.zz[:nc],
        pressure_base=native.pressure_base[:nc],
        pressure_p=native.pressure_p[:nc],
        rvord=constants.RVORD,
    )

    def retain_tail(
        value: torch.Tensor, active_value: torch.Tensor, count: int
    ) -> torch.Tensor:
        torch._assert_async(
            torch.isfinite(active_value[:count]).all(), "nonfinite refreshed diagnostic"
        )
        return torch.cat((active_value[:count], value[count:]), dim=0)

    refreshed = replace(
        native,
        theta=retain_tail(native.theta, thermo["theta"], nc),
        u_reconstruct_zonal=retain_tail(
            native.u_reconstruct_zonal, winds["uReconstructZonal"], owned
        ),
        u_reconstruct_meridional=retain_tail(
            native.u_reconstruct_meridional, winds["uReconstructMeridional"], owned
        ),
    )
    diagnostics = OutputDiagnostics(
        retain_tail(carried.rho, thermo["rho"], nc),
        retain_tail(carried.pressure, thermo["pressure"], nc),
        retain_tail(carried.u_reconstruct_x, winds["uReconstructX"], owned),
        retain_tail(carried.u_reconstruct_y, winds["uReconstructY"], owned),
        retain_tail(carried.u_reconstruct_z, winds["uReconstructZ"], owned),
    )
    return DiagnosticRefresh(refreshed, diagnostics)
