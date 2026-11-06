# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""R4 reverse momentum and surface-pressure updates in source arithmetic order."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from mpas_pytorch.constants import UFS_CONSTANTS, ConstantsProfile
from mpas_pytorch.diagnostics import physics_interface_surface_pressure

from .containers import NativeAtmosphere, ProjectionMesh, Stateout
from .projection import _working_on_device


@dataclass(frozen=True)
class MomentumMesh:
    """Native one-based donors; mesh extents exclude extra allocated padding."""

    n_cells: int
    n_edges_solve: int
    cells_on_edge: torch.Tensor
    east: torch.Tensor
    north: torch.Tensor
    edge_normal_vectors: torch.Tensor


@dataclass(frozen=True)
class CellWindIncrement:
    """Communication buffers; exchange halos before the edge operator consumes them."""

    zonal: torch.Tensor
    meridional: torch.Tensor


@dataclass(frozen=True)
class MomentumCommit:
    atmosphere: NativeAtmosphere
    u: torch.Tensor
    ru: torch.Tensor
    edge_increment: torch.Tensor
    one_sided_edge_count: torch.Tensor
    edge_density: torch.Tensor


def _wind_extents(native: NativeAtmosphere, stateout: Stateout) -> tuple[int, int]:
    n = native.n_cells_solve
    if native.rho_zz.ndim != 2 or type(n) is not int:
        raise ValueError("native wind update requires explicit owned extent")
    levels = native.rho_zz.shape[1]
    if not 0 < n <= native.rho_zz.shape[0] or levels < 1:
        raise ValueError("invalid native wind owned/level extent")
    device = native.rho_zz.device
    # The winds share the working dtype of the native state.
    _working_on_device("rho_zz", native.rho_zz, device)
    dtype = native.rho_zz.dtype
    for name in ("u_reconstruct_zonal", "u_reconstruct_meridional"):
        value = getattr(native, name)
        _working_on_device(name, value, device, dtype)
        if value.ndim != 2 or value.shape[0] < n or value.shape[1] != levels:
            raise ValueError(f"invalid native wind extent: {name}")
    for name in ("gu0", "gv0"):
        value = getattr(stateout, name)
        _working_on_device(name, value, device, dtype)
        if value.ndim != 2 or value.shape[0] < n or value.shape[1] != levels:
            raise ValueError(f"invalid Stateout wind extent: {name}")
        torch._assert_async(
            (torch.isfinite(value[:n]) & (torch.abs(value[:n]) <= 10000.0)).all(),
            "invalid CCPP wind",
        )
    return n, levels


def prepare_cell_wind_increment(
    native: NativeAtmosphere, stateout: Stateout
) -> CellWindIncrement:
    """Match the source temporary buffers, including unchanged allocated tails.

    This does not exchange halo rows. The host must exchange both returned
    buffers, even when a solve edge's second donor is not solve-owned.
    """
    n, _ = _wind_extents(native, stateout)
    return CellWindIncrement(
        torch.cat(
            (
                stateout.gu0[:n] - native.u_reconstruct_zonal[:n],
                native.u_reconstruct_zonal[n:],
            ),
            dim=0,
        ),
        torch.cat(
            (
                stateout.gv0[:n] - native.u_reconstruct_meridional[:n],
                native.u_reconstruct_meridional[n:],
            ),
            dim=0,
        ),
    )


def commit_momentum(
    native: NativeAtmosphere,
    stateout: Stateout,
    mesh: MomentumMesh,
    *,
    u: torch.Tensor,
    ru: torch.Tensor,
    exchanged_increment: CellWindIncrement,
) -> MomentumCommit:
    """Apply the exact two-donor loop after the host's increment halo exchange.

    Donor validity uses nCells, not nCellsSolve or allocated buffer length.
    One valid donor is legal. No valid donor is fatal, as in the original source.
    The returned cell winds precede the host's subsequent vector reconstruction.
    """
    n, levels = _wind_extents(native, stateout)
    nc, ne = mesh.n_cells, mesh.n_edges_solve
    if (
        type(nc) is not int
        or type(ne) is not int
        or not n <= nc <= native.rho_zz.shape[0]
        or ne < 1
    ):
        raise ValueError("invalid native momentum mesh extents")
    device = native.rho_zz.device
    for name, value in (
        ("u", u),
        ("ru", ru),
        ("east", mesh.east),
        ("north", mesh.north),
        ("edge_normal_vectors", mesh.edge_normal_vectors),
        ("zonal_increment", exchanged_increment.zonal),
        ("meridional_increment", exchanged_increment.meridional),
    ):
        _working_on_device(name, value, device, native.rho_zz.dtype)
    if u.ndim != 2 or u.shape[0] < ne or u.shape[1] != levels or ru.shape != u.shape:
        raise ValueError("invalid native edge momentum extent")
    for name in ("east", "north"):
        value = getattr(mesh, name)
        if value.ndim != 2 or value.shape[0] < nc or value.shape[1] != 3:
            raise ValueError("invalid cell Cartesian basis extent")
    normals = mesh.edge_normal_vectors
    if normals.ndim != 2 or normals.shape[0] < ne or normals.shape[1] != 3:
        raise ValueError("invalid edge-normal extent")
    pairs = mesh.cells_on_edge
    if (
        not isinstance(pairs, torch.Tensor)
        or pairs.dtype != torch.int64
        or pairs.device != device
        or pairs.ndim != 2
        or pairs.shape[0] < ne
        or pairs.shape[1] != 2
    ):
        raise ValueError("native edge donors require device int64 one-based pairs")
    for value, old in (
        (exchanged_increment.zonal, native.u_reconstruct_zonal),
        (exchanged_increment.meridional, native.u_reconstruct_meridional),
    ):
        if value.shape != old.shape or value.shape[0] < nc:
            raise ValueError(
                "exchanged wind increment extent does not match native buffer"
            )
    torch._assert_async(
        (
            (
                exchanged_increment.zonal[:n]
                == stateout.gu0[:n] - native.u_reconstruct_zonal[:n]
            )
            & (
                exchanged_increment.meridional[:n]
                == stateout.gv0[:n] - native.u_reconstruct_meridional[:n]
            )
        ).all(),
        "halo exchange changed owned CCPP wind increments",
    )
    pairs = pairs[:ne]
    valid = (pairs >= 1) & (pairs <= nc)
    torch._assert_async(
        valid.any(dim=1).all(), "native solve edge has no valid CCPP wind donor"
    )
    # Safe gather indices do not turn an invalid donor into a physical donor.
    indices = (pairs - 1).clamp(0, nc - 1)
    normal = normals[:ne]
    increment = torch.zeros_like(u[:ne])
    density_sum = torch.zeros_like(increment)
    donor_count = torch.zeros_like(increment)
    for donor in range(2):
        mask = valid[:, donor, None]
        selected = indices[:, donor]
        east = torch.where(mask, mesh.east[selected], torch.zeros_like(normal))
        north = torch.where(mask, mesh.north[selected], torch.zeros_like(normal))
        zonal = torch.where(
            mask, exchanged_increment.zonal[selected], torch.zeros_like(increment)
        )
        meridional = torch.where(
            mask, exchanged_increment.meridional[selected], torch.zeros_like(increment)
        )
        density = torch.where(
            mask, native.rho_zz[selected], torch.zeros_like(increment)
        )
        normal_zonal = (
            normal[:, 0] * east[:, 0]
            + normal[:, 1] * east[:, 1]
            + normal[:, 2] * east[:, 2]
        )
        normal_meridional = (
            normal[:, 0] * north[:, 0]
            + normal[:, 1] * north[:, 1]
            + normal[:, 2] * north[:, 2]
        )
        # Preserve the source's running +u then +v association, including donor 2.
        increment = torch.where(
            mask,
            increment
            + zonal * normal_zonal[:, None]
            + meridional * normal_meridional[:, None],
            increment,
        )
        density_sum = torch.where(mask, density_sum + density, density_sum)
        donor_count = donor_count + mask.to(dtype=u.dtype)
    increment = increment / torch.clamp_min(donor_count, 1.0)
    updated_u = u[:ne] + increment
    updated_ru = updated_u * density_sum / torch.clamp_min(donor_count, 1.0)
    for value in (increment, updated_u, updated_ru):
        torch._assert_async(
            torch.isfinite(value).all(), "nonfinite committed edge momentum"
        )
    atmosphere = replace(
        native,
        u_reconstruct_zonal=torch.cat(
            (stateout.gu0[:n], native.u_reconstruct_zonal[n:]), dim=0
        ),
        u_reconstruct_meridional=torch.cat(
            (stateout.gv0[:n], native.u_reconstruct_meridional[n:]), dim=0
        ),
    )
    return MomentumCommit(
        atmosphere,
        torch.cat((updated_u, u[ne:]), dim=0),
        torch.cat((updated_ru, ru[ne:]), dim=0),
        increment,
        (valid.sum(dim=1) == 1).sum(),
        torch.cat(
            (density_sum / torch.clamp_min(donor_count, 1.0), torch.zeros_like(u[ne:])),
            dim=0,
        ),
    )


def commit_surface_pressure(
    native: NativeAtmosphere,
    mesh: ProjectionMesh,
    *,
    native_qv_index: int,
    constants: ConstantsProfile = UFS_CONSTANTS,
) -> NativeAtmosphere:
    """Use maintained native surface-pressure numerics after thermodynamic commit."""
    n = native.n_cells_solve
    if (
        native.rho_zz.ndim != 2
        or type(n) is not int
        or not 0 < n <= native.rho_zz.shape[0]
    ):
        raise ValueError("invalid surface commit owned extent")
    count, levels = native.rho_zz.shape
    if (
        native.scalars.ndim != 3
        or native.scalars.shape[:2] != (count, levels)
        or type(native_qv_index) is not int
        or not 0 <= native_qv_index < native.scalars.shape[2]
    ):
        raise ValueError("invalid native surface commit vapor slot")
    _working_on_device("rho_zz", native.rho_zz, native.rho_zz.device)
    for name, value in (
        ("scalars", native.scalars),
        ("zz", mesh.zz),
        ("zgrid", mesh.zgrid),
        ("pressure_p", native.pressure_p),
        ("pressure_base", native.pressure_base),
        ("surface_pressure", native.surface_pressure),
    ):
        _working_on_device(name, value, native.rho_zz.device, native.rho_zz.dtype)
    if native.surface_pressure.shape != (count,):
        raise ValueError("invalid native surface pressure extent")
    result = physics_interface_surface_pressure(
        nCells=n,
        rho_zz=native.rho_zz[:n],
        zz=mesh.zz[:n],
        qv=native.scalars[:n, :, native_qv_index],
        pressure_p=native.pressure_p[:n],
        pressure_base=native.pressure_base[:n],
        zgrid=mesh.zgrid[:n],
        gravity=constants.G,
    )
    torch._assert_async(
        torch.isfinite(result).all(), "nonfinite committed surface pressure"
    )
    return replace(
        native, surface_pressure=torch.cat((result, native.surface_pressure[n:]), dim=0)
    )
