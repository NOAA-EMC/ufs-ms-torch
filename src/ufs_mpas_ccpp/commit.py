# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Functional R4 native commit boundary, using the maintained host operators."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
from mpas_pytorch.constants import UFS_CONSTANTS, ConstantsProfile
from mpas_pytorch.state import bind_ufs_physics_commit

from .containers import NativeAtmosphere, ProjectionMesh, Statein, Stateout
from .distributed import partition_scope
from .ledger import CommitLedgers, CumulativeLimiter, commit_ledgers
from .momentum import (
    MomentumMesh,
    commit_momentum,
    commit_surface_pressure,
    prepare_cell_wind_increment,
)
from .projection import TracerContract
from .refresh import OutputDiagnostics, ReconstructionMesh, refresh_host_diagnostics
from .reverse import CommitPhase, commit_thermodynamics
from .runtime_policy import cycle_observers_enabled, glue
from .surface_containers import NativeSurface
from .surface_mirror import SurfaceMirror, SurfaceMirrorInputs, mirror_surface

NATIVE_SURFACE_UPDATE_KEYS = ("skintemp", "sst", "snow", "snowh", "snowc", "smois", "sh2o", "tslb")


def finalize_stateout(prefix, *, dtp: torch.Tensor):
    """Apply GFS_mpas_stateout_update without erasing direct process updates.

    The source leaves the ledger unchanged and does not floor tracers here.
    Native tracer limiting belongs to the subsequent reverse bridge.
    """
    stateout, work = prefix.stateout, prefix.interstitial
    reference = stateout.gt0
    if dtp.shape != () or dtp.dtype != reference.dtype or dtp.device != reference.device:
        raise ValueError("Stateout finalization requires a device-local timestep scalar")
    updated = {}
    for state, ledger in (("gt0", "dtdt"), ("gu0", "dudt"), ("gv0", "dvdt"), ("gq0", "dqdt")):
        value, tendency = getattr(stateout, state), work[ledger]
        if value.shape != tendency.shape or value.dtype != tendency.dtype or value.device != tendency.device:
            raise ValueError(f"Stateout finalization ledger differs: {ledger}")
        updated[state] = value + tendency * dtp
    return replace(prefix, stateout=replace(stateout, **updated))


@dataclass(frozen=True)
class CommitDomain:
    """Physical extents, distinct from allocated padding and solve extents."""

    n_cells: int
    n_edges: int
    mpi_size: int


@dataclass(frozen=True)
class CommitGeometry:
    projection: ProjectionMesh
    momentum: MomentumMesh
    reconstruction: ReconstructionMesh
    area_cell: torch.Tensor
    rtheta_base: torch.Tensor

    @classmethod
    def from_mpas_state(
        cls,
        state: Mapping[str, Any],
        mesh: Mapping[str, Any],
        *,
        on_a_sphere: bool = True,
    ):
        """Bind the initialized canonical geometry without numerical preparation."""
        values = {**mesh, **state}
        nc, ne = values["nCells"], values["nEdges"]
        return cls(
            ProjectionMesh.from_mpas_state(values),
            MomentumMesh(
                nc,
                # A partition's edge update covers its owned edges; one rank owns them all.
                values.get("nEdgesSolve", ne),
                values["cellsOnEdge"],
                values["east"],
                values["north"],
                values["edgeNormalVectors"],
            ),
            ReconstructionMesh(
                nc,
                ne,
                values["edgesOnCell"],
                values["nEdgesOnCell"],
                values["coeffs_reconstruct"],
                values["latCell"],
                values["lonCell"],
                on_a_sphere,
            ),
            values["areaCell"],
            values["rtheta_base"],
        )


@dataclass(frozen=True)
class NativeCommitState:
    atmosphere: NativeAtmosphere
    surface: NativeSurface
    diagnostics: OutputDiagnostics
    u: torch.Tensor
    ru: torch.Tensor
    rtheta_p: torch.Tensor
    cumulative: CumulativeLimiter | None


@dataclass(frozen=True)
class NativeCommitResult:
    state: NativeCommitState
    ledgers: CommitLedgers | None  # None with observers off: no physics reads the ledgers
    surface_mirror: SurfaceMirror | None
    edge_density: torch.Tensor | None = None  # donor-mean rho_zz of the owned edges, zero on the others


def commit_mpas_state(
    before: dict[str, Any],
    surface: NativeSurface,
    statein: Statein,
    stateout: Stateout,
    *,
    mesh: dict[str, Any],
    config: dict[str, Any],
    domain: CommitDomain,
    phase: CommitPhase,
    tracers: TracerContract,
    water_species: tuple[bool, ...],
    cumulative: CumulativeLimiter | None,
    sfcprop: SurfaceMirrorInputs | None,
) -> tuple[dict[str, Any], NativeCommitResult]:
    """Commit live physics into the maintained MPAS state and carry map.

    Both phases run the same operators; ``result.state`` holds the state they propose. Microphysics
    installs it and leaves its heating rate in ``rt_diabatic_tend`` for the next dycore step, as
    ``microphysics_to_MPAS`` does. Radiation/physics installs its tracers and surface, and hands its
    theta_m and u changes to the dycore as the RK forcing ``tend_rtheta_physics`` and
    ``tend_ru_physics``, as ``physics_get_tend`` does in standalone MPAS. Heating written directly at
    constant density is a pressure jump, largest in the thinnest layers, and the horizontal pressure
    gradient answers it with a wind impulse that grows with the timestep; through the acoustic steps
    the vertically implicit solve adjusts it as it arrives. Both paths are needed at the official
    timestep (6 x spacing in km): without ``rt_diabatic_tend``, surface fluxes in the 47 m layer feed a
    condensation jump, the wind spike follows, and sea spray closes the loop.

    The caller must supply a completed initialization/step boundary. The next
    public dycore step authenticates this entry and uses R4 refresh semantics.
    Surface/limiter state remains in the returned host result for the next phase.
    This does not declare a completed coupled cycle or a restart boundary.
    """
    geometry = CommitGeometry.from_mpas_state(
        before,
        mesh,
        on_a_sphere=config.get("config_on_a_sphere", True),
    )
    diagnostics = OutputDiagnostics(
        before["rho"],
        before["pressure"],
        before["uReconstructX"],
        before["uReconstructY"],
        before["uReconstructZ"],
    )
    native = NativeCommitState(
        NativeAtmosphere.from_mpas_state(before, n_cells_solve=domain.n_cells),
        surface,
        diagnostics,
        before["u"],
        before["ru"],
        before["rtheta_p"],
        cumulative,
    )
    result = commit_native(
        native,
        geometry,
        statein,
        stateout,
        domain=domain,
        phase=phase,
        tracers=tracers,
        water_species=water_species,
        sfcprop=sfcprop,
        halo=config.get("_halo_exchanger"),
    )
    atmosphere = result.state.atmosphere
    if phase is CommitPhase.RADIATION_PHYSICS:
        dt = config["config_dt"]
        # The dycore's no-grad step and its halo exchanges reuse these work tensors in place; the previous
        # state keeps its own.
        fresh = {name: before[name].clone() for name in ("theta_m", "theta", "exner", "pressure_p", "rtheta_p", "u", "ru")}
        after = {
            **before,
            **fresh,
            "scalars": atmosphere.scalars,
            "qv": atmosphere.scalars[..., 0],
            "t": fresh["theta_m"],
            "tend_rtheta_physics": before["rho_zz"] * (atmosphere.theta_m - before["theta_m"]) / dt,
            "tend_ru_physics": result.edge_density * (result.state.u - before["u"]) / dt,
        }
    else:
        after = {
            **before,
            **{
                name: getattr(atmosphere, name)
                for name in (
                    "scalars",
                    "theta_m",
                    "theta",
                    "exner",
                    "pressure_p",
                    "surface_pressure",
                )
            },
            "u": result.state.u,
            "ru": result.state.ru,
            "rtheta_p": result.state.rtheta_p,
            "rho": result.state.diagnostics.rho,
            "pressure": result.state.diagnostics.pressure,
            "uReconstructX": result.state.diagnostics.u_reconstruct_x,
            "uReconstructY": result.state.diagnostics.u_reconstruct_y,
            "uReconstructZ": result.state.diagnostics.u_reconstruct_z,
            "uReconstructZonal": atmosphere.u_reconstruct_zonal,
            "uReconstructMeridional": atmosphere.u_reconstruct_meridional,
            "ur_cell": atmosphere.u_reconstruct_zonal,
            "vr_cell": atmosphere.u_reconstruct_meridional,
            "qv": atmosphere.scalars[..., 0],
            "t": atmosphere.theta_m,
            "rt_diabatic_tend": (atmosphere.theta_m - before["theta_m"]) / config["config_dt"],
        }
    for name in NATIVE_SURFACE_UPDATE_KEYS:
        if name in before:
            value = getattr(result.state.surface, name)
            if (
                value.shape != before[name].shape
                or value.dtype != before[name].dtype
                or value.device != before[name].device
            ):
                raise ValueError(f"native surface binding differs: {name}")
            after[name] = value
    bind_ufs_physics_commit(
        before,
        after,
        mesh=mesh,
        config=config,
        dt=config["config_dt"],
    )
    return after, result


def commit_native(
    before: NativeCommitState,
    geometry: CommitGeometry,
    statein: Statein,
    stateout: Stateout,
    *,
    domain: CommitDomain,
    phase: CommitPhase,
    tracers: TracerContract,
    water_species: tuple[bool, ...],
    sfcprop: SurfaceMirrorInputs | None,
    constants: ConstantsProfile = UFS_CONSTANTS,
    halo=None,
) -> NativeCommitResult:
    """Compose the actual TL1 boundary without captured intermediate injection.

    One fully owned rank has no received wind-increment or primary halo rows.
    A partitioned domain (``domain.mpi_size`` ranks) commits its owned cells and
    edges and exchanges, with its ``halo`` (the dycore's ``DynamicsHaloExchange``),
    the cell wind increments before the edge update reads its donor cells, and the
    source eight-field commit group (theta_m, rho_zz, scalars, u, ru, pressure_p,
    rtheta_p, exner) before the output refresh; no mutation intervenes. Ledger sums
    run over all cells in global-ID order (``ufs_mpas_ccpp.distributed``).
    This operator does not initialize geometry, run CCPP groups, synchronize
    dycore time levels, or qualify a complete forecast/restart boundary.
    """
    nc, ne, ranks = domain.n_cells, domain.n_edges, domain.mpi_size
    partitioned = halo is not None
    local_cells, local_edges = geometry.momentum.n_cells, geometry.reconstruction.n_edges
    if (
        any(type(v) is not int for v in (nc, ne, ranks))
        or min(nc, ne, ranks) < 1
        or partitioned != (ranks > 1)
        or before.atmosphere.n_cells_solve != nc
        or geometry.momentum.n_edges_solve != ne
        or geometry.reconstruction.n_cells != local_cells
        or (not partitioned and (local_cells != nc or local_edges != ne))
        or (partitioned and (
            halo.transport.world_size != ranks
            or halo.partition.cells.owned_count != nc
            or halo.partition.edges.owned_count != ne
            or halo.partition.cells.local_count != local_cells
            or halo.partition.edges.local_count != local_edges))
    ):
        raise ValueError("native commit requires a fully owned one-rank domain, "
                         "or a partitioned domain with its halo exchanger")
    if not isinstance(phase, CommitPhase):
        raise ValueError("native commit requires an explicit CommitPhase")
    if tracers.native_qv_index != 0:
        raise ValueError("selected R4 output refresh requires native vapor slot zero")
    if phase is CommitPhase.RADIATION_PHYSICS:
        if not isinstance(sfcprop, SurfaceMirrorInputs):
            raise ValueError("phase1 native commit requires live Sfcprop mirror inputs")
    elif sfcprop is not None:
        raise ValueError("phase2 native commit must not mirror Sfcprop")

    observe = cycle_observers_enabled()
    cells = glue(commit_thermodynamics)(
        before.atmosphere,
        geometry.projection,
        statein,
        stateout,
        area_cell=geometry.area_cell,
        rtheta_base=geometry.rtheta_base,
        rtheta_p=before.rtheta_p,
        phase=phase,
        tracers=tracers,
        constants=constants,
        ledgers=observe,
    )
    ledgers = None
    if observe:
        with partition_scope(halo):  # sums over all cells in global-ID order
            ledgers = commit_ledgers(
                before.atmosphere,
                cells,
                tracers=tracers,
                water_species=water_species,
                cumulative=before.cumulative,
            )
    increments = prepare_cell_wind_increment(cells.atmosphere, stateout)
    if partitioned:
        exchanged = _exchanged(halo, "ufs_commit:wind_increment",
                               zonal=increments.zonal, meridional=increments.meridional)
        increments = replace(increments, **exchanged)
    winds = glue(commit_momentum)(
        cells.atmosphere,
        stateout,
        geometry.momentum,
        u=before.u,
        ru=before.ru,
        exchanged_increment=increments,
    )
    atmosphere = commit_surface_pressure(
        winds.atmosphere,
        geometry.projection,
        native_qv_index=tracers.native_qv_index,
        constants=constants,
    )
    u, ru, rtheta_p = winds.u, winds.ru, cells.rtheta_p
    if partitioned:
        primaries = _exchanged(halo, "ufs_commit:primaries", theta_m=atmosphere.theta_m,
            rho_zz=atmosphere.rho_zz, scalars=atmosphere.scalars, pressure_p=atmosphere.pressure_p,
            rtheta_p=rtheta_p, exner=atmosphere.exner, u=u, ru=ru)
        u, ru, rtheta_p = primaries.pop("u"), primaries.pop("ru"), primaries.pop("rtheta_p")
        atmosphere = replace(atmosphere, **primaries)
    mirrored = (
        glue(mirror_surface)(before.surface, sfcprop, n_cells_solve=nc)
        if phase is CommitPhase.RADIATION_PHYSICS
        else None
    )
    refreshed = glue(refresh_host_diagnostics)(
        atmosphere,
        geometry.projection,
        geometry.reconstruction,
        before.diagnostics,
        exchanged_u=u,
        constants=constants,
    )
    return NativeCommitResult(
        NativeCommitState(
            refreshed.atmosphere,
            before.surface if mirrored is None else mirrored.native,
            refreshed.diagnostics,
            u,
            ru,
            rtheta_p,
            ledgers.cumulative if observe else before.cumulative,
        ),
        ledgers,
        mirrored,
        winds.edge_density,
    )


def _exchanged(halo, group: str, **fields: torch.Tensor) -> dict[str, torch.Tensor]:
    """The fields with their halo rows from their owners (new tensors; the inputs are not written)."""
    halo.exchange_group(group, fields, caller_owned=True)
    return fields
