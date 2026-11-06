# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Control vector of a coupled state: read the controls, or return a state with controls replaced.

The controls are the dycore prognostics (native ``u``, ``w``, ``rw``, ``theta_m``, ``rho_zz``, ``scalars``) and
every floating physics-side tensor of the step boundary with more than one element (land, surface,
NSST, carried radiation and cloud state, physics tables), except the model clock ``sec``. A
replacement may be a new tensor, a leaf that requires grad, a dual tensor or a perturbed copy.

``with_controls`` keeps the state a valid continuation:
- Aliases move with their control (the native state binds one tensor under several names, such as
  ``t`` and ``theta_m``, or ``sst`` and the surface container's member).
- The native diagnostics that the next cycle reads before it recomputes them follow the native
  prognostics through their MPAS relations ``f``: the coupled diagnostics (``ru``,
  ``rho_p``, ``rtheta_p``, ``exner``, ``pressure_p``), the output refresh (``theta``, rho,
  pressure, reconstructed winds) and the surface pressure. Each is anchored to its carried value,
  ``d = d0 - (f(x0) - f(x))``: unchanged controls give the carried bytes exactly (the carried value
  comes from the commit's own operation order, not from ``f``), and derivatives are those of ``f``.
- ``rw`` is a control of its own: the split-explicit solver carries it beside ``w``, and at a cycle
  boundary it departs from the coupled-diagnostic relation (relative 1.5e-3 at x1.10242, cycle 3),
  while every anchored diagnostic agrees with its relation to rounding (``anchored_residuals``).
- On a partition (``config["_halo_exchanger"]``) the owned rows are the controls: the halo rows of the
  native controls are taken from their owners (the dycore's ``ufs_controls`` halo group), with their
  tangents and cotangents, so a replaced control's own halo rows have no effect. The anchored wind
  and surface-pressure diagnostics follow on owned cells (as the commit refreshes them).
- The commit and boundary receipts are rebound to the new tensors, as an ordinary step leaves them.
  A cold-start state carries the dycore's complete-boundary marker instead of a commit receipt;
  the marker is bound again to the new native state, so the dycore reuses the anchored diagnostics
  as it reuses the carried ones of an unedited boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch
from mpas_pytorch.atmosphere_model import _init_coupled_diagnostics
from mpas_pytorch.state import _BOUNDARY_MARKER

from .commit import CommitGeometry
from .containers import NativeAtmosphere
from .continuation import bind_coupled_boundary, require_coupled_boundary
from .cycle import CoupledCycleState, mark_cold_boundary
from .momentum import commit_surface_pressure
from .refresh import OutputDiagnostics, refresh_host_diagnostics
from .step_graphs import flatten_state, restamp, stamp_names, unflatten_state

NATIVE_CONTROLS = tuple(f"/native_state/{name}" for name in ("u", "w", "rw", "theta_m", "rho_zz", "scalars"))
_COUPLED = ("ru", "rho_p", "rtheta_p", "exner", "pressure_p")
_REFRESHED = ("theta", "uReconstructZonal", "uReconstructMeridional", "uReconstructX",
              "uReconstructY", "uReconstructZ", "rho", "pressure", "surface_pressure")


def control_paths(state: CoupledCycleState) -> tuple[str, ...]:
    """Boundary paths of the controls, in boundary order."""
    boundary, _ = flatten_state(state)
    return tuple(path for path, value in boundary.items()
                 if isinstance(value, torch.Tensor) and value.is_floating_point()
                 and (path in NATIVE_CONTROLS
                      or (not path.startswith("/native_state/") and path != "/sec" and value.numel() > 1)))


def controls(state: CoupledCycleState) -> dict[str, torch.Tensor]:
    """The control tensors of ``state`` by path (the state's own tensors, not copies)."""
    boundary, _ = flatten_state(state)
    return {path: boundary[path] for path in control_paths(state)}


def _native_diagnostics(native: Mapping, mesh: Mapping, config: Mapping) -> dict[str, torch.Tensor]:
    """The MPAS relations ``f`` from the native prognostics in ``native`` to the anchored diagnostics."""
    coupled = {key: value for key, value in native.items() if key not in ("theta", "rho", "qv")}
    _init_coupled_diagnostics(coupled, dict(mesh), config=config)
    values = {**native, **{key: coupled[key] for key in _COUPLED}}
    nc = values["nCells"] if "nCells" in values else mesh["nCells"]
    nc = values.get("nCellsSolve", nc)  # a partition refreshes its owned cells (one rank: all cells)
    geometry = CommitGeometry.from_mpas_state(values, mesh, on_a_sphere=config.get("config_on_a_sphere", True))
    atmosphere = commit_surface_pressure(NativeAtmosphere.from_mpas_state(values, n_cells_solve=nc),
                                         geometry.projection, native_qv_index=0)
    carried = OutputDiagnostics(values["rho"], values["pressure"], values["uReconstructX"],
                                values["uReconstructY"], values["uReconstructZ"])
    refreshed = refresh_host_diagnostics(atmosphere, geometry.projection, geometry.reconstruction, carried,
                                         exchanged_u=values["u"])
    out = {key: coupled[key] for key in _COUPLED}
    out.update(theta=refreshed.atmosphere.theta, uReconstructZonal=refreshed.atmosphere.u_reconstruct_zonal,
               uReconstructMeridional=refreshed.atmosphere.u_reconstruct_meridional,
               uReconstructX=refreshed.diagnostics.u_reconstruct_x,
               uReconstructY=refreshed.diagnostics.u_reconstruct_y,
               uReconstructZ=refreshed.diagnostics.u_reconstruct_z,
               rho=refreshed.diagnostics.rho, pressure=refreshed.diagnostics.pressure,
               surface_pressure=atmosphere.surface_pressure)
    return out


def anchored_residuals(state: CoupledCycleState, *, mesh: Mapping, config: Mapping) -> dict[str, float]:
    """max |d0 - f(x0)| / max |d0| per anchored diagnostic: how far the carried value is from ``f``.

    Rounding-level values confirm that ``f`` is the relation the model maintains."""
    native = state.native_state
    f0 = _native_diagnostics(native, mesh, config)
    return {key: float((native[key] - f0[key]).abs().max() / native[key].abs().max().clamp_min(1e-300))
            for key in f0}


def with_controls(state: CoupledCycleState, values: Mapping[str, torch.Tensor], *,
                  mesh: Mapping, config: Mapping) -> CoupledCycleState:
    """``state`` with the given controls replaced (see the module docstring)."""
    unknown = sorted(set(values) - set(control_paths(state)))
    if unknown:
        raise KeyError(f"not controls of the coupled state: {unknown[:5]}")
    halo = config.get("_halo_exchanger")
    if halo is not None and any(path in values for path in NATIVE_CONTROLS):
        names = [path.rsplit("/", 1)[1] for path in NATIVE_CONTROLS]
        fields = {name: values.get(path, state.native_state[name]) for name, path in zip(names, NATIVE_CONTROLS)}
        halo.exchange_group("ufs_controls", fields, caller_owned=True)
        values = {**values, **{path: fields[name] for name, path in zip(names, NATIVE_CONTROLS) if path in values}}
    return _replace(state, values, mesh=mesh, config=config, anchor=True)


def _replace(state: CoupledCycleState, values: Mapping[str, torch.Tensor], *, mesh: Mapping, config: Mapping,
             anchor: bool) -> CoupledCycleState:
    """Any boundary tensors replaced (aliases follow, receipts rebound); with ``anchor`` the native
    diagnostics follow the native prognostics. Diagnosis tools use it to perturb a diagnostic alone."""
    boundary, stamps = flatten_state(state)
    replacement: dict[int, torch.Tensor] = {}

    def assign(old: torch.Tensor, new: torch.Tensor, path: str) -> None:
        if new.shape != old.shape or new.dtype != old.dtype or new.device != old.device:
            raise ValueError(f"control {path} must keep shape, dtype and device")
        if replacement.setdefault(id(old), new) is not new:
            raise ValueError(f"aliased controls disagree at {path}")

    for path, new in values.items():
        assign(boundary[path], new, path)
    native = state.native_state
    if anchor and any(path in values for path in NATIVE_CONTROLS):
        moved = {key: replacement.get(id(value), value) for key, value in native.items()}
        f0 = _native_diagnostics(native, mesh, config)
        f1 = _native_diagnostics(moved, mesh, config)
        for key in f0:
            assign(native[key], native[key] - (f0[key] - f1[key]), f"/native_state/{key}")
        # The commit binds the vapor name to a view of the scalars (never read as a source).
        if "qv" in native and "/native_state/scalars" in values:
            assign(native["qv"], replacement.get(id(native["scalars"]), native["scalars"])[..., 0],
                   "/native_state/qv")
    named = stamp_names(boundary, stamps)
    new_boundary = {path: replacement.get(id(value), value) if isinstance(value, torch.Tensor) else value
                    for path, value in boundary.items()}
    result = unflatten_state(state, {**new_boundary, **restamp(new_boundary, named)}, state.step_index)
    if _BOUNDARY_MARKER in native and any(id(value) in replacement for value in native.values()):
        marker = native[_BOUNDARY_MARKER]
        mark_cold_boundary(result.native_state, mesh=mesh if marker["mesh"] is not None else None,
                           config=config if marker["config"] is not None else None, dt=marker["dt"])
    if state.continuation is not None:
        require_coupled_boundary(state)
        result = bind_coupled_boundary(replace(result, continuation=None))
    return result
