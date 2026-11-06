# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""R4 native commit operators; no Fortran or host-valued tensor controls."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum

import torch
from mpas_pytorch.constants import UFS_CONSTANTS, ConstantsProfile

from .containers import NativeAtmosphere, ProjectionMesh, Statein, Stateout
from .projection import TracerContract, _working_on_device, _pressure_power
from .precision import eps_scaled, roundoff_floor


class CommitPhase(str, Enum):
    RADIATION_PHYSICS = "phase1_radiation_physics"
    MICROPHYSICS = "phase2_microphysics"


@dataclass(frozen=True)
class ThermodynamicCommit:
    """Cell thermodynamics plus unreduced, owned-cell conservation operands."""

    atmosphere: NativeAtmosphere
    rtheta_p: torch.Tensor
    dry_layer_mass: torch.Tensor
    entry_layer_mass: torch.Tensor
    mass_scale: torch.Tensor
    # Ledger operands; None when the commit runs without ledgers (observers off).
    proposed_tracers: torch.Tensor | None
    limiter_correction: torch.Tensor | None
    native_tracer_increment: torch.Tensor | None
    ccpp_tracer_increment: torch.Tensor | None


def _commit_layer_masses(rho, rdzw, area, prsi, gravity, pressure_thickness=None):
    dry = rho / rdzw[None, :] * area[:, None]
    dp = (prsi[:, :-1] - prsi[:, 1:]).abs() if pressure_thickness is None else pressure_thickness
    return dry, dp / gravity * area[:, None]


def commit_thermodynamics(
    native: NativeAtmosphere,
    mesh: ProjectionMesh,
    statein: Statein,
    stateout: Stateout,
    *,
    area_cell: torch.Tensor,
    rtheta_base: torch.Tensor,
    rtheta_p: torch.Tensor,
    phase: CommitPhase,
    tracers: TracerContract,
    constants: ConstantsProfile = UFS_CONSTANTS,
    ledgers: bool = True,
    pressure_thickness: torch.Tensor | None = None,
) -> ThermodynamicCommit:
    """Port the original cell loop, before edge momentum and diagnostic refresh.

    Layer masses retain the source's coordinate thickness and expression order.
    The first phase applies absolute Stateout on entry GFS pressure mass. TEMPO
    returns its completed state on post-process dry-plus-vapor mass instead.
    The returned ledger operands have not undergone a global/MPI reduction.
    With ``ledgers=False`` they are not formed (None); the committed state is unchanged.
    Experimental ``pressure_thickness`` retains the live hydrostatic producer's
    layer pressure before interface rounding. Only phase one admits that representation.
    The normal coordinator does not supply it.
    """
    if not isinstance(phase, CommitPhase):
        raise ValueError("native commit requires an explicit CommitPhase")
    if native.rho_zz.ndim != 2:
        raise ValueError("native commit requires column/level rho_zz")
    count, levels = native.rho_zz.shape
    n = native.n_cells_solve
    if type(n) is not int or not 0 < n <= count:
        raise ValueError("invalid native commit owned extent")
    device = native.rho_zz.device
    for name, value in (
        ("rho_zz", native.rho_zz),
        ("scalars", native.scalars),
        ("zz", mesh.zz),
        ("rdzw", mesh.rdzw),
        ("area_cell", area_cell),
        ("rtheta_base", rtheta_base),
        ("rtheta_p", rtheta_p),
        ("statein.prsi", statein.prsi),
        ("statein.qgrs", statein.qgrs),
    ):
        _working_on_device(name, value, device)
    if native.scalars.ndim != 3 or native.scalars.shape[:2] != (count, levels):
        raise ValueError("native scalar extents do not match commit atmosphere")
    if (
        mesh.zz.shape != (count, levels)
        or mesh.rdzw.shape != (levels,)
        or area_cell.shape != (count,)
    ):
        raise ValueError("invalid native commit mass geometry")
    if rtheta_base.shape != (count, levels) or rtheta_p.shape != (count, levels):
        raise ValueError("invalid native density-weighted thermodynamic state")
    for name in ("theta", "theta_m", "exner", "pressure_base", "pressure_p"):
        value = getattr(native, name)
        _working_on_device(name, value, device)
        if value.shape != (count, levels):
            raise ValueError(f"invalid native thermodynamic extent: {name}")
    for field in fields(stateout):
        value = getattr(stateout, field.name)
        _working_on_device("stateout." + field.name, value, device)
        if (
            value.ndim != (3 if field.name == "gq0" else 2)
            or value.shape[0] < n
            or value.shape[1] != levels
        ):
            raise ValueError(f"invalid native commit Stateout extent: {field.name}")
    n_apply = tracers.validate(
        native_count=native.scalars.shape[2], target_count=stateout.gq0.shape[2]
    )
    bindings = tracers.bindings[:n_apply]
    if bindings[tracers.ccpp_qv_index].mpas_index != tracers.native_qv_index:
        raise ValueError("CCPP vapor must map to the native vapor scalar")
    owned_slots = [binding.mpas_index for binding in bindings if binding.ccpp_owned]
    if len(set(owned_slots)) != len(owned_slots):
        raise ValueError(
            "selected reverse bridge requires unique owned scalar bindings"
        )
    if (
        statein.prsi.ndim != 2
        or statein.qgrs.ndim != 3
        or statein.prsi.shape[0] < n
        or statein.prsi.shape[1:] != (levels + 1,)
        or statein.qgrs.shape[0] < n
        or statein.qgrs.shape[1] != levels
        or statein.qgrs.shape[2] < n_apply
    ):
        raise ValueError("CCPP entry state does not cover native commit")

    rho = native.rho_zz[:n]
    if pressure_thickness is not None:
        if (not isinstance(pressure_thickness, torch.Tensor)
                or phase is not CommitPhase.RADIATION_PHYSICS or pressure_thickness.shape != rho.shape
                or pressure_thickness.dtype != rho.dtype or pressure_thickness.device != rho.device
                or statein.prsi.dtype != rho.dtype):
            raise ValueError("unrounded layer pressure requires phase-one native extents, dtype and device")
        producer_dp = constants.G * rho / mesh.rdzw * (
            1.0 + native.scalars[:n, :, tracers.native_qv_index])
        torch._assert_async((pressure_thickness == producer_dp).all(),
                            "unrounded layer pressure differs from its native hydrostatic producer")
        # p[k] = fl(p[k+1] + dp[k]); bound the two subsequent subtractions too.
        u = torch.finfo(rho.dtype).eps / 2
        radius = (3 * u / (1 - 3 * u)) * (
            statein.prsi[:n, :-1].abs() + statein.prsi[:n, 1:].abs() + pressure_thickness.abs())
        difference = (statein.prsi[:n, :-1] - statein.prsi[:n, 1:]).abs()
        torch._assert_async(torch.isfinite(statein.prsi[:n]).all() & torch.isfinite(radius).all()
                            & ((difference - pressure_thickness).abs() <= radius).all(),
                            "unrounded layer pressure disagrees with stored interfaces")
    dry_mass, entry_mass = _commit_layer_masses(
        rho, mesh.rdzw, area_cell[:n], statein.prsi[:n], constants.G
    )
    torch._assert_async(
        (
            torch.isfinite(dry_mass)
            & (dry_mass > 0.0)
            & torch.isfinite(entry_mass)
            & (entry_mass > 0.0)
        ).all(),
        "invalid native/CCPP commit layer masses",
    )
    scale = entry_mass / dry_mass
    expected_scale = 1.0 + native.scalars[:n, :, tracers.native_qv_index]
    torch._assert_async(
        (torch.abs(scale - expected_scale) <= eps_scaled(1.0e-10, scale.dtype) * expected_scale).all(),
        "CCPP pressure mass disagrees with native dry-plus-vapor mass",
    )
    if pressure_thickness is not None:
        _, entry_mass = _commit_layer_masses(
            rho, mesh.rdzw, area_cell[:n], statein.prsi[:n], constants.G, pressure_thickness)
        scale = entry_mass / dry_mass
        torch._assert_async((torch.isfinite(entry_mass) & (entry_mass > 0.0)
            & ((scale - expected_scale).abs() <= eps_scaled(1.0e-10, scale.dtype) * expected_scale)).all(),
            "invalid unrounded native/CCPP commit layer mass")
    vapor_out = stateout.gq0[:n, :, tracers.ccpp_qv_index]
    torch._assert_async(
        (torch.isfinite(vapor_out) & (vapor_out < 1.0)).all(),
        "invalid provisional CCPP vapor",
    )
    temperature = stateout.gt0[:n]
    torch._assert_async(
        (
            torch.isfinite(temperature) & (temperature > 0.0) & (temperature <= 2000.0)
        ).all(),
        "invalid CCPP temperature",
    )
    for name in ("gu0", "gv0"):
        value = getattr(stateout, name)[:n]
        torch._assert_async(
            (torch.isfinite(value) & (torch.abs(value) <= 10000.0)).all(),
            "invalid CCPP wind",
        )

    values_by_native_slot = {}
    proposed, correction, native_increment, ccpp_increment = [], [], [], []
    for index, binding in enumerate(bindings):
        old = native.scalars[:n, :, binding.mpas_index]
        if not binding.ccpp_owned:
            if not ledgers:
                continue
            proposed.append(old)
            correction.append(torch.zeros_like(old))
            native_increment.append(torch.zeros_like(old))
            ccpp_increment.append(torch.zeros_like(old))
            continue
        value = stateout.gq0[:n, :, index]
        if ledgers:
            ccpp_increment.append((value - statein.qgrs[:n, :, index]) * entry_mass)
        if index == tracers.ccpp_qv_index:
            value = (
                value / (1.0 - value)
                if phase is CommitPhase.MICROPHYSICS
                else value * scale
            )
        elif binding.uses_dry_plus_vapor_mass_basis:
            value = (
                value / torch.clamp_min(1.0 - vapor_out, 1.0e-12)
                if phase is CommitPhase.MICROPHYSICS
                else value * scale
            )
        torch._assert_async(
            (
                torch.isfinite(value)
                & (torch.abs(value) <= 1.0e30)
                & (value >= roundoff_floor(binding.negative_floor, value))
            ).all(),
            "invalid converted CCPP tracer",
        )
        # Floor after conversion; other species still use raw provisional vapor.
        floored = torch.where(value < 0.0, torch.zeros_like(value), value)
        if ledgers:
            proposed.append(value)
            correction.append(torch.where(value < 0.0, -value, torch.zeros_like(value)))
            native_increment.append((floored - old) * dry_mass)
        values_by_native_slot[binding.mpas_index] = floored
    columns = [
        values_by_native_slot.get(index, native.scalars[:n, :, index])
        for index in range(native.scalars.shape[2])
    ]
    scalars_owned = torch.stack(columns, dim=2)
    moist_factor = 1.0 + constants.RVORD * scalars_owned[:, :, tracers.native_qv_index]
    pressure = rho * mesh.zz[:n] * constants.RD * temperature * moist_factor
    torch._assert_async(
        (torch.isfinite(pressure) & (pressure > 0.0)).all(),
        "invalid reconstructed pressure",
    )
    exner = _pressure_power(pressure, constants)
    theta = temperature / exner
    theta_m = theta * moist_factor
    rp = rho * theta_m - rtheta_base[:n]
    pp = pressure - native.pressure_base[:n]
    residual = torch.abs(theta * exner - temperature)
    torch._assert_async(
        (
            residual
            <= 128.0
            * torch.finfo(temperature.dtype).eps
            * torch.clamp_min(torch.abs(temperature), 1.0)
        ).all(),
        "post-commit temperature inconsistency",
    )
    for value in (exner, theta, theta_m, pp, rp):
        torch._assert_async(
            torch.isfinite(value).all(), "nonfinite committed thermodynamics"
        )

    def owned(value: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
        return torch.cat((value, old[n:]), dim=0)

    atmosphere = replace(
        native,
        scalars=owned(scalars_owned, native.scalars),
        theta=owned(theta, native.theta),
        theta_m=owned(theta_m, native.theta_m),
        exner=owned(exner, native.exner),
        pressure_p=owned(pp, native.pressure_p),
    )
    return ThermodynamicCommit(
        atmosphere,
        owned(rp, rtheta_p),
        dry_mass,
        entry_mass,
        scale,
        *((torch.stack(proposed, dim=2), torch.stack(correction, dim=2),
           torch.stack(native_increment, dim=2), torch.stack(ccpp_increment, dim=2))
          if ledgers else (None, None, None, None)),
    )
