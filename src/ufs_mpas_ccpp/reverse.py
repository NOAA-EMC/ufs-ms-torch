# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""R4 native commit operators; no Fortran or host-valued tensor controls."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum

import torch
from mpas_pytorch import _torch_compat as torch_compat
from mpas_pytorch.constants import UFS_CONSTANTS, ConstantsProfile
from ufs_physics_common.compile_policy import stored

from .containers import NativeAtmosphere, ProjectionMesh, Statein, Stateout, static_tensor
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


def physics_temperature_to_native(temperature_in, temperature_out, vapor_dry, gas_mass_scale,
                                  *, constants=UFS_CONSTANTS):
    """Convert GFS Cp heating to MPAS constant-kappa internal-energy heating.

    GFS temperatures use dry-plus-vapor mass. MPAS internal energy has coefficient
    Cv*(1+Rv/Rd*rv) per dry mass (Eldred et al., 2022, constant-kappa system).
    ``gas_mass_scale`` is the frozen entry pressure mass divided by native dry
    mass, not the final vapor-dependent mass. Changing composition cannot change
    an independently supplied heat increment. This does not close composition
    exchanges or replace each scheme's caloric equation.
    """
    capacity = constants.CV * (1.0 + constants.RVORD * vapor_dry)
    scale = constants.CP * gas_mass_scale / capacity
    return temperature_in + scale * (temperature_out - temperature_in)


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
    The first phase applies tracer Stateout on entry GFS pressure mass. TEMPO
    returns its completed tracers on post-process dry-plus-vapor mass instead.
    Both phases convert GFS temperature increments to native internal-energy heating.
    The returned ledger operands have not undergone a global/MPI reduction.
    With ``ledgers=False`` they are not formed (None); the committed state is unchanged.
    Experimental ``pressure_thickness`` retains the live hydrostatic producer's
    layer pressure before interface rounding. Only phase one accepts that representation.
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
        ("statein.tgrs", statein.tgrs),
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
        or statein.tgrs.ndim != 2
        or statein.tgrs.shape[0] < n
        or statein.tgrs.shape[1] != levels
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
        torch_compat.assert_async((pressure_thickness == producer_dp).all(),
                            "unrounded layer pressure differs from its native hydrostatic producer")
        # p[k] = fl(p[k+1] + dp[k]); bound the two subsequent subtractions too.
        u = torch.finfo(rho.dtype).eps / 2
        radius = (3 * u / (1 - 3 * u)) * (
            statein.prsi[:n, :-1].abs() + statein.prsi[:n, 1:].abs() + pressure_thickness.abs())
        difference = (statein.prsi[:n, :-1] - statein.prsi[:n, 1:]).abs()
        torch_compat.assert_async(torch.isfinite(statein.prsi[:n]).all() & torch.isfinite(radius).all()
                            & ((difference - pressure_thickness).abs() <= radius).all(),
                            "unrounded layer pressure disagrees with stored interfaces")
    dry_mass, entry_mass = _commit_layer_masses(
        rho, mesh.rdzw, area_cell[:n], statein.prsi[:n], constants.G
    )
    torch_compat.assert_async(
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
    torch_compat.assert_async(
        (torch.abs(scale - expected_scale) <= eps_scaled(1.0e-10, scale.dtype) * expected_scale).all(),
        "CCPP pressure mass disagrees with native dry-plus-vapor mass",
    )
    if pressure_thickness is not None:
        _, entry_mass = _commit_layer_masses(
            rho, mesh.rdzw, area_cell[:n], statein.prsi[:n], constants.G, pressure_thickness)
        scale = entry_mass / dry_mass
        torch_compat.assert_async((torch.isfinite(entry_mass) & (entry_mass > 0.0)
            & ((scale - expected_scale).abs() <= eps_scaled(1.0e-10, scale.dtype) * expected_scale)).all(),
            "invalid unrounded native/CCPP commit layer mass")
    vapor_out = stateout.gq0[:n, :, tracers.ccpp_qv_index]
    torch_compat.assert_async(
        (torch.isfinite(vapor_out) & (vapor_out < 1.0)).all(),
        "invalid provisional CCPP vapor",
    )
    temperature = stateout.gt0[:n]
    torch_compat.assert_async(
        (
            torch.isfinite(temperature) & (temperature > 0.0) & (temperature <= 2000.0)
        ).all(),
        "invalid CCPP temperature",
    )
    for name in ("gu0", "gv0"):
        value = getattr(stateout, name)[:n]
        torch_compat.assert_async(
            (torch.isfinite(value) & (torch.abs(value) <= 10000.0)).all(),
            "invalid CCPP wind",
        )

    # All applied tracers at once, in CCPP order; each element sees the same operations as the per-tracer
    # source loop (static per-tracer selections; the divisor of an unconverted tracer is a finite placeholder).
    # The asserts and ledger operands stay per tracer.
    values = stateout.gq0[:n, :, :n_apply]
    is_qv = static_tensor(tuple(i == tracers.ccpp_qv_index for i in range(n_apply)), dtype=torch.bool, device=device)
    converts = static_tensor(tuple(i == tracers.ccpp_qv_index or b.uses_dry_plus_vapor_mass_basis
                                   for i, b in enumerate(bindings)), dtype=torch.bool, device=device)
    if phase is CommitPhase.MICROPHYSICS:
        divisor = torch.where(is_qv, 1.0 - values, torch.clamp_min(1.0 - vapor_out, 1.0e-12)[..., None])
        converted = torch.where(converts, values / divisor, values)
    else:
        converted = torch.where(converts, values * scale[..., None], values)
    # Floor after conversion; other species still use raw provisional vapor.
    floored = torch.where(converted < 0.0, 0.0, converted)
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
        value = converted[:, :, index]
        torch_compat.assert_async(
            (
                torch.isfinite(value)
                & (torch.abs(value) <= 1.0e30)
                & (value >= roundoff_floor(binding.negative_floor, value))
            ).all(),
            "invalid converted CCPP tracer",
        )
        if ledgers:
            ccpp_increment.append((stateout.gq0[:n, :, index] - statein.qgrs[:n, :, index]) * entry_mass)
            proposed.append(value)
            correction.append(torch.where(value < 0.0, -value, torch.zeros_like(value)))
            native_increment.append((floored[:, :, index] - old) * dry_mass)
    # Each native slot takes its owned CCPP tracer, else keeps its value; written once, contiguously.
    native_count = native.scalars.shape[2]
    source = [0] * native_count
    for index, binding in enumerate(bindings):
        if binding.ccpp_owned:
            source[binding.mpas_index] = index
    owned_slot = tuple(any(b.ccpp_owned and b.mpas_index == j for b in bindings) for j in range(native_count))
    if source != list(range(native_count)):
        floored = floored.index_select(2, static_tensor(tuple(source), dtype=torch.int64, device=device))
    scalars_owned = (floored if all(owned_slot) else torch.where(
        static_tensor(owned_slot, dtype=torch.bool, device=device), floored, native.scalars[:n]))
    vapor_dry = scalars_owned[:, :, tracers.native_qv_index]
    moist_factor = 1.0 + constants.RVORD * vapor_dry
    # Physics change: upstream MPAS-in-UFS sets the native temperature to the GFS temperature (Cp heating on the
    # GFS gas mass). Here the GFS increment becomes MPAS constant-kappa internal energy, Cv*(1+Rv/Rd*rv) per dry
    # mass, on the frozen physics entry mass, so the dycore keeps the heat each scheme supplies; the increment
    # scales by Cp*entry mass/(Cv*(1+Rv/Rd*rv)*dry mass). The Fortran reference has the same change
    # (the ``mpas_heat_capacity`` Fortran patch).
    temperature = physics_temperature_to_native(
        statein.tgrs[:n], temperature, vapor_dry, scale, constants=constants)
    torch_compat.assert_async((torch.isfinite(temperature) & (temperature > 0.0)
                         & (temperature <= 2000.0)).all(), "invalid native heating temperature")
    pressure = rho * mesh.zz[:n] * constants.RD * temperature * moist_factor
    torch_compat.assert_async(
        (torch.isfinite(pressure) & (pressure > 0.0)).all(),
        "invalid reconstructed pressure",
    )
    # Stored once (compiled): the halo-row cats are pointwise on CUDA, and Inductor evaluated the FP64 power again
    # in the cat kernel (one more FP64 power over [ncol, nlev] per commit).
    exner = stored(_pressure_power(pressure, constants))
    theta = temperature / exner
    theta_m = theta * moist_factor
    rp = rho * theta_m - rtheta_base[:n]
    pp = pressure - native.pressure_base[:n]
    residual = torch.abs(theta * exner - temperature)
    torch_compat.assert_async(
        (
            residual
            <= 128.0
            * torch.finfo(temperature.dtype).eps
            * torch.clamp_min(torch.abs(temperature), 1.0)
        ).all(),
        "post-commit temperature inconsistency",
    )
    for value in (exner, theta, theta_m, pp, rp):
        torch_compat.assert_async(
            torch.isfinite(value).all(), "nonfinite committed thermodynamics"
        )

    def owned(value: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
        # Unowned rows (halo) keep their previous values; a fully owned domain needs no copy.
        return value if n == count else torch.cat((value, old[n:]), dim=0)

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
