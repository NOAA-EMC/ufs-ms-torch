# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Port of R4 atmos_coupling.F90 atmospheric projection, without surface reset."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
from mpas_pytorch.constants import UFS_CONSTANTS, ConstantsProfile
from mpas_pytorch.diagnostics import physics_interface_surface_pressure
from ufs_physics_common.precision import working_power

from .containers import NativeAtmosphere, ProjectionMesh, Statein, static_tensor
from .precision import WORKING_DTYPES, roundoff_floor


_TRACER_REDUCTION_ROWS = 8192


def _pressure_power(pressure, constants):
    return working_power(pressure / constants.P0, constants.RD / constants.CP)


def _tracer_reduction_tiles(values, *, minimum):
    flat = values.reshape(-1, values.shape[-1])
    full = flat.shape[0] // _TRACER_REDUCTION_ROWS * _TRACER_REDUCTION_ROWS
    if full == 0:
        return flat.amin(dim=0) if minimum else flat.sum(dim=0)
    tiles = flat[:full].reshape(-1, _TRACER_REDUCTION_ROWS, flat.shape[-1])
    partial = tiles.amin(dim=1) if minimum else tiles.sum(dim=1)
    if full < flat.shape[0]:
        tail = flat[full:].amin(dim=0) if minimum else flat[full:].sum(dim=0)
        partial = torch.cat((partial, tail[None]), dim=0)
    return partial.amin(dim=0) if minimum else partial.sum(dim=0)


class _TracerMinimum(torch.autograd.Function):
    """Tiled primal reduction with the original global ``amin`` tie derivative."""

    generate_vmap_rule = True

    @staticmethod
    def forward(values):
        return _tracer_reduction_tiles(values, minimum=True)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0], output)
        ctx.save_for_forward(inputs[0], output)

    @staticmethod
    def backward(ctx, grad):
        values, minimum = ctx.saved_tensors
        selected = values == minimum
        return (grad / selected.sum(dim=(0, 1), keepdim=True)) * selected

    @staticmethod
    def jvp(ctx, tangent):
        values, minimum = ctx.saved_tensors
        selected = values == minimum
        return (tangent * selected).sum(dim=(0, 1)) / selected.sum(dim=(0, 1))


@torch.compiler.allow_in_graph
def _tracer_minimum(values):
    return _TracerMinimum.apply(values)


def _negative_tracer_statistics(values, owned):
    negative = (values < 0.0) & owned
    counts = _tracer_reduction_tiles(negative, minimum=False)
    minima = _tracer_minimum(torch.where(negative, values, 0.0))
    return negative, counts, minima


@dataclass(frozen=True)
class TracerBinding:
    """Static constituent metadata and its zero-based native scalar index."""

    name: str
    units: str
    mpas_index: int
    ccpp_owned: bool

    @property
    def uses_dry_plus_vapor_mass_basis(self) -> bool:
        return self.name != "qv" and "kg-1" in self.units.strip()

    @property
    def negative_floor(self) -> float:
        return -1.0e-8 if self.name in ("ni", "nr") else -1.0e-10


@dataclass(frozen=True)
class TracerContract:
    """Source map and qv indices; metadata contains no live tensor controls."""

    bindings: tuple[TracerBinding, ...]
    ccpp_qv_index: int
    native_qv_index: int

    def validate(self, *, native_count: int, target_count: int) -> int:
        n_apply = min(native_count, target_count)
        if len(self.bindings) < n_apply:
            raise ValueError("constituent-to-MPAS scalar map is too small")
        if not 0 <= self.ccpp_qv_index < n_apply:
            raise ValueError("selected atmospheric projection requires a mapped qv")
        if not 0 <= self.native_qv_index < native_count:
            raise ValueError("invalid native qv scalar index")
        if not any(binding.ccpp_owned for binding in self.bindings[:n_apply]):
            raise ValueError("no MPAS tracers map to CCPP-owned fields")
        for binding in self.bindings[:n_apply]:
            if (
                type(binding.mpas_index) is not int
                or not 0 <= binding.mpas_index < native_count
            ):
                raise ValueError("invalid constituent-to-MPAS scalar index")
            if type(binding.ccpp_owned) is not bool:
                raise TypeError("tracer ownership must be static")
        return n_apply


def r4_tracer_contract() -> TracerContract:
    """Executed R4 tracer_table/map and post-init nt* controls: all ten owned."""
    names = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "sgs_tke", "o3mr")
    units = ("kg kg-1",) * 6 + ("kg-1", "kg-1", "m2 s-2", "kg kg-1")
    return TracerContract(
        tuple(
            TracerBinding(name, unit, index, True)
            for index, (name, unit) in enumerate(zip(names, units))
        ),
        ccpp_qv_index=0,
        native_qv_index=0,
    )


@dataclass(frozen=True)
class HydrostaticPressure:
    pmiddry: torch.Tensor
    pintdry: torch.Tensor
    pmid: torch.Tensor
    pint: torch.Tensor


@dataclass(frozen=True)
class AtmosphericProjection:
    statein: Statein
    surface_pressure: torch.Tensor
    floored_tracer_count: torch.Tensor
    floored_tracer_min: torch.Tensor


def _working_on_device(name: str, value: torch.Tensor, device: torch.device, dtype: torch.dtype | None = None) -> None:
    """``value`` is a tensor on ``device`` in ``dtype`` (default: either working dtype, float32 or float64)."""
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype not in ((dtype,) if dtype is not None else WORKING_DTYPES)
        or value.device != device
    ):
        raise ValueError(f"{name} must be {dtype or 'float32/float64'} on {device}")


def hydrostatic_pressure(
    *,
    rho_zz: torch.Tensor,
    theta_m: torch.Tensor,
    exner: torch.Tensor,
    qv: torch.Tensor,
    rdzw: torch.Tensor,
    zgrid: torch.Tensor,
    constants: ConstantsProfile = UFS_CONSTANTS,
) -> HydrostaticPressure:
    """Match source finite-volume mass and ordered downward pressure integration."""
    if rho_zz.ndim != 2 or rho_zz.shape[1] < 1:
        raise ValueError("hydrostatic pressure requires a nonempty level axis")
    shape = rho_zz.shape
    for name, value in (("theta_m", theta_m), ("exner", exner), ("qv", qv)):
        if value.shape != shape:
            raise ValueError(f"{name} shape does not match rho_zz")
    if rdzw.shape != (shape[1],) or zgrid.shape != (shape[0], shape[1] + 1):
        raise ValueError("hydrostatic coordinate thickness/height extents do not match")
    for name, value in (
        ("rho_zz", rho_zz),
        ("theta_m", theta_m),
        ("exner", exner),
        ("qv", qv),
        ("rdzw", rdzw),
        ("zgrid", zgrid),
    ):
        _working_on_device(name, value, rho_zz.device, rho_zz.dtype)
        torch._assert_async(
            torch.isfinite(value).all(), f"nonfinite hydrostatic {name}"
        )
    dz = zgrid[:, 1:] - zgrid[:, :-1]
    torch._assert_async((dz > 0.0).all(), "nonpositive geometric layer thickness")
    torch._assert_async(
        (rdzw > 0.0).all(), "nonpositive coordinate reciprocal thickness"
    )
    torch._assert_async((rho_zz > 0.0).all(), "nonpositive dry mass density")
    torch._assert_async((qv > -1.0).all(), "invalid hydrostatic water-vapor mass basis")

    dpdry = constants.G * rho_zz / rdzw
    water_mass = 1.0 + qv
    dp = dpdry * water_mass
    tv = (theta_m / water_mass) * exner
    temperature = tv * water_mass / (1.0 + constants.RVORD * qv)
    pmid_raw = dp * constants.RD * tv / (constants.G * dz)
    pmiddry_raw = dpdry * constants.RD * temperature / (constants.G * dz)
    scale = constants.RD * tv[:, -1] / constants.G
    dry_scale = constants.RD * temperature[:, -1] / constants.G
    scale = torch.where(scale >= 1.0, scale, 1.0)
    dry_scale = torch.where(dry_scale >= 1.0, dry_scale, 1.0)
    top = pmid_raw[:, -1] * torch.exp(-0.5 * dz[:, -1] / scale)
    dry_top = pmiddry_raw[:, -1] * torch.exp(-0.5 * dz[:, -1] / dry_scale)
    top = torch.where(top >= 1.0e-3, top, 1.0e-3)
    dry_top = torch.where(dry_top >= 1.0e-3, dry_top, 1.0e-3)

    # The source adds one layer at a time from the top; cumsum changes association, so the two
    # interface sums stay ordered. The midpoint bounds are elementwise per level (vectorized).
    interfaces, dry_interfaces = [top], [dry_top]
    for level in range(shape[1] - 1, -1, -1):
        interfaces.append(interfaces[-1] + dp[:, level])
        dry_interfaces.append(dry_interfaces[-1] + dpdry[:, level])
    pint = torch.stack(interfaces[::-1], dim=1)
    pintdry = torch.stack(dry_interfaces[::-1], dim=1)

    def bounded_midpoint(raw, interface, thickness):
        # Level k lies between interface k (below) and k + 1 (above).
        eps = thickness * 0.05
        upper_limit = interface[:, :-1] - eps
        bounded = torch.where(raw <= upper_limit, raw, upper_limit)
        lower_limit = interface[:, 1:] + eps
        return torch.where(bounded >= lower_limit, bounded, lower_limit)

    return HydrostaticPressure(
        pmiddry=bounded_midpoint(pmiddry_raw, pintdry, dpdry),
        pintdry=pintdry,
        pmid=bounded_midpoint(pmid_raw, pint, dp),
        pint=pint,
    )


def _validate_projection_shapes(
    native: NativeAtmosphere, mesh: ProjectionMesh, target: Statein
) -> None:
    if native.rho_zz.ndim != 2:
        raise ValueError("native atmospheric fields must have cell-by-level axes")
    cells, levels = native.rho_zz.shape
    owned = native.n_cells_solve
    if type(owned) is not int or not 0 < owned <= cells or levels < 2:
        raise ValueError(
            "projection requires an explicit owned count and at least two levels"
        )
    for field in fields(native):
        if field.name == "n_cells_solve":
            continue
        value = getattr(native, field.name)
        _working_on_device(field.name, value, native.rho_zz.device, native.rho_zz.dtype)
        expected = (cells, levels)
        if field.name == "w":
            expected = (cells, levels + 1)
        elif field.name == "surface_pressure":
            expected = (cells,)
        elif field.name == "scalars":
            if value.ndim != 3 or value.shape[:2] != expected or value.shape[2] < 1:
                raise ValueError("native scalars must have cell, level and tracer axes")
            continue
        if value.shape != expected:
            raise ValueError(f"native {field.name} has the wrong shape")
    for name, shape in (
        ("zz", (cells, levels)),
        ("zgrid", (cells, levels + 1)),
        ("rdzw", (levels,)),
    ):
        value = getattr(mesh, name)
        _working_on_device(name, value, native.rho_zz.device, native.rho_zz.dtype)
        if value.shape != shape:
            raise ValueError(f"mesh {name} has the wrong shape")
    if (
        target.qgrs.ndim != 3
        or target.qgrs.shape[0] < owned
        or target.qgrs.shape[1] != levels
        or target.qgrs.shape[2] < 1
    ):
        raise ValueError("Statein qgrs extents do not cover owned atmospheric cells")
    target_cells = target.qgrs.shape[0]
    for field in fields(target):
        value = getattr(target, field.name)
        if field.name == "wgrs" and value is None:
            continue
        _working_on_device(f"Statein.{field.name}", value, native.rho_zz.device, native.rho_zz.dtype)
        expected = (target_cells, levels)
        if field.name in ("phii", "prsi", "prsik"):
            expected = (target_cells, levels + 1)
        elif field.name == "pgr":
            expected = (target_cells,)
        elif field.name == "qgrs":
            continue
        elif field.name in ("smc", "stc", "slc"):
            if value.ndim != 2 or value.shape[0] != target_cells or value.shape[1] < 1:
                raise ValueError(f"Statein.{field.name} has the wrong soil shape")
            continue
        if value.shape != expected:
            raise ValueError(f"Statein.{field.name} has the wrong shape")
    if target.smc.shape != target.stc.shape or target.smc.shape != target.slc.shape:
        raise ValueError("Statein soil extents must match")


def project_atmosphere(
    native: NativeAtmosphere,
    mesh: ProjectionMesh,
    target: Statein,
    *,
    tracers: TracerContract,
    constants: ConstantsProfile = UFS_CONSTANTS,
) -> AtmosphericProjection:
    """Reproject current atmosphere; preserve surface state and unwritten members."""
    _validate_projection_shapes(native, mesh, target)
    n_apply = tracers.validate(
        native_count=native.scalars.shape[2], target_count=target.qgrs.shape[2]
    )
    owned = native.n_cells_solve
    raw = native.scalars[:owned]
    qv = raw[..., tracers.native_qv_index]
    mapped_qv = raw[..., tracers.bindings[tracers.ccpp_qv_index].mpas_index]
    qv_denominator = torch.where(
        torch.isfinite(mapped_qv) & (mapped_qv >= 0.0), mapped_qv, 0.0
    )
    # All applied tracers at once; each element sees the same operations as the per-tracer
    # source loop (static per-tracer selections). The asserts stay per tracer (named messages).
    bindings = tracers.bindings[:n_apply]
    device = raw.device
    values = raw.index_select(2, static_tensor(tuple(b.mpas_index for b in bindings), dtype=torch.int64, device=device))
    for index, binding in enumerate(bindings):
        if binding.ccpp_owned:
            value = values[..., index]
            torch._assert_async(
                (torch.isfinite(value) & (value.abs() <= 1.0e30)).all(),
                f"invalid MPAS tracer {binding.name}",
            )
            torch._assert_async(
                (value >= roundoff_floor(binding.negative_floor, value)).all(),
                f"negative MPAS tracer {binding.name}",
            )
    owned_tracer = static_tensor(tuple(b.ccpp_owned for b in bindings), dtype=torch.bool, device=device)
    negative, counts, minima = _negative_tracer_statistics(values, owned_tracer)
    values = torch.where(negative, 0.0, values)
    is_qv = static_tensor(tuple(i == tracers.ccpp_qv_index for i in range(n_apply)), dtype=torch.bool, device=device)
    dry_basis = static_tensor(tuple(i != tracers.ccpp_qv_index and b.uses_dry_plus_vapor_mass_basis
                                    for i, b in enumerate(bindings)), dtype=torch.bool, device=device)
    qgrs = torch.where(is_qv, values / (1.0 + values),
                       torch.where(dry_basis, values / (1.0 + qv_denominator[..., None]), values))
    if n_apply < target.qgrs.shape[2]:
        qgrs = torch.cat((qgrs, target.qgrs[:owned, :, n_apply:]), dim=2)

    rho_zz = native.rho_zz[:owned]
    zgrid = mesh.zgrid[:owned]
    zz = mesh.zz[:owned]
    hydrostatic = hydrostatic_pressure(
        rho_zz=rho_zz,
        theta_m=native.theta_m[:owned],
        exner=native.exner[:owned],
        qv=qv,
        rdzw=mesh.rdzw,
        zgrid=zgrid,
        constants=constants,
    )
    for name, value in (("prsl", hydrostatic.pmid), ("prsi", hydrostatic.pint)):
        torch._assert_async(
            (torch.isfinite(value) & (value > 0.0)).all(), f"invalid projected {name}"
        )
    surface_pressure = physics_interface_surface_pressure(
        nCells=owned,
        rho_zz=rho_zz,
        zz=zz,
        qv=qv,
        pressure_p=native.pressure_p[:owned],
        pressure_base=native.pressure_base[:owned],
        zgrid=zgrid,
        gravity=constants.G,
    )
    height = zgrid - zgrid[:, :1]
    w_layer = 0.5 * (native.w[:owned, :-1] + native.w[:owned, 1:])
    values = {
        "tgrs": native.theta[:owned] * native.exner[:owned],
        "ugrs": native.u_reconstruct_zonal[:owned],
        "vgrs": native.u_reconstruct_meridional[:owned],
        "phii": constants.G * height,
        "phil": 0.5 * constants.G * (height[:, :-1] + height[:, 1:]),
        "vvl": -(rho_zz * zz * (1.0 + qv)) * constants.G * w_layer,
        "pgr": surface_pressure,
        "prsl": hydrostatic.pmid,
        "prsi": hydrostatic.pint,
        "prslk": _pressure_power(hydrostatic.pmid, constants),
        "prsik": _pressure_power(hydrostatic.pint, constants),
        "qgrs": qgrs,
    }
    if target.wgrs is not None:
        values["wgrs"] = w_layer
    # Unowned rows (halo) keep their previous values; a fully owned domain needs no copy.
    full = owned == target.tgrs.shape[0]
    for name, active in values.items():
        torch._assert_async(torch.isfinite(active).all(), f"nonfinite projected {name}")
        if not full:
            values[name] = torch.cat((active, getattr(target, name)[owned:]), dim=0)
    return AtmosphericProjection(
        statein=replace(target, **values),
        surface_pressure=surface_pressure if owned == native.surface_pressure.shape[0] else torch.cat(
            (surface_pressure, native.surface_pressure[owned:]), dim=0
        ),
        floored_tracer_count=counts,
        floored_tracer_min=minima,
    )
