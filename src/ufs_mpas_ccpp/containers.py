# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Source-shaped atmospheric containers, with explicit native owned extents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.autograd.forward_ad as forward_ad


def static_tensor(values, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """Device copy of a static host tuple, built once per values, dtype and device.

    A per-call host tuple costs a pageable copy and a stream sync on CUDA and
    blocks graph capture. Callers must not modify the result in place.
    """
    if torch.compiler.is_compiling():  # traced as a graph constant; no cache lookup
        return torch.tensor(values, dtype=dtype, device=device)
    return _static_tensor(_freeze(values), dtype, torch.device(device), torch.is_inference_mode_enabled())


def _freeze(values):
    return tuple(_freeze(v) for v in values) if isinstance(values, (list, tuple)) else values


def _finite_flags(primals):
    # Return row flags across the compiled boundary. A fused global reduction
    # otherwise scans every field with one CUDA block on the 60km workload.
    return tuple(torch.isfinite(value).all(dim=tuple(range(1, value.ndim)))
                 if value.ndim > 1 else torch.isfinite(value).reshape(-1)
                 for value in primals)


def assert_all_finite(values: Mapping[str, torch.Tensor], label: str, *, exclude: tuple[str, ...] = (),
                      compile_check: bool = False) -> None:
    """One device assertion for many fields; the message lists the checked fields.

    ``exclude`` names fields not checked here (the pass-through ``dtend`` diagnostic array, whose
    full read would dominate the check). CUDA rejects assertion messages of 255 characters or
    more, so there the list is cut short.
    """
    names = [name for name, value in values.items()
             if value.is_floating_point() and value.numel() and name not in exclude]
    if names:
        with torch.no_grad():
            # Aliased names need one scan, but every distinct view still needs coverage.
            checked = (tuple(values[name] for name in names) if torch.compiler.is_compiling()
                       else tuple({id(values[name]): values[name] for name in names}.values()))
            primals = tuple(forward_ad.unpack_dual(value).primal for value in checked)
            if compile_check and not torch.compiler.is_compiling():
                from ufs_physics_common.compile_policy import compiled

                rows = compiled(_finite_flags, normalize_layouts=False)(primals)
                flags = torch.cat(rows).all()
            else:
                extrema = [bound for value in primals for bound in torch.aminmax(value)]
                flags = torch.isfinite(torch.stack(extrema)).all()
        message = f"nonfinite {label} (one of: {', '.join(names)})"
        cut = flags.is_cuda and len(message) >= 255
        # Keep the assertion outside the compiled predicate, independent of compiled-assert policy.
        torch._assert_async(flags, message[:250] + "...)" if cut else message)


def _exact(values):
    """Cache key that keeps -0.0 apart from 0.0 and matches NaN to itself."""
    if isinstance(values, tuple):
        return tuple(_exact(v) for v in values)
    return values.hex() if isinstance(values, float) else values


_STATIC_TENSORS: dict = {}


def _static_tensor(values, dtype, device, inference):
    key = (_exact(values), dtype, device, inference)
    cached = _STATIC_TENSORS.get(key)
    if cached is None:
        cached = _STATIC_TENSORS[key] = torch.tensor(values, dtype=dtype, device=device)
    return cached


@dataclass(frozen=True)
class NativeAtmosphere:
    """Current TL1 primaries and diagnostics in cell-major, bottom-up layout."""

    n_cells_solve: int
    rho_zz: torch.Tensor
    theta_m: torch.Tensor
    scalars: torch.Tensor
    w: torch.Tensor
    theta: torch.Tensor
    exner: torch.Tensor
    u_reconstruct_zonal: torch.Tensor
    u_reconstruct_meridional: torch.Tensor
    pressure_base: torch.Tensor
    pressure_p: torch.Tensor
    surface_pressure: torch.Tensor

    @classmethod
    def from_mpas_state(
        cls, state: Mapping[str, torch.Tensor], *, n_cells_solve: int
    ) -> NativeAtmosphere:
        """Bind live canonical fields; never use the potentially stale qv alias."""
        return cls(
            n_cells_solve=n_cells_solve,
            rho_zz=state["rho_zz"],
            theta_m=state["theta_m"],
            scalars=state["scalars"],
            w=state["w"],
            theta=state["theta"],
            exner=state["exner"],
            u_reconstruct_zonal=state["uReconstructZonal"],
            u_reconstruct_meridional=state["uReconstructMeridional"],
            pressure_base=state["pressure_base"],
            pressure_p=state["pressure_p"],
            surface_pressure=state["surface_pressure"],
        )


@dataclass(frozen=True)
class ProjectionMesh:
    """MPAS geometry; rdzw is coordinate thickness, not geometric thickness."""

    zz: torch.Tensor
    zgrid: torch.Tensor
    rdzw: torch.Tensor

    @classmethod
    def from_mpas_state(cls, mesh: Mapping[str, torch.Tensor]) -> ProjectionMesh:
        return cls(zz=mesh["zz"], zgrid=mesh["zgrid"], rdzw=mesh["rdzw"])


@dataclass(frozen=True)
class Statein:
    """GFS_statein_type members; projection leaves deformation/soil state intact."""

    phii: torch.Tensor
    prsi: torch.Tensor
    prsik: torch.Tensor
    phil: torch.Tensor
    prsl: torch.Tensor
    prslk: torch.Tensor
    pgr: torch.Tensor
    ugrs: torch.Tensor
    vgrs: torch.Tensor
    vvl: torch.Tensor
    tgrs: torch.Tensor
    qgrs: torch.Tensor
    def_1: torch.Tensor
    def_2: torch.Tensor
    def_3: torch.Tensor
    diss_est: torch.Tensor
    smc: torch.Tensor
    stc: torch.Tensor
    slc: torch.Tensor
    wgrs: torch.Tensor | None

    @classmethod
    def allocate(
        cls,
        *,
        ncol: int,
        nlev: int,
        ntrac: int,
        soil_levels: int,
        lightning_threat: bool,
        device: torch.device | str,
        dtype: torch.dtype = torch.float64,
    ) -> Statein:
        """Match GFS_typedefs.F90 statein_create and its clear_val=zero."""
        if any(
            type(value) is not int or value <= 0
            for value in (ncol, nlev, ntrac, soil_levels)
        ):
            raise ValueError("Statein allocation requires positive integer extents")
        if type(lightning_threat) is not bool:
            raise TypeError("lightning_threat must be a static bool")

        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            phii=zeros(ncol, nlev + 1),
            prsi=zeros(ncol, nlev + 1),
            prsik=zeros(ncol, nlev + 1),
            phil=zeros(ncol, nlev),
            prsl=zeros(ncol, nlev),
            prslk=zeros(ncol, nlev),
            pgr=zeros(ncol),
            ugrs=zeros(ncol, nlev),
            vgrs=zeros(ncol, nlev),
            vvl=zeros(ncol, nlev),
            tgrs=zeros(ncol, nlev),
            qgrs=zeros(ncol, nlev, ntrac),
            def_1=zeros(ncol, nlev),
            def_2=zeros(ncol, nlev),
            def_3=zeros(ncol, nlev),
            diss_est=zeros(ncol, nlev),
            smc=zeros(ncol, soil_levels),
            stc=zeros(ncol, soil_levels),
            slc=zeros(ncol, soil_levels),
            wgrs=zeros(ncol, nlev) if lightning_threat else None,
        )


@dataclass(frozen=True)
class Stateout:
    """All four GFS_stateout_type members, in source column-first layout."""

    gu0: torch.Tensor
    gv0: torch.Tensor
    gt0: torch.Tensor
    gq0: torch.Tensor

    @classmethod
    def allocate(
        cls, *, ncol: int, nlev: int, ntrac: int, device: torch.device | str,
        dtype: torch.dtype = torch.float64,
    ) -> Stateout:
        if any(type(value) is not int or value <= 0 for value in (ncol, nlev, ntrac)):
            raise ValueError("Stateout allocation requires positive integer extents")
        return cls(
            gu0=torch.zeros((ncol, nlev), dtype=dtype, device=device),
            gv0=torch.zeros((ncol, nlev), dtype=dtype, device=device),
            gt0=torch.zeros((ncol, nlev), dtype=dtype, device=device),
            gq0=torch.zeros((ncol, nlev, ntrac), dtype=dtype, device=device),
        )

    @classmethod
    def from_statein(cls, statein: Statein) -> Stateout:
        """Functional equivalent of ufs_statein_to_stateout, before group execution."""
        return cls(
            statein.ugrs.clone(),
            statein.vgrs.clone(),
            statein.tgrs.clone(),
            statein.qgrs.clone(),
        )
