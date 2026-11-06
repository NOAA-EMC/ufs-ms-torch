# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Differentiable diagnostics derived from MPAS dynamics state."""

from __future__ import annotations

import torch

from mpas_pytorch.constants import GRAVITY, constants_for_config


_OUTPUT_DIAGNOSTIC_NAMES = ("theta", "rho", "pressure")


def refresh_output_diagnostics(result: dict, config: dict) -> None:
    """Refresh active cells while preserving native padding and unrelated carries."""
    fields = ("theta_m", "rho_zz", "scalars", "zz", "pressure_base", "pressure_p")
    if not all(isinstance(result.get(name), torch.Tensor) for name in fields):
        return
    n_cells = result.get("nCells", result["theta_m"].shape[0])
    if not isinstance(n_cells, int):
        raise TypeError("prepared nCells must be an integer")
    paddings = {}
    for name in _OUTPUT_DIAGNOSTIC_NAMES:
        original = result.get(name)
        paddings[f"{name}_padding"] = (
            original[n_cells:]
            if isinstance(original, torch.Tensor)
            else torch.zeros_like(result["theta_m"][n_cells:])
        )
    numeric = _output_diagnostics_compiled(config, result["theta_m"].device)
    outputs = (numeric or _output_diagnostics_padded_numeric)(
        **{name: result[name] for name in fields},
        **paddings,
        n_cells=n_cells,
        rvord=constants_for_config(config).RVORD,
    )
    for name, value in zip(_OUTPUT_DIAGNOSTIC_NAMES, outputs):
        result[name] = value


def _output_diagnostics_padded_numeric(
    *,
    theta_m: torch.Tensor,
    rho_zz: torch.Tensor,
    scalars: torch.Tensor,
    zz: torch.Tensor,
    pressure_base: torch.Tensor,
    pressure_p: torch.Tensor,
    theta_padding: torch.Tensor,
    rho_padding: torch.Tensor,
    pressure_padding: torch.Tensor,
    n_cells: int,
    rvord: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Output diagnostics of the active cells, each followed by its carried padding rows."""
    diagnostics = native_output_diagnostics(
        theta_m=theta_m[:n_cells],
        rho_zz=rho_zz[:n_cells],
        scalars=scalars[:n_cells],
        zz=zz[:n_cells],
        pressure_base=pressure_base[:n_cells],
        pressure_p=pressure_p[:n_cells],
        rvord=rvord,
    )
    return (
        torch.cat((diagnostics["theta"], theta_padding), dim=0),
        torch.cat((diagnostics["rho"], rho_padding), dim=0),
        torch.cat((diagnostics["pressure"], pressure_padding), dim=0),
    )


def _output_diagnostics_compiled(config, device):
    """The compiled output-diagnostics region when the coupled-diagnostics region
    is a compile target (forward only, as ``forward_compiled_region``), else None."""
    from collections.abc import Mapping

    if not isinstance(config, Mapping) or torch.is_grad_enabled():
        return None
    from mpas_pytorch.dynamics.compile_utils import (
        compile_backend,
        compile_enabled,
        compile_mode,
        kernel_compile_enabled_for_device,
        maybe_compile_kernel,
    )

    run_config = config if isinstance(config, dict) else dict(config)
    if not compile_enabled(run_config) or not kernel_compile_enabled_for_device(
        "coupled_diagnostics_region", run_config, device=device
    ):
        return None
    return maybe_compile_kernel(
        "output_diagnostics_region",
        _output_diagnostics_padded_numeric,
        enabled=True,
        mode=compile_mode(run_config),
        backend=compile_backend(run_config),
    )


def native_output_diagnostics(
    *,
    theta_m: torch.Tensor,
    rho_zz: torch.Tensor,
    scalars: torch.Tensor,
    zz: torch.Tensor,
    pressure_base: torch.Tensor,
    pressure_p: torch.Tensor,
    rvord: float,
) -> dict[str, torch.Tensor]:
    """Match atm_compute_output_diagnostics from current primary fields."""
    shape = theta_m.shape
    if theta_m.ndim != 2 or any(
        value.shape != shape for value in (rho_zz, zz, pressure_base, pressure_p)
    ):
        raise ValueError("output diagnostics require matching cell-by-level fields")
    if scalars.ndim != 3 or scalars.shape[:2] != shape or scalars.shape[2] < 1:
        raise ValueError("output diagnostics require the native qv tracer slot")
    return {
        "theta": theta_m / (1.0 + rvord * scalars[..., 0]),
        "rho": rho_zz * zz,
        "pressure": pressure_base + pressure_p,
    }


def physics_interface_surface_pressure(
    *,
    nCells: int,
    rho_zz: torch.Tensor,
    zz: torch.Tensor,
    qv: torch.Tensor,
    pressure_p: torch.Tensor,
    pressure_base: torch.Tensor,
    zgrid: torch.Tensor,
    gravity: float = GRAVITY,
) -> torch.Tensor:
    """Diagnose hydrostatic surface pressure using the MPAS physics interface.

    This is the evolved-state formula in ``mpas_atmphys_interface.F`` rather
    than the ad-hoc initialization extrapolation.  All inputs use cell-major
    layout and may contain either active cells only or one trailing padding
    row. The active count is explicit so a zero padding row cannot create a
    ``0/0`` hydrostatic thickness.

    Args:
        nCells: Number of active cell rows.
        rho_zz: Jacobian-weighted density, ``(nCells[+1], nVertLevels)``.
        zz: Vertical metric, same shape as ``rho_zz``.
        qv: Water-vapor mixing ratio, same shape as ``rho_zz``.
        pressure_p: Perturbation pressure, same shape as ``rho_zz``.
        pressure_base: Base pressure, same shape as ``rho_zz``.
        zgrid: W-level geometric height, ``(nCells[+1], nVertLevels+1)``.
        gravity: Gravitational acceleration in ``m s-2``.

    Returns:
        Hydrostatic surface pressure in Pa, one value per input cell row.
    """
    fields = {
        "rho_zz": rho_zz,
        "zz": zz,
        "qv": qv,
        "pressure_p": pressure_p,
        "pressure_base": pressure_base,
        "zgrid": zgrid,
    }
    for name, value in fields.items():
        if value.ndim != 2:
            raise ValueError(f"{name} must be rank-2, got shape {tuple(value.shape)}")

    if rho_zz.shape[0] not in (nCells, nCells + 1):
        raise ValueError(
            f"rho_zz first dimension must be nCells ({nCells}) or nCells+1 "
            f"({nCells + 1}), got {rho_zz.shape[0]}"
        )
    state_shape = tuple(rho_zz.shape)
    for name in ("zz", "qv", "pressure_p", "pressure_base"):
        value = fields[name]
        if tuple(value.shape) != state_shape:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} does not match rho_zz {state_shape}"
            )
    if zgrid.shape[0] != rho_zz.shape[0]:
        raise ValueError(
            "zgrid cell extent must match rho_zz: "
            f"{zgrid.shape[0]} != {rho_zz.shape[0]}"
        )
    if rho_zz.shape[1] < 2 or zgrid.shape[1] < 3:
        raise ValueError("surface pressure requires two mass levels and three W levels")

    zgrid_active = zgrid[:nCells, :]
    tem1 = zgrid_active[:, 1] - zgrid_active[:, 0]
    tem2 = zgrid_active[:, 2] - zgrid_active[:, 1]
    rho1 = rho_zz[:nCells, 0] * zz[:nCells, 0] * (1.0 + qv[:nCells, 0])
    rho2 = rho_zz[:nCells, 1] * zz[:nCells, 1] * (1.0 + qv[:nCells, 1])
    hydrostatic = (
        0.5 * gravity * tem1 * (rho1 - 0.5 * (rho2 - rho1) * tem1 / (tem1 + tem2))
    )
    surface_pressure = hydrostatic + pressure_p[:nCells, 0] + pressure_base[:nCells, 0]
    if rho_zz.shape[0] == nCells + 1:
        return torch.cat((surface_pressure, torch.zeros_like(surface_pressure[:1])))
    return surface_pressure
