# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Prognostic-ozone and stratospheric-H2O grid setup and forcing (R4 time-vary lifecycle), and the
h2o photochemistry kernel (GFS v17 h2o_phys)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .containers import static_tensor
from .distributed import rank_values
from .grid import Grid
from .precision import WORKING_DTYPES


@dataclass(frozen=True)
class PrognosticOzone:
    """Promoted source REAL4 coefficients, with source surface-to-top ordering.

    Also holds the h2o table (global_h2oprdlos.f77): ty_h2ophys%load reads the same layout.
    """

    latitude: torch.Tensor
    pressure: torch.Tensor
    time: torch.Tensor
    data: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.latitude.ndim != 1
            or self.latitude.numel() < 1
            or self.pressure.ndim != 1
            or self.pressure.numel() < 1
            or self.time.ndim != 1
            or self.time.numel() < 3
            or self.data.ndim != 4
            or self.data.shape[:2] != (self.latitude.numel(), self.pressure.numel())
            or self.data.shape[2] < 1
            or self.data.shape[3] != self.time.numel() - 1
        ):
            raise ValueError("invalid source prognostic ozone table dimensions")
        for value in (self.latitude, self.pressure, self.time, self.data):
            if (value.dtype not in WORKING_DTYPES or value.dtype != self.data.dtype
                    or value.device != self.data.device):
                raise ValueError("ozone tables require colocated tensors of one working dtype")
            torch._assert_async(torch.isfinite(value).all(), "nonfinite ozone table")
        torch._assert_async(
            (self.latitude[1:] > self.latitude[:-1]).all()
            & (self.time[1:] > self.time[:-1]).all()
            & (self.pressure > 0.0).all(),
            "invalid ozone latitude, time or pressure coordinates",
        )

    @property
    def po3(self) -> torch.Tensor:
        return torch.log(100.0 * self.pressure)


def photochemistry_latitude_weights(
    latitude: torch.Tensor, table: PrognosticOzone
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """setup_o3prog / ty_h2ophys%setup (identical): one-based indices, strict upper bound and
    polar weight=1. Returns (jindx1, jindx2, ddy) for the columns of ``latitude``."""
    if (
        latitude.ndim != 1
        or latitude.dtype != table.data.dtype
        or latitude.device != table.data.device
    ):
        raise ValueError("invalid model photochemistry latitude")
    torch._assert_async(torch.isfinite(latitude).all(), "nonfinite model latitude")
    upper = torch.searchsorted(table.latitude, latitude.contiguous(), right=True)
    lower = torch.clamp_min(upper - 1, 0)
    upper = torch.clamp_max(upper, table.latitude.numel() - 1)
    distinct = upper != lower
    denominator = torch.where(
        distinct,
        table.latitude[upper] - table.latitude[lower],
        torch.ones_like(latitude),
    )
    weight = torch.where(
        distinct,
        (latitude - table.latitude[lower]) / denominator,
        torch.ones_like(latitude),
    )
    return (lower + 1).to(torch.int32), (upper + 1).to(torch.int32), weight


def initialize_ozone_grid(grid: Grid, ozone: PrognosticOzone) -> Grid:
    """setup_o3prog: one-based indices, strict upper bound and polar weight=1."""
    latitude = grid.xlat_d
    if (
        latitude.ndim != 1
        or latitude.dtype != ozone.data.dtype
        or latitude.device != ozone.data.device
    ):
        raise ValueError("invalid model ozone latitude")
    for name in ("jindx1_o3", "jindx2_o3", "ddy_o3"):
        value = getattr(grid, name)
        dtype = latitude.dtype if name == "ddy_o3" else torch.int32
        if (
            value.shape != latitude.shape
            or value.dtype != dtype
            or value.device != latitude.device
        ):
            raise ValueError(f"invalid ozone grid member: {name}")
    lower, upper, weight = photochemistry_latitude_weights(latitude, ozone)
    return replace(grid, jindx1_o3=lower, jindx2_o3=upper, ddy_o3=weight)


def ozone_forecast_day(jdat: torch.Tensor) -> torch.Tensor:
    """W3DOXDAT day-of-year plus whole UTC hour; minutes/seconds are omitted."""
    if jdat.shape != (8,) or jdat.dtype != torch.int32:
        raise ValueError("ozone forecast date requires source eight-element int32 jdat")
    year, month, day, hour = (jdat[i].to(torch.int64) for i in (0, 1, 2, 4))
    leap = (torch.remainder(year, 4) == 0) & (
        (torch.remainder(year, 100) != 0) | (torch.remainder(year, 400) == 0)
    )
    offsets = static_tensor(
        (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334),
        dtype=torch.int64,
        device=jdat.device,
    )
    lengths = static_tensor(
        (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
        dtype=torch.int64,
        device=jdat.device,
    )
    torch._assert_async(
        (year >= 1583)
        & (month >= 1)
        & (month <= 12)
        & (day >= 1)
        & (hour >= 0)
        & (hour <= 23),
        "invalid Gregorian ozone date",
    )
    index = torch.clamp(month - 1, 0, 11)
    month_length = lengths.index_select(0, index.reshape(1)).reshape(())
    month_offset = offsets.index_select(0, index.reshape(1)).reshape(())
    torch._assert_async(
        day <= month_length + (leap & (month == 2)).to(torch.int64),
        "invalid Gregorian ozone day",
    )
    doy = month_offset + day + (leap & (month > 2)).to(torch.int64)
    return doy.to(torch.float64) + hour.to(torch.float64) / 24.0


def update_ozone_forcing(
    grid: Grid, ozone: PrognosticOzone, forecast_day: torch.Tensor
) -> torch.Tensor:
    """find_photochem_time_index/update_o3prog, returning [ncol,nlev,ncf]."""
    return _update_photochemistry_forcing(
        (grid.jindx1_o3, grid.jindx2_o3, grid.ddy_o3), ozone,
        _photochemistry_day(ozone, _checked_forecast_day(ozone, forecast_day)))


def update_h2o_forcing(
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    h2o: PrognosticOzone,
    forecast_day: torch.Tensor,
    *,
    ozone: PrognosticOzone | None,
) -> torch.Tensor:
    """GFS_phys_time_vary h2o block: h2ophys%update at the day the ozone block left.

    The source adds 365 to the shared rjday in the ozone block (when ntoz > 0, ``ozone``) and
    tests it again against the h2o table. ``weights`` are photochemistry_latitude_weights of the
    h2o table (ty_h2ophys%setup). Returns h2opl [ncol,levh2o,h2o_coeff].
    """
    day = _checked_forecast_day(h2o, forecast_day)
    if ozone is not None:
        day = _photochemistry_day(ozone, day)
    return _update_photochemistry_forcing(weights, h2o, _photochemistry_day(h2o, day))


def _checked_forecast_day(table: PrognosticOzone, forecast_day: torch.Tensor) -> torch.Tensor:
    if (
        forecast_day.numel() != 1
        or forecast_day.dtype != torch.float64
        or forecast_day.device != table.data.device
    ):
        raise ValueError("photochemistry forecast day requires a colocated float64 scalar")
    return forecast_day.reshape(())


def _photochemistry_day(table: PrognosticOzone, day: torch.Tensor) -> torch.Tensor:
    """``if (rjday < time(1)) rjday = rjday + 365.``"""
    return torch.where(day < table.time[0], day + 365.0, day)


def _update_photochemistry_forcing(
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    table: PrognosticOzone,
    day: torch.Tensor,
) -> torch.Tensor:
    """find_photochem_time_index then update_o3prog / ty_h2ophys%update (identical)."""
    device = table.data.device
    jindx1, jindx2, weight = weights
    if weight.ndim != 1 or weight.dtype != table.data.dtype or weight.device != device:
        raise ValueError("invalid initialized photochemistry latitude weights")
    for index in (jindx1, jindx2):
        if (
            index.shape != weight.shape
            or index.dtype != torch.int32
            or index.device != device
        ):
            raise ValueError("invalid initialized photochemistry latitude indices")
        torch._assert_async(
            ((index >= 1) & (index <= table.latitude.numel())).all(),
            "uninitialized or out-of-range photochemistry latitude indices",
        )
    torch._assert_async(
        torch.isfinite(day) & torch.isfinite(weight).all(),
        "nonfinite photochemistry interpolation input",
    )
    first = torch.searchsorted(table.time[1:-1], day.reshape(1), right=True)
    second = torch.remainder(first + 1, table.data.shape[3])
    # The source wraps the data and TIME index together, not via TIME(ntime+1).
    t1 = table.time.index_select(0, first).reshape(())
    t2 = table.time.index_select(0, second).reshape(())
    tx1 = (t2 - day) / (t2 - t1)
    tx2 = 1.0 - tx1
    j1, j2 = jindx1.to(torch.int64) - 1, jindx2.to(torch.int64) - 1
    h = weight[:, None, None]
    tem = 1.0 - h
    return tx1 * (
        tem * table.data[j1, :, :, first] + h * table.data[j2, :, :, first]
    ) + tx2 * (tem * table.data[j1, :, :, second] + h * table.data[j2, :, :, second])


def forcing_at_model_levels(
    logp: torch.Tensor, log_pressure: torch.Tensor, forcing: torch.Tensor
) -> torch.Tensor:
    """The vertical interpolation of run_o3prog_2015 and ty_h2ophys%run (identical loops).

    ``logp`` [ncol,nlev] is log(model pressure), ``log_pressure`` the descending forcing levels,
    ``forcing`` [ncol,nforce,ncf]. The source kmin..kmax search visits exactly the bracketing
    interval of every in-range column; values below the lowest or at/above the highest forcing
    level take that end level. Returns [ncol,nlev,ncf].
    """
    ncol = logp.shape[0]
    lower = (torch.searchsorted(-log_pressure, -logp.contiguous(), right=False) - 1).clamp(
        0, log_pressure.numel() - 2
    )
    upper = lower + 1
    columns = torch.arange(ncol, device=logp.device)[:, None]
    inverse_width = 1.0 / (log_pressure[lower] - log_pressure[upper])
    weight = (logp - log_pressure[upper]) * inverse_width
    coefficients = weight[..., None] * forcing[columns, lower] + (
        1.0 - weight
    )[..., None] * forcing[columns, upper]
    coefficients = torch.where(
        (logp < log_pressure[-1])[..., None], forcing[:, -1:, :], coefficients
    )
    return torch.where(
        (logp >= log_pressure[0])[..., None], forcing[:, :1, :], coefficients
    )


def ozone_2015(
    pressure: torch.Tensor,
    temperature: torch.Tensor,
    pressure_thickness: torch.Tensor,
    ozone: torch.Tensor,
    *,
    log_pressure: torch.Tensor,
    forcing: torch.Tensor,
    dtp: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """run_o3prog_2015: implicit ozone loss and ordered overhead columns.

    Model levels and forcing coordinates run from surface to top. The overhead
    recurrence uses the initial ozone, never the updated ozone above this level.
    """
    from gfs_surface_layer_torch.driver import GRAV

    if pressure.ndim != 2 or min(pressure.shape) < 1:
        raise ValueError("ozone2015 requires nonempty [ncol,nlev] pressure")
    ncol, nlev = pressure.shape
    if log_pressure.ndim != 1 or log_pressure.numel() < 2:
        raise ValueError("ozone2015 requires at least two forcing levels")
    if forcing.shape != (ncol, log_pressure.numel(), 6):
        raise ValueError("ozone2015 requires six complete forcing coefficients")
    if dtp.ndim != 0:
        raise ValueError("ozone2015 timestep must be a scalar tensor")
    for name, value in (
        ("pressure", pressure), ("temperature", temperature),
        ("pressure thickness", pressure_thickness), ("ozone", ozone),
        ("log pressure", log_pressure), ("forcing", forcing), ("timestep", dtp),
    ):
        if (value.dtype not in WORKING_DTYPES or (value.ndim and value.dtype != pressure.dtype)
                or value.device != pressure.device):
            raise ValueError(f"ozone2015 requires colocated {name} in the working dtype")
        torch._assert_async(torch.isfinite(value).all(), f"nonfinite ozone2015 {name}")
    if any(value.shape != pressure.shape for value in (temperature, pressure_thickness, ozone)):
        raise ValueError("ozone2015 model field extents differ")
    torch._assert_async(
        (pressure > 0.).all() & (temperature > 0.).all()
        & (pressure_thickness > 0.).all() & (dtp > 0.)
        & (log_pressure[:-1] > log_pressure[1:]).all(),
        "invalid ozone2015 thermodynamics, timestep or descending forcing grid",
    )

    logp = torch.log(pressure)
    coefficients = forcing_at_model_levels(logp, log_pressure, forcing)
    one_over_g = 1.0 / GRAV
    # Only the two overhead columns are ordered recurrences (top down, inclusive of level k);
    # every other term is elementwise per level and runs on all levels at once.
    ozone_layer = ozone * pressure_thickness * one_over_g
    reference_layer = coefficients[:, :, 5] * pressure_thickness * one_over_g
    column_ozone, column_reference = ozone.new_zeros(ncol), ozone.new_zeros(ncol)
    columns_ozone, columns_reference = [], []
    for k in range(nlev - 1, -1, -1):
        column_ozone = column_ozone + ozone_layer[:, k]
        column_reference = column_reference + reference_layer[:, k]
        columns_ozone.append(column_ozone)
        columns_reference.append(column_reference)
    column_ozone = torch.stack(columns_ozone[::-1], dim=1)
    column_reference = torch.stack(columns_reference[::-1], dim=1)
    prod = coefficients
    loss = torch.where(prod[..., 1] <= 0., prod[..., 1], 0.)
    thermal = prod[..., 2] * (temperature - prod[..., 4])
    overhead = prod[..., 3] * (column_ozone - column_reference)
    source = prod[..., 0] - loss * prod[..., 5] + thermal + overhead
    updated = (ozone + source * dtp) / (1.0 - loss * dtp)
    result = {"oz": updated, "do3_dt_prd": prod[..., 0] * dtp,
              "do3_dt_ozmx": loss * (updated - prod[..., 5]) * dtp,
              "do3_dt_temp": thermal * dtp, "do3_dt_ohoz": overhead * dtp}
    for value in result.values():
        torch._assert_async(torch.isfinite(value).all(), "nonfinite ozone2015 output")
    return result


# module_h2ophys run: real(kind_phys), parameter :: prsmax=10000.0, pmaxl=log(prsmax)
H2O_PRSMAX = 10000.0
H2O_PMAXL = math.log(H2O_PRSMAX)


def h2o_photochemistry(
    pressure: torch.Tensor,
    h2o: torch.Tensor,
    *,
    log_pressure: torch.Tensor,
    forcing: torch.Tensor,
    dtp: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """ty_h2ophys%run: NRL stratospheric water-vapour photochemistry (implicit relaxation).

    ``forcing`` is h2opl [ncol,levh2o,ncf>=3] (production, relaxation time scale, climatological
    mixing ratio). A level gets interpolated coefficients only if the lowest log-pressure over the
    columns of the call (the source block) is below log(prsmax); otherwise its coefficients stay zero.
    A column is updated where p < prsmax and the time scale is nonzero. Returns the updated
    ``h2o`` and the optional diagnostics ``dqv_dt_prd`` and ``dqv_dt_qvmx``.
    """
    if pressure.ndim != 2 or min(pressure.shape) < 1:
        raise ValueError("h2o photochemistry requires nonempty [ncol,nlev] pressure")
    ncol = pressure.shape[0]
    if log_pressure.ndim != 1 or log_pressure.numel() < 2:
        raise ValueError("h2o photochemistry requires at least two forcing levels")
    if forcing.ndim != 3 or forcing.shape[:2] != (ncol, log_pressure.numel()) or forcing.shape[2] < 3:
        raise ValueError("h2o photochemistry requires three forcing coefficients on every level")
    if dtp.ndim != 0 or h2o.shape != pressure.shape:
        raise ValueError("invalid h2o photochemistry timestep or water-vapour extent")
    for name, value in (("pressure", pressure), ("water vapour", h2o), ("log pressure", log_pressure),
                        ("forcing", forcing), ("timestep", dtp)):
        if (value.dtype not in WORKING_DTYPES or (value.ndim and value.dtype != pressure.dtype)
                or value.device != pressure.device):
            raise ValueError(f"h2o photochemistry requires colocated {name} in the working dtype")
        torch._assert_async(torch.isfinite(value).all(), f"nonfinite h2o photochemistry {name}")
    torch._assert_async(
        (pressure > 0.).all() & (dtp > 0.) & (log_pressure[:-1] > log_pressure[1:]).all(),
        "invalid h2o photochemistry pressure, timestep or descending forcing grid",
    )
    logp = torch.log(pressure)
    coefficients = forcing_at_model_levels(logp, log_pressure, forcing)
    # Per level, `if (pmin < pmaxl)` over the columns of the call; pmin is a reduction of values.
    # On a partition the columns of the call are all cells (one rank's decision, on every rank).
    interpolated = rank_values(torch.amin(logp, dim=0)).amin(dim=0) < H2O_PMAXL
    pltc = torch.where(interpolated[None, :, None], coefficients, 0.0)
    production, scale, climatology = pltc[..., 0], pltc[..., 1], pltc[..., 2]
    nonzero = scale != 0.0
    update = (pressure < H2O_PRSMAX) & nonzero
    temp = 1.0 / torch.where(update, scale, 1.0)
    updated = torch.where(update, (h2o + (production + climatology * temp) * dtp) / (1.0 + temp * dtp), h2o)
    result = {
        "h2o": updated,
        "dqv_dt_prd": production * dtp,
        "dqv_dt_qvmx": torch.where(
            nonzero, (updated - climatology) * dtp / torch.where(nonzero, scale, 1.0), 0.0),
    }
    for value in result.values():
        torch._assert_async(torch.isfinite(value).all(), "nonfinite h2o photochemistry output")
    return result
