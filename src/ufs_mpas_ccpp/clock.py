# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Elapsed-time controls shared by the R4 MPAS host and CCPP timestep setup."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .containers import static_tensor
from rrtmgp_radiation_torch.setup import _iw3jdn


def _nint_nonnegative(value: torch.Tensor) -> torch.Tensor:
    """Fortran NINT without rounding a value just below a half-integer upward."""
    integral = torch.floor(value)
    return integral.to(torch.int64) + ((value - integral) >= 0.5).to(torch.int64)


def advance_utc_date(
    initial_date: torch.Tensor, forecast_hours: torch.Tensor
) -> torch.Tensor:
    """W3MOVDAT/W3REDDAT hour interval, preserving source millisecond rounding."""
    if (
        initial_date.shape != (8,)
        or initial_date.dtype != torch.int32
        or forecast_hours.numel() != 1
        or forecast_hours.dtype != torch.float64
        or forecast_hours.device != initial_date.device
    ):
        raise ValueError("UTC advancement requires int32 NCEP date and float64 hours")
    date = initial_date.to(torch.int64)
    year, month, day, zone, hour, minute, second, millisecond = date.unbind()
    torch._assert_async(
        (year >= 1583)
        & (year <= 3300)
        & (month >= 1)
        & (month <= 12)
        & (day >= 1)
        & (day <= 31)
        & (zone == 0)
        & (hour >= 0)
        & (hour < 24)
        & (minute >= 0)
        & (minute < 60)
        & (second >= 0)
        & (second < 60)
        & (millisecond >= 0)
        & (millisecond < 1000),
        "invalid UTC initial date",
    )
    hours = forecast_hours.reshape(()) + hour.to(torch.float64)
    torch._assert_async(
        torch.isfinite(hours) & (torch.abs(hours) < 2147483647.0),
        "UTC forecast interval exceeds source int32 hours",
    )
    integral_hours = torch.floor(hours).to(torch.int64)
    remainder_ms = (hours - integral_hours.to(torch.float64)) * 3600000.0
    remainder_ms = _nint_nonnegative(remainder_ms + millisecond.to(torch.float64))
    # Source canonicalization rounds fractional units before the integer carry chain.
    ms = torch.remainder(remainder_ms, 1000)
    seconds = second + (remainder_ms - ms) // 1000
    ss = torch.remainder(seconds, 60)
    minutes = minute + (seconds - ss) // 60
    mm = torch.remainder(minutes, 60)
    hours_int = integral_hours + (minutes - mm) // 60
    hh = torch.remainder(hours_int, 24)
    jdn = _iw3jdn(year, month, day) + (hours_int - hh) // 24

    # W3FS26 uses positive integer quotients throughout this Gregorian range.
    length = jdn + 68569
    century = 4 * length // 146097
    length = length - (146097 * century + 3) // 4
    y = 4000 * (length + 1) // 1461001
    length = length - 1461 * y // 4 + 31
    m = 80 * length // 2447
    d = length - 2447 * m // 80
    length = m // 11
    m = m + 2 - 12 * length
    y = 100 * (century - 49) + y + length
    torch._assert_async(
        (y >= 1583) & (y <= 3300), "UTC date exceeds source calendar range"
    )
    return torch.stack((y, m, d, zone, hh, mm, ss, ms)).to(torch.int32)


@dataclass(frozen=True)
class StepClock:
    sec: torch.Tensor
    phour: torch.Tensor
    zhour: torch.Tensor
    fhour: torch.Tensor
    kdt: torch.Tensor
    first_time_step: torch.Tensor
    lsswr: torch.Tensor
    lslwr: torch.Tensor
    solhr: torch.Tensor


def prepare_physics_calendar(
    jdat: torch.Tensor, yearlen: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source zero-based Noah-MP day and pinned-host year-length carry.

    GFS_time_vary_pre assigns yearlen only in years divisible by four.
    Preserve the supplied host buffer in other years, including its source
    cold value 365. This follows the pinned build, not a portable definition
    of the source's unassigned intent(out) value. Seconds do not enter julian.
    """
    if (
        jdat.shape != (8,)
        or jdat.dtype != torch.int32
        or yearlen.shape != ()
        or yearlen.dtype != torch.int32
        or yearlen.device != jdat.device
    ):
        raise ValueError("physics calendar requires int32 date and year-length carry")
    date = jdat.to(torch.int64)
    year, month, day, hour, minute = (date[i] for i in (0, 1, 2, 4, 5))
    fourth = torch.remainder(year, 4) == 0
    leap = fourth & ((torch.remainder(year, 100) != 0) | (torch.remainder(year, 400) == 0))
    month_days = static_tensor((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
                               dtype=torch.int64, device=jdat.device)
    days = month_days.gather(0, torch.clamp(month - 1, 0, 11).reshape(1)).reshape(())
    days = days + ((month == 2) & leap)
    torch._assert_async(
        (year >= 1583) & (year <= 3300) & (month >= 1) & (month <= 12)
        & (day >= 1) & (day <= days) & (date[3] == 0)
        & (hour >= 0) & (hour < 24) & (minute >= 0) & (minute < 60)
        & (date[6] >= 0) & (date[6] < 60) & (date[7] >= 0) & (date[7] < 1000)
        & ((yearlen == 365) | (yearlen == 366)),
        "invalid source physics calendar",
    )
    day_index = _iw3jdn(year, month, day) - _iw3jdn(year, 1, 1)
    fraction = hour.to(torch.float64) / 24.0 + minute.to(torch.float64) / 1440.0
    julian = day_index.to(torch.float64) + fraction
    updated_yearlen = torch.where(fourth, 365 + leap.to(torch.int32), yearlen)
    return julian, updated_yearlen


def prepare_timestep_clock(
    sec: torch.Tensor,
    dtp: torch.Tensor,
    *,
    initial_hour: int,
    nsswr: int,
    nslwr: int,
    nhfrad: int,
) -> StepClock:
    """Source clock arithmetic after elapsed seconds, before first radiation.

    GFS_time_vary_pre_timestep_init sets zhour=phour, replacing the earlier
    host bucket-hour reduction. UTC date construction, Julian day/year length,
    diagnostic resets and lifecycle initialization remain separate operations.
    """
    if any(type(v) is not int for v in (initial_hour, nsswr, nslwr, nhfrad)):
        raise TypeError("timestep clock configuration must be static integers")
    if not 0 <= initial_hour < 24 or nsswr < 1 or nslwr < 1 or nhfrad < 0:
        raise ValueError("invalid timestep clock configuration")
    if (
        sec.numel() != 1
        or dtp.numel() != 1
        or sec.dtype != torch.float64
        or dtp.dtype != torch.float64
        or sec.device != dtp.device
    ):
        raise ValueError("invalid elapsed time/timestep dtype, shape or device")
    torch._assert_async(
        (torch.isfinite(sec) & (sec >= 0.0)).all()
        & (torch.isfinite(dtp) & (dtp > 0.0)).all(),
        "invalid elapsed time or physics timestep",
    )
    sec, dtp = sec.reshape(()), dtp.reshape(())
    phour = sec / 3600.0
    step = (sec + dtp) / dtp
    torch._assert_async(
        step < 2147483647.5, "physics step counter exceeds source int32"
    )
    kdt = _nint_nonnegative(step).to(torch.int32)
    lsswr = (torch.fmod(kdt, nsswr) == 1) | (nsswr == 1) | (kdt <= nhfrad)
    lslwr = (torch.fmod(kdt, nslwr) == 1) | (nslwr == 1) | (kdt <= nhfrad)
    return StepClock(
        sec,
        phour,
        phour,
        (sec + dtp) / 3600.0,
        kdt,
        kdt == 1,
        lsswr,
        lslwr,
        torch.fmod(phour + initial_hour, 24.0),
    )
