# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Gas volume mixing ratios (``getgases``, ``ico2``) and the saturation vapour pressure table (``fpvs``) of
``GFS_rrtmgp_pre``."""

from __future__ import annotations

from functools import lru_cache
import math
import re
from pathlib import Path

import torch

from .reference_inputs import physics_input_dir as _physics_input_dir


TensorOrString = torch.Tensor | str

CON_RV = 461.50
CON_PSAT = 6.1078e2
CON_TTP = 2.7316e2
CON_HVAP = 2.5e6
CON_HFUS = 3.3358e5
CON_CVAP = 1.8460e3
CON_CLIQ = 4.1855e3
CON_CSOL = 2.1060e3
CON_AMD = 28.9644
CON_AMW = 18.0154
CON_AMO3 = 47.9982
AMDW = CON_AMD / CON_AMW
AMDO3 = CON_AMD / CON_AMO3

DEFAULT_CO2_VMR = 350.0e-6
DEFAULT_N2O_VMR = 0.31e-6
DEFAULT_CH4_VMR = 1.50e-6
DEFAULT_O2_VMR = 0.209
CO2_PRESSURE_LIMIT_PA = 78800.0
CO2_TABLE_IMAX = 24
CO2_TABLE_JMAX = 12
CO2_TABLE_RESLONLAT_DEG = 15.0
CO2_TABLE_EXPECTED_SHA256 = {
    1992: "bba504f731fd5e72c32a0ded020a174a079154b5d81729b455e79cdbcbe7f17c",
}


def _as_real(value: TensorOrString, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64 (the working dtype), got {value.dtype}")
    return value


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
@torch.inference_mode(False)
def _load_co2_monthly_table_cpu(year: int) -> tuple[torch.Tensor, float]:
    path = _physics_input_dir() / f"co2historicaldata_{year:04d}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    expected_sha = CO2_TABLE_EXPECTED_SHA256.get(year)
    if expected_sha is not None:
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(f"{path.name} sha256 {actual_sha} does not match frozen {expected_sha}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = lines[0] if lines else ""
    match = re.search(r"GLB ANNUAL MEAN =\s*([0-9.]+)", header)
    if match is None:
        raise ValueError(f"could not parse global annual CO2 mean from {path}")
    global_mean_vmr = float(match.group(1)) * 1.0e-6
    values: list[float] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == CO2_TABLE_IMAX:
            values.extend(float(part) for part in parts)
    expected_count = 12 * CO2_TABLE_JMAX * CO2_TABLE_IMAX
    if len(values) != expected_count:
        raise ValueError(f"{path} has {len(values)} CO2 values, expected {expected_count}")
    table = torch.tensor(values, dtype=torch.float64).reshape(12, CO2_TABLE_JMAX, CO2_TABLE_IMAX) * 1.0e-6
    return table, global_mean_vmr


def compute_nonprognostic_gases(
    p_lev: torch.Tensor,
    xlon: torch.Tensor,
    xlat: torch.Tensor,
    *,
    ico2: int,
    top_at_1: bool,
    con_pi: float = math.pi,
    co2_year: int = 1992,
    co2_month: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port `module_radiation_gases::getgases` for gases used by RRTMGP pre.

    The active sealed profile uses `ico2=2`, which selects a 15-degree monthly
    CO2 table below 788 hPa and a global annual mean above that pressure. Other
    non-prognostic gases are constants in the upstream routine.
    """

    p_int = _as_real(p_lev, name="p_lev")
    lon = _as_real(xlon, name="xlon").reshape(-1)
    lat = _as_real(xlat, name="xlat").reshape(-1)
    nlay = p_int.shape[0] - 1
    ncol = p_int.shape[1]
    if lon.numel() != ncol or lat.numel() != ncol:
        raise ValueError("xlon/xlat length must match p_lev columns")
    co2 = torch.full((nlay, ncol), DEFAULT_CO2_VMR, dtype=p_int.dtype, device=p_int.device)
    if ico2 == 1:
        _, global_mean = _load_co2_monthly_table_cpu(co2_year)
        co2 = torch.full_like(co2, global_mean)
    elif ico2 == 2:
        if not (1 <= co2_month <= 12):
            raise ValueError(f"co2_month must be 1..12, got {co2_month}")
        table, global_mean = _co2_table_on_device(co2_year, p_int.device, p_int.dtype)
        raddeg = 180.0 / con_pi
        tmp = raddeg / CO2_TABLE_RESLONLAT_DEG
        lon_wrapped = torch.where(lon < 0.0, lon + con_pi, lon)
        lat_index_value = (0.5 * con_pi - lat) * tmp + 1.0
        lon_index_value = lon_wrapped * tmp + 1.0
        ilon = torch.clamp(torch.floor(lon_index_value).to(dtype=torch.int64), min=1, max=CO2_TABLE_IMAX) - 1
        ilat = torch.clamp(torch.floor(lat_index_value).to(dtype=torch.int64), min=1, max=CO2_TABLE_JMAX) - 1
        column_co2 = table[co2_month - 1, ilat, ilon]
        if top_at_1:
            lower_atmosphere = p_int[:-1, :] >= CO2_PRESSURE_LIMIT_PA
        else:
            lower_atmosphere = p_int[1:, :] >= CO2_PRESSURE_LIMIT_PA
        co2 = torch.where(lower_atmosphere, column_co2.reshape(1, ncol), torch.full_like(co2, global_mean))
    elif ico2 != 0:
        raise NotImplementedError(f"unsupported ico2 branch for RRTMGP pre gas lookup: {ico2}")
    n2o = _repeat_gas(DEFAULT_N2O_VMR, co2)
    ch4 = _repeat_gas(DEFAULT_CH4_VMR, co2)
    o2 = _repeat_gas(DEFAULT_O2_VMR, co2)
    return o2, ch4, n2o, co2


@lru_cache(maxsize=1)
@torch.inference_mode(False)
def _fpvs_table_cpu() -> torch.Tensor:
    n = 7501
    xmin = 180.0
    xmax = 330.0
    # funcphys::gpvs builds this ordered grid; linspace uses different rounding.
    x = xmin + torch.arange(n, dtype=torch.float64) * ((xmax - xmin) / (n - 1))
    tr = torch.div(CON_TTP, x)
    tliq = CON_TTP
    tice = CON_TTP - 20.0
    dldtl = CON_CVAP - CON_CLIQ
    dldti = CON_CVAP - CON_CSOL
    xponal = -dldtl / CON_RV
    xponbl = -dldtl / CON_RV + CON_HVAP / (CON_RV * CON_TTP)
    xponai = -dldti / CON_RV
    xponbi = -dldti / CON_RV + (CON_HVAP + CON_HFUS) / (CON_RV * CON_TTP)
    liquid = CON_PSAT * (tr**xponal) * torch.exp(xponbl * (1.0 - tr))
    ice = CON_PSAT * (tr**xponai) * torch.exp(xponbi * (1.0 - tr))
    weight = (x - tice) / (tliq - tice)
    return torch.where(x >= tliq, liquid, torch.where(x < tice, ice, weight * liquid + (1.0 - weight) * ice))


def fpvs(t: torch.Tensor) -> torch.Tensor:
    """Match `funcphys::fpvs` linear lookup table in Pascals."""

    temp = _as_real(t, name="t")
    table = _fpvs_table_on_device(temp.device, temp.dtype)
    c2 = 1.0 / ((330.0 - 180.0) / 7500.0)
    c1 = 1.0 - 180.0 * c2
    xj = torch.clamp(c1 + c2 * temp, min=1.0, max=7501.0)
    jx = torch.minimum(torch.floor(xj).to(torch.int64), torch.full_like(torch.floor(xj).to(torch.int64), 7500))
    lower = table[jx - 1]
    upper = table[jx]
    return lower + (xj - jx.to(dtype=temp.dtype)) * (upper - lower)


_FPVS_DEVICE_TABLES: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _fpvs_table_on_device(device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Prebuilt ``fpvs`` table per device and working dtype; a compiled region reads it as an input.

    Built in float64 on the host (funcphys::gpvs) by preparation or the eager first call. A plain
    dict, not ``lru_cache``: Dynamo traces through ``lru_cache`` and rebuilt the table inside every
    graph that reached it.
    """
    key = (device, dtype)
    table = _FPVS_DEVICE_TABLES.get(key)
    if table is None:
        table = _FPVS_DEVICE_TABLES[key] = _fpvs_device_copy(device, dtype)
    return table


@torch.inference_mode(False)
def _fpvs_device_copy(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _fpvs_table_cpu().to(device=device, dtype=dtype)


@lru_cache(maxsize=8)
@torch.inference_mode(False)
def _co2_table_on_device(year: int, device: torch.device, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, float]:
    table, global_mean = _load_co2_monthly_table_cpu(year)
    return table.to(device=device, dtype=dtype), global_mean


def _repeat_gas(value: float, template: torch.Tensor) -> torch.Tensor:
    return torch.full_like(template, float(value))


__all__ = [
    "AMDO3",
    "AMDW",
    "compute_nonprognostic_gases",
    "fpvs",
]
