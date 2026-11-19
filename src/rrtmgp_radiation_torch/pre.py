# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Gas volume mixing ratios (``getgases``) and the saturation vapour pressure table (``fpvs``) of
``GFS_rrtmgp_pre``."""

from __future__ import annotations

from functools import lru_cache

import torch


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


def _as_real(value: TensorOrString, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64 (the working dtype), got {value.dtype}")
    return value


def compute_nonprognostic_gases(p_lev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`module_radiation_gases::getgases` with ``ico2 = 0``: O2, CH4, N2O and CO2 volume mixing ratios.

    Each is the upstream constant on the ``[nlay, ncol]`` layers of ``p_lev``. With ``ico2 = 2`` the
    radiation group replaces CO2 from the monthly tables the host prepares (``RadiationConfig.co2_monthly``).
    """

    p_int = _as_real(p_lev, name="p_lev")
    nlay = p_int.shape[0] - 1
    ncol = p_int.shape[1]
    co2 = torch.full((nlay, ncol), DEFAULT_CO2_VMR, dtype=p_int.dtype, device=p_int.device)
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
    dict, not ``lru_cache``: Dynamo traces through ``lru_cache`` and rebuilds the table inside every
    graph that reaches it.
    """
    key = (device, dtype)
    table = _FPVS_DEVICE_TABLES.get(key)
    if table is None:
        table = _FPVS_DEVICE_TABLES[key] = _fpvs_device_copy(device, dtype)
    return table


@torch.inference_mode(False)
def _fpvs_device_copy(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _fpvs_table_cpu().to(device=device, dtype=dtype)


def _repeat_gas(value: float, template: torch.Tensor) -> torch.Tensor:
    return torch.full_like(template, float(value))


__all__ = [
    "AMDO3",
    "AMDW",
    "compute_nonprognostic_gases",
    "fpvs",
]
