# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Torch-native thermodynamic helpers used by the SA-SAS profile harness."""

from __future__ import annotations

import math

import torch

from sa_sas_deep_convection_torch.constants import (
    CLIQ,
    CSOL,
    CVAP,
    HFUS,
    HVAP,
    PSAT,
    RV,
    TTP,
)

NXPVS = 7501
PVS_XMIN = 180.0
PVS_XMAX = 330.0
_FPVS_TABLES: dict[tuple[str, torch.dtype], torch.Tensor] = {}


def _fpvs_table_for_device(device: str, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Prebuilt ``fpvs`` table in the working dtype; a compiled region reads it as a tensor input.

    Built in float64 on the host (funcphys::gpvs) at import, held per device and dtype. A plain
    dict, not ``lru_cache``: Dynamo traces through ``lru_cache``, so the cached builder re-ran its
    7,501-point Python loop at each call site of every compile and baked the values into the graph.
    """
    key = (device, dtype)
    table = _FPVS_TABLES.get(key)
    if table is None:
        table = _FPVS_TABLES[key] = _FPVS_TABLE_CPU.to(device=torch.device(device), dtype=dtype)
    return table


def _fpvsx_scalar(temperature: float) -> float:
    tr = TTP / temperature
    tice = TTP - 20.0
    xponal = -(CVAP - CLIQ) / RV
    xponbl = -(CVAP - CLIQ) / RV + HVAP / (RV * TTP)
    xponai = -(CVAP - CSOL) / RV
    xponbi = -(CVAP - CSOL) / RV + (HVAP + HFUS) / (RV * TTP)
    if temperature >= TTP:
        return PSAT * (tr**xponal) * math.exp(xponbl * (1.0 - tr))
    if temperature < tice:
        return PSAT * (tr**xponai) * math.exp(xponbi * (1.0 - tr))
    weight = (temperature - tice) / 20.0
    pvl = PSAT * (tr**xponal) * math.exp(xponbl * (1.0 - tr))
    pvi = PSAT * (tr**xponai) * math.exp(xponbi * (1.0 - tr))
    return weight * pvl + (1.0 - weight) * pvi


def _fpvs_table_cpu() -> torch.Tensor:
    xinc = (PVS_XMAX - PVS_XMIN) / (NXPVS - 1)
    values = [_fpvsx_scalar(PVS_XMIN + index * xinc) for index in range(NXPVS)]
    with torch.inference_mode(False):
        return torch.tensor(values, dtype=torch.float64)


_FPVS_TABLE_CPU = _fpvs_table_cpu()


def _fpvsx(temperature: torch.Tensor) -> torch.Tensor:
    tr = TTP / temperature
    tice = TTP - 20.0
    xponal = -(CVAP - CLIQ) / RV
    xponbl = -(CVAP - CLIQ) / RV + HVAP / (RV * TTP)
    xponai = -(CVAP - CSOL) / RV
    xponbi = -(CVAP - CSOL) / RV + (HVAP + HFUS) / (RV * TTP)
    pvl = PSAT * torch.pow(tr, xponal) * torch.exp(torch.as_tensor(xponbl, dtype=temperature.dtype, device=temperature.device) * (1.0 - tr))
    pvi = PSAT * torch.pow(tr, xponai) * torch.exp(torch.as_tensor(xponbi, dtype=temperature.dtype, device=temperature.device) * (1.0 - tr))
    weight = (temperature - tice) / 20.0
    return torch.where(temperature >= TTP, pvl, torch.where(temperature < tice, pvi, weight * pvl + (1.0 - weight) * pvi))


def fpvs(temperature: torch.Tensor) -> torch.Tensor:
    """Match ``funcphys.fpvs`` table lookup in the working dtype (float64 parity, or float32)."""

    if temperature.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"temperature must be torch.float32 or torch.float64, got {temperature.dtype}")
    table = _fpvs_table_for_device(str(temperature.device), temperature.dtype)
    xinc = (PVS_XMAX - PVS_XMIN) / (NXPVS - 1)
    c2 = 1.0 / xinc
    c1 = 1.0 - PVS_XMIN * c2
    xj = torch.clamp(c1 + c2 * temperature, 1.0, float(NXPVS))
    jx = torch.minimum(xj.to(torch.int64), torch.full_like(xj, NXPVS - 1, dtype=torch.int64))
    return table[jx - 1] + (xj - jx.to(xj.dtype)) * (table[jx] - table[jx - 1])
