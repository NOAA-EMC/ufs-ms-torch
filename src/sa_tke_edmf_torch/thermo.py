# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Prepared funcphys coefficients and PyTorch-native thermodynamic lookup."""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from sa_tke_edmf_torch.constants import CLIQ, CSOL, CVAP, HFUS, HVAP, PSAT, RV, TTP

_FPVS_N = 7501
_FPVS_XMIN = 180.0
_FPVS_XMAX = 330.0
_FPVS_XINC = (_FPVS_XMAX - _FPVS_XMIN) / (_FPVS_N - 1)
_FPVS_C2 = (_FPVS_N - 1) / (_FPVS_XMAX - _FPVS_XMIN)
_FPVS_C1 = 1.0 - _FPVS_XMIN * _FPVS_C2
_FPVS_TABLE_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _fpvsx(t: float) -> float:
    """Generate an immutable coefficient using funcphys's scalar association.

    This runs only during table preparation, without state tensors. Tensor
    pow/exp differ from scalar libm at some nodes; using those coefficients
    changed the ocean plume's height even with identical lookup operands.
    Current-temperature interpolation and its derivatives stay in torch.
    """
    tliq = TTP
    tice = TTP - 20.0
    dldtl = CVAP - CLIQ
    xponal = -dldtl / RV
    xponbl = -dldtl / RV + HVAP / (RV * TTP)
    dldti = CVAP - CSOL
    heati = HVAP + HFUS
    xponai = -dldti / RV
    xponbi = -dldti / RV + heati / (RV * TTP)
    tr = TTP / t
    if t >= tliq:
        return PSAT * (tr ** xponal) * math.exp(xponbl * (1.0 - tr))
    if t < tice:
        return PSAT * (tr ** xponai) * math.exp(xponbi * (1.0 - tr))
    pvl = PSAT * (tr ** xponal) * math.exp(xponbl * (1.0 - tr))
    pvi = PSAT * (tr ** xponai) * math.exp(xponbi * (1.0 - tr))
    w = (t - tice) / (tliq - tice)
    return w * pvl + (1.0 - w) * pvi


@lru_cache(maxsize=1)
def _fpvs_coefficients() -> tuple[float, ...]:
    return tuple(_fpvsx(_FPVS_XMIN + j * _FPVS_XINC) for j in range(_FPVS_N))


def _fpvs_table(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    table = _FPVS_TABLE_CACHE.get(key)
    if table is None:
        table = torch.tensor(_fpvs_coefficients(), dtype=dtype, device=device)
        _FPVS_TABLE_CACHE[key] = table
    return table


def fpvs_table_like(reference: torch.Tensor) -> torch.Tensor:
    """Return the cached saturation-vapor-pressure lookup table for a tensor."""

    return _fpvs_table(reference.device, reference.dtype)


def fpvs_from_table(t: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Linear-table saturation vapor pressure with the lookup table passed in."""

    xj = torch.clamp(_FPVS_C1 + _FPVS_C2 * t, min=1.0, max=float(_FPVS_N))
    jx = torch.minimum(torch.floor(xj).to(torch.int64), torch.full_like(xj, _FPVS_N - 1, dtype=torch.int64))
    idx0 = jx - 1
    left = table[idx0]
    right = table[idx0 + 1]
    return left + (xj - jx.to(t.dtype)) * (right - left)


def fpvs(t: torch.Tensor) -> torch.Tensor:
    """Linear-table saturation vapor pressure used by Fortran ``fpvs``."""

    return fpvs_from_table(t, fpvs_table_like(t))
