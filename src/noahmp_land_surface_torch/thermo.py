# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Immutable funcphys gpvs coefficients and differentiable tensor lookup.

Scalar coefficient generation is setup, independent of atmospheric state.
Runtime interpolation stays in torch. The scalar associations match the
vendored fpvsx and gpvs implementations, including mixed-phase weighting.
"""
from functools import lru_cache
import math
import functools

import torch


def static_tensor(values, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """Device copy of a static host constant, built once per values, dtype and device.

    A per-call host constant costs a pageable copy and a stream sync on CUDA and
    blocks graph capture. Callers must not modify the result in place.
    """
    if torch.compiler.is_compiling():  # traced as a graph constant; no cache lookup
        return torch.tensor(values, dtype=dtype, device=device)
    return _static_tensor(_freeze(values), dtype, torch.device(device), torch.is_inference_mode_enabled())


def _freeze(values):
    return tuple(_freeze(v) for v in values) if isinstance(values, (list, tuple)) else values


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

N = 7501
XMIN = 180.0
XINC = (330.0 - XMIN) / (N - 1)
C2 = 1.0 / XINC
C1 = 1.0 - XMIN * C2
_TABLES = {}


def _fpvsx(t):
    ttp = 273.16
    rv, hvap, hfus = 461.5, 2.5e6, 3.3358e5
    cvap, cliq, csol, psat = 1846.0, 4185.5, 2106.0, 610.78
    a = -(cvap - cliq) / rv
    b = a + hvap / (rv * ttp)
    ai = -(cvap - csol) / rv
    bi = ai + (hvap + hfus) / (rv * ttp)
    tr = ttp / t
    if t >= ttp:
        return psat * (tr ** a) * math.exp(b * (1.0 - tr))
    if t < ttp - 20.0:
        return psat * (tr ** ai) * math.exp(bi * (1.0 - tr))
    liquid = psat * (tr ** a) * math.exp(b * (1.0 - tr))
    ice = psat * (tr ** ai) * math.exp(bi * (1.0 - tr))
    weight = (t - (ttp - 20.0)) / 20.0
    return weight * liquid + (1.0 - weight) * ice


@lru_cache(maxsize=1)
def coefficients():
    return tuple(_fpvsx(XMIN + j * XINC) for j in range(N))


def fpvs_table_like(reference):
    key = (reference.device, reference.dtype)
    if key not in _TABLES:
        _TABLES[key] = torch.tensor(coefficients(), device=reference.device, dtype=reference.dtype)
    return _TABLES[key]


def fpvs_from_table(temperature, table):
    position = torch.clamp(C1 + C2 * temperature, min=1.0, max=float(N))
    index = torch.clamp(torch.floor(position).to(torch.int64), max=N - 1)
    left = table[index - 1]
    right = table[index]
    return left + (position - index.to(temperature.dtype)) * (right - left)
