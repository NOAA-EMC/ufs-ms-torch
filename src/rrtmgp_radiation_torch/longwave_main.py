# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Longwave cloud optics by band (``rrtmgp_lw_cloud_optics``: liquid and ice look-up tables)."""

from __future__ import annotations

from typing import Any

import torch

from .shortwave import (
    _combine_cloud_components,
    _interp_cloud_table,
)
from .tables import RrtmgpTableBundle

TensorOrString = torch.Tensor | str
_LONGWAVE_CLOUD_TABLE_CACHE: dict[tuple[int, int], tuple[Any, ...]] = {}


def _to_real(value: TensorOrString, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64 (the working dtype), got {value.dtype}")
    return value


@torch.inference_mode(False)
def _longwave_cloud_table_inputs(
    table: Any,
    *,
    ice_roughness_index: int | None,
) -> tuple[Any, ...]:
    """Prepare immutable longwave cloud-table views outside compiled kernels."""

    tensors = table.tensors
    roughness = int(table.dimensions.get("nrghice", tensors["lut_extice"].shape[0])) - 1
    if ice_roughness_index is not None:
        roughness = int(ice_roughness_index)
    key = (id(table), roughness)
    cached = _LONGWAVE_CLOUD_TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    liq_ext = tensors["lut_extliq"].transpose(0, 1).contiguous()
    liq_ssa = tensors["lut_ssaliq"].transpose(0, 1).contiguous()
    liq_asy = tensors["lut_asyliq"].transpose(0, 1).contiguous()
    ice_ext = tensors["lut_extice"][roughness].transpose(0, 1).contiguous()
    ice_ssa = tensors["lut_ssaice"][roughness].transpose(0, 1).contiguous()
    ice_asy = tensors["lut_asyice"][roughness].transpose(0, 1).contiguous()
    radliq_lwr = tensors["radliq_lwr"]
    radice_lwr = tensors["radice_lwr"]
    liq_step = (tensors["radliq_upr"] - radliq_lwr) / (liq_ext.shape[0] - 1)
    ice_step = (tensors["radice_upr"] - radice_lwr) / (ice_ext.shape[0] - 1)
    prepared = (
        liq_ext,
        liq_ssa,
        liq_asy,
        ice_ext,
        ice_ssa,
        ice_asy,
        radliq_lwr,
        radice_lwr,
        liq_step,
        ice_step,
    )
    _LONGWAVE_CLOUD_TABLE_CACHE[key] = prepared
    return prepared


def compute_longwave_cloud_optics_byband(
    cld_lwp: torch.Tensor,
    cld_iwp: torch.Tensor,
    cld_reliq: torch.Tensor,
    cld_reice: torch.Tensor,
    *,
    tables: RrtmgpTableBundle,
    ice_roughness_index: int | None = None,
    band_axis: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute `lw_cloud_props%cloud_optics` by band for liquid plus ice (``band_axis``: ``_interp_cloud_table``)."""

    lwp = _to_real(cld_lwp, name="cld_lwp")
    prepared = _longwave_cloud_table_inputs(
        tables.files["lw_cloud"],
        ice_roughness_index=ice_roughness_index,
    )
    liquid = _interp_cloud_table(
        lwp,
        cld_reliq,
        prepared[0],
        prepared[1],
        prepared[2],
        offset=prepared[6],
        step_size=prepared[8],
        band_axis=band_axis,
    )
    ice = _interp_cloud_table(
        cld_iwp,
        cld_reice,
        prepared[3],
        prepared[4],
        prepared[5],
        offset=prepared[7],
        step_size=prepared[9],
        band_axis=band_axis,
    )
    return _combine_cloud_components(liquid, ice)


__all__ = [
    "compute_longwave_cloud_optics_byband",
]
