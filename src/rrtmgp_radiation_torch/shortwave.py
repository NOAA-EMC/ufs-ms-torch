# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Shortwave cloud optics by band (``rrtmgp_sw_cloud_optics``), the cloud look-up interpolation shared with the
longwave, and the two-stream increment of same-spectral optics (``inc_2stream_by_2stream``)."""

from __future__ import annotations

import torch

from .tables import RrtmgpTableBundle, RrtmgpTableFile, load_rrtmgp_tables

TensorOrString = torch.Tensor | str


_SHORTWAVE_CLOUD_TABLE_CACHE: dict[tuple[int, int], tuple[object, ...]] = {}


def _to_real(value: TensorOrString, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64 (the working dtype), got {value.dtype}")
    return value


def _increment_shortwave_same_spectral_kernel(
    tau1: torch.Tensor,
    ssa1: torch.Tensor,
    g1: torch.Tensor,
    tau2: torch.Tensor,
    ssa2: torch.Tensor,
    g2: torch.Tensor,
    eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse same-grid two-stream property combination."""

    tau12 = tau1 + tau2
    tauscat12 = tau1 * ssa1 + tau2 * ssa2
    g12 = (tau1 * ssa1 * g1 + tau2 * ssa2 * g2) / torch.maximum(eps, tauscat12)
    ssa12 = tauscat12 / torch.maximum(eps, tau12)
    return tau12, ssa12, g12


def _interp_cloud_table(
    water_path: torch.Tensor,
    radius: torch.Tensor,
    ext_table: torch.Tensor,
    ssa_table: torch.Tensor,
    asy_table: torch.Tensor,
    *,
    offset: torch.Tensor,
    step_size: float | torch.Tensor,
    band_axis: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """By-band optics ``[nband, nlay, ncol]``; ``band_axis=1``: ``[nlay, nband, ncol]``, the same elements (a compiled
    consumer that stores them level-major then computes them in that order)."""
    wp = _to_real(water_path, name="water_path")
    re = _to_real(radius, name="radius").to(device=wp.device)
    ext = ext_table.to(device=wp.device, dtype=wp.dtype)
    ssa = ssa_table.to(device=wp.device, dtype=wp.dtype)
    asy = asy_table.to(device=wp.device, dtype=wp.dtype)
    if wp.shape != re.shape or wp.ndim != 2:
        raise ValueError("cloud water path and radius must have identical [nlay,ncol] shapes")
    if ext.shape != ssa.shape or ext.shape != asy.shape or ext.ndim != 2:
        raise ValueError("cloud LUT tables must use [nsize,nband] shapes")
    nsteps, nband = ext.shape
    step = torch.as_tensor(step_size, dtype=wp.dtype, device=wp.device)
    x = (re - offset.to(device=wp.device, dtype=wp.dtype)) / step
    index_1based = torch.floor(x).to(dtype=torch.int64) + 1
    index_1based = torch.minimum(index_1based, torch.full_like(index_1based, nsteps - 1))
    index_1based = torch.maximum(index_1based, torch.ones_like(index_1based))
    index0 = index_1based - 1
    fint = x - (index_1based.to(dtype=wp.dtype) - 1.0)
    flat_index = index0.reshape(-1)
    lower_index = flat_index
    upper_index = flat_index + 1
    shape = (*wp.shape, nband)
    order = (2, 0, 1) if band_axis == 0 else (0, 2, 1)
    ext0 = ext.index_select(0, lower_index).reshape(shape).permute(order)
    ext1 = ext.index_select(0, upper_index).reshape(shape).permute(order)
    ssa0 = ssa.index_select(0, lower_index).reshape(shape).permute(order)
    ssa1 = ssa.index_select(0, upper_index).reshape(shape).permute(order)
    asy0 = asy.index_select(0, lower_index).reshape(shape).permute(order)
    asy1 = asy.index_select(0, upper_index).reshape(shape).permute(order)
    fint3 = fint.unsqueeze(band_axis)
    tau_raw = wp.unsqueeze(band_axis) * (ext0 + fint3 * (ext1 - ext0))
    taussa_raw = tau_raw * (ssa0 + fint3 * (ssa1 - ssa0))
    taussag_raw = taussa_raw * (asy0 + fint3 * (asy1 - asy0))
    mask = (wp > 0.0).unsqueeze(band_axis)
    zeros = torch.zeros_like(tau_raw)
    return (
        torch.where(mask, tau_raw, zeros),
        torch.where(mask, taussa_raw, zeros),
        torch.where(mask, taussag_raw, zeros),
    )


def _combine_cloud_components(
    liquid: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ice: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ltau, ltaussa, ltaussag = liquid
    itau, itaussa, itaussag = ice
    tau = ltau + itau
    taussa = ltaussa + itaussa
    taussag = ltaussag + itaussag
    eps = torch.full((), torch.finfo(tau.dtype).eps, dtype=tau.dtype, device=tau.device)
    g = taussag / torch.maximum(eps, taussa)
    ssa = taussa / torch.maximum(eps, tau)
    return tau, ssa, g


def _sw_cloud_table(bundle: RrtmgpTableBundle | None, *, device: torch.device) -> RrtmgpTableFile:
    tables = bundle if bundle is not None else load_rrtmgp_tables(device=device)
    return tables.files["sw_cloud"]


@torch.inference_mode(False)
def _shortwave_cloud_table_inputs(
    table: RrtmgpTableFile,
    *,
    ice_roughness_index: int | None,
) -> tuple[object, ...]:
    """Prepare immutable SW cloud LUT views and device scalars once per table."""

    tensors = table.tensors
    roughness = int(table.dimensions.get("nrghice", tensors["lut_extice"].shape[0])) - 1
    if ice_roughness_index is not None:
        roughness = int(ice_roughness_index)
    key = (id(table), roughness)
    cached = _SHORTWAVE_CLOUD_TABLE_CACHE.get(key)
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
    prepared: tuple[object, ...] = (
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
    _SHORTWAVE_CLOUD_TABLE_CACHE[key] = prepared
    return prepared


def compute_shortwave_cloud_optics_byband(
    cld_lwp: torch.Tensor,
    cld_iwp: torch.Tensor,
    cld_reliq: torch.Tensor,
    cld_reice: torch.Tensor,
    *,
    tables: RrtmgpTableBundle | None = None,
    ice_roughness_index: int | None = None,
    band_axis: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute `sw_cloud_props%cloud_optics` by band for liquid plus ice (``band_axis``: ``_interp_cloud_table``)."""

    lwp = _to_real(cld_lwp, name="cld_lwp")
    table = _sw_cloud_table(tables, device=lwp.device)
    (
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
    ) = _shortwave_cloud_table_inputs(table, ice_roughness_index=ice_roughness_index)
    liquid = _interp_cloud_table(
        lwp,
        cld_reliq,
        liq_ext,
        liq_ssa,
        liq_asy,
        offset=radliq_lwr,
        step_size=liq_step,
        band_axis=band_axis,
    )
    ice = _interp_cloud_table(
        cld_iwp,
        cld_reice,
        ice_ext,
        ice_ssa,
        ice_asy,
        offset=radice_lwr,
        step_size=ice_step,
        band_axis=band_axis,
    )
    return _combine_cloud_components(liquid, ice)


__all__ = [
    "compute_shortwave_cloud_optics_byband",
]
