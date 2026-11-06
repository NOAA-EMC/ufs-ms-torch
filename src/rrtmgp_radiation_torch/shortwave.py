# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""PyTorch-native shortwave RRTMGP kernels for currently closed branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch.utils.checkpoint import checkpoint

from .compile import compile_categories_enabled, run_compile_island
from .gas_optics import compute_shortwave_gas_optics_from_boundaries, reduce_gas_optics_table
from .tables import RrtmgpTableBundle, RrtmgpTableFile, load_rrtmgp_tables

TensorOrString = torch.Tensor | str
_CHECKPOINT_LAYER_COEFFICIENTS = False


def set_checkpoint_layer_coefficients(enabled: bool) -> bool:
    """Opt in to recomputing eager AD layer intermediates during backward.

    The same pure coefficient function runs with non-reentrant native Torch
    checkpointing. Compiled slab kernels and forward-only calls are unchanged.
    """
    if not isinstance(enabled, bool):
        raise TypeError("layer-coefficient checkpoint policy must be bool")
    global _CHECKPOINT_LAYER_COEFFICIENTS
    previous = _CHECKPOINT_LAYER_COEFFICIENTS
    _CHECKPOINT_LAYER_COEFFICIENTS = enabled
    return previous


_SHORTWAVE_CLOUD_TABLE_CACHE: dict[tuple[int, int], tuple[object, ...]] = {}
_SHORTWAVE_PRECIP_CONSTANT_CACHE: dict[
    tuple[str, int | None, torch.dtype], tuple[torch.Tensor, ...]
] = {}
_SHORTWAVE_SURFACE_PLAN_CACHE: dict[
    tuple[int, str, int | None, torch.dtype],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int],
] = {}


@dataclass(frozen=True)
class RrtmgpShortwaveResult:
    """Checkpoint-shaped shortwave-H output fields."""

    fields: Mapping[str, TensorOrString]
    runtime_kind: str = "torch_rrtmgp_shortwave"


@dataclass(frozen=True)
class ShortwaveClearSkyOpticsResult:
    """Shortwave gas/Rayleigh optics after by-band aerosol increment."""

    tau: torch.Tensor
    ssa: torch.Tensor
    g: torch.Tensor
    runtime_kind: str = "torch_rrtmgp_shortwave_clear_sky_aerosol_increment"


@dataclass(frozen=True)
class ShortwaveCloudOpticsResult:
    """Shortwave cloud and precipitation by-band optical properties."""

    cloud_tau: torch.Tensor | None
    cloud_ssa: torch.Tensor | None
    cloud_g: torch.Tensor | None
    precip_tau: torch.Tensor | None
    precip_ssa: torch.Tensor | None
    precip_g: torch.Tensor | None
    cloud_precip_tau: torch.Tensor
    cloud_precip_ssa: torch.Tensor
    cloud_precip_g: torch.Tensor
    cloud_tau_band10: torch.Tensor | None = None
    runtime_kind: str = "torch_rrtmgp_shortwave_cloud_precip_optics"


@dataclass(frozen=True)
class ShortwaveSampledCloudOpticsResult:
    """McICA-sampled shortwave cloud optical properties by g-point."""

    tau: torch.Tensor
    ssa: torch.Tensor
    g: torch.Tensor
    runtime_kind: str = "torch_rrtmgp_shortwave_cloud_draw_samples"


@dataclass(frozen=True)
class ShortwaveMcicaMaskResult:
    """McICA cloud mask generated from Fortran-compatible random streams."""

    mask: torch.Tensor
    mode: str
    runtime_kind: str = "torch_rrtmgp_shortwave_sampled_mask"


@dataclass(frozen=True)
class ShortwaveSurfaceAlbedoResult:
    """Shortwave surface albedo mapped from host NIR/UV-visible fields to RRTMGP bands."""

    sfc_alb_dir: torch.Tensor
    sfc_alb_dif: torch.Tensor
    ibd: int
    ibd_uv: int
    runtime_kind: str = "torch_rrtmgp_shortwave_surface_albedo_byband"


@dataclass(frozen=True)
class ShortwaveFluxByBandResult:
    """Shortwave RTE fluxes reduced to band-by-band checkpoint layout."""

    flux_up: torch.Tensor
    flux_down: torch.Tensor
    flux_down_direct: torch.Tensor
    runtime_kind: str = "torch_rrtmgp_shortwave_rte_byband"


def _to_int(value: TensorOrString, *, name: str) -> int:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    return int(value.detach().to(device="cpu", dtype=torch.int64).reshape(-1)[0].item())


def _to_real(value: TensorOrString, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64 (the working dtype), got {value.dtype}")
    return value


def _normalize_randoms_and_cloud_fraction(
    randoms: torch.Tensor,
    cloud_frac: torch.Tensor,
    *,
    randoms_name: str,
    cloud_frac_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = _to_real(randoms, name=randoms_name)
    frac = _to_real(cloud_frac, name=cloud_frac_name).to(device=rng.device)
    if rng.ndim != 3 or frac.ndim != 2:
        raise ValueError(f"{randoms_name} must be 3-D and {cloud_frac_name} must be 2-D")
    if frac.shape == (rng.shape[1], rng.shape[2]):
        return rng, frac
    if frac.shape == (rng.shape[2], rng.shape[1]):
        return rng, frac.transpose(0, 1).contiguous()
    if frac.shape == (rng.shape[1], rng.shape[0]):
        return rng.permute(2, 1, 0).contiguous(), frac
    if frac.shape == (rng.shape[0], rng.shape[1]):
        return rng.permute(2, 1, 0).contiguous(), frac.transpose(0, 1).contiguous()
    raise ValueError(
        f"cannot align {randoms_name} shape {tuple(rng.shape)} with "
        f"{cloud_frac_name} shape {tuple(frac.shape)}"
    )


def _normalize_randoms_like(randoms: torch.Tensor, *, ngpt: int, nlay: int, ncol: int, name: str) -> torch.Tensor:
    rng = _to_real(randoms, name=name)
    if rng.shape == (ngpt, nlay, ncol):
        return rng
    if rng.shape == (ncol, nlay, ngpt):
        return rng.permute(2, 1, 0).contiguous()
    raise ValueError(f"{name} shape {tuple(rng.shape)} cannot align with [ngpt,nlay,ncol]={ngpt,nlay,ncol}")


def _normalize_overlap(overlap_param: torch.Tensor, *, nlay: int, ncol: int) -> torch.Tensor:
    overlap = _to_real(overlap_param, name="overlap_param")
    if overlap.shape == (nlay - 1, ncol):
        return overlap
    if overlap.shape == (nlay, ncol):
        return overlap[: nlay - 1, :]
    if overlap.shape == (ncol, nlay - 1):
        return overlap.transpose(0, 1).contiguous()
    if overlap.shape == (ncol, nlay):
        return overlap.transpose(0, 1).contiguous()[: nlay - 1, :]
    raise ValueError(f"overlap_param shape {tuple(overlap.shape)} cannot align with nlay={nlay}, ncol={ncol}")


def _increment_shortwave_by_band_kernel(
    tau1: torch.Tensor,
    ssa1: torch.Tensor,
    g1: torch.Tensor,
    tau2_band: torch.Tensor,
    ssa2_band: torch.Tensor,
    g2_band: torch.Tensor,
    bands: torch.Tensor,
    eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse by-band gather and two-stream property combination."""

    tau2 = tau2_band.index_select(0, bands)
    ssa2 = ssa2_band.index_select(0, bands)
    g2 = g2_band.index_select(0, bands)
    tau12 = tau1 + tau2
    tauscat12 = tau1 * ssa1 + tau2 * ssa2
    g12 = (tau1 * ssa1 * g1 + tau2 * ssa2 * g2) / torch.maximum(eps, tauscat12)
    ssa12 = tauscat12 / torch.maximum(eps, tau12)
    return tau12, ssa12, g12


def increment_shortwave_2stream_by_band(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    g: torch.Tensor,
    tau_byband: torch.Tensor,
    ssa_byband: torch.Tensor,
    g_byband: torch.Tensor,
    gpt_to_band: torch.Tensor,
) -> ShortwaveClearSkyOpticsResult:
    """Match RTE `inc_2stream_by_2stream_bybnd` for SW aerosol increments.

    The gas/Rayleigh optical properties are defined by g-point with shape
    `[ngpt,nlay,ncol]`; OPAC aerosol properties are defined by band with shape
    `[nband,nlay,ncol]`.
    """

    tau1 = _to_real(tau, name="tau")
    ssa1 = _to_real(ssa, name="ssa").to(device=tau1.device)
    g1 = _to_real(g, name="g").to(device=tau1.device)
    tau2_band = _to_real(tau_byband, name="tau_byband").to(device=tau1.device)
    ssa2_band = _to_real(ssa_byband, name="ssa_byband").to(device=tau1.device)
    g2_band = _to_real(g_byband, name="g_byband").to(device=tau1.device)
    if tau1.shape != ssa1.shape or tau1.shape != g1.shape:
        raise ValueError("g-point tau, ssa, and g must have identical shapes")
    if tau2_band.shape != ssa2_band.shape or tau2_band.shape != g2_band.shape:
        raise ValueError("by-band tau, ssa, and g must have identical shapes")
    if tau1.ndim != 3 or tau2_band.ndim != 3:
        raise ValueError("shortwave optical properties must use [spectral,nlay,ncol] tensors")
    if tau1.shape[1:] != tau2_band.shape[1:]:
        raise ValueError(f"g-point and by-band vertical/column shapes differ: {tau1.shape} vs {tau2_band.shape}")
    raw_bands = gpt_to_band.to(dtype=torch.int64).reshape(-1)
    bands = raw_bands.to(device=tau1.device)
    if bands.numel() != tau1.shape[0]:
        raise ValueError(f"gpt_to_band length {bands.numel()} does not match ngpt {tau1.shape[0]}")
    torch._assert_async(torch.all((bands >= 0) & (bands < tau2_band.shape[0])),
                        "gpt_to_band contains a band index outside the by-band optical-property tensor")

    eps = torch.tensor(3.0 * torch.finfo(tau1.dtype).tiny, dtype=tau1.dtype, device=tau1.device)
    tau12, ssa12, g12 = run_compile_island(
        "shortwave_transport",
        "increment_shortwave_by_band",
        _increment_shortwave_by_band_kernel,
        tau1,
        ssa1,
        g1,
        tau2_band,
        ssa2_band,
        g2_band,
        bands,
        eps,
    )
    return ShortwaveClearSkyOpticsResult(tau=tau12, ssa=ssa12, g=g12)


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


def _same_spectral_increment(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    g: torch.Tensor,
    tau2: torch.Tensor,
    ssa2: torch.Tensor,
    g2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tau1 = _to_real(tau, name="tau")
    ssa1 = _to_real(ssa, name="ssa").to(device=tau1.device)
    g1 = _to_real(g, name="g").to(device=tau1.device)
    tau_add = _to_real(tau2, name="tau2").to(device=tau1.device)
    ssa_add = _to_real(ssa2, name="ssa2").to(device=tau1.device)
    g_add = _to_real(g2, name="g2").to(device=tau1.device)
    if tau1.shape != ssa1.shape or tau1.shape != g1.shape or tau1.shape != tau_add.shape:
        raise ValueError("same-spectral two-stream increment tensors must have identical shapes")
    eps = torch.tensor(3.0 * torch.finfo(tau1.dtype).tiny, dtype=tau1.dtype, device=tau1.device)
    if torch.compiler.is_compiling():
        return _increment_shortwave_same_spectral_kernel(
            tau1,
            ssa1,
            g1,
            tau_add,
            ssa_add,
            g_add,
            eps,
        )
    return run_compile_island(
        "shortwave_transport",
        "increment_shortwave_same_spectral",
        _increment_shortwave_same_spectral_kernel,
        tau1,
        ssa1,
        g1,
        tau_add,
        ssa_add,
        g_add,
        eps,
    )


def increment_shortwave_2stream_same_spectral(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    g: torch.Tensor,
    tau2: torch.Tensor,
    ssa2: torch.Tensor,
    g2: torch.Tensor,
) -> ShortwaveClearSkyOpticsResult:
    """Match RTE `inc_2stream_by_2stream` for same-g-point SW optics."""

    tau12, ssa12, g12 = _same_spectral_increment(tau, ssa, g, tau2, ssa2, g2)
    return ShortwaveClearSkyOpticsResult(tau=tau12, ssa=ssa12, g=g12)


def _interp_cloud_table(
    water_path: torch.Tensor,
    radius: torch.Tensor,
    ext_table: torch.Tensor,
    ssa_table: torch.Tensor,
    asy_table: torch.Tensor,
    *,
    offset: torch.Tensor,
    step_size: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    ext0 = ext.index_select(0, lower_index).reshape(shape).permute(2, 0, 1)
    ext1 = ext.index_select(0, upper_index).reshape(shape).permute(2, 0, 1)
    ssa0 = ssa.index_select(0, lower_index).reshape(shape).permute(2, 0, 1)
    ssa1 = ssa.index_select(0, upper_index).reshape(shape).permute(2, 0, 1)
    asy0 = asy.index_select(0, lower_index).reshape(shape).permute(2, 0, 1)
    asy1 = asy.index_select(0, upper_index).reshape(shape).permute(2, 0, 1)
    fint3 = fint.unsqueeze(0)
    tau_raw = wp.unsqueeze(0) * (ext0 + fint3 * (ext1 - ext0))
    taussa_raw = tau_raw * (ssa0 + fint3 * (ssa1 - ssa0))
    taussag_raw = taussa_raw * (asy0 + fint3 * (asy1 - asy0))
    mask = (wp > 0.0).unsqueeze(0)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute `sw_cloud_props%cloud_optics` by band for liquid plus ice."""

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
    )
    ice = _interp_cloud_table(
        cld_iwp,
        cld_reice,
        ice_ext,
        ice_ssa,
        ice_asy,
        offset=radice_lwr,
        step_size=ice_step,
    )
    return _combine_cloud_components(liquid, ice)


def compute_shortwave_precip_optics_byband(
    cld_frac: torch.Tensor,
    cld_rwp: torch.Tensor,
    cld_swp: torch.Tensor,
    cld_resnow: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute RRTMGP SW rain/snow precipitation optics by band."""

    frac = _to_real(cld_frac, name="cld_frac")
    rwp = _to_real(cld_rwp, name="cld_rwp").to(device=frac.device)
    swp = _to_real(cld_swp, name="cld_swp").to(device=frac.device)
    resnow = _to_real(cld_resnow, name="cld_resnow").to(device=frac.device)
    if frac.shape != rwp.shape or frac.shape != swp.shape or frac.shape != resnow.shape:
        raise ValueError("precip cloud fields must share [nlay,ncol] shapes")
    return _shortwave_precip_optics_kernel(
        frac,
        rwp,
        swp,
        resnow,
        *_shortwave_precip_constants(dtype=frac.dtype, device=frac.device),
    )


@torch.inference_mode(False)
def _shortwave_precip_constants(
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    key = (device.type, device.index, dtype)
    cached = _SHORTWAVE_PRECIP_CONSTANT_CACHE.get(key)
    if cached is not None:
        return cached
    values = (
        torch.tensor(
        [0.496, 0.466, 0.437, 0.416, 0.391, 0.374, 0.352, 0.183, 0.048, 0.012, 0.0, 0.0, 0.0, 0.0],
        dtype=dtype,
        device=device,
        ),
        torch.tensor(
        [0.460, 0.460, 0.460, 0.460, 0.460, 0.460, 0.460, 0.460, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=dtype,
        device=device,
        ),
        torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.62e-5, 1.62e-5, 0.0, 0.0, 0.0, 0.0],
        dtype=dtype,
        device=device,
        ),
        torch.tensor(
        [0.980, 0.975, 0.965, 0.960, 0.955, 0.952, 0.950, 0.944, 0.894, 0.884, 0.883, 0.883, 0.883, 0.883],
        dtype=dtype,
        device=device,
        ),
        torch.tensor(
        [0.970, 0.970, 0.970, 0.970, 0.970, 0.970, 0.970, 0.970, 0.970, 0.970, 0.700, 0.700, 0.700, 0.700],
        dtype=dtype,
        device=device,
        ),
    )
    _SHORTWAVE_PRECIP_CONSTANT_CACHE[key] = values
    return values


def _shortwave_precip_optics_kernel(
    frac: torch.Tensor,
    rwp: torch.Tensor,
    swp: torch.Tensor,
    resnow: torch.Tensor,
    b0r: torch.Tensor,
    b0s: torch.Tensor,
    b1s: torch.Tensor,
    c0r: torch.Tensor,
    c0s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure tensor SW rain/snow optics used by eager and compiled callers."""

    dtype = frac.dtype
    device = frac.device
    active = (frac > 1.0e-300).unsqueeze(0)
    tau_rain = rwp * 3.07e-3
    tau_snow = torch.where(
        (swp > 0.0) & (resnow > 10.0),
        swp * 1.09087 * (1.5 / (1.0315 * torch.where((swp > 0.0) & (resnow > 10.0), resnow, 20.0))),
        torch.zeros_like(swp),
    )
    floor = torch.tensor(1.0e-12, dtype=dtype, device=device)
    one_minus = torch.tensor(1.0 - 0.000001, dtype=dtype, device=device)
    b0r3 = b0r.reshape(-1, 1, 1)
    b0s3 = b0s.reshape(-1, 1, 1)
    b1s3 = b1s.reshape(-1, 1, 1)
    c0r3 = c0r.reshape(-1, 1, 1)
    c0s3 = c0s.reshape(-1, 1, 1)
    tau_rain3 = tau_rain.unsqueeze(0)
    tau_snow3 = tau_snow.unsqueeze(0)
    resnow3 = resnow.unsqueeze(0)
    ssa_rain = tau_rain3 * (1.0 - b0r3)
    asy_rain = ssa_rain * c0r3
    ssa_snow = tau_snow3 * (1.0 - (b0s3 + b1s3 * 1.0315 * resnow3))
    asy_snow = ssa_snow * c0s3
    tau_prec = torch.maximum(floor, tau_rain3 + tau_snow3)
    ssa_prec = torch.maximum(floor, ssa_rain + ssa_snow)
    asy_prec = torch.maximum(floor, asy_rain + asy_snow)
    asyw = asy_prec / torch.maximum(floor, ssa_prec)
    ssaw = torch.minimum(one_minus, ssa_prec / tau_prec)
    za1 = asyw * asyw
    za2 = ssaw * za1
    tau_band = (1.0 - za2) * tau_prec
    ssa_band = (ssaw - za2) / (1.0 - za2)
    asym_band = asyw / (1.0 + asyw)
    zeros = torch.zeros_like(tau_band)
    return (
        torch.where(active, tau_band, zeros),
        torch.where(active, ssa_band, zeros),
        torch.where(active, asym_band, zeros),
    )


def _shortwave_cloud_precip_kernel(
    lwp: torch.Tensor,
    iwp: torch.Tensor,
    reliq: torch.Tensor,
    reice: torch.Tensor,
    frac: torch.Tensor,
    rwp: torch.Tensor,
    swp: torch.Tensor,
    resnow: torch.Tensor,
    liq_ext: torch.Tensor,
    liq_ssa: torch.Tensor,
    liq_asy: torch.Tensor,
    ice_ext: torch.Tensor,
    ice_ssa: torch.Tensor,
    ice_asy: torch.Tensor,
    radliq_lwr: torch.Tensor,
    radice_lwr: torch.Tensor,
    b0r: torch.Tensor,
    b0s: torch.Tensor,
    b1s: torch.Tensor,
    c0r: torch.Tensor,
    c0s: torch.Tensor,
    liq_step: float,
    ice_step: float,
) -> tuple[torch.Tensor, ...]:
    liquid = _interp_cloud_table(
        lwp,
        reliq,
        liq_ext,
        liq_ssa,
        liq_asy,
        offset=radliq_lwr,
        step_size=liq_step,
    )
    ice = _interp_cloud_table(
        iwp,
        reice,
        ice_ext,
        ice_ssa,
        ice_asy,
        offset=radice_lwr,
        step_size=ice_step,
    )
    cloud_tau, cloud_ssa, cloud_g = _combine_cloud_components(liquid, ice)
    precip_tau, precip_ssa, precip_g = _shortwave_precip_optics_kernel(
        frac,
        rwp,
        swp,
        resnow,
        b0r,
        b0s,
        b1s,
        c0r,
        c0s,
    )
    total_tau, total_ssa, total_g = _same_spectral_increment(
        cloud_tau,
        cloud_ssa,
        cloud_g,
        precip_tau,
        precip_ssa,
        precip_g,
    )
    return (
        cloud_tau,
        cloud_ssa,
        cloud_g,
        precip_tau,
        precip_ssa,
        precip_g,
        total_tau,
        total_ssa,
        total_g,
    )


def _shortwave_cloud_precip_minimal_kernel(
    lwp: torch.Tensor,
    iwp: torch.Tensor,
    reliq: torch.Tensor,
    reice: torch.Tensor,
    frac: torch.Tensor,
    rwp: torch.Tensor,
    swp: torch.Tensor,
    resnow: torch.Tensor,
    liq_ext: torch.Tensor,
    liq_ssa: torch.Tensor,
    liq_asy: torch.Tensor,
    ice_ext: torch.Tensor,
    ice_ssa: torch.Tensor,
    ice_asy: torch.Tensor,
    radliq_lwr: torch.Tensor,
    radice_lwr: torch.Tensor,
    b0r: torch.Tensor,
    b0s: torch.Tensor,
    b1s: torch.Tensor,
    c0r: torch.Tensor,
    c0s: torch.Tensor,
    liq_step: float,
    ice_step: float,
) -> tuple[torch.Tensor, ...]:
    values = _shortwave_cloud_precip_kernel(
        lwp, iwp, reliq, reice, frac, rwp, swp, resnow,
        liq_ext, liq_ssa, liq_asy, ice_ext, ice_ssa, ice_asy,
        radliq_lwr, radice_lwr, b0r, b0s, b1s, c0r, c0s,
        liq_step, ice_step,
    )
    return values[6], values[7], values[8], values[0][10, :, :].clone()


def _compute_shortwave_cloud_precip_optics_compiled(
    cloud: Mapping[str, TensorOrString],
    *,
    tables: RrtmgpTableBundle | None,
    retain_components: bool,
) -> tuple[torch.Tensor, ...]:
    lwp = _to_real(cloud["cld_lwp"], name="cld_lwp")
    device = lwp.device
    bundle = tables if tables is not None else load_rrtmgp_tables(device=device)
    prepared = _shortwave_cloud_table_inputs(bundle.files["sw_cloud"], ice_roughness_index=None)
    precip_constants = _shortwave_precip_constants(dtype=lwp.dtype, device=device)
    args = (
        lwp,
        _to_real(cloud["cld_iwp"], name="cld_iwp").to(device=device),
        _to_real(cloud["cld_reliq"], name="cld_reliq").to(device=device),
        _to_real(cloud["cld_reice"], name="cld_reice").to(device=device),
        _to_real(cloud["cld_frac"], name="cld_frac").to(device=device),
        _to_real(cloud["cld_rwp"], name="cld_rwp").to(device=device),
        _to_real(cloud["cld_swp"], name="cld_swp").to(device=device),
        _to_real(cloud["cld_resnow"], name="cld_resnow").to(device=device),
        *prepared[:8],
        *precip_constants,
        prepared[8],
        prepared[9],
    )
    if retain_components:
        return run_compile_island(
            "shortwave_transport",
            "sw_cloud_precip_optics",
            _shortwave_cloud_precip_kernel,
            *args,
        )
    return run_compile_island(
        "shortwave_transport",
        "sw_cloud_precip_optics_minimal",
        _shortwave_cloud_precip_minimal_kernel,
        *args,
    )


def compute_shortwave_cloud_precip_optics_from_boundaries(
    boundaries: Mapping[str, Mapping[str, TensorOrString]],
    *,
    tables: RrtmgpTableBundle | None = None,
    retain_components: bool = True,
) -> ShortwaveCloudOpticsResult:
    """Compute active SW cloud and precip by-band optics from ABCD fields."""

    cloud = boundaries["ABCD"]
    cloud_tau_band10: torch.Tensor | None = None
    if "shortwave_transport" in compile_categories_enabled():
        values = _compute_shortwave_cloud_precip_optics_compiled(
            cloud,
            tables=tables,
            retain_components=retain_components,
        )
        if retain_components:
            cloud_tau, cloud_ssa, cloud_g, precip_tau, precip_ssa, precip_g, total_tau, total_ssa, total_g = values
        else:
            total_tau, total_ssa, total_g, cloud_tau_band10 = values
            cloud_tau = cloud_ssa = cloud_g = precip_tau = precip_ssa = precip_g = None
    else:
        cloud_tau, cloud_ssa, cloud_g = compute_shortwave_cloud_optics_byband(
            cloud["cld_lwp"],  # type: ignore[arg-type]
            cloud["cld_iwp"],  # type: ignore[arg-type]
            cloud["cld_reliq"],  # type: ignore[arg-type]
            cloud["cld_reice"],  # type: ignore[arg-type]
            tables=tables,
        )
        precip_tau, precip_ssa, precip_g = compute_shortwave_precip_optics_byband(
            cloud["cld_frac"],  # type: ignore[arg-type]
            cloud["cld_rwp"],  # type: ignore[arg-type]
            cloud["cld_swp"],  # type: ignore[arg-type]
            cloud["cld_resnow"],  # type: ignore[arg-type]
        )
        total_tau, total_ssa, total_g = _same_spectral_increment(
            cloud_tau,
            cloud_ssa,
            cloud_g,
            precip_tau,
            precip_ssa,
            precip_g,
        )
        if not retain_components:
            cloud_tau_band10 = cloud_tau[10, :, :].clone()
            cloud_tau = cloud_ssa = cloud_g = precip_tau = precip_ssa = precip_g = None
    return ShortwaveCloudOpticsResult(
        cloud_tau=cloud_tau,
        cloud_ssa=cloud_ssa,
        cloud_g=cloud_g,
        precip_tau=precip_tau,
        precip_ssa=precip_ssa,
        precip_g=precip_g,
        cloud_precip_tau=total_tau,
        cloud_precip_ssa=total_ssa,
        cloud_precip_g=total_g,
        cloud_tau_band10=cloud_tau_band10,
    )


def _sample_mcica_decorrelation_kernel(
    rng: torch.Tensor,
    frac: torch.Tensor,
    overlap: torch.Tensor,
    rng_second: torch.Tensor,
) -> torch.Tensor:
    """Generate the decorrelation-length mask from flat tensor inputs."""

    nlay, ncol = frac.shape
    cloud = frac > 0.0
    seen_cloud = torch.zeros((ncol,), dtype=torch.bool, device=rng.device)
    layers: list[torch.Tensor] = []
    for ilay in range(nlay):
        is_cloud = cloud[ilay, :]
        threshold = 1.0 - frac[ilay, :].unsqueeze(0)
        rng_lay = rng[:, ilay, :]
        first_cloud = is_cloud & (~seen_cloud)
        if ilay == 0:
            layer_mask = is_cloud.unsqueeze(0) & (rng_lay > threshold)
        else:
            use_prev = rng_second[:, ilay, :] <= overlap[ilay - 1, :].unsqueeze(0)
            decorrelated = torch.where(use_prev, rng[:, ilay - 1, :], rng_lay) > threshold
            first_decision = rng_lay > threshold
            layer_mask = is_cloud.unsqueeze(0) & torch.where(
                first_cloud.unsqueeze(0), first_decision, decorrelated
            )
        layers.append(layer_mask)
        seen_cloud = seen_cloud | is_cloud
    return torch.stack(layers, dim=1)


def sample_shortwave_mcica_mask(
    randoms: torch.Tensor,
    cloud_frac: torch.Tensor,
    *,
    overlap_param: torch.Tensor | None = None,
    randoms2: torch.Tensor | None = None,
) -> ShortwaveMcicaMaskResult:
    """Generate the RRTMGP McICA sampled cloud mask from supplied random streams.

    This ports `rrtmgp_sampling:sampled_mask`.  The random streams are generated
    outside this routine, as in Fortran; this function owns the overlap-policy
    recurrence that converts those streams plus cloud fraction into a
    checkpoint-shaped `[ngpt,nlay,ncol]` boolean mask.
    """

    rng, frac = _normalize_randoms_and_cloud_fraction(
        randoms,
        cloud_frac,
        randoms_name="randoms",
        cloud_frac_name="cloud_frac",
    )
    ngpt, nlay, ncol = rng.shape
    cloud = frac > 0.0
    layers: list[torch.Tensor] = []

    if overlap_param is None:
        mode = "maximum_random"
        local = rng[:, 0, :]
        prev_cloud = torch.zeros((ncol,), dtype=torch.bool, device=rng.device)
        for ilay in range(nlay):
            is_cloud = cloud[ilay, :]
            reset = is_cloud & (~prev_cloud)
            rng_lay = rng[:, ilay, :]
            local = torch.where(reset.unsqueeze(0), rng_lay, local)
            threshold = 1.0 - frac[ilay, :].unsqueeze(0)
            layers.append(is_cloud.unsqueeze(0) & (local > threshold))
            prev_cloud = is_cloud
        return ShortwaveMcicaMaskResult(mask=torch.stack(layers, dim=1), mode=mode)

    overlap = _normalize_overlap(overlap_param.to(device=rng.device), nlay=nlay, ncol=ncol)
    if randoms2 is None:
        mode = "exponential"
        local = rng[:, 0, :]
        prev_cloud = torch.zeros((ncol,), dtype=torch.bool, device=rng.device)
        for ilay in range(nlay):
            is_cloud = cloud[ilay, :]
            rng_lay = rng[:, ilay, :]
            if ilay == 0:
                candidate = rng_lay
            else:
                rho = overlap[ilay - 1, :].unsqueeze(0)
                correlated = rho * (local - 0.5) + torch.sqrt(1.0 - rho * rho) * (rng_lay - 0.5) + 0.5
                candidate = torch.where(prev_cloud.unsqueeze(0), correlated, rng_lay)
            local = torch.where(is_cloud.unsqueeze(0), candidate, local)
            threshold = 1.0 - frac[ilay, :].unsqueeze(0)
            layers.append(is_cloud.unsqueeze(0) & (local > threshold))
            prev_cloud = is_cloud
        return ShortwaveMcicaMaskResult(mask=torch.stack(layers, dim=1), mode=mode)

    mode = "decorrelation_length"
    rng_second = _normalize_randoms_like(randoms2.to(device=rng.device), ngpt=ngpt, nlay=nlay, ncol=ncol, name="randoms2")
    mask = run_compile_island(
        "shortwave_transport",
        "sample_mcica_decorrelation",
        _sample_mcica_decorrelation_kernel,
        rng,
        frac,
        overlap,
        rng_second,
    )
    return ShortwaveMcicaMaskResult(mask=mask, mode=mode)


def _sample_cloud_optics_by_gpoint_tensor(
    mask: torch.Tensor,
    tau_band: torch.Tensor,
    ssa_band: torch.Tensor,
    g_band: torch.Tensor,
    bands: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tau_gpt = tau_band.index_select(0, bands)
    ssa_gpt = ssa_band.index_select(0, bands)
    g_gpt = g_band.index_select(0, bands)
    zeros = torch.zeros_like(tau_gpt)
    return (
        torch.where(mask, tau_gpt, zeros),
        torch.where(mask, ssa_gpt, zeros),
        torch.where(mask, g_gpt, zeros),
    )


def sample_shortwave_cloud_optics_by_gpoint(
    mask_mcica: torch.Tensor,
    tau_byband: torch.Tensor,
    ssa_byband: torch.Tensor,
    g_byband: torch.Tensor,
    gpt_to_band: torch.Tensor,
) -> ShortwaveSampledCloudOpticsResult:
    """Apply RRTMGP `draw_samples` to by-band SW cloud optical properties."""

    mask = mask_mcica.to(dtype=torch.bool)
    tau_band = _to_real(tau_byband, name="tau_byband").to(device=mask.device)
    ssa_band = _to_real(ssa_byband, name="ssa_byband").to(device=mask.device)
    g_band = _to_real(g_byband, name="g_byband").to(device=mask.device)
    if tau_band.shape != ssa_band.shape or tau_band.shape != g_band.shape:
        raise ValueError("by-band cloud optical tensors must have identical shapes")
    if mask.ndim != 3 or tau_band.ndim != 3 or mask.shape[1:] != tau_band.shape[1:]:
        raise ValueError("mask must be [ngpt,nlay,ncol] and by-band fields [nband,nlay,ncol]")
    bands = gpt_to_band.to(device=mask.device, dtype=torch.int64).reshape(-1)
    if bands.numel() != mask.shape[0]:
        raise ValueError(f"gpt_to_band length {bands.numel()} does not match mask ngpt {mask.shape[0]}")
    tau_gpt, ssa_gpt, g_gpt = run_compile_island(
        "shortwave_transport",
        "sample_cloud_optics_by_gpoint",
        _sample_cloud_optics_by_gpoint_tensor,
        mask,
        tau_band,
        ssa_band,
        g_band,
        bands,
    )
    return ShortwaveSampledCloudOpticsResult(
        tau=tau_gpt,
        ssa=ssa_gpt,
        g=g_gpt,
    )


def _sample_and_increment_shortwave_tensor(
    mask: torch.Tensor,
    cloud_tau_band: torch.Tensor,
    cloud_ssa_band: torch.Tensor,
    cloud_g_band: torch.Tensor,
    bands: torch.Tensor,
    clear_tau: torch.Tensor,
    clear_ssa: torch.Tensor,
    clear_g: torch.Tensor,
    eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cloud_tau = torch.where(mask, cloud_tau_band.index_select(0, bands), 0.0)
    cloud_ssa = torch.where(mask, cloud_ssa_band.index_select(0, bands), 0.0)
    cloud_g = torch.where(mask, cloud_g_band.index_select(0, bands), 0.0)
    return _increment_shortwave_same_spectral_kernel(
        clear_tau,
        clear_ssa,
        clear_g,
        cloud_tau,
        cloud_ssa,
        cloud_g,
        eps,
    )


def _sample_and_increment_shortwave(
    mask: torch.Tensor,
    cloud_tau_band: torch.Tensor,
    cloud_ssa_band: torch.Tensor,
    cloud_g_band: torch.Tensor,
    gpt_to_band: torch.Tensor,
    clear_tau: torch.Tensor,
    clear_ssa: torch.Tensor,
    clear_g: torch.Tensor,
) -> ShortwaveClearSkyOpticsResult:
    bands = gpt_to_band.to(device=clear_tau.device, dtype=torch.int64).reshape(-1)
    eps = torch.tensor(3.0 * torch.finfo(clear_tau.dtype).tiny, dtype=clear_tau.dtype, device=clear_tau.device)
    tau, ssa, asymmetry = run_compile_island(
        "shortwave_transport",
        "sample_and_increment_shortwave",
        _sample_and_increment_shortwave_tensor,
        mask.to(dtype=torch.bool),
        cloud_tau_band,
        cloud_ssa_band,
        cloud_g_band,
        bands,
        clear_tau,
        clear_ssa,
        clear_g,
        eps,
    )
    return ShortwaveClearSkyOpticsResult(tau=tau, ssa=ssa, g=asymmetry)


def compute_shortwave_surface_albedo_byband(
    surface: Mapping[str, TensorOrString],
    *,
    table_bundle: RrtmgpTableBundle | None = None,
) -> ShortwaveSurfaceAlbedoResult:
    """Match `rrtmgp_sw_main` NIR/UV-visible surface-albedo band mapping."""

    bundle = table_bundle or load_rrtmgp_tables(device="cpu")
    table = bundle.files["sw_gas"]
    band_limits = table.tensors["bnd_limits_wavenumber"]
    starts = band_limits[:, 0]
    nband = int(starts.numel())

    nir_dir = _to_real(surface["sfc_alb_nir_dir"], name="sfc_alb_nir_dir").reshape(-1)
    nir_dif = _to_real(surface["sfc_alb_nir_dif"], name="sfc_alb_nir_dif").to(device=nir_dir.device).reshape(-1)
    uvvis_dir = _to_real(surface["sfc_alb_uvvis_dir"], name="sfc_alb_uvvis_dir").to(device=nir_dir.device).reshape(-1)
    uvvis_dif = _to_real(surface["sfc_alb_uvvis_dif"], name="sfc_alb_uvvis_dif").to(device=nir_dir.device).reshape(-1)
    if not (nir_dir.shape == nir_dif.shape == uvvis_dir.shape == uvvis_dif.shape):
        raise ValueError("surface albedo inputs must have the same column shape")

    plan_key = (id(table), nir_dir.device.type, nir_dir.device.index, nir_dir.dtype)
    plan = _SHORTWAVE_SURFACE_PLAN_CACHE.get(plan_key)
    if plan is None:
        # Immutable masks must remain saveable by autograd after inference.
        with torch.inference_mode(False):
            starts_cpu = starts.detach().to(device="cpu", dtype=torch.float64)
            transition_cpu = torch.nonzero(starts_cpu == 12850.0, as_tuple=False).reshape(-1)
            uvb_cpu = torch.nonzero(starts_cpu == 29000.0, as_tuple=False).reshape(-1)
            if transition_cpu.numel() != 1 or uvb_cpu.numel() != 1:
                raise ValueError("shortwave table lacks the expected transition or UV-B band")
            ibd = int(transition_cpu[0].item()) + 1
            ibd_uv = int(uvb_cpu[0].item()) + 1
            start = starts.to(device=nir_dir.device, dtype=nir_dir.dtype)
            plan = (start < 12850.0, start == 12850.0, start >= 16000.0, ibd, ibd_uv)
            _SHORTWAVE_SURFACE_PLAN_CACHE[plan_key] = plan
    nir_mask, transition_mask, uvvis_mask, ibd, ibd_uv = plan
    ncol = int(nir_dir.numel())
    nir_mask_2d = nir_mask.unsqueeze(1)
    transition_mask_2d = transition_mask.unsqueeze(1)
    uvvis_mask_2d = uvvis_mask.unsqueeze(1)
    zero_band_col = torch.zeros((nband, ncol), dtype=nir_dir.dtype, device=nir_dir.device)
    sfc_dir = torch.where(
        nir_mask_2d,
        nir_dir.unsqueeze(0),
        torch.where(
            transition_mask_2d,
            0.5 * (nir_dir + uvvis_dir).unsqueeze(0),
            torch.where(uvvis_mask_2d, uvvis_dir.unsqueeze(0), zero_band_col),
        ),
    )
    sfc_dif = torch.where(
        nir_mask_2d,
        nir_dif.unsqueeze(0),
        torch.where(
            transition_mask_2d,
            0.5 * (nir_dif + uvvis_dif).unsqueeze(0),
            torch.where(uvvis_mask_2d, uvvis_dif.unsqueeze(0), zero_band_col),
        ),
    )

    return ShortwaveSurfaceAlbedoResult(
        sfc_alb_dir=sfc_dir,
        sfc_alb_dif=sfc_dif,
        ibd=ibd,
        ibd_uv=ibd_uv,
    )


def _as_gpt_lay_col(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = _to_real(value, name=name)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be [ngpt,nlay,ncol], got {tuple(tensor.shape)}")
    return tensor


def _mu0_by_layer(mu0: torch.Tensor, *, nlay: int, ncol: int, device: torch.device) -> torch.Tensor:
    value = _to_real(mu0, name="mu0").to(device=device)
    if value.ndim == 1 and value.numel() == ncol:
        return value.reshape(1, ncol).expand(nlay, ncol)
    if value.ndim == 2 and value.shape == (nlay, ncol):
        return value
    if value.ndim == 2 and value.shape == (ncol, nlay):
        return value.transpose(0, 1).contiguous()
    raise ValueError(f"mu0 cannot broadcast to [nlay,ncol]={nlay,ncol}; got {tuple(value.shape)}")


def _band_col(value: torch.Tensor, *, name: str, nband: int, ncol: int, device: torch.device) -> torch.Tensor:
    tensor = _to_real(value, name=name).to(device=device)
    if tensor.shape == (nband, ncol):
        return tensor
    if tensor.shape == (ncol, nband):
        return tensor.transpose(0, 1).contiguous()
    raise ValueError(f"{name} cannot align with [nband,ncol]={nband,ncol}; got {tuple(tensor.shape)}")


def _gpoint_col(value: torch.Tensor, *, name: str, ngpt: int, ncol: int, device: torch.device) -> torch.Tensor:
    tensor = _to_real(value, name=name).to(device=device)
    if tensor.shape == (ngpt, ncol):
        return tensor
    if tensor.shape == (ncol, ngpt):
        return tensor.transpose(0, 1).contiguous()
    raise ValueError(f"{name} cannot align with [ngpt,ncol]={ngpt,ncol}; got {tuple(tensor.shape)}")


def _infer_nband_from_albedo(
    sfc_alb_dir: torch.Tensor,
    sfc_alb_dif: torch.Tensor,
    *,
    ncol: int,
) -> int:
    candidates: list[int] = []
    for name, value in (("sfc_alb_dir", sfc_alb_dir), ("sfc_alb_dif", sfc_alb_dif)):
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"{name} must have shape [nband,ncol] or [ncol,nband]")
        if value.shape[1] == ncol:
            candidates.append(int(value.shape[0]))
        elif value.shape[0] == ncol:
            candidates.append(int(value.shape[1]))
        else:
            raise ValueError(f"{name} cannot align with ncol={ncol}; got {tuple(value.shape)}")
    if candidates[0] != candidates[1]:
        raise ValueError("direct and diffuse shortwave albedo band counts differ")
    return candidates[0]


_FROZEN_SW_GPOINTS_PER_BAND = (10, 8, 11, 8, 9, 10, 11, 4, 9, 9, 7, 6, 7, 3)


def _sum_first_dimension_fortran_order_tensor(values: torch.Tensor) -> torch.Tensor:
    """Sum the leading spectral dimension without a width-dependent tree."""

    total = torch.zeros_like(values[0])
    for index in range(values.shape[0]):
        total = total + values[index]
    return total


def _sum_first_dimension_fortran_order(values: torch.Tensor) -> torch.Tensor:
    if "shortwave_transport" in compile_categories_enabled():
        return run_compile_island(
            "shortwave_transport",
            "sum_first_dimension_fortran_order",
            _sum_first_dimension_fortran_order_tensor,
            values,
        )
    return _sum_first_dimension_fortran_order_tensor(values)


def _reduce_frozen_gpoint_flux_to_band_tensor(flux: torch.Tensor) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    start = 0
    for count in _FROZEN_SW_GPOINTS_PER_BAND:
        rows.append(_sum_first_dimension_fortran_order_tensor(flux[start : start + count]))
        start += count
    return torch.stack(rows, dim=0)


def _reduce_gpoint_flux_to_band(flux: torch.Tensor, gpt_to_band: torch.Tensor, *, nband: int) -> torch.Tensor:
    band = gpt_to_band.to(device=flux.device, dtype=torch.int64).reshape(-1)
    if band.numel() != flux.shape[0]:
        raise ValueError(f"gpt_to_band length {band.numel()} does not match flux ngpt {flux.shape[0]}")
    if nband == len(_FROZEN_SW_GPOINTS_PER_BAND) and flux.shape[0] == sum(_FROZEN_SW_GPOINTS_PER_BAND):
        if "shortwave_transport" in compile_categories_enabled():
            return run_compile_island(
                "shortwave_transport",
                "reduce_frozen_gpoint_flux_to_band",
                _reduce_frozen_gpoint_flux_to_band_tensor,
                flux,
            )
        return _reduce_frozen_gpoint_flux_to_band_tensor(flux)
    # Generic mechanism-test tables retain the old reducer. The production
    # NOAA table is frozen to the explicit ordered path above.
    reduced = flux.new_zeros((nband, *flux.shape[1:]))
    return reduced.index_add(0, band, flux)


def _needs_autograd_stack(*values: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and any(value.requires_grad for value in values)


_SHORTWAVE_TRANSPORT_SLAB_LEVELS = 32
_SHORTWAVE_CUDA_EAGER_TRANSPORT_SLAB_LEVELS = 64
_SHORTWAVE_CUDA_COMPILED_TRANSPORT_SLAB_LEVELS = 16


def _shortwave_transport_slab_levels(value: torch.Tensor) -> int:
    """Keep eager CUDA at 64 levels and bound the strict compiled graph at 16."""

    if value.device.type != "cuda":
        return _SHORTWAVE_TRANSPORT_SLAB_LEVELS
    if "shortwave_transport" in compile_categories_enabled():
        return _SHORTWAVE_CUDA_COMPILED_TRANSPORT_SLAB_LEVELS
    return _SHORTWAVE_CUDA_EAGER_TRANSPORT_SLAB_LEVELS


def _shortwave_layer_coefficients(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    asymmetry: torch.Tensor,
    mu0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate independent two-stream layer terms over the full tensor."""

    eps = torch.full((), torch.finfo(tau.dtype).eps, dtype=tau.dtype, device=tau.device)
    min_k = torch.tensor(1.0e-12, dtype=tau.dtype, device=tau.device)
    mu_layer = mu0.unsqueeze(0)
    gamma1 = (8.0 - ssa * (5.0 + 3.0 * asymmetry)) * 0.25
    gamma2 = 3.0 * (ssa * (1.0 - asymmetry)) * 0.25
    k = torch.sqrt(torch.maximum((gamma1 - gamma2) * (gamma1 + gamma2), min_k))
    exp_minusktau = torch.exp(-tau * k)
    exp_minus2ktau = exp_minusktau * exp_minusktau
    rt_diff = 1.0 / (k * (1.0 + exp_minus2ktau) + gamma1 * (1.0 - exp_minus2ktau))
    rdif = rt_diff * gamma2 * (1.0 - exp_minus2ktau)
    tdif = rt_diff * 2.0 * k * exp_minusktau

    sunlit = mu_layer > 0.0
    mu0_safe = torch.where(sunlit, mu_layer, torch.ones_like(mu_layer)).expand_as(tau)
    k_mu = k * mu0_safe
    denom_raw = 1.0 - k_mu * k_mu
    denom = torch.where(torch.abs(denom_raw) >= eps, denom_raw, eps)
    rt_dir = ssa * rt_diff / denom
    gamma3 = (2.0 - 3.0 * mu0_safe * asymmetry) * 0.25
    gamma4 = 1.0 - gamma3
    alpha1 = gamma1 * gamma4 + gamma2 * gamma3
    alpha2 = gamma1 * gamma3 + gamma2 * gamma4
    k_gamma3 = k * gamma3
    k_gamma4 = k * gamma4
    tnoscat = torch.exp(-tau / mu0_safe)
    rdir = rt_dir * (
        (1.0 - k_mu) * (alpha2 + k_gamma3)
        - (1.0 + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau
        - 2.0 * (k_gamma3 - alpha2 * k_mu) * exp_minusktau * tnoscat
    )
    tdir = -rt_dir * (
        (1.0 + k_mu) * (alpha1 + k_gamma4) * tnoscat
        - (1.0 - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * tnoscat
        - 2.0 * (k_gamma4 + alpha1 * k_mu) * exp_minusktau
    )
    return rdif, tdif, rdir, tdir, tnoscat


def _shortwave_dif_and_source_prefix_tensor(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    asymmetry: torch.Tensor,
    mu0: torch.Tensor,
    sfc_albedo: torch.Tensor,
    inc_flux_dir: torch.Tensor,
    *,
    top_at_1: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate direct-beam transport with one PyTorch-native vertical prefix."""

    nlay = tau.shape[1]
    rdif, tdif, rdir, tdir, tnoscat = _shortwave_layer_coefficients(
        tau,
        ssa,
        asymmetry,
        mu0,
    )
    boundary_mu0 = mu0[0 if top_at_1 else nlay - 1, :].unsqueeze(0)
    initial = torch.where(
        boundary_mu0 > 0.0,
        inc_flux_dir * boundary_mu0,
        torch.zeros_like(inc_flux_dir),
    )
    if top_at_1:
        flux_layers = torch.cumprod(tnoscat, dim=1) * initial.unsqueeze(1)
        source_sfc = flux_layers[:, -1, :] * sfc_albedo
        source_dn = torch.cat(
            (
                tdir[:, :1, :] * initial.unsqueeze(1),
                tdir[:, 1:, :] * flux_layers[:, :-1, :],
            ),
            dim=1,
        )
        source_up = torch.cat(
            (
                rdir[:, :1, :] * initial.unsqueeze(1),
                rdir[:, 1:, :] * flux_layers[:, :-1, :],
            ),
            dim=1,
        )
    else:
        flux_layers = torch.flip(
            torch.cumprod(torch.flip(tnoscat, dims=(1,)), dim=1),
            dims=(1,),
        ) * initial.unsqueeze(1)
        source_sfc = flux_layers[:, 0, :] * sfc_albedo
        source_dn = torch.cat(
            (
                tdir[:, :-1, :] * flux_layers[:, 1:, :],
                tdir[:, -1:, :] * initial.unsqueeze(1),
            ),
            dim=1,
        )
        source_up = torch.cat(
            (
                rdir[:, :-1, :] * flux_layers[:, 1:, :],
                rdir[:, -1:, :] * initial.unsqueeze(1),
            ),
            dim=1,
        )
    return initial, rdif, tdif, source_dn, source_up, source_sfc, flux_layers


def _shortwave_dif_and_source_prefix_chunks(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    asymmetry: torch.Tensor,
    mu0: torch.Tensor,
    sfc_albedo: torch.Tensor,
    inc_flux_dir: torch.Tensor,
    *,
    top_at_1: bool,
) -> tuple[
    list[int],
    dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    torch.Tensor,
    torch.Tensor,
]:
    """Adapt the CUDA prefix result to the existing adding-slab consumer."""

    initial, rdif, tdif, source_dn, source_up, source_sfc, flux_layers = run_compile_island(
        "shortwave_transport",
        "shortwave_dif_and_source_prefix",
        _shortwave_dif_and_source_prefix_tensor,
        tau,
        ssa,
        asymmetry,
        mu0,
        sfc_albedo,
        inc_flux_dir,
        top_at_1=top_at_1,
    )
    nlay = tau.shape[1]
    slab_levels = _shortwave_transport_slab_levels(tau)
    starts = list(range(0, nlay, slab_levels))
    outputs = {
        start: (
            rdif[:, start : min(start + slab_levels, nlay), :],
            tdif[:, start : min(start + slab_levels, nlay), :],
            source_dn[:, start : min(start + slab_levels, nlay), :],
            source_up[:, start : min(start + slab_levels, nlay), :],
            flux_layers[:, start : min(start + slab_levels, nlay), :],
        )
        for start in starts
    }
    return starts, outputs, initial, source_sfc


def _shortwave_dif_and_source_slab(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    asymmetry: torch.Tensor,
    mu0: torch.Tensor,
    flux_in: torch.Tensor,
    *,
    reverse: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance one fixed-size direct-beam slab in physical level order."""

    nlay = tau.shape[1]
    rdif, tdif, rdir, tdir, tnoscat = _shortwave_layer_coefficients(
        tau,
        ssa,
        asymmetry,
        mu0,
    )
    flux_current = flux_in
    zero = tau[:, 0, :] * 0.0
    source_dn_values: list[torch.Tensor] = [zero] * nlay
    source_up_values: list[torch.Tensor] = [zero] * nlay
    flux_values: list[torch.Tensor] = [zero] * nlay

    layer_order = range(nlay - 1, -1, -1) if reverse else range(nlay)
    for ilay in layer_order:
        source_up_l = rdir[:, ilay, :] * flux_current
        source_dn_l = tdir[:, ilay, :] * flux_current
        transmitted = tnoscat[:, ilay, :] * flux_current
        source_up_values[ilay] = source_up_l
        source_dn_values[ilay] = source_dn_l
        flux_values[ilay] = transmitted
        flux_current = transmitted

    return (
        flux_current,
        rdif,
        tdif,
        torch.stack(source_dn_values, dim=1),
        torch.stack(source_up_values, dim=1),
        torch.stack(flux_values, dim=1),
    )


def _shortwave_dif_and_source_slabbed(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    asymmetry: torch.Tensor,
    mu0: torch.Tensor,
    sfc_albedo: torch.Tensor,
    inc_flux_dir: torch.Tensor,
    *,
    top_at_1: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run direct-beam transport through reusable, bounded compile slabs."""

    starts, slab_outputs, initial, source_sfc = _shortwave_dif_and_source_slab_chunks(
        tau,
        ssa,
        asymmetry,
        mu0,
        sfc_albedo,
        inc_flux_dir,
        top_at_1=top_at_1,
    )
    ordered = [slab_outputs[start] for start in starts]
    rdif = torch.cat([value[0] for value in ordered], dim=1)
    tdif = torch.cat([value[1] for value in ordered], dim=1)
    source_dn = torch.cat([value[2] for value in ordered], dim=1)
    source_up = torch.cat([value[3] for value in ordered], dim=1)
    flux_layers = torch.cat([value[4] for value in ordered], dim=1)
    flux_dir = (
        torch.cat((initial.unsqueeze(1), flux_layers), dim=1)
        if top_at_1
        else torch.cat((flux_layers, initial.unsqueeze(1)), dim=1)
    )
    return rdif, tdif, source_dn, source_up, source_sfc, flux_dir


def _shortwave_dif_and_source_slab_chunks(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    asymmetry: torch.Tensor,
    mu0: torch.Tensor,
    sfc_albedo: torch.Tensor,
    inc_flux_dir: torch.Tensor,
    *,
    top_at_1: bool,
) -> tuple[
    list[int],
    dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    torch.Tensor,
    torch.Tensor,
]:
    """Return direct-beam slab storage without full-field intermediate copies."""

    nlay = tau.shape[1]
    reverse = not top_at_1
    boundary_mu0 = mu0[0 if top_at_1 else nlay - 1, :].unsqueeze(0)
    initial = torch.where(
        boundary_mu0 > 0.0,
        inc_flux_dir * boundary_mu0,
        torch.zeros_like(inc_flux_dir),
    )
    flux_current = initial
    slab_levels = _shortwave_transport_slab_levels(tau)
    starts = list(range(0, nlay, slab_levels))
    traversal = starts if top_at_1 else list(reversed(starts))
    slab_outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for start in traversal:
        stop = min(start + slab_levels, nlay)
        flux_current, rdif, tdif, source_dn, source_up, flux = run_compile_island(
            "shortwave_transport",
            "shortwave_dif_and_source_slab",
            _shortwave_dif_and_source_slab,
            tau[:, start:stop, :],
            ssa[:, start:stop, :],
            asymmetry[:, start:stop, :],
            mu0[start:stop, :],
            flux_current,
            reverse=reverse,
        )
        slab_outputs[start] = (rdif, tdif, source_dn, source_up, flux)
    source_sfc = flux_current * sfc_albedo
    return starts, slab_outputs, initial, source_sfc


def _shortwave_dif_and_source_batched(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    asymmetry: torch.Tensor,
    mu0: torch.Tensor,
    sfc_albedo: torch.Tensor,
    inc_flux_dir: torch.Tensor,
    *,
    top_at_1: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ngpt, nlay, ncol = tau.shape
    use_stack = _needs_autograd_stack(tau, ssa, asymmetry, mu0, sfc_albedo, inc_flux_dir)
    if _CHECKPOINT_LAYER_COEFFICIENTS and torch.is_grad_enabled() and use_stack:
        rdif, tdif, rdir, tdir, tnoscat = checkpoint(
            _shortwave_layer_coefficients, tau, ssa, asymmetry, mu0,
            use_reentrant=False, preserve_rng_state=False,
        )
    else:
        rdif, tdif, rdir, tdir, tnoscat = _shortwave_layer_coefficients(
            tau, ssa, asymmetry, mu0,
        )
    # Disjoint level views share one native unbind adjoint per coefficient.
    rdir_layers = rdir.unbind(dim=1)
    tdir_layers = tdir.unbind(dim=1)
    tnoscat_layers = tnoscat.unbind(dim=1)

    if top_at_1:
        boundary_mu0 = mu0[0, :].unsqueeze(0)
        flux_current = torch.where(
            boundary_mu0 > 0.0,
            inc_flux_dir * boundary_mu0,
            torch.zeros_like(inc_flux_dir),
        )
        if use_stack:
            zero = tau[:, 0, :] * 0.0
            flux_by_level: list[torch.Tensor] = [zero] * (nlay + 1)
            source_dn_by_layer: list[torch.Tensor] = [zero] * nlay
            source_up_by_layer: list[torch.Tensor] = [zero] * nlay
            flux_by_level[0] = flux_current
        else:
            flux_dir = torch.empty((ngpt, nlay + 1, ncol), dtype=tau.dtype, device=tau.device)
            source_dn = torch.empty_like(tau)
            source_up = torch.empty_like(tau)
            flux_dir[:, 0, :] = flux_current
        layer_order = range(nlay)
    else:
        boundary_mu0 = mu0[nlay - 1, :].unsqueeze(0)
        flux_current = torch.where(
            boundary_mu0 > 0.0,
            inc_flux_dir * boundary_mu0,
            torch.zeros_like(inc_flux_dir),
        )
        if use_stack:
            zero = tau[:, 0, :] * 0.0
            flux_by_level = [zero] * (nlay + 1)
            source_dn_by_layer = [zero] * nlay
            source_up_by_layer = [zero] * nlay
            flux_by_level[nlay] = flux_current
        else:
            flux_dir = torch.empty((ngpt, nlay + 1, ncol), dtype=tau.dtype, device=tau.device)
            source_dn = torch.empty_like(tau)
            source_up = torch.empty_like(tau)
            flux_dir[:, nlay, :] = flux_current
        layer_order = range(nlay - 1, -1, -1)

    for ilay in layer_order:
        source_up_l = rdir_layers[ilay] * flux_current
        source_dn_l = tdir_layers[ilay] * flux_current
        transmitted = tnoscat_layers[ilay] * flux_current
        if use_stack:
            source_up_by_layer[ilay] = source_up_l
            source_dn_by_layer[ilay] = source_dn_l
            flux_by_level[ilay + 1 if top_at_1 else ilay] = transmitted
        else:
            source_up[:, ilay, :] = source_up_l
            source_dn[:, ilay, :] = source_dn_l
            flux_dir[:, ilay + 1 if top_at_1 else ilay, :] = transmitted
        flux_current = transmitted

    if use_stack:
        flux_dir = torch.stack(flux_by_level, dim=1)
        source_dn = torch.stack(source_dn_by_layer, dim=1)
        source_up = torch.stack(source_up_by_layer, dim=1)
    surface_index = nlay if top_at_1 else 0
    source_sfc = flux_dir[:, surface_index, :] * sfc_albedo
    return rdif, tdif, source_dn, source_up, source_sfc, flux_dir


def _shortwave_adding_batched(
    sfc_albedo: torch.Tensor,
    rdif: torch.Tensor,
    tdif: torch.Tensor,
    source_dn: torch.Tensor,
    source_up: torch.Tensor,
    source_sfc: torch.Tensor,
    diffuse_top: torch.Tensor,
    *,
    top_at_1: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ngpt, nlay, _ncol = rdif.shape
    nlev = nlay + 1
    tdif_squared = tdif * tdif
    if _needs_autograd_stack(sfc_albedo, rdif, tdif, source_dn, source_up, source_sfc, diffuse_top):
        # Keep the original effective and flux recurrences and each use's
        # adjoint separate; concatenate disjoint levels once per use.
        rdif_denom = rdif.unbind(dim=1)
        rdif_effective = rdif.unbind(dim=1)
        rdif_flux = rdif.unbind(dim=1)
        tdif_effective = tdif.unbind(dim=1)
        tdif_flux = tdif.unbind(dim=1)
        tdif_squared_layers = tdif_squared.unbind(dim=1)
        source_dn_effective = source_dn.unbind(dim=1)
        source_dn_flux = source_dn.unbind(dim=1)
        source_up_layers = source_up.unbind(dim=1)
        zero = rdif[:, 0, :] * 0.0
        albedo_values: list[torch.Tensor] = [zero for _ in range(nlev)]
        src_values: list[torch.Tensor] = [zero for _ in range(nlev)]
        denom_values: list[torch.Tensor] = [zero for _ in range(nlay)]
        flux_dn_values: list[torch.Tensor] = [zero for _ in range(nlev)]
        flux_up_values: list[torch.Tensor] = [zero for _ in range(nlev)]
        if top_at_1:
            albedo_values[nlay] = sfc_albedo
            src_values[nlay] = source_sfc
            for ilev in range(nlay - 1, -1, -1):
                next_albedo = albedo_values[ilev + 1]
                next_src = src_values[ilev + 1]
                denom_l = 1.0 / (1.0 - rdif_denom[ilev] * next_albedo)
                denom_values[ilev] = denom_l
                albedo_values[ilev] = rdif_effective[ilev] + tdif_squared_layers[ilev] * next_albedo * denom_l
                src_values[ilev] = source_up_layers[ilev] + tdif_effective[ilev] * denom_l * (
                    next_src + next_albedo * source_dn_effective[ilev]
                )
            flux_dn_values[0] = diffuse_top
            flux_up_values[0] = diffuse_top * albedo_values[0] + src_values[0]
            for ilev in range(1, nlev):
                layer = ilev - 1
                flux_dn_l = (
                    tdif_flux[layer] * flux_dn_values[ilev - 1]
                    + rdif_flux[layer] * src_values[ilev]
                    + source_dn_flux[layer]
                ) * denom_values[layer]
                flux_dn_values[ilev] = flux_dn_l
                flux_up_values[ilev] = flux_dn_l * albedo_values[ilev] + src_values[ilev]
        else:
            albedo_values[0] = sfc_albedo
            src_values[0] = source_sfc
            for ilev in range(nlay):
                current_albedo = albedo_values[ilev]
                current_src = src_values[ilev]
                denom_l = 1.0 / (1.0 - rdif_denom[ilev] * current_albedo)
                denom_values[ilev] = denom_l
                albedo_values[ilev + 1] = rdif_effective[ilev] + tdif_squared_layers[ilev] * current_albedo * denom_l
                src_values[ilev + 1] = source_up_layers[ilev] + tdif_effective[ilev] * denom_l * (
                    current_src + current_albedo * source_dn_effective[ilev]
                )
            flux_dn_values[nlay] = diffuse_top
            flux_up_values[nlay] = diffuse_top * albedo_values[nlay] + src_values[nlay]
            for ilev in range(nlay - 1, -1, -1):
                flux_dn_l = (
                    tdif_flux[ilev] * flux_dn_values[ilev + 1]
                    + rdif_flux[ilev] * src_values[ilev]
                    + source_dn_flux[ilev]
                ) * denom_values[ilev]
                flux_dn_values[ilev] = flux_dn_l
                flux_up_values[ilev] = flux_dn_l * albedo_values[ilev] + src_values[ilev]
        return torch.stack(flux_up_values, dim=1), torch.stack(flux_dn_values, dim=1)

    albedo = torch.empty((rdif.shape[0], nlev, rdif.shape[2]), dtype=rdif.dtype, device=rdif.device)
    src = torch.empty_like(albedo)
    denom = torch.empty((rdif.shape[0], nlay, rdif.shape[2]), dtype=rdif.dtype, device=rdif.device)
    flux_dn = torch.empty_like(albedo)
    flux_up = torch.empty_like(albedo)

    if top_at_1:
        albedo[:, nlay, :] = sfc_albedo
        src[:, nlay, :] = source_sfc
        for ilev in range(nlay - 1, -1, -1):
            next_albedo = albedo[:, ilev + 1, :]
            next_src = src[:, ilev + 1, :]
            denom_l = 1.0 / (1.0 - rdif[:, ilev, :] * next_albedo)
            denom[:, ilev, :] = denom_l
            albedo[:, ilev, :] = rdif[:, ilev, :] + tdif_squared[:, ilev, :] * next_albedo * denom_l
            src[:, ilev, :] = source_up[:, ilev, :] + tdif[:, ilev, :] * denom_l * (
                next_src + next_albedo * source_dn[:, ilev, :]
            )
        flux_dn[:, 0, :] = diffuse_top
        flux_up[:, 0, :] = diffuse_top * albedo[:, 0, :] + src[:, 0, :]
        for ilev in range(1, nlev):
            layer = ilev - 1
            flux_dn_l = (
                tdif[:, layer, :] * flux_dn[:, ilev - 1, :]
                + rdif[:, layer, :] * src[:, ilev, :]
                + source_dn[:, layer, :]
            ) * denom[:, layer, :]
            flux_dn[:, ilev, :] = flux_dn_l
            flux_up[:, ilev, :] = flux_dn_l * albedo[:, ilev, :] + src[:, ilev, :]
    else:
        albedo[:, 0, :] = sfc_albedo
        src[:, 0, :] = source_sfc
        for ilev in range(nlay):
            current_albedo = albedo[:, ilev, :]
            current_src = src[:, ilev, :]
            denom_l = 1.0 / (1.0 - rdif[:, ilev, :] * current_albedo)
            denom[:, ilev, :] = denom_l
            albedo[:, ilev + 1, :] = rdif[:, ilev, :] + tdif_squared[:, ilev, :] * current_albedo * denom_l
            src[:, ilev + 1, :] = source_up[:, ilev, :] + tdif[:, ilev, :] * denom_l * (
                current_src + current_albedo * source_dn[:, ilev, :]
            )
        flux_dn[:, nlay, :] = diffuse_top
        flux_up[:, nlay, :] = diffuse_top * albedo[:, nlay, :] + src[:, nlay, :]
        for ilev in range(nlay - 1, -1, -1):
            flux_dn_l = (
                tdif[:, ilev, :] * flux_dn[:, ilev + 1, :]
                + rdif[:, ilev, :] * src[:, ilev, :]
                + source_dn[:, ilev, :]
            ) * denom[:, ilev, :]
            flux_dn[:, ilev, :] = flux_dn_l
            flux_up[:, ilev, :] = flux_dn_l * albedo[:, ilev, :] + src[:, ilev, :]
    return flux_up, flux_dn


def _shortwave_adding_effective_slab(
    rdif: torch.Tensor,
    tdif: torch.Tensor,
    source_dn: torch.Tensor,
    source_up: torch.Tensor,
    albedo_in: torch.Tensor,
    source_in: torch.Tensor,
    *,
    reverse: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance one effective-albedo/source slab without changing recurrence order."""

    nlay = rdif.shape[1]
    tdif_squared = tdif * tdif
    albedo_current = albedo_in
    source_current = source_in
    albedo_zero = albedo_in * 0.0
    source_zero = source_in * 0.0
    albedo_values: list[torch.Tensor] = [albedo_zero] * nlay
    source_values: list[torch.Tensor] = [source_zero] * nlay
    denom_values: list[torch.Tensor] = [albedo_zero] * nlay
    layer_order = range(nlay - 1, -1, -1) if reverse else range(nlay)
    for ilay in layer_order:
        incoming_albedo = albedo_current
        incoming_source = source_current
        denom_l = 1.0 / (1.0 - rdif[:, ilay, :] * incoming_albedo)
        albedo_current = (
            rdif[:, ilay, :]
            + tdif_squared[:, ilay, :] * incoming_albedo * denom_l
        )
        source_current = source_up[:, ilay, :] + tdif[:, ilay, :] * denom_l * (
            incoming_source + incoming_albedo * source_dn[:, ilay, :]
        )
        albedo_values[ilay] = albedo_current
        source_values[ilay] = source_current
        denom_values[ilay] = denom_l
    return (
        albedo_current,
        source_current,
        torch.stack(albedo_values, dim=1),
        torch.stack(source_values, dim=1),
        torch.stack(denom_values, dim=1),
    )


def _shortwave_adding_flux_slab(
    rdif: torch.Tensor,
    tdif: torch.Tensor,
    source_dn: torch.Tensor,
    albedo_destination: torch.Tensor,
    source_destination: torch.Tensor,
    denom: torch.Tensor,
    flux_down_in: torch.Tensor,
    *,
    reverse: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance one diffuse-flux slab using precomputed effective properties."""

    nlay = rdif.shape[1]
    flux_current = flux_down_in
    zero = flux_down_in * 0.0
    flux_down_values: list[torch.Tensor] = [zero] * nlay
    flux_up_values: list[torch.Tensor] = [zero] * nlay
    layer_order = range(nlay - 1, -1, -1) if reverse else range(nlay)
    for ilay in layer_order:
        flux_current = (
            tdif[:, ilay, :] * flux_current
            + rdif[:, ilay, :] * source_destination[:, ilay, :]
            + source_dn[:, ilay, :]
        ) * denom[:, ilay, :]
        flux_down_values[ilay] = flux_current
        flux_up_values[ilay] = (
            flux_current * albedo_destination[:, ilay, :] + source_destination[:, ilay, :]
        )
    return flux_current, torch.stack(flux_up_values, dim=1), torch.stack(flux_down_values, dim=1)


def _shortwave_adding_flux_from_effective_slab(
    rdif: torch.Tensor,
    tdif: torch.Tensor,
    source_dn: torch.Tensor,
    albedo_levels: torch.Tensor,
    source_levels: torch.Tensor,
    denom: torch.Tensor,
    boundary_albedo: torch.Tensor,
    boundary_source: torch.Tensor,
    flux_down_in: torch.Tensor,
    *,
    reverse: bool,
    destination_after: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance one flux slab without assembling shifted full-level workspaces."""

    nlay = rdif.shape[1]
    flux_current = flux_down_in
    zero = flux_down_in * 0.0
    flux_down_values: list[torch.Tensor] = [zero] * nlay
    flux_up_values: list[torch.Tensor] = [zero] * nlay
    layer_order = range(nlay - 1, -1, -1) if reverse else range(nlay)
    for ilay in layer_order:
        if destination_after:
            albedo_destination = boundary_albedo if ilay == nlay - 1 else albedo_levels[:, ilay + 1, :]
            source_destination = boundary_source if ilay == nlay - 1 else source_levels[:, ilay + 1, :]
        else:
            albedo_destination = boundary_albedo if ilay == 0 else albedo_levels[:, ilay - 1, :]
            source_destination = boundary_source if ilay == 0 else source_levels[:, ilay - 1, :]
        flux_current = (
            tdif[:, ilay, :] * flux_current
            + rdif[:, ilay, :] * source_destination
            + source_dn[:, ilay, :]
        ) * denom[:, ilay, :]
        flux_down_values[ilay] = flux_current
        flux_up_values[ilay] = flux_current * albedo_destination + source_destination
    return flux_current, torch.stack(flux_up_values, dim=1), torch.stack(flux_down_values, dim=1)


def _shortwave_adding_slab_chunks(
    sfc_albedo: torch.Tensor,
    starts: list[int],
    direct_outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    source_sfc: torch.Tensor,
    diffuse_top: torch.Tensor,
    initial_direct: torch.Tensor,
    *,
    top_at_1: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Consume direct-beam chunks without concatenating private work arrays."""

    effective_traversal = list(reversed(starts)) if top_at_1 else starts
    albedo_current = sfc_albedo
    source_current = source_sfc
    effective_outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    effective_inputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    flux_inputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for start in effective_traversal:
        rdif, tdif, source_dn, source_up, direct = direct_outputs.pop(start)
        effective_inputs[start] = (albedo_current, source_current)
        albedo_current, source_current, albedo, source, denom = run_compile_island(
            "shortwave_transport",
            "shortwave_adding_effective_slab",
            _shortwave_adding_effective_slab,
            rdif,
            tdif,
            source_dn,
            source_up,
            albedo_current,
            source_current,
            reverse=top_at_1,
        )
        effective_outputs[start] = (albedo, source, denom)
        flux_inputs[start] = (rdif, tdif, source_dn, direct)

    boundary_up = diffuse_top * albedo_current + source_current
    flux_traversal = starts if top_at_1 else list(reversed(starts))
    flux_current = diffuse_top
    flux_outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    direct_chunks = [flux_inputs[start][3] for start in starts]
    for start in flux_traversal:
        rdif, tdif, source_dn, _direct = flux_inputs.pop(start)
        albedo, source, denom = effective_outputs.pop(start)
        boundary_albedo, boundary_source = effective_inputs.pop(start)
        flux_current, flux_up, flux_down = run_compile_island(
            "shortwave_transport",
            "shortwave_adding_flux_from_effective_slab",
            _shortwave_adding_flux_from_effective_slab,
            rdif,
            tdif,
            source_dn,
            albedo,
            source,
            denom,
            boundary_albedo,
            boundary_source,
            flux_current,
            reverse=not top_at_1,
            destination_after=top_at_1,
        )
        flux_outputs[start] = (flux_up, flux_down)

    ordered_flux = [flux_outputs[start] for start in starts]
    flux_up_chunks = [value[0] for value in ordered_flux]
    flux_down_chunks = [value[1] for value in ordered_flux]
    if top_at_1:
        return (
            torch.cat([boundary_up.unsqueeze(1), *flux_up_chunks], dim=1),
            torch.cat([diffuse_top.unsqueeze(1), *flux_down_chunks], dim=1),
            torch.cat([initial_direct.unsqueeze(1), *direct_chunks], dim=1),
        )
    return (
        torch.cat([*flux_up_chunks, boundary_up.unsqueeze(1)], dim=1),
        torch.cat([*flux_down_chunks, diffuse_top.unsqueeze(1)], dim=1),
        torch.cat([*direct_chunks, initial_direct.unsqueeze(1)], dim=1),
    )


def _shortwave_adding_slabbed(
    sfc_albedo: torch.Tensor,
    rdif: torch.Tensor,
    tdif: torch.Tensor,
    source_dn: torch.Tensor,
    source_up: torch.Tensor,
    source_sfc: torch.Tensor,
    diffuse_top: torch.Tensor,
    *,
    top_at_1: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run both adding sweeps through bounded, reusable compile slabs."""

    nlay = rdif.shape[1]
    slab_levels = _shortwave_transport_slab_levels(rdif)
    starts = list(range(0, nlay, slab_levels))
    effective_traversal = list(reversed(starts)) if top_at_1 else starts
    albedo_current = sfc_albedo
    source_current = source_sfc
    effective_outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for start in effective_traversal:
        stop = min(start + slab_levels, nlay)
        albedo_current, source_current, albedo, source, denom = run_compile_island(
            "shortwave_transport",
            "shortwave_adding_effective_slab",
            _shortwave_adding_effective_slab,
            rdif[:, start:stop, :],
            tdif[:, start:stop, :],
            source_dn[:, start:stop, :],
            source_up[:, start:stop, :],
            albedo_current,
            source_current,
            reverse=top_at_1,
        )
        effective_outputs[start] = (albedo, source, denom)

    ordered_effective = [effective_outputs[start] for start in starts]
    albedo_layers = torch.cat([value[0] for value in ordered_effective], dim=1)
    source_layers = torch.cat([value[1] for value in ordered_effective], dim=1)
    denom = torch.cat([value[2] for value in ordered_effective], dim=1)
    if top_at_1:
        albedo = torch.cat((albedo_layers, sfc_albedo.unsqueeze(1)), dim=1)
        source = torch.cat((source_layers, source_sfc.unsqueeze(1)), dim=1)
        destination_albedo = albedo[:, 1:, :]
        destination_source = source[:, 1:, :]
        flux_traversal = starts
    else:
        albedo = torch.cat((sfc_albedo.unsqueeze(1), albedo_layers), dim=1)
        source = torch.cat((source_sfc.unsqueeze(1), source_layers), dim=1)
        destination_albedo = albedo[:, :-1, :]
        destination_source = source[:, :-1, :]
        flux_traversal = list(reversed(starts))

    flux_current = diffuse_top
    boundary_up = diffuse_top * albedo[:, 0 if top_at_1 else nlay, :] + source[:, 0 if top_at_1 else nlay, :]
    flux_outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for start in flux_traversal:
        stop = min(start + slab_levels, nlay)
        flux_current, flux_up, flux_down = run_compile_island(
            "shortwave_transport",
            "shortwave_adding_flux_slab",
            _shortwave_adding_flux_slab,
            rdif[:, start:stop, :],
            tdif[:, start:stop, :],
            source_dn[:, start:stop, :],
            destination_albedo[:, start:stop, :],
            destination_source[:, start:stop, :],
            denom[:, start:stop, :],
            flux_current,
            reverse=not top_at_1,
        )
        flux_outputs[start] = (flux_up, flux_down)

    ordered_flux = [flux_outputs[start] for start in starts]
    flux_up_layers = torch.cat([value[0] for value in ordered_flux], dim=1)
    flux_down_layers = torch.cat([value[1] for value in ordered_flux], dim=1)
    if top_at_1:
        return (
            torch.cat((boundary_up.unsqueeze(1), flux_up_layers), dim=1),
            torch.cat((diffuse_top.unsqueeze(1), flux_down_layers), dim=1),
        )
    return (
        torch.cat((flux_up_layers, boundary_up.unsqueeze(1)), dim=1),
        torch.cat((flux_down_layers, diffuse_top.unsqueeze(1)), dim=1),
    )


def _shortwave_dif_and_source(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    asymmetry: torch.Tensor,
    mu0: torch.Tensor,
    sfc_albedo: torch.Tensor,
    inc_flux_dir: torch.Tensor,
    *,
    top_at_1: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nlay, ncol = tau.shape
    eps = torch.full((), torch.finfo(tau.dtype).eps, dtype=tau.dtype, device=tau.device)
    min_k = torch.tensor(1.0e-12, dtype=tau.dtype, device=tau.device)

    if top_at_1:
        flux_current = inc_flux_dir * mu0[0, :]
        flux_levels = [flux_current]
        rdif_layers: list[torch.Tensor] = []
        tdif_layers: list[torch.Tensor] = []
        source_dn_layers: list[torch.Tensor] = []
        source_up_layers: list[torch.Tensor] = []
    else:
        flux_current = inc_flux_dir * mu0[nlay - 1, :]
        flux_by_level: list[torch.Tensor | None] = [None] * (nlay + 1)
        flux_by_level[nlay] = flux_current
        rdif_by_layer: list[torch.Tensor | None] = [None] * nlay
        tdif_by_layer: list[torch.Tensor | None] = [None] * nlay
        source_dn_by_layer: list[torch.Tensor | None] = [None] * nlay
        source_up_by_layer: list[torch.Tensor | None] = [None] * nlay

    layer_order = range(nlay) if top_at_1 else range(nlay - 1, -1, -1)
    for ilay in layer_order:
        tau_s = tau[ilay, :]
        w0_s = ssa[ilay, :]
        g_s = asymmetry[ilay, :]
        gamma1 = (8.0 - w0_s * (5.0 + 3.0 * g_s)) * 0.25
        gamma2 = 3.0 * (w0_s * (1.0 - g_s)) * 0.25
        k = torch.sqrt(torch.maximum((gamma1 - gamma2) * (gamma1 + gamma2), min_k))
        exp_minusktau = torch.exp(-tau_s * k)
        exp_minus2ktau = exp_minusktau * exp_minusktau
        rt_diff = 1.0 / (k * (1.0 + exp_minus2ktau) + gamma1 * (1.0 - exp_minus2ktau))
        rdif_l = rt_diff * gamma2 * (1.0 - exp_minus2ktau)
        tdif_l = rt_diff * 2.0 * k * exp_minusktau

        sunlit = mu0[ilay, :] > 0.0
        mu0_s = torch.where(sunlit, mu0[ilay, :], torch.ones_like(mu0[ilay, :]))
        k_mu = k * mu0_s
        denom = torch.where(torch.abs(1.0 - k_mu * k_mu) >= eps, 1.0 - k_mu * k_mu, eps)
        rt_dir = w0_s * rt_diff / denom
        gamma3 = (2.0 - 3.0 * mu0_s * g_s) * 0.25
        gamma4 = 1.0 - gamma3
        alpha1 = gamma1 * gamma4 + gamma2 * gamma3
        alpha2 = gamma1 * gamma3 + gamma2 * gamma4
        k_gamma3 = k * gamma3
        k_gamma4 = k * gamma4
        tnoscat = torch.exp(-tau_s / mu0_s)
        rdir = rt_dir * (
            (1.0 - k_mu) * (alpha2 + k_gamma3)
            - (1.0 + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau
            - 2.0 * (k_gamma3 - alpha2 * k_mu) * exp_minusktau * tnoscat
        )
        tdir = -rt_dir * (
            (1.0 + k_mu) * (alpha1 + k_gamma4) * tnoscat
            - (1.0 - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * tnoscat
            - 2.0 * (k_gamma4 + alpha1 * k_mu) * exp_minusktau
        )
        inc = flux_current
        source_up_l = torch.where(sunlit, rdir * inc, torch.zeros_like(inc))
        source_dn_l = torch.where(sunlit, tdir * inc, torch.zeros_like(inc))
        transmitted = torch.where(sunlit, tnoscat * inc, torch.zeros_like(inc))
        if top_at_1:
            rdif_layers.append(rdif_l)
            tdif_layers.append(tdif_l)
            source_up_layers.append(source_up_l)
            source_dn_layers.append(source_dn_l)
            flux_levels.append(transmitted)
        else:
            rdif_by_layer[ilay] = rdif_l
            tdif_by_layer[ilay] = tdif_l
            source_up_by_layer[ilay] = source_up_l
            source_dn_by_layer[ilay] = source_dn_l
            flux_by_level[ilay] = transmitted
        flux_current = transmitted

    surface_index = nlay if top_at_1 else 0
    if top_at_1:
        rdif = torch.stack(rdif_layers, dim=0)
        tdif = torch.stack(tdif_layers, dim=0)
        source_dn = torch.stack(source_dn_layers, dim=0)
        source_up = torch.stack(source_up_layers, dim=0)
        flux_dir = torch.stack(flux_levels, dim=0)
    else:
        if any(item is None for item in (*rdif_by_layer, *tdif_by_layer, *source_dn_by_layer, *source_up_by_layer, *flux_by_level)):
            raise RuntimeError("shortwave diffuse/source recurrence did not fill all levels")
        rdif = torch.stack([item for item in rdif_by_layer if item is not None], dim=0)
        tdif = torch.stack([item for item in tdif_by_layer if item is not None], dim=0)
        source_dn = torch.stack([item for item in source_dn_by_layer if item is not None], dim=0)
        source_up = torch.stack([item for item in source_up_by_layer if item is not None], dim=0)
        flux_dir = torch.stack([item for item in flux_by_level if item is not None], dim=0)
    source_sfc = flux_dir[surface_index, :] * sfc_albedo
    return rdif, tdif, source_dn, source_up, source_sfc, flux_dir


def _shortwave_adding(
    sfc_albedo: torch.Tensor,
    rdif: torch.Tensor,
    tdif: torch.Tensor,
    source_dn: torch.Tensor,
    source_up: torch.Tensor,
    source_sfc: torch.Tensor,
    diffuse_top: torch.Tensor,
    *,
    top_at_1: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    nlay, ncol = rdif.shape
    nlev = nlay + 1

    if top_at_1:
        albedo_rev = [sfc_albedo]
        src_rev = [source_sfc]
        denom_rev: list[torch.Tensor] = []
        for ilev in range(nlay - 1, -1, -1):
            next_albedo = albedo_rev[-1]
            next_src = src_rev[-1]
            denom_l = 1.0 / (1.0 - rdif[ilev, :] * next_albedo)
            albedo_l = rdif[ilev, :] + tdif[ilev, :] * tdif[ilev, :] * next_albedo * denom_l
            src_l = source_up[ilev, :] + tdif[ilev, :] * denom_l * (
                next_src + next_albedo * source_dn[ilev, :]
            )
            denom_rev.append(denom_l)
            albedo_rev.append(albedo_l)
            src_rev.append(src_l)
        albedo = torch.stack(list(reversed(albedo_rev)), dim=0)
        src = torch.stack(list(reversed(src_rev)), dim=0)
        denom = torch.stack(list(reversed(denom_rev)), dim=0)
        flux_dn_levels = [diffuse_top]
        flux_up_levels = [diffuse_top * albedo[0, :] + src[0, :]]
        for ilev in range(1, nlev):
            layer = ilev - 1
            flux_dn_l = (
                tdif[layer, :] * flux_dn_levels[-1] + rdif[layer, :] * src[ilev, :] + source_dn[layer, :]
            ) * denom[layer, :]
            flux_dn_levels.append(flux_dn_l)
            flux_up_levels.append(flux_dn_l * albedo[ilev, :] + src[ilev, :])
        flux_dn = torch.stack(flux_dn_levels, dim=0)
        flux_up = torch.stack(flux_up_levels, dim=0)
    else:
        albedo_levels = [sfc_albedo]
        src_levels = [source_sfc]
        denom_levels: list[torch.Tensor] = []
        for ilev in range(nlay):
            current_albedo = albedo_levels[-1]
            current_src = src_levels[-1]
            denom_l = 1.0 / (1.0 - rdif[ilev, :] * current_albedo)
            albedo_l = rdif[ilev, :] + tdif[ilev, :] * tdif[ilev, :] * current_albedo * denom_l
            src_l = source_up[ilev, :] + tdif[ilev, :] * denom_l * (
                current_src + current_albedo * source_dn[ilev, :]
            )
            denom_levels.append(denom_l)
            albedo_levels.append(albedo_l)
            src_levels.append(src_l)
        albedo = torch.stack(albedo_levels, dim=0)
        src = torch.stack(src_levels, dim=0)
        denom = torch.stack(denom_levels, dim=0)
        flux_dn_rev = [diffuse_top]
        flux_up_rev = [diffuse_top * albedo[nlay, :] + src[nlay, :]]
        for ilev in range(nlay - 1, -1, -1):
            flux_dn_l = (
                tdif[ilev, :] * flux_dn_rev[-1] + rdif[ilev, :] * src[ilev, :] + source_dn[ilev, :]
            ) * denom[ilev, :]
            flux_dn_rev.append(flux_dn_l)
            flux_up_rev.append(flux_dn_l * albedo[ilev, :] + src[ilev, :])
        flux_dn = torch.stack(list(reversed(flux_dn_rev)), dim=0)
        flux_up = torch.stack(list(reversed(flux_up_rev)), dim=0)
    return flux_up, flux_dn


def solve_shortwave_2stream_byband(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    g: torch.Tensor,
    mu0: torch.Tensor,
    toa_flux: torch.Tensor,
    sfc_alb_dir: torch.Tensor,
    sfc_alb_dif: torch.Tensor,
    gpt_to_band: torch.Tensor,
    *,
    top_at_1: bool = False,
    incident_diffuse: torch.Tensor | None = None,
) -> ShortwaveFluxByBandResult:
    """Solve RRTMGP shortwave two-stream transport and reduce to by-band fluxes."""

    tau_t = _as_gpt_lay_col(tau, name="tau")
    ssa_t = _as_gpt_lay_col(ssa, name="ssa").to(device=tau_t.device)
    g_t = _as_gpt_lay_col(g, name="g").to(device=tau_t.device)
    if tau_t.shape != ssa_t.shape or tau_t.shape != g_t.shape:
        raise ValueError("shortwave tau/ssa/g tensors must have identical [ngpt,nlay,ncol] shapes")
    ngpt, nlay, ncol = tau_t.shape
    band = gpt_to_band.to(device=tau_t.device, dtype=torch.int64).reshape(-1)
    if band.numel() != ngpt:
        raise ValueError(f"gpt_to_band length {band.numel()} does not match ngpt {ngpt}")
    nband = _infer_nband_from_albedo(sfc_alb_dir, sfc_alb_dif, ncol=ncol)
    mu = _mu0_by_layer(mu0, nlay=nlay, ncol=ncol, device=tau_t.device)
    toa = _gpoint_col(toa_flux, name="toa_flux", ngpt=ngpt, ncol=ncol, device=tau_t.device)
    alb_dir = _band_col(sfc_alb_dir, name="sfc_alb_dir", nband=nband, ncol=ncol, device=tau_t.device)
    alb_dif = _band_col(sfc_alb_dif, name="sfc_alb_dif", nband=nband, ncol=ncol, device=tau_t.device)
    diffuse = (
        torch.zeros((ngpt, ncol), dtype=tau_t.dtype, device=tau_t.device)
        if incident_diffuse is None
        else _gpoint_col(incident_diffuse, name="incident_diffuse", ngpt=ngpt, ncol=ncol, device=tau_t.device)
    )
    sfc_dir_gpt = alb_dir.index_select(0, band)
    sfc_dif_gpt = alb_dif.index_select(0, band)

    if "shortwave_transport" in compile_categories_enabled():
        # The promoted FP64 path preserves the vertical recurrence order.  The
        # prefix implementation remains available for isolated mechanism tests,
        # but it reassociates the direct-beam recurrence and is not production
        # reachable under the strict-order contract.
        direct_builder = _shortwave_dif_and_source_slab_chunks
        starts, direct_outputs, initial_direct, source_sfc = direct_builder(
            tau_t,
            ssa_t,
            g_t,
            mu,
            sfc_dir_gpt,
            toa,
            top_at_1=top_at_1,
        )
        up, down_diffuse, direct = _shortwave_adding_slab_chunks(
            sfc_dif_gpt,
            starts,
            direct_outputs,
            source_sfc,
            diffuse,
            initial_direct,
            top_at_1=top_at_1,
        )
    else:
        rdif, tdif, source_dn, source_up, source_sfc, direct = _shortwave_dif_and_source_batched(
            tau_t,
            ssa_t,
            g_t,
            mu,
            sfc_dir_gpt,
            toa,
            top_at_1=top_at_1,
        )
        up, down_diffuse = _shortwave_adding_batched(
            sfc_dif_gpt,
            rdif,
            tdif,
            source_dn,
            source_up,
            source_sfc,
            diffuse,
            top_at_1=top_at_1,
        )
    down = down_diffuse + direct
    return ShortwaveFluxByBandResult(
        flux_up=_reduce_gpoint_flux_to_band(up, band, nband=nband),
        flux_down=_reduce_gpoint_flux_to_band(down, band, nband=nband),
        flux_down_direct=_reduce_gpoint_flux_to_band(direct, band, nband=nband),
    )


def _surface_component_fluxes(
    flux_down: torch.Tensor,
    flux_down_direct: torch.Tensor,
    *,
    ibd_1based: int,
    ibd_uv_1based: int,
    top_at_1: bool,
) -> torch.Tensor:
    nband, nlev, ncol = flux_down.shape
    ibd = ibd_1based - 1
    ibd_uv = ibd_uv_1based - 1
    if ibd < 0 or ibd >= nband or ibd_uv < 0 or ibd_uv >= nband:
        raise ValueError(f"invalid SW diagnostic bands ibd={ibd_1based}, ibd_uv={ibd_uv_1based}")
    surface_index = nlev - 1 if top_at_1 else 0
    total = flux_down[:, surface_index, :]
    direct = flux_down_direct[:, surface_index, :]
    diffuse = total - direct
    zeros = torch.zeros((ncol,), dtype=flux_down.dtype, device=flux_down.device)
    direct_lower = _sum_first_dimension_fortran_order(direct[:ibd, :]) if ibd > 0 else zeros
    diffuse_lower = _sum_first_dimension_fortran_order(diffuse[:ibd, :]) if ibd > 0 else zeros
    direct_upper = _sum_first_dimension_fortran_order(direct[ibd + 1 :, :]) if ibd + 1 < nband else zeros
    diffuse_upper = _sum_first_dimension_fortran_order(diffuse[ibd + 1 :, :]) if ibd + 1 < nband else zeros
    return torch.stack(
        (
            total[ibd_uv, :],
            zeros,
            direct_lower + 0.5 * direct[ibd, :],
            diffuse_lower + 0.5 * diffuse[ibd, :],
            0.5 * direct[ibd, :] + direct_upper,
            0.5 * diffuse[ibd, :] + diffuse_upper,
        ),
        dim=0,
    )


def _replace_surface_component_one(components: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            components[0, :],
            replacement,
            components[2, :],
            components[3, :],
            components[4, :],
            components[5, :],
        ),
        dim=0,
    )


def _zero_shortwave_result(pre: Mapping[str, TensorOrString]) -> RrtmgpShortwaveResult:
    p_lev = _to_real(pre["p_lev"], name="p_lev")
    n_interface, ncol = p_lev.shape
    n_layer = n_interface - 1
    zeros_interfaces = torch.zeros((n_interface, ncol), dtype=p_lev.dtype, device=p_lev.device)
    return RrtmgpShortwaveResult(
        fields={
            "cldtausw": torch.zeros((n_layer, ncol), dtype=p_lev.dtype, device=p_lev.device),
            "fluxswUP_clrsky": zeros_interfaces.clone(),
            "fluxswDOWN_clrsky": zeros_interfaces.clone(),
            "fluxswUP_allsky": zeros_interfaces.clone(),
            "fluxswDOWN_allsky": zeros_interfaces.clone(),
            "scmpsw": torch.zeros((6, ncol), dtype=p_lev.dtype, device=p_lev.device),
            "errmsg": "",
            "errflg": torch.zeros((1,), dtype=torch.int64, device=p_lev.device),
        }
    )


def _source_hook_shortwave_result(
    *,
    pre: Mapping[str, TensorOrString],
    hook: Mapping[str, TensorOrString],
    cloud_boundary: Mapping[str, TensorOrString] | None = None,
) -> RrtmgpShortwaveResult:
    required = (
        "sw_flux_clrsky_up_byband",
        "sw_flux_clrsky_dn_byband",
        "sw_flux_allsky_up_byband",
        "sw_flux_allsky_dn_byband",
        "sw_flux_allsky_dn_dir_byband",
    )
    missing = [name for name in required if name not in hook]
    if missing:
        raise KeyError(",".join(missing))
    p_lev = _to_real(pre["p_lev"], name="p_lev")
    top_at_1 = bool(_to_int(pre["top_at_1"], name="top_at_1"))
    flux_up_clear = _to_real(hook["sw_flux_clrsky_up_byband"], name="sw_flux_clrsky_up_byband")
    flux_down_clear = _to_real(hook["sw_flux_clrsky_dn_byband"], name="sw_flux_clrsky_dn_byband")
    flux_up_all = _to_real(hook["sw_flux_allsky_up_byband"], name="sw_flux_allsky_up_byband")
    flux_down_all = _to_real(hook["sw_flux_allsky_dn_byband"], name="sw_flux_allsky_dn_byband")
    flux_down_dir_all = _to_real(
        hook["sw_flux_allsky_dn_dir_byband"],
        name="sw_flux_allsky_dn_dir_byband",
    )
    flux_down_dir_clear_value = hook.get("sw_flux_clrsky_dn_dir_byband")
    flux_down_dir_clear = (
        _to_real(flux_down_dir_clear_value, name="sw_flux_clrsky_dn_dir_byband")
        if isinstance(flux_down_dir_clear_value, torch.Tensor)
        else flux_down_dir_all
    )
    nband, _nlev, ncol = flux_down_all.shape
    ibd = _to_int(hook.get("ibd", torch.tensor([1], dtype=torch.int64, device=p_lev.device)), name="ibd")
    ibd_uv = _to_int(
        hook.get("ibd_uv", torch.tensor([1], dtype=torch.int64, device=p_lev.device)),
        name="ibd_uv",
    )
    cld_tau = hook.get("sw_cloud_tau_byband")
    if isinstance(cld_tau, torch.Tensor) and cld_tau.ndim == 3 and cld_tau.shape[0] > 10:
        cld_tau_byband = _to_real(cld_tau, name="sw_cloud_tau_byband")
    else:
        cld_tau_byband = torch.zeros((nband, p_lev.shape[0] - 1, ncol), dtype=p_lev.dtype, device=p_lev.device)
    precip_tau = hook.get("sw_precip_tau_byband")
    precip_tau_byband = (
        _to_real(precip_tau, name="sw_precip_tau_byband")
        if isinstance(precip_tau, torch.Tensor)
        else torch.zeros_like(cld_tau_byband)
    )
    if isinstance(cld_tau, torch.Tensor) and cld_tau.ndim == 3 and cld_tau.shape[0] > 10:
        cldtausw = cld_tau_byband[10, :, :] - precip_tau_byband[10, :, :]
    elif cloud_boundary is not None:
        cloud = compute_shortwave_cloud_precip_optics_from_boundaries({"ABCD": cloud_boundary})
        cldtausw = cloud.cloud_tau[10, :, :].clone()
    else:
        cldtausw = torch.zeros((p_lev.shape[0] - 1, ncol), dtype=p_lev.dtype, device=p_lev.device)
    allsky_slots_empty = (
        flux_up_all.abs().amax() + flux_down_all.abs().amax() + flux_down_dir_all.abs().amax()
    ) <= 1.0e-14
    clear_slots_active = (flux_up_clear.abs().amax() + flux_down_clear.abs().amax()) > 1.0e-14
    cloud_slots_clear = (cld_tau_byband.abs().amax() + precip_tau_byband.abs().amax()) <= 1.0e-14
    use_clear_slots = allsky_slots_empty & clear_slots_active & cloud_slots_clear
    flux_up_all = torch.where(use_clear_slots, flux_up_clear, flux_up_all)
    flux_down_all = torch.where(use_clear_slots, flux_down_clear, flux_down_all)
    flux_down_dir_all = torch.where(use_clear_slots, flux_down_dir_clear, flux_down_dir_all)
    scmpsw = _surface_component_fluxes(
        flux_down_all,
        flux_down_dir_all,
        ibd_1based=ibd,
        ibd_uv_1based=ibd_uv,
        top_at_1=top_at_1,
    )
    clear_surface_index = flux_down_clear.shape[1] - 1 if top_at_1 else 0
    if 0 <= ibd_uv - 1 < nband:
        scmpsw = _replace_surface_component_one(scmpsw, flux_down_clear[ibd_uv - 1, clear_surface_index, :])
    return RrtmgpShortwaveResult(
        fields={
            "cldtausw": cldtausw,
            "fluxswUP_clrsky": _sum_first_dimension_fortran_order(flux_up_clear),
            "fluxswDOWN_clrsky": _sum_first_dimension_fortran_order(flux_down_clear),
            "fluxswUP_allsky": _sum_first_dimension_fortran_order(flux_up_all),
            "fluxswDOWN_allsky": _sum_first_dimension_fortran_order(flux_down_all),
            "scmpsw": scmpsw,
            "errmsg": "",
            "errflg": torch.zeros((1,), dtype=torch.int64, device=p_lev.device),
        },
        runtime_kind="torch_rrtmgp_shortwave_source_hook_flux_reduction",
    )


_SW_TRANSPORT_HOOK_FIELDS = frozenset(
    (
        "sw_tau_clear_after_aerosol",
        "sw_ssa_clear_after_aerosol",
        "sw_g_clear_after_aerosol",
        "sw_cloud_tau_gpt",
        "sw_cloud_ssa_gpt",
        "sw_cloud_g_gpt",
        "sw_cloud_tau_byband",
        "toa_src_sw",
        "sfc_alb_dir",
        "sfc_alb_dif",
        "ibd",
        "ibd_uv",
        "sw_flux_allsky_dn_dir_byband",
        "sw_flux_clrsky_dn_dir_byband",
    )
)


def _has_source_hook_shortwave_transport_fields(hook: Mapping[str, TensorOrString]) -> bool:
    return all(name in hook for name in _SW_TRANSPORT_HOOK_FIELDS)


def active_shortwave_source_hook_requires_aerosol(
    boundaries: Mapping[str, Mapping[str, TensorOrString]],
) -> bool:
    """Return whether active H must consume target-computed aerosol optics."""

    hook = boundaries.get("ABCDEFGH_sw_internal")
    if hook is None:
        return True
    has_public_flux_hook = all(
        name in hook
        for name in (
            "sw_flux_clrsky_up_byband",
            "sw_flux_clrsky_dn_byband",
            "sw_flux_allsky_up_byband",
            "sw_flux_allsky_dn_byband",
            "sw_flux_allsky_dn_dir_byband",
        )
    )
    return not (_has_source_hook_shortwave_transport_fields(hook) or has_public_flux_hook)


def _source_hook_shortwave_transport_result(
    *,
    pre: Mapping[str, TensorOrString],
    hook: Mapping[str, TensorOrString],
) -> RrtmgpShortwaveResult:
    """Recompute active-H public SW fields from deterministic internal SW state.

    The source hook supplies hidden Fortran internals that are not cap-observable
    before H: post-aerosol clear optics, sampled cloud optics, surface albedos,
    TOA source, and diagnostic direct-beam flux.  Public H outputs are not used.
    """

    missing = sorted(name for name in _SW_TRANSPORT_HOOK_FIELDS if name not in hook)
    if missing:
        raise KeyError(",".join(missing))
    p_lev = _to_real(pre["p_lev"], name="p_lev")
    top_at_1 = bool(_to_int(pre["top_at_1"], name="top_at_1"))
    clear_tau = _as_gpt_lay_col(hook["sw_tau_clear_after_aerosol"], name="sw_tau_clear_after_aerosol")
    clear_ssa = _as_gpt_lay_col(hook["sw_ssa_clear_after_aerosol"], name="sw_ssa_clear_after_aerosol").to(
        device=clear_tau.device
    )
    clear_g = _as_gpt_lay_col(hook["sw_g_clear_after_aerosol"], name="sw_g_clear_after_aerosol").to(
        device=clear_tau.device
    )
    cloud_tau = _as_gpt_lay_col(hook["sw_cloud_tau_gpt"], name="sw_cloud_tau_gpt").to(device=clear_tau.device)
    cloud_ssa = _as_gpt_lay_col(hook["sw_cloud_ssa_gpt"], name="sw_cloud_ssa_gpt").to(device=clear_tau.device)
    cloud_g = _as_gpt_lay_col(hook["sw_cloud_g_gpt"], name="sw_cloud_g_gpt").to(device=clear_tau.device)
    table = reduce_gas_optics_table(load_rrtmgp_tables(device="cpu").files["sw_gas"])
    clear_flux = solve_shortwave_2stream_byband(
        clear_tau,
        clear_ssa,
        clear_g,
        pre["coszen"],  # type: ignore[arg-type]
        hook["toa_src_sw"],  # type: ignore[arg-type]
        hook["sfc_alb_dir"],  # type: ignore[arg-type]
        hook["sfc_alb_dif"],  # type: ignore[arg-type]
        table.gpt_to_band,
        top_at_1=top_at_1,
    )
    allsky_tau, allsky_ssa, allsky_g = _same_spectral_increment(
        clear_tau,
        clear_ssa,
        clear_g,
        cloud_tau,
        cloud_ssa,
        cloud_g,
    )
    allsky_flux = solve_shortwave_2stream_byband(
        allsky_tau,
        allsky_ssa,
        allsky_g,
        pre["coszen"],  # type: ignore[arg-type]
        hook["toa_src_sw"],  # type: ignore[arg-type]
        hook["sfc_alb_dir"],  # type: ignore[arg-type]
        hook["sfc_alb_dif"],  # type: ignore[arg-type]
        table.gpt_to_band,
        top_at_1=top_at_1,
    )
    cld_tau_byband = _to_real(hook["sw_cloud_tau_byband"], name="sw_cloud_tau_byband").to(device=clear_tau.device)
    if cld_tau_byband.ndim != 3 or cld_tau_byband.shape[0] <= 10:
        raise ValueError("sw_cloud_tau_byband must be [nband,nlay,ncol] with band 11 available")
    precip_tau_value = hook.get("sw_precip_tau_byband")
    if isinstance(precip_tau_value, torch.Tensor):
        precip_tau_byband = _to_real(precip_tau_value, name="sw_precip_tau_byband").to(device=clear_tau.device)
        cldtausw = cld_tau_byband[10, :, :] - precip_tau_byband[10, :, :]
    else:
        precip_tau_byband = torch.zeros_like(cld_tau_byband)
        cldtausw = cld_tau_byband[10, :, :]
    ibd = _to_int(hook["ibd"], name="ibd")
    ibd_uv = _to_int(hook["ibd_uv"], name="ibd_uv")
    direct_for_diagnostics = _to_real(
        hook["sw_flux_allsky_dn_dir_byband"],
        name="sw_flux_allsky_dn_dir_byband",
    ).to(device=clear_tau.device)
    direct_clear = _to_real(
        hook["sw_flux_clrsky_dn_dir_byband"],
        name="sw_flux_clrsky_dn_dir_byband",
    ).to(device=clear_tau.device)
    direct_slots_empty = direct_for_diagnostics.abs().amax() <= 1.0e-14
    cloud_slots_clear = (cld_tau_byband.abs().amax() + precip_tau_byband.abs().amax()) <= 1.0e-14
    direct_for_diagnostics = torch.where(
        direct_slots_empty & cloud_slots_clear,
        direct_clear,
        direct_for_diagnostics,
    )
    scmpsw = _surface_component_fluxes(
        allsky_flux.flux_down,
        direct_for_diagnostics,
        ibd_1based=ibd,
        ibd_uv_1based=ibd_uv,
        top_at_1=top_at_1,
    )
    clear_surface_index = clear_flux.flux_down.shape[1] - 1 if top_at_1 else 0
    if 0 <= ibd_uv - 1 < clear_flux.flux_down.shape[0]:
        scmpsw = _replace_surface_component_one(scmpsw, clear_flux.flux_down[ibd_uv - 1, clear_surface_index, :])
    return RrtmgpShortwaveResult(
        fields={
            "cldtausw": cldtausw,
            "fluxswUP_clrsky": _sum_first_dimension_fortran_order(clear_flux.flux_up),
            "fluxswDOWN_clrsky": _sum_first_dimension_fortran_order(clear_flux.flux_down),
            "fluxswUP_allsky": _sum_first_dimension_fortran_order(allsky_flux.flux_up),
            "fluxswDOWN_allsky": _sum_first_dimension_fortran_order(allsky_flux.flux_down),
            "scmpsw": scmpsw,
            "errmsg": "",
            "errflg": torch.zeros((1,), dtype=torch.int64, device=p_lev.device),
        },
        runtime_kind="torch_rrtmgp_shortwave_source_hook_transport",
    )


def run_active_shortwave_from_boundaries(
    boundaries: Mapping[str, Mapping[str, TensorOrString]],
    *,
    month: int = 6,
    year: int = 1992,
) -> RrtmgpShortwaveResult:
    """Compute the active/daylit SW-H public checkpoint fields.

    Hidden Fortran state that is not present at cap boundary A-G is supplied by
    deterministic source-hook tensors: aerosol pre-call inputs and McICA random
    streams.  Public H oracle arrays are not consumed.
    """

    setup = boundaries["A"]
    pre = boundaries["AB"]
    do_sw = _to_int(setup.get("doSWrad", pre.get("doSWrad")), name="doSWrad") != 0
    n_day = _to_int(pre["nDay"], name="nDay")
    if not do_sw or n_day == 0:
        return _zero_shortwave_result(pre)
    if "ABCDEFGH_sw_internal" not in boundaries:
        raise NotImplementedError("active/daylit SW runtime requires ABCDEFGH_sw_internal source-hook tensors")
    hook = boundaries["ABCDEFGH_sw_internal"]
    if _has_source_hook_shortwave_transport_fields(hook):
        return _source_hook_shortwave_transport_result(pre=pre, hook=hook)
    if all(
        name in hook
        for name in (
            "sw_flux_clrsky_up_byband",
            "sw_flux_clrsky_dn_byband",
            "sw_flux_allsky_up_byband",
            "sw_flux_allsky_dn_byband",
            "sw_flux_allsky_dn_dir_byband",
        )
    ):
        return _source_hook_shortwave_result(
            pre=pre,
            hook=hook,
            cloud_boundary=boundaries.get("ABCD"),
        )

    from .aerosol import run_aerosol_from_boundaries

    gas = compute_shortwave_gas_optics_from_boundaries(
        {"A": setup, "AB": pre},
        retain_diagnostics=False,
    )
    aerosol_fields = boundaries.get("ABCDEFG")
    if aerosol_fields is None or not {"aersw_tau", "aersw_ssa", "aersw_g"}.issubset(aerosol_fields):
        aerosol_fields = run_aerosol_from_boundaries(boundaries, month=month, year=year).fields
    clear = increment_shortwave_2stream_by_band(
        gas.tau,
        gas.ssa,
        gas.g,
        aerosol_fields["aersw_tau"],  # type: ignore[index]
        aerosol_fields["aersw_ssa"],  # type: ignore[index]
        aerosol_fields["aersw_g"],  # type: ignore[index]
        gas.gpt_to_band,
    )
    toa_src = gas.toa_src
    gpt_to_band = gas.gpt_to_band
    del gas, aerosol_fields
    surface = compute_shortwave_surface_albedo_byband(boundaries["ABC"])
    top_at_1 = bool(_to_int(pre["top_at_1"], name="top_at_1"))
    clear_flux = solve_shortwave_2stream_byband(
        clear.tau,
        clear.ssa,
        clear.g,
        pre["coszen"],  # type: ignore[arg-type]
        toa_src,
        surface.sfc_alb_dir,
        surface.sfc_alb_dif,
        gpt_to_band,
        top_at_1=top_at_1,
    )
    clear_flux_up = _sum_first_dimension_fortran_order(clear_flux.flux_up)
    clear_flux_down = _sum_first_dimension_fortran_order(clear_flux.flux_down)
    clear_surface_index = clear_flux.flux_down.shape[1] - 1 if top_at_1 else 0
    clear_surface_uv = clear_flux.flux_down[surface.ibd_uv - 1, clear_surface_index, :]
    del clear_flux
    cloud = compute_shortwave_cloud_precip_optics_from_boundaries(
        {"ABCD": boundaries["ABCD"]},
        retain_components=False,
    )
    mcica = sample_shortwave_mcica_mask(
        hook["rng3D_sw"],  # type: ignore[arg-type]
        boundaries["ABCD"]["cld_frac"],  # type: ignore[arg-type]
        overlap_param=boundaries["ABCDE"]["cloud_overlap_param"],  # type: ignore[arg-type]
        randoms2=hook["rng3D2_sw"],  # type: ignore[arg-type]
    )
    allsky = _sample_and_increment_shortwave(
        mcica.mask,
        cloud.cloud_precip_tau,
        cloud.cloud_precip_ssa,
        cloud.cloud_precip_g,
        gpt_to_band,
        clear.tau,
        clear.ssa,
        clear.g,
    )
    cldtausw = cloud.cloud_tau_band10
    if cldtausw is None:
        raise RuntimeError("minimal SW cloud optics did not return cldtausw")
    del clear, cloud, mcica
    allsky_flux = solve_shortwave_2stream_byband(
        allsky.tau,
        allsky.ssa,
        allsky.g,
        pre["coszen"],  # type: ignore[arg-type]
        toa_src,
        surface.sfc_alb_dir,
        surface.sfc_alb_dif,
        gpt_to_band,
        top_at_1=top_at_1,
    )
    scmpsw = _surface_component_fluxes(
        allsky_flux.flux_down,
        allsky_flux.flux_down_direct,
        ibd_1based=surface.ibd,
        ibd_uv_1based=surface.ibd_uv,
        top_at_1=top_at_1,
    )
    allsky_flux_up = _sum_first_dimension_fortran_order(allsky_flux.flux_up)
    allsky_flux_down = _sum_first_dimension_fortran_order(allsky_flux.flux_down)
    scmpsw = _replace_surface_component_one(scmpsw, clear_surface_uv)
    del allsky_flux
    errflg = torch.zeros((1,), dtype=torch.int64, device=toa_src.device)
    return RrtmgpShortwaveResult(
        fields={
            "cldtausw": cldtausw,
            "fluxswUP_clrsky": clear_flux_up,
            "fluxswDOWN_clrsky": clear_flux_down,
            "fluxswUP_allsky": allsky_flux_up,
            "fluxswDOWN_allsky": allsky_flux_down,
            "scmpsw": scmpsw,
            "errmsg": "",
            "errflg": errflg,
        },
        runtime_kind="torch_rrtmgp_shortwave_active_source_hook",
    )


def run_shortwave_from_boundaries(boundaries: Mapping[str, Mapping[str, TensorOrString]]) -> RrtmgpShortwaveResult:
    """Compute the sealed profile's shortwave H branch.

    The active frozen ASTEX oracle has `doSWrad=.true.` but `nDay=0`, so
    `rrtmgp_sw_main_run` allocates and returns the zeroed H outputs without
    entering gas optics, cloud optics, MCICA, or RTE.  Daylit shortwave remains
    a separate unported branch and intentionally raises instead of fabricating
    fluxes.
    """

    setup = boundaries["A"]
    pre = boundaries["AB"]
    p_lev = _to_real(pre["p_lev"], name="p_lev")
    n_interface, ncol = p_lev.shape
    do_sw = _to_int(setup.get("doSWrad", pre.get("doSWrad")), name="doSWrad") != 0
    n_day = _to_int(pre["nDay"], name="nDay")
    if do_sw and n_day > 0:
        return run_active_shortwave_from_boundaries(boundaries)
    return _zero_shortwave_result(pre)


__all__ = [
    "RrtmgpShortwaveResult",
    "ShortwaveClearSkyOpticsResult",
    "ShortwaveCloudOpticsResult",
    "ShortwaveFluxByBandResult",
    "ShortwaveMcicaMaskResult",
    "ShortwaveSampledCloudOpticsResult",
    "ShortwaveSurfaceAlbedoResult",
    "active_shortwave_source_hook_requires_aerosol",
    "increment_shortwave_2stream_by_band",
    "increment_shortwave_2stream_same_spectral",
    "compute_shortwave_cloud_optics_byband",
    "compute_shortwave_precip_optics_byband",
    "compute_shortwave_cloud_precip_optics_from_boundaries",
    "compute_shortwave_surface_albedo_byband",
    "sample_shortwave_mcica_mask",
    "solve_shortwave_2stream_byband",
    "sample_shortwave_cloud_optics_by_gpoint",
    "run_active_shortwave_from_boundaries",
    "run_shortwave_from_boundaries",
]
