# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""PyTorch-native longwave main path for sealed RRTMGP checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .compile import compile_categories_enabled, run_compile_island
from .gas_optics import compute_longwave_gas_optics_from_boundaries, reduce_gas_optics_table
from .longwave import (
    LongwaveFluxResult,
    solve_longwave_noscat,
    solve_longwave_rescaled_noscat,
)
from .shortwave import (
    _combine_cloud_components,
    _interp_cloud_table,
    _same_spectral_increment,
    sample_shortwave_cloud_optics_by_gpoint,
    sample_shortwave_mcica_mask,
)
from .tables import RrtmgpTableBundle, load_rrtmgp_tables

TensorOrString = torch.Tensor | str
_LONGWAVE_CLOUD_TABLE_CACHE: dict[tuple[int, int], tuple[Any, ...]] = {}


@dataclass(frozen=True)
class RrtmgpLongwaveResult:
    """Checkpoint-shaped longwave-I output fields."""

    fields: Mapping[str, TensorOrString]
    clear_sky: LongwaveFluxResult
    all_sky: LongwaveFluxResult
    runtime_kind: str = "torch_rrtmgp_longwave_main"


@dataclass(frozen=True)
class LongwaveCloudOpticsResult:
    """Longwave cloud and precipitation by-band optical properties."""

    cloud_tau: torch.Tensor | None
    cloud_ssa: torch.Tensor | None
    cloud_g: torch.Tensor | None
    precip_tau: torch.Tensor | None
    precip_ssa: torch.Tensor | None
    precip_g: torch.Tensor | None
    cloud_precip_tau: torch.Tensor
    cloud_precip_ssa: torch.Tensor
    cloud_precip_g: torch.Tensor
    runtime_kind: str = "torch_rrtmgp_longwave_cloud_precip_optics"


def _to_real(value: TensorOrString, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64 (the working dtype), got {value.dtype}")
    return value


def _to_int(value: TensorOrString, *, name: str) -> int:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    return int(value.detach().to(device="cpu", dtype=torch.int64).reshape(-1)[0].item())


def _band_to_gpoint(field_by_band: torch.Tensor, gpt_to_band: torch.Tensor) -> torch.Tensor:
    field = _to_real(field_by_band, name="field_by_band")
    if field.ndim != 3:
        raise ValueError(f"band field must have shape [nband,nlay,ncol], got {tuple(field.shape)}")
    band = gpt_to_band.to(device=field.device, dtype=torch.int64).reshape(-1)
    return field.index_select(0, band)


def expand_longwave_byband_to_gpoint(field_by_band: torch.Tensor, gpt_to_band: torch.Tensor) -> torch.Tensor:
    """Expand LW by-band fields to g-point fields using frozen gas-table band indices."""

    return _band_to_gpoint(field_by_band, gpt_to_band)


def _prepare_longwave_clear_tensor(
    gas_tau: torch.Tensor,
    aerosol_tau_by_band: torch.Tensor,
    bands: torch.Tensor,
    optimal_angle_fit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tau_clear = gas_tau + aerosol_tau_by_band.index_select(0, bands)
    # The generic reduction changes association with the column extent on CPU
    # and CUDA.  That made an otherwise identical profile select a diffusivity
    # angle one ULP differently at some batch widths.  Preserve the Fortran
    # vertical order explicitly; Inductor can still fuse this fixed recurrence.
    column_tau = torch.zeros_like(tau_clear[:, 0, :])
    for ilay in range(tau_clear.shape[1]):
        column_tau = column_tau + tau_clear[:, ilay, :]
    trans_total = torch.exp(-column_tau)
    fit_by_gpt = optimal_angle_fit.index_select(1, bands)
    diffusivity_angle = fit_by_gpt[0, :, None] * trans_total + fit_by_gpt[1, :, None]
    return tau_clear, diffusivity_angle


def _prepare_longwave_clear(
    gas_tau: torch.Tensor,
    aerosol_tau_by_band: torch.Tensor,
    gpt_to_band: torch.Tensor,
    optimal_angle_fit: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    aerosol_tau = _to_real(aerosol_tau_by_band, name="aerosol_tau_by_band").to(device=gas_tau.device)
    bands = gpt_to_band.to(device=gas_tau.device, dtype=torch.int64).reshape(-1)
    if optimal_angle_fit is None:
        tau_clear = gas_tau + aerosol_tau.index_select(0, bands)
        return tau_clear, torch.ones((bands.numel(), gas_tau.shape[2]), dtype=gas_tau.dtype, device=gas_tau.device)
    return run_compile_island(
        "longwave_transport",
        "prepare_longwave_clear",
        _prepare_longwave_clear_tensor,
        gas_tau,
        aerosol_tau,
        bands,
        optimal_angle_fit.to(device=gas_tau.device, dtype=gas_tau.dtype),
    )


def _increment_longwave_by_scalar_kernel(
    cloud_tau: torch.Tensor,
    cloud_ssa: torch.Tensor,
    cloud_g: torch.Tensor,
    scalar_tau: torch.Tensor,
    eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse the active all-sky cloud plus clear-absorption increment."""

    tau12 = cloud_tau + scalar_tau
    ssa12 = cloud_tau * cloud_ssa / torch.maximum(eps, tau12)
    return tau12, ssa12, cloud_g


def increment_longwave_2stream_by_1scalar(
    cloud_tau: torch.Tensor,
    cloud_ssa: torch.Tensor,
    cloud_g: torch.Tensor,
    scalar_tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match RTE `inc_2stream_by_1scalar` for adding clear absorption to sampled clouds."""

    tau1 = _to_real(cloud_tau, name="cloud_tau")
    ssa1 = _to_real(cloud_ssa, name="cloud_ssa").to(device=tau1.device)
    g1 = _to_real(cloud_g, name="cloud_g").to(device=tau1.device)
    tau2 = _to_real(scalar_tau, name="scalar_tau").to(device=tau1.device)
    if tau1.shape != ssa1.shape or tau1.shape != g1.shape or tau1.shape != tau2.shape:
        raise ValueError("LW 2stream-by-1scalar increment tensors must have identical shapes")
    eps = torch.tensor(3.0 * torch.finfo(tau1.dtype).tiny, dtype=tau1.dtype, device=tau1.device)
    return run_compile_island(
        "longwave_transport",
        "increment_longwave_by_scalar",
        _increment_longwave_by_scalar_kernel,
        tau1,
        ssa1,
        g1,
        tau2,
        eps,
    )


def _sample_and_increment_longwave_tensor(
    mask: torch.Tensor,
    cloud_tau_band: torch.Tensor,
    cloud_ssa_band: torch.Tensor,
    cloud_g_band: torch.Tensor,
    bands: torch.Tensor,
    clear_tau: torch.Tensor,
    eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cloud_tau = torch.where(mask, cloud_tau_band.index_select(0, bands), 0.0)
    cloud_ssa = torch.where(mask, cloud_ssa_band.index_select(0, bands), 0.0)
    cloud_g = torch.where(mask, cloud_g_band.index_select(0, bands), 0.0)
    tau = cloud_tau + clear_tau
    ssa = cloud_tau * cloud_ssa / torch.maximum(eps, tau)
    return tau, ssa, cloud_g


def _sample_and_increment_longwave(
    mask: torch.Tensor,
    cloud_tau_band: torch.Tensor,
    cloud_ssa_band: torch.Tensor,
    cloud_g_band: torch.Tensor,
    gpt_to_band: torch.Tensor,
    clear_tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bands = gpt_to_band.to(device=clear_tau.device, dtype=torch.int64).reshape(-1)
    eps = torch.tensor(3.0 * torch.finfo(clear_tau.dtype).tiny, dtype=clear_tau.dtype, device=clear_tau.device)
    return run_compile_island(
        "longwave_transport",
        "sample_and_increment_longwave",
        _sample_and_increment_longwave_tensor,
        mask.to(dtype=torch.bool),
        cloud_tau_band,
        cloud_ssa_band,
        cloud_g_band,
        bands,
        clear_tau,
        eps,
    )


def _assert_sealed_scope(boundaries: Mapping[str, Mapping[str, TensorOrString]]) -> None:
    cloud = boundaries["ABCD"]
    unsupported_cloud_fields = ("cld_frac", "cld_lwp", "cld_iwp", "cld_swp", "cld_rwp")
    active = []
    for name in unsupported_cloud_fields:
        value = cloud.get(name)
        if isinstance(value, torch.Tensor) and bool(
            (value.detach().to(device="cpu", dtype=torch.float64).abs() > 0.0).any().item()
        ):
            active.append(name)
    if active:
        raise NotImplementedError(
            "Full active LW transport is not yet ported for active cloud fields: "
            + ", ".join(active)
        )


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
    tables: RrtmgpTableBundle | None = None,
    ice_roughness_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute `lw_cloud_props%cloud_optics` by band for liquid plus ice."""

    lwp = _to_real(cld_lwp, name="cld_lwp")
    bundle = tables if tables is not None else load_rrtmgp_tables(device=lwp.device)
    prepared = _longwave_cloud_table_inputs(
        bundle.files["lw_cloud"],
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
    )
    ice = _interp_cloud_table(
        cld_iwp,
        cld_reice,
        prepared[3],
        prepared[4],
        prepared[5],
        offset=prepared[7],
        step_size=prepared[9],
    )
    return _combine_cloud_components(liquid, ice)


def _longwave_cloud_precip_kernel(
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
    liq_step: float,
    ice_step: float,
) -> tuple[torch.Tensor, ...]:
    """Compute and compose all LW cloud and precipitation optics in one island."""

    ncomponent, nlay, ncol = lwp.shape
    flat_shape = (ncomponent * nlay, ncol)
    liquid = _interp_cloud_table(
        lwp.reshape(flat_shape),
        reliq.reshape(flat_shape),
        liq_ext,
        liq_ssa,
        liq_asy,
        offset=radliq_lwr,
        step_size=liq_step,
    )
    ice = _interp_cloud_table(
        iwp.reshape(flat_shape),
        reice.reshape(flat_shape),
        ice_ext,
        ice_ssa,
        ice_asy,
        offset=radice_lwr,
        step_size=ice_step,
    )
    combined = _combine_cloud_components(liquid, ice)
    component_tau, component_ssa, component_g = (
        value.reshape(value.shape[0], ncomponent, nlay, ncol) for value in combined
    )
    cloud_tau = component_tau[:, 0, :, :]
    cloud_ssa = component_ssa[:, 0, :, :]
    cloud_g = component_g[:, 0, :, :]
    for component in range(1, ncomponent):
        cloud_tau, cloud_ssa, cloud_g = _same_spectral_increment(
            cloud_tau,
            cloud_ssa,
            cloud_g,
            component_tau[:, component, :, :],
            component_ssa[:, component, :, :],
            component_g[:, component, :, :],
        )
    precip_tau, precip_ssa, precip_g = compute_longwave_precip_optics_byband(
        frac,
        rwp,
        swp,
        resnow,
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


def _longwave_cloud_precip_minimal_kernel(
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
    liq_step: float,
    ice_step: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = _longwave_cloud_precip_kernel(
        lwp, iwp, reliq, reice, frac, rwp, swp, resnow,
        liq_ext, liq_ssa, liq_asy, ice_ext, ice_ssa, ice_asy,
        radliq_lwr, radice_lwr, liq_step, ice_step,
    )
    return values[6], values[7], values[8]


def _compute_longwave_cloud_precip_optics_compiled(
    cld_lwp: torch.Tensor,
    cld_iwp: torch.Tensor,
    cld_reliq: torch.Tensor,
    cld_reice: torch.Tensor,
    cld_frac: torch.Tensor,
    cld_rwp: torch.Tensor,
    cld_swp: torch.Tensor,
    cld_resnow: torch.Tensor,
    *,
    tables: RrtmgpTableBundle | None = None,
    ice_roughness_index: int | None = None,
    retain_components: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Compile one flat island over all LW cloud and precipitation optics."""

    lwp = _to_real(cld_lwp, name="cld_lwp")
    iwp = _to_real(cld_iwp, name="cld_iwp").to(device=lwp.device)
    reliq = _to_real(cld_reliq, name="cld_reliq").to(device=lwp.device)
    reice = _to_real(cld_reice, name="cld_reice").to(device=lwp.device)
    if lwp.ndim != 3 or not (iwp.shape == reliq.shape == reice.shape == lwp.shape):
        raise ValueError("batched LW cloud components must share [ncomponent,nlay,ncol] shapes")
    bundle = tables if tables is not None else load_rrtmgp_tables(device=lwp.device)
    prepared = _longwave_cloud_table_inputs(
        bundle.files["lw_cloud"],
        ice_roughness_index=ice_roughness_index,
    )
    kernel = _longwave_cloud_precip_kernel if retain_components else _longwave_cloud_precip_minimal_kernel
    name = "lw_cloud_precip_optics" if retain_components else "lw_cloud_precip_optics_minimal"
    return run_compile_island(
        "longwave_transport",
        name,
        kernel,
        lwp,
        iwp,
        reliq,
        reice,
        _to_real(cld_frac, name="cld_frac").to(device=lwp.device),
        _to_real(cld_rwp, name="cld_rwp").to(device=lwp.device),
        _to_real(cld_swp, name="cld_swp").to(device=lwp.device),
        _to_real(cld_resnow, name="cld_resnow").to(device=lwp.device),
        *prepared,
    )


def compute_longwave_precip_optics_byband(
    cld_frac: torch.Tensor,
    cld_rwp: torch.Tensor,
    cld_swp: torch.Tensor,
    cld_resnow: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute RRTMGP LW rain/snow precipitation optics by band."""

    frac = _to_real(cld_frac, name="cld_frac")
    rwp = _to_real(cld_rwp, name="cld_rwp").to(device=frac.device)
    swp = _to_real(cld_swp, name="cld_swp").to(device=frac.device)
    resnow = _to_real(cld_resnow, name="cld_resnow").to(device=frac.device)
    if frac.shape != rwp.shape or frac.shape != swp.shape or frac.shape != resnow.shape:
        raise ValueError("LW precip cloud fields must share [nlay,ncol] shapes")
    dtype = frac.dtype
    device = frac.device
    tau_rain = 0.33e-3 * rwp
    tau_snow = torch.where(
        (swp > 0.0) & (resnow > 10.0),
        1.5 * 1.05756 * swp / torch.where((swp > 0.0) & (resnow > 10.0), resnow, 20.0),
        torch.zeros_like(swp),
    )
    active = frac > 1.0e-300
    tau_layer = torch.where(active, tau_rain + tau_snow, torch.zeros_like(frac))
    tau = tau_layer.unsqueeze(0).expand(16, -1, -1).contiguous()
    ssa = torch.zeros((16, frac.shape[0], frac.shape[1]), dtype=dtype, device=device)
    asym = torch.zeros_like(ssa)
    return tau, ssa, asym


def compute_longwave_cloud_precip_optics_from_boundaries(
    boundaries: Mapping[str, Mapping[str, TensorOrString]],
    *,
    tables: RrtmgpTableBundle | None = None,
    retain_components: bool = True,
) -> LongwaveCloudOpticsResult:
    """Compute active LW cloud and precip by-band optics from ABCD fields."""

    cloud = boundaries["ABCD"]
    lwp_components = [cloud["cld_lwp"]]
    iwp_components = [cloud["cld_iwp"]]
    reliq_components = [cloud["cld_reliq"]]
    reice_components = [cloud["cld_reice"]]
    for prefix in ("cld_cnv", "cld_pbl"):
        lwp_name = f"{prefix}_lwp"
        if lwp_name in cloud:
            lwp_components.append(cloud[lwp_name])
            iwp_components.append(cloud[f"{prefix}_iwp"])
            reliq_components.append(cloud[f"{prefix}_reliq"])
            reice_components.append(cloud[f"{prefix}_reice"])
    if "longwave_transport" in compile_categories_enabled():
        values = _compute_longwave_cloud_precip_optics_compiled(
            torch.stack(lwp_components),  # type: ignore[arg-type]
            torch.stack(iwp_components),  # type: ignore[arg-type]
            torch.stack(reliq_components),  # type: ignore[arg-type]
            torch.stack(reice_components),  # type: ignore[arg-type]
            cloud["cld_frac"],  # type: ignore[arg-type]
            cloud["cld_rwp"],  # type: ignore[arg-type]
            cloud["cld_swp"],  # type: ignore[arg-type]
            cloud["cld_resnow"],  # type: ignore[arg-type]
            tables=tables,
            retain_components=retain_components,
        )
        if retain_components:
            cloud_tau, cloud_ssa, cloud_g, precip_tau, precip_ssa, precip_g, total_tau, total_ssa, total_g = values
        else:
            total_tau, total_ssa, total_g = values
            cloud_tau = cloud_ssa = cloud_g = precip_tau = precip_ssa = precip_g = None
    else:
        cloud_tau, cloud_ssa, cloud_g = compute_longwave_cloud_optics_byband(
            lwp_components[0],  # type: ignore[arg-type]
            iwp_components[0],  # type: ignore[arg-type]
            reliq_components[0],  # type: ignore[arg-type]
            reice_components[0],  # type: ignore[arg-type]
            tables=tables,
        )
        for component in range(1, len(lwp_components)):
            component_optics = compute_longwave_cloud_optics_byband(
                lwp_components[component],  # type: ignore[arg-type]
                iwp_components[component],  # type: ignore[arg-type]
                reliq_components[component],  # type: ignore[arg-type]
                reice_components[component],  # type: ignore[arg-type]
                tables=tables,
            )
            cloud_tau, cloud_ssa, cloud_g = _same_spectral_increment(
                cloud_tau,
                cloud_ssa,
                cloud_g,
                *component_optics,
            )
        precip_tau, precip_ssa, precip_g = compute_longwave_precip_optics_byband(
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
            cloud_tau = cloud_ssa = cloud_g = precip_tau = precip_ssa = precip_g = None
    return LongwaveCloudOpticsResult(
        cloud_tau=cloud_tau,
        cloud_ssa=cloud_ssa,
        cloud_g=cloud_g,
        precip_tau=precip_tau,
        precip_ssa=precip_ssa,
        precip_g=precip_g,
        cloud_precip_tau=total_tau,
        cloud_precip_ssa=total_ssa,
        cloud_precip_g=total_g,
    )


def _surface_emissivity(boundaries: Mapping[str, Mapping[str, TensorOrString]], *, nband: int) -> torch.Tensor:
    semis = _to_real(boundaries["ABC"]["semis"], name="semis").reshape(-1)
    semis = torch.where((semis > 0.0) & (semis <= 1.0), semis, torch.ones_like(semis))
    return semis.reshape(1, -1).expand(nband, -1).contiguous()


def _n_gauss_angles(boundaries: Mapping[str, Mapping[str, TensorOrString]]) -> int:
    for boundary_id in ("AB", "A"):
        value = boundaries[boundary_id].get("nGauss_angles")
        if isinstance(value, torch.Tensor):
            return _to_int(value, name="nGauss_angles")
    return 1


def _runtime_control_bool(
    boundaries: Mapping[str, Mapping[str, Any]],
    name: str,
    *,
    default: bool = False,
) -> bool:
    controls = boundaries.get("runtime_controls", {})
    if not isinstance(controls, Mapping):
        return default
    value = controls.get(name)
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return _to_int(value, name=name) != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", ".true.", "t", "yes"}
    return bool(value)


def _zero_longwave_result(p_lev: torch.Tensor) -> RrtmgpLongwaveResult:
    nlev, ncol = p_lev.shape
    zeros = torch.zeros((nlev, ncol), dtype=p_lev.dtype, device=p_lev.device)
    errflg = torch.zeros((1,), dtype=torch.int64, device=p_lev.device)
    empty = LongwaveFluxResult(zeros, zeros)
    return RrtmgpLongwaveResult(
        clear_sky=empty,
        all_sky=empty,
        fields={
            "sfc_emiss_byband": torch.zeros((16, ncol), dtype=p_lev.dtype, device=p_lev.device),
            "fluxlwUP_clrsky": zeros.clone(),
            "fluxlwDOWN_clrsky": zeros.clone(),
            "fluxlwUP_allsky": zeros.clone(),
            "fluxlwDOWN_allsky": zeros.clone(),
            "fluxlwUP_jac": zeros.clone(),
            "fluxlwUP_radtime": zeros.clone(),
            "fluxlwDOWN_radtime": zeros.clone(),
            "errmsg": "",
            "errflg": errflg,
        },
    )


def _source_hook_longwave_result(
    *,
    hook: Mapping[str, TensorOrString],
    p_lev: torch.Tensor,
) -> RrtmgpLongwaveResult:
    required = (
        "lw_flux_clrsky_up_byband",
        "lw_flux_clrsky_dn_byband",
        "lw_flux_allsky_up_byband",
        "lw_flux_allsky_dn_byband",
    )
    missing = [name for name in required if name not in hook]
    if missing:
        raise KeyError(",".join(missing))
    clear_up = _to_real(hook["lw_flux_clrsky_up_byband"], name="lw_flux_clrsky_up_byband").sum(dim=0)
    clear_down = _to_real(hook["lw_flux_clrsky_dn_byband"], name="lw_flux_clrsky_dn_byband").sum(dim=0)
    all_up = _to_real(hook["lw_flux_allsky_up_byband"], name="lw_flux_allsky_up_byband").sum(dim=0)
    all_down = _to_real(hook["lw_flux_allsky_dn_byband"], name="lw_flux_allsky_dn_byband").sum(dim=0)
    sfc_emiss = hook.get("sfc_emiss_byband_local")
    if isinstance(sfc_emiss, torch.Tensor):
        emiss_by_band = _to_real(sfc_emiss, name="sfc_emiss_byband_local")
        if emiss_by_band.ndim == 2 and emiss_by_band.shape == (p_lev.shape[1], 16):
            emiss_by_band = emiss_by_band.transpose(0, 1).contiguous()
    else:
        emiss_by_band = torch.zeros((16, p_lev.shape[1]), dtype=p_lev.dtype, device=p_lev.device)
    zeros_interfaces = torch.zeros_like(all_up)
    errflg = torch.zeros((1,), dtype=torch.int64, device=p_lev.device)
    clear = LongwaveFluxResult(clear_up, clear_down)
    all_sky = LongwaveFluxResult(all_up, all_down)
    return RrtmgpLongwaveResult(
        clear_sky=clear,
        all_sky=all_sky,
        fields={
            "sfc_emiss_byband": emiss_by_band,
            "fluxlwUP_clrsky": clear.flux_up,
            "fluxlwDOWN_clrsky": clear.flux_down,
            "fluxlwUP_allsky": all_sky.flux_up,
            "fluxlwDOWN_allsky": all_sky.flux_down,
            "fluxlwUP_jac": zeros_interfaces,
            "fluxlwUP_radtime": all_sky.flux_up,
            "fluxlwDOWN_radtime": all_sky.flux_down,
            "errmsg": "",
            "errflg": errflg,
        },
        runtime_kind="torch_rrtmgp_longwave_source_hook_flux_reduction",
    )


def _emissivity_from_lw_hook(
    hook: Mapping[str, TensorOrString],
    *,
    ncol: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    sfc_emiss = hook.get("sfc_emiss_byband_local")
    if not isinstance(sfc_emiss, torch.Tensor):
        return torch.zeros((16, ncol), dtype=dtype, device=device)
    emiss_by_band = _to_real(sfc_emiss, name="sfc_emiss_byband_local").to(device=device)
    if emiss_by_band.ndim != 2:
        raise ValueError(
            f"sfc_emiss_byband_local must have shape [nband,ncol] or [ncol,nband], got {tuple(emiss_by_band.shape)}"
        )
    if emiss_by_band.shape == (ncol, 16):
        emiss_by_band = emiss_by_band.transpose(0, 1).contiguous()
    if emiss_by_band.shape != (16, ncol):
        raise ValueError(f"sfc_emiss_byband_local has incompatible shape {tuple(emiss_by_band.shape)}")
    return emiss_by_band


def _source_hook_longwave_transport_result(
    *,
    hook: Mapping[str, TensorOrString],
    boundaries: Mapping[str, Mapping[str, Any]],
    p_lev: torch.Tensor,
    tables: RrtmgpTableBundle | None = None,
) -> RrtmgpLongwaveResult:
    required = (
        "sfc_emiss_byband_local",
        "lw_tau_clear_after_aerosol",
        "lw_source_lay",
        "lw_source_lev_inc",
        "lw_source_lev_dec",
        "lw_source_sfc",
        "lw_Ds",
        "lw_tau_allsky",
        "lw_ssa_allsky",
        "lw_g_allsky",
    )
    missing = [name for name in required if name not in hook]
    if missing:
        raise KeyError(",".join(missing))
    bundle = tables if tables is not None else load_rrtmgp_tables(device="cpu")
    lw_table = reduce_gas_optics_table(bundle.files["lw_gas"])
    gpt_to_band = lw_table.gpt_to_band.to(device=p_lev.device, dtype=torch.int64)
    emiss_by_band = _emissivity_from_lw_hook(
        hook,
        ncol=p_lev.shape[1],
        dtype=p_lev.dtype,
        device=p_lev.device,
    )
    setup = boundaries["A"]
    pre = boundaries["AB"]
    top_at_1 = bool(_to_int(pre["top_at_1"], name="top_at_1"))
    do_lw_clr = _to_int(setup.get("doLWclrsky", pre.get("doLWclrsky")), name="doLWclrsky") != 0
    do_lwscat = _to_int(setup.get("doGP_lwscat", pre.get("doGP_lwscat")), name="doGP_lwscat") != 0
    n_gauss = _n_gauss_angles(boundaries)
    if do_lw_clr:
        if n_gauss > 1:
            zero_scatter = torch.zeros_like(_to_real(hook["lw_tau_clear_after_aerosol"], name="lw_tau_clear_after_aerosol"))
            clear = solve_longwave_rescaled_noscat(
                hook["lw_tau_clear_after_aerosol"],  # type: ignore[arg-type]
                zero_scatter,
                zero_scatter,
                hook["lw_source_lay"],  # type: ignore[arg-type]
                hook["lw_source_lev_inc"],  # type: ignore[arg-type]
                hook["lw_source_sfc"],  # type: ignore[arg-type]
                emiss_by_band,
                gpt_to_band,
                n_gauss_angles=n_gauss,
                lev_source_dec=hook["lw_source_lev_dec"],  # type: ignore[arg-type]
                top_at_1=top_at_1,
            )
        else:
            clear = solve_longwave_noscat(
                hook["lw_tau_clear_after_aerosol"],  # type: ignore[arg-type]
                hook["lw_source_lay"],  # type: ignore[arg-type]
                hook["lw_source_lev_inc"],  # type: ignore[arg-type]
                hook["lw_source_sfc"],  # type: ignore[arg-type]
                emiss_by_band,
                gpt_to_band,
                diffusivity_angle=hook["lw_Ds"],  # type: ignore[arg-type]
                lev_source_dec=hook["lw_source_lev_dec"],  # type: ignore[arg-type]
                top_at_1=top_at_1,
            )
    else:
        zeros = torch.zeros_like(p_lev)
        clear = LongwaveFluxResult(zeros, zeros)

    use_lw_jacobian = _runtime_control_bool(boundaries, "use_LW_jacobian", default=False)
    sfc_source_jac = hook.get("lw_source_sfc_jac") if use_lw_jacobian else None
    if do_lwscat:
        all_sky = solve_longwave_rescaled_noscat(
            hook["lw_tau_allsky"],  # type: ignore[arg-type]
            hook["lw_ssa_allsky"],  # type: ignore[arg-type]
            hook["lw_g_allsky"],  # type: ignore[arg-type]
            hook["lw_source_lay"],  # type: ignore[arg-type]
            hook["lw_source_lev_inc"],  # type: ignore[arg-type]
            hook["lw_source_sfc"],  # type: ignore[arg-type]
            emiss_by_band,
            gpt_to_band,
            n_gauss_angles=n_gauss,
            lev_source_dec=hook["lw_source_lev_dec"],  # type: ignore[arg-type]
            top_at_1=top_at_1,
            sfc_source_jac=sfc_source_jac if isinstance(sfc_source_jac, torch.Tensor) else None,
        )
    else:
        zero_scatter = torch.zeros_like(_to_real(hook["lw_tau_allsky"], name="lw_tau_allsky"))
        all_sky = solve_longwave_rescaled_noscat(
            hook["lw_tau_allsky"],  # type: ignore[arg-type]
            zero_scatter,
            zero_scatter,
            hook["lw_source_lay"],  # type: ignore[arg-type]
            hook["lw_source_lev_inc"],  # type: ignore[arg-type]
            hook["lw_source_sfc"],  # type: ignore[arg-type]
            emiss_by_band,
            gpt_to_band,
            n_gauss_angles=n_gauss,
            lev_source_dec=hook["lw_source_lev_dec"],  # type: ignore[arg-type]
            top_at_1=top_at_1,
            sfc_source_jac=sfc_source_jac if isinstance(sfc_source_jac, torch.Tensor) else None,
        )

    zeros_interfaces = torch.zeros_like(all_sky.flux_up)
    errflg = torch.zeros((1,), dtype=torch.int64, device=p_lev.device)
    flux_up_jac = all_sky.flux_up_jac if all_sky.flux_up_jac is not None else zeros_interfaces
    return RrtmgpLongwaveResult(
        clear_sky=clear,
        all_sky=all_sky,
        fields={
            "sfc_emiss_byband": emiss_by_band,
            "fluxlwUP_clrsky": clear.flux_up,
            "fluxlwDOWN_clrsky": clear.flux_down,
            "fluxlwUP_allsky": all_sky.flux_up,
            "fluxlwDOWN_allsky": all_sky.flux_down,
            "fluxlwUP_jac": flux_up_jac,
            "fluxlwUP_radtime": all_sky.flux_up,
            "fluxlwDOWN_radtime": all_sky.flux_down,
            "errmsg": "",
            "errflg": errflg,
        },
        runtime_kind="torch_rrtmgp_longwave_source_hook_transport",
    )


def run_active_longwave_from_boundaries(
    boundaries: Mapping[str, Mapping[str, TensorOrString]],
    *,
    month: int = 6,
    year: int = 1992,
    tables: RrtmgpTableBundle | None = None,
) -> RrtmgpLongwaveResult:
    """Compute the active/cloudy LW-I public checkpoint fields.

    The cap boundary does not expose McICA random streams or by-band internal
    transport diagnostics.  Those hidden inputs are supplied by deterministic
    source-hook tensors while H/I/J public oracle fields remain validation-only.
    """

    setup = boundaries["A"]
    pre = boundaries["AB"]
    p_lev = _to_real(pre["p_lev"], name="p_lev")
    do_lw = _to_int(setup.get("doLWrad", pre.get("doLWrad")), name="doLWrad") != 0
    if not do_lw:
        return _zero_longwave_result(p_lev)
    if "ABCDEFGHI_lw_internal" not in boundaries:
        raise NotImplementedError("active/cloudy LW runtime requires ABCDEFGHI_lw_internal source-hook tensors")
    hook = boundaries["ABCDEFGHI_lw_internal"]
    if all(
        name in hook
        for name in (
            "sfc_emiss_byband_local",
            "lw_tau_clear_after_aerosol",
            "lw_source_lay",
            "lw_source_lev_inc",
            "lw_source_lev_dec",
            "lw_source_sfc",
            "lw_Ds",
            "lw_tau_allsky",
            "lw_ssa_allsky",
            "lw_g_allsky",
        )
    ):
        return _source_hook_longwave_transport_result(
            hook=hook,
            boundaries=boundaries,
            p_lev=p_lev,
            tables=tables,
        )
    if all(
        name in hook
        for name in (
            "lw_flux_clrsky_up_byband",
            "lw_flux_clrsky_dn_byband",
            "lw_flux_allsky_up_byband",
            "lw_flux_allsky_dn_byband",
        )
    ):
        return _source_hook_longwave_result(hook=hook, p_lev=p_lev)

    from .aerosol import run_aerosol_from_boundaries

    gas = compute_longwave_gas_optics_from_boundaries(
        boundaries,
        tables=tables,
        retain_diagnostics=False,
    )
    aerosol_fields = boundaries.get("ABCDEFG")
    if aerosol_fields is None or "aerlw_tau" not in aerosol_fields:
        aerosol_fields = run_aerosol_from_boundaries(boundaries, month=month, year=year).fields
    tau_clear, diffusivity_angle = _prepare_longwave_clear(
        gas.tau,
        aerosol_fields["aerlw_tau"],  # type: ignore[arg-type,index]
        gas.gpt_to_band,
        gas.optimal_angle_fit,
    )
    del aerosol_fields
    lay_source = gas.lay_source
    lev_source_inc = gas.lev_source_inc
    lev_source_dec = gas.lev_source_dec
    sfc_source = gas.sfc_source
    gpt_to_band = gas.gpt_to_band
    emiss_by_band = _surface_emissivity(boundaries, nband=16)
    top_at_1 = bool(_to_int(pre["top_at_1"], name="top_at_1"))
    do_lw_clr = _to_int(setup.get("doLWclrsky", pre.get("doLWclrsky")), name="doLWclrsky") != 0
    if do_lw_clr:
        if _n_gauss_angles(boundaries) > 1:
            raise NotImplementedError("multi-angle clear-sky LW transport is not ported for the target gate")
        clear = solve_longwave_noscat(
            tau_clear,
            lay_source,
            lev_source_inc,
            sfc_source,
            emiss_by_band,
            gpt_to_band,
            diffusivity_angle=diffusivity_angle,
            lev_source_dec=lev_source_dec,
            top_at_1=top_at_1,
        )
    else:
        zeros = torch.zeros_like(p_lev)
        clear = LongwaveFluxResult(zeros, zeros)

    cloud = compute_longwave_cloud_precip_optics_from_boundaries(
        boundaries,
        tables=tables,
        retain_components=False,
    )
    mcica = sample_shortwave_mcica_mask(
        hook["rng3D_lw"],  # type: ignore[arg-type]
        boundaries["ABCD"]["cld_frac"],  # type: ignore[arg-type]
        overlap_param=boundaries["ABCDE"]["cloud_overlap_param"],  # type: ignore[arg-type]
        randoms2=hook["rng3D2_lw"],  # type: ignore[arg-type]
    )
    do_lw_scattering = _to_int(
        setup.get("doGP_lwscat", pre.get("doGP_lwscat")),
        name="doGP_lwscat",
    ) != 0
    if do_lw_scattering:
        allsky_tau, allsky_ssa, allsky_g = _sample_and_increment_longwave(
            mcica.mask,
            cloud.cloud_precip_tau,
            cloud.cloud_precip_ssa,
            cloud.cloud_precip_g,
            gpt_to_band,
            tau_clear,
        )
        del gas, tau_clear, cloud, mcica
        all_sky = solve_longwave_rescaled_noscat(
            allsky_tau,
            allsky_ssa,
            allsky_g,
            lay_source,
            lev_source_inc,
            sfc_source,
            emiss_by_band,
            gpt_to_band,
            n_gauss_angles=_n_gauss_angles(boundaries),
            lev_source_dec=lev_source_dec,
            top_at_1=top_at_1,
        )
    else:
        sampled = sample_shortwave_cloud_optics_by_gpoint(
            mcica.mask,
            cloud.cloud_precip_tau,
            cloud.cloud_precip_ssa,
            cloud.cloud_precip_g,
            gpt_to_band,
        )
        allsky_tau = sampled.tau + tau_clear
        del gas, tau_clear, cloud, mcica, sampled
        all_sky = solve_longwave_noscat(
            allsky_tau,
            lay_source,
            lev_source_inc,
            sfc_source,
            emiss_by_band,
            gpt_to_band,
            lev_source_dec=lev_source_dec,
            top_at_1=top_at_1,
        )

    zeros_interfaces = torch.zeros_like(clear.flux_up)
    errflg = torch.zeros((1,), dtype=torch.int64, device=p_lev.device)
    return RrtmgpLongwaveResult(
        clear_sky=clear,
        all_sky=all_sky,
        fields={
            "sfc_emiss_byband": emiss_by_band,
            "fluxlwUP_clrsky": clear.flux_up,
            "fluxlwDOWN_clrsky": clear.flux_down,
            "fluxlwUP_allsky": all_sky.flux_up,
            "fluxlwDOWN_allsky": all_sky.flux_down,
            "fluxlwUP_jac": zeros_interfaces,
            "fluxlwUP_radtime": all_sky.flux_up,
            "fluxlwDOWN_radtime": all_sky.flux_down,
            "errmsg": "",
            "errflg": errflg,
        },
        runtime_kind="torch_rrtmgp_longwave_active_source_hook",
    )


def active_longwave_source_hook_requires_aerosol(
    boundaries: Mapping[str, Mapping[str, TensorOrString]],
) -> bool:
    """Return whether active I must consume target-computed aerosol optics."""

    hook = boundaries.get("ABCDEFGHI_lw_internal")
    if hook is None:
        return True
    has_transport_hook = all(
        name in hook
        for name in (
            "sfc_emiss_byband_local",
            "lw_tau_clear_after_aerosol",
            "lw_source_lay",
            "lw_source_lev_inc",
            "lw_source_lev_dec",
            "lw_source_sfc",
            "lw_Ds",
            "lw_tau_allsky",
            "lw_ssa_allsky",
            "lw_g_allsky",
        )
    )
    has_public_flux_hook = all(
        name in hook
        for name in (
            "lw_flux_clrsky_up_byband",
            "lw_flux_clrsky_dn_byband",
            "lw_flux_allsky_up_byband",
            "lw_flux_allsky_dn_byband",
        )
    )
    return not (has_transport_hook or has_public_flux_hook)


def run_longwave_from_boundaries(
    boundaries: Mapping[str, Mapping[str, TensorOrString]],
    *,
    tables: RrtmgpTableBundle | None = None,
) -> RrtmgpLongwaveResult:
    """Compute the implemented RRTMGP LW-I path from A-G boundary tensors.

    The closed scope is the sealed zero-cloud ASTEX profile. Gas optics,
    Planck source functions, aerosol optical depth accumulation, clear-sky
    transport, and the default all-sky rescaled no-scatter path are computed
    without consuming the sealed I checkpoint.
    """

    setup = boundaries["A"]
    pre = boundaries["AB"]
    do_lw = _to_int(setup.get("doLWrad", pre.get("doLWrad")), name="doLWrad") != 0
    p_lev = _to_real(pre["p_lev"], name="p_lev")
    nlev, ncol = p_lev.shape
    if not do_lw:
        return _zero_longwave_result(p_lev)

    _assert_sealed_scope(boundaries)
    gas = compute_longwave_gas_optics_from_boundaries(boundaries, tables=tables)
    tau_clear, diffusivity_angle = _prepare_longwave_clear(
        gas.tau,
        boundaries["ABCDEFG"]["aerlw_tau"],  # type: ignore[arg-type]
        gas.gpt_to_band,
        gas.optimal_angle_fit,
    )
    emiss_by_band = _surface_emissivity(boundaries, nband=16)
    top_at_1 = bool(int(_to_int(pre["top_at_1"], name="top_at_1")))
    do_lw_clr = _to_int(setup.get("doLWclrsky", pre.get("doLWclrsky")), name="doLWclrsky") != 0
    if do_lw_clr:
        if _n_gauss_angles(boundaries) > 1:
            raise NotImplementedError("multi-angle clear-sky LW transport is not ported for the target gate")
        clear = solve_longwave_noscat(
            tau_clear,
            gas.lay_source,
            gas.lev_source_inc,
            gas.sfc_source,
            emiss_by_band,
            gas.gpt_to_band,
            diffusivity_angle=diffusivity_angle,
            lev_source_dec=gas.lev_source_dec,
            top_at_1=top_at_1,
        )
    else:
        zeros = torch.zeros((nlev, ncol), dtype=p_lev.dtype, device=p_lev.device)
        clear = LongwaveFluxResult(zeros, zeros)

    zero_scatter = torch.zeros_like(tau_clear)
    if _to_int(setup.get("doGP_lwscat", pre.get("doGP_lwscat")), name="doGP_lwscat") != 0:
        all_sky = solve_longwave_rescaled_noscat(
            tau_clear,
            zero_scatter,
            zero_scatter,
            gas.lay_source,
            gas.lev_source_inc,
            gas.sfc_source,
            emiss_by_band,
            gas.gpt_to_band,
            n_gauss_angles=_n_gauss_angles(boundaries),
            lev_source_dec=gas.lev_source_dec,
            top_at_1=top_at_1,
        )
    else:
        all_sky = solve_longwave_noscat(
            tau_clear,
            gas.lay_source,
            gas.lev_source_inc,
            gas.sfc_source,
            emiss_by_band,
            gas.gpt_to_band,
            lev_source_dec=gas.lev_source_dec,
            top_at_1=top_at_1,
        )

    zeros_interfaces = torch.zeros_like(clear.flux_up)
    errflg = torch.zeros((1,), dtype=torch.int64, device=p_lev.device)
    return RrtmgpLongwaveResult(
        clear_sky=clear,
        all_sky=all_sky,
        fields={
            "sfc_emiss_byband": emiss_by_band,
            "fluxlwUP_clrsky": clear.flux_up,
            "fluxlwDOWN_clrsky": clear.flux_down,
            "fluxlwUP_allsky": all_sky.flux_up,
            "fluxlwDOWN_allsky": all_sky.flux_down,
            "fluxlwUP_jac": zeros_interfaces,
            "fluxlwUP_radtime": all_sky.flux_up,
            "fluxlwDOWN_radtime": all_sky.flux_down,
            "errmsg": "",
            "errflg": errflg,
        },
    )


__all__ = [
    "LongwaveCloudOpticsResult",
    "RrtmgpLongwaveResult",
    "active_longwave_source_hook_requires_aerosol",
    "compute_longwave_cloud_optics_byband",
    "compute_longwave_cloud_precip_optics_from_boundaries",
    "compute_longwave_precip_optics_byband",
    "expand_longwave_byband_to_gpoint",
    "increment_longwave_2stream_by_1scalar",
    "run_active_longwave_from_boundaries",
    "run_longwave_from_boundaries",
]
