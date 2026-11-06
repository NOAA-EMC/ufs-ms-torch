# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""PyTorch-native RRTMGP cloud preprocessing kernels for closed profile branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


TensorOrString = torch.Tensor | str

RADLIQ_LWR = 2.5
RADICE_LWR = 10.0
DEFAULT_DECORRELATION_LENGTH_KM = 1.911111111111111


@dataclass(frozen=True)
class RrtmgpCloudMpResult:
    """Checkpoint-shaped cloud microphysics-to-radiation fields."""

    fields: Mapping[str, TensorOrString]
    runtime_kind: str = "torch_rrtmgp_cloud_mp_clear"


@dataclass(frozen=True)
class RrtmgpCloudOverlapResult:
    """Checkpoint-shaped cloud overlap fields."""

    fields: Mapping[str, TensorOrString]
    runtime_kind: str = "torch_rrtmgp_cloud_overlap"


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
    return int(value.reshape(-1)[0].item())


def _to_bool(value: TensorOrString, *, name: str) -> bool:
    return _to_int(value, name=name) != 0


def _tracer_field(source: Mapping[str, TensorOrString], index_name: str) -> torch.Tensor:
    tracer = _to_real(source["tracer"], name="tracer")
    if tracer.ndim != 3:
        raise ValueError(f"tracer must be [ntrac,nlay,ncol], got {tuple(tracer.shape)}")
    index = _to_int(source[index_name], name=index_name) - 1
    if index < 0 or index >= tracer.shape[0]:
        raise ValueError(f"{index_name}={index + 1} is outside tracer dimension {tracer.shape[0]}")
    return tracer[index, :, :]


def _zero_layers(template: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(template)


def run_clear_cloud_mp_from_boundaries(
    boundaries: Mapping[str, Mapping[str, TensorOrString]],
    *,
    retain_diagnostics: bool = True,
) -> RrtmgpCloudMpResult:
    """Port the closed clear-cloud `GFS_rrtmgp_cloud_mp_run` branch.

    The sealed active RRTMGP profile has no grid-mean, convective, PBL, rain, or
    snow cloud water paths. The Fortran branch still clips grid-mean and
    convective effective radii to the RRTMGP LUT lower bounds, which is why those
    radius fields are nonzero while all water paths and cloud fractions are zero.
    """

    pre = boundaries["AB"]
    p_lay = _to_real(pre["p_lay"], name="p_lay")
    nlay, ncol = p_lay.shape
    zeros = torch.zeros((nlay, ncol), dtype=p_lay.dtype, device=p_lay.device)
    errflg = torch.zeros((1,), dtype=torch.int64, device=p_lay.device)
    cld_reliq = torch.full_like(zeros, RADLIQ_LWR)
    cld_reice = torch.full_like(zeros, RADICE_LWR)
    if not retain_diagnostics:
        return RrtmgpCloudMpResult(
            fields={
                "cld_frac": zeros,
                "cld_lwp": zeros,
                "cld_iwp": zeros,
                "cld_swp": zeros,
                "cld_rwp": zeros,
                "cld_reliq": cld_reliq,
                "cld_reice": cld_reice,
                "cld_resnow": zeros,
            }
        )
    one_col_zero = torch.zeros((ncol,), dtype=p_lay.dtype, device=p_lay.device)
    return RrtmgpCloudMpResult(
        fields={
            "cld_frac": _zero_layers(zeros),
            "cld_lwp": _zero_layers(zeros),
            "cld_iwp": _zero_layers(zeros),
            "cld_swp": _zero_layers(zeros),
            "cld_rwp": _zero_layers(zeros),
            "cld_reliq": cld_reliq,
            "cld_reice": cld_reice,
            "cld_resnow": _zero_layers(zeros),
            "cld_rerain": _zero_layers(zeros),
            "cld_pbl_frac": _zero_layers(zeros),
            "cld_pbl_lwp": _zero_layers(zeros),
            "cld_pbl_iwp": _zero_layers(zeros),
            "cld_pbl_reliq": _zero_layers(zeros),
            "cld_pbl_reice": _zero_layers(zeros),
            "cld_cnv_frac": _zero_layers(zeros),
            "cld_cnv_lwp": _zero_layers(zeros),
            "cld_cnv_iwp": _zero_layers(zeros),
            "cld_cnv_reliq": cld_reliq.clone(),
            "cld_cnv_reice": cld_reice.clone(),
            "effrin_cldliq": _zero_layers(zeros),
            "effrin_cldice": _zero_layers(zeros),
            "effrin_cldsnow": _zero_layers(zeros),
            "effrin_cldrain": _zero_layers(zeros),
            "lwp_ex": one_col_zero.clone(),
            "iwp_ex": one_col_zero.clone(),
            "lwp_fc": one_col_zero.clone(),
            "iwp_fc": one_col_zero.clone(),
            "cldfra2d": one_col_zero.clone(),
            "precip_frac": _zero_layers(zeros),
            "errmsg": "",
            "errflg": errflg,
        }
    )


def _xu_randall_cloud_fraction(
    p_lay: torch.Tensor,
    qs_lay: torch.Tensor,
    relhum: torch.Tensor,
    cloud_mixing_ratio: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    clwt = 1.0e-6 * (p_lay * 0.001)
    active = cloud_mixing_ratio > clwt
    onemrh = torch.maximum(
        torch.tensor(1.0e-10, dtype=relhum.dtype, device=relhum.device),
        1.0 - relhum,
    )
    denominator = torch.minimum(
        torch.maximum(torch.sqrt(onemrh * qs_lay), torch.tensor(0.0001, dtype=relhum.dtype, device=relhum.device)),
        torch.tensor(1.0, dtype=relhum.dtype, device=relhum.device),
    )
    tem1 = float(alpha) / denominator
    tem2 = torch.maximum(
        torch.minimum(tem1 * (cloud_mixing_ratio - clwt), torch.tensor(50.0, dtype=relhum.dtype, device=relhum.device)),
        torch.zeros_like(relhum),
    )
    tem3 = torch.sqrt(torch.sqrt(relhum))
    frac = torch.maximum(tem3 * (1.0 - torch.exp(-tem2)), torch.zeros_like(relhum))
    return torch.where(active, frac, torch.zeros_like(frac))


def run_active_gfdl_cloud_mp_from_source_hook(
    source: Mapping[str, TensorOrString],
    *,
    retain_diagnostics: bool = True,
) -> RrtmgpCloudMpResult:
    """Port the active GFDL-compatible `cloud_mp_uni` radiation-cloud branch."""

    do_sw = _to_bool(source["doSWrad"], name="doSWrad")
    do_lw = _to_bool(source["doLWrad"], name="doLWrad")
    p_lay = _to_real(source["p_lay"], name="p_lay")
    nlay, ncol = p_lay.shape
    zeros = torch.zeros((nlay, ncol), dtype=p_lay.dtype, device=p_lay.device)
    errflg = torch.zeros((1,), dtype=torch.int64, device=p_lay.device)
    if not (do_sw or do_lw):
        return RrtmgpCloudMpResult(fields={"errmsg": "", "errflg": errflg})

    imp_physics = _to_int(source["imp_physics"], name="imp_physics")
    imp_gfdl = _to_int(source["imp_physics_gfdl"], name="imp_physics_gfdl")
    if imp_physics != imp_gfdl:
        raise NotImplementedError(
            f"active cloud-MP source-hook target currently closes GFDL branch only, got imp_physics={imp_physics}"
        )
    if not _to_bool(source["lgfdlmprad"], name="lgfdlmprad"):
        return RrtmgpCloudMpResult(fields={"errmsg": "ERROR: MP choice not available with RRTMGP", "errflg": torch.ones_like(errflg)})

    p_lev = _to_real(source["p_lev"], name="p_lev")
    if p_lev.shape[0] != nlay + 1:
        raise ValueError(f"p_lev must have nlay+1 interfaces, got {tuple(p_lev.shape)} for nlay={nlay}")
    con_g = _to_real(source["con_g"], name="con_g").reshape(-1)[0]
    delta_p_hpa = (p_lev[1:, :] - p_lev[:-1, :]).abs() * 0.01
    tem1 = 1.0e5 / con_g

    cld_frac = _tracer_field(source, "i_cldtot")
    q_liq = _tracer_field(source, "i_cldliq")
    q_ice = _tracer_field(source, "i_cldice")
    q_rain = _tracer_field(source, "i_cldrain")
    q_snow = _tracer_field(source, "i_cldsnow")
    q_graupel = _tracer_field(source, "i_cldgrpl")
    active = cld_frac > 0.001
    cld_lwp = torch.where(active, torch.maximum(q_liq * tem1 * delta_p_hpa, zeros), zeros)
    cld_iwp = torch.where(active, torch.maximum(q_ice * tem1 * delta_p_hpa, zeros), zeros)
    cld_rwp = torch.where(active, torch.maximum(q_rain * tem1 * delta_p_hpa, zeros), zeros)
    cld_swp = torch.where(active, torch.maximum((q_snow + q_graupel) * tem1 * delta_p_hpa, zeros), zeros)

    if _to_bool(source["effr_in"], name="effr_in"):
        cld_reliq = _to_real(source["effrin_cldliq_initial"], name="effrin_cldliq_initial").clone()
        cld_reice = torch.clamp(
            _to_real(source["effrin_cldice_initial"], name="effrin_cldice_initial"),
            min=10.0,
            max=150.0,
        ).clone()
        cld_resnow = _to_real(source["effrin_cldsnow_initial"], name="effrin_cldsnow_initial").clone()
        cld_rerain = (
            _to_real(source["effrin_cldrain_initial"], name="effrin_cldrain_initial").clone()
            if retain_diagnostics
            else None
        )
    else:
        raise NotImplementedError("active GFDL cloud-MP gate currently covers effr_in=.true. source-hook cases")

    if _to_bool(source["doGP_cldoptics_PADE"], name="doGP_cldoptics_PADE") or _to_bool(
        source["doGP_cldoptics_LUT"], name="doGP_cldoptics_LUT"
    ):
        cld_reliq = torch.clamp(cld_reliq, min=RADLIQ_LWR, max=21.5)
        cld_reice = torch.clamp(cld_reice, min=RADICE_LWR, max=180.0)

    if not retain_diagnostics:
        return RrtmgpCloudMpResult(
            fields={
                "cld_frac": cld_frac,
                "cld_lwp": cld_lwp,
                "cld_iwp": cld_iwp,
                "cld_swp": cld_swp,
                "cld_rwp": cld_rwp,
                "cld_reliq": cld_reliq,
                "cld_reice": cld_reice,
                "cld_resnow": cld_resnow,
            },
            runtime_kind="torch_rrtmgp_cloud_mp_gfdl_active_source_hook_minimal",
        )

    one_col_zero = torch.zeros((ncol,), dtype=p_lay.dtype, device=p_lay.device)
    conv_reliq = zeros.clone()
    conv_reice = zeros.clone()
    imfdeep = _to_int(source["imfdeepcnv"], name="imfdeepcnv")
    if imfdeep in {
        _to_int(source["imfdeepcnv_gf"], name="imfdeepcnv_gf"),
        _to_int(source["imfdeepcnv_samf"], name="imfdeepcnv_samf"),
    }:
        conv_reliq = torch.full_like(zeros, RADLIQ_LWR)
        conv_reice = torch.full_like(zeros, RADICE_LWR)

    return RrtmgpCloudMpResult(
        fields={
            "cld_frac": cld_frac,
            "cld_lwp": cld_lwp,
            "cld_iwp": cld_iwp,
            "cld_swp": cld_swp,
            "cld_rwp": cld_rwp,
            "cld_reliq": cld_reliq,
            "cld_reice": cld_reice,
            "cld_resnow": cld_resnow,
            "cld_rerain": cld_rerain,
            "cld_pbl_frac": zeros.clone(),
            "cld_pbl_lwp": zeros.clone(),
            "cld_pbl_iwp": zeros.clone(),
            "cld_pbl_reliq": zeros.clone(),
            "cld_pbl_reice": zeros.clone(),
            "cld_cnv_frac": zeros.clone(),
            "cld_cnv_lwp": zeros.clone(),
            "cld_cnv_iwp": zeros.clone(),
            "cld_cnv_reliq": conv_reliq,
            "cld_cnv_reice": conv_reice,
            "effrin_cldliq": _to_real(source["effrin_cldliq_initial"], name="effrin_cldliq_initial"),
            "effrin_cldice": _to_real(source["effrin_cldice_initial"], name="effrin_cldice_initial"),
            "effrin_cldsnow": _to_real(source["effrin_cldsnow_initial"], name="effrin_cldsnow_initial"),
            "effrin_cldrain": _to_real(source["effrin_cldrain_initial"], name="effrin_cldrain_initial"),
            "lwp_ex": one_col_zero.clone(),
            "iwp_ex": one_col_zero.clone(),
            "lwp_fc": one_col_zero.clone(),
            "iwp_fc": one_col_zero.clone(),
            "cldfra2d": cld_frac[:-1, :].max(dim=0).values if nlay > 1 else cld_frac.max(dim=0).values,
            "precip_frac": cld_frac.clone(),
            "errmsg": "",
            "errflg": errflg,
        },
        runtime_kind="torch_rrtmgp_cloud_mp_gfdl_active_source_hook",
    )


def compute_exponential_cloud_overlap(
    delta_z_center: torch.Tensor,
    cld_frac: torch.Tensor,
    *,
    decorrelation_length_km: float | torch.Tensor = DEFAULT_DECORRELATION_LENGTH_KM,
    exponential_random: bool = False,
) -> torch.Tensor:
    """Match `module_radiation_cloud_overlap::get_alpha_exper`.

    Checkpoint tensors use `[nlay, ncol]`, while the Fortran routine uses
    `(ncol,nlay)`. The first vertical level is exactly zero; lower levels are
    `exp(-(deltaZc*0.001)/dcorr_lgth)`, with the optional exponential-random
    clear-after-cloud reset.
    """

    dz = _to_real(delta_z_center, name="delta_z_center")
    frac = _to_real(cld_frac, name="cld_frac")
    if dz.shape != frac.shape:
        raise ValueError(f"delta_z_center and cld_frac shapes differ: {dz.shape} != {frac.shape}")
    if isinstance(decorrelation_length_km, torch.Tensor):
        decorrelation = _to_real(decorrelation_length_km, name="decorrelation_length_km").to(device=dz.device)
    else:
        decorrelation = torch.tensor(float(decorrelation_length_km), dtype=dz.dtype, device=dz.device)
    if decorrelation.ndim == 0 or decorrelation.numel() == 1:
        decorrelation = decorrelation.reshape(1, 1)
    elif decorrelation.ndim == 1 and decorrelation.numel() == dz.shape[1]:
        decorrelation = decorrelation.reshape(1, dz.shape[1])
    else:
        raise ValueError(
            "decorrelation_length_km must be scalar or one value per column; "
            f"got {tuple(decorrelation.shape)} for cloud field {tuple(dz.shape)}"
        )
    alpha = torch.zeros_like(dz)
    if dz.shape[0] > 1:
        alpha[1:, :] = torch.exp(-(dz[1:, :] * 0.001) / decorrelation)
    if exponential_random and dz.shape[0] > 1:
        clear_after_cloud = (frac[1:, :] == 0.0) & (frac[:-1, :] > 0.0)
        alpha[1:, :] = torch.where(clear_after_cloud, torch.zeros_like(alpha[1:, :]), alpha[1:, :])
    return alpha


def compute_hogan_decorrelation_length(xlat: torch.Tensor, con_pi: torch.Tensor) -> torch.Tensor:
    """Match Hogan/Shonk latitude-dependent cloud decorrelation length."""

    lat = _to_real(xlat, name="xlat")
    pi = _to_real(con_pi, name="con_pi").reshape(-1)[0].to(device=lat.device)
    return torch.maximum(
        torch.full((), 0.6, dtype=lat.dtype, device=lat.device),
        2.78 - 4.6 * torch.abs(lat / pi),
    )


def run_active_cloud_overlap_from_boundaries(
    boundaries: Mapping[str, Mapping[str, TensorOrString]],
    *,
    exponential_random: bool = False,
) -> RrtmgpCloudOverlapResult:
    """Port active RRTMGP cloud overlap using pre-call latitude controls."""

    if "AB_pre_inputs" not in boundaries:
        raise NotImplementedError("active cloud overlap requires AB_pre_inputs for latitude decorrelation controls")
    controls = boundaries["AB_pre_inputs"]
    decorrelation = compute_hogan_decorrelation_length(
        controls["xlat"],  # type: ignore[arg-type]
        controls["con_pi"],  # type: ignore[arg-type]
    )
    return run_cloud_overlap_from_boundaries(
        boundaries,
        decorrelation_length_km=decorrelation,
        exponential_random=exponential_random,
    )


def run_cloud_overlap_from_boundaries(
    boundaries: Mapping[str, Mapping[str, TensorOrString]],
    *,
    decorrelation_length_km: float = DEFAULT_DECORRELATION_LENGTH_KM,
    exponential_random: bool = False,
) -> RrtmgpCloudOverlapResult:
    """Port the closed RRTMGP cloud-overlap branch for the active profile."""

    surface = boundaries["ABC"]
    pre = boundaries["AB"]
    cloud = boundaries["ABCD"]
    delta_zc = _to_real(pre["deltaZc"], name="deltaZc")
    cld_frac = _to_real(cloud["cld_frac"], name="cld_frac")
    overlap = compute_exponential_cloud_overlap(
        delta_zc,
        cld_frac,
        decorrelation_length_km=decorrelation_length_km,
        exponential_random=exponential_random,
    )
    errflg = torch.zeros((1,), dtype=torch.int64, device=delta_zc.device)
    return RrtmgpCloudOverlapResult(
        fields={
            "sfc_alb_nir_dir": _to_real(surface["sfc_alb_nir_dir"], name="sfc_alb_nir_dir"),
            "sfc_alb_nir_dif": _to_real(surface["sfc_alb_nir_dif"], name="sfc_alb_nir_dif"),
            "sfc_alb_uvvis_dir": _to_real(surface["sfc_alb_uvvis_dir"], name="sfc_alb_uvvis_dir"),
            "sfc_alb_uvvis_dif": _to_real(surface["sfc_alb_uvvis_dif"], name="sfc_alb_uvvis_dif"),
            "semis": _to_real(surface["semis"], name="semis"),
            "tsfg": _to_real(surface["tsfg"], name="tsfg"),
            "cloud_overlap_param": overlap,
            "errmsg": "",
            "errflg": errflg,
        }
    )


__all__ = [
    "DEFAULT_DECORRELATION_LENGTH_KM",
    "RADICE_LWR",
    "RADLIQ_LWR",
    "RrtmgpCloudMpResult",
    "RrtmgpCloudOverlapResult",
    "compute_hogan_decorrelation_length",
    "compute_exponential_cloud_overlap",
    "run_active_cloud_overlap_from_boundaries",
    "run_active_gfdl_cloud_mp_from_source_hook",
    "run_clear_cloud_mp_from_boundaries",
    "run_cloud_overlap_from_boundaries",
]
