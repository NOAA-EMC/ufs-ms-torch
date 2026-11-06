# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Torch helpers for the Noah-MP land-body H/I/J checkpoint surfaces.

This module is intentionally separate from ``driver.py`` so it can be wired into
the cumulative target after the H/I/J implementation is reviewed.  The routines
below follow the source-smoke option corridor only and reject unsupported
branches instead of broadening the target claim silently.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Mapping, NamedTuple, Sequence

import torch
from ufs_physics_common.precision import working_power
from .thermo import static_tensor

from .compile import compile_categories_enabled, run_compile_island
from .land_energy import (
    DZSOIL_4L,
    NOAHMP_SB as LAND_ENERGY_SB,
    bare_flux_opt_sfc3,
    btran_opt_btr1_source_corridor,
    compute_energy_geometry,
    load_land_energy_tables,
    radiation_bats_two_stream,
    soil_parameter_by_column,
    soilwater_opt_run1_inf1_source_corridor,
    thermoprop_source_corridor,
    _state_cached_tensor,
    _tsnosoi_source_corridor_flat_impl,
    tsnosoi_source_corridor,
    vege_flux_opt_sfc3,
    _positive_soil_ice,
)
from .tables import GFS_V17_PROFILE_SCM_TABLE_VARIANT, NoahMpTableBundle
from .snow_layers import combine_snow_layers


SMOOTH_TENSOR_OVERRIDE_KEY = "_smooth_tensor_overrides"
PREPARED_TENSOR_COLUMNS_KEY = "_prepared_tensor_columns"

CON_RD = 287.05
CON_RV = 461.50
CON_EPS = CON_RD / CON_RV
CON_EPSM1 = CON_EPS - 1.0
CON_HFUS = 3.3358e5
NOAHMP_RAIR = 287.04
NOAHMP_CPAIR = 1004.64
NOAHMP_TFRZ = 273.16
NOAHMP_HVAP = 2.5104e06
NOAHMP_HSUB = 2.8440e06
NOAHMP_HFUS = 0.3336e06
NOAHMP_CWAT = 4.188e06
NOAHMP_CICE = 2.094e06
NOAHMP_DENH2O = 1000.0
NOAHMP_DENICE = 917.0
NOAHMP_GRAV = 9.80616
NOAHMP_SB = 5.67e-08
NOAHMP_VKC = 0.40


def _pow2(value: torch.Tensor) -> torch.Tensor:
    return value * value


def _pow3(value: torch.Tensor) -> torch.Tensor:
    return value * value * value


def _pow4(value: torch.Tensor) -> torch.Tensor:
    squared = value * value
    return squared * squared


LAND_BODY_SUPPORTED_OPTIONS = {
    "idveg": 4,
    "iopt_alb": 1,
    "iopt_btr": 1,
    "iopt_crs": 2,
    "iopt_diag": 2,
    "iopt_frz": 1,
    "iopt_inf": 1,
    "iopt_rad": 3,
    "iopt_run": 1,
    "iopt_sfc": 3,
    "iopt_snf": 4,
    "iopt_stc": 3,
    "iopt_tbot": 2,
    "iopt_trs": 2,
}

H_SCALAR_FIELDS = (
    "fsa",
    "fsr",
    "fira",
    "fsh",
    "fcev",
    "fgev",
    "fctr",
    "ssoil",
    "trad",
    "psn",
    "apar",
    "ponding",
)
I_SCALAR_FIELDS = ("qvap", "qdew", "edir", "ecan", "etran", "runsrf", "runsub", "qsnbot", "esnow")
J_SCALAR_FIELDS = ("gpp", "npp", "nee", "errwat", "fveg", "albedo", "snowh", "sneqv", "qsfc")


@dataclass(frozen=True)
class LandBodyState:
    """Column-major state needed immediately after Noah-MP land boundary G."""

    column_indices: torch.Tensor
    vegtype: torch.Tensor
    soiltyp: torch.Tensor
    soilcol: torch.Tensor
    slopetyp: torch.Tensor
    temperature_forcing: torch.Tensor
    q1: torch.Tensor
    ps: torch.Tensor
    prsl1: torch.Tensor
    wind: torch.Tensor
    u1: torch.Tensor
    v1: torch.Tensor
    cm: torch.Tensor
    ch: torch.Tensor
    dlwflx: torch.Tensor
    xcoszin: torch.Tensor
    garea: torch.Tensor
    zf: torch.Tensor
    pblh: torch.Tensor
    shdfac: torch.Tensor
    sfalb: torch.Tensor
    snoalb: torch.Tensor
    tskin: torch.Tensor
    tg3: torch.Tensor
    snow_depth: torch.Tensor
    snow_water_equiv: torch.Tensor
    snow_levels: torch.Tensor
    temperature_snow_soil: torch.Tensor
    interface_depth: torch.Tensor
    snow_level_ice: torch.Tensor
    snow_level_liquid: torch.Tensor
    snow_ice_frac_old: torch.Tensor
    soil_moisture_vol: torch.Tensor
    soil_liquid_vol: torch.Tensor
    tair: torch.Tensor
    qair: torch.Tensor
    eair: torch.Tensor
    rhoair: torch.Tensor
    swdown: torch.Tensor
    precipitation: torch.Tensor
    rain: torch.Tensor
    snow: torch.Tensor
    fpice: torch.Tensor
    fveg: torch.Tensor
    elai: torch.Tensor
    esai: torch.Tensor
    cmc: torch.Tensor
    canopy_liquid_begin: torch.Tensor
    canopy_ice_begin: torch.Tensor
    canopy_liquid_initial: torch.Tensor
    canopy_ice_initial: torch.Tensor
    qrain_ground: torch.Tensor
    qsnow_ground: torch.Tensor
    snow_depth_rate_ground: torch.Tensor
    fwet: torch.Tensor
    bdfall: torch.Tensor
    pahv: torch.Tensor
    pahg: torch.Tensor
    pahb: torch.Tensor
    dzsnso: torch.Tensor
    prslki: torch.Tensor | None = None
    prslk1: torch.Tensor | None = None
    prsik1: torch.Tensor | None = None
    temperature_leaf: torch.Tensor | None = None
    temperature_ground: torch.Tensor | None = None
    temperature_canopy_air: torch.Tensor | None = None
    vapor_pres_canopy_air: torch.Tensor | None = None
    snow_age: torch.Tensor | None = None
    snow_water_equiv_old: torch.Tensor | None = None
    aquifer_water: torch.Tensor | None = None
    depth_water_table: torch.Tensor | None = None
    saturated_water: torch.Tensor | None = None
    friction_velocity: torch.Tensor | None = None
    veg_idx: torch.Tensor | None = None
    soil_idx: torch.Tensor | None = None
    soilcol_idx: torch.Tensor | None = None
    soil_layer_idx: torch.Tensor | None = None
    mfsno_by_column: torch.Tensor | None = None
    scffac_by_column: torch.Tensor | None = None
    z0mvt_by_column: torch.Tensor | None = None
    hvt_by_column: torch.Tensor | None = None
    cwpvt_by_column: torch.Tensor | None = None
    ch2op_by_column: torch.Tensor | None = None
    dleaf_by_column: torch.Tensor | None = None
    cbiom_by_column: torch.Tensor | None = None
    xl_by_column: torch.Tensor | None = None
    rhol_by_column: torch.Tensor | None = None
    rhos_by_column: torch.Tensor | None = None
    taul_by_column: torch.Tensor | None = None
    taus_by_column: torch.Tensor | None = None
    rgl_by_column: torch.Tensor | None = None
    rsmin_by_column: torch.Tensor | None = None
    rsmax_by_column: torch.Tensor | None = None
    topt_by_column: torch.Tensor | None = None
    hs_by_column: torch.Tensor | None = None
    nroot_by_column: torch.Tensor | None = None
    nroot_int_by_column: torch.Tensor | None = None
    smcmax_surface_by_column: torch.Tensor | None = None
    smcwlt_surface_by_column: torch.Tensor | None = None
    bexp_surface_by_column: torch.Tensor | None = None
    psisat_surface_by_column: torch.Tensor | None = None
    smcmax_by_column: torch.Tensor | None = None
    smcref_by_column: torch.Tensor | None = None
    smcwlt_by_column: torch.Tensor | None = None
    psisat_by_column: torch.Tensor | None = None
    bexp_by_column: torch.Tensor | None = None
    quartz_by_column: torch.Tensor | None = None
    dksat_by_column: torch.Tensor | None = None
    dwsat_by_column: torch.Tensor | None = None
    albsat_by_column: torch.Tensor | None = None
    albdry_by_column: torch.Tensor | None = None
    albsat_soilcol_by_column: torch.Tensor | None = None
    albdry_soilcol_by_column: torch.Tensor | None = None


@dataclass(frozen=True)
class LandBodyHIJ:
    """Vectorized H/I/J fields before checkpoint-shaped emission."""

    h: Mapping[str, torch.Tensor]
    i: Mapping[str, torch.Tensor]
    j: Mapping[str, torch.Tensor]
    tail: Mapping[str, torch.Tensor]


class LandBodyTailOutput(NamedTuple):
    """Flat K-O tail state needed by the production final-O surface."""

    temperature_radiative: torch.Tensor
    surface_temperature: torch.Tensor
    sensible_heat_total: torch.Tensor
    latent_heat_total: torch.Tensor
    ground_heat_total: torch.Tensor
    z0_total: torch.Tensor
    z0h_total: torch.Tensor
    exchange_heat_total: torch.Tensor
    soil_moisture_vol: torch.Tensor
    temperature_snow_soil: torch.Tensor
    spec_humidity_surface: torch.Tensor
    sw_absorbed_total: torch.Tensor
    emissivity_total: torch.Tensor
    runoff_surface: torch.Tensor
    vegetation_fraction: torch.Tensor


LAND_BODY_TAIL_OUTPUT_FIELDS = LandBodyTailOutput._fields


def _tensor(values: Any, *, device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=dtype)


def _int_tensor(values: Any, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=torch.int64)


def _column(case_data: Mapping[str, Any], name: str) -> list[Any]:
    return [column[name] for column in case_data["columns"]]


def _device_matches(actual: torch.device, requested: torch.device) -> bool:
    if actual.type != requested.type:
        return False
    if requested.type == "cuda" and requested.index is None:
        return torch.cuda.is_available() and actual.index == torch.cuda.current_device()
    return actual.index == requested.index


def _to_device_dtype(value: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if value.dtype == dtype and _device_matches(value.device, device):
        return value
    return value.to(device=device, dtype=dtype)


def _case_real(case_data: Mapping[str, Any]) -> torch.dtype:
    """The working dtype of a case: its live temperature column's (the coupled state), else float64."""
    for key in (SMOOTH_TENSOR_OVERRIDE_KEY, PREPARED_TENSOR_COLUMNS_KEY):
        group = case_data.get(key)
        value = group.get("t1") if isinstance(group, Mapping) else None
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.dtype
    return torch.float64


def _smooth_override_tensor(
    case_data: Mapping[str, Any],
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor | None:
    overrides = case_data.get(SMOOTH_TENSOR_OVERRIDE_KEY)
    if not isinstance(overrides, Mapping) or name not in overrides:
        return None
    value = overrides[name]
    if isinstance(value, torch.Tensor):
        return _to_device_dtype(value, device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _prepared_column_tensor(
    case_data: Mapping[str, Any],
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    prepared = case_data.get(PREPARED_TENSOR_COLUMNS_KEY)
    if not isinstance(prepared, Mapping) or name not in prepared:
        return None
    value = prepared[name]
    if isinstance(value, torch.Tensor):
        return _to_device_dtype(value, device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _column_tensor(case_data: Mapping[str, Any], name: str, *, device: torch.device) -> torch.Tensor:
    real = _case_real(case_data)
    override = _smooth_override_tensor(case_data, name, device=device, dtype=real)
    if override is not None:
        return override
    prepared = _prepared_column_tensor(case_data, name, device=device, dtype=real)
    if prepared is not None:
        return prepared
    return _tensor(_column(case_data, name), device=device, dtype=real)


def _column_int_tensor(case_data: Mapping[str, Any], name: str, *, device: torch.device) -> torch.Tensor:
    prepared = _prepared_column_tensor(case_data, name, device=device, dtype=torch.int64)
    if prepared is not None:
        return prepared
    return _int_tensor(_column(case_data, name), device=device)


def _has_column(case_data: Mapping[str, Any], name: str) -> bool:
    """Whether the host columns carry ``name``; from the prepared column names when the case has them.

    Dynamo guards every value of a host column dict that a membership test reads, so a compiled call
    would recompile whenever its first column changed; the prepared name set is one guard.
    """
    metadata = case_data.get("_prepared_metadata")
    if isinstance(metadata, Mapping) and "column_names" in metadata:
        return name in metadata["column_names"]
    return name in case_data["columns"][0]


def _optional_column(case_data: Mapping[str, Any], name: str, default: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if _has_column(case_data, name) or name in case_data.get(SMOOTH_TENSOR_OVERRIDE_KEY, {}):
        return _column_tensor(case_data, name, device=device)
    return default


def _matrix_tensor(case_data: Mapping[str, Any], name: str, *, device: torch.device) -> torch.Tensor:
    real = _case_real(case_data)
    override = _smooth_override_tensor(case_data, name, device=device, dtype=real)
    if override is not None:
        return override
    prepared = _prepared_column_tensor(case_data, name, device=device, dtype=real)
    if prepared is not None:
        return prepared
    return _tensor(_column(case_data, name), device=device, dtype=real)


def _fortran_nint(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0.0, torch.floor(x + 0.5), torch.ceil(x - 0.5)).to(torch.int64)


def _replace_column_tuple(columns: tuple[torch.Tensor, ...], index: int, values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(values if pos == int(index) else column for pos, column in enumerate(columns))


def _columns_to_matrix(columns: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.stack(columns, dim=1)


def _replace_matrix_column(values: torch.Tensor, index: int, column: torch.Tensor) -> torch.Tensor:
    column_index = int(index)
    selector = torch.arange(values.shape[1], device=values.device) == column_index
    return values + (column.unsqueeze(1) - values[:, column_index : column_index + 1]) * selector.unsqueeze(0)


def _replace_first_three_columns(values: torch.Tensor, first_three: torch.Tensor) -> torch.Tensor:
    if values.shape[1] == 3:
        return first_three
    return torch.column_stack(
        (
            first_three[:, 0],
            first_three[:, 1],
            first_three[:, 2],
            values[:, 3],
            values[:, 4],
            values[:, 5],
            values[:, 6],
        )
    )


def _g_boundary_matrix(g_outputs: Mapping[str, torch.Tensor], field: str, indices: torch.Tensor) -> torch.Tensor:
    compact_key = f"G__{field}"
    if compact_key in g_outputs:
        values = g_outputs[compact_key]
        return values[indices]
    if field in g_outputs:
        return g_outputs[field][indices]
    # Legacy checkpoint keys carry static column labels, not tensor values.
    prefix = f"G__{field}__col"
    labels = sorted(int(key[len(prefix):]) for key in g_outputs if key.startswith(prefix))
    if not labels:
        raise KeyError(f"Missing {field} checkpoint columns")
    rows = torch.stack([g_outputs[f"{prefix}{label}"] for label in labels], dim=0)
    label_tensor = torch.tensor(labels, dtype=indices.dtype, device=indices.device)
    positions = torch.searchsorted(label_tensor, indices + 1)
    found = (positions < len(labels)) & (label_tensor.index_select(0, positions.clamp(max=len(labels) - 1)) == indices + 1)
    torch._assert_async(torch.all(found), f"Missing {field} checkpoint columns")
    return rows.index_select(0, positions)


def _g_boundary_vector(g_outputs: Mapping[str, torch.Tensor], field: str, indices: torch.Tensor) -> torch.Tensor:
    return _g_boundary_matrix(g_outputs, field, indices)


def _g_private_tensor(g_outputs: Mapping[str, torch.Tensor], field: str, indices: torch.Tensor) -> torch.Tensor | None:
    values = g_outputs.get(f"_{field}")
    if values is None:
        return None
    return values[indices]


def _select_active_columns(values: torch.Tensor, indices: torch.Tensor, *, indices_cover_all: bool) -> torch.Tensor:
    if indices_cover_all:
        return values
    return values[indices]


def _g_boundary_active(
    g_outputs: Mapping[str, torch.Tensor],
    field: str,
    indices: torch.Tensor,
    *,
    indices_cover_all: bool,
) -> torch.Tensor:
    compact_key = f"G__{field}"
    if compact_key in g_outputs:
        return _select_active_columns(g_outputs[compact_key], indices, indices_cover_all=indices_cover_all)
    if field in g_outputs:
        return _select_active_columns(g_outputs[field], indices, indices_cover_all=indices_cover_all)
    return _g_boundary_matrix(g_outputs, field, indices)


def _g_private_active(
    g_outputs: Mapping[str, torch.Tensor],
    field: str,
    indices: torch.Tensor,
    *,
    indices_cover_all: bool,
) -> torch.Tensor | None:
    values = g_outputs.get(f"_{field}")
    if values is None:
        return None
    return _select_active_columns(values, indices, indices_cover_all=indices_cover_all)


def _fresh_snow_density(
    case_data: Mapping[str, Any], temperature: torch.Tensor, *, device: torch.device,
) -> torch.Tensor:
    bdfall = torch.clamp(67.92 + 51.25 * torch.exp((temperature - NOAHMP_TFRZ) / 2.59), max=120.0)
    opt_snf = int(case_data["driver_options"]["iopt_snf"])
    if opt_snf in {4, 5}:
        snow = _column_tensor(case_data, "snow_mp", device=device)
        graupel = _column_tensor(case_data, "graupel_mp", device=device)
        hail = _column_tensor(case_data, "ice_mp", device=device)
        frozen = snow + graupel + hail
        frozen_input = (_column_tensor(case_data, "rainn_mp", device=device) > 0.0) & (frozen > 0.0)
        safe_frozen = torch.where(frozen_input, frozen, 1.0)
        if opt_snf == 4:
            mixed = bdfall * (snow / safe_frozen) + 500.0 * (graupel / safe_frozen) + 917.0 * (hail / safe_frozen)
        else:
            mixed = _column_tensor(case_data, "rhonewsn1", device=device)
        bdfall = torch.where(frozen_input, mixed, bdfall)
    return bdfall


def _ground_precip_after_interception(
    case_data: Mapping[str, Any],
    tables: NoahMpTableBundle,
    *,
    indices: torch.Tensor,
    indices_cover_all: bool = False,
    fveg: torch.Tensor,
    elai: torch.Tensor,
    esai: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    return_heat: bool = False,
    bdfall: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    opts = case_data["driver_options"]
    t1_all = _column_tensor(case_data, "t1", device=device)
    u1_all = _column_tensor(case_data, "u1", device=device)
    v1_all = _column_tensor(case_data, "v1", device=device)
    rainc_mp = _column_tensor(case_data, "rainc_mp", device=device)
    rainn_mp = _column_tensor(case_data, "rainn_mp", device=device)
    snow_mp = _column_tensor(case_data, "snow_mp", device=device)
    graupel_mp = _column_tensor(case_data, "graupel_mp", device=device)
    ice_mp = _column_tensor(case_data, "ice_mp", device=device)
    canopy_liquid_initial = _optional_column(case_data, "canliqxy", _column_tensor(case_data, "canopy", device=device), device=device)
    canopy_ice_initial = _optional_column(case_data, "canicexy", canopy_liquid_initial * 0.0, device=device)
    vegtype_all = _column_int_tensor(case_data, "vegtype", device=device)
    veg_index = (vegtype_all if vegtype_all.dtype == torch.int64 else vegtype_all.to(torch.int64)) - 1

    prcpconv = rainc_mp
    prcpnonc = rainn_mp
    prcpshcv = 0.0
    prcpsnow = snow_mp
    prcpgrpl = graupel_mp
    prcphail = ice_mp
    prcp = prcpconv + prcpnonc + prcpshcv
    opt_snf = int(opts["iopt_snf"])
    if opt_snf == 4:
        qprecc = prcpconv + prcpshcv
        qprecl = prcpnonc
    else:
        qprecc = 0.10 * prcp
        qprecl = 0.90 * prcp
    precip_area = qprecc + qprecl
    fp_denom = 10.0 * qprecc + qprecl
    fp_denom = torch.where((precip_area > 0.0) & (torch.abs(fp_denom) > 1.0e-30), fp_denom, 1.0)
    fp = torch.where(precip_area > 0.0, precip_area / fp_denom, 0.0)

    if opt_snf == 1:
        fpice = torch.where(
            t1_all > NOAHMP_TFRZ + 2.5,
            0.0,
            torch.where(
                t1_all <= NOAHMP_TFRZ + 0.5,
                1.0,
                torch.where(t1_all <= NOAHMP_TFRZ + 2.0, 1.0 - (-54.632 + 0.2 * t1_all), 0.6),
            ),
        )
    elif opt_snf == 2:
        fpice = torch.where(t1_all >= NOAHMP_TFRZ + 2.2, 0.0, 1.0)
    elif opt_snf == 3:
        fpice = torch.where(t1_all >= NOAHMP_TFRZ, 0.0, 1.0)
    else:
        fpice = 0.0

    if bdfall is None:
        bdfall = _fresh_snow_density(case_data, t1_all, device=device)
    if opt_snf in {4, 5}:
        prcp_frozen = prcpsnow + prcpgrpl + prcphail
        frozen_input = (prcpnonc > 0.0) & (prcp_frozen > 0.0)
        safe_nonc = torch.where(frozen_input, prcpnonc, 1.0)
        fpice = torch.where(frozen_input, torch.clamp(prcp_frozen / safe_nonc, min=0.0, max=1.0), 0.0)

    rain = prcp * (1.0 - fpice)
    snow = prcp * fpice
    select = (lambda value: value) if indices_cover_all else (lambda value: value[indices])
    rain = select(rain)
    snow = select(snow)
    fp = select(fp)
    bdfall = select(bdfall)
    canopy_liquid = select(canopy_liquid_initial)
    canopy_ice = select(canopy_ice_initial)
    temperature_forcing = select(t1_all)
    temperature_leaf = select(_optional_column(
        case_data, "tvxy", _column_tensor(case_data, "tskin", device=device), device=device
    ))
    veg_index = select(veg_index)
    u1 = select(u1_all)
    v1 = select(v1_all)

    dt = float(opts["delt"])
    positive_canopy = (elai + esai) > 0.0
    maxliq = tables.ch2op[veg_index] * (elai + esai)
    safe_maxliq = torch.where(positive_canopy & (maxliq > 1.0e-30), maxliq, 1.0)
    qintr_canopy = fveg * rain * fp
    qintr_limit = (maxliq - canopy_liquid) / dt * (1.0 - torch.exp(-rain * dt / safe_maxliq))
    qintr_canopy = torch.clamp(torch.minimum(qintr_canopy, qintr_limit), min=0.0)
    qdripr_canopy = fveg * rain - qintr_canopy
    qthror_canopy = (1.0 - fveg) * rain
    canopy_liquid_after = torch.clamp(canopy_liquid + qintr_canopy * dt, min=0.0)
    qdripr = torch.where(positive_canopy, qdripr_canopy, torch.where(canopy_liquid > 0.0, canopy_liquid / dt, 0.0))
    qthror = torch.where(positive_canopy, qthror_canopy, rain)
    canopy_liquid_after = torch.where(positive_canopy, canopy_liquid_after, 0.0)

    maxsno = 6.6 * (0.27 + torch.div(46.0, bdfall)) * (elai + esai)
    safe_maxsno = torch.where(positive_canopy & (maxsno > 1.0e-30), maxsno, 1.0)
    qints_canopy = fveg * snow * fp
    qints_limit = (maxsno - canopy_ice) / dt * (1.0 - torch.exp(-snow * dt / safe_maxsno))
    qints_canopy = torch.clamp(torch.minimum(qints_canopy, qints_limit), min=0.0)
    ft = torch.clamp((temperature_leaf - 270.15) / 1.87e5, min=0.0)
    fv = torch.sqrt(u1 * u1 + v1 * v1) / 1.56e5
    icedrip = torch.clamp(canopy_ice, min=0.0) * (fv + ft)
    qdrips_canopy = (fveg * snow - qints_canopy) + icedrip
    qthros_canopy = (1.0 - fveg) * snow
    canopy_ice_after = torch.clamp(canopy_ice + (qints_canopy - icedrip) * dt, min=0.0)
    qdrips = torch.where(positive_canopy, qdrips_canopy, torch.where(canopy_ice > 0.0, canopy_ice / dt, 0.0))
    qthros = torch.where(positive_canopy, qthros_canopy, snow)
    canopy_ice_after = torch.where(positive_canopy, canopy_ice_after, 0.0)
    fwet_ice = torch.clamp(canopy_ice_after, min=0.0) / torch.clamp(maxsno, min=1.0e-6)
    fwet_liq = torch.clamp(canopy_liquid_after, min=0.0) / torch.clamp(maxliq, min=1.0e-6)
    fwet = torch.where(canopy_ice_after > 0.0, fwet_ice, fwet_liq)
    fwet_limited = torch.clamp(fwet, min=0.0, max=1.0)
    fwet = torch.where(
        fwet_limited > 0.0,
        working_power(torch.clamp(fwet_limited, min=1.0e-30), 0.667),
        0.0,
    )
    qsnow_ground = qdrips + qthros
    snow_depth_rate = qsnow_ground / torch.clamp(bdfall, min=1.0e-12)
    _ = dtype
    water = (qdripr + qthror, qsnow_ground, canopy_liquid_after, canopy_ice_after, snow_depth_rate, fwet)
    if not return_heat:
        return water

    temperature_ground = select(_optional_column(
        case_data, "tgxy", _column_tensor(case_data, "tskin", device=device), device=device
    ))
    pah_ac = fveg * rain * (NOAHMP_CWAT / 1000.0) * (temperature_forcing - temperature_leaf)
    pah_cg = qdripr * (NOAHMP_CWAT / 1000.0) * (temperature_leaf - temperature_ground)
    pah_ag = qthror * (NOAHMP_CWAT / 1000.0) * (temperature_forcing - temperature_ground)
    pah_ac = pah_ac + fveg * snow * (NOAHMP_CICE / 1000.0) * (temperature_forcing - temperature_leaf)
    pah_cg = pah_cg + qdrips * (NOAHMP_CICE / 1000.0) * (temperature_leaf - temperature_ground)
    pah_ag = pah_ag + qthros * (NOAHMP_CICE / 1000.0) * (temperature_forcing - temperature_ground)
    fractional = (fveg > 0.0) & (fveg < 1.0)
    canopy_denominator = torch.where(fractional, fveg, 1.0)
    bare_denominator = torch.where(fractional, 1.0 - fveg, 1.0)
    pahv = torch.where(fveg <= 0.0, 0.0, pah_ac - pah_cg)
    pahg = torch.where(fveg <= 0.0, 0.0, pah_cg / canopy_denominator)
    pahb = torch.where(fveg <= 0.0, pah_cg + pah_ag,
                       torch.where(fveg >= 1.0, 0.0, pah_ag / bare_denominator))
    return (*water, torch.clamp(pahv, min=-20.0, max=20.0),
            torch.clamp(pahg, min=-20.0, max=20.0),
            torch.clamp(pahb, min=-20.0, max=20.0))




def _require_real(name: str, value: torch.Tensor) -> None:
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must be torch.float64 (checkpoint parity) or torch.float32; got {value.dtype}")


def _validate_state(opts: Mapping[str, Any], tables: NoahMpTableBundle, state: LandBodyState) -> None:
    bad = {
        name: opts.get(name)
        for name, expected in LAND_BODY_SUPPORTED_OPTIONS.items()
        if int(opts.get(name, expected)) != expected
        and not (
            name == "iopt_sfc"
            and int(opts.get(name, expected)) == 4
        )
    }
    if bad:
        raise NotImplementedError(f"Noah-MP land-body H/I/J helper supports only the sealed source-smoke options; got {bad}")
    if int(opts.get("opt_crop", 0)) != 0:
        raise NotImplementedError("Noah-MP crop carbon branches are not supported by the H/I/J helper.")
    if bool(opts.get("cpllnd", False)) or bool(opts.get("cpllnd2atm", False)):
        raise NotImplementedError("Noah-MP land IAU coupling branches are outside the sealed H/I/J source-smoke corridor.")
    if not bool(opts.get("thsfc_loc", True)):
        raise NotImplementedError("The H/I/J helper supports only the sealed thsfc_loc=.true. surface-temperature mode.")
    _ = tables
    for field in fields(state):
        value = getattr(state, field.name)
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            _require_real(field.name, value)


def _host_active_land_indices(case_data: Mapping[str, Any], tables: NoahMpTableBundle) -> list[int]:
    metadata = case_data.get("_prepared_metadata")
    if isinstance(metadata, Mapping) and metadata.get("active_land_indices") is not None:
        return [int(index) for index in metadata["active_land_indices"]]
    dry_flag = None
    if isinstance(metadata, Mapping):
        dry_flag = metadata.get("dry_flag_iter_indices")
    columns = list(case_data["columns"])
    if dry_flag is None:
        dry_flag = [
            index for index, column in enumerate(columns)
            if bool(column.get("dry", False)) and bool(column.get("flag_iter", False))
        ]
    return [
        int(index) for index in dry_flag
        if int(float(columns[int(index)].get("vegtype", 0))) != int(tables.isice)
    ]


def _active_land_indices_cover_all_columns(case_data: Mapping[str, Any], tables: NoahMpTableBundle) -> bool:
    metadata = case_data.get("_prepared_metadata")
    if isinstance(metadata, Mapping) and metadata.get("active_land_indices_cover_all_columns") is not None:
        return bool(metadata["active_land_indices_cover_all_columns"])
    im = int(case_data["dimensions"]["im"])
    active_land_indices = _host_active_land_indices(case_data, tables)
    return len(active_land_indices) == im and all(index == position for position, index in enumerate(active_land_indices))


def _active_land_selection(
    case_data: Mapping[str, Any], tables: NoahMpTableBundle, *, device: torch.device
) -> tuple[torch.Tensor | None, bool]:
    """The active non-glacier land columns: (device index tensor, whether they are all columns in order).

    The index tensor is None when there are none. A prepared case holds both in its metadata, so a
    compiled call takes the indices as one tensor input and traces and guards no host index list.
    """
    metadata = case_data.get("_prepared_metadata")
    if isinstance(metadata, Mapping) and metadata.get("active_land_index_tensor") is not None:
        indices = metadata["active_land_index_tensor"]
        if indices.shape[0] == 0:
            return None, False
        return indices.to(device), bool(metadata["active_land_indices_cover_all_columns"])
    active_land_indices = _host_active_land_indices(case_data, tables)
    if not active_land_indices:
        return None, False
    return (static_tensor(active_land_indices, device=device, dtype=torch.int64),
            _active_land_indices_cover_all_columns(case_data, tables))


def _active_land_snowxy_round_below_minus2(case_data: Mapping[str, Any], active_land_indices: list[int]) -> bool:
    metadata = case_data.get("_prepared_metadata")
    if isinstance(metadata, Mapping) and metadata.get("active_land_snowxy_round_below_minus2") is not None:
        return bool(metadata["active_land_snowxy_round_below_minus2"])
    columns = list(case_data["columns"])
    return any(round(float(columns[index].get("snowxy", 0.0))) < -2 for index in active_land_indices)


def state_from_case_after_g(
    case_data: Mapping[str, Any],
    g_outputs: Mapping[str, torch.Tensor],
    tables: NoahMpTableBundle,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> LandBodyState:
    """Build H/I/J helper inputs from a materialized case and current G output.

    This reads only case JSON fields and the target's own G emissions.  It does
    not read or infer from checkpoint/oracle values. ``dtype`` defaults to the case's
    working dtype (``_case_real``): float64 for checkpoint parity, float32 when the
    coupled state is float32.
    """
    dtype = _case_real(case_data) if dtype is None else dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Noah-MP runs in float64 (checkpoint parity) or float32.")
    dev = torch.device("cpu") if device is None else torch.device(device)
    dims = case_data["dimensions"]
    if int(dims["km"]) != 4 or int(dims["lsnowl"]) != -2:
        raise NotImplementedError("The sealed Noah-MP source-smoke corpus uses km=4 and lsnowl=-2.")

    indices, active_land_indices_cover_all = _active_land_selection(case_data, tables, device=dev)
    if indices is None:
        raise NotImplementedError("No active non-glacier land columns are available for H/I/J.")
    vegtype_all = _column_int_tensor(case_data, "vegtype", device=dev)
    active_vegtype = _select_active_columns(vegtype_all, indices, indices_cover_all=active_land_indices_cover_all)
    active_soiltyp = _select_active_columns(
        _column_int_tensor(case_data, "soiltyp", device=dev)[:, 0],
        indices,
        indices_cover_all=active_land_indices_cover_all,
    )
    active_soilcol = _select_active_columns(
        _column_int_tensor(case_data, "soilcol", device=dev),
        indices,
        indices_cover_all=active_land_indices_cover_all,
    )
    active_slopetyp = _select_active_columns(
        _column_int_tensor(case_data, "slopetyp", device=dev),
        indices,
        indices_cover_all=active_land_indices_cover_all,
    )
    energy_tables = load_land_energy_tables(
        device=dev,
        dtype=dtype,
        table_variant=tables.table_variant,
    )
    veg_idx = active_vegtype - 1
    soil_idx = active_soiltyp - 1
    soilcol_idx = active_soilcol - 1
    soil_layer_idx = soil_idx[:, None].expand(active_soiltyp.shape[0], 4)

    q1_all = _column_tensor(case_data, "q1", device=dev)
    ps_all = _column_tensor(case_data, "ps", device=dev)
    t1_all = _column_tensor(case_data, "t1", device=dev)
    prsl1_all = _optional_column(case_data, "prsl1", ps_all * 0.995, device=dev)
    private_snow_depth = _g_private_active(g_outputs, "snow_depth", indices, indices_cover_all=active_land_indices_cover_all)
    private_snow_water_equiv = _g_private_active(g_outputs, "snow_water_equiv", indices, indices_cover_all=active_land_indices_cover_all)
    private_snow_levels = _g_private_active(g_outputs, "snow_levels", indices, indices_cover_all=active_land_indices_cover_all)
    private_temperature_snow_soil = _g_private_active(g_outputs, "temperature_snow_soil", indices, indices_cover_all=active_land_indices_cover_all)
    private_interface_depth = _g_private_active(g_outputs, "interface_depth", indices, indices_cover_all=active_land_indices_cover_all)
    private_snow_level_ice = _g_private_active(g_outputs, "snow_level_ice", indices, indices_cover_all=active_land_indices_cover_all)
    private_snow_level_liquid = _g_private_active(g_outputs, "snow_level_liquid", indices, indices_cover_all=active_land_indices_cover_all)
    private_snow_ice_frac_old = _g_private_active(g_outputs, "snow_ice_frac_old", indices, indices_cover_all=active_land_indices_cover_all)
    if (
        private_snow_depth is None
        or private_snow_water_equiv is None
        or private_snow_levels is None
        or private_temperature_snow_soil is None
        or private_interface_depth is None
        or private_snow_level_ice is None
        or private_snow_level_liquid is None
        or private_snow_ice_frac_old is None
    ):
        im = int(dims["im"])
        snow_depth_all = _column_tensor(case_data, "snwdph", device=dev) * 0.001
        weasd_all = _column_tensor(case_data, "weasd", device=dev)
        snow_depth_all = torch.where((weasd_all != 0.0) & (snow_depth_all == 0.0), 10.0 * weasd_all / 1000.0, snow_depth_all)
        prepared_snow_levels = _prepared_column_tensor(case_data, "snow_levels", device=dev, dtype=torch.int64)
        snow_levels_all = prepared_snow_levels if prepared_snow_levels is not None else _fortran_nint(_column_tensor(case_data, "snowxy", device=dev))
        stc_all = _matrix_tensor(case_data, "stc", device=dev)
        z_profile = torch.tensor([-0.03, -0.07, -0.12, -0.1, -0.4, -1.0, -2.0], device=dev, dtype=dtype)
        interface_depth_all = z_profile.expand(im, 7)
        temperature_snow_soil_all = torch.column_stack((t1_all, t1_all, t1_all, stc_all[:, 0], stc_all[:, 1], stc_all[:, 2], stc_all[:, 3]))
        snow_level_ice_all = torch.where(
            (_column_tensor(case_data, "snowxy", device=dev) < 0.0)[:, None],
            (weasd_all / 3.0)[:, None].expand(im, 3),
            0.0,
        )
        snow_level_liquid_all = snow_level_ice_all * 0.0
        positive_snow_ice = snow_level_ice_all > 0.0
        frac_columns = []
        for snow_index in [-2, -1, 0]:
            layer = snow_index + 2
            participates = snow_levels_all + 1 <= snow_index
            denom = snow_level_ice_all[:, layer] + snow_level_liquid_all[:, layer]
            frac = torch.where(positive_snow_ice[:, layer], snow_level_ice_all[:, layer] / denom, 0.0)
            frac_columns.append(torch.where(participates, frac, 0.0))
        snow_ice_frac_old_all = torch.column_stack(tuple(frac_columns))
        snow_depth = _select_active_columns(snow_depth_all, indices, indices_cover_all=active_land_indices_cover_all)
        snow_water_equiv = _select_active_columns(weasd_all, indices, indices_cover_all=active_land_indices_cover_all)
        snow_levels = _select_active_columns(snow_levels_all, indices, indices_cover_all=active_land_indices_cover_all)
        temperature_snow_soil = _select_active_columns(
            temperature_snow_soil_all,
            indices,
            indices_cover_all=active_land_indices_cover_all,
        )
        interface_depth = _select_active_columns(interface_depth_all, indices, indices_cover_all=active_land_indices_cover_all)
        snow_level_ice = _select_active_columns(snow_level_ice_all, indices, indices_cover_all=active_land_indices_cover_all)
        snow_level_liquid = _select_active_columns(snow_level_liquid_all, indices, indices_cover_all=active_land_indices_cover_all)
        snow_ice_frac_old = _select_active_columns(snow_ice_frac_old_all, indices, indices_cover_all=active_land_indices_cover_all)
    else:
        snow_depth = private_snow_depth
        snow_water_equiv = private_snow_water_equiv
        snow_levels = private_snow_levels
        temperature_snow_soil = private_temperature_snow_soil
        interface_depth = private_interface_depth
        snow_level_ice = private_snow_level_ice
        snow_level_liquid = private_snow_level_liquid
        snow_ice_frac_old = private_snow_ice_frac_old

    fveg_g = _g_boundary_active(g_outputs, "fveg", indices, indices_cover_all=active_land_indices_cover_all)
    elai_g = _g_boundary_active(g_outputs, "elai", indices, indices_cover_all=active_land_indices_cover_all)
    esai_g = _g_boundary_active(g_outputs, "esai", indices, indices_cover_all=active_land_indices_cover_all)
    canopy_liquid_begin = _optional_column(case_data, "canliqxy", _column_tensor(case_data, "canopy", device=dev), device=dev)
    canopy_ice_begin = _optional_column(case_data, "canicexy", canopy_liquid_begin * 0.0, device=dev)
    qrain_ground = _g_private_active(g_outputs, "qrain_ground", indices, indices_cover_all=active_land_indices_cover_all)
    qsnow_ground = _g_private_active(g_outputs, "qsnow_ground", indices, indices_cover_all=active_land_indices_cover_all)
    canopy_liquid_after_g = _g_private_active(g_outputs, "canopy_liquid_after_g", indices, indices_cover_all=active_land_indices_cover_all)
    canopy_ice_after_g = _g_private_active(g_outputs, "canopy_ice_after_g", indices, indices_cover_all=active_land_indices_cover_all)
    snow_depth_rate_ground = _g_private_active(g_outputs, "snow_depth_rate_ground", indices, indices_cover_all=active_land_indices_cover_all)
    fwet_after_g = _g_private_active(g_outputs, "fwet_after_g", indices, indices_cover_all=active_land_indices_cover_all)
    bdfall = _g_private_active(g_outputs, "bdfall", indices, indices_cover_all=active_land_indices_cover_all)
    if bdfall is None:
        bdfall = _select_active_columns(_fresh_snow_density(case_data, t1_all, device=dev),
            indices, indices_cover_all=active_land_indices_cover_all)
    if (
        qrain_ground is None
        or qsnow_ground is None
        or canopy_liquid_after_g is None
        or canopy_ice_after_g is None
        or snow_depth_rate_ground is None
        or fwet_after_g is None
    ):
        qrain_ground, qsnow_ground, canopy_liquid_after_g, canopy_ice_after_g, snow_depth_rate_ground, fwet_after_g = _ground_precip_after_interception(
            case_data,
            tables,
            indices=indices,
            indices_cover_all=active_land_indices_cover_all,
            fveg=fveg_g,
            elai=elai_g,
            esai=esai_g,
            device=dev,
            dtype=dtype,
        )

    return LandBodyState(
        **{field: _select_active_columns(_column_tensor(case_data, arg, device=dev), indices, indices_cover_all=active_land_indices_cover_all)
           for field, arg in {"prslki": "prslki", "prslk1": "prslk1", "prsik1": "prsik1",
                              "temperature_leaf": "tvxy", "temperature_ground": "tgxy",
                              "temperature_canopy_air": "tahxy", "vapor_pres_canopy_air": "eahxy",
                              "snow_age": "taussxy", "snow_water_equiv_old": "sneqvoxy",
                              "aquifer_water": "waxy", "depth_water_table": "zwtxy", "saturated_water": "wtxy",
                              "friction_velocity": "ustar1"}.items()
           if _has_column(case_data, arg)},
        column_indices=indices,
        vegtype=active_vegtype,
        soiltyp=active_soiltyp,
        soilcol=active_soilcol,
        slopetyp=active_slopetyp,
        temperature_forcing=_select_active_columns(t1_all, indices, indices_cover_all=active_land_indices_cover_all),
        q1=_select_active_columns(q1_all, indices, indices_cover_all=active_land_indices_cover_all),
        ps=_select_active_columns(ps_all, indices, indices_cover_all=active_land_indices_cover_all),
        prsl1=_select_active_columns(prsl1_all, indices, indices_cover_all=active_land_indices_cover_all),
        wind=_select_active_columns(_column_tensor(case_data, "wind", device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        u1=_select_active_columns(_column_tensor(case_data, "u1", device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        v1=_select_active_columns(_column_tensor(case_data, "v1", device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        cm=_select_active_columns(_optional_column(case_data, "cmxy", _column_tensor(case_data, "cm", device=dev), device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        ch=_select_active_columns(_optional_column(case_data, "chxy", _column_tensor(case_data, "ch", device=dev), device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        dlwflx=_select_active_columns(_column_tensor(case_data, "dlwflx", device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        xcoszin=_select_active_columns(_column_tensor(case_data, "xcoszin", device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        garea=_select_active_columns(_column_tensor(case_data, "garea", device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        zf=_select_active_columns(_column_tensor(case_data, "zf", device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        pblh=_select_active_columns(_column_tensor(case_data, "pblh", device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        shdfac=_select_active_columns(_column_tensor(case_data, "sigmaf", device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        sfalb=_select_active_columns(_column_tensor(case_data, "sfalb", device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        snoalb=_select_active_columns(_column_tensor(case_data, "snoalb", device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        tskin=_select_active_columns(_column_tensor(case_data, "tskin", device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        tg3=_select_active_columns(_column_tensor(case_data, "tg3", device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        snow_depth=snow_depth,
        snow_water_equiv=snow_water_equiv,
        snow_levels=snow_levels,
        temperature_snow_soil=temperature_snow_soil,
        interface_depth=interface_depth,
        snow_level_ice=snow_level_ice,
        snow_level_liquid=snow_level_liquid,
        snow_ice_frac_old=snow_ice_frac_old,
        soil_moisture_vol=_select_active_columns(_matrix_tensor(case_data, "smc", device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        soil_liquid_vol=_select_active_columns(_matrix_tensor(case_data, "slc", device=dev), indices, indices_cover_all=active_land_indices_cover_all),
        tair=_g_boundary_active(g_outputs, "thair", indices, indices_cover_all=active_land_indices_cover_all),
        qair=_g_boundary_active(g_outputs, "qair", indices, indices_cover_all=active_land_indices_cover_all),
        eair=_g_boundary_active(g_outputs, "eair", indices, indices_cover_all=active_land_indices_cover_all),
        rhoair=_g_boundary_active(g_outputs, "rhoair", indices, indices_cover_all=active_land_indices_cover_all),
        swdown=_g_boundary_active(g_outputs, "swdown", indices, indices_cover_all=active_land_indices_cover_all),
        precipitation=_select_active_columns(
            _column_tensor(case_data, "rainc_mp", device=dev)
            + _column_tensor(case_data, "rainn_mp", device=dev),
            indices, indices_cover_all=active_land_indices_cover_all),
        rain=_g_boundary_active(g_outputs, "rain", indices, indices_cover_all=active_land_indices_cover_all),
        snow=_g_boundary_active(g_outputs, "snow", indices, indices_cover_all=active_land_indices_cover_all),
        fpice=_g_boundary_active(g_outputs, "fpice", indices, indices_cover_all=active_land_indices_cover_all),
        fveg=fveg_g,
        elai=elai_g,
        esai=esai_g,
        cmc=_g_boundary_vector(g_outputs, "cmc", indices),
        canopy_liquid_begin=_select_active_columns(canopy_liquid_begin, indices, indices_cover_all=active_land_indices_cover_all),
        canopy_ice_begin=_select_active_columns(canopy_ice_begin, indices, indices_cover_all=active_land_indices_cover_all),
        canopy_liquid_initial=canopy_liquid_after_g,
        canopy_ice_initial=canopy_ice_after_g,
        qrain_ground=qrain_ground,
        qsnow_ground=qsnow_ground,
        snow_depth_rate_ground=snow_depth_rate_ground,
        fwet=fwet_after_g,
        bdfall=bdfall,
        pahv=_g_boundary_active(g_outputs, "pahv", indices, indices_cover_all=active_land_indices_cover_all),
        pahg=_g_boundary_active(g_outputs, "pahg", indices, indices_cover_all=active_land_indices_cover_all),
        pahb=_g_boundary_active(g_outputs, "pahb", indices, indices_cover_all=active_land_indices_cover_all),
        dzsnso=_g_boundary_active(g_outputs, "dzsnso", indices, indices_cover_all=active_land_indices_cover_all),
        veg_idx=veg_idx,
        soil_idx=soil_idx,
        soilcol_idx=soilcol_idx,
        soil_layer_idx=soil_layer_idx,
        mfsno_by_column=energy_tables.mfsno[veg_idx],
        scffac_by_column=energy_tables.scffac[veg_idx],
        z0mvt_by_column=energy_tables.z0mvt[veg_idx],
        hvt_by_column=energy_tables.hvt[veg_idx],
        cwpvt_by_column=energy_tables.cwpvt[veg_idx],
        ch2op_by_column=energy_tables.ch2op[veg_idx],
        dleaf_by_column=energy_tables.dleaf[veg_idx],
        cbiom_by_column=energy_tables.cbiom[veg_idx],
        xl_by_column=energy_tables.xl[veg_idx],
        rhol_by_column=energy_tables.rhol[veg_idx],
        rhos_by_column=energy_tables.rhos[veg_idx],
        taul_by_column=energy_tables.taul[veg_idx],
        taus_by_column=energy_tables.taus[veg_idx],
        rgl_by_column=energy_tables.rgl[veg_idx],
        rsmin_by_column=energy_tables.rsmin[veg_idx],
        rsmax_by_column=energy_tables.rsmax[veg_idx],
        topt_by_column=energy_tables.topt[veg_idx],
        hs_by_column=energy_tables.hs[veg_idx],
        nroot_by_column=energy_tables.nroot[veg_idx],
        nroot_int_by_column=energy_tables.nroot_int[veg_idx],
        smcmax_surface_by_column=soil_parameter_by_column(energy_tables, "smcmax", soil_idx, active_vegtype),
        smcwlt_surface_by_column=soil_parameter_by_column(energy_tables, "smcwlt", soil_idx, active_vegtype),
        bexp_surface_by_column=energy_tables.bexp[soil_idx],
        psisat_surface_by_column=energy_tables.psisat[soil_idx],
        smcmax_by_column=soil_parameter_by_column(energy_tables, "smcmax", soil_layer_idx, active_vegtype),
        smcref_by_column=soil_parameter_by_column(energy_tables, "smcref", soil_layer_idx, active_vegtype),
        smcwlt_by_column=soil_parameter_by_column(energy_tables, "smcwlt", soil_layer_idx, active_vegtype),
        psisat_by_column=energy_tables.psisat[soil_layer_idx],
        bexp_by_column=energy_tables.bexp[soil_layer_idx],
        quartz_by_column=energy_tables.quartz[soil_layer_idx],
        dksat_by_column=energy_tables.dksat[soil_layer_idx],
        dwsat_by_column=energy_tables.dwsat[soil_layer_idx],
        albsat_by_column=energy_tables.albsat[soil_idx],
        albdry_by_column=energy_tables.albdry[soil_idx],
        albsat_soilcol_by_column=energy_tables.albsat[soilcol_idx],
        albdry_soilcol_by_column=energy_tables.albdry[soilcol_idx],
    )


def _esat(t_celsius: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    esw = 100.0 * (
        6.107799961
        + t_celsius
        * (
            4.436518521e-01
            + t_celsius
            * (
                1.428945805e-02
                + t_celsius
                * (
                    2.650648471e-04
                    + t_celsius * (3.031240396e-06 + t_celsius * (2.034080948e-08 + t_celsius * 6.136820929e-11))
                )
            )
        )
    )
    esi = 100.0 * (
        6.109177956
        + t_celsius
        * (
            5.034698970e-01
            + t_celsius
            * (
                1.886013408e-02
                + t_celsius
                * (
                    4.176223716e-04
                    + t_celsius * (5.824720280e-06 + t_celsius * (4.838803174e-08 + t_celsius * 1.838826904e-10))
                )
            )
        )
    )
    desw = 100.0 * (
        4.438099984e-01
        + t_celsius
        * (
            2.857002636e-02
            + t_celsius
            * (
                7.938054040e-04
                + t_celsius
                * (
                    1.215215065e-05
                    + t_celsius * (1.036561403e-07 + t_celsius * (3.532421810e-10 - t_celsius * 7.090244804e-13))
                )
            )
        )
    )
    desi = 100.0 * (
        5.030305237e-01
        + t_celsius
        * (
            3.773255020e-02
            + t_celsius
            * (
                1.267995369e-03
                + t_celsius
                * (
                    2.477563108e-05
                    + t_celsius * (3.005693132e-07 + t_celsius * (2.158542548e-09 + t_celsius * 7.131097725e-12))
                )
            )
        )
    )
    return esw, esi, desw, desi


def _snow_cover_fraction(state: LandBodyState) -> torch.Tensor:
    bdsno = torch.where(state.snow_depth > 0.0, state.snow_water_equiv / torch.clamp(state.snow_depth, min=1.0e-12), 0.0)
    fmelt = _pow2(torch.clamp(bdsno / 100.0, min=1.0e-12))
    fsno = torch.tanh(state.snow_depth / torch.clamp(0.020 * fmelt, min=1.0e-12))
    return torch.where((state.snow_depth <= 1.0e-6) | (state.snow_water_equiv <= 1.0e-3), 0.0, fsno)


def _radiation_source_corridor(
    state: LandBodyState,
    *,
    dt: torch.Tensor,
    table_variant: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    object,
    object,
    object,
]:
    energy_tables = load_land_energy_tables(
        device=state.temperature_forcing.device,
        dtype=state.temperature_forcing.dtype,
        table_variant=table_variant,
    )
    geometry = compute_energy_geometry(
        state,
        energy_tables,
        options=LAND_BODY_SUPPORTED_OPTIONS,
        soilcolor=state.soilcol,
    )
    radiation = radiation_bats_two_stream(
        state,
        energy_tables,
        geometry,
        dt=dt,
        soilcolor=state.soilcol,
    )
    apar = radiation.parsun * radiation.laisun + radiation.parsha * radiation.laisha
    psn = -999.99 * (radiation.laisun + radiation.laisha)
    psn = torch.where((state.elai + state.esai) > 0.0, psn, 0.0)
    return radiation.fsa, radiation.fsr, radiation.sav, radiation.sag, apar, psn, geometry.fsno, energy_tables, geometry, radiation


def _soil_heat_update(state: LandBodyState, ssoil: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    dz = torch.clamp(torch.abs(state.dzsnso), min=1.0e-6)
    hcp = state.temperature_snow_soil * 0.0 + 2.0e6
    top = 3
    top_update = state.temperature_snow_soil[:, top] + dt * ssoil / (hcp[:, top] * dz[:, top])
    stc = state.temperature_snow_soil
    return _columns_to_matrix((stc[:, 0], stc[:, 1], stc[:, 2], top_update, stc[:, 4], stc[:, 5], stc[:, 6]))


def _surface_flux_source_corridor_flat_impl(
    *,
    u1: torch.Tensor,
    v1: torch.Tensor,
    snow_depth: torch.Tensor,
    zf: torch.Tensor,
    dzsnso: torch.Tensor,
    tskin: torch.Tensor,
    cm: torch.Tensor,
    ch: torch.Tensor,
    prsl1: torch.Tensor,
    rhoair: torch.Tensor,
    temperature_forcing: torch.Tensor,
    eair: torch.Tensor,
    dlwflx: torch.Tensor,
    temperature_snow_soil: torch.Tensor,
    pahb: torch.Tensor,
    sag: torch.Tensor,
    fsno: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ur = torch.clamp(torch.sqrt(u1 * u1 + v1 * v1), min=1.0)
    z0m = torch.clamp(0.01 * (1.0 - fsno) + 0.002 * fsno, min=1.0e-6)
    z0h = torch.clamp(z0m * 0.1, min=1.0e-6)
    zlvl = torch.clamp(snow_depth, min=0.0) + zf
    cgh = 2.0 * 0.35 / torch.clamp(torch.abs(dzsnso[:, 3]), min=1.0e-6)
    latent_zero = tskin * 0.0
    latheag = torch.where(tskin > NOAHMP_TFRZ, latent_zero + NOAHMP_HVAP, latent_zero + NOAHMP_HSUB)
    gamma = NOAHMP_CPAIR * prsl1 / (CON_EPS * latheag)
    emg = 0.97 * (1.0 - fsno) + 0.95 * fsno
    cir = emg * NOAHMP_SB
    tgb = tskin
    cm = torch.clamp(cm, min=1.0e-5)
    ch = torch.clamp(ch, min=1.0e-5)
    fv = torch.sqrt(torch.clamp(cm * ur * ur, min=1.0e-12))
    fh2 = torch.log((2.0 + z0h) / z0h)
    for _ in range(5):
        rahb = torch.clamp(1.0 / (ch * ur), min=1.0)
        rawb = rahb
        temp_c = torch.clamp(tgb - NOAHMP_TFRZ, min=-50.0, max=50.0)
        esw, esi, desw, desi = _esat(temp_c)
        estg = torch.where(temp_c > 0.0, esw, esi)
        destg = torch.where(temp_c > 0.0, desw, desi)
        csh = rhoair * NOAHMP_CPAIR / rahb
        cev = rhoair * NOAHMP_CPAIR / gamma / (50.0 + rawb)
        tgb_pow3 = _pow3(tgb)
        tgb_pow4 = _pow4(tgb)
        irb = cir * tgb_pow4 - emg * dlwflx
        shb = csh * (tgb - temperature_forcing)
        evb = cev * (estg - eair)
        ghb = cgh * (tgb - temperature_snow_soil[:, 3])
        residual = sag - irb - shb - evb - ghb + pahb
        coeff = 4.0 * cir * tgb_pow3 + csh + cev * destg + cgh
        dtg = residual / torch.clamp(coeff, min=1.0e-12)
        irb = irb + 4.0 * cir * tgb_pow3 * dtg
        shb = shb + csh * dtg
        evb = evb + cev * destg * dtg
        ghb = ghb + cgh * dtg
        tgb = tgb + dtg
    fire = dlwflx + irb
    trad = torch.pow(torch.clamp((fire - (1.0 - emg) * dlwflx) / (emg * NOAHMP_SB), min=1.0e-12), 0.25)
    estg_nonnegative = torch.clamp(estg, min=0.0)
    qsfc = CON_EPS * estg_nonnegative / (prsl1 + CON_EPSM1 * estg_nonnegative)
    ehb2 = fv * NOAHMP_VKC / torch.clamp(fh2, min=1.0e-6)
    t2m = torch.where(ehb2 < 1.0e-5, tgb, tgb - shb / (rhoair * NOAHMP_CPAIR) / torch.where(ehb2 < 1.0e-5, 1.0, ehb2))
    _ = (z0m, zlvl, t2m)
    return irb, shb, evb, ghb, trad, qsfc, tgb, latheag, emg


def _surface_flux_source_corridor(
    state: LandBodyState,
    sag: torch.Tensor,
    fsno: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ground_temperature = state.tskin if state.temperature_ground is None else state.temperature_ground
    return run_compile_island(
        "land_surface_flux",
        "surface_flux_source_corridor",
        _surface_flux_source_corridor_flat_impl,
        u1=state.u1,
        v1=state.v1,
        snow_depth=state.snow_depth,
        zf=state.zf,
        dzsnso=state.dzsnso,
        tskin=ground_temperature,
        cm=state.cm,
        ch=state.ch,
        prsl1=state.prsl1,
        rhoair=state.rhoair,
        temperature_forcing=state.temperature_forcing,
        eair=state.eair,
        dlwflx=state.dlwflx,
        temperature_snow_soil=state.temperature_snow_soil,
        pahb=state.pahb,
        sag=sag,
        fsno=fsno,
    )


def _canwater_source_corridor(
    state: LandBodyState,
    energy_tables: Any,
    fcev: torch.Tensor,
    fctr: torch.Tensor,
    latheav: torch.Tensor,
    frozen_canopy: torch.Tensor,
    dt: torch.Tensor,
    temperature_leaf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    veg_idx = (state.vegtype if state.vegtype.dtype == torch.int64 else state.vegtype.to(torch.int64)) - 1
    ch2op = state.ch2op_by_column if state.ch2op_by_column is not None else energy_tables.ch2op[veg_idx]
    maxliq = ch2op * (state.elai + state.esai)
    return run_compile_island(
        "land_canwater",
        "canwater_source_corridor",
        _canwater_source_corridor_flat_impl,
        canopy_liquid_initial=state.canopy_liquid_initial,
        canopy_ice_initial=state.canopy_ice_initial,
        bdfall=state.bdfall,
        tskin=temperature_leaf,
        elai=state.elai,
        esai=state.esai,
        maxliq=maxliq,
        fcev=fcev,
        fctr=fctr,
        frozen_canopy=frozen_canopy,
        dt=dt,
    )


def _canwater_source_corridor_flat_impl(
    *,
    canopy_liquid_initial: torch.Tensor,
    canopy_ice_initial: torch.Tensor,
    bdfall: torch.Tensor,
    tskin: torch.Tensor,
    elai: torch.Tensor,
    esai: torch.Tensor,
    maxliq: torch.Tensor,
    fcev: torch.Tensor,
    fctr: torch.Tensor,
    frozen_canopy: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    canliq = torch.clamp(canopy_liquid_initial, min=0.0)
    canice = torch.clamp(canopy_ice_initial, min=0.0)
    etran_unfrozen = torch.clamp(fctr / NOAHMP_HVAP, min=0.0)
    qevac_unfrozen = torch.clamp(fcev / NOAHMP_HVAP, min=0.0)
    qdewc_unfrozen = torch.abs(torch.clamp(fcev / NOAHMP_HVAP, max=0.0))
    etran_frozen = torch.clamp(fctr / NOAHMP_HSUB, min=0.0)
    qsubc_frozen = torch.clamp(fcev / NOAHMP_HSUB, min=0.0)
    qfroc_frozen = torch.abs(torch.clamp(fcev / NOAHMP_HSUB, max=0.0))
    etran = torch.where(frozen_canopy, etran_frozen, etran_unfrozen)
    qevac = torch.where(frozen_canopy, 0.0, qevac_unfrozen)
    qdewc = torch.where(frozen_canopy, 0.0, qdewc_unfrozen)
    qsubc = torch.where(frozen_canopy, qsubc_frozen, 0.0)
    qfroc = torch.where(frozen_canopy, qfroc_frozen, 0.0)
    qevac = torch.minimum(canliq / dt, qevac)
    canliq = torch.clamp(canliq + (qdewc - qevac) * dt, min=0.0)
    canliq = torch.where(canliq <= 1.0e-6, 0.0, canliq)
    qsubc = torch.minimum(canice / dt, qsubc)
    canice = torch.clamp(canice + (qfroc - qsubc) * dt, min=0.0)
    canice = torch.where(canice <= 1.0e-6, 0.0, canice)
    maxsno = 6.6 * (0.27 + torch.div(46.0, bdfall)) * (elai + esai)
    fwet = torch.where(
        canice > 0.0,
        canice / torch.clamp(maxsno, min=1.0e-6),
        canliq / torch.clamp(maxliq, min=1.0e-6),
    )
    fwet_limited = torch.clamp(fwet, min=0.0, max=1.0)
    fwet = torch.where(
        fwet_limited > 0.0,
        working_power(torch.clamp(fwet_limited, min=1.0e-30), 0.667),
        0.0,
    )

    melts = (canice > 1.0e-6) & (tskin > NOAHMP_TFRZ)
    qmeltc = torch.minimum(canice / dt, (tskin - NOAHMP_TFRZ) * NOAHMP_CICE * canice / NOAHMP_DENICE / (dt * NOAHMP_HFUS))
    canice_melt = torch.clamp(canice - qmeltc * dt, min=0.0)
    canliq_melt = torch.clamp(canliq + qmeltc * dt, min=0.0)
    canice = torch.where(melts, canice_melt, canice)
    canliq = torch.where(melts, canliq_melt, canliq)
    freezes = (canliq > 1.0e-6) & (tskin < NOAHMP_TFRZ)
    qfrzc = torch.minimum(canliq / dt, (NOAHMP_TFRZ - tskin) * NOAHMP_CWAT * canliq / NOAHMP_DENH2O / (dt * NOAHMP_HFUS))
    canliq_freeze = torch.clamp(canliq - qfrzc * dt, min=0.0)
    canice_freeze = torch.clamp(canice + qfrzc * dt, min=0.0)
    canliq = torch.where(freezes, canliq_freeze, canliq)
    canice = torch.where(freezes, canice_freeze, canice)
    cmc = canliq + canice
    ecan = qevac + qsubc - qdewc - qfroc
    tskin = torch.where(melts | freezes, fwet * NOAHMP_TFRZ + (1.0 - fwet) * tskin, tskin)
    return canliq, canice, cmc, ecan, etran, fwet, tskin


def _combo_snow_nodes(
    dz: torch.Tensor,
    wliq: torch.Tensor,
    wice: torch.Tensor,
    temp: torch.Tensor,
    dz2: torch.Tensor,
    wliq2: torch.Tensor,
    wice2: torch.Tensor,
    temp2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized Noah-MP ``combo`` enthalpy merge for two snow nodes."""

    dzc = dz + dz2
    wicec = wice + wice2
    wliqc = wliq + wliq2
    enthalpy1 = (NOAHMP_CICE * wice + NOAHMP_CWAT * wliq) * (temp - NOAHMP_TFRZ) + NOAHMP_HFUS * wliq
    enthalpy2 = (NOAHMP_CICE * wice2 + NOAHMP_CWAT * wliq2) * (temp2 - NOAHMP_TFRZ) + NOAHMP_HFUS * wliq2
    enthalpy = enthalpy1 + enthalpy2
    heat_capacity = torch.maximum(
        NOAHMP_CICE * wicec + NOAHMP_CWAT * wliqc,
        enthalpy * 0.0 + 1.0e-30,
    )
    # Float64 quotients rounded once (no-op in float64): float32 (a/b)/b overflows at the 1e-30 floor.
    temp_cold = NOAHMP_TFRZ + (enthalpy.double() / heat_capacity.double()).to(enthalpy.dtype)
    temp_warm = NOAHMP_TFRZ + ((enthalpy - NOAHMP_HFUS * wliqc).double() / heat_capacity.double()).to(enthalpy.dtype)
    tempc = torch.where(
        enthalpy < 0.0,
        temp_cold,
        torch.where(enthalpy <= NOAHMP_HFUS * wliqc, NOAHMP_TFRZ, temp_warm),
    )
    return dzc, wliqc, wicec, tempc


def _snow_slots_from_fortran_order(
    isnow: torch.Tensor,
    dzsnso: torch.Tensor,
    snice: torch.Tensor,
    snliq: torch.Tensor,
    stc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return active snow nodes in top-to-bottom slots, independent of ``isnow``."""

    msno = torch.clamp(-isnow, min=0, max=3)
    slots = torch.arange(3, device=isnow.device, dtype=torch.int64)
    source = torch.clamp((3 - msno)[:, None] + slots[None, :], min=0, max=2)
    active = slots[None, :] < msno[:, None]

    def gather_active(values: torch.Tensor) -> torch.Tensor:
        gathered = torch.gather(values[:, :3], 1, source)
        return gathered * active.to(gathered.dtype)

    return (
        gather_active(dzsnso),
        gather_active(snice),
        gather_active(snliq),
        gather_active(stc),
    )


def _write_snow_slots_to_fortran_order(
    isnow: torch.Tensor,
    dz_slots: torch.Tensor,
    ice_slots: torch.Tensor,
    liq_slots: torch.Tensor,
    temp_slots: torch.Tensor,
    dzsnso: torch.Tensor,
    snice: torch.Tensor,
    snliq: torch.Tensor,
    stc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write top-to-bottom snow slots back to Fortran-indexed arrays."""

    msno = torch.clamp(-isnow, min=0, max=3)
    targets = torch.arange(3, device=isnow.device, dtype=torch.int64)
    slot_index = targets[None, :] - (3 - msno)[:, None]
    active = (slot_index >= 0) & (slot_index < msno[:, None])
    slot_index = torch.clamp(slot_index, min=0, max=2)

    def write_slots(values: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        first_three = torch.where(active, torch.gather(slots, 1, slot_index), values[:, :3])
        if values.shape[1] == 3:
            return first_three
        return torch.column_stack(
            (
                first_three[:, 0],
                first_three[:, 1],
                first_three[:, 2],
                values[:, 3],
                values[:, 4],
                values[:, 5],
                values[:, 6],
            )
        )

    return (
        write_slots(dzsnso, dz_slots),
        write_slots(snice, ice_slots),
        write_slots(snliq, liq_slots),
        write_slots(stc, temp_slots),
    )


def _divide_snow_source_corridor(
    isnow: torch.Tensor,
    dzsnso: torch.Tensor,
    snice: torch.Tensor,
    snliq: torch.Tensor,
    stc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port Noah-MP ``divide`` for the sealed one/two-layer land snow corridor."""

    msno = torch.clamp(-isnow, min=0, max=3)
    dz, ice, liq, temp = _snow_slots_from_fortran_order(isnow, dzsnso, snice, snliq, stc)
    dz0, dz1, dz2 = dz[:, 0], dz[:, 1], dz[:, 2]
    ice0, ice1, ice2 = ice[:, 0], ice[:, 1], ice[:, 2]
    liq0, liq1, liq2 = liq[:, 0], liq[:, 1], liq[:, 2]
    temp0, temp1, temp2 = temp[:, 0], temp[:, 1], temp[:, 2]

    split_one = (msno == 1) & (dz0 > 0.05)
    dz_half = dz0 * 0.5
    ice_half = ice0 * 0.5
    liq_half = liq0 * 0.5
    dz0 = torch.where(split_one, dz_half, dz0)
    ice0 = torch.where(split_one, ice_half, ice0)
    liq0 = torch.where(split_one, liq_half, liq0)
    dz1 = torch.where(split_one, dz_half, dz1)
    ice1 = torch.where(split_one, ice_half, ice1)
    liq1 = torch.where(split_one, liq_half, liq1)
    temp1 = torch.where(split_one, temp0, temp1)
    msno = torch.where(split_one, 2, msno)

    trim_top = (msno > 1) & (dz0 > 0.05)
    drr = dz0 - 0.05
    # Float64 quotient rounded once (no-op in float64): float32 (a/b)/b overflows at the 1e-30 floor.
    propor_rem = (drr.double() / torch.clamp(dz0, min=1.0e-30).double()).to(drr.dtype)
    rem_ice = propor_rem * ice0
    rem_liq = propor_rem * liq0
    propor_top = torch.div(0.05, torch.clamp(dz0, min=1.0e-30).double()).to(dz0.dtype)
    top_ice = propor_top * ice0
    top_liq = propor_top * liq0
    merged_dz, merged_liq, merged_ice, merged_temp = _combo_snow_nodes(
        dz1,
        liq1,
        ice1,
        temp1,
        drr,
        rem_liq,
        rem_ice,
        temp0,
    )
    dz0 = torch.where(trim_top, 0.05, dz0)
    ice0 = torch.where(trim_top, top_ice, ice0)
    liq0 = torch.where(trim_top, top_liq, liq0)
    dz1 = torch.where(trim_top, merged_dz, dz1)
    ice1 = torch.where(trim_top, merged_ice, ice1)
    liq1 = torch.where(trim_top, merged_liq, liq1)
    temp1 = torch.where(trim_top, merged_temp, temp1)

    split_second = (msno <= 2) & (msno > 1) & (dz1 > 0.20)
    # Float64 quotient rounded once (no-op in float64): float32 a/b**2 underflows at the 1e-30 floor in backward.
    dtdz = ((temp0 - temp1).double() / torch.clamp((dz0 + dz1) * 0.5, min=1.0e-30).double()).to(temp0.dtype)
    dz2_half = dz1 * 0.5
    ice2_half = ice1 * 0.5
    liq2_half = liq1 * 0.5
    temp3_candidate = temp1 - dtdz * dz2_half * 0.5
    temp2_candidate = torch.where(temp3_candidate >= NOAHMP_TFRZ, temp1, temp1 + dtdz * dz2_half * 0.5)
    temp3_candidate = torch.where(temp3_candidate >= NOAHMP_TFRZ, temp1, temp3_candidate)
    dz1 = torch.where(split_second, dz2_half, dz1)
    ice1 = torch.where(split_second, ice2_half, ice1)
    liq1 = torch.where(split_second, liq2_half, liq1)
    dz2 = torch.where(split_second, dz2_half, dz2)
    ice2 = torch.where(split_second, ice2_half, ice2)
    liq2 = torch.where(split_second, liq2_half, liq2)
    temp1 = torch.where(split_second, temp2_candidate, temp1)
    temp2 = torch.where(split_second, temp3_candidate, temp2)
    msno = torch.where(split_second, 3, msno)

    trim_second = (msno > 2) & (dz1 > 0.20)
    drr2 = dz1 - 0.20
    prop_rem2 = (drr2.double() / torch.clamp(dz1, min=1.0e-30).double()).to(drr2.dtype)
    rem_ice2 = prop_rem2 * ice1
    rem_liq2 = prop_rem2 * liq1
    prop_keep2 = torch.div(0.20, torch.clamp(dz1, min=1.0e-30).double()).to(dz1.dtype)
    keep_ice2 = prop_keep2 * ice1
    keep_liq2 = prop_keep2 * liq1
    merged_dz3, merged_liq3, merged_ice3, merged_temp3 = _combo_snow_nodes(
        dz2,
        liq2,
        ice2,
        temp2,
        drr2,
        rem_liq2,
        rem_ice2,
        temp1,
    )
    dz1 = torch.where(trim_second, 0.20, dz1)
    ice1 = torch.where(trim_second, keep_ice2, ice1)
    liq1 = torch.where(trim_second, keep_liq2, liq1)
    dz2 = torch.where(trim_second, merged_dz3, dz2)
    ice2 = torch.where(trim_second, merged_ice3, ice2)
    liq2 = torch.where(trim_second, merged_liq3, liq2)
    temp2 = torch.where(trim_second, merged_temp3, temp2)

    isnow_out = torch.where(msno > 0, -msno, isnow)
    isnow_out = isnow_out if isnow_out.dtype == torch.int64 else isnow_out.to(torch.int64)
    dz = _columns_to_matrix((dz0, dz1, dz2))
    ice = _columns_to_matrix((ice0, ice1, ice2))
    liq = _columns_to_matrix((liq0, liq1, liq2))
    temp = _columns_to_matrix((temp0, temp1, temp2))
    return (isnow_out,) + _write_snow_slots_to_fortran_order(isnow_out, dz, ice, liq, temp, dzsnso, snice, snliq, stc)


_divide_snow_source_corridor_impl = _divide_snow_source_corridor


def _divide_snow_source_corridor(
    isnow: torch.Tensor,
    dzsnso: torch.Tensor,
    snice: torch.Tensor,
    snliq: torch.Tensor,
    stc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return run_compile_island(
        "land_snow_divide",
        "divide_snow_source_corridor",
        _divide_snow_source_corridor_impl,
        isnow,
        dzsnso,
        snice,
        snliq,
        stc,
    )


def _snowwater_source_corridor_flat_impl(
    *,
    snow_levels: torch.Tensor,
    dzsnso: torch.Tensor,
    interface_depth: torch.Tensor,
    snow_depth_rate_ground: torch.Tensor,
    temperature_forcing: torch.Tensor,
    snow_ice_frac_old: torch.Tensor,
    ssi: torch.Tensor,
    snow_ret_fac: torch.Tensor,
    imelt: torch.Tensor,
    stc: torch.Tensor,
    snice: torch.Tensor,
    snliq: torch.Tensor,
    smc: torch.Tensor,
    sh2o: torch.Tensor,
    sneqv: torch.Tensor,
    snowh: torch.Tensor,
    qsnfro: torch.Tensor,
    qsnsub: torch.Tensor,
    qrain: torch.Tensor,
    qsnow: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Port Noah-MP ``snowwater`` with fixed, batched snow slots."""

    isnow = snow_levels
    zsnso = interface_depth
    dz_work = dzsnso
    z_work = zsnso
    stc_work = stc
    snice_work = snice
    snliq_work = snliq
    sh2o_work = sh2o
    sice_work = _positive_soil_ice(smc - sh2o)
    zero = snowh * 0.0
    ponding1 = zero
    ponding2 = zero
    snoflow = zero
    qsnbot = zero

    shallow_snowfall = (isnow == 0) & (qsnow > 0.0)
    snowh = torch.where(shallow_snowfall, snowh + snow_depth_rate_ground * dt, snowh)
    sneqv = torch.where(shallow_snowfall, sneqv + qsnow * dt, sneqv)
    newnode = shallow_snowfall & (snowh >= 0.025)
    isnow = torch.where(newnode, -1, isnow)
    dz_work = _replace_matrix_column(dz_work, 2, torch.where(newnode, snowh, dz_work[:, 2]))
    stc_work = _replace_matrix_column(stc_work, 2, torch.where(newnode, torch.clamp(temperature_forcing, max=NOAHMP_TFRZ), stc_work[:, 2]))
    snice_work = _replace_matrix_column(snice_work, 2, torch.where(newnode, sneqv, snice_work[:, 2]))
    snliq_work = _replace_matrix_column(snliq_work, 2, torch.where(newnode, 0.0, snliq_work[:, 2]))
    snowh = torch.where(newnode, 0.0, snowh)

    fidx3 = torch.arange(-2, 1, device=isnow.device, dtype=isnow.dtype)
    snowfall_slot = (isnow + 1)[:, None] == fidx3[None, :]
    add_to_layer = ((isnow < 0) & (~newnode) & (qsnow > 0.0))[:, None] & snowfall_slot
    snice_work = snice_work + (qsnow * dt)[:, None] * add_to_layer
    dz_work = _replace_first_three_columns(
        dz_work,
        dz_work[:, :3] + (snow_depth_rate_ground * dt)[:, None] * add_to_layer,
    )

    active_snow = isnow < 0
    active3_predivide = active_snow[:, None] & (fidx3[None, :] >= (isnow + 1)[:, None])
    wx = snice_work + snliq_work
    wx_active = wx * active3_predivide
    burden = torch.cumsum(wx_active, dim=1) - wx_active
    dz_snow = dz_work[:, :3]
    stc_snow = stc_work[:, :3]
    # Float64 quotient rounded once (no-op in float64): float32 a/b**2 underflows at the 1e-30 floor in backward.
    fice = (snice_work.double() / torch.clamp(wx, min=1.0e-30).double()).to(snice_work.dtype)
    void = 1.0 - ((snice_work / NOAHMP_DENICE + snliq_work / NOAHMP_DENH2O).double()
                  / torch.clamp(dz_snow, min=1.0e-30).double()).to(snice_work.dtype)
    compact = active3_predivide & (void > 0.001) & (snice_work > 0.1)
    bi = (snice_work.double() / torch.clamp(dz_snow, min=1.0e-30).double()).to(snice_work.dtype)
    td = torch.clamp(NOAHMP_TFRZ - stc_snow, min=0.0)
    ddz1 = -2.5e-6 * torch.exp(-0.04 * td)
    ddz1 = torch.where(bi > 100.0, ddz1 * torch.exp(-46.0e-3 * (bi - 100.0)), ddz1)
    ddz1 = ddz1 * (1.0 + (snliq_work > 0.01 * dz_snow))
    ddz2 = -(burden + 0.5 * wx) * torch.exp(-0.08 * td - 21.0e-3 * bi) / 1.8e6
    ficeold = snow_ice_frac_old[:, :3]
    ddz3 = torch.clamp((ficeold - fice) / torch.clamp(ficeold, min=1.0e-6), min=0.0)
    ddz3 = (-ddz3 / dt[:, None]) * (imelt[:, :3] == 1.0)
    pdzdtc = torch.clamp((ddz1 + ddz2 + ddz3) * dt[:, None], min=-0.5)
    dz_new = dz_snow * (1.0 + pdzdtc)
    dz_new = torch.minimum(torch.maximum(dz_new, wx / 500.0), wx / 50.0)
    dz_work = _replace_first_three_columns(dz_work, dz_snow + (dz_new - dz_snow) * compact)

    (stc_work, dz_work, snice_work, snliq_work, isnow, snowh, sneqv,
     sh2o_work, sice_work, ponding1, ponding2) = combine_snow_layers(
        stc=stc_work, dzsnso=dz_work, snice=snice_work, snliq=snliq_work,
        snow_levels=isnow, snowh=snowh, sneqv=sneqv,
        sh2o=sh2o_work, sice=sice_work, ponding1=ponding1, ponding2=ponding2,
        accumulate_ponding=False,
    )

    isnow, dzsnso, snice_work, snliq_work, stc_work = _divide_snow_source_corridor_impl(
        isnow,
        dz_work,
        snice_work,
        snliq_work,
        stc_work,
    )

    sneqv_zero = sneqv == 0.0
    first_soil_dz = torch.clamp(dzsnso[:, 3], min=1.0e-30)
    sice_work = _replace_matrix_column(sice_work, 0, torch.where(sneqv_zero, sice_work[:, 0] + (qsnfro - qsnsub) * dt / (first_soil_dz * 1000.0), sice_work[:, 0]))
    sice_negative = sice_work[:, 0] < 0.0
    sh2o_work = _replace_matrix_column(sh2o_work, 0, torch.where(sice_negative, sh2o_work[:, 0] + sice_work[:, 0], sh2o_work[:, 0]))
    sice_work = _replace_matrix_column(sice_work, 0, torch.where(sice_negative, 0.0, sice_work[:, 0]))

    shallow_no_layer = (isnow == 0) & (sneqv > 0.0)
    temp_mass = torch.clamp(sneqv, min=1.0e-30)
    sneqv_no_layer = sneqv - qsnsub * dt + qsnfro * dt
    propor = sneqv_no_layer / temp_mass
    snowh_no_layer = torch.clamp(propor * snowh, min=0.0)
    snowh_no_layer = torch.minimum(torch.maximum(snowh_no_layer, sneqv_no_layer / 500.0), sneqv_no_layer / 50.0)
    snow_negative = sneqv_no_layer < 0.0
    sice_no_layer = sice_work[:, 0] + sneqv_no_layer / (first_soil_dz * 1000.0)
    sice_work = _replace_matrix_column(sice_work, 0, torch.where(shallow_no_layer & snow_negative, sice_no_layer, sice_work[:, 0]))
    sneqv_no_layer = torch.where(snow_negative, 0.0, sneqv_no_layer)
    snowh_no_layer = torch.where(snow_negative, 0.0, snowh_no_layer)
    sneqv = torch.where(shallow_no_layer, sneqv_no_layer, sneqv)
    snowh = torch.where(shallow_no_layer, snowh_no_layer, snowh)
    sice_negative = sice_work[:, 0] < 0.0
    sh2o_work = _replace_matrix_column(sh2o_work, 0, torch.where(sice_negative, sh2o_work[:, 0] + sice_work[:, 0], sh2o_work[:, 0]))
    sice_work = _replace_matrix_column(sice_work, 0, torch.where(sice_negative, 0.0, sice_work[:, 0]))

    no_snowpack = (snowh <= 1.0e-6) | (sneqv <= 1.0e-3)
    snowh = torch.where(no_snowpack, 0.0, snowh)
    sneqv = torch.where(no_snowpack, 0.0, sneqv)

    deep_snow = isnow < 0
    active3 = deep_snow[:, None] & (fidx3[None, :] >= (isnow + 1)[:, None])
    top3 = deep_snow[:, None] & ((isnow + 1)[:, None] == fidx3[None, :])
    wgdif3 = snice_work - (qsnsub * dt)[:, None] + (qsnfro * dt)[:, None]
    snice_work = snice_work + (wgdif3 - snice_work) * top3
    # Sublimation can remove the top layer; rain uses the surviving top slot.
    combine_after_sublimation = (top3 & (wgdif3 < 1.0e-6)).any(dim=1)
    (stc_work, dzsnso, snice_work, snliq_work, isnow, snowh, sneqv,
     sh2o_work, sice_work, ponding1, ponding2) = combine_snow_layers(
        stc=stc_work, dzsnso=dzsnso, snice=snice_work, snliq=snliq_work,
        snow_levels=isnow, snowh=snowh, sneqv=sneqv,
        sh2o=sh2o_work, sice=sice_work, ponding1=ponding1, ponding2=ponding2,
        accumulate_ponding=False, activity=combine_after_sublimation,
    )
    deep_snow = isnow < 0
    active3 = deep_snow[:, None] & (fidx3[None, :] >= (isnow + 1)[:, None])
    top3 = deep_snow[:, None] & ((isnow + 1)[:, None] == fidx3[None, :])
    snliq_candidate = torch.clamp(snliq_work + (qrain * dt)[:, None], min=0.0)
    snliq_work = snliq_work + (snliq_candidate - snliq_work) * top3

    qin = zero
    qout = zero
    active = active3[:, 0]
    dz = torch.clamp(dzsnso[:, 0], min=1.0e-30)
    vol_ice = torch.clamp(snice_work[:, 0] / (dz * NOAHMP_DENICE), max=1.0)
    epore = 1.0 - vol_ice
    snliq_added = snliq_work[:, 0] + qin
    vol_liq = snliq_added / (dz * NOAHMP_DENH2O)
    qout_candidate = torch.clamp((vol_liq - ssi * epore) * dz, min=0.0) * NOAHMP_DENH2O
    snliq_after = snliq_added - qout_candidate
    max_fraction = 0.4 / (1.0 - 0.4) * snice_work[:, 0]
    too_liquid = snliq_after / torch.clamp(snice_work[:, 0] + snliq_after, min=1.0e-30) > 0.4
    qout_next = qout_candidate + (snliq_after - max_fraction) * too_liquid
    snliq_next = snliq_after + (max_fraction - snliq_after) * too_liquid
    snliq0 = snliq_work[:, 0] + (snliq_next - snliq_work[:, 0]) * active
    qout = torch.where(active, qout_next, qout)
    qin = torch.where(active, qout_next, qin)

    active = active3[:, 1]
    dz = torch.clamp(dzsnso[:, 1], min=1.0e-30)
    vol_ice = torch.clamp(snice_work[:, 1] / (dz * NOAHMP_DENICE), max=1.0)
    epore = 1.0 - vol_ice
    snliq_added = snliq_work[:, 1] + qin
    vol_liq = snliq_added / (dz * NOAHMP_DENH2O)
    qout_candidate = torch.clamp((vol_liq - ssi * epore) * dz, min=0.0) * NOAHMP_DENH2O
    snliq_after = snliq_added - qout_candidate
    max_fraction = 0.4 / (1.0 - 0.4) * snice_work[:, 1]
    too_liquid = snliq_after / torch.clamp(snice_work[:, 1] + snliq_after, min=1.0e-30) > 0.4
    qout_next = qout_candidate + (snliq_after - max_fraction) * too_liquid
    snliq_next = snliq_after + (max_fraction - snliq_after) * too_liquid
    snliq1 = snliq_work[:, 1] + (snliq_next - snliq_work[:, 1]) * active
    qout = torch.where(active, qout_next, qout)
    qin = torch.where(active, qout_next, qin)

    active = active3[:, 2]
    dz = torch.clamp(dzsnso[:, 2], min=1.0e-30)
    vol_ice = torch.clamp(snice_work[:, 2] / (dz * NOAHMP_DENICE), max=1.0)
    epore = 1.0 - vol_ice
    snliq_added = snliq_work[:, 2] + qin
    vol_liq = snliq_added / (dz * NOAHMP_DENH2O)
    qout_candidate = torch.maximum(
        (vol_liq - epore) * dz,
        snow_ret_fac * dt * torch.clamp((vol_liq - ssi * epore) * dz, min=0.0),
    ) * NOAHMP_DENH2O
    snliq_after = snliq_added - qout_candidate
    max_fraction = 0.4 / (1.0 - 0.4) * snice_work[:, 2]
    too_liquid = snliq_after / torch.clamp(snice_work[:, 2] + snliq_after, min=1.0e-30) > 0.4
    qout_next = qout_candidate + (snliq_after - max_fraction) * too_liquid
    snliq_next = snliq_after + (max_fraction - snliq_after) * too_liquid
    snliq2 = snliq_work[:, 2] + (snliq_next - snliq_work[:, 2]) * active
    qout = torch.where(active, qout_next, qout)
    qin = torch.where(active, qout_next, qin)
    _ = qin
    snliq_work = torch.column_stack((snliq0, snliq1, snliq2))

    mass3 = snliq_work + snice_work
    dz_limited3 = torch.minimum(torch.maximum(dzsnso[:, :3], mass3 / 500.0), mass3 / 50.0)
    dzsnso = _replace_first_three_columns(dzsnso, dzsnso[:, :3] + (dz_limited3 - dzsnso[:, :3]) * active3)

    qsnbot = torch.where(deep_snow, qout / dt, qsnbot)

    empty3 = fidx3[None, :] <= isnow[:, None]
    active_not_empty3 = ~empty3
    snice_work = snice_work * active_not_empty3
    snliq_work = snliq_work * active_not_empty3
    stc_work = _replace_first_three_columns(stc_work, stc_work[:, :3] * active_not_empty3)
    dzsnso = _replace_first_three_columns(dzsnso, dzsnso[:, :3] * active_not_empty3)
    z_work = _replace_first_three_columns(z_work, z_work[:, :3] * active_not_empty3)

    too_much_snow = sneqv > 5000.0
    # Float64 quotient rounded once (no-op in float64): float32 (a/b)/b overflows at the 1e-30 floor.
    bdsnow = (snice_work[:, 2].double() / torch.clamp(dzsnso[:, 2], min=1.0e-30).double()).to(snice_work.dtype)
    snoflow_mass = sneqv - 5000.0
    snice_work = _replace_matrix_column(snice_work, 2, torch.where(too_much_snow, snice_work[:, 2] - snoflow_mass, snice_work[:, 2]))
    bdsnow_safe = torch.clamp(torch.abs(bdsnow), min=1.0e-30)
    overflow_depth = (snoflow_mass.double() / bdsnow_safe.double()).to(snoflow_mass.dtype)
    dzsnso = _replace_matrix_column(dzsnso, 2, torch.where(too_much_snow, dzsnso[:, 2] - overflow_depth, dzsnso[:, 2]))
    snoflow = torch.where(too_much_snow, snoflow_mass / dt, snoflow)

    fidx3 = torch.arange(-2, 1, device=isnow.device, dtype=isnow.dtype)
    active3 = (isnow[:, None] < 0) & (fidx3[None, :] >= (isnow + 1)[:, None])
    snow_depth = dzsnso[:, :3] * active3
    active_mass = torch.zeros_like(sneqv)
    for row in range(3):
        active_mass = active_mass + torch.where(active3[:, row], snice_work[:, row], 0.0)
        active_mass = active_mass + torch.where(active3[:, row], snliq_work[:, row], 0.0)
    active_depth = snow_depth[:, 0] + snow_depth[:, 1] + snow_depth[:, 2]
    sneqv = torch.where(isnow < 0, active_mass, sneqv)
    snowh = torch.where(isnow < 0, active_depth, snowh)

    soil_depths = dzsnso[:, 3:7] * 0.0 - static_tensor(DZSOIL_4L, dtype=dzsnso.dtype, device=dzsnso.device)
    dz_reset_snow = torch.where(active3, -dzsnso[:, :3], dzsnso[:, :3])
    dz_reset = torch.column_stack(
        (
            dz_reset_snow[:, 0],
            dz_reset_snow[:, 1],
            dz_reset_snow[:, 2],
            soil_depths[:, 0],
            soil_depths[:, 1],
            soil_depths[:, 2],
            soil_depths[:, 3],
        )
    )
    fidx7 = torch.arange(-2, 5, device=isnow.device, dtype=isnow.dtype)
    active7 = fidx7[None, :] >= (isnow + 1)[:, None]
    zsnso_reset = torch.where(active7, torch.cumsum(dz_reset, dim=1), z_work)
    dzsnso_out = torch.where(active7, -dz_reset, dz_reset)

    smc_work = sh2o_work + sice_work
    _ = snoflow
    return (
        isnow,
        snowh,
        sneqv,
        snice_work,
        snliq_work,
        stc_work,
        smc_work,
        sh2o_work,
        sice_work,
        zsnso_reset,
        dzsnso_out,
        qsnbot,
        ponding1,
        ponding2,
    )


_SNOWWATER_RESULT_FIELDS = (
    "snow_levels",
    "snowh",
    "sneqv",
    "snice",
    "snliq",
    "stc",
    "smc",
    "sh2o",
    "sice",
    "zsnso",
    "dzsnso",
    "qsnbot",
    "ponding1",
    "ponding2",
)


def _snowwater_source_corridor(
    state: LandBodyState,
    energy_tables: Any,
    *,
    imelt: torch.Tensor,
    stc: torch.Tensor,
    snice: torch.Tensor,
    snliq: torch.Tensor,
    smc: torch.Tensor,
    sh2o: torch.Tensor,
    sneqv: torch.Tensor,
    snowh: torch.Tensor,
    qsnfro: torch.Tensor,
    qsnsub: torch.Tensor,
    qrain: torch.Tensor,
    qsnow: torch.Tensor,
    dt: torch.Tensor,
) -> dict[str, torch.Tensor]:
    snow_levels = state.snow_levels if state.snow_levels.dtype == torch.int64 else state.snow_levels.to(torch.int64)
    snowwater = run_compile_island(
        "land_snowwater",
        "snowwater_source_corridor",
        _snowwater_source_corridor_flat_impl,
        snow_levels=snow_levels,
        dzsnso=state.dzsnso,
        interface_depth=state.interface_depth,
        snow_depth_rate_ground=state.snow_depth_rate_ground,
        temperature_forcing=state.temperature_forcing,
        snow_ice_frac_old=state.snow_ice_frac_old,
        ssi=energy_tables.ssi,
        snow_ret_fac=energy_tables.snow_ret_fac,
        imelt=imelt,
        stc=stc,
        snice=snice,
        snliq=snliq,
        smc=smc,
        sh2o=sh2o,
        sneqv=sneqv,
        snowh=snowh,
        qsnfro=qsnfro,
        qsnsub=qsnsub,
        qrain=qrain,
        qsnow=qsnow,
        dt=dt,
    )
    return dict(zip(_SNOWWATER_RESULT_FIELDS, snowwater, strict=True))


def _water_source_corridor(
    state: LandBodyState,
    energy_tables: Any,
    fgev: torch.Tensor,
    fcev: torch.Tensor,
    fctr: torch.Tensor,
    latheag: torch.Tensor,
    frozen_canopy: torch.Tensor,
    dt: torch.Tensor,
    bare_only: torch.Tensor,
    active_canopy_no_snow: torch.Tensor,
    active_canopy_source: torch.Tensor,
    btrani: torch.Tensor,
    phase_stc: torch.Tensor,
    phase_snice: torch.Tensor,
    phase_snliq: torch.Tensor,
    phase_smc: torch.Tensor,
    phase_sh2o: torch.Tensor,
    phase_sneqv: torch.Tensor,
    phase_snowh: torch.Tensor,
    phase_imelt: torch.Tensor,
    phase_ponding: torch.Tensor,
    soilwater_iteration_count: int,
    temperature_leaf: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    zsnso_no_snow = static_tensor((0.0, 0.0, 0.0, -0.1, -0.4, -1.0, -2.0), dtype=state.interface_depth.dtype, device=state.interface_depth.device).expand_as(state.interface_depth)
    dzsnso_no_snow = static_tensor((0.0, 0.0, 0.0, *DZSOIL_4L), dtype=state.dzsnso.dtype, device=state.dzsnso.device).expand_as(state.dzsnso)
    (
        qvap,
        qdew,
        edir,
        latheav,
        qsnsub,
        qsnfro,
        qsdew,
        qseva,
        esnow,
        zsnso,
        dzsnso,
    ) = run_compile_island(
        "land_water_merge",
        "water_pre_snow",
        _water_pre_snow_flat_impl,
        fgev=fgev,
        latheag=latheag,
        frozen_canopy=frozen_canopy,
        snow_water_equiv=phase_sneqv,
        snow_levels=state.snow_levels,
        dt=dt,
        interface_depth=state.interface_depth,
        dzsnso=state.dzsnso,
        zsnso_no_snow_template=zsnso_no_snow,
        dzsnso_no_snow_template=dzsnso_no_snow,
    )
    canliq, canice, cmc, ecan, etran, _fwet, temperature_leaf = _canwater_source_corridor(state, energy_tables, fcev, fctr, latheav, frozen_canopy, dt, temperature_leaf)
    no_snow = state.snow_levels == 0
    snow_active = ~no_snow
    snowwater = _snowwater_source_corridor(
        state,
        energy_tables,
        imelt=phase_imelt,
        stc=phase_stc,
        snice=phase_snice,
        snliq=phase_snliq,
        smc=phase_smc,
        sh2o=phase_sh2o,
        sneqv=phase_sneqv,
        snowh=phase_snowh,
        qsnfro=qsnfro,
        qsnsub=qsnsub,
        qrain=state.qrain_ground,
        qsnow=state.qsnow_ground,
        dt=dt,
    )
    snow_smc = snowwater["smc"]
    snow_sh2o = snowwater["sh2o"]
    snow_sice = snowwater["sice"]
    snow_zsnso = snowwater["zsnso"]
    snow_dzsnso = snowwater["dzsnso"]
    smc, sh2o, sice, zsnso, dzsnso, qinsur, qseva_input = run_compile_island(
        "land_water_merge",
        "water_snow_to_soil",
        _water_snow_to_soil_flat_impl,
        phase_smc=phase_smc,
        phase_sh2o=phase_sh2o,
        zsnso=zsnso,
        dzsnso=dzsnso,
        snow_levels=state.snow_levels,
        latheag=latheag,
        qvap=qvap,
        qsdew=qsdew,
        qseva=qseva,
        phase_ponding=phase_ponding,
        qrain_ground=state.qrain_ground,
        snowwater_smc=snow_smc,
        snowwater_sh2o=snow_sh2o,
        snowwater_sice=snow_sice,
        snowwater_zsnso=snow_zsnso,
        snowwater_dzsnso=snow_dzsnso,
        snowwater_ponding1=snowwater["ponding1"],
        snowwater_ponding2=snowwater["ponding2"],
        snowwater_qsnbot=snowwater["qsnbot"],
        snowwater_snow_levels=snowwater["snow_levels"],
        dt=dt,
    )
    etrani = etran[:, None] * btrani * 0.001
    water = soilwater_opt_run1_inf1_source_corridor(
        state,
        energy_tables,
        qinsur=qinsur,
        qseva=qseva_input * 0.001,
        etrani=etrani,
        dt=dt,
        smc_initial=smc,
        sh2o_initial=sh2o,
        sice_initial=sice,
        dzsnso_soil_initial=dzsnso[:, 3:7],
        iteration_count=soilwater_iteration_count,
        wa_initial=4900.0 if state.aquifer_water is None else state.aquifer_water,
        zwt_initial=2.0 if state.depth_water_table is None else state.depth_water_table,
        wt_initial=500.0 if state.saturated_water is None else state.saturated_water,
    )
    active_source_water = bare_only | active_canopy_source
    i_values = run_compile_island(
        "land_water_merge",
        "water_final_merge",
        _water_final_merge_flat_impl,
        active_source_water=active_source_water,
        snow_active=snow_active,
        qvap=qvap,
        qdew=qdew,
        edir=edir,
        ecan=ecan,
        etran=etran,
        esnow=esnow,
        smc_before_soilwater=smc,
        sh2o_before_soilwater=sh2o,
        zsnso=zsnso,
        dzsnso=dzsnso,
        phase_stc=phase_stc,
        phase_snowh=phase_snowh,
        phase_sneqv=phase_sneqv,
        snow_levels=state.snow_levels,
        snowwater_qsnbot=snowwater["qsnbot"],
        snowwater_stc=snowwater["stc"],
        snowwater_snowh=snowwater["snowh"],
        snowwater_sneqv=snowwater["sneqv"],
        snowwater_snow_levels=snowwater["snow_levels"],
        soilwater_smc=water.smc,
        soilwater_sh2o=water.sh2o,
        soilwater_runsrf=water.runsrf,
        soilwater_runsub=water.runsub,
        soilwater_wa=water.wa,
        canliq=canliq,
        canice=canice,
        cmc=cmc,
    )
    i = dict(zip(_I_RESULT_FIELDS, i_values, strict=True))
    i.update(snice=snowwater["snice"], snliq=snowwater["snliq"],
             wt=water.wt, zwt=water.zwt, fwet=_fwet, tv=temperature_leaf)
    return (
        i,
        qvap,
        qdew,
        edir,
    )


def _phasechange_source_corridor_flat_impl(
    *,
    snow_water_equiv: torch.Tensor,
    snow_depth: torch.Tensor,
    dzsnso: torch.Tensor,
    soiltyp: torch.Tensor,
    snow_levels: torch.Tensor,
    psisat: torch.Tensor,
    smcmax_by_column: torch.Tensor,
    bexp_table: torch.Tensor,
    fact_matrix: torch.Tensor,
    stc: torch.Tensor,
    snice: torch.Tensor,
    snliq: torch.Tensor,
    smc: torch.Tensor,
    sh2o: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port Noah-MP ``phasechange`` for the sealed ``opt_frz=1`` corridor."""

    sneqv = snow_water_equiv
    snowh = snow_depth
    dt_col = dt[:, None] if dt.ndim > 0 else dt
    zero7 = stc * 0.0
    zero4 = smc * 0.0
    fidx = torch.arange(-2, 5, device=stc.device, dtype=snow_levels.dtype)
    active = fidx[None, :] >= (snow_levels + 1)[:, None]

    soil_dz = dzsnso[:, 3:7]
    soil_idx = soiltyp - 1
    cold_soil = stc[:, 3:7] < NOAHMP_TFRZ
    smp = NOAHMP_HFUS * (NOAHMP_TFRZ - stc[:, 3:7]) / (NOAHMP_GRAV * stc[:, 3:7])
    psisat_soil = psisat[soil_idx][:, None]
    smcmax_soil = smcmax_by_column[:, None]
    bexp_soil = bexp_table[soil_idx][:, None]
    power_base = smp / psisat_soil
    power_base = torch.where(cold_soil & (power_base > 0.0), power_base, 1.0)
    supercooled = smcmax_soil * working_power(power_base, -1.0 / bexp_soil)

    soil_mliq = sh2o * soil_dz * 1000.0
    soil_mice = (smc - sh2o) * soil_dz * 1000.0
    soil_supercool = supercooled * soil_dz * 1000.0 * cold_soil.to(stc.dtype)
    mice = torch.column_stack(
        (
            snice[:, 0],
            snice[:, 1],
            snice[:, 2],
            soil_mice[:, 0],
            soil_mice[:, 1],
            soil_mice[:, 2],
            soil_mice[:, 3],
        )
    )
    mliq = torch.column_stack(
        (
            snliq[:, 0],
            snliq[:, 1],
            snliq[:, 2],
            soil_mliq[:, 0],
            soil_mliq[:, 1],
            soil_mliq[:, 2],
            soil_mliq[:, 3],
        )
    )
    supercool = torch.column_stack(
        (
            zero7[:, 0],
            zero7[:, 1],
            zero7[:, 2],
            soil_supercool[:, 0],
            soil_supercool[:, 1],
            soil_supercool[:, 2],
            soil_supercool[:, 3],
        )
    )

    melting = (mice > 0.0) & (stc >= NOAHMP_TFRZ)
    freezing = (mliq > supercool) & (stc < NOAHMP_TFRZ)
    thin_snow_melt = (snow_levels[:, None] == 0) & (sneqv[:, None] > 0.0) & (fidx[None, :] == 1) & (stc >= NOAHMP_TFRZ)
    melt_mask = active & (melting | thin_snow_melt)
    freeze_mask = active & freezing & (~melting) & (~thin_snow_melt)
    imelt = melt_mask.to(stc.dtype) + 2.0 * freeze_mask.to(stc.dtype)

    wice0 = mice
    wliq0 = mliq
    wmass0 = mice + mliq

    changing = active & (imelt > 0.0)
    fact = torch.where(torch.abs(fact_matrix) > 1.0e-30, fact_matrix, 1.0e-30)
    # Float64 quotient rounded once (no-op in float64): at the 1e-30 floor (no phase change) the float32 backward
    # (a/b)/b overflows and the mask multiplies it by 0 (NaN).
    hm = ((stc - NOAHMP_TFRZ).double() / fact.double()).to(stc.dtype) * changing.to(stc.dtype)
    stc_work = torch.where(changing, zero7 + NOAHMP_TFRZ, stc)
    melt_wrong_sign = (imelt == 1.0) & (hm < 0.0)
    freeze_wrong_sign = (imelt == 2.0) & (hm > 0.0)
    reset = changing & (melt_wrong_sign | freeze_wrong_sign)
    keep_after_reset = (~reset).to(stc.dtype)
    hm = hm * keep_after_reset
    imelt = imelt * keep_after_reset
    xm = hm * dt_col / NOAHMP_HFUS

    # Thin snow with no explicit snow layer.  This corridor is included for
    # completeness; the currently sealed snow cases both have explicit layers.
    thin_active = (snow_levels == 0) & (sneqv > 0.0) & (xm[:, 3] > 0.0)
    temp1 = sneqv
    sneqv_thin = torch.clamp(temp1 - xm[:, 3], min=0.0)
    # Float64 quotient rounded once (no-op in float64): float32 a/b**2 underflows at the 1e-30 floor in backward.
    propor = (sneqv_thin.double() / torch.clamp(temp1, min=1.0e-30).double()).to(sneqv_thin.dtype)
    snowh_thin = torch.clamp(propor * snowh, min=0.0)
    snowh_thin = torch.minimum(torch.maximum(snowh_thin, sneqv_thin / 500.0), sneqv_thin / 50.0)
    heatr_thin = hm[:, 3] - NOAHMP_HFUS * (temp1 - sneqv_thin) / dt
    remaining_heat = heatr_thin > 0.0
    xm_thin = torch.where(remaining_heat, heatr_thin * dt / NOAHMP_HFUS, 0.0)
    hm_thin = torch.where(remaining_heat, heatr_thin, 0.0)
    ponding = torch.where(thin_active, temp1 - sneqv_thin, 0.0)
    qmelt = torch.where(thin_active, torch.clamp(temp1 - sneqv_thin, min=0.0) / dt, 0.0)
    sneqv = torch.where(thin_active, sneqv_thin, sneqv)
    snowh = torch.where(thin_active, snowh_thin, snowh)
    xm_col3 = torch.where(thin_active, xm_thin, xm[:, 3])
    hm_col3 = torch.where(thin_active, hm_thin, hm[:, 3])
    xm = torch.column_stack((xm[:, 0], xm[:, 1], xm[:, 2], xm_col3, xm[:, 4], xm[:, 5], xm[:, 6]))
    hm = torch.column_stack((hm[:, 0], hm[:, 1], hm[:, 2], hm_col3, hm[:, 4], hm[:, 5], hm[:, 6]))

    ice_layers = []
    liquid_layers = []
    temperature_layers = []
    transferred = torch.zeros_like(active[:, 0])
    residual_heat = zero7[:, 0]
    # A depleted snow node updates the next node's HM/XM before that node changes phase.
    for layer in range(7):
        layer_hm = torch.where(transferred, hm[:, layer] + residual_heat, hm[:, layer])
        layer_xm = torch.where(transferred, layer_hm * dt / NOAHMP_HFUS, xm[:, layer])
        changing = active[:, layer] & (imelt[:, layer] > 0.0) & (torch.abs(layer_hm) > 0.0)
        ice_melt = torch.clamp(wice0[:, layer] - layer_xm, min=0.0)
        if layer < 3:
            ice_freeze = torch.minimum(wmass0[:, layer], wice0[:, layer] - layer_xm)
        else:
            ice_freeze = torch.clamp(torch.minimum(
                wmass0[:, layer] - supercool[:, layer], wice0[:, layer] - layer_xm
            ), min=0.0)
            ice_freeze = torch.where(wmass0[:, layer] < supercool[:, layer], 0.0, ice_freeze)
        ice_candidate = torch.where(
            layer_xm > 0.0, ice_melt,
            torch.where(layer_xm < 0.0, ice_freeze, mice[:, layer]),
        )
        heatr = torch.where(
            layer_xm != 0.0,
            layer_hm - NOAHMP_HFUS * (wice0[:, layer] - ice_candidate) / dt,
            0.0,
        )
        layer_ice = torch.where(changing, ice_candidate, mice[:, layer])
        layer_liquid = torch.where(
            changing, torch.clamp(wmass0[:, layer] - layer_ice, min=0.0), mliq[:, layer]
        )
        heating = changing & (torch.abs(heatr) > 0.0)
        temperature_candidate = stc_work[:, layer] + fact[:, layer] * heatr
        if layer < 3:
            temperature_candidate = torch.where(
                (layer_liquid * layer_ice > 0.0) | (layer_ice == 0.0),
                zero7[:, layer] + NOAHMP_TFRZ, temperature_candidate,
            )
            transferred = heating & (layer_ice == 0.0)
            residual_heat = heatr
            qmelt = qmelt + torch.where(changing, torch.clamp(
                wice0[:, layer] - layer_ice, min=0.0
            ) / dt, 0.0)
        else:
            transferred = torch.zeros_like(transferred)
        ice_layers.append(layer_ice)
        liquid_layers.append(layer_liquid)
        temperature_layers.append(torch.where(heating, temperature_candidate, stc_work[:, layer]))
    mice = torch.stack(ice_layers, dim=1)
    mliq = torch.stack(liquid_layers, dim=1)
    stc_work = torch.stack(temperature_layers, dim=1)

    active_snow = active[:, :3]
    snliq_out = torch.where(active_snow, mliq[:, :3], snliq)
    snice_out = torch.where(active_snow, mice[:, :3], snice)
    sh2o_out = mliq[:, 3:7] / (1000.0 * soil_dz)
    smc_out = (mliq[:, 3:7] + mice[:, 3:7]) / (1000.0 * soil_dz)

    return (
        stc_work,
        snice_out,
        snliq_out,
        smc_out,
        sh2o_out,
        sneqv,
        snowh,
        imelt,
        ponding,
    )


_tsnosoi_source_corridor_impl = tsnosoi_source_corridor


def tsnosoi_source_corridor(
    state: Any,
    tables: LandEnergyTables,
    thermo: ThermopropResult,
    *,
    ssoil: torch.Tensor,
    dt: torch.Tensor | float,
) -> torch.Tensor:
    if "land_tsnosoi" not in compile_categories_enabled():
        return _tsnosoi_source_corridor_impl(state, tables, thermo, ssoil=ssoil, dt=dt)
    return run_compile_island(
        "land_tsnosoi",
        "tsnosoi_source_corridor",
        _tsnosoi_source_corridor_flat_impl,
        temperature_snow_soil=state.temperature_snow_soil,
        interface_depth=state.interface_depth,
        df=thermo.df,
        hcpct=thermo.hcpct,
        temperature_forcing=state.temperature_forcing,
        snow_levels=state.snow_levels,
        tg3=state.tg3,
        snow_depth=state.snow_depth,
        zbot=tables.zbot,
        ssoil=ssoil,
        dt=dt,
    )


def _phasechange_source_corridor(
    state: LandBodyState,
    energy_tables: Any,
    thermo: Any,
    *,
    stc: torch.Tensor,
    snice: torch.Tensor,
    snliq: torch.Tensor,
    smc: torch.Tensor,
    sh2o: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    soiltyp = state.soiltyp if state.soiltyp.dtype == torch.int64 else state.soiltyp.to(torch.int64)
    return run_compile_island(
        "land_phasechange",
        "phasechange_source_corridor",
        _phasechange_source_corridor_flat_impl,
        snow_water_equiv=state.snow_water_equiv,
        snow_depth=state.snow_depth,
        dzsnso=state.dzsnso,
        soiltyp=soiltyp,
        snow_levels=state.snow_levels,
        psisat=energy_tables.psisat,
        smcmax_by_column=_state_cached_tensor(state, "smcmax_surface_by_column",
                                              lambda: soil_parameter_by_column(energy_tables, "smcmax", soiltyp - 1, state.vegtype)),
        bexp_table=energy_tables.bexp,
        fact_matrix=thermo.fact,
        stc=stc,
        snice=snice,
        snliq=snliq,
        smc=smc,
        sh2o=sh2o,
        dt=dt,
    )


def _bare_flux_merge_flat_impl(
    *,
    fira: torch.Tensor,
    fsh: torch.Tensor,
    fgev: torch.Tensor,
    ssoil: torch.Tensor,
    trad: torch.Tensor,
    qsfc: torch.Tensor,
    tv: torch.Tensor,
    latheag: torch.Tensor,
    geometry_emg: torch.Tensor,
    geometry_latheag: torch.Tensor,
    geometry_z0mg: torch.Tensor,
    bare_z0h: torch.Tensor,
    bare_irb: torch.Tensor,
    bare_shb: torch.Tensor,
    bare_evb: torch.Tensor,
    bare_ghb: torch.Tensor,
    bare_qsfc: torch.Tensor,
    bare_tgb: torch.Tensor,
    dlwflx: torch.Tensor,
    bare_only: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    bare_fire = dlwflx + bare_irb
    bare_trad_base = (bare_fire - (1.0 - geometry_emg) * dlwflx) / torch.clamp(geometry_emg * LAND_ENERGY_SB, min=1.0e-30)
    bare_trad = torch.pow(torch.clamp(bare_trad_base, min=1.0e-30), 0.25)
    return (
        torch.where(bare_only, bare_irb, fira),
        torch.where(bare_only, bare_shb, fsh),
        torch.where(bare_only, bare_evb, fgev),
        torch.where(bare_only, bare_ghb, ssoil),
        torch.where(bare_only, bare_trad, trad),
        torch.where(bare_only, bare_qsfc, qsfc),
        torch.where(bare_only, bare_tgb, tv),
        torch.where(bare_only, geometry_latheag, latheag),
        geometry_emg,
        geometry_z0mg,
        bare_z0h,
    )


def _canopy_flux_merge_flat_impl(
    *,
    active_canopy_source: torch.Tensor,
    fveg: torch.Tensor,
    prsl1: torch.Tensor,
    dlwflx: torch.Tensor,
    geometry_emg: torch.Tensor,
    geometry_emv: torch.Tensor,
    geometry_z0m: torch.Tensor,
    geometry_z0mg: torch.Tensor,
    hvt_by_column: torch.Tensor,
    laisun: torch.Tensor,
    laisha: torch.Tensor,
    bare_irb: torch.Tensor,
    bare_shb: torch.Tensor,
    bare_evb: torch.Tensor,
    bare_ghb: torch.Tensor,
    bare_ch: torch.Tensor,
    bare_tgb: torch.Tensor,
    bare_qsfc: torch.Tensor,
    veg_irg: torch.Tensor,
    veg_irc: torch.Tensor,
    veg_shg: torch.Tensor,
    veg_shc: torch.Tensor,
    veg_evg: torch.Tensor,
    veg_evc: torch.Tensor,
    veg_tr: torch.Tensor,
    veg_gh: torch.Tensor,
    veg_ch: torch.Tensor,
    veg_tah: torch.Tensor,
    veg_eah: torch.Tensor,
    veg_psnsun: torch.Tensor,
    veg_psnsha: torch.Tensor,
    fira: torch.Tensor,
    fsh: torch.Tensor,
    fgev: torch.Tensor,
    ssoil: torch.Tensor,
    trad: torch.Tensor,
    qsfc: torch.Tensor,
    tv: torch.Tensor,
    fcev: torch.Tensor,
    fctr: torch.Tensor,
    psn: torch.Tensor,
    emissivity_total: torch.Tensor,
    z0_total: torch.Tensor,
    z0h_total: torch.Tensor,
    selected_gfs_v17: bool = False,
    friction_velocity: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    fira_canopy = fveg * veg_irg + (1.0 - fveg) * bare_irb + veg_irc
    fsh_canopy = fveg * veg_shg + (1.0 - fveg) * bare_shb + veg_shc
    fgev_canopy = fveg * veg_evg + (1.0 - fveg) * bare_evb
    ssoil_canopy = fveg * veg_gh + (1.0 - fveg) * bare_ghb
    fcev_canopy = veg_evc
    fctr_canopy = veg_tr
    ch_canopy = fveg * veg_ch + (1.0 - fveg) * bare_ch
    fire_canopy = dlwflx + fira_canopy
    emissi_canopy = fveg * (
        geometry_emg * (1.0 - geometry_emv) + geometry_emv + geometry_emv * (1.0 - geometry_emv) * (1.0 - geometry_emg)
    ) + (1.0 - fveg) * geometry_emg
    trad_canopy_base = (fire_canopy - (1.0 - emissi_canopy) * dlwflx) / torch.clamp(emissi_canopy * LAND_ENERGY_SB, min=1.0e-30)
    trad_canopy = working_power(torch.clamp(trad_canopy_base, min=1.0e-30), 0.25)
    psn_canopy = veg_psnsun * laisun + veg_psnsha * laisha
    denom = torch.where(active_canopy_source, prsl1 + CON_EPSM1 * veg_eah, 1.0)
    qsfc_canopy = fveg * (veg_eah * CON_EPS / denom) + (1.0 - fveg) * bare_qsfc
    ch_canopy_safe = torch.where(active_canopy_source & (torch.abs(ch_canopy) > 1.0e-30), ch_canopy, 1.0)
    tv_canopy_numerator = fveg * veg_ch * veg_tah + (1.0 - fveg) * bare_ch * bare_tgb
    tv_canopy_numerator = torch.where(active_canopy_source, tv_canopy_numerator, 0.0)
    z0m_composite = fveg * geometry_z0m + (1.0 - fveg) * geometry_z0mg
    composite_ustarx = torch.full_like(z0m_composite, 0.1) if friction_velocity is None else friction_velocity
    czil_composite = torch.pow(torch.full_like(z0m_composite, 10.0), -0.4 * hvt_by_column)
    reyn_composite = composite_ustarx * z0m_composite / 1.5e-5
    kb_sigma_f0_composite = torch.where(
        reyn_composite > 2.0,
        2.46 * torch.pow(torch.where(reyn_composite > 2.0, reyn_composite, 4.0), 0.25) - torch.log(torch.full_like(reyn_composite, 7.4)),
        -torch.log(torch.full_like(reyn_composite, 0.397)),
    )
    z0h_vegetated = geometry_z0m * torch.exp(-czil_composite * 0.4 * 258.2 * torch.sqrt(composite_ustarx * geometry_z0m))
    z0h_bare_roughness = geometry_z0m if selected_gfs_v17 else geometry_z0mg
    z0h_bare_for_composite = torch.maximum(
        z0h_bare_roughness / torch.exp(kb_sigma_f0_composite),
        torch.full_like(geometry_z0mg, 1.0e-6),
    )
    z0h_composite = torch.exp(fveg * torch.log(z0h_vegetated) + (1.0 - fveg) * torch.log(z0h_bare_for_composite))
    return (
        torch.where(active_canopy_source, fira_canopy, fira),
        torch.where(active_canopy_source, fsh_canopy, fsh),
        torch.where(active_canopy_source, fgev_canopy, fgev),
        torch.where(active_canopy_source, ssoil_canopy, ssoil),
        torch.where(active_canopy_source, trad_canopy, trad),
        torch.where(active_canopy_source, qsfc_canopy, qsfc),
        torch.where(active_canopy_source, tv_canopy_numerator / ch_canopy_safe, tv),
        torch.where(active_canopy_source, fcev_canopy, fcev),
        torch.where(active_canopy_source, fctr_canopy, fctr),
        torch.where(active_canopy_source, psn_canopy, psn),
        torch.where(active_canopy_source, emissi_canopy, emissivity_total),
        torch.where(active_canopy_source, z0m_composite, z0_total),
        torch.where(active_canopy_source, z0h_composite, z0h_total),
        torch.where(active_canopy_source, ch_canopy, bare_ch),
    )


def _temperature_merge_flat_impl(
    *,
    source_temperature_columns: torch.Tensor,
    phase_stc: torch.Tensor,
    phase_snice: torch.Tensor,
    phase_snliq: torch.Tensor,
    phase_imelt: torch.Tensor,
    phase_ponding: torch.Tensor,
    temperature_snow_soil: torch.Tensor,
    snow_level_ice: torch.Tensor,
    snow_level_liquid: torch.Tensor,
    dzsnso: torch.Tensor,
    ssoil: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dz = torch.clamp(torch.abs(dzsnso), min=1.0e-6)
    hcp = temperature_snow_soil * 0.0 + 2.0e6
    top_update = temperature_snow_soil[:, 3] + dt * ssoil / (hcp[:, 3] * dz[:, 3])
    soil_stc = _columns_to_matrix(
        (
            temperature_snow_soil[:, 0],
            temperature_snow_soil[:, 1],
            temperature_snow_soil[:, 2],
            top_update,
            temperature_snow_soil[:, 4],
            temperature_snow_soil[:, 5],
            temperature_snow_soil[:, 6],
        )
    )
    return (
        torch.where(source_temperature_columns[:, None], phase_stc, soil_stc),
        torch.where(source_temperature_columns[:, None], phase_snice, snow_level_ice),
        torch.where(source_temperature_columns[:, None], phase_snliq, snow_level_liquid),
        torch.where(source_temperature_columns[:, None], phase_imelt, 0.0),
        torch.where(source_temperature_columns, phase_ponding, 0.0),
    )


def _water_pre_snow_flat_impl(
    *,
    fgev: torch.Tensor,
    latheag: torch.Tensor,
    frozen_canopy: torch.Tensor,
    snow_water_equiv: torch.Tensor,
    snow_levels: torch.Tensor,
    dt: torch.Tensor,
    interface_depth: torch.Tensor,
    dzsnso: torch.Tensor,
    zsnso_no_snow_template: torch.Tensor,
    dzsnso_no_snow_template: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    qvap = torch.clamp(fgev / latheag, min=0.0)
    qdew = torch.abs(torch.clamp(fgev / latheag, max=0.0))
    edir = qvap - qdew
    latent_zero = fgev * 0.0
    latheav = torch.where(frozen_canopy, latent_zero + NOAHMP_HSUB, latent_zero + NOAHMP_HVAP)
    qsnsub = torch.where(snow_water_equiv > 0.0, torch.minimum(qvap, snow_water_equiv / dt), 0.0)
    qsnfro = torch.where(snow_water_equiv > 0.0, qdew, 0.0)
    qsdew = qdew - qsnfro
    qseva = qvap - qsnsub
    esnow = qsnsub * NOAHMP_HSUB
    no_snow = snow_levels == 0
    zsnso = torch.where(no_snow[:, None], zsnso_no_snow_template, interface_depth)
    dzsnso = torch.where(no_snow[:, None], dzsnso_no_snow_template, dzsnso)
    return qvap, qdew, edir, latheav, qsnsub, qsnfro, qsdew, qseva, esnow, zsnso, dzsnso


def _water_snow_to_soil_flat_impl(
    *,
    phase_smc: torch.Tensor,
    phase_sh2o: torch.Tensor,
    zsnso: torch.Tensor,
    dzsnso: torch.Tensor,
    snow_levels: torch.Tensor,
    latheag: torch.Tensor,
    qvap: torch.Tensor,
    qsdew: torch.Tensor,
    qseva: torch.Tensor,
    phase_ponding: torch.Tensor,
    qrain_ground: torch.Tensor,
    snowwater_smc: torch.Tensor,
    snowwater_sh2o: torch.Tensor,
    snowwater_sice: torch.Tensor,
    snowwater_zsnso: torch.Tensor,
    snowwater_dzsnso: torch.Tensor,
    snowwater_ponding1: torch.Tensor,
    snowwater_ponding2: torch.Tensor,
    snowwater_qsnbot: torch.Tensor,
    snowwater_snow_levels: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    no_snow = snow_levels == 0
    snow_active = ~no_snow
    frozen_ground = latheag == NOAHMP_HSUB
    top_soil_dz = torch.clamp(snowwater_dzsnso[:, 3], min=1.0e-30)
    sice_fg0 = snowwater_sice[:, 0] + (qsdew - qseva) * dt / (top_soil_dz * 1000.0)
    negative_sice = sice_fg0 < 0.0
    sh2o_fg0 = torch.where(negative_sice, snowwater_sh2o[:, 0] + sice_fg0, snowwater_sh2o[:, 0])
    sice_fg0 = torch.where(negative_sice, 0.0, sice_fg0)
    sice_fg = _columns_to_matrix((sice_fg0, snowwater_sice[:, 1], snowwater_sice[:, 2], snowwater_sice[:, 3]))
    sh2o_fg = _columns_to_matrix((sh2o_fg0, snowwater_sh2o[:, 1], snowwater_sh2o[:, 2], snowwater_sh2o[:, 3]))
    snow_sh2o = torch.where(frozen_ground[:, None], sh2o_fg, snowwater_sh2o)
    snow_sice = torch.where(frozen_ground[:, None], sice_fg, snowwater_sice)
    qseva_soil = torch.where(frozen_ground, 0.0, qseva)
    qsdew_soil = torch.where(frozen_ground, 0.0, qsdew)
    qinsur_snow = (phase_ponding + snowwater_ponding1 + snowwater_ponding2) / dt * 0.001
    qinsur_snow = torch.where(
        snowwater_snow_levels == 0,
        qinsur_snow + (snowwater_qsnbot + qsdew_soil + qrain_ground) * 0.001,
        qinsur_snow + (snowwater_qsnbot + qsdew_soil) * 0.001,
    )
    smc = phase_smc
    sh2o = snow_sh2o
    zsnso = snowwater_zsnso
    dzsnso = snowwater_dzsnso
    qinsur = qinsur_snow
    qseva_input = qseva_soil
    return smc, sh2o, snow_sice, zsnso, dzsnso, qinsur, qseva_input


def _water_final_merge_flat_impl(
    *,
    active_source_water: torch.Tensor,
    snow_active: torch.Tensor,
    qvap: torch.Tensor,
    qdew: torch.Tensor,
    edir: torch.Tensor,
    ecan: torch.Tensor,
    etran: torch.Tensor,
    esnow: torch.Tensor,
    smc_before_soilwater: torch.Tensor,
    sh2o_before_soilwater: torch.Tensor,
    zsnso: torch.Tensor,
    dzsnso: torch.Tensor,
    phase_stc: torch.Tensor,
    phase_snowh: torch.Tensor,
    phase_sneqv: torch.Tensor,
    snow_levels: torch.Tensor,
    snowwater_qsnbot: torch.Tensor,
    snowwater_stc: torch.Tensor,
    snowwater_snowh: torch.Tensor,
    snowwater_sneqv: torch.Tensor,
    snowwater_snow_levels: torch.Tensor,
    soilwater_smc: torch.Tensor,
    soilwater_sh2o: torch.Tensor,
    soilwater_runsrf: torch.Tensor,
    soilwater_runsub: torch.Tensor,
    soilwater_wa: torch.Tensor,
    canliq: torch.Tensor,
    canice: torch.Tensor,
    cmc: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    zeros = qvap * 0.0
    smc = torch.where(active_source_water[:, None], soilwater_smc, smc_before_soilwater)
    sh2o = torch.where(active_source_water[:, None], soilwater_sh2o, sh2o_before_soilwater)
    runsrf = torch.where(active_source_water, soilwater_runsrf, zeros)
    runsub = torch.where(active_source_water, soilwater_runsub, zeros)
    wa = torch.where(active_source_water, soilwater_wa, 4900.0)
    return (
        qvap,
        qdew,
        edir,
        ecan,
        etran,
        runsrf,
        runsub,
        snowwater_qsnbot,
        esnow,
        smc,
        sh2o,
        snowwater_stc,
        zsnso,
        dzsnso,
        wa,
        canliq,
        canice,
        cmc,
        snowwater_snowh,
        snowwater_sneqv,
        snowwater_snow_levels,
    )


_I_RESULT_FIELDS = (
    "qvap",
    "qdew",
    "edir",
    "ecan",
    "etran",
    "runsrf",
    "runsub",
    "qsnbot",
    "esnow",
    "smc",
    "sh2o",
    "stc",
    "zsnso",
    "dzsnso",
    "wa",
    "canliq",
    "canice",
    "cmc",
    "snowh",
    "sneqv",
    "snow_levels",
)


def _water_balance_error(
    state: LandBodyState,
    i: Mapping[str, torch.Tensor],
    *,
    canliq: torch.Tensor,
    canice: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """sflx ``error``: end minus begin storage minus net forcing, in mm per step."""
    default_wa = state.cmc * 0.0 + 4900.0 if state.aquifer_water is None else state.aquifer_water
    beg_wb = state.canopy_liquid_begin + state.canopy_ice_begin + state.snow_water_equiv + default_wa
    for layer in range(4):
        beg_wb = beg_wb + state.soil_moisture_vol[:, layer] * state.dzsnso[:, layer + 3] * 1000.0
    end_wb = canliq + canice + i.get("sneqv", state.snow_water_equiv) + i.get("wa", default_wa)
    dzsnso = i.get("dzsnso", state.dzsnso)
    for layer in range(4):
        end_wb = end_wb + i["smc"][:, layer] * dzsnso[:, layer + 3] * 1000.0
    prcp = state.precipitation
    return end_wb - beg_wb - (prcp - i["ecan"] - i["etran"] - i["edir"] - i["runsrf"] - i["runsub"]) * dt


def _numerical_water_loss(errwat: torch.Tensor) -> torch.Tensor:
    """Keep numerical storage loss in the audit ledger, not physical runoff."""
    return torch.where(errwat < 0.0, -errwat, errwat * 0.0)


def compute_land_body_hij(
    state: LandBodyState,
    tables: NoahMpTableBundle,
    opts: Mapping[str, Any],
    *,
    include_checkpoint_diagnostics: bool = True,
) -> LandBodyHIJ:
    """Compute H/I/J source-smoke land-body fields without reading oracles."""
    hij = land_body_hij_before_canopy(state, tables, opts)
    veg_flux = None
    if hij["has_canopy"]:
        veg_flux = vege_flux_opt_sfc3(
            hij["state"],
            hij["energy_tables"],
            hij["geometry"],
            hij["thermo"],
            hij["_radiation"],
            dt=hij["dt"],
            canliq=hij["canliq"],
            canice=hij["canice"],
            fwet=hij["fwet"],
            btran=hij["btran"].btran,
            surface_exchange_option=int(opts.get("iopt_sfc", 3)),
        )
    return land_body_hij_after_canopy(hij, veg_flux, include_checkpoint_diagnostics=include_checkpoint_diagnostics)


def land_body_hij_before_canopy(state: LandBodyState, tables: NoahMpTableBundle, opts: Mapping[str, Any]) -> dict[str, Any]:
    """``compute_land_body_hij`` up to its canopy flux call: every value the rest reads, and the call's
    canopy water inputs (``has_canopy`` false: no canopy call)."""
    _validate_state(opts, tables, state)
    soilwater_iteration_count = int(opts.get("soilwater_iteration_count", opts.get("_soilwater_iteration_count", 6)))
    if soilwater_iteration_count != 6:
        raise ValueError("soilwater_iteration_count must be 6; source activity selects three or six iterations")
    # Source energy clears tiny snow after the beginning-of-step water budget.
    initial_state = state
    snow_exists = (state.snow_depth > 1.0e-6) & (state.snow_water_equiv > 1.0e-3)
    state = replace(
        state,
        snow_depth=torch.where(snow_exists, state.snow_depth, torch.zeros_like(state.snow_depth)),
        snow_water_equiv=torch.where(
            snow_exists, state.snow_water_equiv, torch.zeros_like(state.snow_water_equiv)
        ),
    )
    dt = torch.full_like(state.temperature_forcing, float(opts["delt"]))
    fsa, fsr, sav, sag, apar, psn, fsno, energy_tables, geometry, _radiation = _radiation_source_corridor(
        state,
        dt=dt,
        table_variant=tables.table_variant,
    )
    fira, fsh, fgev, ssoil, trad, qsfc, tv, latheag, _emg = _surface_flux_source_corridor(state, sag, fsno)
    thermo = thermoprop_source_corridor(state, energy_tables, geometry, dt=dt, shdfac=state.shdfac)
    bare = bare_flux_opt_sfc3(
        state,
        geometry,
        sag=sag,
        df=thermo.df,
        dt=dt,
        selected_gfs_v17=tables.table_variant == "gfs_v17_profile_scm_da75531",
        surface_exchange_option=int(opts.get("iopt_sfc", 3)),
    )
    bare_only = ((state.elai + state.esai) <= 0.0) | (state.fveg <= 0.0)
    (
        fira,
        fsh,
        fgev,
        ssoil,
        trad,
        qsfc,
        tv,
        latheag,
        emissivity_total,
        z0_total,
        z0h_total,
    ) = run_compile_island(
        "land_flux_merge",
        "bare_flux_merge",
        _bare_flux_merge_flat_impl,
        fira=fira,
        fsh=fsh,
        fgev=fgev,
        ssoil=ssoil,
        trad=trad,
        qsfc=qsfc,
        tv=tv,
        latheag=latheag,
        geometry_emg=geometry.emg,
        geometry_latheag=geometry.latheag,
        geometry_z0mg=geometry.z0mg,
        bare_z0h=bare.z0h,
        bare_irb=bare.irb,
        bare_shb=bare.shb,
        bare_evb=bare.evb,
        bare_ghb=bare.ghb,
        bare_qsfc=bare.qsfc,
        bare_tgb=bare.tgb,
        dlwflx=state.dlwflx,
        bare_only=bare_only,
    )
    exchange_heat_total = bare.ch
    active_canopy = (state.elai + state.esai > 0.0) & (state.fveg > 0.0)
    no_snow = (state.snow_levels == 0) & (state.snow_depth <= 1.0e-6) & (state.snow_water_equiv <= 1.0e-3)
    active_canopy_no_snow = active_canopy & no_snow
    active_canopy_source = active_canopy
    btran = btran_opt_btr1_source_corridor(state, energy_tables)
    zero_flux = fgev * 0.0
    fcev = zero_flux
    fctr = zero_flux
    has_canopy = bool(opts.get("_has_active_canopy_source", True))
    veg_idx = canliq = canice = fwet = None
    if has_canopy:
        veg_idx = state.veg_idx if state.veg_idx is not None else (state.vegtype if state.vegtype.dtype == torch.int64 else state.vegtype.to(torch.int64)) - 1
        canliq = torch.clamp(state.canopy_liquid_initial, min=0.0)
        canice = torch.clamp(state.canopy_ice_initial, min=0.0)
        fwet = state.fwet
    return {
        "_radiation": _radiation, "active_canopy": active_canopy,
        "active_canopy_no_snow": active_canopy_no_snow, "active_canopy_source": active_canopy_source,
        "apar": apar, "bare": bare, "bare_only": bare_only, "btran": btran, "canice": canice, "canliq": canliq,
        "dt": dt, "emissivity_total": emissivity_total, "energy_tables": energy_tables,
        "exchange_heat_total": exchange_heat_total, "fcev": fcev, "fctr": fctr, "fgev": fgev, "fira": fira,
        "fsa": fsa, "fsh": fsh, "fsno": fsno, "fsr": fsr, "fwet": fwet, "geometry": geometry,
        "has_canopy": has_canopy, "initial_state": initial_state, "latheag": latheag, "opts": opts, "psn": psn,
        "qsfc": qsfc, "soilwater_iteration_count": soilwater_iteration_count, "ssoil": ssoil, "state": state,
        "tables": tables, "thermo": thermo, "trad": trad, "tv": tv, "veg_idx": veg_idx, "z0_total": z0_total,
        "z0h_total": z0h_total,
    }


def land_body_hij_after_canopy(
    hij: Mapping[str, Any], veg_flux: Any, *, include_checkpoint_diagnostics: bool = True
) -> LandBodyHIJ:
    """``compute_land_body_hij`` from its canopy flux result (None without a canopy call)."""
    (
        _radiation, active_canopy, active_canopy_no_snow, active_canopy_source, apar, bare, bare_only, btran, dt,
        emissivity_total, energy_tables, exchange_heat_total, fcev, fctr, fgev, fira, fsa, fsh, fsno, fsr, geometry,
        initial_state, latheag, opts, psn, qsfc, soilwater_iteration_count, ssoil, state, tables, thermo, trad, tv,
        veg_idx, z0_total, z0h_total,
    ) = (hij[name] for name in (
        "_radiation active_canopy active_canopy_no_snow active_canopy_source apar bare bare_only btran dt "
        "emissivity_total energy_tables exchange_heat_total fcev fctr fgev fira fsa fsh fsno fsr geometry "
        "initial_state latheag opts psn qsfc soilwater_iteration_count ssoil state tables thermo trad tv "
        "veg_idx z0_total z0h_total"
    ).split())
    if veg_flux is not None:
        (
            fira,
            fsh,
            fgev,
            ssoil,
            trad,
            qsfc,
            tv,
            fcev,
            fctr,
            psn,
            emissivity_total,
            z0_total,
            z0h_total,
            exchange_heat_total,
        ) = run_compile_island(
            "land_flux_merge",
            "canopy_flux_merge",
            _canopy_flux_merge_flat_impl,
            active_canopy_source=active_canopy_source,
            fveg=state.fveg,
            prsl1=state.prsl1,
            dlwflx=state.dlwflx,
            geometry_emg=geometry.emg,
            geometry_emv=geometry.emv,
            geometry_z0m=geometry.z0m,
            geometry_z0mg=geometry.z0mg,
            hvt_by_column=state.hvt_by_column if state.hvt_by_column is not None else energy_tables.hvt[veg_idx],
            laisun=_radiation.laisun,
            laisha=_radiation.laisha,
            bare_irb=bare.irb,
            bare_shb=bare.shb,
            bare_evb=bare.evb,
            bare_ghb=bare.ghb,
            bare_ch=bare.ch,
            bare_tgb=bare.tgb,
            bare_qsfc=bare.qsfc,
            veg_irg=veg_flux.irg,
            veg_irc=veg_flux.irc,
            veg_shg=veg_flux.shg,
            veg_shc=veg_flux.shc,
            veg_evg=veg_flux.evg,
            veg_evc=veg_flux.evc,
            veg_tr=veg_flux.tr,
            veg_gh=veg_flux.gh,
            veg_ch=veg_flux.ch,
            veg_tah=veg_flux.tah,
            veg_eah=veg_flux.eah,
            veg_psnsun=veg_flux.psnsun,
            veg_psnsha=veg_flux.psnsha,
            fira=fira,
            fsh=fsh,
            fgev=fgev,
            ssoil=ssoil,
            trad=trad,
            qsfc=qsfc,
            tv=tv,
            fcev=fcev,
            fctr=fctr,
            psn=psn,
            emissivity_total=emissivity_total,
            z0_total=z0_total,
            z0h_total=z0h_total,
            selected_gfs_v17=tables.table_variant == GFS_V17_PROFILE_SCM_TABLE_VARIANT,
            friction_velocity=state.friction_velocity,
        )
    source_stc = tsnosoi_source_corridor(state, energy_tables, thermo, ssoil=ssoil, dt=dt)
    (
        phase_stc,
        phase_snice,
        phase_snliq,
        phase_smc,
        phase_sh2o,
        phase_sneqv,
        phase_snowh,
        phase_imelt,
        phase_ponding,
    ) = _phasechange_source_corridor(
        state,
        energy_tables,
        thermo,
        stc=source_stc,
        snice=state.snow_level_ice,
        snliq=state.snow_level_liquid,
        smc=state.soil_moisture_vol,
        sh2o=state.soil_liquid_vol,
        dt=dt,
    )
    source_temperature_columns = bare_only | active_canopy_source
    stc, snice, snliq, imelt, ponding = run_compile_island(
        "land_temperature_merge",
        "temperature_phase_merge",
        _temperature_merge_flat_impl,
        source_temperature_columns=source_temperature_columns,
        phase_stc=phase_stc,
        phase_snice=phase_snice,
        phase_snliq=phase_snliq,
        phase_imelt=phase_imelt,
        phase_ponding=phase_ponding,
        temperature_snow_soil=state.temperature_snow_soil,
        snow_level_ice=state.snow_level_ice,
        snow_level_liquid=state.snow_level_liquid,
        dzsnso=state.dzsnso,
        ssoil=ssoil,
        dt=dt,
    )
    h = {}
    if include_checkpoint_diagnostics:
        h = {
            "fsa": fsa,
            "fsr": fsr,
            "fira": fira,
            "fsh": fsh,
            "fcev": fcev,
            "fgev": fgev,
            "fctr": fctr,
            "ssoil": ssoil,
            "trad": trad,
            "psn": psn,
            "apar": apar,
            "ponding": ponding,
            "imelt": imelt,
            "stc": stc,
            "snice": snice,
            "snliq": snliq,
        }

    initial_leaf = state.tskin if state.temperature_leaf is None else state.temperature_leaf
    energy_leaf = initial_leaf if veg_flux is None else torch.where(active_canopy, veg_flux.tv, initial_leaf)
    frozen_canopy = initial_leaf <= NOAHMP_TFRZ
    i, _qvap, _qdew, _edir = _water_source_corridor(
        state,
        energy_tables,
        fgev,
        fcev,
        fctr,
        latheag,
        frozen_canopy,
        dt,
        bare_only,
        active_canopy_no_snow,
        active_canopy_source,
        btran.btrani,
        phase_stc,
        phase_snice,
        phase_snliq,
        phase_smc,
        phase_sh2o,
        phase_sneqv,
        phase_snowh,
        phase_imelt,
        phase_ponding,
        soilwater_iteration_count,
        energy_leaf,
    )

    canliq = i.get("canliq", torch.clamp(state.cmc, min=0.0))
    canice = i.get("canice", canliq * 0.0)
    urban = (state.vegtype == tables.isurban) | (state.vegtype == 31) | (state.vegtype == 32) | (state.vegtype == 33)
    qfx = i["etran"] + i["ecan"] + i["edir"]
    # noahmp_sflx: ch is the energy output (chb on urban columns, whose fveg is zero), not the input.
    qsfc_urban = qfx / (state.rhoair * torch.clamp(exchange_heat_total, min=1.0e-12)) + state.qair
    qsfc_j = torch.where(urban, qsfc_urban, qsfc)
    j = {"qsfc": qsfc_j}
    q2b = torch.where(urban, qsfc_j, bare.q2b)  # noahmp_sflx: urban q2b = qsfc after the energy call
    errwat = _water_balance_error(initial_state, i, canliq=canliq, canice=canice, dt=dt)
    if include_checkpoint_diagnostics:
        snowh = torch.where((i["snowh"] <= 1.0e-6) | (i["sneqv"] <= 1.0e-3), 0.0, i["snowh"])
        sneqv = torch.where((i["snowh"] <= 1.0e-6) | (i["sneqv"] <= 1.0e-3), 0.0, i["sneqv"])
        albedo = torch.where(state.swdown != 0.0, fsr / torch.where(state.swdown != 0.0, state.swdown, 1.0), -999.9)
        zero_j = fsa * 0.0
        j = {
            "dveg_active": torch.zeros_like(state.vegtype, dtype=torch.int64),
            "crop_active": torch.zeros_like(state.vegtype, dtype=torch.int64),
            "gpp": zero_j,
            "npp": zero_j,
            "nee": zero_j,
            "errwat": errwat,
            "fveg": state.fveg,
            "albedo": albedo,
            "snowh": snowh,
            "sneqv": sneqv,
            "qsfc": qsfc_j,
            "errflg": torch.zeros_like(state.vegtype, dtype=torch.int64),
        }
    tail = {
        "temperature_radiative": trad,
        "surface_temperature": tv,
        "sensible_heat_total": fsh,
        "latent_heat_total": fcev + fgev + fctr,
        "ground_heat_total": ssoil,
        "snow_cover_fraction": fsno,
        "vegetation_fraction": state.fveg,
        "z0_total": z0_total,
        "z0h_total": z0h_total,
        "exchange_heat_total": exchange_heat_total,
        "soil_moisture_vol": i["smc"],
        "soil_liquid_vol": i["sh2o"],
        "temperature_snow_soil": i.get("stc", stc),
        "spec_humidity_surface": j["qsfc"],
        "sw_absorbed_total": fsa,
        "emissivity_total": emissivity_total,
        "runoff_surface": i["runsrf"],
        "evaporation_soil": i["edir"],
        "snow_levels": i["snow_levels"],
    }
    clear_snow = (i["snowh"] <= 1.0e-6) | (i["sneqv"] <= 1.0e-3)
    tail.update(budget__raw_residual_mm=errwat,
        budget__numerical_loss_mm=_numerical_water_loss(errwat),
        budget__snow_cleanup_sink_mm=torch.where(clear_snow, i["sneqv"], 0.0))
    leaf_temperature = i["tv"]
    tail.update({
        "driver__snowxy": i["snow_levels"].to(trad.dtype),
        "driver__weasd": torch.where(clear_snow, 0.0, i["sneqv"]),
        "driver__snwdph": torch.where(clear_snow, 0.0, i["snowh"]) * 1000.0,
        "driver__tsnoxy": i["stc"][:, :3], "driver__snicexy": i["snice"],
        "driver__snliqxy": i["snliq"], "driver__zsnsoxy": i["zsnso"],
        "driver__slc": i["sh2o"], "driver__canopy": i["cmc"],
        "driver__canliqxy": canliq, "driver__canicexy": canice,
        "driver__waxy": i["wa"], "driver__wtxy": i["wt"], "driver__zwtxy": i["zwt"],
        "driver__ecan": i["ecan"], "driver__etran": i["etran"], "driver__edir": i["edir"],
        "driver__drain": i["runsub"], "driver__sbsno": i["esnow"],
        "driver__snohf": i["qsnbot"] * CON_HFUS,
        "driver__evbs": fgev, "driver__evcw": fcev, "driver__trans": fctr,
        "driver__gflux": -ssoil, "driver__pah": state.fveg * state.pahg + (1.0 - state.fveg) * state.pahb + state.pahv,
        "driver__fwetxy": i["fwet"], "driver__taussxy": _radiation.tauss,
        "driver__sneqvoxy": phase_sneqv,
        "driver__snowc": fsno, "driver__sncovr1": fsno,
        "driver__emiss": emissivity_total, "driver__sfalb": torch.where(state.swdown != 0.0, fsr / torch.where(state.swdown != 0.0, state.swdown, 1.0), -999.9),
        "driver__albdvis": _radiation.albd[:, 0], "driver__albdnir": _radiation.albd[:, 1],
        "driver__albivis": _radiation.albi[:, 0], "driver__albinir": _radiation.albi[:, 1],
        "driver__tvxy": leaf_temperature,
        "driver__tgxy": bare.tgb if veg_flux is None else torch.where(active_canopy, state.fveg * veg_flux.tgv + (1.0 - state.fveg) * bare.tgb, bare.tgb),
        "driver__tahxy": (state.tskin if state.temperature_canopy_air is None else state.temperature_canopy_air) if veg_flux is None else torch.where(active_canopy, veg_flux.tah, state.tskin if state.temperature_canopy_air is None else state.temperature_canopy_air),
        "driver__eahxy": (state.eair if state.vapor_pres_canopy_air is None else state.vapor_pres_canopy_air) if veg_flux is None else torch.where(active_canopy, veg_flux.eah, state.eair if state.vapor_pres_canopy_air is None else state.vapor_pres_canopy_air),
        "driver__t2mmp": bare.t2mb if veg_flux is None else torch.where(active_canopy, state.fveg * veg_flux.t2mv + (1.0 - state.fveg) * bare.t2mb, bare.t2mb),
        "driver__q2mp": q2b if veg_flux is None else torch.where(active_canopy, state.fveg * veg_flux.q2v + (1.0 - state.fveg) * q2b, q2b),
    })
    if veg_flux is not None:
        valid_resistance = active_canopy & (_radiation.laisun != 0.0) & (_radiation.laisha != 0.0) & (veg_flux.rssun > 0.0) & (veg_flux.rssha > 0.0)
        conductance = 1.0 / (veg_flux.rssun + veg_flux.rb) * _radiation.laisun + 1.0 / (veg_flux.rssha + veg_flux.rb) * _radiation.laisha
        resistance = 1.0 / torch.where(valid_resistance, conductance, 1.0)
        idx = state.vegtype.to(torch.int64) - 1
        tail["driver__rca"] = torch.where(valid_resistance, torch.maximum(resistance, energy_tables.rsmin[idx]), energy_tables.rsmax[idx])
    return LandBodyHIJ(h=h, i=i, j=j, tail=tail)






def expand_land_body_tail_state(
    case_data: Mapping[str, Any],
    state: LandBodyState,
    result: LandBodyHIJ,
    *,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Expand active-column H/I/J internals into the post-merge K-O tail shape."""
    dev = torch.device("cpu") if device is None else torch.device(device)
    im = int(case_data["dimensions"]["im"])
    dtype = state.temperature_forcing.dtype
    indices = state.column_indices.to(device=dev, dtype=torch.int64)
    mask = torch.zeros(im, device=dev, dtype=torch.bool).scatter(0, indices, torch.ones_like(indices, dtype=torch.bool))

    def full_vector(values: torch.Tensor, *, fill: float = 0.0) -> torch.Tensor:
        expanded = torch.full((im,), fill, device=dev, dtype=values.dtype)
        return expanded.scatter(0, indices, values)

    def full_matrix(values: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
        if base is None:
            expanded = torch.zeros((im, values.shape[1]), device=dev, dtype=values.dtype)
        else:
            expanded = base
        scatter_indices = indices[:, None].expand_as(values)
        return expanded.scatter(0, scatter_indices, values)

    tail = result.tail
    return {
        "mask": mask,
        "indices": state.column_indices,
        "temperature_radiative": full_vector(tail["temperature_radiative"]),
        "surface_temperature": full_vector(tail["surface_temperature"]),
        "sensible_heat_total": full_vector(tail["sensible_heat_total"]),
        "latent_heat_total": full_vector(tail["latent_heat_total"]),
        "ground_heat_total": full_vector(tail["ground_heat_total"]),
        "snow_cover_fraction": full_vector(tail["snow_cover_fraction"]),
        "vegetation_fraction": full_vector(tail["vegetation_fraction"]),
        "z0_total": full_vector(tail["z0_total"]),
        "z0h_total": full_vector(tail["z0h_total"]),
        "exchange_heat_total": full_vector(tail["exchange_heat_total"]),
        "soil_moisture_vol": full_matrix(tail["soil_moisture_vol"], _matrix_tensor(case_data, "smc", device=dev)),
        "soil_liquid_vol": full_matrix(tail["soil_liquid_vol"], _matrix_tensor(case_data, "slc", device=dev)),
        "temperature_snow_soil": full_matrix(tail["temperature_snow_soil"]),
        "spec_humidity_surface": full_vector(tail["spec_humidity_surface"]),
        "sw_absorbed_total": full_vector(tail["sw_absorbed_total"]),
        "emissivity_total": full_vector(tail["emissivity_total"]),
        "runoff_surface": full_vector(tail["runoff_surface"]),
        "evaporation_soil": full_vector(tail["evaporation_soil"]),
        "snow_levels": full_vector(tail["snow_levels"].to(dtype), fill=0.0).to(torch.int64),
    }


# (boundary, field, integer) of the H/I/J checkpoint fields, in emission order.
_HIJ_EMISSION_FIELDS = (
    *(("H", field, False) for field in (*H_SCALAR_FIELDS, "imelt", "stc", "snice", "snliq")),
    *(("I", field, False) for field in (*I_SCALAR_FIELDS, "smc", "sh2o", "zsnso")),
    ("J", "dveg_active", True),
    ("J", "crop_active", True),
    *(("J", field, False) for field in J_SCALAR_FIELDS),
    ("J", "errflg", True),
)


def emit_land_body_hij(result: LandBodyHIJ, column_indices: Sequence[int]) -> dict[str, torch.Tensor]:
    """Checkpoint-shaped per-column H/I/J fields: ``<B>__<field>__col<k>`` (1-based source column).

    ``column_indices`` is the host list of the active columns (from the case), so naming the
    fields reads no device value.
    """
    groups = {"H": result.h, "I": result.i, "J": result.j}
    out: dict[str, torch.Tensor] = {}
    for boundary, field, integer in _HIJ_EMISSION_FIELDS:
        values = groups[boundary][field]
        if values.ndim == 0 or values.shape[0] != len(column_indices):
            raise ValueError(f"{boundary}.{field}: expected one row per active column")
        values = values.to(torch.int64) if integer else values
        out.update({f"{boundary}__{field}__col{int(col) + 1}": values[row] for row, col in enumerate(column_indices)})
    return out


def run_land_body_hij_with_tail_state(
    case_data: Mapping[str, Any],
    g_outputs: Mapping[str, torch.Tensor],
    tables: NoahMpTableBundle,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    driver_options: Mapping[str, Any] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return H/I/J checkpoint emissions plus the K-O post-merge state."""
    state = state_from_case_after_g(case_data, g_outputs, tables, device=device, dtype=dtype)
    opts = case_data["driver_options"] if driver_options is None else driver_options
    result = compute_land_body_hij(state, tables, opts)
    return emit_land_body_hij(result, _host_active_land_indices(case_data, tables)), expand_land_body_tail_state(
        case_data,
        state,
        result,
        device=device,
    )


def run_land_body_tail_state(
    case_data: Mapping[str, Any],
    g_outputs: Mapping[str, torch.Tensor],
    tables: NoahMpTableBundle,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    expand: bool = True,
    driver_options: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Return only the K-O post-merge state for production/benchmark runs."""
    state = state_from_case_after_g(case_data, g_outputs, tables, device=device, dtype=dtype)
    opts = case_data["driver_options"] if driver_options is None else driver_options
    if not expand and _active_land_indices_cover_all_columns(case_data, tables):
        result = compute_land_body_hij(
            state,
            tables,
            opts,
            include_checkpoint_diagnostics=False,
        )
        return {
            **result.tail,
            "mask": torch.ones((int(case_data["dimensions"]["im"]),), device=state.column_indices.device, dtype=torch.bool),
            "indices": state.column_indices,
        }
    result = compute_land_body_hij(state, tables, opts)
    return expand_land_body_tail_state(case_data, state, result, device=device)


def run_land_body_tail_output_from_state(
    state: LandBodyState,
    tables: NoahMpTableBundle,
    opts: Mapping[str, Any],
) -> LandBodyTailOutput:
    """Run the production H-O land-body tail from an already prepared G state."""
    result = compute_land_body_hij(
        state,
        tables,
        opts,
        include_checkpoint_diagnostics=False,
    )
    return _tail_output_from_mapping(result.tail)


def _tail_output_from_mapping(tail: Mapping[str, torch.Tensor]) -> LandBodyTailOutput:
    return LandBodyTailOutput(*(tail[field] for field in LAND_BODY_TAIL_OUTPUT_FIELDS))


def run_land_body_tail_output(
    case_data: Mapping[str, Any],
    g_outputs: Mapping[str, torch.Tensor],
    tables: NoahMpTableBundle,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    expand: bool = True,
    driver_options: Mapping[str, Any] | None = None,
) -> LandBodyTailOutput:
    """Return only the K-O production tail as a flat tuple."""
    state = state_from_case_after_g(case_data, g_outputs, tables, device=device, dtype=dtype)
    opts = case_data["driver_options"] if driver_options is None else driver_options
    if not expand and _active_land_indices_cover_all_columns(case_data, tables):
        result = compute_land_body_hij(
            state,
            tables,
            opts,
            include_checkpoint_diagnostics=False,
        )
        return _tail_output_from_mapping(result.tail)
    result = compute_land_body_hij(state, tables, opts)
    return _tail_output_from_mapping(expand_land_body_tail_state(case_data, state, result, device=device))


def run_land_body_hij(
    case_data: Mapping[str, Any],
    g_outputs: Mapping[str, torch.Tensor],
    tables: NoahMpTableBundle,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Convenience adapter for later driver integration after boundary G."""
    emissions, _tail_state = run_land_body_hij_with_tail_state(
        case_data,
        g_outputs,
        tables,
        device=device,
        dtype=dtype,
    )
    return emissions


__all__ = [
    "H_SCALAR_FIELDS",
    "I_SCALAR_FIELDS",
    "J_SCALAR_FIELDS",
    "LAND_BODY_SUPPORTED_OPTIONS",
    "LAND_BODY_TAIL_OUTPUT_FIELDS",
    "LandBodyHIJ",
    "LandBodyState",
    "LandBodyTailOutput",
    "compute_land_body_hij",
    "expand_land_body_tail_state",
    "run_land_body_hij",
    "run_land_body_hij_with_tail_state",
    "run_land_body_tail_output",
    "run_land_body_tail_output_from_state",
    "run_land_body_tail_state",
    "state_from_case_after_g",
]
