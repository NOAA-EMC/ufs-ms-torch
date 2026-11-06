# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""PyTorch-native RRTMGP gas-optics kernels.

This module ports the RRTMGP table-reduction and longwave gas/source
calculation surface used by ``mo_gas_optics_rrtmgp``.  The implementation is
intentionally direct: indexes remain Fortran-like internally and coefficient
tables are loaded once by :mod:`rrtmgp_radiation_torch.tables`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .compile import run_compile_island
from .tables import RrtmgpTableBundle, RrtmgpTableFile, load_rrtmgp_tables


DEFAULT_ACTIVE_GASES = ("h2o", "co2", "o3", "n2o", "ch4", "o2")
GAS_FIELD_NAMES = {
    "h2o": "vmr_h2o",
    "co2": "vmr_co2",
    "o3": "vmr_o3",
    "n2o": "vmr_n2o",
    "ch4": "vmr_ch4",
    "o2": "vmr_o2",
}

AVOGADRO = 6.02214076e23
M_DRY = 0.028964
M_H2O = 0.018016
GRAV = 9.80665
_MinorTauRecord = tuple[int, int, int, bool, int, bool, int, int]
_REDUCED_GAS_TABLE_CACHE: dict[tuple[str, str, str, torch.dtype, tuple[str, ...], str], "ReducedGasOpticsTable"] = {}
_GAS_TABLE_LAYOUT = "gpoint_contiguous"
_MINOR_GAS_COMPILE_MODE = "joint"
_LW_GAS_COMPILE_MODE = "joint"


def set_lw_gas_compile_mode(mode: str) -> str:
    """Select joint LW compilation or only its immutable corner gathers.

    Corner mode keeps interpolation-plan, weight/scale adjoints and ordered
    floating accumulation in native Torch. Defaults remain unchanged.
    """
    if mode not in {"joint", "corner_gathers"}:
        raise ValueError(f"unknown LW-gas compile mode {mode!r}")
    global _LW_GAS_COMPILE_MODE
    previous = _LW_GAS_COMPILE_MODE
    _LW_GAS_COMPILE_MODE = mode
    return previous


def set_minor_gas_compile_mode(mode: str) -> str:
    """Select the existing joint island or pointwise record interpolation.

    Record mode leaves density scaling, broadcasting adjoints and ordered record
    accumulation in native eager Torch. Only interpolation is compiled. The
    default remains joint until whole-call numerical/performance qualification.
    """

    if mode not in {"joint", "record_interpolation"}:
        raise ValueError(f"unknown minor-gas compile mode {mode!r}")
    global _MINOR_GAS_COMPILE_MODE
    previous = _MINOR_GAS_COMPILE_MODE
    _MINOR_GAS_COMPILE_MODE = mode
    return previous


def set_gas_table_layout(layout: str) -> str:
    """Select an experimental immutable coefficient layout; return the old policy.

    Logical table dimensions, coefficient values and interpolation order stay
    unchanged.  Reduced tables are cached separately by layout.  CUDA locality
    and whole-call numerical/performance qualification are required before
    changing the default.
    """

    if layout not in {"gpoint_contiguous", "gpoint_major"}:
        raise ValueError(f"unknown gas coefficient layout {layout!r}")
    global _GAS_TABLE_LAYOUT
    previous = _GAS_TABLE_LAYOUT
    _GAS_TABLE_LAYOUT = layout
    return previous


def _prepare_coefficient_layout(value: torch.Tensor, layout: str) -> torch.Tensor:
    """Prepare spectral blocks once, retaining the existing logical index API."""

    if layout == "gpoint_contiguous":
        return value
    if layout != "gpoint_major":
        raise ValueError(f"unknown gas coefficient layout {layout!r}")
    # Each spectral point's temperature/eta/pressure neighborhood becomes
    # contiguous.  Restore logical axes as a view, so science keeps the same
    # coefficient indexing and eight-corner association.  No column sorting or
    # expanded per-column coefficient table is involved.
    return value.movedim(-1, 0).contiguous().movedim(0, -1)


@dataclass(frozen=True)
class ReducedGasOpticsTable:
    """Frozen gas table reduced to the active host gases."""

    gas_names: tuple[str, ...]
    band_lims_gpt: torch.Tensor
    gpt_to_band: torch.Tensor
    key_species: torch.Tensor
    flavor: torch.Tensor
    gpoint_flavor: torch.Tensor
    vmr_ref: torch.Tensor
    press_ref: torch.Tensor
    temp_ref: torch.Tensor
    press_ref_trop: torch.Tensor
    kmajor: torch.Tensor
    kminor_lower: torch.Tensor
    kminor_upper: torch.Tensor
    minor_limits_gpt_lower: torch.Tensor
    minor_limits_gpt_upper: torch.Tensor
    minor_scales_with_density_lower: torch.Tensor
    minor_scales_with_density_upper: torch.Tensor
    scale_by_complement_lower: torch.Tensor
    scale_by_complement_upper: torch.Tensor
    idx_minor_lower: torch.Tensor
    idx_minor_upper: torch.Tensor
    idx_minor_scaling_lower: torch.Tensor
    idx_minor_scaling_upper: torch.Tensor
    kminor_start_lower: torch.Tensor
    kminor_start_upper: torch.Tensor
    planck_fraction: torch.Tensor | None = None
    totplnk: torch.Tensor | None = None
    optimal_angle_fit: torch.Tensor | None = None
    rayl_lower: torch.Tensor | None = None
    rayl_upper: torch.Tensor | None = None
    solar_source_quiet: torch.Tensor | None = None
    solar_source_facular: torch.Tensor | None = None
    solar_source_sunspot: torch.Tensor | None = None
    mg_default: torch.Tensor | None = None
    sb_default: torch.Tensor | None = None
    tsi_default: torch.Tensor | None = None
    minor_records_lower: tuple[_MinorTauRecord, ...] = ()
    minor_records_upper: tuple[_MinorTauRecord, ...] = ()

    @property
    def ngpt(self) -> int:
        return int(self.gpt_to_band.numel())

    @property
    def nlay_atmos(self) -> int:
        return 2

    @property
    def nband(self) -> int:
        return int(self.band_lims_gpt.shape[0])

    @property
    def neta(self) -> int:
        return int(self.kmajor.shape[1])

    @property
    def npres(self) -> int:
        return int(self.press_ref.numel())

    @property
    def ntemp(self) -> int:
        return int(self.temp_ref.numel())


@dataclass(frozen=True)
class LongwaveGasOpticsResult:
    """Longwave gas optical depth and Planck source tensors."""

    tau: torch.Tensor
    lay_source: torch.Tensor
    lev_source_inc: torch.Tensor
    lev_source_dec: torch.Tensor
    sfc_source: torch.Tensor
    sfc_source_jac: torch.Tensor
    col_dry: torch.Tensor | None
    gpt_to_band: torch.Tensor
    optimal_angle_fit: torch.Tensor | None
    diffusivity_angle: torch.Tensor | None
    runtime_kind: str = "torch_rrtmgp_longwave_gas_optics"

    @property
    def lev_source(self) -> torch.Tensor:
        """Compatibility interface-source tensor derived from directional sources."""

        ngpt, nlay, ncol = self.lev_source_inc.shape
        result = torch.empty((ngpt, nlay + 1, ncol), dtype=self.lev_source_inc.dtype, device=self.lev_source_inc.device)
        result[:, 0, :] = self.lev_source_dec[:, 0, :]
        result[:, 1:, :] = self.lev_source_inc
        return result


@dataclass(frozen=True)
class ShortwaveGasOpticsResult:
    """Shortwave gas/Rayleigh optical properties and TOA solar source."""

    tau: torch.Tensor
    ssa: torch.Tensor
    g: torch.Tensor
    tau_absorption: torch.Tensor | None
    tau_rayleigh: torch.Tensor | None
    toa_src: torch.Tensor
    col_dry: torch.Tensor | None
    gpt_to_band: torch.Tensor
    runtime_kind: str = "torch_rrtmgp_shortwave_gas_optics"


@dataclass(frozen=True)
class _GasInterpolationPlan:
    itropo: torch.Tensor
    jtemp: torch.Tensor
    jpress_arg: torch.Tensor
    jeta: torch.Tensor
    fmajor: torch.Tensor
    fminor: torch.Tensor
    col_mix: torch.Tensor
    flavor: torch.Tensor | None = None  # the table's gpoint_flavor [2, ngpt]


def _as_real(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64 (the working dtype), got {value.dtype}")
    return value


def _device_of(table: RrtmgpTableFile) -> torch.device:
    for tensor in table.tensors.values():
        return tensor.device
    return torch.device("cpu")


def _table_tensor(table: RrtmgpTableFile, name: str) -> torch.Tensor:
    try:
        return table.tensors[name]
    except KeyError as exc:
        raise KeyError(f"frozen RRTMGP table {table.path.name} lacks {name}") from exc


def _table_strings(table: RrtmgpTableFile, name: str) -> tuple[str, ...]:
    try:
        return tuple(table.strings[name])
    except KeyError as exc:
        raise KeyError(f"frozen RRTMGP table {table.path.name} lacks {name}") from exc


def _gpt_to_band_from_limits(band_lims_gpt: torch.Tensor) -> torch.Tensor:
    limits = band_lims_gpt.to(dtype=torch.int64)
    counts = limits[:, 1] - limits[:, 0] + 1
    bands = torch.arange(limits.shape[0], dtype=torch.int64, device=limits.device)
    return torch.repeat_interleave(bands, counts)


def _rewrite_key_pair(pair: Sequence[int]) -> tuple[int, int]:
    first, second = int(pair[0]), int(pair[1])
    if first == 0 and second == 0:
        return (2, 2)
    return (first, second)


def _reduce_key_species(
    key_species: torch.Tensor,
    *,
    table_gas_names: Sequence[str],
    active_gases: Sequence[str],
) -> torch.Tensor:
    active_index = {name: idx + 1 for idx, name in enumerate(active_gases)}
    reduced = torch.zeros_like(key_species, dtype=torch.int64)
    key_species_host = key_species.detach().to(device="cpu", dtype=torch.int64)
    for iband in range(key_species.shape[0]):
        for iatm in range(key_species.shape[1]):
            for ipair in range(key_species.shape[2]):
                raw_index = int(key_species_host[iband, iatm, ipair].item())
                if raw_index == 0:
                    continue
                gas = table_gas_names[raw_index - 1].strip().lower()
                if gas not in active_index:
                    raise ValueError(f"required RRTMGP key gas {gas!r} is not active")
                reduced[iband, iatm, ipair] = active_index[gas]
    return reduced


def _create_flavors(key_species: torch.Tensor) -> torch.Tensor:
    key_species_host = key_species.detach().to(device="cpu", dtype=torch.int64)
    flavors: list[tuple[int, int]] = []
    for iband in range(key_species.shape[0]):
        for iatm in range(key_species.shape[1]):
            pair = _rewrite_key_pair(key_species_host[iband, iatm, :].tolist())
            if pair not in flavors:
                flavors.append(pair)
    return torch.tensor(flavors, dtype=torch.int64, device=key_species.device)


def _create_gpoint_flavor(
    key_species: torch.Tensor,
    gpt_to_band: torch.Tensor,
    flavor: torch.Tensor,
) -> torch.Tensor:
    key_species_host = key_species.detach().to(device="cpu", dtype=torch.int64)
    gpt_to_band_host = gpt_to_band.detach().to(device="cpu", dtype=torch.int64)
    flavor_host = flavor.detach().to(device="cpu", dtype=torch.int64)
    lookup = {tuple(pair.tolist()): idx for idx, pair in enumerate(flavor_host)}
    result = torch.empty((2, gpt_to_band.numel()), dtype=torch.int64, device=key_species.device)
    for igpt, iband in enumerate(gpt_to_band_host.tolist()):
        for iatm in range(2):
            pair = _rewrite_key_pair(key_species_host[int(iband), iatm, :].tolist())
            result[iatm, igpt] = lookup[pair]
    return result


def _reduce_vmr_ref(
    table: RrtmgpTableFile,
    *,
    active_gases: Sequence[str],
) -> torch.Tensor:
    gas_names = _table_strings(table, "gas_names")
    original_index = {name.strip().lower(): idx + 1 for idx, name in enumerate(gas_names)}
    raw = _table_tensor(table, "vmr_ref")
    device = raw.device
    ntemp = raw.shape[0]
    reduced = torch.empty((2, len(active_gases) + 1, ntemp), dtype=raw.dtype, device=device)
    reduced[:, 0, :] = raw[:, 0, :].permute(1, 0)
    for idx, gas in enumerate(active_gases, start=1):
        try:
            source_index = original_index[gas]
        except KeyError as exc:
            raise ValueError(f"active gas {gas!r} is not present in frozen RRTMGP table") from exc
        reduced[:, idx, :] = raw[:, source_index, :].permute(1, 0)
    return reduced


def _minor_gas_source(
    *,
    identifier: str,
    gas_minor: Sequence[str],
    identifier_minor: Sequence[str],
) -> str:
    try:
        index = [item.strip().lower() for item in identifier_minor].index(identifier.strip().lower())
    except ValueError as exc:
        raise ValueError(f"minor gas identifier {identifier!r} missing from table identifier list") from exc
    return gas_minor[index].strip().lower()


def _reduce_minor_arrays(
    table: RrtmgpTableFile,
    *,
    suffix: str,
    active_gases: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active_index = {name: idx + 1 for idx, name in enumerate(active_gases)}
    gas_minor = _table_strings(table, "gas_minor")
    identifier_minor = _table_strings(table, "identifier_minor")
    minor_gases = _table_strings(table, f"minor_gases_{suffix}")
    scaling_gases = _table_strings(table, f"scaling_gas_{suffix}")
    device = _device_of(table)
    limits_raw = _table_tensor(table, f"minor_limits_gpt_{suffix}").to(device=device, dtype=torch.int64)
    scales_raw = _table_tensor(table, f"minor_scales_with_density_{suffix}").to(device=device, dtype=torch.int64)
    complement_raw = _table_tensor(table, f"scale_by_complement_{suffix}").to(device=device, dtype=torch.int64)
    start_raw = _table_tensor(table, f"kminor_start_{suffix}").to(device=device, dtype=torch.int64)
    kminor_raw = _table_tensor(table, f"kminor_{suffix}").to(device=device)
    limits_host = limits_raw.detach().to(device="cpu", dtype=torch.int64)
    scales_host = scales_raw.detach().to(device="cpu", dtype=torch.int64)
    complement_host = complement_raw.detach().to(device="cpu", dtype=torch.int64)
    start_host = start_raw.detach().to(device="cpu", dtype=torch.int64)

    kept_coefficients: list[torch.Tensor] = []
    kept_limits: list[torch.Tensor] = []
    kept_scales: list[int] = []
    kept_complement: list[int] = []
    kept_start: list[int] = []
    kept_idx_minor: list[int] = []
    kept_idx_scaling: list[int] = []
    next_start = 1
    for imnr, identifier in enumerate(minor_gases):
        source_gas = _minor_gas_source(
            identifier=identifier,
            gas_minor=gas_minor,
            identifier_minor=identifier_minor,
        )
        gpt_start = int(limits_host[imnr, 0].item())
        gpt_end = int(limits_host[imnr, 1].item())
        count = gpt_end - gpt_start + 1
        if source_gas not in active_index:
            continue
        source_start = int(start_host[imnr].item()) - 1
        kept_coefficients.append(kminor_raw[:, :, source_start : source_start + count])
        kept_limits.append(limits_raw[imnr])
        kept_scales.append(int(scales_host[imnr].item()))
        kept_complement.append(int(complement_host[imnr].item()))
        kept_start.append(next_start)
        kept_idx_minor.append(active_index[source_gas])
        scaling_gas = scaling_gases[imnr].strip().lower()
        kept_idx_scaling.append(active_index.get(scaling_gas, -1))
        next_start += count

    if kept_coefficients:
        kminor = torch.cat(kept_coefficients, dim=2)
        limits = torch.stack(kept_limits).to(device=device, dtype=torch.int64)
    else:
        kminor = kminor_raw[:, :, :0]
        limits = torch.empty((0, 2), dtype=torch.int64, device=device)

    return (
        kminor,
        limits,
        torch.tensor(kept_scales, dtype=torch.int64, device=device),
        torch.tensor(kept_complement, dtype=torch.int64, device=device),
        torch.tensor(kept_idx_minor, dtype=torch.int64, device=device),
        torch.tensor(kept_idx_scaling, dtype=torch.int64, device=device),
        torch.tensor(kept_start, dtype=torch.int64, device=device),
    )


def _minor_tau_records(
    *,
    limits: torch.Tensor,
    scales: torch.Tensor,
    complement: torch.Tensor,
    idx_minor: torch.Tensor,
    idx_scaling: torch.Tensor,
    starts: torch.Tensor,
    gpt_flavor: torch.Tensor,
) -> tuple[_MinorTauRecord, ...]:
    records: list[_MinorTauRecord] = []
    limits_host = limits.detach().to(device="cpu", dtype=torch.int64)
    scales_host = scales.detach().to(device="cpu", dtype=torch.int64)
    complement_host = complement.detach().to(device="cpu", dtype=torch.int64)
    idx_minor_host = idx_minor.detach().to(device="cpu", dtype=torch.int64)
    idx_scaling_host = idx_scaling.detach().to(device="cpu", dtype=torch.int64)
    starts_host = starts.detach().to(device="cpu", dtype=torch.int64)
    gpt_flavor_host = gpt_flavor.detach().to(device="cpu", dtype=torch.int64)
    for imnr in range(int(limits.shape[0])):
        gpt_start = int(limits_host[imnr, 0].item()) - 1
        gpt_end = int(limits_host[imnr, 1].item())
        records.append(
            (
                gpt_start,
                gpt_end,
                int(idx_minor_host[imnr].item()),
                bool(int(scales_host[imnr].item())),
                int(idx_scaling_host[imnr].item()),
                bool(int(complement_host[imnr].item())),
                int(gpt_flavor_host[gpt_start].item()),
                int(starts_host[imnr].item()) - 1,
            )
        )
    return tuple(records)


@torch.inference_mode(False)
def reduce_gas_optics_table(
    table: RrtmgpTableFile,
    *,
    active_gases: Sequence[str] = DEFAULT_ACTIVE_GASES,
) -> ReducedGasOpticsTable:
    """Reduce frozen coefficients, retaining tensors usable by later backward calls."""

    active = tuple(name.strip().lower() for name in active_gases)
    device = _device_of(table)
    layout = _GAS_TABLE_LAYOUT
    cache_key = (str(table.path), table.sha256, str(device), _table_tensor(table, "kmajor").dtype, active, layout)
    cached = _REDUCED_GAS_TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    gas_names = tuple(name.strip().lower() for name in _table_strings(table, "gas_names"))
    band_lims = _table_tensor(table, "bnd_limits_gpt").to(device=device, dtype=torch.int64)
    gpt_to_band = _gpt_to_band_from_limits(band_lims)
    key_species = _reduce_key_species(
        _table_tensor(table, "key_species").to(device=device, dtype=torch.int64),
        table_gas_names=gas_names,
        active_gases=active,
    )
    flavor = _create_flavors(key_species)
    gpoint_flavor = _create_gpoint_flavor(key_species, gpt_to_band, flavor)
    (
        kminor_lower,
        minor_limits_lower,
        minor_scales_lower,
        complement_lower,
        idx_minor_lower,
        idx_scaling_lower,
        kminor_start_lower,
    ) = _reduce_minor_arrays(table, suffix="lower", active_gases=active)
    (
        kminor_upper,
        minor_limits_upper,
        minor_scales_upper,
        complement_upper,
        idx_minor_upper,
        idx_scaling_upper,
        kminor_start_upper,
    ) = _reduce_minor_arrays(table, suffix="upper", active_gases=active)

    kmajor = _table_tensor(table, "kmajor").to(device=device).permute(0, 2, 1, 3).contiguous()
    planck_raw = table.tensors.get("plank_fraction")
    planck_fraction = (
        planck_raw.to(device=device).permute(0, 2, 1, 3).contiguous()
        if planck_raw is not None
        else None
    )
    totplnk_raw = table.tensors.get("totplnk")
    totplnk = totplnk_raw.to(device=device).t().contiguous() if totplnk_raw is not None else None
    fit = table.tensors.get("optimal_angle_fit")
    optimal_angle_fit = fit.to(device=device).t().contiguous() if fit is not None else None
    rayl_lower_raw = table.tensors.get("rayl_lower")
    rayl_upper_raw = table.tensors.get("rayl_upper")
    solar_quiet = table.tensors.get("solar_source_quiet")
    solar_facular = table.tensors.get("solar_source_facular")
    solar_sunspot = table.tensors.get("solar_source_sunspot")
    minor_records_lower = _minor_tau_records(
        limits=minor_limits_lower,
        scales=minor_scales_lower,
        complement=complement_lower,
        idx_minor=idx_minor_lower,
        idx_scaling=idx_scaling_lower,
        starts=kminor_start_lower,
        gpt_flavor=gpoint_flavor[0],
    )
    minor_records_upper = _minor_tau_records(
        limits=minor_limits_upper,
        scales=minor_scales_upper,
        complement=complement_upper,
        idx_minor=idx_minor_upper,
        idx_scaling=idx_scaling_upper,
        starts=kminor_start_upper,
        gpt_flavor=gpoint_flavor[1],
    )

    result = ReducedGasOpticsTable(
        gas_names=active,
        band_lims_gpt=band_lims,
        gpt_to_band=gpt_to_band,
        key_species=key_species,
        flavor=flavor,
        gpoint_flavor=gpoint_flavor,
        vmr_ref=_reduce_vmr_ref(table, active_gases=active).to(device=device),
        press_ref=_table_tensor(table, "press_ref").to(device=device),
        temp_ref=_table_tensor(table, "temp_ref").to(device=device),
        press_ref_trop=_table_tensor(table, "press_ref_trop").to(device=device),
        kmajor=_prepare_coefficient_layout(kmajor, layout),
        kminor_lower=_prepare_coefficient_layout(kminor_lower, layout),
        kminor_upper=_prepare_coefficient_layout(kminor_upper, layout),
        minor_limits_gpt_lower=minor_limits_lower,
        minor_limits_gpt_upper=minor_limits_upper,
        minor_scales_with_density_lower=minor_scales_lower,
        minor_scales_with_density_upper=minor_scales_upper,
        scale_by_complement_lower=complement_lower,
        scale_by_complement_upper=complement_upper,
        idx_minor_lower=idx_minor_lower,
        idx_minor_upper=idx_minor_upper,
        idx_minor_scaling_lower=idx_scaling_lower,
        idx_minor_scaling_upper=idx_scaling_upper,
        kminor_start_lower=kminor_start_lower,
        kminor_start_upper=kminor_start_upper,
        planck_fraction=_prepare_coefficient_layout(planck_fraction, layout) if planck_fraction is not None else None,
        totplnk=totplnk,
        optimal_angle_fit=optimal_angle_fit,
        rayl_lower=_prepare_coefficient_layout(rayl_lower_raw.to(device=device).contiguous(), layout) if rayl_lower_raw is not None else None,
        rayl_upper=_prepare_coefficient_layout(rayl_upper_raw.to(device=device).contiguous(), layout) if rayl_upper_raw is not None else None,
        solar_source_quiet=solar_quiet.to(device=device) if solar_quiet is not None else None,
        solar_source_facular=solar_facular.to(device=device) if solar_facular is not None else None,
        solar_source_sunspot=solar_sunspot.to(device=device) if solar_sunspot is not None else None,
        mg_default=table.tensors.get("mg_default").to(device=device) if table.tensors.get("mg_default") is not None else None,
        sb_default=table.tensors.get("sb_default").to(device=device) if table.tensors.get("sb_default") is not None else None,
        tsi_default=table.tensors.get("tsi_default").to(device=device) if table.tensors.get("tsi_default") is not None else None,
        minor_records_lower=minor_records_lower,
        minor_records_upper=minor_records_upper,
    )
    _REDUCED_GAS_TABLE_CACHE[cache_key] = result
    return result


def _compute_col_dry(p_lev: torch.Tensor, vmr_h2o: torch.Tensor) -> torch.Tensor:
    delta_p = torch.abs(p_lev[1:, :] - p_lev[:-1, :])
    denominator = 10000.0 * (M_DRY + M_H2O * vmr_h2o) * GRAV
    return delta_p * AVOGADRO / denominator


def _vmr_from_boundaries(
    fields: Mapping[str, torch.Tensor | str],
    *,
    active_gases: Sequence[str],
    shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    nlay, ncol = shape
    layers = [torch.ones((nlay, ncol), dtype=dtype, device=device)]
    for idx, gas in enumerate(active_gases, start=1):
        name = GAS_FIELD_NAMES.get(gas)
        if name is None or name not in fields:
            layers.append(torch.zeros((nlay, ncol), dtype=dtype, device=device))
            continue
        layers.append(_as_real(fields[name], name=name).to(device=device))
    return torch.stack(layers, dim=0)


def _interpolation_coordinates_tensor(
    play: torch.Tensor,
    tlay: torch.Tensor,
    temp_ref: torch.Tensor,
    press_ref: torch.Tensor,
    press_ref_trop: torch.Tensor,
    *,
    ntemp: int,
    npres: int,
) -> tuple[torch.Tensor, ...]:
    """Column/layer coordinates, shared by every spectral point."""
    temp_delta = (temp_ref[-1] - temp_ref[0]) / float(ntemp - 1)
    press_delta = (torch.log(press_ref[-1]) - torch.log(press_ref[0])) / float(npres - 1)
    jtemp_float = torch.floor((tlay - (temp_ref[0] - temp_delta)) / temp_delta)
    jtemp_float = torch.clamp(jtemp_float, 1, ntemp - 1)
    jtemp = jtemp_float.to(dtype=torch.int64)
    ftemp = (tlay - temp_ref.index_select(0, (jtemp - 1).reshape(-1)).reshape_as(tlay)) / temp_delta
    locpress = 1.0 + (torch.log(play) - torch.log(press_ref[0])) / press_delta
    jpress_float = torch.trunc(locpress)
    jpress_clamped = torch.clamp(jpress_float, 1, npres - 1)
    jpress = jpress_clamped.to(dtype=torch.int64)
    fpress = locpress - jpress_clamped
    itropo = (play <= press_ref_trop).to(dtype=torch.int64)
    jpress_arg = jpress + itropo + 1
    return jtemp, ftemp, itropo, jpress_arg, fpress


def _build_interpolation_plan_tensor(
    play: torch.Tensor,
    tlay: torch.Tensor,
    col_gas: torch.Tensor,
    temp_ref: torch.Tensor,
    press_ref: torch.Tensor,
    press_ref_trop: torch.Tensor,
    gas1: torch.Tensor,
    gas2: torch.Tensor,
    vmr_ref: torch.Tensor,
    gpoint_flavor: torch.Tensor,
    tiny: torch.Tensor,
    *,
    ntemp: int,
    npres: int,
    neta: int,
    coordinates: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Build all major/minor interpolation coordinates from flat tensors."""

    if coordinates is None:
        coordinates = _interpolation_coordinates_tensor(
            play, tlay, temp_ref, press_ref, press_ref_trop, ntemp=ntemp, npres=npres,
        )
    jtemp, ftemp, itropo, jpress_arg, fpress = coordinates
    flavor = gpoint_flavor  # the static [2, ngpt] table: no per-element flavor index

    col1 = col_gas.index_select(0, gas1)
    col2 = col_gas.index_select(0, gas2)
    itropo_i = itropo.unsqueeze(0)
    gas1_i = gas1.reshape(-1, 1, 1)
    gas2_i = gas2.reshape(-1, 1, 1)

    fminor_by_temp: list[torch.Tensor] = []
    fmajor_by_temp: list[torch.Tensor] = []
    jeta_by_temp: list[torch.Tensor] = []
    col_mix_by_temp: list[torch.Tensor] = []
    for itemp in range(2):
        ref_temp_index = (jtemp + itemp - 1).unsqueeze(0)
        ratio = vmr_ref[itropo_i, gas1_i, ref_temp_index] / vmr_ref[itropo_i, gas2_i, ref_temp_index]
        mix = col1 + ratio * col2
        col_mix_by_temp.append(mix)
        eta = torch.where(mix > tiny, col1 / torch.where(mix > tiny, mix, 1.0), torch.full_like(mix, 0.5))
        loceta = eta * float(neta - 1)
        trunc_eta = torch.trunc(loceta)
        jeta_value = torch.minimum(
            trunc_eta.to(dtype=torch.int64) + 1,
            torch.full_like(trunc_eta.to(dtype=torch.int64), neta - 1),
        )
        jeta_by_temp.append(jeta_value)
        feta = loceta - trunc_eta
        ftemp_term = ((1.0 - ftemp) if itemp == 0 else ftemp).unsqueeze(0)
        fminor_eta0 = (1.0 - feta) * ftemp_term
        fminor_eta1 = feta * ftemp_term
        fminor_temp = torch.stack((fminor_eta0, fminor_eta1), dim=1)
        fminor_by_temp.append(fminor_temp)
        p0 = (1.0 - fpress).unsqueeze(0)
        p1 = fpress.unsqueeze(0)
        fmajor_by_temp.append(
            torch.stack(
                (
                    torch.stack((p0 * fminor_eta0, p1 * fminor_eta0), dim=1),
                    torch.stack((p0 * fminor_eta1, p1 * fminor_eta1), dim=1),
                ),
                dim=1,
            )
        )
    return (
        itropo,
        jtemp,
        jpress_arg,
        torch.stack(jeta_by_temp, dim=1),
        torch.stack(fmajor_by_temp, dim=3),
        torch.stack(fminor_by_temp, dim=2),
        torch.stack(col_mix_by_temp, dim=1),
        flavor,
    )


def _build_interpolation_plan(
    table: ReducedGasOpticsTable,
    *,
    play: torch.Tensor,
    tlay: torch.Tensor,
    col_gas: torch.Tensor,
    coordinates: tuple[torch.Tensor, ...] | None = None,
) -> _GasInterpolationPlan:
    temp_ref = table.temp_ref.to(device=play.device, dtype=play.dtype)
    press_ref = table.press_ref.to(device=play.device, dtype=play.dtype)
    press_ref_trop = table.press_ref_trop.to(device=play.device, dtype=play.dtype)
    gas1 = table.flavor[:, 0].to(device=play.device, dtype=torch.int64)
    gas2 = table.flavor[:, 1].to(device=play.device, dtype=torch.int64)
    vmr_ref = table.vmr_ref.to(device=play.device, dtype=play.dtype)
    tiny = torch.full((), 2.0 * torch.finfo(play.dtype).tiny, dtype=play.dtype, device=play.device)
    arguments = (
        play,
        tlay,
        col_gas,
        temp_ref,
        press_ref,
        press_ref_trop,
        gas1,
        gas2,
        vmr_ref,
        table.gpoint_flavor.to(device=play.device, dtype=torch.int64),
        tiny,
    )
    options = dict(ntemp=table.ntemp, npres=table.npres, neta=table.neta, coordinates=coordinates)
    values = (_build_interpolation_plan_tensor(*arguments, **options)
              if _LW_GAS_COMPILE_MODE == "corner_gathers" else run_compile_island(
                  "gas_optics", "interpolation_plan", _build_interpolation_plan_tensor,
                  *arguments, **options))
    return _GasInterpolationPlan(
        itropo=values[0],
        jtemp=values[1],
        jpress_arg=values[2],
        jeta=values[3],
        fmajor=values[4],
        fminor=values[5],
        col_mix=values[6],
        flavor=values[7],
    )


def _flavor_indices(gpoint_flavor: torch.Tensor, itropo: torch.Tensor) -> torch.Tensor:
    """One shared interpolation index, instead of two spectral gathers per operand."""
    return torch.where(itropo.unsqueeze(0) != 0,
                       gpoint_flavor[1, :, None, None], gpoint_flavor[0, :, None, None])


def _gather_flavor(values: torch.Tensor, gpoint_flavor: torch.Tensor, itropo: torch.Tensor,
                   *, indices: torch.Tensor | None = None) -> torch.Tensor:
    """Select flavors from the original broadcast layout, retaining its VJP reduction order."""
    ngpt = gpoint_flavor.shape[1]
    expanded = values.unsqueeze(0).expand(ngpt, -1, -1, -1)
    if indices is None:
        indices = _flavor_indices(gpoint_flavor, itropo)
    return torch.gather(expanded, 1, indices.unsqueeze(1)).squeeze(1)


def _gpoint_flavor(table: ReducedGasOpticsTable, plan: _GasInterpolationPlan) -> torch.Tensor:
    if plan.flavor is not None:
        return plan.flavor
    return table.gpoint_flavor.to(device=plan.itropo.device, dtype=torch.int64)


def _interp3d_all_gpoints_tensor(
    coeff_t: torch.Tensor,
    flavor: torch.Tensor,
    itropo: torch.Tensor,
    jtemp: torch.Tensor,
    jpress_arg: torch.Tensor,
    jeta: torch.Tensor,
    fmajor: torch.Tensor,
    col_mix: torch.Tensor,
    *,
    scale_by_col_mix: bool,
) -> torch.Tensor:
    ngpt = coeff_t.shape[-1]
    nlay, ncol = jtemp.shape
    gpt_index = torch.arange(ngpt, dtype=torch.int64, device=jtemp.device).reshape(ngpt, 1, 1)
    p_indices = (jpress_arg - 2, jpress_arg - 1)
    t_indices = (jtemp - 1, jtemp)
    # Backward saves gather indices: share that storage under AD. In inference
    # each gather releases its index before the larger coefficient temporaries.
    flavor_indices = (_flavor_indices(flavor, itropo) if torch.is_grad_enabled()
                      and (fmajor.requires_grad or (scale_by_col_mix and col_mix.requires_grad)) else None)
    # interpolate3D_byflav sums eta first, then pressure, before scaling: ((c0 + c1) + c2) + c3,
    # accumulated as each corner is formed so no corner outlives its addition. One eta gather
    # per temperature (the integer eta - 1 + slot gathered equals the gathered eta, minus 1 + slot).
    total = None
    for itemp, t_index in enumerate(t_indices):
        t_index_gpt = t_index.unsqueeze(0)
        eta_dtype = torch.int32 if coeff_t.shape[1] <= 2**31 else torch.int64
        eta_hi = _gather_flavor(jeta[:, itemp, :, :].to(dtype=eta_dtype),
                                flavor, itropo, indices=flavor_indices)
        subtotal = None
        for press_slot, p_index in enumerate(p_indices):
            p_index_gpt = p_index.unsqueeze(0)
            for eta_slot in range(2):
                # The integer gather owns its workspace. Restore it after each pair.
                eta_index = eta_hi.add_(-1 if eta_slot == 0 else 1)
                weight = _gather_flavor(fmajor[:, eta_slot, press_slot, itemp, :, :], flavor, itropo,
                                        indices=flavor_indices)
                corner = weight * _coefficient_corner_tensor(
                    coeff_t, t_index_gpt, eta_index, p_index_gpt, gpt_index)
                del weight, eta_index
                subtotal = corner if subtotal is None else subtotal + corner
                del corner
        del eta_hi
        scale = (_gather_flavor(col_mix[:, itemp, :, :], flavor, itropo, indices=flavor_indices)
                 if scale_by_col_mix else 1.0)
        scaled = scale * subtotal
        del scale, subtotal
        total = scaled if total is None else total + scaled
        del scaled
    return total


def _interp3d_all_gpoints(
    coeff: torch.Tensor,
    table: ReducedGasOpticsTable,
    plan: _GasInterpolationPlan,
    *,
    scale_by_col_mix: bool,
) -> torch.Tensor:
    coeff_t = coeff.to(device=plan.itropo.device)
    return run_compile_island(
        "gas_optics",
        "interp3d_all_gpoints",
        _interp3d_all_gpoints_tensor,
        coeff_t,
        _gpoint_flavor(table, plan),
        plan.itropo,
        plan.jtemp,
        plan.jpress_arg,
        plan.jeta,
        plan.fmajor,
        plan.col_mix,
        scale_by_col_mix=scale_by_col_mix,
    )


def _coefficient_corner_tensor(
    coefficient: torch.Tensor,
    temperature: torch.Tensor, eta: torch.Tensor,
    pressure: torch.Tensor, gpoint: torch.Tensor,
    *, linear_index: torch.Tensor | None = None,
) -> torch.Tensor:
    if torch.compiler.is_compiling():
        # Inductor fuses these broadcast addresses. Explicit linear indices
        # instead become retained full-spectral buffers at shared corners.
        return coefficient[temperature, eta, pressure, gpoint]
    if linear_index is not None:
        return coefficient.reshape(-1).index_select(0, linear_index.reshape(-1)).reshape_as(linear_index)
    # CUDA advanced indexing copies differently-strided broadcast indices.
    # One owned linear index avoids three full spectral index copies.
    npres, ngpt = coefficient.shape[2:]
    index_dtype = torch.int32 if coefficient.numel() <= 2**31 else torch.int64
    # Cast the compact operands before arithmetic. Casting only the destination
    # lets Inductor retain full spectral int64 intermediates for every corner.
    temperature = temperature.to(dtype=index_dtype)
    pressure = pressure.to(dtype=index_dtype)
    gpoint = gpoint.to(dtype=index_dtype)
    offset = (temperature * (coefficient.shape[1] * npres) + pressure) * ngpt
    index = eta.to(dtype=index_dtype, copy=True)
    index.mul_(npres * ngpt).add_(offset).add_(gpoint)
    return coefficient.reshape(-1).index_select(0, index.reshape(-1)).reshape_as(index)


def _interp3d_lw_major_planck_tensor(
    kmajor_t: torch.Tensor, planck_fraction_t: torch.Tensor,
    flavor: torch.Tensor, itropo: torch.Tensor, jtemp: torch.Tensor,
    jpress_arg: torch.Tensor, jeta: torch.Tensor,
    fmajor: torch.Tensor, col_mix: torch.Tensor,
    *, gpt_range: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _interp3d_lw_major_planck_body(
        kmajor_t, planck_fraction_t, flavor, itropo, jtemp, jpress_arg,
        jeta, fmajor, col_mix, gpt_range=gpt_range,
    )


def _interp3d_lw_major_planck_body(
    kmajor_t: torch.Tensor,
    planck_fraction_t: torch.Tensor,
    flavor: torch.Tensor,
    itropo: torch.Tensor,
    jtemp: torch.Tensor,
    jpress_arg: torch.Tensor,
    jeta: torch.Tensor,
    fmajor: torch.Tensor,
    col_mix: torch.Tensor,
    *,
    compile_gathers: bool = False,
    gpt_range: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate LW major absorption and Planck fractions together.

    Both tables use the same eight interpolation corners.  Keeping the two
    accumulators in one tensor island shares all flavor, eta, pressure, and
    temperature index work without changing either accumulator's addition
    order.
    """

    table_ngpt = kmajor_t.shape[-1]
    start, end = (0, table_ngpt) if gpt_range is None else gpt_range
    ngpt = end - start
    nlay, ncol = jtemp.shape
    gpt_index = torch.arange(start, end, dtype=torch.int64, device=jtemp.device).reshape(ngpt, 1, 1)
    p_indices = (jpress_arg - 2, jpress_arg - 1)
    t_indices = (jtemp - 1, jtemp)
    flavor_indices = (_flavor_indices(flavor, itropo) if torch.is_grad_enabled()
                      and (fmajor.requires_grad or col_mix.requires_grad) else None)

    def corner(coefficient, indices, linear_index):
        if compile_gathers:
            return run_compile_island(
                "gas_optics", "lw_coefficient_corner",
                _coefficient_corner_tensor, coefficient, *indices,
            )
        return _coefficient_corner_tensor(coefficient, *indices, linear_index=linear_index)

    # Reverse mode must retain each lookup's index. Only the no-grad eager
    # path can reuse and advance an owned address buffer between corners.
    reuse_addresses = (not torch.compiler.is_compiling() and not torch.is_grad_enabled()
                       and not compile_gathers and kmajor_t.numel() <= 2**31
                       and kmajor_t.shape == planck_fraction_t.shape)

    # Each accumulator keeps ((c0 + c1) + c2) + c3 per temperature, then scale * sum, then the
    # two temperatures in order; corners are added as they are formed (no corner list).
    absorption_total = fraction_total = None
    for itemp, t_index in enumerate(t_indices):
        t_index_gpt = t_index.unsqueeze(0)
        eta_dtype = torch.int32 if kmajor_t.shape[1] <= 2**31 else torch.int64
        eta_hi = _gather_flavor(jeta[:, itemp, :, :].to(dtype=eta_dtype),
                                flavor, itropo, indices=flavor_indices)
        linear_index = None
        if reuse_addresses:
            npres = kmajor_t.shape[2]
            offset = (t_index_gpt.to(torch.int32) * (kmajor_t.shape[1] * npres)
                      + p_indices[0].unsqueeze(0).to(torch.int32)) * table_ngpt
            linear_index = eta_hi.mul_(npres * table_ngpt).add_(offset).add_(gpt_index.to(torch.int32))
            del offset
        absorption_sum = fraction_sum = None
        for press_slot, p_index in enumerate(p_indices):
            p_index_gpt = p_index.unsqueeze(0)
            for eta_slot in range(2):
                # The integer gather owns its workspace. Restore it after each pair.
                if reuse_addresses:
                    delta = (npres * table_ngpt if eta_slot
                             else (table_ngpt if press_slot else 0) - npres * table_ngpt)
                    linear_index.add_(delta)
                    eta_index = eta_hi
                else:
                    eta_index = eta_hi.add_(-1 if eta_slot == 0 else 1)
                weight = _gather_flavor(fmajor[:, eta_slot, press_slot, itemp, :, :], flavor, itropo,
                                        indices=flavor_indices)
                table_index = (t_index_gpt, eta_index, p_index_gpt, gpt_index)

                # Consume absorption before gathering the Planck fraction: the
                # two spectral coefficient arrays need not coexist.
                # Advanced indexing owns these buffers. Reuse that workspace;
                # autograd saves any original values required by the reverse pass.
                absorption = corner(kmajor_t, table_index, linear_index).mul_(weight)
                absorption_sum = absorption if absorption_sum is None else absorption_sum.add_(absorption)
                del absorption
                fraction = corner(planck_fraction_t, table_index, linear_index).mul_(weight)
                del weight, table_index, eta_index
                fraction_sum = fraction if fraction_sum is None else fraction_sum.add_(fraction)
                del fraction
        del eta_hi, linear_index
        scale = _gather_flavor(col_mix[:, itemp, :, :], flavor, itropo, indices=flavor_indices)
        absorption_sum = scale.mul_(absorption_sum)
        del scale
        absorption_total = absorption_sum if absorption_total is None else absorption_total.add_(absorption_sum)
        fraction_total = fraction_sum if fraction_total is None else fraction_total.add_(fraction_sum)
        del absorption_sum, fraction_sum
    return absorption_total, fraction_total


def _interp3d_lw_major_planck(
    table: ReducedGasOpticsTable,
    plan: _GasInterpolationPlan,
    *, gpt_range: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if table.planck_fraction is None:
        raise ValueError("longwave table does not include Planck fractions")
    arguments = (
        table.kmajor.to(device=plan.itropo.device),
        table.planck_fraction.to(device=plan.itropo.device),
        _gpoint_flavor(table, plan),
        plan.itropo,
        plan.jtemp,
        plan.jpress_arg,
        plan.jeta,
        plan.fmajor,
        plan.col_mix,
    )
    if _LW_GAS_COMPILE_MODE == "corner_gathers":
        return _interp3d_lw_major_planck_body(*arguments, compile_gathers=True, gpt_range=gpt_range)
    return run_compile_island("gas_optics", "interp3d_lw_major_planck", _interp3d_lw_major_planck_tensor,
                              *arguments, gpt_range=gpt_range)


def _build_plan_and_shortwave_major_tensor(
    play: torch.Tensor,
    tlay: torch.Tensor,
    col_gas: torch.Tensor,
    temp_ref: torch.Tensor,
    press_ref: torch.Tensor,
    press_ref_trop: torch.Tensor,
    gas1: torch.Tensor,
    gas2: torch.Tensor,
    vmr_ref: torch.Tensor,
    gpoint_flavor: torch.Tensor,
    tiny: torch.Tensor,
    kmajor: torch.Tensor,
    solar_source_quiet: torch.Tensor,
    solar_source_facular: torch.Tensor,
    solar_source_sunspot: torch.Tensor,
    mg: torch.Tensor,
    sb: torch.Tensor,
    tsi: torch.Tensor,
    mg_ref: torch.Tensor,
    sb_ref: torch.Tensor,
    solcon: torch.Tensor,
    *,
    ntemp: int,
    npres: int,
    neta: int,
    apply_solcon: bool,
    coordinates: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    plan_values = _build_interpolation_plan_tensor(
        play,
        tlay,
        col_gas,
        temp_ref,
        press_ref,
        press_ref_trop,
        gas1,
        gas2,
        vmr_ref,
        gpoint_flavor,
        tiny,
        ntemp=ntemp,
        npres=npres,
        neta=neta,
        coordinates=coordinates,
    )
    tau_major = _interp3d_all_gpoints_tensor(
        kmajor,
        plan_values[7],
        plan_values[0],
        plan_values[1],
        plan_values[2],
        plan_values[3],
        plan_values[4],
        plan_values[6],
        scale_by_col_mix=True,
    )
    source = _shortwave_solar_source_tensor(
        solar_source_quiet,
        solar_source_facular,
        solar_source_sunspot,
        mg,
        sb,
        tsi,
        mg_ref,
        sb_ref,
        solcon,
        apply_solcon=apply_solcon,
    )
    return (*plan_values, tau_major, source)


def _build_plan_and_lw_major_planck_tensor(
    play: torch.Tensor,
    tlay: torch.Tensor,
    col_gas: torch.Tensor,
    temp_ref: torch.Tensor,
    press_ref: torch.Tensor,
    press_ref_trop: torch.Tensor,
    gas1: torch.Tensor,
    gas2: torch.Tensor,
    vmr_ref: torch.Tensor,
    gpoint_flavor: torch.Tensor,
    tiny: torch.Tensor,
    kmajor: torch.Tensor,
    planck_fraction: torch.Tensor,
    *,
    ntemp: int,
    npres: int,
    neta: int,
    coordinates: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    plan_values = _build_interpolation_plan_tensor(
        play,
        tlay,
        col_gas,
        temp_ref,
        press_ref,
        press_ref_trop,
        gas1,
        gas2,
        vmr_ref,
        gpoint_flavor,
        tiny,
        ntemp=ntemp,
        npres=npres,
        neta=neta,
        coordinates=coordinates,
    )
    tau_major, pfrac = _interp3d_lw_major_planck_tensor(
        kmajor,
        planck_fraction,
        plan_values[7],
        plan_values[0],
        plan_values[1],
        plan_values[2],
        plan_values[3],
        plan_values[4],
        plan_values[6],
    )
    return (*plan_values, tau_major, pfrac)


def _build_plan_and_major(
    table: ReducedGasOpticsTable,
    *,
    play: torch.Tensor,
    tlay: torch.Tensor,
    col_gas: torch.Tensor,
    longwave: bool,
    solcon: torch.Tensor | float | None = None,
    coordinates: tuple[torch.Tensor, ...] | None = None,
) -> tuple[_GasInterpolationPlan, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    common_args = (
        play,
        tlay,
        col_gas,
        table.temp_ref.to(device=play.device, dtype=play.dtype),
        table.press_ref.to(device=play.device, dtype=play.dtype),
        table.press_ref_trop.to(device=play.device, dtype=play.dtype),
        table.flavor[:, 0].to(device=play.device, dtype=torch.int64),
        table.flavor[:, 1].to(device=play.device, dtype=torch.int64),
        table.vmr_ref.to(device=play.device, dtype=play.dtype),
        table.gpoint_flavor.to(device=play.device, dtype=torch.int64),
        torch.full((), 2.0 * torch.finfo(play.dtype).tiny, dtype=play.dtype, device=play.device),
        table.kmajor.to(device=play.device, dtype=play.dtype),
    )
    if longwave:
        if table.planck_fraction is None:
            raise ValueError("longwave table does not include Planck fractions")
        if _LW_GAS_COMPILE_MODE == "corner_gathers":
            plan_values = _build_interpolation_plan_tensor(
                *common_args[:11], ntemp=table.ntemp,
                npres=table.npres, neta=table.neta, coordinates=coordinates,
            )
            tau_major, pfrac = _interp3d_lw_major_planck_body(
                common_args[11],
                table.planck_fraction.to(device=play.device, dtype=play.dtype),
                plan_values[7], plan_values[0], plan_values[1], plan_values[2],
                plan_values[3], plan_values[4], plan_values[6],
                compile_gathers=True,
            )
            values = (*plan_values, tau_major, pfrac)
        else:
            values = run_compile_island(
                "gas_optics",
                "plan_and_lw_major_planck",
                _build_plan_and_lw_major_planck_tensor,
                *common_args,
                table.planck_fraction.to(device=play.device, dtype=play.dtype),
                ntemp=table.ntemp,
                npres=table.npres,
                neta=table.neta,
                coordinates=coordinates,
            )
        tau_major, pfrac, source = values[8], values[9], None
    else:
        if (
            table.solar_source_quiet is None
            or table.solar_source_facular is None
            or table.solar_source_sunspot is None
        ):
            raise ValueError("shortwave gas table does not include solar source components")
        dtype = play.dtype
        device = play.device
        mg = torch.as_tensor(0.1567652, dtype=dtype, device=device) if table.mg_default is None else table.mg_default.to(device=device, dtype=dtype)
        sb = torch.as_tensor(902.7126, dtype=dtype, device=device) if table.sb_default is None else table.sb_default.to(device=device, dtype=dtype)
        tsi = torch.as_tensor(1360.85767381726, dtype=dtype, device=device) if table.tsi_default is None else table.tsi_default.to(device=device, dtype=dtype)
        apply_solcon = solcon is not None
        solcon_t = tsi if solcon is None else torch.as_tensor(solcon, dtype=dtype, device=device)
        values = run_compile_island(
            "gas_optics",
            "plan_and_shortwave_major",
            _build_plan_and_shortwave_major_tensor,
            *common_args,
            table.solar_source_quiet.to(device=device, dtype=dtype),
            table.solar_source_facular.to(device=device, dtype=dtype),
            table.solar_source_sunspot.to(device=device, dtype=dtype),
            mg,
            sb,
            tsi,
            torch.full((), 0.1495954, dtype=dtype, device=device),
            torch.full((), 0.00066696, dtype=dtype, device=device),
            solcon_t,
            ntemp=table.ntemp,
            npres=table.npres,
            neta=table.neta,
            apply_solcon=apply_solcon,
            coordinates=coordinates,
        )
        tau_major, pfrac, source = values[8], None, values[9]
    plan = _GasInterpolationPlan(
        itropo=values[0],
        jtemp=values[1],
        jpress_arg=values[2],
        jeta=values[3],
        fmajor=values[4],
        fminor=values[5],
        col_mix=values[6],
        flavor=values[7],
    )
    return plan, tau_major, pfrac, source


def _interp2d_all_gpoints_tensor(
    coeff_t: torch.Tensor,
    flavor: torch.Tensor,
    itropo: torch.Tensor,
    jtemp: torch.Tensor,
    jeta: torch.Tensor,
    fminor: torch.Tensor,
    *,
    upper_coeff: torch.Tensor | None = None,
) -> torch.Tensor:
    ngpt = coeff_t.shape[-1]
    nlay, ncol = jtemp.shape
    gpt_index = torch.arange(ngpt, dtype=torch.int64, device=jtemp.device).reshape(ngpt, 1, 1).expand(ngpt, nlay, ncol)
    result = torch.zeros((ngpt, nlay, ncol), dtype=coeff_t.dtype, device=coeff_t.device)
    flavor_indices = (_flavor_indices(flavor, itropo) if torch.is_grad_enabled()
                      and fminor.requires_grad else None)
    # The domain is constant across the four corners. Select table entries
    # before interpolation, not two full spectral interpolants afterward.
    if upper_coeff is not None:
        coefficients = torch.stack((coeff_t, upper_coeff))
        domain = (itropo != 0).to(dtype=torch.int64).unsqueeze(0)
    for itemp in range(2):
        t_index_gpt = (jtemp + itemp - 1).unsqueeze(0).expand_as(gpt_index)
        eta_hi = _gather_flavor(jeta[:, itemp, :, :], flavor, itropo, indices=flavor_indices)
        for eta_slot in range(2):
            eta_index = eta_hi - 1 if eta_slot == 0 else eta_hi
            weight = _gather_flavor(fminor[:, eta_slot, itemp, :, :], flavor, itropo, indices=flavor_indices)
            corner = (coeff_t[t_index_gpt, eta_index, gpt_index] if upper_coeff is None
                      else coefficients[domain, t_index_gpt, eta_index, gpt_index])
            result = result + weight * corner
    return result


def _interp2d_all_gpoints(
    coeff: torch.Tensor,
    table: ReducedGasOpticsTable,
    plan: _GasInterpolationPlan,
) -> torch.Tensor:
    coeff_t = coeff.to(device=plan.itropo.device)
    return run_compile_island(
        "gas_optics",
        "interp2d_all_gpoints",
        _interp2d_all_gpoints_tensor,
        coeff_t,
        _gpoint_flavor(table, plan),
        plan.itropo,
        plan.jtemp,
        plan.jeta,
        plan.fminor,
    )


def _minor_record_interpolation_tensor(
    coeff: torch.Tensor,
    minor_index: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
    e10: torch.Tensor,
    e11: torch.Tensor,
    e20: torch.Tensor,
    e21: torch.Tensor,
    weight00: torch.Tensor,
    weight10: torch.Tensor,
    weight01: torch.Tensor,
    weight11: torch.Tensor,
) -> torch.Tensor:
    """Original four-corner association with already prepared tensor operands."""

    return (
        weight00 * coeff[t0, e10, minor_index]
        + weight10 * coeff[t0, e11, minor_index]
        + weight01 * coeff[t1, e20, minor_index]
        + weight11 * coeff[t1, e21, minor_index]
    )


def _column_reciprocal_tensor(column: torch.Tensor) -> torch.Tensor:
    # Reciprocal AD squares the inverse before applying its cotangent. Keep
    # that square normal with exact binary scaling. Normalize toward one but
    # not past it: overscaling can overflow/underflow a finite tangent or VJP.
    step = 256 if column.dtype == torch.float64 else 32
    magnitude = column.abs()
    scale = torch.ones_like(column)
    for exponent in (step, 2 * step, 3 * step):
        scale = torch.where(
            magnitude > 2.**exponent, 2.**-exponent,
            torch.where(magnitude < 2.**-exponent, 2.**exponent, scale),
        )
    return (1. / (column * scale)) * scale


def _minor_tau_delta_for_domain_tensor(
    coeff_t: torch.Tensor,
    itropo: torch.Tensor,
    jtemp: torch.Tensor,
    jeta: torch.Tensor,
    fminor: torch.Tensor,
    col_gas: torch.Tensor,
    play: torch.Tensor,
    tlay: torch.Tensor,
    records: tuple[_MinorTauRecord, ...],
    *,
    lower: bool,
    ngpt: int,
    record_interpolation_islands: bool = False,
    initial_tau: torch.Tensor | None = None,
    initial_slices: dict[tuple[int, int], torch.Tensor] | None = None,
    return_slices: bool = False,
) -> torch.Tensor | dict[tuple[int, int], torch.Tensor]:
    nlay, ncol = play.shape
    domain_scale = (itropo == (0 if lower else 1)).to(dtype=coeff_t.dtype).unsqueeze(0)
    base = (torch.zeros((ngpt, nlay, ncol), dtype=coeff_t.dtype, device=coeff_t.device)
            if initial_tau is None else initial_tau)
    # Each record covers whole g-point bands: a band's slice accumulates its records in order
    # (Fortran order), then the bands are joined once. No in-place write, no full copy per record.
    slices = {} if initial_slices is None else dict(initial_slices)
    for gpt_start, gpt_end, gas_index, scales_with_density, scale_index, scale_by_complement, iflav, start in records:
        count = gpt_end - gpt_start
        scaling = col_gas[gas_index, :, :]
        if scales_with_density:
            scaling = scaling * (0.01 * play / tlay)
            if scale_index > 0:
                vmr_fact = _column_reciprocal_tensor(col_gas[0, :, :])
                dry_fact = 1.0 / (1.0 + col_gas[1, :, :] * vmr_fact)
                scale_value = col_gas[scale_index, :, :] * vmr_fact * dry_fact
                scaling = scaling * (1.0 - scale_value if scale_by_complement else scale_value)
        minor_index = torch.arange(start, start + count, dtype=torch.int64, device=coeff_t.device).reshape(count, 1, 1).expand(count, nlay, ncol)
        t0 = (jtemp - 1).unsqueeze(0).expand_as(minor_index)
        t1 = jtemp.unsqueeze(0).expand_as(minor_index)
        e10 = (jeta[iflav, 0, :, :] - 1).unsqueeze(0).expand_as(minor_index)
        e11 = jeta[iflav, 0, :, :].unsqueeze(0).expand_as(minor_index)
        e20 = (jeta[iflav, 1, :, :] - 1).unsqueeze(0).expand_as(minor_index)
        e21 = jeta[iflav, 1, :, :].unsqueeze(0).expand_as(minor_index)
        weights = (
            fminor[iflav, 0, 0, :, :].unsqueeze(0),
            fminor[iflav, 1, 0, :, :].unsqueeze(0),
            fminor[iflav, 0, 1, :, :].unsqueeze(0),
            fminor[iflav, 1, 1, :, :].unsqueeze(0),
        )
        if record_interpolation_islands:
            # This branch runs only in the eager ordered outer reducer. Expanded
            # weights are views; their adjoint reductions stay outside the
            # pointwise compiled helper. It neither stacks all records nor
            # changes pressure-domain/gas summation order.
            values = run_compile_island(
                "gas_optics", "minor_record_interpolation",
                _minor_record_interpolation_tensor,
                coeff_t, minor_index, t0, t1, e10, e11, e20, e21,
                *(weight.expand_as(minor_index) for weight in weights),
            )
        else:
            values = _minor_record_interpolation_tensor(
                coeff_t, minor_index, t0, t1, e10, e11, e20, e21, *weights,
            )
        contribution = scaling.unsqueeze(0) * values * domain_scale.expand_as(values)
        # Fortran updates the existing major absorption after each minor gas.
        key = (gpt_start, gpt_end)
        slices[key] = (slices[key] if key in slices else base[gpt_start:gpt_end]) + contribution
    if return_slices:
        return slices
    parts, position = [], 0
    for start, end in sorted(slices):
        if start < position:
            raise ValueError("minor-gas records must cover disjoint g-point ranges")
        if start > position:
            parts.append(base[position:start])
        parts.append(slices[(start, end)])
        position = end
    if position < ngpt:
        parts.append(base[position:ngpt])
    return torch.cat(parts, dim=0) if len(parts) > 1 else (parts[0] if parts else base)


def _minor_tau_delta_for_domain(
    table: ReducedGasOpticsTable,
    plan: _GasInterpolationPlan,
    *,
    lower: bool,
    play: torch.Tensor,
    tlay: torch.Tensor,
    col_gas: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if lower:
        coeff = table.kminor_lower
        records = table.minor_records_lower
    else:
        coeff = table.kminor_upper
        records = table.minor_records_upper
    coeff_t = coeff.to(device=device, dtype=dtype)
    if not records:
        nlay, ncol = play.shape
        return torch.zeros((table.ngpt, nlay, ncol), dtype=dtype, device=device)
    return run_compile_island(
        "gas_optics",
        "minor_tau_delta_lower" if lower else "minor_tau_delta_upper",
        _minor_tau_delta_for_domain_tensor,
        coeff_t,
        plan.itropo,
        plan.jtemp,
        plan.jeta,
        plan.fminor,
        col_gas,
        play,
        tlay,
        records,
        lower=lower,
        ngpt=table.ngpt,
    )


def _accumulate_minor_tau_tensor(
    coeff_lower: torch.Tensor,
    coeff_upper: torch.Tensor,
    initial_tau: torch.Tensor,
    itropo: torch.Tensor,
    jtemp: torch.Tensor,
    jeta: torch.Tensor,
    fminor: torch.Tensor,
    col_gas: torch.Tensor,
    play: torch.Tensor,
    tlay: torch.Tensor,
    records_lower: tuple[_MinorTauRecord, ...],
    records_upper: tuple[_MinorTauRecord, ...],
    *,
    ngpt: int,
    record_interpolation_islands: bool = False,
) -> torch.Tensor:
    """Update spectral slices in lower/upper order, then join once."""

    boundaries = sorted({edge for record in (*records_lower, *records_upper) for edge in record[:2]})

    def partition(records):
        aligned = []
        for first, last, gas, density, scaling, complement, flavor, offset in records:
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                if first <= left < right <= last:
                    aligned.append((left, right, gas, density, scaling, complement, flavor, offset + left - first))
        return tuple(aligned)

    lower = _minor_tau_delta_for_domain_tensor(
        coeff_lower,
        itropo,
        jtemp,
        jeta,
        fminor,
        col_gas,
        play,
        tlay,
        partition(records_lower),
        lower=True,
        ngpt=ngpt,
        initial_tau=initial_tau,
        record_interpolation_islands=record_interpolation_islands,
        return_slices=True,
    )
    upper = _minor_tau_delta_for_domain_tensor(
        coeff_upper,
        itropo,
        jtemp,
        jeta,
        fminor,
        col_gas,
        play,
        tlay,
        partition(records_upper),
        lower=False,
        ngpt=ngpt,
        initial_tau=initial_tau,
        initial_slices=lower,
        record_interpolation_islands=record_interpolation_islands,
    )
    return upper


def _accumulate_minor_tau(
    table: ReducedGasOpticsTable,
    plan: _GasInterpolationPlan,
    *,
    play: torch.Tensor,
    tlay: torch.Tensor,
    col_gas: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    initial_tau: torch.Tensor,
) -> torch.Tensor:
    """Dispatch the same ordered minor-gas science at the selected granularity."""

    if _MINOR_GAS_COMPILE_MODE == "record_interpolation":
        # Keep the loop/scaling shell eager; true selects compiled pointwise
        # interpolation inside it. The old joint compiled body uses false and
        # consequently contains only pure tensor operations.
        return _accumulate_minor_tau_tensor(
            table.kminor_lower.to(device=device, dtype=dtype),
            table.kminor_upper.to(device=device, dtype=dtype), initial_tau,
            plan.itropo, plan.jtemp, plan.jeta, plan.fminor, col_gas, play, tlay,
            table.minor_records_lower, table.minor_records_upper, ngpt=table.ngpt,
            record_interpolation_islands=True,
        )

    return run_compile_island(
        "gas_optics",
        "minor_tau_deltas",
        _accumulate_minor_tau_tensor,
        table.kminor_lower.to(device=device, dtype=dtype),
        table.kminor_upper.to(device=device, dtype=dtype),
        initial_tau,
        plan.itropo,
        plan.jtemp,
        plan.jeta,
        plan.fminor,
        col_gas,
        play,
        tlay,
        table.minor_records_lower,
        table.minor_records_upper,
        ngpt=table.ngpt,
    )


def _interp_planck_tensor(
    totplnk: torch.Tensor,
    temp_ref: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    temp_delta = (temp_ref[-1] - temp_ref[0]) / float(totplnk.shape[0] - 1)
    loc = (temperature - temp_ref[0]) / temp_delta
    frac = loc - torch.trunc(loc)
    index = torch.clamp(torch.trunc(loc) + 1, 1, totplnk.shape[0] - 1).to(dtype=torch.int64) - 1
    lower = totplnk.index_select(0, index.reshape(-1)).reshape(*temperature.shape, totplnk.shape[1])
    upper = totplnk.index_select(0, (index + 1).reshape(-1)).reshape(*temperature.shape, totplnk.shape[1])
    return lower + frac.unsqueeze(-1) * (upper - lower)


def _interp_planck(table: ReducedGasOpticsTable, temperature: torch.Tensor) -> torch.Tensor:
    if table.totplnk is None:
        raise ValueError("longwave table does not include total Planck source table")
    return _interp_planck_tensor(table.totplnk, table.temp_ref, temperature)


def _finalize_longwave_gas_tensor(
    tau_abs: torch.Tensor,
    zero: torch.Tensor,
    pfrac: torch.Tensor,
    totplnk: torch.Tensor,
    temp_ref: torch.Tensor,
    tsfc: torch.Tensor,
    tlay: torch.Tensor,
    tlev: torch.Tensor,
    bands: torch.Tensor,
    sfc_lay: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tau = torch.maximum(tau_abs, zero)
    sfc_planck = _interp_planck_tensor(totplnk, temp_ref, tsfc)
    sfc_planck_plus = _interp_planck_tensor(totplnk, temp_ref, tsfc + 1.0)
    sfc_planck_gpt = sfc_planck.index_select(1, bands).transpose(0, 1).contiguous()
    sfc_planck_plus_gpt = sfc_planck_plus.index_select(1, bands).transpose(0, 1).contiguous()
    sfc_pfrac = pfrac[:, sfc_lay, :]
    sfc_source = sfc_pfrac * sfc_planck_gpt
    sfc_source_jac = sfc_pfrac * (sfc_planck_plus_gpt - sfc_planck_gpt)
    lay_planck = _interp_planck_tensor(totplnk, temp_ref, tlay)
    lay_source = pfrac * lay_planck.index_select(2, bands).permute(2, 0, 1).contiguous()
    lev_planck_dec = _interp_planck_tensor(totplnk, temp_ref, tlev[:-1, :])
    lev_planck_inc = _interp_planck_tensor(totplnk, temp_ref, tlev[1:, :])
    lev_source_dec = pfrac * lev_planck_dec.index_select(2, bands).permute(2, 0, 1).contiguous()
    lev_source_inc = pfrac * lev_planck_inc.index_select(2, bands).permute(2, 0, 1).contiguous()
    return tau, lay_source, lev_source_inc, lev_source_dec, sfc_source, sfc_source_jac


def _finalize_longwave_gas(
    table: ReducedGasOpticsTable,
    *,
    tau_abs: torch.Tensor,
    pfrac: torch.Tensor,
    tsfc: torch.Tensor,
    tlay: torch.Tensor,
    tlev: torch.Tensor,
    bands: torch.Tensor,
    sfc_lay: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if table.totplnk is None:
        raise ValueError("longwave table does not include total Planck source table")
    return run_compile_island(
        "gas_optics",
        "finalize_longwave_gas",
        _finalize_longwave_gas_tensor,
        tau_abs,
        torch.zeros((), dtype=tau_abs.dtype, device=tau_abs.device),
        pfrac,
        table.totplnk,
        table.temp_ref,
        tsfc,
        tlay,
        tlev,
        bands,
        sfc_lay,
    )


def compute_longwave_gas_optics(
    table: ReducedGasOpticsTable,
    *,
    p_lay: torch.Tensor,
    p_lev: torch.Tensor,
    t_lay: torch.Tensor,
    t_lev: torch.Tensor,
    tsfg: torch.Tensor,
    vmr: torch.Tensor,
    top_at_1: bool = False,
    retain_diagnostics: bool = True,
) -> LongwaveGasOpticsResult:
    """Compute LW gas optical depth and Planck source tensors.

    Tensor orientation follows the sealed checkpoint convention:
    ``[nlay,ncol]`` for layer fields and bottom-at-index-0 when
    ``top_at_1=False``.
    """

    if table.planck_fraction is None or table.totplnk is None:
        raise ValueError("compute_longwave_gas_optics requires a longwave gas table")
    play = _as_real(p_lay, name="p_lay")
    plev = _as_real(p_lev, name="p_lev").to(device=play.device)
    tlay = _as_real(t_lay, name="t_lay").to(device=play.device)
    tlev = _as_real(t_lev, name="t_lev").to(device=play.device)
    tsfc = _as_real(tsfg, name="tsfg").reshape(-1).to(device=play.device)
    vmr_t = _as_real(vmr, name="vmr").to(device=play.device)
    if play.ndim != 2 or plev.ndim != 2 or tlay.shape != play.shape or plev.shape != (play.shape[0] + 1, play.shape[1]):
        raise ValueError("longwave gas-optics fields must use [nlay,ncol] and [nlay+1,ncol] checkpoint shapes")
    nlay, ncol = play.shape
    if vmr_t.shape != (len(table.gas_names) + 1, nlay, ncol):
        raise ValueError(f"vmr must have shape [{len(table.gas_names)+1},{nlay},{ncol}], got {tuple(vmr_t.shape)}")

    col_dry = _compute_col_dry(plev, vmr_t[1])
    # ``vmr_t[0]`` is the prepared dry-air unit plane.  One broadcast multiply
    # produces the same dry and trace-gas columns without a second full-field
    # concatenation at every SW/LW call.
    col_gas = vmr_t * col_dry.unsqueeze(0)
    plan, tau, pfrac_value, _ = _build_plan_and_major(
        table,
        play=play,
        tlay=tlay,
        col_gas=col_gas,
        longwave=True,
    )
    if pfrac_value is None:
        raise AssertionError("longwave gas interpolation did not return Planck fractions")
    pfrac = pfrac_value
    tau = _accumulate_minor_tau(
        table,
        plan,
        play=play,
        tlay=tlay,
        col_gas=col_gas,
        dtype=tau.dtype,
        device=tau.device,
        initial_tau=tau,
    )
    sfc_lay = nlay - 1 if top_at_1 else 0
    bands = table.gpt_to_band.to(device=play.device, dtype=torch.int64)
    tau, lay_source, lev_source_inc, lev_source_dec, sfc_source, sfc_source_jac = _finalize_longwave_gas(
        table,
        tau_abs=tau,
        pfrac=pfrac,
        tsfc=tsfc,
        tlay=tlay,
        tlev=tlev,
        bands=bands,
        sfc_lay=sfc_lay,
    )

    diffusivity_angle = compute_longwave_diffusivity_angles(table, tau) if retain_diagnostics else None
    return LongwaveGasOpticsResult(
        tau=tau,
        lay_source=lay_source,
        lev_source_inc=lev_source_inc,
        lev_source_dec=lev_source_dec,
        sfc_source=sfc_source,
        sfc_source_jac=sfc_source_jac,
        col_dry=col_dry if retain_diagnostics else None,
        gpt_to_band=table.gpt_to_band,
        optimal_angle_fit=table.optimal_angle_fit,
        diffusivity_angle=diffusivity_angle,
    )


def _shortwave_solar_source_tensor(
    quiet: torch.Tensor,
    facular: torch.Tensor,
    sunspot: torch.Tensor,
    mg: torch.Tensor,
    sb: torch.Tensor,
    tsi: torch.Tensor,
    mg_ref: torch.Tensor,
    sb_ref: torch.Tensor,
    solcon: torch.Tensor,
    *,
    apply_solcon: bool,
    normalize_tsi: bool = True,
) -> torch.Tensor:
    source = quiet + (mg - mg_ref) * facular + (sb - sb_ref) * sunspot
    if normalize_tsi:
        source = source * tsi / source.sum()
    if apply_solcon:
        source = source * solcon / source.sum()
    return source


def compute_shortwave_solar_source(
    table: ReducedGasOpticsTable,
    *,
    solcon: torch.Tensor | float | None = None,
    normalize_tsi: bool = True,
) -> torch.Tensor:
    """Compute the RRTMGP SW g-point solar source vector.

    This mirrors `set_solar_variability(mg_default, sb_default, tsi_default)`
    followed by the GFS SW-main `solcon / sum(toa_src)` scaling. UFS v1.7
    initializes variability without the optional TSI argument.
    """

    if table.solar_source_quiet is None or table.solar_source_facular is None or table.solar_source_sunspot is None:
        raise ValueError("shortwave gas table does not include solar source components")
    device = table.solar_source_quiet.device
    dtype = table.solar_source_quiet.dtype
    mg = torch.as_tensor(0.1567652, dtype=dtype, device=device) if table.mg_default is None else table.mg_default.to(device=device, dtype=dtype)
    sb = torch.as_tensor(902.7126, dtype=dtype, device=device) if table.sb_default is None else table.sb_default.to(device=device, dtype=dtype)
    tsi = torch.as_tensor(1360.85767381726, dtype=dtype, device=device) if table.tsi_default is None else table.tsi_default.to(device=device, dtype=dtype)
    apply_solcon = solcon is not None
    solcon_t = tsi if solcon is None else torch.as_tensor(solcon, dtype=dtype, device=device)
    return _shortwave_solar_source_tensor(
        table.solar_source_quiet,
        table.solar_source_facular,
        table.solar_source_sunspot,
        mg,
        sb,
        tsi,
        torch.full((), 0.1495954, dtype=dtype, device=device),
        torch.full((), 0.00066696, dtype=dtype, device=device),
        solcon_t,
        apply_solcon=apply_solcon,
        normalize_tsi=normalize_tsi,
    )


def _finalize_shortwave_optics_tensor(
    tau_abs: torch.Tensor,
    rayleigh_lower: torch.Tensor,
    rayleigh_upper: torch.Tensor,
    itropo: torch.Tensor,
    moist_plus_dry_column: torch.Tensor,
    zero: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tau_rayleigh = torch.where(
        itropo.unsqueeze(0) == 0,
        rayleigh_lower,
        rayleigh_upper,
    ) * moist_plus_dry_column.unsqueeze(0)
    tau_abs = torch.maximum(tau_abs, zero)
    tau = tau_abs + tau_rayleigh
    ssa = torch.where(tau > threshold, tau_rayleigh / torch.where(tau > threshold, tau, 1.0), torch.zeros_like(tau))
    return tau_abs, tau_rayleigh, tau, ssa, torch.zeros_like(tau)


def _finalize_shortwave_optics(
    tau_abs: torch.Tensor,
    rayleigh_lower: torch.Tensor,
    rayleigh_upper: torch.Tensor,
    plan: _GasInterpolationPlan,
    moist_plus_dry_column: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = torch.zeros((), dtype=tau_abs.dtype, device=tau_abs.device)
    threshold = 2.0 * torch.finfo(tau_abs.dtype).tiny
    return run_compile_island(
        "gas_optics",
        "finalize_shortwave_optics",
        _finalize_shortwave_optics_tensor,
        tau_abs,
        rayleigh_lower,
        rayleigh_upper,
        plan.itropo,
        moist_plus_dry_column,
        zero,
        threshold,
    )


def _rayleigh_and_finalize_shortwave_tensor(
    rayleigh_lower_coeff: torch.Tensor,
    rayleigh_upper_coeff: torch.Tensor,
    flavor: torch.Tensor,
    jtemp: torch.Tensor,
    jeta: torch.Tensor,
    fminor: torch.Tensor,
    tau_abs: torch.Tensor,
    itropo: torch.Tensor,
    moist_plus_dry_column: torch.Tensor,
    zero: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rayleigh = _interp2d_all_gpoints_tensor(
        rayleigh_lower_coeff,
        flavor,
        itropo,
        jtemp,
        jeta,
        fminor,
        upper_coeff=rayleigh_upper_coeff,
    )
    return _finalize_shortwave_optics_tensor(
        tau_abs,
        rayleigh,
        rayleigh,
        itropo,
        moist_plus_dry_column,
        zero,
        threshold,
    )


def _rayleigh_and_finalize_shortwave(
    table: ReducedGasOpticsTable,
    plan: _GasInterpolationPlan,
    *,
    tau_abs: torch.Tensor,
    moist_plus_dry_column: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if table.rayl_lower is None or table.rayl_upper is None:
        raise ValueError("shortwave gas table does not include Rayleigh coefficients")
    return run_compile_island(
        "gas_optics",
        "rayleigh_and_finalize_shortwave",
        _rayleigh_and_finalize_shortwave_tensor,
        table.rayl_lower,
        table.rayl_upper,
        _gpoint_flavor(table, plan),
        plan.jtemp,
        plan.jeta,
        plan.fminor,
        tau_abs,
        plan.itropo,
        moist_plus_dry_column,
        torch.zeros((), dtype=tau_abs.dtype, device=tau_abs.device),
        2.0 * torch.finfo(tau_abs.dtype).tiny,
    )


def compute_shortwave_gas_optics(
    table: ReducedGasOpticsTable,
    *,
    p_lay: torch.Tensor,
    p_lev: torch.Tensor,
    t_lay: torch.Tensor,
    vmr: torch.Tensor,
    solcon: torch.Tensor | float | None = None,
    retain_diagnostics: bool = True,
) -> ShortwaveGasOpticsResult:
    """Compute SW gas absorption, Rayleigh scattering, and TOA source.

    Tensor orientation follows checkpoint convention: `[nlay,ncol]` for layer
    fields and `[nlay+1,ncol]` for interfaces.  Returned optical properties use
    `[ngpt,nlay,ncol]` to match the existing target gas-optics convention.
    """

    if table.rayl_lower is None or table.rayl_upper is None:
        raise ValueError("compute_shortwave_gas_optics requires a shortwave gas table with Rayleigh coefficients")
    play = _as_real(p_lay, name="p_lay")
    plev = _as_real(p_lev, name="p_lev").to(device=play.device)
    tlay = _as_real(t_lay, name="t_lay").to(device=play.device)
    vmr_t = _as_real(vmr, name="vmr").to(device=play.device)
    if play.ndim != 2 or plev.ndim != 2 or tlay.shape != play.shape or plev.shape != (play.shape[0] + 1, play.shape[1]):
        raise ValueError("shortwave gas-optics fields must use [nlay,ncol] and [nlay+1,ncol] checkpoint shapes")
    nlay, ncol = play.shape
    if vmr_t.shape != (len(table.gas_names) + 1, nlay, ncol):
        raise ValueError(f"vmr must have shape [{len(table.gas_names)+1},{nlay},{ncol}], got {tuple(vmr_t.shape)}")

    col_dry = _compute_col_dry(plev, vmr_t[1])
    col_gas = vmr_t * col_dry.unsqueeze(0)
    plan, tau_abs, _, source = _build_plan_and_major(
        table,
        play=play,
        tlay=tlay,
        col_gas=col_gas,
        longwave=False,
        solcon=solcon,
    )
    tau_abs = _accumulate_minor_tau(
        table,
        plan,
        play=play,
        tlay=tlay,
        col_gas=col_gas,
        dtype=tau_abs.dtype,
        device=tau_abs.device,
        initial_tau=tau_abs,
    )
    tau_abs, tau_rayleigh, tau, ssa, asymmetry = _rayleigh_and_finalize_shortwave(
        table,
        plan,
        tau_abs=tau_abs,
        moist_plus_dry_column=col_gas[1, :, :] + col_dry,
    )
    if source is None:
        raise AssertionError("shortwave gas interpolation did not return a solar source")
    toa_src = source.reshape(table.ngpt, 1).expand(table.ngpt, ncol).contiguous()
    return ShortwaveGasOpticsResult(
        tau=tau,
        ssa=ssa,
        g=asymmetry,
        tau_absorption=tau_abs if retain_diagnostics else None,
        tau_rayleigh=tau_rayleigh if retain_diagnostics else None,
        toa_src=toa_src,
        col_dry=col_dry if retain_diagnostics else None,
        gpt_to_band=table.gpt_to_band,
    )


def compute_shortwave_gas_optics_from_boundaries(
    boundaries: Mapping[str, Mapping[str, torch.Tensor | str]],
    *,
    tables: RrtmgpTableBundle | None = None,
    active_gases: Sequence[str] = DEFAULT_ACTIVE_GASES,
    retain_diagnostics: bool = True,
) -> ShortwaveGasOpticsResult:
    """Compute SW gas/Rayleigh optics from A/AB checkpoint inputs."""

    pre = boundaries["AB"]
    setup = boundaries["A"]
    p_lay = _as_real(pre["p_lay"], name="p_lay")
    bundle = tables if tables is not None else load_rrtmgp_tables(device=p_lay.device)
    table = reduce_gas_optics_table(bundle.files["sw_gas"], active_gases=active_gases)
    vmr = _vmr_from_boundaries(
        pre,
        active_gases=table.gas_names,
        shape=tuple(p_lay.shape),
        device=p_lay.device,
        dtype=p_lay.dtype,
    )
    solcon_value = setup.get("solcon", pre.get("solcon"))
    solcon_t = _as_real(solcon_value, name="solcon").reshape(-1)[0] if isinstance(solcon_value, torch.Tensor) else None
    return compute_shortwave_gas_optics(
        table,
        p_lay=p_lay,
        p_lev=_as_real(pre["p_lev"], name="p_lev"),
        t_lay=_as_real(pre["t_lay"], name="t_lay"),
        vmr=vmr,
        solcon=solcon_t,
        retain_diagnostics=retain_diagnostics,
    )


def compute_longwave_diffusivity_angles_from_fit(
    optimal_angle_fit: torch.Tensor | None,
    gpt_to_band: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Compute RRTMGP optimal LW diffusivity secants from a supplied tau field."""

    if optimal_angle_fit is None:
        return torch.ones((gpt_to_band.numel(), tau.shape[2]), dtype=tau.dtype, device=tau.device)
    column_tau = tau.sum(dim=1)
    trans_total = torch.exp(-column_tau)
    fit = optimal_angle_fit.to(device=tau.device, dtype=tau.dtype)
    bands = gpt_to_band.to(device=tau.device, dtype=torch.int64).reshape(-1)
    fit_by_gpt = fit.index_select(1, bands)
    return fit_by_gpt[0, :, None] * trans_total + fit_by_gpt[1, :, None]


def compute_longwave_diffusivity_angles(table: ReducedGasOpticsTable, tau: torch.Tensor) -> torch.Tensor:
    """Compute RRTMGP optimal LW diffusivity secants from column transmissivity."""

    return compute_longwave_diffusivity_angles_from_fit(table.optimal_angle_fit, table.gpt_to_band, tau)


def compute_longwave_gas_optics_from_boundaries(
    boundaries: Mapping[str, Mapping[str, torch.Tensor | str]],
    *,
    tables: RrtmgpTableBundle | None = None,
    active_gases: Sequence[str] = DEFAULT_ACTIVE_GASES,
    retain_diagnostics: bool = True,
) -> LongwaveGasOpticsResult:
    """Compute LW gas optics from the sealed A/AB/ABC checkpoint inputs."""

    pre = boundaries["AB"]
    surface = boundaries["ABC"]
    p_lay = _as_real(pre["p_lay"], name="p_lay")
    bundle = tables if tables is not None else load_rrtmgp_tables(device=p_lay.device)
    table = reduce_gas_optics_table(bundle.files["lw_gas"], active_gases=active_gases)
    top_at_1_value = pre["top_at_1"]
    if not isinstance(top_at_1_value, torch.Tensor):
        raise TypeError("top_at_1 must be a torch tensor")
    vmr = _vmr_from_boundaries(
        pre,
        active_gases=table.gas_names,
        shape=tuple(p_lay.shape),
        device=p_lay.device,
        dtype=p_lay.dtype,
    )
    return compute_longwave_gas_optics(
        table,
        p_lay=p_lay,
        p_lev=_as_real(pre["p_lev"], name="p_lev"),
        t_lay=_as_real(pre["t_lay"], name="t_lay"),
        t_lev=_as_real(pre["t_lev"], name="t_lev"),
        tsfg=_as_real(surface["tsfg"], name="tsfg"),
        vmr=vmr,
        top_at_1=bool(int(top_at_1_value.detach().to(device="cpu", dtype=torch.int64).reshape(-1)[0].item())),
        retain_diagnostics=retain_diagnostics,
    )


__all__ = [
    "DEFAULT_ACTIVE_GASES",
    "LongwaveGasOpticsResult",
    "ReducedGasOpticsTable",
    "ShortwaveGasOpticsResult",
    "compute_longwave_diffusivity_angles",
    "compute_longwave_diffusivity_angles_from_fit",
    "compute_longwave_gas_optics",
    "compute_longwave_gas_optics_from_boundaries",
    "compute_shortwave_gas_optics",
    "compute_shortwave_gas_optics_from_boundaries",
    "compute_shortwave_solar_source",
    "reduce_gas_optics_table",
]
