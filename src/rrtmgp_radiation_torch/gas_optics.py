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
from typing import Sequence

import torch

from .tables import RrtmgpTableFile


DEFAULT_ACTIVE_GASES = ("h2o", "co2", "o3", "n2o", "ch4", "o2")

AVOGADRO = 6.02214076e23
M_DRY = 0.028964
M_H2O = 0.018016
GRAV = 9.80665
_MinorTauRecord = tuple[int, int, int, bool, int, bool, int, int]
_REDUCED_GAS_TABLE_CACHE: dict[tuple[str, str, str, torch.dtype, tuple[str, ...]], "ReducedGasOpticsTable"] = {}


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
class _GasInterpolationPlan:
    itropo: torch.Tensor
    jtemp: torch.Tensor
    jpress_arg: torch.Tensor
    jeta: torch.Tensor
    fmajor: torch.Tensor
    fminor: torch.Tensor
    col_mix: torch.Tensor
    flavor: torch.Tensor | None = None  # the table's gpoint_flavor [2, ngpt]


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
    cache_key = (str(table.path), table.sha256, str(device), _table_tensor(table, "kmajor").dtype, active)
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
        kmajor=kmajor,
        kminor_lower=kminor_lower,
        kminor_upper=kminor_upper,
        planck_fraction=planck_fraction,
        totplnk=totplnk,
        optimal_angle_fit=optimal_angle_fit,
        rayl_lower=rayl_lower_raw.to(device=device).contiguous() if rayl_lower_raw is not None else None,
        rayl_upper=rayl_upper_raw.to(device=device).contiguous() if rayl_upper_raw is not None else None,
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


def _interpolation_coordinates_tensor(
    play: torch.Tensor,
    tlay: torch.Tensor,
    temp_ref: torch.Tensor,
    press_ref: torch.Tensor,
    press_ref_trop: torch.Tensor,
    *,
    ntemp: int,
    npres: int,
    compact: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Column/layer coordinates, shared by every spectral point.

    ``compact`` (inference: no derivative reads them): the same index values as int32 (``jtemp``,
    ``jpress_arg``) and the troposphere flag as bool (``itropo``), so every spectral kernel reads 9 instead of
    24 bytes of them per layer and column; index sites read a bool ``itropo`` through ``_as_index``."""
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
    if compact:
        return jtemp.to(torch.int32), ftemp, itropo.to(torch.bool), jpress_arg.to(torch.int32), fpress
    return jtemp, ftemp, itropo, jpress_arg, fpress


def _as_index(itropo: torch.Tensor) -> torch.Tensor:
    """``itropo`` as an integer index (a compact plan stores it as bool, which indexing reads as a mask)."""
    return itropo.to(torch.int64) if itropo.dtype == torch.bool else itropo


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
    major_weights: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Build all major/minor interpolation coordinates from flat tensors.

    ``major_weights=False`` leaves out ``fmajor`` (None): each of its elements is the pressure weight
    times the ``fminor`` element, which the banded inference interpolation forms per band.
    """

    if coordinates is None:
        coordinates = _interpolation_coordinates_tensor(
            play, tlay, temp_ref, press_ref, press_ref_trop, ntemp=ntemp, npres=npres,
        )
    jtemp, ftemp, itropo, jpress_arg, fpress = coordinates
    flavor = gpoint_flavor  # the static [2, ngpt] table: no per-element flavor index

    col1 = col_gas.index_select(0, gas1)
    col2 = col_gas.index_select(0, gas2)
    itropo_i = _as_index(itropo).unsqueeze(0)
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
        if jtemp.dtype != torch.int64:  # compact coordinates: eta indices in the same integer width
            jeta_value = jeta_value.to(jtemp.dtype)
        jeta_by_temp.append(jeta_value)
        feta = loceta - trunc_eta
        ftemp_term = ((1.0 - ftemp) if itemp == 0 else ftemp).unsqueeze(0)
        fminor_eta0 = (1.0 - feta) * ftemp_term
        fminor_eta1 = feta * ftemp_term
        fminor_temp = torch.stack((fminor_eta0, fminor_eta1), dim=1)
        fminor_by_temp.append(fminor_temp)
        if not major_weights:
            continue
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
        torch.stack(fmajor_by_temp, dim=3) if major_weights else None,
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
    major_weights: bool = True,
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
    values = _build_interpolation_plan_tensor(*arguments, ntemp=table.ntemp, npres=table.npres, neta=table.neta, coordinates=coordinates, major_weights=major_weights)
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


def _flavor_indices(gpoint_flavor: torch.Tensor, itropo: torch.Tensor, *, level_major: bool = False) -> torch.Tensor:
    """One shared interpolation index, instead of two spectral gathers per operand (``level_major``:
    ``[nlay, ngpt, ncol]``)."""
    if level_major:
        return torch.where(itropo.unsqueeze(1) != 0,
                           gpoint_flavor[1, None, :, None], gpoint_flavor[0, None, :, None])
    return torch.where(itropo.unsqueeze(0) != 0,
                       gpoint_flavor[1, :, None, None], gpoint_flavor[0, :, None, None])


def _gather_flavor(values: torch.Tensor, gpoint_flavor: torch.Tensor, itropo: torch.Tensor,
                   *, indices: torch.Tensor | None = None, level_major: bool = False) -> torch.Tensor:
    """Select flavors from the original broadcast layout, retaining its VJP reduction order.

    ``level_major`` (compiled inference): the same selection with the g-points on the middle axis,
    ``[nlay, ngpt, ncol]``."""
    ngpt = gpoint_flavor.shape[1]
    if indices is None:
        indices = _flavor_indices(gpoint_flavor, itropo, level_major=level_major)
    if level_major:
        nflav, nlay, ncol = values.shape
        expanded = values.transpose(0, 1).unsqueeze(1).expand(nlay, ngpt, nflav, ncol)
        return torch.gather(expanded, 2, indices.unsqueeze(2)).squeeze(2)
    expanded = values.unsqueeze(0).expand(ngpt, -1, -1, -1)
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
    fmajor: torch.Tensor | None,
    col_mix: torch.Tensor,
    *,
    scale_by_col_mix: bool,
    fminor: torch.Tensor | None = None,
    fpress: torch.Tensor | None = None,
    level_major: bool = False,
    inference: bool = False,
) -> torch.Tensor:
    """Without ``fmajor`` (an inference plan), each corner weight is formed as the plan forms it: the pressure weight
    (``1 - fpress`` or ``fpress``) times the flavor-selected ``fminor`` element. ``level_major`` (that form, as the
    compiled inference regions take it): the result is computed in a ``[nlay, ngpt, ncol]`` buffer and returned as its
    ``[ngpt, nlay, ncol]`` transposed view, so each layer's ``[ncol]`` index and weight rows are read once for all
    g-points. Every element is the same expression either way. ``inference``: no derivative is formed (the caller
    decides it once, ``radiation_group._derivative_free``)."""
    if level_major and fmajor is not None:
        raise ValueError("the level-major interpolation forms its weights from fminor (a plan without fmajor)")
    ngpt = coeff_t.shape[-1]
    nlay, ncol = jtemp.shape
    g_axis = 1 if level_major else 0
    gpt_index = torch.arange(ngpt, dtype=torch.int64, device=jtemp.device).reshape(
        (1, ngpt, 1) if level_major else (ngpt, 1, 1))
    p_indices = (jpress_arg - 2, jpress_arg - 1)
    t_indices = (jtemp - 1, jtemp)
    # Backward saves gather indices: share that storage under AD. In inference
    # each gather releases its index before the larger coefficient temporaries.
    flavor_indices = (_flavor_indices(flavor, itropo) if not inference and fmajor is not None
                      and (fmajor.requires_grad or (scale_by_col_mix and col_mix.requires_grad)) else None)
    # interpolate3D_byflav sums eta first, then pressure, before scaling: ((c0 + c1) + c2) + c3,
    # accumulated as each corner is formed so no corner outlives its addition. One eta gather
    # per temperature (the integer eta - 1 + slot gathered equals the gathered eta, minus 1 + slot).
    total = None
    for itemp, t_index in enumerate(t_indices):
        t_index_gpt = t_index.unsqueeze(g_axis)
        eta_dtype = torch.int32 if coeff_t.shape[1] <= 2**31 else torch.int64
        eta_hi = _gather_flavor(jeta[:, itemp, :, :].to(dtype=eta_dtype),
                                flavor, itropo, indices=flavor_indices, level_major=level_major)
        subtotal = None
        for press_slot, p_index in enumerate(p_indices):
            p_index_gpt = p_index.unsqueeze(g_axis)
            for eta_slot in range(2):
                # The integer gather owns its workspace. Restore it after each pair.
                eta_index = eta_hi.add_(-1 if eta_slot == 0 else 1)
                if fmajor is not None:
                    weight = _gather_flavor(fmajor[:, eta_slot, press_slot, itemp, :, :], flavor, itropo,
                                            indices=flavor_indices)
                else:
                    weight = ((1.0 - fpress) if press_slot == 0 else fpress).unsqueeze(g_axis) * _gather_flavor(
                        fminor[:, eta_slot, itemp, :, :], flavor, itropo, level_major=level_major)
                corner = weight * _coefficient_corner_tensor(
                    coeff_t, t_index_gpt, eta_index, p_index_gpt, gpt_index)
                del weight, eta_index
                subtotal = corner if subtotal is None else subtotal + corner
                del corner
        del eta_hi
        scale = (_gather_flavor(col_mix[:, itemp, :, :], flavor, itropo, indices=flavor_indices,
                                level_major=level_major)
                 if scale_by_col_mix else 1.0)
        scaled = scale * subtotal
        del scale, subtotal
        total = scaled if total is None else total + scaled
        del scaled
    return total.transpose(0, 1) if level_major else total


def _interp3d_all_gpoints(
    coeff: torch.Tensor,
    table: ReducedGasOpticsTable,
    plan: _GasInterpolationPlan,
    *,
    scale_by_col_mix: bool,
    fpress: torch.Tensor | None = None,
    level_major: bool = False,
    inference: bool = False,
) -> torch.Tensor:
    """``fpress``: the pressure fraction of the interpolation coordinates, for an inference plan (no ``fmajor``);
    ``level_major`` and ``inference``: ``_interp3d_all_gpoints_tensor``."""
    coeff_t = coeff.to(device=plan.itropo.device)
    return _interp3d_all_gpoints_tensor(coeff_t, _gpoint_flavor(table, plan), plan.itropo, plan.jtemp, plan.jpress_arg, plan.jeta, plan.fmajor, plan.col_mix, scale_by_col_mix=scale_by_col_mix, fminor=plan.fminor, fpress=fpress, level_major=level_major, inference=inference)


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
    kmajor_t: torch.Tensor,
    planck_fraction_t: torch.Tensor,
    flavor: torch.Tensor,
    itropo: torch.Tensor,
    jtemp: torch.Tensor,
    jpress_arg: torch.Tensor,
    jeta: torch.Tensor,
    fmajor: torch.Tensor | None,
    col_mix: torch.Tensor,
    *,
    gpt_range: tuple[int, int] | None = None,
    fminor: torch.Tensor | None = None,
    fpress: torch.Tensor | None = None,
    level_major: bool = False,
    inference: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate LW major absorption and Planck fractions together.

    Without ``fmajor`` (an inference plan), each corner weight is formed as the plan forms it: the
    pressure weight (``1 - fpress`` or ``fpress``) times the flavor-selected ``fminor`` element.
    ``level_major`` and ``inference``: as ``_interp3d_all_gpoints_tensor`` (the arrays are returned as
    ``[ngpt, nlay, ncol]`` transposed views of level-major buffers).

    Both tables use the same eight interpolation corners.  Keeping the two
    accumulators in one tensor island shares all flavor, eta, pressure, and
    temperature index work without changing either accumulator's addition
    order.
    """

    table_ngpt = kmajor_t.shape[-1]
    start, end = (0, table_ngpt) if gpt_range is None else gpt_range
    ngpt = end - start
    nlay, ncol = jtemp.shape
    if level_major and fmajor is not None:
        raise ValueError("the level-major interpolation forms its weights from fminor (a plan without fmajor)")
    g_axis = 1 if level_major else 0
    gpt_index = torch.arange(start, end, dtype=torch.int64, device=jtemp.device).reshape(
        (1, ngpt, 1) if level_major else (ngpt, 1, 1))
    p_indices = (jpress_arg - 2, jpress_arg - 1)
    t_indices = (jtemp - 1, jtemp)
    flavor_indices = (_flavor_indices(flavor, itropo) if not inference and fmajor is not None
                      and (fmajor.requires_grad or col_mix.requires_grad) else None)

    # Reverse mode must retain each lookup's index. Only the no-grad eager
    # path can reuse and advance an owned address buffer between corners.
    reuse_addresses = (not torch.compiler.is_compiling() and inference
                       and kmajor_t.numel() <= 2**31
                       and kmajor_t.shape == planck_fraction_t.shape)

    # Each accumulator keeps ((c0 + c1) + c2) + c3 per temperature, then scale * sum, then the
    # two temperatures in order; corners are added as they are formed (no corner list).
    absorption_total = fraction_total = None
    for itemp, t_index in enumerate(t_indices):
        t_index_gpt = t_index.unsqueeze(g_axis)
        eta_dtype = torch.int32 if kmajor_t.shape[1] <= 2**31 else torch.int64
        eta_hi = _gather_flavor(jeta[:, itemp, :, :].to(dtype=eta_dtype),
                                flavor, itropo, indices=flavor_indices, level_major=level_major)
        linear_index = None
        if reuse_addresses:
            npres = kmajor_t.shape[2]
            offset = (t_index_gpt.to(torch.int32) * (kmajor_t.shape[1] * npres)
                      + p_indices[0].unsqueeze(0).to(torch.int32)) * table_ngpt
            linear_index = eta_hi.mul_(npres * table_ngpt).add_(offset).add_(gpt_index.to(torch.int32))
            del offset
        absorption_sum = fraction_sum = None
        for press_slot, p_index in enumerate(p_indices):
            p_index_gpt = p_index.unsqueeze(g_axis)
            for eta_slot in range(2):
                # The integer gather owns its workspace. Restore it after each pair.
                if reuse_addresses:
                    delta = (npres * table_ngpt if eta_slot
                             else (table_ngpt if press_slot else 0) - npres * table_ngpt)
                    linear_index.add_(delta)
                    eta_index = eta_hi
                else:
                    eta_index = eta_hi.add_(-1 if eta_slot == 0 else 1)
                if fmajor is not None:
                    weight = _gather_flavor(fmajor[:, eta_slot, press_slot, itemp, :, :], flavor, itropo,
                                            indices=flavor_indices)
                else:
                    weight = ((1.0 - fpress) if press_slot == 0 else fpress).unsqueeze(g_axis) * _gather_flavor(
                        fminor[:, eta_slot, itemp, :, :], flavor, itropo, indices=flavor_indices,
                        level_major=level_major)
                table_index = (t_index_gpt, eta_index, p_index_gpt, gpt_index)

                # Consume absorption before gathering the Planck fraction: the
                # two spectral coefficient arrays need not coexist.
                # Advanced indexing owns these buffers. Reuse that workspace;
                # autograd saves any original values required by the reverse pass.
                absorption = _coefficient_corner_tensor(kmajor_t, *table_index, linear_index=linear_index).mul_(weight)
                absorption_sum = absorption if absorption_sum is None else absorption_sum.add_(absorption)
                del absorption
                fraction = _coefficient_corner_tensor(planck_fraction_t, *table_index, linear_index=linear_index).mul_(weight)
                del weight, table_index, eta_index
                fraction_sum = fraction if fraction_sum is None else fraction_sum.add_(fraction)
                del fraction
        del eta_hi, linear_index
        scale = _gather_flavor(col_mix[:, itemp, :, :], flavor, itropo, indices=flavor_indices,
                               level_major=level_major)
        absorption_sum = scale.mul_(absorption_sum)
        del scale
        absorption_total = absorption_sum if absorption_total is None else absorption_total.add_(absorption_sum)
        fraction_total = fraction_sum if fraction_total is None else fraction_total.add_(fraction_sum)
        del absorption_sum, fraction_sum
    if level_major:
        return absorption_total.transpose(0, 1), fraction_total.transpose(0, 1)
    return absorption_total, fraction_total


def _interp3d_lw_major_planck_banded(
    kmajor_t: torch.Tensor, planck_fraction_t: torch.Tensor, itropo: torch.Tensor, jtemp: torch.Tensor,
    jpress_arg: torch.Tensor, jeta: torch.Tensor, fminor: torch.Tensor, fpress: torch.Tensor,
    col_mix: torch.Tensor, band_flavors: tuple[tuple[int, int, int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``_interp3d_lw_major_planck_tensor`` for inference (no derivative, eager), with less live memory.

    Every g-point of a band shares that band's flavor pair (``band_flavors``: first and end g-point, lower
    and upper flavor, host constants of the table), so the flavor-selected eta, weights and column
    mix are formed once per band at ``[nlay, ncol]`` and applied to the band's g-points, instead of as
    ``[ngpt, nlay, ncol]`` gathers. Absorption and Planck fraction then run one after the other.
    Each element sees the same products and the same addition order as the joint form:
    ``((c0 + c1) + c2) + c3`` per temperature, the column mix times the absorption sum, the two
    temperatures in order. The results are the same bits. A corner weight is the plan's ``fmajor``
    element, formed here as it is there: the pressure weight times the ``fminor`` element.
    """
    table_ngpt = kmajor_t.shape[-1]
    nlay, ncol = jtemp.shape
    npres = kmajor_t.shape[2]
    upper = (itropo != 0).reshape(nlay, ncol)
    p_first = jpress_arg - 2
    t_indices = (jtemp - 1, jtemp)

    def per_band(values, band):
        _, _, lower_flavor, upper_flavor = band
        return torch.where(upper, values[upper_flavor], values[lower_flavor])

    def interpolate(coefficient, scale_by_col_mix):
        flat = coefficient.reshape(-1)
        total = None
        linear = torch.empty((table_ngpt, nlay, ncol), dtype=torch.int32, device=jtemp.device)
        for itemp, t_index in enumerate(t_indices):
            offset = (t_index.to(torch.int32) * (kmajor_t.shape[1] * npres) + p_first.to(torch.int32)) * table_ngpt
            for band in band_flavors:
                first, end = band[0], band[1]
                eta = per_band(jeta[:, itemp], band).to(torch.int32)
                gpoints = torch.arange(first, end, dtype=torch.int32, device=jtemp.device).view(-1, 1, 1)
                linear[first:end] = (eta * (npres * table_ngpt) + offset).unsqueeze(0) + gpoints
                del eta, gpoints
            del offset
            subtotal = None
            for press_slot in range(2):
                for eta_slot in range(2):
                    linear.add_(npres * table_ngpt if eta_slot
                                else (table_ngpt if press_slot else 0) - npres * table_ngpt)
                    corner = flat.index_select(0, linear.reshape(-1)).view(table_ngpt, nlay, ncol)
                    pressure = (1.0 - fpress) if press_slot == 0 else fpress
                    for band in band_flavors:
                        weight = pressure * per_band(fminor[:, eta_slot, itemp], band)
                        corner[band[0]:band[1]].mul_(weight)
                        del weight
                    subtotal = corner if subtotal is None else subtotal.add_(corner)
                    del corner
            if scale_by_col_mix:
                # scale * sum, as the joint form (the product commutes bit for bit)
                for band in band_flavors:
                    subtotal[band[0]:band[1]].mul_(per_band(col_mix[:, itemp], band))
            total = subtotal if total is None else total.add_(subtotal)
            del subtotal
        return total

    absorption = interpolate(kmajor_t, True)
    fraction = interpolate(planck_fraction_t, False)
    return absorption, fraction


def _interp3d_lw_major_planck(
    table: ReducedGasOpticsTable,
    plan: _GasInterpolationPlan,
    *, gpt_range: tuple[int, int] | None = None,
    band_flavors: tuple[tuple[int, int, int, int], ...] | None = None,
    fpress: torch.Tensor | None = None,
    level_major: bool = False,
    inference: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``band_flavors`` (``radiation_group.table_metadata``): each band's g-points and flavor pair, for the banded
    eager inference form (``_interp3d_lw_major_planck_banded``); ``level_major``, ``inference``:
    ``_interp3d_lw_major_planck_tensor``."""
    if table.planck_fraction is None:
        raise ValueError("longwave table does not include Planck fractions")
    if (band_flavors is not None and fpress is not None and gpt_range is None and inference and not level_major
            and table.kmajor.numel() <= 2**31 and table.kmajor.shape == table.planck_fraction.shape):
        return _interp3d_lw_major_planck_banded(
            table.kmajor.to(device=plan.itropo.device), table.planck_fraction.to(device=plan.itropo.device),
            plan.itropo, plan.jtemp, plan.jpress_arg, plan.jeta, plan.fminor, fpress, plan.col_mix, band_flavors)
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
    return _interp3d_lw_major_planck_tensor(*arguments, gpt_range=gpt_range, fminor=plan.fminor, fpress=fpress, level_major=level_major, inference=inference)


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
        values = _build_plan_and_lw_major_planck_tensor(*common_args, table.planck_fraction.to(device=play.device, dtype=play.dtype), ntemp=table.ntemp, npres=table.npres, neta=table.neta, coordinates=coordinates)
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
        values = _build_plan_and_shortwave_major_tensor(*common_args, table.solar_source_quiet.to(device=device, dtype=dtype), table.solar_source_facular.to(device=device, dtype=dtype), table.solar_source_sunspot.to(device=device, dtype=dtype), mg, sb, tsi, torch.full((), 0.1495954, dtype=dtype, device=device), torch.full((), 0.00066696, dtype=dtype, device=device), solcon_t, ntemp=table.ntemp, npres=table.npres, neta=table.neta, apply_solcon=apply_solcon, coordinates=coordinates)
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
    level_major: bool = False,
    inference: bool = False,
) -> torch.Tensor:
    """``level_major`` (the compiled inference form): the same elements in a ``[nlay, ngpt, ncol]`` array;
    ``inference``: no derivative is formed."""
    ngpt = coeff_t.shape[-1]
    nlay, ncol = jtemp.shape
    g_axis = 1 if level_major else 0
    shape = (nlay, ngpt, ncol) if level_major else (ngpt, nlay, ncol)
    gpt_index = torch.arange(ngpt, dtype=torch.int64, device=jtemp.device).reshape(
        (1, ngpt, 1) if level_major else (ngpt, 1, 1)).expand(shape)
    result = torch.zeros(shape, dtype=coeff_t.dtype, device=coeff_t.device)
    flavor_indices = (_flavor_indices(flavor, itropo) if not inference
                      and fminor.requires_grad else None)
    # The domain is constant across the four corners. Select table entries
    # before interpolation, not two full spectral interpolants afterward.
    if upper_coeff is not None:
        coefficients = torch.stack((coeff_t, upper_coeff))
        domain = (itropo != 0).to(dtype=torch.int64).unsqueeze(g_axis)
    for itemp in range(2):
        t_index_gpt = (jtemp + itemp - 1).unsqueeze(g_axis).expand_as(gpt_index)
        eta_hi = _gather_flavor(jeta[:, itemp, :, :], flavor, itropo, indices=flavor_indices,
                                level_major=level_major)
        for eta_slot in range(2):
            eta_index = eta_hi - 1 if eta_slot == 0 else eta_hi
            weight = _gather_flavor(fminor[:, eta_slot, itemp, :, :], flavor, itropo, indices=flavor_indices,
                                    level_major=level_major)
            corner = (coeff_t[t_index_gpt, eta_index, gpt_index] if upper_coeff is None
                      else coefficients[domain, t_index_gpt, eta_index, gpt_index])
            result = result + weight * corner
    return result


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
    initial_tau: torch.Tensor | None = None,
    initial_slices: dict[tuple[int, int], torch.Tensor] | None = None,
    return_slices: bool = False,
    floor: float | None = None,
    level_major: bool = False,
    in_place: bool = False,
) -> torch.Tensor | dict[tuple[int, int], torch.Tensor]:
    """The records' minor-gas optical depths of one pressure domain added to ``initial_tau`` (zeros if None).

    Each record covers whole g-point bands: a band's slice accumulates its records in order (Fortran order) and the
    bands are joined once, or, with ``return_slices``, the band slices are returned for the next domain
    (``initial_slices``). ``in_place`` (eager inference, no derivative): each record is added into the storage of
    ``initial_tau`` itself and that tensor is returned (the caller's ``initial_tau`` holds the result): the same
    additions in the same order, without the band slices and the joined copy. ``floor``: the result is
    ``clamp_min(result, floor)``, applied to each part before the join (elementwise, so the same bits), so a compiled
    graph writes one clamped array instead of the join and its clamped copy. ``level_major``: ``initial_tau`` is the
    ``[ngpt, nlay, ncol]`` transposed view of a level-major buffer (the compiled inference major absorption); it is
    updated level-major and the result returned as the same kind of view (slices stay level-major)."""
    if in_place and (initial_tau is None or initial_slices or return_slices):
        raise ValueError("in_place adds into initial_tau and returns it: no slices")
    nlay, ncol = play.shape
    g_axis = 1 if level_major else 0

    def rows(values, first, last):  # g-points first:last
        return values[:, first:last] if level_major else values[first:last]

    domain_scale = (itropo == (0 if lower else 1)).to(dtype=coeff_t.dtype).unsqueeze(g_axis)
    base = (torch.zeros((ngpt, nlay, ncol), dtype=coeff_t.dtype, device=coeff_t.device)
            if initial_tau is None else initial_tau.transpose(0, 1) if level_major else initial_tau)
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
        minor_index = torch.arange(start, start + count, dtype=torch.int64, device=coeff_t.device)
        minor_index = (minor_index.reshape(1, count, 1).expand(nlay, count, ncol) if level_major
                       else minor_index.reshape(count, 1, 1).expand(count, nlay, ncol))
        t0 = (jtemp - 1).unsqueeze(g_axis).expand_as(minor_index)
        t1 = jtemp.unsqueeze(g_axis).expand_as(minor_index)
        e10 = (jeta[iflav, 0, :, :] - 1).unsqueeze(g_axis).expand_as(minor_index)
        e11 = jeta[iflav, 0, :, :].unsqueeze(g_axis).expand_as(minor_index)
        e20 = (jeta[iflav, 1, :, :] - 1).unsqueeze(g_axis).expand_as(minor_index)
        e21 = jeta[iflav, 1, :, :].unsqueeze(g_axis).expand_as(minor_index)
        # The four-corner association of the source: (w00 c00 + w10 c10) + w01 c01 + w11 c11.
        values = (
            fminor[iflav, 0, 0, :, :].unsqueeze(g_axis) * coeff_t[t0, e10, minor_index]
            + fminor[iflav, 1, 0, :, :].unsqueeze(g_axis) * coeff_t[t0, e11, minor_index]
            + fminor[iflav, 0, 1, :, :].unsqueeze(g_axis) * coeff_t[t1, e20, minor_index]
            + fminor[iflav, 1, 1, :, :].unsqueeze(g_axis) * coeff_t[t1, e21, minor_index]
        )
        contribution = scaling.unsqueeze(g_axis) * values * domain_scale.expand_as(values)
        # Fortran updates the existing major absorption after each minor gas.
        if in_place:
            rows(base, gpt_start, gpt_end).add_(contribution)
            del contribution, values
            continue
        key = (gpt_start, gpt_end)
        slices[key] = (slices[key] if key in slices else rows(base, gpt_start, gpt_end)) + contribution

    def spectral(values):
        return values.transpose(0, 1) if level_major else values

    if in_place:
        return spectral(base if floor is None else base.clamp_min_(floor))
    if return_slices:
        return slices
    parts, position = [], 0
    for start, end in sorted(slices):
        if start < position:
            raise ValueError("minor-gas records must cover disjoint g-point ranges")
        if start > position:
            parts.append(rows(base, position, start))
        parts.append(slices[(start, end)])
        position = end
    if position < ngpt:
        parts.append(rows(base, position, ngpt))
    if floor is not None:
        parts = [part.clamp_min(floor) for part in parts]
    return spectral(torch.cat(parts, dim=g_axis) if len(parts) > 1 else parts[0])


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
    floor: float | None = None,
    level_major: bool = False,
    inference: bool = False,
) -> torch.Tensor:
    """The lower then the upper domain's records added to ``initial_tau`` (``floor``: clamp_min of the result). Eager
    inference adds them into ``initial_tau`` in place and returns it; otherwise the band slices of both domains are
    joined once."""

    boundaries = sorted({edge for record in (*records_lower, *records_upper) for edge in record[:2]})

    def partition(records):
        aligned = []
        for first, last, gas, density, scaling, complement, flavor, offset in records:
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                if first <= left < right <= last:
                    aligned.append((left, right, gas, density, scaling, complement, flavor, offset + left - first))
        return tuple(aligned)

    common = (itropo, jtemp, jeta, fminor, col_gas, play, tlay)
    if inference and not torch.compiler.is_compiling():
        tau = _minor_tau_delta_for_domain_tensor(coeff_lower, *common, partition(records_lower), lower=True,
                                                 ngpt=ngpt, initial_tau=initial_tau, level_major=level_major,
                                                 in_place=True)
        return _minor_tau_delta_for_domain_tensor(coeff_upper, *common, partition(records_upper), lower=False,
                                                  ngpt=ngpt, initial_tau=tau, floor=floor, level_major=level_major,
                                                  in_place=True)
    lower = _minor_tau_delta_for_domain_tensor(coeff_lower, *common, partition(records_lower), lower=True, ngpt=ngpt,
                                               initial_tau=initial_tau, return_slices=True, level_major=level_major)
    return _minor_tau_delta_for_domain_tensor(coeff_upper, *common, partition(records_upper), lower=False, ngpt=ngpt,
                                              initial_tau=initial_tau, initial_slices=lower, floor=floor,
                                              level_major=level_major)


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
    floor: float | None = None,
    level_major: bool = False,
    inference: bool = False,
) -> torch.Tensor:
    """``initial_tau`` plus the minor-gas optical depths of both pressure domains, in the source order (``floor``,
    ``level_major``: ``_minor_tau_delta_for_domain_tensor``). With ``inference`` (no derivative) the eager call adds
    them into ``initial_tau`` in place and returns that tensor: the caller must not read the major optical depth
    again."""

    return _accumulate_minor_tau_tensor(table.kminor_lower.to(device=device, dtype=dtype), table.kminor_upper.to(device=device, dtype=dtype), initial_tau, plan.itropo, plan.jtemp, plan.jeta, plan.fminor, col_gas, play, tlay, table.minor_records_lower, table.minor_records_upper, ngpt=table.ngpt, floor=floor, level_major=level_major, inference=inference)


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
    *,
    g_axis: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``g_axis``: the spectral arrays' g-point axis (1: level-major ``[nlay, ngpt, ncol]``)."""
    tau_rayleigh = torch.where(
        itropo.unsqueeze(g_axis) == 0,
        rayleigh_lower,
        rayleigh_upper,
    ) * moist_plus_dry_column.unsqueeze(g_axis)
    tau_abs = torch.maximum(tau_abs, zero)
    tau = tau_abs + tau_rayleigh
    ssa = torch.where(tau > threshold, tau_rayleigh / torch.where(tau > threshold, tau, 1.0), torch.zeros_like(tau))
    return tau_abs, tau_rayleigh, tau, ssa, torch.zeros_like(tau)


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
    *,
    level_major: bool = False,
    inference: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``level_major``: ``tau_abs`` is the transposed view of a level-major buffer (the compiled inference form); the
    results are level-major too, returned as ``[ngpt, nlay, ncol]`` views: the same elements, each layer's rows read
    once for all g-points."""
    rayleigh = _interp2d_all_gpoints_tensor(
        rayleigh_lower_coeff,
        flavor,
        itropo,
        jtemp,
        jeta,
        fminor,
        upper_coeff=rayleigh_upper_coeff,
        level_major=level_major,
        inference=inference,
    )
    results = _finalize_shortwave_optics_tensor(
        tau_abs.transpose(0, 1) if level_major else tau_abs,
        rayleigh,
        rayleigh,
        itropo,
        moist_plus_dry_column,
        zero,
        threshold,
        g_axis=1 if level_major else 0,
    )
    return tuple(value.transpose(0, 1) for value in results) if level_major else results


def _rayleigh_and_finalize_shortwave(
    table: ReducedGasOpticsTable,
    plan: _GasInterpolationPlan,
    *,
    tau_abs: torch.Tensor,
    moist_plus_dry_column: torch.Tensor,
    level_major: bool = False,
    inference: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if table.rayl_lower is None or table.rayl_upper is None:
        raise ValueError("shortwave gas table does not include Rayleigh coefficients")
    return _rayleigh_and_finalize_shortwave_tensor(table.rayl_lower, table.rayl_upper, _gpoint_flavor(table, plan), plan.jtemp, plan.jeta, plan.fminor, tau_abs, plan.itropo, moist_plus_dry_column, torch.zeros((), dtype=tau_abs.dtype, device=tau_abs.device), 2.0 * torch.finfo(tau_abs.dtype).tiny, level_major=level_major, inference=inference)


__all__ = [
    "DEFAULT_ACTIVE_GASES",
    "ReducedGasOpticsTable",
    "compute_shortwave_solar_source",
    "reduce_gas_optics_table",
]
