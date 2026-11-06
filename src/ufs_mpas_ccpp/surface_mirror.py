# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Source-ordered R4 Sfcprop copyback to native MPAS surface buffers."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch

from .projection import _working_on_device
from .surface_containers import NativeSurface, Sfcprop


@dataclass(frozen=True)
class SurfaceMirrorInputs:
    """Live view of the twelve allocated Sfcprop pointers consumed by R4."""

    tsfcl: torch.Tensor
    tisfc: torch.Tensor
    tsfco: torch.Tensor
    weasdl: torch.Tensor
    snodl: torch.Tensor
    sncovr: torch.Tensor
    weasdi: torch.Tensor
    snodi: torch.Tensor
    sncovr_ice: torch.Tensor
    smc: torch.Tensor
    slc: torch.Tensor
    stc: torch.Tensor

    @classmethod
    def from_sfcprop(cls, source: Sfcprop) -> SurfaceMirrorInputs:
        return cls(**{field.name: getattr(source, field.name) for field in fields(cls)})


@dataclass(frozen=True)
class SurfaceMirror:
    native: NativeSurface
    updated_skin: torch.Tensor
    updated_snow: torch.Tensor
    updated_soil: torch.Tensor
    max_skin_delta: torch.Tensor
    max_snow_delta: torch.Tensor
    max_soil_moisture_delta: torch.Tensor
    max_soil_temperature_delta: torch.Tensor


def mirror_surface(
    native: NativeSurface, sfcprop: SurfaceMirrorInputs, *, n_cells_solve: int
) -> SurfaceMirror:
    """Match ufs_sfcprop_to_mpas for the allocated R4 Noah-MP/NSST members.

    The source calls this after phase1, never after phase2. It preserves all
    unowned rows and unwritten fields. Source guards reject invalid selected
    skin, snow and soil values; they do not clip a returned physics result.
    """
    n = n_cells_solve
    if type(n) is not int or n < 1:
        raise ValueError("surface mirror requires a positive static owned extent")
    device = native.skintemp.device
    if (
        native.landmask.ndim != 1
        or native.landmask.shape[0] < n
        or native.landmask.dtype != torch.int32
        or native.landmask.device != device
    ):
        raise ValueError("surface mirror requires native int32 landmask")

    # Every mirrored field shares the working dtype of the native skin temperature.
    _working_on_device("skintemp", native.skintemp, device)
    dtype = native.skintemp.dtype

    def field(name: str, value: torch.Tensor, rank: int) -> None:
        _working_on_device(name, value, device, dtype)
        if value.ndim != rank or value.shape[0] < n:
            raise ValueError(f"invalid surface mirror field extent: {name}")

    for name in ("skintemp", "sst", "snow", "snowh", "snowc", "xice"):
        field(name, getattr(native, name), 1)
    for name in ("smois", "sh2o", "tslb"):
        field(name, getattr(native, name), 2)
    for name in (
        "tsfcl",
        "tisfc",
        "tsfco",
        "weasdl",
        "snodl",
        "sncovr",
        "weasdi",
        "snodi",
        "sncovr_ice",
    ):
        field(name, getattr(sfcprop, name), 1)
    for name in ("smc", "slc", "stc"):
        field(name, getattr(sfcprop, name), 2)

    land = native.landmask[:n] == 1
    ice = native.xice[:n] >= 0.5
    ocean = (native.landmask[:n] == 0) & (native.xice[:n] < 0.5)
    skin = torch.where(
        land, sfcprop.tsfcl[:n], torch.where(ice, sfcprop.tisfc[:n], sfcprop.tsfco[:n])
    )
    skin_upper = torch.where(
        land, torch.full_like(skin, 400.0), torch.full_like(skin, 350.0)
    )
    torch._assert_async(
        (torch.isfinite(skin) & (skin >= 150.0) & (skin <= skin_upper)).all(),
        "invalid CCPP surface temperature",
    )
    torch._assert_async(
        (
            ~ocean
            | (
                torch.isfinite(sfcprop.tsfco[:n])
                & (sfcprop.tsfco[:n] >= 150.0)
                & (sfcprop.tsfco[:n] <= 350.0)
            )
        ).all(),
        "invalid CCPP sea-surface temperature",
    )
    sst = torch.where(ocean, sfcprop.tsfco[:n], native.sst[:n])
    snow = torch.where(
        land, sfcprop.weasdl[:n], torch.where(ice, sfcprop.weasdi[:n], 0.0)
    )
    snowh = torch.where(
        land,
        0.001 * sfcprop.snodl[:n],
        torch.where(ice, 0.001 * sfcprop.snodi[:n], 0.0),
    )
    snowc = torch.where(
        land, sfcprop.sncovr[:n], torch.where(ice, sfcprop.sncovr_ice[:n], 0.0)
    )
    torch._assert_async(
        (
            torch.isfinite(snow)
            & (snow >= 0.0)
            & torch.isfinite(snowh)
            & (snowh >= 0.0)
            & torch.isfinite(snowc)
            & (snowc >= 0.0)
            & (snowc <= 1.0)
        ).all(),
        "invalid CCPP snow state",
    )
    levels = min(
        native.smois.shape[1],
        native.sh2o.shape[1],
        native.tslb.shape[1],
        sfcprop.smc.shape[1],
        sfcprop.slc.shape[1],
        sfcprop.stc.shape[1],
    )
    smc, slc, stc = (
        getattr(sfcprop, name)[:n, :levels] for name in ("smc", "slc", "stc")
    )
    torch._assert_async(
        (
            ~land[:, None]
            | (
                torch.isfinite(smc)
                & torch.isfinite(slc)
                & torch.isfinite(stc)
                & (smc >= 0.0)
                & (smc <= 1.0)
                & (slc >= 0.0)
                & (slc <= smc + 1.0e-12)
                & (stc >= 150.0)
                & (stc <= 400.0)
            )
        ).all(),
        "invalid CCPP soil state",
    )

    def scalar_copy(value: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        return torch.cat((active, value[n:]), dim=0)

    def soil_copy(value: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        owned = torch.cat(
            (
                torch.where(land[:, None], active, value[:n, :levels]),
                value[:n, levels:],
            ),
            dim=1,
        )
        return torch.cat((owned, value[n:]), dim=0)

    # Mask inactive operands before reductions. Native ocean soil can be undefined.
    def land_delta(new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
        return torch.where(land[:, None], new, 0.0) - torch.where(
            land[:, None], old[:n, :levels], 0.0
        )

    moisture_delta = torch.cat(
        (land_delta(smc, native.smois), land_delta(slc, native.sh2o)), dim=1
    )
    temperature_delta = land_delta(stc, native.tslb)
    max_moisture = moisture_delta.abs().amax() if levels else skin.new_zeros(())
    max_temperature = temperature_delta.abs().amax() if levels else skin.new_zeros(())
    return SurfaceMirror(
        replace(
            native,
            skintemp=scalar_copy(native.skintemp, skin),
            sst=scalar_copy(native.sst, sst),
            snow=scalar_copy(native.snow, snow),
            snowh=scalar_copy(native.snowh, snowh),
            snowc=scalar_copy(native.snowc, snowc),
            smois=soil_copy(native.smois, smc),
            sh2o=soil_copy(native.sh2o, slc),
            tslb=soil_copy(native.tslb, stc),
        ),
        torch.full((), n, dtype=torch.int32, device=device),
        torch.full((), n, dtype=torch.int32, device=device),
        land.to(torch.int32).sum(dtype=torch.int32) * levels,
        (skin - native.skintemp[:n]).abs().amax(),
        (snow - native.snow[:n]).abs().amax(),
        max_moisture,
        max_temperature,
    )
