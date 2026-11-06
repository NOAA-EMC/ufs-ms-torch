# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Selected R4 uncoupled Noah-MP surface composites pre-call: nonfractional and frac_grid (GFS v17)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

import torch

from .clock import _nint_nonnegative
from .containers import Statein
from .surface_containers import Sfcprop


@dataclass(frozen=True)
class SurfaceCompositesControl:
    frac_grid: bool
    cplflx: bool
    cplice: bool
    use_cdeps_data: bool
    lkm: int
    lsm: int
    lsm_ruc: int
    min_seaice: float
    min_lakeice: float
    huge: float

    def __post_init__(self):
        for name in ("frac_grid", "cplflx", "cplice", "use_cdeps_data"):
            if type(getattr(self, name)) is not bool:
                raise TypeError("surface composite controls must be static booleans")
        if any((self.cplflx, self.cplice, self.use_cdeps_data)):
            raise NotImplementedError(
                "only the selected R4 uncoupled surface policy is ported (frac_grid either way)"
            )
        if any(type(v) is not int for v in (self.lkm, self.lsm, self.lsm_ruc)):
            raise TypeError("surface model identifiers must be static integers")
        if self.lkm != 0 or self.lsm != 2 or self.lsm == self.lsm_ruc:
            raise NotImplementedError(
                "only the selected R4 Noah-MP/no-lake-model policy is ported"
            )
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in (self.min_seaice, self.min_lakeice, self.huge)
        ):
            raise ValueError("invalid source surface composite thresholds")


@dataclass(frozen=True)
class SurfaceCompositesResult:
    sfcprop: Sfcprop
    interstitial: Mapping[str, torch.Tensor]


def prepare_surface_composites(
    statein: Statein,
    sfcprop: Sfcprop,
    interstitial: Mapping[str, torch.Tensor],
    control: SurfaceCompositesControl,
) -> SurfaceCompositesResult:
    """Source writes before dcyc2t3; caller supplies freshly reset group scratch."""
    torch._assert_async(
        (sfcprop.use_lake_model == 0).all(), "unexpected active lake model"
    )
    torch._assert_async((sfcprop.slmsk >= 0).all(), "negative source surface mask")
    if control.frac_grid:
        # frac_grid branch (cice is the ice fraction of the water area); cplflx is off.
        frland = sfcprop.landfrac
        water_part = frland < 1.0
        land = ~water_part
        ocean = water_part & (sfcprop.oceanfrac > 0.0)
        lake = water_part & ~ocean
        dry = torch.where(frland > 0.0, True, interstitial["dry"])
    else:
        islmsk = interstitial["islmsk"]
        land = islmsk == 1
        water_part = ~land
        ocean = (~land) & (sfcprop.oceanfrac > 0.0)
        lake = (~land) & (~ocean)
        frland = land.to(sfcprop.oceanfrac.dtype)
        dry = torch.where(land, True, interstitial["dry"])
    threshold = torch.where(
        ocean,
        control.min_seaice,
        torch.full_like(sfcprop.fice, control.min_lakeice),
    )
    icy = water_part & (sfcprop.fice >= threshold)
    fice = torch.where(icy, sfcprop.fice, 0.0)
    hice = torch.where(icy, sfcprop.hice, 0.0)
    wet = torch.where(ocean, fice < 1.0, interstitial["wet"])
    wet = torch.where(lake & (fice < 1.0), True, wet)
    if control.frac_grid:
        # Some open water sets wet; ice-covered water keeps it; all-land clears it.
        wet = torch.where(ocean & (fice >= 1.0), interstitial["wet"], wet)
        wet = torch.where(land, False, wet)
    islmsk = torch.where(land, 1, torch.where(icy, 2, 0)).to(torch.int32)
    islmsk_cice = torch.where(
        water_part, islmsk, 1 if control.frac_grid else interstitial["islmsk_cice"]
    )
    flag_cice = torch.where(water_part, False, False if control.frac_grid else interstitial["flag_cice"])
    tsfco = torch.where(
        water_part & icy & (fice < 1.0),
        torch.clamp_min(sfcprop.tisfc, 271.2),
        sfcprop.tsfco,
    )
    if not control.frac_grid:
        tsfco = torch.where(ocean & wet & (tsfco < 0.0), statein.tgrs[:, 0], tsfco)
    scratch = dict(interstitial)
    scratch.update(
        dry=dry,
        wet=wet,
        icy=icy,
        islmsk=islmsk,
        islmsk_cice=islmsk_cice,
        flag_cice=flag_cice,
        frland=frland,
    )
    for suffix in ("water", "land", "ice"):
        scratch[f"tprcp_{suffix}"] = sfcprop.tprcp
    for suffix, active, temperature in (
        ("water", wet, tsfco),
        ("land", dry, sfcprop.tsfcl),
        ("ice", icy, sfcprop.tisfc),
    ):
        scratch[f"tsurf_{suffix}"] = torch.where(
            active, temperature, interstitial[f"tsurf_{suffix}"]
        )
        friction_active = (icy | wet) if suffix == "ice" else active
        scratch[f"uustar_{suffix}"] = torch.where(
            friction_active, sfcprop.uustar, interstitial[f"uustar_{suffix}"]
        )
    scratch["tsfc_water"] = torch.where(wet, tsfco, interstitial["tsfc_water"])
    for name in ("ep1d_ice", "gflx_ice"):
        scratch[name] = torch.where(icy, 0.0, interstitial[name])
    # Snow on ice (kdt == 1 or uncoupled: always here). frac_grid: ice in a cell with some land
    # keeps the land snow and gets the rest per unit ice area; ice alone as in the nonfractional case.
    alone = icy & ~dry if control.frac_grid else icy
    inverse_ice = 1.0 / torch.where(icy, fice, 1.0)
    snow_ice = sfcprop.snowd * inverse_ice
    water_ice = sfcprop.weasd * inverse_ice
    weasd_land = torch.where(dry, sfcprop.weasd, sfcprop.weasdl)
    if control.frac_grid:
        shared = icy & dry
        inverse_shared = 1.0 / torch.where(shared, fice * (1.0 - frland), 1.0)
        snow_ice = torch.where(shared, torch.clamp_min(
            (sfcprop.snowd - sfcprop.snodl * frland) * inverse_shared, 0.0), snow_ice)
        water_ice = torch.where(shared, torch.clamp_min(
            (sfcprop.weasd - weasd_land * frland) * inverse_shared, 0.0), water_ice)
    return SurfaceCompositesResult(
        replace(
            sfcprop,
            fice=fice,
            hice=hice,
            tsfco=tsfco,
            tsfcl=torch.where(dry, sfcprop.tsfcl, control.huge),
            zorlw=torch.where(
                wet, torch.clamp(sfcprop.zorlw, min=1.0e-5, max=1.0), control.huge
            ),
            zorll=torch.where(dry, sfcprop.zorll, control.huge),
            zorli=torch.where(
                icy, torch.clamp(sfcprop.zorli, min=1.0e-5, max=1.0), control.huge
            ),
            weasdl=torch.where(alone, 0.0, weasd_land),
            weasdi=torch.where(icy, water_ice, sfcprop.weasdi),
            snodl=torch.where(alone, 0.0, sfcprop.snodl),
            snodi=torch.where(icy, snow_ice, sfcprop.snodi),
            slmsk=torch.where(
                _nint_nonnegative(sfcprop.slmsk) != 1,
                islmsk.to(sfcprop.slmsk.dtype),
                sfcprop.slmsk,
            ),
        ),
        scratch,
    )
