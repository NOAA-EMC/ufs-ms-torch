# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""One-time R4 MPAS-to-CCPP surface initialization, without host numerics."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import torch

from .containers import Statein
from .precision import WORKING_DTYPES
from .surface_containers import (
    R4_SURFACE_INTEGER_MEMBERS,
    R4_SURFACE_TAIL_SHAPES,
    NativeSurface,
    Sfcprop,
    SoilParameters,
    SurfaceControl,
)


@dataclass(frozen=True)
class SurfaceInitialization:
    sfcprop: Sfcprop
    radiation_albedo: torch.Tensor


def clamp_surface_fraction(value: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.where(value > 1.0, 0.01 * value, value), 0.0, 1.0)


def bounded_surface_temperature(
    value: torch.Tensor, fallback: torch.Tensor
) -> torch.Tensor:
    def valid(x: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(x) & (x >= 150.0) & (x <= 380.0)

    temperature = torch.where(valid(value), value, fallback)
    return torch.where(valid(temperature), temperature, 273.15)


def initialize_surface(
    native: NativeSurface,
    physics: Statein,
    target: Sfcprop,
    radiation_albedo: torch.Tensor,
    control: SurfaceControl,
    soil: SoilParameters,
) -> SurfaceInitialization:
    """Execute an explicit cold-start projection, never a timestep refresh.

    The R4 host calls this before physics_init, then again after physics_init.
    Only the second cold-start call reconstructs liquid water and glacier state.
    Restart sidecar restoration follows this operator and owns evolved values.
    """
    n = control.n_cells_solve
    if type(n) is not int or n <= 0 or n > physics.tgrs.shape[0]:
        raise ValueError("invalid static surface owned extent")
    if type(control.initialize_noahmp_liquid) is not bool:
        raise TypeError("initialize_noahmp_liquid must be a static bool")
    if any(type(x) is not int for x in (control.jdat_month, control.imn, control.ntqv)):
        raise TypeError("surface month and tracer controls must be static integers")
    device, real = physics.tgrs.device, physics.tgrs.dtype
    if (
        physics.tgrs.ndim != 2
        or real not in WORKING_DTYPES
        or physics.qgrs.ndim != 3
        or physics.qgrs.dtype != real
        or physics.qgrs.device != device
        or physics.qgrs.shape[:2] != physics.tgrs.shape
    ):
        raise ValueError("invalid surface atmospheric Statein")
    integer_fields = {"isltyp", "ivgtyp", "landmask"}
    matrix_fields = {"sh2o", "smois", "tslb", "greenfrac", "albedo12m"}
    for field in fields(native):
        value = getattr(native, field.name)
        dtype = torch.int32 if field.name in integer_fields else real
        if value.device != device or value.dtype != dtype:
            raise ValueError(f"invalid surface dtype/device: {field.name}")
        if (
            value.ndim != (2 if field.name in matrix_fields else 1)
            or value.shape[0] < n
        ):
            raise ValueError(f"invalid column-first surface shape: {field.name}")
    for field in fields(target):
        value = getattr(target, field.name)
        tail = R4_SURFACE_TAIL_SHAPES.get(field.name, ())
        dtype = (
            torch.int32 if field.name in R4_SURFACE_INTEGER_MEMBERS else real
        )
        if (
            value.device != device
            or value.dtype != dtype
            or value.ndim != 1 + len(tail)
            or value.shape[0] < n
            or value.shape[1:] != tail
        ):
            raise ValueError(f"invalid R4 Sfcprop shape/dtype/device: {field.name}")
    if (
        control.dzs.shape != (4,)
        or control.dzs.dtype != real
        or control.dzs.device != device
    ):
        raise ValueError("R4 dzs must have four soil thicknesses in the working dtype")
    if (
        radiation_albedo.ndim != 1
        or radiation_albedo.shape[0] < n
        or radiation_albedo.dtype != real
        or radiation_albedo.device != device
    ):
        raise ValueError("invalid allocated Radtend sfalb")
    for field in fields(soil):
        value = getattr(soil, field.name)
        if (
            value.ndim != 1
            or value.device != device
            or value.dtype != real
            or value.shape != soil.bexp.shape
        ):
            raise ValueError("invalid initialized Noah-MP soil table")
    if any(getattr(native, name).shape[1] < 4 for name in ("sh2o", "smois", "tslb")):
        raise ValueError("R4 native surface requires four soil layers")

    def x(name: str) -> torch.Tensor:
        return getattr(native, name)[:n]

    air = physics.tgrs[:n, 0]
    skin = bounded_surface_temperature(x("skintemp"), air)
    sst = bounded_surface_temperature(x("sst"), skin)
    qv = (
        torch.clamp(physics.qgrs[:n, 0, control.ntqv - 1], 0.0, 0.08)
        if 0 < control.ntqv <= physics.qgrs.shape[2]
        else torch.zeros_like(air)
    )
    ice = clamp_surface_fraction(torch.maximum(x("seaice"), x("xice")))
    land_mask = (x("landmask") == 1) & (ice < 0.5)
    land = land_mask.to(dtype=air.dtype)
    glacier = land_mask & (x("ivgtyp") == 15) & control.initialize_noahmp_liquid
    month = control.jdat_month if 1 <= control.jdat_month <= 12 else control.imn
    month = month if 1 <= month <= 12 else 1

    def seasonal(name: str, fallback: str) -> torch.Tensor:
        value = x(name)
        return clamp_surface_fraction(
            value[:, month - 1] if value.shape[1] >= month else x(fallback)
        )

    albedo = seasonal("albedo12m", "sfc_albbck")
    green = seasonal("greenfrac", "vegfra")
    lai = torch.clamp_min(6.0 * green, 0.05)
    sai = torch.clamp_min(0.10 * lai, 0.05)
    carbon = 100.0 * lai
    water = torch.clamp_min(x("snow"), 0.0)
    depth_mm = 1000.0 * torch.clamp_min(x("snowh"), 0.0)
    depth_mm = torch.where((water > 0.0) & (depth_mm == 0.0), 10.0 * water, depth_mm)
    water = torch.where(glacier, 600.0, water)
    depth_mm = torch.where(glacier, 2000.0, depth_mm)
    depth = 0.001 * depth_mm
    snowxy = torch.where(
        depth < 0.025,
        torch.zeros_like(depth),
        torch.where(
            depth <= 0.05,
            torch.full_like(depth, -1.0),
            torch.where(
                depth <= 0.25,
                torch.full_like(depth, -2.0),
                torch.full_like(depth, -3.0),
            ),
        ),
    )
    dz_top = torch.where(
        depth > 0.25, torch.full_like(depth, 0.05), torch.zeros_like(depth)
    )
    dz_middle = torch.where(
        depth <= 0.05,
        0.0,
        torch.where(
            depth <= 0.10,
            0.5 * depth,
            torch.where(
                depth <= 0.25,
                0.05,
                torch.where(depth <= 0.45, 0.5 * (depth - 0.05), 0.20),
            ),
        ),
    )
    dz_bottom = torch.where(
        depth < 0.025,
        0.0,
        torch.where(
            depth <= 0.05,
            depth,
            torch.where(
                depth <= 0.10,
                0.5 * depth,
                torch.where(
                    depth <= 0.25,
                    depth - 0.05,
                    torch.where(depth <= 0.45, 0.5 * (depth - 0.05), depth - 0.25),
                ),
            ),
        ),
    )
    torch._assert_async(
        (torch.isfinite(x("ufs_sfc_znt")) & (x("ufs_sfc_znt") > 0.0)).all(),
        "invalid analyzed znt",
    )
    updates = {}

    def set_members(value: torch.Tensor | float | int, *names: str) -> None:
        for name in names:
            old = getattr(target, name)
            written = torch.as_tensor(value, dtype=old.dtype, device=device).expand_as(
                old[:n]
            )
            updates[name] = torch.cat((written, old[n:]), dim=0)

    set_members(torch.where((ice >= 0.5) & ~land_mask, 2.0, land), "slmsk")
    set_members(land, "landfrac")
    set_members(torch.clamp_min(1.0 - land, 0.0), "oceanfrac")
    set_members(skin, "tsfc", "tsfcl", "tvxy", "tgxy")
    set_members(sst, "tsfco", "tref")
    ti = bounded_surface_temperature(x("ufs_sfc_tisfc"), skin)
    set_members(ti, "tisfc")
    set_members(ti[:, None], "tiice")
    set_members(ice, "fice")
    set_members(torch.clamp_min(x("ufs_sfc_hice"), 0.0), "hice")
    set_members(water, "snowd", "weasd", "weasdl", "weasdi", "sneqvoxy")
    set_members(clamp_surface_fraction(x("snowc")), "sncovr", "sncovr_ice")
    set_members(depth_mm, "snodl", "snodi")
    for name in ("snoalb", "shdmin", "shdmax"):
        set_members(clamp_surface_fraction(x(name)), name)
    for name in ("alvsf", "alnsf", "alvwf", "alnwf"):
        set_members(clamp_surface_fraction(x("ufs_sfc_" + name)), name)
    set_members(
        albedo,
        "albdirvis_lnd",
        "albdirnir_lnd",
        "albdifvis_lnd",
        "albdifnir_lnd",
        "alboldxy",
    )
    set_members(1.0, "facsf", "slope", "slope_save")
    set_members(4, "scolor", "scolor_save")
    set_members(0.97, "emis_lnd")
    set_members(0.98, "emis_ice", "semisbase")
    set_members(0.99, "emis_wat")
    set_members(0.01, "zorlw", "zorlwav")
    set_members(10.0, "zorll")
    set_members(0.10, "zorli")
    set_members(100.0 * x("ufs_sfc_znt"), "zorl")
    set_members(x("ivgtyp"), "vtype", "vtype_save")
    set_members(x("isltyp"), "stype", "stype_save")
    set_members(green, "vfrac")
    tg3 = bounded_surface_temperature(x("ufs_sfc_tg3"), x("tmn"))
    set_members(torch.where(glacier, torch.clamp_max(tg3, 263.15), tg3), "tg3")
    set_members(x("ter"), "oro", "oro_uf")
    set_members(torch.clamp_min(x("ufs_sfc_ust"), 0.0), "uustar")
    set_members(qv, "qss")
    canopy = torch.where(land_mask, torch.clamp_min(x("ufs_sfc_canwat"), 0.0), 0.0)
    set_members(canopy, "canopy", "canliqxy")
    for name in ("ffmm", "ffhh", "f10m"):
        set_members(torch.clamp_min(x("ufs_sfc_" + name), 0.0), name)
    set_members(bounded_surface_temperature(x("ufs_sfc_t2m"), air), "t2m", "th2m")
    set_members(torch.clamp(x("ufs_sfc_q2"), 0.0, 0.08), "q2m")
    set_members(100.0, "rca")
    set_members(1.0e-3, "flhc", "flqc", "cmxy", "chxy")
    set_members(snowxy, "snowxy")
    set_members(1000.0, "eahxy")
    set_members(air, "tahxy")
    set_members(2.0, "zwtxy")
    set_members(carbon, "lfmassxy", "rtmassxy")
    set_members(0.5 * carbon, "stmassxy", "woodxy")
    set_members(lai, "xlaixy")
    set_members(sai, "xsaixy")
    set_members(0.20, "smcwtdxy")
    set_members(
        0.0,
        "lakefrac",
        "lakedepth",
        "use_lake_model",
        "usfco",
        "vsfco",
        "facwf",
        "evap",
        "hflx",
        "tprcp",
        "srflag",
        "rmol",
        "raincprv",
        "rainncprv",
        "iceprv",
        "snowprv",
        "graupelprv",
        "draincprv",
        "drainncprv",
        "diceprv",
        "dsnowprv",
        "dgraupelprv",
        "canicexy",
        "fwetxy",
        "qsnowxy",
        "wslakexy",
        "waxy",
        "wtxy",
        "stblcpxy",
        "fastcpxy",
        "taussxy",
        "deeprechxy",
        "rechxy",
        "z_c",
        "c_0",
        "c_d",
        "w_0",
        "w_d",
        "xt",
        "xs",
        "d_conv",
        "ifd",
        "dt_cool",
        "qrain",
    )

    for name, width in (("vegtype_frac", 20), ("soiltype_frac", 19)):
        category = (
            x("ivgtyp" if name == "vegtype_frac" else "isltyp").clamp(1, width) - 1
        )
        set_members(
            (category[:, None] == torch.arange(width, device=device)[None, :]).to(
                physics.tgrs.dtype
            ),
            name,
        )

    for name in ("sh2o", "smois", "tslb"):
        torch._assert_async(
            (
                torch.isfinite(x(name)[:, :4]) & (torch.abs(x(name)[:, :4]) <= 1.0e30)
            ).all(),
            f"invalid MPAS {name}",
        )
    liquid = torch.clamp(x("sh2o")[:, :4], 0.0, 1.0)
    total = torch.clamp(x("smois")[:, :4], 0.0, 1.0)
    if control.initialize_noahmp_liquid:
        soil_idx = (x("isltyp").clamp(1, soil.bexp.numel()) - 1).to(torch.int64)
        b = soil.bexp[soil_idx][:, None]
        maximum = soil.smcmax[soil_idx][:, None]
        psi = soil.psisat[soil_idx][:, None]
        torch._assert_async(
            (~land_mask[:, None] | ((b > 0.0) & (maximum > 0.0) & (psi > 0.0))).all(),
            "invalid initialized Noah-MP soil parameters",
        )
        active_frozen = land_mask[:, None] & (x("tslb")[:, :4] < 273.15 - 1.0e-3)
        # Safe inactive operands preserve the source branch and reverse-mode AD (A1).
        safe_t = torch.where(active_frozen, x("tslb")[:, :4], 260.0)
        safe_b = torch.where(active_frozen, b, 1.0)
        safe_psi = torch.where(active_frozen, psi, 1.0)
        fk = ((333580.0 / (9.80665 * (-safe_psi))) * ((safe_t - 273.15) / safe_t)) ** (
            -1.0 / safe_b
        ) * torch.where(active_frozen, maximum, 1.0)
        reconstructed_total = torch.minimum(
            torch.clamp_min(x("smois")[:, :4], 0.0), maximum
        )
        reconstructed_liquid = torch.where(
            active_frozen,
            torch.minimum(torch.clamp_min(fk, 0.02), reconstructed_total),
            reconstructed_total,
        )
        liquid = torch.where(land_mask[:, None], reconstructed_liquid, liquid)
        total = torch.where(land_mask[:, None], torch.minimum(total, maximum), total)
    liquid = torch.where(glacier[:, None], 0.0, liquid)
    total = torch.where(glacier[:, None], 1.0, total)
    temperature = bounded_surface_temperature(x("tslb")[:, :4], skin[:, None])
    glacier_t = torch.minimum(
        bounded_surface_temperature(x("tmn"), skin), torch.full_like(skin, 263.15)
    )
    temperature = torch.where(
        glacier[:, None], torch.minimum(temperature, glacier_t[:, None]), temperature
    )
    set_members(liquid, "slc")
    set_members(total, "smc")
    set_members(temperature, "stc")
    set_members(
        torch.where(glacier[:, None], 1.0, torch.clamp(x("smois")[:, :4], 0.0, 1.0)),
        "smoiseq",
    )

    ice_layers, snow_temperatures, snow_depths = [], [], []
    cumulative = torch.zeros_like(depth)
    for level, dz in zip((-2, -1, 0), (dz_top, dz_middle, dz_bottom)):
        active = snowxy < float(level)
        mass_active = active & (depth > 0.0) & (water > 0.0) & (dz > 0.0)
        safe_depth = torch.where(mass_active, depth, 1.0)
        ice_layers.append(torch.where(mass_active, water * dz / safe_depth, 0.0))
        snow_temperatures.append(
            torch.where(active, torch.clamp_max(skin, 273.16), skin)
        )
        cumulative = torch.where(active, cumulative - dz, cumulative)
        snow_depths.append(torch.where(active, cumulative, 0.0))
    cumulative = torch.minimum(cumulative, snow_depths[-1])
    for layer in range(4):
        cumulative = cumulative - control.dzs[layer]
        snow_depths.append(cumulative)
    set_members(torch.stack(ice_layers, dim=1), "snicexy")
    set_members(0.0, "snliqxy")
    set_members(torch.stack(snow_temperatures, dim=1), "tsnoxy")
    set_members(torch.stack(snow_depths, dim=1), "zsnsoxy")

    theta, gamma, sigma, elevation = (
        x("gwd_" + name) for name in ("theta", "gamma", "sigma", "elvmax")
    )
    torch._assert_async(
        (torch.isfinite(theta) & (torch.abs(theta) <= 0.5 * math.pi)).all(),
        "invalid GWD principal-axis angle",
    )
    torch._assert_async(
        (torch.isfinite(gamma) & (gamma >= 0.0) & (gamma <= 1.0)).all(),
        "invalid GWD anisotropy",
    )
    torch._assert_async(
        (torch.isfinite(sigma) & (sigma >= 0.0)).all(), "invalid GWD slope"
    )
    torch._assert_async(
        (torch.isfinite(elevation) & (elevation >= x("ter"))).all(),
        "invalid GWD maximum elevation",
    )
    terrain_names = (
        "var2d",
        "con",
        "oa1",
        "oa2",
        "oa3",
        "oa4",
        "ol1",
        "ol2",
        "ol3",
        "ol4",
    )
    hprime = [x(name) for name in terrain_names]
    hprime.extend((theta * 180.0 / math.pi, gamma, sigma, elevation))
    hprime.extend(x(name + "ss") for name in terrain_names)
    set_members(torch.stack(hprime, dim=1), "hprime")
    return SurfaceInitialization(
        replace(target, **updates), torch.cat((albedo, radiation_albedo[n:]), dim=0)
    )
