# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Selected R4 cold Noah-MP state written by GFS_phys_time_vary_init."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch

from .surface_containers import (
    R4_SURFACE_INTEGER_MEMBERS,
    R4_SURFACE_TAIL_SHAPES,
    Sfcprop,
)


@dataclass(frozen=True)
class LandInitializationTables:
    isbarren: int
    isice: int
    isurban: int
    iswater: int
    laim: torch.Tensor
    sla: torch.Tensor
    bexp: torch.Tensor
    smcmax: torch.Tensor
    dwsat: torch.Tensor
    dksat: torch.Tensor
    psisat: torch.Tensor


def _equilibrium_moisture(
    bexp: torch.Tensor,
    dwsat: torch.Tensor,
    dksat: torch.Tensor,
    ddz: torch.Tensor,
    maximum: torch.Tensor,
) -> torch.Tensor:
    """Source Newton order and per-element first-convergence return, without sync."""
    expon = bexp + 1.0
    aa = dwsat / ddz
    bb = dksat / maximum**expon
    moisture = 0.5 * maximum
    active = torch.ones_like(moisture, dtype=torch.bool)
    for _ in range(100):
        func = (moisture - maximum) * aa + bb * moisture**expon
        derivative = aa + bb * expon * moisture**bexp
        dx = func / derivative
        moisture = torch.where(active, moisture - dx, moisture)
        active = active & (torch.abs(dx) >= 1.0e-6)
    return moisture


def initialize_noahmp_cold_surface(
    target: Sfcprop,
    tables: LandInitializationTables,
    *,
    month: int,
    zs: torch.Tensor,
    dzs: torch.Tensor,
    n_cells_solve: int,
) -> Sfcprop:
    """Port the cold land branch, not the full CCPP lifecycle or restart path.

    The caller supplies the first MPAS surface projection, initialized tables and
    static source controls. This branch precedes radiation and physics init.
    All nonland missing values and all source-unwritten members are preserved.
    """
    n = n_cells_solve
    if type(n) is not int or not 0 < n <= target.landfrac.shape[0]:
        raise ValueError("invalid cold land owned extent")
    if type(month) is not int or not 1 <= month <= 12:
        raise ValueError("invalid cold land month")
    device = target.landfrac.device
    for f in fields(target):
        value = getattr(target, f.name)
        tail = R4_SURFACE_TAIL_SHAPES.get(f.name, ())
        dtype = torch.int32 if f.name in R4_SURFACE_INTEGER_MEMBERS else target.landfrac.dtype
        if (
            value.device != device
            or value.dtype != dtype
            or value.ndim != 1 + len(tail)
            or value.shape[0] < n
            or value.shape[1:] != tail
        ):
            raise ValueError(f"invalid R4 Sfcprop shape/dtype/device: {f.name}")
    if zs.shape != (4,) or dzs.shape != (4,):
        raise ValueError("R4 cold land requires four soil layers")
    for f in fields(tables):
        value = getattr(tables, f.name)
        if isinstance(value, torch.Tensor):
            if value.dtype != target.landfrac.dtype or value.device != device:
                raise ValueError(f"invalid cold land table dtype/device: {f.name}")
    for value in (zs, dzs):
        if value.dtype != target.landfrac.dtype or value.device != device:
            raise ValueError("invalid cold land soil geometry dtype/device")
    if tables.laim.ndim != 2 or tables.laim.shape[1] != 12:
        raise ValueError("invalid monthly leaf-area table")
    if tables.sla.shape != (tables.laim.shape[0],):
        raise ValueError("invalid specific leaf-area table")
    for name in ("bexp", "smcmax", "dwsat", "dksat", "psisat"):
        if (
            getattr(tables, name).shape != tables.smcmax.shape
            or tables.smcmax.ndim != 1
        ):
            raise ValueError("inconsistent cold land soil tables")

    def x(name: str) -> torch.Tensor:
        return getattr(target, name)[:n]

    land = x("landfrac") >= 1.0e-4
    veg = torch.where(x("vtype") == 0, 7, x("vtype")).to(torch.int64)
    soil = x("stype").to(torch.int64)
    torch._assert_async(
        (~land | ((veg >= 1) & (veg <= tables.sla.numel()))).all(),
        "invalid active cold land vegetation category",
    )
    torch._assert_async(
        (~land | ((soil >= 1) & (soil <= tables.smcmax.numel()))).all(),
        "undefined active cold land soil category",
    )
    vi = torch.where(land, veg - 1, 0)
    si = torch.where(land, soil - 1, 0)
    glacier = veg == tables.isice
    barren = (
        glacier
        | (veg == tables.isbarren)
        | (veg == tables.isurban)
        | (veg == tables.iswater)
    )
    updates = {}

    def write(name: str, value: torch.Tensor | float, *, land_only: bool = True):
        old = getattr(target, name)
        value = torch.as_tensor(value, dtype=old.dtype, device=device).expand_as(
            old[:n]
        )
        if land_only:
            mask = land.reshape((n,) + (1,) * (value.ndim - 1))
            value = torch.where(mask, value, 9.99e20)
        updates[name] = torch.cat((value, old[n:]), dim=0)

    for name in ("albdirvis_lnd", "albdirnir_lnd", "albdifvis_lnd", "albdifnir_lnd"):
        write(name, 0.2, land_only=False)
    write("emis_lnd", 0.95, land_only=False)
    temperature = torch.where(
        (x("snowd") > 0.01) & (x("tsfcl") > 273.15), 273.15, x("tsfcl")
    )
    for name in ("tvxy", "tgxy", "tahxy"):
        write(name, temperature)
    for name in (
        "canicexy",
        "cmxy",
        "chxy",
        "fwetxy",
        "qsnowxy",
        "wslakexy",
        "taussxy",
        "deeprechxy",
        "rechxy",
    ):
        write(name, 0.0)
    write("canliqxy", x("canopy"))
    write("eahxy", 2000.0)
    write("sneqvoxy", x("weasd"))
    write("alboldxy", 0.65)
    write("waxy", 4900.0)
    write("wtxy", 4900.0)
    water = torch.full_like(x("waxy"), 4900.0)
    write("zwtxy", (25.0 + 2.0) - water / 1000.0 / 0.2)
    lai = torch.where(barren, 0.0, torch.clamp_min(tables.laim[vi, month - 1], 0.05))
    sai = torch.where(barren, 0.0, torch.clamp_min(lai * 0.1, 0.05))
    masslai = torch.full_like(lai, 1000.0) / torch.clamp_min(tables.sla[vi], 1.0)
    write("xlaixy", lai)
    write("xsaixy", sai)
    write("lfmassxy", lai * masslai)
    write("stmassxy", sai * (1000.0 / 3.0))
    for name, value in (
        ("rtmassxy", 500.0),
        ("woodxy", 500.0),
        ("stblcpxy", 1000.0),
        ("fastcpxy", 1000.0),
    ):
        write(name, torch.where(barren, 0.0, torch.full_like(lai, value)))
    stc = torch.where(
        (land & glacier)[:, None],
        torch.minimum(
            x("stc"),
            torch.minimum(x("tg3"), torch.full_like(x("tg3"), 263.15))[:, None],
        ),
        x("stc"),
    )
    for name, value in (
        ("stc", stc),
        ("smc", torch.where((land & glacier)[:, None], 1.0, x("smc"))),
        ("slc", torch.where((land & glacier)[:, None], 0.0, x("slc"))),
    ):
        write(name, value, land_only=False)

    depth = x("snowd") / 1000.0
    depth = torch.where(
        (x("weasd") != 0.0) & (depth == 0.0), x("weasd") / 1000.0, depth
    )
    depth = torch.where(veg == 15, 2.0, depth)
    swe = torch.where(land & (veg == 15), 600.0, x("weasd"))
    write("weasd", swe, land_only=False)
    snow = torch.where(
        depth < 0.025,
        0.0,
        torch.where(
            depth <= 0.05,
            -1.0,
            torch.where(depth <= 0.25, -2.0, torch.full_like(depth, -3.0)),
        ),
    )
    write("snowxy", snow)
    top = torch.where(depth > 0.25, torch.full_like(depth, 0.05), 0.0)
    middle = torch.where(
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
    bottom = torch.where(
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
                    torch.where(
                        depth <= 0.45, 0.5 * (depth - 0.05), depth - 0.05 - 0.20
                    ),
                ),
            ),
        ),
    )
    cumulative = torch.zeros_like(depth)
    ice, temperatures, depths = [], [], []
    safe_depth = torch.where(land & (depth > 0.0), depth, 1.0)
    for level, thickness in zip((-2, -1, 0), (top, middle, bottom), strict=True):
        active = snow < float(level)
        cumulative = cumulative + torch.where(active, thickness, 0.0)
        temperatures.append(
            torch.where(
                active,
                temperature
                + ((cumulative - 0.5 * thickness) / safe_depth)
                * (stc[:, 0] - temperature),
                0.0,
            )
        )
        ice.append(torch.where(active, (1.0 * thickness) * swe / safe_depth, 0.0))
        depths.append(torch.where(active, -cumulative, 0.0))
    z = -cumulative
    for level in range(4):
        z = z + (-dzs[level])
        depths.append(z)
    write("tsnoxy", torch.stack(temperatures, dim=1))
    write("snicexy", torch.stack(ice, dim=1))
    write("snliqxy", 0.0)
    write("zsnsoxy", torch.stack(depths, dim=1))

    b, maximum, dw, dk, psi = (
        getattr(tables, name)[si]
        for name in ("bexp", "smcmax", "dwsat", "dksat", "psisat")
    )
    maximum = torch.where(veg == tables.isurban, 0.45, maximum)
    active = land & (b > 0.0) & (maximum > 0.0) & (psi > 0.0)
    ddz = torch.stack(
        (-zs[1] * 0.5, (zs[0] - zs[2]) * 0.5, (zs[1] - zs[3]) * 0.5, zs[2] - zs[3])
    )
    safe_b, safe_maximum, safe_dw, safe_dk = (
        torch.where(active, value, 1.0)[:, None] for value in (b, maximum, dw, dk)
    )
    equilibrium = _equilibrium_moisture(
        safe_b, safe_dw, safe_dk, ddz[None, :], safe_maximum
    )
    equilibrium = torch.minimum(
        torch.clamp_min(equilibrium, 1.0e-4), maximum[:, None] * 0.99
    )
    write("smoiseq", torch.where(active[:, None], equilibrium, maximum[:, None]))
    write("smcwtdxy", maximum)
    return replace(target, **updates)
