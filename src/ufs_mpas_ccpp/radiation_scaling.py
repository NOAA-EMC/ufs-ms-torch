# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""R4 dcyc2t3: consume held radiation inside the physics group.

The selected source disables the LW Jacobian adjustment. Its optional Jacobian
branch remains unsupported; it must not silently use unadjusted heating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

import torch
from sa_tke_edmf_torch.diagnostics import add_diagnostic_

from .containers import Statein, Stateout
from .grid import Grid
from .radiation import RadiationCarry
from .surface_containers import Sfcprop
from .runtime_policy import glue
from .precision import working_like


@dataclass(frozen=True)
class RadiationScalingControl:
    dtf: float
    dtp: float
    fhswr: float
    tend_opt_rad_scaler: int
    lssav: bool
    ldiag3d: bool
    lsidea: bool
    use_LW_jacobian: bool
    use_med_flux: bool
    do_sppt: bool
    ca_global: bool
    pert_radtend: bool
    index_of_temperature: int
    index_of_process_longwave: int
    index_of_process_shortwave: int

    def __post_init__(self):
        for name in (
            "lssav",
            "ldiag3d",
            "lsidea",
            "use_LW_jacobian",
            "use_med_flux",
            "do_sppt",
            "ca_global",
            "pert_radtend",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError("radiation scaling switches must be static booleans")
        if self.use_LW_jacobian:
            raise NotImplementedError("R4 LW Jacobian adjustment is not ported")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in (self.dtf, self.dtp, self.fhswr)
        ):
            raise ValueError("radiation scaling intervals must be positive and static")
        if type(
            self.tend_opt_rad_scaler
        ) is not int or self.tend_opt_rad_scaler not in (1, 2, 3, 4):
            raise ValueError("radiation tendency application control must be 1-4")
        if any(
            type(v) is not int or v < 1
            for v in (
                self.index_of_temperature,
                self.index_of_process_longwave,
                self.index_of_process_shortwave,
            )
        ):
            raise ValueError(
                "radiation diagnostic indices use source one-based numbering"
            )


def scale_radiation(
    inputs: Mapping[str, torch.Tensor],
    control: RadiationScalingControl,
) -> dict[str, torch.Tensor]:
    """Column-first source arguments and writes, without a host numerical bridge."""
    ncol, nlev = inputs["gt0"].shape
    device = inputs["gt0"].device
    for name, value in inputs.items():
        if value.device != device or (value.dtype not in (torch.int32, torch.bool)
                                      and not working_like(value, inputs["gt0"])):
            raise ValueError(f"invalid radiation scaling operand {name}")
    for name in ("gt0", "gq0", "dqdt"):
        torch._assert_async(
            torch.isfinite(inputs[name]).all(), f"non-finite dcyc2t3 entry {name}"
        )
    torch._assert_async(
        (inputs["tsflw"] > 0).all(), "invalid held LW surface-air temperature"
    )

    # These loop bounds depend only on the source's static timestep configuration.
    tem1 = control.fhswr / control.dtf
    nstp = max(6, math.floor(tem1) + int(tem1 - math.floor(tem1) >= 0.5))
    ratio = nstp / tem1
    nstl = max(1, math.floor(ratio) + int(ratio - math.floor(ratio) >= 0.5))
    pid12 = math.pi / 12.0
    if nstl == 1:
        cns = (
            pid12 * (inputs["solhr"] + control.dtf * (1.0 / 7200.0) - 12.0)
            + inputs["slag"]
        )
        xcosz = inputs["sdec"] * inputs["sinlat"] + inputs["cdec"] * inputs[
            "coslat"
        ] * torch.cos(cns + inputs["xlon"])
    elif nstl == nstp:
        xcosz = inputs["coszen"]
    else:
        rstl = 1.0 / nstl
        solang = pid12 * (inputs["solhr"] - 12.0)
        anginc = pid12 * control.dtf * (1.0 / 3600.0) * rstl
        xcosz = torch.zeros_like(inputs["coszen"])
        istsun = torch.zeros_like(inputs["coszen"], dtype=torch.int32)
        for it in range(1, nstl + 1):
            cns = solang + (it - 0.5) * anginc + inputs["slag"]
            coszn = inputs["sdec"] * inputs["sinlat"] + inputs["cdec"] * inputs[
                "coslat"
            ] * torch.cos(cns + inputs["xlon"])
            xcosz = xcosz + torch.clamp_min(coszn, 0.0)
            istsun = istsun + (coszn > 0.0001).to(torch.int32)
        xcosz = torch.where(istsun > 0, xcosz / torch.clamp_min(istsun, 1), xcosz)

    daylight = (xcosz > 0.0001) & (inputs["coszen"] > 0.0001)
    xmu = torch.where(
        daylight, xcosz / torch.where(daylight, inputs["coszen"], 1.0), 0.0
    )
    tem1 = inputs["tf"] / inputs["tsflw"]
    tem2 = tem1 * tem1
    down_lw = inputs["sfcdlw"] * tem2 * tem2
    out = dict(inputs)
    out.update(xcosz=xcosz, xmu=xmu, adjsfcdlw=down_lw)
    for surface, mask in (("lnd", "dry"), ("ice", "icy"), ("wat", "wet")):
        active = inputs[mask]
        skin = torch.where(active, inputs[f"tsfc_{surface}"], 1.0)
        squared = skin * skin
        emissivity = inputs[f"sfcemis_{surface}"]
        up_lw = (
            emissivity * 5.670400e-8 * squared * squared + (1.0 - emissivity) * down_lw
        )
        if surface == "wat" and control.use_med_flux:
            up_lw = torch.where(
                inputs["sfculw_med"] > 0.0001, inputs["sfculw_med"], up_lw
            )
        name = f"adjsfculw_{surface}"
        out[name] = torch.where(active, up_lw, inputs[name])
    for output, source in (
        ("adjsfcnsw", "sfcnsw"),
        ("adjsfcdsw", "sfcdsw"),
        ("adjsfcdswc", "sfcdswc"),
        ("adjnirbmu", "sfcnirbmu"),
        ("adjnirdfu", "sfcnirdfu"),
        ("adjvisbmu", "sfcvisbmu"),
        ("adjvisdfu", "sfcvisdfu"),
        ("adjnirbmd", "sfcnirbmd"),
        ("adjnirdfd", "sfcnirdfd"),
        ("adjvisbmd", "sfcvisbmd"),
        ("adjvisdfd", "sfcvisdfd"),
    ):
        out[output] = inputs[source] * xmu

    sw = inputs["swh"] * xmu[:, None]
    out["ten_t"] = sw + inputs["hlw"]
    for name in ("ten_u", "ten_v"):
        out[name] = torch.zeros_like(inputs["gt0"])
    out["ten_q"] = torch.zeros_like(inputs["gq0"])
    if control.do_sppt or control.ca_global:
        if control.pert_radtend:
            out["dtdtnp"] = (
                inputs["dtdtnp"] + inputs["swhc"] * xmu[:, None] + inputs["hlwc"]
            )
        else:
            out["dtdtnp"] = inputs["dtdtnp"] + sw + inputs["hlw"]
    for state, accumulated, tendency in (
        ("gt0", "dtdt", "ten_t"),
        ("gu0", "dudt", "ten_u"),
        ("gv0", "dvdt", "ten_v"),
        ("gq0", "dqdt", "ten_q"),
    ):
        option = control.tend_opt_rad_scaler
        if option == 1:
            out[state] = inputs[state] + control.dtp * out[tendency]
        elif option == 2:
            out[accumulated] = inputs[accumulated] + out[tendency]
        elif option == 3:
            out[state] = inputs[state] + control.dtp * (
                inputs[accumulated] + out[tendency]
            )
            out[accumulated] = torch.zeros_like(inputs[accumulated])

    if control.lssav and control.ldiag3d and not control.lsidea:
        dtend = inputs["dtend"].clone()
        if dtend.shape[:2] != (ncol, nlev) or inputs["dtidx"].dtype != torch.int32:
            raise ValueError("invalid source radiation diagnostic layout")
        for process, heating in (
            (control.index_of_process_longwave, inputs["hlw"]),
            (control.index_of_process_shortwave, sw),
        ):
            index = inputs["dtidx"][control.index_of_temperature - 1, process - 1]
            torch._assert_async(
                (index <= dtend.shape[2]),
                "radiation diagnostic slot outside allocation",
            )
            add_diagnostic_(dtend, heating * control.dtf, index)
        out["dtend"] = dtend
    for name in ("gt0", "gq0", "dqdt"):
        torch._assert_async(
            torch.isfinite(out[name]).all(), f"non-finite dcyc2t3 exit {name}"
        )
    return out


@dataclass(frozen=True)
class RadiationScalingResult:
    stateout: Stateout
    carry: RadiationCarry
    interstitial: Mapping[str, torch.Tensor]


def apply_held_radiation(
    statein: Statein,
    stateout: Stateout,
    grid: Grid,
    sfcprop: Sfcprop,
    carry: RadiationCarry,
    interstitial: Mapping[str, torch.Tensor],
    solar: Mapping[str, torch.Tensor],
    dtidx: torch.Tensor,
    control: RadiationScalingControl,
) -> RadiationScalingResult:
    """Generated-cap bindings; caller owns pre-surface scratch and diagnostic reset."""
    if control.do_sppt or control.ca_global:
        raise NotImplementedError("native Tbd.dtdtnp copyback is not implemented")
    inputs = {name: solar[name] for name in ("solhr", "slag", "sdec", "cdec")}
    inputs.update({name: getattr(grid, name) for name in ("sinlat", "coslat", "xlon")})
    inputs.update(
        coszen=carry.radtend["coszen"],
        tsflw=carry.radtend["tsflw"],
        tf=statein.tgrs[:, 0],
        tsfc_lnd=sfcprop.tsfcl,
        tsfc_ice=sfcprop.tisfc,
        tsfc_wat=interstitial["tsfc_water"],
        dtidx=dtidx,
    )
    for surface in ("lnd", "ice", "wat"):
        inputs[f"sfcemis_{surface}"] = getattr(sfcprop, f"emis_{surface}")
        suffix = {"lnd": "land", "ice": "ice", "wat": "water"}[surface]
        inputs[f"adjsfculw_{surface}"] = interstitial[f"adjsfculw_{suffix}"]
    for name in ("sfcdsw", "sfcnsw", "sfcdswc", "sfcdlw"):
        inputs[name] = carry.coupling[name]
    for wave, held in (
        ("swh", "htrsw"),
        ("hlw", "htrlw"),
        ("swhc", "swhc"),
        ("hlwc", "lwhc"),
    ):
        inputs[wave] = carry.radtend[held]
    for component in (
        "nirbmu",
        "nirdfu",
        "visbmu",
        "visdfu",
        "nirbmd",
        "nirdfd",
        "visbmd",
        "visdfd",
    ):
        inputs[f"sfc{component}"] = carry.coupling[component + "i"]
    for name in ("gt0", "gu0", "gv0", "gq0"):
        inputs[name] = getattr(stateout, name)
    for name in ("dudt", "dvdt", "dtdt", "dqdt", "dry", "icy", "wet"):
        inputs[name] = interstitial[name]
    if control.use_med_flux:
        inputs["sfculw_med"] = carry.coupling["ulwsfcin_med"]
    if control.lssav and control.ldiag3d and not control.lsidea:
        inputs["dtend"] = carry.intdiag["dtend"]
    out = glue(scale_radiation)(inputs, control)
    diag = dict(carry.intdiag)
    for source, target in (
        ("adjsfcdsw", "dswsfci"),
        ("adjsfcdswc", "dswsfcci"),
        ("adjsfcnsw", "nswsfci"),
        ("adjsfcdlw", "dlwsfci"),
    ):
        diag[target] = out[source]
    if "dtend" in out:
        diag["dtend"] = out["dtend"]
    scratch = dict(interstitial)
    for name in (
        "dudt",
        "dvdt",
        "dtdt",
        "dqdt",
        "ten_u",
        "ten_v",
        "ten_t",
        "ten_q",
        "xmu",
        "xcosz",
        "adjnirbmu",
        "adjnirdfu",
        "adjvisbmu",
        "adjvisdfu",
        "adjnirbmd",
        "adjnirdfd",
        "adjvisbmd",
        "adjvisdfd",
        "dtdtnp",
    ):
        if name in out:
            scratch[name] = out[name]
    for source, target in (("lnd", "land"), ("ice", "ice"), ("wat", "water")):
        scratch[f"adjsfculw_{target}"] = out[f"adjsfculw_{source}"]
    return RadiationScalingResult(
        replace(stateout, **{n: out[n] for n in ("gt0", "gu0", "gv0", "gq0")}),
        replace(carry, intdiag=diag),
        scratch,
    )
