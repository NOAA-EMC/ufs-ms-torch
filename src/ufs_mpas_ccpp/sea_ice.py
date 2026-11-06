# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""R4 sfc_sice fluxes and Winton ice3lay, in source arithmetic order."""

from __future__ import annotations

import torch
from gfs_surface_layer_torch.driver import CP, EPS, EPSM1, RD, RVRDM1, fpvs

from .physics_prefix import PhysicsPrefixResult, merge_prefix_writes
from .precision import WORKING_DTYPES
from .runtime_policy import scheme

T0C = 273.15
TGICE = 271.2
HVAP = 2.5e6
SBC = 5.670400e-8
DS, DW, KI, KS, DI, CI, LI = 330.0, 1000.0, 2.03, 0.31, 917.0, 2054.0, 3.34e5
TFI, TFW = -0.054, -1.8
DICI, DILI, DSLI, KI4 = DI * CI, DI * LI, DS * LI, KI * 4.0

SURFACE_BINDINGS = {
    "hice": "hice",
    "fice": "fice",
    "tice": "tisfc",
    "weasd": "weasdi",
    "tiice": "tiice",
    "snwdph": "snodi",
}
WORK_BINDINGS = {
    "tsfc_wat": "tsfc_water",
    "tprcp": "tprcp_ice",
    "ep": "ep1d_ice",
    "qss_i": "qss_ice",
    "qss_w": "qss_water",
    "snowmt": "snowmt",
    "gflux": "gflx_ice",
    "cmm": "cmm_ice",
    "chh": "chh_ice",
    "evapi": "evap_ice",
    "evapw": "evap_water",
    "hflxi": "hflx_ice",
    "hflxw": "hflx_water",
}
OUTPUT_FIELDS = tuple(SURFACE_BINDINGS) + tuple(WORK_BINDINGS)


def sea_ice_fluxes(fields, table, *, thsfc_loc: bool):
    """Pre-flux stage, including precipitation and source ice/snow limits."""
    if type(thsfc_loc) is not bool:
        raise TypeError("sea-ice potential-temperature control must be static")
    active = (
        (fields["islmsk"] == 2) & fields["flag_iter"] & (fields["use_lake_model"] != 1)
    )
    # A1: source-inactive columns must not evaluate singular flux operands.
    defaults = {
        "ps": 100000.0,
        "t1": 260.0,
        "q1": 0.001,
        "sfcemis": 0.98,
        "dlwflx": 200.0,
        "sfcnsw": 0.0,
        "sfcdsw": 0.0,
        "srflag": 0.0,
        "cm": 0.001,
        "ch": 0.001,
        "prsl1": 98000.0,
        "prslki": 1.0,
        "prsik1": 1.0,
        "prslk1": 1.0,
        "wind": 5.0,
        "hice": 1.0,
        "fice": 1.0,
        "tice": 260.0,
        "weasd": 0.0,
        "tprcp": 0.0,
        "ep": 0.0,
    }
    f = {n: torch.where(active, fields[n], d) for n, d in defaults.items()}
    f["tiice"] = torch.where(active[:, None], fields["tiice"], 260.0)
    torch._assert_async(
        (~active | ((f["fice"] > 0.0) & (f["ch"] * f["wind"] > 0.0))).all(),
        "active sea ice needs nonzero concentration and heat exchange",
    )
    precip_snow = f["srflag"] > 0.0
    f["ep"] = torch.where(precip_snow, f["ep"] * (1.0 - f["srflag"]), f["ep"])
    f["weasd"] = torch.where(
        precip_snow, f["weasd"] + 1000.0 * f["tprcp"] * f["srflag"], f["weasd"]
    )
    f["tprcp"] = torch.where(precip_snow, f["tprcp"] * (1.0 - f["srflag"]), f["tprcp"])
    q0 = torch.clamp_min(f["q1"], 1.0e-8)
    theta1 = f["t1"] * f["prslki"] if thsfc_loc else f["t1"] / f["prslk1"]
    rho = f["prsl1"] / (RD * f["t1"] * (1.0 + RVRDM1 * q0))
    vapor = fpvs(f["t1"], table)
    qs1 = torch.clamp_min(EPS * vapor / (f["prsl1"] + EPSM1 * vapor), 1.0e-8)
    q0 = torch.minimum(qs1, q0)
    ffw = 1.0 - f["fice"]
    vi = fpvs(f["tice"], table)
    vw = fpvs(torch.full_like(f["tice"], TGICE), table)
    qssi = EPS * vi / (f["ps"] + EPSM1 * vi)
    qssw = EPS * vw / (f["ps"] + EPSM1 * vw)
    cmm = f["cm"] * f["wind"]
    chh = rho * f["ch"] * f["wind"]
    rch = chh * CP
    evapi = (HVAP / CP) * rch * (qssi - q0)
    evapw = (HVAP / CP) * rch * (qssw - q0)
    snetw = torch.minimum(
        3.0 * f["sfcnsw"] / (1.0 + 2.0 * ffw), f["sfcdsw"] * (1.0 - 0.06)
    )
    sneti = (f["sfcnsw"] - ffw * snetw) / f["fice"]
    t12 = f["tice"] * f["tice"]
    t14 = t12 * t12
    surface_t = f["tice"] if thsfc_loc else f["tice"] / f["prsik1"]
    hfi = -f["dlwflx"] + f["sfcemis"] * SBC * t14 + evapi + rch * (surface_t - theta1)
    hfd = (
        4.0 * f["sfcemis"] * SBC * f["tice"] * t12
        + (1.0 + (HVAP / CP) * EPS * HVAP * qs1 / (RD * t12)) * rch
    )
    hice = torch.clamp(f["hice"], min=0.1, max=8.0)
    snowd = torch.clamp_max(f["weasd"] * 0.001, 2.0)
    snowd = torch.where(snowd > 2.0 * hice, hice + hice, snowd)
    return {
        **f,
        "flag": active,
        "theta1": theta1,
        "rho": rho,
        "rch": rch,
        "cmm": cmm,
        "chh": chh,
        "evapi": evapi,
        "evapw": evapw,
        "sneti": sneti,
        "hfi": hfi,
        "hfd": hfd,
        "snowd": snowd,
        "hice": hice,
        "stsice": f["tiice"],
        "focn": torch.full_like(hice, 2.0),
        "snof": torch.zeros_like(hice),
    }


def ice_temperature(f, delt: torch.Tensor):
    """ice3lay flooding, temperature solve, and top/bottom melt energies."""
    dt2 = delt + delt
    dt4 = dt2 + dt2
    dt6 = dt2 + dt4
    dt2i = 1.0 / dt2
    snowd = f["snowd"] * (DW / DS)
    hice = f["hice"]
    hdi = (DS / DW) * snowd + (DI / DW) * hice
    flooded = hice < hdi
    snowd = torch.where(flooded, snowd + hice - hdi, snowd)
    hice = torch.where(flooded, hice + (hdi - hice) * (DS / DI), hice)
    snof = f["snof"] * (DW / DS)
    t1 = torch.clamp_max(f["stsice"][:, 0] - T0C, TFI - 0.0001)
    t2 = torch.clamp_max(f["stsice"][:, 1] - T0C, TFI - 0.0001)
    snowy = snowd > 0.0
    tsf = torch.where(snowy, torch.zeros_like(hice), torch.full_like(hice, TFI))
    ip = torch.where(snowy, 0.0, 0.3 * f["sneti"])
    tice = torch.minimum(f["tice"] - T0C, tsf)
    bi = f["hfd"]
    ai = f["hfi"] - f["sneti"] + ip - tice * bi
    k12 = torch.div(KI4 * KS, KS * hice + KI4 * snowd)
    k32 = torch.div(KI + KI, hice)
    wrk = 1.0 / (dt6 * k32 + DICI * hice)
    a10 = DICI * hice * dt2i + k32 * (dt4 * k32 + DICI * hice) * wrk
    b10 = (
        -DI * hice * (CI * t1 + torch.div(LI * TFI, t1)) * dt2i
        - ip
        - k32 * (dt4 * k32 * TFW + DICI * hice * t2) * wrk
    )
    wrk1 = k12 / (k12 + bi)
    a1 = a10 + bi * wrk1
    b1 = b10 + ai * wrk1
    c1 = DILI * TFI * dt2i * hice
    t1 = -(torch.sqrt(b1 * b1 - 4.0 * a1 * c1) + b1) / (a1 + a1)
    tice = (k12 * t1 - ai) / (k12 + bi)
    melting = tice > tsf
    melt_a1 = a10 + k12
    melt_b1 = b10 - k12 * tsf
    melt_t1 = -(torch.sqrt(melt_b1 * melt_b1 - 4.0 * melt_a1 * c1) + melt_b1) / (
        melt_a1 + melt_a1
    )
    t1 = torch.where(melting, melt_t1, t1)
    tice = torch.where(melting, tsf, tice)
    tmelt = torch.where(melting, (k12 * (t1 - tsf) - (ai + bi * tsf)) * delt, 0.0)
    snowd = torch.where(melting, snowd, snowd + snof * delt)
    t2 = (dt2 * k32 * (t1 + TFW + TFW) + DICI * hice * t2) * wrk
    bmelt = (f["focn"] + KI4 * (t2 - TFW) / hice) * delt
    return {
        **f,
        "hice": hice,
        "snowd": snowd,
        "snof": snof,
        "tice": tice,
        "t1": t1,
        "t2": t2,
        "tmelt": tmelt,
        "bmelt": bmelt,
    }


def ice_mass_changes(f, delt: torch.Tensor):
    """ice3lay snow/top/bottom melt, layer redistribution and heat copyback."""
    h1, h2 = 0.5 * f["hice"], 0.5 * f["hice"]
    snowd, t1, t2 = f["snowd"], f["t1"], f["t2"]
    snow_only = f["tmelt"] <= snowd * DSLI
    snowmt = torch.where(snow_only, f["tmelt"] / DSLI, snowd)
    # The rejected snow-only branch must not divide at the melting point (A1).
    melt_t1 = torch.where(~snow_only, t1, -1.0)
    h1 = torch.where(
        snow_only,
        h1,
        torch.clamp_min(
            h1
            - (f["tmelt"] - snowd * DSLI)
            / (DI * (CI - torch.div(LI, melt_t1)) * (TFI - melt_t1)),
            0.0,
        ),
    )
    snowd = torch.where(snow_only, snowd - snowmt, 0.0)
    freezing = f["bmelt"] < 0.0
    dh = torch.where(freezing, -f["bmelt"] / (DILI + DICI * (TFI - TFW)), 0.0)
    t2 = torch.where(freezing, (h2 * t2 + dh * TFW) / (h2 + dh), t2)
    h2 = torch.where(freezing, h2 + dh, h2 - f["bmelt"] / (DILI + DICI * (TFI - t2)))
    h2 = torch.clamp_min(h2, 0.0)
    hice = h1 + h2
    remaining = hice > 0.0
    upper_larger = remaining & (h1 > 0.5 * hice)
    lower_larger = remaining & ~upper_larger
    safe_hice = torch.where(remaining, hice, 1.0)
    f1 = 1.0 - (h2 + h2) / safe_hice
    redistributed_t2 = f1 * (t1 + torch.div(LI * TFI, CI * t1)) + (1.0 - f1) * t2
    t2 = torch.where(upper_larger, redistributed_t2, t2)
    overheated = upper_larger & (t2 > TFI)
    hice = torch.where(overheated, hice - h2 * CI * (t2 - TFI) / (LI * delt), hice)
    t2 = torch.where(overheated, TFI, t2)
    f1 = (h1 + h1) / safe_hice
    mixed_t1 = f1 * (t1 + torch.div(LI * TFI, CI * t1)) + (1.0 - f1) * t2
    redistributed_t1 = (
        mixed_t1 - torch.sqrt(mixed_t1 * mixed_t1 - 4.0 * TFI * LI / CI)
    ) * 0.5
    t1 = torch.where(lower_larger, redistributed_t1, t1)
    denominator = torch.where(remaining, KS * hice + KI4 * snowd, 1.0)
    gflux = torch.where(remaining, torch.div(KI4 * KS, denominator) * (t1 - f["tice"]), 0.0)
    vanished_snow = (
        snowd
        + (h1 * (CI * (t1 - TFI) - LI * (1.0 - TFI / t1)) + h2 * (CI * (t2 - TFI) - LI))
        / LI
    )
    hice = torch.where(remaining, hice, torch.clamp_min(vanished_snow * (DS / DI), 0.0))
    snowd = torch.where(remaining, snowd, 0.0)
    t1 = torch.where(remaining, t1, TFW)
    t2 = torch.where(remaining, t2, TFW)
    return {
        **f,
        "hice": hice,
        "snowd": snowd * (DS / DW),
        "tice": f["tice"] + T0C,
        "stsice": torch.stack((t1 + T0C, t2 + T0C), dim=1),
        "snowmt": snowmt * (DS / DW),
        "gflux": f["fice"] * gflux,
    }


def ice3lay(f, delt: torch.Tensor):
    return ice_mass_changes(ice_temperature(f, delt), delt)


def sea_ice_copyback(fields, flux, ice, *, thsfc_loc: bool):
    """Post-thermodynamics limits, flux units, and exact inactive retention."""
    tice = torch.clamp_min(ice["tice"], 173.0)
    tiice = torch.clamp(ice["stsice"], min=173.0, max=T0C)
    if thsfc_loc:
        hflxi = flux["rch"] * (tice - flux["theta1"])
        hflxw = flux["rch"] * (TGICE - flux["theta1"])
    else:
        tem = 1.0 / flux["prsik1"]
        hflxi = flux["rch"] * (tice * tem - flux["theta1"])
        hflxw = flux["rch"] * (TGICE * tem - flux["theta1"])
    qss_i = flux["q1"] + flux["evapi"] / ((HVAP / CP) * flux["rch"])
    qss_w = flux["q1"] + flux["evapw"] / ((HVAP / CP) * flux["rch"])
    weasd = ice["snowd"] * 1000.0
    tem = 1.0 / flux["rho"]
    values = {
        "hice": ice["hice"],
        "fice": flux["fice"],
        "tice": tice,
        "weasd": weasd,
        "tiice": tiice,
        "snwdph": weasd * (1.0 / 0.33),
        "tsfc_wat": torch.full_like(tice, TGICE),
        "tprcp": flux["tprcp"],
        "ep": flux["ep"],
        "qss_i": qss_i,
        "qss_w": qss_w,
        "snowmt": ice["snowmt"],
        "gflux": ice["gflux"],
        "cmm": flux["cmm"],
        "chh": flux["chh"],
        "evapi": flux["evapi"] * tem * (1.0 / HVAP),
        "evapw": flux["evapw"] * tem * (1.0 / HVAP),
        "hflxi": hflxi * tem * (1.0 / CP),
        "hflxw": hflxw * tem * (1.0 / CP),
    }
    return {
        n: torch.where(
            flux["flag"][:, None] if v.ndim == 2 else flux["flag"], v, fields[n]
        )
        for n, v in values.items()
    }


def sfc_sice_run(fields, table, delt: torch.Tensor, *, thsfc_loc: bool):
    """Complete selected two-ice-layer sfc_sice_run, without host activity reads."""
    if fields["tiice"].shape != (fields["t1"].numel(), 2):
        raise ValueError("the selected sea-ice model requires two ice layers")
    if (
        delt.shape != ()
        or delt.device != fields["t1"].device
        or delt.dtype not in WORKING_DTYPES
    ):
        raise ValueError(
            "sea-ice timestep must be a scalar float32/float64 tensor on the numerical device"
        )
    torch._assert_async(
        (torch.isfinite(delt) & (delt > 0.0)), "invalid sea-ice timestep"
    )
    flux = sea_ice_fluxes(fields, table, thsfc_loc=thsfc_loc)
    return sea_ice_copyback(fields, flux, ice3lay(flux, delt), thsfc_loc=thsfc_loc)


def sea_ice_inputs(statein, prefix: PhysicsPrefixResult):
    f = {n: getattr(prefix.sfcprop, m) for n, m in SURFACE_BINDINGS.items()}
    f.update({n: prefix.interstitial[m] for n, m in WORK_BINDINGS.items()})
    f.update(
        ps=statein.pgr,
        t1=statein.tgrs[:, 0],
        q1=statein.qgrs[:, 0, 0],
        sfcemis=prefix.sfcprop.emis_ice,
        dlwflx=prefix.interstitial["gabsbdlw_ice"],
        sfcnsw=prefix.carry.intdiag["nswsfci"],
        sfcdsw=prefix.carry.intdiag["dswsfci"],
        srflag=prefix.sfcprop.srflag,
        cm=prefix.interstitial["cd_ice"],
        ch=prefix.interstitial["cdq_ice"],
        prsl1=statein.prsl[:, 0],
        prslki=prefix.interstitial["work3"],
        prsik1=statein.prsik[:, 0],
        prslk1=statein.prslk[:, 0],
        wind=prefix.interstitial["wind"],
        flag_iter=prefix.interstitial["flag_iter"],
        use_lake_model=prefix.sfcprop.use_lake_model,
        islmsk=prefix.interstitial["islmsk_cice"],
    )
    return f


def apply_sea_ice(
    statein, prefix: PhysicsPrefixResult, table, delt: torch.Tensor, *, thsfc_loc: bool
):
    """sfc_sice on the live prefix; the compiled region (``scheme``) is ``sea_ice_writes``."""
    return merge_prefix_writes(prefix, scheme(sea_ice_writes)(statein, prefix, table, delt, thsfc_loc=thsfc_loc))


def sea_ice_writes(
    statein, prefix: PhysicsPrefixResult, table, delt: torch.Tensor, *, thsfc_loc: bool
) -> tuple[dict, dict]:
    """The sfcprop and interstitial fields sfc_sice writes (``apply_sea_ice``)."""
    out = sfc_sice_run(
        sea_ice_inputs(statein, prefix), table, delt, thsfc_loc=thsfc_loc
    )
    return ({m: out[n] for n, m in SURFACE_BINDINGS.items()},
            {m: out[n] for n, m in WORK_BINDINGS.items()})
