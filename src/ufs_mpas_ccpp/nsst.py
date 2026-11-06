# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Live R4 NSST surface update, with source-ordered masks and guess restoration."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from gfs_surface_layer_torch.driver import CP, EPS, EPSM1, RD, RVRDM1, fpvs
from ufs_physics_common.precision import working_power

from . import nsst_thermal as thermal
from .nsst_dtm import WARM_FIELDS, dtl_reset, dtm_1p
from .physics_prefix import merge_prefix_writes
from .runtime_policy import scheme

NSST_FIELDS = WARM_FIELDS + (
    "zm",
    "dt_cool",
    "z_c",
    "c_0",
    "c_d",
    "w_0",
    "w_d",
    "d_conv",
    "ifd",
    "qrain",
)
FLUX_FIELDS = ("qsurf", "gflux", "cmm", "chh", "evap", "hflx", "ep")
_SAVED = WARM_FIELDS + ("zm", "ifd", "tskin", "dt_cool", "z_c")
HVAP = 2.5e6
SBC = 5.670400e-8
TGICE = 271.2


@dataclass(frozen=True)
class NSSTControl:
    nstf_name: tuple[int, ...]
    timestep: float
    use_oceanuv: bool
    thsfc_loc: bool
    lseaspray: bool
    cplflx: bool = False

    def __post_init__(self):
        if type(self.nstf_name) is not tuple or any(
            type(n) is not int for n in self.nstf_name
        ):
            raise TypeError("NSST control vector must contain static integers")
        if self.nstf_name != (2, 1, 0, 0, 0):
            raise NotImplementedError("NSST requires the selected R4 control vector")
        if self.cplflx:
            raise NotImplementedError("mediator-coupled NSST pre-call is not ported")
        if (
            type(self.timestep) not in (int, float)
            or not math.isfinite(self.timestep)
            or self.timestep <= 0.0
        ):
            raise ValueError("invalid NSST timestep")
        for name in ("use_oceanuv", "thsfc_loc", "lseaspray", "cplflx"):
            if type(getattr(self, name)) is not bool:
                raise TypeError("NSST switches must be static booleans")


def sfc_nst_run(fields, solhr, table, control: NSSTControl):
    """Update all 26 source writes, retaining untouched input buffers."""
    active = fields["wet"] & fields["flag_iter"] & (fields["use_lake_model"] != 1)
    eligible = fields["wet"] & (fields["use_lake_model"] != 1)
    # Neutralize only source-inactive rows before singular arithmetic (A1).
    f = {
        name: torch.where(active, fields[name], value)
        for name, value in {
            "u1": 1.0,
            "v1": 0.0,
            "usfco": 0.0,
            "vsfco": 0.0,
            "ps": 100000.0,
            "prsl1": 100000.0,
            "t1": 290.0,
            "q1": 0.01,
            "prslki": 1.0,
            "prslk1": 1.0,
            "prsik1": 1.0,
            "cm": 0.001,
            "ch": 0.001,
            "wind": 1.0,
            "stress": 0.005,
            "fm": 1.0,
            "fm10": 1.0,
            "tsurf": 290.0,
            "tref": 290.0,
            "sfcemis": 1.0,
            "dlwflx": 400.0,
            "sfcnsw": 0.0,
            "rain": 0.0,
            "xlon": 0.0,
            "sinlat": 0.0,
        }.items()
    }
    result = {n: fields[n] for n in NSST_FIELDS + FLUX_FIELDS + ("tskin", "tsurf")}
    q0 = torch.clamp_min(f["q1"], 1.0e-8)
    theta1 = f["t1"] * f["prslki"] if control.thsfc_loc else f["t1"] / f["prslk1"]
    rho_a = f["prsl1"] / (RD * (f["t1"] * (1.0 + RVRDM1 * q0)))

    def saturation(t, pressure):
        vapor = fpvs(t, table)
        return EPS * vapor / (pressure + EPSM1 * vapor)

    def sensible(t):
        temperature = t if control.thsfc_loc else t / f["prsik1"]
        return rch * (temperature - theta1)

    qss = saturation(f["tsurf"], f["ps"])
    wind = (
        thermal.positive_sqrt((f["u1"] - f["usfco"]) ** 2 + (f["v1"] - f["vsfco"]) ** 2)
        if control.use_oceanuv
        else f["wind"]
    )
    rch = rho_a * CP * f["ch"] * wind
    evap = (HVAP / CP) * rch * (qss - q0)
    hflx = sensible(f["tsurf"])
    tsea = f["tsurf"]
    t12 = tsea * tsea
    ulw = f["sfcemis"] * SBC * t12 * t12
    alon = f["xlon"] * (180.0 / math.pi)
    gravity = thermal.grv(f["sinlat"])
    soltim = torch.fmod(alon / 15.0 + solhr, 24.0) * 3600.0
    salinity = torch.full_like(tsea, 34.0)
    rho_w = thermal.density(tsea, salinity)
    alpha, beta = thermal.rhocoef(tsea, salinity, rho_w)
    le = (2.501 - 0.00237 * tsea) * 1.0e6
    dwat = 2.11e-5 * (f["t1"] / 273.16) ** 1.94
    tc = f["t1"] - 273.16
    dtmp = (1.0 + 3.309e-3 * tc - 1.44e-6 * tc * tc) * 0.02411 / (rho_a * CP)
    wetc = 622.0 * le * qss / (RD * f["t1"] * f["t1"])
    alfac = 1.0 / (1.0 + wetc * le * dwat / (CP * dtmp))
    rainfactor = (1.0e3 * f["rain"] / rho_w) * alfac * thermal.CP_W
    qrain = rainfactor * (tsea - f["t1"] + 1.0e3 * (qss - q0) * le / CP)
    nonsolar = hflx + evap + ulw - f["dlwflx"] + qrain
    sep = salinity * (evap / le - f["rain"]) / rho_w
    ustar = thermal.positive_sqrt(f["stress"] / rho_a)
    hl_ts = rch * (HVAP / CP) * EPS * HVAP * qss / (RD * t12)
    q_ts = (
        4.0 * f["sfcemis"] * SBC * tsea * tsea * tsea
        + rch
        + hl_ts
        + rainfactor * (1.0 + rch * hl_ts)
    )
    cool = thermal.cool_skin(
        ustar,
        nonsolar,
        f["sfcnsw"],
        evap,
        salinity,
        alpha,
        beta,
        rho_w,
        rho_a,
        tsea,
        q_ts,
        hl_ts,
        gravity,
        le,
    )
    for name, value in zip(("dt_cool", "z_c", "c_0", "c_d"), cool):
        result[name] = torch.where(active, value, result[name])
    result["qrain"] = torch.where(active, qrain, result["qrain"])
    norm = thermal.positive_sqrt(f["u1"] * f["u1"] + f["v1"] * f["v1"])
    torch._assert_async(
        ((~active) | (norm > 0.0)).all(), "NSST wind direction is undefined"
    )
    inverse = 1.0 / torch.where(active, norm, 1.0)
    stress = torch.clamp_min(f["stress"], 0.005)
    evolve = active & ~((soltim > 21600.0) & (fields["ifd"] == 0.0))
    result["ifd"] = torch.where(evolve, 1.0, result["ifd"])
    warm = {
        n: torch.where(evolve, fields[n], 20.0 if n == "xz" else 0.0)
        for n in WARM_FIELDS
    }
    conv = (nonsolar > 0.0) & (warm["xt"] > 0.0) & evolve
    depth = thermal.convdepth(
        control.timestep,
        f["sfcnsw"],
        nonsolar,
        salinity,
        sep,
        rho_w,
        alpha,
        beta,
        warm["xt"],
        warm["xs"],
        warm["xz"],
    )
    d_conv = torch.where(conv, depth, 0.0)
    result["d_conv"] = torch.where(evolve, d_conv, result["d_conv"])
    params = dict(
        timestep=control.timestep,
        rich=0.65,
        tox=stress * (f["u1"] * inverse),
        toy=stress * (f["v1"] * inverse),
        i0=torch.where(evolve, f["sfcnsw"], 0.0),
        q=torch.where(evolve, nonsolar, 0.0),
        sss=salinity,
        sep=torch.where(evolve, sep, 0.0),
        q_ts=q_ts,
        hl_ts=hl_ts,
        rho=rho_w,
        alpha=alpha,
        beta=beta,
        sinlat=f["sinlat"],
        grav=gravity,
        le=le,
    )
    warm = dtm_1p(params, warm)
    adjust = evolve & (warm["xt"] > 0.0)

    def set_depth(new, mask):
        nonlocal warm
        warm["xz"] = torch.where(mask, new, warm["xz"])
        warm = dtl_reset(warm, mask & (warm["xz"] >= 20.0))

    set_depth(thermal.dtm_1p_mda(warm["xz"]), adjust)
    mask = adjust & (d_conv > 0.0) & (warm["xt"] > 0.0)
    safe_xt = torch.where(mask, warm["xt"], 1.0)
    safe_xz = torch.where(mask, warm["xz"], 1.0)
    set_depth(
        thermal.dtm_1p_fca(torch.where(mask, d_conv, 0.0), safe_xt, safe_xz), mask
    )
    safe_xz = torch.where(adjust, warm["xz"], 1.0)
    dz = torch.minimum(safe_xz, torch.clamp_min(d_conv, 0.1))
    q_warm = thermal.sw_ps_9b(torch.full_like(tsea, 0.1)) * f["sfcnsw"] - nonsolar
    ttop0 = thermal.cal_ttop(control.timestep, q_warm, rho_w, dz, warm["xt"], safe_xz)
    ttop = ((warm["xt"] + warm["xt"]) / safe_xz) * (1.0 - dz / (safe_xz + safe_xz))
    mask = adjust & (q_warm > 0.0) & (ttop > ttop0)
    set_depth(thermal.dtm_1p_tla(dz, ttop0, torch.where(mask, warm["xt"], 0.0)), mask)
    safe_xz = torch.where(adjust, warm["xz"], 1.0)
    t0 = (warm["xt"] + warm["xt"]) / safe_xz
    set_depth(thermal.dtm_1p_mwa(warm["xt"]), adjust & (t0 > 5.0))
    safe_xz = torch.where(adjust, warm["xz"], 1.0)
    sstc = f["tref"] + (warm["xt"] + warm["xt"]) / safe_xz - result["dt_cool"]
    mask = adjust & (sstc > 308.16)
    set_depth(thermal.dtm_1p_mta(sstc - 308.16, warm["xt"], safe_xz), mask)
    warm = dtl_reset(warm, evolve & (torch.abs(soltim) < 2.0 * control.timestep))
    for name in WARM_FIELDS:
        result[name] = torch.where(evolve, warm[name], result[name])
    z1 = tsea.new_full((), 0.001 * control.nstf_name[3])
    z2 = tsea.new_full((), 0.001 * control.nstf_name[4])
    dtz = thermal.get_dtzm_point(
        torch.where(active, result["xt"], 0.0),
        torch.where(active, result["xz"], 1.0),
        torch.where(active, result["dt_cool"], 0.0),
        torch.where(active, result["z_c"], 1.0),
        z1,
        z2,
    )
    result["tsurf"] = torch.where(
        active, torch.clamp_min(f["tref"] + dtz, TGICE), result["tsurf"]
    )
    wmask = active & (result["xt"] > 0.0)
    w0, wd = thermal.cal_w(
        torch.where(wmask, result["xz"], 1.0),
        torch.where(wmask, result["xt"], 0.0),
        torch.where(wmask, result["xzts"], 0.0),
        torch.where(wmask, result["xtts"], 0.0),
    )
    result["w_0"] = torch.where(active, torch.where(wmask, w0, 0.0), result["w_0"])
    result["w_d"] = torch.where(active, torch.where(wmask, wd, 0.0), result["w_d"])
    result["zm"] = torch.where(active, f["wind"], result["zm"])
    guess = eligible & fields["flag_guess"]
    for name in _SAVED:
        result[name] = torch.where(guess, fields[name], result[name])
    result["tskin"] = torch.where(
        eligible & ~fields["flag_guess"] & active.any(),
        result["tsurf"],
        result["tskin"],
    )
    skin = torch.where(active, result["tskin"], 290.0)
    qss = saturation(skin, f["ps"])
    evap = (HVAP / CP) * rch * (qss - q0)
    hflx = sensible(skin)
    if control.lseaspray:
        f10 = f["fm10"] / f["fm"]
        u10, v10 = f10 * f["u1"], f10 * f["v1"]
        ws10 = torch.clamp(thermal.positive_sqrt(u10 * u10 + v10 * v10), 1.0, 30.0)
        tem = 0.015 * ws10 * ws10
        # A Python-scalar numerator would divide as 10 * (1/tem) (one rounding more than the
        # source 10./tem); divide two tensors instead.
        ru10 = 1.0 - 0.087 * torch.log(torch.full_like(tem, 10.0) / tem)
        qss1 = saturation(f["t1"], f["prsl1"])
        tem = RD * CP * f["t1"] * f["t1"]
        bb1 = 1.0 / (1.0 + EPS * HVAP * HVAP * qss1 / tem)
        evaps = 7.2e-9 * working_power(ws10, 5.4) * ru10 * bb1
        evaps = evaps * rho_a * HVAP * (qss1 - q0)
        evap = evap + 0.75 * evaps
        hflxs = 6.4e-8 * working_power(ws10, 3.4) * ru10
        hflxs = hflxs * rho_a * CP * (skin - f["t1"])
        hflx = hflx + 0.75 * hflxs - (0.75 - 0.15) * evaps
    flux = dict(
        qsurf=qss,
        gflux=torch.zeros_like(tsea),
        cmm=f["cm"] * wind,
        chh=rho_a * f["ch"] * wind,
        evap=evap * (1.0 / rho_a) * (1.0 / HVAP),
        hflx=hflx * (1.0 / rho_a) * (1.0 / CP),
        ep=torch.zeros_like(tsea),
    )
    result.update(
        {n: torch.where(active, value, result[n]) for n, value in flux.items()}
    )
    return result


def apply_nsst(
    statein, grid, prefix, solhr, table, control: NSSTControl, *, iteration: int
):
    """Bind loop-control1, NSST pre/run/post to live surface and work buffers.

    The first-iteration guess runs here; the compiled region (``scheme``) is ``nsst_writes``, one
    variant for both iterations, returning only the fields it writes.
    """
    if type(iteration) is not int or iteration not in (1, 2):
        raise ValueError("surface iteration must be 1 or 2")
    if iteration == 1:
        prefix = nsst_first_guess(prefix)
    return merge_prefix_writes(prefix, scheme(nsst_writes)(statein, grid, prefix, solhr, table, control))


def nsst_first_guess(prefix):
    """loop-control1 of the first surface iteration: ``flag_guess`` also where the wind is below 2 m/s."""
    work = prefix.interstitial
    return replace(prefix, interstitial={**work, "flag_guess": work["flag_guess"] | (work["wind"] < 2.0)})


def nsst_writes(statein, grid, prefix, solhr, table, control: NSSTControl) -> tuple[dict, dict]:
    """The sfcprop and interstitial fields NSST pre/run/post write (``apply_nsst`` after loop-control1)."""
    surface, work = prefix.sfcprop, prefix.interstitial  # read by key only: unread fields stay unguarded
    written = {}
    pre = work["wet"] & (surface.oceanfrac > 0.0)
    tseal = torch.where(pre, surface.tsfco, work["tseal"])
    f = {
        n: getattr(surface, n)
        for n in NSST_FIELDS + ("tref", "use_lake_model", "usfco", "vsfco")
    }
    f.update(
        ps=statein.pgr,
        u1=statein.ugrs[:, 0],
        v1=statein.vgrs[:, 0],
        t1=statein.tgrs[:, 0],
        q1=statein.qgrs[:, 0, 0],
        prsl1=statein.prsl[:, 0],
        prslki=work["work3"],
        prsik1=statein.prsik[:, 0],
        prslk1=statein.prslk[:, 0],
        xlon=grid.xlon,
        sinlat=grid.sinlat,
        sfcemis=surface.emis_wat,
        dlwflx=work["gabsbdlw_water"],
        sfcnsw=prefix.carry.intdiag["nswsfci"],
        rain=work["tprcp_water"],
        tskin=tseal,
        tsurf=work["tsurf_water"],
    )
    for name in ("wet", "flag_iter", "flag_guess", "wind"):
        f[name] = work[name]
    bindings = dict(
        cm="cd",
        ch="cdq",
        fm="ffmm",
        fm10="fm10",
        stress="stress",
        qsurf="qss",
        gflux="gflx",
        cmm="cmm",
        chh="chh",
        evap="evap",
        hflx="hflx",
        ep="ep1d",
    )
    for name, member in bindings.items():
        f[name] = work[member + "_water"]
    out = sfc_nst_run(f, solhr, table, control)
    nsst = {n: out[n] for n in NSST_FIELDS}
    written.update(tseal=out["tskin"], tsurf_water=out["tsurf"])
    for name in FLUX_FIELDS:
        written[bindings[name] + "_water"] = out[name]
    z1 = solhr.new_full((), 0.001 * control.nstf_name[3])
    z2 = solhr.new_full((), 0.001 * control.nstf_name[4])
    written["dtzm"] = thermal.get_dtzm_2d(
        nsst["xt"], nsst["xz"], nsst["dt_cool"], nsst["z_c"], work["wet"], z1, z2
    )
    post = work["wet"] & (surface.use_lake_model != 1)
    written["tsfc_water"] = torch.where(
        post, torch.clamp_min(surface.tref + written["dtzm"], TGICE), work["tsfc_water"]
    )
    return nsst, written
