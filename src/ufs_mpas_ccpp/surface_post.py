# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Selected R4 surface composites (nonfractional and frac_grid), diagnostics and generic post."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from gfs_surface_layer_torch.driver import EPS, EPSM1, RD, RVRDM1, sfc_diag_run, stability

from .land import (
    apply_noahmp,
    noahmp_case_with_plan,
    apply_sfc_land,
    finish_surface_iteration,
    skip_inactive_noahmp_pass,
)
from .nsst import apply_nsst
from .physics_prefix import PhysicsPrefixResult, apply_surface_layer
from .sea_ice import apply_sea_ice
from .runtime_policy import glue, scheme


@dataclass(frozen=True)
class SurfacePostControl:
    frac_grid: bool
    use_cdeps_data: bool
    cplflx: bool
    cplwav2atm: bool
    cpl_fire: bool
    cplaqm: bool
    cplchm: bool
    cplcat: bool
    cplwav: bool
    cpllnd: bool
    lkm: int
    lsm: int
    lsm_noahmp: int
    opt_diag: int
    lssav: bool
    lheatstrg: bool
    h0facu: float
    h0facs: float

    def __post_init__(self):
        for name in (
            "frac_grid",
            "use_cdeps_data",
            "cplflx",
            "cplwav2atm",
            "cpl_fire",
            "cplaqm",
            "cplchm",
            "cplcat",
            "cplwav",
            "cpllnd",
            "lssav",
            "lheatstrg",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError("surface post switches must be static booleans")
        if any(
            getattr(self, name)
            for name in (
                "use_cdeps_data",
                "cplflx",
                "cplwav2atm",
                "cpl_fire",
                "cplaqm",
                "cplchm",
                "cplcat",
                "cplwav",
                "cpllnd",
            )
        ):
            raise NotImplementedError(
                "only the selected R4 uncoupled surface post is ported (frac_grid either way)"
            )
        if any(
            type(getattr(self, n)) is not int
            for n in ("lkm", "lsm", "lsm_noahmp", "opt_diag")
        ):
            raise TypeError(
                "surface model and diagnostic identifiers must be static integers"
            )
        if self.lkm != 0 or self.lsm != 2 or self.lsm_noahmp != self.lsm:
            raise NotImplementedError(
                "surface post requires the selected no-lake Noah-MP policy"
            )


_TRANSFER_FIELDS = (
    "cd",
    "cdq",
    "rb",
    "stress",
    "ffmm",
    "ffhh",
    "uustar",
    "fm10",
    "fh2",
    "cmm",
    "chh",
    "gflx",
    "ep1d",
)
COMPOSITE_SURFACE_BINDINGS = {
    **{
        n: n
        for n in "zorl ffmm ffhh uustar weasd tprcp evap hflx qss tsfc tsfco tsfcl tisfc hice stc".split()
    },
    "cice": "fice",
    "snowd": "snowd",
    "zorlo": "zorlw",
    "zorll": "zorll",
    "zorli": "zorli",
    "weasd_lnd": "weasdl",
    "weasd_ice": "weasdi",
    "snowd_lnd": "snodl",
    "snowd_ice": "snodi",
    "tiice": "tiice",
}
COMPOSITE_WORK_BINDINGS = {
    **{
        n: n
        for n in "islmsk use_lake_model wet sigmaf zvfun hflxq hffac".split()
        if n != "use_lake_model"
    },
    **{
        n: n
        for n in _TRANSFER_FIELDS
        if n not in ("ffmm", "ffhh", "uustar", "cmm", "chh")
    },
    "tsfc_wat": "tsfc_water",
    **{
        n + "_" + short: n + "_" + long
        for short, long in (("wat", "water"), ("lnd", "land"), ("ice", "ice"))
        for n in _TRANSFER_FIELDS + ("evap", "hflx", "qss")
    },
}
COMPOSITE_WRITES = _TRANSFER_FIELDS + (
    "zorl",
    "weasd",
    "snowd",
    "tprcp",
    "evap",
    "hflx",
    "qss",
    "tsfc",
    "tsfco",
    "tsfcl",
    "tisfc",
    "hice",
    "cice",
    "stc",
    "sigmaf",
    "zvfun",
    "hflxq",
    "hffac",
)


def surface_composites_post(f, *, huge: float):
    """Source composite_land/wet/icy/wet_and_icy for the selected policy."""
    torch._assert_async(
        (f["use_lake_model"] == 0).all(), "unexpected active lake model"
    )
    land, water = f["islmsk"] == 1, f["islmsk"] == 0
    ice = ~(land | water)
    out = {n: f[n] for n in COMPOSITE_WRITES}
    for n in _TRANSFER_FIELDS:
        out[n] = torch.where(
            land, f[n + "_lnd"], torch.where(water, f[n + "_wat"], f[n + "_ice"])
        )
    for name, lnd, wat, icy in (
        ("zorl", "zorll", "zorlo", "zorli"),
        ("weasd", "weasd_lnd", None, "weasd_ice"),
        ("snowd", "snowd_lnd", None, "snowd_ice"),
        ("evap", "evap_lnd", "evap_wat", "evap_ice"),
        ("hflx", "hflx_lnd", "hflx_wat", "hflx_ice"),
        ("qss", "qss_lnd", "qss_wat", "qss_ice"),
    ):
        ice_value = f[icy] * f["cice"] if name in ("weasd", "snowd") else f[icy]
        out[name] = torch.where(
            land, f[lnd], torch.where(water, 0.0 if wat is None else f[wat], ice_value)
        )
    out["tsfc"] = torch.where(land, f["tsfcl"], f["tsfc_wat"])
    out["tsfco"] = out["tsfc"]
    out["tsfcl"] = torch.where(land, f["tsfcl"], out["tsfc"])
    out["tisfc"] = torch.where(ice, f["tisfc"], out["tsfc"])
    out["hice"] = torch.where(ice, f["hice"], 0.0)
    out["cice"] = torch.where(ice, f["cice"], 0.0)
    # A1: huge inactive roughness is an omitted logarithm, not log(huge).
    ln_ice = torch.log(torch.where(ice & (f["zorli"] != huge), f["zorli"], 1.0))
    ln_water = torch.log(torch.where(ice & (f["zorlo"] != huge), f["zorlo"], 1.0))
    txi, txo = f["cice"], 1.0 - f["cice"]

    def water(value, ice_value):
        # A1: a huge placeholder (no open water) enters as the ice value; txo * huge would give a
        # ~huge derivative with respect to cice. The forward value is unchanged where txo == 0.
        return torch.where(value == huge, ice_value, value)

    for name in ("evap", "hflx", "stress", "qss", "ep1d"):
        out[name] = torch.where(
            ice, txi * f[name + "_ice"] + txo * water(f[name + "_wat"], f[name + "_ice"]), out[name]
        )
    out["tsfc"] = torch.where(ice, txi * f["tisfc"] + txo * water(f["tsfc_wat"], f["tisfc"]), out["tsfc"])
    out["zorl"] = torch.where(
        ice, torch.exp(txi * ln_ice + txo * ln_water), out["zorl"]
    )
    out["tsfco"] = torch.where(
        ice, torch.where(f["wet"], f["tsfc_wat"], out["tsfc"]), out["tsfco"]
    )
    out["tsfcl"] = torch.where(ice, out["tsfc"], out["tsfcl"])
    out["stc"] = torch.cat(
        (torch.where(ice[:, None], f["tiice"], f["stc"][:, :2]), f["stc"][:, 2:]), dim=1
    )
    return out


def surface_composites_post_fractional(f, *, huge: float, thsfc_loc: bool, lheatstrg: bool,
                                       h0facu: float, h0facs: float):
    """Source frac_grid branch: three-way composites by land, ice and open-water area.

    Uncoupled and without CDEPS (the control rejects those). Cells of one surface type take that
    type's transfer coefficients; cells with several types (fractional sea ice) call sfc_diff's
    stability with the composite roughness, temperatures and the recomputed zvfun.
    """
    torch._assert_async((f["use_lake_model"] == 0).all(), "unexpected active lake model")
    out = {n: f[n] for n in COMPOSITE_WRITES}
    txl = f["landfrac"]
    wfrac = 1.0 - txl
    txi = f["cice"] * wfrac
    txo = torch.clamp_min(wfrac - txi, 0.0)
    weights = {"lnd": txl, "ice": txi, "wat": txo}

    def term(kind, *values):
        # A1: a surface with zero area contributes zero; its placeholder (huge) values do not
        # enter, so no huge derivative reaches cice. Forward: 0 * finite = 0, as in the source.
        weight = weights[kind]
        result = weight
        for value in values:
            result = result * torch.where(weight != 0.0, value, 0.0)
        return result

    def composite(name, kinds=("lnd", "ice", "wat")):
        total = None
        for kind in kinds:
            part = term(kind, f[f"{name}_{kind}"])
            total = part if total is None else total + part
        return total

    for name in ("ep1d", "evap", "hflx", "qss", "gflx"):
        out[name] = composite(name)
    out["weasd"] = composite("weasd", ("lnd", "ice"))
    out["snowd"] = composite("snowd", ("lnd", "ice"))
    out["sigmaf"] = txl * f["sigmaf"]
    conductance = term("lnd", f["cdq_lnd"]) + term("ice", f["cdq_ice"]) + term("wat", f["cdq_wat"])
    out["tsfc"] = (
        term("lnd", f["cdq_lnd"], f["tsfcl"]) + term("ice", f["cdq_ice"], f["tisfc"])
        + term("wat", f["cdq_wat"], f["tsfc_wat"])
    ) / conductance
    tsurf = (
        term("lnd", f["cdq_lnd"], f["tsurf_lnd"]) + term("ice", f["cdq_ice"], f["tsurf_ice"])
        + term("wat", f["cdq_wat"], f["tsurf_wat"])
    ) / conductance
    virtfac = 1.0 + RVRDM1 * torch.clamp_min(f["q1"], 1.0e-8)
    tv1 = f["t1"] * virtfac
    if thsfc_loc:
        thv1 = f["t1"] * f["prslki"] * virtfac
        tvs = 0.5 * (out["tsfc"] + tsurf) * virtfac
    else:
        thv1 = f["t1"] / f["prslk1"] * virtfac
        tvs = 0.5 * (out["tsfc"] + tsurf) / f["prsik1"] * virtfac

    def log_roughness(value):
        # The source omits the logarithm of a huge (inactive) roughness: zero.
        active = value != huge
        return torch.where(active, torch.log(torch.where(active, value, 1.0)), 0.0)

    out["zorl"] = torch.exp(txl * log_roughness(f["zorll"]) + txi * log_roughness(f["zorli"])
                            + txo * log_roughness(f["zorlo"]))
    z0max = 0.01 * out["zorl"]
    ztmax = torch.exp(txl * torch.log(f["ztmax_lnd"]) + txi * torch.log(f["ztmax_ice"])
                      + txo * torch.log(f["ztmax_wat"]))
    land, water = txl == 1.0, (txl != 1.0) & (txo == 1.0)
    ice = (txl != 1.0) & (txo != 1.0) & (txi == 1.0)
    mixed = ~(land | water | ice)
    tem1 = torch.clamp((z0max - 0.1) / (1.0 - 0.1), 0.0, 1.0)
    roughness_vegetation = tem1 * torch.clamp_min(out["sigmaf"], 0.1)
    positive = roughness_vegetation > 0.0
    zvfun = torch.where(positive, torch.sqrt(torch.where(positive, roughness_vegetation, 1.0)), 0.0)
    hffac = torch.ones_like(zvfun)
    hflxq = out["hflx"]
    if lheatstrg:
        hffac = 1.0 + torch.where(out["hflx"] > 0.0, h0facu * zvfun, h0facs * zvfun)
        hflxq = out["hflx"] / hffac
    out["zvfun"] = torch.where(mixed, zvfun, f["zvfun"])
    out["hflxq"] = torch.where(mixed, hflxq, f["hflxq"])
    out["hffac"] = torch.where(mixed, hffac, f["hffac"])
    # Every stability input is finite and physical in every cell; only mixed cells keep the result.
    mix = stability(f["z1"], zvfun, torch.sqrt(f["garea"]), tv1, thv1, f["wind"], z0max, ztmax, tvs,
                    thsfc_loc=thsfc_loc)
    for name, source in (("rb", "rb"), ("ffmm", "fm"), ("ffhh", "fh"), ("fm10", "fm10"),
                         ("fh2", "fh2"), ("cd", "cm"), ("cdq", "ch"), ("stress", "stress"),
                         ("uustar", "ustar")):
        out[name] = torch.where(land, f[name + "_lnd"], torch.where(
            water, f[name + "_wat"], torch.where(ice, f[name + "_ice"], mix[source])))
    rho = f["prsl1"] / (RD * tv1)
    out["cmm"] = out["cd"] * f["wind"]
    out["chh"] = rho * out["cdq"] * f["wind"]
    dry, wet, icy = f["dry"], f["wet"], f["icy"]
    out["tsfcl"] = torch.where(dry, f["tsfcl"], torch.where(wet, f["tsfc_wat"], f["tisfc"]))
    out["tsfco"] = torch.where(wet, f["tsfc_wat"], torch.where(icy, f["tisfc"], out["tsfcl"]))
    out["tisfc"] = torch.where(icy, f["tisfc"], torch.where(wet, f["tsfc_wat"], out["tsfcl"]))
    out["hice"] = torch.where(icy, f["hice"], 0.0)
    out["cice"] = torch.where(icy, f["cice"], 0.0)
    return out


def apply_surface_composites_post(
    prefix: PhysicsPrefixResult, control: SurfacePostControl, *, huge: float,
    statein=None, grid=None, thsfc_loc: bool | None = None,
):
    """``statein``, ``grid`` and ``thsfc_loc`` are read by the frac_grid branch only."""
    f = {n: getattr(prefix.sfcprop, m) for n, m in COMPOSITE_SURFACE_BINDINGS.items()}
    f.update({n: prefix.interstitial[m] for n, m in COMPOSITE_WORK_BINDINGS.items()})
    f.update(
        use_lake_model=prefix.sfcprop.use_lake_model,
        cmm=prefix.carry.intdiag["cmm"],
        chh=prefix.carry.intdiag["chh"],
    )
    if control.frac_grid:
        if statein is None or grid is None or type(thsfc_loc) is not bool:
            raise ValueError("the frac_grid composites need Statein, Grid and a static thsfc_loc")
        work = prefix.interstitial
        f.update(
            landfrac=prefix.sfcprop.landfrac, wind=work["wind"], t1=statein.tgrs[:, 0],
            q1=statein.qgrs[:, 0, 0], prsl1=statein.prsl[:, 0], garea=grid.area,
            prsik1=statein.prsik[:, 0], prslk1=statein.prslk[:, 0], prslki=work["work3"],
            z1=prefix.carry.intdiag["zlvl"], dry=work["dry"], icy=work["icy"],
            **{f"{n}_{short}": work[f"{n}_{long}"] for n in ("ztmax", "tsurf")
               for short, long in (("wat", "water"), ("lnd", "land"), ("ice", "ice"))},
        )
        out = surface_composites_post_fractional(
            f, huge=huge, thsfc_loc=thsfc_loc, lheatstrg=control.lheatstrg,
            h0facu=control.h0facu, h0facs=control.h0facs)
    else:
        out = surface_composites_post(f, huge=huge)
    surface, work, diag = {}, dict(prefix.interstitial), dict(prefix.carry.intdiag)
    for n, value in out.items():
        if n in COMPOSITE_SURFACE_BINDINGS:
            surface[COMPOSITE_SURFACE_BINDINGS[n]] = value
        elif n in COMPOSITE_WORK_BINDINGS:
            work[COMPOSITE_WORK_BINDINGS[n]] = value
        else:
            diag[n] = value
    return replace(
        prefix,
        sfcprop=replace(prefix.sfcprop, **surface),
        interstitial=work,
        carry=replace(prefix.carry, intdiag=diag),
    )


def apply_surface_diagnostics(statein, grid, prefix, table, config):
    """Bind the maintained diagnostic kernel to source Stateout winds."""
    f = {
        n: getattr(prefix.sfcprop, n)
        for n in ("usfco", "vsfco", "evap", "f10m", "t2m", "q2m", "use_lake_model")
    }
    f.update(
        xlat_d=grid.xlat_d,
        xlon_d=grid.xlon_d,
        zf=prefix.carry.intdiag["zlvl"],
        ps=statein.pgr,
        u1=prefix.stateout.gu0[:, 0],
        v1=prefix.stateout.gv0[:, 0],
        t1=statein.tgrs[:, 0],
        q1=statein.qgrs[:, 0, 0],
        prslki=prefix.interstitial["work3"],
        fm=prefix.sfcprop.ffmm,
        fh=prefix.sfcprop.ffhh,
        fm10=prefix.interstitial["fm10"],
        fh2=prefix.interstitial["fh2"],
        ust=prefix.sfcprop.uustar,
        tskin=prefix.sfcprop.tsfc,
        qsurf=prefix.sfcprop.qss,
        shflx=prefix.sfcprop.hflx,
        cdq=prefix.interstitial["cdq"],
        wind=prefix.interstitial["wind"],
    )
    out = sfc_diag_run(f, config, fpvs_table=table)
    diag = {**prefix.carry.intdiag, "u10m": out["u10m_diag"], "v10m": out["v10m_diag"]}
    # Non-RUC source does not assign dpt2m. The following post owns its write.
    if config.lsm == config.lsm_ruc:
        diag["dpt2m"] = out["dpt2m"]
    return replace(
        prefix,
        sfcprop=replace(prefix.sfcprop, **{n: out[n] for n in ("f10m", "t2m", "q2m")}),
        carry=replace(prefix.carry, intdiag=diag),
    )


DIAG_POST_FIELDS = (
    "tmpmin",
    "tmpmax",
    "spfhmin",
    "spfhmax",
    "wind10mmax",
    "u10mmax",
    "v10mmax",
    "dpt2m",
)


def surface_diagnostic_post(f, control: SurfacePostControl):
    out = {n: f[n] for n in ("t2m", "q2m") + DIAG_POST_FIELDS}
    if control.opt_diag in (2, 3):
        out.update(
            t2m=torch.where(f["dry"], f["t2mmp"], f["t2m"]),
            q2m=torch.where(f["dry"], f["q2mp"], f["q2m"]),
        )
    if control.lssav:
        for extreme, base, op in (
            ("tmpmin", "t2m", torch.minimum),
            ("tmpmax", "t2m", torch.maximum),
            ("spfhmin", "q2m", torch.minimum),
            ("spfhmax", "q2m", torch.maximum),
        ):
            out[extreme] = op(f[extreme], out[base])
        square = f["u10m"] * f["u10m"] + f["v10m"] * f["v10m"]
        wind = torch.where(
            square > 0.0, torch.sqrt(torch.where(square > 0.0, square, 1.0)), 0.0
        )
        larger = wind > f["wind10mmax"]
        out.update(
            {
                n: torch.where(larger, v, f[n])
                for n, v in (
                    ("wind10mmax", wind),
                    ("u10mmax", f["u10m"]),
                    ("v10mmax", f["v10m"]),
                )
            }
        )
        vapor = torch.clamp_min(
            f["pgr"] * out["q2m"] / (EPS - EPSM1 * out["q2m"]), 1e-8
        )
        logv = torch.log(vapor / 611.2)
        safe_log = torch.where(logv != 0.0, logv, 1.0)
        dpt = torch.div(243.5, torch.div(17.67, safe_log) - 1.0) + 273.14
        # The native infinity at logv=0 has this finite value and derivative limit.
        limit = 243.5 * logv / (17.67 - logv) + 273.14
        out["dpt2m"] = torch.where(logv != 0.0, dpt, limit)
    return out


def apply_surface_diagnostic_post(statein, prefix, control: SurfacePostControl):
    f = {n: prefix.carry.intdiag[n] for n in DIAG_POST_FIELDS + ("u10m", "v10m")}
    f.update({n: prefix.interstitial[n] for n in ("dry", "t2mmp", "q2mp")})
    f.update(pgr=statein.pgr, t2m=prefix.sfcprop.t2m, q2m=prefix.sfcprop.q2m)
    out = surface_diagnostic_post(f, control)
    return replace(
        prefix,
        sfcprop=replace(prefix.sfcprop, t2m=out["t2m"], q2m=out["q2m"]),
        carry=replace(
            prefix.carry,
            intdiag={**prefix.carry.intdiag, **{n: out[n] for n in DIAG_POST_FIELDS}},
        ),
    )


ACCUMULATORS = {
    "gflux": "gflx",
    "evbsa": "evbs",
    "evcwa": "evcw",
    "transa": "trans",
    "sbsnoa": "sbsno",
    "snowca": "snowc",
    "snohfa": "snohf",
    "ep": "ep1d",
    "srunoff": "runof",
    "tecan": "ecan",
    "tetran": "etran",
    "tedir": "edir",
    "paha": "pah",
}
GENERIC_DIAG_WRITES = (
    "epi",
    "gfluxi",
    "pahi",
    "t1",
    "q1",
    "u1",
    "v1",
    *ACCUMULATORS,
    "runoff",
    "twa",
)


def surface_generic_post(f, delt: torch.Tensor, control: SurfacePostControl):
    out = {n: f[n] for n in GENERIC_DIAG_WRITES}
    out.update(
        epi=f["ep1d"],
        gfluxi=f["gflx"],
        pahi=f["pah"],
        t1=f["tgrs_1"],
        q1=f["qgrs_1"],
        u1=f["ugrs_1"],
        v1=f["vgrs_1"],
    )
    if control.lssav:
        out.update({n: f[n] + f[forcing] * delt for n, forcing in ACCUMULATORS.items()})
        out["runoff"] = f["runoff"] + (f["drain"] + f["runof"]) * delt
        out["twa"] = f["waxy"]
    out.update(hflxq=f["hflx"], hffac=torch.ones_like(f["hflx"]))
    if control.lheatstrg:
        factor = 1.0 + torch.where(
            f["hflx"] > 0.0,
            control.h0facu * f["zvfun"],
            control.h0facs * f["zvfun"],
        )
        out["hffac"] = torch.where(f["dry"], factor, out["hffac"])
        out["hflxq"] = f["hflx"] / out["hffac"]
    out.update({n: f[n + "_save"] for n in ("vtype", "stype", "scolor", "slope")})
    return out


def apply_surface_generic_post(
    statein, prefix, delt: torch.Tensor, control: SurfacePostControl
):
    f = {
        n: prefix.carry.intdiag[n]
        for n in GENERIC_DIAG_WRITES + ("evbs", "evcw", "trans", "sbsno")
    }
    f.update(
        {
            n: prefix.interstitial[n]
            for n in (
                "ep1d",
                "gflx",
                "pah",
                "snowc",
                "snohf",
                "ecan",
                "etran",
                "edir",
                "drain",
                "zvfun",
                "dry",
            )
        }
    )
    f["runof"] = prefix.interstitial["runoff"]
    f.update(
        {
            n: getattr(prefix.sfcprop, n)
            for n in (
                "waxy",
                "hflx",
                "vtype_save",
                "stype_save",
                "scolor_save",
                "slope_save",
            )
        }
    )
    f.update(
        tgrs_1=statein.tgrs[:, 0],
        qgrs_1=statein.qgrs[:, 0, 0],
        ugrs_1=statein.ugrs[:, 0],
        vgrs_1=statein.vgrs[:, 0],
    )
    out = surface_generic_post(f, delt, control)
    return replace(
        prefix,
        sfcprop=replace(
            prefix.sfcprop, **{n: out[n] for n in ("vtype", "stype", "scolor", "slope")}
        ),
        interstitial={**prefix.interstitial, **{n: out[n] for n in ("hflxq", "hffac")}},
        carry=replace(
            prefix.carry,
            intdiag={
                **prefix.carry.intdiag,
                **{n: out[n] for n in GENERIC_DIAG_WRITES},
            },
        ),
    )


def run_surface_sequence(
    statein,
    grid,
    prefix,
    hpbl,
    noah_case,
    *,
    solhr,
    table,
    delt,
    julian,
    yearlen,
    surface_config,
    nsst_control,
    post_control: SurfacePostControl,
    huge: float,
    water_budgets=None,
):
    """Exactly two live passes, followed by the four selected R4 post calls.

    The surface layer, NSST and sea ice compile (``scheme``) as regions that return only the fields
    they write (``surface_layer_writes``, ``nsst_writes``, ``sea_ice_writes``).
    """
    for iteration in (1, 2):
        prefix = apply_surface_layer(statein, grid, prefix, surface_config)
        prefix = apply_nsst(
            statein, grid, prefix, solhr, table, nsst_control, iteration=iteration
        )
        budget = {} if water_budgets is not None else None
        if iteration == 1:
            prefix = scheme(apply_noahmp)(
                statein, grid, prefix, hpbl, noahmp_case_with_plan(noah_case, device=statein.pgr.device),
                julian=julian, yearlen=yearlen, water_budget=budget,
            )
        else:
            prefix = skip_inactive_noahmp_pass(statein, prefix, water_budget=budget)
        if water_budgets is not None:
            water_budgets.append(budget)
        prefix = apply_sfc_land(
            prefix, cpllnd2atm=noah_case["driver_options"]["cpllnd2atm"]
        )
        prefix = apply_sea_ice(
            statein, prefix, table, delt, thsfc_loc=surface_config.thsfc_loc
        )
        prefix = finish_surface_iteration(
            prefix,
            iteration=iteration,
            lsm=post_control.lsm,
            lsm_noahmp=post_control.lsm_noahmp,
            nstf_name1=nsst_control.nstf_name[0],
        )
    prefix = glue(apply_surface_composites_post)(prefix, post_control, huge=huge, statein=statein, grid=grid,
                                           thsfc_loc=surface_config.thsfc_loc)
    prefix = glue(apply_surface_diagnostics)(statein, grid, prefix, table, surface_config)
    prefix = glue(apply_surface_diagnostic_post)(statein, prefix, post_control)
    return glue(apply_surface_generic_post)(statein, prefix, delt, post_control)
