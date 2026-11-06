# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Selected R4 physics prefix through interstitial2 and one surface-layer call."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

import torch
from gfs_surface_layer_torch.cases import SurfaceCaseConfig
from gfs_surface_layer_torch.driver import GRAV, RVRDM1, sfc_diff_run
from sa_sas_deep_convection_torch.samf_interstitials import mpas_tempo_layout

from .clock import _nint_nonnegative
from .containers import Statein, Stateout, static_tensor
from .grid import Grid
from .radiation import RadiationCarry
from .radiation_scaling import RadiationScalingControl, apply_held_radiation
from .runtime_policy import scheme
from .surface_composites import SurfaceCompositesControl, prepare_surface_composites
from .surface_containers import Sfcprop
from .precision import WORKING_DTYPES


@dataclass(frozen=True)
class PhysicsPrefixControl:
    composites: SurfaceCompositesControl
    radiation: RadiationScalingControl
    surface_layer: SurfaceCaseConfig
    isot: int
    ivegsrc: int
    ncnvwind: int
    lndp_type: int
    dxmin: float | torch.Tensor
    dxinv: float | torch.Tensor
    shal_cnv: bool
    old_monin: bool
    mstrat: bool
    do_shoc: bool
    imfshalcnv: int

    def __post_init__(self):
        for name in ("isot", "ivegsrc", "ncnvwind", "lndp_type", "imfshalcnv"):
            if type(getattr(self, name)) is not int:
                raise TypeError("physics prefix identifiers must be static integers")
        for name in ("shal_cnv", "old_monin", "mstrat", "do_shoc"):
            if type(getattr(self, name)) is not bool:
                raise TypeError("physics prefix switches must be static booleans")
        if self.isot not in (1, 2) or self.ivegsrc not in range(6) or self.ncnvwind < 1:
            raise ValueError("invalid physics prefix soil/vegetation/wind control")
        if self.lndp_type != 0:
            raise NotImplementedError("surface perturbation pre-call is not ported")
        inversion = (
            ((self.imfshalcnv == 0 and self.shal_cnv) or self.old_monin) and self.mstrat
        ) or self.do_shoc
        if inversion or self.radiation.lsidea:
            raise NotImplementedError(
                "CTEI inversion and IDEA diagnostics are not ported"
            )
        for v in (self.dxmin, self.dxinv):
            if isinstance(v, torch.Tensor):
                if v.shape != () or v.dtype not in WORKING_DTYPES:
                    raise ValueError("invalid source area scaling tensor")
                torch._assert_async(torch.isfinite(v), "nonfinite source area scaling tensor")
            elif type(v) not in (float, int) or not math.isfinite(v):
                raise ValueError("invalid source area scaling constants")
        if (
            self.surface_layer.ivegsrc != self.ivegsrc
            or self.surface_layer.lsm != self.composites.lsm
            or self.surface_layer.lsm_ruc != self.composites.lsm_ruc
            or self.surface_layer.default_real_bytes != 8
        ):
            raise ValueError("inconsistent physics prefix surface controls")
        if self.surface_layer.sfc_z0_type not in (-1, 0):
            raise NotImplementedError("R4 Wang roughness coefficients are not rebased")


@dataclass(frozen=True)
class PhysicsPrefixResult:
    stateout: Stateout
    sfcprop: Sfcprop
    carry: RadiationCarry
    phy_f2d: torch.Tensor
    interstitial: Mapping[str, torch.Tensor]


def merge_prefix_writes(prefix: PhysicsPrefixResult, writes: tuple[dict, dict]) -> PhysicsPrefixResult:
    """``prefix`` with a region's written ``(sfcprop, interstitial)`` fields (the ``*_writes`` regions).

    A compiled region that returns only what it writes reads (and so guards) only those fields; one
    that returned the whole carried prefix recompiled when an unread field changed layout.
    """
    surface, work = writes
    return replace(prefix, sfcprop=replace(prefix.sfcprop, **surface),
                   interstitial={**prefix.interstitial, **work})


def allocate_physics_prefix_scratch(
    statein: Statein, *, huge: float
) -> dict[str, torch.Tensor]:
    """The source group's reset values for fields used through the surface layer."""
    ncol, nlev = statein.tgrs.shape
    device, dtype = statein.tgrs.device, statein.tgrs.dtype
    if dtype not in WORKING_DTYPES or min(ncol, nlev) < 1:
        raise ValueError("physics prefix requires nonempty float32/float64 column/layer state")

    def zeros(*shape):
        return torch.zeros(shape, dtype=dtype, device=device)

    scratch = {
        n: zeros(ncol)
        for n in (
            "adjsfculw_land",
            "adjsfculw_ice",
            "adjsfculw_water",
            "gflx_ice",
            "gflx_water",
            "tseal",
            "dtzm",
            "gabsbdlw_land",
            "gabsbdlw_ice",
            "gabsbdlw_water",
            "work1",
            "work2",
            "work3",
            "sigmaf",
            "z01d",
            "zt1d",
            "bexp1d",
            "xlai1d",
            "vegf1d",
            "ctei_r",
            "ctei_rml",
            "zvfun",
            "gflx_land",
            "drain",
            "runoff",
            "pah",
            "ecan",
            "etran",
            "edir",
            "snowc",
            "snohf",
            "t2mmp",
            "q2mp",
            "snowmt",
            "cd",
            "cdq",
            "rb",
            "stress",
            "fm10",
            "fh2",
            "gflx",
            "ep1d",
            "hffac",
            "hflxq",
            "prcpmp",
            "rainmp",
            "icemp",
            "snowmp",
            "graupelmp",
        )
    }
    scratch.update(
        {
            n: torch.full((ncol,), huge, dtype=dtype, device=device)
            for n in (
                "ep1d_land",
                "cmm_land",
                "chh_land",
                "evap_land",
                "hflx_land",
                "qss_land",
                "tsfc_water",
                "tsurf_water",
                "tsurf_land",
                "tsurf_ice",
                "uustar_water",
                "uustar_land",
                "uustar_ice",
                "tprcp_water",
                "tprcp_land",
                "tprcp_ice",
                "ep1d_ice",
                "ep1d_water",
                "cmm_water",
                "chh_water",
                "evap_water",
                "hflx_water",
                "qss_water",
                "cmm_ice",
                "chh_ice",
                "evap_ice",
                "hflx_ice",
                "qss_ice",
            )
        }
    )
    for surface in ("water", "land", "ice"):
        for field in (
            "cd",
            "cdq",
            "rb",
            "stress",
            "ffmm",
            "ffhh",
            "fm10",
            "fh2",
            "ztmax",
        ):
            scratch[f"{field}_{surface}"] = (
                zeros(ncol)
                if field == "ztmax"
                else torch.full((ncol,), huge, dtype=dtype, device=device)
            )
    scratch.update(
        {
            n: torch.zeros(ncol, dtype=torch.bool, device=device)
            for n in (
                "dry",
                "wet",
                "icy",
                "flag_cice",
                "lake",
                "flag_lakefreeze",
                "flag_guess",
            )
        }
    )
    scratch["flag_iter"] = torch.ones(ncol, dtype=torch.bool, device=device)
    for name in ("islmsk", "islmsk_cice"):
        scratch[name] = torch.zeros(ncol, dtype=torch.int32, device=device)
    scratch["kinver"] = torch.full((ncol,), nlev, dtype=torch.int32, device=device)
    scratch["lndp_vgf"] = zeros()
    scratch["del"] = zeros(ncol, nlev)
    scratch["del_gz"] = zeros(ncol, nlev + 1)
    for name in ("dudt", "dvdt", "dtdt"):
        scratch[name] = zeros(ncol, nlev)
    scratch["dqdt"] = torch.zeros_like(statein.qgrs)
    layout = mpas_tempo_layout()
    if statein.qgrs.shape != (ncol, nlev, layout.ntrac):
        raise ValueError("physics scratch requires the R4 ten-tracer layout")
    scratch["otsptflag"] = static_tensor(layout.otsptflag, dtype=torch.bool, device=device)
    for name in ("vdftra", "dvdftra", "ten_q"):
        scratch[name] = torch.zeros_like(statein.qgrs)
    for name in ("ten_t", "ten_u", "ten_v", "ten_t_pbl", "ten_q_pbl"):
        scratch[name] = zeros(ncol, nlev)
    scratch["tkeh"] = zeros(ncol, nlev + 1)
    scratch["kpbl"] = torch.zeros(ncol, dtype=torch.int32, device=device)
    for name in ("dusfc1", "dvsfc1", "dtsfc1", "dqsfc1"):
        scratch[name] = zeros(ncol)
    return scratch


def _get_prs(
    state: Statein, scratch: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    virtual = state.tgrs * (1.0 + RVRDM1 * torch.clamp_min(state.qgrs[:, :, 0], 0.0))
    hydro = (state.phii[:, 1:] - state.phii[:, :-1]) / virtual
    return {
        **scratch,
        "del": state.prsi[:, :-1] - state.prsi[:, 1:],
        "del_gz": torch.cat((hydro, scratch["del_gz"][:, -1:]), dim=1),
    }


def _interstitial_1(
    state: Statein,
    surface: Sfcprop,
    grid: Grid,
    carry: RadiationCarry,
    scratch: Mapping[str, torch.Tensor],
    control: PhysicsPrefixControl,
):
    torch._assert_async((grid.area > 0.0).all(), "nonpositive physics cell area")
    torch._assert_async((surface.slmsk >= 0).all(), "negative source surface mask")
    work1 = torch.clamp(
        (torch.log(grid.area) - control.dxmin) * control.dxinv, 0.0, 1.0
    )
    work = {
        **scratch,
        "work1": work1,
        "work2": 1.0 - work1,
        "islmsk": _nint_nonnegative(surface.slmsk).to(torch.int32),
    }
    for name in ("dudt", "dvdt", "dtdt"):
        work[name] = torch.zeros_like(state.tgrs)
    work["dqdt"] = torch.zeros_like(state.qgrs)
    return replace(carry, intdiag={**carry.intdiag, "psurf": state.pgr}), work


def _surface_generic_pre(
    state: Statein,
    surface: Sfcprop,
    carry: RadiationCarry,
    phy_f2d: torch.Tensor,
    scratch: Mapping[str, torch.Tensor],
    control: PhysicsPrefixControl,
):
    if (
        phy_f2d.ndim != 2
        or phy_f2d.shape[0] != state.tgrs.shape[0]
        or control.ncnvwind > phy_f2d.shape[1]
        or phy_f2d.dtype != state.tgrs.dtype
        or phy_f2d.device != state.tgrs.device
    ):
        raise ValueError("invalid source Tbd.phy_f2d convection-wind allocation")
    ice = scratch["islmsk"] == 2
    ice_vegetation = {0: 24, 1: 15, 2: 13, 3: 15, 4: 24, 5: 15}[control.ivegsrc]
    surface = replace(
        surface,
        vtype_save=surface.vtype,
        stype_save=surface.stype,
        scolor_save=surface.scolor,
        slope_save=surface.slope,
        stype=torch.where(ice, 16 if control.isot == 1 else 9, surface.stype),
        vtype=torch.where(
            ice, ice_vegetation, torch.where(surface.vtype < 1, 17, surface.vtype)
        ),
        slope=torch.where(ice, 9, torch.where(surface.slope < 1, 1, surface.slope)),
    )
    speed_squared = (
        state.ugrs[:, 0] * state.ugrs[:, 0] + state.vgrs[:, 0] * state.vgrs[:, 0]
    )
    speed = torch.where(
        speed_squared > 0.0,
        torch.sqrt(torch.where(speed_squared > 0.0, speed_squared, 1.0)),
        0.0,
    )
    cnvwind = phy_f2d[:, control.ncnvwind - 1]
    wind = torch.clamp_min(speed + torch.clamp(cnvwind, 0.0, 30.0), 1.0)
    phy_f2d = torch.cat(
        (
            phy_f2d[:, : control.ncnvwind - 1],
            torch.zeros_like(cnvwind[:, None]),
            phy_f2d[:, control.ncnvwind :],
        ),
        dim=1,
    )
    work = {
        **scratch,
        "lndp_vgf": torch.full_like(scratch["lndp_vgf"], -999.0),
        "sigmaf": torch.clamp_min(surface.vfrac, 0.01),
        "islmsk_cice": scratch["islmsk"],
        "work3": state.prsik[:, 0] / state.prslk[:, 0],
        "wind": wind,
    }
    diag = {
        **carry.intdiag,
        "zlvl": state.phil[:, 0] * (1.0 / GRAV),
        "smcwlt2": torch.zeros_like(state.pgr),
        "smcref2": torch.zeros_like(state.pgr),
    }
    return surface, replace(carry, intdiag=diag), phy_f2d, work


def _surface_composites_inter(
    surface: Sfcprop, carry: RadiationCarry, scratch: Mapping[str, torch.Tensor]
):
    work = dict(scratch)
    down = carry.intdiag["dlwsfci"]
    for suffix, mask, emissivity in (
        ("land", "dry", surface.emis_lnd),
        ("ice", "icy", surface.emis_ice),
        ("water", "wet", surface.emis_wat),
    ):
        name = "gabsbdlw_" + suffix
        work[name] = torch.where(scratch[mask], emissivity * down, scratch[name])
    diag = {
        **carry.intdiag,
        "uswsfci": carry.intdiag["dswsfci"] - carry.intdiag["nswsfci"],
    }
    return replace(carry, intdiag=diag), work


def _interstitial_2(
    state: Statein,
    surface: Sfcprop,
    carry: RadiationCarry,
    scratch: Mapping[str, torch.Tensor],
    control: PhysicsPrefixControl,
):
    diag = dict(carry.intdiag)
    if control.radiation.lssav:
        cosine = scratch["xcosz"]
        daytime = cosine >= 0.0001
        irradiance = diag["dswsfci"] / torch.where(daytime, cosine, 1.0)
        diag["suntim"] = torch.where(
            daytime & (irradiance >= 120.0),
            diag["suntim"] + control.radiation.dtf,
            diag["suntim"],
        )
        if control.composites.frac_grid:
            # Area-weighted over land, ice and open water (flag_cice is false: uncoupled).
            frland = scratch["frland"]
            ice_fraction = (1.0 - frland) * surface.fice
            # A1: a huge placeholder (no open water) has zero weight; it enters as the ice value so
            # that the derivative with respect to fice stays finite. The forward value is unchanged.
            water = scratch["adjsfculw_water"]
            water = torch.where(water == control.composites.huge, scratch["adjsfculw_ice"], water)
            diag["ulwsfci"] = (
                scratch["adjsfculw_land"] * frland
                + scratch["adjsfculw_ice"] * ice_fraction
                + water * (1.0 - frland - ice_fraction)
            )
        else:
            mixed = scratch["wet"] & (scratch["adjsfculw_water"] != control.composites.huge)
            ice_flux = torch.where(
                mixed,
                scratch["adjsfculw_ice"] * surface.fice
                + scratch["adjsfculw_water"] * (1.0 - surface.fice),
                scratch["adjsfculw_ice"],
            )
            diag["ulwsfci"] = torch.where(
                scratch["dry"],
                scratch["adjsfculw_land"],
                torch.where(scratch["icy"], ice_flux, scratch["adjsfculw_water"]),
            )
        for bucket, current in (
            ("dlwsfc", diag["dlwsfci"]),
            ("ulwsfc", diag["ulwsfci"]),
            ("psmean", state.pgr),
        ):
            diag[bucket] = diag[bucket] + current * control.radiation.dtf
    work = {**scratch, "ctei_r": torch.full_like(scratch["ctei_r"], 10.0)}
    return replace(carry, intdiag=diag), work


def prepare_physics_prefix(
    statein: Statein,
    stateout: Stateout,
    grid: Grid,
    surface: Sfcprop,
    carry: RadiationCarry,
    phy_f2d: torch.Tensor,
    solar: Mapping[str, torch.Tensor],
    dtidx: torch.Tensor,
    control: PhysicsPrefixControl,
) -> PhysicsPrefixResult:
    """Execute the source's seven calls once; no Stateout reseed or second pass."""
    if statein.qgrs.shape[2] != 10:
        raise ValueError("physics prefix requires the selected R4 ten-tracer layout")
    for value in (control.dxmin, control.dxinv):
        if isinstance(value, torch.Tensor) and value.device != statein.tgrs.device:
            raise ValueError("physics prefix area scaling must share the state device")
    scratch = allocate_physics_prefix_scratch(statein, huge=control.composites.huge)
    scratch = _get_prs(statein, scratch)
    carry, scratch = _interstitial_1(statein, surface, grid, carry, scratch, control)
    surface, carry, phy_f2d, scratch = _surface_generic_pre(
        statein, surface, carry, phy_f2d, scratch, control
    )
    pre = prepare_surface_composites(statein, surface, scratch, control.composites)
    scaled = apply_held_radiation(
        statein,
        stateout,
        grid,
        pre.sfcprop,
        carry,
        pre.interstitial,
        solar,
        dtidx,
        control.radiation,
    )
    carry, scratch = _surface_composites_inter(
        pre.sfcprop, scaled.carry, scaled.interstitial
    )
    carry, scratch = _interstitial_2(statein, pre.sfcprop, carry, scratch, control)
    return PhysicsPrefixResult(scaled.stateout, pre.sfcprop, carry, phy_f2d, scratch)


def apply_surface_layer(
    statein: Statein, grid: Grid, prefix: PhysicsPrefixResult, config: SurfaceCaseConfig
) -> PhysicsPrefixResult:
    """Bind one existing sfc_diff call; NSST/land/ice must precede another pass.

    The compiled region (``scheme``) is ``surface_layer_writes``: it returns only the fields it writes.
    """
    return merge_prefix_writes(prefix, scheme(surface_layer_writes)(statein, grid, prefix, config))


def surface_layer_writes(
    statein: Statein, grid: Grid, prefix: PhysicsPrefixResult, config: SurfaceCaseConfig
) -> tuple[dict, dict]:
    """The sfcprop and interstitial fields one sfc_diff call writes (``apply_surface_layer``)."""
    surface, work = prefix.sfcprop, prefix.interstitial
    fields = {
        "ps": statein.pgr,
        "t1": statein.tgrs[:, 0],
        "q1": statein.qgrs[:, 0, 0],
        "z1": prefix.carry.intdiag["zlvl"],
        "garea": grid.area,
        "wind": work["wind"],
        "prsl1": statein.prsl[:, 0],
        "prslki": work["work3"],
        "prsik1": statein.prsik[:, 0],
        "prslk1": statein.prslk[:, 0],
        "sigmaf": work["sigmaf"],
        "lakefrac": surface.lakefrac,
        "fice": surface.fice,
        "vegtype": surface.vtype,
        "shdmax": surface.shdmax,
        "z0pert": work["z01d"],
        "ztpert": work["zt1d"],
        "u10m": prefix.carry.intdiag["u10m"],
        "v10m": prefix.carry.intdiag["v10m"],
        "u1": statein.ugrs[:, 0],
        "v1": statein.vgrs[:, 0],
        "usfco": surface.usfco,
        "vsfco": surface.vsfco,
        "tskin_wat": work["tsfc_water"],
        "tskin_lnd": surface.tsfcl,
        "tskin_ice": surface.tisfc,
        "tsurf_wat": work["tsurf_water"],
        "tsurf_lnd": work["tsurf_land"],
        "tsurf_ice": work["tsurf_ice"],
        "z0rl_wat": surface.zorlw,
        "z0rl_lnd": surface.zorll,
        "z0rl_ice": surface.zorli,
        "z0rl_wav": surface.zorlwav,
    }
    fields.update(
        {
            n: work[n]
            for n in ("flag_iter", "flag_lakefreeze", "dry", "icy", "wet", "zvfun")
        }
    )
    for short, long in (("wat", "water"), ("lnd", "land"), ("ice", "ice")):
        for field, member in (
            ("ustar", "uustar"),
            ("cm", "cd"),
            ("ch", "cdq"),
            ("rb", "rb"),
            ("stress", "stress"),
            ("fm", "ffmm"),
            ("fh", "ffhh"),
            ("fm10", "fm10"),
            ("fh2", "fh2"),
            ("ztmax", "ztmax"),
        ):
            fields[field + "_" + short] = work[member + "_" + long]
    out = sfc_diff_run(fields, config)
    torch._assert_async(out["errflg"] == 0, "source surface-layer configuration failed")
    surface = dict(zorlw=out["z0rl_wat"], zorll=out["z0rl_lnd"], zorli=out["z0rl_ice"])
    work = {"zvfun": out["zvfun"]}
    for short, long in (("wat", "water"), ("lnd", "land"), ("ice", "ice")):
        for field, member in (
            ("ustar", "uustar"),
            ("cm", "cd"),
            ("ch", "cdq"),
            ("rb", "rb"),
            ("stress", "stress"),
            ("fm", "ffmm"),
            ("fh", "ffhh"),
            ("fm10", "fm10"),
            ("fh2", "fh2"),
            ("ztmax", "ztmax"),
        ):
            work[member + "_" + long] = out[field + "_" + short]
    return surface, work
