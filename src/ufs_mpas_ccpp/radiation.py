# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Native bridge to the existing R4 radiation group and its persistent writes.

The selected MPAS driver creates/resets/destroys Interstitial for every group.
Only Radtend, radiation Coupling/Intdiag members, Cldprop and Tbd persist here.
This is not the full CCPP initialization or the one-wave radiation scheduler.
"""

from __future__ import annotations

import math
from functools import partial

from dataclasses import dataclass, replace
from typing import Mapping

import torch
from rrtmgp_radiation_torch.radiation_group import (
    CON_PI,
    CZLIMT_DAY,
    RadiationConfig,
    _longwave_absorption_v17,
    _longwave_sources_tensor,
    _lw_columns,
    _sw_daylit_columns,
    coszmn_with_daily,
    run_radiation_group,
)
from rrtmgp_radiation_torch.setup import _solar_state
from rrtmgp_radiation_torch.tables import RrtmgpTableBundle

from .clock import StepClock, _nint_nonnegative, advance_utc_date
from .containers import Statein, Stateout
from .grid import Grid
from .radiation_seeds import RadiationSeeds
from ufs_physics_common.compile_policy import compiled
from ufs_physics_common import compile_policy

from .column_chunks import run_column_chunks
from .runtime_policy import (scheme_column_chunk, cycle_observers_enabled, scheme_compile_selected, step_graphs_enabled,
    sw_daylit_fraction)
from .surface_containers import Sfcprop


def _sw_column_regions(*args, **kwargs):
    return _sw_daylit_columns(*args, **kwargs, region=compiled)


_sw_column_regions.__wrapped__ = _sw_daylit_columns
_sw_column_regions.warm = lambda *args, **kwargs: None
compile_policy._compiled[_sw_daylit_columns] = _sw_column_regions


def _lw_tensor_region(fn):
    # Avoid repeated forward copies of call-local preparation. AD keeps the
    # canonical layouts: strided reductions need a separate adjoint envelope.
    borrowed = (not torch.is_grad_enabled()
                and fn in (_longwave_absorption_v17, _longwave_sources_tensor))
    return compiled(fn, normalize_layouts=not borrowed)


def _lw_column_regions(columns, **kwargs):
    region = _lw_tensor_region if columns["p_lay"].device.type == "mps" else compiled
    return _lw_columns(columns, **kwargs, region=region)


_lw_column_regions.__wrapped__ = _lw_columns
_lw_column_regions.warm = lambda *args, **kwargs: None
compile_policy._compiled[_lw_columns] = _lw_column_regions

_SFC_FLUXES = tuple(
    f"{wave}_{component}"
    for wave in ("sfcfsw", "sfcflw")
    for component in ("upfxc", "upfx0", "dnfxc", "dnfx0")
)
_TOP_FLUXES = (
    "topfsw_upfxc",
    "topfsw_dnfxc",
    "topfsw_upfx0",
    "topflw_upfxc",
    "topflw_upfx0",
)
_COUPLING_SURFACE = (
    "nirbmdi",
    "nirdfdi",
    "visbmdi",
    "visdfdi",
    "nirbmui",
    "nirdfui",
    "visbmui",
    "visdfui",
    "sfcdswc",
    "sfcdsw",
    "sfcnsw",
    "sfcdlw",
    "sfculw",
    "tsfc_radtime",
)
_CLOUD_DIAGNOSTICS = ("lwp_ex", "iwp_ex", "lwp_fc", "iwp_fc", "cldfra2d")
_SURFACE_INPUTS = (
    "tsfc",
    "slmsk",
    "fice",
    "landfrac",
    "lakefrac",
    "snodl",
    "snodi",
    "sncovr",
    "sncovr_ice",
    "zorl",
    "tisfc",
    "emis_lnd",
    "emis_ice",
    "emis_wat",
    "semisbase",
    "albdirvis_lnd",
    "albdirnir_lnd",
    "albdifvis_lnd",
    "albdifnir_lnd",
)


@dataclass(frozen=True)
class RadiationCarry:
    """Column-first persistent radiation members of the selected R4 containers."""

    radtend: Mapping[str, torch.Tensor]
    coupling: Mapping[str, torch.Tensor]
    intdiag: Mapping[str, torch.Tensor]
    cnvw: torch.Tensor
    phy_f3d: torch.Tensor

    @classmethod
    def allocate_r4(cls, *, ncol: int, nlev: int, device, ndtend: int | None = None,
                    h2o_diagnostics: bool = False, dtype: torch.dtype = torch.float64) -> RadiationCarry:
        if any(type(n) is not int or n < 1 for n in (ncol, nlev)):
            raise ValueError("radiation allocation requires positive static dimensions")
        if type(h2o_diagnostics) is not bool:
            raise TypeError("h2o diagnostic allocation switch must be a static boolean")
        if ndtend is not None and (type(ndtend) is not int or ndtend < 0):
            raise ValueError("diagnostic tendency allocation requires nonnegative ndtend")

        def zeros(*shape):
            return torch.zeros(shape, dtype=dtype, device=device)

        radtend = {
            name: zeros(ncol)
            for name in (*_SFC_FLUXES, "sfalb", "coszen", "coszdg", "tsflw", "semis")
        }
        radtend.update(
            {
                name: zeros(ncol, nlev)
                for name in ("htrsw", "htrlw", "ext550", "swhc", "lwhc")
            }
        )
        radtend["lwhd"] = zeros(ncol, nlev, 6)
        coupling = {name: zeros(ncol) for name in _COUPLING_SURFACE}
        coupling["condition"] = zeros(ncol)
        coupling.update(
            {
                name: zeros(ncol, nlev + 1)
                for name in ("fluxlwUP_radtime", "fluxlwDOWN_radtime")
            }
        )
        coupling["htrlw"] = zeros(ncol, nlev)
        intdiag = {
            name: zeros(ncol)
            for name in (
                *_TOP_FLUXES,
                *_CLOUD_DIAGNOSTICS,
                "total_albedo",
                "trans",
                "evbs",
                "evcw",
                "sbsno",
                "soilm",
                "smcwlt2",
                "smcref2",
                "wet1",
                "dusfc", "dvsfc", "dtsfc", "dqsfc",
                "dusfci", "dvsfci", "dtsfci", "dqsfci",
                "dugwd", "dvgwd", "zmtnblck",
                "rainc", "cldwrk", "cnvprcp",
                "rain", "ice", "snow", "graupel", "frzr", "frzrb",
                "frozr", "frozrb", "tsnowp", "tsnowpb", "pwat",
                "tdomr", "tdomzr", "tdomip", "tdoms", "totprcp", "totice",
                "totsnw", "totgrp", "toticeb", "totsnwb", "totgrpb",
                "u10max", "v10max", "spd10max", "pratemax",
                "suntim", "dlwsfc", "ulwsfc", "psmean", "u10m", "v10m",
            )
        }
        if ndtend is not None:
            intdiag["dtend"] = zeros(ncol, nlev, ndtend)
        for name in ("do3_dt_prd", "do3_dt_ozmx", "do3_dt_temp", "do3_dt_ohoz"):
            intdiag[name] = zeros(ncol, nlev)
        if h2o_diagnostics:  # GFS_typedefs: allocated when ldiag3d and h2o_phys
            for name in ("dqv_dt_prd", "dqv_dt_qvmx"):
                intdiag[name] = zeros(ncol, nlev)
        for name in (
            "cmm",
            "chh",
            "dpt2m",
            "tmpmax",
            "spfhmax",
            "u10mmax",
            "v10mmax",
            "wind10mmax",
            "epi",
            "gfluxi",
            "pahi",
            "t1",
            "q1",
            "u1",
            "v1",
            "gflux",
            "evbsa",
            "evcwa",
            "transa",
            "sbsnoa",
            "snowca",
            "snohfa",
            "ep",
            "srunoff",
            "tecan",
            "tetran",
            "tedir",
            "paha",
            "runoff",
            "twa",
        ):
            intdiag[name] = zeros(ncol)
        for name in ("tmpmin", "spfhmin"):
            intdiag[name] = torch.full(
                (ncol,), 9.969209968386869e36, dtype=dtype, device=device
            )
        intdiag["rhonewsn1"] = torch.full(
            (ncol,), 200.0, dtype=dtype, device=device
        )
        intdiag["fluxr"] = zeros(ncol, 45)
        intdiag["sr"] = zeros(ncol)
        for name in ("cnvprcpb", "totprcpb"):
            intdiag[name] = zeros(ncol, 1)
        intdiag["refl_10cm"] = torch.full(
            (ncol, nlev), -35.0, dtype=dtype, device=device
        )
        for name,value in (("refdmax",-35.),("refdmax263k",-35.),
                           ("t02max",-999.),("rh02max",-999.),
                           ("t02min",999.),("rh02min",999.)):
            intdiag[name] = torch.full((ncol,),value,dtype=dtype,device=device)
        intdiag["kdis_gw"] = zeros(ncol, nlev)
        intdiag["dudt_ogw"] = zeros(ncol, nlev)
        for name in ("upd_mf", "dwn_mf", "det_mf"):
            intdiag[name] = zeros(ncol, nlev)
        # Each field is replaced before its first read. Stable keys avoid cold-step dict guards.
        zero = zeros()
        for name in ("dlwsfci", "ulwsfci", "dswsfci", "dswsfcci", "nswsfci", "uswsfci", "psurf", "zlvl",
                     "du_ogwcol", "dv_ogwcol", "du_oblcol", "dv_oblcol", "du_osscol", "dv_osscol",
                     "du_ofdcol", "dv_ofdcol"):
            intdiag[name] = zero.expand(ncol)
        for name in ("dkt", "dku", "dvdt_ogw", "dudt_obl", "dvdt_obl", "dudt_oss", "dvdt_oss",
                     "dudt_ofd", "dvdt_ofd"):
            intdiag[name] = zero.expand(ncol, nlev)
        return cls(radtend, coupling, intdiag, zeros(ncol, nlev), zeros(ncol, nlev, 4))


def _host_nint_nonnegative(value: float) -> int:
    """Host twin of clock._nint_nonnegative for static intervals (same floor and half test)."""
    integral = math.floor(value)
    return integral + (value - integral >= 0.5)


def _radiation_reset_divisor(*, fhzero: float, fhswr: float, fhlwr: float, dtp: float, nszero: int) -> int:
    rad_steps = _host_nint_nonnegative(min(fhswr, fhlwr) / dtp)
    if rad_steps <= 0:
        raise ValueError("radiation reset interval rounds to zero")
    return nszero if _host_nint_nonnegative(fhzero * 3600.0) >= _host_nint_nonnegative(max(fhswr, fhlwr)) else rad_steps


def radiation_reset_step(step_index: int, *, fhzero: float, fhswr: float, fhlwr: float, dtp: float,
                         nszero: int, **_unused) -> bool:
    """Whether the radiation buckets reset at static step ``step_index`` (MOD(kdt, divisor) == 1)."""
    if type(step_index) is not int or step_index < 1:
        raise ValueError("radiation reset requires a positive static step_index")
    return step_index % _radiation_reset_divisor(fhzero=fhzero, fhswr=fhswr, fhlwr=fhlwr, dtp=dtp,
                                                 nszero=nszero) == 1


def physics_reset_step(step_index: int, *, nszero: int) -> bool:
    """Whether diag_phys_zero runs at static step ``step_index`` (MOD(kdt - 1, nszero) == 0)."""
    if type(step_index) is not int or step_index < 1:
        raise ValueError("physics reset requires a positive static step_index")
    return (step_index - 1) % nszero == 0


def _diagnostic_reset_values(values, defaults, clear):
    output = []
    for value, default in zip(values, defaults):
        reset_value = torch.zeros_like(value) if default == 0.0 else torch.full_like(value, default)
        output.append(reset_value if clear is None else torch.where(clear, reset_value, value))
    return tuple(output)


def reset_radiation_diagnostics(
    carry: RadiationCarry,
    kdt: torch.Tensor,
    *,
    fhzero: float,
    fhswr: float,
    fhlwr: float,
    dtp: float,
    nszero: int,
    step_index: int | None = None,
    reset: bool | None = None,
    region=None,
) -> RadiationCarry:
    """CCPP_driver timestep bucket reset; never clears held heating or cloud carries.

    With the static ``step_index`` the reset step is decided on the host (the device asserts
    ``kdt == step_index``, or with step graphs that ``kdt`` makes the same decision): other steps
    return ``carry`` unchanged and launch no kernels. ``reset`` instead gives that host decision
    itself (``radiation_reset_step``; the caller checks ``kdt`` against its step number), so a
    compiled call keys on the decision, not on the step.
    """
    if kdt.shape != () or kdt.dtype != torch.int32:
        raise ValueError("radiation reset requires the scalar int32 source timestep")
    if type(nszero) is not int or nszero < 1 or min(fhzero, fhswr, fhlwr, dtp) <= 0:
        raise ValueError("radiation reset requires positive static intervals")
    if kdt.device != carry.cnvw.device:
        raise ValueError("radiation reset operands must share a numerical device")
    if step_index is not None or reset is not None:
        if step_index is not None:
            decided = radiation_reset_step(step_index, fhzero=fhzero, fhswr=fhswr, fhlwr=fhlwr, dtp=dtp,
                                           nszero=nszero)
            if reset is not None and reset != decided:
                raise ValueError("radiation reset decision disagrees with step_index")
            reset = decided
            if not step_graphs_enabled():
                torch._assert_async(kdt == step_index, "radiation reset step_index disagrees with kdt")
        if type(reset) is not bool:
            raise TypeError("radiation reset decision must be a static bool")
        if step_graphs_enabled():  # the device clock must make the same decision (no step number)
            torch._assert_async((torch.fmod(kdt.to(torch.int64), _radiation_reset_divisor(
                fhzero=fhzero, fhswr=fhswr, fhlwr=fhlwr, dtp=dtp, nszero=nszero)) == 1) == reset,
                "radiation reset step disagrees with kdt")
        if not reset:
            return carry
        diag = dict(carry.intdiag)
        names = ("fluxr", *_TOP_FLUXES)
        call = _diagnostic_reset_values if region is None else region(_diagnostic_reset_values)
        values = call(tuple(diag[name] for name in names), (0.0,) * len(names), None)
        diag.update(zip(names, values))
        return replace(carry, intdiag=diag)
    hours = torch.full((), fhzero * 3600.0, dtype=torch.float64, device=kdt.device)
    rad = torch.full((), max(fhswr, fhlwr), dtype=torch.float64, device=kdt.device)
    short = torch.full(
        (), min(fhswr, fhlwr) / dtp, dtype=torch.float64, device=kdt.device
    )
    rad_steps = _nint_nonnegative(short)
    torch._assert_async(rad_steps > 0, "radiation reset interval rounds to zero")
    divisor = torch.where(
        _nint_nonnegative(hours) >= _nint_nonnegative(rad), nszero, rad_steps
    )
    clear = torch.fmod(kdt.to(torch.int64), divisor) == 1
    diag = dict(carry.intdiag)
    names = ("fluxr", *_TOP_FLUXES)
    call = _diagnostic_reset_values if region is None else region(_diagnostic_reset_values)
    values = call(tuple(diag[name] for name in names), (0.0,) * len(names), clear)
    diag.update(zip(names, values))
    return replace(carry, intdiag=diag)


def reset_physics_diagnostics(
    carry: RadiationCarry, kdt: torch.Tensor, *, nszero: int,
    ldiag3d: bool, qdiag3d: bool, oz_phys_2015: bool, huge: float,
    step_index: int | None = None, h2o_phys: bool = False, reset: bool | None = None,
    region=None,
) -> RadiationCarry:
    """CCPP_driver/diag_phys_zero for represented selected R4 diagnostics.

    This scope has Noah-MP, no IAU, MYNN or optional UGWP diagnostics; H2O physics with h2o_phys.
    Forecast-total precipitation survives a normal bucket reset. Held radiation,
    convective cloud carry and phy_f3d also survive; diagnostic cloud fields do not.
    With the static ``step_index`` the reset step is decided on the host (the device asserts
    ``kdt == step_index``, or with step graphs that ``kdt`` makes the same decision): other steps
    return ``carry`` unchanged, reset steps write the reset values directly. ``reset`` instead gives
    that host decision itself (``physics_reset_step``; the caller checks ``kdt`` against its step
    number), so a compiled call keys on the decision, not on the step.
    """
    if kdt.shape != () or kdt.dtype != torch.int32 or kdt.device != carry.cnvw.device:
        raise ValueError("physics reset requires device-local scalar int32 kdt")
    if type(nszero) is not int or nszero < 1:
        raise ValueError("physics reset requires positive static nszero")
    if any(type(v) is not bool for v in (ldiag3d,qdiag3d,oz_phys_2015,h2o_phys)):
        raise TypeError("physics reset flags must be static booleans")
    if step_index is None and reset is None:
        clear = torch.remainder(kdt.to(torch.int64)-1,nszero) == 0
    else:
        if step_index is not None:
            decided = physics_reset_step(step_index, nszero=nszero)
            if reset is not None and reset != decided:
                raise ValueError("physics reset decision disagrees with step_index")
            reset = decided
            if not step_graphs_enabled():
                torch._assert_async(kdt == step_index, "physics reset step_index disagrees with kdt")
        if type(reset) is not bool:
            raise TypeError("physics reset decision must be a static bool")
        if step_graphs_enabled():  # the device clock must make the same decision (no step number)
            torch._assert_async((torch.remainder(kdt.to(torch.int64)-1,nszero) == 0) == reset,
                                "physics reset step disagrees with kdt")
        if not reset:
            return carry
        clear = None
    zero = (
        "srunoff","evbsa","evcwa","snohfa","transa","snowca","sbsnoa",
        "sbsno","evbs","evcw","trans","snowmt_land","snowmt_ice","soilm",
        "tmpmax","dusfc","dvsfc","dtsfc","dqsfc","gflux","dlwsfc","ulwsfc",
        "suntim","runoff","tecan","tetran","tedir","ep","cldwrk","dugwd",
        "dvgwd","psmean","spfhmax","u10mmax","v10mmax","wind10mmax",
        "u10max","v10max","spd10max","rain","rainc","ice","snow","graupel",
        "u10m","v10m","dpt2m","zlvl","psurf","pwat","t1","q1","u1","v1",
        "chh","cmm","dlwsfci","ulwsfci","dswsfci","dswsfcci","nswsfci",
        "uswsfci","dusfci","dvsfci","dtsfci","dqsfci","gfluxi","epi",
        "smcwlt2","smcref2","wet1","sr","tdomr","tdomzr","tdomip","tdoms",
        "zmtnblck","paha","twa","pahi","cldfra","cldfra2d","total_albedo",
        "lwp_ex","iwp_ex","lwp_fc","iwp_fc","totprcpb","cnvprcpb","toticeb",
        "totsnwb","totgrpb","frzrb","frozrb","tsnowpb","zmtb","zogw","zlwb",
        "tau_mtb","tau_ogw","tau_ngw","tau_tofd","dudt_gw","dvdt_gw",
        "dtdt_gw","kdis_gw","dudt_ogw","dkt","dku","pratemax",
    )
    if ldiag3d:
        zero += ("dtend",)
        if qdiag3d:
            zero += ("upd_mf","dwn_mf","det_mf")
        if oz_phys_2015:
            zero += ("do3_dt_prd","do3_dt_ozmx","do3_dt_temp","do3_dt_ohoz")
        if h2o_phys:
            zero += ("dqv_dt_prd","dqv_dt_qvmx")
    defaults = {n:0. for n in zero}
    defaults.update(tmpmin=huge,spfhmin=huge,refl_10cm=-35.,refdmax=-35.,
        refdmax263k=-35.,t02max=-999.,t02min=999.,rh02max=-999.,rh02min=999.,
        max_hail_diam_sfc=-999.,rhonewsn1=200.)
    diag = dict(carry.intdiag)
    names = tuple(name for name in defaults if name in diag)
    call = _diagnostic_reset_values if region is None else region(_diagnostic_reset_values)
    values = call(tuple(diag[name] for name in names), tuple(defaults[name] for name in names), clear)
    diag.update(zip(names, values))
    return replace(carry,intdiag=diag)


for _reset in (reset_radiation_diagnostics, reset_physics_diagnostics):
    _regions = partial(_reset, region=compiled)
    _regions.__wrapped__ = _reset
    _regions.warm = lambda *args, **kwargs: None
    compile_policy._compiled[_reset] = _regions


def radiation_group_inputs(
    statein: Statein,
    stateout: Stateout,
    grid: Grid,
    sfcprop: Sfcprop,
    carry: RadiationCarry,
    seeds: RadiationSeeds,
) -> dict[str, torch.Tensor]:
    """Map the generated cap's inputs; scratch inouts start at source reset values."""
    ncol, nlev = statein.prsl.shape
    if (
        stateout.gq0.shape != (ncol, nlev, 10)
        or carry.cnvw.shape != (ncol, nlev)
        or carry.phy_f3d.shape != (ncol, nlev, 4)
        or seeds.icsdsw.shape != (ncol,)
        or seeds.icsdlw.shape != (ncol,)
    ):
        raise ValueError(
            "native radiation requires the R4 ten-tracer/four-phy-field layout"
        )
    state = {name: getattr(sfcprop, name) for name in _SURFACE_INPUTS}
    state.update(
        {
            name: getattr(grid, name)
            for name in ("xlat", "xlon", "sinlat", "coslat", "dx")
        }
    )
    state.update({name: getattr(statein, name).T for name in ("prsi", "prsl", "prslk")})
    state.update({name: getattr(stateout, name).T for name in ("gt0", "gu0", "gv0")})
    state.update(
        gq0=stateout.gq0.permute(2, 1, 0),
        hprime=sfcprop.hprime[:, 0],
        cnvw=carry.cnvw.T,
        phy_f3d_prev=carry.phy_f3d.permute(2, 1, 0),
        icsdsw=seeds.icsdsw,
        icsdlw=seeds.icsdlw,
        si=torch.ones(nlev + 1, dtype=statein.prsl.dtype, device=statein.prsl.device),
        semis_prev=carry.radtend["semis"],
        lwhc_prev=carry.radtend["lwhc"].T,
        swhc_prev=carry.radtend["swhc"].T,
        fluxr_prev=carry.intdiag["fluxr"].T,
    )
    # MPAS creates and destroys Interstitial around EACH group, including radiation.
    zero_layer = torch.zeros_like(state["prsl"])
    state["clouds_prev"] = zero_layer.unsqueeze(0).expand(9, nlev, ncol)
    for name in (
        "cld_cnv_frac",
        "cld_cnv_lwp",
        "cld_cnv_iwp",
        "cld_cnv_reliq",
        "cld_cnv_reice",
        "cld_pbl_lwp",
        "cld_pbl_iwp",
        "cld_pbl_reliq",
        "cld_pbl_reice",
        "dtdt",
        "dudt",
        "dvdt",
        "cldtausw",
        "cldtaulw",
    ):
        state[name + "_prev"] = zero_layer
    state["dqdt_prev"] = torch.zeros_like(state["gq0"])
    for name in ("UP_allsky", "UP_clrsky", "DOWN_allsky", "DOWN_clrsky"):
        state["fluxsw" + name + "_prev"] = torch.zeros_like(state["prsi"])
    for name in ("nirbm", "nirdf", "visbm", "visdf", "uvbfc", "uvbf0"):
        state["scmpsw_" + name + "_prev"] = torch.zeros_like(state["tsfc"])
    for value in state.values():
        if value.device != statein.prsl.device or value.dtype not in (
            statein.prsl.dtype,
            torch.int32,
        ):
            raise ValueError(
                "native radiation operands must be device-local int32 or the working dtype"
            )
    return state


def _radiation_group(inputs, config, tables):
    """The radiation group; with column chunks (runtime_policy.set_column_chunk) its two mains run
    their column work over chunks of that size.

    The shortwave first gathers the daylit columns of the whole mesh, then runs them in chunks, so
    chunks do not reduce the daylit compaction. Under grad each chunk is a checkpoint; with the
    radiation entry selected for compile, column work uses the same tensor regions with or without
    column chunks. The whole-mesh parts before and after the mains (setup to cloud diagnostics;
    post and fluxr) compile only without gradients (their compiled reverse is not qualified).
    """
    chunk = scheme_column_chunk("run_native_radiation")
    fraction = sw_daylit_fraction()
    compile_blocks = scheme_compile_selected("run_native_radiation")

    def blocks(fn, columns, **options):
        call = compiled(fn) if compile_blocks else fn
        if chunk is None:
            return call(columns, **options)
        return run_column_chunks(lambda part: call(part, **options), columns,
                                 ncol=columns["p_lay"].shape[-1], size=chunk, axis=-1, views=True,
                                 warm=(lambda part: call.warm(part, **options)) if compile_blocks else None)

    return run_radiation_group(inputs, config, tables=tables, sw_day_fraction=fraction, blocks=blocks,
                                      region=compiled if compile_blocks and not torch.is_grad_enabled() else None)


def sw_daylit_bound(previous, *, duration_seconds: int, grid: Grid, initial_date: torch.Tensor,
                    clock_control: Mapping, radiation_config: RadiationConfig,
                    physics_arguments: Mapping, config: Mapping, **_unused) -> float:
    """Geometric bound of the shortwave daylit share over the shortwave steps of an interval.

    Run setup (host work, outside the numerics): for every shortwave step of the interval that
    starts at ``previous`` (the arguments of ``run_coupled_interval``), count the columns with
    ``coszen >= 0.0001`` from the same solar formulas the radiation group uses. Returns the
    largest count plus 1 % and 512 columns, as a fraction of the columns, at most 1.0. A
    variable-resolution mesh with a dense region in daylight gets a share above one half; the
    bound follows it. Set it with ``runtime_policy.set_sw_daylit_fraction``.
    """
    if type(duration_seconds) is not int or duration_seconds <= 0:
        raise ValueError("daylit bound requires a positive static duration_seconds")
    host = torch.device("cpu")
    xlon, sinlat, coslat = (getattr(grid, name).to(host) for name in ("xlon", "sinlat", "coslat"))
    ncol = xlon.shape[0]
    dtp = float(physics_arguments["dtp"])
    date0 = initial_date.to(host)
    first = previous.step_index
    steps = range(first, first + int(duration_seconds // config["config_dt"]))  # one step per cycle
    nsswr, nhfrad = clock_control["nsswr"], clock_control["nhfrad"]
    most = 0
    for k in steps:
        if not (k % nsswr == 1 or nsswr == 1 or k <= nhfrad):
            continue
        phour = torch.tensor((k - 1) * dtp / 3600.0, dtype=torch.float64)
        date = advance_utc_date(date0, phour)
        _, sollag, sindec, cosdec = _solar_state(date, con_pi=CON_PI, device=host)
        coszen, _ = coszmn_with_daily(xlon=xlon, sinlat=sinlat, coslat=coslat,
            solhr=torch.fmod(phour + clock_control["initial_hour"], 24.0), sollag=sollag,
            sindec=sindec, cosdec=cosdec, deltsw=radiation_config.fhswr, deltim=radiation_config.dtf)
        most = max(most, int((coszen >= CZLIMT_DAY).sum()))
    return min(1.0, (math.ceil(most * 1.01) + 512) / ncol)


@dataclass(frozen=True)
class NativeRadiationResult:
    stateout: Stateout
    sfcprop: Sfcprop
    carry: RadiationCarry
    stages: Mapping[str, Mapping[str, torch.Tensor]]


def run_native_radiation(
    statein: Statein,
    grid: Grid,
    sfcprop: Sfcprop,
    carry: RadiationCarry,
    seeds: RadiationSeeds,
    config: RadiationConfig,
    *,
    clock: StepClock,
    tables: RrtmgpTableBundle | None,
) -> NativeRadiationResult:
    """Dispatch the source radiation group, preserving carry on skipped steps.

    The caller supplies live UTC/solar-hour controls and prepared coefficient tables.
    Static wave selection must agree with the device clock. One-wave calls and
    full physics_init remain unsupported; both-off calls execute no radiation.
    Scratch tendencies die with this group, as in CCPP_driver; held Radtend survives.
    Active outputs can share unchanged input fields; callers treat both as immutable.
    """
    if any(type(flag) is not bool for flag in (config.lsswr, config.lslwr)):
        raise TypeError("native radiation entry requires static active-wave selection")
    for flag in (clock.lsswr, clock.lslwr):
        if flag.shape != () or flag.dtype != torch.bool or flag.device != statein.prsl.device:
            raise ValueError("radiation clock flags must be scalar device-local booleans")
    torch._assert_async(
        (clock.lsswr == config.lsswr) & (clock.lslwr == config.lslwr),
        "static radiation selection disagrees with the device clock",
    )
    # atmos_model seeds Stateout before the gate, but calls no cap on skipped steps.
    if not (config.lsswr or config.lslwr):
        return NativeRadiationResult(Stateout.from_statein(statein), sfcprop, carry, {})
    config.require_supported()
    indices = (
        config.imp_physics,
        config.ntcw,
        config.ntrw,
        config.ntiw,
        config.ntsw,
        config.ntgl,
        config.ntinc,
        config.ntoz,
        config.ntrac,
        config.nleffr,
        config.nieffr,
        config.nseffr,
    )
    if indices != (88, 2, 3, 4, 5, 6, 7, 10, 10, 1, 2, 3):
        raise ValueError(
            "native radiation requires the selected R4 tracer/radius indices"
        )
    if tables is None or tables.device != statein.prsl.device:
        raise ValueError("native radiation coefficients must share the state device")
    # The radiation group reads these buffers and constructs functional replacements.
    stateout = Stateout(statein.ugrs, statein.vgrs, statein.tgrs, statein.qgrs)
    inputs = radiation_group_inputs(statein, stateout, grid, sfcprop, carry, seeds)
    stages = _radiation_group(inputs, config, tables)
    pre, surface, cloud, lw, post = (
        stages[name] for name in ("pre", "surface", "cloud_mp", "lw", "post")
    )
    rad = dict(carry.radtend)
    for name in (*_SFC_FLUXES, "tsflw"):
        rad[name] = post[name]
    for name in ("htrlw", "htrsw", "lwhc", "swhc"):
        rad[name] = post[name].T
    for name in ("semis", "sfalb"):
        rad[name] = surface[name]
    for name in ("coszen", "coszdg"):
        rad[name] = pre[name]
    if "ext550" in post:  # rrtmgp_aerosol_optics (iaerflg /= 0); zero otherwise
        rad["ext550"] = post["ext550"].T
    coupling = dict(carry.coupling)
    for name in _COUPLING_SURFACE:
        if name not in ("sfcdswc", "tsfc_radtime"):
            coupling[name] = post[name]
    coupling["tsfc_radtime"] = pre["tsfc_radtime"]
    coupling["htrlw"] = post["htrlwu"].T
    for name in ("fluxlwUP_radtime", "fluxlwDOWN_radtime"):
        coupling[name] = lw[name].T
    diag = dict(carry.intdiag)
    for name in (*_TOP_FLUXES, "total_albedo"):
        diag[name] = post[name]
    for name in _CLOUD_DIAGNOSTICS:
        diag[name] = cloud[name]
    diag["fluxr"] = post["fluxr"].T
    updated = replace(
        carry,
        radtend=rad,
        coupling=coupling,
        intdiag=diag,
        phy_f3d=cloud["phy_f3d"].permute(2, 1, 0),
    )
    stateout = Stateout(
        post["gu0"].T, post["gv0"].T, post["gt0"].T, post["gq0"].permute(2, 1, 0)
    )
    sfcprop = replace(
        sfcprop,
        **{
            name: surface[name]
            for name in (
                "emis_ice",
                "emis_lnd",
                "emis_wat",
                "semisbase",
                "albdirvis_lnd",
                "albdirnir_lnd",
                "albdifvis_lnd",
                "albdifnir_lnd",
            )
        },
    )
    if not cycle_observers_enabled():
        # Only the held solar controls are read downstream; returning every stage would keep
        # (and, compiled, materialize) all intermediate radiation fields.
        stages = {"setup": {n: stages["setup"][n] for n in ("slag", "sdec", "cdec")}}
    return NativeRadiationResult(stateout, sfcprop, updated, stages)

