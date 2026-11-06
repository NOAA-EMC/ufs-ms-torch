# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Complete Noah-MP copyback bound to the selected R4 generated cap."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch
from noahmp_land_surface_torch.driver import (
    NOAHMP_DRIVER_PLAN_KEY,
    PREPARED_METADATA_KEY,
    prepare_noahmp_driver_plan,
    NOAHMP_DRIVER_OUTPUT_FIELDS,
    NOAHMP_WATER_BUDGET_TERMS,
    PREPARED_TENSOR_COLUMNS_KEY,
    SMOOTH_TENSOR_OVERRIDE_KEY,
    noahmp_driver_begin,
    noahmp_driver_finish,
)
from noahmp_land_surface_torch.land_energy import (
    _vege_flux_iteration, load_land_energy_tables, vege_flux_opt_sfc3_iterations,
)
from ufs_physics_common import compile_policy

from .clock import _nint_nonnegative
from .physics_prefix import PhysicsPrefixResult

SURFACE_BINDINGS = {
    **{
        name: name
        for name in (
            "tg3 shdmin shdmax snoalb srflag smc stc slc canopy snowxy tvxy tgxy "
            "canicexy canliqxy eahxy tahxy cmxy chxy fwetxy sneqvoxy alboldxy "
            "qsnowxy wslakexy zwtxy waxy wtxy tsnoxy zsnsoxy snicexy snliqxy "
            "lfmassxy rtmassxy stmassxy woodxy stblcpxy fastcpxy xlaixy xsaixy "
            "taussxy smoiseq smcwtdxy deeprechxy rechxy rca"
        ).split()
    },
    "soiltyp": "stype",
    "soilcol": "scolor",
    "vegtype": "vtype",
    "slopetyp": "slope",
    "rainn_mp": "drainncprv",
    "rainc_mp": "draincprv",
    "snow_mp": "dsnowprv",
    "graupel_mp": "dgraupelprv",
    "ice_mp": "diceprv",
    "weasd": "weasdl",
    "snwdph": "snodl",
    "tskin": "tsfcl",
    "zorl": "zorll",
    "rmol1": "rmol",
    "flhc1": "flhc",
    "flqc1": "flqc",
    "albdvis": "albdirvis_lnd",
    "albdnir": "albdirnir_lnd",
    "albivis": "albdifvis_lnd",
    "albinir": "albdifnir_lnd",
    "emiss": "emis_lnd",
    "sncovr1": "sncovr",
}
INTERSTITIAL_BINDINGS = {
    **{
        name: name
        for name in (
            "sigmaf dry wind flag_iter drain runoff pah ecan etran edir snowc snohf t2mmp q2mp zvfun"
        ).split()
    },
    "cm": "cd_land",
    "ch": "cdq_land",
    "prslki": "work3",
    "xcoszin": "xcosz",
    "tprcp": "tprcp_land",
    "tsurf": "tsurf_land",
    "rb1": "rb_land",
    "fm1": "ffmm_land",
    "fh1": "ffhh_land",
    "ustar1": "uustar_land",
    "stress1": "stress_land",
    "fm101": "fm10_land",
    "fh21": "fh2_land",
    "qsurf": "qss_land",
    "gflux": "gflx_land",
    "evap": "evap_land",
    "hflx": "hflx_land",
    "ep": "ep1d_land",
    "cmm": "cmm_land",
    "chh": "chh_land",
    "ztmax": "ztmax_land",
}
DIAGNOSTIC_BINDINGS = {
    **{
        name: name
        for name in "rhonewsn1 trans evbs evcw sbsno smcwlt2 smcref2 wet1".split()
    },
    "dlwflx": "dlwsfci",
    "dswsfc": "dswsfci",
    "snet": "nswsfci",
    "zf": "zlvl",
    "stm": "soilm",
}

_COUPLED_COLUMN_NAMES = "_coupled_live_column_names"
NOAHMP_COUPLED_INPUT_FIELDS = frozenset(SURFACE_BINDINGS) | frozenset(INTERSTITIAL_BINDINGS) | frozenset(
    DIAGNOSTIC_BINDINGS) | frozenset(("ps", "u1", "v1", "t1", "q1", "prsl1", "prslk1", "prsik1",
                                    "xlatin", "garea", "pblh", "sfalb"))


def prepare_noahmp_coupled_case(surface, *, driver_options: Mapping[str, Any]) -> dict[str, Any]:
    """Offline grid contract only; every numerical input comes from the live coupling.

    Do not materialize cold forcing, run radiation, or prepare unused land tensors here.
    These five host fields are exactly the driver's immutable partition/branch metadata.
    """
    fields = {"dry": _nint_nonnegative(surface.slmsk).to(torch.int32) == 1,
              "flag_iter": torch.ones_like(surface.slmsk, dtype=torch.bool),
              "vegtype": surface.vtype, "snowxy": surface.snowxy, "shdmax": surface.shdmax}
    columns = [dict(zip(fields, values, strict=True)) for values in zip(
        *(value.tolist() for value in fields.values()), strict=True)]
    return {"name": "coupled_live_land", "columns": columns,
            "dimensions": {"im": surface.slmsk.numel(), "km": 4, "lsnowl": -2},
            "driver_options": dict(driver_options), _COUPLED_COLUMN_NAMES: NOAHMP_COUPLED_INPUT_FIELDS}


def noahmp_surface_inputs(
    statein, grid, prefix: PhysicsPrefixResult, hpbl: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Bind every array argument; never substitute a case's cold forcing or state."""
    fields = {
        name: getattr(prefix.sfcprop, member)
        for name, member in SURFACE_BINDINGS.items()
    }
    fields.update(
        {
            name: prefix.interstitial[member]
            for name, member in INTERSTITIAL_BINDINGS.items()
        }
    )
    fields.update(
        {
            name: prefix.carry.intdiag[member]
            for name, member in DIAGNOSTIC_BINDINGS.items()
        }
    )
    fields.update(
        ps=statein.pgr,
        u1=statein.ugrs[:, 0],
        v1=statein.vgrs[:, 0],
        t1=statein.tgrs[:, 0],
        q1=statein.qgrs[:, 0, 0],
        prsl1=statein.prsl[:, 0],
        prslk1=statein.prslk[:, 0],
        prsik1=statein.prsik[:, 0],
        xlatin=grid.xlat,
        garea=grid.area,
        pblh=hpbl,
        sfalb=prefix.carry.radtend["sfalb"],
    )
    fields["soiltyp"] = fields["soiltyp"][:, None].expand(-1, 4)
    return fields


_PLANNED_CASES: dict[tuple[int, str], tuple[Mapping[str, Any], dict[str, Any]]] = {}


def noahmp_case_with_plan(prepared_case: Mapping[str, Any], *, device: torch.device) -> dict[str, Any]:
    """``prepared_case`` with the driver's per-grid plan (``prepare_noahmp_driver_plan``).

    Built once per prepared case and device outside the compiled Noah-MP region, so a call runs no
    per-column host work and a compiled call traces none. The caller's case is not changed (it
    enters the continuation digest); the results equal the driver without the plan.
    """
    key = (id(prepared_case), str(device))
    hit = _PLANNED_CASES.get(key)
    if hit is None or hit[0] is not prepared_case:
        plan = prepare_noahmp_driver_plan(prepared_case, device=device, live_activity=True)
        hit = _PLANNED_CASES[key] = (prepared_case, {**prepared_case, NOAHMP_DRIVER_PLAN_KEY: plan})
    return hit[1]


def release_noahmp_prepared_columns(prepared_case: Mapping[str, Any]) -> dict[str, Any]:
    """``prepared_case`` without its resident prepared tensor columns, for a caller that keeps it.

    ``apply_noahmp`` replaces them with the live host fields on every call, and its driver plan
    comes from the host columns, so it never reads these full-grid copies (0.2 GB at x1.163842);
    its results are the same bits. The offline driver re-prepares a case without them.
    """
    return {key: value for key, value in prepared_case.items() if key != PREPARED_TENSOR_COLUMNS_KEY}


def apply_noahmp(
    statein,
    grid,
    prefix: PhysicsPrefixResult,
    hpbl: torch.Tensor,
    prepared_case: Mapping[str, Any],
    *,
    julian: torch.Tensor,
    yearlen: torch.Tensor,
    water_budget: dict[str, torch.Tensor] | None = None,
) -> PhysicsPrefixResult:
    """Use the existing complete driver and its live per-pass activity mask.

    Preparation owns immutable grid/options and coefficient tables, not physics.
    All array arguments and both calendar values come from the live host state.
    """
    ctx = _apply_noahmp_begin(statein, grid, prefix, hpbl, prepared_case, julian=julian, yearlen=yearlen,
                              water_budget=water_budget)
    if ctx["canopy"] is not None:
        ctx = {**ctx, "canopy": vege_flux_opt_sfc3_iterations(ctx["canopy"])}
    return _apply_noahmp_finish(prefix, ctx)


def _apply_noahmp_begin(statein, grid, prefix: PhysicsPrefixResult, hpbl: torch.Tensor,
                        prepared_case: Mapping[str, Any], *, julian: torch.Tensor, yearlen: torch.Tensor,
                        water_budget: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
    """``apply_noahmp`` up to the canopy iteration loop (``noahmp_driver_begin``)."""
    live = noahmp_surface_inputs(statein, grid, prefix, hpbl)
    required = (prepared_case[_COUPLED_COLUMN_NAMES] if _COUPLED_COLUMN_NAMES in prepared_case
                else set(prepared_case["columns"][0]))
    if required != set(live):
        raise ValueError(
            f"unbound Noah-MP arguments: missing={sorted(required - set(live))}, extra={sorted(set(live) - required)}"
        )
    ncol = statein.pgr.numel()
    if len(prepared_case["columns"]) != ncol:
        raise ValueError("Noah-MP grid preparation has a different column extent")
    for name, value in live.items():
        if value.shape[0] != ncol or value.device != statein.pgr.device:
            raise ValueError(f"invalid live Noah-MP field {name}")
    for value in (julian, yearlen):
        if value.shape != () or value.device != statein.pgr.device:
            raise ValueError(
                "Noah-MP calendar values must be scalar tensors on the numerical device"
            )
    activity = prefix.interstitial["dry"] & prefix.interstitial["flag_iter"]
    prepared = {**live, "snow_levels": _nint_nonnegative(-live["snowxy"]).neg()}
    case = {
        **prepared_case,
        PREPARED_TENSOR_COLUMNS_KEY: prepared,
        SMOOTH_TENSOR_OVERRIDE_KEY: live,
        "driver_options": _live_driver_options(prepared_case, julian, yearlen),
    }
    return noahmp_driver_begin(
        case, device=statein.pgr.device, activity=activity, water_budget=water_budget
    )


def _live_driver_options(prepared_case: Mapping[str, Any], julian: torch.Tensor, yearlen: torch.Tensor) -> dict[str, Any]:
    return {**prepared_case["driver_options"], "julian": julian, "iyrlen": yearlen}


def _apply_noahmp_finish(prefix: PhysicsPrefixResult, ctx: Mapping[str, Any]) -> PhysicsPrefixResult:
    """``apply_noahmp`` after the canopy iteration loop (``noahmp_driver_finish``)."""
    return _apply_noahmp_result(prefix, noahmp_driver_finish(ctx))


def _apply_noahmp_result(prefix: PhysicsPrefixResult, result: Mapping[str, torch.Tensor]) -> PhysicsPrefixResult:
    """The driver's copyback bound back into the carried prefix (no tensor work)."""
    if set(result) != set(NOAHMP_DRIVER_OUTPUT_FIELDS):
        raise RuntimeError("Noah-MP complete copyback surface changed")
    surface, work, diag, radtend = (
        {},
        dict(prefix.interstitial),
        dict(prefix.carry.intdiag),
        dict(prefix.carry.radtend),
    )
    for name, value in result.items():
        if name in SURFACE_BINDINGS:
            surface[SURFACE_BINDINGS[name]] = value
        elif name in INTERSTITIAL_BINDINGS:
            work[INTERSTITIAL_BINDINGS[name]] = value
        elif name in DIAGNOSTIC_BINDINGS:
            diag[DIAGNOSTIC_BINDINGS[name]] = value
        elif name == "sfalb":
            radtend[name] = value
        else:
            raise RuntimeError(f"unbound Noah-MP output {name}")
    return replace(
        prefix,
        sfcprop=replace(prefix.sfcprop, **surface),
        interstitial=work,
        carry=replace(prefix.carry, intdiag=diag, radtend=radtend),
    )


def _apply_noahmp_compiled(statein, grid, prefix: PhysicsPrefixResult, hpbl: torch.Tensor,
                           prepared_case: Mapping[str, Any], *, julian: torch.Tensor, yearlen: torch.Tensor,
                           water_budget: dict[str, torch.Tensor] | None = None) -> PhysicsPrefixResult:
    """The compiled form of ``apply_noahmp``.

    Without a recorded gradient: three exact-policy regions, the part before the canopy iteration
    loop, one loop pass (four flag specializations, run 20 times from Python) and the part after.
    One region unrolled the 20 passes, which held 40 % of its FX graph and of its cold compile. Each
    part equals eager bit for bit. With a gradient recorded: one region, as before. Region
    boundaries regroup the sums of gradient contributions, so the parts' reverse mode differed
    from eager by up to 4 ulp (aot_eager, 80-column capture) where the single region's is exact.
    The prefix is bound outside the regions (no tensor work).
    """
    # The calendar enters the regions as 1-element views: see _region_calendar.
    calendar = (julian.reshape(1), yearlen.reshape(1))
    if torch.is_grad_enabled():
        result = compile_policy.compiled(_apply_noahmp_region)(statein, grid, prefix, hpbl, prepared_case, calendar,
                                                              water_budget=water_budget)
        return _apply_noahmp_result(prefix, result)
    ctx = compile_policy.compiled(_apply_noahmp_begin_region)(statein, grid, prefix, hpbl, prepared_case, calendar,
                                                             water_budget=water_budget)
    if ctx["canopy"] is not None:
        ctx = {**ctx, "canopy": vege_flux_opt_sfc3_iterations(
            ctx["canopy"], step=compile_policy.compiled(_vege_flux_iteration))}
    # The finish does not read full-grid cold columns when every partition is already live.
    # Keep those tensors and the unused plan entries out of tracing and layout normalization.
    if all(subset is not None for _, subset, _ in ctx["groups"]):
        prepared_case = {"driver_options": prepared_case["driver_options"],
            NOAHMP_DRIVER_PLAN_KEY: {"groups": prepared_case[NOAHMP_DRIVER_PLAN_KEY]["groups"]}}
        ctx = {**ctx, "case": {}}
    result = compile_policy.compiled(_apply_noahmp_finish_region)(prepared_case, calendar, ctx)
    return _apply_noahmp_result(prefix, result)


# What crosses a region boundary. Host records of the plan's partitions stay out: Dynamo realizes a
# host list it returns and guards every value in it (480,000 guards, 58 s at x1.10242). So do 0-d
# CPU float64 tensors (the calendar, the energy tables' scalars): torch 2.8 gives each such region
# input a float symbol, specializes it after tracing and restarts the whole trace (9 + 20 s at
# x1.10242). Each part takes them from the same source as eager instead.
_HOST_PARTITION_ENTRIES = ("columns", PREPARED_METADATA_KEY)


def _region_calendar(calendar):
    julian, yearlen = calendar
    return julian.reshape(()), yearlen.reshape(())


def _apply_noahmp_region(statein, grid, prefix, hpbl, prepared_case, calendar, *, water_budget=None):
    julian, yearlen = _region_calendar(calendar)
    ctx = _apply_noahmp_begin(statein, grid, prefix, hpbl, prepared_case, julian=julian, yearlen=yearlen,
                              water_budget=water_budget)
    if ctx["canopy"] is not None:
        ctx = {**ctx, "canopy": vege_flux_opt_sfc3_iterations(ctx["canopy"])}
    return noahmp_driver_finish(ctx)


def _apply_noahmp_begin_region(statein, grid, prefix, hpbl, prepared_case, calendar, *, water_budget=None):
    julian, yearlen = _region_calendar(calendar)
    ctx = _apply_noahmp_begin(statein, grid, prefix, hpbl, prepared_case, julian=julian, yearlen=yearlen,
                              water_budget=water_budget)
    host = (*_HOST_PARTITION_ENTRIES, "driver_options")
    groups = [(glacier, None if subset is None else {key: value for key, value in subset.items() if key not in host},
               selector) for glacier, subset, selector in ctx["groups"]]
    hij = None if ctx["hij"] is None else {key: value for key, value in ctx["hij"].items()
                                           if key not in ("opts", "energy_tables")}
    # The live case crosses as its full-grid columns only (inputs or one computed field); the finish
    # part rebuilds the rest from prepared_case for the deferred glacier partition.
    case = ctx["case"]
    live = {key: case[key] for key in (PREPARED_TENSOR_COLUMNS_KEY, SMOOTH_TENSOR_OVERRIDE_KEY)}
    return {**ctx, "groups": groups, "hij": hij, "case": live}


def _apply_noahmp_finish_region(prepared_case, calendar, ctx):
    options = _live_driver_options(prepared_case, *_region_calendar(calendar))
    templates = [template for _, template, _ in prepared_case[NOAHMP_DRIVER_PLAN_KEY]["groups"]]
    groups = [(glacier, None if subset is None else
               {**subset, **{key: template[key] for key in _HOST_PARTITION_ENTRIES}, "driver_options": options},
               selector) for (glacier, subset, selector), template in zip(ctx["groups"], templates, strict=True)]
    case = ({**prepared_case, **ctx["case"], "driver_options": options}
            if any(subset is None for _, subset, _ in groups) else {})
    hij = ctx["hij"]
    if hij is not None:
        temperature = hij["state"].temperature_forcing
        hij = {**hij, "opts": options, "energy_tables": load_land_energy_tables(
            device=temperature.device, dtype=temperature.dtype, table_variant=hij["tables"].table_variant)}
    return noahmp_driver_finish({**ctx, "groups": groups, "hij": hij, "case": case})


# The scheme switch (runtime_policy.scheme) compiles apply_noahmp through compile_policy.compiled:
# register the partitioned form as its compiled form. Each part makes its own eager first call.
_apply_noahmp_compiled.__wrapped__ = apply_noahmp
_apply_noahmp_compiled.warm = lambda *args, **kwargs: None
compile_policy._compiled[apply_noahmp] = _apply_noahmp_compiled


def skip_inactive_noahmp_pass(
    statein,
    prefix: PhysicsPrefixResult,
    *,
    water_budget: dict[str, torch.Tensor] | None = None,
) -> PhysicsPrefixResult:
    """Second R4 surface pass: noahmpdrv writes nothing, so the call is skipped.

    Pass 1 clears flag_iter on every dry column because lsm == lsm_noahmp, and
    composites make dry (land) and wet (ocean/lake) disjoint. The device assertion
    keeps that proof checked. An inactive driver call leaves every copyback member
    unchanged and its budget at zero, so the result is the same.
    """
    work = prefix.interstitial
    torch._assert_async(
        ~(work["dry"] & work["flag_iter"]).any(), "second-pass Noah-MP column is active"
    )
    if water_budget is not None:
        if water_budget:
            raise ValueError("Noah-MP water_budget must be an empty output mapping")
        zero = statein.pgr
        water_budget.update({name: torch.zeros_like(zero) for name in NOAHMP_WATER_BUDGET_TERMS})
        water_budget.update(active=torch.zeros_like(zero, dtype=torch.bool),
            glacier=torch.zeros_like(zero, dtype=torch.bool))
    return prefix

def finish_surface_iteration(
    prefix: PhysicsPrefixResult,
    *,
    iteration: int,
    lsm: int,
    lsm_noahmp: int,
    nstf_name1: int,
) -> PhysicsPrefixResult:
    """GFS_surface_loop_control_part2: reset guesses and select the next pass."""
    if type(iteration) is not int or iteration not in (1, 2):
        raise ValueError("surface iteration must be 1 or 2")
    if any(type(value) is not int for value in (lsm, lsm_noahmp, nstf_name1)):
        raise TypeError("surface loop model controls must be static integers")
    work = prefix.interstitial
    active = (
        (work["wind"] < 2.0)
        & ((work["dry"] & (lsm != lsm_noahmp)) | (work["wet"] & (nstf_name1 > 0)))
        & (iteration == 1)
    )
    return replace(
        prefix,
        interstitial={
            **work,
            "flag_iter": active,
            "flag_guess": torch.zeros_like(active),
        },
    )


def apply_sfc_land(
    prefix: PhysicsPrefixResult, *, cpllnd2atm: bool
) -> PhysicsPrefixResult:
    """The selected uncoupled sfc_land_run returns without writes."""
    if type(cpllnd2atm) is not bool:
        raise TypeError("land-to-atmosphere coupling must be a static boolean")
    if cpllnd2atm:
        raise NotImplementedError("active externally coupled sfc_land is not ported")
    return prefix
