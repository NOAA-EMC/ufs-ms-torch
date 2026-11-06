# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Torch-native Noah-MP land energy helpers for the sealed land corridor.

This module ports the radiation/albedo and bare-ground flux pieces used by
``module_sf_noahmplsm.F90`` for the current sealed Noah-MP corpus:

* ``idveg=4``
* ``opt_rad=3``
* ``opt_alb=1``
* ``opt_sfc=3``
* ``opt_trs=2``
* ``opt_stc=3``

It deliberately does not read checkpoint or NetCDF oracle files.  It reads only
the frozen Noah-MP table file and accepts tensor state prepared by the caller.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import torch
from ufs_physics_common.precision import working_power
from .thermo import static_tensor

from .compile import compile_categories_enabled, run_compile_island, vegetation_chunk_size
from .tables import GFS_V17_PROFILE_SCM_TABLE_VARIANT, PACKAGE_TABLE_VARIANT, resolve_noahmp_table_path


NOAHMP_GRAV = 9.80616
NOAHMP_SB = 5.67e-08
NOAHMP_VKC = 0.40
NOAHMP_TFRZ = 273.16
NOAHMP_HSUB = 2.8440e06
NOAHMP_HVAP = 2.5104e06
NOAHMP_CPAIR = 1004.64
NOAHMP_RAIR = 287.04
NOAHMP_RW = 461.269
NOAHMP_CWAT = 4.188e06
NOAHMP_CICE = 2.094e06
NOAHMP_TKWAT = 0.6
NOAHMP_TKICE = 2.2
NOAHMP_DENICE = 917.0
NOAHMP_DENH2O = 1000.0

CON_RD = 287.05
CON_RV = 461.50
CON_CP = 1004.6
CON_EPS = CON_RD / CON_RV
CON_EPSM1 = CON_EPS - 1.0
CON_FVIRT = CON_RV / CON_RD - 1.0
NOAHMP_EXNER_EXP = 0.2854

MPE = 1.0e-6
Z0_BARE_SOIL = 0.015
ZSOIL_4L = (-0.1, -0.4, -1.0, -2.0)
DZSOIL_4L = (-ZSOIL_4L[0],) + tuple(
    ZSOIL_4L[layer - 1] - ZSOIL_4L[layer] for layer in range(1, 4)
)


def _positive_soil_ice(delta: torch.Tensor) -> torch.Tensor:
    """MAX(total-liquid,0), with the ice-free one-sided derivative at a tie.

    Preserve clamp's signed-zero bits. The inactive branch is constant zero;
    no finite-width dead band or change to the positive-ice derivative is used.
    """
    zero = torch.zeros_like(delta)
    nonpositive = torch.where(delta <= 0.0, delta, zero)
    inactive = torch.clamp(nonpositive, min=0.0) * 0.0
    return torch.where((delta > 0.0) | torch.isnan(delta), delta, inactive)


def _pow2(value: torch.Tensor) -> torch.Tensor:
    return value * value


def _pow3(value: torch.Tensor) -> torch.Tensor:
    return value * value * value


def _pow4(value: torch.Tensor) -> torch.Tensor:
    squared = value * value
    return squared * squared


def _replace_column_tuple(columns: tuple[torch.Tensor, ...], index: int, values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(values if pos == int(index) else column for pos, column in enumerate(columns))


def _columns_to_matrix(columns: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.stack(columns, dim=1)


@dataclass(frozen=True)
class LandEnergyTables:
    """Noah-MP table fields needed by the energy/radiation/bare path."""

    table_variant: str
    isurban: int
    iswater: int
    isbarren: int
    isice: int
    z0mvt: torch.Tensor
    hvt: torch.Tensor
    hvb: torch.Tensor
    mfsno: torch.Tensor
    scffac: torch.Tensor
    cwpvt: torch.Tensor
    ch2op: torch.Tensor
    dleaf: torch.Tensor
    cbiom: torch.Tensor
    nroot: torch.Tensor
    nroot_int: torch.Tensor
    rgl: torch.Tensor
    rsmin: torch.Tensor
    hs: torch.Tensor
    topt: torch.Tensor
    rsmax: torch.Tensor
    rc: torch.Tensor
    xl: torch.Tensor
    rhol: torch.Tensor
    rhos: torch.Tensor
    taul: torch.Tensor
    taus: torch.Tensor
    albsat: torch.Tensor
    albdry: torch.Tensor
    alblak: torch.Tensor
    omegas: torch.Tensor
    betads: torch.Tensor
    betais: torch.Tensor
    eg: torch.Tensor
    ssi: torch.Tensor
    snow_ret_fac: torch.Tensor
    z0sno: torch.Tensor
    snow_emis: torch.Tensor
    swemx: torch.Tensor
    tau0: torch.Tensor
    grain_growth: torch.Tensor
    extra_growth: torch.Tensor
    dirt_soot: torch.Tensor
    bats_cosz: torch.Tensor
    bats_vis_new: torch.Tensor
    bats_nir_new: torch.Tensor
    bats_vis_age: torch.Tensor
    bats_nir_age: torch.Tensor
    bats_vis_dir: torch.Tensor
    bats_nir_dir: torch.Tensor
    rsurf_snow: torch.Tensor
    rsurf_exp: torch.Tensor
    csoil: torch.Tensor
    zbot: torch.Tensor
    smcmax: torch.Tensor
    smcref: torch.Tensor
    smcwlt: torch.Tensor
    psisat: torch.Tensor
    bexp: torch.Tensor
    quartz: torch.Tensor
    dksat: torch.Tensor
    dwsat: torch.Tensor
    timean: torch.Tensor
    fsatmx: torch.Tensor


@dataclass(frozen=True)
class SoilWaterOptRun1Result:
    """Outputs from the fixed ``opt_run=1``/``opt_inf=1`` water corridor."""

    sh2o: torch.Tensor
    smc: torch.Tensor
    runsrf: torch.Tensor
    runsub: torch.Tensor
    qin: torch.Tensor
    qdis: torch.Tensor
    wa: torch.Tensor
    wt: torch.Tensor
    zwt: torch.Tensor
    wcnd: torch.Tensor
    fcrmax: torch.Tensor


@dataclass(frozen=True)
class LandEnergyGeometry:
    """Energy-subroutine geometry and scalar factors before radiation/flux."""

    snowh: torch.Tensor
    sneqv: torch.Tensor
    fsno: torch.Tensor
    ur: torch.Tensor
    vai: torch.Tensor
    veg: torch.Tensor
    z0mg: torch.Tensor
    zpdg: torch.Tensor
    z0m: torch.Tensor
    zpd: torch.Tensor
    zlvl: torch.Tensor
    emv: torch.Tensor
    emg: torch.Tensor
    latheag: torch.Tensor
    frozen_ground: torch.Tensor
    rsurf: torch.Tensor
    rhsur: torch.Tensor
    gamma_ground: torch.Tensor


@dataclass(frozen=True)
class RadiationResult:
    """Outputs from radiation/albedo/surrad for two Noah-MP wavebands."""

    fsun: torch.Tensor
    laisun: torch.Tensor
    laisha: torch.Tensor
    parsun: torch.Tensor
    parsha: torch.Tensor
    sav: torch.Tensor
    sag: torch.Tensor
    fsr: torch.Tensor
    fsa: torch.Tensor
    fsrv: torch.Tensor
    fsrg: torch.Tensor
    albd: torch.Tensor
    albi: torch.Tensor
    albsnd: torch.Tensor
    albsni: torch.Tensor
    albgrd: torch.Tensor
    albgri: torch.Tensor
    bgap: torch.Tensor
    wgap: torch.Tensor
    tauss: torch.Tensor
    fage: torch.Tensor


@dataclass(frozen=True)
class BareFluxResult:
    """Outputs from the ``opt_sfc=3`` bare-ground flux path."""

    tgb: torch.Tensor
    cm: torch.Tensor
    ch: torch.Tensor
    ustarx: torch.Tensor
    z0h: torch.Tensor
    tauxb: torch.Tensor
    tauyb: torch.Tensor
    irb: torch.Tensor
    shb: torch.Tensor
    evb: torch.Tensor
    ghb: torch.Tensor
    t2mb: torch.Tensor
    q2b: torch.Tensor
    qsfc: torch.Tensor
    ehb2: torch.Tensor
    csigmaf0: torch.Tensor


@dataclass(frozen=True)
class BtranResult:
    """Soil water stress factors for the fixed ``opt_btr=1`` corridor."""

    btran: torch.Tensor
    btrani: torch.Tensor


@dataclass(frozen=True)
class RagrbResult:
    """Under-canopy aerodynamic resistances from ``ragrb``."""

    ramg: torch.Tensor
    rahg: torch.Tensor
    rawg: torch.Tensor
    rb: torch.Tensor
    mozg: torch.Tensor
    fhg: torch.Tensor
    fhgh: torch.Tensor


@dataclass(frozen=True)
class CanopyResistanceResult:
    """Jarvis canopy resistance result for one PAR stream."""

    rc: torch.Tensor
    psn: torch.Tensor


@dataclass(frozen=True)
class VegFluxResult:
    """Outputs from the ``opt_sfc=3`` vegetated flux path."""

    tgv: torch.Tensor
    tv: torch.Tensor
    tah: torch.Tensor
    eah: torch.Tensor
    cm: torch.Tensor
    ch: torch.Tensor
    chleaf: torch.Tensor
    chuc: torch.Tensor
    rb: torch.Tensor
    rssun: torch.Tensor
    rssha: torch.Tensor
    psnsun: torch.Tensor
    psnsha: torch.Tensor
    irg: torch.Tensor
    irc: torch.Tensor
    shg: torch.Tensor
    shc: torch.Tensor
    evg: torch.Tensor
    evc: torch.Tensor
    tr: torch.Tensor
    gh: torch.Tensor
    tauxv: torch.Tensor
    tauyv: torch.Tensor
    t2mv: torch.Tensor
    q2v: torch.Tensor
    qsfc: torch.Tensor
    canhs: torch.Tensor
    cah2: torch.Tensor
    z0h: torch.Tensor
    z0hg: torch.Tensor
    csigmaf1: torch.Tensor


@dataclass(frozen=True)
class LandEnergyPathResult:
    """Combined helper result suitable for later H assembly in land_body."""

    geometry: LandEnergyGeometry
    radiation: RadiationResult
    bare_flux: BareFluxResult


@dataclass(frozen=True)
class ThermopropResult:
    """Outputs from the fixed-corpus ``thermoprop`` source corridor."""

    df: torch.Tensor
    hcpct: torch.Tensor
    fact: torch.Tensor
    snicev: torch.Tensor
    snliqv: torch.Tensor
    epore: torch.Tensor


def _section(text: str, name: str) -> str:
    match = re.search(rf"&{re.escape(name)}(?P<body>.*?)\n/", text, re.S)
    if match is None:
        raise ValueError(f"missing noahmptable section {name}")
    return match.group("body")


def _raw_value(body: str, name: str) -> str:
    pattern = rf"^\s*{re.escape(name)}\s*=\s*(?P<value>.*?)(?=^\s*[A-Za-z_][A-Za-z0-9_]*\s*=|^\s*/|\Z)"
    match = re.search(pattern, body, re.S | re.M)
    if match is None:
        raise ValueError(f"missing noahmptable value {name}")
    value = re.sub(r"!.*", "", match.group("value"))
    return value.replace("\n", " ").strip().rstrip(",")


def _numbers(body: str, name: str) -> list[float]:
    raw = _raw_value(body, name).replace("d", "e").replace("D", "e")
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _integer(body: str, name: str) -> int:
    return int(float(_raw_value(body, name)))


# Static parsed file; traced compiles treat the result as a constant.
@lru_cache(maxsize=None)
@torch.compiler.assume_constant_result
def _parsed_table(table_variant: str) -> dict[str, Any]:
    text = resolve_noahmp_table_path(table_variant).read_text(encoding="utf-8")
    veg = _section(text, "noahmp_modis_parameters")
    rad = _section(text, "noahmp_rad_parameters")
    global_params = _section(text, "noahmp_global_parameters")
    general_params = _section(text, "noahmp_general_parameters")
    soil = _section(text, "noahmp_soil_stas_parameters")
    return {
        "isurban": _integer(veg, "isurban"),
        "iswater": _integer(veg, "iswater"),
        "isbarren": _integer(veg, "isbarren"),
        "isice": _integer(veg, "isice"),
        "z0mvt": _numbers(veg, "z0mvt"),
        "hvt": _numbers(veg, "hvt"),
        "hvb": _numbers(veg, "hvb"),
        "mfsno": _numbers(veg, "mfsno"),
        "scffac": _numbers(veg, "scffac"),
        "cwpvt": _numbers(veg, "cwpvt"),
        "ch2op": _numbers(veg, "ch2op"),
        "dleaf": _numbers(veg, "dleaf"),
        "cbiom": _numbers(veg, "cbiom"),
        "nroot": _numbers(veg, "nroot"),
        "rgl": _numbers(veg, "rgl"),
        "rsmin": _numbers(veg, "rs"),
        "hs": _numbers(veg, "hs"),
        "topt": _numbers(veg, "topt"),
        "rsmax": _numbers(veg, "rsmax"),
        "rc": _numbers(veg, "rc"),
        "xl": _numbers(veg, "xl"),
        "rhol": list(zip(_numbers(veg, "rhol_vis"), _numbers(veg, "rhol_nir"), strict=True)),
        "rhos": list(zip(_numbers(veg, "rhos_vis"), _numbers(veg, "rhos_nir"), strict=True)),
        "taul": list(zip(_numbers(veg, "taul_vis"), _numbers(veg, "taul_nir"), strict=True)),
        "taus": list(zip(_numbers(veg, "taus_vis"), _numbers(veg, "taus_nir"), strict=True)),
        "albsat": list(zip(_numbers(rad, "albsat_vis"), _numbers(rad, "albsat_nir"), strict=True)),
        "albdry": list(zip(_numbers(rad, "albdry_vis"), _numbers(rad, "albdry_nir"), strict=True)),
        "alblak": _numbers(rad, "alblak"),
        "omegas": _numbers(rad, "omegas"),
        "betads": _numbers(rad, "betads")[0],
        "betais": _numbers(rad, "betais")[0],
        "eg": _numbers(rad, "eg"),
        "ssi": _numbers(global_params, "ssi")[0],
        "snow_ret_fac": _numbers(global_params, "snow_ret_fac")[0],
        "z0sno": _numbers(global_params, "z0sno")[0],
        "snow_emis": _numbers(global_params, "snow_emis")[0],
        "swemx": _numbers(global_params, "swemx")[0],
        "tau0": _numbers(global_params, "tau0")[0],
        "grain_growth": _numbers(global_params, "grain_growth")[0],
        "extra_growth": _numbers(global_params, "extra_growth")[0],
        "dirt_soot": _numbers(global_params, "dirt_soot")[0],
        "bats_cosz": _numbers(global_params, "bats_cosz")[0],
        "bats_vis_new": _numbers(global_params, "bats_vis_new")[0],
        "bats_nir_new": _numbers(global_params, "bats_nir_new")[0],
        "bats_vis_age": _numbers(global_params, "bats_vis_age")[0],
        "bats_nir_age": _numbers(global_params, "bats_nir_age")[0],
        "bats_vis_dir": _numbers(global_params, "bats_vis_dir")[0],
        "bats_nir_dir": _numbers(global_params, "bats_nir_dir")[0],
        "rsurf_snow": _numbers(global_params, "rsurf_snow")[0],
        "rsurf_exp": _numbers(global_params, "rsurf_exp")[0],
        "csoil": _numbers(general_params, "csoil_data")[0],
        "zbot": _numbers(general_params, "zbot_data")[0],
        "smcmax": _numbers(soil, "maxsmc"),
        "smcref": _numbers(soil, "refsmc"),
        "smcwlt": _numbers(soil, "wltsmc"),
        "psisat": _numbers(soil, "satpsi"),
        "bexp": _numbers(soil, "bb"),
        "quartz": _numbers(soil, "qtz"),
        "dksat": _numbers(soil, "satdk"),
        "dwsat": _numbers(soil, "satdw"),
        "timean": _numbers(global_params, "timean")[0],
        "fsatmx": _numbers(global_params, "fsatmx")[0],
    }


def _canonical_device(device: torch.device | str) -> tuple[str, int | None]:
    dev = torch.device(device)
    index = dev.index
    if dev.type == "cuda" and index is None and torch.cuda.is_available():
        index = int(torch.cuda.current_device())
    return dev.type, index


def _device_from_key(device_type: str, device_index: int | None) -> torch.device:
    if device_index is None:
        return torch.device(device_type)
    return torch.device(f"{device_type}:{device_index}")


@lru_cache(maxsize=None)
def _cached_land_energy_tables(
    device_type: str,
    device_index: int | None,
    dtype: torch.dtype,
    table_variant: str,
) -> LandEnergyTables:
    dev = _device_from_key(device_type, device_index)
    parsed = _parsed_table(table_variant)

    def tensor(name: str) -> torch.Tensor:
        return torch.as_tensor(parsed[name], device=dev, dtype=dtype)

    def scalar(name: str) -> torch.Tensor:
        return torch.tensor(float(parsed[name]), device=dev, dtype=dtype)

    return LandEnergyTables(
        table_variant=table_variant,
        isurban=int(parsed["isurban"]),
        iswater=int(parsed["iswater"]),
        isbarren=int(parsed["isbarren"]),
        isice=int(parsed["isice"]),
        z0mvt=tensor("z0mvt"),
        hvt=tensor("hvt"),
        hvb=tensor("hvb"),
        mfsno=tensor("mfsno"),
        scffac=tensor("scffac"),
        cwpvt=tensor("cwpvt"),
        ch2op=tensor("ch2op"),
        dleaf=tensor("dleaf"),
        cbiom=tensor("cbiom"),
        nroot=tensor("nroot"),
        nroot_int=torch.as_tensor(parsed["nroot"], device=dev, dtype=torch.int64),
        rgl=tensor("rgl"),
        rsmin=tensor("rsmin"),
        hs=tensor("hs"),
        topt=tensor("topt"),
        rsmax=tensor("rsmax"),
        rc=tensor("rc"),
        xl=tensor("xl"),
        rhol=tensor("rhol"),
        rhos=tensor("rhos"),
        taul=tensor("taul"),
        taus=tensor("taus"),
        albsat=tensor("albsat"),
        albdry=tensor("albdry"),
        alblak=tensor("alblak"),
        omegas=tensor("omegas"),
        betads=scalar("betads"),
        betais=scalar("betais"),
        eg=tensor("eg"),
        ssi=scalar("ssi"),
        snow_ret_fac=scalar("snow_ret_fac"),
        z0sno=scalar("z0sno"),
        snow_emis=scalar("snow_emis"),
        swemx=scalar("swemx"),
        tau0=scalar("tau0"),
        grain_growth=scalar("grain_growth"),
        extra_growth=scalar("extra_growth"),
        dirt_soot=scalar("dirt_soot"),
        bats_cosz=scalar("bats_cosz"),
        bats_vis_new=scalar("bats_vis_new"),
        bats_nir_new=scalar("bats_nir_new"),
        bats_vis_age=scalar("bats_vis_age"),
        bats_nir_age=scalar("bats_nir_age"),
        bats_vis_dir=scalar("bats_vis_dir"),
        bats_nir_dir=scalar("bats_nir_dir"),
        rsurf_snow=scalar("rsurf_snow"),
        rsurf_exp=scalar("rsurf_exp"),
        csoil=scalar("csoil"),
        zbot=scalar("zbot"),
        smcmax=tensor("smcmax"),
        smcref=tensor("smcref"),
        smcwlt=tensor("smcwlt"),
        psisat=tensor("psisat"),
        bexp=tensor("bexp"),
        quartz=tensor("quartz"),
        dksat=tensor("dksat"),
        dwsat=tensor("dwsat"),
        timean=scalar("timean"),
        fsatmx=scalar("fsatmx"),
    )


def load_land_energy_tables(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    table_variant: str = PACKAGE_TABLE_VARIANT,
) -> LandEnergyTables:
    """Load frozen Noah-MP table fields as cached read-only tensors."""

    return _cached_land_energy_tables(*_canonical_device(device), dtype, table_variant)


def _option_check(options: Mapping[str, Any] | None) -> None:
    if options is None:
        return
    expected = {
        "idveg": 4,
        "iopt_rad": 3,
        "iopt_alb": 1,
        "iopt_sfc": 3,
        "iopt_stc": 3,
        "iopt_btr": 1,
        "iopt_run": 1,
        "iopt_trs": 2,
    }
    for key, value in expected.items():
        if key in options and int(options[key]) != value:
            raise NotImplementedError(f"land_energy.py supports {key}={value}; got {options[key]!r}")
    if "km" in options and int(options["km"]) != 4:
        raise NotImplementedError(f"land_energy.py supports km=4; got {options['km']!r}")
    if "nsnow" in options and int(options["nsnow"]) != 3:
        raise NotImplementedError(f"land_energy.py supports nsnow=3; got {options['nsnow']!r}")
    if "lsnowl" in options and int(options["lsnowl"]) != -2:
        raise NotImplementedError(f"land_energy.py supports lsnowl=-2; got {options['lsnowl']!r}")


def _as_vector(value: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and value.device == like.device and value.dtype == like.dtype:
        return value.expand_as(like)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype).expand_as(like)


def _as_like(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if value.device == like.device and value.dtype == like.dtype:
        return value
    return value.to(device=like.device, dtype=like.dtype)


def _band_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.column_stack((a, b))


def _veg_index(vegtype: torch.Tensor) -> torch.Tensor:
    return (vegtype if vegtype.dtype == torch.int64 else vegtype.to(torch.int64)) - 1


def _soil_index(soil: torch.Tensor) -> torch.Tensor:
    soil_first = soil[:, 0] if soil.ndim == 2 else soil
    return (soil_first if soil_first.dtype == torch.int64 else soil_first.to(torch.int64)) - 1


def _soil_layer_indices(soil: torch.Tensor, *, nlayer: int = 4) -> torch.Tensor:
    if soil.ndim == 2:
        sliced = soil[:, :nlayer]
        return (sliced if sliced.dtype == torch.int64 else sliced.to(torch.int64)) - 1
    expanded = soil[:, None].expand(soil.shape[0], nlayer)
    return (expanded if expanded.dtype == torch.int64 else expanded.to(torch.int64)) - 1


# noahmpdrv.F90 transfer_mp_parameters hard-codes these on every soil layer of an urban column.
URBAN_SOIL_PARAMETERS = {"smcmax": 0.45, "smcref": 0.42, "smcwlt": 0.40, "smcdry": 0.40, "csoil": 3.0e6}


def soil_parameter_by_column(tables: Any, name: str, index: torch.Tensor | None, vegtype: torch.Tensor) -> torch.Tensor:
    """``parameters%<name>``: the soil-table row (``index`` [ncol] or [ncol, nsoil]; None for a scalar
    parameter), replaced by the urban value on urban columns (vegtype isurban, 31, 32 or 33)."""
    values = getattr(tables, name)
    values = values[index] if index is not None else values.expand(vegtype.shape)
    urban = (vegtype == tables.isurban) | (vegtype == 31) | (vegtype == 32) | (vegtype == 33)
    return torch.where(urban.reshape(urban.shape + (1,) * (values.ndim - urban.ndim)), URBAN_SOIL_PARAMETERS[name], values)


def _state_cached_tensor(state: Any, name: str, fallback: Any) -> torch.Tensor:
    cached = getattr(state, name, None)
    if cached is not None:
        return cached
    return fallback() if callable(fallback) else fallback


def _safe_sign_floor(value: torch.Tensor, floor: float) -> torch.Tensor:
    return torch.where(
        torch.abs(value) < floor,
        torch.where(value >= 0.0, floor, -floor),
        value,
    )


def _mask_blend(mask: torch.Tensor, true_value: torch.Tensor, false_value: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, true_value, false_value)


def _geometry_result(values: tuple[torch.Tensor, ...]) -> LandEnergyGeometry:
    return LandEnergyGeometry(
        snowh=values[0],
        sneqv=values[1],
        fsno=values[2],
        ur=values[3],
        vai=values[4],
        veg=values[5],
        z0mg=values[6],
        zpdg=values[7],
        z0m=values[8],
        zpd=values[9],
        zlvl=values[10],
        emv=values[11],
        emg=values[12],
        latheag=values[13],
        frozen_ground=values[14],
        rsurf=values[15],
        rhsur=values[16],
        gamma_ground=values[17],
    )


def _radiation_result(values: tuple[torch.Tensor, ...]) -> RadiationResult:
    return RadiationResult(
        fsun=values[0],
        laisun=values[1],
        laisha=values[2],
        parsun=values[3],
        parsha=values[4],
        sav=values[5],
        sag=values[6],
        fsr=values[7],
        fsa=values[8],
        fsrv=values[9],
        fsrg=values[10],
        albd=values[11],
        albi=values[12],
        albsnd=values[13],
        albsni=values[14],
        albgrd=values[15],
        albgri=values[16],
        bgap=values[17],
        wgap=values[18],
        tauss=values[19],
        fage=values[20],
    )


def _energy_geometry_flat_kwargs(
    state: Any,
    tables: LandEnergyTables,
    *,
    soilcolor: torch.Tensor | None,
) -> dict[str, Any]:
    veg_idx = _state_cached_tensor(state, "veg_idx", lambda: _veg_index(state.vegtype))
    # transfer_mp_parameters uses soiltype for hydraulic properties. Soil color
    # selects albedo only; using it here silently changes surface resistance
    # whenever the two categories differ.
    soil_idx = _state_cached_tensor(state, "soil_idx", lambda: _soil_index(state.soiltyp))
    smcmax = _state_cached_tensor(state, "smcmax_surface_by_column", lambda: soil_parameter_by_column(tables, "smcmax", soil_idx, state.vegtype))
    smcwlt = _state_cached_tensor(state, "smcwlt_surface_by_column", lambda: soil_parameter_by_column(tables, "smcwlt", soil_idx, state.vegtype))
    bexp = _state_cached_tensor(state, "bexp_surface_by_column", lambda: tables.bexp[soil_idx])
    psisat = _state_cached_tensor(state, "psisat_surface_by_column", lambda: tables.psisat[soil_idx])
    ground_temperature = getattr(state, "temperature_ground", None)
    if ground_temperature is None:
        ground_temperature = state.tskin
    return {
        "temperature_forcing": state.temperature_forcing,
        "vegtype": state.vegtype,
        "snow_depth": state.snow_depth,
        "snow_water_equiv": state.snow_water_equiv,
        "u1": state.u1,
        "v1": state.v1,
        "wind": state.wind,
        "elai": state.elai,
        "esai": state.esai,
        "zf": state.zf,
        "soil_liquid_vol": state.soil_liquid_vol,
        "tskin": ground_temperature,
        "prsl1": state.prsl1,
        "mfsno": _state_cached_tensor(state, "mfsno_by_column", lambda: tables.mfsno[veg_idx]),
        "scffac": _state_cached_tensor(state, "scffac_by_column", lambda: tables.scffac[veg_idx]),
        "z0mvt": _state_cached_tensor(state, "z0mvt_by_column", lambda: tables.z0mvt[veg_idx]),
        "hvt": _state_cached_tensor(state, "hvt_by_column", lambda: tables.hvt[veg_idx]),
        "eg0": tables.eg[0],
        "z0sno": tables.z0sno,
        "snow_emis": tables.snow_emis,
        "smcmax": smcmax,
        "smcwlt": smcwlt,
        "bexp": bexp,
        "psisat": psisat,
        "rsurf_snow": tables.rsurf_snow,
        "rsurf_exp": tables.rsurf_exp,
        "isurban": int(tables.isurban),
        "z0_bare_soil": 0.002 if tables.table_variant == GFS_V17_PROFILE_SCM_TABLE_VARIANT else Z0_BARE_SOIL,
    }


def _compute_energy_geometry_flat_impl(
    *,
    temperature_forcing: torch.Tensor,
    vegtype: torch.Tensor,
    snow_depth: torch.Tensor,
    snow_water_equiv: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    wind: torch.Tensor,
    elai: torch.Tensor,
    esai: torch.Tensor,
    zf: torch.Tensor,
    soil_liquid_vol: torch.Tensor,
    tskin: torch.Tensor,
    prsl1: torch.Tensor,
    mfsno: torch.Tensor,
    scffac: torch.Tensor,
    z0mvt: torch.Tensor,
    hvt: torch.Tensor,
    eg0: torch.Tensor,
    z0sno: torch.Tensor,
    snow_emis: torch.Tensor,
    smcmax: torch.Tensor,
    smcwlt: torch.Tensor,
    bexp: torch.Tensor,
    psisat: torch.Tensor,
    rsurf_snow: torch.Tensor,
    rsurf_exp: torch.Tensor,
    isurban: int,
    z0_bare_soil: float,
) -> tuple[torch.Tensor, ...]:
    dtype = temperature_forcing.dtype
    device = temperature_forcing.device
    zsoil = static_tensor(ZSOIL_4L, device=device, dtype=dtype)

    snowh = snow_depth
    sneqv = snow_water_equiv
    snow_exists = (snowh > 1.0e-6) & (sneqv > 1.0e-3)
    zero = snowh * 0.0
    snowh = torch.where(snow_exists, snowh, zero)
    sneqv = torch.where(snow_exists, sneqv, zero)
    bdsno = torch.where(snowh > 0.0, sneqv / torch.where(snowh > 0.0, snowh, 1.0), 1.0)
    fmelt = torch.pow(bdsno / 100.0, mfsno)
    fsno = torch.where(snowh > 0.0, torch.tanh(snowh / (scffac * fmelt)), zero)

    ur = torch.clamp(torch.sqrt(u1 * u1 + v1 * v1), min=1.0)
    vai = elai + esai
    veg = vai > 0.0

    z0mg = z0_bare_soil * (1.0 - fsno) + fsno * z0sno
    zpdg = snowh
    z0m_veg = z0mvt
    zpd_veg = 0.65 * hvt
    zpd_veg = torch.where(snowh > zpd_veg, snowh, zpd_veg)
    z0m = torch.where(veg, z0m_veg, z0mg)
    zpd = torch.where(veg, zpd_veg, zpdg)

    urban = (vegtype == isurban) | (vegtype == 31) | (vegtype == 32) | (vegtype == 33)
    z0mg = torch.where(urban, z0m_veg, z0mg)
    zpdg = torch.where(urban, 0.65 * hvt, zpdg)
    z0m = torch.where(urban, z0mg, z0m)
    zpd = torch.where(urban, zpdg, zpd)

    zlvl = torch.maximum(zpd, hvt) + zf
    zlvl = torch.where(zpdg >= zlvl, zpdg + zf, zlvl)

    emv = 1.0 - torch.exp(-vai)
    emg = eg0 * (1.0 - fsno) + snow_emis * fsno

    sh2o1 = soil_liquid_vol[:, 0]
    one_minus_wetness = 1.0 - torch.minimum(torch.ones_like(sh2o1), sh2o1 / smcmax)
    l_rsurf = (-zsoil[0]) * (torch.exp(torch.pow(one_minus_wetness, rsurf_exp)) - 1.0) / (2.71828 - 1.0)
    d_rsurf = 2.2e-5 * smcmax * smcmax * torch.pow(1.0 - smcwlt / smcmax, 2.0 + 3.0 / bexp)
    rsurf_dry = l_rsurf / d_rsurf
    rsurf = 1.0 / (fsno * (1.0 / rsurf_snow) + (1.0 - fsno) * (1.0 / torch.clamp(rsurf_dry, min=0.001)))
    rsurf = torch.where((sh2o1 < 0.01) & (snowh == 0.0), 1.0e6, rsurf)
    rsurf = torch.where(urban & (snowh == 0.0), 1.0e6, rsurf)

    psi = -psisat * torch.pow(torch.clamp(sh2o1, min=0.01) / smcmax, -bexp)
    rhsur = fsno + (1.0 - fsno) * torch.exp(psi * NOAHMP_GRAV / (NOAHMP_RW * tskin))

    frozen_ground = tskin <= NOAHMP_TFRZ
    latheag = torch.where(
        frozen_ground,
        torch.full_like(tskin, NOAHMP_HSUB),
        torch.full_like(tskin, NOAHMP_HVAP),
    )
    gamma_ground = NOAHMP_CPAIR * prsl1 / (CON_EPS * latheag)

    return (
        snowh,
        sneqv,
        fsno,
        ur,
        vai,
        veg,
        z0mg,
        zpdg,
        z0m,
        zpd,
        zlvl,
        emv,
        emg,
        latheag,
        frozen_ground,
        rsurf,
        rhsur,
        gamma_ground,
    )


def esat_noahmp(t_celsius: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fortran ``esat`` polynomial over water and ice."""

    # The four Horner chains run together; each keeps the source's evaluation order.
    t = t_celsius
    coefficients = static_tensor(_ESAT_COEFFICIENTS, dtype=t.dtype, device=t.device).reshape(4, 7, *([1] * t.ndim))
    value = t * coefficients[:, 6]
    for order in range(5, -1, -1):
        value = coefficients[:, order] + value
        if order:
            value = t * value
    esw, esi, desw, desi = (100.0 * value).unbind(0)
    return esw, esi, desw, desi


_ESAT_COEFFICIENTS = (
    (6.107799961, 4.436518521e-01, 1.428945805e-02, 2.650648471e-04, 3.031240396e-06, 2.034080948e-08, 6.136820929e-11),
    (6.109177956, 5.034698970e-01, 1.886013408e-02, 4.176223716e-04, 5.824720280e-06, 4.838803174e-08, 1.838826904e-10),
    (4.438099984e-01, 2.857002636e-02, 7.938054040e-04, 1.215215065e-05, 1.036561403e-07, 3.532421810e-10, -7.090244804e-13),
    (5.030305237e-01, 3.773255020e-02, 1.267995369e-03, 2.477563108e-05, 3.005693132e-07, 2.158542548e-09, 7.131097725e-12),
)


def split_shortwave(swdown: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``solad`` and ``solai`` using Noah-MP's fixed VIS/NIR split."""

    return _band_pair(swdown * 0.35, swdown * 0.35), _band_pair(swdown * 0.15, swdown * 0.15)


def snow_age_bats(
    *,
    dt: torch.Tensor,
    tg: torch.Tensor,
    sneqvo: torch.Tensor,
    sneqv: torch.Tensor,
    tauss: torch.Tensor,
    tau0: torch.Tensor,
    grain_growth: torch.Tensor,
    extra_growth: torch.Tensor,
    dirt_soot: torch.Tensor,
    swemx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fortran ``snow_age`` for the BATS albedo option."""

    old_tauss = tauss
    dela0 = dt / tau0
    arg = grain_growth * (1.0 / NOAHMP_TFRZ - 1.0 / tg)
    age1 = torch.exp(arg)
    age2 = torch.exp(torch.clamp(extra_growth * arg, max=0.0))
    tage = age1 + age2 + dirt_soot
    dela = dela0 * tage
    dels = torch.clamp(sneqv - sneqvo, min=0.0) / swemx
    sge = (old_tauss + dela) * (1.0 - dels)
    new_tauss = torch.where(sneqv <= 0.0, 0.0, torch.clamp(sge, min=0.0))
    fage = new_tauss / (new_tauss + 1.0)
    return new_tauss, fage


def snowalb_bats(
    *,
    cosz: torch.Tensor,
    fage: torch.Tensor,
    bats_cosz: torch.Tensor,
    bats_vis_new: torch.Tensor,
    bats_nir_new: torch.Tensor,
    bats_vis_age: torch.Tensor,
    bats_nir_age: torch.Tensor,
    bats_vis_dir: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fortran ``snowalb_bats`` for two bands."""

    sl1 = 1.0 / bats_cosz
    sl2 = 2.0 * bats_cosz
    cf1 = ((1.0 + sl1) / (1.0 + sl2 * cosz) - sl1)
    fzen = torch.clamp(cf1, min=0.0)
    albsni_vis = bats_vis_new * (1.0 - bats_vis_age * fage)
    albsni_nir = bats_nir_new * (1.0 - bats_nir_age * fage)
    albsni = _band_pair(albsni_vis, albsni_nir)
    albsnd = _band_pair(
        albsni_vis + bats_vis_dir * fzen * (1.0 - albsni_vis),
        albsni_nir + bats_vis_dir * fzen * (1.0 - albsni_nir),
    )
    return albsnd, albsni


def ground_albedo(
    *,
    albsat: torch.Tensor,
    albdry: torch.Tensor,
    smc: torch.Tensor,
    fsno: torch.Tensor,
    albsnd: torch.Tensor,
    albsni: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fortran ``groundalb`` for the sealed soil-surface corridor."""

    inc = torch.clamp(0.11 - 0.40 * smc[:, 0], min=0.0)
    albsod = torch.minimum(albsat + inc[:, None], albdry)
    albsoi = albsod
    albgrd = albsod * (1.0 - fsno[:, None]) + albsnd * fsno[:, None]
    albgri = albsoi * (1.0 - fsno[:, None]) + albsni * fsno[:, None]
    return albgrd, albgri


def _twostream(
    *,
    cosz: torch.Tensor,
    vai: torch.Tensor,
    fwet: torch.Tensor,
    t: torch.Tensor,
    albgrd: torch.Tensor,
    albgri: torch.Tensor,
    rho: torch.Tensor,
    tau: torch.Tensor,
    fveg: torch.Tensor,
    xl: torch.Tensor,
    omegas: torch.Tensor,
    betads: torch.Tensor,
    betais: torch.Tensor,
    diffuse: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = cosz.shape[0]
    one = torch.ones((n, 2), device=cosz.device, dtype=cosz.dtype)
    cosz2 = cosz[:, None]
    vai2 = vai[:, None]
    fwet2 = fwet[:, None]
    fveg2 = fveg[:, None]
    t2 = t[:, None]

    gap = torch.where(vai2 == 0.0, one, one * (1.0 - fveg2))
    kopen = gap
    vai2 = torch.where(vai2 > 0.0, vai2, 1.0)

    coszi = torch.clamp(cosz2, min=0.001)
    chil = torch.clamp(xl[:, None], min=-0.4, max=0.6)
    chil = torch.where(torch.abs(chil) <= 0.01, 0.01, chil)
    phi1 = 0.5 - 0.633 * chil - 0.330 * chil * chil
    phi2 = 0.877 * (1.0 - 2.0 * phi1)
    gdir = phi1 + phi2 * coszi
    ext = gdir / coszi
    avmu = (1.0 - phi1 / phi2 * torch.log((phi1 + phi2) / phi1)) / phi2
    omegal = rho + tau
    tmp0 = gdir + phi2 * coszi
    tmp1 = phi1 * coszi
    asu = 0.5 * omegal * gdir / tmp0 * (1.0 - tmp1 / tmp0 * torch.log((tmp1 + tmp0) / tmp1))
    betadl = (1.0 + avmu * ext) / (omegal * avmu * ext) * asu
    chil_factor = (1.0 + chil) / 2.0
    betail = 0.5 * (rho + tau + (rho - tau) * _pow2(chil_factor)) / omegal

    warm = t2 > NOAHMP_TFRZ
    omega = torch.where(warm, omegal, (1.0 - fwet2) * omegal + fwet2 * omegas[None, :])
    betad = torch.where(
        warm,
        betadl,
        ((1.0 - fwet2) * omegal * betadl + fwet2 * omegas[None, :] * betads) / omega,
    )
    betai = torch.where(
        warm,
        betail,
        ((1.0 - fwet2) * omegal * betail + fwet2 * omegas[None, :] * betais) / omega,
    )

    b = 1.0 - omega + omega * betai
    c = omega * betai
    tmp0 = avmu * ext
    d = tmp0 * omega * betad
    f = tmp0 * omega * (1.0 - betad)
    tmp1 = b * b - c * c
    h = torch.sqrt(torch.clamp(tmp1, min=1.0e-30)) / avmu
    sigma = _safe_sign_floor(tmp0 * tmp0 - tmp1, 1.0e-6)
    p1 = b + avmu * h
    p2 = b - avmu * h
    p3 = b + tmp0
    p4 = b - tmp0
    s1 = torch.exp(-h * vai2)
    s2 = torch.exp(-ext * vai2)
    alb = albgri if diffuse else albgrd
    u1 = b - c / alb
    u2 = b - c * alb
    u3 = f + c * alb
    tmp2 = u1 - avmu * h
    tmp3 = u1 + avmu * h
    d1 = p1 * tmp2 / s1 - p2 * tmp3 * s1
    tmp4 = u2 + avmu * h
    tmp5 = u2 - avmu * h
    d2 = tmp4 / s1 - tmp5 * s1
    h1 = -d * p4 - c * f
    tmp6 = d - h1 * p3 / sigma
    tmp7 = (d - c - h1 / sigma * (u1 + tmp0)) * s2
    h2 = (tmp6 * tmp2 / s1 - p2 * tmp7) / d1
    h3 = -(tmp6 * tmp3 * s1 - p1 * tmp7) / d1
    h4 = -f * p3 - c * d
    tmp8 = h4 / sigma
    tmp9 = (u3 - tmp8 * (u2 - tmp0)) * s2
    h5 = -(tmp8 * tmp4 / s1 + tmp9) / d2
    h6 = (tmp8 * tmp5 * s1 + tmp9) / d2
    h7 = (c * tmp2) / (d1 * s1)
    h8 = (-c * tmp3 * s1) / d1
    h9 = tmp4 / (d2 * s1)
    h10 = (-tmp5 * s1) / d2

    if diffuse:
        ftd = alb * 0.0
        fti = (h9 * s1 + h10 / s1) * (1.0 - kopen) + kopen
        fre = (h7 + h8) * (1.0 - kopen) + albgri * kopen
        frev = fre
        freg = fre * 0.0
    else:
        ftd = s2 * (1.0 - gap) + gap
        fti = (h4 * s2 / sigma + h5 * s1 + h6 / s1) * (1.0 - gap)
        fre = (h1 / sigma + h2 + h3) * (1.0 - gap) + albgrd * gap
        frev = (h1 / sigma + h2 + h3) * (1.0 - gap)
        freg = albgrd * gap
    fab = 1.0 - fre - (1.0 - albgrd) * ftd - (1.0 - albgri) * fti
    bgap = gap[:, 0]
    wgap = bgap * 0.0
    return fab, fre, ftd, fti, gdir[:, 0], frev, freg, bgap, wgap


def _radiation_flat_kwargs(
    state: Any,
    tables: LandEnergyTables,
    *,
    geometry: LandEnergyGeometry,
    dt: torch.Tensor | float,
    soilcolor: torch.Tensor | None = None,
    tauss: torch.Tensor | None = None,
    sneqvo: torch.Tensor | None = None,
) -> dict[str, Any]:
    veg_idx = _state_cached_tensor(state, "veg_idx", lambda: _veg_index(state.vegtype))
    if soilcolor is None:
        soil_idx = _state_cached_tensor(state, "soil_idx", lambda: _soil_index(state.soiltyp))
        albsat = _state_cached_tensor(state, "albsat_by_column", lambda: tables.albsat[soil_idx])
        albdry = _state_cached_tensor(state, "albdry_by_column", lambda: tables.albdry[soil_idx])
    elif soilcolor is getattr(state, "soilcol", None):
        soil_idx = _state_cached_tensor(state, "soilcol_idx", lambda: _soil_index(soilcolor))
        albsat = _state_cached_tensor(state, "albsat_soilcol_by_column", lambda: tables.albsat[soil_idx])
        albdry = _state_cached_tensor(state, "albdry_soilcol_by_column", lambda: tables.albdry[soil_idx])
    else:
        soil_idx = _soil_index(soilcolor)
        albsat = tables.albsat[soil_idx]
        albdry = tables.albdry[soil_idx]
    fwet_state = getattr(state, "fwet", None)
    if fwet_state is None:
        fwet_state = state.temperature_forcing * 0.0
    if tauss is None:
        tauss = getattr(state, "snow_age", None)
        if tauss is None:
            tauss = state.temperature_forcing * 0.0
    if sneqvo is None:
        sneqvo = getattr(state, "snow_water_equiv_old", None)
    ground_temperature = getattr(state, "temperature_ground", None)
    if ground_temperature is None:
        ground_temperature = state.tskin
    leaf_temperature = getattr(state, "temperature_leaf", None)
    if leaf_temperature is None:
        leaf_temperature = state.tskin
    return {
        "temperature_forcing": state.temperature_forcing,
        "dt_tensor": _as_vector(dt, state.temperature_forcing),
        "tskin": ground_temperature,
        "temperature_leaf": leaf_temperature,
        "xcoszin": state.xcoszin,
        "soil_moisture_vol": state.soil_moisture_vol,
        "elai": state.elai,
        "esai": state.esai,
        "fveg": state.fveg,
        "swdown": state.swdown,
        "fwet": fwet_state,
        "sneqv": geometry.sneqv,
        "fsno": geometry.fsno,
        "vai": geometry.vai,
        "tauss": tauss,
        "sneqvo": geometry.sneqv if sneqvo is None else sneqvo,
        "albsat": albsat,
        "albdry": albdry,
        "xl": _state_cached_tensor(state, "xl_by_column", lambda: tables.xl[veg_idx]),
        "rhol": _state_cached_tensor(state, "rhol_by_column", lambda: tables.rhol[veg_idx]),
        "rhos": _state_cached_tensor(state, "rhos_by_column", lambda: tables.rhos[veg_idx]),
        "taul": _state_cached_tensor(state, "taul_by_column", lambda: tables.taul[veg_idx]),
        "taus": _state_cached_tensor(state, "taus_by_column", lambda: tables.taus[veg_idx]),
        "tau0": tables.tau0,
        "grain_growth": tables.grain_growth,
        "extra_growth": tables.extra_growth,
        "dirt_soot": tables.dirt_soot,
        "swemx": tables.swemx,
        "bats_cosz": tables.bats_cosz,
        "bats_vis_new": tables.bats_vis_new,
        "bats_nir_new": tables.bats_nir_new,
        "bats_vis_age": tables.bats_vis_age,
        "bats_nir_age": tables.bats_nir_age,
        "bats_vis_dir": tables.bats_vis_dir,
        "omegas": tables.omegas,
        "betads": tables.betads,
        "betais": tables.betais,
    }


def _radiation_bats_two_stream_flat_impl(
    *,
    temperature_forcing: torch.Tensor,
    dt_tensor: torch.Tensor,
    tskin: torch.Tensor,
    temperature_leaf: torch.Tensor,
    xcoszin: torch.Tensor,
    soil_moisture_vol: torch.Tensor,
    elai: torch.Tensor,
    esai: torch.Tensor,
    fveg: torch.Tensor,
    swdown: torch.Tensor,
    fwet: torch.Tensor,
    sneqv: torch.Tensor,
    fsno: torch.Tensor,
    vai: torch.Tensor,
    tauss: torch.Tensor,
    sneqvo: torch.Tensor,
    albsat: torch.Tensor,
    albdry: torch.Tensor,
    xl: torch.Tensor,
    rhol: torch.Tensor,
    rhos: torch.Tensor,
    taul: torch.Tensor,
    taus: torch.Tensor,
    tau0: torch.Tensor,
    grain_growth: torch.Tensor,
    extra_growth: torch.Tensor,
    dirt_soot: torch.Tensor,
    swemx: torch.Tensor,
    bats_cosz: torch.Tensor,
    bats_vis_new: torch.Tensor,
    bats_nir_new: torch.Tensor,
    bats_vis_age: torch.Tensor,
    bats_nir_age: torch.Tensor,
    bats_vis_dir: torch.Tensor,
    omegas: torch.Tensor,
    betads: torch.Tensor,
    betais: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Flat tensor/literal radiation kernel for ``opt_rad=3,opt_alb=1``."""

    dtype = temperature_forcing.dtype
    tauss_new, fage = snow_age_bats(
        dt=dt_tensor,
        tg=tskin,
        sneqvo=sneqvo,
        sneqv=sneqv,
        tauss=tauss,
        tau0=tau0,
        grain_growth=grain_growth,
        extra_growth=extra_growth,
        dirt_soot=dirt_soot,
        swemx=swemx,
    )

    _ = dtype
    zero = temperature_forcing * 0.0
    day = xcoszin > 0.0
    cosz_safe = torch.clamp(xcoszin, min=1.0e-12)
    albsnd_day, albsni_day = snowalb_bats(
        cosz=cosz_safe,
        fage=fage,
        bats_cosz=bats_cosz,
        bats_vis_new=bats_vis_new,
        bats_nir_new=bats_nir_new,
        bats_vis_age=bats_vis_age,
        bats_nir_age=bats_nir_age,
        bats_vis_dir=bats_vis_dir,
    )
    albsnd = torch.where(day[:, None], albsnd_day, 0.0)
    albsni = torch.where(day[:, None], albsni_day, 0.0)
    albgrd_day, albgri_day = ground_albedo(
        albsat=albsat,
        albdry=albdry,
        smc=soil_moisture_vol,
        fsno=fsno,
        albsnd=albsnd,
        albsni=albsni,
    )
    albgrd = torch.where(day[:, None], albgrd_day, 0.0)
    albgri = torch.where(day[:, None], albgri_day, 0.0)

    wl = elai / torch.clamp(vai, min=MPE)
    ws = esai / torch.clamp(vai, min=MPE)
    rho = torch.clamp(rhol * wl[:, None] + rhos * ws[:, None], min=MPE)
    tau = torch.clamp(taul * wl[:, None] + taus * ws[:, None], min=MPE)
    albgrd_stream = torch.where(day[:, None], albgrd, 0.2)
    albgri_stream = torch.where(day[:, None], albgri, 0.2)

    fabd, albd, ftdd, ftid, gdir, frevd, fregd, bgap, wgap = _twostream(
        cosz=cosz_safe,
        vai=vai,
        fwet=fwet,
        t=temperature_leaf,
        albgrd=albgrd_stream,
        albgri=albgri_stream,
        rho=rho,
        tau=tau,
        fveg=fveg,
        xl=xl,
        omegas=omegas,
        betads=betads,
        betais=betais,
        diffuse=False,
    )
    fabi, albi, _ftdi, ftii, _gdir_i, frevi, fregi, _bgap_i, _wgap_i = _twostream(
        cosz=cosz_safe,
        vai=vai,
        fwet=fwet,
        t=temperature_leaf,
        albgrd=albgrd_stream,
        albgri=albgri_stream,
        rho=rho,
        tau=tau,
        fveg=fveg,
        xl=xl,
        omegas=omegas,
        betads=betads,
        betais=betais,
        diffuse=True,
    )
    fabd = torch.where(day[:, None], fabd, 0.0)
    fabi = torch.where(day[:, None], fabi, 0.0)
    ftdd = torch.where(day[:, None], ftdd, 0.0)
    ftid = torch.where(day[:, None], ftid, 0.0)
    ftii = torch.where(day[:, None], ftii, 0.0)
    albd = torch.where(day[:, None], albd, 0.0)
    albi = torch.where(day[:, None], albi, 0.0)
    frevd = torch.where(day[:, None], frevd, 0.0)
    frevi = torch.where(day[:, None], frevi, 0.0)
    fregd = torch.where(day[:, None], fregd, 0.0)
    fregi = torch.where(day[:, None], fregi, 0.0)
    bgap = torch.where(day, bgap, zero)
    wgap = torch.where(day, wgap, zero)

    ext = gdir / torch.clamp(xcoszin, min=MPE) * torch.sqrt(torch.clamp(1.0 - rho[:, 0] - tau[:, 0], min=0.0))
    fsun_raw = (1.0 - torch.exp(-ext * vai)) / torch.clamp(ext * vai, min=MPE)
    fsun = torch.where((day) & (fsun_raw >= 0.01), fsun_raw, zero)
    fsha = 1.0 - fsun
    laisun = elai * fsun
    laisha = elai * fsha
    solad, solai = split_shortwave(swdown)

    cad = solad * fabd
    cai = solai * fabi
    trd = solad * ftdd
    tri = solad * ftid + solai * ftii
    ground_abs = trd * (1.0 - albgrd) + tri * (1.0 - albgri)
    sav = torch.zeros_like(swdown)
    sag = torch.zeros_like(swdown)
    fsa = torch.zeros_like(swdown)
    for band in range(2):
        sav = sav + cad[:, band] + cai[:, band]
        fsa = fsa + cad[:, band] + cai[:, band]
        sag = sag + ground_abs[:, band]
        fsa = fsa + ground_abs[:, band]

    laifra = elai / torch.clamp(vai, min=MPE)
    parsun = torch.where(
        fsun > 0.0,
        (cad[:, 0] + fsun * cai[:, 0]) * laifra / torch.clamp(laisun, min=MPE),
        zero,
    )
    parsha = torch.where(
        fsun > 0.0,
        (fsha * cai[:, 0]) * laifra / torch.clamp(laisha, min=MPE),
        (cad[:, 0] + cai[:, 0]) * laifra / torch.clamp(laisha, min=MPE),
    )
    rvis = albd[:, 0] * solad[:, 0] + albi[:, 0] * solai[:, 0]
    rnir = albd[:, 1] * solad[:, 1] + albi[:, 1] * solai[:, 1]
    fsr = rvis + rnir
    fsrv = frevd[:, 0] * solad[:, 0] + frevi[:, 0] * solai[:, 0] + frevd[:, 1] * solad[:, 1] + frevi[:, 1] * solai[:, 1]
    fsrg = fregd[:, 0] * solad[:, 0] + fregi[:, 0] * solai[:, 0] + fregd[:, 1] * solad[:, 1] + fregi[:, 1] * solai[:, 1]

    return (
        fsun,
        laisun,
        laisha,
        parsun,
        parsha,
        sav,
        sag,
        fsr,
        fsa,
        fsrv,
        fsrg,
        albd,
        albi,
        albsnd,
        albsni,
        albgrd,
        albgri,
        bgap,
        wgap,
        tauss_new,
        fage,
    )


def _tdfcnd_flat(
    *,
    smcmax: torch.Tensor,
    quartz: torch.Tensor,
    smc: torch.Tensor,
    sh2o: torch.Tensor,
) -> torch.Tensor:
    satratio = smc / smcmax
    thks = torch.pow(7.7, quartz) * torch.pow(2.0, 1.0 - quartz)
    xunfroz = torch.where(smc > 0.0, sh2o / torch.where(smc > 0.0, smc, 1.0), 1.0)
    xu = xunfroz * smcmax
    thksat = torch.pow(thks, 1.0 - smcmax) * torch.pow(NOAHMP_TKICE, smcmax - xu) * torch.pow(0.57, xu)
    gammd = (1.0 - smcmax) * 2700.0
    thkdry = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)
    frozen = (sh2o + 0.0005) < smc
    ake_unfrozen = torch.where(satratio > 0.1, torch.log10(torch.where(satratio > 0.1, satratio, 1.0)) + 1.0, 0.0)
    ake = torch.where(frozen, satratio, ake_unfrozen)
    return ake * (thksat - thkdry) + thkdry


def _soil_carbon_by_veg_index(veg_idx: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
    soil_carbon = static_tensor(
        [
            [90, 65, 90, 65, 90, 40, 50, 50, 40, 50, 90, 60, 60, 60, 0, 20, 0, 90, 90, 60],
            [40, 30, 40, 30, 40, 25, 30, 30, 25, 30, 40, 30, 30, 30, 0, 15, 0, 60, 60, 40],
            [20, 15, 20, 15, 20, 15, 20, 15, 15, 15, 25, 20, 20, 20, 0, 10, 0, 40, 40, 30],
            [15, 10, 15, 10, 15, 10, 15, 10, 10, 10, 20, 10, 10, 10, 0, 10, 0, 40, 30, 20],
        ],
        device=like.device,
        dtype=like.dtype,
    ) / 130.0
    return soil_carbon[:, veg_idx].T


def _thermoprop_flat_kwargs(
    state: Any,
    tables: LandEnergyTables,
    geometry: LandEnergyGeometry,
    *,
    dt: torch.Tensor | float,
    shdfac: torch.Tensor,
) -> dict[str, torch.Tensor]:
    soil_layers = _state_cached_tensor(state, "soil_layer_idx", lambda: _soil_layer_indices(state.soiltyp, nlayer=4))
    veg_idx = _state_cached_tensor(state, "veg_idx", lambda: _veg_index(state.vegtype))
    urban = (state.vegtype == tables.isurban) | (state.vegtype == 31) | (state.vegtype == 32) | (state.vegtype == 33)
    selected_gfs_v17 = tables.table_variant == GFS_V17_PROFILE_SCM_TABLE_VARIANT
    dz_snow = state.dzsnso[:, :3]
    snow_mass = state.snow_level_ice[:, :3] + state.snow_level_liquid[:, :3]
    if selected_gfs_v17:
        dz_safe = torch.where(torch.abs(dz_snow) > 1.0e-30, dz_snow, torch.ones_like(dz_snow))
        bulk_density = snow_mass / dz_safe
        snow_conductivity = 2.576e-6 * bulk_density * bulk_density + 0.074
        soil_carbon_layers = torch.zeros_like(state.soil_moisture_vol)
    else:
        snow_conductivity = torch.full_like(dz_snow, 0.35)
        soil_carbon_layers = _soil_carbon_by_veg_index(veg_idx, like=state.temperature_forcing)
    return {
        "temperature_forcing": state.temperature_forcing,
        "snow_levels": state.snow_levels,
        "dzsnso": state.dzsnso,
        "snow_level_ice": state.snow_level_ice,
        "snow_level_liquid": state.snow_level_liquid,
        "soil_moisture_vol": state.soil_moisture_vol,
        "soil_liquid_vol": state.soil_liquid_vol,
        "snowh": geometry.snowh,
        "dt_tensor": _as_vector(dt, state.temperature_forcing),
        "shdfac": shdfac,
        "smcmax_layers": _state_cached_tensor(state, "smcmax_by_column", lambda: soil_parameter_by_column(tables, "smcmax", soil_layers, state.vegtype)),
        "quartz_layers": _state_cached_tensor(state, "quartz_by_column", lambda: tables.quartz[soil_layers]),
        "snow_conductivity": snow_conductivity,
        "soil_carbon_layers": soil_carbon_layers,
        "csoil": soil_parameter_by_column(tables, "csoil", None, state.vegtype)[:, None],
        "urban": urban,
    }


def _thermoprop_result(values: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> ThermopropResult:
    df, hcpct, fact, snicev, snliqv, epore = values
    return ThermopropResult(
        df=df,
        hcpct=hcpct,
        fact=fact,
        snicev=snicev,
        snliqv=snliqv,
        epore=epore,
    )


def _thermoprop_source_corridor_flat_impl(
    *,
    temperature_forcing: torch.Tensor,
    snow_levels: torch.Tensor,
    dzsnso: torch.Tensor,
    snow_level_ice: torch.Tensor,
    snow_level_liquid: torch.Tensor,
    soil_moisture_vol: torch.Tensor,
    soil_liquid_vol: torch.Tensor,
    snowh: torch.Tensor,
    dt_tensor: torch.Tensor,
    shdfac: torch.Tensor,
    smcmax_layers: torch.Tensor,
    quartz_layers: torch.Tensor,
    snow_conductivity: torch.Tensor,
    soil_carbon_layers: torch.Tensor,
    csoil: torch.Tensor,
    urban: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zero3 = dzsnso[:, :3] * 0.0
    snow_idx = torch.arange(-2, 1, device=snow_levels.device, dtype=snow_levels.dtype)
    active_snow = snow_idx[None, :] >= (snow_levels + 1)[:, None]
    dz_snow = dzsnso[:, :3]
    dz_safe = torch.where(torch.abs(dz_snow) > 1.0e-30, dz_snow, 1.0)
    ice_vol = torch.minimum(dz_snow * 0.0 + 1.0, snow_level_ice[:, :3] / (dz_safe * NOAHMP_DENICE))
    pore = 1.0 - ice_vol
    liq_vol = torch.minimum(pore, snow_level_liquid[:, :3] / (dz_safe * NOAHMP_DENH2O))
    active_snow_real = active_snow.to(temperature_forcing.dtype)
    snicev = ice_vol * active_snow_real
    snliqv = liq_vol * active_snow_real
    epore = pore * active_snow_real
    df_snow = snow_conductivity * active_snow_real
    hcpct_snow = (NOAHMP_CICE * ice_vol + NOAHMP_CWAT * liq_vol) * active_snow_real

    sice = soil_moisture_vol - soil_liquid_vol
    hcpct_soil = (
        soil_liquid_vol * NOAHMP_CWAT
        + (1.0 - smcmax_layers) * csoil
        + (smcmax_layers - soil_moisture_vol) * NOAHMP_CPAIR
        + sice * NOAHMP_CICE
    )
    df_soil = _tdfcnd_flat(smcmax=smcmax_layers, quartz=quartz_layers, smc=soil_moisture_vol, sh2o=soil_liquid_vol)
    df_soil = (1.0 - soil_carbon_layers) * df_soil + soil_carbon_layers * 0.25
    df_soil = torch.where(urban[:, None], 3.24, df_soil)
    soil_damping = torch.column_stack(
        (
            torch.exp(-2.0 * shdfac),
            shdfac * 0.0 + 1.0,
            shdfac * 0.0 + 1.0,
            shdfac * 0.0 + 1.0,
        )
    )
    df_soil = df_soil * soil_damping

    df = torch.column_stack(
        (
            df_snow[:, 0],
            df_snow[:, 1],
            df_snow[:, 2],
            df_soil[:, 0],
            df_soil[:, 1],
            df_soil[:, 2],
            df_soil[:, 3],
        )
    )
    hcpct = torch.column_stack(
        (
            hcpct_snow[:, 0],
            hcpct_snow[:, 1],
            hcpct_snow[:, 2],
            hcpct_soil[:, 0],
            hcpct_soil[:, 1],
            hcpct_soil[:, 2],
            hcpct_soil[:, 3],
        )
    )
    no_snow = snow_levels == 0
    blended_no_snow = (df[:, 3] * dzsnso[:, 3] + 0.35 * snowh) / (snowh + dzsnso[:, 3])
    blended_snow = (df[:, 3] * dzsnso[:, 3] + df[:, 2] * dzsnso[:, 2]) / (dzsnso[:, 2] + dzsnso[:, 3])
    df3 = torch.where(no_snow, blended_no_snow, blended_snow)
    df = torch.column_stack((df[:, 0], df[:, 1], df[:, 2], df3, df[:, 4], df[:, 5], df[:, 6]))

    layers = torch.arange(7, device=snow_levels.device, dtype=snow_levels.dtype)
    participates = layers[None, :] >= (snow_levels + 3)[:, None]
    fact_denom = hcpct * dzsnso
    fact_denom_safe = torch.where(participates & (torch.abs(fact_denom) > 1.0e-30), fact_denom, 1.0)
    fact = torch.where(participates, dt_tensor[:, None] / fact_denom_safe, fact_denom * 0.0)

    return (df, hcpct, fact, snicev, snliqv, epore)


def thermoprop_source_corridor(
    state: Any,
    tables: LandEnergyTables,
    geometry: LandEnergyGeometry,
    *,
    dt: torch.Tensor | float,
    shdfac: torch.Tensor,
) -> ThermopropResult:
    """Port ``thermoprop``/``csnow``/``tdfcnd`` for the sealed land corridor."""

    return _thermoprop_result(_thermoprop_source_corridor_flat_impl(**_thermoprop_flat_kwargs(state, tables, geometry, dt=dt, shdfac=shdfac)))


def tsnosoi_source_corridor(
    state: Any,
    tables: LandEnergyTables,
    thermo: ThermopropResult,
    *,
    ssoil: torch.Tensor,
    dt: torch.Tensor | float,
) -> torch.Tensor:
    """Port ``tsnosoi``/``hrt``/``hstep``/``rosr12`` for sealed opt_stc=3 cases."""

    return _tsnosoi_source_corridor_flat_impl(
        temperature_snow_soil=state.temperature_snow_soil,
        interface_depth=state.interface_depth,
        df=thermo.df,
        hcpct=thermo.hcpct,
        temperature_forcing=state.temperature_forcing,
        snow_levels=state.snow_levels,
        tg3=state.tg3,
        snow_depth=state.snow_depth,
        zbot=tables.zbot,
        ssoil=ssoil,
        dt=dt,
    )


def _tsnosoi_source_corridor_flat_impl(
    *,
    temperature_snow_soil: torch.Tensor,
    interface_depth: torch.Tensor,
    df: torch.Tensor,
    hcpct: torch.Tensor,
    temperature_forcing: torch.Tensor,
    snow_levels: torch.Tensor,
    tg3: torch.Tensor,
    snow_depth: torch.Tensor,
    zbot: torch.Tensor,
    ssoil: torch.Tensor,
    dt: torch.Tensor | float,
) -> torch.Tensor:
    """Flat tensor/literal ``tsnosoi`` kernel used by the scoped compile path."""

    stc = temperature_snow_soil
    zsnso = interface_depth
    dt_tensor = _as_vector(dt, temperature_forcing)
    ntop = snow_levels + 3
    tbot = tg3
    zbot_work = _as_like(zbot, stc) - snow_depth

    zero = stc[:, 0] * 0.0
    zero5 = stc[:, 1:6] * 0.0
    layer5 = torch.arange(1, 6, device=stc.device, dtype=ntop.dtype)
    active5 = layer5[None, :] >= ntop[:, None]
    top5 = layer5[None, :] == ntop[:, None]
    middle5 = active5 & (~top5)
    # Fortran evaluates gradients only in the active top/middle layer branch.
    temp1_zero = torch.where(ntop == 0, -zsnso[:, 1], 1.0)
    ddz0 = torch.where(ntop == 0, 2.0 / temp1_zero, zero)
    dtsdz0 = torch.where(ntop == 0, 2.0 * (stc[:, 0] - stc[:, 1]) / temp1_zero, zero)
    denom0 = torch.where(ntop == 0, -zsnso[:, 0] * hcpct[:, 0], zero)
    eflux0 = torch.where(ntop == 0, df[:, 0] * dtsdz0 - ssoil, zero)

    temp1_top = torch.where(top5, -zsnso[:, 2:7], 1.0)
    ddz_top = 2.0 / temp1_top
    dtsdz_top = 2.0 * (stc[:, 1:6] - stc[:, 2:7]) / temp1_top
    denom_top = -zsnso[:, 1:6] * hcpct[:, 1:6]
    eflux_top = df[:, 1:6] * dtsdz_top - ssoil[:, None]

    temp1_mid = torch.where(middle5, zsnso[:, 0:5] - zsnso[:, 2:7], 1.0)
    ddz_mid = 2.0 / temp1_mid
    dtsdz_mid = 2.0 * (stc[:, 1:6] - stc[:, 2:7]) / temp1_mid
    dtsdz5 = torch.where(top5, dtsdz_top, torch.where(middle5, dtsdz_mid, zero5))
    previous_dtsdz5 = torch.column_stack((dtsdz0, dtsdz5[:, 0], dtsdz5[:, 1], dtsdz5[:, 2], dtsdz5[:, 3]))
    denom_mid = (zsnso[:, 0:5] - zsnso[:, 1:6]) * hcpct[:, 1:6]
    eflux_mid = df[:, 1:6] * dtsdz_mid - df[:, 0:5] * previous_dtsdz5

    ddz5 = torch.where(top5, ddz_top, torch.where(middle5, ddz_mid, zero5))
    denom5 = torch.where(top5, denom_top, torch.where(middle5, denom_mid, zero5))
    eflux5 = torch.where(top5, eflux_top, torch.where(middle5, eflux_mid, zero5))

    active6 = 6 >= ntop
    denom6 = (zsnso[:, 5] - zsnso[:, 6]) * hcpct[:, 6]
    dtsdz6 = (stc[:, 6] - tbot) / (0.5 * (zsnso[:, 5] + zsnso[:, 6]) - zbot_work)
    botflx = -df[:, 6] * dtsdz6
    eflux6 = -botflx - df[:, 5] * dtsdz5[:, 4]
    active6_real = active6.to(stc.dtype)
    denom6 = denom6 * active6_real
    eflux6 = eflux6 * active6_real

    ddz = torch.column_stack((ddz0, ddz5[:, 0], ddz5[:, 1], ddz5[:, 2], ddz5[:, 3], ddz5[:, 4], zero))
    denom = torch.column_stack((denom0, denom5[:, 0], denom5[:, 1], denom5[:, 2], denom5[:, 3], denom5[:, 4], denom6))
    eflux = torch.column_stack((eflux0, eflux5[:, 0], eflux5[:, 1], eflux5[:, 2], eflux5[:, 3], eflux5[:, 4], eflux6))

    layers = torch.arange(7, device=stc.device, dtype=ntop.dtype)
    active = layers[None, :] >= ntop[:, None]
    top = layers[None, :] == ntop[:, None]
    middle = active & (~top) & (layers[None, :] != 6)
    bottom = active & (layers[None, :] == 6)
    denom_safe = torch.where(torch.abs(denom) > 1.0e-30, denom, 1.0)
    df_prev = torch.column_stack((zero, df[:, 0], df[:, 1], df[:, 2], df[:, 3], df[:, 4], df[:, 5]))
    ddz_prev = torch.column_stack((zero, ddz[:, 0], ddz[:, 1], ddz[:, 2], ddz[:, 3], ddz[:, 4], ddz[:, 5]))

    ai_mid = -df_prev * ddz_prev / denom_safe
    ci_mid = -df * ddz / denom_safe
    middle_or_bottom = middle | bottom
    ai = ai_mid * middle_or_bottom.to(stc.dtype)
    ci = ci_mid * (top | middle).to(stc.dtype)
    bi = (-ci) * top.to(stc.dtype) + (-(ai + ci)) * middle_or_bottom.to(stc.dtype)
    rhsts = eflux / (-denom_safe) * active.to(stc.dtype)

    dt_col = dt_tensor[:, None]
    rhsts = torch.where(active, rhsts * dt_col, rhsts)
    ai = torch.where(active, ai * dt_col, ai)
    bi = torch.where(active, 1.0 + bi * dt_col, bi)
    ci = torch.where(active, ci * dt_col, ci)

    rhstsin = rhsts
    ciin = ci
    b0_safe = torch.where(torch.abs(bi[:, 0]) > 1.0e-30, bi[:, 0], 1.0)
    c0 = torch.where(ntop == 0, -ciin[:, 0] / b0_safe, zero)
    r0 = torch.where(ntop == 0, rhstsin[:, 0] / b0_safe, zero)
    a1, a2, a3, a4, a5, a6 = ai[:, 1], ai[:, 2], ai[:, 3], ai[:, 4], ai[:, 5], ai[:, 6]
    b1, b2, b3, b4, b5, b6 = bi[:, 1], bi[:, 2], bi[:, 3], bi[:, 4], bi[:, 5], bi[:, 6]
    c1, c2, c3, c4, c5, c6 = ci[:, 1], ci[:, 2], ci[:, 3], ci[:, 4], ci[:, 5], ci[:, 6]
    r1, r2, r3, r4, r5, r6 = rhsts[:, 1], rhsts[:, 2], rhsts[:, 3], rhsts[:, 4], rhsts[:, 5], rhsts[:, 6]
    cin1, cin2, cin3, cin4, cin5, cin6 = ciin[:, 1], ciin[:, 2], ciin[:, 3], ciin[:, 4], ciin[:, 5], ciin[:, 6]
    rin1, rin2, rin3, rin4, rin5, rin6 = rhstsin[:, 1], rhstsin[:, 2], rhstsin[:, 3], rhstsin[:, 4], rhstsin[:, 5], rhstsin[:, 6]

    b1_safe = torch.where(torch.abs(b1) > 1.0e-30, b1, 1.0)
    c1 = torch.where(ntop == 1, -cin1 / b1_safe, c1)
    r1 = torch.where(ntop == 1, rin1 / b1_safe, r1)
    solve = b1 + a1 * c0
    solve_safe = torch.where(torch.abs(solve) > 1.0e-30, solve, 1.0)
    # rosr12 multiplies by the reciprocal after the first active row.
    inverse_solve = torch.div(1.0, solve_safe)
    c1 = torch.where(ntop == 0, -cin1 * inverse_solve, c1)
    r1 = torch.where(ntop == 0, (rin1 - a1 * r0) * inverse_solve, r1)

    b2_safe = torch.where(torch.abs(b2) > 1.0e-30, b2, 1.0)
    c2 = torch.where(ntop == 2, -cin2 / b2_safe, c2)
    r2 = torch.where(ntop == 2, rin2 / b2_safe, r2)
    solve = b2 + a2 * c1
    solve_safe = torch.where(torch.abs(solve) > 1.0e-30, solve, 1.0)
    inverse_solve = torch.div(1.0, solve_safe)
    c2 = torch.where(2 > ntop, -cin2 * inverse_solve, c2)
    r2 = torch.where(2 > ntop, (rin2 - a2 * r1) * inverse_solve, r2)

    b3_safe = torch.where(torch.abs(b3) > 1.0e-30, b3, 1.0)
    c3 = torch.where(ntop == 3, -cin3 / b3_safe, c3)
    r3 = torch.where(ntop == 3, rin3 / b3_safe, r3)
    solve = b3 + a3 * c2
    solve_safe = torch.where(torch.abs(solve) > 1.0e-30, solve, 1.0)
    inverse_solve = torch.div(1.0, solve_safe)
    c3 = torch.where(3 > ntop, -cin3 * inverse_solve, c3)
    r3 = torch.where(3 > ntop, (rin3 - a3 * r2) * inverse_solve, r3)

    solve = b4 + a4 * c3
    solve_safe = torch.where(torch.abs(solve) > 1.0e-30, solve, 1.0)
    inverse_solve = torch.div(1.0, solve_safe)
    c4 = torch.where(4 > ntop, -cin4 * inverse_solve, c4)
    r4 = torch.where(4 > ntop, (rin4 - a4 * r3) * inverse_solve, r4)

    solve = b5 + a5 * c4
    solve_safe = torch.where(torch.abs(solve) > 1.0e-30, solve, 1.0)
    inverse_solve = torch.div(1.0, solve_safe)
    c5 = torch.where(5 > ntop, -cin5 * inverse_solve, c5)
    r5 = torch.where(5 > ntop, (rin5 - a5 * r4) * inverse_solve, r5)

    solve = b6 + a6 * c5
    solve_safe = torch.where(torch.abs(solve) > 1.0e-30, solve, 1.0)
    inverse_solve = torch.div(1.0, solve_safe)
    c6 = torch.where(6 > ntop, -cin6 * inverse_solve, c6)
    r6 = torch.where(6 > ntop, (rin6 - a6 * r5) * inverse_solve, r6)

    x6 = r6
    x5 = torch.where(5 >= ntop, c5 * x6 + r5, c5)
    x4 = torch.where(4 >= ntop, c4 * x5 + r4, c4)
    x3 = torch.where(3 >= ntop, c3 * x4 + r3, c3)
    x2 = torch.where(2 >= ntop, c2 * x3 + r2, c2)
    x1 = torch.where(1 >= ntop, c1 * x2 + r1, c1)
    x0 = c0 * x1 + r0

    return _columns_to_matrix(
        (
            torch.where(ntop == 0, stc[:, 0] + x0, stc[:, 0]),
            torch.where(1 >= ntop, stc[:, 1] + x1, stc[:, 1]),
            torch.where(2 >= ntop, stc[:, 2] + x2, stc[:, 2]),
            torch.where(3 >= ntop, stc[:, 3] + x3, stc[:, 3]),
            torch.where(4 >= ntop, stc[:, 4] + x4, stc[:, 4]),
            torch.where(5 >= ntop, stc[:, 5] + x5, stc[:, 5]),
            torch.where(6 >= ntop, stc[:, 6] + x6, stc[:, 6]),
        )
    )


def _soilwater_srt_opt_run1_inf1(
    *,
    sh2o: torch.Tensor,
    smc: torch.Tensor,
    sice: torch.Tensor,
    smcmax: torch.Tensor,
    bexp: torch.Tensor,
    dksat: torch.Tensor,
    dwsat: torch.Tensor,
    fcr: torch.Tensor,
    pddum: torch.Tensor,
    qseva: torch.Tensor,
    etrani: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized ``srt`` corridor for ``opt_run=1`` and ``opt_inf=1``."""

    factr = torch.clamp(smc / smcmax, min=0.01)
    wdf = dwsat * working_power(factr, bexp + 2.0) * (1.0 - fcr)
    wcnd = dksat * working_power(factr, 2.0 * bexp + 3.0) * (1.0 - fcr)

    denom0 = pddum * 0.0 + DZSOIL_4L[0]
    temp0 = -ZSOIL_4L[1]
    ddz0 = pddum * 0.0 + 2.0 / temp0
    dsmdz0 = 2.0 * (smc[:, 0] - smc[:, 1]) / temp0
    wflux0 = wdf[:, 0] * dsmdz0 + wcnd[:, 0] - pddum + etrani[:, 0] + qseva

    denom1 = pddum * 0.0 + DZSOIL_4L[1]
    temp1 = ZSOIL_4L[0] - ZSOIL_4L[2]
    ddz1 = pddum * 0.0 + 2.0 / temp1
    dsmdz1 = 2.0 * (smc[:, 1] - smc[:, 2]) / temp1
    wflux1 = wdf[:, 1] * dsmdz1 + wcnd[:, 1] - wdf[:, 0] * dsmdz0 - wcnd[:, 0] + etrani[:, 1]

    denom2 = pddum * 0.0 + DZSOIL_4L[2]
    temp2 = ZSOIL_4L[1] - ZSOIL_4L[3]
    ddz2 = pddum * 0.0 + 2.0 / temp2
    dsmdz2 = 2.0 * (smc[:, 2] - smc[:, 3]) / temp2
    wflux2 = wdf[:, 2] * dsmdz2 + wcnd[:, 2] - wdf[:, 1] * dsmdz1 - wcnd[:, 1] + etrani[:, 2]

    denom3 = pddum * 0.0 + DZSOIL_4L[3]
    qdrain = pddum * 0.0
    wflux3 = -(wdf[:, 2] * dsmdz2) - wcnd[:, 2] + etrani[:, 3] + qdrain

    zero = pddum * 0.0
    ai0 = zero
    bi0 = wdf[:, 0] * ddz0 / denom0
    ci0 = -bi0
    ai1 = -wdf[:, 0] * ddz0 / denom1
    ci1 = -wdf[:, 1] * ddz1 / denom1
    bi1 = -(ai1 + ci1)
    ai2 = -wdf[:, 1] * ddz1 / denom2
    ci2 = -wdf[:, 2] * ddz2 / denom2
    bi2 = -(ai2 + ci2)
    ai3 = -wdf[:, 2] * ddz2 / denom3
    ci3 = zero
    bi3 = -(ai3 + ci3)
    ai = _columns_to_matrix((ai0, ai1, ai2, ai3))
    bi = _columns_to_matrix((bi0, bi1, bi2, bi3))
    ci = _columns_to_matrix((ci0, ci1, ci2, ci3))
    rhstt = _columns_to_matrix((wflux0 / (-denom0), wflux1 / (-denom1), wflux2 / (-denom2), wflux3 / (-denom3)))
    _ = sice
    return rhstt, ai, bi, ci, qdrain, wcnd


def _soilwater_rosr12_four_layer(
    ai: torch.Tensor,
    bi: torch.Tensor,
    ci: torch.Tensor,
    rhstt: torch.Tensor,
) -> torch.Tensor:
    """Fortran ``rosr12`` specialized to four soil layers and ``ntop=1``."""

    p0 = -ci[:, 0] / bi[:, 0]
    d0 = rhstt[:, 0] / bi[:, 0]
    denom1 = bi[:, 1] + ai[:, 1] * p0
    # rosr12 divides only the first row; subsequent rows multiply a reciprocal.
    inverse1 = torch.reciprocal(denom1)
    p1 = -ci[:, 1] * inverse1
    d1 = (rhstt[:, 1] - ai[:, 1] * d0) * inverse1
    denom2 = bi[:, 2] + ai[:, 2] * p1
    inverse2 = torch.reciprocal(denom2)
    p2 = -ci[:, 2] * inverse2
    d2 = (rhstt[:, 2] - ai[:, 2] * d1) * inverse2
    denom3 = bi[:, 3] + ai[:, 3] * p2
    inverse3 = torch.reciprocal(denom3)
    d3 = (rhstt[:, 3] - ai[:, 3] * d2) * inverse3
    x3 = d3
    x2 = p2 * x3 + d2
    x1 = p1 * x2 + d1
    x0 = p0 * x1 + d0
    return _columns_to_matrix((x0, x1, x2, x3))


def _soilwater_sstep_opt_run1(
    *,
    sh2o: torch.Tensor,
    smc: torch.Tensor,
    sice: torch.Tensor,
    smcmax: torch.Tensor,
    dzsnso_soil: torch.Tensor,
    ai: torch.Tensor,
    bi: torch.Tensor,
    ci: torch.Tensor,
    rhstt: torch.Tensor,
    dtfine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized ``sstep`` corridor for four-layer ``opt_run=1`` soil water."""

    rhstt_scaled = rhstt * dtfine[:, None]
    ai_scaled = ai * dtfine[:, None]
    bi_scaled = 1.0 + bi * dtfine[:, None]
    ci_scaled = ci * dtfine[:, None]
    delta = _soilwater_rosr12_four_layer(ai_scaled, bi_scaled, ci_scaled, rhstt_scaled)
    sh2o_trial = sh2o + delta
    h0, h1, h2, h3 = sh2o_trial[:, 0], sh2o_trial[:, 1], sh2o_trial[:, 2], sh2o_trial[:, 3]

    col3 = h3
    epore = torch.clamp(smcmax[:, 3] - sice[:, 3], min=1.0e-4)
    wplus = torch.clamp(col3 - epore, min=0.0) * dzsnso_soil[:, 3]
    h3 = torch.minimum(epore, col3)
    h2 = h2 + wplus / dzsnso_soil[:, 2]

    col2 = h2
    epore = torch.clamp(smcmax[:, 2] - sice[:, 2], min=1.0e-4)
    wplus = torch.clamp(col2 - epore, min=0.0) * dzsnso_soil[:, 2]
    h2 = torch.minimum(epore, col2)
    h1 = h1 + wplus / dzsnso_soil[:, 1]

    col1 = h1
    epore = torch.clamp(smcmax[:, 1] - sice[:, 1], min=1.0e-4)
    wplus = torch.clamp(col1 - epore, min=0.0) * dzsnso_soil[:, 1]
    h1 = torch.minimum(epore, col1)
    h0 = h0 + wplus / dzsnso_soil[:, 0]

    col0 = h0
    epore_top = torch.clamp(smcmax[:, 0] - sice[:, 0], min=1.0e-4)
    wplus_top = torch.clamp(col0 - epore_top, min=0.0) * dzsnso_soil[:, 0]
    h0 = torch.minimum(epore_top, col0)

    top_excess = wplus_top > 0.0
    d0 = h0
    d1 = h1 + wplus_top / dzsnso_soil[:, 1]
    d2 = h2
    d3 = h3

    col1_down = d1
    epore = torch.clamp(smcmax[:, 1] - sice[:, 1], min=1.0e-4)
    wplus_down = torch.clamp(col1_down - epore, min=0.0) * dzsnso_soil[:, 1]
    d1 = torch.minimum(epore, col1_down)
    d2 = d2 + wplus_down / dzsnso_soil[:, 2]

    col2_down = d2
    epore = torch.clamp(smcmax[:, 2] - sice[:, 2], min=1.0e-4)
    wplus_down = torch.clamp(col2_down - epore, min=0.0) * dzsnso_soil[:, 2]
    d2 = torch.minimum(epore, col2_down)
    d3 = d3 + wplus_down / dzsnso_soil[:, 3]

    col3_down = d3
    epore_bottom = torch.clamp(smcmax[:, 3] - sice[:, 3], min=1.0e-4)
    wplus_bottom = torch.clamp(col3_down - epore_bottom, min=0.0) * dzsnso_soil[:, 3]
    d3 = torch.minimum(epore_bottom, col3_down)

    sh2o_next = _columns_to_matrix((h0, h1, h2, h3))
    sh2o_down = _columns_to_matrix((d0, d1, d2, d3))
    sh2o_next = torch.where(top_excess[:, None], sh2o_down, sh2o_next)
    wplus = torch.where(top_excess, wplus_bottom, wplus_top)
    smc_next = sh2o_next + sice
    _ = smc
    return sh2o_next, smc_next, wplus


def _groundwater_opt_run1_source_corridor(
    *,
    sh2o: torch.Tensor,
    sice: torch.Tensor,
    smcmax: torch.Tensor,
    psisat: torch.Tensor,
    bexp: torch.Tensor,
    dksat: torch.Tensor,
    wcnd: torch.Tensor,
    fcrmax: torch.Tensor,
    timean: torch.Tensor,
    dt: torch.Tensor,
    zwt_initial: torch.Tensor,
    wa_initial: torch.Tensor,
    wt_initial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized ``groundwater`` source corridor used by the default Noah-MP run."""

    device = sh2o.device
    dtype = sh2o.dtype
    dzmm = static_tensor(tuple(1000.0 * dz for dz in DZSOIL_4L), device=device, dtype=dtype)
    znode = static_tensor(
        (-ZSOIL_4L[0] / 2.0,) + tuple(
            -ZSOIL_4L[layer - 1] + 0.5 * DZSOIL_4L[layer] for layer in range(1, 4)
        ), device=device, dtype=dtype,
    )
    epore = torch.clamp(smcmax - sice, min=0.01)
    smc = sh2o + sice
    mliq = sh2o * dzmm
    hk = 1000.0 * wcnd

    # Match the source's first layer-bottom crossing, including equality.
    iwt = torch.full_like(zwt_initial, 3, dtype=torch.int64)
    for layer in range(3, 0, -1):
        iwt = torch.where(zwt_initial <= -ZSOIL_4L[layer], layer - 1, iwt)
    index = iwt[:, None]
    bexp_node = torch.gather(bexp, 1, index).squeeze(1)
    hk_node = torch.gather(hk, 1, index).squeeze(1)
    node_depth = znode[iwt]
    fff = bexp_node / 3.0
    rsbmx = hk_node * 1000.0 * 20.085536923187668
    qdis = (1.0 - fcrmax) * rsbmx * torch.exp(-timean) * torch.exp(-fff * zwt_initial)

    s_node = torch.clamp(
        torch.gather(smc, 1, index).squeeze(1)
        / torch.gather(smcmax, 1, index).squeeze(1), min=0.01, max=1.0,
    )
    smpfz = -torch.gather(psisat, 1, index).squeeze(1) * 1000.0 * torch.pow(s_node, -bexp_node)
    smpfz = torch.clamp(0.80 * smpfz, min=-120000.0)
    ka = 0.5 * (hk_node + torch.gather(dksat, 1, index).squeeze(1) * 1000.0)
    wh_zwt = -zwt_initial * 1000.0
    wh = smpfz - node_depth * 1000.0
    qin = -ka * (wh_zwt - wh) / ((zwt_initial - node_depth) * 1000.0)
    recharge_limit = torch.full_like(dt, 10.0) / dt
    qin = torch.clamp(qin, min=-recharge_limit, max=recharge_limit)

    wt = wt_initial + (qin - qdis) * dt
    aquifer = iwt == 3
    wa_deep = wa_initial + (qin - qdis) * dt
    zwt_deep = (-ZSOIL_4L[3] + 25.0) - wa_deep / 1000.0 / 0.2
    deep_bottom = mliq[:, 3] - qin * dt + torch.clamp(wa_deep - 5000.0, min=0.0)
    wt = torch.where(aquifer, wa_deep, wt)
    wa = torch.where(aquifer, torch.clamp(wa_deep, max=5000.0), wa_initial)

    next_layer = torch.clamp(iwt + 1, max=3)
    storage_below = torch.zeros_like(dt)
    wtsub = torch.zeros_like(dt)
    for layer in range(4):
        storage_below = storage_below + torch.where(
            layer > iwt + 1, epore[:, layer] * dzmm[layer], 0.0,
        )
        wtsub = wtsub + hk[:, layer] * dzmm[layer]
    storage_change = wt - 0.2 * 1000.0 * 25.0
    storage_change = torch.where(iwt == 2, storage_change, storage_change - storage_below)
    next_depth = static_tensor(tuple(-z for z in ZSOIL_4L), device=device, dtype=dtype)[next_layer]
    next_porosity = torch.gather(epore, 1, next_layer[:, None]).squeeze(1)
    zwt_shallow = next_depth - storage_change / next_porosity / 1000.0
    zwt = torch.where(aquifer, zwt_deep, zwt_shallow)

    # Float64 quotient rounded once (no-op in float64): float32 a/b**2 underflows at the floor in backward.
    removal = ((qdis[:, None] * dt[:, None] * hk * dzmm).double() / torch.clamp(wtsub[:, None], min=1.0e-300 if wtsub.dtype == torch.float64 else float(torch.finfo(wtsub.dtype).tiny)).double()).to(wtsub.dtype)
    deep_mliq = torch.cat((mliq[:, :3], deep_bottom[:, None]), dim=1)
    mliq = torch.where(aquifer[:, None], deep_mliq, mliq - removal)
    zwt = torch.clamp(zwt, min=1.5)

    watmin = wa * 0.0 + 0.01
    mliq0, mliq1, mliq2, mliq3 = mliq[:, 0], mliq[:, 1], mliq[:, 2], mliq[:, 3]
    xs = torch.where(mliq0 < 0.0, watmin - mliq0, 0.0)
    mliq0 = mliq0 + xs
    mliq1 = mliq1 - xs
    xs = torch.where(mliq1 < 0.0, watmin - mliq1, 0.0)
    mliq1 = mliq1 + xs
    mliq2 = mliq2 - xs
    xs = torch.where(mliq2 < 0.0, watmin - mliq2, 0.0)
    mliq2 = mliq2 + xs
    mliq3 = mliq3 - xs
    xs = torch.where(mliq3 < watmin, watmin - mliq3, 0.0)
    mliq3 = mliq3 + xs
    wa = wa - xs
    wt = wt - xs

    mliq = _columns_to_matrix((mliq0, mliq1, mliq2, mliq3))
    sh2o_next = mliq / dzmm
    return sh2o_next, qin, qdis, wa, wt, zwt, hk


def _soilwater_opt_run1_inf1_source_corridor_flat_impl(
    *,
    vegtype: torch.Tensor,
    smcmax: torch.Tensor,
    psisat: torch.Tensor,
    bexp: torch.Tensor,
    dksat: torch.Tensor,
    dwsat: torch.Tensor,
    fsatmx: torch.Tensor,
    isurban: int,
    timean: torch.Tensor,
    dzsnso_soil: torch.Tensor,
    smc: torch.Tensor,
    sh2o: torch.Tensor,
    qinsur: torch.Tensor,
    qseva: torch.Tensor,
    etrani: torch.Tensor,
    dt: torch.Tensor,
    zwt: torch.Tensor,
    wa: torch.Tensor,
    wt: torch.Tensor,
    iteration_count: int = 6,
    sice_initial: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Port ``soilwater`` + ``groundwater`` for sealed ``opt_run=1`` cases.

    This is intentionally narrow: it covers the fixed four-soil-layer,
    ``opt_inf=1``/``opt_run=1`` source-smoke corridor used by the bare/no-canopy
    checkpoint cases.  It does not implement alternate runoff, infiltration, lake,
    crop, or dynamic water-table options.
    """

    dt_tensor = dt
    sice = _positive_soil_ice(smc - sh2o) if sice_initial is None else sice_initial
    epore = torch.clamp(smcmax - sice, min=1.0e-4)
    # soilwater accumulates rsat layer by layer (k = 1..nsoil). torch.sum adds in another,
    # vector-width dependent order on CPU, and a compiled sum adds in this order.
    excess = torch.clamp(sh2o - epore, min=0.0) * dzsnso_soil
    rsat = ((excess[:, 0] + excess[:, 1]) + excess[:, 2]) + excess[:, 3]
    sh2o = torch.minimum(epore, sh2o)

    fice = torch.clamp(sice / smcmax, max=1.0)
    exp_minus_a = 0.01831563888873418
    fcr = torch.clamp(torch.exp(-4.0 * (1.0 - fice)) - exp_minus_a, min=0.0) / (1.0 - exp_minus_a)
    urban = (vegtype == isurban) | (vegtype == 31) | (vegtype == 32) | (vegtype == 33)
    fcrmax = torch.max(fcr, dim=1).values  # soilwater takes the maximum before the urban impermeable override
    fcr = _columns_to_matrix((torch.where(urban, 0.95, fcr[:, 0]), fcr[:, 1], fcr[:, 2], fcr[:, 3]))

    fff = bexp[:, 0] / 3.0
    fsat = fsatmx * torch.exp(-0.5 * fff * zwt)
    positive_input = qinsur > 0.0
    runsrf = torch.where(positive_input, qinsur * ((1.0 - fcr[:, 0]) * fsat + fcr[:, 0]), 0.0)
    pddum = torch.where(positive_input, qinsur - runsrf, 0.0)
    niter = torch.where(pddum * dt_tensor > dzsnso_soil[:, 0] * smcmax[:, 0], 6.0, 3.0)
    dtfine = dt_tensor / niter

    qdrain_save = dt_tensor * 0.0
    runsrf_save = dt_tensor * 0.0
    wcnd_last = smc * 0.0
    if iteration_count not in {3, 6}:
        raise ValueError(f"soilwater iteration_count must be 3 or 6 for the sealed opt_run=1 corridor; got {iteration_count}")
    for iteration in range(iteration_count):
        active = iteration < niter
        rhstt, ai, bi, ci, qdrain, wcnd = _soilwater_srt_opt_run1_inf1(
            sh2o=sh2o,
            smc=smc,
            sice=sice,
            smcmax=smcmax,
            bexp=bexp,
            dksat=dksat,
            dwsat=dwsat,
            fcr=fcr,
            pddum=pddum,
            qseva=qseva,
            etrani=etrani,
        )
        sh2o_next, smc_next, wplus = _soilwater_sstep_opt_run1(
            sh2o=sh2o,
            smc=smc,
            sice=sice,
            smcmax=smcmax,
            dzsnso_soil=dzsnso_soil,
            ai=ai,
            bi=bi,
            ci=ci,
            rhstt=rhstt,
            dtfine=dtfine,
        )
        sh2o = torch.where(active[:, None], sh2o_next, sh2o)
        smc = torch.where(active[:, None], smc_next, smc)
        wcnd_last = torch.where(active[:, None], wcnd, wcnd_last)
        rsat = rsat + torch.where(active, wplus, 0.0)
        qdrain_save = qdrain_save + torch.where(active, qdrain, 0.0)
        runsrf_save = runsrf_save + torch.where(active, runsrf, 0.0)

    qdrain = qdrain_save / niter
    runsrf_mm = runsrf_save / niter * 1000.0 + rsat * 1000.0 / dt_tensor
    _ = qdrain * 1000.0
    sh2o, qin, qdis, wa, wt, zwt, _hk = _groundwater_opt_run1_source_corridor(
        sh2o=sh2o,
        sice=sice,
        smcmax=smcmax,
        psisat=psisat,
        bexp=bexp,
        dksat=dksat,
        wcnd=wcnd_last,
        fcrmax=fcrmax,
        timean=timean,
        dt=dt_tensor,
        zwt_initial=zwt,
        wa_initial=wa,
        wt_initial=wt,
    )
    smc = sh2o + sice
    return (
        sh2o,
        smc,
        runsrf_mm,
        qdis,
        qin,
        qdis,
        wa,
        wt,
        zwt,
        wcnd_last,
        fcrmax,
    )


def soilwater_opt_run1_inf1_source_corridor(
    state: Any,
    tables: LandEnergyTables,
    *,
    qinsur: torch.Tensor,
    qseva: torch.Tensor,
    etrani: torch.Tensor,
    dt: torch.Tensor | float,
    zwt_initial: torch.Tensor | float = 2.0,
    wa_initial: torch.Tensor | float = 4900.0,
    wt_initial: torch.Tensor | float = 500.0,
    smc_initial: torch.Tensor | None = None,
    sh2o_initial: torch.Tensor | None = None,
    dzsnso_soil_initial: torch.Tensor | None = None,
    iteration_count: int = 6,
    sice_initial: torch.Tensor | None = None,
) -> SoilWaterOptRun1Result:
    """Port ``soilwater`` + ``groundwater`` for sealed ``opt_run=1`` cases."""

    dt_tensor = _as_vector(dt, state.temperature_forcing)
    zwt = _as_vector(zwt_initial, state.temperature_forcing)
    wa = _as_vector(wa_initial, state.temperature_forcing)
    wt = _as_vector(wt_initial, state.temperature_forcing)
    soil_layers = _state_cached_tensor(state, "soil_layer_idx", lambda: _soil_layer_indices(state.soiltyp, nlayer=4))
    dzsnso_soil = torch.abs(state.dzsnso[:, 3:7]) if dzsnso_soil_initial is None else torch.abs(dzsnso_soil_initial)
    smc = state.soil_moisture_vol if smc_initial is None else smc_initial
    sh2o = state.soil_liquid_vol if sh2o_initial is None else sh2o_initial
    result = run_compile_island(
        "land_soilwater",
        "soilwater_opt_run1_inf1_source_corridor",
        _soilwater_opt_run1_inf1_source_corridor_flat_impl,
        vegtype=state.vegtype,
        smcmax=_state_cached_tensor(state, "smcmax_by_column", lambda: soil_parameter_by_column(tables, "smcmax", soil_layers, state.vegtype)),
        psisat=_state_cached_tensor(state, "psisat_by_column", lambda: tables.psisat[soil_layers]),
        bexp=_state_cached_tensor(state, "bexp_by_column", lambda: tables.bexp[soil_layers]),
        dksat=_state_cached_tensor(state, "dksat_by_column", lambda: tables.dksat[soil_layers]),
        dwsat=_state_cached_tensor(state, "dwsat_by_column", lambda: tables.dwsat[soil_layers]),
        fsatmx=tables.fsatmx,
        isurban=tables.isurban,
        timean=tables.timean,
        dzsnso_soil=dzsnso_soil,
        smc=smc,
        sh2o=sh2o,
        qinsur=qinsur,
        qseva=qseva,
        etrani=etrani,
        dt=dt_tensor,
        zwt=zwt,
        wa=wa,
        wt=wt,
        iteration_count=int(iteration_count),
        sice_initial=sice_initial,
    )
    return SoilWaterOptRun1Result(
        sh2o=result[0],
        smc=result[1],
        runsrf=result[2],
        runsub=result[3],
        qin=result[4],
        qdis=result[5],
        wa=result[6],
        wt=result[7],
        zwt=result[8],
        wcnd=result[9],
        fcrmax=result[10],
    )


def btran_opt_btr1_source_corridor(state: Any, tables: LandEnergyTables) -> BtranResult:
    """Port the Noah soil-moisture stress factor used by ``opt_btr=1``."""

    soil_layers = _state_cached_tensor(state, "soil_layer_idx", lambda: _soil_layer_indices(state.soiltyp, nlayer=4))
    veg_idx = _state_cached_tensor(state, "veg_idx", lambda: _veg_index(state.vegtype))
    nroot = _state_cached_tensor(state, "nroot_int_by_column", lambda: tables.nroot_int[veg_idx])
    nroot_clamped = torch.clamp(nroot, min=1, max=4)
    root_depths = static_tensor((-ZSOIL_4L[0], -ZSOIL_4L[1], -ZSOIL_4L[2], -ZSOIL_4L[3]), device=state.temperature_forcing.device, dtype=state.temperature_forcing.dtype)
    zroot = root_depths[nroot_clamped - 1]
    dz_soil = state.dzsnso[:, 3:7]
    smcwlt = _state_cached_tensor(state, "smcwlt_by_column", lambda: soil_parameter_by_column(tables, "smcwlt", soil_layers, state.vegtype))
    smcref = _state_cached_tensor(state, "smcref_by_column", lambda: soil_parameter_by_column(tables, "smcref", soil_layers, state.vegtype))
    gx = (state.soil_liquid_vol - smcwlt) / (smcref - smcwlt)
    gx = torch.clamp(gx, min=0.0, max=1.0)
    btrani_raw = torch.clamp(dz_soil / zroot[:, None] * gx, min=MPE)
    layer_index = torch.arange(4, device=state.temperature_forcing.device, dtype=torch.int64)[None, :]
    active = layer_index < nroot[:, None]
    btrani = torch.where(active, btrani_raw, 0.0)
    # The source accumulates root layers in order before normalizing each layer.
    btran = torch.zeros_like(btrani[:, 0])
    for k in range(4):
        btran = btran + btrani[:, k]
    btran = torch.clamp(btran, min=MPE)
    btrani = torch.where(active, btrani / btran[:, None], 0.0)
    return BtranResult(btran=btran, btrani=btrani)


def _thermalz0_bare_chen09_impl(
    *,
    z0mg: torch.Tensor,
    zlvl: torch.Tensor,
    zpd: torch.Tensor,
    ustarx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fortran ``thermalz0`` bare-flag branch for ``opt_trs=2``."""

    z0m_out = z0mg
    reyn = ustarx * z0m_out / 1.5e-5
    kb_sigma_f0 = torch.where(
        reyn > 2.0,
        2.46 * working_power(torch.where(reyn > 2.0, reyn, 4.0), 0.25) - torch.log(torch.full_like(reyn, 7.4)),
        -torch.log(torch.full_like(reyn, 0.397)),
    )
    z0h_out = torch.clamp(z0m_out / torch.exp(kb_sigma_f0), min=1.0e-6)
    csigmaf0 = torch.log((zlvl - zpd) / z0m_out) * (torch.log((zlvl - zpd) / z0m_out) + kb_sigma_f0)
    return z0m_out, z0h_out, csigmaf0


def thermalz0_bare_chen09(
    *,
    z0mg: torch.Tensor,
    zlvl: torch.Tensor,
    zpd: torch.Tensor,
    ustarx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return run_compile_island(
        "land_stability",
        "thermalz0_bare_chen09",
        _thermalz0_bare_chen09_impl,
        z0mg=z0mg,
        zlvl=zlvl,
        zpd=zpd,
        ustarx=ustarx,
    )


def thermalz0_vegetated_chen09(
    tables: LandEnergyTables,
    *,
    vegtype: torch.Tensor,
    z0m: torch.Tensor,
    ustarx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fortran ``thermalz0`` vegetated-flag branch for ``opt_trs=2``."""

    veg_idx = _veg_index(vegtype)
    czil = torch.pow(torch.full_like(z0m, 10.0), -0.4 * tables.hvt[veg_idx])
    z0h = z0m * torch.exp(-czil * 0.4 * 258.2 * torch.sqrt(ustarx * z0m))
    return z0m, z0h, torch.zeros_like(z0m)


def calhum_noahmp(sfctmp: torch.Tensor, sfcprs: torch.Tensor) -> torch.Tensor:
    """Return ``calhum`` saturated mixing ratio in kg/kg."""

    es = 0.611 * torch.exp(2.501e6 / 461.0 * (1.0 / 273.15 - 1.0 / sfctmp))
    sfcprsx = sfcprs * 1.0e-3
    q2sat = 0.622 * es / (sfcprsx - es)
    q2sat = q2sat * 1.0e3
    return q2sat / 1.0e3


def canres_jarvis_source_corridor(
    tables: LandEnergyTables,
    *,
    vegtype: torch.Tensor,
    par: torch.Tensor,
    sfctmp: torch.Tensor,
    rcsoil: torch.Tensor,
    eah: torch.Tensor,
    sfcprs: torch.Tensor,
) -> CanopyResistanceResult:
    """Port ``canres`` for the sealed Jarvis ``opt_crs=2`` path."""

    veg_idx = _veg_index(vegtype)
    q2 = CON_EPS * eah / (sfcprs + CON_EPSM1 * eah)
    q2 = q2 / (1.0 - q2)
    q2sat = calhum_noahmp(sfctmp, sfcprs)
    ff = 2.0 * par / tables.rgl[veg_idx]
    rcs = torch.clamp((ff + tables.rsmin[veg_idx] / tables.rsmax[veg_idx]) / (1.0 + ff), min=0.0001)
    rct = torch.clamp(1.0 - 0.0016 * _pow2(tables.topt[veg_idx] - sfctmp), min=0.0001)
    rcq = torch.maximum(
        1.0 / (1.0 + tables.hs[veg_idx] * torch.clamp(q2sat - q2, min=0.0)),
        par * 0.0 + 0.01,
    )
    rc = tables.rsmin[veg_idx] / (rcs * rct * rcq * rcsoil)
    psn = par * 0.0 - 999.99
    return CanopyResistanceResult(rc=rc, psn=psn)


def ragrb_source_corridor(
    tables: LandEnergyTables,
    *,
    iter_number: int,
    vegtype: torch.Tensor,
    vai: torch.Tensor,
    rhoair: torch.Tensor,
    hg: torch.Tensor,
    tah: torch.Tensor,
    zpd: torch.Tensor,
    z0mg: torch.Tensor,
    z0hg: torch.Tensor,
    hcan: torch.Tensor,
    uc: torch.Tensor,
    z0h: torch.Tensor,
    fv: torch.Tensor,
    cwp: torch.Tensor,
    tv: torch.Tensor,
    mozg: torch.Tensor,
    fhg: torch.Tensor,
    fhgh: torch.Tensor,
) -> RagrbResult:
    """Vectorized ``ragrb`` under-canopy and leaf boundary resistance."""

    if iter_number > 1:
        tmp1 = NOAHMP_VKC * (torch.full_like(tah, NOAHMP_GRAV) / tah) * hg / (rhoair * NOAHMP_CPAIR)
        tmp1 = torch.where(torch.abs(tmp1) <= MPE, MPE, tmp1)
        molg = -_pow3(fv) / tmp1
        mozg_new = torch.clamp((zpd - z0mg) / molg, max=1.0)
        mozgh = torch.clamp((hcan - zpd) / molg, max=1.0)
    else:
        mozg_new = mozg * 0.0
        mozgh = mozg * 0.0

    unstable = mozg_new < 0.0
    one = torch.ones_like(mozg_new)
    fhg_power_base = torch.where(unstable, torch.clamp(1.0 - 15.0 * mozg_new, min=1.0e-30), one)
    fhgh_power_base = torch.where(unstable, torch.clamp(1.0 - 9.0 * mozg_new, min=1.0e-30), one)
    fhgnew = torch.where(
        unstable,
        working_power(fhg_power_base, -0.25),
        1.0 + 4.7 * mozg_new,
    )
    fhgnewh = torch.where(
        unstable,
        0.74 * working_power(fhgh_power_base, torch.full_like(fhgh_power_base, -0.5)),
        0.74 + 4.7 * mozgh,
    )
    if iter_number == 1:
        fhg_out = fhgnew
        fhgh_out = fhgnewh
    else:
        fhg_out = 0.5 * (fhg + fhgnew)
        fhgh_out = 0.5 * (fhgh + fhgnewh)

    cwpc_base = torch.clamp(cwp * vai * hcan * fhg_out, min=1.0)
    # RAGRB uses a real exponent, whose source rounding differs from sqrt.
    cwpc = working_power(cwpc_base, torch.full_like(cwpc_base, 0.5))
    cwpc = torch.clamp(cwpc, min=1.0, max=5.0)
    tmp1 = torch.exp(-cwpc * z0hg / hcan)
    tmp2 = torch.exp(-cwpc * (z0h + zpd) / hcan)
    tmprah2 = hcan * torch.exp(cwpc) / cwpc * (tmp1 - tmp2)
    kh = torch.clamp(NOAHMP_VKC * fv * (hcan - zpd) / torch.clamp(fhgh_out, min=0.1), min=MPE)
    rahg = tmprah2 / kh
    rawg = rahg
    ramg = rahg * 0.0
    tmprb = cwpc * 50.0 / (1.0 - torch.exp(-cwpc / 2.0))
    dleaf = tables.dleaf[_veg_index(vegtype)]
    rb = tmprb * torch.sqrt(dleaf / uc)
    rb = torch.clamp(rb, min=5.0, max=50.0)
    _ = tv
    return RagrbResult(ramg=ramg, rahg=rahg, rawg=rawg, rb=rb, mozg=mozg_new, fhg=fhg_out, fhgh=fhgh_out)


def _gather_top_layer(values: torch.Tensor, snow_levels: torch.Tensor) -> torch.Tensor:
    row = torch.arange(values.shape[0], device=values.device, dtype=torch.int64)
    idx = torch.clamp(snow_levels + 3, min=0, max=values.shape[1] - 1)
    return values[row, idx]


def _vege_flux_update_state_impl(
    *,
    active: torch.Tensor,
    use_active_mask: bool,
    z0h: torch.Tensor,
    z0hg: torch.Tensor,
    csigmaf1: torch.Tensor,
    fm: torch.Tensor,
    fh: torch.Tensor,
    fh2: torch.Tensor,
    fv: torch.Tensor,
    cm: torch.Tensor,
    ch: torch.Tensor,
    mozg: torch.Tensor,
    fhg: torch.Tensor,
    fhgh: torch.Tensor,
    rahg_final: torch.Tensor,
    rssun: torch.Tensor,
    rssha: torch.Tensor,
    psnsun: torch.Tensor,
    psnsha: torch.Tensor,
    tv: torch.Tensor,
    tg: torch.Tensor,
    tah: torch.Tensor,
    eah: torch.Tensor,
    irc: torch.Tensor,
    shc: torch.Tensor,
    evc: torch.Tensor,
    tr: torch.Tensor,
    irg: torch.Tensor,
    shg: torch.Tensor,
    evg: torch.Tensor,
    gh: torch.Tensor,
    canhs: torch.Tensor,
    h: torch.Tensor,
    hg: torch.Tensor,
    qsfc: torch.Tensor,
    estg: torch.Tensor,
    z0h_new: torch.Tensor,
    z0hg_new: torch.Tensor,
    csigmaf1_new: torch.Tensor,
    fm_new: torch.Tensor,
    fh_new: torch.Tensor,
    fh2_new: torch.Tensor,
    fv_new: torch.Tensor,
    cm_new: torch.Tensor,
    ch_new: torch.Tensor,
    rag_mozg: torch.Tensor,
    rag_fhg: torch.Tensor,
    rag_fhgh: torch.Tensor,
    rag_rahg: torch.Tensor,
    rssun_work: torch.Tensor,
    rssha_work: torch.Tensor,
    psnsun_work: torch.Tensor,
    psnsha_work: torch.Tensor,
    tv_new: torch.Tensor,
    tg_new: torch.Tensor,
    tah_new: torch.Tensor,
    eah_new: torch.Tensor,
    irc_new: torch.Tensor,
    shc_new: torch.Tensor,
    evc_new: torch.Tensor,
    tr_new: torch.Tensor,
    irg_new: torch.Tensor,
    shg_new: torch.Tensor,
    evg_new: torch.Tensor,
    gh_new: torch.Tensor,
    canhs_new: torch.Tensor,
    h_new: torch.Tensor,
    hg_new: torch.Tensor,
    qsfc_new: torch.Tensor,
    estg_work: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Merge one vegetated-flux fixed-point iteration into the carried state."""

    if not use_active_mask:
        return (
            z0h_new,
            z0hg_new,
            csigmaf1_new,
            fm_new,
            fh_new,
            fh2_new,
            fv_new,
            cm_new,
            ch_new,
            rag_mozg,
            rag_fhg,
            rag_fhgh,
            rag_rahg,
            rssun_work,
            rssha_work,
            psnsun_work,
            psnsha_work,
            tv_new,
            tg_new,
            tah_new,
            eah_new,
            irc_new,
            shc_new,
            evc_new,
            tr_new,
            irg_new,
            shg_new,
            evg_new,
            gh_new,
            canhs_new,
            h_new,
            hg_new,
            qsfc_new,
            estg_work,
        )
    return (
        torch.where(active, z0h_new, z0h),
        torch.where(active, z0hg_new, z0hg),
        torch.where(active, csigmaf1_new, csigmaf1),
        torch.where(active, fm_new, fm),
        torch.where(active, fh_new, fh),
        torch.where(active, fh2_new, fh2),
        torch.where(active, fv_new, fv),
        torch.where(active, cm_new, cm),
        torch.where(active, ch_new, ch),
        torch.where(active, rag_mozg, mozg),
        torch.where(active, rag_fhg, fhg),
        torch.where(active, rag_fhgh, fhgh),
        torch.where(active, rag_rahg, rahg_final),
        torch.where(active, rssun_work, rssun),
        torch.where(active, rssha_work, rssha),
        torch.where(active, psnsun_work, psnsun),
        torch.where(active, psnsha_work, psnsha),
        torch.where(active, tv_new, tv),
        torch.where(active, tg_new, tg),
        torch.where(active, tah_new, tah),
        torch.where(active, eah_new, eah),
        torch.where(active, irc_new, irc),
        torch.where(active, shc_new, shc),
        torch.where(active, evc_new, evc),
        torch.where(active, tr_new, tr),
        torch.where(active, irg_new, irg),
        torch.where(active, shg_new, shg),
        torch.where(active, evg_new, evg),
        torch.where(active, gh_new, gh),
        torch.where(active, canhs_new, canhs),
        torch.where(active, h_new, h),
        torch.where(active, hg_new, hg),
        torch.where(active, qsfc_new, qsfc),
        torch.where(active, estg_work, estg),
    )


def _vege_flux_update_state(*, use_compile: bool, **kwargs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if use_compile:
        return run_compile_island(
            "land_vege_flux_update",
            "vege_flux_update_state",
            _vege_flux_update_state_impl,
            **kwargs,
        )
    return _vege_flux_update_state_impl(**kwargs)


def _vege_flux_iteration_physics_impl(
    *,
    after_first: bool,
    first_iteration: bool,
    sfctmp: torch.Tensor,
    ps: torch.Tensor,
    prsl1: torch.Tensor,
    rhoair: torch.Tensor,
    fveg: torch.Tensor,
    pahv: torch.Tensor,
    pahg: torch.Tensor,
    eair: torch.Tensor,
    dlwflx: torch.Tensor,
    zpd: torch.Tensor,
    z0mg: torch.Tensor,
    hcan: torch.Tensor,
    emv: torch.Tensor,
    emg: torch.Tensor,
    gamma_ground: torch.Tensor,
    rsurf: torch.Tensor,
    rhsur: torch.Tensor,
    df_top: torch.Tensor,
    dz_top: torch.Tensor,
    stc_top: torch.Tensor,
    ur: torch.Tensor,
    laisune: torch.Tensor,
    laishae: torch.Tensor,
    parsun: torch.Tensor,
    parsha: torch.Tensor,
    sav: torch.Tensor,
    sag: torch.Tensor,
    cwp: torch.Tensor,
    dleaf: torch.Tensor,
    cbiom: torch.Tensor,
    rgl: torch.Tensor,
    rsmin: torch.Tensor,
    rsmax: torch.Tensor,
    topt: torch.Tensor,
    hs: torch.Tensor,
    dt_tensor: torch.Tensor,
    canliq: torch.Tensor,
    canice: torch.Tensor,
    fwet: torch.Tensor,
    btran: torch.Tensor,
    latheav: torch.Tensor,
    gammav: torch.Tensor,
    uc: torch.Tensor,
    vaie: torch.Tensor,
    zero: torch.Tensor,
    one: torch.Tensor,
    z0h_static: torch.Tensor,
    z0hg_static: torch.Tensor,
    fv_new: torch.Tensor,
    ch_new: torch.Tensor,
    tah: torch.Tensor,
    hg: torch.Tensor,
    fv: torch.Tensor,
    mozg: torch.Tensor,
    fhg: torch.Tensor,
    fhgh: torch.Tensor,
    rssun: torch.Tensor,
    rssha: torch.Tensor,
    psnsun: torch.Tensor,
    psnsha: torch.Tensor,
    tv: torch.Tensor,
    tg: torch.Tensor,
    eah: torch.Tensor,
    estg: torch.Tensor,
    selected_decoupled: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Dense vegetated-flux iteration algebra as a flat tensor kernel."""

    rahc = torch.maximum(one, 1.0 / (ch_new * ur))
    rawc = rahc
    if after_first:
        tmp1 = NOAHMP_VKC * (torch.full_like(tah, NOAHMP_GRAV) / tah) * hg / (rhoair * NOAHMP_CPAIR)
        tmp1 = torch.where(torch.abs(tmp1) <= MPE, MPE, tmp1)
        molg = -_pow3(fv_new) / tmp1
        rag_mozg = torch.clamp((zpd - z0mg) / molg, max=1.0)
        mozgh = torch.clamp((hcan - zpd) / molg, max=1.0)
    else:
        rag_mozg = zero
        mozgh = zero

    unstable = rag_mozg < 0.0
    fhg_power_base = torch.where(
        unstable,
        torch.clamp(1.0 - 15.0 * rag_mozg, min=1.0e-30),
        one,
    )
    fhgh_power_base = torch.where(
        unstable,
        torch.clamp(1.0 - 9.0 * rag_mozg, min=1.0e-30),
        one,
    )
    fhgnew = torch.where(unstable, working_power(fhg_power_base, -0.25), 1.0 + 4.7 * rag_mozg)
    fhgnewh = torch.where(
        unstable,
        0.74 * working_power(fhgh_power_base, torch.full_like(fhgh_power_base, -0.5)),
        0.74 + 4.7 * mozgh,
    )
    if first_iteration:
        rag_fhg = fhgnew
        rag_fhgh = fhgnewh
    else:
        rag_fhg = 0.5 * (fhg + fhgnew)
        rag_fhgh = 0.5 * (fhgh + fhgnewh)

    cwpc_base = torch.clamp(cwp * vaie * hcan * rag_fhg, min=1.0)
    cwpc = working_power(cwpc_base, torch.full_like(cwpc_base, 0.5))
    cwpc = torch.clamp(cwpc, min=1.0, max=5.0)
    tmp1 = torch.exp(-cwpc * z0hg_static / hcan)
    tmp2 = torch.exp(-cwpc * (z0h_static + zpd) / hcan)
    tmprah2 = hcan * torch.exp(cwpc) / cwpc * (tmp1 - tmp2)
    kh = torch.clamp(NOAHMP_VKC * fv_new * (hcan - zpd) / torch.clamp(rag_fhgh, min=0.1), min=MPE)
    rag_rahg = tmprah2 / kh
    rag_rawg = rag_rahg
    tmprb = cwpc * 50.0 / (1.0 - torch.exp(-cwpc / 2.0))
    rag_rb = torch.clamp(tmprb * torch.sqrt(dleaf / uc), min=5.0, max=50.0)

    t_c = torch.clamp(tv - NOAHMP_TFRZ, min=-50.0, max=50.0)
    esatw, esati, dsatw, dsati = esat_noahmp(t_c)
    warm_tv = t_c > 0.0
    estv = _mask_blend(warm_tv, esatw, esati)
    destv = _mask_blend(warm_tv, dsatw, dsati)
    q2_can = CON_EPS * eah / (prsl1 + CON_EPSM1 * eah)
    q2_can = q2_can / (1.0 - q2_can)
    q2sat_can = calhum_noahmp(tv, prsl1)
    topt_minus_tv_sq = _pow2(topt - tv)
    rct_can = torch.clamp(1.0 - 0.0016 * topt_minus_tv_sq, min=0.0001)
    rcq_can = torch.clamp(1.0 / (1.0 + hs * torch.clamp(q2sat_can - q2_can, min=0.0)), min=0.01)
    ff_sun = 2.0 * parsun / rgl
    rcs_sun = torch.clamp((ff_sun + rsmin / rsmax) / (1.0 + ff_sun), min=0.0001)
    sun_rc = rsmin / (rcs_sun * rct_can * rcq_can * btran)
    sun_psn = parsun * 0.0 - 999.99
    ff_sha = 2.0 * parsha / rgl
    rcs_sha = torch.clamp((ff_sha + rsmin / rsmax) / (1.0 + ff_sha), min=0.0001)
    sha_rc = rsmin / (rcs_sha * rct_can * rcq_can * btran)
    sha_psn = parsha * 0.0 - 999.99
    if first_iteration:
        rssun_work = sun_rc
        rssha_work = sha_rc
        psnsun_work = sun_psn
        psnsha_work = sha_psn
    else:
        rssun_work = rssun
        rssha_work = rssha
        psnsun_work = psnsun
        psnsha_work = psnsha

    tg_pow4 = _pow4(tg)
    air = -emv * (1.0 + (1.0 - emv) * (1.0 - emg)) * dlwflx - emv * emg * NOAHMP_SB * tg_pow4
    cir = (2.0 - emv * (1.0 - emg)) * emv * NOAHMP_SB
    cah = 1.0 / rahc
    cvh = 2.0 * vaie / rag_rb
    cgh = 1.0 / rag_rahg
    cond_h = cah + cvh + cgh
    ata = (sfctmp * cah + tg * cgh) / cond_h
    bta = cvh / cond_h
    csh_canopy = (1.0 - bta) * rhoair * NOAHMP_CPAIR * cvh

    evpot = fveg * rhoair * NOAHMP_CPAIR * vaie / rag_rb * (estv - eah) / gammav
    caw = 1.0 / rawc
    cew_cap_liq = canliq * latheav / dt_tensor / torch.clamp(evpot, min=MPE)
    cew_cap_ice = canice * latheav / dt_tensor / torch.clamp(evpot, min=MPE)
    cew_cap = _mask_blend(tv > NOAHMP_TFRZ, cew_cap_liq, cew_cap_ice)
    cew_wet = torch.minimum(fwet, cew_cap) * vaie / rag_rb
    cew_dry = fwet * vaie / rag_rb
    cew = torch.where((evpot > 0.0) & (fwet > 0.0), cew_wet, cew_dry)
    ctw = (1.0 - fwet) * (laisune / (rag_rb + rssun_work) + laishae / (rag_rb + rssha_work))
    cgw = 1.0 / (rag_rawg + rsurf)
    cond_w = caw + cew + ctw + cgw
    aea = (eair * caw + estg * cgw) / cond_w
    bea = (cew + ctw) / cond_w
    cev_canopy = (1.0 - bea) * cew * rhoair * NOAHMP_CPAIR / gammav
    ctr = (1.0 - bea) * ctw * rhoair * NOAHMP_CPAIR / gammav

    tah_new = ata + bta * tv
    eah_new = aea + bea * estv
    tv_pow4 = _pow4(tv)
    tv_pow3 = _pow3(tv)
    irc_new = fveg * (air + cir * tv_pow4)
    shc_new = fveg * rhoair * NOAHMP_CPAIR * cvh * (tv - tah_new)
    evc_new = fveg * rhoair * NOAHMP_CPAIR * cew * (estv - eah_new) / gammav
    tr_new = fveg * rhoair * NOAHMP_CPAIR * ctw * (estv - eah_new) / gammav
    evc_cap = _mask_blend(tv > NOAHMP_TFRZ, canliq * latheav / dt_tensor, canice * latheav / dt_tensor)
    evc_new = torch.minimum(evc_cap, evc_new)
    hcv = fveg * (cbiom * vaie * NOAHMP_CWAT + canliq * NOAHMP_CWAT / NOAHMP_DENH2O + canice * NOAHMP_CICE / NOAHMP_DENICE)
    balance_v = sav - irc_new - shc_new - evc_new - tr_new + pahv
    jac_v = fveg * (4.0 * cir * tv_pow3 + csh_canopy + (cev_canopy + ctr) * destv) + hcv / dt_tensor
    dtv = balance_v / jac_v
    irc_new = irc_new + fveg * 4.0 * cir * tv_pow3 * dtv
    shc_new = shc_new + fveg * csh_canopy * dtv
    evc_new = evc_new + fveg * cev_canopy * destv * dtv
    tr_new = tr_new + fveg * ctr * destv * dtv
    canhs_new = dtv * hcv / dt_tensor
    tv_new = tv + dtv
    if not selected_decoupled:
        tah_new = ata + bta * tv_new
    h_new = rhoair * NOAHMP_CPAIR * (tah_new - sfctmp) / rahc
    hg_new = rhoair * NOAHMP_CPAIR * (tg - tah_new) / rag_rahg
    qsfc_new = CON_EPS * eah_new / (prsl1 + CON_EPSM1 * eah_new)

    tv_new_pow4 = _pow4(tv_new)
    air_g = -emg * (1.0 - emv) * dlwflx - emg * emv * NOAHMP_SB * tv_new_pow4
    cir_g = emg * NOAHMP_SB
    csh_ground = rhoair * NOAHMP_CPAIR / rag_rahg
    cev_ground = rhoair * NOAHMP_CPAIR / (gamma_ground * (rag_rawg + rsurf))
    cgh_ground = 2.0 * df_top / dz_top
    t_cg = torch.clamp(tg - NOAHMP_TFRZ, min=-50.0, max=50.0)
    esatw_g, esati_g, dsatw_g, dsati_g = esat_noahmp(t_cg)
    warm_tg = t_cg > 0.0
    estg_work = _mask_blend(warm_tg, esatw_g, esati_g)
    destg = _mask_blend(warm_tg, dsatw_g, dsati_g)
    tg_pow3 = _pow3(tg)
    irg_new = cir_g * tg_pow4 + air_g
    shg_new = csh_ground * (tg - tah_new)
    evg_new = cev_ground * (estg_work * rhsur - eah_new)
    gh_new = cgh_ground * (tg - stc_top)
    balance_g = sag - irg_new - shg_new - evg_new - gh_new + pahg
    jac_g = 4.0 * cir_g * tg_pow3 + csh_ground + cev_ground * destg + cgh_ground
    dtg = balance_g / jac_g
    irg_new = irg_new + 4.0 * cir_g * tg_pow3 * dtg
    shg_new = shg_new + csh_ground * dtg
    evg_new = evg_new + cev_ground * destg * dtg
    gh_new = gh_new + cgh_ground * dtg
    tg_new = tg + dtg

    return (
        rag_mozg,
        rag_fhg,
        rag_fhgh,
        rag_rahg,
        rssun_work,
        rssha_work,
        psnsun_work,
        psnsha_work,
        tv_new,
        tg_new,
        tah_new,
        eah_new,
        irc_new,
        shc_new,
        evc_new,
        tr_new,
        irg_new,
        shg_new,
        evg_new,
        gh_new,
        canhs_new,
        h_new,
        hg_new,
        qsfc_new,
        estg_work,
        dtv,
        dtg,
    )


def _vege_flux_iteration_physics(*, use_compile: bool, **kwargs: torch.Tensor | bool) -> tuple[torch.Tensor, ...]:
    if use_compile:
        return run_compile_island(
            "land_vege_flux_iteration",
            "vege_flux_iteration_physics",
            _vege_flux_iteration_physics_impl,
            **kwargs,
        )
    return _vege_flux_iteration_physics_impl(**kwargs)


def _vege_flux_loop_step_impl(
    *,
    after_first: bool,
    first_iteration: bool,
    can_converge: bool,
    use_active_mask: bool,
    thsfc_loc: bool,
    sfctmp: torch.Tensor,
    qair: torch.Tensor,
    ps: torch.Tensor,
    prsl1: torch.Tensor,
    rhoair: torch.Tensor,
    fveg: torch.Tensor,
    shdfac: torch.Tensor,
    garea: torch.Tensor,
    pahv: torch.Tensor,
    pahg: torch.Tensor,
    eair: torch.Tensor,
    dlwflx: torch.Tensor,
    zpd: torch.Tensor,
    z0m: torch.Tensor,
    z0mg: torch.Tensor,
    hcan: torch.Tensor,
    emv: torch.Tensor,
    emg: torch.Tensor,
    gamma_ground: torch.Tensor,
    rsurf: torch.Tensor,
    rhsur: torch.Tensor,
    df_top: torch.Tensor,
    dz_top: torch.Tensor,
    stc_top: torch.Tensor,
    ur: torch.Tensor,
    zlvl: torch.Tensor,
    vaie: torch.Tensor,
    laisune: torch.Tensor,
    laishae: torch.Tensor,
    parsun: torch.Tensor,
    parsha: torch.Tensor,
    sav: torch.Tensor,
    sag: torch.Tensor,
    cwp: torch.Tensor,
    dleaf: torch.Tensor,
    cbiom: torch.Tensor,
    rgl: torch.Tensor,
    rsmin: torch.Tensor,
    rsmax: torch.Tensor,
    topt: torch.Tensor,
    hs: torch.Tensor,
    dt_tensor: torch.Tensor,
    canliq: torch.Tensor,
    canice: torch.Tensor,
    fwet: torch.Tensor,
    btran: torch.Tensor,
    latheav: torch.Tensor,
    gammav: torch.Tensor,
    uc: torch.Tensor,
    zero: torch.Tensor,
    one: torch.Tensor,
    z0h_static: torch.Tensor,
    z0hg_static: torch.Tensor,
    prslkix: torch.Tensor,
    prsik1x: torch.Tensor,
    prslk1x: torch.Tensor,
    liter: torch.Tensor,
    done: torch.Tensor,
    z0h: torch.Tensor,
    z0hg: torch.Tensor,
    csigmaf1: torch.Tensor,
    fm: torch.Tensor,
    fh: torch.Tensor,
    fh2: torch.Tensor,
    fv: torch.Tensor,
    cm: torch.Tensor,
    ch: torch.Tensor,
    mozg: torch.Tensor,
    fhg: torch.Tensor,
    fhgh: torch.Tensor,
    rahg_final: torch.Tensor,
    rssun: torch.Tensor,
    rssha: torch.Tensor,
    psnsun: torch.Tensor,
    psnsha: torch.Tensor,
    tv: torch.Tensor,
    tg: torch.Tensor,
    tah: torch.Tensor,
    eah: torch.Tensor,
    irc: torch.Tensor,
    shc: torch.Tensor,
    evc: torch.Tensor,
    tr: torch.Tensor,
    irg: torch.Tensor,
    shg: torch.Tensor,
    evg: torch.Tensor,
    gh: torch.Tensor,
    canhs: torch.Tensor,
    h: torch.Tensor,
    hg: torch.Tensor,
    qsfc: torch.Tensor,
    estg: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """One fixed-point pass of vegetated surface flux as a flat tensor island."""

    active = ~done
    fm_new, fh_new, fh2_new, fv_new, cm_new, ch_new = _sfcdif3_impl(
        sfctmp=sfctmp,
        qair=qair,
        ur=ur,
        zlvl=zlvl,
        tgb=tah,
        thsfc_loc=thsfc_loc,
        prslkix=prslkix,
        prsik1x=prsik1x,
        prslk1x=prslk1x,
        z0m=z0m,
        z0h=z0h_static,
        zpd=zpd,
        fveg=shdfac,
        garea1=garea,
    )
    (
        rag_mozg,
        rag_fhg,
        rag_fhgh,
        rag_rahg,
        rssun_work,
        rssha_work,
        psnsun_work,
        psnsha_work,
        tv_new,
        tg_new,
        tah_new,
        eah_new,
        irc_new,
        shc_new,
        evc_new,
        tr_new,
        irg_new,
        shg_new,
        evg_new,
        gh_new,
        canhs_new,
        h_new,
        hg_new,
        qsfc_new,
        estg_work,
        dtv,
        dtg,
    ) = _vege_flux_iteration_physics_impl(
        after_first=after_first,
        first_iteration=first_iteration,
        sfctmp=sfctmp,
        ps=ps,
        prsl1=prsl1,
        rhoair=rhoair,
        fveg=fveg,
        pahv=pahv,
        pahg=pahg,
        eair=eair,
        dlwflx=dlwflx,
        zpd=zpd,
        z0mg=z0mg,
        hcan=hcan,
        emv=emv,
        emg=emg,
        gamma_ground=gamma_ground,
        rsurf=rsurf,
        rhsur=rhsur,
        df_top=df_top,
        dz_top=dz_top,
        stc_top=stc_top,
        ur=ur,
        laisune=laisune,
        laishae=laishae,
        parsun=parsun,
        parsha=parsha,
        sav=sav,
        sag=sag,
        cwp=cwp,
        dleaf=dleaf,
        cbiom=cbiom,
        rgl=rgl,
        rsmin=rsmin,
        rsmax=rsmax,
        topt=topt,
        hs=hs,
        dt_tensor=dt_tensor,
        canliq=canliq,
        canice=canice,
        fwet=fwet,
        btran=btran,
        latheav=latheav,
        gammav=gammav,
        uc=uc,
        vaie=vaie,
        zero=zero,
        one=one,
        z0h_static=z0h_static,
        z0hg_static=z0hg_static,
        fv_new=fv_new,
        ch_new=ch_new,
        tah=tah,
        hg=hg,
        fv=fv,
        mozg=mozg,
        fhg=fhg,
        fhgh=fhgh,
        rssun=rssun,
        rssha=rssha,
        psnsun=psnsun,
        psnsha=psnsha,
        tv=tv,
        tg=tg,
        eah=eah,
        estg=estg,
    )
    (
        z0h,
        z0hg,
        csigmaf1,
        fm,
        fh,
        fh2,
        fv,
        cm,
        ch,
        mozg,
        fhg,
        fhgh,
        rahg_final,
        rssun,
        rssha,
        psnsun,
        psnsha,
        tv,
        tg,
        tah,
        eah,
        irc,
        shc,
        evc,
        tr,
        irg,
        shg,
        evg,
        gh,
        canhs,
        h,
        hg,
        qsfc,
        estg,
    ) = _vege_flux_update_state_impl(
        active=active,
        use_active_mask=use_active_mask,
        z0h=z0h,
        z0hg=z0hg,
        csigmaf1=csigmaf1,
        fm=fm,
        fh=fh,
        fh2=fh2,
        fv=fv,
        cm=cm,
        ch=ch,
        mozg=mozg,
        fhg=fhg,
        fhgh=fhgh,
        rahg_final=rahg_final,
        rssun=rssun,
        rssha=rssha,
        psnsun=psnsun,
        psnsha=psnsha,
        tv=tv,
        tg=tg,
        tah=tah,
        eah=eah,
        irc=irc,
        shc=shc,
        evc=evc,
        tr=tr,
        irg=irg,
        shg=shg,
        evg=evg,
        gh=gh,
        canhs=canhs,
        h=h,
        hg=hg,
        qsfc=qsfc,
        estg=estg,
        z0h_new=z0h_static,
        z0hg_new=z0hg_static,
        csigmaf1_new=zero,
        fm_new=fm_new,
        fh_new=fh_new,
        fh2_new=fh2_new,
        fv_new=fv_new,
        cm_new=cm_new,
        ch_new=ch_new,
        rag_mozg=rag_mozg,
        rag_fhg=rag_fhg,
        rag_fhgh=rag_fhgh,
        rag_rahg=rag_rahg,
        rssun_work=rssun_work,
        rssha_work=rssha_work,
        psnsun_work=psnsun_work,
        psnsha_work=psnsha_work,
        tv_new=tv_new,
        tg_new=tg_new,
        tah_new=tah_new,
        eah_new=eah_new,
        irc_new=irc_new,
        shc_new=shc_new,
        evc_new=evc_new,
        tr_new=tr_new,
        irg_new=irg_new,
        shg_new=shg_new,
        evg_new=evg_new,
        gh_new=gh_new,
        canhs_new=canhs_new,
        h_new=h_new,
        hg_new=hg_new,
        qsfc_new=qsfc_new,
        estg_work=estg_work,
    )
    completed_final_loop = active & liter
    converged = active & (~liter) & can_converge & (torch.abs(dtv) <= 0.01) & (torch.abs(dtg) <= 0.01)
    done = done | completed_final_loop
    liter = liter | converged
    return (
        z0h,
        z0hg,
        csigmaf1,
        fm,
        fh,
        fh2,
        fv,
        cm,
        ch,
        mozg,
        fhg,
        fhgh,
        rahg_final,
        rssun,
        rssha,
        psnsun,
        psnsha,
        tv,
        tg,
        tah,
        eah,
        irc,
        shc,
        evc,
        tr,
        irg,
        shg,
        evg,
        gh,
        canhs,
        h,
        hg,
        qsfc,
        estg,
        done,
        liter,
    )


def _vege_flux_loop_step(*, use_compile: bool, **kwargs: torch.Tensor | bool) -> tuple[torch.Tensor, ...]:
    if use_compile:
        return run_compile_island(
            "land_vege_flux_loop_step",
            "vege_flux_sfc3_loop_step",
            _vege_flux_loop_step_impl,
            **kwargs,
        )
    return _vege_flux_loop_step_impl(**kwargs)


def _vege_flux_loop_chunk_impl(
    *,
    base_iter_index: int,
    chunk_count: int,
    thsfc_loc: bool,
    sfctmp: torch.Tensor,
    qair: torch.Tensor,
    ps: torch.Tensor,
    prsl1: torch.Tensor,
    rhoair: torch.Tensor,
    fveg: torch.Tensor,
    shdfac: torch.Tensor,
    garea: torch.Tensor,
    pahv: torch.Tensor,
    pahg: torch.Tensor,
    eair: torch.Tensor,
    dlwflx: torch.Tensor,
    zpd: torch.Tensor,
    z0m: torch.Tensor,
    z0mg: torch.Tensor,
    hcan: torch.Tensor,
    emv: torch.Tensor,
    emg: torch.Tensor,
    gamma_ground: torch.Tensor,
    rsurf: torch.Tensor,
    rhsur: torch.Tensor,
    df_top: torch.Tensor,
    dz_top: torch.Tensor,
    stc_top: torch.Tensor,
    ur: torch.Tensor,
    zlvl: torch.Tensor,
    vaie: torch.Tensor,
    laisune: torch.Tensor,
    laishae: torch.Tensor,
    parsun: torch.Tensor,
    parsha: torch.Tensor,
    sav: torch.Tensor,
    sag: torch.Tensor,
    cwp: torch.Tensor,
    dleaf: torch.Tensor,
    cbiom: torch.Tensor,
    rgl: torch.Tensor,
    rsmin: torch.Tensor,
    rsmax: torch.Tensor,
    topt: torch.Tensor,
    hs: torch.Tensor,
    dt_tensor: torch.Tensor,
    canliq: torch.Tensor,
    canice: torch.Tensor,
    fwet: torch.Tensor,
    btran: torch.Tensor,
    latheav: torch.Tensor,
    gammav: torch.Tensor,
    uc: torch.Tensor,
    zero: torch.Tensor,
    one: torch.Tensor,
    z0h_static: torch.Tensor,
    z0hg_static: torch.Tensor,
    prslkix: torch.Tensor,
    prsik1x: torch.Tensor,
    prslk1x: torch.Tensor,
    liter: torch.Tensor,
    done: torch.Tensor,
    z0h: torch.Tensor,
    z0hg: torch.Tensor,
    csigmaf1: torch.Tensor,
    fm: torch.Tensor,
    fh: torch.Tensor,
    fh2: torch.Tensor,
    fv: torch.Tensor,
    cm: torch.Tensor,
    ch: torch.Tensor,
    mozg: torch.Tensor,
    fhg: torch.Tensor,
    fhgh: torch.Tensor,
    rahg_final: torch.Tensor,
    rssun: torch.Tensor,
    rssha: torch.Tensor,
    psnsun: torch.Tensor,
    psnsha: torch.Tensor,
    tv: torch.Tensor,
    tg: torch.Tensor,
    tah: torch.Tensor,
    eah: torch.Tensor,
    irc: torch.Tensor,
    shc: torch.Tensor,
    evc: torch.Tensor,
    tr: torch.Tensor,
    irg: torch.Tensor,
    shg: torch.Tensor,
    evg: torch.Tensor,
    gh: torch.Tensor,
    canhs: torch.Tensor,
    h: torch.Tensor,
    hg: torch.Tensor,
    qsfc: torch.Tensor,
    estg: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Fixed-count vegetated-flux passes as one flat compile island."""

    local_iter_index = 0
    while local_iter_index < chunk_count:
        iter_index = base_iter_index + local_iter_index
        iter_number = iter_index + 1
        (
            z0h,
            z0hg,
            csigmaf1,
            fm,
            fh,
            fh2,
            fv,
            cm,
            ch,
            mozg,
            fhg,
            fhgh,
            rahg_final,
            rssun,
            rssha,
            psnsun,
            psnsha,
            tv,
            tg,
            tah,
            eah,
            irc,
            shc,
            evc,
            tr,
            irg,
            shg,
            evg,
            gh,
            canhs,
            h,
            hg,
            qsfc,
            estg,
            done,
            liter,
        ) = _vege_flux_loop_step_impl(
            after_first=iter_number > 1,
            first_iteration=iter_number == 1,
            can_converge=iter_number >= 5,
            use_active_mask=iter_index >= 6,
            thsfc_loc=thsfc_loc,
            sfctmp=sfctmp,
            qair=qair,
            ps=ps,
            prsl1=prsl1,
            rhoair=rhoair,
            fveg=fveg,
            shdfac=shdfac,
            garea=garea,
            pahv=pahv,
            pahg=pahg,
            eair=eair,
            dlwflx=dlwflx,
            zpd=zpd,
            z0m=z0m,
            z0mg=z0mg,
            hcan=hcan,
            emv=emv,
            emg=emg,
            gamma_ground=gamma_ground,
            rsurf=rsurf,
            rhsur=rhsur,
            df_top=df_top,
            dz_top=dz_top,
            stc_top=stc_top,
            ur=ur,
            zlvl=zlvl,
            vaie=vaie,
            laisune=laisune,
            laishae=laishae,
            parsun=parsun,
            parsha=parsha,
            sav=sav,
            sag=sag,
            cwp=cwp,
            dleaf=dleaf,
            cbiom=cbiom,
            rgl=rgl,
            rsmin=rsmin,
            rsmax=rsmax,
            topt=topt,
            hs=hs,
            dt_tensor=dt_tensor,
            canliq=canliq,
            canice=canice,
            fwet=fwet,
            btran=btran,
            latheav=latheav,
            gammav=gammav,
            uc=uc,
            zero=zero,
            one=one,
            z0h_static=z0h_static,
            z0hg_static=z0hg_static,
            prslkix=prslkix,
            prsik1x=prsik1x,
            prslk1x=prslk1x,
            liter=liter,
            done=done,
            z0h=z0h,
            z0hg=z0hg,
            csigmaf1=csigmaf1,
            fm=fm,
            fh=fh,
            fh2=fh2,
            fv=fv,
            cm=cm,
            ch=ch,
            mozg=mozg,
            fhg=fhg,
            fhgh=fhgh,
            rahg_final=rahg_final,
            rssun=rssun,
            rssha=rssha,
            psnsun=psnsun,
            psnsha=psnsha,
            tv=tv,
            tg=tg,
            tah=tah,
            eah=eah,
            irc=irc,
            shc=shc,
            evc=evc,
            tr=tr,
            irg=irg,
            shg=shg,
            evg=evg,
            gh=gh,
            canhs=canhs,
            h=h,
            hg=hg,
            qsfc=qsfc,
            estg=estg,
        )
        local_iter_index += 1
    return (
        z0h,
        z0hg,
        csigmaf1,
        fm,
        fh,
        fh2,
        fv,
        cm,
        ch,
        mozg,
        fhg,
        fhgh,
        rahg_final,
        rssun,
        rssha,
        psnsun,
        psnsha,
        tv,
        tg,
        tah,
        eah,
        irc,
        shc,
        evc,
        tr,
        irg,
        shg,
        evg,
        gh,
        canhs,
        h,
        hg,
        qsfc,
        estg,
        done,
        liter,
    )


def _vege_flux_loop_chunk(*, use_compile: bool, **kwargs: torch.Tensor | bool | int) -> tuple[torch.Tensor, ...]:
    if use_compile:
        return run_compile_island(
            "land_vege_flux_loop_chunk",
            "vege_flux_sfc3_loop_chunk",
            _vege_flux_loop_chunk_impl,
            **kwargs,
        )
    return _vege_flux_loop_chunk_impl(**kwargs)


def _vege_flux_finalize_impl(
    *,
    snowh: torch.Tensor,
    tg: torch.Tensor,
    fsno: torch.Tensor,
    tv: torch.Tensor,
    emg: torch.Tensor,
    emv: torch.Tensor,
    dlwflx: torch.Tensor,
    shg: torch.Tensor,
    evg: torch.Tensor,
    sag: torch.Tensor,
    pahg: torch.Tensor,
    rhoair: torch.Tensor,
    cm: torch.Tensor,
    ur: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    fv: torch.Tensor,
    fh2: torch.Tensor,
    tah: torch.Tensor,
    shc: torch.Tensor,
    fveg: torch.Tensor,
    qsfc: torch.Tensor,
    evc: torch.Tensor,
    tr: torch.Tensor,
    latheav: torch.Tensor,
    ch: torch.Tensor,
    vaie: torch.Tensor,
    rb: torch.Tensor,
    rahg_final: torch.Tensor,
    eah: torch.Tensor,
    rssun: torch.Tensor,
    rssha: torch.Tensor,
    psnsun: torch.Tensor,
    psnsha: torch.Tensor,
    irg: torch.Tensor,
    irc: torch.Tensor,
    gh: torch.Tensor,
    canhs: torch.Tensor,
    z0h: torch.Tensor,
    z0hg: torch.Tensor,
    csigmaf1: torch.Tensor,
    gamma_ground: torch.Tensor,
    rsurf: torch.Tensor,
    rhsur: torch.Tensor,
    estg: torch.Tensor,
    sfcdif4_wspd: torch.Tensor | None = None,
    sfcdif4_ch2v: torch.Tensor | None = None,
    sfcdif4_cq2v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Final vegetated-flux diagnostics after the iteration loop.

    With ``sfcdif4_*`` (opt_sfc=4) the 2-m and output conductances use sfcdif4's wind:
    cah2 = 1/max(1, 1/(ch2v*wspdv)), cq2v = 1/max(1, 1/(cq2v*wspdv)), ch = 1/max(1, 1/(ch*wspdv)).
    """

    snow_melt_reset = (snowh > 0.05) & (tg > NOAHMP_TFRZ)
    tg_reset = (1.0 - fsno) * tg + fsno * NOAHMP_TFRZ
    irg_reset = NOAHMP_SB * emg * _pow4(tg_reset) - emg * (1.0 - emv) * dlwflx - emg * emv * NOAHMP_SB * _pow4(tv)
    shg = torch.where(snow_melt_reset, rhoair * NOAHMP_CPAIR / rahg_final * (tg_reset - tah), shg)
    evg = torch.where(snow_melt_reset, rhoair * NOAHMP_CPAIR / (gamma_ground * (rahg_final + rsurf)) * (estg * rhsur - eah), evg)
    gh_reset = sag + pahg - (irg_reset + shg + evg)
    tg = torch.where(snow_melt_reset, tg_reset, tg)
    irg = torch.where(snow_melt_reset, irg_reset, irg)
    gh = torch.where(snow_melt_reset, gh_reset, gh)

    tauxv = -rhoair * cm * ur * u1
    tauyv = -rhoair * cm * ur * v1
    if sfcdif4_wspd is None:
        cah2 = fv * NOAHMP_VKC / fh2
        cq2v = cah2
        ch_wind = ur
    else:
        cah2 = 1.0 / torch.clamp(1.0 / (sfcdif4_ch2v * sfcdif4_wspd), min=1.0)
        cq2v = 1.0 / torch.clamp(1.0 / (sfcdif4_cq2v * sfcdif4_wspd), min=1.0)
        ch_wind = sfcdif4_wspd
    low_cah2 = cah2 < 1.0e-5
    t2mv = torch.where(low_cah2, tah, tah - (shg + shc / fveg) / (rhoair * NOAHMP_CPAIR) / cah2)
    q2v = torch.where(low_cah2, qsfc, qsfc - ((evc + tr) / fveg + evg) / (latheav * rhoair) / cq2v)
    chleaf = 2.0 * vaie / rb
    chuc = 1.0 / rahg_final
    ch = 1.0 / torch.clamp(1.0 / (ch * ch_wind), min=1.0)
    return (
        tg,
        tv,
        tah,
        eah,
        cm,
        ch,
        chleaf,
        chuc,
        rb,
        rssun,
        rssha,
        psnsun,
        psnsha,
        irg,
        irc,
        shg,
        shc,
        evg,
        evc,
        tr,
        gh,
        tauxv,
        tauyv,
        t2mv,
        q2v,
        qsfc,
        canhs,
        cah2,
        z0h,
        z0hg,
        csigmaf1,
    )


def _vege_flux_finalize(*, use_compile: bool, **kwargs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if use_compile:
        return run_compile_island(
            "land_vege_flux_finalize",
            "vege_flux_sfc3_finalize",
            _vege_flux_finalize_impl,
            **kwargs,
        )
    return _vege_flux_finalize_impl(**kwargs)


def _vege_flux_setup(
    *,
    temperature_forcing: torch.Tensor,
    tskin: torch.Tensor,
    cm_state: torch.Tensor,
    ch_state: torch.Tensor,
    vegtype: torch.Tensor,
    ps: torch.Tensor,
    prsl1: torch.Tensor,
    rhoair: torch.Tensor,
    qair: torch.Tensor,
    fveg: torch.Tensor,
    shdfac: torch.Tensor,
    garea: torch.Tensor,
    pblh: torch.Tensor,
    pahv: torch.Tensor,
    pahg: torch.Tensor,
    eair: torch.Tensor,
    dlwflx: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    snow_levels: torch.Tensor,
    dzsnso: torch.Tensor,
    temperature_snow_soil: torch.Tensor,
    ur: torch.Tensor,
    vai: torch.Tensor,
    z0m: torch.Tensor,
    z0mg: torch.Tensor,
    zpd: torch.Tensor,
    zlvl: torch.Tensor,
    emv: torch.Tensor,
    emg: torch.Tensor,
    gamma_ground: torch.Tensor,
    rsurf: torch.Tensor,
    rhsur: torch.Tensor,
    snowh: torch.Tensor,
    fsno: torch.Tensor,
    df: torch.Tensor,
    laisun: torch.Tensor,
    laisha: torch.Tensor,
    parsun: torch.Tensor,
    parsha: torch.Tensor,
    sav: torch.Tensor,
    sag: torch.Tensor,
    hcan: torch.Tensor,
    cwp: torch.Tensor,
    dleaf: torch.Tensor,
    cbiom: torch.Tensor,
    rgl: torch.Tensor,
    rsmin: torch.Tensor,
    rsmax: torch.Tensor,
    topt: torch.Tensor,
    hs: torch.Tensor,
    dt_tensor: torch.Tensor,
    canliq: torch.Tensor,
    canice: torch.Tensor,
    fwet: torch.Tensor,
    btran: torch.Tensor,
    thsfc_loc: bool = True,
    niter: int = 20,
    use_nested_stability_compile: bool = True,
    use_inner_compile_islands: bool = True,
    selected_gfs_v17: bool = False,
    use_sfcdif4: bool = False,
    temperature_leaf: torch.Tensor | None = None,
    temperature_ground: torch.Tensor | None = None,
    temperature_canopy_air: torch.Tensor | None = None,
    vapor_pres_canopy_air: torch.Tensor | None = None,
    prslki_input: torch.Tensor | None = None,
    prslk1_input: torch.Tensor | None = None,
    prsik1_input: torch.Tensor | None = None,
    friction_velocity: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """``vege_flux`` up to its canopy iteration loop (``opt_sfc=3,opt_crs=2`` land corridor): the loop
    state. ``_vege_flux_iteration`` runs one pass and ``_vege_flux_finish`` the rest."""

    sfctmp = temperature_forcing
    tv = tskin if temperature_leaf is None else temperature_leaf
    tg = tskin if temperature_ground is None else temperature_ground
    tah = tskin if temperature_canopy_air is None else temperature_canopy_air
    eah = torch.full_like(sfctmp, 1000.0) if vapor_pres_canopy_air is None else vapor_pres_canopy_air
    cm = cm_state
    ch = ch_state
    ustarx = torch.full_like(sfctmp, 0.1) if friction_velocity is None else friction_velocity
    fv = ustarx
    z0h = torch.clamp(z0m, min=1.0e-6)
    z0hg = torch.clamp(z0mg, min=1.0e-6)
    zero = torch.zeros_like(sfctmp)
    one = torch.ones_like(sfctmp)
    csigmaf1 = zero
    fm = zero
    fh = zero
    fh2 = zero
    hg = zero
    h = zero
    qfx = zero
    mozg = zero
    fhg = zero
    fhgh = zero
    irc = zero
    shc = zero
    evc = zero
    tr = zero
    irg = zero
    shg = zero
    evg = zero
    gh = zero
    canhs = zero
    rssun = zero
    rssha = zero
    psnsun = zero
    psnsha = zero
    rb = zero
    rahg_final = one
    cah2 = zero
    q2v = zero

    vaie = torch.clamp(vai, max=6.0)
    laisune = torch.clamp(laisun, max=6.0)
    laishae = torch.clamp(laisha, max=6.0)
    t_c = torch.clamp(tg - NOAHMP_TFRZ, min=-50.0, max=50.0)
    esatw, esati, _dsatw, _dsati = esat_noahmp(t_c)
    estg = _mask_blend(t_c > 0.0, esatw, esati)
    qsfc = CON_EPS * eah / (ps + CON_EPSM1 * eah)
    uc = ur * torch.log((hcan - zpd + z0m) / z0m) / torch.log(zlvl / z0m)
    latheav = torch.where(
        tv > NOAHMP_TFRZ,
        tv.new_full((), NOAHMP_HVAP),
        tv.new_full((), NOAHMP_HSUB),
    )
    gammav = NOAHMP_CPAIR * prsl1 / (CON_EPS * latheav)
    gammag = gamma_ground
    top_df = _gather_top_layer(df, snow_levels)
    top_dz = _gather_top_layer(dzsnso, snow_levels)
    top_stc = _gather_top_layer(temperature_snow_soil, snow_levels)
    prslk1x = torch.pow(prsl1 / 100000.0, NOAHMP_EXNER_EXP)
    prslkix = 1.0 / prslk1x
    psfc_exner = torch.pow(ps / 100000.0, NOAHMP_EXNER_EXP)
    prsik1x = 1.0 / psfc_exner
    prslk1x = prslk1x if prslk1_input is None else prslk1_input
    prslkix = prslkix if prslki_input is None else prslki_input
    prsik1x = prsik1x if prsik1_input is None else prsik1_input
    if selected_gfs_v17:
        _z0m_g_static = z0mg
        z0hg_static = torch.where(vegtype <= 5, z0mg, z0mg * 0.01)
        _csigmaf0_static = torch.zeros_like(z0mg)
    elif use_nested_stability_compile:
        _z0m_g_static, z0hg_static, _csigmaf0_static = thermalz0_bare_chen09(
            z0mg=z0mg,
            zlvl=zlvl,
            zpd=zpd,
            ustarx=ustarx,
        )
    else:
        _z0m_g_static, z0hg_static, _csigmaf0_static = _thermalz0_bare_chen09_impl(
            z0mg=z0mg,
            zlvl=zlvl,
            zpd=zpd,
            ustarx=ustarx,
        )
    czil_static = torch.pow(torch.full_like(z0m, 10.0), -0.4 * hcan)
    z0h_static = z0m * torch.exp(-czil_static * 0.4 * 258.2 * torch.sqrt(ustarx * z0m))
    _ = (_z0m_g_static, _csigmaf0_static)

    liter = torch.zeros_like(sfctmp, dtype=torch.bool)
    done = torch.zeros_like(sfctmp, dtype=torch.bool)
    # The loop chunk/step islands implement the opt_sfc=3 stability only; sfcdif4 columns iterate in vege_flux_iteration.
    use_loop_chunk_compile = (not selected_gfs_v17) and (not use_sfcdif4) and use_inner_compile_islands and "land_vege_flux_loop_chunk" in compile_categories_enabled()
    use_loop_step_compile = (not selected_gfs_v17) and (not use_sfcdif4) and use_inner_compile_islands and "land_vege_flux_loop_step" in compile_categories_enabled()
    use_update_compile = use_inner_compile_islands and "land_vege_flux_update" in compile_categories_enabled()
    use_iteration_compile = use_inner_compile_islands and "land_vege_flux_iteration" in compile_categories_enabled()
    use_finalize_compile = use_inner_compile_islands and "land_vege_flux_finalize" in compile_categories_enabled()
    if use_loop_chunk_compile and niter == 20:
        loop_common_kwargs = {
            "sfctmp": sfctmp,
            "qair": qair,
            "ps": ps,
            "prsl1": prsl1,
            "rhoair": rhoair,
            "fveg": fveg,
            "shdfac": shdfac,
            "garea": garea,
            "pahv": pahv,
            "pahg": pahg,
            "eair": eair,
            "dlwflx": dlwflx,
            "zpd": zpd,
            "z0m": z0m,
            "z0mg": z0mg,
            "hcan": hcan,
            "emv": emv,
            "emg": emg,
            "gamma_ground": gammag,
            "rsurf": rsurf,
            "rhsur": rhsur,
            "df_top": top_df,
            "dz_top": top_dz,
            "stc_top": top_stc,
            "ur": ur,
            "zlvl": zlvl,
            "vaie": vaie,
            "laisune": laisune,
            "laishae": laishae,
            "parsun": parsun,
            "parsha": parsha,
            "sav": sav,
            "sag": sag,
            "cwp": cwp,
            "dleaf": dleaf,
            "cbiom": cbiom,
            "rgl": rgl,
            "rsmin": rsmin,
            "rsmax": rsmax,
            "topt": topt,
            "hs": hs,
            "dt_tensor": dt_tensor,
            "canliq": canliq,
            "canice": canice,
            "fwet": fwet,
            "btran": btran,
            "latheav": latheav,
            "gammav": gammav,
            "uc": uc,
            "zero": zero,
            "one": one,
            "z0h_static": z0h_static,
            "z0hg_static": z0hg_static,
            "prslkix": prslkix,
            "prsik1x": prsik1x,
            "prslk1x": prslk1x,
        }
        chunk_start = 0
        selected_chunk_size = vegetation_chunk_size()
        while chunk_start < 20:
            if selected_chunk_size == 20:
                chunk_count = 20
                chunk_base = chunk_start
            else:
                # Flags change at indices 4 (convergence permitted) and 6
                # (completed points masked). After 6 the step uses the index
                # only for those constant flags; reuse that exact specialization.
                phase_end = 4 if chunk_start < 4 else 6 if chunk_start < 6 else 20
                chunk_count = min(selected_chunk_size, phase_end - chunk_start)
                chunk_base = min(chunk_start, 6)
            (
                z0h,
                z0hg,
                csigmaf1,
                fm,
                fh,
                fh2,
                fv,
                cm,
                ch,
                mozg,
                fhg,
                fhgh,
                rahg_final,
                rssun,
                rssha,
                psnsun,
                psnsha,
                tv,
                tg,
                tah,
                eah,
                irc,
                shc,
                evc,
                tr,
                irg,
                shg,
                evg,
                gh,
                canhs,
                h,
                hg,
                qsfc,
                estg,
                done,
                liter,
            ) = _vege_flux_loop_chunk(
                use_compile=True,
                base_iter_index=chunk_base,
                chunk_count=chunk_count,
                thsfc_loc=thsfc_loc,
                **loop_common_kwargs,
                liter=liter,
                done=done,
                z0h=z0h,
                z0hg=z0hg,
                csigmaf1=csigmaf1,
                fm=fm,
                fh=fh,
                fh2=fh2,
                fv=fv,
                cm=cm,
                ch=ch,
                mozg=mozg,
                fhg=fhg,
                fhgh=fhgh,
                rahg_final=rahg_final,
                rssun=rssun,
                rssha=rssha,
                psnsun=psnsun,
                psnsha=psnsha,
                tv=tv,
                tg=tg,
                tah=tah,
                eah=eah,
                irc=irc,
                shc=shc,
                evc=evc,
                tr=tr,
                irg=irg,
                shg=shg,
                evg=evg,
                gh=gh,
                canhs=canhs,
                h=h,
                hg=hg,
                qsfc=qsfc,
                estg=estg,
            )
            chunk_start += chunk_count
        niter = 0
    return {
        "btran": btran, "canhs": canhs, "canice": canice, "canliq": canliq, "cbiom": cbiom, "ch": ch,
        "cm": cm, "csigmaf1": csigmaf1, "cwp": cwp, "dleaf": dleaf, "dlwflx": dlwflx, "done": done,
        "dt_tensor": dt_tensor, "eah": eah, "eair": eair, "emg": emg, "emv": emv, "estg": estg, "evc": evc,
        "evg": evg, "fh": fh, "fh2": fh2, "fhg": fhg, "fhgh": fhgh, "fm": fm, "fsno": fsno, "fv": fv,
        "fveg": fveg, "fwet": fwet, "gamma_ground": gamma_ground, "gammag": gammag, "gammav": gammav,
        "garea": garea, "gh": gh, "h": h, "hcan": hcan, "hg": hg, "hs": hs, "irc": irc, "irg": irg,
        "laishae": laishae, "laisune": laisune, "latheav": latheav, "liter": liter, "mozg": mozg,
        "niter": niter, "one": one, "pahg": pahg, "pahv": pahv, "parsha": parsha, "parsun": parsun,
        "pblh": pblh, "prsik1x": prsik1x, "prsl1": prsl1, "prslk1x": prslk1x, "prslkix": prslkix, "ps": ps,
        "psnsha": psnsha, "psnsun": psnsun, "qair": qair, "qfx": qfx, "qsfc": qsfc, "rahg_final": rahg_final,
        "rb": rb, "rgl": rgl, "rhoair": rhoair, "rhsur": rhsur, "rsmax": rsmax, "rsmin": rsmin,
        "rssha": rssha, "rssun": rssun, "rsurf": rsurf, "sag": sag, "sav": sav,
        "selected_gfs_v17": selected_gfs_v17, "sfctmp": sfctmp, "shc": shc, "shdfac": shdfac, "shg": shg,
        "snowh": snowh, "tah": tah, "tg": tg, "thsfc_loc": thsfc_loc, "top_df": top_df, "top_dz": top_dz,
        "top_stc": top_stc, "topt": topt, "tr": tr, "tv": tv, "u1": u1, "uc": uc, "ur": ur,
        "use_finalize_compile": use_finalize_compile, "use_iteration_compile": use_iteration_compile,
        "use_loop_step_compile": use_loop_step_compile,
        "use_nested_stability_compile": use_nested_stability_compile, "use_sfcdif4": use_sfcdif4,
        "use_update_compile": use_update_compile, "v1": v1, "vaie": vaie, "z0h": z0h,
        "z0h_static": z0h_static, "z0hg": z0hg, "z0hg_static": z0hg_static, "z0m": z0m, "z0mg": z0mg,
        "zero": zero, "zlvl": zlvl, "zpd": zpd,
        # sfcdif4 (opt_sfc=4): wspdv, ch2v (/wspdv) and cq2v of the last iteration.
        "wspd4": ur, "ch2v4": zero, "cq2v4": zero,
    }


def _vege_flux_iteration(
    env: dict[str, Any], *, first_iteration: bool, can_converge: bool, use_active_mask: bool
) -> dict[str, Any]:
    """One canopy iteration pass of ``vege_flux``: ``env`` with the carried fields updated.

    Iteration 1 is the first pass; convergence may end a column from iteration 5; from iteration 7
    the completed columns keep their values. The loop gives these flags; nothing else varies."""
    (
        btran, canhs, canice, canliq, cbiom, ch, cm, csigmaf1, cwp, dleaf, dlwflx, done, dt_tensor, eah,
        eair, emg, emv, estg, evc, evg, fh, fh2, fhg, fhgh, fm, fv, fveg, fwet, gammag, gammav, garea, gh, h,
        hcan, hg, hs, irc, irg, laishae, laisune, latheav, liter, mozg, one, pahg, pahv, parsha, parsun,
        pblh, prsik1x, prsl1, prslk1x, prslkix, ps, psnsha, psnsun, qair, qfx, qsfc, rahg_final, rgl, rhoair,
        rhsur, rsmax, rsmin, rssha, rssun, rsurf, sag, sav, selected_gfs_v17, sfctmp, shc, shdfac, shg,
        snowh, tah, tg, thsfc_loc, top_df, top_dz, top_stc, topt, tr, tv, u1, uc, ur, use_iteration_compile,
        use_loop_step_compile, use_nested_stability_compile, use_sfcdif4, use_update_compile, v1, vaie, z0h,
        z0h_static, z0hg, z0hg_static, z0m, z0mg, zero, zlvl, zpd,
    ) = _env_values(env, (
        "btran canhs canice canliq cbiom ch cm csigmaf1 cwp dleaf dlwflx done dt_tensor eah eair emg emv "
        "estg evc evg fh fh2 fhg fhgh fm fv fveg fwet gammag gammav garea gh h hcan hg hs irc irg laishae "
        "laisune latheav liter mozg one pahg pahv parsha parsun pblh prsik1x prsl1 prslk1x prslkix ps psnsha "
        "psnsun qair qfx qsfc rahg_final rgl rhoair rhsur rsmax rsmin rssha rssun rsurf sag sav "
        "selected_gfs_v17 sfctmp shc shdfac shg snowh tah tg thsfc_loc top_df top_dz top_stc topt tr tv u1 "
        "uc ur use_iteration_compile use_loop_step_compile use_nested_stability_compile use_sfcdif4 "
        "use_update_compile v1 vaie z0h z0h_static z0hg z0hg_static z0m z0mg zero zlvl zpd"
    ))
    active = ~done
    z0hg_new = z0hg_static
    z0h_new = z0h_static
    csigmaf1_new = zero
    if use_loop_step_compile:
        (
            z0h,
            z0hg,
            csigmaf1,
            fm,
            fh,
            fh2,
            fv,
            cm,
            ch,
            mozg,
            fhg,
            fhgh,
            rahg_final,
            rssun,
            rssha,
            psnsun,
            psnsha,
            tv,
            tg,
            tah,
            eah,
            irc,
            shc,
            evc,
            tr,
            irg,
            shg,
            evg,
            gh,
            canhs,
            h,
            hg,
            qsfc,
            estg,
            done,
            liter,
        ) = _vege_flux_loop_step(
            use_compile=True,
            after_first=not first_iteration,
            first_iteration=first_iteration,
            can_converge=can_converge,
            use_active_mask=use_active_mask,
            thsfc_loc=thsfc_loc,
            sfctmp=sfctmp,
            qair=qair,
            ps=ps,
            prsl1=prsl1,
            rhoair=rhoair,
            fveg=fveg,
            shdfac=shdfac,
            garea=garea,
            pahv=pahv,
            pahg=pahg,
            eair=eair,
            dlwflx=dlwflx,
            zpd=zpd,
            z0m=z0m,
            z0mg=z0mg,
            hcan=hcan,
            emv=emv,
            emg=emg,
            gamma_ground=gammag,
            rsurf=rsurf,
            rhsur=rhsur,
            df_top=top_df,
            dz_top=top_dz,
            stc_top=top_stc,
            ur=ur,
            zlvl=zlvl,
            vaie=vaie,
            laisune=laisune,
            laishae=laishae,
            parsun=parsun,
            parsha=parsha,
            sav=sav,
            sag=sag,
            cwp=cwp,
            dleaf=dleaf,
            cbiom=cbiom,
            rgl=rgl,
            rsmin=rsmin,
            rsmax=rsmax,
            topt=topt,
            hs=hs,
            dt_tensor=dt_tensor,
            canliq=canliq,
            canice=canice,
            fwet=fwet,
            btran=btran,
            latheav=latheav,
            gammav=gammav,
            uc=uc,
            zero=zero,
            one=one,
            z0h_static=z0h_static,
            z0hg_static=z0hg_static,
            prslkix=prslkix,
            prsik1x=prsik1x,
            prslk1x=prslk1x,
            liter=liter,
            done=done,
            z0h=z0h,
            z0hg=z0hg,
            csigmaf1=csigmaf1,
            fm=fm,
            fh=fh,
            fh2=fh2,
            fv=fv,
            cm=cm,
            ch=ch,
            mozg=mozg,
            fhg=fhg,
            fhgh=fhgh,
            rahg_final=rahg_final,
            rssun=rssun,
            rssha=rssha,
            psnsun=psnsun,
            psnsha=psnsha,
            tv=tv,
            tg=tg,
            tah=tah,
            eah=eah,
            irc=irc,
            shc=shc,
            evc=evc,
            tr=tr,
            irg=irg,
            shg=shg,
            evg=evg,
            gh=gh,
            canhs=canhs,
            h=h,
            hg=hg,
            qsfc=qsfc,
            estg=estg,
        )
        return {**env,
                "canhs": canhs, "ch": ch, "cm": cm, "csigmaf1": csigmaf1, "done": done, "eah": eah, "estg": estg,
                "evc": evc, "evg": evg, "fh": fh, "fh2": fh2, "fhg": fhg, "fhgh": fhgh, "fm": fm, "fv": fv,
                "gh": gh, "h": h, "hg": hg, "irc": irc, "irg": irg, "liter": liter, "mozg": mozg,
                "psnsha": psnsha, "psnsun": psnsun, "qfx": qfx, "qsfc": qsfc, "rahg_final": rahg_final,
                "rssha": rssha, "rssun": rssun, "shc": shc, "shg": shg, "tah": tah, "tg": tg, "tr": tr, "tv": tv,
                "z0h": z0h, "z0hg": z0hg,
        }
    sfcdif3_args = {
        "sfctmp": sfctmp,
        "qair": qair,
        "ur": ur,
        "zlvl": zlvl,
        "tgb": tah,
        "thsfc_loc": thsfc_loc,
        "prslkix": prslkix,
        "prsik1x": prsik1x,
        "prslk1x": prslk1x,
        "z0m": z0m,
        "z0h": z0h_new,
        "zpd": zpd,
        "fveg": shdfac,
        "garea1": garea,
    }
    if use_sfcdif4:
        (
            _qsfc_exchange,
            fv_new,
            cm_speed,
            ch_speed,
            ch2_speed,
            _cq2_speed,
            _rmol,
            _rb,
            fm_new,
            fh_new,
            _stress,
            _fm10,
            _fh2_exchange,
            exchange_wind,
            _flhc,
            _flqc,
        ) = _sfcdif4_impl(
            temperature_forcing=sfctmp,
            prsl1=prsl1,
            ps=ps,
            pblh=pblh,
            garea=garea,
            u1=u1,
            v1=v1,
            qair=qair,
            tsk=tah,
            zlvl=zlvl,
            z0m=z0m,
            snowh=snowh,
            hfx=h,
            qfx=qfx,
            qsfc=qsfc,
            ustar=fv,
        )
        # vege_flux, opt_sfc=4: ch = ch/wspdv, cm = cm/wspdv, ch2v = ch2v/wspdv (cq2v unscaled);
        # rahc = max(1, 1/(ch*wspdv)) in the iteration step; fh2 = psit2 from sfcdif4.
        cm_new = cm_speed / exchange_wind
        ch_new = ch_speed / exchange_wind
        fh2_new = _fh2_exchange
        ch2v_new = ch2_speed / exchange_wind
        cq2v_new = _cq2_speed
    elif use_nested_stability_compile:
        fm_new, fh_new, fh2_new, fv_new, cm_new, ch_new = sfcdif3(**sfcdif3_args)
    else:
        fm_new, fh_new, fh2_new, fv_new, cm_new, ch_new = _sfcdif3_impl(**sfcdif3_args)
    (
        rag_mozg,
        rag_fhg,
        rag_fhgh,
        rag_rahg,
        rssun_work,
        rssha_work,
        psnsun_work,
        psnsha_work,
        tv_new,
        tg_new,
        tah_new,
        eah_new,
        irc_new,
        shc_new,
        evc_new,
        tr_new,
        irg_new,
        shg_new,
        evg_new,
        gh_new,
        canhs_new,
        h_new,
        hg_new,
        qsfc_new,
        estg_work,
        dtv,
        dtg,
    ) = _vege_flux_iteration_physics(
        use_compile=use_iteration_compile,
        after_first=not first_iteration,
        first_iteration=first_iteration,
        sfctmp=sfctmp,
        ps=ps,
        prsl1=prsl1,
        rhoair=rhoair,
        fveg=fveg,
        pahv=pahv,
        pahg=pahg,
        eair=eair,
        dlwflx=dlwflx,
        zpd=zpd,
        z0mg=z0mg,
        hcan=hcan,
        emv=emv,
        emg=emg,
        gamma_ground=gammag,
        rsurf=rsurf,
        rhsur=rhsur,
        df_top=top_df,
        dz_top=top_dz,
        stc_top=top_stc,
        ur=exchange_wind if use_sfcdif4 else ur,
        laisune=laisune,
        laishae=laishae,
        parsun=parsun,
        parsha=parsha,
        sav=sav,
        sag=sag,
        cwp=cwp,
        dleaf=dleaf,
        cbiom=cbiom,
        rgl=rgl,
        rsmin=rsmin,
        rsmax=rsmax,
        topt=topt,
        hs=hs,
        dt_tensor=dt_tensor,
        canliq=canliq,
        canice=canice,
        fwet=fwet,
        btran=btran,
        latheav=latheav,
        gammav=gammav,
        uc=uc,
        vaie=vaie,
        zero=zero,
        one=one,
        z0h_static=z0h_new,
        z0hg_static=z0hg_new,
        fv_new=fv_new,
        ch_new=ch_new,
        tah=tah,
        hg=hg,
        fv=fv,
        mozg=mozg,
        fhg=fhg,
        fhgh=fhgh,
        rssun=rssun,
        rssha=rssha,
        psnsun=psnsun,
        psnsha=psnsha,
        tv=tv,
        tg=tg,
        eah=eah,
        estg=estg,
        selected_decoupled=selected_gfs_v17,
    )

    if selected_gfs_v17:
        tg_new = tg
        irg_new = irg
        shg_new = shg
        evg_new = evg
        gh_new = gh
        estg_work = estg

    (
        z0h,
        z0hg,
        csigmaf1,
        fm,
        fh,
        fh2,
        fv,
        cm,
        ch,
        mozg,
        fhg,
        fhgh,
        rahg_final,
        rssun,
        rssha,
        psnsun,
        psnsha,
        tv,
        tg,
        tah,
        eah,
        irc,
        shc,
        evc,
        tr,
        irg,
        shg,
        evg,
        gh,
        canhs,
        h,
        hg,
        qsfc,
        estg,
    ) = _vege_flux_update_state(
        use_compile=use_update_compile,
        active=active,
        use_active_mask=use_active_mask,
        z0h=z0h,
        z0hg=z0hg,
        csigmaf1=csigmaf1,
        fm=fm,
        fh=fh,
        fh2=fh2,
        fv=fv,
        cm=cm,
        ch=ch,
        mozg=mozg,
        fhg=fhg,
        fhgh=fhgh,
        rahg_final=rahg_final,
        rssun=rssun,
        rssha=rssha,
        psnsun=psnsun,
        psnsha=psnsha,
        tv=tv,
        tg=tg,
        tah=tah,
        eah=eah,
        irc=irc,
        shc=shc,
        evc=evc,
        tr=tr,
        irg=irg,
        shg=shg,
        evg=evg,
        gh=gh,
        canhs=canhs,
        h=h,
        hg=hg,
        qsfc=qsfc,
        estg=estg,
        z0h_new=z0h_new,
        z0hg_new=z0hg_new,
        csigmaf1_new=csigmaf1_new,
        fm_new=fm_new,
        fh_new=fh_new,
        fh2_new=fh2_new,
        fv_new=fv_new,
        cm_new=cm_new,
        ch_new=ch_new,
        rag_mozg=rag_mozg,
        rag_fhg=rag_fhg,
        rag_fhgh=rag_fhgh,
        rag_rahg=rag_rahg,
        rssun_work=rssun_work,
        rssha_work=rssha_work,
        psnsun_work=psnsun_work,
        psnsha_work=psnsha_work,
        tv_new=tv_new,
        tg_new=tg_new,
        tah_new=tah_new,
        eah_new=eah_new,
        irc_new=irc_new,
        shc_new=shc_new,
        evc_new=evc_new,
        tr_new=tr_new,
        irg_new=irg_new,
        shg_new=shg_new,
        evg_new=evg_new,
        gh_new=gh_new,
        canhs_new=canhs_new,
        h_new=h_new,
        hg_new=hg_new,
        qsfc_new=qsfc_new,
        estg_work=estg_work,
    )

    completed_final_loop = active & liter
    if selected_gfs_v17:
        converged = active & (~liter) & can_converge & (torch.abs(dtv) <= 0.01)
    else:
        converged = active & (~liter) & can_converge & (torch.abs(dtv) <= 0.01) & (torch.abs(dtg) <= 0.01)
    done = done | completed_final_loop
    liter = liter | converged
    if use_sfcdif4:
        wspd4 = torch.where(active, exchange_wind, env["wspd4"])
        ch2v4 = torch.where(active, ch2v_new, env["ch2v4"])
        cq2v4 = torch.where(active, cq2v_new, env["cq2v4"])
        # qfx = (qsfc-qair)*rhoair*caw, caw = 1/rawc, rawc = rahc = max(1, 1/(ch*wspdv))
        qfx = (qsfc - qair) * rhoair * (1.0 / torch.maximum(one, 1.0 / (ch * wspd4)))
        env = {**env, "wspd4": wspd4, "ch2v4": ch2v4, "cq2v4": cq2v4}
    return {**env,
            "canhs": canhs, "ch": ch, "cm": cm, "csigmaf1": csigmaf1, "done": done, "eah": eah, "estg": estg,
            "evc": evc, "evg": evg, "fh": fh, "fh2": fh2, "fhg": fhg, "fhgh": fhgh, "fm": fm, "fv": fv,
            "gh": gh, "h": h, "hg": hg, "irc": irc, "irg": irg, "liter": liter, "mozg": mozg,
            "psnsha": psnsha, "psnsun": psnsun, "qfx": qfx, "qsfc": qsfc, "rahg_final": rahg_final,
            "rssha": rssha, "rssun": rssun, "shc": shc, "shg": shg, "tah": tah, "tg": tg, "tr": tr, "tv": tv,
            "z0h": z0h, "z0hg": z0hg,
    }


def _vege_flux_finish(env: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    """``vege_flux`` after its canopy iteration loop."""
    (
        canhs, ch, cm, csigmaf1, cwp, dleaf, dlwflx, eah, emg, emv, estg, evc, evg, fh, fh2, fhg, fm, fsno,
        fv, fveg, gamma_ground, gammag, gh, h, hcan, irc, irg, latheav, pahg, psnsha, psnsun, qsfc,
        rahg_final, rb, rhoair, rhsur, rssha, rssun, rsurf, sag, selected_gfs_v17, shc, shg, snowh, tah, tg,
        top_df, top_dz, top_stc, tr, tv, u1, uc, ur, use_finalize_compile, v1, vaie, z0h, z0hg,
    ) = _env_values(env, (
        "canhs ch cm csigmaf1 cwp dleaf dlwflx eah emg emv estg evc evg fh fh2 fhg fm fsno fv fveg "
        "gamma_ground gammag gh h hcan irc irg latheav pahg psnsha psnsun qsfc rahg_final rb rhoair rhsur "
        "rssha rssun rsurf sag selected_gfs_v17 shc shg snowh tah tg top_df top_dz top_stc tr tv u1 uc ur "
        "use_finalize_compile v1 vaie z0h z0hg"
    ))
    if selected_gfs_v17:
        air_g = -emg * (1.0 - emv) * dlwflx - emg * emv * NOAHMP_SB * _pow4(tv)
        cir_g = emg * NOAHMP_SB
        csh_ground = rhoair * NOAHMP_CPAIR / rahg_final
        cev_ground = rhoair * NOAHMP_CPAIR / (gammag * (rahg_final + rsurf))
        cgh_ground = 2.0 * top_df / top_dz
        for _ in range(5):
            t_cg = torch.clamp(tg - NOAHMP_TFRZ, min=-50.0, max=50.0)
            esatw_g, esati_g, dsatw_g, dsati_g = esat_noahmp(t_cg)
            warm_tg = t_cg > 0.0
            estg = _mask_blend(warm_tg, esatw_g, esati_g)
            destg = _mask_blend(warm_tg, dsatw_g, dsati_g)
            tg_pow4 = _pow4(tg)
            tg_pow3 = _pow3(tg)
            irg = cir_g * tg_pow4 + air_g
            shg = csh_ground * (tg - tah)
            evg = cev_ground * (estg * rhsur - eah)
            gh = cgh_ground * (tg - top_stc)
            balance_g = sag - irg - shg - evg - gh + pahg
            jac_g = 4.0 * cir_g * tg_pow3 + csh_ground + cev_ground * destg + cgh_ground
            dtg = balance_g / jac_g
            irg = irg + 4.0 * cir_g * tg_pow3 * dtg
            shg = shg + csh_ground * dtg
            evg = evg + cev_ground * destg * dtg
            gh = gh + cgh_ground * dtg
            tg = tg + dtg

    _ = (fm, fh, h)
    # Reconstruct the final RAGRB resistance with its real-exponent operation.
    cwpc_base = torch.clamp(cwp * vaie * hcan * fhg, min=1.0)
    cwpc_final = torch.clamp(working_power(cwpc_base, torch.full_like(cwpc_base, 0.5)), min=1.0, max=5.0)
    rb = torch.clamp(cwpc_final * 50.0 / (1.0 - torch.exp(-cwpc_final / 2.0)) * torch.sqrt(dleaf / uc), min=5.0, max=50.0)
    return _vege_flux_finalize(
        use_compile=use_finalize_compile,
        gamma_ground=gamma_ground,
        rsurf=rsurf,
        rhsur=rhsur,
        estg=estg,
        snowh=snowh,
        tg=tg,
        fsno=fsno,
        tv=tv,
        emg=emg,
        emv=emv,
        dlwflx=dlwflx,
        shg=shg,
        evg=evg,
        sag=sag,
        pahg=pahg,
        rhoair=rhoair,
        cm=cm,
        ur=ur,
        u1=u1,
        v1=v1,
        fv=fv,
        fh2=fh2,
        tah=tah,
        shc=shc,
        fveg=fveg,
        qsfc=qsfc,
        evc=evc,
        tr=tr,
        latheav=latheav,
        ch=ch,
        vaie=vaie,
        rb=rb,
        rahg_final=rahg_final,
        eah=eah,
        rssun=rssun,
        rssha=rssha,
        psnsun=psnsun,
        psnsha=psnsha,
        irg=irg,
        irc=irc,
        gh=gh,
        canhs=canhs,
        z0h=z0h,
        z0hg=z0hg,
        csigmaf1=csigmaf1,
        sfcdif4_wspd=env["wspd4"] if env["use_sfcdif4"] else None,
        sfcdif4_ch2v=env["ch2v4"] if env["use_sfcdif4"] else None,
        sfcdif4_cq2v=env["cq2v4"] if env["use_sfcdif4"] else None,
    )


def _env_values(env: dict[str, Any], names: str) -> tuple[Any, ...]:
    return tuple(env[name] for name in names.split())


def _vege_flux_iterations(env: dict[str, Any], step: Any = None) -> dict[str, Any]:
    """The canopy iteration loop of ``vege_flux``: ``env["niter"]`` passes of ``step`` (default
    ``_vege_flux_iteration``; a caller may pass its compiled form)."""
    step = _vege_flux_iteration if step is None else step
    for iter_index in range(env["niter"]):
        env = step(env, first_iteration=iter_index == 0, can_converge=iter_index >= 4, use_active_mask=iter_index >= 6)
    return env


def _vege_flux_opt_sfc3_flat_impl(**kwargs: Any) -> tuple[torch.Tensor, ...]:
    """Port ``vege_flux`` for the sealed ``opt_sfc=3,opt_crs=2`` land corridor (``_vege_flux_setup`` arguments)."""
    return _vege_flux_finish(_vege_flux_iterations(_vege_flux_setup(**kwargs)))


def _gfs_stability_impl(
    *,
    z1: torch.Tensor,
    tv1: torch.Tensor,
    thv1: torch.Tensor,
    wind: torch.Tensor,
    z0max: torch.Tensor,
    ztmax: torch.Tensor,
    tvs: torch.Tensor,
    thsfc_loc: bool,
    gravity: float | torch.Tensor = NOAHMP_GRAV,
    zvfun: torch.Tensor | None = None,
    gdx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized GFS stability routine used by ``sfcdif3``."""

    ca = 0.4
    alpha4 = 20.0
    xkrefsqr = 0.3
    zolmin = -10.0
    z1i = 1.0 / z1
    xkzo = torch.ones_like(z1) if gdx is None else torch.minimum(gdx / 3000.0, torch.ones_like(gdx))
    if zvfun is not None:
        xkzo = torch.where(tv1 > tvs, torch.minimum(torch.clamp(xkzo * zvfun, min=0.05), xkzo), xkzo)
    zolmax = torch.full_like(xkzo, xkrefsqr) / torch.sqrt(xkzo)

    dtv = thv1 - tvs
    adtv = torch.clamp(torch.abs(dtv), min=0.001)
    dtv = torch.sign(dtv) * adtv
    dtv = torch.where(dtv == 0.0, adtv, dtv)
    grav = gravity if isinstance(gravity, torch.Tensor) else float(gravity)
    if thsfc_loc:
        rb = torch.clamp(2.0 * grav * dtv * z1 / ((thv1 + tvs) * wind * wind), min=-5000.0)
    else:
        rb = torch.clamp(grav * dtv * z1 / (tv1 * wind * wind), min=-5000.0)

    z0mi = torch.reciprocal(z0max)
    z0hi = torch.reciprocal(ztmax)
    fm_neu = torch.log((z0max + z1) * z0mi)
    fh_neu = torch.log((ztmax + z1) * z0hi)
    fm10_neu = torch.log((z0max + 10.0) * z0mi)
    fh2_neu = torch.log((ztmax + 2.0) * z0hi)
    hlinf = rb * fm_neu * fm_neu / fh_neu
    hlinf = torch.minimum(torch.clamp(hlinf, min=zolmin), zolmax)

    stable = dtv >= 0.0
    hlinf_st = torch.clamp(hlinf, min=0.0)
    hl1 = hlinf_st
    high_stable = hlinf_st > 0.25
    tem1 = hlinf_st * z1i
    hl0inf = z0max * tem1
    hltinf = ztmax * tem1
    aa = torch.sqrt(1.0 + alpha4 * hlinf_st)
    aa0 = torch.sqrt(1.0 + alpha4 * hl0inf)
    bb = aa
    bb0 = torch.sqrt(1.0 + alpha4 * hltinf)
    pm = aa0 - aa + torch.log((aa + 1.0) / (aa0 + 1.0))
    ph = bb0 - bb + torch.log((bb + 1.0) / (bb0 + 1.0))
    fms = fm_neu - pm
    fhs = fh_neu - ph
    hl1_iter = torch.minimum(fms * fms * rb / fhs, zolmax)
    hl1 = torch.where(high_stable, hl1_iter, hl1)
    tem1 = hl1 * z1i
    hl0 = z0max * tem1
    hlt = ztmax * tem1
    aa = torch.sqrt(1.0 + alpha4 * hl1)
    aa0 = torch.sqrt(1.0 + alpha4 * hl0)
    bb = aa
    bb0 = torch.sqrt(1.0 + alpha4 * hlt)
    pm_st = aa0 - aa + torch.log((1.0 + aa) / (1.0 + aa0))
    ph_st = bb0 - bb + torch.log((1.0 + bb) / (1.0 + bb0))
    hl110 = hl1 * 10.0 * z1i
    aa10 = torch.sqrt(1.0 + alpha4 * hl110)
    pm10_st = aa0 - aa10 + torch.log((1.0 + aa10) / (1.0 + aa0))
    hl12 = (hl1 + hl1) * z1i
    bb2 = torch.sqrt(1.0 + alpha4 * hl12)
    ph2_st = bb0 - bb2 + torch.log((1.0 + bb2) / (1.0 + bb0))

    hlinf_un = torch.clamp(hlinf, max=-1.0e-12)
    olinf = z1 / hlinf_un
    tem1 = 50.0 * z0max
    hlinf_un = torch.where(torch.abs(olinf) <= tem1, torch.clamp(-z1 / tem1, min=zolmin), hlinf_un)
    mild = hlinf_un >= -0.5
    hl1_mild = hlinf_un
    pm_mild = (-3.975 + 12.32 * hl1_mild) * hl1_mild / (1.0 + (-7.755 + 6.041 * hl1_mild) * hl1_mild)
    ph_mild = (-7.941 + 24.75 * hl1_mild) * hl1_mild / (1.0 + (-8.705 + 7.899 * hl1_mild) * hl1_mild)
    hl110_mild = hl1_mild * 10.0 * z1i
    pm10_mild = (-3.975 + 12.32 * hl110_mild) * hl110_mild / (1.0 + (-7.755 + 6.041 * hl110_mild) * hl110_mild)
    hl12_mild = (hl1_mild + hl1_mild) * z1i
    ph2_mild = (-7.941 + 24.75 * hl12_mild) * hl12_mild / (1.0 + (-8.705 + 7.899 * hl12_mild) * hl12_mild)

    hl1_strong = torch.clamp(-hlinf_un, min=1.0e-12)
    tem1 = 1.0 / torch.sqrt(hl1_strong)
    pm_strong = torch.log(hl1_strong) + 2.0 * torch.sqrt(tem1) - 0.8776
    ph_strong = torch.log(hl1_strong) + 0.5 * tem1 + 1.386
    hl110_strong = torch.clamp(hl1_strong * 10.0 * z1i, min=1.0e-12)
    pm10_strong = (
        torch.log(hl110_strong)
        + torch.full_like(hl110_strong, 2.0) / torch.sqrt(torch.sqrt(hl110_strong))
        - 0.8776
    )
    hl12_strong = torch.clamp((hl1_strong + hl1_strong) * z1i, min=1.0e-12)
    ph2_strong = (
        torch.log(hl12_strong)
        + torch.full_like(hl12_strong, 0.5) / torch.sqrt(hl12_strong)
        + 1.386
    )

    pm_un = torch.where(mild, pm_mild, pm_strong)
    ph_un = torch.where(mild, ph_mild, ph_strong)
    pm10_un = torch.where(mild, pm10_mild, pm10_strong)
    ph2_un = torch.where(mild, ph2_mild, ph2_strong)

    pm = torch.where(stable, pm_st, pm_un)
    ph = torch.where(stable, ph_st, ph_un)
    pm10 = torch.where(stable, pm10_st, pm10_un)
    ph2_corr = torch.where(stable, ph2_st, ph2_un)
    fm = fm_neu - pm
    fh = fh_neu - ph
    fm10 = fm10_neu - pm10
    fh2 = fh2_neu - ph2_corr
    # Tensor numerators retain Fortran division, not scalar rdiv's reciprocal product.
    minimum = torch.full_like(z1, 0.00001) / z1
    ca_squared = torch.full_like(fm, ca * ca)
    cm = torch.clamp(ca_squared / (fm * fm), min=minimum)
    ch = torch.clamp(ca_squared / (fm * fh), min=minimum)
    stress = cm * wind * wind
    ustar = torch.sqrt(stress)
    return rb, fm, fh, fm10, fh2, cm, ch, stress, ustar


def gfs_stability(
    *,
    z1: torch.Tensor,
    tv1: torch.Tensor,
    thv1: torch.Tensor,
    wind: torch.Tensor,
    z0max: torch.Tensor,
    ztmax: torch.Tensor,
    tvs: torch.Tensor,
    thsfc_loc: bool,
    gravity: float | torch.Tensor = NOAHMP_GRAV,
    zvfun: torch.Tensor | None = None,
    gdx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return run_compile_island(
        "land_stability",
        "gfs_stability",
        _gfs_stability_impl,
        z1=z1,
        tv1=tv1,
        thv1=thv1,
        wind=wind,
        z0max=z0max,
        ztmax=ztmax,
        tvs=tvs,
        thsfc_loc=thsfc_loc,
        gravity=gravity,
        zvfun=zvfun,
        gdx=gdx,
    )


def _sfcdif3_impl(
    *,
    sfctmp: torch.Tensor,
    qair: torch.Tensor,
    ur: torch.Tensor,
    zlvl: torch.Tensor,
    tgb: torch.Tensor,
    thsfc_loc: bool,
    prslkix: torch.Tensor,
    prsik1x: torch.Tensor,
    prslk1x: torch.Tensor,
    z0m: torch.Tensor,
    z0h: torch.Tensor,
    zpd: torch.Tensor,
    fveg: torch.Tensor,
    garea1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fortran ``sfcdif3`` with the hard-coded GFS stability call."""

    virtfac = 1.0 + 0.61 * torch.clamp(qair, min=1.0e-8)
    tv1 = sfctmp * virtfac
    thv1 = sfctmp * prslkix * virtfac if thsfc_loc else sfctmp / prslk1x * virtfac
    _ = (fveg, garea1)
    tvs = tgb * virtfac if thsfc_loc else tgb / prsik1x * virtfac
    _rb, fm, fh, _fm10, fh2, cm, ch, _stress, fv = _gfs_stability_impl(
        z1=zlvl - zpd,
        tv1=tv1,
        thv1=thv1,
        wind=ur,
        z0max=z0m,
        ztmax=z0h,
        tvs=tvs,
        thsfc_loc=thsfc_loc,
    )
    return fm, fh, fh2, fv, cm, ch


def sfcdif3(
    *,
    sfctmp: torch.Tensor,
    qair: torch.Tensor,
    ur: torch.Tensor,
    zlvl: torch.Tensor,
    tgb: torch.Tensor,
    thsfc_loc: bool,
    prslkix: torch.Tensor,
    prsik1x: torch.Tensor,
    prslk1x: torch.Tensor,
    z0m: torch.Tensor,
    z0h: torch.Tensor,
    zpd: torch.Tensor,
    fveg: torch.Tensor,
    garea1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return run_compile_island(
        "land_stability",
        "sfcdif3",
        _sfcdif3_impl,
        sfctmp=sfctmp,
        qair=qair,
        ur=ur,
        zlvl=zlvl,
        tgb=tgb,
        thsfc_loc=thsfc_loc,
        prslkix=prslkix,
        prsik1x=prsik1x,
        prslk1x=prslk1x,
        z0m=z0m,
        z0h=z0h,
        zpd=zpd,
        fveg=fveg,
        garea1=garea1,
    )


def _mynn_psim_stable_full(zolf: torch.Tensor) -> torch.Tensor:
    return -6.1 * torch.log(zolf + torch.pow(1.0 + torch.pow(zolf, 2.5), 0.4))


def _mynn_psih_stable_full(zolf: torch.Tensor) -> torch.Tensor:
    return -5.3 * torch.log(zolf + torch.pow(1.0 + torch.pow(zolf, 1.1), 0.9090909090909090909))


def _mynn_psim_unstable_full(zolf: torch.Tensor) -> torch.Tensor:
    # module_sf_noahmplsm: onethird = 1./3. multiplies, as in the source (not a division by 3).
    atan1 = 0.785398163397
    sqrt3 = 1.7320508075688773
    x = torch.pow(1.0 - 16.0 * zolf, 0.25)
    psimk = 2.0 * torch.log(0.5 * (1.0 + x)) + torch.log(0.5 * (1.0 + x * x)) - 2.0 * torch.atan(x) + 2.0 * atan1
    ym = torch.pow(1.0 - 10.0 * zolf, 1.0 / 3.0)
    psimc = 1.5 * torch.log((ym * ym + ym + 1.0) * (1.0 / 3.0)) - sqrt3 * torch.atan((2.0 * ym + 1.0) / sqrt3) + 4.0 * atan1 / sqrt3
    zolf2 = zolf * zolf
    return (psimk + zolf2 * psimc) / (1.0 + zolf2)


def _mynn_psih_unstable_full(zolf: torch.Tensor) -> torch.Tensor:
    atan1 = 0.785398163397
    sqrt3 = 1.7320508075688773
    # Fortran (1.-16.*zolf)**.5 is libm pow; a scalar 0.5 exponent would run as sqrt in ATen.
    y_base = 1.0 - 16.0 * zolf
    y = torch.pow(y_base, torch.full_like(y_base, 0.5))
    psihk = 2.0 * torch.log(0.5 * (1.0 + y))
    yh = torch.pow(1.0 - 34.0 * zolf, 1.0 / 3.0)
    psihc = 1.5 * torch.log((yh * yh + yh + 1.0) * (1.0 / 3.0)) - sqrt3 * torch.atan((2.0 * yh + 1.0) / sqrt3) + 4.0 * atan1 / sqrt3
    zolf2 = zolf * zolf
    return (psihk + zolf2 * psihc) / (1.0 + zolf2)


def _mynn_psi_lookup(
    zolf: torch.Tensor,
    *,
    heat: bool,
    stable: bool,
    tables_initialized: bool,
) -> torch.Tensor:
    scaled = zolf * 100.0 if stable else -zolf * 100.0
    index = torch.trunc(scaled).to(torch.int64)
    fraction = scaled - index.to(zolf.dtype)
    base_zolf = index.to(zolf.dtype) * (0.01 if stable else -0.01)
    next_zolf = base_zolf + (0.01 if stable else -0.01)
    full = (
        _mynn_psih_stable_full if stable and heat else
        _mynn_psim_stable_full if stable else
        _mynn_psih_unstable_full if heat else
        _mynn_psim_unstable_full
    )
    if tables_initialized:
        interpolated = full(base_zolf) + fraction * (full(next_zolf) - full(base_zolf))
    else:
        # In the selected GFS-v17 profile lifecycle ``psi_init`` is guarded by
        # do_mynnsfclay=.false.; the static lookup arrays therefore retain their
        # Fortran zero initialization even when iopt_sfc=4 is selected.
        interpolated = torch.zeros_like(zolf)
    uses_lookup = index + 1 < 1000
    # Fortran evaluates the full formula only outside the lookup range. An
    # unused stable formula at negative z/l otherwise poisons reverse mode.
    fallback_zolf = torch.where(uses_lookup, 1.0 if stable else -1.0, zolf)
    return torch.where(uses_lookup, interpolated, full(fallback_zolf))


def _li_etal_2010(rib: torch.Tensor, zaz0: torch.Tensor, z0zt: torch.Tensor) -> torch.Tensor:
    alfa = torch.log(torch.clamp(zaz0, min=100.0, max=100000.0))
    beta = torch.log(torch.clamp(z0zt, min=0.5, max=100.0))
    unstable = 0.045 * alfa * rib * rib + (
        (0.003 * beta + 0.0059) * alfa * alfa
        + (-0.0828 * beta + 0.8845) * alfa
        + (0.1739 * beta * beta - 0.9213 * beta - 0.1057)
    ) * rib
    weak_stable = ((0.5738 * beta - 0.4399) * alfa + (-4.901 * beta + 52.50)) * rib * rib + (
        (-0.0539 * beta + 1.540) * alfa + (-0.669 * beta - 3.282)
    ) * rib
    strong_stable = (0.7529 * alfa + 14.94) * rib + 0.1569 * alfa - 0.3091 * beta - 1.303
    return torch.where(
        rib <= 0.0,
        torch.clamp(unstable, min=-15.0, max=0.0),
        torch.where(rib <= 0.2, torch.clamp(weak_stable, min=0.0, max=4.0), torch.clamp(strong_stable, min=1.0, max=20.0)),
    )


def _zolrib(
    rib: torch.Tensor,
    za: torch.Tensor,
    z0: torch.Tensor,
    zt: torch.Tensor,
    logz0: torch.Tensor,
    logzt: torch.Tensor,
    initial: torch.Tensor,
    *,
    tables_initialized: bool,
) -> torch.Tensor:
    zol = torch.where(initial * rib < 0.0, torch.zeros_like(initial), initial)
    previous = torch.where(rib < 0.0, torch.full_like(zol, -99999.0), torch.full_like(zol, 99999.0))
    result = torch.where(rib < 0.0, torch.full_like(zol, -66666.0), torch.full_like(zol, 66666.0))
    active = torch.ones_like(rib, dtype=torch.bool)
    for iteration in range(19):
        old = zol if iteration == 0 else result
        previous = torch.where(active, old, previous)
        zol20 = old * z0 / za
        zol3 = old + zol20
        zolt = old * zt / za
        unstable_psit = torch.clamp(logzt - (_mynn_psi_lookup(zol3, heat=True, stable=False, tables_initialized=tables_initialized) - _mynn_psi_lookup(zolt, heat=True, stable=False, tables_initialized=tables_initialized)), min=1.0)
        unstable_psix = torch.clamp(logz0 - (_mynn_psi_lookup(zol3, heat=False, stable=False, tables_initialized=tables_initialized) - _mynn_psi_lookup(zol20, heat=False, stable=False, tables_initialized=tables_initialized)), min=1.0)
        stable_psit = torch.clamp(logzt - (_mynn_psi_lookup(zol3, heat=True, stable=True, tables_initialized=tables_initialized) - _mynn_psi_lookup(zolt, heat=True, stable=True, tables_initialized=tables_initialized)), min=1.0)
        stable_psix = torch.clamp(logz0 - (_mynn_psi_lookup(zol3, heat=False, stable=True, tables_initialized=tables_initialized) - _mynn_psi_lookup(zol20, heat=False, stable=True, tables_initialized=tables_initialized)), min=1.0)
        psit = torch.where(rib < 0.0, unstable_psit, stable_psit)
        psix = torch.where(rib < 0.0, unstable_psix, stable_psix)
        candidate = rib * (psix * psix) / psit  # zolrib: ri*psix2**2/psit2
        result = torch.where(active, candidate, result)
        active = active & (torch.abs(old - candidate) > 0.01)
    fallback = _li_etal_2010(rib, za / z0, z0 / zt)
    return torch.where(active & (torch.abs(previous - result) > 0.01), fallback, result)


def _sfcdif4_impl(
    *,
    temperature_forcing: torch.Tensor,
    prsl1: torch.Tensor,
    ps: torch.Tensor,
    pblh: torch.Tensor,
    garea: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    qair: torch.Tensor,
    tsk: torch.Tensor,
    zlvl: torch.Tensor,
    z0m: torch.Tensor,
    snowh: torch.Tensor,
    hfx: torch.Tensor,
    qfx: torch.Tensor,
    qsfc: torch.Tensor,
    ustar: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Vectorized historical Noah-MP/MYNN ``sfcdif4`` for land, ``itime=1``.

    A Python number over a tensor (``c / x``) runs in eager torch as ``x.reciprocal() * c``,
    one rounding more than Fortran's ``c/x``; such quotients use ``torch.div(c, x)``.
    """

    psfc_kpa = ps / 1000.0
    tabs = 0.5 * (tsk + temperature_forcing)
    e_ice = 0.6112 * torch.exp(4648.0 * (1.0 / 273.15 - torch.div(1.0, tabs)) - 11.64 * torch.log(torch.div(273.15, tabs)) + 0.02265 * (273.15 - tabs))
    e_water = 0.6112 * torch.exp(17.67 * (tabs - 273.15) / (tabs - 29.65))
    e1 = torch.where(tabs < 273.15, e_ice, e_water)
    qsfc = CON_EPS * e1 / (psfc_kpa - (1.0 - CON_EPS) * e1)

    exponent = NOAHMP_RAIR / NOAHMP_CPAIR
    thgb = tsk * torch.pow(torch.div(100000.0, ps), exponent)
    # sfcdif4: pl = p1d/1000.; thx = t1d*(p1000mb*0.001/pl)**(rair/cpair), p1000mb*0.001 == 100.
    thx = temperature_forcing * torch.pow(torch.div(100.0, prsl1 / 1000.0), exponent)
    thvx = thx * (1.0 + CON_FVIRT * qair)
    tvir = temperature_forcing * (1.0 + CON_FVIRT * qair)
    rho = psfc_kpa * 1000.0 / (NOAHMP_RAIR * tvir)  # sfcdif4: rhox=psfcx*1000./(rair*tvir)
    za = zlvl
    t1dc = temperature_forcing - 273.15
    e1_air_ice = 0.6112 * torch.exp(4648.0 * (1.0 / 273.15 - torch.div(1.0, temperature_forcing)) - 11.64 * torch.log(torch.div(273.15, temperature_forcing)) + 0.02265 * (273.15 - temperature_forcing))
    e1_air_water = 0.6112 * torch.exp(17.67 * (temperature_forcing - 273.15) / (temperature_forcing - 29.65))
    e1_air = torch.where(temperature_forcing < 273.15, e1_air_ice, e1_air_water)
    _qgh = CON_EPS * e1_air / (prsl1 / 1000.0 - e1_air)
    cpm = CON_CP * (1.0 + 0.84 * qair / (1.0 - qair))
    wind = torch.sqrt(u1 * u1 + v1 * v1)
    tskv = thgb * (1.0 + CON_FVIRT * qsfc)
    dthvdz = thvx - tskv
    fluxc = torch.clamp(hfx / rho / CON_CP + CON_FVIRT * tskv * qfx / rho, min=0.0)
    convective_base = torch.div(NOAHMP_GRAV, tsk) * torch.minimum(1.5 * pblh, torch.full_like(pblh, 4000.0)) * fluxc
    nonzero_convection = convective_base != 0.0
    vconv = torch.where(nonzero_convection, 1.25 * torch.pow(torch.where(nonzero_convection, convective_base, 1.0), 0.33), 0.0)
    dx = torch.sqrt(garea)
    gustiness_base = torch.clamp(dx / 5000.0 - 1.0, min=0.0)
    nonzero_gustiness = gustiness_base != 0.0
    vsgd = torch.clamp(torch.where(nonzero_gustiness, 0.32 * torch.pow(torch.where(nonzero_gustiness, gustiness_base, 1.0), 0.33), 0.0), max=0.5)
    wind = torch.clamp(torch.sqrt(wind * wind + vconv * vconv + vsgd * vsgd), min=0.1)
    rib = torch.clamp(torch.div(NOAHMP_GRAV, thx) * za * dthvdz / (wind * wind), min=-2.0, max=2.0)
    visc = 1.326e-5 * (1.0 + 6.542e-3 * t1dc + 8.301e-6 * t1dc * t1dc - 4.84e-9 * t1dc * t1dc * t1dc)
    restar = torch.clamp(ustar * z0m / visc, min=0.1)
    zt = torch.minimum(z0m * torch.exp(-NOAHMP_VKC * 0.085 * torch.sqrt(restar)), 0.75 * z0m)
    zq = zt
    gz1oz0 = torch.log((za + z0m) / z0m)
    gz1ozt = torch.log((za + z0m) / zt)
    gz2oz0 = torch.log((2.0 + z0m) / z0m)
    gz2ozt = torch.log((2.0 + z0m) / zt)
    gz10oz0 = torch.log((10.0 + z0m) / z0m)
    initial_zol = _li_etal_2010(rib, za / z0m, z0m / zt)
    zol = torch.clamp(
        _zolrib(
            rib,
            za,
            z0m,
            zt,
            gz1oz0,
            gz1ozt,
            initial_zol,
            tables_initialized=False,
        ),
        min=-20.0,
        max=20.0,
    )
    zolzt = zol * zt / za
    zolz0 = zol * z0m / za
    zolza = zol * (za + z0m) / za
    zol10 = zol * (10.0 + z0m) / za
    zol2 = zol * (2.0 + z0m) / za
    stable = rib > 0.0
    unstable = rib < 0.0
    lookup = False
    psim_st = _mynn_psi_lookup(zolza, heat=False, stable=True, tables_initialized=lookup) - _mynn_psi_lookup(zolz0, heat=False, stable=True, tables_initialized=lookup)
    psih_st = _mynn_psi_lookup(zolza, heat=True, stable=True, tables_initialized=lookup) - _mynn_psi_lookup(zolzt, heat=True, stable=True, tables_initialized=lookup)
    psim10_st = _mynn_psi_lookup(zol10, heat=False, stable=True, tables_initialized=lookup) - _mynn_psi_lookup(zolz0, heat=False, stable=True, tables_initialized=lookup)
    psih2_st = _mynn_psi_lookup(zol2, heat=True, stable=True, tables_initialized=lookup) - _mynn_psi_lookup(zolzt, heat=True, stable=True, tables_initialized=lookup)
    psim_un = _mynn_psi_lookup(zolza, heat=False, stable=False, tables_initialized=lookup) - _mynn_psi_lookup(zolz0, heat=False, stable=False, tables_initialized=lookup)
    psih_un = torch.minimum(_mynn_psi_lookup(zolza, heat=True, stable=False, tables_initialized=lookup) - _mynn_psi_lookup(zolzt, heat=True, stable=False, tables_initialized=lookup), 0.9 * gz1ozt)
    psim_un = torch.minimum(psim_un, 0.9 * gz1oz0)
    psim10_un = torch.minimum(_mynn_psi_lookup(zol10, heat=False, stable=False, tables_initialized=lookup) - _mynn_psi_lookup(zolz0, heat=False, stable=False, tables_initialized=lookup), 0.9 * gz10oz0)
    psih2_un = torch.minimum(_mynn_psi_lookup(zol2, heat=True, stable=False, tables_initialized=lookup) - _mynn_psi_lookup(zolzt, heat=True, stable=False, tables_initialized=lookup), 0.9 * gz2ozt)
    zero = torch.zeros_like(rib)
    psim = torch.where(stable, psim_st, torch.where(unstable, psim_un, zero))
    psih = torch.where(stable, psih_st, torch.where(unstable, psih_un, zero))
    psim10 = torch.where(stable, psim10_st, torch.where(unstable, psim10_un, zero))
    psih2 = torch.where(stable, psih2_st, torch.where(unstable, psih2_un, zero))
    rmol = torch.where(rib == 0.0, zero, zol / za)
    psix = torch.clamp(gz1oz0 - psim, min=1.0)
    psix10 = torch.clamp(gz10oz0 - psim10, min=1.0)
    psit = torch.clamp(gz1ozt - psih, min=1.0)
    psit2 = torch.clamp(gz2ozt - psih2, min=1.0)
    psiq = torch.clamp(torch.log((za + zq) / zq) - psih, min=1.0)
    psiq2 = torch.clamp(torch.log((2.0 + zq) / zq) - psih2, min=1.0)
    ustar = torch.clamp(0.5 * ustar + 0.5 * NOAHMP_VKC * wind / psix, min=0.005)
    cm_speed = torch.div(NOAHMP_VKC, psix) * torch.div(NOAHMP_VKC, psix) * wind
    ch_speed = ustar * NOAHMP_VKC / psit
    cq2_speed = ustar * NOAHMP_VKC / psiq2
    ch2_speed = ustar * NOAHMP_VKC / psit2
    flhc = rho * cpm * ustar * NOAHMP_VKC / psit
    flqc = rho * ustar * NOAHMP_VKC / psiq
    stress = ustar * ustar
    return qsfc, ustar, cm_speed, ch_speed, ch2_speed, cq2_speed, rmol, rib, psix, psit, stress, psix10, psit2, wind, flhc, flqc


def bare_flux_opt_sfc3(
    state: Any,
    geometry: LandEnergyGeometry,
    *,
    sag: torch.Tensor,
    df: torch.Tensor,
    dt: torch.Tensor | float,
    tgb: torch.Tensor | None = None,
    cm: torch.Tensor | None = None,
    ch: torch.Tensor | None = None,
    ustarx: torch.Tensor | None = None,
    thsfc_loc: bool = True,
    niter: int = 5,
    selected_gfs_v17: bool = False,
    surface_exchange_option: int = 3,
) -> BareFluxResult:
    """Port ``bare_flux`` for ``opt_sfc=3`` and ``opt_stc=3``.

    ``df`` is the snow/soil thermal conductivity from ``thermoprop`` with shape
    ``[ncol, 7]`` for ``nsnow=3,nsoil=4``.  The current helper intentionally
    keeps thermoprop outside this module.
    """
    _ = dt
    return _bare_flux_result(
        _bare_flux_opt_sfc3_flat_impl(
            **_bare_flux_flat_kwargs(
                state,
                geometry,
                sag=sag,
                df=df,
                tgb=tgb,
                cm=cm,
                ch=ch,
                ustarx=ustarx,
                thsfc_loc=thsfc_loc,
                niter=niter,
                selected_gfs_v17=selected_gfs_v17,
                surface_exchange_option=surface_exchange_option,
            )
        )
    )


def _bare_flux_flat_kwargs(
    state: Any,
    geometry: LandEnergyGeometry,
    *,
    sag: torch.Tensor,
    df: torch.Tensor,
    tgb: torch.Tensor | None,
    cm: torch.Tensor | None,
    ch: torch.Tensor | None,
    ustarx: torch.Tensor | None,
    thsfc_loc: bool,
    niter: int,
    use_nested_stability_compile: bool = True,
    selected_gfs_v17: bool = False,
    surface_exchange_option: int = 3,
) -> dict[str, Any]:
    return {
        "prslki_input": getattr(state, "prslki", None),
        "prslk1_input": getattr(state, "prslk1", None),
        "prsik1_input": getattr(state, "prsik1", None),
        "temperature_forcing": state.temperature_forcing,
        "vegtype": state.vegtype,
        "tskin": state.tskin,
        "cm_state": state.cm,
        "ch_state": state.ch,
        "qair": state.qair,
        "prsl1": state.prsl1,
        "ps": state.ps,
        "snow_levels": state.snow_levels,
        "dzsnso": state.dzsnso,
        "temperature_snow_soil": state.temperature_snow_soil,
        "fveg": state.fveg,
        "garea": state.garea,
        "pblh": state.pblh,
        "rhoair": state.rhoair,
        "dlwflx": state.dlwflx,
        "eair": state.eair,
        "pahb": state.pahb,
        "u1": state.u1,
        "v1": state.v1,
        "ur": geometry.ur,
        "z0mg": geometry.z0mg,
        "zpdg": geometry.zpdg,
        "zlvl": geometry.zlvl,
        "emg": geometry.emg,
        "gamma_ground": geometry.gamma_ground,
        "rsurf": geometry.rsurf,
        "rhsur": geometry.rhsur,
        "snowh": geometry.snowh,
        "fsno": geometry.fsno,
        "latheag": geometry.latheag,
        "sag": sag,
        "df": df,
        "tgb": getattr(state, "temperature_ground", None) if tgb is None else tgb,
        "cm": cm,
        "ch": ch,
        "ustarx": getattr(state, "friction_velocity", None) if ustarx is None else ustarx,
        "thsfc_loc": thsfc_loc,
        "niter": int(niter),
        "use_nested_stability_compile": bool(use_nested_stability_compile),
        "selected_gfs_v17": bool(selected_gfs_v17),
        "use_sfcdif4": bool(surface_exchange_option == 4),
    }


def _bare_flux_result(values: tuple[torch.Tensor, ...]) -> BareFluxResult:
    return BareFluxResult(
        tgb=values[0],
        cm=values[1],
        ch=values[2],
        ustarx=values[3],
        z0h=values[4],
        tauxb=values[5],
        tauyb=values[6],
        irb=values[7],
        shb=values[8],
        evb=values[9],
        ghb=values[10],
        t2mb=values[11],
        q2b=values[12],
        qsfc=values[13],
        ehb2=values[14],
        csigmaf0=values[15],
    )


def _vege_flux_flat_kwargs(
    state: Any,
    tables: LandEnergyTables,
    geometry: LandEnergyGeometry,
    thermo: ThermopropResult,
    radiation: RadiationResult,
    *,
    dt: torch.Tensor | float,
    canliq: torch.Tensor,
    canice: torch.Tensor,
    fwet: torch.Tensor,
    btran: torch.Tensor,
    thsfc_loc: bool,
    niter: int,
    use_nested_stability_compile: bool = True,
    use_inner_compile_islands: bool = True,
    surface_exchange_option: int = 3,
) -> dict[str, Any]:
    veg_idx = _state_cached_tensor(state, "veg_idx", lambda: _veg_index(state.vegtype))
    kwargs = {
        "friction_velocity": getattr(state, "friction_velocity", None),
        "prslki_input": getattr(state, "prslki", None),
        "prslk1_input": getattr(state, "prslk1", None),
        "prsik1_input": getattr(state, "prsik1", None),
        "temperature_leaf": getattr(state, "temperature_leaf", None),
        "temperature_ground": getattr(state, "temperature_ground", None),
        "temperature_canopy_air": getattr(state, "temperature_canopy_air", None),
        "vapor_pres_canopy_air": getattr(state, "vapor_pres_canopy_air", None),
        "temperature_forcing": state.temperature_forcing,
        "tskin": state.tskin,
        "cm_state": state.cm,
        "ch_state": state.ch,
        "vegtype": state.vegtype,
        "ps": state.ps,
        "prsl1": state.prsl1,
        "rhoair": state.rhoair,
        "qair": state.qair,
        "fveg": state.fveg,
        "shdfac": state.shdfac,
        "garea": state.garea,
        "pblh": state.pblh,
        "pahv": state.pahv,
        "pahg": state.pahg,
        "eair": state.eair,
        "dlwflx": state.dlwflx,
        "u1": state.u1,
        "v1": state.v1,
        "snow_levels": state.snow_levels,
        "dzsnso": state.dzsnso,
        "temperature_snow_soil": state.temperature_snow_soil,
        "ur": geometry.ur,
        "vai": geometry.vai,
        "z0m": geometry.z0m,
        "z0mg": geometry.z0mg,
        "zpd": geometry.zpd,
        "zlvl": geometry.zlvl,
        "emv": geometry.emv,
        "emg": geometry.emg,
        "gamma_ground": geometry.gamma_ground,
        "rsurf": geometry.rsurf,
        "rhsur": geometry.rhsur,
        "snowh": geometry.snowh,
        "fsno": geometry.fsno,
        "df": thermo.df,
        "laisun": radiation.laisun,
        "laisha": radiation.laisha,
        "parsun": radiation.parsun,
        "parsha": radiation.parsha,
        "sav": radiation.sav,
        "sag": radiation.sag,
        "hcan": _state_cached_tensor(state, "hvt_by_column", lambda: tables.hvt[veg_idx]),
        "cwp": _state_cached_tensor(state, "cwpvt_by_column", lambda: tables.cwpvt[veg_idx]),
        "dleaf": _state_cached_tensor(state, "dleaf_by_column", lambda: tables.dleaf[veg_idx]),
        "cbiom": _state_cached_tensor(state, "cbiom_by_column", lambda: tables.cbiom[veg_idx]),
        "rgl": _state_cached_tensor(state, "rgl_by_column", lambda: tables.rgl[veg_idx]),
        "rsmin": _state_cached_tensor(state, "rsmin_by_column", lambda: tables.rsmin[veg_idx]),
        "rsmax": _state_cached_tensor(state, "rsmax_by_column", lambda: tables.rsmax[veg_idx]),
        "topt": _state_cached_tensor(state, "topt_by_column", lambda: tables.topt[veg_idx]),
        "hs": _state_cached_tensor(state, "hs_by_column", lambda: tables.hs[veg_idx]),
        "dt_tensor": _as_vector(dt, state.temperature_forcing),
        "canliq": canliq,
        "canice": canice,
        "fwet": fwet,
        "btran": btran,
        "thsfc_loc": thsfc_loc,
        "niter": int(niter),
        "use_nested_stability_compile": bool(use_nested_stability_compile),
        "use_inner_compile_islands": bool(use_inner_compile_islands),
        "selected_gfs_v17": tables.table_variant == GFS_V17_PROFILE_SCM_TABLE_VARIANT,
        "use_sfcdif4": surface_exchange_option == 4,
    }
    # Fortran does not call vege_flux on these rows. Guard its inactive operands.
    canopy = (state.fveg > 0.0) & (geometry.vai > 0.0)
    safe_controls = {"fveg": 1.0, "vai": 2.0, "z0m": 0.1, "z0mg": 0.01,
                     "zpd": 0.0, "hcan": 1.0, "cwp": 1.0, "dleaf": 0.05,
                     "cbiom": 300.0, "rgl": 50.0, "rsmin": 100.0,
                     "rsmax": 5000.0, "topt": 298.0, "hs": 36.0,
                     "laisun": 1.0, "laisha": 1.0}
    for name, fallback in safe_controls.items():
        kwargs[name] = torch.where(canopy, kwargs[name], fallback)
    return kwargs


def _vege_flux_result(values: tuple[torch.Tensor, ...]) -> VegFluxResult:
    return VegFluxResult(
        tgv=values[0],
        tv=values[1],
        tah=values[2],
        eah=values[3],
        cm=values[4],
        ch=values[5],
        chleaf=values[6],
        chuc=values[7],
        rb=values[8],
        rssun=values[9],
        rssha=values[10],
        psnsun=values[11],
        psnsha=values[12],
        irg=values[13],
        irc=values[14],
        shg=values[15],
        shc=values[16],
        evg=values[17],
        evc=values[18],
        tr=values[19],
        gh=values[20],
        tauxv=values[21],
        tauyv=values[22],
        t2mv=values[23],
        q2v=values[24],
        qsfc=values[25],
        canhs=values[26],
        cah2=values[27],
        z0h=values[28],
        z0hg=values[29],
        csigmaf1=values[30],
    )


def _bare_flux_opt_sfc3_flat_impl(
    *,
    temperature_forcing: torch.Tensor,
    vegtype: torch.Tensor,
    tskin: torch.Tensor,
    cm_state: torch.Tensor,
    ch_state: torch.Tensor,
    qair: torch.Tensor,
    prsl1: torch.Tensor,
    ps: torch.Tensor,
    snow_levels: torch.Tensor,
    dzsnso: torch.Tensor,
    temperature_snow_soil: torch.Tensor,
    fveg: torch.Tensor,
    garea: torch.Tensor,
    pblh: torch.Tensor,
    rhoair: torch.Tensor,
    dlwflx: torch.Tensor,
    eair: torch.Tensor,
    pahb: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    ur: torch.Tensor,
    z0mg: torch.Tensor,
    zpdg: torch.Tensor,
    zlvl: torch.Tensor,
    emg: torch.Tensor,
    gamma_ground: torch.Tensor,
    rsurf: torch.Tensor,
    rhsur: torch.Tensor,
    snowh: torch.Tensor,
    fsno: torch.Tensor,
    latheag: torch.Tensor,
    sag: torch.Tensor,
    df: torch.Tensor,
    tgb: torch.Tensor | None,
    cm: torch.Tensor | None,
    ch: torch.Tensor | None,
    ustarx: torch.Tensor | None,
    thsfc_loc: bool,
    niter: int,
    use_nested_stability_compile: bool = True,
    selected_gfs_v17: bool = False,
    use_sfcdif4: bool = False,
    prslki_input: torch.Tensor | None = None,
    prslk1_input: torch.Tensor | None = None,
    prsik1_input: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    sfctmp = temperature_forcing
    tgb_work = tskin if tgb is None else tgb
    cm_work = cm_state if cm is None else cm
    ch_work = ch_state if ch is None else ch
    ustar_work = torch.sqrt(torch.clamp(cm_work, min=0.0)) * ur if ustarx is None else ustarx
    prslk1x = torch.pow(prsl1 / 100000.0, NOAHMP_EXNER_EXP)
    prslkix = 1.0 / prslk1x
    psfc_exner = torch.pow(ps / 100000.0, NOAHMP_EXNER_EXP)
    prsik1x = 1.0 / psfc_exner
    prslk1x = prslk1x if prslk1_input is None else prslk1_input
    prslkix = prslkix if prslki_input is None else prslki_input
    prsik1x = prsik1x if prsik1_input is None else prsik1_input
    top_df = _gather_top_layer(df, snow_levels)
    top_dz = _gather_top_layer(dzsnso, snow_levels)
    top_stc = _gather_top_layer(temperature_snow_soil, snow_levels)
    cgh = 2.0 * top_df / top_dz
    cir = emg * NOAHMP_SB
    if selected_gfs_v17:
        _z0m_out_static = z0mg
        z0h_static = torch.where(vegtype <= 5, z0mg, z0mg * 0.01)
        csigmaf0_static = torch.zeros_like(z0mg)
    elif use_nested_stability_compile:
        _z0m_out_static, z0h_static, csigmaf0_static = thermalz0_bare_chen09(
            z0mg=z0mg,
            zlvl=zlvl,
            zpd=zpdg,
            ustarx=ustar_work,
        )
    else:
        _z0m_out_static, z0h_static, csigmaf0_static = _thermalz0_bare_chen09_impl(
            z0mg=z0mg,
            zlvl=zlvl,
            zpd=zpdg,
            ustarx=ustar_work,
        )
    _ = _z0m_out_static

    zero = torch.zeros_like(sfctmp)
    fm = zero
    fh = zero
    fh2 = zero
    evb = zero
    shb = zero
    h_mo = zero  # bare_flux h: m-o length input of the next sfcdif4 call
    irb = zero
    ghb = zero
    qfx = zero
    z0h = torch.clamp(z0mg, min=1.0e-6)
    csigmaf0 = zero
    qsfc = zero
    fv = ustar_work
    csh = zero
    cev = zero
    estg = zero
    exchange_wind = ur
    ch2_work = zero
    cq2_work = zero

    for _ in range(niter):
        z0h = z0h_static
        csigmaf0 = csigmaf0_static
        sfcdif3_args = {
            "sfctmp": sfctmp,
            "qair": qair,
            "ur": ur,
            "zlvl": zlvl,
            "tgb": tgb_work,
            "thsfc_loc": thsfc_loc,
            "prslkix": prslkix,
            "prsik1x": prsik1x,
            "prslk1x": prslk1x,
            "z0m": z0mg,
            "z0h": z0h,
            "zpd": zpdg,
            "fveg": fveg,
            "garea1": garea,
        }
        if use_sfcdif4:
            (
                qsfc,
                fv,
                cm_speed,
                ch_speed,
                ch2_speed,
                cq2_speed,
                _rmol,
                _rb,
                fm,
                fh,
                _stress,
                _fm10,
                fh2,
                exchange_wind,
                _flhc,
                _flqc,
            ) = _sfcdif4_impl(
                temperature_forcing=sfctmp,
                prsl1=prsl1,
                ps=ps,
                pblh=pblh,
                garea=garea,
                u1=u1,
                v1=v1,
                qair=qair,
                tsk=tgb_work,
                zlvl=zlvl,
                z0m=z0mg,
                snowh=snowh,
                hfx=h_mo,
                qfx=qfx,
                qsfc=qsfc,
                ustar=fv,
            )
            cm_work = cm_speed / exchange_wind
            ch_work = ch_speed / exchange_wind
            ch2_work = ch2_speed / exchange_wind
            cq2_work = cq2_speed / exchange_wind
            snow_present = snowh * 1000.0 > 0.0
            cm_work = torch.where(snow_present, torch.clamp(cm_work, max=0.01), cm_work)
            ch_work = torch.where(snow_present, torch.clamp(ch_work, max=0.01), ch_work)
            ch2_work = torch.where(snow_present, torch.clamp(ch2_work, max=0.01), ch2_work)
            cq2_work = torch.where(snow_present, torch.clamp(cq2_work, max=0.01), cq2_work)
        elif use_nested_stability_compile:
            fm, fh, fh2, fv, cm_work, ch_work = sfcdif3(**sfcdif3_args)
        else:
            fm, fh, fh2, fv, cm_work, ch_work = _sfcdif3_impl(**sfcdif3_args)
        ramb = torch.clamp(1.0 / (cm_work * exchange_wind), min=1.0)
        rahb = torch.clamp(1.0 / (ch_work * exchange_wind), min=1.0)
        rawb = rahb
        _ = ramb

        t_c = torch.clamp(tgb_work - NOAHMP_TFRZ, min=-50.0, max=50.0)
        esatw, esati, dsatw, dsati = esat_noahmp(t_c)
        warm = t_c > 0.0
        estg = torch.where(warm, esatw, esati)
        destg = torch.where(warm, dsatw, dsati)

        csh = rhoair * NOAHMP_CPAIR / rahb
        cev = rhoair * NOAHMP_CPAIR / gamma_ground / (rsurf + rawb)
        tgb_pow4 = _pow4(tgb_work)
        tgb_pow3 = _pow3(tgb_work)
        irb = cir * tgb_pow4 - emg * dlwflx
        shb = csh * (tgb_work - sfctmp)
        evb = cev * (estg * rhsur - eair)
        ghb = cgh * (tgb_work - top_stc)
        balance = sag - irb - shb - evb - ghb + pahb
        jac = 4.0 * cir * tgb_pow3 + csh + cev * destg + cgh
        dtg = balance / jac
        irb = irb + 4.0 * cir * tgb_pow3 * dtg
        shb = shb + csh * dtg
        evb = evb + cev * destg * dtg
        ghb = ghb + cgh * dtg
        tgb_work = tgb_work + dtg
        h_mo = csh * (tgb_work - sfctmp)  # bare_flux: h = csh*(tgb - sfctmp) after the update
        t_c = torch.clamp(tgb_work - NOAHMP_TFRZ, min=-50.0, max=50.0)
        esatw, esati, _dsatw, _dsati = esat_noahmp(t_c)
        estg = torch.where(t_c > 0.0, esatw, esati)
        qsfc = CON_EPS * (estg * rhsur) / (ps + CON_EPSM1 * (estg * rhsur))
        qfx = (qsfc - qair) * cev * gamma_ground / NOAHMP_CPAIR
        _ = (fm, fh, qfx)

    snow_melt_reset = (snowh > 0.05) & (tgb_work > NOAHMP_TFRZ)
    tgb_reset = (1.0 - fsno) * tgb_work + fsno * NOAHMP_TFRZ
    tgb_work = torch.where(snow_melt_reset, tgb_reset, tgb_work)
    irb_reset = cir * _pow4(tgb_work) - emg * dlwflx
    shb_reset = csh * (tgb_work - sfctmp)
    evb_reset = cev * (estg * rhsur - eair)
    ghb_reset = sag + pahb - (irb_reset + shb_reset + evb_reset)
    irb = torch.where(snow_melt_reset, irb_reset, irb)
    shb = torch.where(snow_melt_reset, shb_reset, shb)
    evb = torch.where(snow_melt_reset, evb_reset, evb)
    ghb = torch.where(snow_melt_reset, ghb_reset, ghb)

    tauxb = -rhoair * cm_work * ur * u1
    tauyb = -rhoair * cm_work * ur * v1
    if use_sfcdif4:
        ehb2 = 1.0 / torch.clamp(1.0 / (ch2_work * exchange_wind), min=1.0)
        cq2b = 1.0 / torch.clamp(1.0 / (cq2_work * exchange_wind), min=1.0)
    else:
        ehb2 = fv * NOAHMP_VKC / fh2
        cq2b = ehb2
    # Benign divisors in the unselected branch keep its backward finite (forward unchanged where selected).
    t2mb = torch.where(ehb2 < 1.0e-5, tgb_work, tgb_work - shb / (rhoair * NOAHMP_CPAIR) / torch.where(ehb2 < 1.0e-5, 1.0, ehb2))
    q2b = torch.where(
        ehb2 < 1.0e-5,
        qsfc,
        qsfc - evb / (latheag * rhoair) * (1.0 / torch.where(ehb2 < 1.0e-5, 1.0, cq2b) + rsurf),
    )
    ch_work = 1.0 / torch.clamp(1.0 / (ch_work * exchange_wind), min=1.0)
    return (
        tgb_work,
        cm_work,
        ch_work,
        fv,
        z0h,
        tauxb,
        tauyb,
        irb,
        shb,
        evb,
        ghb,
        t2mb,
        q2b,
        qsfc,
        ehb2,
        csigmaf0,
    )


_compute_energy_geometry_impl = _compute_energy_geometry_flat_impl
_thermoprop_source_corridor_impl = thermoprop_source_corridor
_radiation_bats_two_stream_impl = _radiation_bats_two_stream_flat_impl
_bare_flux_opt_sfc3_impl = bare_flux_opt_sfc3
_soilwater_opt_run1_inf1_source_corridor_impl = soilwater_opt_run1_inf1_source_corridor
_vege_flux_opt_sfc3_impl = _vege_flux_opt_sfc3_flat_impl


def compute_energy_geometry(
    state: Any,
    tables: LandEnergyTables,
    *,
    options: Mapping[str, Any] | None = None,
    soilcolor: torch.Tensor | None = None,
) -> LandEnergyGeometry:
    _option_check(options)
    return _geometry_result(
        run_compile_island(
            "land_geometry",
            "compute_energy_geometry",
            _compute_energy_geometry_impl,
            **_energy_geometry_flat_kwargs(state, tables, soilcolor=soilcolor),
        )
    )


def thermoprop_source_corridor(
    state: Any,
    tables: LandEnergyTables,
    geometry: LandEnergyGeometry,
    *,
    dt: torch.Tensor | float,
    shdfac: torch.Tensor,
) -> ThermopropResult:
    return _thermoprop_result(
        run_compile_island(
            "land_thermoprop",
            "thermoprop_source_corridor",
            _thermoprop_source_corridor_flat_impl,
            **_thermoprop_flat_kwargs(state, tables, geometry, dt=dt, shdfac=shdfac),
        )
    )


def radiation_bats_two_stream(
    state: Any,
    tables: LandEnergyTables,
    geometry: LandEnergyGeometry,
    *,
    dt: torch.Tensor | float,
    soilcolor: torch.Tensor | None = None,
    tauss: torch.Tensor | None = None,
    sneqvo: torch.Tensor | None = None,
) -> RadiationResult:
    return _radiation_result(
        run_compile_island(
            "land_radiation",
            "radiation_bats_two_stream",
            _radiation_bats_two_stream_impl,
            **_radiation_flat_kwargs(
                state,
                tables,
                geometry=geometry,
                dt=dt,
                soilcolor=soilcolor,
                tauss=tauss,
                sneqvo=sneqvo,
            ),
        )
    )


def bare_flux_opt_sfc3(
    state: Any,
    geometry: LandEnergyGeometry,
    *,
    sag: torch.Tensor,
    df: torch.Tensor,
    dt: torch.Tensor | float,
    tgb: torch.Tensor | None = None,
    cm: torch.Tensor | None = None,
    ch: torch.Tensor | None = None,
    ustarx: torch.Tensor | None = None,
    thsfc_loc: bool = True,
    niter: int = 5,
    selected_gfs_v17: bool = False,
    surface_exchange_option: int = 3,
) -> BareFluxResult:
    _ = dt
    use_nested_stability_compile = "land_bare_flux" not in compile_categories_enabled()
    return _bare_flux_result(
        run_compile_island(
            "land_bare_flux",
            "bare_flux_opt_sfc3",
            _bare_flux_opt_sfc3_flat_impl,
            **_bare_flux_flat_kwargs(
                state,
                geometry,
                sag=sag,
                df=df,
                tgb=tgb,
                cm=cm,
                ch=ch,
                ustarx=ustarx,
                thsfc_loc=thsfc_loc,
                niter=niter,
                use_nested_stability_compile=use_nested_stability_compile,
                selected_gfs_v17=selected_gfs_v17,
                surface_exchange_option=surface_exchange_option,
            ),
        )
    )


def soilwater_opt_run1_inf1_source_corridor(
    state: Any,
    tables: LandEnergyTables,
    *,
    qinsur: torch.Tensor,
    qseva: torch.Tensor,
    etrani: torch.Tensor,
    dt: torch.Tensor | float,
    zwt_initial: torch.Tensor | float = 2.0,
    wa_initial: torch.Tensor | float = 4900.0,
    wt_initial: torch.Tensor | float = 500.0,
    smc_initial: torch.Tensor | None = None,
    sh2o_initial: torch.Tensor | None = None,
    dzsnso_soil_initial: torch.Tensor | None = None,
    iteration_count: int = 6,
    sice_initial: torch.Tensor | None = None,
) -> SoilWaterOptRun1Result:
    return _soilwater_opt_run1_inf1_source_corridor_impl(
        state,
        tables,
        qinsur=qinsur,
        qseva=qseva,
        etrani=etrani,
        dt=dt,
        zwt_initial=zwt_initial,
        wa_initial=wa_initial,
        wt_initial=wt_initial,
        smc_initial=smc_initial,
        sh2o_initial=sh2o_initial,
        dzsnso_soil_initial=dzsnso_soil_initial,
        iteration_count=iteration_count,
        sice_initial=sice_initial,
    )


def vege_flux_opt_sfc3(
    state: Any,
    tables: LandEnergyTables,
    geometry: LandEnergyGeometry,
    thermo: ThermopropResult,
    radiation: RadiationResult,
    *,
    dt: torch.Tensor | float,
    canliq: torch.Tensor,
    canice: torch.Tensor,
    fwet: torch.Tensor,
    btran: torch.Tensor,
    thsfc_loc: bool = True,
    niter: int = 20,
    surface_exchange_option: int = 3,
) -> VegFluxResult:
    use_full_compile = "land_vege_flux" in compile_categories_enabled()
    use_nested_stability_compile = not use_full_compile
    return _vege_flux_result(
        run_compile_island(
            "land_vege_flux",
            "vege_flux_opt_sfc3",
            _vege_flux_opt_sfc3_impl,
            **_vege_flux_flat_kwargs(
                state,
                tables,
                geometry,
                thermo,
                radiation,
                dt=dt,
                canliq=canliq,
                canice=canice,
                fwet=fwet,
                btran=btran,
                thsfc_loc=thsfc_loc,
                niter=niter,
                use_nested_stability_compile=use_nested_stability_compile,
                use_inner_compile_islands=not use_full_compile,
                surface_exchange_option=surface_exchange_option,
            ),
        )
    )


def vege_flux_opt_sfc3_begin(
    state: Any,
    tables: LandEnergyTables,
    geometry: LandEnergyGeometry,
    thermo: ThermopropResult,
    radiation: RadiationResult,
    *,
    dt: torch.Tensor | float,
    canliq: torch.Tensor,
    canice: torch.Tensor,
    fwet: torch.Tensor,
    btran: torch.Tensor,
    thsfc_loc: bool = True,
    niter: int = 20,
    surface_exchange_option: int = 3,
) -> dict[str, Any]:
    """``vege_flux_opt_sfc3`` up to its canopy iteration loop: the loop state.

    ``vege_flux_opt_sfc3_iterations`` runs the loop and ``vege_flux_opt_sfc3_finish`` gives the result;
    the three equal ``vege_flux_opt_sfc3``. With the ``land_vege_flux`` island on, the state holds the
    island's whole result.
    """
    arguments = dict(dt=dt, canliq=canliq, canice=canice, fwet=fwet, btran=btran, thsfc_loc=thsfc_loc, niter=niter,
                     surface_exchange_option=surface_exchange_option)
    if "land_vege_flux" in compile_categories_enabled():
        return {"result": vege_flux_opt_sfc3(state, tables, geometry, thermo, radiation, **arguments)}
    return _vege_flux_setup(**_vege_flux_flat_kwargs(
        state, tables, geometry, thermo, radiation, **arguments,
        use_nested_stability_compile=True, use_inner_compile_islands=True))


def vege_flux_opt_sfc3_iterations(env: dict[str, Any], step: Any = None) -> dict[str, Any]:
    """The canopy iteration loop of ``vege_flux_opt_sfc3_begin``'s state (``step``: see ``_vege_flux_iterations``)."""
    return env if "result" in env else _vege_flux_iterations(env, step)


def vege_flux_opt_sfc3_finish(env: dict[str, Any]) -> VegFluxResult:
    """``vege_flux_opt_sfc3`` from the state after its canopy iteration loop."""
    return env["result"] if "result" in env else _vege_flux_result(_vege_flux_finish(env))


def compute_radiation_and_bare_flux(
    state: Any,
    *,
    tables: LandEnergyTables | None = None,
    options: Mapping[str, Any] | None = None,
    soilcolor: torch.Tensor | None = None,
    df: torch.Tensor,
    dt: torch.Tensor | float,
    tauss: torch.Tensor | None = None,
    sneqvo: torch.Tensor | None = None,
) -> LandEnergyPathResult:
    """Convenience wrapper for the requested H energy helper surface."""

    _option_check(options)
    local_tables = load_land_energy_tables(device=state.temperature_forcing.device, dtype=state.temperature_forcing.dtype) if tables is None else tables
    geometry = compute_energy_geometry(state, local_tables, options=options, soilcolor=soilcolor)
    radiation = radiation_bats_two_stream(
        state,
        local_tables,
        geometry,
        dt=dt,
        soilcolor=soilcolor,
        tauss=tauss,
        sneqvo=sneqvo,
    )
    bare_flux = bare_flux_opt_sfc3(state, geometry, sag=radiation.sag, df=df, dt=dt)
    return LandEnergyPathResult(geometry=geometry, radiation=radiation, bare_flux=bare_flux)


def result_as_dict(result: LandEnergyPathResult) -> dict[str, torch.Tensor]:
    """Flatten core helper outputs into names matching later H assembly needs."""

    radiation = result.radiation
    bare = result.bare_flux
    geometry = result.geometry
    return {
        "fsno": geometry.fsno,
        "emg": geometry.emg,
        "z0mg": geometry.z0mg,
        "zpdg": geometry.zpdg,
        "zlvl": geometry.zlvl,
        "fsa": radiation.fsa,
        "fsr": radiation.fsr,
        "sag": radiation.sag,
        "sav": radiation.sav,
        "fsrv": radiation.fsrv,
        "fsrg": radiation.fsrg,
        "albdvis": radiation.albd[:, 0],
        "albdnir": radiation.albd[:, 1],
        "albivis": radiation.albi[:, 0],
        "albinir": radiation.albi[:, 1],
        "albsnd_vis": radiation.albsnd[:, 0],
        "albsnd_nir": radiation.albsnd[:, 1],
        "albsni_vis": radiation.albsni[:, 0],
        "albsni_nir": radiation.albsni[:, 1],
        "fsun": radiation.fsun,
        "laisun": radiation.laisun,
        "laisha": radiation.laisha,
        "parsun": radiation.parsun,
        "parsha": radiation.parsha,
        "apar": radiation.parsun * radiation.laisun + radiation.parsha * radiation.laisha,
        "tgb": bare.tgb,
        "cm_bare": bare.cm,
        "ch_bare": bare.ch,
        "z0h_bare": bare.z0h,
        "tauxb": bare.tauxb,
        "tauyb": bare.tauyb,
        "irb": bare.irb,
        "shb": bare.shb,
        "evb": bare.evb,
        "ghb": bare.ghb,
        "t2mb": bare.t2mb,
        "q2b": bare.q2b,
        "qsfc": bare.qsfc,
    }
