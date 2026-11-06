# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""R4 Noah-MP/NSST surface allocation and native input bindings."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import torch

R4_SURFACE_TAIL_SHAPES = {
    "vegtype_frac": (20,),
    "soiltype_frac": (19,),
    "tiice": (2,),
    "hprime": (24,),
    "dust12m_in": (12, 5),
    "smoke_RRFS": (24, 2),
    "smoke2d_RRFS": (5,),
    "emi_in": (1,),
    **{name: (4,) for name in ("slc", "smc", "stc", "smoiseq")},
    **{name: (3,) for name in ("snicexy", "snliqxy", "tsnoxy")},
    "zsnsoxy": (7,),
}
R4_SURFACE_INTEGER_MEMBERS = frozenset(
    {
        "use_lake_model",
        "slope",
        "slope_save",
        "vtype",
        "vtype_save",
        "stype",
        "stype_save",
        "scolor",
        "scolor_save",
    }
)


@dataclass(frozen=True)
class NativeSurface:
    """MPAS sfc_input fields; all tensor axes are column-first."""

    isltyp: torch.Tensor
    ivgtyp: torch.Tensor
    landmask: torch.Tensor
    sfc_albbck: torch.Tensor
    skintemp: torch.Tensor
    snow: torch.Tensor
    snowc: torch.Tensor
    snowh: torch.Tensor
    sst: torch.Tensor
    ter: torch.Tensor
    tmn: torch.Tensor
    vegfra: torch.Tensor
    seaice: torch.Tensor
    xice: torch.Tensor
    xland: torch.Tensor
    shdmin: torch.Tensor
    shdmax: torch.Tensor
    snoalb: torch.Tensor
    ufs_sfc_znt: torch.Tensor
    ufs_sfc_z0: torch.Tensor
    ufs_sfc_alvsf: torch.Tensor
    ufs_sfc_alnsf: torch.Tensor
    ufs_sfc_alvwf: torch.Tensor
    ufs_sfc_alnwf: torch.Tensor
    ufs_sfc_canwat: torch.Tensor
    ufs_sfc_f10m: torch.Tensor
    ufs_sfc_ffhh: torch.Tensor
    ufs_sfc_ffmm: torch.Tensor
    ufs_sfc_ust: torch.Tensor
    ufs_sfc_hice: torch.Tensor
    ufs_sfc_q2: torch.Tensor
    ufs_sfc_t2m: torch.Tensor
    ufs_sfc_tg3: torch.Tensor
    ufs_sfc_tisfc: torch.Tensor
    greenfrac: torch.Tensor
    albedo12m: torch.Tensor
    sh2o: torch.Tensor
    smois: torch.Tensor
    tslb: torch.Tensor
    var2d: torch.Tensor
    con: torch.Tensor
    oa1: torch.Tensor
    oa2: torch.Tensor
    oa3: torch.Tensor
    oa4: torch.Tensor
    ol1: torch.Tensor
    ol2: torch.Tensor
    ol3: torch.Tensor
    ol4: torch.Tensor
    gwd_theta: torch.Tensor
    gwd_gamma: torch.Tensor
    gwd_sigma: torch.Tensor
    gwd_elvmax: torch.Tensor
    var2dss: torch.Tensor
    conss: torch.Tensor
    oa1ss: torch.Tensor
    oa2ss: torch.Tensor
    oa3ss: torch.Tensor
    oa4ss: torch.Tensor
    ol1ss: torch.Tensor
    ol2ss: torch.Tensor
    ol3ss: torch.Tensor
    ol4ss: torch.Tensor

    @classmethod
    def from_mpas_state(cls, state: Mapping[str, torch.Tensor]) -> NativeSurface:
        values = {field.name: state[field.name] for field in fields(cls)}
        for name in ("isltyp", "ivgtyp", "landmask"):
            if values[name].dtype not in (torch.int32, torch.int64):
                raise ValueError(f"native surface category is not an integer: {name}")
            limits = torch.iinfo(torch.int32)
            torch._assert_async(
                ((values[name] >= limits.min) & (values[name] <= limits.max)).all(),
                f"native surface category exceeds source int32: {name}",
            )
            values[name] = values[name].to(torch.int32)
        return cls(**values)


@dataclass(frozen=True)
class SurfaceControl:
    """Static host controls for the selected R4 surface mapping."""

    n_cells_solve: int
    jdat_month: int
    imn: int
    ntqv: int
    initialize_noahmp_liquid: bool
    dzs: torch.Tensor


@dataclass(frozen=True)
class SoilParameters:
    """Initialized noahmp_tables hydraulic arrays, including unused slots."""

    bexp: torch.Tensor
    smcmax: torch.Tensor
    psisat: torch.Tensor


@dataclass(frozen=True)
class Sfcprop:
    """Allocated pointer members for R4: Noah-MP, NSST, no lake/fire/MYNN."""

    slmsk: torch.Tensor
    oceanfrac: torch.Tensor
    landfrac: torch.Tensor
    vegtype_frac: torch.Tensor
    soiltype_frac: torch.Tensor
    lakefrac: torch.Tensor
    lakedepth: torch.Tensor
    use_lake_model: torch.Tensor
    tsfc: torch.Tensor
    usfco: torch.Tensor
    vsfco: torch.Tensor
    tsfco: torch.Tensor
    tsfcl: torch.Tensor
    tisfc: torch.Tensor
    tiice: torch.Tensor
    snowd: torch.Tensor
    zorl: torch.Tensor
    zorlw: torch.Tensor
    zorll: torch.Tensor
    zorli: torch.Tensor
    zorlwav: torch.Tensor
    fice: torch.Tensor
    snodl: torch.Tensor
    weasdl: torch.Tensor
    snodi: torch.Tensor
    weasdi: torch.Tensor
    hprime: torch.Tensor
    dust12m_in: torch.Tensor
    smoke_RRFS: torch.Tensor
    smoke2d_RRFS: torch.Tensor
    emi_in: torch.Tensor
    albdirvis_lnd: torch.Tensor
    albdirnir_lnd: torch.Tensor
    albdifvis_lnd: torch.Tensor
    albdifnir_lnd: torch.Tensor
    emis_lnd: torch.Tensor
    emis_ice: torch.Tensor
    emis_wat: torch.Tensor
    snoalb: torch.Tensor
    alvsf: torch.Tensor
    alnsf: torch.Tensor
    alvwf: torch.Tensor
    alnwf: torch.Tensor
    facsf: torch.Tensor
    facwf: torch.Tensor
    slope: torch.Tensor
    slope_save: torch.Tensor
    shdmin: torch.Tensor
    shdmax: torch.Tensor
    tg3: torch.Tensor
    vfrac: torch.Tensor
    vtype: torch.Tensor
    vtype_save: torch.Tensor
    stype: torch.Tensor
    stype_save: torch.Tensor
    scolor: torch.Tensor
    scolor_save: torch.Tensor
    uustar: torch.Tensor
    oro: torch.Tensor
    oro_uf: torch.Tensor
    evap: torch.Tensor
    hflx: torch.Tensor
    qss: torch.Tensor
    hice: torch.Tensor
    weasd: torch.Tensor
    sncovr: torch.Tensor
    sncovr_ice: torch.Tensor
    canopy: torch.Tensor
    ffmm: torch.Tensor
    ffhh: torch.Tensor
    f10m: torch.Tensor
    tprcp: torch.Tensor
    srflag: torch.Tensor
    slc: torch.Tensor
    smc: torch.Tensor
    stc: torch.Tensor
    t2m: torch.Tensor
    th2m: torch.Tensor
    q2m: torch.Tensor
    tref: torch.Tensor
    z_c: torch.Tensor
    c_0: torch.Tensor
    c_d: torch.Tensor
    w_0: torch.Tensor
    w_d: torch.Tensor
    xt: torch.Tensor
    xs: torch.Tensor
    xu: torch.Tensor
    xv: torch.Tensor
    xz: torch.Tensor
    zm: torch.Tensor
    xtts: torch.Tensor
    xzts: torch.Tensor
    d_conv: torch.Tensor
    ifd: torch.Tensor
    dt_cool: torch.Tensor
    qrain: torch.Tensor
    xlaixy: torch.Tensor
    rca: torch.Tensor
    raincprv: torch.Tensor
    rainncprv: torch.Tensor
    iceprv: torch.Tensor
    snowprv: torch.Tensor
    graupelprv: torch.Tensor
    snowxy: torch.Tensor
    tvxy: torch.Tensor
    tgxy: torch.Tensor
    canicexy: torch.Tensor
    canliqxy: torch.Tensor
    eahxy: torch.Tensor
    tahxy: torch.Tensor
    cmxy: torch.Tensor
    chxy: torch.Tensor
    fwetxy: torch.Tensor
    sneqvoxy: torch.Tensor
    alboldxy: torch.Tensor
    qsnowxy: torch.Tensor
    wslakexy: torch.Tensor
    zwtxy: torch.Tensor
    waxy: torch.Tensor
    wtxy: torch.Tensor
    lfmassxy: torch.Tensor
    rtmassxy: torch.Tensor
    stmassxy: torch.Tensor
    woodxy: torch.Tensor
    stblcpxy: torch.Tensor
    fastcpxy: torch.Tensor
    xsaixy: torch.Tensor
    taussxy: torch.Tensor
    smcwtdxy: torch.Tensor
    deeprechxy: torch.Tensor
    rechxy: torch.Tensor
    snicexy: torch.Tensor
    snliqxy: torch.Tensor
    tsnoxy: torch.Tensor
    smoiseq: torch.Tensor
    zsnsoxy: torch.Tensor
    draincprv: torch.Tensor
    drainncprv: torch.Tensor
    diceprv: torch.Tensor
    dsnowprv: torch.Tensor
    dgraupelprv: torch.Tensor
    semisbase: torch.Tensor
    rmol: torch.Tensor
    flhc: torch.Tensor
    flqc: torch.Tensor

    @classmethod
    def allocate_r4(cls, *, ncol: int, device: torch.device | str, dtype: torch.dtype = torch.float64) -> Sfcprop:
        """Selected sfcprop_create: four soils, three snow layers, 24 GWD slots."""
        if type(ncol) is not int or ncol <= 0:
            raise ValueError("Sfcprop allocation requires a positive integer ncol")
        return cls(
            **{
                field.name: torch.zeros(
                    (ncol, *R4_SURFACE_TAIL_SHAPES.get(field.name, ())),
                    dtype=torch.int32
                    if field.name in R4_SURFACE_INTEGER_MEMBERS
                    else dtype,
                    device=device,
                )
                for field in fields(cls)
            }
        )
