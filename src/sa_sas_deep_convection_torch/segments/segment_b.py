# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment B: pressure-grid and workspace seed."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch.compile import run_compile_island
from sa_sas_deep_convection_torch.constants import KM, NTC, NTR
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real

CHECKPOINT_FIELDS = (
    "ps",
    "prsl",
    "del",
    "do_aerosols",
    "cnvw",
    "cnvc",
    "QLCN",
    "QICN",
    "w_upi",
    "cf_upi",
)


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 346-502 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    psp = state.require("psp")
    prslp = state.require("prslp")
    delp = state.require("delp")
    t1 = state.require("t1")
    hpbl = state.require("hpbl")
    garea = state.require("garea")
    islimsk = state.require("islimsk")
    max_mf = state.require("maxMF")
    qtr = state.require("qtr")
    ncol = psp.shape[0]
    require_real("psp", psp, (ncol,))
    require_real("prslp", prslp, (ncol, KM))
    require_real("delp", delp, (ncol, KM))
    require_real("t1", t1, (ncol, KM))
    require_real("hpbl", hpbl, (ncol,))
    require_real("garea", garea, (ncol,))
    require_real("islimsk", islimsk, (ncol,))
    require_real("maxMF", max_mf, (ncol,))

    hwrf_samfdeep = bool(state.config.get("hwrf_samfdeep", False))
    do_mynnedmf = bool(state.config.get("do_mynnedmf", False))
    itc = int(state.config.get("itc", 0))
    ntc = int(state.config.get("ntc", 0))
    ntr = int(state.config.get("ntr", 0))
    mp_phys = int(state.config.get("mp_phys", 0))
    mp_phys_mg = int(state.config.get("mp_phys_mg", -1))

    (
        ps,
        prsl,
        del_,
        do_aerosols,
        chem_c,
        chem_pw,
        wet_dep,
        cnvflg,
        sfcpbl,
        rn,
        mbdt,
        kbot,
        ktop,
        kbcon,
        ktcon,
        ktconn,
        dtconv,
        cldwrk,
        zero_col,
        xlamud,
        xlamue,
        c0,
        c0t,
        cnvw,
        cnvc,
        ud_mf,
        dd_mf,
        dt_mf,
        qlcn,
        qicn,
        km1,
        scaldfunc,
        sigmagfm,
    ) = run_compile_island(
        "b_workspace_seed",
        "workspace_seed",
        _workspace_seed_block,
        psp,
        prslp,
        delp,
        t1,
        hpbl,
        garea,
        islimsk,
        max_mf,
        qtr,
        state.require("c0s"),
        state.require("asolfac"),
        hwrf_samfdeep,
        do_mynnedmf,
        itc,
        ntc,
        ntr,
        mp_phys,
        mp_phys_mg,
    )
    return state.with_updates(
        {
            "ps": ps,
            "prsl": prsl,
            "del": del_,
            "do_aerosols": do_aerosols,
            "km1": km1,
            "chem_c": chem_c,
            "chem_pw": chem_pw,
            "wet_dep": wet_dep,
            "cnvflg": cnvflg,
            "sfcpbl": sfcpbl,
            "rn": rn,
            "mbdt": mbdt,
            "kbot": kbot,
            "ktop": ktop,
            "kbcon": kbcon,
            "ktcon": ktcon,
            "ktconn": ktconn,
            "dtconv": dtconv,
            "cldwrk": cldwrk,
            "pdot": zero_col.clone(),
            "lmin": torch.ones_like(rn),
            "jmin": torch.ones_like(rn),
            "qlko_ktcon": zero_col.clone(),
            "edt": zero_col.clone(),
            "edto": zero_col.clone(),
            "edtx": zero_col.clone(),
            "aa1": zero_col.clone(),
            "aa2": zero_col.clone(),
            "xaa0": zero_col.clone(),
            "cina": zero_col.clone(),
            "pwavo": zero_col.clone(),
            "pwevo": zero_col.clone(),
            "xmb": zero_col.clone(),
            "xpwav": zero_col.clone(),
            "xpwev": zero_col.clone(),
            "vshear": zero_col.clone(),
            "advfac": zero_col.clone(),
            "rainevap": zero_col.clone(),
            "omegac": zero_col.clone(),
            "sigmab": zero_col.clone(),
            "gdx": torch.sqrt(garea),
            "xlamud": xlamud,
            "xlamue": xlamue,
            "scaldfunc": scaldfunc,
            "sigmagfm": sigmagfm,
            "c0": c0,
            "c0t": c0t,
            "cnvw": cnvw,
            "cnvc": cnvc,
            "ud_mf": ud_mf,
            "dd_mf": dd_mf,
            "dt_mf": dt_mf,
            "QLCN": qlcn,
            "QICN": qicn,
            "w_upi": torch.zeros_like(xlamud),
            "cf_upi": torch.zeros_like(xlamud),
            "CNV_MFD": torch.zeros_like(xlamud),
            "CNV_DQLDT": torch.zeros_like(xlamud),
            "CLCN": torch.zeros_like(xlamud),
            "CNV_FICE": torch.zeros_like(xlamud),
            "CNV_NDROP": torch.zeros_like(xlamud),
            "CNV_NICE": torch.zeros_like(xlamud),
        }
    )


def _workspace_seed_block(
    psp: torch.Tensor,
    prslp: torch.Tensor,
    delp: torch.Tensor,
    t1: torch.Tensor,
    hpbl: torch.Tensor,
    garea: torch.Tensor,
    islimsk: torch.Tensor,
    max_mf: torch.Tensor,
    qtr: torch.Tensor,
    c0s: torch.Tensor,
    asolfac: torch.Tensor,
    hwrf_samfdeep: bool,
    do_mynnedmf: bool,
    itc: int,
    ntc: int,
    ntr: int,
    mp_phys: int,
    mp_phys_mg: int,
) -> tuple[torch.Tensor, ...]:
    KM = t1.shape[1]
    NTC = ntc
    ncol = psp.shape[0]
    do_aerosols_scalar = False if hwrf_samfdeep else (itc > 0 and ntc > 0 and ntr > 0 and ntr >= itc + ntc - 3)
    do_aerosols = torch.full((ncol,), 1.0 if do_aerosols_scalar else 0.0, dtype=psp.dtype, device=psp.device)
    ps = psp * 0.001
    prsl = prslp * 0.001
    del_ = delp * 0.001
    chem_c = torch.zeros((ncol, KM, NTC), dtype=psp.dtype, device=psp.device)
    chem_pw = torch.zeros_like(chem_c)
    wet_dep = torch.zeros_like(chem_c)
    cnvflg = torch.ones((ncol,), dtype=psp.dtype, device=psp.device)
    if do_mynnedmf:
        cnvflg = torch.where(max_mf > 0.0, torch.zeros_like(cnvflg), cnvflg)
    sfcpbl = 0.2 * hpbl
    rn = torch.zeros((ncol,), dtype=psp.dtype, device=psp.device)
    mbdt = torch.full_like(rn, 10.0)
    kbot = torch.full_like(rn, float(KM + 1))
    ktop = torch.zeros_like(rn)
    kbcon = torch.full_like(rn, float(KM))
    ktcon = torch.ones_like(rn)
    ktconn = torch.ones_like(rn)
    dtconv = torch.full_like(rn, 3600.0)
    cldwrk = torch.zeros_like(rn)
    zero_col = torch.zeros_like(rn)
    xlamud = torch.zeros((ncol, KM), dtype=psp.dtype, device=psp.device)
    xlamue = torch.zeros_like(xlamud)
    c0 = torch.where(islimsk == 1.0, c0s * asolfac, c0s.expand_as(islimsk))
    tem = 0.001 * (t1 - 273.16)
    c0t = torch.where(t1 > 273.16, c0[:, None], c0[:, None] * torch.exp(tem))
    cnvw = torch.zeros_like(xlamud)
    cnvc = torch.zeros_like(xlamud)
    ud_mf = torch.zeros_like(xlamud)
    dd_mf = torch.zeros_like(xlamud)
    dt_mf = torch.zeros_like(xlamud)
    qlcn = torch.zeros_like(xlamud)
    qicn = torch.zeros_like(xlamud)
    if mp_phys == mp_phys_mg:
        qlcn = qtr[:, :, 1].clone()
        qicn = qtr[:, :, 0].clone()
    km1 = torch.full((), float(KM - 1), dtype=psp.dtype, device=psp.device)
    scaldfunc = torch.full_like(rn, -1.0) if hwrf_samfdeep else zero_col.clone()
    sigmagfm = torch.full_like(rn, -1.0) if hwrf_samfdeep else zero_col.clone()
    return (
        ps,
        prsl,
        del_,
        do_aerosols,
        chem_c,
        chem_pw,
        wet_dep,
        cnvflg,
        sfcpbl,
        rn,
        mbdt,
        kbot,
        ktop,
        kbcon,
        ktcon,
        ktconn,
        dtconv,
        cldwrk,
        zero_col,
        xlamud,
        xlamue,
        c0,
        c0t,
        cnvw,
        cnvc,
        ud_mf,
        dd_mf,
        dt_mf,
        qlcn,
        qicn,
        km1,
        scaldfunc,
        sigmagfm,
    )


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
