# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment R: cloud-base mass flux and scale-aware reduction."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch.compile import run_compile_island
from sa_sas_deep_convection_torch.segments.segment_d import _gather_column
from sa_sas_deep_convection_torch.state import SaSasDeepConvectionState, require_real

CHECKPOINT_FIELDS = ("xmb", "xmbmax", "sigmagfm", "scaldfunc", "cnvflg", "return_stage")


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 3020-3112 of ``samfdeepcnv_run`` for the approved active case."""

    cnvflg = state.require("cnvflg").clone()
    asqecflg = state.require("asqecflg").clone()
    kbcon = state.require("kbcon")
    po = state.require("po")
    to = state.require("to")
    xlamue = state.require("xlamue")
    garea = state.require("garea")
    gdx = state.require("gdx")
    dxcrtuf = state.require("dxcrtuf")
    advfac = state.require("advfac")
    aa1 = state.require("aa1")
    xaa0 = state.require("xaa0")
    mbdt = state.require("mbdt")
    dtconv = state.require("dtconv")
    wc = state.require("wc")
    sigmab = state.require("sigmab")
    sigmagfm = state.require("sigmagfm").clone()
    scaldfunc = state.require("scaldfunc").clone()
    xmb = state.require("xmb").clone()
    xmbmax = state.require("xmbmax")
    ncol = cnvflg.shape[0]
    for name, tensor in {
        "cnvflg": cnvflg,
        "asqecflg": asqecflg,
        "kbcon": kbcon,
        "garea": garea,
        "gdx": gdx,
        "dxcrtuf": dxcrtuf,
        "advfac": advfac,
        "aa1": aa1,
        "xaa0": xaa0,
        "mbdt": mbdt,
        "dtconv": dtconv,
        "wc": wc,
        "sigmab": sigmab,
        "sigmagfm": sigmagfm,
        "scaldfunc": scaldfunc,
        "xmb": xmb,
        "xmbmax": xmbmax,
    }.items():
        require_real(name, tensor, (ncol,))
    require_real("po", po, (ncol, state.config.get("km", po.shape[1])))
    require_real("to", to, (ncol, state.config.get("km", to.shape[1])))
    require_real("xlamue", xlamue, (ncol, state.config.get("km", xlamue.shape[1])))

    fld, xk, xmb, sigmagfm, scaldfunc, cnvflg, asqecflg = run_compile_island(
        "r_mass_flux_scale",
        "mass_flux_scale",
        _mass_flux_scale_block,
        cnvflg,
        asqecflg,
        kbcon,
        po,
        to,
        xlamue,
        garea,
        gdx,
        dxcrtuf,
        advfac,
        aa1,
        xaa0,
        mbdt,
        dtconv,
        wc,
        sigmab,
        sigmagfm,
        scaldfunc,
        xmb,
        xmbmax,
        state.require("rd"),
        state.require("omegac"),
        state.require("gravinv"),
        bool(state.config.get("progsigma", False)),
    )

    return state.with_updates(
        {
            "fld": fld,
            "xk": xk,
            "xmb": xmb,
            "sigmagfm": sigmagfm,
            "scaldfunc": scaldfunc,
            "cnvflg": cnvflg,
            "asqecflg": asqecflg,
            "return_stage": torch.zeros_like(cnvflg),
        }
    )


def _mass_flux_scale_block(
    cnvflg: torch.Tensor,
    asqecflg: torch.Tensor,
    kbcon: torch.Tensor,
    po: torch.Tensor,
    to: torch.Tensor,
    xlamue: torch.Tensor,
    garea: torch.Tensor,
    gdx: torch.Tensor,
    dxcrtuf: torch.Tensor,
    advfac: torch.Tensor,
    aa1: torch.Tensor,
    xaa0: torch.Tensor,
    mbdt: torch.Tensor,
    dtconv: torch.Tensor,
    wc: torch.Tensor,
    sigmab: torch.Tensor,
    sigmagfm: torch.Tensor,
    scaldfunc: torch.Tensor,
    xmb: torch.Tensor,
    xmbmax: torch.Tensor,
    rd: torch.Tensor,
    omegac: torch.Tensor,
    gravinv: torch.Tensor,
    progsigma: bool,
) -> tuple[torch.Tensor, ...]:
    active = cnvflg != 0.0
    asqec = asqecflg != 0.0
    non_asqec = active & ~asqec
    rho = _gather_column(po, kbcon) * 100.0 / (rd * _gather_column(to, kbcon))
    if progsigma:
        xmb_non_asqec = advfac * sigmab * (-omegac * gravinv)
    else:
        xmb_non_asqec = advfac * 0.03 * rho * wc
    xmb = torch.where(non_asqec, xmb_non_asqec, xmb)

    fld = aa1 / dtconv
    bad_fld = asqec & (fld <= 0.0)
    asqecflg = torch.where(bad_fld, torch.zeros_like(asqecflg), asqecflg)
    cnvflg = torch.where(bad_fld, torch.zeros_like(cnvflg), cnvflg)
    asqec = asqecflg != 0.0
    xk = (xaa0 - aa1) / mbdt
    bad_xk = asqec & (xk >= 0.0)
    asqecflg = torch.where(bad_xk, torch.zeros_like(asqecflg), asqecflg)
    cnvflg = torch.where(bad_xk, torch.zeros_like(cnvflg), cnvflg)
    asqec = asqecflg != 0.0
    # Inactive columns have xk=0. Mask the denominator before division so
    # their unused quotient cannot introduce NaNs in forward or reverse AD.
    xmb = torch.where(asqec, -advfac * fld / torch.where(asqec, xk, 1.0), xmb)

    active = cnvflg != 0.0
    xlamue_base = _gather_column(xlamue, kbcon)
    tem = torch.minimum(torch.maximum(xlamue_base, torch.full_like(xlamue_base, 7.0e-5)), torch.full_like(xlamue_base, 3.0e-4))
    tem = 0.2 / tem
    sigmagfm_candidate = torch.clamp(3.14 * tem * tem / garea, min=0.001, max=0.999)
    sigmagfm = torch.where(active, sigmagfm_candidate, sigmagfm)
    if progsigma:
        scale_candidate = (1.0 - sigmab) * (1.0 - sigmab)
    else:
        scale_candidate = (1.0 - sigmagfm) * (1.0 - sigmagfm)
    scale_candidate = torch.clamp(scale_candidate, min=0.0, max=1.0)
    scaldfunc_candidate = torch.where(gdx < dxcrtuf, scale_candidate, torch.ones_like(scale_candidate))
    scaldfunc = torch.where(active, scaldfunc_candidate, scaldfunc)
    xmb = torch.where(active, torch.minimum(xmb * scaldfunc, xmbmax), xmb)
    return fld, xk, xmb, sigmagfm, scaldfunc, cnvflg, asqecflg


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
