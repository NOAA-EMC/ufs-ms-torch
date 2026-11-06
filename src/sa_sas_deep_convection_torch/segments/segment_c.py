# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment C: vertical geometry and initial thermodynamic state."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch.compile import run_compile_island
from sa_sas_deep_convection_torch.constants import EPS, EPSM1, FV, GRAV, KM, NN, NTR
from sa_sas_deep_convection_torch.segments.segment_d import _fpvs_from_table
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real
from sa_sas_deep_convection_torch.thermo import _fpvs_table_for_device

CHECKPOINT_FIELDS = ("qeso", "heo", "heso", "dbyo", "zo", "kmax", "kbm", "kbmax", "cnvflg")


def _masked_fill_field(base: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, values, base)


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 516-690 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    ps = state.require("ps")
    prsl = state.require("prsl")
    phil = state.require("phil")
    t1 = state.require("t1")
    q1 = state.require("q1")
    u1 = state.require("u1")
    v1 = state.require("v1")
    qtr = state.require("qtr")
    cnvflg = state.require("cnvflg")
    sfcpbl = state.require("sfcpbl")
    ncol = ps.shape[0]
    require_real("ps", ps, (ncol,))
    require_real("prsl", prsl, (ncol, KM))
    require_real("phil", phil, (ncol, KM))
    require_real("t1", t1, (ncol, KM))
    require_real("q1", q1, (ncol, KM))
    require_real("u1", u1, (ncol, KM))
    require_real("v1", v1, (ncol, KM))
    require_real("qtr", qtr, (ncol, KM, NN))
    require_real("cnvflg", cnvflg, (ncol,))
    require_real("sfcpbl", sfcpbl, (ncol,))

    hwrf_samfdeep = bool(state.config.get("hwrf_samfdeep", False))
    (
        kmax,
        kbm,
        kbmax,
        kd94,
        tx1,
        zo,
        zi,
        xlamue,
        pfld,
        eta,
        fent1,
        fent2,
        rh,
        frh,
        hcko,
        qcko,
        qrcko,
        ucko,
        vcko,
        etad,
        hcdo,
        qcdo,
        ucdo,
        vcdo,
        qrcd,
        qrcdo,
        dbyo,
        pwo,
        pwdo,
        dellal,
        to,
        qo,
        uo,
        vo,
        wu2,
        buo,
        wush,
        drag,
        cnvwt,
        dbyo1,
        zdqca,
        omega_u,
        zeta,
        ctr,
        ctro,
        ecko,
        ercko,
        ecdo,
        qeso,
        heo,
        heso,
        tvo,
        scalar_constants,
    ) = run_compile_island(
        "c_geometry_thermo",
        "geometry_thermo_seed",
        _geometry_thermo_seed_block,
        ps,
        prsl,
        phil,
        t1,
        q1,
        u1,
        v1,
        # samf transports qtr(:,:,3:ntr+2); slots above ntr+2 (layout nn > ntr+2) pass through.
        qtr[:, :, : NTR + 2],
        cnvflg,
        sfcpbl,
        state.require("xlamue"),
        state.require("clam"),
        state.require("cp"),
        state.require("hvap"),
        _fpvs_table_for_device(str(ps.device), ps.dtype),
        hwrf_samfdeep,
    )
    (
        cxlame,
        cxlamd,
        crtlame,
        crtlamd,
        xlamde,
        xlamdd,
        w1l,
        w2l,
        w3l,
        w4l,
        w1s,
        w2s,
        w3s,
        w4s,
    ) = scalar_constants

    return state.with_updates(
        {
            "cxlame": cxlame,
            "cxlamd": cxlamd,
            "crtlame": crtlame,
            "crtlamd": crtlamd,
            "xlamde": xlamde,
            "xlamdd": xlamdd,
            "w1l": w1l,
            "w2l": w2l,
            "w3l": w3l,
            "w4l": w4l,
            "w1s": w1s,
            "w2s": w2s,
            "w3s": w3s,
            "w4s": w4s,
            "kmax": kmax,
            "kbm": kbm,
            "kbmax": kbmax,
            "kd94": kd94,
            "tx1": tx1,
            "zo": zo,
            "zi": zi,
            "xlamue": xlamue,
            "pfld": pfld,
            "eta": eta,
            "fent1": fent1,
            "fent2": fent2,
            "rh": rh,
            "frh": frh,
            "hcko": hcko,
            "qcko": qcko,
            "qrcko": qrcko,
            "ucko": ucko,
            "vcko": vcko,
            "etad": etad,
            "hcdo": hcdo,
            "qcdo": qcdo,
            "ucdo": ucdo,
            "vcdo": vcdo,
            "qrcd": qrcd,
            "qrcdo": qrcdo,
            "dbyo": dbyo,
            "pwo": pwo,
            "pwdo": pwdo,
            "dellal": dellal,
            "to": to,
            "qo": qo,
            "uo": uo,
            "vo": vo,
            "wu2": wu2,
            "buo": buo,
            "wush": wush,
            "drag": drag,
            "cnvwt": cnvwt,
            "dbyo1": dbyo1,
            "zdqca": zdqca,
            "omega_u": omega_u,
            "zeta": zeta,
            "ctr": ctr,
            "ctro": ctro,
            "ecko": ecko,
            "ercko": ercko,
            "ecdo": ecdo,
            "qeso": qeso,
            "heo": heo,
            "heso": heso,
            "tvo": tvo,
        }
    )


def _geometry_thermo_seed_block(
    ps: torch.Tensor,
    prsl: torch.Tensor,
    phil: torch.Tensor,
    t1: torch.Tensor,
    q1: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    qtr: torch.Tensor,
    cnvflg: torch.Tensor,
    sfcpbl: torch.Tensor,
    xlamue_input: torch.Tensor,
    clam: torch.Tensor,
    cp: torch.Tensor,
    hvap: torch.Tensor,
    fpvs_table: torch.Tensor,
    hwrf_samfdeep: bool,
) -> tuple[torch.Tensor, ...]:
    KM = q1.shape[1]
    NTR = qtr.shape[2] - 2
    ncol = ps.shape[0]
    one_col = torch.ones((ncol,), dtype=ps.dtype, device=ps.device)
    kmax = torch.full_like(one_col, float(KM))
    kbm = torch.full_like(one_col, float(KM))
    kbmax = torch.full_like(one_col, float(KM))
    kd94 = torch.full_like(one_col, float(KM))
    tx1 = 1.0 / ps
    ratios = prsl * tx1[:, None]
    for level in range(KM):
        one_based_next = float(level + 2)
        kmax = torch.where(ratios[:, level] > 0.04, torch.full_like(kmax, one_based_next), kmax)
        kbmax = torch.where(ratios[:, level] > 0.45, torch.full_like(kbmax, one_based_next), kbmax)
        kbm = torch.where(ratios[:, level] > 0.70, torch.full_like(kbm, one_based_next), kbm)
        kd94 = torch.where(ratios[:, level] > 0.94, torch.full_like(kd94, one_based_next), kd94)
    kmax = torch.minimum(kmax, torch.full_like(kmax, float(KM)))
    kbmax = torch.minimum(kbmax, kmax)
    kbm = torch.minimum(kbm, kmax)
    kd94 = torch.minimum(kd94, kmax)

    zo = phil / GRAV
    zi = torch.zeros_like(zo)
    zi[:, : KM - 1] = 0.5 * (zo[:, : KM - 1] + zo[:, 1:KM])
    xlamue = xlamue_input.clone()
    if hwrf_samfdeep:
        xlamue[:, : KM - 1] = clam / zi[:, : KM - 1]

    level_ids = torch.arange(1, KM + 1, dtype=ps.dtype, device=ps.device).reshape(1, KM)
    active = level_ids <= kmax[:, None]
    zeros = torch.zeros((ncol, KM), dtype=ps.dtype, device=ps.device)
    ones = torch.ones_like(zeros)
    pfld = _masked_fill_field(zeros, prsl * 10.0, active)
    eta = _masked_fill_field(zeros, ones, active)
    fent1 = eta.clone()
    fent2 = eta.clone()
    rh = zeros.clone()
    frh = zeros.clone()
    hcko = zeros.clone()
    qcko = zeros.clone()
    qrcko = zeros.clone()
    ucko = zeros.clone()
    vcko = zeros.clone()
    etad = eta.clone()
    hcdo = zeros.clone()
    qcdo = zeros.clone()
    ucdo = zeros.clone()
    vcdo = zeros.clone()
    qrcd = zeros.clone()
    qrcdo = zeros.clone()
    dbyo = zeros.clone()
    pwo = zeros.clone()
    pwdo = zeros.clone()
    dellal = zeros.clone()
    to = _masked_fill_field(zeros, t1, active)
    qo = _masked_fill_field(zeros, q1, active)
    uo = _masked_fill_field(zeros, u1, active)
    vo = _masked_fill_field(zeros, v1, active)
    wu2 = zeros.clone()
    buo = zeros.clone()
    wush = zeros.clone()
    drag = zeros.clone()
    cnvwt = zeros.clone()
    dbyo1 = zeros.clone()
    zdqca = zeros.clone()
    omega_u = zeros.clone()
    zeta = ones.clone()

    ctr = torch.zeros((ncol, KM, NTR), dtype=ps.dtype, device=ps.device)
    ctro = torch.zeros_like(ctr)
    ecko = torch.zeros_like(ctr)
    ercko = torch.zeros_like(ctr)
    ecdo = torch.zeros_like(ctr)
    if not hwrf_samfdeep:
        ctr = qtr[:, :, 2 : 2 + NTR].clone()
        ctro = ctr.clone()
        ctr = torch.where(active[:, :, None], ctr, torch.zeros_like(ctr))
        ctro = torch.where(active[:, :, None], ctro, torch.zeros_like(ctro))

    qeso = torch.zeros_like(pfld)
    qeso_candidate = 0.01 * _fpvs_from_table(to, fpvs_table)
    qeso_candidate = EPS * qeso_candidate / (pfld + EPSM1 * qeso_candidate)
    qeso_candidate = torch.clamp_min(qeso_candidate, 1.0e-8)
    qeso = _masked_fill_field(qeso, qeso_candidate, active)
    qo = _masked_fill_field(qo, torch.clamp_min(qo, 1.0e-10), active)
    moist_static_base = phil + cp * to
    heo = _masked_fill_field(zeros, moist_static_base + hvap * qo, active)
    heso = _masked_fill_field(zeros, moist_static_base + hvap * qeso, active)
    tvo = to + FV * to * qo
    scalar_constants = (
        torch.full((), 1.0e-3 if hwrf_samfdeep else 1.0e-4, dtype=ps.dtype, device=ps.device),
        torch.full((), 0.75e-4, dtype=ps.dtype, device=ps.device),
        torch.full((), 1.0e-4, dtype=ps.dtype, device=ps.device),
        torch.full((), 1.0e-4, dtype=ps.dtype, device=ps.device),
        torch.full((), 1.0e-4, dtype=ps.dtype, device=ps.device),
        torch.full((), 1.0e-4, dtype=ps.dtype, device=ps.device),
        torch.full((), -8.0e-3, dtype=ps.dtype, device=ps.device),
        torch.full((), -4.0e-2, dtype=ps.dtype, device=ps.device),
        torch.full((), -5.0e-3, dtype=ps.dtype, device=ps.device),
        torch.full((), -5.0e-4, dtype=ps.dtype, device=ps.device),
        torch.full((), -2.0e-4, dtype=ps.dtype, device=ps.device),
        torch.full((), -2.0e-3, dtype=ps.dtype, device=ps.device),
        torch.full((), -1.0e-3, dtype=ps.dtype, device=ps.device),
        torch.full((), -2.0e-5, dtype=ps.dtype, device=ps.device),
    )
    return (
        kmax,
        kbm,
        kbmax,
        kd94,
        tx1,
        zo,
        zi,
        xlamue,
        pfld,
        eta,
        fent1,
        fent2,
        rh,
        frh,
        hcko,
        qcko,
        qrcko,
        ucko,
        vcko,
        etad,
        hcdo,
        qcdo,
        ucdo,
        vcdo,
        qrcd,
        qrcdo,
        dbyo,
        pwo,
        pwdo,
        dellal,
        to,
        qo,
        uo,
        vo,
        wu2,
        buo,
        wush,
        drag,
        cnvwt,
        dbyo1,
        zdqca,
        omega_u,
        zeta,
        ctr,
        ctro,
        ecko,
        ercko,
        ecdo,
        qeso,
        heo,
        heso,
        tvo,
        scalar_constants,
    )


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
