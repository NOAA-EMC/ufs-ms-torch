# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment M: TKE/TTE production, dissipation, and predictor update."""

from __future__ import annotations

import torch
from ufs_physics_common.precision import working_power

from sa_tke_edmf_torch.autodiff import safe_positive_denominator, safe_sqrt_nonnegative
from sa_tke_edmf_torch.compile import run_compile_island
from sa_tke_edmf_torch.segments.segment_g import levels_over
from sa_tke_edmf_torch.constants import F32_0P19, F32_0P2, F32_0P4, F32_0P51, F32_ONE_THIRD, IM, KM, NTKE, source_real
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64

CHECKPOINT_FIELDS = ("prod", "diss", "te", "tke")


def _tke_production_inputs(
    dku: torch.Tensor,
    dkt: torch.Tensor,
    bf: torch.Tensor,
    shr2: torch.Tensor,
    ckz: torch.Tensor,
    xkzmo: torch.Tensor,
    elm: torch.Tensor,
    ele: torch.Tensor,
    gotvx: torch.Tensor,
    sflux: torch.Tensor,
    xmf: torch.Tensor,
    buou: torch.Tensor,
    xmfd: torch.Tensor,
    buod: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    ucko: torch.Tensor,
    vcko: torch.Tensor,
    ucdo: torch.Tensor,
    vcdo: torch.Tensor,
    zi: torch.Tensor,
    zl: torch.Tensor,
    rdzt: torch.Tensor,
    stress: torch.Tensor,
    ustar: torch.Tensor,
    phims: torch.Tensor,
    dku_h: torch.Tensor,
    def_1: torch.Tensor,
    def_2: torch.Tensor,
    def_3: torch.Tensor,
    garea: torch.Tensor,
    ele_les: torch.Tensor,
    pftke: torch.Tensor,
    kpbl: torch.Tensor,
    mrad: torch.Tensor,
    krad: torch.Tensor,
    pcnvflg: torch.Tensor,
    scuflg: torch.Tensor,
    dtn: torch.Tensor,
    ce0: torch.Tensor,
    sa3dtke: bool,
    tte_edmf: bool,
    coefficients: tuple[float, ...] = (
        F32_0P19, F32_0P2, F32_0P4, F32_0P51, F32_ONE_THIRD,
    ),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    F32_0P19, F32_0P2, F32_0P4, F32_0P51, F32_ONE_THIRD = coefficients
    KM = dku.shape[1]
    dtype = dku.dtype
    device = dku.device
    ncol = dku.shape[0]
    km1 = KM - 1

    first_tkmnz = 0.5 * xkzmo[:, 0] / safe_positive_denominator(ckz[:, 0] * elm[:, 0])
    tkmnz = torch.empty((ncol, km1), dtype=dtype, device=device)
    tkmnz[:, 0] = torch.clamp(first_tkmnz * first_tkmnz, min=1.0e-9, max=F32_0P2)
    tem_tkmnz = 0.5 * (ckz[:, : km1 - 1] + ckz[:, 1:km1])
    tem1_tkmnz = 0.5 * (xkzmo[:, : km1 - 1] + xkzmo[:, 1:km1])
    ptem_tkmnz = tem1_tkmnz / safe_positive_denominator(tem_tkmnz * elm[:, 1:km1])
    tkmnz[:, 1:km1] = torch.clamp(ptem_tkmnz * ptem_tkmnz, min=1.0e-9, max=F32_0P2)

    prod = torch.empty((ncol, km1), dtype=dtype, device=device)
    buoy0 = -dkt[:, 0] * bf[:, 0]
    downdraft_buoy0 = torch.where(scuflg & (mrad == 1), xmfd[:, 0] * buod[:, 0], torch.zeros((ncol,), dtype=dtype, device=device))
    buop0 = 0.5 * (gotvx[:, 0] * sflux + (buoy0 + downdraft_buoy0))
    if sa3dtke:
        shear0 = dku[:, 0] * def_1[:, 0] + 2.0 * dku_h[:, 0] * def_2[:, 0]
    else:
        shear0 = dku[:, 0] * shr2[:, 0]
    du0 = (u1[:, 1] - u1[:, 0]) * rdzt[:, 0]
    dv0 = (v1[:, 1] - v1[:, 0]) * rdzt[:, 0]
    downdraft_u0 = torch.where(
        scuflg & (mrad == 1),
        0.5 * du0 * xmfd[:, 0] * (ucdo[:, 0] + ucdo[:, 1] - u1[:, 0] - u1[:, 1]),
        torch.zeros((ncol,), dtype=dtype, device=device),
    )
    downdraft_v0 = torch.where(
        scuflg & (mrad == 1),
        0.5 * dv0 * xmfd[:, 0] * (vcdo[:, 0] + vcdo[:, 1] - v1[:, 0] - v1[:, 1]),
        torch.zeros((ncol,), dtype=dtype, device=device),
    )
    surface_denominator = F32_0P4 * zl[:, 0]
    finite_surface_shear = stress * ustar * phims / safe_positive_denominator(surface_denominator)
    surface_shear = torch.where(
        torch.abs(surface_denominator) > 1.0e-30,
        finite_surface_shear,
        torch.full_like(finite_surface_shear, torch.inf),
    )
    shrp0 = 0.5 * (shear0 + downdraft_u0 + downdraft_v0 + surface_shear)
    prod[:, 0] = torch.where(buop0 > 0.0, 2.0 * buop0 + shrp0, shrp0) if tte_edmf else buop0 + shrp0

    levels = torch.arange(2, KM, dtype=torch.int64, device=device)
    updraft_active = pcnvflg[:, None] & (levels[None, :] <= kpbl[:, None])
    downdraft_active = scuflg[:, None] & (levels[None, :] >= mrad[:, None]) & (levels[None, :] < krad[:, None])
    zeros_2d = torch.zeros((ncol, km1 - 1), dtype=dtype, device=device)
    buoy = 0.5 * (-dkt[:, : km1 - 1] * bf[:, : km1 - 1] - dkt[:, 1:km1] * bf[:, 1:km1])
    updraft_buoy = torch.where(
        updraft_active,
        0.5 * (xmf[:, : km1 - 1] + xmf[:, 1:km1]) * buou[:, 1:km1],
        zeros_2d,
    )
    downdraft_buoy = torch.where(
        downdraft_active,
        0.5 * (xmfd[:, : km1 - 1] + xmfd[:, 1:km1]) * buod[:, 1:km1],
        zeros_2d,
    )
    buop = buoy + updraft_buoy + downdraft_buoy
    if sa3dtke:
        shear = 0.5 * (
            dku[:, : km1 - 1] * def_1[:, : km1 - 1]
            + dku[:, 1:km1] * def_1[:, 1:km1]
            + 2.0 * dku_h[:, 1:km1] * def_2[:, 1:km1]
        )
    else:
        shear = 0.5 * (dku[:, : km1 - 1] * shr2[:, : km1 - 1] + dku[:, 1:km1] * shr2[:, 1:km1])
    du_up = (u1[:, 2:KM] - u1[:, 1:km1]) * rdzt[:, 1:km1]
    du_dn = (u1[:, 1:km1] - u1[:, : km1 - 1]) * rdzt[:, : km1 - 1]
    updraft_u = torch.where(
        updraft_active,
        0.5 * (xmf[:, 1:km1] * du_up + xmf[:, : km1 - 1] * du_dn) * (u1[:, 1:km1] - ucko[:, 1:km1]),
        zeros_2d,
    )
    downdraft_u = torch.where(
        downdraft_active,
        0.5 * (xmfd[:, 1:km1] * du_up + xmfd[:, : km1 - 1] * du_dn) * (ucdo[:, 1:km1] - u1[:, 1:km1]),
        zeros_2d,
    )
    dv_up = (v1[:, 2:KM] - v1[:, 1:km1]) * rdzt[:, 1:km1]
    dv_dn = (v1[:, 1:km1] - v1[:, : km1 - 1]) * rdzt[:, : km1 - 1]
    updraft_v = torch.where(
        updraft_active,
        0.5 * (xmf[:, 1:km1] * dv_up + xmf[:, : km1 - 1] * dv_dn) * (v1[:, 1:km1] - vcko[:, 1:km1]),
        zeros_2d,
    )
    downdraft_v = torch.where(
        downdraft_active,
        0.5 * (xmfd[:, 1:km1] * dv_up + xmfd[:, : km1 - 1] * dv_dn) * (vcdo[:, 1:km1] - v1[:, 1:km1]),
        zeros_2d,
    )
    shrp = shear + updraft_u + downdraft_u + updraft_v + downdraft_v
    prod[:, 1:km1] = torch.where(buop > 0.0, 2.0 * buop + shrp, shrp) if tte_edmf else buop + shrp

    if sa3dtke:
        dz = zi[:, 1:KM] - zi[:, :km1]
        tem1 = working_power(torch.clamp(garea[:, None] * dz, min=0.0), F32_ONE_THIRD)
        ptem1 = ce0 / safe_positive_denominator(ele[:, :km1])
        ptem2 = (F32_0P19 + F32_0P51 * ele_les[:, :km1] / safe_positive_denominator(tem1)) / safe_positive_denominator(ele_les[:, :km1])
        pftke_column = pftke[:, None]
        ptem_diss = (1.0 - pftke_column) * ptem2 + pftke_column * ptem1
        def_3_head = def_3[:, :km1]
    else:
        ptem_diss = ce0 / safe_positive_denominator(ele[:, :km1])
        def_3_head = torch.zeros((ncol, km1), dtype=dtype, device=device)
    return tkmnz, prod, ptem_diss, def_3_head


def _advance_te_substeps_base(
    te_head: torch.Tensor,
    prod: torch.Tensor,
    tkmnz: torch.Tensor,
    ptem_diss: torch.Tensor,
    def_3_head: torch.Tensor,
    dtn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    diss = torch.zeros_like(te_head)
    for _ in range(20):
        disste = ptem_diss * te_head * safe_sqrt_nonnegative(te_head)
        limit = prod + te_head / dtn
        disste = torch.maximum(torch.minimum(disste, limit), torch.zeros_like(disste))
        diss = torch.where(torch.isposinf(prod), torch.full_like(disste, torch.inf), disste)
        tendency = prod - disste + def_3_head
        finite_tendency = torch.where(torch.isfinite(tendency), tendency, torch.zeros_like(tendency))
        next_te = torch.where(torch.isfinite(tendency), te_head + dtn * finite_tendency, tkmnz)
        te_head = torch.maximum(next_te, tkmnz)
    return te_head, diss


def _advance_te_substeps_tte(
    te_head: torch.Tensor,
    tke_head: torch.Tensor,
    prod: torch.Tensor,
    tkmnz: torch.Tensor,
    ptem_diss: torch.Tensor,
    def_3_head: torch.Tensor,
    dtn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    for _ in range(20):
        disste = ptem_diss * te_head * safe_sqrt_nonnegative(te_head)
        limit = prod + te_head / dtn
        disste = torch.maximum(torch.minimum(disste, limit), torch.zeros_like(disste))
        tendency = prod - disste + def_3_head
        finite_tendency = torch.where(torch.isfinite(tendency), tendency, torch.zeros_like(tendency))
        next_te = torch.where(torch.isfinite(tendency), te_head + dtn * finite_tendency, tkmnz)
        te_head = torch.maximum(next_te, tkmnz)
    diss = ptem_diss * tke_head * safe_sqrt_nonnegative(tke_head)
    return te_head, diss


def _update_tke_plume_recurrences(
    te: torch.Tensor,
    qcko_tke: torch.Tensor,
    qcdo_tke: torch.Tensor,
    xlamue: torch.Tensor,
    xlamde: torch.Tensor,
    zl: torch.Tensor,
    kpbl: torch.Tensor,
    mrad: torch.Tensor,
    krad: torch.Tensor,
    pcnvflg: torch.Tensor,
    scuflg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    KM = te.shape[1]
    kmpbl = KM // 2
    qcko_tke_cols = [qcko_tke[:, 0]]
    for k in range(1, kmpbl):
        active = pcnvflg & (k + 1 <= kpbl)
        dz = zl[:, k] - zl[:, k - 1]
        tem = 0.5 * xlamue[:, k - 1] * dz
        factor = 1.0 + tem
        qcko_tke_cols.append(
            torch.where(
                active,
                ((1.0 - tem) * qcko_tke_cols[-1] + tem * (te[:, k] + te[:, k - 1])) / factor,
                qcko_tke[:, k],
            )
        )
    qcko_tke = levels_over(qcko_tke, 1, qcko_tke_cols[1:])

    qcdo_tke_cols = [qcdo_tke[:, k] for k in range(kmpbl + 1)]
    for k in range(kmpbl - 1, -1, -1):
        active = scuflg & (k + 1 < krad) & (k + 1 >= mrad)
        if k + 1 < KM:
            dz = zl[:, k + 1] - zl[:, k]
            tem = 0.5 * xlamde[:, k] * dz
            factor = 1.0 + tem
            qcdo_tke_cols[k] = torch.where(
                active,
                ((1.0 - tem) * qcdo_tke_cols[k + 1] + tem * (te[:, k] + te[:, k + 1])) / factor,
                qcdo_tke_cols[k],
            )
    qcdo_tke = levels_over(qcdo_tke, 0, qcdo_tke_cols[:kmpbl])
    return qcko_tke, qcdo_tke


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    NTKE = state.require("q1").shape[2]
    KM = state.require("q1").shape[1]
    dku = state.require("dku")
    dkt = state.require("dkt")
    bf = state.require("bf")
    shr2 = state.require("shr2")
    ckz = state.require("ckz")
    xkzmo = state.require("xkzmo")
    elm = state.require("elm")
    ele = state.require("ele")
    te = state.require("te").clone()
    tke = state.require("tke").clone()
    gotvx = state.require("gotvx")
    sflux = state.require("sflux")
    xmf = state.require("xmf")
    buou = state.require("buou")
    xmfd = state.require("xmfd")
    buod = state.require("buod")
    u1 = state.require("u1")
    v1 = state.require("v1")
    ucdo = state.fields.get("ucdo")
    if ucdo is None:
        ucdo = torch.zeros_like(u1)
    vcdo = state.fields.get("vcdo")
    if vcdo is None:
        vcdo = torch.zeros_like(v1)
    ucko = state.fields.get("ucko")
    if ucko is None:
        ucko = torch.zeros_like(u1)
    vcko = state.fields.get("vcko")
    if vcko is None:
        vcko = torch.zeros_like(v1)
    qcko = state.require("qcko").clone()
    qcdo = state.require("qcdo").clone()
    xlamue = state.require("xlamue")
    xlamde = state.require("xlamde")
    zi = state.require("zi")
    zl = state.require("zl")
    rdzt = state.require("rdzt")
    stress = state.require("stress")
    ustar = state.require("ustar")
    phims = state.require("phims")
    dku_h = state.require("dku_h")
    def_1 = state.require("def_1")
    def_2 = state.require("def_2")
    def_3 = state.require("def_3")
    garea = state.require("garea")
    ele_les = state.require("ele_les")
    pftke = state.require("pftke")
    kpbl = state.require("kpbl").to(torch.int64)
    mrad = state.require("mrad").to(torch.int64)
    krad = state.require("krad").to(torch.int64)
    pcnvflg = state.require("pcnvflg") > 0.5
    scuflg = state.require("scuflg") > 0.5
    IM = dku.shape[0]
    require_float64("dku", dku, (IM, KM))
    require_float64("dkt", dkt, (IM, KM))
    require_float64("bf", bf, (IM, KM - 1))
    require_float64("shr2", shr2, (IM, KM - 1))
    require_float64("ckz", ckz, (IM, KM))
    require_float64("xkzmo", xkzmo, (IM, KM - 1))
    require_float64("elm", elm, (IM, KM))
    require_float64("ele", ele, (IM, KM))
    require_float64("te", te, (IM, KM))
    require_float64("tke", tke, (IM, KM))
    require_float64("gotvx", gotvx, (IM, KM))
    require_float64("xmf", xmf, (IM, KM))
    require_float64("buou", buou, (IM, KM))
    require_float64("xmfd", xmfd, (IM, KM))
    require_float64("buod", buod, (IM, KM))

    dtype = te.dtype
    device = te.device
    km1 = KM - 1
    kmpbl = KM // 2
    tte_edmf = bool(state.config.get("tte_edmf", False))
    sa3dtke = bool(state.config.get("sa3dtke", False))
    tc_pbl = bool(state.config.get("tc_pbl", False))
    ce0 = torch.full((), source_real(0.12 if tc_pbl else 0.4, state.config),
                       dtype=dtype, device=device)
    dt2 = state.require("delt")[0]
    dtn = dt2 / 20.0

    tkmnz, prod, ptem_diss, def_3_head = run_compile_island(
        "m_tendency_block",
        "production_inputs",
        _tke_production_inputs,
        dku,
        dkt,
        bf,
        shr2,
        ckz,
        xkzmo,
        elm,
        ele,
        gotvx,
        sflux,
        xmf,
        buou,
        xmfd,
        buod,
        u1,
        v1,
        ucko,
        vcko,
        ucdo,
        vcdo,
        zi,
        zl,
        rdzt,
        stress,
        ustar,
        phims,
        dku_h,
        def_1,
        def_2,
        def_3,
        garea,
        ele_les,
        pftke,
        kpbl,
        mrad,
        krad,
        pcnvflg,
        scuflg,
        dtn,
        ce0,
        sa3dtke,
        tte_edmf,
        tuple(source_real(value, state.config) for value in
              (0.19, 0.2, 0.4, 0.51, 0.33333333)),
    )
    te_head = te[:, :km1]
    te_tail = te[:, KM - 1]
    if tte_edmf:
        te_head, diss = run_compile_island(
            "m_tke_recurrence",
            "advance_te_substeps_tte",
            _advance_te_substeps_tte,
            te_head,
            tke[:, :km1],
            prod,
            tkmnz,
            ptem_diss,
            def_3_head,
            dtn,
        )
    else:
        te_head, diss = run_compile_island(
            "m_tke_recurrence",
            "advance_te_substeps_base",
            _advance_te_substeps_base,
            te_head,
            prod,
            tkmnz,
            ptem_diss,
            def_3_head,
            dtn,
        )
    te_next = torch.empty_like(te)
    te_next[:, :km1] = te_head
    te_next[:, KM - 1] = te_tail
    te = te_next

    qcko_tke = torch.where(pcnvflg[:, None], te, qcko[:, :, NTKE - 1])
    qcdo_tke = torch.where(scuflg[:, None], te, qcdo[:, :, NTKE - 1])
    qcko_tke, qcdo_tke = run_compile_island(
        "m_plume_recurrence",
        "update_tke_plumes",
        _update_tke_plume_recurrences,
        te,
        qcko_tke,
        qcdo_tke,
        xlamue,
        xlamde,
        zl,
        kpbl,
        mrad,
        krad,
        pcnvflg,
        scuflg,
    )
    qcko[:, :, NTKE - 1] = qcko_tke
    qcdo[:, :, NTKE - 1] = qcdo_tke

    return state.with_updates(
        {
            "prod": prod,
            "diss": diss,
            "te": te,
            "tke": tke,
            "qcko": qcko,
            "qcdo": qcdo,
            "tkmnz": tkmnz,
        }
    )


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
