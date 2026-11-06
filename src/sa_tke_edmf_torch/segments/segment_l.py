# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment L: non-canopy eddy diffusivity and SA-3D-TKE blending."""

from __future__ import annotations

import torch
from ufs_physics_common.precision import working_power

from sa_tke_edmf_torch.autodiff import safe_positive_denominator, safe_sqrt_nonnegative
from sa_tke_edmf_torch.compile import run_compile_island
from sa_tke_edmf_torch.constants import (
    CP,
    F32_0P1,
    F32_0P15,
    F32_0P16,
    F32_0P4,
    F32_0P55,
    F32_0P76,
    F32_ONE_THIRD,
    FV,
    GRAV,
    HFUS,
    HVAP,
    IM,
    KM,
    NTCW,
    NTIW,
    NTRAC,
    QLMIN,
    source_real,
)
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64
from sa_tke_edmf_torch.tracers import satmedmf_hydrometeor_indices

CHECKPOINT_FIELDS = ("dku", "dkt", "dkq", "dku_h", "dkq_h")


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    KM = state.require("q1").shape[1]
    NTRAC = state.require("q1").shape[2]
    q1 = state.require("q1")
    t1 = state.require("t1")
    prslk = state.require("prslk")
    psk = state.require("psk")
    zl = state.require("zl")
    zi = state.require("zi")
    theta = state.require("theta")
    thvs = state.require("thvs")
    bf = state.require("bf")
    shr2 = state.require("shr2")
    tke = state.require("tke")
    tkeh = state.require("tkeh")
    elm = state.require("elm")
    ckz = state.require("ckz")
    chz = state.require("chz")
    prn = state.require("prn")
    xkzo = state.require("xkzo")
    xkzmo = state.require("xkzmo")
    scuflg = state.require("scuflg") > 0.5
    mrad = state.require("mrad").to(torch.int64)
    krad = state.require("krad").to(torch.int64)
    kpbl = state.require("kpbl").to(torch.int64)
    hpbl = state.require("hpbl")
    pcnvflg = state.require("pcnvflg") > 0.5
    gotvx = state.require("gotvx")
    garea = state.require("garea")
    elmh = state.require("elmh")
    IM = q1.shape[0]
    require_float64("q1", q1, (IM, KM, NTRAC))
    require_float64("t1", t1, (IM, KM))
    require_float64("prslk", prslk, (IM, KM))
    require_float64("zl", zl, (IM, KM))
    require_float64("zi", zi, (IM, KM + 1))
    require_float64("theta", theta, (IM, KM))
    require_float64("bf", bf, (IM, KM - 1))
    require_float64("shr2", shr2, (IM, KM - 1))
    require_float64("tke", tke, (IM, KM))
    require_float64("tkeh", tkeh, (IM, KM))
    require_float64("elm", elm, (IM, KM))
    require_float64("ckz", ckz, (IM, KM))
    require_float64("chz", chz, (IM, KM))
    require_float64("prn", prn, (IM, KM - 1))
    require_float64("xkzo", xkzo, (IM, KM - 1))
    require_float64("xkzmo", xkzmo, (IM, KM - 1))

    tte_edmf = bool(state.config.get("tte_edmf", False))
    sa3dtke = bool(state.config.get("sa3dtke", False))
    tc_pbl = bool(state.config.get("tc_pbl", False))
    if tte_edmf:
        require_float64("tteh", state.require("tteh"), (IM, KM))
    dku, dkt, dkq, dku_h, dkq_h, tkeh, tesq, ele_les_state, pftke_state = run_compile_island(
        "l_diffusivity",
        "eddy_diffusivity",
        _diffusivity_block,
        q1,
        t1,
        prslk,
        psk,
        zl,
        zi,
        theta,
        thvs,
        bf,
        shr2,
        tke,
        tkeh,
        elm,
        ckz,
        chz,
        prn,
        xkzo,
        xkzmo,
        scuflg,
        mrad,
        krad,
        kpbl,
        hpbl,
        pcnvflg,
        gotvx,
        garea,
        elmh,
        state.require("phihs"),
        state.require("phims"),
        state.require("zol"),
        state.fields.get("tteh", tkeh),
        tte_edmf,
        sa3dtke,
        tc_pbl,
        *satmedmf_hydrometeor_indices(state.config, NTRAC)[:2],
        tuple(source_real(value, state.config) for value in
              (0.1, 0.15, 0.16, 0.4, 0.55, 0.76, 0.33333333, 1.0e-12, 0.67)),
    )
    return state.with_updates(
        {
            "dku": dku,
            "dkt": dkt,
            "dkq": dkq,
            "dku_h": dku_h,
            "dkq_h": dkq_h,
            "tkeh": tkeh,
            "tesq": tesq,
            "ele_les": ele_les_state,
            "pftke": pftke_state,
        }
    )


def _diffusivity_block(
    q1: torch.Tensor,
    t1: torch.Tensor,
    prslk: torch.Tensor,
    psk: torch.Tensor,
    zl: torch.Tensor,
    zi: torch.Tensor,
    theta: torch.Tensor,
    thvs: torch.Tensor,
    bf: torch.Tensor,
    shr2: torch.Tensor,
    tke: torch.Tensor,
    tkeh: torch.Tensor,
    elm: torch.Tensor,
    ckz: torch.Tensor,
    chz: torch.Tensor,
    prn: torch.Tensor,
    xkzo: torch.Tensor,
    xkzmo: torch.Tensor,
    scuflg: torch.Tensor,
    mrad: torch.Tensor,
    krad: torch.Tensor,
    kpbl: torch.Tensor,
    hpbl: torch.Tensor,
    pcnvflg: torch.Tensor,
    gotvx: torch.Tensor,
    garea: torch.Tensor,
    elmh: torch.Tensor,
    phihs: torch.Tensor,
    phims: torch.Tensor,
    zol: torch.Tensor,
    tteh: torch.Tensor,
    tte_edmf: bool,
    sa3dtke: bool,
    tc_pbl: bool,
    ntcw: int = NTCW,
    ntiw: int = NTIW,
    coefficients: tuple[float, ...] = (
        F32_0P1, F32_0P15, F32_0P16, F32_0P4, F32_0P55, F32_0P76,
        F32_ONE_THIRD, QLMIN, 0.6700000166893005,
    ),
) -> tuple[torch.Tensor, ...]:
    (F32_0P1, F32_0P15, F32_0P16, F32_0P4, F32_0P55, F32_0P76,
     F32_ONE_THIRD, QLMIN, prscu_default) = coefficients
    KM = q1.shape[1]
    dtype = q1.dtype
    device = q1.device
    ncol = q1.shape[0]
    km1 = KM - 1
    kmpbl = KM // 2
    zero_2 = torch.zeros((ncol, KM), dtype=dtype, device=device)
    zero_i = torch.zeros((ncol, km1), dtype=dtype, device=device)

    ck0 = F32_0P55 if tc_pbl else F32_0P4
    ck1 = F32_0P16 if tte_edmf else F32_0P15
    ch1 = F32_0P16 if tte_edmf else F32_0P15
    prmax = 6.0 if tte_edmf else 4.0
    prscu = F32_0P4 if tte_edmf else prscu_default

    if tte_edmf:
        ris = torch.maximum(zol * phihs / (phims * phims), torch.full((ncol,), -100.0, dtype=dtype, device=device))
        ri_tte = torch.maximum(bf / shr2, torch.full_like(bf, -100.0))
        below_surface_fraction = zi[:, 1:KM] <= (F32_0P1 * hpbl)[:, None]
        ri_tte = torch.where(below_surface_fraction, ris[:, None], ri_tte)
        # Each branch sees only its own sign of ri_tte (finite backward at the unselected pole).
        ri_neg, ri_pos = torch.where(ri_tte < 0.0, ri_tte, -1.0), torch.where(ri_tte < 0.0, 1.0, ri_tte)
        epotte = torch.where(ri_tte < 0.0, ri_neg / (2.0 * ri_neg - 1.0), ri_pos / (1.0 + 3.0 * ri_pos))
        tkeh_int = tteh[:, :km1] * (1.0 - epotte)
        tesq = tkeh_int / safe_positive_denominator(safe_sqrt_nonnegative(tteh[:, :km1]))
        tkeh_next = tkeh.clone()
        tkeh_next[:, :km1] = tkeh_int
        tkeh = tkeh_next
        tesq = torch.nan_to_num(tesq, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        tesq = safe_sqrt_nonnegative(tkeh[:, :km1])

    tem = 0.5 * (elm[:, :km1] + elm[:, 1:]) * tesq
    ri = torch.maximum(bf / shr2, torch.full_like(bf, -100.0))
    inside_pbl = torch.arange(1, KM, dtype=torch.int64, device=device).unsqueeze(0) < kpbl[:, None]
    unstable = ri < 0.0

    dku = zero_2.clone()
    dkt = zero_2.clone()
    dkq = zero_i.clone()
    convective = inside_pbl & pcnvflg[:, None]
    local_unstable = inside_pbl & (~pcnvflg[:, None]) & unstable
    local_stable = inside_pbl & (~pcnvflg[:, None]) & (~unstable)
    free_unstable = (~inside_pbl) & unstable
    free_stable = (~inside_pbl) & (~unstable)

    dku_i = torch.zeros_like(tem)
    dkt_i = torch.zeros_like(tem)
    dku_i = torch.where(convective | local_unstable, ckz[:, :km1] * tem, dku_i)
    dkt_i = torch.where(convective | local_unstable, dku_i / prn, dkt_i)
    dkt_i = torch.where(local_stable, chz[:, :km1] * tem, dkt_i)
    dku_i = torch.where(local_stable, dkt_i * prn, dku_i)
    dku_i = torch.where(free_unstable, ck1 * tem, dku_i)
    dkt_i = torch.where(free_unstable, 1.5 * dku_i, dkt_i)
    prnum = torch.minimum(1.0 + 2.1 * ri, torch.full_like(ri, prmax))
    dkt_i = torch.where(free_stable, ch1 * tem, dkt_i)
    dku_i = torch.where(free_stable, dkt_i * prnum, dku_i)

    level = torch.arange(1, KM, dtype=torch.int64, device=device).unsqueeze(0)
    scu_band = scuflg[:, None] & (level >= mrad[:, None]) & (level < krad[:, None])
    tem1 = ck0 * tem if tte_edmf else ckz[:, :km1] * tem
    ptem1 = tem1 / prscu
    dku_i = torch.where(scu_band, torch.maximum(dku_i, tem1), dku_i)
    dkt_i = torch.where(scu_band, torch.maximum(dkt_i, ptem1), dkt_i)

    dkq = torch.maximum(torch.minimum(dkt_i, torch.full_like(dkt_i, 1000.0)), xkzo)
    dkt_i = torch.maximum(torch.minimum(dkt_i, torch.full_like(dkt_i, 1000.0)), xkzo)
    dku_i = torch.maximum(torch.minimum(dku_i, torch.full_like(dku_i, 1000.0)), xkzmo)
    dku[:, :km1] = dku_i
    dkt[:, :km1] = dkt_i

    dku_h = zero_2.clone()
    dkq_h = zero_2.clone()
    ele_les_state = zero_2.clone()
    pftke_state = torch.zeros((ncol,), dtype=dtype, device=device)
    if sa3dtke:
        qliq = torch.clamp_min(q1[:, :, ntcw - 1], QLMIN)
        qice_source = q1[:, :, ntiw - 1] + q1[:, :, 4] + q1[:, :, 5]
        qice = torch.clamp_min(qice_source, QLMIN)
        pix = psk[:, None] / prslk
        thetal = theta - (HVAP + HFUS) / CP * pix * (qliq + qice)

        dz_interface = zl[:, 1:] - zl[:, :-1]
        stable_les = gotvx[:, :km1] * (thetal[:, 1:] - thetal[:, :-1]) / dz_interface
        tem1_les = working_power(garea[:, None] * dz_interface, F32_ONE_THIRD)
        elm_les = torch.where(
            stable_les > 0.0,
            torch.minimum(
                F32_0P76
                * safe_sqrt_nonnegative(tke[:, :km1])
                / safe_positive_denominator(safe_sqrt_nonnegative(stable_les)),
                tem1_les,
            ),
            tem1_les,
        )
        dku_les = torch.minimum(F32_0P1 * elm_les * safe_sqrt_nonnegative(tkeh[:, :km1]), torch.full_like(elm_les, 300.0))
        dkt_les = torch.minimum((1.0 + 2.0 * elm_les / tem1_les) * dku_les, torch.full_like(elm_les, 300.0))
        dkq_les = dkt_les

        tke_pbl = tke[:, :kmpbl]
        tkemax, ktkemax0 = torch.max(tke_pbl, dim=1)
        ktkemax = ktkemax0 + 1
        scl = torch.zeros((ncol,), dtype=dtype, device=device)
        flg = _take_level(zl, ktkemax) <= 2500.0
        scl = torch.where(flg, scl, torch.full_like(scl, 500.0))
        for k in range(kmpbl):
            active = flg & (k + 1 > ktkemax)
            scl = torch.where(active, zl[:, k], scl)
            flg = torch.where(active & (tke[:, k] < 0.5 * tkemax), torch.zeros_like(flg), flg)
        scl = torch.maximum(torch.minimum(torch.maximum(scl, torch.full_like(scl, 500.0)), torch.full_like(scl, 2500.0)), hpbl)
        pfdx = torch.sqrt(garea) / scl
        pfdx2 = pfdx * pfdx
        pfdx_sqrt = torch.sqrt(pfdx)
        pfl = 0.280 * (pfdx2 + 0.870 * pfdx_sqrt - 0.913) / (pfdx2 + 0.153 * pfdx_sqrt + 0.278) + 0.720
        pfl = torch.clamp(pfl, min=0.0, max=1.0)
        pfdx_two_thirds = working_power(pfdx, 0.6666666865348816)
        pftke_state = (pfdx2 + 0.07000000029802322 * pfdx_two_thirds) / (
            pfdx2 + 0.1420000046491623 * pfdx_two_thirds + 0.07100000232458115
        )
        pftke_state = torch.clamp(pftke_state, min=0.0, max=1.0)

        dku_body = (1.0 - pfl[:, None]) * dku_les + pfl[:, None] * dku[:, :km1]
        dkt_body = (1.0 - pfl[:, None]) * dkt_les + pfl[:, None] * dkt[:, :km1]
        dku_next = dku.clone()
        dku_next[:, :km1] = dku_body
        dku = dku_next
        dkt_next = dkt.clone()
        dkt_next[:, :km1] = dkt_body
        dkt = dkt_next
        dkq = (1.0 - pfl[:, None]) * dkq_les + pfl[:, None] * dkq

        dku_h = F32_0P4 * elmh * safe_sqrt_nonnegative(tke)
        dkq_h = dku_h.clone()

        dz_first = zl[:, 1]
        stable_first = gotvx[:, 0] * (thetal[:, 1] - thvs) / safe_positive_denominator(dz_first)
        dz_i = zi[:, 1:KM] - zi[:, :km1]
        if km1 > 1:
            dz_rest = zl[:, 2:] - zl[:, :-2]
            stable_rest = gotvx[:, 1:km1] * (thetal[:, 2:] - thetal[:, : km1 - 1]) / safe_positive_denominator(dz_rest)
            dz_layer = torch.empty_like(dz_i)
            dz_layer[:, 0] = dz_first
            dz_layer[:, 1:] = dz_rest
            stable_layer = torch.empty_like(dz_i)
            stable_layer[:, 0] = stable_first
            stable_layer[:, 1:] = stable_rest
        else:
            dz_layer = dz_first[:, None]
            stable_layer = stable_first[:, None]
        tem1_layer = working_power(garea[:, None] * dz_i, F32_ONE_THIRD)
        ele_les = torch.where(
            stable_layer > 0.0,
            torch.minimum(
                F32_0P76
                * safe_sqrt_nonnegative(tke[:, :km1])
                / safe_positive_denominator(safe_sqrt_nonnegative(stable_layer)),
                tem1_layer,
            ),
            tem1_layer,
        )
        ele_les_next = ele_les_state.clone()
        ele_les_next[:, :km1] = ele_les
        ele_les_state = ele_les_next
        dku_les_layer = torch.minimum(F32_0P1 * ele_les * safe_sqrt_nonnegative(tke[:, :km1]), torch.full_like(ele_les, 300.0))
        dkq_les_layer = torch.minimum((1.0 + 2.0 * ele_les / tem1_layer) * dku_les_layer, torch.full_like(ele_les, 300.0))
        dku_h_body = (1.0 - pfl[:, None]) * dku_les_layer + pfl[:, None] * dku_h[:, :km1]
        dkq_h_body = (1.0 - pfl[:, None]) * dkq_les_layer + pfl[:, None] * dkq_h[:, :km1]
        dku_h_next = torch.empty((ncol, KM), dtype=dtype, device=device)
        dku_h_next[:, :km1] = dku_h_body
        dku_h_next[:, KM - 1] = dku_h_body[:, -1]
        dku_h = dku_h_next
        dkq_h_next = torch.empty((ncol, KM), dtype=dtype, device=device)
        dkq_h_next[:, :km1] = dkq_h_body
        dkq_h_next[:, KM - 1] = dkq_h_body[:, -1]
        dkq_h = dkq_h_next

    return dku, dkt, dkq, dku_h, dkq_h, tkeh, tesq, ele_les_state, pftke_state


def _take_level(values: torch.Tensor, one_based_index: torch.Tensor) -> torch.Tensor:
    index = torch.clamp(one_based_index.to(torch.int64) - 1, min=0, max=values.shape[1] - 1)
    return values.gather(1, index[:, None]).squeeze(1)


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
