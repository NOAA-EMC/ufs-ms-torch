# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment P: ASQEC downdraft recompute and advective factor."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch import recurrence_chunks as chunks
from sa_sas_deep_convection_torch.compile import run_compile_island
from sa_sas_deep_convection_torch.constants import KM
from sa_sas_deep_convection_torch.ops import sqrt_zero_grad
from sa_sas_deep_convection_torch.segments.segment_d import _gather_column
from ufs_physics_common.ordered_sum import masked_sum
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real

CHECKPOINT_FIELDS = ("advfac", "dtconv", "xpwav", "xpwev", "mbdt", "edtx", "asqecflg")


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 2777-2986 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    asqecflg = state.require("asqecflg")
    cnvflg = state.require("cnvflg")
    jmin = state.require("jmin")
    kd94 = state.require("kd94")
    kbcon1 = state.require("kbcon1")
    ktcon1 = state.require("ktcon1")
    zi = state.require("zi")
    zo = state.require("zo")
    heo = state.require("heo")
    heso = state.require("heso")
    qo = state.require("qo")
    qeso = state.require("qeso")
    to = state.require("to")
    etad = state.require("etad")
    xlamdet = state.require("xlamdet")
    xlamddt = state.require("xlamddt")
    xlamd = state.require("xlamd")
    hcdo = state.require("hcdo").clone()
    qcdo = state.require("qcdo").clone()
    qrcd = state.require("qrcd").clone()
    xpwev = state.require("xpwev").clone()
    edtx = state.require("edtx").clone()
    xaa0 = state.require("xaa0").clone()
    dtconv = state.require("dtconv").clone()
    advfac = state.require("advfac").clone()
    u1 = state.require("u1")
    v1 = state.require("v1")
    gdx = state.require("gdx")
    wc = state.require("wc")
    ncol = cnvflg.shape[0]
    for name, tensor in {
        "asqecflg": asqecflg,
        "cnvflg": cnvflg,
        "jmin": jmin,
        "kd94": kd94,
        "kbcon1": kbcon1,
        "ktcon1": ktcon1,
        "xlamdet": xlamdet,
        "xlamddt": xlamddt,
        "xlamd": xlamd,
        "xpwev": xpwev,
        "edtx": edtx,
        "dtconv": dtconv,
        "advfac": advfac,
        "gdx": gdx,
        "wc": wc,
    }.items():
        require_real(name, tensor, (ncol,))
    for name, tensor in {
        "zi": zi,
        "zo": zo,
        "heo": heo,
        "heso": heso,
        "qo": qo,
        "qeso": qeso,
        "to": to,
        "etad": etad,
        "hcdo": hcdo,
        "qcdo": qcdo,
        "qrcd": qrcd,
        "u1": u1,
        "v1": v1,
    }.items():
        require_real(name, tensor, (ncol, KM))

    hcdo, qcdo, qrcd, xpwev, edtx, xaa0, dtconv, sumx, umean, advfac = run_compile_island(
        "p_asqec_downdraft",
        "asqec_downdraft_advective",
        _asqec_downdraft_advective_block,
        asqecflg,
        cnvflg,
        jmin,
        kd94,
        kbcon1,
        ktcon1,
        zi,
        zo,
        heo,
        heso,
        qo,
        qeso,
        to,
        etad,
        xlamdet,
        xlamddt,
        xlamd,
        hcdo,
        qcdo,
        qrcd,
        xpwev,
        edtx,
        xaa0,
        dtconv,
        advfac,
        u1,
        v1,
        gdx,
        wc,
        state.require("xpwav"),
        state.require("el2orc"),
        state.require("hvap"),
        state.require("grav"),
        state.require("cp"),
        state.require("fv"),
        state.require("delt"),
        state.require("cat_adj_deep"),
        bool(state.config.get("hwrf_samfdeep", False)),
    )

    return state.with_updates(
        {
            "hcdo": hcdo,
            "qcdo": qcdo,
            "qrcd": qrcd,
            "xpwev": xpwev,
            "edtx": edtx,
            "xaa0": xaa0,
            "dtconv": dtconv,
            "sumx": sumx,
            "umean": umean,
            "advfac": advfac,
        }
    )


def _asqec_downdraft_advective_block(
    asqecflg: torch.Tensor,
    cnvflg: torch.Tensor,
    jmin: torch.Tensor,
    kd94: torch.Tensor,
    kbcon1: torch.Tensor,
    ktcon1: torch.Tensor,
    zi: torch.Tensor,
    zo: torch.Tensor,
    heo: torch.Tensor,
    heso: torch.Tensor,
    qo: torch.Tensor,
    qeso: torch.Tensor,
    to: torch.Tensor,
    etad: torch.Tensor,
    xlamdet: torch.Tensor,
    xlamddt: torch.Tensor,
    xlamd: torch.Tensor,
    hcdo: torch.Tensor,
    qcdo: torch.Tensor,
    qrcd: torch.Tensor,
    xpwev: torch.Tensor,
    edtx: torch.Tensor,
    xaa0: torch.Tensor,
    dtconv: torch.Tensor,
    advfac: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    gdx: torch.Tensor,
    wc: torch.Tensor,
    xpwav: torch.Tensor,
    el2orc: torch.Tensor,
    hvap: torch.Tensor,
    grav: torch.Tensor,
    cp: torch.Tensor,
    fv: torch.Tensor,
    delt: torch.Tensor,
    cat_adj_deep: torch.Tensor,
    hwrf_samfdeep: bool,
) -> tuple[torch.Tensor, ...]:
    KM = zo.shape[1]
    asqec = asqecflg != 0.0
    xpwev = torch.where(asqec, torch.zeros_like(xpwev), xpwev)
    # Each recurrence value keeps its own autograd lifetime. Reading a view
    # of the subsequently mutated matrix breaks derivatives of geometry inputs.
    # Carry-independent level terms are evaluated for all levels at once; only
    # the hcdo recurrence and the ordered sums loop over levels.
    level_values = torch.arange(1, KM + 1, dtype=jmin.dtype, device=jmin.device)
    start = asqec[:, None] & (jmin[:, None] == level_values)
    hcdo = torch.where(start, heo, hcdo)
    qcdo = torch.where(start, qo, qcdo)
    qrcd = torch.where(start, qo, qrcd)

    head = slice(0, KM - 1)
    lower = level_values[head]
    layer = asqec[:, None] & (lower < jmin[:, None])
    layer_rows = layer.t().contiguous()
    dz = zi[:, 1:KM] - zi[:, head]
    tem = xlamdet[:, None] * dz
    tem1 = torch.where(lower >= kd94[:, None], 0.5 * xlamddt[:, None] * dz, 0.5 * (xlamd + xlamddt)[:, None] * dz)
    keep = (1.0 - tem1).t().contiguous()
    source = (tem * 0.5 * (heo[:, head] + heo[:, 1:KM])).t().contiguous()
    denominator = torch.where(layer, 1.0 + tem - tem1, 1.0).t().contiguous()
    carry, cells = hcdo[:, KM - 1], []
    for index in range(KM - 2, -1, -1):
        candidate = (keep[index] * carry + source[index]) / denominator[index]
        carry = torch.where(layer_rows[index], candidate, hcdo[:, index])
        cells.append(carry)
    hcdo = chunks.place_cells(hcdo, cells[::-1], 0)

    # qrcd and qcdo read the finished hcdo, never their own updates: no recurrence.
    temperature = torch.where(layer, to[:, head], 1.0)
    gamma = el2orc * qeso[:, head] / (temperature * temperature)
    qrcd_candidate = qeso[:, head] + (1.0 / hvap) * (gamma / (1.0 + gamma)) * (hcdo[:, head] - heso[:, head])
    qrcd_head = torch.where(layer, qrcd_candidate, qrcd[:, head])
    qrcd = chunks.place_levels(qrcd, qrcd_head, 0)
    tem = 0.5 * xlamdet[:, None] * dz
    qcdo_candidate = ((1.0 - tem) * qrcd[:, 1:KM] + tem * (qo[:, head] + qo[:, 1:KM])) / (1.0 + tem)
    qcdo_head = torch.where(layer, qcdo_candidate, qcdo[:, head])
    qcdo = chunks.place_levels(qcdo, qcdo_head, 0)
    xpwd = (etad[:, head] * (qcdo_head - qrcd_head)).t().contiguous()
    xpwev = masked_sum(xpwev, layer_rows, xpwd, descending=True)

    edtmax = torch.full_like(edtx, 0.3)
    edtx_candidate = torch.where(xpwev >= 0.0, torch.zeros_like(edtx), torch.minimum(-edtx * xpwav / torch.where(asqec & (xpwev < 0.0), xpwev, torch.ones_like(xpwev)), edtmax))
    edtx = torch.where(asqec, edtx_candidate, edtx)

    dz_work = zo[:, head] - zo[:, 1:KM]
    heat = edtx[:, None] * dz_work * (grav / (cp * temperature)) * ((hcdo[:, head] - heso[:, head]) / (1.0 + gamma)) * (
        1.0 + fv * cp * gamma * to[:, head] / hvap
    )
    moisture = edtx[:, None] * dz_work * grav * fv * torch.maximum(torch.zeros_like(cnvflg)[:, None], qeso[:, head] - qo[:, head])
    heat, moisture = heat.t().contiguous(), moisture.t().contiguous()
    xaa0 = masked_sum(xaa0, layer_rows, heat, moisture, descending=True)

    depth = _gather_column(zi, ktcon1) - _gather_column(zi, kbcon1)
    dtconv_candidate = depth / torch.where(cnvflg != 0.0, wc, torch.ones_like(wc))
    if not hwrf_samfdeep:
        dtconv_candidate = (1.0 + gdx / 75000.0) * dtconv_candidate
    dtmin = torch.maximum(delt, torch.full_like(delt, 600.0))
    dtmax = torch.maximum(delt, torch.full_like(delt, 10800.0))
    dtconv_candidate = torch.minimum(torch.maximum(dtconv_candidate, dtmin), dtmax)
    dtconv = torch.where(cnvflg != 0.0, dtconv_candidate, dtconv)

    sumx = torch.zeros_like(cnvflg)
    umean = torch.zeros_like(cnvflg)
    middle = level_values[1:KM - 1]
    wind_layer = ((cnvflg != 0.0)[:, None] & (middle >= kbcon1[:, None]) & (middle < ktcon1[:, None])).t().contiguous()
    dz_wind = zi[:, 1:KM - 1] - zi[:, 0:KM - 2]
    speed = sqrt_zero_grad(u1[:, 1:KM - 1] * u1[:, 1:KM - 1] + v1[:, 1:KM - 1] * v1[:, 1:KM - 1])
    wind_term = (speed * dz_wind).t().contiguous()
    dz_wind = dz_wind.t().contiguous()
    for index in range(KM - 2):
        umean = torch.where(wind_layer[index], umean + wind_term[index], umean)
        sumx = torch.where(wind_layer[index], sumx + dz_wind[index], sumx)
    umean = torch.where(cnvflg != 0.0, torch.maximum(umean / torch.where(cnvflg != 0.0, sumx, torch.ones_like(sumx)), torch.ones_like(umean)), umean)
    tauadv = gdx / torch.where(cnvflg != 0.0, umean, torch.ones_like(umean))
    advfac = torch.where(cnvflg != 0.0, torch.minimum(cat_adj_deep * tauadv / torch.where(cnvflg != 0.0, dtconv, 1.0), torch.ones_like(advfac)), advfac)
    return hcdo, qcdo, qrcd, xpwev, edtx, xaa0, dtconv, sumx, umean, advfac


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
