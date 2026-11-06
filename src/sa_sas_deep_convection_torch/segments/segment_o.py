# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment O: ASQEC environment and updraft recompute."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch import recurrence_chunks as chunks
from sa_sas_deep_convection_torch.compile import run_compile_island
from sa_sas_deep_convection_torch.constants import EPS, EPSM1, FV, GRAV, KM
from sa_sas_deep_convection_torch.segments.segment_d import _fpvs_from_table
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real
from sa_sas_deep_convection_torch.thermo import _fpvs_table_for_device

CHECKPOINT_FIELDS = ("qeso", "heo", "heso", "dbyo", "hcko", "qcko", "eta", "asqecflg")


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 2573-2776 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    cnvflg = state.require("cnvflg")
    kmax = state.require("kmax")
    kb = state.require("kb")
    kbcon = state.require("kbcon")
    jmin = state.require("jmin")
    ktcon = state.require("ktcon")
    ktcon1 = state.require("ktcon1")
    gdx = state.require("gdx")
    q1 = state.require("q1")
    t1 = state.require("t1")
    dellaq = state.require("dellaq")
    dellah = state.require("dellah")
    mbdt = state.require("mbdt")
    pfld = state.require("pfld")
    zo = state.require("zo")
    zi = state.require("zi")
    c0t = state.require("c0t")
    xlamue = state.require("xlamue")
    xlamud = state.require("xlamud")
    eta = state.require("eta")
    qeso = state.require("qeso").clone()
    qo = state.require("qo").clone()
    to = state.require("to").clone()
    po = state.require("po").clone()
    heo = state.require("heo").clone()
    heso = state.require("heso").clone()
    hcko = state.require("hcko").clone()
    qcko = state.require("qcko").clone()
    dbyo = state.require("dbyo").clone()
    xaa0 = state.require("xaa0").clone()
    xpwav = state.require("xpwav").clone()
    ncol = cnvflg.shape[0]
    for name, tensor in {
        "cnvflg": cnvflg,
        "kmax": kmax,
        "kb": kb,
        "kbcon": kbcon,
        "jmin": jmin,
        "ktcon": ktcon,
        "ktcon1": ktcon1,
        "gdx": gdx,
        "mbdt": mbdt,
    }.items():
        require_real(name, tensor, (ncol,))
    for name, tensor in {
        "q1": q1,
        "t1": t1,
        "dellaq": dellaq,
        "dellah": dellah,
        "pfld": pfld,
        "zo": zo,
        "zi": zi,
        "c0t": c0t,
        "xlamue": xlamue,
        "xlamud": xlamud,
        "eta": eta,
        "qeso": qeso,
        "qo": qo,
        "to": to,
        "po": po,
        "heo": heo,
        "heso": heso,
        "hcko": hcko,
        "qcko": qcko,
        "dbyo": dbyo,
    }.items():
        require_real(name, tensor, (ncol, KM))

    dxcrtuf, asqecflg, qo, to, po, qeso, heo, heso, hcko, qcko, dbyo, xaa0, xpwav = run_compile_island(
        "o_asqec_updraft",
        "asqec_environment_updraft",
        _asqec_environment_updraft_block,
        cnvflg,
        kmax,
        kb,
        kbcon,
        jmin,
        ktcon,
        ktcon1,
        gdx,
        q1,
        t1,
        dellaq,
        dellah,
        mbdt,
        pfld,
        zo,
        zi,
        c0t,
        xlamue,
        xlamud,
        eta,
        qeso,
        qo,
        to,
        po,
        heo,
        heso,
        hcko,
        qcko,
        dbyo,
        xaa0,
        xpwav,
        _fpvs_table_for_device(str(cnvflg.device), cnvflg.dtype),
        state.require("hvap"),
        state.require("cp"),
        state.require("fact1"),
        state.require("fact2"),
        state.require("el2orc"),
        state.require("c1"),
        bool(state.config.get("progsigma", False)),
        int(state.config.get("ncloud", 0)),
    )
    return state.with_updates(
        {
            "dxcrtuf": dxcrtuf,
            "asqecflg": asqecflg,
            "qo": qo,
            "to": to,
            "po": po,
            "qeso": qeso,
            "heo": heo,
            "heso": heso,
            "hcko": hcko,
            "qcko": qcko,
            "dbyo": dbyo,
            "xaa0": xaa0,
            "xpwav": xpwav,
        }
    )


def _asqec_environment_updraft_block(
    cnvflg: torch.Tensor,
    kmax: torch.Tensor,
    kb: torch.Tensor,
    kbcon: torch.Tensor,
    jmin: torch.Tensor,
    ktcon: torch.Tensor,
    ktcon1: torch.Tensor,
    gdx: torch.Tensor,
    q1: torch.Tensor,
    t1: torch.Tensor,
    dellaq: torch.Tensor,
    dellah: torch.Tensor,
    mbdt: torch.Tensor,
    pfld: torch.Tensor,
    zo: torch.Tensor,
    zi: torch.Tensor,
    c0t: torch.Tensor,
    xlamue: torch.Tensor,
    xlamud: torch.Tensor,
    eta: torch.Tensor,
    qeso: torch.Tensor,
    qo: torch.Tensor,
    to: torch.Tensor,
    po: torch.Tensor,
    heo: torch.Tensor,
    heso: torch.Tensor,
    hcko: torch.Tensor,
    qcko: torch.Tensor,
    dbyo: torch.Tensor,
    xaa0: torch.Tensor,
    xpwav: torch.Tensor,
    fpvs_table: torch.Tensor,
    hvap: torch.Tensor,
    cp: torch.Tensor,
    fact1: torch.Tensor,
    fact2: torch.Tensor,
    el2orc: torch.Tensor,
    c1: torch.Tensor,
    progsigma: bool,
    ncloud: int,
    *,
    heat_microblock: int = 0,
) -> tuple[torch.Tensor, ...]:
    KM = q1.shape[1]
    dxcrtas = 500.0e3 if progsigma else 8.0e3
    dxcrtuf_value = 10.0e3 if progsigma else 15.0e3
    dxcrtuf = torch.full_like(cnvflg, dxcrtuf_value)
    asqecflg = torch.where((cnvflg != 0.0) & (gdx >= dxcrtas), torch.ones_like(cnvflg), torch.zeros_like(cnvflg))
    asqec = asqecflg != 0.0

    level_values = torch.arange(1, KM + 1, dtype=cnvflg.dtype, device=cnvflg.device).reshape(1, KM)
    active_levels = level_values <= kmax[:, None]
    layer = asqec[:, None] & active_levels
    above_top = layer & (level_values > ktcon[:, None])
    in_cloud = layer & (level_values <= ktcon[:, None])
    qo = torch.where(above_top, q1, qo)
    to = torch.where(above_top, t1, to)
    qo_candidate = torch.maximum(dellaq * mbdt[:, None] + q1, torch.full_like(qo, 1.0e-10))
    dellat = (dellah - hvap * dellaq) / cp
    to_candidate = dellat * mbdt[:, None] + t1
    qo = torch.where(in_cloud, qo_candidate, qo)
    to = torch.where(in_cloud, to_candidate, to)

    mask = asqec[:, None] & active_levels
    qeso_candidate = 0.01 * _fpvs_from_table(to, fpvs_table)
    qeso_candidate = EPS * qeso_candidate / (pfld + EPSM1 * qeso_candidate)
    qeso = torch.where(mask, torch.clamp_min(qeso_candidate, 1.0e-8), qeso)

    dz = 0.5 * (zo[:, 1:KM] - zo[:, : KM - 1])
    dp = 0.5 * (pfld[:, 1:KM] - pfld[:, : KM - 1])
    es = 0.01 * _fpvs_from_table(to[:, 1:KM], fpvs_table)
    pprime = pfld[:, 1:KM] + EPSM1 * es
    qs = EPS * es / pprime
    dqsdp = -qs / pprime
    interface_active = asqec[:, None] & (level_values[:, :KM - 1] <= (kmax[:, None] - 1.0))
    temperature = torch.where(interface_active, to[:, 1:KM], 1.0)
    desdt = es * (fact1 / temperature + fact2 / (temperature * temperature))
    dqsdt = qs * pfld[:, 1:KM] * desdt / (es * pprime)
    gamma = el2orc * qeso[:, 1:KM] / (temperature * temperature)
    dt = (GRAV * dz + hvap * dqsdp * dp) / (cp * (1.0 + gamma))
    dq = dqsdt * dt + dqsdp * dp
    level_index = torch.arange(KM, dtype=torch.long, device=cnvflg.device)
    lower_index = torch.clamp(level_index, max=KM - 2)
    next_index = torch.clamp(level_index + 1, max=KM - 1)
    interface_mask_full = asqec[:, None] & (level_values < float(KM)) & (level_values <= (kmax[:, None] - 1.0))
    to = torch.where(interface_mask_full, to[:, next_index] + dt[:, lower_index], to)
    qo = torch.where(interface_mask_full, qo[:, next_index] + dq[:, lower_index], qo)
    po = torch.where(interface_mask_full, 0.5 * (pfld[:, lower_index] + pfld[:, next_index]), po)

    qeso_interface = 0.01 * _fpvs_from_table(to, fpvs_table)
    qeso_interface = EPS * qeso_interface / (po + EPSM1 * qeso_interface)
    qeso = torch.where(interface_mask_full, torch.clamp_min(qeso_interface, 1.0e-8), qeso)
    qo = torch.where(interface_mask_full, torch.clamp_min(qo, 1.0e-10), qo)
    heo_interface = 0.5 * GRAV * (zo + zo[:, next_index]) + cp * to + hvap * qo
    heso_interface = 0.5 * GRAV * (zo + zo[:, next_index]) + cp * to + hvap * qeso
    heo = torch.where(interface_mask_full, heo_interface, heo)
    heso = torch.where(interface_mask_full, heso_interface, heso)
    kmax_layer = asqec[:, None] & (kmax[:, None] == level_values)
    heo_kmax = GRAV * zo + cp * to + hvap * qo
    heso_kmax = GRAV * zo + cp * to + hvap * qeso
    heo = torch.where(kmax_layer, heo_kmax, heo)
    heso = torch.where(kmax_layer, heso_kmax, heso)

    xaa0 = torch.where(asqec, torch.zeros_like(xaa0), xaa0)
    xpwav = torch.where(asqec, torch.zeros_like(xpwav), xpwav)
    kb_layer = asqec[:, None] & (kb[:, None] == level_values)
    hcko = torch.where(kb_layer, heo, hcko)
    qcko = torch.where(kb_layer, qo, qcko)

    # Carry-independent level terms for all levels at once; only the hcko and
    # qcko recurrences and the ordered sums loop over levels.
    cur, prev = slice(1, KM - 1), slice(0, KM - 2)
    lv = level_values[:, cur]
    rows = lambda x: x.t().contiguous()  # noqa: E731
    dz = zi[:, cur] - zi[:, prev]
    tem = 0.5 * (xlamue[:, cur] + xlamue[:, prev]) * dz
    tem1 = 0.25 * (xlamud[:, cur] + xlamud[:, prev]) * dz
    hlayer = asqec[:, None] & (lv > kb[:, None]) & (lv <= ktcon[:, None])
    hkeep, hsource = rows(1.0 - tem1), rows(tem * 0.5 * (heo[:, cur] + heo[:, prev]))
    hden, hlayer_rows = rows(torch.where(hlayer, 1.0 + tem - tem1, 1.0)), rows(hlayer)
    if heat_microblock:
        # Keep the heat anomaly through the buoyancy consumer, not only the scan.
        anchor = heo[:, :1]
        source = rows(tem * 0.5 * ((heo[:, cur] - anchor) + (heo[:, prev] - anchor)))
        anomaly = chunks._blocked_affine_recurrence(
            hkeep, (source,), hden, hlayer_rows, rows(hcko[:, cur] - anchor),
            hcko[:, 0] - anchor[:, 0], microblock=heat_microblock).movedim(0, 1)
        updated = torch.where(hlayer, anomaly + anchor, hcko[:, cur])
        hcko = chunks.place_levels(hcko, updated, 1)
        xdby = torch.where(hlayer, anomaly + (anchor - heso[:, cur]), hcko[:, cur] - heso[:, cur])
    else:
        carry, cells = hcko[:, 0], []
        for index in range(KM - 2):
            candidate = (hkeep[index] * carry + hsource[index]) / hden[index]
            carry = torch.where(hlayer_rows[index], candidate, hcko[:, index + 1])
            cells.append(carry)
        hcko = chunks.place_cells(hcko, cells, 1)
        xdby = hcko[:, cur] - heso[:, cur]

    in_updraft = asqec[:, None] & (lv > kb[:, None]) & (lv < ktcon[:, None])
    buoy = asqec[:, None] & (lv >= kbcon[:, None]) & (lv < ktcon1[:, None])
    temperature = torch.where(in_updraft | buoy, to[:, cur], 1.0)
    gamma = el2orc * qeso[:, cur] / (temperature * temperature)
    xqrch = qeso[:, cur] + gamma * xdby / (hvap * (1.0 + gamma))
    tem = 0.25 * (xlamue[:, cur] + xlamue[:, prev]) * dz
    eta_cur = eta[:, cur]
    etah = 0.5 * (eta_cur + eta[:, prev])
    dz1 = zo[:, 2:KM] - zo[:, cur]
    # MPS forward AD needs a tensor literal when cp is a live scalar dual.
    fv = torch.as_tensor(FV, dtype=cp.dtype, device=cp.device)
    rfact = 1.0 + fv * cp * gamma * to[:, cur] / hvap
    heat = dz1 * torch.div(GRAV, cp * temperature) * xdby / (1.0 + gamma) * rfact
    moisture = dz1 * GRAV * FV * torch.maximum(torch.zeros_like(cnvflg)[:, None], qeso[:, cur] - qo[:, cur])
    t = {name: rows(value) for name, value in dict(
        keep=1.0 - tem, source=tem * (qo[:, cur] + qo[:, prev]), den=torch.where(in_updraft, 1.0 + tem, 1.0),
        in_updraft=in_updraft, buoy=buoy, xqrch=xqrch, eta=eta_cur, precip_window=in_updraft & (lv >= kbcon[:, None]),
        cloud_den=eta_cur + etah * (c0t[:, cur] + c1) * dz, plain_den=eta_cur + etah * c0t[:, cur] * dz,
        cloud_level=(ncloud > 0) & (lv > jmin[:, None]), below_top=lv < ktcon1[:, None], dz_grav=dz * GRAV,
        rain_factor=etah * c0t[:, cur] * dz, heat=heat, moisture=moisture).items()}
    carry, cells = qcko[:, 0], []
    for index in range(KM - 2):
        up = t["in_updraft"][index]
        qcko_candidate = (t["keep"][index] * carry + t["source"][index]) / t["den"][index]
        qcko_current = torch.where(up, qcko_candidate, qcko[:, index + 1])
        dq_cloud = t["eta"][index] * (qcko_current - t["xqrch"][index])
        precip = t["precip_window"][index] & (dq_cloud > 0.0)
        qlk_cloud = dq_cloud / torch.where(precip, t["cloud_den"][index], 1.0)
        qlk_plain = dq_cloud / torch.where(precip, t["plain_den"][index], 1.0)
        qlk = torch.where(t["cloud_level"][index], qlk_cloud, qlk_plain)
        xaa0 = torch.where(precip & t["below_top"][index], xaa0 - t["dz_grav"][index] * qlk, xaa0)
        carry = torch.where(precip, qlk + t["xqrch"][index], qcko_current)
        cells.append(carry)
        xpwav = torch.where(precip, xpwav + t["rain_factor"][index] * qlk, xpwav)
        xaa0 = torch.where(t["buoy"][index], xaa0 + t["heat"][index], xaa0)
        xaa0 = torch.where(t["buoy"][index], xaa0 + t["moisture"][index], xaa0)
    qcko = chunks.place_cells(qcko, cells, 1)
    return dxcrtuf, asqecflg, qo, to, po, qeso, heo, heso, hcko, qcko, dbyo, xaa0, xpwav


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
