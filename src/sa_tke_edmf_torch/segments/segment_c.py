# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment C: geometry, liquid water, buoyancy, and virtual potential temperature."""

from __future__ import annotations

import torch
from ufs_physics_common.precision import working_power

from sa_tke_edmf_torch.constants import (
    CP,
    EL2ORC,
    ELOCP,
    EPS,
    EPSM1,
    F32_0P001,
    F32_0P01,
    F32_0P49,
    F32_1E_M4,
    F32_1E_M6,
    F32_1E_M10,
    FV,
    GRAV,
    HFUS,
    HVAP,
    IM,
    KM,
    NTRAC,
    NTQV,
    QLMIN,
    QMIN,
    source_real,
)
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64
from sa_tke_edmf_torch.thermo import fpvs
from sa_tke_edmf_torch.tracers import satmedmf_hydrometeor_indices

CHECKPOINT_FIELDS = ("zl", "zi", "theta", "qlx", "thvx")


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    (F32_0P01, F32_0P001, F32_0P49, F32_1E_M4, F32_1E_M6,
     F32_1E_M10, QMIN, QLMIN) = tuple(source_real(v, state.config) for v in
        (0.01, 0.001, 0.49, 1e-4, 1e-6, 1e-10, 1e-8, 1e-12))
    KM = state.require("q1").shape[1]
    NTRAC = state.require("q1").shape[2]
    q1 = state.require("q1")
    t1 = state.require("t1")
    psk = state.require("psk")
    prsl = state.require("prsl")
    prslk = state.require("prslk")
    phii = state.require("phii")
    phil = state.require("phil")
    IM = q1.shape[0]
    require_float64("q1", q1, (IM, KM, NTRAC))
    require_float64("t1", t1, (IM, KM))
    require_float64("psk", psk, (IM,))
    require_float64("prsl", prsl, (IM, KM))
    require_float64("prslk", prslk, (IM, KM))
    require_float64("phii", phii, (IM, KM + 1))
    require_float64("phil", phil, (IM, KM))

    zi = phii * (1.0 / GRAV)
    zl = phil * (1.0 / GRAV)
    zm = zi[:, 1:]
    pix = psk[:, None] / prslk
    plyr = F32_0P01 * prsl
    theta = t1 * pix
    ntcw, ntiw, _ = satmedmf_hydrometeor_indices(state.config, NTRAC)
    qliq = torch.clamp_min(q1[:, :, ntcw - 1], QLMIN)
    if bool(state.config.get("sa3dtke", False)):
        qice_source = q1[:, :, ntiw - 1] + q1[:, :, 4] + q1[:, :, 5]
    else:
        qice_source = q1[:, :, ntiw - 1]
    qice = torch.clamp_min(qice_source, QLMIN)
    qlx = qliq + qice
    qv = torch.clamp_min(q1[:, :, NTQV - 1], QMIN)
    thlx = theta - pix * ELOCP * qlx
    qtx = qv + qlx
    thetae = theta + ELOCP * pix * qv
    thlvx = thlx * (1.0 + FV * qtx)
    thvx = theta * (1.0 + FV * qv - qlx)
    tvx = t1 * (1.0 + FV * qv - qlx)
    svx = CP * tvx
    slx = CP * t1 + phil - (HVAP * qliq + (HVAP + HFUS) * qice)

    es = F32_0P01 * fpvs(t1)
    qstl = torch.clamp_min(EPS * es / (plyr + EPSM1 * es), QMIN)
    rhly = torch.clamp(qv / qstl, min=0.0, max=1.0)
    clwt = F32_1E_M6 * (plyr * F32_0P001)
    onemrh = torch.clamp_min(1.0 - rhly, F32_1E_M10)
    tem1 = torch.clamp(working_power(onemrh * qstl, F32_0P49), min=F32_1E_M4, max=1.0)
    value = torch.clamp(torch.div(100.0, tem1) * qlx, min=0.0, max=50.0)
    cfly_candidate = torch.clamp(torch.sqrt(torch.sqrt(rhly)) * (1.0 - torch.exp(-value)), min=0.0, max=1.0)
    cfly = torch.where(qlx > clwt, cfly_candidate, torch.zeros_like(qlx))

    rdzt = 1.0 / (zl[:, 1:] - zl[:, :-1])
    tem = 0.5 * (svx[:, :-1] + svx[:, 1:])
    tem_t = 0.5 * (t1[:, :-1] + t1[:, 1:])
    tem_qs = 0.5 * (qstl[:, :-1] + qstl[:, 1:])
    cfh = torch.minimum(cfly[:, 1:], 0.5 * (cfly[:, :-1] + cfly[:, 1:]))
    alp = torch.div(GRAV, tem)
    gamma = EL2ORC * tem_qs / (tem_t * tem_t)
    epsi = tem_t / ELOCP
    beta = (1.0 + gamma * epsi * (1.0 + FV)) / (1.0 + gamma)
    chx = cfh * alp * beta + (1.0 - cfh) * alp
    cqx = cfh * alp * HVAP * (beta - epsi) + (1.0 - cfh) * FV * GRAV
    ptem1 = (slx[:, 1:] - slx[:, :-1]) * rdzt
    ptem2 = (qtx[:, 1:] - qtx[:, :-1]) * rdzt
    bf = chx * ptem1 + cqx * ptem2
    return state.with_updates(
        {
            "zl": zl,
            "zi": zi,
            "zm": zm,
            "plyr": plyr,
            "pix": pix,
            "theta": theta,
            "thetae": thetae,
            "qlx": qlx,
            "qtx": qtx,
            "thlx": thlx,
            "thlvx": thlvx,
            "thvx": thvx,
            "tvx": tvx,
            "svx": svx,
            "slx": slx,
            "qstl": qstl,
            "cfly": cfly,
            "rdzt": rdzt,
            "bf": bf,
        }
    )


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
