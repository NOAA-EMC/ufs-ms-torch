# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment R: momentum tridiagonal solve and diagnostics."""

from __future__ import annotations

import torch

from sa_tke_edmf_torch.compile import run_compile_island
from sa_tke_edmf_torch.constants import FV, IM, KM, NDTEND, NTRAC, NTQV, QMIN, RD
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64
from sa_tke_edmf_torch.diagnostics import diagnostic_slots, add_diagnostic_

CHECKPOINT_FIELDS = ("ten_u", "ten_v", "dusfc", "dvsfc", "hpbl", "kpbl", "dku3d_h", "dku3d_e")


def _tridi2_momentum(
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
    rhs_u: torch.Tensor,
    rhs_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    KM = diagonal.shape[1]
    # Level-major rows: each sweep step reads one contiguous row per coefficient.
    lower, diagonal, upper, rhs_u, rhs_v = (x.t().contiguous() for x in (lower, diagonal, upper, rhs_u, rhs_v))
    fk = 1.0 / diagonal[0]
    upper_cols = [fk * upper[0]]
    u_cols = [fk * rhs_u[0]]
    v_cols = [fk * rhs_v[0]]
    for k in range(1, KM - 1):
        fk = 1.0 / (diagonal[k] - lower[k - 1] * upper_cols[-1])
        upper_cols.append(fk * upper[k])
        u_cols.append(fk * (rhs_u[k] - lower[k - 1] * u_cols[-1]))
        v_cols.append(fk * (rhs_v[k] - lower[k - 1] * v_cols[-1]))
    fk = 1.0 / (diagonal[KM - 1] - lower[KM - 2] * upper_cols[-1])
    u_cols.append(fk * (rhs_u[KM - 1] - lower[KM - 2] * u_cols[-1]))
    v_cols.append(fk * (rhs_v[KM - 1] - lower[KM - 2] * v_cols[-1]))
    u_solution_cols = [u_cols[-1]]
    v_solution_cols = [v_cols[-1]]
    for k in range(KM - 2, -1, -1):
        u_solution_cols.append(u_cols[k] - upper_cols[k] * u_solution_cols[-1])
        v_solution_cols.append(v_cols[k] - upper_cols[k] * v_solution_cols[-1])
    return (torch.stack(list(reversed(u_solution_cols)), dim=0).t().contiguous(),
            torch.stack(list(reversed(v_solution_cols)), dim=0).t().contiguous())


def _momentum_system_rhs(
    u1: torch.Tensor,
    v1: torch.Tensor,
    delp: torch.Tensor,
    prsl: torch.Tensor,
    zi: torch.Tensor,
    rdzt: torch.Tensor,
    dku: torch.Tensor,
    xmf: torch.Tensor,
    xmfd: torch.Tensor,
    ucko: torch.Tensor,
    vcko: torch.Tensor,
    ucdo: torch.Tensor,
    vcdo: torch.Tensor,
    stress: torch.Tensor,
    spd1: torch.Tensor,
    kpbl: torch.Tensor,
    mrad: torch.Tensor,
    krad: torch.Tensor,
    pcnvflg: torch.Tensor,
    scuflg: torch.Tensor,
    dt2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    KM = u1.shape[1]
    dtype = u1.dtype
    device = u1.device
    ncol = u1.shape[0]
    km1 = KM - 1

    dtdz1 = dt2 / (zi[:, 1] - zi[:, 0])
    diagonal = torch.zeros((ncol, KM), dtype=dtype, device=device)
    dtodsd = dt2 / delp[:, :km1]
    dtodsu = dt2 / delp[:, 1:KM]
    dsig = prsl[:, :km1] - prsl[:, 1:KM]
    rdz = rdzt[:, :km1]
    dsdz2 = dsig * dku[:, :km1] * rdz * rdz
    upper = -dtodsd * dsdz2
    lower = -dtodsu * dsdz2
    diagonal[:, 0] = 1.0 + dtdz1 * stress / spd1 - upper[:, 0]
    diagonal[:, 1:km1] = 1.0 - lower[:, : km1 - 1] - upper[:, 1:km1]
    diagonal[:, KM - 1] = 1.0 - lower[:, km1 - 1]
    tem2 = dsig * rdz

    levels = torch.arange(1, KM, dtype=torch.int64, device=device)
    updraft_active = pcnvflg[:, None] & (levels[None, :] < kpbl[:, None])
    ptem = 0.5 * tem2 * xmf[:, :km1]
    ptem1 = dtodsd * ptem
    ptem2 = dtodsu * ptem
    delta_u = (ucko[:, :km1] + ucko[:, 1:KM]) - (u1[:, :km1] + u1[:, 1:KM])
    delta_v = (vcko[:, :km1] + vcko[:, 1:KM]) - (v1[:, :km1] + v1[:, 1:KM])
    zeros_2d = torch.zeros((ncol, km1), dtype=dtype, device=device)
    rhs_u_left_delta = torch.where(updraft_active, -delta_u * ptem1, zeros_2d)
    rhs_u_right_delta = torch.where(updraft_active, delta_u * ptem2, zeros_2d)
    rhs_v_left_delta = torch.where(updraft_active, -delta_v * ptem1, zeros_2d)
    rhs_v_right_delta = torch.where(updraft_active, delta_v * ptem2, zeros_2d)

    downdraft_active = scuflg[:, None] & (levels[None, :] >= mrad[:, None]) & (levels[None, :] < krad[:, None])
    ptem = 0.5 * tem2 * xmfd[:, :km1]
    ptem1 = dtodsd * ptem
    ptem2 = dtodsu * ptem
    delta_u = (ucdo[:, :km1] + ucdo[:, 1:KM]) - (u1[:, :km1] + u1[:, 1:KM])
    delta_v = (vcdo[:, :km1] + vcdo[:, 1:KM]) - (v1[:, :km1] + v1[:, 1:KM])
    down_u_left = torch.where(downdraft_active, delta_u * ptem1, zeros_2d)
    down_u_right = torch.where(downdraft_active, -delta_u * ptem2, zeros_2d)
    down_v_left = torch.where(downdraft_active, delta_v * ptem1, zeros_2d)
    down_v_right = torch.where(downdraft_active, -delta_v * ptem2, zeros_2d)

    rhs_u = u1.clone()
    rhs_v = v1.clone()
    # Keep below/above interface order and separate source plume additions.
    rhs_u[:, 1:KM] = rhs_u[:, 1:KM] + rhs_u_right_delta
    rhs_u[:, 1:KM] = rhs_u[:, 1:KM] + down_u_right
    rhs_u[:, :km1] = rhs_u[:, :km1] + rhs_u_left_delta
    rhs_u[:, :km1] = rhs_u[:, :km1] + down_u_left
    rhs_v[:, 1:KM] = rhs_v[:, 1:KM] + rhs_v_right_delta
    rhs_v[:, 1:KM] = rhs_v[:, 1:KM] + down_v_right
    rhs_v[:, :km1] = rhs_v[:, :km1] + rhs_v_left_delta
    rhs_v[:, :km1] = rhs_v[:, :km1] + down_v_left
    return lower, diagonal, upper, rhs_u, rhs_v


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    KM = state.require("q1").shape[1]
    NTRAC = state.require("q1").shape[2]
    q1 = state.require("q1")
    t1 = state.require("t1")
    u1 = state.require("u1")
    v1 = state.require("v1")
    delp = state.require("del")
    prsl = state.require("prsl")
    zi = state.require("zi")
    rdzt = state.require("rdzt")
    dku = state.require("dku")
    xmf = state.require("xmf")
    xmfd = state.require("xmfd")
    ucko = state.require("ucko")
    vcko = state.require("vcko")
    ucdo = state.fields.get("ucdo")
    if ucdo is None:
        ucdo = torch.zeros_like(u1)
    vcdo = state.fields.get("vcdo")
    if vcdo is None:
        vcdo = torch.zeros_like(v1)
    stress = state.require("stress")
    spd1 = state.require("spd1")
    usfco = state.require("usfco")
    vsfco = state.require("vsfco")
    hpbl = state.fields.get("hpblx", state.require("hpbl"))
    kpbl_out = state.fields.get("kpblx", state.require("kpbl"))
    dku_h = state.require("dku_h")
    dkq_h = state.require("dkq_h")
    ten_u = state.require("ten_u").clone()
    ten_v = state.require("ten_v").clone()
    dtend = state.require("dtend").clone()
    delt = state.require("delt")
    kpbl = state.require("kpbl").to(torch.int64)
    krad = state.require("krad").to(torch.int64)
    mrad = state.require("mrad").to(torch.int64)
    pcnvflg = state.require("pcnvflg") > 0.5
    scuflg = state.require("scuflg") > 0.5
    IM = q1.shape[0]
    require_float64("q1", q1, (IM, KM, NTRAC))
    require_float64("t1", t1, (IM, KM))
    require_float64("u1", u1, (IM, KM))
    require_float64("v1", v1, (IM, KM))
    require_float64("del", delp, (IM, KM))
    require_float64("prsl", prsl, (IM, KM))
    require_float64("zi", zi, (IM, KM + 1))
    require_float64("rdzt", rdzt, (IM, KM - 1))
    require_float64("dku", dku, (IM, KM))
    require_float64("xmf", xmf, (IM, KM))
    require_float64("xmfd", xmfd, (IM, KM))
    require_float64("ucko", ucko, (IM, KM))
    require_float64("vcko", vcko, (IM, KM))
    require_float64("ucdo", ucdo, (IM, KM))
    require_float64("vcdo", vcdo, (IM, KM))
    require_float64("spd1", spd1, (IM,))
    require_float64("usfco", usfco, (IM,))
    require_float64("vsfco", vsfco, (IM,))
    require_float64("dku_h", dku_h, (IM, KM))
    require_float64("dkq_h", dkq_h, (IM, KM))
    require_float64("ten_u", ten_u, (IM, KM))
    require_float64("ten_v", ten_v, (IM, KM))
    slots = diagnostic_slots(state)

    dtype = u1.dtype
    device = u1.device
    dt2 = delt[0]
    rdt = 1.0 / dt2
    lower, diagonal, upper, rhs_u, rhs_v = run_compile_island(
        "tridiagonal",
        "momentum_system_rhs",
        _momentum_system_rhs,
        u1,
        v1,
        delp,
        prsl,
        zi,
        rdzt,
        dku,
        xmf,
        xmfd,
        ucko,
        vcko,
        ucdo,
        vcdo,
        stress,
        spd1,
        kpbl,
        mrad,
        krad,
        pcnvflg,
        scuflg,
        dt2,
    )

    solved_u, solved_v = run_compile_island(
        "tridiagonal",
        "momentum_thomas",
        _tridi2_momentum,
        lower,
        diagonal,
        upper,
        rhs_u,
        rhs_v,
    )
    ten_u = (solved_u - u1) * rdt
    ten_v = (solved_v - v1) * rdt

    qv_surface = torch.clamp_min(q1[:, 0, NTQV - 1], QMIN)
    rho_a = prsl[:, 0] / (RD * t1[:, 0] * (1.0 + FV * qv_surface))
    if bool(state.config.get("use_oceanuv", False)):
        spd1_m = torch.sqrt((u1[:, 0] - usfco) ** 2 + (v1[:, 0] - vsfco) ** 2)
        dusfc = -rho_a * stress * (u1[:, 0] - usfco) / spd1_m
        dvsfc = -rho_a * stress * (v1[:, 0] - vsfco) / spd1_m
    else:
        dusfc = -rho_a * stress * u1[:, 0] / spd1
        dvsfc = -rho_a * stress * v1[:, 0] / spd1

    if bool(state.config.get("ldiag3d", False)) and not bool(state.config.get("gen_tend", False)):
        dtend = dtend.clone()
        add_diagnostic_(dtend, ten_u * dt2, slots[1])
        add_diagnostic_(dtend, ten_v * dt2, slots[2])

    dku3d_h = torch.zeros((IM, KM), dtype=dtype, device=device)
    dku3d_e = torch.zeros((IM, KM), dtype=dtype, device=device)
    if bool(state.config.get("sa3dtke", False)):
        dku3d_h = dku_h
        dku3d_e = dkq_h

    return state.with_updates(
        {
            "ten_u": ten_u,
            "ten_v": ten_v,
            "dusfc": dusfc,
            "dvsfc": dvsfc,
            "hpbl": hpbl,
            "kpbl": kpbl_out,
            "dku3d_h": dku3d_h,
            "dku3d_e": dku3d_e,
            "momentum_u_solution": solved_u,
            "momentum_v_solution": solved_v,
            "dtend": dtend,
        }
    )


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
