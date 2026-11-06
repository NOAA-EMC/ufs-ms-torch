# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment M: unit-mass feedback increments."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch.compile import run_compile_island
from sa_sas_deep_convection_torch.constants import KM, NTR
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real

CHECKPOINT_FIELDS = ("dellah", "dellaq", "dellau", "dellav", "dellae", "delubar", "delvbar")


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 2242-2400 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    cnvflg = state.require("cnvflg")
    kmax = state.require("kmax")
    kb = state.require("kb")
    jmin = state.require("jmin")
    ktcon = state.require("ktcon")
    kd94 = state.require("kd94")
    del_ = state.require("del")
    zi = state.require("zi")
    heo = state.require("heo")
    heso = state.require("heso")
    hcko = state.require("hcko")
    hcdo = state.require("hcdo")
    qcko = state.require("qcko")
    qlko_ktcon = state.require("qlko_ktcon")
    qrcko = state.require("qrcko")
    qrcdo = state.require("qrcdo")
    qcdo = state.require("qcdo")
    uo = state.require("uo")
    vo = state.require("vo")
    ucko = state.require("ucko")
    vcko = state.require("vcko")
    ucdo = state.require("ucdo")
    vcdo = state.require("vcdo")
    eta = state.require("eta")
    etad = state.require("etad")
    edto = state.require("edto")
    xlamue = state.require("xlamue")
    xlamud = state.require("xlamud")
    xlamdet = state.require("xlamdet")
    xlamddt = state.require("xlamddt")
    xlamd = state.require("xlamd")
    ecko = state.require("ecko")
    ercko = state.require("ercko")
    ecdo = state.require("ecdo")
    dellal = state.require("dellal").clone()
    ncol = cnvflg.shape[0]
    for name, tensor in {
        "cnvflg": cnvflg,
        "kmax": kmax,
        "kb": kb,
        "jmin": jmin,
        "ktcon": ktcon,
        "kd94": kd94,
        "xlamdet": xlamdet,
        "xlamddt": xlamddt,
        "xlamd": xlamd,
    }.items():
        require_real(name, tensor, (ncol,))
    for name, tensor in {
        "del": del_,
        "zi": zi,
        "heo": heo,
        "heso": heso,
        "hcko": hcko,
        "hcdo": hcdo,
        "qcko": qcko,
        "qrcko": qrcko,
        "qrcdo": qrcdo,
        "qcdo": qcdo,
        "uo": uo,
        "vo": vo,
        "ucko": ucko,
        "vcko": vcko,
        "ucdo": ucdo,
        "vcdo": vcdo,
        "eta": eta,
        "etad": etad,
        "xlamue": xlamue,
        "xlamud": xlamud,
        "dellal": dellal,
    }.items():
        require_real(name, tensor, (ncol, KM))
    for name, tensor in {"ecko": ecko, "ercko": ercko, "ecdo": ecdo}.items():
        require_real(name, tensor, (ncol, KM, NTR))

    grav = state.require("grav")
    hwrf_samfdeep = bool(state.config.get("hwrf_samfdeep", False))
    dellah, dellaq, dellau, dellav, dellae, dellal, delubar, delvbar = run_compile_island(
        "m_feedback",
        "feedback_increment",
        _feedback_increment_block,
        cnvflg,
        kb,
        jmin,
        ktcon,
        kd94,
        del_,
        zi,
        heo,
        hcko,
        hcdo,
        qcko,
        qlko_ktcon,
        qrcko,
        qrcdo,
        qcdo,
        uo,
        vo,
        ucko,
        vcko,
        ucdo,
        vcdo,
        eta,
        etad,
        edto,
        xlamue,
        xlamud,
        xlamdet,
        xlamddt,
        xlamd,
        ecko,
        ercko,
        ecdo,
        dellal,
        grav,
        hwrf_samfdeep,
    )
    return state.with_updates(
        {
            "dellah": dellah,
            "dellaq": dellaq,
            "dellau": dellau,
            "dellav": dellav,
            "dellae": dellae,
            "dellal": dellal,
            "delubar": delubar,
            "delvbar": delvbar,
        }
    )


def _feedback_increment_block(
    cnvflg: torch.Tensor,
    kb: torch.Tensor,
    jmin: torch.Tensor,
    ktcon: torch.Tensor,
    kd94: torch.Tensor,
    del_: torch.Tensor,
    zi: torch.Tensor,
    heo: torch.Tensor,
    hcko: torch.Tensor,
    hcdo: torch.Tensor,
    qcko: torch.Tensor,
    qlko_ktcon: torch.Tensor,
    qrcko: torch.Tensor,
    qrcdo: torch.Tensor,
    qcdo: torch.Tensor,
    uo: torch.Tensor,
    vo: torch.Tensor,
    ucko: torch.Tensor,
    vcko: torch.Tensor,
    ucdo: torch.Tensor,
    vcdo: torch.Tensor,
    eta: torch.Tensor,
    etad: torch.Tensor,
    edto: torch.Tensor,
    xlamue: torch.Tensor,
    xlamud: torch.Tensor,
    xlamdet: torch.Tensor,
    xlamddt: torch.Tensor,
    xlamd: torch.Tensor,
    ecko: torch.Tensor,
    ercko: torch.Tensor,
    ecdo: torch.Tensor,
    dellal: torch.Tensor,
    grav: torch.Tensor,
    hwrf_samfdeep: bool,
    *,
    conditioned_heat: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    KM = zi.shape[1]
    ncol = cnvflg.shape[0]
    active = cnvflg != 0.0
    dtype = cnvflg.dtype
    device = cnvflg.device
    levels = torch.arange(1, KM + 1, dtype=dtype, device=device).reshape(1, KM)
    mid = slice(1, KM - 1)

    dellah = torch.zeros_like(heo)
    dellaq = torch.zeros_like(heo)
    dellau = torch.zeros_like(heo)
    dellav = torch.zeros_like(heo)
    dellae = torch.zeros_like(ecko)

    surface_mask = active[:, None] & (levels == 1.0)
    dp_all = 1000.0 * del_
    tem_surface = edto * etad[:, 0] * grav / dp_all[:, 0]
    dellah = torch.where(surface_mask, (tem_surface * (hcdo[:, 0] - heo[:, 0]))[:, None], dellah)
    dellaq = torch.where(surface_mask, (tem_surface * qrcdo[:, 0])[:, None], dellaq)
    dellau = torch.where(surface_mask, (tem_surface * (ucdo[:, 0] - uo[:, 0]))[:, None], dellau)
    dellav = torch.where(surface_mask, (tem_surface * (vcdo[:, 0] - vo[:, 0]))[:, None], dellav)

    if not hwrf_samfdeep:
        surface_e = edto[:, None] * etad[:, 0:1] * ecdo[:, 0, :] * grav / dp_all[:, 0:1]
        dellae = torch.where(surface_mask[:, :, None], surface_e[:, None, :], dellae)

    mid_levels = levels[:, mid]
    layer_mid = active[:, None] & (mid_levels < ktcon[:, None])
    aup = torch.where(mid_levels <= kb[:, None], torch.zeros((ncol, KM - 2), dtype=dtype, device=device), torch.ones((ncol, KM - 2), dtype=dtype, device=device))
    adw = torch.where(mid_levels > jmin[:, None], torch.zeros((ncol, KM - 2), dtype=dtype, device=device), torch.ones((ncol, KM - 2), dtype=dtype, device=device))
    dp_mid = dp_all[:, mid]
    dz_mid = zi[:, mid] - zi[:, : KM - 2]
    dv1h = heo[:, mid]
    dv2h = 0.5 * (heo[:, mid] + heo[:, : KM - 2])
    dv3h = heo[:, : KM - 2]
    tem = 0.5 * (xlamue[:, mid] + xlamue[:, : KM - 2])
    tem1 = 0.5 * (xlamud[:, mid] + xlamud[:, : KM - 2])
    ptem1 = torch.where(mid_levels <= kd94[:, None], xlamd[:, None] + xlamddt[:, None], xlamddt[:, None])
    factor = grav / dp_mid
    if conditioned_heat:
        h_term = _conditioned_heat_feedback(heo, hcko, hcdo, eta, etad, aup, adw,
            edto[:, None], tem, tem1, ptem1, xlamdet[:, None], dz_mid, factor)
    else:
        h_term = (
            (aup * eta[:, mid] - adw * edto[:, None] * etad[:, mid]) * dv1h
            - (aup * eta[:, : KM - 2] - adw * edto[:, None] * etad[:, : KM - 2]) * dv3h
            - (aup * tem * eta[:, : KM - 2] + adw * edto[:, None] * xlamdet[:, None] * etad[:, mid]) * dv2h * dz_mid
            + aup * tem1 * eta[:, : KM - 2] * 0.5 * (hcko[:, mid] + hcko[:, : KM - 2]) * dz_mid
            + adw * edto[:, None] * ptem1 * etad[:, mid] * 0.5 * (hcdo[:, mid] + hcdo[:, : KM - 2]) * dz_mid
        ) * factor
    mid_mask = torch.zeros((ncol, KM), dtype=torch.bool, device=device)
    mid_mask[:, mid] = layer_mid
    mid_update = torch.zeros_like(heo)
    mid_update[:, mid] = h_term
    dellah = torch.where(mid_mask, mid_update, dellah)

    q_term = (aup * (-eta[:, mid] * qrcko[:, mid] + eta[:, : KM - 2] * qcko[:, : KM - 2]) - adw * edto[:, None] * (-etad[:, mid] * qrcdo[:, mid] + etad[:, : KM - 2] * qcdo[:, : KM - 2])) * factor
    u_term = (
        aup * (eta[:, mid] * (uo[:, mid] - ucko[:, mid]) - eta[:, : KM - 2] * (uo[:, : KM - 2] - ucko[:, : KM - 2]))
        - adw * edto[:, None] * (etad[:, mid] * (uo[:, mid] - ucdo[:, mid]) - etad[:, : KM - 2] * (uo[:, : KM - 2] - ucdo[:, : KM - 2]))
    ) * factor
    v_term = (
        aup * (eta[:, mid] * (vo[:, mid] - vcko[:, mid]) - eta[:, : KM - 2] * (vo[:, : KM - 2] - vcko[:, : KM - 2]))
        - adw * edto[:, None] * (etad[:, mid] * (vo[:, mid] - vcdo[:, mid]) - etad[:, : KM - 2] * (vo[:, : KM - 2] - vcdo[:, : KM - 2]))
    ) * factor
    q_update = torch.zeros_like(dellaq)
    u_update = torch.zeros_like(dellau)
    v_update = torch.zeros_like(dellav)
    q_update[:, mid] = q_term
    u_update[:, mid] = u_term
    v_update[:, mid] = v_term
    dellaq = torch.where(mid_mask, q_update, dellaq)
    dellau = torch.where(mid_mask, u_update, dellau)
    dellav = torch.where(mid_mask, v_update, dellav)

    if not hwrf_samfdeep:
        # Tracer source arithmetic multiplies by gravity before pressure division.
        e_delta = (
            aup[:, :, None] * (-eta[:, mid, None] * ercko[:, mid, :] + eta[:, : KM - 2, None] * ecko[:, : KM - 2, :])
            - adw[:, :, None]
            * edto[:, None, None]
            * (-etad[:, mid, None] * ecdo[:, mid, :] + etad[:, : KM - 2, None] * ecdo[:, : KM - 2, :])
        ) * grav / dp_mid[:, :, None]
        e_update = torch.zeros_like(dellae)
        e_update[:, mid, :] = e_delta
        dellae = torch.where(mid_mask[:, :, None], e_update, dellae)

    top_mask = active[:, None] & (levels >= 2.0) & (ktcon[:, None] == levels)
    top_tem = eta[:, : KM - 1] * grav / dp_all[:, 1:]
    top_full = torch.zeros_like(heo)
    top_full[:, 1:] = top_tem
    h_top = torch.zeros_like(heo)
    q_top = torch.zeros_like(dellaq)
    u_top = torch.zeros_like(dellau)
    v_top = torch.zeros_like(dellav)
    h_top[:, 1:] = top_tem * (hcko[:, : KM - 1] - heo[:, : KM - 1])
    q_top[:, 1:] = top_tem * qcko[:, : KM - 1]
    u_top[:, 1:] = top_tem * (ucko[:, : KM - 1] - uo[:, : KM - 1])
    v_top[:, 1:] = top_tem * (vcko[:, : KM - 1] - vo[:, : KM - 1])
    dellah = torch.where(top_mask, h_top, dellah)
    dellaq = torch.where(top_mask, q_top, dellaq)
    dellau = torch.where(top_mask, u_top, dellau)
    dellav = torch.where(top_mask, v_top, dellav)
    dellal = torch.where(top_mask, top_full * qlko_ktcon[:, None], dellal)
    if not hwrf_samfdeep:
        e_top = torch.zeros_like(dellae)
        e_top[:, 1:, :] = eta[:, : KM - 1, None] * ecko[:, : KM - 1, :] * grav / dp_all[:, 1:, None]
        dellae = torch.where(top_mask[:, :, None], e_top, dellae)

    zero_col = torch.zeros_like(cnvflg)
    return dellah, dellaq, dellau, dellav, dellae, dellal, zero_col.clone(), zero_col.clone()


def _conditioned_heat_feedback(heo, hcko, hcdo, eta, etad, aup, adw, edto,
                               entr, detr, down_det, down_entr, dz, factor):
    """Experimental shifted linear heat flux, not selected by the coordinator.

    Substitute each heat H=A+(H-A), with A the lower environmental heat.
    The coefficient sum multiplies A. Preserve its mass-flux residual: rounded
    stored eta values do not satisfy an exact zero-residual identity.
    """
    upper, lower = slice(1, -1), slice(None, -2)
    residual = aup * ((eta[:, upper] - eta[:, lower])
        - entr * eta[:, lower] * dz + detr * eta[:, lower] * dz)
    residual = residual + adw * edto * ((etad[:, lower] - etad[:, upper])
        - down_entr * etad[:, upper] * dz + down_det * etad[:, upper] * dz)
    anchor = heo[:, lower]
    delta = heo[:, upper] - anchor
    return ((aup * eta[:, upper] - adw * edto * etad[:, upper]) * delta
        - (aup * entr * eta[:, lower] + adw * edto * down_entr * etad[:, upper]) * (.5 * delta) * dz
        + aup * detr * eta[:, lower] * (.5 * ((hcko[:, upper] - anchor) + (hcko[:, lower] - anchor))) * dz
        + adw * edto * down_det * etad[:, upper] * (.5 * ((hcdo[:, upper] - anchor) + (hcdo[:, lower] - anchor))) * dz
        + anchor * residual) * factor


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
