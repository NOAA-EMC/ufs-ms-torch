# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Post-dycore projection and the selected R4 pre-convection cap sequence."""

from dataclasses import replace

import torch
from gfs_surface_layer_torch.driver import RVRDM1

from .containers import NativeAtmosphere, ProjectionMesh, Stateout
from .physics_prefix import _get_prs
from .photochemistry import apply_photochemistry
from .projection import project_atmosphere, r4_tracer_contract
from .precision import working_like


def reset_microphysics_scratch(statein, prefix, *, nn):
    """CCPP_step resets each group; retain initialized tracer transport metadata.

    These are the reset members consumed through convection. Later cap stages
    must bind their own complete reset inventory before they become runnable.
    """
    if type(nn) is not int or nn < 2:
        raise ValueError("microphysics requires a static clw extent >=2")
    shape = statein.tgrs.shape
    work = dict(prefix.interstitial)
    for name in ("dtdt", "dudt", "dvdt", "del", "rhc", "save_tcp"):
        work[name] = torch.zeros_like(statein.tgrs)
    for name in ("dqdt", "save_q"):
        work[name] = torch.zeros_like(statein.qgrs)
    work["del_gz"] = torch.zeros_like(statein.phii)
    clw = statein.tgrs.new_zeros((*shape, nn))
    channels = torch.arange(nn, device=clw.device)
    work["clw"] = torch.where(channels == 1, -999.9, clw)
    work["dclw"] = torch.zeros_like(clw)
    work["dvdftra"] = torch.zeros_like(statein.qgrs)
    work["tkeh"] = torch.zeros_like(statein.phii)
    for name in ("dd_mf", "dt_mf", "cnvc", "ten_t", "ten_u", "ten_v"):
        work[name] = torch.zeros_like(statein.tgrs)
    work["ten_q"] = torch.zeros_like(statein.qgrs)
    for name in ("raincd", "raincs", "cld1d"):
        work[name] = statein.tgrs.new_zeros(shape[0])
    for name in ("kbot", "ktop", "kcnv"):
        work[name] = torch.zeros(shape[0], dtype=torch.int32, device=clw.device)
    for name in ("work1", "work2"):
        work[name] = statein.tgrs.new_zeros(shape[0])
    for name in ("kpbl", "islmsk"):
        work[name] = torch.zeros(shape[0], dtype=torch.int32, device=clw.device)
    work["kinver"] = torch.full((shape[0],), shape[1], dtype=torch.int32, device=clw.device)
    return replace(prefix, interstitial=work)


def get_phi(statein, prefix):
    """get_phi_fv3: reset surface geopotential and retain the top del_gz carry."""
    state = prefix.stateout
    work = prefix.interstitial
    increments = work["del_gz"][:, :-1] * state.gt0 * (
        1.0 + RVRDM1 * torch.clamp_min(state.gq0[:, :, 0], 0.0)
    )
    interfaces = [torch.zeros_like(state.gt0[:, 0])]
    for k in range(state.gt0.shape[1]):
        interfaces.append(interfaces[-1] + increments[:, k])
    phii = torch.stack(interfaces, dim=1)
    phil = 0.5 * (phii[:, :-1] + phii[:, 1:])
    return replace(statein, phii=phii, phil=phil), replace(prefix,
        interstitial={**work, "del_gz": torch.cat((increments, work["del_gz"][:, -1:]), dim=1)})


def interstitial_3(statein, prefix, *, control, crtrh, rhcmax):
    """TEMPO ice/liquid save fields, masked tracer packing and critical RH."""
    for name in ("imp_physics", "imp_physics_tempo"):
        if type(control[name]) is not int:
            raise TypeError("interstitial3 microphysics identifiers must be static integers")
    if control["imp_physics"] != 88 or control["imp_physics_tempo"] != 88:
        raise NotImplementedError("interstitial3 selects the R4 TEMPO branch")
    for name in ("progsigma", "progomega", "cscnv", "satmedmf", "trans_trac", "ras", "ltaerosol"):
        if type(control[name]) is not bool:
            raise TypeError("interstitial3 switches must be static booleans")
    if control["progsigma"] or control["progomega"]:
        raise NotImplementedError("prognostic convection fractions/velocities are not bound")
    work, state = prefix.interstitial, prefix.stateout
    q, t = state.gq0, state.gt0
    if q.ndim != 3 or q.shape[:2] != t.shape or min(q.shape) < 1 or q.shape[2] != 10:
        raise ValueError("invalid interstitial3 model dimensions")
    ncol, nlev, ntrac = q.shape
    ice, liquid = control["ntiw"], control["ntcw"]
    if any(type(v) is not int or not 1 <= v <= ntrac for v in (ice, liquid)):
        raise ValueError("invalid TEMPO cloud tracer indices")
    if work["clw"].ndim != 3 or work["clw"].shape[:2] != t.shape or work["clw"].shape[2] < 2:
        raise ValueError("invalid interstitial3 clw dimensions")
    for name, value, shape in (
        ("Stateout tracers", q, q.shape), ("temperature", t, t.shape),
        ("prslk", statein.prslk, t.shape), ("clw", work["clw"], work["clw"].shape),
        ("save_q", work["save_q"], q.shape), ("save_tcp", work["save_tcp"], t.shape),
        ("rhc", work["rhc"], t.shape), ("work1", work["work1"], (ncol,)),
        ("work2", work["work2"], (ncol,)), ("crtrh", crtrh, (3,)), ("rhcmax", rhcmax, ()),
    ):
        if value.shape != shape or not working_like(value, q) or value.device != q.device:
            raise ValueError(f"invalid interstitial3 {name}")
        torch._assert_async(torch.isfinite(value).all(), f"nonfinite interstitial3 {name}")
    flags, kpbl = work["otsptflag"], work["kpbl"]
    if flags.shape != (ntrac,) or flags.dtype != torch.bool or flags.device != q.device:
        raise ValueError("invalid interstitial3 tracer transport flags")
    if kpbl.shape != (ncol,) or kpbl.dtype != torch.int32 or kpbl.device != q.device:
        raise ValueError("invalid interstitial3 PBL indices")
    kk = torch.clamp_min(kpbl.to(torch.int64), 10)
    torch._assert_async((kk <= nlev).all(), "interstitial3 PBL top exceeds model levels")
    pbl_pressure = torch.gather(statein.prslk, 1, (kk - 1)[:, None])
    torch._assert_async(((pbl_pressure > 0.) & (pbl_pressure < 1.)).all(), "invalid interstitial3 PBL pressure")
    levels = torch.arange(1, nlev + 1, device=q.device)[None, :]
    bottom = crtrh[0] - (crtrh[0] - crtrh[1]) * (1.0 - statein.prslk) / (1.0 - pbl_pressure)
    top = crtrh[1] - (crtrh[1] - crtrh[2]) * (pbl_pressure - statein.prslk) / pbl_pressure
    rhc = torch.clamp(rhcmax * work["work1"][:, None]
        + torch.where(levels < kk[:, None], bottom, top) * work["work2"][:, None], 0., 1.)

    clw = work["clw"]
    channels = torch.arange(clw.shape[2], device=q.device)
    if control["cscnv"] or control["satmedmf"] or control["trans_trac"] or control["ras"]:
        destinations = torch.cumsum(flags[1:].to(torch.int64), dim=0) + 1
        torch._assert_async((destinations[-1] < clw.shape[2]), "interstitial3 transport exceeds clw allocation")
        for n in range(1, ntrac):
            mask = flags[n] & (channels == destinations[n - 1])
            clw = torch.where(mask, q[:, :, n:n+1], clw)
    clw = torch.cat((q[:, :, ice-1:ice], q[:, :, liquid-1:liquid], clw[:, :, 2:]), dim=2)
    slots = torch.arange(ntrac, device=q.device)
    save_q = torch.where(slots == ice - 1, q, work["save_q"])
    if control["ltaerosol"]:
        save_q = torch.where(slots == liquid - 1, q, save_q)
    return replace(prefix, interstitial={**work, "rhc": rhc, "clw": clw,
        "save_q": save_q, "save_tcp": t.clone()})


def prepare_microphysics_prefix(native_state, mesh, statein, grid, prefix, *,
    log_pressure, forcing, dtp, control, crtrh, rhcmax, h2o_log_pressure=None, h2o_forcing=None):
    """Project live dycore output, seed Stateout and execute E6a in cap order.

    The caller retains the projection's native surface pressure. Deformation,
    soil state and all unrelated radiation/surface carries remain live. Source
    CCPP_step resets the microphysics scratch before entering this cap.
    """
    projection = project_atmosphere(
        NativeAtmosphere.from_mpas_state(native_state, n_cells_solve=statein.tgrs.shape[0]),
        ProjectionMesh.from_mpas_state({**mesh, **native_state}), statein,
        tracers=r4_tracer_contract())
    projected = projection.statein
    reset = reset_microphysics_scratch(projected, prefix, nn=control["nn"])
    seeded = replace(reset, stateout=Stateout.from_statein(projected),
        interstitial=_get_prs(projected, reset.interstitial))
    chemistry = apply_photochemistry(projected, grid, seeded, log_pressure=log_pressure,
        forcing=forcing, dtp=dtp, control=control,
        h2o_log_pressure=h2o_log_pressure, h2o_forcing=h2o_forcing)
    heights, chemistry = get_phi(projected, chemistry)
    result = interstitial_3(heights, chemistry, control=control, crtrh=crtrh, rhcmax=rhcmax)
    return replace(projection, statein=heights), result
