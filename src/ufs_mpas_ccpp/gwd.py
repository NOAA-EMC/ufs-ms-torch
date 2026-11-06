# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""R4 topographic binding and legacy mountain-blocking/orographic-wave solver.

This module does not yet implement small-scale/nonstationary drag or UGWP post.
"""

from dataclasses import replace
from collections.abc import Mapping
import math

import torch
from gfs_surface_layer_torch.driver import CP, GRAV, RD, RVRDM1

from .physics_prefix import PhysicsPrefixResult
from .containers import Statein, static_tensor
from .precision import WORKING_DTYPES


def gwd_topography(mntvar: torch.Tensor) -> dict[str, torch.Tensor]:
    """Retain the selected nmtvr=24 GFS_GWD_generic_pre column layout."""
    if mntvar.ndim != 2 or mntvar.shape[1] != 24 or mntvar.dtype not in WORKING_DTYPES:
        raise ValueError("R4 GWD pre requires float32/float64 [ncol,24] topography")
    return {
        "oc": mntvar[:, 1], "oa4": mntvar[:, 2:6], "clx": mntvar[:, 6:10],
        "theta": mntvar[:, 10], "gamma": mntvar[:, 11],
        "sigma": mntvar[:, 12], "elvmax": mntvar[:, 13],
        "varss": mntvar[:, 14], "ocss": mntvar[:, 15],
        "oa4ss": mntvar[:, 16:20], "clxss": mntvar[:, 20:24],
    }


def apply_gwd_pre(prefix: PhysicsPrefixResult) -> PhysicsPrefixResult:
    return replace(prefix, interstitial={
        **prefix.interstitial, **gwd_topography(prefix.sfcprop.hprime),
    })


def orographic_gwd_inputs(statein: Statein, prefix: PhysicsPrefixResult) -> dict[str, torch.Tensor]:
    """Bind the cap's Statein ownership, not PBL Stateout, to the legacy solver.

    unified_ugwp initializes its reaction stress before gwdps. This binding does
    not overwrite PBL tendency ledgers or combine the legacy stress with GSL.
    """
    work = prefix.interstitial
    return {
        "t1": statein.tgrs, "u1": statein.ugrs, "v1": statein.vgrs,
        "q1": statein.qgrs[:, :, 0], "hprime": prefix.sfcprop.hprime[:, 0],
        **{n: getattr(statein, n) for n in ("prsi", "prsl", "prslk", "phil", "phii")},
        **{n: work[n] for n in ("del", "kpbl", "oc", "oa4", "clx", "theta", "sigma", "gamma", "elvmax")},
        "dusfc": torch.zeros_like(statein.pgr), "dvsfc": torch.zeros_like(statein.pgr),
        "ten_q": torch.zeros_like(statein.qgrs),
    }


def mountain_blocking(
    fields: Mapping[str, torch.Tensor], *, imx: int | torch.Tensor, cdmbgwd: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Port gwdps's nmtvr=14 blocking section with full-batch active masks.

    unified_ugwp passes 14 for the host's selected 24-field terrain. The means
    and dividing-streamline integral retain the source's vertical order.
    Returned level indices remain Fortran one-based, with zero meaning no hit.
    The caller supplies validated bottom-up profiles; undefined source workspace
    outside the active columns/levels is represented by zero, not read.
    """
    t, u, v, q = (fields[n] for n in ("t1", "u1", "v1", "q1"))
    phil, phii, prsl, prsi, prslk, dp = (fields[n] for n in (
        "phil", "phii", "prsl", "prsi", "prslk", "del",
    ))
    h, height, sigma, gamma, theta = (fields[n] for n in (
        "hprime", "elvmax", "sigma", "gamma", "theta",
    ))
    if t.ndim != 2 or t.shape[1] < 4 or t.dtype not in WORKING_DTYPES:
        raise ValueError("blocking requires float32/float64 [ncol,nlev>=4] and positive imx")
    if isinstance(imx, torch.Tensor):
        if imx.shape != () or imx.dtype not in (torch.int32, torch.int64) or imx.device != t.device:
            raise ValueError("GWD resolution requires a device-local integer scalar")
    elif type(imx) is not int or imx <= 0:
        raise ValueError("GWD resolution requires a positive static integer or device scalar")
    resolution = torch.as_tensor(imx, dtype=t.dtype, device=t.device)
    torch._assert_async(resolution > 0, "nonpositive GWD resolution")
    ncol, km = t.shape
    for name in ("u1", "v1", "q1", "phil", "prsl", "prslk", "del"):
        value = fields[name]
        if value.shape != t.shape or value.dtype != t.dtype or value.device != t.device:
            raise ValueError(f"invalid blocking field {name}")
    for name in ("prsi", "phii", "hprime", "elvmax", "sigma", "gamma", "theta"):
        value = fields[name]
        shape = (ncol, km + 1) if name in ("prsi", "phii") else (ncol,)
        if value.shape != shape or value.dtype != t.dtype or value.device != t.device:
            raise ValueError(f"invalid blocking field {name}")
    if cdmbgwd.shape != (4,) or cdmbgwd.dtype != t.dtype or cdmbgwd.device != t.device:
        raise ValueError("cdmbgwd requires a device-local [4] tensor in the working dtype")
    zero = torch.zeros_like(height)
    active = (height > 50.0) & (h > 1.0)
    height = torch.where(active, torch.minimum(height + 4.0 * h, height.new_full((), 8000.0)), height)
    h_safe = torch.where(active, h, 1.0)
    cdmb = (t.new_full((), 4.0 * 192.0) / resolution) * torch.where(cdmbgwd[0] >= 0.0, cdmbgwd[0], 1.0)
    cdsigohp = torch.where(active, 0.25 * cdmb * sigma / h_safe, zero)
    onebg = 1.0 / GRAV
    torch._assert_async((~active | (height < phil[:, -3] * onebg)).all(),
                        "mountain top exceeds source-defined blocking stratification")
    # iwklm is the running maximum of hit levels: one exact integer max over levels.
    level_phi = phil * onebg
    hit = active[:, None] & (height[:, None] <= level_phi[:, 1:]) & (height[:, None] >= level_phi[:, :-1])
    hit_levels = torch.arange(2, km + 1, dtype=torch.int64, device=t.device)
    iwklm = torch.where(hit, hit_levels, 2).amax(dim=1)

    vtj = t[:, :-1] * (1.0 + RVRDM1 * q[:, :-1])
    vtk = vtj / prslk[:, :-1]
    ro = (1.0 / RD) * prsl[:, :-1] / vtj
    rdz = t.new_full((), GRAV) / (phil[:, 1:-1] - phil[:, :-2])
    bnv2lm = (GRAV + GRAV) * rdz * (vtk[:, 1:] - vtk[:, :-1]) / (vtk[:, 1:] + vtk[:, :-1])
    bnv2lm = torch.clamp_min(bnv2lm, 1.0e-5)
    top_pressure = prsi.gather(1, (iwklm - 1)[:, None]).squeeze(1)
    delks = height.new_full((), 1.0) / (prsi[:, 0] - top_pressure)
    ubar, vbar, roll = zero, zero, zero
    bnv2bar = (prsi[:, 0] - prsl[:, 0]) * delks * bnv2lm[:, 0]
    # Terms for all levels at once; only the ordered sums loop.
    lower = torch.arange(1, km - 1, device=t.device)
    inside_levels = (active[:, None] & (lower < iwklm[:, None])).t().contiguous()
    weight = dp[:, :km - 2] * delks[:, None]
    u_terms = (weight * u[:, :km - 2]).t().contiguous()
    v_terms = (weight * v[:, :km - 2]).t().contiguous()
    ro_terms = (weight * ro[:, :km - 2]).t().contiguous()
    next_pressure = torch.where(lower < (iwklm - 1)[:, None], prsl[:, 1:km - 1], prsi[:, 1:km - 1])
    bnv2_terms = (bnv2lm[:, :km - 2] * ((prsl[:, :km - 2] - next_pressure) * delks[:, None])).t().contiguous()
    for k in range(km - 2):
        inside = inside_levels[k]
        ubar = torch.where(inside, ubar + u_terms[k], ubar)
        vbar = torch.where(inside, vbar + v_terms[k], vbar)
        roll = torch.where(inside, roll + ro_terms[k], roll)
        bnv2bar = torch.where(inside, bnv2bar + bnv2_terms[k], bnv2bar)

    levels = torch.arange(1, km + 1, device=t.device)[None, :]
    inside = active[:, None] & (levels <= iwklm[:, None])
    # Guard atan2 and sqrt before masking; calm inactive columns remain AD-safe.
    moving = inside & ((u != 0.0) | (v != 0.0))
    phiang = torch.atan2(torch.where(moving, v, 0.0), torch.where(moving, u, 1.0)) * (180.0 / 3.1415926535897931)
    angle = theta[:, None] - phiang
    angle = torch.where(angle > 90.0, angle - 180.0, angle)
    angle = torch.where(angle < -90.0, angle + 180.0, angle)
    angle = angle * (3.1415926535897931 / 180.0)
    angle = torch.where(inside, angle, 0.0)
    speed2 = u * u + v * v
    speed = torch.sqrt(torch.where(inside & (speed2 > 0.01), speed2, 1.0))
    speed = torch.where(speed2 > 0.01, speed, 0.1)
    uds = torch.where(inside, speed, 0.0)
    pe, ek, up = zero, zero, zero
    idxzb = torch.zeros_like(iwklm)
    # The ordered integral stops at the first PE >= EK in each active column.
    span = slice(0, km - 2)
    in_column = (active[:, None] & (lower <= iwklm[:, None])).t().contiguous()
    increment = (bnv2lm[:, span] * (GRAV * height[:, None] - phil[:, span]) * (phii[:, 1:km - 1] - phii[:, span]) * (onebg * onebg)).t().contiguous()
    up_level = (uds[:, span] * torch.cos(angle[:, span])).t().contiguous()
    ek_level = 0.5 * up_level * up_level
    for k in range(km - 3, -1, -1):
        pending = in_column[k] & (idxzb == 0)
        pe = torch.where(pending, pe + increment[k], pe)
        up = torch.where(pending, up_level[k], up)
        ek = torch.where(pending, ek_level[k], ek)
        idxzb = torch.where(pending & (pe >= ek), k + 1, idxzb)

    dividing_phi = phil.gather(1, torch.clamp_min(idxzb - 1, 0)[:, None])
    blocked = active[:, None] & (idxzb[:, None] > 0) & (levels <= idxzb[:, None]) & (dividing_phi > phil)
    numerator = torch.where(blocked, dividing_phi - phil, 1.0)
    denominator = torch.where(blocked, phil + GRAV * h[:, None], 1.0)
    zlen = torch.sqrt(numerator / denominator)
    cosang, sinang = torch.cos(angle), torch.sin(angle)
    cos2a, sin2a = cosang * cosang, sinang * sinang
    tem = cos2a + gamma[:, None] * sin2a
    factor = gamma[:, None] * cos2a + sin2a
    regular = torch.abs(tem) > 1.0e-6
    rtrm = torch.where(regular, factor / torch.where(regular, tem, 1.0),
                       torch.where(tem > 0.0, factor * 1.0e6, -factor * 1.0e6))
    zr = torch.clamp_min(2.0 - rtrm, 0.0)
    ro_full = torch.cat((ro, torch.zeros_like(ro[:, :1])), dim=1)
    db = cdsigohp[:, None] * zr * ro_full * zlen * torch.maximum(cosang, gamma[:, None] * sinang) * uds
    output = {
        "active": active, "elvmax": height, "iwklm": torch.where(active, iwklm, 0),
        "idxzb": idxzb, "rdxzb": idxzb.to(t.dtype), "cdsigohp": cdsigohp,
        "vtj": vtj, "vtk": vtk, "ro": ro, "bnv2lm": bnv2lm,
        "delks": delks, "ubar": ubar, "vbar": vbar, "roll": roll,
        "bnv2bar": bnv2bar, "pe": pe, "ek": ek, "up": up,
        "ang": angle, "uds": uds, "db": torch.where(blocked, db, 0.0),
    }
    for name in ("vtj", "vtk", "ro", "bnv2lm", "delks", "bnv2bar"):
        value = output[name]
        mask = active[:, None] if value.ndim == 2 else active
        output[name] = torch.where(mask, value, 0.0)
    return output


def gwdps(
    fields: Mapping[str, torch.Tensor], *, imx: int | torch.Tensor, cdmbgwd: torch.Tensor,
    dtp: torch.Tensor, ldiag_ugwp: bool = False,
) -> dict[str, torch.Tensor]:
    """Selected R4 legacy mesoscale wave drag plus mountain blocking.

    This solver returns tendencies and finalized reaction stress, not committed
    winds. unified_ugwp owns combination with small-scale and nonstationary drag.
    Optional diagnostics start from caller-owned arrays, as in gwdps_run.
    """
    blocking = mountain_blocking(fields, imx=imx, cdmbgwd=cdmbgwd)
    t, u, v, q = (fields[n] for n in ("t1", "u1", "v1", "q1"))
    prsi, prsl, prslk, phil, dp = (fields[n] for n in ("prsi", "prsl", "prslk", "phil", "del"))
    ncol, km = t.shape
    resolution = torch.as_tensor(imx, dtype=t.dtype, device=t.device)
    if dtp.shape != () or dtp.dtype != t.dtype or dtp.device != t.device:
        raise ValueError("GWD timestep requires a device-local scalar in the working dtype")
    if fields["kpbl"].shape != (ncol,) or fields["kpbl"].device != t.device:
        raise ValueError("invalid GWD PBL indices")
    active = blocking["active"]
    zero = torch.zeros_like(fields["hprime"])
    vtj = t * (1.0 + RVRDM1 * q)
    vtk = vtj / prslk
    ro = (1.0 / RD) * prsl / vtj
    rdz = t.new_full((), GRAV) / (phil[:, 1:] - phil[:, :-1])
    ti = t.new_full((), 2.0) / (t[:, :-1] + t[:, 1:])
    du, dv = u[:, :-1] - u[:, 1:], v[:, :-1] - v[:, 1:]
    shr2 = torch.clamp_min(du * du + dv * dv, 1.0) * rdz * rdz
    bvf2 = GRAV * (GRAV / CP + rdz * (vtj[:, 1:] - vtj[:, :-1])) * ti
    ri = torch.clamp_min(bvf2 / shr2, -100.0)
    bnv2 = (GRAV + GRAV) * rdz * (vtk[:, 1:] - vtk[:, :-1]) / (vtk[:, 1:] + vtk[:, :-1])
    bnv2 = torch.clamp_min(bnv2, 1.0e-5)
    iwk = torch.full_like(fields["kpbl"], 2, dtype=torch.int64)
    for k in range(2, km // 2):
        iwk = torch.where(prsi[:, 0] - prsi[:, k] < 5000.0, k + 1, iwk)
    kref = torch.maximum(iwk, fields["kpbl"].to(torch.int64) + 1)
    kref = torch.where(active, kref, 2)
    torch._assert_async(((kref >= 2) & (kref <= km - 1)).all(), "invalid source GWD reference level")
    delks = t.new_full((), 1.0) / (prsi[:, 0] - prsi.gather(1, (kref - 1)[:, None]).squeeze(1))
    ubar, vbar, roll = zero, zero, zero
    bnv2bar = (prsi[:, 0] - prsl[:, 0]) * delks * bnv2[:, 0]
    for k in range(km - 1):
        below = active & (k + 1 < kref)
        weight = dp[:, k] * delks
        ubar = torch.where(below, ubar + weight * u[:, k], ubar)
        vbar = torch.where(below, vbar + weight * v[:, k], vbar)
        roll = torch.where(below, roll + weight * ro[:, k], roll)
        pnext = torch.where(k + 1 < kref - 1, prsl[:, k + 1], prsi[:, k + 1])
        weight = (prsl[:, k] - pnext) * delks
        bnv2bar = torch.where(below, bnv2bar + bnv2[:, k] * weight, bnv2bar)

    moving = active & ((ubar != 0.0) | (vbar != 0.0))
    wdir = torch.atan2(torch.where(moving, ubar, 0.0), torch.where(moving, vbar, 1.0)) + math.pi
    idir = torch.remainder(torch.floor((8.0 / (math.pi + math.pi)) * wdir + 0.5).to(torch.int64), 8)
    nwdir = static_tensor((6, 7, 5, 8, 2, 3, 1, 4), device=t.device, dtype=torch.int64)
    nwd = nwdir[idir]
    terrain_index = torch.remainder(nwd - 1, 4)[:, None]
    oa = (1 - 2 * torch.div(nwd - 1, 4, rounding_mode="trunc")) * fields["oa4"].gather(1, terrain_index).squeeze(1)
    clx = fields["clx"].gather(1, terrain_index).squeeze(1)
    low2 = ubar * ubar + vbar * vbar
    ulow = torch.sqrt(torch.where(low2 > 1.0, low2, 1.0))
    uloi = t.new_full((), 1.0) / ulow
    velco = 0.5 * ((u[:, :-1] + u[:, 1:]) * ubar[:, None] + (v[:, :-1] + v[:, 1:]) * vbar[:, None])
    velco = velco * uloi[:, None]
    bnv = torch.sqrt(torch.where(active, bnv2bar, 1.0))
    fr = torch.clamp_max(bnv * uloi * torch.clamp_max(fields["hprime"], 2400.0), 100.0)
    xn, yn = ubar * uloi, vbar * uloi
    efact = torch.clamp(torch.pow(oa + 2.0, 0.8 * fr), min=0.0, max=10.0)
    coefm = torch.pow(1.0 + clx, oa + 1.0)
    cleff = (t.new_full((), 0.5e-5) / torch.sqrt(resolution / 192.0)) * torch.where(cdmbgwd[1] >= 0.0, cdmbgwd[1], 1.0)
    xlinv = coefm * cleff
    tem = fr * fr * fields["oc"]
    gfobnv = tem / ((tem + 0.5) * bnv)
    taub = xlinv * roll * ulow * ulow * ulow * gfobnv * efact
    taub = torch.where(active, taub, 0.0)
    critical = torch.zeros_like(active)
    taup = []
    incoming = taub
    one = t.new_full((), 1.0)
    span = slice(0, km - 1)
    level_numbers = torch.arange(1, km, device=t.device)
    rows = lambda x: x.t().contiguous()  # noqa: E731
    level_active_rows = rows(active[:, None] & (level_numbers >= kref[:, None]))
    below_ref_rows = rows(level_numbers <= kref[:, None])
    critical_rows = rows(active[:, None] & (level_numbers >= kref[:, None]) & ((ri[:, span] < 0.25) | (velco[:, span] <= 0.0)))
    wind_all = torch.clamp_min(velco[:, span], 0.01)
    temv_rows = rows(one / wind_all)
    brvf_all = torch.sqrt(bnv2[:, span])
    tem1_rows = rows(xlinv[:, None] * (ro[:, 1:km] + ro[:, span]) * brvf_all * 0.5 * wind_all)
    brvf_rows, ri_rows, velco_rows = rows(brvf_all), rows(ri[:, span]), rows(velco[:, span])
    for k in range(km - 1):
        current = torch.where(below_ref_rows[k], taub, incoming)
        taup.append(current)
        critical = critical | critical_rows[k]
        propagating = level_active_rows[k] & ~critical & (current > 0.0)
        brvf, tem1 = brvf_rows[k], tem1_rows[k]
        hd = torch.sqrt(torch.where(propagating, current, 1.0) / torch.where(propagating, tem1, 1.0))
        fro = brvf * hd * temv_rows[k]
        tem2 = torch.sqrt(torch.where(propagating, ri_rows[k], 1.0))
        tem = 1.0 + tem2 * fro
        rim = ri_rows[k] * (1.0 - fro) / (tem * tem)
        temc = 2.0 + one / tem2
        saturated_hd = velco_rows[k] * (2.0 * torch.sqrt(temc) - temc) / brvf
        saturated_tau = tem1 * saturated_hd * saturated_hd
        # kint == kref makes the source's downstream Scorer branch unreachable.
        next_tau = torch.where(rim <= 0.25, saturated_tau, current)
        incoming = torch.where(propagating, torch.minimum(next_tau, current), 0.0)
    taup.append(incoming)
    taup.append(prsi[:, km] / prsi[:, km - 1] * incoming)
    taup = torch.stack(taup, dim=1)
    taud = GRAV * (taup[:, 1:] - taup[:, :-1]) / dp
    taud = torch.cat((taud[:, :-1], taud[:, -1:] * 0.5), dim=1)
    dtfac = torch.ones_like(zero)
    limiting_all = active[:, None] & (level_numbers > kref[:, None]) & (prsi[:, span] >= 50000.0) & (taud[:, span] != 0.0)
    candidate_rows = rows(torch.abs(velco[:, span] / torch.where(limiting_all, dtp * taud[:, span], 1.0)))
    limiting_rows = rows(limiting_all)
    for k in range(km - 1):
        candidate = candidate_rows[k]
        # minimum's native JVP cancels small winning tangents against huge losers.
        # Equal nonnegative factors retain the symmetric tie derivative and bits.
        minimum = torch.where(candidate < dtfac, candidate,
            torch.where(candidate > dtfac, dtfac, (dtfac + candidate) * 0.5))
        dtfac = torch.where(limiting_rows[k], minimum, dtfac)
    taud = taud * dtfac[:, None]
    dtaux, dtauy = taud * xn[:, None], taud * yn[:, None]
    levels = torch.arange(1, km + 1, device=t.device)[None, :]
    blocked = active[:, None] & (levels < blocking["idxzb"][:, None])
    dbim = blocking["db"] / (1.0 + blocking["db"] * dtp)
    a = torch.where(active[:, None], torch.where(blocked, -dbim * v, dtauy), 0.0)
    b = torch.where(active[:, None], torch.where(blocked, -dbim * u, dtaux), 0.0)
    eng0 = 0.5 * (u * u + v * v)
    eng1_block = eng0 * (1.0 - dbim * dtp) * (1.0 - dbim * dtp)
    tem1, tem2 = u + dtaux * dtp, v + dtauy * dtp
    eng1_wave = 0.5 * (tem1 * tem1 + tem2 * tem2)
    eng1 = torch.where(blocked, eng1_block, eng1_wave)
    c = torch.where(active[:, None], torch.clamp_min(eng0 - eng1, 0.0) * (t.new_full((), 1.0) / (CP * dtp)), 0.0)
    dusfc, dvsfc = fields["dusfc"], fields["dvsfc"]
    diagnostics = {}
    if ldiag_ugwp:
        diagnostics = {n: fields[n] for n in ("dtaux2d_bl", "dtauy2d_bl", "dtaux2d_ms", "dtauy2d_ms",
                                             "dusfc_bl", "dvsfc_bl", "dusfc_ms", "dvsfc_ms")}
        diagnostics.update({
            "dtaux2d_bl": torch.where(blocked, -dbim * u, fields["dtaux2d_bl"]),
            "dtauy2d_bl": torch.where(blocked, -dbim * v, fields["dtauy2d_bl"]),
            "dtaux2d_ms": torch.where(active[:, None] & ~blocked, dtaux, fields["dtaux2d_ms"]),
            "dtauy2d_ms": torch.where(active[:, None] & ~blocked, dtauy, fields["dtauy2d_ms"]),
        })
    tem1 = dbim * dp
    blocked_rows = rows(blocked)
    u_block, v_block = rows((1.0 / GRAV) * tem1 * u), rows((1.0 / GRAV) * tem1 * v)
    u_wave, v_wave = rows((1.0 / GRAV) * dtaux * dp), rows((1.0 / GRAV) * dtauy * dp)
    for k in range(km):
        dusfc = torch.where(active, torch.where(blocked_rows[k], dusfc + u_block[k], dusfc - u_wave[k]), dusfc)
        dvsfc = torch.where(active, torch.where(blocked_rows[k], dvsfc + v_block[k], dvsfc - v_wave[k]), dvsfc)
        if ldiag_ugwp:
            for name, tendency, mask in (("dusfc_bl", -dbim[:, k] * u[:, k], blocked[:, k]),
                                        ("dvsfc_bl", -dbim[:, k] * v[:, k], blocked[:, k]),
                                        ("dusfc_ms", dtaux[:, k], active & ~blocked[:, k]),
                                        ("dvsfc_ms", dtauy[:, k], active & ~blocked[:, k])):
                diagnostics[name] = torch.where(mask, diagnostics[name] + tendency * dp[:, k], diagnostics[name])
    if ldiag_ugwp:
        for name in ("dusfc_bl", "dvsfc_bl", "dusfc_ms", "dvsfc_ms"):
            diagnostics[name] = torch.where(active, -(1.0 / GRAV) * diagnostics[name], diagnostics[name])
    return {"a": a, "b": b, "c": c, "ten_q": torch.zeros_like(fields["ten_q"]),
            "dusfc": dusfc, "dvsfc": dvsfc, "rdxzb": blocking["rdxzb"],
            "elvmax": blocking["elvmax"], **diagnostics}
