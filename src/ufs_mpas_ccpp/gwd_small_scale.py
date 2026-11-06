# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Selected R4 PSL small-scale gravity-wave drag and turbulent form drag."""

import math
from collections.abc import Mapping

import torch
from gfs_surface_layer_torch.driver import CP, GRAV, RD, RVRDM1
from sa_tke_edmf_torch.diagnostics import add_diagnostic_

from .containers import Statein, static_tensor
from .grid import Grid
from .physics_prefix import PhysicsPrefixResult
from .precision import WORKING_DTYPES


def small_scale_gwd_inputs(
    statein: Statein, grid: Grid, prefix: PhysicsPrefixResult,
    legacy: Mapping[str, torch.Tensor], hpbl: torch.Tensor, *,
    dtp: torch.Tensor, control: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Bind PSL after legacy drag without reinitializing its updated elevation.

    Sequential updating, when selected, applies legacy drag once to private
    winds. The next solver always consumes legacy's inout elvmax.
    """
    work = prefix.interstitial
    u, v = statein.ugrs, statein.vgrs
    if control["ugwp_seq_update"] and (control["do_gsl_drag_ss"] or control["do_gsl_drag_tofd"]):
        u = u + legacy["b"] * dtp
        v = v + legacy["a"] * dtp
    return {
        "u1": u, "v1": v, "t1": statein.tgrs,
        "q1": statein.qgrs[:, :, 0], "hpbl": hpbl, "br1": work["rb"],
        "slmsk": prefix.sfcprop.slmsk, "vtype": prefix.sfcprop.vtype,
        "dx": grid.dx, "var": prefix.sfcprop.hprime[:, 0],
        "elvmax": legacy["elvmax"],
        **{n: getattr(statein, n) for n in ("prsi", "prsl", "prslk", "phil", "phii")},
        **{n: work[n] for n in ("del", "kpbl", "varss", "oa4ss", "clxss")},
        "dtend": prefix.carry.intdiag["dtend"],
    }


def drag_suite_psl(
    fields: Mapping[str, torch.Tensor], *, dtp: torch.Tensor,
    alpha_fd: torch.Tensor, control: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Port drag_suite_psl: SS, then TOFD, then large-scale GWD plus blocking (LS/BL).

    R4 (unified_ugwp) runs SS/TOFD after legacy gwdps; GFS v17 (ugwpv1_gsldrag) runs
    LS/BL and TOFD here. The source's gsd_diss_ht_opt=0 leaves temperature tendency at zero.
    """
    t, u, v, q = (fields[n] for n in ("t1", "u1", "v1", "q1"))
    prsi, prsl, prslk, phil, dp = (fields[n] for n in ("prsi", "prsl", "prslk", "phil", "del"))
    if t.ndim != 2 or t.dtype not in WORKING_DTYPES or t.shape[1] < 4:
        raise ValueError("PSL requires float32/float64 [ncol,nlev>=4]")
    ncol, km = t.shape
    for name in ("u1", "v1", "q1", "prsl", "prslk", "phil", "del"):
        value = fields[name]
        if value.shape != t.shape or value.dtype != t.dtype or value.device != t.device:
            raise ValueError(f"invalid PSL field {name}")
    for name, value in (("dtp", dtp), ("alpha_fd", alpha_fd)):
        if value.shape != () or value.dtype != t.dtype or value.device != t.device:
            raise ValueError(f"PSL {name} requires a device-local scalar in the working dtype")
    kpbl = fields["kpbl"].to(torch.int64)
    torch._assert_async(((kpbl >= 1) & (kpbl <= km)).all(), "invalid PSL PBL index")
    vtj = t * (1.0 + RVRDM1 * q)
    ro = (1.0 / RD) * prsl / vtj
    zl = phil * (1.0 / GRAV)
    land = (fields["slmsk"] == 1.0) | (fields["slmsk"] == 2.0)
    var, varss = fields["var"], fields["varss"]
    if control["spp_gwd"] == 1:
        weight = fields["spp_wts_gwd"][:, 0]
        var = var + var * 0.75 * weight
        varss = varss + varss * 0.75 * weight
    zlowtop = torch.where(fields["vtype"] == 15, var, 2.0 * var)
    # klowtop/komax keep the first level that meets the test: argmax returns the first hit.
    def first_level(hit):
        return torch.where(hit.any(dim=1), hit.to(torch.int8).argmax(dim=1).to(kpbl.dtype) + 3, 0)
    klowtop = first_level(zl[:, 1:] >= zlowtop[:, None])
    komax = first_level(zl[:, 1:] >= fields["elvmax"][:, None])
    kbl = torch.clamp(torch.maximum(torch.maximum(komax, klowtop), kpbl), min=2, max=km // 2)
    delks = t.new_full((), 1.0) / (prsi[:, 0] - prsi.gather(1, (kbl - 1)[:, None]).squeeze(1))
    ubar, vbar = torch.zeros_like(var), torch.zeros_like(var)
    half = km // 2
    weight = dp[:, :half] * delks[:, None]
    below_rows = (torch.arange(1, half + 1, device=t.device) < kbl[:, None]).t().contiguous()
    u_terms, v_terms = (weight * u[:, :half]).t().contiguous(), (weight * v[:, :half]).t().contiguous()
    for k in range(half):
        ubar = torch.where(below_rows[k], ubar + u_terms[k], ubar)
        vbar = torch.where(below_rows[k], vbar + v_terms[k], vbar)
    moving = (ubar != 0.0) | (vbar != 0.0)
    wdir = torch.atan2(torch.where(moving, ubar, 0.0), torch.where(moving, vbar, 1.0)) + math.pi
    idir = torch.remainder(torch.floor((8.0 / (2.0 * math.pi)) * wdir + 0.5).to(torch.int64), 8)
    nwd = static_tensor((6, 7, 5, 8, 2, 3, 1, 4), device=t.device, dtype=torch.int64)[idir]
    index = torch.remainder(nwd - 1, 4)[:, None]
    oass = (1 - 2 * torch.div(nwd - 1, 4, rounding_mode="trunc")) * fields["oa4ss"].gather(1, index).squeeze(1)
    olss = fields["clxss"].gather(1, index).squeeze(1)
    dx = fields["dx"]
    cleff_ss = 0.1 * torch.clamp_min(torch.sqrt(dx * dx + dx * dx), 12000.0)
    large = None
    if control["do_gsl_drag_ls_bl"]:
        large = _large_scale_and_blocking(fields, t=t, u=u, v=v, var=var, zl=zl, ro=ro, vtj=vtj,
            land=land, kbl=kbl, zlowtop=zlowtop, delks=delks, ubar=ubar, vbar=vbar, nwd=nwd, dtp=dtp,
            control=control)
    wave_u, wave_v = torch.zeros_like(t), torch.zeros_like(t)
    form_u, form_v = torch.zeros_like(t), torch.zeros_like(t)
    if control["do_gsl_drag_ss"]:
        thvx = (t / prslk) * (1.0 + RVRDM1 * q)
        hpbl2, kpbl2 = fields["hpbl"] + 10.0, kpbl
        found = torch.zeros_like(land)
        for k in range(1, km):
            hit = ~found & (k + 1 <= torch.clamp_min(kpbl, 2)) & (zl[:, k] > 300.0)
            kpbl2 = torch.where(hit, k + 1, kpbl2)
            hpbl2 = torch.where(hit & (k + 1 != kpbl), zl[:, k] + 10.0, hpbl2)
            found = found | hit
        top = thvx.gather(1, (kpbl2 - 1)[:, None]).squeeze(1)
        bottom = thvx[:, 0]
        stable = land & (2.0 * varss <= fields["hpbl"]) & (fields["br1"] > 0.0) & (top - bottom > 0.0)
        xlinv = torch.pow(1.0 + olss, oass + 1.0) / cleff_ss
        govrth = t.new_full((), GRAV) / (0.5 * (top + bottom))
        argument = govrth * (top - bottom) / hpbl2
        xnbv = torch.sqrt(torch.where(stable, argument, 1.0))
        coefficient = 0.5 * xnbv * xlinv * (2.0 * varss).square() * ro[:, 0]
        taper = torch.clamp_min(1.0 - zl / hpbl2[:, None], 0.0)
        in_pbl = torch.arange(1, km + 1, device=t.device)[None, :] <= kpbl[:, None]
        waves = []
        for wind in (u, v):
            top_wind = wind.gather(1, (kpbl2 - 1)[:, None]).squeeze(1)
            ratio = torch.abs(xnbv / torch.where(top_wind != 0.0, top_wind, 1.0))
            excited = stable & ((ratio > xlinv) | (top_wind == 0.0))
            stress = torch.where(excited, coefficient * wind[:, 0] / (1.0 + coefficient * dtp), 0.0)
            # Every stable PBL level gets -1*tau*2*taper/hpbl2: a signed zero where tau = 0.
            tendency = -1.0 * stress[:, None] * 2.0 * taper / hpbl2[:, None]
            waves.append(torch.where(stable[:, None] & in_pbl, tendency, 0.0))
        wave_u, wave_v = waves
    if control["do_gsl_drag_tofd"]:
        a1 = 0.00026615161 * varss.square()
        a2 = a1 * 0.005363
        safe_z = torch.where(land[:, None], zl, 1.0)
        factor = alpha_fd * 6.325e-3 * torch.exp(-torch.pow(safe_z / 1500.0, t.new_full((), 1.5))) * a2[:, None]
        factor = factor * torch.pow(safe_z, t.new_full((), -1.2))
        speed2 = u.square() + v.square()
        nonzero = land[:, None] & (speed2 > 0.0)
        speed = torch.where(nonzero, torch.sqrt(torch.where(nonzero, speed2, 1.0)), 0.0)
        form_u = torch.where(land[:, None], -factor * speed * u / (1.0 + factor * dtp * speed), 0.0)
        form_v = torch.where(land[:, None], -factor * speed * v / (1.0 + factor * dtp * speed), 0.0)
    dudt = (torch.zeros_like(t) + wave_u) + form_u
    dvdt = (torch.zeros_like(t) + wave_v) + form_v
    if large is not None:
        # dudt = dtaux + dtauxb + dudt, in the tapered columns only.
        lsm = large["lsm"][:, None]
        dudt = torch.where(lsm, (large["dtaux"] + large["dtauxb"]) + dudt, dudt)
        dvdt = torch.where(lsm, (large["dtauy"] + large["dtauyb"]) + dvdt, dvdt)
    def integrate(tendency):
        result = torch.zeros_like(var)
        terms = (tendency * dp).t().contiguous()
        for k in range(km):
            result = result + terms[k]
        return result
    # The source accumulates the entire SS column before the TOFD column.
    dusfc, dvsfc = integrate(wave_u), integrate(wave_v)
    for k in range(km):
        dusfc = dusfc + form_u[:, k] * dp[:, k]
        dvsfc = dvsfc + form_v[:, k] * dp[:, k]
    rdxzb = torch.zeros_like(var)
    if large is not None:
        # The LS/BL column follows the TOFD column: taud_ls*xn*del, then taud_bl*xn*del, per level.
        lsm = large["lsm"]
        ls_u, bl_u = (large["dtaux"] * dp).t().contiguous(), (large["dtauxb"] * dp).t().contiguous()
        ls_v, bl_v = (large["dtauy"] * dp).t().contiguous(), (large["dtauyb"] * dp).t().contiguous()
        for k in range(km):
            dusfc = torch.where(lsm, (dusfc + ls_u[k]) + bl_u[k], dusfc)
            dvsfc = torch.where(lsm, (dvsfc + ls_v[k]) + bl_v[k], dvsfc)
        rdxzb = large["rdxzb"]
    result = {"dudt": dudt, "dvdt": dvdt, "dtdt": torch.zeros_like(t),
              "dusfc": -(1.0 / GRAV) * dusfc, "dvsfc": -(1.0 / GRAV) * dvsfc,
              "rdxzb": rdxzb}
    if "dtend" in fields:
        dtend = fields["dtend"]
        if control["ldiag3d"] and not control["flag_for_gwd_generic_tend"]:
            dtend = dtend.clone()
            process = control["index_of_process_orographic_gwd"] - 1
            for name, wave, form in (("index_of_x_wind", wave_u, form_u), ("index_of_y_wind", wave_v, form_v)):
                slot = fields["dtidx"][control[name] - 1, process]
                torch._assert_async(slot <= dtend.shape[2], "PSL diagnostic slot outside allocation")
                if control["do_gsl_drag_ss"]:
                    add_diagnostic_(dtend, wave * dtp, slot)
                if control["do_gsl_drag_tofd"]:
                    add_diagnostic_(dtend, form * dtp, slot)
                if large is not None:
                    ls, bl = (large["dtaux"], large["dtauxb"]) if name == "index_of_x_wind" else \
                        (large["dtauy"], large["dtauyb"])
                    add_diagnostic_(dtend, torch.where(large["lsm"][:, None], (ls + bl) * dtp, 0.0), slot)
        result["dtend"] = dtend
    elif control["ldiag3d"] and not control["flag_for_gwd_generic_tend"]:
        raise ValueError("active PSL process diagnostics require dtend")
    if control["ldiag_ugwp"]:
        zero = torch.zeros_like(t)
        lsm = large["lsm"][:, None] if large is not None else None
        for tag, x, y in (("ls", *((torch.where(lsm, large[n], 0.0) for n in ("dtaux", "dtauy"))
                                   if large is not None else (zero, zero))),
                          ("bl", *((torch.where(lsm, large[n], 0.0) for n in ("dtauxb", "dtauyb"))
                                   if large is not None else (zero, zero))),
                          ("ss", wave_u, wave_v), ("fd", form_u, form_v)):
            result["dtaux2d_" + tag], result["dtauy2d_" + tag] = x, y
            result["dusfc_" + tag] = -(1.0 / GRAV) * integrate(x)
            result["dvsfc_" + tag] = -(1.0 / GRAV) * integrate(y)
    return result


def _first_hit(hit: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """One-based level of the first True along dim 1 (levels holds those level numbers), else 0."""
    return torch.where(hit.any(dim=1), levels[hit.to(torch.int8).argmax(dim=1)], 0)


def _minimum(current: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """min(current, candidate) with the symmetric tie derivative (gwdps limiter form)."""
    return torch.where(candidate < current, candidate,
                       torch.where(candidate > current, current, (current + candidate) * 0.5))


def _large_scale_and_blocking(fields, *, t, u, v, var, zl, ro, vtj, land, kbl, zlowtop, delks,
                              ubar, vbar, nwd, dtp, control):
    """drag_suite_psl large-scale GWD (Kim-Arakawa) and flow blocking (Kim-Doyle), gwd_opt_ls=bl=1.

    Returns the per-level tendencies dtaux/dtauy (LS) and dtauxb/dtauyb (BL), the tapered-column
    mask and the blocking level. Every column-dependent branch is a mask; recurrences run over
    levels in source order. rstoch = 0 (no SPP pattern on the drag) and lcap = km.
    """
    prsi, prsl, prslk, dp = (fields[n] for n in ("prsi", "prsl", "prslk", "del"))
    ncol, km = t.shape
    kpblmax = km // 2
    dev = t.device
    g = GRAV
    one = t.new_full((), 1.0)
    dx = fields["dx"]
    # Scale-aware taper of the large-scale components (dxmin_ls = 3 km, dxmax_ls = 13 km).
    taper = torch.where(dx >= 13000.0, 1.0, torch.where(dx <= 3000.0, 0.0,
        0.5 * (torch.sin(math.pi * (dx - 0.5 * (13000.0 + 3000.0)) / (13000.0 - 3000.0)) + 1.0)))
    lsm = taper > 1.0e-2
    # Orographic statistics in the low-level wind direction.
    index = torch.remainder(nwd - 1, 4)[:, None]
    sign = 1 - 2 * torch.div(nwd - 1, 4, rounding_mode="trunc")
    oa = sign * fields["oa4"].gather(1, index).squeeze(1)
    ol = torch.clamp_min(fields["clx"].gather(1, index).squeeze(1), 1.0e-5)
    oc = torch.clamp(fields["oc"], min=1.0, max=10.0)
    ol4p = fields["clx"].index_select(1, static_tensor((1, 0, 3, 2), dtype=torch.int64, device=dev))
    olp = torch.clamp_min(ol4p.gather(1, index).squeeze(1), 1.0e-5)
    od = torch.clamp(torch.clamp_max(olp / ol, 10.0), min=0.1)
    diagonal = torch.sqrt(dx * dx + dx * dx)
    along = torch.where(index.squeeze(1) >= 2, diagonal, dx)  # dxy4(dir) = dxy4p(dir) for delx = dely
    cleff = control["psl_gwd_dx_factor"] * (dx + dx) * 0.5
    # Reference and blocking heights.
    levels = torch.arange(1, km + 1, device=dev)
    kbmax = _first_hit(zl[:, 1:] >= (fields["elvmax"] + zlowtop)[:, None], levels[1:] + 1)
    hmax = torch.maximum(fields["elvmax"], zlowtop)
    href = torch.maximum(hmax, fields["hpbl"])
    delks1 = one / (prsl[:, 0] - prsl.gather(1, (kbl - 1)[:, None]).squeeze(1))
    roll = torch.zeros_like(var)
    weight_rows = (dp[:, :kpblmax] * delks[:, None] * ro[:, :kpblmax]).t().contiguous()
    below = (levels[:kpblmax][:, None] < kbl[None, :])
    for k in range(kpblmax):
        roll = torch.where(below[k], roll + weight_rows[k], roll)
    # Richardson number and squared Brunt-Vaisala frequency between levels k and k+1 (k = 1..km-1).
    vtk = vtj / prslk
    ti = t.new_full((), 2.0) / (t[:, :-1] + t[:, 1:])
    rdz = one / (zl[:, 1:] - zl[:, :-1])
    du, dv = u[:, :-1] - u[:, 1:], v[:, :-1] - v[:, 1:]
    shr2 = torch.clamp_min(du * du + dv * dv, 1.0) * rdz * rdz
    bvf2 = g * (g / CP + rdz * (vtj[:, 1:] - vtj[:, :-1])) * ti
    usqj = torch.clamp_min(bvf2 / shr2, -100.0)
    bnv2 = torch.clamp_min(2.0 * g * rdz * (vtk[:, 1:] - vtk[:, :-1]) / (vtk[:, 1:] + vtk[:, :-1]), 1.0e-5)
    low2 = ubar * ubar + vbar * vbar
    ulow = torch.sqrt(torch.where(low2 > 1.0, low2, 1.0))  # max(sqrt(low2), 1) exactly
    rulow = one / ulow
    velco = 0.5 * ((u[:, :-1] + u[:, 1:]) * ubar[:, None] + (v[:, :-1] + v[:, 1:]) * vbar[:, None])
    velco = velco * rulow[:, None]
    velco = torch.where((velco < 1.0) & (velco > 0.0), 1.0, velco)
    under = levels[None, :km - 1] < kbl[:, None]
    ldrag = (href <= 50.0) | (velco[:, 0] <= 0.0)
    ldrag = ldrag | ((levels[None, :km - 1] >= 2) & (levels[None, :km - 1] <= kpblmax) & under
                     & (velco <= 0.0)).any(dim=1)
    ldrag = ldrag | ((levels[None, :km - 1] <= kpblmax) & under & (bnv2 < 0.0)).any(dim=1)
    wtkbj = (prsl[:, 0] - prsl[:, 1]) * delks1
    bnv2_low, usqj_low = wtkbj * bnv2[:, 0], wtkbj * usqj[:, 0]
    for k in range(2, kpblmax + 1):
        inside = k < kbl
        rdelks = (prsl[:, k - 1] - prsl[:, k]) * delks1
        bnv2_low = torch.where(inside, bnv2_low + bnv2[:, k - 1] * rdelks, bnv2_low)
        usqj_low = torch.where(inside, usqj_low + usqj[:, k - 1] * rdelks, usqj_low)
    ldrag = ldrag | (bnv2_low <= 0.0) | (ulow == 1.0) | (var <= 0.0) | ~land
    # Base-level stress taub (drag columns only).
    drag = lsm & ~ldrag
    bnv = torch.sqrt(torch.where(drag, bnv2_low, 1.0))
    fr = torch.clamp_max(bnv * rulow * 2.0 * var * od, 10.0)
    xn = torch.where(drag, ubar * rulow, 0.0)
    yn = torch.where(drag, vbar * rulow, 0.0)
    efact = torch.clamp(torch.pow(oa + 2.0, 0.8 * fr / 1.0), min=0.0, max=10.0)
    coefm = torch.pow(1.0 + ol, oa + 1.0)
    xlinv = coefm / cleff
    tem = fr * fr * oc
    gfobnv = 1.0 * tem / ((tem + 1.0) * bnv)
    taub = torch.where(drag, xlinv * roll * ulow * ulow * ulow * gfobnv * efact, 0.0)
    # Vertical structure of the large-scale stress: saturation (Lindzen) above kbl.
    taup = [torch.where((k <= kbl) & (k <= kpblmax), taub, 0.0) for k in range(1, km + 2)]
    taup[km] = torch.zeros_like(var)
    icrilv = torch.zeros_like(lsm)
    rows = lambda x: x.t().contiguous()  # noqa: E731
    usqj_rows, velco_rows, bnv2_rows = rows(usqj), rows(velco), rows(bnv2)
    density_rows = rows(ro[:, 1:] + ro[:, :-1])
    scale = coefm / along
    for k in range(2, km):
        above = k >= kbl
        uk, vk = usqj_rows[k - 1], velco_rows[k - 1]
        icrilv = torch.where(above, icrilv | (uk < 0.25) | (vk <= 0.0), icrilv)
        brvf = torch.sqrt(torch.clamp_min(bnv2_rows[k - 1], 1.0e-5))
        current = taup[k - 1]
        active = lsm & above & ~ldrag & ~icrilv & (current > 0.0)
        safe_v = torch.where(active, vk, 1.0)
        temv = one / safe_v
        tem1 = torch.where(active, scale * density_rows[k - 1] * brvf * vk * 0.5, 1.0)
        hd = torch.sqrt(torch.where(active, current, 1.0) / tem1)
        fro = brvf * hd * temv
        tem2 = torch.sqrt(torch.where(active, uk, 1.0))
        tem = 1.0 + tem2 * fro
        rim = uk * (1.0 - fro) / (tem * tem)
        temc = 2.0 + one / tem2
        hd_sat = safe_v * (2.0 * torch.sqrt(temc) - temc) / brvf
        saturated = tem1 * hd_sat * hd_sat
        taup[k] = torch.where(active, torch.where(rim <= 0.25, saturated, current), taup[k])
    # Flow blocking: dividing-streamline level kblk where potential energy exceeds kinetic energy.
    # kbmax = km + 1 (blocking top above the model top) reads past the source array; not reached.
    torch._assert_async((~lsm | (kbmax <= km)).all(), "PSL blocking launch level above the model top")
    zl_bmax = zl.gather(1, torch.clamp(kbmax - 1, min=0, max=km - 1)[:, None]).squeeze(1)
    pe = torch.zeros_like(var)
    kblk = torch.zeros_like(kbl)
    zblk = torch.zeros_like(var)
    pending = drag.clone()
    energy_rows = rows(bnv2 * (zl_bmax[:, None] - zl[:, :-1]) * dp[:, :-1] * (1.0 / g)
                       / ro[:, :-1])
    ke_rows = rows(0.5 * (u * u + v * v))
    zl_rows = rows(zl)
    for k in range(km, 1, -1):
        step = pending & (k <= kbmax)
        # bnv2(km) is never set by the source (zero): its potential-energy term is zero.
        increment = energy_rows[k - 1] if k < km else torch.zeros_like(pe)
        pe = torch.where(step, pe + increment, pe)
        hit = step & (pe >= ke_rows[k - 1]) & (zl_rows[k - 1] <= hmax)
        kblk = torch.where(hit, k, kblk)
        zblk = torch.where(hit, zl_rows[k - 1], zblk)
        pending = pending & ~hit
    blocked = drag & (kblk > 0)
    cd = torch.clamp_min(2.0 - one / od, 0.0)
    span = torch.clamp_min(along, 13000.0)
    taufb_low = torch.where(blocked, 0.5 * roll * coefm / (span * span) * cd * along * olp * zblk
                            * (ulow * ulow), 0.0)
    tautem = taufb_low / torch.where(blocked, kblk - 1, 1).to(t.dtype)
    taufb = [taufb_low] + [torch.zeros_like(var) for _ in range(km)]
    for k in range(2, kpblmax + 1):
        taufb[k - 1] = torch.where(blocked & (k <= kblk), taufb[k - 2] - tautem, taufb[k - 1])
    taup_block = torch.stack(taup, dim=1).gather(1, torch.clamp_min(kblk - 1, 0)[:, None]).squeeze(1)
    for k in range(1, kpblmax + 1):
        taup[k - 1] = torch.where(blocked & (k <= kblk), taup_block, taup[k - 1])
    # Deceleration -g d(tau)/dp, top half-limit, then the critical-line limiter dtfac.
    taup, taufb = torch.stack(taup, dim=1), torch.stack(taufb, dim=1)
    taud_ls = (taup[:, 1:] - taup[:, :-1]) * g / dp
    taud_bl = (taufb[:, 1:] - taufb[:, :-1]) * g / dp
    taud_ls = torch.cat((taud_ls[:, :-1], taud_ls[:, -1:] * 0.5), dim=1)
    taud_bl = torch.cat((taud_bl[:, :-1], taud_bl[:, -1:] * 0.5), dim=1)
    dtfac = torch.ones_like(var)
    lower = torch.ones_like(lsm)
    total = taud_ls + taud_bl
    for k in range(1, kpblmax):
        lower = lower & (prsi[:, k - 1] >= 0.5 * prsi[:, 0])
        limit = lsm & lower & (total[:, k - 1] != 0.0)
        candidate = torch.abs(velco[:, k - 1] / (dtp * torch.where(limit, total[:, k - 1], 1.0)))
        dtfac = torch.where(limit, _minimum(dtfac, candidate), dtfac)
    meso_velco = torch.cat((velco, velco[:, -1:]), dim=1)
    mesosphere = (prsl <= 70.0) & (taud_ls != 0.0)
    meso = torch.where(mesosphere, _minimum(torch.ones_like(t), 0.5 * torch.abs(
        meso_velco / (dtp * torch.where(mesosphere, taud_ls, 1.0)))), 1.0)
    taud_ls = taud_ls * dtfac[:, None] * meso * taper[:, None] * (1.0 - 0.0)
    taud_bl = taud_bl * dtfac[:, None] * taper[:, None] * (1.0 - 0.0)
    return {"lsm": lsm, "dtaux": taud_ls * xn[:, None], "dtauy": taud_ls * yn[:, None],
            "dtauxb": taud_bl * xn[:, None], "dtauyb": taud_bl * yn[:, None],
            "rdxzb": torch.where(blocked & lsm, kblk.to(t.dtype), 0.0)}
