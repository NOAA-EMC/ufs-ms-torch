# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Model-facing surface and cloud diagnostics for the GFS v17 oracle profile.

Equations follow radiation_surface.f setalb/setemis (ialb=iems=2,
Noah-MP, uncoupled ice) and radiation_clouds.f gethml (bottom-up,
maximum-random). Tensor axes match radiation_group: [layer, column].
"""
from __future__ import annotations
import torch
from ufs_physics_common.precision import working_power


def radiation_surface(state, pre, config):
    if (config.ialb, config.iems, config.lsm,
        config.cplice, config.use_cice_alb, config.lndp_type) != (2, 2, 2, False, False, 0):
        raise ValueError('G3.4 surface supports the oracle Noah-MP uncoupled ialb=iems=2 profile only')
    cimin = torch.where(state['lakefrac'] > 0, torch.full_like(state['fice'], config.min_lakeice),
                        torch.full_like(state['fice'], config.min_seaice))
    if config.frac_grid:
        # GFS_radiation_surface frac_grid branch: land fraction from landfrac, ice on the rest
        fracl = state['landfrac']
        fraco = torch.clamp_min(1.0 - fracl, 0.0)
        icy = state['fice'] >= cimin
        fraci = torch.where(icy, fraco * state['fice'], torch.zeros_like(fracl))
        fraco = torch.clamp_min(fraco - fraci, 0.0)
    else:
        land = state['slmsk'] == 1
        fracl = land.to(pre['coszen'].dtype)
        fraci = torch.where(~land & (state['fice'] >= cimin), state['fice'], torch.zeros_like(fracl))
        fraco = torch.clamp(1.0 - fracl - fraci, min=0.0)
        icy = (~land) & (fraci > 0.0)
    asnow = 0.02 * state['snodi']
    argh = torch.clamp(0.01 * state['zorl'], 0.025, 0.50)
    hrgh = torch.clamp(1.0577 - 1.1538e-3 * state['hprime'], 0.20, 1.0)
    fsno = asnow / (argh + asnow) * hrgh
    ice_emis = torch.where(state['sncovr_ice'] > 0,
                          0.96 * (1.0 - state['sncovr_ice']) + 0.99 * state['sncovr_ice'],
                          torch.where(state['snodi'] > 0, 0.96 * (1.0 - fsno) + 0.99 * fsno,
                                      torch.full_like(fracl, 0.96)))
    semis_ice = torch.where(icy, ice_emis, state['emis_ice'])
    semis = fracl * state['emis_lnd'] + fraco * 0.97 + fraci * ice_emis
    mu = pre['coszen']
    # Scalar / tensor rounds the reciprocal first; source setalb divides directly.
    water_dir = torch.maximum(torch.full_like(mu, 0.06),
        torch.div(0.026, working_power(torch.clamp(mu, min=0.0), 1.7) + 0.065)
        + 0.15 * (mu - 0.1) * (mu - 0.5) * (mu - 1.0))
    water_dir = torch.where((fraco > 0) & (mu > 0.0001), water_dir, torch.full_like(mu, 0.06))
    puddle = (pre['tsfg'] > 271.1) & (pre['tsfg'] < 271.5)
    a1 = (pre['tsfg'] - 271.1) ** 2
    vis_dif = torch.where(puddle, 0.7 - 4.0 * a1, torch.full_like(mu, 0.70))
    nir_dif = torch.where(puddle, 0.65 - 3.6875 * a1, torch.full_like(mu, 0.65))
    b1 = 0.03 * torch.clamp(273.16 - state['tisfc'], 0.0, 5.0)
    snow_vis, snow_nir = vis_dif + b1, nir_dif + b1
    csnow = 0.5 * (torch.div(3.0, 1.0 + 4.0 * mu) - 1.0)
    snow_vis_dir = torch.where((mu > 0.0001) & (mu < 0.5),
                               torch.clamp(snow_vis + (1.0 - snow_vis) * csnow, max=0.98), snow_vis)
    snow_nir_dir = torch.where((mu > 0.0001) & (mu < 0.5),
                               torch.clamp(snow_nir + (1.0 - snow_nir) * csnow, max=0.98), snow_nir)
    ice = [nir_dif * (1.0 - fsno) + snow_nir_dir * fsno,
           nir_dif * (1.0 - fsno) + snow_nir * fsno,
           vis_dif * (1.0 - fsno) + snow_vis_dir * fsno,
           vis_dif * (1.0 - fsno) + snow_vis * fsno]
    defaults = (0.65, 0.65, 0.70, 0.70)
    land_names = ('albdirnir_lnd', 'albdifnir_lnd', 'albdirvis_lnd', 'albdifvis_lnd')
    sfcalb = torch.stack([torch.clamp(state[name], 0.01, 0.99) * fracl
                         + (water_dir if j in (0, 2) else 0.06) * fraco
                         + torch.where(icy, value, torch.full_like(mu, defaults[j])) * fraci
                         for j, (name, value) in enumerate(zip(land_names, ice))])
    return {'sfcalb': sfcalb, 'sfalb': torch.clamp(0.5 * (sfcalb[1] + sfcalb[3]), min=0.01),
            'semis': semis, 'emis_ice': semis_ice, 'emis_lnd': state['emis_lnd'],
            'emis_wat': torch.full_like(fracl, 0.97), 'semisbase': state['semisbase'],
            **{name: state[name] for name in land_names}}


def cloud_diagnostics(state, pre, cloud, config, overlap=None):
    """Port gethml's original ordered vertical recurrences and one-based indices.

    Total and boundary-layer cloud follow ``iovr`` 1 (maximum-random) or 3 (decorrelation
    length: ``alfa = exp(-0.5*(dz1 + dz)/de_lgth)`` with ``dz`` the layer thickness in m and
    ``de_lgth`` in km, as in the source).
    """
    cf = torch.clamp(cloud['cld_frac'], min=0.0, max=1.0 - 1.0e-8)
    p = pre['p_lay'] * 0.01
    nlay, ncol = p.shape
    one = torch.ones_like(p[0])
    zero = torch.zeros_like(one)
    izero = torch.zeros_like(one, dtype=torch.int64)
    ione = torch.ones_like(izero)
    # gethml searches si(2:nlay), defaults to the final layer if none falls below .9.
    indices = torch.arange(1, nlay, dtype=torch.int64, device=p.device)
    llyr = torch.amin(torch.where(state['si'][1:nlay] < 0.9, indices, nlay - 1))
    cl1, cl2, bl = one, one, zero
    cloudy_layers, clear_fraction = cf >= 0.001, 1.0 - cf
    if config.iovr == 3:
        dz, de_lgth = pre['deltaZ'], overlap['de_lgth']
        dz1 = -dz[0]
        for k in range(nlay):
            cloudy = cloudy_layers[k]
            alfa = torch.exp(-(0.5 * (dz1 + dz[k])) / de_lgth)
            mixed = alfa * torch.minimum(cl2, clear_fraction[k]) + (1.0 - alfa) * (cl2 * clear_fraction[k])
            cl1 = torch.where(cloudy, cl1, cl1 * cl2)
            cl2 = torch.where(cloudy, mixed, one)
            dz1 = torch.where(cloudy, dz[k], -dz[k + 1] if k + 1 < nlay else dz1)
            bl = torch.where(llyr == k + 1, 1.0 - cl1 * cl2, bl)
    else:
        for k in range(nlay):
            cloudy = cloudy_layers[k]
            cl1 = torch.where(cloudy, cl1, cl1 * cl2)
            cl2 = torch.where(cloudy, torch.minimum(cl2, clear_fraction[k]), one)
            bl = torch.where(llyr == k + 1, 1.0 - cl1 * cl2, bl)
    total = 1.0 - cl1 * cl2
    latfac = torch.clamp(4.0 * torch.abs(state['xlat'] / config.con_pi) - 1.0, min=0.0)
    thresholds = torch.stack((1050.0 * one, 650.0 + 100.0 * latfac, 400.0 + 100.0 * latfac, zero))
    frac = [zero, zero, zero]
    tops, bots = [ione, ione * 2, ione * 2], [ione, ione * 2, ione * 2]
    cl1, cl2, kbt1, kbt2, kth1, kth2, domain = zero, zero, ione, ione, izero, izero, izero
    for k in range(nlay):
        pnext = p[k + 1] if k + 1 < nlay else -one
        cnext = cf[k + 1] if k + 1 < nlay else zero
        threshold = torch.gather(thresholds, 0, (domain + 1).clamp(max=3).unsqueeze(0))[0]
        domain = domain + (p[k] < threshold).to(torch.int64)
        threshold = torch.gather(thresholds, 0, (domain + 1).clamp(max=3).unsqueeze(0))[0]
        cloudy = cf[k] >= 0.001
        kbt2 = torch.where(cloudy & (kth2 == 0), k + 1, kbt2)
        kth2 = kth2 + cloudy.to(torch.int64)
        cl2 = torch.where(cloudy, torch.maximum(cl2, cf[k]), cl2)
        endcloud = cloudy & ((cnext < 0.001) | (pnext < threshold))
        denom = torch.where(endcloud, cl1 + cl2, one)
        kb_new = torch.floor((cl1 * kbt1 + cl2 * kbt2) / denom + 0.5).to(torch.int64)
        kh_new = torch.floor((cl1 * kth1 + cl2 * kth2) / denom + 0.5).to(torch.int64)
        kbt1, kth1 = torch.where(endcloud, kb_new, kbt1), torch.where(endcloud, kh_new, kth1)
        cl1 = torch.where(endcloud, cl1 + cl2 - cl1 * cl2, cl1)
        kbt2, kth2 = torch.where(endcloud, k + 2, kbt2), torch.where(endcloud, 0, kth2)
        cl2 = torch.where(endcloud, zero, cl2)
        enddomain = pnext < threshold
        for d in range(3):
            write = enddomain & (domain == d)
            frac[d] = torch.where(write, cl1, frac[d])
            tops[d] = torch.where(write, torch.maximum(kbt1, kbt1 + kth1 - 1), tops[d])
            bots[d] = torch.where(write, kbt1, bots[d])
        cl1 = torch.where(enddomain, zero, cl1)
        kbt1, kth1 = torch.where(enddomain, min(k + 2, nlay), kbt1), torch.where(enddomain, 0, kth1)
        for d in range(1, 3):
            write = enddomain & (domain + 1 == d)
            tops[d], bots[d] = torch.where(write, kbt1, tops[d]), torch.where(write, kbt1, bots[d])
    return {'cldsa': torch.stack((*frac, total, bl)), 'mtopa': torch.stack(tops), 'mbota': torch.stack(bots)}


def radiation_fluxr(state, pre, sw, lw, post, diag, config):
    slots = list(state['fluxr_prev'].unbind(0))
    if not config.lssav:
        return state['fluxr_prev']
    for slot, field in ((1, 'topflw_upfxc'), (19, 'sfcflw_dnfxc'), (20, 'sfcflw_upfxc'),
                        (28, 'topflw_upfx0'), (30, 'sfcflw_dnfx0'), (33, 'sfcflw_upfx0')):
        slots[slot-1] = slots[slot-1] + config.fhlwr * post[field]
    mu, dg = pre['coszen'], pre['coszdg']
    w = config.fhswr * dg / torch.where(mu > 0, mu, torch.ones_like(mu))
    for slot, field in ((2, 'topfsw_upfxc'), (3, 'sfcfsw_upfxc'), (4, 'sfcfsw_dnfxc'),
                        (23, 'topfsw_dnfxc'), (29, 'topfsw_upfx0'), (31, 'sfcfsw_upfx0'), (32, 'sfcfsw_dnfx0')):
        slots[slot-1] = torch.where(mu > 0, slots[slot-1] + post[field] * w, slots[slot-1])
    for slot, field in ((21, 'uvbfc'), (22, 'uvbf0'), (24, 'visbm'), (25, 'visdf'), (26, 'nirbm'), (27, 'nirdf')):
        slots[slot-1] = torch.where(mu > 0, slots[slot-1] + sw['scmpsw_'+field] * w, slots[slot-1])
    aerodp = post.get('aerodp')
    for slot in range(34, 40):  # aerosol optical depth at 550 nm (zero with iaerflg = 0)
        slots[slot-1] = torch.zeros_like(mu) if aerodp is None else aerodp[slot - 34]
    for slot, j in ((17, 3), (18, 4)):
        slots[slot-1] = slots[slot-1] + pre['raddt'] * diag['cldsa'][j]
    for j in range(3):
        wt = pre['raddt'] * diag['cldsa'][j]
        itop, ibot = diag['mtopa'][j]-config.kd-1, diag['mbota'][j]-config.kd-1
        top_p = torch.gather(pre['p_lev'], 0, (itop+config.kt).unsqueeze(0))[0]
        bot_p = torch.gather(pre['p_lev'], 0, (ibot+config.kb).unsqueeze(0))[0]
        top_t = torch.gather(state['gt0'], 0, itop.unsqueeze(0))[0]
        for slot, val in ((7-j, wt), (10-j, wt*top_p), (13-j, wt*bot_p), (16-j, wt*top_t)):
            slots[slot-1] = slots[slot-1] + val
        tau_sw, tau_lw = torch.zeros_like(mu), torch.zeros_like(mu)
        # Masked layer terms at once; the ordered layer sums stay sequential.
        layers = torch.arange(pre['p_lay'].shape[0], device=mu.device)[:, None]
        mask = (layers >= ibot) & (layers <= itop)
        sw_terms = torch.where(mask, sw['cldtausw'], 0.0)
        lw_terms = torch.where(mask, lw['cldtaulw'], 0.0)
        for k in range(pre['p_lay'].shape[0]):
            tau_sw = tau_sw + sw_terms[k]
            tau_lw = tau_lw + lw_terms[k]
        slots[41-j] = torch.where(torch.any(pre['day']), slots[41-j] + wt * tau_sw, slots[41-j])
        slots[44-j] = slots[44-j] + wt * (1.0-torch.exp(-tau_lw))
    return torch.stack(slots)
