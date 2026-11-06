# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Bounded pure-tensor recurrence chunks; ordered scalar carries stay live.

Chunk inputs are immutable slices. Results are assembled once by the caller,
so no saved view can be invalidated by a later level update.
"""
from __future__ import annotations

import torch
from ufs_physics_common.ordered_sum import masked_sum, parallel_masked_sums


def _divide(mask, numerator, denominator, other):
    return torch.where(mask, numerator / torch.where(mask, denominator, 1.0), other)


def _sqrt(argument):
    positive = argument > 0.0
    return torch.where(positive, torch.sqrt(torch.where(positive, argument, 1.0)), 0.0)


def _levels(first, width):
    # ``first + offset`` for every chunk offset; small integers are exact in first's dtype.
    return (first + torch.arange(width, dtype=first.dtype, device=first.device))[:, None]


def pack_profiles(*values):
    """One contiguous [level, profile, column, ...] packet per ordered pass."""
    return torch.stack(tuple(value.movedim(1, 0) for value in values), dim=1)


def _stack(values):
    # Levels before columns, then a column-leading view: with the caller's slice_scatter Inductor
    # writes the chunk in one kernel (a dim-1 stack cost one more kernel and 34% more bytes in a
    # 55-level mimic). Values are unchanged.
    return torch.stack(values, dim=0).movedim(0, 1)


def ordered_pair_sums(mask, terms, first_sum, second_sum, *, masked_add):
    """Two ordered sums over contiguous level rows; only column carries remain live."""
    if parallel_masked_sums():
        return masked_sum(first_sum, mask, terms[:, 0]), masked_sum(second_sum, mask, terms[:, 1])
    for offset in range(mask.shape[0]):
        active = mask[offset]
        if masked_add:
            first_sum = first_sum + torch.where(active, terms[offset, 0], 0.0)
            second_sum = second_sum + torch.where(active, terms[offset, 1], 0.0)
        else:
            first_sum = torch.where(active, first_sum + terms[offset, 0], first_sum)
            second_sum = torch.where(active, second_sum + terms[offset, 1], second_sum)
    return first_sum, second_sum


def eta_lower(cnv, kb, kbcon, first, profiles, xlamud, carry):
    # Carry-independent terms for all levels of the chunk; only the recurrence loops.
    width = profiles.shape[0] - 1
    level = _levels(first, width)
    cond = cnv & (level < kbcon) & (level >= kb)
    z, lam, old = (profiles[:, i] for i in range(3))
    dz = z[1:] - z[:-1]
    ptem = 0.5 * (lam[:-1] + lam[1:]) - xlamud
    denominator = torch.where(cond, 1.0 + ptem * dz, 1.0)
    old_rows = old[:-1]
    values = [None] * width
    for offset in range(width - 1, -1, -1):
        carry = torch.where(cond[offset], carry / denominator[offset], old_rows[offset])
        values[offset] = carry
    return _stack(values)


def eta_upper(flg, kbcon, kmax, kbm, first, profiles, xlamud, carry):
    width = profiles.shape[0] - 1
    level = _levels(first, width)
    # kmax changes only where flg turns false, so the level window is fixed by the input kmax.
    window = (level > kbcon) & (level < kmax)
    z, lam, old = (profiles[:, i] for i in range(3))
    dz = z[1:] - z[:-1]
    ptem = 0.5 * (lam[1:] + lam[:-1]) - xlamud
    factor = 1.0 + ptem * dz
    old_rows = old[1:]
    values = []
    for offset in range(width):
        cond = flg & window[offset]
        candidate = carry * factor[offset]
        carry = torch.where(cond, candidate, old_rows[offset])
        bad = cond & (candidate <= 0.0)
        kmax = torch.where(bad, level[offset, 0], kmax)
        kbm = torch.where(bad, torch.minimum(kbm, kmax), kbm)
        flg = torch.where(bad, False, flg)
        values.append(carry)
    return _stack(values), kmax, kbm, flg


def thermo(cnv, kb, kmax, first, profiles, xlamud, carry, *, cm, pgcon):
    width = profiles.shape[0] - 1
    level = _levels(first, width)
    cond = cnv & (level > kb) & (level < kmax)
    z, lam, h, hs, u, v = (profiles[:, i] for i in range(6))
    hs = hs[1:]
    old_rows = tuple(profiles[1:, i] for i in range(6, 10))
    dz = z[1:] - z[:-1]
    tem = 0.5 * (lam[1:] + lam[:-1]) * dz
    tem1 = 0.5 * xlamud * dz
    heat_keep = 1.0 - tem1
    heat_source = tem * 0.5 * (h[1:] + h[:-1])
    heat_denominator = torch.where(cond, 1.0 + tem - tem1, 1.0)
    tem_m = 0.5 * cm * tem
    ptem, ptem1 = tem_m + pgcon, tem_m - pgcon
    momentum_keep = 1.0 - tem_m
    momentum_denominator = torch.where(cond, 1.0 + tem_m, 1.0)
    u_upper, u_lower = ptem * u[1:], ptem1 * u[:-1]
    v_upper, v_lower = ptem * v[1:], ptem1 * v[:-1]
    outputs = [[], [], [], []]
    hprev, uprev, vprev = carry
    for offset in range(width):
        c = cond[offset]
        hprev = torch.where(c, (heat_keep[offset] * hprev + heat_source[offset]) / heat_denominator[offset], old_rows[0][offset])
        dby = torch.where(c, hprev - hs[offset], old_rows[3][offset])
        uprev = torch.where(c, (momentum_keep[offset] * uprev + u_upper[offset] + u_lower[offset]) / momentum_denominator[offset], old_rows[1][offset])
        vprev = torch.where(c, (momentum_keep[offset] * vprev + v_upper[offset] + v_lower[offset]) / momentum_denominator[offset], old_rows[2][offset])
        for values, value in zip(outputs, (hprev, uprev, vprev, dby)):
            values.append(value)
    return tuple(_stack(values) for values in outputs)


def tracer(cnv, kb, kmax, first, profiles, tracers, carry, *, cq):
    width = profiles.shape[0] - 1
    level = _levels(first, width)
    cond = (cnv & (level > kb) & (level < kmax))[:, :, None]
    z, lam = (profiles[:, i] for i in range(2))
    c = tracers[:, 0]
    dz = z[1:] - z[:-1]
    tem = (cq * 0.25 * (lam[1:] + lam[:-1]) * dz)[:, :, None]
    keep, source = 1.0 - tem, tem * (c[1:] + c[:-1])
    denominator = torch.where(cond, 1.0 + tem, 1.0)
    old0, old1 = tracers[1:, 1], tracers[1:, 2]
    transported, raw = [], []
    for offset in range(width):
        carry = torch.where(cond[offset], (keep[offset] * carry + source[offset]) / denominator[offset], old0[offset])
        transported.append(carry)
        raw.append(torch.where(cond[offset], carry, old1[offset]))
    return _stack(transported), _stack(raw)


def aerosol(cnv, kb, kmax, first, profiles, tracers, fscav, carry, *, cq):
    zi, xlamue, c0t, eta = (profiles[:, i] for i in range(4))
    ctro = tracers[:, 0]
    old = tuple(tracers[1:, i] for i in range(1, 5))
    outputs = [[], [], [], []]
    for offset in range(profiles.shape[0] - 1):
        level = first + offset
        cond = cnv & (level > kb) & (level < kmax)
        dz = zi[offset + 1] - zi[offset]
        tem = cq * 0.25 * (xlamue[offset + 1] + xlamue[offset]) * dz
        new = _divide(cond[:, None], (1.0 - tem)[:, None] * carry + tem[:, None] * (ctro[offset + 1] + ctro[offset]),
                      (1.0 + tem)[:, None], old[0][offset])
        chem = torch.where(cond[:, None], 0.8 * fscav[None, :] * new, old[2][offset])
        scav = chem / (1.0 + c0t[offset + 1, :, None] * dz[:, None])
        pw = torch.where(cond[:, None], c0t[offset + 1, :, None] * dz[:, None] * scav * eta[offset, :, None], old[3][offset])
        carry = torch.where(cond[:, None], scav + new - chem, new)
        raw = torch.where(cond[:, None], new, old[1][offset])
        for values, value in zip(outputs, (carry, raw, chem, pw)):
            values.append(value)
    return tuple(_stack(values) for values in outputs)


def _precip_terms(cond, moist_level, profiles, el2orc, *, cq, c1, hvap):
    """Carry-independent precipitation terms, level-major; rows match the source per-level values."""
    z, qs, temp, dby, l, e, q, dp, c0t = (profiles[:, i] for i in range(9))
    dz = z[1:] - z[:-1]
    safe_to = torch.where(cond, temp[1:], 1.0)
    q_sat = qs[1:]
    gamma = el2orc * q_sat / (safe_to * safe_to)
    qrch = q_sat + gamma * dby[1:] / (hvap * (1.0 + gamma))
    tem = cq * 0.25 * (l[1:] + l[:-1]) * dz
    eta_cur = e[1:]
    etah = 0.5 * (eta_cur + e[:-1])
    c0 = c0t[1:]
    return dict(
        dz=dz, safe_to=safe_to, gamma=gamma, qrch=qrch, keep=1.0 - tem, source=tem * (q[1:] + q[:-1]),
        denominator=torch.where(cond, 1.0 + tem, 1.0), cond=cond, moist_window=cond & moist_level,
        eta=eta_cur, etah=etah, dp=1000.0 * dp[1:], liquid_denominator=eta_cur + etah * (c0 + c1) * dz,
        detrain_factor=etah * c1 * dz, rain_factor=etah * c0 * dz)


def _precip(t, offset, old, qprev, *, grav):
    qold, qrold, pwold, cwold, dlold, zdold = old
    cond = t["cond"][offset]
    qk = torch.where(cond, (t["keep"][offset] * qprev + t["source"][offset]) / t["denominator"][offset], qold)
    qr = torch.where(cond, qk, qrold)
    eta, qrch, dp = t["eta"][offset], t["qrch"][offset], t["dp"][offset]
    dq = eta * (qk - qrch)
    moist = t["moist_window"][offset] & (dq > 0.0)
    qlk = torch.where(moist, dq / torch.where(moist, t["liquid_denominator"][offset], 1.0), 0.0)
    dl = torch.where(moist, t["detrain_factor"][offset] * qlk * grav / dp, dlold)
    qk = torch.where(moist, qlk + qrch, qk)
    pw = torch.where(moist, t["rain_factor"][offset] * qlk, pwold)
    cw = torch.where(moist, t["etah"][offset] * qlk * grav / dp, cwold)
    zd = torch.where(moist, torch.where(moist, dq / torch.where(moist, eta, 1.0), 0.0), zdold)
    return (qk, qr, pw, cw, dl, zd), moist, qlk


def precip(cnv, lower, upper, first, profiles, qprev, el2orc,
           *, cq, c1, hvap, grav):
    width = profiles.shape[0] - 1
    level = _levels(first, width)
    cond = cnv & (level >= lower) & (level < upper)
    t = _precip_terms(cond, True, profiles, el2orc, cq=cq, c1=c1, hvap=hvap)
    old_rows = tuple(profiles[1:, i] for i in range(9, 15))
    outputs = [[] for _ in old_rows]
    for offset in range(width):
        values, _, _ = _precip(t, offset, tuple(x[offset] for x in old_rows), qprev, grav=grav)
        qprev = values[0]
        for cells, value in zip(outputs, values):
            cells.append(value)
    return tuple(_stack(values) for values in outputs)


def buoyancy(cnv, kb, kbcon, ktcon, first, profiles, xlamud, qprev, el2orc,
             *, cq, c1, csmf, hvap, grav, cp, fv):
    width = profiles.shape[0] - 1
    level = _levels(first, width)
    cond = cnv & (level > kb) & (level < ktcon)
    moist_level = level >= kbcon
    t = _precip_terms(cond, moist_level, profiles, el2orc, cq=cq, c1=c1, hvap=hvap)
    old_rows = tuple(profiles[1:, i] for i in range(9, 18))
    above = cond & moist_level
    gamma, safe_to = t["gamma"], t["safe_to"]
    rfact = 1.0 + fv * cp * gamma * safe_to / hvap
    heat = (grav / (cp * safe_to)) * profiles[1:, 3] / (1.0 + gamma) * rfact
    q, l = profiles[:, 6], profiles[:, 4]
    moisture = grav * fv * torch.clamp_min(profiles[1:, 1] - q[1:], 0.0)
    drag = torch.where(above, torch.maximum(l[1:], xlamud), old_rows[7])
    safe_dz = torch.where(above, t["dz"], 1.0)
    u, v = profiles[:, 18], profiles[:, 19]
    shear = ((u[1:] - u[:-1]) / safe_dz) * ((u[1:] - u[:-1]) / safe_dz)
    shear = shear + ((v[1:] - v[:-1]) / safe_dz) * ((v[1:] - v[:-1]) / safe_dz)
    wush = torch.where(above, csmf * _sqrt(shear), old_rows[8])
    outputs = [[] for _ in range(7)]
    for offset in range(width):
        values, moist, qlk = _precip(t, offset, tuple(x[offset] for x in old_rows[:6]), qprev, grav=grav)
        buo = torch.where(moist, old_rows[6][offset] - grav * qlk, old_rows[6][offset])
        buo = torch.where(above[offset], buo + heat[offset], buo)
        buo = torch.where(above[offset], buo + moisture[offset], buo)
        qprev = values[0]
        for cells, value in zip(outputs, (*values, buo)):
            cells.append(value)
    return (*(_stack(values) for values in outputs), drag.movedim(0, 1), wush.movedim(0, 1))


def velocity(cnv, kbcon1, ktcon, first, profiles, carry, *, bb1, bb2):
    width = profiles.shape[0] - 1
    level = _levels(first, width)
    cond = cnv & (level > kbcon1) & (level < ktcon)
    z, d, b = (profiles[:, i] for i in range(3))
    w = profiles[1:, 3]
    dz = z[1:] - z[:-1]
    tem = 0.25 * bb1 * (d[:-1] + d[1:]) * dz
    tem1 = 0.5 * bb2 * (b[:-1] + b[1:])
    keep, denominator = 1.0 - tem, 1.0 + tem
    old_rows = profiles[1:, 4]
    cells = []
    for offset in range(width):
        tem2 = w[offset] * _sqrt(torch.clamp_min(carry, 0.0))
        wu = (keep[offset] * carry + (tem1[offset] - tem2) * dz[offset]) / denominator[offset]
        carry = torch.where(cond[offset], torch.clamp_min(wu, 0.0), old_rows[offset])
        cells.append(carry)
    return _stack(cells)


def evaporation(cnv, kb, ktcon, kmax, first, profiles, xmb, carry, rntot,
                *, shevf, evef, delt, el2orc, elocp, grav):
    rn, qcond, qevap, delqev, delq2, delqbar, deltbar, flg = carry
    width = profiles.shape[0]
    level = _levels(first, width)
    cond_rn = cnv & (level < ktcon) & (level > kb)
    pwo, dp, qeso, tk, qk = (profiles[:, i] for i in range(5))
    rain = (pwo * xmb * 0.001 * delt).masked_fill(~cond_rn, 0.0)
    window = (level < ktcon) & (level <= kmax)
    tk2 = tk * tk
    qcond_numerator = shevf * evef * (qk - qeso)
    saturation_term = el2orc * qeso
    dp_k = 1000.0 * dp
    factor = dp_k / grav
    tem = 0.001 * factor
    t_cells, q_cells = [None] * width, [None] * width
    for offset in range(width - 1, -1, -1):
        rn = rn + rain[offset]
        cond = flg & window[offset]
        qcond_den = 1.0 + saturation_term[offset] / torch.where(cond, tk2[offset], 1.0)
        qcond = torch.where(cond, qcond_numerator[offset] / qcond_den, qcond)
        evap_cond = cond & (rn > 0.0) & (qcond < 0.0)
        qevap = torch.where(evap_cond, -qcond * (1.0 - torch.exp(-0.32 * _sqrt(delt * rn))), 0.0)
        qevap = torch.where(evap_cond, torch.minimum(qevap, rn * 1000.0 * grav / dp_k[offset]), qevap)
        delq2 = torch.where(evap_cond, delqev + 0.001 * qevap * factor[offset], delq2)
        over = evap_cond & (delq2 > rntot)
        qevap = torch.where(over, 1000.0 * grav * (rntot - delqev) / dp_k[offset], qevap)
        flg = torch.where(over, False, flg)
        apply_evap = cond & (rn > 0.0) & (qevap > 0.0)
        tem1 = qevap * tem[offset]
        too_much = apply_evap & (tem1 > rn)
        qevap = torch.where(too_much, torch.where(too_much, rn / torch.where(too_much, tem[offset], 1.0), 0.0), qevap)
        rn = torch.where(too_much, 0.0, torch.where(apply_evap, rn - tem1, rn))
        q_cells[offset] = torch.where(apply_evap, qk[offset] + qevap, qk[offset])
        t_cells[offset] = torch.where(apply_evap, tk[offset] - elocp * qevap, tk[offset])
        deltv = torch.where(apply_evap, -elocp * qevap / delt, 0.0)
        delq = torch.where(apply_evap, qevap / delt, 0.0)
        delqev = torch.where(apply_evap, delqev + tem[offset] * qevap, delqev)
        delqbar = delqbar + (delq * factor[offset]).masked_fill(~cond, 0.0)
        deltbar = deltbar + (deltv * factor[offset]).masked_fill(~cond, 0.0)
    return (_stack(t_cells), _stack(q_cells),
            (rn, qcond, qevap, delqev, delq2, delqbar, deltbar, flg))
