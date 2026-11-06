# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Frozen R4 NSST water properties and thermal helpers, in column-batched Torch."""

from __future__ import annotations

import torch

from .containers import static_tensor

CP_W = 4000.0
Z_W_MAX = 20.0
Z_W_MIN = 0.2
_F = (0.237, 0.36, 0.179, 0.087, 0.08, 0.0246, 0.025, 0.007, 0.0004)
_GAMMA = (34.8, 2.27, 3.15e-2, 5.48e-3, 8.32e-4, 1.26e-4, 3.13e-4, 7.82e-5, 1.44e-5)


def positive_sqrt(x):
    return torch.where(x > 0.0, torch.sqrt(torch.where(x > 0.0, x, 1.0)), 0.0)


def density(t, s):
    tc = t - 273.16
    tc2 = tc * tc
    tc4 = tc2 * tc2
    return (
        999.842594
        + 6.793952e-2 * tc
        - 9.095290e-3 * tc**2
        + 1.001685e-4 * tc**3
        - 1.120083e-6 * tc4
        + 6.536332e-9 * (tc2 * (tc * tc2))
        + 8.24493e-1 * s
        - 4.0899e-3 * tc * s
        + 7.6438e-5 * tc**2 * s
        - 8.2467e-7 * tc**3 * s
        + 5.3875e-9 * tc4 * s
        - 5.72466e-3 * s**1.5
        + 1.0227e-4 * tc * s**1.5
        - 1.6546e-6 * tc**2 * s**1.5
        + 4.8314e-4 * s**2
    )


def rhocoef(t, s, rho):
    tc = t - 273.16
    tc2 = tc * tc
    tc4 = tc2 * tc2
    alpha = (
        6.793952e-2
        - 2.0 * 9.095290e-3 * tc
        + 3.0 * 1.001685e-4 * tc**2
        - 4.0 * 1.120083e-6 * tc**3
        + 5.0 * 6.536332e-9 * tc4
        - 4.0899e-3 * s
        + 2.0 * 7.6438e-5 * tc * s
        - 3.0 * 8.2467e-7 * tc**2 * s
        + 4.0 * 5.3875e-9 * tc**3 * s
        + 1.0227e-4 * s**1.5
        - 2.0 * 1.6546e-6 * tc * s**1.5
    )
    beta = (
        8.24493e-1
        - 4.0899e-3 * tc
        + 7.6438e-5 * tc**2
        - 8.2467e-7 * tc**3
        + 5.3875e-9 * tc4
        - 1.5 * 5.72466e-3 * s**0.5
        + 1.5 * 1.0227e-4 * tc * s**0.5
        - 1.5 * 1.6546e-6 * tc**2 * s**0.5
        + 2.0 * 4.8314e-4 * s
    )
    return -alpha / rho, beta / rho


def grv(sinlat):
    return 9.7803267715 * (
        1.0
        + 0.0052790414 * sinlat**2
        + 0.0000232718 * sinlat**4
        + 0.0000001262 * sinlat**6
        + 0.0000000007 * sinlat**8
    )


def _band_decay(depth):
    # exp(-depth / gamma) for all nine bands at once, band-major.
    gamma = static_tensor(_GAMMA, dtype=depth.dtype, device=depth.device).reshape(-1, *([1] * depth.ndim))
    return torch.exp(-depth / gamma)


def _ordered_band_sum(terms):
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def sw_ps_9b(z):
    depth = torch.where(z > 0.0, z, 0.0)
    weights = static_tensor(_F, dtype=depth.dtype, device=depth.device).reshape(-1, *([1] * depth.ndim))
    total = _ordered_band_sum(weights * _band_decay(depth))
    return torch.where(z > 0.0, 1.0 - total, 0.0)


def sw_ps_9b_aw(z):
    depth = torch.where(z > 0.0, z, 0.0)
    # The source repeats f(1)/f(2) in bands 4/5/7/8. Preserve the actual formula.
    weights = (_F[0], _F[1], _F[2], _F[0], _F[1], _F[5], _F[0], _F[1], _F[8])
    scale = static_tensor(tuple(f / gamma for f, gamma in zip(weights, _GAMMA)), dtype=depth.dtype, device=depth.device)
    total = _ordered_band_sum(scale.reshape(-1, *([1] * depth.ndim)) * _band_decay(depth))
    return torch.where(z > 0.0, total, 0.0)


def sw_rad_skin(z):
    depth = torch.where(z > 0.0, z, 1.0)
    absorbed = (
        0.065 + 11.0 * depth - torch.div(6.6e-5, depth) * (1.0 - torch.exp(-depth / 8.0e-4))
    )
    return torch.where(z > 0.0, absorbed, 0.0)


def cool_skin(
    ustar_a,
    f_nsol,
    f_sol_0,
    evap,
    sss,
    alpha,
    beta,
    rho_w,
    rho_a,
    ts,
    q_ts,
    hl_ts,
    grav,
    le,
):
    ustar = torch.clamp_min(ustar_a, 0.031)
    ustar2 = ustar * ustar
    ustar4 = ustar2 * ustar2
    deltaf = f_sol_0 * sw_rad_skin(torch.full_like(ustar, 0.001))
    hb = alpha * (f_nsol - deltaf) + beta * sss * CP_W * evap / le
    bigc = 16.0 * grav * CP_W * (rho_w * 1.0e-6) ** 3 / (rho_a * rho_a * 0.6 * 0.6)
    positive = hb > 0.0
    safe_hb = torch.where(positive, hb, 1.0)
    # A Python scalar numerator evaluates reciprocal-multiply instead of division.
    xi = torch.where(
        positive,
        ustar.new_full((), 6.0)
        / (1.0 + (bigc * safe_hb / ustar4) ** 0.75) ** 0.3333333,
        6.0,
    )
    z_c = torch.clamp_max(xi * 1.0e-6 / (torch.sqrt(rho_a / rho_w) * ustar), 0.01)
    deltaf = f_nsol - f_sol_0 * sw_rad_skin(z_c)
    cooling = deltaf > 0.0
    dt_cool = torch.where(cooling, deltaf * z_c / 0.6, 0.0)
    z_c = torch.where(cooling, z_c, 0.0)
    active = z_c > 0.0
    coupled = active & (hb > 0.0) & (alpha > 0.0)
    depth = torch.where(active, z_c, 1.0)
    zcsq = depth * depth
    a_c = (
        11.0
        + torch.div(6.6e-5, zcsq)
        - (torch.div(6.6e-5, 8.0e-4 * depth) + torch.div(6.6e-5, zcsq))
        * torch.exp(-depth / 8.0e-4)
    )
    safe_alpha = torch.where(coupled, alpha, 1.0)
    safe_hb = torch.where(coupled, hb, 1.0)
    cc1 = torch.div(6.0 * 1.0e-6, 0.6 * ustar * torch.sqrt(rho_a / rho_w))
    cc2 = bigc * safe_alpha / ustar4
    cc3 = beta * sss * CP_W / (safe_alpha * le)
    denominator = f_sol_0 * a_c - 4.0 * (cc1 * 0.6) ** 3 * (
        safe_hb / safe_alpha
    ) ** 0.25 / (cc2**0.75 * zcsq * zcsq)
    b_c = (q_ts + cc3 * hl_ts) / torch.where(coupled, denominator, 1.0)
    c0 = (z_c * q_ts + (f_nsol - deltaf - f_sol_0 * a_c * z_c) * b_c) * (1.0 / 0.6)
    cd = (f_sol_0 * a_c * z_c * b_c - q_ts) * (1.0 / 0.6)
    c0 = torch.where(coupled, c0, z_c * q_ts * (1.0 / 0.6))
    cd = torch.where(coupled, cd, -q_ts * (1.0 / 0.6))
    return dt_cool, z_c, torch.where(active, c0, 0.0), torch.where(active, cd, 0.0)


def convdepth(timestep, i0, q, sss, sep, rho, alpha, beta, xt, xs, xz):
    depth = torch.where(xz != 0.0, xz, 1.0)
    t = 2.0 * xt / depth
    s = 2.0 * xs / depth
    s1 = alpha * rho * t - beta * rho * s
    valid = s1 != 0.0
    factor = 2.0 * xz * timestep / torch.where(valid, s1, 1.0)
    fac1 = alpha * q / CP_W + beta * rho * sep
    night = positive_sqrt(factor * fac1)
    estimate = torch.zeros_like(xt)
    result = torch.zeros_like(xt)
    running = valid & (i0 > 0.0)
    for _ in range(5):
        absorbed = sw_ps_9b(estimate)
        aw = sw_ps_9b_aw(estimate)
        s2 = alpha * (q - (absorbed - aw * estimate) * i0) / CP_W + beta * rho * sep
        squared = factor * s2
        current = positive_sqrt(squared)
        result = torch.where(running, current, result)
        converged = (squared < 0.0) | (torch.abs(current - estimate) < 0.01)
        estimate = torch.where(running & ~converged, current, estimate)
        running = running & ~converged
    result = torch.clamp(result, 0.0, Z_W_MAX)
    return torch.where(valid, torch.where(i0 <= 0.0, night, result), 0.0)


def cal_w(xz, xt, xzts, xtts):
    w0 = 2.0 * (xtts - xt * xzts / xz) / xz
    wd = (2.0 * xt * xzts / xz**2 - w0) / xz
    return w0, wd


def cal_ttop(timestep, q_warm, rho, dz, xt, xz):
    warm = (xt + xt) / xz
    t0 = warm * (1.0 - dz / (xz + xz))
    return t0 + q_warm * timestep / (rho * CP_W * dz)


def dtm_1p_fca(d_conv, xt, xz):
    t0 = 2.0 * xt / xz
    t_fcl = t0 * (1.0 - d_conv / (2.0 * xz))
    return 2.0 * xt / t_fcl


def dtm_1p_tla(dz, te, xt):
    argument = xt * (xt - dz * te)
    return torch.where(
        argument > 0.0,
        (xt + positive_sqrt(argument)) / torch.where(argument > 0.0, te, 1.0),
        Z_W_MAX,
    )


def dtm_1p_mwa(xt):
    return 2.0 * xt / 5.0


def dtm_1p_mda(xz):
    return torch.clamp_min(xz, Z_W_MIN)


def dtm_1p_mta(dta, xt, xz):
    ta = torch.clamp_min(2.0 * xt / xz - dta, 0.0)
    return torch.where(ta > 0.0, 2.0 * xt / torch.where(ta > 0.0, ta, 1.0), Z_W_MAX)


def get_dtzm_point(xt, xz, dt_cool, zc, z1, z2):
    def contribution(value, depth, active):
        safe = torch.where(active, depth, 1.0)
        interval = z1 < z2
        span = torch.where(interval, z2 - z1, 1.0)
        full = value * (1.0 - (z1 + z2) / (safe + safe))
        partial = 0.5 * (1.0 - z1 / safe) * value * (depth - z1) / span
        mean = torch.where(z2 < depth, full, torch.where(z1 < depth, partial, 0.0))
        point = torch.where(z1 < depth, value * (1.0 - z1 / safe), 0.0)
        return torch.where(
            active, torch.where(interval, mean, torch.where(z1 == z2, point, 0.0)), 0.0
        )

    warm = (xt + xt) / torch.where(xt > 0.0, xz, 1.0)
    return contribution(warm, xz, xt > 0.0) - contribution(dt_cool, zc, zc > 0.0)


def get_dtzm_2d(xt, xz, dt_cool, zc, wet, z1, z2):
    # The pre/post source uses reciprocal-multiply for its warming profile.
    safe = torch.where((xt > 0.0) & wet, xz, 1.0)
    inverse = 1.0 / safe
    warm = (xt + xt) * inverse
    interval = z1 < z2
    span = torch.where(interval, z2 - z1, 1.0)
    full = warm * (1.0 - 0.5 * (z1 + z2) * inverse)
    partial = 0.5 * (1.0 - z1 * inverse) * warm * (xz - z1) / span
    mean = torch.where(z2 < xz, full, torch.where(z1 < xz, partial, 0.0))
    point = torch.where(z1 < xz, warm * (1.0 - z1 * inverse), 0.0)
    dtw = torch.where(
        (xt > 0.0) & wet,
        torch.where(interval, mean, torch.where(z1 == z2, point, 0.0)),
        0.0,
    )
    # Only the cooling part uses the same arithmetic as the point helper.
    dtc = -get_dtzm_point(torch.zeros_like(xt), xz, dt_cool, zc, z1, z2)
    return torch.where(wet, dtw - dtc, 0.0)
