# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Ordered R4 diurnal warm-layer onset and modified Euler integration."""

from __future__ import annotations

import torch

from .nsst_thermal import CP_W, Z_W_MAX, Z_W_MIN, positive_sqrt, sw_ps_9b, sw_ps_9b_aw

WARM_FIELDS = ("xt", "xs", "xu", "xv", "xz", "xzts", "xtts")


def dtl_reset(state, mask):
    return {
        name: torch.where(mask, Z_W_MAX if name == "xz" else 0.0, state[name])
        for name in WARM_FIELDS
    }


def _sensitivity(p, state, drho, aw, norm, mask):
    xz = state["xz"]
    alpha, beta, rho, rich, grav = (
        p[n] for n in ("alpha", "beta", "rho", "rich", "grav")
    )
    denominator = torch.where(mask, norm, 1.0)
    torch._assert_async(
        ((~mask) | (norm > 0.0)).all(), "NSST warm-layer current norm is zero"
    )
    return (1.0 / denominator) * (
        (alpha * p["q_ts"] / CP_W + beta * p["sss"] * p["hl_ts"] / p["le"])
        * grav
        * xz**3
        / (4.0 * rich * rho)
        + (
            (p["tox"] * state["xu"] + p["toy"] * state["xv"]) / rho
            + (3.0 * drho - alpha * p["i0"] * aw * xz / (rho * CP_W))
            * grav
            * xz
            * xz
            / (4.0 * rich)
        )
        * state["xzts"]
    )


def eulerm(p, state):
    dt, rho, rich, grav = (p[n] for n in ("timestep", "rho", "rich", "grav"))
    xz0, xu0, xv0 = (state[n] for n in ("xz", "xu", "xv"))
    norm0 = xu0 * xu0 + xv0 * xv0
    speed = torch.clamp_min(norm0, 1.0e-8)
    fc = 0.000073 * p["sinlat"]
    q_warm = sw_ps_9b(xz0) * p["i0"] - p["q"]
    aw = sw_ps_9b_aw(xz0)
    drho = -p["alpha"] * q_warm / (rho * CP_W) + p["beta"] * p["sep"]
    dzw = xz0 * (
        (p["tox"] * xu0 + p["toy"] * xv0) / (rho * speed)
        + xz0 * xz0 * drho * grav / (4.0 * rich * speed)
    )
    first = {
        "xt": state["xt"] + dt * q_warm / (rho * CP_W),
        "xs": state["xs"] + dt * p["sep"],
        "xu": xu0 + dt * (fc * xv0 + p["tox"] / rho),
        "xv": xv0 + dt * (-fc * xu0 + p["toy"] / rho),
        "xz": xz0 + dt * dzw,
    }
    valid1 = (first["xt"] > 0.0) & (first["xz"] > 0.0) & (first["xz"] <= Z_W_MAX)
    first["xzts"] = state["xzts"] + dt * _sensitivity(p, state, drho, aw, norm0, valid1)
    first["xtts"] = state["xtts"] + dt * (p["i0"] * aw * state["xzts"] - p["q_ts"]) / (
        rho * CP_W
    )
    # Rows that returned after stage 1 must not enter singular stage-2 arithmetic.
    safe = {
        n: torch.where(valid1, first[n], 1.0 if n in ("xt", "xz", "xu", "xv") else 0.0)
        for n in WARM_FIELDS
    }
    xz1, xu1, xv1 = (safe[n] for n in ("xz", "xu", "xv"))
    q_warm = sw_ps_9b(xz1) * p["i0"] - p["q"]
    aw = sw_ps_9b_aw(xz1)
    drho = -p["alpha"] * q_warm / (rho * CP_W) + p["beta"] * p["sep"]
    norm1 = xu1 * xu1 + xv1 * xv1
    norm_safe = torch.where(valid1, norm1, 1.0)
    dzw = xz1 * (p["tox"] * xu1 + p["toy"] * xv1) / (
        rho * norm_safe
    ) + xz1 * xz1 * xz1 * drho * grav / (4.0 * rich * norm_safe)
    second = {
        "xt": state["xt"] + dt * q_warm / (rho * CP_W),
        "xs": state["xs"] + dt * p["sep"],
        "xu": xu0 + dt * (fc * xv1 + p["tox"] / rho),
        "xv": xv0 + dt * (-fc * xu1 + p["toy"] / rho),
        "xz": xz0 + dt * dzw,
    }
    valid2 = (
        valid1 & (second["xt"] > 0.0) & (second["xz"] > 0.0) & (second["xz"] <= Z_W_MAX)
    )
    second["xzts"] = state["xzts"] + dt * _sensitivity(p, safe, drho, aw, norm1, valid2)
    second["xtts"] = state["xtts"] + dt * (p["i0"] * aw * safe["xzts"] - p["q_ts"]) / (
        rho * CP_W
    )
    result = {n: 0.5 * (first[n] + second[n]) for n in WARM_FIELDS}
    reset = (
        ~valid2
        | (result["xt"] <= 0.0)
        | (result["xz"] < 0.0)
        | (result["xz"] > Z_W_MAX)
    )
    return dtl_reset(result, reset)


def dtm_onset(p, state):
    dt, rho, rich, grav = (p[n] for n in ("timestep", "rho", "rich", "grav"))
    fc = 1.46 / 10000.0 / 2.0 * p["sinlat"]
    high_latitude = torch.abs(torch.asin(p["sinlat"])) > 1.0
    rotation_time = positive_sqrt(
        (2.0 - 2.0 * torch.cos(fc * dt)) / torch.where(high_latitude, fc * fc * dt, 1.0)
    )
    ftime = torch.where(high_latitude, rotation_time, dt)
    coeff1 = p["alpha"] * grav / CP_W
    coeff2 = p["beta"] * grav * rho
    depth = torch.full_like(state["xz"], Z_W_MIN)
    q_warm = sw_ps_9b(depth) * p["i0"] - p["q"]
    warml = coeff1 * q_warm - coeff2 * p["sep"]
    running = (warml > 0.0) & (q_warm > 0.0)
    z_w = torch.full_like(depth, Z_W_MAX)
    for iteration in range(1, 6):
        valid = (warml > 0.0) & (q_warm > 0.0)
        trial = (
            positive_sqrt(2.0 * rich * ftime / rho)
            * positive_sqrt(p["tox"] ** 2 + p["toy"] ** 2)
            / torch.sqrt(torch.where(valid, warml, 1.0))
        )
        current = torch.where(valid, trial, Z_W_MAX)
        z_w = torch.where(running, current, z_w)
        converged = (
            (torch.abs(current - depth) < 0.01) & (current != Z_W_MAX) & (iteration < 5)
        )
        update = running & valid & ~converged
        depth = torch.where(update, current, depth)
        q_warm = torch.where(update, sw_ps_9b(depth) * p["i0"] - p["q"], q_warm)
        warml = torch.where(update, coeff1 * q_warm - coeff2 * p["sep"], warml)
        running = update
    xz0 = torch.clamp_min(z_w, Z_W_MIN)
    entered = (z_w < Z_W_MAX) & (q_warm > 0.0)
    q_warm = sw_ps_9b(z_w) * p["i0"] - p["q"]
    ft0 = q_warm / (rho * CP_W)
    fs0 = p["sep"]
    fu0 = p["tox"] / rho
    fv0 = p["toy"] / rho
    xt1, xu1, xv1 = dt * ft0, dt * fu0, dt * fv0
    norm1 = xu1 * xu1 + xv1 * xv1
    torch._assert_async(
        ((~entered) | (norm1 > 0.0)).all(), "NSST onset stress norm is zero"
    )
    divisor = torch.where(entered, norm1, 1.0)
    fz0 = (
        xz0
        * (
            (p["tox"] * xu1 + p["toy"] * xv1) / rho
            + p["beta"] * grav * p["sep"] * xz0 * xz0 / (4.0 * rich)
            - p["alpha"] * grav * q_warm * xz0 * xz0 / (4.0 * rich * CP_W * rho)
        )
        / divisor
    )
    xz1 = torch.clamp_min(xz0 + dt * fz0, Z_W_MIN)
    valid = entered & (xt1 >= 0.0) & (xz1 <= Z_W_MAX)
    stage_depth = torch.where(valid, xz1, Z_W_MIN)
    q_warm = sw_ps_9b(stage_depth) * p["i0"] - p["q"]
    ft1 = q_warm / (rho * CP_W)
    fs1 = p["sep"]
    fu1 = fc * xv1 + p["tox"] / rho
    fv1 = -fc * xu1 + p["toy"] / rho
    fz1 = (
        stage_depth
        * (
            (p["tox"] * xu1 + p["toy"] * xv1) / rho
            + p["beta"] * grav * p["sep"] * stage_depth * stage_depth / (4.0 * rich)
            - p["alpha"]
            * grav
            * q_warm
            * stage_depth
            * stage_depth
            / (4.0 * rich * CP_W * rho)
        )
        / divisor
    )
    candidate = {
        "xt": 0.5 * dt * (ft0 + ft1),
        "xs": 0.5 * dt * (fs0 + fs1),
        "xu": 0.5 * dt * (fu0 + fu1),
        "xv": 0.5 * dt * (fv0 + fv1),
        "xz": torch.clamp_min(xz0 + 0.5 * dt * (fz0 + fz1), Z_W_MIN),
    }
    aw = sw_ps_9b_aw(candidate["xz"])
    alpha = torch.where(valid, p["alpha"], 1.0)
    extra = rho * CP_W * p["beta"] * p["sss"]
    numerator = p["q_ts"] + extra * p["hl_ts"] * candidate["xz"] / (p["le"] * alpha)
    denominator = (
        p["i0"] * candidate["xz"] * aw
        + 2.0 * q_warm
        - 2.0 * extra * p["sep"] / (p["le"] * alpha)
    )
    candidate["xzts"] = numerator / torch.where(valid, denominator, 1.0)
    candidate["xtts"] = (
        dt * (p["i0"] * aw * candidate["xzts"] - p["q_ts"]) / (rho * CP_W)
    )
    # A failed onset leaves the actual incoming frozen-source buffers untouched.
    result = {n: torch.where(valid, candidate[n], state[n]) for n in WARM_FIELDS}
    reset = (entered & ~valid) | (result["xt"] < 0.0) | (result["xz"] > Z_W_MAX)
    return dtl_reset(result, reset)


def dtm_1p(p, state):
    onset = state["xt"] <= 0.0
    onset_state = {
        n: torch.where(onset, state[n], Z_W_MAX if n == "xz" else 0.0)
        for n in WARM_FIELDS
    }
    euler_state = {
        n: torch.where(~onset, state[n], 1.0 if n in ("xt", "xz", "xu", "xv") else 0.0)
        for n in WARM_FIELDS
    }
    first = dtm_onset(p, onset_state)
    second = eulerm(p, euler_state)
    return {n: torch.where(onset, first[n], second[n]) for n in WARM_FIELDS}
