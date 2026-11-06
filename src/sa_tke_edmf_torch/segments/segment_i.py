# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment I: cloud-top-driven downdraft mass flux and parcel transport."""

from __future__ import annotations

import torch

from sa_tke_edmf_torch.compile import run_compile_island
from sa_tke_edmf_torch.constants import GRAV, CP, HVAP, FV, EPS, EPSM1, ELOCP, EL2ORC, source_real
from sa_tke_edmf_torch.segments.segment_g import levels_over
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64
from sa_tke_edmf_torch.thermo import fpvs_from_table, fpvs_table_like
from sa_tke_edmf_torch.tracers import satmedmf_hydrometeor_indices

CHECKPOINT_FIELDS = ("xmfd", "xlamde", "tcdo", "qcdo", "buod", "krad", "mrad")


def _safe(value, mask, fallback=1.0):
    return torch.where(mask, value, fallback)


def _root(value):
    positive = value > 0.0
    return torch.where(positive, torch.sqrt(_safe(value, positive)), 0.0)


def _parcel(mask, previous_th, previous_qt, base_th, base_qt, lower, upper,
            entrainment, table, es_scale, qmin):
    # Profile rows hold zl, pix, plyr, thlx, qtx, thvx, u1, v1, in that order.
    tem = 0.5 * entrainment * (upper[0] - lower[0])
    next_th = ((1.0 - tem) * previous_th + tem * (lower[3] + upper[3])) / (1.0 + tem)
    next_qt = ((1.0 - tem) * previous_qt + tem * (lower[4] + upper[4])) / (1.0 + tem)
    th = torch.where(mask, next_th, base_th)
    tld = _safe(next_th, mask, 280.0) / _safe(lower[1], mask)
    es = es_scale * fpvs_from_table(tld, table)
    qs = torch.clamp_min(EPS * es / (_safe(lower[2], mask, 1000.0) + EPSM1 * es), qmin)
    dq = next_qt - qs
    qld = dq / (1.0 + EL2ORC * qs / (tld * tld))
    saturated = dq > 0.0
    qt = torch.where(mask, torch.where(saturated, qs + qld, next_qt), base_qt)
    return th, qt, tld, qs, qld, saturated, next_qt


def _buoyancy_levels(previous, profiles, entrainment, base, masks, table, es_scale, qmin):
    values = []
    th, qt = previous[0], previous[1]
    for j in range(base.shape[0] - 1, -1, -1):
        mask = masks[j]
        th, qt, tld, qs, qld, saturated, next_qt = _parcel(
            mask, th, qt, base[j, 0], base[j, 1], profiles[j], profiles[j + 1],
            entrainment[j], table, es_scale, qmin)
        thvd = torch.where(saturated,
            (th + profiles[j, 1] * ELOCP * qld) * (1.0 + FV * qs - qld),
            th * (1.0 + FV * next_qt))
        buoy = torch.where(mask, GRAV * (1.0 - thvd / _safe(profiles[j, 5], mask)), base[j, 2])
        values.append(torch.stack((th, qt, buoy)))
    return torch.stack(values[::-1])


def _velocity_levels(previous, zm, entrainment, buoy, wush, masks, seeds, initial):
    values = []
    for j in range(masks.shape[0] - 1, -1, -1):
        dz = zm[j + 1] - zm[j]
        base = torch.where(seeds[j], initial, torch.zeros_like(initial))
        tem = 0.5 * (entrainment[j] + entrainment[j + 1]) * dz
        force = 4.0 * buoy[j + 1] - wush[j + 1] * _root(previous)
        value = ((1.0 - tem) * previous + force * dz) / (1.0 + tem)
        previous = torch.where(masks[j], value, base)
        values.append(previous)
    return torch.stack(values[::-1])


def _transport_levels(previous, previous_water, profiles, entrainment, base, water,
                      environment_water, masks, table, tracer, transported,
                      es_scale, qmin, pgcon, cloud):
    values, waters = [], []
    th, qt, uc, vc = previous[0], previous[1], previous[3], previous[4]
    for j in range(base.shape[0] - 1, -1, -1):
        mask = masks[j]
        th, qt, tld, qs, qld, saturated, next_qt = _parcel(
            mask, th, qt, base[j, 0], base[j, 1], profiles[j], profiles[j + 1],
            entrainment[j], table, es_scale, qmin)
        qv = torch.where(saturated, qs, next_qt)
        ql = torch.where(saturated, qld, torch.zeros_like(qld))
        temp = torch.where(mask, torch.where(saturated, tld + ELOCP * qld, tld), base[j, 2])
        current_water = torch.where(mask[:, None] & (tracer[None, :] == 0), qv[:, None], water[j])
        current_water = torch.where(mask[:, None] & (tracer[None, :] == cloud), ql[:, None], current_water)
        tem = 0.5 * entrainment[j] * (profiles[j + 1, 0] - profiles[j, 0])
        uc = torch.where(mask,
            ((1.0 - tem) * uc + (tem - pgcon) * profiles[j + 1, 6]
             + (tem + pgcon) * profiles[j, 6]) / (1.0 + tem), base[j, 3])
        vc = torch.where(mask,
            ((1.0 - tem) * vc + (tem - pgcon) * profiles[j + 1, 7]
             + (tem + pgcon) * profiles[j, 7]) / (1.0 + tem), base[j, 4])
        value = ((1.0 - tem[:, None]) * previous_water + tem[:, None] *
                 (environment_water[j] + environment_water[j + 1])) / (1.0 + tem[:, None])
        previous_water = torch.where(mask[:, None] & transported[None, :], value, current_water)
        values.append(torch.stack((th, qt, temp, uc, vc)))
        waters.append(previous_water)
    return torch.stack(values[::-1]), torch.stack(waters[::-1])


def _descending_blocks(count):
    # Bound live FP64 outputs; full-depth recurrence fusion exhausted H100 driver memory (G158).
    for stop in range(count, 0, -4):
        yield max(0, stop - 4), stop


def _find_bottom(mrad, cnv, r, extends):
    searching = cnv
    for k in range(extends.shape[0] - 1, -1, -1):
        test = searching & (k < r)
        mrad = torch.where(test & extends[k], torch.full_like(mrad, k + 1), mrad)
        searching = searching & ~(test & ~extends[k])
    return mrad


def _depth_sums(mean, depth, mean_terms, depth_terms):
    # Only two live carries; retain the descending source sum, not a tree reduction.
    for k in range(mean_terms.shape[0] - 1, -1, -1):
        mean = mean + mean_terms[k]
        depth = depth + depth_terms[k]
    return mean, depth


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    q1 = state.require("q1")
    ncol, km, ntrac = q1.shape
    ntcw, _, _ = satmedmf_hydrometeor_indices(state.config, ntrac)
    cloud = ntcw - 1
    require_float64("q1", q1, (ncol, km, ntrac))
    for name in ("xmfd", "tcdo", "buod", "thlx", "thvx", "thlvx", "thetae",
                 "zl", "zm", "pix", "plyr", "wush", "u1", "v1"):
        require_float64(name, state.require(name), (ncol, km))
    require_float64("qcdo", state.require("qcdo"), (ncol, km, ntrac))
    require_float64("xlamde", state.require("xlamde"), (ncol, km - 1))
    literal = lambda v: source_real(v, state.config)
    ce0, pgcon, qmin = literal(0.4), literal(0.55), literal(1e-8)
    a1 = state.config.get("bl_dnfr", literal(0.1))
    zl, zm, pix, plyr = (state.require(n) for n in ("zl", "zm", "pix", "plyr"))
    thlx, thvx, thlvx = (state.require(n) for n in ("thlx", "thvx", "thlvx"))
    krad = state.require("krad")
    r = (krad.to(torch.int64) - 1).clamp(0, km - 2)
    cnv = state.require("scuflg") > 0.5
    entered = cnv.any()
    levels = torch.arange(km, device=q1.device)
    def at(field, index):
        return field.gather(1, index[:, None]).squeeze(1)
    def seed(field, value, mask):
        return torch.where(mask[:, None] & (levels[None, :] == r[:, None]), value[:, None], field)
    safe, root = _safe, _root

    # Keep collective early-return effects, including mrad writes to inactive lanes.
    mrad_in = state.fields.get("mrad", torch.full_like(krad, float(km - 1)))
    mrad = torch.where(entered, krad, mrad_in).to(torch.int64)
    qtx = q1[:, :, 0] + q1[:, :, cloud]
    dz = at(zm, r + 1) - at(zm, r)
    cooling = torch.clamp_min(500.0 * state.require("radmin") / safe(dz, cnv), -3.0)
    thld = seed(torch.zeros_like(thlx), at(thlx, r) + cooling, cnv)
    qtd = seed(torch.zeros_like(qtx), at(qtx, r), cnv)
    thlvd = at(thlvx, r) + cooling
    buo = torch.where(cnv[:, None], torch.zeros_like(thvx), state.require("buod"))
    buo = seed(buo, -GRAV * cooling / safe(at(thvx, r), cnv), cnv)
    thetae = state.require("thetae")
    dtheta = at(thetae, r) - at(thetae, r + 1)
    dqtop = at(qtx, r) - at(qtx, r + 1)
    instability = cnv & (dtheta > 0.0) & (dqtop > 0.0)
    ratio = CP * dtheta / (HVAP * safe(dqtop, instability))
    ra1 = torch.where(instability & (ratio > literal(0.7)),
                      torch.full_like(krad, 0.5), torch.full_like(krad, a1))
    mrad = run_compile_island("i_downdraft", "find_bottom", _find_bottom,
        mrad, cnv, r, (thlvd[None, :] <= thlvx[:, :km // 2].t()).contiguous())
    cnv = cnv & (krad - mrad >= 1)
    ce = ce0 * state.require("vez0fun")
    tke = state.require("tkemean")
    ce = torch.where(tke > 2.0, torch.maximum(ce, torch.minimum(root(tke / 2.0),
                     torch.full_like(tke, 5.0)) * ce0), ce)
    hrad = at(zm, r)
    xlamde = state.require("xlamde")
    half = km // 2
    def entrain(mask, bottom, current):
        mid = (bottom + (krad.to(torch.int64) - bottom) // 2 - 1).clamp(0, km - 2)
        delta = at(zl, mid + 1) - at(zl, mid)
        base = at(zm, (bottom - 2).clamp(0, km - 1))
        # No level depends on another: all lower-half levels at once.
        half = km // 2
        k = torch.arange(half, device=zm.device)
        inside = (k + 1 >= bottom[:, None]) & (k < r[:, None])
        z = zm[:, :half]
        low = torch.where((bottom == 1)[:, None], z + delta[:, None], z - base[:, None] + delta[:, None])
        high = torch.maximum(hrad[:, None] - z + delta[:, None], delta[:, None])
        candidate = ce[:, None] * (1.0 / safe(low, mask[:, None]) + 1.0 / safe(high, mask[:, None]))
        candidate = torch.where(inside, candidate, (ce / safe(delta, mask))[:, None])
        return torch.where(mask[:, None], candidate, current)
    entrainment_lower = entrain(cnv, mrad, xlamde[:, :km // 2])
    # Pack once. Every block then receives contiguous level rows without boundary copies.
    profiles = torch.stack(tuple(x[:, :half + 1].t() for x in
        (zl, pix, plyr, thlx, qtx, thvx, state.require("u1"), state.require("v1"))), dim=1)
    base = torch.stack(tuple(x[:, :half + 1].t() for x in (thld, qtd, buo)), dim=1)
    entrainment_rows = entrainment_lower.t().contiguous()
    level_rows = torch.arange(half, device=q1.device)[:, None]
    masks = cnv[None, :] & (level_rows < r[None, :])
    table, es_scale = fpvs_table_like(q1), literal(0.01)
    previous, pieces = base[half], []
    for first, stop in _descending_blocks(half):
        piece = run_compile_island(
            "i_downdraft", "buoyancy_levels", _buoyancy_levels,
            previous, profiles[first:stop + 1], entrainment_rows[first:stop],
            base[first:stop], masks[first:stop], table, es_scale, qmin)
        pieces.append(piece)
        previous = piece[0]
    parcel_rows = torch.cat(pieces[::-1], dim=0)
    th = list(parcel_rows[:, 0].unbind(0))
    initial_k = (r - 1).clamp(0, km - 2)
    xlam = torch.slice_scatter(xlamde, entrainment_lower, dim=1, start=0, end=km // 2)
    delta = at(zm, initial_k + 1) - at(zm, initial_k)
    buod = levels_over(buo, 0, list(parcel_rows[:, 2].unbind(0)))
    initial_wd = 4.0 * at(buod, initial_k + 1) * delta / (
        1.0 + at(xlam, initial_k) * delta)
    zm_rows, xlam_rows, buoy_rows, wush_rows = (
        x[:, :half + 1].t().contiguous() for x in (zm, xlam, buod, state.require("wush")))
    masks = cnv[None, :] & (level_rows < r[None, :] - 1)
    seeds = cnv[None, :] & (level_rows == initial_k[None, :])
    previous, pieces = torch.zeros_like(krad), []
    for first, stop in _descending_blocks(half):
        piece = run_compile_island(
            "i_downdraft", "velocity_levels", _velocity_levels,
            previous, zm_rows[first:stop + 1], xlam_rows[first:stop + 1],
            buoy_rows[first:stop + 1], wush_rows[first:stop + 1],
            masks[first:stop], seeds[first:stop], initial_wd)
        pieces.append(piece)
        previous = piece[0]
    wd_rows = torch.cat(pieces[::-1], dim=0)
    wd = list(wd_rows.unbind(0))
    mrad = torch.where(cnv, krad.to(torch.int64), mrad)
    mrad = run_compile_island("i_downdraft", "find_bottom", _find_bottom,
        mrad, cnv, r, wd_rows > 0.0)
    cnv = cnv & (krad - mrad >= 1)
    entrainment_lower = entrain(cnv, mrad, entrainment_lower)
    mean = torch.zeros_like(krad)
    depth = torch.zeros_like(krad)
    k_half = torch.arange(half, device=zl.device)
    in_layer = cnv[:, None] & (k_half + 1 >= mrad[:, None]) & (k_half < r[:, None])
    dz_half = zl[:, 1:half + 1] - zl[:, :half]
    mean_terms = torch.where(in_layer, entrainment_lower * dz_half, 0.0).t().contiguous()
    depth_terms = torch.where(in_layer, dz_half, 0.0).t().contiguous()
    mean, depth = run_compile_island("i_downdraft", "depth_sums", _depth_sums,
        mean, depth, mean_terms, depth_terms)
    mean = mean / safe(depth, cnv)
    radius = literal(0.2) / safe(mean, cnv)
    sigma = (literal(3.14) * radius * radius / safe(state.require("gdx") ** 2, cnv)).clamp(
        literal(0.001), literal(0.999))
    scale = torch.where(sigma > ra1, ((1.0 - sigma) * (1.0 - sigma)).clamp(0.0, 1.0),
                        torch.ones_like(sigma))
    dt = state.require("delt")
    # Each level is independent: all lower-half levels at once.
    value = ra1[:, None] * root(torch.stack(wd[:half], dim=1))
    value = torch.where((sigma > ra1)[:, None], sigma[:, None] * value / ra1[:, None], value)
    value = torch.minimum(scale[:, None] * value, dz_half / (dt[..., None] if dt.ndim else dt))
    xmfd = state.require("xmfd")
    mass_lower = torch.where(in_layer, value, xmfd[:, :half])

    th_seeded = seed(levels_over(thld, 0, th), at(thlx, r), cnv)
    qt_rows = torch.cat((parcel_rows[:, 1], qtd[:, half:half + 1].t()), dim=0)
    base = torch.stack((th_seeded[:, :half + 1].t(), qt_rows,
        *(state.require(n)[:, :half + 1].t() for n in ("tcdo", "ucdo", "vcdo"))), dim=1)
    water_rows = state.require("qcdo")[:, :half + 1].movedim(1, 0).contiguous()
    environment_water = q1[:, :half + 1].movedim(1, 0).contiguous()
    tracer = torch.arange(ntrac, device=q1.device)
    transported = (tracer != 0) & (tracer != cloud) & (tracer < ntrac - 1)
    masks = cnv[None, :] & (level_rows + 1 >= mrad[None, :]) & (level_rows < r[None, :])
    entrainment_rows = entrainment_lower.t().contiguous()
    previous, previous_water, pieces, waters = base[half], water_rows[half], [], []
    for first, stop in _descending_blocks(half):
        piece, water = run_compile_island(
            "i_downdraft", "transport_levels", _transport_levels,
            previous, previous_water, profiles[first:stop + 1], entrainment_rows[first:stop],
            base[first:stop], water_rows[first:stop], environment_water[first:stop + 1],
            masks[first:stop], table, tracer, transported, es_scale, qmin, pgcon, cloud)
        pieces.append(piece)
        waters.append(water)
        previous, previous_water = piece[0], water[0]
    transport_rows = torch.cat(pieces[::-1], dim=0)
    water = torch.cat(waters[::-1], dim=0)
    return state.with_updates(
        {
            "xmfd": torch.slice_scatter(xmfd, mass_lower, dim=1, start=0, end=half),
            "xlamde": torch.slice_scatter(xlamde, entrainment_lower, dim=1, start=0, end=half),
            "tcdo": levels_over(state.require("tcdo"), 0, list(transport_rows[:, 2].unbind(0))),
            "qcdo": torch.slice_scatter(state.require("qcdo"), water.movedim(0, 1), dim=1, start=0, end=half),
            "ucdo": levels_over(state.require("ucdo"), 0, list(transport_rows[:, 3].unbind(0))),
            "vcdo": levels_over(state.require("vcdo"), 0, list(transport_rows[:, 4].unbind(0))),
            "buod": buod,
            "krad": krad,
            "mrad": mrad.to(q1.dtype),
            "scuflg": cnv.to(q1.dtype),
            "wd2": levels_over(krad.new_zeros((ncol, km)), 0, wd[:half]),
            "downdraft_fraction": ra1,
        }
    )


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
