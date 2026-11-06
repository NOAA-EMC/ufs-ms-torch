# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Bounded tensor recurrences with immutable inputs and ordered level carries."""
from __future__ import annotations

import torch
from sa_sas_deep_convection_torch.constants import FV, GRAV
from sa_sas_deep_convection_torch.ops import sqrt_zero_grad
from ufs_physics_common.ordered_sum import masked_sum


def _divide(mask, numerator, denominator, other):
    return torch.where(mask, numerator / torch.where(mask, denominator, 1.0), other)


def _stack_levels(values):
    # Materialize levels before columns so the next ordered carry is a
    # contiguous column row. Public outputs retain the leading column axis.
    return torch.stack(values, dim=0).movedim(0, 1)


def place_levels(base, value, start):
    """``base`` with levels ``start:start + value.shape[1]`` (dim 1) replaced by ``value``, one slice.

    Assembling chunk results as per-level views into ``torch.stack`` made Inductor emit one copy
    kernel per level on CUDA (offline sm_80 codegen, x1.10242); values are unchanged.
    """
    # Inductor can retain a broadcast base's zero strides in the scatter output.
    # Materialize that layout only; ordinary recurrence fields keep their layout.
    if 0 in base.stride():
        base = base.contiguous()
    return torch.slice_scatter(base, value, dim=1, start=start, end=start + value.shape[1])


def join_levels(base, pieces, start):
    """Assemble adjacent recurrence blocks once, without copying the full field per block."""
    if not pieces:
        return base
    from sa_sas_deep_convection_torch.compile import run_compile_island

    return run_compile_island("g_ordered_chunks", "join_levels", _join_levels, base, tuple(pieces), start)


def _join_levels(base, pieces, start):
    value = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
    return place_levels(base, value, start)


def place_cells(base, cells, start, width=8):
    """``base`` with levels ``start:start + len(cells)`` replaced by ``cells`` (one ``[ncol]`` tensor per level).

    Stack at most ``width`` cells per piece to bound Inductor's stack inputs. Join the
    pieces and replace the completed field once, not once per piece.
    """
    pieces = tuple(_stack_levels(cells[lo:lo + width]) for lo in range(0, len(cells), width))
    return join_levels(base, pieces, start)


def _levels(first, width):
    # ``first + offset`` for every chunk offset; small integers are exact in first's dtype.
    return (first + torch.arange(width, dtype=first.dtype, device=first.device))[:, None]


def seed_updraft(active, kb, levels, heo, uo, vo, ctro, hcko, ucko, vcko, ecko, ercko, pwavo,
                 hwrf_samfdeep):
    mask = active[:, None] & (kb[:, None] == levels[None, :])
    hcko, ucko, vcko = (torch.where(mask, env, old)
                        for env, old in ((heo, hcko), (uo, ucko), (vo, vcko)))
    if not hwrf_samfdeep:
        ecko = torch.where(mask[:, :, None], ctro, ecko)
        ercko = torch.where(mask[:, :, None], ctro, ercko)
    return hcko, ucko, vcko, ecko, ercko, torch.where(active, torch.zeros_like(pwavo), pwavo)


def _rows(x):
    # Level-major contiguous copy: row ``offset`` holds the level slice ``x[:, offset]``.
    return x.transpose(0, 1).contiguous()


def pack_profiles(*values):
    """One contiguous [level, profile, column, ...] packet for repeated bounded regions."""
    return torch.stack(tuple(value.movedim(1, 0) for value in values), dim=1)


def _pad_levels(value, count, fill=0.):
    return torch.nn.functional.pad(value, (0, 0) * (value.ndim - 1) + (0, count), value=fill)


def _two_sum(a, b):
    total = a + b
    z = total - a
    return total, (a - (total - z)) + (b - z)


def _two_product(a, b):
    # Dekker splitting requires finite normal products and no splitter overflow.
    splitter = 134217729. if a.dtype == torch.float64 else 4097.
    ca, cb = splitter * a, splitter * b
    ah, bh = ca - (ca - a), cb - (cb - b)
    al, bl = a - ah, b - bh
    product = a * b
    error = al * bl - (((product - ah * bh) - al * bh) - ah * bl)
    return product, error


def _carry_map(function, value):
    return tuple(function(part) for part in value) if isinstance(value, tuple) else function(value)


def _carry_pair(value):
    if isinstance(value, tuple):
        return value
    return value, torch.zeros_like(value) if isinstance(value, torch.Tensor) else 0.


def _carry_stack(values, dim=0):
    if any(isinstance(value, tuple) for value in values):
        return tuple(torch.stack(parts, dim=dim) for parts in zip(*map(_carry_pair, values), strict=True))
    return torch.stack(values, dim=dim)


def _carry_where(mask, value, other):
    if isinstance(value, tuple):
        return tuple(torch.where(mask, a, b) for a, b in zip(value, _carry_pair(other), strict=True))
    return torch.where(mask, value, other)


def _carry_sum(a, b):
    if not isinstance(a, tuple) and not isinstance(b, tuple):
        return a + b
    a, b = _carry_pair(a), _carry_pair(b)
    high, low = _two_sum(a[0], b[0])
    return _two_sum(high, low + (a[1] + b[1]))


def _carry_difference(a, b):
    if not isinstance(a, tuple) and not isinstance(b, tuple):
        return a - b
    return _carry_sum(a, _carry_map(lambda value: -value, b))


def _carry_product(a, b):
    if not isinstance(a, tuple) and not isinstance(b, tuple):
        return a * b
    # These literal factors scale both components exactly in the normal domain.
    for factor, value in ((a, b), (b, a)):
        if isinstance(factor, (int, float)):
            if factor not in (-1., .25, .5, 1.):
                raise ValueError("paired literal product requires an exact power-of-two scale")
            return _carry_map(lambda part: factor * part, value)
    a, b = _carry_pair(a), _carry_pair(b)
    high, low = _two_product(a[0], b[0])
    low = low + (a[0] * b[1] + a[1] * b[0])
    return _two_sum(high, low + a[1] * b[1])


def _carry_divide(numerator, denominator):
    if isinstance(denominator, tuple):
        high = numerator[0] / denominator[0]
        product = _carry_product((high, torch.zeros_like(high)), denominator)
        residual = _carry_difference(numerator, product)
        low = (residual[0] + residual[1]) / denominator[0]
        return _two_sum(high, low)
    high = numerator[0] / denominator
    product, error = _two_product(high, denominator)
    residual, remainder = _two_sum(numerator[0], -product)
    low = (residual + ((numerator[1] - error) + remainder)) / denominator
    return _two_sum(high, low)


def _affine_prefix(linear, bias, initial):
    """Inclusive affine scan with linear work, including signed and reset maps.

    Pair contraction computes odd-index prefixes recursively. Each intervening
    even prefix then needs one map application. No coefficient is inverted.
    Arithmetic association differs from an ordered recurrence.
    """
    field = linear[0] if isinstance(linear, tuple) else linear
    count = field.shape[0]
    seed = _carry_map(lambda value: value.unsqueeze(0), initial)
    if count == 1:
        return _carry_sum(_carry_product(linear, seed), bias)
    pairs = count // 2
    right = _carry_map(lambda value: value[1:2 * pairs:2], linear)
    left = _carry_map(lambda value: value[:2 * pairs:2], linear)
    left_bias = _carry_map(lambda value: value[:2 * pairs:2], bias)
    right_bias = _carry_map(lambda value: value[1:2 * pairs:2], bias)
    odd = _affine_prefix(_carry_product(right, left),
                         _carry_sum(_carry_product(right, left_bias), right_bias), initial)
    index = torch.arange((count + 1) // 2, device=field.device)
    previous = _carry_map(lambda value: value.index_select(0, (index - 1).clamp_min(0)), odd)
    first = (index == 0).reshape(-1, *([1] * (field.ndim - 1)))
    if isinstance(linear, tuple):
        previous = tuple(torch.where(first, start, value) for start, value in zip(seed, previous, strict=True))
    else:
        previous = torch.where(first, seed, previous)
    even = _carry_sum(_carry_product(_carry_map(lambda value: value[::2], linear), previous),
                      _carry_map(lambda value: value[::2], bias))
    if count % 2:
        odd = _carry_map(lambda value: _pad_levels(value, 1), odd)
    if isinstance(linear, tuple):
        return tuple(torch.stack((a, b), dim=1).flatten(0, 1)[:count]
                     for a, b in zip(even, odd, strict=True))
    return torch.stack((even, odd), dim=1).flatten(0, 1)[:count]


def _blocked_affine_recurrence(keep, sources, denominator, update, old, initial, *, microblock=4,
                               compensated=False, return_components=False):
    """Scan block-entry carries, then evaluate short ordered microblocks.

    Inputs are level-major and equally shaped; ``sources`` preserves successive
    numerator additions. Inactive levels overwrite the carry, not skip a map.
    This experimental helper is not selected by the production coordinator.
    """
    count = (keep[0] if isinstance(keep, tuple) else keep).shape[0]
    if return_components and not compensated:
        raise ValueError("component output requires compensated carries")
    padding = (-count) % microblock
    def blocks(value, fill):
        if isinstance(value, tuple):
            return blocks(value[0], fill), blocks(value[1], 0.)
        if padding:
            value = _pad_levels(value, padding, fill)
        return value.reshape(-1, microblock, *value.shape[1:]).movedim(1, 0)
    keep = blocks(keep, 1.)
    denominator = blocks(denominator, 1.)
    update = blocks(update, True)
    old = blocks(old, 0.)
    sources = tuple(blocks(source, 0.) for source in sources)
    safe_denominator = _carry_where(update, denominator, 1.)
    if compensated:
        keep = _carry_where(update, _carry_pair(keep), 0.)
        sources = tuple(_carry_where(update, _carry_pair(source), 0.) for source in sources)
        initial = _carry_pair(initial)
        linear = _carry_divide(keep, safe_denominator)
    else:
        linear = torch.where(update, keep / safe_denominator, 0.)
    source = sources[0]
    for term in sources[1:]:
        source = _carry_sum(source, term)
    if compensated:
        bias = _carry_divide(source, safe_denominator)
        bias = _carry_where(update, bias, old)
    else:
        bias = torch.where(update, source / safe_denominator, old)
    block_linear = _carry_map(lambda value: value[0], linear)
    block_bias = _carry_map(lambda value: value[0], bias)
    for offset in range(1, microblock):
        coefficient = _carry_map(lambda value: value[offset], linear)
        block_linear = _carry_product(coefficient, block_linear)
        block_bias = _carry_sum(_carry_product(coefficient, block_bias),
                                _carry_map(lambda value: value[offset], bias))
    completed = _affine_prefix(block_linear, block_bias, initial)
    field = completed[0] if compensated else completed
    first = (torch.arange(field.shape[0], device=field.device) == 0).reshape(
        -1, *([1] * (field.ndim - 1)))
    if compensated:
        carry = tuple(torch.where(first, seed.unsqueeze(0), torch.roll(value, 1, dims=0))
                      for seed, value in zip(initial, completed, strict=True))
    else:
        carry = torch.where(first, initial.unsqueeze(0), torch.roll(completed, 1, dims=0))
    values = []
    for offset in range(microblock):
        numerator = _carry_product(_carry_map(lambda value: value[offset], keep), carry)
        for term in sources:
            numerator = _carry_sum(numerator, _carry_map(lambda value: value[offset], term))
        if compensated:
            candidate = _carry_divide(numerator, _carry_map(lambda value: value[offset], safe_denominator))
            reset = _carry_map(lambda value: value[offset], old)
            carry = _carry_where(update[offset], candidate, reset)
            reset_value = reset[0] + reset[1] if isinstance(reset, tuple) else reset
            values.append(carry if return_components else
                          torch.where(update[offset], carry[0] + carry[1], reset_value))
        else:
            carry = torch.where(update[offset], numerator / safe_denominator[offset], old[offset])
            values.append(carry)
    if return_components:
        return tuple(torch.stack(tuple(value[part] for value in values), dim=1).flatten(0, 1)[:count]
                     for part in range(2))
    return torch.stack(values, dim=1).flatten(0, 1)[:count]


def tke_mean(active, kb, kbcon, first, profile_rows, tkemean, sumx):
    # Carry-independent terms for all levels at once; only the ordered sums loop.
    z, tke = profile_rows.unbind(1)
    width = z.shape[0] - 1
    level = _levels(first, width)
    layer = active & (level >= kb) & (level < kbcon)
    dz = z[1:] - z[:-1]
    term = tke[:-1] * dz
    return masked_sum(tkemean, layer, term), masked_sum(sumx, layer, dz)


def sigma_average(active, kbcon1, ktcon, first, profile_rows, omegac, sumx):
    del_, omega = profile_rows.unbind(1)
    level = _levels(first, del_.shape[0] - 1)
    layer = active & (level >= kbcon1) & (level < ktcon)
    dp = 1000.0 * del_[1:]
    tem = 0.5 * (omega[1:] + omega[:-1])
    return masked_sum(omegac, layer, tem * dp), masked_sum(sumx, layer, dp)


def shear_sum(active, kb, ktcon, first, profile_rows, vshear):
    u, v = profile_rows.unbind(1)
    level = _levels(first, u.shape[0] - 1)
    layer = active & (level > kb) & (level <= ktcon)
    shear = sqrt_zero_grad((u[1:] - u[:-1]) ** 2 + (v[1:] - v[:-1]) ** 2)
    return masked_sum(vshear, layer, shear)


def downdraft_depth(active, kd94, first, profile_rows, sumx):
    z, = profile_rows.unbind(1)
    level = _levels(first, z.shape[0] - 1)
    layer = active & (level < kd94)
    dz = z[1:] - z[:-1]
    return masked_sum(sumx, layer, dz)


def downdraft_eta(active, kmax, kd94, jmin, first, profile_rows, carry,
                  xlamd, xlamddt, xlamdet):
    z, old = profile_rows.unbind(1)
    width = z.shape[0] - 1
    level = _levels(first, width)
    layer = active & (level <= kmax - 1.0)
    dz = z[1:] - z[:-1]
    ptem_upper = xlamddt - xlamdet
    ptem_lower = xlamd + xlamddt - xlamdet
    ptem = torch.where(level < kd94, ptem_lower, ptem_upper)
    factor = 1.0 - ptem * dz
    update = layer & (((level < jmin) & (level >= kd94)) | (level < kd94))
    old_rows = old[:-1]
    values = [None] * width
    for offset in range(width - 1, -1, -1):
        carry = torch.where(update[offset], carry * factor[offset], old_rows[offset])
        values[offset] = carry
    return _stack_levels(values)


def eta_lower(active, kb, kbcon, first, profile_rows, carry):
    z, e, d, old = profile_rows.unbind(1)
    width = z.shape[0] - 1
    level = _levels(first, width)
    mask = active & (level < kbcon) & (level >= kb)
    dz = z[1:] - z[:-1]
    tem = 0.5 * (d[:-1] + d[1:])
    ptem = 0.5 * (e[:-1] + e[1:]) - tem
    denominator = torch.where(mask, 1.0 + ptem * dz, 1.0)
    old_rows = old[:-1]
    values = [None] * width
    for offset in range(width - 1, -1, -1):
        carry = torch.where(mask[offset], carry / denominator[offset], old_rows[offset])
        values[offset] = carry
    return _stack_levels(values)


def eta_upper(flg, kbcon, kmax, ktconn, first, profile_rows, carry):
    z, e, d, old = profile_rows.unbind(1)
    width = z.shape[0] - 1
    level = _levels(first, width)
    # kmax changes only where flg turns false, so the level window is fixed by the input kmax.
    window = (level > kbcon) & (level < kmax)
    dz = z[1:] - z[:-1]
    tem = 0.5 * (d[1:] + d[:-1])
    ptem = 0.5 * (e[1:] + e[:-1]) - tem
    factor = 1.0 + ptem * dz
    old_rows = old[1:]
    values = []
    for offset in range(width):
        mask = flg & window[offset]
        candidate = carry * factor[offset]
        carry = torch.where(mask, candidate, old_rows[offset])
        failed = mask & (candidate <= 0.0)
        kmax = torch.where(failed, level[offset, 0], kmax)
        ktconn = torch.where(failed, level[offset, 0], ktconn)
        flg = torch.where(failed, False, flg)
        values.append(carry)
    return _stack_levels(values), kmax, ktconn, flg


def updraft(active, kb, kmax, first, profile_rows, carry, pgcon, *, microblock=0, compensated=False):
    if compensated and not microblock:
        raise ValueError("compensated updraft requires a blocked recurrence")
    z, e, d, h, hs, u, v, hcko, ucko, vcko, dbyo = profile_rows.unbind(1)
    width = z.shape[0] - 1
    level = _levels(first, width)
    mask = active & (level > kb) & (level < kmax)
    hs = hs[1:]
    old_rows = tuple(x[1:] for x in (hcko, ucko, vcko, dbyo))
    prepare = _carry_pair if compensated else lambda value: value
    dz = _carry_difference(prepare(z[1:]), prepare(z[:-1]))
    tem = _carry_product(_carry_product(.5, _carry_sum(prepare(e[1:]), prepare(e[:-1]))), dz)
    tem1 = _carry_product(_carry_product(.25, _carry_sum(prepare(d[1:]), prepare(d[:-1]))), dz)
    heat_keep = _carry_difference(1., tem1)
    heat_denominator = _carry_where(mask, _carry_difference(_carry_sum(1., tem), tem1), 1.)
    tem_mom = _carry_product(.5, tem)
    momentum_denominator = _carry_where(mask, _carry_sum(1., tem_mom), 1.)
    ptem, ptem1 = _carry_sum(tem_mom, pgcon), _carry_difference(tem_mom, pgcon)
    momentum_keep = _carry_difference(1., tem_mom)
    u_upper, u_lower = _carry_product(ptem, u[1:]), _carry_product(ptem1, u[:-1])
    v_upper, v_lower = _carry_product(ptem, v[1:]), _carry_product(ptem1, v[:-1])
    if microblock:
        anchor = h[:1]
        centered_source = _carry_product(_carry_product(tem, .5), _carry_sum(
            _carry_difference(prepare(h[1:]), anchor), _carry_difference(prepare(h[:-1]), anchor)))
        keep = _carry_stack((heat_keep, momentum_keep, momentum_keep), dim=1)
        source = _carry_stack((centered_source, u_upper, v_upper), dim=1)
        zero_source = torch.zeros_like(centered_source[0] if compensated else centered_source)
        second_source = _carry_stack((zero_source, u_lower, v_lower), dim=1)
        denominator = _carry_stack((heat_denominator, momentum_denominator, momentum_denominator), dim=1)
        old = _carry_stack((_carry_difference(prepare(old_rows[0]), anchor), old_rows[1], old_rows[2]), dim=1)
        initial = _carry_stack((_carry_difference(prepare(carry[0]), anchor[0]), carry[1], carry[2]))
        result = _blocked_affine_recurrence(keep, (source, second_source), denominator,
            mask[:, None].expand_as(keep[0] if compensated else keep), old, initial,
            microblock=microblock, compensated=compensated,
            return_components=compensated)
        if compensated:
            anomaly, up, vp = result[0].unbind(1)
            anomaly_low, up_low, vp_low = result[1].unbind(1)
            # Keep both anomaly components through the cancelling consumers.
            heat = _carry_sum((anomaly, anomaly_low), (anchor, torch.zeros_like(anchor)))
            buoyancy = _carry_sum((anomaly, anomaly_low), _two_sum(anchor, -hs))
            values = (torch.where(mask, heat[0] + heat[1], old_rows[0]),
                      torch.where(mask, up + up_low, old_rows[1]),
                      torch.where(mask, vp + vp_low, old_rows[2]),
                      torch.where(mask, buoyancy[0] + buoyancy[1], old_rows[3]))
            return tuple(value.movedim(0, 1) for value in values)
        anomaly, up, vp = result.unbind(1)
        values = (torch.where(mask, anomaly + anchor, old_rows[0]),
                  torch.where(mask, up, old_rows[1]), torch.where(mask, vp, old_rows[2]),
                  torch.where(mask, anomaly + (anchor - hs), old_rows[3]))
        return tuple(value.movedim(0, 1) for value in values)
    outputs = [[], [], [], []]
    heat_source = tem * 0.5 * (h[1:] + h[:-1])
    hprev, uprev, vprev = carry
    for offset in range(width):
        m = mask[offset]
        hprev = torch.where(m, (heat_keep[offset] * hprev + heat_source[offset]) / heat_denominator[offset], old_rows[0][offset])
        dby = torch.where(m, hprev - hs[offset], old_rows[3][offset])
        uprev = torch.where(m, (momentum_keep[offset] * uprev + u_upper[offset] + u_lower[offset]) / momentum_denominator[offset], old_rows[1][offset])
        vprev = torch.where(m, (momentum_keep[offset] * vprev + v_upper[offset] + v_lower[offset]) / momentum_denominator[offset], old_rows[2][offset])
        for values, value in zip(outputs, (hprev, uprev, vprev, dby)):
            values.append(value)
    return tuple(_stack_levels(values) for values in outputs)


def tracer_up(active, kb, kmax, first, profile_rows, tracer_rows, carry):
    z, e = profile_rows.unbind(1)
    env, old, original = tracer_rows.unbind(1)
    width = z.shape[0] - 1
    level = _levels(first, width)
    mask = (active & (level > kb) & (level < kmax))[:, :, None]
    dz = z[1:] - z[:-1]
    tem = (0.25 * (e[1:] + e[:-1]) * dz)[:, :, None]
    keep, source = 1.0 - tem, tem * (env[1:] + env[:-1])
    denominator = torch.where(mask, 1.0 + tem, 1.0)
    old_rows, original_rows = old[1:], original[1:]
    values, originals = [], []
    for offset in range(width):
        carry = torch.where(mask[offset], (keep[offset] * carry + source[offset]) / denominator[offset], old_rows[offset])
        values.append(carry)
        originals.append(torch.where(mask[offset], carry, original_rows[offset]))
    return _stack_levels(values), _stack_levels(originals)


def aerosol_up(active, kb, kmax, first, profile_rows, tracer_rows, fscav, carry):
    zi, ue, eta, c0t = profile_rows.unbind(1)
    environment, *old = tracer_rows.unbind(1)
    outputs = [[], [], [], []]
    for offset in range(zi.shape[0] - 1):
        level = first + offset
        mask = (active & (level > kb) & (level < kmax))[:, None]
        dz = zi[offset + 1] - zi[offset]
        tem_recur = (0.25 * (ue[offset + 1] + ue[offset]) * dz)[:, None]
        candidate = ((1.0 - tem_recur) * carry + tem_recur * (environment[offset + 1] + environment[offset]))
        recur = _divide(mask, candidate, 1.0 + tem_recur, old[0][offset + 1])
        original = torch.where(mask, recur, old[1][offset + 1])
        chem = fscav * recur
        tem = chem / (1.0 + c0t[offset + 1] * dz)[:, None]
        # Source eta uses the preceding level; c0t uses the current level.
        chem_pw = (c0t[offset + 1] * dz)[:, None] * tem * eta[offset, :, None]
        candidate = tem + recur - chem
        carry = torch.where(mask, candidate, recur)
        values = (carry, original, torch.where(mask, chem, old[2][offset + 1]), torch.where(mask, chem_pw, old[3][offset + 1]))
        for output, value in zip(outputs, values):
            output.append(value)
    return tuple(_stack_levels(values) for values in outputs)


def downdraft_heat(active, jmin, kd94, first, profile_rows, carry, xlamdet, xlamddt, xlamd, pgcon):
    z, h, hs, u, v, hcdo, ucdo, vcdo, dbyo = profile_rows.unbind(1)
    width = z.shape[0] - 1
    level = _levels(first, width)
    mask = active & (level < jmin)
    hs = hs[:-1]
    old_rows = tuple(x[:-1] for x in (hcdo, ucdo, vcdo, dbyo))
    dz = z[1:] - z[:-1]
    tem = xlamdet * dz
    tem1 = torch.where(level >= kd94, 0.5 * xlamddt * dz, 0.5 * (xlamd + xlamddt) * dz)
    heat_keep = 1.0 - tem1
    heat_source = tem * 0.5 * (h[:-1] + h[1:])
    heat_denominator = torch.where(mask, 1.0 + tem - tem1, 1.0)
    tem_mom = 0.5 * tem
    momentum_denominator = torch.where(mask, 1.0 + tem_mom, 1.0)
    ptem, ptem1 = tem_mom - pgcon, tem_mom + pgcon
    momentum_keep = 1.0 - tem_mom
    u_upper, u_lower = ptem * u[1:], ptem1 * u[:-1]
    v_upper, v_lower = ptem * v[1:], ptem1 * v[:-1]
    outputs = [[None] * width for _ in range(4)]
    hnext, unext, vnext = carry
    for offset in range(width - 1, -1, -1):
        m = mask[offset]
        hnext = torch.where(m, (heat_keep[offset] * hnext + heat_source[offset]) / heat_denominator[offset], old_rows[0][offset])
        dby = torch.where(m, hnext - hs[offset], old_rows[3][offset])
        unext = torch.where(m, (momentum_keep[offset] * unext + u_upper[offset] + u_lower[offset]) / momentum_denominator[offset], old_rows[1][offset])
        vnext = torch.where(m, (momentum_keep[offset] * vnext + v_upper[offset] + v_lower[offset]) / momentum_denominator[offset], old_rows[2][offset])
        for values, value in zip(outputs, (hnext, unext, vnext, dby)):
            values[offset] = value
    return tuple(_stack_levels(values) for values in outputs)


def tracer_down(active, jmin, first, profile_rows, tracer_rows, carry, xlamdet):
    z = profile_rows[:, 0]
    env, old = tracer_rows.unbind(1)
    width = z.shape[0] - 1
    level = _levels(first, width)
    mask = (active & (level < jmin))[:, :, None]
    dz = z[1:] - z[:-1]
    tem = (0.5 * xlamdet * dz)[:, :, None]
    keep, source = 1.0 - tem, tem * (env[:-1] + env[1:])
    denominator = torch.where(mask, 1.0 + tem, 1.0)
    old_rows = old[:-1]
    values = [None] * width
    for offset in range(width - 1, -1, -1):
        carry = torch.where(mask[offset], (keep[offset] * carry + source[offset]) / denominator[offset], old_rows[offset])
        values[offset] = carry
    return _stack_levels(values)


def downdraft_moisture(active, jmin, first, profile_rows, carry, pwevo, xlamdet, el2orc, hvap):
    # The carried value is the previous (upper) level's saturation moisture, which does
    # not depend on the recurrence; only the ordered pwevo sum loops.
    z, q_sat, to, dbyo, q, etad, qrcdo, qcdo, pwdo = profile_rows.unbind(1)
    width = z.shape[0] - 1
    level = _levels(first, width)
    mask = active & (level < jmin)
    q_sat = q_sat[:-1]
    temperature = torch.where(mask, to[:-1], 1.0)
    gamma = el2orc * q_sat / (temperature * temperature)
    hvap = torch.as_tensor(hvap, dtype=q_sat.dtype, device=q_sat.device)
    candidate = q_sat + torch.reciprocal(hvap) * (gamma / (1.0 + gamma)) * dbyo[:-1]
    current = torch.where(mask, candidate, qrcdo[:-1])
    previous = torch.stack((*current.unbind(0)[1:], carry), dim=0)
    dz = z[1:] - z[:-1]
    tem = 0.5 * xlamdet * dz
    candidate = ((1.0 - tem) * previous + tem * (q[:-1] + q[1:]))
    moisture = torch.where(mask, candidate / torch.where(mask, 1.0 + tem, 1.0), qcdo[:-1])
    precipitation = etad[:-1] * (moisture - current)
    pwevo = masked_sum(pwevo, mask, precipitation, descending=True)
    outputs = (current, moisture, torch.where(mask, precipitation, pwdo[:-1]))
    return tuple(value.movedim(0, 1) for value in outputs), pwevo


def downdraft_work(active, jmin, first, profile_rows, edto, aa1, el2orc, grav, cp, fv, hvap):
    z, q_sat, to, hcdo, heso, qo = profile_rows.unbind(1)
    width = z.shape[0] - 1
    level = _levels(first, width)
    mask = active & (level < jmin)
    q_sat = q_sat[:-1]
    temperature = torch.where(mask, to[:-1], 1.0)
    gamma = el2orc * q_sat / (temperature * temperature)
    dz = z[:-1] - z[1:]
    heat = edto * dz * (grav / (cp * temperature)) * ((hcdo[:-1] - heso[:-1]) / (1.0 + gamma)) * (1.0 + fv * cp * gamma * temperature / hvap)
    moisture = edto * dz * grav * fv * torch.maximum(torch.zeros_like(edto), q_sat - qo[:-1])
    return masked_sum(aa1, mask, heat, moisture, descending=True)


def updraft_minimum(active, kbmax, kbcon1, first, profile_rows, hmin, lmin):
    heo = profile_rows[:, 0, :]
    level = _levels(first, heo.shape[0])
    window = active & (level <= kbmax) & (level > kbcon1)
    for offset in range(heo.shape[0]):
        layer = window[offset] & (heo[offset] < hmin)
        lmin = torch.where(layer, level[offset] + 1.0, lmin)
        hmin = torch.where(layer, heo[offset], hmin)
    return hmin, lmin


def updraft_work(active, kbcon, ktcon, first, profile_rows, aa1):
    zo, buo = profile_rows.unbind(1)
    level = _levels(first, zo.shape[0] - 1)
    layer = active & (level >= kbcon) & (level < ktcon)
    dz = zo[1:] - zo[:-1]
    term = buo[:-1] * dz
    return masked_sum(aa1, layer, term)


def velocity_average(active, kbcon1, ktcon, first, profile_rows, wc, sumx):
    zi, wu2 = profile_rows.unbind(1)
    level = _levels(first, zi.shape[0] - 1)
    layer = active & (level > kbcon1) & (level < ktcon)
    dz = zi[1:] - zi[:-1]
    tem = 0.5 * (sqrt_zero_grad(wu2[1:]) + sqrt_zero_grad(wu2[:-1]))
    return masked_sum(wc, layer, tem * dz), masked_sum(sumx, layer, dz)


def precip_updraft(active, kb, ktcon, kbcon, jmin, first, profile_rows, qcko_previous, pwavo, el2orc, hvap, c1, cp, ncloud):
    (zi, qeso, to, dbyo, xlamue, qo, eta, del_, c0t, qcko, qrcko, pwo, buo,
     dellal, cnvwt, zdqca, drag, wush, xlamud, uo, vo) = profile_rows.unbind(1)
    count = qcko.shape[0] - 2
    level = _levels(first, count)
    cur, prev = slice(1, count + 1), slice(0, count)
    z, q_sat, t, dby, lam, q, et, dl, c0 = zi, qeso, to, dbyo, xlamue, qo, eta, del_, c0t
    old = tuple(x[cur] for x in (qcko, qrcko, pwo, buo, dellal, cnvwt, zdqca, drag, wush))
    layer = active & (level > kb) & (level < ktcon)
    dz = z[cur] - z[prev]
    temperature = torch.where(layer, t[cur], 1.0)
    gamma = el2orc * q_sat[cur] / (temperature * temperature)
    qrch = q_sat[cur] + gamma * dby[cur] / (hvap * (1.0 + gamma))
    tem = 0.25 * (lam[cur] + lam[prev]) * dz
    keep, source = 1.0 - tem, tem * (q[cur] + q[prev])
    denominator = torch.where(layer, 1.0 + tem, 1.0)
    above = layer & (level >= kbcon)
    eta_cur = et[cur]
    etah = 0.5 * (eta_cur + et[prev])
    dp1000 = 1000.0 * dl[cur]
    cloud_den = eta_cur + etah * (c0[cur] + c1) * dz
    plain_den = eta_cur + etah * c0[cur] * dz
    cloud_level = (ncloud > 0) & (level > jmin)
    dellal_factor = etah * c1 * dz
    pwo_factor = etah * c0[cur] * dz
    rfact = 1.0 + FV * cp * gamma * t[cur] / hvap
    buoy_heat = torch.div(GRAV, cp * temperature) * dby[cur] / (1.0 + gamma) * rfact
    buoy_moist = GRAV * FV * torch.maximum(torch.zeros_like(pwavo), q_sat[cur] - q[cur])
    drag_layer = torch.where(above, torch.maximum(lam[cur], xlamud[cur]), old[7])
    u, v = uo, vo
    shear = ((u[cur] - u[prev]) / dz) ** 2
    shear = shear + ((v[cur] - v[prev]) / dz) ** 2
    wush_layer = torch.where(above, 0.2 * sqrt_zero_grad(shear), old[8])
    raw_values, moisture_values, excess_values, values = [], [], [], []
    for offset in range(count):
        lay = layer[offset]
        raw = torch.where(lay, (keep[offset] * qcko_previous + source[offset]) / denominator[offset], old[0][offset])
        dq = eta_cur[offset] * (raw - qrch[offset])
        precip = above[offset] & (dq > 0.0)
        qlk_cloud = dq / torch.where(precip, cloud_den[offset], 1.0)
        qlk_plain = dq / torch.where(precip, plain_den[offset], 1.0)
        cloud = precip & cloud_level[offset]
        qlk = torch.where(cloud, qlk_cloud, qlk_plain)
        qcko_previous = torch.where(precip, qlk + qrch[offset], raw)
        raw_values.append(raw)
        moisture_values.append(qlk)
        excess_values.append(dq)
        values.append(qcko_previous)

    # Share the recurrence's moisture intermediates with its packed diagnostics.
    # Recomputing them splits their pullbacks before cancellation-sensitive consumers.
    raw = torch.stack(raw_values, dim=0)
    dq = torch.stack(excess_values, dim=0)
    qlk = torch.stack(moisture_values, dim=0)
    precip = above & (dq > 0.0)
    dp = torch.where(precip, dp1000, 1.0)
    cloud = precip & cloud_level
    qrcko = torch.where(layer, raw, old[1])
    dellal = torch.where(cloud, dellal_factor * qlk * GRAV / dp, old[4])
    buo = torch.where(precip, old[3] - GRAV * qlk, old[3])
    pwo_val = pwo_factor * qlk
    pwo = torch.where(precip, pwo_val, old[2])
    for offset in range(count):
        pwavo = torch.where(precip[offset], pwavo + pwo_val[offset], pwavo)
    cnvwt = torch.where(precip, etah * qlk * GRAV / dp, old[5])
    zdqca = torch.where(precip, dq / torch.where(precip, eta_cur, 1.0), old[6])
    buo = torch.where(above, buo + buoy_heat, buo)
    buo = torch.where(above, buo + buoy_moist, buo)
    diagnostics = (qrcko, pwo, buo, dellal, cnvwt, zdqca, drag_layer, wush_layer)
    return (_stack_levels(values), *(value.movedim(0, 1) for value in diagnostics)), pwavo


def overshoot_top(flg, ktcon, kmax, first, profile_rows, aa2, ktcon1, cloud_top_height, cloud_depth_half, el2orc, fv, cp, hvap, grav):
    zo, qeso, to, dbyo, zi = profile_rows.unbind(1)
    for offset in range(1, to.shape[0] - 1):
        level = first + offset - 1
        layer = flg & (level >= ktcon) & (level < kmax)
        dz1 = zo[offset + 1] - zo[offset]
        temperature = torch.where(layer, to[offset], 1.0)
        gamma = el2orc * qeso[offset] / (temperature * temperature)
        rfact = 1.0 + fv * cp * gamma * to[offset] / hvap
        aa2_term = dz1 * (grav / (cp * temperature)) * dbyo[offset] / (1.0 + gamma) * rfact
        aa2 = torch.where(layer, aa2 + aa2_term, aa2)
        above_top = zi[offset] - cloud_top_height
        stop = layer & ((aa2 < 0.0) | (above_top >= cloud_depth_half))
        ktcon1 = torch.where(stop, level, ktcon1)
        flg = torch.where(stop, torch.zeros_like(flg), flg)
    return aa2, ktcon1, flg


def overshoot_moisture(active, ktcon, ktcon1, first, profile_rows, qcko_previous, pwavo, el2orc, hvap, c1, grav, ncloud_positive):
    zi, qeso, to, dbyo, xlamue, qo, eta, del_, c0t, qcko, qrcko, dellal, pwo, cnvwt, zdqca = profile_rows.unbind(1)
    count = qcko.shape[0] - 2
    level = _levels(first, count)
    cur, prev = slice(1, count + 1), slice(0, count)
    z, q_sat, t, dby, lam, q, et, dl, c0 = zi, qeso, to, dbyo, xlamue, qo, eta, del_, c0t
    old = tuple(x[cur] for x in (qcko, qrcko, dellal, pwo, cnvwt, zdqca))
    layer = active & (level >= ktcon) & (level < ktcon1)
    dz = z[cur] - z[prev]
    temperature = torch.where(layer, t[cur], 1.0)
    gamma = el2orc * q_sat[cur] / (temperature * temperature)
    qrch = q_sat[cur] + gamma * dby[cur] / (hvap * (1.0 + gamma))
    tem = 0.25 * (lam[cur] + lam[prev]) * dz
    keep, source = 1.0 - tem, tem * (q[cur] + q[prev])
    denominator = torch.where(layer, 1.0 + tem, 1.0)
    eta_cur = et[cur]
    etah = 0.5 * (eta_cur + et[prev])
    dp1000 = 1000.0 * dl[cur]
    # The source evaluates both forms; only the selected one reaches the outputs.
    qlk_den = eta_cur + etah * (c0[cur] + c1) * dz if ncloud_positive else eta_cur + etah * c0[cur] * dz
    dellal_factor = etah * c1 * dz
    pwo_factor = etah * c0[cur] * dz
    outputs = [[] for _ in range(6)]
    for offset in range(count):
        lay = layer[offset]
        qcko_layer = torch.where(lay, (keep[offset] * qcko_previous + source[offset]) / denominator[offset], old[0][offset])
        qrcko_layer = torch.where(lay, qcko_layer, old[1][offset])
        dq = eta_cur[offset] * (qcko_layer - qrch[offset])
        precip = lay & (dq > 0.0)
        dp = torch.where(precip, dp1000[offset], 1.0)
        qlk = dq / torch.where(precip, qlk_den[offset], 1.0)
        dellal_layer = torch.where(precip & ncloud_positive, dellal_factor[offset] * qlk * grav / dp, old[2][offset])
        qcko_layer = torch.where(precip, qlk + qrch[offset], qcko_layer)
        pwo_val = pwo_factor[offset] * qlk
        pwo_layer = torch.where(precip, pwo_val, old[3][offset])
        pwavo = torch.where(precip, pwavo + pwo_val, pwavo)
        cnvwt_layer = torch.where(precip, etah[offset] * qlk * grav / dp, old[4][offset])
        zdqca_layer = torch.where(precip, dq / torch.where(precip, eta_cur[offset], 1.0), old[5][offset])
        for values, value in zip(outputs, (qcko_layer, qrcko_layer, dellal_layer, pwo_layer, cnvwt_layer, zdqca_layer)):
            values.append(value)
        qcko_previous = qcko_layer
    return tuple(_stack_levels(values) for values in outputs), pwavo


def updraft_velocity(active, kbcon1, ktcon, first, profile_rows, wu2_previous):
    z, dr, b, wu, old = profile_rows.unbind(1)
    count = old.shape[0] - 2
    level = _levels(first, count)
    cur, prev = slice(1, count + 1), slice(0, count)
    layer = active & (level > kbcon1) & (level < ktcon)
    dz = z[cur] - z[prev]
    tem = 0.25 * 4.0 * (dr[prev] + dr[cur]) * dz
    tem1 = 0.5 * 0.8 * (b[prev] + b[cur])
    keep, denominator = 1.0 - tem, 1.0 + tem
    zero = torch.zeros_like(wu2_previous)
    values = []
    for offset in range(count):
        tem2 = wu[cur][offset] * sqrt_zero_grad(wu2_previous)
        tem2 = (tem1[offset] - tem2) * dz[offset]
        ptem = keep[offset] * wu2_previous
        candidate = torch.maximum((ptem + tem2) / denominator[offset], zero)
        wu2_layer = torch.where(layer[offset], candidate, old[cur][offset])
        values.append(wu2_layer)
        wu2_previous = wu2_layer
    return _stack_levels(values)


def rain_total(active, kmax, ktcon, kb, jmin, first, profile_rows, edto, xmb, delt, rntot):
    pwo, pwdo = profile_rows.unbind(1)
    width = pwo.shape[0]
    level = _levels(first, width)
    layer = active & (level <= kmax) & (level < ktcon)
    aup = (level > kb).to(pwo.dtype)  # 0 at or below kb, else 1; float64 like the products it scales
    adw = (level < jmin).to(pwo.dtype)
    rain = aup * pwo + adw * edto * pwdo
    increment = rain * xmb * 0.001 * delt
    return masked_sum(rntot, layer, increment, descending=True)


def evaporation(active, kmax, ktcon, kb, jmin, first, profile_rows, carry, rntot, edto, xmb, delt, evef, el2orc, grav, elocp):
    del_, q1, t1, q_sat, pwo, pwdo = profile_rows.unbind(1)
    rn, qevap, delqev, delq2, qcond, delqbar, deltbar, flg = carry
    width = q1.shape[0]
    level = _levels(first, width)
    in_domain = level <= kmax
    below_top = in_domain & (level < ktcon)
    active_rain = active & below_top
    aup = (level > kb).to(pwo.dtype)  # 0 at or below kb, else 1; float64 like the products it scales
    adw = (level < jmin).to(pwo.dtype)
    rain = (aup * pwo + adw * edto * pwdo) * xmb * 0.001 * delt
    qcond_value = evef * (q1 - q_sat) / (1.0 + el2orc * q_sat / (t1 * t1))
    dp = 1000.0 * del_
    tem = grav / dp
    tem1 = dp / grav
    cap_scale = 1000.0 * tem
    qvalues, tvalues = [None] * width, [None] * width
    for offset in range(width - 1, -1, -1):
        qevap = torch.where(in_domain[offset], 0.0, qevap)
        rn = torch.where(active_rain[offset], rn + rain[offset], rn)
        evap_layer = flg & below_top[offset]
        qcond = torch.where(evap_layer, qcond_value[offset], qcond)
        possible = evap_layer & (rn > 0.0) & (qcond < 0.0)
        sqrt_arg = torch.where(possible, delt * rn, 1.0)
        qevap_candidate = -qcond * (1.0 - torch.exp(-0.32 * torch.sqrt(sqrt_arg)))
        qevap_candidate = torch.minimum(qevap_candidate, rn * 1000.0 * tem[offset])
        qevap = torch.where(possible, qevap_candidate, qevap)
        delq2 = torch.where(possible, delqev + 0.001 * qevap * tem1[offset], delq2)
        cap = possible & (delq2 > rntot)
        qevap = torch.where(cap, cap_scale[offset] * (rntot - delqev), qevap)
        flg = torch.where(cap, False, flg)
        apply_evap = evap_layer & (rn > 0.0) & (qevap > 0.0)
        qvalues[offset] = torch.where(apply_evap, q1[offset] + qevap, q1[offset])
        tvalues[offset] = torch.where(apply_evap, t1[offset] - elocp * qevap, t1[offset])
        removed = 0.001 * qevap * tem1[offset]
        rn = torch.where(apply_evap, rn - removed, rn)
        deltv = torch.where(apply_evap, -elocp * qevap / delt, 0.0)
        delq = torch.where(apply_evap, qevap / delt, 0.0)
        delqev = torch.where(apply_evap, delqev + removed, delqev)
        delqbar = torch.where(evap_layer, delqbar + delq * tem1[offset], delqbar)
        deltbar = torch.where(evap_layer, deltbar + deltv * tem1[offset], deltbar)
    return (_stack_levels(qvalues), _stack_levels(tvalues)), (rn, qevap, delqev, delq2, qcond, delqbar, deltbar, flg)
