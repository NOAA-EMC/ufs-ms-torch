# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Exact column compaction with a recoverable overflow (coupled GPU plan, direction D3).

A compacted region runs part of a scheme only on the columns that need it: a stable ``argsort``
gathers them into a buffer of static capacity, the region runs there, and the results are copied
back. Results equal the dense code bit for bit as long as the buffer holds every column that
needs it. When it does not, the region does not stop the run: it sets its bit in one device-side
overflow flag. The driver reads the flag once per step; on overflow it moves that region up the
capacity ladder and reruns the step from its unchanged input, so the kept result is again exact.

Capacities are fractions of the column count, at least ``MIN_COLUMNS``. A region starts at a
validated fraction and steps up to the next ``LADDER`` value above it, so a region compiles at
most a few distinct shapes.
"""

from __future__ import annotations

import math

import torch
from torch._functorch.pyfunctorch import temporarily_clear_interpreter_stack

LADDER = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
MIN_COLUMNS = 512

_flags: dict[torch.device, torch.Tensor] = {}
_bits: dict[str, int] = {}


def capacity(ncol: int, fraction: float) -> int:
    """Static buffer size for ``fraction`` of ``ncol`` columns."""
    return max(math.ceil(fraction * ncol), MIN_COLUMNS)


def rung_up(fraction: float) -> float:
    """The next ladder fraction above ``fraction`` (1.0 stays 1.0)."""
    return next((rung for rung in LADDER if rung > fraction), 1.0)


def region_bit(name: str) -> int:
    """Stable bit of a named compacted region (at most 63 regions)."""
    if name not in _bits:
        if len(_bits) >= 63:
            raise RuntimeError("too many compacted regions for one overflow flag")
        _bits[name] = len(_bits)
    return _bits[name]


def overflow_flag(device: torch.device | str) -> torch.Tensor:
    """The persistent int64 overflow flag of ``device`` (one bit per region)."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "mps" and device.index is None:
        device = torch.device("mps", 0)
    flag = _flags.get(device)
    if flag is None:
        flag = _flags[device] = torch.zeros((), dtype=torch.int64, device=device)
    return flag


def note_overflow(name: str, overflow: torch.Tensor) -> None:
    """OR ``overflow`` (a bool scalar) into region ``name``'s bit of the device flag.

    Inside a function transform (``torch.func`` jvp, vjp, grad, vmap) the flag is a captured
    tensor that the transform may not mutate. The overflow condition has no derivative, so
    its plain value (under vmap: any batch member) goes into the flag outside the
    transform's interpreter stack, on the device, without a host read.
    """
    bit = 1 << region_bit(name)
    if (
        not torch.compiler.is_compiling()
        and torch._C._functorch.maybe_current_level() is not None
    ):
        while torch._C._functorch.is_functorch_wrapped_tensor(overflow):
            overflow = torch._C._functorch.get_unwrapped(overflow)
        with temporarily_clear_interpreter_stack():
            overflow_flag(overflow.device).bitwise_or_(
                overflow.any().to(torch.int64) * bit
            )
        return
    flag = overflow_flag(overflow.device)
    flag.bitwise_or_(overflow.to(torch.int64) * bit)


def reset_overflow(device: torch.device | str) -> None:
    overflow_flag(device).zero_()


def overflowed_regions(device: torch.device | str) -> tuple[str, ...]:
    """Regions whose buffer overflowed since the last reset (one host read; drivers only)."""
    value = int(overflow_flag(device))
    return tuple(name for name, bit in _bits.items() if value >> bit & 1)


def _compact_select(values, axes, order):
    return tuple(value.index_select(axis, order) for value, axis in zip(values, axes))


def _compact_merge(originals, updates, axes, order, keep):
    result = []
    for original, update, axis in zip(originals, updates, axes):
        shape = [1] * update.ndim
        shape[axis] = order.shape[0]
        old = original.index_select(axis, order)
        result.append(original.contiguous().index_copy(
            axis, order, torch.where(keep.view(shape), update, old)))
    return tuple(result)


def run_compacted(run, columns: dict[str, torch.Tensor], need: torch.Tensor, *, region: str,
                  fraction: float, axes: dict[str, int] | None = None,
                  tensor_call=None, inputs: tuple[str, ...] | None = None) -> dict[str, torch.Tensor]:
    """Run ``run(columns)`` only on the columns where ``need`` is true; return the merged columns.

    ``columns`` maps names to tensors with a column axis (``axes[name]``, default 0); ``run``
    returns a subset of them, updated. Columns outside ``need`` keep their inputs. The buffer
    holds ``capacity(ncol, fraction)`` columns: the needed ones first (stable order), then filler
    columns whose results are discarded. With ``fraction`` 1.0 (or a buffer as large as the
    columns) ``run`` sees every column, as the dense code. ``tensor_call(fn, *args)`` optionally
    dispatches the read-only gather and merge through the caller's existing tensor regions.
    ``inputs`` narrows the gathered contract; other columns remain available only as merge bases.
    """
    axes = {name: (axes[name] if axes is not None and name in axes else 0) for name in columns}
    ncol = need.shape[0]
    size = capacity(ncol, fraction)
    names = tuple(columns) if inputs is None else inputs
    if size >= ncol:
        return {**columns, **run({name: columns[name] for name in names})}
    note_overflow(region, need.sum() > size)
    order = torch.argsort(need.to(torch.int8), descending=True, stable=True)[:size]
    keep = need.index_select(0, order)
    call = tensor_call if tensor_call is not None else lambda fn, *args: fn(*args)
    selected = call(_compact_select, tuple(columns[name] for name in names),
                    tuple(axes[name] for name in names), order)
    part = run(dict(zip(names, selected)))
    merged = dict(columns)
    # Keep originals separate from the gathered buffers: a scheme can mutate those buffers.
    replacements = call(_compact_merge, tuple(columns[name] for name in part),
                        tuple(part.values()), tuple(axes[name] for name in part), order, keep)
    merged.update(zip(part, replacements))
    return merged
