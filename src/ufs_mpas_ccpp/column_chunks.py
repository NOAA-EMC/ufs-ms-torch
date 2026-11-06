# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Equal column chunks for column-local kernels: bounded memory and checkpointed reverse mode.

The kernels run here (the radiation mains, the TEMPO core) compute each column from that
column alone, so the chunked result equals one call. Chunks have one size, at most the
requested size and balanced over the chunks: the last repeats the final column up to it (fewer
columns than there are chunks), and that padding is discarded (it receives no cotangent), so a
compiled kernel builds one graph. Under grad each chunk is a non-reentrant checkpoint: only
its inputs stay alive and backward recomputes it, one chunk at a time. A compiled kernel is
warmed once outside the checkpoints, on a slice of WARM_COLUMNS columns, so every chunk (forward
and recompute) runs the same compiled region.
An output that every chunk returns as its own unmodified input (the same tensor, version counter
unchanged) is the whole input, not a concatenated copy: a scheme that passes most of a container
through then adds only the fields it computes.

With ``views=True`` each chunk reads a window of rows as ``narrow`` views (no copy) where the view is
contiguous; a non-contiguous view (column-last arrays, strided fields) is copied to the contiguous
layout ``index_select`` gives. Instead of padding, the last window ends at the last column and
overlaps the one before it; the join keeps the overlapped rows of the earlier window (the kernels
are column-local, so both computed the same values). Only the fields a chunk reads cost bytes.
The kernel must not write its inputs in place (a view would carry the write into the caller's
tensor, and a recompute under grad would repeat it): a changed version counter raises.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint
from ufs_physics_common.compaction import overflow_flag
from ufs_physics_common.compile_policy import eager_regions, eager_regions_active


def take(value: Any, index: torch.Tensor, *, axis: int, ncol: int) -> Any:
    """Select ``index`` along ``axis`` of every tensor in a tree whose ``axis`` extent is ``ncol``."""
    return _map_columns(value, lambda t: t.index_select(axis, index), axis=axis, ncol=ncol)


def rows_view(value: Any, start: int, size: int, *, axis: int, ncol: int) -> Any:
    """Rows ``start:start + size`` along ``axis`` of every ``ncol``-extent tensor in a tree, in the layout
    ``take`` gives: a ``narrow`` view where that is contiguous (no copy), else a contiguous copy."""

    def rows(t: torch.Tensor) -> torch.Tensor:
        part = t.narrow(axis, start, size)
        return part if part.is_contiguous() else part.contiguous()

    return _map_columns(value, rows, axis=axis, ncol=ncol)


def _map_columns(value: Any, select: Callable[[torch.Tensor], torch.Tensor], *, axis: int, ncol: int,
                 memo: dict[int, torch.Tensor] | None = None) -> Any:
    if memo is None:
        memo = {}
    if isinstance(value, torch.Tensor):
        if not (value.ndim and value.shape[axis] == ncol):
            return value
        if id(value) not in memo:
            memo[id(value)] = select(value)
        return memo[id(value)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, **{f.name: _map_columns(getattr(value, f.name), select, axis=axis, ncol=ncol, memo=memo)
                                             for f in dataclasses.fields(value) if f.init})
    if isinstance(value, Mapping):
        return {k: _map_columns(v, select, axis=axis, ncol=ncol, memo=memo) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_map_columns(v, select, axis=axis, ncol=ncol, memo=memo) for v in value)
    return value


class _Passed:
    """A chunk output that is the chunk's own unmodified input ``whole[rows]``."""

    __slots__ = ("whole",)

    def __init__(self, whole: torch.Tensor):
        self.whole = whole


def _inputs(whole: Any, part: Any, out: dict) -> dict:
    """id of each chunk input tensor -> (whole tensor, version after the selection)."""
    if isinstance(part, torch.Tensor):
        if part is not whole:
            out[id(part)] = (whole, part._version)
    elif dataclasses.is_dataclass(part) and not isinstance(part, type):
        for f in dataclasses.fields(part):
            if f.init:
                _inputs(getattr(whole, f.name), getattr(part, f.name), out)
    elif isinstance(part, Mapping):
        for k, v in part.items():
            _inputs(whole[k], v, out)
    elif isinstance(part, (list, tuple)):
        for w, v in zip(whole, part):
            _inputs(w, v, out)
    return out


def _mark_passed(value: Any, inputs: dict) -> Any:
    if isinstance(value, torch.Tensor):
        whole, version = inputs.get(id(value), (None, None))
        return _Passed(whole) if whole is not None and value._version == version else value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, **{f.name: _mark_passed(getattr(value, f.name), inputs)
                                             for f in dataclasses.fields(value) if f.init})
    if isinstance(value, Mapping):
        return {k: _mark_passed(v, inputs) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_mark_passed(v, inputs) for v in value)
    return value


def join(parts: list, *, axis: int, size: int, ncol: int, rows: torch.Tensor | None = None,
         starts: list[int] | None = None) -> Any:
    """Concatenate chunk outputs along ``axis`` (extent ``size``) and drop the padding.

    With ``starts`` (row windows, ``views``) each chunk adds the rows past the previous window's end.
    """
    if any(isinstance(p, _Passed) for p in parts):
        if all(isinstance(p, _Passed) and p.whole is parts[0].whole for p in parts):
            return parts[0].whole
        parts = [(p.whole.narrow(axis, starts[i], size) if starts is not None
                  else p.whole.index_select(axis, rows[i * size:(i + 1) * size])) if isinstance(p, _Passed) else p
                 for i, p in enumerate(parts)]
    first = parts[0]
    if isinstance(first, torch.Tensor):
        if first.ndim and first.shape[axis] == size:
            if starts is None:
                return torch.cat(parts, dim=axis).narrow(axis, 0, ncol)
            skip = [0] + [max(starts[i - 1] + size - starts[i], 0) for i in range(1, len(parts))]
            return torch.cat([p if k == 0 else p.narrow(axis, k, size - k) for p, k in zip(parts, skip)], dim=axis)
        return first  # column-independent output (e.g. solar constants)
    options = dict(axis=axis, size=size, ncol=ncol, rows=rows, starts=starts)
    if dataclasses.is_dataclass(first) and not isinstance(first, type):
        return dataclasses.replace(first, **{f.name: join([getattr(p, f.name) for p in parts], **options)
                                             for f in dataclasses.fields(first) if f.init})
    if isinstance(first, Mapping):
        return {k: join([p[k] for p in parts], **options) for k in first}
    if isinstance(first, (list, tuple)):
        return type(first)(join([p[i] for p in parts], **options) for i in range(len(first)))
    if any(p != first for p in parts[1:]):
        raise ValueError("column chunks disagree on a static output")
    return first


# Columns of the eager first call that primes a compiled chunk kernel (run_column_chunks ``warm``).
# 1024 = 2 x compaction.MIN_COLUMNS: at every capacity fraction below 1 a compacted region of the slice runs
# its compacted path, as a whole chunk does, so the warm call registers the same overflow-region bits that
# compiled regions read (64 columns ran dense and left them unset: convection and TEMPO recompiled).
WARM_COLUMNS = 1024


def _join_piece(result, part, *, first, axis, size, ncol, destination, offset, count):
    """Copy one piece into its final storage, without retaining earlier outputs.

    copy_ records both AD modes. Only fresh destination storage is written;
    carried inputs stay immutable. Passed-through fields need no storage at all.
    """
    if isinstance(part, _Passed):
        if first:
            return part
        if isinstance(result, _Passed) and result.whole is part.whole:
            return result
        part = part.whole.narrow(axis, destination - offset, size)
    if isinstance(part, torch.Tensor):
        if not (part.ndim and part.shape[axis] == size):
            return part if first else result
        if first or isinstance(result, _Passed):
            shape = list(part.shape)
            shape[axis] = ncol
            output = part.new_empty(shape)
            if not first and destination:
                output.narrow(axis, 0, destination).copy_(result.whole.narrow(axis, 0, destination))
            result = output
        result.narrow(axis, destination, count).copy_(part.narrow(axis, offset, count))
        return result
    options = dict(first=first, axis=axis, size=size, ncol=ncol,
                   destination=destination, offset=offset, count=count)
    if dataclasses.is_dataclass(part) and not isinstance(part, type):
        return dataclasses.replace(part, **{
            f.name: _join_piece(None if first else getattr(result, f.name), getattr(part, f.name), **options)
            for f in dataclasses.fields(part) if f.init})
    if isinstance(part, Mapping):
        return {k: _join_piece(None if first else result[k], v, **options) for k, v in part.items()}
    if isinstance(part, (list, tuple)):
        return type(part)(_join_piece(None if first else result[i], v, **options) for i, v in enumerate(part))
    if not first and part != result:
        raise ValueError("column chunks disagree on a static output")
    return part


def chunk_size(ncol: int, size: int) -> int:
    """Columns per chunk for ``ncol`` columns and at most ``size`` per chunk: balanced over the chunks."""
    count = -(-ncol // size)
    return -(-ncol // count)


def run_column_chunks(call: Callable[[Any], Any], columns: Any, *, ncol: int, size: int, axis: int,
                      warm: Callable[[Any], Any] | None = None, views: bool = False,
                      warm_columns: int | None = WARM_COLUMNS) -> Any:
    """``call(columns)`` over equal chunks of ``size`` columns; ``warm`` primes a compiled ``call``.

    ``views``: full chunks read their rows as views (module docstring); ``call`` must not write its inputs.
    ``warm_columns``: columns of the warm call (None: a whole chunk, for a kernel whose compiled regions
    read state that only a whole chunk sets, e.g. TEMPO's nested compaction tiers).
    """
    if ncol <= size:
        return call(columns)
    device = next(t for t in _tensors(columns) if t.ndim and t.shape[axis] == ncol).device
    size = chunk_size(ncol, size)  # balanced: the padding is below the chunk count
    padded = -(-ncol // size) * size
    rows = torch.clamp(torch.arange(padded, device=device), max=ncol - 1)

    starts = list(range(0, padded, size))
    if views:
        starts[-1] = ncol - size  # the last window ends at the last column instead of padding past it
    eager = eager_regions_active()

    def piece(start: int) -> Any:
        if views:
            part = rows_view(columns, start, size, axis=axis, ncol=ncol)
        else:
            part = take(columns, rows[start:start + size], axis=axis, ncol=ncol)
        inputs = _inputs(columns, part, {})  # versions before the call
        # Backward can replay this piece after the cold step's eager scope ends.
        with eager_regions(eager):
            out = call(part)
        if views and any(t._version != inputs[id(t)][1] for t in _tensors(part) if id(t) in inputs):
            raise RuntimeError("a column-chunk kernel wrote one of its inputs in place (views=True forbids it)")
        return _mark_passed(out, inputs)

    grad = torch.is_grad_enabled()
    if warm is not None and not eager_regions_active():
        # The eager first call only fills caches; on WARM_COLUMNS columns it does not set the memory
        # peak, as a whole eager chunk did. A slice may overflow a compaction buffer the chunks fit:
        # its overflow bits are discarded (the flag is put back), so it never forces a rerun.
        count = size if warm_columns is None else min(warm_columns, size)
        flag = overflow_flag(device)
        before = flag.clone()
        with torch.no_grad():
            warm(rows_view(columns, 0, count, axis=axis, ncol=ncol) if views
                 else take(columns, rows[:count], axis=axis, ncol=ncol))
        flag.copy_(before)
    result = None
    for index, start in enumerate(starts):
        part = checkpoint(piece, start, use_reentrant=False) if grad else piece(start)
        offset = max(starts[index - 1] + size - start, 0) if views and index else 0
        destination = start + offset
        count = min(size - offset, ncol - destination)
        result = _join_piece(result, part, first=index == 0, axis=axis, size=size, ncol=ncol,
                             destination=destination, offset=offset, count=count)
        del part
    return join([result], axis=axis, size=size, ncol=ncol)


def _tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            yield from _tensors(getattr(value, f.name))
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from _tensors(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _tensors(v)
