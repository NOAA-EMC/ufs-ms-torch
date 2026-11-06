# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Plans of ordered index sums for devices without ``segment_reduce`` (host preparation).

A plan is built once per static index set and cached by ``stencil_ops``; building it reads the
pass widths on the host. The sums themselves (``stencil_ops._ordered_segment_sum_by_plan``) read
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OrderedSumPlan:
    """Static plan of a sequential ordered index sum for devices without ``segment_reduce``.

    ``gather_index`` lists, pass by pass, the value row that each source adds: pass k
    holds the k-th term (stable index order) of every source that still has terms,
    sources ranked by term count so that pass k is the prefix ``[:widths[k]]``.
    ``slot`` maps each source to its ranked position.
    """

    gather_index: torch.Tensor
    slot: torch.Tensor
    widths: tuple[int, ...]


def build_ordered_sum_plan(indices: torch.Tensor, n_source_rows: int) -> OrderedSumPlan:
    """Plan an index set once: the host reads the pass widths here, never per call."""
    order = torch.argsort(indices, stable=True)
    sources = indices[order]
    offsets = torch.searchsorted(
        sources, torch.arange(n_source_rows + 1, device=indices.device, dtype=indices.dtype)
    )
    counts = offsets[1:] - offsets[:-1]
    by_count = torch.argsort(counts, descending=True, stable=True)
    slot = torch.empty_like(by_count)
    slot[by_count] = torch.arange(n_source_rows, device=counts.device)
    passes = int(counts.max()) if n_source_rows else 0
    steps = torch.arange(passes, device=counts.device)
    widths = (counts[by_count][None, :] > steps[:, None]).sum(1)
    rank = torch.arange(sources.numel(), device=sources.device) - offsets[sources]
    starts = torch.cumsum(widths, 0) - widths
    position = starts[rank] + slot[sources]
    gather_index = torch.empty_like(order)
    gather_index[position] = order
    return OrderedSumPlan(gather_index, slot, tuple(widths.tolist()))
