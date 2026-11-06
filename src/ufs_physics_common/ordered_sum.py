# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Masked running sums whose order sets only the rounding of the result.

A column total accumulated over selected levels, ``where(mask[k], total + term[k], total)``,
is one masked sum. The source schemes add the levels in sequence, and the default keeps that
order: this is the form the Fortran oracle gates compare bit for bit. Each unrolled level adds
code to one thread per column, which the kernel then has to execute level by level.

``set_parallel_masked_sums(True)`` selects one reduction over the levels. The value differs from
the sequential sum by at most ``gamma(n) * sum(|terms|)``, with n the number of added terms: the
equations and every term are unchanged, only the rounding of the sum. The Fortran comparisons
are exact, so they apply to the default only. Terms must not read the total.
"""

from __future__ import annotations

import torch

_parallel_masked_sums = False


def set_parallel_masked_sums(enabled: bool) -> bool:
    """Select the reduction form; returns the previous setting. Set it before the first compile."""
    global _parallel_masked_sums
    previous, _parallel_masked_sums = _parallel_masked_sums, bool(enabled)
    return previous


def parallel_masked_sums() -> bool:
    return _parallel_masked_sums


def masked_sum(
    total: torch.Tensor,
    mask: torch.Tensor,
    *terms: torch.Tensor,
    descending: bool = False,
) -> torch.Tensor:
    """``total`` plus the ``terms`` rows that ``mask`` selects; rows are the first dimension.

    Row by row each term of ``terms`` is added in turn, in row order or, with ``descending``,
    in reverse row order.
    """
    if _parallel_masked_sums:
        for term in terms:
            total = total + torch.where(mask, term, 0.0).sum(dim=0)
        return total
    rows = range(mask.shape[0])
    for row in reversed(rows) if descending else rows:
        for term in terms:
            total = torch.where(mask[row], total + term[row], total)
    return total
