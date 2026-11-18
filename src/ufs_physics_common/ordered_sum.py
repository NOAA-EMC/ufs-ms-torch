# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Masked running sums whose order sets only the rounding of the result.

A column total accumulated over selected levels, ``where(mask[k], total + term[k], total)``,
is one masked sum. The source schemes add the levels in sequence, and the default keeps that
order: this is the form the Fortran oracle gates compare bit for bit. Each unrolled level adds
code to one thread per column, which the kernel then has to execute level by level.
"""

from __future__ import annotations

import torch

def level_sum(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Sum along ``dim`` in index order from +0.0: ``((0 + v0) + v1) + ...``.

    The source's loop order (``s = 0; do k: s = s + v(k)``), equal to ATen's CPU ``cumsum(dim)`` at the last
    index. CUDA ``cumsum`` is a tree scan (``tensor_kernel_scan_innermost_dim``) and Inductor's scan an
    associative one, so neither keeps this order on a GPU. Unrolled, the terms fuse into one kernel.
    """
    terms = values.unbind(dim)
    if not terms:
        return values.sum(dim=dim)
    total = torch.zeros_like(terms[0])  # +0.0, as the source and ATen's CPU scan: -0.0 terms give +0.0
    for term in terms:
        total = total + term
    return total


def level_prefix(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Inclusive running sums along ``dim`` in index order from +0.0 (``level_sum`` at every index); equal to
    ATen's CPU ``cumsum(dim)``."""
    terms = values.unbind(dim)
    if not terms:
        return values.clone()
    running, out = torch.zeros_like(terms[0]), []
    for term in terms:
        running = running + term
        out.append(running)
    return torch.stack(out, dim=dim)


def level_product_prefix(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Inclusive running products along ``dim`` in index order from 1.0; equal to ATen's CPU ``cumprod(dim)``."""
    terms = values.unbind(dim)
    if not terms:
        return values.clone()
    running, out = torch.ones_like(terms[0]), []
    for term in terms:
        running = running * term
        out.append(running)
    return torch.stack(out, dim=dim)


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
    rows = range(mask.shape[0])
    for row in reversed(rows) if descending else rows:
        for term in terms:
            total = torch.where(mask[row], total + term[row], total)
    return total
