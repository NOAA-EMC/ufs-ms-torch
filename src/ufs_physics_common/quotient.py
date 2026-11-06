# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""``n / d`` with derivatives evaluated in a range-safe order, for quotients of possibly subnormal operands.

The forward is the ordinary division (same bits). PyTorch's division backward forms ``-g * n / d**2``
(or ``(n / d) / d``): for d near 1e-316, as with sums of tracer round-off, ``d**2`` underflows or
``(n / d) / d`` overflows and the derivative becomes inf or NaN, although the chain-rule product
``(g / d) * (n / d)`` is O(1) when the upstream gradient ``g`` is as small as the operands. This
Function evaluates exactly that product: d(n/d)/dn = 1/d, d(n/d)/dd = -(n/d)/d, applied as
``g / d`` and ``-(g / d) * q`` in reverse mode and ``(t_n - q * t_d) / d`` in forward mode.

Dynamo (torch 2.8) does not trace an autograd.Function that defines ``jvp`` ("Unsupported custom
jvp"), so the forward-mode rule lives in a subclass used only under an active forward-AD level;
compiled regions run eagerly there (``compile_policy.compiled``). Both classes give the same bits.
"""

from __future__ import annotations

from typing import Any

import torch


class _Quotient(torch.autograd.Function):
    @staticmethod
    def forward(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return numerator / denominator

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor], output: torch.Tensor) -> None:
        numerator, denominator = inputs
        ctx.shapes = (numerator.shape, denominator.shape)
        ctx.save_for_backward(denominator, output)
        ctx.save_for_forward(denominator, output)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        denominator, quotient_value = ctx.saved_tensors
        scaled = grad / denominator
        grad_numerator = scaled.sum_to_size(ctx.shapes[0]) if ctx.needs_input_grad[0] else None
        grad_denominator = (-(scaled * quotient_value)).sum_to_size(ctx.shapes[1]) if ctx.needs_input_grad[1] else None
        return grad_numerator, grad_denominator


class _DualQuotient(_Quotient):
    @staticmethod
    def jvp(ctx: Any, tangent_numerator: torch.Tensor | None, tangent_denominator: torch.Tensor | None) -> torch.Tensor:
        denominator, quotient_value = ctx.saved_tensors
        change = torch.zeros_like(quotient_value) if tangent_numerator is None else tangent_numerator
        if tangent_denominator is not None:
            change = change - quotient_value * tangent_denominator
        return change / denominator


def quotient(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """``numerator / denominator`` (same forward bits) with range-safe reverse and forward derivatives."""
    if torch.autograd.forward_ad._current_level >= 0:
        return _DualQuotient.apply(numerator, denominator)
    return _Quotient.apply(numerator, denominator)
