# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Custom autograd for batched tridiagonal LU factorization."""

from __future__ import annotations

from typing import Any, cast

import torch

from mpas_pytorch.autodiff.tridiagonal_scan import (
    affine_forward_scan,
    affine_reverse_scan,
    compile_active,
    mobius_prefix_scan_zero,
    tridiag_scan_preferred,
)


def _tridiag_factor_forward(
    a_tri: torch.Tensor,
    b_tri: torch.Tensor,
    c_tri: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordered column-owned sweep shared by inference and custom autograd.

    Only the completed alpha/gamma arrays are packed. This exposes the sequential
    recurrence to fusion without repeatedly functionalizing whole-array writes.
    """
    n_cells = a_tri.shape[0]
    n_levels = a_tri.shape[1]
    if n_levels == 0:
        return torch.zeros_like(a_tri), torch.zeros_like(a_tri)
    zero = a_tri.new_zeros(n_cells)
    alpha_columns = [zero]
    gamma_columns = [zero]
    for k in range(1, n_levels):
        alpha = 1.0 / (b_tri[:, k] - a_tri[:, k] * gamma_columns[k - 1])
        alpha_columns.append(alpha)
        gamma_columns.append(c_tri[:, k] * alpha)
    return torch.stack(alpha_columns, dim=1), torch.stack(gamma_columns, dim=1)


def _tridiag_factor_forward_scan(
    a_tri: torch.Tensor,
    b_tri: torch.Tensor,
    c_tri: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallel-prefix form of the Thomas factorization recurrence."""
    n_cells, n_levels = a_tri.shape
    alpha_tri = torch.zeros((n_cells, n_levels), dtype=a_tri.dtype, device=a_tri.device)
    gamma_tri = torch.zeros((n_cells, n_levels), dtype=a_tri.dtype, device=a_tri.device)
    if n_levels <= 1:
        return alpha_tri, gamma_tri

    gamma_steps = mobius_prefix_scan_zero(
        torch.zeros_like(a_tri[:, 1:n_levels]),
        c_tri[:, 1:n_levels],
        -a_tri[:, 1:n_levels],
        b_tri[:, 1:n_levels],
    )
    gamma_tri[:, 1:n_levels] = gamma_steps
    gamma_prev = gamma_steps.new_empty((n_cells, n_levels - 1))
    gamma_prev[:, :1] = gamma_tri[:, :1]
    if n_levels > 2:
        gamma_prev[:, 1:] = gamma_steps[:, :-1]
    alpha_tri[:, 1:n_levels] = 1.0 / (
        b_tri[:, 1:n_levels] - a_tri[:, 1:n_levels] * gamma_prev
    )
    return alpha_tri, gamma_tri


def _tridiag_factor_backward_scan(
    a_tri: torch.Tensor,
    c_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    gamma_tri: torch.Tensor,
    grad_alpha: torch.Tensor,
    grad_gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scan form of the adjoint recurrence for factorization."""
    _n_cells, n_levels = a_tri.shape
    grad_a = torch.zeros_like(a_tri)
    grad_b = torch.zeros_like(a_tri)
    grad_c = torch.zeros_like(a_tri)
    if n_levels <= 1:
        return grad_a, grad_b, grad_c

    alpha_tail = alpha_tri[:, 1:n_levels]
    a_tail = a_tri[:, 1:n_levels]
    c_tail = c_tri[:, 1:n_levels]
    gamma_prev = gamma_tri[:, 0 : n_levels - 1]

    reverse_linear = a_tail * alpha_tail.square() * c_tail
    reverse_bias = (
        grad_gamma[:, 0 : n_levels - 1]
        + a_tail * alpha_tail.square() * grad_alpha[:, 1:n_levels]
    )
    adj_lower = affine_reverse_scan(
        reverse_linear,
        reverse_bias,
        grad_gamma[:, n_levels - 1],
    )
    adj_gamma = grad_gamma.clone()
    adj_gamma[:, : n_levels - 1] = adj_lower

    adj_tail = adj_gamma[:, 1:n_levels]
    total_adj_alpha = grad_alpha[:, 1:n_levels] + adj_tail * c_tail
    d_adj = -total_adj_alpha * alpha_tail.square()
    grad_c[:, 1:n_levels] = adj_tail * alpha_tail
    grad_b[:, 1:n_levels] = d_adj
    grad_a[:, 1:n_levels] = -d_adj * gamma_prev
    return grad_a, grad_b, grad_c


def _tridiag_factor_jvp_scan(
    a_tri: torch.Tensor,
    c_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    gamma_tri: torch.Tensor,
    grad_a: torch.Tensor | None,
    grad_b: torch.Tensor | None,
    grad_c: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan form of the forward-mode factorization recurrence."""
    n_cells, n_levels = a_tri.shape
    da = torch.zeros_like(a_tri) if grad_a is None else grad_a
    db = torch.zeros_like(a_tri) if grad_b is None else grad_b
    dc = torch.zeros_like(a_tri) if grad_c is None else grad_c
    d_alpha = torch.zeros((n_cells, n_levels), dtype=a_tri.dtype, device=a_tri.device)
    d_gamma = torch.zeros((n_cells, n_levels), dtype=a_tri.dtype, device=a_tri.device)
    if n_levels <= 1:
        return d_alpha, d_gamma

    alpha_tail = alpha_tri[:, 1:n_levels]
    gamma_prev = gamma_tri[:, 0 : n_levels - 1]
    p = c_tri[:, 1:n_levels] * alpha_tail.square() * a_tri[:, 1:n_levels]
    q = dc[:, 1:n_levels] * alpha_tail - c_tri[:, 1:n_levels] * alpha_tail.square() * (
        db[:, 1:n_levels] - da[:, 1:n_levels] * gamma_prev
    )
    d_gamma_steps = affine_forward_scan(p, q, d_gamma[:, 0])
    d_gamma[:, 1:n_levels] = d_gamma_steps
    d_gamma_prev = d_gamma_steps.new_empty((n_cells, n_levels - 1))
    d_gamma_prev[:, :1] = d_gamma[:, :1]
    if n_levels > 2:
        d_gamma_prev[:, 1:] = d_gamma_steps[:, :-1]
    term = (
        db[:, 1:n_levels]
        - da[:, 1:n_levels] * gamma_prev
        - a_tri[:, 1:n_levels] * d_gamma_prev
    )
    d_alpha[:, 1:n_levels] = -alpha_tail.square() * term
    return d_alpha, d_gamma


def _validate_factor_inputs(
    a_tri: torch.Tensor, b_tri: torch.Tensor, c_tri: torch.Tensor
) -> None:
    if a_tri.shape != b_tri.shape or a_tri.shape != c_tri.shape:
        raise ValueError("a_tri, b_tri, c_tri must have identical shapes")
    if a_tri.ndim != 2:
        raise ValueError("expected shape (nCells, nVertLevels)")


class TridiagonalFactorize(torch.autograd.Function):
    """Factorize tridiagonal coefficients into alpha/gamma sweep arrays."""

    # Second-order derivatives (double backward):
    # 1) Keep custom Function but remove backward graph blockers (e.g., once_differentiable)
    #    and ensure backward computations stay differentiable.
    #    Pro: smallest API change. Con: requires careful backward graph auditing.
    # 2) Rewrite backward math in fully functional tensor ops (no in-place mutations),
    #    so autograd can trace backward-of-backward directly.
    #    Pro: easiest to validate with gradgradcheck. Con: may increase memory/compute.
    # 3) Introduce dedicated custom backward Functions (e.g., TridiagonalFactorizeBackward)
    #    with explicit forward/backward equations.
    #    Pro: maximal control and performance tuning. Con: highest maintenance cost.

    @staticmethod
    def forward(
        a_tri: torch.Tensor,
        b_tri: torch.Tensor,
        c_tri: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_factor_inputs(a_tri, b_tri, c_tri)

        use_scan = tridiag_scan_preferred(a_tri)
        if use_scan:
            alpha_tri, gamma_tri = _tridiag_factor_forward_scan(a_tri, b_tri, c_tri)
        else:
            alpha_tri, gamma_tri = _tridiag_factor_forward(a_tri, b_tri, c_tri)

        return alpha_tri, gamma_tri

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Save the same recurrence operands for ordinary and functional AD."""
        a_tri, _b_tri, c_tri = inputs
        alpha_tri, gamma_tri = output
        ctx.use_scan = tridiag_scan_preferred(a_tri)
        ctx.save_for_backward(a_tri, c_tri, alpha_tri, gamma_tri)
        ctx.save_for_forward(a_tri, c_tri, alpha_tri, gamma_tri)

    @staticmethod
    def backward(
        ctx: Any,
        grad_alpha: torch.Tensor,
        grad_gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reverse sweep for:
        alpha_k = 1 / (b_k - a_k * gamma_{k-1}), gamma_k = c_k * alpha_k.

        The recurrence is derived by chain rule on each level k and then swept
        backwards because gamma_{k-1} influences alpha_k. Correctness is
        checked with gradcheck.
        """
        a_tri, c_tri, alpha_tri, gamma_tri = ctx.saved_tensors
        _n_cells, n_levels = a_tri.shape
        if ctx.use_scan:
            return _tridiag_factor_backward_scan(
                a_tri,
                c_tri,
                alpha_tri,
                gamma_tri,
                grad_alpha,
                grad_gamma,
            )

        grad_a = torch.zeros_like(a_tri)
        grad_b = torch.zeros_like(a_tri)
        grad_c = torch.zeros_like(a_tri)

        # Accumulate adjoints to gamma because gamma[k-1] feeds alpha[k].
        adj_gamma = grad_gamma.clone()

        for k in range(n_levels - 1, 0, -1):
            alpha_k = alpha_tri[:, k]
            gamma_prev = gamma_tri[:, k - 1]
            a_k = a_tri[:, k]
            c_k = c_tri[:, k]

            # gamma_k = c_k * alpha_k
            grad_c[:, k] = grad_c[:, k] + adj_gamma[:, k] * alpha_k
            total_adj_alpha = grad_alpha[:, k] + adj_gamma[:, k] * c_k

            # alpha_k = 1 / d_k, d_k = b_k - a_k * gamma_{k-1}
            d_adj = -total_adj_alpha * (alpha_k * alpha_k)
            grad_b[:, k] = grad_b[:, k] + d_adj
            grad_a[:, k] = grad_a[:, k] - d_adj * gamma_prev
            adj_gamma[:, k - 1] = adj_gamma[:, k - 1] - d_adj * a_k

        return grad_a, grad_b, grad_c

    @staticmethod
    def jvp(
        ctx: Any,
        grad_a: torch.Tensor | None,
        grad_b: torch.Tensor | None,
        grad_c: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward-mode recurrence for alpha/gamma sweeps.

        Equations:
          alpha_k = 1 / (b_k - a_k * gamma_{k-1})
          gamma_k = c_k * alpha_k
          d(alpha_k) = -alpha_k^2 * (d(b_k) - d(a_k)*gamma_{k-1} - a_k*d(gamma_{k-1}))
          d(gamma_k) = d(c_k)*alpha_k + c_k*d(alpha_k)
        """
        a_tri, c_tri, alpha_tri, gamma_tri = ctx.saved_tensors
        n_cells, n_levels = a_tri.shape
        if ctx.use_scan:
            return _tridiag_factor_jvp_scan(
                a_tri,
                c_tri,
                alpha_tri,
                gamma_tri,
                grad_a,
                grad_b,
                grad_c,
            )

        da = torch.zeros_like(a_tri) if grad_a is None else grad_a
        db = torch.zeros_like(a_tri) if grad_b is None else grad_b
        dc = torch.zeros_like(a_tri) if grad_c is None else grad_c

        d_alpha = torch.zeros(
            (n_cells, n_levels), dtype=a_tri.dtype, device=a_tri.device
        )
        d_gamma = torch.zeros(
            (n_cells, n_levels), dtype=a_tri.dtype, device=a_tri.device
        )
        for k in range(1, n_levels):
            alpha_k = alpha_tri[:, k]
            gamma_prev = gamma_tri[:, k - 1]
            d_gamma_prev = d_gamma[:, k - 1]
            term = db[:, k] - da[:, k] * gamma_prev - a_tri[:, k] * d_gamma_prev
            d_alpha[:, k] = -(alpha_k * alpha_k) * term
            d_gamma[:, k] = dc[:, k] * alpha_k + c_tri[:, k] * d_alpha[:, k]
        return d_alpha, d_gamma


def tridiagonal_factorize(
    a_tri: torch.Tensor,
    b_tri: torch.Tensor,
    c_tri: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose the shared forward recurrence to compiler-derived AD rules."""
    if compile_active():
        _validate_factor_inputs(a_tri, b_tri, c_tri)
        if tridiag_scan_preferred(a_tri):
            return _tridiag_factor_forward_scan(a_tri, b_tri, c_tri)
        return _tridiag_factor_forward(a_tri, b_tri, c_tri)
    return cast(
        tuple[torch.Tensor, torch.Tensor],
        TridiagonalFactorize.apply(a_tri, b_tri, c_tri),
    )


def tridiagonal_factorize_scan(
    a_tri: torch.Tensor,
    b_tri: torch.Tensor,
    c_tri: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Public scan implementation for focused validation and profiling."""
    return _tridiag_factor_forward_scan(a_tri, b_tri, c_tri)


__all__ = [
    "TridiagonalFactorize",
    "tridiagonal_factorize",
    "tridiagonal_factorize_scan",
]
