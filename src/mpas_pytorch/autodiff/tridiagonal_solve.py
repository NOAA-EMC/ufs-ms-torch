# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Custom autograd for acoustic-step-style tridiagonal forward/backward sweeps."""

from __future__ import annotations

from typing import Any, cast

import torch

from mpas_pytorch.autodiff.tridiagonal_scan import (
    ORDERED_VERTICAL_BLOCK_LEVELS,
    affine_forward_scan,
    affine_reverse_scan,
    compile_active,
    replace_vertical_interval,
    run_ordered_vertical_block,
    tridiag_scan_preferred,
    vertical_sweep_level_major,
)


def _tridiag_forward_elimination_scan(
    a_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Scan form of ``u_k = (rhs_k - a_k*u_{k-1}) * alpha_k``."""
    n_levels = a_tri.shape[1]
    if n_levels == 0:
        return rhs
    if n_levels == 1:
        return rhs
    p = -a_tri[:, 1:n_levels] * alpha_tri[:, 1:n_levels]
    q = rhs[:, 1:n_levels] * alpha_tri[:, 1:n_levels]
    interior = affine_forward_scan(p, q, rhs[:, 0])
    return replace_vertical_interval(rhs, interior, 1)


def _tridiag_back_substitution_scan(
    gamma_tri: torch.Tensor,
    eliminated: torch.Tensor,
) -> torch.Tensor:
    """Scan form of ``out_k = eliminated_k - gamma_k*out_{k+1}``."""
    n_levels = gamma_tri.shape[1]
    if n_levels == 0:
        return eliminated
    interior = affine_reverse_scan(
        -gamma_tri[:, :n_levels],
        eliminated[:, :n_levels],
        eliminated[:, n_levels],
    )
    return replace_vertical_interval(eliminated, interior, 0)


def _tridiag_solve_forward(
    a_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    gamma_tri: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Sequential Thomas sweeps with column-owned intermediate values.

    Keep the original association at every level. Building the final tensor
    once avoids whole-array slice-scatter temporaries when Inductor functionalizes
    an in-place sweep; no parallel scan or alternative numerical equation is used.
    """
    n_levels = a_tri.shape[1]
    if n_levels == 0:
        return rhs.clone()
    if vertical_sweep_level_major(a_tri.device.type):
        return _tridiag_solve_forward_level_major(a_tri, alpha_tri, gamma_tri, rhs)
    columns = list(rhs.unbind(dim=1))
    for k in range(1, n_levels):
        columns[k] = (columns[k] - a_tri[:, k] * columns[k - 1]) * alpha_tri[:, k]
    for k in range(n_levels - 1, -1, -1):
        columns[k] = columns[k] - gamma_tri[:, k] * columns[k + 1]
    return torch.stack(columns, dim=1)


def _tridiag_solve_forward_level_major(
    a_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    gamma_tri: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """The same sweeps over contiguous level rows of transposed copies.

    Inductor folds a bare transpose back into its consumer, so the graph breaks
    make the copies real buffers: the sweep kernel then reads each level row
    with neighbouring columns adjacent in memory.
    """
    compile_blocks = torch.compiler.is_compiling()
    a_rows = a_tri.t().contiguous()
    alpha_rows = alpha_tri.t().contiguous()
    gamma_rows = gamma_tri.t().contiguous()
    rhs_rows = rhs.t().contiguous()
    torch._dynamo.graph_break()
    solved = _tridiag_solve_level_rows(
        a_rows, alpha_rows, gamma_rows, rhs_rows, compile_blocks=compile_blocks,
    )
    torch._dynamo.graph_break()
    return solved.t().contiguous()


@torch.compiler.disable
def _tridiag_solve_level_rows(a_rows, alpha_rows, gamma_rows, rhs_rows, *, compile_blocks=False):
    n_levels = a_rows.shape[0]
    rows = [rhs_rows[:1]]
    for start in range(1, n_levels, ORDERED_VERTICAL_BLOCK_LEVELS):
        stop = min(start + ORDERED_VERTICAL_BLOCK_LEVELS, n_levels)
        rows.append(run_ordered_vertical_block(
            _tridiag_eliminate_level_block,
            a_rows[start:stop], alpha_rows[start:stop], rhs_rows[start:stop], rows[-1][-1],
            compiled=compile_blocks,
        ))
    eliminated = torch.cat((*rows, rhs_rows[n_levels:]), dim=0)
    solved = [eliminated[n_levels:]]
    for stop in range(n_levels, 0, -ORDERED_VERTICAL_BLOCK_LEVELS):
        start = max(0, stop - ORDERED_VERTICAL_BLOCK_LEVELS)
        solved.append(run_ordered_vertical_block(
            _tridiag_substitute_level_block,
            gamma_rows[start:stop], eliminated[start:stop], solved[-1][0],
            compiled=compile_blocks,
        ))
    return torch.cat(list(reversed(solved)), dim=0)


def _tridiag_eliminate_level_block(a_rows, alpha_rows, rhs_rows, previous):
    rows = []
    for k in range(a_rows.shape[0]):
        previous = (rhs_rows[k] - a_rows[k] * previous) * alpha_rows[k]
        rows.append(previous)
    return torch.stack(rows)


def _tridiag_substitute_level_block(gamma_rows, eliminated, following):
    rows = []
    for k in range(gamma_rows.shape[0] - 1, -1, -1):
        following = eliminated[k] - gamma_rows[k] * following
        rows.append(following)
    return torch.stack(list(reversed(rows)))


def _tridiag_solve_forward_scan(
    a_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    gamma_tri: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Parallel-prefix form of acoustic-step tridiagonal solve sweeps."""
    if a_tri.shape[1] == 0:
        return rhs.clone()
    eliminated = _tridiag_forward_elimination_scan(a_tri, alpha_tri, rhs)
    return _tridiag_back_substitution_scan(gamma_tri, eliminated)


def _tridiag_solve_backward_scan(
    a_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    gamma_tri: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scan form of the reverse-mode solve sweeps."""
    _n_cells, n_levels = a_tri.shape
    grad_a = torch.zeros_like(a_tri)
    grad_alpha = torch.zeros_like(alpha_tri)
    grad_gamma = torch.zeros_like(gamma_tri)
    grad_rhs = torch.zeros_like(rhs)
    if n_levels == 0:
        return grad_a, grad_alpha, grad_gamma, grad_out.clone()

    if n_levels > 0:
        adj_tail = affine_forward_scan(
            -gamma_tri[:, :n_levels],
            grad_out[:, 1 : n_levels + 1],
            grad_out[:, 0],
        )
        adj_from_back_sub = grad_out.clone()
        adj_from_back_sub[:, 1 : n_levels + 1] = adj_tail
        grad_gamma[:, :n_levels] = (
            -adj_from_back_sub[:, :n_levels] * out[:, 1 : n_levels + 1]
        )
    else:
        adj_from_back_sub = grad_out

    u = _tridiag_forward_elimination_scan(a_tri, alpha_tri, rhs)
    if n_levels <= 1:
        grad_rhs[:, 0] = adj_from_back_sub[:, 0]
        if rhs.shape[1] > 1:
            grad_rhs[:, 1] = adj_from_back_sub[:, 1]
        return grad_a, grad_alpha, grad_gamma, grad_rhs

    p = -a_tri[:, 1:n_levels] * alpha_tri[:, 1:n_levels]
    q = adj_from_back_sub[:, 0 : n_levels - 1]
    adj_lower = affine_reverse_scan(
        p,
        q,
        adj_from_back_sub[:, n_levels - 1],
    )
    adj_u = adj_from_back_sub[:, :n_levels].clone()
    adj_u[:, : n_levels - 1] = adj_lower

    t = rhs[:, 1:n_levels] - a_tri[:, 1:n_levels] * u[:, 0 : n_levels - 1]
    adj_t = adj_u[:, 1:n_levels] * alpha_tri[:, 1:n_levels]
    grad_alpha[:, 1:n_levels] = adj_u[:, 1:n_levels] * t
    grad_rhs[:, 1:n_levels] = adj_t
    grad_a[:, 1:n_levels] = -adj_t * u[:, 0 : n_levels - 1]
    grad_rhs[:, 0] = adj_u[:, 0]
    grad_rhs[:, n_levels] = adj_from_back_sub[:, n_levels]
    return grad_a, grad_alpha, grad_gamma, grad_rhs


def _tridiag_solve_jvp_scan(
    a_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    gamma_tri: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    grad_a: torch.Tensor | None,
    grad_alpha: torch.Tensor | None,
    grad_gamma: torch.Tensor | None,
    grad_rhs: torch.Tensor | None,
) -> torch.Tensor:
    """Scan form of the forward-mode solve recurrence."""
    _n_cells, n_levels = a_tri.shape
    da = torch.zeros_like(a_tri) if grad_a is None else grad_a
    d_alpha = torch.zeros_like(alpha_tri) if grad_alpha is None else grad_alpha
    d_gamma = torch.zeros_like(gamma_tri) if grad_gamma is None else grad_gamma
    d_rhs = torch.zeros_like(rhs) if grad_rhs is None else grad_rhs
    if n_levels == 0:
        return d_rhs

    u = _tridiag_forward_elimination_scan(a_tri, alpha_tri, rhs)
    d_u = torch.zeros_like(rhs)
    d_u[:, 0] = d_rhs[:, 0]
    d_u[:, n_levels] = d_rhs[:, n_levels]
    if n_levels > 1:
        t = rhs[:, 1:n_levels] - a_tri[:, 1:n_levels] * u[:, 0 : n_levels - 1]
        p = -a_tri[:, 1:n_levels] * alpha_tri[:, 1:n_levels]
        q = (
            d_rhs[:, 1:n_levels] - da[:, 1:n_levels] * u[:, 0 : n_levels - 1]
        ) * alpha_tri[:, 1:n_levels] + t * d_alpha[:, 1:n_levels]
        d_u[:, 1:n_levels] = affine_forward_scan(p, q, d_u[:, 0])

    d_out_interior = affine_reverse_scan(
        -gamma_tri[:, :n_levels],
        d_u[:, :n_levels] - d_gamma[:, :n_levels] * out[:, 1 : n_levels + 1],
        d_u[:, n_levels],
    )
    d_out = d_u.clone()
    d_out[:, :n_levels] = d_out_interior
    return d_out


def _validate_solve_inputs(
    a_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    gamma_tri: torch.Tensor,
    rhs: torch.Tensor,
) -> None:
    if a_tri.shape != alpha_tri.shape or a_tri.shape != gamma_tri.shape:
        raise ValueError("a_tri, alpha_tri, gamma_tri must have identical shapes")
    if a_tri.ndim != 2:
        raise ValueError(
            "expected tridiagonal tensors with shape (nCells, nVertLevels)"
        )
    if (
        rhs.ndim != 2
        or rhs.shape[0] != a_tri.shape[0]
        or rhs.shape[1] != a_tri.shape[1] + 1
    ):
        raise ValueError("rhs must have shape (nCells, nVertLevels + 1)")


class TridiagonalSolve(torch.autograd.Function):
    """Solve tridiagonal system using acoustic-step sweep ordering.

    Notes:
    - Input `rhs` is expected to have shape `(nCells, nVertLevels + 1)`.
    - The extra last level is the boundary value used during back substitution.
    - Compatible with gradient checkpointing in later phases when
      `torch.utils.checkpoint(..., use_reentrant=False)` is used.

    Second-order extensibility (double backward) options:
    1) Keep this custom Function and make backward-path ops fully differentiable
       (remove blockers and avoid non-traceable backward logic).
       Pro: minimal interface churn. Con: subtle to get complete coverage.
    2) Convert backward implementation to purely functional tensor expressions,
       avoiding in-place updates so autograd can build grad-of-grad graph.
       Pro: straightforward gradgradcheck workflow. Con: potential perf overhead.
    3) Add explicit custom backward Function(s) for this solver path
       (e.g., TridiagonalSolveBackward) with hand-derived formulas.
       Pro: strongest control of stability/performance. Con: most complex to maintain.
    """

    @staticmethod
    def forward(
        a_tri: torch.Tensor,
        alpha_tri: torch.Tensor,
        gamma_tri: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        _validate_solve_inputs(a_tri, alpha_tri, gamma_tri, rhs)

        use_scan = tridiag_scan_preferred(a_tri)
        if use_scan:
            out = _tridiag_solve_forward_scan(a_tri, alpha_tri, gamma_tri, rhs)
        else:
            out = _tridiag_solve_forward(a_tri, alpha_tri, gamma_tri, rhs)

        return out

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Bind the unchanged sweep rules on both autograd capture surfaces."""
        a_tri, alpha_tri, gamma_tri, rhs = inputs
        ctx.use_scan = tridiag_scan_preferred(a_tri)
        ctx.save_for_backward(a_tri, alpha_tri, gamma_tri, rhs, output)
        ctx.save_for_forward(a_tri, alpha_tri, gamma_tri, rhs, output)

    @staticmethod
    def backward(
        ctx: Any,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reverse sweeps for acoustic-step solve:
        1) u_k = (rhs_k - a_k * u_{k-1}) * alpha_k
        2) out_k = u_k - gamma_k * out_{k+1}

        We first backprop through step (2), then through step (1). The
        `grad_gamma` term intentionally uses `out_{k+1}` from the final solution
        (saved as `out`), which is the exact dependency in step (2). Correctness
        is checked with gradcheck.
        """
        a_tri, alpha_tri, gamma_tri, rhs, out = ctx.saved_tensors
        _n_cells, n_levels = a_tri.shape
        if ctx.use_scan:
            return _tridiag_solve_backward_scan(
                a_tri,
                alpha_tri,
                gamma_tri,
                rhs,
                out,
                grad_out,
            )

        grad_a = torch.zeros_like(a_tri)
        grad_alpha = torch.zeros_like(alpha_tri)
        grad_gamma = torch.zeros_like(gamma_tri)
        grad_rhs = torch.zeros_like(rhs)

        if n_levels == 0:
            return grad_a, grad_alpha, grad_gamma, grad_out.clone()

        # Backward through back-substitution:
        # out_k = u_k - gamma_k * out_{k+1}
        adj_out = grad_out.clone()
        adj_u = torch.zeros_like(rhs)
        for k in range(n_levels):
            adj_u[:, k] = adj_u[:, k] + adj_out[:, k]
            grad_gamma[:, k] = grad_gamma[:, k] - adj_out[:, k] * out[:, k + 1]
            adj_out[:, k + 1] = adj_out[:, k + 1] - gamma_tri[:, k] * adj_out[:, k]
        adj_u[:, n_levels] = adj_u[:, n_levels] + adj_out[:, n_levels]

        # Recompute forward-elimination intermediate u.
        u = rhs.clone()
        for k in range(1, n_levels):
            u[:, k] = (u[:, k] - a_tri[:, k] * u[:, k - 1]) * alpha_tri[:, k]

        # Backward through forward-elimination:
        # u_k = (rhs_k - a_k * u_{k-1}) * alpha_k
        for k in range(n_levels - 1, 0, -1):
            t = rhs[:, k] - a_tri[:, k] * u[:, k - 1]
            grad_alpha[:, k] = grad_alpha[:, k] + adj_u[:, k] * t
            adj_t = adj_u[:, k] * alpha_tri[:, k]
            grad_rhs[:, k] = grad_rhs[:, k] + adj_t
            grad_a[:, k] = grad_a[:, k] - adj_t * u[:, k - 1]
            adj_u[:, k - 1] = adj_u[:, k - 1] - adj_t * a_tri[:, k]

        # Boundary equations: u_0 = rhs_0, u_K = rhs_K
        grad_rhs[:, 0] = grad_rhs[:, 0] + adj_u[:, 0]
        grad_rhs[:, n_levels] = grad_rhs[:, n_levels] + adj_u[:, n_levels]

        return grad_a, grad_alpha, grad_gamma, grad_rhs

    @staticmethod
    def jvp(
        ctx: Any,
        grad_a: torch.Tensor | None,
        grad_alpha: torch.Tensor | None,
        grad_gamma: torch.Tensor | None,
        grad_rhs: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward-mode recurrence through elimination + back-substitution.

        Equations:
          u_k = (rhs_k - a_k * u_{k-1}) * alpha_k
          out_k = u_k - gamma_k * out_{k+1}
          d(u_k) = (d(rhs_k) - d(a_k)*u_{k-1} - a_k*d(u_{k-1})) * alpha_k
                   + (rhs_k - a_k*u_{k-1}) * d(alpha_k)
          d(out_k) = d(u_k) - d(gamma_k)*out_{k+1} - gamma_k*d(out_{k+1})
        """
        a_tri, alpha_tri, gamma_tri, rhs, out = ctx.saved_tensors
        _n_cells, n_levels = a_tri.shape
        if ctx.use_scan:
            return _tridiag_solve_jvp_scan(
                a_tri,
                alpha_tri,
                gamma_tri,
                rhs,
                out,
                grad_a,
                grad_alpha,
                grad_gamma,
                grad_rhs,
            )

        da = torch.zeros_like(a_tri) if grad_a is None else grad_a
        d_alpha = torch.zeros_like(alpha_tri) if grad_alpha is None else grad_alpha
        d_gamma = torch.zeros_like(gamma_tri) if grad_gamma is None else grad_gamma
        d_rhs = torch.zeros_like(rhs) if grad_rhs is None else grad_rhs

        # Recompute forward elimination intermediate `u`.
        u = rhs.clone()
        for k in range(1, n_levels):
            u[:, k] = (u[:, k] - a_tri[:, k] * u[:, k - 1]) * alpha_tri[:, k]

        # Forward-mode through elimination.
        d_u = torch.zeros_like(rhs)
        d_u[:, 0] = d_rhs[:, 0]
        d_u[:, n_levels] = d_rhs[:, n_levels]
        for k in range(1, n_levels):
            t = rhs[:, k] - a_tri[:, k] * u[:, k - 1]
            d_t = d_rhs[:, k] - da[:, k] * u[:, k - 1] - a_tri[:, k] * d_u[:, k - 1]
            d_u[:, k] = d_t * alpha_tri[:, k] + t * d_alpha[:, k]

        # Forward-mode through back-substitution.
        d_out = d_u.clone()
        for k in range(n_levels - 1, -1, -1):
            d_out[:, k] = (
                d_out[:, k]
                - d_gamma[:, k] * out[:, k + 1]
                - gamma_tri[:, k] * d_out[:, k + 1]
            )
        return d_out


def tridiagonal_solve(
    a_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    gamma_tri: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Use the shared recurrence directly during compiler tracing.

    Dynamo cannot capture a custom Function with a JVP on reverse-mode paths.
    Tracing the SAME forward helper lets PyTorch differentiate its tensor program,
    without an opaque frontend escape or an inference-only numerical engine.
    Policy inspection remains visible to Dynamo and its guards. Eager execution
    retains the memory-efficient custom rules.
    """
    if compile_active():
        _validate_solve_inputs(a_tri, alpha_tri, gamma_tri, rhs)
        if tridiag_scan_preferred(a_tri):
            return _tridiag_solve_forward_scan(a_tri, alpha_tri, gamma_tri, rhs)
        return _tridiag_solve_forward(a_tri, alpha_tri, gamma_tri, rhs)
    return cast(torch.Tensor, TridiagonalSolve.apply(a_tri, alpha_tri, gamma_tri, rhs))


def tridiagonal_solve_scan(
    a_tri: torch.Tensor,
    alpha_tri: torch.Tensor,
    gamma_tri: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Public scan implementation for focused validation and profiling."""
    return _tridiag_solve_forward_scan(a_tri, alpha_tri, gamma_tri, rhs)


__all__ = ["TridiagonalSolve", "tridiagonal_solve", "tridiagonal_solve_scan"]
