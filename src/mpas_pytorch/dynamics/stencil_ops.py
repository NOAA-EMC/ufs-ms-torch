# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Reusable stencil operators for vectorized MPAS dynamics kernels.

These helpers provide gather-reduce and CSR-SpMM implementations for common
connectivity patterns used by diagnostic-solve and upcoming vectorized kernels.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast
import weakref

import torch
from torch._subclasses.fake_tensor import is_fake

from mpas_pytorch.dynamics.ordered_sum_plans import OrderedSumPlan, build_ordered_sum_plan

_STENCIL_SLOT_GATHER_BATCH = 4
_AUDIT_ALLOW_STATIC_METADATA_UNDER_COMPILE = False
_STENCIL_COMPILE_TARGETS = (
    "stencil_ordered_weighted_sum",
    "stencil_ordered_weighted_components",
    "column_ordered_factor",
)
_STENCIL_COMPILE_POLICY: ContextVar[tuple | None] = ContextVar(
    "mpas_stencil_compile_policy", default=None
)


@contextmanager
def use_prepared_stencil_compile_policy(policy):
    """Replay a call-owned compile selection, including checkpoint recomputation."""
    token = _STENCIL_COMPILE_POLICY.set(policy)
    try:
        yield
    finally:
        _STENCIL_COMPILE_POLICY.reset(token)


@contextmanager
def use_stencil_compile_policy(config, *, device):
    """Select shared reducers using the existing explicit kernel policy."""
    from .compile_utils import (
        compile_backend,
        compile_enabled,
        compile_mode,
        kernel_compile_enabled_for_device,
        kernel_compiled_reverse_ad_enabled_for_device,
    )

    selected = ()
    if compile_enabled(dict(config)):
        selected = tuple(
            (
                name,
                kernel_compiled_reverse_ad_enabled_for_device(
                    name, config, device=device
                ),
            )
            for name in _STENCIL_COMPILE_TARGETS
            if kernel_compile_enabled_for_device(name, config, device=device)
        )
    policy = (
        (selected, compile_mode(config), compile_backend(config)) if selected else None
    )
    with use_prepared_stencil_compile_policy(policy):
        yield


def _compiled_stencil_reducer(name, function, *tensors, metadata=None):
    # Compose the same expressions when an enclosing numeric region is tracing.
    # Never dispatch, inspect a context variable, or mutate caches in that graph.
    if _compile_active():
        return None
    policy = _STENCIL_COMPILE_POLICY.get()
    if policy is None:
        return None
    selected, mode, backend = policy
    admission = dict(selected)
    if name not in admission:
        return None
    from .compile_utils import maybe_compile_kernel

    signature = tuple(
        (
            tuple(t.shape),
            tuple(t.stride()),
            str(t.dtype),
            str(t.device),
            t.requires_grad,
        )
        if isinstance(t, torch.Tensor)
        else None
        for t in tensors
    )
    return maybe_compile_kernel(
        name,
        function,
        enabled=True,
        mode=mode,
        backend=backend,
        specialization_key=(signature, metadata, torch.is_grad_enabled()),
        allow_reverse_autodiff=admission[name],
    )


def _torch_compiler_module() -> object | None:
    try:
        return torch.compiler
    except AttributeError:
        return None


def _torch_dynamo_module() -> object | None:
    try:
        return torch._dynamo
    except AttributeError:
        return None


def _compile_active() -> bool:
    compiler = _torch_compiler_module()
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return bool(compiler.is_compiling())
    dynamo = _torch_dynamo_module()
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return bool(dynamo.is_compiling())
    return False


def _tensor_carries_autodiff(tensor: torch.Tensor) -> bool:
    """Return whether a tensor participates in reverse-mode or forward-mode AD."""
    if tensor.requires_grad:
        return True
    try:
        from torch.autograd import forward_ad as fwAD

        _, tangent = fwAD.unpack_dual(tensor)
    except RuntimeError:
        return False
    return tangent is not None


def _static_connectivity_cache_allowed(tensors: Iterable[Any]) -> bool:
    """Allow static topology cache reuse unless compile or AD touches topology."""
    if _compile_active() and not _AUDIT_ALLOW_STATIC_METADATA_UNDER_COMPILE:
        return False
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and _tensor_carries_autodiff(tensor):
            return False
    return True


def _active_slot_count(
    counts: torch.Tensor, available: int, device: torch.device
) -> int:
    """Keep a shape-owned width; device masks exclude inactive slots."""
    return available


def _slot_batches(slot_count: int, batch_size: int = _STENCIL_SLOT_GATHER_BATCH):
    width = max(1, min(int(batch_size), max(int(slot_count), 1)))
    for start in range(0, int(slot_count), width):
        yield start, min(start + width, int(slot_count))


def _slot_gather_batch(field: torch.Tensor, slot_count: int) -> int:
    # One slot per gather on every device. Wider batches give the same bytes
    # but make Inductor materialize [rows, width, levels] gathers on CUDA:
    # A100 60 km split 482 -> 419 ms/step at width 1 (kernel-efficiency ke3).
    return 1


def _gather_slot_batch(
    field: torch.Tensor,
    indices: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    if stop - start == 1:
        return gather_rows(field, indices[:, start]).unsqueeze(1)
    return gather_rows(field, indices[:, start:stop])


def _static_mask_all_true(
    mask: torch.Tensor | None,
    *,
    allow_cuda_sync: bool = False,
) -> bool | None:
    """Infer only shape-proven shortcuts; otherwise retain the tensor mask.

    The compatibility argument no longer permits tensor-to-host inference.
    Nonempty masks remain authoritative in the existing masked numerical path.
    """
    del allow_cuda_sync
    if mask is None:
        return None
    if mask.numel() == 0:
        return True
    return None


def _static_slot_all_true(
    mask: torch.Tensor | None,
    *,
    allow_cuda_sync: bool = False,
) -> tuple[bool, ...] | None:
    """Keep nonempty per-slot masks on device instead of deriving host hints."""
    del allow_cuda_sync
    if mask is None:
        return None
    if mask.shape[1] == 0:
        return ()
    return None


def sum_fortran_order(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Deterministic left-to-right reduction to mirror Fortran loop order."""
    if dim < 0:
        dim += x.ndim
    if dim < 0 or dim >= x.ndim:
        raise IndexError(f"sum dim out of range: dim={dim}, ndim={x.ndim}")
    n = int(x.shape[dim])
    if n == 0:
        out_shape = list(x.shape)
        del out_shape[dim]
        return torch.zeros(out_shape, dtype=x.dtype, device=x.device)
    if not torch.is_grad_enabled() and not _compile_active():
        # Preserve the exact Fortran association while avoiding one full output
        # allocation per slot in eager inference.  Clone the first slot so the
        # input remains immutable, then reuse that private accumulator.
        total = x.select(dim, 0).clone()
        for idx in range(1, n):
            total.add_(x.select(dim, idx))
        return total
    total = x.select(dim, 0)
    for idx in range(1, n):
        total = total + x.select(dim, idx)
    return total


# Backward-compatible alias for existing call sites/tests.
_sum_fortran_order = sum_fortran_order


def _accumulate_ordered_slot(
    total: torch.Tensor | None, contrib: torch.Tensor
) -> torch.Tensor:
    """Left-to-right slot accumulation with inference-only local in-place reuse."""
    if total is None:
        return contrib
    if not torch.is_grad_enabled() and not _compile_active():
        total.add_(contrib)
        return total
    return total + contrib


class PinnedTensorKey:
    """Cache-key element that compares by an address-like value and pins it.

    Holding the tensor keeps its storage alive for as long as the key is held
    (e.g. by a cache entry), so the allocator cannot hand the same address to
    a different tensor while a cached result keyed on it exists. Without the
    pin, a freed mesh's address and version could be reused by a new tensor
    with different contents and hit a stale entry.
    """

    __slots__ = ("_tensor", "value")

    def __init__(self, tensor: torch.Tensor, value: int) -> None:
        self._tensor = tensor
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PinnedTensorKey) and self.value == other.value

    def __hash__(self) -> int:
        return hash(("pinned_tensor", self.value))

    def __repr__(self) -> str:
        return f"pinned_tensor({self.value})"


def _pinned_identity(tensor: torch.Tensor) -> PinnedTensorKey:
    """Object-identity key element that keeps the object (and its id) alive."""
    return PinnedTensorKey(tensor, id(tensor))


def _tensor_version_cache_key(tensor: torch.Tensor) -> tuple[object, ...] | None:
    """Return a mutation-sensitive cache key, or ``None`` when unsafe to cache.

    The address element pins the tensor (see PinnedTensorKey). Layout is part
    of identity because equal-pointer views can expose different values.
    """
    try:
        version = int(tensor._version)
        pointer = int(tensor.data_ptr())
    except RuntimeError:
        return None
    return (
        PinnedTensorKey(tensor, pointer),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        int(tensor.storage_offset()),
        str(tensor.device),
        str(tensor.dtype),
        version,
    )


def _tensor_version_or_none(tensor: torch.Tensor) -> int | None:
    """Version counter (bumped by every in-place write to the tensor or a view
    of it), or None for inference tensors, which carry none. Host metadata
    only: no device synchronization."""
    try:
        return int(tensor._version)
    except RuntimeError:
        return None


class _IndexCopyRows(torch.autograd.Function):
    """Memory-efficient row replacement for autograd.

    Standard ``torch.index_copy`` saves the full output tensor in the
    backward graph (~5 MB per call on the coarse mesh). This custom Function saves
    only ``row_idx`` (tiny) and propagates gradients sparsely: the backward
    for ``base`` scatters the output gradient at non-replaced rows, and
    the backward for ``values`` gathers from the replaced rows.
    """

    @staticmethod
    def forward(
        ctx, base: torch.Tensor, row_idx: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(row_idx)
        ctx.save_for_forward(row_idx)
        ctx.base_shape = base.shape
        return torch.index_copy(base, 0, row_idx, values)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (row_idx,) = ctx.saved_tensors
        grad_base = None
        if ctx.needs_input_grad[0]:
            grad_base = grad_output.clone()
            grad_base[row_idx] = 0
        grad_values = None
        if ctx.needs_input_grad[2]:
            grad_values = grad_output.index_select(0, row_idx)
        return grad_base, None, grad_values

    @staticmethod
    def jvp(
        ctx: Any,
        grad_base: torch.Tensor | None,
        grad_row_idx: torch.Tensor | None,
        grad_values: torch.Tensor | None,
    ) -> torch.Tensor:
        (row_idx,) = ctx.saved_tensors
        del grad_row_idx
        if grad_base is None:
            if grad_values is None:
                raise RuntimeError(
                    "index-copy JVP requires one differentiable tensor input"
                )
            out = torch.zeros(
                ctx.base_shape, dtype=grad_values.dtype, device=grad_values.device
            )
            return torch.index_copy(out, 0, row_idx, grad_values)
        out = grad_base.clone()
        if grad_values is not None:
            out = torch.index_copy(out, 0, row_idx, grad_values)
        else:
            out[row_idx] = 0
        return out


def _functional_replace_rows(
    base: torch.Tensor,
    row_idx: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Functional dim-0 row replacement without custom autograd wrapping."""
    return torch.index_copy(base, 0, row_idx, values)


def _shares_storage(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Return True when tensors use the same backing storage."""
    try:
        lhs_ptr = lhs.untyped_storage().data_ptr()
        rhs_ptr = rhs.untyped_storage().data_ptr()
    except RuntimeError:
        return False
    return lhs_ptr == rhs_ptr


def _clone_if_storage_overlap(base: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Clone source values before in-place writes when they alias the target."""
    return values.clone() if _shares_storage(base, values) else values


def _replace_rows(
    base: torch.Tensor, row_idx: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Functional dim-0 row replacement with an inference fast path."""
    if int(row_idx.numel()) == 0 and not _tensor_carries_autodiff(values):
        return base
    if _compile_active():
        return _functional_replace_rows(base, row_idx, values)
    if not torch.is_grad_enabled():
        values = _clone_if_storage_overlap(base, values)
        base[row_idx] = values
        return base
    return cast(torch.Tensor, _IndexCopyRows.apply(base, row_idx, values))


def _replace_all_rows(base: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Replace every dim-0 row without constructing an all-row index tensor."""
    if not torch.is_grad_enabled() and not _compile_active():
        values = _clone_if_storage_overlap(base, values)
        base[...] = values
        return base
    return values


def _replace_row_slice(
    base: torch.Tensor,
    row_start: int,
    row_end: int,
    values: torch.Tensor,
    *,
    preserve_base: bool = False,
) -> torch.Tensor:
    """Replace contiguous rows, optionally preserving inputs even in eager no-grad mode."""
    row_start = int(row_start)
    row_end = int(row_end)
    if row_start < 0 or row_end < row_start or row_end > base.shape[0]:
        raise ValueError(
            f"invalid row slice [{row_start}:{row_end}] for dim-0 size {base.shape[0]}"
        )
    if (row_end - row_start) != values.shape[0]:
        raise ValueError(
            f"row slice length {row_end - row_start} does not match values rows {values.shape[0]}"
        )
    if row_start == row_end and not _tensor_carries_autodiff(values):
        return base
    if row_start == 0 and row_end == base.shape[0]:
        if preserve_base:
            return values
        return _replace_all_rows(base, values)
    if not preserve_base and not torch.is_grad_enabled() and not _compile_active():
        values = _clone_if_storage_overlap(base, values)
        base[row_start:row_end, ...] = values
        return base
    # Functional concatenation writes the result once. Cloning ``base`` and
    # then copying an almost-full active patch writes most rows twice, which is
    # especially costly for MPAS's one-row-padded public fields. There is no
    # reduction or gradient accumulation here, so concatenation preserves the
    # exact replacement and its VJP/JVP.
    pieces = []
    if row_start:
        pieces.append(base[:row_start, ...])
    pieces.append(values)
    if row_end < base.shape[0]:
        pieces.append(base[row_end:, ...])
    return torch.cat(tuple(pieces), dim=0)


def _zero_row_slice(base: torch.Tensor, row_start: int, row_end: int) -> torch.Tensor:
    """Set a contiguous dim-0 row slice to zero.

    Same result as ``_replace_row_slice`` with a zero patch. Outside autograd
    and compilation the rows are zeroed in place, without allocating and
    copying a zero patch.
    """
    row_start = int(row_start)
    row_end = int(row_end)
    if row_start < 0 or row_end < row_start or row_end > base.shape[0]:
        raise ValueError(
            f"invalid row slice [{row_start}:{row_end}] for dim-0 size {base.shape[0]}"
        )
    if not torch.is_grad_enabled() and not _compile_active():
        base[row_start:row_end, ...].zero_()
        return base
    return _replace_row_slice(
        base,
        row_start,
        row_end,
        base.new_zeros((row_end - row_start, *base.shape[1:])),
    )


def _replace_row_slice_cols(
    base: torch.Tensor,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    values: torch.Tensor,
) -> torch.Tensor:
    """Replace a contiguous dim-0 row slice and dim-1 column slice."""
    row_start = int(row_start)
    row_end = int(row_end)
    col_start = int(col_start)
    col_end = int(col_end)
    if row_start < 0 or row_end < row_start or row_end > base.shape[0]:
        raise ValueError(
            f"invalid row slice [{row_start}:{row_end}] for dim-0 size {base.shape[0]}"
        )
    if col_start < 0 or col_end < col_start or col_end > base.shape[1]:
        raise ValueError(
            f"invalid column slice [{col_start}:{col_end}] for dim-1 size {base.shape[1]}"
        )
    if (row_end - row_start) != values.shape[0]:
        raise ValueError(
            f"row slice length {row_end - row_start} does not match values rows {values.shape[0]}"
        )
    if (col_end - col_start) != values.shape[1]:
        raise ValueError(
            f"column slice length {col_end - col_start} does not match values columns {values.shape[1]}"
        )
    if (row_start == row_end or col_start == col_end) and not _tensor_carries_autodiff(
        values
    ):
        return base
    if col_start == 0 and col_end == base.shape[1]:
        return _replace_row_slice(base, row_start, row_end, values)
    if not torch.is_grad_enabled() and not _compile_active():
        values = _clone_if_storage_overlap(base, values)
        base[row_start:row_end, col_start:col_end, ...] = values
        return base
    active_columns = []
    if col_start:
        active_columns.append(base[row_start:row_end, :col_start, ...])
    active_columns.append(values)
    if col_end < base.shape[1]:
        active_columns.append(base[row_start:row_end, col_end:, ...])
    active_rows = torch.cat(tuple(active_columns), dim=1)
    output_rows = []
    if row_start:
        output_rows.append(base[:row_start, ...])
    output_rows.append(active_rows)
    if row_end < base.shape[0]:
        output_rows.append(base[row_end:, ...])
    return torch.cat(tuple(output_rows), dim=0)


def _functional_replace_rows_cols(
    base: torch.Tensor,
    row_idx: torch.Tensor,
    col_start: int,
    col_end: int,
    values: torch.Tensor,
) -> torch.Tensor:
    """Functional dim-0 + dim-1 slice replacement without materializing cats."""
    if col_start == 0 and col_end == base.shape[1]:
        updated = values
    else:
        updated = base.index_select(0, row_idx).clone()
        updated[:, col_start:col_end, ...] = values
    return torch.index_copy(base, 0, row_idx, updated)


class _IndexCopyRowsCols(torch.autograd.Function):
    """Memory-efficient row+col slice replacement for autograd."""

    @staticmethod
    def forward(
        ctx,
        base: torch.Tensor,
        row_idx: torch.Tensor,
        col_start: int,
        col_end: int,
        values: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(row_idx)
        ctx.save_for_forward(row_idx)
        ctx.col_start = col_start
        ctx.col_end = col_end
        ctx.base_shape = base.shape
        return _functional_replace_rows_cols(base, row_idx, col_start, col_end, values)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (row_idx,) = ctx.saved_tensors
        col_start, col_end = ctx.col_start, ctx.col_end
        grad_base = None
        if ctx.needs_input_grad[0]:
            grad_base = grad_output.clone()
            grad_base[row_idx, col_start:col_end, ...] = 0
        grad_values = None
        if ctx.needs_input_grad[4]:
            grad_values = grad_output.index_select(0, row_idx)[
                :, col_start:col_end, ...
            ]
        return grad_base, None, None, None, grad_values

    @staticmethod
    def jvp(
        ctx: Any,
        grad_base: torch.Tensor | None,
        grad_row_idx: torch.Tensor | None,
        grad_col_start: int | None,
        grad_col_end: int | None,
        grad_values: torch.Tensor | None,
    ) -> torch.Tensor:
        (row_idx,) = ctx.saved_tensors
        del grad_row_idx, grad_col_start, grad_col_end
        col_start, col_end = ctx.col_start, ctx.col_end
        if grad_base is None:
            if grad_values is None:
                raise RuntimeError(
                    "IndexCopyRowsCols.jvp requires a tangent for base or values"
                )
            out = torch.zeros(
                ctx.base_shape, dtype=grad_values.dtype, device=grad_values.device
            )
        else:
            out = grad_base.clone()
        if grad_values is not None:
            out = _functional_replace_rows_cols(
                out, row_idx, col_start, col_end, grad_values
            )
        else:
            out[row_idx, col_start:col_end, ...] = 0
        return out


def _replace_rows_cols(
    base: torch.Tensor,
    row_idx: torch.Tensor,
    col_start: int,
    col_end: int,
    values: torch.Tensor,
) -> torch.Tensor:
    """Functional dim-0 + dim-1 slice replacement with an inference fast path."""
    if (
        int(row_idx.numel()) == 0 or col_start == col_end
    ) and not _tensor_carries_autodiff(values):
        return base
    if _compile_active():
        return _functional_replace_rows_cols(base, row_idx, col_start, col_end, values)
    if not torch.is_grad_enabled():
        values = _clone_if_storage_overlap(base, values)
        base[row_idx, col_start:col_end, ...] = values
        return base
    return cast(
        torch.Tensor,
        _IndexCopyRowsCols.apply(base, row_idx, col_start, col_end, values),
    )


def _zero_rows(base: torch.Tensor, row_idx: torch.Tensor) -> torch.Tensor:
    """Replace dim-0 rows by zeros without gathering those rows first."""
    zeros = base.new_zeros((int(row_idx.numel()), *base.shape[1:]))
    return _replace_rows(base, row_idx.reshape(-1), zeros)


def _replace_all_rows_cols(
    base: torch.Tensor,
    col_start: int,
    col_end: int,
    values: torch.Tensor,
) -> torch.Tensor:
    """Replace one column slice for every row without constructing row indices."""
    if col_start == 0 and col_end == base.shape[1]:
        if not torch.is_grad_enabled() and not _compile_active():
            values = _clone_if_storage_overlap(base, values)
            base[...] = values
            return base
        return values
    if not torch.is_grad_enabled() and not _compile_active():
        values = _clone_if_storage_overlap(base, values)
        base[:, col_start:col_end, ...] = values
        return base
    pieces = []
    if col_start:
        pieces.append(base[:, :col_start, ...])
    pieces.append(values)
    if col_end < base.shape[1]:
        pieces.append(base[:, col_end:, ...])
    return torch.cat(tuple(pieces), dim=1)


def _clone_for_grad(base: torch.Tensor) -> torch.Tensor:
    """Clone only when grad mode needs alias isolation across live graphs."""
    return base.clone() if torch.is_grad_enabled() else base


# Eager only (a traced region keeps the guard in its graph, index_proof.py): the checked int64 gather
# indices of each index tensor, by id of its storage owner (the base of a view) and then by view layout.
# An entry holds the owner weakly, keeps the owner's current version (its views share it) and leaves
# with the owner, so index tensors that a caller rebuilds every call do not accumulate.
_GATHER_INDICES: dict[int, tuple[Any, int, dict[tuple[object, ...], torch.Tensor]]] = {}
# Eager index_select rejects every row outside [0, rows) alike (IndexError on CPU, a device assert on
# CUDA, zeros on MPS), so one rejected row serves every row count an index tensor gathers from.
_REJECTED_ROW = torch.iinfo(torch.int64).max


def _gather_index_entry(row_idx: torch.Tensor) -> dict[tuple[object, ...], torch.Tensor] | None:
    """Checked indices of ``row_idx``'s storage owner at its current version; None without a version."""
    owner = row_idx if row_idx._base is None else row_idx._base
    try:
        version = int(owner._version)
    except RuntimeError:  # inference tensors carry no version counter
        return None
    held = _GATHER_INDICES.get(id(owner))
    if held is None or held[0]() is not owner:
        weakref.finalize(owner, _GATHER_INDICES.pop, id(owner), None)
    elif held[1] == version:
        return held[2]
    _GATHER_INDICES[id(owner)] = (weakref.ref(owner), version, {})
    return _GATHER_INDICES[id(owner)][2]


def _checked_gather_index(row_idx: torch.Tensor, rows: int) -> torch.Tensor:
    """``row_idx`` as int64 with negative entries mapped to a rejected row (``rows`` when traced)."""
    eager = not torch.compiler.is_compiling() and not is_fake(row_idx)
    checked = _gather_index_entry(row_idx) if eager else None
    view = (tuple(row_idx.shape), tuple(row_idx.stride()), int(row_idx.storage_offset()))
    if checked is not None and view in checked:
        return checked[view]
    index = row_idx if row_idx.dtype == torch.int64 else row_idx.to(dtype=torch.int64)
    # Inductor wraps negative index_select indices, unlike eager PyTorch.
    # Map them to a rejected row so both paths retain checked indexing.
    index = index.masked_fill(index < 0, _REJECTED_ROW if eager else rows)
    if checked is not None:
        checked[view] = index
    return index


def _gather_rows_numeric(field: torch.Tensor, row_idx: torch.Tensor) -> torch.Tensor:
    """Gather dim-0 rows for arbitrary-shaped row indices via ``index_select``."""
    row_idx = _checked_gather_index(row_idx, field.shape[0])
    if row_idx.ndim == 1:
        return field.index_select(0, row_idx)
    flat_idx = row_idx.reshape(-1)
    gathered = field.index_select(0, flat_idx)
    return gathered.reshape(*row_idx.shape, *field.shape[1:])


# Least recently used last. One x1.40962 adjoint step uses about 180 plans of stable index sets; the
# limit keeps them while index sets of one call age out.
_ORDERED_SUM_PLANS: dict[tuple[object, ...], OrderedSumPlan] = {}
_ORDERED_SUM_PLAN_LIMIT = 512


def _static_index_plan_key(
    tag: str, device: torch.device, *tensors: torch.Tensor
) -> tuple[object, ...] | None:
    """Plan-cache name of an index set that a caller rebuilds per call from static index tensors.

    Only plan-cached devices (no ``segment_reduce``) outside tracing use plans; elsewhere None.
    """
    if device.type in ("cpu", "cuda") or torch.compiler.is_compiling():
        return None
    keys = tuple(_tensor_version_cache_key(tensor) for tensor in tensors)
    return None if any(key is None for key in keys) else (tag, *keys)


def _ordered_sum_plan(
    indices: torch.Tensor, n_source_rows: int, plan_key: tuple[object, ...] | None
) -> OrderedSumPlan:
    """The cached plan of an index set (keyed by its mutation-sensitive identity)."""
    if plan_key is None:
        tensor_key = _tensor_version_cache_key(indices)
        plan_key = None if tensor_key is None else ("indices", tensor_key)
    if plan_key is None:
        return build_ordered_sum_plan(indices, n_source_rows)
    key = (*plan_key, int(n_source_rows), str(indices.device))
    plan = _ORDERED_SUM_PLANS.pop(key, None)
    if plan is None:
        plan = build_ordered_sum_plan(indices, n_source_rows)
        if len(_ORDERED_SUM_PLANS) >= _ORDERED_SUM_PLAN_LIMIT:
            _ORDERED_SUM_PLANS.pop(next(iter(_ORDERED_SUM_PLANS)))
    _ORDERED_SUM_PLANS[key] = plan
    return plan


def _ordered_segment_sum_by_plan(
    plan: OrderedSumPlan, values: torch.Tensor, n_source_rows: int
) -> torch.Tensor:
    """Sequential ordered sums from +0.0, as CPU ``index_add``: pass k adds the k-th term
    of every source that still has one. No host read and no sort per call."""
    packed = _gather_rows_numeric(values, plan.gather_index)
    total = values.new_zeros((n_source_rows, *values.shape[1:]))
    begin = 0
    for width in plan.widths:
        total[:width] += packed[begin : begin + width]
        begin += width
    return _gather_rows_numeric(total, plan.slot)


def _ordered_index_sum_numeric(
    indices: torch.Tensor,
    values: torch.Tensor,
    n_source_rows: int,
    plan_key: tuple[object, ...] | None = None,
) -> torch.Tensor:
    """Sum each source's contributions in their original, stable order.

    ``plan_key`` names the index set for the plan cache of devices without
    ``segment_reduce`` when ``indices`` is rebuilt per call from a static set.
    """
    feature_width = 1
    for dimension in values.shape[1:]:
        feature_width *= dimension
    if feature_width == 0:
        return values.new_zeros((n_source_rows, *values.shape[1:]))
    indices = indices.to(device=values.device, dtype=torch.int64)
    if values.device.type == "cpu" and not torch.compiler.is_compiling():
        # CPU index_add adds rows in index order from +0.0: the same stable order
        # and the same bits as the segment sum below, without the sort.
        return torch.index_add(
            values.new_zeros((n_source_rows, *values.shape[1:])), 0, indices, values
        )
    if values.device.type not in ("cpu", "cuda"):
        # Torch 2.8 has segment_reduce kernels for CPU and CUDA only (MPS raises).
        if indices.numel() == 0:
            return values.new_zeros((n_source_rows, *values.shape[1:]))
        if torch.compiler.is_compiling() or is_fake(values):
            # A traced region (Dynamo, or AOT tracing a backward) cannot plan on the host: one
            # scatter-add, summed in the device's order (compiled derivatives are judged by the
            # derivative checks, not bit for bit).
            return torch.index_add(values.new_zeros((n_source_rows, *values.shape[1:])), 0, indices, values)
        plan = _ordered_sum_plan(indices, n_source_rows, plan_key)
        return _ordered_segment_sum_by_plan(plan, values, n_source_rows)
    order = torch.argsort(indices, stable=True)
    sources = _gather_rows_numeric(indices, order)
    offsets = torch.searchsorted(
        sources,
        torch.arange(n_source_rows + 1, device=indices.device, dtype=indices.dtype),
    )
    ordered = _gather_rows_numeric(values, order)
    # Torch 2.8 uses a tree reduction for one feature. A second, unused feature
    # selects its sequential per-feature CUDA sum.
    if feature_width == 1:
        ordered = torch.stack((ordered, torch.zeros_like(ordered)), dim=-1)
    # Searchsorted constructs bounded, nondecreasing offsets from checked indices.
    result = torch.segment_reduce(
        ordered, "sum", offsets=offsets, axis=0, initial=0.0, unsafe=True
    )
    return result[..., 0] if feature_width == 1 else result


class _OrderedIndexSumFunction(torch.autograd.Function):
    """The fixed-order sum is linear; retain its exact gather transpose."""

    generate_vmap_rule = True

    @staticmethod
    def forward(indices, values, n_source_rows, plan_key=None):
        return _ordered_index_sum_numeric(indices, values, n_source_rows, plan_key)

    @staticmethod
    def setup_context(ctx, inputs, output):
        indices, values, n_source_rows, *_ = inputs
        ctx.save_for_backward(indices)
        ctx.save_for_forward(indices)
        ctx.output_shape = (n_source_rows, *values.shape[1:])
        ctx.dtype = values.dtype
        ctx.device = values.device
        ctx.plan_key = inputs[3] if len(inputs) > 3 else None

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        return None, gather_rows(grad_output, indices), None, None

    @staticmethod
    def jvp(ctx, grad_indices, grad_values, grad_rows, grad_plan_key=None):
        del grad_indices, grad_rows, grad_plan_key
        (indices,) = ctx.saved_tensors
        if grad_values is None:
            return torch.zeros(ctx.output_shape, device=ctx.device, dtype=ctx.dtype)
        return _OrderedIndexSumFunction.apply(
            indices, grad_values, ctx.output_shape[0], ctx.plan_key
        )


def _ordered_index_add_rows(
    initial: torch.Tensor, indices: torch.Tensor, contribution: torch.Tensor
) -> torch.Tensor:
    """Add checked row contributions after the existing accumulator, in order."""
    rows = initial.shape[0]
    # The concatenated index set is new per call; name it by its static part.
    plan_key = _static_index_plan_key("index_add_rows", initial.device, indices)
    indices = indices.to(device=initial.device, dtype=torch.int64)
    # A separately reduced slot plus initial changes cancellation and rounding.
    ordered_indices = torch.cat(
        (torch.arange(rows, device=initial.device, dtype=torch.int64), indices)
    )
    ordered_values = torch.cat((initial, contribution))
    return _OrderedIndexSumFunction.apply(ordered_indices, ordered_values, rows, plan_key)


class _GatherRowsFunction(torch.autograd.Function):
    """Keep repeated-index transposes independent of CUDA atomic order."""

    generate_vmap_rule = True

    @staticmethod
    def forward(field, indices):
        return _gather_rows_numeric(field, indices)

    @staticmethod
    def setup_context(ctx, inputs, output):
        field, indices = inputs
        ctx.save_for_backward(indices)
        ctx.save_for_forward(indices)
        ctx.field_shape = tuple(field.shape)
        ctx.output_shape = tuple(output.shape)
        ctx.dtype = field.dtype
        ctx.device = field.device

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        return _OrderedIndexSumFunction.apply(
            indices.reshape(-1),
            grad_output.reshape(indices.numel(), *ctx.field_shape[1:]),
            ctx.field_shape[0],
            # A strided index (a connectivity column) flattens to a new tensor per call.
            _static_index_plan_key("gather_transpose", grad_output.device, indices),
        ), None

    @staticmethod
    def jvp(ctx, grad_field, grad_indices):
        (indices,) = ctx.saved_tensors
        if grad_field is None:
            return torch.zeros(ctx.output_shape, device=ctx.device, dtype=ctx.dtype)
        return _GatherRowsFunction.apply(grad_field, indices)


@torch.compiler.allow_in_graph
def gather_rows(field: torch.Tensor, row_idx: torch.Tensor) -> torch.Tensor:
    """Gather checked dim-0 rows with the same ordered transpose in both modes."""
    if _tensor_carries_autodiff(field):
        return _GatherRowsFunction.apply(field, row_idx)
    return _gather_rows_numeric(field, row_idx)


def sum_pair_rows(field: torch.Tensor, pair_idx: torch.Tensor) -> torch.Tensor:
    """Gather two row slots once and return ``slot0 + slot1`` in Fortran order."""
    if pair_idx.ndim != 2 or pair_idx.shape[1] != 2:
        raise ValueError(
            f"pair_idx must have shape (n, 2), got {tuple(pair_idx.shape)}"
        )
    pair = gather_rows(field, pair_idx)
    return pair[:, 0, ...] + pair[:, 1, ...]


def average_pair_rows(field: torch.Tensor, pair_idx: torch.Tensor) -> torch.Tensor:
    """Gather two row slots once and return their ordered arithmetic mean."""
    return 0.5 * sum_pair_rows(field, pair_idx)


def build_sparse_csr_stencil(
    indices0: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: torch.Tensor,
    n_cols: int,
) -> torch.Tensor:
    """Build a CSR sparse matrix from dense stencil indices/weights.

    Args:
        indices0: 0-based column indices, shape (n_rows, n_stencil)
        weights: per-stencil weights, same shape as indices0
        valid_mask: True where stencil entry is active, same shape
        n_cols: number of source columns in the sparse matrix

    Returns:
        torch.sparse_csr_tensor of shape (n_rows, n_cols)
    """
    if indices0.ndim != 2:
        raise ValueError(f"indices0 must be rank-2, got shape {tuple(indices0.shape)}")
    if weights.shape != indices0.shape:
        raise ValueError("weights shape must match indices0")
    if valid_mask.shape != indices0.shape:
        raise ValueError("valid_mask shape must match indices0")

    rows = indices0.shape[0]
    counts = valid_mask.sum(dim=1)
    crow = torch.zeros(rows + 1, dtype=torch.int64, device=indices0.device)
    # GPU NOTE: integer CSR row-pointer prefix sum (exact for int64).
    crow[1:] = torch.cumsum(counts, dim=0)
    col = indices0[valid_mask]
    values = weights[valid_mask]
    return torch.sparse_csr_tensor(
        crow,
        col,
        values,
        size=(rows, n_cols),
        dtype=weights.dtype,
        device=indices0.device,
    )


def apply_sparse_stencil(matrix_csr: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    """Apply a CSR stencil matrix to a field with shape (n_src, n_levels)."""
    if field.ndim != 2:
        raise ValueError(f"field must be rank-2, got shape {tuple(field.shape)}")
    if matrix_csr.shape[1] != field.shape[0]:
        raise ValueError(
            f"matrix columns ({matrix_csr.shape[1]}) must match field rows ({field.shape[0]})"
        )
    return cast(torch.Tensor, torch.sparse.mm(matrix_csr, field))


def edge_to_cell_interp(
    cell_pairs_idx: torch.Tensor,
    cell_pairs_valid: torch.Tensor,
    cell_field: torch.Tensor,
) -> torch.Tensor:
    """Pattern A: interpolate cell-centered field to edges via 0.5*(c1+c2)."""
    c1 = cell_pairs_idx[:, 0]
    c2 = cell_pairs_idx[:, 1]
    vals = 0.5 * (cell_field[c1, :] + cell_field[c2, :])
    mask = cell_pairs_valid[:, 0] & cell_pairs_valid[:, 1]
    return torch.where(mask[:, None], vals, torch.zeros_like(vals))


def cell_divergence_gather(
    edge_field: torch.Tensor,
    edge_metric: torch.Tensor,
    cell_edges_idx: torch.Tensor,
    cell_edges_sign: torch.Tensor,
    cell_edges_valid: torch.Tensor,
    inv_area_cell: torch.Tensor,
    cell_edges_valid_all_by_slot: tuple[bool, ...] | None = None,
) -> torch.Tensor:
    """Pattern B gather-reduce: cell divergence-like sum over edge stencil."""
    n_cells = cell_edges_idx.shape[0]
    # Only prepared payloads may omit a mask; live calls must not inspect it on host.
    total: torch.Tensor | None = None
    for batch_start, batch_stop in _slot_batches(
        cell_edges_idx.shape[1],
        _slot_gather_batch(edge_field, cell_edges_idx.shape[1]),
    ):
        edge_batch = _gather_slot_batch(
            edge_field, cell_edges_idx, batch_start, batch_stop
        )
        metric_batch = _gather_slot_batch(
            edge_metric, cell_edges_idx, batch_start, batch_stop
        )
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            weight = cell_edges_sign[:, slot] * metric_batch[:, local_slot]
            valid_all = (
                cell_edges_valid_all_by_slot is not None
                and cell_edges_valid_all_by_slot[slot]
            )
            if not valid_all:
                weight = torch.where(
                    cell_edges_valid[:, slot], weight, torch.zeros_like(weight)
                )
            contrib = weight.unsqueeze(1) * edge_batch[:, local_slot, :]
            total = _accumulate_ordered_slot(total, contrib)
    if total is None:
        total = edge_field.new_zeros((n_cells, edge_field.shape[1]))
    return total * inv_area_cell[:, None]


def _sum_weighted_stencil_slots(
    *,
    field: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    valid_all_by_slot: tuple[bool, ...] | None = None,
    slot_factor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Accumulate a fixed-width ordered stencil without materializing all slots."""
    compiled = _compiled_stencil_reducer(
        "stencil_ordered_weighted_sum",
        _ordered_weighted_stencil_program,
        field,
        indices,
        weights,
        valid_mask,
        slot_factor,
        metadata=valid_all_by_slot,
    )
    program = compiled if compiled is not None else _ordered_weighted_stencil_program
    return program(
        field=field,
        indices=indices,
        weights=weights,
        valid_mask=valid_mask,
        valid_all_by_slot=valid_all_by_slot,
        slot_factor=slot_factor,
    )


def _ordered_weighted_stencil_program(
    *,
    field: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    valid_all_by_slot: tuple[bool, ...] | None = None,
    slot_factor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single tensor program for eager, compiled and differentiated reducers."""
    if field.ndim != 2:
        raise ValueError(f"field must be rank-2, got shape {tuple(field.shape)}")
    if indices.ndim != 2:
        raise ValueError(f"indices must be rank-2, got shape {tuple(indices.shape)}")
    if weights is not None and weights.shape != indices.shape:
        raise ValueError("weights shape must match indices")
    if valid_mask is not None and valid_mask.shape != indices.shape:
        raise ValueError("valid_mask shape must match indices")
    if (
        field.device.type == "cuda"
        and not torch.is_grad_enabled()
        and not _compile_active()
    ):
        return _sum_weighted_stencil_view_reduce(
            field=field,
            indices=indices,
            weights=weights,
            valid_mask=valid_mask,
            valid_all_by_slot=valid_all_by_slot,
            slot_factor=slot_factor,
        )

    n_rows = indices.shape[0]
    total: torch.Tensor | None = None
    for batch_start, batch_stop in _slot_batches(
        indices.shape[1],
        _slot_gather_batch(field, indices.shape[1]),
    ):
        values_batch = _gather_slot_batch(field, indices, batch_start, batch_stop)
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            contrib = values_batch[:, local_slot, :]
            if weights is not None:
                contrib = weights[:, slot].unsqueeze(1) * contrib
            valid_all = valid_all_by_slot is not None and valid_all_by_slot[slot]
            if valid_mask is not None and not valid_all:
                contrib = torch.where(
                    valid_mask[:, slot].unsqueeze(1),
                    contrib,
                    torch.zeros_like(contrib),
                )
            if slot_factor is not None:
                contrib = contrib * slot_factor
            total = _accumulate_ordered_slot(total, contrib)
    if total is None:
        return field.new_zeros((n_rows, field.shape[1]))
    return total


def _sum_weighted_stencil_view_reduce(
    *,
    field: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    valid_all_by_slot: tuple[bool, ...] | None = None,
    slot_factor: torch.Tensor | None = None,
) -> torch.Tensor:
    """No-grad row-owned stencil reduction without atomics.

    The row/slot relation is already grouped by row, so the flattened gather can
    be viewed as ``(n_rows, n_slots, n_levels)`` and reduced in upstream slot
    order.  This avoids atomic row accumulation on CUDA and remains deterministic.
    """
    n_rows = indices.shape[0]
    n_slots = indices.shape[1]
    if n_slots == 0:
        return field.new_zeros((n_rows, field.shape[1]))
    contrib = gather_rows(field, indices.reshape(-1)).reshape(
        n_rows, n_slots, field.shape[1]
    )
    if weights is not None:
        contrib = weights.unsqueeze(2) * contrib
    valid_all = (
        valid_all_by_slot is not None
        and len(valid_all_by_slot) == n_slots
        and all(valid_all_by_slot)
    )
    if valid_mask is not None and not valid_all:
        contrib = contrib.masked_fill(~valid_mask.unsqueeze(2), 0.0)
    if slot_factor is not None:
        factor = slot_factor.expand(n_rows, field.shape[1]).unsqueeze(1)
        contrib = contrib * factor
    return sum_fortran_order(contrib, dim=1)


def _sum_weighted_stencil_components_view_reduce(
    *,
    field: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    valid_all_by_slot: tuple[bool, ...] | None = None,
) -> torch.Tensor:
    """Ordered components with one live slot, not a [rows, slots, levels, xyz] product."""
    n_rows = indices.shape[0]
    n_slots = indices.shape[1]
    n_components = weights.shape[2]
    if n_slots == 0:
        return field.new_zeros((n_rows, field.shape[1], n_components))
    total = None
    for slot in range(n_slots):
        values = gather_rows(field, indices[:, slot])
        contrib = values.unsqueeze(2) * weights[:, slot, :].unsqueeze(1)
        valid_all = (valid_all_by_slot is not None and len(valid_all_by_slot) == n_slots
                     and valid_all_by_slot[slot])
        if valid_mask is not None and not valid_all:
            contrib = contrib.masked_fill(~valid_mask[:, slot, None, None], 0.0)
        total = _accumulate_ordered_slot(total, contrib)
    # MPAS initializes reconstruction components to positive zero. For finite
    # sums this final addition preserves the same sign at an all-zero stencil.
    return total + field.new_zeros(
        (n_rows, field.shape[1], n_components)
    )


def _sum_weighted_stencil_components(
    *,
    field: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    valid_all_by_slot: tuple[bool, ...] | None = None,
) -> torch.Tensor:
    """Accumulate ordered stencil components with one source gather per slot."""
    compiled = _compiled_stencil_reducer(
        "stencil_ordered_weighted_components",
        _ordered_weighted_components_program,
        field,
        indices,
        weights,
        valid_mask,
        metadata=valid_all_by_slot,
    )
    program = compiled if compiled is not None else _ordered_weighted_components_program
    return program(
        field=field,
        indices=indices,
        weights=weights,
        valid_mask=valid_mask,
        valid_all_by_slot=valid_all_by_slot,
    )


def _ordered_weighted_components_program(
    *,
    field: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    valid_all_by_slot: tuple[bool, ...] | None = None,
) -> torch.Tensor:
    """One component tensor program; policy dispatch never enters the graph."""
    if field.ndim != 2:
        raise ValueError(f"field must be rank-2, got shape {tuple(field.shape)}")
    if indices.ndim != 2:
        raise ValueError(f"indices must be rank-2, got shape {tuple(indices.shape)}")
    if weights.ndim != 3 or weights.shape[:2] != indices.shape:
        raise ValueError("weights must have shape (*indices.shape, n_components)")
    if valid_mask is not None and valid_mask.shape != indices.shape:
        raise ValueError("valid_mask shape must match indices")
    if (
        field.device.type == "cuda" or _compile_active()
    ) and not torch.is_grad_enabled():
        return _sum_weighted_stencil_components_view_reduce(
            field=field,
            indices=indices,
            weights=weights,
            valid_mask=valid_mask,
            valid_all_by_slot=valid_all_by_slot,
        )

    n_rows = indices.shape[0]
    n_components = weights.shape[2]
    total = field.new_zeros((n_rows, field.shape[1], n_components))
    for batch_start, batch_stop in _slot_batches(
        indices.shape[1],
        _slot_gather_batch(field, indices.shape[1]),
    ):
        values_batch = _gather_slot_batch(field, indices, batch_start, batch_stop)
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            values = values_batch[:, local_slot, :]
            contrib = values.unsqueeze(2) * weights[:, slot, :].unsqueeze(1)
            valid_all = valid_all_by_slot is not None and valid_all_by_slot[slot]
            if valid_mask is not None and not valid_all:
                contrib = torch.where(
                    valid_mask[:, slot].view(n_rows, 1, 1),
                    contrib,
                    torch.zeros_like(contrib),
                )
            total = _accumulate_ordered_slot(total, contrib)
    return total


def vertex_vorticity_gather(
    edge_field: torch.Tensor,
    edge_metric: torch.Tensor,
    vertex_edges_idx: torch.Tensor,
    vertex_edges_sign: torch.Tensor,
    vertex_edges_valid: torch.Tensor,
    inv_area_triangle: torch.Tensor,
) -> torch.Tensor:
    """Pattern D gather-reduce: vertex circulation-like sum over edge stencil."""
    weights = vertex_edges_sign * edge_metric[vertex_edges_idx]
    return (
        _sum_weighted_stencil_slots(
            field=edge_field,
            indices=vertex_edges_idx,
            weights=weights,
            valid_mask=vertex_edges_valid,
        )
        * inv_area_triangle[:, None]
    )


def edge_tangential_velocity_gather(
    edge_field: torch.Tensor,
    edge_edges_idx: torch.Tensor,
    edge_edges_weights: torch.Tensor,
    edge_edges_valid: torch.Tensor,
) -> torch.Tensor:
    """Pattern E gather-reduce: tangential reconstruction from edge-neighbor stencil."""
    return _sum_weighted_stencil_slots(
        field=edge_field,
        indices=edge_edges_idx,
        weights=edge_edges_weights,
        valid_mask=edge_edges_valid,
    )


def build_divergence_csr(
    cell_edges_idx: torch.Tensor,
    cell_edges_sign: torch.Tensor,
    cell_edges_valid: torch.Tensor,
    dv_edge: torch.Tensor,
    inv_area_cell: torch.Tensor,
    n_edges: int,
) -> torch.Tensor:
    """Build CSR operator for divergence-like Pattern B."""
    weights = cell_edges_sign * dv_edge[cell_edges_idx] * inv_area_cell[:, None]
    return build_sparse_csr_stencil(
        cell_edges_idx, weights, cell_edges_valid, n_cols=n_edges
    )


def build_vorticity_csr(
    vertex_edges_idx: torch.Tensor,
    vertex_edges_sign: torch.Tensor,
    vertex_edges_valid: torch.Tensor,
    dc_edge: torch.Tensor,
    inv_area_triangle: torch.Tensor,
    n_edges: int,
) -> torch.Tensor:
    """Build CSR operator for vorticity-like Pattern D."""
    weights = vertex_edges_sign * dc_edge[vertex_edges_idx] * inv_area_triangle[:, None]
    return build_sparse_csr_stencil(
        vertex_edges_idx, weights, vertex_edges_valid, n_cols=n_edges
    )


def build_tangential_velocity_csr(
    edge_edges_idx: torch.Tensor,
    weights_on_edge: torch.Tensor,
    edge_edges_valid: torch.Tensor,
    n_edges: int,
) -> torch.Tensor:
    """Build CSR operator for tangential reconstruction Pattern E."""
    return build_sparse_csr_stencil(
        edge_edges_idx, weights_on_edge, edge_edges_valid, n_cols=n_edges
    )
