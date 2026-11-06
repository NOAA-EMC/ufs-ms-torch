# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Consumed-index proof for compiling CUDA regions without bounds asserts.

Inductor checks every indirect load (``tl.device_assert``) unless it can prove
the index range itself, which it cannot for indices loaded from memory. The
checks cost about 6-7% of a forecast step. A compiled graph may run without
them only when, for the call at hand:

1. every index it consumes is a graph input, reached through view-only steps
   (and the negative-index guard of ``_gather_rows_numeric``), and the graph
   contains no other indexing operation; and
2. each such input tensor, as passed to this call, is validated: all values are
   in ``[0, rows)`` of the dimension it indexes. Validation runs once per tensor
   identity, version and storage, then is reused.

The backend analyzes each graph Dynamo produces and compiles the kernels its
calls run: unchecked for a proven no-grad graph, checked otherwise. Dynamo
guards grad mode and ``requires_grad``, so a graph compiled with grad never
runs unchecked kernels. A dispatcher checks the actual graph inputs of every
call; a call that fails validation runs checked kernels, which a proven graph
compiles on that first call only. Anything unproven keeps the checked kernels.
Only nondifferentiable calls use unchecked kernels.
"""

from __future__ import annotations

import copy
import inspect
import operator
import threading
import weakref
from collections.abc import Callable
from typing import Any

import torch

# (index input position, operand input position or None, operand dim, static rows)
Requirement = tuple[int, int | None, int, int]
# Process-wide evidence of which kernels CUDA region calls actually used.
COUNTERS = {"checked_calls": 0, "unchecked_calls": 0}

# Methods and functions that only reshape or select without new values.
_VIEW_METHODS = {
    "view",
    "reshape",
    "flatten",
    "unsqueeze",
    "squeeze",
    "contiguous",
    "expand",
    "select",
    "narrow",
    "t",
    "permute",
    "transpose",
}
_VIEW_FUNCTIONS = {
    torch.reshape,
    torch.flatten,
    torch.unsqueeze,
    torch.squeeze,
    torch.select,
    torch.narrow,
    torch.permute,
    torch.transpose,
}
_INDEXING = {
    "index_select",
    "index_add",
    "index_add_",
    "index_copy",
    "index_copy_",
    "index_put",
    "index_put_",
    "index_fill",
    "index_fill_",
    "scatter",
    "scatter_",
    "scatter_add",
    "scatter_add_",
    "scatter_reduce",
    "gather",
    "take",
    "take_along_dim",
    "embedding",
    "repeat_interleave",
    "searchsorted",
    "bucketize",
    "nonzero",
    "masked_select",
    "index_reduce",
}


def _target_name(node: torch.fx.Node) -> str:
    target = node.target
    return target if isinstance(target, str) else getattr(target, "__name__", "")


def _is_integer(node: torch.fx.Node) -> bool:
    value = node.meta.get("example_value")
    return (
        isinstance(value, torch.Tensor)
        and not value.is_floating_point()
        and (value.dtype != torch.bool)
    )


def _static_size(node: torch.fx.Node, dim: int) -> int | None:
    value = node.meta.get("example_value")
    if not isinstance(value, torch.Tensor) or dim >= value.ndim:
        return None
    size = value.shape[dim]
    return size if isinstance(size, int) else None


def _step(node: Any) -> Any:
    """The input of a view-only step (or the negative guard), else None."""
    name = _target_name(node)
    args: Any = node.args
    if node.op == "call_method" and name in _VIEW_METHODS:
        return args[0]
    if node.op == "call_method" and name in ("to", "long"):
        return args[0] if _is_integer(node) else None
    if node.op == "call_function" and node.target in _VIEW_FUNCTIONS:
        return args[0]
    if node.op == "call_function" and node.target is operator.getitem:
        items = args[1] if isinstance(args[1], tuple) else (args[1],)
        if any(isinstance(item, torch.fx.Node) for item in items):
            return None
        return args[0]
    if node.op in ("call_method", "call_function") and name == "masked_fill":
        # _gather_rows_numeric: idx.masked_fill(idx < 0, n) is the identity
        # once the input minimum is proven non-negative.
        mask = args[1]
        if (
            isinstance(mask, torch.fx.Node)
            and _target_name(mask) == "lt"
            and mask.args[0] is args[0]
            and mask.args[1] == 0
        ):
            return args[0]
    return None


def _clamp_bounds(node: Any) -> tuple[Any, Any] | None:
    name = _target_name(node)
    if name not in ("clamp", "clip"):
        return None
    args: Any = node.args
    kwargs: Any = node.kwargs
    low = args[1] if len(args) > 1 else kwargs.get("min")
    high = args[2] if len(args) > 2 else kwargs.get("max")
    return low, high


def _prove(
    index: Any,
    operand: torch.fx.Node,
    dim: int,
    positions: dict[torch.fx.Node, int],
    depth: int = 0,
) -> list[Requirement] | None:
    """Requirements under which ``index`` addresses rows of ``operand``.

    The index must reach an integer input through view-only steps, a clamp
    to constant bounds inside the rows, or a lookup in an integer input table
    whose own index is proven the same way.
    """
    rows = _static_size(operand, dim)
    if rows is None or depth > 2:
        return None
    operand_position = positions.get(operand)
    node = index
    while isinstance(node, torch.fx.Node):
        if node.op == "placeholder":
            if not _is_integer(node):
                return None
            return [(positions[node], operand_position, dim, rows)]
        if _target_name(node) == "where" and len(node.args) == 3:
            # An upwind choice between two proven indices stays in range.
            chosen = [
                _prove(branch, operand, dim, positions, depth + 1)
                for branch in node.args[1:]
            ]
            if any(found is None for found in chosen):
                return None
            return [item for found in chosen if found is not None for item in found]
        bounds = _clamp_bounds(node)
        if bounds is not None:
            low, high = bounds
            if (
                isinstance(low, int)
                and isinstance(high, int)
                and 0 <= low <= high < rows
            ):
                return []
            return None
        table = _uses(node)
        if (
            table
            and len(table) == 1
            and _target_name(node) in ("index_select", "getitem")
        ):
            inner, table_node, table_dim = table[0]
            # Values come from the table: the whole table must fit the rows,
            # and the inner index must fit the table.
            values = _prove(table_node, operand, dim, positions, depth + 1)
            selected = _prove(inner, table_node, table_dim, positions, depth + 1)
            if values is None or selected is None:
                return None
            return values + selected
        node = _step(node)
    return None


def _uses(node: torch.fx.Node) -> list[tuple[Any, torch.fx.Node, int]] | None:
    """(index, indexed operand, dim) per index of an indexing node; [] if none.

    None marks indexing that cannot be mapped to a dimension.
    """
    name = _target_name(node)
    args: Any = node.args
    if name == "index_select":
        kwargs: Any = node.kwargs
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        index = args[2] if len(args) > 2 else kwargs.get("index")
        if not isinstance(dim, int) or not isinstance(args[0], torch.fx.Node):
            return None
        return [(index, args[0], dim)]
    if name in ("index_copy", "index_copy_", "index_add", "index_add_"):
        # Writes through an index: rows of the destination along ``dim``.
        dim = args[1] if len(args) > 1 else None
        index = args[2] if len(args) > 2 else None
        if not isinstance(dim, int) or not isinstance(args[0], torch.fx.Node):
            return None
        return [(index, args[0], dim)]
    if name in ("gather_rows", "static_pair_gather"):
        indices = [args[1]] if name == "gather_rows" else [args[1], args[2]]
        return [(index, args[0], 0) for index in indices]
    if node.target is operator.getitem and isinstance(args[0], torch.fx.Node):
        items = args[1] if isinstance(args[1], tuple) else (args[1],)
        uses = []
        for dim, item in enumerate(items):
            if isinstance(item, torch.fx.Node):
                uses.append((item, args[0], dim))
        if uses and any(item is None or item is Ellipsis for item in items):
            return None
        return uses
    return []


def analyze(gm: torch.fx.GraphModule) -> tuple[Requirement, ...] | None:
    """Requirements for an assert-free compile of ``gm``, or None if unproven."""
    requirements: list[Requirement] = []
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    positions = {node: position for position, node in enumerate(placeholders)}
    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_method"):
            continue
        name = _target_name(node)
        uses = _uses(node)
        if uses is None:
            return None
        if not uses:
            if name in _INDEXING:
                return None
            target = node.target
            # Unknown Python functions (allow_in_graph helpers) may index.
            if (
                node.op == "call_function"
                and inspect.isfunction(target)
                and not (getattr(target, "__module__", "") or "").startswith(
                    ("torch", "_operator", "operator", "math")
                )
            ):
                return None
            continue
        for index, operand, dim in uses:
            found = _prove(index, operand, dim, positions)
            if found is None:
                return None
            requirements.extend(found)
    return tuple(requirements)


class _Validated:
    """Index range per tensor identity, version and storage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ranges: dict[int, tuple[tuple[object, ...], int, int]] = {}

    def range(self, tensor: torch.Tensor) -> tuple[int, int] | None:
        try:
            stamp = (tensor._version, tensor.data_ptr(), tuple(tensor.shape))
        except RuntimeError:  # inference tensors carry no version counter
            return None
        key = id(tensor)
        with self._lock:
            cached = self._ranges.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1], cached[2]
        if tensor.numel() == 0:
            low, high = 0, -1
        else:
            # One host read per static index tensor version, not per call.
            low, high = (int(v) for v in torch.aminmax(tensor))
        with self._lock:
            self._ranges[key] = (stamp, low, high)
        weakref.finalize(tensor, self._ranges.pop, key, None)
        return low, high


VALIDATED = _Validated()


def _proven(requirements: tuple[Requirement, ...], inputs: Any) -> bool:
    for index_position, operand_position, dim, rows in requirements:
        index = inputs[index_position]
        if not isinstance(index, torch.Tensor):
            return False
        if operand_position is not None:
            operand = inputs[operand_position]
            if not isinstance(operand, torch.Tensor) or operand.ndim <= dim:
                return False
            rows = operand.shape[dim]
        bounds = VALIDATED.range(index)
        if bounds is None or bounds[0] < 0 or bounds[1] >= rows:
            return False
    return True


def _needs_grad(inputs: Any) -> bool:
    return torch.is_grad_enabled() and any(
        isinstance(value, torch.Tensor) and value.requires_grad for value in inputs
    )


def _invoke(fn: Callable[..., Any], inputs: list[Any]) -> Any:
    return fn(inputs) if getattr(fn, "_boxed_call", False) else fn(*inputs)


class _OnDemand:
    """Checked kernels of a proven graph, compiled on the first call that needs them.

    A proven no-grad graph runs them only when validation fails (a bad index
    or an unvalidated input). They come from Dynamo tracing the graph's
    unchanged copy, so guards and dynamic shapes follow the actual inputs.
    """

    def __init__(self, gm: torch.fx.GraphModule, options: dict[str, Any]) -> None:
        self._gm = gm
        self._options = options
        self._compiled: Callable[..., Any] | None = None
        self._lock = threading.Lock()

    def __call__(self, *inputs: Any) -> Any:
        with self._lock:
            if self._compiled is None:
                options, gm = self._options, self._gm
                self._compiled = torch.compile(
                    gm,
                    backend=lambda graph, example: _compile_fx(graph, example, options),
                )
        return self._compiled(*inputs)


def _dispatch(
    checked: Callable[..., Any],
    unchecked: Callable[..., Any] | None,
    requirements: tuple[Requirement, ...],
) -> Callable[..., Any]:
    """One graph's call: unchecked kernels only for validated, no-grad inputs."""

    def run_list(inputs: list[Any]) -> Any:
        if (
            unchecked is not None
            and not _needs_grad(inputs)
            and _proven(requirements, inputs)
        ):
            COUNTERS["unchecked_calls"] += 1
            return _invoke(unchecked, inputs)
        COUNTERS["checked_calls"] += 1
        return _invoke(checked, inputs)

    compiled = unchecked if unchecked is not None else checked
    if getattr(compiled, "_boxed_call", False):
        # Boxed graphs take one list (and may clear it): choose first.
        run_list._boxed_call = True  # type: ignore[attr-defined]
        return run_list

    def run(*inputs: Any) -> Any:
        return run_list(list(inputs))

    return run


def _compile_fx(
    gm: torch.fx.GraphModule, example_inputs: Any, options: dict[str, Any]
) -> Callable[..., Any]:
    # Looked up per compile: tests count compiles with a stand-in.
    from torch._inductor import compile_fx as inductor

    return inductor.compile_fx(gm, example_inputs, config_patches=options)


_BACKENDS: dict[str, Callable[..., Any]] = {}
_BACKENDS_LOCK = threading.Lock()


def proven_index_backend(options: dict[str, Any]) -> Callable[..., Any]:
    """The Dynamo backend for ``options``; equal options share one object.

    Dynamo keys cached graphs by backend identity. compile_utils creates one
    wrapper per specialization key, and per-wrapper backends filled each
    region's cache until Dynamo hit recompile_limit and ran it eager (GPU
    session nccl-r3b: 7194 instead of 1888 kernels per step).
    """
    checked_options = {**options, "assert_indirect_indexing": True}
    unchecked_options = {**options, "assert_indirect_indexing": False}
    key = repr(sorted(checked_options.items()))
    with _BACKENDS_LOCK:
        backend = _BACKENDS.get(key)
        if backend is not None:
            return backend

        def backend(gm, example_inputs):
            # The proofs and the unchecked kernels are validated for CUDA graphs only (which may
            # also take CPU scalars).
            devices = {t.device.type for t in example_inputs if isinstance(t, torch.Tensor)}
            if "cuda" not in devices:
                raise ValueError(f"the proven-index backend compiles CUDA graphs only, got {sorted(devices)}")
            requirements = analyze(gm)
            # Grad mode and requires_grad are guarded: a graph compiled with
            # grad, like an unproven one, never runs unchecked kernels.
            if requirements is None or _needs_grad(example_inputs):
                checked = _compile_fx(gm, example_inputs, checked_options)
                return _dispatch(checked, None, ())
            # Inductor may rewrite the graph it compiles: copy before.
            checked = _OnDemand(copy.deepcopy(gm), checked_options)
            unchecked = _compile_fx(gm, example_inputs, unchecked_options)
            return _dispatch(checked, unchecked, requirements)

        # Exposed for policy checks: the variants differ only in the flag.
        backend.inductor_options = checked_options  # type: ignore[attr-defined]
        backend.unchecked_options = unchecked_options  # type: ignore[attr-defined]
        _BACKENDS[key] = backend
        return backend


class ProvenIndexCompiled:
    """A CUDA region compiled with the shared proven-index backend."""

    def __init__(
        self,
        fn: Callable[..., Any],
        options: dict[str, Any],
        compile_fn: Callable[..., Any],
    ) -> None:
        self.compiled = compile_fn(fn, backend=proven_index_backend(options))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.compiled(*args, **kwargs)
