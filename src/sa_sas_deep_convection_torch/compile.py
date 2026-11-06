# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Scoped ``torch.compile`` controls for SA-SAS pure tensor islands."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
import dataclasses
import hashlib
import types
from typing import Any, TypeVar

import torch
from ufs_physics_common.ordered_sum import parallel_masked_sums

TensorResult = TypeVar("TensorResult")

COMPILE_CATEGORIES = frozenset(
    {
        "a_entry_init",
        "b_workspace_seed",
        "c_geometry_thermo",
        "d_source_lfc",
        "e_trigger_filters",
        "f_entrainment_rates",
        "f_ordered_chunks",
        "g_entrainment_formula",
        "g_mass_flux_recurrence",
        "i_ordered_chunks",
        "j_ordered_chunks",
        "g_ordered_chunks",
        "h_lfc_cloudtop",
        "i_updraft_precip",
        "j_diagnostics",
        "k_shear_downdraft_seed",
        "k_ordered_chunks",
        "l_downdraft",
        "l_ordered_chunks",
        "m_feedback",
        "n_tvd_limiter",
        "n_tensor_backward",
        "o_asqec_updraft",
        "p_asqec_downdraft",
        "q_progsigma_inputs",
        "r_mass_flux_scale",
        "s_feedback",
        "t_ordered_chunks",
        "t_evap_wetdep",
        "u_outputs",
        "v_post",
    }
)
DEFAULT_COMPILE_CATEGORIES = frozenset(
    {
        "d_source_lfc",
        "e_trigger_filters",
        "f_entrainment_rates",
        "g_entrainment_formula",
        "g_mass_flux_recurrence",
        "i_updraft_precip",
        "j_diagnostics",
        "k_shear_downdraft_seed",
        "l_downdraft",
        "m_feedback",
        "n_tvd_limiter",
        "o_asqec_updraft",
        "p_asqec_downdraft",
        "s_feedback",
        "t_evap_wetdep",
    }
)

_enabled_categories: frozenset[str] = frozenset()
_explicit_categories = False
_recurrence_chunk_size = 8
_symbolic_columns = False
_COLUMN_HELPER_CATEGORIES = frozenset({
    "f_ordered_chunks", "g_ordered_chunks", "i_ordered_chunks",
    "j_ordered_chunks", "k_ordered_chunks", "l_ordered_chunks",
    "n_tensor_backward", "t_ordered_chunks",
})
_compile_cache: dict[tuple[Any, ...], Callable[..., Any]] = {}
_compile_status: dict[str, dict[str, Any]] = {}
_regional = False


@contextmanager
def regional_compile():
    """Bounded recurrence graphs under the coupled exact-numerics policy."""
    global _regional
    previous, _regional = _regional, True
    try:
        with compile_categories(COMPILE_CATEGORIES, recurrence_chunk_size=8):
            yield
    finally:
        _regional = previous


def normalize_compile_categories(categories: Iterable[str] | str | None) -> frozenset[str]:
    if categories is None:
        return frozenset()
    if isinstance(categories, str):
        value = categories.strip().lower()
        if value == "default":
            return DEFAULT_COMPILE_CATEGORIES
        if value in {"all", "on", "true"}:
            return COMPILE_CATEGORIES
        if value in {"", "none", "off", "false"}:
            return frozenset()
        raw = [item.strip() for item in categories.split(",")]
    else:
        raw = [str(item).strip() for item in categories]
    normalized = frozenset(item for item in raw if item)
    unknown = normalized - COMPILE_CATEGORIES
    if unknown:
        raise ValueError(f"unknown SA-SAS compile categories: {sorted(unknown)}")
    return normalized


def compile_categories_enabled() -> frozenset[str]:
    return _enabled_categories


def compile_categories_explicitly_set() -> bool:
    return _explicit_categories


def set_compile_categories(categories: Iterable[str] | str | None) -> tuple[frozenset[str], bool]:
    global _enabled_categories, _explicit_categories
    previous = (_enabled_categories, _explicit_categories)
    _enabled_categories = normalize_compile_categories(categories)
    _explicit_categories = True
    return previous


def restore_compile_categories(previous: Iterable[str] | str | tuple[frozenset[str], bool] | None) -> None:
    global _enabled_categories, _explicit_categories
    if isinstance(previous, tuple) and len(previous) == 2 and isinstance(previous[0], frozenset):
        _enabled_categories = previous[0]
        _explicit_categories = bool(previous[1])
        return
    set_compile_categories(previous)


@contextmanager
def compile_categories(categories: Iterable[str] | str | None, *, recurrence_chunk_size: int = 8,
                       symbolic_columns: bool = False):
    global _recurrence_chunk_size, _symbolic_columns
    if type(recurrence_chunk_size) is not int or recurrence_chunk_size not in (4, 8):
        raise ValueError("deep recurrence chunk size must be4 or8")
    if type(symbolic_columns) is not bool:
        raise ValueError("symbolic_columns must be a Boolean")
    previous_size = _recurrence_chunk_size
    previous_symbolic = _symbolic_columns
    previous = set_compile_categories(categories)
    _recurrence_chunk_size = recurrence_chunk_size
    _symbolic_columns = symbolic_columns
    try:
        yield
    finally:
        restore_compile_categories(previous)
        _recurrence_chunk_size = previous_size
        _symbolic_columns = previous_symbolic


def clear_compile_cache() -> None:
    _compile_cache.clear()
    _compile_status.clear()


def compile_island_status() -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in _compile_status.items()}


def _tensor_spec(value: torch.Tensor, columns: int | None = None, column_axis: int | None = 0) -> tuple[Any, ...]:
    shape = tuple(value.shape)
    strides = tuple(value.stride())
    if columns is not None and column_axis is not None and value.ndim > column_axis and shape[column_axis] == columns:
        shape = (*shape[:column_axis], "columns", *shape[column_axis + 1:])
        if column_axis and value.is_contiguous():
            # Packed rows have column-dependent outer strides, fixed by their contiguous shape.
            strides = ("contiguous",)
    return (
        "tensor",
        shape,
        str(value.dtype),
        str(value.device),
        bool(value.requires_grad),
        int(value.layout == torch.strided),
        strides,
    )


def _literal_spec(value: Any, columns: int | None = None, column_axis: int | None = 0) -> tuple[Any, ...]:
    if isinstance(value, torch.Tensor):
        return _tensor_spec(value, columns, column_axis)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return ("literal", type(value).__name__, value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_literal_spec(item, columns) for item in value))
    if dataclasses.is_dataclass(value) and not isinstance(value, type) and value.__dataclass_params__.frozen:
        # Frozen host-side descriptors (e.g. the convective tracer layout) are literals.
        return ("dataclass", type(value).__name__,
                tuple(_literal_spec(getattr(value, f.name), columns) for f in dataclasses.fields(value)))
    raise TypeError(
        "compiled SA-SAS islands accept only tensors and stable Python literals; "
        f"got {type(value).__name__}"
    )


def _has_gradient_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.requires_grad)
    if isinstance(value, tuple):
        return any(_has_gradient_tensor(item) for item in value)
    return False


def _column_count(category: str, args: tuple[Any, ...]) -> int | None:
    # The first mask/vector declares the column count; level/tracer axes stay fixed.
    if _symbolic_columns and category in _COLUMN_HELPER_CATEGORIES and args:
        first = args[0]
        if isinstance(first, torch.Tensor) and first.ndim == 1 and first.shape[0] > 1:
            return first.shape[0]
    return None


def _column_axis(name):
    if name in {"profile_rows", "tracer_rows"}:
        return 2
    return None if name == "fscav" else 0


def _column_axes(fn, args):
    # Packet columns are axis two; fscav is a fixed tracer vector, not columns.
    code = getattr(fn, "__code__", None)
    names = code.co_varnames[:len(args)] if code is not None else (None,) * len(args)
    return tuple(_column_axis(name) for name in names)


def _cache_key(category: str, name: str, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    columns = _column_count(category, args)
    kw_specs = tuple((key, _literal_spec(value, columns, _column_axis(key)))
                     for key, value in sorted(kwargs.items()))
    return (category, name, id(fn), torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
            columns is not None,
            tuple(_literal_spec(value, columns, axis) for value, axis in zip(args, _column_axes(fn, args), strict=True)),
            kw_specs)


@contextmanager
def _temporary_column_marks(args, kwargs, columns, fn=None):
    missing = object()
    saved = []
    seen = set()
    names = ("_dynamo_dynamic_indices", "_dynamo_dynamic_range", "_specialize_on")

    def mark(value, axis=0):
        if isinstance(value, tuple):
            for item in value:
                mark(item, axis)
        elif (isinstance(value, torch.Tensor) and axis is not None and value.ndim > axis
              and value.shape[axis] == columns and id(value) not in seen):
            seen.add(id(value))
            original = tuple((name, getattr(value, name, missing)) for name in names)
            saved.append((value, original))
            # mark_dynamic mutates existing sets/dicts. Work on copies and
            # restore the caller's original objects even when tracing fails.
            for name, previous in original:
                if isinstance(previous, (set, dict)):
                    setattr(value, name, previous.copy())
            torch._dynamo.mark_dynamic(value, axis)

    try:
        for value, axis in zip(args, _column_axes(fn, args), strict=True):
            mark(value, axis)
        for name, value in kwargs.items():
            mark(value, _column_axis(name))
        yield
    finally:
        for value, original in saved:
            for name, previous in original:
                if previous is missing:
                    if hasattr(value, name):
                        delattr(value, name)
                else:
                    setattr(value, name, previous)


def _compile_specialized_callable(fn: Callable[..., TensorResult], key: tuple[Any, ...]) -> Callable[..., TensorResult]:
    code = getattr(fn, "__code__", None)
    if code is None:
        return fn
    suffix = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:16]
    name = f"{getattr(fn, '__name__', 'sa_sas_island')}_{suffix}"
    replacements: dict[str, str] = {"co_name": name}
    if hasattr(code, "co_qualname"):
        replacements["co_qualname"] = name
    specialized_code = code.replace(**replacements)
    specialized = types.FunctionType(
        specialized_code,
        fn.__globals__,
        name,
        fn.__defaults__,
        fn.__closure__,
    )
    specialized.__kwdefaults__ = getattr(fn, "__kwdefaults__", None)
    specialized.__annotations__ = getattr(fn, "__annotations__", {})
    specialized.__module__ = getattr(fn, "__module__", None)
    return specialized


def run_compile_island(
    category: str,
    name: str,
    fn: Callable[..., TensorResult],
    *args: Any,
    **kwargs: Any,
) -> TensorResult:
    if category not in COMPILE_CATEGORIES:
        raise ValueError(f"unknown SA-SAS compile category {category!r}")
    if torch.compiler.is_compiling():
        return fn(*args, **kwargs)
    if _regional:
        from ufs_physics_common.compile_policy import compiled

        if name in {"join_levels", "_compact_select", "_compact_merge"}:
            return compiled(fn, normalize_layouts=False)(*args, **kwargs)
        return compiled(fn)(*args, **kwargs)
    status_key = f"{category}:{name}"
    if category not in _enabled_categories:
        _compile_status[status_key] = {"status": "eager_disabled"}
        return fn(*args, **kwargs)
    if category not in {"f_ordered_chunks", "g_ordered_chunks", "l_ordered_chunks", "n_tensor_backward", "i_ordered_chunks", "j_ordered_chunks", "k_ordered_chunks", "t_ordered_chunks"} and torch.is_grad_enabled() and (any(_has_gradient_tensor(arg) for arg in args) or any(_has_gradient_tensor(value) for value in kwargs.values())):
        _compile_status[status_key] = {
            "status": "eager_grad_enabled",
            "reason": "compiled SA-SAS islands are forward-only until backward compile closure is profiled",
        }
        return fn(*args, **kwargs)
    if not hasattr(torch, "compile"):
        _compile_status[status_key] = {"status": "compile_unavailable"}
        raise RuntimeError("torch.compile is unavailable")

    key = _cache_key(category, name, fn, args, kwargs)
    columns = _column_count(category, args)
    compiled = _compile_cache.get(key)
    created = compiled is None
    if compiled is None:
        _compile_status[status_key] = {
            "status": "compiling",
            "cache_key": repr(key),
            "autodiff_stabilized": False,
        }
        compiled = torch.compile(_compile_specialized_callable(fn, key), fullgraph=True,
                                 dynamic=None if columns is not None else False, mode=None)
        _compile_cache[key] = compiled
    _compile_status[status_key] = {
        "status": "compiled",
        "cache_key": repr(key),
        "autodiff_stabilized": False,
        "symbolic_columns": columns is not None,
    }
    try:
        if created and columns is not None:
            with _temporary_column_marks(args, kwargs, columns, fn):
                return compiled(*args, **kwargs)
        return compiled(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - compile failures are evidence.
        _compile_status[status_key] = {
            "status": "runtime_failed",
            "exception_type": type(exc).__name__,
            "exception": str(exc).splitlines()[0],
            "cache_key": repr(key),
        }
        raise


def recurrence_chunk_size() -> int:
    return _recurrence_chunk_size


def masked_sum_chunk_size() -> int:
    """Levels per chunk of a masked sum; its reduction form runs all levels in one region."""
    return 1 << 30 if parallel_masked_sums() else _recurrence_chunk_size


def ordered_chunks_enabled(category: str) -> bool:
    return category in _enabled_categories


def run_recurrence_chunk(category: str, name: str, fn, *args):
    if _regional and not torch.compiler.is_compiling():
        # Float scalar inputs are value-specialized by Dynamo 2.8. Keep the
        # varying first level as a one-element device tensor, not a host value.
        parameters = fn.__code__.co_varnames[:fn.__code__.co_argcount]
        args = tuple(value.reshape(1) if parameter == "first" else value
                     for parameter, value in zip(parameters[:len(args)], args, strict=True))
        return run_compile_island(category, name, fn, *args)
    if torch.compiler.is_compiling() or category not in _enabled_categories:
        return fn(*args)
    return run_compile_island(category, name, fn, *args)
