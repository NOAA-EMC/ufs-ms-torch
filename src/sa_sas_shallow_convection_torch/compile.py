# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Scoped ``torch.compile`` controls for SA-SAS shallow recurrence islands."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
import hashlib
import types
from typing import Any, TypeVar

import torch

TensorResult = TypeVar("TensorResult")

COMPILE_CATEGORIES = frozenset(
    {
        "f_recurrence",
        "h_recurrence",
        "i_recurrence",
        "o_recurrence",
        "recurrence_chunks",
        "trigger",
    }
)
DEFAULT_COMPILE_CATEGORIES = COMPILE_CATEGORIES - {"recurrence_chunks", "trigger"}

_enabled_categories: frozenset[str] = frozenset()
_explicit_categories = False
_recurrence_chunk_size: int | None = None
_compile_cache: dict[tuple[Any, ...], Callable[..., Any]] = {}
_compile_status: dict[str, dict[str, Any]] = {}
_regional = False


@contextmanager
def regional_compile():
    global _regional
    previous, _regional = _regional, True
    try:
        with compile_categories({"recurrence_chunks", "trigger"}, recurrence_chunk_size=8):
            yield
    finally:
        _regional = previous


def normalize_compile_categories(categories: Iterable[str] | str | None) -> frozenset[str]:
    if categories is None:
        return frozenset()
    if isinstance(categories, str):
        value = categories.strip().lower()
        if value in {"default", "all", "on", "true"}:
            return DEFAULT_COMPILE_CATEGORIES
        if value in {"", "none", "off", "false"}:
            return frozenset()
        raw = [item.strip() for item in categories.split(",")]
    else:
        raw = [str(item).strip() for item in categories]
    normalized = frozenset(item for item in raw if item)
    unknown = normalized - COMPILE_CATEGORIES
    if unknown:
        raise ValueError(f"unknown SA-SAS shallow compile categories: {sorted(unknown)}")
    return normalized


def compile_categories_enabled() -> frozenset[str]:
    return _enabled_categories


def compile_categories_explicitly_set() -> bool:
    return _explicit_categories


def selected_recurrence_chunk_size() -> int | None:
    """Static wrapper granularity; pure kernels receive immutable shaped slices."""
    return _recurrence_chunk_size


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
def compile_categories(categories: Iterable[str] | str | None, *, recurrence_chunk_size: int | None = None):
    global _recurrence_chunk_size
    if recurrence_chunk_size is not None and (type(recurrence_chunk_size) is not int or recurrence_chunk_size not in (4, 8)):
        raise ValueError("shallow recurrence_chunk_size must be 4 or 8")
    previous = set_compile_categories(categories)
    previous_chunk_size = _recurrence_chunk_size
    if recurrence_chunk_size is not None:
        _recurrence_chunk_size = recurrence_chunk_size
    try:
        yield
    finally:
        _recurrence_chunk_size = previous_chunk_size
        restore_compile_categories(previous)


def clear_compile_cache() -> None:
    _compile_cache.clear()
    _compile_status.clear()


def compile_island_status() -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in _compile_status.items()}


def _tensor_spec(value: torch.Tensor) -> tuple[Any, ...]:
    return (
        "tensor",
        tuple(value.shape),
        tuple(value.stride()),
        str(value.dtype),
        str(value.device),
        bool(value.requires_grad),
        int(value.layout == torch.strided),
    )


def _literal_spec(value: Any) -> tuple[Any, ...]:
    if isinstance(value, torch.Tensor):
        return _tensor_spec(value)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return ("literal", type(value).__name__, value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_literal_spec(item) for item in value))
    raise TypeError(
        "compiled SA-SAS shallow islands accept only tensors and stable Python literals; "
        f"got {type(value).__name__}"
    )


def _has_gradient_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.is_floating_point(value) and value.requires_grad)
    if isinstance(value, tuple):
        return any(_has_gradient_tensor(item) for item in value)
    return False


def _cache_key(category: str, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    kw_specs = tuple((key, _literal_spec(value)) for key, value in sorted(kwargs.items()))
    return (category, name, tuple(_literal_spec(value) for value in args), kw_specs)


def _compile_specialized_callable(fn: Callable[..., TensorResult], key: tuple[Any, ...]) -> Callable[..., TensorResult]:
    code = getattr(fn, "__code__", None)
    if code is None:
        return fn
    suffix = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:16]
    name = f"{getattr(fn, '__name__', 'sa_sas_shallow_island')}_{suffix}"
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
    # An outer compiled region composes the same pure chunk equations directly.
    if torch.compiler.is_compiling():
        return fn(*args, **kwargs)
    if category not in COMPILE_CATEGORIES:
        raise ValueError(f"unknown SA-SAS shallow compile category {category!r}")
    if _regional:
        if category not in {"recurrence_chunks", "trigger"}:
            return fn(*args, **kwargs)
        from ufs_physics_common.compile_policy import compiled

        parameters = fn.__code__.co_varnames[:fn.__code__.co_argcount]
        args = tuple(value.reshape(1) if parameter == "first" else value
                     for parameter, value in zip(parameters[:len(args)], args, strict=True))
        if name in {"assemble_levels", "_compact_select", "_compact_merge"}:
            return compiled(fn, normalize_layouts=False)(*args, **kwargs)
        return compiled(fn)(*args, **kwargs)
    status_key = f"{category}:{name}"
    if category not in _enabled_categories:
        _compile_status[status_key] = {"status": "eager_disabled"}
        return fn(*args, **kwargs)
    if category != "recurrence_chunks" and torch.is_grad_enabled() and (any(_has_gradient_tensor(arg) for arg in args) or any(_has_gradient_tensor(value) for value in kwargs.values())):
        _compile_status[status_key] = {
            "status": "eager_grad_enabled",
            "reason": "compiled shallow recurrence islands are forward-only until backward compile closure is profiled",
        }
        return fn(*args, **kwargs)
    if not hasattr(torch, "compile"):
        _compile_status[status_key] = {"status": "compile_unavailable"}
        raise RuntimeError("torch.compile is unavailable")

    key = (fn, torch.is_grad_enabled(), _cache_key(category, name, args, kwargs))
    compiled = _compile_cache.get(key)
    if compiled is None:
        _compile_status[status_key] = {"status": "compiling", "cache_key": repr(key)}
        compiled = torch.compile(_compile_specialized_callable(fn, key), fullgraph=True, dynamic=False, mode=None)
        _compile_cache[key] = compiled
    _compile_status[status_key] = {"status": "compiled", "cache_key": repr(key)}
    try:
        return compiled(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - compile failures are benchmark evidence.
        _compile_status[status_key] = {
            "status": "runtime_failed",
            "exception_type": type(exc).__name__,
            "exception": str(exc).splitlines()[0],
            "cache_key": repr(key),
        }
        raise
