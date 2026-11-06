# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Scoped ``torch.compile`` controls for RRTMGP pure tensor islands.

The scientific target remains PyTorch-native.  These helpers deliberately
compile only tensor-only islands; the cumulative checkpoint/orchestration shell
continues to run eagerly because it contains mappings, source-hook selection,
and validation-oriented control flow.
"""

from __future__ import annotations

import hashlib
import types
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any, TypeVar

import torch

TensorResult = TypeVar("TensorResult")

COMPILE_CATEGORIES = frozenset(
    {
        "aerosol",
        "gas_optics",
        "longwave_transport",
        "post",
        "shortwave_transport",
    }
)
DEFAULT_COMPILE_CATEGORIES = frozenset()

_enabled_categories: frozenset[str] = DEFAULT_COMPILE_CATEGORIES
_explicit_categories = False
_compile_cache: dict[tuple[Any, ...], Callable[..., Any]] = {}
_compile_status: dict[str, dict[str, Any]] = {}
_tuple_literal_spec_cache: dict[int, tuple[tuple[Any, ...], tuple[Any, ...]]] = {}


def normalize_compile_categories(categories: Iterable[str] | str | None) -> frozenset[str]:
    if categories is None:
        return DEFAULT_COMPILE_CATEGORIES
    if isinstance(categories, str):
        value = categories.strip().lower()
        if value == "default":
            return DEFAULT_COMPILE_CATEGORIES
        if value in {"all", "on", "true", "transport"}:
            return COMPILE_CATEGORIES
        if value in {"", "none", "off", "false"}:
            return frozenset()
        raw = [item.strip() for item in value.split(",")]
    else:
        raw = [str(item).strip() for item in categories]
    normalized = frozenset(item for item in raw if item)
    unknown = normalized - COMPILE_CATEGORIES
    if unknown:
        raise ValueError(f"unknown RRTMGP compile categories: {sorted(unknown)}")
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
def compile_categories(categories: Iterable[str] | str | None):
    previous = set_compile_categories(categories)
    try:
        yield
    finally:
        restore_compile_categories(previous)


def clear_compile_cache() -> None:
    _compile_cache.clear()
    _compile_status.clear()
    _tuple_literal_spec_cache.clear()


def compile_island_status() -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in _compile_status.items()}


def compile_cache_snapshot() -> dict[str, Any]:
    """Return a stable, non-owning description of the process-local cache.

    The callable objects themselves are deliberately not exposed.  The key
    digests make an unexpected specialization visible without serializing the
    (potentially very large) tensor-signature representations into benchmark
    artifacts.
    """

    key_sha256 = sorted(
        hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
        for key in _compile_cache
    )
    return {
        "entry_count": len(_compile_cache),
        "key_sha256": key_sha256,
        "island_status": compile_island_status(),
    }


def _tensor_spec(value: torch.Tensor) -> tuple[Any, ...]:
    return (
        "tensor",
        tuple(value.shape),
        tuple(value.stride()),
        value.dtype,
        value.device.type,
        value.device.index,
        bool(value.requires_grad),
        value.layout,
    )


def _spec_holds_tensor(spec: tuple[Any, ...]) -> bool:
    return spec[0] == "tensor" or (spec[0] == "tuple" and any(_spec_holds_tensor(item) for item in spec[1]))


def _literal_spec(value: Any) -> tuple[Any, ...]:
    if isinstance(value, torch.Tensor):
        return _tensor_spec(value)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return ("literal", type(value).__name__, value)
    if isinstance(value, tuple):
        cached = _tuple_literal_spec_cache.get(id(value))
        if cached is not None and cached[0] is value:
            return cached[1]
        spec = ("tuple", tuple(_literal_spec(item) for item in value))
        # Keep the immutable tuple alive with its specification so ``id`` reuse
        # cannot alias a later compile signature. A tuple that holds tensors is not
        # kept: the cache would hold its tensors (and, under grad, their whole
        # graph) for the life of the process.
        if not _spec_holds_tensor(spec):
            _tuple_literal_spec_cache[id(value)] = (value, spec)
        return spec
    raise TypeError(
        "compiled RRTMGP islands accept only tensors and stable Python literals; "
        f"got {type(value).__name__}"
    )


def _cache_key(category: str, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    kw_specs = tuple((key, _literal_spec(value)) for key, value in sorted(kwargs.items()))
    return (category, name, tuple(_literal_spec(value) for value in args), kw_specs)


def _compile_specialized_callable(fn: Callable[..., TensorResult], key: tuple[Any, ...]) -> Callable[..., TensorResult]:
    code = getattr(fn, "__code__", None)
    if code is None:
        return fn
    suffix = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:16]
    name = f"{getattr(fn, '__name__', 'rrtmgp_island')}_{suffix}"
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
        raise ValueError(f"unknown RRTMGP compile category {category!r}")
    status_key = f"{category}:{name}"
    if category not in _enabled_categories:
        _compile_status.setdefault(status_key, {"status": "eager_disabled"})
        return fn(*args, **kwargs)
    if not hasattr(torch, "compile"):
        _compile_status[status_key] = {"status": "compile_unavailable"}
        raise RuntimeError("torch.compile is unavailable")

    key = _cache_key(category, name, args, kwargs)
    compiled = _compile_cache.get(key)
    if compiled is None:
        cache_key_repr = repr(key)
        _compile_status[status_key] = {"status": "compiling", "cache_key": cache_key_repr}
        compiled = torch.compile(_compile_specialized_callable(fn, key), fullgraph=True, dynamic=False, mode=None)
        _compile_cache[key] = compiled
        _compile_status[status_key] = {"status": "compiled", "cache_key": cache_key_repr}
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
