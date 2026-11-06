# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Opt-in scoped ``torch.compile`` controls for pure tensor recurrence islands."""

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
        "d_pbl_scan",
        "e_surface_similarity",
        "f_cloud_seed",
        "g_plume_recurrence",
        "i_downdraft",
        "k_mixing_length",
        "l_diffusivity",
        "m_tendency_block",
        "m_tke_recurrence",
        "m_plume_recurrence",
        "n_tvd_limiter",
        "tridiagonal",
        "interstitial",
    }
)
DEFAULT_COMPILE_CATEGORIES = COMPILE_CATEGORIES

_enabled_categories: frozenset[str] = frozenset()
_explicit_categories = False
_compile_cache: dict[tuple[Any, ...], Callable[..., Any]] = {}
_compile_status: dict[str, dict[str, Any]] = {}
_regional = False


@contextmanager
def regional_compile():
    """Keep the coupled PBL driver eager and compile its tensor blocks exactly."""
    global _regional
    previous, _regional = _regional, True
    try:
        yield
    finally:
        _regional = previous


def normalize_compile_categories(categories: Iterable[str] | str | None) -> frozenset[str]:
    if categories is None:
        return frozenset()
    if isinstance(categories, str):
        if categories.strip().lower() in {"default", "all", "on", "true"}:
            return DEFAULT_COMPILE_CATEGORIES
        if categories.strip().lower() in {"", "none", "off", "false"}:
            return frozenset()
        raw = [item.strip() for item in categories.split(",")]
    else:
        raw = [str(item).strip() for item in categories]
    normalized = frozenset(item for item in raw if item)
    unknown = normalized - COMPILE_CATEGORIES
    if unknown:
        raise ValueError(f"unknown SA-TKE compile categories: {sorted(unknown)}")
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


def compile_island_status() -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in _compile_status.items()}


def _tensor_spec(value: torch.Tensor) -> tuple[Any, ...]:
    return (
        "tensor",
        tuple(value.shape),
        str(value.dtype),
        str(value.device),
        bool(value.requires_grad),
        int(value.layout == torch.strided),
        tuple(value.stride()) if value.layout == torch.strided else None,
    )


def _literal_spec(value: Any) -> tuple[Any, ...]:
    if isinstance(value, torch.Tensor):
        return _tensor_spec(value)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return ("literal", type(value).__name__, value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_literal_spec(item) for item in value))
    raise TypeError(
        "compiled SA-TKE islands accept only tensors and stable Python literals; "
        f"got {type(value).__name__}"
    )


def _cache_key(category: str, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    kw_specs = tuple((key, _literal_spec(value)) for key, value in sorted(kwargs.items()))
    return (category, name, torch.is_grad_enabled(), tuple(_literal_spec(value) for value in args), kw_specs)


def _compile_specialized_callable(fn: Callable[..., TensorResult], key: tuple[Any, ...]) -> Callable[..., TensorResult]:
    code = getattr(fn, "__code__", None)
    if code is None:
        return fn
    suffix = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:16]
    name = f"{getattr(fn, '__name__', 'sa_tke_island')}_{suffix}"
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
        raise ValueError(f"unknown SA-TKE compile category {category!r}")
    # Nested islands compose their pure functions when an outer region traces.
    if torch.compiler.is_compiling():
        return fn(*args, **kwargs)
    if _regional:
        # These are orchestration over separately compiled blocks, not tensor regions.
        if category == "g_plume_recurrence" and name in {"plume_block", "tracer_recurrence"}:
            return fn(*args, **kwargs)
        if category == "tridiagonal" and name in {"heat_tracer_recovery", "heat_tracer_thomas"}:
            return fn(*args, **kwargs)
        if category == "k_mixing_length" and name in {"seed_block", "parcel_chunk"}:
            return fn(*args, **kwargs)
        from ufs_physics_common.compile_policy import compiled

        if name == "assemble_levels":
            return compiled(fn, normalize_layouts=False)(*args, **kwargs)
        return compiled(fn)(*args, **kwargs)
    status_key = f"{category}:{name}"
    if category not in _enabled_categories:
        _compile_status[status_key] = {"status": "eager_disabled"}
        return fn(*args, **kwargs)
    if not hasattr(torch, "compile"):
        _compile_status[status_key] = {"status": "compile_unavailable"}
        raise RuntimeError("torch.compile is unavailable")

    # Keep the caller's gradient set. The signature already distinguishes it.
    autodiff_stabilized = False
    key = _cache_key(category, name, args, kwargs)
    compiled = _compile_cache.get(key)
    if compiled is None:
        _compile_status[status_key] = {
            "status": "compiling",
            "cache_key": repr(key),
            "autodiff_stabilized": autodiff_stabilized,
        }
        compiled = torch.compile(_compile_specialized_callable(fn, key), fullgraph=True, dynamic=False, mode=None)
        _compile_cache[key] = compiled
    _compile_status[status_key] = {
        "status": "compiled",
        "cache_key": repr(key),
        "autodiff_stabilized": autodiff_stabilized,
    }
    try:
        return compiled(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - compile failures are evidence.
        _compile_status[status_key] = {
            "status": "runtime_failed",
            "exception_type": type(exc).__name__,
            "exception": str(exc).splitlines()[0],
            "cache_key": repr(key),
        }
        raise
