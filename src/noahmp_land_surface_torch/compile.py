# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Scoped PyTorch-native compile controls for Noah-MP target islands."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import hashlib
import types
from typing import Any, TypeVar

import torch

TensorResult = TypeVar("TensorResult")

COMPILE_CATEGORIES = frozenset(
    {
        "land_geometry",
        "land_canwater",
        "land_thermoprop",
        "land_radiation",
        "land_bare_flux",
        "land_vege_flux_update",
        "land_snow_divide",
        "land_stability",
        "land_snowwater",
        "land_soilwater",
        "land_surface_flux",
        "land_flux_merge",
        "land_temperature_merge",
        "land_water_merge",
        "land_vege_flux",
        "land_vege_flux_iteration",
        "land_vege_flux_finalize",
        "land_vege_flux_loop_chunk",
        "land_vege_flux_loop_step",
        "land_tsnosoi",
        "land_phasechange",
        "land_tail",
        "glacier_tsnosoi",
        "glacier_repartition",
    }
)
MECHANISM_CERTIFIED_COMPILE_CATEGORIES = frozenset(
    {
        "land_bare_flux",
        "land_flux_merge",
        "land_tail",
        "land_temperature_merge",
        "land_water_merge",
        "land_vege_flux_finalize",
        "land_vege_flux_loop_chunk",
    }
)
RATCHET_BLOCKED_COMPILE_CATEGORIES = frozenset()
PROFILER_BLOCKED_COMPILE_CATEGORIES = frozenset({"land_vege_flux"})
STRUCTURED_LEGACY_COMPILE_CATEGORIES = frozenset()
FLAT_INPUT_COMPILE_CATEGORIES = COMPILE_CATEGORIES - STRUCTURED_LEGACY_COMPILE_CATEGORIES
EXPERIMENTAL_FULL_COMPILE_CATEGORIES = FLAT_INPUT_COMPILE_CATEGORIES - PROFILER_BLOCKED_COMPILE_CATEGORIES
BASE_DEFAULT_COMPILE_CATEGORIES = frozenset(
    {
        "land_stability",
        "land_vege_flux_finalize",
        "land_vege_flux_loop_chunk",
        "glacier_tsnosoi",
    }
)
PROFILED_LAND_FLAT_EXTRA_CATEGORIES = frozenset(
    {
        "land_geometry",
        "land_bare_flux",
        "land_canwater",
        "land_flux_merge",
        "land_tail",
        "land_temperature_merge",
        "land_water_merge",
        "land_phasechange",
        "land_radiation",
        "land_snowwater",
        "land_soilwater",
        "land_stability",
        "land_surface_flux",
        "land_thermoprop",
        "land_tsnosoi",
    }
)
PROFILED_LAND_FLAT_COMPILE_CATEGORIES = BASE_DEFAULT_COMPILE_CATEGORIES | PROFILED_LAND_FLAT_EXTRA_CATEGORIES
DEFAULT_COMPILE_CATEGORIES = PROFILED_LAND_FLAT_COMPILE_CATEGORIES
PRODUCTION_SAFE_COMPILE_CATEGORIES = DEFAULT_COMPILE_CATEGORIES

_enabled_categories: frozenset[str] = frozenset()
_gradient_categories: frozenset[str] = frozenset()
_vegetation_chunk_size = 20
_explicit_categories = False
_compile_cache: dict[tuple[Any, ...], Callable[..., Any]] = {}
_compile_status: dict[str, dict[str, Any]] = {}


def normalize_compile_categories(categories: Iterable[str] | str | None) -> frozenset[str]:
    if categories is None:
        return frozenset()
    if isinstance(categories, str):
        value = categories.strip().lower()
        if value in {"", "none", "off", "false"}:
            return frozenset()
        if value in {"default", "on", "true"}:
            return DEFAULT_COMPILE_CATEGORIES
        if value in {"profiled_land_flat", "land_flat_no_flux"}:
            return PROFILED_LAND_FLAT_COMPILE_CATEGORIES
        if value in {"experimental", "experimental_full", "full"}:
            return EXPERIMENTAL_FULL_COMPILE_CATEGORIES
        if value == "all":
            return PRODUCTION_SAFE_COMPILE_CATEGORIES
        raw = [item.strip() for item in categories.split(",")]
    else:
        raw = [str(item).strip() for item in categories]
    normalized = frozenset(item for item in raw if item)
    unknown = normalized - COMPILE_CATEGORIES
    if unknown:
        raise ValueError(f"unknown Noah-MP compile categories: {sorted(unknown)}")
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
def compile_categories(categories: Iterable[str] | str | None, *, gradients: bool = False, vegetation_chunk_size: int = 20):
    """Select islands; compiled derivatives are an explicit experimental opt-in.

    Existing callers retain eager derivative execution. A gradient-enabled
    policy requires independent numerical and derivative qualification; enabling
    this switch alone does not qualify it.
    """
    global _gradient_categories, _vegetation_chunk_size
    if type(vegetation_chunk_size) is not int or vegetation_chunk_size not in (4, 20):
        raise ValueError("vegetation_chunk_size must be 4 or 20")
    previous = set_compile_categories(categories)
    previous_gradients = _gradient_categories
    previous_chunk_size = _vegetation_chunk_size
    _gradient_categories = _enabled_categories if gradients else frozenset()
    _vegetation_chunk_size = vegetation_chunk_size
    try:
        yield
    finally:
        _gradient_categories = previous_gradients
        _vegetation_chunk_size = previous_chunk_size
        restore_compile_categories(previous)


def vegetation_chunk_size() -> int:
    """Selected compile granularity; the source iteration count stays unchanged."""
    return _vegetation_chunk_size


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
        str(value.layout),
        tuple(value.stride()) if value.layout == torch.strided else None,
    )


def _literal_spec(value: Any) -> tuple[Any, ...]:
    if isinstance(value, torch.Tensor):
        return _tensor_spec(value)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return ("literal", type(value).__name__, value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_literal_spec(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_literal_spec(item) for item in value))
    if isinstance(value, Mapping):
        return ("mapping", tuple((str(key), _literal_spec(item)) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))))
    if is_dataclass(value):
        return (
            "dataclass",
            type(value).__module__,
            type(value).__qualname__,
            tuple((field.name, _literal_spec(getattr(value, field.name))) for field in fields(value)),
        )
    raise TypeError(
        "compiled Noah-MP islands accept tensors, stable literals, dataclasses, and mappings of those values; "
        f"got {type(value).__name__}"
    )


def _cache_key(category: str, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    kw_specs = tuple((key, _literal_spec(value)) for key, value in sorted(kwargs.items()))
    return (category, name, tuple(_literal_spec(value) for value in args), kw_specs)


def _flatten_for_compile(value: Any) -> tuple[Any, tuple[Any, ...], int]:
    if isinstance(value, torch.Tensor):
        return _tensor_spec(value), (value,), 0
    if isinstance(value, (bool, int, float, str)) or value is None:
        return ("literal", type(value).__name__, value), (value,), 0
    if isinstance(value, tuple):
        flattened = tuple(_flatten_for_compile(item) for item in value)
        specs = tuple(spec for spec, _values, _count in flattened)
        flat = tuple(item for _spec, values, _count in flattened for item in values)
        return ("tuple", specs), flat, 1 + sum(count for _spec, _values, count in flattened)
    if isinstance(value, list):
        flattened = tuple(_flatten_for_compile(item) for item in value)
        specs = tuple(spec for spec, _values, _count in flattened)
        flat = tuple(item for _spec, values, _count in flattened for item in values)
        return ("list", specs), flat, 1 + sum(count for _spec, _values, count in flattened)
    if isinstance(value, Mapping):
        flattened = tuple((key, _flatten_for_compile(item)) for key, item in sorted(value.items(), key=lambda pair: str(pair[0])))
        specs = tuple((key, spec) for key, (spec, _values, _count) in flattened)
        flat = tuple(item for _key, (_spec, values, _count) in flattened for item in values)
        return ("mapping", specs), flat, 1 + sum(count for _key, (_spec, _values, count) in flattened)
    if is_dataclass(value):
        flattened = tuple((field.name, _flatten_for_compile(getattr(value, field.name))) for field in fields(value))
        specs = tuple((field_name, spec) for field_name, (spec, _values, _count) in flattened)
        flat = tuple(item for _field_name, (_spec, values, _count) in flattened for item in values)
        return ("dataclass", type(value), specs), flat, 1 + sum(count for _field_name, (_spec, _values, count) in flattened)
    raise TypeError(
        "compiled Noah-MP islands accept tensors, stable literals, dataclasses, and mappings of those values; "
        f"got {type(value).__name__}"
    )


def _rebuild_from_compile_flat(spec: Any, values: tuple[Any, ...], index: int) -> tuple[Any, int]:
    kind = spec[0]
    if kind in {"tensor", "literal"}:
        return values[index], index + 1
    if kind == "tuple":
        return _rebuild_sequence_from_compile_flat(spec[1], values, index)
    if kind == "list":
        rebuilt, index = _rebuild_sequence_from_compile_flat(spec[1], values, index)
        return list(rebuilt), index
    if kind == "mapping":
        rebuilt_values, index = _rebuild_sequence_from_compile_flat(tuple(item_spec for _key, item_spec in spec[1]), values, index)
        return dict(zip((key for key, _item_spec in spec[1]), rebuilt_values, strict=True)), index
    if kind == "dataclass":
        cls = spec[1]
        rebuilt_values, index = _rebuild_sequence_from_compile_flat(tuple(field_spec for _field_name, field_spec in spec[2]), values, index)
        return cls(**dict(zip((field_name for field_name, _field_spec in spec[2]), rebuilt_values, strict=True))), index
    raise TypeError(f"unknown Noah-MP compile flatten spec kind {kind!r}")


def _rebuild_sequence_from_compile_flat(specs: tuple[Any, ...], values: tuple[Any, ...], index: int) -> tuple[tuple[Any, ...], int]:
    if not specs:
        return (), index
    head, index = _rebuild_from_compile_flat(specs[0], values, index)
    tail, index = _rebuild_sequence_from_compile_flat(specs[1:], values, index)
    return (head, *tail), index


def _flatten_call_for_compile(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], tuple[tuple[str, Any], ...], tuple[Any, ...], int]:
    if not args and all(isinstance(value, torch.Tensor) or isinstance(value, (bool, int, float, str)) or value is None for value in kwargs.values()):
        sorted_kwargs = tuple(sorted(kwargs.items()))
        return (
            (),
            tuple((key, _tensor_spec(value) if isinstance(value, torch.Tensor) else ("literal", type(value).__name__, value)) for key, value in sorted_kwargs),
            tuple(value for _key, value in sorted_kwargs),
            0,
        )
    flattened_args = tuple(_flatten_for_compile(arg) for arg in args)
    flattened_kwargs = tuple((key, _flatten_for_compile(value)) for key, value in sorted(kwargs.items()))
    arg_specs = tuple(spec for spec, _values, _count in flattened_args)
    kw_specs = tuple((key, spec) for key, (spec, _values, _count) in flattened_kwargs)
    flat_args = tuple(item for _spec, values, _count in flattened_args for item in values)
    flat_kwargs = tuple(item for _key, (_spec, values, _count) in flattened_kwargs for item in values)
    structured_count = sum(count for _spec, _values, count in flattened_args) + sum(
        count for _key, (_spec, _values, count) in flattened_kwargs
    )
    return arg_specs, kw_specs, flat_args + flat_kwargs, structured_count


def _flat_compile_callable(
    fn: Callable[..., TensorResult],
    arg_specs: tuple[Any, ...],
    kw_specs: tuple[tuple[str, Any], ...],
) -> Callable[..., TensorResult]:
    def flat_callable(*flat_values: Any) -> TensorResult:
        rebuilt_args, index = _rebuild_sequence_from_compile_flat(arg_specs, flat_values, 0)
        rebuilt_values, index = _rebuild_sequence_from_compile_flat(tuple(spec for _key, spec in kw_specs), flat_values, index)
        rebuilt_kwargs = dict(zip((key for key, _spec in kw_specs), rebuilt_values, strict=True))
        return fn(*rebuilt_args, **rebuilt_kwargs)

    flat_callable.__name__ = f"{getattr(fn, '__name__', 'noahmp_island')}_flat"
    return flat_callable


def _compile_specialized_callable(fn: Callable[..., TensorResult], key: tuple[Any, ...]) -> Callable[..., TensorResult]:
    code = getattr(fn, "__code__", None)
    if code is None:
        return fn
    suffix = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:16]
    name = f"{getattr(fn, '__name__', 'noahmp_island')}_{suffix}"
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


def _cache_key_digest(key: tuple[Any, ...]) -> str:
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:16]


def _has_gradient_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.is_floating_point(value) and value.requires_grad)
    if isinstance(value, (tuple, list)):
        return any(_has_gradient_tensor(item) for item in value)
    if isinstance(value, Mapping):
        return any(_has_gradient_tensor(item) for item in value.values())
    if is_dataclass(value):
        return any(_has_gradient_tensor(getattr(value, field.name)) for field in fields(value))
    return False


def run_compile_island(
    category: str,
    name: str,
    fn: Callable[..., TensorResult],
    *args: Any,
    **kwargs: Any,
) -> TensorResult:
    if category not in COMPILE_CATEGORIES:
        raise ValueError(f"unknown Noah-MP compile category {category!r}")
    status_key = f"{category}:{name}"
    if category not in _enabled_categories:
        _compile_status[status_key] = {"status": "eager_disabled"}
        return fn(*args, **kwargs)
    if category in STRUCTURED_LEGACY_COMPILE_CATEGORIES:
        _compile_status[status_key] = {
            "status": "eager_structured_legacy",
            "reason": "this island still accepts dataclass/table inputs; compile requires a hand-purified flat tensor/literal kernel",
        }
        return fn(*args, **kwargs)
    if category not in _gradient_categories and torch.is_grad_enabled() and (any(_has_gradient_tensor(arg) for arg in args) or any(_has_gradient_tensor(value) for value in kwargs.values())):
        _compile_status[status_key] = {
            "status": "eager_grad_enabled",
            "reason": "compiled Noah-MP islands are forward-performance paths; CPU autodiff gates validate eager smooth-input paths",
        }
        return fn(*args, **kwargs)
    if not hasattr(torch, "compile"):
        _compile_status[status_key] = {"status": "compile_unavailable"}
        raise RuntimeError("torch.compile is unavailable")

    arg_specs, kw_specs, flat_values, structured_input_count = _flatten_call_for_compile(args, kwargs)
    if structured_input_count:
        _compile_status.update(
            {
                status_key: {
                    "status": "eager_unpurified_structured_inputs",
                    "reason": "compiled Noah-MP islands must be hand-purified flat tensor/literal kernels",
                    "original_structured_input_count": structured_input_count,
                    "legacy_wrapper_rebuild_count": structured_input_count,
                }
            }
        )
        return fn(*args, **kwargs)
    direct_kwargs_call = not args and structured_input_count == 0
    key = (category, name, id(fn), torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
           "direct_kwargs" if direct_kwargs_call else "flat_values", arg_specs, kw_specs)
    compiled = _compile_cache.get(key)
    if compiled is None:
        cache_hit = False
        key_repr = repr(key)
        key_digest = hashlib.sha256(key_repr.encode("utf-8")).hexdigest()[:16]
        _compile_status[status_key] = {
            "status": "compiling",
            "cache_key": key_repr,
            "cache_key_digest": key_digest,
            "cache_hit": False,
            "input_form": "flat_tensor_literal_args",
            "flat_arg_count": len(flat_values),
            "compiled_structured_input_count": 0,
            "original_structured_input_count": structured_input_count,
            "legacy_wrapper_rebuild_count": structured_input_count,
        }
        target = fn if direct_kwargs_call else _flat_compile_callable(fn, arg_specs, kw_specs)
        # Inductor's default lowerings (x / c as x * (1 / c), integer powers as product
        # chains, vectorized CPU code) moved H__fctr by 64 ulp on case01; under the shared
        # exact-numerics policy every island output equals eager bit for bit.
        from ufs_physics_common import compile_policy

        compile_policy.install_exact_numerics()
        compiled = torch.compile(
            _compile_specialized_callable(target, key),
            backend=compile_policy._inductor_backend,
            fullgraph=True,
            dynamic=False,
        )
        _compile_cache[key] = compiled
    else:
        cache_hit = True
        previous_status = _compile_status.get(status_key, {})
        key_digest = str(previous_status.get("cache_key_digest") or _cache_key_digest(key))
        key_repr = previous_status.get("cache_key") or f"<sha256:{key_digest}>"
    _compile_status[status_key] = {
        "status": "compiled",
        "cache_key": key_repr,
        "cache_key_digest": key_digest,
        "cache_hit": cache_hit,
        "input_form": "flat_tensor_literal_args",
        "flat_arg_count": len(flat_values),
        "compiled_structured_input_count": 0,
        "original_structured_input_count": structured_input_count,
        "legacy_wrapper_rebuild_count": structured_input_count,
    }
    try:
        if direct_kwargs_call:
            return compiled(**kwargs)
        return compiled(*flat_values)
    except Exception as exc:  # noqa: BLE001 - compile failures are benchmark evidence.
        _compile_status[status_key] = {
            "status": "runtime_failed",
            "exception_type": type(exc).__name__,
            "exception": str(exc).splitlines()[0],
            "cache_key": repr(key),
        }
        raise
