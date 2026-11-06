# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Working precision of the coupled model: the dtype of the state, not a module constant.

The coupled model runs in the floating dtype of its case. Float64 is the default and the parity
dtype (cases are prepared in float64 and stay bitwise to the Fortran-derived references); float32
is separately qualified. Runtime code takes its working dtype from a state tensor it already
holds (``x.dtype``). CPU FP32 uses FP64 evaluation at selected powers, then rounds once;
PyTorch 2.8 scalar/SIMD power differences otherwise change results with spatial ownership.
CUDA retains native FP32 powers. ``ufs_physics_common.precision.working_power`` defines this
on-device, differentiable precision policy. It does not change FP64 arithmetic.

``case_as_dtype`` makes a case of another precision from a prepared case in memory: every floating
tensor of the whole case, including configuration tensors, is rounded to ``dtype``
(integer and boolean tensors, Python scalars and strings are kept). The native MPAS boundary marker
is rebound to the converted tensors with its configuration digest and timestep, so the first step
takes the same restart path as the float64 case.

Float64 by design in every working dtype: the clock and calendar (``clock.py``, the cycle's ``sec``,
solar orbit and Julian dates in ``rrtmgp_radiation_torch/setup.py``, ozone and MERRA-2 time
interpolation). McICA uses the same integer Mersenne-Twister words in both precisions;
FP64 uniforms divide by ``2**32 - 1``, while FP32 uses exact integer threshold predicates.
Host tables (fpvs, CO2) are built in float64 and held in the working dtype.

Where float32 loses meaning (not tuned; revisit before float32 science runs):
- ``eps_scaled`` guards keep the float64 count of rounding steps: ``reverse.py`` mass check (1e-10 is
  4.5e5 ulp, 5.4e-2 relative in float32: the check no longer catches a vapour-basis error) and the
  SAMF water/energy reconciliation (``samf_interstitials.py`` 5e-13, 2.7e-4 in float32).
- ``roundoff_floor``: the absolute tracer floors (``projection.py`` -1e-10, -1e-8 for ni/nr) are below
  float32 resolution; float32 uses 64 ulp of the field maximum (ni reached -0.03 kg-1 at the first commit after the physics group).
- Divide guards of 1e-300 round to 0 in float32 (0/0 in masked lanes): TEMPO ``torch_native.py``
  (42 sites, e.g. 4667, 6103, 14543), Noah-MP ``land_energy.py`` 1974; ``frac > 1e-300`` in RRTMGP
  ``shortwave.py`` 568 and ``longwave_main.py`` 488 becomes ``frac > 0``.
- Additive guards below float32 epsilon are no-ops: ``surface_mirror.py`` 161 (``smc + 1e-12``),
  surface layer ``driver.py`` 333/337 (``wind2 >= 1e-12`` fallback).
- RRTMGP thresholds from float64 epsilon (``radiation_group.py`` 125-135: WP_EPS, LW_TAU_THRESH,
  SW_MIN_K, SW_MIN_MU0); a float32 Fortran build would take epsilon(1._sp).
- The CCPP fill value ``huge`` (9.97e36) times a squared wind overflows float32 in unused surface
  lanes (surface layer ``driver.py`` 332-334, masked afterwards).
- Ordered column sums (``ledger._ordered_totals``: limiter ledgers, reference geometry) accumulate
  ~n*eps relative error over 10,242 columns in float32.
- TEMPO collection and freezing tables (``real(table_dp)`` in module_mp_tempo_params.F90) and its
  ``real(dp)`` diagnostics are float32 here.
- The dycore's compiled normal-mode AD is qualified for float64 only (``compile_utils.py`` 3863);
  float32 runs the no-grad and eager differentiable paths.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

WORKING_DTYPES = (torch.float32, torch.float64)
_HOST_SCALAR_TYPES = frozenset((int, float, bool, str, bytes, complex, type(None)))


def working_like(value: Any, reference: torch.Tensor) -> bool:
    """True when ``value`` is a tensor of the working dtype of ``reference`` (float32 or float64).

    A zero-dimensional ``value`` may have either working dtype: scalars such as the clock's float64
    timestep never promote the arrays they meet (PyTorch type promotion), so they cannot upcast.
    """
    return (isinstance(value, torch.Tensor) and reference.dtype in WORKING_DTYPES and value.dtype in WORKING_DTYPES
            and (value.ndim == 0 or value.dtype == reference.dtype))


def eps_scaled(tolerance64: float, dtype: torch.dtype) -> float:
    """A float64 runtime tolerance scaled by the machine epsilon of ``dtype`` (``tolerance64`` itself in float64).

    For checks whose float64 bound counts rounding steps; float32 meets the same number of steps.
    """
    return tolerance64 * (torch.finfo(dtype).eps / 2.0 ** -52)


def roundoff_floor(floor64: float, value: torch.Tensor) -> float | torch.Tensor:
    """The negative floor of a fail-closed check on ``value``: ``floor64`` in float64.

    In float32 the floor is at least 64 ulp of the largest ``|value|`` below zero, the roundoff of a
    field of that size (a float64 absolute floor such as -1e-10 is below float32 resolution).
    """
    if value.dtype == torch.float64:
        return floor64
    return torch.clamp_max(value.abs().amax() * (-64.0 * torch.finfo(value.dtype).eps), floor64)


def resolve_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """Resolve the public precision names; reject reduced precision and spelling errors."""
    from mpas_pytorch.precision import resolve_dtype as resolve

    return resolve({"f32": "float32", "f64": "float64"}.get(dtype, dtype))


def move_tree(tree: Any, *, dtype: torch.dtype | str | None = None,
              device: torch.device | str | None = None, memo: dict | None = None) -> Any:
    """Convert setup tensors once, preserving object aliases and the autograd graph.

    Integer/bool tensors keep their dtype. Named tuples and device metadata are
    included. No conversion occurs inside a timestep or a compiled region.
    """
    dtype = resolve_dtype(dtype) if dtype is not None else None
    # The device tensors report (e.g. cuda:0 for "cuda"): device metadata is compared with tensor devices.
    device = torch.empty(0, device=device).device if device is not None else None
    memo = {} if memo is None else memo

    def build(value: Any) -> Any:
        # Host column metadata contains millions of immutable scalar leaves.
        if type(value) in _HOST_SCALAR_TYPES:
            return value
        key = id(value)
        if key in memo:
            return memo[key]
        if isinstance(value, torch.Tensor):
            if value.is_complex():
                raise ValueError("complex tensors are not supported in a coupled case")
            out = value.to(device=device, dtype=dtype if value.is_floating_point() else None)
        elif isinstance(value, torch.device):
            out = value if device is None else device
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            changes = {f.name: build(getattr(value, f.name)) for f in dataclasses.fields(value) if f.init}
            out = value if all(v is getattr(value, k) for k, v in changes.items()) else dataclasses.replace(value, **changes)
        elif isinstance(value, Mapping):
            items = {k: build(v) for k, v in value.items()}
            out = value if all(items[k] is value[k] for k in value) else type(value)(items)
        elif isinstance(value, (tuple, list)):
            items = [build(v) for v in value]
            out = (value if all(a is b for a, b in zip(items, value)) else
                   type(value)(*items) if hasattr(value, "_fields") else type(value)(items))
        else:
            out = value
        memo[key] = out
        return out

    return build(tree)


def cast_floating(tree: Any, dtype: torch.dtype) -> Any:
    """``tree`` with each floating tensor in ``dtype``; dicts, lists, tuples and dataclasses are rebuilt.

    Tensors shared by several leaves stay shared. Other leaves are returned unchanged.
    """
    return move_tree(tree, dtype=dtype)


def case_as_dtype(case: Mapping[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    """A prepared coupled case (``native``, ``mesh``, ``config``, ``arguments``, ``cold_arguments``) in ``dtype``.

    Returns a shallow copy of ``case`` when it already has that dtype. The arguments' ``mesh`` is the
    converted case mesh and their ``config`` the case config, as in the prepared case.
    """
    return prepare_case(case, dtype=dtype)


def prepare_case(case: Mapping[str, Any], *, dtype: torch.dtype | str | None = None,
                 device: torch.device | str | None = None) -> dict[str, Any]:
    """Convert one complete case and rebind its restart boundary to the new tensors.

    Inspect every leaf, even when the primary state already has the requested
    dtype. A single memo keeps aliases shared across native, mesh and arguments.
    The input case is never modified.
    """
    from mpas_pytorch.state import (
        _BOUNDARY_MARKER, _live_boundary_snapshot, has_complete_step_boundary,
        continuation_configuration_digest,
    )

    native = dict(case["native"])
    marker = native.pop(_BOUNDARY_MARKER, None)
    out = dict(move_tree({**case, "native": native}, dtype=dtype, device=device))
    native, mesh = out["native"], out["mesh"]
    arguments = dict(out["arguments"])
    if "mesh" in arguments:
        arguments["mesh"] = mesh
    if "config" in arguments:
        arguments["config"] = out["config"]
    out["arguments"] = arguments
    if marker is not None:
        unchanged = (mesh is case["mesh"] and out["config"] is case["config"]
                     and all(v is case["native"][k] for k, v in native.items()))
        if unchanged:
            out["native"] = case["native"]
            return out
        if not has_complete_step_boundary(
                case["native"], mesh=case["mesh"] if marker.get("mesh") is not None else None,
                config=case["config"] if marker.get("config") is not None else None,
                dt=marker.get("dt")):
            raise ValueError("cannot convert an edited complete-step boundary")
        # Never mutate an unchanged input native mapping while rebinding metadata.
        native = out["native"] = dict(native)
        native[_BOUNDARY_MARKER] = {
            **marker,
            "state": _live_boundary_snapshot(native),
            "mesh": _live_boundary_snapshot(mesh, all_keys=True, device=native["u"].device)
            if marker.get("mesh") is not None else None,
            "config": continuation_configuration_digest(out["config"])
            if marker.get("config") is not None else None,
        }
    return out
