# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Main entry point for MPAS-Atmosphere PyTorch dynamics core."""

import contextlib
import logging
import math
import os
import resource
import warnings
import weakref
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

from mpas_pytorch.autodiff import differentiable_mode
from mpas_pytorch.autodiff.checkpoint import checkpoint_state_dict
from mpas_pytorch.config_defaults import (
    FORTRAN_DEFAULTS,
    get_config_value_safe,
    validate_config,
)
from mpas_pytorch.constants import constants_for_config
from mpas_pytorch.cuda_graph_timestep import (
    BoundarySignature,
    TimestepGraph,
    boundary_signature,
    signature_changes,
)
from mpas_pytorch.diagnostics import refresh_output_diagnostics
from mpas_pytorch.dynamics import time_integration as _time_integration
from mpas_pytorch.dynamics.compile_utils import (
    compile_backend,
    compile_enabled,
    compile_mode,
    cuda_graph_forward_scope,
    cuda_graph_step_boundary,
    forward_compiled_region,
    own_cuda_graph_outputs,
)
from mpas_pytorch.dynamics.init_coupled_diagnostics import (
    _init_coupled_connectivity_payload,
    _init_coupled_connectivity_payload_cache_key,
    atm_init_coupled_diagnostics,
    coupled_exner_pressure,
)
from mpas_pytorch.dynamics.recover_large_step import (
    _large_step_recovery_connectivity_payload,
    decode_native_fields,
)
from mpas_pytorch.dynamics.solve_diagnostics import (
    _rebind_diagnostic_solve_receipt,
    diagnostic_solve_checkpoint_metadata,
)
from mpas_pytorch.dynamics.static_payloads import (
    DycoreStaticPayloads,
    ensure_static_payload_store,
)
from mpas_pytorch.dynamics.time_integration import (
    _checkpoint_context_for_policy,
    _checkpoint_non_tensor_signature_for_policy,
    atm_srk3,
)
from mpas_pytorch.dynamics.vector_reconstruction import (
    CELL_PADDING_OFFSET_KEY,
    COEFFS_RECONSTRUCT_ALL_ZERO_KEY,
    _vector_reconstruction_numeric,
    _vector_reconstruction_payload,
    _vector_reconstruction_payload_cache_key,
    coeffs_reconstruct_all_zero,
    mpas_reconstruct_2d,
)
from mpas_pytorch.exceptions import MpasConfigError, MpasDeviceError
from mpas_pytorch.initialization import initialize_from_state
from mpas_pytorch.precision import (
    assert_async,
    floating_dtype,
    require_uniform_precision,
    require_workspace_precision,
)
from mpas_pytorch.profiling import profiled_call
from mpas_pytorch.release_capabilities import validate_runtime_request
from mpas_pytorch.state import (
    NATIVE_STATE_KEYS,
    STEP_FORCING_KEYS,
    _require_tensor_predicate,
    clone_tree,
    consume_ufs_physics_commit,
    dict_to_device,
    mark_complete_step_boundary,
    require_boundary_continuation_identity,
    timestep_boundary_predicate,
)
from mpas_pytorch.utils.checks import require_consistent_device

LOGGER = logging.getLogger(__name__)
_CPU_DEVICE = torch.device("cpu")
# Preserve the historical private alias for eager checkpoint tests/tools.
_SAVE_NOTHING_CONTEXT = _time_integration._SAVE_NOTHING_CONTEXT


def _validate_public_inputs(
    state: Mapping[str, Any],
    mesh: Mapping[str, Any],
    device: torch.device | str,
    *,
    experimental: bool = False,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Reject unsupported execution before preparing or converting model fields."""
    require_consistent_device("forecast inputs", state, mesh)
    compute_dtype = require_uniform_precision(state, mesh)
    target = _check_device_available(device, dtype=compute_dtype)
    require_workspace_precision(
        (config or {}).get("_dycore_workspace"), compute_dtype, target
    )
    validate_runtime_request(
        device=target.type,
        dtype=str(compute_dtype).removeprefix("torch."),
        physics_mode="dycore",
        experimental=experimental,
    )
    for mapping in (state, mesh):
        for value in mapping.values():
            if isinstance(value, torch.Tensor):
                dtype = (
                    str(value.dtype).removeprefix("torch.")
                    if value.is_floating_point() or value.is_complex()
                    else str(compute_dtype).removeprefix("torch.")
                )
                validate_runtime_request(
                    device=value.device.type,
                    dtype=dtype,
                    physics_mode="dycore",
                    experimental=experimental,
                )


def _finite_flags_numeric(*fields: torch.Tensor) -> torch.Tensor:
    """True when every element of every field is finite (one region)."""
    return torch.stack([torch.isfinite(field).all() for field in fields]).all()


def _compiled_finite_check(
    config: Mapping[str, Any] | None, values: list[torch.Tensor]
) -> Callable[..., torch.Tensor] | None:
    """The fused finite check, compiled with the dycore (no autograd, one
    device), else None."""
    if (
        not isinstance(config, Mapping)
        or len(values) < 2
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or not compile_enabled(dict(config))
        or len({value.device for value in values}) != 1
    ):
        return None
    from mpas_pytorch.dynamics.compile_utils import maybe_compile_kernel

    return maybe_compile_kernel(
        "model_finite_check_region",
        _finite_flags_numeric,
        enabled=True,
        mode=compile_mode(dict(config)),
        backend=compile_backend(dict(config)),
        specialization_key=tuple(
            (tuple(value.shape), str(value.dtype)) for value in values
        ),
    )


def _require_finite_model_fields(
    state: dict[str, Any],
    context: str,
    *,
    fields: Iterable[str] | None = None,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Reject nonfinite model fields, on device (no host read).

    Fields whose version stamp shows them unchanged since a passing check are
    skipped. With ``config`` compiling the dycore, the remaining fields are
    checked by one compiled reduction and one assertion; if it fails, the
    per-field checks below name the field (as without compilation).
    """
    fields = (
        fields
        if fields is not None
        else (
            "u",
            "w",
            "theta",
            "rho",
            "theta_m",
            "rho_zz",
            "scalars",
            "qv",
            "rho_base",
            "theta_base",
            "zz",
            "pressure_p",
            "pressure_base",
            "pressure",
            "surface_pressure",
        )
    )
    selected = {name: state[name] for name in fields if name in state}
    require_consistent_device(context, selected)
    stamps = not torch.compiler.is_compiling()
    pending = {
        name: value
        for name, value in selected.items()
        if isinstance(value, torch.Tensor)
        and not (stamps and _finite_stamp_matches(value))
    }
    fused = _compiled_finite_check(config, list(pending.values()))
    if fused is not None:
        message = f"{context}: {', '.join(pending)} contain nonfinite values"
        try:
            assert_async(fused(*pending.values()), message)
        except RuntimeError as error:
            if str(error) != message:
                raise
            # Failed (CPU raises here): the per-field checks name the field.
        else:
            if stamps:
                for value in pending.values():
                    _record_finite_stamp(value)
            return
    for name, value in selected.items():
        if not isinstance(value, torch.Tensor):
            continue
        if stamps and _finite_stamp_matches(value):
            continue
        message = f"{context}: {name} contains nonfinite values"
        try:
            # Keep the predicate on-device, including under forward/reverse AD.
            # CUDA reports assertion failures asynchronously; never read it back.
            # Cached validation must not add reverse saves only on its first execution.
            with torch.no_grad():
                assert_async(torch.isfinite(value).all(), message)
        except RuntimeError as error:
            if str(error) != message:
                raise
            raise MpasConfigError(message) from error
        if stamps:
            _record_finite_stamp(value)


# Tensors already checked finite: id -> (weak reference, version counter at the
# check). A tensor object whose version counter has not moved since its check
# holds the same values (every in-place write, eager or compiled, through any
# view, bumps the shared counter), so the check is not repeated. This skips the
# static base fields on every step and the step-input check of fields that the
# previous step's result check covered. Inference tensors have no version
# counter and are always checked.
_FINITE_CHECKED: dict[int, tuple[weakref.ref, int]] = {}


def _finite_stamp_matches(value: torch.Tensor) -> bool:
    if value.is_inference():
        return False
    entry = _FINITE_CHECKED.get(id(value))
    return entry is not None and entry[0]() is value and entry[1] == value._version


def _record_finite_stamp(value: torch.Tensor) -> None:
    if value.is_inference():
        return
    key = id(value)

    def _forget(ref: weakref.ref, key: int = key) -> None:
        entry = _FINITE_CHECKED.get(key)
        if entry is not None and entry[0] is ref:
            del _FINITE_CHECKED[key]

    _FINITE_CHECKED[key] = (weakref.ref(value, _forget), value._version)


__all__ = [
    "CheckpointPlan",
    "_run_simulation_jvp",
    "_run_simulation_vjp",
    "build_checkpoint_plan",
    "run_simulation",
    "run_timestep",
]


@dataclass(frozen=True)
class CheckpointPlan:
    """Immutable recomputation policy selected before numeric execution."""

    outer_timestep: bool
    nested_rk: bool
    outer_use_reentrant: bool
    materialize_final: bool
    nested_selection: str
    production_qualified: bool


def build_checkpoint_plan(
    *,
    differentiable: bool,
    checkpoint_timesteps: bool | None,
    checkpoint_use_reentrant: bool,
    checkpoint_materialize_final: bool,
    config: Mapping[str, Any] | None,
) -> CheckpointPlan:
    """Resolve outer and nested checkpointing without an implicit double layer."""
    if not differentiable:
        return CheckpointPlan(
            False,
            False,
            False,
            bool(checkpoint_materialize_final),
            "disabled",
            True,
        )

    outer = True if checkpoint_timesteps is None else bool(checkpoint_timesteps)
    nested_explicit = config is not None and "gradient_checkpointing" in config
    if config is not None and nested_explicit:
        nested = bool(config["gradient_checkpointing"])
        selection = "explicit_config"
    else:
        nested = not outer
        selection = "outer_only_default" if outer else "nested_only_default"
    return CheckpointPlan(
        outer_timestep=outer,
        nested_rk=nested,
        outer_use_reentrant=bool(checkpoint_use_reentrant) if outer else False,
        materialize_final=bool(checkpoint_materialize_final),
        nested_selection=selection,
        production_qualified=not (outer and nested),
    )


_INTER_STEP_PROGNOSTIC_KEYS = frozenset(
    {
        "u",
        "w",
        "theta_m",
        "rho_zz",
        "scalars",
        # Microphysics updates this dynamics tendency and the next RK step
        # consumes it in acoustic/pressure coupling, matching MPAS' tend pool.
        "rt_diabatic_tend",
    }
)
# These diagnostics are part of the discrete inter-timestep state even though
# they are not independently restartable prognostics.  ``atm_srk3`` carries
# their evolved values into the next step.  Reconstructing them from the
# prognostics changes both a handful of float64 values and, more importantly,
# the autograd graph being differentiated.  An outer checkpoint must therefore
# return these exact tensors and recompute them only as part of the checkpointed
# timestep itself.
_INTER_STEP_EXACT_DIAGNOSTIC_KEYS = frozenset(
    {
        "divergence",
        "exner",
        "exner_base",
        "h_edge",
        "ke",
        "pressure_base",
        "pressure_p",
        "pv_edge",
        "rho_edge",
        "rho_p",
        "rtheta_base",
        "rtheta_p",
        "ru",
        "rw",
        "uReconstructMeridional",
        "uReconstructX",
        "uReconstructY",
        "uReconstructZ",
        "uReconstructZonal",
        "ur_cell",
        "v",
        "vorticity",
        "vr_cell",
    }
)
_EXACT_CHECKPOINT_DIAGNOSTIC_MARKER = "_mpas_exact_checkpoint_diagnostic_keys"
_STATIC_BOUNDARY_SUPPORT_KEYS = frozenset(
    {
        "rho_base",
        "theta_base",
        "zz",
        "cellsOnEdge",
        "edgesOnCell",
        "edgesOnCell_sign",
        "nEdgesOnCell",
        "fzm",
        "fzp",
        "zb_cell",
        "zb3_cell",
        "coeffs_reconstruct",
        "latCell",
        "lonCell",
        "u_init",
        "v_init",
        "t_init",
        "qv_init",
        "ur_cell",
        "vr_cell",
        "uReconstructX",
        "uReconstructY",
        "uReconstructZ",
        "uReconstructZonal",
        "uReconstructMeridional",
    }
)
_FORWARD_INTERSTEP_STATIC_KEYS = _STATIC_BOUNDARY_SUPPORT_KEYS - frozenset(
    {
        "ur_cell",
        "vr_cell",
        "uReconstructX",
        "uReconstructY",
        "uReconstructZ",
        "uReconstructZonal",
        "uReconstructMeridional",
    }
)
_FORWARD_INTERSTEP_STATIC_KEYS = _FORWARD_INTERSTEP_STATIC_KEYS | frozenset(
    {
        "advCellsForEdge",
        "adv_coefs",
        "adv_coefs_3rd",
        "angleEdge",
        "areaCell",
        "bdyMaskCell",
        "bdyMaskEdge",
        "bdyMaskVertex",
        "cellsOnCell",
        "cellsOnEdge",
        "cellsOnVertex",
        "cf1",
        "cf2",
        "cf3",
        "dcEdge",
        "defc_a",
        "defc_b",
        "deriv_two",
        "dss",
        "dzu",
        "dzw",
        "dvEdge",
        "edgesOnCell",
        "edgesOnCell_sign",
        "edgesOnEdge",
        "edgesOnVertex",
        "edgesOnVertex_sign",
        "fEdge",
        "fVertex",
        "fnm",
        "fnp",
        "invAreaCell",
        "invAreaTriangle",
        "invDcEdge",
        "invDvEdge",
        "kiteAreasOnVertex",
        "kiteForCell",
        "latEdge",
        "meshScalingDel2",
        "meshScalingDel4",
        "nAdvCellsForEdge",
        "nEdgesOnEdge",
        "rdnw",
        "rdzu",
        "rdzw",
        "t_init",
        "u_init",
        "v_init",
        "verticesOnCell",
        "verticesOnEdge",
        "weightsOnEdge",
        "zb",
        "zb3",
        "zgrid",
        "zxu",
    }
)
_STATIC_BOUNDARY_SUPPORT_KEYS = (
    _STATIC_BOUNDARY_SUPPORT_KEYS | _FORWARD_INTERSTEP_STATIC_KEYS
)

# Public read-only ownership vocabulary used by benchmark runners. Keeping the
# aliases here prevents performance harnesses from inventing a second, drifting
# definition of the timestep boundary.
INTER_STEP_PROGNOSTIC_KEYS = _INTER_STEP_PROGNOSTIC_KEYS
INTER_STEP_EXACT_DIAGNOSTIC_KEYS = _INTER_STEP_EXACT_DIAGNOSTIC_KEYS
FORWARD_INTERSTEP_STATIC_KEYS = _FORWARD_INTERSTEP_STATIC_KEYS


def _checkpoint_exact_diagnostic_keys(state: Mapping[str, Any]) -> tuple[str, ...]:
    """Return and validate the exact diagnostics owned by a checkpoint boundary."""
    marker = state.get(_EXACT_CHECKPOINT_DIAGNOSTIC_MARKER)
    if marker is None:
        return tuple(
            sorted(
                key
                for key in _INTER_STEP_EXACT_DIAGNOSTIC_KEYS
                if isinstance(state.get(key), torch.Tensor)
            )
        )
    if not isinstance(marker, tuple) or marker != tuple(sorted(marker)):
        raise MpasConfigError("invalid exact checkpoint diagnostic marker")
    unknown = set(marker) - _INTER_STEP_EXACT_DIAGNOSTIC_KEYS
    if unknown:
        raise MpasConfigError(
            "exact checkpoint diagnostic marker contains unknown keys: "
            + ", ".join(sorted(unknown))
        )
    missing = [key for key in marker if not isinstance(state.get(key), torch.Tensor)]
    if missing:
        raise MpasConfigError(
            "exact checkpoint boundary lacks diagnostics: " + ", ".join(missing)
        )
    return marker


def _exact_checkpoint_boundary(
    previous_state: Mapping[str, Any],
    step_result: dict[str, Any],
) -> dict[str, Any]:
    """Return the exact tensor state consumed by the following timestep.

    The first boundary derives its diagnostic ownership from the prepared input;
    later boundaries inherit the frozen marker.  This makes deletion of any
    required evolved diagnostic a hard error instead of silently switching the
    discrete program to diagnostic rematerialization.
    """
    exact_diagnostic_keys = _checkpoint_exact_diagnostic_keys(previous_state)
    missing = [
        key
        for key in exact_diagnostic_keys
        if not isinstance(step_result.get(key), torch.Tensor)
    ]
    if missing:
        raise MpasConfigError(
            "checkpointed timestep failed to return exact diagnostics: "
            + ", ".join(missing)
        )
    boundary: dict[str, Any] = {}
    for key, value in step_result.items():
        if (
            not isinstance(value, torch.Tensor)
            or key in _INTER_STEP_PROGNOSTIC_KEYS
            or key in exact_diagnostic_keys
        ):
            boundary[key] = value
    boundary[_EXACT_CHECKPOINT_DIAGNOSTIC_MARKER] = exact_diagnostic_keys
    if "num_scalars" not in boundary and "scalars" in boundary:
        boundary["num_scalars"] = int(boundary["scalars"].shape[-1])
    return boundary


def _own_continuation_buffers(state: dict[str, Any]) -> dict[str, Any]:
    """Own writable restart fields without copying immutable mesh support."""
    copies: dict[int, torch.Tensor] = {}
    result = dict(state)
    for key, value in state.items():
        if (
            isinstance(value, torch.Tensor)
            and key not in _FORWARD_INTERSTEP_STATIC_KEYS
        ):
            identity = id(value)
            if identity not in copies:
                copies[identity] = value.clone()
            result[key] = copies[identity]
    return result


def _capture_static_boundary_support(
    prepared_state: dict[str, Any],
    mesh: dict[str, Any],
) -> dict[str, Any]:
    """Capture static tensor inputs needed to rematerialize future timesteps."""
    support: dict[str, Any] = {}
    for key in _STATIC_BOUNDARY_SUPPORT_KEYS:
        if key in prepared_state and key not in mesh:
            support[key] = prepared_state[key]
    return support


def _restore_static_boundary_support(
    state: dict[str, Any],
    static_support: dict[str, Any],
) -> None:
    """Inject missing static rematerialization inputs back into a boundary state."""
    for key, value in static_support.items():
        state.setdefault(key, value)


def _prepare_materialization_input(
    state: dict[str, Any],
    *,
    checkpoint_timesteps: bool,
    differentiable: bool,
    checkpoint_use_reentrant: bool,
    static_support: dict[str, Any],
) -> dict[str, Any]:
    """Provide a state dict with static support restored before rematerializing."""
    if not (
        checkpoint_timesteps
        and differentiable
        and not checkpoint_use_reentrant
        and static_support
    ):
        return state
    materialize_state = dict(state)
    _restore_static_boundary_support(materialize_state, static_support)
    return materialize_state


def _normalize_checkpointed_state(
    state: dict[str, Any],
    *,
    between_steps: bool,
) -> dict[str, Any]:
    """Normalize checkpointed outputs for continued differentiable use.

    Reentrant checkpoint forward executes under ``torch.no_grad()`` and the
    current dual-mode timestep path uses ``torch.inference_mode()`` for the
    non-differentiable branch. Any tensor that leaves the checkpoint boundary
    and will be used in later differentiable code must therefore be materialized
    back into a regular tensor. Between timesteps, prognostics and the evolved
    diagnostics must retain their history. A returned diagnostic can also be
    an objective or a later physics input; normalization cannot detach it.
    """
    normalized: dict[str, Any] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            normalized[key] = value
            continue
        normalized[key] = value.clone() if value.is_inference() else value
    return normalized


def _memory_probe_enabled() -> bool:
    """Return whether timestep RSS logging is enabled."""
    return os.environ.get("MPAS_MEM_PROBE") == "1"


def _log_timestep_rss(step: int, phase: str) -> None:
    """Emit the current process RSS high-water mark for the given timestep phase."""
    if not _memory_probe_enabled():
        return
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    LOGGER.info("mem_probe step=%d phase=%s ru_maxrss=%s", step, phase, rss)


def _compile_requested(config: Mapping[str, Any] | None) -> bool:
    """Return whether raw caller config requests torch.compile."""
    if os.environ.get("MPAS_DISABLE_TORCH_COMPILE", "0") == "1":
        return False
    if not config:
        return False
    for key in ("config_use_torch_compile", "use_torch_compile"):
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise MpasConfigError(
                    f"{key} must be a Python bool or CPU tensor scalar; "
                    f"got {value.device.type} tensor"
                )
            if value.numel() != 1:
                return bool(value.any().item())
            value = value.item()
        return bool(value)
    return False


def _nondifferentiable_context(config: Mapping[str, Any] | None):
    if _compile_requested(config):
        return torch.no_grad()
    return torch.inference_mode()


def _graph_capturable_halo(exchanger: Any) -> bool:
    """Only NCCL transfers are device work that a CUDA graph records."""
    import torch.distributed as dist

    from mpas_pytorch.distributed.dynamics import DynamicsHaloExchange
    from mpas_pytorch.distributed.halo import HaloTransport

    return (
        isinstance(exchanger, DynamicsHaloExchange)
        and isinstance(exchanger.transport, HaloTransport)
        and dist.is_initialized()
        and dist.get_backend() == "nccl"
    )


def _ranks_agree_to_record(config: Mapping[str, Any], steady: bool) -> bool:
    """Record only when every rank is steady, so all record the same step.

    One host read of a control flag per step until recording; model values
    never leave the device.
    """
    exchanger = config.get("_halo_exchanger")
    if exchanger is None:
        return steady
    import torch.distributed as dist

    flag = torch.tensor(
        [int(steady)], dtype=torch.int64, device=exchanger.transport.device
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(int(flag))


def _graph_cache_view(config: Mapping[str, Any]) -> tuple[Any, ...]:
    """Sizes of the runner caches that a recorded step must find warm."""
    from mpas_pytorch.dynamics.compile_utils import _COMPILED_CACHE

    store = config.get("_dycore_static_payloads")
    exchanger = config.get("_halo_exchanger")
    plans = (
        (*exchanger.plans.values(), *exchanger.group_plans.values())
        if exchanger is not None
        else ()
    )
    return (
        tuple(sorted(config)),
        id(store),
        store.snapshot() if store is not None else None,
        len(_COMPILED_CACHE),
        tuple(
            (len(getattr(plan, "buffers", ())), len(getattr(plan, "schedules", ())))
            for plan in plans
        ),
    )


def _require_timestep_graph_policy(
    config: Mapping[str, Any],
    *,
    device: torch.device | str,
    differentiable: bool,
    prune_forward_intersteps: bool,
    prepared_once: bool,
    on_step: Callable[[int, dict[str, Any]], None] | None,
) -> None:
    """Reject requests the steady-boundary graph policy cannot honor."""
    missing = []
    if torch.device(device).type != "cuda":
        missing.append("a CUDA device")
    if differentiable:
        missing.append("a nondifferentiable forecast")
    if not prune_forward_intersteps or not prepared_once:
        missing.append("pruned forward interstep boundaries")
    if on_step is not None:
        missing.append("no per-step observer")
    if (
        compile_enabled(dict(config))
        and compile_backend(dict(config)) == "inductor"
        and bool(
            torch._inductor.list_mode_options(compile_mode(dict(config))).get(
                "triton.cudagraphs", torch._inductor.config.triton.cudagraphs
            )
        )
    ):
        missing.append("a compile mode without Inductor CUDA graphs")
    if (
        config.get("_halo_observer") is not None
        or config.get("_mpas_section_timer") is not None
        or bool(config.get("config_profile_sections", False))
    ):
        missing.append("no halo observer or section timing")
    exchanger = config.get("_halo_exchanger")
    if exchanger is not None and not _graph_capturable_halo(exchanger):
        # Host-side transports would run once while recording, never on replay.
        missing.append("an NCCL halo transport")
    if bool(config.get("config_apply_lbcs", False)):
        # Boundary forcing changes every step; a replay would freeze it.
        missing.append("no lateral boundary forcing")
    if config.get("_dycore_workspace") is not None or bool(
        config.get("config_use_workspace", False)
    ):
        missing.append("no workspace scratch reuse")
    if missing:
        raise MpasConfigError(
            "config_cuda_graph_timestep requires " + ", ".join(missing)
        )


def _check_device_available(
    device: torch.device | str, *, dtype: torch.dtype = torch.float64
) -> torch.device:
    """Reject unavailable accelerator devices at the public API boundary."""
    requested = device if isinstance(device, torch.device) else torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise MpasDeviceError(
            requested=str(device),
            message=(
                f"CUDA requested (device={str(device)!r}) but torch.cuda.is_available() is False. "
                f"Install PyTorch with CUDA support, or pass device='cpu'."
            ),
        )
    if requested.type == "mps":
        mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if mps_available and dtype == torch.float32:
            return requested
        if mps_available:
            raise MpasDeviceError(
                requested=str(device),
                message=(
                    "MPS requested but MPAS-PyTorch requires float64 dynamics, "
                    "which Apple MPS does not support. Pass device='cpu' or use CUDA float64."
                ),
            )
        raise MpasDeviceError(
            requested=str(device),
            message=(
                f"MPS requested (device={str(device)!r}) but torch.backends.mps.is_available() is False. "
                f"Enable Apple Metal support, or pass device='cpu'."
            ),
        )
    return requested


def _restore_padding_sentinels(state: dict[str, Any]) -> dict[str, Any]:
    """Restore fixed padding-row sentinels expected by downstream kernels/tests."""
    rho_zz = state.get("rho_zz")
    n_cells = state.get("nCells")
    if not isinstance(rho_zz, torch.Tensor) or n_cells is None:
        return state
    n_cells_int = int(n_cells)
    if rho_zz.ndim < 2 or rho_zz.shape[0] <= n_cells_int:
        return state
    # Never stamp the sentinel row in place on a caller-owned tensor. This
    # helper is used at preparation/output boundaries, so a fresh tensor here
    # preserves the public no-mutation contract for run_timestep/run_simulation
    # without relying on whole-state deepcopy guards.
    rho_zz = rho_zz.clone()
    rho_zz[n_cells_int, :] = 1.0
    state["rho_zz"] = rho_zz
    return state


def _project_checkpoint_boundary_state(
    previous_state: dict[str, Any],
    step_state: dict[str, Any],
) -> dict[str, Any]:
    """Project a full timestep result onto the exact reentrant boundary."""
    boundary = _exact_checkpoint_boundary(previous_state, step_state)
    for key, value in previous_state.items():
        if not isinstance(value, torch.Tensor) and key not in boundary:
            boundary[key] = value
    for key in _STATIC_BOUNDARY_SUPPORT_KEYS:
        if key in previous_state:
            boundary.setdefault(key, previous_state[key])
    return boundary


def _project_forward_interstep_state(
    previous_state: dict[str, Any],
    step_state: dict[str, Any],
    *,
    step_number: int,
    nsteps: int,
) -> dict[str, Any]:
    """Drop transient tensors between nondifferentiable forward timesteps."""
    if step_number >= nsteps:
        return step_state
    boundary: dict[str, Any] = {}
    for key, value in previous_state.items():
        if not isinstance(value, torch.Tensor):
            boundary[key] = value
    for key in _FORWARD_INTERSTEP_STATIC_KEYS:
        if key in previous_state:
            boundary[key] = previous_state[key]
    boundary.update(_exact_checkpoint_boundary(previous_state, step_state))
    # Recovery already owns and pads this step's density. Cloning it here
    # expires the exact diagnostic receipt and forces an unnecessary rebuild.
    return boundary


def _materialize_step_state(
    state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict[str, Any],
    dt: float,
    *,
    strict: bool,
) -> dict[str, Any]:
    """Recompute diagnostics/vector outputs from a checkpoint boundary state."""
    materialized = dict(state)
    profiled_call(
        "mpas.init_coupled_diagnostics",
        config,
        _init_coupled_diagnostics,
        materialized,
        mesh,
        config=config,
        strict=strict,
    )
    profiled_call(
        "mpas.restore_padding_sentinels",
        config,
        _restore_padding_sentinels,
        materialized,
    )
    profiled_call(
        "mpas.init_solve_diagnostics",
        config,
        _init_solve_diagnostics,
        materialized,
        mesh,
        config=config,
        dt=dt,
        strict=strict,
    )
    profiled_call(
        "mpas.init_vector_reconstruction",
        config,
        _init_vector_reconstruction,
        materialized,
        mesh,
        config=config,
        strict=strict,
    )
    if "num_scalars" not in materialized and "scalars" in materialized:
        materialized["num_scalars"] = int(materialized["scalars"].shape[-1])
    return materialized


def _merge_materialized_state(
    authoritative_state: dict[str, Any],
    materialized_state: dict[str, Any],
) -> dict[str, Any]:
    """Fill missing diagnostics from a materialized state without overwriting real outputs."""
    merged = dict(materialized_state)
    merged.update(authoritative_state)
    return merged


def _rematerialize_interstep_diagnostics(
    state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict[str, Any],
    dt: float,
    *,
    strict: bool,
) -> None:
    """Refresh only the diagnostics required at the next timestep boundary."""
    if "num_scalars" not in state and "scalars" in state:
        state["num_scalars"] = int(state["scalars"].shape[-1])
    profiled_call(
        "mpas.init_coupled_diagnostics",
        config,
        _init_coupled_diagnostics,
        state,
        mesh,
        config=config,
        strict=strict,
    )
    profiled_call(
        "mpas.restore_padding_sentinels", config, _restore_padding_sentinels, state
    )
    profiled_call(
        "mpas.init_solve_diagnostics",
        config,
        _init_solve_diagnostics,
        state,
        mesh,
        config=config,
        dt=dt,
        strict=strict,
    )
    profiled_call(
        "mpas.init_vector_reconstruction",
        config,
        _init_vector_reconstruction,
        state,
        mesh,
        config=config,
        strict=strict,
    )


_DIMENSION_INTS: dict[int, tuple[weakref.ReferenceType[torch.Tensor], int, int]] = {}


def _dimension_int(value: Any, default: int) -> int:
    """Read a mesh dimension once per live, unmodified 0-d tensor.

    Dimensions are stored as 0-d tensors that may live on the accelerator; each
    ``int()`` of such a tensor is a device synchronization. The cached value is
    keyed by the tensor object and its version counter, so it is exact.
    """
    if value is None:
        return default
    if not isinstance(value, torch.Tensor):
        return int(value)
    try:
        version = int(value._version)
    except RuntimeError:  # inference tensors carry no version counter
        return int(value)
    key = id(value)
    cached = _DIMENSION_INTS.get(key)
    if cached is not None and cached[0]() is value and cached[1] == version:
        return cached[2]
    result = int(value)

    def discard(reference: weakref.ReferenceType[torch.Tensor]) -> None:
        current = _DIMENSION_INTS.get(key)
        if current is not None and current[0] is reference:
            del _DIMENSION_INTS[key]

    # WeakKeyDictionary compares tensor referents and synchronizes on equality.
    _DIMENSION_INTS[key] = (weakref.ref(value, discard), version, result)
    return result


# State entries the no-grad RK step writes in place.
_NO_GRAD_WORK_BUFFERS = (
    "tend_rho", "tend_rho_zz", "tend_u_euler", "tend_w_euler", "tend_theta_euler",
    "ru_p", "rw_p", "rho_pp", "rtheta_pp", "kdiff", "flux_arr", "flux_tmp",
    "flux_upwind_tmp", "s_max", "s_min", "wdtn", "scale_arr", "scalar_tend",
)


def _without_step_local_fields(result: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """``result`` without the fields a caller declared step-local (private config key
    ``_step_local_fields``): work buffers and time levels that the next step creates again
    and never reads, so the boundary does not carry them."""
    names = config.get("_step_local_fields", ())
    return {k: v for k, v in result.items() if k not in names} if names else result


def _run_timestep_impl(
    state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict | None = None,
    dt: float | None = None,
    device: torch.device | str = _CPU_DEVICE,
    *,
    differentiable: bool = False,
    config_validated: bool = False,
    experimental: bool = False,
) -> dict[str, Any]:
    """Advance model state by one timestep using RK3."""
    # Isolate caller-owned mappings while still sharing the underlying tensors.
    # Production prep/rematerialization code may replace entries such as rho_zz
    # inside the working state dict; a shallow copy preserves the public
    # no-mutation contract without deep-copying every field.
    state = dict(state)
    mesh = dict(mesh)
    boundary_mesh = dict(mesh)
    state, full_config, strict = _prepare_timestep_state(
        state=state,
        mesh=mesh,
        config=config,
        dt=dt,
        device=device,
        config_validated=config_validated,
    )
    if dt is None:
        dt = float(full_config["config_dt"])

    # The no-grad integrator reuses these work buffers in place (_clone_for_grad
    # clones only under grad). Private copies keep the caller's tensors unchanged.
    if not differentiable:
        for name in _NO_GRAD_WORK_BUFFERS:
            if name in state:
                state[name] = state[name].clone()
    result = atm_srk3(
        state,
        mesh,
        full_config,
        dt,
        strict=strict,
        differentiable=differentiable,
        **({"experimental": True} if experimental else {}),
    )
    if full_config.get("_native_boundary_output", False):
        _require_finite_model_fields(result, "native timestep result",
            fields=NATIVE_STATE_KEYS, config=full_config)
        native = {name: result[name] for name in NATIVE_STATE_KEYS}
        return native if differentiable else own_cuda_graph_outputs(native)
    _require_finite_model_fields(result, "timestep result", config=full_config)
    # A completed RK step already carries the coupled and solve diagnostics
    # needed by the next step. Re-deriving them on a later public call changes
    # the floating-point trajectory at the call boundary.
    result = _without_step_local_fields(result, full_config)
    if not differentiable:
        result = own_cuda_graph_outputs(result)
    mark_complete_step_boundary(result, mesh=boundary_mesh, config=full_config, dt=dt)
    return result


def _retain_boundary_diagnostics_when(
    state: dict[str, Any], carried: dict[str, Any], predicate: torch.Tensor
) -> None:
    for name, value in tuple(state.items()):
        previous = carried.get(name)
        if (
            isinstance(value, torch.Tensor)
            and isinstance(previous, torch.Tensor)
            and value is not previous
        ):
            if value.shape != previous.shape or value.dtype != previous.dtype:
                raise MpasConfigError(
                    f"complete boundary rederivation changed {name} shape/dtype"
                )
            state[name] = torch.where(predicate, previous, value)


def _prepare_timestep_state(
    state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict | None = None,
    dt: float | None = None,
    device: torch.device | str = _CPU_DEVICE,
    *,
    config_validated: bool = False,
) -> tuple[dict[str, Any], dict, bool]:
    """Prepare state/mesh for numerical integration (pre-atm_srk3 stage)."""
    full_config = dict(config or {}) if config_validated else validate_config(config)
    target_device = torch.device(device)
    input_device = require_consistent_device("run_timestep inputs", state, mesh)
    _require_finite_model_fields(state, "timestep input", config=full_config)

    if input_device != target_device:
        state = dict_to_device(state, target_device)
        mesh_on_target = dict_to_device(mesh, target_device)
        mesh.clear()
        mesh.update(mesh_on_target)

    if consume_ufs_physics_commit(
        state,
        mesh=mesh,
        config=full_config,
        dt=dt if dt is not None else float(full_config["config_dt"]),
    ):
        # R4 synchronizes time levels, then refreshes output/solve diagnostics.
        # The integrator creates its saved time levels from these live primaries.
        # Cold initialization would replace source-owned rw and base fields.
        state["nVertLevels"] = _dimension_from_shape(
            {**mesh, **state}, "nVertLevels", state["u"].shape[1]
        )
        # The live UFS commit retains native topology. The merge verifies this
        # contract with tensor assertions instead of inferring a host policy.
        state["_edge_ordering_mismatch"] = False
        refresh_output_diagnostics(state, full_config)
        _init_vector_reconstruction(state, mesh, config=full_config, strict=True)
        _init_solve_diagnostics(state, mesh, config=full_config, dt=dt, strict=True)
        return state, full_config, True

    if isinstance(config, Mapping) and "strict" in config:
        strict = bool(config["strict"])
    else:
        strict = bool(
            full_config.get("strict_initialization", full_config.get("strict", True))
        )

    use_rcm_val = full_config.get("config_mesh_reordering_rcm", False)
    if isinstance(use_rcm_val, torch.Tensor):
        if use_rcm_val.numel() != 1:
            raise MpasConfigError(
                "config_mesh_reordering_rcm must be a scalar bool/int tensor "
                f"(got shape {tuple(use_rcm_val.shape)})"
            )
        use_rcm = bool(use_rcm_val)
    else:
        use_rcm = bool(use_rcm_val)
    if use_rcm and not mesh.get("_rcm_applied", False):
        from mpas_pytorch.mesh_reordering import apply_rcm_reordering

        state_reordered, mesh_reordered = apply_rcm_reordering(state, mesh)
        state_reordered = dict_to_device(state_reordered, target_device)
        mesh_reordered = dict_to_device(mesh_reordered, target_device)
        state = state_reordered
        mesh.clear()
        mesh.update(mesh_reordered)

    require_boundary_continuation_identity(
        state,
        mesh=mesh,
        config=full_config,
        dt=dt if dt is not None else float(full_config["config_dt"]),
    )
    complete_boundary = timestep_boundary_predicate(state)
    if complete_boundary is not None:
        # Eager scratch writes also run outside compiled regions. A retained
        # public result must not become the next invocation's writable pool.
        # The on-device predicate is evaluated first, on the caller's tensors.
        state = profiled_call(
            "mpas.continuation_input_ownership",
            full_config,
            _own_continuation_buffers,
            state,
        )
    state = profiled_call(
        "mpas.initialize_from_state",
        full_config,
        initialize_from_state,
        state,
        mesh,
        full_config,
        strict=strict,
    )
    if full_config.get("_halo_exchanger") is not None:
        from mpas_pytorch.distributed.dynamics import exchange_halo_group

        exchange_halo_group(
            full_config,
            "initialization:prognostics"
            if "theta_m" in state and "rho_zz" in state
            else "initialization:raw_prognostics",
            state,
        )
    target_coef = get_config_value_safe(full_config, "config_coef_3rd_order", None)
    if target_coef is None:
        target_coef = get_config_value_safe(full_config, "coef_3rd_order", None)
    if target_coef is not None:
        from mpas_pytorch.ic_loader import apply_coef_3rd_order_override_fields

        apply_coef_3rd_order_override_fields(
            state,
            mesh,
            coef_3rd_order=float(target_coef),
        )
    carried_boundary = dict(state) if complete_boundary is not None else None
    state.pop("_restart_skip_rederivation", None)
    profiled_call(
        "mpas.init_coupled_diagnostics",
        full_config,
        _init_coupled_diagnostics,
        state,
        mesh,
        config=full_config,
        strict=strict,
    )
    profiled_call(
        "mpas.restore_padding_sentinels", full_config, _restore_padding_sentinels, state
    )
    profiled_call(
        "mpas.init_solve_diagnostics",
        full_config,
        _init_solve_diagnostics,
        state,
        mesh,
        config=full_config,
        dt=dt,
        strict=strict,
    )
    profiled_call(
        "mpas.init_vector_reconstruction",
        full_config,
        _init_vector_reconstruction,
        state,
        mesh,
        config=full_config,
        strict=strict,
    )
    if full_config.get("_halo_exchanger") is not None:
        exchange_halo_group(full_config, "initialization:pv_edge,ru,rw", state)
    if "num_scalars" not in state and "scalars" in state:
        state["num_scalars"] = int(state["scalars"].shape[-1])
    if carried_boundary is not None:
        # Complete diagnostics remain source-owned carry. Primary edits use the
        # same cold rederivation, selected on-device instead of a Python read.
        _retain_boundary_diagnostics_when(state, carried_boundary, complete_boundary)
    return state, full_config, strict


def prepare_initial_state(
    state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict | None = None,
    dt: float | None = None,
    device: torch.device | str = _CPU_DEVICE,
    *,
    differentiable: bool = False,
    experimental: bool = False,
) -> dict[str, Any]:
    """Materialize the time-zero state that ``run_timestep`` integrates, without advancing it.

    Runs the same preparation as a step (initialization, diagnostic solves, vector
    reconstruction) on private copies and seals the result as a complete step
    boundary, so typed diagnostics at the initial valid time come from the model.
    Caller mappings and tensors are unchanged.
    """
    _validate_public_inputs(state, mesh, device, experimental=experimental, config=config)
    config = validate_config(config)
    # Derivatives through the preparation only when asked (the model's explicit gradient mode);
    # otherwise the caller's mode applies, as for a plain step.
    with differentiable_mode() if differentiable else contextlib.nullcontext():
        prepared_mesh = clone_tree(mesh)
        # Continuation binds the caller's mesh, before preparation adds private support.
        boundary_mesh = dict(prepared_mesh)
        prepared, full_config, _ = _prepare_timestep_state(
            state=clone_tree(state),
            mesh=prepared_mesh,
            config=config,
            dt=dt,
            device=device,
            config_validated=True,
        )
    mark_complete_step_boundary(
        prepared,
        mesh=boundary_mesh,
        config=full_config,
        dt=float(full_config["config_dt"] if dt is None else dt),
    )
    return prepared

def run_timestep(
    state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict | None = None,
    dt: float | None = None,
    device: torch.device | str = _CPU_DEVICE,
    differentiable: bool = False,
    experimental: bool = False,
) -> dict[str, Any]:
    """Advance model state by one timestep using RK3.

    Strict-mode behavior:
    - If `config['strict']` is provided, that value is used.
    - Otherwise strictness mirrors `strict_initialization` when present.
    - Fallback remains strict (`True`) by default.
    """
    _validate_public_inputs(
        state, mesh, device, experimental=experimental, config=config
    )
    config = validate_config(config)
    effective_dt = (
        dt
        if dt is not None
        else (config or {}).get("config_dt", FORTRAN_DEFAULTS["config_dt"])
    )
    if not math.isfinite(float(effective_dt)) or effective_dt <= 0.0:
        raise MpasConfigError("dt must be finite and positive")
    config_validated = True
    if differentiable:
        checkpoint_plan = build_checkpoint_plan(
            differentiable=True,
            checkpoint_timesteps=False,
            checkpoint_use_reentrant=False,
            checkpoint_materialize_final=True,
            config=config,
        )
        effective_config = dict(config) if config else {}
        effective_config["gradient_checkpointing"] = checkpoint_plan.nested_rk
        return _run_timestep_impl(
            state=state,
            mesh=mesh,
            config=effective_config,
            dt=dt,
            device=device,
            differentiable=True,
            config_validated=config_validated,
            experimental=experimental,
        )
    # no_grad, not inference_mode: a forward step's result stays an ordinary tensor (version
    # counters; usable as the start of a differentiable window). run_simulation keeps
    # inference mode for standalone forecasts.
    with torch.no_grad(), cuda_graph_forward_scope(config, device):
        return _run_timestep_impl(
            state=state,
            mesh=mesh,
            config=config,
            dt=dt,
            device=device,
            config_validated=config_validated,
            experimental=experimental,
        )


_INIT_COUPLED_DIAGNOSTIC_KEYS = (
    "theta_m",
    "rho_zz",
    "ru",
    "rw",
    "rho_p",
    "rtheta_base",
    "rtheta_p",
    "exner",
    "exner_base",
    "pressure_p",
    "pressure_base",
)


def _pad_init_coupled_diagnostics(
    result: Mapping[str, torch.Tensor],
    *,
    compute_dtype: torch.dtype,
    nCells: int,
    nEdges: int,
    nVertLevels: int,
) -> dict[str, torch.Tensor]:
    """Embed diagnostics in append-padded arrays.

    Valid data at indices 0..N-1, trailing zero padding at index N. The pad row is concatenated, not
    scattered, so both AD modes and compiled regions see one functional copy.
    """
    padded_result: dict[str, torch.Tensor] = {}
    for key in _INIT_COUPLED_DIAGNOSTIC_KEYS:
        if key not in result:
            continue
        val = result[key]
        rows = nEdges if key == "ru" else nCells
        columns = nVertLevels + 1 if key == "rw" else (nVertLevels if key == "ru" else val.shape[-1])
        if tuple(val.shape) != (rows, columns) or val.dtype != compute_dtype:
            raise ValueError(f"{key} diagnostic has shape {tuple(val.shape)} and dtype {val.dtype}")
        padded_result[key] = torch.cat((val, val.new_zeros((1, columns))))
    return padded_result


def _init_coupled_diagnostics_padded_numeric(
    *, compute_dtype: torch.dtype, _exner_pressure: bool = True, **kwargs: Any
) -> dict[str, torch.Tensor]:
    """Diagnostics embedded in append-padded arrays (also the compile region).

    The compile region passes ``_exner_pressure=False``: the caller then
    computes Exner and pressure eagerly (see coupled_exner_pressure).
    """
    result = atm_init_coupled_diagnostics(**kwargs, _exner_pressure=_exner_pressure)
    return _pad_init_coupled_diagnostics(
        result,
        compute_dtype=compute_dtype,
        nCells=kwargs["nCells"],
        nEdges=kwargs["nEdges"],
        nVertLevels=kwargs["nVertLevels"],
    )


def _init_coupled_diagnostics(
    state: dict[str, Any],
    mesh: dict[str, Any],
    strict: bool = True,
    *,
    config: dict | None = None,
) -> None:
    """Compute coupled diagnostic fields from prognostic state.

    This wraps atm_init_coupled_diagnostics and is
    called once before the first RK3 timestep. Diagnostics are ALWAYS
    recomputed from the current prognostic state.

    The function handles the index convention mismatch: atm_init_coupled_diagnostics
    operates on 0-based arrays without padding, while the orchestration state uses
    1-based arrays with +1 padding. Results are embedded into padded arrays.

    Args:
        state: Model state dictionary (modified in place).
        mesh: Mesh connectivity and geometry dictionary.
        strict: If True, raise ValueError when required inputs are missing.
    """
    # Diagnostic fields produced by init_coupled_diagnostics
    compute_dtype = floating_dtype(state, mesh)
    diag_keys = _INIT_COUPLED_DIAGNOSTIC_KEYS

    # Always compute; do not retain externally supplied derived diagnostics.

    # Collect all sources (state and mesh may overlap)
    combined = {}
    combined.update(mesh)
    combined.update(state)

    # Also need nCells, nEdges, nVertLevels
    nCells = _dimension_int(combined.get("nCells"), 0)
    nEdges = _dimension_int(combined.get("nEdges"), 0)
    nVertLevels = _dimension_int(combined.get("nVertLevels"), 0)

    # Check required inputs for the kernel
    required_state = ["u", "w"]
    raw_inputs = ["theta", "rho", "qv"]
    derived_inputs = ["theta_m", "rho_zz"]
    required_base = ["rho_base", "theta_base", "zz"]
    required_mesh = [
        "cellsOnEdge",
        "edgesOnCell",
        "edgesOnCell_sign",
        "nEdgesOnCell",
        "fzm",
        "fzp",
        "zb_cell",
        "zb3_cell",
    ]

    missing = []
    for k in required_state + required_base + required_mesh:
        if k not in combined:
            missing.append(k)
    if not all(k in combined for k in raw_inputs) and not all(
        k in combined for k in derived_inputs
    ):
        missing.append("theta/rho/qv or theta_m/rho_zz")

    if nCells == 0 or nEdges == 0 or nVertLevels == 0:
        missing.append("nCells/nEdges/nVertLevels")

    if missing:
        if strict:
            raise MpasConfigError(
                f"Cannot compute init_coupled_diagnostics: missing inputs: {missing}. "
                f"Provide these from checkpoint or set strict=False to skip."
            )
        return  # Non-strict: silently skip

    # Extract arrays. State arrays use append-padding: (N+1, ...) where
    # valid data is at indices 0..N-1 and index N is the garbage/padding row.
    # The kernel expects 0-based without padding: (N, ...).
    # Slice off the trailing padding row.
    theta_m_padded = combined.get("theta_m")  # (nCells+1, nVertLevels)
    rho_zz_padded = combined.get("rho_zz")  # (nCells+1, nVertLevels)
    theta_padded = combined.get("theta")
    rho_padded = combined.get("rho")
    qv_padded = combined.get("qv")
    u_padded = combined["u"]  # (nEdges+1, nVertLevels)
    w_padded = combined["w"]  # (nCells+1, nVertLevels+1)
    rho_base_padded = combined["rho_base"]  # (nCells+1, nVertLevels)
    theta_base_padded = combined["theta_base"]  # (nCells+1, nVertLevels)
    zz_padded = combined["zz"]  # (nCells+1, nVertLevels+1) or (nCells+1, nVertLevels)

    # Convert from append-padded to unpadded: valid data at 0..N-1
    theta_m = theta_m_padded[:nCells] if theta_m_padded is not None else None
    rho_zz = rho_zz_padded[:nCells] if rho_zz_padded is not None else None
    theta = theta_padded[:nCells] if theta_padded is not None else None
    rho = rho_padded[:nCells] if rho_padded is not None else None
    qv = qv_padded[:nCells] if qv_padded is not None else None
    u = u_padded[:nEdges]  # (nEdges, nVertLevels)
    w = w_padded[:nCells]  # (nCells, nVertLevels+1)
    rho_base = rho_base_padded[:nCells]  # (nCells, nVertLevels)
    theta_base = theta_base_padded[:nCells]  # (nCells, nVertLevels)

    # zz may be (nCells+1, nVertLevels+1) or (nCells+1, nVertLevels)
    zz = zz_padded[:nCells]  # (nCells, ...)

    # Connectivity arrays stay as-is (1-based padded, kernel expects this)
    cellsOnEdge = combined["cellsOnEdge"]  # (nEdges+1, 2)
    edgesOnCell = combined["edgesOnCell"]  # (nCells+1, maxEdges)
    edgesOnCell_sign = combined["edgesOnCell_sign"]  # (nCells+1, maxEdges)
    nEdgesOnCell = combined["nEdgesOnCell"]  # (nCells+1,)

    init_connectivity_payload = None
    init_payload_kwargs = {
        "device": u.device,
        "nCells": nCells,
        "nEdges": nEdges,
        "nVertLevels": nVertLevels,
        "cellsOnEdge": cellsOnEdge,
        "edgesOnCell": edgesOnCell,
        "edgesOnCell_sign": edgesOnCell_sign,
        "nEdgesOnCell": nEdgesOnCell,
    }
    if isinstance(config, Mapping):
        init_key = _init_coupled_connectivity_payload_cache_key(**init_payload_kwargs)
        if init_key is not None:
            store = config.get("_dycore_static_payloads")
            if isinstance(store, DycoreStaticPayloads):
                init_connectivity_payload = store.get_or_build_typed(
                    "init_coupled_diagnostics",
                    init_key,
                    lambda: _init_coupled_connectivity_payload(**init_payload_kwargs),
                )
            elif isinstance(config, dict):
                init_connectivity_payload = ensure_static_payload_store(
                    config
                ).get_or_build_typed(
                    "init_coupled_diagnostics",
                    init_key,
                    lambda: _init_coupled_connectivity_payload(**init_payload_kwargs),
                )

    # fzm, fzp: may be (nVertLevels+1,) with padding or (nVertLevels,)
    fzm = combined["fzm"]
    fzp = combined["fzp"]
    # If padded (nVertLevels+1,), strip trailing padding to (nVertLevels,)
    if fzm.shape[0] == nVertLevels + 1:
        fzm = fzm[:nVertLevels]
    if fzp.shape[0] == nVertLevels + 1:
        fzp = fzp[:nVertLevels]

    zb_cell = combined["zb_cell"]  # (nCells+1, maxEdges, nVertLevels+1)
    zb3_cell = combined["zb3_cell"]  # (nCells+1, maxEdges, nVertLevels+1)

    constants = constants_for_config(config)
    init_kwargs: dict[str, Any] = {
        "nCells": nCells,
        "nEdges": nEdges,
        "nVertLevels": nVertLevels,
        "rgas": constants.RD,
        "cp": constants.CP,
        "p0": constants.P0,
        "rvord": constants.RVORD,
        "theta_m": theta_m,
        "rho_zz": rho_zz,
        "theta": theta,
        "rho": rho,
        "qv": qv,
        "u": u,
        "w": w,
        "rho_base": rho_base,
        "theta_base": theta_base,
        "zz": zz,
        "cellsOnEdge": cellsOnEdge,
        "edgesOnCell": edgesOnCell,
        "edgesOnCell_sign": edgesOnCell_sign,
        "nEdgesOnCell": nEdgesOnCell,
        "fzm": fzm,
        "fzp": fzp,
        "zb_cell": zb_cell,
        "zb3_cell": zb3_cell,
        "_connectivity_payload": init_connectivity_payload,
    }
    compiled = (
        forward_compiled_region(
            "coupled_diagnostics_region",
            _init_coupled_diagnostics_padded_numeric,
            config,
            u.device,
        )
        if init_connectivity_payload is not None
        else None
    )
    if compiled is None:
        padded = _init_coupled_diagnostics_padded_numeric(
            compute_dtype=compute_dtype, **init_kwargs
        )
    else:
        # Exner and pressure stay eager (one pow implementation in both modes),
        # computed from the region's own rtheta fields.
        padded = compiled(
            compute_dtype=compute_dtype, _exner_pressure=False, **init_kwargs
        )
        padded.update(
            _pad_init_coupled_diagnostics(
                coupled_exner_pressure(
                    zz,
                    padded["rtheta_p"][:nCells],
                    padded["rtheta_base"][:nCells],
                    rgas=constants.RD,
                    cp=constants.CP,
                    p0=constants.P0,
                ),
                compute_dtype=compute_dtype,
                nCells=nCells,
                nEdges=nEdges,
                nVertLevels=nVertLevels,
            )
        )
    for key in diag_keys:
        if key in padded:
            state[key] = padded[key]


def _init_solve_diagnostics_compiled_plan(
    config: dict | None,
    connectivity_payload: Any,
    combined: Mapping[str, Any],
    *,
    nCells: int,
    nEdges: int,
    nVertices: int,
    nVertLevels: int,
    dt: float,
    config_apvm_upwinding: float,
) -> Any:
    """Store-cached compiled diagnostic-solve plan for the refresh, when targeted.

    Returns None (the work function then builds its eager plan as before) unless
    the run explicitly compiles ``diagnostic_solve_solve_diagnostics`` for this
    device. The plan is the same one the timestep uses; it is cached so the
    compiled wrapper is created once, not per refresh.
    """
    if not isinstance(config, dict) or connectivity_payload is None:
        return None
    if torch.is_grad_enabled():
        return None
    from mpas_pytorch.dynamics.compile_utils import (
        compile_backend,
        compile_enabled,
        compile_mode,
        kernel_compile_enabled_for_device,
        maybe_compile_kernel,
    )
    from mpas_pytorch.dynamics.solve_diagnostics import (
        build_diagnostic_solve_numeric_plan,
        compile_diagnostic_solve_numeric_plan,
    )

    device = combined["u"].device
    if not compile_enabled(config) or not kernel_compile_enabled_for_device(
        "diagnostic_solve_solve_diagnostics", config, device=device
    ):
        return None
    store = ensure_static_payload_store(config)
    plan = build_diagnostic_solve_numeric_plan(
        connectivity_payload=connectivity_payload,
        nCells=nCells,
        nEdges=nEdges,
        nVertices=nVertices,
        nVertLevels=nVertLevels,
        dt=dt,
        config_apvm_upwinding=config_apvm_upwinding,
        fVertex=combined["fVertex"],
        invAreaTriangle=combined["invAreaTriangle"],
        invAreaCell=combined["invAreaCell"],
        invDvEdge=combined["invDvEdge"],
        invDcEdge=combined["invDcEdge"],
        rk_step=None,
        stage_key=("init_refresh",),
    )
    plan = store.get_or_build_numeric_plan(
        "diagnostic_solve_init_numeric", plan.key, lambda: plan
    )
    mode, backend = compile_mode(config), compile_backend(config)
    return store.get_or_build_numeric_plan(
        "diagnostic_solve_init_numeric_compiled",
        (plan.key, mode, backend),
        lambda: compile_diagnostic_solve_numeric_plan(
            plan, compile_kernel=maybe_compile_kernel, mode=mode, backend=backend
        ),
    )


def _dimension_from_shape(values: Mapping[str, Any], name: str, size: int, *, default: int = 0) -> int:
    declared = values.get(name, default)
    if isinstance(declared, torch.Tensor):
        if declared.shape != () or declared.dtype not in (torch.int32, torch.int64):
            raise MpasConfigError(f"{name} requires scalar integer metadata")
        _require_tensor_predicate(
            declared == size, f"{name} differs from tensor shape"
        )
        return size
    return int(declared)


def _init_solve_diagnostics(
    state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict | None = None,
    dt: float | None = None,
    strict: bool = True,
) -> None:
    """Compute solve_diagnostics fields from prognostic state.

    Upstream MPAS calls atm_compute_solve_diagnostics during initialization
    to compute v (tangential velocity), vorticity, divergence, ke, and pv_edge
    from the initial state.

    Always recomputes instead of retaining externally supplied diagnostics.

    Args:
        state: Model state dictionary (modified in place).
        mesh: Mesh connectivity and geometry dictionary.
        strict: If True, raise ValueError when required inputs are missing.
    """

    from mpas_pytorch.dynamics.solve_diagnostics import (
        _diagnostic_solve_connectivity_payload,
        _diagnostic_solve_connectivity_payload_cache_key,
        _mark_diagnostic_solve_diagnostics_current,
        atm_compute_solve_diagnostics_work,
    )

    # Merge state and mesh for lookups
    combined = {}
    combined.update(mesh)
    combined.update(state)

    # Required inputs for solve_diagnostics
    required = [
        "u",
        "rho_zz",
        "cellsOnEdge",
        "edgesOnCell",
        "verticesOnEdge",
        "edgesOnVertex",
        "nEdgesOnCell",
        "nEdgesOnEdge",
        "edgesOnEdge",
        "verticesOnCell",
        "dvEdge",
        "dcEdge",
        "invDvEdge",
        "invDcEdge",
        "invAreaCell",
        "invAreaTriangle",
        "fVertex",
        "fEdge",
        "weightsOnEdge",
        "kiteAreasOnVertex",
        "kiteForCell",
        "edgesOnCell_sign",
        "edgesOnVertex_sign",
    ]

    nCells = _dimension_int(combined.get("nCells"), 0)
    nEdges = _dimension_int(combined.get("nEdges"), 0)
    wind = combined.get("u")
    nVertLevels = _dimension_from_shape(combined, "nVertLevels",
        wind.shape[1] if isinstance(wind, torch.Tensor) and wind.ndim == 2 else 0)

    if nCells == 0 or nEdges == 0 or nVertLevels == 0:
        if strict:
            raise MpasConfigError(
                "Cannot compute solve_diagnostics: nCells/nEdges/nVertLevels missing."
            )
        return

    missing = [k for k in required if k not in combined]
    if missing:
        if strict:
            raise MpasConfigError(
                f"Cannot compute solve_diagnostics: missing inputs: {missing}. "
                f"Provide these from checkpoint or set strict=False to skip."
            )
        return

    # Extract dimensions
    nVertices = _dimension_from_shape(combined, "nVertices", combined["areaTriangle"].shape[0] - 1)
    vertexDegree = _dimension_from_shape(combined, "vertexDegree", combined["edgesOnVertex"].shape[1], default=3)
    state["nVertices"], state["vertexDegree"] = nVertices, vertexDegree

    # Call solve_diagnostics kernel.
    # Use runtime dt/apvm defaults to match Fortran initialization behavior.
    # h = rho_zz (Jacobian-weighted density used as mass proxy)
    cfg = config or {}
    dt_diag = float(
        dt
        if dt is not None
        else cfg.get(
            "config_dt", cfg.get("dt", FORTRAN_DEFAULTS.get("config_dt", 720.0))
        )
    )
    apvm_diag = float(
        cfg.get(
            "config_apvm_upwinding",
            cfg.get(
                "apvm_upwinding", FORTRAN_DEFAULTS.get("config_apvm_upwinding", 0.5)
            ),
        )
    )
    # Static topology payload: built once and reused while the mesh tensors are
    # unchanged (version-keyed), instead of being rebuilt with host reads on
    # every forward inter-step rematerialization.
    connectivity_payload = None
    if isinstance(config, dict):
        payload_kwargs = {
            "device": combined["u"].device,
            "nCells": nCells,
            "nEdges": nEdges,
            "nVertices": nVertices,
            "vertexDegree": vertexDegree,
            "cellStart": 1,
            "cellEnd": nCells,
            "vertexStart": 1,
            "vertexEnd": nVertices,
            "edgeStart": 1,
            "edgeEnd": nEdges,
            **{
                name: combined[name]
                for name in (
                    "dvEdge",
                    "dcEdge",
                    "weightsOnEdge",
                    "kiteAreasOnVertex",
                    "cellsOnEdge",
                    "verticesOnEdge",
                    "edgesOnCell",
                    "edgesOnCell_sign",
                    "edgesOnVertex",
                    "edgesOnVertex_sign",
                    "verticesOnCell",
                    "kiteForCell",
                    "nEdgesOnCell",
                    "nEdgesOnEdge",
                    "edgesOnEdge",
                )
            },
        }
        connectivity_payload = _time_integration._prebound_static_payload(
            config,
            "diagnostic_solve_init",
            _diagnostic_solve_connectivity_payload_cache_key,
            _diagnostic_solve_connectivity_payload,
            payload_kwargs,
        )
    numeric_plan = _init_solve_diagnostics_compiled_plan(
        config,
        connectivity_payload,
        combined,
        nCells=nCells,
        nEdges=nEdges,
        nVertices=nVertices,
        nVertLevels=nVertLevels,
        dt=dt_diag,
        config_apvm_upwinding=apvm_diag,
    )
    diag_result = atm_compute_solve_diagnostics_work(
        nCells=nCells,
        nEdges=nEdges,
        nVertices=nVertices,
        nVertLevels=nVertLevels,
        vertexDegree=vertexDegree,
        _connectivity_payload=connectivity_payload,
        _numeric_plan=numeric_plan,
        dt=dt_diag,
        config_apvm_upwinding=apvm_diag,
        cellStart=1,
        cellEnd=nCells,
        vertexStart=1,
        vertexEnd=nVertices,
        edgeStart=1,
        edgeEnd=nEdges,
        fVertex=combined["fVertex"],
        fEdge=combined["fEdge"],
        invAreaTriangle=combined["invAreaTriangle"],
        invAreaCell=combined["invAreaCell"],
        dvEdge=combined["dvEdge"],
        dcEdge=combined["dcEdge"],
        invDvEdge=combined["invDvEdge"],
        invDcEdge=combined["invDcEdge"],
        weightsOnEdge=combined["weightsOnEdge"],
        kiteAreasOnVertex=combined["kiteAreasOnVertex"],
        cellsOnEdge=combined["cellsOnEdge"],
        verticesOnEdge=combined["verticesOnEdge"],
        edgesOnCell=combined["edgesOnCell"],
        edgesOnCell_sign=combined["edgesOnCell_sign"],
        edgesOnVertex=combined["edgesOnVertex"],
        edgesOnVertex_sign=combined["edgesOnVertex_sign"],
        verticesOnCell=combined["verticesOnCell"],
        kiteForCell=combined["kiteForCell"],
        nEdgesOnCell=combined["nEdgesOnCell"],
        nEdgesOnEdge=combined["nEdgesOnEdge"],
        edgesOnEdge=combined["edgesOnEdge"],
        u=combined["u"],
        h=combined["rho_zz"],
    )

    # Store computed fields in state.
    # Refresh diagnostics from the current state and replace stale file values.
    for key in ["v", "vorticity", "divergence", "ke", "pv_edge", "h_edge"]:
        if key in diag_result:
            state[key] = diag_result[key]
    # h_edge is rho_edge in the state convention
    if "h_edge" in diag_result:
        state["rho_edge"] = diag_result["h_edge"]
    _mark_diagnostic_solve_diagnostics_current(
        state, u=combined["u"], h=combined["rho_zz"]
    )


def _init_vector_reconstruction(
    state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict | None = None,
    strict: bool = True,
) -> None:
    """Compute ur_cell/vr_cell from the initial edge-normal wind.

    Always recomputes — no existence guard. Prevents Fortran value injection.
    """

    combined = {}
    combined.update(mesh)
    combined.update(state)
    required = [
        "u",
        "coeffs_reconstruct",
        "edgesOnCell",
        "nEdgesOnCell",
        "latCell",
        "lonCell",
        "nCells",
        "nEdges",
        "nVertLevels",
    ]
    missing = [k for k in required if k not in combined]
    if missing:
        if strict:
            raise MpasConfigError(
                f"Cannot compute initial vector reconstruction: missing inputs: {missing}"
            )
        return

    nCells = _dimension_int(combined["nCells"], 0)
    nEdges = _dimension_int(combined["nEdges"], 0)
    nVertLevels = _dimension_from_shape(combined, "nVertLevels", combined["u"].shape[1])
    if nCells <= 0 or nEdges <= 0 or nVertLevels <= 0:
        if strict:
            raise MpasConfigError(
                "Cannot compute initial vector reconstruction: invalid dimensions "
                f"(nCells={nCells}, nEdges={nEdges}, nVertLevels={nVertLevels})"
            )
        return

    on_a_sphere = True
    if isinstance(config, Mapping):
        on_a_sphere = bool(config.get("config_on_a_sphere", True))

    coeffs_reconstruct = combined["coeffs_reconstruct"]
    coeffs_all_zero = coeffs_reconstruct_all_zero(state, coeffs_reconstruct, nCells)
    if coeffs_all_zero and "ur_cell" in combined and "vr_cell" in combined:
        warnings.warn(
            "coeffs_reconstruct is all zeros; preserving existing ur_cell/vr_cell "
            "for uReconstruct diagnostics.",
            RuntimeWarning,
        )
        state["ur_cell"] = combined["ur_cell"]
        state["vr_cell"] = combined["vr_cell"]
        for key in ("uReconstructX", "uReconstructY", "uReconstructZ"):
            if key in combined:
                state[key] = combined[key]
        state["uReconstructZonal"] = combined.get(
            "uReconstructZonal", combined["ur_cell"].clone()
        )
        state["uReconstructMeridional"] = combined.get(
            "uReconstructMeridional", combined["vr_cell"].clone()
        )
        return

    vector_connectivity_payload = None
    if isinstance(config, Mapping):
        vector_payload_kwargs = {
            "device": combined["u"].device,
            "nCells": nCells,
            "nEdges": nEdges,
            "nVertLevels": nVertLevels,
            "u_rows": int(combined["u"].shape[0]),
            "coeffs_reconstruct": coeffs_reconstruct,
            "edgesOnCell": combined["edgesOnCell"],
            "nEdgesOnCell": combined["nEdgesOnCell"],
            "latCell": combined["latCell"],
            "lonCell": combined["lonCell"],
            "on_a_sphere": on_a_sphere,
            "cell_padding_offset": combined.get(CELL_PADDING_OFFSET_KEY),
        }
        vector_key = _vector_reconstruction_payload_cache_key(**vector_payload_kwargs)
        if vector_key is not None:
            store = config.get("_dycore_static_payloads")
            if isinstance(store, DycoreStaticPayloads):
                vector_connectivity_payload = store.get_or_build_typed(
                    "vector_reconstruction",
                    vector_key,
                    lambda: _vector_reconstruction_payload(**vector_payload_kwargs),
                )
            elif isinstance(config, dict):
                vector_connectivity_payload = ensure_static_payload_store(
                    config
                ).get_or_build_typed(
                    "vector_reconstruction",
                    vector_key,
                    lambda: _vector_reconstruction_payload(**vector_payload_kwargs),
                )

    vector_numeric = (
        forward_compiled_region(
            "vector_reconstruction_region",
            _vector_reconstruction_numeric,
            config,
            combined["u"].device,
        )
        if vector_connectivity_payload is not None
        else None
    )
    recon = mpas_reconstruct_2d(
        nCells=nCells,
        nEdges=nEdges,
        nVertLevels=nVertLevels,
        u=combined["u"],
        coeffs_reconstruct=coeffs_reconstruct,
        edgesOnCell=combined["edgesOnCell"],
        nEdgesOnCell=combined["nEdgesOnCell"],
        latCell=combined["latCell"],
        lonCell=combined["lonCell"],
        on_a_sphere=on_a_sphere,
        cell_padding_offset=combined.get(CELL_PADDING_OFFSET_KEY),
        _connectivity_payload=vector_connectivity_payload,
        _numeric=vector_numeric,
    )
    state["ur_cell"] = recon["ur_cell"]
    state["vr_cell"] = recon["vr_cell"]
    state["uReconstructX"] = recon["uReconstructX"]
    state["uReconstructY"] = recon["uReconstructY"]
    state["uReconstructZ"] = recon["uReconstructZ"]
    state["uReconstructZonal"] = recon["uReconstructZonal"]
    state["uReconstructMeridional"] = recon["uReconstructMeridional"]

    for name in ("ur_cell", "vr_cell"):
        _require_tensor_predicate(
            torch.isfinite(state[name][:nCells, :]).all(),
            f"Initial vector reconstruction produced non-finite {name} values.",
        )


_NATIVE_FORCING_KEYS = STEP_FORCING_KEYS
_NATIVE_PARTITION_DIMENSIONS = (
    ("cell", "nCells"), ("edge", "nEdges"), ("vertex", "nVertices")
)
_NATIVE_PARTITION_KEYS = frozenset(
    key
    for kind, dimension in _NATIVE_PARTITION_DIMENSIONS
    for key in (
        f"{dimension}Solve", f"{dimension}Global", f"{kind}Start", f"{kind}End",
        f"{kind}SolveStart", f"{kind}SolveEnd",
    )
)
_NATIVE_CONTEXT_KEYS = FORWARD_INTERSTEP_STATIC_KEYS | frozenset(
    {
        COEFFS_RECONSTRUCT_ALL_ZERO_KEY,
        _EXACT_CHECKPOINT_DIAGNOSTIC_MARKER,
        "c_s",
        "coef_3rd_order",
        "rtheta_base",
        "exner_base",
        "pressure_base",
        "nCells",
        "nEdges",
        "nVertices",
        "nVertLevels",
        "maxEdges",
        "maxEdges2",
        "vertexDegree",
        "num_scalars",
        "moist_start",
        "moist_end",
        "nRelaxZone",
    }
) | _NATIVE_PARTITION_KEYS


@dataclass(frozen=True)
class _NativeStepContext:
    support: dict[str, Any]
    recovery_topology: dict[str, Any]


def _native_step_context(
    prepared: Mapping[str, Any], mesh: Mapping[str, Any]
) -> _NativeStepContext:
    """Bind fixed support only; never capture a previous step's workspace or winds."""
    combined = {**mesh, **prepared}
    support = {
        key: value for key, value in combined.items() if key in _NATIVE_CONTEXT_KEYS
    }
    if support.get(COEFFS_RECONSTRUCT_ALL_ZERO_KEY) is True:
        raise MpasConfigError(
            "native step closure requires nonzero wind reconstruction coefficients"
        )
    shape_bindings = {
        "nCells": ("rho_p", 0, -1), "nEdges": ("ru", 0, -1),
        "nVertLevels": ("ru", 1, 0), "nVertices": ("areaTriangle", 0, -1),
        "maxEdges": ("edgesOnCell", 1, 0), "maxEdges2": ("edgesOnEdge", 1, 0),
        "vertexDegree": ("edgesOnVertex", 1, 0), "num_scalars": ("scalars", 2, 0),
    }
    for key in (
        "nCells",
        "nEdges",
        "nVertices",
        "nVertLevels",
        "maxEdges",
        "maxEdges2",
        "vertexDegree",
        "num_scalars",
        "moist_start",
        "moist_end",
        "nRelaxZone",
    ):
        if key in support:
            if key in shape_bindings:
                field, axis, offset = shape_bindings[key]
                support[key] = _dimension_from_shape(
                    support, key, combined[field].shape[axis] + offset
                )
            else:
                support[key] = _dimension_int(support[key], 0)
    nc, ne = support["nCells"], support["nEdges"]
    topology = _large_step_recovery_connectivity_payload(
        device=prepared["ru"].device,
        nCells=nc,
        nEdges=ne,
        maxEdges=support["maxEdges"],
        nRelaxZone=support.get("nRelaxZone", 0),
        cell_start=0,
        cell_end=nc,
        edge_start=0,
        edge_end=ne,
        **{
            key: combined[key]
            for key in (
                "cellsOnEdge",
                "edgesOnCell",
                "nEdgesOnCell",
                "edgesOnCell_sign",
                "bdyMaskCell",
            )
        },
    )
    return _NativeStepContext(support, topology)


def _materialize_native_step(
    state: Mapping[str, torch.Tensor],
    context: _NativeStepContext,
    mesh: dict[str, Any],
    config: dict[str, Any],
    dt: float,
    forcing: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Build a warm step from exactly five native producer fields and explicit forcing."""
    if set(state) != set(NATIVE_STATE_KEYS):
        raise MpasConfigError(f"native state must contain exactly {NATIVE_STATE_KEYS}")
    if forcing is not None and set(forcing) - _NATIVE_FORCING_KEYS:
        raise MpasConfigError("native step forcing contains non-forcing fields")
    if set(context.support) - _NATIVE_CONTEXT_KEYS:
        raise MpasConfigError("native context contains non-static fields")
    if config.get("config_apply_lbcs", False):
        raise MpasConfigError(
            "native step closure is not yet qualified for LBCs"
        )
    constants = constants_for_config(config)
    support = context.support
    exchanger = config.get("_halo_exchanger")
    if exchanger is not None:
        from mpas_pytorch.distributed.dynamics import (
            DynamicsHaloExchange,
            exchange_halo_group,
        )

        if (not isinstance(exchanger, DynamicsHaloExchange)
                or state["ru"].device.type != "cpu"):
            raise MpasConfigError("native step closure is not yet qualified for this halo policy")
        for kind, dimension in _NATIVE_PARTITION_DIMENSIONS:
            entity = exchanger.partition.entity(kind)
            required = {
                dimension: entity.local_count,
                f"{dimension}Solve": entity.owned_count,
                f"{dimension}Global": entity.global_count,
                f"{kind}Start": 1,
                f"{kind}End": entity.local_count,
                f"{kind}SolveStart": 1,
                f"{kind}SolveEnd": entity.owned_count,
            }
            if any(type(support.get(key)) is not int or support[key] != value
                   for key, value in required.items()):
                raise MpasConfigError("native partition context disagrees with static ownership bounds")
        state = dict(state)
        exchange_halo_group(config, "native_step:prognostics", state, caller_owned=True)
    decoded = profiled_call(
        "mpas.native_state_decode",
        config,
        decode_native_fields,
        **{key: state[key] for key in NATIVE_STATE_KEYS[:-1]},
        **{
            key: support[key]
            for key in (
                "rho_base",
                "rtheta_base",
                "exner_base",
                "zz",
                "fzm",
                "fzp",
                "zb_cell",
                "zb3_cell",
            )
        },
        **{key: support[key] for key in ("cf1", "cf2", "cf3")},
        connectivity=context.recovery_topology,
        rgas=constants.RD,
        cp=constants.CP,
        p0=constants.P0,
    )
    result = {**support, **state, **decoded, **(forcing or {})}
    if exchanger is not None:
        # Outer edge stencils omit donor cells. Their owners decode the winds
        # before the solve diagnostics consume those halo values.
        exchange_halo_group(config, "native_step:winds", result, caller_owned=True)
    profiled_call(
        "mpas.init_solve_diagnostics",
        config,
        _init_solve_diagnostics,
        result,
        mesh,
        config=config,
        dt=dt,
        strict=True,
    )
    profiled_call(
        "mpas.init_vector_reconstruction",
        config,
        _init_vector_reconstruction,
        result,
        mesh,
        config=config,
        strict=True,
    )
    return result


def _native_output_fields(
    state: Mapping[str, torch.Tensor],
    context: _NativeStepContext,
    mesh: dict[str, Any],
    config: dict[str, Any],
    dt: float,
    fields: Iterable[str],
) -> dict[str, torch.Tensor]:
    """Project requested current-state output without sharing writable state buffers.

    Step-history work, such as ruAvg and kdiff, requires the step's output
    record. A final native snapshot alone cannot recover that history.
    """
    from mpas_pytorch.io_schema import HISTORY_FIELD_METADATA

    requested = tuple(fields)
    supported = (
        set(HISTORY_FIELD_METADATA)
        | set(NATIVE_STATE_KEYS)
        | set(INTER_STEP_EXACT_DIAGNOSTIC_KEYS)
        | {"rho_zz", "theta_m"}
    )
    if set(requested) - supported:
        raise MpasConfigError(
            "native output contains unsupported or step-history fields"
        )
    if not requested:
        return {}
    output_config = dict(config)
    output = _materialize_native_step(state, context, mesh, output_config, dt)
    refresh_output_diagnostics(output, output_config)
    output["qv"] = output["scalars"][..., 0]
    return {name: output[name].clone() for name in requested}


def _native_timestep(
    state: Mapping[str, torch.Tensor],
    context: _NativeStepContext,
    mesh: dict[str, Any],
    config: dict[str, Any],
    dt: float,
    *,
    forcing: Mapping[str, torch.Tensor] | None = None,
    differentiable: bool = False,
    initial: bool = False,
    experimental: bool = False,
) -> dict[str, torch.Tensor]:
    """Advance five fields through the maintained RK implementation.

    Cold input uses u/w/rho_zz/theta_m/scalars. Integrated continuation uses
    the five native producers; cold values must not undergo an inverse round trip.
    """
    with torch.set_grad_enabled(differentiable):
        if initial:
            if set(state) != {"u", "w", "rho_zz", "theta_m", "scalars"}:
                raise MpasConfigError(
                    "initial native step requires five physical input fields"
                )
            if forcing is not None and set(forcing) - _NATIVE_FORCING_KEYS:
                raise MpasConfigError("native step forcing contains non-forcing fields")
            if set(context.support) - _NATIVE_CONTEXT_KEYS:
                raise MpasConfigError("native context contains non-static fields")
            if (
                config.get("config_apply_lbcs", False)
                or config.get("_halo_exchanger") is not None
            ):
                raise MpasConfigError(
                    "native step closure is not yet qualified for LBCs or partitions"
                )
            entry = _materialize_step_state(
                {**context.support, **state, **(forcing or {})},
                mesh,
                config,
                dt,
                strict=True,
            )
        else:
            entry = _materialize_native_step(state, context, mesh, config, dt, forcing)
        advanced = atm_srk3(
            entry,
            mesh,
            {**config, "_native_boundary_output": True},
            dt,
            strict=True,
            differentiable=differentiable,
            experimental=experimental,
        )
        _require_finite_model_fields(
            advanced, "native timestep result", fields=NATIVE_STATE_KEYS, config=config
        )
        return {key: advanced[key] for key in NATIVE_STATE_KEYS}


def run_simulation(
    initial_state: dict[str, Any],
    mesh: dict[str, Any],
    config: dict | None = None,
    dt: float | None = None,
    nsteps: int = 1,
    device: torch.device | str = _CPU_DEVICE,
    on_step: Callable[[int, dict[str, Any]], None] | None = None,
    differentiable: bool = False,
    checkpoint_timesteps: bool | None = None,
    checkpoint_use_reentrant: bool = False,
    checkpoint_materialize_final: bool = True,
    on_phase: Callable[[str], None] | None = None,
    on_initial: Callable[[dict[str, Any]], None] | None = None,
    experimental: bool = False,
) -> dict[str, Any]:
    """Run simulation for specified number of timesteps.

    Args:
        initial_state: Initial model state dictionary.
        mesh: Mesh connectivity and geometry dictionary.
        config: Optional configuration overrides. If None or partial,
            FORTRAN_DEFAULTS values are used for missing keys.
            The 'strict' key controls validation behavior (default True).
        dt: Timestep size in seconds.
        nsteps: Number of timesteps to run.
        device: Target device for state/mesh tensors (default CPU).
        on_initial: Observe the prepared state used by timestep one; must not mutate it.
        on_step: Optional callback invoked after each completed timestep as
            ``on_step(step_number, state_dict)``.
        differentiable: Enable autodiff-compatible execution.
        checkpoint_timesteps: Whether to recompute each full timestep during
            backward. ``None`` enables the current default policy: auto-enable
            for differentiable runs, including one-step TL/AD probes, so memory
            is bounded by recomputation unless callers explicitly opt out.
            Pass ``False`` to disable timestep checkpointing for exactness-
            sensitive validation runs.
        checkpoint_use_reentrant: Use reentrant outer timestep checkpoints.
            Non-reentrant checkpoints support both ``torch.autograd.grad`` and
            ``.backward()`` and should remain the default for nearly all
            workflows. Reentrant checkpoints are kept only as an opt-in
            compatibility path for experiments that explicitly want them.
        checkpoint_materialize_final: Recompute the full final diagnostic state
            before returning. Disable this for prognostic-only adjoint
            objectives to avoid rebuilding a large final diagnostic graph
            unnecessarily. This controls returned output fields, not the
            five-field continuation used by qualified global forecasts.
        on_phase: Optional observation callback. Production benchmark runners
            use ``forecast_integration_begin``/``forecast_integration_end`` to
            time the same post-initialization forecast interval as native MPAS.

    Returns:
        Final model state dictionary after all timesteps.
    """
    _validate_public_inputs(
        initial_state, mesh, device, experimental=experimental, config=config
    )
    config = validate_config(config)
    if dt is None:
        dt = float((config or {}).get("config_dt", FORTRAN_DEFAULTS["config_dt"]))
    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps < 0:
        raise MpasConfigError("nsteps must be a nonnegative integer")
    if not math.isfinite(float(dt)) or dt <= 0.0:
        raise MpasConfigError("dt must be finite and positive")
    config_validated = True
    mesh = dict(mesh)
    boundary_mesh = dict(mesh)
    checkpoint_plan = build_checkpoint_plan(
        differentiable=differentiable,
        checkpoint_timesteps=checkpoint_timesteps,
        checkpoint_use_reentrant=checkpoint_use_reentrant,
        checkpoint_materialize_final=checkpoint_materialize_final,
        config=config,
    )
    checkpoint_timesteps = checkpoint_plan.outer_timestep
    checkpoint_use_reentrant = checkpoint_plan.outer_use_reentrant
    checkpoint_materialize_final = checkpoint_plan.materialize_final

    def _simulate() -> dict[str, Any]:
        state = dict(initial_state)
        full_config = (
            dict(config or {}) if config_validated else validate_config(config)
        )
        prune_forward_intersteps = (
            bool(full_config.get("config_prune_forward_interstep_state", False))
            and not differentiable
        )
        boundary_static_support: dict[str, Any] = {}
        native_context: _NativeStepContext | None = None
        native_forcing: dict[str, torch.Tensor] = {}
        if differentiable:
            full_config["gradient_checkpointing"] = checkpoint_plan.nested_rk
        required_init_dims = {"nCells", "nEdges", "nVertLevels"}
        can_prepare_once = required_init_dims.issubset(
            mesh.keys()
        ) or required_init_dims.issubset(state.keys())
        forecast_integration_started = False
        previous_state: dict[str, Any] = {}
        step_state: dict[str, Any] = {}
        # Execution policy: replay steady pruned forward steps as one CUDA
        # graph per process (halo messages included). The first steps and the
        # final full-state step run ordinarily.
        graph_requested = bool(full_config.get("config_cuda_graph_timestep", False))
        if graph_requested:
            _require_timestep_graph_policy(
                full_config,
                device=device,
                differentiable=differentiable,
                prune_forward_intersteps=prune_forward_intersteps,
                prepared_once=can_prepare_once,
                on_step=on_step,
            )
        timestep_graph: TimestepGraph | None = None
        # Layout and host values of the last ordinary step's entry; no tensors.
        graph_signature: BoundarySignature | None = None

        def _native_forecast_step(
            inputs: dict[str, Any], *, prepared_entry: bool, step_number: int
        ) -> dict[str, Any]:
            """Keep output history outside the five-field continuation boundary."""
            assert native_context is not None
            full_output = on_step is not None or (
                step_number == nsteps
                and (
                    not differentiable
                    or not checkpoint_timesteps
                    or checkpoint_materialize_final
                )
            )
            step_config = {**full_config, "_native_boundary_output": not full_output}
            fixed_tensor_ids = {
                id(value)
                for value in (*mesh.values(), *native_context.support.values())
                if isinstance(value, torch.Tensor)
            }
            static_outputs: dict[str, Any] = {}

            def advance(sd: dict[str, Any]) -> dict[str, Any]:
                entry = (
                    dict(sd)
                    if prepared_entry
                    else _materialize_native_step(
                        {key: sd[key] for key in NATIVE_STATE_KEYS},
                        native_context,
                        mesh,
                        step_config,
                        dt,
                        {key: sd[key] for key in STEP_FORCING_KEYS if key in sd},
                    )
                )
                result = atm_srk3(
                    entry,
                    mesh,
                    step_config,
                    dt,
                    strict=strict,
                    differentiable=(
                        torch.is_grad_enabled()
                        if differentiable and checkpoint_use_reentrant
                        else differentiable
                    ),
                    **({"experimental": True} if experimental else {}),
                )
                if full_output and checkpoint_use_reentrant:
                    # The no-grad solver stamps density padding in place;
                    # replay builds it out of place. Own the published density
                    # and bind its receipt to that buffer in both executions.
                    view = diagnostic_solve_checkpoint_metadata(result)
                    density = result["rho_zz"]
                    owned_density = density.clone()
                    for key in ("rho_zz", "h"):
                        if result.get(key) is density:
                            result[key] = owned_density
                    _rebind_diagnostic_solve_receipt(result, view)
                # Reentrant checkpoints attach gradient outputs even to fixed
                # floating-point tensors returned by identity. Keep topology
                # outside that boundary; restore output views after replay.
                boundary = {}
                for key, value in result.items():
                    if key in native_context.support or id(value) in fixed_tensor_ids:
                        static_outputs[key] = value
                    else:
                        boundary[key] = value
                return boundary

            advanced = checkpoint_state_dict(
                advance,
                inputs if prepared_entry else {**inputs, **native_forcing},
                enabled=differentiable and checkpoint_timesteps,
                use_reentrant=checkpoint_use_reentrant,
                context_fn=_checkpoint_context_for_policy(step_config),
                early_stop=False if step_config.get("_halo_exchanger") else None,
                non_tensor_signature=diagnostic_solve_checkpoint_metadata,
            )
            advanced = _normalize_checkpointed_state(
                advanced, between_steps=step_number != nsteps
            )
            if full_output:
                advanced = {**static_outputs, **advanced}
            _require_finite_model_fields(
                advanced,
                "native forecast result",
                fields=NATIVE_STATE_KEYS,
                config=step_config,
            )
            continuation = {key: advanced[key] for key in NATIVE_STATE_KEYS}
            if on_step is not None:
                on_step(step_number, own_cuda_graph_outputs(advanced))
            _log_timestep_rss(step_number, "after")
            if step_number != nsteps:
                return continuation
            if full_output:
                return advanced
            # Preserve the public compact-output contract. These diagnostic
            # views are final consumers, never inputs to another model step.
            output = _materialize_native_step(
                continuation, native_context, mesh, full_config, dt, native_forcing
            )
            keys = _INTER_STEP_PROGNOSTIC_KEYS | _INTER_STEP_EXACT_DIAGNOSTIC_KEYS
            return {
                key: value
                for key, value in output.items()
                if not isinstance(value, torch.Tensor) or key in keys
            }

        for step in range(nsteps):
            if step > 0 and not differentiable:
                # No borrowed prior-step output may remain live when the next
                # generation executes, including callback and preparation state.
                previous_state = {}
                step_state = {}
                state = cuda_graph_step_boundary(state)
            step_number = step + 1
            _log_timestep_rss(step_number, "before")
            resume_from_restart = (
                timestep_boundary_predicate(state) if step == 0 else None
            )
            want_final_full_state = (
                checkpoint_timesteps
                and differentiable
                and checkpoint_materialize_final
                and step_number == nsteps
            )
            if (
                step > 0
                and differentiable
                and checkpoint_timesteps
                and not checkpoint_use_reentrant
                and boundary_static_support
                and native_context is None
            ):
                profiled_call(
                    "mpas.static_boundary_restore",
                    full_config,
                    _restore_static_boundary_support,
                    state,
                    boundary_static_support,
                )
            if can_prepare_once and step == 0:
                state, full_config, strict = _prepare_timestep_state(
                    state=state,
                    mesh=mesh,
                    config=full_config,
                    dt=dt,
                    device=device,
                    config_validated=True,
                )
                if resume_from_restart is not None and differentiable:
                    resume_from_restart = resume_from_restart.to(state["u"].device)
                    cold_or_carried = dict(state)
                    _rematerialize_interstep_diagnostics(
                        state,
                        mesh,
                        full_config,
                        dt,
                        strict=strict,
                    )
                    _retain_boundary_diagnostics_when(
                        state, cold_or_carried, ~resume_from_restart
                    )
                # Ordinary forward continuation consumes the complete boundary
                # snapshot directly. Re-deriving rho_p/rtheta_p/pressure_p here
                # was measured to change the next timestep by roundoff.
                # Own preparation outputs before starting the recurring timestep
                # root, independent of the preparation graph's recording phase.
                state = cuda_graph_step_boundary(state)
                if on_initial is not None:
                    on_initial(state)
                if (
                    set(NATIVE_STATE_KEYS).issubset(state)
                    and not full_config.get("config_apply_lbcs", False)
                    and (full_config.get("_halo_exchanger") is None
                         or state["ru"].device.type == "cpu")
                    and not graph_requested
                ):
                    native_context = _native_step_context(state, mesh)
                    if prune_forward_intersteps and nsteps > 1:
                        # Preserve the carried path's fixed ownership metadata,
                        # without retaining its diagnostic tensors between steps.
                        native_context.support.setdefault(
                            _EXACT_CHECKPOINT_DIAGNOSTIC_MARKER,
                            _checkpoint_exact_diagnostic_keys(state),
                        )
                    native_forcing = {
                        key: state[key] for key in STEP_FORCING_KEYS if key in state
                    }
                if (
                    differentiable
                    and checkpoint_timesteps
                    and not checkpoint_use_reentrant
                    and native_context is None
                ):
                    # Only checkpointed reverse mode rematerializes from this
                    # support; elsewhere it kept step-0 diagnostics alive.
                    boundary_static_support = profiled_call(
                        "mpas.static_boundary_capture",
                        full_config,
                        _capture_static_boundary_support,
                        state,
                        mesh,
                    )
                if on_phase is not None:
                    on_phase("forecast_integration_begin")
                    forecast_integration_started = True
                if native_context is not None:
                    state = _native_forecast_step(
                        state, prepared_entry=True, step_number=step_number
                    )
                    continue
                previous_state = state
                if differentiable:
                    if checkpoint_timesteps:

                        def _one_step_materialized(
                            sd: dict[str, Any],
                            full_config_local: dict[str, Any] = full_config,
                            strict_local: bool = strict,
                            want_final_full_state_local: bool = want_final_full_state,
                        ) -> dict[str, Any]:
                            step_result = atm_srk3(
                                sd,
                                mesh,
                                full_config_local,
                                dt,
                                strict=strict_local,
                                **({"experimental": True} if experimental else {}),
                                differentiable=(
                                    torch.is_grad_enabled()
                                    if checkpoint_use_reentrant
                                    else True
                                ),
                            )
                            if want_final_full_state_local:
                                return step_result
                            if checkpoint_use_reentrant:
                                return _project_checkpoint_boundary_state(
                                    sd, step_result
                                )
                            return _exact_checkpoint_boundary(sd, step_result)

                        state = checkpoint_state_dict(
                            _one_step_materialized,
                            state,
                            enabled=True,
                            use_reentrant=checkpoint_use_reentrant,
                            context_fn=_checkpoint_context_for_policy(full_config),
                            early_stop=False
                            if full_config.get("_halo_exchanger")
                            else None,
                            non_tensor_signature=_checkpoint_non_tensor_signature_for_policy(
                                full_config
                            ),
                        )
                        state = _normalize_checkpointed_state(
                            state,
                            between_steps=not want_final_full_state,
                        )
                        if checkpoint_use_reentrant:
                            state = profiled_call(
                                "mpas.restore_padding_sentinels",
                                full_config,
                                _restore_padding_sentinels,
                                state,
                            )
                        # Exact non-reentrant outputs already carry the fixed
                        # density padding. Re-stamping adds a gradient node and
                        # changes coupled reverse accumulation unnecessarily.
                    else:
                        state = atm_srk3(
                            state,
                            mesh,
                            full_config,
                            dt,
                            strict=strict,
                            differentiable=True,
                            **({"experimental": True} if experimental else {}),
                        )
                else:
                    state = atm_srk3(
                        state,
                        mesh,
                        full_config,
                        dt,
                        strict=strict,
                        **({"experimental": True} if experimental else {}),
                    )
                step_state = state
                need_materialized_step = (
                    checkpoint_timesteps
                    and differentiable
                    and not want_final_full_state
                    and (
                        on_step is not None
                        or (step + 1 == nsteps and checkpoint_materialize_final)
                    )
                )
                if need_materialized_step:
                    materialize_state = profiled_call(
                        "mpas.materialization_input_prepare",
                        full_config,
                        _prepare_materialization_input,
                        state,
                        checkpoint_timesteps=checkpoint_timesteps,
                        differentiable=differentiable,
                        checkpoint_use_reentrant=checkpoint_use_reentrant,
                        static_support=boundary_static_support,
                    )
                    step_state = _materialize_step_state(
                        materialize_state,
                        mesh,
                        full_config,
                        dt,
                        strict=strict,
                    )
                elif want_final_full_state and (
                    on_step is not None or checkpoint_materialize_final
                ):
                    materialize_state = profiled_call(
                        "mpas.materialization_input_prepare",
                        full_config,
                        _prepare_materialization_input,
                        state,
                        checkpoint_timesteps=checkpoint_timesteps,
                        differentiable=differentiable,
                        checkpoint_use_reentrant=checkpoint_use_reentrant,
                        static_support=boundary_static_support,
                    )
                    step_state = _merge_materialized_state(
                        state,
                        _materialize_step_state(
                            materialize_state,
                            mesh,
                            full_config,
                            dt,
                            strict=strict,
                        ),
                    )
                if on_step is not None:
                    step_state = own_cuda_graph_outputs(step_state)
                    on_step(step_number, step_state)
                if step_number == nsteps and checkpoint_materialize_final:
                    state = step_state
                elif prune_forward_intersteps:
                    state = profiled_call(
                        "mpas.forward_interstep_projection",
                        full_config,
                        _project_forward_interstep_state,
                        previous_state,
                        step_state,
                        step_number=step_number,
                        nsteps=nsteps,
                    )
                _log_timestep_rss(step_number, "after")
                continue
            if can_prepare_once and step > 0:
                strict = bool(full_config.get("strict", True))
                if native_context is not None:
                    state = _native_forecast_step(
                        state, prepared_entry=False, step_number=step_number
                    )
                    continue
                if timestep_graph is not None and step_number == nsteps:
                    # The final full-state step runs ordinarily; free the
                    # graph pool first so both do not hold memory at once.
                    timestep_graph.release()
                    timestep_graph = None
                    torch.cuda.empty_cache()
                if graph_requested and step_number < nsteps:
                    # Recording costs about one step; it needs two replays to pay.
                    if timestep_graph is None and nsteps - step_number >= 2:
                        steady = graph_signature is not None and not signature_changes(
                            graph_signature,
                            boundary_signature(
                                state, diagnostic_solve_checkpoint_metadata
                            ),
                        )
                        record_now = _ranks_agree_to_record(full_config, steady)
                    else:
                        record_now = False
                    if record_now:
                        graph_signature = None

                        def _recorded_step(
                            sd: dict[str, Any],
                            full_config_local: dict[str, Any] = full_config,
                            strict_local: bool = strict,
                            step_local: int = step_number,
                        ) -> dict[str, Any]:
                            # The same calls as the ordinary pruned step below.
                            _checkpoint_exact_diagnostic_keys(sd)
                            result = atm_srk3(
                                sd,
                                mesh,
                                full_config_local,
                                dt,
                                strict=strict_local,
                                **({"experimental": True} if experimental else {}),
                            )
                            return _project_forward_interstep_state(
                                sd, result, step_number=step_local, nsteps=nsteps
                            )

                        timestep_graph = TimestepGraph(
                            _recorded_step,
                            device,
                            protected=initial_state,
                            host_view=diagnostic_solve_checkpoint_metadata,
                            rebind=_rebind_diagnostic_solve_receipt,
                            cache_view=partial(_graph_cache_view, full_config),
                        )
                    if timestep_graph is not None:
                        state = timestep_graph(state)
                        _log_timestep_rss(step_number, "after")
                        continue
                    graph_signature = boundary_signature(
                        state, diagnostic_solve_checkpoint_metadata
                    )
                if differentiable:
                    if checkpoint_timesteps:

                        def _one_step_pruned(
                            sd: dict[str, Any],
                            full_config_local: dict[str, Any] = full_config,
                            strict_local: bool = strict,
                            want_final_full_state_local: bool = want_final_full_state,
                        ) -> dict[str, Any]:
                            # A checkpoint boundary is an exact discrete-model
                            # boundary, not a compact restart. Rebuilding these
                            # diagnostics changed the forward trajectory from
                            # step two onward and differentiated a different map.
                            _checkpoint_exact_diagnostic_keys(sd)
                            working_state = dict(sd)
                            step_result = atm_srk3(
                                working_state,
                                mesh,
                                full_config_local,
                                dt,
                                strict=strict_local,
                                **({"experimental": True} if experimental else {}),
                                differentiable=(
                                    torch.is_grad_enabled()
                                    if checkpoint_use_reentrant
                                    else True
                                ),
                            )
                            if want_final_full_state_local:
                                return step_result
                            if checkpoint_use_reentrant:
                                return _project_checkpoint_boundary_state(
                                    sd, step_result
                                )
                            return _exact_checkpoint_boundary(sd, step_result)

                        state = checkpoint_state_dict(
                            _one_step_pruned,
                            state,
                            enabled=True,
                            use_reentrant=checkpoint_use_reentrant,
                            context_fn=_checkpoint_context_for_policy(full_config),
                            early_stop=False
                            if full_config.get("_halo_exchanger")
                            else None,
                            non_tensor_signature=_checkpoint_non_tensor_signature_for_policy(
                                full_config
                            ),
                        )
                        state = _normalize_checkpointed_state(
                            state,
                            between_steps=not want_final_full_state,
                        )
                        if checkpoint_use_reentrant:
                            state = profiled_call(
                                "mpas.restore_padding_sentinels",
                                full_config,
                                _restore_padding_sentinels,
                                state,
                            )
                    else:
                        # The uncheckpointed differentiable path carries the
                        # complete diagnostic state returned by ``atm_srk3``,
                        # just like the ordinary forward path. Rebuilding the
                        # coupled diagnostics here changes that state before
                        # every later timestep and makes the AD primal a
                        # different program. Non-reentrant checkpoint and
                        # forward-pruned boundaries retain those exact fields.
                        state = atm_srk3(
                            state,
                            mesh,
                            full_config,
                            dt,
                            strict=strict,
                            differentiable=True,
                            **({"experimental": True} if experimental else {}),
                        )
                else:
                    if prune_forward_intersteps:
                        _checkpoint_exact_diagnostic_keys(state)
                    # The ordinary forward path carries the complete diagnostic
                    # state returned by atm_srk3 into the next timestep.  Native
                    # MPAS initializes coupled diagnostics only at cold start;
                    # rebuilding rho_p/rtheta_p here from rho_zz/theta_m adds an
                    # extra floating-point round trip before every later step.
                    # Forward pruning drops transient buffers, not the evolved
                    # diagnostics required by the next timestep.
                    previous_state = state
                    state = atm_srk3(
                        state,
                        mesh,
                        full_config,
                        dt,
                        strict=strict,
                        **({"experimental": True} if experimental else {}),
                    )
                step_state = state
                need_materialized_step = (
                    checkpoint_timesteps
                    and differentiable
                    and not want_final_full_state
                    and (
                        on_step is not None
                        or (step + 1 == nsteps and checkpoint_materialize_final)
                    )
                )
                if need_materialized_step:
                    materialize_state = profiled_call(
                        "mpas.materialization_input_prepare",
                        full_config,
                        _prepare_materialization_input,
                        state,
                        checkpoint_timesteps=checkpoint_timesteps,
                        differentiable=differentiable,
                        checkpoint_use_reentrant=checkpoint_use_reentrant,
                        static_support=boundary_static_support,
                    )
                    step_state = _materialize_step_state(
                        materialize_state,
                        mesh,
                        full_config,
                        dt,
                        strict=strict,
                    )
                elif want_final_full_state and (
                    on_step is not None or checkpoint_materialize_final
                ):
                    materialize_state = profiled_call(
                        "mpas.materialization_input_prepare",
                        full_config,
                        _prepare_materialization_input,
                        state,
                        checkpoint_timesteps=checkpoint_timesteps,
                        differentiable=differentiable,
                        checkpoint_use_reentrant=checkpoint_use_reentrant,
                        static_support=boundary_static_support,
                    )
                    step_state = _merge_materialized_state(
                        state,
                        _materialize_step_state(
                            materialize_state,
                            mesh,
                            full_config,
                            dt,
                            strict=strict,
                        ),
                    )
                if on_step is not None:
                    step_state = own_cuda_graph_outputs(step_state)
                    on_step(step + 1, step_state)
                if step + 1 == nsteps and checkpoint_materialize_final:
                    state = step_state
                elif prune_forward_intersteps:
                    state = profiled_call(
                        "mpas.forward_interstep_projection",
                        full_config,
                        _project_forward_interstep_state,
                        previous_state,
                        step_state,
                        step_number=step + 1,
                        nsteps=nsteps,
                    )
                continue
            if differentiable:
                if checkpoint_timesteps:

                    def _one_step_checkpointed(
                        sd: dict[str, Any],
                        full_config_local: dict[str, Any] = full_config,
                        want_final_full_state_local: bool = want_final_full_state,
                    ) -> dict[str, Any]:
                        step_result = run_timestep(
                            sd,
                            mesh,
                            full_config_local,
                            dt,
                            device=device,
                            experimental=experimental,
                            differentiable=(
                                torch.is_grad_enabled()
                                if checkpoint_use_reentrant
                                else True
                            ),
                        )
                        if want_final_full_state_local:
                            return step_result
                        if checkpoint_use_reentrant:
                            return _project_checkpoint_boundary_state(sd, step_result)
                        return _exact_checkpoint_boundary(sd, step_result)

                    state = checkpoint_state_dict(
                        _one_step_checkpointed,
                        state,
                        enabled=True,
                        use_reentrant=checkpoint_use_reentrant,
                        context_fn=_checkpoint_context_for_policy(full_config),
                        early_stop=False
                        if full_config.get("_halo_exchanger")
                        else None,
                        non_tensor_signature=_checkpoint_non_tensor_signature_for_policy(
                            full_config
                        ),
                    )
                    state = _normalize_checkpointed_state(
                        state,
                        between_steps=not want_final_full_state,
                    )
                    if checkpoint_use_reentrant:
                        state = profiled_call(
                            "mpas.restore_padding_sentinels",
                            full_config,
                            _restore_padding_sentinels,
                            state,
                        )
                else:
                    state = run_timestep(
                        state,
                        mesh,
                        full_config,
                        dt,
                        device=device,
                        differentiable=True,
                        experimental=experimental,
                    )
            else:
                previous_state = state
                state = run_timestep(
                    state,
                    mesh,
                    full_config,
                    dt,
                    device=device,
                    experimental=experimental,
                )
            step_state = state
            strict = bool(full_config.get("strict", True))
            need_materialized_step = (
                checkpoint_timesteps
                and differentiable
                and not want_final_full_state
                and (
                    on_step is not None
                    or (step + 1 == nsteps and checkpoint_materialize_final)
                )
            )
            if need_materialized_step:
                materialize_state = profiled_call(
                    "mpas.materialization_input_prepare",
                    full_config,
                    _prepare_materialization_input,
                    state,
                    checkpoint_timesteps=checkpoint_timesteps,
                    differentiable=differentiable,
                    checkpoint_use_reentrant=checkpoint_use_reentrant,
                    static_support=boundary_static_support,
                )
                step_state = _materialize_step_state(
                    materialize_state,
                    mesh,
                    full_config,
                    dt,
                    strict=strict,
                )
            elif want_final_full_state and (
                on_step is not None or checkpoint_materialize_final
            ):
                materialize_state = profiled_call(
                    "mpas.materialization_input_prepare",
                    full_config,
                    _prepare_materialization_input,
                    state,
                    checkpoint_timesteps=checkpoint_timesteps,
                    differentiable=differentiable,
                    checkpoint_use_reentrant=checkpoint_use_reentrant,
                    static_support=boundary_static_support,
                )
                step_state = _merge_materialized_state(
                    state,
                    _materialize_step_state(
                        materialize_state,
                        mesh,
                        full_config,
                        dt,
                        strict=strict,
                    ),
                )
            if on_step is not None:
                step_state = own_cuda_graph_outputs(step_state)
                on_step(step_number, step_state)
            if step_number == nsteps and checkpoint_materialize_final:
                state = step_state
            elif prune_forward_intersteps:
                state = profiled_call(
                    "mpas.forward_interstep_projection",
                    full_config,
                    _project_forward_interstep_state,
                    previous_state,
                    step_state,
                    step_number=step_number,
                    nsteps=nsteps,
                )
            _log_timestep_rss(step_number, "after")
        if forecast_integration_started and on_phase is not None:
            on_phase("forecast_integration_end")
        _require_finite_model_fields(state, "forecast result")
        complete_result = (
            not differentiable
            or not checkpoint_timesteps
            or checkpoint_materialize_final
        )
        if nsteps > 0 and complete_result:
            # Make the complete returned state reusable by a later call to
            # run_simulation, just like a state loaded from write_restart.
            state = profiled_call(
                "mpas.cuda_graph_boundary_ownership",
                full_config,
                own_cuda_graph_outputs,
                dict(state),
            )
            mark_complete_step_boundary(
                state, mesh=boundary_mesh, config=full_config, dt=dt
            )
        return state

    if differentiable:
        return _simulate()
    with _nondifferentiable_context(config), cuda_graph_forward_scope(config, device):
        return _simulate()


def _ordinary_ad_inputs(state: dict[str, Any]) -> dict[str, Any]:
    """Own inference-only forecast tensors before reusing them in either AD mode."""
    copies: dict[int, torch.Tensor] = {}
    result = dict(state)
    with torch.inference_mode(False):
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_inference():
                if id(value) not in copies:
                    copies[id(value)] = value.clone()
                result[key] = copies[id(value)]
    return result


def _run_simulation_jvp(
    state: dict[str, Any],
    mesh: dict[str, Any],
    perturbation: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Tangent-linear model: compute Jacobian-vector product through run_simulation.

    Args:
        state: Initial model state dictionary.
        mesh: Mesh connectivity and geometry dictionary.
        perturbation: Dict mapping prognostic keys to tangent vectors.
            Only keys present in perturbation receive a tangent;
            other state variables are treated as constants.
        **kwargs: Forwarded to run_simulation (dt, nsteps, config, device, etc.).

    Returns:
        Dict mapping state keys to their output tangent vectors (JVP).
    """
    from torch.autograd import forward_ad

    kwargs.setdefault("differentiable", True)
    _require_finite_model_fields(perturbation, "JVP perturbation", fields=perturbation)
    # A preceding inference forecast can supply inference tensors. Forward AD
    # requires ordinary primals (including versioned numerical-plan inputs).
    # Preserve their bytes/ownership; do not mutate or reinitialize the caller.
    state = _ordinary_ad_inputs(state)
    with forward_ad.dual_level():
        dual_state = dict(state)
        for key, tangent in perturbation.items():
            if not isinstance(tangent, torch.Tensor):
                raise TypeError(
                    f"Perturbation '{key}' must be a tensor, got {type(tangent).__name__}"
                )
            if key not in dual_state:
                raise KeyError(f"Perturbation key '{key}' is missing from state")
            state_value = dual_state[key]
            if not isinstance(state_value, torch.Tensor):
                raise TypeError(
                    f"State entry '{key}' is not a tensor and cannot receive a tangent"
                )
            if tangent.shape != state_value.shape:
                raise MpasConfigError(
                    f"Perturbation '{key}' has shape {tuple(tangent.shape)}; "
                    f"expected {tuple(state_value.shape)}"
                )
            if tangent.dtype != state_value.dtype:
                raise MpasConfigError(
                    f"Perturbation '{key}' has dtype {tangent.dtype}; expected {state_value.dtype}"
                )
            if tangent.device != state_value.device:
                raise MpasConfigError(
                    f"Perturbation '{key}' is on {tangent.device}; expected {state_value.device}"
                )
            dual_state[key] = forward_ad.make_dual(state_value, tangent)
        out = run_simulation(dual_state, mesh, **kwargs)
        result: dict[str, Any] = {}
        for key, val in out.items():
            if isinstance(val, torch.Tensor):
                _, tangent_out = forward_ad.unpack_dual(val)
                if tangent_out is not None:
                    result[key] = tangent_out
        _require_finite_model_fields(result, "JVP result", fields=result)
        return result


def _run_simulation_vjp(
    state: dict[str, Any],
    mesh: dict[str, Any],
    objective: Callable[[dict[str, Any]], torch.Tensor],
    track_keys: Iterable[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Adjoint model: compute vector-Jacobian product through run_simulation.

    Args:
        state: Initial model state dictionary.
        mesh: Mesh connectivity and geometry dictionary.
        objective: Callable that maps the output state dict to a scalar loss.
            The adjoint computes d(objective)/d(state) for each prognostic.
        track_keys: Optional iterable of state keys to differentiate with
            respect to. Defaults to the inter-step prognostic set.
        **kwargs: Forwarded to run_simulation (dt, nsteps, config, device, etc.).

    Returns:
        Dict mapping prognostic keys to their adjoint sensitivity (gradient).

    Memory is not bounded at operational scales. This helper is intended for
    short, small-mesh gradient workflows; gradient checkpointing can reduce the
    retained graph when memory is constrained.
    """
    kwargs.setdefault("differentiable", True)
    clone_state = _ordinary_ad_inputs(state)
    tracked: dict[str, Any] = {}
    if track_keys is None:
        requested_keys = tuple(
            key for key in _INTER_STEP_PROGNOSTIC_KEYS if key in clone_state
        )
    else:
        requested_keys = tuple(track_keys)
    for key in requested_keys:
        if key not in clone_state:
            raise KeyError(f"Tracked VJP key '{key}' is missing from state")
        state_value = clone_state[key]
        if not isinstance(state_value, torch.Tensor):
            raise TypeError(f"Tracked VJP key '{key}' is not a tensor")
        leaf = state_value.clone().requires_grad_(True)
        clone_state[key] = leaf
        tracked[key] = leaf
    out = run_simulation(clone_state, mesh, **kwargs)
    loss = objective(out)
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"objective must return a tensor, got {type(loss).__name__}")
    if loss.ndim != 0:
        raise MpasConfigError("objective must return a scalar tensor")
    _require_tensor_predicate(torch.isfinite(loss), "objective must be finite")
    grads = torch.autograd.grad(loss, list(tracked.values()), allow_unused=True)
    result = {key: grad for key, grad in zip(tracked.keys(), grads) if grad is not None}
    _require_finite_model_fields(result, "VJP result", fields=result)
    return result


run_simulation.jvp = _run_simulation_jvp  # type: ignore[attr-defined]
run_simulation.vjp = _run_simulation_vjp  # type: ignore[attr-defined]
