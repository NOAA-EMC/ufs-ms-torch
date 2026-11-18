# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
Dycore state helpers: device moves, tree copies and the step-boundary identity.

A state is a dict of tensors. Entity arrays use zero-based active rows followed
by one padding row:
- Cell arrays: shape (nCells+1, nVertLevels)
- Edge arrays: shape (nEdges+1, nVertLevels)
- W-level arrays: shape (nCells+1, nVertLevels+1)

The initial-condition loader (``ic_loader``) translates upstream connectivity
into this convention.
"""

import contextvars
import hashlib
import json
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from mpas_pytorch.exceptions import MpasConfigError
from mpas_pytorch.io_schema import continuation_configuration_digest, mesh_digest
from mpas_pytorch.precision import assert_async

NATIVE_STATE_KEYS = ("ru", "rw", "rho_p", "rtheta_p", "scalars")
STEP_FORCING_KEYS = frozenset(
    {"rt_diabatic_tend", "tend_ru_physics", "tend_rtheta_physics", "tend_rho_physics"}
)


def dict_to_device(data: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    """
    Return a new dict with tensor values moved to ``device``.

    Non-tensor values are copied through unchanged.
    """
    target_device = torch.device(device)
    # Transfers are reusable inputs, not inference-only temporaries. Static
    # connectivity plans need version counters on the destination as well.
    grad_enabled = torch.is_grad_enabled()
    with torch.inference_mode(False), torch.set_grad_enabled(grad_enabled):
        return {
            key: (
                _boundary_receipt_to_device(value, target_device)
                if key == _BOUNDARY_MARKER
                else value.to(target_device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in data.items()
        }


def clone_tree(value: Any) -> Any:
    """Recursively clone tensors while preserving container structure."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    return value


_BOUNDARY_PROGNOSTICS = ("u", "w", "rho_zz", "theta_m", "scalars")
_BOUNDARY_PRIMARY_ALIASES = {"t": "theta_m"}
_BOUNDARY_MARKER = "_restart_skip_rederivation"
_BOUNDARY_BINDING = "mpas-boundary-tensors.v2"
_NumericalSnapshot = dict[str, tuple[str, tuple[int, ...], bytes]]
_BOUNDARY_TRANSFER_BATCH_BYTES = 64 * 1024 * 1024


def _no_referent() -> None:
    return None


@dataclass(frozen=True)
class _TensorStamp:
    """Host identity of a bound tensor: the same object, storage view and version counter.

    Any in-place write through PyTorch bumps ``_version``; rebinding a key changes the object.
    Writes that bypass the version counter (``.data``, NumPy or DLPack views) are not seen;
    the byte guard mode sees them.
    """

    ref: Any
    data_ptr: int
    storage_offset: int
    stride: tuple[int, ...]
    version: int

    def __reduce__(self):
        # A pickled or reloaded boundary holds new tensors: its stamps never match again.
        return (_TensorStamp, (_no_referent, self.data_ptr, self.storage_offset, self.stride, self.version))

    def matches(self, value: torch.Tensor) -> bool:
        return (
            self.ref() is value
            and value._version == self.version
            and value.data_ptr() == self.data_ptr
            and value.storage_offset() == self.storage_offset
            and tuple(value.stride()) == self.stride
        )


@dataclass(frozen=True)
class _LiveBoundarySnapshot:
    entries: tuple[tuple[str, str, tuple[int, ...], torch.Tensor | bytes | _TensorStamp], ...]


_BOUNDARY_GUARD_MODES = ("bytes", "stamps")
_BOUNDARY_GUARD: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mpas_boundary_guard", default="bytes"
)


@contextmanager
def boundary_guard(mode: str):
    """Select how live boundary snapshots bind tensors inside this context.

    ``"bytes"`` (default) copies each tensor's bytes on its device and compares them later: it
    sees every write, at the cost of a full copy and compare. ``"stamps"`` records a host stamp
    (object, storage view, version counter) and compares it on the host: no kernels and no
    device memory, but writes that bypass the version counter are not seen. Inference tensors
    have no version counter and keep the byte payload in either mode. A snapshot keeps the kind
    it was taken with, so a check in one mode reads a snapshot taken in the other.
    """
    if mode not in _BOUNDARY_GUARD_MODES:
        raise MpasConfigError(f"unknown boundary guard mode {mode!r}")
    token = _BOUNDARY_GUARD.set(mode)
    try:
        yield
    finally:
        _BOUNDARY_GUARD.reset(token)


@dataclass(frozen=True)
class _UfsCommitEntry:
    state: _LiveBoundarySnapshot
    mesh: _LiveBoundarySnapshot
    config: str
    dt: float


_UFS_COMMIT_ENTRY = "_ufs_physics_commit_entry"
_UFS_COMMIT_FIELDS = (
    "tend_ru_physics",
    "tend_rtheta_physics",
    "rt_diabatic_tend",
    "scalars",
    "theta_m",
    "theta",
    "exner",
    "pressure_p",
    "rtheta_p",
    "u",
    "ru",
    "surface_pressure",
    "rho",
    "pressure",
    "uReconstructX",
    "uReconstructY",
    "uReconstructZ",
    "uReconstructZonal",
    "uReconstructMeridional",
    "ur_cell",
    "vr_cell",
    "qv",
    "t",
    "skintemp",
    "sst",
    "snow",
    "snowh",
    "snowc",
    "smois",
    "sh2o",
    "tslb",
)


def bind_ufs_physics_commit(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    mesh: dict[str, Any],
    config: dict[str, Any],
    dt: float,
) -> None:
    """Authenticate a source commit without declaring a completed dycore step."""
    if config.get("physical_constants_profile") != "ufs":
        raise MpasConfigError("UFS physics commit requires UFS constants")
    if _UFS_COMMIT_ENTRY in before:
        parent = dict(before)
        snapshot = before[_UFS_COMMIT_ENTRY]
        consume_ufs_physics_commit(parent, mesh=mesh, config=config, dt=dt)
        saved = snapshot.state
    else:
        require_boundary_continuation_identity(before, mesh=mesh, config=config, dt=dt)
        predicate = timestep_boundary_predicate(before)
        if predicate is None:
            raise MpasConfigError("UFS physics commit requires an initialized boundary")
        _require_tensor_predicate(predicate, "UFS physics commit parent was edited")
        saved = _live_boundary_marker(before)["state"]
    _require_tensor_predicate(
        _live_boundary_matches(
            saved,
            after,
            device=after["u"].device,
            exclude=_UFS_COMMIT_FIELDS,
        ),
        "UFS physics commit changed source-unwritten carry state",
    )
    after.pop(_BOUNDARY_MARKER, None)
    after[_UFS_COMMIT_ENTRY] = _UfsCommitEntry(
        _live_boundary_snapshot(after),
        _live_boundary_snapshot(mesh, all_keys=True, device=after["u"].device),
        continuation_configuration_digest(config),
        float(dt),
    )


def consume_ufs_physics_commit(
    state: dict[str, Any],
    *,
    mesh: dict[str, Any],
    config: dict[str, Any],
    dt: float,
) -> bool:
    """Validate exact entry ownership before selecting source-specific preparation."""
    entry = state.get(_UFS_COMMIT_ENTRY)
    if entry is None:
        return False
    if not isinstance(entry, _UfsCommitEntry):
        raise MpasConfigError("UFS physics commit entry is not authenticated")
    if (
        entry.config != continuation_configuration_digest(config)
        or entry.dt != float(dt)
        or config.get("physical_constants_profile") != "ufs"
    ):
        raise MpasConfigError("UFS physics commit continuation identity differs")
    device = state["u"].device
    _require_tensor_predicate(
        _live_boundary_matches(entry.state, state, device=device)
        & _live_boundary_matches(entry.mesh, mesh, device=device, all_keys=True),
        "UFS physics commit state or mesh was edited",
    )
    state.pop(_UFS_COMMIT_ENTRY)
    return True


def _boundary_receipt_to_device(marker: Any, device: torch.device) -> Any:
    if not isinstance(marker, dict):
        return marker
    result = dict(marker)
    for key, snapshot in marker.items():
        if isinstance(snapshot, _LiveBoundarySnapshot):
            if any(isinstance(entry[3], _TensorStamp) for entry in snapshot.entries):
                raise MpasConfigError(
                    "a stamp-bound boundary cannot move between devices; bind it in byte mode"
                )
            result[key] = _LiveBoundarySnapshot(
                tuple(
                    (
                        name,
                        dtype,
                        shape,
                        payload.to(device)
                        if isinstance(payload, torch.Tensor)
                        else payload,
                    )
                    for name, dtype, shape, payload in snapshot.entries
                )
            )
    return result


def _boundary_items(state: dict[str, Any], *, all_keys: bool = False):
    for name, value in sorted(state.items()):
        if (
            not all_keys
            and name.startswith("_")
            and name not in {"_edge_ordering_mismatch", "_mesh_connectivity_preferred"}
        ):
            continue
        if isinstance(
            value, (torch.Tensor, np.ndarray, bool, int, float, str, np.generic)
        ):
            yield name, value


def _live_boundary_snapshot(
    state: dict[str, Any],
    *,
    all_keys: bool = False,
    device: torch.device | None = None,
    mode: str | None = None,
) -> _LiveBoundarySnapshot:
    """Own exact logical bytes on-device (or host stamps), without graph detachment.

    ``mode`` overrides the active ``boundary_guard`` mode for this snapshot.
    """
    stamps = (mode or _BOUNDARY_GUARD.get()) == "stamps"
    entries = []
    copies = {}
    for name, value in _boundary_items(state, all_keys=all_keys):
        if isinstance(value, (torch.Tensor, np.ndarray)):
            original = value
            value = torch.as_tensor(value)
            if value.is_conj() or value.is_neg():
                raise MpasConfigError(
                    "boundary tensors must resolve conjugate/negative views"
                )
            key = id(original)
            if (
                key not in copies
                and stamps
                and isinstance(original, torch.Tensor)
                and not original.is_inference()
            ):
                copies[key] = _TensorStamp(
                    weakref.ref(original),
                    original.data_ptr(),
                    original.storage_offset(),
                    tuple(original.stride()),
                    original._version,
                )
            if key not in copies:
                # view_copy retains NaN/zero bits without view(dtype)'s implicit
                # autograd detach. Only ownership metadata uses integer bytes.
                payload = torch.view_copy(value.reshape(-1).contiguous(), dtype=torch.uint8)
                copies[key] = payload.to(device) if device is not None else payload
            entries.append((name, str(value.dtype), tuple(value.shape), copies[key]))
        else:
            entries.append(
                (
                    name,
                    "scalar",
                    (),
                    json.dumps(value, sort_keys=True, default=str).encode(),
                )
            )
    return _LiveBoundarySnapshot(tuple(entries))


def _live_boundary_matches(
    saved: _LiveBoundarySnapshot,
    state: dict[str, Any],
    *,
    device: torch.device,
    exclude: tuple[str, ...] = (),
    all_keys: bool = False,
) -> torch.Tensor:
    current = {
        name: value
        for name, value in _boundary_items(state, all_keys=all_keys)
        if name not in exclude
    }
    entries = tuple(entry for entry in saved.entries if entry[0] not in exclude)
    if set(current) != {entry[0] for entry in entries}:
        return torch.zeros((), dtype=torch.bool, device=device)
    checks = []
    for name, dtype, shape, payload in entries:
        value = current[name]
        if isinstance(payload, _TensorStamp):
            if (
                not isinstance(value, torch.Tensor)
                or str(value.dtype) != dtype
                or tuple(value.shape) != shape
                or not payload.matches(value)
            ):
                return torch.zeros((), dtype=torch.bool, device=device)
            continue
        if isinstance(payload, torch.Tensor):
            if not isinstance(value, (torch.Tensor, np.ndarray)):
                return torch.zeros((), dtype=torch.bool, device=device)
            value = torch.as_tensor(value)
            if str(value.dtype) != dtype or tuple(value.shape) != shape:
                return torch.zeros((), dtype=torch.bool, device=device)
            if value.is_conj() or value.is_neg():
                return torch.zeros((), dtype=torch.bool, device=device)
            checks.append(
                (
                    torch.view_copy(value.reshape(-1).contiguous(), dtype=torch.uint8).to(device)
                    == payload.to(device)
                )
                .all()
                .to(device)
            )
        elif (
            isinstance(value, (torch.Tensor, np.ndarray))
            or json.dumps(value, sort_keys=True, default=str).encode() != payload
        ):
            return torch.zeros((), dtype=torch.bool, device=device)
    return (
        torch.stack(checks).all()
        if checks
        else torch.ones((), dtype=torch.bool, device=device)
    )


def boundary_snapshot_bytes(
    snapshot: _LiveBoundarySnapshot, current: dict[str, Any], *, all_keys: bool = False
) -> _LiveBoundarySnapshot:
    """Byte form of ``snapshot`` for persistence: each stamp entry becomes the bytes of its
    live tensor in ``current``, which must still match the stamp (no edit since binding)."""
    items = dict(_boundary_items(current, all_keys=all_keys))
    entries = []
    for name, dtype, shape, payload in snapshot.entries:
        if isinstance(payload, _TensorStamp):
            value = items.get(name)
            if (
                not isinstance(value, torch.Tensor)
                or str(value.dtype) != dtype
                or tuple(value.shape) != shape
                or not payload.matches(value)
            ):
                raise MpasConfigError("a stamp-bound boundary was edited; it cannot be persisted")
            payload = torch.view_copy(value.reshape(-1).contiguous(), dtype=torch.uint8)
        entries.append((name, dtype, shape, payload))
    return _LiveBoundarySnapshot(tuple(entries))


def _require_tensor_predicate(predicate: torch.Tensor, message: str) -> None:
    try:
        assert_async(predicate, message)
    except RuntimeError as error:
        if str(error) != message:
            raise
        raise MpasConfigError(message) from error


def _live_boundary_marker(state: dict[str, Any]) -> dict[str, Any] | None:
    marker = state.get(_BOUNDARY_MARKER)
    if marker is None:
        return None
    if (
        not isinstance(marker, dict)
        or marker.get("binding") != _BOUNDARY_BINDING
        or not isinstance(marker.get("state"), _LiveBoundarySnapshot)
    ):
        raise MpasConfigError(
            "complete boundary binding is unsupported; reload a current restart"
        )
    return marker


def _boundary_reference_device(state: dict[str, Any]) -> torch.device:
    for name in _BOUNDARY_PROGNOSTICS:
        if not isinstance(state.get(name), torch.Tensor):
            raise MpasConfigError(f"complete step boundary lacks tensor {name!r}")
    return state["u"].device


def timestep_boundary_predicate(state: dict[str, Any]) -> torch.Tensor | None:
    """Authorize diagnostics on-device; primary edits select rederivation."""
    marker = _live_boundary_marker(state)
    if marker is None:
        return None
    require_unedited_boundary_support(state)
    return _live_boundary_matches(
        marker["state"], state, device=_boundary_reference_device(state)
    )


def _boundary_prognostic_digest(state: dict[str, Any]) -> str:
    """Bind a complete step snapshot to the exact prognostic tensor values.

    A value digest survives a deep clone and catches both tensor replacement and
    in-place edits. A pointer/version receipt would incorrectly reject a clone.
    """
    digest = hashlib.sha256()
    for name in _BOUNDARY_PROGNOSTICS:
        value = state.get(name)
        if not isinstance(value, torch.Tensor):
            raise MpasConfigError(f"complete step boundary lacks tensor {name!r}")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.data.cast("B"))
    return digest.hexdigest()


def _boundary_state_snapshot(
    state: dict[str, Any], *, support_only: bool = False
) -> _NumericalSnapshot:
    """Copy immutable bytes; comparisons bind values without version-counter gaps.

    A call-local cache shares bytes for identical tensor views. The cache never
    survives a call, so numpy/.data edits and cloned tensors are checked afresh.
    """
    snapshot: _NumericalSnapshot = {}
    copies: dict[tuple[Any, ...], tuple[str, tuple[int, ...], bytes]] = {}
    pending: dict[tuple[Any, ...], tuple[torch.Tensor, list[str]]] = {}
    streams: dict[torch.device, torch.cuda.Stream] = {}
    pending_bytes = 0

    def flush() -> None:
        nonlocal pending_bytes
        # This is host boundary serialization, not a numerical execution path.
        # Wait before reading pinned destinations; retain only immutable bytes.
        for stream in streams.values():
            stream.synchronize()
        for view, (buffer, names) in pending.items():
            array = buffer.numpy()
            copies[view] = (str(array.dtype), tuple(array.shape), array.tobytes())
            for name in names:
                snapshot[name] = copies[view]
        pending.clear()
        streams.clear()
        pending_bytes = 0

    identity: tuple[Any, ...]
    for key, value in state.items():
        if key.startswith("_") and key not in {
            "_edge_ordering_mismatch",
            "_mesh_connectivity_preferred",
        }:
            continue
        if support_only and (
            key in _BOUNDARY_PROGNOSTICS or key in _BOUNDARY_PRIMARY_ALIASES
        ):
            continue
        if isinstance(value, torch.Tensor):
            identity = (
                "tensor",
                str(value.device),
                value.data_ptr(),
                str(value.dtype),
                tuple(value.shape),
                tuple(value.stride()),
                value.is_conj(),
                value.is_neg(),
            )
            # Check before .cpu(): aliases of a CUDA view otherwise each incur
            # a device transfer even though their bytes are shared below. This
            # cache remains call-local so edits outside version counters are
            # still observed on every subsequent boundary check.
            if identity in copies:
                snapshot[key] = copies[identity]
                continue
            if identity in pending:
                pending[identity][1].append(key)
                continue
            if value.device.type == "cuda" and not (value.is_conj() or value.is_neg()):
                size = value.numel() * value.element_size()
                if pending and pending_bytes + size > _BOUNDARY_TRANSFER_BATCH_BYTES:
                    flush()
                buffer = torch.empty(
                    value.shape, dtype=value.dtype, device="cpu", pin_memory=True
                )
                stream = torch.cuda.current_stream(value.device)
                buffer.copy_(value.detach(), non_blocking=True)
                pending[identity] = (buffer, [key])
                del buffer
                streams[value.device] = stream
                pending_bytes += size
                # One field can exceed the cap, but never shares that batch.
                if pending_bytes >= _BOUNDARY_TRANSFER_BATCH_BYTES:
                    flush()
                continue
            if pending:
                flush()
            array = value.detach().cpu().contiguous().numpy()
        elif isinstance(value, np.ndarray):
            identity = (
                "array",
                value.__array_interface__["data"][0],
                str(value.dtype),
                value.shape,
                value.strides,
            )
            if identity in copies:
                snapshot[key] = copies[identity]
                continue
            array = np.ascontiguousarray(value)
        elif isinstance(value, (bool, int, float, str, np.generic)):
            snapshot[key] = (
                "scalar",
                (),
                json.dumps(value, sort_keys=True, default=str).encode(),
            )
            continue
        else:
            continue
        copies[identity] = (str(array.dtype), tuple(array.shape), array.tobytes())
        snapshot[key] = copies[identity]
    flush()
    return {key: snapshot[key] for key in state if key in snapshot}


def require_unedited_boundary_support(state: dict[str, Any]) -> None:
    """Primary edits may be rederived; diagnostic/support edits are invalid."""
    marker = _live_boundary_marker(state)
    if marker is None:
        return
    device = _boundary_reference_device(state)
    _require_tensor_predicate(
        _live_boundary_matches(
            marker["state"],
            state,
            device=device,
            exclude=(*_BOUNDARY_PROGNOSTICS, *_BOUNDARY_PRIMARY_ALIASES),
        ),
        "complete boundary diagnostics or support inputs were edited",
    )
    saved = {entry[0]: entry for entry in marker["state"].entries}
    for alias, primary in _BOUNDARY_PRIMARY_ALIASES.items():
        if (alias in saved) != (alias in state):
            raise MpasConfigError(
                "complete boundary diagnostic alias inventory changed"
            )
        if alias not in saved:
            continue
        value, primary_value = state[alias], state.get(primary)
        if not isinstance(value, torch.Tensor):
            raise MpasConfigError("complete boundary diagnostic alias was edited")
        unchanged = _live_boundary_matches(
            _LiveBoundarySnapshot((saved[alias],)), {alias: value}, device=device
        )
        following = torch.zeros((), dtype=torch.bool, device=device)
        if (
            isinstance(primary_value, torch.Tensor)
            and value.dtype == primary_value.dtype
            and value.device == primary_value.device
            and value.shape == primary_value.shape
        ):
            following = (
                torch.view_copy(value.reshape(-1).contiguous(), dtype=torch.uint8)
                == torch.view_copy(primary_value.reshape(-1).contiguous(), dtype=torch.uint8)
            ).all()
        _require_tensor_predicate(
            unchanged | following, "complete boundary diagnostic alias was edited"
        )


def mark_complete_step_boundary(
    state: dict[str, Any],
    *,
    mesh: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    dt: float | None = None,
) -> None:
    """Bind a reusable boundary to its values and continuation identity."""
    for name in _BOUNDARY_PROGNOSTICS:
        if not isinstance(state.get(name), torch.Tensor):
            raise MpasConfigError(f"complete step boundary lacks tensor {name!r}")
    snapshot = _live_boundary_snapshot(state)
    state[_BOUNDARY_MARKER] = {
        "binding": _BOUNDARY_BINDING,
        "state": snapshot,
        "mesh": _live_boundary_snapshot(mesh, all_keys=True, device=state["u"].device)
        if mesh is not None
        else None,
        "config": continuation_configuration_digest(config)
        if config is not None
        else None,
        "dt": float(dt) if dt is not None else None,
    }


def require_boundary_continuation_identity(
    state: dict[str, Any], *, mesh: dict[str, Any], config: dict[str, Any], dt: float
) -> None:
    """Primary edits require rederivation, never a different continuation identity."""
    marker = _live_boundary_marker(state)
    if marker is None:
        return
    if marker.get("config") != continuation_configuration_digest(config) or marker.get(
        "dt"
    ) != float(dt):
        raise MpasConfigError("complete boundary continuation identity mismatch")
    saved_mesh = marker.get("mesh")
    if isinstance(saved_mesh, _LiveBoundarySnapshot):
        _require_tensor_predicate(
            _live_boundary_matches(
                saved_mesh,
                mesh,
                device=_boundary_reference_device(state),
                all_keys=True,
            ),
            "complete boundary continuation identity mismatch",
        )
    elif saved_mesh != mesh_digest(mesh):
        # A restart read without mesh keeps its authenticated disk identity.
        # Normal live boundaries and mesh-bound restart reads never hash here.
        raise MpasConfigError("complete boundary continuation identity mismatch")


def has_complete_step_boundary(
    state: dict[str, Any],
    *,
    mesh: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    dt: float | None = None,
) -> bool:
    """Offline I/O predicate; live numerics use timestep_boundary_predicate."""
    try:
        marker = _live_boundary_marker(state)
        if marker is None:
            return False
        if mesh is not None:
            saved_mesh = marker.get("mesh")
            mesh_matches = (
                bool(
                    _live_boundary_matches(
                        saved_mesh,
                        mesh,
                        device=_boundary_reference_device(state),
                        all_keys=True,
                    )
                )
                if isinstance(saved_mesh, _LiveBoundarySnapshot)
                else saved_mesh == mesh_digest(mesh)
            )
        else:
            mesh_matches = True
        return (
            bool(
                _live_boundary_matches(
                    marker["state"], state, device=_boundary_reference_device(state)
                )
            )
            and mesh_matches
            and (
                config is None
                or marker.get("config") == continuation_configuration_digest(config)
            )
            and (dt is None or marker.get("dt") == float(dt))
        )
    except (MpasConfigError, ValueError):
        return False


def _snapshot_mesh_digest(snapshot: _LiveBoundarySnapshot) -> str:
    """``io_schema.mesh_digest`` of the mesh a byte-mode snapshot bound, rebuilt from its own bytes."""
    mesh: dict[str, Any] = {}
    for name, dtype, shape, payload in snapshot.entries:
        if dtype == "scalar":
            mesh[name] = json.loads(cast(bytes, payload))
        elif isinstance(payload, torch.Tensor):
            kind = getattr(torch, dtype.removeprefix("torch."))
            mesh[name] = torch.view_copy(payload, dtype=kind).reshape(shape)
        else:
            raise MpasConfigError("a stamp-bound mesh has no portable identity; bind it in byte mode")
    try:
        return mesh_digest(mesh)
    except ValueError as error:
        raise MpasConfigError("complete boundary identity lacks a context digest") from error


def complete_step_boundary_identity(state: dict[str, Any]) -> dict[str, Any]:
    """Portable continuation context (mesh and configuration digests, dt) of a sealed boundary.

    Raises unless the state is an unedited complete step boundary bound to a mesh,
    configuration and timestep.
    """
    if not has_complete_step_boundary(state):
        raise MpasConfigError("cannot export an edited or incomplete step boundary")
    marker = cast(dict[str, Any], _live_boundary_marker(state))
    mesh = marker.get("mesh")
    identity = {
        "mesh": _snapshot_mesh_digest(mesh) if isinstance(mesh, _LiveBoundarySnapshot) else mesh,
        "config": marker.get("config"),
        "dt": marker.get("dt"),
    }
    validate_complete_step_boundary_identity(identity)
    return identity


def validate_complete_step_boundary_identity(identity: dict[str, Any]) -> None:
    """Validate portable context syntax without creating or changing a seal."""
    if not isinstance(identity, dict):
        raise MpasConfigError("complete boundary identity is not a mapping")
    if set(identity) != {"mesh", "config", "dt"}:
        raise MpasConfigError("complete boundary identity inventory mismatch")
    for name in ("mesh", "config"):
        value = identity[name]
        if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise MpasConfigError("complete boundary identity lacks a context digest")
    dt = identity["dt"]
    if isinstance(dt, bool) or not isinstance(dt, (float, int)) or not np.isfinite(dt) or dt <= 0:
        raise MpasConfigError("complete boundary identity has an invalid timestep")


def restore_complete_step_boundary_identity(
    state: dict[str, Any], identity: dict[str, Any]
) -> None:
    """Seal already authenticated continuation values with an exported identity.

    The caller verifies its retained values and static support first. The next
    step still compares mesh, configuration and dt with its own execution context.
    """
    validate_complete_step_boundary_identity(identity)
    mark_complete_step_boundary(state)
    cast(dict[str, Any], state[_BOUNDARY_MARKER]).update(
        mesh=identity["mesh"], config=identity["config"], dt=float(identity["dt"])
    )


