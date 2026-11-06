# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Replay a steady forward timestep as one CUDA graph per process.

This is an execution policy around the ordinary timestep program, not a second
numerical path. The recorded work is exactly one call of the same timestep and
interstep projection, including any NCCL halo messages issued inside it. Each
replay launches the recorded kernels with no Python dispatch, which removes the
host cost that otherwise bounds strong scaling.

Forward only. The recorded step must be steady: the same keys, the same
non-tensor values and the same aliasing on entry and exit. The boundary values
are copied back into the recorded inputs at the end of the graph, so every
replay returns the same set of tensors. Non-tensor boundary values must be
plain host values: a container that holds tensors would pass the steady check
by identity while its tensors go stale.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

StepFn = Callable[[dict[str, Any]], dict[str, Any]]
# Host values compared and restored by name, e.g. storage/version receipts.
HostView = Callable[[Mapping[str, Any]], Mapping[str, Any]]
Rebind = Callable[[dict[str, Any], Mapping[str, Any]], None]
# Comparable description of caches that recording must not fill.
CacheView = Callable[[], Any]
# Process-wide evidence that recording and replay actually happened.
COUNTERS = {"captures": 0, "replays": 0}


_PLAIN = (type(None), bool, int, float, complex, str, bytes, torch.dtype, torch.device)


def _storage(value: torch.Tensor) -> int:
    return value.untyped_storage().data_ptr()


def _is_plain(value: Any) -> bool:
    """True for host values that hold no tensor and compare by value."""
    if isinstance(value, _PLAIN):
        return True
    if isinstance(value, (tuple, list, set, frozenset)):
        return all(_is_plain(item) for item in value)
    if isinstance(value, dict):
        return all(_is_plain(k) and _is_plain(v) for k, v in value.items())
    return False


def _same_value(first: Any, second: Any) -> bool:
    if first is second:
        return True
    try:
        return bool(first == second)
    except Exception:  # noqa: BLE001 - uncomparable values are not steady
        return False


def _shared_storage_groups(
    state: Mapping[str, Any],
) -> frozenset[tuple[tuple[str, int], ...]]:
    """Keys that share a storage, with each view's offset into it."""
    owners: dict[int, list[tuple[str, int]]] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            owners.setdefault(_storage(value), []).append(
                (key, int(value.storage_offset()))
            )
    return frozenset(
        tuple(sorted(views)) for views in owners.values() if len(views) > 1
    )


@dataclass(frozen=True)
class BoundarySignature:
    """What the steady check compares, without holding any boundary tensor."""

    values: dict[str, tuple[Any, ...]]
    aliasing: frozenset[tuple[tuple[str, int], ...]]


def boundary_signature(
    state: Mapping[str, Any], host_view: HostView | None = None
) -> BoundarySignature:
    """Tensor layouts, host values and storage sharing of a boundary.

    ``host_view`` maps a boundary to comparable host values, for example
    receipts that name tensors by key instead of by allocation address.
    """
    host = host_view(state) if host_view else state
    values: dict[str, tuple[Any, ...]] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            values[key] = (
                "tensor",
                tuple(value.shape),
                value.stride(),
                value.dtype,
                value.device,
            )
            continue
        viewed = host.get(key)
        if not (_is_plain(value) and _is_plain(viewed)):
            raise TypeError(
                f"timestep graph boundary value {key!r} is not a plain host value"
            )
        values[key] = ("host", viewed)
    return BoundarySignature(values, _shared_storage_groups(state))


def signature_changes(before: BoundarySignature, after: BoundarySignature) -> list[str]:
    """Keys whose presence, kind, tensor signature or host value differs."""
    changed = sorted(set(before.values) ^ set(after.values))
    for key in sorted(set(before.values) & set(after.values)):
        value, other = before.values[key], after.values[key]
        if value[0] != other[0]:
            same = False
        elif value[0] == "tensor":
            same = value == other
        else:
            same = _same_value(value[1], other[1])
        if not same:
            changed.append(key)
    if not changed and before.aliasing != after.aliasing:
        changed.append("<tensor aliasing>")
    return changed


def boundary_changes(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    host_view: HostView | None = None,
) -> list[str]:
    """Keys whose presence, kind, tensor signature or host value differs."""
    return signature_changes(
        boundary_signature(before, host_view), boundary_signature(after, host_view)
    )


def _released_step(state: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError("timestep graph was released")


def _capture_device(device: torch.device) -> torch.device:
    """The device whose stream records; work elsewhere is not replayed."""
    if device.index is not None:
        return device
    return torch.device("cuda", torch.cuda.current_device())


# Per device: the shared pool's handle and the live graphs recorded into it.
_GRAPH_POOLS: dict[int, tuple[Any, weakref.WeakSet]] = {}


def _cuda_capture(record: Callable[[], None]) -> Callable[[], None]:
    """Record ``record`` without running it; return the replay launcher.

    The live step graphs of a device share one memory pool: a replay copies its outputs into the
    boundary (outside the pool) and nothing in the pool is read after its replay, so a pool sized for the
    largest step serves every step variant (one pool per graph holds one step's transients per variant).
    A pool is shared only while a graph recorded into it is alive: the allocator counts a private pool's
    live graphs and keeps a pool whose count reached 0 while it still holds memory, and a capture into
    such a pool fails its ``use_count > 0`` assert (CUDACachingAllocator.cpp create_or_incref_pool,
    torch 2.8). Once every graph of the pool is gone the next capture takes a new handle; the
    ``empty_cache`` before each capture returns the old pool's free memory.
    """
    graph = torch.cuda.CUDAGraph()
    device = torch.cuda.current_device()
    entry = _GRAPH_POOLS.get(device)
    if entry is None or not len(entry[1]):
        entry = _GRAPH_POOLS[device] = (torch.cuda.graph_pool_handle(), weakref.WeakSet())
    pool, owners = entry
    # NCCL's watchdog thread queries events; only this thread is recorded.
    with torch.cuda.graph(graph, pool=pool, capture_error_mode="thread_local"):
        record()
    owners.add(graph)
    return graph.replay


class TimestepGraph:
    """Record ``step`` once on a steady boundary, then replay it."""

    def __init__(
        self,
        step: StepFn,
        device: torch.device | str,
        *,
        protected: Mapping[str, Any] | None = None,
        host_view: HostView | None = None,
        rebind: Rebind | None = None,
        cache_view: CacheView | None = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("timestep graphs require a CUDA device")
        self._step = step
        self._host_view = host_view
        self._rebind = rebind
        self._cache_view = cache_view
        self._recorded_view: Mapping[str, Any] = {}
        # Storages a replay must never write, e.g. the caller's input state.
        self._protected = {
            _storage(value)
            for value in (protected or {}).values()
            if isinstance(value, torch.Tensor)
        }
        self._replay: Callable[[], None] | None = None
        self._returned: dict[str, Any] = {}
        self._written: tuple[torch.Tensor, ...] = ()
        self.captures = 0
        self.replays = 0

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        if self._step is _released_step:
            raise RuntimeError("timestep graph was released")
        recording = self._replay is None
        if recording:
            caches = self._cache_view() if self._cache_view else None
            self._record(state)
        else:
            self._require_recorded_inputs(state)
        assert self._replay is not None
        self._replay()
        if recording and self._cache_view is not None and self._cache_view() != caches:
            # Recording runs no kernels: a cache filled then holds unwritten
            # graph-pool memory that later ordinary steps would read.
            raise RuntimeError("timestep graph recording filled a cache")
        self.replays += 1
        COUNTERS["replays"] += 1
        # A replay changes values like an in-place copy; keep version-keyed
        # caches from treating the rewritten boundary as unchanged.
        for value in self._written:
            if not value.is_inference():
                torch.autograd.graph.increment_version(value)
        returned = dict(self._returned)
        if self._rebind is not None:
            # Host receipts name the recorded tensors, as after an ordinary step.
            self._rebind(returned, self._recorded_view)
        return returned

    def release(self) -> None:
        """Free the recorded graph (the shared pool keeps its memory while other graphs use it); the boundary stays."""
        self._replay = None
        self._step = _released_step

    def _require_recorded_inputs(self, state: Mapping[str, Any]) -> None:
        if set(state) != set(self._returned) or any(
            state[key] is not value
            for key, value in self._returned.items()
            if isinstance(value, torch.Tensor)
        ):
            raise RuntimeError(
                "timestep graph input is not the boundary its replay returned"
            )

    def _record(self, state: dict[str, Any]) -> None:
        recorded = dict(state)
        device = _capture_device(self.device)

        def record() -> None:
            output = self._step(dict(recorded))
            destinations, sources = _copy_plan(
                recorded, output, self._host_view, device
            )
            if any(_storage(value) in self._protected for value in destinations):
                raise RuntimeError("timestep graph would overwrite a protected input")
            if destinations:
                torch._foreach_copy_(list(destinations), list(sources))
            self._written = destinations

        self._recorded_view = dict(self._host_view(recorded)) if self._host_view else {}
        torch.cuda.synchronize(self.device)
        # Cached ordinary-pool blocks cannot serve the graph pool.
        torch.cuda.empty_cache()
        self._replay = _cuda_capture(record)
        self.captures += 1
        COUNTERS["captures"] += 1
        self._returned = recorded


def _copy_plan(
    recorded: Mapping[str, Any],
    output: Mapping[str, Any],
    host_view: HostView | None = None,
    device: torch.device | None = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Copies that return ``output``'s values in ``recorded``'s tensors."""
    changed = boundary_changes(recorded, output, host_view)
    if changed:
        raise RuntimeError(
            f"timestep graph boundary is not steady; changed: {changed[:10]}"
        )
    if device is not None:
        elsewhere = sorted(
            key
            for key, target in recorded.items()
            if isinstance(target, torch.Tensor)
            and output[key] is not target
            and not target.device == output[key].device == device
        )
        if elsewhere:
            # Copies off the recording device run once, never on replay.
            raise RuntimeError(
                f"timestep graph boundary leaves {device}: {elsewhere[:10]}"
            )
    destinations: list[torch.Tensor] = []
    sources: list[torch.Tensor] = []
    seen: dict[int, torch.Tensor] = {}
    for key, target in recorded.items():
        value = output[key]
        if not isinstance(target, torch.Tensor) or value is target:
            continue
        if id(target) in seen:
            # Aliased keys must stay one tensor on both sides.
            if seen[id(target)] is not value:
                raise RuntimeError(f"timestep graph aliasing changes at {key}")
            continue
        seen[id(target)] = value
        destinations.append(target)
        sources.append(value)
    targets = {_storage(value) for value in destinations}
    # A source that lives in a destination would be overwritten first: copy it before any copy-back.
    # Other sources are read directly (a clone of each would hold the step's outputs twice).
    sources = [value.clone() if _storage(value) in targets else value for value in sources]
    return tuple(destinations), tuple(sources)
