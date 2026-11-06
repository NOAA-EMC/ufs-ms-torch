# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Coupled steps replayed as CUDA graphs, one graph per step variant (``cycle.step_variant``).

An execution policy around ``run_coupled_cycle``, not a second numerical path: each graph records
one ordinary step and copies its boundary back into the recorded inputs
(``mpas_pytorch.cuda_graph_timestep.TimestepGraph``), so every replay advances the same tensors.
The boundary is the coupled state flattened to path -> leaf, without the static step index, which
the driver keeps and uses only to pick the graph. ``runtime_policy.set_step_graphs`` must be on:
the step checks then hold no step number.

Host receipts stay out of the boundary, because a replay cannot update them: MPAS's live-boundary
stamps (object, storage view and version of a bound tensor) and the dycore's diagnostic-solve
source receipt (allocation addresses). Recording names each receipt's tensors by boundary key;
after every replay the driver rebuilds the receipts for the recorded tensors, as an ordinary step
would (the diagnostic receipt with the dycore's own rebind). Stamps of static tensors (the mesh)
pass through unchanged.

A replay records no autograd graph and no forward-mode tangents. Under a derivative request (grad
mode or an active dual level) the driver therefore advances with ordinary steps
(``run_coupled_interval``), the qualified differentiable path.

Replays accumulate the compaction overflow flag; the driver reads it every
``runtime_policy.overflow_check_steps()`` replays (one host sync per window). On overflow it restores
the window's input, moves the regions one rung up (``runtime_policy.raise_compaction``), drops the
recorded graphs (their buffer shapes changed) and repeats the window; compaction is exact at every rung,
so results equal those of a per-step check. After a capacity change every variant runs one ordinary
step before it records again, so compiled regions that read the raised capacity compile outside a
CUDA-graph capture. Every ordinary step drops the recorded graphs first: a graph keeps its private
memory pool reserved, so no ordinary step runs beside one (each variant records once more after it).
The restore point holds only the boundary tensors the steady variants' ordinary steps wrote (the mesh
and other read-only tensors are left out), in buffers kept across windows (pinned host memory on
CUDA, filled and read back with stream-ordered copies).

``StepGraphRunner`` keeps the graphs, the steady variants and the boundary across calls, so a
forecast advanced interval by interval records each variant once.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
import torch.autograd.forward_ad as forward_ad
from mpas_pytorch.cuda_graph_timestep import TimestepGraph, _copy_plan, boundary_changes
from mpas_pytorch.atmosphere_model import (
    _finite_stamp_matches, _record_finite_stamp, _rebind_diagnostic_solve_receipt,
)
from mpas_pytorch.dynamics.solve_diagnostics import (
    DIAGNOSTIC_SOLVE_SOURCE_RECEIPT_KEY,
    diagnostic_solve_checkpoint_metadata,
)
from mpas_pytorch.state import _TensorStamp
from ufs_physics_common import cost_meter
from ufs_physics_common.compaction import overflowed_regions, reset_overflow

from .cycle import (COMPACTION_LOG, CoupledCycleState, frozen_seed_stream, prepare_step, run_coupled_cycle,
    run_coupled_interval, step_variant)
from .distributed import all_ranks, consensus_regions
from .runtime_policy import overflow_check_steps, raise_compaction, step_graphs_enabled

_PLAIN = (type(None), bool, int, float, str, bytes)
# Steps run as ordinary steps, graphs recorded and replays, over the process (evidence of use).
COUNTS = {"ordinary": 0, "recorded": 0, "replayed": 0}


def flatten_state(state: CoupledCycleState) -> tuple[dict[str, Any], dict[str, _TensorStamp]]:
    """The step boundary (every leaf of ``state`` by path, without ``step_index``) and its stamps."""
    leaves: dict[str, Any] = {}
    stamps: dict[str, _TensorStamp] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, _TensorStamp) or path == _DIAGNOSTIC:
            stamps[path] = value
        elif isinstance(value, torch.Tensor) or value is None or type(value) in _PLAIN:
            leaves[path] = value
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                if not (value is state and field.name == "step_index"):
                    visit(getattr(value, field.name), f"{path}/{field.name}")
        elif isinstance(value, Mapping):
            for key in value:
                if type(key) is not str or "/" in key:
                    raise TypeError(f"step boundary keys must be plain strings at {path}")
                visit(value[key], f"{path}/{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}/{index}")
        else:
            raise TypeError(f"unsupported step boundary value at {path}: {type(value).__name__}")

    visit(state, "")
    return leaves, stamps


def unflatten_state(template: CoupledCycleState, leaves: Mapping[str, Any], step_index: int) -> CoupledCycleState:
    """``template``'s structure with the given leaves (boundary and stamps) and static step index."""

    def build(value: Any, path: str) -> Any:
        if path in leaves:
            return leaves[path]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            changes = {f.name: build(getattr(value, f.name), f"{path}/{f.name}")
                       for f in dataclasses.fields(value) if f.init and not (value is template and f.name == "step_index")}
            return dataclasses.replace(value, **changes)
        if isinstance(value, Mapping):
            return type(value)({key: build(item, f"{path}/{key}") for key, item in value.items()})
        if isinstance(value, (tuple, list)):
            return type(value)(build(item, f"{path}/{index}") for index, item in enumerate(value))
        raise KeyError(f"step boundary lacks {path}")

    return dataclasses.replace(build(template, ""), step_index=step_index)


_DIAGNOSTIC = f"/native_state/{DIAGNOSTIC_SOLVE_SOURCE_RECEIPT_KEY}"
_DIAGNOSTIC_SOURCES = ("u", "rho_zz", "h")


def _diagnostic_mapping(boundary: Mapping[str, Any], receipt: Any) -> dict[str, Any]:
    mapping = {name: boundary[f"/native_state/{name}"] for name in _DIAGNOSTIC_SOURCES
               if f"/native_state/{name}" in boundary}
    mapping[DIAGNOSTIC_SOLVE_SOURCE_RECEIPT_KEY] = receipt
    return mapping


def stamp_names(boundary: Mapping[str, Any], stamps: Mapping[str, Any]) -> dict[str, Any]:
    """Each receipt by the boundary keys it names: a stamp as the key of its tensor (the stamp
    itself for a static tensor), the diagnostic receipt as the dycore's host view of it."""
    names = {}
    for key, value in boundary.items():
        if isinstance(value, torch.Tensor):
            names.setdefault(id(value), key)
    named = {}
    for path, stamp in stamps.items():
        if path == _DIAGNOSTIC:
            mapping = _diagnostic_mapping(boundary, stamp)
            named[path] = ("diagnostic", diagnostic_solve_checkpoint_metadata(mapping)[DIAGNOSTIC_SOLVE_SOURCE_RECEIPT_KEY],
                           stamp)
        else:
            named[path] = names.get(id(stamp.ref()), stamp)
    return named


def _comparable(named: Mapping[str, Any]) -> dict[str, Any]:
    """Receipt names without allocation addresses (what must agree for a steady boundary).

    The diagnostic receipt is compared by each source's layout and the boundary keys it names, not by
    the source's version counter: copy-backs and replays raise the boundary tensors' versions, so after
    replays an ordinary step's fresh outputs never matched (every radiation step stayed ordinary and
    dropped the recorded graphs). Whether a source is fresh is in the names (none when stale)."""
    out = {}
    for path, name in named.items():
        if isinstance(name, _TensorStamp):
            out[path] = ("static", id(name.ref()), name.version)
        elif isinstance(name, tuple) and name[:1] == ("diagnostic",):
            out[path] = ("diagnostic", tuple((layout, keys) for layout, _version, keys in name[1]))
        else:
            out[path] = name
    return out


def restamp(boundary: Mapping[str, Any], named: Mapping[str, Any]) -> dict[str, Any]:
    """Receipts of the boundary's current tensors and versions, as an ordinary step leaves them."""
    stamps = {}
    for path, name in named.items():
        if isinstance(name, _TensorStamp):
            stamps[path] = name
        elif isinstance(name, tuple) and name[:1] == ("diagnostic",):
            mapping = _diagnostic_mapping(boundary, name[2])
            _rebind_diagnostic_solve_receipt(mapping, {DIAGNOSTIC_SOLVE_SOURCE_RECEIPT_KEY: name[1]})
            stamps[path] = mapping[DIAGNOSTIC_SOLVE_SOURCE_RECEIPT_KEY]
        else:
            tensor = boundary[name]
            stamps[path] = _TensorStamp(weakref.ref(tensor), tensor.data_ptr(), tensor.storage_offset(),
                                        tuple(tensor.stride()), tensor._version)
    return stamps


def _finite_paths(boundary: Mapping[str, Any]) -> tuple[str, ...]:
    """Only outputs whose current values passed the dycore's finite check."""
    return tuple(path for path, value in boundary.items()
                 if isinstance(value, torch.Tensor) and _finite_stamp_matches(value))


def _restamp_finite(boundary: Mapping[str, Any], paths: Iterable[str]) -> None:
    # Copy-back changes identities/versions, not the checked values. A recorded graph's
    # output assertions run on every replay, so the same proof follows its returned boundary.
    for path in paths:
        _record_finite_stamp(boundary[path])


class _EmulatedGraph:
    """CPU stand-in for TimestepGraph with the same boundary rules: the step reruns eagerly.

    It checks the driver's host logic (boundary, copy-back, receipts) where no CUDA graph exists.
    """

    def __init__(self, step, device, **_unused):
        self._step = step
        self._recorded: dict[str, Any] | None = None

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        if self._recorded is None:
            self._recorded = dict(state)
        elif any(state[k] is not v for k, v in self._recorded.items() if isinstance(v, torch.Tensor)):
            raise RuntimeError("step graph input is not the boundary its replay returned")
        output = self._step(dict(state))
        destinations, sources = _copy_plan(state, output)
        if destinations:
            torch._foreach_copy_(list(destinations), list(sources))
        return dict(self._recorded)


def _extent_bytes(value: torch.Tensor) -> int:
    """Bytes from the start of ``value``'s storage through its last element."""
    last = value.storage_offset() + (1 + sum((size - 1) * stride for size, stride in zip(value.size(), value.stride()))
                                     if value.numel() else 0)
    return last * value.element_size()


def _clone_keeping_views(tensors: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    """Copies of ``tensors`` (by id) with their layouts and storage sharing: each shared storage is copied
    once (from its start through the last element any of them reaches) and every tensor is rebuilt on
    the copy with its size, stride, dtype and storage offset. ``clone()`` of a strided view
    (``qv = scalars[..., 0]``) returns a contiguous tensor apart from ``scalars``, and the boundary,
    whose signature holds strides and sharing, would no longer be steady. Tensors that share a storage
    without one being a view of the other (``set_``) are covered too: a view's ``_base`` need not span
    the storage."""
    groups: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for i, value in tensors.items():
        groups.setdefault(value.untyped_storage().data_ptr(), []).append((i, value))
    copies = {}
    for members in groups.values():
        first = members[0][1]
        end = max(_extent_bytes(value) for _, value in members)
        source = torch.empty(0, dtype=torch.uint8, device=first.device).set_(first.untyped_storage(), 0, (end,), (1,))
        buffer = torch.empty(-(-end // 16) * 16, dtype=torch.uint8, device=first.device)
        buffer[:end].copy_(source)
        for i, value in members:
            copies[i] = buffer.view(value.dtype).as_strided(value.size(), value.stride(), value.storage_offset())
    return copies


def _written_paths(boundary: Mapping[str, Any], output: Mapping[str, Any], versions: Mapping[str, int | None]
                   ) -> set[str]:
    """Boundary tensors an ordinary step wrote: replaced in its output (the copy-back writes them) or
    written in place (version changed; an inference tensor has no version and counts as written), with
    every boundary tensor that shares storage with a written one."""
    written = {k for k, v in boundary.items() if isinstance(v, torch.Tensor)
               and (output[k] is not v or versions[k] is None or v._version != versions[k])}
    shared = {boundary[k].untyped_storage().data_ptr() for k in written}
    return written | {k for k, v in boundary.items()
                      if isinstance(v, torch.Tensor) and v.untyped_storage().data_ptr() in shared}


def _versions(boundary: Mapping[str, Any]) -> dict[str, int | None]:
    return {k: None if v.is_inference() else v._version for k, v in boundary.items() if isinstance(v, torch.Tensor)}


class StepGraphRunner:
    """Step graphs kept across calls: one runner advances a forecast interval by interval.

    The runner holds the recorded graphs, their receipt names, the steady step variants, the boundary
    and the window restore buffers. A call that receives the state the runner returned last (same
    tensors, receipts and step) continues on that boundary: no ordinary step, no recording and no
    boundary copy. Any other state starts the runner afresh (as ``run_step_graphs``). The returned
    state is the runner's boundary: the next call advances it in place, so a caller that keeps
    values across calls copies them first.

    ``graph`` builds a recorder with TimestepGraph's interface (default: TimestepGraph on CUDA, the
    eager emulator elsewhere). ``donated`` names tensors of the input state that the caller hands over:
    the runner writes them in place. Every other input tensor belongs to the caller and is copied before
    its first write. The runner keeps no reference to the donated tensors beyond its boundary. ``events`` lists (step, action) with action ordinary, recorded, replayed
    or overflow (the window of that step reruns); ``restore`` counts the bytes the window restore points
    copy and the bytes a copy of every boundary tensor would take.
    """

    def __init__(self, *, graph: Callable | None = None, donated: Iterable[torch.Tensor] = (), **arguments):
        self.arguments = arguments
        self._graph = graph
        self._donated = tuple(donated)
        self._last: CoupledCycleState | None = None
        self.events: list[tuple[int, str]] = []
        self.restore = {"windows": 0, "bytes": 0, "full_bytes": 0}
        self._start(None)

    def _start(self, previous: CoupledCycleState | None) -> None:
        self.graphs: dict[tuple, Any] = {}
        self.named: dict[tuple, dict[str, Any]] = {}
        self.finite: dict[tuple, tuple[str, ...]] = {}
        self.steady: set[tuple] = set()  # keys whose ordinary step left the boundary steady
        # Per steady key, the boundary tensors its ordinary step read without writing (by path).
        self.read_only: dict[tuple, dict[str, torch.Tensor]] = {}
        self._saved: dict[str, torch.Tensor] = {}  # window restore buffers by path, reused
        self.window: dict[str, Any] | None = None
        if previous is None:
            return
        self.template, self.step = previous, previous.step_index
        self.boundary, self.stamps = flatten_state(previous)
        # The caller's tensors (the input state, and whatever it shares, such as a case's initial fields)
        # are never written: before the first copy-back into one of them the driver takes its own copy.
        self.caller_owned = ({id(v) for v in self.boundary.values() if isinstance(v, torch.Tensor)}
                             - {id(v) for v in self._donated})
        self._donated = ()  # handed over with the first state: the runner keeps no reference to them
        self.live = {"stamps": self.stamps}  # the receipts a recording (or the emulator's rerun) reads

    def _continues(self, previous: CoupledCycleState) -> bool:
        """``previous`` is the state this runner returned last, unchanged."""
        if self._last is None or previous is not self._last or previous.step_index != self.step:
            return False
        leaves, stamps = flatten_state(previous)
        return (leaves.keys() == self.boundary.keys() and stamps.keys() == self.stamps.keys()
                and all(v is self.boundary[k] if isinstance(v, torch.Tensor) else v == self.boundary[k]
                        for k, v in leaves.items())
                and all(v is self.stamps[k] for k, v in stamps.items()))

    def _capacity_raised(self) -> None:
        """Buffer shapes changed: run every variant as an ordinary step before it records again, so
        regions that read a raised capacity compile outside a CUDA-graph capture. The next step is
        therefore ordinary, and it drops the graphs first (``_drop_graphs``)."""
        self.steady.clear()

    def _drop_graphs(self, device: torch.device) -> None:
        """Drop the recorded graphs before an ordinary step: a graph keeps its private pool reserved,
        and an ordinary step beside it would need the eager peak on top of that reserve. Each variant
        records again at its next step (once per variant: after every variant ran an ordinary step,
        none runs again unless the capacity is raised). On CUDA the freed pool returns to the device."""
        if not self.graphs:
            return
        self.graphs.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def _restore_paths(self) -> list[str]:
        """Boundary tensors a window's replays may write (every steady key's step is considered), one
        path per tensor, views of a saved tensor left to it."""
        read_only = None
        for key in self.steady:
            found = {k for k, v in self.read_only[key].items() if self.boundary.get(k) is v}
            read_only = found if read_only is None else read_only & found
        read_only = read_only or set()
        chosen: dict[int, str] = {}
        for k, v in self.boundary.items():
            if isinstance(v, torch.Tensor) and k not in read_only:
                chosen.setdefault(id(v), k)
        return [k for i, k in chosen.items()
                if not (self.boundary[k]._base is not None and id(self.boundary[k]._base) in chosen)]

    def _open_window(self, step: int, key: tuple, device: torch.device) -> None:
        reset_overflow(device)
        cuda = device.type == "cuda"
        paths = self._restore_paths()
        saved = {}
        for k in paths:
            value, buffer = self.boundary[k], self._saved.get(k)
            if buffer is None or buffer.size() != value.size() or buffer.dtype != value.dtype:
                # On CUDA the restore point is pinned host memory (device memory of large meshes).
                buffer = torch.empty(value.size(), dtype=value.dtype, device="cpu" if cuda else value.device,
                                     pin_memory=cuda)
            saved[k] = buffer.copy_(value, non_blocking=cuda)
        self._saved = saved
        # Tensors outside the restore point (no saved path holds them or their base): no replay may write them.
        held = {id(self.boundary[k]) for k in saved}
        excluded = {k: v for k, v in self.boundary.items() if isinstance(v, torch.Tensor) and id(v) not in held
                    and (v._base is None or id(v._base) not in held)}
        self.window = {"step": step, "key": key, "excluded": excluded, "versions": _versions(excluded),
                       "finite": _finite_paths(self.boundary)}
        self.restore["windows"] += 1
        self.restore["bytes"] += sum(b.numel() * b.element_size() for b in saved.values())
        self.restore["full_bytes"] += sum(v.numel() * v.element_size() for v in self.boundary.values()
                                          if isinstance(v, torch.Tensor))

    def _window_overflowed(self, device: torch.device, exchanger: Any) -> bool:
        if self.window is None:
            return False
        start, self.window = self.window, None
        if _versions(start["excluded"]) != start["versions"]:
            raise RuntimeError("a step graph wrote a boundary tensor left out of its restore point")
        regions = consensus_regions(overflowed_regions(device), exchanger)
        if not regions:
            return False
        for name in regions:
            COMPACTION_LOG.append((start["step"], raise_compaction(name)))
        cuda = device.type == "cuda"
        for k, v in self._saved.items():
            self.boundary[k].copy_(v, non_blocking=cuda)
        _restamp_finite(self.boundary, start["finite"])
        self.stamps = self.live["stamps"] = restamp(self.boundary, self.named[start["key"]])  # new versions
        self._capacity_raised()
        self.events.append((start["step"], "overflow"))
        self.step = start["step"]
        return True

    def advance(self, previous: CoupledCycleState, *, duration_seconds: int) -> CoupledCycleState:
        """Advance ``duration_seconds`` with one recorded graph per step variant; returns the last state."""
        arguments = self.arguments
        if not step_graphs_enabled():
            raise RuntimeError("step graphs need runtime_policy.set_step_graphs(True)")
        dt = arguments["config"]["config_dt"]
        if type(duration_seconds) is not int or duration_seconds <= 0 or duration_seconds % dt:
            raise ValueError("step graphs advance whole positive dycore timesteps")
        if torch.is_grad_enabled() or forward_ad._current_level >= 0:
            # Replays and their copy-back are invisible to autograd and forward AD.
            return run_coupled_interval(previous, duration_seconds=duration_seconds, on_overflow="rerun",
                                        **arguments).state
        device = previous.sec.device
        graph = self._graph
        if graph is None:
            graph = TimestepGraph if device.type == "cuda" else _EmulatedGraph
        if not self._continues(previous):
            self._start(previous)
        # The runner's boundary and template hold what the steps need; a cold state the first step
        # replaces is freed once the caller drops it too (no copy is held across a capture).
        previous = self._last = None
        end_step = self.step + int(duration_seconds // dt)
        # Replays accumulate the device overflow flag; it is read (one host sync) every
        # overflow_check_steps() replays, or before an ordinary step or a recording. On overflow the
        # boundary returns to the window's restore point and the window reruns one rung up: compaction is
        # exact at every rung, so the result equals the per-step protocol's bit for bit.
        # On a partition every rank takes each host decision of this driver together (overflow, steady).
        exchanger = arguments["config"].get("_halo_exchanger")
        replays = 0
        while self.step < end_step:
            step = self.step
            key = step_variant(step, **arguments)
            if key not in self.steady:
                if self._window_overflowed(device, exchanger):
                    continue
                # An ordinary step; a graph is recorded only for a key whose step is steady.
                self._drop_graphs(device)
                reset_overflow(device)
                versions = _versions(self.boundary)
                result = None  # the previous step's intermediates (template keeps its state) before this one runs
                result = run_coupled_cycle(unflatten_state(self.template, {**self.boundary, **self.stamps}, step),
                                           **arguments)
                COUNTS["ordinary"] += 1
                self.events.append((step, "ordinary"))
                regions = consensus_regions(overflowed_regions(device), exchanger)
                if regions:
                    for name in regions:
                        COMPACTION_LOG.append((step, raise_compaction(name)))
                    self._capacity_raised()
                    self.events.append((step, "overflow"))
                    continue
                output, out_stamps = flatten_state(result.state)
                finite = _finite_paths(output)
                out_named = stamp_names(output, out_stamps)
                if all_ranks(not boundary_changes(self.boundary, output) and (
                        _comparable(stamp_names(self.boundary, self.stamps)) == _comparable(out_named)), exchanger):
                    # Steady: the step's values go into the boundary's own tensors, which the
                    # recorded graphs read, and the receipts name them.
                    self.steady.add(key)
                    written_paths = _written_paths(self.boundary, output, versions)
                    self.read_only[key] = {k: v for k, v in self.boundary.items()
                                           if isinstance(v, torch.Tensor) and k not in written_paths}
                    written = {id(v): v for k, v in self.boundary.items()
                               if isinstance(v, torch.Tensor) and id(v) in self.caller_owned and output[k] is not v}
                    if written:
                        # Tensors sharing storage with a replaced one are replaced with it (views of one copy).
                        shared = {v.untyped_storage().data_ptr() for v in written.values()}
                        written.update({id(v): v for v in self.boundary.values() if isinstance(v, torch.Tensor)
                                        and v.untyped_storage().data_ptr() in shared})
                        copies = _clone_keeping_views(written)
                        self.boundary = {k: copies.get(id(v), v) if isinstance(v, torch.Tensor) else v
                                         for k, v in self.boundary.items()}
                        self.caller_owned -= set(copies)
                        self.graphs.clear()  # recorded graphs read the replaced tensors
                    destinations, sources = _copy_plan(self.boundary, output)
                    if destinations:
                        torch._foreach_copy_(list(destinations), list(sources))
                    self.stamps = restamp(self.boundary, out_named)
                else:
                    self.boundary, self.stamps, self.template = output, out_stamps, result.state
                    self.graphs.clear()  # they read the previous boundary's tensors
                _restamp_finite(self.boundary, finite)
                self.live["stamps"] = self.stamps
                # The step's intermediates and outputs (copied back) are not needed by a recording or a
                # replay: nothing of them is held across the next capture.
                result = output = out_stamps = out_named = destinations = sources = written = copies = None
                self.step += 1
                continue
            # Per-run data the step reads (the radiation seed stream) is prepared here, outside the
            # graph: a capture must not record its preparation, and a replay runs no Python.
            prepare_step(self.boundary, step, **arguments)
            if key not in self.graphs:
                if self._window_overflowed(device, exchanger):
                    continue
                def one_step(state, _step=step, _key=key):
                    result = run_coupled_cycle(unflatten_state(self.template, {**state, **self.live["stamps"]}, _step),
                                               **arguments)
                    output, out_stamps = flatten_state(result.state)
                    self.finite[_key] = _finite_paths(output)
                    self.named[_key] = stamp_names(output, out_stamps)
                    return output
                self.graphs[key] = graph(one_step, device)
                COUNTS["recorded"] += 1
                self.events.append((step, "recorded"))
            if self.window is None:
                self._open_window(step, key, device)
                replays = 0
            # cost_meter: the replay is timed whole (the first one also records); off, a null context.
            with frozen_seed_stream(), cost_meter.replay(step, recorded=self.events[-1:] == [(step, "recorded")]):
                self.boundary = self.graphs[key](self.boundary)
            _restamp_finite(self.boundary, self.finite[key])
            COUNTS["replayed"] += 1
            self.events.append((step, "replayed"))
            self.stamps = self.live["stamps"] = restamp(self.boundary, self.named[key])
            replays += 1
            self.step += 1
            if replays >= overflow_check_steps() or self.step == end_step:
                self._window_overflowed(device, exchanger)
        self._last = unflatten_state(self.template, {**self.boundary, **self.stamps}, self.step)
        return self._last


def run_step_graphs(previous: CoupledCycleState, *, duration_seconds: int, graph: Callable | None = None,
                    **arguments) -> CoupledCycleState:
    """Advance ``duration_seconds`` with one recorded graph per step variant; returns the last state.

    One call of a fresh ``StepGraphRunner``: every call runs an ordinary step and records a graph per
    variant. A driver that advances one state over several calls keeps one runner instead.
    ``graph`` builds a recorder with TimestepGraph's interface (default: TimestepGraph on CUDA, the
    eager emulator elsewhere).
    """
    return StepGraphRunner(graph=graph, **arguments).advance(previous, duration_seconds=duration_seconds)
