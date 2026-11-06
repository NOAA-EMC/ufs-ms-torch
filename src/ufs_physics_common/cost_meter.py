# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Cost meter: where compile time and compiled-step time go, per compiled entry, glue block and dycore region.

Off by default (``METER is None``). Off, every hook is one global read and a branch in Python outside
compiled code: no tensor op, no compiled graph, guard or recompile changes.
``enable()`` turns it on for the process; ``report()`` / ``write()`` give the JSON and a short text table.

Hooks (each existing single entry point, nothing wrapped call site by call site):

* ``compile_policy.compiled()``: every compiled physics region and glue block (and its eager first call) is a
  region named by the function. Calls made while a region is traced inline as before and are not hooks.
* ``cycle.run_coupled_cycle`` (``step``): one coupled step, kind ``first``, ``radiation`` (SW or LW step) or
  ``ordinary``. ``dynamics.advance_physics_dynamics`` (``region("dycore_step")``): the dycore step.
* The dycore's compiled regions: mpas-pytorch's compile event sink (``compile_utils.set_compile_event_sink``),
  which the dycore calls at the start and end of every compiled-region call when a sink is set.
* ``step_graphs.StepGraphRunner`` (``replay``): a step-graph replay (the first one records the graph). Inside a
  replay the regions are not timed (a CUDA capture must not record timing events; the CPU emulator follows the
  same rule): the replay is timed as a whole.

Execution clocks: ``time.perf_counter`` at region boundaries (host), and on CUDA also a pair of timing events per
region recorded on the current stream (device). Events are read once in ``report()`` (one synchronize at the end
of the run), so the meter adds no host synchronization inside a step. A region's self time is its time minus its
timed children; shares are self time over the step time.
MPS regions measure host dispatch only, not completed GPU execution. Use synchronized
outer workload timings or a separate Metal trace for MPS execution costs.

Compile clocks come from torch 2.8's own instrumentation, attributed per compile id:

* ``torch._dynamo.callback_handler`` start/end callbacks bracket each compile (trigger DYNAMO: inside
  ``_compile.compile_inner``; LAZY_BACKWARD: a deferred backward compile; TRITON_AUTOTUNING:
  ``CachingAutotuner.benchmark_all_configs`` at a kernel's first launch). The meter takes the
  ``torch._dynamo.utils.compilation_time_metrics`` entries (``dynamo_timed`` events) appended inside the bracket,
  and the bracket's outer event (``_compile.compile_inner``, ``backward._backward_impl`` or
  ``CachingAutotuner.benchmark_all_configs``, appended just after its end callback) in order. The entry is the
  innermost timed region when the compile started (the compiled function, or the dycore region).
* ``torch._dynamo.utils.get_compilation_metrics()``: co_name, recompile reason, guard and graph-op counts per
  compile id. ``counters["inductor"]`` / ``counters["aot_autograd"]``: FX-graph, AOT, Triton-kernel and generated
  module cache hits and misses inside the bracket.
* Coordinate-descent tuning is not a ``dynamo_timed`` event in torch 2.8 (its kernel benchmarks are, through
  ``Benchmarker.benchmark_gpu``, but not the compiles of the configs it tries), so the meter times
  ``CachingAutotuner.coordinate_descent_tuning`` itself while it is on. It runs at a kernel's first launch.

``phase_split`` turns ``dynamo_timed`` sums into exclusive phases. Nesting in torch 2.8.0 (checked by probes):
``_compile.compile_inner`` holds ``bytecode_tracing`` (it stops before the graph is compiled),
``OutputGraph.call_user_compiler`` (the backend) and ``build_guards``; the backend holds
``create_aot_dispatcher_function``, which holds ``compile_fx_inner`` (Inductor, forward and backward); Inductor
holds ``_recursive_post_grad_passes`` and ``GraphLowering.run`` (lowering), ``GraphLowering.codegen`` (scheduler
with fusion, kernel and wrapper code generation) and ``PyCodeCache.load_by_key_path`` (module load and build).
An AOT cache hit instead restores modules under ``AOTAutogradCache.inductor_load``, outside the AOT dispatcher
and ``compile_fx_inner``. Worker ``compile_file`` timers overlap ``async_compile.wait``; their sums cannot
split module-load wall time into exclusive build and load phases. Keep that phase combined and expose raw,
nonexclusive build timers only in ``detail``. The exclusive phases add up to the outer events exactly;
a negative phase means the nesting differs (listed under ``nesting_violations``).

What the meter does not time: Triton launcher precompiles at a first launch outside autotuning, and work in
compile worker processes beyond the parent's wait.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch

METER: "Meter | None" = None
_NULL = contextlib.nullcontext()

# Outer dynamo_timed event of each callback trigger; appended just after the trigger's end callback.
_OUTER = {"DYNAMO": "_compile.compile_inner", "LAZY_BACKWARD": "backward._backward_impl",
          "TRITON_AUTOTUNING": "CachingAutotuner.benchmark_all_configs"}
_CACHE_COUNTERS = {
    "inductor": ("fxgraph_cache_hit", "fxgraph_cache_miss", "fxgraph_cache_bypass", "async_compile_cache_hit",
                 "async_compile_cache_miss", "triton_bundler_read_and_emit_kernel", "generated_module_cache_hit",
                 "generated_module_cache_miss"),
    "aot_autograd": ("autograd_cache_hit", "autograd_cache_miss", "autograd_cache_bypass",
                     "autograd_cache_guard_miss")}
COMPILE_PHASES = ("dynamo_bytecode_tracing", "dynamo_guards", "dynamo_other", "backend_other", "aot_autograd",
                  "inductor_lowering", "inductor_scheduling_codegen", "inductor_load_and_build", "inductor_other")
RUNTIME_PHASES = ("autotune", "coordinate_descent")


def phase_split(sums: dict[str, float]) -> dict[str, Any]:
    """Exclusive phases (seconds) from ``dynamo_timed`` sums (key -> seconds); see the module docstring."""
    s = lambda *keys: float(sum(sums.get(k, 0.0) for k in keys))  # noqa: E731
    frame, backward = s("_compile.compile_inner"), s("backward._backward_impl")
    trace, guards, backend = s("bytecode_tracing"), s("build_guards"), s("OutputGraph.call_user_compiler")
    aot, inductor = s("create_aot_dispatcher_function"), s("compile_fx_inner")
    lowering = s("GraphLowering.run", "_recursive_post_grad_passes")
    codegen, load = s("GraphLowering.codegen"), s("PyCodeCache.load_by_key_path")
    cache_load = s("AOTAutogradCache.inductor_load")
    build = s("compile_file", "async_compile.precompile", "async_compile.wait", "CachingAutotuner.slow_precompile_config")
    phases = {"dynamo_bytecode_tracing": trace, "dynamo_guards": guards,
              "dynamo_other": frame - trace - guards - backend, "backend_other": backend - aot - cache_load,
              "aot_autograd": aot + backward - inductor, "inductor_lowering": lowering,
              "inductor_scheduling_codegen": codegen, "inductor_load_and_build": load,
              "inductor_other": inductor + cache_load - lowering - codegen - load,
              "autotune": s("CachingAutotuner.benchmark_all_configs"), "coordinate_descent": s("coordinate_descent_tuning")}
    return {"compile_s": frame + backward, "phases": phases,
            "detail": {"scheduler_init_s": s("Scheduler.__init__"), "scheduler_codegen_s": s("Scheduler.codegen"),
                       "fused_nodes_s": s("Scheduler.fused_nodes"), "wrapper_generate_s": s("PythonWrapperCodegen.generate"),
                       "fx_codegen_and_compile_s": s("fx_codegen_and_compile"), "aot_cache_load_s": cache_load,
                       "build_timer_sum_nonexclusive_s": build,
                       "kernel_benchmarks_s": s("TritonBenchmarker.benchmark_gpu", "InductorBenchmarker.benchmark_gpu")},
            "nesting_violations": sorted(k for k, v in phases.items() if v < -1e-6)}


def process_phase_split() -> dict[str, Any]:
    """``phase_split`` of every ``dynamo_timed`` event of this process (no meter needed)."""
    from torch._dynamo.utils import compilation_time_metrics

    return phase_split({k: float(sum(v)) for k, v in compilation_time_metrics.items()})


def cache_counters() -> dict[str, int]:
    """FX-graph, AOT, Triton-kernel and generated-module cache counters of this process."""
    from torch._dynamo.utils import counters

    return {name: int(counters[group][name]) for group, names in _CACHE_COUNTERS.items() for name in names}


class Meter:
    """The process's meter; make it with ``enable()``."""

    def __init__(self, device=None, *, dycore_regions: bool = True):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.cuda = self.device.type == "cuda"
        self.regions: list[list] = []  # [name, parent, step ordinal, host start, host end, event start, event end]
        self.stack: list[int] = []
        self.steps: list[dict] = []
        self._partial_steps: dict[tuple, dict] = {}
        self.replays: list[dict] = []
        self.cycles: list[dict] = []
        self.compiles: list[dict] = []
        self.autotunes: list[dict] = []
        self.runtime: list[dict] = []  # dynamo_timed events and tuning outside a compile bracket
        self.cycle_label = None
        self.step_ordinal = None
        self.suspended = 0
        self.overhead_s = 0.0
        self._window = None
        self._pending = {trigger: [] for trigger in _OUTER}
        self._cursor: dict[str, int] = {}
        self._dirty = False
        self._seen_metrics: set = set()
        self._dycore_calls: dict[int, bool] = {}
        self._restore: list[Callable[[], None]] = []
        self._dycore_regions = dycore_regions

    # ------------------------------------------------------------------ execution regions
    def _clock(self):
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return None

    def enter(self, name: str) -> bool:
        """Open a region; False (nothing opened) inside a replay. Pair a True with ``exit()``."""
        if self.suspended:
            return False
        begin = time.perf_counter()
        self.regions.append([name, self.stack[-1] if self.stack else None, self.step_ordinal, 0.0, 0.0,
                             self._clock(), None])
        self.stack.append(len(self.regions) - 1)
        now = time.perf_counter()
        self.regions[-1][3] = now
        self.overhead_s += now - begin
        return True

    def exit(self) -> None:
        now = time.perf_counter()
        record = self.regions[self.stack.pop()]
        record[4] = now
        record[6] = self._clock()
        if self._dirty:
            self._sweep(record[0])
        self.overhead_s += time.perf_counter() - now

    @contextlib.contextmanager
    def region(self, name: str) -> Iterator[None]:
        timed = self.enter(name)
        try:
            yield
        finally:
            if timed:
                self.exit()

    @contextlib.contextmanager
    def step(self, step_index: int, kind: Callable[[], str], *, part: str | None = None) -> Iterator[None]:
        """Time a whole step, or pair its head and tail across a native-state boundary."""
        if part not in (None, "head", "tail"):
            raise ValueError("step part must be head, tail or None")
        if self.suspended:
            yield
            return
        key = (self.cycle_label, int(step_index))
        record = self._partial_steps.pop(key, None) if part == "tail" else None
        if record is None:
            record = {"ordinal": len(self.steps), "step_index": int(step_index), "kind": kind(),
                      "cycle": self.cycle_label, "segments": [], "parts": []}
            self.steps.append(record)
        if part == "head":
            self._partial_steps[key] = record
        segment = [len(self.regions), None]
        record["segments"].append(segment)
        record["parts"].append(part)
        outer, self.step_ordinal = self.step_ordinal, record["ordinal"]
        timed = self.enter("<step>")
        try:
            yield
        finally:
            if timed:
                self.exit()
            self.step_ordinal = outer
            segment[1] = len(self.regions)

    @contextlib.contextmanager
    def replay(self, step_index: int, recorded: bool) -> Iterator[None]:
        begin, start = time.perf_counter(), self._clock()
        self.suspended += 1
        try:
            yield
        finally:
            self.suspended -= 1
            self.replays.append({"step_index": int(step_index), "recorded": bool(recorded), "cycle": self.cycle_label,
                                 "host_s": time.perf_counter() - begin, "events": (start, self._clock())})

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        """No step or region timing inside (e.g. a profiled cycle); compiles are still attributed."""
        self.suspended += 1
        try:
            yield
        finally:
            self.suspended -= 1

    @contextlib.contextmanager
    def cycle(self, label) -> Iterator[None]:
        """A driver's cycle (interval); its host wall time is the denominator of the compile accounting. Drivers
        that time a CUDA cycle synchronize inside it, as they do for their own clocks."""
        previous, self.cycle_label = self.cycle_label, label
        record = {"label": label, "first_compile": len(self.compiles)}
        begin = time.perf_counter()
        try:
            yield
        finally:
            if self._dirty:
                self._sweep(None)
            record["wall_s"] = time.perf_counter() - begin
            self.cycles.append(record)
            self.cycle_label = previous

    def _dycore_event(self, event: dict) -> None:
        phase = event.get("phase")
        if phase == "compiled_call_start":
            self._dycore_calls[event["call_id"]] = self.enter(str(event.get("fn_name")))
        elif phase in ("compiled_call_success", "compiled_call_error", "compiled_call_checkpoint_stop"):
            if self._dycore_calls.pop(event.get("call_id"), False):
                self.exit()

    # ------------------------------------------------------------------ compile accounting
    def _owner(self):
        return self.regions[self.stack[-1]][0] if self.stack else None

    def _sweep(self, owner) -> None:
        """Take the dynamo_timed entries appended since the last sweep: outer events to their pending brackets
        (in order), the rest to the open bracket or else to the runtime list under ``owner``."""
        from torch._dynamo.utils import compilation_time_metrics

        loose: dict[str, float] = {}
        for key, values in compilation_time_metrics.items():
            start = self._cursor.get(key, 0)
            if len(values) <= start:
                continue
            self._cursor[key] = len(values)
            new = values[start:]
            trigger = next((t for t, k in _OUTER.items() if k == key), None)
            while trigger is not None and new and self._pending[trigger]:
                window = self._pending[trigger].pop(0)
                window["outer_s"] = new.pop(0)
                window["sums"][key] = window["outer_s"]
            if new:
                target = self._window["sums"] if self._window is not None else loose
                target[key] = target.get(key, 0.0) + float(sum(new))
        if loose:
            self._runtime(owner, loose)
        self._read_metrics()
        self._dirty = self._window is not None or any(self._pending.values())

    def _runtime(self, owner, sums: dict[str, float]) -> None:
        self.runtime.append({"entry": owner, "cycle": self.cycle_label, "step": self.step_ordinal, "sums": sums})

    def _read_metrics(self) -> None:
        from torch._dynamo.utils import get_compilation_metrics

        by_id = {}
        for window in self.compiles:
            if window["trigger"] != "TRITON_AUTOTUNING" and "co_name" not in window:
                by_id.setdefault(window["compile_id"], window)
        for record in get_compilation_metrics():
            key = (str(record.compile_id), record.start_time_us, record.is_runtime)
            if key in self._seen_metrics:
                continue
            self._seen_metrics.add(key)
            window = by_id.pop(str(record.compile_id), None)
            if window is None or record.is_runtime:
                continue
            window.update(co_name=record.co_name, recompile_reason=record.recompile_reason,
                          guard_count=record.guard_count, graph_op_count=record.graph_op_count,
                          fail_type=record.fail_type)

    def _start(self, args) -> None:
        begin = time.perf_counter()
        self._sweep(self._owner())
        self._window = {"compile_id": str(args.compile_id), "trigger": args.callback_trigger.name,
                        "entry": self._owner(), "cycle": self.cycle_label, "step": self.step_ordinal,
                        "step_index": self.steps[self.step_ordinal]["step_index"] if self.step_ordinal is not None else None,
                        "sums": {}, "cache_before": cache_counters(), "start": time.perf_counter()}
        self._dirty = True
        self.overhead_s += time.perf_counter() - begin

    def _end(self, args) -> None:
        now = time.perf_counter()
        window = self._window
        if window is None:
            return
        self._sweep(window["entry"])
        window["window_s"] = now - window.pop("start")
        before = window.pop("cache_before")
        window["cache"] = {k: v - before[k] for k, v in cache_counters().items() if v != before[k]}
        self._window = None
        # A kernel's first-launch autotuning is runtime tuning of the region that launched it, not a compile.
        (self.autotunes if window["trigger"] == "TRITON_AUTOTUNING" else self.compiles).append(window)
        self._pending[window["trigger"]].append(window)
        self._dirty = True
        self.overhead_s += time.perf_counter() - now

    def _coordinate_descent(self, original):
        meter = self

        def coordinate_descent_tuning(tuner, *args, **kwargs):
            begin = time.perf_counter()
            try:
                return original(tuner, *args, **kwargs)
            finally:
                meter._runtime(meter._owner(), {"coordinate_descent_tuning": time.perf_counter() - begin})
        return coordinate_descent_tuning

    def install(self) -> None:
        from torch._dynamo.callback import callback_handler
        from torch._dynamo.utils import compilation_time_metrics

        self._cursor = {k: len(v) for k, v in compilation_time_metrics.items()}
        callback_handler.register_start_callback(self._start)
        callback_handler.register_end_callback(self._end)


        def remove():  # torch._dynamo.reset() clears the handler's callbacks too
            if self._start in callback_handler.start_callbacks:
                callback_handler.remove_start_callback(self._start)
            if self._end in callback_handler.end_callbacks:
                callback_handler.remove_end_callback(self._end)
        self._restore.append(remove)
        try:
            from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        except ImportError:
            CachingAutotuner = None
        if CachingAutotuner is not None:
            original = CachingAutotuner.coordinate_descent_tuning
            CachingAutotuner.coordinate_descent_tuning = self._coordinate_descent(original)
            self._restore.append(lambda: setattr(CachingAutotuner, "coordinate_descent_tuning", original))
        self.dycore_sink = False
        if self._dycore_regions:
            try:
                from mpas_pytorch.dynamics import compile_utils
            except ImportError:
                compile_utils = None
            if compile_utils is not None:
                previous = compile_utils._COMPILE_EVENT_SINK

                def sink(event, _previous=previous):
                    self._dycore_event(event)
                    if _previous is not None:
                        _previous(event)
                compile_utils.set_compile_event_sink(sink)
                self._restore.append(lambda: compile_utils.set_compile_event_sink(previous))
                self.dycore_sink = True

    def uninstall(self) -> None:
        while self._restore:
            self._restore.pop()()

    # ------------------------------------------------------------------ report
    def _seconds(self, start, end, host: float) -> float:
        return start.elapsed_time(end) / 1e3 if self.cuda and start is not None and end is not None else host

    def report(self) -> dict[str, Any]:
        """The measurements so far (one device synchronize on CUDA, here, after the run)."""
        if self._dirty:
            self._sweep(None)
        if self.cuda:
            torch.cuda.synchronize(self.device)
        n = len(self.regions)
        host = [r[4] - r[3] for r in self.regions]
        total = [self._seconds(r[5], r[6], host[i]) for i, r in enumerate(self.regions)]
        child = [0.0] * n
        child_host = [0.0] * n
        for i, r in enumerate(self.regions):
            if r[1] is not None:
                child[r[1]] += total[i]
                child_host[r[1]] += host[i]
        compiles = []
        for w in self.compiles:
            split = phase_split(w["sums"])
            compiles.append({k: w.get(k) for k in ("compile_id", "trigger", "entry", "co_name", "cycle", "step",
                                                    "step_index", "recompile_reason", "guard_count", "graph_op_count",
                                                    "fail_type", "window_s", "outer_s", "cache")}
                            | {"compile_s": split["compile_s"], "phases": split["phases"], "detail": split["detail"],
                               "nesting_violations": split["nesting_violations"],
                               "callback_gap_s": None if w.get("outer_s") is None else w["outer_s"] - w["window_s"]})
        runtime = []
        for r in [*self.runtime, *({"entry": w["entry"], "cycle": w["cycle"], "step": w["step"], "sums": w["sums"]}
                                   for w in self.autotunes)]:
            p = phase_split(r["sums"])["phases"]
            runtime.append({"entry": r["entry"], "cycle": r["cycle"], "step": r["step"],
                            "autotune_s": p["autotune"], "coordinate_descent_s": p["coordinate_descent"],
                            "other_s": {k: v for k, v in r["sums"].items() if k not in (
                                "CachingAutotuner.benchmark_all_configs", "coordinate_descent_tuning")}})
        by_entry: dict[str, dict] = {}
        for c in compiles:
            e = by_entry.setdefault(_entry_name(c["entry"], c["co_name"]),
                                    {"compiles": 0, "compile_s": 0.0, "phases": Counter(), "cache": Counter(),
                                     "when": [], "recompile_reasons": []})
            e["compiles"] += 1
            e["compile_s"] += c["compile_s"]
            e["phases"].update(c["phases"])
            e["cache"].update(c["cache"] or {})
            e["when"].append([c["cycle"], c["step_index"]])
            if c["recompile_reason"]:
                e["recompile_reasons"].append(c["recompile_reason"][:300])
        for r in runtime:
            e = by_entry.setdefault(_entry_name(r["entry"], None), {"compiles": 0, "compile_s": 0.0, "phases": Counter(),
                                                       "cache": Counter(), "when": [], "recompile_reasons": []})
            e["phases"].update({"autotune": r["autotune_s"], "coordinate_descent": r["coordinate_descent_s"]})
        for e in by_entry.values():
            e["phases"], e["cache"] = dict(e["phases"]), dict(e["cache"])
            e["runtime_tuning_s"] = e["phases"].get("autotune", 0.0) + e["phases"].get("coordinate_descent", 0.0)
        steps = []
        compiling_steps = {c["step"] for c in compiles} | {r["step"] for r in runtime}
        for s in self.steps:
            regions: dict[str, dict] = {}
            roots = []
            for first, end in s["segments"]:
                roots.append(first)  # each segment's "<step>" region
                for i in range(first + 1, n if end is None else end):
                    d = regions.setdefault(self.regions[i][0], {"calls": 0, "self_s": 0.0, "total_s": 0.0, "host_self_s": 0.0})
                    d["calls"] += 1
                    d["self_s"] += total[i] - child[i]
                    d["total_s"] += total[i]
                    d["host_self_s"] += host[i] - child_host[i]
            steps.append({"ordinal": s["ordinal"], "step_index": s["step_index"], "kind": s["kind"], "cycle": s["cycle"],
                          "seconds": sum(total[root] for root in roots),
                          "host_seconds": sum(host[root] for root in roots),
                          "outside_regions_s": sum(total[root] - child[root] for root in roots),
                          "parts": s["parts"],
                          "complete": s["parts"] in ([None], ["head", "tail"]),
                          "compiling": s["ordinal"] in compiling_steps, "regions": regions})
        steady: dict[str, dict] = {}
        for s in steps:
            if s["compiling"] or s["kind"] == "first" or not s["complete"]:
                continue
            g = steady.setdefault(s["kind"], {"steps": 0, "seconds": 0.0, "outside_regions_s": 0.0, "regions": {}})
            g["steps"] += 1
            g["seconds"] += s["seconds"]
            g["outside_regions_s"] += s["outside_regions_s"]
            for name, d in s["regions"].items():
                a = g["regions"].setdefault(name, {"calls": 0, "self_s": 0.0, "host_self_s": 0.0})
                a["calls"] += d["calls"]
                a["self_s"] += d["self_s"]
                a["host_self_s"] += d["host_self_s"]
        for g in steady.values():
            k = g["steps"]
            g["step_s"] = g["seconds"] / k
            g["outside_regions_share"] = g["outside_regions_s"] / g["seconds"]
            g["regions"] = {name: {"calls_per_step": a["calls"] / k, "self_s_per_step": a["self_s"] / k,
                                   "host_self_s_per_step": a["host_self_s"] / k, "share": a["self_s"] / g["seconds"]}
                            for name, a in sorted(g["regions"].items(), key=lambda kv: -kv[1]["self_s"])}
        replays = [{k: v for k, v in r.items() if k != "events"} | {"seconds": self._seconds(*r["events"], r["host_s"])}
                   for r in self.replays]
        timed = sorted(r["seconds"] for r in replays if not r["recorded"])
        replay_summary = {"replays": len(timed), "median_s": timed[len(timed) // 2] if timed else None,
                          "mean_s": sum(timed) / len(timed) if timed else None,
                          "record_and_first_replay_s": [r["seconds"] for r in replays if r["recorded"]]}
        cycles = []
        for c in self.cycles:
            mine = [x for x in compiles if x["cycle"] == c["label"]]
            compile_s = sum(x["compile_s"] for x in mine)
            tuning = sum(x["autotune_s"] + x["coordinate_descent_s"] for x in runtime if x["cycle"] == c["label"])
            in_steps = sum(s["host_seconds"] for s in steps if s["cycle"] == c["label"]) + sum(
                r["host_s"] for r in replays if r["cycle"] == c["label"])
            phases = Counter()
            for x in mine:
                phases.update({k: x["phases"][k] for k in COMPILE_PHASES})
            cycles.append({"label": c["label"], "wall_s": c["wall_s"], "compiles": sum(
                1 for x in mine if x["trigger"] == "DYNAMO"), "compile_s": compile_s, "phases": dict(phases),
                "runtime_tuning_s": tuning, "outside_compile_in_steps_s": in_steps - compile_s - tuning,
                "outside_steps_s": c["wall_s"] - in_steps})
        process_split = process_phase_split()
        # Torch's process timers omit coordinate descent; this meter times it during its own lifetime.
        process_split["phases"]["coordinate_descent"] = sum(r["coordinate_descent_s"] for r in runtime)
        return {"device": str(self.device), "torch": torch.__version__,
                "clock": ("CUDA events (device) and perf_counter (host)" if self.cuda else
                          "perf_counter (host dispatch only; not GPU execution)" if self.device.type == "mps" else
                          "perf_counter (host)"),
                "dycore_regions": bool(getattr(self, "dycore_sink", False)),
                "cycles": cycles, "compiles": compiles, "runtime_tuning": runtime,
                "compile_by_entry": dict(sorted(by_entry.items(), key=lambda kv: -kv[1]["compile_s"])),
                "steps": steps, "steady": steady, "replays": replays, "replay_summary": replay_summary,
                "overhead": {"meter_host_s": self.overhead_s, "region_calls": n,
                             "per_step_s": self.overhead_s / max(1, len(self.steps)),
                             "note": "host seconds inside the meter's own hooks; the dycore sink's payload (built by "
                                     "mpas-pytorch before it calls the sink) is not included: compare on and off runs"},
                "process_split": process_split, "cache": cache_counters()}

    def write(self, directory, *, top: int = 10) -> dict[str, Any]:
        """Write ``cost_meter.json`` and ``cost_meter.txt`` into ``directory``; returns the report."""
        report = self.report()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "cost_meter.json").write_text(json.dumps(report, indent=1, default=str))
        (directory / "cost_meter.txt").write_text(text_table(report, top=top))
        return report


def _entry_name(entry, co_name) -> str:
    """A compile's entry: the innermost timed region, or the compiled code's name outside any region."""
    return str(entry) if entry not in (None, "<step>") else f"[{co_name}]"


def text_table(report: dict[str, Any], *, top: int = 10) -> str:
    """Top entries by compile seconds, compiling cycles' accounting, top regions by step share."""
    lines = [f"cost meter ({report['clock']}, torch {report['torch']})", "",
             f"Compile, top {top} entries by compile seconds (phases exclusive, seconds):"]
    short = {"dynamo_bytecode_tracing": "trace", "dynamo_guards": "guards", "dynamo_other": "dyn.oth",
             "backend_other": "bk.oth", "aot_autograd": "aot", "inductor_lowering": "lower",
             "inductor_scheduling_codegen": "sched+cg", "inductor_load_and_build": "load+bld",
             "inductor_other": "ind.oth", "autotune": "autotune", "coordinate_descent": "coord"}
    head = f"{'entry':44s} {'n':>3s} {'total':>8s} " + " ".join(f"{v:>8s}" for v in short.values()) + "  fx hit/miss  first(c,s)"
    lines.append(head)
    for name, e in list(report["compile_by_entry"].items())[:top]:
        p = e["phases"]
        cache = e["cache"]
        lines.append(f"{name[:44]:44s} {e['compiles']:3d} {e['compile_s']:8.1f} "
                     + " ".join(f"{p.get(k, 0.0):8.1f}" for k in short)
                     + f"  {cache.get('fxgraph_cache_hit', 0):>4d}/{cache.get('fxgraph_cache_miss', 0):<4d}"
                     + f"  {e['when'][0] if e['when'] else '-'}")
    lines += ["", "Compiling cycles (wall = compile phases + runtime tuning + outside compile in steps + outside steps):",
              f"{'cycle':>6s} {'wall':>8s} {'compiles':>8s} {'compile':>8s} {'tuning':>8s} {'outside':>8s} {'remain':>8s}"]
    for c in report["cycles"]:
        if c["compiles"] or c["runtime_tuning_s"]:
            lines.append(f"{str(c['label']):>6s} {c['wall_s']:8.1f} {c['compiles']:8d} {c['compile_s']:8.1f} "
                         f"{c['runtime_tuning_s']:8.1f} {c['outside_compile_in_steps_s']:8.1f} {c['outside_steps_s']:8.2f}")
    for kind, g in report["steady"].items():
        lines += ["", f"Steady {kind} steps ({g['steps']}, {g['step_s']:.3f} s/step), top {top} regions by self share:",
                  f"{'region':60s} {'calls':>6s} {'self s':>8s} {'share':>7s}"]
        for name, r in list(g["regions"].items())[:top]:
            lines.append(f"{name[:60]:60s} {r['calls_per_step']:6.1f} {r['self_s_per_step']:8.3f} {100 * r['share']:6.1f}%")
        lines.append(f"{'(step outside timed regions)':60s} {'':6s} {g['outside_regions_s'] / g['steps']:8.3f} "
                     f"{100 * g['outside_regions_share']:6.1f}%")
    if report["replays"]:
        r = [x["seconds"] for x in report["replays"] if not x["recorded"]]
        if r:
            lines += ["", f"Step-graph replays (timed whole): {len(r)}, median {sorted(r)[len(r) // 2]:.3f} s"]
    o = report["overhead"]
    lines += ["", f"Meter hooks: {o['meter_host_s']:.3f} s host over {o['region_calls']} region calls "
                  f"({o['per_step_s'] * 1e3:.2f} ms per step)"]
    return "\n".join(lines) + "\n"


def enable(device=None, *, dycore_regions: bool = True) -> Meter:
    """Turn the meter on for this process (idempotent); ``device`` selects CUDA events. ``dycore_regions=False``
    leaves the dycore's compile event sink unset (the dycore step is then timed as a whole).
    ``torch._dynamo.reset()`` removes the compile callbacks: enable after a reset."""
    global METER
    if METER is None:
        meter = Meter(device, dycore_regions=dycore_regions)
        meter.install()
        METER = meter
    return METER


def disable() -> Meter | None:
    """Turn the meter off; returns it (its measurements stay readable)."""
    global METER
    meter, METER = METER, None
    if meter is not None:
        meter.uninstall()
    return meter


def region(name: str):
    """A timed region when the meter is on, else a shared null context."""
    meter = METER
    return _NULL if meter is None else meter.region(name)


def step(step_index: int, kind: Callable[[], str], *, part: str | None = None):
    meter = METER
    return _NULL if meter is None else meter.step(step_index, kind, part=part)


def replay(step_index: int, recorded: bool):
    meter = METER
    return _NULL if meter is None else meter.replay(step_index, recorded)


def cycle(label):
    meter = METER
    return _NULL if meter is None else meter.cycle(label)
