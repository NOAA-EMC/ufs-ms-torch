# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Exact-numerics policy for ``torch.compile`` over whole physics schemes.

Measured on torch 2.8.0 with captured coupled inputs:

* Inductor lowers ``x / c`` for a Python constant ``c`` to ``x * (1 / c)``, and
  ``x ** n`` for scalar ``n`` to multiplication chains (|n| >= 4, n <= -2) or to
  ``pow`` (n = -0.5); both differ from eager in the last bit (shared reference G34).
* Inductor's vectorized CPU C++ code returned a wrong water-vapour tendency for
  the whole-scheme SATMEDMF graph (5184 of 140800 values, up to q/dt), while
  ``eager``, ``aot_eager`` and scalar CPU code (``cpp.simdlen = 1``) matched eager.
* Clang merged ``std::sin``/``std::cos`` of one argument into ``sincos`` and
  ``pow(10, x)`` into ``exp10`` in Inductor's CPU kernels, and Inductor lowers
  ``pow(2, x)`` to ``exp2``; each differs from eager by an ulp.
* CPU FP32 transcendental functions differ between ATen's vector path and scalar libm. The
  compiled 240km pressure projection differed at 1594 interface values. FP32
  CPU functions with measured mismatches retain ATen calls; FP64 and CUDA keep their existing lowering.
  This includes the three power overloads and the FP32 tanh backward tail. CPU FP64 tanh backward
  also keeps ATen's kernel, whose rounding depends on the wheel's build (FMA on linux-aarch64).
* MPS FP32 powers use native ATen too. Its scalar square/cube differ from multiplication
  chains, including in reciprocal-square-root backward. Native ``rsqrt`` also differs
  from Inductor's Metal intrinsic. Native MPS sums retain eager's accumulation order.
  Native MPS ``log1p`` preserves its dedicated evaluation; generic codegen evaluates
  ``log(1 + x)``, changing cloud cover on the independent late-state packet.
* Torch2.8's MPS ``any`` codegen can corrupt a fused strided producer by nesting independent
  reduction-index loops. A fused Boolean consumer can also receive a string dtype.
  Native ATen MPS ``any`` avoids both defects without changing the assertion or moving data to CPU.
* MPS masked concat branches can use untyped ``0.0`` padding for integer/Boolean values.
  Metal then rounds random words through FP32 or rejects Boolean consumers. Cast the padding
  to the computed branch's type; do not change the integer stream or force extra stores.
* Metal permits 31 separately bound buffers. A four-node fusion of recurrence assembly can
  still require 76 buffers. Bound MPS fusion by external reads plus all produced buffers,
  not only node count; this conservative estimate includes stores that codegen may remove.
  Static-shape regions can then retain useful fusion without a global four-node cap (G225).
  Scalar backward addition chains can already exceed this limit before scheduling. Realize
  their existing operands before combining more than 30 distinct reads; addition order stays fixed.
* Torch 2.8's ``pointless_convert`` removes CPU FP64 -> FP32 -> FP64 chains,
  losing the required intermediate rounding between mixed-precision powers.
  TEMPO's sedimentation snow moment differed at 2128 values. Keep that rounding.
* Triton contracts ``a * b + c`` into FMA unless ``TRITON_DEFAULT_FP_FUSION=0``;
  eager CUDA runs the two operations as separate kernels.
* Inductor's in-place buffer reuse (``inplace_buffers``) gave wrong gradients for the RRTMGP SW
  two-stream solver (``sw_solver_2stream_v17``): its compiled backward overwrote a saved
  activation that other saved views (``unbind`` of one buffer) still read. Compiled VJPs were off
  by up to 7 % on random inputs and by 1e7 relative in the coupled radiation entry, while the
  forward, the FX graphs run eagerly and each compiled graph on cloned inputs were exact
  (torch 2.8.0, CPU). Off in this policy; it changes where results are stored, not
  their values.
* Inductor decomposes ``aten._assert_async`` to nothing (torch 2.8 ``_inductor/decomposition.py``),
  and AOT tracing drops the one-argument form as dead code, so every finiteness, error-flag and
  bound check inside a compiled region is off: a NaN raises in eager, not compiled.
  ``set_compiled_asserts(True)`` keeps both forms as side-effecting ATen fallback calls
  (``_keep_device_asserts``). That cost 9.7 % and 25 % of the compiled x1.10242 cycle (3.10 ->
  3.40 s and 3.31 -> 4.14 s, CPU, 4 threads, two back-to-back pairs; 353 assert calls per plain
  step), so it is off by default; the coupled interval driver checks the cycle state's finiteness
  outside compiled code instead (``cycle.run_coupled_interval``).

CPU backward reductions use ATen's tensor sum so parameter gradients keep eager's
accumulation order and operand layout. Full backward reductions use it on CUDA too. Functional VJP regions
use the same backward compilation policy. Accumulating index puts and scatter adds use
functional ATen outputs: repeated-index gradients keep the native accumulation order,
and consumers depend on the completed accumulation rather than a mutable zero buffer.
CPU loop reshaping preserves manually installed allocation and mutation-order dependencies.
Torch 2.8 otherwise drops these edges and can schedule a backward write before allocation.
Floating zero additions/subtractions remain in the graph: Inductor's identity
removal otherwise changes +0 + -0 to -0 (four 60 km surface-w values).

CPU FP32 and FP64 forward sums also retain ATen's order and operand layout. In FP32 the PBL
negative-tracer recovery otherwise differs at six captured values, with later
differences surviving in subnormal ice-number tracers. In FP64 Inductor's scalar CPU code adds
``torch.sum`` over a dimension in index order, while eager ATen adds the vector lanes first (four
float64 values on arm64: ``(x0 + x2) + (x1 + x3)``); a recompiled variant can switch to the scalar
code. The SA-TKE negative-energy borrowing sum then gave 1 ulp of TKE in one x1.10242 column at
cycle 15 and a different 25-cycle state; Noah-MP soilwater ``rsat`` gave 1 ulp of
``runoff`` in one x1.163842 column.

With this policy the compiled PBL, NSST, sea-ice and surface-layer calls are
bitwise equal to eager on CPU. Install it before the first compilation. The
policy id and a hash of this file enter Inductor's FX-graph cache key, so graphs compiled under
different lowerings never share a cache entry, and a cache kept between runs never serves a graph
lowered by an earlier version of this file.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import hashlib
import logging
import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar

import torch
import torch.autograd.forward_ad as forward_ad
from torch._inductor.custom_graph_pass import CustomGraphPass

from ufs_physics_common import cost_meter

POLICY_ID = "ufs-exact-numerics-v34"
# Every lowering and patch of the policy is in this file; its bytes key the compile cache.
POLICY_KEY = POLICY_ID + "+src." + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
Result = TypeVar("Result")
_log = logging.getLogger(__name__)

_installed = False
_pointwise_factory_installed = False
_compiled_asserts = False
# Graphs one compiled region may hold, scoped to its calls (the process-wide limit stays 8). The
# compaction capacity ladders give a region a static variant per rung: TEMPO's sedimentation tiers up
# to 24 per shape (48 with column chunks), convection 5 or 10, the shortwave daylit buffer up to 6.
REGION_RECOMPILE_LIMIT = 64
_compiled: dict[Callable[..., object] | tuple[Callable[..., object], str], Callable[..., object]] = {}
_eager = contextvars.ContextVar("ufs_compile_policy_eager", default=False)


@contextlib.contextmanager
def eager_regions(enabled: bool = True) -> Iterator[None]:
    """Fill caches without compiling cold-step-only variants, including nested dycore regions.

    Region implementations must preserve their ownership contracts in eager mode too.
    The acoustic pack regions explicitly preserve their shared inputs for this reason.
    ``False`` pins normal execution during replay, even inside an outer eager scope.
    """
    token = _eager.set(enabled)
    try:
        with torch.compiler.set_stance("force_eager" if enabled else "default"):
            yield
    finally:
        _eager.reset(token)


def eager_regions_active() -> bool:
    return _eager.get()


class _PolicyKey(CustomGraphPass):
    """No-op post-grad pass whose uuid separates cache entries by policy."""

    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        return None

    def uuid(self) -> str:
        return POLICY_KEY + ("+asserts" if _compiled_asserts else "")


def _exact_div_prim(a, b):
    from torch._inductor import lowering

    if all(lowering.is_boolean_type(x) or lowering.is_integer_type(x) for x in (a, b)):
        return lowering.truncdiv(a, b)
    return lowering.make_pointwise(lambda *args: lowering.ops.truediv(*args))(a, b)


def _make_exact_pow(original):
    """Scalar exponents as eager ATen: x*x, x*x*x, sqrt, rsqrt, 1/x, 1/(x*x), else libm pow.

    Inductor otherwise expands integer exponents into multiplication chains and
    sends -0.5 to pow; both differ from eager in the last bit.
    """
    from torch._inductor import lowering

    native = {name: lowering.fallback_handler(getattr(torch.ops.aten.pow, name), add_to_fallback_set=False)
              for name in ("Tensor_Scalar", "Tensor_Tensor", "Scalar")}

    def exact_pow(a, b):
        tensor = b if isinstance(a, (int, float)) else a
        if (not isinstance(tensor, (int, float)) and tensor.get_device().type in ("cpu", "mps")
                and tensor.get_dtype() == torch.float32):
            overload = ("Scalar" if isinstance(a, (int, float)) else
                        "Tensor_Scalar" if isinstance(b, (int, float)) else "Tensor_Tensor")
            return native[overload](a, b)
        if isinstance(a, (int, float)) and not isinstance(a, bool) and not isinstance(b, (int, float)):
            if a != 1 and lowering.is_float_dtype(b.get_dtype()):
                return lowering.pow_native(float(a), b)  # not exp2 for a == 2
            return original(a, b)
        if (not isinstance(b, (int, float)) or isinstance(b, bool) or isinstance(a, (int, float))
                or not lowering.is_float_dtype(a.get_dtype())):
            return original(a, b)
        if b == 2:
            return lowering.mul(a, a)
        if b == 3:
            return lowering.mul(lowering.mul(a, a), a)
        if b == 0.5:
            return lowering.sqrt(a)
        if b == -0.5:
            return lowering.rsqrt(a)
        if b == -1:
            return lowering.reciprocal(a)
        if b == -2:
            return lowering.reciprocal(lowering.mul(a, a))
        if b in (0, 1):
            return original(a, b)
        return lowering.pow_native(a, float(b))

    return exact_pow


def _make_float32_unary(original, aten_op, *, device_type):
    """Keep measured ATen FP32 rounding differences scoped to one device."""
    from torch._inductor import lowering

    native = lowering.fallback_handler(aten_op, add_to_fallback_set=False)

    def exact_unary(x):
        if x.get_device().type == device_type and x.get_dtype() == torch.float32:
            return native(x)
        return original(x)

    return exact_unary


def _make_backward_sum(original):
    """Keep native backward sums and CPU/MPS forward sums, including operand layout."""
    from torch._inductor import lowering
    from torch._inductor.virtualized import V

    native = lowering.fallback_handler(torch.ops.aten.sum.default, add_to_fallback_set=False)
    native_dim = lowering.fallback_handler(torch.ops.aten.sum.dim_IntList, add_to_fallback_set=False)

    def exact_sum(x, axis=None, keepdims=False, *, dtype=None):
        rank = len(x.get_size())
        selected = (axis,) if isinstance(axis, int) else axis
        axes = tuple(range(rank)) if selected is None or selected == [] else tuple(a % rank for a in selected)
        device = x.get_device().type
        native_forward = ((device == "cpu" and x.get_dtype() in (torch.float32, torch.float64))
                          or (device == "mps" and x.get_dtype() == torch.float32))
        if (_backward_codegen[0] or native_forward) and lowering.is_float_dtype(x.get_dtype()):
            if device in ("cpu", "mps"):
                # ATen's reduction order also depends on strides. Inductor can otherwise
                # materialize a pointwise producer in a different layout than eager FX.
                x = lowering.constrain_to_fake_tensor(x, V.graph.current_node.args[0].meta["val"])
            if set(axes) == set(range(rank)):
                result = native(x, dtype=dtype)
                return lowering.view(result, [1] * rank) if keepdims else result
            if device in ("cpu", "mps"):
                return native_dim(x, list(axes), keepdims, dtype=dtype)
        return original(x, axis, keepdims, dtype=dtype)

    return exact_sum


def _make_mps_any(original, aten_op):
    """Avoid Torch2.8's strided any fusion and string-dtype reduction defects."""
    from torch._inductor import lowering

    native = lowering.fallback_handler(aten_op, add_to_fallback_set=False)

    def exact_any(x, dim=None, keepdim=False):
        if x.get_device().type == "mps":
            return native(x) if dim is None else native(x, dim, keepdim)
        return original(x, dim, keepdim)

    return exact_any


def _keep_mps_mask_types() -> None:
    """Keep concat/padding values in the computed branch's Metal type."""
    from torch._inductor.codegen.mps import MetalOverrides, value_to_metal
    original = MetalOverrides.where

    def where(mask, value, other):
        # Keep the literal numeric until after Inductor's dtype propagation.
        # Only the emitted Metal conditional needs its explicit branch type.
        if isinstance(other, (bool, int, float)):
            return f"{mask} ? {value} : static_cast<decltype({value})>({value_to_metal(other)})"
        return original(mask, value, other)

    MetalOverrides.where = staticmethod(where)


def _functional_accumulate() -> None:
    """Use a functional ATen producer for repeated-index sums (torch 2.8).

    Generated CPU atomic adds change order between calls. IndexPutFallback also loses a
    read-after-write dependency in large graphs. Fusing consumers of scatter_add's mutable
    reduction buffer changed the actual TEMPO substep VJP by 26.6%. A functional result
    owns fresh storage, so consumers depend on the completed accumulation.
    ATen controls device determinism; this does not change the process-wide setting.
    """
    from torch._inductor import lowering

    native = lowering.fallback_handler(torch.ops.aten.index_put.default, add_to_fallback_set=False)
    original_impl = lowering.index_put_impl_

    def functional_fallback(target, indices, values, accumulate):
        result = native(target, indices, values, accumulate)
        return lowering.mutate_to(target, result)

    def implementation(target, indices, values, accumulate, check, may_realize=False):
        if accumulate:
            return functional_fallback(target, indices, values, accumulate)
        return original_impl(target, indices, values, accumulate, check, may_realize)

    lowering.index_put_fallback = functional_fallback
    lowering.index_put_impl_ = implementation
    for op in (torch.ops.aten.index_put.default, torch.ops.aten._unsafe_index_put.default):
        original = lowering.lowerings[op]

        def functional(target, indices, values, accumulate=False, _original=original):
            return native(target, indices, values, True) if accumulate else _original(target, indices, values, False)

        lowering.register_lowering(op, type_promotion_kind=None)(functional)

    scatter = lowering.fallback_handler(torch.ops.aten.scatter_add.default, add_to_fallback_set=False)

    def scatter_inplace(target, dim, index, source):
        return lowering.mutate_to(target, scatter(target, dim, index, source))

    lowering.register_lowering(torch.ops.aten.scatter_add.default, type_promotion_kind=None)(scatter)
    lowering.register_lowering(torch.ops.aten.scatter_add_.default, type_promotion_kind=None)(scatter_inplace)


def _keep_separate_libm_calls() -> None:
    """Append -fmath-errno -fno-builtin-pow -fsigned-zeros to Inductor's C++ flags (CPU).

    With -fno-math-errno clang merges std::sin and std::cos of one argument into a
    sincos call that can differ from the separate calls eager makes by one ulp
    (measured in GWD mountain blocking on macOS arm64), and rewrites pow(10, x)
    into exp10(x) (Noah-MP czil). Inductor's -fno-signed-zeros turned -0.0 into
    +0.0 in Noah-MP etran. Triton ignores these flags.
    """
    from torch._inductor import cpp_builder

    original = cpp_builder._get_optimization_cflags

    def optimization_cflags(*args, **kwargs):
        return [*original(*args, **kwargs), "fmath-errno", "fno-builtin-pow", "fsigned-zeros"]

    cpp_builder._get_optimization_cflags = optimization_cflags


def _no_cpu_outer_loop_fusion() -> None:
    """Turn off Inductor's CPU outer-loop fusion (torch 2.8.0), whose local buffers can be undersized.

    A compiled-reverse coupling test (x1.10242, column chunk 2561) aborted on heap corruption: the
    generated backward kernel allocated ``make_unique<double []>(1)`` for an outer-loop local buffer and
    wrote it at ``[x1]`` for x1 in 0..52 (the size is sliced from the global layout with a negative
    offset). Without outer-loop fusion every node keeps its global buffer; the arithmetic is unchanged.
    Triton (CUDA) does not use this path.
    """
    from torch._inductor.codegen import cpp

    cpp.CppScheduling.can_fuse_vertical_outer_loop = lambda self, node1, node2: False


_FUSION_ORIGINALS: dict[str, Callable[..., object]] = {}


def _preserve_scheduling_dependencies() -> None:
    """Keep Torch 2.8 CPU loop reshaping from discarding mutation-order edges.

    recompute_size_and_body rebuilds tensor reads from the loop body, unlike
    refresh_dependencies, which retains StarDep/WeakDep. The actual deep-convection
    backward lost 15 allocation edges and wrote an unallocated buffer. Retain those
    manual edges while allowing the reshaped tensor reads and writes to change.
    """
    from torch._inductor import scheduler
    from torch._inductor.dependencies import StarDep, WeakDep
    from torch.utils._ordered_set import OrderedSet

    if torch.__version__.split("+")[0] != "2.8.0":
        return
    original = scheduler.SchedulerNode.recompute_size_and_body

    @functools.wraps(original)
    def recompute(self, *args, **kwargs):
        manual = OrderedSet(dep for dep in self.read_writes.reads if isinstance(dep, (StarDep, WeakDep)))
        result = original(self, *args, **kwargs)
        if manual - self.read_writes.reads:
            self.set_read_writes(self.read_writes.with_read(manual))
        return result

    scheduler.SchedulerNode.recompute_size_and_body = recompute


def _cached_fusion_scores() -> None:
    """Per-node caches for two of Inductor's fusion heuristics (torch 2.8.0); same results, less compile time.

    For every candidate pair, ``score_fusion_memory`` rebuilt both nodes' read|write dependency sets and
    ``can_fusion_increase_peak_memory`` rebuilt their single-user inputs and a sympy reuse key per buffer.
    On the TEMPO and convection graphs (30-85k FX nodes) this was 26-40% of the whole compile (CPU,
    2026-10-20). Here each node's dependency ids and reuse keys are formed once (buffer users do not change
    while nodes fuse) and a pair compares them: the same integers and decisions, so the same generated code
    (TEMPO 103/103 and convection 25/25 code files identical; fusion 219 -> 54 s and 157 -> 50 s).
    """
    import itertools

    from torch._inductor import scheduler
    from torch._inductor.codegen.wrapper import buffer_reuse_key
    from torch._inductor.virtualized import V
    from torch.utils._ordered_set import OrderedSet

    if torch.__version__.split("+")[0] != "2.8.0":
        return  # the replacements mirror the 2.8.0 bodies
    _FUSION_ORIGINALS.update(score_fusion_memory=scheduler.Scheduler.score_fusion_memory,
                             can_fusion_increase_peak_memory=scheduler.Scheduler.can_fusion_increase_peak_memory)

    def dep_ids(sched, node):
        rw = node.read_writes
        hit = node.__dict__.get("_policy_dep_ids")
        if hit is None or hit[0] is not rw:
            table = sched.__dict__.setdefault("_policy_dep_table", {})
            sizes = sched.__dict__.setdefault("_policy_dep_sizes", [])
            ids = set()
            for dep in itertools.chain(rw.reads, rw.writes):
                if dep not in table:
                    table[dep] = len(sizes)
                    sizes.append(sched.dep_size_hint(dep))
                ids.add(table[dep])
            hit = node.__dict__["_policy_dep_ids"] = (rw, frozenset(ids))
        return hit[1]

    def score_fusion_memory(self, node1, node2):
        small, large = dep_ids(self, node1), dep_ids(self, node2)
        if len(small) > len(large):
            small, large = large, small
        sizes = self._policy_dep_sizes
        return sum(sizes[i] for i in small if i in large)

    def single_user_reuse_keys(sched, node):
        rw = node.read_writes
        hit = node.__dict__.get("_policy_reuse_keys")
        if hit is None or hit[0] is not rw:
            cache = sched.__dict__.setdefault("_policy_reuse_key", {})
            keys = OrderedSet()
            for read in rw.reads:
                buf = sched.name_to_buf.get(read.name)
                if buf and len(buf.users) == 1 and buf.node.has_tensor_output():
                    if read.name not in cache:
                        cache[read.name] = buffer_reuse_key(buf.node)
                    keys.add(cache[read.name])
            hit = node.__dict__["_policy_reuse_keys"] = (rw, keys)
        return hit[1]

    def can_fusion_increase_peak_memory(self, node1, node2):
        overhead = 0
        for key in single_user_reuse_keys(self, node1).intersection(single_user_reuse_keys(self, node2)):
            try:
                overhead += int(key[2])
            except ValueError:
                return False  # not an integer: the original falls back to fusing
        return bool(V.graph.sizevars.statically_known_gt(overhead, 32 * self.score_fusion_memory(node1, node2)))

    scheduler.Scheduler.score_fusion_memory = score_fusion_memory
    scheduler.Scheduler.can_fusion_increase_peak_memory = can_fusion_increase_peak_memory


def _make_mps_bounded_fusion(original):
    """Conservatively bound static Metal buffer arguments before nodes fuse."""
    @functools.wraps(original)
    def can_fuse(self, node1, node2):
        device = node1.get_device()
        if device is not None and device.type == "mps":
            writes = set(node1.get_buffer_names()) | set(node2.get_buffer_names())
            reads = {dep.name for node in (node1, node2) for dep in node.read_writes.reads}
            if len(reads - writes) + len(writes) > 31:
                return False
        return original(self, node1, node2)

    return can_fuse


def _make_mps_bounded_add(original):
    """Keep ordered lazy addition chains within one Metal argument table."""
    @functools.wraps(original)
    def add(a, b, **kwargs):
        from torch._inductor.ir import TensorBox

        if isinstance(a, TensorBox) and isinstance(b, TensorBox) and a.get_device().type == "mps":
            reads = (set(a.get_read_names()), set(b.get_read_names()))
            if len(reads[0] | reads[1]) > 30:
                for value, names in zip((a, b), reads, strict=True):
                    if len(names) > 1:
                        value.realize()
        return original(a, b, **kwargs)

    return add


def _bound_mps_fusion() -> None:
    from torch._inductor import lowering, scheduler

    if torch.__version__.split("+")[0] == "2.8.0":
        scheduler.Scheduler.can_fuse = _make_mps_bounded_fusion(scheduler.Scheduler.can_fuse)
        lowering.add = _make_mps_bounded_add(lowering.add)
        for op in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar):
            lowering.lowerings[op] = _make_mps_bounded_add(lowering.lowerings[op])


def _make_cpu_clamp(original, bound_wins):
    """CPU clamp as eager: the argument wins a tie, so clamp_min(-0.0, 0.0) stays -0.0.

    Inductor lowers clamp_min/clamp_max to maximum/minimum, where the bound wins
    the tie (+0.0). NaN arguments pass through both forms. CUDA keeps Inductor's
    lowering until its own eager tie rule is measured.
    """
    from torch._inductor import lowering

    def clamp(a, bound):
        if (not isinstance(bound, (int, float)) or isinstance(bound, bool)
                or a.get_device().type != "cpu" or not lowering.is_float_dtype(a.get_dtype())):
            return original(a, bound)
        def fn(x):
            limit = lowering.ops.constant(bound, a.get_dtype())
            beyond = bound_wins(x, limit)
            return lowering.ops.where(beyond, limit, x)
        return lowering.make_pointwise(fn)(a)

    return clamp


_NUMERIC_INDEX = re.compile(r"^[0-9\s+\-*/()]+$")


def _tensor_constant_triton_index() -> None:
    """Make a constant Triton load/store index a tensor before ``tl.broadcast_to``.

    Inductor 2.8 tests only ``sympy.Integer`` for constant indices. A gather whose
    indirect index folds to a literal prints as e.g. ``1800 + 18*(9)`` and is passed
    to ``tl.broadcast_to``, which Triton rejects (whole-scheme SATMEDMF on an A100).
    Only purely numeric index strings change; arithmetic on data is untouched.
    """
    import dataclasses

    from torch._inductor.codegen import triton

    original = triton.TritonKernel.indexing

    def indexing(self, *args, **kwargs):
        options = original(self, *args, **kwargs)
        text = getattr(options, "index_str", None)
        if not (isinstance(text, str) and text.startswith("tl.broadcast_to(") and text.endswith(")")):
            return options
        body = text[len("tl.broadcast_to("):-1]
        shape = options.expand_str
        if shape is None or not body.endswith(", " + shape):
            inner, _, shape = body.rpartition(", ")
        else:
            inner = body[:-len(", " + shape)]
        if not _NUMERIC_INDEX.match(inner):
            return options
        index_str = f"tl.full({shape}, {inner}, {self.index_dtype})"  # as Inductor's Integer branch
        return dataclasses.replace(options, index_str=index_str)

    triton.TritonKernel.indexing = indexing


_backward_codegen = [False]  # set while _policy_inner_compile generates a backward graph
_PERSISTENT_DECORATOR = "@triton_heuristics.persistent_reduction("
# Compile workers load kernel source with the parent's sys.path, so they can import this module.
_WIDE_DECORATOR = "@__import__('ufs_physics_common.compile_policy', fromlist=['_']).wide_persistent_reduction("


def _normalize_tiling_solutions() -> None:
    """Remove torch's non-evaluating Identity from integer tiling solutions (2.8).

    ``55 * Identity(3)`` reports constant/integer/number, but ``int`` raises
    ``Expected a number but got Mul``. Identity has no indexing semantics.
    Normalize before solving too: FloorDiv/ModularIndexing validation substitutes
    the solution back into the index expression. Keep all coalescing heuristics.
    A zero solution is no tile: the analyzer would divide the variable's range by it
    (ZeroDivisionError compiling the UGWP v1 ``ngw_solv2`` training graph on CUDA).
    """
    from torch._inductor import tiling_utils
    from torch.utils._sympy.functions import Identity

    original = tiling_utils.solve_for_tiling

    @functools.wraps(original)
    def solve(expr):
        expr = expr.replace(Identity, lambda value: value)
        result = original(expr)
        if result is None:
            return None
        result = result.replace(Identity, lambda value: value)
        return None if result == 0 else result

    tiling_utils.solve_for_tiling = solve


def _keep_signed_zero_operations() -> None:
    """Floating +/- zero are not identities: +0 + -0 is +0, not -0."""
    from torch._inductor.fx_passes import joint_graph

    original = joint_graph.remove_no_ops

    @functools.wraps(original)
    def remove(gm, zeros, ones):
        integer_zeros = type(zeros)(node for node in zeros
            if not (node.meta["val"].is_floating_point() or node.meta["val"].is_complex()))
        return original(gm, integer_zeros, ones)

    joint_graph.remove_no_ops = remove


def _keep_cpu_narrowing_casts() -> None:
    """A CPU FP64 -> FP32 -> FP64 chain is rounding, not an AMP identity."""
    from torch._inductor.fx_passes import joint_graph

    found = False
    for entries in joint_graph.patterns.patterns.values():
        for entry in entries:
            if getattr(entry, "handler", None) is not joint_graph.pointless_convert:
                continue
            original = entry.extra_check

            def safe(match, original=original):
                value = match.kwargs["arg"].meta.get("val")
                if (isinstance(value, torch.Tensor) and value.device.type == "cpu"
                        and value.dtype == torch.float64 and match.kwargs["dtype1"] == torch.float32
                        and match.kwargs["dtype2"] == torch.float64):
                    return False
                return original(match)

            entry.extra_check = safe
            found = True
    if not found:
        raise RuntimeError("exact-numerics policy requires the Torch 2.8 conversion-chain pattern")


def _widen_reduction_configs(configs, size_hints, max_warps=32):
    """Give each persistent-reduction config at least two threads per block element.

    Triton 3.4 (sm_80) fails `TritonGPURemoveLayoutConversions` ("operand does not dominate
    this use") on large fused backward reductions whenever a block holds more than half an
    element per thread (SATMEDMF r=9 and SA-SAS deep r=55 kernels, every XBLOCK/num_warps
    pair checked); with at least two threads per element every checked layout compiled.
    """
    from torch._inductor.runtime.runtime_utils import next_power_of_2
    from torch._inductor.runtime.triton_heuristics import get_total_reduction_numel

    rblock = next_power_of_2(get_total_reduction_numel(size_hints))
    for cfg in configs:
        tile = 1
        for name in ("XBLOCK", "YBLOCK"):
            tile *= cfg.kwargs.get(name, 1)
        needed = next_power_of_2(-(-2 * tile * rblock // 32))
        cfg.num_warps = max(cfg.num_warps, min(needed, max_warps))
    return configs


def wide_persistent_reduction(size_hints, reduction_hint=False, triton_meta=None, filename=None,
                              inductor_meta=None):
    """Inductor's persistent_reduction heuristic with _widen_reduction_configs applied."""
    from torch._inductor.runtime import triton_heuristics as heuristics

    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints["x"] = 1
    configs = _widen_reduction_configs(
        heuristics._persistent_reduction_configs(size_hints, reduction_hint, inductor_meta), size_hints)
    return heuristics.cached_autotune(size_hints, configs, triton_meta=triton_meta, inductor_meta=inductor_meta,
                                      filename=filename, heuristic_type=heuristics.HeuristicType.PERSISTENT_REDUCTION)


def _route_backward_persistent_reductions() -> None:
    """While a backward graph is generated, persistent reductions use wide_persistent_reduction.

    The decorator is rewritten in the kernel source before Inductor embeds it in the wrapper,
    so FX-cache hits reuse the rewritten source. Forward kernels are unchanged.
    """
    from torch._inductor.codegen import triton

    original = triton.TritonScheduling.define_kernel

    def define_kernel(self, src_code, node_schedule, kernel):
        if _backward_codegen[0] and _PERSISTENT_DECORATOR in src_code:
            src_code = src_code.replace(_PERSISTENT_DECORATOR, _WIDE_DECORATOR)
        return original(self, src_code, node_schedule, kernel)

    triton.TritonScheduling.define_kernel = define_kernel


_ASSERT_OPS = (torch.ops.aten._assert_async.msg, torch.ops.aten._functional_assert_async.msg)
# Eager's message for a failed one-argument torch._assert_async (aten/src/ATen/native/TensorCompare.cpp).
_ASSERT_DEFAULT_MESSAGE = "Expected Tensor with single nonzero value, but got zero"


def _keep_device_asserts() -> None:
    """Lower ``aten._assert_async.msg`` to a side-effecting ATen fallback call (kept in the wrapper code).

    The policy's decomposition table (``_policy_decompositions``) leaves the op in the graph; Inductor has
    no lowering of its own, and a fallback kernel without outputs counts as dead unless it reports a side
    effect. CPU raises in the call as eager does; CUDA runs the device assert kernel eager runs.
    """
    from torch._inductor import ir, lowering

    lowering.make_fallback(torch.ops.aten._assert_async.msg, override_decomp=True)
    original = ir.FallbackKernel.has_side_effects

    def has_side_effects(self):
        return getattr(self, "op_overload", None) is torch.ops.aten._assert_async.msg or original(self)

    ir.FallbackKernel.has_side_effects = has_side_effects


def set_compiled_asserts(enabled: bool) -> bool:
    """Keep ``torch._assert_async`` checks inside compiled regions (default off); return the previous value.

    Set it before the first compilation: regions compiled earlier keep their graphs. The setting enters
    the policy's cache key, so graphs with and without the checks never share a cache entry.
    """
    global _compiled_asserts
    if type(enabled) is not bool:
        raise TypeError("compiled asserts switch requires a bool")
    previous, _compiled_asserts = _compiled_asserts, enabled
    return previous


def _policy_decompositions(example_inputs=()) -> dict:
    """Inductor's decomposition table; with ``set_compiled_asserts`` without the asserts' no-op decompositions."""
    from torch._inductor.decomposition import select_decomp_table

    table = select_decomp_table()
    excluded = set(_ASSERT_OPS) if _compiled_asserts else set()
    if any(isinstance(x, torch.Tensor) and x.device.type == "cpu" and x.dtype in (torch.float32, torch.float64)
           for x in example_inputs):
        # ATen's FP32 tail rounds differently from its decomposed (1 - y*y), and the GCC-built
        # linux-aarch64 wheel's FP64 kernel contracts it into fma(-y, y, 1) (torch 2.8.0).
        excluded.add(torch.ops.aten.tanh_backward.default)
    return {op: fn for op, fn in table.items() if op not in excluded} if excluded else table


def _assert_messages(gm: torch.fx.GraphModule) -> int:
    """Give each one-argument ``torch._assert_async`` eager's message: AOT tracing keeps only ``.msg``."""
    if not _compiled_asserts:
        return 0
    count = 0
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch._assert_async and len(node.args) == 1:
            node.args = (*node.args, _ASSERT_DEFAULT_MESSAGE)
            count += 1
    if count:
        gm.recompile()
    return count


def _reclaim_driver_autotune_oom(original):
    """Release unused Torch cache when CUDA needs memory outside its allocator."""
    @functools.wraps(original)
    def bench(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except RuntimeError as error:
            if (self.device_props.type != "cuda"
                    or "CUDA driver error: out of memory" not in str(error)):
                raise
            # Spilled kernel locals use driver memory, not the Torch allocator's retry path.
            with torch.cuda.device(self.device_props.index):
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                if reserved <= allocated:
                    raise
                free_before, _ = torch.cuda.mem_get_info()
                torch.cuda.empty_cache()
                free_after, _ = torch.cuda.mem_get_info()
                _log.warning("CUDA autotune driver OOM for %s: allocated=%d reserved=%d; "
                             "released unused cache, driver free=%d -> %d; retry once",
                             self.inductor_meta.get("kernel_name", "unknown"),
                             allocated, reserved, free_before, free_after)
                return original(self, *args, **kwargs)

    return bench


def _benchmark_spilling_pointwise(original):
    """Measure pointwise candidates when Torch would reject every candidate for spills.

    Torch 2.8 returns infinity above its spill threshold, then selects the first candidate
    if every cost is infinite. TEMPO selected n_spills=1976 without measuring its
    n_spills=134 alternative. This driver metric includes local stack bytes / 4.
    Keep reduction configurations and their arithmetic unchanged.
    """
    from torch._inductor.runtime.hints import HeuristicType

    @functools.wraps(original)
    def benchmark(self, *args, **kwargs):
        threshold = self.inductor_meta.get("spill_threshold", 16)
        if (self.heuristic_type is not HeuristicType.POINTWISE or self.custom_kernel
                or not self.launchers or not all(x.n_spills > threshold for x in self.launchers)):
            return original(self, *args, **kwargs)
        had_threshold = "spill_threshold" in self.inductor_meta
        self.inductor_meta["spill_threshold"] = max(x.n_spills for x in self.launchers)
        try:
            timings = original(self, *args, **kwargs)
        finally:
            if had_threshold:
                self.inductor_meta["spill_threshold"] = threshold
            else:
                del self.inductor_meta["spill_threshold"]
        _log.warning("Measured all-spilling pointwise candidates for %s: %s",
                     self.inductor_meta.get("kernel_name", "unknown"),
                     [(str(x.config), x.n_spills, milliseconds) for x, milliseconds in timings.items()])
        return timings

    return benchmark


def _overflow_pointwise(original):
    """Do not enlarge Y tiles for a limit already handled by the overflow grid.

    Torch 2.8's pointwise factory increases YBLOCK to keep grid Y <= 65535,
    even for Grid2DWithYZOverflow. At 60 km this creates 64x1024 tiles and
    multi-minute native builds. Only candidate sizing uses the capped hint;
    the autotuner and generated grid retain the complete problem size.
    """
    from torch._inductor.runtime import triton_heuristics as heuristics
    from torch._inductor.runtime.hints import HeuristicType, TileHint

    @functools.wraps(original)
    def pointwise(size_hints, triton_meta, tile_hint=None, filename=None,
                  min_elem_per_thread=0, inductor_meta=None):
        meta = {} if inductor_meta is None else inductor_meta
        if (len(size_hints) != 2 or meta.get("grid_type") != "Grid2DWithYZOverflow"
                or not 65535 < size_hints.get("y", 0) <= 65535**2
                or min_elem_per_thread or meta.get("autotune_hints") or meta.get("no_x_dim")
                or getattr(triton_meta.get("device"), "type", None) != "cuda"):
            return original(size_hints, triton_meta, tile_hint, filename,
                            min_elem_per_thread, inductor_meta)
        hints = {**size_hints, "y": 65535}
        bs = max(256, min(size_hints["x"] * size_hints["y"] // 128, 1024))
        pairs = ((32, 32),)
        if not ((heuristics.disable_pointwise_autotuning(meta) or tile_hint == TileHint.SQUARE)
                and not (meta.get("max_autotune") or meta.get("max_autotune_pointwise"))):
            pairs = ((32, 32), (64, 64), (256, 16), (16, 256), (bs, 1), (1, bs))
        configs = [heuristics.triton_config(hints, x, y) for x, y in pairs]
        return heuristics.cached_autotune(
            size_hints, configs, triton_meta=triton_meta, inductor_meta=meta,
            heuristic_type=HeuristicType.POINTWISE, filename=filename)

    return pointwise


def _install_pointwise_factory() -> None:
    global _pointwise_factory_installed
    if not _pointwise_factory_installed:
        from torch._inductor.runtime import triton_heuristics

        triton_heuristics.pointwise = _overflow_pointwise(triton_heuristics.pointwise)
        _pointwise_factory_installed = True


def _policy_worker_compile_triton(load_kernel, extra_env, extra_config):
    # Torch's fresh compiler subprocess does not inherit parent monkeypatches.
    # This module-level callable survives the existing pool's pickle boundary.
    from torch._inductor.runtime.compile_tasks import _worker_compile_triton

    _install_pointwise_factory()
    return _worker_compile_triton(load_kernel, extra_env, extra_config)


def install_exact_numerics() -> None:
    """Make Inductor keep eager's arithmetic for the measured cases; idempotent."""
    global _installed
    if _installed:
        return
    # Triton workers read this when they compile; it must precede the first compile.
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "0")
    from torch._inductor import config, lowering

    if config.post_grad_custom_pre_pass is not None:
        raise RuntimeError("exact-numerics policy needs the free post_grad_custom_pre_pass slot")
    lowering.div_prim = _exact_div_prim  # aten.div/true_divide call it by module name
    # Lookups use overload keys; registering a packet would add an unused key.
    lowering.register_lowering([torch.ops.prims.div.default], broadcast=True)(_exact_div_prim)
    exact_pow = _make_exact_pow(lowering.pow)
    lowering.pow = exact_pow  # the original recurses through this module name
    pow_ops = torch.ops.aten.pow
    lowering.register_lowering([pow_ops.Tensor_Scalar, pow_ops.Tensor_Tensor, pow_ops.Scalar],
                               broadcast=True)(exact_pow)
    for name in ("exp", "log", "log10", "log1p", "expm1", "sin", "cos", "tan",
                 "asin", "acos", "atan", "tanh", "erf", "erfc"):
        aten_op = getattr(torch.ops.aten, name).default
        for op in (aten_op, getattr(torch.ops.prims, name).default):
            lowering.register_lowering([op])(
                _make_float32_unary(lowering.lowerings[op], aten_op, device_type="cpu"))
    aten_log1p = torch.ops.aten.log1p.default
    for op in (aten_log1p, torch.ops.prims.log1p.default):
        lowering.register_lowering([op])(
            _make_float32_unary(lowering.lowerings[op], aten_log1p, device_type="mps"))
    aten_rsqrt = torch.ops.aten.rsqrt.default
    lowering.rsqrt = _make_float32_unary(lowering.rsqrt, aten_rsqrt, device_type="mps")
    lowering.register_lowering([aten_rsqrt, torch.ops.prims.rsqrt.default])(lowering.rsqrt)
    lowering.register_lowering([torch.ops.aten.tanh_backward.default])(
        lowering.fallback_handler(torch.ops.aten.tanh_backward.default, add_to_fallback_set=False))
    aten = torch.ops.aten
    lowering.register_lowering([aten.sum.default, aten.sum.dim_IntList, torch.ops.prims.sum.default])(
        _make_backward_sum(lowering.sum_))
    for op in (aten.any.default, aten.any.dim, aten.any.dims):
        lowering.register_lowering([op])(_make_mps_any(lowering.lowerings[op], op))
    lowering.register_lowering([aten.clamp_min.default])(
        _make_cpu_clamp(lowering.maximum, lambda x, m: lowering.ops.lt(x, m)))
    lowering.register_lowering([aten.clamp_max.default])(
        _make_cpu_clamp(lowering.minimum, lambda x, m: lowering.ops.gt(x, m)))
    config.cpp.simdlen = 1
    config.inplace_buffers = False  # in-place reuse overwrote aliased saved views (module docstring)
    _tensor_constant_triton_index()
    _normalize_tiling_solutions()
    _keep_signed_zero_operations()
    _keep_cpu_narrowing_casts()
    _route_backward_persistent_reductions()
    _keep_separate_libm_calls()
    _preserve_scheduling_dependencies()
    _cached_fusion_scores()
    _bound_mps_fusion()
    _keep_mps_mask_types()
    _no_cpu_outer_loop_fusion()
    _functional_accumulate()
    _keep_device_asserts()
    from torch._inductor import async_compile
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    _install_pointwise_factory()
    async_compile._worker_compile_triton = _policy_worker_compile_triton
    CachingAutotuner.bench = _reclaim_driver_autotune_oom(CachingAutotuner.bench)
    CachingAutotuner.benchmark_all_configs = _benchmark_spilling_pointwise(
        CachingAutotuner.benchmark_all_configs)
    config.post_grad_custom_pre_pass = _PolicyKey()
    _installed = True


def _global_tables_as_device_constants(gm: torch.fx.GraphModule, example_inputs, target=None):
    """Turn CPU tensors lifted from module globals into constants on the graph's device.

    Dynamo lifts static NumPy tables (``G[...]._STATIC_*``) as CPU graph inputs; a
    CUDA graph then copies them host-to-device on every call, which syncs and
    blocks CUDA-graph capture. The tables are read-only, so one device copy made
    at compile time is equivalent. The original inputs stay in the signature,
    unused. ``target`` defaults to the single CUDA device of the other inputs.
    """
    if target is None:
        devices = {t.device for t in example_inputs if isinstance(t, torch.Tensor) and t.device.type == "cuda"}
        if len(devices) != 1:
            return 0
        target = next(iter(devices))
    count = 0
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for node, value in zip(placeholders, example_inputs):
        grapharg = node.meta.get("grapharg")
        source = grapharg.source.name() if grapharg is not None and grapharg.source is not None else ""
        if not (isinstance(value, torch.Tensor) and value.device.type == "cpu" and "G['" in source):
            continue
        name = f"_policy_global_table_{count}"
        gm.register_buffer(name, value.detach().clone().to(target))
        if hasattr(gm, "_param_name_to_source"):  # AOT dispatch maps every lifted buffer to a source
            from torch._dynamo.source import ConstantSource

            gm._param_name_to_source[name] = ConstantSource(name)
        with gm.graph.inserting_after(node):
            constant = gm.graph.get_attr(name)
        node.replace_all_uses_with(constant)
        count += 1
    if count:
        gm.recompile()
    return count


# Backward graphs only: one launch config per Triton kernel instead of runtime autotuning.
# Triton 3.4 (sm_80) fails `TritonGPURemoveLayoutConversions` ("operand #0 does not dominate
# this use") on some XBLOCK/num_warps configs of large fused backward reductions; Inductor
# precompiles every candidate config, so one bad config aborts the compile. With one config,
# wide_persistent_reduction gives it enough warps to compile, and the fixed layout keeps one
# summation order from run to run. CPU code ignores both.
_BACKWARD_CONFIG = {"triton.autotune_pointwise": False}


def _policy_inner_compile(gm, example_inputs, *, functional_vjp=False, **kwargs):
    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx_inner

    if kwargs.get("is_backward") or functional_vjp:
        _backward_codegen[0] = True
        try:
            with config.patch(_BACKWARD_CONFIG):
                return compile_fx_inner(gm, example_inputs, **kwargs)
        finally:
            _backward_codegen[0] = False
    return compile_fx_inner(gm, example_inputs, **kwargs)


def _inductor_backend(gm, example_inputs):
    from torch._inductor.compile_fx import compile_fx

    _global_tables_as_device_constants(gm, example_inputs)
    _assert_messages(gm)
    return compile_fx(gm, example_inputs, inner_compile=_policy_inner_compile,
                      decompositions=_policy_decompositions(example_inputs))


def _inductor_vjp_backend(gm, example_inputs):
    """A functional VJP is a forward FX graph containing backward arithmetic."""
    from torch._inductor.compile_fx import compile_fx

    _global_tables_as_device_constants(gm, example_inputs)
    _assert_messages(gm)
    return compile_fx(gm, example_inputs,
                      inner_compile=functools.partial(_policy_inner_compile, functional_vjp=True),
                      decompositions=_policy_decompositions(example_inputs))


def _canonical_layout(value, seen: set[int]):
    """``value`` with each tensor contiguous and no tensor object twice (later occurrences are aliases).

    A scheme region takes the whole carried state, and Dynamo guards every tensor it passes through on
    strides and on object identity between inputs. The carry alternates layouts (a contiguous cold-start
    field, a strided view of qgrs after Noah-MP, a fresh field after a radiation step; two names bound
    to one object or not), so one region collected a variant per combination: apply_surface_layer
    reached torch's recompile limit after the first radiation step (x1.10242, 13 cycles), a hard error
    under fullgraph. Only views into a larger buffer are copied (their strides depend on the producer);
    a tensor that owns its whole storage keeps its layout, so the column-major TEMPO tables (0.9 GiB) are
    never copied, and broadcast (stride-0) views stay as they are. Values are unchanged.
    A later occurrence of one tensor object is a distinct alias (``view_as``, no copy; a copied view is
    copied again): Dynamo then sees distinct objects whatever the carry binds twice, and no bytes move
    (a copy of each repeat cost 67 mesh tensors per microphysics-prefix call at 60 km).
    """
    if isinstance(value, torch.Tensor):
        copy = (not value.is_contiguous() and 0 not in value.stride()
                and value.untyped_storage().nbytes() > value.numel() * value.element_size())
        if id(value) in seen:
            return value.contiguous() if copy else value.view_as(value)
        seen.add(id(value))
        return value.contiguous() if copy else value
    if isinstance(value, dict):
        out = {k: _canonical_layout(v, seen) for k, v in value.items()}
        return out if any(out[k] is not v for k, v in value.items()) else value
    if isinstance(value, tuple) and not hasattr(value, "_fields"):
        # Lists stay opaque: in scheme inputs they hold per-column Python records (Noah-MP's case has
        # 10,243 column dicts; walking them cost 1.1 s per call), not carried tensors.
        out = [_canonical_layout(v, seen) for v in value]
        return tuple(out) if any(a is not b for a, b in zip(out, value)) else value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changed = {}
        for f in dataclasses.fields(value):
            current = getattr(value, f.name)
            new = _canonical_layout(current, seen)
            if new is not current:
                changed[f.name] = new
        if not changed:
            return value
        if not hasattr(value, "__dict__"):  # slotted dataclass
            return dataclasses.replace(value, **changed)
        copy = object.__new__(type(value))  # field values only; no __init__/__post_init__ re-validation
        copy.__dict__.update(value.__dict__)
        copy.__dict__.update(changed)
        return copy
    return value


def compiled(fn: Callable[..., Result], *, normalize_layouts: bool = True) -> Callable[..., Result]:
    """Whole-call ``torch.compile`` of ``fn`` under the exact policy, built once per function.

    The first call runs ``fn`` eagerly: schemes fill their table caches there (file
    parsing, coefficient reduction), which Dynamo must not trace. Later calls run the
    compiled region, which equals eager bit for bit under this policy. With ``cost_meter`` on,
    each call outside a traced region is a timed region named ``fn.__name__``.
    ``normalize_layouts=False`` is for qualified read-only tensor islands/assembly only. Their
    producer layouts stay explicit in guards; all carried-state regions keep normalization.
    """
    cache_key = fn if normalize_layouts else (fn, "borrowed_layouts")
    found = _compiled.get(cache_key)
    if found is None:
        install_exact_numerics()
        region = torch.compile(fn, backend=_inductor_backend, fullgraph=True, dynamic=False)
        # Config.patch creates a context class per call; prepare its critical-path helper once.
        patch_limit = torch._dynamo.config._make_closure_patcher(recompile_limit=REGION_RECOMPILE_LIMIT)
        warmed: set[tuple] = set()

        def signature(kwargs):
            # One eager call per static keyword signature (e.g. deep vs shallow convection).
            return tuple(sorted((k, v) for k, v in kwargs.items() if type(v) in (bool, int, str)))

        def eager_call(*args, **kwargs):
            # Nested regions become part of the outer graph. Warm their tables,
            # not compiled variants that the outer graph will never call.
            if _eager.get():
                return fn(*args, **kwargs)
            with eager_regions():
                return fn(*args, **kwargs)

        def call(*args, **kwargs):
            # Called while an enclosing region is traced: inline, so that region holds one graph.
            if torch.compiler.is_compiling():
                return fn(*args, **kwargs)
            # torch 2.8's compiled regions drop forward-mode tangents without an error (their
            # outputs come back as plain tensors). Under an active dual level run the eager
            # function, which equals the region bit for bit under this policy.
            if forward_ad._current_level >= 0:
                return fn(*args, **kwargs)
            meter = cost_meter.METER  # None (off): no timing, nothing else changes
            timed = meter is not None and meter.enter(fn.__name__)
            try:
                key = signature(kwargs)
                if key not in warmed or _eager.get():
                    warmed.add(key)
                    return eager_call(*args, **kwargs)
                seen: set[int] = set()
                restore_limit = patch_limit()
                try:
                    if normalize_layouts:
                        return region(*_canonical_layout(args, seen), **_canonical_layout(kwargs, seen))
                    return region(*args, **kwargs)
                finally:
                    restore_limit()
            finally:
                if timed:
                    meter.exit()

        def warm(*args, **kwargs):
            """Make the eager first call now (e.g. before a checkpoint), unless this signature had it."""
            key = signature(kwargs)
            if key not in warmed:
                warmed.add(key)
                meter = cost_meter.METER
                timed = meter is not None and meter.enter(fn.__name__)
                try:
                    eager_call(*args, **kwargs)
                finally:
                    if timed:
                        meter.exit()

        call.__wrapped__ = fn
        call.warm = warm
        found = _compiled[cache_key] = call
    return found  # type: ignore[return-value]


def stored(x: torch.Tensor) -> torch.Tensor:
    """``x`` itself (an identity view); under ``torch.compile`` Inductor stores it in its own buffer.

    Inductor inlines a cheap elementwise result (a select, a column gather) into each consumer
    instead of storing it; for a field read by many kernels that repeats the work and re-reads
    its inputs in every consumer. Inductor realizes the input of ``as_strided``, so this view
    marks the store point. Use it on the direct result of a computing op. A view, or a tensor
    that is not contiguous, is returned unchanged: Inductor's ``as_strided`` lowering drops the
    view's storage offset. (So does a clone or alias of such a view, which Inductor removes as
    a no-op: do not pass one.)
    """
    if x._base is not None or not x.is_contiguous():
        return x
    return x.as_strided(x.size(), x.stride())
