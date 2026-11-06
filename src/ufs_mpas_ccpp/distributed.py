# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The coupled model on spatial MPAS partitions (CPU Gloo or NCCL ranks).

Physics runs on each rank's owned columns; the dycore runs on owned plus halo rows with the
maintained distributed dycore (``mpas_pytorch.distributed``). This module adds what the coupling
needs beyond the dycore: the case partition (setup only), the commit halo groups (defined with the
dycore's groups), sums in global-ID order that do not depend on the rank count, and one decision
for all ranks where a rank-local value would otherwise steer the run (compaction overflow, the
per-level h2o test).

The run's ``config["_halo_exchanger"]`` (a ``DynamicsHaloExchange``) selects the partitioned path;
a one-rank run has no exchanger and none of this code runs.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
from contextvars import ContextVar
from datetime import timedelta
from typing import Any, Mapping

import torch
import torch.distributed as dist

_SCOPE: ContextVar[Any] = ContextVar("coupled_partition_exchanger", default=None)


# The scope as a plain value for compiled regions: Dynamo cannot read a ContextVar, and it guards
# this list's item, so a region traced under one scope is not reused under another.
_TRACED_SCOPE: list[Any] = [None]


@contextlib.contextmanager
def partition_scope(exchanger):
    """Make ``exchanger`` (or None: one rank) the partition of the numerics inside."""
    token = _SCOPE.set(exchanger)
    previous = _TRACED_SCOPE[0]
    _TRACED_SCOPE[0] = exchanger
    try:
        yield
    finally:
        _SCOPE.reset(token)
        _TRACED_SCOPE[0] = previous


def current():
    """The active partition's halo exchanger, or None on one rank."""
    if torch.compiler.is_compiling():
        return _TRACED_SCOPE[0]
    return _SCOPE.get()


@dataclasses.dataclass(frozen=True)
class _Layout:
    rank: int
    world_size: int
    owned_ids: tuple[torch.Tensor, ...]  # zero-based global cell IDs of each rank's owned rows
    n_global: int
    max_owned: int


_LAYOUTS: dict[int, tuple[Any, _Layout]] = {}


def _layout(exchanger) -> _Layout:
    hit = _LAYOUTS.get(id(exchanger))
    if hit is not None and hit[0] is exchanger:
        return hit[1]
    partitions = exchanger.partitions
    device = exchanger.transport.device
    owned = tuple(p.cells.global_ids[: p.cells.owned_count].to(device) for p in partitions)
    layout = _Layout(exchanger.transport.rank, len(partitions), owned,
                     partitions[0].cells.global_count, max(int(ids.numel()) for ids in owned))
    _LAYOUTS[id(exchanger)] = (exchanger, layout)
    return layout


def _all_gather_rows(local: torch.Tensor, layout: _Layout) -> list[torch.Tensor]:
    """Every rank's leading rows (padded to the largest owned count on the wire)."""
    count = local.shape[0]
    padded = torch.cat((local, local.new_zeros((layout.max_owned - count, *local.shape[1:])))).contiguous()
    parts = [torch.empty_like(padded) for _ in range(layout.world_size)]
    dist.all_gather(parts, padded)
    return parts


def _to_global(local: torch.Tensor, layout: _Layout) -> torch.Tensor:
    parts = _all_gather_rows(local, layout)
    result = local.new_empty((layout.n_global, *local.shape[1:]))
    for ids, part in zip(layout.owned_ids, parts, strict=True):
        result.index_copy_(0, ids, part[: ids.numel()])
    return result


def _owned_sum(grad: torch.Tensor, layout: _Layout) -> torch.Tensor:
    """Adjoint of a replicated gather: the ranks' cotangents summed in rank order, owned rows."""
    parts = [torch.empty_like(grad) for _ in range(layout.world_size)]
    dist.all_gather(parts, grad.contiguous())
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return total.index_select(0, layout.owned_ids[layout.rank])


class _GlobalRows(torch.autograd.Function):
    """Owned cell rows -> all cells in global-ID order, replicated on every rank.

    Backward sums the ranks' cotangents of the replicated result (rank order) and returns the
    owned rows; forward mode gathers the tangents the same way.
    """

    @staticmethod
    def forward(local, layout):
        return _to_global(local, layout)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.layout = inputs[1]

    @staticmethod
    def backward(ctx, grad):
        return _owned_sum(grad, ctx.layout), None

    @staticmethod
    def jvp(ctx, tangent, _layout_tangent):
        return _to_global(tangent, ctx.layout)


def global_rows(local: torch.Tensor, exchanger=None) -> torch.Tensor:
    """All cells' rows in global-ID order from each rank's owned rows (identity on one rank)."""
    exchanger = current() if exchanger is None else exchanger
    if exchanger is None:
        return local
    layout = _layout(exchanger)
    if local.shape[0] != layout.owned_ids[layout.rank].numel():
        raise ValueError("global rows require exactly the owned cell rows")
    return _GlobalRows.apply(local, layout)


def rank_values(local: torch.Tensor, exchanger=None) -> torch.Tensor:
    """Each rank's ``local`` stacked in rank order ([world, ...], replicated; [1, ...] on one rank)."""
    exchanger = current() if exchanger is None else exchanger
    if exchanger is None:
        return local[None]
    layout = _layout(exchanger)
    ranks = dataclasses.replace(layout, owned_ids=tuple(
        torch.tensor([r], device=local.device) for r in range(layout.world_size)),
        n_global=layout.world_size, max_owned=1)
    return _GlobalRows.apply(local[None], ranks)


def all_ranks(flag: bool, exchanger) -> bool:
    """True when ``flag`` holds on every rank (host decisions that steer collective work)."""
    if exchanger is None:
        return flag
    gathered: list[Any] = [None] * exchanger.transport.world_size
    dist.all_gather_object(gathered, bool(flag), group=getattr(exchanger, "setup_group", None))
    return all(gathered)


def consensus_regions(regions: tuple[str, ...], exchanger) -> tuple[str, ...]:
    """The union of the ranks' overflowed regions, so that every rank takes the same rerun decision."""
    if exchanger is None:
        return regions
    gathered: list[Any] = [None] * exchanger.transport.world_size
    dist.all_gather_object(gathered, tuple(regions), group=getattr(exchanger, "setup_group", None))
    return tuple(sorted(set().union(*gathered)))


# Case partition (setup). -------------------------------------------------------------------------

# Integer extents of the physics arguments that count the call's columns.
_COLUMN_COUNT_NAMES = ("im", "ncol", "n_cells_solve")


def _owned_columns(value, *, index: torch.Tensor, n_global: int, owned: int, path: str):
    if isinstance(value, torch.Tensor):
        # The column axis is the leading one, or the only column-sized one (MERRA-2 fields [.., ncol]).
        axes = [axis for axis, size in enumerate(value.shape) if size in (n_global, n_global + 1)]
        if not axes:
            return value
        axis = 0 if axes[0] == 0 else axes[0]
        if (axis != 0 and len(axes) > 1) or value.shape[axis] != n_global:
            raise ValueError(f"unclassified or padded physics column axis at {path}: {tuple(value.shape)}")
        return value.index_select(axis, index.to(value.device))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {}
        for field in dataclasses.fields(value):
            if not field.init:
                continue
            old = getattr(value, field.name)
            new = (_owned_count(old, field.name, n_global, owned, f"{path}.{field.name}")
                   if type(old) is int else
                   _owned_columns(old, index=index, n_global=n_global, owned=owned, path=f"{path}.{field.name}"))
            if new is not old:
                changes[field.name] = new
        return dataclasses.replace(value, **changes) if changes else value
    if isinstance(value, dict):
        result = {}
        for key, old in value.items():
            name = f"{path}[{key!r}]"
            result[key] = (_owned_count(old, key, n_global, owned, name) if type(old) is int else
                           _owned_columns(old, index=index, n_global=n_global, owned=owned, path=name))
        return type(value)(result) if type(value) is not dict else result
    if isinstance(value, list) and len(value) == n_global:  # one host entry per column (Noah-MP columns)
        return [value[i] for i in index.tolist()]
    if isinstance(value, (tuple, list)):
        items = [_owned_columns(v, index=index, n_global=n_global, owned=owned, path=f"{path}[{i}]")
                 for i, v in enumerate(value)]
        return type(value)(*items) if hasattr(value, "_fields") else type(value)(items)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        _require_no_columns(value, n_global, path)
    return value


def _owned_count(value: int, name, n_global: int, owned: int, path: str) -> int:
    if value != n_global:
        return value
    if name in _COLUMN_COUNT_NAMES:
        return owned
    raise ValueError(f"unclassified global column count at {path}")


def _require_no_columns(value, n_global: int, path: str, depth: int = 0) -> None:
    """Objects this partition does not rebuild must hold no per-column tensors."""
    if depth > 4:
        return
    items = (vars(value).items() if hasattr(value, "__dict__") and not isinstance(value, type)
             else value.items() if isinstance(value, dict)
             else enumerate(value) if isinstance(value, (tuple, list)) else ())
    for key, item in items:
        if isinstance(item, torch.Tensor):
            if item.ndim and item.shape[0] in (n_global, n_global + 1):
                raise ValueError(f"per-column tensor in an object the partition keeps: {path}.{key}")
        elif isinstance(item, (dict, tuple, list)) or hasattr(item, "__dict__"):
            _require_no_columns(item, n_global, f"{path}.{key}", depth + 1)


def _partition_input(case):
    from mpas_pytorch.state import _BOUNDARY_MARKER, has_complete_step_boundary

    marker = case["native"].get(_BOUNDARY_MARKER)
    if marker is None:
        return case
    if not has_complete_step_boundary(case["native"], mesh=case["mesh"], config=case["config"], dt=marker.get("dt")):
        raise ValueError("cannot partition an edited complete-step boundary")
    # A whole-mesh receipt cannot authenticate local indexing. The cold cycle
    # binds local state; no shard needs the global receipt's byte snapshots.
    return {**case, "native": {k: v for k, v in case["native"].items() if k != _BOUNDARY_MARKER}}


def partition_coupled_case(case: Mapping[str, Any], partitions, rank: int) -> dict[str, Any]:
    """One rank's coupled case from the global case (setup; no numerics).

    Native state and mesh: owned plus halo rows of every entity and the padding row, with local
    connectivity (``mpas_pytorch.distributed.partition_initial_state``). Physics arguments: the
    owned columns in owned-row order, column counts set to the owned count, the commit domain to
    the owned cells and edges, radiation seeds keyed by the owned global cell IDs.
    """
    from mpas_pytorch.distributed.partition import partition_initial_state

    from .commit import CommitDomain

    part = partitions[rank]
    case = _partition_input(case)
    native, mesh = partition_initial_state(case["native"], case["mesh"], part)
    cells, edges = part.cells, part.edges
    owned_ids = cells.global_ids[: cells.owned_count]
    n_global, owned = cells.global_count, cells.owned_count
    if case["arguments"]["n_cells_global"] != n_global:
        raise ValueError("case and partition disagree on the global cell count")
    if not torch.equal(case["arguments"]["global_ids"].cpu(), torch.arange(1, n_global + 1)):
        raise ValueError("the global case must list global cell IDs 1..nCells in order")
    config = dict(case["config"])
    special = {"mesh", "config", "domain", "global_ids", "n_cells_global"}
    arguments = {name: _owned_columns(value, index=owned_ids, n_global=n_global, owned=owned,
                                      path=f"arguments.{name}")
                 for name, value in case["arguments"].items() if name not in special}
    arguments.update(mesh=mesh, config=config, n_cells_global=n_global,
                     domain=CommitDomain(owned, edges.owned_count, len(partitions)),
                     global_ids=(owned_ids + 1).to(case["arguments"]["global_ids"].dtype))
    cold = {name: _owned_columns(value, index=owned_ids, n_global=n_global, owned=owned,
                                 path=f"cold_arguments.{name}")
            for name, value in case["cold_arguments"].items() if name not in ("n_cells_global", "mpi_size")}
    cold.update(n_cells_global=n_global, mpi_size=len(partitions))
    return {**case, "native": native, "mesh": mesh, "config": config, "arguments": arguments,
            "cold_arguments": cold}


def attach_exchanger(local_case: dict[str, Any], exchanger, *, setup_group=None) -> None:
    """Bind the run's halo exchanger (a private config key: outside the continuation identity)."""
    setup_group = setup_group if setup_group is not None else getattr(exchanger, "setup_group", None)
    if dist.is_initialized():
        if dist.get_backend() == "nccl" and setup_group is None:
            raise ValueError("NCCL coupling requires a separate Gloo setup/control group")
        if setup_group is not None and (
                dist.get_backend(setup_group) != "gloo"
                or dist.get_rank(setup_group) != dist.get_rank()
                or dist.get_world_size(setup_group) != dist.get_world_size()):
            raise ValueError("setup/control group must be the complete Gloo world in rank order")
    exchanger.setup_group = setup_group
    local_case["config"]["_halo_exchanger"] = exchanger
    local_case["arguments"]["config"] = local_case["config"]


@contextlib.contextmanager
def forecast_processes(device: str, *, timeout_seconds: int = 300):
    """Bind torchrun ranks to exclusive GPUs, with Gloo only for CPU setup/control.

    NCCL carries numerical CUDA communication. CPU uses the same spatial path
    with Gloo for CPU runs. Only groups created here are destroyed here.
    """
    target = torch.device(device)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 1 or timeout_seconds <= 0:
        raise ValueError("world size and distributed timeout must be positive")
    if world == 1:
        yield target, None
        return
    if dist.is_initialized():
        raise ValueError("the CLI must own its process groups; use run_forecast for caller-owned groups")
    if target.type not in ("cpu", "cuda"):
        raise ValueError("distributed forecasts require CPU/Gloo or CUDA/NCCL")
    backend = "gloo"
    if target.type == "cuda":
        local = int(os.environ["LOCAL_RANK"])
        if not torch.cuda.is_available() or not dist.is_nccl_available():
            raise RuntimeError("CUDA distributed forecasts require CUDA and NCCL")
        if local < 0 or local >= torch.cuda.device_count() or target.index not in (None, local):
            raise ValueError("each rank must use its LOCAL_RANK CUDA device")
        target = torch.device("cuda", local)
        torch.cuda.set_device(target)
        backend = "nccl"
    timeout = timedelta(seconds=timeout_seconds)
    setup_group = None
    try:
        dist.init_process_group(backend, timeout=timeout)
        if backend == "nccl":
            setup_group = dist.new_group(backend="gloo", timeout=timeout)
        yield target, setup_group
    finally:
        if setup_group is not None:
            dist.destroy_process_group(setup_group)
        if dist.is_initialized():
            dist.destroy_process_group()


def load_distributed_case(path, *, dtype, device, setup_group=None, physics_overrides=None, partition_path=None):
    """Load once on CPU root, then place only each rank's spatial shard on its device.

    ``partition_path`` names an MPAS ``graph.info.part.N`` file (N = world size) that sets the cell
    owners, as METIS writes it; without one, balanced spatial bisection decides."""
    from mpas_pytorch.distributed.dynamics import DynamicsHaloExchange
    from mpas_pytorch.distributed.halo import HaloTransport
    from mpas_pytorch.distributed.partition import build_mesh_partition, cell_owners
    from mpas_pytorch.distributed.setup import scatter_cpu_shards
    from .precision import prepare_case, resolve_dtype
    from .case_io import load_cpu_case

    if not dist.is_initialized() or dist.get_backend(setup_group) != "gloo":
        raise ValueError("distributed case loading requires a full-world Gloo setup group")
    rank, world = dist.get_rank(), dist.get_world_size()
    if dist.get_world_size(setup_group) != world or dist.get_rank(setup_group) != rank:
        raise ValueError("distributed case loading requires a full-world Gloo setup group")
    failure = None
    try:
        dtype = resolve_dtype(dtype) if dtype is not None else None
    except ValueError as error:
        failure = str(error)
    requests = [None] * world
    dist.all_gather_object(requests, (str(dtype), torch.device(device).type, failure), group=setup_group)
    if any(request[2] is not None for request in requests):
        raise ValueError(f"distributed precision setup failed: {requests}")
    if len(set(requests)) != 1:
        raise ValueError("all ranks must request the same precision and device type")
    payload, failure = None, [None]
    if rank == 0:
        try:
            case = _partition_input(load_cpu_case(path, physics_overrides=physics_overrides))
            case = prepare_case(case, dtype=dtype)
            owners = cell_owners(case["mesh"], world, partition_path)
            partitions = [build_mesh_partition(case["mesh"], owners, rank=r, world_size=world)
                          for r in range(world)]
            payload = [(partition_coupled_case(case, partitions, r), partitions) for r in range(world)]
            del case
        except Exception as error:
            failure[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(failure, src=0, group=setup_group)
    if failure[0] is not None:
        raise ValueError(f"distributed case setup failed: {failure[0]}")
    received = [None]
    scatter_cpu_shards(received, payload, group=setup_group)
    del payload
    case, partitions = received.pop()
    failure = None
    try:
        case = prepare_case(case, device=device)
    except Exception as error:
        failure = f"rank {rank}: {type(error).__name__}: {error}"
    failures = [None] * world
    dist.all_gather_object(failures, failure, group=setup_group)
    if any(failures):
        raise ValueError(f"distributed device placement failed: {failures}")
    exchanger = DynamicsHaloExchange(partitions, HaloTransport(device))
    attach_exchanger(case, exchanger, setup_group=setup_group)
    return case
