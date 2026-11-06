# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Packed neighbor copies and their transpose for spatial MPAS partitions.

Communication is a policy around the same tensor arithmetic. It does not
average gradients, replicate the global state, or implement numerical kernels.
The initial policy completes each phase before returning; overlap comes after
complete distributed timestep correctness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from math import prod
from typing import Any, cast

import torch
import torch.distributed as dist

from .partition import MeshPartition


@dataclass(frozen=True)
class _TransferSchedule:
    """Peer order, message sizes and per-field slices for one field signature."""

    peers: tuple[int, ...]
    # peer -> (message elements, ((field position, rows, offset, elements), ...))
    send: dict[int, tuple[int, tuple[tuple[int, torch.Tensor, int, int], ...]]]
    recv: dict[int, tuple[int, tuple[tuple[int, torch.Tensor, int, int], ...]]]


@dataclass(frozen=True)
class HaloPlan:
    rank: int
    world_size: int
    local_count: int
    send: dict[int, torch.Tensor]
    recv: dict[int, torch.Tensor]
    index_device: torch.device | None = None
    returned_rows: torch.Tensor = dataclass_field(init=False, repr=False, compare=False)
    buffers: dict[object, tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]] = (
        dataclass_field(default_factory=dict, init=False, repr=False, compare=False)
    )
    schedules: dict[object, _TransferSchedule] = dataclass_field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not 0 <= self.rank < self.world_size or self.local_count < 0:
            raise ValueError("invalid halo rank or local entity count")
        for lists in (self.send, self.recv):
            for peer, indices in lists.items():
                if not 0 <= peer < self.world_size or peer == self.rank:
                    raise ValueError("invalid halo peer")
                if indices.dtype != torch.int64 or indices.ndim != 1:
                    raise ValueError("halo row indices must be int64 vectors")
                if bool(((indices < 0) | (indices >= self.local_count)).any()):
                    raise ValueError(
                        "halo row index addresses padding or is out of range"
                    )
                if indices.unique().numel() != indices.numel():
                    raise ValueError("duplicate row in a peer's halo list")
        if self.recv:
            rows = torch.cat(tuple(self.recv.values()))
            if rows.unique().numel() != rows.numel():
                raise ValueError("a halo row has more than one incoming owner")
        devices = {
            ids.device for lists in (self.send, self.recv) for ids in lists.values()
        }
        if len(devices) > 1:
            raise ValueError("halo index plans must share a device")
        device = self.index_device or next(iter(devices), torch.device("cpu"))
        if any(value != device for value in devices):
            raise ValueError("halo index plan differs from its bound device")
        object.__setattr__(
            self,
            "returned_rows",
            torch.unique(torch.cat(tuple(self.send.values())))
            if self.send
            else torch.empty(0, dtype=torch.int64, device=device),
        )


@dataclass(frozen=True)
class HaloGroupPlan:
    """One field-ordered mixed-entity message, with each field's native halo depth."""

    field_plans: tuple[HaloPlan, ...]
    buffers: dict[object, tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]] = (
        dataclass_field(default_factory=dict, init=False, repr=False, compare=False)
    )
    schedules: dict[object, _TransferSchedule] = dataclass_field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.field_plans or any(
            (p.rank, p.world_size) != (self.rank, self.world_size)
            for p in self.field_plans
        ):
            raise ValueError("mixed halo fields must share a nonempty process group")

    @property
    def rank(self) -> int:
        return self.field_plans[0].rank

    @property
    def world_size(self) -> int:
        return self.field_plans[0].world_size


def build_halo_plan(
    partitions: Sequence[MeshPartition],
    *,
    rank: int,
    entity: str,
    layers: tuple[int, ...],
) -> HaloPlan:
    """Address requested halo layers directly to their authoritative owner."""
    world_size = len(partitions)
    if not 0 <= rank < world_size or tuple(p.rank for p in partitions) != tuple(
        range(world_size)
    ):
        raise ValueError("partitions must be in unique rank order")
    if any(p.world_size != world_size for p in partitions):
        raise ValueError("partition process groups differ")
    if not layers or len(set(layers)) != len(layers) or min(layers) < 1:
        raise ValueError("halo layers must be distinct positive integers")
    here = partitions[rank].entity(entity)
    all_owned = torch.cat(
        [
            p.entity(entity).global_ids[: p.entity(entity).owned_count]
            for p in partitions
        ]
    )
    if not torch.equal(all_owned.sort().values, torch.arange(here.global_count)):
        raise ValueError("entity ownership is incomplete or duplicated")
    global_owners = torch.empty(here.global_count, dtype=torch.int64)
    for partition in partitions:
        e = partition.entity(entity)
        global_owners[e.global_ids[: e.owned_count]] = partition.rank
    send, recv = {}, {}
    for peer, partition in enumerate(partitions):
        there = partition.entity(entity)
        if there.global_count != here.global_count:
            raise ValueError("partitions refer to different global meshes")
        if not torch.equal(there.owners, global_owners[there.global_ids]):
            raise ValueError("halo ID and authoritative owner disagree")
        if peer == rank:
            continue
        recv_mask = (here.owners == peer) & torch.isin(
            here.layers, torch.tensor(layers)
        )
        incoming = torch.nonzero(recv_mask).flatten()
        send_mask = (there.owners == rank) & torch.isin(
            there.layers, torch.tensor(layers)
        )
        requested = there.global_ids[send_mask]
        outgoing = here.global_to_local[requested + 1] - 1
        if bool(((outgoing < 0) | (outgoing >= here.owned_count)).any()):
            raise ValueError("halo request does not address an owned source")
        if incoming.numel():
            recv[peer] = incoming
        if outgoing.numel():
            send[peer] = outgoing
    return HaloPlan(rank, world_size, here.local_count, send, recv)


class HaloTransport:
    """One collectively initialized transport, reused by all numerical phases."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        if not dist.is_initialized():
            raise RuntimeError("initialize the process group before halo transport")
        self.device = torch.device(device)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # Keep both objects: identity-key reuse must never alias an unrelated plan.
        self._bound_plans: dict[int, tuple[HaloPlan, HaloPlan]] = {}
        self._bound_groups: dict[int, tuple[HaloGroupPlan, HaloGroupPlan]] = {}
        if dist.get_backend() == "nccl":
            if self.device.type != "cuda":
                raise ValueError("NCCL halo transport requires a CUDA device")
            torch.cuda.set_device(self.device)
            self.device = torch.device("cuda", torch.cuda.current_device())
        ready = torch.ones(1, device=self.device, dtype=torch.int64)
        dist.all_reduce(ready)
        if int(ready) != self.world_size:
            raise RuntimeError("halo transport collective initialization failed")

    def bind_plan(self, plan: HaloPlan) -> HaloPlan:
        """Copy immutable topology once, including the reverse returned-row union."""
        cached = self._bound_plans.get(id(plan))
        if cached is not None:
            assert cached[0] is plan
            return cached[1]
        bound = HaloPlan(
            plan.rank,
            plan.world_size,
            plan.local_count,
            {
                peer: ids.to(device=self.device, copy=True)
                for peer, ids in plan.send.items()
            },
            {
                peer: ids.to(device=self.device, copy=True)
                for peer, ids in plan.recv.items()
            },
            index_device=self.device,
        )
        self._bound_plans[id(plan)] = (plan, bound)
        self._bound_plans[id(bound)] = (bound, bound)
        return bound

    def exchange(
        self,
        plan: HaloPlan | HaloGroupPlan,
        *fields: torch.Tensor,
        inplace: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "halo exchange must remain outside compiled numeric regions"
            )
        if plan.rank != self.rank or plan.world_size != self.world_size:
            raise ValueError("halo plan does not match the active process group")
        if not fields:
            raise ValueError("a halo phase must have at least one field")
        plans = (
            plan.field_plans
            if isinstance(plan, HaloGroupPlan)
            else (plan,) * len(fields)
        )
        if len(plans) != len(fields):
            raise ValueError("mixed halo plan and field count differ")
        for field, field_plan in zip(fields, plans, strict=True):
            # Plans never address the padding row, so it is optional; padding
            # an unpadded field would copy the whole field at every exchange.
            if field.ndim < 1 or field.shape[0] not in (
                field_plan.local_count,
                field_plan.local_count + 1,
            ):
                raise ValueError(
                    "halo fields must have their local entity rows, "
                    "optionally followed by one padding row"
                )
            if field.device != self.device or field.dtype != fields[0].dtype:
                raise ValueError("packed halo fields must share device and dtype")
        bound = (
            self.bind_group(plan)
            if isinstance(plan, HaloGroupPlan)
            else self.bind_plan(plan)
        )
        if inplace:
            if torch.is_grad_enabled() or any(
                field.requires_grad
                or torch.autograd.forward_ad.unpack_dual(field).tangent is not None
                or torch._C._functorch.is_functorch_wrapped_tensor(field)
                for field in fields
            ):
                raise ValueError("in-place halo writes require inference-owned buffers")
            return _transfer(bound, tuple(fields), transpose=False, inplace=True)
        return cast(tuple[torch.Tensor, ...], _HaloExchange.apply(bound, *fields))

    def bind_group(self, plan: HaloGroupPlan) -> HaloGroupPlan:
        cached = self._bound_groups.get(id(plan))
        if cached is not None:
            assert cached[0] is plan
            return cached[1]
        bound = HaloGroupPlan(tuple(self.bind_plan(p) for p in plan.field_plans))
        self._bound_groups[id(plan)] = (plan, bound)
        self._bound_groups[id(bound)] = (bound, bound)
        return bound


def _transfer(
    plan: HaloPlan | HaloGroupPlan,
    fields: tuple[torch.Tensor, ...],
    *,
    transpose: bool,
    inplace: bool = False,
) -> tuple[torch.Tensor, ...]:
    plans = (
        plan.field_plans if isinstance(plan, HaloGroupPlan) else (plan,) * len(fields)
    )
    tails = tuple(tuple(field.shape[1:]) for field in fields)
    schedule = plan.schedules.get((transpose, tails))
    if schedule is None:
        schedule = _build_schedule(plans, tails, transpose=transpose)
        plan.schedules[transpose, tails] = schedule
    stream = (
        torch.cuda.current_stream(fields[0].device).cuda_stream
        if fields[0].device.type == "cuda"
        else None
    )
    key = (
        transpose,
        tuple((tail, f.dtype, f.device) for tail, f in zip(tails, fields)),
        stream,
    )
    transformed = any(
        torch._C._functorch.is_functorch_wrapped_tensor(f) for f in fields
    )
    buffers = None if transformed else plan.buffers.get(key)
    if buffers is None:
        buffers = (
            {
                peer: fields[0].new_empty(size)
                for peer, (size, _) in schedule.send.items()
            },
            {
                peer: fields[0].new_empty(size)
                for peer, (size, _) in schedule.recv.items()
            },
        )
        # Function transforms cannot mutate a tensor captured from an earlier
        # invocation. They use fresh packing storage, with identical wire values.
        # Buffers allocated while a CUDA graph records belong to that graph's
        # pool; caching them would keep the pool alive after its release.
        recording = (
            fields[0].device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        )
        if not transformed and not recording:
            plan.buffers[key] = buffers
    sends, receives = buffers
    operations = []
    for peer in schedule.peers:
        if peer in sends:
            for position, ids, offset, size in schedule.send[peer][1]:
                field = fields[position]
                destination = sends[peer][offset : offset + size]
                if inplace:
                    # Inference-owned scratch can receive the gather directly.
                    # AD and function transforms keep their functional gather.
                    torch.index_select(
                        field,
                        0,
                        ids,
                        out=destination.view((ids.numel(),) + tails[position]),
                    )
                else:
                    destination.copy_(field.index_select(0, ids).reshape(-1))
            operations.append(dist.P2POp(dist.isend, sends[peer], peer))
        if peer in receives:
            operations.append(dist.P2POp(dist.irecv, receives[peer], peer))
    if operations:
        requests = dist.batch_isend_irecv(operations)
        for request in requests:
            request.wait()
    result = fields if inplace else tuple(field.clone() for field in fields)
    if transpose:
        # Incoming forward values overwrite previous halo values. Their VJP
        # has no identity contribution to the overwritten local input.
        for field, field_plan in zip(result, plans, strict=True):
            for indices in field_plan.recv.values():
                field.index_fill_(0, indices, 0)
            field.index_fill_(0, field_plan.returned_rows, 0)
    # Include the owner's direct contribution in global rank order, not always
    # first. The returned rows are unique within a peer, including in backward.
    peers = set(receives)
    if transpose and any(p.send for p in plans):
        peers.add(plan.rank)
    for peer in sorted(peers):
        if peer == plan.rank:
            for field, original, field_plan in zip(result, fields, plans, strict=True):
                returned_rows = field_plan.returned_rows
                if not returned_rows.numel():
                    continue
                field.index_copy_(
                    0,
                    returned_rows,
                    field.index_select(0, returned_rows)
                    + original.index_select(0, returned_rows),
                )
            continue
        for position, incoming_rows, offset, size in schedule.recv[peer][1]:
            field = result[position]
            values = receives[peer][offset : offset + size].reshape(
                (incoming_rows.numel(),) + tails[position]
            )
            if transpose:
                # A peer requests unique rows; peer contributions are added in
                # rank order. No atomics or averaging are needed here.
                field.index_copy_(
                    0, incoming_rows, field.index_select(0, incoming_rows) + values
                )
            else:
                field.index_copy_(0, incoming_rows, values)
    return result


def _build_schedule(
    plans: tuple[HaloPlan, ...],
    tails: tuple[tuple[int, ...], ...],
    *,
    transpose: bool,
) -> _TransferSchedule:
    """Field-ordered message layout; rebuilt only for a new field signature."""

    def layout(
        tables: tuple[dict[int, torch.Tensor], ...],
    ) -> dict[int, tuple[int, tuple[tuple[int, torch.Tensor, int, int], ...]]]:
        items: dict[int, list[tuple[int, torch.Tensor, int, int]]] = {}
        sizes: dict[int, int] = {}
        for position, (table, tail) in enumerate(zip(tables, tails, strict=True)):
            width = prod(tail)
            for peer, rows in table.items():
                size = rows.numel() * width
                offset = sizes.get(peer, 0)
                items.setdefault(peer, []).append((position, rows, offset, size))
                sizes[peer] = offset + size
        return {peer: (sizes[peer], tuple(items[peer])) for peer in items}

    send = layout(tuple(p.recv if transpose else p.send for p in plans))
    recv = layout(tuple(p.send if transpose else p.recv for p in plans))
    return _TransferSchedule(tuple(sorted(set(send) | set(recv))), send, recv)


class _HaloExchange(torch.autograd.Function):
    @staticmethod
    def forward(
        plan: HaloPlan | HaloGroupPlan, *fields: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return _transfer(plan, tuple(fields), transpose=False)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        ctx.plan = inputs[0]
        # The copy is linear; retaining the full pre-exchange tensors would
        # unnecessarily multiply reverse-mode storage at every halo boundary.
        ctx.field_specs = tuple((tuple(x.shape), x.dtype, x.device) for x in inputs[1:])

    @staticmethod
    def backward(ctx: Any, *gradients: torch.Tensor | None) -> tuple[Any, ...]:
        values = tuple(
            torch.zeros(shape, dtype=dtype, device=device) if grad is None else grad
            for (shape, dtype, device), grad in zip(
                ctx.field_specs, gradients, strict=True
            )
        )
        return (None, *_transfer(ctx.plan, values, transpose=True))

    @staticmethod
    def jvp(
        ctx: Any, plan_tangent: None, *tangents: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        values = tuple(
            torch.zeros(shape, dtype=dtype, device=device)
            if tangent is None
            else tangent
            for (shape, dtype, device), tangent in zip(
                ctx.field_specs, tangents, strict=True
            )
        )
        return _transfer(ctx.plan, values, transpose=False)
