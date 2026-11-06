# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""MPAS ownership and local indexing, without a second numerical model.

Setup operates on the existing loader's one-based, last-row-padded mesh. Entity
IDs in a partition are zero-based global row positions; stencil entries remain
one-based. Owned entities precede successive halos and the sentinel is last.
Full vertical columns are never split. Outer, unused stencil entries may be
truncated, but required stencil rows fail loudly if their sources are absent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_DIMENSIONS = {"cell": "nCells", "edge": "nEdges", "vertex": "nVertices"}
# Source entity and the entity addressed by each stencil slot. Slot order is
# deliberately not sorted during remapping.
_RELATIONS = {
    "cellsOnCell": ("cell", "cell"),
    "edgesOnCell": ("cell", "edge"),
    "verticesOnCell": ("cell", "vertex"),
    "cellsOnEdge": ("edge", "cell"),
    "verticesOnEdge": ("edge", "vertex"),
    "edgesOnEdge": ("edge", "edge"),
    "advCellsForEdge": ("edge", "cell"),
    "cellsOnVertex": ("vertex", "cell"),
    "edgesOnVertex": ("vertex", "edge"),
}


@dataclass(frozen=True)
class EntityPartition:
    global_count: int
    global_ids: torch.Tensor
    owners: torch.Tensor
    layers: torch.Tensor
    global_to_local: torch.Tensor
    owned_count: int

    @property
    def local_count(self) -> int:
        return self.global_ids.numel()

    def through_layer(self, layer: int) -> int:
        return int((self.layers <= layer).sum())

    def select(self, tensor: torch.Tensor) -> torch.Tensor:
        """Select horizontal rows, retaining an existing padding row exactly."""
        if tensor.ndim == 0 or tensor.shape[0] not in (
            self.global_count,
            self.global_count + 1,
        ):
            raise ValueError("field does not match its global entity dimension")
        indices = self.global_ids
        if tensor.shape[0] == self.global_count + 1:
            indices = torch.cat((indices, torch.tensor([self.global_count])))
        return tensor.index_select(0, indices.to(device=tensor.device))

    def remap(self, entries: torch.Tensor) -> torch.Tensor:
        """Convert global one-based slots without changing their order."""
        if entries.dtype not in (torch.int32, torch.int64):
            raise ValueError("connectivity must be integer-valued")
        if bool(((entries < 0) | (entries > self.global_count + 1)).any()):
            raise ValueError("connectivity contains an invalid global ID")
        lookup = self.global_to_local.to(device=entries.device)
        return lookup[entries.long()].to(dtype=entries.dtype)


@dataclass(frozen=True)
class MeshPartition:
    rank: int
    world_size: int
    num_cell_halos: int
    cells: EntityPartition
    edges: EntityPartition
    vertices: EntityPartition

    def entity(self, kind: str) -> EntityPartition:
        return {"cell": self.cells, "edge": self.edges, "vertex": self.vertices}[kind]

    def required_rows(self, relation: str) -> int:
        source, _ = _RELATIONS[relation]
        entity = self.entity(source)
        # All incident edges/vertices of retained cells must exist. For the
        # neighbor-cell relation, checking halo one establishes halo two.
        if relation in ("edgesOnCell", "verticesOnCell"):
            return entity.local_count
        return entity.through_layer(1)


def read_cell_partition(
    path: str | Path, *, n_cells: int, world_size: int
) -> torch.Tensor:
    """Read the ordinary graph.info.part.N format (one rank per cell row).

    N must equal ``world_size``: every rank owns at least one cell, as METIS writes it."""
    values = [int(line) for line in Path(path).read_text().splitlines() if line.strip()]
    result = torch.tensor(values, dtype=torch.int64)
    _validate_owners(result, n_cells=n_cells, world_size=world_size)
    empty = (torch.bincount(result, minlength=world_size) == 0).nonzero().flatten()
    if empty.numel():
        raise ValueError(
            f"partition {path} gives no cells to ranks {empty.tolist()}: "
            f"use a graph.info.part.{world_size} file for {world_size} ranks"
        )
    return result


def cell_owners(
    mesh: Mapping[str, Any], world_size: int, partition_path: str | Path | None = None
) -> torch.Tensor:
    """Cell owners from an MPAS graph.info.part.N file, or balanced spatial bisection without one."""
    if partition_path is None:
        return spatial_cell_owners(mesh, world_size)
    return read_cell_partition(
        partition_path, n_cells=int(mesh["nCells"]), world_size=world_size
    )


def _validate_owners(owners: torch.Tensor, *, n_cells: int, world_size: int) -> None:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if owners.dtype != torch.int64 or owners.shape != (n_cells,):
        raise ValueError("cell owners must be an int64 vector with one entry per cell")
    if bool(((owners < 0) | (owners >= world_size)).any()):
        raise ValueError("cell owner is outside the process group")


def spatial_cell_owners(mesh: Mapping[str, Any], world_size: int) -> torch.Tensor:
    """Balanced spatial bisection when an existing METIS partition is unavailable.

    This is a setup policy, not a claim of METIS-quality edge cuts. Coordinates
    are consumed only here; the numerical kernels are unchanged.
    """
    n_cells = int(mesh["nCells"])
    if not 1 <= world_size <= n_cells:
        raise ValueError("spatial partition count must be between one and nCells")
    xyz = torch.stack(
        [mesh[key][:n_cells].detach().cpu() for key in ("xCell", "yCell", "zCell")],
        dim=1,
    )
    if not bool(torch.isfinite(xyz).all()):
        raise ValueError("partition coordinates must be finite")
    owners = torch.empty(n_cells, dtype=torch.int64)

    def split(ids: torch.Tensor, first_rank: int, count: int) -> None:
        if count == 1:
            owners[ids] = first_rank
            return
        points = xyz[ids]
        axis = int((points.amax(dim=0) - points.amin(dim=0)).argmax())
        ordered = ids[torch.argsort(points[:, axis], stable=True)]
        left_count = count // 2
        cut = ordered.numel() * left_count // count
        split(ordered[:cut], first_rank, left_count)
        split(ordered[cut:], first_rank + left_count, count - left_count)

    split(torch.arange(n_cells), 0, world_size)
    return owners


OWNED_ORDERS = ("global", "hilbert")


def _hilbert_keys(xyz: torch.Tensor, bits: int = 16) -> torch.Tensor:
    """Index of each point along a 3-D Hilbert curve (Skilling's method).

    Used only to order owned rows at setup; numerical kernels never see it.
    """
    span = (xyz.amax(dim=0) - xyz.amin(dim=0)).clamp_min(1e-300)
    q = ((xyz - xyz.amin(dim=0)) / span * (2**bits - 1)).to(torch.int64)
    x = [q[:, i].clone() for i in range(3)]
    m = 1 << (bits - 1)
    bit = m
    while bit > 1:
        low = bit - 1
        for i in range(3):
            high = (x[i] & bit) != 0
            x[0] = torch.where(high, x[0] ^ low, x[0])
            swap = torch.where(high, torch.zeros_like(x[0]), (x[0] ^ x[i]) & low)
            x[0] = x[0] ^ swap
            x[i] = x[i] ^ swap
        bit >>= 1
    for i in range(1, 3):
        x[i] = x[i] ^ x[i - 1]
    flip = torch.zeros_like(x[0])
    bit = m
    while bit > 1:
        flip = torch.where((x[2] & bit) != 0, flip ^ (bit - 1), flip)
        bit >>= 1
    for i in range(3):
        x[i] = x[i] ^ flip
    key = torch.zeros_like(x[0])
    for b in range(bits - 1, -1, -1):
        for i in range(3):
            key = (key << 1) | ((x[i] >> b) & 1)
    return key


def _owned_order_keys(
    mesh: Mapping[str, Any], relations: Mapping[str, torch.Tensor], n_cells: int
) -> dict[str, torch.Tensor]:
    """Hilbert keys of cell centres; edges and vertices use their cells' mean."""
    xyz = torch.stack(
        [mesh[key][:n_cells].detach().cpu() for key in ("xCell", "yCell", "zCell")],
        dim=1,
    ).to(torch.float64)
    keys = {"cell": _hilbert_keys(xyz)}
    for kind, relation in (("edge", "cellsOnEdge"), ("vertex", "cellsOnVertex")):
        cells = relations[relation]
        valid = (cells > 0) & (cells <= n_cells)
        points = xyz[(cells - 1).clamp(0, n_cells - 1)] * valid[..., None]
        centre = points.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        keys[kind] = _hilbert_keys(centre)
    return keys


def _owner_from_first_cell(
    entries: torch.Tensor, cell_owners: torch.Tensor
) -> torch.Tensor:
    valid = (entries > 0) & (entries <= cell_owners.numel())
    if not bool(valid.any(dim=1).all()):
        raise ValueError("edge or vertex has no valid adjacent cell")
    first = valid.long().argmax(dim=1)
    cells = entries.gather(1, first[:, None]).squeeze(1) - 1
    return cell_owners[cells.long()]


# Layer of clone rows: copies of owned rows appended to reach a row count that
# lets compiled kernels drop bounds masks. Never owned, exchanged or output.
CLONE_LAYER = 99


def _clone_rows(
    count: int, owned: int, row_multiple: int | None, max_padding: float
) -> int:
    """Rows to append so ``count`` becomes a multiple of a block size.

    Tries ``row_multiple`` and its halves down to 16 and takes the first whose
    padding stays within ``max_padding`` of ``count`` (and of the owned rows
    available to clone).
    """
    multiple = row_multiple or 0
    while multiple >= 16:
        extra = -count % multiple
        if extra <= max_padding * count and extra <= owned:
            return extra
        multiple //= 2
    return 0


def _entity_partition(
    owners: torch.Tensor,
    layers: torch.Tensor,
    rank: int,
    row_multiple: int | None = None,
    max_padding: float = 0.05,
    owned_key: torch.Tensor | None = None,
) -> EntityPartition:
    # Stable order within a layer is global row order. This does not reorder
    # neighbor slots, edge orientations, or coefficients.
    present = torch.nonzero(layers >= 0).flatten()
    ids = present[torch.argsort(layers[present], stable=True)]
    owned = int((owners == rank).sum())
    if owned_key is not None:
        # Owned rows (layer 0) follow the key; halo layers keep global order.
        head = ids[:owned]
        ids = torch.cat((head[torch.argsort(owned_key[head], stable=True)], ids[owned:]))
    lookup = torch.full((owners.numel() + 2,), -1, dtype=torch.int64)
    lookup[0] = 0
    lookup[ids + 1] = torch.arange(1, ids.numel() + 1)
    extra = _clone_rows(ids.numel(), owned, row_multiple, max_padding)
    # Clone rows repeat the first owned rows; the lookup keeps the originals.
    local_layers = torch.cat(
        (layers[ids], torch.full((extra,), CLONE_LAYER, dtype=layers.dtype))
    )
    ids = torch.cat((ids, ids[:extra]))
    lookup[-1] = ids.numel() + 1
    return EntityPartition(
        owners.numel(),
        ids,
        owners[ids],
        local_layers,
        lookup,
        owned,
    )


def build_mesh_partition(
    mesh: Mapping[str, Any],
    cell_owners: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    num_cell_halos: int = 2,
    row_multiple: int | None = None,
    max_padding: float = 0.05,
    owned_order: str = "global",
) -> MeshPartition:
    """Construct cell rings and incident edge/vertex rings following MPAS.

    ``owned_order="hilbert"`` orders each entity's owned rows along a Hilbert
    curve of their centres instead of by global ID, so neighbouring rows sit
    close in memory. Halo layers, slots and arithmetic are unchanged.

    ``row_multiple`` appends clone rows (``CLONE_LAYER``) so each entity's
    local row count is a multiple of it, or of a power-of-two divisor, when
    that adds at most ``max_padding`` of the rows. Kernels over all local rows
    then need no bounds masks. Owned, halo and output rows are unchanged.
    """
    counts = {kind: int(mesh[name]) for kind, name in _DIMENSIONS.items()}
    owners = cell_owners.detach().cpu()
    _validate_owners(owners, n_cells=counts["cell"], world_size=world_size)
    if not 0 <= rank < world_size or num_cell_halos < 2:
        raise ValueError("invalid rank or fewer than the required two cell halos")
    relations = {}
    for name, (source, _) in _RELATIONS.items():
        if name not in mesh:
            continue
        value = mesh[name]
        if value.dtype not in (torch.int32, torch.int64) or value.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional integer stencil")
        if value.shape[0] not in (counts[source], counts[source] + 1):
            raise ValueError(f"{name} has the wrong source dimension")
        relations[name] = value[: counts[source]].detach().cpu().long()
    for name in (
        "cellsOnCell",
        "edgesOnCell",
        "verticesOnCell",
        "cellsOnEdge",
        "cellsOnVertex",
    ):
        if name not in relations:
            raise ValueError(f"partition setup requires {name}")
    for name, (_, target) in _RELATIONS.items():
        if name in relations and bool(
            ((relations[name] < 0) | (relations[name] > counts[target] + 1)).any()
        ):
            raise ValueError(f"{name} contains an invalid global ID")
    cell_layers = torch.full((counts["cell"],), -1, dtype=torch.int64)
    cell_layers[owners == rank] = 0
    for layer in range(1, num_cell_halos + 1):
        slots = relations["cellsOnCell"][cell_layers == layer - 1].flatten()
        ids = slots[(slots > 0) & (slots <= counts["cell"])] - 1
        ids = ids[cell_layers[ids] < 0]
        cell_layers[ids] = layer

    if owned_order not in OWNED_ORDERS:
        raise ValueError(f"owned_order must be one of {OWNED_ORDERS}")
    keys = (
        _owned_order_keys(mesh, relations, counts["cell"])
        if owned_order == "hilbert"
        else {}
    )
    entity_parts = {
        "cell": _entity_partition(
            owners, cell_layers, rank, row_multiple, max_padding, keys.get("cell")
        )
    }
    for kind, relation, incident in (
        ("edge", "cellsOnEdge", "edgesOnCell"),
        ("vertex", "cellsOnVertex", "verticesOnCell"),
    ):
        entity_owners = _owner_from_first_cell(relations[relation], owners)
        layers = torch.full((counts[kind],), -1, dtype=torch.int64)
        layers[entity_owners == rank] = 0
        # Halo one contains non-owned entities touching owned cells; each
        # subsequent cell ring contributes the next edge/vertex layer.
        for cell_layer in range(num_cell_halos + 1):
            slots = relations[incident][cell_layers == cell_layer].flatten()
            ids = slots[(slots > 0) & (slots <= counts[kind])] - 1
            ids = ids[layers[ids] < 0]
            layers[ids] = cell_layer + 1
        entity_parts[kind] = _entity_partition(
            entity_owners, layers, rank, row_multiple, max_padding, keys.get(kind)
        )
    return MeshPartition(
        rank,
        world_size,
        num_cell_halos,
        entity_parts["cell"],
        entity_parts["edge"],
        entity_parts["vertex"],
    )


def partition_initial_state(
    state: Mapping[str, Any],
    mesh: Mapping[str, Any],
    partition: MeshPartition,
    *,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Slice existing cold-start results before moving a shard to its device.

    No initialization equations are duplicated. Call the ordinary loader on CPU
    once at setup, then release the global cold-start dictionaries after sharding.
    This is not yet a distributed run_timestep entry point.
    """
    target_device = torch.device(device)
    for kind, dimension in _DIMENSIONS.items():
        if int(mesh[dimension]) != partition.entity(kind).global_count:
            raise ValueError(f"partition and mesh disagree on {dimension}")
    converted: dict[int, torch.Tensor] = {}

    def localize(values: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                result[name] = value
                continue
            if name in _DIMENSIONS.values():
                kind = next(k for k, dim in _DIMENSIONS.items() if dim == name)
                result[name] = value.new_tensor(partition.entity(kind).local_count).to(
                    target_device
                )
                continue
            if id(value) in converted:
                result[name] = converted[id(value)]
                continue
            kinds = [
                kind
                for kind in _DIMENSIONS
                if value.ndim > 0
                and value.shape[0]
                in (
                    partition.entity(kind).global_count,
                    partition.entity(kind).global_count + 1,
                )
            ]
            if len(kinds) > 1:
                raise ValueError(f"ambiguous horizontal dimension for {name}")
            local = partition.entity(kinds[0]).select(value) if kinds else value.clone()
            if name in _RELATIONS:
                source, destination = _RELATIONS[name]
                if kinds != [source]:
                    raise ValueError(f"{name} has the wrong source dimension")
                local = partition.entity(destination).remap(local)
                required = partition.required_rows(name)
                if bool((local[:required] < 0).any()):
                    raise ValueError(f"missing required stencil source in {name}")
                # MPAS block_creator.F maps out-of-block IDs to the final
                # garbage row, not to zero (which becomes -1 on conversion).
                # Required rows were checked above; only outer rows may use it.
                local = torch.where(
                    local < 0,
                    partition.entity(destination).local_count + 1,
                    local,
                )
            local = local.to(device=target_device)
            converted[id(value)] = local
            result[name] = local
        for kind, dimension in _DIMENSIONS.items():
            entity = partition.entity(kind)
            result[dimension] = entity.local_count
            result[f"{dimension}Solve"] = entity.owned_count
            result[f"{dimension}Global"] = entity.global_count
            # MPAS separates storage/diagnostic bounds from owned solve bounds.
            # Single-process defaults use every row, so they cannot be inherited
            # by a shard, including when the input contains explicit bounds.
            result[f"{kind}Start"] = 1
            result[f"{kind}End"] = entity.local_count
            result[f"{kind}SolveStart"] = 1
            result[f"{kind}SolveEnd"] = entity.owned_count
        return result

    return localize(state), localize(mesh)
