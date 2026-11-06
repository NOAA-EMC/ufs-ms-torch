# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""R4 owned-cell native commit ledgers and persistent limiter state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .containers import NativeAtmosphere, assert_all_finite, static_tensor
from .distributed import global_rows, rank_values
from .projection import TracerContract, _working_on_device
from .reverse import ThermodynamicCommit


@dataclass(frozen=True)
class CumulativeLimiter:
    count: torch.Tensor
    added_dry_weighted: torch.Tensor

    @classmethod
    def allocate(cls, *, ntrac: int, device: torch.device | str, dtype: torch.dtype = torch.float64) -> CumulativeLimiter:
        if type(ntrac) is not int or ntrac < 1:
            raise ValueError("limiter allocation requires a positive tracer extent")
        return cls(
            torch.zeros(ntrac, dtype=torch.int32, device=device),
            torch.zeros(ntrac, dtype=dtype, device=device),
        )


@dataclass(frozen=True)
class CommitLedgers:
    """Totals over all cells: on a partition, sums in global-ID order and global counts and minima."""

    limiter_count: torch.Tensor
    limiter_min_proposed: torch.Tensor
    limiter_added_mixing_ratio: torch.Tensor
    limiter_added_dry_weighted: torch.Tensor
    dry_air_mass: torch.Tensor
    water_before_mass: torch.Tensor
    water_after_mass: torch.Tensor
    water_delta_mass: torch.Tensor
    tracer_before_dry_weighted: torch.Tensor
    tracer_after_dry_weighted: torch.Tensor
    tracer_delta_dry_weighted: torch.Tensor
    ccpp_tracer_delta: torch.Tensor
    basis_reconciliation: torch.Tensor
    cumulative: CumulativeLimiter


def _ordered_totals(values: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
    # Prefix the source accumulator, including positive zero and restart carries.
    rows = values.reshape(-1, initial.numel())
    return torch.cumsum(torch.cat((initial[None, :], rows), dim=0), dim=0)[-1]


def _observer_totals(values: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
    """Source-ordered observer totals; off (NaN) on CUDA.

    The ordered scan runs serially per column on CUDA (about 5.5 s of an A100
    cycle), and no physics reads these totals.
    """
    if values.is_cuda:
        return torch.full_like(initial, float("nan"))
    # On a partition: every cell's rows in global-ID order, as one rank sums them.
    return _ordered_totals(global_rows(values), initial)


def commit_ledgers(
    before: NativeAtmosphere,
    committed: ThermodynamicCommit,
    *,
    tracers: TracerContract,
    water_species: tuple[bool, ...],
    cumulative: CumulativeLimiter | None = None,
) -> CommitLedgers:
    """Accumulate iCol -> iLay -> iTracer, using the source's commit operands.

    Per-tracer sums retain cell/level order; water sums also retain tracer order.
    Direct increments do not subtract two globally integrated reservoirs.
    This does not perform MPI transport, mutate carries or accept host-valued
    tensor controls. CPU accumulation is checked against the original loop.
    On CUDA the ordered observer totals are off: they and the carried
    cumulative limiter mass are NaN by design (no physics reads them), so a
    CUDA-run carry cannot resume a CPU run. The CUDA path instead checks that
    every per-cell operand is finite.
    """
    n, levels = committed.dry_layer_mass.shape
    if (
        type(before.n_cells_solve) is not int
        or n != before.n_cells_solve
        or n < 1
        or levels < 1
    ):
        raise ValueError("commit ledger operands do not cover owned cells")
    device = committed.dry_layer_mass.device
    _working_on_device("dry_layer_mass", committed.dry_layer_mass, device)
    # Every operand and the carried mass share the working dtype; a mixed carry would promote silently.
    dtype = committed.dry_layer_mass.dtype
    _working_on_device("native scalars", before.scalars, device, dtype)
    _working_on_device("committed scalars", committed.atmosphere.scalars, device, dtype)
    if (
        before.scalars.ndim != 3
        or before.scalars.shape[0] < n
        or before.scalars.shape[1] != levels
        or committed.atmosphere.scalars.shape != before.scalars.shape
        or committed.proposed_tracers.ndim != 3
    ):
        raise ValueError("invalid commit ledger scalar extents")
    nt = committed.proposed_tracers.shape[2]
    if tracers.validate(native_count=before.scalars.shape[2], target_count=nt) != nt:
        raise ValueError("commit ledger tracer extent differs from native commit")
    if len(water_species) != nt or any(type(v) is not bool for v in water_species):
        raise ValueError("commit ledger requires the source water-species map")
    shape = (n, levels, nt)
    for name in (
        "proposed_tracers",
        "limiter_correction",
        "native_tracer_increment",
        "ccpp_tracer_increment",
    ):
        value = getattr(committed, name)
        _working_on_device(name, value, device, dtype)
        if value.shape != shape:
            raise ValueError(f"invalid commit ledger operand extent: {name}")
    previous = cumulative or CumulativeLimiter.allocate(ntrac=nt, device=device, dtype=dtype)
    _working_on_device("cumulative limiter mass", previous.added_dry_weighted, device, dtype)
    if (
        previous.count.dtype != torch.int32
        or previous.count.device != device
        or previous.count.shape != (nt,)
        or previous.added_dry_weighted.shape != (nt,)
    ):
        raise ValueError("invalid cumulative limiter metadata")
    bindings = tracers.bindings[:nt]
    owned = static_tensor(
        tuple(b.ccpp_owned for b in bindings), dtype=torch.bool, device=device
    )
    slots = static_tensor(tuple(b.mpas_index for b in bindings), dtype=torch.int64, device=device)
    old = torch.where(owned, before.scalars[:n].index_select(2, slots), 0.0)
    new = torch.where(owned, committed.atmosphere.scalars[:n].index_select(2, slots), 0.0)
    mass = committed.dry_layer_mass[..., None]
    before_terms, after_terms = old * mass, new * mass
    negative = owned & (committed.proposed_tracers < 0.0)
    correction = torch.where(negative, committed.limiter_correction, 0.0)
    weighted = correction * mass
    native_delta = torch.where(owned, committed.native_tracer_increment, 0.0)
    ccpp_delta = torch.where(owned, committed.ccpp_tracer_increment, 0.0)
    zeros = mass.new_zeros((nt,))
    before_sum = _observer_totals(before_terms, zeros)
    after_sum = _observer_totals(after_terms, zeros)
    delta_sum = _observer_totals(native_delta, zeros)
    ccpp_sum = _observer_totals(ccpp_delta, zeros)
    added = _observer_totals(correction, zeros)
    added_weighted = _observer_totals(weighted, zeros)
    # Over all ranks (one rank: unchanged); integer counts and minima do not depend on the order.
    count = rank_values(negative.to(torch.int32).sum(dim=(0, 1), dtype=torch.int32)).sum(dim=0, dtype=torch.int32)
    minimum = rank_values(torch.where(negative, committed.proposed_tracers, 0.0).amin(dim=(0, 1))).amin(dim=0)
    cumulative_added = _observer_totals(weighted, previous.added_dry_weighted)
    # The source does not touch an inactive cumulative slot, even its zero sign.
    cumulative_added = torch.where(
        count > 0, cumulative_added, previous.added_dry_weighted
    )
    water = tuple(i for i, active in enumerate(water_species) if active)
    if water and mass.is_cuda:
        water_sums = mass.new_full((3,), float("nan"))  # observer totals are off on CUDA
    elif water:
        water_index = static_tensor(water, dtype=torch.int64, device=device)
        water_terms = torch.stack(
            (
                before_terms.index_select(-1, water_index),
                after_terms.index_select(-1, water_index),
                native_delta.index_select(-1, water_index),
            ),
            dim=-1,
        )
        water_sums = _observer_totals(water_terms, mass.new_zeros((3,)))
    else:
        water_sums = mass.new_zeros((3,))
    dry_sum = _observer_totals(
        committed.dry_layer_mass[..., None], mass.new_zeros((1,))
    )[0]
    result = CommitLedgers(
        count,
        minimum,
        added,
        added_weighted,
        dry_sum,
        water_sums[0],
        water_sums[1],
        water_sums[2],
        before_sum,
        after_sum,
        delta_sum,
        ccpp_sum,
        delta_sum - ccpp_sum,
        CumulativeLimiter(previous.count + count, cumulative_added),
    )
    if mass.is_cuda:  # the totals are off; check the operands they would sum instead
        assert_all_finite(dict(before_terms=before_terms, after_terms=after_terms, weighted=weighted,
                               native_delta=native_delta, ccpp_delta=ccpp_delta, correction=correction,
                               dry_layer_mass=committed.dry_layer_mass), "native commit ledger operand")
    # On CUDA only the limiter minimum is computed; the ordered totals are off.
    for value in (minimum,) if mass.is_cuda else (
        minimum,
        added,
        added_weighted,
        dry_sum,
        water_sums,
        before_sum,
        after_sum,
        delta_sum,
        ccpp_sum,
        result.basis_reconciliation,
        cumulative_added,
    ):
        torch._assert_async(
            torch.isfinite(value).all(), "nonfinite native commit ledger"
        )
    return result
