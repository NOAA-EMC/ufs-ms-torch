# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
Scalar advancement - monotonic (monotonic-scalar-transport).

Advances scalar (tracer) fields using explicit time integration with
flux-corrected transport (FCT) for monotonicity preservation.

This kernel corresponds to Fortran subroutine atm_advance_scalars_mono_work.

Algorithm overview (Flux-Corrected Transport):
1. Apply source tendencies to scalars_old (initial update)
2. Compute rho_zz_int (intermediate density) from divergence
3. For each scalar:
   a. Copy scalar_old/scalar_new from scalars arrays
   b. Compute vertical fluxes (wdtn) and s_max/s_min bounds
   c. Compute high-order horizontal fluxes (flux_arr)
   d. Compute upwind-updated scalar_new and accumulate flux scales
   e. Compute flux limiters (scale_arr) based on s_max/s_min bounds
   f. Rescale fluxes to maintain monotonicity
   g. Final scalar update with limited fluxes
   h. Copy back to scalars_new with positive-definiteness enforcement

Index Convention:
    - Loop indices (cellStart, edgeStart, etc.) are one-based values
    - Arrays use 0-based Python indexing:
      * Valid cells at indices 0 to nCells-1 (corresponding to MPAS cells 1..nCells)
      * Padding cell at index nCells (corresponding to MPAS cell nCells+1)
    - Connectivity arrays (cellsOnEdge, etc.) contain one-based cell/edge indices
      -> Must subtract 1 to get Python indices
    - Vertical level arrays (fnm, rdnw): 0-indexed (0 to nVertLevels-1)
"""

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import torch

from mpas_pytorch.autodiff import is_grad_mode
from mpas_pytorch.autodiff.soft_limiter import (
    fct_abs,
    fct_clamp_01,
    fct_max,
    fct_max_zero,
    fct_min,
    fct_neighbor_bounds,
    fct_quotient,
    fct_sign_pm,
)
from mpas_pytorch.distributed.dynamics import current_dynamics_halo_exchange
from mpas_pytorch.dynamics.dyn_tend import _make_flux_divisor, _validate_flux_divisor
from mpas_pytorch.dynamics.pattern_c import (
    PatternCOperator,
    _build_pattern_c_connectivity_payload,
)
from mpas_pytorch.dynamics.static_payloads import (
    StaticPairRelation,
    StaticRange,
    StaticSlotRelation,
)
from mpas_pytorch.dynamics.stencil_ops import (
    _accumulate_ordered_slot,
    _active_slot_count,
    _clone_for_grad,
    _compile_active,
    _gather_slot_batch,
    _replace_all_rows_cols,
    _replace_row_slice,
    _replace_rows,
    _slot_batches,
    _slot_gather_batch,
    _static_connectivity_cache_allowed,
    _static_mask_all_true,
    _static_slot_all_true,
    _tensor_carries_autodiff,
    _tensor_version_cache_key,
    gather_rows,
    sum_fortran_order,
)
from mpas_pytorch.profiling import profile_section
from mpas_pytorch.utils.checks import require_consistent_device

_MONOTONIC_SCALAR_TRANSPORT_CONNECTIVITY_TENSOR_NAMES = (
    "edgesOnCell",
    "nEdgesOnCell",
    "edgesOnCell_sign",
    "cellsOnCell",
    "cellsOnEdge",
    "bdyMaskCell",
    "nAdvCellsForEdge",
    "advCellsForEdge",
    "adv_coefs",
    "adv_coefs_3rd",
    "dvEdge",
    "invAreaCell",
)

_MONOTONIC_SCALAR_TRANSPORT_REGION_COMPILE_TARGETS = frozenset(
    {
        "monotonic_scalar_transport_pattern_c_flux_region",
        "monotonic_scalar_transport_streaming_scalar_transport_region",
        "monotonic_scalar_transport_upwind_scale_prep_region",
        "monotonic_scalar_transport_limiter_flux_region",
        "monotonic_scalar_transport_final_update_region",
        "monotonic_scalar_transport_vertical_neighbor_bounds_region",
        "monotonic_scalar_transport_source_tendency_region",
        "monotonic_scalar_transport_density_update_region",
    }
)


def _flux4(
    q_im2: torch.Tensor,
    q_im1: torch.Tensor,
    q_i: torch.Tensor,
    q_ip1: torch.Tensor,
    ua: torch.Tensor,
    divisor: float | torch.Tensor = 12.0,
) -> torch.Tensor:
    """
    Fourth-order advection flux.

    MPAS expression: flux4(q_im2, q_im1, q_i, q_ip1, ua) = ua*(7*(q_i + q_im1) - (q_ip1 + q_im2))/12.0
    """
    return ua * (7.0 * (q_i + q_im1) - (q_ip1 + q_im2)) / divisor


def _flux3(
    q_im2: torch.Tensor,
    q_im1: torch.Tensor,
    q_i: torch.Tensor,
    q_ip1: torch.Tensor,
    ua: torch.Tensor,
    coef3: float,
    divisor: float | torch.Tensor = 12.0,
) -> torch.Tensor:
    """
    Third-order advection flux with upwind correction.

    MPAS expression:
        flux3(q_im2, q_im1, q_i, q_ip1, ua, coef3) =
            flux4(q_im2, q_im1, q_i, q_ip1, ua) +
            coef3*abs(ua)*((q_ip1 - q_im2) - 3*(q_i - q_im1))/12.0
    """
    f4 = _flux4(q_im2, q_im1, q_i, q_ip1, ua, divisor)
    upwind_corr = (
        coef3 * fct_abs(ua) * ((q_ip1 - q_im2) - 3.0 * (q_i - q_im1)) / divisor
    )
    return f4 + upwind_corr


def _monotonic_scalar_transport_positive_scalar_all_cell_patch(
    *,
    scalar_new_all_patch: torch.Tensor,
    scalar_stage2: torch.Tensor,
    solve_rel_start: int,
    solve_rel_end: int,
    tau: float,
) -> torch.Tensor:
    """Apply positivity without cloning an all-cell tensor for solve-row replacement."""
    solve_rel_start = int(solve_rel_start)
    solve_rel_end = int(solve_rel_end)
    if solve_rel_start < 0 or solve_rel_end < solve_rel_start:
        raise ValueError(f"invalid solve row slice [{solve_rel_start}:{solve_rel_end}]")
    if solve_rel_end > scalar_new_all_patch.shape[0]:
        raise ValueError(
            "solve row slice exceeds all-cell patch rows: "
            f"{solve_rel_end} > {scalar_new_all_patch.shape[0]}"
        )
    if scalar_stage2.shape[0] != solve_rel_end - solve_rel_start:
        raise ValueError(
            "solve row slice length does not match scalar_stage2 rows: "
            f"{solve_rel_end - solve_rel_start} != {scalar_stage2.shape[0]}"
        )

    positive_solve = fct_max_zero(scalar_stage2, tau=tau)
    if solve_rel_start == 0 and solve_rel_end == scalar_new_all_patch.shape[0]:
        return positive_solve

    parts: list[torch.Tensor] = []
    if solve_rel_start > 0:
        parts.append(
            fct_max_zero(scalar_new_all_patch[:solve_rel_start, :, :], tau=tau)
        )
    parts.append(positive_solve)
    if solve_rel_end < scalar_new_all_patch.shape[0]:
        parts.append(fct_max_zero(scalar_new_all_patch[solve_rel_end:, :, :], tau=tau))
    return torch.cat(parts, dim=0)


def _full_level_vertical_flux_delta(
    interface_flux: torch.Tensor,
    n_levels: int,
) -> torch.Tensor:
    """Match padded[:, 1:n+1, :] - padded[:, :n, :] with zero boundary fluxes."""
    expected_interfaces = max(n_levels - 1, 0)
    if interface_flux.shape[1] != expected_interfaces:
        raise ValueError(
            "interface_flux has wrong vertical width: "
            f"got {interface_flux.shape[1]}, expected {expected_interfaces}"
        )
    if n_levels == 0:
        return interface_flux.new_zeros(
            (interface_flux.shape[0], 0, interface_flux.shape[2])
        )
    if expected_interfaces == 0:
        return interface_flux.new_zeros(
            (interface_flux.shape[0], n_levels, interface_flux.shape[2])
        )
    out = interface_flux.new_empty(
        (interface_flux.shape[0], n_levels, interface_flux.shape[2])
    )
    out[:, 0, :] = interface_flux[:, 0, :]
    if n_levels > 2:
        out[:, 1:-1, :] = interface_flux[:, 1:, :] - interface_flux[:, :-1, :]
    out[:, -1, :] = -interface_flux[:, -1, :]
    return out


def _monotonic_scalar_transport_apply_vertical_upwind_updates(
    *,
    scalar_old_rows: torch.Tensor,
    rho_zz_old_rows: torch.Tensor,
    flux_upwind_mid: torch.Tensor,
    rdnw: torch.Tensor,
) -> torch.Tensor:
    """Apply the two Fortran vertical upwind loops in their original order."""
    n_levels = int(scalar_old_rows.shape[1])
    scalar_new_rows = scalar_old_rows * rho_zz_old_rows.unsqueeze(2)
    if n_levels <= 1:
        return scalar_new_rows
    outgoing = scalar_new_rows[:, : n_levels - 1, :] - flux_upwind_mid * rdnw[
        : n_levels - 1
    ].view(1, n_levels - 1, 1)
    scalar_new_rows = _replace_all_rows_cols(
        scalar_new_rows,
        0,
        n_levels - 1,
        outgoing,
    )
    incoming = scalar_new_rows[:, 1:n_levels, :] + flux_upwind_mid * rdnw[
        1:n_levels
    ].view(1, n_levels - 1, 1)
    return _replace_all_rows_cols(
        scalar_new_rows,
        1,
        n_levels,
        incoming,
    )


def _monotonic_scalar_transport_scale_patches_from_mid_flux(
    *,
    wdtn_patch: torch.Tensor,
    wdtn_mid_patch: torch.Tensor,
    rdnw: torch.Tensor,
    n_vert_levels: int,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute limiter scale patches without materializing padded wdtn rows."""
    shape = (wdtn_patch.shape[0], n_vert_levels, wdtn_patch.shape[2])
    scale_in_patch = wdtn_patch.new_empty(shape)
    scale_out_patch = wdtn_patch.new_empty(shape)
    if n_vert_levels == 0:
        return scale_in_patch, scale_out_patch

    first_prev = wdtn_patch[:, 0:1, :]
    first_next = (
        wdtn_mid_patch[:, 0:1, :] if n_vert_levels > 1 else wdtn_patch[:, 1:2, :]
    )
    first_rdnw = rdnw[0:1].view(1, 1, 1)
    scale_in_patch[:, 0:1, :] = -first_rdnw * (
        (-fct_max_zero(-first_next, tau=tau)) - fct_max_zero(first_prev, tau=tau)
    )
    scale_out_patch[:, 0:1, :] = -first_rdnw * (
        fct_max_zero(first_next, tau=tau) - (-fct_max_zero(-first_prev, tau=tau))
    )

    if n_vert_levels > 2:
        interior_prev = wdtn_mid_patch[:, 0 : n_vert_levels - 2, :]
        interior_next = wdtn_mid_patch[:, 1 : n_vert_levels - 1, :]
        interior_rdnw = rdnw[1 : n_vert_levels - 1].view(1, n_vert_levels - 2, 1)
        scale_in_patch[:, 1 : n_vert_levels - 1, :] = -interior_rdnw * (
            (-fct_max_zero(-interior_next, tau=tau))
            - fct_max_zero(interior_prev, tau=tau)
        )
        scale_out_patch[:, 1 : n_vert_levels - 1, :] = -interior_rdnw * (
            fct_max_zero(interior_next, tau=tau)
            - (-fct_max_zero(-interior_prev, tau=tau))
        )

    if n_vert_levels > 1:
        last_prev = wdtn_mid_patch[:, n_vert_levels - 2 : n_vert_levels - 1, :]
        last_next = wdtn_patch[:, n_vert_levels : n_vert_levels + 1, :]
        last_rdnw = rdnw[n_vert_levels - 1 : n_vert_levels].view(1, 1, 1)
        scale_in_patch[:, n_vert_levels - 1 : n_vert_levels, :] = -last_rdnw * (
            (-fct_max_zero(-last_next, tau=tau)) - fct_max_zero(last_prev, tau=tau)
        )
        scale_out_patch[:, n_vert_levels - 1 : n_vert_levels, :] = -last_rdnw * (
            fct_max_zero(last_next, tau=tau) - (-fct_max_zero(-last_prev, tau=tau))
        )
    return scale_in_patch, scale_out_patch


def _monotonic_scalar_transport_limited_vertical_delta_from_mid(
    *,
    wdtn_patch: torch.Tensor,
    wdtn_limited_mid: torch.Tensor,
    n_vert_levels: int,
) -> torch.Tensor:
    """Match padded limited-wdtn row differences without the padded tensor."""
    shape = (wdtn_patch.shape[0], n_vert_levels, wdtn_patch.shape[2])
    if n_vert_levels == 0:
        return wdtn_patch.new_empty(shape)
    out = wdtn_patch.new_empty(shape)
    if n_vert_levels == 1:
        out[:, 0:1, :] = wdtn_patch[:, 1:2, :] - wdtn_patch[:, 0:1, :]
        return out
    out[:, 0:1, :] = wdtn_limited_mid[:, 0:1, :] - wdtn_patch[:, 0:1, :]
    if n_vert_levels > 2:
        out[:, 1 : n_vert_levels - 1, :] = (
            wdtn_limited_mid[:, 1 : n_vert_levels - 1, :]
            - wdtn_limited_mid[:, 0 : n_vert_levels - 2, :]
        )
    out[:, n_vert_levels - 1 : n_vert_levels, :] = (
        wdtn_patch[:, n_vert_levels : n_vert_levels + 1, :]
        - wdtn_limited_mid[:, n_vert_levels - 2 : n_vert_levels - 1, :]
    )
    return out


def _monotonic_scalar_transport_last_scalar_limited_rows(
    *,
    wdtn_patch: torch.Tensor,
    wdtn_limited_mid: torch.Tensor,
    last_scalar: int,
    n_vert_levels: int,
) -> torch.Tensor:
    """Build the final-scalar wdtn scratch rows without chunk-wide row scratch."""
    wdtn_last = wdtn_patch[:, :, last_scalar].clone()
    if n_vert_levels > 1:
        wdtn_last[:, 1:n_vert_levels] = wdtn_limited_mid[:, :, last_scalar]
    return wdtn_last


def _sum_cell_edge_slots(
    *,
    field: torch.Tensor,
    cell_edge_relation: StaticSlotRelation,
    prefer_low_memory: bool = False,
    reduction_slot_tile_width: int | None = None,
) -> torch.Tensor:
    """Accumulate cell-edge flux slots without materializing a 4D gather."""
    cell_edges_0based = cell_edge_relation.indices
    edge_sign = cell_edge_relation.sign
    edge_valid = cell_edge_relation.valid
    edge_valid_all_by_slot = cell_edge_relation.valid_all_by_slot
    signed_inv_area_weight = cell_edge_relation.weights
    if edge_sign is None or edge_valid is None or signed_inv_area_weight is None:
        raise ValueError(
            "monotonic-scalar-transport cell-edge reduction requires relation sign, valid mask, and weights"
        )
    return _sum_cell_edge_slots_from_static(
        field=field,
        cell_edges_0based=cell_edges_0based,
        edge_sign=edge_sign,
        edge_valid=edge_valid,
        edge_valid_all_by_slot=edge_valid_all_by_slot,
        signed_inv_area_weight=signed_inv_area_weight,
        prefer_low_memory=prefer_low_memory,
        reduction_slot_tile_width=reduction_slot_tile_width,
    )


def _monotonic_scalar_transport_apply_cell_edge_upwind_updates_from_static(
    *,
    scalar_new_rows: torch.Tensor,
    scale_in_patch: torch.Tensor,
    scale_out_patch: torch.Tensor,
    flux_upwind_edge_patch: torch.Tensor,
    flux_tmp_edge_patch: torch.Tensor,
    cell_edges_0based: torch.Tensor,
    edge_sign: torch.Tensor,
    signed_inv_area_weight: torch.Tensor,
    edge_valid_weight: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply each ordered cell-edge update to its running accumulator."""
    n_cells = int(cell_edges_0based.shape[0])
    for slot in range(cell_edges_0based.shape[1]):
        edge_index = cell_edges_0based[:, slot]
        scalar_contribution = gather_rows(
            flux_upwind_edge_patch, edge_index
        ) * signed_inv_area_weight[:, slot].view(n_cells, 1, 1)
        signed_flux = edge_sign[:, slot].view(n_cells, 1, 1) * gather_rows(
            flux_tmp_edge_patch, edge_index
        )
        area_weight = edge_valid_weight[:, slot].view(n_cells, 1, 1)
        outgoing_contribution = fct_max_zero(signed_flux, tau=tau) * area_weight
        incoming_contribution = (-fct_max_zero(-signed_flux, tau=tau)) * area_weight
        if not torch.is_grad_enabled() and not _compile_active():
            scalar_new_rows.sub_(scalar_contribution)
            scale_out_patch.sub_(outgoing_contribution)
            scale_in_patch.sub_(incoming_contribution)
        else:
            scalar_new_rows = scalar_new_rows - scalar_contribution
            scale_out_patch = scale_out_patch - outgoing_contribution
            scale_in_patch = scale_in_patch - incoming_contribution
    return scalar_new_rows, scale_in_patch, scale_out_patch


def _monotonic_scalar_transport_apply_cell_edge_flux_updates_from_static(
    *,
    scalar_new_rows: torch.Tensor,
    flux_edge_patch: torch.Tensor,
    cell_edges_0based: torch.Tensor,
    signed_inv_area_weight: torch.Tensor,
) -> torch.Tensor:
    """Subtract each limited edge flux from the running cell state."""
    n_cells = int(cell_edges_0based.shape[0])
    for slot in range(cell_edges_0based.shape[1]):
        contribution = gather_rows(
            flux_edge_patch, cell_edges_0based[:, slot]
        ) * signed_inv_area_weight[:, slot].view(n_cells, 1, 1)
        if not torch.is_grad_enabled() and not _compile_active():
            scalar_new_rows.sub_(contribution)
        else:
            scalar_new_rows = scalar_new_rows - contribution
    return scalar_new_rows


def _sum_cell_edge_slots_from_static(
    *,
    field: torch.Tensor,
    cell_edges_0based: torch.Tensor,
    edge_sign: torch.Tensor,
    edge_valid: torch.Tensor,
    edge_valid_all_by_slot: tuple[bool, ...] | None,
    signed_inv_area_weight: torch.Tensor,
    prefer_low_memory: bool = False,
    reduction_slot_tile_width: int | None = None,
) -> torch.Tensor:
    """Ordered cell-edge reduction over prebound topology tensors."""
    if _monotonic_scalar_transport_can_use_full_view_reduce(
        field,
        cell_edges_0based,
        edge_sign,
        edge_valid,
        None,
        signed_inv_area_weight,
        prefer_low_memory=prefer_low_memory,
        reduction_slot_tile_width=reduction_slot_tile_width,
        slot_count=cell_edges_0based.shape[1],
    ):
        return _sum_cell_edge_slots_view_reduce(
            field=field,
            cell_edges_0based=cell_edges_0based,
            edge_sign=edge_sign,
            edge_valid=edge_valid,
            inv_area_cell=None,
            edge_valid_all_by_slot=edge_valid_all_by_slot,
            signed_inv_area_weight=signed_inv_area_weight,
        )
    n_cells = cell_edges_0based.shape[0]
    total: torch.Tensor | None = None
    gather_batch = _monotonic_scalar_transport_reduction_gather_batch(
        field,
        cell_edges_0based.shape[1],
        prefer_low_memory=prefer_low_memory,
        reduction_slot_tile_width=reduction_slot_tile_width,
        tensors=(cell_edges_0based, edge_sign, edge_valid, signed_inv_area_weight),
    )
    for batch_start, batch_stop in _slot_batches(
        cell_edges_0based.shape[1],
        gather_batch,
    ):
        field_batch = _gather_slot_batch(
            field, cell_edges_0based, batch_start, batch_stop
        )
        weight_batch = signed_inv_area_weight[:, batch_start:batch_stop]
        for local_slot in range(batch_stop - batch_start):
            contrib = field_batch[:, local_slot, :, :] * weight_batch[
                :, local_slot
            ].view(n_cells, 1, 1)
            total = _accumulate_ordered_slot(total, contrib)
    if total is None:
        return field.new_zeros((n_cells, *field.shape[1:]))
    return total


def _sum_cell_edge_limiter_slots(
    *,
    field: torch.Tensor,
    cell_edge_relation: StaticSlotRelation,
    tau: float = 0.01,
    prefer_low_memory: bool = False,
    reduction_slot_tile_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate positive and negative limiter flux slots in one ordered pass."""
    cell_edges_0based = cell_edge_relation.indices
    edge_sign = cell_edge_relation.sign
    edge_valid = cell_edge_relation.valid
    edge_valid_all_by_slot = cell_edge_relation.valid_all_by_slot
    edge_valid_weight = cell_edge_relation.valid_weight
    if edge_sign is None or edge_valid is None or edge_valid_weight is None:
        raise ValueError(
            "monotonic-scalar-transport limiter reduction requires relation sign, valid mask, and weights"
        )
    return _sum_cell_edge_limiter_slots_from_static(
        field=field,
        cell_edges_0based=cell_edges_0based,
        edge_sign=edge_sign,
        edge_valid=edge_valid,
        edge_valid_all_by_slot=edge_valid_all_by_slot,
        edge_valid_weight=edge_valid_weight,
        tau=tau,
        prefer_low_memory=prefer_low_memory,
        reduction_slot_tile_width=reduction_slot_tile_width,
    )


def _sum_cell_edge_limiter_slots_from_static(
    *,
    field: torch.Tensor,
    cell_edges_0based: torch.Tensor,
    edge_sign: torch.Tensor,
    edge_valid: torch.Tensor,
    edge_valid_all_by_slot: tuple[bool, ...] | None,
    edge_valid_weight: torch.Tensor,
    tau: float = 0.01,
    prefer_low_memory: bool = False,
    reduction_slot_tile_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordered limiter reduction over prebound topology tensors."""
    if _monotonic_scalar_transport_can_use_full_view_reduce(
        field,
        cell_edges_0based,
        edge_sign,
        edge_valid,
        None,
        edge_valid_weight,
        prefer_low_memory=prefer_low_memory,
        reduction_slot_tile_width=reduction_slot_tile_width,
        slot_count=cell_edges_0based.shape[1],
    ):
        return _sum_cell_edge_limiter_slots_view_reduce(
            field=field,
            cell_edges_0based=cell_edges_0based,
            edge_sign=edge_sign,
            edge_valid=edge_valid,
            inv_area_cell=None,
            edge_valid_all_by_slot=edge_valid_all_by_slot,
            edge_valid_weight=edge_valid_weight,
            tau=tau,
        )
    n_cells = cell_edges_0based.shape[0]
    positive: torch.Tensor | None = None
    negative: torch.Tensor | None = None
    gather_batch = _monotonic_scalar_transport_reduction_gather_batch(
        field,
        cell_edges_0based.shape[1],
        prefer_low_memory=prefer_low_memory,
        reduction_slot_tile_width=reduction_slot_tile_width,
        tensors=(cell_edges_0based, edge_sign, edge_valid, edge_valid_weight),
    )
    for batch_start, batch_stop in _slot_batches(
        cell_edges_0based.shape[1],
        gather_batch,
    ):
        field_batch = _gather_slot_batch(
            field, cell_edges_0based, batch_start, batch_stop
        )
        for local_slot in range(batch_stop - batch_start):
            slot = batch_start + local_slot
            signed = (
                edge_sign[:, slot].view(n_cells, 1, 1)
                * field_batch[:, local_slot, :, :]
            )
            weight = edge_valid_weight[:, slot].view(n_cells, 1, 1)
            positive_contrib = fct_max_zero(signed, tau=tau) * weight
            negative_contrib = (-fct_max_zero(-signed, tau=tau)) * weight
            positive = _accumulate_ordered_slot(positive, positive_contrib)
            negative = _accumulate_ordered_slot(negative, negative_contrib)
    if positive is None or negative is None:
        shape = (n_cells, *field.shape[1:])
        return field.new_zeros(shape), field.new_zeros(shape)
    return positive, negative


def _sum_density_flux_slots(
    *,
    edge_field: torch.Tensor,
    cell_edge_relation: StaticSlotRelation,
) -> torch.Tensor:
    """Accumulate density horizontal flux divergence without a full cell-edge tensor."""
    cell_edges_0based = cell_edge_relation.indices
    edge_sign = cell_edge_relation.sign
    edge_valid = cell_edge_relation.valid
    edge_valid_all_by_slot = cell_edge_relation.valid_all_by_slot
    signed_edge_metric = cell_edge_relation.weights
    edge_valid_inv_area_weight = cell_edge_relation.valid_weight
    if edge_sign is None or edge_valid is None or signed_edge_metric is None:
        raise ValueError(
            "monotonic-scalar-transport density reduction requires relation sign, valid mask, and metric weights"
        )
    if edge_valid_inv_area_weight is None:
        raise ValueError(
            "monotonic-scalar-transport density reduction requires relation inverse-area weights"
        )
    if _monotonic_scalar_transport_can_use_full_view_reduce(
        edge_field,
        None,
        cell_edges_0based,
        edge_sign,
        edge_valid,
        None,
        signed_edge_metric,
        edge_valid_inv_area_weight,
    ):
        return _sum_density_flux_slots_view_reduce(
            edge_field=edge_field,
            edge_metric=None,
            cell_edges_0based=cell_edges_0based,
            edge_sign=edge_sign,
            edge_valid=edge_valid,
            inv_area_cell=None,
            edge_valid_all_by_slot=edge_valid_all_by_slot,
            signed_edge_metric=signed_edge_metric,
            edge_valid_inv_area_weight=edge_valid_inv_area_weight,
        )
    n_cells = cell_edges_0based.shape[0]
    total: torch.Tensor | None = None
    for batch_start, batch_stop in _slot_batches(
        cell_edges_0based.shape[1],
        _slot_gather_batch(edge_field, cell_edges_0based.shape[1]),
    ):
        edge_batch = _gather_slot_batch(
            edge_field, cell_edges_0based, batch_start, batch_stop
        )
        metric_batch = signed_edge_metric[:, batch_start:batch_stop]
        weight_batch = edge_valid_inv_area_weight[:, batch_start:batch_stop]
        for local_slot in range(batch_stop - batch_start):
            base = edge_batch[:, local_slot, :] * metric_batch[:, local_slot].view(
                n_cells, 1
            )
            contrib = base * weight_batch[:, local_slot].view(n_cells, 1)
            total = _accumulate_ordered_slot(total, contrib)
    if total is None:
        return edge_field.new_zeros((n_cells, edge_field.shape[1]))
    return total


_MONOTONIC_SCALAR_TRANSPORT_FLUX_DIVISORS: dict[
    tuple[torch.dtype, torch.device], torch.Tensor
] = {}


def _monotonic_scalar_transport_flux_divisor(reference: torch.Tensor) -> torch.Tensor:
    """Exact 12 as a tensor input so Inductor keeps the division.

    Created once per dtype/device: building a device tensor from a host scalar
    is a blocking host-to-device copy, which must not happen on every call.
    """
    key = (reference.dtype, reference.device)
    divisor = _MONOTONIC_SCALAR_TRANSPORT_FLUX_DIVISORS.get(key)
    if divisor is None:
        divisor = _make_flux_divisor(reference)
        _MONOTONIC_SCALAR_TRANSPORT_FLUX_DIVISORS[key] = divisor
    # Shared across calls: an in-place write or grad flag would silently change
    # every later flux, so fail loudly (no device read, no sync).
    _validate_flux_divisor(divisor)
    return divisor


def _monotonic_scalar_transport_vertical_neighbor_bounds_numeric(
    *,
    scalar_old_all: torch.Tensor,
    scalar_new_all: torch.Tensor,
    ww: torch.Tensor,
    fnm: torch.Tensor,
    fnp: torch.Tensor,
    neighbor_relation: StaticSlotRelation,
    flux_divisor: torch.Tensor,
    cell_start: int,
    cell_stop: int,
    n_vert_levels: int,
    coef_3rd_order: float,
    prefer_low_memory: bool,
    reduction_slot_tile_width: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local vertical bounds, neighbor bounds and the vertical scalar flux.

    Same expressions and order as the former inline block: levels are joined
    with ``torch.cat`` instead of slice stores, which moves no arithmetic.
    """
    so = scalar_old_all[cell_start:cell_stop, :, :]
    sn = scalar_new_all[cell_start:cell_stop, :, :]
    n = n_vert_levels
    k_last = n - 1

    s_max_parts = [
        fct_max(so[:, 0, :], so[:, 1, :]).unsqueeze(1),
        fct_max(fct_max(so[:, 0, :], so[:, 1, :]), so[:, 2, :]).unsqueeze(1),
    ]
    s_min_parts = [
        fct_min(so[:, 0, :], so[:, 1, :]).unsqueeze(1),
        fct_min(fct_min(so[:, 0, :], so[:, 1, :]), so[:, 2, :]).unsqueeze(1),
    ]
    zero_level = torch.zeros_like(sn[:, 0:1, :])
    wdtn_parts = [
        zero_level,
        (
            ww[:, 1].unsqueeze(1) * (fnm[1] * sn[:, 1, :] + fnp[1] * sn[:, 0, :])
        ).unsqueeze(1),
    ]
    if n > 3:
        wdtn_parts.append(
            _flux3(
                sn[:, 0 : n - 3, :],
                sn[:, 1 : n - 2, :],
                sn[:, 2 : n - 1, :],
                sn[:, 3:n, :],
                ww[:, 2 : n - 1].unsqueeze(2),
                coef_3rd_order,
                flux_divisor,
            )
        )
        s_max_parts.append(
            fct_max(fct_max(so[:, 1 : n - 2, :], so[:, 2 : n - 1, :]), so[:, 3:n, :])
        )
        s_min_parts.append(
            fct_min(fct_min(so[:, 1 : n - 2, :], so[:, 2 : n - 1, :]), so[:, 3:n, :])
        )
    wdtn_parts.append(
        (
            ww[:, k_last].unsqueeze(1)
            * (fnm[k_last] * sn[:, k_last, :] + fnp[k_last] * sn[:, k_last - 1, :])
        ).unsqueeze(1)
    )
    wdtn_parts.append(zero_level)
    s_max_parts.append(fct_max(so[:, k_last, :], so[:, k_last - 1, :]).unsqueeze(1))
    s_min_parts.append(fct_min(so[:, k_last, :], so[:, k_last - 1, :]).unsqueeze(1))

    neighbor_max, neighbor_min = _neighbor_scalar_bounds_slots(
        scalar_field=scalar_old_all,
        neighbor_relation=neighbor_relation,
        prefer_low_memory=prefer_low_memory,
        reduction_slot_tile_width=reduction_slot_tile_width,
    )
    s_max = fct_max(torch.cat(s_max_parts, dim=1), neighbor_max)
    s_min = fct_min(torch.cat(s_min_parts, dim=1), neighbor_min)
    return s_max, s_min, torch.cat(wdtn_parts, dim=1)


def _neighbor_scalar_bounds_slots(
    *,
    scalar_field: torch.Tensor,
    neighbor_relation: StaticSlotRelation,
    prefer_low_memory: bool = False,
    reduction_slot_tile_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce neighbor scalar bounds without materializing a cell-slot field."""
    neighbors_0based = neighbor_relation.indices
    neighbor_valid = neighbor_relation.valid
    if neighbor_valid is None:
        raise ValueError(
            "monotonic-scalar-transport neighbor bounds require a relation valid mask"
        )
    slot_tile_width = _monotonic_scalar_transport_effective_reduction_slot_tile_width(
        neighbors_0based.shape[1],
        prefer_low_memory=prefer_low_memory,
        reduction_slot_tile_width=reduction_slot_tile_width,
    )
    if (
        slot_tile_width == 0
        and not _monotonic_scalar_transport_can_use_full_view_reduce(
            scalar_field,
            neighbors_0based,
            neighbor_valid,
            slot_count=neighbors_0based.shape[1],
        )
    ):
        slot_tile_width = 1
    return fct_neighbor_bounds(
        scalar_field,
        neighbors_0based,
        neighbor_valid,
        slot_tile_width=slot_tile_width,
    )


def _gather_cell_pair_patch_or_zero(
    patch: torch.Tensor,
    row_idx: torch.Tensor,
    *,
    row_start: int,
    row_end: int,
) -> torch.Tensor:
    """Gather cell rows from a contiguous patch; zero rows outside it."""
    local_valid = (row_idx >= row_start) & (row_idx < row_end)
    local_idx = (row_idx - row_start).clamp(min=0, max=max(row_end - row_start - 1, 0))
    gathered = gather_rows(patch, local_idx)
    if (
        row_idx.device.type == "cpu"
        and not _compile_active()
        and bool(local_valid.all())
    ):
        return gathered
    return torch.where(
        local_valid.view(*row_idx.shape, *([1] * (patch.ndim - 1))),
        gathered,
        torch.zeros_like(gathered),
    )


def _monotonic_scalar_transport_safe_zero_invalid(
    indices0: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    out = torch.where(valid, indices0, torch.zeros_like(indices0))
    return out if out.dtype == torch.int64 else out.to(dtype=torch.int64)


def _monotonic_scalar_transport_static_all_true(
    mask: torch.Tensor, *, allow_cuda_sync: bool = False
) -> bool:
    return bool(_static_mask_all_true(mask, allow_cuda_sync=allow_cuda_sync))


def _monotonic_scalar_transport_slot_all_true(
    mask: torch.Tensor,
    *,
    allow_cuda_sync: bool = False,
) -> tuple[bool, ...]:
    metadata = _static_slot_all_true(mask, allow_cuda_sync=allow_cuda_sync)
    if metadata is None:
        return tuple(False for _ in range(mask.shape[1]))
    return metadata


def _monotonic_scalar_transport_masked_value_or_zero(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    mask_all: bool | None = None,
) -> torch.Tensor:
    if mask_all is True or (
        mask_all is None and _monotonic_scalar_transport_static_all_true(mask)
    ):
        return value
    return torch.where(mask, value, torch.zeros_like(value))


def _monotonic_scalar_transport_masked_select(
    mask: torch.Tensor,
    true_value: torch.Tensor,
    false_value: torch.Tensor,
    *,
    mask_all: bool | None = None,
) -> torch.Tensor:
    if mask_all is True or (
        mask_all is None and _monotonic_scalar_transport_static_all_true(mask)
    ):
        return true_value
    return torch.where(mask, true_value, false_value)


def _monotonic_scalar_transport_can_use_full_view_reduce(
    reference: torch.Tensor,
    *tensors: torch.Tensor | None,
    prefer_low_memory: bool = False,
    reduction_slot_tile_width: int | None = None,
    slot_count: int | None = None,
) -> bool:
    """Return True when the ordered additive reducer may gather all slots."""
    if slot_count is not None:
        tile_width = _monotonic_scalar_transport_effective_reduction_slot_tile_width(
            int(slot_count),
            prefer_low_memory=prefer_low_memory,
            reduction_slot_tile_width=reduction_slot_tile_width,
        )
        if tile_width not in (0, int(slot_count)):
            return False
    elif prefer_low_memory:
        return False
    return reference.device.type == "cuda" or _compile_active()


def _monotonic_scalar_transport_effective_reduction_slot_tile_width(
    slot_count: int,
    *,
    prefer_low_memory: bool = False,
    reduction_slot_tile_width: int | None = None,
) -> int:
    """Return 0 for default/full reducer selection, or a bounded slot tile width."""
    slots = max(int(slot_count), 1)
    if reduction_slot_tile_width is None:
        return 1 if prefer_low_memory else 0
    requested = int(reduction_slot_tile_width)
    if requested <= 0:
        return 1 if prefer_low_memory else 0
    return max(1, min(requested, slots))


def _monotonic_scalar_transport_reduction_gather_batch(
    reference: torch.Tensor,
    slot_count: int,
    *,
    prefer_low_memory: bool = False,
    reduction_slot_tile_width: int | None = None,
    tensors: tuple[torch.Tensor | None, ...] = (),
) -> int:
    """Choose slot gather width for monotonic-scalar-transport's memory/throughput policy."""
    tile_width = _monotonic_scalar_transport_effective_reduction_slot_tile_width(
        slot_count,
        prefer_low_memory=prefer_low_memory,
        reduction_slot_tile_width=reduction_slot_tile_width,
    )
    if tile_width > 0:
        return tile_width
    return _slot_gather_batch(reference, slot_count)


def _sum_cell_edge_slots_view_reduce(
    *,
    field: torch.Tensor,
    cell_edges_0based: torch.Tensor,
    edge_sign: torch.Tensor,
    edge_valid: torch.Tensor,
    inv_area_cell: torch.Tensor | None,
    edge_valid_all_by_slot: tuple[bool, ...] | None = None,
    signed_inv_area_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Row-owned monotonic-scalar-transport cell-edge reduction without slot-loop launches."""
    n_cells = cell_edges_0based.shape[0]
    n_slots = cell_edges_0based.shape[1]
    if n_slots == 0:
        return field.new_zeros((n_cells, *field.shape[1:]))
    gathered = gather_rows(field, cell_edges_0based)
    tail = (1,) * (field.ndim - 1)
    if signed_inv_area_weight is not None:
        weight = signed_inv_area_weight
    else:
        if inv_area_cell is None:
            raise ValueError(
                "monotonic-scalar-transport reduction requires inverse area or precomputed weights"
            )
        weight = edge_sign * inv_area_cell.unsqueeze(1)
        valid_all = (
            edge_valid_all_by_slot is not None
            and len(edge_valid_all_by_slot) == n_slots
            and all(edge_valid_all_by_slot)
        )
        if not valid_all:
            weight = weight * edge_valid.to(dtype=field.dtype)
    contrib = gathered * weight.view(n_cells, n_slots, *tail)
    return sum_fortran_order(contrib, dim=1)


def _sum_cell_edge_limiter_slots_view_reduce(
    *,
    field: torch.Tensor,
    cell_edges_0based: torch.Tensor,
    edge_sign: torch.Tensor,
    edge_valid: torch.Tensor,
    inv_area_cell: torch.Tensor | None,
    edge_valid_all_by_slot: tuple[bool, ...] | None = None,
    edge_valid_weight: torch.Tensor | None = None,
    tau: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-owned monotonic-scalar-transport limiter flux reduction without slot-loop launches."""
    n_cells = cell_edges_0based.shape[0]
    n_slots = cell_edges_0based.shape[1]
    if n_slots == 0:
        shape = (n_cells, *field.shape[1:])
        return field.new_zeros(shape), field.new_zeros(shape)
    gathered = gather_rows(field, cell_edges_0based)
    tail = (1,) * (field.ndim - 1)
    signed = edge_sign.view(n_cells, n_slots, *tail) * gathered
    if edge_valid_weight is not None:
        weight = edge_valid_weight
    else:
        if inv_area_cell is None:
            raise ValueError(
                "monotonic-scalar-transport limiter reduction requires inverse area or precomputed weights"
            )
        weight = inv_area_cell.unsqueeze(1)
        valid_all = (
            edge_valid_all_by_slot is not None
            and len(edge_valid_all_by_slot) == n_slots
            and all(edge_valid_all_by_slot)
        )
        if not valid_all:
            weight = weight * edge_valid.to(dtype=field.dtype)
    weight_view = weight.view(n_cells, n_slots, *tail)
    positive = sum_fortran_order(fct_max_zero(signed, tau=tau) * weight_view, dim=1)
    negative = sum_fortran_order((-fct_max_zero(-signed, tau=tau)) * weight_view, dim=1)
    return positive, negative


def _sum_density_flux_slots_view_reduce(
    *,
    edge_field: torch.Tensor,
    edge_metric: torch.Tensor | None,
    cell_edges_0based: torch.Tensor,
    edge_sign: torch.Tensor,
    edge_valid: torch.Tensor,
    inv_area_cell: torch.Tensor | None,
    edge_valid_all_by_slot: tuple[bool, ...] | None = None,
    signed_edge_metric: torch.Tensor | None = None,
    edge_valid_inv_area_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Row-owned monotonic-scalar-transport density divergence reduction without slot-loop launches."""
    n_cells = cell_edges_0based.shape[0]
    n_slots = cell_edges_0based.shape[1]
    if n_slots == 0:
        return edge_field.new_zeros((n_cells, edge_field.shape[1]))
    edge_batch = gather_rows(edge_field, cell_edges_0based)
    if signed_edge_metric is None:
        if edge_metric is None:
            raise ValueError(
                "monotonic-scalar-transport density reduction requires edge metric or precomputed weights"
            )
        metric = -edge_sign * gather_rows(edge_metric, cell_edges_0based)
    else:
        metric = signed_edge_metric
    base = edge_batch * metric.unsqueeze(2)
    if edge_valid_inv_area_weight is not None:
        weight = edge_valid_inv_area_weight
    else:
        if inv_area_cell is None:
            raise ValueError(
                "monotonic-scalar-transport density reduction requires inverse area or precomputed weights"
            )
        weight = inv_area_cell.unsqueeze(1)
        valid_all = (
            edge_valid_all_by_slot is not None
            and len(edge_valid_all_by_slot) == n_slots
            and all(edge_valid_all_by_slot)
        )
        if not valid_all:
            weight = weight * edge_valid.to(dtype=edge_field.dtype)
    return sum_fortran_order(base * weight.unsqueeze(2), dim=1)


def _gather_limiter_scale_pairs(
    *,
    scale_in_limited: torch.Tensor,
    scale_out_limited: torch.Tensor,
    edge_cell1_safe: torch.Tensor,
    edge_cell2_safe: torch.Tensor,
    row_start: int,
    row_end: int,
    edge_cell1_local_idx: torch.Tensor | None = None,
    edge_cell2_local_idx: torch.Tensor | None = None,
    edge_cell1_local_valid: torch.Tensor | None = None,
    edge_cell2_local_valid: torch.Tensor | None = None,
    edge_cell1_local_all: bool | None = None,
    edge_cell2_local_all: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather monotonic-scalar-transport limiter scale pairs while deriving local row masks once."""
    row_tail = [1] * (scale_in_limited.ndim - 1)
    if (
        edge_cell1_local_idx is None
        or edge_cell2_local_idx is None
        or edge_cell1_local_valid is None
        or edge_cell2_local_valid is None
    ):
        row_span = max(row_end - row_start - 1, 0)
        c1_valid = (edge_cell1_safe >= row_start) & (edge_cell1_safe < row_end)
        c2_valid = (edge_cell2_safe >= row_start) & (edge_cell2_safe < row_end)
        c1_idx = (edge_cell1_safe - row_start).clamp(min=0, max=row_span)
        c2_idx = (edge_cell2_safe - row_start).clamp(min=0, max=row_span)
    else:
        c1_valid = edge_cell1_local_valid
        c2_valid = edge_cell2_local_valid
        c1_idx = edge_cell1_local_idx
        c2_idx = edge_cell2_local_idx

    c1_in = gather_rows(scale_in_limited, c1_idx)
    c1_out = gather_rows(scale_out_limited, c1_idx)
    c2_in = gather_rows(scale_in_limited, c2_idx)
    c2_out = gather_rows(scale_out_limited, c2_idx)

    c1_mask = c1_valid.view(*edge_cell1_safe.shape, *row_tail)
    c2_mask = c2_valid.view(*edge_cell2_safe.shape, *row_tail)
    c1_all = (
        edge_cell1_local_all
        if edge_cell1_local_all is not None
        else _monotonic_scalar_transport_static_all_true(c1_valid)
    )
    c2_all = (
        edge_cell2_local_all
        if edge_cell2_local_all is not None
        else _monotonic_scalar_transport_static_all_true(c2_valid)
    )
    return (
        _monotonic_scalar_transport_masked_value_or_zero(
            c1_in, c1_mask, mask_all=c1_all
        ),
        _monotonic_scalar_transport_masked_value_or_zero(
            c2_in, c2_mask, mask_all=c2_all
        ),
        _monotonic_scalar_transport_masked_value_or_zero(
            c1_out, c1_mask, mask_all=c1_all
        ),
        _monotonic_scalar_transport_masked_value_or_zero(
            c2_out, c2_mask, mask_all=c2_all
        ),
    )


def _monotonic_scalar_transport_pattern_c_flux_region(
    *,
    scalar_field: torch.Tensor,
    uh_avg: torch.Tensor,
    one: torch.Tensor,
    pattern_c_operator: PatternCOperator,
    edge_cell_relation: StaticPairRelation,
    edge_active_all: bool | None = None,
) -> torch.Tensor:
    """Compute monotonic-scalar-transport Pattern-C edge flux for one scalar chunk."""
    payload = pattern_c_operator.payload
    edge_slice = payload["edge_slice"]
    flux_candidate = _monotonic_scalar_transport_horizontal_source_flux(
        scalar_field=scalar_field,
        uh_avg=uh_avg,
        one=one,
        edge_start=int(edge_slice.start),
        edge_end=int(edge_slice.stop),
        slot_count=int(payload["slot_count"]),
        adv_coef=payload["adv_coef"],
        adv_coef_3rd=payload["adv_coef_3rd"],
        slot_cells_matrix=payload["slot_cells_matrix"],
        slot_valid_matrix=payload["slot_valid_matrix"],
    )
    edge_active_mask = edge_cell_relation.active_3d
    if edge_active_mask is None:
        raise ValueError(
            "monotonic-scalar-transport Pattern-C flux region requires an active edge mask"
        )
    if edge_active_all is None:
        edge_active_all = edge_cell_relation.active_all
    return _monotonic_scalar_transport_masked_value_or_zero(
        flux_candidate,
        edge_active_mask,
        mask_all=edge_active_all,
    )


def _monotonic_scalar_transport_horizontal_source_flux(
    *,
    scalar_field: torch.Tensor,
    uh_avg: torch.Tensor,
    one: torch.Tensor,
    edge_start: int,
    edge_end: int,
    slot_count: int,
    adv_coef: torch.Tensor,
    adv_coef_3rd: torch.Tensor,
    slot_cells_matrix: torch.Tensor,
    slot_valid_matrix: torch.Tensor,
) -> torch.Tensor:
    """Preserve SELECT CASE(10) association with fixed-shape tensor masks."""
    velocity = uh_avg[edge_start:edge_end, :]
    ten_slots = (slot_valid_matrix.sum(dim=1) == 10).view(-1, 1)
    # copysign(one, velocity) as a select (same bits, signed zeros included; Metal codegen has no copysign).
    velocity_sign = torch.where(torch.signbit(velocity), -one, one)
    total = torch.zeros(
        velocity.shape[0],
        scalar_field.shape[1],
        scalar_field.shape[2],
        dtype=scalar_field.dtype,
        device=scalar_field.device,
    )
    for slot in range(slot_count):
        a = adv_coef[:, slot].view(-1, 1)
        b = adv_coef_3rd[:, slot].view(-1, 1)
        scalar = gather_rows(scalar_field, slot_cells_matrix[:, slot])
        # Select completed contributions, not a broadcast weight field. This keeps the
        # ordered products together instead of storing one [edge, level] weight per slot.
        contribution = torch.where(
            ten_slots.unsqueeze(2),
            torch.where((velocity > 0.0).unsqueeze(2),
                        (a + b).unsqueeze(2) * scalar, (a - b).unsqueeze(2) * scalar),
            (velocity * (a + velocity_sign * b)).unsqueeze(2) * scalar,
        )
        contribution = torch.where(
            slot_valid_matrix[:, slot].view(-1, 1, 1),
            contribution,
            torch.zeros_like(contribution),
        )
        added = total + contribution
        total = torch.where(ten_slots.unsqueeze(2), contribution, added) if slot == 0 else added
    return torch.where(ten_slots.unsqueeze(2), velocity.unsqueeze(2) * total, total)


def _monotonic_scalar_transport_upwind_scale_prep_region(
    *,
    scalar_old_rows: torch.Tensor,
    scalar_old_all: torch.Tensor,
    rho_zz_old_rows: torch.Tensor,
    ww_avg_rows: torch.Tensor,
    uh_avg_edges: torch.Tensor,
    dv_edge_patch: torch.Tensor,
    flux_edge_patch: torch.Tensor,
    wdtn_patch: torch.Tensor,
    rdnw: torch.Tensor,
    edge_cell_relation: StaticPairRelation,
    cell_edge_relation: StaticSlotRelation,
    dt: float,
    tau: float,
    edge_interior_all: bool | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Prepare monotonic-scalar-transport upwind updates and limiter scale sums for one scalar chunk."""
    n_vert_levels = scalar_old_rows.shape[1]
    flux_upwind_mid = dt * _monotonic_scalar_transport_upwind_value(
        ww_avg_rows[:, 1:n_vert_levels],
        scalar_old_rows[:, 0 : n_vert_levels - 1, :],
        scalar_old_rows[:, 1:n_vert_levels, :],
        tau=tau,
    )
    scalar_new_rows = _monotonic_scalar_transport_apply_vertical_upwind_updates(
        scalar_old_rows=scalar_old_rows,
        rho_zz_old_rows=rho_zz_old_rows,
        flux_upwind_mid=flux_upwind_mid,
        rdnw=rdnw,
    )

    wdtn_mid_patch = dt * wdtn_patch[:, 1:n_vert_levels, :] - flux_upwind_mid
    wdtn_rows = _replace_all_rows_cols(wdtn_patch, 1, n_vert_levels, wdtn_mid_patch)

    scale_in_patch = -rdnw.view(1, n_vert_levels, 1) * (
        (-fct_max_zero(-wdtn_rows[:, 1 : n_vert_levels + 1, :], tau=tau))
        - fct_max_zero(wdtn_rows[:, 0:n_vert_levels, :], tau=tau)
    )
    scale_out_patch = -rdnw.view(1, n_vert_levels, 1) * (
        fct_max_zero(wdtn_rows[:, 1 : n_vert_levels + 1, :], tau=tau)
        - (-fct_max_zero(-wdtn_rows[:, 0:n_vert_levels, :], tau=tau))
    )

    edge_cell1_safe = edge_cell_relation.first
    edge_cell2_safe = edge_cell_relation.second
    edge_interior = edge_cell_relation.valid
    if edge_interior is None:
        raise ValueError(
            "monotonic-scalar-transport upwind scale prep requires edge-cell interior mask"
        )
    flux_upwind_candidate = (
        dv_edge_patch.view(-1, 1, 1)
        * dt
        * _monotonic_scalar_transport_horizontal_upwind_value(
            scalar_old_all=scalar_old_all,
            velocity=uh_avg_edges,
            edge_cell_first=edge_cell1_safe,
            edge_cell_second=edge_cell2_safe,
            tau=tau,
        )
    )
    edge_interior_mask = edge_interior.unsqueeze(1).unsqueeze(2)
    flux_upwind_edge_patch = _monotonic_scalar_transport_masked_value_or_zero(
        flux_upwind_candidate,
        edge_interior_mask,
        mask_all=edge_interior_all,
    )
    flux_tmp_edge_patch = dt * flux_edge_patch - flux_upwind_edge_patch

    if (
        cell_edge_relation.sign is None
        or cell_edge_relation.weights is None
        or cell_edge_relation.valid_weight is None
    ):
        raise ValueError(
            "monotonic-scalar-transport cell-edge update requires relation sign and precomputed weights"
        )
    scalar_cell_patch, scale_in_cell_patch, scale_out_cell_patch = (
        _monotonic_scalar_transport_apply_cell_edge_upwind_updates_from_static(
            scalar_new_rows=scalar_new_rows,
            scale_in_patch=scale_in_patch,
            scale_out_patch=scale_out_patch,
            flux_upwind_edge_patch=flux_upwind_edge_patch,
            flux_tmp_edge_patch=flux_tmp_edge_patch,
            cell_edges_0based=cell_edge_relation.indices,
            edge_sign=cell_edge_relation.sign,
            signed_inv_area_weight=cell_edge_relation.weights,
            edge_valid_weight=cell_edge_relation.valid_weight,
            tau=tau,
        )
    )
    return (
        scalar_cell_patch,
        scale_in_cell_patch,
        scale_out_cell_patch,
        flux_upwind_edge_patch,
        flux_tmp_edge_patch,
        wdtn_rows,
    )


def _monotonic_scalar_transport_limiter_flux_region(
    *,
    s_max_patch: torch.Tensor,
    s_min_patch: torch.Tensor,
    rho_zz_int_rows: torch.Tensor,
    scalar_new_rows: torch.Tensor,
    scale_in_cell_patch: torch.Tensor,
    scale_out_cell_patch: torch.Tensor,
    wdtn_rows: torch.Tensor,
    flux_edge_patch: torch.Tensor,
    flux_tmp_edge_patch: torch.Tensor,
    edge_cell_relation: StaticPairRelation,
    row_start: int,
    row_end: int,
    eps: float,
    tau: float,
    edge_interior_all: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the monotonic-scalar-transport limiter scales, limited fluxes, and limited vertical fluxes."""
    edge_cell1_safe = edge_cell_relation.first
    edge_cell2_safe = edge_cell_relation.second
    edge_cell1_local_idx = edge_cell_relation.first_local
    edge_cell2_local_idx = edge_cell_relation.second_local
    edge_cell1_local_valid = edge_cell_relation.first_local_valid
    edge_cell2_local_valid = edge_cell_relation.second_local_valid
    edge_cell1_local_all = edge_cell_relation.first_local_all
    edge_cell2_local_all = edge_cell_relation.second_local_all
    if edge_cell_relation.valid is None:
        raise ValueError(
            "monotonic-scalar-transport limiter region requires a relation valid mask"
        )
    edge_interior_mask = edge_cell_relation.valid.unsqueeze(1).unsqueeze(2)
    if edge_interior_all is None:
        edge_interior_all = edge_cell_relation.valid_all
    edge_active_mask = edge_cell_relation.active_3d
    edge_active_all = edge_cell_relation.active_all
    if edge_active_mask is None:
        raise ValueError(
            "monotonic-scalar-transport limiter region requires a relation active mask"
        )
    n_vert_levels = scalar_new_rows.shape[1]
    scale_factor_in = (s_max_patch * rho_zz_int_rows.unsqueeze(2) - scalar_new_rows) / (
        scale_in_cell_patch + eps
    )
    scale_in_limited = fct_clamp_01(scale_factor_in, tau=tau)
    scale_factor_out = (
        s_min_patch * rho_zz_int_rows.unsqueeze(2) - scalar_new_rows
    ) / (scale_out_cell_patch - eps)
    scale_out_limited = fct_clamp_01(scale_factor_out, tau=tau)

    flux_after = _monotonic_scalar_transport_masked_value_or_zero(
        flux_edge_patch,
        edge_interior_mask,
        mask_all=edge_interior_all,
    )
    flux_after = _monotonic_scalar_transport_masked_select(
        edge_active_mask,
        flux_tmp_edge_patch,
        flux_after,
        mask_all=edge_active_all,
    )
    c1_in, c2_in, c1_out, c2_out = _gather_limiter_scale_pairs(
        scale_in_limited=scale_in_limited,
        scale_out_limited=scale_out_limited,
        edge_cell1_safe=edge_cell1_safe,
        edge_cell2_safe=edge_cell2_safe,
        row_start=row_start,
        row_end=row_end,
        edge_cell1_local_idx=edge_cell1_local_idx,
        edge_cell2_local_idx=edge_cell2_local_idx,
        edge_cell1_local_valid=edge_cell1_local_valid,
        edge_cell2_local_valid=edge_cell2_local_valid,
        edge_cell1_local_all=edge_cell1_local_all,
        edge_cell2_local_all=edge_cell2_local_all,
    )
    flux_limited = _monotonic_scalar_transport_signed_limiter_value(
        flux_after,
        fct_min(c1_out, c2_in),
        fct_min(c1_in, c2_out),
        tau=tau,
    )
    flux_limited_patch = _monotonic_scalar_transport_masked_select(
        edge_active_mask,
        flux_limited,
        flux_after,
        mask_all=edge_active_all,
    )

    wdtn_mid = wdtn_rows[:, 1:n_vert_levels, :]
    wdtn_limited_patch = _monotonic_scalar_transport_signed_limiter_value(
        wdtn_mid,
        fct_min(
            scale_out_limited[:, 0 : n_vert_levels - 1, :],
            scale_in_limited[:, 1:n_vert_levels, :],
        ),
        fct_min(
            scale_out_limited[:, 1:n_vert_levels, :],
            scale_in_limited[:, 0 : n_vert_levels - 1, :],
        ),
        tau=tau,
    )
    wdtn_limited_rows = _replace_all_rows_cols(
        wdtn_rows, 1, n_vert_levels, wdtn_limited_patch
    )
    return scale_in_limited, scale_out_limited, flux_limited_patch, wdtn_limited_rows


def _monotonic_scalar_transport_final_update_region(
    *,
    scalar_new_all_patch: torch.Tensor,
    scalars_new_base_patch: torch.Tensor,
    scalar_cell_patch: torch.Tensor,
    flux_limited_patch: torch.Tensor,
    wdtn_limited_rows: torch.Tensor,
    rho_zz_int_rows: torch.Tensor,
    rdnw: torch.Tensor,
    cell_edge_relation: StaticSlotRelation,
    bdy_mask: torch.Tensor,
    solve_rel_start: int,
    solve_rel_end: int,
    tau: float,
    bdy_mask_all: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply monotonic-scalar-transport limited horizontal/vertical fluxes and positive scalar mask."""
    n_vert_levels = scalar_cell_patch.shape[1]
    if cell_edge_relation.weights is None:
        raise ValueError(
            "monotonic-scalar-transport final update requires signed inverse-area weights"
        )
    scalar_stage1 = (
        _monotonic_scalar_transport_apply_cell_edge_flux_updates_from_static(
            scalar_new_rows=scalar_cell_patch,
            flux_edge_patch=flux_limited_patch,
            cell_edges_0based=cell_edge_relation.indices,
            signed_inv_area_weight=cell_edge_relation.weights,
        )
    )
    scalar_stage2 = (
        scalar_stage1
        - rdnw.view(1, n_vert_levels, 1)
        * (
            wdtn_limited_rows[:, 1 : n_vert_levels + 1, :]
            - wdtn_limited_rows[:, 0:n_vert_levels, :]
        )
    ) / rho_zz_int_rows.unsqueeze(2)

    positive_scalar = _monotonic_scalar_transport_positive_scalar_all_cell_patch(
        scalar_new_all_patch=scalar_new_all_patch,
        scalar_stage2=scalar_stage2,
        solve_rel_start=solve_rel_start,
        solve_rel_end=solve_rel_end,
        tau=tau,
    )
    scalars_new_patch = _monotonic_scalar_transport_masked_select(
        bdy_mask,
        positive_scalar,
        scalars_new_base_patch,
        mask_all=bdy_mask_all,
    )
    return scalar_stage2, scalars_new_patch


def _monotonic_scalar_transport_effective_streaming_scalar_tile_width(
    num_scalars: int,
    requested_width: int | None,
) -> int:
    """Return a compile-stable scalar subtile width for the monotonic-scalar-transport streaming helper."""
    scalars = max(int(num_scalars), 1)
    if requested_width is None:
        return scalars
    requested = int(requested_width)
    if requested <= 0:
        return scalars
    width = max(1, min(requested, scalars))
    if scalars % width == 0:
        return width
    for candidate in range(width - 1, 0, -1):
        if scalars % candidate == 0:
            return candidate
    return 1


def _monotonic_scalar_transport_slice_scalar_dim(
    value: torch.Tensor,
    scalar_start: int,
    scalar_end: int,
) -> torch.Tensor:
    """Slice tensors whose final dimension carries scalar lanes."""
    if value.ndim >= 3 and value.shape[-1] > 1:
        return value[..., scalar_start:scalar_end]
    return value


def _monotonic_scalar_transport_upwind_value(
    velocity: torch.Tensor,
    positive_value: torch.Tensor,
    negative_value: torch.Tensor,
    *,
    tau: float,
) -> torch.Tensor:
    """Apply the hard upwind primal and its piecewise-exact derivative."""
    velocity_lanes = velocity.unsqueeze(2)
    if not is_grad_mode():
        return velocity_lanes * torch.where(
            velocity_lanes >= 0.0,
            positive_value,
            negative_value,
        )
    return (
        fct_max_zero(velocity, tau=tau).unsqueeze(2) * positive_value
        + (-fct_max_zero(-velocity, tau=tau)).unsqueeze(2) * negative_value
    )


def _monotonic_scalar_transport_horizontal_upwind_value(
    *,
    scalar_old_all: torch.Tensor,
    velocity: torch.Tensor,
    edge_cell_first: torch.Tensor,
    edge_cell_second: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Gather one upwind cell in forward mode and preserve the full AD formula."""
    if is_grad_mode():
        return _monotonic_scalar_transport_upwind_value(
            velocity,
            gather_rows(scalar_old_all, edge_cell_first),
            gather_rows(scalar_old_all, edge_cell_second),
            tau=tau,
        )

    n_levels = int(scalar_old_all.shape[1])
    level = torch.arange(n_levels, device=velocity.device, dtype=torch.long).view(1, -1)
    flat_index = torch.where(
        velocity >= 0.0,
        edge_cell_first.view(-1, 1),
        edge_cell_second.view(-1, 1),
    )
    flat_index.mul_(n_levels).add_(level)
    upstream_value = gather_rows(
        scalar_old_all.reshape(-1, scalar_old_all.shape[2]),
        flat_index.reshape(-1),
    )
    upstream_value = upstream_value.view(
        velocity.shape[0],
        velocity.shape[1],
        scalar_old_all.shape[2],
    )
    return velocity.unsqueeze(2) * upstream_value


def _monotonic_scalar_transport_signed_limiter_value(
    value: torch.Tensor,
    positive_limit: torch.Tensor,
    negative_limit: torch.Tensor,
    *,
    tau: float,
) -> torch.Tensor:
    """Select one hard limiter with its piecewise-exact derivative."""
    if not is_grad_mode():
        return value * torch.where(value >= 0.0, positive_limit, negative_limit)
    return (
        fct_max_zero(value, tau=tau) * positive_limit
        + (-fct_max_zero(-value, tau=tau)) * negative_limit
    )


def _monotonic_scalar_transport_streaming_scale_prepare_numeric(
    pattern_edge_start: int,
    pattern_edge_end: int,
    pattern_edge_count: int,
    pattern_slot_count: int,
    pattern_adv_coef: torch.Tensor,
    pattern_adv_coef_3rd: torch.Tensor,
    pattern_slot_cells: torch.Tensor,
    pattern_slot_valid: torch.Tensor,
    pattern_slot_valid_all: bool | None,
    pattern_use_full_width: bool,
    pattern_fallback_rows: torch.Tensor,
    one: torch.Tensor,
    edge_cell_first: torch.Tensor,
    edge_cell_second: torch.Tensor,
    edge_cell_valid: torch.Tensor,
    edge_active_mask: torch.Tensor,
    edge_cell_first_local: torch.Tensor,
    edge_cell_second_local: torch.Tensor,
    edge_cell_first_local_valid: torch.Tensor,
    edge_cell_second_local_valid: torch.Tensor,
    edge_cell_first_local_all: bool,
    edge_cell_second_local_all: bool,
    cell_edge_indices: torch.Tensor,
    cell_edge_sign: torch.Tensor,
    cell_edge_valid: torch.Tensor,
    cell_edge_valid_all: tuple[bool, ...] | None,
    cell_edge_weights: torch.Tensor,
    cell_edge_valid_weight: torch.Tensor,
    bdy_mask: torch.Tensor,
    solve_rel_start: int,
    solve_rel_end: int,
    row_start: int,
    row_end: int,
    dt: float,
    eps: float,
    tau: float,
    edge_interior_all: bool,
    edge_active_all: bool,
    bdy_mask_all: bool,
    prefer_low_memory_reductions: bool,
    reduction_slot_tile_width: int | None,
    return_scratch: bool,
    /,
    *,
    scalar_field: torch.Tensor,
    uh_avg: torch.Tensor,
    scalar_old_rows: torch.Tensor,
    scalar_old_all: torch.Tensor,
    rho_zz_old_rows: torch.Tensor,
    ww_avg_rows: torch.Tensor,
    uh_avg_edges: torch.Tensor,
    dv_edge_patch: torch.Tensor,
    wdtn_patch: torch.Tensor,
    s_max_patch: torch.Tensor,
    s_min_patch: torch.Tensor,
    rho_zz_int_rows: torch.Tensor,
    scalar_new_all_patch: torch.Tensor,
    scalars_new_base_patch: torch.Tensor,
    rdnw: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Functional monotonic scalar transport streaming phase tile used by forward, reverse AD, and JVP.

    The public chunk loop remains outside compiled regions.  This helper covers
    one scalar chunk and returns only the final scalar scratch needed by the
    Fortran-compatible public outputs, avoiding chunk-wide scratch returns.
    """
    n_vert_levels = scalar_old_rows.shape[1]
    last_scalar = scalar_old_rows.shape[2] - 1

    flux_candidate = _monotonic_scalar_transport_horizontal_source_flux(
        scalar_field=scalar_field,
        uh_avg=uh_avg,
        one=one,
        edge_start=pattern_edge_start,
        edge_end=pattern_edge_end,
        slot_count=pattern_slot_count,
        adv_coef=pattern_adv_coef,
        adv_coef_3rd=pattern_adv_coef_3rd,
        slot_cells_matrix=pattern_slot_cells,
        slot_valid_matrix=pattern_slot_valid,
    )
    flux_edge_patch = _monotonic_scalar_transport_masked_value_or_zero(
        flux_candidate,
        edge_active_mask,
        mask_all=edge_active_all,
    )
    del flux_candidate

    flux_upwind_mid = dt * _monotonic_scalar_transport_upwind_value(
        ww_avg_rows[:, 1:n_vert_levels],
        scalar_old_rows[:, 0 : n_vert_levels - 1, :],
        scalar_old_rows[:, 1:n_vert_levels, :],
        tau=tau,
    )
    scalar_new_rows = _monotonic_scalar_transport_apply_vertical_upwind_updates(
        scalar_old_rows=scalar_old_rows,
        rho_zz_old_rows=rho_zz_old_rows,
        flux_upwind_mid=flux_upwind_mid,
        rdnw=rdnw,
    )

    wdtn_mid_patch = dt * wdtn_patch[:, 1:n_vert_levels, :] - flux_upwind_mid
    del flux_upwind_mid
    scale_in_patch, scale_out_patch = (
        _monotonic_scalar_transport_scale_patches_from_mid_flux(
            wdtn_patch=wdtn_patch,
            wdtn_mid_patch=wdtn_mid_patch,
            rdnw=rdnw,
            n_vert_levels=n_vert_levels,
            tau=tau,
        )
    )

    edge_interior_mask = edge_cell_valid.unsqueeze(1).unsqueeze(2)

    flux_upwind_candidate = (
        dv_edge_patch.view(-1, 1, 1)
        * dt
        * _monotonic_scalar_transport_horizontal_upwind_value(
            scalar_old_all=scalar_old_all,
            velocity=uh_avg_edges,
            edge_cell_first=edge_cell_first,
            edge_cell_second=edge_cell_second,
            tau=tau,
        )
    )
    flux_upwind_edge_patch = _monotonic_scalar_transport_masked_value_or_zero(
        flux_upwind_candidate,
        edge_interior_mask,
        mask_all=edge_interior_all,
    )
    del flux_upwind_candidate
    flux_tmp_edge_patch = dt * flux_edge_patch - flux_upwind_edge_patch
    if return_scratch:
        flux_upwind_last = flux_upwind_edge_patch[:, :, last_scalar].clone()
    if return_scratch:
        flux_tmp_last = flux_tmp_edge_patch[:, :, last_scalar].clone()
    scalar_cell_patch, scale_in_cell_patch, scale_out_cell_patch = (
        _monotonic_scalar_transport_apply_cell_edge_upwind_updates_from_static(
            scalar_new_rows=scalar_new_rows,
            scale_in_patch=scale_in_patch,
            scale_out_patch=scale_out_patch,
            flux_upwind_edge_patch=flux_upwind_edge_patch,
            flux_tmp_edge_patch=flux_tmp_edge_patch,
            cell_edges_0based=cell_edge_indices,
            edge_sign=cell_edge_sign,
            signed_inv_area_weight=cell_edge_weights,
            edge_valid_weight=cell_edge_valid_weight,
            tau=tau,
        )
    )
    del flux_upwind_edge_patch
    del scale_out_patch, scale_in_patch

    if torch.is_grad_enabled():
        # Under AD (eager, or a compiled region tracing for reverse mode) the quotients divide in the
        # working dtype through fct_quotient, whose derivatives never square the denominator: near eps
        # that square underflows in float32, and the clamp's zero cotangent then made 0/0. The quotient
        # equals the no-grad path's on every device (IEEE division is correctly rounded).
        scale_in_limited = fct_clamp_01(fct_quotient(
            s_max_patch * rho_zz_int_rows.unsqueeze(2) - scalar_cell_patch,
            scale_in_cell_patch + eps,
        ), tau=tau)
        del scale_in_cell_patch
        scale_out_limited = fct_clamp_01(fct_quotient(
            s_min_patch * rho_zz_int_rows.unsqueeze(2) - scalar_cell_patch,
            scale_out_cell_patch - eps,
        ), tau=tau)
        del scale_out_cell_patch
    else:
        # Both denominator patches are dead after this point. Keep the exact
        # multiply/subtract/divide/clamp ordering, but retain the numerator as
        # the quotient and clamp it in place instead of materializing separate
        # denominator-adjustment, quotient, and clamp outputs.
        scale_factor_in = s_max_patch * rho_zz_int_rows.unsqueeze(2)
        scale_factor_in.sub_(scalar_cell_patch)
        scale_in_cell_patch.add_(eps)
        scale_in_limited = scale_factor_in.div_(scale_in_cell_patch).clamp_(min=0.0, max=1.0)
        del scale_in_cell_patch

        scale_factor_out = s_min_patch * rho_zz_int_rows.unsqueeze(2)
        scale_factor_out.sub_(scalar_cell_patch)
        scale_out_cell_patch.sub_(eps)
        scale_out_limited = scale_factor_out.div_(scale_out_cell_patch).clamp_(min=0.0, max=1.0)
        del scale_out_cell_patch

    flux_after = _monotonic_scalar_transport_masked_value_or_zero(
        flux_edge_patch,
        edge_interior_mask,
        mask_all=edge_interior_all,
    )
    flux_after = _monotonic_scalar_transport_masked_select(
        edge_active_mask,
        flux_tmp_edge_patch,
        flux_after,
        mask_all=edge_active_all,
    )
    del flux_edge_patch, flux_tmp_edge_patch
    if not return_scratch:
        flux_upwind_last = flux_tmp_last = scalar_cell_patch.new_empty((0,))
    return (
        scalar_cell_patch,
        wdtn_mid_patch,
        scale_in_limited,
        scale_out_limited,
        flux_after,
        flux_upwind_last,
        flux_tmp_last,
    )


def _monotonic_scalar_transport_streaming_limited_update_numeric(
    pattern_edge_start: int,
    pattern_edge_end: int,
    pattern_edge_count: int,
    pattern_slot_count: int,
    pattern_adv_coef: torch.Tensor,
    pattern_adv_coef_3rd: torch.Tensor,
    pattern_slot_cells: torch.Tensor,
    pattern_slot_valid: torch.Tensor,
    pattern_slot_valid_all: bool | None,
    pattern_use_full_width: bool,
    pattern_fallback_rows: torch.Tensor,
    one: torch.Tensor,
    edge_cell_first: torch.Tensor,
    edge_cell_second: torch.Tensor,
    edge_cell_valid: torch.Tensor,
    edge_active_mask: torch.Tensor,
    edge_cell_first_local: torch.Tensor,
    edge_cell_second_local: torch.Tensor,
    edge_cell_first_local_valid: torch.Tensor,
    edge_cell_second_local_valid: torch.Tensor,
    edge_cell_first_local_all: bool,
    edge_cell_second_local_all: bool,
    cell_edge_indices: torch.Tensor,
    cell_edge_sign: torch.Tensor,
    cell_edge_valid: torch.Tensor,
    cell_edge_valid_all: tuple[bool, ...] | None,
    cell_edge_weights: torch.Tensor,
    cell_edge_valid_weight: torch.Tensor,
    bdy_mask: torch.Tensor,
    solve_rel_start: int,
    solve_rel_end: int,
    row_start: int,
    row_end: int,
    dt: float,
    eps: float,
    tau: float,
    edge_interior_all: bool,
    edge_active_all: bool,
    bdy_mask_all: bool,
    prefer_low_memory_reductions: bool,
    reduction_slot_tile_width: int | None,
    return_scratch: bool,
    /,
    *,
    scalar_field: torch.Tensor,
    uh_avg: torch.Tensor,
    scalar_old_rows: torch.Tensor,
    scalar_old_all: torch.Tensor,
    rho_zz_old_rows: torch.Tensor,
    ww_avg_rows: torch.Tensor,
    uh_avg_edges: torch.Tensor,
    dv_edge_patch: torch.Tensor,
    wdtn_patch: torch.Tensor,
    s_max_patch: torch.Tensor,
    s_min_patch: torch.Tensor,
    rho_zz_int_rows: torch.Tensor,
    scalar_new_all_patch: torch.Tensor,
    scalars_new_base_patch: torch.Tensor,
    rdnw: torch.Tensor,
    prepared: tuple[torch.Tensor, ...],
    scale_row_end: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Apply limited fluxes after the Fortran scale exchange seam."""
    (
        scalar_cell_patch,
        wdtn_mid_patch,
        scale_in_limited,
        scale_out_limited,
        flux_after,
        flux_upwind_last,
        flux_tmp_last,
    ) = prepared
    n_vert_levels = scalar_old_rows.shape[1]
    last_scalar = scalar_old_rows.shape[2] - 1
    grad_mode = is_grad_mode()
    if scale_row_end is not None:
        row_end = scale_row_end
    c1_in, c2_in, c1_out, c2_out = _gather_limiter_scale_pairs(
        scale_in_limited=scale_in_limited,
        scale_out_limited=scale_out_limited,
        edge_cell1_safe=edge_cell_first,
        edge_cell2_safe=edge_cell_second,
        row_start=row_start,
        row_end=row_end,
        edge_cell1_local_idx=edge_cell_first_local if scale_row_end is None else None,
        edge_cell2_local_idx=edge_cell_second_local if scale_row_end is None else None,
        edge_cell1_local_valid=edge_cell_first_local_valid
        if scale_row_end is None
        else None,
        edge_cell2_local_valid=edge_cell_second_local_valid
        if scale_row_end is None
        else None,
        edge_cell1_local_all=edge_cell_first_local_all
        if scale_row_end is None
        else None,
        edge_cell2_local_all=edge_cell_second_local_all
        if scale_row_end is None
        else None,
    )
    # Only the gather consumes ghost scales. Vertical updates remain owned-cell work.
    scale_in_limited = scale_in_limited[: scalar_cell_patch.shape[0]]
    scale_out_limited = scale_out_limited[: scalar_cell_patch.shape[0]]
    flux_limited = _monotonic_scalar_transport_signed_limiter_value(
        flux_after,
        fct_min(c1_out, c2_in),
        fct_min(c1_in, c2_out),
        tau=tau,
    )
    flux_limited_patch = _monotonic_scalar_transport_masked_select(
        edge_active_mask,
        flux_limited,
        flux_after,
        mask_all=edge_active_all,
    )
    if return_scratch:
        flux_limited_last = flux_limited_patch[:, :, last_scalar].clone()
    del flux_after, flux_limited, c1_in, c2_in, c1_out, c2_out

    wdtn_limited_mid = _monotonic_scalar_transport_signed_limiter_value(
        wdtn_mid_patch,
        fct_min(
            scale_out_limited[:, 0 : n_vert_levels - 1, :],
            scale_in_limited[:, 1:n_vert_levels, :],
        ),
        fct_min(
            scale_out_limited[:, 1:n_vert_levels, :],
            scale_in_limited[:, 0 : n_vert_levels - 1, :],
        ),
        tau=tau,
    )
    vertical_limited_delta = (
        _monotonic_scalar_transport_limited_vertical_delta_from_mid(
            wdtn_patch=wdtn_patch,
            wdtn_limited_mid=wdtn_limited_mid,
            n_vert_levels=n_vert_levels,
        )
    )

    scalar_stage1 = (
        _monotonic_scalar_transport_apply_cell_edge_flux_updates_from_static(
            scalar_new_rows=scalar_cell_patch,
            flux_edge_patch=flux_limited_patch,
            cell_edges_0based=cell_edge_indices,
            signed_inv_area_weight=cell_edge_weights,
        )
    )
    del flux_limited_patch
    if grad_mode:
        scalar_stage2 = (
            scalar_stage1 - rdnw.view(1, n_vert_levels, 1) * vertical_limited_delta
        ) / rho_zz_int_rows.unsqueeze(2)
        del scalar_stage1, vertical_limited_delta
    else:
        # These two cell patches are private and dead at their respective
        # updates. Reuse them while retaining the established subtract,
        # multiply/subtract, then divide association exactly.
        scalar_stage2 = scalar_stage1
        vertical_limited_delta.mul_(rdnw.view(1, n_vert_levels, 1))
        scalar_stage2.sub_(vertical_limited_delta)
        scalar_stage2.div_(rho_zz_int_rows.unsqueeze(2))
        del vertical_limited_delta
    positive_scalar = _monotonic_scalar_transport_positive_scalar_all_cell_patch(
        scalar_new_all_patch=scalar_new_all_patch,
        scalar_stage2=scalar_stage2,
        solve_rel_start=solve_rel_start,
        solve_rel_end=solve_rel_end,
        tau=tau,
    )
    scalars_new_patch = _monotonic_scalar_transport_masked_select(
        bdy_mask,
        positive_scalar,
        scalars_new_base_patch,
        mask_all=bdy_mask_all,
    )

    if return_scratch:
        scale_arr_last_patch = _monotonic_scalar_transport_scale_arr_last_scalar_patch(
            scale_in_limited,
            scale_out_limited,
            last_scalar,
        )
        wdtn_last_rows = _monotonic_scalar_transport_last_scalar_limited_rows(
            wdtn_patch=wdtn_patch,
            wdtn_limited_mid=wdtn_limited_mid,
            last_scalar=last_scalar,
            n_vert_levels=n_vert_levels,
        )
    else:
        empty_scratch = scalar_stage2.new_empty((0,))
        flux_limited_last = empty_scratch
        flux_upwind_last = empty_scratch
        flux_tmp_last = empty_scratch
        wdtn_last_rows = empty_scratch
        scale_arr_last_patch = empty_scratch
    return (
        scalar_stage2,
        scalars_new_patch,
        flux_limited_last,
        flux_upwind_last,
        flux_tmp_last,
        wdtn_last_rows,
        scale_arr_last_patch,
    )


def _monotonic_scalar_transport_streaming_scalar_transport_tile_numeric(
    pattern_edge_start: int,
    pattern_edge_end: int,
    pattern_edge_count: int,
    pattern_slot_count: int,
    pattern_adv_coef: torch.Tensor,
    pattern_adv_coef_3rd: torch.Tensor,
    pattern_slot_cells: torch.Tensor,
    pattern_slot_valid: torch.Tensor,
    pattern_slot_valid_all: bool | None,
    pattern_use_full_width: bool,
    pattern_fallback_rows: torch.Tensor,
    one: torch.Tensor,
    edge_cell_first: torch.Tensor,
    edge_cell_second: torch.Tensor,
    edge_cell_valid: torch.Tensor,
    edge_active_mask: torch.Tensor,
    edge_cell_first_local: torch.Tensor,
    edge_cell_second_local: torch.Tensor,
    edge_cell_first_local_valid: torch.Tensor,
    edge_cell_second_local_valid: torch.Tensor,
    edge_cell_first_local_all: bool,
    edge_cell_second_local_all: bool,
    cell_edge_indices: torch.Tensor,
    cell_edge_sign: torch.Tensor,
    cell_edge_valid: torch.Tensor,
    cell_edge_valid_all: tuple[bool, ...] | None,
    cell_edge_weights: torch.Tensor,
    cell_edge_valid_weight: torch.Tensor,
    bdy_mask: torch.Tensor,
    solve_rel_start: int,
    solve_rel_end: int,
    row_start: int,
    row_end: int,
    dt: float,
    eps: float,
    tau: float,
    edge_interior_all: bool,
    edge_active_all: bool,
    bdy_mask_all: bool,
    prefer_low_memory_reductions: bool,
    reduction_slot_tile_width: int | None,
    return_scratch: bool,
    /,
    *,
    scalar_field: torch.Tensor,
    uh_avg: torch.Tensor,
    scalar_old_rows: torch.Tensor,
    scalar_old_all: torch.Tensor,
    rho_zz_old_rows: torch.Tensor,
    ww_avg_rows: torch.Tensor,
    uh_avg_edges: torch.Tensor,
    dv_edge_patch: torch.Tensor,
    wdtn_patch: torch.Tensor,
    s_max_patch: torch.Tensor,
    s_min_patch: torch.Tensor,
    rho_zz_int_rows: torch.Tensor,
    scalar_new_all_patch: torch.Tensor,
    scalars_new_base_patch: torch.Tensor,
    rdnw: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compose the same pure phases when no network exchange is required."""
    static_args = (
        pattern_edge_start,
        pattern_edge_end,
        pattern_edge_count,
        pattern_slot_count,
        pattern_adv_coef,
        pattern_adv_coef_3rd,
        pattern_slot_cells,
        pattern_slot_valid,
        pattern_slot_valid_all,
        pattern_use_full_width,
        pattern_fallback_rows,
        one,
        edge_cell_first,
        edge_cell_second,
        edge_cell_valid,
        edge_active_mask,
        edge_cell_first_local,
        edge_cell_second_local,
        edge_cell_first_local_valid,
        edge_cell_second_local_valid,
        edge_cell_first_local_all,
        edge_cell_second_local_all,
        cell_edge_indices,
        cell_edge_sign,
        cell_edge_valid,
        cell_edge_valid_all,
        cell_edge_weights,
        cell_edge_valid_weight,
        bdy_mask,
        solve_rel_start,
        solve_rel_end,
        row_start,
        row_end,
        dt,
        eps,
        tau,
        edge_interior_all,
        edge_active_all,
        bdy_mask_all,
        prefer_low_memory_reductions,
        reduction_slot_tile_width,
        return_scratch,
    )
    dynamic_tensors = {
        "scalar_field": scalar_field,
        "uh_avg": uh_avg,
        "scalar_old_rows": scalar_old_rows,
        "scalar_old_all": scalar_old_all,
        "rho_zz_old_rows": rho_zz_old_rows,
        "ww_avg_rows": ww_avg_rows,
        "uh_avg_edges": uh_avg_edges,
        "dv_edge_patch": dv_edge_patch,
        "wdtn_patch": wdtn_patch,
        "s_max_patch": s_max_patch,
        "s_min_patch": s_min_patch,
        "rho_zz_int_rows": rho_zz_int_rows,
        "scalar_new_all_patch": scalar_new_all_patch,
        "scalars_new_base_patch": scalars_new_base_patch,
        "rdnw": rdnw,
    }
    prepared = _monotonic_scalar_transport_streaming_scale_prepare_numeric(
        *static_args, **dynamic_tensors
    )
    return _monotonic_scalar_transport_streaming_limited_update_numeric(
        *static_args,
        prepared=prepared,
        **dynamic_tensors,  # type: ignore[arg-type]
    )


def _monotonic_scalar_transport_streaming_scalar_transport_store_numeric(
    *static_args: object,
    scalars_new_base: torch.Tensor,
    store_start: int,
    store_end: int,
    **dynamic_tensors: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """The single-tile streaming transport followed by the scalars_new row store.

    Returns the tile outputs with the new-scalar patch replaced by
    ``scalars_new_base`` with rows ``store_start:store_end`` set to that patch,
    written once inside the region instead of a patch copied into the base
    afterwards. The base is only read.
    """
    outputs = _monotonic_scalar_transport_streaming_scalar_transport_tile_numeric(
        *static_args,  # type: ignore[arg-type]
        **dynamic_tensors,
    )
    return (
        outputs[0],
        _replace_row_slice(scalars_new_base, store_start, store_end, outputs[1]),
        *outputs[2:],
    )


@dataclass(frozen=True)
class MonotonicScalarTransportStreamingTilePlan:
    """One monotonic-scalar-transport tile callable with static topology and choices prebound."""

    static_key: tuple[object, ...]
    eager_callable: Callable[..., tuple[torch.Tensor, ...]]
    prepare_callable: Callable[..., tuple[torch.Tensor, ...]]
    update_callable: Callable[..., tuple[torch.Tensor, ...]]
    return_scratch: bool
    # The same single tile plus the scalars_new row store (see
    # _monotonic_scalar_transport_streaming_scalar_transport_store_numeric).
    store_callable: Callable[..., tuple[torch.Tensor, ...]] | None = None


@dataclass(frozen=True)
class MonotonicScalarTransportNumericPlan:
    """Runner-owned immutable monotonic-scalar-transport topology and scalar execution plan."""

    key: tuple[object, ...]
    connectivity_payload: Mapping[str, Any]
    scratch_tile_plan: MonotonicScalarTransportStreamingTilePlan
    no_scratch_tile_plan: MonotonicScalarTransportStreamingTilePlan
    num_scalars: int
    chunk_size: int
    scalar_tile_width: int
    scalar_partition_policy: str
    reduction_slot_tile_width: int | None
    advance_density: bool
    prefer_low_memory_reductions: bool


def _monotonic_scalar_transport_streaming_static_arguments(
    *,
    one: torch.Tensor,
    pattern_c_operator: PatternCOperator,
    edge_cell_relation: StaticPairRelation,
    cell_edge_relation: StaticSlotRelation,
    bdy_mask: torch.Tensor,
    solve_rel_start: int,
    solve_rel_end: int,
    row_start: int,
    row_end: int,
    dt: float,
    eps: float,
    tau: float,
    edge_interior_all: bool | None,
    edge_active_all: bool | None,
    bdy_mask_all: bool | None,
    prefer_low_memory_reductions: bool,
    reduction_slot_tile_width: int | None,
    return_scratch: bool,
) -> tuple[object, ...]:
    """Flatten monotonic-scalar-transport static objects once, outside the compiled numeric region."""
    payload = pattern_c_operator.payload
    edge_slice = pattern_c_operator.edge_slice
    pattern_tensors = (
        payload.get("adv_coef"),
        payload.get("adv_coef_3rd"),
        payload.get("slot_cells_matrix"),
        payload.get("slot_valid_matrix"),
        payload.get("fallback_rows"),
    )
    if not all(isinstance(value, torch.Tensor) for value in pattern_tensors):
        raise TypeError(
            "monotonic-scalar-transport Pattern-C operator must contain tensor-only static topology"
        )

    edge_cell_valid = edge_cell_relation.valid
    edge_active_mask = edge_cell_relation.active_3d
    if edge_cell_valid is None or edge_active_mask is None:
        raise ValueError(
            "monotonic-scalar-transport edge-cell relation requires valid and active masks"
        )
    edge_interior_all = (
        bool(edge_cell_relation.valid_all)
        if edge_interior_all is None
        else bool(edge_interior_all)
    )
    edge_active_all = (
        bool(edge_cell_relation.active_all)
        if edge_active_all is None
        else bool(edge_active_all)
    )

    first_local = edge_cell_relation.first_local
    second_local = edge_cell_relation.second_local
    first_local_valid = edge_cell_relation.first_local_valid
    second_local_valid = edge_cell_relation.second_local_valid
    row_span = max(int(row_end) - int(row_start) - 1, 0)
    if first_local is None or first_local_valid is None:
        first_local_valid = (edge_cell_relation.first >= row_start) & (
            edge_cell_relation.first < row_end
        )
        first_local = (edge_cell_relation.first - row_start).clamp(min=0, max=row_span)
    if second_local is None or second_local_valid is None:
        second_local_valid = (edge_cell_relation.second >= row_start) & (
            edge_cell_relation.second < row_end
        )
        second_local = (edge_cell_relation.second - row_start).clamp(
            min=0, max=row_span
        )
    first_local_all = (
        bool(edge_cell_relation.first_local_all)
        if edge_cell_relation.first_local_all is not None
        else bool(_monotonic_scalar_transport_static_all_true(first_local_valid))
    )
    second_local_all = (
        bool(edge_cell_relation.second_local_all)
        if edge_cell_relation.second_local_all is not None
        else bool(_monotonic_scalar_transport_static_all_true(second_local_valid))
    )

    cell_edge_sign = cell_edge_relation.sign
    cell_edge_valid = cell_edge_relation.valid
    cell_edge_weights = cell_edge_relation.weights
    cell_edge_valid_weight = cell_edge_relation.valid_weight
    if any(
        value is None
        for value in (
            cell_edge_sign,
            cell_edge_valid,
            cell_edge_weights,
            cell_edge_valid_weight,
        )
    ):
        raise ValueError(
            "monotonic-scalar-transport cell-edge relation lacks required masks or weights"
        )
    if bdy_mask_all is None:
        bdy_mask_all = bool(_monotonic_scalar_transport_static_all_true(bdy_mask))

    return (
        int(edge_slice.start),
        int(edge_slice.stop),
        int(pattern_c_operator.edge_count),
        int(pattern_c_operator.slot_count),
        *pattern_tensors[:4],
        payload.get("slot_valid_matrix_all"),
        bool(payload.get("use_full_width_patch_path", False)),
        pattern_tensors[4],
        one,
        edge_cell_relation.first,
        edge_cell_relation.second,
        edge_cell_valid,
        edge_active_mask,
        first_local,
        second_local,
        first_local_valid,
        second_local_valid,
        first_local_all,
        second_local_all,
        cell_edge_relation.indices,
        cell_edge_sign,
        cell_edge_valid,
        cell_edge_relation.valid_all_by_slot,
        cell_edge_weights,
        cell_edge_valid_weight,
        bdy_mask,
        int(solve_rel_start),
        int(solve_rel_end),
        int(row_start),
        int(row_end),
        float(dt),
        float(eps),
        float(tau),
        edge_interior_all,
        edge_active_all,
        bool(bdy_mask_all),
        bool(prefer_low_memory_reductions),
        None if reduction_slot_tile_width is None else int(reduction_slot_tile_width),
        bool(return_scratch),
    )


def _monotonic_scalar_transport_streaming_tile_plan(
    **static_kwargs: object,
) -> MonotonicScalarTransportStreamingTilePlan:
    """Build a stateless callable whose public signature contains dynamic tensors only."""
    static_args = _monotonic_scalar_transport_streaming_static_arguments(
        **static_kwargs  # type: ignore[arg-type]
    )
    tensor_keys: list[tuple[object, ...]] = []
    for value in static_args:
        if not isinstance(value, torch.Tensor):
            continue
        tensor_key = _tensor_version_cache_key(value)
        if tensor_key is None or _tensor_carries_autodiff(value):
            tensor_keys = []
            break
        tensor_keys.append(tensor_key)
    static_scalars = tuple(
        value for value in static_args if not isinstance(value, torch.Tensor)
    )
    cache_key: tuple[object, ...] = (
        "monotonic_scalar_transport_streaming_tile_plan_v1",
        *static_scalars,
        *tensor_keys,
    )
    return MonotonicScalarTransportStreamingTilePlan(
        static_key=cache_key,
        eager_callable=cast(
            Callable[..., tuple[torch.Tensor, ...]],
            functools.partial(
                _monotonic_scalar_transport_streaming_scalar_transport_tile_numeric,
                *static_args,  # type: ignore[arg-type]
            ),
        ),
        return_scratch=bool(static_args[-1]),
        prepare_callable=cast(
            Callable[..., tuple[torch.Tensor, ...]],
            functools.partial(
                _monotonic_scalar_transport_streaming_scale_prepare_numeric,
                *static_args,  # type: ignore[arg-type]
            ),
        ),
        update_callable=cast(
            Callable[..., tuple[torch.Tensor, ...]],
            functools.partial(
                _monotonic_scalar_transport_streaming_limited_update_numeric,
                *static_args,  # type: ignore[arg-type]
            ),
        ),
        store_callable=cast(
            Callable[..., tuple[torch.Tensor, ...]],
            functools.partial(
                _monotonic_scalar_transport_streaming_scalar_transport_store_numeric,
                *static_args,  # type: ignore[arg-type]
            ),
        ),
    )


def _monotonic_scalar_transport_streaming_scalar_transport_tile_region(
    *,
    scalar_field: torch.Tensor,
    uh_avg: torch.Tensor,
    one: torch.Tensor,
    pattern_c_operator: PatternCOperator,
    scalar_old_rows: torch.Tensor,
    scalar_old_all: torch.Tensor,
    rho_zz_old_rows: torch.Tensor,
    ww_avg_rows: torch.Tensor,
    uh_avg_edges: torch.Tensor,
    dv_edge_patch: torch.Tensor,
    wdtn_patch: torch.Tensor,
    s_max_patch: torch.Tensor,
    s_min_patch: torch.Tensor,
    rho_zz_int_rows: torch.Tensor,
    scalar_new_all_patch: torch.Tensor,
    scalars_new_base_patch: torch.Tensor,
    rdnw: torch.Tensor,
    edge_cell_relation: StaticPairRelation,
    cell_edge_relation: StaticSlotRelation,
    bdy_mask: torch.Tensor,
    solve_rel_start: int,
    solve_rel_end: int,
    row_start: int,
    row_end: int,
    dt: float,
    eps: float,
    tau: float,
    edge_interior_all: bool | None = None,
    edge_active_all: bool | None = None,
    bdy_mask_all: bool | None = None,
    prefer_low_memory_reductions: bool = False,
    reduction_slot_tile_width: int | None = None,
    return_scratch: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Compatibility boundary around the plan-owned tensor-only monotonic-scalar-transport tile."""
    plan = _monotonic_scalar_transport_streaming_tile_plan(
        one=one,
        pattern_c_operator=pattern_c_operator,
        edge_cell_relation=edge_cell_relation,
        cell_edge_relation=cell_edge_relation,
        bdy_mask=bdy_mask,
        solve_rel_start=solve_rel_start,
        solve_rel_end=solve_rel_end,
        row_start=row_start,
        row_end=row_end,
        dt=dt,
        eps=eps,
        tau=tau,
        edge_interior_all=edge_interior_all,
        edge_active_all=edge_active_all,
        bdy_mask_all=bdy_mask_all,
        prefer_low_memory_reductions=prefer_low_memory_reductions,
        reduction_slot_tile_width=reduction_slot_tile_width,
        return_scratch=return_scratch,
    )
    return plan.eager_callable(
        scalar_field=scalar_field,
        uh_avg=uh_avg,
        scalar_old_rows=scalar_old_rows,
        scalar_old_all=scalar_old_all,
        rho_zz_old_rows=rho_zz_old_rows,
        ww_avg_rows=ww_avg_rows,
        uh_avg_edges=uh_avg_edges,
        dv_edge_patch=dv_edge_patch,
        wdtn_patch=wdtn_patch,
        s_max_patch=s_max_patch,
        s_min_patch=s_min_patch,
        rho_zz_int_rows=rho_zz_int_rows,
        scalar_new_all_patch=scalar_new_all_patch,
        scalars_new_base_patch=scalars_new_base_patch,
        rdnw=rdnw,
    )


def _monotonic_scalar_transport_streaming_scalar_transport_region(
    *,
    scalar_field: torch.Tensor,
    uh_avg: torch.Tensor,
    one: torch.Tensor,
    pattern_c_operator: PatternCOperator,
    scalar_old_rows: torch.Tensor,
    scalar_old_all: torch.Tensor,
    rho_zz_old_rows: torch.Tensor,
    ww_avg_rows: torch.Tensor,
    uh_avg_edges: torch.Tensor,
    dv_edge_patch: torch.Tensor,
    wdtn_patch: torch.Tensor,
    s_max_patch: torch.Tensor,
    s_min_patch: torch.Tensor,
    rho_zz_int_rows: torch.Tensor,
    scalar_new_all_patch: torch.Tensor,
    scalars_new_base_patch: torch.Tensor,
    rdnw: torch.Tensor,
    edge_cell_relation: StaticPairRelation,
    cell_edge_relation: StaticSlotRelation,
    bdy_mask: torch.Tensor,
    solve_rel_start: int,
    solve_rel_end: int,
    row_start: int,
    row_end: int,
    dt: float,
    eps: float,
    tau: float,
    edge_interior_all: bool | None = None,
    edge_active_all: bool | None = None,
    bdy_mask_all: bool | None = None,
    prefer_low_memory_reductions: bool = False,
    reduction_slot_tile_width: int | None = None,
    scalar_tile_width: int | None = None,
    return_scratch: bool = True,
    _tile_region=None,
    _no_scratch_tile_region=None,
    _tile_static_bound: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Orchestrate monotonic scalar transport streaming phase scalar tiles outside compiled numeric code.

    Both the public chunk loop and this scalar-tile loop stay outside compile.
    Every tile uses the same functional numeric callable; concatenation and
    final-scalar scratch selection are orchestration-only operations.
    """
    tile_region = _monotonic_scalar_transport_streaming_scalar_transport_tile_region
    if _tile_region is not None:
        tile_region = _tile_region
    no_scratch_tile_region = tile_region
    if _no_scratch_tile_region is not None:
        no_scratch_tile_region = _no_scratch_tile_region

    def _invoke_tile(region, *, emit_scratch: bool, **dynamic_tensors: torch.Tensor):
        if _tile_static_bound:
            return region(**dynamic_tensors)
        return region(
            **dynamic_tensors,
            one=one,
            pattern_c_operator=pattern_c_operator,
            edge_cell_relation=edge_cell_relation,
            cell_edge_relation=cell_edge_relation,
            bdy_mask=bdy_mask,
            solve_rel_start=solve_rel_start,
            solve_rel_end=solve_rel_end,
            row_start=row_start,
            row_end=row_end,
            dt=dt,
            eps=eps,
            tau=tau,
            edge_interior_all=edge_interior_all,
            edge_active_all=edge_active_all,
            bdy_mask_all=bdy_mask_all,
            prefer_low_memory_reductions=prefer_low_memory_reductions,
            reduction_slot_tile_width=reduction_slot_tile_width,
            return_scratch=emit_scratch,
        )

    num_scalars = int(scalar_old_rows.shape[2])
    tile_width = _monotonic_scalar_transport_effective_streaming_scalar_tile_width(
        num_scalars,
        scalar_tile_width,
    )
    if tile_width >= num_scalars:
        current_tile_region = tile_region if return_scratch else no_scratch_tile_region
        return cast(
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
            _invoke_tile(
                current_tile_region,
                emit_scratch=return_scratch,
                scalar_field=scalar_field,
                uh_avg=uh_avg,
                scalar_old_rows=scalar_old_rows,
                scalar_old_all=scalar_old_all,
                rho_zz_old_rows=rho_zz_old_rows,
                ww_avg_rows=ww_avg_rows,
                uh_avg_edges=uh_avg_edges,
                dv_edge_patch=dv_edge_patch,
                wdtn_patch=wdtn_patch,
                s_max_patch=s_max_patch,
                s_min_patch=s_min_patch,
                rho_zz_int_rows=rho_zz_int_rows,
                scalar_new_all_patch=scalar_new_all_patch,
                scalars_new_base_patch=scalars_new_base_patch,
                rdnw=rdnw,
            ),
        )

    assemble_functionally = torch.is_grad_enabled()
    if assemble_functionally:
        scalar_stage2_chunks: list[torch.Tensor] = []
        scalars_new_patch_chunks: list[torch.Tensor] = []
    else:
        # Each tile result is owned and dead after assembly. Copy it immediately
        # into one final destination instead of retaining every tile and then
        # allocating another full pair of outputs through torch.cat. Besides
        # lowering peak memory, this releases any compiler-owned tile output
        # before the next invocation.
        scalar_stage2_result = torch.empty_like(scalar_old_rows)
        scalars_new_patch_result = torch.empty_like(scalars_new_base_patch)
    final_scratch: (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ) = None
    for scalar_start in range(0, num_scalars, tile_width):
        scalar_end = min(scalar_start + tile_width, num_scalars)
        emit_scratch = return_scratch and scalar_end == num_scalars
        current_tile_region = tile_region if emit_scratch else no_scratch_tile_region
        (
            scalar_stage2_tile,
            scalars_new_patch_tile,
            flux_limited_last,
            flux_upwind_last,
            flux_tmp_last,
            wdtn_last_rows,
            scale_arr_last_patch,
        ) = _invoke_tile(
            current_tile_region,
            emit_scratch=emit_scratch,
            scalar_field=_monotonic_scalar_transport_slice_scalar_dim(
                scalar_field, scalar_start, scalar_end
            ),
            uh_avg=uh_avg,
            scalar_old_rows=scalar_old_rows[..., scalar_start:scalar_end],
            scalar_old_all=scalar_old_all[..., scalar_start:scalar_end],
            rho_zz_old_rows=rho_zz_old_rows,
            ww_avg_rows=ww_avg_rows,
            uh_avg_edges=uh_avg_edges,
            dv_edge_patch=dv_edge_patch,
            wdtn_patch=wdtn_patch[..., scalar_start:scalar_end],
            s_max_patch=s_max_patch[..., scalar_start:scalar_end],
            s_min_patch=s_min_patch[..., scalar_start:scalar_end],
            rho_zz_int_rows=rho_zz_int_rows,
            scalar_new_all_patch=scalar_new_all_patch[..., scalar_start:scalar_end],
            scalars_new_base_patch=scalars_new_base_patch[..., scalar_start:scalar_end],
            rdnw=rdnw,
        )
        if assemble_functionally:
            scalar_stage2_chunks.append(scalar_stage2_tile)
            scalars_new_patch_chunks.append(scalars_new_patch_tile)
        else:
            scalar_stage2_result[..., scalar_start:scalar_end].copy_(scalar_stage2_tile)
            scalars_new_patch_result[..., scalar_start:scalar_end].copy_(
                scalars_new_patch_tile
            )
        if scalar_end == num_scalars:
            final_scratch = (
                flux_limited_last,
                flux_upwind_last,
                flux_tmp_last,
                wdtn_last_rows,
                scale_arr_last_patch,
            )

    if final_scratch is None:
        raise RuntimeError(
            "monotonic-scalar-transport streaming scalar tiling produced no final scalar scratch"
        )
    if assemble_functionally:
        scalar_stage2_result = torch.cat(scalar_stage2_chunks, dim=2)
        scalars_new_patch_result = torch.cat(scalars_new_patch_chunks, dim=2)
    return (
        scalar_stage2_result,
        scalars_new_patch_result,
        final_scratch[0],
        final_scratch[1],
        final_scratch[2],
        final_scratch[3],
        final_scratch[4],
    )


def _normalize_scalar_chunk_size(
    scalar_chunk_size: int | None, num_scalars: int
) -> int:
    """Return a compile-stable monotonic-scalar-transport scalar chunk width.

    A non-divisor chunk width leaves a smaller final chunk and therefore a
    second compiled shape for the same production stage. Use the largest
    divisor not exceeding the request so compiled transport has one scalar
    shape per scratch-emission variant.
    """
    if scalar_chunk_size is None:
        return num_scalars
    chunk_size = int(scalar_chunk_size)
    if chunk_size <= 0 or chunk_size >= num_scalars:
        return num_scalars
    return _monotonic_scalar_transport_effective_streaming_scalar_tile_width(
        num_scalars, chunk_size
    )


def _zero_scratch_for_output(tensor: torch.Tensor) -> torch.Tensor:
    """Reuse explicit monotonic-scalar-transport scratch arrays in inference mode; allocate in grad mode."""
    if not torch.is_grad_enabled() and not _compile_active():
        tensor.zero_()
        return tensor
    return torch.zeros_like(tensor)


def _monotonic_scalar_transport_stencil_field_with_zero_padding(
    field: torch.Tensor,
    padding_row: int,
) -> torch.Tensor:
    """Return a scalar stencil field with the unused padding row set to zero.

    monotonic-scalar-transport uses the trailing cell row as the safe gather target for absent
    neighbors.  Differentiable and compiled execution keep the old functional
    clone boundary; forward-only eager execution canonicalizes that unused row in
    place instead of cloning the full 3D scalar field.
    """
    if padding_row < 0 or padding_row >= field.shape[0]:
        return field
    out = _clone_for_grad(field)
    zeros = out.new_zeros((1, *out.shape[1:]))
    return _replace_row_slice(out, padding_row, padding_row + 1, zeros)


def _monotonic_scalar_transport_pack_or_return_scalars_new_patch(
    *,
    scalars_new_out: torch.Tensor,
    row_start: int,
    row_end: int,
    scalars_new_patch: torch.Tensor,
    return_compact_patch: bool = False,
) -> torch.Tensor:
    """Update or return monotonic-scalar-transport scalar rows without per-chunk full-field cloning."""
    if return_compact_patch:
        return scalars_new_patch
    if not torch.is_grad_enabled() and not _compile_active():
        scalars_new_out[row_start:row_end, :, :] = scalars_new_patch
        return scalars_new_out
    return _replace_row_slice(scalars_new_out, row_start, row_end, scalars_new_patch)


def _monotonic_scalar_transport_zero_row_patch_like(
    field: torch.Tensor,
    row_start: int,
    row_end: int,
) -> torch.Tensor:
    return field.new_zeros((int(row_end) - int(row_start), *field.shape[1:]))


def _monotonic_scalar_transport_source_tendency_numeric(
    *,
    scalars_old: torch.Tensor,
    scalar_tend: torch.Tensor,
    rho_zz_old: torch.Tensor,
    dt: float,
    read_start: int,
    read_stop: int,
    write_start: int,
    write_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the source tendency to the solve rows, then zero the tendency.

    Same expression and order as the former inline block.
    """
    rows = slice(read_start, read_stop)
    scalars_old_patch = scalars_old[rows, :, :] + dt * scalar_tend[
        rows, :, :
    ] / rho_zz_old[rows, :].unsqueeze(2)
    return (
        _replace_row_slice(scalars_old, write_start, write_end, scalars_old_patch),
        _replace_row_slice(
            scalar_tend,
            write_start,
            write_end,
            _monotonic_scalar_transport_zero_row_patch_like(
                scalar_tend, write_start, write_end
            ),
        ),
    )


def _monotonic_scalar_transport_density_update_numeric(
    *,
    uhAvg: torch.Tensor,
    cell_edge_relation: StaticSlotRelation,
    rdnw: torch.Tensor,
    wwAvg: torch.Tensor,
    rho_zz_old: torch.Tensor,
    rho_zz_int: torch.Tensor,
    dt: float,
    read_start: int,
    read_stop: int,
    write_start: int,
    write_end: int,
    n_vert_levels: int,
) -> torch.Tensor:
    """rho_zz_int on the solve rows from the flux divergences.

    Same expressions and order as the former inline block.
    """
    rows = slice(read_start, read_stop)
    horiz_div = _sum_density_flux_slots(
        edge_field=uhAvg,
        cell_edge_relation=cell_edge_relation,
    )
    vert_div = rdnw.view(1, n_vert_levels) * (
        wwAvg[rows, 1 : n_vert_levels + 1] - wwAvg[rows, 0:n_vert_levels]
    )
    rho_zz_patch = rho_zz_old[rows, :] + dt * (horiz_div - vert_div)
    return _replace_row_slice(rho_zz_int, write_start, write_end, rho_zz_patch)


def _monotonic_scalar_transport_scale_arr_last_scalar_patch(
    scale_in_limited: torch.Tensor,
    scale_out_limited: torch.Tensor,
    last_scalar: int,
) -> torch.Tensor:
    """Pack the final scalar's scale_arr scratch (one functional copy of the two slices)."""
    return torch.stack(
        (scale_in_limited[:, :, last_scalar], scale_out_limited[:, :, last_scalar]), dim=1
    )


def _monotonic_scalar_transport_tensor_cache_key(
    tensor: torch.Tensor,
) -> tuple[object, ...] | None:
    return _tensor_version_cache_key(tensor)


def _monotonic_scalar_transport_connectivity_payload_cache_key(
    **kwargs: object,
) -> tuple[object, ...] | None:
    """Return the mutation-sensitive cache key for a monotonic-scalar-transport static payload."""
    tensors = tuple(
        kwargs.get(name)
        for name in _MONOTONIC_SCALAR_TRANSPORT_CONNECTIVITY_TENSOR_NAMES
    )
    if not _static_connectivity_cache_allowed(tensors):
        return None

    tensor_keys: list[tuple[object, ...]] = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor_key = _monotonic_scalar_transport_tensor_cache_key(tensor)
            if tensor_key is None:
                return None
            tensor_keys.append(tensor_key)

    return (
        kwargs["device"],
        kwargs["nCells"],
        kwargs["nEdges"],
        kwargs["nCellsSolve"],
        kwargs["cell_start"],
        kwargs["cell_end"],
        kwargs["edge_start"],
        kwargs["edge_end"],
        kwargs["cell_solve_start"],
        kwargs["cell_solve_end"],
        kwargs["nSpecZone"],
        *tensor_keys,
    )


def _build_monotonic_scalar_transport_connectivity_payload(
    *,
    device: torch.device,
    nCells: int,
    nEdges: int,
    nCellsSolve: int,
    cell_start: int,
    cell_end: int,
    edge_start: int,
    edge_end: int,
    cell_solve_start: int,
    cell_solve_end: int,
    nSpecZone: int,
    edgesOnCell: torch.Tensor,
    nEdgesOnCell: torch.Tensor,
    edgesOnCell_sign: torch.Tensor,
    cellsOnCell: torch.Tensor,
    cellsOnEdge: torch.Tensor,
    bdyMaskCell: torch.Tensor,
    nAdvCellsForEdge: torch.Tensor,
    advCellsForEdge: torch.Tensor,
    adv_coefs: torch.Tensor,
    adv_coefs_3rd: torch.Tensor,
    dvEdge: torch.Tensor | None = None,
    invAreaCell: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build scalar-independent monotonic-scalar-transport transport topology for one mesh/domain slice."""
    compute_dtype = edgesOnCell_sign.dtype
    solve_cell_slice = slice(cell_solve_start, cell_solve_end)
    all_cell_slice = slice(cell_start, cell_end)
    edge_slice = slice(edge_start, edge_end)
    n_solve_cells = cell_solve_end - cell_solve_start

    padding_idx = torch.full((1,), nCells, dtype=torch.int64, device=device)  # device fill, no H2D

    density_n_edges_cell = nEdgesOnCell[solve_cell_slice]
    density_max_edges = _active_slot_count(
        density_n_edges_cell, edgesOnCell.shape[1], device
    )
    density_edge_slots = torch.arange(density_max_edges, device=device).unsqueeze(0)
    density_edge_valid = density_edge_slots < density_n_edges_cell.unsqueeze(1)
    density_cell_edges_1based = edgesOnCell[solve_cell_slice, :density_max_edges]
    density_cell_edges_0based = _monotonic_scalar_transport_safe_zero_invalid(
        density_cell_edges_1based - 1,
        density_edge_valid,
    )
    density_edge_sign = edgesOnCell_sign[solve_cell_slice, :density_max_edges].to(
        dtype=compute_dtype
    )

    neighbor_max_edges = _active_slot_count(
        nEdgesOnCell[solve_cell_slice], cellsOnCell.shape[1], device
    )
    neighbors_1based = cellsOnCell[solve_cell_slice, :neighbor_max_edges]
    neighbor_valid = (neighbors_1based >= 1) & (neighbors_1based <= nCells)
    neighbors_0based = _monotonic_scalar_transport_safe_zero_invalid(
        neighbors_1based - 1, neighbor_valid
    )
    neighbor_valid_all_by_slot = _monotonic_scalar_transport_slot_all_true(
        neighbor_valid,
        allow_cuda_sync=True,
    )

    edge_cell1_1based = cellsOnEdge[edge_slice, 0]
    edge_cell2_1based = cellsOnEdge[edge_slice, 1]
    edge_cell1 = edge_cell1_1based - 1
    edge_cell2 = edge_cell2_1based - 1
    edge_interior = (
        (edge_cell1_1based >= 1)
        & (edge_cell1_1based <= nCells)
        & (edge_cell2_1based >= 1)
        & (edge_cell2_1based <= nCells)
    )
    edge_active = edge_interior & (
        (edge_cell1 < nCellsSolve) | (edge_cell2 < nCellsSolve)
    )
    edge_cell1_safe = _monotonic_scalar_transport_safe_zero_invalid(
        edge_cell1, edge_interior
    )
    edge_cell2_safe = _monotonic_scalar_transport_safe_zero_invalid(
        edge_cell2, edge_interior
    )
    edge_cell1_local_valid = (edge_cell1_safe >= cell_solve_start) & (
        edge_cell1_safe < cell_solve_end
    )
    edge_cell2_local_valid = (edge_cell2_safe >= cell_solve_start) & (
        edge_cell2_safe < cell_solve_end
    )
    local_row_span = max(cell_solve_end - cell_solve_start - 1, 0)
    edge_cell1_local_idx = (edge_cell1_safe - cell_solve_start).clamp(
        min=0, max=local_row_span
    )
    edge_cell2_local_idx = (edge_cell2_safe - cell_solve_start).clamp(
        min=0, max=local_row_span
    )

    edge_slots = torch.arange(neighbor_max_edges, device=device).unsqueeze(0)
    n_edges_cell = nEdgesOnCell[solve_cell_slice]
    edge_valid = edge_slots < n_edges_cell.unsqueeze(1)
    cell_edges_1based = edgesOnCell[solve_cell_slice, :neighbor_max_edges]
    cell_edges_0based = _monotonic_scalar_transport_safe_zero_invalid(
        cell_edges_1based - 1, edge_valid
    )
    edge_sign = edgesOnCell_sign[solve_cell_slice, :neighbor_max_edges].to(
        dtype=compute_dtype
    )
    edge_reduce_valid = (
        edge_valid & (cell_edges_0based >= edge_start) & (cell_edges_0based < edge_end)
    )
    cell_edges_local = (cell_edges_0based - edge_start).clamp(
        min=0,
        max=max(edge_end - edge_start - 1, 0),
    )
    bdy_mask = (bdyMaskCell[all_cell_slice] <= nSpecZone).unsqueeze(1).unsqueeze(2)
    pattern_c_payload = _build_pattern_c_connectivity_payload(
        scalar_field=adv_coefs,
        n_adv_cells_for_edge=nAdvCellsForEdge,
        adv_cells_for_edge=advCellsForEdge,
        adv_coefs=adv_coefs,
        adv_coefs_3rd=adv_coefs_3rd,
        edge_start=edge_start + 1,
        edge_end=edge_end,
        n_adv_cells_max=advCellsForEdge.shape[1],
    )
    pattern_c_operator = PatternCOperator(pattern_c_payload)

    density_signed_edge_metric = None
    edge_dv = None
    if dvEdge is not None:
        density_signed_edge_metric = (
            -density_edge_sign * dvEdge[density_cell_edges_0based]
        )
        edge_dv = dvEdge[edge_slice]

    density_valid_inv_area = None
    edge_reduce_signed_inv_area = None
    edge_reduce_valid_inv_area = None
    if invAreaCell is not None:
        inv_area_solve = invAreaCell[solve_cell_slice].unsqueeze(1)
        density_valid_inv_area = (
            density_edge_valid.to(dtype=compute_dtype) * inv_area_solve
        )
        edge_reduce_valid_inv_area = (
            edge_reduce_valid.to(dtype=compute_dtype) * inv_area_solve
        )
        edge_reduce_signed_inv_area = edge_sign * edge_reduce_valid_inv_area

    density_edge_valid_all_by_slot = _monotonic_scalar_transport_slot_all_true(
        density_edge_valid,
        allow_cuda_sync=True,
    )
    edge_reduce_valid_all_by_slot = _monotonic_scalar_transport_slot_all_true(
        edge_reduce_valid,
        allow_cuda_sync=True,
    )
    edge_cell1_local_all = _monotonic_scalar_transport_static_all_true(
        edge_cell1_local_valid,
        allow_cuda_sync=True,
    )
    edge_cell2_local_all = _monotonic_scalar_transport_static_all_true(
        edge_cell2_local_valid,
        allow_cuda_sync=True,
    )
    edge_interior_all = _monotonic_scalar_transport_static_all_true(
        edge_interior, allow_cuda_sync=True
    )
    edge_active_all = _monotonic_scalar_transport_static_all_true(
        edge_active, allow_cuda_sync=True
    )
    bdy_mask_all = _monotonic_scalar_transport_static_all_true(
        bdy_mask, allow_cuda_sync=True
    )
    solve_cell_range = StaticRange(
        "monotonic_scalar_transport.solve_cells", cell_solve_start, cell_solve_end
    )
    all_cell_range = StaticRange(
        "monotonic_scalar_transport.all_cells", cell_start, cell_end
    )
    edge_range = StaticRange("monotonic_scalar_transport.edges", edge_start, edge_end)
    density_cell_edge_relation = StaticSlotRelation(
        "monotonic_scalar_transport.density_cell_edges",
        indices=density_cell_edges_0based,
        valid=density_edge_valid,
        sign=density_edge_sign,
        weights=density_signed_edge_metric,
        valid_all_by_slot=density_edge_valid_all_by_slot,
        valid_weight=density_valid_inv_area,
    )
    neighbor_cell_relation = StaticSlotRelation(
        "monotonic_scalar_transport.neighbor_cells",
        indices=neighbors_0based,
        valid=neighbor_valid,
        valid_all_by_slot=neighbor_valid_all_by_slot,
    )
    edge_cell_relation = StaticPairRelation(
        "monotonic_scalar_transport.edge_cells",
        first=edge_cell1_safe,
        second=edge_cell2_safe,
        valid=edge_interior,
        valid_all=edge_interior_all,
        first_local=edge_cell1_local_idx,
        second_local=edge_cell2_local_idx,
        first_local_valid=edge_cell1_local_valid,
        second_local_valid=edge_cell2_local_valid,
        first_local_all=edge_cell1_local_all,
        second_local_all=edge_cell2_local_all,
        active=edge_active,
        active_3d=edge_active.unsqueeze(1).unsqueeze(2),
        active_all=edge_active_all,
    )
    cell_edge_reduce_relation = StaticSlotRelation(
        "monotonic_scalar_transport.cell_edge_reduce",
        indices=cell_edges_local,
        valid=edge_reduce_valid,
        sign=edge_sign,
        weights=edge_reduce_signed_inv_area,
        valid_all_by_slot=edge_reduce_valid_all_by_slot,
        valid_weight=edge_reduce_valid_inv_area,
    )

    return {
        "solve_cell_slice": solve_cell_slice,
        "solve_cell_range": solve_cell_range,
        "all_cell_slice": all_cell_slice,
        "all_cell_range": all_cell_range,
        "edge_slice": edge_slice,
        "edge_range": edge_range,
        "n_solve_cells": n_solve_cells,
        "padding_row": nCells,
        "padding_idx": padding_idx,
        "density_edge_valid": density_edge_valid,
        "density_edge_valid_all_by_slot": density_edge_valid_all_by_slot,
        "density_cell_edges_0based": density_cell_edges_0based,
        "density_edge_sign": density_edge_sign,
        "density_signed_edge_metric": density_signed_edge_metric,
        "density_valid_inv_area": density_valid_inv_area,
        "density_cell_edge_relation": density_cell_edge_relation,
        "neighbors_0based": neighbors_0based,
        "neighbor_valid": neighbor_valid,
        "neighbor_valid_all_by_slot": neighbor_valid_all_by_slot,
        "neighbor_cell_relation": neighbor_cell_relation,
        "edge_cell1": edge_cell1,
        "edge_cell2": edge_cell2,
        "edge_interior": edge_interior,
        "edge_active": edge_active,
        "edge_cell1_safe": edge_cell1_safe,
        "edge_cell2_safe": edge_cell2_safe,
        "edge_cell1_local_idx": edge_cell1_local_idx,
        "edge_cell2_local_idx": edge_cell2_local_idx,
        "edge_cell1_local_valid": edge_cell1_local_valid,
        "edge_cell2_local_valid": edge_cell2_local_valid,
        "edge_dv": edge_dv,
        "edge_cell1_local_all": edge_cell1_local_all,
        "edge_cell2_local_all": edge_cell2_local_all,
        "edge_interior_all": edge_interior_all,
        "edge_active_all": edge_active_all,
        "edge_cell_relation": edge_cell_relation,
        "edge_reduce_valid": edge_reduce_valid,
        "edge_reduce_valid_all_by_slot": edge_reduce_valid_all_by_slot,
        "cell_edges_local": cell_edges_local,
        "edge_sign": edge_sign,
        "edge_reduce_signed_inv_area": edge_reduce_signed_inv_area,
        "edge_reduce_valid_inv_area": edge_reduce_valid_inv_area,
        "cell_edge_reduce_relation": cell_edge_reduce_relation,
        "bdy_mask": bdy_mask,
        "bdy_mask_all": bdy_mask_all,
        "pattern_c_payload": pattern_c_payload,
        "pattern_c_operator": pattern_c_operator,
        "one": pattern_c_payload["one"],
    }


def _monotonic_scalar_transport_connectivity_payload(
    **kwargs: object,
) -> dict[str, Any]:
    """Build monotonic-scalar-transport topology for direct callers; runner caching lives in DycoreStaticPayloads."""
    return _build_monotonic_scalar_transport_connectivity_payload(**kwargs)  # type: ignore[arg-type]


def monotonic_scalar_transport_numeric_plan_owner_key(
    *,
    connectivity_key: tuple[object, ...] | None,
    nCells: int,
    nEdges: int,
    nVertLevels: int,
    num_scalars: int,
    cellStart: int,
    cellEnd: int,
    edgeStart: int,
    edgeEnd: int,
    cellSolveStart: int,
    cellSolveEnd: int,
    dt: float,
    soft_limiter_tau: float,
    scalar_chunk_size: int,
    streaming_scalar_tile_width: int | None,
    reduction_slot_tile_width: int | None,
    advance_density: bool,
    prefer_low_memory_reductions: bool = False,
) -> tuple[object, ...]:
    """Return the runner cache key without constructing tile closures."""
    chunk_size = _normalize_scalar_chunk_size(scalar_chunk_size, num_scalars)
    tile_width = _monotonic_scalar_transport_effective_streaming_scalar_tile_width(
        chunk_size,
        streaming_scalar_tile_width,
    )
    reduction_width = (
        0
        if reduction_slot_tile_width is None or int(reduction_slot_tile_width) <= 0
        else int(reduction_slot_tile_width)
    )
    return (
        "MonotonicScalarTransportNumericPlan.v2",
        connectivity_key,
        int(nCells),
        int(nEdges),
        int(nVertLevels),
        int(num_scalars),
        int(chunk_size),
        int(tile_width),
        "largest_divisor_no_remainder",
        reduction_width,
        bool(advance_density),
        bool(prefer_low_memory_reductions) or chunk_size < int(num_scalars),
        float(dt),
        float(soft_limiter_tau),
        int(cellStart),
        int(cellEnd),
        int(edgeStart),
        int(edgeEnd),
        int(cellSolveStart),
        int(cellSolveEnd),
    )


def build_monotonic_scalar_transport_numeric_plan(
    *,
    connectivity_payload: Mapping[str, Any],
    connectivity_key: tuple[object, ...] | None,
    nCells: int,
    nEdges: int,
    nVertLevels: int,
    num_scalars: int,
    cellStart: int,
    cellEnd: int,
    edgeStart: int,
    edgeEnd: int,
    cellSolveStart: int,
    cellSolveEnd: int,
    dt: float,
    soft_limiter_tau: float,
    scalar_chunk_size: int,
    streaming_scalar_tile_width: int | None,
    reduction_slot_tile_width: int | None,
    advance_density: bool,
    prefer_low_memory_reductions: bool = False,
) -> MonotonicScalarTransportNumericPlan:
    """Bind monotonic-scalar-transport topology and all shape-changing choices into one runner plan."""
    payload = MappingProxyType(dict(connectivity_payload))
    pattern_c_operator = payload.get("pattern_c_operator")
    edge_cell_relation = payload.get("edge_cell_relation")
    cell_edge_relation = payload.get("cell_edge_reduce_relation")
    bdy_mask = payload.get("bdy_mask")
    if not isinstance(pattern_c_operator, PatternCOperator):
        raise TypeError("monotonic-scalar-transport plan requires pattern_c_operator")
    if not isinstance(edge_cell_relation, StaticPairRelation):
        raise TypeError("monotonic-scalar-transport plan requires edge_cell_relation")
    if not isinstance(cell_edge_relation, StaticSlotRelation):
        raise TypeError(
            "monotonic-scalar-transport plan requires cell_edge_reduce_relation"
        )
    if not isinstance(bdy_mask, torch.Tensor):
        raise TypeError("monotonic-scalar-transport plan requires bdy_mask")

    chunk_size = _normalize_scalar_chunk_size(scalar_chunk_size, num_scalars)
    tile_width = _monotonic_scalar_transport_effective_streaming_scalar_tile_width(
        chunk_size,
        streaming_scalar_tile_width,
    )
    low_memory = bool(prefer_low_memory_reductions) or chunk_size < int(num_scalars)
    reduction_width = (
        None
        if reduction_slot_tile_width is None or int(reduction_slot_tile_width) <= 0
        else int(reduction_slot_tile_width)
    )
    py_cell_start = int(cellStart) - 1
    py_solve_start = int(cellSolveStart) - 1
    py_solve_end = int(cellSolveEnd)
    static_kwargs = {
        "one": payload["one"],
        "pattern_c_operator": pattern_c_operator,
        "edge_cell_relation": edge_cell_relation,
        "cell_edge_relation": cell_edge_relation,
        "bdy_mask": bdy_mask,
        "solve_rel_start": py_solve_start - py_cell_start,
        "solve_rel_end": py_solve_end - py_cell_start,
        "row_start": py_solve_start,
        "row_end": py_solve_end,
        "dt": float(dt),
        "eps": 1.0e-20,
        "tau": float(soft_limiter_tau),
        "edge_interior_all": payload.get("edge_interior_all"),
        "edge_active_all": payload.get("edge_active_all"),
        "bdy_mask_all": payload.get("bdy_mask_all"),
        "prefer_low_memory_reductions": low_memory,
        "reduction_slot_tile_width": reduction_width,
    }
    scratch_plan = _monotonic_scalar_transport_streaming_tile_plan(
        **static_kwargs, return_scratch=True
    )
    no_scratch_plan = _monotonic_scalar_transport_streaming_tile_plan(
        **static_kwargs, return_scratch=False
    )
    key = monotonic_scalar_transport_numeric_plan_owner_key(
        connectivity_key=connectivity_key,
        nCells=nCells,
        nEdges=nEdges,
        nVertLevels=nVertLevels,
        num_scalars=num_scalars,
        cellStart=cellStart,
        cellEnd=cellEnd,
        edgeStart=edgeStart,
        edgeEnd=edgeEnd,
        cellSolveStart=cellSolveStart,
        cellSolveEnd=cellSolveEnd,
        dt=dt,
        soft_limiter_tau=soft_limiter_tau,
        scalar_chunk_size=scalar_chunk_size,
        streaming_scalar_tile_width=streaming_scalar_tile_width,
        reduction_slot_tile_width=reduction_slot_tile_width,
        advance_density=advance_density,
        prefer_low_memory_reductions=prefer_low_memory_reductions,
    )
    return MonotonicScalarTransportNumericPlan(
        key=key,
        connectivity_payload=payload,
        scratch_tile_plan=scratch_plan,
        no_scratch_tile_plan=no_scratch_plan,
        num_scalars=int(num_scalars),
        chunk_size=int(chunk_size),
        scalar_tile_width=int(tile_width),
        scalar_partition_policy="largest_divisor_no_remainder",
        reduction_slot_tile_width=reduction_width,
        advance_density=bool(advance_density),
        prefer_low_memory_reductions=bool(low_memory),
    )


def atm_advance_scalars_mono_work(
    nCells: int,
    nEdges: int,
    nVertLevels: int,
    num_scalars: int,
    dt: float,
    coef_3rd_order: float,
    nCellsSolve: int,
    cellStart: int,  # Fortran 1-based
    cellEnd: int,  # Fortran 1-based
    edgeStart: int,  # Fortran 1-based
    edgeEnd: int,  # Fortran 1-based
    cellSolveStart: int,  # Fortran 1-based
    cellSolveEnd: int,  # Fortran 1-based
    fnm: torch.Tensor,
    fnp: torch.Tensor,
    rdnw: torch.Tensor,
    dvEdge: torch.Tensor,
    invAreaCell: torch.Tensor,
    nEdgesOnCell: torch.Tensor,
    nAdvCellsForEdge: torch.Tensor,
    bdyMaskCell: torch.Tensor,
    bdyMaskEdge: torch.Tensor,
    cellsOnEdge: torch.Tensor,
    cellsOnCell: torch.Tensor,
    edgesOnCell: torch.Tensor,
    advCellsForEdge: torch.Tensor,
    edgesOnCell_sign: torch.Tensor,
    adv_coefs: torch.Tensor,
    adv_coefs_3rd: torch.Tensor,
    uhAvg: torch.Tensor,
    wwAvg: torch.Tensor,
    rho_zz_old: torch.Tensor,
    rho_zz_new: torch.Tensor,
    scalar_old: torch.Tensor,
    scalar_new: torch.Tensor,
    s_max: torch.Tensor,
    s_min: torch.Tensor,
    wdtn: torch.Tensor,
    flux_arr: torch.Tensor,
    flux_upwind_tmp: torch.Tensor,
    flux_tmp: torch.Tensor,
    scalar_tend: torch.Tensor,
    scalars_old: torch.Tensor,
    scalars_new: torch.Tensor,
    scale_arr: torch.Tensor,
    rho_zz_int: torch.Tensor,
    advance_density: bool = True,
    soft_limiter_tau: float = 0.01,
    nSpecZone: int = 2,
    scalar_chunk_size: int = 0,
    _precomputed_rho_zz_int: torch.Tensor | None = None,
    _compile_region_names: tuple[str, ...] | frozenset[str] | None = None,
    # Regions admitted for compiled reverse AD (see compile_utils admission).
    _compile_reverse_ad_region_names: tuple[str, ...] | frozenset[str] = (),
    _compile_mode: str = "reduce-overhead",
    _compile_backend: str = "inductor",
    _connectivity_payload: Mapping[str, Any] | None = None,
    _profile_sections: bool = False,
    _prefer_low_memory_reductions: bool = False,
    _reduction_slot_tile_width: int | None = None,
    _streaming_scalar_tile_width: int | None = None,
    _numeric_plan: MonotonicScalarTransportNumericPlan | None = None,
    _chunk_checkpoint: Callable[[Callable[[], dict]], dict] | None = None,
    _emit_scratch_outputs: bool = True,
) -> dict:
    """Public monotonic-scalar-transport entrypoint with runner-owned scalar chunk orchestration.

    ``_emit_scratch_outputs=False`` (a runner whose caller drops the Fortran
    scratch fields after the step) runs every chunk through the no-scratch
    tile region; the scratch entries of the result are then the inputs,
    unchanged. Scalar, tendency and density outputs are identical either way.
    """
    device = require_consistent_device(
        "atm_advance_scalars_mono_work",
        fnm,
        fnp,
        rdnw,
        dvEdge,
        invAreaCell,
        nEdgesOnCell,
        nAdvCellsForEdge,
        bdyMaskCell,
        bdyMaskEdge,
        cellsOnEdge,
        cellsOnCell,
        edgesOnCell,
        advCellsForEdge,
        edgesOnCell_sign,
        adv_coefs,
        adv_coefs_3rd,
        uhAvg,
        wwAvg,
        rho_zz_old,
        rho_zz_new,
        scalar_old,
        scalar_new,
        s_max,
        s_min,
        wdtn,
        flux_arr,
        flux_upwind_tmp,
        flux_tmp,
        scalar_tend,
        scalars_old,
        scalars_new,
        scale_arr,
        rho_zz_int,
    )
    chunk_size = _normalize_scalar_chunk_size(scalar_chunk_size, num_scalars)
    if _compile_active():
        raise RuntimeError(
            "Public monotonic-scalar-transport scalar chunk orchestration is not a torch.compile "
            "region. Compile per-chunk monotonic-scalar-transport numeric helpers instead."
        )
    if _numeric_plan is not None:
        requested_tile_width = (
            _monotonic_scalar_transport_effective_streaming_scalar_tile_width(
                chunk_size,
                _streaming_scalar_tile_width,
            )
        )
        requested_reduction_width = (
            None
            if _reduction_slot_tile_width is None
            or int(_reduction_slot_tile_width) <= 0
            else int(_reduction_slot_tile_width)
        )
        requested_choices = (
            int(num_scalars),
            int(chunk_size),
            int(requested_tile_width),
            requested_reduction_width,
            bool(advance_density),
            bool(_prefer_low_memory_reductions) or chunk_size < num_scalars,
        )
        plan_choices = (
            _numeric_plan.num_scalars,
            _numeric_plan.chunk_size,
            _numeric_plan.scalar_tile_width,
            _numeric_plan.reduction_slot_tile_width,
            _numeric_plan.advance_density,
            _numeric_plan.prefer_low_memory_reductions,
        )
        if plan_choices != requested_choices:
            raise ValueError(
                f"monotonic-scalar-transport numeric plan mismatch: expected {requested_choices}, got {plan_choices}"
            )
    if chunk_size >= num_scalars:
        whole = functools.partial(
            _atm_advance_scalars_mono_work_no_chunk,
            nCells=nCells,
            nEdges=nEdges,
            nVertLevels=nVertLevels,
            num_scalars=num_scalars,
            dt=dt,
            coef_3rd_order=coef_3rd_order,
            nCellsSolve=nCellsSolve,
            cellStart=cellStart,
            cellEnd=cellEnd,
            edgeStart=edgeStart,
            edgeEnd=edgeEnd,
            cellSolveStart=cellSolveStart,
            cellSolveEnd=cellSolveEnd,
            fnm=fnm,
            fnp=fnp,
            rdnw=rdnw,
            dvEdge=dvEdge,
            invAreaCell=invAreaCell,
            nEdgesOnCell=nEdgesOnCell,
            nAdvCellsForEdge=nAdvCellsForEdge,
            bdyMaskCell=bdyMaskCell,
            bdyMaskEdge=bdyMaskEdge,
            cellsOnEdge=cellsOnEdge,
            cellsOnCell=cellsOnCell,
            edgesOnCell=edgesOnCell,
            advCellsForEdge=advCellsForEdge,
            edgesOnCell_sign=edgesOnCell_sign,
            adv_coefs=adv_coefs,
            adv_coefs_3rd=adv_coefs_3rd,
            uhAvg=uhAvg,
            wwAvg=wwAvg,
            rho_zz_old=rho_zz_old,
            rho_zz_new=rho_zz_new,
            scalar_old=scalar_old,
            scalar_new=scalar_new,
            s_max=s_max,
            s_min=s_min,
            wdtn=wdtn,
            flux_arr=flux_arr,
            flux_upwind_tmp=flux_upwind_tmp,
            flux_tmp=flux_tmp,
            scalar_tend=scalar_tend,
            scalars_old=scalars_old,
            scalars_new=scalars_new,
            scale_arr=scale_arr,
            rho_zz_int=rho_zz_int,
            advance_density=advance_density,
            soft_limiter_tau=soft_limiter_tau,
            nSpecZone=nSpecZone,
            scalar_chunk_size=0,
            _precomputed_rho_zz_int=_precomputed_rho_zz_int,
            _compile_region_names=_compile_region_names,
            _compile_reverse_ad_region_names=_compile_reverse_ad_region_names,
            _compile_mode=_compile_mode,
            _compile_backend=_compile_backend,
            _connectivity_payload=(
                _numeric_plan.connectivity_payload
                if _numeric_plan is not None
                else _connectivity_payload
            ),
            _profile_sections=_profile_sections,
            _validated_device=device,
            _pack_scratch_outputs=bool(_emit_scratch_outputs),
            _return_compact_scalars_new_patch=False,
            _store_scalars_new_in_region=not torch.is_grad_enabled(),
            _prefer_low_memory_reductions=bool(_prefer_low_memory_reductions),
            _reduction_slot_tile_width=_reduction_slot_tile_width,
            _streaming_scalar_tile_width=_streaming_scalar_tile_width,
            _numeric_plan=_numeric_plan,
        )
        return whole() if _chunk_checkpoint is None else _chunk_checkpoint(whole)

    py_cellStart = cellStart - 1
    py_cellEnd = cellEnd
    py_edgeStart = edgeStart - 1
    py_edgeEnd = edgeEnd
    py_cellSolveStart = cellSolveStart - 1
    py_cellSolveEnd = cellSolveEnd
    chunk_payload = (
        _numeric_plan.connectivity_payload
        if _numeric_plan is not None
        else _connectivity_payload
    )
    if chunk_payload is None:
        with profile_section(
            "mpas.monotonic_scalar_transport.topology",
            enabled=bool(_profile_sections) and not _compile_active(),
        ):
            chunk_payload = _monotonic_scalar_transport_connectivity_payload(
                device=device,
                nCells=nCells,
                nEdges=nEdges,
                nCellsSolve=nCellsSolve,
                cell_start=py_cellStart,
                cell_end=py_cellEnd,
                edge_start=py_edgeStart,
                edge_end=py_edgeEnd,
                cell_solve_start=py_cellSolveStart,
                cell_solve_end=py_cellSolveEnd,
                nSpecZone=nSpecZone,
                edgesOnCell=edgesOnCell,
                nEdgesOnCell=nEdgesOnCell,
                edgesOnCell_sign=edgesOnCell_sign,
                cellsOnCell=cellsOnCell,
                cellsOnEdge=cellsOnEdge,
                bdyMaskCell=bdyMaskCell,
                nAdvCellsForEdge=nAdvCellsForEdge,
                advCellsForEdge=advCellsForEdge,
                adv_coefs=adv_coefs,
                adv_coefs_3rd=adv_coefs_3rd,
                dvEdge=dvEdge,
                invAreaCell=invAreaCell,
            )

    if _numeric_plan is None:
        connectivity_key = _monotonic_scalar_transport_connectivity_payload_cache_key(
            device=device,
            nCells=nCells,
            nEdges=nEdges,
            nCellsSolve=nCellsSolve,
            cell_start=py_cellStart,
            cell_end=py_cellEnd,
            edge_start=py_edgeStart,
            edge_end=py_edgeEnd,
            cell_solve_start=py_cellSolveStart,
            cell_solve_end=py_cellSolveEnd,
            nSpecZone=nSpecZone,
            edgesOnCell=edgesOnCell,
            nEdgesOnCell=nEdgesOnCell,
            edgesOnCell_sign=edgesOnCell_sign,
            cellsOnCell=cellsOnCell,
            cellsOnEdge=cellsOnEdge,
            bdyMaskCell=bdyMaskCell,
            nAdvCellsForEdge=nAdvCellsForEdge,
            advCellsForEdge=advCellsForEdge,
            adv_coefs=adv_coefs,
            adv_coefs_3rd=adv_coefs_3rd,
            dvEdge=dvEdge,
            invAreaCell=invAreaCell,
        )
        _numeric_plan = build_monotonic_scalar_transport_numeric_plan(
            connectivity_payload=chunk_payload,
            connectivity_key=connectivity_key,
            nCells=nCells,
            nEdges=nEdges,
            nVertLevels=nVertLevels,
            num_scalars=num_scalars,
            cellStart=cellStart,
            cellEnd=cellEnd,
            edgeStart=edgeStart,
            edgeEnd=edgeEnd,
            cellSolveStart=cellSolveStart,
            cellSolveEnd=cellSolveEnd,
            dt=dt,
            soft_limiter_tau=soft_limiter_tau,
            scalar_chunk_size=chunk_size,
            streaming_scalar_tile_width=_streaming_scalar_tile_width,
            reduction_slot_tile_width=_reduction_slot_tile_width,
            advance_density=advance_density,
            prefer_low_memory_reductions=bool(_prefer_low_memory_reductions),
        )
    expected_tile_width = (
        _monotonic_scalar_transport_effective_streaming_scalar_tile_width(
            chunk_size,
            _streaming_scalar_tile_width,
        )
    )
    expected_reduction_width = (
        None
        if _reduction_slot_tile_width is None or int(_reduction_slot_tile_width) <= 0
        else int(_reduction_slot_tile_width)
    )
    expected_plan_choices = (
        int(num_scalars),
        int(chunk_size),
        int(expected_tile_width),
        expected_reduction_width,
        bool(advance_density),
        bool(_prefer_low_memory_reductions) or chunk_size < num_scalars,
    )
    actual_plan_choices = (
        _numeric_plan.num_scalars,
        _numeric_plan.chunk_size,
        _numeric_plan.scalar_tile_width,
        _numeric_plan.reduction_slot_tile_width,
        _numeric_plan.advance_density,
        _numeric_plan.prefer_low_memory_reductions,
    )
    if actual_plan_choices != expected_plan_choices:
        raise ValueError(
            f"monotonic-scalar-transport numeric plan mismatch: expected {expected_plan_choices}, "
            f"got {actual_plan_choices}"
        )

    assemble_functionally = torch.is_grad_enabled()
    if assemble_functionally:
        scalars_new_chunks: list[torch.Tensor] = []
        scalars_old_chunks: list[torch.Tensor] = []
        scalar_tend_chunks: list[torch.Tensor] = []
    else:
        scalars_new_result = scalars_new
        scalars_old_result = scalars_old
        scalar_tend_result = scalar_tend
    last_result: dict | None = None
    rho_zz_int_for_chunks = _precomputed_rho_zz_int
    prefer_low_memory_reductions = bool(_prefer_low_memory_reductions) or (
        chunk_size < num_scalars
    )
    for scalar_start in range(0, num_scalars, chunk_size):
        scalar_end = min(scalar_start + chunk_size, num_scalars)
        rho_zz_int_input = (
            rho_zz_int if rho_zz_int_for_chunks is None else rho_zz_int_for_chunks
        )
        scalar_tend_chunk = scalar_tend[:, :, scalar_start:scalar_end]
        scalars_old_chunk = scalars_old[:, :, scalar_start:scalar_end]
        scalars_new_chunk = scalars_new[:, :, scalar_start:scalar_end]
        # Under grad, each scalar chunk may run as its own checkpoint (``_chunk_checkpoint``):
        # its activations are recomputed in backward, so the retained and the backward-peak
        # memory scale with the chunk, not with all scalars.
        run_chunk = functools.partial(
            _atm_advance_scalars_mono_work_no_chunk,
            nCells=nCells,
            nEdges=nEdges,
            nVertLevels=nVertLevels,
            num_scalars=scalar_end - scalar_start,
            dt=dt,
            coef_3rd_order=coef_3rd_order,
            nCellsSolve=nCellsSolve,
            cellStart=cellStart,
            cellEnd=cellEnd,
            edgeStart=edgeStart,
            edgeEnd=edgeEnd,
            cellSolveStart=cellSolveStart,
            cellSolveEnd=cellSolveEnd,
            fnm=fnm,
            fnp=fnp,
            rdnw=rdnw,
            dvEdge=dvEdge,
            invAreaCell=invAreaCell,
            nEdgesOnCell=nEdgesOnCell,
            nAdvCellsForEdge=nAdvCellsForEdge,
            bdyMaskCell=bdyMaskCell,
            bdyMaskEdge=bdyMaskEdge,
            cellsOnEdge=cellsOnEdge,
            cellsOnCell=cellsOnCell,
            edgesOnCell=edgesOnCell,
            advCellsForEdge=advCellsForEdge,
            edgesOnCell_sign=edgesOnCell_sign,
            adv_coefs=adv_coefs,
            adv_coefs_3rd=adv_coefs_3rd,
            uhAvg=uhAvg,
            wwAvg=wwAvg,
            rho_zz_old=rho_zz_old,
            rho_zz_new=rho_zz_new,
            scalar_old=scalar_old,
            scalar_new=scalar_new,
            s_max=s_max,
            s_min=s_min,
            wdtn=wdtn,
            flux_arr=flux_arr,
            flux_upwind_tmp=flux_upwind_tmp,
            flux_tmp=flux_tmp,
            scalar_tend=scalar_tend_chunk,
            scalars_old=scalars_old_chunk,
            scalars_new=scalars_new_chunk,
            scale_arr=scale_arr,
            rho_zz_int=rho_zz_int_input,
            advance_density=advance_density,
            soft_limiter_tau=soft_limiter_tau,
            nSpecZone=nSpecZone,
            scalar_chunk_size=0,
            _precomputed_rho_zz_int=rho_zz_int_for_chunks,
            _compile_region_names=_compile_region_names,
            _compile_reverse_ad_region_names=_compile_reverse_ad_region_names,
            _compile_mode=_compile_mode,
            _compile_backend=_compile_backend,
            _connectivity_payload=chunk_payload,
            _profile_sections=_profile_sections,
            _validated_device=device,
            _pack_scratch_outputs=scalar_end == num_scalars
            and bool(_emit_scratch_outputs),
            _return_compact_scalars_new_patch=assemble_functionally,
            _prefer_low_memory_reductions=prefer_low_memory_reductions,
            _reduction_slot_tile_width=_reduction_slot_tile_width,
            _streaming_scalar_tile_width=_streaming_scalar_tile_width,
            _numeric_plan=_numeric_plan,
        )
        chunk_result = run_chunk() if _chunk_checkpoint is None else _chunk_checkpoint(run_chunk)
        if rho_zz_int_for_chunks is None:
            rho_zz_int_for_chunks = chunk_result["rho_zz_int"]
        if assemble_functionally:
            scalars_new_chunks.append(chunk_result["scalars_new"])
            scalars_old_chunks.append(chunk_result["scalars_old"])
            scalar_tend_chunks.append(chunk_result["scalar_tend"])
        else:
            # Without grad an eager chunk updates its slice views in place and
            # returns them. A compiled region returns new tensors and leaves the
            # views unchanged, so its results are written into the slices here.
            for view, name in (
                (scalars_new_chunk, "scalars_new"),
                (scalars_old_chunk, "scalars_old"),
                (scalar_tend_chunk, "scalar_tend"),
            ):
                if chunk_result[name] is not view:
                    view.copy_(chunk_result[name])
        last_result = chunk_result

    if last_result is None:
        raise ValueError(
            "monotonic-scalar-transport scalar chunking requires at least one scalar"
        )
    if assemble_functionally:
        scalars_new_cells = torch.cat(scalars_new_chunks, dim=2)
        scalars_new_result = _replace_row_slice(
            _clone_for_grad(scalars_new),
            py_cellStart,
            py_cellEnd,
            scalars_new_cells,
        )
        scalars_old_result = torch.cat(scalars_old_chunks, dim=2)
        scalar_tend_result = torch.cat(scalar_tend_chunks, dim=2)
    return {
        "scalars_new": scalars_new_result,
        "flux_arr": last_result["flux_arr"],
        "scalars_old": scalars_old_result,
        "scalar_tend": scalar_tend_result,
        "scalar_old": last_result["scalar_old"],
        "scalar_new": last_result["scalar_new"],
        "s_max": last_result["s_max"],
        "s_min": last_result["s_min"],
        "wdtn": last_result["wdtn"],
        "scale_arr": last_result["scale_arr"],
        "flux_upwind_tmp": last_result["flux_upwind_tmp"],
        "flux_tmp": last_result["flux_tmp"],
        "rho_zz_int": last_result["rho_zz_int"],
    }


def _atm_advance_scalars_mono_work_no_chunk(
    nCells: int,
    nEdges: int,
    nVertLevels: int,
    num_scalars: int,
    dt: float,
    coef_3rd_order: float,
    nCellsSolve: int,
    cellStart: int,  # Fortran 1-based
    cellEnd: int,  # Fortran 1-based
    edgeStart: int,  # Fortran 1-based
    edgeEnd: int,  # Fortran 1-based
    cellSolveStart: int,  # Fortran 1-based
    cellSolveEnd: int,  # Fortran 1-based
    # 1D arrays - 0-indexed for vertical levels
    fnm: torch.Tensor,  # (nVertLevels,) - 0-indexed
    fnp: torch.Tensor,  # (nVertLevels,) - 0-indexed
    rdnw: torch.Tensor,  # (nVertLevels,) - 0-indexed
    # 1D arrays - Python 0-based: valid cells at 0..nCells-1, padding at nCells
    dvEdge: torch.Tensor,  # (nEdges+1,)
    invAreaCell: torch.Tensor,  # (nCells+1,)
    nEdgesOnCell: torch.Tensor,  # (nCells+1,)
    nAdvCellsForEdge: torch.Tensor,  # (nEdges+1,)
    bdyMaskCell: torch.Tensor,  # (nCells+1,)
    bdyMaskEdge: torch.Tensor,  # (nEdges+1,)
    # Connectivity arrays (contain Fortran 1-based indices - subtract 1 for Python)
    cellsOnEdge: torch.Tensor,  # (nEdges+1, 2) - values are Fortran 1-based
    cellsOnCell: torch.Tensor,  # (nCells+1, maxEdges) - values are Fortran 1-based
    edgesOnCell: torch.Tensor,  # (nCells+1, maxEdges) - values are Fortran 1-based
    advCellsForEdge: torch.Tensor,  # (nEdges+1, nAdvCellsMax) - values are Fortran 1-based
    edgesOnCell_sign: torch.Tensor,  # (nCells+1, maxEdges)
    adv_coefs: torch.Tensor,  # (nEdges+1, nAdvCellsMax)
    adv_coefs_3rd: torch.Tensor,  # (nEdges+1, nAdvCellsMax)
    # 2D arrays - Python 0-based for cell/edge index
    uhAvg: torch.Tensor,  # (nEdges+1, nVertLevels)
    wwAvg: torch.Tensor,  # (nCells+1, nVertLevels+1)
    rho_zz_old: torch.Tensor,  # (nCells+1, nVertLevels)
    rho_zz_new: torch.Tensor,  # (nCells+1, nVertLevels)
    # Working arrays (input/output)
    scalar_old: torch.Tensor,  # (nCells+1, nVertLevels)
    scalar_new: torch.Tensor,  # (nCells+1, nVertLevels)
    s_max: torch.Tensor,  # (nCells+1, nVertLevels)
    s_min: torch.Tensor,  # (nCells+1, nVertLevels)
    wdtn: torch.Tensor,  # (nCells+1, nVertLevels+1)
    flux_arr: torch.Tensor,  # (nEdges+1, nVertLevels)
    flux_upwind_tmp: torch.Tensor,  # (nEdges+1, nVertLevels)
    flux_tmp: torch.Tensor,  # (nEdges+1, nVertLevels)
    # 3D arrays
    scalar_tend: torch.Tensor,  # (nCells+1, nVertLevels, num_scalars)
    scalars_old: torch.Tensor,  # (nCells+1, nVertLevels, num_scalars)
    scalars_new: torch.Tensor,  # (nCells+1, nVertLevels, num_scalars)
    scale_arr: torch.Tensor,  # (nCells+1, 2, nVertLevels)
    rho_zz_int: torch.Tensor,  # (nCells+1, nVertLevels)
    advance_density: bool = True,  # Fortran config_split_dynamics_transport default
    soft_limiter_tau: float = 0.01,
    nSpecZone: int = 2,
    scalar_chunk_size: int = 0,
    _precomputed_rho_zz_int: torch.Tensor | None = None,
    _compile_region_names: tuple[str, ...] | frozenset[str] | None = None,
    # Regions admitted for compiled reverse AD (see compile_utils admission).
    _compile_reverse_ad_region_names: tuple[str, ...] | frozenset[str] = (),
    _compile_mode: str = "reduce-overhead",
    _compile_backend: str = "inductor",
    _connectivity_payload: Mapping[str, Any] | None = None,
    _profile_sections: bool = False,
    _validated_device: torch.device | None = None,
    _pack_scratch_outputs: bool = True,
    _return_compact_scalars_new_patch: bool = False,
    _store_scalars_new_in_region: bool = False,
    _prefer_low_memory_reductions: bool = False,
    _reduction_slot_tile_width: int | None = None,
    _streaming_scalar_tile_width: int | None = None,
    _numeric_plan: MonotonicScalarTransportNumericPlan | None = None,
) -> dict:
    """
    Advance scalar fields using flux-corrected transport (FCT) for monotonicity.

    This routine uses a flux-limiting algorithm to ensure the updated scalar
    values remain within the bounds of the old values in the local stencil,
    preventing spurious oscillations (Gibbs phenomena) near sharp gradients.

    Returns:
        Dictionary containing updated output arrays:
            - scalars_new, flux_arr, scalars_old, scalar_tend, scalar_old,
              scalar_new, s_max, s_min, wdtn, scale_arr, flux_upwind_tmp,
              flux_tmp, rho_zz_int
    """
    compute_dtype = scalar_old.dtype
    if _validated_device is None:
        device = require_consistent_device(
            "atm_advance_scalars_mono_work",
            fnm,
            fnp,
            rdnw,
            dvEdge,
            invAreaCell,
            nEdgesOnCell,
            nAdvCellsForEdge,
            bdyMaskCell,
            bdyMaskEdge,
            cellsOnEdge,
            cellsOnCell,
            edgesOnCell,
            advCellsForEdge,
            edgesOnCell_sign,
            adv_coefs,
            adv_coefs_3rd,
            uhAvg,
            wwAvg,
            rho_zz_old,
            rho_zz_new,
            scalar_old,
            scalar_new,
            s_max,
            s_min,
            wdtn,
            flux_arr,
            flux_upwind_tmp,
            flux_tmp,
            scalar_tend,
            scalars_old,
            scalars_new,
            scale_arr,
            rho_zz_int,
        )
    else:
        device = _validated_device

    # Constants
    eps = 1.0e-20
    tau = soft_limiter_tau

    # Clone all output arrays to avoid modifying inputs
    scalars_new_out = _clone_for_grad(scalars_new)
    scalars_old_out = _clone_for_grad(scalars_old)
    scalar_tend_out = _clone_for_grad(scalar_tend)
    scalar_old_out = (
        _clone_for_grad(scalar_old) if _pack_scratch_outputs else scalar_old
    )
    scalar_new_out = (
        _clone_for_grad(scalar_new) if _pack_scratch_outputs else scalar_new
    )
    s_max_out = _clone_for_grad(s_max) if _pack_scratch_outputs else s_max
    s_min_out = _clone_for_grad(s_min) if _pack_scratch_outputs else s_min
    wdtn_out = _clone_for_grad(wdtn) if _pack_scratch_outputs else wdtn
    scale_arr_out = _clone_for_grad(scale_arr) if _pack_scratch_outputs else scale_arr
    flux_arr_out = _clone_for_grad(flux_arr) if _pack_scratch_outputs else flux_arr
    flux_upwind_tmp_out = (
        _zero_scratch_for_output(flux_upwind_tmp)
        if _pack_scratch_outputs
        else flux_upwind_tmp
    )
    flux_tmp_out = (
        _zero_scratch_for_output(flux_tmp) if _pack_scratch_outputs else flux_tmp
    )
    rho_zz_int_out = _clone_for_grad(rho_zz_int)

    # Convert Fortran 1-based loop bounds to Python 0-based
    # Fortran: do iCell = cellStart, cellEnd  (1-based, inclusive)
    # Python: for iCell in range(cellStart-1, cellEnd)  (0-based, exclusive end)
    py_cellStart = cellStart - 1
    py_cellEnd = cellEnd  # Python range is exclusive, so cellEnd works directly
    py_edgeStart = edgeStart - 1
    py_edgeEnd = edgeEnd
    py_cellSolveStart = cellSolveStart - 1
    py_cellSolveEnd = cellSolveEnd

    conn: Mapping[str, Any]
    if _connectivity_payload is None:
        with profile_section(
            "mpas.monotonic_scalar_transport.topology",
            enabled=bool(_profile_sections) and not _compile_active(),
        ):
            conn = _monotonic_scalar_transport_connectivity_payload(
                device=device,
                nCells=nCells,
                nEdges=nEdges,
                nCellsSolve=nCellsSolve,
                cell_start=py_cellStart,
                cell_end=py_cellEnd,
                edge_start=py_edgeStart,
                edge_end=py_edgeEnd,
                cell_solve_start=py_cellSolveStart,
                cell_solve_end=py_cellSolveEnd,
                nSpecZone=nSpecZone,
                edgesOnCell=edgesOnCell,
                nEdgesOnCell=nEdgesOnCell,
                edgesOnCell_sign=edgesOnCell_sign,
                cellsOnCell=cellsOnCell,
                cellsOnEdge=cellsOnEdge,
                bdyMaskCell=bdyMaskCell,
                nAdvCellsForEdge=nAdvCellsForEdge,
                advCellsForEdge=advCellsForEdge,
                adv_coefs=adv_coefs,
                adv_coefs_3rd=adv_coefs_3rd,
                dvEdge=dvEdge,
                invAreaCell=invAreaCell,
            )
    else:
        conn = _connectivity_payload
    one = conn.get("one")
    if not isinstance(one, torch.Tensor):
        one = torch.tensor(1.0, dtype=compute_dtype, device=device)
    compile_region_names = (
        frozenset(_compile_region_names or ())
        & _MONOTONIC_SCALAR_TRANSPORT_REGION_COMPILE_TARGETS
    )
    profile_sections = bool(_profile_sections) and not _compile_active()
    prefer_low_memory_reductions = bool(_prefer_low_memory_reductions)
    reduction_slot_tile_width = (
        None if _reduction_slot_tile_width is None else int(_reduction_slot_tile_width)
    )
    requested_streaming_scalar_tile_width = (
        reduction_slot_tile_width
        if _streaming_scalar_tile_width is None
        else int(_streaming_scalar_tile_width)
    )
    streaming_scalar_tile_width = (
        _monotonic_scalar_transport_effective_streaming_scalar_tile_width(
            num_scalars,
            requested_streaming_scalar_tile_width,
        )
    )

    def _monotonic_scalar_transport_profile(name: str):
        return profile_section(name, enabled=profile_sections)

    def _maybe_compile_monotonic_scalar_transport_region(
        fn_name: str, fn, specialization_key=None
    ):
        selected = fn_name in compile_region_names or (
            fn_name
            in {
                "monotonic_scalar_transport_streaming_scale_prepare_region",
                "monotonic_scalar_transport_streaming_limited_update_region",
                "monotonic_scalar_transport_streaming_scalar_transport_store_region",
            }
            and "monotonic_scalar_transport_streaming_scalar_transport_region"
            in compile_region_names
        )
        if not selected:
            return fn
        from mpas_pytorch.dynamics.compile_utils import maybe_compile_kernel

        return maybe_compile_kernel(
            fn_name,
            fn,
            enabled=True,
            mode=_compile_mode,
            backend=_compile_backend,
            specialization_key=specialization_key,
            allow_reverse_autodiff=fn_name in _compile_reverse_ad_region_names,
        )

    solve_cell_range = conn.get("solve_cell_range")
    solve_cell_slice = (
        solve_cell_range.slice
        if isinstance(solve_cell_range, StaticRange)
        else conn["solve_cell_slice"]
    )
    all_cell_range = conn.get("all_cell_range")
    all_cell_slice = (
        all_cell_range.slice
        if isinstance(all_cell_range, StaticRange)
        else conn["all_cell_slice"]
    )
    edge_range = conn.get("edge_range")
    edge_slice = (
        edge_range.slice if isinstance(edge_range, StaticRange) else conn["edge_slice"]
    )
    n_solve_cells = conn["n_solve_cells"]

    density_cell_edge_relation = conn.get("density_cell_edge_relation")
    neighbor_cell_relation = conn.get("neighbor_cell_relation")
    edge_cell_relation = conn.get("edge_cell_relation")
    cell_edge_reduce_relation = conn.get("cell_edge_reduce_relation")
    if not isinstance(density_cell_edge_relation, StaticSlotRelation):
        raise TypeError(
            "monotonic-scalar-transport connectivity payload must include density_cell_edge_relation"
        )
    if not isinstance(neighbor_cell_relation, StaticSlotRelation):
        raise TypeError(
            "monotonic-scalar-transport connectivity payload must include neighbor_cell_relation"
        )
    if not isinstance(edge_cell_relation, StaticPairRelation):
        raise TypeError(
            "monotonic-scalar-transport connectivity payload must include edge_cell_relation"
        )
    if not isinstance(cell_edge_reduce_relation, StaticSlotRelation):
        raise TypeError(
            "monotonic-scalar-transport connectivity payload must include cell_edge_reduce_relation"
        )
    pattern_c_operator = conn.get("pattern_c_operator")
    if not isinstance(pattern_c_operator, PatternCOperator):
        pattern_c_operator = PatternCOperator(conn["pattern_c_payload"])  # type: ignore[arg-type]
    pattern_c_signature = tuple(sorted(pattern_c_operator.snapshot().items()))

    edge_interior = edge_cell_relation.valid
    if edge_interior is None:
        raise ValueError(
            "monotonic-scalar-transport edge_cell_relation must include an interior valid mask"
        )
    edge_interior_all = (
        bool(edge_cell_relation.valid_all)
        if edge_cell_relation.valid_all is not None
        else bool(edge_interior.all())
    )
    edge_dv = conn.get("edge_dv")
    edge_active_mask = edge_cell_relation.active_3d
    edge_active_all = edge_cell_relation.active_all
    if edge_active_mask is None:
        raise ValueError(
            "monotonic-scalar-transport edge_cell_relation must include an active mask"
        )

    # =========================================================================
    # PART 1: Apply source tendencies to scalars_old
    # =========================================================================
    with _monotonic_scalar_transport_profile(
        "mpas.monotonic_scalar_transport.source_tendency"
    ):
        # Source tendency applied, then zeroed: one pure region (compiled
        # when selected).
        source_tendency_region = _maybe_compile_monotonic_scalar_transport_region(
            "monotonic_scalar_transport_source_tendency_region",
            _monotonic_scalar_transport_source_tendency_numeric,
        )
        scalars_old_out, scalar_tend_out = source_tendency_region(
            scalars_old=scalars_old_out,
            scalar_tend=scalar_tend_out,
            rho_zz_old=rho_zz_old,
            dt=dt,
            read_start=int(solve_cell_slice.start),
            read_stop=int(solve_cell_slice.stop),
            write_start=int(py_cellSolveStart),
            write_end=int(py_cellSolveEnd),
        )

    exchanger = current_dynamics_halo_exchange()
    if exchanger is not None:
        old_halos = {"scalars_old": scalars_old_out}
        exchanger.exchange_group("scalars_old", old_halos)
        scalars_old_out = old_halos["scalars_old"]

    # =========================================================================
    # PART 2: Set rho_zz_int
    # When advance_density=True (Fortran default for split dynamics/transport),
    # compute rho_zz_int from horizontal and vertical flux divergence.
    # When advance_density=False, set rho_zz_int = rho_zz_new directly.
    # =========================================================================
    with _monotonic_scalar_transport_profile(
        "mpas.monotonic_scalar_transport.density_update"
    ):
        if _precomputed_rho_zz_int is not None:
            rho_zz_int_out = _replace_row_slice(
                rho_zz_int_out,
                py_cellSolveStart,
                py_cellSolveEnd,
                _precomputed_rho_zz_int[solve_cell_slice, :],
            )
        elif advance_density:
            density_update_region = _maybe_compile_monotonic_scalar_transport_region(
                "monotonic_scalar_transport_density_update_region",
                _monotonic_scalar_transport_density_update_numeric,
            )
            rho_zz_int_out = density_update_region(
                uhAvg=uhAvg,
                cell_edge_relation=density_cell_edge_relation,
                rdnw=rdnw,
                wwAvg=wwAvg,
                rho_zz_old=rho_zz_old,
                rho_zz_int=rho_zz_int_out,
                dt=dt,
                read_start=int(solve_cell_slice.start),
                read_stop=int(solve_cell_slice.stop),
                write_start=int(py_cellSolveStart),
                write_end=int(py_cellSolveEnd),
                n_vert_levels=int(nVertLevels),
            )
        else:
            rho_zz_int_out = _replace_row_slice(
                rho_zz_int_out,
                py_cellSolveStart,
                py_cellSolveEnd,
                rho_zz_new[solve_cell_slice, :],
            )

    # =========================================================================
    # PART 3: Vectorized over scalar dimension
    # =========================================================================
    with _monotonic_scalar_transport_profile(
        "mpas.monotonic_scalar_transport.vertical_and_neighbor_bounds"
    ):
        padding_row = int(conn["padding_row"])
        # Step 3a
        scalar_old_all = _monotonic_scalar_transport_stencil_field_with_zero_padding(
            scalars_old_out, padding_row
        )
        scalar_new_all = _monotonic_scalar_transport_stencil_field_with_zero_padding(
            scalars_new_out, padding_row
        )

        # Step 3b: vertical/local and neighbor bounds plus the vertical flux,
        # one pure numeric region (compiled when selected).
        cell_slice = solve_cell_slice
        bounds_region = _maybe_compile_monotonic_scalar_transport_region(
            "monotonic_scalar_transport_vertical_neighbor_bounds_region",
            _monotonic_scalar_transport_vertical_neighbor_bounds_numeric,
        )
        s_max_patch, s_min_patch, wdtn_patch = bounds_region(
            scalar_old_all=scalar_old_all,
            scalar_new_all=scalar_new_all,
            ww=wwAvg[solve_cell_slice, :],
            fnm=fnm,
            fnp=fnp,
            neighbor_relation=neighbor_cell_relation,
            flux_divisor=_monotonic_scalar_transport_flux_divisor(scalar_old_all),
            cell_start=int(solve_cell_slice.start),
            cell_stop=int(solve_cell_slice.stop),
            n_vert_levels=int(nVertLevels),
            coef_3rd_order=float(coef_3rd_order),
            prefer_low_memory=prefer_low_memory_reductions,
            reduction_slot_tile_width=reduction_slot_tile_width,
        )

    edge_dv_patch = edge_dv if isinstance(edge_dv, torch.Tensor) else dvEdge[edge_slice]
    solve_rel_start = py_cellSolveStart - py_cellStart
    solve_rel_end = solve_rel_start + n_solve_cells
    streaming_scratch: (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None
    ) = None

    def _streaming_transport_tile_variant(return_scratch: bool, store: bool = False):
        if _numeric_plan is not None:
            plan = (
                _numeric_plan.scratch_tile_plan
                if return_scratch
                else _numeric_plan.no_scratch_tile_plan
            )
        else:
            plan = _monotonic_scalar_transport_streaming_tile_plan(
                one=one,
                pattern_c_operator=pattern_c_operator,
                edge_cell_relation=edge_cell_relation,
                cell_edge_relation=cell_edge_reduce_relation,
                bdy_mask=conn["bdy_mask"],
                solve_rel_start=solve_rel_start,
                solve_rel_end=solve_rel_end,
                row_start=py_cellSolveStart,
                row_end=py_cellSolveEnd,
                dt=dt,
                eps=eps,
                tau=tau,
                edge_interior_all=edge_interior_all,
                edge_active_all=edge_active_all,
                bdy_mask_all=conn["bdy_mask_all"],
                prefer_low_memory_reductions=prefer_low_memory_reductions,
                reduction_slot_tile_width=reduction_slot_tile_width,
                return_scratch=return_scratch,
            )
        if exchanger is not None:
            if (
                py_cellSolveStart != 0
                or py_cellSolveEnd != exchanger.partition.cells.owned_count
            ):
                raise ValueError(
                    "distributed limiter requires the owned-first cell patch"
                )
            phase_key = (
                "distributed_scale_seam",
                plan.static_key,
                exchanger.partition.cells.local_count,
            )
            prepare = _maybe_compile_monotonic_scalar_transport_region(
                "monotonic_scalar_transport_streaming_scale_prepare_region",
                plan.prepare_callable,
                specialization_key=phase_key,
            )
            update = _maybe_compile_monotonic_scalar_transport_region(
                "monotonic_scalar_transport_streaming_limited_update_region",
                plan.update_callable,
                specialization_key=phase_key,
            )

            def distributed_tile(**dynamic_tensors):
                prepared = prepare(**dynamic_tensors)
                scales = exchanger.exchange_cell_patches(*prepared[2:4])
                return update(
                    prepared=(*prepared[:2], *scales, *prepared[4:]),
                    scale_row_end=exchanger.partition.cells.local_count,
                    **dynamic_tensors,
                )

            return distributed_tile
        if store:
            return _maybe_compile_monotonic_scalar_transport_region(
                "monotonic_scalar_transport_streaming_scalar_transport_store_region",
                plan.store_callable,
                specialization_key=(
                    *streaming_specialization_key(return_scratch),
                    "store",
                ),
            )
        return _maybe_compile_monotonic_scalar_transport_region(
            "monotonic_scalar_transport_streaming_scalar_transport_region",
            plan.eager_callable,
            specialization_key=streaming_specialization_key(return_scratch),
        )

    def streaming_specialization_key(return_scratch: bool) -> tuple[object, ...]:
        return (
            "nVertLevels",
            nVertLevels,
            "chunk_scalars",
            num_scalars,
            "tile_scalars",
            streaming_scalar_tile_width,
            "cell_count",
            py_cellSolveEnd - py_cellSolveStart,
            "all_cell_count",
            py_cellEnd - py_cellStart,
            "edge_count",
            py_edgeEnd - py_edgeStart,
            "edge_interior_all",
            edge_interior_all,
            "edge_active_all",
            bool(edge_active_all),
            "bdy_mask_all",
            conn["bdy_mask_all"],
            "cell_edge_slots",
            cell_edge_reduce_relation.indices.shape[1],
            "pattern_c_operator",
            pattern_c_signature,
            "reduction_slot_tile_width",
            0 if reduction_slot_tile_width is None else int(reduction_slot_tile_width),
            "scalar_tile_width",
            0
            if streaming_scalar_tile_width is None
            else int(streaming_scalar_tile_width),
            "return_scratch",
            return_scratch,
        )

    # Compiled single-tile forward of the unchunked transport
    # (_store_scalars_new_in_region, set by atm_advance_scalars_mono_work
    # outside autograd): the region also stores the new-scalar rows into a new
    # full field (scalars_new_out is only read), replacing the patch copy
    # below. scalars_new_out is step-owned (see the caller's holder check in
    # time_integration). Scalar chunks keep their in-place slice writes.
    store_in_region = (
        _store_scalars_new_in_region
        and "monotonic_scalar_transport_streaming_scalar_transport_region"
        in compile_region_names
        and exchanger is None
        and not _pack_scratch_outputs
        and not _return_compact_scalars_new_patch
        and streaming_scalar_tile_width >= num_scalars
    )
    streaming_transport_tile_region = (
        _monotonic_scalar_transport_streaming_scalar_transport_tile_region
    )
    streaming_transport_no_scratch_tile_region = (
        _monotonic_scalar_transport_streaming_scalar_transport_tile_region
    )
    if _pack_scratch_outputs:
        streaming_transport_tile_region = _streaming_transport_tile_variant(True)
    if not store_in_region and (
        not _pack_scratch_outputs or streaming_scalar_tile_width < num_scalars
    ):
        streaming_transport_no_scratch_tile_region = _streaming_transport_tile_variant(
            False
        )
    if store_in_region:
        store_tile_tensors = {
            "scalar_field": scalar_new_all,
            "uh_avg": uhAvg,
            "scalar_old_rows": scalar_old_all[cell_slice, :, :],
            "scalar_old_all": scalar_old_all,
            "rho_zz_old_rows": rho_zz_old[cell_slice, :],
            "ww_avg_rows": wwAvg[cell_slice, :],
            "uh_avg_edges": uhAvg[edge_slice, :],
            "dv_edge_patch": edge_dv_patch,
            "wdtn_patch": wdtn_patch,
            "s_max_patch": s_max_patch,
            "s_min_patch": s_min_patch,
            "rho_zz_int_rows": rho_zz_int_out[cell_slice, :],
            "scalar_new_all_patch": scalar_new_all[all_cell_slice, :, :],
            "scalars_new_base_patch": scalars_new_out[all_cell_slice, :, :],
            "rdnw": rdnw,
        }
        with _monotonic_scalar_transport_profile(
            "mpas.monotonic_scalar_transport.streaming_scalar_transport"
        ):
            (
                scalar_stage2,
                scalars_new_out,
                *streaming_scratch,
            ) = _streaming_transport_tile_variant(False, store=True)(
                **store_tile_tensors,
                scalars_new_base=scalars_new_out,
                store_start=py_cellStart,
                store_end=py_cellEnd,
            )
    else:
        with _monotonic_scalar_transport_profile(
            "mpas.monotonic_scalar_transport.streaming_scalar_transport"
        ):
            (
                scalar_stage2,
                scalars_new_patch,
                flux_limited_last,
                flux_upwind_last,
                flux_tmp_last,
                wdtn_last_rows,
                scale_arr_last_patch,
            ) = _monotonic_scalar_transport_streaming_scalar_transport_region(
                scalar_field=scalar_new_all,
                uh_avg=uhAvg,
                one=one,
                pattern_c_operator=pattern_c_operator,
                scalar_old_rows=scalar_old_all[cell_slice, :, :],
                scalar_old_all=scalar_old_all,
                rho_zz_old_rows=rho_zz_old[cell_slice, :],
                ww_avg_rows=wwAvg[cell_slice, :],
                uh_avg_edges=uhAvg[edge_slice, :],
                dv_edge_patch=edge_dv_patch,
                wdtn_patch=wdtn_patch,
                s_max_patch=s_max_patch,
                s_min_patch=s_min_patch,
                rho_zz_int_rows=rho_zz_int_out[cell_slice, :],
                scalar_new_all_patch=scalar_new_all[all_cell_slice, :, :],
                scalars_new_base_patch=scalars_new_out[all_cell_slice, :, :],
                rdnw=rdnw,
                edge_cell_relation=edge_cell_relation,
                cell_edge_relation=cell_edge_reduce_relation,
                bdy_mask=conn["bdy_mask"],
                solve_rel_start=solve_rel_start,
                solve_rel_end=solve_rel_end,
                row_start=py_cellSolveStart,
                row_end=py_cellSolveEnd,
                dt=dt,
                eps=eps,
                tau=tau,
                edge_interior_all=edge_interior_all,
                edge_active_all=edge_active_all,
                bdy_mask_all=conn["bdy_mask_all"],
                prefer_low_memory_reductions=prefer_low_memory_reductions,
                reduction_slot_tile_width=reduction_slot_tile_width,
                scalar_tile_width=streaming_scalar_tile_width,
                return_scratch=_pack_scratch_outputs,
                _tile_region=streaming_transport_tile_region,
                _no_scratch_tile_region=streaming_transport_no_scratch_tile_region,
                _tile_static_bound=True,
            )
            streaming_scratch = (
                flux_limited_last,
                flux_upwind_last,
                flux_tmp_last,
                wdtn_last_rows,
                scale_arr_last_patch,
            )
            scalars_new_out = _monotonic_scalar_transport_pack_or_return_scalars_new_patch(
                scalars_new_out=scalars_new_out,
                row_start=py_cellStart,
                row_end=py_cellEnd,
                scalars_new_patch=scalars_new_patch,
                return_compact_patch=_return_compact_scalars_new_patch,
            )

    with _monotonic_scalar_transport_profile(
        "mpas.monotonic_scalar_transport.output_pack"
    ):
        if _pack_scratch_outputs:
            # Fortran scratch arrays correspond to the final scalar iteration.
            last_scalar = num_scalars - 1
            if streaming_scratch is None:
                raise RuntimeError(
                    "monotonic-scalar-transport streaming transport did not return scratch"
                )
            (
                flux_limited_last,
                flux_upwind_last,
                flux_tmp_last,
                wdtn_last_rows,
                scale_arr_last_patch,
            ) = streaming_scratch
            scalar_old_out = scalar_old_all[:, :, last_scalar]
            scalar_new_out = _replace_row_slice(
                scalar_new_all[:, :, last_scalar].clone(),
                py_cellSolveStart,
                py_cellSolveEnd,
                scalar_stage2[:, :, last_scalar],
            )
            s_max_out = _replace_row_slice(
                _zero_scratch_for_output(s_max_out),
                py_cellSolveStart,
                py_cellSolveEnd,
                s_max_patch[:, :, last_scalar],
            )
            s_min_out = _replace_row_slice(
                _zero_scratch_for_output(s_min_out),
                py_cellSolveStart,
                py_cellSolveEnd,
                s_min_patch[:, :, last_scalar],
            )
            wdtn_out = _replace_row_slice(
                _zero_scratch_for_output(wdtn_out),
                py_cellSolveStart,
                py_cellSolveEnd,
                wdtn_last_rows,
            )
            flux_arr_out = _replace_row_slice(
                _zero_scratch_for_output(flux_arr_out),
                py_edgeStart,
                py_edgeEnd,
                flux_limited_last,
            )
            flux_upwind_tmp_out = _replace_row_slice(
                flux_upwind_tmp_out,
                py_edgeStart,
                py_edgeEnd,
                flux_upwind_last,
            )
            flux_tmp_out = _replace_row_slice(
                flux_tmp_out,
                py_edgeStart,
                py_edgeEnd,
                flux_tmp_last,
            )
            scale_arr_out = _replace_row_slice(
                _zero_scratch_for_output(scale_arr_out),
                py_cellSolveStart,
                py_cellSolveEnd,
                scale_arr_last_patch,
            )

    # Return all output arrays
    return {
        "scalars_new": scalars_new_out,
        "flux_arr": flux_arr_out,
        "scalars_old": scalars_old_out,
        "scalar_tend": scalar_tend_out,
        "scalar_old": scalar_old_out,
        "scalar_new": scalar_new_out,
        "s_max": s_max_out,
        "s_min": s_min_out,
        "wdtn": wdtn_out,
        "scale_arr": scale_arr_out,
        "flux_upwind_tmp": flux_upwind_tmp_out,
        "flux_tmp": flux_tmp_out,
        "rho_zz_int": rho_zz_int_out,
    }
