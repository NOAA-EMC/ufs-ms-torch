# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""PyTorch-native longwave RTE transport kernels.

These kernels operate on explicit optical/source-function tensors.  They do not
compute RRTMGP gas optics; that remains a separate upstream kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .compile import compile_categories_enabled, run_compile_island


@dataclass(frozen=True)
class LongwaveFluxResult:
    """Broadband longwave fluxes summed over g-points."""

    flux_up: torch.Tensor
    flux_down: torch.Tensor
    runtime_kind: str = "torch_rrtmgp_longwave_transport"
    flux_up_jac: torch.Tensor | None = None


def _real(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64 (the working dtype), got {value.dtype}")
    return value


def _tau_thresh(dtype: torch.dtype) -> float:
    return float(torch.finfo(dtype).eps ** 0.5)


def _exp_neg(value: torch.Tensor) -> torch.Tensor:
    return torch.exp(-value)


def _as_gpt_lay_col(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = _real(value, name=name)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [ngpt,nlay,ncol], got {tuple(tensor.shape)}")
    return tensor


def _as_gpt_lev_col(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = _real(value, name=name)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [ngpt,nlev,ncol], got {tuple(tensor.shape)}")
    return tensor


def _as_gpt_lay_col_or_from_interfaces(
    lev_source: torch.Tensor,
    lev_source_dec: torch.Tensor | None,
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if lev_source_dec is not None:
        inc = _as_gpt_lay_col(lev_source, name=name)
        dec = _as_gpt_lay_col(lev_source_dec, name="lev_source_dec")
        if dec.shape != inc.shape:
            raise ValueError("lev_source_inc and lev_source_dec shapes are inconsistent")
        return inc, dec
    lev = _as_gpt_lev_col(lev_source, name=name)
    return lev[:, 1:, :], lev[:, :-1, :]


def _as_gpt_col(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = _real(value, name=name)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [ngpt,ncol], got {tuple(tensor.shape)}")
    return tensor


def _gpt_to_band(gpt_to_band: torch.Tensor, *, ngpt: int, nband: int, device: torch.device) -> torch.Tensor:
    if not isinstance(gpt_to_band, torch.Tensor):
        raise TypeError("gpt_to_band must be a torch tensor")
    raw_index = gpt_to_band.to(dtype=torch.int64).reshape(-1)
    index = raw_index.to(device=device)
    if index.numel() != ngpt:
        raise ValueError(f"gpt_to_band must have {ngpt} entries, got {index.numel()}")
    torch._assert_async(torch.all((index >= 0) & (index < nband)),
                        "gpt_to_band entries must be zero-based band indices")
    return index


def _sum_gpoints_fortran_order_tensor(values: torch.Tensor) -> torch.Tensor:
    """Flat tensor kernel with an explicit, dependency-ordered spectral sum."""

    total = torch.zeros_like(values[0])
    for igpt in range(values.shape[0]):
        total = total + values[igpt]
    return total


def _sum_gpoints_fortran_order(values: torch.Tensor) -> torch.Tensor:
    """Accumulate spectral points in the historical scalar-loop order.

    A generic CUDA ``sum`` is a tree reduction whose association changes with
    launch shape.  That error is visible in clear-sky heating rates because the
    post kernel differences adjacent broadband fluxes.  Forward compiled runs
    therefore lower the explicit dependency chain as one fixed-shape island;
    eager and differentiable runs retain the same source-order chain directly.
    """

    if "longwave_transport" in compile_categories_enabled():
        return run_compile_island(
            "longwave_transport",
            "sum_gpoints_fortran_order",
            _sum_gpoints_fortran_order_tensor,
            values,
        )
    return _sum_gpoints_fortran_order_tensor(values)




def _needs_autograd_stack(*values: torch.Tensor | None) -> bool:
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    return torch.is_grad_enabled() and any(value.requires_grad for value in tensors)


_LONGWAVE_TRANSPORT_SLAB_LEVELS = 32
_LONGWAVE_CUDA_EAGER_TRANSPORT_SLAB_LEVELS = 64
_LONGWAVE_CUDA_COMPILED_TRANSPORT_SLAB_LEVELS = 32


def _longwave_transport_slab_levels(value: torch.Tensor) -> int:
    """Select the frozen eager control or retained compiled LW32 candidate."""

    if value.device.type != "cuda":
        return _LONGWAVE_TRANSPORT_SLAB_LEVELS
    if "longwave_transport" in compile_categories_enabled():
        return _LONGWAVE_CUDA_COMPILED_TRANSPORT_SLAB_LEVELS
    return _LONGWAVE_CUDA_EAGER_TRANSPORT_SLAB_LEVELS


def _pad_longwave_layer_axis(
    value: torch.Tensor,
    target_levels: int,
    *,
    fill_value: float,
) -> torch.Tensor:
    """Pad a tail slab with a recurrence-neutral constant level."""

    missing = target_levels - value.shape[1]
    if missing < 0:
        raise ValueError("longwave layer padding cannot truncate an input")
    if missing == 0:
        return value
    padding = value.new_full((value.shape[0], missing, value.shape[2]), fill_value)
    return torch.cat((value, padding), dim=1)


def _trim_longwave_layer_chunks(chunks: list[torch.Tensor], nlay: int) -> torch.Tensor:
    """Join fixed-shape slabs and remove any neutral tail levels."""

    return torch.cat(chunks, dim=1)[:, :nlay, :]


def _longwave_noscat_down_slab(
    tau_scaled: torch.Tensor,
    transmittance: torch.Tensor,
    lay_source: torch.Tensor,
    lev_source_down: torch.Tensor,
    intensity_in: torch.Tensor,
    *,
    reverse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one no-scatter downward slab in source recurrence order."""

    nlay = tau_scaled.shape[1]
    intensity_to_flux = torch.as_tensor(torch.pi, dtype=tau_scaled.dtype, device=tau_scaled.device)
    intensity = intensity_in
    flux_values: list[torch.Tensor] = [intensity_in * 0.0] * nlay
    layer_order = range(nlay - 1, -1, -1) if reverse else range(nlay)
    for ilay in layer_order:
        tau_loc = tau_scaled[:, ilay, :]
        trans = transmittance[:, ilay, :]
        intensity = trans * intensity + lw_noscat_source_down(
            lev_source_down[:, ilay, :],
            lay_source[:, ilay, :],
            tau_loc,
            trans,
        )
        flux_values[ilay] = intensity * intensity_to_flux
    return intensity, torch.stack(flux_values, dim=1)


def _longwave_noscat_up_slab(
    tau_scaled: torch.Tensor,
    transmittance: torch.Tensor,
    lay_source: torch.Tensor,
    lev_source_up: torch.Tensor,
    intensity_in: torch.Tensor,
    jacobian_in: torch.Tensor,
    *,
    reverse: bool,
    has_jacobian: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance one no-scatter upward slab, including the optional Jacobian carry."""

    nlay = tau_scaled.shape[1]
    intensity_to_flux = torch.as_tensor(torch.pi, dtype=tau_scaled.dtype, device=tau_scaled.device)
    intensity = intensity_in
    jacobian = jacobian_in
    flux_values: list[torch.Tensor] = [intensity_in * 0.0] * nlay
    jacobian_values: list[torch.Tensor] = [jacobian_in * 0.0] * nlay
    layer_order = range(nlay - 1, -1, -1) if reverse else range(nlay)
    for ilay in layer_order:
        tau_loc = tau_scaled[:, ilay, :]
        trans = transmittance[:, ilay, :]
        intensity = trans * intensity + lw_noscat_source_up(
            lev_source_up[:, ilay, :],
            lay_source[:, ilay, :],
            tau_loc,
            trans,
        )
        flux_values[ilay] = intensity * intensity_to_flux
        if has_jacobian:
            jacobian = trans * jacobian
            jacobian_values[ilay] = jacobian * intensity_to_flux
    return intensity, jacobian, torch.stack(flux_values, dim=1), torch.stack(jacobian_values, dim=1)


def _longwave_noscat_down_autograd_slab(
    tau: torch.Tensor,
    lay_source: torch.Tensor,
    lev_source_down: torch.Tensor,
    diffusivity: torch.Tensor,
    intensity_in: torch.Tensor,
    *,
    reverse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradient-active down sweep with the historical independent exp graph."""

    nlay = tau.shape[1]
    intensity_to_flux = torch.as_tensor(torch.pi, dtype=tau.dtype, device=tau.device)
    intensity = intensity_in
    flux_values: list[torch.Tensor] = [intensity_in * 0.0] * nlay
    layer_order = range(nlay - 1, -1, -1) if reverse else range(nlay)
    for ilay in layer_order:
        tau_loc = tau[:, ilay, :] * diffusivity
        trans = _exp_neg(tau_loc)
        intensity = trans * intensity + lw_noscat_source_down(
            lev_source_down[:, ilay, :], lay_source[:, ilay, :], tau_loc, trans
        )
        flux_values[ilay] = intensity * intensity_to_flux
    return intensity, torch.stack(flux_values, dim=1)


def _longwave_noscat_up_autograd_slab(
    tau: torch.Tensor,
    lay_source: torch.Tensor,
    lev_source_up: torch.Tensor,
    diffusivity: torch.Tensor,
    intensity_in: torch.Tensor,
    jacobian_in: torch.Tensor,
    *,
    reverse: bool,
    has_jacobian: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradient-active up sweep with the historical independent exp graph."""

    nlay = tau.shape[1]
    intensity_to_flux = torch.as_tensor(torch.pi, dtype=tau.dtype, device=tau.device)
    intensity = intensity_in
    jacobian = jacobian_in
    flux_values: list[torch.Tensor] = [intensity_in * 0.0] * nlay
    jacobian_values: list[torch.Tensor] = [jacobian_in * 0.0] * nlay
    layer_order = range(nlay - 1, -1, -1) if reverse else range(nlay)
    for ilay in layer_order:
        tau_loc = tau[:, ilay, :] * diffusivity
        trans = _exp_neg(tau_loc)
        intensity = trans * intensity + lw_noscat_source_up(
            lev_source_up[:, ilay, :], lay_source[:, ilay, :], tau_loc, trans
        )
        flux_values[ilay] = intensity * intensity_to_flux
        if has_jacobian:
            jacobian = trans * jacobian
            jacobian_values[ilay] = jacobian * intensity_to_flux
    return intensity, jacobian, torch.stack(flux_values, dim=1), torch.stack(jacobian_values, dim=1)


def _solve_longwave_noscat_slabbed(
    tau_t: torch.Tensor,
    lay_t: torch.Tensor,
    lev_inc_t: torch.Tensor,
    lev_dec_t: torch.Tensor,
    sfc_t: torch.Tensor,
    sfc_jac_t: torch.Tensor | None,
    emiss: torch.Tensor,
    band: torch.Tensor,
    ds: torch.Tensor,
    *,
    top_at_1: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run no-scatter LW transport through bounded, reusable compile slabs."""

    ngpt, nlay, ncol = tau_t.shape
    intensity_to_flux = torch.as_tensor(torch.pi, dtype=tau_t.dtype, device=tau_t.device)
    lev_up_t = lev_dec_t if top_at_1 else lev_inc_t
    lev_dn_t = lev_inc_t if top_at_1 else lev_dec_t
    sfc_emis = emiss.index_select(0, band)
    zero = torch.zeros((ngpt, ncol), dtype=tau_t.dtype, device=tau_t.device)
    slab_levels = _longwave_transport_slab_levels(tau_t)
    starts = list(range(0, nlay, slab_levels))
    gradient_active = _needs_autograd_stack(tau_t, lay_t, lev_inc_t, lev_dec_t, sfc_t, sfc_jac_t, emiss, ds)
    padded_nlay = starts[-1] + slab_levels
    lay_slab = _pad_longwave_layer_axis(lay_t, padded_nlay, fill_value=0.0)
    lev_up_slab = _pad_longwave_layer_axis(lev_up_t, padded_nlay, fill_value=0.0)
    lev_dn_slab = _pad_longwave_layer_axis(lev_dn_t, padded_nlay, fill_value=0.0)
    if gradient_active:
        tau_slab = _pad_longwave_layer_axis(tau_t, padded_nlay, fill_value=0.0)
    else:
        tau_scaled = tau_t * ds.unsqueeze(1)
        transmittance = _exp_neg(tau_scaled)
        tau_scaled_slab = _pad_longwave_layer_axis(tau_scaled, padded_nlay, fill_value=0.0)
        transmittance_slab = _pad_longwave_layer_axis(
            transmittance, padded_nlay, fill_value=1.0
        )

    down_current = zero
    down_traversal = starts if top_at_1 else list(reversed(starts))
    down_outputs: dict[int, torch.Tensor] = {}
    for start in down_traversal:
        stop = start + slab_levels
        if gradient_active:
            down_current, down_flux = run_compile_island(
                "longwave_transport",
                "longwave_noscat_down_autograd_slab",
                _longwave_noscat_down_autograd_slab,
                tau_slab[:, start:stop, :],
                lay_slab[:, start:stop, :],
                lev_dn_slab[:, start:stop, :],
                ds,
                down_current,
                reverse=not top_at_1,
            )
        else:
            down_current, down_flux = run_compile_island(
                "longwave_transport",
                "longwave_noscat_down_slab",
                _longwave_noscat_down_slab,
                tau_scaled_slab[:, start:stop, :],
                transmittance_slab[:, start:stop, :],
                lay_slab[:, start:stop, :],
                lev_dn_slab[:, start:stop, :],
                down_current,
                reverse=not top_at_1,
            )
        down_outputs[start] = down_flux
    down_chunks = [down_outputs[start] for start in starts]
    down_layers = _trim_longwave_layer_chunks(down_chunks, nlay)
    down_gpt = (
        torch.cat([zero.unsqueeze(1), down_layers], dim=1)
        if top_at_1
        else torch.cat([down_layers, zero.unsqueeze(1)], dim=1)
    )
    del down_outputs, down_chunks

    sfc_index = nlay if top_at_1 else 0
    up_surface = down_gpt[:, sfc_index, :] / intensity_to_flux * (1.0 - sfc_emis) + sfc_emis * sfc_t
    has_jacobian = sfc_jac_t is not None
    jacobian_surface = sfc_emis * sfc_jac_t if sfc_jac_t is not None else zero
    up_current = up_surface
    jacobian_current = jacobian_surface
    up_traversal = list(reversed(starts)) if top_at_1 else starts
    up_outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for start in up_traversal:
        stop = start + slab_levels
        if gradient_active:
            up_current, jacobian_current, up_flux, jacobian_flux = run_compile_island(
                "longwave_transport",
                "longwave_noscat_up_autograd_slab",
                _longwave_noscat_up_autograd_slab,
                tau_slab[:, start:stop, :],
                lay_slab[:, start:stop, :],
                lev_up_slab[:, start:stop, :],
                ds,
                up_current,
                jacobian_current,
                reverse=top_at_1,
                has_jacobian=has_jacobian,
            )
        else:
            up_current, jacobian_current, up_flux, jacobian_flux = run_compile_island(
                "longwave_transport",
                "longwave_noscat_up_slab",
                _longwave_noscat_up_slab,
                tau_scaled_slab[:, start:stop, :],
                transmittance_slab[:, start:stop, :],
                lay_slab[:, start:stop, :],
                lev_up_slab[:, start:stop, :],
                up_current,
                jacobian_current,
                reverse=top_at_1,
                has_jacobian=has_jacobian,
            )
        up_outputs[start] = (up_flux, jacobian_flux)
    ordered_up = [up_outputs[start] for start in starts]
    up_chunks = [value[0] for value in ordered_up]
    up_layers = _trim_longwave_layer_chunks(up_chunks, nlay)
    up_surface_flux = up_surface * intensity_to_flux
    up_gpt = (
        torch.cat([up_layers, up_surface_flux.unsqueeze(1)], dim=1)
        if top_at_1
        else torch.cat([up_surface_flux.unsqueeze(1), up_layers], dim=1)
    )

    broadband_up_jac = None
    if has_jacobian:
        jacobian_chunks = [value[1] for value in ordered_up]
        jacobian_layers = _trim_longwave_layer_chunks(jacobian_chunks, nlay)
        jacobian_surface_flux = jacobian_surface * intensity_to_flux
        jacobian_gpt = (
            torch.cat([jacobian_layers, jacobian_surface_flux.unsqueeze(1)], dim=1)
            if top_at_1
            else torch.cat([jacobian_surface_flux.unsqueeze(1), jacobian_layers], dim=1)
        )
        broadband_up_jac = _sum_gpoints_fortran_order(jacobian_gpt)
    del up_outputs, ordered_up, up_chunks
    return (
        _sum_gpoints_fortran_order(up_gpt),
        _sum_gpoints_fortran_order(down_gpt),
        broadband_up_jac,
    )


def lw_noscat_source_up(
    lev_source_inc: torch.Tensor,
    lay_source: torch.Tensor,
    tau_loc: torch.Tensor,
    trans: torch.Tensor,
) -> torch.Tensor:
    """Longwave no-scattering upward source for a linear-in-tau layer source."""

    threshold = _tau_thresh(tau_loc.dtype)
    safe_tau = torch.where(tau_loc > 0.0, tau_loc, torch.ones_like(tau_loc))
    fact_large = (1.0 - trans) / safe_tau - trans
    fact_small = tau_loc * (0.5 - tau_loc / 3.0)
    fact = torch.where(tau_loc > threshold, fact_large, fact_small)
    return (1.0 - trans) * lev_source_inc + 2.0 * fact * (lay_source - lev_source_inc)


def lw_noscat_source_down(
    lev_source_dec: torch.Tensor,
    lay_source: torch.Tensor,
    tau_loc: torch.Tensor,
    trans: torch.Tensor,
) -> torch.Tensor:
    """Longwave no-scattering downward source for a linear-in-tau layer source."""

    threshold = _tau_thresh(tau_loc.dtype)
    safe_tau = torch.where(tau_loc > 0.0, tau_loc, torch.ones_like(tau_loc))
    fact_large = (1.0 - trans) / safe_tau - trans
    fact_small = tau_loc * (0.5 - tau_loc / 3.0)
    fact = torch.where(tau_loc > threshold, fact_large, fact_small)
    return (1.0 - trans) * lev_source_dec + 2.0 * fact * (lay_source - lev_source_dec)




def _solve_longwave_noscat_tensor(
    tau_t: torch.Tensor,
    lay_t: torch.Tensor,
    lev_inc_t: torch.Tensor,
    lev_dec_t: torch.Tensor,
    sfc_t: torch.Tensor,
    sfc_jac_t: torch.Tensor | None,
    emiss: torch.Tensor,
    band: torch.Tensor,
    ds: torch.Tensor,
    *,
    top_at_1: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    ngpt, nlay, ncol = tau_t.shape
    nlev = nlay + 1
    intensity_to_flux = torch.as_tensor(torch.pi, dtype=tau_t.dtype, device=tau_t.device)
    lev_up_t = lev_dec_t if top_at_1 else lev_inc_t
    lev_dn_t = lev_inc_t if top_at_1 else lev_dec_t
    sfc_emis = emiss.index_select(0, band)
    zero_gpt_col = torch.zeros((ngpt, ncol), dtype=tau_t.dtype, device=tau_t.device)
    use_stack = _needs_autograd_stack(tau_t, lay_t, lev_inc_t, lev_dec_t, sfc_t, sfc_jac_t, emiss, ds)
    if not use_stack:
        tau_scaled = tau_t * ds.unsqueeze(1)
        trans_layers = _exp_neg(tau_scaled)
    if use_stack:
        # One unbind per directional pass lets native autograd assemble the
        # disjoint layer adjoints once. Independent [:, level, :] chains
        # otherwise allocate a full spectral/level/column gradient per use.
        # Keep the two passes separate, including their original exp/order.
        tau_dn_layers = tau_t.unbind(dim=1)
        tau_up_layers = tau_t.unbind(dim=1)
        lay_dn_layers = lay_t.unbind(dim=1)
        lay_up_layers = lay_t.unbind(dim=1)
        lev_dn_layers = lev_dn_t.unbind(dim=1)
        lev_up_layers = lev_up_t.unbind(dim=1)
        zero = zero_gpt_col
        if top_at_1:
            down_values: list[torch.Tensor] = [zero for _ in range(nlev)]
            down_values[0] = zero
            intensity_dn = zero
            for ilay in range(nlay):
                tau_loc = tau_dn_layers[ilay] * ds
                trans = _exp_neg(tau_loc)
                intensity_dn = trans * intensity_dn + lw_noscat_source_down(
                    lev_dn_layers[ilay],
                    lay_dn_layers[ilay],
                    tau_loc,
                    trans,
                )
                down_values[ilay + 1] = intensity_dn * intensity_to_flux
            sfc_index = nlay
            intensity_up = down_values[sfc_index] / intensity_to_flux * (1.0 - sfc_emis) + sfc_emis * sfc_t
            up_values: list[torch.Tensor] = [zero for _ in range(nlev)]
            up_jac_values: list[torch.Tensor] | None = [zero for _ in range(nlev)] if sfc_jac_t is not None else None
            up_values[sfc_index] = intensity_up * intensity_to_flux
            intensity_up_jac = sfc_emis * sfc_jac_t if sfc_jac_t is not None else None
            if intensity_up_jac is not None and up_jac_values is not None:
                up_jac_values[sfc_index] = intensity_up_jac * intensity_to_flux
            for ilay in range(nlay - 1, -1, -1):
                tau_loc = tau_up_layers[ilay] * ds
                trans = _exp_neg(tau_loc)
                intensity_up = trans * intensity_up + lw_noscat_source_up(
                    lev_up_layers[ilay],
                    lay_up_layers[ilay],
                    tau_loc,
                    trans,
                )
                up_values[ilay] = intensity_up * intensity_to_flux
                if intensity_up_jac is not None and up_jac_values is not None:
                    intensity_up_jac = trans * intensity_up_jac
                    up_jac_values[ilay] = intensity_up_jac * intensity_to_flux
        else:
            down_values = [zero for _ in range(nlev)]
            down_values[nlay] = zero
            intensity_dn = zero
            for ilay in range(nlay - 1, -1, -1):
                tau_loc = tau_dn_layers[ilay] * ds
                trans = _exp_neg(tau_loc)
                intensity_dn = trans * intensity_dn + lw_noscat_source_down(
                    lev_dn_layers[ilay],
                    lay_dn_layers[ilay],
                    tau_loc,
                    trans,
                )
                down_values[ilay] = intensity_dn * intensity_to_flux
            sfc_index = 0
            intensity_up = down_values[sfc_index] / intensity_to_flux * (1.0 - sfc_emis) + sfc_emis * sfc_t
            up_values = [zero for _ in range(nlev)]
            up_jac_values = [zero for _ in range(nlev)] if sfc_jac_t is not None else None
            up_values[sfc_index] = intensity_up * intensity_to_flux
            intensity_up_jac = sfc_emis * sfc_jac_t if sfc_jac_t is not None else None
            if intensity_up_jac is not None and up_jac_values is not None:
                up_jac_values[sfc_index] = intensity_up_jac * intensity_to_flux
            for ilay in range(nlay):
                tau_loc = tau_up_layers[ilay] * ds
                trans = _exp_neg(tau_loc)
                intensity_up = trans * intensity_up + lw_noscat_source_up(
                    lev_up_layers[ilay],
                    lay_up_layers[ilay],
                    tau_loc,
                    trans,
                )
                up_values[ilay + 1] = intensity_up * intensity_to_flux
                if intensity_up_jac is not None and up_jac_values is not None:
                    intensity_up_jac = trans * intensity_up_jac
                    up_jac_values[ilay + 1] = intensity_up_jac * intensity_to_flux
        down_gpt_stack = torch.stack(down_values, dim=1)
        up_gpt_stack = torch.stack(up_values, dim=1)
        up_jac_gpt_stack = torch.stack(up_jac_values, dim=1) if up_jac_values is not None else None
        broadband_up = _sum_gpoints_fortran_order(up_gpt_stack)
        broadband_down = _sum_gpoints_fortran_order(down_gpt_stack)
        broadband_up_jac = _sum_gpoints_fortran_order(up_jac_gpt_stack) if up_jac_gpt_stack is not None else None
        return broadband_up, broadband_down, broadband_up_jac

    down_gpt = torch.empty((ngpt, nlev, ncol), dtype=tau_t.dtype, device=tau_t.device)
    up_gpt = torch.empty_like(down_gpt)
    up_jac_gpt = torch.empty_like(down_gpt) if sfc_jac_t is not None else None

    if top_at_1:
        down_gpt[:, 0, :] = zero_gpt_col
        intensity_dn = zero_gpt_col
        for ilay in range(nlay):
            tau_loc = tau_scaled[:, ilay, :]
            trans = trans_layers[:, ilay, :]
            intensity_dn = trans * intensity_dn + lw_noscat_source_down(
                lev_dn_t[:, ilay, :],
                lay_t[:, ilay, :],
                tau_loc,
                trans,
            )
            down_gpt[:, ilay + 1, :] = intensity_dn * intensity_to_flux
        sfc_index = nlay
        intensity_up = down_gpt[:, sfc_index, :] / intensity_to_flux * (1.0 - sfc_emis) + sfc_emis * sfc_t
        up_gpt[:, sfc_index, :] = intensity_up * intensity_to_flux
        intensity_up_jac = sfc_emis * sfc_jac_t if sfc_jac_t is not None else None
        if intensity_up_jac is not None:
            up_jac_gpt[:, sfc_index, :] = intensity_up_jac * intensity_to_flux
        for ilay in range(nlay - 1, -1, -1):
            tau_loc = tau_scaled[:, ilay, :]
            trans = trans_layers[:, ilay, :]
            intensity_up = trans * intensity_up + lw_noscat_source_up(
                lev_up_t[:, ilay, :],
                lay_t[:, ilay, :],
                tau_loc,
                trans,
            )
            up_gpt[:, ilay, :] = intensity_up * intensity_to_flux
            if intensity_up_jac is not None:
                intensity_up_jac = trans * intensity_up_jac
                up_jac_gpt[:, ilay, :] = intensity_up_jac * intensity_to_flux
    else:
        down_gpt[:, nlay, :] = zero_gpt_col
        intensity_dn = zero_gpt_col
        for ilay in range(nlay - 1, -1, -1):
            tau_loc = tau_scaled[:, ilay, :]
            trans = trans_layers[:, ilay, :]
            intensity_dn = trans * intensity_dn + lw_noscat_source_down(
                lev_dn_t[:, ilay, :],
                lay_t[:, ilay, :],
                tau_loc,
                trans,
            )
            down_gpt[:, ilay, :] = intensity_dn * intensity_to_flux
        sfc_index = 0
        intensity_up = down_gpt[:, sfc_index, :] / intensity_to_flux * (1.0 - sfc_emis) + sfc_emis * sfc_t
        up_gpt[:, sfc_index, :] = intensity_up * intensity_to_flux
        intensity_up_jac = sfc_emis * sfc_jac_t if sfc_jac_t is not None else None
        if intensity_up_jac is not None:
            up_jac_gpt[:, sfc_index, :] = intensity_up_jac * intensity_to_flux
        for ilay in range(nlay):
            tau_loc = tau_scaled[:, ilay, :]
            trans = trans_layers[:, ilay, :]
            intensity_up = trans * intensity_up + lw_noscat_source_up(
                lev_up_t[:, ilay, :],
                lay_t[:, ilay, :],
                tau_loc,
                trans,
            )
            up_gpt[:, ilay + 1, :] = intensity_up * intensity_to_flux
            if intensity_up_jac is not None:
                intensity_up_jac = trans * intensity_up_jac
                up_jac_gpt[:, ilay + 1, :] = intensity_up_jac * intensity_to_flux

    broadband_up = _sum_gpoints_fortran_order(up_gpt)
    broadband_down = _sum_gpoints_fortran_order(down_gpt)
    broadband_up_jac = None
    if up_jac_gpt is not None:
        broadband_up_jac = _sum_gpoints_fortran_order(up_jac_gpt)
    return broadband_up, broadband_down, broadband_up_jac


def solve_longwave_noscat(
    tau: torch.Tensor,
    lay_source: torch.Tensor,
    lev_source: torch.Tensor,
    sfc_source: torch.Tensor,
    sfc_emiss_byband: torch.Tensor,
    gpt_to_band: torch.Tensor,
    *,
    diffusivity_angle: torch.Tensor | float | None = None,
    lev_source_dec: torch.Tensor | None = None,
    top_at_1: bool = False,
    sfc_source_jac: torch.Tensor | None = None,
) -> LongwaveFluxResult:
    """Solve broadband non-scattering LW transport from explicit source hooks.

    Shapes are `[ngpt,nlay,ncol]` for `tau` and `lay_source`,
    `[ngpt,nlay,ncol]` for directional `lev_source`/`lev_source_dec`,
    `[ngpt,ncol]` for `sfc_source`, and `[nband,ncol]` for
    `sfc_emiss_byband`.  For compatibility, omitting `lev_source_dec` allows
    `lev_source` to be supplied as `[ngpt,nlev,ncol]`.  The returned fluxes use
    the checkpoint convention `[nlev,ncol]`.
    """

    tau_t = _as_gpt_lay_col(tau, name="tau")
    lay_t = _as_gpt_lay_col(lay_source, name="lay_source")
    lev_inc_t, lev_dec_t = _as_gpt_lay_col_or_from_interfaces(
        lev_source,
        lev_source_dec,
        name="lev_source_inc" if lev_source_dec is not None else "lev_source",
    )
    sfc_t = _as_gpt_col(sfc_source, name="sfc_source")
    sfc_jac_t = _as_gpt_col(sfc_source_jac, name="sfc_source_jac") if sfc_source_jac is not None else None
    emiss = _real(sfc_emiss_byband, name="sfc_emiss_byband")
    if emiss.ndim != 2:
        raise ValueError(f"sfc_emiss_byband must have shape [nband,ncol], got {tuple(emiss.shape)}")
    ngpt, nlay, ncol = tau_t.shape
    if (
        lay_t.shape != tau_t.shape
        or lev_inc_t.shape != tau_t.shape
        or lev_dec_t.shape != tau_t.shape
        or sfc_t.shape != (ngpt, ncol)
        or (sfc_jac_t is not None and sfc_jac_t.shape != (ngpt, ncol))
    ):
        raise ValueError("longwave no-scattering source/optics shapes are inconsistent")
    band = _gpt_to_band(gpt_to_band, ngpt=ngpt, nband=emiss.shape[0], device=tau_t.device)
    if diffusivity_angle is None:
        ds = torch.ones((ngpt, ncol), dtype=tau_t.dtype, device=tau_t.device)
    else:
        ds = torch.as_tensor(diffusivity_angle, dtype=tau_t.dtype, device=tau_t.device)
        ds = ds.expand((ngpt, ncol)) if ds.ndim == 0 else ds
        if ds.shape != (ngpt, ncol):
            raise ValueError(f"diffusivity_angle must broadcast to [ngpt,ncol], got {tuple(ds.shape)}")
    if "longwave_transport" in compile_categories_enabled():
        broadband_up, broadband_down, broadband_up_jac = _solve_longwave_noscat_slabbed(
            tau_t,
            lay_t,
            lev_inc_t,
            lev_dec_t,
            sfc_t,
            sfc_jac_t,
            emiss,
            band,
            ds,
            top_at_1=top_at_1,
        )
    else:
        broadband_up, broadband_down, broadband_up_jac = _solve_longwave_noscat_tensor(
            tau_t,
            lay_t,
            lev_inc_t,
            lev_dec_t,
            sfc_t,
            sfc_jac_t,
            emiss,
            band,
            ds,
            top_at_1=top_at_1,
        )

    return LongwaveFluxResult(
        flux_up=broadband_up,
        flux_down=broadband_down,
        flux_up_jac=broadband_up_jac,
    )


_GAUSS_JACOBI_5_D = {
    1: (1.66,),
    2: (1.18350343, 2.81649655),
    3: (1.09719858, 1.69338507, 4.70941630),
    4: (1.06056257, 1.38282560, 2.40148179, 7.15513024),
}
_GAUSS_JACOBI_5_W = {
    1: (0.5,),
    2: (0.3180413817, 0.1819586183),
    3: (0.2009319137, 0.2292411064, 0.0698269799),
    4: (0.1355069134, 0.2034645680, 0.1298475476, 0.0311809710),
}


def _quadrature(n_gauss_angles: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if n_gauss_angles not in _GAUSS_JACOBI_5_D:
        raise ValueError(f"n_gauss_angles must be 1..4, got {n_gauss_angles}")
    return _GAUSS_JACOBI_5_D[n_gauss_angles], _GAUSS_JACOBI_5_W[n_gauss_angles]




def _longwave_rescaled_source_terms(
    tau_scaled: torch.Tensor,
    transmittance: torch.Tensor,
    wb: torch.Tensor,
    scale_tau: torch.Tensor,
    lay_source: torch.Tensor,
    lev_source_up: torch.Tensor,
    lev_source_down: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build rescaled source tensors from externally evaluated transmittance."""

    a = 1.0 - transmittance * transmittance
    c = 0.4 * wb / scale_tau
    source_up = lw_noscat_source_up(lev_source_up, lay_source, tau_scaled, transmittance)
    source_down = lw_noscat_source_down(lev_source_down, lay_source, tau_scaled, transmittance)
    return a, c, source_up, source_down


def _longwave_rescaled_prepare_autograd(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    asymmetry: torch.Tensor,
    lay_source: torch.Tensor,
    lev_source_up: torch.Tensor,
    lev_source_down: torch.Tensor,
    diffusivity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build rescaled tensors with the historical gradient-active graph."""

    wb = ssa * (1.0 - asymmetry) * 0.5
    scale_tau = 1.0 - ssa + wb
    tau_scaled = tau * diffusivity * scale_tau
    trans = _exp_neg(tau_scaled)
    a = 1.0 - trans * trans
    c = 0.4 * wb / scale_tau
    source_up = lw_noscat_source_up(lev_source_up, lay_source, tau_scaled, trans)
    source_down = lw_noscat_source_down(lev_source_down, lay_source, tau_scaled, trans)
    return trans, a, c, source_up, source_down


def _longwave_rescaled_initial_down_slab(
    trans: torch.Tensor,
    source_down: torch.Tensor,
    down_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the source-only downward carry from high to low indices."""

    nlay = trans.shape[1]
    down = down_in
    values: list[torch.Tensor] = [down_in * 0.0] * nlay
    for ilay in range(nlay - 1, -1, -1):
        down = trans[:, ilay, :] * down + source_down[:, ilay, :]
        values[ilay] = down
    return down, torch.stack(values, dim=1)


def _longwave_rescaled_up_slab(
    trans: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    source_up: torch.Tensor,
    source_down: torch.Tensor,
    down_reference: torch.Tensor,
    up_in: torch.Tensor,
    jacobian_in: torch.Tensor,
    *,
    reverse: bool,
    has_jacobian: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the adjusted upward recurrence through one fixed slab."""

    nlay = trans.shape[1]
    up = up_in
    jacobian = jacobian_in
    up_values: list[torch.Tensor] = [up_in * 0.0] * nlay
    jacobian_values: list[torch.Tensor] = [jacobian_in * 0.0] * nlay
    layer_order = range(nlay - 1, -1, -1) if reverse else range(nlay)
    for ilay in layer_order:
        adjustment = c[:, ilay, :] * (
            a[:, ilay, :] * down_reference[:, ilay, :]
            - trans[:, ilay, :] * source_down[:, ilay, :]
            - source_up[:, ilay, :]
        )
        up = trans[:, ilay, :] * up + source_up[:, ilay, :] + adjustment
        up_values[ilay] = up
        if has_jacobian:
            jacobian = trans[:, ilay, :] * jacobian
            jacobian_values[ilay] = jacobian
    return up, jacobian, torch.stack(up_values, dim=1), torch.stack(jacobian_values, dim=1)


def _longwave_rescaled_up_from_initial_slab(
    trans: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    source_up: torch.Tensor,
    source_down: torch.Tensor,
    initial_down_levels: torch.Tensor,
    initial_down_after: torch.Tensor,
    up_in: torch.Tensor,
    jacobian_in: torch.Tensor,
    *,
    reverse: bool,
    has_jacobian: bool,
    down_reference_same_index: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance one up slab using local initial-down values and one carry."""

    nlay = trans.shape[1]
    up = up_in
    jacobian = jacobian_in
    up_values: list[torch.Tensor] = [up_in * 0.0] * nlay
    jacobian_values: list[torch.Tensor] = [jacobian_in * 0.0] * nlay
    layer_order = range(nlay - 1, -1, -1) if reverse else range(nlay)
    for ilay in layer_order:
        down_reference = (
            initial_down_levels[:, ilay, :]
            if down_reference_same_index
            else (
                initial_down_levels[:, ilay + 1, :]
                if ilay + 1 < nlay
                else initial_down_after
            )
        )
        adjustment = c[:, ilay, :] * (
            a[:, ilay, :] * down_reference
            - trans[:, ilay, :] * source_down[:, ilay, :]
            - source_up[:, ilay, :]
        )
        up = trans[:, ilay, :] * up + source_up[:, ilay, :] + adjustment
        up_values[ilay] = up
        if has_jacobian:
            jacobian = trans[:, ilay, :] * jacobian
            jacobian_values[ilay] = jacobian
    return up, jacobian, torch.stack(up_values, dim=1), torch.stack(jacobian_values, dim=1)


def _longwave_rescaled_final_down_slab(
    trans: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    source_up: torch.Tensor,
    source_down: torch.Tensor,
    up_reference: torch.Tensor,
    down_in: torch.Tensor,
    *,
    reverse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the corrected downward recurrence through one fixed slab."""

    nlay = trans.shape[1]
    down = down_in
    values: list[torch.Tensor] = [down_in * 0.0] * nlay
    layer_order = range(nlay - 1, -1, -1) if reverse else range(nlay)
    for ilay in layer_order:
        adjustment = c[:, ilay, :] * (
            a[:, ilay, :] * up_reference[:, ilay, :]
            - trans[:, ilay, :] * source_up[:, ilay, :]
            - source_down[:, ilay, :]
        )
        down = trans[:, ilay, :] * down + source_down[:, ilay, :] + adjustment
        values[ilay] = down
    return down, torch.stack(values, dim=1)


def _longwave_rescaled_final_down_from_up_slab(
    trans: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    source_up: torch.Tensor,
    source_down: torch.Tensor,
    up_levels: torch.Tensor,
    up_before: torch.Tensor,
    down_in: torch.Tensor,
    *,
    reverse: bool,
    up_reference_same_index: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one down slab using local up values and one boundary carry."""

    nlay = trans.shape[1]
    down = down_in
    values: list[torch.Tensor] = [down_in * 0.0] * nlay
    layer_order = range(nlay - 1, -1, -1) if reverse else range(nlay)
    for ilay in layer_order:
        up_reference = (
            up_levels[:, ilay, :]
            if up_reference_same_index
            else (up_before if ilay == 0 else up_levels[:, ilay - 1, :])
        )
        adjustment = c[:, ilay, :] * (
            a[:, ilay, :] * up_reference
            - trans[:, ilay, :] * source_up[:, ilay, :]
            - source_down[:, ilay, :]
        )
        down = trans[:, ilay, :] * down + source_down[:, ilay, :] + adjustment
        values[ilay] = down
    return down, torch.stack(values, dim=1)


def _solve_longwave_rescaled_noscat_slabbed(
    tau_t: torch.Tensor,
    ssa_t: torch.Tensor,
    g_t: torch.Tensor,
    lay_t: torch.Tensor,
    lev_inc_t: torch.Tensor,
    lev_dec_t: torch.Tensor,
    sfc_t: torch.Tensor,
    sfc_jac_t: torch.Tensor | None,
    emiss: torch.Tensor,
    band: torch.Tensor,
    quad_ds: tuple[float, ...],
    quad_weights: tuple[float, ...],
    *,
    top_at_1: bool,
    stream_private_references: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run rescaled no-scatter LW transport through bounded compile slabs."""

    ngpt, nlay, ncol = tau_t.shape
    nlev = nlay + 1
    lev_up_t = lev_dec_t if top_at_1 else lev_inc_t
    lev_dn_t = lev_inc_t if top_at_1 else lev_dec_t
    sfc_emis = emiss.index_select(0, band)
    zero = torch.zeros((ngpt, ncol), dtype=tau_t.dtype, device=tau_t.device)
    broadband_up = torch.zeros((nlev, ncol), dtype=tau_t.dtype, device=tau_t.device)
    broadband_down = torch.zeros_like(broadband_up)
    broadband_up_jac = torch.zeros_like(broadband_up) if sfc_jac_t is not None else None
    slab_levels = _longwave_transport_slab_levels(tau_t)
    starts = list(range(0, nlay, slab_levels))
    padded_nlay = starts[-1] + slab_levels
    previous_start = {
        start: starts[index - 1]
        for index, start in enumerate(starts)
        if index > 0
    }
    gradient_active = _needs_autograd_stack(
        tau_t,
        ssa_t,
        g_t,
        lay_t,
        lev_inc_t,
        lev_dec_t,
        sfc_t,
        sfc_jac_t,
        emiss,
    )

    for d_value, weight_value in zip(quad_ds, quad_weights, strict=True):
        diffusivity = torch.as_tensor(d_value, dtype=tau_t.dtype, device=tau_t.device)
        weight = torch.as_tensor(2.0 * weight_value * torch.pi, dtype=tau_t.dtype, device=tau_t.device)
        if gradient_active:
            trans, a, c, source_up, source_down = run_compile_island(
                "longwave_transport",
                "longwave_rescaled_prepare_autograd",
                _longwave_rescaled_prepare_autograd,
                tau_t,
                ssa_t,
                g_t,
                lay_t,
                lev_up_t,
                lev_dn_t,
                diffusivity,
            )
        else:
            wb = ssa_t * (1.0 - g_t) * 0.5
            scale_tau = 1.0 - ssa_t + wb
            tau_scaled = tau_t * diffusivity * scale_tau
            trans = _exp_neg(tau_scaled)
            a, c, source_up, source_down = run_compile_island(
                "longwave_transport",
                "longwave_rescaled_source_terms",
                _longwave_rescaled_source_terms,
                tau_scaled,
                trans,
                wb,
                scale_tau,
                lay_t,
                lev_up_t,
                lev_dn_t,
            )

        trans_slab = _pad_longwave_layer_axis(trans, padded_nlay, fill_value=1.0)
        a_slab = _pad_longwave_layer_axis(a, padded_nlay, fill_value=0.0)
        c_slab = _pad_longwave_layer_axis(c, padded_nlay, fill_value=0.0)
        source_up_slab = _pad_longwave_layer_axis(source_up, padded_nlay, fill_value=0.0)
        source_down_slab = _pad_longwave_layer_axis(source_down, padded_nlay, fill_value=0.0)

        initial_down_current = zero
        initial_down_outputs: dict[int, torch.Tensor] = {}
        for start in reversed(starts):
            stop = start + slab_levels
            initial_down_current, down_values = run_compile_island(
                "longwave_transport",
                "longwave_rescaled_initial_down_slab",
                _longwave_rescaled_initial_down_slab,
                trans_slab[:, start:stop, :],
                source_down_slab[:, start:stop, :],
                initial_down_current,
            )
            initial_down_outputs[start] = down_values

        has_jacobian = sfc_jac_t is not None
        jacobian_surface = sfc_emis * sfc_jac_t if sfc_jac_t is not None else zero
        if stream_private_references:
            surface_initial_down = zero if top_at_1 else initial_down_current
            up_surface = surface_initial_down * (1.0 - sfc_emis) + sfc_emis * sfc_t
            up_current = up_surface
            jacobian_current = jacobian_surface
            up_outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            up_traversal = list(reversed(starts)) if top_at_1 else starts
            for start in up_traversal:
                stop = start + slab_levels
                valid_stop = min(stop, nlay)
                initial_down_after = (
                    zero
                    if top_at_1 or valid_stop == nlay
                    else initial_down_outputs[valid_stop][:, 0, :]
                )
                up_current, jacobian_current, up_values, jacobian_values = run_compile_island(
                    "longwave_transport",
                    "longwave_rescaled_up_slab",
                    _longwave_rescaled_up_from_initial_slab,
                    trans_slab[:, start:stop, :],
                    a_slab[:, start:stop, :],
                    c_slab[:, start:stop, :],
                    source_up_slab[:, start:stop, :],
                    source_down_slab[:, start:stop, :],
                    initial_down_outputs[start],
                    initial_down_after,
                    up_current,
                    jacobian_current,
                    reverse=top_at_1,
                    has_jacobian=has_jacobian,
                    down_reference_same_index=top_at_1,
                )
                up_outputs[start] = (up_values, jacobian_values)
            ordered_up = [up_outputs[start] for start in starts]
            up_chunks = [value[0] for value in ordered_up]
            up_layers = _trim_longwave_layer_chunks(up_chunks, nlay)
            up_gpt = (
                torch.cat([up_layers, up_surface.unsqueeze(1)], dim=1)
                if top_at_1
                else torch.cat([up_surface.unsqueeze(1), up_layers], dim=1)
            )
            del initial_down_outputs
            final_down_current = initial_down_current if top_at_1 else zero
            final_down_outputs: dict[int, torch.Tensor] = {}
            final_down_traversal = starts if top_at_1 else list(reversed(starts))
            for start in final_down_traversal:
                stop = start + slab_levels
                up_before = (
                    zero
                    if top_at_1
                    else (
                        up_surface
                        if start == 0
                        else up_outputs[previous_start[start]][0][:, -1, :]
                    )
                )
                final_down_current, down_values = run_compile_island(
                    "longwave_transport",
                    "longwave_rescaled_final_down_slab",
                    _longwave_rescaled_final_down_from_up_slab,
                    trans_slab[:, start:stop, :],
                    a_slab[:, start:stop, :],
                    c_slab[:, start:stop, :],
                    source_up_slab[:, start:stop, :],
                    source_down_slab[:, start:stop, :],
                    up_outputs[start][0],
                    up_before,
                    final_down_current,
                    reverse=not top_at_1,
                    up_reference_same_index=top_at_1,
                )
                final_down_outputs[start] = down_values
            final_down_chunks = [final_down_outputs[start] for start in starts]
            final_down_layers = _trim_longwave_layer_chunks(final_down_chunks, nlay)
            down_gpt = (
                torch.cat([initial_down_current.unsqueeze(1), final_down_layers], dim=1)
                if top_at_1
                else torch.cat([final_down_layers, zero.unsqueeze(1)], dim=1)
            )
            del final_down_outputs, final_down_chunks
        else:
            initial_down_layers = _trim_longwave_layer_chunks(
                [initial_down_outputs[start] for start in starts], nlay
            )
            initial_down = torch.cat([initial_down_layers, zero.unsqueeze(1)], dim=1)
            del initial_down_outputs
            sfc_index = nlay if top_at_1 else 0
            up_surface = initial_down[:, sfc_index, :] * (1.0 - sfc_emis) + sfc_emis * sfc_t
            down_reference = initial_down[:, :-1, :] if top_at_1 else initial_down[:, 1:, :]
            up_current = up_surface
            jacobian_current = jacobian_surface
            up_outputs = {}
            up_traversal = list(reversed(starts)) if top_at_1 else starts
            down_reference_slab = _pad_longwave_layer_axis(
                down_reference, padded_nlay, fill_value=0.0
            )
            for start in up_traversal:
                stop = start + slab_levels
                up_current, jacobian_current, up_values, jacobian_values = run_compile_island(
                    "longwave_transport",
                    "longwave_rescaled_up_slab",
                    _longwave_rescaled_up_slab,
                    trans_slab[:, start:stop, :],
                    a_slab[:, start:stop, :],
                    c_slab[:, start:stop, :],
                    source_up_slab[:, start:stop, :],
                    source_down_slab[:, start:stop, :],
                    down_reference_slab[:, start:stop, :],
                    up_current,
                    jacobian_current,
                    reverse=top_at_1,
                    has_jacobian=has_jacobian,
                )
                up_outputs[start] = (up_values, jacobian_values)
            ordered_up = [up_outputs[start] for start in starts]
            up_chunks = [value[0] for value in ordered_up]
            up_layers = _trim_longwave_layer_chunks(up_chunks, nlay)
            up_gpt = (
                torch.cat([up_layers, up_surface.unsqueeze(1)], dim=1)
                if top_at_1
                else torch.cat([up_surface.unsqueeze(1), up_layers], dim=1)
            )
            final_down_current = initial_down_current if top_at_1 else zero
            up_reference = up_gpt[:, :-1, :]
            up_reference_slab = _pad_longwave_layer_axis(
                up_reference, padded_nlay, fill_value=0.0
            )
            del down_reference, initial_down
            final_down_outputs = {}
            final_down_traversal = starts if top_at_1 else list(reversed(starts))
            for start in final_down_traversal:
                stop = start + slab_levels
                final_down_current, down_values = run_compile_island(
                    "longwave_transport",
                    "longwave_rescaled_final_down_slab",
                    _longwave_rescaled_final_down_slab,
                    trans_slab[:, start:stop, :],
                    a_slab[:, start:stop, :],
                    c_slab[:, start:stop, :],
                    source_up_slab[:, start:stop, :],
                    source_down_slab[:, start:stop, :],
                    up_reference_slab[:, start:stop, :],
                    final_down_current,
                    reverse=not top_at_1,
                )
                final_down_outputs[start] = down_values
            final_down_chunks = [final_down_outputs[start] for start in starts]
            final_down_layers = _trim_longwave_layer_chunks(final_down_chunks, nlay)
            down_gpt = (
                torch.cat([initial_down_current.unsqueeze(1), final_down_layers], dim=1)
                if top_at_1
                else torch.cat([final_down_layers, zero.unsqueeze(1)], dim=1)
            )
            del final_down_outputs, final_down_chunks

        broadband_up = broadband_up + _sum_gpoints_fortran_order(up_gpt * weight)
        broadband_down = broadband_down + _sum_gpoints_fortran_order(down_gpt * weight)
        if broadband_up_jac is not None:
            jacobian_chunks = [value[1] for value in ordered_up]
            jacobian_layers = _trim_longwave_layer_chunks(jacobian_chunks, nlay)
            jacobian_gpt = (
                torch.cat([jacobian_layers, jacobian_surface.unsqueeze(1)], dim=1)
                if top_at_1
                else torch.cat([jacobian_surface.unsqueeze(1), jacobian_layers], dim=1)
            )
            broadband_up_jac = broadband_up_jac + _sum_gpoints_fortran_order(jacobian_gpt * weight)
        del up_outputs, ordered_up, up_chunks

    return broadband_up, broadband_down, broadband_up_jac


def _solve_longwave_rescaled_noscat_tensor(
    tau_t: torch.Tensor,
    ssa_t: torch.Tensor,
    g_t: torch.Tensor,
    lay_t: torch.Tensor,
    lev_inc_t: torch.Tensor,
    lev_dec_t: torch.Tensor,
    sfc_t: torch.Tensor,
    sfc_jac_t: torch.Tensor | None,
    emiss: torch.Tensor,
    band: torch.Tensor,
    quad_ds: tuple[float, ...],
    quad_weights: tuple[float, ...],
    *,
    top_at_1: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    ngpt, nlay, ncol = tau_t.shape
    nlev = nlay + 1
    lev_up_t = lev_dec_t if top_at_1 else lev_inc_t
    lev_dn_t = lev_inc_t if top_at_1 else lev_dec_t
    broadband_up = torch.zeros((nlev, ncol), dtype=tau_t.dtype, device=tau_t.device)
    broadband_down = torch.zeros_like(broadband_up)
    broadband_up_jac = torch.zeros_like(broadband_up) if sfc_jac_t is not None else None
    sfc_emis = emiss.index_select(0, band)
    zero_gpt_col = torch.zeros((ngpt, ncol), dtype=tau_t.dtype, device=tau_t.device)
    use_stack = _needs_autograd_stack(tau_t, ssa_t, g_t, lay_t, lev_inc_t, lev_dec_t, sfc_t, sfc_jac_t, emiss)
    for d_value, weight_value in zip(quad_ds, quad_weights):
        weight = torch.as_tensor(2.0 * weight_value * torch.pi, dtype=tau_t.dtype, device=tau_t.device)
        d_scalar = torch.as_tensor(d_value, dtype=tau_t.dtype, device=tau_t.device)
        wb = ssa_t * (1.0 - g_t) * 0.5
        scale_tau = 1.0 - ssa_t + wb
        tau_scaled = tau_t * d_scalar * scale_tau
        trans_layers = _exp_neg(tau_scaled)
        a_layers = 1.0 - trans_layers * trans_layers
        c_layers = 0.4 * wb / scale_tau
        source_up_layers = lw_noscat_source_up(lev_up_t, lay_t, tau_scaled, trans_layers)
        source_down_layers = lw_noscat_source_down(lev_dn_t, lay_t, tau_scaled, trans_layers)

        if use_stack:
            zero = zero_gpt_col
            down_values: list[torch.Tensor] = [zero for _ in range(nlev)]
            down_values[nlay] = zero
            for ilay in range(nlay - 1, -1, -1):
                down_values[ilay] = (
                    trans_layers[:, ilay, :] * down_values[ilay + 1] + source_down_layers[:, ilay, :]
                )
            up_values: list[torch.Tensor] = [zero for _ in range(nlev)]
            up_jac_values: list[torch.Tensor] | None = [zero for _ in range(nlev)] if sfc_jac_t is not None else None
            if top_at_1:
                up_values[nlay] = down_values[nlay] * (1.0 - sfc_emis) + sfc_emis * sfc_t
                if sfc_jac_t is not None and up_jac_values is not None:
                    up_jac_values[nlay] = sfc_emis * sfc_jac_t
                for ilay in range(nlay - 1, -1, -1):
                    adjustment = c_layers[:, ilay, :] * (
                        a_layers[:, ilay, :] * down_values[ilay]
                        - trans_layers[:, ilay, :] * source_down_layers[:, ilay, :]
                        - source_up_layers[:, ilay, :]
                    )
                    up_values[ilay] = (
                        trans_layers[:, ilay, :] * up_values[ilay + 1]
                        + source_up_layers[:, ilay, :]
                        + adjustment
                    )
                    if sfc_jac_t is not None and up_jac_values is not None:
                        up_jac_values[ilay] = trans_layers[:, ilay, :] * up_jac_values[ilay + 1]
                for ilay in range(nlay):
                    adjustment = c_layers[:, ilay, :] * (
                        a_layers[:, ilay, :] * up_values[ilay]
                        - trans_layers[:, ilay, :] * source_up_layers[:, ilay, :]
                        - source_down_layers[:, ilay, :]
                    )
                    down_values[ilay + 1] = (
                        trans_layers[:, ilay, :] * down_values[ilay]
                        + source_down_layers[:, ilay, :]
                        + adjustment
                    )
            else:
                up_values[0] = down_values[0] * (1.0 - sfc_emis) + sfc_emis * sfc_t
                if sfc_jac_t is not None and up_jac_values is not None:
                    up_jac_values[0] = sfc_emis * sfc_jac_t
                for ilay in range(nlay):
                    adjustment = c_layers[:, ilay, :] * (
                        a_layers[:, ilay, :] * down_values[ilay + 1]
                        - trans_layers[:, ilay, :] * source_down_layers[:, ilay, :]
                        - source_up_layers[:, ilay, :]
                    )
                    up_values[ilay + 1] = (
                        trans_layers[:, ilay, :] * up_values[ilay]
                        + source_up_layers[:, ilay, :]
                        + adjustment
                    )
                    if sfc_jac_t is not None and up_jac_values is not None:
                        up_jac_values[ilay + 1] = trans_layers[:, ilay, :] * up_jac_values[ilay]
                for ilay in range(nlay - 1, -1, -1):
                    adjustment = c_layers[:, ilay, :] * (
                        a_layers[:, ilay, :] * up_values[ilay]
                        - trans_layers[:, ilay, :] * source_up_layers[:, ilay, :]
                        - source_down_layers[:, ilay, :]
                    )
                    down_values[ilay] = (
                        trans_layers[:, ilay, :] * down_values[ilay + 1]
                        + source_down_layers[:, ilay, :]
                        + adjustment
                    )
            up_gpt_stack = torch.stack(up_values, dim=1)
            down_gpt_stack = torch.stack(down_values, dim=1)
            broadband_up = broadband_up + _sum_gpoints_fortran_order(up_gpt_stack * weight)
            broadband_down = broadband_down + _sum_gpoints_fortran_order(down_gpt_stack * weight)
            if broadband_up_jac is not None and up_jac_values is not None:
                broadband_up_jac = broadband_up_jac + _sum_gpoints_fortran_order(torch.stack(up_jac_values, dim=1) * weight)
            continue

        down_gpt = torch.empty((ngpt, nlev, ncol), dtype=tau_t.dtype, device=tau_t.device)
        up_gpt = torch.empty_like(down_gpt)
        up_jac_gpt = torch.empty_like(down_gpt) if sfc_jac_t is not None else None
        down_gpt[:, nlay, :] = zero_gpt_col
        for ilay in range(nlay - 1, -1, -1):
            down_gpt[:, ilay, :] = (
                trans_layers[:, ilay, :] * down_gpt[:, ilay + 1, :] + source_down_layers[:, ilay, :]
            )

        if top_at_1:
            up_gpt[:, nlay, :] = down_gpt[:, nlay, :] * (1.0 - sfc_emis) + sfc_emis * sfc_t
            if sfc_jac_t is not None:
                up_jac_gpt[:, nlay, :] = sfc_emis * sfc_jac_t
            for ilay in range(nlay - 1, -1, -1):
                adjustment = c_layers[:, ilay, :] * (
                    a_layers[:, ilay, :] * down_gpt[:, ilay, :]
                    - trans_layers[:, ilay, :] * source_down_layers[:, ilay, :]
                    - source_up_layers[:, ilay, :]
                )
                up_gpt[:, ilay, :] = (
                    trans_layers[:, ilay, :] * up_gpt[:, ilay + 1, :]
                    + source_up_layers[:, ilay, :]
                    + adjustment
                )
                if sfc_jac_t is not None:
                    up_jac_gpt[:, ilay, :] = trans_layers[:, ilay, :] * up_jac_gpt[:, ilay + 1, :]
            for ilay in range(nlay):
                adjustment = c_layers[:, ilay, :] * (
                    a_layers[:, ilay, :] * up_gpt[:, ilay, :]
                    - trans_layers[:, ilay, :] * source_up_layers[:, ilay, :]
                    - source_down_layers[:, ilay, :]
                )
                down_gpt[:, ilay + 1, :] = (
                    trans_layers[:, ilay, :] * down_gpt[:, ilay, :]
                    + source_down_layers[:, ilay, :]
                    + adjustment
                )
        else:
            up_gpt[:, 0, :] = down_gpt[:, 0, :] * (1.0 - sfc_emis) + sfc_emis * sfc_t
            if sfc_jac_t is not None:
                up_jac_gpt[:, 0, :] = sfc_emis * sfc_jac_t
            for ilay in range(nlay):
                adjustment = c_layers[:, ilay, :] * (
                    a_layers[:, ilay, :] * down_gpt[:, ilay + 1, :]
                    - trans_layers[:, ilay, :] * source_down_layers[:, ilay, :]
                    - source_up_layers[:, ilay, :]
                )
                up_gpt[:, ilay + 1, :] = (
                    trans_layers[:, ilay, :] * up_gpt[:, ilay, :]
                    + source_up_layers[:, ilay, :]
                    + adjustment
                )
                if sfc_jac_t is not None:
                    up_jac_gpt[:, ilay + 1, :] = trans_layers[:, ilay, :] * up_jac_gpt[:, ilay, :]
            for ilay in range(nlay - 1, -1, -1):
                adjustment = c_layers[:, ilay, :] * (
                    a_layers[:, ilay, :] * up_gpt[:, ilay, :]
                    - trans_layers[:, ilay, :] * source_up_layers[:, ilay, :]
                    - source_down_layers[:, ilay, :]
                )
                down_gpt[:, ilay, :] = (
                    trans_layers[:, ilay, :] * down_gpt[:, ilay + 1, :]
                    + source_down_layers[:, ilay, :]
                    + adjustment
                )

        broadband_up = broadband_up + _sum_gpoints_fortran_order(up_gpt * weight)
        broadband_down = broadband_down + _sum_gpoints_fortran_order(down_gpt * weight)
        if broadband_up_jac is not None and up_jac_gpt is not None:
            broadband_up_jac = broadband_up_jac + _sum_gpoints_fortran_order(up_jac_gpt * weight)

    return broadband_up, broadband_down, broadband_up_jac


def solve_longwave_rescaled_noscat(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    asymmetry: torch.Tensor,
    lay_source: torch.Tensor,
    lev_source: torch.Tensor,
    sfc_source: torch.Tensor,
    sfc_emiss_byband: torch.Tensor,
    gpt_to_band: torch.Tensor,
    *,
    n_gauss_angles: int = 1,
    lev_source_dec: torch.Tensor | None = None,
    top_at_1: bool = False,
    sfc_source_jac: torch.Tensor | None = None,
) -> LongwaveFluxResult:
    """Solve the RTE/RRTMGP default LW 2-stream-object path.

    `mo_rte_lw` dispatches `ty_optical_props_2str` to the no-scatter solver
    with Tang-style optical-depth rescaling unless `use_2stream=true` is
    explicitly supplied. GFS RRTMGP calls this default path for all-sky LW when
    `doGP_lwscat` is true.
    """

    tau_t = _as_gpt_lay_col(tau, name="tau")
    ssa_t = _as_gpt_lay_col(ssa, name="ssa")
    g_t = _as_gpt_lay_col(asymmetry, name="asymmetry")
    lay_t = _as_gpt_lay_col(lay_source, name="lay_source")
    lev_inc_t, lev_dec_t = _as_gpt_lay_col_or_from_interfaces(
        lev_source,
        lev_source_dec,
        name="lev_source_inc" if lev_source_dec is not None else "lev_source",
    )
    sfc_t = _as_gpt_col(sfc_source, name="sfc_source")
    sfc_jac_t = _as_gpt_col(sfc_source_jac, name="sfc_source_jac") if sfc_source_jac is not None else None
    emiss = _real(sfc_emiss_byband, name="sfc_emiss_byband")
    if emiss.ndim != 2:
        raise ValueError(f"sfc_emiss_byband must have shape [nband,ncol], got {tuple(emiss.shape)}")
    ngpt, nlay, ncol = tau_t.shape
    if (
        ssa_t.shape != tau_t.shape
        or g_t.shape != tau_t.shape
        or lay_t.shape != tau_t.shape
        or lev_inc_t.shape != tau_t.shape
        or lev_dec_t.shape != tau_t.shape
        or sfc_t.shape != (ngpt, ncol)
        or (sfc_jac_t is not None and sfc_jac_t.shape != (ngpt, ncol))
    ):
        raise ValueError("longwave rescaled no-scattering source/optics shapes are inconsistent")
    band = _gpt_to_band(gpt_to_band, ngpt=ngpt, nband=emiss.shape[0], device=tau_t.device)
    quad_ds, quad_weights = _quadrature(n_gauss_angles)
    if "longwave_transport" in compile_categories_enabled():
        broadband_up, broadband_down, broadband_up_jac = _solve_longwave_rescaled_noscat_slabbed(
            tau_t,
            ssa_t,
            g_t,
            lay_t,
            lev_inc_t,
            lev_dec_t,
            sfc_t,
            sfc_jac_t,
            emiss,
            band,
            tuple(quad_ds),
            tuple(quad_weights),
            top_at_1=top_at_1,
            stream_private_references=tau_t.device.type == "cuda",
        )
    else:
        broadband_up, broadband_down, broadband_up_jac = _solve_longwave_rescaled_noscat_tensor(
            tau_t,
            ssa_t,
            g_t,
            lay_t,
            lev_inc_t,
            lev_dec_t,
            sfc_t,
            sfc_jac_t,
            emiss,
            band,
            tuple(quad_ds),
            tuple(quad_weights),
            top_at_1=top_at_1,
        )

    return LongwaveFluxResult(flux_up=broadband_up, flux_down=broadband_down, flux_up_jac=broadband_up_jac)


def lw_2stream_coeffs(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    asymmetry: torch.Tensor,
    lev_src_bot: torch.Tensor,
    lev_src_top: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Layer reflectance, transmittance, and LW two-stream source terms."""

    lw_diff_sec = torch.as_tensor(1.66, dtype=tau.dtype, device=tau.device)
    k_min = torch.as_tensor(torch.finfo(tau.dtype).tiny, dtype=tau.dtype, device=tau.device)
    gamma1 = lw_diff_sec * (1.0 - 0.5 * ssa * (1.0 + asymmetry))
    gamma2 = lw_diff_sec * 0.5 * ssa * (1.0 - asymmetry)
    k = torch.sqrt(torch.maximum(lw_diff_sec * (1.0 - ssa) * (gamma1 + gamma2), k_min))
    e1 = torch.exp(-tau * k)
    om1 = -torch.expm1(-tau * k)
    coeff = e1 * e1
    one_minus_e2kt = om1 * (1.0 + e1)
    rt_term = 1.0 / (k * (1.0 + coeff) + gamma1 * one_minus_e2kt)
    rdif = rt_term * gamma2 * one_minus_e2kt
    tdif = rt_term * 2.0 * k * e1

    delta_b = lev_src_bot - lev_src_top
    gamma_sum = gamma1 + gamma2
    one_p_e1 = 1.0 + e1
    emis_fac = om1 * (k * om1 + lw_diff_sec * (1.0 - ssa) * one_p_e1) * rt_term
    safe_tau = torch.where(tau > 0.0, tau, torch.ones_like(tau))
    delta_bz = (
        delta_b
        * (om1 / safe_tau)
        * (k * om1 + gamma_sum * one_p_e1)
        * rt_term
        / torch.maximum(gamma_sum, torch.as_tensor(torch.finfo(tau.dtype).eps, dtype=tau.dtype, device=tau.device))
    )
    src_up_nonzero = torch.pi * (lev_src_top * emis_fac - tdif * delta_b + delta_bz)
    src_dn_nonzero = torch.pi * (lev_src_bot * emis_fac + tdif * delta_b - delta_bz)
    zero = torch.zeros((), dtype=tau.dtype, device=tau.device)
    src_up = torch.where(tau > 0.0, src_up_nonzero, zero)
    src_dn = torch.where(tau > 0.0, src_dn_nonzero, zero)
    return rdif, tdif, src_up, src_dn


def _solve_longwave_2stream_tensor(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    asymmetry: torch.Tensor,
    lev_source: torch.Tensor,
    sfc_source: torch.Tensor,
    emissivity: torch.Tensor,
    gpt_to_band: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized two-stream reference over g-points and columns."""

    _ngpt, nlay, _ncol = tau.shape
    rdif, tdif, source_up, source_down = lw_2stream_coeffs(
        tau,
        ssa,
        asymmetry,
        lev_source[:, :-1, :],
        lev_source[:, 1:, :],
    )
    sfc_emissivity = emissivity.index_select(0, gpt_to_band)
    albedo_current = 1.0 - sfc_emissivity
    source_current = torch.pi * sfc_emissivity * sfc_source
    albedo_values = [albedo_current]
    source_values = [source_current]
    for ilay in range(nlay):
        incoming_albedo = albedo_current
        incoming_source = source_current
        denom = 1.0 / (1.0 - rdif[:, ilay, :] * incoming_albedo)
        albedo_current = (
            rdif[:, ilay, :]
            + tdif[:, ilay, :] * tdif[:, ilay, :] * incoming_albedo * denom
        )
        source_current = source_up[:, ilay, :] + tdif[:, ilay, :] * denom * (
            incoming_source + incoming_albedo * source_down[:, ilay, :]
        )
        albedo_values.append(albedo_current)
        source_values.append(source_current)

    zero = tau[:, 0, :] * 0.0
    flux_down_current = zero
    flux_up_values: list[torch.Tensor] = [zero for _ in range(nlay + 1)]
    flux_down_values: list[torch.Tensor] = [zero for _ in range(nlay + 1)]
    flux_up_values[nlay] = flux_down_current * albedo_values[nlay] + source_values[nlay]
    flux_down_values[nlay] = flux_down_current
    for ilay in range(nlay - 1, -1, -1):
        denom = 1.0 / (1.0 - rdif[:, ilay, :] * albedo_values[ilay])
        flux_down_current = (
            tdif[:, ilay, :] * flux_down_current
            + rdif[:, ilay, :] * source_values[ilay]
            + source_down[:, ilay, :]
        ) * denom
        flux_up_values[ilay] = flux_down_current * albedo_values[ilay] + source_values[ilay]
        flux_down_values[ilay] = flux_down_current
    return (
        _sum_gpoints_fortran_order(torch.stack(flux_up_values, dim=1)),
        _sum_gpoints_fortran_order(torch.stack(flux_down_values, dim=1)),
    )


def _longwave_2stream_effective_slab(
    rdif: torch.Tensor,
    tdif: torch.Tensor,
    source_up: torch.Tensor,
    source_down: torch.Tensor,
    albedo_in: torch.Tensor,
    source_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance effective albedo/source through one low-to-high slab."""

    nlay = rdif.shape[1]
    albedo = albedo_in
    source = source_in
    albedo_values: list[torch.Tensor] = [albedo_in * 0.0] * nlay
    source_values: list[torch.Tensor] = [source_in * 0.0] * nlay
    for ilay in range(nlay):
        incoming_albedo = albedo
        incoming_source = source
        denom = 1.0 / (1.0 - rdif[:, ilay, :] * incoming_albedo)
        albedo = rdif[:, ilay, :] + tdif[:, ilay, :] * tdif[:, ilay, :] * incoming_albedo * denom
        source = source_up[:, ilay, :] + tdif[:, ilay, :] * denom * (
            incoming_source + incoming_albedo * source_down[:, ilay, :]
        )
        albedo_values[ilay] = albedo
        source_values[ilay] = source
    return albedo, source, torch.stack(albedo_values, dim=1), torch.stack(source_values, dim=1)


def _longwave_2stream_flux_slab(
    rdif: torch.Tensor,
    tdif: torch.Tensor,
    source_down: torch.Tensor,
    albedo: torch.Tensor,
    source: torch.Tensor,
    flux_down_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance two-stream fluxes through one high-to-low slab."""

    nlay = rdif.shape[1]
    flux_down = flux_down_in
    zero = flux_down_in * 0.0
    up_values: list[torch.Tensor] = [zero] * nlay
    down_values: list[torch.Tensor] = [zero] * nlay
    for ilay in range(nlay - 1, -1, -1):
        denom = 1.0 / (1.0 - rdif[:, ilay, :] * albedo[:, ilay, :])
        flux_down = (
            tdif[:, ilay, :] * flux_down
            + rdif[:, ilay, :] * source[:, ilay, :]
            + source_down[:, ilay, :]
        ) * denom
        up_values[ilay] = flux_down * albedo[:, ilay, :] + source[:, ilay, :]
        down_values[ilay] = flux_down
    return flux_down, torch.stack(up_values, dim=1), torch.stack(down_values, dim=1)


def _solve_longwave_2stream_slabbed(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    asymmetry: torch.Tensor,
    lev_source: torch.Tensor,
    sfc_source: torch.Tensor,
    emissivity: torch.Tensor,
    gpt_to_band: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run full two-stream LW transport through bounded compile slabs."""

    _ngpt, nlay, _ncol = tau.shape
    rdif, tdif, source_up, source_down = run_compile_island(
        "longwave_transport",
        "longwave_2stream_coefficients",
        lw_2stream_coeffs,
        tau,
        ssa,
        asymmetry,
        lev_source[:, :-1, :],
        lev_source[:, 1:, :],
    )
    sfc_emissivity = emissivity.index_select(0, gpt_to_band)
    surface_albedo = 1.0 - sfc_emissivity
    surface_source = torch.pi * sfc_emissivity * sfc_source
    slab_levels = _longwave_transport_slab_levels(tau)
    starts = list(range(0, nlay, slab_levels))
    padded_nlay = starts[-1] + slab_levels
    rdif_slab = _pad_longwave_layer_axis(rdif, padded_nlay, fill_value=0.0)
    tdif_slab = _pad_longwave_layer_axis(tdif, padded_nlay, fill_value=1.0)
    source_up_slab = _pad_longwave_layer_axis(source_up, padded_nlay, fill_value=0.0)
    source_down_slab = _pad_longwave_layer_axis(source_down, padded_nlay, fill_value=0.0)

    albedo_current = surface_albedo
    source_current = surface_source
    effective_outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for start in starts:
        stop = start + slab_levels
        albedo_current, source_current, albedo_values, source_values = run_compile_island(
            "longwave_transport",
            "longwave_2stream_effective_slab",
            _longwave_2stream_effective_slab,
            rdif_slab[:, start:stop, :],
            tdif_slab[:, start:stop, :],
            source_up_slab[:, start:stop, :],
            source_down_slab[:, start:stop, :],
            albedo_current,
            source_current,
        )
        effective_outputs[start] = (albedo_values, source_values)
    ordered_effective = [effective_outputs[start] for start in starts]
    albedo_layers = _trim_longwave_layer_chunks(
        [value[0] for value in ordered_effective], nlay
    )
    source_layers = _trim_longwave_layer_chunks(
        [value[1] for value in ordered_effective], nlay
    )
    albedo = torch.cat(
        (surface_albedo.unsqueeze(1), albedo_layers),
        dim=1,
    )
    source = torch.cat(
        (surface_source.unsqueeze(1), source_layers),
        dim=1,
    )
    albedo_slab = _pad_longwave_layer_axis(albedo, padded_nlay, fill_value=0.0)
    source_slab = _pad_longwave_layer_axis(source, padded_nlay, fill_value=0.0)

    zero = tau[:, 0, :] * 0.0
    flux_down_current = zero
    flux_outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for start in reversed(starts):
        stop = start + slab_levels
        flux_down_current, up_values, down_values = run_compile_island(
            "longwave_transport",
            "longwave_2stream_flux_slab",
            _longwave_2stream_flux_slab,
            rdif_slab[:, start:stop, :],
            tdif_slab[:, start:stop, :],
            source_down_slab[:, start:stop, :],
            albedo_slab[:, start:stop, :],
            source_slab[:, start:stop, :],
            flux_down_current,
        )
        flux_outputs[start] = (up_values, down_values)
    ordered_flux = [flux_outputs[start] for start in starts]
    flux_up_layers = _trim_longwave_layer_chunks([value[0] for value in ordered_flux], nlay)
    flux_down_layers = _trim_longwave_layer_chunks([value[1] for value in ordered_flux], nlay)
    flux_up_gpt = torch.cat(
        (flux_up_layers, source[:, nlay:, :]),
        dim=1,
    )
    flux_down_gpt = torch.cat(
        (flux_down_layers, zero.unsqueeze(1)),
        dim=1,
    )
    return _sum_gpoints_fortran_order(flux_up_gpt), _sum_gpoints_fortran_order(flux_down_gpt)




def solve_longwave_2stream(
    tau: torch.Tensor,
    ssa: torch.Tensor,
    asymmetry: torch.Tensor,
    lev_source: torch.Tensor,
    sfc_source: torch.Tensor,
    sfc_emiss_byband: torch.Tensor,
    gpt_to_band: torch.Tensor,
) -> LongwaveFluxResult:
    """Solve broadband two-stream LW transport from explicit source hooks."""

    tau_t = _as_gpt_lay_col(tau, name="tau")
    ssa_t = _as_gpt_lay_col(ssa, name="ssa")
    g_t = _as_gpt_lay_col(asymmetry, name="asymmetry")
    lev_t = _as_gpt_lev_col(lev_source, name="lev_source")
    sfc_t = _as_gpt_col(sfc_source, name="sfc_source")
    emiss = _real(sfc_emiss_byband, name="sfc_emiss_byband")
    ngpt, nlay, ncol = tau_t.shape
    if ssa_t.shape != tau_t.shape or g_t.shape != tau_t.shape or lev_t.shape != (ngpt, nlay + 1, ncol):
        raise ValueError("longwave two-stream source/optics shapes are inconsistent")
    band = _gpt_to_band(gpt_to_band, ngpt=ngpt, nband=emiss.shape[0], device=tau_t.device)
    if "longwave_transport" in compile_categories_enabled():
        flux_up, flux_down = _solve_longwave_2stream_slabbed(
            tau_t,
            ssa_t,
            g_t,
            lev_t,
            sfc_t,
            emiss,
            band,
        )
    else:
        flux_up, flux_down = _solve_longwave_2stream_tensor(
            tau_t,
            ssa_t,
            g_t,
            lev_t,
            sfc_t,
            emiss,
            band,
        )
    return LongwaveFluxResult(flux_up=flux_up, flux_down=flux_down)


__all__ = [
    "LongwaveFluxResult",
    "lw_2stream_coeffs",
    "lw_noscat_source_down",
    "lw_noscat_source_up",
    "solve_longwave_rescaled_noscat",
    "solve_longwave_2stream",
    "solve_longwave_noscat",
]
