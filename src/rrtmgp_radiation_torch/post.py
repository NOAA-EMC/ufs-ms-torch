# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""PyTorch-native RRTMGP post-processing kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .compile import run_compile_island

GRAV = 9.80665
CP_DRY = 1004.64


TensorOrString = torch.Tensor | str


@dataclass(frozen=True)
class RrtmgpPostResult:
    """Checkpoint-shaped final-J output fields computed from post inputs."""

    fields: Mapping[str, TensorOrString]
    runtime_kind: str = "torch_rrtmgp_post"
    # Validation-only references to the exact live inputs from which the four
    # heating fields were formed.  This adds no scientific work and is not part
    # of the public field surface; it lets the gate prove the flux/heating
    # invariant without rerunning H and I or accepting injected diagnostics.
    accuracy_context: Mapping[str, torch.Tensor] | None = None


def _to_real(value: TensorOrString, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64 (the working dtype), got {value.dtype}")
    return value


def _index_tensor(
    value: TensorOrString, *, size: int, name: str, device: torch.device
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    index = value.to(device=device, dtype=torch.int64).reshape(-1)[:1] - 1
    torch._assert_async(
        ((index >= 0) & (index < size)).all(), f"{name} is outside the interface range"
    )
    return index


def compute_heating_rate(
    flux_up: torch.Tensor,
    flux_down: torch.Tensor,
    p_lev: torch.Tensor,
    *,
    grav: float = GRAV,
    cp_dry: float = CP_DRY,
) -> torch.Tensor:
    """Match RTE/RRTMGP `mo_heating_rates::compute_heating_rate_general`.

    The checkpoint convention stores Fortran `(ncol,nlev)` arrays as
    `(nlev,ncol)`, so the vertical interface axis is axis 0.
    """

    up = _to_real(flux_up, name="flux_up")
    down = _to_real(flux_down, name="flux_down")
    pressure = _to_real(p_lev, name="p_lev")
    if up.shape != down.shape or up.shape != pressure.shape:
        raise ValueError(
            "flux_up, flux_down, and p_lev must have identical checkpoint-shaped "
            f"interfaces, got {up.shape}, {down.shape}, {pressure.shape}"
        )
    if up.ndim != 2 or up.shape[0] < 2:
        raise ValueError(f"interface fields must have shape [nlev, ncol], got {up.shape}")
    return (
        (up[1:, :] - up[:-1, :] - down[1:, :] + down[:-1, :])
        * grav
        / (cp_dry * (pressure[1:, :] - pressure[:-1, :]))
    )


def _zero_like_layers(p_lev: torch.Tensor) -> torch.Tensor:
    pressure = _to_real(p_lev, name="p_lev")
    return torch.zeros(
        (pressure.shape[0] - 1, pressure.shape[1]), dtype=pressure.dtype, device=pressure.device
    )


def _copy_scalar_from(
    mapping: Mapping[str, TensorOrString], name: str, template: torch.Tensor
) -> torch.Tensor:
    value = mapping.get(name)
    if isinstance(value, torch.Tensor):
        return value.to(dtype=template.dtype, device=template.device).reshape(-1)
    return torch.zeros((1,), dtype=template.dtype, device=template.device)


def _final_j_tensor_kernel(
    p_lev: torch.Tensor,
    fluxlw_up_all: torch.Tensor,
    fluxlw_down_all: torch.Tensor,
    fluxlw_up_clr: torch.Tensor,
    fluxlw_down_clr: torch.Tensor,
    fluxsw_up_all: torch.Tensor,
    fluxsw_down_all: torch.Tensor,
    fluxsw_up_clr: torch.Tensor,
    fluxsw_down_clr: torch.Tensor,
    day_columns: torch.Tensor,
    *,
    i_sfc: int | torch.Tensor,
    i_toa: int | torch.Tensor,
    use_shortwave: bool | torch.Tensor,
    day_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Assemble the differentiable final-J tensor surface in one flat island."""

    htrlw = compute_heating_rate(fluxlw_up_all, fluxlw_down_all, p_lev)
    htrlwc = compute_heating_rate(fluxlw_up_clr, fluxlw_down_clr, p_lev)
    sfc = torch.as_tensor(i_sfc, dtype=torch.int64, device=p_lev.device).reshape(1)
    toa = torch.as_tensor(i_toa, dtype=torch.int64, device=p_lev.device).reshape(1)
    enabled = torch.as_tensor(use_shortwave, dtype=torch.bool, device=p_lev.device)

    def interface(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return value.index_select(0, index).squeeze(0)

    sfcflw = torch.stack(
        (
            interface(fluxlw_up_all, sfc),
            interface(fluxlw_up_clr, sfc),
            interface(fluxlw_down_all, sfc),
            interface(fluxlw_down_clr, sfc),
        ),
        dim=0,
    )
    topflw = torch.stack((interface(fluxlw_up_all, toa), interface(fluxlw_up_clr, toa)), dim=0)
    ncol = p_lev.shape[1]
    if day_mask is None:
        day_mask = torch.zeros((ncol,), dtype=torch.bool, device=p_lev.device).index_fill(
            0, day_columns, True
        )
    active = enabled & day_mask
    # Inactive pressure differences must be safe in both AD modes.
    sw_pressure = torch.where(
        active.unsqueeze(0),
        p_lev,
        torch.arange(p_lev.shape[0], dtype=p_lev.dtype, device=p_lev.device).unsqueeze(1),
    )
    up_all = torch.where(active.unsqueeze(0), fluxsw_up_all, 0.0)
    down_all = torch.where(active.unsqueeze(0), fluxsw_down_all, 0.0)
    up_clr = torch.where(active.unsqueeze(0), fluxsw_up_clr, 0.0)
    down_clr = torch.where(active.unsqueeze(0), fluxsw_down_clr, 0.0)
    htrsw = compute_heating_rate(up_all, down_all, sw_pressure)
    htrswc = compute_heating_rate(up_clr, down_clr, sw_pressure)
    topfsw = torch.where(
        enabled,
        torch.stack(
            (
                interface(fluxsw_up_all, toa),
                interface(fluxsw_down_all, toa),
                interface(fluxsw_up_clr, toa),
            ),
            dim=0,
        ),
        0.0,
    )
    sfcfsw = torch.where(
        enabled,
        torch.stack(
            (
                interface(fluxsw_up_all, sfc),
                interface(fluxsw_down_all, sfc),
                interface(fluxsw_up_clr, sfc),
                interface(fluxsw_down_clr, sfc),
            ),
            dim=0,
        ),
        0.0,
    )
    return htrsw, htrlw, htrswc, htrlwc, topfsw, sfcfsw, sfcflw, topflw


def run_post_from_boundaries(
    boundaries: Mapping[str, Mapping[str, TensorOrString]],
) -> RrtmgpPostResult:
    """Compute final-J post fields from upstream radiation boundary state.

    This ports the scientific post-processing formulas. It intentionally consumes
    SW/LW fluxes produced upstream; it does not compute gas optics, cloud optics,
    or the radiative-transfer solve.
    """

    setup = boundaries["A"]
    pre = boundaries["AB"]
    sw = boundaries["ABCDEFGH"]
    lw = boundaries["ABCDEFGHI"]
    final_inputs = boundaries.get(
        "runtime_controls",
        boundaries.get("AB_pre_inputs", boundaries.get("ABCDEFGHIJ", {})),
    )

    p_lev = _to_real(pre["p_lev"], name="p_lev")
    n_interface, ncol = p_lev.shape
    i_sfc = _index_tensor(pre["iSFC"], size=n_interface, name="iSFC", device=p_lev.device)
    i_toa = _index_tensor(pre["iTOA"], size=n_interface, name="iTOA", device=p_lev.device)
    n_day = pre["nDay"].to(device=p_lev.device, dtype=torch.int64).reshape(())

    fluxlw_up_all = _to_real(lw["fluxlwUP_allsky"], name="fluxlwUP_allsky")
    fluxlw_down_all = _to_real(lw["fluxlwDOWN_allsky"], name="fluxlwDOWN_allsky")
    fluxlw_up_clr = _to_real(lw["fluxlwUP_clrsky"], name="fluxlwUP_clrsky")
    fluxlw_down_clr = _to_real(lw["fluxlwDOWN_clrsky"], name="fluxlwDOWN_clrsky")

    do_sw = (
        setup.get("doSWrad", pre.get("doSWrad"))
        .to(device=p_lev.device, dtype=torch.bool)
        .reshape(())
    )
    use_shortwave = do_sw & (n_day > 0)

    def sw_flux(name: str) -> torch.Tensor:
        value = sw.get(name)
        if not isinstance(value, torch.Tensor):
            torch._assert_async(~use_shortwave, f"missing active shortwave flux {name}")
        field = (
            _to_real(value, name=name)
            if isinstance(value, torch.Tensor)
            else torch.zeros_like(fluxlw_up_all)
        )
        return torch.where(use_shortwave, field, 0.0)

    fluxsw_up_all = sw_flux("fluxswUP_allsky")
    fluxsw_down_all = sw_flux("fluxswDOWN_allsky")
    fluxsw_up_clr = sw_flux("fluxswUP_clrsky")
    fluxsw_down_clr = sw_flux("fluxswDOWN_clrsky")
    idxday = pre["idxday"]
    if not isinstance(idxday, torch.Tensor):
        raise TypeError("idxday must be a torch tensor")
    day_columns = idxday.to(device=p_lev.device, dtype=torch.int64).reshape(-1) - 1
    valid = torch.arange(day_columns.numel(), device=p_lev.device) < n_day
    torch._assert_async(
        (~valid | ((day_columns >= 0) & (day_columns < ncol))).all(),
        "idxday is outside the column range",
    )
    torch._assert_async(
        (n_day >= 0) & (n_day <= day_columns.numel()), "nDay is outside the index length"
    )
    day_mask = (
        torch.zeros((ncol,), dtype=torch.int64, device=p_lev.device).index_add(
            0, day_columns.clamp(0, ncol - 1), valid.to(dtype=torch.int64)
        )
        > 0
    )

    htrsw, htrlw, htrswc, htrlwc, topfsw, sfcfsw, sfcflw, topflw = run_compile_island(
        "post",
        "final_j_tensor",
        _final_j_tensor_kernel,
        p_lev,
        fluxlw_up_all,
        fluxlw_down_all,
        fluxlw_up_clr,
        fluxlw_down_clr,
        fluxsw_up_all,
        fluxsw_down_all,
        fluxsw_up_clr,
        fluxsw_down_clr,
        day_columns,
        i_sfc=i_sfc,
        i_toa=i_toa,
        use_shortwave=use_shortwave,
        day_mask=day_mask,
    )

    fhswr = _copy_scalar_from(final_inputs, "fhswr", p_lev)
    fhlwr = _copy_scalar_from(final_inputs, "fhlwr", p_lev)
    errflg = torch.zeros((1,), dtype=torch.int64, device=p_lev.device)
    return RrtmgpPostResult(
        fields={
            "fhswr": fhswr,
            "fhlwr": fhlwr,
            "topfsw": topfsw,
            "htrsw": htrsw,
            "htrlw": htrlw,
            "htrswc": htrswc,
            "htrlwc": htrlwc,
            "sfcfsw": sfcfsw,
            "sfcflw": sfcflw,
            "topflw": topflw,
            "errmsg": "",
            "errflg": errflg,
        },
        accuracy_context={
            "p_lev": p_lev,
            "fluxlwUP_allsky": fluxlw_up_all,
            "fluxlwDOWN_allsky": fluxlw_down_all,
            "fluxlwUP_clrsky": fluxlw_up_clr,
            "fluxlwDOWN_clrsky": fluxlw_down_clr,
            "fluxswUP_allsky": fluxsw_up_all,
            "fluxswDOWN_allsky": fluxsw_down_all,
            "fluxswUP_clrsky": fluxsw_up_clr,
            "fluxswDOWN_clrsky": fluxsw_down_clr,
        },
    )
