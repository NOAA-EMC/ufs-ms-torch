# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Source process-ledger routing for PBL diagnostic increments."""

from __future__ import annotations

import torch
from .autodiff import static_tensor

from .state import SaTkeEdmfState, require_float64


def diagnostic_slots(state: SaTkeEdmfState) -> torch.Tensor:
    """Return source one-based T/U/V and packed-tracer slots; nonpositive means off."""
    dtend = state.require("dtend")
    ncol, nlev, ntrac = state.require("q1").shape
    # ldiag3d off: no slots are allocated (ndtend 0, as GFS_typedefs) and nothing is written.
    least = 1 if bool(state.config.get("ldiag3d", False)) else 0
    if dtend.ndim != 3 or dtend.shape[:2] != (ncol, nlev) or dtend.shape[2] < least:
        raise ValueError("PBL dtend requires [ncol,nlev,ndtend]")
    require_float64("dtend", dtend)
    if "dtidx" not in state.fields:
        slots = torch.arange(1, 4 + ntrac, dtype=torch.int64, device=dtend.device)
    else:
        dtidx = state.require("dtidx")
        if dtidx.dtype not in (torch.int32, torch.int64) or dtidx.device != dtend.device:
            raise ValueError("PBL dtidx must be device-local integers")
        process = state.config["index_of_process_pbl"] - 1
        rows = [state.config[n] - 1 for n in
                ("index_of_temperature", "index_of_x_wind", "index_of_y_wind")]
        if dtidx.ndim != 2 or min(rows) < 0 or max(rows) >= dtidx.shape[0] or not 0 <= process < dtidx.shape[1] or dtidx.shape[0] < 100 + ntrac:
            raise ValueError("invalid source PBL diagnostic indices")
        slots = torch.cat((dtidx[static_tensor(rows, dtype=torch.int64, device=dtidx.device), process], dtidx[100:100+ntrac, process])).to(torch.int64)
    torch._assert_async((slots <= dtend.shape[2]).all(), "PBL diagnostic slot outside allocation")
    return slots


def _slot_index(dtend: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    # Nonpositive slots are off; they read and rewrite column 0 unchanged.
    torch._assert_async(slot <= dtend.shape[2], "diagnostic slot outside allocation")
    return (slot.clamp(min=1) - 1).to(torch.int64).reshape(1)


def add_diagnostic_(dtend: torch.Tensor, increment: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    """Source ``dtend(:,:,idtend) += increment`` in place, reading and writing only the selected slot.

    ``dtend`` must be a buffer the caller owns: one ``clone()`` per caller, not per slot. An
    out-of-place update copies all ``ndtend`` slots for every increment.
    """
    index = _slot_index(dtend, slot)
    current = dtend.index_select(2, index)
    return dtend.index_copy_(2, index, torch.where(slot >= 1, current + increment[:, :, None], current))


def set_diagnostic_(dtend: torch.Tensor, value: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    """Source ``dtend(:,:,idtend) = value`` in place on a caller-owned buffer (see ``add_diagnostic_``)."""
    index = _slot_index(dtend, slot)
    current = dtend.index_select(2, index)
    return dtend.index_copy_(2, index, torch.where(slot >= 1, value[:, :, None], current))
