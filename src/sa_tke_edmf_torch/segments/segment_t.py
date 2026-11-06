# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment T: post-interstitial tendency application and output fields."""

from __future__ import annotations

import torch

from sa_tke_edmf_torch.constants import IM, KM, NTQV, NTRAC
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_float64
from sa_tke_edmf_torch.tracers import specific_humidity_index
from sa_tke_edmf_torch.diagnostics import diagnostic_slots, set_diagnostic_

CHECKPOINT_FIELDS = ("gt0", "gq0", "gu0", "gv0", "dqdt", "dtdt", "dudt", "dvdt", "ten_t_pbl", "ten_q_pbl")


def run(state: SaTkeEdmfState) -> SaTkeEdmfState:
    """``GFS_PBL_generic_post_run`` tendency handling.

    ``config["tend_opt_pbl"]`` (default 1) selects the oracle ``case_PBL_ten`` branch:
    1 apply now, 2 add to the dtdt/dudt/dvdt/dqdt sums, 3 add to the sums and apply
    them, 4 leave state and sums unchanged. With ``ldiag3d`` the PBL slices of dtend
    are overwritten using source dtidx and globally unpacked tracers. Profiles
    without dtidx retain their declared 3+n layout.
    """

    KM = state.require("q1").shape[1]
    NTRAC = state.require("q1").shape[2]
    # Modes 1 and 3 make new state tensors; modes 2 and 4 return the inputs unchanged (no copies).
    gt0 = state.require("gt0")
    gq0 = state.require("gq0")
    gu0 = state.require("gu0")
    gv0 = state.require("gv0")
    ten_t = state.require("ten_t")
    ten_u = state.require("ten_u")
    ten_v = state.require("ten_v")
    ten_q = state.require("ten_q")
    delt = state.require("delt")
    IM = gt0.shape[0]
    require_float64("gt0", gt0, (IM, KM))
    require_float64("gq0", gq0, (IM, KM, NTRAC))
    require_float64("gu0", gu0, (IM, KM))
    require_float64("gv0", gv0, (IM, KM))
    require_float64("ten_t", ten_t, (IM, KM))
    require_float64("ten_u", ten_u, (IM, KM))
    require_float64("ten_v", ten_v, (IM, KM))
    require_float64("ten_q", ten_q, (IM, KM, NTRAC))

    dtype = gt0.dtype
    device = gt0.device
    dtdt = state.fields.get("dtdt")
    if dtdt is None:
        dtdt = torch.zeros((IM, KM), dtype=dtype, device=device)
    dudt = state.fields.get("dudt")
    if dudt is None:
        dudt = torch.zeros((IM, KM), dtype=dtype, device=device)
    dvdt = state.fields.get("dvdt")
    if dvdt is None:
        dvdt = torch.zeros((IM, KM), dtype=dtype, device=device)
    dqdt = state.fields.get("dqdt")
    if dqdt is None:
        dqdt = torch.zeros((IM, KM, NTRAC), dtype=dtype, device=device)
    require_float64("dtdt", dtdt, (IM, KM))
    require_float64("dudt", dudt, (IM, KM))
    require_float64("dvdt", dvdt, (IM, KM))
    require_float64("dqdt", dqdt, (IM, KM, NTRAC))

    dtp = delt[0]
    tend_opt_pbl = int(state.config.get("tend_opt_pbl", 1))
    if tend_opt_pbl == 1:
        gt0 = gt0 + dtp * ten_t
        gu0 = gu0 + dtp * ten_u
        gv0 = gv0 + dtp * ten_v
        gq0 = gq0 + dtp * ten_q
    elif tend_opt_pbl == 2:
        dtdt = dtdt + ten_t
        dudt = dudt + ten_u
        dvdt = dvdt + ten_v
        dqdt = dqdt + ten_q
    elif tend_opt_pbl == 3:
        gt0 = gt0 + dtp * (dtdt + ten_t)
        gu0 = gu0 + dtp * (dudt + ten_u)
        gv0 = gv0 + dtp * (dvdt + ten_v)
        gq0 = gq0 + dtp * (dqdt + ten_q)
        dtdt = torch.zeros_like(dtdt)
        dudt = torch.zeros_like(dudt)
        dvdt = torch.zeros_like(dvdt)
        dqdt = torch.zeros_like(dqdt)
    elif tend_opt_pbl != 4:
        raise ValueError(f"tend_opt_pbl must be 1-4, got {tend_opt_pbl}")

    updates = {
        "gt0": gt0,
        "gq0": gq0,
        "gu0": gu0,
        "gv0": gv0,
        "dtdt": dtdt,
        "dudt": dudt,
        "dvdt": dvdt,
        "dqdt": dqdt,
        "ten_t_pbl": ten_t,
        "ten_q_pbl": ten_q[:, :, specific_humidity_index(state.config) - 1],
    }
    dtend = state.fields.get("dtend")
    if bool(state.config.get("ldiag3d", False)) and dtend is not None:
        slots = diagnostic_slots(state)
        temperature_dt = dtp
        if bool(state.config.get("lsidea", False)):
            dtf = state.require("dtf")
            require_float64("dtf", dtf)
            if dtf.ndim != 0:
                raise ValueError("PBL post dtf requires a scalar tensor")
            temperature_dt = dtf
        dtend = dtend.clone()
        set_diagnostic_(dtend, ten_t * temperature_dt, slots[0])
        set_diagnostic_(dtend, ten_u * dtp, slots[1])
        set_diagnostic_(dtend, ten_v * dtp, slots[2])
        for n in range(NTRAC):
            set_diagnostic_(dtend, ten_q[:, :, n] * dtp, slots[3 + n])
        updates["dtend"] = dtend
    return state.with_updates(updates)


def checkpoint_fields(state: SaTkeEdmfState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
