# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Selected R4 physics cap after its prefix, before the native state finalizer."""

from dataclasses import dataclass

import torch

from .gwd import apply_gwd_pre
from .gwd_post import apply_gwd_post
from .gwd_unified import apply_unified_gwd
from .gwd_v1 import apply_ugwpv1_gsldrag
from .pbl import run_pbl_sequence
from .physics_prefix import PhysicsPrefixResult
from .runtime_policy import scheme
from .surface_post import run_surface_sequence


@dataclass(frozen=True)
class PhysicsSequenceResult:
    prefix: PhysicsPrefixResult
    hpbl: torch.Tensor
    noah_water_budgets: tuple[dict[str, torch.Tensor], ...]


def run_physics_sequence(statein, grid, prefix, hpbl, noah_case, *,
    solhr, table, dtp, dtf, julian, yearlen, surface_config, nsst_control,
    surface_post_control, huge, dtidx, pbl_control, gwd_control, cdmbgwd, alpha_fd, forecast_day=None):
    """Run two surface passes, PBL and GWD once, with explicit persistent carry.

    The caller prepares radiation and the seven-call physics prefix. This call
    does not initialize cold state, finalize Stateout or commit to the dycore.
    advance_physics_dynamics owns that finalizer and the first native commit.
    """
    water_budgets = []
    surface = run_surface_sequence(statein, grid, prefix, hpbl, noah_case,
        solhr=solhr, table=table, delt=dtf, julian=julian, yearlen=yearlen,
        surface_config=surface_config, nsst_control=nsst_control,
        post_control=surface_post_control, huge=huge, water_budgets=water_budgets)
    pbl = scheme(run_pbl_sequence)(statein, grid, surface, dtp=dtp, dtf=dtf,
        dtidx=dtidx, control=pbl_control)
    prepared = apply_gwd_pre(pbl.prefix)
    if gwd_control.get("do_ugwp_v1", False):
        # GFS v17: ugwpv1_gsldrag; the tau file is read at the GFS_phys_time_vary forecast day.
        drag = scheme(apply_ugwpv1_gsldrag)(statein, grid, prepared, pbl.hpbl, dtp=dtp,
            alpha_fd=alpha_fd, dtidx=dtidx, control=gwd_control, forecast_day=forecast_day)
    else:
        drag = scheme(apply_unified_gwd)(statein, grid, prepared, pbl.hpbl,
            dtp=dtp, cdmbgwd=cdmbgwd, alpha_fd=alpha_fd, dtidx=dtidx, control=gwd_control)
    post = apply_gwd_post(drag, dtp=dtp, dtf=dtf, dtidx=dtidx, control=gwd_control)
    return PhysicsSequenceResult(post, pbl.hpbl, tuple(water_budgets))
