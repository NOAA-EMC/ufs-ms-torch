# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Selected R4 physics cap after its prefix, before the native state finalizer."""

from dataclasses import dataclass

import torch

from . import parameters
from .gwd import apply_gwd_pre
from .gwd_post import apply_gwd_post
from .gwd_unified import apply_unified_gwd
from .gwd_v1 import apply_ugwpv1_gsldrag
from .land import noahmp_parameter_scales
from .pbl import run_pbl_sequence
from .physics_prefix import PhysicsPrefixResult
from .runtime_policy import scheme
from .surface_post import run_surface_sequence

# Registry key -> name: the surface layer's Charnock scale, and the GWD scales (alpha_fd scales the TOFD
# coefficient on both GWD paths; dx_factor_scale and tauamp_scale are ugwpv1_gsldrag arguments).
SURFACE_PARAMETER_KEYS = {"sfc.charnock": "charnock"}
GWD_PARAMETER_KEYS = {"gwd.alpha_fd": "alpha_fd", "gwd.dx_factor": "dx_factor_scale", "gwd.tauamp": "tauamp_scale"}

@dataclass(frozen=True)
class PhysicsSequenceResult:
    prefix: PhysicsPrefixResult
    hpbl: torch.Tensor
    noah_water_budgets: tuple[dict[str, torch.Tensor], ...]


def run_physics_sequence(statein, grid, prefix, hpbl, noah_case, *,
    solhr, table, dtp, dtf, julian, yearlen, surface_config, nsst_control,
    surface_post_control, huge, dtidx, pbl_control, gwd_control, cdmbgwd, alpha_fd, forecast_day=None,
    parameter_scales=None):
    """Run two surface passes, PBL and GWD once, with explicit persistent carry.

    The caller prepares radiation and the seven-call physics prefix. This call
    does not initialize cold state, finalize Stateout or commit to the dycore.
    advance_physics_dynamics owns that finalizer and the first native commit.
    """
    water_budgets = []
    # Scales (0-d tensors) reach the schemes as tensor arguments.
    charnock = parameters.select(parameter_scales, SURFACE_PARAMETER_KEYS).get("charnock")
    surface = run_surface_sequence(statein, grid, prefix, hpbl, noah_case,
        solhr=solhr, table=table, delt=dtf, julian=julian, yearlen=yearlen,
        surface_config=surface_config, nsst_control=nsst_control,
        post_control=surface_post_control, huge=huge, water_budgets=water_budgets, charnock_scale=charnock,
        noahmp_scales=noahmp_parameter_scales(parameter_scales))
    pbl = scheme(run_pbl_sequence)(statein, grid, surface, dtp=dtp, dtf=dtf,
        dtidx=dtidx, control=pbl_control, parameter_scales=parameter_scales)
    prepared = apply_gwd_pre(pbl.prefix)
    # gwd.dx_factor and gwd.tauamp have sites on the GFS v17 path only (R4 runs UGWP v0 without LS/BL drag); the gates
    # judge them on the stored v17 case.
    gwd = parameters.select(parameter_scales, GWD_PARAMETER_KEYS)
    if "alpha_fd" in gwd:
        # TOFD alpha_fd*s, on alpha_fd's device and dtype (drag_suite_psl validates both); both GWD paths.
        alpha_fd = alpha_fd * gwd["alpha_fd"].to(device=alpha_fd.device, dtype=alpha_fd.dtype)
    if gwd_control.get("do_ugwp_v1", False):
        # GFS v17: ugwpv1_gsldrag; the tau file is read at the GFS_phys_time_vary forecast day.
        # gwd.dx_factor and gwd.tauamp have sites only here: R4 has no LS/BL drag and a fixed v0 wave amplitude.
        v1_scales = {name: gwd[name] for name in ("dx_factor_scale", "tauamp_scale") if name in gwd}
        drag = scheme(apply_ugwpv1_gsldrag)(statein, grid, prepared, pbl.hpbl, dtp=dtp,
            alpha_fd=alpha_fd, dtidx=dtidx, control=gwd_control, forecast_day=forecast_day, **v1_scales)
    else:
        drag = scheme(apply_unified_gwd)(statein, grid, prepared, pbl.hpbl,
            dtp=dtp, cdmbgwd=cdmbgwd, alpha_fd=alpha_fd, dtidx=dtidx, control=gwd_control)
    post = apply_gwd_post(drag, dtp=dtp, dtf=dtf, dtidx=dtidx, control=gwd_control)
    return PhysicsSequenceResult(post, pbl.hpbl, tuple(water_budgets))
