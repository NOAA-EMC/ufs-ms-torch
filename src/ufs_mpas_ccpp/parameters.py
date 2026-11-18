# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Registry of the physics tuning parameters, each a multiplicative 0-d tensor scale.

A scale ``s`` multiplies the parameter's default where the scheme uses it, so ``s = 1`` is the unchanged model.
Physics scales enter a coupled run as one ``parameter_scales`` dict, {registry key: 0-d floating tensor}, shared by
the run's ``physics_arguments`` and ``microphysics_arguments``; they reach the boundaries as the ``parameter_scales``
keyword (``run_physics_sequence``, ``run_microphysics_sequence``, ``run_native_radiation``). Each scheme reads its
keys through ``select`` with its own map from registry key to scheme name, so an absent key adds no operation.
Dycore scales stay in the dycore registry (``PARAMETER_SCALE_KEYS``) and travel in ``config["parameter_scales"]``.
"""
from typing import Mapping

import torch

PHYSICS_PARAMETERS: tuple[str, ...] = (
    "samf.clam_deep", "samf.c0s_deep", "samf.pgcon_deep", "samf.clam_shal", "samf.evef", "samf.tau_deep",
    "samf.tau_shal", "samf.cinpcr_deep", "samf.cinpcr_shal", "samf.rhcrt_deep", "samf.rhcrt_shal",
    "pbl.xkzm_m", "pbl.xkzm_h", "pbl.bl_upfr", "pbl.ce0_updraft", "pbl.rlmx", "pbl.elmx",
    "sfc.charnock",
    "noahmp.czil", "noahmp.rsmin", "noahmp.rsurf", "noahmp.z0mvt",
    "tempo.nt_c_land", "tempo.nt_c_ocean", "tempo.demott_nuc_ssati", "tempo.av_s", "tempo.av_g",
    "rad.reliq_conv", "rad.reice_conv", "rad.alpha0", "rad.decorr",
    "gwd.alpha_fd", "gwd.dx_factor", "gwd.tauamp",
)


def scale(scales: Mapping[str, torch.Tensor] | None, key: str) -> torch.Tensor | None:
    """The scale tensor for a registered key, or None when it is absent; KeyError for an unregistered key."""
    if key not in PHYSICS_PARAMETERS:
        raise KeyError(f"unregistered parameter scale {key!r}")
    return None if scales is None else scales.get(key)


def select(scales: Mapping[str, torch.Tensor] | None, names: Mapping[str, str],
           like: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """The scales of ``names`` (registry key -> scheme name) present in ``scales``, by scheme name, in the order of
    ``names``; empty when none is present. With ``like``, as 1-element views on its device and dtype
    (``region_views``), for the arguments of a compiled region."""
    picked = {name: value for key, name in names.items() if (value := scale(scales, key)) is not None}
    return picked if like is None else region_views(picked, like)


def region_view(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """One scale as a 1-element view on ``like``'s device and dtype (see ``region_views``)."""
    return value.reshape(1).to(device=like.device, dtype=like.dtype)


def region_views(scales: Mapping[str, torch.Tensor] | None, like: torch.Tensor) -> dict[str, torch.Tensor] | None:
    """``scales`` as 1-element views on ``like``'s device and dtype, for the arguments of a compiled region.

    A 0-d CPU tensor argument costs a host read in Dynamo's guard evaluation (``aten::item`` under ``TorchDynamo
    Cache Lookup``) on every call, and a device sync on CUDA; a 1-element view on the numerical device costs
    neither. The region takes each value back with ``reshape(())``. None and an empty mapping stay as they are, so
    an absent key adds no op and no tensor argument; a region with a slot for that scale receives the constant None
    there (Dynamo guards it with ``is None``)."""
    if not scales:
        return scales
    return {name: region_view(value, like) for name, value in scales.items()}
