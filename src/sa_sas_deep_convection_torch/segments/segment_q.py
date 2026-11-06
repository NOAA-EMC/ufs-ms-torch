# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment Q: progsigma helper inputs and inactive-path sigma state."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch.compile import run_compile_island
from sa_sas_deep_convection_torch.constants import KM
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real

CHECKPOINT_FIELDS = ("qadv", "tmfq", "qmicro", "omega_u", "zeta", "sigmain", "sigmaout", "sigmab")


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 2960-3017 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    cnvflg = state.require("cnvflg")
    q = state.require("q")
    prevsq = state.require("prevsq")
    tmf = state.require("tmf")
    qmicro = state.require("qmicro")
    omega_u = state.require("omega_u")
    zeta = state.require("zeta")
    sigmain = state.require("sigmain")
    sigmaout = state.require("sigmaout").clone()
    sigmab = state.require("sigmab").clone()
    ncol = cnvflg.shape[0]
    require_real("cnvflg", cnvflg, (ncol,))
    for name, tensor in {
        "q": q,
        "prevsq": prevsq,
        "qmicro": qmicro,
        "omega_u": omega_u,
        "zeta": zeta,
        "sigmain": sigmain,
        "sigmaout": sigmaout,
    }.items():
        require_real(name, tensor, (ncol, KM))

    qadv, tmfq, sigmaout, sigmab = run_compile_island(
        "q_progsigma_inputs",
        "progsigma_inputs",
        _progsigma_inputs_block,
        cnvflg,
        q,
        prevsq,
        tmf,
        sigmaout,
        sigmab,
        state.require("delt"),
        bool(state.config.get("progsigma", False)),
        bool(state.config.get("first_time_step", False)),
        bool(state.config.get("restart", False)),
        bool(state.config.get("sigmab_coldstart", False)),
    )

    return state.with_updates({"qadv": qadv, "tmfq": tmfq, "sigmaout": sigmaout, "sigmab": sigmab})


def _progsigma_inputs_block(
    cnvflg: torch.Tensor,
    q: torch.Tensor,
    prevsq: torch.Tensor,
    tmf: torch.Tensor,
    sigmaout: torch.Tensor,
    sigmab: torch.Tensor,
    delt: torch.Tensor,
    progsigma: bool,
    first_time_step: bool,
    restart: bool,
    sigmab_coldstart: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qadv = torch.zeros_like(q)
    tmfq = torch.zeros_like(q)
    if progsigma:
        if first_time_step and ((not restart) or sigmab_coldstart):
            qadv = torch.zeros_like(q)
        else:
            qadv = (q - prevsq) / delt
        tmfq = tmf[:, :, 0].clone()
        sigmaout = torch.where((cnvflg != 0.0)[:, None], sigmaout, sigmaout)
        sigmab = torch.where(cnvflg != 0.0, sigmab, sigmab)
    return qadv, tmfq, sigmaout, sigmab


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
