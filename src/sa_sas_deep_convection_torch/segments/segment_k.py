# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment K: velocity averages, sigma masks, cloud-top split, and downdraft seed."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch import recurrence_chunks as chunks
from sa_sas_deep_convection_torch.compile import (
    masked_sum_chunk_size, run_compile_island, run_recurrence_chunk, recurrence_chunk_size,
    ordered_chunks_enabled,
)
from sa_sas_deep_convection_torch.constants import KM
from sa_sas_deep_convection_torch.ops import sqrt_zero_grad
from sa_sas_deep_convection_torch.segments.segment_d import _gather_column
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real

CHECKPOINT_FIELDS = ("omegac", "sigmab", "scaldfunc", "sigmagfm", "xmbmax", "ktcon", "ktcon1")


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run lines 1901-2083 of ``samfdeepcnv_run`` for the approved active case."""
    _, KM, NN, NTR, NTC = runtime_dimensions(state)

    cnvflg = state.require("cnvflg")
    kb = state.require("kb")
    kbcon = state.require("kbcon")
    kbcon1 = state.require("kbcon1")
    ktcon = state.require("ktcon").clone()
    ktcon1 = state.require("ktcon1").clone()
    kmax = state.require("kmax")
    kd94 = state.require("kd94")
    zi = state.require("zi")
    zo = state.require("zo")
    qeso = state.require("qeso")
    qo = state.require("qo")
    qcko = state.require("qcko").clone()
    zdqca = state.require("zdqca").clone()
    dbyo = state.require("dbyo")
    uo = state.require("uo")
    vo = state.require("vo")
    eta = state.require("eta")
    etad = state.require("etad").clone()
    xlamdet = state.require("xlamdet")
    xlamddt = state.require("xlamddt")
    xlamd = state.require("xlamd").clone()
    vshear = state.require("vshear").clone()
    edt = state.require("edt").clone()
    edto = state.require("edto").clone()
    edtx = state.require("edtx").clone()
    qlko_ktcon = state.require("qlko_ktcon").clone()
    omegac = state.require("omegac").clone()
    omegac_k_checkpoint = omegac.clone()
    sigmab = state.require("sigmab").clone()
    scaldfunc = state.require("scaldfunc").clone()
    sigmagfm = state.require("sigmagfm").clone()
    scaldfunc_k_checkpoint = torch.zeros_like(scaldfunc) if bool(state.config.get("hwrf_samfdeep", False)) else scaldfunc.clone()
    sigmagfm_k_checkpoint = torch.zeros_like(sigmagfm) if bool(state.config.get("hwrf_samfdeep", False)) else sigmagfm.clone()
    ncol = cnvflg.shape[0]
    for name, tensor in {
        "cnvflg": cnvflg,
        "kb": kb,
        "kbcon": kbcon,
        "kbcon1": kbcon1,
        "ktcon": ktcon,
        "ktcon1": ktcon1,
        "kmax": kmax,
        "kd94": kd94,
    }.items():
        require_real(name, tensor, (ncol,))
    for name, tensor in {
        "zi": zi,
        "zo": zo,
        "qeso": qeso,
        "qo": qo,
        "qcko": qcko,
        "zdqca": zdqca,
        "dbyo": dbyo,
        "uo": uo,
        "vo": vo,
        "eta": eta,
        "etad": etad,
    }.items():
        require_real(name, tensor, (ncol, KM))

    (
        cnvflg,
        omegac,
        zeta,
        ktcon,
        ktcon1,
        qcko,
        zdqca,
        qlko_ktcon,
        vshear,
        edt,
        edto,
        edtx,
        sumx,
        xlamd,
        etad,
    ) = _recurrence_coordinator(
        "k_shear_downdraft_seed",
        "shear_downdraft_seed",
        _shear_downdraft_seed_block,
        cnvflg,
        kb,
        kbcon1,
        ktcon,
        ktcon1,
        kmax,
        kd94,
        state.require("jmin"),
        zi,
        qeso,
        qcko,
        zdqca,
        dbyo,
        uo,
        vo,
        eta,
        etad,
        xlamdet,
        xlamddt,
        xlamd,
        vshear,
        edt,
        edto,
        edtx,
        qlko_ktcon,
        omegac,
        state.require("zeta"),
        state.require("sumx"),
        state.require("del"),
        state.require("omega_u"),
        state.require("to"),
        state.require("islimsk"),
        state.require("betal"),
        state.require("betas"),
        state.require("el2orc"),
        state.require("hvap"),
        bool(state.config.get("progsigma", False)),
        int(state.config.get("ncloud", 0)),
    )

    return state.with_updates(
        {
            "omegac": omegac,
            "omegac_k_checkpoint": omegac_k_checkpoint,
            "sigmab": sigmab,
            "scaldfunc": scaldfunc,
            "scaldfunc_k_checkpoint": scaldfunc_k_checkpoint,
            "sigmagfm": sigmagfm,
            "sigmagfm_k_checkpoint": sigmagfm_k_checkpoint,
            "zeta": zeta,
            "ktcon": ktcon,
            "ktcon1": ktcon1,
            "qcko": qcko,
            "zdqca": zdqca,
            "qlko_ktcon": qlko_ktcon,
            "vshear": vshear,
            "edt": edt,
            "edto": edto,
            "edtx": edtx,
            "sumx": sumx,
            "xlamd": xlamd,
            "etad": etad,
        }
    )


def _recurrence_coordinator(category, name, fn, *args):
    if ordered_chunks_enabled("k_ordered_chunks"):
        return fn(*args)
    return run_compile_island(category, name, fn, *args)


def _shear_downdraft_seed_block(
    cnvflg: torch.Tensor,
    kb: torch.Tensor,
    kbcon1: torch.Tensor,
    ktcon: torch.Tensor,
    ktcon1: torch.Tensor,
    kmax: torch.Tensor,
    kd94: torch.Tensor,
    jmin: torch.Tensor,
    zi: torch.Tensor,
    qeso: torch.Tensor,
    qcko: torch.Tensor,
    zdqca: torch.Tensor,
    dbyo: torch.Tensor,
    uo: torch.Tensor,
    vo: torch.Tensor,
    eta: torch.Tensor,
    etad: torch.Tensor,
    xlamdet: torch.Tensor,
    xlamddt: torch.Tensor,
    xlamd: torch.Tensor,
    vshear: torch.Tensor,
    edt: torch.Tensor,
    edto: torch.Tensor,
    edtx: torch.Tensor,
    qlko_ktcon: torch.Tensor,
    omegac: torch.Tensor,
    zeta: torch.Tensor,
    sumx_input: torch.Tensor,
    del_: torch.Tensor,
    omega_u: torch.Tensor,
    to: torch.Tensor,
    islimsk: torch.Tensor,
    betal: torch.Tensor,
    betas: torch.Tensor,
    el2orc: torch.Tensor,
    hvap: torch.Tensor,
    progsigma: bool,
    ncloud: int,
) -> tuple[torch.Tensor, ...]:
    KM = zi.shape[1]
    chunk_size = recurrence_chunk_size()
    levels = torch.arange(1, KM + 1, dtype=kb.dtype, device=kb.device)
    active = cnvflg != 0.0
    if progsigma:
        omegac_work = torch.zeros_like(omegac)
        sumx_work = torch.zeros_like(sumx_input)
        sigma_rows = chunks.pack_profiles(del_, omega_u)
        sum_size = masked_sum_chunk_size()
        for start in range(1, KM - 1, sum_size):
            end = min(start + sum_size, KM - 1)
            omegac_work, sumx_work = run_recurrence_chunk(
                "k_ordered_chunks", "sigma_average", chunks.sigma_average,
                active, kbcon1, ktcon, levels[start], sigma_rows[start - 1:end],
                omegac_work, sumx_work,
            )
        del sigma_rows
        valid_sum = active & (sumx_work != 0.0)
        omegac = torch.where(valid_sum, omegac_work / sumx_work, omegac)
        cnvflg = torch.where(active & (sumx_work == 0.0), torch.zeros_like(cnvflg), cnvflg)
        cnvflg = torch.where((cnvflg != 0.0) & (omegac > -1.2), torch.zeros_like(cnvflg), cnvflg)
        active = cnvflg != 0.0
        layer = (active[:, None] & (levels[None, :] >= kbcon1[:, None])
                 & (levels[None, :] < ktcon[:, None])
                 & (levels[None, :] > 1.0) & (levels[None, :] < float(KM)))
        zeta_candidate = torch.where(
            omega_u != 0.0,
            eta * (omegac[:, None] / torch.where(omega_u != 0.0, omega_u, 1.0)),
            torch.zeros_like(omega_u),
        )
        zeta = torch.where(layer, torch.clamp(zeta_candidate, min=0.0, max=1.0), zeta)

    old_ktcon = ktcon.clone()
    ktcon = torch.where(active, ktcon1, ktcon)
    ktcon1 = torch.where(active, old_ktcon, ktcon1)

    if ncloud > 0:
        # Fortran evaluates the cloud-top split only for cnvflg columns.
        # An inactive column can retain ktcon=1, whose ktcon-1 is an unset
        # one-based index. Supply a finite operand there before evaluating
        # the masked expression; active indices are never clamped.
        top_idx = torch.where(active, ktcon - 1.0, torch.ones_like(ktcon))
        top_temperature = torch.where(active, _gather_column(to, top_idx), torch.ones_like(ktcon))
        gamma = el2orc * _gather_column(qeso, top_idx) / (top_temperature ** 2)
        qrch = _gather_column(qeso, top_idx) + gamma * _gather_column(dbyo, top_idx) / (hvap * (1.0 + gamma))
        dq = _gather_column(qcko, top_idx) - qrch
        split = active & (dq > 0.0)
        qlko_ktcon = torch.where(split, dq, qlko_ktcon)
        layer = split[:, None] & (top_idx[:, None] == levels[None, :])
        qcko = torch.where(layer, qrch[:, None], qcko)
        zdqca = torch.where(layer, dq[:, None], zdqca)

    vshear = torch.where(active, torch.zeros_like(vshear), vshear)
    shear_rows = chunks.pack_profiles(uo, vo)
    sum_size = masked_sum_chunk_size()
    for start in range(1, KM, sum_size):
        end = min(start + sum_size, KM)
        vshear = run_recurrence_chunk(
            "k_ordered_chunks", "shear_sum", chunks.shear_sum,
            active, kb, ktcon, levels[start], shear_rows[start - 1:end], vshear,
        )
    del shear_rows
    depth = _gather_column(zi, ktcon) - _gather_column(zi, kb)
    vshear_active = 1.0e3 * vshear / torch.where(active, depth, torch.ones_like(depth))
    shear_squared = vshear_active * vshear_active
    shear_cubed = shear_squared * vshear_active
    e1 = 1.591 - 0.639 * vshear_active + 0.0953 * shear_squared - 0.00496 * shear_cubed
    edt_candidate = torch.clamp(1.0 - e1, min=0.0, max=0.9)
    edt = torch.where(active, edt_candidate, edt)
    edto = torch.where(active, edt, edto)
    edtx = torch.where(active, edt, edtx)

    sumx = torch.where(active, torch.zeros_like(sumx_input), sumx_input)
    depth_rows = chunks.pack_profiles(zi)
    for start in range(0, KM - 1, sum_size):
        end = min(start + sum_size, KM - 1)
        sumx = run_recurrence_chunk(
            "k_ordered_chunks", "downdraft_depth", chunks.downdraft_depth,
            active, kd94, levels[start], depth_rows[start:end + 1], sumx,
        )
    del depth_rows
    beta = torch.where(islimsk == 1.0, betal.expand_as(cnvflg), betas.expand_as(cnvflg))
    dz_mean = (sumx + zi[:, 0]) / kd94
    xlamd_candidate = (1.0 - beta ** (1.0 / kd94)) / dz_mean
    xlamd = torch.where(active, xlamd_candidate, xlamd)

    pieces = []
    carry = etad[:, -1].contiguous()
    eta_rows = chunks.pack_profiles(zi, etad)
    end = KM - 1
    while end > 0:
        start = max(0, end - chunk_size)
        values = run_recurrence_chunk(
            "k_ordered_chunks", "downdraft_eta", chunks.downdraft_eta,
            active, kmax, kd94, jmin, levels[start], eta_rows[start:end + 1],
            carry, xlamd, xlamddt, xlamdet,
        )
        pieces.append(values)
        carry = values[:, 0]
        end = start
    etad = chunks.join_levels(etad, pieces[::-1], 0)
    return cnvflg, omegac, zeta, ktcon, ktcon1, qcko, zdqca, qlko_ktcon, vshear, edt, edto, edtx, sumx, xlamd, etad


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {
        name: state.fields.get("omegac_k_checkpoint", state.require("omegac"))
        if name == "omegac"
        else state.fields.get("scaldfunc_k_checkpoint", state.require("scaldfunc"))
        if name == "scaldfunc"
        else state.fields.get("sigmagfm_k_checkpoint", state.require("sigmagfm"))
        if name == "sigmagfm"
        else state.require(name)
        for name in CHECKPOINT_FIELDS
    }
