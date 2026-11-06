# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Integrated SA-SAS shallow-convection PyTorch driver for cumulative port gates."""

from __future__ import annotations

from typing import Any

import torch

from sa_sas_shallow_convection_torch import recurrence_chunks
from ufs_physics_common.ordered_sum import parallel_masked_sums

from sa_sas_shallow_convection_torch.compile import (
    DEFAULT_COMPILE_CATEGORIES,
    compile_categories,
    compile_categories_explicitly_set,
    run_compile_island,
    selected_recurrence_chunk_size,
)
from sa_sas_shallow_convection_torch.constants import (
    ASOLFAC,
    BETADCU,
    BETAMCU,
    BETASCU,
    C0S,
    C1,
    CAT_ADJ_SHAL,
    CLIQ,
    CLAM,
    CP,
    CSCALE,
    CVAP,
    DELT,
    EPS,
    EPSM1,
    EVEF,
    FV,
    GRAV,
    HVAP,
    IMPLEMENTED_SEGMENTS,
    IM,
    ITC as ITC_DEFAULT,
    KM,
    NCLOUD,
    NDTEND,
    NN,
    NTC,
    NTK,
    NTR,
    PGCON,
    RD,
    RV,
    SEGMENT_IDS,
    T0C,
)
from sa_sas_shallow_convection_torch.state import SaSasShallowConvectionState, require_real, require_same_device
from sa_sas_deep_convection_torch.samf_interstitials import (
    IMP_PHYSICS_CODES,
    ConvectionTracerLayout,
    build_clw,
    interface_pressure,
    layout_from_config,
    legacy_layout,
    mpas_tempo_layout,
    samf_water_reconciliation,
    tendencies_from_dclw,
)
from sa_sas_shallow_convection_torch.thermo import fpvs, _fpvs_lookup, _fpvs_table_for_device


BOUNDARY_FIELDS: dict[str, tuple[str, ...]] = {
    "A": ("errflg", "errmsg", "ten_t", "ten_u", "ten_v", "ten_q", "dqtr", "new_t1", "new_q1", "new_qtr"),
    "B": ("do_aerosols", "ps", "prsl", "del", "cnvflg", "c0", "c0t", "cnvw", "cnvc", "ud_mf", "dt_mf"),
    "C": ("zo", "zi", "qeso", "heo", "heso", "to", "qo", "uo", "vo", "kpbl", "kbm", "kmax"),
    "D": ("kb", "kb1", "kbcon", "cnvflg", "pfld", "po", "to", "qo", "heo", "heso"),
    "E": ("pdot", "cina", "rhbar", "tkemean", "clamt", "xlamue", "xlamud", "cnvflg", "return_stage"),
    "F": ("eta", "hcko", "qcko", "ucko", "vcko", "ecko", "ercko", "chem_c", "chem_pw", "xlamue"),
    "G": ("kbcon1", "ktcon", "ktcon1", "xmbmax", "dbyo", "cnvflg", "return_stage"),
    "H": ("qcko", "qrcko", "pwo", "cnvwt", "dellal", "zdqca", "buo", "drag", "aa1", "dbyo1"),
    "I": ("ktcon1", "qcko", "qrcko", "pwo", "cnvwt", "wu2", "omega_u", "omegaout", "cnvflg"),
    "J": ("wc", "omegac", "zeta", "ktcon", "ktcon1", "qlko_ktcon", "qcko", "zdqca", "cnvflg"),
    "K": ("dellah", "dellaq", "dellau", "dellav", "dellal", "dellae", "delubar", "delvbar"),
    "L": ("q_diff", "e_diff", "flxtvd", "dtconv", "umean", "advfac", "dellaq", "dellae"),
    "M": ("qadv", "tmfq", "sigmaout", "sigmab", "xmb", "sigmagfm", "scaldfunc", "new_t1", "new_q1", "new_u1", "new_v1"),
    "N": ("new_q1", "new_qtr", "ctr", "wet_dep", "tsumn", "tsump", "rtnp", "delebar"),
    "O": ("qeso", "rntot", "rn", "qevap", "delqev", "delqbar", "deltbar", "new_t1", "new_q1"),
    "P": ("rn", "kbot", "ktop", "kcnv", "cnvw", "cnvc", "ud_mf", "dt_mf", "new_qtr", "ten_t", "ten_u", "ten_v", "ten_q", "dqtr"),
    "Q": ("gt0", "gu0", "gv0", "gq0", "dtdt", "dudt", "dvdt", "dqdt", "dtend", "rainc", "cnvprcp", "cnvw_phy_f3d", "cnvc_phy_f3d", "ten_q", "errflg"),
}

PRODUCTION_FIELDS: tuple[str, ...] = (
    "gt0",
    "gu0",
    "gv0",
    "gq0",
    "dtdt",
    "dudt",
    "dvdt",
    "dqdt",
    "dtend",
    "rainc",
    "cnvprcp",
    "cnvw_phy_f3d",
    "cnvc_phy_f3d",
    "ten_q",
    "ten_t",
    "ten_u",
    "ten_v",
    "dqtr",
    "kcnv",
    "kbot",
    "ktop",
    "rn",
    "ud_mf",
    "dt_mf",
    "cnvw",
    "cnvc",
    "errflg",
    "errmsg",
)

_ORDER = {segment_id: index for index, segment_id in enumerate(SEGMENT_IDS)}

_INPUT_GROUPS = {
    "scalar": ("cliq", "cp", "cvap", "eps", "epsm1", "fv", "grav", "hvap", "rd", "rv", "t0c", "delt", "cscale", "clam", "c0s", "c1", "evef", "pgcon", "asolfac", "betadcu", "betamcu", "betascu", "cat_adj_shal"),
    "column": ("psp", "rn", "garea", "hpbl", "kcnv", "kbot", "ktop", "islimsk", "rainc", "cnvprcp"),
    "level": ("qmicro", "delp", "ten_t", "ten_u", "ten_v", "prslp", "phil", "tkeh", "prevsq", "q", "q1", "t1", "u1", "v1", "dot", "ud_mf", "dt_mf", "cnvw", "cnvc", "sigmain", "sigmaout", "omegain", "omegaout", "gt0", "gu0", "gv0", "dudt", "dvdt", "dtdt", "cnvw_phy_f3d", "cnvc_phy_f3d"),
    "source_tracer": ("tmf", "qtr", "dqtr"),
    "tracer": ("ten_q", "gq0", "dqdt"),
    "chemistry": ("fscav",),
    "diagnostic": ("dtend",),
}

CASE_PROFILE_VARIANTS: dict[str, int] = {
    "case01_no_lfc_early_return": 1,
    "case02_pressure_depth_rejection": 2,
    "case03_rh_rejection": 3,
    "case04_entrained_lfc_rejection": 5,
    "case05_cin_rejection": 6,
    "case06_cloud_depth_rejection": 7,
    "case07_negative_cloudwork_rejection": 8,
}

CASE_CONFIGS: dict[str, dict[str, Any]] = {
    "case00_kcnv_precluded_return": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case01_no_lfc_early_return": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case02_pressure_depth_rejection": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case03_rh_rejection": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case04_entrained_lfc_rejection": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case05_cin_rejection": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case06_cloud_depth_rejection": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case07_negative_cloudwork_rejection": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case08_active_maritime_default": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case09_active_land_default": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case10_hwrf_samfshal_active": {"hwrf_samfshal": True, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case11_tke_entrainment_active": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case12_convective_tracer_transport": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case13_aerosol_wetdep_active": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case14_ncloud_cloudwater_detrainment": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case15_progsigma_first_step": {"hwrf_samfshal": False, "progsigma": True, "progomega": False, "tend_opt_scnv": 1},
    "case16_progsigma_warm_restart": {"hwrf_samfshal": False, "progsigma": True, "progomega": False, "tend_opt_scnv": 1},
    "case17_progomega_active": {"hwrf_samfshal": False, "progsigma": False, "progomega": True, "tend_opt_scnv": 1},
    "case18_small_grid_scaleaware": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case19_tend_opt_scnv_2_accumulate": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 2},
    "case20_tend_opt_scnv_3_apply_reset": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 3},
    "case21_tend_opt_scnv_4_noop": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 4},
    "case22_scnv_post_gfdl_cloud_diag": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case23_scnv_post_pdf_cloud_diag": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
    "case24_scnv_post_dtend_tracer": {"hwrf_samfshal": False, "progsigma": False, "progomega": False, "tend_opt_scnv": 1},
}


def _scalar(value: float, *, device: torch.device | str | None) -> torch.Tensor:
    return torch.as_tensor(float(value), dtype=torch.float64, device=device)


def _zeros(shape: tuple[int, ...], *, device: torch.device | str | None) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float64, device=device)


def _full(shape: tuple[int, ...], value: float, *, device: torch.device | str | None) -> torch.Tensor:
    return torch.full(shape, float(value), dtype=torch.float64, device=device)


def _level_values(start: int, stop: int, *, reference: torch.Tensor) -> torch.Tensor:
    return torch.arange(start, stop, dtype=reference.dtype, device=reference.device).reshape(1, stop - start)


def _gather2(field: torch.Tensor, index1: torch.Tensor) -> torch.Tensor:
    idx = (index1.to(torch.long) - 1).clamp(0, field.shape[1] - 1).reshape(-1, 1)
    return field.gather(1, idx).squeeze(1)


def _gather2_many(index1: torch.Tensor, *fields: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if not fields:
        return ()
    idx = (index1.to(torch.long) - 1).clamp(0, fields[0].shape[1] - 1).reshape(-1, 1, 1)
    packed = torch.stack(fields, dim=2)
    gathered = packed.gather(1, idx.expand(-1, 1, len(fields))).squeeze(1)
    return tuple(gathered[:, item] for item in range(len(fields)))


def _gather2_pair(field: torch.Tensor, index1_a: torch.Tensor, index1_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.stack(
        (
            (index1_a.to(torch.long) - 1).clamp(0, field.shape[1] - 1),
            (index1_b.to(torch.long) - 1).clamp(0, field.shape[1] - 1),
        ),
        dim=1,
    )
    gathered = field.gather(1, idx)
    return gathered[:, 0], gathered[:, 1]


def _gather_dynamic3_all(field: torch.Tensor, index1: torch.Tensor) -> torch.Tensor:
    idx = (index1.to(torch.long) - 1).clamp(0, field.shape[1] - 1).reshape(-1, 1, 1)
    return field.gather(1, idx.expand(-1, 1, field.shape[2])).squeeze(1)


def _level_hit(field: torch.Tensor, index1: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # [ncol, km]: the level ``index1`` (1-based, clamped as the gathers) of the masked columns.
    idx = (index1.to(torch.long) - 1).clamp(0, field.shape[1] - 1)
    return (torch.arange(field.shape[1], device=field.device) == idx[:, None]) & mask[:, None]


# The puts select in place of gather + out-of-place scatter (a full copy of the field plus a
# scatter); unselected levels keep their values, as the scatter of the current values did.
def _put_dynamic2(field: torch.Tensor, index1: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(_level_hit(field, index1, mask), values[:, None], field)


def _put_dynamic2_many(
    fields: tuple[torch.Tensor, ...],
    index1: torch.Tensor,
    values: tuple[torch.Tensor, ...],
    mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if len(fields) != len(values):
        raise ValueError("fields and values must have matching lengths")
    if not fields:
        return ()
    hit = _level_hit(fields[0], index1, mask)
    return tuple(torch.where(hit, value[:, None], field) for field, value in zip(fields, values))


def _put_dynamic3(field: torch.Tensor, index1: torch.Tensor, tracer0: int, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    tracer = torch.arange(field.shape[2], device=field.device) == int(tracer0)
    return torch.where(_level_hit(field, index1, mask)[:, :, None] & tracer, values[:, None, None], field)


def _put_dynamic3_all(field: torch.Tensor, index1: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(_level_hit(field, index1, mask)[:, :, None], values[:, None, :], field)


def _where(mask: torch.Tensor, true_value: torch.Tensor | float, false_value: torch.Tensor | float) -> torch.Tensor:
    return torch.where(mask, true_value, false_value)


def _masked_values(mask: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return values.masked_fill(~mask, 0.0)


def _masked_sum(mask: torch.Tensor, values: torch.Tensor, *, dim: int = 1) -> torch.Tensor:
    masked = _masked_values(mask, values)
    if masked.shape[dim] == 0:
        return masked.sum(dim=dim)
    # The source's loop order: torch.sum groups terms differently, which changes the last bit.
    return masked.cumsum(dim=dim).select(dim, -1).clone()


def _first_true_level(mask: torch.Tensor, levels: torch.Tensor, default: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 0:
        return default
    level_values = levels.expand_as(mask)
    sentinel = torch.full_like(level_values, torch.inf)
    candidate = torch.where(mask, level_values, sentinel).amin(dim=1)
    return torch.where(mask.any(dim=1), candidate, default)


def _last_prefix_true_level(mask: torch.Tensor, levels: torch.Tensor, default: torch.Tensor) -> torch.Tensor:
    # Last level of the leading run of true levels (``levels`` consecutive, shape [1, n]): the first
    # false level minus one, or the last level when all are true; exact small integers, no scan.
    if mask.shape[1] == 0:
        return default
    first_false = _first_true_level(~mask, levels, levels[0, -1] + 1.0)
    return torch.where(mask[:, 0], first_false - 1.0, default)


def _masked_argmax_level(values: torch.Tensor, mask: torch.Tensor, levels: torch.Tensor, default: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    masked = torch.where(mask, values, -torch.inf)
    max_values, max_idx = masked.max(dim=1)
    level_values = levels.expand_as(values).gather(1, max_idx.reshape(-1, 1)).squeeze(1)
    return torch.where(mask.any(dim=1), level_values, default), torch.where(mask.any(dim=1), max_values, _gather2(values, default))


def _where_div(
    mask: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    false_value: torch.Tensor | float,
) -> torch.Tensor:
    safe_denominator = torch.where(mask, denominator, 1.0)
    return torch.where(mask, numerator / safe_denominator, false_value)


def _sqrt_zero_grad(argument: torch.Tensor) -> torch.Tensor:
    positive = argument > 0.0
    safe = torch.where(positive, argument, 1.0)
    return torch.where(positive, torch.sqrt(safe), 0.0)


def _d_environment_block(
    cnvflg_bool, sfcpbl, kpbl, kbm, kmax, kbcon, zo, pfld,
    to, qo, qeso, po, rh, uo, vo, ctro, heo, heso,
    cp, eps, epsm1, grav, hvap, fact1, fact2, el2orc, table, *, hwrf,
):
    """Vectorized segment D with immutable inputs and no carried-state guards."""
    km = to.shape[1]
    kb1 = torch.ones_like(kpbl)
    levels_full = _level_values(1, km + 1, reference=kpbl)
    kb1_candidate = _last_prefix_true_level(zo <= sfcpbl[:, None], levels_full, kb1)
    kb1 = _where(cnvflg_bool, kb1_candidate, kb1)
    kb1 = torch.minimum(kb1, kpbl)
    kb, _ = _masked_argmax_level(
        heo, cnvflg_bool[:, None] & (levels_full >= kb1[:, None]) & (levels_full <= kpbl[:, None]),
        levels_full, kb1)
    levels_left = _level_values(1, km, reference=kpbl)
    cond_left = cnvflg_bool[:, None] & (levels_left <= (kmax - 1.0)[:, None])
    dz = 0.5 * (zo[:, 1:km] - zo[:, :km - 1])
    dp = 0.5 * (pfld[:, 1:km] - pfld[:, :km - 1])
    to_next = torch.where(cond_left, to[:, 1:km], 1.0)
    qo_next = qo[:, 1:km]
    qeso_next = qeso[:, 1:km]
    es = 0.01 * _fpvs_lookup(to_next, table)
    pprime = pfld[:, 1:km] + epsm1 * es
    qs = eps * es / pprime
    dqsdp = -qs / pprime
    desdt = es * (fact1 / to_next + fact2 / (to_next * to_next))
    dqsdt = qs * pfld[:, 1:km] * desdt / (es * pprime)
    gamma = el2orc * qeso_next / (to_next * to_next)
    dt = (grav * dz + hvap * dqsdp * dp) / (cp * (1.0 + gamma))
    dq = dqsdt * dt + dqsdp * dp
    to = _place(to, _where(cond_left, to_next + dt, to[:, :km - 1]), 0)
    qo = _place(qo, _where(cond_left, qo_next + dq, qo[:, :km - 1]), 0)
    po = _place(po, _where(cond_left, 0.5 * (pfld[:, :km - 1] + pfld[:, 1:km]), po[:, :km - 1]), 0)

    to_k, qo_k = to[:, :km - 1], qo[:, :km - 1]
    uo_k, vo_k = uo[:, :km - 1], vo[:, :km - 1]
    es = 0.01 * _fpvs_lookup(to_k, table)
    qes = torch.clamp_min(eps * es / (po[:, :km - 1] + epsm1 * es), 1.0e-8)
    qeso_k = _where(cond_left, qes, qeso[:, :km - 1])
    qo_k = _where(cond_left, torch.clamp_min(qo_k, 1.0e-10), qo_k)
    qeso = _place(qeso, qeso_k, 0)
    qo = _place(qo, qo_k, 0)
    rh_candidate = torch.clamp_max(qo_k / torch.where(cond_left, qeso_k, 1.0), 1.0)
    rh = _place(rh, _where(cond_left, rh_candidate, rh[:, :km - 1]), 0)
    height = 0.5 * grav * (zo[:, :km - 1] + zo[:, 1:km])
    heo = _place(heo, _where(cond_left, height + cp * to_k + hvap * qo_k, heo[:, :km - 1]), 0)
    heso = _place(heso, _where(cond_left, height + cp * to_k + hvap * qeso_k, heso[:, :km - 1]), 0)
    uo = _place(uo, _where(cond_left, 0.5 * (uo_k + uo[:, 1:km]), uo_k), 0)
    vo = _place(vo, _where(cond_left, 0.5 * (vo_k + vo[:, 1:km]), vo_k), 0)
    if not hwrf:
        ctro = _place(ctro, _where(cond_left[:, :, None],
            0.5 * (ctro[:, :km - 1] + ctro[:, 1:km]), ctro[:, :km - 1]), 0)
    hb = _gather2(heo, kb)
    levels_inner = _level_values(2, km, reference=kpbl)
    kbcon = _first_true_level(
        cnvflg_bool[:, None] & (levels_inner < kbm[:, None]) & (levels_inner > kb[:, None])
        & (hb[:, None] > heso[:, 1:km - 1]),
        levels_inner, _where(cnvflg_bool, kmax, kbcon))
    cnvflg_bool = cnvflg_bool & (kbcon != kmax)
    return kb, kb1, kbcon, cnvflg_bool, to, qo, qeso, po, rh, heo, heso, uo, vo, ctro



# Static candidate granularity; exact short tails use the same equations.
RECURRENCE_CHUNK_SIZE = 8


def _e_cloud_base_block(cnvflg_bool, kb, kbcon, kpbl, kbm, kmax, dot, zi, pfld,
                        heo, heso, hpbl, islimsk, pdot):
    dot_kbcon, zi_kbcon = _gather2_many(kbcon, dot, zi)
    pdot = _where(cnvflg_bool, 0.01 * dot_kbcon, pdot)
    w3 = _where(islimsk == 1.0, -5.0e-3, -1.0e-3)
    w4 = _where(islimsk == 1.0, -5.0e-4, -2.0e-5)
    tem = _where(pdot <= w4, (pdot - w4) / (w3 - w4),
                 _where(pdot >= -w4, -(pdot + w4) / (w4 - w3), 0.0))
    tem = torch.clamp(tem, min=-1.0, max=1.0)
    cinpcr = 180.0 - (1.0 - tem) * (0.5 * (180.0 - 120.0))
    pfld_kb, pfld_kbcon = _gather2_pair(pfld, kb, kbcon)
    pressure_depth = pfld_kb - pfld_kbcon
    cnvflg_bool = cnvflg_bool & ~((pressure_depth > cinpcr) & (zi_kbcon > hpbl))
    levels_full = _level_values(1, heo.shape[1] + 1, reference=pdot)
    levels_inner = _level_values(2, heo.shape[1], reference=pdot)
    kb = _where(cnvflg_bool, 1.0, kb)
    kb, _ = _masked_argmax_level(heo, cnvflg_bool[:, None] & (levels_full <= kpbl[:, None]), levels_full, kb)
    hb = _gather2(heo, kb)
    kbcon = _first_true_level(
        cnvflg_bool[:, None] & (levels_inner < kbm[:, None]) & (levels_inner > kb[:, None]) & (hb[:, None] > heso[:, 1:-1]),
        levels_inner, _where(cnvflg_bool, kmax, kbcon))
    cnvflg_bool = cnvflg_bool & (kbcon != kmax)
    pdot = _where(cnvflg_bool, 0.01 * _gather2(dot, kbcon), pdot)
    return cnvflg_bool, kb, kbcon, pdot, w3, w4


def _recurrence_ranges(start: int, stop: int) -> list[tuple[int, int]]:
    configured = selected_recurrence_chunk_size()
    width = RECURRENCE_CHUNK_SIZE if configured is None else configured
    return [(lo, min(lo + width, stop)) for lo in range(start, stop, width)]


def _recurrence_chunk(name, fn, *args, **kwargs):
    return run_compile_island("recurrence_chunks", name, fn, *args, **kwargs)


def _ordered_pair_sums(mask, first_terms, second_terms, *, masked_add):
    if mask.shape[1] == 0:
        return first_terms.sum(dim=1), second_terms.sum(dim=1)
    # Pack once so contiguous level windows cross the compile boundary without copies.
    rows = mask.t().contiguous()
    terms = torch.stack((first_terms.t(), second_terms.t()), dim=1)
    first = torch.zeros_like(first_terms[:, 0])
    second = torch.zeros_like(second_terms[:, 0])
    ranges = [(0, mask.shape[1])] if parallel_masked_sums() else _recurrence_ranges(0, mask.shape[1])
    for lo, hi in ranges:
        first, second = _recurrence_chunk(
            "ordered_pair_sums", recurrence_chunks.ordered_pair_sums,
            rows[lo:hi], terms[lo:hi], first, second, masked_add=masked_add)
    return first, second


def _place(base, value, start, *, dim=1):
    """``base`` with ``value`` in place of its indices ``start:start + value.shape[dim]`` along ``dim``.

    Joining per-level cells (``unbind`` then one ``torch.stack``) made Inductor emit one copy kernel
    per level and field (1,837 kernels in the compiled shallow variant at x1.10242); a
    chunk result enters as one slice. Values are unchanged.
    """
    return torch.slice_scatter(base, value, dim=dim, start=start, end=start + value.shape[dim])


def _place_chunks(base, pieces, start):
    return run_compile_island("recurrence_chunks", "assemble_levels", _place_chunks_tensor,
                              base, tuple(pieces), start)


def _place_chunks_tensor(base, pieces, start):
    # Join equal-width level-major blocks before copying the complete destination.
    # Normal full blocks and their short tail need at most two profile updates.
    pieces = tuple(pieces)
    lo = 0
    while lo < len(pieces):
        hi = lo + 1
        while hi < len(pieces) and pieces[hi].shape[1] == pieces[lo].shape[1]:
            hi += 1
        if hi == lo + 1:
            joined = pieces[lo]
        else:
            rows = torch.stack(tuple(piece.movedim(1, 0) for piece in pieces[lo:hi]), dim=0)
            joined = rows.flatten(0, 1).movedim(0, 1)
        base = _place(base, joined, start)
        start += joined.shape[1]
        lo = hi
    return base


def _assemble_recurrence(old, parts):
    # Interior levels 1..km-2 from the chunks; the end levels keep their old values.
    return tuple(_place_chunks(value, pieces, 1) for value, pieces in zip(old, parts))


def _transport_tracers(cnv, kb, kmax, levels, zi, xlamue, ctro, ecko, ercko, start, stop, *, cq, km):
    old = (ecko[:, :, start:stop], ercko[:, :, start:stop])
    profiles = recurrence_chunks.pack_profiles(zi, xlamue)
    tracers = recurrence_chunks.pack_profiles(ctro[:, :, start:stop], *old)
    parts = [[], []]
    carry = old[0][:, 0].contiguous()
    for lo, hi in _recurrence_ranges(1, km - 1):
        values = _recurrence_chunk("tracer", recurrence_chunks.tracer,
            cnv, kb, kmax, levels[lo], profiles[lo-1:hi], tracers[lo-1:hi], carry, cq=cq)
        for cells, value in zip(parts, values):
            cells.append(value)
        carry = values[0][:, -1]
    del profiles, tracers
    transported, raw = _assemble_recurrence(old, parts)
    return _place(ecko, transported, start, dim=2), _place(ercko, raw, start, dim=2)


def _f_seed_profiles(cnvflg_bool, kb, heo, uo, vo, hcko, ucko, vcko, tracers):
    heo_kb, uo_kb, vo_kb = _gather2_many(kb, heo, uo, vo)
    seeded = _put_dynamic2_many((hcko, ucko, vcko), kb, (heo_kb, uo_kb, vo_kb), cnvflg_bool)
    if tracers:
        ctro, ecko, ercko = tracers
        ctro_kb = _gather_dynamic3_all(ctro, kb)
        seeded += (_put_dynamic3_all(ecko, kb, ctro_kb, cnvflg_bool),
                   _put_dynamic3_all(ercko, kb, ctro_kb, cnvflg_bool))
    return seeded


def _k_tendencies(cnvflg_bool, kb, ktcon, del_, zi, xlamue, xlamud, eta,
                  heo, hcko, qrcko, qcko, uo, ucko, vo, vcko, dellal,
                  qlko_ktcon, grav, tracers, *, ntr):
    km = del_.shape[1]
    levels = _level_values(2, km, reference=del_)
    active = cnvflg_bool[:, None] & (levels > kb[:, None]) & (levels < ktcon[:, None])
    dp_mid = 1000.0 * del_[:, 1:km-1]
    dz_mid = zi[:, 1:km-1] - zi[:, :km-2]
    tem_mid = 0.5 * (xlamue[:, 1:km-1] + xlamue[:, :km-2])
    factor_mid = grav / dp_mid
    heat = (
        eta[:, 1:km-1] * heo[:, 1:km-1]
        - eta[:, :km-2] * heo[:, :km-2]
        - tem_mid * eta[:, :km-2] * 0.5 * (heo[:, 1:km-1] + heo[:, :km-2]) * dz_mid
        + xlamud[:, None] * eta[:, :km-2] * 0.5 * (hcko[:, 1:km-1] + hcko[:, :km-2]) * dz_mid
    ) * factor_mid
    moisture = (-eta[:, 1:km-1] * qrcko[:, 1:km-1] + eta[:, :km-2] * qcko[:, :km-2]) * factor_mid
    zonal = (
        eta[:, 1:km-1] * (uo[:, 1:km-1] - ucko[:, 1:km-1])
        - eta[:, :km-2] * (uo[:, :km-2] - ucko[:, :km-2])
    ) * factor_mid
    meridional = (
        eta[:, 1:km-1] * (vo[:, 1:km-1] - vcko[:, 1:km-1])
        - eta[:, :km-2] * (vo[:, :km-2] - vcko[:, :km-2])
    ) * factor_mid
    # Functional assembly keeps compiled forward-mode derivatives through new buffers.
    dellah, dellaq, dellau, dellav = tuple(
        _place(torch.zeros_like(del_), _where(active, value, 0.0), 1)
        for value in (heat, moisture, zonal, meridional)
    )
    dellae = del_.new_zeros((del_.shape[0], km, ntr))
    if tracers:
        ercko, ecko = tracers
        tracer_mid = (
            -eta[:, 1:km-1, None] * ercko[:, 1:km-1, :]
            + eta[:, :km-2, None] * ecko[:, :km-2, :]
        ) * grav / dp_mid[:, :, None]
        dellae = _place(dellae, _where(active[:, :, None], tracer_mid, 0.0), 1)
    dp = 1000.0 * _gather2(del_, ktcon)
    hcko_prev, heo_prev, qcko_prev, ucko_prev, uo_prev, vcko_prev, vo_prev, eta_prev = _gather2_many(
        ktcon - 1.0, hcko, heo, qcko, ucko, uo, vcko, vo, eta)
    tem = eta_prev * grav / dp
    dellah, dellaq, dellau, dellav, dellal = _put_dynamic2_many(
        (dellah, dellaq, dellau, dellav, dellal), ktcon,
        (tem * (hcko_prev - heo_prev), tem * qcko_prev,
         tem * (ucko_prev - uo_prev), tem * (vcko_prev - vo_prev), tem * qlko_ktcon),
        cnvflg_bool,
    )
    if tracers:
        ecko_prev = _gather_dynamic3_all(ecko, ktcon - 1.0)
        dellae_top = eta_prev[:, None] * ecko_prev * grav / dp[:, None]
        dellae = _put_dynamic3_all(dellae, ktcon, dellae_top, cnvflg_bool)
    return dellah, dellaq, dellau, dellav, dellal, dellae


def _l_tvd(cnvflg_bool, kb, ktcon, del_, eta, q1, qo, dellaq, dellae, grav, tracers):
    """Independent TVD interfaces and their tendency updates, without mutating inputs."""
    km = q1.shape[1]
    levels_left = _level_values(1, km, reference=q1)
    cond_left = cnvflg_bool[:, None] & (levels_left <= ktcon[:, None])
    q_diff = _place(torch.zeros_like(q1),
                    _where(cond_left, q1[:, :km-1] - q1[:, 1:km], 0.0), 1)
    bottom = _where(cnvflg_bool & (q1[:, 0] >= 0.0),
                    torch.clamp_min(2.0 * q1[:, 0] - q1[:, 1], 0.0) - q1[:, 0], q_diff[:, 0])
    q_diff = _place(q_diff, bottom[:, None], 0)
    next_idx = torch.clamp(torch.arange(2, km + 1, dtype=torch.long, device=q1.device), max=km - 1)
    qden = q_diff[:, 1:km]
    qvalid = torch.abs(qden) > 1.0e-22
    rrkp = _where(qvalid, q_diff[:, next_idx] / torch.where(qvalid, qden, 1.0), 0.0)
    phkp = (rrkp + torch.abs(rrkp)) / (1.0 + torch.abs(rrkp))
    tem1_flux = q1[:, 1:km] + phkp * (qo[:, :km-1] - q1[:, 1:km])
    cond_flux = cnvflg_bool[:, None] & (levels_left >= kb[:, None]) & (levels_left < ktcon[:, None]) & (eta[:, :km-1] > 0.0)
    flxtvd = _where(cond_flux, eta[:, :km-1] * tem1_flux, 0.0)
    levels_mid = _level_values(2, km, reference=q1)
    cond_mid = cnvflg_bool[:, None] & (levels_mid > kb[:, None]) & (levels_mid <= ktcon[:, None])
    dp_mid = 1000.0 * del_[:, 1:km-1]
    dellaq = _place(dellaq, dellaq[:, 1:km-1] + _where(
        cond_mid, (flxtvd[:, 1:km-1] - flxtvd[:, :km-2]) * grav / dp_mid, 0.0), 1)
    e_diff = torch.zeros_like(dellae)
    if tracers:
        ctr, ctro = tracers
        e_diff = _place(e_diff, _where(cond_left[:, :, None],
            ctr[:, :km-1, :] - ctr[:, 1:km, :], 0.0), 1)
        bottom = _where(cnvflg_bool[:, None] & (ctr[:, 0, :] >= 0.0),
            torch.clamp_min(2.0 * ctr[:, 0, :] - ctr[:, 1, :], 0.0) - ctr[:, 0, :],
            e_diff[:, 0, :])
        e_diff = _place(e_diff, bottom[:, None, :], 0)
        eden = e_diff[:, 1:km, :]
        evalid = torch.abs(eden) > 1.0e-22
        errkp = _where(evalid, e_diff[:, next_idx, :] / torch.where(evalid, eden, 1.0), 0.0)
        ephkp = (errkp + torch.abs(errkp)) / (1.0 + torch.abs(errkp))
        etem1 = ctr[:, 1:km, :] + ephkp * (ctro[:, :km-1, :] - ctr[:, 1:km, :])
        flxtvd_all = _where(cond_flux[:, :, None], eta[:, :km-1, None] * etem1, 0.0)
        flxtvd = flxtvd_all[:, :, -1]
        dellae = _place(dellae, dellae[:, 1:km-1, :] + _where(
            cond_mid[:, :, None], (flxtvd_all[:, 1:km-1, :] - flxtvd_all[:, :km-2, :]) * grav / dp_mid[:, :, None],
            0.0), 1)
    return q_diff, e_diff, flxtvd, dellaq, dellae


def _f_recurrence_block(
    cnvflg_bool: torch.Tensor,
    kb: torch.Tensor,
    kbcon: torch.Tensor,
    kbm: torch.Tensor,
    kmax: torch.Tensor,
    zi: torch.Tensor,
    xlamue: torch.Tensor,
    xlamud: torch.Tensor,
    eta: torch.Tensor,
    heo: torch.Tensor,
    heso: torch.Tensor,
    hcko: torch.Tensor,
    ucko: torch.Tensor,
    vcko: torch.Tensor,
    uo: torch.Tensor,
    vo: torch.Tensor,
    dbyo: torch.Tensor,
    ctro: torch.Tensor,
    ecko: torch.Tensor,
    ercko: torch.Tensor,
    chem_c: torch.Tensor,
    chem_pw: torch.Tensor,
    c0t: torch.Tensor,
    fscav: torch.Tensor,
    *,
    cm: float,
    cq: float,
    pgcon: float,
    hwrf: bool,
    do_aerosols_bool: bool,
    itc: int,
    ntc: int,
    ntk: int,
    ntr: int,
    km: int,
) -> tuple[torch.Tensor, ...]:
    levels = _level_values(1, km + 1, reference=eta)[0]
    profiles = recurrence_chunks.pack_profiles(zi, xlamue, eta)
    parts = []
    carry = eta[:, -1].contiguous()
    for lo, hi in reversed(_recurrence_ranges(0, km - 1)):
        piece = _recurrence_chunk("eta_lower", recurrence_chunks.eta_lower,
            cnvflg_bool, kb, kbcon, levels[lo], profiles[lo:hi+1], xlamud, carry)
        parts.append(piece)
        carry = piece[:, 0]
    del profiles
    eta = _place_chunks(eta, reversed(parts), 0)
    profiles = recurrence_chunks.pack_profiles(zi, xlamue, eta)
    parts = []
    carry, flg = eta[:, 0].contiguous(), cnvflg_bool.clone()
    for lo, hi in _recurrence_ranges(1, km - 1):
        piece, kmax, kbm, flg = _recurrence_chunk("eta_upper", recurrence_chunks.eta_upper,
            flg, kbcon, kmax, kbm, levels[lo], profiles[lo-1:hi], xlamud, carry)
        parts.append(piece)
        carry = piece[:, -1]
    del profiles
    eta = _place_chunks(eta, parts, 1)

    tracers = () if hwrf else (ctro, ecko, ercko)
    seeded = _recurrence_chunk("seed_profiles", _f_seed_profiles, cnvflg_bool, kb,
                               heo, uo, vo, hcko, ucko, vcko, tracers)
    hcko, ucko, vcko = seeded[:3]
    if not hwrf:
        ecko, ercko = seeded[3:]

    old = (hcko, ucko, vcko, dbyo)
    profiles = recurrence_chunks.pack_profiles(zi, xlamue, heo, heso, uo, vo, *old)
    parts = [[] for _ in old]
    carry = tuple(t[:, 0].contiguous() for t in old[:3])
    for lo, hi in _recurrence_ranges(1, km - 1):
        values = _recurrence_chunk("thermo", recurrence_chunks.thermo, cnvflg_bool, kb, kmax,
            levels[lo], profiles[lo-1:hi], xlamud, carry, cm=cm, pgcon=pgcon)
        for cells, value in zip(parts, values):
            cells.append(value)
        carry = tuple(t[:, -1] for t in values[:3])
    del profiles
    hcko, ucko, vcko, dbyo = _assemble_recurrence(old, parts)

    if not hwrf:
        tracer_count = itc - 3 if do_aerosols_bool else ntr
        if tracer_count > 0:
            ecko, ercko = _transport_tracers(cnvflg_bool, kb, kmax, levels, zi, xlamue,
                ctro, ecko, ercko, 0, tracer_count, cq=cq, km=km)
        if do_aerosols_bool:
            aer_slice = slice(itc - 3, itc - 3 + ntc)
            old = (ecko[:, :, aer_slice], ercko[:, :, aer_slice], chem_c, chem_pw)
            profiles = recurrence_chunks.pack_profiles(zi, xlamue, c0t, eta)
            tracers = recurrence_chunks.pack_profiles(ctro[:, :, aer_slice], *old)
            parts = [[] for _ in old]
            carry = old[0][:, 0].contiguous()
            for lo, hi in _recurrence_ranges(1, km - 1):
                values = _recurrence_chunk("aerosol", recurrence_chunks.aerosol,
                    cnvflg_bool, kb, kmax, levels[lo], profiles[lo-1:hi], tracers[lo-1:hi],
                    fscav, carry, cq=cq)
                for cells, value in zip(parts, values):
                    cells.append(value)
                carry = values[0][:, -1]
            del profiles, tracers
            transported, raw, chem_c, chem_pw = _assemble_recurrence(old, parts)
            ecko = _place(ecko, transported, itc - 3, dim=2)
            ercko = _place(ercko, raw, itc - 3, dim=2)
            # The source's TKE pass follows the COMPLETE aerosol pass. Their
            # species ranges may overlap; interleaving would change recurrence.
            if ntk > 2:
                ecko, ercko = _transport_tracers(cnvflg_bool, kb, kmax, levels, zi, xlamue,
                    ctro, ecko, ercko, ntk-3, ntk-2, cq=cq, km=km)
    return eta, kbm, kmax, hcko, ucko, vcko, ecko, ercko, dbyo, chem_c, chem_pw


def _h_recurrence_block(
    cnvflg_bool: torch.Tensor,
    kb: torch.Tensor,
    kbcon: torch.Tensor,
    ktcon: torch.Tensor,
    zi: torch.Tensor,
    qeso: torch.Tensor,
    to: torch.Tensor,
    dbyo: torch.Tensor,
    xlamue: torch.Tensor,
    xlamud: torch.Tensor,
    eta: torch.Tensor,
    qo: torch.Tensor,
    del_: torch.Tensor,
    c0t: torch.Tensor,
    qcko: torch.Tensor,
    qrcko: torch.Tensor,
    pwo: torch.Tensor,
    cnvwt: torch.Tensor,
    dellal: torch.Tensor,
    zdqca: torch.Tensor,
    buo: torch.Tensor,
    drag: torch.Tensor,
    wush: torch.Tensor,
    uo: torch.Tensor,
    vo: torch.Tensor,
    *,
    cq: float,
    c1: float,
    csmf: float,
    el2orc: torch.Tensor,
    hvap: float,
    grav: float,
    cp: float,
    fv: float,
    km: int,
) -> tuple[torch.Tensor, ...]:
    levels = _level_values(1, km + 1, reference=qcko)[0]
    old = (qcko, qrcko, pwo, cnvwt, dellal, zdqca, buo, drag, wush)
    profiles = recurrence_chunks.pack_profiles(zi, qeso, to, dbyo, xlamue, eta,
                                               qo, del_, c0t, *old, uo, vo)
    parts = [[] for _ in old]
    carry = qcko[:, 0].contiguous()
    for lo, hi in _recurrence_ranges(1, km - 1):
        values = _recurrence_chunk("buoyancy", recurrence_chunks.buoyancy,
            cnvflg_bool, kb, kbcon, ktcon, levels[lo], profiles[lo-1:hi], xlamud, carry,
            el2orc, cq=cq, c1=c1, csmf=csmf, hvap=hvap, grav=grav, cp=cp, fv=fv)
        for cells, value in zip(parts, values):
            cells.append(value)
        carry = values[0][:, -1]
    del profiles
    return _assemble_recurrence(old, parts)


def _i_precip_recurrence_block(
    cnvflg_bool: torch.Tensor,
    ktcon: torch.Tensor,
    ktcon1: torch.Tensor,
    zi: torch.Tensor,
    qeso: torch.Tensor,
    to: torch.Tensor,
    dbyo: torch.Tensor,
    xlamue: torch.Tensor,
    eta: torch.Tensor,
    qo: torch.Tensor,
    del_: torch.Tensor,
    c0t: torch.Tensor,
    qcko: torch.Tensor,
    qrcko: torch.Tensor,
    pwo: torch.Tensor,
    cnvwt: torch.Tensor,
    dellal: torch.Tensor,
    zdqca: torch.Tensor,
    *,
    cq: float,
    c1: float,
    el2orc: torch.Tensor,
    hvap: float,
    grav: float,
    km: int,
) -> tuple[torch.Tensor, ...]:
    levels = _level_values(1, km + 1, reference=qcko)[0]
    old = (qcko, qrcko, pwo, cnvwt, dellal, zdqca)
    profiles = recurrence_chunks.pack_profiles(zi, qeso, to, dbyo, xlamue, eta, qo, del_, c0t, *old)
    parts = [[] for _ in old]
    carry = qcko[:, 0].contiguous()
    for lo, hi in _recurrence_ranges(1, km - 1):
        values = _recurrence_chunk("precip", recurrence_chunks.precip,
            cnvflg_bool, ktcon, ktcon1, levels[lo], profiles[lo-1:hi], carry, el2orc,
            cq=cq, c1=c1, hvap=hvap, grav=grav)
        for cells, value in zip(parts, values):
            cells.append(value)
        carry = values[0][:, -1]
    del profiles
    return _assemble_recurrence(old, parts)


def _i_wu2_recurrence_block(
    cnvflg_bool: torch.Tensor,
    kbcon1: torch.Tensor,
    ktcon: torch.Tensor,
    zi: torch.Tensor,
    drag: torch.Tensor,
    buo: torch.Tensor,
    wush: torch.Tensor,
    po: torch.Tensor,
    to: torch.Tensor,
    wu2: torch.Tensor,
    omega_u: torch.Tensor,
    *,
    bb1: float,
    bb2: float,
    rd: float,
    grav: float,
    km: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    levels = _level_values(1, km + 1, reference=wu2)[0]
    profiles = recurrence_chunks.pack_profiles(zi, drag, buo, wush, wu2)
    parts = []
    carry = wu2[:, 0].contiguous()
    for lo, hi in _recurrence_ranges(1, km - 1):
        piece = _recurrence_chunk("velocity", recurrence_chunks.velocity,
            cnvflg_bool, kbcon1, ktcon, levels[lo], profiles[lo-1:hi], carry, bb1=bb1, bb2=bb2)
        parts.append(piece)
        carry = piece[:, -1]
    del profiles
    wu2 = _place_chunks(wu2, parts, 1)
    levels_omega = levels[None, 1:km-1]
    cond_omega = cnvflg_bool[:, None] & (levels_omega > kbcon1[:, None]) & (levels_omega < ktcon[:, None])
    safe_to = torch.where(cond_omega, to[:, 1:km-1], 1.0)
    rho = po[:, 1:km-1] * 100.0 / (rd * safe_to)
    omega = -_sqrt_zero_grad(torch.clamp_min(wu2[:, 1:km-1], 0.0)) * rho * grav
    interior = _where(cond_omega, torch.clamp_min(omega, -80.0), omega_u[:, 1:km-1])
    omega_u = _place(omega_u, interior, 1)
    return wu2, omega_u


def _o_evap_recurrence_block(
    cnvflg_bool: torch.Tensor,
    kb: torch.Tensor,
    ktcon: torch.Tensor,
    kmax: torch.Tensor,
    pwo: torch.Tensor,
    xmb: torch.Tensor,
    del_: torch.Tensor,
    qeso: torch.Tensor,
    new_t1: torch.Tensor,
    new_q1: torch.Tensor,
    rn: torch.Tensor,
    rntot: torch.Tensor,
    delqbar: torch.Tensor,
    deltbar: torch.Tensor,
    *,
    shevf: float,
    evef: float,
    delt: float,
    el2orc: torch.Tensor,
    elocp: torch.Tensor,
    grav: float,
    km: int,
) -> tuple[torch.Tensor, ...]:
    levels = _level_values(1, km + 1, reference=new_t1)[0]
    profiles = recurrence_chunks.pack_profiles(pwo, del_, qeso, new_t1, new_q1)
    zero = torch.zeros_like(rn)
    carry = (rn, zero, zero, zero, zero, delqbar, deltbar, cnvflg_bool.clone())
    t_parts, q_parts = [], []
    for lo, hi in reversed(_recurrence_ranges(0, km)):
        t_chunk, q_chunk, carry = _recurrence_chunk("evaporation", recurrence_chunks.evaporation,
            cnvflg_bool, kb, ktcon, kmax, levels[lo], profiles[lo:hi], xmb,
            carry, rntot, shevf=shevf, evef=evef, delt=delt, el2orc=el2orc, elocp=elocp, grav=grav)
        t_parts.append(t_chunk)
        q_parts.append(q_chunk)
    del profiles
    rn, _, qevap, delqev, _, delqbar, deltbar, flg = carry
    new_t1 = _place_chunks(new_t1, reversed(t_parts), 0)
    new_q1 = _place_chunks(new_q1, reversed(q_parts), 0)
    return rn, qevap, delqev, delqbar, deltbar, new_t1, new_q1, flg


def _progomega_calc(
    *,
    first_time_step: bool,
    restart: bool,
    kbcon1: torch.Tensor,
    ktcon: torch.Tensor,
    omegain: torch.Tensor,
    del_: torch.Tensor,
    zi: torch.Tensor,
    cnvflg_bool: torch.Tensor,
    omegaout: torch.Tensor,
    buo: torch.Tensor,
    drag: torch.Tensor,
    wush: torch.Tensor,
    ps: torch.Tensor,
) -> torch.Tensor:
    KM = del_.shape[1]
    omega = omegain.clone()
    omega = _where(cnvflg_bool[:, None] & (omega < 1.0e-5), 0.0, omega)
    if first_time_step and not restart:
        omega = _where(cnvflg_bool[:, None], -1.2, omega)
    out = omegaout.clone()
    levels_inner = _level_values(2, KM, reference=ps)
    cond = cnvflg_bool[:, None] & (levels_inner >= kbcon1[:, None]) & (levels_inner < ktcon[:, None])
    dp = 1000.0 * del_[:, 1 : KM - 1]
    dz = zi[:, 2:KM] - zi[:, 1 : KM - 1]
    dpodz = dp / dz
    term_a = DELT * (1.5 * drag[:, 1 : KM - 1] * dpodz)
    term_b = 1.0 - DELT * 1.2 * wush[:, 1 : KM - 1] * dpodz
    omega_k = omega[:, 1 : KM - 1]
    term_c = omega_k - DELT * 0.6 * buo[:, 1 : KM - 1] * dpodz - DELT * omega_k * (omega[:, : KM - 2] - omega_k) / dp
    discr = term_b * term_b - 4.0 * term_a * term_c
    safe_term_a = torch.where(term_a != 0.0, term_a, 1.0)
    root = (-term_b - _sqrt_zero_grad(discr)) / (2.0 * safe_term_a)
    candidate = _where(discr >= 0.0, root, omega_k)
    candidate = torch.clamp(candidate, min=-80.0, max=-1.2)
    out[:, 1 : KM - 1] = _where(cond, candidate, out[:, 1 : KM - 1].clone())
    return out


def _progsigma_calc(
    *,
    first_time_step: bool,
    restart: bool,
    del_: torch.Tensor,
    tmfq: torch.Tensor,
    qmicro: torch.Tensor,
    dbyo1: torch.Tensor,
    zdqca: torch.Tensor,
    omega_u: torch.Tensor,
    zeta: torch.Tensor,
    qadv: torch.Tensor,
    kb: torch.Tensor,
    kbcon1: torch.Tensor,
    ktcon: torch.Tensor,
    cnvflg_bool: torch.Tensor,
    sigmain: torch.Tensor,
    sigmaout: torch.Tensor,
    ps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    KM = del_.shape[1]
    sigmind_new = 0.0 if first_time_step and not restart else 0.01
    sigmins = 0.03
    inbu = torch.zeros_like(del_)
    levels_full = _level_values(1, KM + 1, reference=ps)
    dp = _masked_values(cnvflg_bool[:, None] & (levels_full > 1.0) & (levels_full < float(KM)), 1000.0 * del_)

    sigmab = torch.zeros_like(ps)
    levels_inner = _level_values(2, KM, reference=ps)
    cond_cloud = cnvflg_bool[:, None] & (levels_inner > kbcon1[:, None]) & (levels_inner < ktcon[:, None])
    sigmab = _masked_sum(cond_cloud, sigmain[:, 1 : KM - 1] * dp[:, 1 : KM - 1])
    sumx = _masked_sum(cond_cloud, dp[:, 1 : KM - 1])
    zero_sum = cnvflg_bool & (sumx == 0.0)
    sigmab = _where(zero_sum, _gather2(sigmain, kbcon1), sigmab)
    nonzero_sum = cnvflg_bool & (sumx != 0.0)
    sigmab_avg = torch.clamp_max(sigmab / torch.where(nonzero_sum, sumx, 1.0), 1.0)
    sigmab = _where(nonzero_sum, _where(sigmab_avg < 1.0e-5, 0.0, sigmab_avg), sigmab)

    term_a = torch.zeros_like(ps)
    term_b = torch.zeros_like(ps)
    term_c = torch.zeros_like(ps)
    term_d = torch.zeros_like(ps)
    fdqa = torch.zeros_like(ps)
    for k0 in range(KM - 1):
        k1 = float(k0 + 1)
        cond = cnvflg_bool & (k1 >= kb) & (k1 < ktcon)
        mcon = HVAP * (qadv[:, k0] + tmfq[:, k0] + qmicro[:, k0]) * dp[:, k0]
        buy2 = term_d + mcon
        active = cond & (buy2 > 0.0)
        inbu[:, k0] = _where(active, 1.0, inbu[:, k0])
        term_d = term_d + _masked_values(active, mcon)
    cond_buoy = cnvflg_bool[:, None] & (levels_inner >= kbcon1[:, None]) & (levels_inner < ktcon[:, None])
    common = zeta[:, 1 : KM - 1] * dbyo1[:, 1 : KM - 1] * inbu[:, 1 : KM - 1] * dp[:, 1 : KM - 1]
    term_a = _masked_sum(cond_buoy, sigmab[:, None] * common)
    term_b = _masked_sum(cond_buoy, common)
    for k0 in range(1, KM - 1):
        k1 = float(k0 + 1)
        cond = cnvflg_bool & (k1 >= kbcon1) & (k1 < ktcon)
        form = -1.0 * inbu[:, k0] * (omega_u[:, k0] * DELT)
        fdqb = 0.5 * form * zdqca[:, k0]
        term_c = term_c + _masked_values(cond, inbu[:, k0] * (fdqb + fdqa) * HVAP * zeta[:, k0])
        fdqa = _where(cond, fdqb, fdqa)
    den = torch.clamp_max(term_c + term_b, 1.0e8)
    cvg = torch.clamp_min(term_d * DELT, 0.0)
    zz = ((term_a >= 0.0) & (term_b >= 0.0) & ((term_c - 1.0e-11) >= 0.0)).to(ps.dtype)
    candidate = zz * (term_a + cvg) / (den + (1.0 - zz))
    positive = cnvflg_bool & (candidate > 0.0)
    candidate = _where(positive, torch.clamp(candidate, min=sigmind_new, max=0.95), candidate)
    sigmab = _where(cnvflg_bool, candidate, sigmab)
    sigmaout = _where(cnvflg_bool[:, None], sigmab[:, None].expand_as(sigmaout), sigmaout)
    sigmab = _where(cnvflg_bool, torch.clamp_min(sigmab, sigmins), sigmab)
    sigmab = torch.clamp_max(sigmab, 0.95)
    return sigmab, sigmaout


def _state_from_profile_variant(
    profile_variant: int,
    *,
    device: torch.device | str | None = None,
    ncol: int | None = None,
    ntrac: int = NTR,
) -> SaSasShallowConvectionState:
    ncol = IM if ncol is None else int(ncol)
    if ncol <= 0:
        raise ValueError(f"ncol must be positive, got {ncol}")
    if not isinstance(ntrac, int) or ntrac < 8:
        raise ValueError("native profile needs at least eight declared tracer slots")
    NTR = ntrac
    NN = NTR + 2
    levels = torch.arange(KM, dtype=torch.float64, device=device).reshape(1, KM)
    columns = torch.arange(ncol, dtype=torch.int64, device=device)

    prslp = (100000.0 - levels * 5500.0).repeat(ncol, 1)
    delp = _full((ncol, KM), 5000.0, device=device)
    phil = (GRAV * levels * 450.0).repeat(ncol, 1)
    t1 = (304.0 - 4.0 * levels).repeat(ncol, 1)
    if profile_variant == 1:
        t1 = (302.0 - 3.8 * levels).repeat(ncol, 1)
        t1[:, :4] = t1[:, :4] + 4.0 * (5.0 - torch.arange(1, 5, dtype=torch.float64, device=device))
        t1[:, 0] = 310.0
        t1[:, 1] = 300.0
        t1[:, 2] = 309.0
        t1[:, 3] = 320.0
        t1[:, 4] = 315.0
    elif profile_variant == 2:
        t1 = (302.0 - 4.0 * levels).repeat(ncol, 1)
        t1[:, 0] = 311.0
        t1[:, 1] = 294.0
        t1[:, 2] = 304.0
        t1[:, 3] = 316.0
        t1[:, 4] = 312.0
    elif profile_variant == 3:
        t1 = (301.0 - 3.5 * levels).repeat(ncol, 1)
        t1[:, 0] = 307.0
        t1[:, 1] = 302.0
        t1[:, 2] = 301.0
        t1[:, 3] = 306.0
        t1[:, 4] = 310.0
    elif profile_variant == 5:
        t1 = (302.0 - 3.8 * levels).repeat(ncol, 1)
        t1[:, 0] = 310.0
        t1[:, 1] = 300.0
        t1[:, 2] = 309.0
        t1[:, 3] = 305.0
        t1[:, 4] = 300.0
    elif profile_variant == 6:
        t1 = (302.0 - 3.8 * levels).repeat(ncol, 1)
        t1[:, 0] = 312.0
        t1[:, 1] = 298.0
        t1[:, 2] = 308.0
        t1[:, 3] = 304.0
        t1[:, 4] = 299.0
    elif profile_variant == 7:
        t1[:, 3] = 330.0
        t1[:, 4] = 318.0
    elif profile_variant == 8:
        t1[:, 0] = 310.0
        t1[:, 3] = 330.0
        t1[:, 4] = 318.0
    elif profile_variant == 9:
        t1[:, 0] = 310.0
        t1[:, 4] = 330.0
        t1[:, 5] = 318.0
    esat = 0.01 * fpvs(t1)
    pfmb = 0.01 * prslp
    qsat = EPS * esat / (pfmb + EPSM1 * esat)
    q1 = torch.clamp(0.98 * qsat, min=1.0e-8, max=0.05)
    if profile_variant == 1:
        q1[:, 0] = torch.clamp_min(q1[:, 0], 0.026)
        q1[:, 1] = torch.clamp_min(q1[:, 1], 0.020)
        q1[:, 2] = torch.clamp_min(q1[:, 2], 0.018)
        q1[:, 3] = torch.clamp_max(q1[:, 3], 0.001)
        q1[:, 4:] = torch.clamp_max(q1[:, 4:], 0.004)
    elif profile_variant == 2:
        q1[:, 0] = torch.clamp_min(q1[:, 0], 0.030)
        q1[:, 1] = torch.clamp_min(q1[:, 1], 0.022)
        q1[:, 2] = torch.clamp_min(q1[:, 2], 0.018)
        q1[:, 3] = torch.clamp_max(q1[:, 3], 0.004)
        q1[:, 4:] = torch.clamp_max(q1[:, 4:], 0.002)
    elif profile_variant == 3:
        q1[:, 0] = torch.clamp_min(q1[:, 0], 0.028)
        q1[:, 1] = torch.clamp_min(q1[:, 1], 0.022)
        q1[:, 2] = torch.clamp_min(q1[:, 2], 0.018)
        q1[:, 3] = torch.clamp_max(q1[:, 3], 0.010)
        q1[:, 4:] = torch.clamp_max(q1[:, 4:], 0.004)
    elif profile_variant == 5:
        q1[:, 0] = torch.clamp_min(q1[:, 0], 0.028)
        q1[:, 1] = torch.clamp_min(q1[:, 1], 0.018)
        q1[:, 2] = torch.clamp_max(q1[:, 2], 0.004)
        q1[:, 3:] = torch.clamp_max(q1[:, 3:], 0.003)
    elif profile_variant == 6:
        q1[:, 0] = torch.clamp_min(q1[:, 0], 0.032)
        q1[:, 1] = torch.clamp_min(q1[:, 1], 0.018)
        q1[:, 2] = torch.clamp_max(q1[:, 2], 0.003)
        q1[:, 3:] = torch.clamp_max(q1[:, 3:], 0.002)
    elif profile_variant == 7:
        q1[:, :3] = torch.maximum(
            q1[:, :3],
            0.020 - 0.0012 * torch.arange(0, 3, dtype=torch.float64, device=device).reshape(1, 3),
        )
        q1[:, 3] = torch.clamp_max(q1[:, 3], 0.001)
        q1[:, 4:] = torch.clamp_max(q1[:, 4:], 0.003)
    elif profile_variant == 8:
        q1[:, 0] = torch.clamp_min(q1[:, 0], 0.028)
        q1[:, 1] = torch.clamp_min(q1[:, 1], 0.020)
        q1[:, 2] = torch.clamp_max(q1[:, 2], 0.004)
        q1[:, 3] = torch.clamp_max(q1[:, 3], 0.001)
        q1[:, 4:] = torch.clamp_max(q1[:, 4:], 0.003)
    elif profile_variant == 9:
        q1[:, 0] = torch.clamp_min(q1[:, 0], 0.028)
        q1[:, 1] = torch.clamp_min(q1[:, 1], 0.020)
        q1[:, 2] = torch.clamp_min(q1[:, 2], 0.016)
        q1[:, 3] = torch.clamp_max(q1[:, 3], 0.004)
        q1[:, 4] = torch.clamp_max(q1[:, 4], 0.001)
        q1[:, 5:] = torch.clamp_max(q1[:, 5:], 0.003)
    else:
        q1[:, :5] = torch.maximum(
            q1[:, :5],
            0.020 - 0.0012 * torch.arange(0, 5, dtype=torch.float64, device=device).reshape(1, 5),
        )

    q = q1.clone()
    prevsq = q1 * 0.98
    u1 = (5.0 + 0.2 * levels).repeat(ncol, 1)
    v1 = _full((ncol, KM), 1.0, device=device)
    dot = _full((ncol, KM), -0.025, device=device)
    qtr = _zeros((ncol, KM, NN), device=device)
    qtr[:, :, 0] = q1
    qtr[:, :, 1] = 1.0e-5
    qtr[:, :, 2:] = 1.0e-6
    qtr[:, :, NTK - 1] = 0.1

    islimsk = (((columns + 1) % 2) == 0).to(torch.float64)
    fields: dict[str, torch.Tensor] = {
        "tmf": _zeros((ncol, KM, NN), device=device),
        "qmicro": _zeros((ncol, KM), device=device),
        "delp": delp,
        "ten_t": _zeros((ncol, KM), device=device),
        "ten_u": _zeros((ncol, KM), device=device),
        "ten_v": _zeros((ncol, KM), device=device),
        "ten_q": _zeros((ncol, KM, NTR), device=device),
        "prslp": prslp,
        "psp": _full((ncol,), 100000.0, device=device),
        "phil": phil,
        "tkeh": _full((ncol, KM), 0.1, device=device),
        "qtr": qtr,
        "dqtr": _zeros((ncol, KM, NN), device=device),
        "prevsq": prevsq,
        "q": q,
        "q1": q1,
        "t1": t1,
        "u1": u1,
        "v1": v1,
        "fscav": _full((NTC,), 0.1, device=device),
        "rn": _zeros((ncol,), device=device),
        "garea": _full((ncol,), 1.0e8, device=device),
        "dot": dot,
        "hpbl": _full((ncol,), 700.0 if profile_variant >= 7 else 1800.0, device=device),
        "ud_mf": _zeros((ncol, KM), device=device),
        "dt_mf": _zeros((ncol, KM), device=device),
        "cnvw": _zeros((ncol, KM), device=device),
        "cnvc": _zeros((ncol, KM), device=device),
        "sigmain": _zeros((ncol, KM), device=device),
        "sigmaout": _zeros((ncol, KM), device=device),
        "omegain": _zeros((ncol, KM), device=device),
        "omegaout": _zeros((ncol, KM), device=device),
        "kcnv": _zeros((ncol,), device=device),
        "kbot": _zeros((ncol,), device=device),
        "ktop": _zeros((ncol,), device=device),
        "islimsk": islimsk,
        "gt0": t1.clone(),
        "gu0": u1.clone(),
        "gv0": v1.clone(),
        "gq0": _zeros((ncol, KM, NTR), device=device),
        "dudt": _zeros((ncol, KM), device=device),
        "dvdt": _zeros((ncol, KM), device=device),
        "dtdt": _zeros((ncol, KM), device=device),
        "dqdt": _zeros((ncol, KM, NTR), device=device),
        "rainc": _zeros((ncol,), device=device),
        "cnvprcp": _zeros((ncol,), device=device),
        "cnvw_phy_f3d": _zeros((ncol, KM), device=device),
        "cnvc_phy_f3d": _zeros((ncol, KM), device=device),
        "dtend": _zeros((ncol, KM, NDTEND), device=device),
        "cliq": _scalar(CLIQ, device=device),
        "cp": _scalar(CP, device=device),
        "cvap": _scalar(CVAP, device=device),
        "eps": _scalar(EPS, device=device),
        "epsm1": _scalar(EPSM1, device=device),
        "fv": _scalar(FV, device=device),
        "grav": _scalar(GRAV, device=device),
        "hvap": _scalar(HVAP, device=device),
        "rd": _scalar(RD, device=device),
        "rv": _scalar(RV, device=device),
        "t0c": _scalar(T0C, device=device),
        "delt": _scalar(DELT, device=device),
        "cscale": _scalar(CSCALE, device=device),
        "clam": _scalar(CLAM, device=device),
        "c0s": _scalar(C0S, device=device),
        "c1": _scalar(C1, device=device),
        "evef": _scalar(EVEF, device=device),
        "pgcon": _scalar(PGCON, device=device),
        "asolfac": _scalar(ASOLFAC, device=device),
        "betadcu": _scalar(BETADCU, device=device),
        "betamcu": _scalar(BETAMCU, device=device),
        "betascu": _scalar(BETASCU, device=device),
        "cat_adj_shal": _scalar(CAT_ADJ_SHAL, device=device),
    }
    fields["gq0"][:, :, 0] = q1

    config: dict[str, Any] = {
        "im": ncol,
        "km": KM,
        "nn": NN,
        "ntr": NTR,
        "ntc": NTC,
        "itc": ITC_DEFAULT,
        "ntk": NTK,
        "ncloud": NCLOUD,
        "ndtend": NDTEND,
        "first_time_step": True,
        "restart": False,
        "hwrf_samfshal": False,
        "progsigma": False,
        "progomega": False,
        "tend_opt_scnv": 1,
        "imp_physics": 8,
        "imp_physics_gfdl": 9,
        "npdf3d": 3,
        "num_p3d": 4,
        "ncnvcld3d": 1,
        "ntrac": NTR,
        "nsamftrac": NTR,
        "lssav": True,
        "ldiag3d": True,
        "qdiag3d": True,
        "shcnvcw": True,
        "flag_for_scnv_generic_tend": True,
        "cscnv": False,
        "satmedmf": True,
        "trans_trac": True,
        "ras": False,
    }
    return SaSasShallowConvectionState(fields, config)


def initial_state_case08_active_maritime_default(
    *,
    device: torch.device | str | None = None,
    ncol: int | None = None,
    ntrac: int = NTR,
) -> SaSasShallowConvectionState:
    """Build the deterministic active shallow profile used by the source driver."""

    return _state_from_profile_variant(9, device=device, ncol=ncol, ntrac=ntrac)


def initial_state_for_case(
    case_id: str,
    *,
    device: torch.device | str | None = None,
    ncol: int | None = None,
    ntrac: int = NTR,
) -> SaSasShallowConvectionState:
    """Build a materialized scenario-policy case with explicit branch config."""

    if case_id not in CASE_CONFIGS:
        raise KeyError(f"unknown SA-SAS shallow-convection case {case_id!r}")
    state = _state_from_profile_variant(CASE_PROFILE_VARIANTS.get(case_id, 9), device=device, ncol=ncol, ntrac=ntrac)
    fields = dict(state.fields)
    config = dict(state.config)
    config.update(CASE_CONFIGS[case_id])

    if config["progsigma"]:
        fields["sigmain"] = torch.full_like(fields["sigmain"], 0.06)
        fields["sigmaout"] = torch.zeros_like(fields["sigmaout"])
        fields["prevsq"] = fields["q1"] * 0.97
        if case_id == "case16_progsigma_warm_restart":
            config["first_time_step"] = False
            config["restart"] = False
    if config["progomega"]:
        fields["omegain"] = torch.full_like(fields["omegain"], -0.25)
    if case_id == "case00_kcnv_precluded_return":
        fields["kcnv"] = torch.ones_like(fields["kcnv"])
    if case_id in {"case02_pressure_depth_rejection", "case03_rh_rejection", "case05_cin_rejection", "case09_active_land_default"}:
        fields["islimsk"] = torch.ones_like(fields["islimsk"])
    if case_id == "case01_no_lfc_early_return":
        fields["q1"] = torch.full_like(fields["q1"], 1.0e-8)
        fields["q"] = fields["q1"].clone()
        fields["qtr"] = fields["qtr"].clone()
        fields["qtr"][:, :, 0] = fields["q1"]
        fields["dot"] = torch.full_like(fields["dot"], 0.02)
    if case_id == "case03_rh_rejection":
        fields["q1"] = torch.clamp_min(fields["q1"] * 0.05, 1.0e-8)
        fields["q"] = fields["q1"].clone()
        fields["qtr"] = fields["qtr"].clone()
        fields["qtr"][:, :, 0] = fields["q1"]
    if case_id in {"case02_pressure_depth_rejection", "case05_cin_rejection"}:
        fields["dot"] = torch.full_like(fields["dot"], 0.08)
    if case_id == "case04_entrained_lfc_rejection":
        fields["q1"] = fields["q1"].clone()
        fields["q1"][:, 3:] = torch.clamp_min(fields["q1"][:, 3:] * 0.1, 1.0e-8)
    if case_id == "case06_cloud_depth_rejection":
        fields["hpbl"] = torch.full_like(fields["hpbl"], 120.0)
    if case_id == "case07_negative_cloudwork_rejection":
        fields["t1"] = fields["t1"] + 8.0
    if case_id == "case18_small_grid_scaleaware":
        fields["garea"] = torch.full_like(fields["garea"], 1.0e6)
    if case_id == "case13_aerosol_wetdep_active":
        fields["fscav"] = torch.full_like(fields["fscav"], 0.5)
    if case_id == "case11_tke_entrainment_active":
        fields["qtr"] = fields["qtr"].clone()
        fields["qtr"][:, :, NTK - 1] = 0.2
    if case_id == "case22_scnv_post_gfdl_cloud_diag":
        config["imp_physics"] = config["imp_physics_gfdl"]
    if case_id == "case23_scnv_post_pdf_cloud_diag":
        config.update({"shcnvcw": True, "npdf3d": 3, "num_p3d": 4, "ncnvcld3d": 1})
    if case_id == "case24_scnv_post_dtend_tracer":
        config.update({"ldiag3d": True, "qdiag3d": True, "flag_for_scnv_generic_tend": True})
    return SaSasShallowConvectionState(fields, config)


def _runs_through(through: str, segment_id: str) -> bool:
    return _ORDER[through] >= _ORDER[segment_id]


def run_sa_sas_shallow_convection(state: SaSasShallowConvectionState, *, through: str = "A") -> SaSasShallowConvectionState:
    """Run cumulative SA-SAS shallow-convection segments A-Q for the sealed active path."""

    return _run_sa_sas_shallow_convection_mutable(state, through=through).immutable_copy()


def _run_sa_sas_shallow_convection_mutable(state: SaSasShallowConvectionState, *, through: str) -> SaSasShallowConvectionState:
    """Run cumulative segments and return the mutable working state."""

    if through not in SEGMENT_IDS:
        raise ValueError(f"unknown SA-SAS shallow-convection segment {through!r}")
    if through not in IMPLEMENTED_SEGMENTS:
        raise NotImplementedError(f"SA-SAS shallow-convection segment {through!r} is not implemented yet")
    return _in_scheme_context(state, lambda: _run_scheme(state.mutable_copy(), through))


def _in_scheme_context(state: SaSasShallowConvectionState, run):
    device = require_same_device(state.fields)
    if device.type == "cuda" and not compile_categories_explicitly_set() and hasattr(torch, "compile"):
        with compile_categories(DEFAULT_COMPILE_CATEGORIES):
            return run()
    return run()


def run_sa_sas_shallow_convection_production(state: SaSasShallowConvectionState) -> SaSasShallowConvectionState:
    """Run the full scheme and retain only host-visible production outputs."""

    full = _run_sa_sas_shallow_convection_mutable(state, through="Q")
    compact = {name: full.require(name) for name in PRODUCTION_FIELDS}
    return SaSasShallowConvectionState(compact, full.config)


def run_sa_sas_shallow_convection_compacted(
    state: SaSasShallowConvectionState,
    *,
    through: str,
    outputs: tuple[str, ...],
    fraction: float,
    region: str = "convection.shallow",
) -> SaSasShallowConvectionState:
    """Run A-E on every column and segments F onwards only on the triggered columns.

    As in ``samfshalcnv_run``, segments F onwards act only on columns with ``cnvflg`` set after
    the trigger (segment E); other columns keep their values at E in every field the source
    writes back. The triggered columns' fields and A-E locals at E are gathered into a buffer of
    ``fraction`` of the columns (``ufs_physics_common.compaction``) and run from F through
    ``through`` there; each column computes from its own values, so its result does not depend on
    the buffer. Returns ``outputs`` only: per-column outputs equal the dense run bit for bit; batch
    scalars (``errflg``) come from the compacted run. A buffer too small for the triggered columns
    sets ``region``'s overflow bit.
    """
    from ufs_physics_common.compaction import run_compacted

    if through not in _ORDER or _ORDER[through] <= _ORDER["E"]:
        raise ValueError("shallow compaction ends after the trigger (segment E)")
    if through not in IMPLEMENTED_SEGMENTS:
        raise NotImplementedError(f"SA-SAS shallow-convection segment {through!r} is not implemented yet")
    at_e = state.mutable_copy()
    env = _in_scheme_context(at_e, lambda: _run_scheme_through_e(at_e, through))
    fields = at_e.fields
    ncol = fields["cnvflg"].shape[0]
    missing = [name for name in outputs if name not in fields]
    if missing:
        raise KeyError(f"shallow compaction outputs must exist at the trigger: {missing}")

    def is_column(value) -> bool:
        return isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == ncol

    # Each column tensor at E enters the buffer once: a field and the local bound to it stay one
    # tensor there, as in the dense run (segments F onwards update some in place).
    columns: dict[str, torch.Tensor] = {}

    def slot(value: torch.Tensor) -> str:
        for key, seen in columns.items():
            if seen is value:
                return key
        key = f"column{len(columns)}"
        columns[key] = value
        return key

    field_slots = {name: slot(value) for name, value in fields.items() if is_column(value)}
    env_slots = {name: slot(value) for name, value in env.items() if is_column(value)}
    gathered = tuple(columns)
    # Non-triggered columns take their values at E: those enter as the merge's old values.
    kept = [name for name in outputs if is_column(fields[name])]
    columns.update({f"at_e.{name}": fields[name] for name in kept})
    batch = {}

    def scheme(part: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        size = part[field_slots["q1"]].shape[0]
        config = state.config
        if "im" in config:  # the declared column count is the buffer's
            config = {**config, "im": size}
        buffer = SaSasShallowConvectionState(
            {name: part[field_slots[name]] if name in field_slots else value for name, value in fields.items()},
            config, mutable=True)
        locals_e = {name: part[env_slots[name]] if name in env_slots else value for name, value in env.items()}
        locals_e["ncol"] = size
        current = _in_scheme_context(buffer, lambda: _run_scheme(buffer, through, locals_e))
        batch.update({name: current.require(name) for name in outputs if name not in kept})
        return {f"at_e.{name}": current.require(name) for name in kept}

    def tensor_call(fn, *args):
        return run_compile_island("recurrence_chunks", fn.__name__, fn, *args)

    merged = run_compacted(scheme, columns, fields["cnvflg"] != 0.0, region=region,
                           fraction=fraction, tensor_call=tensor_call, inputs=gathered)
    return SaSasShallowConvectionState(
        {name: batch[name] if name in batch else merged[f"at_e.{name}"] for name in outputs}, state.config)


def _run_scheme_through_e(state: SaSasShallowConvectionState, through: str) -> dict[str, Any] | None:
    """Segments A-E on ``state`` (fields updated in place); the locals segment F reads, or None
    when ``through`` ends at or before E."""
    fields = state.fields
    device = require_same_device(fields)
    q1 = fields["q1"]
    if q1.ndim != 2 or q1.shape[0] < 1 or q1.shape[1] < 2:
        raise ValueError("q1 needs a nonempty column axis and at least two levels")
    ncol, KM = q1.shape
    qtr_shape = fields["qtr"].shape
    if len(qtr_shape) != 3 or qtr_shape[:2] != q1.shape or qtr_shape[2] < 3:
        raise ValueError("qtr must have shape [ncol, nz, ntr+2]")
    NN = qtr_shape[2]
    NTC = fields["fscav"].numel()
    layout = tracer_layout(state)
    # Legacy harness table: ntr = ntrac = nn - 2. A declared layout (G3.2 oracle MPAS
    # table) gives ntr = nsamftrac independent of nn and of the host tracer count.
    NTR = layout.ntr
    NTRAC = layout.ntrac
    ITC = int(state.config.get("itc", ITC_DEFAULT if layout.name == "legacy" else layout.itc))
    if layout.name != "legacy" and NN != layout.nn:
        raise ValueError(f"nn metadata disagrees with input shapes: expected {layout.nn}")
    for name, value in (("im", ncol), ("km", KM), ("nn", NN), ("ntr", NTR),
                        ("ntrac", NTRAC), ("nsamftrac", NTR), ("ntc", NTC)):
        if name in state.config and state.config[name] != value:
            raise ValueError(f"{name} metadata disagrees with input shapes: expected {value}")
    ntk = int(state.config.get("ntk", NTK))
    if ntk < 0 or ntk > NN or (NTC and ITC and (ITC < 1 or ITC + NTC - 1 > NN)) or NTR + 2 > NN:
        raise ValueError("tracer slot metadata exceeds the qtr axis")
    shapes = {"scalar": (), "column": (ncol,), "level": (ncol, KM),
              "source_tracer": (ncol, KM, NN), "tracer": (ncol, KM, NTRAC),
              "chemistry": (NTC,), "diagnostic": (ncol, KM, int(state.config.get("ndtend", NDTEND)))}
    for group, names in _INPUT_GROUPS.items():
        for name in names:
            require_real(name, fields[name], shapes[group])
    # The generic post (segment Q) writes dtend; a call that stops earlier needs no slots
    # (ldiag3d off allocates none).
    if (layout.name == "legacy" and NTR < 8) or (_runs_through(through, "Q") and fields["dtend"].shape[2] < NTRAC + 4):
        raise ValueError("the declared generic-post diagnostic contract needs eight tracers and ntr+4 dtend slots")
    dtype = q1.dtype
    # The source cap owns these scalars. Profile fixtures happen to match the
    # module defaults, but evolved coupled calls use a different dt and clam.
    C1, CAT_ADJ_SHAL, CLAM = (fields[n] for n in ("c1", "cat_adj_shal", "clam"))
    CP, CSCALE, DELT = (fields[n] for n in ("cp", "cscale", "delt"))
    EPS, EPSM1, EVEF = (fields[n] for n in ("eps", "epsm1", "evef"))
    FV, GRAV, HVAP = (fields[n] for n in ("fv", "grav", "hvap"))
    PGCON, RD = (fields[n] for n in ("pgcon", "rd"))

    t1 = fields["t1"]
    u1 = fields["u1"]
    v1 = fields["v1"]
    qtr = fields["qtr"]
    delp = fields["delp"]
    prslp = fields["prslp"]
    psp = fields["psp"]
    phil = fields["phil"]
    tkeh = fields["tkeh"]
    dot = fields["dot"]
    hpbl = fields["hpbl"]
    tmf = fields["tmf"]
    qmicro = fields["qmicro"]
    prevsq = fields["prevsq"]
    q = fields["q"]
    fscav = fields["fscav"]
    garea = fields["garea"]
    islimsk = fields["islimsk"]
    kcnv = fields["kcnv"].clone()
    kbot = fields["kbot"].clone()
    ktop = fields["ktop"].clone()

    ten_t = torch.zeros_like(t1)
    ten_u = torch.zeros_like(u1)
    ten_v = torch.zeros_like(v1)
    ten_q = torch.zeros((ncol, KM, NTRAC), dtype=dtype, device=device)
    dqtr = torch.zeros_like(qtr)
    new_t1 = t1.clone()
    new_u1 = u1.clone()
    new_v1 = v1.clone()
    new_q1 = q1.clone()
    new_qtr = qtr.clone()
    zero_scalar = torch.zeros((), dtype=dtype, device=device)
    fields.update(
        {
            "ten_t": ten_t,
            "ten_u": ten_u,
            "ten_v": ten_v,
            "ten_q": ten_q,
            "dqtr": dqtr,
            "new_t1": new_t1,
            "new_u1": new_u1,
            "new_v1": new_v1,
            "new_q1": new_q1,
            "new_qtr": new_qtr,
            "errmsg": zero_scalar.clone(),
            "errflg": zero_scalar.clone(),
        }
    )
    if through == "A":
        return None

    gravinv = 1.0 / fields["grav"]
    invdelt = 1.0 / fields["delt"]
    elocp = fields["hvap"] / fields["cp"]
    el2orc = fields["hvap"] * fields["hvap"] / (fields["rv"] * fields["cp"])
    fact1 = (fields["cvap"] - fields["cliq"]) / fields["rv"]
    fact2 = fields["hvap"] / fields["rv"] - fact1 * fields["t0c"]
    hwrf = bool(state.config.get("hwrf_samfshal", False))
    progsigma = bool(state.config.get("progsigma", False))
    progomega = bool(state.config.get("progomega", False))
    first_time_step = bool(state.config.get("first_time_step", True))
    restart = bool(state.config.get("restart", False))
    ntk = int(state.config.get("ntk", NTK))
    ntr = int(state.config.get("ntr", NTR))
    ntc = int(state.config.get("ntc", NTC))
    itc = int(state.config.get("itc", ITC))
    ncloud = int(state.config.get("ncloud", NCLOUD))

    km1 = KM - 1
    do_aerosols_bool = False if hwrf else (itc > 0 and ntc > 0 and ntr > 0 and ntr >= itc + ntc - 3)
    do_aerosols = torch.full((ncol,), 1.0 if do_aerosols_bool else 0.0, dtype=dtype, device=device)
    ps = psp * 0.001
    prsl = prslp * 0.001
    del_ = delp * 0.001
    chem_c = torch.zeros((ncol, KM, NTC), dtype=dtype, device=device)
    chem_pw = torch.zeros_like(chem_c)
    wet_dep = torch.zeros_like(chem_c)
    cnvflg_bool = kcnv != 1.0
    cnvflg = cnvflg_bool.to(dtype)
    kbot = _where(cnvflg_bool, float(KM + 1), kbot)
    ktop = _where(cnvflg_bool, 0.0, ktop)
    sfcpbl = 0.2 * hpbl
    rn = torch.zeros_like(ps)
    kbcon = torch.full_like(ps, float(KM))
    ktcon = torch.ones_like(ps)
    kb = torch.full_like(ps, float(KM))
    pdot = torch.zeros_like(ps)
    qlko_ktcon = torch.zeros_like(ps)
    aa1 = torch.zeros_like(ps)
    cina = torch.zeros_like(ps)
    advfac = torch.zeros_like(ps)
    gdx = torch.sqrt(garea)
    xmb = torch.zeros_like(ps)
    c0 = _where(islimsk == 1.0, fields["c0s"] * fields["asolfac"], fields["c0s"].expand_as(ps))
    small_grid = gdx < 9000.0
    c0 = _where(small_grid, c0 * (gdx / 9000.0) ** 3, c0)
    c0t = _where(t1 > 273.16, c0[:, None], c0[:, None] * torch.exp(0.001 * (t1 - 273.16)))
    cnvw = torch.zeros_like(t1)
    cnvc = torch.zeros_like(t1)
    ud_mf = torch.zeros_like(t1)
    dt_mf = torch.zeros_like(t1)
    fields.update(
        {
            "do_aerosols": do_aerosols,
            "ps": ps,
            "prsl": prsl,
            "del": del_,
            "cnvflg": cnvflg,
            "c0": c0,
            "c0t": c0t,
            "cnvw": cnvw,
            "cnvc": cnvc,
            "ud_mf": ud_mf,
            "dt_mf": dt_mf,
            # The source's initial outputs, so that later boundaries hold the values a
            # non-convecting column ends with (run_sa_sas_shallow_convection_compacted).
            "kbot": kbot,
            "ktop": ktop,
            "rn": rn,
            "ten_t": ten_t,
            "ten_u": ten_u,
            "ten_v": ten_v,
            "ten_q": ten_q,
            "dqtr": dqtr,
        }
    )
    if through == "B":
        return None

    aafac = 0.1
    crtlame = 1.0e-4
    w1l, w2l, w3l, w4l = -8.0e-3, -4.0e-2, -5.0e-3, -5.0e-4
    w1s, w2s, w3s, w4s = -2.0e-4, -2.0e-3, -1.0e-3, -2.0e-5
    clamd, tkemx, tkemn = 0.1, 0.65, 0.05
    dtke = tkemx - tkemn
    cthk, cthkmn, dthk = 200.0, 0.0, 25.0
    rhcrt = 0.75
    cinpcrmx, cinpcrmn = 180.0, 120.0
    cinacrmx = -120.0
    cinacrmn = -80.0 if not hwrf else 0.0
    dtmax, dtmin = 10800.0, 600.0
    bb1, bb2, csmf = 4.0, 0.8, 0.2
    tkcrt, cmxfac = 2.0, 10.0
    betaw = 0.03
    dxcrt = 10000.0 if progsigma else 15000.0
    dxcrtas = 500000.0
    wc_min = 0.2
    cm = 1.0
    cq = 1.0
    shevf = 2.0

    tx1 = 1.0 / ps
    levels_full = _level_values(1, KM + 1, reference=ps)
    ratio = prsl * tx1[:, None]
    kbm = (ratio > 0.70).to(dtype).sum(dim=1) + 1.0
    kmax = (ratio > 0.60).to(dtype).sum(dim=1) + 1.0
    kbm = torch.minimum(kbm, kmax)
    zo = phil / fields["grav"]
    zi = torch.zeros_like(zo)
    xlamue = torch.zeros_like(zo)
    zi[:, : KM - 1] = 0.5 * (zo[:, : KM - 1] + zo[:, 1:KM])
    if hwrf:
        xlamue[:, : KM - 1] = fields["clam"] / zi[:, : KM - 1]
    if hwrf:
        xlamue[:, KM - 1] = xlamue[:, KM - 2]
    flg = cnvflg_bool.clone()
    kpbl = torch.ones_like(ps)
    levels_kpbl = _level_values(2, KM, reference=ps)
    kpbl_candidate = _last_prefix_true_level(zo[:, 1 : KM - 1] <= hpbl[:, None], levels_kpbl, kpbl)
    kpbl = _where(flg, kpbl_candidate, kpbl)
    kpbl = torch.minimum(kpbl, kbm)

    pfld = torch.zeros_like(t1)
    eta = torch.zeros_like(t1)
    rh = torch.zeros_like(t1)
    hcko = torch.zeros_like(t1)
    qcko = torch.zeros_like(t1)
    qrcko = torch.zeros_like(t1)
    ucko = torch.zeros_like(t1)
    vcko = torch.zeros_like(t1)
    dbyo = torch.zeros_like(t1)
    pwo = torch.zeros_like(t1)
    dellal = torch.zeros_like(t1)
    to = torch.zeros_like(t1)
    qo = torch.zeros_like(t1)
    uo = torch.zeros_like(t1)
    vo = torch.zeros_like(t1)
    wu2 = torch.zeros_like(t1)
    buo = torch.zeros_like(t1)
    wush = torch.zeros_like(t1)
    drag = torch.zeros_like(t1)
    cnvwt = torch.zeros_like(t1)
    po = torch.zeros_like(t1)
    dbyo1 = torch.zeros_like(t1)
    zdqca = torch.zeros_like(t1)
    omega_u = torch.zeros_like(t1)
    omegaout = fields["omegaout"].clone()
    zeta = torch.ones_like(t1)
    ctr = torch.zeros((ncol, KM, NTR), dtype=dtype, device=device)
    ctro = torch.zeros_like(ctr)
    ecko = torch.zeros_like(ctr)
    ercko = torch.zeros_like(ctr)
    active_grid = cnvflg_bool[:, None] & (torch.arange(1, KM + 1, dtype=dtype, device=device).reshape(1, KM) <= kmax[:, None])
    pfld = _where(active_grid, prsl * 10.0, pfld)
    eta = _where(active_grid, torch.ones_like(eta), eta)
    to = _where(active_grid, t1, to)
    qo = _where(active_grid, q1, qo)
    uo = _where(active_grid, u1, uo)
    vo = _where(active_grid, v1, vo)
    if not hwrf:
        source_tracers = qtr[:, :, 2 : 2 + NTR]
        ctr = _where(active_grid[:, :, None], source_tracers, ctr)
        ctro = _where(active_grid[:, :, None], source_tracers, ctro)
    qeso = torch.zeros_like(t1)
    es = 0.01 * fpvs(to)
    qeso_calc = EPS * es / (pfld + EPSM1 * es)
    qeso = _where(active_grid, torch.clamp_min(qeso_calc, 1.0e-8), qeso)
    qo = _where(active_grid, torch.clamp_min(qo, 1.0e-10), qo)
    tem_grid = phil + CP * to
    heo = _where(active_grid, tem_grid + HVAP * qo, 0.0)
    heso = _where(active_grid, tem_grid + HVAP * qeso, 0.0)
    fields.update(
        {
            "zo": zo,
            "zi": zi,
            "qeso": qeso,
            "heo": heo,
            "heso": heso,
            "to": to,
            "qo": qo,
            "uo": uo,
            "vo": vo,
            "kpbl": kpbl,
            "kbm": kbm,
            "kmax": kmax,
            "pfld": pfld,
            "eta": eta,
            "rh": rh,
            "hcko": hcko,
            "qcko": qcko,
            "qrcko": qrcko,
            "ucko": ucko,
            "vcko": vcko,
            "dbyo": dbyo,
            "pwo": pwo,
            "dellal": dellal,
            "ctr": ctr,
            "ctro": ctro,
            "ecko": ecko,
            "ercko": ercko,
            "chem_c": chem_c,
            "chem_pw": chem_pw,
            "wet_dep": wet_dep,
        }
    )
    if through == "C":
        return None

    levels_full = _level_values(1, KM + 1, reference=ps)
    levels_inner = _level_values(2, KM, reference=ps)
    kb, kb1, kbcon, cnvflg_bool, to, qo, qeso, po, rh, heo, heso, uo, vo, ctro = run_compile_island(
        "trigger", "environment_adjustment", _d_environment_block,
        cnvflg_bool, sfcpbl, kpbl, kbm, kmax, kbcon, zo, pfld, to, qo, qeso, po, rh, uo, vo, ctro, heo, heso,
        CP, EPS, EPSM1, GRAV, HVAP, fact1, fact2, el2orc,
        _fpvs_table_for_device(str(q1.device), q1.dtype), hwrf=hwrf)
    cnvflg = cnvflg_bool.to(dtype)
    fields.update({"kb": kb, "kb1": kb1, "kbcon": kbcon, "cnvflg": cnvflg, "pfld": pfld, "po": po,
                   "to": to, "qo": qo, "qeso": qeso, "rh": rh, "heo": heo, "heso": heso,
                   "uo": uo, "vo": vo, "ctro": ctro})
    if through == "D":
        return None

    cnvflg_bool, kb, kbcon, pdot, w3, w4 = run_compile_island(
        "trigger", "cloud_base", _e_cloud_base_block,
        cnvflg_bool, kb, kbcon, kpbl, kbm, kmax, dot, zi, pfld, heo, heso, hpbl, islimsk, pdot)

    levels_left = _level_values(1, KM, reference=ps)
    dz_left = zo[:, 1:KM] - zo[:, : KM - 1]
    layer_between_kb_kbcon = cnvflg_bool[:, None] & (levels_left >= kb[:, None]) & (levels_left < kbcon[:, None])
    rhbar = _masked_sum(layer_between_kb_kbcon, rh[:, : KM - 1] * dz_left)
    sumx = _masked_sum(layer_between_kb_kbcon, dz_left)
    rhbar = _where_div(cnvflg_bool, rhbar, sumx, rhbar)
    cnvflg_bool = cnvflg_bool & (rhbar >= rhcrt)

    tkemean = torch.zeros_like(ps)
    clamt = torch.zeros_like(ps)
    xlamud = torch.zeros_like(ps)
    if hwrf:
        xlamud = _where(cnvflg_bool, _gather2(xlamue, kbcon), xlamud)
    else:
        if ntk > 0:
            layer_between_kb_kbcon = cnvflg_bool[:, None] & (levels_left >= kb[:, None]) & (levels_left < kbcon[:, None])
            tkemean = _masked_sum(layer_between_kb_kbcon, tkeh[:, : KM - 1] * dz_left)
            sumx = _masked_sum(layer_between_kb_kbcon, dz_left)
            tkemean = _where_div(cnvflg_bool, tkemean, sumx, tkemean)
            clamt = _where(
                cnvflg_bool & (tkemean > tkemx),
                CLAM + clamd,
                _where(
                    cnvflg_bool & (tkemean < tkemn),
                    CLAM - clamd,
                    _where(cnvflg_bool, CLAM + clamd * (1.0 - 2.0 * (tkemx - tkemean) / dtke), clamt),
                ),
            )
            tem = 1.0 + tkemean / tkcrt
            clamt = _where(cnvflg_bool & (tkemean > tkcrt), torch.clamp_max(tem, cmxfac) * CLAM, clamt)
        else:
            clamt = _where(cnvflg_bool, CLAM, clamt)
        val = torch.clamp_min(clamt[:, None] / (zi[:, : KM - 1] + dz_left), crtlame)
        xlamue[:, : KM - 1] = _where(cnvflg_bool[:, None], val, xlamue[:, : KM - 1].clone())
        xlamue[:, KM - 1] = _where(cnvflg_bool, xlamue[:, KM - 2], xlamue[:, KM - 1])
        xlamud = _where(cnvflg_bool, 0.001 * clamt, xlamud)
    cnvflg = cnvflg_bool.to(dtype)
    fields.update({"pdot": pdot, "cina": cina, "rhbar": rhbar, "tkemean": tkemean, "clamt": clamt, "xlamue": xlamue, "xlamud": xlamud, "cnvflg": cnvflg, "return_stage": torch.zeros_like(ps)})
    if through == "E":
        return None
    return dict(zip(_E_LOCALS, (
        C1, CAT_ADJ_SHAL, CP, CSCALE, DELT, EPS, EPSM1, EVEF, FV, GRAV, HVAP, KM, NTC, NTR, PGCON, RD, aa1,
        aafac, advfac, bb1, bb2, betaw, buo, c0t, chem_c, chem_pw, cina, cinacrmn, cinacrmx, cm, cnvc,
        cnvflg_bool, cnvw, cnvwt, cq, csmf, cthk, cthkmn, ctr, ctro, dbyo, dbyo1, del_, dellal, delp, device,
        do_aerosols_bool, dot, drag, dt_mf, dthk, dtmax, dtmin, dtype, dxcrt, dxcrtas, ecko, el2orc,
        elocp, ercko, eta, first_time_step, fscav, garea, gdx, gravinv, hcko, heo, heso, hwrf, invdelt,
        itc, kb, kbcon, kbm, kbot, kcnv, kmax, ktcon, ktop, levels_inner, ncloud, ncol, new_q1, new_qtr,
        new_t1, new_u1, new_v1, ntc, ntk, ntr, omega_u, omegaout, pdot, pfld, po, prevsq, progomega,
        progsigma, ps, pwo, q, q1, qcko, qeso, qlko_ktcon, qmicro, qo, qrcko, qtr, restart, rn, shevf, t1,
        ten_q, tmf, to, u1, ucko, ud_mf, uo, v1, vcko, vo, w3, w4, wc_min, wet_dep, wu2, wush, xlamud, xlamue,
        xmb, zdqca, zeta, zi, zo,)))


# Locals of segments A-E that segments F onwards read (_run_scheme_through_e -> _run_scheme).
_E_LOCALS = (
    "C1", "CAT_ADJ_SHAL", "CP", "CSCALE", "DELT", "EPS", "EPSM1", "EVEF", "FV", "GRAV", "HVAP", "KM", "NTC",
    "NTR", "PGCON", "RD", "aa1", "aafac", "advfac", "bb1", "bb2", "betaw", "buo", "c0t", "chem_c", "chem_pw",
    "cina", "cinacrmn", "cinacrmx", "cm", "cnvc", "cnvflg_bool", "cnvw", "cnvwt", "cq", "csmf", "cthk",
    "cthkmn", "ctr", "ctro", "dbyo", "dbyo1", "del_", "dellal", "delp", "device", "do_aerosols_bool", "dot",
    "drag", "dt_mf", "dthk", "dtmax", "dtmin", "dtype", "dxcrt", "dxcrtas", "ecko", "el2orc", "elocp",
    "ercko", "eta", "first_time_step", "fscav", "garea", "gdx", "gravinv", "hcko", "heo", "heso",
    "hwrf", "invdelt", "itc", "kb", "kbcon", "kbm", "kbot", "kcnv", "kmax", "ktcon", "ktop", "levels_inner",
    "ncloud", "ncol", "new_q1", "new_qtr", "new_t1", "new_u1", "new_v1", "ntc", "ntk", "ntr", "omega_u",
    "omegaout", "pdot", "pfld", "po", "prevsq", "progomega", "progsigma", "ps", "pwo", "q", "q1", "qcko",
    "qeso", "qlko_ktcon", "qmicro", "qo", "qrcko", "qtr", "restart", "rn", "shevf", "t1", "ten_q", "tmf",
    "to", "u1", "ucko", "ud_mf", "uo", "v1", "vcko", "vo", "w3", "w4", "wc_min", "wet_dep", "wu2", "wush",
    "xlamud", "xlamue", "xmb", "zdqca", "zeta", "zi", "zo",
)


def _run_scheme(state: SaSasShallowConvectionState, through: str,
                env: dict[str, Any] | None = None) -> SaSasShallowConvectionState:
    """Segments A through ``through`` on ``state`` (fields updated in place).

    With ``env`` (the locals of ``_run_scheme_through_e``) the run resumes at F from the fields at
    E. Columns are independent from F on: ``env`` and ``state.fields`` may hold any subset of the
    columns at E (``run_sa_sas_shallow_convection_compacted``), with ``env["ncol"]`` their count.
    """
    if env is None:
        env = _run_scheme_through_e(state, through)
        if env is None:
            return state
    fields = state.fields
    (
        C1, CAT_ADJ_SHAL, CP, CSCALE, DELT, EPS, EPSM1, EVEF, FV, GRAV, HVAP, KM, NTC, NTR, PGCON, RD, aa1,
        aafac, advfac, bb1, bb2, betaw, buo, c0t, chem_c, chem_pw, cina, cinacrmn, cinacrmx, cm, cnvc,
        cnvflg_bool, cnvw, cnvwt, cq, csmf, cthk, cthkmn, ctr, ctro, dbyo, dbyo1, del_, dellal, delp, device,
        do_aerosols_bool, dot, drag, dt_mf, dthk, dtmax, dtmin, dtype, dxcrt, dxcrtas, ecko, el2orc,
        elocp, ercko, eta, first_time_step, fscav, garea, gdx, gravinv, hcko, heo, heso, hwrf, invdelt,
        itc, kb, kbcon, kbm, kbot, kcnv, kmax, ktcon, ktop, levels_inner, ncloud, ncol, new_q1, new_qtr,
        new_t1, new_u1, new_v1, ntc, ntk, ntr, omega_u, omegaout, pdot, pfld, po, prevsq, progomega,
        progsigma, ps, pwo, q, q1, qcko, qeso, qlko_ktcon, qmicro, qo, qrcko, qtr, restart, rn, shevf, t1,
        ten_q, tmf, to, u1, ucko, ud_mf, uo, v1, vcko, vo, w3, w4, wc_min, wet_dep, wu2, wush, xlamud, xlamue,
        xmb, zdqca, zeta, zi, zo,
    ) = [env[name] for name in _E_LOCALS]

    (
        eta,
        kbm,
        kmax,
        hcko,
        ucko,
        vcko,
        ecko,
        ercko,
        dbyo,
        chem_c,
        chem_pw,
    ) = run_compile_island(
        "f_recurrence",
        "plume_recurrence",
        _f_recurrence_block,
        cnvflg_bool,
        kb,
        kbcon,
        kbm,
        kmax,
        zi,
        xlamue,
        xlamud,
        eta,
        heo,
        heso,
        hcko,
        ucko,
        vcko,
        uo,
        vo,
        dbyo,
        ctro,
        ecko,
        ercko,
        chem_c,
        chem_pw,
        c0t,
        fscav,
        cm=cm,
        cq=cq,
        pgcon=PGCON,
        hwrf=hwrf,
        do_aerosols_bool=do_aerosols_bool,
        itc=itc,
        ntc=ntc,
        ntk=ntk,
        ntr=ntr,
        km=KM,
    )
    fields.update({"eta": eta, "hcko": hcko, "qcko": qcko, "ucko": ucko, "vcko": vcko, "ecko": ecko, "ercko": ercko, "chem_c": chem_c, "chem_pw": chem_pw, "xlamue": xlamue, "kbm": kbm, "kmax": kmax})
    if through == "F":
        return state

    kbcon1 = _first_true_level(
        cnvflg_bool[:, None] & (levels_inner < kbm[:, None]) & (levels_inner >= kbcon[:, None]) & (dbyo[:, 1 : KM - 1] > 0.0),
        levels_inner,
        _where(cnvflg_bool, kmax, 0.0),
    )
    cnvflg_bool = cnvflg_bool & (kbcon1 != kmax)
    tem_depth = _gather2(pfld, kbcon) - _gather2(pfld, kbcon1)
    cnvflg_bool = cnvflg_bool & ~(tem_depth > dthk)
    levels_g = _level_values(2, KM, reference=ps)
    cond_g_cin = cnvflg_bool[:, None] & (levels_g > kb[:, None]) & (levels_g < kbcon1[:, None])
    dz1_g = zo[:, 2:KM] - zo[:, 1 : KM - 1]
    safe_to_g = torch.where(cond_g_cin, to[:, 1 : KM - 1], 1.0)
    gamma_g = el2orc * qeso[:, 1 : KM - 1] / (safe_to_g * safe_to_g)
    rfact_g = 1.0 + FV * CP * gamma_g * safe_to_g / HVAP
    cin_add_g = dz1_g * (GRAV / (CP * safe_to_g)) * dbyo[:, 1 : KM - 1] / (1.0 + gamma_g) * rfact_g
    cin_add_g = cin_add_g + dz1_g * GRAV * FV * torch.clamp_min(qeso[:, 1 : KM - 1] - qo[:, 1 : KM - 1], 0.0)
    cina = cina + _masked_sum(cond_g_cin, cin_add_g)
    tem = _where(
        pdot <= w4,
        (pdot - w4) / (w3 - w4),
        _where(pdot >= -w4, -(pdot + w4) / (w4 - w3), 0.0),
    )
    tem = torch.clamp(tem, min=-1.0, max=1.0)
    cinacr = cinacrmx - (1.0 - tem) * (0.5 * (cinacrmx - cinacrmn))
    cnvflg_bool = cnvflg_bool & (cina >= cinacr)
    ktcon = _first_true_level(
        cnvflg_bool[:, None] & (levels_inner < kbm[:, None]) & (levels_inner > kbcon1[:, None]) & (dbyo[:, 1 : KM - 1] < 0.0),
        levels_inner,
        _where(cnvflg_bool, 1.0, ktcon),
    )
    del_kbcon = _gather2(del_, kbcon)
    pfld_kbcon, pfld_ktcon = _gather2_pair(pfld, kbcon, ktcon)
    cloud_depth = pfld_kbcon - pfld_ktcon
    cnvflg_bool = cnvflg_bool & ~((cloud_depth > cthk) | (cloud_depth < cthkmn))
    dp_base = 1000.0 * del_kbcon
    xmbmax = _where(cnvflg_bool, dp_base / (GRAV * DELT), 0.0)
    cnvflg = cnvflg_bool.to(dtype)
    fields.update({"kbcon1": kbcon1, "ktcon": ktcon, "ktcon1": torch.zeros_like(ktcon), "xmbmax": xmbmax, "dbyo": dbyo, "cnvflg": cnvflg, "return_stage": torch.zeros_like(ps), "cina": cina})
    if through == "G":
        return state

    qo_kb = _gather2(qo, kb)
    qcko, qrcko = _put_dynamic2_many((qcko, qrcko), kb, (qo_kb, qo_kb), cnvflg_bool)
    aa1 = _where(cnvflg_bool, 0.0, aa1)
    (
        qcko,
        qrcko,
        pwo,
        cnvwt,
        dellal,
        zdqca,
        buo,
        drag,
        wush,
    ) = run_compile_island(
        "h_recurrence",
        "precip_recurrence",
        _h_recurrence_block,
        cnvflg_bool,
        kb,
        kbcon,
        ktcon,
        zi,
        qeso,
        to,
        dbyo,
        xlamue,
        xlamud,
        eta,
        qo,
        del_,
        c0t,
        qcko,
        qrcko,
        pwo,
        cnvwt,
        dellal,
        zdqca,
        buo,
        drag,
        wush,
        uo,
        vo,
        cq=cq,
        c1=C1,
        csmf=csmf,
        el2orc=el2orc,
        hvap=HVAP,
        grav=GRAV,
        cp=CP,
        fv=FV,
        km=KM,
    )
    aa1 = _where(cnvflg_bool, 0.0, aa1)
    levels_h = _level_values(2, KM, reference=ps)
    cond_h_sum = cnvflg_bool[:, None] & (levels_h >= kbcon[:, None]) & (levels_h < ktcon[:, None])
    dz1_h = zo[:, 2 : KM] - zo[:, 1 : KM - 1]
    aa1 = aa1 + _masked_sum(cond_h_sum, buo[:, 1 : KM - 1] * dz1_h)
    dbyo1[:, 1 : KM - 1] = _where(cond_h_sum, hcko[:, 1 : KM - 1] - heso[:, 1 : KM - 1], dbyo1[:, 1 : KM - 1].clone())
    cnvflg_bool = cnvflg_bool & (aa1 > 0.0)
    fields.update({"qcko": qcko, "qrcko": qrcko, "pwo": pwo, "cnvwt": cnvwt, "dellal": dellal, "zdqca": zdqca, "buo": buo, "drag": drag, "aa1": aa1, "dbyo1": dbyo1, "cnvflg": cnvflg_bool.to(dtype)})
    if through == "H":
        return state

    aa1 = _where(cnvflg_bool, aafac * aa1, aa1)
    levels_i = _level_values(2, KM, reference=ps)
    cond_i_cross = cnvflg_bool[:, None] & (levels_i >= ktcon[:, None]) & (levels_i < kbm[:, None])
    dz1_i = zo[:, 2:KM] - zo[:, 1 : KM - 1]
    safe_to_i = torch.where(cond_i_cross, to[:, 1 : KM - 1], 1.0)
    gamma_i = el2orc * qeso[:, 1 : KM - 1] / (safe_to_i * safe_to_i)
    rfact_i = 1.0 + FV * CP * gamma_i * safe_to_i / HVAP
    add_i = dz1_i * (GRAV / (CP * safe_to_i)) * dbyo[:, 1 : KM - 1] / (1.0 + gamma_i) * rfact_i
    add_i = _masked_values(cond_i_cross, add_i)
    stop_i = torch.zeros_like(cond_i_cross)
    if add_i.shape[1]:
        # aa1 enters first: the source adds each level's term to the running aa1.
        aa1_path = _place(add_i, add_i[:, :1] + aa1[:, None], 0).cumsum(dim=1)
        stop_i = cond_i_cross & (aa1_path < 0.0)
    ktcon1 = _first_true_level(stop_i, levels_i, _where(cnvflg_bool, kbm, 0.0))
    if stop_i.shape[1]:
        stop_idx = stop_i.to(torch.long).argmax(dim=1)
        aa1_at_stop = aa1_path.gather(1, stop_idx.reshape(-1, 1)).squeeze(1)
        aa1 = _where(stop_i.any(dim=1), aa1_at_stop, aa1_path[:, -1])
    qcko, qrcko, pwo, cnvwt, dellal, zdqca = run_compile_island(
        "i_recurrence",
        "precip_recurrence",
        _i_precip_recurrence_block,
        cnvflg_bool,
        ktcon,
        ktcon1,
        zi,
        qeso,
        to,
        dbyo,
        xlamue,
        eta,
        qo,
        del_,
        c0t,
        qcko,
        qrcko,
        pwo,
        cnvwt,
        dellal,
        zdqca,
        cq=cq,
        c1=C1,
        el2orc=el2orc,
        hvap=HVAP,
        grav=GRAV,
        km=KM,
    )
    if hwrf:
        kidx = kbcon1
        po_kidx, to_kidx, dot_kidx = _gather2_many(kidx, po, to, dot)
        safe_to_kidx = torch.where(cnvflg_bool, to_kidx, 1.0)
        rho = po_kidx / (RD * safe_to_kidx)
        rho = torch.where(cnvflg_bool, rho, 1.0)
        wucb = -0.01 * dot_kidx / (rho * GRAV)
        wu2 = _put_dynamic2(wu2, kidx, _where(wucb > 0.0, wucb * wucb, 0.0), cnvflg_bool)
    if progomega:
        omegaout = _progomega_calc(
            first_time_step=first_time_step,
            restart=restart,
            kbcon1=kbcon1,
            ktcon=ktcon,
            omegain=fields["omegain"],
            del_=del_,
            zi=zi,
            cnvflg_bool=cnvflg_bool,
            omegaout=omegaout,
            buo=buo,
            drag=drag,
            wush=wush,
            ps=ps,
        )
        levels_progomega = _level_values(1, KM + 1, reference=ps)
        cond_progomega = cnvflg_bool[:, None] & (levels_progomega >= kbcon1[:, None]) & (levels_progomega < ktcon[:, None])
        omega_val = torch.clamp_min(omegaout, -80.0)
        active_density = cond_progomega & (po != 0.0) & (to != 0.0)
        safe_to = torch.where(active_density, to, 1.0)
        rho = po * 100.0 / (RD * safe_to)
        safe_rho = torch.where(active_density & (rho != 0.0), rho, 1.0)
        wu = _where(active_density, (-omega_val) / (safe_rho * GRAV), 0.0)
        omega_u = _where(cond_progomega, omega_val, omega_u)
        wu2 = _where(cond_progomega, torch.clamp_min(wu * wu, 0.0), wu2)
    else:
        wu2, omega_u = run_compile_island(
            "i_recurrence",
            "wu2_recurrence",
            _i_wu2_recurrence_block,
            cnvflg_bool,
            kbcon1,
            ktcon,
            zi,
            drag,
            buo,
            wush,
            po,
            to,
            wu2,
            omega_u,
            bb1=bb1,
            bb2=bb2,
            rd=RD,
            grav=GRAV,
            km=KM,
        )
    fields.update({"ktcon1": ktcon1, "qcko": qcko, "qrcko": qrcko, "pwo": pwo, "cnvwt": cnvwt, "wu2": wu2, "omega_u": omega_u, "omegaout": omegaout, "cnvflg": cnvflg_bool.to(dtype), "aa1": aa1})
    if through == "I":
        return state

    levels_j = _level_values(2, KM, reference=ps)
    cond_j = cnvflg_bool[:, None] & (levels_j > kbcon1[:, None]) & (levels_j < ktcon[:, None])
    dz_j = zi[:, 1 : KM - 1] - zi[:, : KM - 2]
    tem_j = 0.5 * (
        _sqrt_zero_grad(torch.clamp_min(wu2[:, 1 : KM - 1], 0.0))
        + _sqrt_zero_grad(torch.clamp_min(wu2[:, : KM - 2], 0.0))
    )
    # The source accumulates both cloud-depth integrals from bottom to top.
    # A tree reduction changes closure mass flux and detrained condensates.
    wc, sumx = _ordered_pair_sums(cond_j, tem_j * dz_j, dz_j, masked_add=True)
    has_depth = sumx != 0.0
    wc = _where_div(cnvflg_bool & has_depth, wc, sumx, wc)
    cnvflg_bool = cnvflg_bool & has_depth & (wc >= 1.0e-4)
    omegac = torch.zeros_like(ps)
    if progsigma:
        cond_omegac = cnvflg_bool[:, None] & (levels_j >= kbcon1[:, None]) & (levels_j < ktcon[:, None])
        dp_omegac = 1000.0 * del_[:, 1 : KM - 1]
        tem_omegac = 0.5 * (omega_u[:, 1 : KM - 1] + omega_u[:, : KM - 2])
        omegac = _masked_sum(cond_omegac, tem_omegac * dp_omegac)
        sumx = _masked_sum(cond_omegac, dp_omegac)
        has_depth = sumx != 0.0
        omegac = _where_div(cnvflg_bool & has_depth, omegac, sumx, omegac)
        cnvflg_bool = cnvflg_bool & has_depth & (omegac <= -1.2)
        cond_zeta = cnvflg_bool[:, None] & (levels_j >= kbcon1[:, None]) & (levels_j < ktcon[:, None])
        nonzero = omega_u[:, 1 : KM - 1] != 0.0
        raw = _where(
            nonzero,
            eta[:, 1 : KM - 1] * omegac[:, None] / torch.where(nonzero, omega_u[:, 1 : KM - 1], 1.0),
            0.0,
        )
        zeta[:, 1 : KM - 1] = _where(
            cond_zeta,
            torch.clamp(raw, min=0.0, max=1.0),
            zeta[:, 1 : KM - 1].clone(),
        )
    old_ktcon = ktcon.clone()
    ktcon = _where(cnvflg_bool, ktcon1, ktcon)
    ktcon1 = _where(cnvflg_bool, old_ktcon, ktcon1)
    if ncloud > 0:
        k_top_minus = ktcon - 1.0
        qeso_top, to_top, dbyo_top, qcko_top = _gather2_many(k_top_minus, qeso, to, dbyo, qcko)
        safe_to_top = torch.where(cnvflg_bool, to_top, 1.0)
        gamma = el2orc * qeso_top / (safe_to_top * safe_to_top)
        qrch = qeso_top + gamma * dbyo_top / (HVAP * (1.0 + gamma))
        dq = qcko_top - qrch
        moist = cnvflg_bool & (dq > 0.0)
        qlko_ktcon = _where(moist, dq, qlko_ktcon)
        qcko, zdqca = _put_dynamic2_many((qcko, zdqca), k_top_minus, (qrch, dq), moist)
    fields.update({"wc": wc, "omegac": omegac, "zeta": zeta, "ktcon": ktcon, "ktcon1": ktcon1, "qlko_ktcon": qlko_ktcon, "qcko": qcko, "zdqca": zdqca, "cnvflg": cnvflg_bool.to(dtype)})
    if through == "J":
        return state

    dellah, dellaq, dellau, dellav, dellal, dellae = _recurrence_chunk(
        "tendencies", _k_tendencies, cnvflg_bool, kb, ktcon, del_, zi, xlamue, xlamud,
        eta, heo, hcko, qrcko, qcko, uo, ucko, vo, vcko, dellal, qlko_ktcon, GRAV,
        () if hwrf else (ercko, ecko), ntr=NTR,
    )
    delubar = torch.zeros_like(ps)
    delvbar = torch.zeros_like(ps)
    fields.update({"dellah": dellah, "dellaq": dellaq, "dellau": dellau, "dellav": dellav, "dellal": dellal, "dellae": dellae, "delubar": delubar, "delvbar": delvbar})
    if through == "K":
        return state

    q_diff, e_diff, flxtvd, dellaq, dellae = _recurrence_chunk(
        "tvd", _l_tvd, cnvflg_bool, kb, ktcon, del_, eta, q1, qo, dellaq, dellae, GRAV,
        () if hwrf else (ctr, ctro))
    dtconv = torch.zeros_like(ps)
    levels_mid = _level_values(2, KM, reference=ps)
    cond_wind = cnvflg_bool[:, None] & (levels_mid >= kbcon1[:, None]) & (levels_mid < ktcon1[:, None])
    dz_mid = zi[:, 1 : KM - 1] - zi[:, : KM - 2]
    speed_mid = _sqrt_zero_grad(u1[:, 1 : KM - 1] * u1[:, 1 : KM - 1] + v1[:, 1 : KM - 1] * v1[:, 1 : KM - 1])
    # Source wind and depth sums advance from the lowest active level upward.
    wind_products = speed_mid * dz_mid
    umean, sumx = _ordered_pair_sums(cond_wind, wind_products, dz_mid, masked_add=False)
    zi_ktcon1, zi_kbcon1 = _gather2_pair(zi, ktcon1, kbcon1)
    cloud_tem = zi_ktcon1 - zi_kbcon1
    wc_for_dtconv = torch.clamp_min(wc, wc_min) if progomega else wc
    dtconv = _where_div(cnvflg_bool, cloud_tem, wc_for_dtconv, dtconv)
    if not hwrf:
        dtconv = _where(cnvflg_bool, (1.0 + gdx / 75000.0) * dtconv, dtconv)
    dtconv = _where(cnvflg_bool, torch.clamp(torch.clamp_min(dtconv, DELT), min=dtmin, max=dtmax), dtconv)
    umean_avg = _where_div(cnvflg_bool, umean, sumx, 0.0)
    umean = _where(cnvflg_bool, torch.clamp_min(umean_avg, 1.0), umean)
    tauadv = gdx / torch.where(cnvflg_bool, umean, 1.0)
    advfac_raw = tauadv / torch.where(cnvflg_bool, dtconv, 1.0)
    advfac_raw = CAT_ADJ_SHAL * advfac_raw
    advfac = _where(cnvflg_bool, torch.clamp_max(advfac_raw, 1.0), advfac)
    fields.update({"q_diff": q_diff, "e_diff": e_diff, "flxtvd": flxtvd, "dtconv": dtconv, "umean": umean, "advfac": advfac, "dellaq": dellaq, "dellae": dellae})
    if through == "L":
        return state

    qadv = torch.zeros_like(t1)
    if progsigma and not (first_time_step and not restart):
        qadv = (q - prevsq) * invdelt
    tmfq = tmf[:, :, 0].clone()
    sigmaout = fields["sigmaout"].clone()
    sigmab = torch.zeros_like(ps)
    if progsigma:
        sigmab, sigmaout = _progsigma_calc(
            first_time_step=first_time_step,
            restart=restart,
            del_=del_,
            tmfq=tmfq,
            qmicro=qmicro,
            dbyo1=dbyo1,
            zdqca=zdqca,
            omega_u=omega_u,
            zeta=zeta,
            qadv=qadv,
            kb=kb,
            kbcon1=kbcon1,
            ktcon=ktcon,
            cnvflg_bool=cnvflg_bool,
            sigmain=fields["sigmain"],
            sigmaout=sigmaout,
            ps=ps,
    )
    po_kbcon, to_kbcon, xlamue_kbcon = _gather2_many(kbcon, po, to, xlamue)
    safe_to_kbcon = torch.where(cnvflg_bool, to_kbcon, 1.0)
    rho = po_kbcon * 100.0 / (RD * safe_to_kbcon)
    xmb = _where(cnvflg_bool, advfac * betaw * rho * wc, xmb)
    if progsigma:
        xmb = _where(cnvflg_bool & (gdx < dxcrtas), advfac * sigmab * ((-1.0 * omegac) * gravinv), xmb)
    tem = torch.clamp(xlamue_kbcon, min=2.0e-4, max=6.0e-4)
    tem = 0.2 / tem
    sigmagfm = torch.clamp((3.14 * tem * tem) / garea, min=0.001, max=0.999)
    scale_base = (1.0 - sigmab) * (1.0 - sigmab) if progsigma else (1.0 - sigmagfm) * (1.0 - sigmagfm)
    scaldfunc = _where(gdx < dxcrt, torch.clamp(scale_base, min=0.0, max=1.0), 1.0)
    sigmagfm = _where(cnvflg_bool, sigmagfm, 0.0)
    scaldfunc = _where(cnvflg_bool, scaldfunc, 0.0)
    xmb = _where(cnvflg_bool, torch.minimum(xmb * scaldfunc, xmbmax), xmb)
    delhbar = torch.zeros_like(ps)
    delqbar = torch.zeros_like(ps)
    deltbar = torch.zeros_like(ps)
    delubar = torch.zeros_like(ps)
    delvbar = torch.zeros_like(ps)
    qcond = torch.zeros_like(ps)
    delebar = torch.zeros((ncol, NTR), dtype=dtype, device=device)
    levels_full = _level_values(1, KM + 1, reference=ps)
    cond_qeso = cnvflg_bool[:, None] & (levels_full <= kmax[:, None])
    es = 0.01 * fpvs(t1)
    qes = EPS * es / (pfld + EPSM1 * es)
    qeso = _where(cond_qeso, torch.clamp_min(qes, 1.0e-8), qeso)
    cond_update = cnvflg_bool[:, None] & (levels_full > kb[:, None]) & (levels_full <= ktcon[:, None])
    dellat = (dellah - HVAP * dellaq) / CP
    # Preserve the source's left-associated feedback products.
    new_t1 = _where(cond_update, t1 + dellat * xmb[:, None] * DELT, new_t1)
    new_q1 = _where(cond_update, q1 + dellaq * xmb[:, None] * DELT, new_q1)
    new_u1 = _where(cond_update, u1 + dellau * xmb[:, None] * DELT, new_u1)
    new_v1 = _where(cond_update, v1 + dellav * xmb[:, None] * DELT, new_v1)
    tem_mass = xmb[:, None] * (1000.0 * del_) / GRAV
    delhbar = _masked_sum(cond_update, tem_mass * dellah)
    delqbar = _masked_sum(cond_update, tem_mass * dellaq)
    deltbar = _masked_sum(cond_update, tem_mass * dellat)
    delubar = _masked_sum(cond_update, tem_mass * dellau)
    delvbar = _masked_sum(cond_update, tem_mass * dellav)
    fields.update({"qadv": qadv, "tmfq": tmfq, "sigmaout": sigmaout, "sigmab": sigmab, "xmb": xmb, "sigmagfm": sigmagfm, "scaldfunc": scaldfunc, "new_t1": new_t1, "new_q1": new_q1, "new_u1": new_u1, "new_v1": new_v1, "qeso": qeso, "delhbar": delhbar, "delqbar": delqbar, "deltbar": deltbar, "delubar": delubar, "delvbar": delvbar})
    if through == "M":
        return state

    tsumn = torch.zeros_like(ps)
    tsump = torch.zeros_like(ps)
    rtnp = torch.ones_like(ps)
    levels_n = _level_values(1, KM, reference=ps)
    cond_n = cnvflg_bool[:, None] & (levels_n > kb[:, None]) & (levels_n <= ktcon[:, None])
    qk = new_q1[:, : KM - 1].clone()
    q_mass = qk * delp[:, : KM - 1] / GRAV
    tsumn = _masked_sum(cond_n & (new_q1[:, : KM - 1] < 0.0), q_mass)
    tsump = _masked_sum(cond_n & (new_q1[:, : KM - 1] > 0.0), q_mass)
    borrow = cnvflg_bool & (tsump > 0.0) & (tsumn < 0.0)
    borrow_pos = borrow & (tsump > torch.abs(tsumn))
    rtnp_pos = tsumn / torch.where(borrow_pos, tsump, 1.0)
    rtnp_neg = tsump / torch.where(borrow & (~borrow_pos), tsumn, 1.0)
    rtnp = _where(borrow_pos, rtnp_pos, _where(borrow, rtnp_neg, rtnp))
    cond_borrow = cond_n & (rtnp < 0.0)[:, None]
    more_pos = tsump > torch.abs(tsumn)
    updated_q = _where(cond_borrow & more_pos[:, None] & (qk < 0.0), 0.0, qk)
    updated_q = _where(cond_borrow & more_pos[:, None] & (qk > 0.0), (1.0 + rtnp)[:, None] * qk, updated_q)
    updated_q = _where(cond_borrow & (~more_pos)[:, None] & (qk < 0.0), (1.0 + rtnp)[:, None] * qk, updated_q)
    updated_q = _where(cond_borrow & (~more_pos)[:, None] & (qk > 0.0), 0.0, updated_q)
    new_q1[:, : KM - 1] = updated_q
    if not hwrf:
        indx = ntk - 2
        cond_tr = cond_update
        ctr = ctr + _masked_values(cond_tr[:, :, None], dellae * xmb[:, None, None] * DELT)
        dp_mass = xmb[:, None] * (1000.0 * del_) / GRAV
        delebar = _masked_sum(cond_tr[:, :, None], dellae * dp_mass[:, :, None])

        prev_zi = torch.zeros_like(zi[:, : KM - 1])
        prev_zi[:, 1 : KM - 1] = zi[:, : KM - 2]
        dz_borrow = zi[:, : KM - 1] - prev_zi
        tracer_ids = _level_values(1, NTR + 1, reference=ps).reshape(1, 1, NTR)
        ctr_mid = ctr[:, : KM - 1, :].clone()
        tracer_mass = torch.where(
            tracer_ids == float(indx),
            ctr_mid * dz_borrow[:, :, None],
            ctr_mid * delp[:, : KM - 1, None] / GRAV,
        )
        cond_tr_mid = cond_tr[:, : KM - 1, None]
        tsumn_tr = _masked_sum(cond_tr_mid & (ctr_mid < 0.0), tracer_mass)
        tsump_tr = _masked_sum(cond_tr_mid & (ctr_mid > 0.0), tracer_mass)
        borrow_tr = cnvflg_bool[:, None] & (tsump_tr > 0.0) & (tsumn_tr < 0.0)
        borrow_pos_tr = borrow_tr & (tsump_tr > torch.abs(tsumn_tr))
        rtnp_pos_tr = tsumn_tr / torch.where(borrow_pos_tr, tsump_tr, 1.0)
        rtnp_neg_tr = tsump_tr / torch.where(borrow_tr & (~borrow_pos_tr), tsumn_tr, 1.0)
        rtnp_tr = _where(borrow_pos_tr, rtnp_pos_tr, _where(borrow_tr, rtnp_neg_tr, 1.0))
        cond_tr_borrow = cond_tr_mid & (rtnp_tr[:, None, :] < 0.0)
        updated_ctr = _where(cond_tr_borrow & borrow_pos_tr[:, None, :] & (ctr_mid < 0.0), 0.0, ctr_mid)
        updated_ctr = _where(cond_tr_borrow & borrow_pos_tr[:, None, :] & (ctr_mid > 0.0), (1.0 + rtnp_tr)[:, None, :] * ctr_mid, updated_ctr)
        updated_ctr = _where(cond_tr_borrow & (~borrow_pos_tr)[:, None, :] & (ctr_mid < 0.0), (1.0 + rtnp_tr)[:, None, :] * ctr_mid, updated_ctr)
        updated_ctr = _where(cond_tr_borrow & (~borrow_pos_tr)[:, None, :] & (ctr_mid > 0.0), 0.0, updated_ctr)
        ctr[:, : KM - 1, :] = updated_ctr
        new_qtr[:, :, 2 : 2 + NTR] = _where(cond_tr[:, :, None], ctr, new_qtr[:, :, 2 : 2 + NTR].clone())
        tsumn = tsumn_tr[:, NTR - 1]
        tsump = tsump_tr[:, NTR - 1]
        rtnp = rtnp_tr[:, NTR - 1]
        if do_aerosols_bool:
            levels_aer = _level_values(2, KM, reference=ps)
            cond_aer = cnvflg_bool[:, None] & (levels_aer > kb[:, None]) & (levels_aer < ktcon[:, None])
            wet_mid = wet_dep[:, 1 : KM - 1, :].clone()
            wet_mid = _where(
                cond_aer[:, :, None],
                chem_pw[:, 1 : KM - 1, :] * GRAV * xmb[:, None, None] * DELT,
                wet_mid,
            )
            qtr_slice = new_qtr[:, 1 : KM - 1, itc - 1 : itc - 1 + NTC].clone()
            dp_aer = 1000.0 * del_[:, 1 : KM - 1]
            tem = -qtr_slice * dp_aer[:, :, None]
            neg_aer = cond_aer[:, :, None] & (qtr_slice < 0.0)
            enough = neg_aer & (wet_mid >= tem)
            wet_after_enough = _where(enough, wet_mid - tem, wet_mid)
            qtr_after_enough = _where(enough, 0.0, qtr_slice)
            not_enough = neg_aer & (~enough)
            wet_final = _where(not_enough, 0.0, wet_after_enough)
            qtr_final = _where(not_enough, qtr_after_enough + wet_final / dp_aer[:, :, None], qtr_after_enough)
            wet_dep[:, 1 : KM - 1, :] = wet_final
            new_qtr[:, 1 : KM - 1, itc - 1 : itc - 1 + NTC] = qtr_final
    fields.update({"new_q1": new_q1, "new_qtr": new_qtr, "ctr": ctr, "wet_dep": wet_dep, "tsumn": tsumn, "tsump": tsump, "rtnp": rtnp, "delebar": delebar})
    if through == "N":
        return state

    levels_full = _level_values(1, KM + 1, reference=ps)
    cond_o = cnvflg_bool[:, None] & (levels_full > kb[:, None]) & (levels_full <= ktcon[:, None])
    es = 0.01 * fpvs(new_t1)
    qes = EPS * es / (pfld + EPSM1 * es)
    qeso = _where(cond_o, torch.clamp_min(qes, 1.0e-8), qeso)
    cond_rntot = cnvflg_bool[:, None] & (levels_full < ktcon[:, None]) & (levels_full > kb[:, None])
    rntot = _masked_sum(cond_rntot, pwo * xmb[:, None] * 0.001 * DELT)
    deltbar = fields["deltbar"]
    delqbar = fields["delqbar"] if "delqbar" in fields else delqbar
    rn, qevap, delqev, delqbar, deltbar, new_t1, new_q1, flg = run_compile_island(
        "o_recurrence",
        "evap_recurrence",
        _o_evap_recurrence_block,
        cnvflg_bool,
        kb,
        ktcon,
        kmax,
        pwo,
        xmb,
        del_,
        qeso,
        new_t1,
        new_q1,
        rn,
        rntot,
        delqbar,
        deltbar,
        shevf=shevf,
        evef=EVEF,
        delt=DELT,
        el2orc=el2orc,
        elocp=elocp,
        grav=GRAV,
        km=KM,
    )
    fields.update({"qeso": qeso, "rntot": rntot, "rn": rn, "qevap": qevap, "delqev": delqev, "delqbar": delqbar, "deltbar": deltbar, "new_t1": new_t1, "new_q1": new_q1})
    if through == "O":
        return state

    rn = _where(cnvflg_bool & ((rn < 0.0) | (~flg)), 0.0, rn)
    ktop = _where(cnvflg_bool, ktcon, ktop)
    kbot = _where(cnvflg_bool, kbcon, kbot)
    kcnv = _where(cnvflg_bool, 2.0, kcnv)
    levels_full = _level_values(1, KM + 1, reference=ps)
    cond_cloud = cnvflg_bool[:, None] & (levels_full >= kbcon[:, None]) & (levels_full < ktcon[:, None])
    cnvw = _where(cond_cloud, cnvwt * xmb[:, None] * DELT * CSCALE, cnvw)
    cnvc_value = torch.clamp(0.04 * torch.log(1.0 + 675.0 * eta * xmb[:, None]), min=0.0, max=0.2)
    cnvc = _where(cond_cloud, cnvc_value, cnvc)
    if ncloud > 0:
        cond_detrain = cnvflg_bool[:, None] & (levels_full[:, : KM - 1] >= kbcon[:, None]) & (levels_full[:, : KM - 1] <= ktcon[:, None])
        tem = dellal[:, : KM - 1] * xmb[:, None] * DELT
        tem1 = torch.clamp((263.16 - new_t1[:, : KM - 1]) * (1.0 / (263.16 - 233.16)), min=0.0, max=1.0)
        liquid_present = new_qtr[:, : KM - 1, 1] > -999.0
        ice_after = _where(cond_detrain & liquid_present, new_qtr[:, : KM - 1, 0] + tem * tem1, new_qtr[:, : KM - 1, 0].clone())
        liq_after = _where(cond_detrain & liquid_present, new_qtr[:, : KM - 1, 1] + tem * (1.0 - tem1), new_qtr[:, : KM - 1, 1].clone())
        ice_after = _where(cond_detrain & (~liquid_present), ice_after + tem, ice_after)
        new_qtr[:, : KM - 1, 0] = ice_after
        new_qtr[:, : KM - 1, 1] = liq_after
    cond_ud = cnvflg_bool[:, None] & (levels_full >= kb[:, None]) & (levels_full < ktop[:, None])
    ud_mf = _where(cond_ud, eta * xmb[:, None] * DELT, ud_mf)
    dt_mf = _put_dynamic2(dt_mf, ktop - 1.0, _gather2(ud_mf, ktop - 1.0), cnvflg_bool)
    if (not hwrf) and ntk > 0:
        levels_mid = _level_values(2, KM, reference=ps)
        cond_tke = cnvflg_bool[:, None] & (levels_mid > kb[:, None]) & (levels_mid < ktop[:, None])
        tem = 0.5 * (eta[:, : KM - 2] + eta[:, 1 : KM - 1]) * xmb[:, None]
        tem1 = pfld[:, 1 : KM - 1] * 100.0 / (RD * new_t1[:, 1 : KM - 1])
        tem2 = sigmab[:, None] if progsigma else torch.clamp_min(sigmagfm, betaw)[:, None]
        ptem = _where(cond_tke, tem / (tem2 * torch.where(cond_tke, tem1, 1.0)), 0.0)
        new_qtr[:, 1 : KM - 1, ntk - 1] = _where(
            cond_tke,
            new_qtr[:, 1 : KM - 1, ntk - 1].clone() + 0.5 * tem2 * ptem * ptem,
            new_qtr[:, 1 : KM - 1, ntk - 1].clone(),
        )
    ten_t = (new_t1 - t1) / DELT
    ten_u = (new_u1 - u1) / DELT
    ten_v = (new_v1 - v1) / DELT
    ten_q[:, :, 0] = (new_q1 - q1) / DELT
    dqtr = (new_qtr - qtr) / DELT
    fields.update({"rn": rn, "kbot": kbot, "ktop": ktop, "kcnv": kcnv, "cnvw": cnvw, "cnvc": cnvc, "ud_mf": ud_mf, "dt_mf": dt_mf, "new_qtr": new_qtr, "ten_t": ten_t, "ten_u": ten_u, "ten_v": ten_v, "ten_q": ten_q, "dqtr": dqtr})
    if through == "P":
        return state

    _run_post(fields, state.config)
    return state


# GFS_SCNV_generic_post: tracers left out of the tracer dtend under cscnv/satmedmf/trans_trac/ras.
DTEND_EXCLUDED = ("ntcw", "ntiw", "ntclamt", "ntrw", "ntsw", "ntrnc", "ntsnc", "ntgl", "ntgnc", "ntsigma")
# Legacy harness table of the frozen shallow corpus (TKE last, ntclamt=4, ntsigma=7).
LEGACY_INDICES = {"ntqv": 1, "ntcw": 2, "ntiw": 3, "ntclamt": 4, "ntrw": 5, "ntsw": 6, "ntsigma": 7}


def tracer_layout_from(fields: dict[str, torch.Tensor], config: dict[str, Any]) -> ConvectionTracerLayout:
    layout = layout_from_config(config.get("tracer_layout"))
    if layout is not None:
        return layout
    nn = fields["qtr"].shape[2]
    return legacy_layout(
        nn - 2,
        indices=LEGACY_INDICES,
        ntk=int(config.get("ntk", NTK)),
        itc=int(config.get("itc", ITC_DEFAULT)),
        ntc=fields["fscav"].numel(),
    )


def tracer_layout(state: SaSasShallowConvectionState) -> ConvectionTracerLayout:
    """The resolved convective tracer layout (legacy harness table when undeclared)."""

    return tracer_layout_from(state.fields, state.config)


def _run_post(fields: dict[str, torch.Tensor], config: dict[str, Any]) -> None:
    """GFS_SCNV_generic_post (oracle source): layout-driven slots, CCPP cloud rule.

    With ``config['samf_water_reconciliation']`` (oracle runtime
    ``MPAS_UFS_SAMF_WATER_RECONCILIATION=1``) the MPAS water reconciliation runs after
    the tendency application. The commit-residual validation and the ledgers write no
    state and are not ported.
    """
    layout = tracer_layout_from(fields, config)
    tend_opt = int(config.get("tend_opt_scnv", 1))
    reconcile = bool(config.get("samf_water_reconciliation", False))
    if reconcile and tend_opt not in (1, 3):
        raise ValueError(
            "SAMF water reconciliation requires tend_opt_scnv 1 or 3 (the oracle returns errflg otherwise)"
        )
    # Even pass-through outputs must survive a refresh of caller-owned inputs.
    # Updated fields acquire their own storage in the expressions below.
    if tend_opt not in (2, 3):
        for name in ("dtdt", "dudt", "dvdt", "dqdt"):
            fields[name] = fields[name].clone()
    if tend_opt not in (1, 3):
        for name in ("gt0", "gu0", "gv0", "gq0"):
            fields[name] = fields[name].clone()
    fields["cnvprcp"] = fields["cnvprcp"].clone()
    delt = fields["delt"]
    frain = fields.get("frain", torch.ones_like(delt))
    imp_physics = int(config.get("imp_physics", 8))
    imp_physics_gfdl = int(config.get("imp_physics_gfdl", 9))
    ten_q = tendencies_from_dclw(fields["ten_q"], fields["dqtr"], layout, is_gfdl=imp_physics == imp_physics_gfdl)
    fields["ten_q"] = ten_q

    save_q = fields["gq0"]
    if tend_opt == 1:
        fields["gt0"] = fields["gt0"] + delt * fields["ten_t"]
        fields["gu0"] = fields["gu0"] + delt * fields["ten_u"]
        fields["gv0"] = fields["gv0"] + delt * fields["ten_v"]
        fields["gq0"] = fields["gq0"] + delt * ten_q
    elif tend_opt == 2:
        fields["dtdt"] = fields["dtdt"] + fields["ten_t"]
        fields["dudt"] = fields["dudt"] + fields["ten_u"]
        fields["dvdt"] = fields["dvdt"] + fields["ten_v"]
        fields["dqdt"] = fields["dqdt"] + ten_q
    elif tend_opt == 3:
        fields["gt0"] = fields["gt0"] + delt * (fields["dtdt"] + fields["ten_t"])
        fields["gu0"] = fields["gu0"] + delt * (fields["dudt"] + fields["ten_u"])
        fields["gv0"] = fields["gv0"] + delt * (fields["dvdt"] + fields["ten_v"])
        fields["gq0"] = fields["gq0"] + delt * (fields["dqdt"] + ten_q)
        fields["dtdt"] = torch.zeros_like(fields["dtdt"])
        fields["dudt"] = torch.zeros_like(fields["dudt"])
        fields["dvdt"] = torch.zeros_like(fields["dvdt"])
        fields["dqdt"] = torch.zeros_like(fields["dqdt"])
    elif tend_opt != 4:
        fields["errflg"] = torch.ones_like(fields["errflg"])

    if reconcile:
        gq0, gt0, reconcile_errflg = samf_water_reconciliation(
            fields["gq0"], fields["gt0"], save_q, fields["rn"], fields["garea"], fields["prsi"], frain,
            fields["grav"], fields["cp"], fields["hvap"], layout, delt=delt,
        )
        fields["gq0"] = gq0
        fields["gt0"] = gt0
        fields["errflg"] = torch.maximum(fields["errflg"], reconcile_errflg)

    # imfshalcnv is SAMF in every sealed configuration.
    fields["rainc"] = fields["rainc"] + frain * fields["rn"]
    npdf3d = int(config.get("npdf3d", 3))
    num_p3d = int(config.get("num_p3d", 4))
    ncnvcld3d = int(config.get("ncnvcld3d", 1))
    if bool(config.get("shcnvcw", True)) and num_p3d == 4 and npdf3d == 3:
        fields["cnvw_phy_f3d"] = fields["cnvw_phy_f3d"] + fields["cnvw"]
        fields["cnvc_phy_f3d"] = fields["cnvc_phy_f3d"] + fields["cnvc"]
    elif npdf3d == 0 and ncnvcld3d == 1:
        fields["cnvw_phy_f3d"] = fields["cnvw_phy_f3d"] + fields["cnvw"]
        fields["cnvc_phy_f3d"] = fields["cnvc_phy_f3d"].clone()
    else:
        fields["cnvw_phy_f3d"] = fields["cnvw_phy_f3d"].clone()
        fields["cnvc_phy_f3d"] = fields["cnvc_phy_f3d"].clone()
    dtend = fields["dtend"].clone()
    if bool(config.get("lssav", True)) and bool(config.get("flag_for_scnv_generic_tend", True)) and bool(config.get("ldiag3d", True)):
        # dtidx (harness rule): T, u, v -> slots 1-3, qv -> 4, tracer n >= 2 -> slot 4+n.
        dtend[:, :, 0] = dtend[:, :, 0] + (fields["ten_t"] * delt) * frain
        dtend[:, :, 1] = dtend[:, :, 1] + (fields["ten_u"] * delt) * frain
        dtend[:, :, 2] = dtend[:, :, 2] + (fields["ten_v"] * delt) * frain
        transport = any(bool(config.get(name, False)) for name in ("cscnv", "satmedmf", "trans_trac", "ras"))
        excluded = {layout.index(name) for name in DTEND_EXCLUDED} - {0}
        for n in range(2, layout.ntrac + 1):
            if transport and n in excluded:
                continue
            dtend[:, :, n + 3] = dtend[:, :, n + 3] + (ten_q[:, :, n - 1] * delt) * frain
        qv = layout.index("ntqv")
        dtend[:, :, 3] = dtend[:, :, 3] + (ten_q[:, :, qv - 1] * delt) * frain
    fields["dtend"] = dtend


def initial_state_oracle_mpas_case(
    case_id: str,
    *,
    reconciliation: bool,
    tracer_values: dict[str, float] | None = None,
    device: torch.device | str | None = None,
    ncol: int | None = None,
) -> SaSasShallowConvectionState:
    """SCNV-entry state of the G3.2 oracle lane for an approved scenario case.

    Mirrors ``tools/run_sa_sas_convection_oracle_g3_2_mint.py`` (shallow, ``mpas_tempo``):
    the legacy profile and scenario adjustments, then the MPAS tracer state (``gq0``
    with qv = q1, state = samf input), ``clw`` from the ``GFS_suite_interstitial_3``
    rule, TEMPO, satmedmf/trans_trac, ``npdf3d=0, num_p3d=3, ncnvcld3d=1``,
    ``shcnvcw=.false.`` and ``ncloud=ncnd=5``.
    """

    legacy = initial_state_for_case(case_id, device=device, ncol=ncol)
    layout = mpas_tempo_layout()
    fields = {name: tensor.clone() for name, tensor in legacy.fields.items()}
    ncol_runtime, km = fields["q1"].shape
    dtype, dev = fields["q1"].dtype, fields["q1"].device
    values = dict(tracer_values or ORACLE_MPAS_TRACER_VALUES)
    if case_id == "case11_tke_entrainment_active":
        values["ntke"] = 0.2
    gq0 = torch.zeros((ncol_runtime, km, layout.ntrac), dtype=dtype, device=dev)
    gq0[:, :, layout.index("ntqv") - 1] = fields["q1"]
    for name, value in values.items():
        gq0[:, :, layout.index(name) - 1] = value
    clw = build_clw(
        gq0,
        torch.zeros((ncol_runtime, km, layout.nn), dtype=dtype, device=dev),
        layout,
        imp_physics=IMP_PHYSICS_CODES["tempo"],
        transport=True,
    )
    fields.update(
        {
            "qtr": clw,
            "dqtr": torch.zeros_like(clw),
            "tmf": torch.zeros_like(clw),
            "ten_q": torch.zeros_like(gq0),
            "dqdt": torch.zeros_like(gq0),
            "gq0": gq0,
            "gt0": fields["t1"].clone(),
            "gu0": fields["u1"].clone(),
            "gv0": fields["v1"].clone(),
            "fscav": torch.full((layout.ntc,), 0.5 if case_id == "case13_aerosol_wetdep_active" else 0.1, dtype=dtype, device=dev),
            "prsi": interface_pressure(fields["psp"], fields["delp"]),
        }
    )
    config = {
        **legacy.config,
        "tracer_layout": layout.as_config(),
        "nn": layout.nn,
        "ntr": layout.ntr,
        "ntk": layout.ntk,
        "itc": layout.itc,
        "ntc": layout.ntc,
        "ntrac": layout.ntrac,
        "nsamftrac": layout.ntr,
        "ncloud": 2 if case_id == "case14_ncloud_cloudwater_detrainment" else 5,
        "imp_physics": IMP_PHYSICS_CODES["tempo"],
        "imp_physics_gfdl": IMP_PHYSICS_CODES["gfdl"],
        "npdf3d": 0,
        "num_p3d": 3,
        "ncnvcld3d": 1,
        "shcnvcw": False,
        "satmedmf": True,
        "trans_trac": True,
        "samf_water_reconciliation": bool(reconciliation),
    }
    return SaSasShallowConvectionState(fields, config)


# G3.2 oracle lane (MPAS_GFS_v17_tempo_rrtmgp): MPAS tracer values of the approved cases.
ORACLE_MPAS_TRACER_VALUES = {
    "ntcw": 1.0e-5, "ntrw": 1.0e-6, "ntiw": 1.0e-6, "ntsw": 1.0e-6, "ntgl": 1.0e-6,
    "ntinc": 1.0e-6, "ntrnc": 1.0e-6, "ntke": 0.1, "ntoz": 1.0e-6,
}


def checkpoint_fields(state: SaSasShallowConvectionState, boundary_id: str) -> dict[str, torch.Tensor]:
    if boundary_id not in BOUNDARY_FIELDS:
        raise NotImplementedError(f"SA-SAS shallow-convection checkpoint {boundary_id!r} is not implemented yet")
    return {name: state.require(name) for name in BOUNDARY_FIELDS[boundary_id]}
