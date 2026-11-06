# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Segment V: GFS deep-convection generic post application."""

from __future__ import annotations

import torch

from sa_sas_deep_convection_torch.compile import run_compile_island
from sa_sas_deep_convection_torch.constants import KM, NDTEND, NN, NTR
from sa_sas_deep_convection_torch.samf_interstitials import (
    IMP_PHYSICS_CODES,
    ConvectionTracerLayout,
    build_clw,
    samf_water_reconciliation,
    tendencies_from_dclw,
)
from sa_sas_deep_convection_torch.state import runtime_dimensions, SaSasDeepConvectionState, require_real, tracer_layout

CHECKPOINT_FIELDS = (
    "gt0",
    "gu0",
    "gv0",
    "gq0",
    "dtdt",
    "dudt",
    "dvdt",
    "dqdt",
    "dtend",
    "upd_mf",
    "dwn_mf",
    "det_mf",
    "cnvw_phy_f3d",
    "cnvc_phy_f3d",
    "rainc",
    "cldwrk",
    "ten_q",
    "cnvw",
    "cnvc",
    "clw",
    "errflg",
)

# CCPP GFS_typedefs.F90 microphysics codes (shared with the shallow port).
IMP_PHYSICS_GFDL = IMP_PHYSICS_CODES["gfdl"]
IMP_PHYSICS_THOMPSON = IMP_PHYSICS_CODES["thompson"]
# GFS_DCNV_generic_post refreshes clw ice/water for these codes only; any other
# code falls through and leaves clw ice/water untouched. TEMPO is included per the
# GFS v17 TEMPO+RRTMGP reference patch, which adds it to the Thompson branch.
CLW_ICE_WATER_CODES = frozenset(
    IMP_PHYSICS_CODES[name] for name in ("thompson", "tempo", "nssl", "wsm6", "mg", "fer_hires")
)


def imp_physics_code(value: object) -> int | None:
    """Return the CCPP code for an integer, numeric string or scheme name; None if unnamed."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text.lstrip("+-").isdigit():
        return int(text)
    return IMP_PHYSICS_CODES.get(text.removeprefix("imp_physics_"))


def _field_or_zeros(state: SaSasDeepConvectionState, name: str, like: torch.Tensor) -> torch.Tensor:
    field = state.fields.get(name)
    if field is None:
        return torch.zeros_like(like)
    return field.clone()


def cloud_copy_rule(config: dict) -> str:
    """CCPP npdf3d/num_p3d/ncnvcld3d rule of GFS_DCNV_generic_post (cnvw/cnvc copy).

    The rule is host configuration and never inferred from ``imp_physics``. Missing
    fields take the legacy harness values (npdf3d=3, num_p3d=4, ncnvcld3d=1).
    """

    npdf3d = int(config.get("npdf3d", 3))
    num_p3d = int(config.get("num_p3d", 4))
    ncnvcld3d = int(config.get("ncnvcld3d", 1))
    if npdf3d == 3 and num_p3d == 4:
        return "cnvw_cnvc"
    if npdf3d == 0 and ncnvcld3d == 1:
        return "cnvw"
    return "none"


def run(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """GFS_DCNV_generic_post (oracle source): layout-driven tracer slots, CCPP cloud rule.

    With ``config['samf_water_reconciliation']`` (the oracle runtime's
    ``MPAS_UFS_SAMF_WATER_RECONCILIATION=1``) the MPAS water reconciliation runs after
    the tendency application, as in the oracle. The oracle's commit-residual check is a
    validation of the application just performed with the same expressions and is not
    ported; its ledgers write no state.
    """
    _, KM, NN, NTR, NTC = runtime_dimensions(state)
    layout = tracer_layout(state)
    NTRAC = layout.ntrac
    NDTEND_runtime = int(state.config.get("ndtend", max(NDTEND, NTRAC + 3)))

    t1 = state.require("t1")
    u1 = state.require("u1")
    v1 = state.require("v1")
    qtr = state.require("qtr")
    dqtr = state.require("dqtr")
    ten_t = state.require("ten_t")
    ten_u = state.require("ten_u")
    ten_v = state.require("ten_v")
    ten_q = state.require("ten_q").clone()
    rn = state.require("rn")
    cldwrk = state.require("cldwrk")
    ud_mf = state.require("ud_mf")
    dd_mf = state.require("dd_mf")
    dt_mf = state.require("dt_mf")
    cnvw = state.require("cnvw")
    cnvc = state.require("cnvc")

    ncol = t1.shape[0]
    for name, tensor in {
        "t1": t1,
        "u1": u1,
        "v1": v1,
        "ten_t": ten_t,
        "ten_u": ten_u,
        "ten_v": ten_v,
        "ud_mf": ud_mf,
        "dd_mf": dd_mf,
        "dt_mf": dt_mf,
        "cnvw": cnvw,
        "cnvc": cnvc,
    }.items():
        require_real(name, tensor, (ncol, KM))
    for name, (tensor, shape) in {
        "qtr": (qtr, (ncol, KM, NN)),
        "dqtr": (dqtr, (ncol, KM, NN)),
        "ten_q": (ten_q, (ncol, KM, NTRAC)),
    }.items():
        require_real(name, tensor, shape)
    require_real("rn", rn, (ncol,))
    require_real("cldwrk", cldwrk, (ncol,))

    imp_physics = imp_physics_code(state.config.get("imp_physics", "gfdl"))
    is_gfdl = imp_physics == IMP_PHYSICS_GFDL
    tend_opt = int(state.config.get("tend_opt_dcnv", 1))
    trans_trac = bool(state.config.get("trans_trac", True))
    ldiag3d = bool(state.config.get("ldiag3d", True))
    qdiag3d = bool(state.config.get("qdiag3d", True))
    flag_for_dcnv_generic_tend = bool(state.config.get("flag_for_dcnv_generic_tend", True))
    ras = bool(state.config.get("ras", False))
    cscnv = bool(state.config.get("cscnv", False))
    satmedmf = bool(state.config.get("satmedmf", False))
    reconcile = bool(state.config.get("samf_water_reconciliation", False))
    if reconcile and tend_opt not in (1, 3):
        raise ValueError(
            "SAMF water reconciliation requires tend_opt_dcnv 1 or 3 (the oracle returns errflg otherwise)"
        )

    gt0 = state.fields.get("gt0", t1).clone()
    gu0 = state.fields.get("gu0", u1).clone()
    gv0 = state.fields.get("gv0", v1).clone()
    gq0_seed = state.fields.get("gq0")
    gq0 = (qtr[:, :, :NTRAC] if gq0_seed is None else gq0_seed).clone()
    require_real("gq0", gq0, (ncol, KM, NTRAC))
    dtdt = _field_or_zeros(state, "dtdt", gt0)
    dudt = _field_or_zeros(state, "dudt", gu0)
    dvdt = _field_or_zeros(state, "dvdt", gv0)
    dqdt_seed = state.fields.get("dqdt")
    dqdt = torch.zeros_like(gq0) if dqdt_seed is None else dqdt_seed.clone()

    dtend_seed = state.fields.get("dtend")
    dtend = torch.zeros((ncol, KM, NDTEND_runtime), dtype=t1.dtype, device=t1.device) if dtend_seed is None else dtend_seed.clone()

    clw_seed = state.fields.get("clw")
    clw = (qtr.clone() if clw_seed is None else clw_seed.clone())

    upd_mf = _field_or_zeros(state, "upd_mf", ud_mf)
    dwn_mf = _field_or_zeros(state, "dwn_mf", dd_mf)
    det_mf = _field_or_zeros(state, "det_mf", dt_mf)
    cnvw_phy_f3d = _field_or_zeros(state, "cnvw_phy_f3d", cnvw)
    cnvc_phy_f3d = _field_or_zeros(state, "cnvc_phy_f3d", cnvc)

    transport = cscnv or satmedmf or trans_trac or ras
    excluded = {layout.index(name) for name in DTEND_EXCLUDED} - {0}
    if transport:
        dtend_tracers = tuple(n for n in range(2, NTRAC + 1) if n not in excluded)
    else:
        dtend_tracers = tuple(range(2, NTRAC + 1))
    one = torch.ones((), dtype=t1.dtype, device=t1.device)
    frain = state.fields.get("frain", one)
    if reconcile:
        prsi = state.require("prsi")
        area = state.require("garea")
        require_real("prsi", prsi, (ncol, KM + 1))
    else:
        prsi = torch.zeros((ncol, KM + 1), dtype=t1.dtype, device=t1.device)
        area = torch.zeros((ncol,), dtype=t1.dtype, device=t1.device)

    (
        ten_q,
        gt0,
        gu0,
        gv0,
        gq0,
        dtdt,
        dudt,
        dvdt,
        dqdt,
        dtend,
        upd_mf,
        dwn_mf,
        det_mf,
        cnvw_phy_f3d,
        cnvc_phy_f3d,
        cnvw_out,
        cnvc_out,
        clw,
        rainc,
        cldwrk_out,
        errflg,
    ) = run_compile_island(
        "v_post",
        "generic_post",
        _generic_post_block,
        dqtr,
        ten_t,
        ten_u,
        ten_v,
        ten_q,
        rn,
        cldwrk,
        ud_mf,
        dd_mf,
        dt_mf,
        cnvw,
        cnvc,
        gt0,
        gu0,
        gv0,
        gq0,
        dtdt,
        dudt,
        dvdt,
        dqdt,
        dtend,
        clw,
        upd_mf,
        dwn_mf,
        det_mf,
        cnvw_phy_f3d,
        cnvc_phy_f3d,
        state.require("delt"),
        state.require("grav"),
        state.require("cp"),
        state.require("hvap"),
        frain,
        area,
        prsi,
        layout,
        is_gfdl,
        imp_physics,
        tend_opt,
        transport,
        ldiag3d,
        qdiag3d,
        flag_for_dcnv_generic_tend,
        ras,
        cscnv,
        cloud_copy_rule(state.config),
        dtend_tracers,
        reconcile,
    )

    return state.with_updates(
        {
            "ten_q": ten_q,
            "gt0": gt0,
            "gu0": gu0,
            "gv0": gv0,
            "gq0": gq0,
            "dtdt": dtdt,
            "dudt": dudt,
            "dvdt": dvdt,
            "dqdt": dqdt,
            "dtend": dtend,
            "rainc": rainc,
            "cldwrk": cldwrk_out,
            "upd_mf": upd_mf,
            "dwn_mf": dwn_mf,
            "det_mf": det_mf,
            "cnvw_phy_f3d": cnvw_phy_f3d,
            "cnvc_phy_f3d": cnvc_phy_f3d,
            "cnvw": cnvw_out,
            "cnvc": cnvc_out,
            "clw": clw,
            "errflg": errflg,
        }
    )


# GFS_DCNV_generic_post: tracers left out of the tracer dtend under cscnv/satmedmf/trans_trac/ras.
DTEND_EXCLUDED = (
    "ntcw", "ntiw", "ntclamt", "ntrw", "ntsw", "ntrnc", "ntsnc", "ntgl", "ntgnc", "nthl",
    "nthnc", "nthv", "ntrz", "ntgz", "nthz", "ntgv", "ntsigma", "ntomega",
)


def _generic_post_block(
    dqtr: torch.Tensor,
    ten_t: torch.Tensor,
    ten_u: torch.Tensor,
    ten_v: torch.Tensor,
    ten_q: torch.Tensor,
    rn: torch.Tensor,
    cldwrk: torch.Tensor,
    ud_mf: torch.Tensor,
    dd_mf: torch.Tensor,
    dt_mf: torch.Tensor,
    cnvw: torch.Tensor,
    cnvc: torch.Tensor,
    gt0: torch.Tensor,
    gu0: torch.Tensor,
    gv0: torch.Tensor,
    gq0: torch.Tensor,
    dtdt: torch.Tensor,
    dudt: torch.Tensor,
    dvdt: torch.Tensor,
    dqdt: torch.Tensor,
    dtend: torch.Tensor,
    clw: torch.Tensor,
    upd_mf: torch.Tensor,
    dwn_mf: torch.Tensor,
    det_mf: torch.Tensor,
    cnvw_phy_f3d: torch.Tensor,
    cnvc_phy_f3d: torch.Tensor,
    delt: torch.Tensor,
    grav: torch.Tensor,
    cp: torch.Tensor,
    hvap: torch.Tensor,
    frain: torch.Tensor,
    area: torch.Tensor,
    prsi: torch.Tensor,
    layout: ConvectionTracerLayout,
    is_gfdl: bool,
    imp_physics: int | None,
    tend_opt: int,
    transport: bool,
    ldiag3d: bool,
    qdiag3d: bool,
    flag_for_dcnv_generic_tend: bool,
    ras: bool,
    cscnv: bool,
    cloud_rule: str,
    dtend_tracers: tuple[int, ...],
    reconcile: bool,
) -> tuple[torch.Tensor, ...]:
    ten_q = tendencies_from_dclw(ten_q, dqtr, layout, is_gfdl=is_gfdl)

    errflg = torch.zeros((), dtype=ten_t.dtype, device=ten_t.device)
    save_q = gq0
    if tend_opt == 1:
        gt0 = gt0 + delt * ten_t
        gu0 = gu0 + delt * ten_u
        gv0 = gv0 + delt * ten_v
        gq0 = gq0 + delt * ten_q
    elif tend_opt == 2:
        dtdt = dtdt + ten_t
        dudt = dudt + ten_u
        dvdt = dvdt + ten_v
        dqdt = dqdt + ten_q
    elif tend_opt == 3:
        gt0 = gt0 + delt * (dtdt + ten_t)
        gu0 = gu0 + delt * (dudt + ten_u)
        gv0 = gv0 + delt * (dvdt + ten_v)
        gq0 = gq0 + delt * (dqdt + ten_q)
        dtdt = torch.zeros_like(dtdt)
        dudt = torch.zeros_like(dudt)
        dvdt = torch.zeros_like(dvdt)
        dqdt = torch.zeros_like(dqdt)
    elif tend_opt != 4:
        errflg = torch.ones((), dtype=ten_t.dtype, device=ten_t.device)

    if reconcile:
        gq0, gt0, reconcile_errflg = samf_water_reconciliation(
            gq0, gt0, save_q, rn, area, prsi, frain, grav, cp, hvap, layout, delt=delt
        )
        errflg = torch.maximum(errflg, reconcile_errflg)

    clw = build_clw(gq0, clw, layout, imp_physics=imp_physics, transport=transport)

    cnvw_out = cnvw.clone()
    cnvc_out = cnvc.clone()
    if not ras and not cscnv:
        if cloud_rule == "cnvw_cnvc":
            cnvw_phy_f3d = cnvw.clone()
            cnvc_phy_f3d = cnvc.clone()
            cnvw_out = torch.zeros_like(cnvw)
            cnvc_out = torch.zeros_like(cnvc)
        elif cloud_rule == "cnvw":
            cnvw_phy_f3d = cnvw.clone()
            cnvw_out = torch.zeros_like(cnvw)

    rainc = frain * rn
    cldwrk_out = cldwrk + cldwrk * delt
    if ldiag3d and flag_for_dcnv_generic_tend:
        # dtidx (harness rule): T, u, v -> slots 1-3, tracer n -> slot 3+n.
        dtend = dtend.clone()
        dtend[:, :, 0] = dtend[:, :, 0] + (ten_t * delt) * frain
        dtend[:, :, 1] = dtend[:, :, 1] + (ten_u * delt) * frain
        dtend[:, :, 2] = dtend[:, :, 2] + (ten_v * delt) * frain
        for n in dtend_tracers:
            dtend[:, :, n + 2] = dtend[:, :, n + 2] + (ten_q[:, :, n - 1] * delt) * frain
        qv = layout.index("ntqv")
        dtend[:, :, qv + 2] = dtend[:, :, qv + 2] + (ten_q[:, :, qv - 1] * delt) * frain
        if qdiag3d:
            upd_mf = upd_mf + ud_mf * (grav * frain)
            dwn_mf = dwn_mf + dd_mf * (grav * frain)
            det_mf = det_mf + dt_mf * (grav * frain)

    return (
        ten_q,
        gt0,
        gu0,
        gv0,
        gq0,
        dtdt,
        dudt,
        dvdt,
        dqdt,
        dtend,
        upd_mf,
        dwn_mf,
        det_mf,
        cnvw_phy_f3d,
        cnvc_phy_f3d,
        cnvw_out,
        cnvc_out,
        clw,
        rainc,
        cldwrk_out,
        errflg,
    )


def checkpoint_fields(state: SaSasDeepConvectionState) -> dict[str, torch.Tensor]:
    return {name: state.require(name) for name in CHECKPOINT_FIELDS}
