# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Selected R4 unified gravity-wave drag assembly and live host binding."""

from collections.abc import Mapping
from dataclasses import replace

import torch
from sa_tke_edmf_torch.diagnostics import add_diagnostic_

from .containers import static_tensor
from .gwd import gwdps, orographic_gwd_inputs
from .gwd_nonstationary import latitude_wave_stress, nonstationary_gwd
from .gwd_small_scale import drag_suite_psl

GWD_DIAGNOSTIC_ALIASES = {
    "gw_kdis": "kdis_gw", "rdxzb": "zmtnblck",
    **{"dtau"+component+"2d_"+tag: "d"+("u" if component == "x" else "v")+"dt_"+kind
       for tag, kind in (("ms", "ogw"), ("bl", "obl"), ("ss", "oss"), ("fd", "ofd")) for component in ("x", "y")},
    **{"d"+component+"sfc_"+tag: "d"+component+"_"+kind+"col"
       for tag, kind in (("ms", "ogw"), ("bl", "obl"), ("ss", "oss"), ("fd", "ofd")) for component in ("u", "v")},
    **{n: n for n in ("du3dt_mtb", "du3dt_ogw", "du3dt_tms")},
}


def unified_gwd(fields: Mapping[str, torch.Tensor], *, dtp: torch.Tensor,
                cdmbgwd: torch.Tensor, alpha_fd: torch.Tensor,
                control: Mapping[str, object], launch_level: int = 54):
    """Execute legacy -> PSL -> nonstationary drag with source-owned winds.

    The selected R4 parent uses UGWP-v0 plus the PSL SS/TOFD implementation.
    Alternative versions and the alternate PSL LS/BL solver reject execution.
    """
    if not control["do_ugwp_v0"] or not control["do_gwd_opt_psl"] or control["do_gsl_drag_ls_bl"]:
        raise NotImplementedError("unified GWD requires the selected UGWP-v0/PSL SS/TOFD path")
    t, u, v = (fields[n] for n in ("t1", "u1", "v1"))
    ncol, nlev = t.shape
    if cdmbgwd.shape != (4,) or cdmbgwd.dtype != t.dtype or cdmbgwd.device != t.device:
        raise ValueError("unified cdmbgwd requires a device-local [4] tensor in the working dtype")
    # The source skips legacy when both controls are inactive. Guard its unused evaluation.
    enabled = (cdmbgwd[0] > 0.0) | (cdmbgwd[1] > 0.0)
    safe_cdmb = torch.where(enabled, cdmbgwd, static_tensor((0.14, 1.8, 1.0, 1.0), dtype=cdmbgwd.dtype, device=cdmbgwd.device))
    diagnostic_zeros = {n: t.new_zeros((ncol,nlev)) if n.startswith("dtau") else t.new_zeros(ncol)
        for n in ("dtaux2d_ms", "dtauy2d_ms", "dtaux2d_bl", "dtauy2d_bl", "dusfc_ms", "dvsfc_ms", "dusfc_bl", "dvsfc_bl")}
    legacy = gwdps({**fields, **diagnostic_zeros, "dusfc": t.new_zeros(ncol), "dvsfc": t.new_zeros(ncol),
                   "ten_q": t.new_zeros((ncol, nlev, control["ntrac"]))},
                  imx=control["lonr"], cdmbgwd=safe_cdmb, dtp=dtp,
                  ldiag_ugwp=control["ldiag_ugwp"])
    legacy = {n: torch.where(enabled, value, fields["elvmax"] if n == "elvmax" else 0.0)
              for n, value in legacy.items()}
    dtend = fields["dtend"]
    if control["ldiag3d"] and control["lssav"] and not control["flag_for_gwd_generic_tend"]:
        dtend = dtend.clone()
        for name, value in (("index_of_x_wind", legacy["b"]),
                            ("index_of_y_wind", legacy["a"]), ("index_of_temperature", legacy["c"])):
            slot = fields["dtidx"][control[name]-1, control["index_of_process_orographic_gwd"]-1]
            add_diagnostic_(dtend, value * dtp, slot)
    result = dict(elvmax=legacy["elvmax"], gw_dqdt=legacy["ten_q"],
                  dusfcg=legacy["dusfc"], dvsfcg=legacy["dvsfc"], rdxzb=legacy["rdxzb"])
    psl = {"dudt": torch.zeros_like(t), "dvdt": torch.zeros_like(t), "dtdt": torch.zeros_like(t)}
    if control["do_gsl_drag_ss"] or control["do_gsl_drag_tofd"]:
        psl_fields = {**fields, "elvmax": legacy["elvmax"], "var": fields["hprime"], "dtend": dtend}
        if control["ugwp_seq_update"]:
            psl_fields.update(u1=u+legacy["b"]*dtp, v1=v+legacy["a"]*dtp)
        psl = drag_suite_psl(psl_fields, dtp=dtp, alpha_fd=alpha_fd, control=control)
        result.update(dusfcg=legacy["dusfc"]+psl["dusfc"], dvsfcg=legacy["dvsfc"]+psl["dvsfc"], rdxzb=psl["rdxzb"])
        dtend = psl["dtend"]
    ngw_active = cdmbgwd[2] > 0.0
    tau = latitude_wave_stress(fields["xlat_d"])
    modulate = (1.0-cdmbgwd[2]).abs() > 1.e-6
    turbulence = t.new_zeros(ncol)
    if control["ntke"] > 0:
        tke = fields["q_tke"] + fields["dqdt_tke"] * dtp
        mass = t.new_zeros(ncol)
        for k in range((nlev+nlev)//3):
            turbulence = turbulence + fields["del"][:, k] * tke[:, k]
            mass = mass + fields["del"][:, k]
        turbulence = turbulence / mass
    rfac = t.new_full((), 86400000.0) / dtp
    scale = cdmbgwd[3] * torch.clamp_max(torch.maximum(turbulence, fields["rain"]*rfac), 10.0)
    tau = torch.where(modulate & (cdmbgwd[3] > 0.0), tau*torch.clamp(scale, min=0.1, max=5.0), tau)
    tau = torch.where(modulate, tau*cdmbgwd[2], tau)
    tau = torch.where(ngw_active, tau, 0.0)
    # NGW consumes original Statein winds, never the private PSL wind update.
    ngw = nonstationary_gwd(fields, tau, launch_level=launch_level)
    ngw = {n: torch.where(ngw_active, value, 0.0) for n, value in ngw.items()}
    for name, psl_name, legacy_name, ngw_name in (
        ("gw_dudt", "dudt", "b", "pdudt"), ("gw_dvdt", "dvdt", "a", "pdvdt"),
        ("gw_dtdt", "dtdt", "c", "pdtdt")):
        result[name] = torch.where(ngw_active, psl[psl_name]+ngw[ngw_name]+legacy[legacy_name],
                                  psl[psl_name]+legacy[legacy_name])
    result["gw_kdis"] = ngw["dked"] + torch.zeros_like(t)
    result.update(tau_ngw=tau, dudt_ngw=ngw["pdudt"], dvdt_ngw=ngw["pdvdt"], dtdt_ngw=ngw["pdtdt"])
    for name in ("tau_mtb", "tau_ogw", "tau_tofd"):
        result[name] = t.new_zeros(ncol)
    for name in ("dudt_mtb", "dudt_tms"):
        result[name] = torch.zeros_like(t)
    if control["ldiag3d"] and control["lssav"] and not control["flag_for_gwd_generic_tend"]:
        dtend = dtend.clone()
        for name, value in (("index_of_x_wind", ngw["pdudt"]),
                            ("index_of_y_wind", ngw["pdvdt"]), ("index_of_temperature", ngw["pdtdt"])):
            slot = fields["dtidx"][control[name]-1, control["index_of_process_nonorographic_gwd"]-1]
            add_diagnostic_(dtend, value*dtp, slot)
    result["dtend"] = dtend
    if control["ldiag_ugwp"]:
        for tag, source_tag in (("ms", "ls"), ("bl", "bl"), ("ss", "ss"), ("fd", "fd")):
            for component in ("x", "y"):
                name = "dtau"+component+"2d_"+tag
                source = "dtau"+component+"2d_"+source_tag
                result[name] = psl.get(source, legacy.get(name, torch.zeros_like(t)))
            for component in ("u", "v"):
                name = "d"+component+"sfc_"+tag
                result[name] = psl.get("d"+component+"sfc_"+source_tag, legacy.get(name, t.new_zeros(ncol)))
        for name in ("du3dt_mtb", "du3dt_ogw", "du3dt_tms"):
            result[name] = torch.zeros_like(t)
    return result


def apply_unified_gwd(statein, grid, prefix, hpbl, *, dtp, cdmbgwd, alpha_fd,
                      dtidx, control, launch_level=54):
    """Bind the original cap's Statein, live PBL carries and shared diagnostic ledger."""
    work = prefix.interstitial
    fields = {**orographic_gwd_inputs(statein, prefix),
        **{n: work[n] for n in ("varss", "oa4ss", "clxss")},
        "hpbl": hpbl, "br1": work["rb"], "slmsk": prefix.sfcprop.slmsk,
        "vtype": prefix.sfcprop.vtype, "dx": grid.dx, "xlat_d": grid.xlat_d,
        "sinlat": grid.sinlat, "rain": prefix.sfcprop.tprcp,
        "q_tke": statein.qgrs[:, :, control["ntke"]-1],
        "dqdt_tke": work["dqdt"][:, :, control["ntke"]-1],
        "dtend": prefix.carry.intdiag["dtend"], "dtidx": dtidx}
    result = unified_gwd(fields, dtp=dtp, cdmbgwd=cdmbgwd, alpha_fd=alpha_fd,
                         control=control, launch_level=launch_level)
    diag = {**prefix.carry.intdiag, "dtend": result["dtend"]}
    for argument, member in GWD_DIAGNOSTIC_ALIASES.items():
        if argument in result:
            diag[member] = result[argument]
    work_names = ("elvmax", "dusfcg", "dvsfcg", "tau_mtb", "tau_ogw", "tau_tofd", "tau_ngw",
                  "dudt_mtb", "dudt_tms", "dudt_ngw", "dvdt_ngw", "dtdt_ngw")
    return replace(prefix, interstitial={**work, **{n: result[n] for n in work_names},
        "ten_t": result["gw_dtdt"], "ten_u": result["gw_dudt"], "ten_v": result["gw_dvdt"], "ten_q": result["gw_dqdt"]},
        carry=replace(prefix.carry, intdiag=diag))
