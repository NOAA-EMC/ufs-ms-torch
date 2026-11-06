# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Original R4 unified and generic GWD post updates, in native Torch."""

from dataclasses import replace

import torch
from sa_tke_edmf_torch.diagnostics import add_diagnostic_

from .gwd_v1 import ugwpv1_gsldrag_post
from .runtime_policy import glue


def unified_gwd_post(fields, *, dtf, ldiag_ugwp):
    """Retain the original diagnostic early return and accumulation order."""
    names = ("tot_zmtb", "tot_zlwb", "tot_zogw", "tot_tofd", "tot_mtb", "tot_ogw", "tot_ngw",
        "du3dt_mtb", "du3dt_tms", "du3dt_ogw", "du3dt_ngw", "dv3dt_ngw", "dws3dt_ogw", "dws3dt_obl",
        "dws3dt_oss", "dws3dt_ofd", "ldu3dt_ogw", "ldu3dt_obl", "ldu3dt_oss", "ldu3dt_ofd",
        "du3_ogwcol", "dv3_ogwcol", "du3_oblcol", "dv3_oblcol", "du3_osscol", "dv3_osscol",
        "du3_ofdcol", "dv3_ofdcol", "ldu3dt_ngw", "ldv3dt_ngw", "ldt3dt_ngw")
    result = {n: fields[n] for n in names if n in fields}
    if not ldiag_ugwp:
        return result
    pairs = (("tot_zmtb", "zmtb"), ("tot_zlwb", "zlwb"), ("tot_zogw", "zogw"),
        ("tot_tofd", "tau_tofd"), ("tot_mtb", "tau_mtb"), ("tot_ogw", "tau_ogw"), ("tot_ngw", "tau_ngw"),
        ("du3dt_mtb", "dudt_mtb"), ("du3dt_tms", "dudt_tms"), ("du3dt_ogw", "dudt_ogw"),
        ("du3dt_ngw", "gw_dudt"), ("dv3dt_ngw", "gw_dvdt"), ("ldu3dt_ngw", "dudt_ngw"),
        ("ldv3dt_ngw", "dvdt_ngw"), ("ldt3dt_ngw", "dtdt_ngw"))
    for target, source in pairs:
        result[target] = fields[target] + dtf*fields[source]
    for tag in ("ogw", "obl", "oss", "ofd"):
        speed2 = fields["dudt_"+tag].square() + fields["dvdt_"+tag].square()
        speed = torch.where(speed2 > 0.0, torch.sqrt(torch.where(speed2 > 0.0, speed2, 1.0)), 0.0)
        result["dws3dt_"+tag] = fields["dws3dt_"+tag] + dtf*speed
        result["ldu3dt_"+tag] = fields["ldu3dt_"+tag] + dtf*fields["dudt_"+tag]
        for component in ("u", "v"):
            target, source = "d"+component+"3_"+tag+"col", "d"+component+"_"+tag+"col"
            result[target] = fields[target] + dtf*fields[source]
    return result


def generic_gwd_post(fields, *, dtp, dtf, control):
    """Apply the source's four tendency modes and exact nonnegative tracer floor.

    Source stderr accounting and test-only environment counterfactuals are tooling,
    not numerical inputs. State fields and accumulated tendencies remain separate.
    """
    mode = control["tend_opt_gwd"]
    if mode not in (1, 2, 3, 4):
        raise ValueError("GWD tendency mode must be 1..4")
    result = {n: fields[n] for n in ("gt0", "gu0", "gv0", "gq0", "dtdt", "dudt", "dvdt", "dqdt", "dugwd", "dvgwd")}
    for state, ledger, tendency in (("gt0", "dtdt", "ten_t"), ("gu0", "dudt", "ten_u"),
                                    ("gv0", "dvdt", "ten_v"), ("gq0", "dqdt", "ten_q")):
        if mode in (1, 3):
            current = fields[tendency] if mode == 1 else fields[ledger] + fields[tendency]
            proposed = fields[state] + dtp * current
            result[state] = torch.where(proposed < 0.0, 0.0, proposed) if state == "gq0" else proposed
        if mode == 2:
            result[ledger] = fields[ledger] + fields[tendency]
        elif mode == 3:
            result[ledger] = torch.zeros_like(fields[ledger])
    if control["lssav"]:
        result["dugwd"] = fields["dugwd"] + fields["dusfcg"]*dtf
        result["dvgwd"] = fields["dvgwd"] + fields["dvsfcg"]*dtf
    if "dtend" in fields:
        dtend = fields["dtend"]
        if control["lssav"] and control["ldiag3d"] and control["flag_for_gwd_generic_tend"]:
            dtend = dtend.clone()
            for variable, name in (("index_of_temperature", "ten_t"), ("index_of_x_wind", "ten_u"), ("index_of_y_wind", "ten_v")):
                slot = fields["dtidx"][control[variable]-1, control["index_of_process_orographic_gwd"]-1]
                add_diagnostic_(dtend, fields[name]*dtf, slot)
        result["dtend"] = dtend
    elif control["lssav"] and control["ldiag3d"] and control["flag_for_gwd_generic_tend"]:
        raise ValueError("active GWD diagnostics require dtend")
    return result


def apply_gwd_post(prefix, *, dtp, dtf, dtidx, control):
    """Run unified (or, for UGWP v1, ugwpv1_gsldrag) post, then generic post, on the live containers."""
    work, diag = prefix.interstitial, prefix.carry.intdiag
    if control.get("do_ugwp_v1", False):
        unified = ugwpv1_gsldrag_post(work, dtf=dtf, ldiag_ugwp=control["ldiag_ugwp"])
    else:
        aliases = {"tot_zmtb": "zmtb", "tot_zlwb": "zlwb", "tot_zogw": "zogw",
            "tot_tofd": "tau_tofd", "tot_mtb": "tau_mtb", "tot_ogw": "tau_ogw", "tot_ngw": "tau_ngw"}
        post_fields = {**diag, **work, "gw_dudt": work["ten_u"], "gw_dvdt": work["ten_v"],
                       **{argument: diag[member] for argument, member in aliases.items() if member in diag}}
        unified = unified_gwd_post(post_fields, dtf=dtf, ldiag_ugwp=control["ldiag_ugwp"])
        unified = {aliases.get(n, n): value for n, value in unified.items()}
    post = glue(generic_gwd_post)({**work, **diag, "dtidx": dtidx,
        **{n: getattr(prefix.stateout, n) for n in ("gt0", "gu0", "gv0", "gq0")}},
        dtp=dtp, dtf=dtf, control=control)
    return replace(prefix, stateout=replace(prefix.stateout, **{n: post[n] for n in ("gt0", "gu0", "gv0", "gq0")}),
        interstitial={**work, **{n: post[n] for n in ("dtdt", "dudt", "dvdt", "dqdt")}},
        carry=replace(prefix.carry, intdiag={**diag, **unified, **{n: post[n] for n in ("dugwd", "dvgwd", "dtend")}}))
