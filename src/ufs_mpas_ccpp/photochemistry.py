# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Selected R4 ozone photochemistry, the GFS v17 h2o photochemistry, and their unfloored tendency commit."""

from dataclasses import replace

import torch

from .containers import assert_all_finite

from .ozone import h2o_photochemistry, ozone_2015
from .precision import WORKING_DTYPES


def photochemistry_2015(fields, *, log_pressure, forcing, dtp, ntoz, h2o=None):
    """GFS_photochemistry_run for ozone2015 (oz_phys_2006 off).

    With ``h2o`` (mapping with ``log_pressure``, ``forcing`` = h2opl and the static ``ntqv``) the
    h2o_phys block also runs on the initial specific humidity; otherwise it is off.
    """
    q, temperature = fields["gq0"], fields["gt0"]
    if q.ndim != 3 or q.shape[:2] != temperature.shape or type(ntoz) is not int:
        raise ValueError("invalid ozone2015 Stateout extent or static tracer index")
    if not 1 <= ntoz <= q.shape[2] or q.dtype not in WORKING_DTYPES or q.device != temperature.device:
        raise ValueError("invalid ozone2015 tracer binding")
    area = fields["area"]
    if area.shape != (q.shape[0],) or area.dtype != q.dtype or area.device != q.device:
        raise ValueError("invalid ozone2015 cell area")
    torch._assert_async(torch.isfinite(area).all() & (area > 0.).all(), "invalid ozone2015 cell area")
    result = ozone_2015(fields["prsl"], temperature, fields["dp"], q[:, :, ntoz - 1],
        log_pressure=log_pressure, forcing=forcing, dtp=dtp)
    tendency = (result["oz"] - q[:, :, ntoz - 1]) / dtp
    zero = torch.zeros_like(temperature)
    tracers = torch.arange(q.shape[2], device=q.device)
    ten_q = torch.where(tracers == ntoz - 1, tendency[..., None], torch.zeros_like(q))
    out = {n: result[n] for n in result if n != "oz"}
    if h2o is not None:
        ntqv = h2o["ntqv"]
        if type(ntqv) is not int or not 1 <= ntqv <= q.shape[2] or ntqv == ntoz:
            raise ValueError("invalid h2o photochemistry static tracer index")
        water = h2o_photochemistry(fields["prsl"], q[:, :, ntqv - 1],
            log_pressure=h2o["log_pressure"], forcing=h2o["forcing"], dtp=dtp)
        ten_q = torch.where(tracers == ntqv - 1, ((water["h2o"] - q[:, :, ntqv - 1]) / dtp)[..., None], ten_q)
        out.update(dqv_dt_prd=water["dqv_dt_prd"], dqv_dt_qvmx=water["dqv_dt_qvmx"])
    return {**out, "ten_q": ten_q, "ten_u": zero, "ten_v": zero, "ten_t": zero}


def photochemistry_post(fields, *, dtp, mode):
    """Original four post modes; unlike GWD, this post never floors tracers."""
    if type(mode) is not int or mode not in (1, 2, 3, 4):
        raise ValueError("photochemistry tendency mode must be a static integer 1..4")
    q = fields["gq0"]
    if q.ndim != 3 or min(q.shape) < 1 or dtp.ndim != 0:
        raise ValueError("invalid photochemistry post dimensions")
    if dtp.dtype not in WORKING_DTYPES or dtp.device != q.device:
        raise ValueError("photochemistry post requires a colocated float32/float64 timestep")
    torch._assert_async(torch.isfinite(dtp) & (dtp > 0.), "invalid photochemistry post timestep")
    result = {}
    triples = (("gt0", "dtdt", "ten_t"), ("gu0", "dudt", "ten_u"), ("gv0", "dvdt", "ten_v"), ("gq0", "dqdt", "ten_q"))
    assert_all_finite({name: fields[name] for triple in triples for name in triple}, "photochemistry post")
    for state, ledger, tendency in triples:
        for name in (state, ledger, tendency):
            value = fields[name]
            if value.shape != (q.shape if state == "gq0" else q.shape[:2]) or value.dtype != q.dtype or value.device != q.device:
                raise ValueError(f"invalid photochemistry post {name}")
        result[state], result[ledger] = fields[state], fields[ledger]
        if mode in (1, 3):
            current = fields[tendency] if mode == 1 else fields[ledger] + fields[tendency]
            result[state] = fields[state] + dtp * current
        if mode == 2:
            result[ledger] = fields[ledger] + fields[tendency]
        elif mode == 3:
            result[ledger] = torch.zeros_like(fields[ledger])
    assert_all_finite(result, "photochemistry post output")
    return result


def apply_photochemistry(statein, grid, prefix, *, log_pressure, forcing, dtp, control,
                         h2o_log_pressure=None, h2o_forcing=None):
    """Bind the active ozone2015 (and, with h2o_phys, h2o) cap and post without altering other carries."""
    for name in ("oz_phys_2015", "oz_phys", "h2o_phys", "ldiag3d"):
        if type(control[name]) is not bool:
            raise TypeError("photochemistry switches must be static booleans")
    if not control["oz_phys_2015"] or control["oz_phys"]:
        raise NotImplementedError("this cap binding selects ozone2015 (and optionally h2o_phys) only")
    if control["h2o_phys"] != (h2o_forcing is not None) or (h2o_forcing is None) != (h2o_log_pressure is None):
        raise ValueError("h2o_phys requires the h2o forcing (h2opl) and its log-pressure levels, and only then")
    work, diagnostics = prefix.interstitial, prefix.carry.intdiag
    fields = {**work, "area": grid.area, "prsl": statein.prsl, "dp": work["del"],
        **{n: getattr(prefix.stateout, n) for n in ("gt0", "gu0", "gv0", "gq0")}}
    h2o = (dict(log_pressure=h2o_log_pressure, forcing=h2o_forcing, ntqv=control["ntqv"])
           if control["h2o_phys"] else None)
    tendencies = photochemistry_2015(fields, log_pressure=log_pressure, forcing=forcing,
        dtp=dtp, ntoz=control["ntoz"], h2o=h2o)
    post = photochemistry_post({**fields, **tendencies}, dtp=dtp, mode=control["tend_opt_photochem"])
    updates = {n: tendencies[n] for n in tendencies if n.startswith("ten_")}
    if control["ldiag3d"]:
        for name in tendencies:
            if name.startswith(("do3_", "dqv_")):
                if (name not in diagnostics or diagnostics[name].shape != statein.prsl.shape
                    or diagnostics[name].dtype != statein.prsl.dtype or diagnostics[name].device != statein.prsl.device):
                    raise ValueError(f"missing active photochemistry diagnostic {name}")
                diagnostics = {**diagnostics, name: tendencies[name]}
    return replace(prefix,
        stateout=replace(prefix.stateout, **{n: post[n] for n in ("gt0", "gu0", "gv0", "gq0")}),
        interstitial={**work, **updates, **{n: post[n] for n in ("dtdt", "dudt", "dvdt", "dqdt")}},
        carry=replace(prefix.carry, intdiag=diagnostics))
