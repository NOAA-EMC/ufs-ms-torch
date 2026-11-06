# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Selected R4 hourly diagnostics and ordered end-of-physics diagnostics."""

from dataclasses import dataclass, replace
from typing import Mapping

import torch

from ufs_physics_common.compile_policy import compiled

from .containers import static_tensor
from .clock import _nint_nonnegative
from .runtime_policy import scheme_compile_selected
from .precision import WORKING_DTYPES


HOURLY_FIELDS = (
    "refdmax", "refdmax263k", "u10max", "v10max", "spd10max",
    "t02max", "t02min", "rh02max", "rh02min", "pratemax",
)
CHEMISTRY_FIELDS = (
    "do3_dt_prd", "do3_dt_ozmx", "do3_dt_temp", "do3_dt_ohoz",
    "dqv_dt_prd", "dqv_dt_qvmx",
)


def hourly_reset(kdt, *, dtp, avg_max_length):
    """CCPP_Interstitials_reset: MOD(kdt-1,NINT(avg_max_length/dtp))==0."""
    if kdt.shape != () or kdt.dtype != torch.int32:
        raise ValueError("hourly reset requires a scalar int32 cap step")
    for value in (dtp, avg_max_length):
        if value.shape != () or value.dtype not in WORKING_DTYPES or value.device != kdt.device:
            raise ValueError("hourly reset requires device-local float32/float64 intervals")
    torch._assert_async((kdt >= 1) & torch.isfinite(dtp) & (dtp > 0.)
        & torch.isfinite(avg_max_length) & (avg_max_length > 0.), "invalid hourly clock")
    count = _nint_nonnegative(avg_max_length/dtp)
    torch._assert_async(count >= 1, "hourly reset interval rounds to zero")
    return torch.remainder(kdt.to(torch.int64)-1, count) == 0


def maximum_hourly_diagnostics(fields, *, reset, control):
    for name in ("lradar", "lightning_threat"):
        if type(control[name]) is not bool:
            raise TypeError(f"hourly diagnostics requires static {name}")
        if control[name]:
            raise NotImplementedError("selected R4 hourly diagnostics has radar/lightning off")
    f, reference = fields, fields["gt0"]
    if reference.ndim != 2 or min(reference.shape) < 1 or reference.dtype not in WORKING_DTYPES:
        raise ValueError("hourly diagnostics requires a nonempty float32/float64 column/layer state")
    if reset.shape != () or reset.dtype != torch.bool or reset.device != reference.device:
        raise ValueError("hourly reset requires a device-local logical scalar")
    for name in (*HOURLY_FIELDS, "u10m", "v10m", "pgr", "t2m", "q2m", "rain"):
        value = f[name]
        if value.shape != reference.shape[:1] or value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(f"invalid hourly diagnostic {name}")
        torch._assert_async(torch.isfinite(value).all(), f"nonfinite hourly diagnostic {name}")
    dtp = f["dtp"]
    if dtp.shape != () or dtp.dtype != reference.dtype or dtp.device != reference.device:
        raise ValueError("hourly diagnostics requires a device-local timestep")
    torch._assert_async((dtp > 0.) & torch.isfinite(dtp)
        & ((reference[:,0] > 0.) & (f["pgr"] > 0.) & (f["t2m"] > 35.86)).all(),
        "invalid hourly temperature/pressure/timestep")
    out = {n:f[n] for n in HOURLY_FIELDS}
    for name in ("spd10max", "u10max", "v10max", "t02max", "rh02max"):
        out[name] = torch.where(reset, -999., f[name])
    for name in ("t02min", "rh02min"):
        out[name] = torch.where(reset, 999., f[name])
    out["pratemax"] = torch.where(reset, 0., f["pratemax"])
    speed_squared = f["u10m"]*f["u10m"]+f["v10m"]*f["v10m"]
    moving = speed_squared > 0.
    speed = torch.where(moving,torch.sqrt(torch.where(moving,speed_squared,1.)),0.)
    faster = speed > out["spd10max"]
    for name, value in (("spd10max",speed),("u10max",f["u10m"]),("v10max",f["v10m"])):
        out[name] = torch.where(faster,value,out[name])
    shelter = f["pgr"]*torch.exp(torch.div(-.068283,reference[:,0]))
    saturation = torch.div(379.90516,shelter)*torch.exp(17.2693882*(f["t2m"]-273.16)/(f["t2m"]-35.86))
    rh = torch.clamp(f["q2m"]/saturation,1.e-6,1.)
    for name in ("rh02max", "rh02min", "t02max", "t02min", "pratemax"):
        value = rh if name.startswith("rh") else f["t2m"] if name.startswith("t02") else (3.6e6/dtp)*f["rain"]
        out[name] = (torch.minimum if name.endswith("min") else torch.maximum)(out[name],value)
    return out


@dataclass(frozen=True)
class StochasticsResult:
    fields: Mapping[str, torch.Tensor]
    errmsg: str = ""
    errflg: int = 0


def stochastics_off(fields, *, control, initialize=False):
    """Original selected-off init/run: clear errors and preserve every buffer."""
    if type(initialize) is not bool:
        raise TypeError("stochastics requires static initialize")
    names = ("do_ca", "ca_global") if initialize else ("do_sppt", "do_ca", "ca_global", "do_shum", "do_skeb")
    for name in names:
        if type(control[name]) is not bool:
            raise TypeError(f"stochastics requires static {name}")
    active = control["do_ca"] and control["ca_global"] if initialize else (
        control["do_sppt"] or control["do_ca"] or control["do_shum"] or control["do_skeb"])
    if active:
        raise NotImplementedError("active stochastic physics is outside selected R4")
    return StochasticsResult(fields)


def physics_post(fields, *, control):
    """GFS_physics_post, preserving source process order and destination aliases."""
    if type(control["ldiag3d"]) is not bool:
        raise TypeError("physics post requires static ldiag3d")
    if not control["ldiag3d"]:
        return {"dtend":fields["dtend"]} if "dtend" in fields else {}
    for name in ("ntoz", "ntqv", "ntracp100", "nprocess", "nprocess_summed",
                 "ip_physics", "ip_photochem", "ip_prod_loss", "ip_ozmix", "ip_temp", "ip_overhead_ozone"):
        if type(control[name]) is not int:
            raise TypeError(f"physics post requires static {name}")
    f,c = fields,control
    dtend,idx,chem = f["dtend"],f["dtidx"],f["is_photochem"]
    nr,np = c["ntracp100"],c["nprocess"]
    if (dtend.ndim != 3 or dtend.dtype not in WORKING_DTYPES or dtend.shape[2] < 1
        or idx.shape != (nr,np) or idx.dtype not in (torch.int32,torch.int64)
        or chem.shape != (np,) or chem.dtype != torch.bool
        or idx.device != dtend.device or chem.device != dtend.device
        or not 0 <= c["nprocess_summed"] <= np):
        raise ValueError("invalid physics post diagnostic allocation")
    for name in ("ip_physics", "ip_photochem", "ip_prod_loss", "ip_ozmix", "ip_temp", "ip_overhead_ozone"):
        if not 1 <= c[name] <= np:
            raise ValueError(f"invalid physics post process {name}")
    if not all(1 <= 100+c[n] <= nr for n in ("ntoz","ntqv")):
        raise ValueError("invalid physics post tracer index")
    torch._assert_async((idx <= dtend.shape[2]).all(), "physics post slot exceeds allocation")
    dtend = dtend.clone()  # owned: every write below replaces one slot in place

    def read(value,slot):
        return value.index_select(2,(slot-1).clamp_min(0).reshape(1).to(torch.int64)).squeeze(2)

    def write(value,slot,term):
        live = torch.where(slot >= 1,term,read(value,slot))
        return value.index_copy_(2,(slot-1).clamp_min(0).reshape(1).to(torch.int64),live.unsqueeze(2))

    def add(value,row,process,name):
        if name not in f:
            return value
        term = f[name]
        if term.shape != dtend.shape[:2] or term.dtype != dtend.dtype or term.device != dtend.device:
            raise ValueError(f"invalid chemistry tendency {name}")
        slot = idx[row-1,c[process]-1]
        return write(value,slot,read(value,slot)+term)

    def sum_it(value,row,process,mask):
        destination = idx[row-1,c[process]-1]
        count = c["nprocess_summed"]
        sources = idx[row-1,:count]
        # value is unchanged until the final write, so all source slots are read at once.
        reads = value.index_select(2,(sources-1).clamp_min(0).to(torch.int64)).movedim(2,0).contiguous()
        active,alias = mask[:count] & (sources >= 1),sources == destination
        total,first = read(value,destination),torch.ones((),dtype=torch.bool,device=value.device)
        for p in range(count):
            # Source reads the already-written destination if a later term aliases it.
            term = torch.where(alias[p] & ~first,total,reads[p])
            total = torch.where(active[p],torch.where(first,term,total+term),total)
            first = first & ~active[p]
        return write(value,destination,torch.where(first,0.,total))

    for tracer,terms in ((c["ntoz"],CHEMISTRY_FIELDS[:4]),(c["ntqv"],CHEMISTRY_FIELDS[4:])):
        for process,name in zip(("ip_prod_loss","ip_ozmix","ip_temp","ip_overhead_ozone"),terms):
            dtend = add(dtend,100+tracer,process,name)
        dtend = sum_it(dtend,100+tracer,"ip_photochem",chem)
    rows = c.get("physics_variables",tuple(range(2,nr+1)))
    if type(rows) is not tuple or any(type(r) is not int or not 2 <= r <= nr for r in rows) or len(set(rows)) != len(rows) or tuple(sorted(rows)) != rows:
        raise ValueError("physics_variables must be unique ascending static source rows")
    excluded = tuple(r-1 for r in range(2,nr+1) if r not in rows)
    if excluded:
        torch._assert_async((idx[static_tensor(excluded,dtype=torch.int64,device=idx.device),c["ip_physics"]-1] < 1).all(),
            "prepared physics rows omit an active source sum")
    all_true = torch.ones(np,dtype=torch.bool,device=dtend.device)
    for row in rows:
        dtend = sum_it(dtend,row,"ip_physics",all_true)
    return {"dtend":dtend}


def finish_microphysics(statein,prefix,*,dtp,kdt,avg_max_length,dtidx,is_photochem,control):
    """Selected cap tail: hourly -> off stochastics -> ordered physics post."""
    reset = hourly_reset(kdt,dtp=dtp,avg_max_length=avg_max_length)
    diag = prefix.carry.intdiag
    f = {**diag,"gt0":prefix.stateout.gt0,"pgr":statein.pgr,
         "t2m":prefix.sfcprop.t2m,"q2m":prefix.sfcprop.q2m,"dtp":dtp}
    hourly = maximum_hourly_diagnostics(f,reset=reset,control=control)
    stochastics_off({},control=control)
    # The generated R4 cap passes null water-chemistry pointers when h2o_phys is off.
    for flag in ("ldiag3d", "oz_phys_2015", "h2o_phys"):
        if type(control[flag]) is not bool:
            raise TypeError(f"physics post requires static {flag}")
    chemistry = {n:diag[n] for n in CHEMISTRY_FIELDS
        if control["ldiag3d"] and control["oz_phys_2015" if n.startswith("do3") else "h2o_phys"]}
    fields = {"dtidx":dtidx,"is_photochem":is_photochem,**chemistry}
    if "dtend" in diag:
        fields["dtend"] = diag["dtend"]
    # Compiled (exact policy) with the schemes: its slot sums are 15-term ordered chains per row.
    post = compiled(physics_post) if scheme_compile_selected("physics_post") else physics_post
    final = post(fields,control=control)
    return replace(prefix,carry=replace(prefix.carry,intdiag={**diag,**hourly,**final}),
        interstitial={**prefix.interstitial,"max_hourly_reset":reset})
