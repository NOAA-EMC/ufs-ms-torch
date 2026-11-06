# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""R4 interstitial4 number/ledger updates and persistent cnvc90 history."""

from dataclasses import dataclass, replace
from math import pi

import torch

from .containers import static_tensor
from .clock import _nint_nonnegative
from .runtime_policy import glue
from .precision import WORKING_DTYPES, working_like


_ICE_RADIUS = (
    5.92779,6.26422,6.61973,6.99539,7.39234,7.81177,8.25496,8.72323,9.21800,9.74075,
    10.2930,10.8765,11.4929,12.1440,12.8317,13.5581,14.2319,15.0351,15.8799,16.7674,
    17.6986,18.6744,19.6955,20.7623,21.8757,23.0364,24.2452,25.5034,26.8125,27.7895,
    28.6450,29.4167,30.1088,30.7306,31.2943,31.8151,32.3077,32.7870,33.2657,33.7540,
    34.2601,34.7892,35.3442,35.9255,36.5316,37.1602,37.8078,38.4720,39.1508,39.8442,
    40.5552,41.2912,42.0635,42.8876,43.7863,44.7853,45.9170,47.2165,48.7221,50.4710,
    52.4980,54.8315,57.4898,60.4785,63.7898,65.5604,71.2885,75.4113,79.7368,84.2351,
    88.8833,93.6658,98.5739,103.603,108.752,114.025,119.424,124.954,130.630,136.457,
    142.446,148.608,154.956,161.503,168.262,175.248,182.473,189.952,197.699,205.728,
    214.055,222.694,231.661,240.971,250.639,
)


def make_ice_number(ice_mass, temperature, host_pi):
    """TEMPO utils_v2 mass-to-number law after mp_tempo_init sets host pi."""
    index = torch.trunc(temperature-179.).to(torch.int64).clamp(1,94)-1
    table = static_tensor(_ICE_RADIUS, dtype=temperature.dtype, device=temperature.device)
    fraction = temperature-torch.trunc(temperature)
    radius = table[index]*(1.-fraction)+table[index+1]*fraction
    diameter = 2.*radius*1.e-6
    slope = torch.div(3.,diameter)
    number = ice_mass*slope*slope*slope/(host_pi*890.)
    # The source's Q==0 early return is still a continuous linear mass law.
    # Adding positive zero preserves that derivative and canonicalizes -0.
    return number+torch.zeros_like(number)


def interstitial_4(fields, *, control):
    """Selected ten-tracer TEMPO cap, with live source diagnostic indices."""
    expected = dict(imp_physics=88,imp_physics_tempo=88,ntrac=10,nn=9,
        ntcw=2,ntiw=4,ntlnc=0,ntinc=7,ntk=3,ntke=9)
    for name,value in expected.items():
        if type(control[name]) is not int:
            raise TypeError(f"interstitial4 requires static {name}")
        if control[name] != value:
            raise NotImplementedError(f"interstitial4 selects R4 {name}={value}")
    for name in ("convert_dry_rho","ldiag3d","qdiag3d"):
        if type(control[name]) is not bool:
            raise TypeError(f"interstitial4 requires static {name}")
    if type(control["tracers_total"]) is not int or control["tracers_total"] < 0:
        raise ValueError("interstitial4 requires a static transported tracer count")
    q,ledger,indices = fields["gq0"],fields["dtend"],fields["dtidx"]
    if q.ndim != 3 or q.shape[2] != 10 or min(q.shape) < 1:
        raise ValueError("interstitial4 requires ten nonempty column-first tracers")
    if ledger.ndim != 3 or ledger.shape[:2] != q.shape[:2] or ledger.shape[2] < (1 if control["ldiag3d"] else 0):
        raise ValueError("invalid interstitial4 diagnostic allocation")
    if fields["clw"].shape != (*q.shape[:2],9):
        raise ValueError("interstitial4 requires the nine-slot R4 clw layout")
    if indices.ndim != 2 or indices.shape[0] < 110 or indices.dtype not in (torch.int32,torch.int64):
        raise ValueError("invalid interstitial4 diagnostic map")
    flags = fields["otsptflag"]
    if flags.shape != (10,) or flags.dtype != torch.bool:
        raise ValueError("invalid interstitial4 transport flags")
    for name,value in fields.items():
        if value.device != q.device:
            raise ValueError(f"interstitial4 {name} is on a different device")
        if value.is_floating_point():
            if not working_like(value, q):
                raise TypeError(f"interstitial4 {name} requires the working dtype {q.dtype}")
            if name != "dtend":  # pass-through diagnostic array: not re-read for finiteness
                torch._assert_async(torch.isfinite(value).all(),f"nonfinite interstitial4 {name}")
    for name in ("prsl","save_tcp","spechum","save_qi"):
        if fields[name].shape != q.shape[:2]:
            raise ValueError(f"invalid interstitial4 {name} dimensions")
    for name in ("con_rd","con_eps","con_pi"):
        if fields[name].shape != ():
            raise ValueError(f"interstitial4 {name} requires a scalar tensor")
        torch._assert_async(fields[name] > 0.,f"interstitial4 {name} must be positive")
    processes = []
    for name in ("conv_trans","dcnv","scnv"):
        index = control["index_of_process_"+name]
        if type(index) is not int or not 1 <= index <= indices.shape[1]:
            raise ValueError("invalid interstitial4 diagnostic process")
        processes.append(index-1)
    transport,deep,shallow = processes

    def accumulate(current, variable, source_process, enabled=True):
        target,source = indices[100+variable-1,transport],indices[100+variable-1,source_process]
        active = enabled & (target > 0) & (source > 0)
        torch._assert_async(~active | ((target <= ledger.shape[2]) & (source <= ledger.shape[2])),
                            "interstitial4 diagnostic slot exceeds allocation")
        # In place on the owned ledger: read the source and target slots, write the target only.
        address = source.to(torch.int64).clamp(1,ledger.shape[2]).reshape(1)-1
        destination = target.to(torch.int64).clamp(1,ledger.shape[2]).reshape(1)-1
        value = current.index_select(2,address)
        held = current.index_select(2,destination)
        return current.index_copy_(2,destination,torch.where(active,held+value,held))

    # ldiag3d off allocates no slots (every dtidx entry -99): the source writes nothing.
    slotted = ledger.shape[2] > 0
    if slotted and (control["ldiag3d"] or control["tracers_total"] > 0):
        ledger = ledger.clone()
    if control["ldiag3d"]:
        for variable in (control["ntke"],control["ntiw"],control["ntcw"]):
            ledger = accumulate(ledger,variable,deep)
            ledger = accumulate(ledger,variable,shallow)
    if slotted and control["tracers_total"] > 0:
        # Source compares the host tracer index against ntk (the clw slot),
        # not ntke. Thus this layout copies the TKE diagnostic a second time.
        excluded = (control["ntk"],control["ntlnc"],control["ntinc"],control["ntcw"],control["ntiw"])
        for variable in range(2,11):
            if variable not in excluded:
                ledger = accumulate(ledger,variable,deep,flags[variable-1])
                ledger = accumulate(ledger,variable,shallow,flags[variable-1])

    humidity = fields["spechum"]
    temperature = fields["save_tcp"]
    torch._assert_async(((humidity >= 0.) & (humidity < 1.) & (temperature > 0.) & (fields["prsl"] > 0.)).all(),
                        "invalid interstitial4 humidity/temperature/pressure")
    old_number = q[:,:,6]
    mass = q[:,:,3]-fields["save_qi"]
    if control["convert_dry_rho"]:
        vapor = humidity/(1.-humidity)
        density = fields["con_eps"]*fields["prsl"]/(fields["con_rd"]*temperature*(vapor+fields["con_eps"]))
        inverse = torch.reciprocal(density)
        dry_mass,dry_number = mass/(1.-humidity),old_number/(1.-humidity)
        new_number = torch.clamp_min(dry_number+make_ice_number(dry_mass*density,temperature,fields["con_pi"])*inverse,0.)
        new_number = new_number/(1.+vapor)
    else:
        density = fields["con_eps"]*fields["prsl"]/(fields["con_rd"]*temperature*(humidity+fields["con_eps"]))
        inverse = torch.reciprocal(density)
        new_number = torch.clamp_min(old_number+make_ice_number(mass*density,temperature,fields["con_pi"])*inverse,0.)
    if control["ldiag3d"] and control["qdiag3d"]:
        target = indices[100+control["ntinc"]-1,transport]
        torch._assert_async(target <= ledger.shape[2],"interstitial4 ice diagnostic exceeds allocation")
        # Preserve the source's left-associated ledger+new-old cancellation.
        destination = target.to(torch.int64).clamp(1,ledger.shape[2]).reshape(1)-1
        held = ledger.index_select(2,destination)
        ledger.index_copy_(2,destination,torch.where(target > 0,(held+new_number[...,None])-old_number[...,None],held))
    return dict(gq0=torch.cat((q[:,:,:6],new_number[...,None],q[:,:,7:]),dim=2),dtend=ledger)


@dataclass(frozen=True)
class ConvectiveCloudHistory:
    acv: torch.Tensor
    acvb: torch.Tensor
    acvt: torch.Tensor
    cv: torch.Tensor
    cvb: torch.Tensor
    cvt: torch.Tensor

    @classmethod
    def allocate_r4(cls, *, ncol, device, dtype=torch.float64):
        if type(ncol) is not int or ncol < 1:
            raise ValueError("cloud history requires a positive static column count")
        return cls(*(torch.zeros(ncol,dtype=dtype,device=device) for _ in range(6)))


def convective_cloud_clock(kdt,lsswr,fhour,*,nsswr,fhswr):
    """GFS_phys_time_vary's four cnvc90 initialize/accumulate/convert modes."""
    if type(nsswr) is not int or nsswr < 1:
        raise ValueError("cloud clock requires a positive static SW interval")
    # fhour is the float64 clock; fhswr a scalar control in either working dtype.
    for name,value,dtype in (("kdt",kdt,(torch.int32,)),("lsswr",lsswr,(torch.bool,)),
                             ("fhour",fhour,(torch.float64,)),("fhswr",fhswr,WORKING_DTYPES)):
        if value.shape != () or value.dtype not in dtype or value.device != fhour.device:
            raise ValueError(f"invalid cloud clock {name}")
    torch._assert_async(torch.isfinite(fhour) & torch.isfinite(fhswr) & (fhswr > 0.) & (fhour >= 0.) & (kdt > 0),
                        "invalid cloud clock interval")
    elapsed = torch.minimum(torch.minimum(fhswr/3600.,fhour),fhour.new_full((), 99.))
    convert = torch.remainder(kdt,nsswr) == 0
    return torch.where(lsswr,1100.,100.)+torch.where(convert,elapsed,0.)


def cnvc90(history, *, clstp, rainc, kbot, ktop, prsi):
    """Source cloud accumulation, table interpolation and retained pressure bounds."""
    ncol,ninterface = prsi.shape
    if ninterface < 2 or clstp.shape != () or min(ncol,ninterface) < 1:
        raise ValueError("invalid cnvc90 dimensions")
    for name,value in (("clstp",clstp),("prsi",prsi),("rainc",rainc),
                       *((n,getattr(history,n)) for n in history.__dataclass_fields__)):
        shape = () if name == "clstp" else prsi.shape if name == "prsi" else (ncol,)
        if value.shape != shape or not working_like(value, prsi) or value.device != prsi.device:
            raise ValueError(f"invalid cnvc90 {name}")
        torch._assert_async(torch.isfinite(value).all(),f"nonfinite cnvc90 {name}")
    for value in (kbot,ktop):
        if value.shape != (ncol,) or value.dtype not in (torch.int32,torch.int64) or value.device != prsi.device:
            raise ValueError("invalid cnvc90 integer cloud bounds")
    reset = clstp >= 1000.
    accumulate = (clstp >= 1100.) | ((clstp < 1000.) & (clstp >= 100.))
    hours = torch.fmod(clstp,100.)
    convert = (hours > .01) & (hours < 99.99)
    acv = torch.where(reset,0.,history.acv)
    acvb = torch.where(reset,100.,history.acvb)
    acvt = torch.where(reset,0.,history.acvt)
    rainy = accumulate & (rainc > 0.)
    acv = torch.where(rainy,acv+rainc,acv)
    acvb = torch.where(rainy,torch.minimum(acvb,kbot.to(prsi.dtype)),acvb)
    acvt = torch.where(rainy,torch.maximum(acvt,ktop.to(prsi.dtype)),acvt)
    positive = convert & (acv > 0.)
    bottom = _nint_nonnegative(torch.where(positive,acvb,0.))
    top = _nint_nonnegative(torch.where(positive,acvt,0.))
    torch._assert_async((~positive | ((bottom >= 1) & (bottom <= ninterface) & (top >= 0) & (top < ninterface))).all(),
                        "cnvc90 history exceeds interface bounds")
    b_index = torch.where(positive,bottom-1,0).clamp(0,ninterface-1)
    t_index = torch.where(positive,top,0).clamp(0,ninterface-1)
    cloud_bottom = torch.gather(prsi,1,b_index[:,None])[:,0]
    cloud_top = torch.gather(prsi,1,t_index[:,None])[:,0]
    cvb = torch.where(convert,torch.where(positive,cloud_bottom,0.),history.cvb)
    cvt = torch.where(convert,torch.where(positive,cloud_top,0.),history.cvt)
    safe_hours = torch.where(convert,hours,1.)
    precipitation = acv*torch.div(24.e3,safe_hours)
    thresholds = static_tensor((.14,.31,.70,1.6,3.4,7.7,17.,38.,85.),dtype=prsi.dtype,device=prsi.device)
    fractions = static_tensor((0.,.1,.2,.3,.4,.5,.6,.7,.8),dtype=prsi.dtype,device=prsi.device)
    interval = (precipitation[:,None] > thresholds).to(torch.int64).sum(dim=1)
    low = (interval-1).clamp(0,7)
    interpolated = fractions[low]+(fractions[low+1]-fractions[low])*(precipitation-thresholds[low])/(thresholds[low+1]-thresholds[low])
    cloud = torch.where(interval == 0,0.,torch.where(interval == 9,.8,interpolated))
    cv = torch.where(convert,cloud,history.cv)
    cvb = torch.where(convert & (interval == 0),0.,cvb)
    cvt = torch.where(convert & (interval == 0),0.,cvt)
    return ConvectiveCloudHistory(acv,acvb,acvt,cv,cvb,cvt)


def prepare_convective_clouds(statein, prefix, history, *, clstp,dtidx,control):
    """Continue the live shallow post through interstitial4 and cnvc90."""
    work,carry = prefix.interstitial,prefix.carry
    fields = dict(gq0=prefix.stateout.gq0,clw=work["clw"],prsl=statein.prsl,
        save_tcp=work["save_tcp"],save_qi=work["save_q"][:,:,3],
        spechum=statein.qgrs[:,:,0],otsptflag=work["otsptflag"],
        dtend=carry.intdiag["dtend"],dtidx=dtidx,
        con_rd=statein.prsl.new_full((), 287.05),con_eps=statein.prsl.new_full((), 287.05/461.5),
        con_pi=statein.prsl.new_full((), pi))
    out = glue(interstitial_4)(fields,control=control)
    clouds = glue(cnvc90)(history,clstp=clstp,rainc=carry.intdiag["rainc"],
        kbot=work["kbot"],ktop=work["ktop"],prsi=statein.prsi)
    result = replace(prefix,stateout=replace(prefix.stateout,gq0=out["gq0"]),
        carry=replace(carry,intdiag={**carry.intdiag,"dtend":out["dtend"]}))
    return result,clouds
