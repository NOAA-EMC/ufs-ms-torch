# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Live R4 TEMPO cap binding; generic microphysics commit is separate."""

from dataclasses import dataclass, replace
import math

import torch

from sa_tke_edmf_torch.diagnostics import add_diagnostic_
from .containers import assert_all_finite

from tempo_torch import mp_tempo, torch_native
from .photochemistry import photochemistry_post
from .convection import apply_convection
from .convective_clouds import ConvectiveCloudHistory, prepare_convective_clouds
from .physics_post import finish_microphysics
from .physics_prefix import PhysicsPrefixResult
from ufs_physics_common.compile_policy import compiled
from ufs_physics_common.precision import working_power

from .column_chunks import run_column_chunks
from .runtime_policy import (glue, column_chunk, scheme, scheme_column_chunk,
                             scheme_compile_selected, tempo_sedimentation_compaction)
from .precision import working_like


def apply_tempo(statein,prefix,*,tables,min_layer_depth,control):
    """Run the existing TEMPO wrapper/core/limiter on live shallow-post state.

    The source's local initialized itimestep is one on its first call. It
    increments once per wrapper call, not per inner sedimentation step.
    """
    for name in ("imp_physics","ntrac","tempo_call_index"):
        if type(control[name]) is not int:
            raise TypeError(f"TEMPO requires static {name}")
    if control["imp_physics"] != 88 or control["ntrac"] != 10:
        raise NotImplementedError("TEMPO binding selects the R4 ten-tracer cap")
    if control["tempo_call_index"] < 1:
        raise ValueError("TEMPO call index must be positive")
    for name in ("ltaerosol","lthailaware","convert_dry_rho","lradar","semi_sedi","tempo_mp_is_init"):
        if type(control[name]) is not bool:
            raise TypeError(f"TEMPO requires static {name}")
    if control["ltaerosol"] or control["lthailaware"]:
        raise NotImplementedError("TEMPO binding selects the R4 absent aerosol/hail optionals")
    if not control["tempo_mp_is_init"]:
        raise ValueError("TEMPO requires completed source initialization")
    for name,value in (("min_layer_depth",min_layer_depth),*((n,control[n]) for n in ("dtp","dt_inner","ttendlim"))):
        if type(value) not in (int,float):
            raise TypeError(f"TEMPO requires static {name}")
        if not math.isfinite(value) or (name != "ttendlim" and value <= 0.):
            raise ValueError(f"invalid TEMPO {name}")
    q,t = prefix.stateout.gq0,prefix.stateout.gt0
    if q.shape != (*t.shape,10) or t.ndim != 2 or min(t.shape) < 1:
        raise ValueError("TEMPO requires ten column-first source tracers")
    work,diagnostics = prefix.interstitial,prefix.carry.intdiag
    profiles = dict(spechum=q[:,:,0],qc=q[:,:,1],qr=q[:,:,2],qi=q[:,:,3],
        qs=q[:,:,4],qg=q[:,:,5],ni=q[:,:,6],nr=q[:,:,7],tgrs=t,
        prsl=statein.prsl,phii=statein.phii,omega=statein.vvl,
        refl_10cm=diagnostics["refl_10cm"],prslk=statein.prslk)
    for name,value in profiles.items():
        shape = (t.shape[0],t.shape[1]+1) if name == "phii" else t.shape
        if value.shape != shape or value.dtype != t.dtype or value.device != t.device:
            raise ValueError(f"invalid TEMPO {name} allocation")
        torch._assert_async(torch.isfinite(value).all(),f"nonfinite TEMPO {name}")
    for name,value in (("prcp",work["prcpmp"]),("rain",work["rainmp"]),
                       ("ice",work["icemp"]),("snow",work["snowmp"]),("graupel",work["graupelmp"])):
        if value.shape != t.shape[:1] or value.dtype != t.dtype or value.device != t.device:
            raise ValueError(f"invalid TEMPO {name} allocation")
        torch._assert_async(torch.isfinite(value).all(),f"nonfinite TEMPO {name}")
        profiles[name] = value
    dz = (statein.phii[:,1:]-statein.phii[:,:-1])/mp_tempo.CON_G
    torch._assert_async((dz >= min_layer_depth).all(),"TEMPO live layer violates prepared depth bound")
    torch._assert_async(((q[:,:,0] >= 0.) & (q[:,:,0] < 1.) & (t > 0.) & (statein.prsl > 0.)).all(),
                        "invalid TEMPO humidity/temperature/pressure")
    profiles.pop("prslk")
    # The static substep bounds follow fall-speed ceilings (runtime_policy "bound"); a column needing
    # more substeps sets region tempo.bound.<kind> and the interval driver reruns the step one ceiling
    # up (raise_compaction). Not torch._assert_async: compiled code drops it.
    policy = tempo_sedimentation_compaction()
    tiers = {kind: value for kind, value in policy.items() if kind != "bound"}
    ceilings = dict(policy["bound"]) if "bound" in policy else {}
    result = _tempo_core(profiles,dict(dtp=control["dtp"],dt_inner=control["dt_inner"],
        min_layer_depth=min_layer_depth,tables=tables,semi_sedi=control["semi_sedi"],
        convert_dry_rho=control["convert_dry_rho"],do_radar_ref=control["lradar"],
        itimestep=control["tempo_call_index"],sedimentation_compaction=tiers,
        max_sedimentation_fall_speed=(ceilings["rain_graupel"] if "rain_graupel" in ceilings
                                      else torch_native.DEFAULT_MAX_SEDIMENTATION_FALL_SPEED),
        max_snow_sedimentation_fall_speed=(ceilings["snow"] if "snow" in ceilings
                                           else torch_native.DEFAULT_MAX_SEDIMENTATION_FALL_SPEED),
        return_tendencies=False,
        # Compiled, the semi-Lagrangian substeps are one region reused by every substep.
        sedimentation_substep_region=scheme_compile_selected("apply_tempo")))
    if torch_native.SEDIMENTATION_SUBSTEP_OVERFLOW not in result.diagnostics:
        raise RuntimeError("TEMPO omitted its required sedimentation overflow diagnostic")
    ten_t = mp_tempo.mp_tempo_post_run(result.ten_t,statein.prslk,control["ttendlim"])
    for name in ("ten_q","ten_t","ten_u","ten_v","prcp","rain","ice","snow","graupel","sr","refl_10cm"):
        torch._assert_async(torch.isfinite(getattr(result,name)).all(),f"nonfinite TEMPO output {name}")
    torch._assert_async(torch.isfinite(ten_t).all(),"nonfinite TEMPO limited temperature tendency")
    updated = replace(prefix,interstitial={**work,"ten_t":ten_t,
        "ten_u":result.ten_u,"ten_v":result.ten_v,"ten_q":result.ten_q,
        "prcpmp":result.prcp,"rainmp":result.rain,"icemp":result.ice,
        "snowmp":result.snow,"graupelmp":result.graupel},
        carry=replace(prefix.carry,intdiag={**diagnostics,"sr":result.sr,"refl_10cm":result.refl_10cm}))
    return updated,result


def _tempo_core(profiles,static):
    """The TEMPO core in one call, or over column chunks (runtime_policy.set_column_chunk).

    Profiles are column-first; the result's per-column fields join along columns, and its
    static inner-step count and timestep agree across chunks.
    """
    chunk = scheme_column_chunk("apply_tempo")
    if column_chunk() is None:
        return mp_tempo.mp_tempo_run(**profiles,**static)
    core = compiled(mp_tempo.mp_tempo_run) if scheme_compile_selected("apply_tempo") else None
    call = core or mp_tempo.mp_tempo_run
    if chunk is None:
        return call(**profiles, **static)
    return run_column_chunks(lambda columns: call(**columns,**static),profiles,
        ncol=profiles["tgrs"].shape[0],size=chunk,axis=0,views=True,
        # A whole chunk: a slice leaves nested sedimentation tiers dense, so their overflow
        # bits stay unregistered and the compiled core recompiles once they appear.
        warm=None if core is None else lambda columns: core.warm(**columns,**static),warm_columns=None)


def microphysics_post(fields, *, control):
    """Selected R4 GFS_MP_generic_post, including its persistent writes.

    Source-only mass ledgers are validation output, not numerical state.
    Active precip-type algorithms and radar assimilation are not this scope.
    """
    for name in ("imp_physics", "lsm", "lsm_noahmp", "nncl", "ntcw", "num_dfi_radar",
                 "num_diag_buckets", "index_of_temperature", "index_of_process_mp", "tend_opt_mp"):
        if type(control[name]) is not int:
            raise TypeError(f"microphysics post requires static {name}")
    for name in ("cal_pre", "exticeden", "progsigma", "cplflx", "cplchm", "cpllnd",
                 "lssav", "ldiag3d", "qdiag3d"):
        if type(control[name]) is not bool:
            raise TypeError(f"microphysics post requires static {name}")
    if (control["imp_physics"] != 88 or control["lsm"] != control["lsm_noahmp"] or
        control["cal_pre"] or control["exticeden"] or control["num_dfi_radar"] != 0):
        raise NotImplementedError("microphysics post selects R4 TEMPO/Noah-MP without precip-type/radar assimilation")
    f = fields
    q = f["gq0"]
    if q.ndim != 3 or q.shape[2] != 10 or control["ntcw"] != 2 or control["nncl"] != 5:
        raise ValueError("microphysics post requires the R4 five-water-condensate layout")
    for name,value in f.items():
        if value.device != q.device:
            raise ValueError(f"microphysics post {name} has a different device")
        if value.is_floating_point():
            if not working_like(value, q):
                raise TypeError(f"microphysics post {name} requires the working dtype {q.dtype}")
    assert_all_finite(f, "microphysics post", exclude=("dtend",))
    out = {**f, **photochemistry_post(f,dtp=f["dtp"],mode=control["tend_opt_mp"])}
    dtp,frain = f["dtp"],f["frain"]
    torch._assert_async((f["con_g"] > 0.) & (dtp > 0.),"invalid microphysics post gravity/timestep")
    onebg = 1./f["con_g"]
    out["rain"] = f["rainc"]+frain*f["rain1"]

    # The selected SAMF/TEMPO combination adds convective reflectivity even
    # when the TEMPO radar switch is off. First warm level is found top-down.
    for name in ("imfdeepcnv", "imfdeepcnv_samf", "imfdeepcnv_gf", "imfshalcnv", "imfshalcnv_gf"):
        if type(control[name]) is not int:
            raise TypeError(f"microphysics post requires static {name}")
    if (control["imfdeepcnv"] in (control["imfdeepcnv_samf"],control["imfdeepcnv_gf"]) or
        control["imfshalcnv"] == control["imfshalcnv_gf"]):
        zo = f["phil"]*onebg
        levels = torch.arange(1,q.shape[1]+1,device=q.device)
        warm = torch.where(out["gt0"] >= f["con_t0c"],levels,0).amax(dim=1)
        zfrz = zo.gather(1,(warm-1).clamp_min(0)[:,None]).squeeze(1)
        htop = f["htop"].to(torch.int64)
        torch._assert_async(((htop >= 0) & (htop <= q.shape[1])).all(),"invalid microphysics convective top")
        ztop = zo.gather(1,(htop-1).clamp_min(0)[:,None]).squeeze(1)
        convective = (f["rainc"] > 0.) & (htop > 0)
        factor = torch.where(convective,-2./torch.clamp_min(ztop-zfrz,1000.),0.)
        delz = zo-zfrz[:,None]
        fctz = torch.pow(10.,factor[:,None]*torch.where(delz < 0.,0.,delz))
        cuprate = torch.where(convective,f["rainc"]*3.6e6/dtp,1.)
        ze_conv = fctz*(300.*working_power(cuprate,1.4))[:,None]
        ze_mp = torch.pow(10.,.1*f["refl_10cm"])
        dbz_sum = torch.clamp_min(10.*torch.log10(ze_mp+ze_conv),-20.)
        active = convective[:,None] & (levels <= htop[:,None])
        out["refl_10cm"] = torch.where(active,dbz_sum,f["refl_10cm"])

    for name in ("frzr", "frzrb"):
        out[name] = f[name]+torch.where(out["gt0"][:,0] <= 273.,f["rain0"],0.)
    for name,increment in (("tsnowp","snow0"),("tsnowpb","snow0"),
                           ("frozr","graupel0"),("frozrb","graupel0")):
        out[name] = f[name]+f[increment]
    out["tprcp"] = torch.clamp_min(f["rainc"]+frain*f["rain1"],0.)
    for name in ("ice", "snow", "graupel"):
        out[name] = frain*f[name+"0"]
    for name,value in (("raincprv",f["rainc"]),("rainncprv",frain*f["rain1"]),
                       *((n+"prv",out[n]) for n in ("ice","snow","graupel"))):
        out[name] = value
    rate = 1./(dtp*.001)
    for name in ("raincprv","rainncprv","iceprv","snowprv","graupelprv"):
        out["d"+name] = rate*out[name]
    csnow = torch.where(f["tsfc"] >= 273.15,0.,f["rainc"])
    total_precip = (((f["snow0"]+f["ice0"])+f["graupel0"])+f["rain0"])+f["rainc"]
    wet = total_precip > f["rainmin"]
    out["srflag"] = torch.where(wet,
        ((f["snow0"]+f["ice0"])+f["graupel0"]+csnow)/torch.where(wet,total_precip,1.),0.)
    if control["lssav"]:
        for name,increment in (("cnvprcp","rainc"),("totprcp","rain"),
            ("totice","ice"),("totsnw","snow"),("totgrp","graupel"),
            ("toticeb","ice"),("totsnwb","snow"),("totgrpb","graupel")):
            out[name] = f[name]+out[increment]
        for name,increment in (("cnvprcpb","rainc"),("totprcpb","rain")):
            if f[name].shape != (q.shape[0],control["num_diag_buckets"]):
                raise ValueError(f"invalid microphysics bucket {name}")
            out[name] = f[name]+out[increment][:,None]
        if control["ldiag3d"]:
            dtidx,dtend = f["dtidx"],out["dtend"].clone()
            process = control["index_of_process_mp"]
            if not 1 <= process <= dtidx.shape[1]:
                raise ValueError("invalid microphysics diagnostic process")
            terms = [(control["index_of_temperature"],f["ten_t"])]
            if control["qdiag3d"]:
                terms.extend((101+n,f["ten_q"][:,:,n]) for n in range(q.shape[2]))
            for variable,tendency in terms:
                slot = dtidx[variable-1,process-1]
                torch._assert_async(slot <= dtend.shape[2],"microphysics diagnostic slot exceeds allocation")
                increment = tendency*dtp*frain
                add_diagnostic_(dtend,increment,slot)
            out["dtend"] = dtend
    if control["progsigma"]:
        out["dqdt_qmicro"],out["prevsq"] = f["ten_q"][:,:,0],out["gq0"][:,:,0]
    if control["cplflx"] or control["cplchm"] or control["cpllnd"]:
        out["dsnow_cpl"] = torch.clamp_min(out["rain"]*out["srflag"],0.)
        out["drain_cpl"] = torch.clamp_min(out["rain"]-out["dsnow_cpl"],0.)
        out["rain_cpl"] = f["rain_cpl"]+out["drain_cpl"]
        out["snow_cpl"] = f["snow_cpl"]+out["dsnow_cpl"]
    if control["cplchm"] or control["cpllnd"]:
        out["rainc_cpl"] = f["rainc_cpl"]+f["rainc"]
    pwat = torch.zeros_like(f["pwat"])
    # Condensate sums for all levels at once (same tracer order); the column sum stays ordered.
    work = torch.zeros_like(f["del"])
    for ic in range(control["ntcw"]-1,control["ntcw"]+control["nncl"]-1):
        work = work+out["gq0"][:,:,ic]
    terms = (f["del"]*(out["gq0"][:,:,0]+work)).t().contiguous()
    for k in range(q.shape[1]):
        pwat = pwat+terms[k]
    out["pwat"] = pwat*onebg
    assert_all_finite(out, "microphysics post output", exclude=("dtend",))
    return out


def apply_microphysics_post(statein, grid, prefix, *, dtp, frain, dtidx, control):
    """Bind the selected cap's live tendencies, precipitation and carry."""
    work,diag = prefix.interstitial,prefix.carry.intdiag
    state,sfc = prefix.stateout,prefix.sfcprop
    f = {**{n:getattr(state,n) for n in ("gt0","gu0","gv0","gq0")},
        **{n:work[n] for n in ("ten_t","ten_u","ten_v","ten_q","dtdt","dudt","dvdt","dqdt","del")},
        **{n:getattr(statein,n) for n in ("prsl","prsi","phii","phil")},
        **{n:diag[n] for n in ("rainc","rain","ice","snow","graupel","refl_10cm","sr",
            "frzr","frzrb","frozr","frozrb","tsnowp","tsnowpb","rhonewsn1","cnvprcp",
            "totprcp","totice","totsnw","totgrp","cnvprcpb","totprcpb","toticeb","totsnwb",
            "totgrpb","pwat","dtend")},
        **{n:getattr(sfc,n) for n in ("tsfc","tprcp","srflag","raincprv","rainncprv","iceprv",
            "snowprv","graupelprv","draincprv","drainncprv","diceprv","dsnowprv","dgraupelprv")},
        **{n:state.gt0.new_full((), v) for n,v in (("con_g",9.80665),("con_t0c",273.15),("rainmin",1.e-13))},
        "dtp":dtp,"frain":frain,"dtidx":dtidx,"htop":work["ktop"],
        "area":grid.area,"rain1":work["prcpmp"],"rain0":work["rainmp"],
        "ice0":work["icemp"],"snow0":work["snowmp"],"graupel0":work["graupelmp"]}
    for name,member in (("domr_diag","tdomr"),("domzr_diag","tdomzr"),
                        ("domip_diag","tdomip"),("doms_diag","tdoms")):
        f[name] = diag[member]
    for name in ("rain_cpl","rainc_cpl","snow_cpl","drain_cpl","dsnow_cpl","dqdt_qmicro","prevsq"):
        if name in prefix.carry.coupling:
            f[name] = prefix.carry.coupling[name]
    out = microphysics_post(f,control=control)
    result = replace(prefix,
        stateout=replace(state,**{n:out[n] for n in ("gt0","gu0","gv0","gq0")}),
        interstitial={**work,**{n:out[n] for n in ("dtdt","dudt","dvdt","dqdt")}},
        carry=replace(prefix.carry,
            intdiag={**diag,**{n:out[n] for n in diag if n in out}},
            coupling={**prefix.carry.coupling,**{n:out[n] for n in prefix.carry.coupling if n in out}}))
    updated_sfc = replace(sfc,**{n:out[n] for n in ("tprcp","srflag","raincprv","rainncprv","iceprv",
        "snowprv","graupelprv","draincprv","drainncprv","diceprv","dsnowprv","dgraupelprv")})
    return replace(result,sfcprop=updated_sfc)


@dataclass(frozen=True)
class MicrophysicsSequenceResult:
    prefix: PhysicsPrefixResult
    ud_mf: torch.Tensor
    cloud_history: ConvectiveCloudHistory


def run_microphysics_sequence(statein, grid, prefix, *, hpbl, ud_mf, prevsq,
    cloud_history, clstp, dtp, dtf, dtidx, coefficients, convection_control,
    cloud_control, tables, min_layer_depth, tempo_control, frain, post_control,
    kdt, avg_max_length, is_photochem, tail_control):
    """Run the selected R4 cap after prepare_microphysics_prefix.

    The caller owns projection/ozone preparation and the final native commit.
    Persistent updraft and cloud history enter and leave this sequence unchanged
    except for their source updates. No cold initialization occurs here.
    """
    if type(tempo_control["dtp"]) not in (int, float):
        raise TypeError("microphysics sequence requires a static TEMPO timestep")
    torch._assert_async(dtp == tempo_control["dtp"],
        "microphysics sequence and TEMPO timesteps differ")
    deep = scheme(apply_convection)(statein, grid, prefix, hpbl=hpbl, ud_mf=ud_mf,
        prevsq=prevsq, dtp=dtp, dtf=dtf, dtidx=dtidx, coefficients=coefficients,
        control=convection_control)
    shallow = scheme(apply_convection)(statein, grid, deep.prefix, hpbl=hpbl,
        ud_mf=deep.ud_mf, prevsq=prevsq, dtp=dtp, dtf=dtf, dtidx=dtidx,
        coefficients=coefficients, control=convection_control, shallow=True)
    clouds, history = prepare_convective_clouds(statein, shallow.prefix,
        cloud_history, clstp=clstp, dtidx=dtidx, control=cloud_control)
    # With column chunks the TEMPO core per chunk is the compiled region instead.
    tempo_entry = apply_tempo if column_chunk() is not None else scheme(apply_tempo)
    tempo, _ = tempo_entry(statein, clouds, tables=tables,
        min_layer_depth=min_layer_depth, control=tempo_control)
    post = glue(apply_microphysics_post)(statein, grid, tempo, dtp=dtp, frain=frain,
        dtidx=dtidx, control=post_control)
    final = finish_microphysics(statein, post, dtp=dtp, kdt=kdt,
        avg_max_length=avg_max_length, dtidx=dtidx, is_photochem=is_photochem,
        control=tail_control)
    return MicrophysicsSequenceResult(final, shallow.ud_mf, history)
