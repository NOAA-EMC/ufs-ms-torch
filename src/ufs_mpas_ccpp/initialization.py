# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Source-defined R4 controls and cold MPAS bridge from initialized buffers."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
from gfs_surface_layer_torch.cases import SurfaceCaseConfig
from mpas_pytorch.constants import UFS_CONSTANTS
from rrtmgp_radiation_torch.radiation_group import RadiationConfig
from rrtmgp_radiation_torch.setup import _solar_state
from sa_sas_deep_convection_torch import constants as convection_constants
from sa_sas_deep_convection_torch.samf_interstitials import (
    INDEX_NAMES, IMP_PHYSICS_CODES, mpas_tempo_layout,
)

from .clock import _nint_nonnegative
from .containers import NativeAtmosphere, ProjectionMesh, Statein, Stateout
from .grid import Grid, GridProjection, project_grid
from .distributed import global_rows
from .ledger import _ordered_totals
from .nsst import NSSTControl
from .physics_prefix import PhysicsPrefixControl
from .precision import WORKING_DTYPES
from .projection import project_atmosphere, r4_tracer_contract
from .radiation_scaling import RadiationScalingControl
from .refresh import OutputDiagnostics, ReconstructionMesh, refresh_host_diagnostics
from .surface import initialize_surface
from .surface_composites import SurfaceCompositesControl
from .surface_containers import NativeSurface, Sfcprop, SoilParameters, SurfaceControl
from .surface_post import SurfacePostControl


@dataclass(frozen=True)
class SurfaceSchemeControls:
    composites: SurfaceCompositesControl
    surface_layer: SurfaceCaseConfig
    nsst: NSSTControl
    post: SurfacePostControl
    noah_options: Mapping[str, Any]


def _static_namelist_value(namelist, name, default):
    result = namelist.get(name, default)
    if type(default) in (bool, int):
        if type(result) is not type(default):
            raise TypeError(f"physics namelist {name} must be a static {type(default).__name__}")
    elif type(result) not in (float, int) or not math.isfinite(result):
        raise ValueError(f"physics namelist {name} must be a finite static number")
    return result


def prepare_r4_surface_controls(
    physics_namelist: Mapping[str, Any], *, timestep: float
) -> SurfaceSchemeControls:
    """Selected Noah-MP/NSST setup (nonfractional or frac_grid) from static namelist values.

    Defaults and derived identifiers follow GFS_typedefs control_initialize.
    GFS_time_vary_pre sets lssav=true before every surface invocation.
    Read the namelist outside numerics. Calendar values stay live call inputs.
    This does not initialize tables, forcing, atmospheric controls or dtidx.
    """
    if type(timestep) not in (int, float) or not math.isfinite(timestep) or timestep <= 0:
        raise ValueError("surface setup requires a positive static timestep")

    def value(name, default):
        return _static_namelist_value(physics_namelist, name, default)

    lsm = value("lsm", 1)
    ivegsrc = value("ivegsrc", 2)
    isot = value("isot", 0)
    if (lsm, ivegsrc, isot) != (2, 1, 1):
        raise NotImplementedError("R4 surface setup requires Noah-MP, IGBP and STATSGO")
    if value("lsoil", 4) != 4 or value("lsoil_lsm", -1) not in (-1, 4) or value("lsnow_lsm", 3) != 3:
        raise NotImplementedError("R4 surface setup requires four soil and three snow layers")
    common = {n: value(n, False) for n in (
        "frac_grid", "cplflx", "cplice", "cplwav2atm", "cpl_fire", "cplaqm",
        "cplchm", "cplcat", "cplwav", "cpllnd", "cpllnd2atm", "use_cdeps_inline")}
    common["cplchm"] = common["cplchm"] or common["cplaqm"] or common["cplcat"]
    if common["cpllnd2atm"]:
        raise NotImplementedError("R4 surface setup does not support mediator land coupling")
    composites = SurfaceCompositesControl(
        **{n: common[n] for n in ("frac_grid", "cplflx", "cplice")},
        use_cdeps_data=common["use_cdeps_inline"], lkm=value("lkm", 0), lsm=lsm, lsm_ruc=3,
        min_seaice=value("min_seaice", 1.e-11), min_lakeice=value("min_lakeice", .15),
        huge=value("huge", 9.9692099683868690e36))
    surface_layer = SurfaceCaseConfig(
        case_id="r4_native_prefix", default_real_bytes=8, ivegsrc=ivegsrc,
        sfc_z0_type=value("sfc_z0_type", 0), lsm=lsm, lsm_ruc=3,
        iopt_lake=value("iopt_lake", 2), iopt_lake_clm=2,
        **{n: value(n, False) for n in ("redrag", "use_oceanuv", "diag_flux", "diag_log", "use_lake2m")},
        thsfc_loc=value("thsfc_loc", True))
    nstf_name = physics_namelist.get("nstf_name", (0, 0, 1, 0, 5))
    if type(nstf_name) not in (list, tuple) or any(type(v) is not int for v in nstf_name):
        raise TypeError("surface namelist nstf_name must contain static integers")
    nsst = NSSTControl(tuple(nstf_name), timestep,
        use_oceanuv=surface_layer.use_oceanuv, thsfc_loc=surface_layer.thsfc_loc,
        lseaspray=value("lseaspray", False), cplflx=common["cplflx"])
    post = SurfacePostControl(
        **{n: common[n] for n in ("frac_grid", "cplflx", "cplwav2atm", "cpl_fire",
            "cplaqm", "cplchm", "cplcat", "cplwav", "cpllnd")},
        use_cdeps_data=common["use_cdeps_inline"], lkm=composites.lkm, lsm=lsm, lsm_noahmp=2,
        opt_diag=value("iopt_diag", 2), lssav=True,
        lheatstrg=value("lheatstrg", False), h0facu=value("h0facu", .25), h0facs=value("h0facs", 1.))
    noah_options = {n: value(n, default) for n, default in {
        "iopt_crs": 1, "iopt_btr": 1, "iopt_run": 3, "iopt_sfc": 1,
        "iopt_frz": 1, "iopt_inf": 1, "iopt_rad": 3, "iopt_alb": 2,
        "iopt_snf": 1, "iopt_tbot": 2, "iopt_stc": 1, "iopt_trs": 2, "iopt_diag": 2,
        "do_mynnsfclay": False, "do_mynnedmf": False,
    }.items()}
    noah_options.update(idveg=value("iopt_dveg", 4), delt=timestep,
        cpllnd=common["cpllnd"], cpllnd2atm=common["cpllnd2atm"], thsfc_loc=surface_layer.thsfc_loc)
    return SurfaceSchemeControls(composites, surface_layer, nsst, post, noah_options)


@dataclass(frozen=True)
class PhysicsSchemeControls:
    surface: SurfaceSchemeControls
    prefix: PhysicsPrefixControl
    pbl: Mapping[str, Any]
    dtp: torch.Tensor
    dtf: torch.Tensor
    nphy_f2d: int


@dataclass(frozen=True)
class DiagnosticIndices:
    dtidx: torch.Tensor
    ndtend: int


def prepare_r4_diagnostic_indices(physics_namelist: Mapping[str, Any], *,
    frozen_chemistry_rows: tuple[int, ...], device: torch.device | str) -> DiagnosticIndices:
    """Follow GFS_typedefs fill_dtidx call order for the selected R4 layout.

    The source leaves chemistry indices unset when ntchm=0, but consumes them
    during allocation. Require its frozen positive rows explicitly; these are
    not portable chemistry defaults. No captured dtidx supplies the other rows.
    """
    def value(name, default):
        return _static_namelist_value(physics_namelist, name, default)
    if (value("imp_physics", 99) != 88 or not value("satmedmf", False)
        or value("imfdeepcnv", 1) != 2 or value("imfshalcnv", 1) != 2
        or value("ldiag3d", False) != value("qdiag3d", False)
        or value("tend_opt_deep_conv", 1) not in (1, 3)
        or value("tend_opt_shal_conv", 1) not in (1, 3)
        or any(value(n, False) for n in ("cplchm", "cplaqm", "do_mynnedmf", "do_shoc"))):
        raise NotImplementedError("diagnostic setup selects R4 SAMF/TEMPO/SATMEDMF with ldiag3d = qdiag3d")
    if not value("ldiag3d", False):
        # GFS_typedefs fills dtidx only with ldiag3d: every slot stays -99 and nothing is allocated.
        return DiagnosticIndices(torch.full((110, 18), -99, dtype=torch.int32, device=device), 0)
    if (not isinstance(frozen_chemistry_rows, tuple)
        or any(type(row) is not int for row in frozen_chemistry_rows)
        or frozen_chemistry_rows != (100,)):
        raise ValueError("diagnostic setup requires explicit frozen chemistry row (100,)")
    selection = physics_namelist.get("dtend_select", "*")
    if isinstance(selection, str):
        selection = [selection]
    if (not isinstance(selection, (list, tuple))
        or any(type(pattern) is not str for pattern in selection)):
        raise TypeError("diagnostic selection requires static string patterns")
    if (not selection or selection[0] != "*"
        or any(s != " " and s != "" for s in selection[1:])):
        raise NotImplementedError("R4 diagnostic setup selects all source tendency fields")
    if value("num_dfi_radar", 0) != 0:
        raise NotImplementedError("R4 diagnostic setup excludes DFI radar")
    rayleigh = not (value("lsidea", False) or value("ral_ts", 0.) <= 0.
                    or value("prslrd0", 0.) == 0.)
    slots = {}

    def fill(row, process, enabled=True):
        if enabled and (row, process) not in slots:
            slots[row, process] = len(slots) + 1

    # GFS_typedefs:5788-5816: temperature, then interleaved wind components.
    for process in (9, 10, 1, 2, 3, 4, 11, 12, 13, 16, 17):
        fill(10, process, process != 12 or rayleigh)
    for process in (1, 11, 2, 13, 12, 3, 16, 17):
        for row in (11, 12):
            fill(row, process, process != 12 or rayleigh)
    fill(101, 3)
    fill(101, 2)
    # Both SAMF generic tendency flags are true in the selected composition.
    for tracer in range(2, 11):
        if tracer in (7, 9, 10):
            fill(100 + tracer, 3)
            fill(100 + tracer, 2)
        else:
            fill(100 + tracer, 14)
        if tracer in (7, 8):
            fill(100 + tracer, 14)
    for row in frozen_chemistry_rows:
        fill(row, 1)
    ozone = value("oz_phys", False) or value("oz_phys_2015", True)
    for process in (1, 5, 6, 7, 8, 18, 16, 17):
        fill(110, process, process in (1, 16, 17) or ozone)
    for process in (5, 6, 18):
        fill(101, process, value("h2o_phys", False))
    for tracer in range(1, 10):
        for process in (1, 16, 17, 4):
            fill(100 + tracer, process, process != 4 or tracer != 9)
    dtidx = torch.tensor([[slots.get((row, process), -99) for process in range(1, 19)]
        for row in range(1, 111)], dtype=torch.int32, device=device)
    return DiagnosticIndices(dtidx, len(slots))


def prepare_r4_physics_controls(physics_namelist: Mapping[str, Any], *,
    dt_physics: float, dt_dycore: float, device: torch.device | str) -> PhysicsSchemeControls:
    """Defined R4 surface/prefix/PBL setup, without captured host containers.

    GFS_typedefs supplies defaults, indices and source-ordered area scaling.
    The selected TEMPO allocation has one phy_f2d field and no SHOC fields.
    This does not decide uninitialized dtidx chemistry or radiation-cloud controls.
    """
    for timestep in (dt_physics, dt_dycore):
        if type(timestep) not in (int, float) or not math.isfinite(timestep) or timestep <= 0:
            raise ValueError("physics setup requires positive static timesteps")
    def value(name, default):
        return _static_namelist_value(physics_namelist, name, default)
    if (value("imp_physics", 99) != 88 or not value("satmedmf", False)
        or value("isatmedmf", 0) != 1 or value("do_mynnedmf", False)
        or value("do_shoc", False) or value("hybedmf", False)):
        raise NotImplementedError("R4 physics setup requires TEMPO and SATMEDMF isatmedmf=1")
    surface = prepare_r4_surface_controls(physics_namelist, timestep=dt_dycore)
    dimensions = tuple(value(n, default) for n, default in (
        ("max_lon", 5000), ("max_lat", 2000), ("min_lon", 192), ("min_lat", 94)))
    if any(v <= 0 for v in dimensions):
        raise ValueError("physics area scaling requires positive static dimensions")
    max_lon, max_lat, min_lon, min_lat = dimensions
    radius = torch.tensor(UFS_CONSTANTS.EARTH_RADIUS, dtype=torch.float64, device=device)
    area = radius * radius * (UFS_CONSTANTS.PI + UFS_CONSTANTS.PI) * UFS_CONSTANTS.PI
    dxmax = torch.log(area / (max_lon * max_lat))
    dxmin = torch.log(area / (min_lon * min_lat))
    torch._assert_async(dxmax != dxmin, "degenerate physics area scaling")
    dxinv = 1. / (dxmax - dxmin)
    radiation = RadiationScalingControl(
        dtf=dt_dycore, dtp=dt_physics, fhswr=value("fhswr", 3600.),
        tend_opt_rad_scaler=value("tend_opt_rad_scaler", 2), lssav=True,
        **{n: value(n, False) for n in ("ldiag3d", "lsidea", "use_LW_jacobian",
            "use_med_flux", "do_sppt", "ca_global")}, pert_radtend=value("pert_radtend", True),
        index_of_temperature=10, index_of_process_longwave=9, index_of_process_shortwave=10)
    prefix = PhysicsPrefixControl(surface.composites, radiation, surface.surface_layer,
        isot=value("isot", 0), ivegsrc=surface.surface_layer.ivegsrc, ncnvwind=1,
        lndp_type=value("lndp_type", 0), dxmin=dxmin, dxinv=dxinv,
        **{n: value(n, False) for n in ("shal_cnv", "old_monin", "mstrat", "do_shoc")},
        imfshalcnv=value("imfshalcnv", 1))
    pbl = {n: value(n, default) for n, default in {
        "tte_edmf": False, "dspheat": False, "sa3dtke": False, "use_oceanuv": False,
        "xkzm_m": 1., "xkzm_h": 1., "xkzm_s": 1., "dspfac": 1.,
        "bl_upfr": .13, "bl_dnfr": .1, "rlmx": 300., "elmx": 300.,
        "do_canopy": False, "cplaqm": False, "sfc_rlm": 0, "tc_pbl": 0, "use_lpt": 0,
        "ldiag3d": False, "tend_opt_pbl": 2, "cplflx": False, "cplcat": False, "lsidea": False,
    }.items()}
    pbl.update(cplchm=surface.post.cplchm, lssav=True, gen_tend=False, source_default_real_bytes=8,
        index_of_temperature=10, index_of_x_wind=11, index_of_y_wind=12, index_of_process_pbl=1)
    return PhysicsSchemeControls(surface, prefix, pbl,
        radius.new_tensor(dt_physics), radius.new_tensor(dt_dycore), 1)


@dataclass(frozen=True)
class ReferenceGeometry:
    ak: torch.Tensor
    bk: torch.Tensor
    lonr: torch.Tensor
    global_area: torch.Tensor
    mean_area: torch.Tensor
    equivalent_spacing: torch.Tensor


@dataclass(frozen=True)
class ConvectionSchemeControls:
    control: Mapping[str, Any]
    coefficients: Mapping[str, torch.Tensor]


def prepare_r4_convection_controls(physics_namelist: Mapping[str, Any], *,
    dt_physics: float, dt_dycore: float, ncol: int, device: torch.device | str,
    frozen_unassigned_indices: Mapping[str, int], step_index: int = 1) -> ConvectionSchemeControls:
    """Selected nonrestart SAMF core/post setup without captured containers.

    Reuse the frozen R4 tracer contract and source packing rule. Coefficients live on the numerical
    device; step flags use the static scheduler index checked by the device clock.
    Five cap indices are unassigned on the source MPAS/TEMPO path.
    Require the frozen-build zeros explicitly, not as portable source defaults.
    Source setup also reads unset ntccn/ntccna; the frozen packing contract
    qualifies its resulting transport flags, not those members' initialization.
    This does not prepare dtidx, ozone forcing, cloud or TEMPO-tail controls.
    """
    for timestep in (dt_physics,dt_dycore):
        if type(timestep) not in (int,float) or not math.isfinite(timestep) or timestep <= 0:
            raise ValueError("convection setup requires positive static timesteps")
    if type(ncol) is not int or ncol < 1 or type(step_index) is not int or step_index < 1:
        raise ValueError("convection setup requires positive static ncol/step_index")
    def value(name,default):
        return _static_namelist_value(physics_namelist,name,default)
    flags = {n:value(n,False) for n in (
        "satmedmf","trans_trac","ldiag3d","qdiag3d","ras","cscnv","do_shoc",
        "do_mynnedmf","progsigma","progomega","do_ca","ca_closure","ca_entr",
        "ca_trigger","hwrf_samfdeep","hwrf_samfshal","sigmab_coldstart",
        "cplchm","cplaqm","cplcat","cpl_fire","rrfs_sd")}
    if (value("imp_physics",99) != 88 or value("imfdeepcnv",1) != 2
        or value("imfshalcnv",1) != 2 or not value("shal_cnv",False)
        or not flags["satmedmf"] or any(flags[n] for n in
            ("ras","cscnv","do_shoc","do_mynnedmf","progsigma","progomega","do_ca",
             "cplchm","cplaqm","cplcat","cpl_fire","rrfs_sd"))
        or value("pdfcld",False) or not value("cnvcld",False)):
        raise NotImplementedError("R4 convection setup requires selected deterministic SAMF/TEMPO/SATMEDMF")
    if flags["qdiag3d"] and not flags["ldiag3d"]:
        raise ValueError("convection qdiag3d requires ldiag3d")
    unassigned = ("ntclamt","nthv","ntrz","ntgz","nthz")
    if not isinstance(frozen_unassigned_indices, Mapping) or set(frozen_unassigned_indices) != set(unassigned):
        raise ValueError("convection setup requires the five consumed frozen unassigned indices")
    for name in unassigned:
        index = frozen_unassigned_indices[name]
        if type(index) is not int:
            raise TypeError(f"convection setup requires static integer {name}")
        if index != 0:
            raise NotImplementedError(f"R4 frozen convection layout requires {name}=0")
    layout = mpas_tempo_layout(ntc=0)
    values = {n:value(n,default) for n,default in {
        "clam_deep":.1,"c0s_deep":.002,"c1_deep":.002,"betal_deep":.05,"betas_deep":.05,
        "clam_shal":.3,"c0s_shal":.002,"c1_shal":.0005,
        "pgcon_deep":.55,"asolfac_deep":.958,"pgcon_shal":.55,"asolfac_shal":.958,
        "evef":.09,"cscale":1.,"betascu":8.,"betamcu":1.,"betadcu":2.,
        "nthresh":18.,"cat_adj_deep":1.,"cat_adj_shal":1.,
    }.items()}
    values.update({n:getattr(convection_constants,n.upper()) for n in (
        "cliq","cp","cvap","eps","epsm1","fv","grav","hvap","rd","rv","t0c")})
    values.update(delt=dt_physics,dtf=dt_dycore,frain=dt_dycore/dt_physics,
        con_g=convection_constants.GRAV,con_cp=convection_constants.CP)
    # The original merged cap mapping retains shallow aliases; apply_convection
    # replaces these with the selected deep/shallow coefficients for each call.
    values.update({n:values[n+"_shal"] for n in ("clam","c0s","c1","pgcon","asolfac")})
    values.update({n:values[n+"_deep"] for n in ("betal","betas")})
    modes = {n:value(n,1) for n in ("tend_opt_deep_conv","tend_opt_shal_conv")}
    if any(mode not in (1,3) for mode in modes.values()):
        raise NotImplementedError("SAMF water reconciliation requires applied Stateout modes1/3")
    control = dict({**flags,**values,**modes,
        **{n:layout.index(n) for n in INDEX_NAMES if n not in ("ntccn","ntccna")},
        **frozen_unassigned_indices,
        **{"imp_physics_"+n:code for n,code in IMP_PHYSICS_CODES.items()}},
        im=ncol,km=55,levs=55,kdt=step_index,first_time_step=(step_index==1),restart=False,
        imp_physics=88,mp_phys=88,mp_phys_mg=IMP_PHYSICS_CODES["mg"],
        ntrac=layout.ntrac,nn=layout.nn,ntr=layout.ntr,ntk=layout.ntk,itc=layout.itc,
        ntc=0,ntchm=0,nsamftrac=layout.ntr,tracers_total=layout.tracers_total,
        ncnd=5,ncloud=5,npdf3d=0,num_p3d=3,
        ncnvcld3d=1,ncnvc=-999,ncnvw=4,
        shcnvcw=False,lssav=True,samf_water_reconciliation=True,
        index_of_temperature=10,index_of_x_wind=11,index_of_y_wind=12,
        index_of_process_dcnv=2,index_of_process_scnv=3,
        flag_for_dcnv_generic_tend=True,flag_for_scnv_generic_tend=True,
        imfshalcnv=2,imfshalcnv_sas=1,imfshalcnv_samf=2,
        tend_opt_dcnv=modes["tend_opt_deep_conv"],tend_opt_scnv=modes["tend_opt_shal_conv"])
    coefficients = {n:torch.tensor(v,dtype=torch.float64,device=device) for n,v in values.items()}
    coefficients["fscav"] = torch.empty(0,dtype=torch.float64,device=device)
    return ConvectionSchemeControls(control,coefficients)


@dataclass(frozen=True)
class MicrophysicsSchemeControls:
    convection: ConvectionSchemeControls
    photochemistry: Mapping[str, Any]
    crtrh: torch.Tensor
    rhcmax: torch.Tensor
    cloud: Mapping[str, Any]
    tempo: Mapping[str, Any]
    post: Mapping[str, Any]
    tail: Mapping[str, Any]
    frain: torch.Tensor
    avg_max_length: torch.Tensor
    is_photochem: torch.Tensor


def prepare_r4_microphysics_controls(physics_namelist: Mapping[str, Any], *,
    dt_physics: float, dt_dycore: float, ncol: int, device: torch.device | str,
    frozen_unassigned_indices: Mapping[str, int], tempo_initialized: bool,
    step_index: int = 1) -> MicrophysicsSchemeControls:
    """Selected ozone/SAMF/cloud/TEMPO/post controls, not table or state initialization.

    The caller must complete TEMPO preparation before setting tempo_initialized.
    The frozen dtidx remains a separate input. Source-declared rows and chemistry
    process flags do not need captured buffers or host reads from device tensors.
    """
    unset = ("ntclamt","nthv","ntrz","ntgz","nthz","ntccn")
    if not isinstance(frozen_unassigned_indices, Mapping) or set(frozen_unassigned_indices) != set(unset):
        raise ValueError("microphysics setup requires six frozen unassigned cap indices")
    if type(frozen_unassigned_indices["ntccn"]) is not int:
        raise TypeError("microphysics setup requires static integer ntccn")
    if frozen_unassigned_indices["ntccn"] != 0:
        raise NotImplementedError("R4 frozen microphysics layout requires ntccn=0")
    if type(tempo_initialized) is not bool:
        raise TypeError("TEMPO preparation status must be static")
    if not tempo_initialized:
        raise ValueError("complete TEMPO preparation before microphysics setup")
    convection = prepare_r4_convection_controls(physics_namelist,
        dt_physics=dt_physics,dt_dycore=dt_dycore,ncol=ncol,device=device,step_index=step_index,
        frozen_unassigned_indices={n:frozen_unassigned_indices[n] for n in unset[:-1]})
    c = convection.control
    def value(name,default):
        return _static_namelist_value(physics_namelist,name,default)
    flags = {n:value(n,default) for n,default in {
        "ltaerosol":False,"lthailaware":False,"lradar":False,"lightning_threat":False,
        "cal_pre":False,"exticeden":False,"cplflx":False,"cpllnd":False,
        "do_sppt":False,"pert_mp":False,"use_zmtnblck":False,"do_shum":False,
        "do_skeb":False,"ca_global":False,"oz_phys_2015":True,"h2o_phys":False,
    }.items()}
    if (any(flags[n] for n in ("ltaerosol","lthailaware","lradar","lightning_threat",
        "cal_pre","exticeden","cplflx","cpllnd","do_sppt","do_shum","do_skeb"))
        or value("lsm",1) != 2 or value("lkm",0) != 0
        or value("iopt_lake",2) != 2 or value("oz_phys",False)
        or not flags["oz_phys_2015"]):
        raise NotImplementedError("microphysics setup selects R4 TEMPO/Noah-MP with inactive optional schemes")
    radar = physics_namelist.get("fh_dfi_radar",(-2.e10,)*5)
    if type(radar) not in (list,tuple) or len(radar) != 5:
        raise TypeError("radar intervals require five static endpoints")
    radar = tuple(_static_namelist_value({"fh_dfi_radar":v},"fh_dfi_radar",-2.e10) for v in radar)
    if any(a > -1.e10 and b > -1.e10 for a,b in zip(radar,radar[1:])):
        raise NotImplementedError("R4 microphysics does not activate radar-assimilation intervals")
    avg = value("avg_max_length",3600.)
    ratio = avg/dt_physics
    integral = math.floor(ratio)
    hourly_steps = integral + int(ratio-integral >= .5)
    if avg <= 0. or hourly_steps < 1:
        raise ValueError("hourly diagnostic interval rounds to zero")
    inner = value("dt_inner",-999.)
    inner = inner if inner > 0. else dt_physics
    # convert_dry_rho is a declared member initializer, not a namelist option.
    tempo = dict(imp_physics=88,ntrac=10,tempo_call_index=step_index,
        dtp=dt_physics,dt_inner=inner,ttendlim=value("ttendlim",-999.),
        ltaerosol=False,lthailaware=False,lradar=False,convert_dry_rho=True,
        semi_sedi=value("sedi_semi",False),tempo_mp_is_init=tempo_initialized)
    imp = {"imp_physics_"+n:code for n,code in IMP_PHYSICS_CODES.items()}
    indices = {n:c[n] for n in ("ntcw","ntiw","ntrw","ntsw","ntgl","ntrnc",
        "ntsnc","ntgnc","ntlnc","ntinc","ntoz","ntke","ntqv")}
    common = dict(im=ncol,levs=55,imp_physics=88,ntrac=10,**imp,**indices,
        ldiag3d=c["ldiag3d"],qdiag3d=c["qdiag3d"])
    mode = value("tend_opt_photochem",1)
    if mode not in (1,2,3,4):
        raise ValueError("invalid photochemistry tendency mode")
    humidity = physics_namelist.get("crtrh",(.90,.90,.90))
    if type(humidity) not in (list,tuple) or len(humidity) != 3:
        raise TypeError("critical humidity requires three static values")
    humidity = tuple(_static_namelist_value({"crtrh":v},"crtrh",.90) for v in humidity)
    maximum = value("rhcmax",.9999999)
    photochemistry = {**c,**common,"oz_phys_2015":True,"oz_phys":False,"h2o_phys":flags["h2o_phys"],
        "ltaerosol":False,"dtp":dt_physics,"tend_opt_photochem":mode,
        "imfdeepcnv":2,"imfdeepcnv_samf":2,"imfdeepcnv_c3":5,"imfshalcnv_c3":5}
    cloud = {**common,"ltaerosol":False,"tracers_total":c["tracers_total"],
        "ntclamt":frozen_unassigned_indices["ntclamt"],"ntccn":0,"nn":c["nn"],
        "ntk":c["ntk"],"convert_dry_rho":True,"dtf":dt_dycore,
        "con_pi":UFS_CONSTANTS.PI,"con_rd":convection_constants.RD,"con_eps":convection_constants.EPS,
        "nssl_ccn_on":value("nssl_ccn_on",True),"nssl_invertccn":value("nssl_invertccn",True),
        "nssl_cccn":value("nssl_cccn",.6e9),
        "index_of_process_conv_trans":14,"index_of_process_dcnv":2,"index_of_process_scnv":3}
    post = {**common,**{n:flags[n] for n in ("cal_pre","exticeden","cplflx","cpllnd")},
        "cplchm":False,"progsigma":False,"kdt":step_index,"tend_opt_mp":value("tend_opt_mp",1),
        "rainmin":1.e-13,"rhowater":1000.,"nrcm":2,"nncl":5,
        "con_g":convection_constants.GRAV,"con_cp":convection_constants.CP,"con_t0c":273.15,
        "dtf":dt_dycore,"dtp":dt_physics,"frain":dt_dycore/dt_physics,
        "imfshalcnv":2,"imfshalcnv_gf":3,"imfdeepcnv":2,"imfdeepcnv_gf":3,"imfdeepcnv_samf":2,
        "lsm":2,"lsm_ruc":3,"lsm_noahmp":2,"lkm":0,"iopt_lake":2,"iopt_lake_clm":2,
        "num_diag_buckets":1,"num_dfi_radar":0,"fhour":step_index*dt_physics/3600.,
        "index_of_process_dfi_radar":15,"index_of_temperature":10,"index_of_process_mp":4,"lssav":True}
    if post["tend_opt_mp"] not in (1,2,3,4):
        raise ValueError("invalid microphysics tendency mode")
    rows = (10,11,12)+tuple(range(101,111)) if c["qdiag3d"] else (10,11,12)
    tail = {**common,**flags,"km":55,"ncol":ncol,"nlev":55,"kdt":step_index,
        "reset":(step_index-1)%hourly_steps==0,"num_diag_buckets":1,
        "delt":dt_physics,"dtp":dt_physics,"con_g":convection_constants.GRAV,"con_rd":convection_constants.RD,
        "do_ca":False,"ntracp100":110,"nprocess":18,"nprocess_summed":15,
        "ip_physics":16,"ip_photochem":18,"ip_prod_loss":5,"ip_ozmix":6,"ip_temp":7,
        "ip_overhead_ozone":8,"physics_variables":rows if c["ldiag3d"] else ()}
    is_photochem = torch.zeros(18,dtype=torch.bool,device=device)
    is_photochem[4:8] = True
    return MicrophysicsSchemeControls(convection,photochemistry,
        torch.tensor(humidity,dtype=torch.float64,device=device),
        torch.tensor(maximum,dtype=torch.float64,device=device),cloud,tempo,post,tail,
        torch.tensor(dt_dycore/dt_physics,dtype=torch.float64,device=device),
        torch.tensor(avg,dtype=torch.float64,device=device),is_photochem)


V17_DATA_YEARS = 2  # the initial year and the next: a run may cross one new year


def prepare_r4_radiation_config(physics_namelist: Mapping[str, Any], *,
    dt_physics: float, dt_dycore: float, initial_year: int, initial_month: int,
    jdat: torch.Tensor, solhr: torch.Tensor, frozen_smearclds: bool,
    input_dir=None, grid: Grid | None = None, aer_jdat: torch.Tensor | None = None,
    forecast_hours: float = 384.0) -> RadiationConfig:
    """Defined selected radiation controls with live device calendar inputs.

    GFS_typedefs leaves doGP_smearclds unassigned on the implicit-cloud path.
    Require its frozen-build value explicitly; never invent a portable default.
    This does not prepare gas/cloud tables or select a different source oracle.

    The GFS v17 options read data files in ``input_dir`` (the run folder, the names the Fortran
    opens) during this host preparation: ``isol = 2`` (``solarconstant_noaa_an.txt``),
    ``ico2 = 2`` (``co2historicaldata_YYYY.txt``) for the initial year and the next, and
    ``iaer = 1011`` (MERRA-2: ``aeroclim.mMM.nc`` for the months the run of ``forecast_hours``
    reads, interpolated to the columns of ``grid``, and ``optics_*.dat``). ``aer_jdat`` is the
    date at the step's ``fhour`` that ``GFS_phys_time_vary`` uses for the aerosol interpolation.
    """
    for timestep in (dt_physics, dt_dycore):
        if type(timestep) not in (int, float) or not math.isfinite(timestep) or timestep <= 0:
            raise ValueError("radiation setup requires positive static timesteps")
    if (type(initial_year) is not int or not 1583 <= initial_year <= 9999
        or type(initial_month) is not int or not 1 <= initial_month <= 12):
        raise ValueError("radiation setup requires a static initial year/month")
    if type(frozen_smearclds) is not bool:
        raise TypeError("radiation setup requires explicit static frozen_smearclds")
    if (not isinstance(jdat,torch.Tensor) or jdat.shape != (8,)
        or jdat.dtype not in (torch.int32,torch.int64)
        or not isinstance(solhr,torch.Tensor) or solhr.shape != ()
        or solhr.dtype != torch.float64 or solhr.device != jdat.device):
        raise ValueError("radiation calendar requires device-local date and float64 solar hour")
    torch._assert_async(torch.isfinite(solhr) & (solhr >= 0.) & (solhr < 24.),
        "invalid radiation solar hour")
    def value(name, default):
        return _static_namelist_value(physics_namelist,name.lower(),default)
    if (not value("do_rrtmgp",False) or value("imp_physics",99) != 88
        or not value("effr_in",False) or value("lsm",1) != 2
        or value("isol",0) not in (0, 2) or value("ictm",1) != 1 or value("ico2",0) not in (0, 2)
        or value("do_gpsw_glw",False) or value("do_mynnedmf",False)
        or value("dogp_sgs_cnv",False) or value("dogp_sgs_mynn",False)
        or (value("levr",-99) >= 0 and value("levr",-99) != 55)):
        raise NotImplementedError("R4 radiation setup requires selected TEMPO/Noah-MP implicit-cloud RRTMGP")
    dimensions = {n:value(n,-999) for n in (
        "rrtmgp_ngptssw","rrtmgp_ngptslw","rrtmgp_nbandssw","rrtmgp_nbandslw")}
    if tuple(dimensions.values()) != (112,128,14,16):
        raise NotImplementedError("R4 radiation setup requires the selected spectral tables")
    gases = physics_namelist.get("active_gases","")
    if type(gases) is not str:
        raise TypeError("radiation setup requires static active_gases")
    if value("ngases",-999) != 6 or gases != "h2o_co2_o3_n2o_ch4_o2":
        raise NotImplementedError("R4 radiation setup requires the selected six gases")
    for name, selected in {
        "lw_file_gas":"rrtmgp-data-lw-g128-210809.nc",
        "sw_file_gas":"rrtmgp-data-sw-g112-210809.nc",
        "lw_file_clouds":"rrtmgp-cloud-optics-coeffs-lw.nc",
        "sw_file_clouds":"rrtmgp-cloud-optics-coeffs-sw.nc",
    }.items():
        filename = physics_namelist.get(name,"")
        if type(filename) is not str:
            raise TypeError(f"radiation setup requires static {name}")
        if filename != selected:
            raise NotImplementedError(f"R4 radiation setup requires selected {name}")
    r1, slag, sdec, cdec = _solar_state(jdat,con_pi=UFS_CONSTANTS.PI,device=jdat.device)
    config = RadiationConfig(
        imp_physics=88,ntcw=2,ntiw=4,ntrw=3,ntsw=5,ntgl=6,ntinc=7,ntoz=10,ntrac=10,
        nleffr=1,nieffr=2,nseffr=3,
        lmfshal=value("shal_cnv",False) and value("imfshalcnv",1) > 0,
        lmfdeep2=value("imfdeepcnv",1) in (2,3,4,5),
        imfdeepcnv=value("imfdeepcnv",1),imfdeepcnv_samf=2,imfdeepcnv_gf=3,
        do_mynnedmf=False,ltaerosol=value("ltaerosol",False),
        lgfdlmprad=value("lgfdlmprad",False),doGP_smearclds=frozen_smearclds,
        **{n:value(n,default) for n,default in {
            "ico2":0,"iovr":1,"iovr_convcld":1,"idcor":1,
            "isubc_sw":0,"isubc_lw":0,"swhtr":True,"lwhtr":True,
            "tend_opt_lwrad":4,"tend_opt_swrad":4,"fhswr":3600.,"fhlwr":3600.,
            "rrtmgp_sw_phys_blksz":1,"rrtmgp_lw_phys_blksz":1,"use_LW_jacobian":False,
            "ialb":0,"iems":0,"frac_grid":False,"cplice":False,"use_cice_alb":False,
            "lndp_type":0,"min_seaice":1.e-11,"min_lakeice":.15,
        }.items()},
        doGP_lwscat=value("dogp_lwscat",False),
        nGauss_angles=value("rrtmgp_ngauss_ang",1),nrghice=value("rrtmgp_nrghice",3),
        lsswr=True,lslwr=True,lssav=True,lsm=2,dtp=dt_physics,dtf=dt_dycore,
        solhr=solhr,solcon=torch.div(1366.,r1*r1),jdat=jdat,
        co2_year=initial_year,co2_month=initial_month,slag=slag,sdec=sdec,cdec=cdec,
        **_v17_radiation_data(value, input_dir=input_dir, grid=grid, jdat=jdat,
                              aer_jdat=aer_jdat, initial_year=initial_year, forecast_hours=forecast_hours))
    if (config.isubc_sw == 0 or config.isubc_lw == 0) and config.iovr > 2:
        config = replace(config, iovr=1)  # GFS_typedefs: IOVR > 2 needs McICA sampling
    # swhtr/lwhtr false (operational GFS): clear-sky heating rates are not output, so cloudy
    # columns need no clear-sky solve (doSWclrsky/doLWclrsky in rrtmgp_sw/lw_main).
    if (min(config.fhswr,config.fhlwr) <= 0
        or config.tend_opt_lwrad not in (1,2,3,4) or config.tend_opt_swrad not in (1,2,3,4)):
        raise ValueError("invalid radiation frequency, heating or tendency controls")
    config.require_supported()
    return config


def _v17_radiation_data(value, *, input_dir, grid, jdat, aer_jdat, initial_year,
                        forecast_hours) -> dict[str, Any]:
    """Host preparation of the GFS v17 radiation data (GFS_rrtmgp_setup_init/timestep_init and
    GFS_phys_time_vary_init): config fields of ``RadiationConfig``."""
    from pathlib import Path
    from rrtmgp_radiation_torch import merra2
    from rrtmgp_radiation_torch.setup import _load_annual_solar_constant, load_co2_year
    isol, ico2 = value("isol", 0), value("ico2", 0)
    iaer = value("iaer", 1)
    iaermdl, iaerflg = iaer // 1000, iaer % 1000  # GFS_rrtmgp_setup_init with ictm = 1
    out: dict[str, Any] = {"isol": isol, "iaermdl": iaermdl, "iaerflg": iaerflg}
    if not (isol or ico2 or iaerflg):
        return out
    if input_dir is None:
        raise ValueError("isol/ico2/iaer data need input_dir (the run folder)")
    input_dir = Path(input_dir)
    device = jdat.device
    years = range(initial_year, initial_year + V17_DATA_YEARS)
    out["data_year0"] = initial_year
    if isol == 2:
        out["solc0"] = torch.tensor([_load_annual_solar_constant(y, filename="solarconstant_noaa_an.txt",
                                     physics_input_dir=input_dir) for y in years], dtype=torch.float64, device=device)
    if ico2 == 2:
        tables = [load_co2_year(input_dir, y) for y in years]
        out["co2_monthly"] = torch.stack([t for t, _ in tables]).to(device)
        out["co2_global"] = torch.tensor([g for _, g in tables], dtype=torch.float64, device=device)
    if iaerflg:
        if iaermdl != 1 or iaerflg != 11:
            raise NotImplementedError(f"iaer={iaer}: only no aerosol and MERRA-2 iaer=1011 are ported")
        if grid is None or aer_jdat is None:
            raise ValueError("MERRA-2 preparation needs the grid and aer_jdat")
        y, m, d, _, h = (int(v) for v in jdat.detach().cpu().tolist()[:5])
        months = merra2.months_needed(y, m, d, h, hours=forecast_hours)
        out["merra2"] = merra2.merra2_to(
            merra2.prepare_merra2_climatology(input_dir, grid.xlat_d, grid.xlon_d, months), device)
        out["merra2_optics"] = merra2.merra2_to(merra2.load_merra2_optics(input_dir, con_pi=UFS_CONSTANTS.PI), device)
        out["aer_jdat"] = aer_jdat
    return out


@dataclass(frozen=True)
class GWDSchemeControls:
    control: Mapping[str, Any]
    cdmbgwd: torch.Tensor
    alpha_fd: torch.Tensor


def prepare_r4_gwd_controls(physics_namelist: Mapping[str, Any],
    ugwp_namelist: Mapping[str, Any], *, geometry: ReferenceGeometry,
    input_dir=None) -> GWDSchemeControls:
    """Selected UGWP-v0/PSL (R4) or UGWP-v1/GSL (GFS v17) controls from source namelists.

    Resolution stays the device integer from native reference geometry.
    The current nonstationary runtime implements the declared R4 launch/spectrum.
    Reject other initialization knobs rather than silently use those constants.
    UGWP v1 (do_ugwp_v1 without do_ugwp_v0) reads ugwp_limb_tau.nc from ``input_dir``.
    """
    def value(name, default):
        return _static_namelist_value(physics_namelist, name, default)

    if value("do_ugwp_v1", False) and not value("do_ugwp_v0", True):
        return _prepare_ugwpv1_gwd_controls(physics_namelist, ugwp_namelist, geometry=geometry,
                                            input_dir=input_dir)
    flags = {n: value(n, default) for n, default in {
        "do_ugwp_v0": True, "do_gwd_opt_psl": False, "do_gsl_drag_ls_bl": False,
        "do_gsl_drag_ss": False, "do_gsl_drag_tofd": False, "ugwp_seq_update": False,
        "ldiag_ugwp": False, "ldiag3d": False,
    }.items()}
    if (value("gwd_opt",1) not in (2,22) or not flags["do_ugwp_v0"]
        or not flags["do_gwd_opt_psl"] or flags["do_gsl_drag_ls_bl"]
        or any(value(n,False) for n in ("do_ugwp_v0_orog_only", "do_ugwp_v0_nst_only",
            "do_ugwp_v1", "do_ugwp_v1_orog_only", "do_ugwp_v1_w_gsldrag", "do_ngw_ec"))
        or value("spp_gwd",0) != 0):
        raise NotImplementedError("R4 GWD setup requires deterministic UGWP-v0 and PSL SS/TOFD")
    for name, default, selected in (
        ("knob_ugwp_solver",1,2), ("knob_ugwp_doaxyz",1,1), ("knob_ugwp_doheat",1,1),
        ("knob_ugwp_dokdis",0,1), ("knob_ugwp_ndx4lh",2,1),
        ("knob_ugwp_version",0,0), ("launch_level",55,54),
    ):
        if _static_namelist_value(ugwp_namelist,name,default) != selected:
            raise NotImplementedError(f"R4 UGWP initialization requires selected {name}={selected}")
    for name, default, selected in (
        ("knob_ugwp_source",(1,0,1,0),(1,1,0,0)),
        ("knob_ugwp_wvspec",(1,32,32,32),(1,25,25,25)),
        ("knob_ugwp_azdir",(2,4,4,4),(2,4,4,4)),
        ("knob_ugwp_stoch",(0,0,0,0),(0,0,0,0)),
        ("knob_ugwp_effac",(1.,1.,1.,1.),(1.,1.,1.,1.)),
    ):
        actual = ugwp_namelist.get(name,default)
        if type(actual) not in (list,tuple) or len(actual) != 4:
            raise TypeError(f"UGWP namelist {name} requires four static values")
        actual = tuple(_static_namelist_value({name:v},name,d) for v,d in zip(actual,default))
        if actual != selected:
            raise NotImplementedError(f"R4 UGWP initialization requires selected {name}")
    resolution = geometry.lonr
    if (not isinstance(resolution,torch.Tensor) or resolution.shape != () or resolution.dtype not in (torch.int32,torch.int64)
        or geometry.ak.shape != (56,) or geometry.ak.dtype != torch.float64
        or resolution.device != geometry.ak.device):
        raise ValueError("R4 GWD setup requires device-local 55-level reference geometry")
    torch._assert_async(resolution > 0, "nonpositive GWD reference resolution")
    coefficients = physics_namelist.get("cdmbgwd",(2.,.25,1.,1.))
    if type(coefficients) not in (list,tuple) or len(coefficients) != 4:
        raise TypeError("physics namelist cdmbgwd requires four static values")
    coefficients = tuple(_static_namelist_value({"cdmbgwd":v},"cdmbgwd",1.) for v in coefficients)
    control = {**flags, "lonr":resolution, "ntrac":10, "ntke":9,
        "spp_gwd":0, "tend_opt_gwd":value("tend_opt_gwd",3), "lssav":True,
        "flag_for_gwd_generic_tend":True, "index_of_temperature":10,
        "index_of_x_wind":11, "index_of_y_wind":12,
        "index_of_process_orographic_gwd":11, "index_of_process_nonorographic_gwd":13}
    if control["tend_opt_gwd"] not in (1,2,3,4):
        raise ValueError("GWD tendency application control must be 1-4")
    return GWDSchemeControls(control,
        geometry.ak.new_tensor(coefficients), geometry.ak.new_tensor(value("alpha_fd",12.)))


def _prepare_ugwpv1_gwd_controls(physics_namelist, ugwp_namelist, *, geometry, input_dir):
    """ugwpv1_gsldrag_init for the GFS v17 options: UGWP v1 waves with the PSL GSL drag suite.

    Ported: do_ugwp_v1 with do_gwd_opt_psl and any of do_gsl_drag_ls_bl/ss/tofd. Rejected:
    drag_suite_run (do_gwd_opt_psl false), orogw_v1, the ECMWF waves, SPP and ldiag_ugwp.
    """
    from pathlib import Path

    from .gwd_v1 import prepare_ugwpv1

    def value(name, default):
        return _static_namelist_value(physics_namelist, name, default)

    flags = {n: value(n, default) for n, default in {
        "do_ugwp_v1": False, "do_ugwp_v0": False, "do_gwd_opt_psl": False, "do_gsl_drag_ls_bl": False,
        "do_gsl_drag_ss": False, "do_gsl_drag_tofd": False, "ugwp_seq_update": False,
        "ldiag_ugwp": False, "ldiag3d": False, "do_ngw_ec": False, "do_ugwp_v1_orog_only": False,
    }.items()}
    if (value("gwd_opt", 1) not in (2, 22) or not flags["do_gwd_opt_psl"] or flags["do_ngw_ec"]
            or flags["do_ugwp_v1_orog_only"] or flags["ldiag_ugwp"]
            or any(value(n, False) for n in ("do_ugwp_v0_orog_only", "do_ugwp_v0_nst_only",
                                             "do_ugwp_v1_w_gsldrag"))
            or value("spp_gwd", 0) != 0):
        raise NotImplementedError("UGWP v1 setup requires the PSL GSL drag suite with deterministic v1 waves")
    resolution = geometry.lonr
    if (not isinstance(resolution, torch.Tensor) or resolution.shape != ()
            or resolution.dtype not in (torch.int32, torch.int64)
            or geometry.ak.shape != (56,) or geometry.ak.dtype != torch.float64
            or resolution.device != geometry.ak.device):
        raise ValueError("UGWP v1 setup requires device-local 55-level reference geometry")
    if input_dir is None:
        raise ValueError("UGWP v1 setup requires the run folder holding ugwp_limb_tau.nc")
    coefficients = physics_namelist.get("cdmbgwd", (2., .25, 1., 1.))
    if type(coefficients) not in (list, tuple) or len(coefficients) != 4:
        raise TypeError("physics namelist cdmbgwd requires four static values")
    coefficients = tuple(_static_namelist_value({"cdmbgwd": v}, "cdmbgwd", 1.) for v in coefficients)
    # One host read of the reference pressures: the source initializes from host scalars.
    ugwpv1 = prepare_ugwpv1(ugwp_namelist, ak=tuple(geometry.ak.tolist()), bk=tuple(geometry.bk.tolist()),
                            levs=55, tau_file=Path(input_dir) / "ugwp_limb_tau.nc")
    control = {**flags, "lonr": resolution, "ntrac": 10, "ntke": 9, "spp_gwd": 0,
        "tend_opt_gwd": value("tend_opt_gwd", 3), "lssav": True, "flag_for_gwd_generic_tend": True,
        "index_of_temperature": 10, "index_of_x_wind": 11, "index_of_y_wind": 12,
        "index_of_process_orographic_gwd": 11, "index_of_process_nonorographic_gwd": 13,
        "psl_gwd_dx_factor": float(value("psl_gwd_dx_factor", 6.0)), "ugwpv1": ugwpv1}
    if control["tend_opt_gwd"] not in (1, 2, 3, 4):
        raise ValueError("GWD tendency application control must be 1-4")
    return GWDSchemeControls(control, geometry.ak.new_tensor(coefficients),
                             geometry.ak.new_tensor(value("alpha_fd", 12.)))


def initialize_reference_geometry(
    area_cell: torch.Tensor,
    prsi: torch.Tensor,
    *,
    n_cells_solve: int,
    n_cells_global: int,
    mpi_size: int,
    halo=None,
) -> ReferenceGeometry:
    """Source ordered, area-weighted MPAS equivalents before UGWP initialization.

    On a partition (``halo``: the run's ``DynamicsHaloExchange``) the sums run over all cells in
    global-ID order, so every rank count gives the one-rank values.
    """
    n = n_cells_solve
    if (
        any(type(v) is not int for v in (n, n_cells_global, mpi_size))
        or n < 1
        or (halo is None and (n != n_cells_global or mpi_size != 1))
        or (halo is not None and (mpi_size != halo.transport.world_size or mpi_size < 2
                                  or n != halo.partition.cells.owned_count
                                  or n_cells_global != halo.partition.cells.global_count))
    ):
        raise ValueError("reference geometry requires a fully owned one-rank domain or its partition")
    if (
        prsi.ndim != 2
        or prsi.shape[0] < n
        or prsi.shape[1] < 2
        or area_cell.ndim != 1
        or area_cell.shape[0] < n
        or prsi.dtype not in WORKING_DTYPES
        or area_cell.dtype != prsi.dtype
        or prsi.device != area_cell.device
    ):
        raise ValueError("invalid native reference geometry operands")
    area, pressure = area_cell[:n], prsi[:n]
    torch._assert_async(
        (torch.isfinite(area) & (area > 0.0)).all() & torch.isfinite(pressure).all(),
        "invalid native reference geometry inputs",
    )
    totals = _ordered_totals(
        global_rows(torch.cat((area[:, None], area[:, None] * pressure), dim=1), halo),
        torch.zeros(pressure.shape[1] + 1, dtype=area.dtype, device=area.device),
    )
    global_area = totals[0]
    ak = totals[1:] / global_area
    torch._assert_async(
        (ak > 0.0).all() & (ak[1:] < ak[:-1]).all(),
        "nonpositive or nondecreasing reference interface pressure",
    )
    mean_area = global_area / n_cells_global
    radius = torch.sqrt(global_area / (4.0 * UFS_CONSTANTS.PI))
    spacing = torch.sqrt(2.0 * mean_area / (3.0**0.5))
    longitude_count = 2.0 * UFS_CONSTANTS.PI * radius / spacing
    lonr = torch.clamp_min(_nint_nonnegative(longitude_count), 4).to(torch.int32)
    return ReferenceGeometry(
        ak, torch.zeros_like(ak), lonr, global_area, mean_area, spacing
    )


@dataclass(frozen=True)
class BridgeInitialization:
    native_state: dict[str, Any]
    atmosphere: NativeAtmosphere
    native_surface: NativeSurface
    grid: GridProjection
    statein: Statein
    stateout: Stateout
    sfcprop: Sfcprop
    radiation_albedo: torch.Tensor
    reference_geometry: ReferenceGeometry


def initialize_bridge(
    state: Mapping[str, Any],
    mesh: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    surface_control: SurfaceControl,
    soil: SoilParameters,
    n_cells_global: int,
    mpi_size: int,
) -> BridgeInitialization:
    """Allocate/map the first cold bridge, before reference geometry and physics_init.

    MPAS dynamics initialization must precede this call. CCPP lifecycle containers
    outside this bridge remain separate work; this does not execute physics_init.
    The second surface mapping must consume actual lifecycle carries, not repeat
    this allocation. Stateout stays at its source allocation value until seeding.
    """
    if config.get("physical_constants_profile") != "ufs":
        raise ValueError("cold UFS bridge requires UFS physical constants")
    if surface_control.initialize_noahmp_liquid:
        raise ValueError("first cold bridge precedes Noah-MP soil table initialization")
    values = {**mesh, **state}
    nc, nz = surface_control.n_cells_solve, values["rho_zz"].shape[1]
    halo = config.get("_halo_exchanger")
    # One rank owns every cell; a partition (mpi_size ranks, its halo exchanger in the config) owns
    # nCellsSolve of its nCells local rows.
    owned = values.get("nCellsSolve", values.get("nCells")) if halo is not None else values.get("nCells")
    if (
        type(n_cells_global) is not int
        or type(mpi_size) is not int
        or (halo is None and (n_cells_global != nc or mpi_size != 1))
        or (halo is not None and (mpi_size < 2 or n_cells_global <= nc))
    ):
        raise ValueError("first R4 bridge requires a fully owned one-rank domain or a partition with its halo exchanger")
    if type(owned) is not int or owned != nc or type(values.get("nCells")) is not int:
        raise ValueError(
            "first R4 bridge requires static owned native cell metadata"
        )
    if values["scalars"].shape[2] != 10 or surface_control.ntqv != 1:
        raise ValueError(
            "first R4 bridge requires its ten-tracer layout and vapor slot one"
        )
    device, dtype = values["rho_zz"].device, values["rho_zz"].dtype
    # Owned surface-pressure storage is overwritten by the native projection.
    values.setdefault("surface_pressure", torch.zeros_like(values["rho_zz"][:, 0]))
    values.setdefault("pressure", torch.zeros_like(values["theta_m"]))
    atmosphere = NativeAtmosphere.from_mpas_state(values, n_cells_solve=nc)
    # Match after_atm_core_init: cell winds must depend on the current native u.
    refreshed = refresh_host_diagnostics(
        atmosphere,
        ProjectionMesh.from_mpas_state(values),
        ReconstructionMesh(values["nCells"], values["nEdges"], values["edgesOnCell"],
            values["nEdgesOnCell"], values["coeffs_reconstruct"],
            values["latCell"], values["lonCell"], config.get("config_on_a_sphere",True)),
        OutputDiagnostics(*(values[name] for name in (
            "rho", "pressure", "uReconstructX", "uReconstructY", "uReconstructZ"))),
        exchanged_u=values["u"],
    )
    atmosphere = refreshed.atmosphere
    values.update(theta=atmosphere.theta,
        uReconstructZonal=atmosphere.u_reconstruct_zonal,
        uReconstructMeridional=atmosphere.u_reconstruct_meridional,
        **{name:getattr(refreshed.diagnostics,attr) for name,attr in (
            ("rho","rho"),("pressure","pressure"),
            ("uReconstructX","u_reconstruct_x"),("uReconstructY","u_reconstruct_y"),
            ("uReconstructZ","u_reconstruct_z"))})
    surface = NativeSurface.from_mpas_state(values)
    target = Statein.allocate(
        ncol=nc, nlev=nz, ntrac=10, soil_levels=4, lightning_threat=False, device=device, dtype=dtype
    )
    projected = project_atmosphere(
        atmosphere,
        ProjectionMesh.from_mpas_state(values),
        target,
        tracers=r4_tracer_contract(),
    )
    atmosphere = replace(atmosphere, surface_pressure=projected.surface_pressure)
    values["surface_pressure"] = projected.surface_pressure
    grid = project_grid(
        Grid.allocate_r4(ncol=nc, device=device, dtype=dtype),
        n_cells_solve=nc,
        lat_cell=values["latCell"],
        lon_cell=values["lonCell"],
        area_cell=values["areaCell"],
        mesh_density=values["meshDensity"],
        nominal_min_dc=values["nominalMinDc"],
        config_len_disp=config["config_len_disp"],
    )
    values["nominalMinDc"] = grid.length_scale
    initialized_surface = initialize_surface(
        surface,
        projected.statein,
        Sfcprop.allocate_r4(ncol=nc, device=device, dtype=dtype),
        torch.zeros(nc, dtype=dtype, device=device),
        surface_control,
        soil,
    )
    reference = initialize_reference_geometry(
        values["areaCell"],
        projected.statein.prsi,
        n_cells_solve=nc,
        n_cells_global=n_cells_global,
        mpi_size=mpi_size,
        halo=halo,
    )
    return BridgeInitialization(
        values,
        atmosphere,
        surface,
        grid,
        projected.statein,
        Stateout.allocate(ncol=nc, nlev=nz, ntrac=10, device=device, dtype=dtype),
        initialized_surface.sfcprop,
        initialized_surface.radiation_albedo,
        reference,
    )
