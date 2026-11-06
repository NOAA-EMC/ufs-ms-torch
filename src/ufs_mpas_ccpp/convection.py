# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Selected R4 SAMF cores and real-cap post diagnostics."""

from dataclasses import dataclass, replace

import torch
import torch.autograd.forward_ad as forward_ad

from sa_sas_deep_convection_torch.driver import run_sa_sas_deep_convection_compacted
from sa_sas_deep_convection_torch.state import SaSasDeepConvectionState
from sa_sas_deep_convection_torch.samf_interstitials import (
    mpas_tempo_layout, tendencies_from_dclw, build_clw, samf_water_reconciliation,
)
from sa_sas_deep_convection_torch.segments.segment_v import DTEND_EXCLUDED
from sa_sas_deep_convection_torch import constants
from sa_sas_shallow_convection_torch.driver import run_sa_sas_shallow_convection_compacted
from sa_sas_shallow_convection_torch.state import SaSasShallowConvectionState

from sa_tke_edmf_torch.diagnostics import add_diagnostic_
from ufs_physics_common import compile_policy
from .containers import static_tensor, assert_all_finite, _finite_flags
from .photochemistry import photochemistry_post
from .physics_prefix import PhysicsPrefixResult
from .precision import WORKING_DTYPES, working_like
from .runtime_policy import deep_convection_fraction, shallow_convection_fraction

_regional_compile = False
_POST_CONTROLS = (
    "tend_opt_scnv", "tend_opt_dcnv", "imp_physics", "npdf3d", "num_p3d", "ncnvcld3d",
    "lssav", "ldiag3d", "qdiag3d", "samf_water_reconciliation", "satmedmf", "trans_trac",
    "ras", "cscnv", "shcnvcw", "flag_for_scnv_generic_tend", "flag_for_dcnv_generic_tend",
    "index_of_process_scnv", "index_of_process_dcnv", "index_of_temperature",
    "index_of_x_wind", "index_of_y_wind",
)
_POST_FIELDS = (
    "gt0", "gu0", "gv0", "gq0", "ten_t", "ten_u", "ten_v", "ten_q",
    "dtdt", "dudt", "dvdt", "dqdt", "dclw", "dtidx", "dtend", "delt", "dtf",
    "frain", "area", "prsi", "grav", "cp", "hvap", "clw", "rain1", "rainc",
    "cnvprcp", "cldwrk", "cld1d", "cnvw", "cnvc", "cnvw_phy_f3d", "cnvc_phy_f3d",
    "upd_mf", "dwn_mf", "det_mf", "ud_mf", "dd_mf", "dt_mf",
)


def convection_core(fields, *, control, shallow=False):
    """Run existing SAMF numerics, stopping before its standalone post adapter.

    Fields use source argument names. Absent optional arguments have private
    workspace only when their activating controls are explicitly false.
    """
    if type(shallow) is not bool:
        raise TypeError("convection kind must be static")
    for name in ("imp_physics", "npdf3d", "num_p3d"):
        if type(control[name]) is not int:
            raise TypeError(f"convection requires static {name}")
    if control["imp_physics"] != 88:
        raise NotImplementedError("convection core binding selects TEMPO")
    for name in ("progsigma", "progomega", "do_ca", "do_mynnedmf"):
        if type(control[name]) is not bool:
            raise TypeError(f"convection requires static {name}")
        if control[name]:
            raise NotImplementedError(f"R4 convection binding does not activate {name}")
    layout = mpas_tempo_layout(ntc=control["ntc"])
    if layout.ntc not in (0, 7):
        raise NotImplementedError("R4 SAMF chemistry extent is zero; seven is the qualified G3.2 fixture")
    for name, value in (("nn", layout.nn), ("ntr", layout.ntr),
                        ("ntk", layout.ntk), ("itc", layout.itc), ("ntc", layout.ntc)):
        if type(control[name]) is not int or control[name] != value:
            raise ValueError(f"convection requires the source R4 {name}={value}")
    f = dict(fields)
    t = f["t1"]
    if t.ndim != 2 or t.dtype not in WORKING_DTYPES or min(t.shape) < 1:
        raise ValueError("convection requires nonempty float32/float64 column profiles")
    # Source optional sigma/omega, MYNN, MG and CA arguments are null in R4.
    zero = t.new_zeros(())
    zero_profile, zero_column = zero.expand_as(t), zero.expand(t.shape[0])
    for name in ("qmicro", "sigmain", "sigmaout", "omegain", "omegaout",
                 "qlcn", "qicn", "w_upi", "cf_upi", "cnv_mfd", "cnv_dqldt",
                 "clcn", "cnv_fice", "cnv_ndrop", "cnv_nice"):
        f[name] = zero_profile
    for name in ("maxMF", "ca_deep"):
        f[name] = zero_column
    if "cnvc_phy_f3d" not in f:
        if control["npdf3d"] == 3 and control["num_p3d"] == 4:
            raise ValueError("active convection cloud fraction diagnostic is missing")
        f["cnvc_phy_f3d"] = zero_profile
    for name, value in f.items():
        if value.device != t.device:
            raise ValueError(f"convection field {name} has a different device")
        if value.is_floating_point():
            if not working_like(value, t):
                raise TypeError(f"convection field {name} requires the working dtype {t.dtype}")
    assert_all_finite(f, "convection", exclude=("dtend",), compile_check=_regional_compile)
    torch._assert_async((f["delp"] > 0.).all(), "convection requires positive pressure thickness")
    if f["tkeh"].shape != (t.shape[0], t.shape[1]+1):
        raise ValueError("source tkeh requires the complete interface axis")
    f["tkeh"] = f["tkeh"][:, :-1]
    # TMF is only read by inactive prognostic convection branches. The core's
    # private contract is clw-shaped, while the host allocation has ten slots.
    if f["tmf"].shape != (*t.shape, 10):
        raise ValueError("source dvdftra requires ten host tracer slots")
    f["tmf"] = f["tmf"][:, :, :layout.nn]
    for name in ("kbot", "ktop", "kcnv", "islimsk"):
        f[name] = f[name].to(t.dtype)
    config = {**control, "tracer_layout": layout.as_config(),
              "ndtend": f["dtend"].shape[2]}
    names = ("ten_t", "ten_u", "ten_v", "ten_q", "dqtr", "rn", "kbot", "ktop",
             "kcnv", "ud_mf", "dt_mf", "cnvw", "cnvc")
    if not shallow:
        names += ("dd_mf", "cldwrk", "rainevap")
    # Each scheme runs past its trigger on the triggered columns only.
    if shallow:
        result = run_sa_sas_shallow_convection_compacted(SaSasShallowConvectionState(f, config), through="P",
            outputs=(*names, "errflg"), fraction=shallow_convection_fraction())
    else:
        # The core stops before its post (segment V), the only dtend reader; with ldiag3d off
        # there are no slots to pass.
        core = f if control["ldiag3d"] else {k: v for k, v in f.items() if k != "dtend"}
        result = run_sa_sas_deep_convection_compacted(SaSasDeepConvectionState(core, config), through="U",
            outputs=(*names, "errflg"), fraction=deep_convection_fraction())
    torch._assert_async(result.require("errflg") == 0., "SAMF core failed")
    outputs = {name: result.require(name) for name in names}
    for name in ("kbot", "ktop", "kcnv"):
        outputs[name] = outputs[name].to(torch.int32)
    return outputs


def convection_post(fields, *, control, shallow=False):
    """Source post with live diagnostic slots, cloud carry and water correction.

    This selects TEMPO, SAMF and the R4 tracer table. No source stderr ledger
    qualification is implied by the numerical state/diagnostic update.
    """
    if type(shallow) is not bool:
        raise TypeError("convection kind must be static")
    layout = mpas_tempo_layout()
    mode = control["tend_opt_scnv" if shallow else "tend_opt_dcnv"]
    if type(mode) is not int:
        raise TypeError("convection post mode must be static")
    for name in ("imp_physics", "npdf3d", "num_p3d", "ncnvcld3d"):
        if type(control[name]) is not int:
            raise TypeError(f"convection post requires static {name}")
    reconcile = control["samf_water_reconciliation"]
    for name in ("lssav", "ldiag3d", "qdiag3d", "samf_water_reconciliation",
                 "satmedmf", "trans_trac", "ras", "cscnv", *(["shcnvcw"] if shallow else [])):
        if type(control[name]) is not bool:
            raise TypeError(f"convection post requires static {name}")
    if control["imp_physics"] != 88 or control["ras"] or control["cscnv"]:
        raise NotImplementedError("convection post selects R4 TEMPO/SAMF")
    if reconcile and mode not in (1, 3):
        raise ValueError("SAMF water reconciliation requires applied Stateout")
    f = dict(fields)
    q = f["gq0"]
    if q.ndim != 3 or q.shape[2] != layout.ntrac:
        raise ValueError("convection post requires ten R4 tracers")
    dtidx, dtend = f["dtidx"], f["dtend"]
    if (dtidx.ndim != 2 or dtidx.shape[0] < 100+layout.ntrac or
        dtidx.dtype not in (torch.int32, torch.int64) or dtidx.device != q.device or
        dtend.ndim != 3 or dtend.shape[:2] != q.shape[:2]):
        raise ValueError("invalid source convection diagnostic map")
    for name, value in f.items():
        if value.device != q.device:
            raise ValueError(f"convection field {name} has a different device")
        if value.is_floating_point():
            if not working_like(value, q):
                raise TypeError(f"convection field {name} requires the working dtype {q.dtype}")
    if _regional_compile:
        names = tuple(name for name, value in f.items()
                      if value.is_floating_point() and value.numel() and name != "dtend")
        values = (tuple(f[name] for name in names) if torch.compiler.is_compiling()
                  else tuple({id(f[name]): f[name] for name in names}.values()))
        with torch.no_grad():
            primals = tuple(forward_ad.unpack_dual(value).primal for value in values)
        result, finite = compile_policy.compiled(_convection_post_checked, normalize_layouts=False)(
            {name: f[name] for name in _POST_FIELDS if name in f},
            primals,
            control={name: control[name] for name in _POST_CONTROLS if name in control},
            shallow=shallow)
        finite = torch.cat(finite).all()
        message = f"nonfinite convection post (one of: {', '.join(names)})"
        torch._assert_async(finite, message[:250] + "...)" if finite.is_cuda and len(message) >= 255 else message)
        return result
    assert_all_finite(f, "convection post", exclude=("dtend",))
    return _convection_post_tensors(f, control=control, shallow=shallow)


def _convection_post_checked(f, checked, *, control, shallow):
    # Both consumers borrow their inputs. Keep their actual strides and one dispatch.
    with torch.no_grad():
        finite = _finite_flags(checked)
    return _convection_post_tensors(f, control=control, shallow=shallow), finite


def _convection_post_tensors(f, *, control, shallow):
    layout = mpas_tempo_layout()
    mode = control["tend_opt_scnv" if shallow else "tend_opt_dcnv"]
    reconcile = control["samf_water_reconciliation"]
    q, dtidx, dtend = f["gq0"], f["dtidx"], f["dtend"]
    f = dict(f)
    f["ten_q"] = tendencies_from_dclw(f["ten_q"], f["dclw"], layout, is_gfdl=False)
    out = photochemistry_post(f, dtp=f["delt"], mode=mode)
    out["ten_q"] = f["ten_q"]
    if reconcile:
        out["gq0"], out["gt0"], errflg = samf_water_reconciliation(
            out["gq0"], out["gt0"], q, f["rain1"], f["area"], f["prsi"],
            f["frain"], f["grav"], f["cp"], f["hvap"], layout, delt=f["delt"])
        torch._assert_async(errflg == 0., "SAMF water reconciliation failed")
    out["clw"] = (f["clw"] if shallow else build_clw(out["gq0"], f["clw"], layout,
        imp_physics=88, transport=control["satmedmf"] or control["trans_trac"]))
    out["rainc"] = f["frain"] * f["rain1"]
    if shallow:
        out["rainc"] = f["rainc"] + out["rainc"]
        out["cnvprcp"] = f["cnvprcp"]
    else:
        out["cldwrk"] = f["cldwrk"]
        if control["lssav"]:
            out["cldwrk"] = f["cldwrk"] + f["cld1d"] * f["dtf"]
    out["cnvw"], out["cnvc"] = f["cnvw"], f["cnvc"]
    out["cnvw_phy_f3d"] = f["cnvw_phy_f3d"]
    if "cnvc_phy_f3d" in f:
        out["cnvc_phy_f3d"] = f["cnvc_phy_f3d"]
    cloud_water = control["npdf3d"] == 0 and control["ncnvcld3d"] == 1
    cloud_both = control["npdf3d"] == 3 and control["num_p3d"] == 4
    if shallow:
        cloud_both = cloud_both and control["shcnvcw"]
    for name, active in (("cnvw", cloud_water or cloud_both), ("cnvc", cloud_both)):
        if active:
            out[name+"_phy_f3d"] = f[name+"_phy_f3d"] + f[name] if shallow else f[name].clone()
            if not shallow:
                out[name] = torch.zeros_like(f[name])
    for name in (() if shallow else ("upd_mf", "dwn_mf", "det_mf")):
        out[name] = f[name]
    flag = control["flag_for_scnv_generic_tend" if shallow else "flag_for_dcnv_generic_tend"]
    if type(flag) is not bool:
        raise TypeError("convection diagnostic switch must be static")
    out["dtend"] = dtend
    if control["lssav"] and control["ldiag3d"] and flag:
        process = control["index_of_process_scnv" if shallow else "index_of_process_dcnv"]
        if type(process) is not int or not 1 <= process <= dtidx.shape[1]:
            raise ValueError("invalid convection diagnostic process index")
        out["dtend"] = dtend.clone()
        terms = [(control["index_of_"+n], f[ten]) for n, ten in
                 (("temperature", "ten_t"), ("x_wind", "ten_u"), ("y_wind", "ten_v"))]
        excluded = ({layout.index(n) for n in DTEND_EXCLUDED} if not shallow else
                    {layout.index(n) for n in ("ntcw", "ntiw", "ntclamt", "ntrw", "ntsw",
                     "ntrnc", "ntsnc", "ntgl", "ntgnc", "ntsigma")})
        transport = control["satmedmf"] or control["trans_trac"]
        terms += [(100+n, f["ten_q"][:, :, n-1]) for n in range(2, layout.ntrac+1)
                  if not transport or n not in excluded]
        terms.append((100+layout.index("ntqv"), f["ten_q"][:, :, layout.index("ntqv")-1]))
        for index, tendency in terms:
            if type(index) is not int or not 1 <= index <= dtidx.shape[0]:
                raise ValueError("invalid convection diagnostic variable index")
            slot = dtidx[index-1, process-1]
            torch._assert_async(slot <= dtend.shape[2], "convection diagnostic slot exceeds allocation")
            increment = (tendency * f["delt"]) * f["frain"]
            add_diagnostic_(out["dtend"], increment, slot)
        if not shallow and control["qdiag3d"]:
            for name, tendency in (("upd_mf", "ud_mf"), ("dwn_mf", "dd_mf"), ("det_mf", "dt_mf")):
                out[name] = f[name] + f[tendency] * (f["grav"] * f["frain"])
    return out


@dataclass(frozen=True)
class ConvectionResult:
    prefix: PhysicsPrefixResult
    ud_mf: torch.Tensor


def apply_convection(statein, grid, prefix, *, hpbl, ud_mf, prevsq, dtp, dtf,
                     dtidx, coefficients, control, shallow=False):
    """Bind one live SAMF core/post; return its persistent updraft mass flux.

    The deep result feeds shallow directly. hpbl, prevsq and ud_mf belong to
    GFS_Tbd and survive the microphysics group reset.
    """
    if type(shallow) is not bool:
        raise TypeError("convection kind must be static")
    work, state, carry = prefix.interstitial, prefix.stateout, prefix.carry
    ncol, nlev = state.gt0.shape
    for name, value, shape in (("hpbl", hpbl, (ncol,)), ("prevsq", prevsq, (ncol,nlev)),
                               ("ud_mf", ud_mf, (ncol,nlev))):
        if value.shape != shape or value.dtype != state.gt0.dtype or value.device != state.gt0.device:
            raise ValueError(f"invalid persistent convection {name}")
    config = {**control, "nn": control["nn"], "ntr": control["nsamftrac"],
        "ntc": control["ntchm"], "ncloud": control["ncnd"],
        "mp_phys": control["imp_physics"], "mp_phys_mg": control["imp_physics_mg"],
        "tend_opt_dcnv": control["tend_opt_deep_conv"],
        "tend_opt_scnv": control["tend_opt_shal_conv"]}
    flags = work["otsptflag"]
    if flags.shape != (10,) or flags.dtype != torch.bool or flags.device != state.gt0.device:
        raise ValueError("invalid live convection tracer flags")
    torch._assert_async((flags == static_tensor(mpas_tempo_layout().otsptflag, dtype=flags.dtype, device=flags.device)).all(),
                        "live convection tracer flags differ from R4")
    suffix = "shal" if shallow else "deep"
    f = {**coefficients,
        **{name: static_tensor(getattr(constants, name.upper()), dtype=state.gt0.dtype, device=state.gt0.device) for name in
           ("cliq", "cp", "cvap", "eps", "epsm1", "fv", "grav", "hvap", "rd", "rv", "t0c")},
        **{name: coefficients[name+"_"+suffix] for name in ("clam", "c0s", "c1", "pgcon", "asolfac")},
        "delt": dtp, "dtf": dtf, "psp": statein.pgr, "prslp": statein.prsl,
        "prsi": statein.prsi, "phil": statein.phil, "dot": statein.vvl,
        "q": statein.qgrs[:, :, 0], "q1": state.gq0[:, :, 0],
        "t1": state.gt0, "u1": state.gu0, "v1": state.gv0,
        "qtr": work["clw"], "dqtr": work["dclw"], "tmf": work["dvdftra"],
        "delp": work["del"], "tkeh": work["tkeh"], "islimsk": work["islmsk"],
        "hpbl": hpbl, "prevsq": prevsq, "ud_mf": ud_mf,
        "garea": grid.area, "area": grid.area, "dtidx": dtidx,
        "rn": work["raincs" if shallow else "raincd"], "cnvw": carry.cnvw,
        **{name: work[name] for name in ("kbot", "ktop", "kcnv", "dd_mf", "dt_mf", "cnvc",
            "dtdt", "dudt", "dvdt", "dqdt", "ten_t", "ten_u", "ten_v", "ten_q")},
        **{name: getattr(state,name) for name in ("gt0", "gu0", "gv0", "gq0")},
        **{name: carry.intdiag[name] for name in ("dtend", "rainc", "cldwrk", "cnvprcp",
                                                "upd_mf", "dwn_mf", "det_mf")},
        "rainevap": carry.coupling["condition"], "clw": work["clw"]}
    f["cldwrk"] = work["cld1d"]
    for name, index in (("cnvw", control["ncnvw"]), ("cnvc", control["ncnvc"])):
        if type(index) is not int or index > carry.phy_f3d.shape[2]:
            raise ValueError(f"invalid source {name} diagnostic slot")
        if index > 0:
            f[name+"_phy_f3d"] = carry.phy_f3d[:, :, index-1]
    if not shallow:
        for name in ("betal", "betas"):
            f[name] = coefficients[name+"_deep"]
    core = convection_core(f, control=config, shallow=shallow)
    pf = {**f, **core, "dclw": core["dqtr"], "rain1": core["rn"],
          "cld1d": core.get("cldwrk", work["cld1d"]), "cldwrk": carry.intdiag["cldwrk"]}
    for name, index in (("cnvw", control["ncnvw"]), ("cnvc", control["ncnvc"])):
        if type(index) is not int or index > carry.phy_f3d.shape[2]:
            raise ValueError(f"missing source {name} diagnostic slot")
        if index > 0:
            pf[name+"_phy_f3d"] = carry.phy_f3d[:, :, index-1]
    post = convection_post(pf, control={name: config[name] for name in _POST_CONTROLS if name in config},
                           shallow=shallow)
    updated_work = {**work, "dclw": core["dqtr"],
        "raincs" if shallow else "raincd": core["rn"],
        **{name: core[name] for name in ("kbot", "ktop", "kcnv", "dt_mf", "ten_t", "ten_u", "ten_v")},
        **{name: post[name] for name in ("clw", "cnvc", "ten_q", "dtdt", "dudt", "dvdt", "dqdt")}}
    if not shallow:
        updated_work.update(dd_mf=core["dd_mf"], cld1d=core["cldwrk"])
    phy = carry.phy_f3d.clone()
    for name, index in (("cnvw", control["ncnvw"]), ("cnvc", control["ncnvc"])):
        if index > 0:
            phy[:, :, index-1] = post[name+"_phy_f3d"]
    diagnostics = {**carry.intdiag, **{name: post[name] for name in post if name in carry.intdiag}}
    coupling = carry.coupling if shallow else {**carry.coupling, "condition": core["rainevap"]}
    result = replace(prefix,
        stateout=replace(state, **{name: post[name] for name in ("gt0", "gu0", "gv0", "gq0")}),
        interstitial=updated_work,
        carry=replace(carry, cnvw=post["cnvw"], phy_f3d=phy, intdiag=diagnostics, coupling=coupling))
    return ConvectionResult(result, core["ud_mf"])


def _apply_convection_regions(*args, **kwargs):
    from sa_sas_deep_convection_torch.compile import regional_compile as deep_regions
    from sa_sas_shallow_convection_torch.compile import regional_compile as shallow_regions

    global _regional_compile
    previous, _regional_compile = _regional_compile, True
    try:
        with deep_regions(), shallow_regions():
            return apply_convection(*args, **kwargs)
    finally:
        _regional_compile = previous


_apply_convection_regions.__wrapped__ = apply_convection
_apply_convection_regions.warm = lambda *args, **kwargs: None
compile_policy._compiled[apply_convection] = _apply_convection_regions
