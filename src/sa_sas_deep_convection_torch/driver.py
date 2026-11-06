# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Integrated SA-SAS deep-convection PyTorch driver for cumulative port gates."""

from __future__ import annotations

from typing import Any

import torch

from sa_sas_deep_convection_torch.compile import DEFAULT_COMPILE_CATEGORIES, compile_categories, compile_categories_explicitly_set
from sa_sas_deep_convection_torch.constants import (
    ASOLFAC,
    BETADCU,
    BETAL,
    BETAMCU,
    BETAS,
    BETASCU,
    C0S,
    C1,
    CAT_ADJ_DEEP,
    CLAM,
    CLIQ,
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
    ITC,
    KM,
    NCLOUD,
    NDTEND,
    NN,
    NTC,
    NTK,
    NTHRESH,
    NTR,
    PGCON,
    RD,
    RV,
    SEGMENT_IDS,
    T0C,
)
from sa_sas_deep_convection_torch.ops import stored
from sa_sas_deep_convection_torch.segments import segment_a, segment_b, segment_c, segment_d, segment_e, segment_f, segment_g, segment_h, segment_i, segment_j, segment_k, segment_l, segment_m, segment_n, segment_o, segment_p, segment_q, segment_r, segment_s, segment_t, segment_u, segment_v
from sa_sas_deep_convection_torch.state import SaSasDeepConvectionState, require_same_device
from sa_sas_deep_convection_torch.thermo import fpvs

_SEGMENT_ORDER = {segment_id: idx for idx, segment_id in enumerate(SEGMENT_IDS)}
_segment_profiling_enabled = False


def _scalar(value: float, *, device: torch.device | str | None) -> torch.Tensor:
    return torch.as_tensor(float(value), dtype=torch.float64, device=device)


def _zeros(shape: tuple[int, ...], *, device: torch.device | str | None) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float64, device=device)


def _full(shape: tuple[int, ...], value: float, *, device: torch.device | str | None) -> torch.Tensor:
    return torch.full(shape, float(value), dtype=torch.float64, device=device)


def initial_state_case01_active_maritime_deep(
    *,
    device: torch.device | str | None = None,
    ncol: int | None = None,
    ntrac: int = NTR,
) -> SaSasDeepConvectionState:
    """Build the deterministic active DCNV-entry profile used by the source driver."""

    if type(ntrac) is not int or ntrac < NTR:
        raise ValueError("deep fixture ntrac must retain at least eight source species")
    NTR_runtime = ntrac
    NN_runtime = ntrac + 2
    ncol = IM if ncol is None else int(ncol)
    if ncol <= 0:
        raise ValueError(f"ncol must be positive, got {ncol}")
    levels = torch.arange(KM, dtype=torch.float64, device=device).reshape(1, KM)
    columns = torch.arange(ncol, dtype=torch.int64, device=device)

    prslp = 100000.0 - levels * 5500.0
    prslp = prslp.repeat(ncol, 1)
    delp = _full((ncol, KM), 5000.0, device=device)
    phil = GRAV * levels * 500.0
    phil = phil.repeat(ncol, 1)
    t1 = 295.0 - 3.0 * levels
    t1 = t1.repeat(ncol, 1)
    esat = 0.01 * fpvs(t1)
    pfmb = 0.01 * prslp
    qsat = EPS * esat / (pfmb + EPSM1 * esat)
    q1 = torch.clamp(0.85 * qsat, min=1.0e-8, max=0.05)
    q = q1.clone()
    prevsq = q1 * 0.98
    u1 = _full((ncol, KM), 5.0, device=device)
    v1 = _full((ncol, KM), 1.0, device=device)
    dot = _full((ncol, KM), -0.03, device=device)
    qtr = _zeros((ncol, KM, NN_runtime), device=device)
    qtr[:, :, 0] = q1
    qtr[:, :, 1] = 1.0e-5
    qtr[:, :, 2:] = 1.0e-6
    qtr[:, :, NTK - 1] = 0.1

    islimsk = torch.where(columns % 2 == 0, torch.zeros_like(columns), torch.ones_like(columns)).to(torch.float64)
    one_d = {
        "psp": _full((ncol,), 100000.0, device=device),
        "garea": _full((ncol,), 1.0e8, device=device),
        "hpbl": _full((ncol,), 300.0, device=device),
        "ca_deep": _full((ncol,), 1.0, device=device),
        "maxMF": _zeros((ncol,), device=device),
        "cldwrk": _zeros((ncol,), device=device),
        "rn": _zeros((ncol,), device=device),
        "rainevap": _zeros((ncol,), device=device),
    }
    scalar_fields = {
        "cliq": CLIQ,
        "cp": CP,
        "cvap": CVAP,
        "eps": EPS,
        "epsm1": EPSM1,
        "fv": FV,
        "grav": GRAV,
        "hvap": HVAP,
        "rd": RD,
        "rv": RV,
        "t0c": T0C,
        "delt": DELT,
        "clam": CLAM,
        "c0s": C0S,
        "c1": C1,
        "betal": BETAL,
        "betas": BETAS,
        "evef": EVEF,
        "pgcon": PGCON,
        "asolfac": ASOLFAC,
        "cscale": CSCALE,
        "nthresh": NTHRESH,
        "betadcu": BETADCU,
        "betamcu": BETAMCU,
        "betascu": BETASCU,
        "cat_adj_deep": CAT_ADJ_DEEP,
    }

    fields: dict[str, torch.Tensor] = {
        "tmf": _zeros((ncol, KM, NN_runtime), device=device),
        "qmicro": _zeros((ncol, KM), device=device),
        "delp": delp,
        "ten_t": _zeros((ncol, KM), device=device),
        "ten_u": _zeros((ncol, KM), device=device),
        "ten_v": _zeros((ncol, KM), device=device),
        "ten_q": _zeros((ncol, KM, NTR_runtime), device=device),
        "prslp": prslp,
        "phil": phil,
        "tkeh": _full((ncol, KM), 0.1, device=device),
        "qtr": qtr,
        "dqtr": _zeros((ncol, KM, NN_runtime), device=device),
        "prevsq": prevsq,
        "q": q,
        "q1": q1,
        "t1": t1,
        "u1": u1,
        "v1": v1,
        "fscav": _full((NTC,), 0.1, device=device),
        "kcnv": _zeros((ncol,), device=device),
        "kbot": _zeros((ncol,), device=device),
        "ktop": _zeros((ncol,), device=device),
        "islimsk": islimsk,
        "dot": dot,
        "ud_mf": _zeros((ncol, KM), device=device),
        "dd_mf": _zeros((ncol, KM), device=device),
        "dt_mf": _zeros((ncol, KM), device=device),
        "cnvw": _zeros((ncol, KM), device=device),
        "cnvc": _zeros((ncol, KM), device=device),
        "qlcn": _zeros((ncol, KM), device=device),
        "qicn": _zeros((ncol, KM), device=device),
        "w_upi": _zeros((ncol, KM), device=device),
        "cf_upi": _zeros((ncol, KM), device=device),
        "cnv_mfd": _zeros((ncol, KM), device=device),
        "cnv_dqldt": _zeros((ncol, KM), device=device),
        "clcn": _zeros((ncol, KM), device=device),
        "cnv_fice": _zeros((ncol, KM), device=device),
        "cnv_ndrop": _zeros((ncol, KM), device=device),
        "cnv_nice": _zeros((ncol, KM), device=device),
        "sigmain": _zeros((ncol, KM), device=device),
        "sigmaout": _zeros((ncol, KM), device=device),
        "omegain": _zeros((ncol, KM), device=device),
        "omegaout": _zeros((ncol, KM), device=device),
        **one_d,
        **{name: _scalar(value, device=device) for name, value in scalar_fields.items()},
    }

    config: dict[str, Any] = {
        "im": ncol,
        "km": KM,
        "nn": NN_runtime,
        "ntr": NTR_runtime,
        "ntc": NTC,
        "itc": ITC,
        "ntk": NTK,
        "ncloud": NCLOUD,
        "ndtend": max(NDTEND, NTR_runtime + 3),
        "first_time_step": True,
        "restart": False,
        "hwrf_samfdeep": False,
        "progsigma": False,
        "progomega": False,
        "do_mynnedmf": True,
        "sigmab_coldstart": True,
        "do_ca": False,
        "ca_closure": False,
        "ca_entr": False,
        "ca_trigger": False,
        "mp_phys": 8,
        "mp_phys_mg": 99,
        # Legacy harness cloud-copy rule (GFS_DCNV_generic_post npdf3d/num_p3d/ncnvcld3d).
        "npdf3d": 3,
        "num_p3d": 4,
        "ncnvcld3d": 1,
        "trans_trac": True,
        "tend_opt_dcnv": 1,
        "imp_physics": "gfdl",
        "ldiag3d": True,
        "qdiag3d": True,
        "flag_for_dcnv_generic_tend": True,
        "ras": False,
        "cscnv": False,
        "satmedmf": False,
    }
    return SaSasDeepConvectionState(fields, config)


def initial_state_from_scenario_case(
    case: dict[str, Any],
    *,
    device: torch.device | str | None = None,
    ncol: int | None = None,
    ntrac: int = NTR,
) -> SaSasDeepConvectionState:
    """Build the deterministic DCNV-entry state for a scenario-corpus case.

    The adjustments intentionally mirror
    ``tools/run_sa_sas_deep_convection_coverage.py::scenario_adjustments`` and
    ``post_adjustments``.  The caller passes the parsed scenario mapping, not a
    case name, so config is explicit and auditable.
    """

    state = initial_state_case01_active_maritime_deep(device=device, ncol=ncol, ntrac=ntrac)
    fields = {name: tensor.clone() for name, tensor in state.fields.items()}
    cfg = dict(case.get("required_config") or {})
    families = set(case.get("branch_families") or [])
    config = {**state.config, **cfg}
    config["trans_trac"] = bool(cfg.get("trans_trac", False))

    if cfg.get("garea_mode") == "small":
        fields["garea"] = torch.full_like(fields["garea"], 1.0e6)
    elif cfg.get("garea_mode") == "large":
        fields["garea"] = torch.full_like(fields["garea"], 2.5e11)

    if cfg.get("fscav_nonzero"):
        fields["fscav"] = torch.full_like(fields["fscav"], 0.5)

    if cfg.get("ntk"):
        ntk_index = int(cfg["ntk"]) - 1
        if not 0 <= ntk_index < fields["qtr"].shape[2]:
            raise ValueError(f"scenario {case.get('id')} has invalid ntk={cfg['ntk']}")
        fields["qtr"][:, :, ntk_index] = 0.1

    if cfg.get("progsigma"):
        fields["sigmain"] = torch.full_like(fields["sigmain"], 0.06)
        fields["prevsq"] = fields["q1"] * 0.97

    if cfg.get("progomega"):
        fields["omegain"] = torch.full_like(fields["omegain"], -0.25)

    if "no_lfc_early_return" in families:
        fields["q1"] = torch.full_like(fields["q1"], 1.0e-8)
        fields["q"] = fields["q1"].clone()
        fields["qtr"][:, :, 0] = fields["q1"]
        fields["dot"] = torch.full_like(fields["dot"], 0.02)

    if "rh_rejection" in families:
        fields["q1"] = torch.maximum(fields["q1"] * 0.05, torch.full_like(fields["q1"], 1.0e-8))
        fields["q"] = fields["q1"].clone()
        fields["qtr"][:, :, 0] = fields["q1"]

    if "cin_pressure_depth_rejection" in families:
        fields["dot"] = torch.full_like(fields["dot"], 0.08)

    if "cloud_work_rejection" in families:
        fields["t1"] = fields["t1"] + 8.0

    if any(str(family).startswith("cellular_automata") for family in families):
        fields["ca_deep"] = torch.full_like(fields["ca_deep"], 2.0)

    # Harness post adjustments: the scenario's microphysics choice sets the cloud rule.
    if cfg.get("imp_physics") == "thompson":
        config.update({"npdf3d": 0, "ncnvcld3d": 1})
    elif cfg.get("imp_physics") == "gfdl":
        config.update({"npdf3d": 3, "num_p3d": 4})

    return SaSasDeepConvectionState(fields, config)


# G3.2 oracle lane (MPAS_GFS_v17_tempo_rrtmgp): MPAS tracer values of the approved cases.
ORACLE_MPAS_TRACER_VALUES = {
    "ntcw": 1.0e-5, "ntrw": 1.0e-6, "ntiw": 1.0e-6, "ntsw": 1.0e-6, "ntgl": 1.0e-6,
    "ntinc": 1.0e-6, "ntrnc": 1.0e-6, "ntke": 0.1, "ntoz": 1.0e-6,
}


def initial_state_oracle_mpas_case(
    case: dict[str, Any],
    *,
    reconciliation: bool,
    tracer_values: dict[str, float] | None = None,
    device: torch.device | str | None = None,
    ncol: int | None = None,
) -> SaSasDeepConvectionState:
    """DCNV-entry state of the G3.2 oracle lane for an approved scenario case.

    Mirrors ``tools/run_sa_sas_convection_oracle_g3_2_mint.py`` (deep, ``mpas_tempo``):
    the legacy profile and scenario adjustments, then the MPAS tracer state
    (``gq0`` with qv = q1), ``clw`` from the ``GFS_suite_interstitial_3`` rule, TEMPO,
    satmedmf/trans_trac, ``npdf3d=0, num_p3d=3, ncnvcld3d=1`` and ``ncloud=ncnd=5``.
    """

    from sa_sas_deep_convection_torch.samf_interstitials import (
        IMP_PHYSICS_CODES,
        build_clw,
        interface_pressure,
        mpas_tempo_layout,
    )

    legacy = initial_state_from_scenario_case(case, device=device, ncol=ncol)
    layout = mpas_tempo_layout()
    fields = {name: tensor.clone() for name, tensor in legacy.fields.items()}
    cfg = dict(case.get("required_config") or {})
    ncol_runtime, km = fields["q1"].shape
    dtype, dev = fields["q1"].dtype, fields["q1"].device
    gq0 = torch.zeros((ncol_runtime, km, layout.ntrac), dtype=dtype, device=dev)
    gq0[:, :, layout.index("ntqv") - 1] = fields["q1"]
    for name, value in (tracer_values or ORACLE_MPAS_TRACER_VALUES).items():
        gq0[:, :, layout.index(name) - 1] = value
    clw = build_clw(
        gq0,
        torch.zeros((ncol_runtime, km, layout.nn), dtype=dtype, device=dev),
        layout,
        imp_physics=IMP_PHYSICS_CODES["tempo"],
        transport=True,
    )
    fscav_value = 0.5 if cfg.get("fscav_nonzero") else 0.1
    fields.update(
        {
            "qtr": clw,
            "dqtr": torch.zeros_like(clw),
            "tmf": torch.zeros_like(clw),
            "ten_q": torch.zeros_like(gq0),
            "fscav": torch.full((layout.ntc,), fscav_value, dtype=dtype, device=dev),
            "gq0": gq0,
            "gt0": fields["t1"].clone(),
            "gu0": fields["u1"].clone(),
            "gv0": fields["v1"].clone(),
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
        "ncloud": 5,
        "mp_phys": IMP_PHYSICS_CODES["tempo"],
        "mp_phys_mg": IMP_PHYSICS_CODES["mg"],
        "imp_physics": IMP_PHYSICS_CODES["tempo"],
        "satmedmf": True,
        "trans_trac": True,
        "npdf3d": 0,
        "num_p3d": 3,
        "ncnvcld3d": 1,
        "ndtend": NDTEND,
        "samf_water_reconciliation": bool(reconciliation),
    }
    return SaSasDeepConvectionState(fields, config)


def _runs_through(through: str, segment_id: str) -> bool:
    return _SEGMENT_ORDER[through] >= _SEGMENT_ORDER[segment_id]


def unsupported_config_reasons(config: dict[str, Any], *, through: str = "V") -> tuple[str, ...]:
    """Return uncheckpointed target-side branch claims for the requested prefix."""

    reasons: list[str] = []
    # Tracer-index branches sealed by goldens: the legacy harness table (TKE slot 8,
    # chemistry 6:8) and the G3.2 oracle MPAS table (TKE slot 3, itc=0, ntc=nscav=7).
    sealed = {"legacy": (NTK, ITC, NTC), "mpas_tempo": (3, 0, 7)}
    layout_name = "mpas_tempo" if config.get("tracer_layout") is not None else "legacy"
    sealed_ntk, sealed_itc, sealed_ntc = sealed[layout_name]
    if _runs_through(through, "T") and int(config.get("ntk", sealed_ntk)) != sealed_ntk:
        reasons.append("non-default TKE tracer-index branch requires target checkpoints")
    if _runs_through(through, "T") and (
        int(config.get("itc", sealed_itc)) != sealed_itc or int(config.get("ntc", sealed_ntc)) not in
        ((0, sealed_ntc) if layout_name == "mpas_tempo" else (sealed_ntc,))
    ):
        reasons.append("non-default chemistry tracer-index branch requires target checkpoints")
    if _runs_through(through, "V"):
        post_variants = []
        if bool(config.get("ldiag3d", True)) is not True:
            post_variants.append("ldiag3d")
        if bool(config.get("qdiag3d", True)) is not True:
            post_variants.append("qdiag3d")
        if bool(config.get("flag_for_dcnv_generic_tend", True)) is not True:
            post_variants.append("flag_for_dcnv_generic_tend")
        if bool(config.get("ras", False)):
            post_variants.append("ras")
        if bool(config.get("cscnv", False)):
            post_variants.append("cscnv")
        if post_variants:
            reasons.append(
                "post-diagnostic variant is not sealed by the active-case V oracle: "
                + ", ".join(sorted(set(post_variants)))
            )
    return tuple(reasons)


def validate_supported_config(config: dict[str, Any], *, through: str = "V") -> None:
    reasons = unsupported_config_reasons(config, through=through)
    if reasons:
        raise NotImplementedError(
            "unsupported SA-SAS target config branch for the sealed PyTorch checkpoint path "
            f"through={through}: " + "; ".join(reasons)
        )


def set_segment_profiling_enabled(enabled: bool) -> bool:
    global _segment_profiling_enabled
    previous = _segment_profiling_enabled
    _segment_profiling_enabled = bool(enabled)
    return previous


def _run_segment(boundary_id: str, state: SaSasDeepConvectionState, runner: Any) -> SaSasDeepConvectionState:
    if not _segment_profiling_enabled:
        return runner(state)
    with torch.profiler.record_function(f"sa_sas.segment_{boundary_id}"):
        return runner(state)


def run_sa_sas_deep_convection(
    state: SaSasDeepConvectionState,
    *,
    through: str = "A",
    return_trace: bool = False,
    enforce_supported_config: bool = True,
) -> SaSasDeepConvectionState | tuple[SaSasDeepConvectionState, dict[str, dict[str, torch.Tensor]]]:
    """Run cumulative implemented SA-SAS deep-convection segments."""

    if through not in SEGMENT_IDS:
        raise ValueError(f"unknown SA-SAS deep-convection segment {through!r}")
    if through not in IMPLEMENTED_SEGMENTS:
        raise NotImplementedError(f"SA-SAS deep-convection segment {through!r} is not implemented yet")
    require_same_device(state.fields)
    if enforce_supported_config:
        validate_supported_config(state.config, through=through)
    if through == "V" and not return_trace:
        return run_sa_sas_deep_convection_production(state)
    trace: dict[str, dict[str, torch.Tensor]] = {}

    current = _run_segment("A", state, segment_a.run)
    if return_trace:
        trace["A"] = segment_a.checkpoint_fields(current)
    if through == "A":
        return (current, trace) if return_trace else current

    current = _run_segment("B", current, segment_b.run)
    if return_trace:
        trace["B"] = segment_b.checkpoint_fields(current)
    if through == "B":
        return (current, trace) if return_trace else current

    current = _run_segment("C", current, segment_c.run)
    if return_trace:
        trace["C"] = segment_c.checkpoint_fields(current)
    if through == "C":
        return (current, trace) if return_trace else current

    current = _run_segment("D", current, segment_d.run)
    if return_trace:
        trace["D"] = segment_d.checkpoint_fields(current)
    if through == "D":
        return (current, trace) if return_trace else current

    current = _run_segment("E", current, segment_e.run)
    if return_trace:
        trace["E"] = segment_e.checkpoint_fields(current)
    if through == "E":
        return (current, trace) if return_trace else current

    current = _run_segment("F", current, segment_f.run)
    if return_trace:
        trace["F"] = segment_f.checkpoint_fields(current)
    if through == "F":
        return (current, trace) if return_trace else current

    current = _run_segment("G", current, segment_g.run)
    if return_trace:
        trace["G"] = segment_g.checkpoint_fields(current)
    if through == "G":
        return (current, trace) if return_trace else current

    current = _run_segment("H", current, segment_h.run)
    if return_trace:
        trace["H"] = segment_h.checkpoint_fields(current)
    if through == "H":
        return (current, trace) if return_trace else current

    current = _run_segment("I", current, segment_i.run)
    if return_trace:
        trace["I"] = segment_i.checkpoint_fields(current)
    if through == "I":
        return (current, trace) if return_trace else current

    current = _run_segment("J", current, segment_j.run)
    if return_trace:
        trace["J"] = segment_j.checkpoint_fields(current)
    if through == "J":
        return (current, trace) if return_trace else current

    current = _run_segment("K", current, segment_k.run)
    if return_trace:
        trace["K"] = segment_k.checkpoint_fields(current)
    if through == "K":
        return (current, trace) if return_trace else current

    current = _run_segment("L", current, segment_l.run)
    if return_trace:
        trace["L"] = segment_l.checkpoint_fields(current)
    if through == "L":
        return (current, trace) if return_trace else current

    current = _run_segment("M", current, segment_m.run)
    if return_trace:
        trace["M"] = segment_m.checkpoint_fields(current)
    if through == "M":
        return (current, trace) if return_trace else current

    current = _run_segment("N", current, segment_n.run)
    if return_trace:
        trace["N"] = segment_n.checkpoint_fields(current)
    if through == "N":
        return (current, trace) if return_trace else current

    current = _run_segment("O", current, segment_o.run)
    if return_trace:
        trace["O"] = segment_o.checkpoint_fields(current)
    if through == "O":
        return (current, trace) if return_trace else current

    current = _run_segment("P", current, segment_p.run)
    if return_trace:
        trace["P"] = segment_p.checkpoint_fields(current)
    if through == "P":
        return (current, trace) if return_trace else current

    current = _run_segment("Q", current, segment_q.run)
    if return_trace:
        trace["Q"] = segment_q.checkpoint_fields(current)
    if through == "Q":
        return (current, trace) if return_trace else current

    current = _run_segment("R", current, segment_r.run)
    if return_trace:
        trace["R"] = segment_r.checkpoint_fields(current)
    if through == "R":
        return (current, trace) if return_trace else current

    current = _run_segment("S", current, segment_s.run)
    if return_trace:
        trace["S"] = segment_s.checkpoint_fields(current)
    if through == "S":
        return (current, trace) if return_trace else current

    current = _run_segment("T", current, segment_t.run)
    if return_trace:
        trace["T"] = segment_t.checkpoint_fields(current)
    if through == "T":
        return (current, trace) if return_trace else current

    current = _run_segment("U", current, segment_u.run)
    if return_trace:
        trace["U"] = segment_u.checkpoint_fields(current)
    if through == "U":
        return (current, trace) if return_trace else current

    current = _run_segment("V", current, segment_v.run)
    if return_trace:
        trace["V"] = segment_v.checkpoint_fields(current)
    if through == "V":
        return (current, trace) if return_trace else current

    raise NotImplementedError(f"SA-SAS deep-convection segment {through!r} is not implemented yet")


_SEGMENT_RUNNERS = {
    "A": segment_a.run, "B": segment_b.run, "C": segment_c.run, "D": segment_d.run, "E": segment_e.run,
    "F": segment_f.run, "G": segment_g.run, "H": segment_h.run, "I": segment_i.run, "J": segment_j.run,
    "K": segment_k.run, "L": segment_l.run, "M": segment_m.run, "N": segment_n.run, "O": segment_o.run,
    "P": segment_p.run, "Q": segment_q.run, "R": segment_r.run, "S": segment_s.run, "T": segment_t.run,
    "U": segment_u.run, "V": segment_v.run,
}


# Gathered columns that three or more compiled cloud-model kernels read: each is stored once
# instead of being gathered again inside every consumer. The others (one or two consumers, or
# constant workspace) stay fused into their consumers. Measured on the compiled deep
# apply_convection at x1.10242 (consumer kernels per stored column); the set changes
# only performance, not values.
_STORED_COLUMNS = frozenset({
    "buo", "c0t", "cnvflg", "cnvwt", "ctr", "ctro", "dbyo", "del", "dellal", "ecdo", "eta", "etad",
    "gdx", "hcdo", "hcko", "heo", "heso", "kb", "kbcon", "kbcon1", "kd94", "kmax", "ktcon", "ktop", "new_qtr",
    "pfld", "pwo", "q1", "qcdo", "qcko", "qeso", "qo", "qrcdo", "qrcko", "t1", "to", "ucdo", "uo", "vcdo", "vo",
    "wu2", "xlamddt", "xlamdet", "xlamud", "xlamue", "zi", "zo",
})


def run_sa_sas_deep_convection_compacted(
    state: SaSasDeepConvectionState,
    *,
    through: str,
    outputs: tuple[str, ...],
    fraction: float,
    region: str = "convection.deep",
) -> SaSasDeepConvectionState:
    """Run A-H on every column and the cloud model (I onwards) only on the triggered columns.

    As in ``samfdeepcnv_run``, segments I onwards act only on columns with ``cnvflg`` set after
    the trigger (segment H); other columns keep their values at H in every field the source
    writes back. The triggered columns are gathered into a buffer of ``fraction`` of the columns
    (``ufs_physics_common.compaction``); each computes from its own values, so the result does not
    depend on the buffer. Returns ``outputs`` only: per-column outputs equal the dense run bit for
    bit; batch scalars (``errflg``) come from the compacted run. A buffer too small for the
    triggered columns sets ``region``'s overflow bit.
    """
    from ufs_physics_common.compaction import run_compacted
    from sa_sas_deep_convection_torch.compile import run_compile_island

    if through not in _SEGMENT_ORDER or _SEGMENT_ORDER[through] <= _SEGMENT_ORDER["H"]:
        raise ValueError("deep compaction starts after the trigger (segment H)")
    require_same_device(state.fields)
    validate_supported_config(state.config, through=through)
    at_h = run_sa_sas_deep_convection(state, through="H", enforce_supported_config=False)
    fields = at_h.fields
    ncol = fields["cnvflg"].shape[0]
    columns = {k: v for k, v in fields.items() if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == ncol}
    shared = {k: v for k, v in fields.items() if k not in columns}
    missing = [name for name in outputs if name not in fields]
    if missing:
        raise KeyError(f"deep compaction outputs must exist at the trigger: {missing}")
    batch = {}

    def cloud_model(part: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        config = at_h.config
        if "im" in config:  # the declared column count is the buffer's
            config = {**config, "im": part["cnvflg"].shape[0]}
        part = {name: stored(value) if name in _STORED_COLUMNS else value for name, value in part.items()}
        current = SaSasDeepConvectionState({**shared, **part}, config, mutable=True)
        for boundary_id in SEGMENT_IDS[_SEGMENT_ORDER["H"] + 1 : _SEGMENT_ORDER[through] + 1]:
            current = _run_segment(boundary_id, current, _SEGMENT_RUNNERS[boundary_id])
        batch.update({name: current.require(name) for name in outputs if name not in columns})
        return {name: current.require(name) for name in outputs if name in columns}

    def tensor_call(fn, *args):
        return run_compile_island("g_ordered_chunks", fn.__name__, fn, *args)

    merged = run_compacted(cloud_model, columns, fields["cnvflg"] != 0.0, region=region,
                           fraction=fraction, tensor_call=tensor_call)
    return SaSasDeepConvectionState({name: batch[name] if name in batch else merged[name] for name in outputs},
                                    at_h.config)


def run_sa_sas_deep_convection_production(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    """Run the final A-V production path without checkpoint trace plumbing."""

    require_same_device(state.fields)
    device = next(iter(state.fields.values())).device if state.fields else torch.device("cpu")
    if device.type == "cuda" and not compile_categories_explicitly_set() and hasattr(torch, "compile"):
        with compile_categories(DEFAULT_COMPILE_CATEGORIES):
            return _run_sa_sas_deep_convection_production_mutable(state)
    return _run_sa_sas_deep_convection_production_mutable(state)


def _run_sa_sas_deep_convection_production_mutable(state: SaSasDeepConvectionState) -> SaSasDeepConvectionState:
    current = state.mutable_copy()
    current = _run_segment("A", current, segment_a.run)
    current = _run_segment("B", current, segment_b.run)
    current = _run_segment("C", current, segment_c.run)
    current = _run_segment("D", current, segment_d.run)
    current = _run_segment("E", current, segment_e.run)
    current = _run_segment("F", current, segment_f.run)
    current = _run_segment("G", current, segment_g.run)
    current = _run_segment("H", current, segment_h.run)
    current = _run_segment("I", current, segment_i.run)
    current = _run_segment("J", current, segment_j.run)
    current = _run_segment("K", current, segment_k.run)
    current = _run_segment("L", current, segment_l.run)
    current = _run_segment("M", current, segment_m.run)
    current = _run_segment("N", current, segment_n.run)
    current = _run_segment("O", current, segment_o.run)
    current = _run_segment("P", current, segment_p.run)
    current = _run_segment("Q", current, segment_q.run)
    current = _run_segment("R", current, segment_r.run)
    current = _run_segment("S", current, segment_s.run)
    current = _run_segment("T", current, segment_t.run)
    current = _run_segment("U", current, segment_u.run)
    current = _run_segment("V", current, segment_v.run)
    return current.immutable_copy()


def checkpoint_fields(state: SaSasDeepConvectionState, boundary_id: str) -> dict[str, torch.Tensor]:
    if boundary_id == "A":
        return segment_a.checkpoint_fields(state)
    if boundary_id == "B":
        return segment_b.checkpoint_fields(state)
    if boundary_id == "C":
        return segment_c.checkpoint_fields(state)
    if boundary_id == "D":
        return segment_d.checkpoint_fields(state)
    if boundary_id == "E":
        return segment_e.checkpoint_fields(state)
    if boundary_id == "F":
        return segment_f.checkpoint_fields(state)
    if boundary_id == "G":
        return segment_g.checkpoint_fields(state)
    if boundary_id == "H":
        return segment_h.checkpoint_fields(state)
    if boundary_id == "I":
        return segment_i.checkpoint_fields(state)
    if boundary_id == "J":
        return segment_j.checkpoint_fields(state)
    if boundary_id == "K":
        return segment_k.checkpoint_fields(state)
    if boundary_id == "L":
        return segment_l.checkpoint_fields(state)
    if boundary_id == "M":
        return segment_m.checkpoint_fields(state)
    if boundary_id == "N":
        return segment_n.checkpoint_fields(state)
    if boundary_id == "O":
        return segment_o.checkpoint_fields(state)
    if boundary_id == "P":
        return segment_p.checkpoint_fields(state)
    if boundary_id == "Q":
        return segment_q.checkpoint_fields(state)
    if boundary_id == "R":
        return segment_r.checkpoint_fields(state)
    if boundary_id == "S":
        return segment_s.checkpoint_fields(state)
    if boundary_id == "T":
        return segment_t.checkpoint_fields(state)
    if boundary_id == "U":
        return segment_u.checkpoint_fields(state)
    if boundary_id == "V":
        return segment_v.checkpoint_fields(state)
    raise NotImplementedError(f"SA-SAS deep-convection checkpoint {boundary_id!r} is not implemented yet")
