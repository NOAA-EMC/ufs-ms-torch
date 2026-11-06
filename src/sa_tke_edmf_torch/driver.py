# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Integrated SA-TKE-EDMF PyTorch driver for cumulative port gates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from sa_tke_edmf_torch.compile import DEFAULT_COMPILE_CATEGORIES, compile_categories, compile_categories_explicitly_set
from sa_tke_edmf_torch.constants import (
    IMPLEMENTED_SEGMENTS,
    IM,
    KM,
    NDTEND,
    NTCHEM,
    NTOZ,
    NTRAC,
    NTIW,
    NTKE,
    NTCW,
    NTQV,
    NTRW,
    NTSW,
    SEGMENT_IDS,
)
from sa_tke_edmf_torch.segments import (
    segment_a,
    segment_b,
    segment_c,
    segment_d,
    segment_e,
    segment_f,
    segment_g,
    segment_h,
    segment_i,
    segment_j,
    segment_k,
    segment_l,
    segment_m,
    segment_n,
    segment_o,
    segment_p,
    segment_q,
    segment_r,
    segment_s,
    segment_t,
)
from sa_tke_edmf_torch.state import SaTkeEdmfState, require_same_device


def _to_tensor(values: Any, *, device: torch.device | str | None) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float64, device=device)


def _replicate_column(values: Any, *, device: torch.device | str | None, ncol: int) -> torch.Tensor:
    vertical = _to_tensor(values, device=device)
    if vertical.ndim != 1:
        raise ValueError("profile columns must be one-dimensional")
    return vertical.unsqueeze(0).repeat(ncol, 1)


def _replicate_interface(values: Any, *, device: torch.device | str | None, ncol: int) -> torch.Tensor:
    vertical = _to_tensor(values, device=device)
    if vertical.ndim != 1:
        raise ValueError("profile interfaces must be one-dimensional")
    return vertical.unsqueeze(0).repeat(ncol, 1)


def initial_state_from_materialized_profile(
    profile: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
    ncol: int | None = None,
    ntrac: int = NTRAC,
) -> SaTkeEdmfState:
    """Keep species slots 1-7, passive slots before TKE, and TKE last as Fortran requires."""

    ncol = IM if ncol is None else int(ncol)
    if ncol <= 0:
        raise ValueError(f"ncol must be positive, got {ncol}")
    temperature = _to_tensor(profile["temperature"], device=device)
    if temperature.ndim != 1 or temperature.shape[0] < 2:
        raise ValueError("temperature must be a one-dimensional profile with at least two levels")
    KM = temperature.shape[0]
    if not isinstance(ntrac, int) or ntrac < NTRAC:
        raise ValueError(f"ntrac must retain the eight declared species slots, got {ntrac}")
    NTKE = ntrac
    for name in ("qv", "ql", "qi", "ozone", "tke", "u", "v", "pressure", "del", "prslk", "phil"):
        if _to_tensor(profile[name], device=device).shape != (KM,):
            raise ValueError(f"{name} must have shape ({KM},)")
    if "chemistry" in profile and _to_tensor(profile["chemistry"], device=device).shape != (KM,):
        raise ValueError(f"chemistry must have shape ({KM},)")
    for name in ("prsi", "phii"):
        if _to_tensor(profile[name], device=device).shape != (KM + 1,):
            raise ValueError(f"{name} must have shape ({KM + 1},)")

    q1 = torch.zeros((ncol, KM, ntrac), dtype=torch.float64, device=device)
    q1[:, :, NTQV - 1] = _replicate_column(profile["qv"], device=device, ncol=ncol)
    q1[:, :, NTCW - 1] = _replicate_column(profile["ql"], device=device, ncol=ncol)
    q1[:, :, NTIW - 1] = _replicate_column(profile["qi"], device=device, ncol=ncol)
    q1[:, :, NTRW - 1] = 0.0
    q1[:, :, NTSW - 1] = 0.0
    q1[:, :, NTOZ - 1] = _replicate_column(profile["ozone"], device=device, ncol=ncol)
    q1[:, :, NTCHEM - 1] = _replicate_column(profile.get("chemistry", profile["ozone"]), device=device, ncol=ncol)
    q1[:, :, NTKE - 1] = _replicate_column(profile["tke"], device=device, ncol=ncol)

    t1 = _replicate_column(profile["temperature"], device=device, ncol=ncol)
    u1 = _replicate_column(profile["u"], device=device, ncol=ncol)
    v1 = _replicate_column(profile["v"], device=device, ncol=ncol)
    spd1 = torch.maximum(
        torch.sqrt(u1[:, 0] * u1[:, 0] + v1[:, 0] * v1[:, 0]),
        torch.full((ncol,), 0.1, dtype=torch.float64, device=device),
    )
    usfco = torch.zeros((ncol,), dtype=torch.float64, device=device)
    vsfco = torch.zeros((ncol,), dtype=torch.float64, device=device)
    if bool((profile.get("config") or {}).get("use_oceanuv", False)):
        usfco = 0.2 * u1[:, 0]
        vsfco = 0.2 * v1[:, 0]
    psk = torch.full((ncol,), float(profile["psk"]), dtype=torch.float64, device=device)
    prsl = _replicate_column(profile["pressure"], device=device, ncol=ncol)
    prsi = _replicate_interface(profile["prsi"], device=device, ncol=ncol)
    delp = _replicate_column(profile["del"], device=device, ncol=ncol)
    prslk = _replicate_column(profile["prslk"], device=device, ncol=ncol)
    phii = _replicate_interface(profile["phii"], device=device, ncol=ncol)
    phil = _replicate_column(profile["phil"], device=device, ncol=ncol)
    one_d = {
        name: torch.full((ncol,), float(profile[name]), dtype=torch.float64, device=device)
        for name in ("tsea", "heat", "evap", "stress", "zorl")
    }
    canopy = {
        name: torch.full(
            (ncol,),
            float((profile.get("canopy") or {}).get(name, 0.0)),
            dtype=torch.float64,
            device=device,
        )
        for name in ("claie", "cfch", "cfrt", "cclu", "cpopu")
    }
    one_d["rbsoil"] = torch.full((ncol,), -0.1, dtype=torch.float64, device=device)
    zeros_3 = torch.zeros_like(q1)
    weak_forcing = torch.full((ncol, KM), 1.0e-4, dtype=torch.float64, device=device)

    return SaTkeEdmfState(
        {
            "q1": q1,
            "qgrs": q1.clone(),
            "rtg": q1.clone(),
            "vdftra": zeros_3,
            "gq0": q1.clone(),
            "t1": t1,
            "u1": u1,
            "v1": v1,
            "spd1": spd1,
            "usfco": usfco,
            "vsfco": vsfco,
            "psk": psk,
            "prsl": prsl,
            "prsi": prsi,
            "del": delp,
            "prslk": prslk,
            "phii": phii,
            "phil": phil,
            "garea": torch.full((ncol,), float(profile["area"]), dtype=torch.float64, device=device),
            "delt": torch.full((ncol,), 60.0, dtype=torch.float64, device=device),
            **one_d,
            **canopy,
            "gt0": t1.clone(),
            "gu0": u1.clone(),
            "gv0": v1.clone(),
            "tgrs": t1.clone(),
            "ugrs": u1.clone(),
            "vgrs": v1.clone(),
            "tkeh": torch.clamp_min(q1[:, :, NTKE - 1], 1.0e-9),
            "def_1": weak_forcing.clone(),
            "def_2": weak_forcing.clone(),
            "def_3": weak_forcing.clone(),
            "ten_t": torch.zeros((ncol, KM), dtype=torch.float64, device=device),
            "ten_u": torch.zeros((ncol, KM), dtype=torch.float64, device=device),
            "ten_v": torch.zeros((ncol, KM), dtype=torch.float64, device=device),
            "dtend": torch.zeros((ncol, KM, NDTEND), dtype=torch.float64, device=device),
            "tracer_index_policy": torch.full((ncol,), float(NTQV), dtype=torch.float64, device=device),
        },
        dict(profile.get("config") or {}),
    )


def run_sa_tke_edmf(
    state: SaTkeEdmfState,
    *,
    through: str = "A",
    return_trace: bool = False,
) -> SaTkeEdmfState | tuple[SaTkeEdmfState, dict[str, dict[str, torch.Tensor]]]:
    """Run cumulative implemented segments from the frozen initial state."""

    if through not in SEGMENT_IDS:
        raise ValueError(f"unknown SA-TKE-EDMF segment {through!r}")
    require_same_device(state.fields)
    if through == "T" and not return_trace:
        return run_sa_tke_edmf_production(state)
    trace: dict[str, dict[str, torch.Tensor]] = {}

    current = segment_a.run(state)
    if return_trace:
        trace["A"] = segment_a.checkpoint_fields(current)
    if through == "A":
        return (current, trace) if return_trace else current

    current = segment_b.run(current)
    if return_trace:
        trace["B"] = segment_b.checkpoint_fields(current)
    if through == "B":
        return (current, trace) if return_trace else current

    current = segment_c.run(current)
    if return_trace:
        trace["C"] = segment_c.checkpoint_fields(current)
    if through == "C":
        return (current, trace) if return_trace else current

    current = segment_d.run(current)
    if return_trace:
        trace["D"] = segment_d.checkpoint_fields(current)
    if through == "D":
        return (current, trace) if return_trace else current

    current = segment_e.run(current)
    if return_trace:
        trace["E"] = segment_e.checkpoint_fields(current)
    if through == "E":
        return (current, trace) if return_trace else current

    current = segment_f.run(current)
    if return_trace:
        trace["F"] = segment_f.checkpoint_fields(current)
    if through == "F":
        return (current, trace) if return_trace else current

    current = segment_g.run(current)
    if return_trace:
        trace["G"] = segment_g.checkpoint_fields(current)
    if through == "G":
        return (current, trace) if return_trace else current

    current = segment_h.run(current)
    if return_trace:
        trace["H"] = segment_h.checkpoint_fields(current)
    if through == "H":
        return (current, trace) if return_trace else current

    current = segment_i.run(current)
    if return_trace:
        trace["I"] = segment_i.checkpoint_fields(current)
    if through == "I":
        return (current, trace) if return_trace else current

    current = segment_j.run(current)
    if return_trace:
        trace["J"] = segment_j.checkpoint_fields(current)
    if through == "J":
        return (current, trace) if return_trace else current

    current = segment_k.run(current)
    if return_trace:
        trace["K"] = segment_k.checkpoint_fields(current)
    if through == "K":
        return (current, trace) if return_trace else current

    current = segment_l.run(current)
    if return_trace:
        trace["L"] = segment_l.checkpoint_fields(current)
    if through == "L":
        return (current, trace) if return_trace else current

    current = segment_m.run(current)
    if return_trace:
        trace["M"] = segment_m.checkpoint_fields(current)
    if through == "M":
        return (current, trace) if return_trace else current

    current = segment_n.run(current)
    if return_trace:
        trace["N"] = segment_n.checkpoint_fields(current)
    if through == "N":
        return (current, trace) if return_trace else current

    current = segment_o.run(current)
    if return_trace:
        trace["O"] = segment_o.checkpoint_fields(current)
    if through == "O":
        return (current, trace) if return_trace else current

    current = segment_p.run(current)
    if return_trace:
        trace["P"] = segment_p.checkpoint_fields(current)
    if through == "P":
        return (current, trace) if return_trace else current

    current = segment_q.run(current)
    if return_trace:
        trace["Q"] = segment_q.checkpoint_fields(current)
    if through == "Q":
        return (current, trace) if return_trace else current

    current = segment_r.run(current)
    if return_trace:
        trace["R"] = segment_r.checkpoint_fields(current)
    if through == "R":
        return (current, trace) if return_trace else current

    current = apply_canopy_chemistry_tendency(current)
    current = segment_s.run(current)
    if return_trace:
        trace["S"] = segment_s.checkpoint_fields(current)
    if through == "S":
        return (current, trace) if return_trace else current

    current = segment_t.run(current)
    if return_trace:
        trace["T"] = segment_t.checkpoint_fields(current)
    if through == "T":
        return (current, trace) if return_trace else current

    raise RuntimeError(f"unreachable SA-TKE-EDMF segment dispatch for {through!r}; implemented={IMPLEMENTED_SEGMENTS}")


def run_sa_tke_edmf_production(state: SaTkeEdmfState) -> SaTkeEdmfState:
    """Run the final A-T production path without checkpoint trace plumbing."""

    require_same_device(state.fields)
    if not compile_categories_explicitly_set() and hasattr(torch, "compile"):
        with compile_categories(DEFAULT_COMPILE_CATEGORIES):
            return _run_sa_tke_edmf_production_mutable(state)
    return _run_sa_tke_edmf_production_mutable(state)


def _run_sa_tke_edmf_production_mutable(state: SaTkeEdmfState) -> SaTkeEdmfState:
    current = state.mutable_copy()
    current = segment_a.run(current)
    current = segment_b.run(current)
    current = segment_c.run(current)
    current = segment_d.run(current)
    current = segment_e.run(current)
    current = segment_f.run(current)
    current = segment_g.run(current)
    current = segment_h.run(current)
    current = segment_i.run(current)
    current = segment_j.run(current)
    current = segment_k.run(current)
    current = segment_l.run(current)
    current = segment_m.run(current)
    current = segment_n.run(current)
    current = segment_o.run(current)
    current = segment_p.run(current)
    current = segment_q.run(current)
    current = segment_r.run(current)
    current = apply_canopy_chemistry_tendency(current)
    current = segment_s.run(current)
    current = segment_t.run(current)
    return current.immutable_copy()


def apply_canopy_chemistry_tendency(state: SaTkeEdmfState) -> SaTkeEdmfState:
    """Apply the active canopy chemistry update between R and S."""

    if not (bool(state.config.get("do_canopy")) and bool(state.config.get("cplaqm"))):
        return state
    rtg = state.require("rtg").clone()
    cfch = state.require("cfch")
    chemistry_tendency = state.fields.get("rtg_canopy_chemistry")
    if chemistry_tendency is None:
        return state
    rtg[:, :, NTCHEM - 1] = torch.where(
        (cfch > 0.0)[:, None],
        chemistry_tendency,
        rtg[:, :, NTCHEM - 1],
    )
    return state.with_updates({"rtg": rtg})


def checkpoint_fields(state: SaTkeEdmfState, boundary_id: str) -> dict[str, torch.Tensor]:
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
    raise ValueError(f"unknown SA-TKE-EDMF checkpoint boundary {boundary_id!r}")
