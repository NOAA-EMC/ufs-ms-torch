# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Integrated SA-SAS deep-convection PyTorch driver for cumulative port gates."""

from __future__ import annotations

from typing import Any

import torch

from sa_sas_deep_convection_torch.constants import (
    IMPLEMENTED_SEGMENTS,
    ITC,
    NTC,
    NTK,
    SEGMENT_IDS,
)
from sa_sas_deep_convection_torch.ops import stored
from sa_sas_deep_convection_torch.segments import segment_a, segment_b, segment_c, segment_d, segment_e, segment_f, segment_g, segment_h, segment_i, segment_j, segment_k, segment_l, segment_m, segment_n, segment_o, segment_p, segment_q, segment_r, segment_s, segment_t, segment_u, segment_v
from sa_sas_deep_convection_torch.state import SaSasDeepConvectionState, require_same_device

_SEGMENT_ORDER = {segment_id: idx for idx, segment_id in enumerate(SEGMENT_IDS)}


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




_SEGMENT_RUNNERS = {
    "A": segment_a.run, "B": segment_b.run, "C": segment_c.run, "D": segment_d.run, "E": segment_e.run,
    "F": segment_f.run, "G": segment_g.run, "H": segment_h.run, "I": segment_i.run, "J": segment_j.run,
    "K": segment_k.run, "L": segment_l.run, "M": segment_m.run, "N": segment_n.run, "O": segment_o.run,
    "P": segment_p.run, "Q": segment_q.run, "R": segment_r.run, "S": segment_s.run, "T": segment_t.run,
    "U": segment_u.run, "V": segment_v.run,
}


def run_sa_sas_deep_convection(
    state: SaSasDeepConvectionState,
    *,
    through: str,
    enforce_supported_config: bool = True,
) -> SaSasDeepConvectionState:
    """Run segments A through ``through`` on every column."""

    if through not in SEGMENT_IDS:
        raise ValueError(f"unknown SA-SAS deep-convection segment {through!r}")
    if through not in IMPLEMENTED_SEGMENTS:
        raise NotImplementedError(f"SA-SAS deep-convection segment {through!r} is not implemented yet")
    require_same_device(state.fields)
    if enforce_supported_config:
        validate_supported_config(state.config, through=through)
    current = state
    for boundary_id in SEGMENT_IDS[: _SEGMENT_ORDER[through] + 1]:
        current = _SEGMENT_RUNNERS[boundary_id](current)
    return current


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
    missing = [name for name in outputs if name not in fields]
    if missing:
        raise KeyError(f"deep compaction outputs must exist at the trigger: {missing}")

    def is_column(value) -> bool:
        return isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == ncol

    # Each column tensor at H enters the buffer once: names bound to one tensor stay one tensor there, as in the
    # dense run (as in shallow compaction).
    columns: dict[str, torch.Tensor] = {}

    def slot(value: torch.Tensor) -> str:
        for key, seen in columns.items():
            if seen is value:
                return key
        key = f"column{len(columns)}"
        columns[key] = value
        return key

    slots = {name: slot(value) for name, value in fields.items() if is_column(value)}
    shared = {k: v for k, v in fields.items() if k not in slots}
    gathered = tuple(columns)
    # Non-triggered columns take their values at H: those enter as the merge's old values.
    kept = [name for name in outputs if name in slots]
    columns.update({f"at_h.{name}": fields[name] for name in kept})
    batch = {}

    def cloud_model(part: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        size = next(iter(part.values())).shape[0]
        config = at_h.config
        if "im" in config:  # the declared column count is the buffer's
            config = {**config, "im": size}
        named = {name: part[slots[name]] for name in slots}
        named = {name: stored(value) if name in _STORED_COLUMNS else value for name, value in named.items()}
        current = SaSasDeepConvectionState({**shared, **named}, config, mutable=True)
        for boundary_id in SEGMENT_IDS[_SEGMENT_ORDER["H"] + 1 : _SEGMENT_ORDER[through] + 1]:
            current = _SEGMENT_RUNNERS[boundary_id](current)
        batch.update({name: current.require(name) for name in outputs if name not in kept})
        return {f"at_h.{name}": current.require(name) for name in kept}

    def tensor_call(fn, *args):
        return run_compile_island("g_ordered_chunks", fn.__name__, fn, *args)

    merged = run_compacted(cloud_model, columns, fields["cnvflg"] != 0.0, region=region,
                           fraction=fraction, tensor_call=tensor_call, inputs=gathered)
    return SaSasDeepConvectionState({name: batch[name] if name in batch else merged[f"at_h.{name}"]
                                     for name in outputs}, at_h.config)


