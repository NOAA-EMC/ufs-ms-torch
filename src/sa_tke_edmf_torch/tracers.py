# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Tracer-layout rules of GFS_PBL_generic_pre/post and the satmedmfvdifq index arguments.

Without ``config["tracer_layout"]`` the legacy profile layout applies (species slots
from ``constants``, TKE last, ``vdftra = qgrs``). With a layout mapping (1-based
indices ``ntqv ntcw ntiw ntrw ntsw ntgl ntinc ntrnc ntke ntoz``) the oracle rules
apply: when ``nvdiff == ntrac``, ``ntkev = nvdiff /= ntke`` and ``imp_physics`` is
Thompson or TEMPO, ``vdftra`` is packed in a fixed 9-species order plus TKE at
``ntkev`` and ``ten_q`` is unpacked back to the global slots.

All indices are Python integers taken from configuration, never from tensor values.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sa_tke_edmf_torch.constants import NTCW, NTIW, NTOZ, NTQV, NTRW

IMP_PHYSICS_THOMPSON = 8
IMP_PHYSICS_TEMPO = 88

# MPAS tracer table of the GFS v17 TEMPO+RRTMGP oracle lane:
# qv qc qr qi qs qg ni nr sgs_tke o3mr.
MPAS_TEMPO_TRACER_LAYOUT = {
    "ntqv": 1,
    "ntcw": 2,
    "ntrw": 3,
    "ntiw": 4,
    "ntsw": 5,
    "ntgl": 6,
    "ntinc": 7,
    "ntrnc": 8,
    "ntke": 9,
    "ntoz": 10,
}

# GFS_PBL_generic_pre.F90 Thompson-or-TEMPO branch: vdftra(:,:,1..9); TKE goes to ntkev.
THOMPSON_TEMPO_PACK_ORDER = ("ntqv", "ntcw", "ntiw", "ntrw", "ntsw", "ntgl", "ntinc", "ntrnc", "ntoz")


def _layout(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return config.get("tracer_layout")


def uses_thompson_tempo_pack(config: Mapping[str, Any], nvdiff: int) -> bool:
    """True when the oracle pre/post take the packed Thompson-or-TEMPO branch.

    The port always has ``nvdiff == ntrac`` and ``satmedmf``; ``ntkev = nvdiff``.
    """

    layout = _layout(config)
    if layout is None:
        return False
    imp_physics = int(config.get("imp_physics", -1))
    return int(layout["ntke"]) != nvdiff and imp_physics in (IMP_PHYSICS_THOMPSON, IMP_PHYSICS_TEMPO)


def pack_indices(config: Mapping[str, Any], nvdiff: int) -> tuple[list[int], int]:
    """Zero-based global slots for vdftra slots 1..9, and the zero-based global TKE slot."""

    layout = _layout(config)
    if layout is None:
        raise ValueError("pack_indices requires config['tracer_layout']")
    source = [int(layout[name]) - 1 for name in THOMPSON_TEMPO_PACK_ORDER]
    ntke = int(layout["ntke"]) - 1
    if any(slot < 0 or slot >= nvdiff for slot in (*source, ntke)) or len(set(source)) != len(source) or ntke in source:
        raise ValueError(f"invalid Thompson-or-TEMPO/SATMEDMF tracer layout {dict(layout)!r} for nvdiff={nvdiff}")
    if len(source) >= nvdiff:
        raise ValueError(f"nvdiff={nvdiff} leaves no TKE slot after the {len(source)} packed species")
    return source, ntke


def ozone_index(config: Mapping[str, Any]) -> int:
    """1-based ``ntoz`` of the global tracer array."""

    layout = _layout(config)
    return NTOZ if layout is None else int(layout["ntoz"])


def specific_humidity_index(config: Mapping[str, Any]) -> int:
    layout = _layout(config)
    return NTQV if layout is None else int(layout["ntqv"])


def satmedmf_hydrometeor_indices(config: Mapping[str, Any], nvdiff: int) -> tuple[int, int, int]:
    """1-based ``(ntcw, ntiw, ntrw)`` passed to satmedmfvdifq (CCPP ``ntcwx/ntiwx/ntrwx``).

    As in ``CCPP_typedefs.F90``: under the packed
    Thompson-or-TEMPO branch satmedmf indexes the packed vdftra, so the indices are
    the packed positions of ``ntcw/ntiw/ntrw`` (2/3/4); otherwise, with
    ``nvdiff == ntrac``, the global ``ntcw/ntiw/ntrw``.
    """

    layout = _layout(config)
    if layout is None:
        return NTCW, NTIW, NTRW
    if uses_thompson_tempo_pack(config, nvdiff):
        return tuple(THOMPSON_TEMPO_PACK_ORDER.index(name) + 1 for name in ("ntcw", "ntiw", "ntrw"))
    return int(layout["ntcw"]), int(layout["ntiw"]), int(layout["ntrw"])
