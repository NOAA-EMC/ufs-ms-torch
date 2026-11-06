# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""On-device ownership of the consumed CCPP continuation state."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
import json

import torch
from mpas_pytorch.state import (
    _UFS_COMMIT_ENTRY, _LiveBoundarySnapshot, _TensorStamp, _live_boundary_matches, _live_boundary_snapshot,
)


def carry_members(state):
    """Flatten carry with structural tags; native ownership remains MPAS's job."""
    members = {}

    def visit(value, path):
        name = json.dumps(path, separators=(",", ":"))
        if isinstance(value, torch.Tensor):
            members[name] = value
        elif isinstance(value, _TensorStamp):
            # A stamp-bound commit receipt: bind its host identity (the tensors it names are
            # bound by MPAS's own commit checks).
            members[name] = json.dumps(["stamp", value.data_ptr, value.storage_offset,
                                        list(value.stride), value.version])
        elif is_dataclass(value) and not isinstance(value, type):
            members[name] = type(value).__module__ + "." + type(value).__qualname__
            for field in fields(value):
                visit(getattr(value, field.name), (*path, field.name))
        elif isinstance(value, Mapping):
            members[name] = "mapping"
            if any(type(key) is not str for key in value):
                raise TypeError("coupled carry requires string mapping keys")
            for key in sorted(value):
                visit(value[key], (*path, key))
        elif isinstance(value, (tuple, list)):
            members[name] = type(value).__name__
            for index, item in enumerate(value):
                visit(item, (*path, index))
        elif value is None or type(value) in (bool, int, float, str):
            members[name] = json.dumps([type(value).__name__, value], allow_nan=False)
        elif type(value) is bytes:
            members[name] = json.dumps(["bytes", value.hex()])
        else:
            raise TypeError(f"unsupported coupled carry at {name}: {type(value).__name__}")

    for field in fields(state):
        if field.name not in ("native_state", "continuation"):
            visit(getattr(state, field.name), (field.name,))
    visit(state.native_state.get(_UFS_COMMIT_ENTRY), ("native_commit",))
    return members


def bind_coupled_boundary(state):
    """Called by the completed-cycle owner, never by a restart reader/writer."""
    if state.step_index < 2:
        raise ValueError("coupled boundary requires a completed cycle")
    if _UFS_COMMIT_ENTRY not in state.native_state:
        raise ValueError("coupled boundary lacks its original native commit receipt")
    return replace(state, continuation=_live_boundary_snapshot(carry_members(state), all_keys=True, mode="bytes"))


def require_coupled_boundary(state):
    if state.continuation is None:
        raise ValueError("coupled boundary lacks its original carry receipt")
    torch._assert_async(_live_boundary_matches(state.continuation, carry_members(state),
        device=state.sec.device, all_keys=True), "coupled continuation carry was edited")


def portable_continuation(state, native_commit):
    """``state`` with its native commit entry replaced by ``native_commit``, the byte form a
    restart writes (``mpas_pytorch.state.boundary_snapshot_bytes`` checked every stamp against its
    unchanged live tensor). The carry receipt follows that one change: the live carry must match
    the original receipt first, and only the commit-entry members are re-bound, under the same
    member names; every other receipt entry is kept as it was."""
    require_coupled_boundary(state)
    portable = replace(state, native_state={**state.native_state, _UFS_COMMIT_ENTRY: native_commit})
    prefix = json.dumps(["native_commit"], separators=(",", ":"))[:-1]
    members = carry_members(portable)
    commit = {name: value for name, value in members.items() if name.startswith(prefix)}
    if set(members) != {entry[0] for entry in state.continuation.entries}:
        raise ValueError("a restart byte form must keep the carry members")
    fresh = {entry[0]: entry for entry in _live_boundary_snapshot(commit, all_keys=True, mode="bytes").entries}
    return replace(portable, continuation=_LiveBoundarySnapshot(
        tuple(fresh.get(entry[0], entry) for entry in state.continuation.entries)))
