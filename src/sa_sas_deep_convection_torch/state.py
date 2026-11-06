# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""State container and tensor checks for SA-SAS deep convection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class SaSasDeepConvectionState:
    fields: dict[str, torch.Tensor]
    config: dict[str, Any] = field(default_factory=dict)
    mutable: bool = False

    def require(self, name: str) -> torch.Tensor:
        try:
            return self.fields[name]
        except KeyError as exc:
            raise KeyError(f"missing SA-SAS deep-convection state field {name!r}") from exc

    def with_updates(self, updates: Mapping[str, torch.Tensor]) -> "SaSasDeepConvectionState":
        if self.mutable:
            self.fields.update(updates)
            return self
        fields = dict(self.fields)
        fields.update(updates)
        return SaSasDeepConvectionState(fields, self.config)

    def mutable_copy(self) -> "SaSasDeepConvectionState":
        return SaSasDeepConvectionState(dict(self.fields), self.config, mutable=True)

    def immutable_copy(self) -> "SaSasDeepConvectionState":
        return SaSasDeepConvectionState(dict(self.fields), self.config, mutable=False)


def require_real(name: str, tensor: torch.Tensor, shape: tuple[int, ...] | None = None) -> None:
    """``tensor`` is in a working dtype (float64, the parity dtype, or float32) with ``shape`` when given."""
    if tensor.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be torch.float32 or torch.float64, got {tensor.dtype}")
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def require_same_device(fields: Mapping[str, torch.Tensor]) -> torch.device:
    devices = {tensor.device for tensor in fields.values()}
    if not devices:
        return torch.device("cpu")
    if len(devices) != 1:
        raise ValueError(f"SA-SAS deep-convection tensors must share one device, got {sorted(map(str, devices))}")
    return next(iter(devices))


def runtime_dimensions(state: SaSasDeepConvectionState) -> tuple[int, int, int, int, int]:
    """Validate immutable extents from current tensors, without host tensor reads.

    Without ``config['tracer_layout']`` the frozen legacy harness table applies
    (``nn = ntr + 2``, TKE slot 8, chemistry slots 6:8). With a layout (G3.2, e.g. the
    oracle MPAS table) ``nn``, ``ntr``, ``ntk``, ``itc`` and ``ntc`` must equal the
    layout's CCPP values; ``ntr`` (``nsamftrac``) is then independent of ``nn``.
    """
    from sa_sas_deep_convection_torch.samf_interstitials import layout_from_config

    q1, qtr = state.require("q1"), state.require("qtr")
    if q1.ndim != 2 or qtr.ndim != 3 or qtr.shape[:2] != q1.shape:
        raise ValueError("deep q1/qtr require consistent [ncol,nz,(species)] shapes")
    ncol, nz = q1.shape
    nn = qtr.shape[2]
    fscav = state.require("fscav")
    if ncol < 1 or nz < 2 or fscav.ndim != 1:
        raise ValueError("deep requires positive columns, at least two levels and a chemistry axis")
    ntc = fscav.shape[0]
    layout = layout_from_config(state.config.get("tracer_layout"))
    itc = int(state.config.get("itc", 6 if layout is None else layout.itc))
    ntk = int(state.config.get("ntk", 8 if layout is None else layout.ntk))
    if layout is None:
        ntr = nn - 2
        if ntr < 8:
            raise ValueError("deep requires positive columns, at least two levels and the frozen eight-tracer layout")
    else:
        ntr = layout.ntr
        if nn != layout.nn:
            raise ValueError(f"deep qtr axis {nn} disagrees with tracer layout nn={layout.nn}")
    for name, actual in (("im", ncol), ("km", nz), ("nn", nn), ("ntr", ntr), ("ntc", ntc)):
        if name in state.config and (type(state.config[name]) is not int or state.config[name] != actual):
            raise ValueError(f"deep config {name} disagrees with physical tensor extent {actual}")
    if layout is None:
        if ntk != 8 or itc != 6 or ntc != 3 or itc - 1 + ntc > nn:
            raise ValueError("deep runtime retains source TKE slot8 and chemistry slots6:8")
    elif (ntk, itc, ntc) != (layout.ntk, layout.itc, layout.ntc) or ntr + 2 > nn:
        raise ValueError("deep ntk/itc/ntc disagree with the tracer layout")
    if "dtend" in state.fields:
        dtend = state.require("dtend")
        ntrac = ntr if layout is None else layout.ntrac
        if dtend.ndim != 3 or dtend.shape[:2] != (ncol, nz) or dtend.shape[2] < ntrac + 3:
            raise ValueError("deep dtend needs matching columns/levels and all source diagnostic slots")
    return ncol, nz, nn, ntr, ntc


def tracer_layout(state: SaSasDeepConvectionState):
    """The resolved convective tracer layout (legacy harness table when undeclared)."""
    from sa_sas_deep_convection_torch.samf_interstitials import layout_from_config, legacy_layout

    layout = layout_from_config(state.config.get("tracer_layout"))
    if layout is not None:
        return layout
    nn = state.require("qtr").shape[2]
    return legacy_layout(
        nn - 2,
        indices={"ntqv": 1, "ntcw": 2, "ntiw": 3, "ntrw": 4, "ntsw": 5},
        ntk=int(state.config.get("ntk", 8)),
        itc=int(state.config.get("itc", 6)),
        ntc=state.require("fscav").shape[0],
    )
