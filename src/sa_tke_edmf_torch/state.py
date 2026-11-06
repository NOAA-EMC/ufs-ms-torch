# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""State container and tensor checks for SA-TKE-EDMF."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class SaTkeEdmfState:
    fields: dict[str, torch.Tensor]
    config: dict[str, Any] = field(default_factory=dict)
    mutable: bool = False

    def require(self, name: str) -> torch.Tensor:
        try:
            return self.fields[name]
        except KeyError as exc:
            raise KeyError(f"missing SA-TKE-EDMF state field {name!r}") from exc

    def with_updates(self, updates: Mapping[str, torch.Tensor]) -> "SaTkeEdmfState":
        if self.mutable:
            self.fields.update(updates)
            return self
        fields = dict(self.fields)
        fields.update(updates)
        return SaTkeEdmfState(fields, self.config)

    def mutable_copy(self) -> "SaTkeEdmfState":
        return SaTkeEdmfState(dict(self.fields), self.config, mutable=True)

    def immutable_copy(self) -> "SaTkeEdmfState":
        return SaTkeEdmfState(dict(self.fields), self.config, mutable=False)


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
        raise ValueError(f"SA-TKE-EDMF tensors must share one device, got {sorted(map(str, devices))}")
    return next(iter(devices))


# The segments call the check by its historical name; it accepts both working dtypes.
require_float64 = require_real
