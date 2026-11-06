# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Per-run precision. Defaults are immutable; tensors carry the compute dtype."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from mpas_pytorch.constants import DTYPE
from mpas_pytorch.exceptions import MpasConfigError, MpasUnsupportedCapabilityError


def working_power(base: torch.Tensor, exponent: float) -> torch.Tensor:
    """Keep CPU FP32 scalar/SIMD power rounding independent of spatial batch size.

    CPU uses an FP64 power with a working-dtype exponent and rounds once.
    CUDA/MPS retain native FP32 powers. FP64 retains the qualified operation.
    """
    if base.dtype == torch.float32:
        power = torch.scalar_tensor(exponent, dtype=base.dtype, device=base.device)
        if base.device.type == "cpu":
            return torch.float_power(base, power).to(base.dtype)
        return torch.pow(base, power)
    return torch.pow(base, exponent)



def assert_async(predicate: torch.Tensor, message: str) -> None:
    """``torch._assert_async``; MPS has no kernel for it, so there the check reads the predicate.

    Raises ``RuntimeError(message)`` on every device, as ``torch._assert_async`` does.
    """
    if predicate.device.type == "mps":
        if not bool(predicate):
            raise RuntimeError(message)
        return
    torch._assert_async(predicate, message)

def resolve_dtype(dtype: torch.dtype | str = DTYPE) -> torch.dtype:
    if dtype in (torch.float32, "float32"):
        return torch.float32
    if dtype in (torch.float64, "float64"):
        return torch.float64
    raise MpasUnsupportedCapabilityError(
        f"Unsupported compute dtype {dtype!r}; use float32 or float64",
        capability="compute_dtype",
        requested=dtype,
    )


@dataclass(frozen=True)
class ExecutionPolicy:
    """Model-owned numerical placement and explicit differentiable transfers."""

    dtype: torch.dtype | str = DTYPE
    device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        dtype = resolve_dtype(self.dtype)
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda", "mps"):
            raise MpasConfigError(f"Unsupported compute device {device}")
        if device.type == "mps" and dtype != torch.float32:
            raise MpasConfigError("MPS execution requires float32")
        if device.type == "cpu":
            device = torch.device("cpu")
        elif device.type == "mps" and device.index is None:
            device = torch.device("mps", 0)
        elif device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "device", device)

    def as_dict(self) -> dict[str, str]:
        return {"dtype": str(self.dtype).removeprefix("torch."), "device": str(self.device)}

    @classmethod
    def from_tensors(cls, *sources: Mapping[str, Any]) -> "ExecutionPolicy":
        dtype = require_uniform_precision(*sources)
        devices = {value.device for source in sources for value in source.values()
                   if isinstance(value, torch.Tensor) and value.is_floating_point()}
        if len(devices) != 1:
            raise MpasConfigError("execution policy requires one floating-point device")
        return cls(dtype, devices.pop())

    def transfer(self, value: torch.Tensor) -> torch.Tensor:
        """Copy/cast at an explicit host boundary without breaking either AD mode.

        MPS cannot hold float64, including an intermediate in a combined to().
        Widen after leaving MPS; narrow before entering it. Integer fields keep
        their dtype. This operation does not grant runtime/release admission.
        """
        if value.is_complex():
            raise MpasConfigError("complex model transfers are unsupported")
        if not value.is_floating_point():
            return value.to(device=self.device)
        resolve_dtype(value.dtype)
        if value.device.type == "mps":
            return value.to(device=self.device).to(dtype=self.dtype)
        return value.to(dtype=self.dtype).to(device=self.device)


def floating_dtype(*sources: Any) -> torch.dtype:
    """Choose a reference dtype for allocations, including geometry-only helpers."""
    for source in sources:
        if isinstance(source, torch.Tensor) and source.is_floating_point():
            return resolve_dtype(source.dtype)
        if isinstance(source, Mapping):
            for key in ("u", "rho_zz", "theta", "rho", "dcEdge", "areaCell"):
                value = source.get(key)
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    return resolve_dtype(value.dtype)
            for value in source.values():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    return resolve_dtype(value.dtype)
        elif hasattr(source, "u"):
            return resolve_dtype(source.u.dtype)
    return DTYPE


def require_uniform_precision(*sources: Mapping[str, Any]) -> torch.dtype:
    """Reject mixed buffers before any conversion could conceal the mismatch."""
    dtype = floating_dtype(*sources)
    for source in sources:
        for key, value in source.items():
            if isinstance(value, torch.Tensor) and (
                value.is_complex()
                or (value.is_floating_point() and value.dtype != dtype)
            ):
                raise MpasUnsupportedCapabilityError(
                    f"{key}: mixed or unsupported compute dtype {value.dtype}; expected {dtype}",
                    capability="uniform_compute_dtype",
                    requested={"field": key, "dtype": value.dtype, "expected": dtype},
                )
    return dtype


def require_workspace_precision(
    workspace: Any, dtype: torch.dtype, device: torch.device
) -> None:
    """Reject reused scratch from another precision/device before any stepping."""
    if workspace is None:
        return

    def canonical_device(value: torch.device) -> torch.device:
        if value.type == "mps" and value.index is None:
            return torch.device("mps", 0)
        if value.type == "cuda" and value.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return value

    device = canonical_device(device)
    workspace_device = getattr(workspace, "device", None)
    if getattr(workspace, "dtype", None) != dtype or (
        not isinstance(workspace_device, torch.device)
        or canonical_device(workspace_device) != device
    ):
        raise MpasConfigError(f"workspace must use {dtype} on {device}")

    def check(values: Mapping[str, Any], prefix: str) -> None:
        for key, value in values.items():
            name = f"{prefix}.{key}"
            if isinstance(value, torch.Tensor):
                if (
                    value.device != device
                    or value.is_complex()
                    or (value.is_floating_point() and value.dtype != dtype)
                ):
                    raise MpasConfigError(
                        f"{name}: workspace precision/device mismatch"
                    )
            elif isinstance(value, SimpleNamespace):
                check(vars(value), name)
            elif isinstance(value, Mapping):
                check(value, name)

    check(vars(workspace), "workspace")
