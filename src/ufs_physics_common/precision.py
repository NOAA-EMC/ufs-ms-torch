# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Power operands in the state precision, without tensor host reads."""

import torch


def working_power(base: torch.Tensor | float, exponent: torch.Tensor | float) -> torch.Tensor:
    """Evaluate a real power consistently across batch boundaries and devices.

    FP32 scalar/vector pow differ both in exponent precision and evaluation. On CPU and CUDA, round the
    exponent to the state dtype, evaluate in FP64, then round once (one form on both devices), so
    FP32 is mixed precision at these sensitive powers; on CUDA the FP64 power runs at the device's FP64
    throughput. MPS cannot hold FP64 and keeps the native FP32 power, as ``mpas_pytorch.precision.working_power``
    does. The casts remain on-device and preserve JVP/VJP. No tensor values reach the host.
    FP64 operations and their special integer-power paths remain unchanged.
    """
    reference = base if isinstance(base, torch.Tensor) else exponent
    if not isinstance(reference, torch.Tensor):
        raise TypeError("working_power requires a tensor operand")
    if reference.dtype == torch.float32:
        power, value = (_operand(operand, reference) for operand in (exponent, base))
        if reference.device.type == "mps":
            return torch.pow(value, power)
        return torch.float_power(value, power).to(reference.dtype)
    if reference.dtype == torch.float64:
        return torch.pow(base, exponent)
    raise ValueError("power literals require float32 or float64")


def _operand(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    """``value`` in the dtype and on the device of ``reference``. A Python number becomes a device fill
    (``torch.full``, rounded to the dtype as ``torch.as_tensor`` rounds it): no host-to-device copy, which a CUDA
    Graph capture forbids."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=reference.dtype, device=reference.device)
    return torch.full((), value, dtype=reference.dtype, device=reference.device)
