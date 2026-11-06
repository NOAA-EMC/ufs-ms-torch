# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Power operands in the state precision, without tensor host reads."""

import torch


def working_power(base: torch.Tensor | float, exponent: torch.Tensor | float) -> torch.Tensor:
    """Evaluate a real power consistently across CPU batch boundaries.

    FP32 scalar/vector pow differ both in exponent precision and evaluation.
    On CPU, round the exponent to the state dtype, evaluate in FP64, then round once.
    CPU FP32 is therefore mixed precision at these sensitive powers. The casts
    remain on-device and preserve JVP/VJP. No tensor values reach the host.
    FP64 operations and their special integer-power paths remain unchanged.
    """
    reference = base if isinstance(base, torch.Tensor) else exponent
    if not isinstance(reference, torch.Tensor):
        raise TypeError("working_power requires a tensor operand")
    if reference.dtype == torch.float32:
        power = torch.as_tensor(exponent, dtype=reference.dtype, device=reference.device)
        value = torch.as_tensor(base, dtype=reference.dtype, device=reference.device)
        if reference.device.type == "cpu":
            return torch.float_power(value, power).to(reference.dtype)
        return torch.pow(value, power)
    if reference.dtype == torch.float64:
        return torch.pow(base, exponent)
    raise ValueError("power literals require float32 or float64")
