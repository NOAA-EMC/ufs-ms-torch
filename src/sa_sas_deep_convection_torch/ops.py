# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Small tensor helpers for the SA-SAS torch port."""

from __future__ import annotations

import torch

from ufs_physics_common.compile_policy import stored  # noqa: F401  (segment D, compaction)


def sqrt_zero_grad(argument: torch.Tensor) -> torch.Tensor:
    """Return ``sqrt(argument)`` with a finite derivative at exact zero.

    Callers use this only for analytically non-negative expressions such as
    squared wind norms and turbulent kinetic energy. The forward value matches
    ``sqrt`` for positive inputs and exact zero; the zero branch avoids the
    singular derivative that otherwise poisons autodiff.
    """

    positive = argument > 0.0
    safe_argument = torch.where(positive, argument, 1.0)
    return torch.where(positive, torch.sqrt(safe_argument), 0.0)

