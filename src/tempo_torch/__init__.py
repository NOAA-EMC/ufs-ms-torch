# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""PyTorch-native TEMPO microphysics."""

from . import torch_native

run_tempo_native = torch_native.run_tempo_native

__all__ = [
    "run_tempo_native",
    "torch_native",
]
