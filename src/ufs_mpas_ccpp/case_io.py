# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Forecast inputs: native NetCDF runtime directories or existing prepared cases."""

from pathlib import Path
import tempfile

import torch


def load_cpu_case(path, *, physics_overrides=None):
    """Bootstrap native inputs on CPU before the existing precision/partition path.

    Only root calls this in a distributed forecast. Setup temporaries never become
    forecast products. Legacy prepared files remain readable for regression oracles.
    """
    path = Path(path)
    if path.is_dir():
        from ufs_mpas_ccpp.native_case import build_native_case

        with tempfile.TemporaryDirectory(prefix="coupled-bootstrap-") as scratch:
            return build_native_case(path, Path(scratch), physics_overrides=physics_overrides)
    return torch.load(path, map_location="cpu", weights_only=False)
