# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Locations for frozen, radiation-owned runtime input data."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def physics_input_dir() -> Path:
    """Return the checked-in CCPP-SCM v7 radiation text-table directory."""

    return repo_root() / "orchestration" / "rrtmgp_radiation" / "reference_inputs" / "physics_input_data"


__all__ = ["physics_input_dir", "repo_root"]
