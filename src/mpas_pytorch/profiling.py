# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Opt-in profiling ranges for MPAS forward runners and hot sections."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch


def profiling_enabled(config: Mapping[str, Any] | None = None) -> bool:
    """Return whether section-level profiler ranges are requested."""
    if config is None:
        return False
    return bool(
        config.get("config_profile_sections", config.get("profile_sections", False))
    )


@contextmanager
def profile_section(
    name: str,
    config: Mapping[str, Any] | None = None,
    *,
    enabled: bool | None = None,
) -> Iterator[None]:
    """Emit a torch-profiler range when profiling is explicitly enabled."""
    if not (profiling_enabled(config) if enabled is None else bool(enabled)):
        yield
        return
    with torch.profiler.record_function(name):
        yield


def profiled_call(
    name: str,
    config: Mapping[str, Any] | None,
    fn: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call ``fn`` under an optional profiling range."""
    with profile_section(name, config):
        return fn(*args, **kwargs)
