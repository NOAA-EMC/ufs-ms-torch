# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
MPAS-Atmosphere PyTorch port.

This package provides a PyTorch implementation of the MPAS-Atmosphere
dynamics core.

Key modules:
    constants: Physical constants (gravity, gas constants, etc.)
    dynamics: Time integration and dynamical processes
    utils: Checkpoint loading and validation utilities
"""

__version__ = "0.1.7"
__author__ = "Xiaoxu Tian"

from mpas_pytorch.atmosphere_model import (
    prepare_initial_state,
    run_simulation,
    run_timestep,
)
from mpas_pytorch.exceptions import (
    MpasConfigError,
    MpasDeviceError,
    MpasError,
    MpasMeshMismatchError,
    MpasMissingFieldError,
    MpasNumericalAgreementError,
    MpasProvenanceError,
    MpasStrictModeError,
    MpasUnsupportedCapabilityError,
)
from mpas_pytorch.ic_loader import load_config_from_namelist, load_initial_state
from mpas_pytorch.output_writer import read_history, write_history
from mpas_pytorch.restart import read_restart, write_restart


__all__ = [
    "MpasConfigError",
    "MpasDeviceError",
    "MpasError",
    "MpasMeshMismatchError",
    "MpasMissingFieldError",
    "MpasNumericalAgreementError",
    "MpasProvenanceError",
    "MpasStrictModeError",
    "MpasUnsupportedCapabilityError",
    "load_config_from_namelist",
    "load_initial_state",
    "read_history",
    "read_restart",
    "prepare_initial_state",
    "run_simulation",
    "run_timestep",
    "write_history",
    "write_restart",
]
