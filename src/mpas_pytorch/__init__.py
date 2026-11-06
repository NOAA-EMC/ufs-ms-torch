# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
MPAS-Atmosphere PyTorch port.

This package provides a PyTorch implementation of the tested MPAS-Atmosphere
dynamics-core release surface.  Numerical claims are scoped to registered gates.

Key modules:
    constants: Physical constants (gravity, gas constants, etc.)
    mesh: Mesh data structures and connectivity arrays
    dynamics: Time integration and dynamical processes
    boundary: Boundary condition handling
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
from mpas_pytorch.release_capabilities import get_release_capabilities
from mpas_pytorch.restart import read_restart, write_restart

__release_capabilities__ = get_release_capabilities()

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
