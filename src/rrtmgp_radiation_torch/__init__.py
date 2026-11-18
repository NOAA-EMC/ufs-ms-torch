# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""PyTorch RRTMGP radiation group of the GFS v17 suite (``MPAS_GFS_v17_tempo_rrtmgp``).

``run_radiation_group`` runs ``GFS_rrtmgp_setup`` through ``GFS_radiation_post`` on column-batched
tensors (``radiation_group``); ``load_rrtmgp_tables`` loads the RRTMGP coefficient tables it reads.
"""

from .cloud import DEFAULT_DECORRELATION_LENGTH_KM, compute_hogan_decorrelation_length
from .gas_optics import (
    DEFAULT_ACTIVE_GASES,
    ReducedGasOpticsTable,
    compute_shortwave_solar_source,
    reduce_gas_optics_table,
)
from .longwave_main import compute_longwave_cloud_optics_byband
from .radiation_group import RadiationConfig, RadiationConfigError, run_radiation_group
from .post import CP_DRY, GRAV, compute_heating_rate
from .pre import compute_nonprognostic_gases, fpvs
from .shortwave import compute_shortwave_cloud_optics_byband
from .tables import RrtmgpTableBundle, RrtmgpTableFile, load_rrtmgp_tables

__all__ = [
    "CP_DRY",
    "DEFAULT_ACTIVE_GASES",
    "DEFAULT_DECORRELATION_LENGTH_KM",
    "GRAV",
    "RadiationConfig",
    "RadiationConfigError",
    "ReducedGasOpticsTable",
    "RrtmgpTableBundle",
    "RrtmgpTableFile",
    "compute_heating_rate",
    "compute_hogan_decorrelation_length",
    "compute_longwave_cloud_optics_byband",
    "compute_nonprognostic_gases",
    "compute_shortwave_cloud_optics_byband",
    "compute_shortwave_solar_source",
    "fpvs",
    "load_rrtmgp_tables",
    "reduce_gas_optics_table",
    "run_radiation_group",
]
