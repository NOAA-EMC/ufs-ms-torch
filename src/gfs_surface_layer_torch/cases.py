# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Surface-layer tensor interface and static source controls."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class SurfaceCaseConfig:
    """Scalar switches for one source-oracle run."""

    case_id: str
    ivegsrc: int
    sfc_z0_type: int
    lsm: int
    lsm_ruc: int
    iopt_lake: int
    iopt_lake_clm: int
    redrag: bool
    use_oceanuv: bool
    thsfc_loc: bool
    diag_flux: bool
    diag_log: bool
    use_lake2m: bool
    default_real_bytes: int = 4

    def __post_init__(self):
        if type(self.default_real_bytes) is not int or self.default_real_bytes not in (
            4,
            8,
        ):
            raise ValueError("source default REAL must use 4 or 8 bytes")
