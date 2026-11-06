# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Surface-layer tensor interface and static source controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REAL_INPUT_FIELDS = (
    "ps",
    "t1",
    "q1",
    "z1",
    "garea",
    "wind",
    "prsl1",
    "prslki",
    "prsik1",
    "prslk1",
    "sigmaf",
    "shdmax",
    "z0pert",
    "ztpert",
    "lakefrac",
    "fice",
    "u10m",
    "v10m",
    "u1",
    "v1",
    "usfco",
    "vsfco",
    "tskin_wat",
    "tskin_lnd",
    "tskin_ice",
    "tsurf_wat",
    "tsurf_lnd",
    "tsurf_ice",
    "z0rl_wat",
    "z0rl_lnd",
    "z0rl_ice",
    "z0rl_wav",
    "ustar_wat",
    "ustar_lnd",
    "ustar_ice",
    "cm_wat",
    "cm_lnd",
    "cm_ice",
    "ch_wat",
    "ch_lnd",
    "ch_ice",
    "rb_wat",
    "rb_lnd",
    "rb_ice",
    "stress_wat",
    "stress_lnd",
    "stress_ice",
    "fm_wat",
    "fm_lnd",
    "fm_ice",
    "fh_wat",
    "fh_lnd",
    "fh_ice",
    "fm10_wat",
    "fm10_lnd",
    "fm10_ice",
    "fh2_wat",
    "fh2_lnd",
    "fh2_ice",
    "ztmax_wat",
    "ztmax_lnd",
    "ztmax_ice",
    "xlat_d",
    "xlon_d",
    "shflx",
    "cdq",
    "zf",
    "evap",
    "ust",
    "tskin",
    "qsurf",
    "lake_t2m",
    "lake_q2m",
)

INT_INPUT_FIELDS = ("vegtype", "use_lake_model")
BOOL_INPUT_FIELDS = ("flag_iter", "dry", "icy", "flag_lakefreeze", "wet")

SURFACE_DIFF_OUTPUT_FIELDS = (
    "z0rl_wat",
    "z0rl_lnd",
    "z0rl_ice",
    "ustar_wat",
    "ustar_lnd",
    "ustar_ice",
    "cm_wat",
    "cm_lnd",
    "cm_ice",
    "ch_wat",
    "ch_lnd",
    "ch_ice",
    "rb_wat",
    "rb_lnd",
    "rb_ice",
    "stress_wat",
    "stress_lnd",
    "stress_ice",
    "fm_wat",
    "fm_lnd",
    "fm_ice",
    "fh_wat",
    "fh_lnd",
    "fh_ice",
    "fm10_wat",
    "fm10_lnd",
    "fm10_ice",
    "fh2_wat",
    "fh2_lnd",
    "fh2_ice",
    "ztmax_wat",
    "ztmax_lnd",
    "ztmax_ice",
    "zvfun",
)

SURFACE_DIAG_OUTPUT_FIELDS = ("f10m", "u10m_diag", "v10m_diag", "t2m", "q2m", "dpt2m")
CHECKPOINT_OUTPUT_FIELDS = SURFACE_DIFF_OUTPUT_FIELDS + SURFACE_DIAG_OUTPUT_FIELDS


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

    def as_fortran_scalars(self) -> tuple[Any, ...]:
        return (
            self.ivegsrc,
            self.sfc_z0_type,
            self.lsm,
            self.lsm_ruc,
            self.iopt_lake,
            self.iopt_lake_clm,
            self.redrag,
            self.use_oceanuv,
            self.thsfc_loc,
            self.diag_flux,
            self.diag_log,
            self.use_lake2m,
        )
