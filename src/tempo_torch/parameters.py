# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""TEMPO constants, size-distribution parameters and lookup-table setup."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import math

import numpy as np

from tempo_torch.host_constants import ACTIVE as _HOST
from tempo_torch.host_constants import HAIL_AWARE_INIT as _HAIL_AWARE_INIT
from tempo_torch.host_constants import INIT as _INIT


@dataclass(frozen=True)
class TableArraySpec:
    name: str
    filename: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    offset: int = 0

    @property
    def entries(self) -> int:
        return int(np.prod(self.shape))

    @property
    def byte_count(self) -> int:
        return self.entries * self.dtype.itemsize


@dataclass(frozen=True)
class TableFileRecord:
    filename: str
    expected_bytes: int
    observed_bytes: int
    access: str
    status: str


@dataclass(frozen=True)
class TempoFrozenTables:
    arrays: Mapping[str, np.ndarray]
    file_records: tuple[TableFileRecord, ...]
    mmap_enabled: bool

    @property
    def resident_payload_bytes(self) -> int:
        if self.mmap_enabled:
            return 0
        return int(sum(array.nbytes for array in self.arrays.values()))

    @property
    def logical_payload_bytes(self) -> int:
        return int(sum(array.nbytes for array in self.arrays.values()))

    def native_array(self, name: str) -> np.ndarray:
        return np.asarray(self.arrays[name], dtype=self.arrays[name].dtype.newbyteorder("="))


F64 = np.dtype(">f8")


F32 = np.dtype(">f4")


FREEZE_QR_SHAPE = (37, 37, 45, 55)


FREEZE_QC_SHAPE = (37, 100, 45, 55)


QR_ACR_QS_SHAPE = (37, 9, 37, 37)


QR_ACR_QG_SHAPE = (37, 37, 9, 37, 37)


CCN_SHAPE = (7, 9, 7, 5, 4)


TABLE_SPECS: tuple[TableArraySpec, ...] = (
    TableArraySpec("tpi_qrfz", "freeze_water_data_tempo_v3", FREEZE_QR_SHAPE, F64),
    TableArraySpec("tni_qrfz", "freeze_water_data_tempo_v3", FREEZE_QR_SHAPE, F64),
    TableArraySpec("tpg_qrfz", "freeze_water_data_tempo_v3", FREEZE_QR_SHAPE, F64),
    TableArraySpec("tnr_qrfz", "freeze_water_data_tempo_v3", FREEZE_QR_SHAPE, F64),
    TableArraySpec("tpi_qcfz", "freeze_water_data_tempo_v3", FREEZE_QC_SHAPE, F64),
    TableArraySpec("tni_qcfz", "freeze_water_data_tempo_v3", FREEZE_QC_SHAPE, F64),
    TableArraySpec("tcs_racs1", "qr_acr_qs_data_tempo_v3", QR_ACR_QS_SHAPE, F64),
    TableArraySpec("tmr_racs1", "qr_acr_qs_data_tempo_v3", QR_ACR_QS_SHAPE, F64),
    TableArraySpec("tcs_racs2", "qr_acr_qs_data_tempo_v3", QR_ACR_QS_SHAPE, F64),
    TableArraySpec("tmr_racs2", "qr_acr_qs_data_tempo_v3", QR_ACR_QS_SHAPE, F64),
    TableArraySpec("tcr_sacr1", "qr_acr_qs_data_tempo_v3", QR_ACR_QS_SHAPE, F64),
    TableArraySpec("tms_sacr1", "qr_acr_qs_data_tempo_v3", QR_ACR_QS_SHAPE, F64),
    TableArraySpec("tcr_sacr2", "qr_acr_qs_data_tempo_v3", QR_ACR_QS_SHAPE, F64),
    TableArraySpec("tms_sacr2", "qr_acr_qs_data_tempo_v3", QR_ACR_QS_SHAPE, F64),
    TableArraySpec("tnr_racs1", "qr_acr_qs_data_tempo_v3", QR_ACR_QS_SHAPE, F64),
    TableArraySpec("tnr_racs2", "qr_acr_qs_data_tempo_v3", QR_ACR_QS_SHAPE, F64),
    TableArraySpec("tnr_sacr1", "qr_acr_qs_data_tempo_v3", QR_ACR_QS_SHAPE, F64),
    TableArraySpec("tnr_sacr2", "qr_acr_qs_data_tempo_v3", QR_ACR_QS_SHAPE, F64),
    TableArraySpec("tcg_racg", "qr_acr_qg_data_tempo_v3", QR_ACR_QG_SHAPE, F64),
    TableArraySpec("tmr_racg", "qr_acr_qg_data_tempo_v3", QR_ACR_QG_SHAPE, F64),
    TableArraySpec("tcr_gacr", "qr_acr_qg_data_tempo_v3", QR_ACR_QG_SHAPE, F64),
    TableArraySpec("tnr_racg", "qr_acr_qg_data_tempo_v3", QR_ACR_QG_SHAPE, F64),
    TableArraySpec("tnr_gacr", "qr_acr_qg_data_tempo_v3", QR_ACR_QG_SHAPE, F64),
    TableArraySpec("tnccn_act", "ccn_activate.bin", CCN_SHAPE, F32, offset=4),
)


def table_file_expected_bytes() -> dict[str, int]:
    totals: dict[str, int] = {}
    for spec in TABLE_SPECS:
        totals[spec.filename] = max(totals.get(spec.filename, 0), spec.offset)
        totals[spec.filename] += spec.byte_count
    totals["ccn_activate.bin"] += 4
    return totals


def load_tables(
    table_dir: str | Path,
    *,
    mmap: bool = True,
) -> TempoFrozenTables:
    root = Path(table_dir)
    expected_sizes = table_file_expected_bytes()
    file_records: list[TableFileRecord] = []
    arrays: dict[str, np.ndarray] = {}
    offsets = {filename: 0 for filename in expected_sizes}
    offsets["ccn_activate.bin"] = 4

    for filename, expected in expected_sizes.items():
        path = root / filename
        observed = path.stat().st_size if path.exists() else -1
        status = "ok" if observed == expected else "size_mismatch"
        if not path.exists():
            status = "missing"
        file_records.append(
            TableFileRecord(
                filename=filename,
                expected_bytes=expected,
                observed_bytes=observed,
                access="sequential_record" if filename == "ccn_activate.bin" else "stream",
                status=status,
            )
        )
        if status != "ok":
            raise FileNotFoundError(
                f"TEMPO table {path} has status {status}; expected {expected} bytes, observed {observed}"
            )

    for spec in TABLE_SPECS:
        offset = spec.offset if spec.offset else offsets[spec.filename]
        path = root / spec.filename
        arrays[spec.name] = read_table_array(path, spec, offset=offset, mmap=mmap)
        offsets[spec.filename] = offset + spec.byte_count

    validate_ccn_record_markers(root / "ccn_activate.bin", TABLE_SPECS[-1].byte_count)
    return TempoFrozenTables(
        arrays=arrays,
        file_records=tuple(file_records),
        mmap_enabled=mmap,
    )


def read_table_array(
    path: Path,
    spec: TableArraySpec,
    *,
    offset: int,
    mmap: bool,
) -> np.ndarray:
    if mmap:
        return np.memmap(
            path,
            dtype=spec.dtype,
            mode="r",
            offset=offset,
            shape=spec.shape,
            order="F",
        )
    with path.open("rb") as handle:
        handle.seek(offset)
        data = np.frombuffer(handle.read(spec.byte_count), dtype=spec.dtype, count=spec.entries)
    return np.array(data.reshape(spec.shape, order="F"), dtype=spec.dtype.newbyteorder("="))


def validate_ccn_record_markers(path: Path, payload_bytes: int) -> None:
    with path.open("rb") as handle:
        head = int.from_bytes(handle.read(4), byteorder="big", signed=False)
        handle.seek(4 + payload_bytes)
        tail = int.from_bytes(handle.read(4), byteorder="big", signed=False)
    if head != payload_bytes or tail != payload_bytes:
        raise ValueError(
            f"CCN record markers are {head}/{tail}, expected {payload_bytes}"
        )


CHECKPOINT_FIELDS = ("t", "p", "w", "dz", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr")


MIN_QV = np.float64(1.0e-10)


RDRY = _HOST.rdry


ROVERRV = RDRY / _HOST.rv


TEND_WORK_FIELDS = (
    "prr_wau",
    "pnr_wau",
    "pnc_wau",
    "prr_rcw",
    "pnc_rcw",
    "pnr_rcr",
    "prs_scw",
    "pnc_scw",
    "png_scw",
    "pbg_scw",
    "prg_gcw",
    "pnc_gcw",
    "pbg_gcw",
    "pri_ihm",
    "pni_ihm",
    "prs_ihm",
    "prg_ihm",
    "prg_scw",
    "prr_rcs",
    "pnr_rcs",
    "prg_rcs",
    "png_rcs",
    "prs_rcs",
    "pbg_rcs",
    "prr_rcg",
    "pnr_rcg",
    "prg_rcg",
    "png_rcg",
    "pbg_rcg",
    "pri_inu",
    "pni_inu",
    "pri_iha",
    "pni_iha",
    "pri_wfz",
    "pni_wfz",
    "prg_rfz",
    "png_rfz",
    "pnr_rfz",
    "pri_rfz",
    "pni_rfz",
    "pbg_rfz",
    "prs_sde",
    "pri_ide",
    "pni_ide",
    "prs_ide",
    "prg_gde",
    "png_gde",
    "pni_iau",
    "prs_iau",
    "prr_sml",
    "prr_gml",
    "pbg_sml",
    "pbg_gml",
    "pnr_sml",
    "pnr_gml",
    "prr_rci",
    "pnr_rci",
    "pri_rci",
    "pni_rci",
    "prg_rci",
    "png_rci",
    "pbg_rci",
    "pni_sci",
    "prs_sci",
    "prw_vcd",
    "pnc_wcd",
    "prv_rev",
    "pnr_rev",
    "pna_rca",
    "pna_sca",
    "pna_gca",
    "pnd_rcd",
    "pnd_scd",
    "pnd_gcd",
)


TENDENCY_FIELDS = (
    "tten",
    "qvten",
    "qcten",
    "qiten",
    "qrten",
    "qsten",
    "qgten",
    "qbten",
    "niten",
    "nrten",
    "ncten",
    "ngten",
    "nwfaten",
    "nifaten",
)


SNOW_MOMENT_FIELDS = ("smo0", "smo1", "smo2", "smob", "smoc", "smoe", "smof", "smog", "smoz", "ns")


FALLSPEED_FIELDS = ("vtrr", "vtnr", "vtrs", "vtri", "vtni", "vtrc", "vtnc", "vtrg", "vtng")


PRECIP_DIAG_FIELDS = (
    "rain_precip",
    "cloud_precip",
    "ice_liquid_equiv_precip",
    "snow_liquid_equiv_precip",
    "graupel_liquid_equiv_precip",
    "frz_rain_precip",
)


EPS = np.float64(1.0e-15)


NACCN0 = np.float64(300.0e6)


NACCN1 = np.float64(50.0e6)


NAIN0 = np.float64(1.5e6)


NAIN1 = np.float64(0.5e6)


NWFA_DEFAULT = np.float64(11.1e6)


NIFA_DEFAULT = np.float64(0.5e6 * 0.01)


AERO_MAX = np.float64(9999.0e6)


PI = _HOST.pi


T0 = _HOST.t0


RHO_W = _HOST.rho_w


RHO_I = np.float64(890.0)


R1 = np.float64(1.0e-12)


R2 = np.float64(1.0e-6)


MAX_NI = np.float64(4999.0e3)


MU_R = np.float64(0.0)


MU_I = np.float64(0.0)


MU_S = np.float64(0.6357)


BM_R = np.float64(3.0)


BM_I = np.float64(3.0)


BM_S = np.float64(2.0)


BV_I = np.float64(1.0)


BV_S = np.float64(0.55)


AM_R = PI * RHO_W / np.float64(6.0)


AM_I = PI * RHO_I / np.float64(6.0)


# tempo_init-time tables (efrw, efsw, drop evaporation, ice autoconversion, ice bins)
# are built before the CCPP host overrides the constants (see _host_constants).
INIT_PI = _INIT.pi


INIT_RHO_W = _INIT.rho_w


INIT_AM_R = INIT_PI * INIT_RHO_W / np.float64(6.0)


INIT_AM_I = INIT_PI * RHO_I / np.float64(6.0)


AM_S = np.float64(0.069)


OBMR = np.float64(1.0) / BM_R


OBMI = np.float64(1.0) / BM_I


OAMS = np.float64(1.0) / AM_S


D0R = np.float64(50.0e-6)


D0R_MAX = np.float64(2.5e-3)


D0S = np.float64(300.0e-6)


KAP0 = np.float64(490.6)


KAP1 = np.float64(17.46)


LAM0 = np.float64(20.78)


LAM1 = np.float64(3.29)


SA = np.array(
    [5.065339, -0.062659, -3.032362, 0.029469, -0.000285, 0.31255, 0.000204, 0.003199, 0.0, -0.015952],
    dtype=np.float64,
)


SB = np.array(
    [0.476221, -0.015896, 0.165977, 0.007468, -0.000141, 0.060366, 0.000079, 0.000594, 0.0, -0.003577],
    dtype=np.float64,
)


CIE = np.array(
    [
        MU_I + 1.0,
        BM_I + MU_I + 1.0,
        BM_I + MU_I + BV_I + 1.0,
        MU_I + BV_I + 1.0,
        MU_I + 2.0,
        BM_I * 0.5 + MU_I + BV_I + 1.0,
        BM_I * 0.5 + MU_I + 1.0,
    ],
    dtype=np.float64,
)


CIG = np.array([math.gamma(float(value)) for value in CIE], dtype=np.float64)


OIG1 = np.float64(1.0) / CIG[0]


OIG2 = np.float64(1.0) / CIG[1]


CRE = np.array(
    [
        BM_R + 1.0,
        MU_R + 1.0,
        BM_R + MU_R + 1.0,
        BM_R * 2.0 + MU_R + 1.0,
        MU_R + 1.0 + 1.0,
        BM_R + MU_R + 1.0 + 1.0,
        BM_R * 0.5 + MU_R + 1.0 + 1.0,
        BM_R + MU_R + 1.0 + 3.0,
        MU_R + 1.0 + 3.0,
        MU_R + 2.0,
        0.5 * (1.0 + 5.0 + 2.0 * MU_R),
        BM_R * 0.5 + MU_R + 1.0,
        BM_R * 2.0 + MU_R + 1.0 + 1.0,
    ],
    dtype=np.float64,
)


CRG = np.array([math.gamma(float(value)) for value in CRE], dtype=np.float64)


ORG2 = np.float64(1.0) / CRG[1]


ORG3 = np.float64(1.0) / CRG[2]


CSE = np.array(
    [
        BM_S + 1.0,
        BM_S + 2.0,
        BM_S * 2.0,
        BM_S + BV_S + 1.0,
        BM_S * 2.0 + BV_S + 1.0,
        BM_S * 2.0 + 1.0,
        BM_S + MU_S + 1.0,
        BM_S + MU_S + 2.0,
        BM_S + MU_S + 3.0,
        BM_S + MU_S + BV_S + 1.0,
        BM_S * 2.0 + MU_S + BV_S + 1.0,
        BM_S * 2.0 + MU_S + 1.0,
        BV_S + 2.0,
        BM_S + BV_S,
        MU_S + 1.0,
        1.0 + (1.0 + BV_S) / 2.0,
        BM_S + BV_S + 2.0,
    ],
    dtype=np.float64,
)


CSG = np.array([math.gamma(float(value)) for value in CSE], dtype=np.float64)


NT_C_MAX = np.float64(1999.0e6)


NT_C_MIN = np.float64(2.0)


NT_C_L = np.float64(100.0e6)


NT_C_O = np.float64(50.0e6)


NU_C_SCALE = np.float64(1000.0e6)


D0C = np.float64(1.0e-6)


BV_C = np.float64(2.0)


CCE = np.array(
    [
        [n + 1.0 for n in range(1, 16)],
        [BM_R + n + 1.0 for n in range(1, 16)],
        [BM_R + n + 4.0 for n in range(1, 16)],
        [n + BV_C + 1.0 for n in range(1, 16)],
        [BM_R + n + BV_C + 1.0 for n in range(1, 16)],
    ],
    dtype=np.float64,
)


CCG = np.array([[math.gamma(float(value)) for value in row] for row in CCE], dtype=np.float64)


OCG1 = np.float64(1.0) / CCG[0]


OCG2 = np.float64(1.0) / CCG[1]


def get_nuc(nc: float) -> int:
    ratio = NU_C_SCALE / nc
    if ratio >= 12.5:
        return 15
    return min(15, int(math.floor(float(ratio) + 0.5)) + 2)


CP = _HOST.cp


LVAP0 = _HOST.lvap0


RV = _HOST.rv


ORV = np.float64(1.0) / RV


RHO_NOT = np.float64(101325.0) / (RDRY * np.float64(298.0))


METERS3_TO_LITERS = np.float64(1000.0)


NRHG = 9


IDX_BG1 = 6


RHO_G = np.array([50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0], dtype=np.float64)


AV_G = np.array(
    [45.9173813, 67.0867386, 98.0158463, 122.353378, 143.204224, 161.794724, 178.762115, 194.488785, 209.225876],
    dtype=np.float64,
)


BV_G = np.full(NRHG, 0.640961647, dtype=np.float64)


if not _HAIL_AWARE_INIT:
    # initialize_graupel_vars(hail_flag=.false.): av_g_old/bv_g_old at idx_bg1.
    AV_G[IDX_BG1 - 1] = np.float64(442.0)
    BV_G[IDX_BG1 - 1] = np.float64(0.89)


BM_G = np.float64(3.0)


MU_G = np.float64(0.0)


AM_G = PI * RHO_G / np.float64(6.0)


OBMG = np.float64(1.0) / BM_G


D0G = np.float64(350.0e-6)


GONV_MIN = np.float64(1.0e2)


GONV_MAX = np.float64(1.0e6)


CGE = np.zeros((12, NRHG), dtype=np.float64)


CGE[0, :] = BM_G + 1.0


CGE[1, :] = MU_G + 1.0


CGE[2, :] = BM_G + MU_G + 1.0


CGE[3, :] = BM_G * 2.0 + MU_G + 1.0


CGE[9, :] = MU_G + 2.0


CGE[11, :] = BM_G * 0.5 + MU_G + 1.0


for _m in range(NRHG):
    CGE[4, _m] = BM_G * 2.0 + MU_G + BV_G[_m] + 1.0
    CGE[5, _m] = BM_G + MU_G + BV_G[_m] + 1.0
    CGE[6, _m] = BM_G * 0.5 + MU_G + BV_G[_m] + 1.0
    CGE[7, _m] = MU_G + BV_G[_m] + 1.0
    CGE[8, _m] = MU_G + BV_G[_m] + 3.0
    CGE[10, _m] = 0.5 * (BV_G[_m] + 5.0 + 2.0 * MU_G)


CGG = np.array([[math.gamma(float(value)) for value in row] for row in CGE], dtype=np.float64)


OGE1 = np.float64(1.0) / CGE[0, 0]


OGG1 = np.float64(1.0) / CGG[0, 0]


OGG2 = np.float64(1.0) / CGG[1, 0]


OGG3 = np.float64(1.0) / CGG[2, 0]


NBR = 100


NBC = 100


NTB_R = 37


NTB_R1 = 37


NTB_S = 37


NTB_T = 9


NTB_G = 37


NTB_G1 = 37


AV_R = np.float64(4854.0)


FV_R = np.float64(195.0)


AUTOCON_NR_FACTOR = np.float64(10.0)


T1_QR_QC = PI * np.float64(0.25) * AV_R * CRG[8]


NIR2 = -6


NIS2 = -6


NIG2 = -6


NIR3 = 6


NIG3 = 2


R_R = np.array(
    [float(digit) * 10.0**power for power in (-6, -5, -4, -3) for digit in range(1, 10)]
    + [1.0e-2],
    dtype=np.float64,
)


R_S = np.array(R_R, dtype=np.float64, copy=True)


R_G = np.array(R_R, dtype=np.float64, copy=True)


DR = None


T_EFRW = None


def _rain_diameter_bins() -> np.ndarray:
    global DR
    if DR is None:
        DR = _create_bins(NBR, float(D0R), 0.005)
    return DR


def _cloud_diameter_bins() -> np.ndarray:
    # initialize_bins_for_tables: dc(1) = d0c, dc(n) = dc(n-1) + 1.0e-6 (sequential sums, so
    # e.g. dc(6) = 5.999999999999999e-06, which the efsw dc < 6e-6 test sees).
    values = np.empty(NBC, dtype=np.float64)
    values[0] = D0C
    for n in range(1, NBC):
        values[n] = values[n - 1] + np.float64(1.0e-6)
    return values


def _warm_rain_efficiency() -> np.ndarray:
    global T_EFRW
    if T_EFRW is not None:
        return T_EFRW
    dr = _rain_diameter_bins()
    dc = _cloud_diameter_bins()
    values = np.zeros((NBR, NBC), dtype=np.float64)
    for j in range(NBC):
        for i in range(NBR):
            ef_rw = 0.0
            p = dc[j] / dr[i]
            if dr[i] < 50.0e-6 or dc[j] < 3.0e-6:
                values[i, j] = 0.0
            elif p > 0.25:
                x = dc[j] * 1.0e6
                if dr[i] < 75.0e-6:
                    ef_rw = 0.026794 * x - 0.20604
                elif dr[i] < 125.0e-6:
                    ef_rw = -0.00066842 * x * x + 0.061542 * x - 0.37089
                elif dr[i] < 175.0e-6:
                    ef_rw = 4.091e-06 * x * x * x * x - 0.00030908 * x * x * x + 0.0066237 * x * x - 0.0013687 * x - 0.073022
                elif dr[i] < 250.0e-6:
                    ef_rw = 9.6719e-5 * x * x * x - 0.0068901 * x * x + 0.17305 * x - 0.65988
                elif dr[i] < 350.0e-6:
                    ef_rw = 9.0488e-5 * x * x * x - 0.006585 * x * x + 0.16606 * x - 0.56125
                else:
                    ef_rw = 0.00010721 * x * x * x - 0.0072962 * x * x + 0.1704 * x - 0.46929
            else:
                # Preserve the original coefficient-first products before cancellation.
                vtr = -0.1021 + 4.932e3 * dr[i] - 0.9551e6 * dr[i] * dr[i] + 0.07934e9 * dr[i] * dr[i] * dr[i] - 0.002362e12 * dr[i] * dr[i] * dr[i] * dr[i]
                stokes = dc[j] * dc[j] * vtr * INIT_RHO_W / (9.0 * 1.718e-5 * dr[i])
                reynolds = 9.0 * stokes / (p * p * INIT_RHO_W)
                f = math.log(reynolds)
                g = -0.1007 - 0.358 * f + 0.0261 * f * f
                k0 = math.exp(g)
                z = math.log(stokes / (k0 + 1.0e-15))
                h = 0.1465 + 1.302 * z - 0.607 * z * z + 0.293 * z * z * z
                yc0 = 2.0 / INIT_PI * math.atan(h)
                ef_rw = (yc0 + p) * (yc0 + p) / ((1.0 + p) * (1.0 + p))
            values[i, j] = max(0.0, min(ef_rw, 0.95))
    T_EFRW = values
    return values


def _create_bins(numbins: int, lowbin: float, highbin: float) -> np.ndarray:
    xdx = np.empty(numbins + 1, dtype=np.float64)
    xdx[0] = lowbin
    xdx[numbins] = highbin
    for n in range(1, numbins):
        xdx[n] = math.exp(float(n) / float(numbins) * math.log(xdx[numbins] / xdx[0]) + math.log(xdx[0]))
    bins = np.empty(numbins, dtype=np.float64)
    for n in range(numbins):
        bins[n] = math.sqrt(xdx[n] * xdx[n + 1])
    return bins


NTB_C = 37


NTB_IN = 55


NTB_T1 = 45


NIC2 = -6


NIIN2 = 0


TNO = np.float64(5.0)


ATO = np.float64(0.304)


HGFRZ = np.float64(235.16)


XM0I = np.float64(1.0e-12)


ICENUC_MAX = np.float64(1000.0e3)


DEMOTT_NUC_SSATI = np.float64(0.25)


DEMOTT_NUC_TEMPC = np.float64(-20.0)


R_UNI = _HOST.r_uni


RHO_NOT0 = np.float64(101325.0) / (RDRY * T0)


AR_VOLUME = np.float64(4.0) / np.float64(3.0) * PI * np.float64(2.5e-6) ** np.float64(3.0)


R_C = np.array(R_R, dtype=np.float64, copy=True)


T_NC = None


def _cloud_number_bins() -> np.ndarray:
    global T_NC
    if T_NC is None:
        T_NC = _create_bins(NBC, 1.0, 3000.0) * 1.0e6
    return T_NC


NTB_I = 64


NTB_I1 = 55


NBI = 100


NII2 = -10


NII3 = 0


AV_S = np.float64(40.0)


LSUB = _HOST.lsub


SC = np.float64(0.632)


SC3 = SC ** (np.float64(1.0) / np.float64(3.0))


C_CUBE = np.float64(0.5)


C_SQRD = np.float64(0.15)


EF_SI = np.float64(0.05)


EF_RI = np.float64(0.95)


T1_QS_QI = PI * np.float64(0.25) * AV_S


T1_QR_QI = PI * np.float64(0.25) * AV_R * CRG[8]


T2_QR_QI = PI * np.float64(0.25) * AM_R * AV_R * CRG[7]


T1_QS_SD = np.float64(0.86)


T2_QS_SD = np.float64(0.28) * SC3 * np.sqrt(AV_S)


T1_QG_SD = np.float64(0.86) * CGG[9, 0]


R_I = np.array(
    # Source decimal literals round once, unlike digit * rounded powers of ten.
    [float(f"{digit}e{power}") for power in range(-10, -3) for digit in range(1, 10)]
    + [1.0e-3],
    dtype=np.float64,
)


NT_I = np.array(
    [float(digit) * 10.0**power for power in range(0, 6) for digit in range(1, 10)]
    + [1.0e6],
    dtype=np.float64,
)


def calc_gamma_p(a: float, x: float) -> float:
    if x < 0.0 or a <= 0.0:
        raise ValueError("calc_gamma_p requires x >= 0 and a > 0")
    if x < (a + 1.0):
        return _calc_gamma_series(a, x)
    return 1.0 - _calc_gamma_cf(a, x)


def _calc_gamma_series(a: float, x: float) -> float:
    it_max = 100
    smallvalue = 1.0e-7
    aj = a
    sum_term = 1.0 / math.gamma(aj + 1.0)
    total = sum_term
    for _ in range(1, it_max + 1):
        aj += 1.0
        sum_term = sum_term * x / aj
        total += sum_term
        if abs(sum_term) < (abs(total) * smallvalue):
            break
    return total * x**a * math.exp(-x)


def _calc_gamma_cf(a: float, x: float) -> float:
    it_max = 100
    smallvalue = 1.0e-7
    offset = 1.0e-30
    f0 = offset
    b = 1.0 - a + x
    d = 1.0 / b
    c = b + (1.0 / f0)
    delta = c * d
    f = f0 * delta
    for k in range(1, it_max + 1):
        aj = k * (a - k)
        b += 2.0
        d = b + aj * d
        if abs(d) < offset:
            d = offset
        c = b + aj / c
        if abs(c) < offset:
            c = offset
        d = 1.0 / d
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < smallvalue:
            break
    return math.exp(-x + a * math.log(x)) * f / math.gamma(a)


@lru_cache(maxsize=1)
def _ice_autoconversion_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    di = _create_bins(NBI, float(_init_d0i()), float(2.0 * D0S))
    dti = _bin_widths(NBI, float(_init_d0i()), float(2.0 * D0S))
    tpi_ide = np.zeros((NTB_I, NTB_I1), dtype=np.float64)
    tps_iaus = np.zeros((NTB_I, NTB_I1), dtype=np.float64)
    tni_iaus = np.zeros((NTB_I, NTB_I1), dtype=np.float64)
    for j in range(NTB_I1):
        for i in range(NTB_I):
            lami = (INIT_AM_I * CIG[1] * OIG1 * NT_I[j] / R_I[i]) ** OBMI
            di_mean = (BM_I + MU_I + 1.0) / lami
            n0_i = NT_I[j] * OIG1 * lami ** CIE[0]
            t1 = 0.0
            t2 = 0.0
            if di_mean > 5.0 * D0S:
                t1 = R_I[i]
                t2 = NT_I[j]
                tpi_ide[i, j] = 0.0
            elif di_mean < _init_d0i():
                tpi_ide[i, j] = 1.0
            else:
                tpi_ide[i, j] = calc_gamma_p(float(MU_I + 2.0), float(lami * D0S))
                for n2 in range(NBI):
                    n_i = n0_i * di[n2] ** MU_I * math.exp(-lami * di[n2]) * dti[n2]
                    if di[n2] >= D0S:
                        t1 += n_i * INIT_AM_I * di[n2] ** BM_I
                        t2 += n_i
            tps_iaus[i, j] = t1
            tni_iaus[i, j] = t2
    return tpi_ide, tps_iaus, tni_iaus


def _bin_widths(numbins: int, lowbin: float, highbin: float) -> np.ndarray:
    xdx = np.empty(numbins + 1, dtype=np.float64)
    xdx[0] = lowbin
    xdx[numbins] = highbin
    for n in range(1, numbins):
        xdx[n] = math.exp(float(n) / float(numbins) * math.log(xdx[numbins] / xdx[0]) + math.log(xdx[0]))
    widths = np.empty(numbins, dtype=np.float64)
    for n in range(numbins):
        widths[n] = xdx[n + 1] - xdx[n]
    return widths


@lru_cache(maxsize=1)
def _d0i() -> float:
    return float((XM0I / AM_I) ** OBMI)


@lru_cache(maxsize=1)
def _init_d0i() -> float:
    """d0i of the tempo_init-time ice bins and autoconversion table (default constants)."""
    return float((XM0I / INIT_AM_I) ** OBMI)


NBS = 100


FV_S = np.float64(100.0)


RHO_S = np.float64(100.0)


RIME_THRESHOLD = np.float64(2.0)


RIME_CONVERSION = np.float64(0.95)


T1_QS_QC = PI * np.float64(0.25) * AV_S


@lru_cache(maxsize=1)
def _snow_cloud_water_efficiency() -> np.ndarray:
    ds = _snow_diameter_bins()
    dc = _cloud_diameter_bins()
    values = np.zeros((NBS, NBC), dtype=np.float64)
    for j in range(NBC):
        vtc = 1.19e4 * (1.0e4 * dc[j] * dc[j] * 0.25)
        for i in range(NBS):
            vts = AV_S * ds[i] ** BV_S * math.exp(float(-FV_S * ds[i])) - vtc
            ds_m = (AM_S * ds[i] ** BM_S / INIT_AM_R) ** OBMR
            p = dc[j] / ds_m
            if p > 0.25 or ds[i] < D0S or dc[j] < 6.0e-6 or vts < 1.0e-3:
                values[i, j] = 0.0
            else:
                stokes = dc[j] * dc[j] * vts * INIT_RHO_W / (9.0 * 1.718e-5 * ds_m)
                reynolds = 9.0 * stokes / (p * p * INIT_RHO_W)
                f = math.log(float(reynolds))
                g = -0.1007 - 0.358 * f + 0.0261 * f * f
                k0 = math.exp(g)
                z = math.log(float(stokes / (k0 + 1.0e-15)))
                h = 0.1465 + 1.302 * z - 0.607 * z * z + 0.293 * z * z * z
                yc0 = 2.0 / INIT_PI * math.atan(h)
                ef_sw = (yc0 + p) * (yc0 + p) / ((1.0 + p) * (1.0 + p))
                values[i, j] = max(0.0, min(ef_sw, 0.95))
    return values


@lru_cache(maxsize=1)
def _snow_diameter_bins() -> np.ndarray:
    return _create_bins(NBS, float(D0S), 0.02)


OLFUS = np.float64(1.0) / (LSUB - LVAP0)


T1_QS_ME = PI * np.float64(4.0) * C_SQRD * OLFUS * np.float64(0.86)


T2_QS_ME = (
    PI
    * np.float64(4.0)
    * C_SQRD
    * OLFUS
    * np.float64(0.28)
    * SC3
    * np.sqrt(AV_S)
)


T1_QG_ME = (
    PI
    * np.float64(4.0)
    * C_CUBE
    * OLFUS
    * np.float64(0.86)
    * CGG[9, 0]
)


TIMESTEP_CONVERSION_RIME_TO_RAIN = np.float64(120.0)


WF_AEROSOL_SIZE = np.float64(0.04e-6)


IF_AEROSOL_SIZE = np.float64(0.8e-6)


BOLTZMAN = np.float64(1.3806503e-23)


MEANPATH = np.float64(0.0256e-6)


LFUS = LSUB - LVAP0


TA_NA = np.array([10.0, 31.6, 100.0, 316.0, 1000.0, 3160.0, 10000.0], dtype=np.float64)


TA_WW = np.array([0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0, 31.6, 100.0], dtype=np.float64)


TA_TK = np.array([243.15, 253.15, 263.15, 273.15, 283.15, 293.15, 303.15], dtype=np.float64)


T1_QR_EV = np.float64(0.78) * CRG[9]


T2_QR_EV = np.float64(0.308) * SC3 * np.sqrt(AV_R) * CRG[10]


@lru_cache(maxsize=1)
def _drop_evap_number_table() -> np.ndarray:
    cloud_diameters = _cloud_diameter_bins()
    cloud_widths = np.empty(NBC, dtype=np.float64)
    cloud_widths[0] = D0C
    cloud_widths[1:] = cloud_diameters[1:] - cloud_diameters[:-1]
    cloud_number_bins = _cloud_number_bins()
    values = np.zeros((NBC, NTB_C, NBC), dtype=np.float64)
    massc = INIT_AM_R * cloud_diameters**BM_R
    for k in range(NBC):
        nu_c = get_nuc(float(cloud_number_bins[k]))
        for j in range(NTB_C):
            lamc = (
                cloud_number_bins[k]
                * INIT_AM_R
                * CCG[1, nu_c - 1]
                * OCG1[nu_c - 1]
                / R_C[j]
            ) ** OBMR
            n0_c = cloud_number_bins[k] * OCG1[nu_c - 1] * lamc ** CCE[0, nu_c - 1]
            n_c = n0_c * cloud_diameters**nu_c * np.exp(-lamc * cloud_diameters) * cloud_widths
            cumulative = np.cumsum(n_c)
            for i in range(NBC):
                values[i, j, k] = cumulative[i]
    return values


LOW_LIMIT_MASS_FOR_PRECIP = np.float64(1.0e-9)


SEMI_SEDI_FACTOR = np.float64(10.0)


AV_G_OLD = np.float64(442.0)


BV_G_OLD = np.float64(0.89)


A_COEFF = np.float64(0.47244157)


B_COEFF = np.float64(0.54698726)


EARTH_GRAVITY = np.float64(9.8)


AV_I = np.float64(1493.9)


AV_C = np.float64(0.316946e8)


def segment_i_hgfrz() -> np.float64:
    return np.float64(235.16)


NHBINS = 50


HAIL_CONC_THRESHOLD = np.float64(0.0005)


def _hail_bins() -> tuple[np.ndarray, np.ndarray]:
    edges = np.empty(NHBINS + 1, dtype=np.float64)
    edges[0] = 500.0e-6
    edges[-1] = 0.075
    for index in range(1, NHBINS):
        edges[index] = np.exp((index / NHBINS) * np.log(edges[-1] / edges[0]) + np.log(edges[0]))
    bins = np.sqrt(edges[:-1] * edges[1:])
    widths = edges[1:] - edges[:-1]
    return bins, widths


RADAR_BINS = 50


MELT_OUTSIDE = np.float64(0.9)


LAMBDA_RADAR = np.float64(0.10)


_OBMS = np.float64(1.0) / BM_S


_OCMS = OAMS**_OBMS


def complex_water_ray(lambda_radar: float, temperature_celsius: float) -> np.complex128:
    """Complex refractive index of water from Ray (1972)."""

    wavelength = np.float64(lambda_radar)
    temp = np.float64(temperature_celsius)
    epsinf = 5.27137 + 0.02164740 * temp - 0.00131198 * temp * temp
    epss = 78.54 * (
        1.0
        - 4.579e-3 * (temp - 25.0)
        + 1.190e-5 * (temp - 25.0) * (temp - 25.0)
        - 2.800e-8 * (temp - 25.0) * (temp - 25.0) * (temp - 25.0)
    )
    alpha = -16.8129 / (temp + 273.16) + 0.0609265
    lambdas = 0.00033836 * np.exp(2513.98 / (temp + 273.16)) * 1.0e-2
    ratio = lambdas / wavelength
    nenner = 1.0 + 2.0 * ratio ** (1.0 - alpha) * np.sin(alpha * PI * 0.5)
    nenner = nenner + ratio ** (2.0 - 2.0 * alpha)
    epsr = epsinf + ((epss - epsinf) * (ratio ** (1.0 - alpha) * np.sin(alpha * PI * 0.5) + 1.0)) / nenner
    epsi = ((epss - epsinf) * (ratio ** (1.0 - alpha) * np.cos(alpha * PI * 0.5))) / nenner
    epsi = epsi + wavelength * 1.25664 / 1.88496
    return np.complex128(np.sqrt(np.complex128(complex(epsr, -epsi))))


def complex_ice_maetzler(lambda_radar: float, temperature_celsius: float) -> np.complex128:
    """Complex refractive index of ice from Maetzler (1998)."""

    wavelength = np.float64(lambda_radar)
    temp = np.float64(temperature_celsius)
    c_light = 2.99e8
    tk = temp + 273.16
    frequency = c_light / wavelength * 1.0e-9
    b1 = 0.0207
    b2 = 1.16e-11
    b = 335.0
    exp_b = np.exp(b / tk)
    deltabeta = np.exp(-10.02 + 0.0364 * (tk - 273.16))
    betam = (b1 / tk) * (exp_b / ((exp_b - 1.0) ** 2.0)) + b2 * frequency * frequency
    beta = betam + deltabeta
    theta = 300.0 / tk - 1.0
    alfa = (0.00504 + 0.0062 * theta) * np.exp(-22.1 * theta)
    refractive_index = complex(3.1884 + 9.1e-4 * (tk - 273.16), alfa / frequency + beta * frequency)
    return np.complex128(np.sqrt(np.conj(np.complex128(refractive_index))))


def _create_bins_with_widths(numbins: int, lowbin: float, highbin: float) -> tuple[np.ndarray, np.ndarray]:
    xdx = np.empty(numbins + 1, dtype=np.float64)
    xdx[0] = lowbin
    xdx[numbins] = highbin
    for n in range(1, numbins):
        xdx[n] = np.exp(float(n) / float(numbins) * np.log(xdx[numbins] / xdx[0]) + np.log(xdx[0]))
    bins = np.sqrt(xdx[:-1] * xdx[1:])
    widths = xdx[1:] - xdx[:-1]
    return bins, widths


def _simpson_weights(numbins: int) -> np.ndarray:
    weights = np.zeros(numbins + 1, dtype=np.float64)
    for n in range(0, numbins - 1, 2):
        weights[n] = weights[n] + 1.0 / 3.0
        weights[n + 1] = weights[n + 1] + 4.0 / 3.0
        weights[n + 2] = weights[n + 2] + 1.0 / 3.0
    return weights[:numbins]
