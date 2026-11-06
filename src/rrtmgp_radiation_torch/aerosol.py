# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""PyTorch-native aerosol-G kernels for the frozen OPAC RRTMGP branch.

The target-owned surface here is the active CCPP-SCM RRTMGP aerosol path:
``iaermdl=5`` OPAC climatology with volcanic aerosol enabled.  One-time
ingestion of ``aerosol.dat`` and volcanic text tables is treated like the
NetCDF coefficient-table loader: the parser runs on the host, validates frozen
hashes, and produces coefficient tensors.  The scientific G calculation itself
does not consume the sealed ``ABCDEFG`` output oracle.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .compile import compile_categories_enabled, run_compile_island
from .reference_inputs import physics_input_dir as _physics_input_dir

TensorOrString = torch.Tensor | str

NBDSW = 14
NBDLW = 16
NSWBND = NBDSW
NLWBND = NBDLW
NSWLWBD = NBDSW + NBDLW
NWVSOL = 151
NWVTOT = 57600
NWVTIR = 4000
NXC = 5
NAE = 7
NDM = 5
IMXAE = 72
JMXAE = 37
NAERBND = 61
NRHLEV = 8
NCM1 = 6
NCM2 = 4
NCM = NCM1 + NCM2
NSPC = 5
NSPC1 = NSPC + 1
_AEROSOL_REDUCTION_SLAB_LEVELS = 16
_AEROSOL_PROFILE_SLAB_LEVELS = 16
_AEROSOL_CUDA_SLAB_LEVELS = 32


def _aerosol_slab_levels(value: torch.Tensor, cpu_levels: int) -> int:
    """Use bounded wider CUDA slabs while retaining measured CPU specialization."""

    return _AEROSOL_CUDA_SLAB_LEVELS if value.device.type == "cuda" else cpu_levels

RHLV = np.asarray([0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99], dtype=np.float64)
IDXSPC = np.asarray([1, 2, 1, 1, 1, 1, 3, 5, 5, 4], dtype=np.int64) - 1
IDXSPC_TUPLE = tuple(int(value) for value in IDXSPC)
WVNSW1 = np.asarray(
    [2600.0, 3250.0, 4000.0, 4650.0, 5150.0, 6150.0, 7700.0, 8050.0,
     12850.0, 16000.0, 22650.0, 29000.0, 38000.0, 820.0],
    dtype=np.float64,
)
WVNSW2 = np.asarray(
    [3250.0, 4000.0, 4650.0, 5150.0, 6150.0, 7700.0, 8050.0, 12850.0,
     16000.0, 22650.0, 29000.0, 38000.0, 50000.0, 2600.0],
    dtype=np.float64,
)
WVNLW1 = np.asarray(
    [10.0, 350.0, 500.0, 630.0, 700.0, 820.0, 980.0, 1080.0,
     1180.0, 1390.0, 1480.0, 1800.0, 2080.0, 2250.0, 2380.0, 2600.0],
    dtype=np.float64,
)
WVNLW2 = np.asarray(
    [350.0, 500.0, 630.0, 700.0, 820.0, 980.0, 1080.0, 1180.0,
     1390.0, 1480.0, 1800.0, 2080.0, 2250.0, 2380.0, 2600.0, 3250.0],
    dtype=np.float64,
)

# physcons.F90 values used by the active CCPP-SCM source chain.
CON_T0C = 2.7315e2
CON_C = 2.99792458e8
CON_PLNK = 6.6260693e-34
CON_BOLTZ = 1.3806505e-23

AEROSOL_DAT_SHA256 = "ac5e965cc70a25cbb2845848e30c62569d7ea33e719a3f463ba5f56e86755727"
VOLCANIC_AEROSOL_SHA256 = {
    (1990, 1999): "32c4a8e94a755888e30ac1c276541fc4b39cf6b7439f8df8cca594f7cefe535b",
}
MIN_VOLCANIC_YEAR = 1850
MAX_VOLCANIC_YEAR = 1999
_AEROSOL_DEVICE_TABLE_CACHE: dict[tuple[str, int | None, int, int, float], "_AerosolTables"] = {}


@dataclass(frozen=True)
class RrtmgpAerosolResult:
    """Checkpoint-shaped aerosol-G output fields."""

    fields: Mapping[str, TensorOrString]
    runtime_kind: str = "torch_rrtmgp_aerosol_opac_g"


@dataclass(frozen=True)
class _RawAerosolTables:
    haer: np.ndarray
    prsref: np.ndarray
    sigref: np.ndarray
    kprfg: np.ndarray
    idxcg: np.ndarray
    cmixg: np.ndarray
    denng: np.ndarray
    extrhi: np.ndarray
    scarhi: np.ndarray
    ssarhi: np.ndarray
    asyrhi: np.ndarray
    extrhd: np.ndarray
    scarhd: np.ndarray
    ssarhd: np.ndarray
    asyrhd: np.ndarray
    extstra: np.ndarray
    ivolae: np.ndarray
    nv_aod: int
    month: int
    year: int
    aerosol_sha256: str
    volcanic_sha256: str


@dataclass(frozen=True)
class _AerosolTables:
    haer: torch.Tensor
    prsref: torch.Tensor
    sigref: torch.Tensor
    kprfg: torch.Tensor
    idxcg: torch.Tensor
    cmixg: torch.Tensor
    denng: torch.Tensor
    extrhi: torch.Tensor
    scarhi: torch.Tensor
    ssarhi: torch.Tensor
    asyrhi: torch.Tensor
    extrhd: torch.Tensor
    scarhd: torch.Tensor
    ssarhd: torch.Tensor
    asyrhd: torch.Tensor
    extstra: torch.Tensor
    ivolae: torch.Tensor
    nv_aod: int
    month: int
    year: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _fortran_aerosol_source() -> Path:
    return _repo_root() / "physics-fortran" / "ccpp-physics" / "physics" / "Radiation" / "radiation_aerosols.f"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numbers(text: str) -> list[float]:
    return [float(item) for item in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?|[-+]?\.\d+(?:[Ee][+-]?\d+)?", text)]


@lru_cache(maxsize=1)
def _spectrum_constants_from_fortran() -> tuple[np.ndarray, np.ndarray]:
    source = _fortran_aerosol_source().read_text(encoding="utf-8")
    nwv_match = re.search(r"data\s+nwvns0\s*/(.*?)/", source, flags=re.IGNORECASE | re.DOTALL)
    if nwv_match is None:
        raise ValueError("could not parse nwvns0 from radiation_aerosols.f")
    nwvns0 = np.asarray([int(float(item)) for item in _numbers(nwv_match.group(1))], dtype=np.int64)

    start = source.find("data  s0intv")
    end = source.find("real (kind=kind_phys), dimension(NBDSW)", start)
    if start < 0 or end < 0:
        raise ValueError("could not parse s0intv from radiation_aerosols.f")
    values: list[float] = []
    for block in re.findall(r"/([^/]*)/", source[start:end], flags=re.DOTALL):
        values.extend(_numbers(block))
    s0intv = np.asarray(values, dtype=np.float64)
    if nwvns0.shape != (NWVSOL,) or s0intv.shape != (NWVSOL,):
        raise ValueError(f"bad aerosol spectrum constants: {nwvns0.shape}, {s0intv.shape}")
    return nwvns0, s0intv


def _read_float_block(lines: list[str], index: int, count: int) -> tuple[np.ndarray, int]:
    values: list[float] = []
    while len(values) < count:
        values.extend(_numbers(lines[index]))
        index += 1
    if len(values) != count:
        raise ValueError(f"expected {count} values, got {len(values)}")
    return np.asarray(values, dtype=np.float64), index


def _read_int_block(lines: list[str], index: int, count: int) -> tuple[np.ndarray, int]:
    values: list[int] = []
    while len(values) < count:
        values.extend(int(float(item)) for item in _numbers(lines[index]))
        index += 1
    if len(values) != count:
        raise ValueError(f"expected {count} integer values, got {len(values)}")
    return np.asarray(values, dtype=np.int64), index


def _parse_month_line(line: str) -> tuple[list[int], list[float], int, float]:
    idx: list[int] = []
    cmix: list[float] = []
    offset = 0
    for _ in range(NXC):
        idx.append(int(line[offset:offset + 2]))
        offset += 2
        cmix.append(float(line[offset:offset + 11]))
        offset += 11
    kprf = int(line[offset:offset + 2])
    offset += 2
    denn = float(line[offset:offset + 8])
    return idx, cmix, kprf, denn


def _read_monthly_distribution(lines: list[str], month: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 1 <= month <= 12:
        raise ValueError(f"month must be 1..12, got {month}")
    block_size = 2 + JMXAE * IMXAE
    index = (month - 1) * block_size + 2
    kprfg = np.zeros((IMXAE, JMXAE), dtype=np.int64)
    idxcg = np.zeros((NXC, IMXAE, JMXAE), dtype=np.int64)
    cmixg = np.zeros((NXC, IMXAE, JMXAE), dtype=np.float64)
    denng = np.zeros((2, IMXAE, JMXAE), dtype=np.float64)
    for j in range(JMXAE):
        for i in range(IMXAE):
            idx, cmix, kprf, denn = _parse_month_line(lines[index])
            index += 1
            kprfg[i, j] = kprf
            denng[0, i, j] = denn
            denng[1, i, j] = cmix[-1] if kprf >= 6 else 0.0
            remainder = 1.0
            for k in range(NXC - 1):
                idxcg[k, i, j] = idx[k]
                cmixg[k, i, j] = cmix[k]
                remainder -= cmix[k]
            idxcg[NXC - 1, i, j] = idx[NXC - 1]
            cmixg[NXC - 1, i, j] = remainder
    return kprfg, idxcg, cmixg, denng


def _set_spectrum(con_pi: float) -> tuple[np.ndarray, np.ndarray]:
    nwvns0, s0intv = _spectrum_constants_from_fortran()
    solfwv = np.zeros(NWVTOT + 1, dtype=np.float64)
    nw1 = 1
    for nb in range(NWVSOL):
        if nb == 0:
            nw1 = 1
        else:
            nw1 += int(nwvns0[nb - 1])
        nw2 = nw1 + int(nwvns0[nb]) - 1
        solfwv[nw1:nw2 + 1] = s0intv[nb]

    eirfwv = np.zeros(NWVTIR + 1, dtype=np.float64)
    tmp1 = (con_pi + con_pi) * CON_PLNK * CON_C * CON_C
    tmp2 = CON_PLNK * CON_C / (CON_BOLTZ * CON_T0C)
    for nw in range(1, NWVTIR + 1):
        tmp3 = 100.0 * float(nw)
        eirfwv[nw] = (tmp1 * tmp3**3) / (math.exp(tmp2 * tmp3) - 1.0)
    return solfwv, eirfwv


def _spectral_weights(
    iendwv: np.ndarray,
    *,
    con_pi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    solfwv, eirfwv = _set_spectrum(con_pi)
    solwaer = np.zeros((NSWBND, NAERBND), dtype=np.float64)
    solbnd = np.zeros(NSWBND, dtype=np.float64)
    eirwaer = np.zeros((NLWBND, NAERBND), dtype=np.float64)
    eirbnd = np.zeros(NLWBND, dtype=np.float64)
    nv1 = np.zeros(NSWBND, dtype=np.int64)
    nv2 = np.zeros(NSWBND, dtype=np.int64)
    nr1 = np.zeros(NLWBND, dtype=np.int64)
    nr2 = np.zeros(NLWBND, dtype=np.int64)

    ibs = 0
    ibe = 0
    wvs = WVNSW1[0]
    wve = WVNSW1[0]
    nv_aod = 0
    for ib in range(1, NSWBND):
        if WVNSW2[ib] >= 1.0e4 / 0.55 >= WVNSW1[ib]:
            nv_aod = ib
        if WVNSW1[ib] < wvs:
            wvs = WVNSW1[ib]
            ibs = ib
        if WVNSW1[ib] > wve:
            wve = WVNSW1[ib]
            ibe = ib

    for ib in range(NSWBND):
        ii = 0
        iw1 = int(round(float(WVNSW1[ib])))
        iw2 = int(round(float(WVNSW2[ib])))
        while iw1 > int(iendwv[ii]):
            if ii == NAERBND - 1:
                break
            ii += 1
        if ib == ibs:
            sumsol = 0.0
        else:
            sumsol = -0.5 * solfwv[iw1]
        fac = 0.0 if ib == ibe else -0.5
        solbnd[ib] = sumsol
        nv1[ib] = ii
        for iw in range(iw1, iw2 + 1):
            solbnd[ib] += solfwv[iw]
            sumsol += solfwv[iw]
            if iw == int(iendwv[ii]):
                solwaer[ib, ii] = sumsol
                if ii < NAERBND - 1:
                    sumsol = 0.0
                    ii += 1
        if iw2 != int(iendwv[ii]):
            solwaer[ib, ii] = sumsol
        tmp = fac * solfwv[iw2]
        solwaer[ib, ii] += tmp
        solbnd[ib] += tmp
        nv2[ib] = ii

    ibs = 0
    ibe = 0
    wvs = WVNLW1[0]
    wve = WVNLW1[0]
    for ib in range(1, NLWBND):
        if WVNLW1[ib] < wvs:
            wvs = WVNLW1[ib]
            ibs = ib
        if WVNLW1[ib] > wve:
            wve = WVNLW1[ib]
            ibe = ib

    for ib in range(NLWBND):
        ii = 0
        iw1 = int(round(float(WVNLW1[ib])))
        iw2 = int(round(float(WVNLW2[ib])))
        while iw1 > int(iendwv[ii]):
            if ii == NAERBND - 1:
                break
            ii += 1
        if ib == ibs:
            sumir = 0.0
        else:
            sumir = -0.5 * eirfwv[iw1]
        fac = 0.0 if ib == ibe else -0.5
        eirbnd[ib] = sumir
        nr1[ib] = ii
        for iw in range(iw1, iw2 + 1):
            eirbnd[ib] += eirfwv[iw]
            sumir += eirfwv[iw]
            if iw == int(iendwv[ii]):
                eirwaer[ib, ii] = sumir
                if ii < NAERBND - 1:
                    sumir = 0.0
                    ii += 1
        if iw2 != int(iendwv[ii]):
            eirwaer[ib, ii] = sumir
        tmp = fac * eirfwv[iw2]
        eirwaer[ib, ii] += tmp
        eirbnd[ib] += tmp
        nr2[ib] = ii

    return solwaer, solbnd, eirwaer, eirbnd, nv1, nv2, nr1, nr2, nv_aod


def _optavg(
    rhidext0: np.ndarray,
    rhidsca0: np.ndarray,
    rhidssa0: np.ndarray,
    rhidasy0: np.ndarray,
    rhdpext0: np.ndarray,
    rhdpsca0: np.ndarray,
    rhdpssa0: np.ndarray,
    rhdpasy0: np.ndarray,
    straext0: np.ndarray,
    weights: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    solwaer, solbnd, eirwaer, eirbnd, nv1, nv2, nr1, nr2, nv_aod = weights
    extrhi = np.zeros((NCM1, NSWLWBD), dtype=np.float64)
    scarhi = np.zeros((NCM1, NSWLWBD), dtype=np.float64)
    ssarhi = np.zeros((NCM1, NSWLWBD), dtype=np.float64)
    asyrhi = np.zeros((NCM1, NSWLWBD), dtype=np.float64)
    extrhd = np.zeros((NRHLEV, NCM2, NSWLWBD), dtype=np.float64)
    scarhd = np.zeros((NRHLEV, NCM2, NSWLWBD), dtype=np.float64)
    ssarhd = np.zeros((NRHLEV, NCM2, NSWLWBD), dtype=np.float64)
    asyrhd = np.zeros((NRHLEV, NCM2, NSWLWBD), dtype=np.float64)
    extstra = np.zeros(NSWLWBD, dtype=np.float64)

    for nb in range(NSWBND):
        rsolbd = 1.0 / solbnd[nb]
        for nc in range(NCM1):
            sumk = sums = sumok = sumokg = sumreft = 0.0
            for ni in range(int(nv1[nb]), int(nv2[nb]) + 1):
                sp = math.sqrt((1.0 - rhidssa0[ni, nc]) / (1.0 - rhidssa0[ni, nc] * rhidasy0[ni, nc]))
                reft = (1.0 - sp) / (1.0 + sp)
                weight = solwaer[nb, ni]
                sumreft += reft * weight
                sumk += rhidext0[ni, nc] * weight
                sums += rhidsca0[ni, nc] * weight
                sumok += rhidssa0[ni, nc] * weight * rhidext0[ni, nc]
                sumokg += rhidssa0[ni, nc] * weight * rhidext0[ni, nc] * rhidasy0[ni, nc]
            refb = sumreft * rsolbd
            extrhi[nc, nb] = sumk * rsolbd
            scarhi[nc, nb] = sums * rsolbd
            asyrhi[nc, nb] = sumokg / (sumok + 1.0e-10)
            ssarhi[nc, nb] = 4.0 * refb / ((1.0 + refb) ** 2 - asyrhi[nc, nb] * (1.0 - refb) ** 2)

        for nc in range(NCM2):
            for nh in range(NRHLEV):
                sumk = sums = sumok = sumokg = sumreft = 0.0
                for ni in range(int(nv1[nb]), int(nv2[nb]) + 1):
                    sp = math.sqrt((1.0 - rhdpssa0[ni, nh, nc]) / (1.0 - rhdpssa0[ni, nh, nc] * rhdpasy0[ni, nh, nc]))
                    reft = (1.0 - sp) / (1.0 + sp)
                    weight = solwaer[nb, ni]
                    sumreft += reft * weight
                    sumk += rhdpext0[ni, nh, nc] * weight
                    sums += rhdpsca0[ni, nh, nc] * weight
                    sumok += rhdpssa0[ni, nh, nc] * weight * rhdpext0[ni, nh, nc]
                    sumokg += rhdpssa0[ni, nh, nc] * weight * rhdpext0[ni, nh, nc] * rhdpasy0[ni, nh, nc]
                refb = sumreft * rsolbd
                extrhd[nh, nc, nb] = sumk * rsolbd
                scarhd[nh, nc, nb] = sums * rsolbd
                asyrhd[nh, nc, nb] = sumokg / (sumok + 1.0e-10)
                ssarhd[nh, nc, nb] = 4.0 * refb / ((1.0 + refb) ** 2 - asyrhd[nh, nc, nb] * (1.0 - refb) ** 2)

        extstra[nb] = sum(straext0[ni] * solwaer[nb, ni] for ni in range(int(nv1[nb]), int(nv2[nb]) + 1)) * rsolbd

    for nb in range(NLWBND):
        ib = NSWBND + nb
        rirbd = 1.0 / eirbnd[nb]
        for nc in range(NCM1):
            sumk = sums = sumok = sumokg = sumreft = 0.0
            for ni in range(int(nr1[nb]), int(nr2[nb]) + 1):
                sp = math.sqrt((1.0 - rhidssa0[ni, nc]) / (1.0 - rhidssa0[ni, nc] * rhidasy0[ni, nc]))
                reft = (1.0 - sp) / (1.0 + sp)
                weight = eirwaer[nb, ni]
                sumreft += reft * weight
                sumk += rhidext0[ni, nc] * weight
                sums += rhidsca0[ni, nc] * weight
                sumok += rhidssa0[ni, nc] * weight * rhidext0[ni, nc]
                sumokg += rhidssa0[ni, nc] * weight * rhidext0[ni, nc] * rhidasy0[ni, nc]
            refb = sumreft * rirbd
            extrhi[nc, ib] = sumk * rirbd
            scarhi[nc, ib] = sums * rirbd
            asyrhi[nc, ib] = sumokg / (sumok + 1.0e-10)
            ssarhi[nc, ib] = 4.0 * refb / ((1.0 + refb) ** 2 - asyrhi[nc, ib] * (1.0 - refb) ** 2)

        for nc in range(NCM2):
            for nh in range(NRHLEV):
                sumk = sums = sumok = sumokg = sumreft = 0.0
                for ni in range(int(nr1[nb]), int(nr2[nb]) + 1):
                    sp = math.sqrt((1.0 - rhdpssa0[ni, nh, nc]) / (1.0 - rhdpssa0[ni, nh, nc] * rhdpasy0[ni, nh, nc]))
                    reft = (1.0 - sp) / (1.0 + sp)
                    weight = eirwaer[nb, ni]
                    sumreft += reft * weight
                    sumk += rhdpext0[ni, nh, nc] * weight
                    sums += rhdpsca0[ni, nh, nc] * weight
                    sumok += rhdpssa0[ni, nh, nc] * weight * rhdpext0[ni, nh, nc]
                    sumokg += rhdpssa0[ni, nh, nc] * weight * rhdpext0[ni, nh, nc] * rhdpasy0[ni, nh, nc]
                refb = sumreft * rirbd
                extrhd[nh, nc, ib] = sumk * rirbd
                scarhd[nh, nc, ib] = sums * rirbd
                asyrhd[nh, nc, ib] = sumokg / (sumok + 1.0e-10)
                ssarhd[nh, nc, ib] = 4.0 * refb / ((1.0 + refb) ** 2 - asyrhd[nh, nc, ib] * (1.0 - refb) ** 2)

        extstra[ib] = sum(straext0[ni] * eirwaer[nb, ni] for ni in range(int(nr1[nb]), int(nr2[nb]) + 1)) * rirbd

    return extrhi, scarhi, ssarhi, asyrhi, extrhd, scarhd, ssarhd, asyrhd, extstra, int(nv_aod)


def _read_volcanic_file(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    payload = lines[1:1 + 4 * 10]
    values = [int(float(item)) for item in re.findall(r"[-+]?\d+", "\n".join(payload))]
    if len(values) != 12 * 4 * 10:
        raise ValueError(f"expected 480 volcanic aerosol integers, got {len(values)}")
    return np.asarray(values, dtype=np.int64).reshape((10, 4, 12)).transpose(2, 1, 0).copy()


@lru_cache(maxsize=4)
def _load_raw_aerosol_tables(month: int, year: int, con_pi: float) -> _RawAerosolTables:
    aerosol_path = _physics_input_dir() / "aerosol.dat"
    if not aerosol_path.exists():
        raise FileNotFoundError(aerosol_path)
    aerosol_sha = _sha256(aerosol_path)
    if aerosol_sha != AEROSOL_DAT_SHA256:
        raise ValueError(f"frozen aerosol.dat hash mismatch: expected {AEROSOL_DAT_SHA256}, got {aerosol_sha}")

    lines = aerosol_path.read_text(encoding="utf-8").splitlines()
    kprfg, idxcg, cmixg, denng = _read_monthly_distribution(lines, month)
    index = 12 * (2 + JMXAE * IMXAE)
    index += 1
    iendwv, index = _read_int_block(lines, index, NAERBND)
    index += 1
    haer_values, index = _read_float_block(lines, index, NDM * NAE)
    index += 1
    prsref_values, index = _read_float_block(lines, index, NDM * NAE)
    index += 1
    rhidext0_values, index = _read_float_block(lines, index, NAERBND * NCM1)
    index += 1
    rhidsca0_values, index = _read_float_block(lines, index, NAERBND * NCM1)
    index += 1
    rhidssa0_values, index = _read_float_block(lines, index, NAERBND * NCM1)
    index += 1
    rhidasy0_values, index = _read_float_block(lines, index, NAERBND * NCM1)
    index += 1
    rhdpext0_values, index = _read_float_block(lines, index, NAERBND * NRHLEV * NCM2)
    index += 1
    rhdpsca0_values, index = _read_float_block(lines, index, NAERBND * NRHLEV * NCM2)
    index += 1
    rhdpssa0_values, index = _read_float_block(lines, index, NAERBND * NRHLEV * NCM2)
    index += 1
    rhdpasy0_values, index = _read_float_block(lines, index, NAERBND * NRHLEV * NCM2)
    index += 1
    straext0, index = _read_float_block(lines, index, NAERBND)

    haer = haer_values.reshape((NDM, NAE), order="F")
    prsref = prsref_values.reshape((NDM, NAE), order="F")
    sigref = 0.001 * prsref
    rhidext0 = rhidext0_values.reshape((NAERBND, NCM1), order="F")
    rhidsca0 = rhidsca0_values.reshape((NAERBND, NCM1), order="F")
    rhidssa0 = rhidssa0_values.reshape((NAERBND, NCM1), order="F")
    rhidasy0 = rhidasy0_values.reshape((NAERBND, NCM1), order="F")
    rhdpext0 = rhdpext0_values.reshape((NAERBND, NRHLEV, NCM2), order="F")
    rhdpsca0 = rhdpsca0_values.reshape((NAERBND, NRHLEV, NCM2), order="F")
    rhdpssa0 = rhdpssa0_values.reshape((NAERBND, NRHLEV, NCM2), order="F")
    rhdpasy0 = rhdpasy0_values.reshape((NAERBND, NRHLEV, NCM2), order="F")

    weights = _spectral_weights(iendwv, con_pi=con_pi)
    extrhi, scarhi, ssarhi, asyrhi, extrhd, scarhd, ssarhd, asyrhd, extstra, nv_aod = _optavg(
        rhidext0,
        rhidsca0,
        rhidssa0,
        rhidasy0,
        rhdpext0,
        rhdpsca0,
        rhdpssa0,
        rhdpasy0,
        straext0,
        weights,
    )

    if year < MIN_VOLCANIC_YEAR or year > MAX_VOLCANIC_YEAR:
        ivolae = np.ones((12, 4, 10), dtype=np.int64)
        volcanic_sha = "fortran_out_of_range_lowest_value"
    else:
        decade_start = year - (year % 10)
        volcanic_path = _physics_input_dir() / f"volcanic_aerosols_{decade_start:04d}-{decade_start + 9:04d}.txt"
        if not volcanic_path.exists():
            raise FileNotFoundError(volcanic_path)
        volcanic_sha = _sha256(volcanic_path)
        expected_volcanic_sha = VOLCANIC_AEROSOL_SHA256.get((decade_start, decade_start + 9))
        if expected_volcanic_sha is not None and volcanic_sha != expected_volcanic_sha:
            raise ValueError(
                "frozen volcanic aerosol hash mismatch: "
                f"expected {expected_volcanic_sha}, got {volcanic_sha}"
            )
        ivolae = _read_volcanic_file(volcanic_path)

    return _RawAerosolTables(
        haer=haer,
        prsref=prsref,
        sigref=sigref,
        kprfg=kprfg,
        idxcg=idxcg,
        cmixg=cmixg,
        denng=denng,
        extrhi=extrhi,
        scarhi=scarhi,
        ssarhi=ssarhi,
        asyrhi=asyrhi,
        extrhd=extrhd,
        scarhd=scarhd,
        ssarhd=ssarhd,
        asyrhd=asyrhd,
        extstra=extstra,
        ivolae=ivolae,
        nv_aod=nv_aod,
        month=month,
        year=year,
        aerosol_sha256=aerosol_sha,
        volcanic_sha256=volcanic_sha,
    )


@torch.inference_mode(False)
def _tables_for_device(device: torch.device, *, month: int, year: int, con_pi: float) -> _AerosolTables:
    cache_key = (device.type, device.index, month, year, float(con_pi))
    cached = _AEROSOL_DEVICE_TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    raw = _load_raw_aerosol_tables(month, year, con_pi)
    kwargs = {"device": device}
    tables = _AerosolTables(
        haer=torch.as_tensor(raw.haer, dtype=torch.float64, **kwargs),
        prsref=torch.as_tensor(raw.prsref, dtype=torch.float64, **kwargs),
        sigref=torch.as_tensor(raw.sigref, dtype=torch.float64, **kwargs),
        kprfg=torch.as_tensor(raw.kprfg, dtype=torch.int64, **kwargs),
        idxcg=torch.as_tensor(raw.idxcg, dtype=torch.int64, **kwargs),
        cmixg=torch.as_tensor(raw.cmixg, dtype=torch.float64, **kwargs),
        denng=torch.as_tensor(raw.denng, dtype=torch.float64, **kwargs),
        extrhi=torch.as_tensor(raw.extrhi, dtype=torch.float64, **kwargs),
        scarhi=torch.as_tensor(raw.scarhi, dtype=torch.float64, **kwargs),
        ssarhi=torch.as_tensor(raw.ssarhi, dtype=torch.float64, **kwargs),
        asyrhi=torch.as_tensor(raw.asyrhi, dtype=torch.float64, **kwargs),
        extrhd=torch.as_tensor(raw.extrhd, dtype=torch.float64, **kwargs),
        scarhd=torch.as_tensor(raw.scarhd, dtype=torch.float64, **kwargs),
        ssarhd=torch.as_tensor(raw.ssarhd, dtype=torch.float64, **kwargs),
        asyrhd=torch.as_tensor(raw.asyrhd, dtype=torch.float64, **kwargs),
        extstra=torch.as_tensor(raw.extstra, dtype=torch.float64, **kwargs),
        ivolae=torch.as_tensor(raw.ivolae, dtype=torch.int64, **kwargs),
        nv_aod=raw.nv_aod,
        month=raw.month,
        year=raw.year,
    )
    _AEROSOL_DEVICE_TABLE_CACHE[cache_key] = tables
    return tables


def _to_real(value: TensorOrString, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64 (the working dtype), got {value.dtype}")
    return value


def _to_int(value: TensorOrString, *, name: str) -> int:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    return int(value.detach().to(device="cpu", dtype=torch.int64).reshape(-1)[0].item())


def _to_host_float(value: TensorOrString, *, name: str) -> float:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    return float(value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)[0].item())


def _hook(boundaries: Mapping[str, Mapping[str, TensorOrString]]) -> Mapping[str, TensorOrString]:
    hook = boundaries.get("ABCDEFG_aerosol_inputs")
    if hook is not None:
        return hook
    return _aerosol_inputs_from_profile_boundaries(boundaries)


def _control_tensor(
    controls: Mapping[str, object],
    name: str,
    *,
    default: int | float,
    dtype: torch.dtype,
    device: torch.device,
    shape: tuple[int, ...] = (1,),
) -> torch.Tensor:
    value = controls.get(name, default)
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=dtype).reshape(-1)
        size = math.prod(shape)
        if tensor.numel() == size:
            return tensor.reshape(shape)
        if tensor.numel() == 1:
            return tensor.reshape((1,) * len(shape)).expand(shape)
        if size == 1:
            return tensor[:1].reshape(shape)
        raise ValueError(f"runtime control {name} has {tensor.numel()} values, expected 1 or {size}")
    return torch.full(shape, value, dtype=dtype, device=device)


def _aerosol_inputs_from_profile_boundaries(
    boundaries: Mapping[str, Mapping[str, TensorOrString]],
) -> Mapping[str, TensorOrString]:
    """Build the OPAC G input surface from target-owned profile tensors when available.

    The generated cap source hook remains the strict branch-oracle path.  This
    fallback removes a source-hook dependency for the sealed profile/runtime path
    by using target-produced B fields plus raw AB pre-call host tensors.  Discrete
    aerosol controls are runtime literals, not differentiable inputs.
    """

    pre = boundaries.get("AB")
    raw = boundaries.get("AB_pre_inputs")
    if pre is None or raw is None:
        raise KeyError("missing ABCDEFG_aerosol_inputs source-hook boundary")
    p_lay = _to_real(pre["p_lay"], name="AB.p_lay")
    nlay, ncol = p_lay.shape
    device = p_lay.device
    dtype = p_lay.dtype
    controls = boundaries.get("runtime_controls", {})
    if not isinstance(controls, Mapping):
        controls = {}
    lsmask = _control_tensor(
        controls,
        "lsmask",
        default=0.0,
        dtype=dtype,
        device=device,
        shape=(ncol,),
    )
    lon = _to_real(raw.get("xlon", raw.get("lon")), name="AB_pre_inputs.xlon").to(device=device)
    lat = _to_real(raw.get("xlat", raw.get("lat")), name="AB_pre_inputs.xlat").to(device=device)
    if lon.numel() == 1 and ncol != 1:
        lon = lon.reshape(1).expand(ncol)
    if lat.numel() == 1 and ncol != 1:
        lat = lat.reshape(1).expand(ncol)
    return {
        "doSWrad": pre.get("doSWrad", raw["doSWrad"]),
        "doLWrad": pre.get("doLWrad", raw["doLWrad"]),
        "top_at_1": pre.get("top_at_1", raw["top_at_1_initial"]),
        "iaermdl": _control_tensor(controls, "iaermdl", default=5, dtype=torch.int64, device=device),
        "iaerflg": _control_tensor(controls, "iaerflg", default=111, dtype=torch.int64, device=device),
        "nDay": pre.get("nDay", torch.zeros((1,), dtype=torch.int64, device=device)),
        "idxday": pre.get("idxday", torch.zeros((1,), dtype=torch.int64, device=device)),
        "nCol": torch.full((1,), ncol, dtype=torch.int64, device=device),
        "nLev": torch.full((1,), nlay + 1, dtype=torch.int64, device=device),
        "p_lev": _to_real(pre["p_lev"], name="AB.p_lev"),
        "p_lay": p_lay,
        "p_lk": _to_real(raw.get("prslk", p_lay), name="AB_pre_inputs.prslk").to(device=device),
        "tv_lay": _to_real(pre["tv_lay"], name="AB.tv_lay"),
        "relhum": _to_real(pre["relhum"], name="AB.relhum"),
        "lsmask": lsmask,
        "lon": lon.reshape(ncol),
        "lat": lat.reshape(ncol),
        "con_pi": _to_real(raw["con_pi"], name="AB_pre_inputs.con_pi").to(device=device),
        "con_rd": _to_real(raw["con_rd"], name="AB_pre_inputs.con_rd").to(device=device),
        "con_g": _to_real(raw["con_g"], name="AB_pre_inputs.con_g").to(device=device),
        "tracer": _to_real(raw["qgrs"], name="AB_pre_inputs.qgrs").to(device=device),
        "aerfld": _zeros_broadcast_last_dim((NSPC1, nlay, ncol), dtype=dtype, device=device),
    }


def _zeros_broadcast_last_dim(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not shape:
        return torch.zeros(shape, dtype=dtype, device=device)
    base_shape = (*shape[:-1], 1)
    return torch.zeros(base_shape, dtype=dtype, device=device).expand(*shape)


def _expand_last_dim(value: torch.Tensor, ncol: int) -> torch.Tensor:
    if value.ndim == 0 or value.shape[-1] == ncol:
        return value
    if value.shape[-1] != 1:
        raise ValueError(f"cannot expand last dimension {value.shape[-1]} to ncol={ncol}")
    return value.expand(*value.shape[:-1], ncol)


def _is_single_column_broadcast(value: torch.Tensor, *, ncol: int, column_axis: int) -> bool:
    if value.ndim == 0 or ncol <= 1:
        return True
    axis = column_axis if column_axis >= 0 else value.ndim + column_axis
    if axis < 0 or axis >= value.ndim or value.shape[axis] != ncol:
        return True
    return value.stride(axis) == 0


def _aerosol_hook_is_single_column_broadcast(
    *,
    p_lev: torch.Tensor,
    p_lay: torch.Tensor,
    tv_lay: torch.Tensor,
    relhum: torch.Tensor,
    lsmask: torch.Tensor,
    lon: torch.Tensor,
    lat: torch.Tensor,
    ncol: int,
) -> bool:
    return all(
        (
            _is_single_column_broadcast(p_lev, ncol=ncol, column_axis=-1),
            _is_single_column_broadcast(p_lay, ncol=ncol, column_axis=-1),
            _is_single_column_broadcast(tv_lay, ncol=ncol, column_axis=-1),
            _is_single_column_broadcast(relhum, ncol=ncol, column_axis=-1),
            _is_single_column_broadcast(lsmask, ncol=ncol, column_axis=0),
            _is_single_column_broadcast(lon, ncol=ncol, column_axis=0),
            _is_single_column_broadcast(lat, ncol=ncol, column_axis=0),
        )
    )


def _expanded_single_column_aerosol_result(
    *,
    aerosw: torch.Tensor,
    aerolw: torch.Tensor,
    aerodp_col: torch.Tensor,
    nlay: int,
    ncol: int,
    dtype: torch.dtype,
    device: torch.device,
    do_sw: bool,
    do_lw: bool,
    n_day: int,
) -> RrtmgpAerosolResult:
    zero_sw = _zeros_broadcast_last_dim((NBDSW, nlay, ncol), dtype=dtype, device=device)
    zero_lw = _zeros_broadcast_last_dim((NBDLW, nlay, ncol), dtype=dtype, device=device)
    if do_sw:
        aersw2_tau_one = aerosw[:, :, 0].transpose(0, 1).unsqueeze(-1)
        aersw2_ssa_one = aerosw[:, :, 1].transpose(0, 1).unsqueeze(-1)
        aersw2_g_one = aerosw[:, :, 2].transpose(0, 1).unsqueeze(-1)
        if n_day > 0:
            sw_order = (torch.arange(NBDSW, dtype=torch.int64, device=device) - 1) % NBDSW
            aersw_tau = _expand_last_dim(aersw2_tau_one.index_select(0, sw_order), ncol)
            aersw_ssa = _expand_last_dim(aersw2_ssa_one.index_select(0, sw_order), ncol)
            aersw_g = _expand_last_dim(aersw2_g_one.index_select(0, sw_order), ncol)
        else:
            aersw_tau = zero_sw
            aersw_ssa = zero_sw
            aersw_g = zero_sw
        aerodp = _expand_last_dim(aerodp_col.unsqueeze(-1), ncol)
    else:
        aersw_tau = zero_sw
        aersw_ssa = zero_sw
        aersw_g = zero_sw
        aerodp = _zeros_broadcast_last_dim((NSPC1, ncol), dtype=dtype, device=device)
    if do_lw:
        aerlw_tau = _expand_last_dim(aerolw[:, :, 0].transpose(0, 1).unsqueeze(-1), ncol)
        aerlw_ssa = _expand_last_dim(aerolw[:, :, 1].transpose(0, 1).unsqueeze(-1), ncol)
        aerlw_g = _expand_last_dim(aerolw[:, :, 2].transpose(0, 1).unsqueeze(-1), ncol)
    else:
        aerlw_tau = zero_lw
        aerlw_ssa = zero_lw
        aerlw_g = zero_lw
    return RrtmgpAerosolResult(
        fields={
            "aerodp": aerodp,
            "aersw_tau": aersw_tau,
            "aersw_ssa": aersw_ssa,
            "aersw_g": aersw_g,
            "aerlw_tau": aerlw_tau,
            "aerlw_ssa": aerlw_ssa,
            "aerlw_g": aerlw_g,
            "ext550": _zeros_broadcast_last_dim((nlay, ncol), dtype=dtype, device=device),
            "errmsg": "",
            "errflg": torch.zeros((1,), dtype=torch.int64, device=device),
        },
        runtime_kind="torch_rrtmgp_aerosol_opac_g_repeated_column_view",
    )






def _ordered_grid_cells(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Replay the source's carried search index with a tensor prefix scan.

    At a grid line both adjacent cells contain the point. The Fortran search
    starts from the preceding column's cell, so its choice affects the class
    maximum over neighboring cells even when interpolation weights are zero.
    Each search is a clamp; clamp composition is associative. A doubling scan
    preserves that source ordering without reads or a Python loop over columns.
    """
    position = torch.arange(lower.numel(), device=lower.device)
    stride = 1
    while stride < lower.numel():
        previous_lower = torch.roll(lower, stride)
        previous_upper = torch.roll(upper, stride)
        composed_lower = torch.minimum(torch.maximum(previous_lower, lower), upper)
        composed_upper = torch.minimum(torch.maximum(previous_upper, lower), upper)
        valid = position >= stride
        lower = torch.where(valid, composed_lower, lower)
        upper = torch.where(valid, composed_upper, upper)
        stride *= 2
    return torch.minimum(torch.maximum(torch.zeros_like(lower), lower), upper)


def _grid_mapping_batch(
    lon_rad: torch.Tensor,
    lat_rad: torch.Tensor,
    *,
    con_pi: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensorized OPAC grid mapping over columns.

    The scalar Fortran-compatible helper remains the reference for single
    columns.  This batch form keeps all per-column map decisions on device.
    """

    dtype = lon_rad.dtype
    dltg = 360.0 / float(IMXAE)
    alon = lon_rad.reshape(-1) * (180.0 / con_pi)
    alon = torch.where(alon < 0.0, alon + 360.0, alon)
    lat = lat_rad.reshape(-1) * (180.0 / con_pi)

    lon_cell = torch.floor(alon / dltg).to(dtype=torch.int64)
    lon_cell_delta = alon - dltg * lon_cell.to(dtype=dtype)
    lon_cell = torch.where(lon_cell_delta < 0.0, lon_cell - 1,
                           torch.where(lon_cell_delta > dltg, lon_cell + 1, lon_cell))
    # Test the original subtraction, rather than a rounded quotient, to retain
    # the source's inclusive upper boundary and floating-point association.
    lon_previous = torch.clamp(lon_cell - 1, min=0, max=IMXAE - 1)
    lon_delta = alon - dltg * lon_previous.to(dtype=dtype)
    lon_lower = torch.where((lon_delta >= 0.0) & (lon_delta <= dltg), lon_previous, lon_cell)
    i1 = _ordered_grid_cells(torch.clamp(lon_lower, 0, IMXAE - 1), torch.clamp(lon_cell, 0, IMXAE - 1))
    i2 = (i1 + 1) % IMXAE
    wi = (alon - i1.to(dtype=dtype) * dltg) * (1.0 / dltg)
    kpi = torch.where(wi <= 0.5, i1, i2)

    lat_cell = torch.floor((90.0 - lat) / dltg).to(dtype=torch.int64)
    lat_cell_delta = (90.0 - dltg * lat_cell.to(dtype=dtype)) - lat
    lat_cell = torch.where(lat_cell_delta < 0.0, lat_cell - 1,
                           torch.where(lat_cell_delta > dltg, lat_cell + 1, lat_cell))
    lat_previous = torch.clamp(lat_cell - 1, min=0, max=JMXAE - 2)
    lat_delta = (90.0 - dltg * lat_previous.to(dtype=dtype)) - lat
    lat_lower = torch.where((lat_delta >= 0.0) & (lat_delta <= dltg), lat_previous, lat_cell)
    j1 = _ordered_grid_cells(torch.clamp(lat_lower, 0, JMXAE - 2), torch.clamp(lat_cell, 0, JMXAE - 2))
    j2 = j1 + 1
    wj = ((90.0 - dltg * j1.to(dtype=dtype)) - lat) * (1.0 / dltg)
    kpj = torch.clamp(torch.where(wj <= 0.5, j1, j2), min=0, max=JMXAE - 1)
    return i1, i2, j1, j2, kpi, kpj, wi, wj


def _bilinear_grid_values(
    values: torch.Tensor,
    *,
    i1: torch.Tensor,
    i2: torch.Tensor,
    j1: torch.Tensor,
    j2: torch.Tensor,
    w11: torch.Tensor,
    w12: torch.Tensor,
    w21: torch.Tensor,
    w22: torch.Tensor,
) -> torch.Tensor:
    return (
        w11 * values[i1, j1]
        + w12 * values[i1, j2]
        + w21 * values[i2, j1]
        + w22 * values[i2, j2]
    )


def _aerosol_grid_mixture_kernel(
    kprfg: torch.Tensor,
    haer: torch.Tensor,
    denng: torch.Tensor,
    idxcg: torch.Tensor,
    cmixg: torch.Tensor,
    lon_rad: torch.Tensor,
    lat_rad: torch.Tensor,
    lsmask: torch.Tensor,
    con_pi: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve OPAC grid classes and mixtures in one flat tensor island."""

    dtype = lon_rad.dtype
    device = lon_rad.device
    ncol = lon_rad.numel()
    i1, i2, j1, j2, kpi, kpj, wi, wj = _grid_mapping_batch(lon_rad, lat_rad, con_pi=con_pi)
    w11 = (1.0 - wi) * (1.0 - wj)
    w12 = (1.0 - wi) * wj
    w21 = wi * (1.0 - wj)
    w22 = wi * wj
    kp = kprfg[kpi, kpj].to(dtype=torch.int64)
    kpa = torch.maximum(
        torch.maximum(kprfg[i1, j1], kprfg[i1, j2]),
        torch.maximum(kprfg[i2, j1], kprfg[i2, j2]),
    ).to(dtype=torch.int64)
    h1 = haer[0, kp - 1]
    mismatch = kp != kpa
    sea_land = lsmask.reshape(-1).to(device=device, dtype=dtype) > 0.0
    mask_kpa6 = mismatch & (kpa == 6)
    mask_kpa7 = mismatch & (kpa == 7)
    mask_other = mismatch & ~(mask_kpa6 | mask_kpa7)
    kp = torch.where(mask_kpa6 & sea_land, torch.full_like(kp, 7), torch.where(mask_kpa6, kpa, kp))
    kp = torch.where(mask_kpa7 & ~sea_land, torch.full_like(kp, 6), torch.where(mask_kpa7, kpa, kp))
    kp = torch.where(mask_other, kpa, kp)
    h1_default = haer[0, torch.clamp(kpa - 1, 0, NAE - 1)]
    h1 = torch.where(mask_other, h1_default, h1)
    h1 = torch.where(mask_kpa6 & sea_land, 0.5 * (haer[0, 5] + haer[0, 6]), h1)
    h1 = torch.where(mask_kpa6 & ~sea_land, haer[0, 5], h1)
    h1 = torch.where(mask_kpa7 & ~sea_land, 0.5 * (haer[0, 5] + haer[0, 6]), h1)
    h1 = torch.where(mask_kpa7 & sea_land, haer[0, 6], h1)

    blend_two = mask_kpa6 | mask_kpa7
    density_values: list[torch.Tensor] = []
    for mixture_index in range(2):
        values = _bilinear_grid_values(
            denng[mixture_index, :, :],
            i1=i1,
            i2=i2,
            j1=j1,
            j2=j2,
            w11=w11,
            w12=w12,
            w21=w21,
            w22=w22,
        )
        density_values.append(
            values if mixture_index == 0 else torch.where(blend_two, values, torch.zeros_like(values))
        )
    denn = torch.stack(density_values)

    species_ids = torch.arange(1, NCM + 1, dtype=torch.int64, device=device).reshape(NCM, 1)
    cmix = torch.zeros((NCM, ncol), dtype=dtype, device=device)
    for mixture_index in range(NXC):
        for weight, ii_idx, jj_idx in ((w11, i1, j1), (w12, i1, j2), (w21, i2, j1), (w22, i2, j2)):
            idx = idxcg[mixture_index, ii_idx, jj_idx].to(dtype=torch.int64)
            mix = cmixg[mixture_index, ii_idx, jj_idx].to(dtype=dtype) * weight
            cmix = cmix + (idx.reshape(1, ncol) == species_ids).to(dtype=dtype) * mix.reshape(1, ncol)
    return kp - 1, h1, denn, cmix


def _gather_rh_band(table: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, fraction: torch.Tensor) -> torch.Tensor:
    nlay, ncol = lower.shape
    table_t = table.to(device=fraction.device, dtype=fraction.dtype)
    nband = table_t.shape[-1]
    low = table_t.index_select(0, lower.reshape(-1)).reshape(nlay, ncol, nband).permute(0, 2, 1)
    high = table_t.index_select(0, upper.reshape(-1)).reshape(nlay, ncol, nband).permute(0, 2, 1)
    return low + fraction.unsqueeze(1) * (high - low)


def _safe_ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    mask = den != 0.0
    safe_den = torch.where(mask, den, torch.ones_like(den))
    return torch.where(mask, num / safe_den, torch.zeros_like(num))


def _aerosol_optical_depth_redistribution_slab(
    boundary: torch.Tensor,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one OPAC neighbor-limiter slab without packing its carry."""

    values = [boundary, *tau.unbind(dim=0)]
    one = torch.ones_like(values[0])
    for level in range(1, len(values)):
        current = values[level]
        neighbor = values[level - 1]
        ratio = torch.where(current > 0.0, neighbor / torch.where(current > 0.0, current, one), one)
        total = current + neighbor
        high = ratio > 30.0
        low = ratio < 0.03333
        values[level] = torch.where(high, 0.2 * total, torch.where(low, 0.8 * total, current))
        values[level - 1] = torch.where(high, 0.8 * total, torch.where(low, 0.2 * total, neighbor))
    return torch.stack(values[:-1]), values[-1]


def _aerosol_optical_depth_redistribution_compiled(
    tau: torch.Tensor,
    *,
    top_at_1: bool,
) -> torch.Tensor:
    """Run the OPAC vertical limiter in fixed-size PyTorch compile slabs."""

    work = tau if top_at_1 else torch.flip(tau, dims=(0,))
    if work.shape[0] <= 1:
        return tau
    completed: list[torch.Tensor] = []
    boundary = work[0]
    slab_levels = _aerosol_slab_levels(work, _AEROSOL_REDUCTION_SLAB_LEVELS)
    for start in range(1, work.shape[0], slab_levels):
        end = min(start + slab_levels, work.shape[0])
        finished, boundary = run_compile_island(
            "aerosol",
            "optical_depth_redistribution_slab",
            _aerosol_optical_depth_redistribution_slab,
            boundary,
            work[start:end],
        )
        completed.append(finished)
    completed.append(boundary.unsqueeze(0))
    redistributed = torch.cat(completed, dim=0)
    return redistributed if top_at_1 else torch.flip(redistributed, dims=(0,))


def _aerosol_boundary_layer_optics_kernel(
    cmix: torch.Tensor,
    denn0: torch.Tensor,
    delz: torch.Tensor,
    ih1: torch.Tensor,
    ih2: torch.Tensor,
    rdrh: torch.Tensor,
    mask1: torch.Tensor,
    extrhi: torch.Tensor,
    scarhi: torch.Tensor,
    ssarhi: torch.Tensor,
    asyrhi: torch.Tensor,
    extrhd: torch.Tensor,
    scarhd: torch.Tensor,
    ssarhd: torch.Tensor,
    asyrhd: torch.Tensor,
    nv_aod: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accumulate the ten OPAC boundary-layer components in one tensor island."""

    dtype = delz.dtype
    device = delz.device
    nlay, ncol = delz.shape
    delz_b = delz.unsqueeze(1)
    ext1 = torch.zeros((nlay, NSWLWBD, ncol), dtype=dtype, device=device)
    sca1 = torch.zeros_like(ext1)
    ssa_num1 = torch.zeros_like(ext1)
    asy_num1 = torch.zeros_like(ext1)
    spcodp1 = torch.zeros((NSPC, ncol), dtype=dtype, device=device)
    for icmp in range(NCM):
        cm = cmix[icmp, :].reshape(1, 1, ncol)
        if icmp < NCM1:
            ex = extrhi[icmp, :].to(dtype=dtype).reshape(1, NSWLWBD, 1)
            sc = scarhi[icmp, :].to(dtype=dtype).reshape(1, NSWLWBD, 1)
            ss = ssarhi[icmp, :].to(dtype=dtype).reshape(1, NSWLWBD, 1)
            ay = asyrhi[icmp, :].to(dtype=dtype).reshape(1, NSWLWBD, 1)
        else:
            ic1 = icmp - NCM1
            ex = _gather_rh_band(extrhd[:, ic1, :], ih1, ih2, rdrh)
            sc = _gather_rh_band(scarhd[:, ic1, :], ih1, ih2, rdrh)
            ss = _gather_rh_band(ssarhd[:, ic1, :], ih1, ih2, rdrh)
            ay = _gather_rh_band(asyrhd[:, ic1, :], ih1, ih2, rdrh)
        tt0 = cm * ex
        ext1 = ext1 + tt0
        sca1 = sca1 + cm * sc
        ssa_num1 = ssa_num1 + cm * ss * ex
        asy_num1 = asy_num1 + cm * ay * sc
        species_idx = IDXSPC_TUPLE[icmp]
        spcodp1[species_idx, :] = spcodp1[species_idx, :] + (
            tt0[:, nv_aod, :] * denn0.reshape(1, ncol) * delz * mask1.squeeze(1).to(dtype=dtype)
        ).sum(dim=0)
    tau1 = ext1 * denn0 * delz_b
    ssa1 = torch.minimum(torch.ones_like(ext1), _safe_ratio(ssa_num1, ext1))
    asy1 = torch.minimum(torch.ones_like(sca1), _safe_ratio(asy_num1, sca1))
    return tau1, ssa1, asy1, spcodp1


def _aerosol_boundary_layer_optics_minimal_kernel(
    cmix: torch.Tensor,
    denn0: torch.Tensor,
    delz: torch.Tensor,
    ih1: torch.Tensor,
    ih2: torch.Tensor,
    rdrh: torch.Tensor,
    extrhi: torch.Tensor,
    scarhi: torch.Tensor,
    ssarhi: torch.Tensor,
    asyrhi: torch.Tensor,
    extrhd: torch.Tensor,
    scarhd: torch.Tensor,
    ssarhd: torch.Tensor,
    asyrhd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute all-band extinction and only downstream-live SW scattering."""

    dtype = delz.dtype
    device = delz.device
    nlay, ncol = delz.shape
    delz_b = delz.unsqueeze(1)
    ext1 = torch.zeros((nlay, NSWLWBD, ncol), dtype=dtype, device=device)
    sca1_sw = torch.zeros((nlay, NSWBND, ncol), dtype=dtype, device=device)
    ssa_num1_sw = torch.zeros_like(sca1_sw)
    asy_num1_sw = torch.zeros_like(sca1_sw)
    for icmp in range(NCM):
        cm = cmix[icmp, :].reshape(1, 1, ncol)
        if icmp < NCM1:
            ex = extrhi[icmp, :].to(dtype=dtype).reshape(1, NSWLWBD, 1)
            sc_sw = scarhi[icmp, :NSWBND].to(dtype=dtype).reshape(1, NSWBND, 1)
            ss_sw = ssarhi[icmp, :NSWBND].to(dtype=dtype).reshape(1, NSWBND, 1)
            ay_sw = asyrhi[icmp, :NSWBND].to(dtype=dtype).reshape(1, NSWBND, 1)
        else:
            ic1 = icmp - NCM1
            ex = _gather_rh_band(extrhd[:, ic1, :], ih1, ih2, rdrh)
            sc_sw = _gather_rh_band(scarhd[:, ic1, :NSWBND], ih1, ih2, rdrh)
            ss_sw = _gather_rh_band(ssarhd[:, ic1, :NSWBND], ih1, ih2, rdrh)
            ay_sw = _gather_rh_band(asyrhd[:, ic1, :NSWBND], ih1, ih2, rdrh)
        ext1 = ext1 + cm * ex
        sca1_sw = sca1_sw + cm * sc_sw
        ssa_num1_sw = ssa_num1_sw + cm * ss_sw * ex[:, :NSWBND, :]
        asy_num1_sw = asy_num1_sw + cm * ay_sw * sc_sw
    tau1 = ext1 * denn0 * delz_b
    ext1_sw = ext1[:, :NSWBND, :]
    ssa1_sw = torch.minimum(torch.ones_like(ext1_sw), _safe_ratio(ssa_num1_sw, ext1_sw))
    asy1_sw = torch.minimum(torch.ones_like(sca1_sw), _safe_ratio(asy_num1_sw, sca1_sw))
    return tau1, ssa1_sw, asy1_sw


def _aerosol_regime_merge_kernel(
    tau4: torch.Tensor,
    sw_lw_ssa: torch.Tensor,
    sw_lw_asy: torch.Tensor,
    tau3: torch.Tensor,
    ssa3: torch.Tensor,
    asy3: torch.Tensor,
    tau1: torch.Tensor,
    ssa1: torch.Tensor,
    asy1: torch.Tensor,
    tau2: torch.Tensor,
    ssa2: torch.Tensor,
    asy2: torch.Tensor,
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    mask3: torch.Tensor,
    mask4: torch.Tensor,
    mask5: torch.Tensor,
    spcodp1: torch.Tensor,
    ex01: torch.Tensor,
    ex02: torch.Tensor,
    ex03: torch.Tensor,
    delz: torch.Tensor,
    nv_aod: int,
    stratospheric_species_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge OPAC regimes and species diagnostics without wrapper-side mutation."""

    tauae = torch.zeros_like(tau1)
    ssaae = torch.ones_like(tau1)
    asyae = torch.zeros_like(tau1)
    tauae = torch.where(mask4, tau4, tauae)
    ssaae = torch.where(mask4 | mask5, sw_lw_ssa.expand_as(ssaae), ssaae)
    asyae = torch.where(mask4 | mask5, sw_lw_asy.expand_as(asyae), asyae)
    tauae = torch.where(mask3, tau3, tauae)
    ssaae = torch.where(mask3, ssa3, ssaae)
    asyae = torch.where(mask3, asy3, asyae)
    tauae = torch.where(mask1, tau1, tauae)
    ssaae = torch.where(mask1, ssa1, ssaae)
    asyae = torch.where(mask1, asy1, asyae)
    tauae = torch.where(mask2, tau2, tauae)
    ssaae = torch.where(mask2, ssa2, ssaae)
    asyae = torch.where(mask2, asy2, asyae)

    mask1d = mask3.squeeze(1).to(dtype=delz.dtype)
    mask2d = mask2.squeeze(1).to(dtype=delz.dtype)
    mask4d = mask4.squeeze(1).to(dtype=delz.dtype)
    rows = list(spcodp1.unbind(dim=0))
    rows[0] = rows[0] + (0.17e-3 * ex01[:, nv_aod, :] * 730.0 * delz * mask1d).sum(dim=0)
    rows[1] = rows[1] + (0.4 * ex02[:, nv_aod, :] * 730.0 * delz * mask1d).sum(dim=0)
    rows[2] = rows[2] + (0.59983 * ex03[:, nv_aod, :] * 730.0 * delz * mask1d).sum(dim=0)
    rows[stratospheric_species_index] = rows[stratospheric_species_index] + (
        tau4[:, nv_aod, :] * mask4d
    ).sum(dim=0)
    rows[0] = rows[0] + (tau2[:, nv_aod, :] * mask2d).sum(dim=0)
    return tauae, ssaae, asyae, torch.stack(rows)


def _aerosol_regime_merge_minimal_kernel(
    tau4: torch.Tensor,
    sw_ssa: torch.Tensor,
    sw_asy: torch.Tensor,
    tau3: torch.Tensor,
    ssa3_sw: torch.Tensor,
    asy3_sw: torch.Tensor,
    tau1: torch.Tensor,
    ssa1_sw: torch.Tensor,
    asy1_sw: torch.Tensor,
    tau2: torch.Tensor,
    ssa2_sw: torch.Tensor,
    asy2_sw: torch.Tensor,
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    mask3: torch.Tensor,
    mask4: torch.Tensor,
    mask5: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge all-band tau and only the SW scattering surface used by H."""

    tauae = torch.zeros_like(tau1)
    tauae = torch.where(mask4, tau4, tauae)
    tauae = torch.where(mask3, tau3, tauae)
    tauae = torch.where(mask1, tau1, tauae)
    tauae = torch.where(mask2, tau2, tauae)

    tau1_sw = tau1[:, :NSWBND, :]
    ssaae_sw = torch.ones_like(tau1_sw)
    asyae_sw = torch.zeros_like(tau1_sw)
    ssaae_sw = torch.where(mask4 | mask5, sw_ssa.expand_as(ssaae_sw), ssaae_sw)
    asyae_sw = torch.where(mask4 | mask5, sw_asy.expand_as(asyae_sw), asyae_sw)
    ssaae_sw = torch.where(mask3, ssa3_sw, ssaae_sw)
    asyae_sw = torch.where(mask3, asy3_sw, asyae_sw)
    ssaae_sw = torch.where(mask1, ssa1_sw, ssaae_sw)
    asyae_sw = torch.where(mask1, asy1_sw, asyae_sw)
    ssaae_sw = torch.where(mask2, ssa2_sw, ssaae_sw)
    asyae_sw = torch.where(mask2, asy2_sw, asyae_sw)
    return tauae, ssaae_sw, asyae_sw


def _aerosol_profile_assignment_slab(
    pressure_ratio: torch.Tensor,
    height_first: torch.Tensor,
    height_second: torch.Tensor,
    dz: torch.Tensor,
    kp0: torch.Tensor,
    h1: torch.Tensor,
    skip_second: torch.Tensor,
    sigref: torch.Tensor,
    haer: torch.Tensor,
    ii_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance OPAC domain assignment through one fixed vertical slab."""

    ii = ii_in
    idmaer_values: list[torch.Tensor] = []
    delz_values: list[torch.Tensor] = []
    for level in range(pressure_ratio.shape[0]):
        threshold = sigref[ii, kp0]
        ii_next = ii + (pressure_ratio[level] < threshold).to(dtype=torch.int64)
        ii = torch.where((ii_next == 1) & skip_second, torch.full_like(ii_next, 2), ii_next)
        idmaer_values.append(ii + 1)
        tmp1 = torch.where(ii > 0, haer[ii, kp0], h1)
        positive_tmp1 = tmp1 > 0.0
        safe_tmp1 = torch.where(positive_tmp1, tmp1, torch.ones_like(tmp1))
        exp_delz = safe_tmp1 * (
            torch.exp(-height_first[level] / safe_tmp1)
            - torch.exp(-height_second[level] / safe_tmp1)
        )
        delz_values.append(torch.where(positive_tmp1, exp_delz, dz[level]))
    return torch.stack(idmaer_values), torch.stack(delz_values), ii


def _aerosol_profile_assignment_tensor(
    pressure_ratio: torch.Tensor,
    height_first: torch.Tensor,
    height_second: torch.Tensor,
    dz: torch.Tensor,
    kp0: torch.Tensor,
    h1: torch.Tensor,
    skip_second: torch.Tensor,
    sigref: torch.Tensor,
    haer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the five OPAC vertical regimes with four batched transitions."""

    nlay, ncol = pressure_ratio.shape
    levels = torch.arange(nlay, dtype=torch.int64, device=pressure_ratio.device).reshape(nlay, 1)
    state = torch.zeros((ncol,), dtype=torch.int64, device=pressure_ratio.device)
    previous_level = torch.full_like(state, -1)
    state_by_level = torch.zeros((nlay, ncol), dtype=torch.int64, device=pressure_ratio.device)
    no_level = torch.full((nlay, ncol), nlay, dtype=torch.int64, device=pressure_ratio.device)

    # The source recurrence can advance at most once per level and has five
    # states. Locate each of its at-most-four transition levels directly.
    for _transition in range(4):
        active = (state < 4) & (previous_level < nlay)
        threshold = sigref[state.clamp_max(4), kp0]
        crossing = (
            (levels > previous_level.unsqueeze(0))
            & (pressure_ratio < threshold.unsqueeze(0))
            & active.unsqueeze(0)
        )
        transition_level = torch.where(crossing, levels.expand(nlay, ncol), no_level).amin(dim=0)
        has_transition = transition_level < nlay
        advanced = state + 1
        advanced = torch.where((advanced == 1) & skip_second, torch.full_like(advanced, 2), advanced)
        next_state = torch.where(has_transition, advanced, state)
        state_by_level = torch.where(
            has_transition.unsqueeze(0) & (levels >= transition_level.unsqueeze(0)),
            next_state.unsqueeze(0),
            state_by_level,
        )
        state = next_state
        previous_level = torch.where(has_transition, transition_level, torch.full_like(previous_level, nlay))

    kp_by_level = kp0.unsqueeze(0).expand(nlay, ncol)
    scale_height = haer[state_by_level, kp_by_level]
    tmp1 = torch.where(state_by_level > 0, scale_height, h1.unsqueeze(0))
    positive_tmp1 = tmp1 > 0.0
    safe_tmp1 = torch.where(positive_tmp1, tmp1, torch.ones_like(tmp1))
    exp_delz = safe_tmp1 * (
        torch.exp(-height_first / safe_tmp1)
        - torch.exp(-height_second / safe_tmp1)
    )
    return state_by_level + 1, torch.where(positive_tmp1, exp_delz, dz)


def _aerosol_profile_assignment_slabbed_reference(
    p_lev_mb: torch.Tensor,
    hz: torch.Tensor,
    dz: torch.Tensor,
    kp0: torch.Tensor,
    h1: torch.Tensor,
    skip_second: torch.Tensor,
    sigref: torch.Tensor,
    haer: torch.Tensor,
    *,
    top_at_1: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the original fixed-slab OPAC assignment as a test reference."""

    nlay, ncol = dz.shape
    if top_at_1:
        rps = 1.0 / p_lev_mb[nlay, :]
        pressure_ratio = torch.flip(p_lev_mb[:nlay, :], dims=(0,)) * rps
        height_first = torch.flip(hz[1:, :], dims=(0,))
        height_second = torch.flip(hz[:nlay, :], dims=(0,))
        dz_ordered = torch.flip(dz, dims=(0,))
    else:
        rps = 1.0 / p_lev_mb[0, :]
        pressure_ratio = p_lev_mb[1:, :] * rps
        height_first = hz[:nlay, :]
        height_second = hz[1:, :]
        dz_ordered = dz

    ii = torch.zeros((ncol,), dtype=torch.int64, device=dz.device)
    idmaer_slabs: list[torch.Tensor] = []
    delz_slabs: list[torch.Tensor] = []
    slab_levels = _aerosol_slab_levels(dz, _AEROSOL_PROFILE_SLAB_LEVELS)
    for start in range(0, nlay, slab_levels):
        end = min(start + slab_levels, nlay)
        idmaer_slab, delz_slab, ii = run_compile_island(
            "aerosol",
            "profile_assignment_slab",
            _aerosol_profile_assignment_slab,
            pressure_ratio[start:end],
            height_first[start:end],
            height_second[start:end],
            dz_ordered[start:end],
            kp0,
            h1,
            skip_second,
            sigref,
            haer,
            ii,
        )
        idmaer_slabs.append(idmaer_slab)
        delz_slabs.append(delz_slab)
    idmaer = torch.cat(idmaer_slabs, dim=0)
    delz_result = torch.cat(delz_slabs, dim=0)
    if top_at_1:
        return torch.flip(idmaer, dims=(0,)), torch.flip(delz_result, dims=(0,))
    return idmaer, delz_result


def _aerosol_profile_assignment_compiled(
    p_lev_mb: torch.Tensor,
    hz: torch.Tensor,
    dz: torch.Tensor,
    kp0: torch.Tensor,
    h1: torch.Tensor,
    skip_second: torch.Tensor,
    sigref: torch.Tensor,
    haer: torch.Tensor,
    *,
    top_at_1: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the bounded-state OPAC assignment as one pure tensor island."""

    nlay = dz.shape[0]
    if top_at_1:
        rps = 1.0 / p_lev_mb[nlay, :]
        pressure_ratio = torch.flip(p_lev_mb[:nlay, :], dims=(0,)) * rps
        height_first = torch.flip(hz[1:, :], dims=(0,))
        height_second = torch.flip(hz[:nlay, :], dims=(0,))
        dz_ordered = torch.flip(dz, dims=(0,))
    else:
        rps = 1.0 / p_lev_mb[0, :]
        pressure_ratio = p_lev_mb[1:, :] * rps
        height_first = hz[:nlay, :]
        height_second = hz[1:, :]
        dz_ordered = dz

    idmaer, delz = run_compile_island(
        "aerosol",
        "profile_assignment",
        _aerosol_profile_assignment_tensor,
        pressure_ratio,
        height_first,
        height_second,
        dz_ordered,
        kp0,
        h1,
        skip_second,
        sigref,
        haer,
    )
    if top_at_1:
        return torch.flip(idmaer, dims=(0,)), torch.flip(delz, dims=(0,))
    return idmaer, delz


def _radclimaer_batch(
    tables: _AerosolTables,
    *,
    rh1: torch.Tensor,
    delz: torch.Tensor,
    idmaer: torch.Tensor,
    cmix: torch.Tensor,
    denn: torch.Tensor,
    top_at_1: bool,
    retain_diagnostics: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    dtype = rh1.dtype
    device = rh1.device
    nlay, ncol = rh1.shape
    rhlev = torch.as_tensor(RHLV, dtype=dtype, device=device)

    ih2_raw = torch.sum((rh1.unsqueeze(-1) > rhlev.reshape(1, 1, NRHLEV)).to(dtype=torch.int64), dim=-1)
    ih1 = torch.clamp(ih2_raw - 1, min=0, max=NRHLEV - 1)
    ih2 = torch.clamp(ih2_raw, min=0, max=NRHLEV - 1)
    rh1_ref = rhlev.index_select(0, ih1.reshape(-1)).reshape(nlay, ncol)
    rh2_ref = rhlev.index_select(0, ih2.reshape(-1)).reshape(nlay, ncol)
    rdrh = _safe_ratio(rh1 - rh1_ref, rh2_ref - rh1_ref)

    mask1 = (idmaer == 1).unsqueeze(1)
    mask2 = (idmaer == 2).unsqueeze(1)
    mask3 = (idmaer == 3).unsqueeze(1)
    mask4 = (idmaer == 4).unsqueeze(1)
    mask5 = (idmaer == 5).unsqueeze(1)
    delz_b = delz.unsqueeze(1)
    denn0 = denn[0, :].reshape(1, 1, ncol)
    denn1 = denn[1, :].reshape(1, 1, ncol)

    if retain_diagnostics:
        sw_lw_ssa = torch.empty((1, NSWLWBD, 1), dtype=dtype, device=device)
        sw_lw_ssa[:, :NSWBND, :] = 0.99
        sw_lw_ssa[:, NSWBND:, :] = 0.5
        sw_lw_asy = torch.empty_like(sw_lw_ssa)
        sw_lw_asy[:, :NSWBND, :] = 0.696
        sw_lw_asy[:, NSWBND:, :] = 0.3
    else:
        sw_lw_ssa = torch.full((1, NSWBND, 1), 0.99, dtype=dtype, device=device)
        sw_lw_asy = torch.full((1, NSWBND, 1), 0.696, dtype=dtype, device=device)

    # Background stratospheric branch.
    extstra = tables.extstra.to(dtype=dtype).reshape(1, NSWLWBD, 1)
    tau4 = extstra * delz_b

    # Maritime branch.
    ext2 = tables.extrhi[5, :].to(dtype=dtype).reshape(1, NSWLWBD, 1)
    tau2 = ext2 * denn1 * delz_b
    scattering_slice = slice(None) if retain_diagnostics else slice(0, NSWBND)
    scattering_bands = NSWLWBD if retain_diagnostics else NSWBND
    ssa2 = tables.ssarhi[5, scattering_slice].to(dtype=dtype).reshape(1, scattering_bands, 1).expand(nlay, scattering_bands, ncol)
    asy2 = tables.asyrhi[5, scattering_slice].to(dtype=dtype).reshape(1, scattering_bands, 1).expand(nlay, scattering_bands, ncol)

    # Free-tropospheric mixture branch.
    ex03 = _gather_rh_band(tables.extrhd[:, 0, :], ih1, ih2, rdrh)
    sc03 = _gather_rh_band(tables.scarhd[:, 0, scattering_slice], ih1, ih2, rdrh)
    ss03 = _gather_rh_band(tables.ssarhd[:, 0, scattering_slice], ih1, ih2, rdrh)
    as03 = _gather_rh_band(tables.asyrhd[:, 0, scattering_slice], ih1, ih2, rdrh)
    ex01 = tables.extrhi[0, :].to(dtype=dtype).reshape(1, NSWLWBD, 1)
    sc01 = tables.scarhi[0, scattering_slice].to(dtype=dtype).reshape(1, scattering_bands, 1)
    ss01 = tables.ssarhi[0, scattering_slice].to(dtype=dtype).reshape(1, scattering_bands, 1)
    as01 = tables.asyrhi[0, scattering_slice].to(dtype=dtype).reshape(1, scattering_bands, 1)
    ex02 = tables.extrhi[1, :].to(dtype=dtype).reshape(1, NSWLWBD, 1)
    sc02 = tables.scarhi[1, scattering_slice].to(dtype=dtype).reshape(1, scattering_bands, 1)
    ss02 = tables.ssarhi[1, scattering_slice].to(dtype=dtype).reshape(1, scattering_bands, 1)
    as02 = tables.asyrhi[1, scattering_slice].to(dtype=dtype).reshape(1, scattering_bands, 1)
    ext3 = 0.17e-3 * ex01 + 0.4 * ex02 + 0.59983 * ex03
    ex01_scattering = ex01[:, scattering_slice, :]
    ex02_scattering = ex02[:, scattering_slice, :]
    ex03_scattering = ex03[:, scattering_slice, :]
    sca3 = 0.17e-3 * sc01 + 0.4 * sc02 + 0.59983 * sc03
    ssa_num3 = 0.17e-3 * ss01 * ex01_scattering + 0.4 * ss02 * ex02_scattering + 0.59983 * ss03 * ex03_scattering
    asy_num3 = 0.17e-3 * as01 * sc01 + 0.4 * as02 * sc02 + 0.59983 * as03 * sc03
    tau3 = ext3 * 730.0 * delz_b
    ext3_scattering = ext3[:, scattering_slice, :]
    ssa3 = torch.minimum(torch.ones_like(ext3_scattering), _safe_ratio(ssa_num3, ext3_scattering))
    asy3 = torch.minimum(torch.ones_like(sca3), _safe_ratio(asy_num3, sca3))

    # Boundary-layer mixture branch.
    nv_aod = int(tables.nv_aod)
    if retain_diagnostics:
        boundary_layer_args = (
            cmix, denn0, delz, ih1, ih2, rdrh, mask1,
            tables.extrhi, tables.scarhi, tables.ssarhi, tables.asyrhi,
            tables.extrhd, tables.scarhd, tables.ssarhd, tables.asyrhd,
            nv_aod,
        )
        if "aerosol" in compile_categories_enabled():
            tau1, ssa1, asy1, spcodp1 = run_compile_island(
                "aerosol",
                "boundary_layer_optics",
                _aerosol_boundary_layer_optics_kernel,
                *boundary_layer_args,
            )
        else:
            tau1, ssa1, asy1, spcodp1 = _aerosol_boundary_layer_optics_kernel(*boundary_layer_args)
    else:
        boundary_layer_args_minimal = (
            cmix, denn0, delz, ih1, ih2, rdrh,
            tables.extrhi, tables.scarhi, tables.ssarhi, tables.asyrhi,
            tables.extrhd, tables.scarhd, tables.ssarhd, tables.asyrhd,
        )
        if "aerosol" in compile_categories_enabled():
            tau1, ssa1, asy1 = run_compile_island(
                "aerosol",
                "boundary_layer_optics_minimal",
                _aerosol_boundary_layer_optics_minimal_kernel,
                *boundary_layer_args_minimal,
            )
        else:
            tau1, ssa1, asy1 = _aerosol_boundary_layer_optics_minimal_kernel(*boundary_layer_args_minimal)
        spcodp1 = None

    if retain_diagnostics:
        regime_args = (
            tau4, sw_lw_ssa, sw_lw_asy, tau3, ssa3, asy3,
            tau1, ssa1, asy1, tau2, ssa2, asy2,
            mask1, mask2, mask3, mask4, mask5,
            spcodp1, ex01, ex02, ex03, delz, nv_aod, int(IDXSPC[9]),
        )
        if "aerosol" in compile_categories_enabled():
            tauae, ssaae, asyae, spcodp = run_compile_island(
                "aerosol",
                "regime_merge",
                _aerosol_regime_merge_kernel,
                *regime_args,
            )
        else:
            tauae, ssaae, asyae, spcodp = _aerosol_regime_merge_kernel(*regime_args)
    else:
        regime_args_minimal = (
            tau4, sw_lw_ssa, sw_lw_asy, tau3, ssa3, asy3,
            tau1, ssa1, asy1, tau2, ssa2, asy2,
            mask1, mask2, mask3, mask4, mask5,
        )
        if "aerosol" in compile_categories_enabled():
            tauae, ssaae, asyae = run_compile_island(
                "aerosol",
                "regime_merge_minimal",
                _aerosol_regime_merge_minimal_kernel,
                *regime_args_minimal,
            )
        else:
            tauae, ssaae, asyae = _aerosol_regime_merge_minimal_kernel(*regime_args_minimal)
        spcodp = None

    if "aerosol" in compile_categories_enabled():
        tauae = _aerosol_optical_depth_redistribution_compiled(tauae, top_at_1=top_at_1)
    else:
        one = torch.ones((NSWLWBD, ncol), dtype=dtype, device=device)
        if top_at_1:
            for kk in range(1, nlay):
                current = tauae[kk, :, :]
                neighbor = tauae[kk - 1, :, :]
                ratio = torch.where(current > 0.0, neighbor / torch.where(current > 0.0, current, one), one)
                total = current + neighbor
                high = ratio > 30.0
                low = ratio < 0.03333
                tauae[kk, :, :] = torch.where(high, 0.2 * total, torch.where(low, 0.8 * total, current))
                tauae[kk - 1, :, :] = torch.where(high, 0.8 * total, torch.where(low, 0.2 * total, neighbor))
        else:
            for kk in range(nlay - 2, -1, -1):
                current = tauae[kk, :, :]
                neighbor = tauae[kk + 1, :, :]
                ratio = torch.where(current > 0.0, neighbor / torch.where(current > 0.0, current, one), one)
                total = current + neighbor
                high = ratio > 30.0
                low = ratio < 0.03333
                tauae[kk, :, :] = torch.where(high, 0.2 * total, torch.where(low, 0.8 * total, current))
                tauae[kk + 1, :, :] = torch.where(high, 0.8 * total, torch.where(low, 0.2 * total, neighbor))
    return tauae, ssaae, asyae, spcodp


def _add_volcanic_batch_kernel(
    aersw2_tau: torch.Tensor,
    aerlw_tau: torch.Tensor,
    p_lev_mb: torch.Tensor,
    lat_rad: torch.Tensor,
    ivolae: torch.Tensor,
    sw_scale: torch.Tensor,
    lw_scale: torch.Tensor,
    con_pi: float,
    month_index: int,
    year_index: int,
    top_at_1: bool,
    do_sw: bool,
    do_lw: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = p_lev_mb.dtype
    device = p_lev_mb.device
    nlev, ncol = p_lev_mb.shape
    nlay = nlev - 1
    alat = lat_rad.reshape(-1).to(device=device, dtype=dtype) * (180.0 / con_pi)
    abs_lat = torch.abs(alat)
    ivol = ivolae[month_index, :, year_index].to(device=device, dtype=dtype)
    volcae = torch.where(
        alat > 46.0,
        1.0e-4 * ivol[0],
        torch.where(
            alat > 44.0,
            5.0e-5 * (ivol[0] + ivol[1]),
            torch.where(
                alat > 1.0,
                1.0e-4 * ivol[1],
                torch.where(
                    alat > -1.0,
                    5.0e-5 * (ivol[1] + ivol[2]),
                    torch.where(
                        alat > -44.0,
                        1.0e-4 * ivol[2],
                        torch.where(alat > -46.0, 5.0e-5 * (ivol[2] + ivol[3]), 1.0e-4 * ivol[3]),
                    ),
                ),
            ),
        ),
    )
    psrfl = torch.where(abs_lat > 70.0, torch.full_like(abs_lat, 250.0), torch.where(abs_lat < 20.0, torch.full_like(abs_lat, 150.0), 110.0 + 2.0 * abs_lat))

    if top_at_1:
        candidates_h = torch.arange(1, max(nlay - 2, 1), dtype=torch.int64, device=device)
        cond_h = p_lev_mb.index_select(0, candidates_h) >= 5.0
        has_h = cond_h.any(dim=0)
        first_h = candidates_h.index_select(0, torch.argmax(cond_h.to(dtype=torch.int64), dim=0))
        kcuth = torch.where(has_h, first_h - 1, torch.full((ncol,), nlay - 1, dtype=torch.int64, device=device))

        candidates_l = candidates_h
        cond_l = p_lev_mb.index_select(0, candidates_l) >= psrfl.reshape(1, ncol)
        has_l = cond_l.any(dim=0)
        first_l = candidates_l.index_select(0, torch.argmax(cond_l.to(dtype=torch.int64), dim=0))
        kcutl = torch.where(has_l, first_l - 1, torch.ones((ncol,), dtype=torch.int64, device=device))
        p_first_l = p_lev_mb.gather(0, first_l.reshape(1, ncol)).reshape(ncol)
        p_kcuth = p_lev_mb.gather(0, torch.clamp(kcuth, 0, nlay).reshape(1, ncol)).reshape(ncol)
        rdelp = torch.where(has_l, 1.0 / (p_first_l - p_kcuth), 1.0 / p_lev_mb[1, :])
        pd = (p_lev_mb[1:, :] - p_lev_mb[:-1, :]) * rdelp.reshape(1, ncol)
        level_index = torch.arange(nlay, dtype=torch.int64, device=device).reshape(nlay, 1)
        layer_mask = (level_index >= kcuth.reshape(1, ncol)) & (level_index <= kcutl.reshape(1, ncol))
    else:
        candidates_h = torch.arange(nlay - 2, 0, -1, dtype=torch.int64, device=device)
        cond_h = p_lev_mb.index_select(0, candidates_h) >= 5.0
        has_h = cond_h.any(dim=0)
        first_h = candidates_h.index_select(0, torch.argmax(cond_h.to(dtype=torch.int64), dim=0))
        kcuth = torch.where(has_h, first_h, torch.ones((ncol,), dtype=torch.int64, device=device))

        candidates_l = torch.arange(nlay - 1, 0, -1, dtype=torch.int64, device=device)
        cond_l = p_lev_mb.index_select(0, candidates_l) >= psrfl.reshape(1, ncol)
        has_l = cond_l.any(dim=0)
        first_l = candidates_l.index_select(0, torch.argmax(cond_l.to(dtype=torch.int64), dim=0))
        kcutl = torch.where(has_l, first_l, torch.full((ncol,), nlay - 1, dtype=torch.int64, device=device))
        p_kcutl = p_lev_mb.gather(0, torch.clamp(kcutl, 0, nlay).reshape(1, ncol)).reshape(ncol)
        p_kcuthp1 = p_lev_mb.gather(0, torch.clamp(kcuth + 1, 0, nlay).reshape(1, ncol)).reshape(ncol)
        rdelp = torch.where(has_l, 1.0 / (p_kcutl - p_kcuthp1), 1.0 / p_lev_mb[nlay - 2, :])
        pd = (p_lev_mb[:-1, :] - p_lev_mb[1:, :]) * rdelp.reshape(1, ncol)
        level_index = torch.arange(nlay, dtype=torch.int64, device=device).reshape(nlay, 1)
        layer_mask = (level_index >= kcutl.reshape(1, ncol)) & (level_index <= kcuth.reshape(1, ncol))

    pressure_delta = torch.where(layer_mask, pd, torch.zeros_like(pd))
    if do_sw:
        aersw2_tau = aersw2_tau + sw_scale.reshape(NBDSW, 1, 1) * pressure_delta.unsqueeze(0) * volcae.reshape(1, 1, ncol)
        kl = kcutl
        current = aersw2_tau.gather(1, torch.clamp(kl, 0, nlay - 1).reshape(1, 1, ncol).expand(NBDSW, 1, ncol)).squeeze(1)
        if top_at_1:
            neighbor_index = torch.clamp(kl + 1, 0, nlay - 1)
            valid = kl + 1 < nlay
        else:
            neighbor_index = torch.clamp(kl - 1, 0, nlay - 1)
            valid = kl - 1 >= 0
        neighbor = aersw2_tau.gather(1, neighbor_index.reshape(1, 1, ncol).expand(NBDSW, 1, ncol)).squeeze(1)
        cond = (current > 10.0 * neighbor) & valid.reshape(1, ncol)
        total = current + neighbor
        aersw2_tau = aersw2_tau.scatter(
            1,
            torch.clamp(kl, 0, nlay - 1).reshape(1, 1, ncol).expand(NBDSW, 1, ncol),
            torch.where(cond, 0.8 * total, current).unsqueeze(1),
        )
        aersw2_tau = aersw2_tau.scatter(
            1,
            neighbor_index.reshape(1, 1, ncol).expand(NBDSW, 1, ncol),
            torch.where(cond, 0.2 * total, neighbor).unsqueeze(1),
        )
    if do_lw:
        aerlw_tau = aerlw_tau + lw_scale.reshape(NBDLW, 1, 1) * pressure_delta.unsqueeze(0) * volcae.reshape(1, 1, ncol)
    return aersw2_tau, aerlw_tau


_VOLCANIC_SCALE_CACHE: dict[tuple[str, int | None, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


@torch.inference_mode(False)
def _volcanic_scales(*, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device.type, device.index, dtype)
    cached = _VOLCANIC_SCALE_CACHE.get(key)
    if cached is not None:
        return cached
    wv1 = torch.as_tensor(WVNSW1, dtype=dtype, device=device)
    wv2 = torch.as_tensor(WVNSW2, dtype=dtype, device=device)
    exponent = torch.where(
        wv1 > 20000.0,
        torch.full_like(wv1, 0.74),
        torch.where(wv2 < 20000.0, torch.full_like(wv1, 1.14), torch.full_like(wv1, 0.94)),
    )
    sw_scale = torch.pow(0.275e-4 * (wv2 + wv1), exponent)
    wv1_lw = torch.as_tensor(WVNLW1, dtype=dtype, device=device)
    wv2_lw = torch.as_tensor(WVNLW2, dtype=dtype, device=device)
    lw_scale = torch.pow(0.275e-4 * (wv2_lw + wv1_lw), torch.full_like(wv1_lw, 1.2))
    _VOLCANIC_SCALE_CACHE[key] = (sw_scale, lw_scale)
    return sw_scale, lw_scale


def _add_volcanic_batch(
    tables: _AerosolTables,
    *,
    aersw2_tau: torch.Tensor,
    aerlw_tau: torch.Tensor,
    p_lev_mb: torch.Tensor,
    lat_rad: torch.Tensor,
    con_pi: float,
    top_at_1: bool,
    do_sw: bool,
    do_lw: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    sw_scale, lw_scale = _volcanic_scales(dtype=p_lev_mb.dtype, device=p_lev_mb.device)
    args = (
        aersw2_tau,
        aerlw_tau,
        p_lev_mb,
        lat_rad,
        tables.ivolae,
        sw_scale,
        lw_scale,
        con_pi,
        tables.month - 1,
        tables.year % 10,
        top_at_1,
        do_sw,
        do_lw,
    )
    if "aerosol" in compile_categories_enabled():
        return run_compile_island("aerosol", "volcanic_update", _add_volcanic_batch_kernel, *args)
    return _add_volcanic_batch_kernel(*args)


def _aerosol_cumulative_height_tensor(dz: torch.Tensor, *, top_at_1: bool) -> torch.Tensor:
    """Integrate layer thickness in source order without a parallel scan."""

    nlay, ncol = dz.shape
    height = torch.zeros((ncol,), dtype=dz.dtype, device=dz.device)
    height_rows = [height]
    layer_order = range(nlay - 1, -1, -1) if top_at_1 else range(nlay)
    for ilay in layer_order:
        height = height + dz[ilay, :]
        height_rows.append(height)
    return torch.stack(list(reversed(height_rows)) if top_at_1 else height_rows, dim=0)


def _aerosol_cumulative_height(dz: torch.Tensor, *, top_at_1: bool) -> torch.Tensor:
    if "aerosol" in compile_categories_enabled():
        return run_compile_island(
            "aerosol",
            "cumulative_height",
            _aerosol_cumulative_height_tensor,
            dz,
            top_at_1=top_at_1,
        )
    return _aerosol_cumulative_height_tensor(dz, top_at_1=top_at_1)


def _compute_opac_batch(
    tables: _AerosolTables,
    *,
    p_lev_mb: torch.Tensor,
    p_lay_mb: torch.Tensor,
    tv_lay: torch.Tensor,
    relhum: torch.Tensor,
    lsmask: torch.Tensor,
    lon_rad: torch.Tensor,
    lat_rad: torch.Tensor,
    con_pi: float,
    con_rd: float,
    con_g: float,
    top_at_1: bool,
    do_sw: bool,
    do_lw: bool,
    retain_diagnostics: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    dtype = p_lev_mb.dtype
    device = p_lev_mb.device
    nlev, ncol = p_lev_mb.shape
    nlay = nlev - 1
    grid_args = (
        tables.kprfg,
        tables.haer,
        tables.denng,
        tables.idxcg,
        tables.cmixg,
        lon_rad,
        lat_rad,
        lsmask,
        con_pi,
    )
    if "aerosol" in compile_categories_enabled():
        kp0, h1, denn, cmix = run_compile_island(
            "aerosol",
            "grid_mixture",
            _aerosol_grid_mixture_kernel,
            *grid_args,
        )
    else:
        kp0, h1, denn, cmix = _aerosol_grid_mixture_kernel(*grid_args)

    rovg = 0.001 * con_rd / con_g
    if not top_at_1:
        prsln = torch.empty((nlay + 1, ncol), dtype=dtype, device=device)
        prsln[:nlay, :] = torch.log(p_lev_mb[:nlay, :])
        prsln[nlay, :] = torch.log(p_lay_mb[nlay - 1, :])
        dz = rovg * (prsln[:-1, :] - prsln[1:, :]) * tv_lay
        dz[nlay - 1, :] = 2.0 * dz[nlay - 1, :]
        hz = _aerosol_cumulative_height(dz, top_at_1=top_at_1)
    else:
        prsln = torch.empty((nlay + 1, ncol), dtype=dtype, device=device)
        prsln[0, :] = torch.log(p_lay_mb[0, :])
        prsln[1:, :] = torch.log(p_lev_mb[1:, :])
        dz = rovg * (prsln[1:, :] - prsln[:-1, :]) * tv_lay
        dz[0, :] = 2.0 * dz[0, :]
        hz = _aerosol_cumulative_height(dz, top_at_1=top_at_1)

    skip_second = tables.prsref[1, kp0] == tables.prsref[2, kp0]
    if "aerosol" in compile_categories_enabled():
        idmaer, delz = _aerosol_profile_assignment_compiled(
            p_lev_mb,
            hz,
            dz,
            kp0,
            h1,
            skip_second,
            tables.sigref,
            tables.haer,
            top_at_1=top_at_1,
        )
    else:
        delz = torch.empty((nlay, ncol), dtype=dtype, device=device)
        idmaer = torch.empty((nlay, ncol), dtype=torch.int64, device=device)
        ii = torch.zeros((ncol,), dtype=torch.int64, device=device)
        if not top_at_1:
            rps = 1.0 / p_lev_mb[0, :]
            for k in range(nlay):
                threshold = tables.sigref[ii, kp0]
                ii_next = ii + (p_lev_mb[k + 1, :] * rps < threshold).to(dtype=torch.int64)
                ii = torch.where((ii_next == 1) & skip_second, torch.full_like(ii_next, 2), ii_next)
                idmaer[k, :] = ii + 1
                tmp1 = torch.where(ii > 0, tables.haer[ii, kp0], h1)
                positive_tmp1 = tmp1 > 0.0
                safe_tmp1 = torch.where(positive_tmp1, tmp1, torch.ones_like(tmp1))
                exp_delz = safe_tmp1 * (
                    torch.exp(-hz[k, :] / safe_tmp1) - torch.exp(-hz[k + 1, :] / safe_tmp1)
                )
                delz[k, :] = torch.where(
                    positive_tmp1,
                    exp_delz,
                    dz[k, :],
                )
        else:
            rps = 1.0 / p_lev_mb[nlay, :]
            for k in range(nlay - 1, -1, -1):
                threshold = tables.sigref[ii, kp0]
                ii_next = ii + (p_lev_mb[k, :] * rps < threshold).to(dtype=torch.int64)
                ii = torch.where((ii_next == 1) & skip_second, torch.full_like(ii_next, 2), ii_next)
                idmaer[k, :] = ii + 1
                tmp1 = torch.where(ii > 0, tables.haer[ii, kp0], h1)
                positive_tmp1 = tmp1 > 0.0
                safe_tmp1 = torch.where(positive_tmp1, tmp1, torch.ones_like(tmp1))
                exp_delz = safe_tmp1 * (
                    torch.exp(-hz[k + 1, :] / safe_tmp1) - torch.exp(-hz[k, :] / safe_tmp1)
                )
                delz[k, :] = torch.where(
                    positive_tmp1,
                    exp_delz,
                    dz[k, :],
                )

    tauae, ssaae, asyae, spcodp = _radclimaer_batch(
        tables,
        rh1=relhum,
        delz=delz,
        idmaer=idmaer,
        cmix=cmix,
        denn=denn,
        top_at_1=top_at_1,
        retain_diagnostics=retain_diagnostics,
    )
    aersw2_tau = tauae[:, :NBDSW, :].permute(1, 0, 2).contiguous() if do_sw else torch.zeros((NBDSW, nlay, ncol), dtype=dtype, device=device)
    aersw2_ssa = ssaae[:, :NBDSW, :].permute(1, 0, 2).contiguous() if do_sw else torch.zeros((NBDSW, nlay, ncol), dtype=dtype, device=device)
    aersw2_g = asyae[:, :NBDSW, :].permute(1, 0, 2).contiguous() if do_sw else torch.zeros((NBDSW, nlay, ncol), dtype=dtype, device=device)
    aerlw_tau = tauae[:, NBDSW:NBDSW + NBDLW, :].permute(1, 0, 2).contiguous() if do_lw else torch.zeros((NBDLW, nlay, ncol), dtype=dtype, device=device)
    aerlw_ssa = (
        ssaae[:, NBDSW:NBDSW + NBDLW, :].permute(1, 0, 2).contiguous()
        if retain_diagnostics and do_lw
        else torch.zeros((NBDLW, nlay, ncol), dtype=dtype, device=device) if retain_diagnostics else None
    )
    aerlw_g = (
        asyae[:, NBDSW:NBDSW + NBDLW, :].permute(1, 0, 2).contiguous()
        if retain_diagnostics and do_lw
        else torch.zeros((NBDLW, nlay, ncol), dtype=dtype, device=device) if retain_diagnostics else None
    )
    aerodp = torch.zeros((NSPC1, ncol), dtype=dtype, device=device) if retain_diagnostics else None
    if retain_diagnostics and do_sw:
        assert aerodp is not None and spcodp is not None
        aerodp[0, :] = torch.sum(tauae[:, int(tables.nv_aod), :], dim=0)
        aerodp[1:, :] = spcodp
    aersw2_tau, aerlw_tau = _add_volcanic_batch(
        tables,
        aersw2_tau=aersw2_tau,
        aerlw_tau=aerlw_tau,
        p_lev_mb=p_lev_mb,
        lat_rad=lat_rad,
        con_pi=con_pi,
        top_at_1=top_at_1,
        do_sw=do_sw,
        do_lw=do_lw,
    )
    return aersw2_tau, aersw2_ssa, aersw2_g, aerlw_tau, aerlw_ssa, aerlw_g, aerodp






def run_aerosol_from_boundaries(
    boundaries: Mapping[str, Mapping[str, TensorOrString]],
    *,
    month: int = 6,
    year: int = 1992,
    retain_diagnostics: bool = True,
) -> RrtmgpAerosolResult:
    """Compute target-owned aerosol G from sealed source-hook inputs."""

    hook = _hook(boundaries)
    do_sw = _to_int(hook["doSWrad"], name="doSWrad") != 0
    do_lw = _to_int(hook["doLWrad"], name="doLWrad") != 0
    top_at_1 = _to_int(hook["top_at_1"], name="top_at_1") != 0
    iaermdl = _to_int(hook["iaermdl"], name="iaermdl")
    iaerflg = _to_int(hook["iaerflg"], name="iaerflg")
    n_day = _to_int(hook["nDay"], name="nDay")
    p_lev = _to_real(hook["p_lev"], name="p_lev")
    p_lay = _to_real(hook["p_lay"], name="p_lay")
    device = p_lev.device
    dtype = p_lev.dtype
    nlev, ncol = p_lev.shape
    nlay = nlev - 1
    if iaermdl == 0 or iaerflg == 0:
        aersw_tau = _zeros_broadcast_last_dim((NBDSW, nlay, ncol), dtype=dtype, device=device)
        aersw_ssa = _zeros_broadcast_last_dim((NBDSW, nlay, ncol), dtype=dtype, device=device)
        aersw_g = _zeros_broadcast_last_dim((NBDSW, nlay, ncol), dtype=dtype, device=device)
        aerlw_tau = _zeros_broadcast_last_dim((NBDLW, nlay, ncol), dtype=dtype, device=device)
        if not retain_diagnostics:
            return RrtmgpAerosolResult(
                fields={
                    "aersw_tau": aersw_tau,
                    "aersw_ssa": aersw_ssa,
                    "aersw_g": aersw_g,
                    "aerlw_tau": aerlw_tau,
                }
            )
        aerlw_ssa = _zeros_broadcast_last_dim((NBDLW, nlay, ncol), dtype=dtype, device=device)
        aerlw_g = _zeros_broadcast_last_dim((NBDLW, nlay, ncol), dtype=dtype, device=device)
        return RrtmgpAerosolResult(
            fields={
                "aerodp": _zeros_broadcast_last_dim((NSPC1, ncol), dtype=dtype, device=device),
                "aersw_tau": aersw_tau,
                "aersw_ssa": aersw_ssa,
                "aersw_g": aersw_g,
                "aerlw_tau": aerlw_tau,
                "aerlw_ssa": aerlw_ssa,
                "aerlw_g": aerlw_g,
                "ext550": _zeros_broadcast_last_dim((nlay, ncol), dtype=dtype, device=device),
                "errmsg": "",
                "errflg": torch.zeros((1,), dtype=torch.int64, device=device),
            }
        )
    if iaermdl != 5:
        raise NotImplementedError(f"implemented aerosol target supports OPAC iaermdl=5 only, got {iaermdl}")
    if iaerflg != 111:
        raise NotImplementedError(f"implemented aerosol target supports active iaerflg=111 only, got {iaerflg}")

    tv_lay = _to_real(hook["tv_lay"], name="tv_lay")
    relhum = _to_real(hook["relhum"], name="relhum")
    lsmask = _to_real(hook["lsmask"], name="lsmask")
    lon = _to_real(hook["lon"], name="lon")
    lat = _to_real(hook["lat"], name="lat")
    con_pi = _to_host_float(hook["con_pi"], name="con_pi")
    con_rd = _to_host_float(hook["con_rd"], name="con_rd")
    con_g = _to_host_float(hook["con_g"], name="con_g")
    tables = _tables_for_device(device, month=month, year=year, con_pi=con_pi)

    # One-column and batched execution intentionally share this tensor kernel.
    # The former scalar-only branch made the numerical implementation depend on
    # total batch width and prevented a meaningful unique-profile invariant.
    if do_sw or do_lw:
        aersw2_tau, aersw2_ssa, aersw2_g, aerlw_tau, aerlw_ssa, aerlw_g, aerodp = _compute_opac_batch(
            tables,
            p_lev_mb=p_lev * 0.01,
            p_lay_mb=p_lay * 0.01,
            tv_lay=tv_lay,
            relhum=relhum,
            lsmask=lsmask,
            lon_rad=lon,
            lat_rad=lat,
            con_pi=con_pi,
            con_rd=con_rd,
            con_g=con_g,
            top_at_1=top_at_1,
            do_sw=do_sw,
            do_lw=do_lw,
            retain_diagnostics=retain_diagnostics,
        )
    else:
        aersw2_tau = torch.zeros((NBDSW, nlay, ncol), dtype=dtype, device=device)
        aersw2_ssa = torch.zeros_like(aersw2_tau)
        aersw2_g = torch.zeros_like(aersw2_tau)
        aerlw_tau = torch.zeros((NBDLW, nlay, ncol), dtype=dtype, device=device)
        aerlw_ssa = torch.zeros_like(aerlw_tau)
        aerlw_g = torch.zeros_like(aerlw_tau)
        aerodp = torch.zeros((NSPC1, ncol), dtype=dtype, device=device)
    if do_sw and n_day > 0:
        aersw_tau = torch.cat((aersw2_tau[-1:, :, :], aersw2_tau[:-1, :, :]), dim=0)
        aersw_ssa = torch.cat((aersw2_ssa[-1:, :, :], aersw2_ssa[:-1, :, :]), dim=0)
        aersw_g = torch.cat((aersw2_g[-1:, :, :], aersw2_g[:-1, :, :]), dim=0)
    else:
        aersw_tau = torch.zeros_like(aersw2_tau)
        aersw_ssa = torch.zeros_like(aersw2_ssa)
        aersw_g = torch.zeros_like(aersw2_g)

    if not retain_diagnostics:
        return RrtmgpAerosolResult(
            fields={
                "aersw_tau": aersw_tau,
                "aersw_ssa": aersw_ssa,
                "aersw_g": aersw_g,
                "aerlw_tau": aerlw_tau,
            }
        )

    assert aerlw_ssa is not None and aerlw_g is not None and aerodp is not None
    ext550 = torch.zeros((nlay, ncol), dtype=dtype, device=device)

    return RrtmgpAerosolResult(
        fields={
            "aerodp": aerodp,
            "aersw_tau": aersw_tau,
            "aersw_ssa": aersw_ssa,
            "aersw_g": aersw_g,
            "aerlw_tau": aerlw_tau,
            "aerlw_ssa": aerlw_ssa,
            "aerlw_g": aerlw_g,
            "ext550": ext550,
            "errmsg": "",
            "errflg": torch.zeros((1,), dtype=torch.int64, device=device),
        }
    )


__all__ = ["RrtmgpAerosolResult", "run_aerosol_from_boundaries"]
