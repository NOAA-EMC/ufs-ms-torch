# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Spectral constants of ``radiation_aerosols.f`` (``set_spectrum``: the solar and infrared weights of the aerosol
bands), used by the MERRA-2 aerosol optics (``merra2``)."""

from __future__ import annotations

import math
import re

import numpy as np

NWVSOL = 151
NWVTOT = 57600
NWVTIR = 4000


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

# physcons.F90 values.
CON_T0C = 2.7315e2
CON_C = 2.99792458e8
CON_PLNK = 6.6260693e-34
CON_BOLTZ = 1.3806505e-23


# ``set_spectrum`` data of ``radiation_aerosols.f``: the number of 1-cm^-1 intervals in each of the NWVSOL solar
# bands (``nwvns0``) and each band's solar flux per interval (``s0intv``, W/m^2).
NWVNS0 = (
    100, 11, 14, 18, 24, 33, 50, 83, 12, 12, 13, 15, 15, 17, 18, 20,
    21, 24, 26, 30, 32, 37, 42, 47, 55, 64, 76, 91, 111, 139, 179, 238,
    333, 41, 42, 45, 46, 48, 51, 53, 55, 58, 61, 64, 68, 71, 75, 79,
    84, 89, 95, 101, 107, 115, 123, 133, 142, 154, 167, 181, 197, 217, 238, 263,
    293, 326, 368, 417, 476, 549, 641, 758, 909, 101, 103, 105, 108, 109, 112, 115,
    117, 119, 122, 125, 128, 130, 134, 137, 140, 143, 147, 151, 154, 158, 163, 166,
    171, 175, 181, 185, 190, 196, 201, 207, 213, 219, 227, 233, 240, 248, 256, 264,
    274, 282, 292, 303, 313, 325, 337, 349, 363, 377, 392, 408, 425, 444, 462, 483,
    505, 529, 554, 580, 610, 641, 675, 711, 751, 793, 841, 891, 947, 1008, 1075, 1150,
    1231, 1323, 1425, 1538, 1667, 1633, 14300,
)

S0INTV = (
    1.6e-06, 2.88e-05, 3.6e-05, 4.592e-05, 6.132e-05, 8.55e-05, 0.0001286, 0.000216,
    0.00029058, 0.000310184, 0.000334152, 0.000358722, 0.00038805, 0.00042, 0.000457056, 0.000496892,
    0.00054516, 0.0006006, 0.0006536, 0.00072504, 0.00079866, 0.0009112, 0.0010368, 0.0011844,
    0.00136682, 0.0015756, 0.0018744, 0.002255, 0.002745, 0.0033984, 0.00434, 0.005754,
    0.00774, 0.0095305, 0.00990192, 0.0102874, 0.0106803, 0.0111366, 0.011583, 0.0121088,
    0.012642, 0.013225, 0.0138088, 0.0144612, 0.0151164, 0.0158878, 0.01665, 0.017514,
    0.018445, 0.0194106, 0.0204864, 0.0217248, 0.023064, 0.024447, 0.025984, 0.027594,
    0.0294138, 0.031395, 0.03348, 0.0357696, 0.0384054, 0.041349, 0.044688, 0.048222,
    0.0522918, 0.0570078, 0.0619888, 0.065472, 0.066906, 0.0681226, 0.0697788, 0.0712668,
    0.07271, 0.073161, 0.0733471, 0.0734814, 0.0734717, 0.0735072, 0.0734939, 0.0735202,
    0.0733249, 0.0731713, 0.0735462, 0.073692, 0.0723677, 0.0725023, 0.0724258, 0.0720766,
    0.0718284, 0.0732757, 0.0731645, 0.0733277, 0.0736128, 0.0733752, 0.0728965, 0.0724924,
    0.0723307, 0.072105, 0.071262, 0.0710903, 0.0712714, 0.0708012, 0.0703752, 0.070035,
    0.0698639, 0.069069, 0.0687621, 0.065208, 0.0665184, 0.0660038, 0.0647615, 0.0644831,
    0.0637206, 0.0624102, 0.0618698, 0.060632, 0.0583498, 0.0567028, 0.0551232, 0.0548645,
    0.051234, 0.0485581, 0.048501, 0.047922, 0.0444058, 0.0448718, 0.0429373, 0.0415242,
    0.0381744, 0.0316342, 0.0299615, 0.029274, 0.0267484, 0.0176904, 0.0140049, 0.0146224,
    0.0139993, 0.0119574, 0.0106386, 0.010098, 0.00863808, 0.00652736, 0.0049941, 0.0043935,
    0.00221676, 0.00133812, 0.0011232, 0.000559, 0.00036, 0.00029808, 7.46294e-05,
)


def _numbers(text: str) -> list[float]:
    return [float(item) for item in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?|[-+]?\.\d+(?:[Ee][+-]?\d+)?", text)]


def _read_float_block(lines: list[str], index: int, count: int) -> tuple[np.ndarray, int]:
    values: list[float] = []
    while len(values) < count:
        values.extend(_numbers(lines[index]))
        index += 1
    if len(values) != count:
        raise ValueError(f"expected {count} values, got {len(values)}")
    return np.asarray(values, dtype=np.float64), index


def _set_spectrum(con_pi: float) -> tuple[np.ndarray, np.ndarray]:
    nwvns0, s0intv = NWVNS0, S0INTV
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

