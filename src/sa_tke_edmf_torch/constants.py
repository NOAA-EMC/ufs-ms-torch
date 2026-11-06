# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Constants for the approved profile-based SA-TKE-EDMF gate path."""

import struct


def source_real(value: float, config: dict) -> float:
    """Bind default-real literals to the oracle build, without tensor host reads."""
    kind = config.get("source_default_real_bytes", 4)
    if kind == 8:
        return value
    if kind == 4:
        return struct.unpack("f", struct.pack("f", value))[0]
    raise ValueError("source_default_real_bytes must be 4 or 8")

IM = 4
KM = 16
NTRAC = 8
NVDIFF = NTRAC
NDTEND = 16

GRAV = 9.80665
RD = 287.05
CP = 1004.6
RV = 461.5
HVAP = 2.5e6
HFUS = 3.3358e5
PSAT = 6.1078e2
TTP = 2.7316e2
CVAP = 1.8460e3
CLIQ = 4.1855e3
CSOL = 2.1060e3
FV = RV / RD - 1.0
EPS = RD / RV
EPSM1 = EPS - 1.0
ELOCP = HVAP / CP
EL2ORC = HVAP * HVAP / (RV * CP)
QMIN = 9.99999993922529e-09
QLMIN = 9.999999960041972e-13

F32_0P1 = 0.10000000149011612
F32_0P01 = 0.009999999776482582
F32_0P001 = 0.0010000000474974513
F32_1E_M4 = 9.999999747378752e-05
F32_0P15 = 0.15000000596046448
F32_0P16 = 0.1599999964237213
F32_0P19 = 0.1899999976158142
F32_0P2 = 0.20000000298023224
F32_0P3 = 0.30000001192092896
F32_0P4 = 0.4000000059604645
F32_0P49 = 0.49000000953674316
F32_0P51 = 0.5099999904632568
F32_0P55 = 0.550000011920929
F32_0P76 = 0.7599999904632568
F32_0P999 = 0.9990000128746033
F32_2P7 = 2.700000047683716
F32_3P7 = 3.700000047683716
F32_3P14 = 3.140000104904175
F32_ONE_THIRD = 0.3333333432674408
F32_1E_M6 = 9.999999974752427e-07
F32_1E_M10 = 1.000000013351432e-10

NTQV = 1
NTCW = 2
NTIW = 3
NTRW = 4
NTSW = 5
NTOZ = 6
NTCHM = 1
NTCHS = 8
NTCHE = 8
NTCHEM = NTCHS - 1
NTKE = 8
NTKEV = NTKE

SEGMENT_IDS = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
)

IMPLEMENTED_SEGMENTS = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
)
