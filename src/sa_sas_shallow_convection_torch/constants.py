# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Constants for the sealed SA-SAS shallow-convection profile gate path."""

IM = 4
KM = 16
NN = 10
NTR = 8
NTC = 3
ITC = 6
NTK = 8
NCLOUD = 2
NDTEND = 32

CLIQ = 4185.5
CP = 1004.6
CVAP = 1846.0
RD = 287.05
RV = 461.5
EPS = RD / RV
EPSM1 = EPS - 1.0
FV = RV / RD - 1.0
GRAV = 9.80665
HVAP = 2.5e6
HFUS = 3.3358e5
PSAT = 6.1078e2
T0C = 273.15
TTP = 273.16
CSOL = 2106.0
DELT = 300.0

CLAM = 0.1
C0S = 0.002
C1 = 0.002
EVEF = 0.2
PGCON = 0.55
ASOLFAC = 1.0
CSCALE = 1.0
BETADCU = 1.0
BETAMCU = 1.0
BETASCU = 1.0
CAT_ADJ_SHAL = 1.0

SEGMENT_IDS = tuple("ABCDEFGHIJKLMNOPQ")
IMPLEMENTED_SEGMENTS = SEGMENT_IDS
