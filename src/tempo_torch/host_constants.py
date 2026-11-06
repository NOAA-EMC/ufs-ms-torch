# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Physical-constant sets of the TEMPO port.

TEMPO keeps ``pi, t0, rho_w, lsub, lvap0, rv, rdry, cp, r_uni`` as module
variables (``module_mp_tempo_params.F90:150-159``). Standalone ``tempo_init``
uses those defaults throughout. The CCPP wrapper ``mp_tempo_init`` first calls
``tempo_init`` (which builds the collision-efficiency, drop-evaporation and
ice-autoconversion tables and the ice diameter bins with the defaults), then
overwrites the variables with the host constants and calls
``initialize_parameters()`` again, so every runtime parameter (``am_r``,
``roverrv``, ``olfus``, ...) uses the host values while those init-time tables
keep the defaults (``mp_tempo.F90:111-128``).

The profile is selected once, at import, by ``TEMPO_TORCH_HOST_CONSTANTS``
because the compiled kernels capture constants as Python literals:

* ``tempo_default`` (default): standalone TEMPO; every value is unchanged.
* ``ufs_physcons``: the CCPP host constants of ``physcons.F90`` exactly as
  ``mp_tempo_init`` assigns them.

``ACTIVE`` is the runtime set; ``INIT`` is the set the init-time tables use.
``HAIL_AWARE_INIT`` (``TEMPO_TORCH_HAIL_AWARE_INIT``, default 1) is the
``hailaware_flag`` given to ``tempo_init``, which selects the graupel fall-speed
coefficients at ``idx_bg1``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

HOST_CONSTANTS_ENV = "TEMPO_TORCH_HOST_CONSTANTS"
TEMPO_DEFAULT = "tempo_default"
UFS_PHYSCONS = "ufs_physcons"


@dataclass(frozen=True)
class TempoConstants:
    pi: float
    t0: float
    rho_w: float
    lsub: float
    lvap0: float
    rv: float
    rdry: float
    cp: float
    r_uni: float


def _tempo_default() -> TempoConstants:
    return TempoConstants(
        pi=float(3.1415926536),
        t0=float(273.15),
        rho_w=float(1000.0),
        lsub=float(2.834e6),
        lvap0=float(2.5e6),
        rv=float(461.5),
        rdry=float(287.04),
        cp=float(1004.0),
        r_uni=float(8.314),
    )


def _ufs_physcons() -> TempoConstants:
    # physcons.F90: con_pi = 4*atan(1), con_hvap, con_hfus, con_rv, con_rd, con_cp, con_t0c, con_rgas, rhowater.
    con_pi = float(4.0) * math.atan(float(1.0))
    lvap0 = float(2.5e6)
    lfus = float(3.3358e5)
    return TempoConstants(
        pi=con_pi,
        t0=float(273.15),
        rho_w=float(1000.0),
        lsub=lvap0 + lfus,  # mp_tempo.F90:119
        lvap0=lvap0,
        rv=float(461.50),
        rdry=float(287.05),
        cp=float(1004.6),
        r_uni=float(8.314472),
    )


PROFILES = {TEMPO_DEFAULT: _tempo_default(), UFS_PHYSCONS: _ufs_physcons()}

ACTIVE_PROFILE = os.environ.get(HOST_CONSTANTS_ENV, TEMPO_DEFAULT) or TEMPO_DEFAULT
if ACTIVE_PROFILE not in PROFILES:
    raise ValueError(
        f"{HOST_CONSTANTS_ENV}={ACTIVE_PROFILE!r} is not one of {sorted(PROFILES)}"
    )

ACTIVE = PROFILES[ACTIVE_PROFILE]
INIT = PROFILES[TEMPO_DEFAULT]

# tempo_init(hailaware_flag=...) -> initialize_graupel_vars (module_mp_tempo_params.F90:411-426):
# hail-aware keeps av_g/bv_g(idx_bg1) at the new coefficients; otherwise the "old"
# coefficients replace them before initialize_parameters builds cge/cgg. The standalone
# reference driver initializes hail-aware; the oracle GFS v17 suite passes lthailaware=F.
HAIL_AWARE_INIT_ENV = "TEMPO_TORCH_HAIL_AWARE_INIT"
_hail_value = os.environ.get(HAIL_AWARE_INIT_ENV, "1") or "1"
if _hail_value not in ("0", "1"):
    raise ValueError(f"{HAIL_AWARE_INIT_ENV}={_hail_value!r} must be 0 or 1")
HAIL_AWARE_INIT = _hail_value == "1"
