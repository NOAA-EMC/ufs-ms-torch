# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Thermodynamic helpers shared by the physics schemes."""

from __future__ import annotations

TRIPLE_POINT = 273.16


def phase_latent_heat(l0, temperature, cp, rd, rv, ttp=TRIPLE_POINT):
    """Vapour-liquid latent heat at ``temperature`` in MPAS moist thermodynamics.

    ``l0 - rv*ttp + (cp - rd)*(rv/rd)*(temperature - ttp)``: the internal-energy latent heat at the triple point
    ``ttp`` (``l0`` is the scheme's reference latent heat), plus the vapour heat capacity at constant volume that
    one kappa for moist air implies, ``(cp - rd)*rv/rd``, times ``temperature - ttp``. Each caller keeps its own
    operand kinds (Python floats or 0-d tensors), so its operations and their rounding do not change.
    """
    return l0 - rv * ttp + (cp - rd) * (rv / rd) * (temperature - ttp)
