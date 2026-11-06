# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Offline TEMPO coefficient construction from the retained NumPy oracle.

Only immutable constants and prepared coefficient buffers cross into the runtime.
No live model state enters this module.
"""
from __future__ import annotations

import cmath
from types import SimpleNamespace

import numpy as np

from tempo_torch import parameters as _parameters
from tempo_torch.parameters import CHECKPOINT_FIELDS as CHECKPOINT_FIELDS


def _complex_beta_constant_python(m1: complex, m2: complex) -> complex:
    return 2.0 * m1 / (m2 - m1) * (m2 / (m2 - m1) * cmath.log(m2 / m1) - 1.0)


_STATIC_SNOW_A = np.asarray(_parameters.SA, dtype=np.float64)
_STATIC_SNOW_B = np.asarray(_parameters.SB, dtype=np.float64)
_STATIC_SNOW_CSE = np.asarray(_parameters.CSE, dtype=np.float64)
_STATIC_SNOW_CSG = np.asarray(_parameters.CSG, dtype=np.float64)
# Frozen TEMPO REAL64 initialization for the fall-speed numerator. Host
# math.gamma differs at these two arguments; preserve the NumPy oracle below.
_STATIC_SNOW_CSG = _STATIC_SNOW_CSG.copy()
_STATIC_SNOW_CSG[3] = 3.51325214469320013
_STATIC_SNOW_CSG[9] = 7.61279965268895031
_SCALAR_CIE = tuple(float(value) for value in _parameters.CIE)
_SCALAR_CIG = tuple(float(value) for value in _parameters.CIG)
_SCALAR_CRE = tuple(float(value) for value in _parameters.CRE)
_SCALAR_CRG = tuple(float(value) for value in _parameters.CRG)
_SCALAR_CSE = tuple(float(value) for value in _parameters.CSE)
_SCALAR_CSG = tuple(float(value) for value in _parameters.CSG)
_SNOW_MOMENT_COMPILE_FIELDS = _parameters.SNOW_MOMENT_FIELDS
_STATIC_CLOUD_CCE = np.asarray(_parameters.CCE, dtype=np.float64)
_STATIC_CLOUD_CCG = np.asarray(_parameters.CCG, dtype=np.float64)
_STATIC_CLOUD_OCG1 = np.asarray(_parameters.OCG1, dtype=np.float64)
_STATIC_CLOUD_OCG2 = np.asarray(_parameters.OCG2, dtype=np.float64)
_STATIC_GRAUPEL_CGG = np.asarray(_parameters.CGG, dtype=np.float64).copy()
# Frozen REAL64 initialize_graupel_vars values; host math.gamma differs.
_FORTRAN_GRAUPEL_GAMMA = {
    1.6409616470000001: 0.8987805104039659,
    1.8900000000000001: 0.9583793077186229,
    2.8204808235: 1.7057543783678373,
    2.945: 1.9021706839288695,
    3.39: 2.9495407427433666,
    3.89: 5.23476361668989,
    4.640961647: 14.181764706343934,
    7.89: 4041.011183991114,
}
for _argument, _gamma in _FORTRAN_GRAUPEL_GAMMA.items():
    _STATIC_GRAUPEL_CGG[_parameters.CGE == _argument] = _gamma
_STATIC_GRAUPEL_CGE = np.asarray(_parameters.CGE, dtype=np.float64)
_STATIC_GRAUPEL_RHO_G = np.asarray(_parameters.RHO_G, dtype=np.float64)
_STATIC_GRAUPEL_AV_G = np.asarray(_parameters.AV_G, dtype=np.float64)
_STATIC_GRAUPEL_BV_G = np.asarray(_parameters.BV_G, dtype=np.float64)
_STATIC_GRAUPEL_AM_G = np.asarray(_parameters.AM_G, dtype=np.float64)
_SCALAR_GRAUPEL_CGE = tuple(tuple(float(value) for value in row) for row in _parameters.CGE)
_SCALAR_GRAUPEL_CGG = tuple(tuple(float(value) for value in row) for row in _STATIC_GRAUPEL_CGG)
_SCALAR_GRAUPEL_RHO_G = tuple(float(value) for value in _parameters.RHO_G)
_SCALAR_GRAUPEL_AV_G = tuple(float(value) for value in _parameters.AV_G)
_SCALAR_GRAUPEL_BV_G = tuple(float(value) for value in _parameters.BV_G)
_SCALAR_GRAUPEL_AM_G = tuple(float(value) for value in _parameters.AM_G)
_SCALAR_GRAUPEL_SINGLE_MOMENT_LAMG_SCALE = float(
    (_SCALAR_GRAUPEL_CGG[2][0] * _parameters.OGG2 * _parameters.OGG1) ** _parameters.OBMG
)
_STATIC_RAIN_DIAMETER_BINS = np.asarray(_parameters._rain_diameter_bins(), dtype=np.float64)
_STATIC_WARM_RAIN_EFFICIENCY = np.asarray(_parameters._warm_rain_efficiency(), dtype=np.float64)
_SCALAR_RAIN_WATER_THRESHOLDS = tuple(float(value) for value in _parameters.R_R)
_SCALAR_SNOW_WATER_THRESHOLDS = tuple(float(value) for value in _parameters.R_S)
_SCALAR_GRAUPEL_WATER_THRESHOLDS = tuple(float(value) for value in _parameters.R_G)
_STATIC_CLOUD_NUMBER_BINS = np.asarray(_parameters._cloud_number_bins(), dtype=np.float64)
_SCALAR_CLOUD_WATER_THRESHOLDS = tuple(float(value) for value in _parameters.R_C)
_STATIC_DROP_EVAP_NUMBER_TABLE = np.asarray(_parameters._drop_evap_number_table(), dtype=np.float64)
_STATIC_CLOUD_ACTIVATION_NA_AXIS = np.asarray(_parameters.TA_NA, dtype=np.float64)
_STATIC_CLOUD_ACTIVATION_W_AXIS = np.asarray(_parameters.TA_WW, dtype=np.float64)
_STATIC_CLOUD_ACTIVATION_T_AXIS = np.asarray(_parameters.TA_TK, dtype=np.float64)
_STATIC_D0I = float(_parameters._d0i())
_STATIC_ICE_AUTOCONVERSION_TABLES = tuple(
    np.asarray(array, dtype=np.float64)
    for array in _parameters._ice_autoconversion_tables()
)
_STATIC_SNOW_CLOUD_WATER_EFFICIENCY = np.asarray(_parameters._snow_cloud_water_efficiency(), dtype=np.float64)
_STATIC_SNOW_DIAMETER_BINS = np.asarray(_parameters._snow_diameter_bins(), dtype=np.float64)
_STATIC_HAIL_BINS = tuple(
    np.asarray(array, dtype=np.float64)
    for array in _parameters._hail_bins()
)
_STATIC_RADAR_BINS = {
    (_parameters.RADAR_BINS, 100.0e-6, 0.05): tuple(
        np.asarray(array, dtype=np.float64)
        for array in _parameters._create_bins_with_widths(_parameters.RADAR_BINS, 100.0e-6, 0.05)
    ),
    (_parameters.RADAR_BINS, 100.0e-6, 0.02): tuple(
        np.asarray(array, dtype=np.float64)
        for array in _parameters._create_bins_with_widths(_parameters.RADAR_BINS, 100.0e-6, 0.02)
    ),
}
_STATIC_SIMPSON_WEIGHTS = {
    _parameters.RADAR_BINS: np.asarray(_parameters._simpson_weights(_parameters.RADAR_BINS), dtype=np.float64),
}

_STATIC_NDARRAY_IDS = frozenset(
    id(array)
    for array in (
        _STATIC_SNOW_A,
        _STATIC_SNOW_B,
        _STATIC_SNOW_CSE,
        _STATIC_SNOW_CSG,
        _STATIC_CLOUD_CCE,
        _STATIC_CLOUD_CCG,
        _STATIC_CLOUD_OCG1,
        _STATIC_CLOUD_OCG2,
        _STATIC_GRAUPEL_CGG,
        _STATIC_GRAUPEL_CGE,
        _STATIC_GRAUPEL_RHO_G,
        _STATIC_GRAUPEL_AV_G,
        _STATIC_GRAUPEL_BV_G,
        _STATIC_GRAUPEL_AM_G,
        _STATIC_RAIN_DIAMETER_BINS,
        _STATIC_WARM_RAIN_EFFICIENCY,
        _STATIC_CLOUD_NUMBER_BINS,
        _STATIC_DROP_EVAP_NUMBER_TABLE,
        _STATIC_CLOUD_ACTIVATION_NA_AXIS,
        _STATIC_CLOUD_ACTIVATION_W_AXIS,
        _STATIC_CLOUD_ACTIVATION_T_AXIS,
        _STATIC_SNOW_CLOUD_WATER_EFFICIENCY,
        _STATIC_SNOW_DIAMETER_BINS,
        *_STATIC_ICE_AUTOCONVERSION_TABLES,
        *_STATIC_HAIL_BINS,
        *(array for pair in _STATIC_RADAR_BINS.values() for array in pair),
        *_STATIC_SIMPSON_WEIGHTS.values(),
    )
)


def _radar_scalar_bin_weights(numbins: int, lowbin: float, highbin: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    key = (int(numbins), float(lowbin), float(highbin))
    bins, widths = _STATIC_RADAR_BINS[key]
    simpson = _STATIC_SIMPSON_WEIGHTS[int(numbins)]
    return (
        tuple(float(value) for value in bins),
        tuple(float(weight) for weight in np.asarray(simpson, dtype=np.float64) * np.asarray(widths, dtype=np.float64)),
    )


_SCALAR_RADAR_GRAUPEL_BINS, _SCALAR_RADAR_GRAUPEL_QUADRATURE_WEIGHTS = _radar_scalar_bin_weights(
    _parameters.RADAR_BINS,
    100.0e-6,
    0.05,
)
_SCALAR_RADAR_SNOW_BINS, _SCALAR_RADAR_SNOW_QUADRATURE_WEIGHTS = _radar_scalar_bin_weights(
    _parameters.RADAR_BINS,
    100.0e-6,
    0.02,
)
_RADAR_COMPLEX_WATER_0 = complex(_parameters.complex_water_ray(_parameters.LAMBDA_RADAR, 0.0))
_RADAR_COMPLEX_ICE_0 = complex(_parameters.complex_ice_maetzler(_parameters.LAMBDA_RADAR, 0.0))
_HOMOGENEOUS_FREEZING_TEMPERATURE = float(_parameters.segment_i_hgfrz())
_RADAR_COMPLEX_WATER_SQ = _RADAR_COMPLEX_WATER_0 * _RADAR_COMPLEX_WATER_0
_RADAR_COMPLEX_ICE_SQ = _RADAR_COMPLEX_ICE_0 * _RADAR_COMPLEX_ICE_0
_RADAR_BETA_WATER_ICE = _complex_beta_constant_python(_RADAR_COMPLEX_WATER_SQ, _RADAR_COMPLEX_ICE_SQ)
_RADAR_K_W = abs((_RADAR_COMPLEX_WATER_SQ - 1.0) / (_RADAR_COMPLEX_WATER_SQ + 2.0))
_RADAR_K_W = _RADAR_K_W * _RADAR_K_W
_RADAR_REFLECTIVITY_FACTOR = float(
    _parameters.LAMBDA_RADAR**4 / (_parameters.PI**5 * _RADAR_K_W)
)



def _literal(value):
    if isinstance(value, np.ndarray):
        return _literal(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_literal(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


# The runtime reads plain Python values: no NumPy scalars enter the tensor programs.
constants = SimpleNamespace(**{name: _literal(value) for name, value in vars(_parameters).items() if name.isupper()})
