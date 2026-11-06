# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Field provenance classification for anti-injection enforcement.

Every field in the dycore state is classified as either:
- ALLOWED_FROM_FILE: May originate from init.nc / grid.nc / mesh_complete.nc
- MUST_COMPUTE: Must be computed from RAW + MESH_STATIC inputs. Loading these
  derived fields from an input file is injection.
"""

from __future__ import annotations

from typing import Any

from mpas_pytorch.exceptions import MpasConfigError, MpasProvenanceError

# Fields allowed to come from init.nc / grid.nc / mesh_complete.nc
ALLOWED_FROM_FILE: frozenset[str] = frozenset(
    {
        # RAW prognostic (from init.nc)
        "theta",
        "rho",
        "u",
        "w",
        "qv",
        "scalars",
        "rho_base",
        "theta_base",
        "zz",
        "zgrid",
        "fzm",
        "fzp",
        "fnm",
        "fnp",
        "rdzw",
        "rdzu",
        "rdnw",
        "dzu",
        "dzw",
        "cf1",
        "cf2",
        "cf3",
        "u_init",
        "v_init",
        "w_init",
        "qv_init",
        "t_init",
        # RAW mesh (from grid.nc)
        "cellsOnCell",
        "edgesOnCell",
        "verticesOnCell",
        "cellsOnEdge",
        "verticesOnEdge",
        "edgesOnVertex",
        "cellsOnVertex",
        "edgesOnEdge",
        "nEdgesOnCell",
        "nEdgesOnEdge",
        "dvEdge",
        "dcEdge",
        "areaCell",
        "areaTriangle",
        "fEdge",
        "fVertex",
        "weightsOnEdge",
        "kiteAreasOnVertex",
        "latCell",
        "lonCell",
        "latEdge",
        "lonEdge",
        "latVertex",
        "lonVertex",
        "xCell",
        "yCell",
        "zCell",
        "xEdge",
        "yEdge",
        "zEdge",
        "xVertex",
        "yVertex",
        "zVertex",
        "angleEdge",
        "meshDensity",
        "zb",
        "zb3",
        "zxu",
        "coeffs_reconstruct",
        "defc_a",
        "defc_b",
        "localVerticalUnitVectors",
        "edgeNormalVectors",
        "cellTangentPlane",
        # MESH_STATIC (deterministic functions of grid geometry)
        "invAreaCell",
        "invDvEdge",
        "invDcEdge",
        "invAreaTriangle",
        "edgesOnCell_sign",
        "edgesOnVertex_sign",
        "kiteForCell",
        "zb_cell",
        "zb3_cell",
        "nAdvCellsForEdge",
        "advCellsForEdge",
        "adv_coefs",
        "adv_coefs_3rd",
        "deriv_two",
        "meshScalingDel2",
        "meshScalingDel4",
        "meshScalingRegionalCell",
        "meshScalingRegionalEdge",
        "dss",
        # Dimension / config scalars
        "nCells",
        "nEdges",
        "nVertices",
        "nVertLevels",
        "maxEdges",
        "maxEdges2",
        "vertexDegree",
        "moist_start",
        "moist_end",
        "coef_3rd_order",
        "c_s",
        "num_scalars",
        "bdyMaskCell",
        "bdyMaskEdge",
        "on_a_sphere",
        # Index arrays (integer, from grid.nc)
        "indexToCellID",
        "indexToEdgeID",
        "indexToVertexID",
        "bdyMaskVertex",
        "landmask",
        "ivgtyp",
        "isltyp",
        # Physics static inputs from init.nc
        "smois",
        "tslb",
        "skintemp",
        "snow",
        "snowc",
        "snowh",
        "snoalb",
        "tmn",
        "vegfra",
        "xland",
        "xice",
        "greenfrac",
        "var2d",
        "con",
        "oa1",
        "oa2",
        "oa3",
        "oa4",
        "ol1",
        "ol2",
        "ol3",
        "ol4",
    }
)

# Diagnostic fields derived by the runtime from the primary model state.
MUST_COMPUTE: frozenset[str] = frozenset(
    {
        # Init coupled diagnostics
        "theta_m",
        "rho_zz",
        "ru",
        "rw",
        "rho_p",
        "rtheta_base",
        "rtheta_p",
        "exner",
        "exner_base",
        "pressure_p",
        "pressure_base",
        # Init solve diagnostics
        "v",
        "vorticity",
        "divergence",
        "ke",
        "pv_edge",
        "rho_edge",
        "h_edge",
        # Init vector reconstruction
        "ur_cell",
        "vr_cell",
        "uReconstructX",
        "uReconstructY",
        "uReconstructZ",
        "uReconstructZonal",
        "uReconstructMeridional",
        # Dycore-only zeroed field
        "rt_diabatic_tend",
        # Runtime aliases and moisture coefficients
        "cqw",
        "cqu",
        "h",
        "p",
        "pb",
        "pp",
        "pressure_b",
        "t",
        "rb",
        "rr",
        "rr_save",
        "rt",
        "rtb",
        "rho_p_save",
        "rho_zz_driving",
        "rho_zz_init",
        "rho_zz_new",
        "rho_zz_old",
        "rho_zz_old_split",
        "rtheta_m_driving",
        "rtheta_p_save",
        "ruAvg_split",
        "ru_driving",
        "ru_save",
        "rw_driving",
        "rw_save",
        "scalar_new",
        "scalar_old",
        "scalars_new",
        "scalars_old",
        "theta_m_init",
        "theta_m_save",
        "u_rk_save",
        "u_tend",
        "uhAvg",
        "w_tend",
        "wwAvg_split",
        # Physics-init derived static inputs
        "sh2o",
        "o3clim",
        "o3vmr",
        "ozmixm",
        "pin",
        "sfc_albedo",
        "sfc_emibck",
        "sfc_emiss",
        "mavail",
        "m_ps",
        "plrad",
        "cldfrac",
        # Tendencies (computed during integration)
        "tend_u",
        "tend_w",
        "tend_theta_m",
        "tend_rho_zz",
        "rho_pp",
        "rtheta_pp",
        "ru_p",
        "rw_p",
        "ruAvg",
        "wwAvg",
        # Packed vertical implicit coefficients (built each step, never loaded)
        "_vertical_implicit_coefficients_cofwr_active",
        "_vertical_implicit_coefficients_cofwz_active",
        "_vertical_implicit_coefficients_coftz_active",
        "_vertical_implicit_coefficients_cofwt_active",
        "_vertical_implicit_coefficients_a_tri_active",
        "_vertical_implicit_coefficients_alpha_tri_active",
        "_vertical_implicit_coefficients_gamma_tri_active",
    }
)


def validate_no_injection(
    state: dict[str, Any],
    *,
    raise_on_violation: bool = True,
    physical_constants_profile: str = "ncar",
) -> list[str]:
    """Assert no MUST_COMPUTE field was loaded into state before the init chain.

    Call this AFTER loading raw IC/mesh but BEFORE running init_coupled_diagnostics
    and solve_diagnostics. Any MUST_COMPUTE field already present means it was
    injected from an input file.

    Returns list of violating field names.
    """
    if physical_constants_profile not in ("ncar", "ufs"):
        raise MpasConfigError("unknown physical constants profile for input ownership")
    # NCAR derives sh2o; UFS reads it through the native sfc_input stream.
    raw_surface = {"sh2o"} if physical_constants_profile == "ufs" else set()
    violations = sorted(k for k in state if k in MUST_COMPUTE and k not in raw_surface)
    if violations and raise_on_violation:
        raise MpasProvenanceError(
            f"DERIVED field injection detected: {violations} "
            f"found in state before init chain. These must be "
            f"computed by PyTorch, not loaded from checkpoints."
        )
    return violations
