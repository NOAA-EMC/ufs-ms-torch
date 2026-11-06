# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""PyTorch-native host operators for the selected UFS-MPAS/CCPP model."""

import os

# The coupled model runs TEMPO with the CCPP host constants and without the hail-aware
# initialization. TEMPO reads both settings once, when it is first imported.
os.environ.setdefault("TEMPO_TORCH_HOST_CONSTANTS", "ufs_physcons")
os.environ.setdefault("TEMPO_TORCH_HAIL_AWARE_INIT", "0")

from .clock import StepClock, advance_utc_date, prepare_physics_calendar, prepare_timestep_clock
from .commit import (
    CommitDomain,
    CommitGeometry,
    NativeCommitResult,
    NativeCommitState,
    commit_mpas_state,
    commit_native,
    finalize_stateout,
)
from .containers import NativeAtmosphere, ProjectionMesh, Statein, Stateout
from .convection import ConvectionResult, apply_convection, convection_core, convection_post
from .convective_clouds import (
    ConvectiveCloudHistory, cnvc90, convective_cloud_clock, interstitial_4,
    prepare_convective_clouds,
)
from .cycle import (
    CoupledCycleState, CoupledCycleResult, ColdCycleInitialization,
    allocate_cold_cycle_state, initialize_cold_cycle, mark_cold_boundary,
    run_coupled_cycle, run_coupled_interval,
    NativeCoupledPhase, NativeCoupledContext, prepare_native_coupled_context,
    start_native_coupled_phase, advance_native_coupled_phase, finish_native_coupled_phase,
)
from .dynamics import (
    PhysicsDynamicsResult, MicrophysicsCommitResult,
    advance_physics_dynamics, commit_microphysics,
)
from .grid import Grid, GridProjection, project_grid
from .gwd import apply_gwd_pre, orographic_gwd_inputs, gwdps
from .gwd_small_scale import drag_suite_psl, small_scale_gwd_inputs
from .gwd_nonstationary import latitude_wave_stress, nonstationary_gwd, wave_spectrum
from .gwd_unified import apply_unified_gwd, unified_gwd
from .gwd_post import apply_gwd_post, generic_gwd_post, unified_gwd_post
from .gwd_v1 import apply_ugwpv1_gsldrag, ngw_solv2, prepare_ugwpv1, ugwpv1_gsldrag, ugwpv1_gsldrag_post
from .initialization import (
    BridgeInitialization,
    ReferenceGeometry,
    SurfaceSchemeControls,
    PhysicsSchemeControls,
    GWDSchemeControls,
    ConvectionSchemeControls,
    MicrophysicsSchemeControls,
    initialize_bridge,
    initialize_reference_geometry,
    prepare_r4_surface_controls,
    prepare_r4_physics_controls,
    DiagnosticIndices,
    prepare_r4_diagnostic_indices,
    prepare_r4_gwd_controls,
    prepare_r4_radiation_config,
    prepare_r4_convection_controls,
    prepare_r4_microphysics_controls,
)
from .land import (
    apply_noahmp,
    apply_sfc_land,
    finish_surface_iteration,
    noahmp_surface_inputs,
)
from .land_initialization import (
    LandInitializationTables,
    initialize_noahmp_cold_surface,
)
from .ledger import CommitLedgers, CumulativeLimiter, commit_ledgers
from .reservoirs import physical_reservoirs
from .momentum import (
    CellWindIncrement,
    MomentumCommit,
    MomentumMesh,
    commit_momentum,
    commit_surface_pressure,
    prepare_cell_wind_increment,
)
from .microphysics_prefix import get_phi, interstitial_3, prepare_microphysics_prefix
from .microphysics import (
    MicrophysicsSequenceResult, run_microphysics_sequence,
    apply_tempo, apply_microphysics_post, microphysics_post,
)
from .physics_sequence import PhysicsSequenceResult, run_physics_sequence
from .physics_post import (
    finish_microphysics, hourly_reset, maximum_hourly_diagnostics,
    physics_post, stochastics_off,
)
from .nsst import NSSTControl, apply_nsst, sfc_nst_run
from .ozone import (
    PrognosticOzone,
    h2o_photochemistry,
    initialize_ozone_grid,
    ozone_forecast_day,
    ozone_2015,
    photochemistry_latitude_weights,
    update_h2o_forcing,
    update_ozone_forcing,
)
from .photochemistry import apply_photochemistry, photochemistry_2015, photochemistry_post
from .physics_prefix import (
    PhysicsPrefixControl,
    PhysicsPrefixResult,
    allocate_physics_prefix_scratch,
    apply_surface_layer,
    prepare_physics_prefix,
)
from .pbl import apply_pbl_pre, run_pbl_sequence
from .projection import (
    AtmosphericProjection,
    HydrostaticPressure,
    TracerBinding,
    TracerContract,
    hydrostatic_pressure,
    project_atmosphere,
    r4_tracer_contract,
)
from .radiation import (
    NativeRadiationResult,
    RadiationCarry,
    radiation_group_inputs,
    reset_radiation_diagnostics,
    reset_physics_diagnostics,
    run_native_radiation,
)
from .radiation_scaling import (
    RadiationScalingControl,
    RadiationScalingResult,
    apply_held_radiation,
    scale_radiation,
)
from .radiation_seeds import (
    RadiationSeeds,
    advance_mpas_radiation_seeds,
    radiation_permutation_seed,
    radiation_seed_offset,
    random_index,
)
from .refresh import (
    DiagnosticRefresh,
    OutputDiagnostics,
    ReconstructionMesh,
    refresh_host_diagnostics,
)
from .reverse import CommitPhase, ThermodynamicCommit, commit_thermodynamics
from .sea_ice import apply_sea_ice, sea_ice_inputs, sfc_sice_run
from .surface import SurfaceInitialization, initialize_surface
from .surface_composites import (
    SurfaceCompositesControl,
    SurfaceCompositesResult,
    prepare_surface_composites,
)
from .surface_containers import NativeSurface, Sfcprop, SoilParameters, SurfaceControl
from .surface_mirror import SurfaceMirror, SurfaceMirrorInputs, mirror_surface
from .surface_post import (
    SurfacePostControl,
    apply_surface_composites_post,
    apply_surface_diagnostics,
    apply_surface_diagnostic_post,
    apply_surface_generic_post,
    run_surface_sequence,
)

__all__ = [
    "PhysicsSchemeControls",
    "GWDSchemeControls",
    "prepare_r4_physics_controls",
    "DiagnosticIndices",
    "prepare_r4_diagnostic_indices",
    "prepare_r4_gwd_controls",
    "SurfaceSchemeControls",
    "prepare_r4_surface_controls",
    "physical_reservoirs",
    "CoupledCycleState",
    "CoupledCycleResult",
    "allocate_cold_cycle_state",
    "ColdCycleInitialization",
    "initialize_cold_cycle",
    "run_coupled_cycle",
    "run_coupled_interval",
    "NativeCoupledPhase",
    "NativeCoupledContext",
    "prepare_native_coupled_context",
    "start_native_coupled_phase",
    "advance_native_coupled_phase",
    "finish_native_coupled_phase",
    "MicrophysicsSequenceResult",
    "run_microphysics_sequence",
    "PhysicsSequenceResult",
    "run_physics_sequence",
    "run_pbl_sequence",
    "apply_tempo",
    "finish_microphysics",
    "hourly_reset",
    "maximum_hourly_diagnostics",
    "physics_post",
    "stochastics_off",
    "ConvectiveCloudHistory",
    "cnvc90",
    "convective_cloud_clock",
    "interstitial_4",
    "prepare_convective_clouds",
    "ConvectionResult",
    "apply_convection",
    "convection_core",
    "convection_post",
    "apply_gwd_pre",
    "orographic_gwd_inputs",
    "gwdps",
    "latitude_wave_stress",
    "nonstationary_gwd",
    "wave_spectrum",
    "apply_unified_gwd",
    "unified_gwd",
    "apply_gwd_post",
    "generic_gwd_post",
    "unified_gwd_post",
    "apply_ugwpv1_gsldrag",
    "ngw_solv2",
    "prepare_ugwpv1",
    "ugwpv1_gsldrag",
    "ugwpv1_gsldrag_post",
    "finalize_stateout",
    "PhysicsDynamicsResult",
    "MicrophysicsCommitResult",
    "commit_microphysics",
    "advance_physics_dynamics",
    "drag_suite_psl",
    "small_scale_gwd_inputs",
    "apply_pbl_pre",
    "get_phi",
    "interstitial_3",
    "prepare_microphysics_prefix",
    "SurfacePostControl",
    "apply_surface_composites_post",
    "apply_surface_diagnostics",
    "apply_surface_diagnostic_post",
    "apply_surface_generic_post",
    "run_surface_sequence",
    "apply_sea_ice",
    "sea_ice_inputs",
    "sfc_sice_run",
    "apply_noahmp",
    "apply_sfc_land",
    "finish_surface_iteration",
    "noahmp_surface_inputs",
    "NSSTControl",
    "apply_nsst",
    "sfc_nst_run",
    "PhysicsPrefixControl",
    "PhysicsPrefixResult",
    "allocate_physics_prefix_scratch",
    "apply_surface_layer",
    "prepare_physics_prefix",
    "SurfaceCompositesControl",
    "SurfaceCompositesResult",
    "prepare_surface_composites",
    "RadiationScalingControl",
    "RadiationScalingResult",
    "apply_held_radiation",
    "scale_radiation",
    "NativeRadiationResult",
    "RadiationCarry",
    "radiation_group_inputs",
    "reset_radiation_diagnostics",
    "reset_physics_diagnostics",
    "run_native_radiation",
    "PrognosticOzone",
    "ozone_2015",
    "apply_photochemistry",
    "photochemistry_2015",
    "photochemistry_post",
    "initialize_ozone_grid",
    "ozone_forecast_day",
    "update_ozone_forcing",
    "h2o_photochemistry",
    "photochemistry_latitude_weights",
    "update_h2o_forcing",
    "StepClock",
    "advance_utc_date",
    "prepare_physics_calendar",
    "prepare_timestep_clock",
    "RadiationSeeds",
    "advance_mpas_radiation_seeds",
    "radiation_permutation_seed",
    "radiation_seed_offset",
    "random_index",
    "LandInitializationTables",
    "initialize_noahmp_cold_surface",
    "Grid",
    "GridProjection",
    "project_grid",
    "BridgeInitialization",
    "initialize_bridge",
    "ReferenceGeometry",
    "initialize_reference_geometry",
    "CommitDomain",
    "CommitGeometry",
    "NativeCommitResult",
    "NativeCommitState",
    "commit_native",
    "commit_mpas_state",
    "CommitLedgers",
    "CumulativeLimiter",
    "commit_ledgers",
    "SurfaceMirror",
    "SurfaceMirrorInputs",
    "mirror_surface",
    "DiagnosticRefresh",
    "OutputDiagnostics",
    "ReconstructionMesh",
    "refresh_host_diagnostics",
    "AtmosphericProjection",
    "CellWindIncrement",
    "MomentumCommit",
    "MomentumMesh",
    "commit_momentum",
    "commit_surface_pressure",
    "prepare_cell_wind_increment",
    "HydrostaticPressure",
    "NativeAtmosphere",
    "NativeSurface",
    "ProjectionMesh",
    "Statein",
    "Stateout",
    "CommitPhase",
    "ThermodynamicCommit",
    "commit_thermodynamics",
    "Sfcprop",
    "SoilParameters",
    "SurfaceControl",
    "SurfaceInitialization",
    "TracerBinding",
    "TracerContract",
    "hydrostatic_pressure",
    "initialize_surface",
    "project_atmosphere",
    "r4_tracer_contract",
]
