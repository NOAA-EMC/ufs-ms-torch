# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Static runtime switches of the coupled cycle; one numerical code path in all settings.

``set_scheme_compile`` (default off): the ten scheme entry points called by the
cycle (radiation, surface layer, NSST, Noah-MP, sea ice, PBL, GWD, deep and
shallow convection, TEMPO) and ``physics_post`` (selection name "physics_post") each run as
exact-policy compiled regions, built once per static configuration and shape.
PBL, UGWP v1 and convection keep container orchestration eager and reuse bounded
tensor regions. Other entries use their existing whole-entry or regional drivers.
Only Inductor's fusion differs from eager (see ufs_physics_common.compile_policy).

``set_cycle_observers`` (default on): the commit ledgers, the four physical-reservoir ledgers and the
continuation carry receipt (byte snapshot at the end of a cycle, compared at the
start of the next). No physics reads them. Off, the cycle returns no reservoir
totals and neither binds nor checks the carry receipt, and the two native commits form no
ledgers (``NativeCommitResult.ledgers`` is None; the cumulative limiter carry passes through).

``set_column_chunk`` (default None): the column-local kernels (the column work of the two
radiation mains, the TEMPO core, and the PBL, GWD and convection entry points) run over equal
column chunks (``column_chunks``); under grad each chunk is a checkpoint that backward
recomputes. A selected scheme's compiled region is then the chunk; ``run_native_radiation`` and
``apply_tempo`` stay eager around their chunked cores. The shortwave chunks the daylit columns
it gathered from the whole mesh.

``set_unchunked_schemes`` (default empty): selected column-local entries run
as one batch without changing other entries' memory limits. This explicit experiment policy
requires separate memory qualification; checkpoint recomputation retains the forward selection.
Forward steps are decided by ``set_column_chunk(None)`` alone: every entry then runs full batch.
``FULL_BATCH_SCHEMES`` names the entries that a differentiable caller keeps full batch under a column
limit, and ``RADIATION_REVERSE_COLUMNS`` bounds radiation's chunk under grad (``plan_memory``).

The TEMPO compaction (``TEMPO_SEDIMENTATION_COMPACTION`` to start, raised by ``raise_compaction``): the
substep compaction tiers of the coupled TEMPO call (``tempo_torch.torch_native``) and, under
"bound", the fall-speed ceilings of its static substep loop bounds, under "lanes" the lane buffers of its
masked FP64 groups. Each column still runs its own source substep count; later substeps run only on the columns
that need them, and the snow powers only on the lanes that hold snow.
A tier whose buffer is too small sets its bit in the device overflow flag
(``ufs_physics_common.compaction``); forward-only ``run_coupled_interval`` retries only the
affected radiation, physics or microphysics group. AD and graph replay retain their
whole-cycle/window recovery so recomputation uses the forward capacities.

``set_sw_daylit_fraction`` (default 1.0, every column): the buffer of the shortwave column work,
as a fraction of the radiation group's columns. As in the Fortran
(``idxday``), only columns with ``coszen >= 0.0001`` run the shortwave; ``radiation.sw_daylit_bound``
gives the geometric bound of a run's radiation times. Overflow follows the TEMPO tiers
(region ``radiation.sw_day``).

The deep convection buffer (``DEEP_CONVECTION_FRACTION`` to start, raised by ``raise_compaction``): the
buffer of the deep convection cloud model, as a fraction of the columns (or of a column chunk). As in the source,
segments I onwards run only on the columns the trigger (segment H) selects
(``run_sa_sas_deep_convection_compacted``). Overflow follows the TEMPO tiers (region
``convection.deep``).

The shallow convection buffer (``SHALLOW_CONVECTION_FRACTION`` to start, raised by ``raise_compaction``):
the buffer of the shallow convection scheme past its trigger, as a fraction of the columns (or of a column chunk). As
in the source, segments F onwards act only on the columns the trigger (segment E) selects; those
columns run the scheme in the buffer (``run_sa_sas_shallow_convection_compacted``). Overflow follows
the TEMPO tiers (region ``convection.shallow``).

The UGWP v1 caps (``UGWP_V1_NSTAB_CAP`` and the reference start to begin with, raised by
``raise_compaction``): the static loop bounds of the UGWP v1 wave solver (``gwd_v1.ngw_solv2``): the eddy-diffusion substeps it unrolls and whether
the wave loop starts at the reference start level (``kstart``, ``gwd_v1.wave_loop_start`` of the
setup) or at layer 3 as the source.
A column that needs more substeps, or launches below ``kstart``, sets its overflow bit (regions
``gwd.ugwp_v1_nstab``, ``gwd.ugwp_v1_kstart``); the interval driver doubles the substep cap (up to
``UGWP_V1_NSTAB_MAX``) or starts the loop at layer 3 and retries the affected group.
AD and graph replay retain whole-cycle/window recovery, as for the TEMPO tiers.

Dycore checkpointing: a differentiable coupled step runs the dycore with its
nested RK checkpointing (``gradient_checkpointing``): backward recomputes each RK stage from its
inputs instead of keeping the whole RK tape. Exact by recomputation; forward-only steps are not
affected. The key stays out of the dycore's continuation identity.

Cycle checkpointing: a differentiable ``run_coupled_interval`` keeps operation inputs;
backward recomputes them through ``cycle._checkpointed_cycle``. Global multicycle intervals retain
five-field post-dynamics atmospheric phases; initial compatibility inputs and terminal observations
remain separate. Other boundary modes retain completed-cycle inputs. Forward/replay identity is
bitwise; native/carried VJP accumulation can round differently and must pass the established AD gates.
Reverse-mode memory is about one operation's graph plus the carried states.
Each recompute uses the compaction rungs its forward used and
checks on the device that its output state equals the forward's bit for bit (on CUDA both run with
deterministic algorithms). Inside a checkpointed cycle the dycore RK checkpointing and the saved-tensor
offload are suspended (nested selective checkpoints and saved-tensor hooks cannot nest in it).

``set_step_graphs`` (default off): the step-number checks of the cycle compare the device clock
with the carried device step counter (``completed_dycore_steps + 1``) and with the step's static
variant (``cycle.step_variant``) instead of with the static step number, so one captured step
graph replays every step of its variant. Off, they compare ``kdt`` with the static step number.

Boundary guard: the cycle's commits and dycore steps bind the live MPAS boundaries
(``mpas_pytorch.state.boundary_guard``) by host stamps (tensor object, storage view, version
counter): no kernels and no device copies; writes that bypass the version counter (``.data``,
NumPy or DLPack views) are not seen. The observers' carry receipt always compares bytes.

``with_dycore_compile(config)`` is a configuration, not a switch: the dycore's
compile settings enter its continuation identity, so a run selects them before
cold initialization. The same config serves every kind of step (``dynamics.dycore_step_config``):
steps without derivatives run the compiled regions; derivative steps run them only where
qualified (reverse mode with ``reverse=True``) and otherwise eagerly, through a private per-step
key outside the continuation identity. ``reverse=True`` selects the compiled regions under
reverse-mode AD too (see its docstring for the qualification and the forward-mode limit).

The switches are module globals: they hold for the whole process, and no lock guards them. Set them
from one thread, before the run that reads them; two models with different settings need two processes.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeVar

import torch
from ufs_physics_common.compaction import rung_up
from ufs_physics_common.compile_policy import compiled, install_exact_numerics

from .column_chunks import run_column_chunks

Result = TypeVar("Result")

_scheme_compile: bool | frozenset[str] = False
_cycle_observers = True
_step_graphs = False
_saved_tensor_offload = False
_column_chunk: int | None = None
# {kind: ((first substep, column fraction), ...)}. Rain and graupel sediment in one loop (kind
# rain_graupel) that compacts from substep 0 (only columns holding either species sediment); snow
# runs substeps 0-1 on every column. Each buffer holds at least 3x the column share that needed its
# tier after 2 h of the x1.10242 coupled run (2026-10-17): rain any 13.4 %, > 2 substeps 7.0 %,
# > 4 1.1 %; graupel any 4.0 %, > 2 0.23 % (so rain or graupel at most 17.4 %, 7.2 %, 1.1 %);
# snow > 2 8.6 %, > 4 2.0 %, > 8 0.5 %. An undersized buffer sets its overflow bit and the
# interval driver retries the affected group one ladder rung up (raise_compaction).
# "bound": ((kind, fall-speed ceiling m/s), ...) sets the static substep loop bounds
# (torch_native.sedimentation_substep_bound: int(dt*v/dz_min + 1) + 1 substeps, int(n/10) + 1 on
# the semi-Lagrangian rain/graupel path, dz_min the grid's thinnest layer). 20 m/s is the
# worst-case ceiling (DEFAULT_MAX_SEDIMENTATION_FALL_SPEED): 14 rain/graupel and 131 snow
# substeps at dt 300 s, dz_min 46.4 m (x1.10242; the 60 km case, also dt 300 s, has the same 14
# and 131), against at most 6 and 22 needed after 2 h of the x1.10242 run (effective fall speeds
# <= 9.0 and <= 3.2 m/s). The defaults 12 and 5 m/s (1.3x and 1.5x those) give 8 and 34 substeps
# there. A column that
# needs more sets region tempo.bound.<kind>; the driver retries with the next ceiling of
# TEMPO_FALL_SPEED_LADDER, up to the worst case.
TEMPO_FALL_SPEED_LADDER = (5.0, 12.0, 20.0)
# "window": the source cells one semi-Lagrangian destination may span (kt - kb + 1; the static loop of its
# inner-cell sums). Over 5 cycles of the x1.10242 coupled run (2026-11-12, round-4 log entry 8) a destination
# spans at most 3 (1 inner cell, 296 of 35 M destinations in cycle 5); 4 is one cell above. A wider
# destination sets region tempo.window; the driver retries with the next TEMPO_WINDOW_LADDER rung and then
# unbounded (None).
TEMPO_WINDOW_LADDER = (4, 8, 16, 32, 64)
# "lanes": ((group, fraction), ...), the buffers of TEMPO's masked FP64 lane groups (torch_native.set_lane_compaction)
# as fractions of the [column, level] lanes. "snow": the snow moments, sedimentation snow moments and snow fall
# speed (mask l_qs; rs > R1). On the x1.10242 FP32 coupled run of 2024-01-01 (cold start, 2026-11-15) the snow
# share rises to 13.6 % at cycle 6, 21.2 % at 12, 24.0 % at 21 and 24.9 % at 33 (11 h), growing 0.06 % a cycle there
# (fall speed: at most 18.8 %); 0.375 holds 1.5x the largest. 0.25 would overflow after about 10 h and pay a rerun
# and a recompile for about 2 ms of a 60 km step. "cloud": the cloud slope of the cloud checks (mask l_qc); the
# cloud share of the same run is 3.3 % at cycle 1, 10.2 % at 6 and 11.6 % at 10, rising 0.3 % a cycle: 0.25 holds
# 2.2x the largest. "melt": the snow-melt number factor 10^(-0.25 tempc) (mask snow_melt, within warm l_qs lanes);
# warm snow lanes are at most 2.4 % over 12 cycles of the same run: 0.0625 holds 2.6x. "rain": the rain slope of the
# rain checks outside the semi-Lagrangian substeps (their active lanes); over 36 cycles (12 h) of the same run
# (2026-11-16) the rain share rises to 4.3 % at cycle 6, 7.2 % at 12, 9.0 % at 24 and 9.6 % at 36, growing 0.03 % a
# cycle there: 0.1875 holds 1.95x. "ice": the ice slope of the ice checks (their active lanes); 11.4 % at cycle 6,
# 16.5 % at 12, 19.5 % at 24, 20.4 % at 36 (0.04 % a cycle): 0.375 holds 1.84x. The cloud share with the warm-rain
# xdc lanes (cloud_auto) is at most 14.4 % at cycle 36: 0.25 holds 1.74x.
# A larger share sets region tempo.lanes.<group>; the driver retries one LADDER rung up (raise_compaction).
TEMPO_SEDIMENTATION_COMPACTION = {
    "rain_graupel": ((0, 0.5), (2, 0.25), (4, 0.0625)),
    "snow": ((2, 0.375), (4, 0.125), (8, 0.03125)),
    "bound": (("rain_graupel", 12.0), ("snow", 5.0)),
    "window": 4,
    "lanes": (("snow", 0.375), ("cloud", 0.25), ("melt", 0.0625), ("rain", 0.1875), ("ice", 0.375)),
}
_tempo_sedimentation_compaction = TEMPO_SEDIMENTATION_COMPACTION
_sw_daylit_fraction = 1.0
# The lowest ladder rung at or above 3x the triggered deep share of the x1.10242 coupled run
# (about 6 % over 24 cycles, 5.3 % at cycle 1; 2026-10-17/18).
DEEP_CONVECTION_FRACTION = 0.25
_deep_convection_fraction = DEEP_CONVECTION_FRACTION
# The triggered shallow share of the x1.10242 coupled run rises from 24.4 % to 29.7 % over the first
# 24 cycles (2026-10-20): 0.5 holds 1.7x the largest; the 3x rule of the deep buffer would leave the
# scheme dense. A larger share overflows and the driver retries at 1.0.
SHALLOW_CONVECTION_FRACTION = 0.5
_shallow_convection_fraction = SHALLOW_CONVECTION_FRACTION
# UGWP v1 eddy-diffusion substeps: every column of the x1.10242 coupled run needs 1 over 25 cycles
# (2026-10-21, claude/r4-e scratch/m240); 2 is the margin, 16 the ported maximum.
UGWP_V1_NSTAB_CAP = 2
UGWP_V1_NSTAB_MAX = 16
_ugwp_v1_nstab_cap = UGWP_V1_NSTAB_CAP
_ugwp_v1_full_wave_loop = False


def _set(name: str, enabled: bool) -> bool:
    if type(enabled) is not bool:
        raise TypeError(f"{name} switch requires a bool")
    previous = globals()[name]
    globals()[name] = enabled
    return previous


def set_scheme_compile(enabled: bool | Iterable[str]) -> bool | frozenset[str]:
    """Set the whole-scheme compile switch; return the previous value.

    True or False selects every entry point (the ten schemes and the glue) or none; a set of names
    (``fn.__name__``, e.g. ``{"apply_tempo", "run_native_radiation"}``) selects those only.
    """
    global _scheme_compile
    previous = _scheme_compile
    if type(enabled) is bool:
        _scheme_compile = enabled
    else:
        names = frozenset(enabled)
        if not names or any(type(n) is not str for n in names):
            raise TypeError("scheme compile selection requires a bool or a non-empty set of names")
        _scheme_compile = names
    return previous


def set_cycle_observers(enabled: bool) -> bool:
    """Set the reservoir/carry-receipt observer switch; return the previous value."""
    return _set("_cycle_observers", enabled)


# Set while a checkpointed coupled cycle runs or recomputes (cycle._checkpointed_cycle).
_INSIDE_CYCLE_CHECKPOINT: contextvars.ContextVar[bool] = contextvars.ContextVar("inside_cycle_checkpoint", default=False)


def dycore_checkpoint_enabled() -> bool:
    """Whether a differentiable step checkpoints the dycore's RK stages: always, except inside a
    checkpointed cycle (nested selective checkpoints cannot nest in it)."""
    return not _INSIDE_CYCLE_CHECKPOINT.get()


def set_saved_tensor_offload(enabled: bool) -> bool:
    """Keep the tensors a differentiable coupled cycle (radiation, physics, dycore, microphysics)
    saves for backward in host memory (``torch.autograd.graph.save_on_cpu``); return the previous
    value. For meshes whose reverse-mode retained bytes exceed device memory; gradients are unchanged."""
    return _set("_saved_tensor_offload", enabled)


def saved_tensor_offload_enabled() -> bool:
    return _saved_tensor_offload and not _INSIDE_CYCLE_CHECKPOINT.get()


_step_local_fields = True


def set_step_local_fields(enabled: bool) -> bool:
    """Drop the dycore's step-local work buffers from the carried state (default on); off keeps them, so
    comparison tools can check every Fortran exit field (dynamics.STEP_LOCAL_NATIVE_FIELDS). Numerics do
    not change either way. Return the previous value."""
    return _set("_step_local_fields", enabled)


def step_local_fields_enabled() -> bool:
    return _step_local_fields


_overflow_check_steps = 12


def overflow_check_steps() -> int:
    """Replayed steps between reads of the compaction overflow flag (step graphs)."""
    return _overflow_check_steps


def set_step_graphs(enabled: bool) -> bool:
    """Set the graph-replayable step checks (``step_graphs_enabled``); return the previous value."""
    return _set("_step_graphs", enabled)


def step_graphs_enabled() -> bool:
    return _step_graphs


def set_column_chunk(columns: int | None) -> int | None:
    """Columns per chunk of the column-local kernels (None: one call); return the previous value.

    RRTMGP holds [g-point, layer, column] work arrays (about 1.7 MB per column) and the reverse
    pass of TEMPO keeps most of a cycle's saved tensors, so large meshes need chunks.
    """
    global _column_chunk
    # Above every non-column extent (layers, bands, g-points), so the column axis is unambiguous.
    if columns is not None and (type(columns) is not int or columns <= 512):
        raise ValueError("column chunk must be an int above 512 or None")
    previous, _column_chunk = _column_chunk, columns
    return previous


def column_chunk() -> int | None:
    return _column_chunk


def tempo_sedimentation_compaction() -> dict:
    return _tempo_sedimentation_compaction


def set_sw_daylit_fraction(fraction: float) -> float:
    """Set the shortwave daylit buffer fraction (0 < fraction <= 1); return the previous value."""
    global _sw_daylit_fraction
    if type(fraction) is not float or not 0.0 < fraction <= 1.0:
        raise ValueError("shortwave daylit fraction must be a float in (0, 1]")
    previous, _sw_daylit_fraction = _sw_daylit_fraction, fraction
    return previous


def sw_daylit_fraction() -> float:
    return _sw_daylit_fraction


def deep_convection_fraction() -> float:
    return _deep_convection_fraction


def shallow_convection_fraction() -> float:
    return _shallow_convection_fraction


def ugwp_v1_caps() -> tuple[int, bool]:
    return _ugwp_v1_nstab_cap, _ugwp_v1_full_wave_loop


def compaction_state() -> tuple:
    """The compaction capacities in force (TEMPO tiers, SW daylit, deep and shallow convection fractions,
    UGWP v1 caps)."""
    return (_tempo_sedimentation_compaction, _sw_daylit_fraction, _deep_convection_fraction,
            _shallow_convection_fraction, ugwp_v1_caps())


def restore_compaction_state(state: tuple) -> tuple:
    """Put back capacities from ``compaction_state``; return the ones replaced. A state without the
    UGWP v1 caps (four entries) keeps the caps in force."""
    global _tempo_sedimentation_compaction, _sw_daylit_fraction, _deep_convection_fraction
    global _shallow_convection_fraction, _ugwp_v1_nstab_cap, _ugwp_v1_full_wave_loop
    previous = compaction_state()
    (_tempo_sedimentation_compaction, _sw_daylit_fraction, _deep_convection_fraction,
     _shallow_convection_fraction) = state[:4]
    if len(state) > 4:
        _ugwp_v1_nstab_cap, _ugwp_v1_full_wave_loop = state[4]
    return previous


def raise_compaction(region: str) -> str:
    """Move an overflowed compacted region one capacity-ladder rung up; return a log line.

    Regions are named by their scheme: ``tempo.<kind>.<first substep>`` for the TEMPO tiers,
    ``tempo.bound.<kind>`` for the TEMPO substep loop bounds (fall-speed ceiling up one
    ``TEMPO_FALL_SPEED_LADDER`` rung; past the worst case the run stops), ``tempo.window`` for the
    semi-Lagrangian sources of one destination (next ``TEMPO_WINDOW_LADDER`` rung, then unbounded),
    ``tempo.lanes.<group>`` for a TEMPO FP64 lane buffer (next ``LADDER`` rung),
    ``radiation.sw_day`` for the shortwave daylit columns, ``convection.deep`` for the deep cloud model,
    ``convection.shallow`` for the shallow scheme past its trigger, ``gwd.ugwp_v1_nstab`` and
    ``gwd.ugwp_v1_kstart`` for the UGWP v1 substep cap and wave-loop start.
    """
    global _tempo_sedimentation_compaction, _sw_daylit_fraction, _deep_convection_fraction
    global _shallow_convection_fraction, _ugwp_v1_nstab_cap, _ugwp_v1_full_wave_loop
    if region == "gwd.ugwp_v1_nstab":
        if _ugwp_v1_nstab_cap >= UGWP_V1_NSTAB_MAX:
            raise RuntimeError(f"UGWP v1 eddy-diffusion substeps above the ported maximum {UGWP_V1_NSTAB_MAX}")
        old, _ugwp_v1_nstab_cap = _ugwp_v1_nstab_cap, 2 * _ugwp_v1_nstab_cap
        return f"{region}: substep cap {old} -> {_ugwp_v1_nstab_cap}"
    if region == "gwd.ugwp_v1_kstart":
        if _ugwp_v1_full_wave_loop:
            raise RuntimeError("UGWP v1 launch level below layer 3")
        _ugwp_v1_full_wave_loop = True
        return f"{region}: wave loop from the reference start level -> layer 3"
    if region == "radiation.sw_day":
        old, _sw_daylit_fraction = _sw_daylit_fraction, rung_up(_sw_daylit_fraction)
        return f"{region}: column fraction {old} -> {_sw_daylit_fraction}"
    if region == "convection.deep":
        old, _deep_convection_fraction = _deep_convection_fraction, rung_up(_deep_convection_fraction)
        return f"{region}: column fraction {old} -> {_deep_convection_fraction}"
    if region == "convection.shallow":
        old, _shallow_convection_fraction = _shallow_convection_fraction, rung_up(_shallow_convection_fraction)
        return f"{region}: column fraction {old} -> {_shallow_convection_fraction}"
    if region.startswith("tempo.lanes."):
        group = region.removeprefix("tempo.lanes.")
        fractions = dict(_tempo_sedimentation_compaction["lanes"]) if "lanes" in _tempo_sedimentation_compaction else {}
        old = fractions[group] if group in fractions else 1.0
        fractions[group] = rung_up(old)
        _tempo_sedimentation_compaction = {**_tempo_sedimentation_compaction, "lanes": tuple(fractions.items())}
        return f"{region}: lane fraction {old} -> {fractions[group]}"
    if region == "tempo.window":
        old = _tempo_sedimentation_compaction["window"] if "window" in _tempo_sedimentation_compaction else None
        if old is None:
            raise RuntimeError(f"{region}: an unbounded semi-Lagrangian window overflowed")
        new = next((sources for sources in TEMPO_WINDOW_LADDER if sources > old), None)
        _tempo_sedimentation_compaction = {**_tempo_sedimentation_compaction, "window": new}
        return f"{region}: sources {old} -> {new}"
    parts = region.split(".")
    if len(parts) == 3 and parts[0] == "tempo" and parts[1] == "bound":
        kind = parts[2]
        ceilings = dict(_tempo_sedimentation_compaction["bound"]) if "bound" in _tempo_sedimentation_compaction else {}
        old = ceilings[kind] if kind in ceilings else TEMPO_FALL_SPEED_LADDER[-1]
        higher = [speed for speed in TEMPO_FALL_SPEED_LADDER if speed > old]
        if not higher:
            raise RuntimeError(f"{region}: a column needs more TEMPO substeps than the worst-case "
                               f"fall-speed ceiling {old} m/s allows")
        ceilings[kind] = higher[0]
        _tempo_sedimentation_compaction = {**_tempo_sedimentation_compaction, "bound": tuple(ceilings.items())}
        return f"{region}: fall-speed ceiling {old} -> {higher[0]} m/s"
    if len(parts) == 3 and parts[0] == "tempo" and parts[1] in _tempo_sedimentation_compaction:
        kind, first = parts[1], int(parts[2])
        tiers = _tempo_sedimentation_compaction[kind]
        if first not in (f for f, _ in tiers):
            raise KeyError(f"no TEMPO {kind} tier starts at substep {first}")
        old = dict(tiers)[first]
        new = rung_up(old)
        _tempo_sedimentation_compaction = {**_tempo_sedimentation_compaction,
            kind: tuple((f, new if f == first else fraction) for f, fraction in tiers)}
        return f"{region}: column fraction {old} -> {new}"
    raise KeyError(f"unknown compacted region {region!r}")


def scheme_compile_selected(name: str) -> bool:
    return _scheme_compile is True or (type(_scheme_compile) is frozenset and name in _scheme_compile)


def cycle_observers_enabled() -> bool:
    return _cycle_observers


# Dycore regions compiled: mpas-port's qualified list (orchestration/scripts/fusion_cpu_gate.py
# CANDIDATE_TARGETS: the 17 regions of the parallel-efficiency round 3 GPU campaign plus 14 from the
# structural efficiency round). With all ten schemes compiled, 25 coupled CPU cycles (x1.10242, two
# radiation steps) equal eager in all 652 state tensors (2026-10-20); the dycore's per-step bytes
# outside compiled code fall to 4 GiB (inputs), its fused bytes are 26 GiB (Inductor estimate).
DYCORE_COMPILE_TARGETS = (
    "acoustic_step_acoustic_flux_solve_region",
    "dynamics_tendency_normal_velocity_tendency_normal_velocity_section",
    "dynamics_tendency_theta_horizontal_advection_theta_finalization_theta_tendency_region",
    "dynamics_tendency_vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency_region",
    "stencil_ordered_weighted_sum",
    "stencil_ordered_weighted_components",
    "column_ordered_factor",
    "monotonic_scalar_transport_streaming_scalar_transport_region",
    "dynamics_tendency_normal_velocity_mixing_normal_velocity_finalization_u_tendency_region",
    "dynamics_tendency_normal_velocity_finalization_u_final_section",
    "dynamics_tendency_vertical_velocity_horizontal_advection_w_hadv_section",
    "small_step_perturbation_smlstep_pert_w_tendency_region",
    "dynamics_tendency_divergence_damping_region",
    "explicit_scalar_transport_horizontal_transport_composition",
    "explicit_scalar_transport_vertical_finish_region",
    "large_step_recovery_recovery_region",
    "acoustic_step_horizontal_momentum_update",
    "diagnostic_solve_solve_diagnostics",
    "monotonic_scalar_transport_vertical_neighbor_bounds_region",
    "dynamics_tendency_density_tendency_density_section",
    "vertical_implicit_coefficients_vert_imp_coefs",
    "dynamics_tendency_mixing_initialization_section",
    "small_step_perturbation_moist_coefficients_region",
    "large_step_recovery_recovery_pack_region",
    "acoustic_step_acoustic_flux_solve_pack_region",
    "acoustic_step_horizontal_momentum_pack_region",
    "monotonic_scalar_transport_source_tendency_region",
    "coupled_diagnostics_region",
    "vector_reconstruction_region",
    "monotonic_scalar_transport_density_update_region",
)


def with_dycore_compile(config: dict, *, reverse: bool = False) -> dict:
    """Return config with the dycore's compiled regions selected.

    Mode "default": the coupled cycle is captured as one CUDA graph by its caller, so
    the dycore must not record its own. CUDA policy "force": the dycore registry holds
    no GPU-validated region, and "safe" would leave every region eager on CUDA.

    ``reverse=True`` also runs the regions under reverse-mode AD, through the dycore's explicit
    experiment switches. It selects only targets admitted by the dycore's reverse-AD registry;
    forward-only targets keep the same eager PyTorch implementation. Qualified on CPU
    in the coupled x1.10242 case, policy v12 (2026-10-18): forward bitwise to the eager
    differentiable step, wind VJP within 4.1e-14 of eager, adjoint identity 0.0 against the
    eager JVP. Forward-mode AD through compiled dycore regions is not qualified (the dycore
    rejects dual tensors); forward-mode steps run the regions eagerly, as do reverse-mode steps
    of the default (forward-only) config.
    """
    # The dycore's Inductor regions keep eager arithmetic only under the exact policy (signed zeros of
    # `x + 0`, no FMA contraction on CUDA). It is process-wide and must precede the first compile; before
    # this call only a scheme region installed it, so a dycore-only compile gave -0.0 where eager gives
    # +0.0 (w at the surface, the explicit transport tendency without physics).
    install_exact_numerics()
    selected = {**config, "config_use_torch_compile": True, "config_torch_compile_backend": "inductor",
                "config_torch_compile_mode": "default", "config_torch_compile_scope": "all",
                "config_torch_compile_target_kernels": DYCORE_COMPILE_TARGETS,
                "config_torch_compile_cuda_inductor_policy": "force"}
    if reverse:
        from mpas_pytorch.dynamics.compile_utils import kernel_compiled_reverse_ad_enabled_for_device

        selected.update(config_torch_compile_differentiable_policy="experiment",
                        config_torch_compile_differentiable_evidence_generation=True)
        selected["config_torch_compile_target_kernels"] = tuple(
            name for name in DYCORE_COMPILE_TARGETS
            if kernel_compiled_reverse_ad_enabled_for_device(name, selected, device="cpu"))
    return selected


def dycore_forward_only(config: dict) -> bool:
    """True when the config's compiled dycore regions are qualified for steps without derivatives
    only (not ``reverse=True``); derivative steps then run the regions eagerly."""
    return bool(config.get("config_use_torch_compile", False)) and not (
        config.get("config_torch_compile_differentiable_policy") == "experiment"
        and config.get("config_torch_compile_differentiable_evidence_generation") is True)


# Column-local entry points whose first argument is the Statein: with column chunks set, each
# runs over chunks of its columns (every [ncol, ...] input and output).
_STATEIN_CHUNKED = frozenset({"run_pbl_sequence", "apply_unified_gwd", "apply_ugwpv1_gsldrag",
                              "apply_convection"})
_COLUMN_ENTRIES = _STATEIN_CHUNKED | {"apply_tempo", "run_native_radiation"}
_unchunked_schemes: frozenset[str] = frozenset()
# The column-local entries a differentiable run keeps full batch under a column limit
# (set_unchunked_schemes(FULL_BATCH_SCHEMES)); radiation alone keeps the limit.
FULL_BATCH_SCHEMES = frozenset({"run_pbl_sequence", "apply_convection", "apply_unified_gwd",
                                "apply_ugwpv1_gsldrag", "apply_tempo"})
# The most columns one radiation chunk holds under grad: it bounds the reverse graph of the RRTMGP
# [g-point, layer, column] work arrays that one chunk keeps for backward, whatever larger limit is set.
RADIATION_REVERSE_COLUMNS = 1024


def set_unchunked_schemes(names: Iterable[str]) -> frozenset[str]:
    """Run these column-local entries as one batch; retain other memory limits.

    This is an explicit memory policy, not an automatic fit claim. Qualify each
    selected entry's peak before using it on a larger mesh or under derivatives.
    """
    global _unchunked_schemes
    if isinstance(names, str):
        raise TypeError("unchunked schemes must be an iterable of entry names")
    selected = frozenset(names)
    if selected - _COLUMN_ENTRIES:
        raise ValueError(f"unknown column-local entries: {sorted(selected - _COLUMN_ENTRIES)}")
    previous, _unchunked_schemes = _unchunked_schemes, selected
    return previous


def unchunked_schemes() -> frozenset[str]:
    return _unchunked_schemes


def scheme_column_chunk(name: str) -> int | None:
    """The column limit for this entry; full-batch selection preserves other limits."""
    chunk = None if name in _unchunked_schemes else _column_chunk
    if chunk is not None and name == "run_native_radiation" and torch.is_grad_enabled():
        chunk = min(chunk, RADIATION_REVERSE_COLUMNS)
    return chunk


def glue(fn: Callable[..., Result]) -> Callable[..., Result]:
    """``scheme(fn)`` for the coupling glue (projection, prefix preparation, forcing, commits, posts,
    clock): compiled under the scheme switch when no gradient is recorded, else ``fn``. Compiled, the
    glue's ops left outside compiled code fall from 5,214 to about 300 per plain step at x1.10242
    (2026-10-20); its reverse mode is not qualified, so differentiable steps run it eagerly.
    """
    return fn if torch.is_grad_enabled() else scheme(fn)


def scheme(fn: Callable[..., Result]) -> Callable[..., Result]:
    """The compiled form of a scheme entry point when the switch is on, else ``fn``; column-chunked
    (``set_column_chunk``) for the Statein-first column-local entry points."""
    call = compiled(fn) if scheme_compile_selected(fn.__name__) else fn
    size = _column_chunk
    if size is None or fn.__name__ not in _STATEIN_CHUNKED or fn.__name__ in _unchunked_schemes:
        return call

    def chunked(statein, *args, **kwargs):
        ncol = statein.tgrs.shape[0]
        if ncol <= size:
            return call(statein, *args, **kwargs)
        control = kwargs.get("control")
        sized = isinstance(control, dict) and control.get("im") == ncol

        def bound(target):
            def run(c):
                # The static batch size states the chunk's extent; source checks compare it to tensors.
                args, kw = c
                if sized:
                    kw = {**kw, "control": {**kw["control"], "im": args[0].tgrs.shape[0]}}
                return target(*args, **kw)
            return run

        return run_column_chunks(bound(call), ((statein, *args), kwargs), ncol=ncol, size=size, axis=0,
                                 views=True, warm=None if call is fn else bound(call.warm))
    return chunked


# Device memory of one coupled step, per column at 55 levels (scaled linearly in levels), measured
# on CPU at x1.10242 in the performance configuration, 2026-10-18 (ccpp scratch/s60): census peak
# live 5.32 GiB at column chunk 2561 and 14.37 GiB unchunked give the whole-mesh share (dycore
# transients, glue intermediates) and the share per chunk column (largest scheme transient);
# reverse-mode retained bytes after the dycore scalar-transport checkpoint: dycore 3.36 GiB,
# physics and glue 1.35 GiB. Reverse mode adds a backward excess over the forward peak: process RSS
# grew 31.9 GiB during the backward of one x1.163842 cycle (CPU, chunk 73728, 2026-10-19: forward
# peak 203.0, total 234.9 GiB); the same measure at x1.10242 chunk 2561 gives 1.85 GiB against 2.0
# predicted, and 0 unchunked (the forward peak dominates there), so the additive term is
# conservative. A model for planning: it does not hold the allocator's caching or fixed tables;
# check a plan against the device before a long run.
_GIB = 2**30
# Forward, from the census (tools/coupled/census2.py, one cold cycle, x1.163842 GFS v17 performance case,
# dycore tracer tiles of 2, MVP2 M4, 2026-10-20; ccpp-s60 scratch/mvp2/m4_census60_*): peak live 57.26 GiB at
# column chunk 32768 and 69.13 GiB at 43008 (radiation) give the line whole + chunk * columns; at 21504 the
# peak is the dycore step, 45.61 GiB with tracer tiles of 2 and 56.07 GiB untiled: the whole-mesh floor.
_CHUNK_BYTES_PER_COLUMN = (69.13 - 57.26) * _GIB / (43008 - 32768)
_WHOLE_BYTES_PER_COLUMN = (57.26 * _GIB - _CHUNK_BYTES_PER_COLUMN * 32768) / 163843
_FLOOR_BYTES_PER_COLUMN = 45.61 * _GIB / 163843
_UNTILED_FLOOR_BYTES_PER_COLUMN = 56.07 * _GIB / 163843
SCALAR_TRANSPORT_TILE = 2  # tracers per dycore scalar-transport tile when the plan needs the lower floor
_RETAINED_PHYSICS_BYTES_PER_COLUMN = 1.35 * _GIB / 10243
_RETAINED_DYCORE_BYTES_PER_COLUMN = 3.36 * _GIB / 10243
# Resident before the step: the cold-start coupled state (cold boundary under the stamps guard) with the
# case's mesh and cycle arguments, the case's initial fields freed: 16.84 GiB at x1.163842 (107.8 KiB per
# column; scratch/mvp2/resident_union.py). The census counts only tensors made during its cycle, so this term
# is separate (Phase E: the 60 km forward ran out of an 80 GB A100 with the case and state held and no such
# term). Callers that keep the case on the device pass resident_bytes.
_RESIDENT_STATE_BYTES_PER_COLUMN = 16.84 * _GIB / 163843
# Reverse: retained activations from the x1.10242 reverse measurements; the backward excess is fitted so
# that the plan reproduces the x1.163842 reverse cycle at chunk 73728 (CPU, 16 threads, peak RSS 233.9
# GiB, lean-boundary code before the M4 forward changes, 2026-10-20; tools/coupled/smoke.py reverse), so it is
# conservative for the current code.
_BACKWARD_EXCESS_BYTES_PER_COLUMN = (233.9 * _GIB - 163842 * (
    _RESIDENT_STATE_BYTES_PER_COLUMN + _WHOLE_BYTES_PER_COLUMN + _RETAINED_PHYSICS_BYTES_PER_COLUMN
    + _RETAINED_DYCORE_BYTES_PER_COLUMN) - 73728 * _CHUNK_BYTES_PER_COLUMN) / 163842


@dataclass(frozen=True)
class MemoryPlan:
    """Settings for one differentiable coupled step to fit ``budget_bytes``; see ``plan_memory``."""

    column_chunk: int | None
    offload: bool
    predicted_bytes: int
    budget_bytes: int
    feasible: bool
    scalar_chunk: int | None = None  # dycore tracer tile (config_scalar_transport_chunk_size); None: untiled


def _bytes_scale(nlev: int, dtype: torch.dtype) -> float:
    """The per-column constants' scale: linear in levels (55 measured) and in the element size (FP64 measured)."""
    if not dtype.is_floating_point:
        raise ValueError("memory plan needs a floating working dtype")
    return nlev / 55 * torch.empty((), dtype=dtype).element_size() / 8


def resident_state_bytes(*, ncol: int, nlev: int, dtype: torch.dtype = torch.float64) -> int:
    """Planned device bytes of the coupled state held across a step (cold-start state, lean boundary)."""
    return int(_RESIDENT_STATE_BYTES_PER_COLUMN * ncol * _bytes_scale(nlev, dtype))


def plan_memory(*, ncol: int, nlev: int, budget_bytes: int, dtype: torch.dtype = torch.float64,
                min_chunk: int = 2048, granule: int = 1024, resident_bytes: int | None = None) -> MemoryPlan:
    """The largest column chunk (or none) and the saved-tensor offload choice for which one differentiable
    coupled step fits ``budget_bytes``.

    The model is linear in columns, per-column constants measured on CPU in FP64 at 55 levels and scaled
    linearly in levels and in the working dtype's element size (``dtype``): a whole-mesh share, a share per
    chunk column, the activations a step retains for backward and the backward's excess over the forward
    peak. It does not hold the allocator's caching or fixed tables; check a plan against the device before a
    long run. Forward steps need no plan: they run every scheme full batch (``set_column_chunk(None)``).

    ``resident_bytes`` is the device memory held before the step (state, case, tables); by default the
    planned coupled state alone (``resident_state_bytes``). A caller that keeps the loaded case on the
    device passes its measured allocation.

    Offload moves the cycle's retained activations to host memory; it is chosen only when the
    step does not fit without it. Dycore tracer tiles (``scalar_chunk``, bitwise the untiled update) lower
    the whole-mesh floor; they are chosen only when the untiled floor does not fit. ``feasible`` is false
    when even ``min_chunk`` columns do not fit: the mesh then needs more device memory or a split over
    devices. ``planned_config`` applies the tile to the dycore configuration. Under the chunk, a caller
    keeps ``FULL_BATCH_SCHEMES`` full batch with ``set_unchunked_schemes``; radiation's chunk under grad is
    at most ``RADIATION_REVERSE_COLUMNS``.
    """
    if ncol <= 0 or nlev <= 0 or budget_bytes <= 0:
        raise ValueError("memory plan needs positive ncol, nlev and budget")
    scale = _bytes_scale(nlev, dtype)
    whole = _WHOLE_BYTES_PER_COLUMN * ncol * scale
    floor = _UNTILED_FLOOR_BYTES_PER_COLUMN * ncol * scale
    per_chunk_column = _CHUNK_BYTES_PER_COLUMN * scale
    retained = _RETAINED_PHYSICS_BYTES_PER_COLUMN * ncol * scale + _RETAINED_DYCORE_BYTES_PER_COLUMN * ncol * scale
    backward = _BACKWARD_EXCESS_BYTES_PER_COLUMN * ncol * scale
    held = resident_state_bytes(ncol=ncol, nlev=nlev, dtype=dtype) if resident_bytes is None else resident_bytes

    def forward_peak(columns: int) -> float:
        return held + max(whole + per_chunk_column * columns, floor)

    offload = forward_peak(min(min_chunk, ncol)) + retained + backward > budget_bytes
    extra = backward + (0.0 if offload else retained)
    scalar_chunk = None
    if held + floor + extra > budget_bytes:
        scalar_chunk, floor = SCALAR_TRANSPORT_TILE, _FLOOR_BYTES_PER_COLUMN * ncol * scale
    room = budget_bytes - extra - held - whole
    if room >= per_chunk_column * ncol:
        chunk, used = None, ncol
    else:
        chunk = max(int(room // per_chunk_column) // granule * granule, 0)
        used = chunk
    feasible = (chunk is None or chunk >= min(min_chunk, ncol)) and held + floor + extra <= budget_bytes
    return MemoryPlan(chunk, offload, int(forward_peak(used) + extra), int(budget_bytes), feasible, scalar_chunk)


def planned_config(config: dict, plan: MemoryPlan) -> dict:
    """The dycore configuration with the plan's tracer tile (unchanged when the plan has none)."""
    if plan.scalar_chunk is None:
        return config
    return {**config, "config_scalar_transport_chunk_size": plan.scalar_chunk}


def use_expandable_segments(device) -> None:
    """Expandable segments for the CUDA caching allocator (no-op off CUDA); call before the run's first device
    allocation. With fixed segments a large request fails while freed memory sits in partly used segments: the
    60 km forward ran out at a 4.03 GiB request with 15 GiB cached (A100, MVP2 M8), reproduced by replaying the
    step's allocation stream through the allocator's rules (scratch/mvp2/alloc_sim.py); expandable segments
    unmap free pages and map them where the request fits. CUDA graph pools use them as well (PyTorch 2.8)."""
    import torch

    if torch.device(device).type == "cuda":
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")


def apply_memory_plan(plan: MemoryPlan) -> dict:
    """Set the column chunk, saved-tensor offload and observers (off) of ``plan``; return the previous values."""
    if not plan.feasible:
        raise ValueError(f"no single-device memory plan fits {plan.budget_bytes / _GIB:.1f} GiB "
                         f"(the whole-mesh share alone needs {plan.predicted_bytes / _GIB:.1f} GiB)")
    return {"column_chunk": set_column_chunk(plan.column_chunk),
            "offload": set_saved_tensor_offload(plan.offload),
            "cycle_observers": set_cycle_observers(False)}
