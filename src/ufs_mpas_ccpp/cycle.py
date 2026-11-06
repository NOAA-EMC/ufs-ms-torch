# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""One selected R4 coupling cycle, retaining its consumed continuation state."""

import contextlib
import contextvars
import functools
import types
import weakref
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Callable, Mapping

import torch
from tempo_torch.mp_tempo import MpTempoInitialization, mp_tempo_init
from tempo_torch.torch_native import TorchFrozenTables
from rrtmgp_radiation_torch.setup import _solar_state
from mpas_pytorch.state import (_UFS_COMMIT_ENTRY, _LiveBoundarySnapshot, _TensorStamp, boundary_guard,
                                mark_complete_step_boundary)

from .clock import StepClock, advance_utc_date, prepare_physics_calendar, prepare_timestep_clock
from .continuation import bind_coupled_boundary, require_coupled_boundary
from .containers import NativeAtmosphere, ProjectionMesh, Statein, static_tensor
from .grid import Grid
from ufs_physics_common.compaction import overflow_flag, overflowed_regions, reset_overflow
from ufs_physics_common.compile_policy import eager_regions, eager_regions_active
from ufs_physics_common import cost_meter
from .convective_clouds import ConvectiveCloudHistory, convective_cloud_clock
from .distributed import consensus_regions, partition_scope
from .dynamics import (
    MicrophysicsCommitResult, PhysicsDynamicsResult,
    advance_physics_dynamics, commit_microphysics, dycore_step_config, native_boundary_supported,
)
from .ledger import CumulativeLimiter
from .precision import WORKING_DTYPES
from .initialization import BridgeInitialization, initialize_bridge
from .land_initialization import LandInitializationTables, initialize_noahmp_cold_surface
from .microphysics import MicrophysicsSequenceResult, run_microphysics_sequence
from .microphysics_prefix import prepare_microphysics_prefix
from .ozone import (initialize_ozone_grid, ozone_forecast_day, photochemistry_latitude_weights,
    update_h2o_forcing, update_ozone_forcing)
from .physics_prefix import PhysicsPrefixResult, prepare_physics_prefix
from .physics_sequence import PhysicsSequenceResult, run_physics_sequence
from .projection import AtmosphericProjection, project_atmosphere, r4_tracer_contract
from .radiation import (
    RadiationCarry, NativeRadiationResult, reset_radiation_diagnostics,
    reset_physics_diagnostics, run_native_radiation, physics_reset_step, radiation_reset_step)
from .radiation_seeds import (RadiationSeeds, RadiationSeedStream, advance_mpas_radiation_seeds, radiation_seed_draws,
    radiation_seed_offset)
from .reservoirs import physical_reservoirs
from . import runtime_policy
from .runtime_policy import (glue, boundary_guard_mode, saved_tensor_offload_enabled, column_chunk, cycle_observers_enabled, raise_compaction, scheme,
    step_graphs_enabled, compaction_state, cycle_checkpoint_enabled, restore_compaction_state)
from .surface import initialize_surface
from .surface_containers import NativeSurface, SoilParameters, SurfaceControl

_CAPACITY_RETRY = contextvars.ContextVar("coupled_capacity_retry", default=None)


def _run_capacity_region(fn, *args, capacity_check=True, **kwargs):
    """Retry only this immutable-input physics group, outside compiled numerics and graph capture."""
    scope = _CAPACITY_RETRY.get()
    if scope is None or not capacity_check:
        return fn(*args, **kwargs)
    device, step_index, exchanger = scope
    incoming = overflow_flag(device).clone()
    result = None
    while True:
        result = None
        reset_overflow(device)
        result = fn(*args, **kwargs)
        regions = consensus_regions(overflowed_regions(device), exchanger)
        if not regions:
            overflow_flag(device).bitwise_or_(incoming)
            return result
        for region in regions:
            COMPACTION_LOG.append((step_index, raise_compaction(region)))


@dataclass(frozen=True)
class CoupledCycleState:
    """Consumed continuation state with its completed-cycle carry receipt.

    Initialization supplies these members before cycle1. After each cycle the
    native state and all consumed physics carry come from that cycle's outputs.
    step_index is the static scheduler index; sec independently checks it on-device.
    The carried prefix holds Sfcprop, the radiation/diagnostic carry and phy_f2d; its stateout
    and interstitial work arrays are step-local (no step reads them from the previous state;
    the cycle result keeps them as ``result.microphysics.prefix``).
    """

    native_state: dict[str, Any]
    native_surface: NativeSurface
    statein: Statein
    prefix: PhysicsPrefixResult
    hpbl: torch.Tensor
    ud_mf: torch.Tensor
    cloud_history: ConvectiveCloudHistory
    seeds: RadiationSeeds
    solar: Mapping[str, torch.Tensor]
    yearlen: torch.Tensor
    cumulative: CumulativeLimiter
    sec: torch.Tensor
    completed_dycore_steps: torch.Tensor
    step_index: int
    continuation: _LiveBoundarySnapshot | None = None


@dataclass(frozen=True)
class CoupledCycleResult:
    state: CoupledCycleState
    clock: StepClock
    projected_statein: Statein
    radiation: NativeRadiationResult
    physics_entry: PhysicsPrefixResult
    physics: PhysicsSequenceResult
    dynamics: PhysicsDynamicsResult
    microphysics_projection: AtmosphericProjection
    microphysics_entry: PhysicsPrefixResult
    microphysics: MicrophysicsSequenceResult
    final: MicrophysicsCommitResult
    reservoirs: Mapping[str, Mapping[str, torch.Tensor]]


@dataclass(frozen=True)
class ColdCycleInitialization:
    bridge: BridgeInitialization
    state: CoupledCycleState
    tempo: MpTempoInitialization


@dataclass(frozen=True)
class _CoupledDycorePhase:
    """Inputs consumed after dynamics, before convection and microphysics.

    This phase keeps the source's pre-momentum-update vertical velocity.
    It is not the conventional completed-cycle restart boundary.
    """

    native_state: dict[str, Any]
    native_surface: NativeSurface
    statein: Statein | None
    prefix: PhysicsPrefixResult
    hpbl: torch.Tensor
    ud_mf: torch.Tensor
    cloud_history: ConvectiveCloudHistory
    seeds: RadiationSeeds
    solar: Mapping[str, torch.Tensor]
    yearlen: torch.Tensor
    cumulative: CumulativeLimiter
    sec: torch.Tensor
    completed_dycore_steps: torch.Tensor
    step_index: int
    clock: StepClock
    forcing_date: torch.Tensor
    native_context: Any = None
    projection_support: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _CoupledCycleHead:
    phase: _CoupledDycorePhase
    radiation: NativeRadiationResult
    physics_entry: PhysicsPrefixResult
    physics: PhysicsSequenceResult
    dynamics: PhysicsDynamicsResult
    reservoirs: Mapping[str, Mapping[str, torch.Tensor]]


@dataclass(frozen=True)
class _CoupledCycleTail:
    state: CoupledCycleState
    microphysics_projection: AtmosphericProjection
    microphysics_entry: PhysicsPrefixResult
    microphysics: MicrophysicsSequenceResult
    final: MicrophysicsCommitResult
    reservoirs: Mapping[str, Mapping[str, torch.Tensor]]


_PROJECTION_SUPPORT_KEYS = ("def_1", "def_2", "def_3", "diss_est", "smc", "stc", "slc")
_FIXED_PROJECTION_KEYS = ("def_1", "def_2", "def_3", "diss_est", "lightning_threat")
_SOIL_PROJECTION_KEYS = ("smc", "stc", "slc")
# The group reset creates all other consumed work. These inputs survive it.
_POST_DYNAMICS_WORK_KEYS = ("otsptflag", "prcpmp", "rainmp", "icemp", "snowmp", "graupelmp")


# The public phase uses the same implementation as internal interval checkpoints.
NativeCoupledPhase = _CoupledDycorePhase


@dataclass(frozen=True)
class NativeCoupledContext:
    """Fixed dycore support and physics projection controls, outside restart payloads.

    These are live inputs, not detached caches. Soil state belongs to the phase.
    """

    dycore: Any
    projection: Mapping[str, Any]


def prepare_native_coupled_context(state, *, mesh) -> NativeCoupledContext:
    """Prepare fixed context from cold setup or an existing compact phase.

    Cold setup need not integrate a timestep. Neither atmospheric diagnostics
    nor soil state enter this context. Use the run's step-local output policy.
    """
    from mpas_pytorch.atmosphere_model import _native_step_context

    if isinstance(state, NativeCoupledPhase) and state.native_context is not None:
        return NativeCoupledContext(state.native_context,
            {name: state.projection_support[name] for name in _FIXED_PROJECTION_KEYS})
    projection = {name: getattr(state.statein, name) for name in _FIXED_PROJECTION_KEYS[:-1]}
    projection["lightning_threat"] = state.statein.wgrs is not None
    from .dynamics import step_local_fields_enabled

    native = {**mesh, **state.native_state}
    if not step_local_fields_enabled():
        # RK setup adds these fixed aliases; full-output phases retain them.
        for alias, source in (("fnm", "fzm"), ("fnp", "fzp"), ("rdnw", "rdzw")):
            native.setdefault(alias, native[source])
    return NativeCoupledContext(_native_step_context(native, mesh), projection)


def _compact_coupled_dycore_phase(phase: _CoupledDycorePhase, *, mesh) -> _CoupledDycorePhase:
    """Discard atmospheric diagnostics at the post-dynamics phase.

    Physics retains its unfinished group's inputs. Statein's atmospheric
    columns and Stateout are not continuation inputs: the projection replaces them.
    """
    from mpas_pytorch.atmosphere_model import NATIVE_STATE_KEYS, _native_step_context

    if phase.statein is None:
        raise ValueError("phase is already compact")
    context = phase.native_context
    if context is None:
        context = _native_step_context(phase.native_state, mesh)
        pressure_padding = phase.native_state["surface_pressure"][context.support["nCells"]:]
        torch._assert_async(
            ((pressure_padding == 0) & ~torch.signbit(pressure_padding)).all(),
            "native coupled phase requires positive-zero interface-pressure padding")
    elif set(phase.native_state) != set(NATIVE_STATE_KEYS):
        raise ValueError("native dynamics output must contain exactly five fields")
    support = {name: getattr(phase.statein, name) for name in _PROJECTION_SUPPORT_KEYS}
    support["lightning_threat"] = phase.statein.wgrs is not None
    return replace(phase,
        native_state={name: phase.native_state[name] for name in NATIVE_STATE_KEYS},
        statein=None, prefix=replace(phase.prefix, stateout=None,
            interstitial={name: phase.prefix.interstitial[name] for name in _POST_DYNAMICS_WORK_KEYS}),
        native_context=context, projection_support=support)


def _materialize_coupled_dycore_phase(phase: _CoupledDycorePhase, *, mesh, config,
    output_fields=None) -> _CoupledDycorePhase:
    """Decode the phase without a previous atmospheric state or head observation record."""
    from mpas_pytorch.atmosphere_model import _materialize_native_step
    from mpas_pytorch.constants import UFS_CONSTANTS
    from mpas_pytorch.diagnostics import (
        physics_interface_surface_pressure,
        refresh_output_diagnostics,
    )

    if phase.native_context is None:
        return phase
    support = phase.projection_support
    if (phase.statein is not None or support is None
        or set(support) != set(_PROJECTION_SUPPORT_KEYS) | {"lightning_threat"}):
        raise ValueError("compact phase has invalid projection support")
    reference = phase.native_state["rho_p"]
    statein = Statein.allocate(ncol=support["smc"].shape[0], nlev=reference.shape[1],
        ntrac=phase.native_state["scalars"].shape[2], soil_levels=support["smc"].shape[1],
        lightning_threat=support["lightning_threat"], device=reference.device, dtype=reference.dtype)
    statein = replace(statein, **{name: support[name] for name in _PROJECTION_SUPPORT_KEYS})
    forward_mode = torch.autograd.forward_ad._current_level >= 0
    step_config = dycore_step_config(config, mesh,
        differentiable=torch.is_grad_enabled() or forward_mode, forward_mode=forward_mode)
    native = _materialize_native_step(phase.native_state, phase.native_context,
        mesh, step_config, config["config_dt"])
    native = {**(output_fields or {}), **mesh, **native}
    # The tail overwrites active pressure before every consumer. No held value is state.
    native["surface_pressure"] = physics_interface_surface_pressure(
        nCells=native["nCells"], rho_zz=native["rho_zz"], zz=native["zz"],
        qv=native["scalars"][..., r4_tracer_contract().native_qv_index],
        pressure_p=native["pressure_p"], pressure_base=native["pressure_base"],
        zgrid=native["zgrid"], gravity=UFS_CONSTANTS.G)
    if output_fields is not None:
        from .commit import NATIVE_SURFACE_UPDATE_KEYS

        native.update({name: getattr(phase.native_surface, name)
                       for name in NATIVE_SURFACE_UPDATE_KEYS})
    refresh_output_diagnostics(native, config)
    mark_complete_step_boundary(native, mesh=mesh, config=config, dt=config["config_dt"])
    return replace(phase, native_state=native, statein=statein, native_context=None,
        projection_support=None)


def mark_cold_boundary(native_state, *, mesh, config, dt) -> None:
    """``mark_complete_step_boundary`` under the runtime policy's boundary guard (``set_boundary_guard``),
    as the cycle binds its own boundaries: in "stamps" mode no byte copy of the state and mesh is kept."""
    with boundary_guard(boundary_guard_mode()):
        mark_complete_step_boundary(native_state, mesh=mesh, config=config, dt=dt)


def initialize_cold_cycle(native_state, mesh, *, config, surface_control: SurfaceControl,
    land_tables: LandInitializationTables, zs: torch.Tensor,
    tempo_tables: TorchFrozenTables, tempo_semi_sedi: bool,
    initial_date: tuple[int, ...], nphy_f2d: int, ndtend: int,
    n_cells_global: int, mpi_size: int, h2o_diagnostics: bool = False) -> ColdCycleInitialization:
    """Build selected cold CCPP carry from already initialized MPAS buffers.

    Tables and static controls come from offline preparation, never captures.
    Native MPAS initialization and table preparation precede this call. It runs
    the selected Noah surface and TEMPO state initializers, not all physics_init
    routines. This cold-only path must not reset continuing or restarted state.
    """
    if _UFS_COMMIT_ENTRY in native_state:
        raise ValueError("cold cycle cannot reset a coupled commit; use continuation")
    if (not isinstance(initial_date, tuple) or len(initial_date) != 8
        or any(type(v) is not int for v in initial_date)):
        raise ValueError("cold cycle requires eight static NCEP date integers")
    if surface_control.jdat_month != initial_date[1]:
        raise ValueError("cold surface month disagrees with initial_date")
    device = native_state["rho_zz"].device
    date = torch.tensor(initial_date, dtype=torch.int32, device=device)
    _, yearlen = prepare_physics_calendar(date, torch.tensor(365, dtype=torch.int32, device=device))
    _, slag, sdec, cdec = _solar_state(date, con_pi=torch.pi, device=device)
    soil = SoilParameters(land_tables.bexp, land_tables.smcmax, land_tables.psisat)
    bridge = initialize_bridge(native_state, mesh, config=config,
        surface_control=surface_control, soil=soil,
        n_cells_global=n_cells_global, mpi_size=mpi_size)
    land = initialize_noahmp_cold_surface(bridge.sfcprop, land_tables,
        month=initial_date[1], zs=zs, dzs=surface_control.dzs,
        n_cells_solve=surface_control.n_cells_solve)
    surface = initialize_surface(bridge.native_surface, bridge.statein, land,
        bridge.radiation_albedo, replace(surface_control, initialize_noahmp_liquid=True), soil).sfcprop
    tempo = mp_tempo_init(spechum=bridge.statein.qgrs[:,:,0],tables=tempo_tables,
        semi_sedi=tempo_semi_sedi)
    # The host reprojects native MPAS after physics_init; its transient q repair
    # does not change native tracers or the final cold Statein.
    state = allocate_cold_cycle_state(bridge, surface, nphy_f2d=nphy_f2d,
        ndtend=ndtend, yearlen=yearlen, h2o_diagnostics=h2o_diagnostics,
        # Held as the radiation setup stage holds them: in the state dtype (the Julian-date arithmetic is float64).
        solar={n: v.to(bridge.statein.tgrs.dtype) for n, v in (("slag", slag), ("sdec", sdec), ("cdec", cdec))})
    return ColdCycleInitialization(bridge, state, tempo)


def allocate_cold_cycle_state(bridge, surface, *, nphy_f2d: int, ndtend: int,
    yearlen: torch.Tensor, solar: Mapping[str, torch.Tensor], h2o_diagnostics: bool = False) -> CoupledCycleState:
    """Allocate cold carry after native bridge and Noah-MP surface initialization.

    The caller supplies source allocation counts and initialized surface/solar
    controls. This does not initialize schemes or authenticate a native boundary.
    Never use this allocator for an ordinary continuation or restart.
    """
    statein = bridge.statein
    ncol, nlev, ntrac = statein.qgrs.shape
    device, dtype = statein.tgrs.device, statein.tgrs.dtype
    if type(nphy_f2d) is not int or nphy_f2d < 1:
        raise ValueError("cold cycle requires a positive static nphy_f2d")
    if type(ndtend) is not int or ndtend < 0:
        raise ValueError("cold cycle requires a nonnegative static ndtend (0: ldiag3d off)")
    if dtype not in WORKING_DTYPES or ntrac != 10:
        raise ValueError("cold cycle requires a float32/float64 R4 ten-tracer Statein")
    if (yearlen.shape != () or yearlen.dtype != torch.int32 or yearlen.device != device
        or surface.tsfc.shape != (ncol,) or surface.tsfc.dtype != dtype
        or surface.tsfc.device != device):
        raise ValueError("cold cycle requires device-local calendar and initialized surface")
    if set(solar) != {"slag", "sdec", "cdec"} or any(
        v.shape != () or v.dtype != dtype or v.device != device for v in solar.values()
    ):
        raise ValueError("cold cycle requires three device-local solar controls in the state dtype")
    carry = RadiationCarry.allocate_r4(ncol=ncol,nlev=nlev,device=device,ndtend=ndtend,h2o_diagnostics=h2o_diagnostics,
                                       dtype=dtype)
    prefix = PhysicsPrefixResult(bridge.stateout,surface,carry,
        torch.zeros((ncol,nphy_f2d),dtype=dtype,device=device),{})
    return CoupledCycleState(bridge.native_state,bridge.native_surface,statein,prefix,
        torch.zeros(ncol,dtype=dtype,device=device),torch.zeros_like(statein.prsl),
        ConvectiveCloudHistory.allocate_r4(ncol=ncol,device=device,dtype=dtype),
        RadiationSeeds.allocate(ncol=ncol,nrstreams=2,lrseeds=True,device=device),
        solar,yearlen,CumulativeLimiter.allocate(ntrac=ntrac,device=device,dtype=dtype),
        torch.zeros((),dtype=torch.float64,device=device),torch.zeros((),dtype=torch.int32,device=device),1)


def step_variant(step_index: int, *, clock_control, radiation_reset, **_unused) -> tuple[bool, ...]:
    """The static key of a coupled step: (SW step, LW step, radiation reset, physics reset, first).

    Steps with one key run the same kernels with the same static values (the cycle reads the step
    index only through this key), so one captured step graph replays every step of its key.
    """
    k = step_index
    sw, lw = radiation_triggers(k, clock_control=clock_control)
    return (sw, lw, radiation_reset_step(k, **radiation_reset), physics_reset_step(k, nszero=radiation_reset["nszero"]),
            k == 1)


def radiation_triggers(step_index: int, *, clock_control) -> tuple[bool, bool]:
    """Static (SW, LW) radiation calls of a step, as the device clock's lsswr/lslwr."""
    k = step_index
    sw = k % clock_control["nsswr"] == 1 or clock_control["nsswr"] == 1 or k <= clock_control["nhfrad"]
    lw = k % clock_control["nslwr"] == 1 or clock_control["nslwr"] == 1 or k <= clock_control["nhfrad"]
    return sw, lw


@contextlib.contextmanager
def _cycle_context(rungs: tuple, deterministic: bool, chunk: int | None, eager: bool,
                   unchunked: frozenset[str] | None = None):
    """A checkpointed cycle's conditions, in its forward and its recompute: the compaction rungs of the
    forward, column chunks and eager mode, no nested dycore checkpointing or saved-tensor offload,
    and deterministic algorithms when asked (CUDA); all restored on exit."""
    replaced = restore_compaction_state(rungs)
    previous_chunk = runtime_policy.set_column_chunk(chunk)
    previous_unchunked = runtime_policy.set_unchunked_schemes(
        runtime_policy.unchunked_schemes() if unchunked is None else unchunked)
    token = runtime_policy._INSIDE_CYCLE_CHECKPOINT.set(True)
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with eager_regions(eager):
            yield
    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=warn_only)
        runtime_policy._INSIDE_CYCLE_CHECKPOINT.reset(token)
        runtime_policy.set_column_chunk(previous_chunk)
        runtime_policy.set_unchunked_schemes(previous_unchunked)
        restore_compaction_state(replaced)


def require_finite_state(state: "CoupledCycleState") -> None:
    """Device check that every floating tensor of a cycle's output state is finite (no host read).

    Compiled regions drop the schemes' own ``torch._assert_async`` checks by default
    (``compile_policy.set_compiled_asserts``); this one check per kept cycle runs outside them.
    """
    from torch.autograd import forward_ad
    from .step_graphs import flatten_state
    with torch.no_grad():
        # Inspect only the primal. foreach_norm's Torch 2.8 JVP has the wrong scalar shape;
        # the check must not compute derivatives or change the state's dual tensors.
        values = [forward_ad.unpack_dual(value).primal for value in flatten_state(state)[0].values()
                  if isinstance(value, torch.Tensor) and value.is_floating_point() and value.numel()]
        if values:
            # The max-norm of each tensor is NaN or inf exactly when the tensor has a nonfinite element; one
            # read per tensor (0.036 s per x1.10242 state on 4 CPU threads, against 0.091 s for isfinite).
            finite = torch.isfinite(torch.stack(torch._foreach_norm(values, float("inf"))))
            torch._assert_async(finite.all(), "nonfinite coupled state after a cycle")


def _state_digest(state: "CoupledCycleState") -> torch.Tensor:
    """Wrapping int64 sum of the bit patterns of every floating state tensor (order-independent)."""
    from .step_graphs import flatten_state
    with torch.no_grad():
        total = torch.zeros((), dtype=torch.int64, device=state.sec.device)
        for value in flatten_state(state)[0].values():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and value.numel():
                bits = {8: torch.int64, 4: torch.int32}[value.element_size()]
                total = total + value.contiguous().view(bits).sum(dtype=torch.int64)
    return total


def _tensor_leaves(value: Any) -> dict[str, torch.Tensor]:
    """Every tensor reachable through dataclass fields, mappings, sequences and object attributes, by path."""
    leaves: dict[str, torch.Tensor] = {}
    seen: set[int] = set()

    def visit(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            leaves[path] = item
        elif id(item) not in seen:
            seen.add(id(item))
            if is_dataclass(item) and not isinstance(item, type):
                for field in fields(item):
                    visit(getattr(item, field.name), f"{path}/{field.name}")
            elif isinstance(item, Mapping):
                for key, entry in item.items():
                    visit(entry, f"{path}/{key}")
            elif isinstance(item, (tuple, list)):
                for index, entry in enumerate(item):
                    visit(entry, f"{path}/{index}")
            elif hasattr(item, "__dict__") and not isinstance(item, (type, types.ModuleType)) and not callable(item):
                for name, entry in vars(item).items():
                    visit(entry, f"{path}/{name}")

    visit(value, "")
    return leaves


def _differentiable_tensors(value: Any, what: str) -> list[torch.Tensor]:
    """Distinct tensors that require grad in the cycle state or run arguments (mappings, sequences, dataclasses),
    which a recompute can replace; one held in another kind of object cannot be replaced and raises."""
    found: dict[int, torch.Tensor] = {}
    for path, tensor in _tensor_leaves(value).items():
        if tensor.requires_grad:
            found.setdefault(id(tensor), tensor)
    replaceable = _tensor_leaves_replaceable(value)
    hidden = [path for path, t in _tensor_leaves(value).items() if t.requires_grad and id(t) not in replaceable]
    if hidden:
        raise TypeError(f"differentiable cycle {what} inside objects a recompute cannot rebuild: {hidden[:5]}")
    return list(found.values())


def _tensor_leaves_replaceable(value: Any) -> set[int]:
    """ids of the tensors reachable through mappings, sequences and dataclass init fields only (what ``_replaced``
    rebuilds)."""
    ids: set[int] = set()
    seen: set[int] = set()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            ids.add(id(item))
        elif id(item) not in seen:
            seen.add(id(item))
            if is_dataclass(item) and not isinstance(item, type):
                for field in fields(item):
                    if field.init:
                        visit(getattr(item, field.name))
            elif isinstance(item, Mapping):
                for entry in item.values():
                    visit(entry)
            elif isinstance(item, (tuple, list)):
                for entry in item:
                    visit(entry)

    visit(value)
    return ids


def _replaced(value: Any, swap: Mapping[int, torch.Tensor]) -> Any:
    """``value`` with each tensor whose id is in ``swap`` replaced; containers on no replaced path stay the same object.
    A boundary stamp of a replaced tensor binds the replacement (``swap`` keeps the storage and version counter)."""
    if isinstance(value, torch.Tensor):
        return swap.get(id(value), value)
    if isinstance(value, _TensorStamp):
        referent = value.ref()
        return value if referent is None or id(referent) not in swap else replace(
            value, ref=weakref.ref(swap[id(referent)]))
    if is_dataclass(value) and not isinstance(value, type):
        changes = {f.name: _replaced(getattr(value, f.name), swap) for f in fields(value) if f.init}
        changed = {k: v for k, v in changes.items() if v is not getattr(value, k)}
        return replace(value, **changed) if changed else value
    if isinstance(value, Mapping):
        items = {k: _replaced(v, swap) for k, v in value.items()}
        return type(value)(items) if any(items[k] is not v for k, v in value.items()) else value
    if isinstance(value, (tuple, list)):
        items = [_replaced(v, swap) for v in value]
        if all(a is b for a, b in zip(items, value)):
            return value
        return type(value)(*items) if hasattr(value, "_fields") else type(value)(items)
    return value


def _sequence_mark() -> int:
    """The autograd node sequence number now: nodes created later on this thread have larger numbers."""
    return (torch.zeros((), requires_grad=True) * 1).grad_fn._sequence_nr()


def _require_known_inputs(outputs, inputs, mark: int) -> None:
    """Raise if ``outputs`` reach graph created before ``mark`` other than through ``inputs``: a differentiable
    tensor the cycle used that is neither in its state nor in its arguments (a cache, a closure). The recompute
    node would differentiate through that upstream graph a second time, or drop the path."""
    stops = {(t.grad_fn, t.output_nr) for t in inputs if t.grad_fn is not None}
    known = {id(t) for t in inputs}
    seen, stack, unknown = set(), [t.grad_fn for t in outputs], []
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for child, index in node.next_functions:
            if child is None or (child, index) in stops:
                continue
            variable = getattr(child, "variable", None)  # AccumulateGrad
            if variable is not None:
                if id(variable) not in known:
                    unknown.append(f"leaf {tuple(variable.shape)}")
            elif child._sequence_nr() < mark:
                unknown.append(f"{child.name()} (earlier graph)")
            else:
                stack.append(child)
    if unknown:
        raise RuntimeError(f"a cycle depends on differentiable tensors outside its state and arguments: {unknown[:5]}")


def _rng_state(device: torch.device):
    return torch.get_rng_state(), (torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


@contextlib.contextmanager
def _rng_restored(state, device: torch.device):
    cpu, cuda = state
    with torch.random.fork_rng(devices=[device] if cuda is not None else []):
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, device)
        yield


@dataclass
class _RecomputePlan:
    body: Callable[..., Any]
    previous: Any
    arguments: Mapping[str, Any]
    paths: tuple[str, ...]
    rng: tuple
    outputs: tuple[torch.Tensor, ...] | None


def _recompute_leaves(inputs) -> dict[int, torch.Tensor]:
    """Leaves aliasing ``inputs`` (by id) for a recompute's inner backward. The detach aliases storage and does not
    cut the model graph: ``_CycleRecompute.backward`` returns each leaf's gradient to its input. The host-read trace
    exempts this one function, as it exempts ``torch.utils.checkpoint`` storage."""
    leaves = {}
    for tensor in inputs:
        leaves[id(tensor)] = tensor.detach().requires_grad_(True)
    return leaves


class _CycleRecompute(torch.autograd.Function):
    """One graph node for a computed cycle: its outputs keep their objects and take this node as
    ``grad_fn``; backward recomputes the cycle from the plan's state and differentiates it.

    Inputs: the differentiable tensors of the input state and of the run arguments. The recompute replaces
    each by a local leaf (and its boundary stamps by the leaf's), so its inner backward ends at those leaves:
    it never runs graph upstream of an input (an input state from a cycle with a full graph or from a
    differentiable constructor, whose tensors may depend on each other; parameters computed from other
    leaves; an earlier recompute node), which the outer backward reaches once through this node's inputs.
    """

    @staticmethod
    def forward(ctx, plan: _RecomputePlan, *inputs):
        ctx.set_materialize_grads(False)
        ctx.plan, ctx.inputs = plan, inputs
        return plan.outputs

    @staticmethod
    def backward(ctx, *grads):
        plan = ctx.plan
        wanted = [(path, grad) for path, grad in zip(plan.paths, grads) if grad is not None]
        if not wanted:
            return (None,) * (1 + len(ctx.inputs))
        local = _recompute_leaves(ctx.inputs)
        with _rng_restored(plan.rng, plan.previous.sec.device), torch.enable_grad():
            recomputed = _tensor_leaves(plan.body(_replaced(plan.previous, local), _replaced(plan.arguments, local)))
        outputs = [recomputed[path] for path, _ in wanted]
        inner = [local[id(t)] for t in ctx.inputs]
        grads_in = torch.autograd.grad(outputs, inner, [grad for _, grad in wanted], allow_unused=True)
        return (None, *grads_in)


def _checkpointed_cycle(previous, arguments: Mapping[str, Any], *, operator=None, state_of=None):
    """One cycle whose graph backward recomputes from ``previous`` (``set_cycle_checkpoint``).

    The cycle runs under grad, then its graph is cut to one ``_CycleRecompute`` node, so until
    backward the cycle holds ``previous`` and that node. (A non-reentrant ``torch.utils.checkpoint``
    drops the saved tensors but keeps every graph node: about 1.7 GiB per x1.10242 FP32 cycle,
    against 0.17 GiB of new state.) The node's inputs are the differentiable tensors of ``previous``
    and of ``arguments`` (the parameters); a graph walk raises if the cycle reaches any other
    differentiable leaf, so no gradient path is dropped.
    Native-phase callers supply their source operator and evolving-state selector;
    capacity, RNG, policy restoration and input-leaf ownership use the same implementation.
    """
    rungs = compaction_state()
    chunk, eager = column_chunk(), eager_regions_active()
    unchunked = runtime_policy.unchunked_schemes()
    deterministic = previous.sec.device.type == "cuda"
    forward_digest = []
    operator = run_coupled_cycle if operator is None else operator
    state_of = _completed_cycle_state if state_of is None else state_of

    def body(state, run_arguments):
        # Each call (forward and every backward) restores the forward's conditions.
        with _cycle_context(rungs, deterministic, chunk, eager, unchunked):
            result = operator(state, **run_arguments)
            digest = _state_digest(state_of(result))
            if forward_digest:
                torch._assert_async(digest == forward_digest[0], "recomputed coupled cycle differs from its forward")
            else:
                forward_digest.append(digest)
            return result

    rng = _rng_state(previous.sec.device)
    mark = _sequence_mark()
    result = body(previous, arguments)
    inside = {id(t): t for t in _differentiable_tensors(previous, "state")}
    differentiable = tuple(t for t in _differentiable_tensors(arguments, "arguments") if id(t) not in inside)
    inputs = [*inside.values(), *differentiable]
    created = {}
    for path, tensor in _tensor_leaves(result).items():
        if tensor.requires_grad and tensor.grad_fn is not None and id(tensor) not in inside:
            created.setdefault(id(tensor), (path, tensor))
    if created:
        paths, outputs = zip(*created.values())
        _require_known_inputs(outputs, inputs, mark)
        plan = _RecomputePlan(body, previous, arguments, paths, rng, outputs)
        _CycleRecompute.apply(plan, *inputs)
        plan.outputs = None
    return result


def _bound_in_policy_guard_mode(fn):
    """Run a cycle with the dycore's live boundaries bound in the runtime policy's guard mode,
    and with its saved tensors in host memory when the policy offloads them (reverse mode only).

    ``functools.wraps`` keeps the cycle's signature (the restart reads its argument set).
    """
    @functools.wraps(fn)
    def call(previous, **arguments):
        offload = (torch.autograd.graph.save_on_cpu(pin_memory=previous.sec.device.type == "cuda")
                   if torch.is_grad_enabled() and saved_tensor_offload_enabled() else contextlib.nullcontext())
        with boundary_guard(boundary_guard_mode()), offload, partition_scope((arguments.get("config") or {}).get("_halo_exchanger")):
            return fn(previous, **arguments)
    return call


def _metered(fn=None, *, part=None):
    """Time the step in ``cost_meter`` when the meter is on (kind first, radiation or ordinary); off, a direct call."""
    if fn is None:
        return functools.partial(_metered, part=part)
    @functools.wraps(fn)
    def call(previous, **arguments):
        if cost_meter.METER is None:
            return fn(previous, **arguments)
        k = previous.step_index
        kind = lambda: "first" if k == 1 else ("radiation" if any(radiation_triggers(  # noqa: E731
            k, clock_control=arguments["clock_control"])) else "ordinary")
        with cost_meter.step(k, kind, part=part):
            return fn(previous, **arguments)
    return call


def _first_step_eager(fn):
    """Avoid compiled variants used only by the cold step, including checkpoint recomputation."""
    @functools.wraps(fn)
    def call(previous, **arguments):
        with eager_regions() if previous.step_index == 1 else contextlib.nullcontext():
            return fn(previous, **arguments)
    return call


def _run_coupled_cycle_to_dycore(previous: CoupledCycleState, *, mesh, config, domain, grid,
    initial_date, global_ids, n_cells_global, clock_control, seed_control,
    radiation_config, radiation_tables, radiation_reset, diagnostic_reset, prefix_control,
    noah_case, physics_arguments, microphysics_arguments,
    ozone_table, ozone_control, crtrh, rhcmax, water_species,
    _native_boundary_output=False) -> _CoupledCycleHead:
    """Run the source cycle through dynamics, retaining only explicit tail inputs."""
    if type(previous.step_index) is not int or previous.step_index < 1:
        raise ValueError("coupled cycle requires a positive static step_index")
    dtp = physics_arguments["dtp"]
    ncol = previous.statein.tgrs.shape[0]
    # The clock (elapsed seconds, forecast hour, calendar) is float64 in every working dtype.
    clock = glue(prepare_timestep_clock)(previous.sec, dtp.to(torch.float64), **clock_control)
    k = previous.step_index
    sw, lw, radiation_reset_now, physics_reset_now, first = step_variant(
        k, clock_control=clock_control, radiation_reset=radiation_reset)
    if step_graphs_enabled():
        # One captured graph replays every step of its variant, so the checks hold no step number:
        # the carried device step counter and the variant the static step index selected.
        torch._assert_async((clock.kdt == previous.completed_dycore_steps + 1) & (clock.lsswr == sw)
            & (clock.lslwr == lw) & (clock.first_time_step == first),
            "coupled static step variant disagrees with the device clock")
    else:
        torch._assert_async(clock.kdt == previous.step_index,
            "coupled static step_index disagrees with the device clock")
    observe = cycle_observers_enabled()
    if previous.step_index > 1 and observe:
        require_coupled_boundary(previous)
    date = glue(advance_utc_date)(initial_date, clock.phour)
    julian, yearlen = glue(prepare_physics_calendar)(date, previous.yearlen)
    # The seed streams change only on radiation steps (static sw/lw equal the device lsswr/lslwr
    # of this step); other steps keep the carried seeds, including the unread rseeds draw. The
    # step's draws come from the stream prepared ahead (radiation_seed_stream), chosen by clock.sec.
    seeds = glue(advance_mpas_radiation_seeds)(previous.seeds, global_ids,
        n_cells_solve=ncol, n_cells_global=n_cells_global, sec=clock.sec,
        ipsd0=_seed_offset(initial_date),
        lsswr=clock.lsswr, lslwr=clock.lslwr, **seed_control,
        draws=radiation_seed_stream(previous.sec, k, ncol=ncol, initial_date=initial_date, global_ids=global_ids,
            n_cells_global=n_cells_global, clock_control=clock_control, seed_control=seed_control,
            physics_arguments=physics_arguments).select(clock.sec)) if sw or lw else previous.seeds
    # The static reset decisions, not the step number (checked against kdt above), key the compiled
    # resets: two variants each, not one per step.
    carry = glue(reset_radiation_diagnostics)(previous.prefix.carry, clock.kdt, **radiation_reset,
        reset=radiation_reset_now)
    carry = glue(reset_physics_diagnostics)(carry,clock.kdt,
        nszero=radiation_reset["nszero"],**diagnostic_reset,reset=physics_reset_now)
    reservoirs = {}
    if observe:
        reservoirs["after_timestep_init_before_physics"] = physical_reservoirs(
            NativeAtmosphere.from_mpas_state(previous.native_state, n_cells_solve=ncol),
            previous.prefix.sfcprop, carry.intdiag, area_cell=mesh["areaCell"], rdzw=mesh["rdzw"], dtp=dtp)
    projection = glue(project_atmosphere)(
        NativeAtmosphere.from_mpas_state(previous.native_state, n_cells_solve=ncol),
        ProjectionMesh.from_mpas_state({**mesh, **previous.native_state}),
        previous.statein, tracers=r4_tracer_contract())
    statein = projection.statein
    # Radiation owns its tensor regions independently of the column-memory policy.
    # GFS_phys_time_vary interpolates ozone and the MERRA-2 aerosol at fhour, not the radiation/Noah phour.
    forcing_date = glue(advance_utc_date)(initial_date, clock.fhour)
    radiation = _run_capacity_region(run_native_radiation, statein, grid, previous.prefix.sfcprop, carry, seeds,
        replace(radiation_config, jdat=date, solhr=clock.solhr, lsswr=sw, lslwr=lw,
                aer_jdat=forcing_date if radiation_config.iaerflg else None),
        clock=clock, tables=radiation_tables, capacity_check=sw or lw)
    # The original setup updates these held controls only on SW-active calls.
    solar = ({n: radiation.stages["setup"][n] for n in ("slag", "sdec", "cdec")}
             if sw else previous.solar)
    entry = glue(prepare_physics_prefix)(statein, radiation.stateout, grid,
        radiation.sfcprop, radiation.carry, previous.prefix.phy_f2d,
        {**solar, "solhr": clock.solhr}, physics_arguments["dtidx"], prefix_control)
    # UGWP v1 reads the tau file at the GFS_phys_time_vary forecast day (as the ozone forcing).
    v1_day = ({"forecast_day": ozone_forecast_day(forcing_date)}
              if physics_arguments["gwd_control"].get("do_ugwp_v1", False) else {})
    physics = _run_capacity_region(run_physics_sequence, statein, grid, entry, previous.hpbl, noah_case,
        **{**physics_arguments, "solhr": clock.solhr, "julian": julian,
           "yearlen": yearlen.to(dtype=dtp.dtype), **v1_day})
    dynamics = advance_physics_dynamics(previous.native_state, previous.native_surface,
        statein, physics.prefix, mesh=mesh, config=config, domain=domain,
        tracers=r4_tracer_contract(), water_species=water_species,
        cumulative=previous.cumulative, dtp=dtp, cap_step_index=clock.kdt,
        completed_dycore_steps=previous.completed_dycore_steps,
        **({"_native_boundary_output": True} if _native_boundary_output else {}))
    if observe:
        reservoirs["after_phase1_radiation_physics_commit"] = physical_reservoirs(
            dynamics.native_commit.state.atmosphere, physics.prefix.sfcprop, physics.prefix.carry.intdiag,
            area_cell=mesh["areaCell"], rdzw=mesh["rdzw"], dtp=dtp)
        reservoirs["after_dycore_before_microphysics"] = physical_reservoirs(
            NativeAtmosphere.from_mpas_state(dynamics.native_state, n_cells_solve=ncol),
            physics.prefix.sfcprop, physics.prefix.carry.intdiag,
            area_cell=mesh["areaCell"], rdzw=mesh["rdzw"], dtp=dtp)
    phase = _CoupledDycorePhase(dynamics.native_state, dynamics.native_commit.state.surface,
        statein, physics.prefix, physics.hpbl, previous.ud_mf, previous.cloud_history,
        seeds, solar, yearlen, dynamics.native_commit.state.cumulative,
        previous.sec, dynamics.completed_dycore_steps, k, clock, forcing_date)
    return _CoupledCycleHead(phase, radiation, entry, physics, dynamics, reservoirs)


def _run_coupled_cycle_from_dycore(phase: _CoupledDycorePhase, *, mesh, config, domain, grid,
    clock_control, radiation_config, physics_arguments, microphysics_arguments,
    ozone_table, ozone_control, crtrh, rhcmax, water_species, _output_fields=None, **_unused) -> _CoupledCycleTail:
    """Finish the source cycle without the head's outputs or previous atmospheric state."""
    phase = _materialize_coupled_dycore_phase(phase, mesh=mesh, config=config, output_fields=_output_fields)
    dtp = physics_arguments["dtp"]
    device = dtp.device
    ncol = phase.statein.tgrs.shape[0]
    k, clock = phase.step_index, phase.clock
    first = k == 1
    observe = cycle_observers_enabled()
    reservoirs = {}
    ozone_date = phase.forcing_date
    forecast_day = ozone_forecast_day(ozone_date)
    forcing = glue(update_ozone_forcing)(_static_ozone_grid(grid, ozone_table),
        ozone_table, forecast_day)
    # h2o_phys (GFS v17): the h2o table rides in the photochemistry control (``h2o_table``) so that
    # the cycle's argument set (and the restart that reads it) is unchanged.
    photochemistry_control = {n: v for n, v in ozone_control.items() if n != "h2o_table"}
    h2o = {}
    if photochemistry_control["h2o_phys"]:
        h2o_table = ozone_control["h2o_table"]
        h2o = dict(h2o_log_pressure=h2o_table.po3, h2o_forcing=glue(update_h2o_forcing)(
            _static_h2o_weights(grid, h2o_table), h2o_table, forecast_day, ozone=ozone_table))
    micro_projection, micro_entry = glue(prepare_microphysics_prefix)(phase.native_state,
        mesh, phase.statein, grid, phase.prefix, log_pressure=ozone_table.po3,
        forcing=forcing, dtp=dtp,
        control={**photochemistry_control,"first_time_step":first}, crtrh=crtrh, rhcmax=rhcmax, **h2o)
    micro = _run_capacity_region(run_microphysics_sequence, micro_projection.statein, grid, micro_entry,
        **{**microphysics_arguments, "hpbl": phase.hpbl, "ud_mf": phase.ud_mf,
           "cloud_history": phase.cloud_history, "kdt": clock.kdt,
           "convection_control": {**microphysics_arguments["convection_control"],
               "first_time_step": first},
           # TEMPO reads its call index only as "first call or not" (mp_tempo.py is_first_step);
           # one value for all later calls keeps compiled regions from recompiling every cycle.
           "tempo_control": {**microphysics_arguments["tempo_control"], "tempo_call_index": min(k, 2)},
           "clstp": convective_cloud_clock(clock.kdt, clock.lsswr, clock.fhour,
               nsswr=clock_control["nsswr"], fhswr=dtp.new_full((), radiation_config.fhswr))})
    final = commit_microphysics(phase.native_state, phase.native_surface,
        micro_projection.statein, micro.prefix, mesh=mesh, config=config, domain=domain,
        tracers=r4_tracer_contract(), water_species=water_species,
        cumulative=phase.cumulative,
        group_states=torch.full((4,), 2, dtype=torch.int32, device=device))
    if observe:
        reservoirs["after_full_step_coupled"] = physical_reservoirs(
            NativeAtmosphere.from_mpas_state(final.native_state, n_cells_solve=ncol),
            micro.prefix.sfcprop, micro.prefix.carry.intdiag,
            area_cell=mesh["areaCell"], rdzw=mesh["rdzw"], dtp=dtp)
    state = CoupledCycleState(final.native_state, final.native_commit.state.surface,
        micro_projection.statein, replace(micro.prefix, stateout=None, interstitial={}), phase.hpbl, micro.ud_mf,
        micro.cloud_history, phase.seeds, phase.solar, phase.yearlen, final.native_commit.state.cumulative,
        phase.sec + dtp, phase.completed_dycore_steps, k + 1)
    if observe:
        state = bind_coupled_boundary(state)
    return _CoupledCycleTail(state, micro_projection, micro_entry, micro, final, reservoirs)


@_metered
@_first_step_eager
@_bound_in_policy_guard_mode
def run_coupled_cycle(previous: CoupledCycleState, *, mesh, config, domain, grid,
    initial_date, global_ids, n_cells_global, clock_control, seed_control,
    radiation_config, radiation_tables, radiation_reset, diagnostic_reset, prefix_control,
    noah_case, physics_arguments, microphysics_arguments,
    ozone_table, ozone_control, crtrh, rhcmax, water_species) -> CoupledCycleResult:
    """Time-vary -> radiation -> physics -> MPAS -> microphysics -> phase2.

    The two phases use the same source operators, order and refresh schedule.
    The result retains observations separately from numerical continuation inputs.
    """
    arguments = dict(mesh=mesh, config=config, domain=domain, grid=grid,
        initial_date=initial_date, global_ids=global_ids, n_cells_global=n_cells_global,
        clock_control=clock_control, seed_control=seed_control,
        radiation_config=radiation_config, radiation_tables=radiation_tables,
        radiation_reset=radiation_reset, diagnostic_reset=diagnostic_reset,
        prefix_control=prefix_control, noah_case=noah_case, physics_arguments=physics_arguments,
        microphysics_arguments=microphysics_arguments, ozone_table=ozone_table,
        ozone_control=ozone_control, crtrh=crtrh, rhcmax=rhcmax, water_species=water_species)
    head = _run_coupled_cycle_to_dycore(previous, **arguments)
    tail = _run_coupled_cycle_from_dycore(head.phase, **arguments)
    return CoupledCycleResult(tail.state, head.phase.clock, head.phase.statein,
        head.radiation, head.physics_entry, head.physics, head.dynamics,
        tail.microphysics_projection, tail.microphysics_entry, tail.microphysics, tail.final,
        {**head.reservoirs, **tail.reservoirs})


@_metered(part="head")
@_first_step_eager
@_bound_in_policy_guard_mode
def _start_native_coupled_phase(previous: CoupledCycleState, **arguments) -> _CoupledDycorePhase:
    """Initial partial operator: physics and dynamics, before the first microphysics call."""
    scope = _CAPACITY_RETRY.get()
    native_output = not cycle_observers_enabled()
    context = (prepare_native_coupled_context(previous, mesh=arguments["mesh"]).dycore
               if native_output else None)
    token = _CAPACITY_RETRY.set((scope[0], previous.step_index, scope[2]) if scope else None)
    try:
        head = _run_coupled_cycle_to_dycore(previous,
            **({"_native_boundary_output": True} if native_output else {}), **arguments)
    finally:
        _CAPACITY_RETRY.reset(token)
    phase = replace(head.phase, native_context=context) if native_output else head.phase
    return _compact_coupled_dycore_phase(phase, mesh=arguments["mesh"])


@_metered(part="tail")
@_first_step_eager
@_bound_in_policy_guard_mode
def _finish_native_coupled_phase(previous: _CoupledDycorePhase, **arguments) -> _CoupledCycleTail:
    """Final partial operator: finish the pending source cycle without another dynamics call."""
    if previous.native_context is None or previous.statein is not None:
        raise ValueError("native coupled continuation requires a compact post-dynamics phase")
    return _run_coupled_cycle_from_dycore(previous, **arguments)


def _advance_native_coupled_phase(previous: _CoupledDycorePhase, *,
    _on_completed_cycle=None, _output_fields=None, **arguments) -> _CoupledDycorePhase:
    """One periodic native step: microphysics(n), then physics/dynamics(n+1)."""
    # The observation record must not keep microphysics work alive through the next head.
    outputs = {"_output_fields": _output_fields} if _on_completed_cycle is not None else {}
    state = _finish_native_coupled_phase(previous, **outputs, **arguments).state
    if _on_completed_cycle is not None:
        _on_completed_cycle(state)
    return _start_native_coupled_phase(state, **arguments)


def _finish_native_coupled_interval(previous: _CoupledDycorePhase, *, _output_fields,
    _on_completed_cycle=None, **arguments) -> CoupledCycleResult:
    """Finish the pending cycle, then retain the last full cycle's observation record.

    The checkpoint input remains compact. The completed intermediate state lives
    only inside this operator; no periodic checkpoint retains its diagnostics.
    The terminal output metadata excludes atmospheric state and computed work.
    """
    state = _finish_native_coupled_phase(previous, _output_fields=_output_fields, **arguments).state
    if _on_completed_cycle is not None:
        _on_completed_cycle(state)
    scope = _CAPACITY_RETRY.get()
    token = _CAPACITY_RETRY.set((scope[0], state.step_index, scope[2]) if scope else None)
    try:
        return run_coupled_cycle(state, **arguments)
    finally:
        _CAPACITY_RETRY.reset(token)


def _native_phase_state(phase: _CoupledDycorePhase) -> _CoupledDycorePhase:
    # The fixed topology includes slices, which the live step-state flattener rejects.
    # Check the complete evolving phase, not its external static context.
    return replace(phase, native_context=None)


def _completed_cycle_state(result):
    return result.state


def _run_native_coupled_start(previous: CoupledCycleState, *, on_overflow="rerun", **arguments):
    """Prepare the first five-field coupled boundary; no completed cycle is implied."""
    return _run_coupled_operation(previous, arguments, on_overflow=on_overflow,
        operator=_start_native_coupled_phase, state_of=_native_phase_state)


def _run_native_coupled_step(previous: _CoupledDycorePhase, *, on_overflow="rerun", **arguments):
    """Advance between two five-field coupled boundaries, with the existing replay policy."""
    return _run_coupled_operation(previous, arguments, on_overflow=on_overflow,
        operator=_advance_native_coupled_phase, state_of=_native_phase_state)


def _run_native_coupled_finish(previous: _CoupledDycorePhase, *, on_overflow="rerun", **arguments):
    """Return the completed-cycle terminal result, without retaining it in a periodic checkpoint."""
    return _run_coupled_operation(previous, arguments, on_overflow=on_overflow,
        operator=_finish_native_coupled_phase, state_of=_completed_cycle_state)


def start_native_coupled_phase(previous: CoupledCycleState, *, on_overflow="rerun", **arguments) -> NativeCoupledPhase:
    """Run physics and dynamics; stop before convection and microphysics."""
    return _run_native_coupled_start(previous, on_overflow=on_overflow, **arguments)


def advance_native_coupled_phase(previous: NativeCoupledPhase, *, on_overflow="rerun",
    on_completed_cycle=None, **arguments) -> NativeCoupledPhase:
    """Finish the pending cycle and advance to the next post-dynamics phase.

    A forward-only observer can consume the completed cycle before the next head.
    It does not become continuation state and never executes during AD replay.
    """
    if on_completed_cycle is not None:
        if not callable(on_completed_cycle):
            raise TypeError("on_completed_cycle must be callable")
        if torch.is_grad_enabled() or torch.autograd.forward_ad._current_level >= 0:
            raise ValueError("on_completed_cycle requires forward-only no-grad execution")
        arguments = {**arguments, "_on_completed_cycle": on_completed_cycle}
    return _run_native_coupled_step(previous, on_overflow=on_overflow, **arguments)


def finish_native_coupled_phase(previous: NativeCoupledPhase, *, on_overflow="rerun", **arguments) -> CoupledCycleState:
    """Finish the pending cycle without another dynamics call.

    The returned completed-cycle state is not a five-field restart boundary.
    Use run_coupled_interval when the full last-cycle observation record is needed.
    """
    return _run_native_coupled_finish(previous, on_overflow=on_overflow, **arguments).state


_OZONE_GRID: dict[tuple[int, int], tuple[Any, Any, Grid]] = {}
_H2O_WEIGHTS: dict[tuple[int, int], tuple[Any, Any, tuple]] = {}
# (step index, change) for every compaction buffer enlarged after an overflow, in run order.
COMPACTION_LOG: list[tuple[int, str]] = []
_SEED_STREAMS: dict[tuple, tuple[tuple, RadiationSeedStream]] = {}
# Radiation steps per prepared seed stream, and the byte budget of one draw's transients (about 32 bytes
# per drawn value: fewer rows on large meshes; 32 rows at x1.163842).
SEED_STREAM_ROWS, SEED_STREAM_BYTES = 32, 512 * 2**20
_SEED_STREAM_FROZEN = contextvars.ContextVar("seed_stream_frozen", default=False)


@contextlib.contextmanager
def frozen_seed_stream():
    """Steps run inside a step graph (recording, or the CPU emulator's replay) may not fill the seed
    stream: a fill inside a capture would be recorded, and a replay runs no Python. ``prepare_step``
    fills it before."""
    token = _SEED_STREAM_FROZEN.set(True)
    try:
        yield
    finally:
        _SEED_STREAM_FROZEN.reset(token)


def _seed_offset(initial_date: torch.Tensor) -> torch.Tensor:
    return radiation_seed_offset(initial_date[static_tensor((4, 1, 2, 0), dtype=torch.int64, device=initial_date.device)])


def radiation_seed_stream(sec: torch.Tensor, step_index: int, *, ncol: int, initial_date, global_ids,
    n_cells_global, clock_control, seed_control, physics_arguments, **_unused) -> RadiationSeedStream:
    """The seed draws of the radiation steps from ``step_index`` on, prepared once per run and window.

    The draws depend only on the elapsed seconds, the date and the static domain
    (``radiation_seed_draws``), so one batched draw serves ``SEED_STREAM_ROWS`` radiation steps (one
    serial Mersenne-twister pass for all rows; the shared permutation is 526 twist blocks at
    x1.163842). ``sec`` is this step's elapsed seconds; each row's seconds follow the cycle's own
    ``sec + dtp`` chain, so they equal the device clock bit for bit. A step outside the window refills
    the same tensors in place (recorded step graphs read them). Like the ozone grid, the stream is kept
    per run's static inputs (``global_ids``, ``initial_date``, ``dtp``, controls).
    """
    dtp = physics_arguments["dtp"]
    key = (id(global_ids), id(initial_date), id(dtp), ncol, n_cells_global, torch.is_inference_mode_enabled(),
           tuple(sorted(seed_control.items())), tuple(sorted(clock_control.items())))
    refs = (global_ids, initial_date, dtp)
    hit = _SEED_STREAMS.get(key)
    stream = hit[1] if hit is not None and all(r() is v for r, v in zip(hit[0], refs)) else None
    frozen = _SEED_STREAM_FROZEN.get()
    # Inside a step graph the row is chosen on the device by the step's seconds only, as at a replay
    # (the CPU emulator reruns a recorded step with the recording's step index); a missing row fails there.
    if stream is not None and (frozen or step_index in stream.steps):
        return stream
    if frozen:
        raise RuntimeError(f"radiation seed stream not prepared for step {step_index} before its step graph "
                           "(cycle.prepare_step)")
    count = 2 * (n_cells_global if seed_control["lrseeds"] else ncol)  # values drawn per row
    rows = max(1, min(SEED_STREAM_ROWS, SEED_STREAM_BYTES // (32 * count)))
    steps, secs, k, current = [], [], step_index, sec
    with torch.no_grad():
        while len(steps) < rows:
            if any(radiation_triggers(k, clock_control=clock_control)):
                steps.append(k)
                secs.append(current.reshape(()))
            current, k = current + dtp, k + 1
        secs = torch.stack(secs)
        active = seed_control["isubc_sw"] == 2 or seed_control["isubc_lw"] == 2
        draws = radiation_seed_draws(secs if seed_control["lrseeds"] or active else torch.zeros_like(secs),
            _seed_offset(initial_date), global_ids, n_cells_solve=ncol, n_cells_global=n_cells_global,
            ipsdlim=seed_control["ipsdlim"], lrseeds=seed_control["lrseeds"])
        if stream is None:
            stream = RadiationSeedStream(secs, draws, tuple(steps))
        else:  # in place: step graphs recorded on these tensors read the new rows
            stream.secs.copy_(secs)
            stream.draws.copy_(draws)
            stream = RadiationSeedStream(stream.secs, stream.draws, tuple(steps))
    for old in [k for k, (r, _) in _SEED_STREAMS.items() if any(ref() is None for ref in r)]:
        del _SEED_STREAMS[old]  # streams of runs whose static inputs are gone
    _SEED_STREAMS[key] = (tuple(weakref.ref(v) for v in refs), stream)
    return stream


def prepare_step(boundary: Mapping[str, Any], step_index: int, **arguments) -> None:
    """Per-run data the coupled step at ``step_index`` reads, prepared outside the step: the radiation
    seed stream on radiation steps. ``boundary`` is the step's input state flattened by path
    (``step_graphs.flatten_state``). ``run_coupled_cycle`` prepares the stream itself; the step-graph
    driver calls this before it records or replays a step (``frozen_seed_stream``)."""
    if any(radiation_triggers(step_index, clock_control=arguments["clock_control"])):
        radiation_seed_stream(boundary["/sec"], step_index, ncol=boundary["/statein/tgrs"].shape[0], **arguments)


def _static_ozone_grid(grid: Grid, ozone_table) -> Grid:
    """initialize_ozone_grid once per (grid, ozone table): both are static inputs of a run."""
    key = (id(grid), id(ozone_table))
    hit = _OZONE_GRID.get(key)
    if hit is None or hit[0]() is not grid or hit[1]() is not ozone_table:
        hit = _OZONE_GRID[key] = (weakref.ref(grid), weakref.ref(ozone_table), initialize_ozone_grid(grid, ozone_table))
    return hit[2]


def _static_h2o_weights(grid: Grid, h2o_table) -> tuple:
    """ty_h2ophys%setup (jindx1_h, jindx2_h, ddy_h) once per (grid, h2o table)."""
    key = (id(grid), id(h2o_table))
    hit = _H2O_WEIGHTS.get(key)
    if hit is None or hit[0]() is not grid or hit[1]() is not h2o_table:
        hit = _H2O_WEIGHTS[key] = (weakref.ref(grid), weakref.ref(h2o_table),
                                   photochemistry_latitude_weights(grid.xlat_d, h2o_table))
    return hit[2]


def _run_coupled_operation(previous, arguments, *, on_overflow, operator=None, state_of=None):
    """Share capacity recovery and recomputation across the two source cycle boundaries."""
    if on_overflow not in ("assert", "rerun"):
        raise ValueError("on_overflow must be 'assert' or 'rerun'")
    if operator is not None and step_graphs_enabled():
        raise ValueError("native coupled phases do not yet support captured step graphs")
    device = previous.sec.device
    checkpointed = torch.is_grad_enabled() and cycle_checkpoint_enabled()
    selected_state = _completed_cycle_state if state_of is None else state_of
    result = None
    while True:
        result = None
        reset_overflow(device)
        local_retry = on_overflow == "rerun" and not torch.is_grad_enabled() and not step_graphs_enabled()
        token = _CAPACITY_RETRY.set((device, previous.step_index,
            arguments["config"].get("_halo_exchanger")) if local_retry else None)
        try:
            if operator is None:
                result = (_checkpointed_cycle(previous, arguments) if checkpointed
                          else run_coupled_cycle(previous, **arguments))
            else:
                result = (_checkpointed_cycle(previous, arguments, operator=operator, state_of=state_of)
                          if checkpointed else operator(previous, **arguments))
        finally:
            _CAPACITY_RETRY.reset(token)
        if on_overflow == "assert":
            torch._assert_async(overflow_flag(device) == 0,
                "a compacted column buffer overflowed: rerun with on_overflow='rerun' or larger buffers")
            break
        regions = consensus_regions(overflowed_regions(device), arguments["config"].get("_halo_exchanger"))
        if not regions:
            break
        for region in regions:
            COMPACTION_LOG.append((previous.step_index, raise_compaction(region)))
    require_finite_state(selected_state(result))
    return result


def _native_coupled_output_fields(previous: CoupledCycleState, phase: NativeCoupledPhase, *, mesh):
    """Keep only passive output fields, never atmospheric continuation inputs."""
    from mpas_pytorch.atmosphere_model import (
        INTER_STEP_EXACT_DIAGNOSTIC_KEYS,
        INTER_STEP_PROGNOSTIC_KEYS,
    )
    from mpas_pytorch.field_provenance import MUST_COMPUTE
    from mpas_pytorch.state import STEP_FORCING_KEYS

    from .commit import NATIVE_SURFACE_UPDATE_KEYS
    from .dynamics import STEP_LOCAL_NATIVE_FIELDS, step_local_fields_enabled

    passive_work = {"precipw", "refl10cm", "relhum", "rh2", "u10", "v10"}
    excluded = (set(phase.native_context.support) | set(mesh)
        | MUST_COMPUTE | INTER_STEP_PROGNOSTIC_KEYS | INTER_STEP_EXACT_DIAGNOSTIC_KEYS
        | STEP_FORCING_KEYS | set(NATIVE_SURFACE_UPDATE_KEYS)
        | {"rho", "theta", "pressure", "qv", "surface_pressure"}
        | (set(STEP_LOCAL_NATIVE_FIELDS) - (set() if step_local_fields_enabled() else passive_work)))
    return {name: value for name, value in previous.native_state.items()
            if name not in excluded and not name.startswith("_")}


def run_coupled_interval(previous: CoupledCycleState, *, duration_seconds: int,
    on_overflow: str = "rerun", **arguments) -> CoupledCycleResult:
    """Execute whole source coupling cycles; retain only the last cycle result.

    A compacted buffer that held too few columns makes a cycle inexact (its overflow flag is set,
    ``ufs_physics_common.compaction``). ``on_overflow="rerun"`` (default) reads the flag once per
    physics group in forward-only execution, enlarges each overflowed buffer one ladder rung
    (``runtime_policy.raise_compaction``, recorded in ``COMPACTION_LOG``), and retries only that
    group from its unchanged inputs. Radiation, physics, and microphysics have separate checks;
    later overflow never repeats earlier groups or the dycore. Differentiable cycles retain
    whole-cycle recovery so checkpoint recomputation keeps its forward's capacities.
    ``"assert"`` reads nothing on the host and fails on the device at
    the end of an overflowed cycle (for runs whose buffers are known to be large enough).

    Global and CPU-partitioned multicycle intervals retain native-five phases internally.
    The final operator returns the unchanged full last-cycle observation record.
    Captured step graphs, LBCs and accelerator partitions retain their current boundary.
    Under grad, each operation is checkpointed (``set_cycle_checkpoint``).
    Each kept boundary must be finite (``require_finite_state``, a device check).
    """
    if on_overflow not in ("assert", "rerun"):
        raise ValueError("on_overflow must be 'assert' or 'rerun'")
    dt = arguments["config"]["config_dt"]
    if type(duration_seconds) is not int or duration_seconds <= 0:
        raise ValueError("coupled interval requires positive static duration_seconds")
    if type(dt) not in (int, float) or dt <= 0 or duration_seconds % dt != 0:
        raise ValueError("coupled interval must contain whole positive dycore timesteps")
    count = int(duration_seconds // dt)
    if (count > 1 and not step_graphs_enabled()
        and native_boundary_supported(arguments["config"])):
        phase = _run_native_coupled_start(previous, on_overflow=on_overflow, **arguments)
        output_fields = _native_coupled_output_fields(previous, phase, mesh=arguments["mesh"])
        for _ in range(count - 2):
            phase = _run_native_coupled_step(phase, on_overflow=on_overflow, **arguments)
        return _run_coupled_operation(phase, {**arguments, "_output_fields": output_fields}, on_overflow=on_overflow,
            operator=_finish_native_coupled_interval, state_of=_completed_cycle_state)
    result = None
    for _ in range(count):
        result = None
        result = _run_coupled_operation(previous, arguments, on_overflow=on_overflow)
        previous = result.state
    return result
