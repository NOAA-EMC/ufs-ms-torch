# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Selected R4 physics commit followed by the maintained MPAS timestep."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.autograd.forward_ad as forward_ad
from mpas_pytorch.atmosphere_model import run_timestep
from mpas_pytorch.dynamics.static_payloads import DycoreStaticPayloads
from ufs_physics_common import cost_meter
from ufs_physics_common.compile_policy import eager_regions_active

from .commit import CommitDomain, NativeCommitResult, commit_mpas_state, finalize_stateout
from .projection import TracerContract
from .reverse import CommitPhase
from .runtime_policy import step_local_fields_enabled, dycore_checkpoint_enabled, dycore_forward_only, glue
from .surface_containers import NativeSurface
from .surface_mirror import SurfaceMirrorInputs


@dataclass(frozen=True)
class PhysicsDynamicsResult:
    """A dycore boundary, not a completed radiation/physics/microphysics cycle."""

    native_state: dict[str, Any]
    native_commit: NativeCommitResult
    completed_dycore_steps: torch.Tensor


def dycore_experimental(config: dict, device: torch.device, dtype: torch.dtype | None = None) -> bool:
    """Whether the coupled step requests the dycore's experimental capabilities: a partitioned run (the
    distributed dycore, ``mpas_pytorch.distributed``), a non-CPU device or a float32 state (the release
    preview admits CPU float64 only)."""
    return (config.get("_halo_exchanger") is not None or torch.device(device).type != "cpu"
            or (dtype is not None and dtype != torch.float64))


def native_boundary_supported(config: dict, device: torch.device | None = None) -> bool:
    """Keep unqualified LBC and accelerator halo policies on their carried boundary."""
    from mpas_pytorch.distributed.dynamics import DynamicsHaloExchange

    exchanger = config.get("_halo_exchanger")
    if config.get("config_apply_lbcs", False):
        return False
    if exchanger is None:
        return True
    if not isinstance(exchanger, DynamicsHaloExchange):
        return False
    device = device if device is not None else getattr(exchanger.transport, "device", None)
    return device is not None and torch.device(device).type == "cpu"


def advance_physics_dynamics(
    before: dict[str, Any],
    surface: NativeSurface,
    statein,
    prefix,
    *,
    mesh: dict[str, Any],
    config: dict[str, Any],
    domain: CommitDomain,
    tracers: TracerContract,
    water_species: tuple[bool, ...],
    cumulative,
    dtp: torch.Tensor,
    cap_step_index: torch.Tensor,
    completed_dycore_steps: torch.Tensor,
    _native_boundary_output: bool = False,
) -> PhysicsDynamicsResult:
    """Finalize -> phase1 commit -> synchronize/refresh -> one MPAS interval.

    The physics interval equals config_dt: one physics call per dycore step, with no coupling
    subcycle. The authenticated commit entry selects refresh-only dycore preparation, retains
    pre-physics rw and carries the physics forcing of the RK stages (``commit_mpas_state``).
    The dycore owns RK/split time levels; this wrapper does not initialize them
    again or inject diagnostics. The source counter sync uses max(prior, kdt-1).
    Source IAU and native MPAS physics are off; CCPP microphysics follows later.
    """
    # Derivatives in either mode (reverse: grad mode; forward: an active dual level) need the
    # dycore's differentiable path; its no-grad path uses in-place kernels without AD rules.
    forward_mode = forward_ad._current_level >= 0
    differentiable = torch.is_grad_enabled() or forward_mode
    reference = before["u"]
    dt = config["config_dt"]
    if (
        dtp.shape != () or dtp.dtype != reference.dtype or dtp.device != reference.device
        or not isinstance(dt, (float, int)) or isinstance(dt, bool) or dt <= 0
    ):
        raise ValueError("physics/dynamics requires a device-local timestep and static config_dt")
    torch._assert_async(torch.isfinite(dtp) & (dtp == dt),
        "selected R4 physics interval must equal the dycore timestep")
    for count in (cap_step_index, completed_dycore_steps):
        if count.shape != () or count.dtype != torch.int32 or count.device != reference.device:
            raise ValueError("dycore counters require device-local int32 scalars")
    torch._assert_async((cap_step_index >= 1) & (completed_dycore_steps >= 0)
        & (completed_dycore_steps < 2147483647), "invalid or overflowing dycore counter")
    synced = torch.maximum(completed_dycore_steps, torch.clamp_min(cap_step_index - 1, 0))
    # Do not publish a successful interval or increment its carry if integration fails.
    finalized = glue(finalize_stateout)(prefix, dtp=dtp)
    committed, native_commit = commit_mpas_state(
        before, surface, statein, finalized.stateout,
        mesh=mesh, config=config, domain=domain, phase=CommitPhase.RADIATION_PHYSICS,
        tracers=tracers, water_species=water_species, cumulative=cumulative,
        sfcprop=SurfaceMirrorInputs.from_sfcprop(finalized.sfcprop),
    )
    step_config = dycore_step_config(config, mesh, differentiable=differentiable, forward_mode=forward_mode)
    if _native_boundary_output:
        if not native_boundary_supported(config, reference.device):
            raise ValueError("native coupled output is not qualified for this boundary policy")
        # A partition's owned columns end before its halo rows. Only the
        # true sentinel row is interface-pressure padding.
        padding = committed["surface_pressure"][committed["rho_p"].shape[0] - 1:]
        torch._assert_async(((padding == 0) & ~torch.signbit(padding)).all(),
            "native coupled phase requires positive-zero interface-pressure padding")
        step_config["_native_boundary_output"] = True
    # The previous coupled state must stay unchanged (restarts, continuations, boundary guards).
    # The dycore's no-grad step copies the work buffers it reuses; a write into any other tensor
    # of the previous state is an error (version counters: host metadata, no device read).
    shared = {} if differentiable else {
        n: v._version for n, v in committed.items() if isinstance(v, torch.Tensor) and before.get(n) is v}
    with cost_meter.region("dycore_step"):  # a null context when the meter is off
        advanced = run_timestep(committed, mesh, config=step_config,
            dt=dt, device=reference.device, differentiable=differentiable,
            experimental=dycore_experimental(config, reference.device, reference.dtype))
    written = sorted(n for n, version in shared.items() if committed[n]._version != version)
    if written:
        raise RuntimeError(f"the dycore's no-grad step wrote into the previous state: {written}")
    return PhysicsDynamicsResult(advanced, native_commit, synced + 1)


# Dycore work buffers and time levels that each step creates again and no later step, physics or commit
# reads (traced over a radiation and an ordinary cycle, ). The
# dycore drops them from its result (private key ``_step_local_fields``), so the step boundary does not
# carry them; a step that needed one would fail on the missing key.
STEP_LOCAL_NATIVE_FIELDS = (
    "scalars_init", "scalars_old", "scalar_old", "scalars_new", "scalar_new",
    "theta_m_init", "theta_m_save", "rho_zz_init", "rho_zz_old", "rho_zz_int", "rho_zz_old_split",
    "u_rk_save", "ru_save", "rw_save", "rtheta_p_save", "rho_p_save", "rr_save", "ruAvg_split", "wwAvg_split",
    "tend_ru", "tend_rw", "tend_rt", "tend_u", "tend_w", "tend_theta", "tend_theta_m",
    "cqu", "cqw", "cofrz", "coftz", "cofwr", "cofwt", "cofwz",
    "h_edge", "rho_edge", "ke", "pv_edge", "divergence", "vorticity", "v", "h_divergence",
    "rt", "rthdynten", "p", "pp", "relhum", "precipw", "refl10cm", "rh2", "u10", "v10", "qtot",
    # The dycore's work buffers and per-step fields that a step creates (zeros or recomputed) when absent; carried,
    # they stayed live through physics (41.7 KiB/col, 6.5 GiB at x1.163842).
    "a_tri", "alpha_tri", "gamma_tri", "flux_arr", "flux_tmp", "flux_upwind_tmp", "horiz_flux_arr", "h", "kdiff",
    "pb", "pressure_b", "rb", "rtb", "rho_pp", "rho_zz_new", "rr", "rtheta_pp", "ruAvg", "ru_p", "rw_p", "s_max",
    "s_min", "scalar_tend", "scalar_tend_save", "scale_arr", "t", "tend_rho", "tend_rho_zz", "tend_theta_euler",
    "tend_u_euler", "tend_w_euler", "uhAvg", "w_init", "wdtn", "wwAvg", "fnm", "fnp", "rdnw",
    "tend_ru_physics", "tend_rtheta_physics",
)


def dycore_step_config(config: dict[str, Any], mesh: dict, *, differentiable: bool, forward_mode: bool) -> dict:
    """The dycore's config for one step: the run's config plus private per-step execution choices.

    Private keys stay out of the continuation identity, so one run config serves every kind of
    step. One static payload store per mesh serves every step. Differentiable steps checkpoint the
    dycore's RK stages (``runtime_policy.set_dycore_checkpoint``). Compiled dycore regions run under
    derivatives only where qualified: reverse mode with ``with_dycore_compile(reverse=True)``
    (M9.3). Forward-mode steps, and reverse-mode steps of a forward-only compiled config, run the
    regions eagerly: the eager differentiable step, which is the reference. Persist the
    cold-step eager override in this config so later checkpoint recomputation keeps it.
    """
    # The UFS host builds the MPAS core without DO_PHYSICS: no end-of-step zeroing of negative scalars.
    step_config = {**config, "_dycore_static_payloads": _static_payloads(mesh),
                   "_step_local_fields": STEP_LOCAL_NATIVE_FIELDS if step_local_fields_enabled() else (),
                   "_zero_negative_scalars": False}
    if differentiable and dycore_checkpoint_enabled():
        step_config["gradient_checkpointing"] = True
    if differentiable and config.get("config_use_torch_compile", False) and (
            forward_mode or dycore_forward_only(config) or eager_regions_active()):
        step_config["_torch_compile_suspended"] = True
    return step_config


_STATIC_PAYLOADS: dict[int, tuple[dict, DycoreStaticPayloads]] = {}


def _static_payloads(mesh: dict) -> DycoreStaticPayloads:
    """The dycore's connectivity payload store for ``mesh`` (built once, reused every step)."""
    hit = _STATIC_PAYLOADS.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        hit = _STATIC_PAYLOADS[id(mesh)] = (mesh, DycoreStaticPayloads())
    return hit[1]


@dataclass(frozen=True)
class MicrophysicsCommitResult:
    native_state: dict[str, Any]
    native_commit: NativeCommitResult
    group_states: torch.Tensor


def commit_microphysics(before,surface,statein,prefix,*,mesh,config,domain,
                        tracers,water_species,cumulative,group_states):
    """Finalize four R4 cap groups, then commit phase2 and refresh diagnostics.

    The generated timestep_final routines have no numerical scheme calls.
    Unlike phase1, this group has no GFS_mpas_stateout_update and no surface
    mirror. It commits the completed Stateout directly without another dycore.
    """
    if (group_states.shape != (4,) or group_states.dtype != torch.int32
        or group_states.device != before["u"].device):
        raise ValueError("selected cap requires four device-local int32 group states")
    torch._assert_async((group_states == 2).all(),"CCPP timestep_final: invalid group state")
    final_states = torch.ones_like(group_states)
    committed,result = commit_mpas_state(before,surface,statein,prefix.stateout,
        mesh=mesh,config=config,domain=domain,phase=CommitPhase.MICROPHYSICS,
        tracers=tracers,water_species=water_species,cumulative=cumulative,sfcprop=None)
    return MicrophysicsCommitResult(committed,result,final_states)
