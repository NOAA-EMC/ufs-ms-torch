# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Sealed RRTMGP checkpoint loading and cumulative target dispatch.

The checkpoint loader remains an oracle-inspection utility.  The
``run_rrtmgp_radiation`` entry point is the profile-scope scientific target
dispatcher and computes target-owned A/B/D/E/G/H/I/J surfaces when the required
pre-call source-hook tensors are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .aerosol import RrtmgpAerosolResult, run_aerosol_from_boundaries
from .cloud import (
    RrtmgpCloudMpResult,
    RrtmgpCloudOverlapResult,
    run_active_cloud_overlap_from_boundaries,
    run_active_gfdl_cloud_mp_from_source_hook,
    run_clear_cloud_mp_from_boundaries,
    run_cloud_overlap_from_boundaries,
)
from .longwave_main import (
    RrtmgpLongwaveResult,
    active_longwave_source_hook_requires_aerosol,
    run_active_longwave_from_boundaries,
    run_longwave_from_boundaries,
)
from .post import RrtmgpPostResult, run_post_from_boundaries
from .pre import (RrtmgpPreResult, run_pre_from_host_inputs, _date_year_month,
                  _fpvs_table_on_device, _co2_table_on_device)
from .setup import RrtmgpSetupResult, run_setup_timestep_from_host_inputs
from .shortwave import (
    RrtmgpShortwaveResult,
    active_shortwave_source_hook_requires_aerosol,
    run_active_shortwave_from_boundaries,
    run_shortwave_from_boundaries,
)


BOOTSTRAP_STATUS = "scientific_a_j_runtime_closed_for_sealed_clear_and_active_source_hook_profiles"
SCIENTIFIC_RUNTIME = True
FULL_CHAIN_RUNTIME = True
SUPPORTED_SEGMENTS = ("A", "B", "D", "E", "G", "H", "I", "J")
SCHEME = "RRTMGP_SHORTWAVE_LONGWAVE_RADIATION"
HOST_CONTROL_FIELDS = frozenset(
    {
        "con_g",
        "con_pi",
        "con_rd",
        "deltim",
        "doGP_lwscat",
        "doLWclrsky",
        "doLWrad",
        "doSWclrsky",
        "doSWrad",
        "fhlwr",
        "fhswr",
        "i_o3",
        "iaerflg",
        "iaermdl",
        "ibd",
        "ibd_uv",
        "ico2",
        "ictm",
        "idat",
        "isol",
        "jdat",
        "nDay",
        "nGauss_angles",
        "rrtmgp_nGauss_ang",
        "solhr",
        "top_at_1",
        "top_at_1_initial",
        "use_LW_jacobian",
    }
)
PRODUCTION_STOCHASTIC_FIELDS = {
    "ABCDEFGH_sw_internal": frozenset({"rng3D_sw", "rng3D2_sw", "ipseed_sw"}),
    "ABCDEFGHI_lw_internal": frozenset({"rng3D_lw", "rng3D2_lw", "ipseed_lw"}),
}
PRODUCTION_FORBIDDEN_SOURCE_BOUNDARIES = frozenset({"ABCDEFG_aerosol_inputs"})


@dataclass(frozen=True)
class RrtmgpRadiationResult:
    """Device-native representation of the sealed A-J oracle corpus."""

    boundaries: Mapping[str, Mapping[str, torch.Tensor | str]]
    manifest_path: Path
    combined_checkpoint_sha256: str | None
    runtime_kind: str = "sealed_checkpoint_replay"


@dataclass(frozen=True)
class RrtmgpPreparedRadiationCall:
    """Immutable non-scientific preparation for repeated profile A-J calls.

    Physical tensor contents remain live inputs and are read on every call.
    Configuration is an owned host snapshot; changing a configuration tensor
    requires preparation again. Daylight selection is guarded on the device so
    changed coordinates cannot silently reuse a stale day/night partition.
    No layer, gas, cloud, aerosol, transport or final-output answer is cached.
    """

    pre_source_hook: Mapping[str, torch.Tensor | str]
    runtime_controls: Mapping[str, torch.Tensor | str]
    surface_inputs: Mapping[str, torch.Tensor | str] | None
    cloud_mp_inputs: Mapping[str, torch.Tensor | str] | None
    sw_stochastic: Mapping[str, torch.Tensor | str] | None
    lw_stochastic: Mapping[str, torch.Tensor | str] | None
    device: torch.device
    month: int
    year: int
    prepared_daylight: tuple[torch.Tensor, torch.Tensor, int]
    control_versions: tuple[tuple[Mapping[str, Any], torch.Tensor, int, str], ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_manifest() -> Path:
    return _repo_root() / "orchestration" / "rrtmgp_radiation" / "checkpoints" / "manifest.json"


def _tensor_from_variable(var, *, device: torch.device, name: str | None = None) -> torch.Tensor | str:
    import numpy as np

    data = var[...]
    array = np.asarray(data)
    variable_name = str(name or getattr(var, "name", ""))
    # Integer and boolean checkpoint fields are selectors/control state, not
    # differentiable physics arrays.  Keeping every such field on the host
    # prevents repeated CUDA scalar synchronizations in the A-J wrappers.
    tensor_device = (
        torch.device("cpu")
        if variable_name in HOST_CONTROL_FIELDS or array.dtype.kind in {"i", "u", "b"}
        else device
    )
    if array.dtype.kind in {"S", "U", "O"}:
        return "".join(array.astype(str).ravel()).rstrip()
    if array.dtype.kind == "f":
        return torch.as_tensor(array, dtype=torch.float64, device=tensor_device)
    if array.dtype.kind in {"i", "u", "b"}:
        return torch.as_tensor(array, dtype=torch.int64, device=tensor_device)
    raise TypeError(f"unsupported NetCDF variable dtype for {var.name}: {array.dtype}")


def load_checkpoint_corpus(
    checkpoint_manifest: str | Path | None = None,
    *,
    device: str | torch.device | None = None,
) -> RrtmgpRadiationResult:
    """Load sealed A-J NetCDF checkpoints as PyTorch tensors."""

    from netCDF4 import Dataset

    manifest_path = Path(checkpoint_manifest) if checkpoint_manifest is not None else _default_manifest()
    manifest_path = manifest_path.resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "sealed" or manifest.get("determinism_status") != "pass":
        raise ValueError(f"checkpoint manifest is not sealed and deterministic: {manifest_path}")
    if manifest.get("scheme") != SCHEME:
        raise ValueError(f"unexpected checkpoint scheme: {manifest.get('scheme')}")

    root = _repo_root()
    torch_device = torch.device(device) if device is not None else torch.device("cpu")
    boundaries: dict[str, dict[str, torch.Tensor | str]] = {}
    for entry in manifest.get("checkpoint_files", []):
        boundary_id = str(entry["boundary_id"])
        path = root / str(entry["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        expected_fields = list(entry.get("fields", []))
        with Dataset(path) as dataset:
            if getattr(dataset, "boundary_id", None) != boundary_id:
                raise ValueError(f"boundary id mismatch in {path}")
            if getattr(dataset, "scheme", None) != SCHEME:
                raise ValueError(f"scheme mismatch in {path}")
            missing = [name for name in expected_fields if name not in dataset.variables]
            if missing:
                raise ValueError(f"missing checkpoint variables in {path}: {missing}")
            fields = {
                name: _tensor_from_variable(dataset.variables[name], device=torch_device, name=name)
                for name in expected_fields
            }
        boundaries[boundary_id] = fields
    source_hook_entries = list(manifest.get("source_hook_hashes", []))
    if not source_hook_entries:
        freeze_lock_path = root / str(manifest.get("freeze_lock", "orchestration/rrtmgp_radiation/checkpoints/freeze_lock.json"))
        if freeze_lock_path.exists():
            with freeze_lock_path.open("r", encoding="utf-8") as handle:
                freeze_lock = json.load(handle)
            source_hook_entries = list(freeze_lock.get("source_hook_hashes", []))
    for entry in source_hook_entries:
        source_hook_id = str(entry.get("source_hook_id") or entry["boundary_id"])
        path = root / str(entry["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        expected_fields = list(entry.get("fields", []))
        with Dataset(path) as dataset:
            if getattr(dataset, "scheme", None) != SCHEME:
                raise ValueError(f"scheme mismatch in {path}")
            missing = [name for name in expected_fields if name not in dataset.variables]
            if missing:
                raise ValueError(f"missing source-hook variables in {path}: {missing}")
            fields = {
                name: _tensor_from_variable(dataset.variables[name], device=torch_device, name=name)
                for name in expected_fields
            }
        boundaries[source_hook_id] = fields
    return RrtmgpRadiationResult(
        boundaries=boundaries,
        manifest_path=manifest_path,
        combined_checkpoint_sha256=manifest.get("combined_checkpoint_sha256"),
    )


def _pre_inputs_from_ab_source_hook(
    source_hook: Mapping[str, torch.Tensor | str],
    setup_fields: Mapping[str, torch.Tensor | str],
) -> dict[str, torch.Tensor | str]:
    qgrs = source_hook["qgrs"]
    if not isinstance(qgrs, torch.Tensor):
        raise TypeError("AB_pre_inputs.qgrs must be a torch tensor")
    i_o3 = source_hook["i_o3"]
    if not isinstance(i_o3, torch.Tensor):
        raise TypeError("AB_pre_inputs.i_o3 must be a torch tensor")
    i_o3_index = int(i_o3.detach().to(device="cpu", dtype=torch.int64).reshape(-1)[0].item()) - 1
    pre_inputs = {
        "doSWrad": source_hook.get("doSWrad", setup_fields["doSWrad"]),
        "doLWrad": source_hook.get("doLWrad", setup_fields["doLWrad"]),
        "doSWclrsky": setup_fields["doSWclrsky"],
        "doLWclrsky": setup_fields["doLWclrsky"],
        "doGP_lwscat": setup_fields["doGP_lwscat"],
        "nGauss_angles": setup_fields["nGauss_angles"],
        "p_lay": source_hook["prsl"],
        "p_lev": source_hook["prsi"],
        "t_lay": source_hook["tgrs"],
        "q_lay": qgrs[0, :, :],
        "o3_lay": qgrs[i_o3_index, :, :],
        "tsfc": source_hook["tsfc"],
        "xlon": source_hook["xlon"],
        "xlat": source_hook["xlat"],
        "ico2": source_hook["ico2"],
        "ictm": source_hook["ictm"],
        "idat": source_hook["idat"],
        "jdat": source_hook["jdat"],
        "con_pi": source_hook["con_pi"],
        "coszen": setup_fields["coszen"],
    }
    top_at_1 = source_hook.get("top_at_1_initial", source_hook.get("top_at_1"))
    if top_at_1 is not None:
        pre_inputs["top_at_1"] = top_at_1
    return pre_inputs


def _run_setup_and_pre_from_source_hook(
    source_hook: Mapping[str, torch.Tensor | str],
    *,
    device: str | torch.device | None = None,
    runtime_controls: Mapping[str, Any] | None = None,
    prepared_daylight: tuple[torch.Tensor, torch.Tensor, int] | None = None,
) -> tuple[RrtmgpSetupResult, RrtmgpPreResult]:
    controls = runtime_controls or {}
    do_swclrsky = _control_bool(controls, "doSWclrsky", default=True)
    do_lwclrsky = _control_bool(controls, "doLWclrsky", default=True)
    do_gp_lwscat = _control_bool(controls, "doGP_lwscat", default=True)
    n_gauss_angles = _control_int(controls, "nGauss_angles", "rrtmgp_nGauss_ang", default=1)
    isol = _control_int(controls, "isol", default=2)
    setup = run_setup_timestep_from_host_inputs(
        source_hook,
        doSWclrsky=do_swclrsky,
        doLWclrsky=do_lwclrsky,
        doGP_lwscat=do_gp_lwscat,
        nGauss_angles=n_gauss_angles,
        isol=isol,
        device=device,
    )
    pre_inputs = _pre_inputs_from_ab_source_hook(source_hook, setup.fields)
    pre_inputs.update(
        {
            "doSWclrsky": torch.tensor([1 if do_swclrsky else 0], dtype=torch.int64),
            "doLWclrsky": torch.tensor([1 if do_lwclrsky else 0], dtype=torch.int64),
            "doGP_lwscat": torch.tensor([1 if do_gp_lwscat else 0], dtype=torch.int64),
            "nGauss_angles": torch.tensor([n_gauss_angles], dtype=torch.int64),
        }
    )
    pre = run_pre_from_host_inputs(pre_inputs, prepared_daylight=prepared_daylight)
    return setup, pre


def _control_bool(controls: Mapping[str, Any], name: str, *, default: bool) -> bool:
    value = controls.get(name)
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return int(value.detach().to(device="cpu", dtype=torch.int64).reshape(-1)[0].item()) != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", ".true.", "t", "yes"}
    return bool(value)


def _control_int(controls: Mapping[str, Any], name: str, *aliases: str, default: int) -> int:
    value: Any = None
    for key in (name, *aliases):
        if key in controls:
            value = controls[key]
            break
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return int(value.detach().to(device="cpu", dtype=torch.int64).reshape(-1)[0].item())
    return int(value)


def run_rrtmgp_radiation(
    *args: object,
    checkpoint_manifest: str | Path | None = None,
    device: str | torch.device = "cpu",
    boundaries: Mapping[str, Mapping[str, torch.Tensor | str]] | None = None,
    through: str = "J",
    month: int = 6,
    year: int = 1992,
    **kwargs: object,
) -> (
    RrtmgpRadiationResult
    | RrtmgpCloudMpResult
    | RrtmgpCloudOverlapResult
    | RrtmgpAerosolResult
    | RrtmgpShortwaveResult
    | RrtmgpLongwaveResult
    | RrtmgpPostResult
    | RrtmgpPreResult
    | RrtmgpSetupResult
):
    """Run the implemented PyTorch-native RRTMGP target surface.

    The implemented scientific surface computes A, B, D, E, G, H, I, and J
    cumulatively from source-hook inputs for hidden host state and stochastic
    streams, frozen coefficient data, frozen C/F neighbor boundaries, and
    explicit runtime control scalars. Setup/pre A-B use the sealed pre-call
    source hook because their true host-input/update state cannot be
    reconstructed from post-call A/AB cap outputs.
    Calling without `boundaries` preserves the checkpoint loader utility for
    oracle inspection, but that replay path is not a port.
    """

    if args or kwargs:
        raise NotImplementedError(
            "run_rrtmgp_radiation accepts explicit keyword inputs only: "
            "checkpoint_manifest, device, boundaries, through, month, and year"
        )
    if boundaries is not None:
        pre_source_hook = boundaries.get("AB_pre_inputs")
        runtime_controls = boundaries.get("runtime_controls", {})
        target_setup: RrtmgpSetupResult | None = None
        target_pre: RrtmgpPreResult | None = None
        if pre_source_hook is not None:
            target_setup, target_pre = _run_setup_and_pre_from_source_hook(
                pre_source_hook,
                device=device,
                runtime_controls=runtime_controls if isinstance(runtime_controls, Mapping) else {},
            )
        if through in {"A", "B"}:
            if target_setup is None or target_pre is None:
                raise NotImplementedError(
                    f"through={through!r} requires sealed AB_pre_inputs host-call tensors"
                )
            return target_setup if through == "A" else target_pre
        active_cloud_source = "ABCD_cloud_mp_inputs" in boundaries
        active_overlap_source = "AB_pre_inputs" in boundaries and active_cloud_source
        active_sw_source = "ABCDEFGH_sw_internal" in boundaries
        active_lw_source = "ABCDEFGHI_lw_internal" in boundaries
        if active_cloud_source and through != "D" and not active_overlap_source:
            raise NotImplementedError(
                "active cumulative RRTMGP runtime requires AB_pre_inputs for cloud-overlap decorrelation controls"
            )
        if through == "D":
            if active_cloud_source:
                return run_active_gfdl_cloud_mp_from_source_hook(boundaries["ABCD_cloud_mp_inputs"])
            cumulative_d: dict[str, Mapping[str, torch.Tensor | str]] = dict(boundaries)
            if target_setup is not None and target_pre is not None:
                cumulative_d["A"] = target_setup.fields
                cumulative_d["AB"] = target_pre.fields
            return run_clear_cloud_mp_from_boundaries(cumulative_d)
        if through == "E":
            cumulative_e: dict[str, Mapping[str, torch.Tensor | str]] = dict(boundaries)
            if target_setup is not None and target_pre is not None:
                cumulative_e["A"] = target_setup.fields
                cumulative_e["AB"] = target_pre.fields
            cloud_mp_e = (
                run_active_gfdl_cloud_mp_from_source_hook(cumulative_e["ABCD_cloud_mp_inputs"])
                if active_cloud_source
                else run_clear_cloud_mp_from_boundaries(cumulative_e)
            )
            cumulative_e["ABCD"] = cloud_mp_e.fields
            if active_overlap_source:
                return run_active_cloud_overlap_from_boundaries(cumulative_e)
            return run_cloud_overlap_from_boundaries(cumulative_e)
        cumulative: dict[str, Mapping[str, torch.Tensor | str]] = dict(boundaries)
        if target_setup is not None and target_pre is not None:
            cumulative["A"] = target_setup.fields
            cumulative["AB"] = target_pre.fields
        cloud_mp = (
            run_active_gfdl_cloud_mp_from_source_hook(cumulative["ABCD_cloud_mp_inputs"])
            if active_cloud_source
            else run_clear_cloud_mp_from_boundaries(cumulative)
        )
        cumulative["ABCD"] = cloud_mp.fields
        cloud_overlap = (
            run_active_cloud_overlap_from_boundaries(cumulative)
            if active_overlap_source
            else run_cloud_overlap_from_boundaries(cumulative)
        )
        cumulative["ABCDE"] = cloud_overlap.fields
        needs_aerosol = (
            (through in {"H", "J"} and active_shortwave_source_hook_requires_aerosol(cumulative))
            or (through in {"I", "J"} and active_longwave_source_hook_requires_aerosol(cumulative))
        )
        if through == "G":
            aerosol = run_aerosol_from_boundaries(cumulative, month=month, year=year)
            cumulative["ABCDEFG"] = aerosol.fields
            return aerosol
        if needs_aerosol:
            aerosol = run_aerosol_from_boundaries(cumulative, month=month, year=year)
            cumulative["ABCDEFG"] = aerosol.fields
        if through == "H":
            if active_sw_source:
                return run_active_shortwave_from_boundaries(cumulative, month=month, year=year)
            return run_shortwave_from_boundaries(cumulative)
        if through == "I":
            if active_lw_source:
                return run_active_longwave_from_boundaries(cumulative, month=month, year=year)
            return run_longwave_from_boundaries(cumulative)
        if through != "J":
            raise NotImplementedError(
                f"implemented RRTMGP target supports through='A', 'B', 'D', 'E', 'G', 'H', 'I', or 'J', got {through!r}"
            )
        shortwave = (
            run_active_shortwave_from_boundaries(cumulative, month=month, year=year)
            if active_sw_source
            else run_shortwave_from_boundaries(cumulative)
        )
        cumulative["ABCDEFGH"] = shortwave.fields
        longwave = (
            run_active_longwave_from_boundaries(cumulative, month=month, year=year)
            if active_lw_source
            else run_longwave_from_boundaries(cumulative)
        )
        cumulative["ABCDEFGHI"] = longwave.fields
        return run_post_from_boundaries(cumulative)
    return load_checkpoint_corpus(checkpoint_manifest, device=device or "cpu")


def prepare_rrtmgp_radiation_call(
    *,
    boundaries: Mapping[str, Mapping[str, torch.Tensor | str]],
    device: str | torch.device = "cpu",
    month: int = 6,
    year: int = 1992,
) -> RrtmgpPreparedRadiationCall:
    """Validate and project immutable metadata for repeated full A-J calls."""

    injected_boundaries = sorted(PRODUCTION_FORBIDDEN_SOURCE_BOUNDARIES & set(boundaries))
    if injected_boundaries:
        raise ValueError(
            "production A-J forbids injected scientific source boundaries: "
            f"{injected_boundaries}"
        )
    for boundary_id, allowed in PRODUCTION_STOCHASTIC_FIELDS.items():
        fields = boundaries.get(boundary_id)
        if fields is None:
            continue
        injected = sorted(set(fields) - allowed)
        if injected:
            raise ValueError(
                f"production A-J forbids injected {boundary_id} scientific fields: {injected}"
            )
    pre_source_hook = boundaries.get("AB_pre_inputs")
    if pre_source_hook is None:
        raise NotImplementedError("prepared A-J requires AB_pre_inputs host-call tensors")
    runtime_controls = boundaries.get("runtime_controls", {})
    if not isinstance(runtime_controls, Mapping):
        raise TypeError("runtime_controls must be a mapping")
    control_versions = []
    def prepared_controls(fields):
        projected = dict(fields)
        for name, value in fields.items():
            if name in HOST_CONTROL_FIELDS and isinstance(value, torch.Tensor):
                if value.requires_grad:
                    raise ValueError(f"configuration {name} must not require gradients")
                control_versions.append((fields, value, value._version, name))
                # Own immutable configuration on the host. Never move live
                # pressure, temperature, humidity, or coordinate fields here.
                with torch.inference_mode(False):
                    projected[name] = value.detach().to(device="cpu").clone()
        return projected
    pre_source_hook = prepared_controls(pre_source_hook)
    runtime_controls = prepared_controls(runtime_controls)
    torch_device = torch.device(device)
    # Lookup coefficients belong to preparation, not per-call host transfers.
    _fpvs_table_on_device(torch_device)
    if _control_int(pre_source_hook, "ico2", default=0) == 2:
        co2_year, _ = _date_year_month(pre_source_hook)
        _co2_table_on_device(co2_year, torch_device)
    # Only select daylight columns in preparation; no scientific layer/gas,
    # cloud, aerosol, transport, or final-output answers are cached.
    with torch.no_grad(), torch.inference_mode(False):
        setup = run_setup_timestep_from_host_inputs(
            pre_source_hook,
            doSWclrsky=_control_bool(runtime_controls, "doSWclrsky", default=True),
            doLWclrsky=_control_bool(runtime_controls, "doLWclrsky", default=True),
            doGP_lwscat=_control_bool(runtime_controls, "doGP_lwscat", default=True),
            nGauss_angles=_control_int(runtime_controls, "nGauss_angles", "rrtmgp_nGauss_ang", default=1),
            isol=_control_int(runtime_controls, "isol", default=2),
            device=torch_device,
        )
        daylight_mask = (setup.fields["coszen"] >= 0.0001).clone()
        selected = torch.nonzero(daylight_mask, as_tuple=False).reshape(-1) + 1
        nday = selected.numel()
        idxday = selected if nday else torch.zeros(1, device=torch_device, dtype=torch.int64)
    return RrtmgpPreparedRadiationCall(
        pre_source_hook=pre_source_hook,
        runtime_controls=runtime_controls,
        surface_inputs=boundaries.get("ABC"),
        cloud_mp_inputs=boundaries.get("ABCD_cloud_mp_inputs"),
        sw_stochastic=boundaries.get("ABCDEFGH_sw_internal"),
        lw_stochastic=boundaries.get("ABCDEFGHI_lw_internal"),
        device=torch_device,
        month=int(month),
        year=int(year),
        prepared_daylight=(daylight_mask, idxday, nday),
        control_versions=tuple(control_versions),
    )


def run_rrtmgp_radiation_call(prepared: RrtmgpPreparedRadiationCall) -> RrtmgpPostResult:
    """Execute all A-J science from a previously validated prepared call."""

    pre_source_hook = prepared.pre_source_hook
    runtime_controls = prepared.runtime_controls
    for fields, tensor, version, name in prepared.control_versions:
        if fields.get(name) is not tensor or tensor._version != version:
            raise ValueError(f"configuration {name} changed; prepare the call again")
    setup, pre = _run_setup_and_pre_from_source_hook(
        pre_source_hook,
        device=prepared.device,
        runtime_controls=runtime_controls,
        prepared_daylight=prepared.prepared_daylight,
    )

    # This mapping is the live production state, not a checkpoint trace.  Keep
    # only data still needed by a later science call and release stage products
    # as soon as their last consumer has run.
    live: dict[str, Mapping[str, torch.Tensor | str]] = {
        "A": setup.fields,
        "AB": pre.fields,
        "AB_pre_inputs": pre_source_hook,
        "runtime_controls": runtime_controls,
    }
    if prepared.surface_inputs is not None:
        live["ABC"] = prepared.surface_inputs
    if prepared.cloud_mp_inputs is not None:
        live["ABCD_cloud_mp_inputs"] = prepared.cloud_mp_inputs
    if prepared.sw_stochastic is not None:
        live["ABCDEFGH_sw_internal"] = prepared.sw_stochastic
    if prepared.lw_stochastic is not None:
        live["ABCDEFGHI_lw_internal"] = prepared.lw_stochastic
    active_cloud_source = prepared.cloud_mp_inputs is not None
    cloud_mp = (
        run_active_gfdl_cloud_mp_from_source_hook(
            live["ABCD_cloud_mp_inputs"],
            retain_diagnostics=False,
        )
        if active_cloud_source
        else run_clear_cloud_mp_from_boundaries(live, retain_diagnostics=False)
    )
    cloud_fields = cloud_mp.fields
    live["ABCD"] = {
        name: cloud_fields[name]
        for name in (
            "cld_frac",
            "cld_lwp",
            "cld_iwp",
            "cld_swp",
            "cld_rwp",
            "cld_reliq",
            "cld_reice",
            "cld_resnow",
        )
    }
    del cloud_mp, cloud_fields
    cloud_overlap = (
        run_active_cloud_overlap_from_boundaries(live)
        if active_cloud_source
        else run_cloud_overlap_from_boundaries(live)
    )
    live["ABCDE"] = {"cloud_overlap_param": cloud_overlap.fields["cloud_overlap_param"]}
    del cloud_overlap
    needs_aerosol = active_shortwave_source_hook_requires_aerosol(
        live
    ) or active_longwave_source_hook_requires_aerosol(live)
    aerosol = (
        run_aerosol_from_boundaries(
            live,
            month=prepared.month,
            year=prepared.year,
            retain_diagnostics=False,
        )
        if needs_aerosol
        else None
    )
    if aerosol is not None:
        live["ABCDEFG"] = aerosol.fields

    active_sw_source = prepared.sw_stochastic is not None
    shortwave = (
        run_active_shortwave_from_boundaries(live, month=prepared.month, year=prepared.year)
        if active_sw_source
        else run_shortwave_from_boundaries(live)
    )

    # SW consumes all three shortwave aerosol properties. LW needs only optical
    # depth, so the other G outputs must not remain live during LW transport.
    if aerosol is not None:
        aerlw_tau = aerosol.fields["aerlw_tau"]
        live["ABCDEFG"] = {"aerlw_tau": aerlw_tau}
        del aerosol

    active_lw_source = prepared.lw_stochastic is not None
    longwave = (
        run_active_longwave_from_boundaries(live, month=prepared.month, year=prepared.year)
        if active_lw_source
        else run_longwave_from_boundaries(live)
    )

    post_inputs: dict[str, Mapping[str, torch.Tensor | str]] = {
        "A": setup.fields,
        "AB": pre.fields,
        "ABCDEFGH": shortwave.fields,
        "ABCDEFGHI": longwave.fields,
        "AB_pre_inputs": pre_source_hook,
    }
    post_inputs["runtime_controls"] = runtime_controls
    del live
    return run_post_from_boundaries(post_inputs)


def run_rrtmgp_radiation_prepared(
    *,
    boundaries: Mapping[str, Mapping[str, torch.Tensor | str]],
    device: str | torch.device = "cpu",
    month: int = 6,
    year: int = 1992,
) -> RrtmgpPostResult:
    """Validate and execute full A-J without injected scientific outputs."""

    return run_rrtmgp_radiation_call(
        prepare_rrtmgp_radiation_call(
            boundaries=boundaries,
            device=device,
            month=month,
            year=year,
        )
    )
