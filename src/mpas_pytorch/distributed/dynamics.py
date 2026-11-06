# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Field-specific MPAS dynamics halo phases around the existing tensor model."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch

from mpas_pytorch.profiling import profile_section, profiling_enabled

from .halo import HaloGroupPlan, HaloTransport, build_halo_plan
from .partition import MeshPartition

# mpas_atm_halos.F. Python's separate *_old keys represent time level one;
# current prognostic keys represent time level two after RK setup/recovery.
# A group can contain fields with different entity types and halo layers.
HALO_GROUPS = {
    "native_step:prognostics": (
        ("edge", (1, 2, 3), ("ru",)),
        ("cell", (1, 2), ("rw", "rho_p", "rtheta_p", "scalars")),
    ),
    "native_step:winds": (
        ("edge", (1, 2, 3), ("u",)),
        ("cell", (1, 2), ("w",)),
    ),
    # Loading supplies these halos in Fortran. At a public PyTorch boundary
    # owned controls are authoritative, including their tangents/cotangents.
    "initialization:prognostics": (
        ("edge", (1, 2, 3), ("u",)),
        ("cell", (1, 2), ("w", "theta_m", "rho_zz", "scalars")),
    ),
    "initialization:raw_prognostics": (
        ("edge", (1, 2, 3), ("u",)),
        ("cell", (1, 2), ("w", "theta", "rho", "qv", "scalars")),
    ),
    "initialization:pv_edge,ru,rw": (
        ("edge", (1, 2, 3), ("pv_edge", "ru")),
        ("cell", (1, 2), ("rw",)),
    ),
    "diagnostic_refresh:pv_edge,rho_edge": (("edge", (1, 2), ("pv_edge", "rho_edge")),),
    "theta_m,scalars,pressure_p,rtheta_p": (
        ("cell", (1, 2), ("theta_m", "scalars", "pressure_p", "rtheta_p")),
    ),
    "exner": (("cell", (1, 2), ("exner",)),),
    "tend_u": (("edge", (1,), ("tend_u",)),),
    "rtheta_pp": (("cell", (1,), ("rtheta_pp",)),),
    # Fortran's next-step rho_pp exchange, merged into this message.
    "rho_pp,rtheta_pp": (("cell", (1,), ("rho_pp", "rtheta_pp")),),
    "rw_p,ru_p,rho_pp,rtheta_pp": (
        ("cell", (1,), ("rw_p",)),
        ("edge", (2,), ("ru_p",)),
        ("cell", (1, 2), ("rho_pp",)),
        ("cell", (2,), ("rtheta_pp",)),
    ),
    "u_123": (("edge", (1, 2, 3), ("u",)),),
    "u_3": (("edge", (3,), ("u",)),),
    "w,pv_edge,rho_edge": (
        ("cell", (1, 2), ("w",)),
        ("edge", (1, 2), ("pv_edge", "rho_edge")),
    ),
    "w,pv_edge,rho_edge,scalars": (
        ("cell", (1, 2), ("w", "scalars")),
        ("edge", (1, 2), ("pv_edge", "rho_edge")),
    ),
    "theta_m,pressure_p,rtheta_p": (
        ("cell", (1, 2), ("theta_m", "pressure_p", "rtheta_p")),
    ),
    "scalars": (("cell", (1, 2), ("scalars",)),),
    "scalars_old": (("cell", (1, 2), ("scalars_old",)),),
    "w": (("cell", (1, 2), ("w",)),),
    "scale": (("cell", (1, 2), ("scale",)),),
    # The UFS physics commit of the coupled model (ufs_mpas_ccpp.commit): the cell wind
    # increments before the edge update reads its donor cells, then the eight-field commit
    # group before the output refresh. Both cover every halo layer, as the source exchanges.
    "ufs_commit:wind_increment": (("cell", (1, 2), ("zonal", "meridional")),),
    "ufs_commit:primaries": (
        (
            "cell",
            (1, 2),
            ("theta_m", "rho_zz", "scalars", "pressure_p", "rtheta_p", "exner"),
        ),
        ("edge", (1, 2, 3), ("u", "ru")),
    ),
    # The coupled model's native controls (ufs_mpas_ccpp.controls): owned rows are authoritative.
    "ufs_controls": (
        ("edge", (1, 2, 3), ("u",)),
        ("cell", (1, 2), ("w", "rw", "theta_m", "rho_zz", "scalars")),
    ),
}

_CURRENT_EXCHANGER: ContextVar[DynamicsHaloExchange | None] = ContextVar(
    "dynamics_halo_exchanger", default=None
)


@contextmanager
def dynamics_halo_policy(exchanger: DynamicsHaloExchange | None) -> Iterator[None]:
    """Scope numerical producer/consumer communication to one timestep."""
    token = _CURRENT_EXCHANGER.set(exchanger)
    try:
        yield
    finally:
        _CURRENT_EXCHANGER.reset(token)


def current_dynamics_halo_exchange() -> DynamicsHaloExchange | None:
    if torch.compiler.is_compiling():
        raise RuntimeError(
            "halo exchange context must remain outside compiled numeric regions"
        )
    return _CURRENT_EXCHANGER.get()


class DynamicsHaloExchange:
    """Blocking phase policy; numerical expressions stay in the existing model."""

    def __init__(
        self, partitions: Sequence[MeshPartition], transport: HaloTransport
    ) -> None:
        self.transport = transport
        self.partitions = tuple(partitions)
        self.partition = partitions[transport.rank]
        self.plans = {
            (entity, layers): build_halo_plan(
                partitions, rank=transport.rank, entity=entity, layers=layers
            )
            for bindings in HALO_GROUPS.values()
            for entity, layers, _ in bindings
        }
        if isinstance(transport, HaloTransport):
            self.plans = {
                key: transport.bind_plan(plan) for key, plan in self.plans.items()
            }
        self.group_plans = {
            group: HaloGroupPlan(
                tuple(
                    self.plans[entity, layers]
                    for entity, layers, names in bindings
                    for _ in names
                )
            )
            for group, bindings in HALO_GROUPS.items()
        }
        if isinstance(transport, HaloTransport):
            self.group_plans = {
                group: transport.bind_group(plan)
                for group, plan in self.group_plans.items()
            }

    def exchange_cell_patches(
        self, *owned_fields: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Supply cell limiter values to every local incident edge."""
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "halo exchange must remain outside compiled numeric regions"
            )
        cells = self.partition.entity("cell")
        if any(field.shape[0] != cells.owned_count for field in owned_fields):
            raise ValueError("cell patch exchange requires exactly owned cell rows")
        padded = tuple(
            torch.cat(
                (
                    field,
                    field.new_zeros(
                        (cells.local_count + 1 - cells.owned_count, *field.shape[1:])
                    ),
                )
            )
            for field in owned_fields
        )
        return self.transport.exchange(self.plans["cell", (1, 2)], *padded)

    def exchange_group(
        self, group: str, state: dict[str, Any], *, caller_owned: bool = False
    ) -> None:
        """Write halo rows; ``caller_owned`` fields are replaced, never mutated."""
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "halo exchange must remain outside compiled numeric regions"
            )
        bindings = HALO_GROUPS[group]
        # Validate the whole phase before issuing its first transfer.
        for entity, _layers, names in bindings:
            for name in names:
                field = state.get(name)
                if not isinstance(field, torch.Tensor):
                    raise TypeError(f"halo phase {group} requires tensor {name}")
                if field.ndim == 0 or field.shape[0] not in (
                    self.partition.entity(entity).local_count,
                    self.partition.entity(entity).local_count + 1,
                ):
                    raise ValueError(
                        f"halo phase {group}: {name} has wrong entity rows"
                    )
        # Fields with or without their padding row are exchanged in place of
        # a padded copy; the plans never address the padding row.
        fields = tuple(state[name] for _, _, names in bindings for name in names)
        if isinstance(self.transport, HaloTransport):
            private_inference = (
                not caller_owned
                and not torch.is_grad_enabled()
                and not any(
                    field.requires_grad
                    or torch.autograd.forward_ad.unpack_dual(field).tangent is not None
                    or torch._C._functorch.is_functorch_wrapped_tensor(field)
                    for field in fields
                )
            )
            outputs = self.transport.exchange(
                self.group_plans[group], *fields, inplace=private_inference
            )
        else:
            outputs = self.transport.exchange(self.group_plans[group], *fields)
        if all(out is field for out, field in zip(outputs, fields, strict=True)):
            return
        replacements = dict(zip(map(id, fields), outputs, strict=True))
        # Preserve existing pool aliases without retaining stale copies.
        for name, value in tuple(state.items()):
            if isinstance(value, torch.Tensor) and id(value) in replacements:
                state[name] = replacements[id(value)]


def exchange_halo_group(
    config: Mapping[str, Any],
    group: str,
    state: dict[str, Any],
    *,
    caller_owned: bool = False,
) -> None:
    """Ordinary single-process execution is unchanged; optional observer is diagnostic."""
    exchanger = config.get("_halo_exchanger")
    if exchanger is not None:
        if profiling_enabled(config):
            with profile_section(f"mpas.halo.{group}", config):
                exchanger.exchange_group(group, state, caller_owned=caller_owned)
        else:
            exchanger.exchange_group(group, state, caller_owned=caller_owned)
    observer = config.get("_halo_observer")
    if observer is not None:
        observer(group, state, HALO_GROUPS[group])
