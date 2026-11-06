# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Owned forward buffers around the complete Noah-MP science call."""

from collections import defaultdict
import copy
from typing import Any, Mapping

import torch

from .compile import compile_categories, normalize_compile_categories
from .driver import (
    PREPARED_TENSOR_COLUMNS_KEY, SMOOTH_TENSOR_OVERRIDE_KEY,
    NoahMpProductionOutput, load_case, prepare_noahmp_land_surface_case,
    prepare_noahmp_land_surface_call, run_noahmp_land_surface_call_tuple,
)
from .tables import PACKAGE_TABLE_VARIANT


class PreparedNoahRuntime:
    """Fixed physical grid, owned live inputs and packed forward outputs.

    Preparation does no current-input physics. ``run_into_buffers`` returns
    borrowed views; ``step`` returns an owned snapshot. Refresh, output packing
    and required snapshots belong in the declared complete-call timing boundary.
    Snow topology, vegetation/iteration controls and canopy classification are
    immutable; reprepare to change them. Use the ordinary complete-call API for
    gradients. CUDA numerics and actual capture require device qualification.
    """

    def __init__(self, case: str | Mapping[str, Any], *, device="cpu",
                 table_variant=PACKAGE_TABLE_VARIANT, categories="none"):
        case_data = load_case(case)
        if case_data.get(SMOOTH_TENSOR_OVERRIDE_KEY):
            raise ValueError("forward buffers do not accept smooth overrides; use the differentiable call")
        existing = case_data.get(PREPARED_TENSOR_COLUMNS_KEY, {})
        if any(t.requires_grad for t in existing.values()):
            raise ValueError("forward buffers cannot own gradient inputs")
        self.device = torch.device(device)
        if any(t.device != self.device for t in existing.values()):
            raise ValueError("prepared input device changed; reprepare the case")
        case_data = {**case_data, "columns": copy.deepcopy(case_data["columns"]),
                     "dimensions": copy.deepcopy(case_data["dimensions"]),
                     "driver_options": copy.deepcopy(case_data["driver_options"])}
        if existing:
            case_data[PREPARED_TENSOR_COLUMNS_KEY] = {k:t.clone() for k,t in existing.items()}
        else:
            case_data = prepare_noahmp_land_surface_case(case_data, device=self.device,
                                                       table_variant=table_variant)
        self.prepared = prepare_noahmp_land_surface_call(case_data, device=self.device,
                                                       table_variant=table_variant)
        self.inputs = self.prepared.case_data[PREPARED_TENSOR_COLUMNS_KEY]
        if any(t.is_floating_point() and t.dtype != torch.float64 for t in self.inputs.values()):
            raise ValueError("Noah parity requires float64 inputs")
        self.categories = normalize_compile_categories(categories)
        groups = defaultdict(list)
        for name in NoahMpProductionOutput._fields:
            if name == "O__errflg":
                spec = (torch.int64, ())
            else:
                value = self.inputs[name[3:]] if name[3:] in self.inputs else self.inputs["q1"]
                spec = (value.dtype, tuple(value.shape))
            groups[spec].append(name)
        self.groups = tuple((tuple(names), torch.empty((len(names), *shape),
                            dtype=dtype, device=self.device))
                            for (dtype, shape), names in groups.items())
        views = {name:buffer[i] for names,buffer in self.groups for i,name in enumerate(names)}
        self.borrowed = NoahMpProductionOutput(*(views[name] for name in NoahMpProductionOutput._fields))
        self.graph = None
        self.has_output = False

    def refresh(self, fields: Mapping[str, torch.Tensor]) -> None:
        """Update floating forcing/state in place within the fixed regime."""
        for name, value in fields.items():
            if name not in self.inputs or name in {"snowxy", "snow_levels", "shdmax"}:
                raise ValueError("physical configuration changed; reprepare Noah buffers")
            target = self.inputs[name]
            if (not target.is_floating_point() or value.requires_grad or
                value.shape != target.shape or value.dtype != target.dtype or
                value.device != target.device):
                raise ValueError("input signature/control changed; reprepare Noah buffers")
        with torch.no_grad():
            for name, value in fields.items():
                self.inputs[name].copy_(value)

    def run_into_buffers(self) -> NoahMpProductionOutput:
        with compile_categories(self.categories), torch.no_grad():
            actual = run_noahmp_land_surface_call_tuple(self.prepared)._asdict()
            for names, buffer in self.groups:
                torch.stack(tuple(actual[name] for name in names), out=buffer)
        self.has_output = True
        return self.borrowed

    def snapshot(self) -> NoahMpProductionOutput:
        if not self.has_output:
            raise RuntimeError("Noah physics has not produced outputs yet")
        views = {}
        for names, buffer in self.groups:
            owned = buffer.clone()
            views.update((name, owned[i]) for i,name in enumerate(names))
        return NoahMpProductionOutput(*(views[name] for name in NoahMpProductionOutput._fields))

    def step(self) -> NoahMpProductionOutput:
        if self.graph is None:
            self.run_into_buffers()
        else:
            self.graph.replay()
        return self.snapshot()

    def capture(self, *, warmup: int = 3) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("CUDA capture requires a CUDA device")
        if warmup < 1:
            raise ValueError("capture requires at least one warmup call")
        current = torch.cuda.current_stream(self.device)
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self.run_into_buffers()
        current.wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.run_into_buffers()
        self.graph = graph
