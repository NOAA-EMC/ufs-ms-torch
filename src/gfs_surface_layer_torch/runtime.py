# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Owned forward buffers for the existing surface science and CUDA graph replay."""

from collections import defaultdict
from typing import Mapping

import torch

from .cases import SurfaceCaseConfig
from .driver import (
    _surface_output_names,
    fpvs_table_like,
    run_gfs_surface_layer,
    run_gfs_surface_layer_efficient,
)


class PreparedSurfaceRuntime:
    """A fixed configuration and shape, with live owned forcing buffers.

    Preparation allocates buffers and the immutable saturation table; it does
    no current-input physics. ``step`` returns owned outputs. ``run_into_buffers``
    returns borrowed views valid until the next call. Input refresh and output
    snapshots are real work and must be included in their declared timed scope.

    This forward-only capture interface rejects gradient inputs. Use the ordinary
    efficient runner for independent differentiable calls. Actual CUDA capture
    and compiled CUDA numerics require qualification on the target device.
    """

    def __init__(self, fields: Mapping[str, torch.Tensor], cfg: SurfaceCaseConfig,
                 *, compiled: bool = False, max_fusion_size: int = 128,
                 fpvs_table: torch.Tensor | None = None):
        if any(value.requires_grad for value in fields.values()):
            raise ValueError("forward capture buffers cannot own gradient inputs")
        self.cfg = cfg
        self.compiled = compiled
        self.max_fusion_size = max_fusion_size
        self.device = fields["ps"].device
        if any(value.device != self.device for value in fields.values()):
            raise ValueError("surface inputs must share a device")
        if any(value.is_floating_point() and value.dtype != torch.float64
               for value in fields.values()):
            raise ValueError("surface parity requires float64 inputs")
        self.fields = {name: value.clone() for name, value in fields.items()}
        table = fpvs_table_like(self.fields["ps"]) if fpvs_table is None else fpvs_table
        if table.shape != (7501,) or table.device != self.device or table.dtype != torch.float64 or table.requires_grad:
            raise ValueError("prepared surface FPVS table must have7501 immutable float64 entries on the input device")
        self.table = table.clone()
        self.output_names = _surface_output_names(self.fields, None)
        groups = defaultdict(list)
        for name in self.output_names:
            if name in self.fields:
                value = self.fields[name]
                spec = (value.dtype, tuple(value.shape))
            elif name in {"errflg", "errflg_diag"}:
                spec = (torch.int64, ())
            else:
                spec = (fields["ps"].dtype, tuple(fields["ps"].shape))
            groups[spec].append(name)
        self.groups = tuple((tuple(names), torch.empty((len(names), *shape),
                            dtype=dtype, device=self.device))
                            for (dtype, shape), names in groups.items())
        self.borrowed = {name: buffer[index] for names, buffer in self.groups
                         for index, name in enumerate(names)}
        self.graph = None
        self.has_output = False

    def refresh(self, fields: Mapping[str, torch.Tensor]) -> None:
        """Copy current inputs without replacing any prepared tensor storage."""
        if set(fields) != set(self.fields):
            raise ValueError("surface input fields changed; reprepare the runtime")
        for name, value in fields.items():
            target = self.fields[name]
            if value.requires_grad or value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError("surface input signature changed; reprepare the runtime")
        with torch.no_grad():
            for name, value in fields.items():
                self.fields[name].copy_(value)

    def run_into_buffers(self) -> dict[str, torch.Tensor]:
        """Execute current physics into stable, borrowed output buffers."""
        with torch.no_grad():
            actual = (run_gfs_surface_layer_efficient(self.fields, self.cfg,
                        max_fusion_size=self.max_fusion_size, fpvs_table=self.table) if self.compiled else
                      run_gfs_surface_layer(self.fields, self.cfg, fpvs_table=self.table))
            for names, buffer in self.groups:
                torch.stack(tuple(actual[name] for name in names), out=buffer)
        self.has_output = True
        return self.borrowed

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Copy each packed output buffer once, preserving complete ownership."""
        if not self.has_output:
            raise RuntimeError("surface physics has not produced outputs yet")
        result = {}
        for names, buffer in self.groups:
            owned = buffer.clone()
            result.update((name, owned[index]) for index, name in enumerate(names))
        return {name: result[name] for name in self.output_names}

    def step(self) -> dict[str, torch.Tensor]:
        if self.graph is None:
            self.run_into_buffers()
        else:
            self.graph.replay()
        return self.snapshot()

    def capture(self, *, warmup: int = 3) -> None:
        """Warm on a separate stream and capture forward physics only.

        The owning worker must bound cold setup and capture time. ``step`` keeps
        the output ownership copies outside replay and inside the complete call.
        """
        if self.device.type != "cuda":
            raise RuntimeError("CUDA graph capture requires a CUDA device")
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
