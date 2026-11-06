# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""R4 MPAS global-ID radiation seeds and the generated CCPP seed copyback."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from rrtmgp_radiation_torch.radiation_group import gfs_mersenne_twister_uniform

from .clock import _nint_nonnegative


@dataclass(frozen=True)
class RadiationSeeds:
    rseeds: torch.Tensor | None
    icsdsw: torch.Tensor
    icsdlw: torch.Tensor

    @classmethod
    def allocate(
        cls, *, ncol: int, nrstreams: int, lrseeds: bool, device: torch.device | str
    ) -> RadiationSeeds:
        if (
            type(ncol) is not int
            or ncol <= 0
            or type(nrstreams) is not int
            or nrstreams < 2
        ):
            raise ValueError(
                "radiation seed allocation requires positive columns and two streams"
            )
        if type(lrseeds) is not bool:
            raise TypeError("lrseeds must be static configuration")
        return cls(
            torch.zeros((ncol, nrstreams), dtype=torch.int32, device=device)
            if lrseeds
            else None,
            torch.zeros(ncol, dtype=torch.int32, device=device),
            torch.zeros(ncol, dtype=torch.int32, device=device),
        )


def radiation_seed_offset(idate: torch.Tensor) -> torch.Tensor:
    """GFS_rrtmgp_setup_init: idate is [hour, month, day, year], not NCEP jdat."""
    if idate.shape != (4,) or idate.dtype != torch.int32:
        raise ValueError("radiation seed offset requires four source int32 date values")
    date = idate.to(torch.int64)
    return (17 * date[0] + 43 * date[1] + 37 * date[2] + 23 * date[3]).to(torch.int32)


def radiation_permutation_seed(
    sec: torch.Tensor, ipsd0: torch.Tensor, *, ipsdlim: int
) -> torch.Tensor:
    """Source nonnegative NINT/MOD arithmetic and int32 addition semantics."""
    if sec.numel() != 1:
        raise ValueError("invalid radiation seed time/offset dtype, shape or device")
    return _permutation_seeds(sec.reshape(1), ipsd0, ipsdlim=ipsdlim).reshape(())


def _permutation_seeds(
    sec: torch.Tensor, ipsd0: torch.Tensor, *, ipsdlim: int
) -> torch.Tensor:
    """``radiation_permutation_seed`` of each of ``sec[B]`` (elementwise, so each equals the single form)."""
    if (
        sec.ndim != 1
        or sec.dtype != torch.float64
        or ipsd0.numel() != 1
        or ipsd0.dtype != torch.int32
        or ipsd0.device != sec.device
    ):
        raise ValueError("invalid radiation seed time/offset dtype, shape or device")
    if type(ipsdlim) is not int or not 0 < ipsdlim <= torch.iinfo(torch.int32).max:
        raise ValueError("ipsdlim must be a positive source int32 limit")
    torch._assert_async(torch.isfinite(sec).all(), "nonfinite radiation seed time")
    value = 100.0 * torch.sqrt(torch.clamp_min(sec, 0.0))
    torch._assert_async(
        (value < 2147483647.5).all(), "radiation seed NINT exceeds source int32"
    )
    rounded = _nint_nonnegative(value)
    return (
        torch.remainder(rounded, ipsdlim) + 1 + ipsd0.reshape(()).to(torch.int64)
    ).to(torch.int32)


def random_index(seed: torch.Tensor, *, imax: int, count: int) -> torch.Tensor:
    """Source random_index_t: MT stream in 624-value blocks, CEILING then MAX(1)."""
    if seed.dtype != torch.int32 or seed.numel() != 1:
        raise ValueError("random_index requires one source int32 seed")
    return _random_index_rows(seed.reshape(1), imax=imax, count=count).reshape(-1)


def _random_index_rows(seeds: torch.Tensor, *, imax: int, count: int) -> torch.Tensor:
    """``random_index`` of each of ``seeds[B]`` as rows ``[B, count]`` (the twist is row by row)."""
    if type(imax) is not int or not 0 < imax <= torch.iinfo(torch.int32).max:
        raise ValueError("random_index imax must be a positive source int32 limit")
    if type(count) is not int or count < 0:
        raise ValueError("random_index count must be a nonnegative static integer")
    if count == 0:
        return torch.empty((seeds.shape[0], 0), dtype=torch.int32, device=seeds.device)
    uniform = gfs_mersenne_twister_uniform(seeds, count)
    return torch.clamp_min(torch.ceil(uniform * imax), 1.0).to(torch.int32)


def radiation_seed_draws(
    sec: torch.Tensor,
    ipsd0: torch.Tensor,
    index_to_cell_id: torch.Tensor,
    *,
    n_cells_solve: int,
    n_cells_global: int,
    ipsdlim: int,
    lrseeds: bool,
) -> torch.Tensor:
    """The two seed draws ``[B, n_cells_solve, 2]`` of radiation steps at elapsed seconds ``sec[B]``.

    They depend on the time, the date offset ``ipsd0``, the static domain and (host-provided McICA,
    ``lrseeds``) the global cell IDs only, never on the model state: the host's shared
    ``2*nCellsGlobal`` permutation read at each column's global ID, or (CCPP fallback) the local
    ``2*n`` stream. Rows are independent, so each equals the draw of its step alone.
    """
    n = n_cells_solve
    if type(lrseeds) is not bool:
        raise TypeError("radiation seed configuration must be static")
    seeds = _permutation_seeds(sec, ipsd0, ipsdlim=ipsdlim)
    if not lrseeds:
        # The CCPP fallback reads SEC only on active steps (``sec`` is 0 on the others).
        torch._assert_async(
            (sec >= 0.0).all(), "negative active CCPP radiation seed time"
        )
        return _random_index_rows(seeds, imax=ipsdlim, count=2 * n).reshape(-1, n, 2)
    if (
        index_to_cell_id.ndim != 1
        or index_to_cell_id.shape[0] < n
        or index_to_cell_id.dtype not in (torch.int32, torch.int64)
        or index_to_cell_id.device != sec.device
    ):
        raise ValueError("invalid native radiation global cell IDs")
    ids = index_to_cell_id[:n].to(torch.int64)
    torch._assert_async(
        ((ids >= 1) & (ids <= n_cells_global)).all(),
        "radiation global cell ID outside domain",
    )
    permutation = _random_index_rows(seeds, imax=ipsdlim, count=2 * n_cells_global)
    return permutation.reshape(-1, n_cells_global, 2)[:, ids - 1]


@dataclass(frozen=True)
class RadiationSeedStream:
    """Seed draws of upcoming radiation steps, prepared outside the step (``cycle.prepare_step``).

    ``secs[R]`` holds each row's elapsed seconds (the cycle's own ``sec + dtp`` chain) and
    ``draws[R, n, 2]`` its ``radiation_seed_draws``; ``steps`` names the rows' static step indices.
    A step selects its row on the device by its clock seconds, so one recorded step graph reads the
    right row at every replay; a refill writes these tensors in place.
    """

    secs: torch.Tensor
    draws: torch.Tensor
    steps: tuple[int, ...]

    def select(self, sec: torch.Tensor) -> torch.Tensor:
        """The ``[n, 2]`` draws of the step at elapsed seconds ``sec`` (device-side row choice)."""
        hit = self.secs == sec.reshape(())
        torch._assert_async(hit.any(), "radiation seed stream holds no row for this step's time")
        row = torch.argmax(hit.to(torch.int32)).reshape(1)
        return self.draws.index_select(0, row).reshape(self.draws.shape[1:])


def advance_mpas_radiation_seeds(
    previous: RadiationSeeds,
    index_to_cell_id: torch.Tensor,
    *,
    n_cells_solve: int,
    n_cells_global: int,
    sec: torch.Tensor,
    ipsd0: torch.Tensor,
    ipsdlim: int,
    lrseeds: bool,
    lsswr: torch.Tensor,
    lslwr: torch.Tensor,
    isubc_sw: int,
    isubc_lw: int,
    draws: torch.Tensor | None = None,
) -> RadiationSeeds:
    """Populate host seeds each interval, then conditionally copy both CCPP streams.

    Global IDs select one shared 2*nCellsGlobal stream. They are not a hash or
    local permutation. Radiation skipped means ICS carries remain unchanged,
    even though MPAS has already updated rseeds. This is not a full MPI host.
    ``draws`` is this step's ``radiation_seed_draws`` row ``[n, 2]`` when prepared
    ahead (``RadiationSeedStream``); without it the step draws it here.
    """
    n = n_cells_solve
    if (
        type(n) is not int
        or n <= 0
        or type(n_cells_global) is not int
        or not n <= n_cells_global <= torch.iinfo(torch.int32).max // 2
    ):
        raise ValueError("invalid static radiation seed domain")
    if (
        type(lrseeds) is not bool
        or type(isubc_sw) is not int
        or type(isubc_lw) is not int
    ):
        raise TypeError("radiation seed configuration must be static")
    device = sec.device
    for flag in (lsswr, lslwr):
        if flag.dtype != torch.bool or flag.numel() != 1 or flag.device != device:
            raise ValueError("radiation triggers must be scalar device bool tensors")
    for value in (previous.icsdsw, previous.icsdlw):
        if (
            value.ndim != 1
            or value.shape[0] < n
            or value.dtype != torch.int32
            or value.device != device
        ):
            raise ValueError("invalid allocated CCPP radiation seed stream")
    if previous.icsdsw.shape != previous.icsdlw.shape:
        raise ValueError("CCPP radiation seed streams must have identical extents")
    active = (lsswr | lslwr).reshape(()) & (isubc_sw == 2 or isubc_lw == 2)
    rseeds = previous.rseeds
    if lrseeds:
        if (
            rseeds is None
            or rseeds.ndim != 2
            or rseeds.shape[0] != previous.icsdsw.shape[0]
            or rseeds.shape[1] < 2
            or rseeds.dtype != torch.int32
            or rseeds.device != device
        ):
            raise ValueError(
                "host-provided McICA requires allocated two-stream int32 storage"
            )
        if draws is None:
            draws = radiation_seed_draws(sec.reshape(1), ipsd0, index_to_cell_id,
                n_cells_solve=n, n_cells_global=n_cells_global, ipsdlim=ipsdlim, lrseeds=True)[0]
        rseeds = torch.cat(
            (torch.cat((draws, rseeds[:n, 2:]), dim=1), rseeds[n:]), dim=0
        )
    elif draws is None:
        # The CCPP fallback uses local block order and reads SEC only when active.
        safe_sec = torch.where(active, sec, torch.zeros_like(sec))
        draws = radiation_seed_draws(safe_sec.reshape(1), ipsd0, index_to_cell_id,
            n_cells_solve=n, n_cells_global=n_cells_global, ipsdlim=ipsdlim, lrseeds=False)[0]
    if draws.shape != (n, 2) or draws.dtype != torch.int32 or draws.device != device:
        raise ValueError("radiation seed draws must be [n_cells_solve, 2] device int32")
    streams = tuple(
        torch.cat((torch.where(active, draws[:, stream], old[:n]), old[n:]), dim=0)
        for stream, old in enumerate((previous.icsdsw, previous.icsdlw))
    )
    return RadiationSeeds(rseeds, *streams)
