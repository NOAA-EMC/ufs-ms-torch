# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""MERRA-2 aerosol climatology for the RRTMGP group (``iaer = 1011``: ``iaermdl = 1``, ``iaerflg = 11``).

Fortran sources (ccpp-physics of the v17 oracle tree):

* ``MP/Morrison_Gettelman/aerinterp.F90``: ``setindxaer`` (horizontal indices, at init),
  ``read_netfaer`` (monthly fields, float32 pressure sum) and ``aerinterpol`` (time and vertical
  interpolation to ``aer_nm``, called by ``GFS_phys_time_vary`` at every timestep with ``fhour``).
* ``Radiation/radiation_aerosols.f``: ``gocart_aerinit`` (``rd_gocart_luts``, ``optavg_gocart``)
  and ``setaer`` -> ``aer_property_gocart`` -> ``aeropt``; ``RRTMGP/rrtmgp_aerosol_optics.F90``
  maps the aerosol SW bands to the RRTMGP band order.

Host preparation (outside the numerics, like the RRTMGP coefficient tables) reads the files: the
horizontal interpolation of each needed month to the model columns is done once, in the source
order of operations (``aerinterpol`` forms ``tx1*(corner sum) + tx2*(corner sum)``, so the corner
sums of each month are exact precomputations). Runtime work (time weights from the date, vertical
interpolation to the layer pressure, optical properties) is torch tensor operations with masks;
it is differentiable with respect to the layer pressure, virtual temperature and humidity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .aerosol import WVNLW1, WVNLW2, WVNSW1, WVNSW2, _read_float_block, _set_spectrum

SPECIES = ("DU001", "DU002", "DU003", "DU004", "DU005", "SS001", "SS002", "SS003", "SS004", "SS005",
           "SO4", "BCPHOBIC", "BCPHILIC", "OCPHOBIC", "OCPHILIC")  # aerclm_def specname
NTRCAER = len(SPECIES)
LEVSAER = 72  # aerclm_def levsaer
AER_TIME = (15.5, 45., 74.5, 105., 135.5, 166., 196.5, 227.5, 258., 288.5, 319., 349.5, 380.5)
KAERBNDD, KAERBNDI, KRHLEV, KCM1, KCM2 = 61, 56, 36, 5, 10
RHLEV_GRT = (.00, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60, .65, .70, .75, .80, .81, .82,
             .83, .84, .85, .86, .87, .88, .89, .90, .91, .92, .93, .94, .95, .96, .97, .98, .99)
GRIDCOMP = ("DU", "SS", "SU", "BC", "OC")
NUM_RADIUS = (5, 5, 1, 2, 2)
RADIUS_LOWER = (1, 6, 11, 12, 14)
TRC_TO_AOD = (1, 5, 4, 2, 3)  # dust, soot, waso, suso, ssam -> aerodp(:, ktrc + 1)
NBDSW, NBDLW = 14, 16
WVN550 = 1.0e4 / 0.55
CON_RD = 287.05     # physcons con_rd (rrtmgp_aerosol_optics argument)
CON_G = 9.80665     # physcons con_g


@dataclass(frozen=True)
class Merra2Optics:
    """``optavg_gocart`` band means (aerosol band order: 14 SW then 16 LW) and 550 nm values."""

    ext_i: torch.Tensor      # [KCM1, 30]
    sca_i: torch.Tensor
    ssa_i: torch.Tensor
    asy_i: torch.Tensor
    ext_d: torch.Tensor      # [KRHLEV, KCM2, 30]
    sca_d: torch.Tensor
    ssa_d: torch.Tensor
    asy_d: torch.Tensor
    ext_i_550: torch.Tensor  # [KCM1]
    ext_d_550: torch.Tensor  # [KRHLEV, KCM2]
    rhlev: torch.Tensor      # [KRHLEV]
    nv_aod: int              # 0-based SW aerosol band that contains 550 nm


@dataclass(frozen=True)
class Merra2Climatology:
    """Monthly MERRA-2 fields at the model columns.

    ``fields[slot]`` is ``[NTRCAER + 1, LEVSAER, ncol]``: the 15 mixing ratios and, last, the
    pressure (Pa), on the aerosol levels from the surface up, each the source bilinear corner sum
    ``temij*a(i1,j1) + ddxy*a(i2,j2) + temiy*a(i1,j2) + temjx*a(i2,j1)``. ``slot_of_month[m]``
    (m = 1..12) is the slot of month m, -1 when the month was not prepared.
    """

    fields: torch.Tensor
    slot_of_month: torch.Tensor
    months: tuple[int, ...]


# ---------------------------------------------------------------------------
# host preparation
# ---------------------------------------------------------------------------
def months_needed(year: int, month: int, day: int, hour: int, *, hours: float) -> tuple[int, ...]:
    """The months ``aerinterpol`` reads for every forecast hour in ``[0, hours]`` from the given date."""
    import datetime as _dt
    start = _dt.datetime(year, month, day, hour)
    needed = set()
    for h in range(0, int(math.ceil(hours)) + 1):
        t = start + _dt.timedelta(hours=h)
        jdoy = t.timetuple().tm_yday
        n1, n2 = _bracketing_months(jdoy + t.hour / 24.0)
        needed.update((n1, n2))
    return tuple(sorted(needed))


def _bracketing_months(rjday: float) -> tuple[int, int]:
    if rjday < AER_TIME[0]:
        rjday += 365.0
    n2 = 13
    for j in range(2, 13):
        if rjday < AER_TIME[j - 1]:
            n2 = j
            break
    n1 = n2 - 1
    return n1, n2 - 12 if n2 > 12 else n2


def _setindxaer(points: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``setindxaer`` for one axis (ascending grid): 0-based indices and the weight ``ddx``/``ddy``."""
    n = grid.shape[0]
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("MERRA-2 grid axis must increase")
    upper = np.searchsorted(grid, points, side="right") + 1  # 1-based first grid value above the point, or n+1
    lower = np.maximum(upper - 1, 1)
    upper = np.minimum(upper, n)
    weight = np.ones(points.shape, dtype=np.float64)
    differ = upper != lower
    weight[differ] = (points[differ] - grid[lower[differ] - 1]) / (grid[upper[differ] - 1] - grid[lower[differ] - 1])
    return lower - 1, upper - 1, weight


def _month_at_columns(path: Path, corners) -> np.ndarray:
    """``read_netfaer`` (iflip = 1) and the bilinear corner sum: ``[NTRCAER + 1, LEVSAER, ncol]``."""
    from netCDF4 import Dataset
    j1, j2, i1, i2, temij, temiy, temjx, ddxy = corners
    def corner_sum(data):  # data [LEVSAER, lat, lon] float64
        return ((temij * data[:, j1, i1] + ddxy * data[:, j2, i2]) + temiy * data[:, j2, i1]) + temjx * data[:, j1, i2]
    with Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        delp = np.asarray(ds["DELP"][0], dtype=np.float32)  # [lev, lat, lon], top down
        levsw = delp.shape[0]
        flip = levsw - 1 - np.arange(LEVSAER)                  # klev = levsw - k + 1
        pres = np.zeros_like(delp)                          # kind_io4 sum, pres_tmp(:,1) = 0
        for k in range(1, levsw):
            pres[k] = pres[k - 1] + delp[k]
        out = np.empty((NTRCAER + 1, LEVSAER, temij.shape[0]), dtype=np.float64)
        out[NTRCAER] = corner_sum(pres[flip].astype(np.float64))
        del delp, pres
        for s, name in enumerate(SPECIES):
            values = np.asarray(ds[name][0], dtype=np.float32)[flip].astype(np.float64)
            values[(values < 0.0) | (values > 1.0)] = 1.0e-15
            out[s] = corner_sum(values)
    return out


def prepare_merra2_climatology(input_dir: str | Path, xlat_d: torch.Tensor, xlon_d: torch.Tensor,
                               months: Sequence[int]) -> Merra2Climatology:
    """Read ``aeroclim.mMM.nc`` for ``months`` and interpolate each to the columns (host side)."""
    from netCDF4 import Dataset
    input_dir = Path(input_dir)
    months = tuple(sorted(set(int(m) for m in months)))
    if not months or any(not 1 <= m <= 12 for m in months):
        raise ValueError("MERRA-2 preparation needs months in 1..12")
    with Dataset(input_dir / "aeroclim.m01.nc") as ds:  # read_aerdata takes lat/lon from m01
        aer_lat = np.asarray(ds["lat"][:], dtype=np.float64)
        aer_lon = np.asarray(ds["lon"][:], dtype=np.float64)
    lat = xlat_d.detach().to("cpu", torch.float64).numpy()
    lon = xlon_d.detach().to("cpu", torch.float64).numpy()
    j1, j2, ddy = _setindxaer(lat, aer_lat)
    i1, i2, ddx = _setindxaer(lon, aer_lon)
    temj = 1.0 - ddy
    temi = 1.0 - ddx
    corners = (j1, j2, i1, i2, temi * temj, temi * ddy, temj * ddx, ddx * ddy)
    fields = np.empty((len(months), NTRCAER + 1, LEVSAER, lat.shape[0]), dtype=np.float64)
    for slot, month in enumerate(months):
        fields[slot] = _month_at_columns(input_dir / f"aeroclim.m{month:02d}.nc", corners)
    slot_of_month = torch.full((13,), -1, dtype=torch.int64)
    for slot, month in enumerate(months):
        slot_of_month[month] = slot
    return Merra2Climatology(torch.from_numpy(fields), slot_of_month, months)


def _gocart_spectral_weights(iendwv: np.ndarray, iw1: int, iw2: int, flux: np.ndarray):
    """The band loop of ``gocart_aerinit`` for one radiation band: (start, end, weights, total)."""
    nbin = iendwv.shape[0]
    weights = np.zeros(nbin, dtype=np.float64)
    ii = 0
    while iw1 > iendwv[ii]:
        if ii == nbin - 1:
            break
        ii += 1
    start = ii
    total = 0.0
    partial = 0.0
    for iw in range(iw1, iw2 + 1):
        total = total + flux[iw]
        partial = partial + flux[iw]
        if iw == iendwv[ii]:
            weights[ii] = partial
            if ii < nbin - 1:
                partial = 0.0
                ii += 1
    if iw2 != iendwv[ii]:
        weights[ii] = partial
    return start, ii, weights, total


def _optavg(ext, sca, ssa, asy, start, end, weights, total):
    """One ``optavg_gocart`` band mean (source accumulation order)."""
    rbd = 1.0 / total
    sumk = sums = sumok = sumokg = sumreft = 0.0
    for ni in range(start, end + 1):
        sp = math.sqrt((1.0 - ssa[ni]) / (1.0 - ssa[ni] * asy[ni]))
        reft = (1.0 - sp) / (1.0 + sp)
        sumreft = sumreft + reft * weights[ni]
        sumk = sumk + ext[ni] * weights[ni]
        sums = sums + sca[ni] * weights[ni]
        sumok = sumok + ssa[ni] * weights[ni] * ext[ni]
        sumokg = sumokg + ssa[ni] * weights[ni] * ext[ni] * asy[ni]
    refb = sumreft * rbd
    asyb = sumokg / (sumok + 1.0e-10)
    ssab = 4.0 * refb / ((1.0 + refb) * (1.0 + refb) - asyb * ((1.0 - refb) * (1.0 - refb)))
    return sumk * rbd, sums * rbd, ssab, asyb


def _read_lut(path: Path, nbin: int, nradius: int):
    """``rd_gocart_luts`` for one file: lambda and [nbin, KRHLEV, nradius] bext, bsca, g."""
    lines = path.read_text().splitlines()
    index = 1
    lam, index = _read_float_block(lines, index, nbin)
    index += 1
    _, index = _read_float_block(lines, index, KRHLEV)
    blocks = []
    for _ in range(3):
        block = np.empty((nbin, KRHLEV, nradius), dtype=np.float64)
        for k in range(nradius):
            index += 1
            for j in range(KRHLEV):
                block[:, j, k], index = _read_float_block(lines, index, nbin)
        blocks.append(block)
    return lam, blocks


def load_merra2_optics(input_dir: str | Path, *, con_pi: float) -> Merra2Optics:
    """``gocart_aerinit`` (laswflg, lalwflg, lalw1bd = F): optics tables by aerosol band (host side)."""
    input_dir = Path(input_dir)
    solfwv, eirfwv = _set_spectrum(con_pi)
    # dust (rh independent), flip to increasing wavenumber
    lam, (bext, bsca, g) = _read_lut(input_dir / "optics_DU.dat", KAERBNDI, NUM_RADIUS[0])
    wavelength_du = (1.0e6 * lam)[::-1]
    id550 = None
    for j, w in enumerate(wavelength_du):
        if int(w * 100) == 55:
            id550 = j
    ext_i0 = bext[::-1, 0, :].copy()
    sca_i0 = bsca[::-1, 0, :].copy()
    ssa_i0 = np.where(ext_i0 != 0.0, sca_i0 / np.where(ext_i0 != 0.0, ext_i0, 1.0), 1.0)
    asy_i0 = g[::-1, 0, :].copy()
    ext_d0 = np.empty((KAERBNDD, KRHLEV, KCM2))
    sca_d0, ssa_d0, asy_d0 = np.empty_like(ext_d0), np.empty_like(ext_d0), np.empty_like(ext_d0)
    i550 = None
    for ib in range(1, len(GRIDCOMP)):
        lam, (bext, bsca, g) = _read_lut(input_dir / f"optics_{GRIDCOMP[ib]}.dat", KAERBNDD, NUM_RADIUS[ib])
        wavelength = (1.0e6 * lam)[::-1]
        for j, w in enumerate(wavelength):
            if int(w * 100) == 55:
                i550 = j
        ibeg = RADIUS_LOWER[ib] - KCM1 - 1
        for k in range(NUM_RADIUS[ib]):
            e, s = bext[::-1, :, k], bsca[::-1, :, k]
            ext_d0[:, :, ibeg + k] = e
            sca_d0[:, :, ibeg + k] = s
            ssa_d0[:, :, ibeg + k] = np.where(e != 0.0, s / np.where(e != 0.0, e, 1.0), 1.0)
            asy_d0[:, :, ibeg + k] = g[::-1, :, k]
    if id550 is None or i550 is None:
        raise ValueError("MERRA-2 optics tables lack the 550 nm wavelength")
    iendwv = np.asarray([int(10000.0 / w) for w in wavelength], dtype=np.int64)
    iendwv_du = np.asarray([int(10000.0 / w) for w in wavelength_du], dtype=np.int64)

    nb_all = NBDSW + NBDLW
    ext_i, sca_i, ssa_i, asy_i = (np.empty((KCM1, nb_all)) for _ in range(4))
    ext_d, sca_d, ssa_d, asy_d = (np.empty((KRHLEV, KCM2, nb_all)) for _ in range(4))
    nv_aod = 0
    bands = []
    for ib in range(NBDSW):
        if WVNSW2[ib] >= WVN550 >= WVNSW1[ib]:
            nv_aod = ib
        bands.append((int(round_half_away(WVNSW1[ib])), int(round_half_away(WVNSW2[ib])), solfwv))
    for ib in range(NBDLW):
        bands.append((int(round_half_away(WVNLW1[ib])), int(round_half_away(WVNLW2[ib])), eirfwv))
    for nb, (iw1, iw2, flux) in enumerate(bands):
        s0, e0, w0, t0 = _gocart_spectral_weights(iendwv_du, iw1, iw2, flux)
        s1, e1, w1, t1 = _gocart_spectral_weights(iendwv, iw1, iw2, flux)
        for nc in range(KCM1):
            ext_i[nc, nb], sca_i[nc, nb], ssa_i[nc, nb], asy_i[nc, nb] = _optavg(
                ext_i0[:, nc], sca_i0[:, nc], ssa_i0[:, nc], asy_i0[:, nc], s0, e0, w0, t0)
        for nc in range(KCM2):
            for nh in range(KRHLEV):
                ext_d[nh, nc, nb], sca_d[nh, nc, nb], ssa_d[nh, nc, nb], asy_d[nh, nc, nb] = _optavg(
                    ext_d0[:, nh, nc], sca_d0[:, nh, nc], ssa_d0[:, nh, nc], asy_d0[:, nh, nc], s1, e1, w1, t1)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64))
    return Merra2Optics(t(ext_i), t(sca_i), t(ssa_i), t(asy_i), t(ext_d), t(sca_d), t(ssa_d), t(asy_d),
                        t(ext_i0[id550]), t(ext_d0[i550]), torch.tensor(RHLEV_GRT, dtype=torch.float64), nv_aod)


def round_half_away(x: float) -> int:
    """Fortran ``nint``."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def merra2_to(value, device):
    """Move a prepared climatology or optics table to ``device`` (host preparation)."""
    if isinstance(value, Merra2Climatology):
        return Merra2Climatology(value.fields.to(device), value.slot_of_month.to(device), value.months)
    fields = {f: getattr(value, f) for f in value.__dataclass_fields__}
    return Merra2Optics(**{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in fields.items()})


# ---------------------------------------------------------------------------
# runtime (torch)
# ---------------------------------------------------------------------------
def _day_of_year(year: torch.Tensor, month: torch.Tensor, day: torch.Tensor) -> torch.Tensor:
    from .setup import _iw3jdn
    return _iw3jdn(year, month, day) - _iw3jdn(year, torch.ones_like(month), torch.ones_like(day)) + 1


def merra2_time_weights(climatology: Merra2Climatology, date: torch.Tensor):
    """``aerinterpol`` time part: slots of months n1, n2 and the weights ``tx1``, ``tx2``.

    ``date`` is the NCEP date (8 integers) at ``fhour`` (W3MOVDAT of idate by fhour).
    """
    vals = date.to(torch.int64)
    year, month, day, hour = vals[0], vals[1], vals[2], vals[4]
    rjday = _day_of_year(year, month, day).to(torch.float64) + hour.to(torch.float64) / 24.0
    rjday = torch.where(rjday < AER_TIME[0], rjday + 365.0, rjday)
    from .radiation_group import static_tensor  # cached device constant (radiation_group imports this module)

    times = static_tensor(AER_TIME, dtype=torch.float64, device=date.device)
    n2 = 2 + (times[1:12] <= rjday).sum()              # first j in 2..12 with rjday < aer_time(j), else 13
    n1 = n2 - 1
    n2 = torch.where(n2 > 12, n2 - 12, n2)              # wrapped before tx1, as in the source
    # Device gathers: indexing with a 0-dim tensor reads the index on the host.
    t2, t1 = times.index_select(0, (n2 - 1).reshape(1))[0], times.index_select(0, (n1 - 1).reshape(1))[0]
    tx1 = (t2 - rjday) / (t2 - t1)
    tx2 = 1.0 - tx1
    slot1 = climatology.slot_of_month.index_select(0, n1.reshape(1))[0]
    slot2 = climatology.slot_of_month.index_select(0, n2.reshape(1))[0]
    torch._assert_async((slot1 >= 0) & (slot2 >= 0), "MERRA-2 month not prepared for this date")
    return slot1, slot2, tx1, tx2


def merra2_aer_nm(climatology: Merra2Climatology, prsl: torch.Tensor, date: torch.Tensor) -> torch.Tensor:
    """``aerinterpol``: ``aer_nm`` ``[NTRCAER, nlay, ncol]`` at layer pressure ``prsl`` ``[nlay, ncol]`` (Pa)."""
    slot1, slot2, tx1, tx2 = merra2_time_weights(climatology, date)
    fields = climatology.fields
    aerpm = tx1 * fields.index_select(0, slot1.reshape(1))[0] + tx2 * fields.index_select(0, slot2.reshape(1))[0]
    aerpres = aerpm[NTRCAER]                                   # [LEVSAER, ncol]
    aerpm = aerpm[:NTRCAER]
    p = prsl.unsqueeze(0)                                       # [1, nlay, ncol]
    upper, lower = aerpres[:-1].unsqueeze(1), aerpres[1:].unsqueeze(1)
    inside = (p <= upper) & (p > lower)                         # [LEVSAER-1, nlay, ncol]
    k1 = torch.argmax(inside.to(torch.int8), dim=0)             # first k (source exits the k loop)
    k2 = k1 + 1
    p1 = aerpres.gather(0, k1)
    p2 = aerpres.gather(0, k2)
    bottom = prsl >= aerpres[0:1]
    top = prsl <= aerpres[LEVSAER - 1:LEVSAER]
    interior = ~(bottom | top)
    tem = 1.0 / torch.where(interior, p1 - p2, torch.ones_like(p1))
    tem1 = (prsl - p2) * tem
    tem2 = (p1 - prsl) * tem
    ncol = prsl.shape[1]
    nlay = prsl.shape[0]
    g1 = aerpm.gather(1, k1.unsqueeze(0).expand(NTRCAER, nlay, ncol))
    g2 = aerpm.gather(1, k2.unsqueeze(0).expand(NTRCAER, nlay, ncol))
    middle = g1 * tem1 + g2 * tem2
    return torch.where(bottom.unsqueeze(0), aerpm[:, 0:1].expand(NTRCAER, nlay, ncol),
                       torch.where(top.unsqueeze(0), aerpm[:, LEVSAER - 1:LEVSAER].expand(NTRCAER, nlay, ncol),
                                   middle))


def _layer_mass_path(aer_nm, p_lev, p_lay, tv_lay):
    """``setaer`` layer thickness and ``aer_property_gocart`` dry mass per area (``cm`` before max)."""
    prsi = p_lev * 0.01
    prsl = p_lay * 0.01
    rovg = 0.001 * CON_RD / CON_G
    nlay = prsl.shape[0]
    prsln = torch.cat((torch.log(prsi[:nlay]), torch.log(prsl[nlay - 1:nlay])), dim=0)
    dz = rovg * (prsln[:-1] - prsln[1:]) * tv_lay
    dz = torch.cat((dz[:-1], 2.0 * dz[-1:]), dim=0)
    dz1 = 1000.0 * dz
    rho = (100.0 * prsl) / (CON_RD * tv_lay)
    aerms = aer_nm * rho
    # A tensor bound avoids MPS's colliding cache keys for small scalar clamps.
    zero = torch.zeros((), dtype=aerms.dtype, device=aerms.device)
    return torch.clamp_min(aerms, zero) * dz1                  # [NTRCAER, nlay, ncol]


def _rh_interpolation(optics: Merra2Optics, relhum: torch.Tensor):
    count = (optics.rhlev.reshape(-1, 1, 1) < relhum.unsqueeze(0)).sum(0)  # ih2 - 1 (0-based)
    ih1 = torch.clamp_min(count - 1, 0)
    ih2 = torch.clamp_max(count, KRHLEV - 1)
    r1 = optics.rhlev[ih1]
    r2 = optics.rhlev[ih2]
    same = ih1 == ih2
    rdrh = torch.where(same, torch.zeros_like(relhum), (relhum - r1) / torch.where(same, torch.ones_like(r2), r2 - r1))
    return ih1, ih2, rdrh


def _aerosol_preparation(optics, aer_nm, p_lev, p_lay, tv_lay, relhum):
    return (_layer_mass_path(aer_nm, p_lev, p_lay, tv_lay),
            *_rh_interpolation(optics, relhum))


def _rh_table(table: torch.Tensor, ih1, ih2, rdrh):
    """``t(ih1) + rdrh*(t(ih2) - t(ih1))`` for a table ``[KRHLEV, nband]`` -> ``[nband, nlay, ncol]``."""
    a = table[ih1].permute(2, 0, 1)
    b = table[ih2].permute(2, 0, 1)
    return a + rdrh.unsqueeze(0) * (b - a)


def _band_view(table, bands):
    """Select static contiguous band runs without a device index buffer."""
    if not bands:
        return table[..., :0]
    if any(band < 0 or band >= table.shape[-1] for band in bands):
        raise IndexError("aerosol band is outside the coefficient table")
    pieces = []
    start = previous = bands[0]
    for band in bands[1:]:
        if band != previous + 1:
            pieces.append(table[..., start:previous + 1])
            start = band
        previous = band
    pieces.append(table[..., start:previous + 1])
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)


def merra2_band_optics(optics: Merra2Optics, aer_nm, p_lev, p_lay, tv_lay, relhum, bands: Sequence[int],
                      *, preparation=None):
    """``aeropt`` for the listed aerosol bands (0-based, 0-13 SW, 14-29 LW): tau, ssa, g ``[nband, nlay, ncol]``."""
    device = p_lay.device
    if preparation is None:
        preparation = _aerosol_preparation(optics, aer_nm, p_lev, p_lay, tv_lay, relhum)
    cm, ih1, ih2, rdrh = preparation
    shape = (len(bands),) + p_lay.shape
    zero = torch.zeros(shape, dtype=p_lay.dtype, device=device)
    sum_tau = sum_ssa = sum_asy = zero
    ssa = asy = zero
    independent = tuple(_band_view(getattr(optics, name), bands)
                        for name in ("ext_i", "sca_i", "ssa_i", "asy_i"))
    dependent = tuple(_band_view(getattr(optics, name), bands)
                      for name in ("ext_d", "sca_d", "ssa_d", "asy_d"))
    groups = [[(m,) + tuple(table[m].reshape(-1, 1, 1) for table in independent)
               for m in range(KCM1)]]
    for ntrc in range(1, len(GRIDCOMP)):
        group = []
        for nbin in range(NUM_RADIUS[ntrc]):
            m1 = RADIUS_LOWER[ntrc] + nbin - 1   # 0-based species
            m = m1 - KCM1
            group.append((m1,) + tuple(_rh_table(table[:, m], ih1, ih2, rdrh) for table in dependent))
        groups.append(group)
    for group in groups:
        ext1 = sca1 = ssa1 = asy1 = zero
        for m1, ext, sca, ssa_t, asy_t in group:
            c = cm[m1].unsqueeze(0)
            ext1 = ext1 + c * ext
            sca1 = sca1 + c * sca
            ssa1 = ssa1 + c * ext * ssa_t
            asy1 = asy1 + c * sca * asy_t
        tau = ext1
        # ssa and asy keep the previous species' values where this species has no extinction
        ssa = torch.where(ext1 > 0.0, torch.clamp_max(ssa1 / torch.where(ext1 > 0.0, ext1, 1.0), 1.0), ssa)
        asy = torch.where(sca1 > 0.0, torch.clamp_max(asy1 / torch.where(sca1 > 0.0, sca1, 1.0), 1.0), asy)
        sum_tau = sum_tau + tau
        sum_ssa = sum_ssa + tau * ssa
        sum_asy = sum_asy + tau * ssa * asy
    ssa_out = torch.where(sum_tau > 0.0, sum_ssa / torch.where(sum_tau > 0.0, sum_tau, 1.0), torch.ones_like(sum_tau))
    asy_out = torch.where(sum_ssa > 0.0, sum_asy / torch.where(sum_ssa > 0.0, sum_ssa, 1.0), zero)
    return sum_tau, ssa_out, asy_out


def merra2_sw_optics(optics, aer_nm, p_lev, p_lay, tv_lay, relhum):
    """SW optics in the RRTMGP band order (``rrtmgp_aerosol_optics``: last aerosol band first)."""
    return merra2_band_optics(optics, aer_nm, p_lev, p_lay, tv_lay, relhum,
                              (NBDSW - 1,) + tuple(range(NBDSW - 1)))


def merra2_lw_optics(optics, aer_nm, p_lev, p_lay, tv_lay, relhum, *, preparation=None):
    return merra2_band_optics(optics, aer_nm, p_lev, p_lay, tv_lay, relhum, tuple(range(NBDSW, NBDSW + NBDLW)),
                              preparation=preparation)


def merra2_aod550(optics: Merra2Optics, aer_nm, p_lev, p_lay, tv_lay, relhum, *, preparation=None):
    """``aerodp`` ``[6, ncol]`` (total, dust, soot, waso, suso, salt) and ``ext550`` ``[nlay, ncol]``."""
    if preparation is None:
        preparation = _aerosol_preparation(optics, aer_nm, p_lev, p_lay, tv_lay, relhum)
    cm, ih1, ih2, rdrh = preparation
    zero = torch.zeros_like(p_lay)
    spcodp = [zero] * 5
    ext1 = zero
    for m in range(KCM1):
        ext1 = ext1 + cm[m] * optics.ext_i_550[m]
    spcodp[0] = ext1
    total = ext1
    for ntrc in range(1, len(GRIDCOMP)):
        ext1 = zero
        for nbin in range(NUM_RADIUS[ntrc]):
            m1 = RADIUS_LOWER[ntrc] + nbin - 1
            table = optics.ext_d_550[:, m1 - KCM1]
            a, b = table[ih1], table[ih2]
            ext1 = ext1 + cm[m1] * (a + rdrh * (b - a))
        spcodp[TRC_TO_AOD[ntrc] - 1] = ext1
        total = total + ext1
    columns = [total] + spcodp
    aerodp = []
    for field in columns:
        acc = torch.zeros_like(p_lay[0])
        for k in range(p_lay.shape[0]):
            acc = acc + field[k]
        aerodp.append(acc)
    return torch.stack(aerodp), total


__all__ = ["Merra2Climatology", "Merra2Optics", "load_merra2_optics", "merra2_aer_nm", "merra2_aod550",
           "merra2_lw_optics", "merra2_sw_optics", "merra2_to", "months_needed", "prepare_merra2_climatology"]
