# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""GFS v17 gravity-wave drag: ugwpv1_gsldrag (+ _post), in native Torch.

ugwpv1_gsldrag runs the GSL drag suite in its PSL form (drag_suite_psl: large-scale GWD and
blocking, small-scale GWD, turbulent form drag; gwd_small_scale.py) and the UGWP v1
non-stationary waves: the GEOS-5 2020 latitude flux (slat_geos5_2020), its precipitation
modulation (ngwflux_update; the limb-sounder flux tau_amf of the tau file enters with weight
w_nomerra = 0) and the v1 spectral solver with eddy diffusion (cires_ugwpv1_ngw_solv2,
knob_ugwp_solver = 2, knob_ugwp_version = 1, knob_ugwp_dokdis = 2, knob_ugwp_doheat = 1).

Setup (prepare_ugwpv1) reads the namelist knobs, the tau file and the reference pressures on
the host, once, with the source's scalar arithmetic and libm functions; the runtime holds only
tensor operations: branches are masks, per-column launch levels and recurrences are masked
level loops, and the eddy-diffusion substep count is a per-column mask under a static cap.

Two static loop bounds come from ``runtime_policy.ugwp_v1_caps`` in the coupled binding: the
substeps unrolled (``nstab_cap``) and the first level of the wave loop (``kstart``: the reference
start ``wave_loop_start`` derives from the setup, or layer 3 as the source). Levels below a column's launch level and substeps
past its count are exact no-ops, so the result equals the full loops while every column fits; a
column that does not sets its overflow bit and the interval driver reruns the cycle with the
bound raised (``ufs_physics_common.compaction``).
"""

import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from ufs_physics_common.precision import working_power

from ufs_physics_common.compaction import note_overflow
from ufs_physics_common import compile_policy

from .containers import static_tensor
from .gwd_small_scale import drag_suite_psl
from .runtime_policy import UGWP_V1_NSTAB_MAX, ugwp_v1_caps

# Physical constants as UFS physcons holds them (ugwpv1_gsldrag_init copies them into ugwp_common).
CON_PI = 4.0 * math.atan(1.0)
CON_G, CON_CP, CON_RD, CON_RV = 9.80665, 1004.6, 287.05, 461.50
CON_FVIRT = CON_RV / CON_RD - 1.0
CON_OMEGA, CON_P0 = 7.2921e-5, 1.01325e5

# cires_ugwpv1_module parameters consumed by the v1 solver.
MAX_KDIS = 450.0
MAX_AXYZ = 450.e-5
MAX_EPS = MAX_KDIS * 4.e-4
MAXDTDT = MAX_EPS * 1.e-3
DKED_MIN, DKED_MAX = 0.01, MAX_KDIS
IPR_KTGW = 1. / 3.
DTURB_MAX = 100.0
ZSP_GW = 106.5e3
# Largest eddy-diffusion substep count per column the port unrolls (runtime_policy caps the loop below it).
NSTAB_MAX = UGWP_V1_NSTAB_MAX
# The wave loop's reference start: the launch layer of a column whose layer pressures are half the
# reference ones (x1.10242: launch layers 25-33 over 25 coupled cycles, reference launch layer 31,
# smallest pressure ratio to the reference 0.58 at layers 19-25).
KSTART_PRESSURE_RATIO = 0.5
_HPSKM = 7000. * 1.e-3  # ugwp_common hpscale in km
_regional_compile = False


def _reference_kvg(p: float) -> float:
    """init_global_gwdis kvg (molecular viscosity plus the 130 km bump) at reference pressure p."""
    zkm = -_HPSKM * math.log(p / CON_P0)
    vumol = 2.e-5 * math.exp(zkm / 7.0)
    ae2 = -((zkm - 130.) / 30.) ** 2
    return vumol + 100. * math.exp(ae2)

_STRUCTURAL = (("knob_ugwp_solver", 1, 2), ("knob_ugwp_version", 1, 1), ("knob_ugwp_dokdis", 0, 2),
               ("knob_ugwp_doheat", 1, 1), ("knob_ugwp_doaxyz", 1, 1))


def _knob(namelist: Mapping[str, Any], name: str, default, kind):
    value = namelist.get(name, default)
    if isinstance(value, torch.Tensor) or type(value) not in (int, float, bool) and kind is not str:
        raise TypeError(f"UGWP v1 namelist {name} requires a static value")
    if kind is str:
        return str(value)
    if kind is int and type(value) is not int:
        raise TypeError(f"UGWP v1 namelist {name} requires an integer")
    value = kind(value)
    if not math.isfinite(value):
        raise ValueError(f"UGWP v1 namelist {name} must be finite")
    return value


def _knob4(namelist: Mapping[str, Any], name: str, default, kind):
    value = namelist.get(name, default)
    if type(value) not in (list, tuple) or len(value) != 4:
        raise TypeError(f"UGWP v1 namelist {name} requires four static values")
    return tuple(_knob({name: v}, name, d, kind) for v, d in zip(value, default))


def read_tau_file(path: Path) -> dict[str, tuple]:
    """read_tau_amf: LATS, DAYS and ABSMF(lat, days) of ugwp_limb_tau.nc, as float64 tuples."""
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        lats = tuple(float(x) for x in ds["LATS"][:].tolist())
        days = tuple(float(x) for x in ds["DAYS"][:].tolist())
        table = ds["ABSMF"][:]  # (days, lat) in the file: Fortran tau_limb(lat, days)
        if table.shape != (len(days), len(lats)) or not lats or not days:
            raise ValueError("ugwp tau file requires ABSMF(days, lat)")
        absmf = tuple(tuple(float(x) for x in row) for row in table.tolist())
    return {"tau_lats": lats, "tau_days": days, "tau_limb": absmf}


def prepare_ugwpv1(ugwp_namelist: Mapping[str, Any], *, ak: tuple, bk: tuple, levs: int,
                   tau_file: Path) -> dict[str, Any]:
    """cires_ugwpv1_init with initsolv_wmsdis (version 1) and init_global_gwdis, host scalars.

    ak/bk are the host's reference interface pressures (bk = 0 on MPAS). Only the v17 solver
    configuration is ported; other structural knobs raise.
    """
    if "launch_level" in ugwp_namelist:
        raise NotImplementedError("the UGWP v1 namelist has no launch_level (the source read fails)")
    for name, default, selected in _STRUCTURAL:
        if _knob(ugwp_namelist, name, default, int) != selected:
            raise NotImplementedError(f"UGWP v1 port requires {name}={selected}")
    stoch = _knob4(ugwp_namelist, "knob_ugwp_stoch", (0, 0, 0, 0), int)
    if stoch[1] != 0:
        raise NotImplementedError("stochastic UGWP v1 non-stationary waves are not ported")
    wvspec = _knob4(ugwp_namelist, "knob_ugwp_wvspec", (1, 32, 32, 32), int)
    azdir = _knob4(ugwp_namelist, "knob_ugwp_azdir", (2, 4, 4, 4), int)
    effac = _knob4(ugwp_namelist, "knob_ugwp_effac", (1., 1., 1., 1.), float)
    if wvspec[1] < 2 or azdir[1] < 1:
        raise NotImplementedError("UGWP v1 solver requires nwav >= 2 and nazd >= 1")
    palaunch = _knob(ugwp_namelist, "knob_ugwp_palaunch", 500.e2, float)
    if palaunch > 900.e2:
        raise ValueError("cires_ugwpv1_init: unrealistic value of knob_ugwp_palaunch")
    nslope = float(_knob(ugwp_namelist, "knob_ugwp_nslope", 1, int))
    lzmax = _knob(ugwp_namelist, "knob_ugwp_lzmax", 12.5e3, float)
    lzmin = _knob(ugwp_namelist, "knob_ugwp_lzmin", 1.5e3, float)
    lzstar = _knob(ugwp_namelist, "knob_ugwp_lzstar", 2.0e3, float)
    taumin = _knob(ugwp_namelist, "knob_ugwp_taumin", 0.25e-3, float)
    tauamp = _knob(ugwp_namelist, "knob_ugwp_tauamp", 7.75e-3, float)
    lhmet = _knob(ugwp_namelist, "knob_ugwp_lhmet", 200.e3, float)
    if len(ak) != levs + 1 or len(bk) != levs + 1:
        raise ValueError("UGWP v1 setup requires levs+1 reference interface coefficients")
    pi = CON_PI
    pi2 = 2. * pi
    # ugwp_common, as ugwpv1_gsldrag_init sets it.
    grav, cpd, rd = CON_G, CON_CP, CON_RD
    rgrav = 1.0 / grav
    rdi = 1.0 / rd
    rcpd = 1. / cpd
    gor = grav / rd
    grcp = grav * rcpd
    rcpdl = cpd * rgrav
    grav2cpd = grav * grcp
    hpscale = 7000.
    rhp = 1. / hpscale
    mkzmin = pi2 / 80.0e3
    mkzmax = pi2 / 500.
    # initsolv_wmsdis, version 1.
    nwav, nazd = wvspec[1], azdir[1]
    kxw = pi2 / lhmet
    zang = pi2 / float(nazd)
    cosang = tuple(math.cos((i - 1) * zang) for i in range(1, nazd + 1))
    sinang = tuple(math.sin((i - 1) * zang) for i in range(1, nazd + 1))
    zms = pi2 / lzstar
    bnfix = 6.28 / 300.
    dlzmet = (lzmax - lzmin) / float(nwav - 1)
    zci = tuple((lzmin + (inc - 1) * dlzmet) / (pi2 / bnfix) for inc in range(1, nwav + 1))
    zdx = (zci[-1] - zci[0]) / float(nwav - 1)
    # init_global_gwdis on pmb = ak + p0*bk.
    pmb = [a + CON_P0 * b for a, b in zip(ak[:levs], bk[:levs])]
    if any(not (p > 0.0) for p in pmb):
        raise ValueError("UGWP v1 setup requires positive reference pressures")
    pa_alp, tau_alp = 750., 10.
    rtau_fv3 = 1. / 86400. / tau_alp
    pih_dlog = (.5 * pi) / math.log(pa_alp / pmb[levs - 1])
    kvg, kion = [], []
    for k in range(levs):
        kvg.append(_reference_kvg(pmb[k]))
        ion = 1. / 86400. / 30.
        if pmb[k] <= pa_alp:
            s = math.sin(pih_dlog * math.log(pa_alp / pmb[k]))
            ion = ion + rtau_fv3 * (s * s)
        kion.append(ion)
    kvg.append(kvg[-1])
    kion.append(kion[-1])
    setup = {
        "nwav": nwav, "nazd": nazd, "zci": zci, "zdci": (zdx,) * nwav, "cosang": cosang, "sinang": sinang,
        "zms": zms, "nslope3": nslope + 3.0, "v_kxw": kxw, "rv_kxw": 1. / kxw, "gw_eff": effac[1],
        "tau_min": taumin, "tamp_mpa": tauamp, "psrc": palaunch, "ucrit": 2.e-2 / mkzmax,
        "mkzmin": mkzmin, "mkz2min": mkzmin * mkzmin, "kvg": tuple(kvg), "kion": tuple(kion),
        "rgrav": rgrav, "rdi": rdi, "rcpd": rcpd, "gor": gor, "rcpdl": rcpdl, "grav2cpd": grav2cpd,
        "bnv2min": (pi2 / 1800.) * (pi2 / 1800.), "bnv2max": (pi2 / 30.) * (pi2 / 30.),
        "rh4": 0.25 * rhp, "omega2": 2. * CON_OMEGA, "gipr": grav * IPR_KTGW, "sc2": 30. * 30.,
    }
    setup.update(read_tau_file(Path(tau_file)))
    return setup


def wave_loop_start(setup: Mapping[str, Any]) -> int:
    """The wave loop's reference start (``kstart``): below the first layer whose reference top interface
    lies above psrc / KSTART_PRESSURE_RATIO (ngw_solv2 launches below the first layer under psrc), at
    least layer 3 and at most levs-2.

    Read from the setup alone, so every setup (also one pickled before this start existed) gets it:
    kvg(k) is init_global_gwdis's kvg at the reference pressure ak(k) + p0 bk(k), the top interface of
    layer k-1 for k = 2..levs, and kvg rises strictly as the pressure falls, so "top of layer k-1 above
    the edge" reads "kvg(k) >= kvg at the edge". A tie of rounded values (never seen) only lowers the
    start; a start that is too high or too low costs time, not bits (the overflow rerun).
    """
    kvg = setup["kvg"]
    levs = len(kvg) - 1
    edge = _reference_kvg(setup["psrc"] / KSTART_PRESSURE_RATIO)
    kfirst = levs
    for k in range(1, levs):
        if kvg[k] >= edge:
            kfirst = k
            break
    return min(max(kfirst - 1, 3), levs - 2)


def tau_amf_interp(xlat_d: torch.Tensor, forecast_day: torch.Tensor, setup: Mapping[str, Any]) -> torch.Tensor:
    """cires_indx_ugwp (latitude weights) and tau_amf_interp (day weights) of the tau file."""
    lats = static_tensor(setup["tau_lats"], dtype=xlat_d.dtype, device=xlat_d.device)
    days = static_tensor(setup["tau_days"], dtype=xlat_d.dtype, device=xlat_d.device)
    table = static_tensor(setup["tau_limb"], dtype=xlat_d.dtype, device=xlat_d.device)  # [days, lat]
    nlat, nday = lats.numel(), days.numel()
    below = xlat_d[:, None] < lats[None, :]
    j2 = torch.where(below.any(dim=1), below.to(torch.int8).argmax(dim=1), nlat - 1)
    j1 = torch.clamp_min(j2 - 1, 0)
    same = j1 == j2
    span = torch.where(same, 1.0, lats[j2] - lats[j1])
    w2 = torch.where(same, 1.0, (xlat_d - lats[j1]) / span)
    w1 = 1.0 - w2
    later = forecast_day < days
    torch._assert_async(later.any(), "tau_amf_interp: forecast day beyond the tau file days")
    it2 = torch.clamp_max(later.to(torch.int8).argmax(), nday - 1)
    it1 = torch.clamp_min(it2 - 1, 0)
    # Device gathers: indexing with a 0-dim tensor reads the index on the host.
    d1, d2 = days.index_select(0, it1.reshape(1))[0], days.index_select(0, it2.reshape(1))[0]
    tw2 = (forecast_day - d1) / (d2 - d1)
    tw1 = 1.0 - tw2
    row1, row2 = table.index_select(0, it1.reshape(1))[0], table.index_select(0, it2.reshape(1))[0]
    tx1 = row1[j1] * w1 + row1[j2] * w2
    tx2 = row2[j1] * w1 + row2[j2] * w2
    return tx1 * tw1 + tw2 * tx2


def slat_geos5_2020(xlat_d: torch.Tensor, amplitude: float) -> torch.Tensor:
    """GEOS-5/MERRA-2 2020 latitude shape of the non-stationary launch flux."""
    lat = xlat_d.abs()
    def gauss(center, width):
        tem = (lat - center) / width
        return torch.exp(-tem * tem)
    tropical = torch.where(lat <= 3.0, 1.25, 1.25 * gauss(3.0, 12.5))
    flat = torch.where(lat < 15.3, tropical, torch.where(lat < 31.0, 0.25,
        torch.where(lat < 60.0, 1.0 * gauss(60.0, 23.0), 1.0 * gauss(60.0, 25.0))))
    south = xlat_d <= -31.0
    flat = torch.where(south & (lat >= 31.0) & (lat < 60.0), 1.0 * gauss(60.0, 23.0), flat)
    flat = torch.where(south & (lat >= 60.0), 1.0 * gauss(60.0, 30.0), flat)
    return amplitude * flat


def ngwflux_update(tau_ngw: torch.Tensor, tau_ddd: torch.Tensor, xlat_d: torch.Tensor,
                   rain: torch.Tensor) -> torch.Tensor:
    """Precipitation modulation of the launch flux (w_merra = 1, w_rain = 1, metoum_rain = 0)."""
    tau_3dt = tau_ngw * 1.0 + 0.0 * tau_ddd
    wlat = xlat_d.abs()
    modulated = (rain > 0.0) & (wlat <= 41.0) & (rain > 1.e-5)
    rw1 = 0.75 * (wlat / 41.0)
    rw2 = 1. - rw1
    tau_rain = tau_3dt * rw1 + rw2 * 1.e-3 * torch.clamp_max(rain, 8.e-5) / 1.e-5
    tau_rain = tau_3dt * (1. - 1.) + 1. * tau_rain
    tau_rain = torch.where(tau_rain < 0.5 * tau_3dt, 0.5 * tau_3dt, tau_rain)
    tau_rain = torch.where(tau_rain > 2. * tau_3dt, 2. * tau_3dt, tau_rain)
    return torch.where(modulated, tau_rain, tau_3dt)


def _clip(value: torch.Tensor, limit: float) -> torch.Tensor:
    """if (abs(x) >= limit) x = sign(limit, x); the bound is a float64 tensor (where of two Python
    scalars is float32, P104)."""
    bound = value.new_full((), limit)
    return torch.where(value.abs() >= limit, torch.where(value < 0.0, -bound, bound), value)


def _wave_level(flux, vueff, ksrc, level, umfp, bn2_next, vzmet, fden_next,
                wrk3, bn2, ion, zci, c2f2, ucrit, zdci, v_kxw, mkz2min,
                previous, fpu_seed, heat_seed, vueff_next):
    """One complete ordered wave level, including its masked output updates."""
    active = level >= ksrc
    act3 = active[None, :, None]
    zci3 = zci[:, None, None]
    flux_m = flux
    live = ~(flux_m < 0.0) & act3
    v_cdp = zci3 - umfp[None, :, :]
    cdf2 = v_cdp * v_cdp - c2f2[None, :, None]
    absorb = live & ((v_cdp <= ucrit[None, :, None]) | (cdf2 <= 0.0))
    prop = live & ~absorb
    v_wdp = v_kxw * v_cdp
    kzw2 = (bn2_next[None, :, None] - v_wdp * v_wdp) / torch.where(prop, cdf2, 1.0)
    good = prop & (kzw2 > mkz2min)
    kzw2_s = torch.where(good, kzw2, 1.0)
    v_kzw = torch.sqrt(kzw2_s)
    cd = torch.sqrt(torch.where(good, cdf2, 1.0))
    wdp = v_kxw * cd
    v_wdi = kzw2_s * vueff[None, :, None] + ion
    v_wdpc = torch.sqrt(wdp * wdp + v_wdi * v_wdi)
    v_kzi = v_kzw * v_wdi / v_wdpc
    ze1 = v_kzi * vzmet[None, :, None]
    strong = ze1 >= 1.e-2
    expdis = torch.where(strong, torch.clamp_min(torch.exp(-ze1), 0.01),
                         flux.new_full((), 1.0) / (1. + torch.where(strong, 0.0, ze1)))
    fdis = torch.where(good, flux * expdis * 1.0, 0.0)
    fluxs = torch.where(good, fden_next[None, :, None] * cdf2 * 1.0, 0.0)
    new = torch.where(fdis - fluxs > 0.0, fluxs, fdis)
    new = torch.where(flux_m - new < 0.0, flux_m, new)
    flux = torch.where(good, new, torch.where(live, -1.0, flux_m))
    diff = flux_m - flux
    vmdiff = torch.where(diff > 0.0, diff, 0.0)
    vc = torch.where(vmdiff <= 0.0, flux_m, flux)
    heat = wrk3[None, :, None] * cd
    fpu_new = heat_sum = flux.new_zeros(flux.shape[1:])
    for inc in range(flux.shape[0]):
        keep = good[inc]
        fpu_new = torch.where(keep, fpu_new + vc[inc] * zdci[inc], fpu_new)
        heat_sum = torch.where(keep, heat_sum + heat[inc] * zdci[inc] * vmdiff[inc], heat_sum)
    sums = torch.stack((fpu_new, heat_sum), dim=-1)
    fpu_new, heat_sum = sums[..., 0], sums[..., 1]
    total = flux.new_zeros(flux.shape[1])
    for iaz in range(flux.shape[2]):
        total = total + heat_sum[:, iaz].abs()
    tx = total / bn2
    mixing = torch.clamp_max(torch.clamp_min(tx, DKED_MIN), DKED_MAX)
    fpu_new = torch.where(fpu_new > previous, previous, fpu_new)
    return (flux, torch.where(active[:, None], fpu_new, fpu_seed),
            torch.where(active[:, None], heat_sum, heat_seed),
            torch.where(active, mixing + vueff_next, vueff_next))


def _eddy_diffusion(uold, vold, told, bdif, bt_dif, adif, cdif, dtdif, inner, nstab,
                    *, nstab_cap: int):
    """The complete ordered diffusion loop over all columns and levels."""
    levs = uold.shape[1] - 1
    prev = lambda x: torch.cat((x[:, :1], x[:, :-1]), dim=1)  # noqa: E731
    nxtk = lambda x: torch.cat((x[:, 1:], x[:, -1:]), dim=1)  # noqa: E731
    for ist in range(1, nstab_cap + 1):
        run = (ist <= nstab)[:, None]
        unew = uold * bdif + prev(uold) * adif + nxtk(uold) * cdif
        vnew = vold * bdif + prev(vold) * adif + nxtk(vold) * cdif
        tnew = told * bt_dif + (prev(told) * adif + nxtk(told) * cdif) * IPR_KTGW
        un = torch.where(inner, unew * dtdif[:, None], uold)
        vn = torch.where(inner, vnew * dtdif[:, None], vold)
        tn = torch.where(inner, tnew * dtdif[:, None], told)
        un = torch.cat((un[:, :levs], un[:, levs - 1:levs]), dim=1)
        vn = torch.cat((vn[:, :levs], vn[:, levs - 1:levs]), dim=1)
        tn = torch.cat((tn[:, :levs], tn[:, levs - 1:levs]), dim=1)
        uold = torch.where(run, un, uold)
        vold = torch.where(run, vn, vold)
        told = torch.where(run, tn, told)
    return uold, vold, told


def _ngw_prepare(fields: Mapping[str, torch.Tensor], tau_ngw: torch.Tensor, *, dtp: torch.Tensor,
              setup: Mapping[str, Any], nstab_cap: int = NSTAB_MAX, kstart: int = 3) -> dict[str, torch.Tensor]:
    """cires_ugwpv1_ngw_solv2 over a column batch; returns pdudt, pdvdt, pdtdt, dked, zngw.

    Arrays carry the source's one-based level numbers: column j of an interface array is
    interface j (1..levs+1), column j of a layer array is layer j (1..levs); column 0 is padding.
    The wave loop runs layers kstart..levs and the eddy diffusion nstab_cap substeps; a column
    launching below kstart or needing more substeps sets its overflow bit (gwd.ugwp_v1_kstart,
    gwd.ugwp_v1_nstab). The defaults are the source's loops.
    """
    atm, aum, avm, aqm = (fields[n] for n in ("t1", "u1", "v1", "q1"))
    prsl, prsi = fields["prsl"], fields["prsi"]  # the source reads prslk only into unused locals
    zmet, zmeti = fields["zmet"], fields["zmeti"]
    ncol, levs = atm.shape
    dev, dt = atm.device, atm.dtype
    s = setup
    if type(nstab_cap) is not int or not 1 <= nstab_cap <= NSTAB_MAX or type(kstart) is not int \
            or not 3 <= kstart <= levs - 2:
        raise ValueError("UGWP v1 solver requires 1 <= nstab_cap <= 16 and 3 <= kstart <= levs-2")
    nwav, nazd = s["nwav"], s["nazd"]
    one = atm.new_full((), 1.0)
    pad = lambda x: torch.cat((x[:, :1], x), dim=1)  # noqa: E731  one-based layers 1..levs
    T, U, V, Q, PL, ZM = (pad(x) for x in (atm, aum, avm, aqm, prsl, zmet))
    PI = pad(prsi)  # interfaces 1..levs+1
    ZI = pad(zmeti)
    zci = static_tensor(s["zci"], dtype=dt, device=dev)
    zdci = s["zdci"]
    cosang, sinang = s["cosang"], s["sinang"]
    kvg = pad(static_tensor(s["kvg"], dtype=dt, device=dev)[None, :])  # interfaces 1..levs+1
    kion = pad(static_tensor(s["kion"], dtype=dt, device=dev)[None, :])
    # Rotational cut-off and the critical-level speed per column.
    tx1 = s["omega2"] * fields["sinlat"] * s["rv_kxw"]
    c2f2 = tx1 * tx1
    ucrit_max = torch.clamp_min(tx1.abs(), s["ucrit"])
    # Launch level: first layer above the launch pressure, at least layer 3.
    levels = torch.arange(1, levs + 1, device=dev)
    hit = prsl < s["psrc"]
    kfirst = torch.where(hit.any(dim=1), levels[hit.to(torch.int8).argmax(dim=1)], levs + 1)
    ksrc = torch.clamp_min(kfirst - 1, 3)
    zngw = ZM.gather(1, ksrc[:, None]).squeeze(1)
    taub_src = torch.clamp_min(tau_ngw, s["tau_min"])
    # Interface mean flow, jk = 2..levs (index jk), with ktop = levs+1 extrapolated.
    TV = T * (1. + CON_FVIRT * Q)
    tvc, tvm = TV[:, 2:], TV[:, 1:-1]
    two = atm.new_full((), 2.0)  # scalar/tensor division would multiply by a reciprocal (P27)
    zthm = two / (tvc + tvm)
    uint = 0.5 * (U[:, 1:-1] + U[:, 2:])
    vint = 0.5 * (V[:, 1:-1] + V[:, 2:])
    tint = 0.5 * (tvc + tvm)
    rhomid = PL[:, 2:] * s["rdi"] / T[:, 2:]
    rhoint = PI[:, 2:levs + 1] * s["rdi"] * zthm
    dz_meti = ZI[:, 3:] - ZI[:, 2:levs + 1]
    dz_met = ZM[:, 2:] - ZI[:, 1:levs]
    zdelm = one / dz_met
    bn2 = s["grav2cpd"] * zthm * (1.0 + s["rcpdl"] * (tvc - tvm) * zdelm)
    bn2 = torch.clamp_min(torch.clamp_max(bn2, s["bnv2max"]), s["bnv2min"])
    wrk3 = one / dz_meti / rhomid
    uz, vz = U[:, 2:] - U[:, 1:-1], V[:, 2:] - V[:, 1:-1]
    shr2 = torch.clamp_min(uz * uz + vz * vz, 1.0) * zdelm * zdelm
    zgrow = torch.exp(ZM[:, 2:] * s["rh4"])
    w1 = one / (1. + 5 * (bn2 / shr2))
    kamp = torch.sqrt(shr2) * s["sc2"] * zgrow
    vueff = torch.clamp_max(torch.clamp_min(kamp * w1 * w1, DKED_MIN), DKED_MAX) + kvg[:, 2:levs + 1]
    akt = atm.new_full((), s["gipr"]) / tvc
    # Full one-based interface arrays: index 0 and 1 copy index 2 (never consumed), levs+1 = ktop.
    def full(x, top):
        return torch.cat((x[:, :1], x[:, :1], x, top[:, None]), dim=1)
    rho_top = 0.5 * prsi[:, levs - 1] * s["rdi"] / atm[:, levs - 1]
    RHOI = full(rhoint, rho_top)
    TINT = full(tint, TV[:, levs])
    UINT = full(uint, aum[:, levs - 1])
    VINT = full(vint, avm[:, levs - 1])
    BN2 = full(bn2, bn2[:, -1])
    BN = torch.sqrt(BN2)
    VZMET = full(2. * dz_meti, 2. * dz_meti[:, -1])
    VUEFF = full(vueff, vueff[:, -1])
    DZI = full(dz_meti, dz_meti[:, -1])
    DZM = full(dz_met, dz_met[:, -1])
    RHOM = full(rhomid, rhomid[:, -1])
    WRK3 = full(wrk3, wrk3[:, -1])
    AKT = full(akt, akt[:, -1])
    AKT = -AKT * (s["gor"] + (torch.cat((TINT[:, 1:], TINT[:, -1:]), dim=1) - TINT) / DZI)
    # Launch spectrum normalized to the launch flux, saturation density fden_bn.
    gather = lambda x: x.gather(1, ksrc[:, None]).squeeze(1)  # noqa: E731
    bvi = gather(BN)
    rstar = one / (bvi / s["zms"])
    rstar2 = rstar * rstar
    zcin = zci[None, :] * rstar[:, None]
    flux1 = rstar[:, None] * (zcin * zcin) / (1. + working_power(zcin, atm.new_full((), s["nslope3"])))
    fpu_src = torch.zeros_like(rstar)
    for inc in range(nwav):
        fpu_src = fpu_src + flux1[:, inc] * zdci[inc]
    flux_norm = taub_src / fpu_src
    ze1 = flux_norm * bvi / gather(RHOI) * rstar * rstar2
    FDEN = ze1[:, None] * RHOI / BN
    # The wave loop keeps waves leading ([nwav, ncol, nazd]: one wave's slice is contiguous) and reads
    # level-major profiles ([levs+2, ncol]: one level's row is contiguous); values are unchanged.
    # Per level it carries only the flux: a wave absorbed or reflected (wave_act = 0) holds the flux -1
    # from then on. An active wave's flux is >= 0 or NaN and an inactive wave's is never read again, so
    # the sign is wave_act. Compiled, a level is one kernel over the spectrum (the update) and one over
    # [ncol, nazd] (the wave-ordered integrals).
    flux = (flux_norm[:, None] * flux1).t()[:, :, None].expand(nwav, ncol, nazd)
    cos_t = static_tensor(cosang, dtype=dt, device=dev)
    sin_t = static_tensor(sinang, dtype=dt, device=dev)
    uint_l, vint_l, bn2_l, vzmet_l, fden_l, wrk3_l = (x.t().contiguous() for x in (UINT, VINT, BN2, VZMET, FDEN, WRK3))
    umfp_l = cos_t[None, None, :] * uint_l[:, :, None] + sin_t[None, None, :] * vint_l[:, :, None]
    launch_levels = torch.arange(levs + 2, device=dev)[:, None]
    fpu_seed = torch.where(ksrc[None, :] == launch_levels, taub_src[None, :], 0.0)
    return (U, V, T, Q, ZM, WRK3, BN2, AKT, DZI, DZM, RHOM, RHOI, ksrc, zngw, taub_src,
            flux, VUEFF.t().contiguous(), umfp_l, bn2_l, vzmet_l, fden_l, wrk3_l, kion,
            zci, c2f2, ucrit_max, (ZI[:, 1:] < ZSP_GW).all(), fpu_seed)


def ngw_solv2(fields: Mapping[str, torch.Tensor], tau_ngw: torch.Tensor, *, dtp: torch.Tensor,
              setup: Mapping[str, Any], nstab_cap: int = NSTAB_MAX, kstart: int = 3) -> dict[str, torch.Tensor]:
    """Ordered UGWP v1 solver, with bounded preparation, wave and diffusion regions."""
    atm = fields["t1"]
    ncol, levs = atm.shape
    dev, s = atm.device, setup
    nazd = s["nazd"]
    prepare = compile_policy.compiled(_ngw_prepare) if _regional_compile else _ngw_prepare
    consumed = {name: fields[name] for name in ("t1", "u1", "v1", "q1", "prsl", "prsi", "zmet", "zmeti", "sinlat")}
    (U, V, T, Q, ZM, WRK3, BN2, AKT, DZI, DZM, RHOM, RHOI, ksrc, zngw, taub_src,
     flux, vueff_rows, umfp_l, bn2_l, vzmet_l, fden_l, wrk3_l, kion, zci, c2f2, ucrit_max, below_sponge, fpu_seed) = prepare(
        consumed, tau_ngw, dtp=dtp, setup=s, nstab_cap=nstab_cap, kstart=kstart)
    torch._assert_async((ksrc <= levs - 2).all(), "UGWP v1 launch level within two layers of the top")
    torch._assert_async(below_sponge, "UGWP v1 wave sponge above 106.5 km is not ported")
    if kstart > 3:
        note_overflow("gwd.ugwp_v1_kstart", (ksrc < kstart).any())
    levels = torch.arange(1, levs + 1, device=dev).unbind(0)
    zdci = s["zdci"]
    zero_az = atm.new_zeros((ncol, nazd))
    # fpu(iaz, jk) for interfaces 1..levs+1 and dfdz(iaz, jk) for layers 1..levs.
    fpu = [row[:, None].expand(ncol, nazd) for row in fpu_seed.unbind(0)]
    dfdz = [zero_az] * (levs + 1)
    vueff_k = list(vueff_rows.unbind(0))
    umfp_k, bn2_k, vzmet_k, fden_k, wrk3_k = (
        value.unbind(0) for value in (umfp_l, bn2_l, vzmet_l, fden_l, wrk3_l))
    kion_k = kion[0].unbind(0)
    level = compile_policy.compiled(_wave_level) if _regional_compile else _wave_level
    for jk in range(kstart, levs + 1):  # layers below ksrc are no-ops
        flux, fpu[jk + 1], dfdz[jk], vueff_k[jk + 1] = level(
            flux, vueff_k[jk], ksrc, levels[jk - 1], umfp_k[jk + 1], bn2_k[jk + 1], vzmet_k[jk],
            fden_k[jk + 1], wrk3_k[jk], bn2_k[jk], kion_k[jk], zci,
            c2f2, ucrit_max, zdci, s["v_kxw"], s["mkz2min"],
            fpu[jk], fpu[jk + 1], dfdz[jk], vueff_k[jk + 1],
        )
    post = compile_policy.compiled(_ngw_diffusion_prepare) if _regional_compile else _ngw_diffusion_prepare
    (uold, vold, told, bdif, bt_dif, adif, cdif, dtdif, inner, nstab, pdudt, pdvdt, pdtdt, dked, kvint, dzm) = post(
        tuple(fpu), tuple(dfdz), ksrc, U, V, T, Q, ZM, WRK3, BN2, AKT, DZI, DZM, RHOM, RHOI, dtp, s)
    torch._assert_async((nstab <= NSTAB_MAX).all(), "UGWP v1 eddy-diffusion substeps above the ported cap")
    note_overflow("gwd.ugwp_v1_nstab", (nstab > nstab_cap).any())
    diffuse = compile_policy.compiled(_eddy_diffusion) if _regional_compile else _eddy_diffusion
    uold, vold, told = diffuse(uold, vold, told, bdif, bt_dif, adif, cdif, dtdif, inner, nstab,
                              nstab_cap=nstab_cap)
    finish = compile_policy.compiled(_ngw_finish) if _regional_compile else _ngw_finish
    pdudt, pdvdt, pdtdt = finish(uold, vold, told, U, V, T, dzm, kvint, inner, pdudt, pdvdt, pdtdt, dtp, s["rcpd"])
    return {"pdudt": pdudt, "pdvdt": pdvdt, "pdtdt": pdtdt, "dked": dked[:, 1:],
            "zngw": zngw, "ksrc": ksrc, "nstab": nstab}


def _ngw_diffusion_prepare(fpu, dfdz, ksrc, U, V, T, Q, ZM, WRK3, BN2, AKT, DZI, DZM, RHOM, RHOI, dtp, s):
    ncol, padded = U.shape
    levs = padded - 1
    atm, dev = U, U.device
    one, two = U.new_full((), 1.0), U.new_full((), 2.0)
    nazd, cosang, sinang = s["nazd"], s["cosang"], s["sinang"]
    gather = lambda x: x.gather(1, ksrc[:, None]).squeeze(1)  # noqa: E731
    # Azimuth sums retain the original left-to-right order.
    FPU = torch.stack(fpu, dim=1)  # [ncol, levs+2, nazd]
    taux, tauy = atm.new_zeros((ncol, levs + 2)), atm.new_zeros((ncol, levs + 2))
    raw = atm.new_zeros((ncol, levs + 1))
    DF = torch.stack(dfdz, dim=1)
    for iaz in range(nazd):
        taux = taux + FPU[:, :, iaz] * cosang[iaz]
        tauy = tauy + FPU[:, :, iaz] * sinang[iaz]
        raw = raw + DF[:, :, iaz]
    lev = torch.arange(levs + 1, device=dev)[None, :]  # one-based layer numbers, 0 = padding
    K = ksrc[:, None]
    above = (lev > K) & (lev <= levs)
    at_src = lev == K
    zdelp = WRK3[:, :levs + 1] * s["gw_eff"]
    ax = _clip((taux[:, 1:] - taux[:, :-1]) * zdelp, MAX_AXYZ)
    ay = _clip((tauy[:, 1:] - tauy[:, :-1]) * zdelp, MAX_AXYZ)
    pdudt = torch.where(above, -ax, 0.0)
    pdvdt = torch.where(above, -ay, 0.0)
    heated = _clip(raw * s["gw_eff"], MAX_EPS)
    pdtdt = torch.where(above, heated, torch.where(at_src, raw, 0.0))
    qmid = pdtdt
    scaled = torch.clamp_max(torch.clamp_min(raw / BN2[:, :levs + 1], DKED_MIN), DKED_MAX)
    dked = torch.where(above, scaled, torch.where(at_src, raw, 0.0))
    # 1-2-1 smoothing of the diffusivity above the launch level.
    nxt = torch.cat((dked[:, 1:], dked[:, -1:]), dim=1)
    prv = torch.cat((dked[:, :1], dked[:, :-1]), dim=1)
    smooth = .25 * (prv + nxt) + .5 * dked
    dktur = torch.where((lev >= K) & (lev <= levs - 1), smooth, dked)
    top = .5 * (dked[:, levs] + dked[:, levs - 1])
    dktur = torch.cat((dktur[:, :levs], top[:, None], top[:, None]), dim=1)  # 0..levs+1
    kv = .5 * (dktur[:, 1:] + dktur[:, :-1])  # kvint(jk) for jk = 1..levs+1 at index jk-1
    kvint = torch.cat((kv[:, :1], kv), dim=1)
    ktint = kvint * IPR_KTGW
    ktint_next = torch.cat((ktint[:, 1:], ktint[:, -1:]), dim=1)
    q2 = qmid + dktur[:, :levs + 1] * AKT[:, :levs + 1] + CON_G * (ktint_next[:, :levs + 1] - ktint[:, :levs + 1]) \
        / DZI[:, :levs + 1]
    q2 = _clip(q2, MAX_EPS)
    pdtdt = torch.where(above, q2 * s["rcpd"], pdtdt)
    dked = torch.where(above, dktur[:, :levs + 1], dked)
    # Updated state, stability-based turbulent diffusivity from km1 = ksrc-1.
    km1, km2 = K - 1, K - 2
    uold = torch.where(lev >= km2, U + pdudt * dtp, U)
    vold = torch.where(lev >= km2, V + pdvdt * dtp, V)
    told = torch.where(lev >= km2, T + pdtdt * dtp, T)
    prev = lambda x: torch.cat((x[:, :1], x[:, :-1]), dim=1)  # noqa: E731  value at layer jk-1
    uzs, vzs = uold - prev(uold), vold - prev(vold)
    dzm = DZM[:, :levs + 1]
    zd = one / dzm
    tvc2 = told * (1. + CON_FVIRT * Q)
    tvm2 = prev(told) * (1. + CON_FVIRT * prev(Q))
    zth2 = two / (tvc2 + tvm2)
    shr2b = torch.clamp_min(uzs * uzs + vzs * vzs, 1.0) * zd * zd
    bn2b = s["grav2cpd"] * zth2 * (1.0 + s["rcpdl"] * (tvc2 - tvm2) * zd)
    bn2b = torch.clamp_min(torch.clamp_max(bn2b, s["bnv2max"]), s["bnv2min"])
    zgrow2 = torch.exp(ZM * s["rh4"])
    w1b = one / (1. + 5 * (bn2b / shr2b))
    ze2b = torch.minimum(s["sc2"] * zgrow2, 4. * dzm * dzm)
    kamp2 = torch.sqrt(shr2b) * ze2b * w1b * w1b
    ktur = torch.clamp_max(torch.clamp_min(kamp2, DKED_MIN), DKED_MAX)
    stable_rows = (lev >= km1) & (lev >= 2)
    dktur2 = torch.where(stable_rows, ktur, DKED_MIN)
    dked = torch.where(stable_rows, dked + ktur, dked)
    # Eddy diffusion of u, v and T by the GW and turbulent diffusivity (knob_ugwp_dokdis = 2).
    kvint = torch.where((lev >= K), kvint[:, :levs + 1] + torch.clamp_max(.5 * (dktur2 + prev(dktur2)), DTURB_MAX),
                        kvint[:, :levs + 1])
    kvint = torch.cat((kvint, kvint[:, levs:levs + 1]), dim=1)  # kvint(ktop) = kvint(levs)
    kvint = torch.where(torch.cat((lev, lev[:, -1:] + 1), dim=1) == km1, gather(kvint)[:, None], kvint)
    nxtk = lambda x: torch.cat((x[:, 1:], x[:, -1:]), dim=1)  # noqa: E731  value at jk+1
    dzmetm = one / dzm
    dzmetp = nxtk(dzmetm)
    dzmetf = one / (DZI[:, :levs + 1] * RHOM[:, :levs + 1])
    turb = kvint[:, :levs + 1] * RHOI[:, :levs + 1] * dzmetf
    turbp = kvint[:, 1:levs + 2] * RHOI[:, 1:levs + 2] * dzmetf
    rows = (lev >= km1) & (lev <= levs - 1)
    adif = torch.where(rows, turb * dzmetm, 0.0)
    cdif = torch.where(rows, turbp * dzmetp, 0.0)
    acdif = adif + cdif
    rdtp = one / dtp
    w1c = acdif * 1.0
    anstab = torch.where(rdtp < w1c, torch.floor(w1c * dtp) + 1.0, 1.0)
    nstab = torch.where((lev >= K) & (lev <= levs - 1), anstab, 1.0).amax(dim=1)
    dtdif = dtp / nstab
    ze1d = one / dtdif
    inner = (lev >= K) & (lev <= levs - 1)
    bdif = ze1d[:, None] - acdif
    bt_dif = ze1d[:, None] - acdif * IPR_KTGW
    return uold, vold, told, bdif, bt_dif, adif, cdif, dtdif, inner, nstab, pdudt, pdvdt, pdtdt, dked, kvint, dzm


def _ngw_finish(uold, vold, told, U, V, T, dzm, kvint, inner, pdudt, pdvdt, pdtdt, dtp, rcpd):
    levs = U.shape[1] - 1
    one = U.new_full((), 1.0)
    rdtp = one / dtp
    prev = lambda x: torch.cat((x[:, :1], x[:, :-1]), dim=1)  # noqa: E731
    nxtk = lambda x: torch.cat((x[:, 1:], x[:, -1:]), dim=1)  # noqa: E731
    ze2 = _clip(rdtp * (uold - U), MAX_AXYZ)
    ze1v = _clip(rdtp * (vold - V), MAX_AXYZ)
    heat_t = _clip(rdtp * (told - T), MAXDTDT)
    uzf, vzf = nxtk(uold) - prev(uold), nxtk(vold) - prev(vold)
    zinv = one / (nxtk(dzm) + dzm)
    mf = rcpd * kvint[:, :levs + 1] * (uzf * uzf + vzf * vzf) * zinv * zinv
    pdudt = torch.where(inner, ze2, pdudt)
    pdvdt = torch.where(inner, ze1v, pdvdt)
    pdtdt = torch.where(inner, heat_t + mf, pdtdt)
    return pdudt[:, 1:], pdvdt[:, 1:], pdtdt[:, 1:]


def ugwpv1_gsldrag(fields: Mapping[str, torch.Tensor], *, dtp: torch.Tensor, alpha_fd: torch.Tensor,
                   control: Mapping[str, Any], forecast_day: torch.Tensor,
                   nstab_cap: int = NSTAB_MAX, kstart: int = 3) -> dict[str, torch.Tensor]:
    """ugwpv1_gsldrag_run for do_ugwp_v1 with the PSL GSL drag (do_gwd_opt_psl), source order.

    Orographic drag (drag_suite_psl) consumes the original Statein winds; the non-stationary
    waves follow. The scheme's own dtend updates apply only without the generic GWD tendency.
    nstab_cap and kstart bound the wave solver's loops (ngw_solv2).
    """
    if not control["do_ugwp_v1"] or not control["do_gwd_opt_psl"] or control["do_ngw_ec"] \
            or control["do_ugwp_v1_orog_only"] or control["ldiag_ugwp"]:
        raise NotImplementedError("ugwpv1_gsldrag port covers UGWP v1 waves with the PSL drag suite only")
    setup = control["ugwpv1"]
    t = fields["t1"]
    ncol, levs = t.shape
    zero2, zero1 = torch.zeros_like(t), t.new_zeros(ncol)
    zmeti = fields["phii"] * setup["rgrav"]
    zmet = fields["phil"] * setup["rgrav"]
    pdudt, pdvdt, pdtdt = zero2, zero2, zero2
    dusfcg, dvsfcg, rdxzb = zero1, zero1, zero1
    dtend = fields["dtend"]
    if control["do_gsl_drag_ls_bl"] or control["do_gsl_drag_ss"] or control["do_gsl_drag_tofd"]:
        drag = compile_policy.compiled(drag_suite_psl) if _regional_compile else drag_suite_psl
        psl = drag({**fields, "var": fields["hprime"]}, dtp=dtp, alpha_fd=alpha_fd, control=control)
        pdudt, pdvdt, pdtdt = psl["dudt"], psl["dvdt"], psl["dtdt"]
        dusfcg, dvsfcg, rdxzb = psl["dusfc"], psl["dvsfc"], psl["rdxzb"]
        dtend = psl.get("dtend", dtend)
    scheme_dtend = control["ldiag3d"] and control["lssav"] and not control["flag_for_gwd_generic_tend"]
    if scheme_dtend:
        dtend = _add_process(dtend, fields["dtidx"], control, "index_of_process_orographic_gwd",
                             (pdudt * dtp, pdvdt * dtp, pdtdt * dtp))
    tau_amf = tau_amf_interp(fields["xlat_d"], forecast_day, setup)
    tau_ngw = slat_geos5_2020(fields["xlat_d"], setup["tamp_mpa"])
    tau_ngw = ngwflux_update(tau_ngw, tau_amf, fields["xlat_d"], fields["rain"])
    ngw = ngw_solv2({**fields, "zmet": zmet, "zmeti": zmeti}, tau_ngw, dtp=dtp, setup=setup,
                    nstab_cap=nstab_cap, kstart=kstart)
    if scheme_dtend:
        dtend = _add_process(dtend, fields["dtidx"], control, "index_of_process_nonorographic_gwd",
                             (ngw["pdudt"] * dtp, ngw["pdvdt"] * dtp, ngw["pdtdt"] * dtp))
    result = {n: zero2 for n in ("dudt_ogw", "dvdt_ogw", "dudt_obl", "dvdt_obl",
                                 "dudt_oss", "dvdt_oss", "dudt_ofd", "dvdt_ofd")}
    result.update({n: zero1 for n in ("du_ogwcol", "dv_ogwcol", "du_oblcol", "dv_oblcol", "du_osscol",
                                      "dv_osscol", "du_ofdcol", "dv_ofdcol", "tau_ogw", "tau_oss",
                                      "zlwb", "zogw", "zobl")})
    result.update(dusfcg=dusfcg, dvsfcg=dvsfcg, rdxzb=rdxzb, tau_ngw=tau_ngw, tau_amf=tau_amf,
                  zngw=ngw["zngw"], dudt_ngw=ngw["pdudt"], dvdt_ngw=ngw["pdvdt"], dtdt_ngw=ngw["pdtdt"],
                  kdis_ngw=ngw["dked"], dudt_gw=pdudt + ngw["pdudt"], dvdt_gw=pdvdt + ngw["pdvdt"],
                  dtdt_gw=pdtdt + ngw["pdtdt"], kdis_gw=zero2 + ngw["dked"],
                  # No tracer tendency: a zero scalar viewed at the tracer shape (consumers only read it).
                  dqdt_gw=t.new_zeros(()).expand(ncol, levs, control["ntrac"]), dtend=dtend)
    return result


def _add_process(dtend, dtidx, control, process, values):
    from sa_tke_edmf_torch.diagnostics import add_diagnostic_
    dtend = dtend.clone()
    for name, value in zip(("index_of_x_wind", "index_of_y_wind", "index_of_temperature"), values):
        add_diagnostic_(dtend, value, dtidx[control[name] - 1, control[process] - 1])
    return dtend


def apply_ugwpv1_gsldrag(statein, grid, prefix, hpbl, *, dtp, alpha_fd, dtidx, control, forecast_day):
    """Bind ugwpv1_gsldrag to Statein, the GWD_generic_pre statistics and the live PBL carries, with the
    wave solver's loop bounds of ``runtime_policy.ugwp_v1_caps``."""
    work, sfcprop = prefix.interstitial, prefix.sfcprop
    fields = {
        "t1": statein.tgrs, "u1": statein.ugrs, "v1": statein.vgrs, "q1": statein.qgrs[:, :, 0],
        **{n: getattr(statein, n) for n in ("prsi", "prsl", "prslk", "phil", "phii")},
        **{n: work[n] for n in ("del", "kpbl", "oc", "oa4", "clx", "theta", "sigma", "gamma", "elvmax",
                                "varss", "oa4ss", "clxss")},
        "hprime": sfcprop.hprime[:, 0], "hpbl": hpbl, "br1": work["rb"], "slmsk": sfcprop.slmsk,
        "vtype": sfcprop.vtype, "dx": grid.dx, "xlat_d": grid.xlat_d, "sinlat": grid.sinlat,
        "rain": sfcprop.tprcp, "dtend": prefix.carry.intdiag["dtend"], "dtidx": dtidx}
    nstab_cap, full_wave_loop = ugwp_v1_caps()
    kstart = 3 if full_wave_loop else wave_loop_start(control["ugwpv1"])
    result = ugwpv1_gsldrag(fields, dtp=dtp, alpha_fd=alpha_fd, control=control, forecast_day=forecast_day,
                            nstab_cap=nstab_cap, kstart=kstart)
    diag_names = ("dudt_ogw", "dvdt_ogw", "dudt_obl", "dvdt_obl", "dudt_oss", "dvdt_oss", "dudt_ofd",
                  "dvdt_ofd", "du_ogwcol", "dv_ogwcol", "du_oblcol", "dv_oblcol", "du_osscol", "dv_osscol",
                  "du_ofdcol", "dv_ofdcol", "kdis_gw", "dtend")
    diag = {**prefix.carry.intdiag, **{n: result[n] for n in diag_names}, "zmtnblck": result["rdxzb"]}
    work_names = ("dusfcg", "dvsfcg", "tau_ogw", "tau_ngw", "tau_oss", "zogw", "zlwb", "zngw",
                  "dudt_ngw", "dvdt_ngw", "dtdt_ngw", "kdis_ngw", "tau_amf")
    return replace(prefix, interstitial={**work, **{n: result[n] for n in work_names}, "zmtb": result["zobl"],
        "ten_t": result["dtdt_gw"], "ten_u": result["dudt_gw"], "ten_v": result["dvdt_gw"],
        "ten_q": result["dqdt_gw"]}, carry=replace(prefix.carry, intdiag=diag))


def ugwpv1_gsldrag_post(fields, *, dtf, ldiag_ugwp):
    """ugwpv1_gsldrag_post: time-averaged diagnostics only when ldiag_ugwp (not ported)."""
    if ldiag_ugwp:
        raise NotImplementedError("ugwpv1_gsldrag_post diagnostics (ldiag_ugwp) are not ported")
    return {}


def _apply_ugwpv1_regions(*args, **kwargs):
    global _regional_compile
    previous, _regional_compile = _regional_compile, True
    try:
        return apply_ugwpv1_gsldrag(*args, **kwargs)
    finally:
        _regional_compile = previous


_apply_ugwpv1_regions.__wrapped__ = apply_ugwpv1_gsldrag
_apply_ugwpv1_regions.warm = lambda *args, **kwargs: None
compile_policy._compiled[apply_ugwpv1_gsldrag] = _apply_ugwpv1_regions
