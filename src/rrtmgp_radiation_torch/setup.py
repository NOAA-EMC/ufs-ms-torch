# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""PyTorch-native RRTMGP timestep setup surface."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import torch

from .reference_inputs import physics_input_dir as _physics_input_dir


TensorOrString = torch.Tensor | str
CON_SOLR_2008 = 1360.8
CON_SOLR_2002 = 1366.0
SOLAR_TABLE_EXPECTED_SHA256 = {
    "solarconstant_noaa_an.txt": "d2177cf38b60ac4fca23524aa1d187bed58dfcd412e96a6a3c50c9cef9b71cb8",
    "solarconstant_noaa_a0.txt": "57432b5a55ab0273e36b7aaa26ab934c1cdb943df3aa19707451d2aacd82da2d",
}
CZLIMT = 0.0001


@dataclass(frozen=True)
class RrtmgpSetupResult:
    """Checkpoint-shaped A-boundary setup fields."""

    fields: Mapping[str, TensorOrString]
    runtime_kind: str = "torch_rrtmgp_setup_timestep"


def _int_flag(value: bool | int, *, device: torch.device) -> torch.Tensor:
    return torch.tensor([1 if bool(value) else 0], dtype=torch.int64, device=device)


def _int_scalar(value: int, *, device: torch.device) -> torch.Tensor:
    return torch.tensor([int(value)], dtype=torch.int64, device=device)


def _float_scalar(value: float | torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float64, device=device).reshape(1)
    return torch.tensor([float(value)], dtype=torch.float64, device=device)


def _to_int(value: TensorOrString, *, name: str) -> int:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    return int(value.detach().to(device="cpu", dtype=torch.int64).reshape(-1)[0].item())


def _to_float(value: TensorOrString, *, name: str) -> float:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    return float(value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)[0].item())


def _to_real_vector(value: TensorOrString, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64 (the working dtype), got {value.dtype}")
    return value.reshape(-1)


def _iw3jdn(year: int, month: int, day: int) -> int:
    """Gregorian Julian day number equivalent to the NCEP W3 helper for SCM dates."""

    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045


def _fortran_mod(a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return torch.fmod(a, p)


def _load_annual_solar_constant(
    year: int,
    *,
    filename: str,
    physics_input_dir: str | Path | None = None,
) -> float:
    """Load the annual TSI value used by `module_radiation_astronomy::sol_update`."""

    path = (Path(physics_input_dir) if physics_input_dir is not None else _physics_input_dir()) / filename
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty solar table: {path}")
    header = lines[0].split()
    if len(header) < 5:
        raise ValueError(f"invalid solar table header in {path}")
    iyr1 = int(header[0])
    iyr2 = int(header[1])
    icy1 = int(header[2])
    icy2 = int(header[3])
    smean = float(header[4])
    lookup_year = int(year)
    if lookup_year < iyr1:
        icy = icy1 - iyr1 + 1
        while lookup_year < iyr1:
            lookup_year += icy
    elif lookup_year > iyr2:
        icy = iyr2 - icy2 + 1
        while lookup_year > iyr2:
            lookup_year -= icy
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        if int(parts[0]) == lookup_year:
            return smean + float(parts[1])
    raise ValueError(f"solar table {path} has no usable year {lookup_year} mapped from {year}")


CO2_MIN_YEAR = 1957  # radiation_gases MINYEAR


def load_co2_year(input_dir: str | Path, year: int, *, ictm: int = 1) -> tuple[torch.Tensor, float]:
    """``gas_update`` (``ico2 = 2``, ``ictm = 1``) for one model year: ``co2vmr_sav`` and ``co2_glb``.

    Host preparation. Returns the monthly 15-degree table ``[month, lat, lon]`` and the global
    annual mean (volume mixing ratios). As in the source, a missing year falls back to the latest
    earlier ``co2historicaldata_YYYY.txt`` in ``input_dir`` and adds 2 ppmv per year
    (``lextpl``); the header and data fields are read at their fixed Fortran columns
    (``(i4,a94,f7.2,16x,f5.2)`` and ``(24f7.2)``).
    """
    if ictm != 1:
        raise NotImplementedError("only ictm = 1 (data at the forecast time, extrapolated) is ported")
    if year < CO2_MIN_YEAR:
        raise NotImplementedError("CO2 years before 1957 use the global-mean table branch (not ported)")
    input_dir = Path(input_dir)
    iyr = int(year)
    path = input_dir / f"co2historicaldata_{iyr:04d}.txt"
    while not path.exists():
        iyr -= 1
        if iyr < CO2_MIN_YEAR:
            raise FileNotFoundError(f"no co2historicaldata_YYYY.txt at or before {year} in {input_dir}")
        path = input_dir / f"co2historicaldata_{iyr:04d}.txt"
    lines = path.read_text().splitlines()
    file_year = int(lines[0][0:4])
    co2g1 = float(lines[0][98:105])
    rate = 2.00 * (year - file_year)
    values = []
    for line in lines[1:1 + 12 * 12]:
        values.extend(float(line[7 * i:7 * i + 7]) for i in range(24))
    table = (torch.tensor(values, dtype=torch.float64).reshape(12, 12, 24) + rate) * 1.0e-6
    return table, (co2g1 + rate) * 1.0e-6


def _initialized_solar_constant(
    *,
    isol: int,
    year: int,
    month: int,
    physics_input_dir: str | Path | None = None,
) -> float:
    if isol == 0:
        return CON_SOLR_2002
    if isol == 10:
        return CON_SOLR_2008
    if isol == 2:
        return _load_annual_solar_constant(
            year,
            filename="solarconstant_noaa_an.txt",
            physics_input_dir=physics_input_dir,
        )
    if isol == 1:
        return _load_annual_solar_constant(
            year,
            filename="solarconstant_noaa_a0.txt",
            physics_input_dir=physics_input_dir,
        )
    if isol == 4:
        raise NotImplementedError("monthly CMIP solar table branch isol=4 is not ported in the current RRTMGP target")
    raise NotImplementedError(f"RRTMGP setup target does not yet support isol={isol}")


def _solar_state(
    jdate: torch.Tensor | tuple[int, ...],
    *,
    con_pi: float,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Source solar-update date arithmetic on the numerical device."""
    if not isinstance(jdate, torch.Tensor):
        jdate = torch.tensor(jdate, dtype=torch.int32, device=device)
    if jdate.numel() != 8 or jdate.dtype not in (torch.int32, torch.int64):
        raise ValueError("jdate must contain eight NCEP date integers")
    if device is not None and jdate.device != torch.device(device):
        raise ValueError("solar date and numerical state must share a device")
    vals = jdate.reshape(8).to(torch.int64)
    iyear, imon, iday, ihr, imin, isec = (vals[i] for i in (0, 1, 2, 4, 5, 6))
    torch._assert_async(
        (iyear >= 1583)
        & (iyear <= 9999)
        & (imon >= 1)
        & (imon <= 12)
        & (iday >= 1)
        & (iday <= 31)
        & (ihr >= 0)
        & (ihr < 24)
        & (imin >= 0)
        & (imin < 60)
        & (isec >= 0)
        & (isec < 60),
        "invalid Gregorian solar date",
    )
    hrday = 1.0 / 24.0
    minday = 1.0 / 1440.0
    secday = 1.0 / 86400.0
    jd1 = _iw3jdn(iyear, imon, iday)
    morning = ihr < 12
    hour = ihr.to(torch.float64)
    minute = imin.to(torch.float64)
    second = isec.to(torch.float64)
    fjd1 = torch.where(
        morning,
        0.5 + hour * hrday + minute * minday + second * secday,
        (hour - 12.0) * hrday + minute * minday + second * secday,
    )
    fjd1 = fjd1 + (jd1 - morning.to(torch.int64)).to(torch.float64)
    jd = torch.trunc(fjd1).to(torch.int64)
    return _solar_orbit(jd, fjd1 - jd.to(torch.float64), con_pi=con_pi)


def _solar_orbit(
    jd: torch.Tensor, fjd: torch.Tensor, *, con_pi: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Source solar orbit, including both eleven-iteration Newton ceilings."""
    if (
        jd.numel() != 1
        or jd.dtype not in (torch.int32, torch.int64)
        or fjd.numel() != 1
        or fjd.dtype != torch.float64
        or jd.device != fjd.device
    ):
        raise ValueError("solar orbit requires scalar integer day/float64 fraction")
    torch._assert_async(torch.isfinite(fjd).all(), "nonfinite solar day fraction")
    jd, fjd = jd.reshape(()), fjd.reshape(())

    cyear = 365.25
    ccr = 1.3e-6
    tpp = 1.55
    svt6 = 78.035
    jdor = 2415020
    tpi = 2.0 * con_pi
    degrad = 180.0 / con_pi
    t1 = (jd - jdor).to(torch.float64) / 36525.0
    year = 0.25964134 + 0.304e-5 * t1
    tyear = 0.24219879e0 - 0.614e-5 * t1
    ec = 0.01675104e0 - (0.418e-4 + 0.126e-6 * t1) * t1
    angin = 23.452294e0 - (0.0130125e0 + 0.164e-5 * t1) * t1
    jdoe = torch.trunc(jdor + torch.div(svt6 * cyear, year - tyear)).to(torch.int64)
    deleqn = (jdoe - jd).to(torch.float64) * (year - tyear) / cyear
    year = year + 365.0
    sni = torch.sin(angin / degrad)
    tini = torch.div(1.0, torch.tan(angin / degrad))
    er = torch.sqrt((1.0 + ec) / (1.0 - ec))
    qq = deleqn * tpi / year

    def eccentric_anomaly(mean: torch.Tensor) -> torch.Tensor:
        e1 = torch.ones_like(mean)
        delta = torch.ones_like(mean)
        for _ in range(11):
            active = delta > ccr
            ep = e1 - (e1 - ec * torch.sin(e1) - mean) / (1.0 - ec * torch.cos(e1))
            delta = torch.where(active, torch.abs(e1 - ep), delta)
            e1 = torch.where(active, ep, e1)
        return e1

    dat = (jd - jdor).to(torch.float64) - tpp + fjd
    date = _fortran_mod(dat, year)
    em = tpi * date / year
    # Both Newton solves (qq, then em) run element by element in one loop: the same arithmetic.
    anomaly = eccentric_anomaly(torch.stack((qq, em)))
    eq = 2.0 * torch.atan(er * torch.tan(0.5 * anomaly[0]))
    e1 = anomaly[1]
    w1 = 2.0 * torch.atan(er * torch.tan(0.5 * e1))
    r1 = 1.0 - ec * torch.cos(e1)
    sindec = sni * torch.sin(w1 - eq)
    cosdec = torch.sqrt(1.0 - sindec * sindec)
    dlt = torch.asin(sindec)
    alp = torch.asin(torch.tan(dlt) * tini)
    alp = torch.where(torch.cos(w1 - eq) < 0.0, con_pi - alp, alp)
    alp = torch.where(alp < 0.0, alp + tpi, alp)
    sun = tpi * (date - deleqn) / year
    sun = torch.where(sun < 0.0, sun + tpi, sun)
    sollag = sun - alp - 0.03255e0
    return r1, sollag, sindec, cosdec


def _coszmn(
    *,
    xlon: torch.Tensor,
    sinlat: torch.Tensor,
    coslat: torch.Tensor,
    solhr: float | torch.Tensor,
    sollag: float | torch.Tensor,
    sindec: float | torch.Tensor,
    cosdec: float | torch.Tensor,
    deltsw: float,
    deltim: float,
    con_pi: float,
) -> torch.Tensor:
    """Port `module_radiation_astronomy::coszmn` for checkpoint-shaped columns."""

    pid12 = con_pi / 12.0
    ratio = deltsw / deltim
    whole = math.floor(ratio)
    nswr = max(1, whole + int(ratio - whole >= 0.5))
    nstp = max(6, nswr)
    anginc = pid12 * (deltsw / 3600.0) / float(nstp)
    solang = pid12 * (solhr - 12.0)
    coszen = torch.zeros_like(xlon)
    istsun = torch.zeros_like(xlon, dtype=torch.int64)
    for it in range(1, nstp + 1):
        cns = solang + (float(it) - 0.5) * anginc + sollag
        coszn = sindec * sinlat + cosdec * coslat * torch.cos(xlon + cns)
        coszen = coszen + torch.clamp(coszn, min=0.0)
        istsun = istsun + (coszn > CZLIMT).to(dtype=torch.int64)
    divisor = torch.clamp_min(istsun, 1).to(dtype=xlon.dtype)
    return torch.where((istsun > 0) & (coszen != 0.0), coszen / divisor, coszen)


def run_setup_timestep_from_host_inputs(
    host_inputs: Mapping[str, TensorOrString],
    *,
    doSWclrsky: bool = True,
    doLWclrsky: bool = True,
    doGP_lwscat: bool = True,
    nGauss_angles: int = 1,
    lsswr: bool | None = None,
    lslwr: bool | None = None,
    isol: int | None = None,
    deltim: float | None = None,
    physics_input_dir: str | Path | None = None,
    device: str | torch.device | None = None,
) -> RrtmgpSetupResult:
    """Compute the A-boundary setup fields from sealed host pre-call inputs.

    This ports the scalar setup pieces that are scientifically relevant to the
    checkpoint boundary: radiation call flags, solar-constant table lookup,
    astronomy update, and column solar-zenith averaging. Aerosol/gas table
    update side effects are validated by their downstream source-hook gates.
    """

    jdate = host_inputs["jdat"]
    if not isinstance(jdate, torch.Tensor):
        raise TypeError("jdat must be a torch tensor")
    torch_device = torch.device(device) if device is not None else jdate.device
    jdate_cpu = jdate.detach().to(device="cpu", dtype=torch.int64)
    idate = host_inputs.get("idat", jdate)
    if not isinstance(idate, torch.Tensor):
        raise TypeError("idat must be a torch tensor")
    idate_cpu = idate.detach().to(device="cpu", dtype=torch.int64)

    do_sw = _to_int(host_inputs["doSWrad"], name="doSWrad") != 0
    do_lw = _to_int(host_inputs["doLWrad"], name="doLWrad") != 0
    lsswr_value = do_sw if lsswr is None else bool(lsswr)
    lslwr_value = do_lw if lslwr is None else bool(lslwr)
    con_pi = _to_float(host_inputs["con_pi"], name="con_pi")
    solhr = _to_float(host_inputs["solhr"], name="solhr")
    deltsw = _to_float(host_inputs["fhswr"], name="fhswr")
    deltim_value = (
        float(deltim) if deltim is not None
        else _to_float(host_inputs.get("deltim", host_inputs["fhswr"]), name="deltim")
    )
    jvals = [int(item) for item in jdate_cpu.reshape(-1).tolist()]
    ivals = [int(item) for item in idate_cpu.reshape(-1).tolist()]
    if len(jvals) < 8 or len(ivals) < 8:
        raise ValueError("idat and jdat must contain 8 NCEP date integers")
    if do_sw:
        isol_value = int(isol if isol is not None else 2)
        kyear = ivals[0] if _to_int(host_inputs.get("ictm", torch.tensor([1])), name="ictm") in (0, -2) else jvals[0]
        solc0 = _initialized_solar_constant(
            isol=isol_value,
            year=kyear,
            month=jvals[1],
            physics_input_dir=physics_input_dir,
        )
        r1, sollag, sindec, cosdec = _solar_state(jdate_cpu.to(device=torch_device), con_pi=con_pi)
        solcon = torch.div(solc0, r1 * r1)
        xlon = _to_real_vector(host_inputs["xlon"], name="xlon").to(device=torch_device)
        sinlat = _to_real_vector(host_inputs["sinlat"], name="sinlat").to(device=torch_device)
        coslat = _to_real_vector(host_inputs["coslat"], name="coslat").to(device=torch_device)
        coszen = _coszmn(
            xlon=xlon,
            sinlat=sinlat,
            coslat=coslat,
            solhr=solhr,
            sollag=sollag,
            sindec=sindec,
            cosdec=cosdec,
            deltsw=deltsw,
            deltim=deltim_value,
            con_pi=con_pi,
        )
    else:
        solcon = 0.0
        coszen = torch.zeros_like(_to_real_vector(host_inputs["xlon"], name="xlon").to(device=torch_device))

    return RrtmgpSetupResult(
        fields={
            "doSWrad": _int_flag(do_sw, device=torch.device("cpu")),
            "doLWrad": _int_flag(do_lw, device=torch.device("cpu")),
            "doSWclrsky": _int_flag(doSWclrsky, device=torch.device("cpu")),
            "doLWclrsky": _int_flag(doLWclrsky, device=torch.device("cpu")),
            "doGP_lwscat": _int_flag(doGP_lwscat, device=torch.device("cpu")),
            "nGauss_angles": _int_scalar(nGauss_angles, device=torch.device("cpu")),
            "lsswr": _int_flag(lsswr_value, device=torch.device("cpu")),
            "lslwr": _int_flag(lslwr_value, device=torch.device("cpu")),
            "solhr": _float_scalar(solhr, device=torch_device),
            "solcon": _float_scalar(solcon, device=torch_device),
            "coszen": coszen.to(dtype=torch.float64).reshape(-1),
            "errmsg": "",
            "errflg": _int_scalar(0, device=torch.device("cpu")),
        },
        runtime_kind="torch_rrtmgp_setup_timestep_from_host_inputs",
    )


def run_setup_timestep_from_config(
    *,
    doSWrad: bool,
    doLWrad: bool,
    doSWclrsky: bool,
    doLWclrsky: bool,
    doGP_lwscat: bool,
    nGauss_angles: int,
    lsswr: bool,
    lslwr: bool,
    solhr: float,
    solcon: float,
    coszen: float,
    device: str | torch.device = "cpu",
) -> RrtmgpSetupResult:
    """Build the A-boundary fields after `GFS_rrtmgp_setup_timestep_init`.

    The upstream Fortran setup routine updates solar, aerosol, gas, and ozone
    climatology state. This target surface represents the host-visible fields
    captured at the A boundary; external update-state provenance must be gated
    separately before a full-chain claim.
    """

    torch_device = torch.device(device)
    return RrtmgpSetupResult(
        fields={
            "doSWrad": _int_flag(doSWrad, device=torch_device),
            "doLWrad": _int_flag(doLWrad, device=torch_device),
            "doSWclrsky": _int_flag(doSWclrsky, device=torch_device),
            "doLWclrsky": _int_flag(doLWclrsky, device=torch_device),
            "doGP_lwscat": _int_flag(doGP_lwscat, device=torch_device),
            "nGauss_angles": _int_scalar(nGauss_angles, device=torch_device),
            "lsswr": _int_flag(lsswr, device=torch_device),
            "lslwr": _int_flag(lslwr, device=torch_device),
            "solhr": _float_scalar(solhr, device=torch_device),
            "solcon": _float_scalar(solcon, device=torch_device),
            "coszen": _float_scalar(coszen, device=torch_device),
            "errmsg": "",
            "errflg": _int_scalar(0, device=torch_device),
        }
    )


__all__ = [
    "RrtmgpSetupResult",
    "SOLAR_TABLE_EXPECTED_SHA256",
    "run_setup_timestep_from_config",
    "run_setup_timestep_from_host_inputs",
]
