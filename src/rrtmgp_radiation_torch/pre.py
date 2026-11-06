# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""PyTorch-native RRTMGP pre-processing kernels for the active profile scope."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re
from pathlib import Path
from typing import Mapping

import torch

from .reference_inputs import physics_input_dir as _physics_input_dir


TensorOrString = torch.Tensor | str

CON_G = 9.80665
CON_RD = 287.05
CON_RV = 461.50
CON_EPS = CON_RD / CON_RV
CON_EPSM1 = CON_EPS - 1.0
CON_FVIRT = CON_RV / CON_RD - 1.0
CON_EPSQS = 1.0e-10
CON_PSAT = 6.1078e2
CON_TTP = 2.7316e2
CON_HVAP = 2.5e6
CON_HFUS = 3.3358e5
CON_CVAP = 1.8460e3
CON_CLIQ = 4.1855e3
CON_CSOL = 2.1060e3
CON_AMD = 28.9644
CON_AMW = 18.0154
CON_AMO3 = 47.9982
AMDW = CON_AMD / CON_AMW
AMDO3 = CON_AMD / CON_AMO3

DEFAULT_ACTIVE_GASES_TEXT = "h2o_co2_o3_n2o_ch4_o2"
DEFAULT_ACTIVE_GASES_ARRAY = ("h2o", "co2", "o3", "n2o", "ch4", "o2")
DEFAULT_ACTIVE_GASES_ARRAY_NETCDF_TEXT = "".join(name.ljust(128) for name in DEFAULT_ACTIVE_GASES_ARRAY).rstrip()
DEFAULT_CO2_VMR = 350.0e-6
DEFAULT_N2O_VMR = 0.31e-6
DEFAULT_CH4_VMR = 1.50e-6
DEFAULT_O2_VMR = 0.209
CO2_PRESSURE_LIMIT_PA = 78800.0
CO2_TABLE_IMAX = 24
CO2_TABLE_JMAX = 12
CO2_TABLE_RESLONLAT_DEG = 15.0
CO2_TABLE_EXPECTED_SHA256 = {
    1992: "bba504f731fd5e72c32a0ded020a174a079154b5d81729b455e79cdbcbe7f17c",
}


@dataclass(frozen=True)
class RrtmgpPreResult:
    """Checkpoint-shaped AB-boundary pre-processing fields."""

    fields: Mapping[str, TensorOrString]
    runtime_kind: str = "torch_rrtmgp_pre"


def _as_real(value: TensorOrString, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64 (the working dtype), got {value.dtype}")
    return value


def _as_int_tensor(value: TensorOrString, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    return value.to(dtype=torch.int64)


def _scalar_flag(fields: Mapping[str, TensorOrString], name: str, default: bool = False) -> torch.Tensor:
    value = fields.get(name)
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.int64).reshape(-1)[:1]
    return torch.tensor([1 if default else 0], dtype=torch.int64)


def _scalar_int(fields: Mapping[str, TensorOrString], name: str, default: int) -> torch.Tensor:
    value = fields.get(name)
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.int64).reshape(-1)[:1]
    return torch.tensor([int(default)], dtype=torch.int64)


def _scalar_int_value(fields: Mapping[str, TensorOrString], name: str, default: int) -> int:
    value = fields.get(name)
    if isinstance(value, torch.Tensor):
        return int(value.detach().to(device="cpu", dtype=torch.int64).reshape(-1)[0].item())
    return int(default)


def _scalar_float_value(fields: Mapping[str, TensorOrString], name: str, default: float) -> float:
    value = fields.get(name)
    if isinstance(value, torch.Tensor):
        return float(value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)[0].item())
    return float(default)


def _date_year_month(fields: Mapping[str, TensorOrString]) -> tuple[int, int]:
    for name in ("jdat", "idat"):
        value = fields.get(name)
        if isinstance(value, torch.Tensor) and value.numel() >= 2:
            flat = value.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
            year = int(flat[0].item())
            month = int(flat[1].item())
            if 1 <= month <= 12:
                return year, month
    return _scalar_int_value(fields, "co2_year", 1992), _scalar_int_value(fields, "co2_month", 6)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
@torch.inference_mode(False)
def _load_co2_monthly_table_cpu(year: int) -> tuple[torch.Tensor, float]:
    path = _physics_input_dir() / f"co2historicaldata_{year:04d}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    expected_sha = CO2_TABLE_EXPECTED_SHA256.get(year)
    if expected_sha is not None:
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(f"{path.name} sha256 {actual_sha} does not match frozen {expected_sha}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = lines[0] if lines else ""
    match = re.search(r"GLB ANNUAL MEAN =\s*([0-9.]+)", header)
    if match is None:
        raise ValueError(f"could not parse global annual CO2 mean from {path}")
    global_mean_vmr = float(match.group(1)) * 1.0e-6
    values: list[float] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == CO2_TABLE_IMAX:
            values.extend(float(part) for part in parts)
    expected_count = 12 * CO2_TABLE_JMAX * CO2_TABLE_IMAX
    if len(values) != expected_count:
        raise ValueError(f"{path} has {len(values)} CO2 values, expected {expected_count}")
    table = torch.tensor(values, dtype=torch.float64).reshape(12, CO2_TABLE_JMAX, CO2_TABLE_IMAX) * 1.0e-6
    return table, global_mean_vmr


def compute_nonprognostic_gases(
    p_lev: torch.Tensor,
    xlon: torch.Tensor,
    xlat: torch.Tensor,
    *,
    ico2: int,
    top_at_1: bool,
    con_pi: float = math.pi,
    co2_year: int = 1992,
    co2_month: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port `module_radiation_gases::getgases` for gases used by RRTMGP pre.

    The active sealed profile uses `ico2=2`, which selects a 15-degree monthly
    CO2 table below 788 hPa and a global annual mean above that pressure. Other
    non-prognostic gases are constants in the upstream routine.
    """

    p_int = _as_real(p_lev, name="p_lev")
    lon = _as_real(xlon, name="xlon").reshape(-1)
    lat = _as_real(xlat, name="xlat").reshape(-1)
    nlay = p_int.shape[0] - 1
    ncol = p_int.shape[1]
    if lon.numel() != ncol or lat.numel() != ncol:
        raise ValueError("xlon/xlat length must match p_lev columns")
    co2 = torch.full((nlay, ncol), DEFAULT_CO2_VMR, dtype=p_int.dtype, device=p_int.device)
    if ico2 == 1:
        _, global_mean = _load_co2_monthly_table_cpu(co2_year)
        co2 = torch.full_like(co2, global_mean)
    elif ico2 == 2:
        if not (1 <= co2_month <= 12):
            raise ValueError(f"co2_month must be 1..12, got {co2_month}")
        table, global_mean = _co2_table_on_device(co2_year, p_int.device, p_int.dtype)
        raddeg = 180.0 / con_pi
        tmp = raddeg / CO2_TABLE_RESLONLAT_DEG
        lon_wrapped = torch.where(lon < 0.0, lon + con_pi, lon)
        lat_index_value = (0.5 * con_pi - lat) * tmp + 1.0
        lon_index_value = lon_wrapped * tmp + 1.0
        ilon = torch.clamp(torch.floor(lon_index_value).to(dtype=torch.int64), min=1, max=CO2_TABLE_IMAX) - 1
        ilat = torch.clamp(torch.floor(lat_index_value).to(dtype=torch.int64), min=1, max=CO2_TABLE_JMAX) - 1
        column_co2 = table[co2_month - 1, ilat, ilon]
        if top_at_1:
            lower_atmosphere = p_int[:-1, :] >= CO2_PRESSURE_LIMIT_PA
        else:
            lower_atmosphere = p_int[1:, :] >= CO2_PRESSURE_LIMIT_PA
        co2 = torch.where(lower_atmosphere, column_co2.reshape(1, ncol), torch.full_like(co2, global_mean))
    elif ico2 != 0:
        raise NotImplementedError(f"unsupported ico2 branch for RRTMGP pre gas lookup: {ico2}")
    n2o = _repeat_gas(DEFAULT_N2O_VMR, co2)
    ch4 = _repeat_gas(DEFAULT_CH4_VMR, co2)
    o2 = _repeat_gas(DEFAULT_O2_VMR, co2)
    return o2, ch4, n2o, co2


@lru_cache(maxsize=1)
@torch.inference_mode(False)
def _fpvs_table_cpu() -> torch.Tensor:
    n = 7501
    xmin = 180.0
    xmax = 330.0
    # funcphys::gpvs builds this ordered grid; linspace uses different rounding.
    x = xmin + torch.arange(n, dtype=torch.float64) * ((xmax - xmin) / (n - 1))
    tr = torch.div(CON_TTP, x)
    tliq = CON_TTP
    tice = CON_TTP - 20.0
    dldtl = CON_CVAP - CON_CLIQ
    dldti = CON_CVAP - CON_CSOL
    xponal = -dldtl / CON_RV
    xponbl = -dldtl / CON_RV + CON_HVAP / (CON_RV * CON_TTP)
    xponai = -dldti / CON_RV
    xponbi = -dldti / CON_RV + (CON_HVAP + CON_HFUS) / (CON_RV * CON_TTP)
    liquid = CON_PSAT * (tr**xponal) * torch.exp(xponbl * (1.0 - tr))
    ice = CON_PSAT * (tr**xponai) * torch.exp(xponbi * (1.0 - tr))
    weight = (x - tice) / (tliq - tice)
    return torch.where(x >= tliq, liquid, torch.where(x < tice, ice, weight * liquid + (1.0 - weight) * ice))


def fpvs(t: torch.Tensor) -> torch.Tensor:
    """Match `funcphys::fpvs` linear lookup table in Pascals."""

    temp = _as_real(t, name="t")
    table = _fpvs_table_on_device(temp.device, temp.dtype)
    c2 = 1.0 / ((330.0 - 180.0) / 7500.0)
    c1 = 1.0 - 180.0 * c2
    xj = torch.clamp(c1 + c2 * temp, min=1.0, max=7501.0)
    jx = torch.minimum(torch.floor(xj).to(torch.int64), torch.full_like(torch.floor(xj).to(torch.int64), 7500))
    lower = table[jx - 1]
    upper = table[jx]
    return lower + (xj - jx.to(dtype=temp.dtype)) * (upper - lower)


_FPVS_DEVICE_TABLES: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _fpvs_table_on_device(device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Prebuilt ``fpvs`` table per device and working dtype; a compiled region reads it as an input.

    Built in float64 on the host (funcphys::gpvs) by preparation or the eager first call. A plain
    dict, not ``lru_cache``: Dynamo traces through ``lru_cache`` and rebuilt the table inside every
    graph that reached it.
    """
    key = (device, dtype)
    table = _FPVS_DEVICE_TABLES.get(key)
    if table is None:
        table = _FPVS_DEVICE_TABLES[key] = _fpvs_device_copy(device, dtype)
    return table


@torch.inference_mode(False)
def _fpvs_device_copy(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _fpvs_table_cpu().to(device=device, dtype=dtype)


@lru_cache(maxsize=8)
@torch.inference_mode(False)
def _co2_table_on_device(year: int, device: torch.device, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, float]:
    table, global_mean = _load_co2_monthly_table_cpu(year)
    return table.to(device=device, dtype=dtype), global_mean


def compute_interface_temperature(
    p_lay: torch.Tensor,
    t_lay: torch.Tensor,
    p_lev: torch.Tensor,
    tsfc: torch.Tensor,
    *,
    top_at_1: bool | None = None,
    min_pressure: float = 1.005183574463,
) -> torch.Tensor:
    """Port `radiation_tools::cmp_tlev` for `[nlay,ncol]` checkpoint layout."""

    p_mid = _as_real(p_lay, name="p_lay")
    t_mid = _as_real(t_lay, name="t_lay")
    p_int = _as_real(p_lev, name="p_lev")
    surface_t = _as_real(tsfc, name="tsfc").reshape(-1)
    if p_mid.shape != t_mid.shape:
        raise ValueError("p_lay and t_lay shapes differ")
    nlay, ncol = p_mid.shape
    if p_int.shape != (nlay + 1, ncol):
        raise ValueError(f"p_lev must have shape {(nlay + 1, ncol)}, got {tuple(p_int.shape)}")
    if surface_t.numel() != ncol:
        raise ValueError(f"tsfc must have {ncol} columns, got {surface_t.numel()}")

    if top_at_1 is None:
        top_at_1 = p_int[0, 0] < p_int[nlay - 1, 0]

    def top_first():
        denom = torch.log(p_mid[:-1, :]) - torch.log(p_mid[1:, :])
        interior = t_mid[1:, :] + (t_mid[:-1, :] - t_mid[1:, :]) * (
            torch.log(p_int[1:nlay, :]) - torch.log(p_mid[1:, :])
        ) / denom
        return torch.cat((t_mid[:1, :], interior, surface_t.unsqueeze(0)), dim=0)

    def bottom_first():
        denom = torch.log(p_mid[1:, :]) - torch.log(p_mid[:-1, :])
        interior = t_mid[:-1, :] + (t_mid[1:, :] - t_mid[:-1, :]) * (
            torch.log(p_int[1:nlay, :]) - torch.log(p_mid[:-1, :])
        ) / denom
        return torch.cat((surface_t.unsqueeze(0), interior, t_mid[-1:, :]), dim=0)

    if isinstance(top_at_1, bool):
        return top_first() if top_at_1 else bottom_first()
    return torch.where(top_at_1, top_first(), bottom_first())


def compute_layer_geometry(
    p_lev: torch.Tensor,
    p_lay: torch.Tensor,
    tv_lay: torch.Tensor,
    *,
    top_at_1: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool | torch.Tensor]:
    """Compute source-ordered geometry with tensor-only pressure arithmetic."""
    p_int = _as_real(p_lev, name="p_lev")
    p_mid = _as_real(p_lay, name="p_lay")
    tv = _as_real(tv_lay, name="tv_lay")
    nlay, ncol = p_mid.shape
    if p_int.shape != (nlay + 1, ncol) or tv.shape != p_mid.shape:
        raise ValueError("geometry input shapes are inconsistent")
    if top_at_1 is None:
        top_at_1 = p_int[0, 0] < p_int[nlay - 1, 0]
    log_int = torch.log(p_int)
    log_mid = torch.log(p_mid)
    log_dp = torch.abs(log_int[1:] - log_int[:-1])
    delta_p = torch.abs(p_int[1:] - p_int[:-1])
    delta_z = (CON_RD / CON_G) * log_dp * tv

    def centers(top: bool) -> torch.Tensor:
        # This is the Fortran vertical recurrence, batched over columns.
        heights = [torch.zeros_like(delta_z[0])]
        order = range(nlay - 1, -1, -1) if top else range(nlay)
        for level in order:
            heights.append(heights[-1] + delta_z[level])
        hgtb = torch.stack(heights[::-1] if top else heights)
        if top:
            pfac = torch.abs(log_int[1:] - log_mid) / log_dp
            hgtc = hgtb[1:] + pfac * (hgtb[:-1] - hgtb[1:])
            return torch.cat((hgtc[:-1] - hgtc[1:], hgtc[-1:] - hgtb[-1:]))
        pfac = torch.abs(log_int[:-1] - log_mid) / log_dp
        hgtc = hgtb[:-1] + pfac * (hgtb[1:] - hgtb[:-1])
        return torch.cat((hgtc[:1] - hgtb[:1], hgtc[1:] - hgtc[:-1]))

    delta_zc = (centers(top_at_1) if isinstance(top_at_1, bool)
                else torch.where(top_at_1, centers(True), centers(False)))
    return delta_p, delta_z, delta_zc, top_at_1


def _repeat_gas(value: float, template: torch.Tensor) -> torch.Tensor:
    return torch.full_like(template, float(value))


def run_pre_from_host_inputs(
    inputs: Mapping[str, TensorOrString],
    *,
    prepared_daylight: tuple[torch.Tensor, torch.Tensor, int] | None = None,
) -> RrtmgpPreResult:
    """Port the observable core of `GFS_rrtmgp_pre_run`.

    Required tensor inputs use checkpoint orientation `[nlay,ncol]` for layer
    fields and `[nlay+1,ncol]` for interface fields. Non-prognostic gas values
    may be supplied explicitly when the host gas-update state is the oracle
    source; otherwise the Fortran default climatology constants are used.
    """

    p_lay = _as_real(inputs["p_lay"], name="p_lay")
    p_lev = _as_real(inputs["p_lev"], name="p_lev")
    t_lay = _as_real(inputs["t_lay"], name="t_lay")
    q_lay = torch.maximum(_as_real(inputs["q_lay"], name="q_lay"), torch.full_like(p_lay, 1.0e-6))
    tsfc = _as_real(inputs["tsfc"], name="tsfc").reshape(-1)
    if p_lay.shape != t_lay.shape or p_lay.shape != q_lay.shape:
        raise ValueError("p_lay, t_lay, and q_lay shapes are inconsistent")
    nlay, ncol = p_lay.shape

    top_at_1_input = inputs.get("top_at_1", inputs.get("top_at_1_initial"))
    top_at_1_hint = bool(_scalar_int_value({"top_at_1": top_at_1_input}, "top_at_1", 0)) if isinstance(top_at_1_input, torch.Tensor) else None
    t_lev = compute_interface_temperature(p_lay, t_lay, p_lev, tsfc, top_at_1=top_at_1_hint)
    es = torch.minimum(p_lay, fpvs(t_lay))
    qs_lay = torch.maximum(
        torch.full_like(p_lay, CON_EPSQS),
        CON_EPS * es / (p_lay + CON_EPSM1 * es),
    )
    relhum = torch.clamp(torch.maximum(torch.full_like(q_lay, CON_EPSQS), q_lay) / qs_lay, min=0.0, max=1.0)
    tv_lay = t_lay * (1.0 + CON_FVIRT * q_lay)
    delta_p, delta_z, delta_zc, top_at_1 = compute_layer_geometry(p_lev, p_lay, tv_lay, top_at_1=top_at_1_hint)

    o3_lay = inputs.get("o3_lay")
    if isinstance(o3_lay, torch.Tensor):
        o3_mass = torch.maximum(_as_real(o3_lay, name="o3_lay"), torch.full_like(q_lay, CON_EPSQS))
        vmr_o3 = torch.where(o3_mass > 0.0, o3_mass * AMDO3, torch.zeros_like(q_lay))
    else:
        vmr_o3 = _as_real(inputs["vmr_o3"], name="vmr_o3") if "vmr_o3" in inputs else torch.zeros_like(q_lay)
    vmr_h2o = torch.where(q_lay != 1.0, (q_lay / (1.0 - q_lay)) * AMDW, torch.zeros_like(q_lay))
    if all(name in inputs for name in ("vmr_co2", "vmr_n2o", "vmr_ch4", "vmr_o2")):
        vmr_co2 = _as_real(inputs["vmr_co2"], name="vmr_co2")
        vmr_n2o = _as_real(inputs["vmr_n2o"], name="vmr_n2o")
        vmr_ch4 = _as_real(inputs["vmr_ch4"], name="vmr_ch4")
        vmr_o2 = _as_real(inputs["vmr_o2"], name="vmr_o2")
    else:
        co2_year, co2_month = _date_year_month(inputs)
        vmr_o2, vmr_ch4, vmr_n2o, vmr_co2 = compute_nonprognostic_gases(
            p_lev,
            _as_real(inputs["xlon"], name="xlon"),
            _as_real(inputs["xlat"], name="xlat"),
            ico2=_scalar_int_value(inputs, "ico2", 0),
            top_at_1=top_at_1,
            con_pi=_scalar_float_value(inputs, "con_pi", math.pi),
            co2_year=co2_year,
            co2_month=co2_month,
        )

    i_sfc = 1 if not top_at_1 else nlay
    i_toa = nlay if not top_at_1 else 1
    i_sfc_ilev = 0 if not top_at_1 else i_sfc
    coszen = inputs.get("coszen")
    if isinstance(coszen, torch.Tensor):
        coszen_t = coszen.to(dtype=p_lay.dtype).reshape(-1)
    else:
        coszen_t = torch.zeros((ncol,), dtype=p_lay.dtype, device=p_lay.device)
    do_sw = _scalar_flag(inputs, "doSWrad", default=True)
    if prepared_daylight is not None:
        expected_mask, idxday, nday_value = prepared_daylight
        # Daylight is a discrete selector. Coordinates and coszen remain live,
        # differentiable inputs; a boundary crossing requires new preparation.
        actual_mask = coszen_t >= 0.0001
        torch._assert_async(torch.all(actual_mask == expected_mask),
                            "radiation daylight mask changed; prepare the call again")
    elif "nDay" in inputs and isinstance(inputs["nDay"], torch.Tensor):
        nday_value = _scalar_int_value(inputs, "nDay", 0)
    else:
        nday_value = int(((coszen_t >= 0.0001).sum()).item()) if bool(do_sw.reshape(-1)[0].item()) else 0
    if prepared_daylight is None:
        idxday = torch.zeros((max(nday_value, 1),), dtype=torch.int64, device=p_lay.device)
        if nday_value:
            idxday[:nday_value] = torch.nonzero(coszen_t >= 0.0001, as_tuple=False).reshape(-1)[:nday_value] + 1
    return RrtmgpPreResult(
        fields={
            "doSWrad": do_sw,
            "doLWrad": _scalar_flag(inputs, "doLWrad", default=True),
            "doSWclrsky": _scalar_flag(inputs, "doSWclrsky", default=True),
            "doLWclrsky": _scalar_flag(inputs, "doLWclrsky", default=True),
            "doGP_lwscat": _scalar_flag(inputs, "doGP_lwscat", default=True),
            "nGauss_angles": _scalar_int(inputs, "nGauss_angles", 1),
            "p_lay": p_lay,
            "p_lev": p_lev,
            "t_lay": t_lay,
            "t_lev": t_lev,
            "deltaZ": delta_z,
            "deltaZc": delta_zc,
            "deltaP": delta_p,
            "iSFC": torch.tensor([i_sfc], dtype=torch.int64),
            "iTOA": torch.tensor([i_toa], dtype=torch.int64),
            "top_at_1": torch.tensor([1 if top_at_1 else 0], dtype=torch.int64),
            "q_lay": q_lay,
            "qs_lay": qs_lay,
            "relhum": relhum,
            "tv_lay": tv_lay,
            "active_gases": DEFAULT_ACTIVE_GASES_TEXT,
            "active_gases_array": DEFAULT_ACTIVE_GASES_ARRAY_NETCDF_TEXT,
            "vmr_o2": vmr_o2,
            "vmr_h2o": vmr_h2o,
            "vmr_o3": vmr_o3,
            "vmr_ch4": vmr_ch4,
            "vmr_n2o": vmr_n2o,
            "vmr_co2": vmr_co2,
            "nDay": torch.tensor([nday_value], dtype=torch.int64),
            "idxday": idxday,
            "coszen": coszen_t,
            "errmsg": "",
            "errflg": torch.tensor([0], dtype=torch.int64),
        }
    )


__all__ = [
    "AMDO3",
    "AMDW",
    "RrtmgpPreResult",
    "compute_interface_temperature",
    "compute_layer_geometry",
    "compute_nonprognostic_gases",
    "fpvs",
    "run_pre_from_host_inputs",
]
