# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Torch-native GFS surface-layer kernels.

The runtime functions in this module are intentionally tensor-only: no NumPy,
NetCDF, source-oracle IO, or host synchronization belongs on the target path.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import struct
import types
from typing import Callable, Mapping, Sequence

import torch

from .cases import SURFACE_DIAG_OUTPUT_FIELDS, SURFACE_DIFF_OUTPUT_FIELDS, SurfaceCaseConfig


GRAV = 9.80665
CP = 1004.6
RD = 287.05
RV = 461.50
EPS = RD / RV
EPSM1 = RD / RV - 1.0
RVRDM1 = RV / RD - 1.0
CON_ROCP = RD / CP
CON_T0C = 273.15
CON_KARMAN = 0.4
_FPVS_TABLE_CACHE: dict[tuple[str, int | None, torch.dtype], torch.Tensor] = {}
_COMPILED_RUN_CACHE: dict[tuple[object, ...], Callable[..., tuple[torch.Tensor, ...]]] = {}
_COMPILED_REGION_CACHE: dict[tuple[object, ...], Callable[..., object]] = {}
_REGION_INPUT_FIELDS = {
    "wind": ("u1", "v1", "usfco", "vsfco"),
    "temperature": ("tskin", "t1", "prslki", "lake_t2m"),
    "humidity": ("q1", "tskin", "ps", "evap", "qsurf", "lake_q2m"),
    "ruc_temperature": ("ps", "tskin", "shflx", "t1", "lake_t2m", "zf"),
    "ruc_humidity": ("t1", "ps", "q1", "qsurf", "cdq", "wind", "evap", "lake_q2m", "zf"),
    "dewpoint": ("ps",),
}
_REGION_CONFIG_FIELDS = {
    "diffusion": ("ivegsrc", "sfc_z0_type", "redrag", "use_oceanuv", "thsfc_loc",
                  "use_lake2m", "iopt_lake", "iopt_lake_clm", "default_real_bytes"),
    "wind": ("use_oceanuv",),
    "temperature": ("thsfc_loc",),
    "humidity": (),
    "ruc_temperature": ("diag_flux", "diag_log"),
    "ruc_humidity": ("diag_flux", "diag_log"),
    "dewpoint": (),
}


def _region_signature(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device,
                value.requires_grad, value.layout)
    if isinstance(value, Mapping):
        return tuple((name, _region_signature(tensor)) for name, tensor in value.items())
    if isinstance(value, tuple):
        return tuple(_region_signature(item) for item in value)
    if isinstance(value, SurfaceCaseConfig):
        return value
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value), value)
    raise TypeError(f"unsupported surface-region argument: {type(value).__name__}")


def _run_surface_region(name: str, fn: Callable, *args: object,
                        max_fusion_size: int | None = None):
    if torch.compiler.is_compiling():
        return fn(*args)
    if name in _REGION_INPUT_FIELDS:
        fields, *rest = args
        args = ({field: fields[field] for field in _REGION_INPUT_FIELDS[name]}, *rest)
    def signature(value):
        if isinstance(value, SurfaceCaseConfig):
            return tuple((field, getattr(value, field))
                         for field in _REGION_CONFIG_FIELDS[name])
        return _region_signature(value)
    key = (name, id(fn), max_fusion_size, torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
           tuple(signature(value) for value in args))
    compiled = _COMPILED_REGION_CACHE.get(key)
    if compiled is None:
        suffix = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        code_name = f"{fn.__name__}_{suffix}"
        code = fn.__code__.replace(co_name=code_name, co_qualname=code_name)
        specialized = types.FunctionType(code, fn.__globals__, code_name,
                                         fn.__defaults__, fn.__closure__)
        specialized.__kwdefaults__ = fn.__kwdefaults__
        options = {} if max_fusion_size is None else {"max_fusion_size": max_fusion_size}
        compiled = torch.compile(specialized, fullgraph=True, dynamic=False, options=options)
        _COMPILED_REGION_CACHE[key] = compiled
    return compiled(*args)


def _c(value: float, like: torch.Tensor) -> torch.Tensor:
    # Filled on the device: no host copy or stream sync on CUDA.
    return torch.full((), value, dtype=like.dtype, device=like.device)


def _where(mask: torch.Tensor, true_value: torch.Tensor, false_value: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, true_value, false_value)


def _min(a: torch.Tensor, b: torch.Tensor | float) -> torch.Tensor:
    return torch.minimum(a, b.to(dtype=a.dtype, device=a.device) if isinstance(b, torch.Tensor) else _c(b, a))


def _max(a: torch.Tensor, b: torch.Tensor | float) -> torch.Tensor:
    return torch.maximum(a, b.to(dtype=a.dtype, device=a.device) if isinstance(b, torch.Tensor) else _c(b, a))


def _clip(a: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(a, min=lo, max=hi)


def _poly(uref: torch.Tensor, coef: Sequence[float]) -> torch.Tensor:
    result = torch.full_like(uref, coef[-1])
    for value in reversed(coef[:-1]):
        result = result * uref + value
    return result


def _round_r4(value: float) -> float:
    """Match unsuffixed Fortran real literals assigned to kind_phys constants."""

    return struct.unpack("f", struct.pack("f", float(value)))[0]


def _r4(value: float) -> float:
    return _round_r4(value)


def _r4_tuple(*values: float) -> tuple[float, ...]:
    return tuple(_round_r4(value) for value in values)


_R4_0P11 = _round_r4(0.11)
_R4_QMIN_DIAG = _round_r4(1.0e-8)

_ZNOT_M_V6_LOW = _r4_tuple(-8.396975715683501e00, -1.597898515251717e00, 2.855780863283819e-01, -1.296521881682694e-02)
_ZNOT_M_V6_MID = _r4_tuple(
    2.147264020369413e-05,
    1.739759082358234e-07,
    -1.240239171056262e-06,
    1.962282433562894e-07,
    3.281964357650687e-09,
    3.790846746036765e-10,
)
_ZNOT_M_V6_HIGH = _r4_tuple(
    -1.663993561652530e01,
    1.255457892775006e00,
    -6.139315534216305e-02,
    1.735308193700643e-03,
    -2.793849676757154e-05,
    1.840430200185075e-07,
)
_ZNOT_M_V6_CAP = _round_r4(4.579369142033410e-04)

_ZNOT_M_V7_LOW = _ZNOT_M_V6_LOW
_ZNOT_M_V7_MID = _ZNOT_M_V6_MID
_ZNOT_M_V7_HIGH = _r4_tuple(
    -1.707846930193362e01,
    1.346757797103756e00,
    -6.797293095862357e-02,
    1.931392924987349e-03,
    -3.019495980684978e-05,
    1.897534489606422e-07,
)
_ZNOT_M_V7_CAP = _round_r4(3.371427455376717e-04)

_ZNOT_T_V6_FLOOR = _round_r4(1.100000000000000e-04)
_ZNOT_T_V6_V10 = _r4_tuple(
    8.634221567969181e-04,
    -2.628501274963990e-04,
    3.333848806567684e-05,
    -2.155602086883837e-06,
    7.020346616456421e-08,
    -9.144581627678278e-10,
)
_ZNOT_T_V6_V20 = _r4_tuple(
    1.428968311457630e-04,
    -2.552246987137160e-05,
    1.871407733439947e-06,
    -6.837922749505057e-08,
    1.232380050058077e-09,
    -8.654513012535990e-12,
)
_ZNOT_T_V6_V30 = _r4_tuple(
    3.436721779020359e-05,
    -2.448983648874671e-06,
    -3.814457439412957e-08,
    8.788972147364181e-09,
    -2.945761895342535e-10,
    3.207515102100162e-12,
)
_ZNOT_T_V6_V40 = _r4_tuple(
    -7.746288511324971e-03,
    5.988240863928883e-04,
    -1.367469811838390e-05,
    -1.227668406985956e-08,
    3.939867958963747e-09,
    -3.530687797132211e-11,
)
_ZNOT_T_V6_V50 = _r4_tuple(
    -1.036679430885215e-02,
    9.503919224192534e-04,
    -3.581694433758150e-05,
    7.169872601310186e-07,
    -8.049200462388188e-09,
    4.801984186231693e-11,
    -1.187982453329086e-13,
)
_ZNOT_T_V6_CAP = _round_r4(4.751256171799112e-05)

_ZNOT_T_V7_FLOOR = _ZNOT_T_V6_FLOOR
_ZNOT_T_V7_V10 = _r4_tuple(
    8.644979973037803e-04,
    -2.633566691328004e-04,
    3.342963077911962e-05,
    -2.163419217747114e-06,
    7.052217518653943e-08,
    -9.193764479895316e-10,
)
_ZNOT_T_V7_V20 = _r4_tuple(
    1.484341646128200e-04,
    -2.680293455916390e-05,
    1.982901461144764e-06,
    -7.299148051141852e-08,
    1.325396583616614e-09,
    -9.402722450219142e-12,
)
_ZNOT_T_V7_V30 = _r4_tuple(
    -7.558911792344770e-05,
    1.659454106237737e-05,
    -1.337841892062716e-06,
    5.251986927351103e-08,
    -1.019028029546602e-09,
    7.921446674311864e-12,
)
_ZNOT_T_V7_V40 = _r4_tuple(
    3.924356617245624e-02,
    -4.588070983722060e-03,
    2.143803523428029e-04,
    -5.000813324746342e-06,
    5.817362913967911e-08,
    -2.694370426850801e-10,
)
_ZNOT_T_V7_V50 = _r4_tuple(
    -1.450062148367566e-02,
    1.329762020689302e-03,
    -5.012618091180904e-05,
    1.003683177025925e-06,
    -1.127030176632823e-08,
    6.724854483077447e-11,
    -1.663918773476178e-13,
)
_ZNOT_T_V7_CAP = _round_r4(6.840803042788488e-05)


def znot_m_v6(uref: torch.Tensor) -> torch.Tensor:
    low = torch.exp(_poly(uref, _ZNOT_M_V6_LOW))
    mid = _poly(uref, _ZNOT_M_V6_MID)
    high = torch.exp(_poly(uref, _ZNOT_M_V6_HIGH))
    return _where(uref <= 6.5, low, _where(uref <= 15.7, mid, _where(uref <= 53.0, high, torch.full_like(uref, _ZNOT_M_V6_CAP))))


def znot_m_v7(uref: torch.Tensor) -> torch.Tensor:
    low = torch.exp(_poly(uref, _ZNOT_M_V7_LOW))
    mid = _poly(uref, _ZNOT_M_V7_MID)
    high = torch.exp(_poly(uref, _ZNOT_M_V7_HIGH))
    return _where(uref <= 6.5, low, _where(uref <= 15.7, mid, _where(uref <= 53.0, high, torch.full_like(uref, _ZNOT_M_V7_CAP))))


def znot_t_v6(uref: torch.Tensor) -> torch.Tensor:
    v10 = _poly(uref, _ZNOT_T_V6_V10)
    v20 = _poly(uref, _ZNOT_T_V6_V20)
    v30 = _poly(uref, _ZNOT_T_V6_V30)
    v40 = _poly(uref, _ZNOT_T_V6_V40)
    v50 = _poly(uref, _ZNOT_T_V6_V50)
    out = torch.full_like(uref, _ZNOT_T_V6_CAP)
    out = _where(uref <= 80.0, v50, out)
    out = _where(uref <= 53.3, v40, out)
    out = _where(uref <= 42.2, v30, out)
    out = _where(uref <= 21.6, v20, out)
    out = _where(uref <= 15.4, v10, out)
    return _where(uref < 5.9, torch.full_like(uref, _ZNOT_T_V6_FLOOR), out)


def znot_t_v7(uref: torch.Tensor) -> torch.Tensor:
    v10 = _poly(uref, _ZNOT_T_V7_V10)
    v20 = _poly(uref, _ZNOT_T_V7_V20)
    v30 = _poly(uref, _ZNOT_T_V7_V30)
    v40 = _poly(uref, _ZNOT_T_V7_V40)
    v50 = _poly(uref, _ZNOT_T_V7_V50)
    out = torch.full_like(uref, _ZNOT_T_V7_CAP)
    out = _where(uref <= 80.0, v50, out)
    out = _where(uref <= 53.0, v40, out)
    out = _where(uref <= 42.6, v30, out)
    out = _where(uref <= 21.6, v20, out)
    out = _where(uref <= 15.4, v10, out)
    return _where(uref < 5.9, torch.full_like(uref, _ZNOT_T_V7_FLOOR), out)


def stability(
    z1: torch.Tensor,
    zvfun: torch.Tensor,
    gdx: torch.Tensor,
    tv1: torch.Tensor,
    thv1: torch.Tensor,
    wind: torch.Tensor,
    z0max: torch.Tensor,
    ztmax: torch.Tensor,
    tvs: torch.Tensor,
    *,
    thsfc_loc: bool,
) -> dict[str, torch.Tensor]:
    one = _c(1.0, z1)
    zero = _c(0.0, z1)
    alpha4 = _c(20.0, z1)
    z1i = one / z1
    xkzo0 = _where(gdx >= 3000.0, one, gdx / 3000.0)
    tem1 = tv1 - tvs
    tem2 = xkzo0 * zvfun
    xkzo = _where(tem1 > zero, _min(_max(tem2, 0.05), xkzo0), xkzo0)
    zolmax = _c(0.3, z1) / torch.sqrt(xkzo)
    dtv0 = thv1 - tvs
    adtv = _max(torch.abs(dtv0), 0.001)
    dtv = _where(dtv0 >= zero, adtv, -adtv)
    wind2 = wind * wind
    if thsfc_loc:
        denominator = (thv1 + tvs) * wind * wind
        denominator = _where(wind2 >= 1.0e-12, denominator, (thv1 + tvs) * 1.0e-12)
        rb = _max((GRAV + GRAV) * dtv * z1 / denominator, -5000.0)
    else:
        denominator = tv1 * wind * wind
        denominator = _where(wind2 >= 1.0e-12, denominator, tv1 * 1.0e-12)
        rb = _max(GRAV * dtv * z1 / denominator, -5000.0)
    z0_inverse = one / z0max
    zt_inverse = one / ztmax
    fm0 = torch.log((z0max + z1) * z0_inverse)
    fh0 = torch.log((ztmax + z1) * zt_inverse)
    fm10_0 = torch.log((z0max + 10.0) * z0_inverse)
    fh2_0 = torch.log((ztmax + 2.0) * zt_inverse)
    hlinf0 = rb * fm0 * fm0 / fh0
    hlinf = _clip(hlinf0, -10.0, 0.0) + _where(hlinf0 > zero, _min(hlinf0, zolmax), torch.zeros_like(hlinf0))

    stable = dtv >= zero
    hl1_init = hlinf
    tem1s = hlinf * z1i
    hl0inf = z0max * tem1s
    hltinf = ztmax * tem1s
    aa = torch.sqrt(_max(one + alpha4 * hlinf, 1.0e-30))
    aa0 = torch.sqrt(_max(one + alpha4 * hl0inf, 1.0e-30))
    bb = aa
    bb0 = torch.sqrt(_max(one + alpha4 * hltinf, 1.0e-30))
    pm = aa0 - aa + torch.log((aa + one) / (aa0 + one))
    ph = bb0 - bb + torch.log((bb + one) / (bb0 + one))
    fms = fm0 - pm
    fhs = fh0 - ph
    hl1_corrected = _min(fms * fms * rb / fhs, zolmax)
    hl1_stable = _where(hlinf > 0.25, hl1_corrected, hl1_init)

    tem1s2 = hl1_stable * z1i
    hl0 = z0max * tem1s2
    hlt = ztmax * tem1s2
    aa = torch.sqrt(_max(one + alpha4 * hl1_stable, 1.0e-30))
    aa0 = torch.sqrt(_max(one + alpha4 * hl0, 1.0e-30))
    bb = aa
    bb0 = torch.sqrt(_max(one + alpha4 * hlt, 1.0e-30))
    pm_stable = aa0 - aa + torch.log((one + aa) / (one + aa0))
    ph_stable = bb0 - bb + torch.log((one + bb) / (one + bb0))
    hl110 = hl1_stable * 10.0 * z1i
    aa10 = torch.sqrt(_max(one + alpha4 * hl110, 1.0e-30))
    pm10_stable = aa0 - aa10 + torch.log((one + aa10) / (one + aa0))
    hl12 = (hl1_stable + hl1_stable) * z1i
    bb2 = torch.sqrt(_max(one + alpha4 * hl12, 1.0e-30))
    ph2_stable = bb0 - bb2 + torch.log((one + bb2) / (one + bb0))

    olinf = z1 / _where(torch.abs(hlinf) < 1.0e-30, torch.full_like(hlinf, -1.0e-30), hlinf)
    tem1u = 50.0 * z0max
    hlinf_unstable = _where(torch.abs(olinf) <= tem1u, _max(-z1 / tem1u, -10.0), hlinf)
    rational = hlinf_unstable >= -0.5
    hl1u = hlinf_unstable
    pmu_r = (-3.975 + 12.32 * hl1u) * hl1u / (one + (-7.755 + 6.041 * hl1u) * hl1u)
    phu_r = (-7.941 + 24.75 * hl1u) * hl1u / (one + (-8.705 + 7.899 * hl1u) * hl1u)
    hl110u = hl1u * 10.0 * z1i
    pm10u_r = (-3.975 + 12.32 * hl110u) * hl110u / (one + (-7.755 + 6.041 * hl110u) * hl110u)
    hl12u = (hl1u + hl1u) * z1i
    ph2u_r = (-7.941 + 24.75 * hl12u) * hl12u / (one + (-8.705 + 7.899 * hl12u) * hl12u)
    hl1_abs = _max(-hlinf_unstable, 1.0e-30)
    sqrt_inv = one / torch.sqrt(hl1_abs)
    pmu_l = torch.log(hl1_abs) + 2.0 * torch.sqrt(sqrt_inv) - 0.8776
    phu_l = torch.log(hl1_abs) + 0.5 * sqrt_inv + 1.386
    hl110_abs = _max(hl1_abs * 10.0 * z1i, 1.0e-30)
    pm10u_l = torch.log(hl110_abs) + 2.0 / torch.sqrt(torch.sqrt(hl110_abs)) - 0.8776
    hl12_abs = _max((hl1_abs + hl1_abs) * z1i, 1.0e-30)
    ph2u_l = torch.log(hl12_abs) + 0.5 / torch.sqrt(hl12_abs) + 1.386
    pm_unstable = _where(rational, pmu_r, pmu_l)
    ph_unstable = _where(rational, phu_r, phu_l)
    pm10_unstable = _where(rational, pm10u_r, pm10u_l)
    ph2_unstable = _where(rational, ph2u_r, ph2u_l)

    pm_final = _where(stable, pm_stable, pm_unstable)
    ph_final = _where(stable, ph_stable, ph_unstable)
    pm10_final = _where(stable, pm10_stable, pm10_unstable)
    ph2_final = _where(stable, ph2_stable, ph2_unstable)
    fm = fm0 - pm_final
    fh = fh0 - ph_final
    fm10 = fm10_0 - pm10_final
    fh2 = fh2_0 - ph2_final
    # A Python scalar numerator uses reciprocal-multiply, not the source division.
    ca_squared = _c(CON_KARMAN * CON_KARMAN, fm)
    floor = _c(0.00001, z1) / z1
    cm = _max(ca_squared / (fm * fm), floor)
    ch = _max(ca_squared / (fm * fh), floor)
    stress = cm * wind * wind
    ustar = torch.sqrt(_max(stress, 0.0))
    return {
        "rb": rb,
        "fm": fm,
        "fh": fh,
        "fm10": fm10,
        "fh2": fh2,
        "cm": cm,
        "ch": ch,
        "stress": stress,
        "ustar": ustar,
    }


def _land_z0max(fields: Mapping[str, torch.Tensor], cfg: SurfaceCaseConfig) -> torch.Tensor:
    q = fields
    z0max = _max(_min(0.01 * q["z0rl_lnd"], q["z1"]), 1.0e-6)
    tem1 = 1.0 - q["shdmax"]
    tem2 = tem1 * tem1
    tem1 = 1.0 - tem2
    if cfg.ivegsrc == 1:
        veg = q["vegtype"]
        adjusted = torch.exp(tem2 * torch.log(_c(0.01, z0max)) + tem1 * torch.log(z0max))
        adjusted = _where(veg == 10, torch.exp(tem2 * torch.log(_c(0.01, z0max)) + tem1 * torch.log(_c(0.07, z0max))), adjusted)
        adjusted = _where(veg == 6, torch.exp(tem2 * torch.log(_c(0.01, z0max)) + tem1 * torch.log(_c(0.05, z0max))), adjusted)
        adjusted = _where((veg == 7) | (veg == 16), torch.full_like(z0max, 0.01), adjusted)
        z0max = adjusted
    elif cfg.ivegsrc == 2:
        veg = q["vegtype"]
        adjusted = torch.exp(tem2 * torch.log(_c(0.01, z0max)) + tem1 * torch.log(z0max))
        adjusted = _where(veg == 7, torch.exp(tem2 * torch.log(_c(0.01, z0max)) + tem1 * torch.log(_c(0.07, z0max))), adjusted)
        adjusted = _where(veg == 8, torch.exp(tem2 * torch.log(_c(0.01, z0max)) + tem1 * torch.log(_c(0.05, z0max))), adjusted)
        adjusted = _where((veg == 9) | (veg == 11), torch.full_like(z0max, 0.01), adjusted)
        z0max = adjusted
    z0max = _where(q["z0pert"] != 0.0, z0max * torch.pow(_c(10.0, z0max), q["z0pert"]), z0max)
    return _max(z0max, 1.0e-6)


def _land_ice_ztmax(z0max: torch.Tensor, sigmaf: torch.Tensor, ustar: torch.Tensor, ztpert: torch.Tensor | None = None) -> torch.Tensor:
    czilc = torch.pow(_c(10.0, z0max), -4.0 * z0max)
    czilc = _clip(czilc, 0.08, 0.8)
    tem1 = 1.0 - sigmaf
    ztmax = z0max * torch.exp(-(czilc * tem1 * tem1) * CON_KARMAN * 258.2 * torch.sqrt(_max(ustar * z0max, 0.0)))
    if ztpert is not None:
        ztmax = _where(ztpert != 0.0, ztmax * torch.pow(_c(10.0, ztmax), ztpert), ztmax)
    return _max(ztmax, 1.0e-6)


def _apply_stability_outputs(
    out: dict[str, torch.Tensor],
    suffix: str,
    mask: torch.Tensor,
    values: Mapping[str, torch.Tensor],
) -> None:
    for name in ("rb", "fm", "fh", "fm10", "fh2", "cm", "ch", "stress", "ustar"):
        field = f"{name}_{suffix}"
        out[field] = _where(mask, values[name], out[field])


_STABILITY_OUTPUT_NAMES = ("rb", "fm", "fh", "fm10", "fh2", "cm", "ch", "stress", "ustar")


def _stability_region(z1, zvfun, gdx, tv1, thv1, wind, z0max, ztmax, tvs, thsfc_loc):
    values = stability(z1, zvfun, gdx, tv1, thv1, wind, z0max, ztmax, tvs,
                       thsfc_loc=thsfc_loc)
    return tuple(values[name] for name in _STABILITY_OUTPUT_NAMES)


def _evaluate_stability(*args, thsfc_loc, compile_stability, max_fusion_size):
    # Float32 evaluates in float64 and rounds each output once (no-op in float64). Every column
    # computes both stability branches and all three surface types; in the discarded lanes float32
    # backward factors (x**-1.5 at the 1e-30 floor, (a/b)/b) overflow, and 0 * inf is NaN.
    working = args[0].dtype
    if working == torch.float32:
        args = tuple(value.double() for value in args)
    if not compile_stability:
        values = stability(*args, thsfc_loc=thsfc_loc)
    else:
        values = dict(zip(_STABILITY_OUTPUT_NAMES, _run_surface_region(
            "stability", _stability_region, *args, thsfc_loc, max_fusion_size=max_fusion_size), strict=True))
    return {name: value.to(working) for name, value in values.items()}


def sfc_diff_run(fields: Mapping[str, torch.Tensor], cfg: SurfaceCaseConfig, *,
                 _compile_stability: bool = False,
                 _region_fusion_size: int | None = None) -> dict[str, torch.Tensor]:
    q = fields
    out = dict(q)
    one = torch.ones_like(q["ps"])
    zero = torch.zeros_like(q["ps"])
    active = q["flag_iter"] | q["flag_lakefreeze"]
    for field in ("ztmax_lnd", "ztmax_ice", "ztmax_wat"):
        out[field] = _where(active, one, out[field])
    virtfac = 1.0 + RVRDM1 * _max(q["q1"], 1.0e-8)
    tv1 = q["t1"] * virtfac
    theta = q["t1"] * q["prslki"] if cfg.thsfc_loc else q["t1"] / q["prslk1"]
    thv1 = theta * virtfac
    gdx = torch.sqrt(q["garea"])
    zvfun = _where(active, zero, q.get("zvfun", zero))

    dry_mask = active & q["dry"]
    z0_lnd = _land_z0max(q, cfg)
    zt_lnd = _land_ice_ztmax(z0_lnd, q["sigmaf"], q["ustar_lnd"], q["ztpert"])
    tvs_lnd = 0.5 * (q["tsurf_lnd"] + q["tskin_lnd"])
    if not cfg.thsfc_loc:
        tvs_lnd = tvs_lnd / q["prsik1"]
    tvs_lnd = tvs_lnd * virtfac
    tem1 = _clip((z0_lnd - 0.1) / (1.0 - 0.1), 0.0, 1.0)
    tem2 = _max(q["sigmaf"], 0.1)
    roughness_vegetation = tem1 * tem2
    # The clipped-zero region is constant; do not evaluate sqrt's singular slope there.
    zvfun_lnd = _where(
        roughness_vegetation > 0.0,
        torch.sqrt(_where(roughness_vegetation > 0.0, roughness_vegetation, one)),
        zero,
    )
    land = _evaluate_stability(q["z1"], zvfun_lnd, gdx, tv1, thv1, q["wind"], z0_lnd, zt_lnd, tvs_lnd,
                               thsfc_loc=cfg.thsfc_loc, compile_stability=_compile_stability,
                               max_fusion_size=_region_fusion_size)
    out["ztmax_lnd"] = _where(dry_mask, zt_lnd, out["ztmax_lnd"])
    zvfun = _where(dry_mask, zvfun_lnd, zvfun)
    _apply_stability_outputs(out, "lnd", dry_mask, land)

    icy_mask = active & q["icy"]
    z0_ice = _max(_min(0.01 * q["z0rl_ice"], q["z1"]), 1.0e-6)
    zt_ice = _land_ice_ztmax(z0_ice, q["sigmaf"], q["ustar_ice"])
    tvs_ice = 0.5 * (q["tsurf_ice"] + q["tskin_ice"])
    if not cfg.thsfc_loc:
        tvs_ice = tvs_ice / q["prsik1"]
    tvs_ice = tvs_ice * virtfac
    ice = _evaluate_stability(q["z1"], zero, gdx, tv1, thv1, q["wind"], z0_ice, zt_ice, tvs_ice,
                              thsfc_loc=cfg.thsfc_loc, compile_stability=_compile_stability,
                              max_fusion_size=_region_fusion_size)
    out["ztmax_ice"] = _where(icy_mask, zt_ice, out["ztmax_ice"])
    zvfun = _where(icy_mask, zero, zvfun)
    _apply_stability_outputs(out, "ice", icy_mask, ice)

    wet_mask = active & q["wet"]
    tvs_wat = 0.5 * (q["tsurf_wat"] + q["tskin_wat"])
    if not cfg.thsfc_loc:
        tvs_wat = tvs_wat / q["prsik1"]
    tvs_wat = tvs_wat * virtfac
    if cfg.use_oceanuv:
        wind10m = torch.sqrt((q["u10m"] - q["usfco"]) * (q["u10m"] - q["usfco"]) + (q["v10m"] - q["vsfco"]) * (q["v10m"] - q["vsfco"]))
        windrel = torch.sqrt((q["u1"] - q["usfco"]) * (q["u1"] - q["usfco"]) + (q["v1"] - q["vsfco"]) * (q["v1"] - q["vsfco"]))
    else:
        wind10m = torch.sqrt(q["u10m"] * q["u10m"] + q["v10m"] * q["v10m"])
        windrel = q["wind"]
    z0_wat = _max(_min(0.01 * q["z0rl_wat"], q["z1"]), 1.0e-6)
    restar = _max(q["ustar_wat"] * z0_wat * (1.0 / 1.4e-5), 1.0e-6)
    rat = _min(2.67 * torch.sqrt(torch.sqrt(restar)) - 2.57, 7.0)
    zt_wat = _max(z0_wat * torch.exp(-rat), 1.0e-6)
    if cfg.sfc_z0_type == 6:
        zt_wat = znot_t_v6(wind10m)
    elif cfg.sfc_z0_type == 7:
        zt_wat = znot_t_v7(wind10m)
    elif cfg.sfc_z0_type > 0:
        out["errflg"] = torch.ones((), dtype=torch.int64, device=q["ps"].device)
        out["zvfun"] = zvfun
        return out
    water = _evaluate_stability(q["z1"], zero, gdx, tv1, thv1, windrel, z0_wat, zt_wat, tvs_wat,
                                thsfc_loc=cfg.thsfc_loc, compile_stability=_compile_stability,
                                max_fusion_size=_region_fusion_size)
    out["ztmax_wat"] = _where(wet_mask, zt_wat, out["ztmax_wat"])
    zvfun = _where(wet_mask, zero, zvfun)
    _apply_stability_outputs(out, "wat", wet_mask, water)

    viscous_coefficient = 0.11 if cfg.default_real_bytes == 8 else _R4_0P11
    viscous = torch.div(viscous_coefficient * 1.4e-5, _max(out["ustar_wat"], 1.0e-30))
    z0_wave = viscous + 0.01 * q["z0rl_wav"]
    z0_charnock = viscous + (0.018 / GRAV) * out["ustar_wat"] * out["ustar_wat"]
    cap = 0.00317 if cfg.redrag else 0.1
    wave_ok = (q["lakefrac"] == 0.0) & (q["fice"] == 0.0) & (q["z0rl_wav"] > 1.0e-7) & (q["z0rl_wav"] < 0.1)
    wav_out = (q["z0rl_wav"] <= 1.0e-7) | (q["z0rl_wav"] >= 0.1)
    z0rl_from_wave = 100.0 * _max(_min(z0_wave, cap), 1.0e-7)
    z0rl_from_charnock = 100.0 * _max(_min(z0_charnock, cap), 1.0e-7)
    z0rl_new = torch.full_like(q["z0rl_wat"], 1.0e-4)
    if cfg.sfc_z0_type == -1:
        z0rl_new = _where(wave_ok, z0rl_from_wave, _where(wav_out, z0rl_from_charnock, z0rl_new))
    elif cfg.sfc_z0_type == 0:
        z0rl_new = z0rl_from_charnock
    elif cfg.sfc_z0_type == 6:
        z0rl_new = 100.0 * znot_m_v6(wind10m)
    elif cfg.sfc_z0_type == 7:
        z0rl_new = 100.0 * znot_m_v7(wind10m)
    out["z0rl_wat"] = _where(wet_mask, z0rl_new, out["z0rl_wat"])
    out["zvfun"] = zvfun
    out["errflg"] = torch.zeros((), dtype=torch.int64, device=q["ps"].device)
    return out


def _fpvsx(t: torch.Tensor) -> torch.Tensor:
    con_psat = _c(610.78, t)
    con_ttp = _c(273.16, t)
    con_rv = _c(461.50, t)
    con_cvap = _c(1846.0, t)
    con_cliq = _c(4185.5, t)
    con_csol = _c(2106.0, t)
    con_hvap = _c(2.5e6, t)
    con_hfus = _c(3.3358e5, t)
    tice = con_ttp - 20.0
    tr = con_ttp / t
    xponal = -(con_cvap - con_cliq) / con_rv
    xponbl = xponal + con_hvap / (con_rv * con_ttp)
    xponai = -(con_cvap - con_csol) / con_rv
    xponbi = xponai + (con_hvap + con_hfus) / (con_rv * con_ttp)
    pvl = con_psat * torch.pow(tr, xponal) * torch.exp(xponbl * (1.0 - tr))
    pvi = con_psat * torch.pow(tr, xponai) * torch.exp(xponbi * (1.0 - tr))
    w = (t - tice) / (con_ttp - tice)
    mixed = w * pvl + (1.0 - w) * pvi
    return _where(t >= con_ttp, pvl, _where(t < tice, pvi, mixed))


def fpvs_table_like(t: torch.Tensor) -> torch.Tensor:
    """Return the cached saturation-vapor-pressure lookup table for a tensor context."""

    key = (t.device.type, t.device.index, t.dtype)
    table = _FPVS_TABLE_CACHE.get(key)
    if table is not None:
        return table
    device = torch.device(t.device.type if t.device.index is None else f"{t.device.type}:{t.device.index}")
    # gpvs forms each point from the lower endpoint, not from both endpoints.
    values = 180.0 + torch.arange(7501, dtype=t.dtype, device=device) * ((330.0 - 180.0) / 7500)
    table = _fpvsx(values)
    _FPVS_TABLE_CACHE[key] = table
    return table


def fpvs(t: torch.Tensor, table: torch.Tensor | None = None) -> torch.Tensor:
    if table is None:
        table = fpvs_table_like(t)
    c2 = (7501.0 - 1.0) / (330.0 - 180.0)
    c1 = 1.0 - 180.0 * c2
    xj = torch.clamp(c1 + c2 * t, min=1.0, max=7501.0)
    jx = torch.minimum(torch.floor(xj), torch.full_like(xj, 7500.0)).to(torch.int64)
    idx = jx - 1
    base = table[idx]
    nxt = table[idx + 1]
    return base + (xj - jx.to(t.dtype)) * (nxt - base)


def _mix_surface_field(state: Mapping[str, torch.Tensor], base: str) -> torch.Tensor:
    dry = state["dry"]
    icy = state["icy"]
    wet = state["wet"]
    land = state[f"{base}_lnd"]
    ice = state[f"{base}_ice"]
    water = state[f"{base}_wat"]
    return _where(wet, water, _where(icy, ice, land))


def _diagnostic_geometry(q, cfg):
    fm = q["fm"] if "fm" in q else _mix_surface_field(q, "fm")
    fh = q["fh"] if "fh" in q else _mix_surface_field(q, "fh")
    fm10 = q["fm10"] if "fm10" in q else _mix_surface_field(q, "fm10")
    fh2 = q["fh2"] if "fh2" in q else _mix_surface_field(q, "fh2")
    ust = q["ust"] if "ust" in q else _mix_surface_field(q, "ustar")

    finite_diag = (torch.abs(fm) > 0.0) & (torch.abs(fh) > 0.0) & (torch.abs(fh2) > 0.0)
    safe_fm = _where(torch.abs(fm) > 0.0, fm, torch.ones_like(fm))
    safe_fh = _where(torch.abs(fh) > 0.0, fh, torch.ones_like(fh))
    safe_fh2 = _where(torch.abs(fh2) > 0.0, fh2, torch.ones_like(fh2))
    have_2m = (q["use_lake_model"] > 0) & cfg.use_lake2m & (cfg.iopt_lake == cfg.iopt_lake_clm)
    fhi = fh2 / safe_fh
    wrk = 1.0 - fhi
    return fm, fh, fm10, fh2, ust, finite_diag, safe_fm, safe_fh, safe_fh2, have_2m, fhi, wrk


def _wind_diagnostics(q, cfg, geometry):
    fm, fh, fm10, fh2, ust, finite_diag, safe_fm, safe_fh, safe_fh2, have_2m, fhi, wrk = geometry
    nan = torch.full_like(fm, float("nan"))
    f10m_raw = fm10 / safe_fm
    f10m = _where(finite_diag, f10m_raw, nan)
    if cfg.use_oceanuv:
        u10m_raw = q["usfco"] + f10m_raw * (q["u1"] - q["usfco"])
        v10m_raw = q["vsfco"] + f10m_raw * (q["v1"] - q["vsfco"])
    else:
        u10m_raw = f10m_raw * q["u1"]
        v10m_raw = f10m_raw * q["v1"]
    u10m_diag = _where(finite_diag, u10m_raw, nan)
    v10m_diag = _where(finite_diag, v10m_raw, nan)

    return f10m, u10m_diag, v10m_diag


def _optional_lake_diagnostic(q, name, have_2m, value):
    if name in q:
        return _where(have_2m, q[name], value)
    torch._assert_async((~have_2m).all(), "active lake diagnostic needs its host buffer")
    return value


def _nonruc_temperature(q, cfg, geometry):
    fm, fh, fm10, fh2, ust, finite_diag, safe_fm, safe_fh, safe_fh2, have_2m, fhi, wrk = geometry
    t2_nonruc_base = q["tskin"] * wrk + q["t1"] * (q["prslki"] if cfg.thsfc_loc else 1.0) * fhi - (GRAV + GRAV) / CP
    t2_nonruc = _optional_lake_diagnostic(q, "lake_t2m", have_2m, t2_nonruc_base)
    return t2_nonruc, _where(finite_diag, t2_nonruc, torch.full_like(fm, float("nan")))


def _nonruc_humidity(q, cfg, geometry, t2_nonruc, fpvs_table):
    fm, fh, fm10, fh2, ust, finite_diag, safe_fm, safe_fh, safe_fh2, have_2m, fhi, wrk = geometry
    qmin_diag = _R4_QMIN_DIAG
    q1_floor = _max(q["q1"], qmin_diag)
    fpvs_tskin = fpvs(q["tskin"], fpvs_table)
    qss_skin = EPS * fpvs_tskin / (q["ps"] + EPSM1 * fpvs_tskin)
    q2_nonruc_base = _where(q["evap"] >= 0.0, q["qsurf"] * wrk + q1_floor * fhi, qss_skin * wrk + q1_floor * fhi)
    q2_nonruc = _optional_lake_diagnostic(q, "lake_q2m", have_2m, q2_nonruc_base)
    fpvs_t2_nonruc = fpvs(t2_nonruc, fpvs_table)
    qss_t2 = EPS * fpvs_t2_nonruc / (q["ps"] + EPSM1 * fpvs_t2_nonruc)
    q2_nonruc = _min(q2_nonruc, qss_t2)
    return _where(finite_diag, q2_nonruc, torch.full_like(fm, float("nan")))


def _ruc_temperature(q, cfg, geometry):
    fm, fh, fm10, fh2, ust, finite_diag, safe_fm, safe_fh, safe_fh2, have_2m, fhi, wrk = geometry
    if cfg.diag_flux:
        thcon = torch.pow(100000.0 / q["ps"], CON_ROCP)
        chs2 = ust * CON_KARMAN / safe_fh2
        th2m = q["tskin"] * thcon - q["shflx"] / chs2
        t2_ruc_base = th2m / thcon
    else:
        t2_ruc_base = q["tskin"] * wrk + q["t1"] * fhi - (GRAV + GRAV) / CP
    t2_ruc = _where(have_2m, q["lake_t2m"], t2_ruc_base)
    if cfg.diag_log:
        d_t = q["t1"] - q["tskin"]
        ff_t = _clip(1.0 - d_t / 10.0, 0.01, 1.0)
        fac_t = torch.log((2.0 + 0.05) / (0.05 + ff_t)) / torch.log((q["zf"] + 0.05) / (0.05 + ff_t))
        t2_alt = q["tskin"] + fac_t * d_t
        t2_ruc = _where((~have_2m) & (d_t > 0.0), t2_alt, t2_ruc)
    lower_t = torch.minimum(q["tskin"], q["t1"])
    upper_t = torch.maximum(q["tskin"], q["t1"])
    t2_ruc_clamped = torch.minimum(upper_t, torch.maximum(lower_t, t2_ruc))
    t2_ruc = _where(~have_2m, t2_ruc_clamped, t2_ruc)
    return t2_ruc, _where(finite_diag, t2_ruc, torch.full_like(fm, float("nan")))


def _ruc_humidity(q, cfg, geometry, t2_ruc, fpvs_table):
    fm, fh, fm10, fh2, ust, finite_diag, safe_fm, safe_fh, safe_fh2, have_2m, fhi, wrk = geometry
    qmin_diag = _R4_QMIN_DIAG
    fpvs_t1 = fpvs(q["t1"], fpvs_table)
    qss_t1 = EPS * fpvs_t1 / (q["ps"] + EPSM1 * fpvs_t1)
    q1c = _min(q["q1"], qss_t1)
    qv1 = q1c / (1.0 - q1c)
    qsfcmr = q["qsurf"] / (1.0 - q["qsurf"])
    if cfg.diag_flux:
        chs = q["cdq"] * q["wind"]
        cqs = chs
        chs2 = ust * CON_KARMAN / safe_fh2
        cqs2 = chs2
        qsfcprox = _max(qv1 + q["evap"] / cqs, qmin_diag)
        x2m = _max(qsfcprox - q["evap"] / cqs2, qmin_diag)
        q2_ruc_base = x2m / (1.0 + x2m)
    else:
        q2_ruc_base = q["qsurf"] * wrk + _max(q1c, qmin_diag) * fhi
    q2_ruc = _where(have_2m, q["lake_q2m"], q2_ruc_base)
    if cfg.diag_log:
        d_q = qv1 - qsfcmr
        ff_q = _clip(1.0 - d_q / 0.003, 0.01, 1.0)
        fac_q = torch.log((2.0 + 0.05) / (0.05 + ff_q)) / torch.log((q["zf"] + 0.05) / (0.05 + ff_q))
        q2_alt = (qsfcmr + fac_q * d_q) / (1.0 + qsfcmr + fac_q * d_q)
        q2_ruc = _where((~have_2m) & (d_q > 0.0), q2_alt, q2_ruc)
    lower_q = torch.minimum(q["qsurf"], q1c)
    upper_q = torch.maximum(q["qsurf"], q1c)
    q2_ruc_clamped = torch.minimum(upper_q, torch.maximum(lower_q, q2_ruc))
    q2_ruc = _where(~have_2m, q2_ruc_clamped, q2_ruc)
    fpvs_t2_ruc = fpvs(t2_ruc, fpvs_table)
    qss_t2_ruc = EPS * fpvs_t2_ruc / (q["ps"] + EPSM1 * fpvs_t2_ruc)
    q2_ruc = _where(~have_2m, _min(q2_ruc, qss_t2_ruc), q2_ruc)
    if cfg.diag_flux:
        q2_ruc = _where(~have_2m, _min(q2_ruc, 1.05 * q1c), q2_ruc)
    return q2_ruc, _where(finite_diag, q2_ruc, torch.full_like(fm, float("nan")))


def _dewpoint(q, geometry, t2_ruc, q2_ruc):
    finite_diag = geometry[5]
    qmin_diag = _R4_QMIN_DIAG
    qv = _max(q2_ruc / (1.0 - q2_ruc), qmin_diag)
    tem = _max(q["ps"] * qv / (EPS - EPSM1 * qv), qmin_diag)
    dpt2m = 243.5 / (17.67 / torch.log(tem / 611.2) - 1.0) + CON_T0C
    dpt2m = _min(dpt2m, t2_ruc)

    return _where(finite_diag, dpt2m, torch.full_like(dpt2m, float("nan")))


def sfc_diag_run(fields: Mapping[str, torch.Tensor], cfg: SurfaceCaseConfig, *,
                 fpvs_table: torch.Tensor | None = None,
                 _geometry: tuple[torch.Tensor, ...] | None = None,
                 _compile_regions: bool = False,
                 _region_fusion_size: int | None = None,
                 _diag_error: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    q = fields
    out = dict(q)
    if fpvs_table is None:
        fpvs_table = fpvs_table_like(q["ps"])
    geometry = _diagnostic_geometry(q, cfg) if _geometry is None else _geometry
    def apply(name, fn, *args):
        return (_run_surface_region(name, fn, *args, max_fusion_size=_region_fusion_size)
                if _compile_regions else fn(*args))
    out["f10m"], out["u10m_diag"], out["v10m_diag"] = apply("wind", _wind_diagnostics, q, cfg, geometry)
    if cfg.lsm == cfg.lsm_ruc:
        t2, out["t2m"] = apply("ruc_temperature", _ruc_temperature, q, cfg, geometry)
        q2, out["q2m"] = apply("ruc_humidity", _ruc_humidity, q, cfg, geometry, t2, fpvs_table)
        out["dpt2m"] = apply("dewpoint", _dewpoint, q, geometry, t2, q2)
    else:
        t2, out["t2m"] = apply("temperature", _nonruc_temperature, q, cfg, geometry)
        out["q2m"] = apply("humidity", _nonruc_humidity, q, cfg, geometry, t2, fpvs_table)
        out["dpt2m"] = torch.full_like(t2, float("nan"))
    out["errflg_diag"] = (torch.zeros((), dtype=torch.int64, device=q["ps"].device)
                           if _diag_error is None else _diag_error)
    return out


def run_sfc_diff_two_passes(fields: Mapping[str, torch.Tensor], cfg: SurfaceCaseConfig, *,
                          _compile_stability: bool = False,
                          _region_fusion_size: int | None = None) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    first = sfc_diff_run(fields, cfg, _compile_stability=_compile_stability,
                         _region_fusion_size=_region_fusion_size)
    second = sfc_diff_run(first, cfg, _compile_stability=_compile_stability,
                          _region_fusion_size=_region_fusion_size)
    return first, second


def run_gfs_surface_layer(
    fields: Mapping[str, torch.Tensor],
    cfg: SurfaceCaseConfig,
    *,
    fpvs_table: torch.Tensor | None = None,
    _compile_regions: bool = False,
    _region_fusion_size: int | None = None,
    _compile_stability: bool = False,
) -> dict[str, torch.Tensor]:
    if _compile_regions and not _compile_stability:
        if fpvs_table is None:
            fpvs_table = fpvs_table_like(fields["ps"])
        values = _run_surface_region("diffusion", _diffusion_geometry_block, fields, cfg,
                                     max_fusion_size=_region_fusion_size)
        names = (*SURFACE_DIFF_OUTPUT_FIELDS, "errflg")
        second = {**fields, **dict(zip(names, values[:len(names)], strict=True))}
        return sfc_diag_run(second, cfg, fpvs_table=fpvs_table,
                            _geometry=values[len(names):-1], _compile_regions=True,
                            _region_fusion_size=_region_fusion_size,
                            _diag_error=values[-1])
    _, second = run_sfc_diff_two_passes(fields, cfg, _compile_stability=_compile_stability,
                                       _region_fusion_size=_region_fusion_size)
    return sfc_diag_run(second, cfg, fpvs_table=fpvs_table,
                        _compile_regions=_compile_regions,
                        _region_fusion_size=_region_fusion_size)


def _diffusion_geometry_block(fields, cfg):
    _, second = run_sfc_diff_two_passes(fields, cfg)
    names = (*SURFACE_DIFF_OUTPUT_FIELDS, "errflg")
    geometry = _diagnostic_geometry(second, cfg)
    diag_error = torch.zeros((), dtype=torch.int64, device=fields["ps"].device)
    return tuple(second[name] for name in names) + geometry + (diag_error,)


def run_gfs_surface_layer_regional_compiled(fields, cfg, *, max_fusion_size: int | None = None,
                                           stability_only: bool = False,
                                           fpvs_table: torch.Tensor | None = None):
    """Opt-in same-science regions with independent diagnostic dependencies."""
    return run_gfs_surface_layer(fields, cfg, fpvs_table=fpvs_table, _compile_regions=True,
                                 _region_fusion_size=max_fusion_size,
                                 _compile_stability=stability_only)


def _surface_output_names(fields: Mapping[str, torch.Tensor], output_names: Sequence[str] | None) -> tuple[str, ...]:
    if output_names is not None:
        return tuple(output_names)
    return tuple(dict.fromkeys((*fields.keys(), *SURFACE_DIFF_OUTPUT_FIELDS, "errflg", *SURFACE_DIAG_OUTPUT_FIELDS, "errflg_diag")))


def _compiled_run_key(
    fields: Mapping[str, torch.Tensor],
    cfg: SurfaceCaseConfig,
    output_names: tuple[str, ...],
    backend: str | None,
    max_fusion_size: int | None = None,
) -> tuple[object, ...]:
    tensor_signature = tuple(
        (name, tuple(value.shape), tuple(value.stride()), value.dtype, value.device,
         value.requires_grad, value.layout)
        for name, value in fields.items()
    )
    return (cfg, tuple(fields.keys()), output_names, backend, max_fusion_size,
            torch.is_grad_enabled(), torch.is_inference_mode_enabled(), tensor_signature)


def run_gfs_surface_layer_compiled(
    fields: Mapping[str, torch.Tensor],
    cfg: SurfaceCaseConfig,
    *,
    output_names: Sequence[str] | None = None,
    backend: str | None = None,
    max_fusion_size: int | None = None,
    fpvs_table: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Run the surface core through a cached fullgraph torch.compile wrapper."""

    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable")
    names = tuple(fields.keys())
    selected_outputs = _surface_output_names(fields, output_names)
    if max_fusion_size is not None and max_fusion_size < 1:
        raise ValueError("max_fusion_size must be positive")
    if fpvs_table is None:
        fpvs_table = fpvs_table_like(fields["ps"])
    if fpvs_table.device != fields["ps"].device or fpvs_table.dtype != fields["ps"].dtype or fpvs_table.shape != (7501,):
        raise ValueError("surface FPVS table must have7501 entries on the input device/dtype")
    table_signature = (tuple(fpvs_table.shape), tuple(fpvs_table.stride()), fpvs_table.dtype,
                       fpvs_table.device, fpvs_table.requires_grad)
    key = (*_compiled_run_key(fields, cfg, selected_outputs, backend, max_fusion_size), table_signature)
    compiled = _COMPILED_RUN_CACHE.get(key)
    if compiled is None:
        def kernel(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
            local_fields = dict(zip(names, args[:-1], strict=True))
            out = run_gfs_surface_layer(local_fields, cfg, fpvs_table=args[-1])
            return tuple(out[name] for name in selected_outputs)

        compile_kwargs: dict[str, object] = {"fullgraph": True, "dynamic": False}
        if max_fusion_size is not None:
            compile_kwargs["options"] = {"max_fusion_size": max_fusion_size}
        if backend is not None:
            compile_kwargs["backend"] = backend
        code_name = "surface_call_" + hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        specialized = types.FunctionType(
            kernel.__code__.replace(co_name=code_name, co_qualname=code_name),
            kernel.__globals__, code_name, kernel.__defaults__, kernel.__closure__)
        compiled = torch.compile(specialized, **compile_kwargs)
        _COMPILED_RUN_CACHE[key] = compiled
    return dict(zip(selected_outputs, compiled(*(fields[name] for name in names), fpvs_table), strict=True))


def config_as_dict(cfg: SurfaceCaseConfig) -> dict[str, object]:
    return asdict(cfg)


def run_gfs_surface_layer_efficient(fields, cfg, *, max_fusion_size: int = 128,
                                    fpvs_table: torch.Tensor | None = None):
    """Opt-in CPU-qualified scheduling; CUDA qualification is still pending.

    Reusable stability regions preserve independent-output unused gradients.
    Both policies execute the same surface equations and return the full map.
    """
    if torch.is_grad_enabled() and any(value.requires_grad for value in fields.values()):
        return run_gfs_surface_layer_regional_compiled(
            fields, cfg, max_fusion_size=max_fusion_size, stability_only=True, fpvs_table=fpvs_table)
    return run_gfs_surface_layer_compiled(fields, cfg, max_fusion_size=max_fusion_size, fpvs_table=fpvs_table)
