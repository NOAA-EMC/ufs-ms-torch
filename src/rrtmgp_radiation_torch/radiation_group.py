# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Oracle-lineage RRTMGP radiation group (G3.4).

This module ports the radiation group of the custom suite
``MPAS_GFS_v17_tempo_rrtmgp`` as compiled in the TEMPO+RRTMGP oracle tree
(ccpp-physics 5ef5adc lineage plus the G3 patch, rte-rrtmgp v1.7):

``GFS_rrtmgp_setup`` -> ``GFS_rrtmgp_pre`` -> ``GFS_radiation_surface`` ->
``GFS_rrtmgp_cloud_mp`` -> ``GFS_rrtmgp_cloud_overlap`` ->
``GFS_cloud_diagnostics`` -> ``rrtmgp_aerosol_optics`` -> ``rrtmgp_sw_main`` -> ``rrtmgp_lw_main`` ->
``GFS_radiation_post``.

Layout follows the package checkpoint convention: layer fields are
``[nlay, ncol]``, interface fields ``[nlay+1, ncol]``, g-point fields
``[ngpt, nlay, ncol]``; the vertical index 0 is the surface (``top_at_1`` is
false for the MPAS host, which is the only orientation supported here).

Scope and declared inputs:

* ``GFS_radiation_surface`` uses the profile's uncoupled Noah-MP
  ``ialb=iems=2`` inputs (``frac_grid`` either way). ``GFS_cloud_diagnostics``
  ports bottom-up ``gethml`` (maximum-random or decorrelation length). ``fluxr``
  retains the source accumulation order. Solar astronomy uses a tensor NCEP date
  (or static date tuple) and the ``isol=0`` source constant or the ``isol=2``
  annual table, with all orbit arithmetic on the numerical device.
* Supported runtime configuration: ``iovr`` maximum-random (1) or
  decorrelation length (3, GFS v17), ``isubc_sw = isubc_lw = 2`` (host seeds),
  ``doGP_lwscat``, one Gauss angle, ``iaerflg = 0`` (no aerosol) or MERRA-2
  ``iaer = 1011`` (``merra2.py``), ``ico2`` 0 or 2, no MYNN, SAMF deep
  convection, ``rrtmgp_*_phys_blksz = 1``. Other values raise.
* McICA random numbers come from a native port of the GFS
  ``mersenne_twister`` generator seeded with ``icsdsw``/``icsdlw``.

The rte-rrtmgp v1.7 kernels used here (level Planck source, LW quadrature,
``tau_thresh`` series, SW ``min_k``/``min_mu0`` and the direct-beam clamps)
are written in this module; the gas-optics interpolation and the cloud
look-up tables reuse the package kernels, whose Fortran counterparts did not
change numerically between 74a0e09 and v1.7 (G3.0 census).

All tensor work is torch; branches are masks. Static configuration and offline
coefficient preparation remain host-side. Date/hour and solar scalars can stay
on the numerical device. This does not close skipped-step lifecycle ownership.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, replace
from typing import Any, Mapping


import torch
from tempo_torch import mp_tempo as tempo_mp_tempo
from tempo_torch import torch_native as tempo_native


def static_tensor(values, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """Device copy of a static host constant, built once per values, dtype and device.

    A per-call host constant costs a pageable copy and a stream sync on CUDA and
    blocks graph capture. Callers must not modify the result in place.
    """
    if torch.compiler.is_compiling():  # traced as a graph constant; no cache lookup
        return torch.tensor(values, dtype=dtype, device=device)
    return _static_tensor(_freeze(values), dtype, torch.device(device), torch.is_inference_mode_enabled())


def _freeze(values):
    return tuple(_freeze(v) for v in values) if isinstance(values, (list, tuple)) else values


def _exact(values):
    """Cache key that keeps -0.0 apart from 0.0 and matches NaN to itself."""
    if isinstance(values, tuple):
        return tuple(_exact(v) for v in values)
    return values.hex() if isinstance(values, float) else values


_STATIC_TENSORS: dict = {}


def _static_tensor(values, dtype, device, inference):
    key = (_exact(values), dtype, device, inference)
    cached = _STATIC_TENSORS.get(key)
    if cached is None:
        cached = _STATIC_TENSORS[key] = torch.tensor(values, dtype=dtype, device=device)
    return cached

from .cloud import compute_hogan_decorrelation_length
from .gas_optics import (
    AVOGADRO,
    GRAV,
    M_DRY,
    M_H2O,
    ReducedGasOpticsTable,
    _GasInterpolationPlan,
    _build_interpolation_plan,
    _build_plan_and_major,
    _interpolation_coordinates_tensor,
    _interp_planck_tensor,
    _interp3d_all_gpoints,
    _interp3d_lw_major_planck,
    _accumulate_minor_tau,
    _rayleigh_and_finalize_shortwave,
    compute_shortwave_solar_source,
    reduce_gas_optics_table,
)
from .longwave_main import compute_longwave_cloud_optics_byband
from .post import compute_heating_rate
from .pre import AMDO3, AMDW, compute_nonprognostic_gases, fpvs
from .setup import _coszmn, _solar_state
from .shortwave import _increment_shortwave_same_spectral_kernel, compute_shortwave_cloud_optics_byband
from .tables import RrtmgpTableBundle, load_rrtmgp_tables
from ufs_physics_common.compaction import run_compacted
from .diagnostics import radiation_surface, cloud_diagnostics, radiation_fluxr
from .merra2 import _aerosol_preparation, merra2_aer_nm, merra2_aod550, merra2_lw_optics, merra2_sw_optics

# --- physical constants (physcons.F90 / GFS_rrtmgp_pre / rte-rrtmgp v1.7) -------
CON_G = 9.80665
CON_RD = 287.05
CON_RV = 461.50
CON_EPS = CON_RD / CON_RV
CON_EPSM1 = CON_EPS - 1.0
CON_FVIRT = CON_RV / CON_RD - 1.0
CON_EPSQS = 1.0e-10
CON_TTP = 2.7316e2
CON_PI = 4.0 * math.atan(1.0)
EPS_PRE = 1.0e-6  # GFS_rrtmgp_pre: eps
ONEMINUS = 1.0 - EPS_PRE
FTINY = 1.0e-12
WP_EPS = 2.220446049250313e-16  # epsilon(1._wp)
OPTICS_EPS = 3.0 * 2.2250738585072014e-308  # mo_optical_props_kernels: 3*tiny (float64; see _optics_eps)


def _optics_eps(dtype: torch.dtype) -> float:
    """``3*tiny(1._wp)`` of the working dtype (OPTICS_EPS in float64; float32's tiny in float32)."""
    return 3.0 * torch.finfo(dtype).tiny
PI_RTE = math.acos(-1.0)
LW_TAU_THRESH = math.sqrt(math.sqrt(WP_EPS))
SW_MIN_K = 1.0e4 * WP_EPS
SW_MIN_MU0 = math.sqrt(WP_EPS)
_SW_LAYER_CHUNK = 8
GAUSS_DS_1 = 1.0 / 0.6096748751  # rte_lw gauss_Ds(1,1)
CZLIMT_DAY = 0.0001  # GFS_rrtmgp_pre daylight threshold

# GFS_rrtmgp_cloud_mp module parameters
RELIQ_DEF = 10.0
REICE_DEF = 50.0
# module_mp_thompson effective-radius bounds used by cmp_reff_TEMPO
RE_QC_MIN, RE_QC_MAX = 2.50e-6, 50.0e-6
RE_QI_MIN, RE_QI_MAX = 2.50e-6, 125.0e-6
RE_QS_MIN, RE_QS_MAX = 5.00e-6, 999.0e-6

# rrtmgp_lw_cloud_optics precipitation absorption
ABSRAIN = 0.33e-3
ABSSNOW0 = 1.5
# rrtmgp_sw_cloud_optics precipitation coefficients
SW_A0R = 3.07e-3
SW_A0S = 0.0
SW_A1S = 1.5
SW_B0R = (0.496, 0.466, 0.437, 0.416, 0.391, 0.374, 0.352, 0.183, 0.048, 0.012, 0.000, 0.000, 0.000, 0.000)
SW_B0S = (0.460, 0.460, 0.460, 0.460, 0.460, 0.460, 0.460, 0.460, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000)
SW_B1S = (0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.62e-5, 1.62e-5, 0.000, 0.000, 0.000, 0.000)
SW_C0R = (0.980, 0.975, 0.965, 0.960, 0.955, 0.952, 0.950, 0.944, 0.894, 0.884, 0.883, 0.883, 0.883, 0.883)
SW_C0S = (0.970, 0.970, 0.970, 0.970, 0.970, 0.970, 0.970, 0.970, 0.970, 0.970, 0.700, 0.700, 0.700, 0.700)
NIR_UVVIS_BND = (12850.0, 16000.0)
UVB_BND_LOWER = 29000.0

IMP_PHYSICS_THOMPSON = 8
IMP_PHYSICS_GFDL = 11
IMP_PHYSICS_TEMPO = 88


class RadiationConfigError(ValueError):
    """Raised for a configuration the oracle radiation group would reject or that is not ported."""


@dataclass(frozen=True)
class RadiationConfig:
    """Host scalars of the radiation group (``GFS_Control`` and constants)."""

    imp_physics: int
    ntcw: int
    ntiw: int
    ntrw: int
    ntsw: int
    ntgl: int
    ntinc: int
    ntoz: int
    ntrac: int
    nleffr: int
    nieffr: int
    nseffr: int
    lmfshal: bool
    lmfdeep2: bool
    imfdeepcnv: int
    imfdeepcnv_samf: int
    imfdeepcnv_gf: int
    do_mynnedmf: bool
    ltaerosol: bool
    lgfdlmprad: bool
    doGP_smearclds: bool
    ico2: int
    iovr: int
    iovr_convcld: int
    idcor: int
    isubc_sw: int
    isubc_lw: int
    iaerflg: int
    swhtr: bool
    lwhtr: bool
    doGP_lwscat: bool
    nGauss_angles: int
    nrghice: int
    tend_opt_lwrad: int
    tend_opt_swrad: int
    lsswr: bool
    lslwr: bool
    fhswr: float
    fhlwr: float
    dtp: float
    dtf: float
    solhr: float | torch.Tensor
    solcon: float | torch.Tensor
    jdat: tuple[int, ...] | torch.Tensor
    co2_year: int
    co2_month: int
    slag: float | None = None
    sdec: float | None = None
    cdec: float | None = None
    rrtmgp_sw_phys_blksz: int = 1
    rrtmgp_lw_phys_blksz: int = 1
    use_LW_jacobian: bool = False
    ialb: int = 2
    iems: int = 2
    lsm: int = 2
    frac_grid: bool = False
    cplice: bool = False
    use_cice_alb: bool = False
    lndp_type: int = 0
    min_seaice: float = 1.0e-11
    min_lakeice: float = 0.15
    lssav: bool = True
    kb: int = 0
    kd: int = 0
    kt: int = 0
    con_pi: float = CON_PI
    # v17 radiation options (host-prepared data; see ``ufs_mpas_ccpp.initialization``):
    # isol = 2 annual solar constant and ico2 = 2 monthly CO2 tables by model year
    # (index year - data_year0), iaermdl = 1 MERRA-2 climatology and optics, and the date at
    # ``fhour`` of the phys_time_vary aerosol interpolation.
    isol: int = 0
    iaermdl: int = 0
    data_year0: int = 0
    solc0: torch.Tensor | None = None
    co2_monthly: torch.Tensor | None = None
    co2_global: torch.Tensor | None = None
    merra2: Any = None
    merra2_optics: Any = None
    aer_jdat: torch.Tensor | None = None

    @staticmethod
    def from_mapping(values: Mapping[str, Any]) -> "RadiationConfig":
        def _int(name: str) -> int:
            return int(values[name])

        def _bool(name: str) -> bool:
            return bool(int(values[name]))

        def _float(name: str) -> float:
            return float(values[name])

        jdat = tuple(int(v) for v in values["jdat"])
        idat = tuple(int(v) for v in values.get("idat", jdat))
        result = RadiationConfig(
            imp_physics=_int("imp_physics"),
            ntcw=_int("ntcw"),
            ntiw=_int("ntiw"),
            ntrw=_int("ntrw"),
            ntsw=_int("ntsw"),
            ntgl=_int("ntgl"),
            ntinc=_int("ntinc"),
            ntoz=_int("ntoz"),
            ntrac=_int("ntrac"),
            nleffr=_int("nleffr"),
            nieffr=_int("nieffr"),
            nseffr=_int("nseffr"),
            lmfshal=_bool("lmfshal"),
            lmfdeep2=_bool("lmfdeep2"),
            imfdeepcnv=_int("imfdeepcnv"),
            imfdeepcnv_samf=_int("imfdeepcnv_samf"),
            imfdeepcnv_gf=_int("imfdeepcnv_gf"),
            do_mynnedmf=_bool("do_mynnedmf"),
            ltaerosol=_bool("ltaerosol"),
            lgfdlmprad=_bool("lgfdlmprad"),
            doGP_smearclds=_bool("doGP_smearclds"),
            ico2=_int("ico2"),
            iovr=_int("iovr"),
            iovr_convcld=_int("iovr_convcld"),
            idcor=_int("idcor"),
            isubc_sw=_int("isubc_sw"),
            isubc_lw=_int("isubc_lw"),
            iaerflg=_int("iaerflg"),
            swhtr=_bool("swhtr"),
            lwhtr=_bool("lwhtr"),
            doGP_lwscat=_bool("doGP_lwscat"),
            nGauss_angles=_int("rrtmgp_nGauss_ang"),
            nrghice=_int("rrtmgp_nrghice"),
            tend_opt_lwrad=_int("tend_opt_lwrad"),
            tend_opt_swrad=_int("tend_opt_swrad"),
            lsswr=_bool("lsswr"),
            lslwr=_bool("lslwr"),
            fhswr=_float("fhswr"),
            fhlwr=_float("fhlwr"),
            dtp=_float("dtp"),
            dtf=float(values.get("dtf", values["dtp"])),
            solhr=_float("solhr"),
            solcon=_float("solcon"),
            jdat=jdat,
            co2_year=idat[0],
            co2_month=idat[1],
            slag=None if values.get("slag") is None else float(values["slag"]),
            sdec=None if values.get("sdec") is None else float(values["sdec"]),
            cdec=None if values.get("cdec") is None else float(values["cdec"]),
            rrtmgp_sw_phys_blksz=int(values.get("rrtmgp_sw_phys_blksz", 1)),
            rrtmgp_lw_phys_blksz=int(values.get("rrtmgp_lw_phys_blksz", 1)),
            use_LW_jacobian=bool(int(values.get("use_LW_jacobian", 0))),
        )
        integers = ('ialb', 'iems', 'lsm', 'lndp_type', 'kb', 'kd', 'kt', 'isol', 'iaermdl')
        logicals = ('frac_grid', 'cplice', 'use_cice_alb', 'lssav')
        reals = ('min_seaice', 'min_lakeice')
        return replace(result, **{k: int(values[k]) for k in integers if k in values},
                       **{k: bool(int(values[k])) for k in logicals if k in values},
                       **{k: float(values[k]) for k in reals if k in values})

    def require_supported(self) -> None:
        """Fail closed outside the ported oracle configuration."""

        problems = []
        if self.iovr not in (1, 3) or self.iovr_convcld != 1:
            problems.append(f"iovr={self.iovr}, iovr_convcld={self.iovr_convcld} (only maximum-random=1 and "
                            "decorrelation-length=3 with maximum-random convective overlap are ported)")
        if self.isubc_sw != 2 or self.isubc_lw != 2:
            problems.append(f"isubc_sw={self.isubc_sw}, isubc_lw={self.isubc_lw} (only host seeds=2 are ported)")
        if self.idcor != 1:
            problems.append(f"idcor={self.idcor} (only Hogan=1 is ported)")
        if self.iaerflg != 0 and not (self.iaermdl == 1 and self.iaerflg == 11):
            problems.append(f"iaermdl={self.iaermdl}, iaerflg={self.iaerflg} (only no aerosol and MERRA-2 "
                            "iaer=1011 are ported)")
        if self.iaerflg != 0 and (self.merra2 is None or self.merra2_optics is None or self.aer_jdat is None):
            problems.append("MERRA-2 aerosol needs the prepared climatology, optics and aer_jdat")
        if self.isol not in (0, 2) or (self.isol == 2 and self.solc0 is None):
            problems.append(f"isol={self.isol} (only 0 and 2 with prepared annual values are ported)")
        if not self.doGP_lwscat or self.nGauss_angles != 1:
            problems.append("only doGP_lwscat=T with one Gauss angle is ported")
        if self.do_mynnedmf:
            problems.append("MYNN PBL clouds are not ported")
        if self.imfdeepcnv != self.imfdeepcnv_samf:
            problems.append(f"imfdeepcnv={self.imfdeepcnv} (only SAMF is ported)")
        if self.doGP_smearclds:
            problems.append("doGP_smearclds=T is not ported")
        if self.ltaerosol:
            problems.append("aerosol-aware TEMPO/Thompson effective radii are not ported")
        if not (self.lsswr and self.lslwr):
            problems.append("radiation-off steps are not ported")
        if self.rrtmgp_sw_phys_blksz != 1 or self.rrtmgp_lw_phys_blksz != 1:
            problems.append("only rrtmgp_{sw,lw}_phys_blksz = 1 is ported (block-level cloud decisions and cldtausw)")
        if self.use_LW_jacobian:
            problems.append("the LW surface-temperature Jacobian is not ported")
        if problems:
            raise RadiationConfigError("; ".join(problems))


# ---------------------------------------------------------------------------
# Mersenne Twister (GFS ``Radiation/mersenne_twister.f``)
# ---------------------------------------------------------------------------
_MT_N = 624
_MT_M = 397
_MASK32 = 0xFFFFFFFF
_MT_UPPER = 0x80000000
_MT_LOWER = 0x7FFFFFFF
_MT_MATA = 0x9908B0DF  # mata = -1727483681
_MT_TMASKB = 0x9D2C5680  # tmaskb = -1658038656
_MT_TMASKC = 0xEFC60000  # tmaskc = -272236544
_MT_DEFAULT_SEED = 4357
_MT_DENOM = 4294967295.0  # 2**32 - 1


def _mt_seed_powers() -> tuple[int, ...]:
    powers = [1]
    for _ in range(1, _MT_N):
        powers.append((powers[-1] * 69069) & _MASK32)
    return tuple(powers)


_MT_SEED_POWERS = _mt_seed_powers()


def _mulmod32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``(a*b) mod 2**32`` for operands in ``[0, 2**32)`` without int64 overflow."""

    lo = b & 0xFFFF
    hi = b >> 16
    return (a * lo + (((a * hi) & 0xFFFF) << 16)) & _MASK32


def _mt_mix(cur: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
    """``(y >> 1) ^ mag01(y & 1)`` with ``y`` the upper bit of ``cur`` and the lower bits of ``nxt``."""

    y = (cur & _MT_UPPER) | (nxt & _MT_LOWER)
    return (y >> 1) ^ ((y & 1) * _MT_MATA)


def _mt_twist(mt: torch.Tensor) -> torch.Tensor:
    """One state regeneration of ``random_number_t`` (``mti >= n``), in four ordered slices.

    ``mt(kk) = mt(kk+m) ^ mix(mt(kk), mt(kk+1))``: the mix terms of ``kk = 0 .. n-2`` read only the
    old state, so they are formed at once; the four XOR slices then follow the source order
    (``kk >= n-m`` reads new words). Integer XOR is exact, so the words equal the per-slice form.
    """

    mix = _mt_mix(mt[:, 0:623], mt[:, 1:624])
    new0 = mt[:, 397:624] ^ mix[:, 0:227]  # kk = 0 .. n-m-1, old mt(kk+m)
    new1 = new0 ^ mix[:, 227:454]  # kk = 227 .. 453, new mt(kk-227)
    new2 = new1[:, 0:169] ^ mix[:, 454:623]  # kk = 454 .. 622
    new3 = new1[:, 169:170] ^ _mt_mix(mt[:, 623:624], new0[:, 0:1])  # kk = n-1, new mt(0), new mt(m-1)
    return torch.cat((new0, new1, new2, new3), dim=1)


def _mt_temper(y: torch.Tensor) -> torch.Tensor:
    y = y ^ (y >> 11)
    y = y ^ ((y << 7) & _MT_TMASKB)
    y = y ^ ((y << 15) & _MT_TMASKC)
    return y ^ (y >> 18)


def gfs_mersenne_twister_words(seeds: torch.Tensor, count: int) -> torch.Tensor:
    """Unsigned source words in int64; keep the 69069 fill and seed-zero substitution."""
    seed = seeds.to(dtype=torch.int64).reshape(-1)
    seed = torch.where(seed == 0, torch.full_like(seed, _MT_DEFAULT_SEED), seed) & _MASK32
    powers = static_tensor(_MT_SEED_POWERS, dtype=torch.int64, device=seed.device)
    state = _mulmod32(seed.unsqueeze(1), powers.unsqueeze(0))
    # The twist is serial over blocks; tempering is word by word, so it runs once on the words read.
    blocks = []
    for _ in range((int(count) + _MT_N - 1) // _MT_N):
        state = _mt_twist(state)
        blocks.append(state)
    del state
    words = torch.cat(blocks, dim=1)[:, : int(count)]
    del blocks
    return _mt_temper(words)


def gfs_mersenne_twister_uniform(seeds: torch.Tensor, count: int) -> torch.Tensor:
    """Source FP64 uniforms: the unsigned words divided by ``2**32 - 1``."""
    return gfs_mersenne_twister_words(seeds, count).to(dtype=torch.float64) / _MT_DENOM


def mcica_randoms_maxrand(seeds: torch.Tensor, *, ngpt: int, nlay: int) -> torch.Tensor:
    """``rng3D(:,iLay,col)`` for ``iovr /= iovr_max``: one ``ngpt`` draw per layer, layer 1 first."""

    stream = gfs_mersenne_twister_uniform(seeds, ngpt * nlay)
    ncol = stream.shape[0]
    return stream.reshape(ncol, nlay, ngpt).permute(2, 1, 0).contiguous()


def _mcica_draws(seeds: torch.Tensor, *, ngpt: int, nlay: int, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float64:
        return mcica_randoms_maxrand(seeds, ngpt=ngpt, nlay=nlay)
    if dtype != torch.float32:
        raise TypeError("McICA physical thresholds must be FP32 or FP64")
    words = gfs_mersenne_twister_words(seeds, ngpt * nlay)
    return words.reshape(-1, nlay, ngpt).permute(2, 1, 0).contiguous()


def _uniform_greater(draws: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    if draws.dtype != torch.int64:
        return draws > threshold
    if threshold.dtype != torch.float32:
        raise TypeError("integer McICA predicates require FP32 thresholds")
    # floor(t*(2**32-1)) without rounding the uniforms or using device FP64.
    # Scaling by 2**32 is exact. Nonintegral q has a fractional part greater
    # than t; integral positive q needs the one-word correction. n=0 also
    # covers tiny thresholds on devices that flush subnormal input values.
    inside = (threshold > 0.0) & (threshold < 1.0)
    t = torch.where(inside, threshold, torch.zeros_like(threshold))
    q = t * 4294967296.0
    n = torch.floor(q).to(torch.int64)
    remainder = q - n.to(torch.float32)
    cutoff = n - ((n > 0) & (remainder < t)).to(torch.int64)
    # MPS comparisons flush negative subnormals. Their sign bits remain exact;
    # exclude -0 explicitly. The final comparison still excludes either NaN.
    bits = threshold.view(torch.int32)
    negative = (bits < 0) & (bits != -2147483648)
    return ((draws > cutoff) | negative) & (threshold < 1.0)


def _uniform_less_equal(draws: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    if draws.dtype != torch.int64:
        return draws <= threshold
    return ~_uniform_greater(draws, threshold) & ~torch.isnan(threshold)


def sampled_mask_maxrand(randoms: torch.Tensor, cld_frac: torch.Tensor) -> torch.Tensor:
    """``rrtmgp_sampling::sampled_mask`` without overlap parameter (maximum-random)."""

    nlay = cld_frac.shape[0]
    cloudy = cld_frac > 0.0
    layer = torch.arange(nlay, device=cld_frac.device).unsqueeze(1)
    previous = (layer[:, 0] - 1).clamp_min(0)
    reset = cloudy & ~cloudy.index_select(0, previous)
    # A cloudy run keeps its first layer's draw. Before the first run the
    # source keeps layer zero's draw, so zero also initializes the prefix.
    owner = torch.cummax(torch.where(reset, layer, torch.zeros_like(layer)), dim=0).values
    local = randoms.gather(1, owner.unsqueeze(0).expand(randoms.shape[0], -1, -1))
    return cloudy.unsqueeze(0) & _uniform_greater(local, (1.0 - cld_frac).unsqueeze(0))


def sampled_mask_dcorr(randoms: torch.Tensor, cld_frac: torch.Tensor, overlap_param: torch.Tensor) -> torch.Tensor:
    """``rrtmgp_sampling::sampled_mask`` with ``overlap_param`` and ``randoms2`` (``iovr_dcorr``).

    The mains draw ``randoms2`` by reseeding with the same seed and taking ``ngpt*nlay`` numbers
    in one call; the generator stream is the same as the per-layer draws of ``randoms``, so
    ``randoms2 == randoms``. In the source the last block overrides the second for every layer
    after the first cloudy one: layer ``l`` (first < l <= last cloudy layer) is cloudy where
    ``r(l-1) > 1 - cf(l)`` if ``r(l) <= overlap(l-1)``, else where ``r(l) > 1 - cf(l)``; the first
    cloudy layer uses its own draw; layers outside are clear.
    """

    nlay = cld_frac.shape[0]
    cloudy = cld_frac > 0.0
    layer = torch.arange(nlay, device=cld_frac.device).unsqueeze(1)
    first = torch.argmax(cloudy.to(torch.int8), dim=0).unsqueeze(0)
    last = (nlay - 1 - torch.argmax(cloudy.flip(0).to(torch.int8), dim=0)).unsqueeze(0)
    threshold = (1.0 - cld_frac).unsqueeze(0)
    own = _uniform_greater(randoms, threshold)
    previous = (layer[:, 0] - 1).clamp_min(0)
    below = _uniform_greater(randoms.index_select(1, previous), threshold)
    use_below = _uniform_less_equal(randoms, overlap_param.index_select(0, previous).unsqueeze(0))
    inner = torch.where(use_below, below, own)
    at_first = (layer == first).unsqueeze(0) & own
    between = ((layer > first) & (layer <= last)).unsqueeze(0) & inner
    return (at_first | between) & cloudy.any(0).reshape(1, 1, -1)


def sampled_mask(randoms: torch.Tensor, cld_frac: torch.Tensor, config: RadiationConfig,
                 overlap_param: torch.Tensor | None) -> torch.Tensor:
    if config.iovr == 3:
        return sampled_mask_dcorr(randoms, cld_frac, overlap_param)
    return sampled_mask_maxrand(randoms, cld_frac)


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------
def _band_to_gpt(values: torch.Tensor, gpt_to_band: torch.Tensor) -> torch.Tensor:
    return values.index_select(0, gpt_to_band.to(device=values.device, dtype=torch.int64))


def _inc_2str(tau1, ssa1, g1, tau2, ssa2, g2):
    eps = torch.full((), _optics_eps(tau1.dtype), dtype=tau1.dtype, device=tau1.device)
    return _increment_shortwave_same_spectral_kernel(tau1, ssa1, g1, tau2, ssa2, g2, eps)


def _sum_gpoints_by_band_then_bands(flux: torch.Tensor, band_lims_gpt: list[tuple[int, int]]) -> torch.Tensor:
    """``sum(bnd_flux, dim=3)`` of ``ty_fluxes_byband``: sequential g-point sums per band, then over bands."""

    limits = [(int(lo) - 1, int(hi)) for lo, hi in band_lims_gpt]
    total = None
    for lo, hi in limits:
        band = flux[lo]
        for igpt in range(lo + 1, hi):
            band = band + flux[igpt]
        total = band if total is None else total + band
    return total


def _band_sums(flux: torch.Tensor, band_lims_gpt: list[tuple[int, int]]) -> list[torch.Tensor]:
    out = []
    for lo, hi in band_lims_gpt:
        lo0 = int(lo) - 1
        band = flux[lo0]
        for igpt in range(lo0 + 1, int(hi)):
            band = band + flux[igpt]
        out.append(band)
    return out


def _tracer(gq0: torch.Tensor, index_1based: int) -> torch.Tensor:
    return gq0[index_1based - 1]


# id(bundle) -> (weak reference to the bundle, metadata). An id is reused once its object is freed, so a hit
# counts only while the reference still points at this bundle; the entry goes when the bundle is freed.
_TABLE_METADATA_CACHE: dict[int, tuple[weakref.ReferenceType[RrtmgpTableBundle], dict[str, Any]]] = {}


@torch.inference_mode(False)
def table_metadata(bundle: RrtmgpTableBundle) -> dict[str, Any]:
    """Host copies of constant coefficient-table metadata, read once per bundle.

    These are configuration constants of the frozen tables (reference
    pressure/temperature limits, LUT radius bounds, band limits), not state;
    the one-time read keeps every later call free of device-to-host transfers.
    """

    key = id(bundle)
    cached = _TABLE_METADATA_CACHE.get(key)
    if cached is not None and cached[0]() is bundle:
        return cached[1]
    lw_gas = bundle.files["lw_gas"].tensors
    lw_cloud = bundle.files["lw_cloud"].tensors
    sw_gas_table = reduce_gas_optics_table(bundle.files["sw_gas"])
    lw_gas_table = reduce_gas_optics_table(bundle.files["lw_gas"])

    def host(t: torch.Tensor) -> list:
        return t.detach().to(device="cpu").tolist()

    meta = {
        "press_min": min(host(lw_gas["press_ref"].reshape(-1))),
        "press_max": max(host(lw_gas["press_ref"].reshape(-1))),
        "temp_min": min(host(lw_gas["temp_ref"].reshape(-1))),
        "temp_max": max(host(lw_gas["temp_ref"].reshape(-1))),
        "radliq_lwr": host(lw_cloud["radliq_lwr"].reshape(-1))[0],
        "radliq_upr": host(lw_cloud["radliq_upr"].reshape(-1))[0],
        "radice_lwr": host(lw_cloud["radice_lwr"].reshape(-1))[0],
        "radice_upr": host(lw_cloud["radice_upr"].reshape(-1))[0],
        "sw_band_lower": [row[0] for row in host(bundle.files["sw_gas"].tensors["bnd_limits_wavenumber"])],
        "sw_band_lims_gpt": [tuple(int(v) for v in row) for row in host(sw_gas_table.band_lims_gpt)],
        "lw_band_lims_gpt": [tuple(int(v) for v in row) for row in host(lw_gas_table.band_lims_gpt)],
    }

    # The callback runs while the bundle is freed, before its id can go to another object.
    _TABLE_METADATA_CACHE[key] = (weakref.ref(bundle, lambda _: _TABLE_METADATA_CACHE.pop(key, None)), meta)
    return meta


# ---------------------------------------------------------------------------
# GFS_rrtmgp_pre
# ---------------------------------------------------------------------------


def run_rrtmgp_pre(
    state: Mapping[str, torch.Tensor],
    config: RadiationConfig,
    *,
    tables: RrtmgpTableBundle,
    solar: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Port ``GFS_rrtmgp_pre_run`` (MPAS bottom-up layout) plus ``coszmn``.

    ``solar`` is ``(sollag, sindec, cosdec)`` of ``config.jdat`` when the caller already has them
    (the group computes the solar state once); otherwise they are computed here.
    """

    prsi = state["prsi"]
    prsl = state["prsl"]
    gt0 = state["gt0"]
    gq0 = state["gq0"]
    nlay, ncol = prsl.shape
    meta = table_metadata(tables)
    press_min, press_max = meta["press_min"], meta["press_max"]
    temp_min, temp_max = meta["temp_min"], meta["temp_max"]

    q_lay = _tracer(gq0, 1)
    q_lay = torch.where(q_lay < 1.0e-6, torch.full_like(q_lay, 1.0e-6), q_lay)
    p_lev = prsi
    p_lay = prsl
    t_lay = gt0
    t_lay = torch.where(t_lay <= temp_min, torch.full_like(t_lay, temp_min + WP_EPS), t_lay)
    p_lay = torch.where(p_lay <= press_min, torch.full_like(p_lay, press_min + WP_EPS), p_lay)
    t_lay = torch.where(t_lay >= temp_max, torch.full_like(t_lay, temp_max - WP_EPS), t_lay)
    p_lay = torch.where(p_lay >= press_max, torch.full_like(p_lay, press_max - WP_EPS), p_lay)

    # radiation_tools::cmp_tlev, bottom-up branch (iTOA = nlay)
    tem2da = torch.log(p_lay)
    tem2db_rows = [torch.log(p_lev[ilev]) for ilev in range(nlay)]
    tem2db_rows[nlay - 1] = torch.log(torch.clamp_min(p_lev[nlay - 1], press_min))
    tem2db = torch.stack(tem2db_rows, dim=0)
    tsfc = state["tsfc"]
    interior = t_lay[:-1] + (t_lay[1:] - t_lay[:-1]) * (tem2db[1:nlay] - tem2da[:-1]) / (tem2da[1:] - tem2da[:-1])
    t_lev = torch.cat((tsfc.unsqueeze(0), interior, t_lay[-1:]), dim=0)
    t_lev = torch.where(t_lev <= temp_min, torch.full_like(t_lev, temp_min + WP_EPS), t_lev)
    t_lev = torch.where(t_lev >= temp_max, torch.full_like(t_lev, temp_max - WP_EPS), t_lev)

    es = torch.minimum(p_lay, fpvs(t_lay))
    qs_lay = torch.maximum(torch.full_like(p_lay, CON_EPSQS), CON_EPS * es / (p_lay + CON_EPSM1 * es))
    relhum = torch.clamp(torch.maximum(torch.full_like(q_lay, CON_EPSQS), q_lay) / qs_lay, min=0.0, max=1.0)
    tv_lay = t_lay * (1.0 + CON_FVIRT * q_lay)

    delta_p = torch.abs(p_lev[1:] - p_lev[:-1])
    con_rdog = CON_RD / CON_G
    log_p_lev = torch.log(p_lev)
    delta_z = con_rdog * torch.abs(log_p_lev[:-1] - log_p_lev[1:]) * tv_lay
    hgtb = [torch.zeros_like(tsfc)]
    for ilay in range(nlay):
        hgtb.append(hgtb[-1] + delta_z[ilay])
    log_p_lay = torch.log(p_lay)
    hgtc = []
    for ilay in range(nlay):
        pfac = torch.abs(log_p_lev[ilay] - log_p_lay[ilay]) / torch.abs(log_p_lev[ilay] - log_p_lev[ilay + 1])
        hgtc.append(hgtb[ilay] + pfac * (hgtb[ilay + 1] - hgtb[ilay]))
    delta_zc = torch.stack([hgtc[0] - hgtb[0]] + [hgtc[ilay] - hgtc[ilay - 1] for ilay in range(1, nlay)], dim=0)

    if config.ntoz <= 0:
        raise RadiationConfigError("climatological ozone (i_o3 <= 0) is not ported")
    o3_lay = torch.clamp_min(_tracer(gq0, config.ntoz), CON_EPSQS)
    vmr_o2, vmr_ch4, vmr_n2o, vmr_co2 = compute_nonprognostic_gases(
        p_lev,
        state["xlon"],
        state["xlat"],
        ico2=0 if config.co2_monthly is not None else config.ico2,
        top_at_1=False,
        con_pi=CON_PI,
        co2_year=config.co2_year,
        co2_month=config.co2_month,
    )
    if config.co2_monthly is not None:
        vmr_co2 = _co2_from_tables(p_lev, state["xlon"], state["xlat"], config)
    vmr_h2o = torch.where(q_lay != 1.0, (q_lay / (1.0 - q_lay)) * AMDW, torch.zeros_like(q_lay))
    vmr_o3 = torch.where(o3_lay > 0.0, o3_lay * AMDO3, torch.zeros_like(o3_lay))

    tsfg = t_lev[0]
    tsfa = t_lay[0]

    if solar is None:
        _, sollag, sindec, cosdec = _solar_state(
            config.jdat, con_pi=CON_PI, device=state["prsl"].device
        )
    else:
        sollag, sindec, cosdec = solar
    coszen, coszdg = coszmn_with_daily(
        xlon=state["xlon"],
        sinlat=state["sinlat"],
        coslat=state["coslat"],
        solhr=config.solhr,
        sollag=sollag,
        sindec=sindec,
        cosdec=cosdec,
        deltsw=config.fhswr,
        deltim=config.dtf,
    )
    day = coszen >= CZLIMT_DAY
    sfc_emiss_byband = state["semis_prev"].unsqueeze(0).expand(len(meta["lw_band_lims_gpt"]), ncol)
    return {
        "p_lay": p_lay,
        "p_lev": p_lev,
        "t_lay": t_lay,
        "t_lev": t_lev,
        "q_lay": q_lay,
        "qs_lay": qs_lay,
        "relhum": relhum,
        "tv_lay": tv_lay,
        "deltaP": delta_p,
        "deltaZ": delta_z,
        "deltaZc": delta_zc,
        "vmr_o2": vmr_o2,
        "vmr_h2o": vmr_h2o,
        "vmr_o3": vmr_o3,
        "vmr_ch4": vmr_ch4,
        "vmr_n2o": vmr_n2o,
        "vmr_co2": vmr_co2,
        "tsfg": tsfg,
        "tsfa": tsfa,
        "tsfc_radtime": tsfc,
        "coszen": coszen,
        "coszdg": coszdg,
        "day": day,
        "raddt": torch.full((), min(config.fhswr, config.fhlwr), dtype=torch.float64, device=prsl.device),
        "sfc_emiss_byband": sfc_emiss_byband,
    }


def _data_year_index(config: RadiationConfig, nyear: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Year index into the host-prepared year tables and the month of ``config.jdat`` (device)."""
    jdat = config.jdat if isinstance(config.jdat, torch.Tensor) else torch.tensor(config.jdat, device=device)
    date = jdat.to(device=device, dtype=torch.int64)
    index = date[0] - config.data_year0
    torch._assert_async((index >= 0) & (index < nyear), "radiation date outside the prepared data years")
    return index, date[1]


def _co2_from_tables(p_lev: torch.Tensor, xlon: torch.Tensor, xlat: torch.Tensor,
                     config: RadiationConfig) -> torch.Tensor:
    """``getgases`` with ``ico2 = 2`` (``ictm = 1``, so ``gco2cyc = 0``) on the prepared tables.

    The source passes ``p_lev/100.`` and compares the upper interface with 788 hPa. ``ilon`` and
    ``ilat`` are Fortran ``int`` truncations clamped to the table range ``1..IMXCO2`` (24) and
    ``1..JMXCO2`` (12). The upstream source clamps only the upper bound: at a pole where
    ``hfpi - xlat < 0`` (float32 grid latitude above pi/2) it gets ``ilat = 0`` and reads
    ``co2vmr_sav`` out of bounds. This port clamps the index, as a patched
    Fortran build does, so the pole column reads the first latitude row of its month.
    """
    monthly, glb = config.co2_monthly, config.co2_global
    year, month = _data_year_index(config, monthly.shape[0], p_lev.device)
    raddeg = 180.0 / CON_PI
    hfpi = 0.5 * CON_PI
    tmp = raddeg / 15.0
    xlon1 = torch.where(xlon < 0.0, xlon + CON_PI, xlon)
    xlat1 = hfpi - xlat
    ilon = torch.clamp(torch.trunc(xlon1 * tmp + 1).to(torch.int64), 1, 24)
    ilat = torch.clamp(torch.trunc(xlat1 * tmp + 1).to(torch.int64), 1, 12)
    flat = (month - 1) * 288 + (ilat - 1) * 24 + (ilon - 1)
    # The clamp keeps ilon/ilat in range; the check still catches a month outside 1..12.
    torch._assert_async((flat >= 0).all() & (flat < 12 * 288).all(), "CO2 table index outside the source array")
    column = monthly.index_select(0, year.reshape(1))[0].reshape(-1)[flat]
    upper = (p_lev / 100.0)[1:] >= 788.0
    return torch.where(upper, column.unsqueeze(0), (glb.index_select(0, year.reshape(1))[0] + 0.0).expand_as(column).unsqueeze(0))


def coszmn_with_daily(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """``module_radiation_astronomy::coszmn``: return ``coszen`` and ``coszdg``."""

    pid12 = CON_PI / 12.0
    ratio = deltsw / deltim
    whole = math.floor(ratio)
    nswr = max(1, whole + int(ratio - whole >= 0.5))
    nstp = max(6, nswr)
    anginc = pid12 * (deltsw / 3600.0) / float(nstp)
    solang = pid12 * (solhr - 12.0)
    rstp = 1.0 / float(nstp)
    total = torch.zeros_like(xlon)
    for it in range(1, nstp + 1):
        cns = solang + (float(it) - 0.5) * anginc + sollag
        coszn = sindec * sinlat + cosdec * coslat * torch.cos(cns + xlon)
        total = total + torch.clamp_min(coszn, 0.0)
    coszen = _coszmn(
        xlon=xlon,
        sinlat=sinlat,
        coslat=coslat,
        solhr=solhr,
        sollag=sollag,
        sindec=sindec,
        cosdec=cosdec,
        deltsw=deltsw,
        deltim=deltim,
        con_pi=CON_PI,
    )
    return coszen, total * rstp


# ---------------------------------------------------------------------------
# GFS_rrtmgp_cloud_mp (Thompson/TEMPO branch with SAMF convective clouds)
# ---------------------------------------------------------------------------
def cld_frac_xu_randall(
    p_lay_hpa: torch.Tensor,
    qs_lay: torch.Tensor,
    relhum: torch.Tensor,
    cld_mr: torch.Tensor,
    alpha: float,
    *,
    cond_cfrac_on_rh: bool,
    lam: float = 0.5,
    factor: float = 1.0,
) -> torch.Tensor:
    """``module_radiation_clouds::cld_frac_XuRandall`` (p in hPa)."""

    clwt = 1.0e-6 * (p_lay_hpa * 0.001)
    clwm = clwt * factor
    onemrh = torch.clamp_min(1.0 - relhum, 1.0e-10)
    # A real Fortran exponent uses pow, not the scalar-exponent sqrt shortcut.
    power = torch.pow(onemrh * qs_lay, torch.full_like(qs_lay, lam))
    tem1 = torch.div(alpha, torch.clamp(power, min=0.0001, max=1.0))
    tem2 = torch.clamp(tem1 * (cld_mr - clwm), min=0.0, max=50.0)
    tem3 = torch.sqrt(torch.sqrt(relhum))
    frac = torch.clamp_min(tem3 * (1.0 - torch.exp(-tem2)), 0.0)
    if cond_cfrac_on_rh:
        frac = torch.where(relhum > 0.99, torch.ones_like(frac), frac)
    return torch.where(cld_mr > clwt, frac, torch.zeros_like(frac))


def cmp_reff_tempo(
    *,
    q_lay: torch.Tensor,
    p_lay: torch.Tensor,
    t_lay: torch.Tensor,
    qc: torch.Tensor,
    qi: torch.Tensor,
    qs: torch.Tensor,
    ni: torch.Tensor,
    lsmask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``GFS_rrtmgp_cloud_mp::cmp_reff_TEMPO`` for ``ltaerosol=F`` (microns).

    Uses the TEMPO port's ``*_check_and_update`` and ``effective_radius``
    kernels; requires the CCPP host-constant profile of the TEMPO port.
    """

    tempo_mp_tempo.require_host_constants(False)
    one_minus_q = 1.0 - q_lay
    qv_mp = q_lay / one_minus_q
    rho = CON_EPS * p_lay / (CON_RD * t_lay * (qv_mp + CON_EPS))
    # TEMPO kernels take [ncol, nz]
    rho_c = rho.transpose(0, 1).contiguous()
    qc1d = (qc / one_minus_q).transpose(0, 1).contiguous()
    qi1d = (qi / one_minus_q).transpose(0, 1).contiguous()
    qs1d = (qs / one_minus_q).transpose(0, 1).contiguous()
    ni1d = (ni / one_minus_q).transpose(0, 1).contiguous()
    zeros = torch.zeros_like(rho_c)
    land = torch.round(lsmask).to(dtype=torch.int64)
    ncsave = tempo_native.get_constant_cloud_number(shape=tuple(rho_c.shape), reference=rho_c, land=land)
    cloud = tempo_native.cloud_check_and_update(
        rho=rho_c, qc1d=qc1d, nc=zeros, qcten=zeros, ncten=zeros, dt=1.0, odt=1.0, ncsave=ncsave
    )
    ice = tempo_native.ice_check_and_update(rho=rho_c, qi1d=qi1d, ni1d=ni1d, qiten=zeros, niten=zeros, dt=1.0, odt=1.0)
    snow = tempo_native.snow_check_and_update(rho=rho_c, qs1d=qs1d, qsten=zeros, dt=1.0, odt=1.0)
    re_qc, re_qi, re_qs = tempo_native.effective_radius_tensor(
        temp=t_lay.transpose(0, 1).contiguous(),
        l_qc=cloud["l_qc"],
        nc=cloud["nc"],
        ilamc=cloud["ilamc"],
        l_qi=ice["l_qi"],
        ilami=ice["ilami"],
        l_qs=snow["l_qs"],
        rs=snow["rs"],
    )
    effr_liq = torch.clamp(torch.clamp_max(re_qc, RE_QC_MAX), min=RE_QC_MIN) * 1.0e6
    effr_ice = torch.clamp(torch.clamp_max(re_qi, RE_QI_MAX), min=RE_QI_MIN) * 1.0e6
    effr_snow = torch.clamp(torch.clamp_max(re_qs, RE_QS_MAX), min=RE_QS_MIN) * 1.0e6
    return (
        effr_liq.transpose(0, 1).contiguous(),
        effr_ice.transpose(0, 1).contiguous(),
        effr_snow.transpose(0, 1).contiguous(),
    )


def run_rrtmgp_cloud_mp(
    state: Mapping[str, torch.Tensor],
    pre: Mapping[str, torch.Tensor],
    config: RadiationConfig,
    *,
    tables: RrtmgpTableBundle,
) -> dict[str, torch.Tensor]:
    """Port ``GFS_rrtmgp_cloud_mp_run`` for the Thompson/TEMPO microphysics branch.

    ``imp_physics`` other than Thompson (8), GFDL (11) or TEMPO (88) is the
    oracle's ``else`` error; GFDL and Thompson are not ported in this lineage
    (Thompson needs ``cmp_reff_Thompson``) and raise as well.
    """

    if config.imp_physics not in (IMP_PHYSICS_THOMPSON, IMP_PHYSICS_GFDL, IMP_PHYSICS_TEMPO):
        raise RadiationConfigError("ERROR: GFS_rrtmgp_cloud_mp has no cloud coupling for this MP choice")
    if config.imp_physics != IMP_PHYSICS_TEMPO:
        raise RadiationConfigError(
            f"imp_physics={config.imp_physics}: only the TEMPO branch is ported in the oracle lineage"
        )
    p_lev = pre["p_lev"]
    p_lay = pre["p_lay"]
    t_lay = pre["t_lay"]
    qs_lay = pre["qs_lay"]
    q_lay = pre["q_lay"]
    relhum = pre["relhum"]
    gq0 = state["gq0"]
    cnvw = state["cnvw"]
    nlay, ncol = p_lay.shape
    p_hpa = p_lay * 0.01
    delta_p_hpa = torch.abs(p_lev[1:] - p_lev[:-1]) * 0.01

    # cloud_mp_SAMF (alpha0 = 200); points without convective water keep their prior values
    clouds_prev = state["clouds_prev"]
    cnv_active = cnvw > 0.0
    tem0 = 1.0e5 / CON_G
    tem1 = torch.clamp(torch.clamp_max((CON_TTP - t_lay) * 0.05, 1.0), min=0.0)
    clwc = torch.clamp_min(cnvw, 0.0) * tem0 * delta_p_hpa
    cnv_iwp = clwc * tem1
    cnv_lwp = clwc - cnv_iwp
    cnv_frac = cld_frac_xu_randall(p_hpa, qs_lay, relhum, cnvw, 200.0, cond_cfrac_on_rh=False)
    cld_cnv_lwp = torch.where(cnv_active, cnv_lwp, state["cld_cnv_lwp_prev"])
    cld_cnv_iwp = torch.where(cnv_active, cnv_iwp, state["cld_cnv_iwp_prev"])
    cld_cnv_reliq = torch.where(cnv_active, torch.full_like(cnvw, RELIQ_DEF), state["cld_cnv_reliq_prev"])
    cld_cnv_reice = torch.where(cnv_active, torch.full_like(cnvw, REICE_DEF), state["cld_cnv_reice_prev"])
    cld_cnv_frac = torch.where(cnv_active, cnv_frac, state["cld_cnv_frac_prev"])

    # cmp_reff_TEMPO
    effr_liq, effr_ice, effr_snow = cmp_reff_tempo(
        q_lay=q_lay,
        p_lay=p_lay,
        t_lay=t_lay,
        qc=_tracer(gq0, config.ntcw),
        qi=_tracer(gq0, config.ntiw),
        qs=_tracer(gq0, config.ntsw),
        ni=_tracer(gq0, config.ntinc),
        lsmask=state["slmsk"],
    )

    # cloud_mp_thompson (alpha0 = 100, or 200 with lmfdeep2; cond_cfrac_onRH present)
    alpha0 = 2000.0
    if config.lmfshal:
        alpha0 = 200.0 if config.lmfdeep2 else 100.0
    qc = _tracer(gq0, config.ntcw)
    qi = _tracer(gq0, config.ntiw)
    qr = _tracer(gq0, config.ntrw)
    qsn = _tracer(gq0, config.ntsw)
    lwp = torch.clamp_min(qc * tem0 * delta_p_hpa, 0.0)
    iwp = torch.clamp_min(qi * tem0 * delta_p_hpa, 0.0)
    rwp = torch.clamp_min(qr * tem0 * delta_p_hpa, 0.0)
    swp = torch.clamp_min(qsn * tem0 * delta_p_hpa, 0.0)
    cld_mr = qc + qi + qr + qsn
    frac = cld_frac_xu_randall(p_hpa, qs_lay, relhum, cld_mr, alpha0, cond_cfrac_on_rh=True)
    top = (torch.arange(nlay, device=p_lay.device) == nlay - 1).unsqueeze(1)  # no host scalar write (graph capture)
    zero = torch.zeros_like(lwp)
    cld_lwp = torch.where(top, zero, lwp)
    cld_iwp = torch.where(top, zero, iwp)
    cld_rwp = torch.where(top, zero, rwp)
    cld_swp = torch.where(top, zero, swp)
    cld_frac = torch.where(top, zero, frac)

    lwp_ex = torch.zeros_like(state["slmsk"])
    iwp_ex = torch.zeros_like(lwp_ex)
    lwp_fc = torch.zeros_like(lwp_ex)
    iwp_fc = torch.zeros_like(lwp_ex)
    for ilay in range(nlay - 1):
        lwp_ex = lwp_ex + cld_lwp[ilay]
        iwp_ex = iwp_ex + cld_iwp[ilay] + cld_swp[ilay]
        in_range = (cld_frac[ilay] >= 0.001) & (cld_frac[ilay] < 1.0 - 1.0e-8)
        lwp_fc = torch.where(in_range, lwp_fc + cld_lwp[ilay], lwp_fc)
        iwp_fc = torch.where(in_range, iwp_fc + cld_iwp[ilay] + cld_swp[ilay], iwp_fc)
    lwp_fc = lwp_fc * 1.0e-3
    iwp_fc = iwp_fc * 1.0e-3
    lwp_ex = lwp_ex * 1.0e-3
    iwp_ex = iwp_ex * 1.0e-3

    # unconditional RRTMGP LUT clamps (rrtmgp_lw_cloud_optics radliq/radice bounds)
    meta = table_metadata(tables)
    radliq_lwr, radliq_upr = meta["radliq_lwr"], meta["radliq_upr"]
    radice_lwr, radice_upr = meta["radice_lwr"], meta["radice_upr"]

    def clamp_liq(values: torch.Tensor) -> torch.Tensor:
        values = torch.where(values < radliq_lwr, torch.full_like(values, radliq_lwr), values)
        return torch.where(values > radliq_upr, torch.full_like(values, radliq_upr), values)

    def clamp_ice(values: torch.Tensor) -> torch.Tensor:
        values = torch.where(values < radice_lwr, torch.full_like(values, radice_lwr), values)
        return torch.where(values > radice_upr, torch.full_like(values, radice_upr), values)

    cld_reliq = clamp_liq(effr_liq)
    cld_reice = clamp_ice(effr_ice)
    if config.imfdeepcnv in (config.imfdeepcnv_samf, config.imfdeepcnv_gf):
        cld_cnv_reliq = clamp_liq(cld_cnv_reliq)
        cld_cnv_reice = clamp_ice(cld_cnv_reice)

    cldfra2d = torch.zeros_like(lwp_ex)
    for ilay in range(nlay - 1):
        cldfra2d = torch.maximum(cldfra2d, cld_frac[ilay])

    phy_f3d = state["phy_f3d_prev"].clone()
    phy_f3d[config.nleffr - 1] = effr_liq
    phy_f3d[config.nieffr - 1] = effr_ice
    phy_f3d[config.nseffr - 1] = effr_snow
    clouds = torch.stack(
        (
            cld_frac,
            cld_lwp,
            cld_reliq,
            cld_iwp,
            cld_reice,
            cld_rwp,
            clouds_prev[6],  # cld_rerain is not written by the Thompson/TEMPO branch
            cld_swp,
            effr_snow,
        ),
        dim=0,
    )
    return {
        "clouds": clouds,
        "cld_frac": cld_frac,
        "cld_lwp": cld_lwp,
        "cld_reliq": cld_reliq,
        "cld_iwp": cld_iwp,
        "cld_reice": cld_reice,
        "cld_rwp": cld_rwp,
        "cld_swp": cld_swp,
        "cld_resnow": effr_snow,
        "precip_frac": cld_frac,
        "cld_cnv_lwp": cld_cnv_lwp,
        "cld_cnv_iwp": cld_cnv_iwp,
        "cld_cnv_reliq": cld_cnv_reliq,
        "cld_cnv_reice": cld_cnv_reice,
        "cld_cnv_frac": cld_cnv_frac,
        "cld_pbl_lwp": state["cld_pbl_lwp_prev"],
        "cld_pbl_iwp": state["cld_pbl_iwp_prev"],
        "cld_pbl_reliq": state["cld_pbl_reliq_prev"],
        "cld_pbl_reice": state["cld_pbl_reice_prev"],
        "lwp_ex": lwp_ex,
        "iwp_ex": iwp_ex,
        "lwp_fc": lwp_fc,
        "iwp_fc": iwp_fc,
        "cldfra2d": cldfra2d,
        "phy_f3d": phy_f3d,
    }


# ---------------------------------------------------------------------------
# GFS_rrtmgp_cloud_overlap (iovr = iovr_convcld = maximum-random, Hogan length)
# ---------------------------------------------------------------------------
def run_rrtmgp_cloud_overlap(
    state: Mapping[str, torch.Tensor],
    cloud: Mapping[str, torch.Tensor],
    config: RadiationConfig,
    *,
    pre: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """``GFS_rrtmgp_cloud_overlap_run`` (Hogan length; ``iovr`` 1 or 3, convective maximum-random).

    For ``iovr = iovr_dcorr`` (3), ``get_alpha_exper``: layer 1 (surface) 0, above it
    ``exp(-(deltaZc*0.001)/de_lgth)``; the precipitation parameter is the same array.
    """
    de_lgth = compute_hogan_decorrelation_length(state["xlat"], torch.full((), CON_PI, dtype=state["xlat"].dtype, device=state["xlat"].device))
    zeros = torch.zeros_like(cloud["cld_frac"])
    overlap = zeros
    if config.iovr == 3:
        dzlay = pre["deltaZc"] * 0.001
        alpha = torch.exp(-dzlay / de_lgth.unsqueeze(0))
        overlap = torch.cat((zeros[:1], alpha[1:]), dim=0)
    return {
        "de_lgth": de_lgth,
        "cloud_overlap_param": overlap,
        "cnv_cloud_overlap_param": zeros,
        "precip_overlap_param": overlap,
    }


# ---------------------------------------------------------------------------
# gas optics (rte-rrtmgp v1.7 col_dry and level Planck source)
# ---------------------------------------------------------------------------
def _col_dry_v17(p_lev: torch.Tensor, vmr_h2o: torch.Tensor) -> torch.Tensor:
    delta_plev = torch.abs(p_lev[:-1] - p_lev[1:])
    fact = 1.0 / (1.0 + vmr_h2o)
    m_air = (M_DRY + M_H2O * vmr_h2o) * fact
    return 10.0 * delta_plev * AVOGADRO * fact / (1000.0 * m_air * 100.0 * GRAV)


def _vmr_stack(pre: Mapping[str, torch.Tensor], gas_names: tuple[str, ...]) -> torch.Tensor:
    layers = [torch.ones_like(pre["p_lay"])]
    for gas in gas_names:
        layers.append(pre[f"vmr_{gas}"])
    return torch.stack(layers, dim=0)


def _planck_by_gpt(totplnk: torch.Tensor, temp_ref: torch.Tensor, temperature: torch.Tensor,
                   bands: torch.Tensor) -> torch.Tensor:
    """Planck function by g-point, ``[ngpt, *temperature.shape]`` and contiguous.

    The interpolation runs by band (band-minor ``[*shape, nband]``), is moved band-major, then
    gathered to g-points along the leading axis (contiguous for the consumers): a pure selection.
    """
    # unbind/stack (not movedim + contiguous) keeps the adjoint of the band-minor interpolation
    # contiguous band-minor, so its band reductions run in the same order as before.
    by_band = torch.stack(_interp_planck_tensor(totplnk, temp_ref, temperature).unbind(-1), dim=0)
    return by_band.index_select(0, bands)


@dataclass(frozen=True)
class _LongwaveGasPreparation:
    col_gas: torch.Tensor
    plan: _GasInterpolationPlan
    planck_by_band: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _longwave_gas_columns_tensor(pre, gas_names):
    vmr = _vmr_stack(pre, gas_names)
    return vmr * _col_dry_v17(pre["p_lev"], pre["vmr_h2o"]).unsqueeze(0)


def _longwave_planck_byband_tensor(totplnk, temp_ref, tlay, tlev, tsfc):
    sfc = _interp_planck_tensor(totplnk, temp_ref, tsfc)
    lay = torch.stack(_interp_planck_tensor(totplnk, temp_ref, tlay).unbind(-1), dim=0)
    lev = torch.stack(_interp_planck_tensor(totplnk, temp_ref, tlev).unbind(-1), dim=0)
    return sfc, lay, lev


def _prepare_longwave_gas_v17(table, pre, *, coordinates=None, region=None):
    """Live call-local preparation shared by spectral consumers, never a state cache."""
    call = (lambda fn: fn) if region is None else region
    col_gas = call(_longwave_gas_columns_tensor)(pre, table.gas_names)
    plan = call(_build_interpolation_plan)(table, play=pre["p_lay"], tlay=pre["t_lay"],
                                          col_gas=col_gas, coordinates=coordinates)
    planck = call(_longwave_planck_byband_tensor)(
        table.totplnk, table.temp_ref, pre["t_lay"], pre["t_lev"], pre["tsfg"])
    return _LongwaveGasPreparation(col_gas, plan, planck)


def _longwave_spectral_view(table, gpt_range):
    """Bound coefficient/record work while keeping original band and flavor numbering."""
    start, end = gpt_range
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= table.ngpt:
        raise ValueError("invalid longwave g-point range")
    records = {}
    for domain in ("lower", "upper"):
        clipped = []
        for first, last, gas, density, scaling, complement, flavor, offset in getattr(table, f"minor_records_{domain}"):
            left, right = max(first, start), min(last, end)
            if left < right:
                clipped.append((left - start, right - start, gas, density, scaling, complement,
                                flavor, offset + left - first))
        records[f"minor_records_{domain}"] = tuple(clipped)
    return replace(table, gpt_to_band=table.gpt_to_band[start:end],
                   gpoint_flavor=table.gpoint_flavor[:, start:end], **records)


def _longwave_absorption_v17(table: ReducedGasOpticsTable, pre: Mapping[str, torch.Tensor],
                             *, coordinates=None, preparation=None, gpt_range=None) -> tuple[torch.Tensor, torch.Tensor]:
    play, tlay, plev = pre["p_lay"], pre["t_lay"], pre["p_lev"]
    if preparation is None:
        vmr = _vmr_stack(pre, table.gas_names)
        col_dry = _col_dry_v17(plev, pre["vmr_h2o"])
        col_gas = vmr * col_dry.unsqueeze(0)
        del vmr
        plan, tau, pfrac, _ = _build_plan_and_major(table, play=play, tlay=tlay, col_gas=col_gas,
                                                   longwave=True, coordinates=coordinates)
    else:
        col_gas = preparation.col_gas
        plan = replace(preparation.plan, flavor=table.gpoint_flavor.to(device=play.device, dtype=torch.int64))
        tau, pfrac = _interp3d_lw_major_planck(table, plan, gpt_range=gpt_range)
    tau = _accumulate_minor_tau(
        table, plan, play=play, tlay=tlay, col_gas=col_gas, dtype=tau.dtype, device=tau.device,
        initial_tau=tau,
    )
    del plan, col_gas
    return torch.clamp_min(tau, 0.0), pfrac


def _longwave_sources_tensor(pfrac, totplnk, temp_ref, bands, tlay, tlev, tsfc, *, planck_by_band=None):
    if planck_by_band is None:
        sfc_planck = _interp_planck_tensor(totplnk, temp_ref, tsfc).index_select(1, bands).transpose(0, 1)
        lay_planck = _planck_by_gpt(totplnk, temp_ref, tlay, bands)
        lev_planck = _planck_by_gpt(totplnk, temp_ref, tlev, bands)
    else:
        sfc_planck = planck_by_band[0].index_select(1, bands).transpose(0, 1)
        lay_planck = planck_by_band[1].index_select(0, bands)
        lev_planck = planck_by_band[2].index_select(0, bands)
    sfc_source = pfrac[:, 0, :] * sfc_planck
    lay_source = pfrac * lay_planck
    nlay = tlay.shape[0]
    interior = torch.sqrt(pfrac[:, :-1, :] * pfrac[:, 1:, :]) * lev_planck[:, 1:nlay, :]
    lev_source = torch.cat(
        (pfrac[:, :1, :] * lev_planck[:, :1, :], interior, pfrac[:, nlay - 1 :, :] * lev_planck[:, nlay:, :]),
        dim=1,
    )
    del lev_planck, interior
    return {"lay_source": lay_source, "lev_source": lev_source, "sfc_source": sfc_source}


def longwave_gas_optics_v17(table: ReducedGasOpticsTable, pre: Mapping[str, torch.Tensor],
                          *, coordinates=None, region=None, preparation=None, gpt_range=None) -> dict[str, torch.Tensor]:
    call = (lambda fn: fn) if region is None else region
    if gpt_range is not None:
        if preparation is None:
            preparation = _prepare_longwave_gas_v17(table, pre, coordinates=coordinates, region=region)
        table = _longwave_spectral_view(table, gpt_range)
    # Finish pfrac before Planck sources: otherwise fusion retains two full partial sums.
    if preparation is None:
        tau, pfrac = call(_longwave_absorption_v17)(table, pre, coordinates=coordinates)
    else:
        tau, pfrac = call(_longwave_absorption_v17)(table, pre, coordinates=coordinates,
                                                  preparation=preparation, gpt_range=gpt_range)
    bands = table.gpt_to_band.to(device=tau.device, dtype=torch.int64)
    if preparation is not None:
        return {"tau": tau, **call(_longwave_sources_tensor)(
            pfrac, table.totplnk, table.temp_ref, bands, pre["t_lay"], pre["t_lev"], pre["tsfg"],
            planck_by_band=preparation.planck_by_band)}
    sources = call(_longwave_sources_tensor)(
        pfrac, table.totplnk, table.temp_ref, bands, pre["t_lay"], pre["t_lev"], pre["tsfg"])
    return {"tau": tau, **sources}


@dataclass(frozen=True)
class _ShortwaveGasPreparation:
    col_gas: torch.Tensor
    col_dry: torch.Tensor
    plan: _GasInterpolationPlan


def _shortwave_gas_columns_tensor(pre, gas_names):
    col_dry = _col_dry_v17(pre["p_lev"], pre["vmr_h2o"])
    return _vmr_stack(pre, gas_names) * col_dry.unsqueeze(0), col_dry


def _prepare_shortwave_gas_v17(table, pre, *, coordinates=None, region=None):
    call = (lambda fn: fn) if region is None else region
    col_gas, col_dry = call(_shortwave_gas_columns_tensor)(pre, table.gas_names)
    plan = call(_build_interpolation_plan)(table, play=pre["p_lay"], tlay=pre["t_lay"],
                                          col_gas=col_gas, coordinates=coordinates)
    return _ShortwaveGasPreparation(col_gas, col_dry, plan)


def shortwave_gas_optics_v17(table: ReducedGasOpticsTable, pre: Mapping[str, torch.Tensor],
                           *, coordinates=None, preparation=None) -> dict[str, torch.Tensor]:
    """SW gas and Rayleigh optics (``g`` zero).

    The zero fields here, of the McICA cloud fields and of ``tau_safe`` stay full ``zeros_like`` /
    ``ones_like`` tensors: with scalar or broadcast zeros the whole-group compiled VJP missed eager
    (cnvw gradient 83 % off on 64 columns). The training forward and its saved
    activations are exact there; the compiled backward graph differs from an eager run of the same
    graph, independent of fusion size, buffer reuse, pattern passes and thread count (Inductor
    defect, cause not isolated).
    """
    play, tlay = pre["p_lay"], pre["t_lay"]
    if preparation is None:
        col_gas, col_dry = _shortwave_gas_columns_tensor(pre, table.gas_names)
        plan, tau_abs, _, _ = _build_plan_and_major(table, play=play, tlay=tlay, col_gas=col_gas,
                                                    longwave=False, coordinates=coordinates)
    else:
        col_gas, col_dry, plan = preparation.col_gas, preparation.col_dry, preparation.plan
        tau_abs = _interp3d_all_gpoints(table.kmajor, table, plan, scale_by_col_mix=True)
    tau_abs = _accumulate_minor_tau(
        table, plan, play=play, tlay=tlay, col_gas=col_gas, dtype=tau_abs.dtype, device=tau_abs.device,
        initial_tau=tau_abs,
    )
    _, _, tau, ssa, _ = _rayleigh_and_finalize_shortwave(
        table,
        plan,
        tau_abs=tau_abs,
        moist_plus_dry_column=col_gas[1] + col_dry,
    )
    return {"tau": tau, "ssa": ssa, "g": torch.zeros_like(tau)}


# ---------------------------------------------------------------------------
# rte-rrtmgp v1.7 solvers (bottom-up layout, top_at_1 = false)
# ---------------------------------------------------------------------------
def _lw_sources(fields: dict[str, torch.Tensor], trans: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``lw_source_noscat`` from ``fields`` (``lay_source``, ``lev_source``, ``tau_loc``), which it
    pops so each is released once read for the last time."""

    tau_loc = fields.pop("tau_loc")
    big = tau_loc > LW_TAU_THRESH
    tau_safe = torch.where(big, tau_loc, torch.ones_like(tau_loc))  # keeps the unused branch finite for autograd
    fact_big = (1.0 - trans) / tau_safe - trans
    del tau_safe
    fact_small = tau_loc * (0.5 + tau_loc * (-1.0 / 3.0 + tau_loc * 1.0 / 8.0))
    del tau_loc
    fact = torch.where(big, fact_big, fact_small)
    del big, fact_big, fact_small
    one_minus = 1.0 - trans
    lay_source = fields.pop("lay_source")
    lev_source = fields.pop("lev_source")
    source_up = one_minus * lev_source[:, 1:, :] + 2.0 * fact * (lay_source - lev_source[:, 1:, :])
    source_dn = one_minus * lev_source[:, :-1, :] + 2.0 * fact * (lay_source - lev_source[:, :-1, :])
    return source_up, source_dn


def _lw_transport_optics(fields, *, level_major=False):
    tau = fields.pop("tau")
    secant = fields.pop("secant")
    ssa = fields.pop("ssa", None)
    g = fields.pop("g", None)
    sec = secant.unsqueeze(1) if secant.ndim == 2 else secant
    del secant
    rescaled = ssa is not None
    if rescaled:
        wb = ssa * (1.0 - g) * 0.5
        del g
        scale_tau = 1.0 - ssa + wb
        del ssa
        cn = 0.4 * wb / scale_tau
        del wb
        tau_loc = tau * sec * scale_tau
        del scale_tau
        trans = torch.exp(-tau_loc)
    else:
        cn = None
        tau_loc = tau * sec
        trans = torch.exp(-tau_loc)
    del tau, sec
    fields["tau_loc"] = tau_loc
    del tau_loc
    source_up, source_dn = _lw_sources(fields, trans)
    # Inductor writes these layouts directly. Eager sweeps need no normalization.
    if level_major:
        trans = trans.transpose(0, 1).contiguous()
        source_up = source_up.transpose(0, 1).contiguous()
        source_dn = source_dn.transpose(0, 1).contiguous()
        cn = None if cn is None else cn.transpose(0, 1).contiguous()
    return trans, source_up, source_dn, cn


def _lw_down_levels(trans, source, downward):
    down = [None] * len(trans)
    for k in range(len(trans) - 1, -1, -1):
        downward = trans[k] * downward + source[k]
        down[k] = downward
    return tuple(down)


def _lw_up_levels(trans, source, upward):
    up = []
    for t, s in zip(trans, source):
        upward = t * upward + s
        up.append(upward)
    return tuple(up)


def _lw_scatter_up_levels(trans, source_up, source_dn, cn, down, upward):
    up = []
    for k in range(len(trans)):
        an = 1.0 - trans[k] * trans[k]
        adj = cn[k] * (an * down[k] - trans[k] * source_dn[k] - source_up[k])
        upward = trans[k] * upward + source_up[k] + adj
        up.append(upward)
    return tuple(up)


def _lw_scatter_down_levels(trans, source_up, source_dn, cn, up, downward):
    down = [None] * len(trans)
    for k in range(len(trans) - 1, -1, -1):
        an = 1.0 - trans[k] * trans[k]
        adj = cn[k] * (an * up[k] - trans[k] * source_up[k] - source_dn[k])
        downward = trans[k] * downward + source_dn[k] + adj
        down[k] = downward
    return tuple(down)


def _lw_finish_levels(up, down, *, scale, band_lims_gpt):
    return (_sum_gpoints_by_band_then_bands(scale * torch.stack(up, dim=1), band_lims_gpt),
            _sum_gpoints_by_band_then_bands(scale * torch.stack(down, dim=1), band_lims_gpt))


def _lw_solve(fields: dict[str, torch.Tensor], *, sfc_emis: torch.Tensor, sfc_src: torch.Tensor,
              weight: float = 1.0, region=None, band_lims_gpt=None,
              transport_optics=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordered longwave sweeps with bounded live level outputs per compiled region."""
    nlay = (fields["tau"].shape[1] if transport_optics is None
            else transport_optics[0].shape[0 if region is not None else 1])
    call = (lambda fn: fn) if region is None else region
    level_axis = 1 if region is None else 0
    trans, source_up, source_dn, cn = (call(_lw_transport_optics)(fields, level_major=region is not None)
                                     if transport_optics is None else transport_optics)
    trans, source_up, source_dn = (x.unbind(level_axis) for x in (trans, source_up, source_dn))
    zero = torch.zeros_like(sfc_src)
    dn = [None] * (nlay + 1)
    dn[nlay] = zero / (PI_RTE * weight)
    for top in range(nlay, 0, -4):
        lo = max(top - 4, 0)
        dn[lo:top] = call(_lw_down_levels)(trans[lo:top], source_dn[lo:top], dn[top])
    up = [None] * (nlay + 1)
    up[0] = dn[0] * (1.0 - sfc_emis) + sfc_emis * sfc_src
    scale = PI_RTE * weight
    reduced_up, reduced_dn = [], []
    scattering = cn is not None
    if cn is None:
        finish_lo = 0
        for lo in range(0, nlay, 4):
            top = min(lo + 4, nlay)
            up[lo + 1:top + 1] = call(_lw_up_levels)(trans[lo:top], source_up[lo:top], up[lo])
            if band_lims_gpt is not None and (top - finish_lo >= 8 or top == nlay):
                uu, dd = call(_lw_finish_levels)(tuple(up[finish_lo:top]), tuple(dn[finish_lo:top]),
                    scale=scale, band_lims_gpt=band_lims_gpt)
                reduced_up.append(uu)
                reduced_dn.append(dd)
                up[finish_lo:top] = dn[finish_lo:top] = [None] * (top - finish_lo)
                finish_lo = top
                del uu, dd
    else:
        cn = cn.unbind(level_axis)
        top_down = dn[nlay]
        dn[0] = None
        for lo in range(0, nlay, 4):
            top = min(lo + 4, nlay)
            up[lo + 1:top + 1] = call(_lw_scatter_up_levels)(trans[lo:top], source_up[lo:top],
                source_dn[lo:top], cn[lo:top], tuple(dn[lo + 1:top + 1]), up[lo])
            dn[lo + 1:top + 1] = [None] * (top - lo)
        dn[nlay] = top_down
        del top_down
        finish_top = nlay
        for top in range(nlay, 0, -4):
            lo = max(top - 4, 0)
            dn[lo:top] = call(_lw_scatter_down_levels)(trans[lo:top], source_up[lo:top],
                source_dn[lo:top], cn[lo:top], tuple(up[lo:top]), dn[top])
            if band_lims_gpt is not None and (finish_top - lo >= 8 or lo == 0):
                # These levels have completed both sweeps; only the lowest down
                # value feeds the next block. Reduce before releasing spectra.
                uu, dd = call(_lw_finish_levels)(tuple(up[lo:finish_top]), tuple(dn[lo + 1:finish_top + 1]),
                    scale=scale, band_lims_gpt=band_lims_gpt)
                reduced_up.append(uu)
                reduced_dn.append(dd)
                up[lo:finish_top] = dn[lo + 1:finish_top + 1] = [None] * (finish_top - lo)
                finish_top = lo
                del uu, dd
        del cn
    del trans, source_up, source_dn
    if band_lims_gpt is not None:
        # The endpoints were excluded above because they carry the recurrence.
        if scattering:
            uu, dd = call(_lw_finish_levels)((up[nlay],), (dn[0],),
                scale=scale, band_lims_gpt=band_lims_gpt)
            return (torch.cat((*reversed(reduced_up), uu), dim=0),
                    torch.cat((dd, *reversed(reduced_dn)), dim=0))
        uu, dd = call(_lw_finish_levels)((up[nlay],), (dn[nlay],),
            scale=scale, band_lims_gpt=band_lims_gpt)
        return torch.cat((*reduced_up, uu), dim=0), torch.cat((*reduced_dn, dd), dim=0)
    stacked = torch.stack(up, dim=1)
    del up
    flux_up = scale * stacked
    stacked = torch.stack(dn, dim=1)
    del dn
    flux_dn = scale * stacked
    return flux_up, flux_dn


def lw_solver_noscat_v17(
    *,
    tau: torch.Tensor,
    lay_source: torch.Tensor,
    lev_source: torch.Tensor,
    sfc_emis: torch.Tensor,
    sfc_src: torch.Tensor,
    secant: torch.Tensor,
    weight: float = 1.0,
    ssa: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``lw_solver_noscat_oneangle`` (rescaled when ``ssa``/``g`` are given), per-g-point fluxes."""

    fields = {"tau": tau, "secant": secant, "lay_source": lay_source, "lev_source": lev_source}
    if ssa is not None:
        fields.update(ssa=ssa, g=g)
    return _lw_solve(fields, sfc_emis=sfc_emis, sfc_src=sfc_src, weight=weight)


def _sw_layer_levels(tau_s, w0, g_s, mu0_s, positive, direct):
    # MPS scalar-clamp cache keys collide for small bounds and zero in Torch 2.8.
    min_k = torch.full((), SW_MIN_K, dtype=tau_s.dtype, device=tau_s.device)
    zero = torch.zeros((), dtype=tau_s.dtype, device=tau_s.device)
    gamma1 = (8.0 - w0 * (5.0 + 3.0 * g_s)) * 0.25
    gamma2 = 3.0 * (w0 * (1.0 - g_s)) * 0.25
    k = torch.sqrt(torch.clamp_min((gamma1 - gamma2) * (gamma1 + gamma2), min_k))
    exp_minusktau = torch.exp(-tau_s * k)
    exp_minus2ktau = exp_minusktau * exp_minusktau
    rt_term = 1.0 / (k * (1.0 + exp_minus2ktau) + gamma1 * (1.0 - exp_minus2ktau))
    rdif = rt_term * gamma2 * (1.0 - exp_minus2ktau)
    tdif = rt_term * 2.0 * k * exp_minusktau
    k_mu = k * mu0_s
    one_minus_kmu2 = 1.0 - k_mu * k_mu
    denom = torch.where(torch.abs(one_minus_kmu2) >= WP_EPS, one_minus_kmu2, WP_EPS)
    rt_term = w0 * rt_term / denom
    gamma3 = (2.0 - 3.0 * mu0_s * g_s) * 0.25
    gamma4 = 1.0 - gamma3
    alpha1 = gamma1 * gamma4 + gamma2 * gamma3
    alpha2 = gamma1 * gamma3 + gamma2 * gamma4
    k_gamma3 = k * gamma3
    k_gamma4 = k * gamma4
    t_noscat = torch.exp(-tau_s / mu0_s)
    rdir = rt_term * (
        (1.0 - k_mu) * (alpha2 + k_gamma3)
        - (1.0 + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau
        - 2.0 * (k_gamma3 - alpha2 * k_mu) * exp_minusktau * t_noscat
    )
    tdir = -rt_term * (
        (1.0 + k_mu) * (alpha1 + k_gamma4) * t_noscat
        - (1.0 - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * t_noscat
        - 2.0 * (k_gamma4 + alpha1 * k_mu) * exp_minusktau
    )
    rdir = torch.clamp_min(torch.minimum(rdir, 1.0 - t_noscat), zero)
    tdir = torch.clamp_min(torch.minimum(tdir, 1.0 - t_noscat - rdir), zero)
    beam = [None] * (tau_s.shape[0] + 1)
    beam[-1] = direct
    transmission = t_noscat.unbind(0)
    for k in range(tau_s.shape[0] - 1, -1, -1):
        beam[k] = transmission[k] * beam[k + 1]
    dir_inc = torch.stack(beam[1:])
    return (rdif, tdif, torch.where(positive, rdir * dir_inc, 0.0),
            torch.where(positive, tdir * dir_inc, 0.0), tuple(beam[:-1]))


def _sw_adding_levels(rdif, tdif, src_up, src_dn, albedo, source):
    aa, ss, dd = [], [], []
    for k in range(len(rdif)):
        denom = 1.0 / (1.0 - rdif[k] * albedo)
        next_albedo = rdif[k] + tdif[k] * tdif[k] * albedo * denom
        source = src_up[k] + tdif[k] * denom * (source + albedo * src_dn[k])
        albedo = next_albedo
        aa.append(albedo)
        ss.append(source)
        dd.append(denom)
    return tuple(aa), tuple(ss), tuple(dd)


def _sw_flux_levels(rdif, tdif, src_dn, denom, albedo, source, downward, direct=None):
    dd, uu = [None] * len(rdif), [None] * len(rdif)
    for k in range(len(rdif) - 1, -1, -1):
        downward = (tdif[k] * downward + rdif[k] * source[k] + src_dn[k]) * denom[k]
        dd[k] = downward if direct is None else downward + direct[k]
        uu[k] = downward * albedo[k] + source[k]
    if direct is not None:
        return tuple(dd), tuple(uu), downward
    return tuple(dd), tuple(uu)


def _sw_solve(fields: dict[str, torch.Tensor], *, mu0: torch.Tensor, sfc_alb_dir: torch.Tensor,
              sfc_alb_dif: torch.Tensor, inc_flux: torch.Tensor, region=None, surface_direct_only=False
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``sw_solver_2stream_v17`` on ``fields`` (``tau``, ``ssa``, ``g``), which it pops: the layer
    optics and each layer's adding terms are released once used. Same arithmetic."""

    tau = fields.pop("tau")
    ssa = fields.pop("ssa")
    g = fields.pop("g")
    ngpt, nlay, ncol = tau.shape
    mu0_row = mu0.unsqueeze(0)
    mu0_s = torch.clamp_min(mu0_row, torch.full((), SW_MIN_MU0, dtype=mu0.dtype, device=mu0.device))
    rdif = [None] * nlay
    tdif = [None] * nlay
    src_up = [None] * nlay
    src_dn = [None] * nlay
    flux_dir = [None] * (nlay + 1)
    flux_dir[nlay] = inc_flux * mu0_row
    positive = mu0_row > 0.0
    call = (lambda fn: fn) if region is None else region
    # Layer optics do not depend on the direct beam: each chunk of layers is
    # evaluated at once, level-major and contiguous; only the beam recurrence loops.
    for top in range(nlay, 0, -_SW_LAYER_CHUNK):
        lo = max(top - _SW_LAYER_CHUNK, 0)
        tau_s, w0, g_s = (x[:, lo:top].transpose(0, 1).contiguous() for x in (tau, ssa, g))
        rr, tt, su, sd, beam = call(_sw_layer_levels)(tau_s, w0, g_s, mu0_s, positive, flux_dir[top])
        rdif[lo:top], tdif[lo:top] = rr.unbind(0), tt.unbind(0)
        src_up[lo:top], src_dn[lo:top], flux_dir[lo:top] = su.unbind(0), sd.unbind(0), beam
        del tau_s, w0, g_s, rr, tt, su, sd, beam
    del tau, ssa, g
    source_sfc = torch.where(positive, flux_dir[0] * sfc_alb_dir, torch.zeros_like(flux_dir[0]))
    # adding (bottom-up)
    albedo = [None] * (nlay + 1)
    src = [None] * (nlay + 1)
    denom_l = [None] * nlay
    albedo[0] = sfc_alb_dif
    src[0] = source_sfc
    del source_sfc
    for lo in range(0, nlay, 4):
        top = min(lo + 4, nlay)
        aa, ss, dd = call(_sw_adding_levels)(tuple(rdif[lo:top]), tuple(tdif[lo:top]),
            tuple(src_up[lo:top]), tuple(src_dn[lo:top]), albedo[lo], src[lo])
        albedo[lo + 1:top + 1], src[lo + 1:top + 1], denom_l[lo:top] = aa, ss, dd
        src_up[lo:top] = [None] * (top - lo)
        del aa, ss, dd
    flux_dn = [None] * (nlay + 1)
    flux_up = [None] * (nlay + 1)
    flux_dn[nlay] = torch.zeros_like(inc_flux)
    flux_up[nlay] = flux_dn[nlay] * albedo[nlay] + src[nlay]
    if surface_direct_only:
        surface_direct = flux_dir[0].unsqueeze(1)
        downward = flux_dn[nlay]
        flux_dn[nlay] = downward + flux_dir[nlay]
        flux_dir[nlay] = None
    albedo[nlay] = src[nlay] = None
    for top in range(nlay, 0, -4):
        lo = max(top - 4, 0)
        args = (tuple(rdif[lo:top]), tuple(tdif[lo:top]), tuple(src_dn[lo:top]),
                tuple(denom_l[lo:top]), tuple(albedo[lo:top]), tuple(src[lo:top]))
        if surface_direct_only:
            dd, uu, downward = call(_sw_flux_levels)(*args, downward, tuple(flux_dir[lo:top]))
            flux_dir[lo:top] = [None] * (top - lo)
        else:
            dd, uu = call(_sw_flux_levels)(*args, flux_dn[top])
        flux_dn[lo:top], flux_up[lo:top] = dd, uu
        for values in (rdif, tdif, src_dn, denom_l, albedo, src):
            values[lo:top] = [None] * (top - lo)
        del dd, uu
    up = torch.stack(flux_up, dim=1)
    del flux_up
    if surface_direct_only:
        return up, torch.stack(flux_dn, dim=1), surface_direct
    direct = torch.stack(flux_dir, dim=1)
    del flux_dir
    down = torch.stack(flux_dn, dim=1)
    del flux_dn
    down = down + direct
    return up, down, direct


def sw_solver_2stream_v17(
    *,
    tau: torch.Tensor,
    ssa: torch.Tensor,
    g: torch.Tensor,
    mu0: torch.Tensor,
    sfc_alb_dir: torch.Tensor,
    sfc_alb_dif: torch.Tensor,
    inc_flux: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``sw_solver_2stream`` with ``sw_dif_and_source`` and ``adding`` (v1.7), per g-point.

    ``mu0`` is per column (``rte_sw_mu0_bycol``); returns up, total down and direct.
    """

    return _sw_solve({"tau": tau, "ssa": ssa, "g": g}, mu0=mu0, sfc_alb_dir=sfc_alb_dir,
                     sfc_alb_dif=sfc_alb_dif, inc_flux=inc_flux)


# ---------------------------------------------------------------------------
# rrtmgp_sw_main
# ---------------------------------------------------------------------------
def _sw_precip_byband(cld_frac, cld_rwp, cld_swp, cld_resnow, *, nband: int):
    active = (cld_frac > FTINY).unsqueeze(0)
    tau_rain = cld_rwp * SW_A0R
    snow_ok = (cld_swp > 0.0) & (cld_resnow > 10.0)
    # Scalar / tensor takes reciprocal-multiply; the source divides directly.
    tau_snow = torch.where(snow_ok,
        cld_swp * 1.09087 * (SW_A0S + torch.div(SW_A1S, 1.0315 * cld_resnow)),
        torch.zeros_like(cld_swp))
    dev, dt = cld_frac.device, cld_frac.dtype
    b0r = static_tensor(SW_B0R[:nband], dtype=dt, device=dev).reshape(-1, 1, 1)
    b0s = static_tensor(SW_B0S[:nband], dtype=dt, device=dev).reshape(-1, 1, 1)
    b1s = static_tensor(SW_B1S[:nband], dtype=dt, device=dev).reshape(-1, 1, 1)
    c0r = static_tensor(SW_C0R[:nband], dtype=dt, device=dev).reshape(-1, 1, 1)
    c0s = static_tensor(SW_C0S[:nband], dtype=dt, device=dev).reshape(-1, 1, 1)
    tr = tau_rain.unsqueeze(0)
    ts = tau_snow.unsqueeze(0)
    rs = cld_resnow.unsqueeze(0)
    ssa_rain = tr * (1.0 - b0r)
    asy_rain = ssa_rain * c0r
    ssa_snow = ts * (1.0 - (b0s + b1s * 1.0315 * rs))
    asy_snow = ssa_snow * c0s
    tau_prec = torch.clamp_min(tr + ts, 1.0e-12)
    ssa_prec = torch.clamp_min(ssa_rain + ssa_snow, 1.0e-12)
    asy_prec = torch.clamp_min(asy_rain + asy_snow, 1.0e-12)
    asyw = asy_prec / torch.clamp_min(ssa_prec, 1.0e-12)
    ssaw = torch.clamp_max(ssa_prec / tau_prec, 1.0 - 0.000001)
    za1 = asyw * asyw
    za2 = ssaw * za1
    zeros = torch.zeros_like(tau_prec)
    return (
        torch.where(active, (1.0 - za2) * tau_prec, zeros),
        torch.where(active, (ssaw - za2) / (1.0 - za2), zeros),
        torch.where(active, asyw / (1.0 + asyw), zeros),
    )


def _column_cloudiness(cld_frac: torch.Tensor) -> torch.Tensor:
    zcf0 = torch.ones_like(cld_frac[0])
    for ilay in range(cld_frac.shape[0]):
        zcf0 = torch.minimum(zcf0, 1.0 - cld_frac[ilay])
    zcf0 = torch.where(zcf0 <= FTINY, torch.zeros_like(zcf0), zcf0)
    zcf0 = torch.where(zcf0 > ONEMINUS, torch.ones_like(zcf0), zcf0)
    return 1.0 - zcf0


def _cloud_optics_sum(cloud_fn, cloud, *, tables, rough):
    """Grid-scale cloud optics plus SAMF and PBL sub-grid increments, by band.

    Also returns the grid-scale optical depth by band (the ``cldtausw`` diagnostic reads it).
    """

    tau, ssa, g = cloud_fn(cloud["cld_lwp"], cloud["cld_iwp"], cloud["cld_reliq"], cloud["cld_reice"], tables=tables, ice_roughness_index=rough)
    grid_tau = tau
    for prefix in ("cld_cnv", "cld_pbl"):
        t2, s2, g2 = cloud_fn(
            cloud[f"{prefix}_lwp"],
            cloud[f"{prefix}_iwp"],
            cloud[f"{prefix}_reliq"],
            cloud[f"{prefix}_reice"],
            tables=tables,
            ice_roughness_index=rough,
        )
        tau, ssa, g = _inc_2str(tau, ssa, g, t2, s2, g2)
    return tau, ssa, g, grid_tau


_SW_SOLAR_SOURCES: dict = {}


def _sw_solar_source(table: ReducedGasOpticsTable, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The SW g-point solar source and its sequential g-point sum, built once per table and device."""

    def build():
        base_src = compute_shortwave_solar_source(table, normalize_tsi=False).to(device=device)
        total = base_src[0]
        for igpt in range(1, table.ngpt):
            total = total + base_src[igpt]
        return base_src, total

    if torch.compiler.is_compiling():  # traced as graph constants; no cache lookup
        return build()
    # The reduced table is itself cached (reduce_gas_optics_table), so its id is stable.
    key = (id(table), torch.device(device), torch.is_inference_mode_enabled())
    if key not in _SW_SOLAR_SOURCES:
        _SW_SOLAR_SOURCES[key] = (table, *build())
    return _SW_SOLAR_SOURCES[key][1:]


_SW_CLOUD_FIELDS = tuple(
    f"{prefix}_{name}" for prefix in ("cld", "cld_cnv", "cld_pbl") for name in ("lwp", "iwp", "reliq", "reice")
) + ("cld_frac", "cld_rwp", "cld_swp", "cld_resnow")
_SW_OUTPUTS = ("fluxswUP_allsky", "fluxswDOWN_allsky", "fluxswUP_clrsky", "fluxswDOWN_clrsky",
               "scmpsw_nirbm", "scmpsw_nirdf", "scmpsw_visbm", "scmpsw_visdf", "scmpsw_uvbfc", "scmpsw_uvbf0")


def _sw_band_split(tables: RrtmgpTableBundle) -> tuple[int, int]:
    """Indices of the NIR/UV-VIS split band and of the UV-B band."""
    ibd = ibd_uv = None
    for iband, lower in enumerate(table_metadata(tables)["sw_band_lower"]):
        if lower == NIR_UVVIS_BND[0]:
            ibd = iband
        elif not (lower < NIR_UVVIS_BND[0] or lower >= NIR_UVVIS_BND[1]):
            raise RadiationConfigError(f"unexpected SW band lower limit {lower}")
        if lower == UVB_BND_LOWER:
            ibd_uv = iband
    if ibd is None or ibd_uv is None:
        raise RadiationConfigError("SW band table lacks the NIR/UV-VIS or UV-B band")
    return ibd, ibd_uv


def _sw_daylit_optics(columns: dict[str, torch.Tensor], *, tables: RrtmgpTableBundle,
                      config: RadiationConfig, coordinates=None, preparation=None):
    """Gas/cloud optics and McICA, before the ordered transport sweeps."""
    table = reduce_gas_optics_table(tables.files["sw_gas"])
    gpt_to_band = table.gpt_to_band.to(device=columns["p_lay"].device, dtype=torch.int64)
    nband, ngpt = table.nband, table.ngpt
    rough = config.nrghice - 1
    col_day = columns["day"]
    nlay, ncol = columns["p_lay"].shape
    mu0 = torch.where(col_day, columns["coszen"], torch.ones_like(columns["coszen"]))
    gas = shortwave_gas_optics_v17(table, columns, coordinates=coordinates, preparation=preparation)

    zcf1 = _column_cloudiness(columns["cld_frac"])
    cloudy = zcf1 > EPS_PRE

    c_tau, c_ssa, c_g, grid_tau = _cloud_optics_sum(
        compute_shortwave_cloud_optics_byband, columns, tables=tables, rough=rough)
    grid_tau_band11 = grid_tau[10]
    del grid_tau
    p_tau, p_ssa, p_g = _sw_precip_byband(columns["cld_frac"], columns["cld_rwp"], columns["cld_swp"],
                                          columns["cld_resnow"], nband=nband)
    c_tau, c_ssa, c_g = _inc_2str(c_tau, c_ssa, c_g, p_tau, p_ssa, p_g)
    del p_tau, p_ssa, p_g

    # G-point-sized fields are released as soon as they are dead (the solve's peak is otherwise
    # set by gas, aerosol, cloud and all-sky optics that are no longer read). The aerosol increment
    # precedes the McICA cloud fields so the gas fields die first; the two share no differentiable
    # operand, so the reverse pass is unchanged (gradients bitwise, scratch/r4 rad_grad check).
    if config.iaerflg != 0:
        # rrtmgp_aerosol_optics (setaer, MERRA-2) by band, incremented into the gas optics by band
        aer = merra2_sw_optics(config.merra2_optics, columns["aer_nm"], columns["p_lev"], columns["p_lay"],
                               columns["tv_lay"], columns["relhum"])
        a_tau, a_ssa, a_g = _inc_2str(gas.pop("tau"), gas.pop("ssa"), gas.pop("g"),
                                      *(_band_to_gpt(x, gpt_to_band) for x in aer))
        del aer
    else:
        # zero aerosol increment (iaerflg = 0 leaves aersw_* zero)
        zero = torch.zeros((), dtype=gas["tau"].dtype, device=gas["tau"].device)
        a_tau, a_ssa, a_g = _inc_2str(gas.pop("tau"), gas.pop("ssa"), gas.pop("g"), zero, zero, zero)
    del gas

    mask = sampled_mask(_mcica_draws(columns["icsdsw"], ngpt=ngpt, nlay=nlay, dtype=columns["cld_frac"].dtype), columns["cld_frac"],
                        config, columns.get("cloud_overlap_param")) & cloudy.reshape(1, 1, ncol)
    s_tau = torch.where(mask, _band_to_gpt(c_tau, gpt_to_band), torch.zeros_like(mask, dtype=c_tau.dtype))
    s_ssa = torch.where(mask, _band_to_gpt(c_ssa, gpt_to_band), torch.zeros_like(mask, dtype=c_tau.dtype))
    s_g = torch.where(mask, _band_to_gpt(c_g, gpt_to_band), torch.zeros_like(mask, dtype=c_tau.dtype))
    del mask, c_tau, c_ssa, c_g

    all_tau, all_ssa, all_g = _inc_2str(a_tau, a_ssa, a_g, s_tau, s_ssa, s_g)
    del s_tau, s_ssa, s_g
    if config.swhtr:
        clear = {"tau": a_tau, "ssa": a_ssa, "g": a_g}
        optics = {"tau": all_tau, "ssa": all_ssa, "g": all_g}
    else:
        # doSWclrsky false (rrtmgp_sw_main.F90:550): the clear-sky solve runs only for clear
        # columns. One column-local solve with all-sky optics in cloudy and clear-sky optics in
        # clear columns gives both, bitwise equal to the two solves where each is used.
        sel = cloudy.reshape(1, 1, ncol)
        optics = {}
        optics["tau"] = torch.where(sel, all_tau, a_tau)
        del all_tau, a_tau
        optics["ssa"] = torch.where(sel, all_ssa, a_ssa)
        del all_ssa, a_ssa
        optics["g"] = torch.where(sel, all_g, a_g)
        del all_g, a_g
        clear = None
    return clear, optics, mu0, cloudy, grid_tau_band11


def _column_gas_coordinates(columns, table, region):
    play = columns["p_lay"]
    call = _interpolation_coordinates_tensor if region is None else region(_interpolation_coordinates_tensor)
    return call(
        play, columns["t_lay"],
        table.temp_ref.to(device=play.device, dtype=play.dtype),
        table.press_ref.to(device=play.device, dtype=play.dtype),
        table.press_ref_trop.to(device=play.device, dtype=play.dtype),
        ntemp=table.ntemp, npres=table.npres,
    )


def _sw_daylit_columns(columns: dict[str, torch.Tensor], *, toa_gpt: torch.Tensor, tables: RrtmgpTableBundle,
                       config: RadiationConfig, region=None) -> dict[str, torch.Tensor]:
    """One column driver; optional regions leave ordered transport orchestration eager."""
    call = (lambda fn: fn) if region is None else region
    # Keep pressure logarithms and temperature coordinates outside spectral fusion.
    table = reduce_gas_optics_table(tables.files["sw_gas"])
    coordinates = _column_gas_coordinates(columns, table, region)
    preparation = _prepare_shortwave_gas_v17(table, columns, coordinates=coordinates, region=region)
    clear, optics, mu0, cloudy, grid_tau = call(_sw_daylit_optics)(
        columns, tables=tables, config=config, coordinates=coordinates, preparation=preparation)
    del coordinates, preparation
    solve = {"mu0": mu0, "sfc_alb_dir": columns["alb_dir"], "sfc_alb_dif": columns["alb_dif"],
             "inc_flux": toa_gpt.unsqueeze(1).expand(toa_gpt.shape[0], mu0.shape[0]), "region": region,
             "surface_direct_only": True}
    clear_bands = None
    if clear is not None:
        clr_up, clr_dn, _ = _sw_solve(clear, **solve)
        clear_bands = call(_sw_clear_band_sums)(clr_up, clr_dn, tables=tables)
        del clr_up, clr_dn, clear
    all_up, all_dn, all_dir = _sw_solve(optics, **solve)
    return call(_sw_daylit_outputs)(columns, all_up, all_dn, all_dir, clear_bands, mu0, cloudy,
                                    grid_tau, tables=tables, config=config)


def _sw_clear_band_sums(up, down, *, tables):
    limits = table_metadata(tables)["sw_band_lims_gpt"]
    return _band_sums(up, limits), _band_sums(down, limits)


def _sw_daylit_outputs(columns, all_up, all_dn, all_dir, clear_bands, mu0, cloudy,
                       grid_tau_band11, *, tables, config):
    band_lims_gpt = table_metadata(tables)["sw_band_lims_gpt"]
    nband = len(band_lims_gpt)
    ibd, ibd_uv = _sw_band_split(tables)
    col_day = columns["day"]
    all_up_b = _band_sums(all_up, band_lims_gpt)
    del all_up
    all_dn_b = _band_sums(all_dn, band_lims_gpt)
    del all_dn
    # Only the surface direct flux is read (the component sums below).
    all_dir_b = _band_sums(all_dir[:, :1], band_lims_gpt)
    del all_dir
    if not config.swhtr:
        clr_up_b, clr_dn_b = all_up_b, all_dn_b
    else:
        clr_up_b, clr_dn_b = clear_bands

    def band_total(bands: list[torch.Tensor]) -> torch.Tensor:
        total = bands[0]
        for item in bands[1:]:
            total = total + item
        return total

    flux_clr_up = band_total(clr_up_b)
    flux_clr_dn = band_total(clr_dn_b)
    flux_all_up = torch.where(cloudy.unsqueeze(0), band_total(all_up_b), flux_clr_up)
    flux_all_dn = torch.where(cloudy.unsqueeze(0), band_total(all_dn_b), flux_clr_dn)

    def components(dir_bands, dif_bands):
        zero = torch.zeros_like(mu0)
        nirbm = nirdf = visbm = visdf = zero
        for iband in range(nband):
            fdir = dir_bands[iband]
            fdif = dif_bands[iband]
            if iband < ibd:
                nirbm = nirbm + fdir
                nirdf = nirdf + fdif
            elif iband == ibd:
                nirbm = nirbm + fdir * 0.5
                nirdf = nirdf + fdif * 0.5
                visbm = visbm + fdir * 0.5
                visdf = visdf + fdif * 0.5
            else:
                visbm = visbm + fdir
                visdf = visdf + fdif
        return nirbm, nirdf, visbm, visdf

    zero_sfc = [torch.zeros_like(mu0) for _ in range(nband)]
    clr_comp = components([b[0] for b in clr_dn_b], zero_sfc)
    all_comp = components([b[0] for b in all_dir_b], [all_dn_b[i][0] - all_dir_b[i][0] for i in range(nband)])
    uvbf0 = clr_dn_b[ibd_uv][0]
    uvbfc = torch.where(cloudy, all_dn_b[ibd_uv][0], uvbf0)
    comps = [torch.where(cloudy, a, c) for a, c in zip(all_comp, clr_comp)]

    if not config.swhtr:
        # Clear-sky outputs are written only for clear columns; cloudy columns keep their values.
        def clear_only(new: torch.Tensor, held: torch.Tensor) -> torch.Tensor:
            return torch.where(cloudy if new.ndim == 1 else cloudy.unsqueeze(0), held, new)

        flux_clr_up = clear_only(flux_clr_up, columns["fluxswUP_clrsky"])
        flux_clr_dn = clear_only(flux_clr_dn, columns["fluxswDOWN_clrsky"])
        uvbf0 = clear_only(uvbf0, columns["scmpsw_uvbf0"])
    new = (flux_all_up, flux_all_dn, flux_clr_up, flux_clr_dn, *comps, uvbfc, uvbf0)
    out = {}
    for name, value in zip(_SW_OUTPUTS, new):
        out[name] = torch.where(col_day if value.ndim == 1 else col_day.unsqueeze(0), value, columns[name])
    out["cldtausw"] = torch.where((col_day & cloudy).unsqueeze(0), grid_tau_band11, columns["cldtausw"])
    return out


def run_rrtmgp_sw_main(
    state: Mapping[str, torch.Tensor],
    pre: Mapping[str, torch.Tensor],
    cloud: Mapping[str, torch.Tensor],
    config: RadiationConfig,
    *,
    tables: RrtmgpTableBundle,
    day_fraction: float = 1.0,
    blocks=None,
    overlap: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Port ``rrtmgp_sw_main_run`` with ``rrtmgp_sw_phys_blksz = 1``.

    Night columns keep their prior interstitial values, as in Fortran where
    only ``idxday`` columns are written. As in Fortran, the column work (gas and cloud
    optics, McICA, the solves and the band sums) runs only on the daylit columns: they are
    gathered into a buffer of ``day_fraction`` of the columns (``ufs_physics_common.compaction``,
    region ``radiation.sw_day``). Every daylit column computes from its own values alone, so the
    result does not depend on the buffer; a buffer too small for the daylit columns sets the
    region's overflow bit. ``day_fraction`` 1.0 runs every column. ``blocks(fn, columns, **kw)``,
    if given, runs ``_sw_daylit_columns`` over blocks of the buffer's columns (the host's column
    chunks); every column computes alone, so blocks do not change the result.
    """

    table = reduce_gas_optics_table(tables.files["sw_gas"])
    gpt_to_band = table.gpt_to_band.to(device=pre["p_lay"].device, dtype=torch.int64)
    day = pre["day"]

    # TOA source: GFS rescales to solcon per column with a sequential g-point sum
    base_src, total = _sw_solar_source(table, day.device)
    toa_gpt = base_src * config.solcon / total

    # surface albedo by band
    sfcalb = state["sfcalb"]
    nir_dir, nir_dif, vis_dir, vis_dif = sfcalb[0], sfcalb[1], sfcalb[2], sfcalb[3]
    alb_dir_rows, alb_dif_rows = [], []
    for lower in table_metadata(tables)["sw_band_lower"]:
        if lower < NIR_UVVIS_BND[0]:
            alb_dir_rows.append(nir_dir)
            alb_dif_rows.append(nir_dif)
        elif lower == NIR_UVVIS_BND[0]:
            alb_dir_rows.append(0.5 * (nir_dir + vis_dir))
            alb_dif_rows.append(0.5 * (nir_dif + vis_dif))
        else:
            alb_dir_rows.append(vis_dir)
            alb_dif_rows.append(vis_dif)
    _sw_band_split(tables)  # rejects an unexpected band layout
    alb_dir = _band_to_gpt(torch.stack(alb_dir_rows), gpt_to_band)
    alb_dif = _band_to_gpt(torch.stack(alb_dif_rows), gpt_to_band)

    columns = {name: pre[name] for name in ("day", "coszen", "p_lay", "t_lay", "p_lev")}
    columns.update({f"vmr_{gas}": pre[f"vmr_{gas}"] for gas in table.gas_names})
    columns.update({name: cloud[name] for name in _SW_CLOUD_FIELDS})
    columns.update(icsdsw=state["icsdsw"], alb_dir=alb_dir, alb_dif=alb_dif)
    columns.update(_optional_columns(pre, config, overlap))
    columns.update({name: state[f"{name}_prev"] for name in _SW_OUTPUTS})
    # Night columns (and every column of a call without daylight) get zero first.
    columns["cldtausw"] = torch.zeros_like(state["cldtausw_prev"])
    options = {"toa_gpt": toa_gpt, "tables": tables, "config": config}

    def daylit(part: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if blocks is None:
            return _sw_daylit_columns(part, **options)
        return blocks(_sw_daylit_columns, part, **options)

    merged = run_compacted(daylit, columns, day, region="radiation.sw_day", fraction=day_fraction,
                           axes={name: value.ndim - 1 for name, value in columns.items()})
    out = {name: merged[name] for name in _SW_OUTPUTS}
    # r4 resets once before the daylit block loop. Zero-day calls retain inout.
    out["cldtausw"] = torch.where(torch.any(day), merged["cldtausw"], state["cldtausw_prev"])
    out["sfc_alb_dir_gpt"] = alb_dir
    return out


def _optional_columns(pre, config, overlap) -> dict[str, torch.Tensor]:
    """Per-column inputs of the v17 options for the mains' column work (column on the last axis)."""
    out = {}
    if config.iovr == 3:
        out["cloud_overlap_param"] = overlap["cloud_overlap_param"]
    if config.iaerflg != 0:
        out.update({name: pre[name] for name in ("aer_nm", "tv_lay", "relhum")})
    return out


# ---------------------------------------------------------------------------
# rrtmgp_lw_main
# ---------------------------------------------------------------------------
_LW_PRE_FIELDS = ("p_lay", "t_lay", "t_lev", "p_lev", "tsfg")
_LW_CLOUD_FIELDS = tuple(
    f"{prefix}_{name}" for prefix in ("cld", "cld_cnv", "cld_pbl") for name in ("lwp", "iwp", "reliq", "reice")
) + ("cld_frac", "cld_rwp", "cld_swp", "cld_resnow")
_LW_OUTPUTS = ("fluxlwUP_allsky", "fluxlwDOWN_allsky", "fluxlwUP_clrsky", "fluxlwDOWN_clrsky")
_AEROSOL_PREPARATION_FIELDS = ("aerosol_cm", "aerosol_ih1", "aerosol_ih2", "aerosol_rdrh")


def run_rrtmgp_lw_main(
    state: Mapping[str, torch.Tensor],
    pre: Mapping[str, torch.Tensor],
    cloud: Mapping[str, torch.Tensor],
    config: RadiationConfig,
    *,
    tables: RrtmgpTableBundle,
    blocks=None,
    overlap: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Port ``rrtmgp_lw_main_run`` (``doGP_lwscat``, one angle, ``rrtmgp_lw_phys_blksz = 1``).

    ``blocks(fn, columns, **kw)``, if given, runs the column work (``_lw_columns``) over blocks of
    the columns (the host's column chunks); every column computes alone, so blocks do not change
    the result.
    """

    table = reduce_gas_optics_table(tables.files["lw_gas"])
    columns = {name: pre[name] for name in _LW_PRE_FIELDS}
    columns.update({f"vmr_{gas}": pre[f"vmr_{gas}"] for gas in table.gas_names})
    columns.update({name: cloud[name] for name in _LW_CLOUD_FIELDS})
    columns.update(icsdlw=state["icsdlw"], semis=state["semis"])
    columns.update(_optional_columns(pre, config, overlap))
    if config.iaerflg != 0 and "aerosol_cm" in pre:
        columns.update({name: pre[name] for name in _AEROSOL_PREPARATION_FIELDS})
    options = {"tables": tables, "config": config}
    fluxes = _lw_columns(columns, **options) if blocks is None else blocks(_lw_columns, columns, **options)
    return {
        **fluxes,
        "fluxlwUP_radtime": fluxes["fluxlwUP_allsky"],
        "fluxlwDOWN_radtime": fluxes["fluxlwDOWN_allsky"],
        "cldtaulw": state["cldtaulw_prev"],
    }


def _lw_column_optics(columns, tau, *, tables, config):
    table = reduce_gas_optics_table(tables.files["lw_gas"])
    gpt_to_band = table.gpt_to_band.to(device=columns["p_lay"].device, dtype=torch.int64)
    nband = table.nband
    ngpt = table.ngpt
    nlay, ncol = columns["p_lay"].shape
    cloud = columns

    semis = columns["semis"]
    emis_col = torch.where((semis > EPS_PRE) & (semis <= 1.0), semis, torch.ones_like(semis))
    sfc_emis = emis_col.unsqueeze(0).expand(ngpt, ncol)

    clr_tau = tau
    if config.iaerflg != 0:
        # 1scl aerosol optical depth (MERRA-2 extinction) incremented into the clear sky by band
        aer_tau, _, _ = merra2_lw_optics(config.merra2_optics, columns["aer_nm"], columns["p_lev"],
                                         columns["p_lay"], columns["tv_lay"], columns["relhum"],
                                         preparation=(tuple(columns[name] for name in _AEROSOL_PREPARATION_FIELDS)
                                                      if "aerosol_cm" in columns else None))
        clr_tau = clr_tau + _band_to_gpt(aer_tau, gpt_to_band)
        del aer_tau
    # else iaerflg = 0: the 1scl aerosol increment adds a zero optical depth

    zcf1 = _column_cloudiness(cloud["cld_frac"])
    cloudy = zcf1 > EPS_PRE
    rough = config.nrghice - 1
    c_tau, c_ssa, c_g, _ = _cloud_optics_sum(compute_longwave_cloud_optics_byband, cloud, tables=tables, rough=rough)
    zero_band = torch.zeros_like(c_tau)
    cloudy_b = cloudy.reshape(1, 1, ncol)
    c_tau = torch.where(cloudy_b, c_tau, zero_band)
    c_ssa = torch.where(cloudy_b, c_ssa, zero_band)
    c_g = torch.where(cloudy_b, c_g, zero_band)
    active = cloud["cld_frac"] > EPS_PRE
    snow_ok = (cloud["cld_swp"] > 0.0) & (cloud["cld_resnow"] > 10.0)
    tau_rain = ABSRAIN * cloud["cld_rwp"]
    tau_snow = torch.where(snow_ok, ABSSNOW0 * 1.05756 * cloud["cld_swp"] / cloud["cld_resnow"], torch.zeros_like(tau_rain))
    p_tau = torch.where(active, tau_rain + tau_snow, torch.zeros_like(tau_rain)).unsqueeze(0).expand(nband, nlay, ncol)
    c_tau, c_ssa, c_g = _inc_2str(c_tau, c_ssa, c_g, p_tau, zero_band, zero_band)
    del p_tau, zero_band
    return clr_tau, sfc_emis, c_tau, c_ssa, c_g, cloudy


def _lw_clear_secant(clr_tau, *, tables):
    table = reduce_gas_optics_table(tables.files["lw_gas"])
    gpt_to_band = table.gpt_to_band.to(device=clr_tau.device, dtype=torch.int64)
    col_tau = clr_tau[:, 0]
    for ilay in range(1, clr_tau.shape[1]):
        col_tau = col_tau + clr_tau[:, ilay]
    trans_total = torch.exp(-col_tau)
    fit = table.optimal_angle_fit.to(device=clr_tau.device, dtype=clr_tau.dtype).index_select(1, gpt_to_band)
    return fit[0].unsqueeze(1) * trans_total + fit[1].unsqueeze(1)


def _lw_allsky_fields(columns, clr_tau, c_tau, c_ssa, c_g, cloudy, *, tables, config):
    table = reduce_gas_optics_table(tables.files["lw_gas"])
    gpt_to_band = table.gpt_to_band.to(device=clr_tau.device, dtype=torch.int64)
    ngpt, nlay, ncol = clr_tau.shape
    cloud = columns
    mask = sampled_mask(_mcica_draws(columns["icsdlw"], ngpt=ngpt, nlay=nlay, dtype=cloud["cld_frac"].dtype), cloud["cld_frac"],
                        config, columns.get("cloud_overlap_param")) & cloudy.reshape(1, 1, ncol)
    s_tau = torch.where(mask, _band_to_gpt(c_tau, gpt_to_band), torch.zeros_like(mask, dtype=c_tau.dtype))
    s_ssa = torch.where(mask, _band_to_gpt(c_ssa, gpt_to_band), torch.zeros_like(mask, dtype=c_tau.dtype))
    s_g = torch.where(mask, _band_to_gpt(c_g, gpt_to_band), torch.zeros_like(mask, dtype=c_tau.dtype))
    del mask, c_tau, c_ssa, c_g
    # clouds (2str) incremented by clear sky (1scl): increment_2stream_by_1scalar
    fields = {"tau": s_tau + clr_tau}
    del clr_tau
    fields["ssa"] = s_tau * s_ssa / torch.clamp_min(fields["tau"], _optics_eps(fields["tau"].dtype))
    del s_tau, s_ssa
    fields["g"] = s_g
    del s_g
    fields["secant"] = fields["tau"].new_full((ngpt, ncol), GAUSS_DS_1)
    return fields


def _lw_clear_transport(clr_tau, lay_source, lev_source, *, tables, level_major=False):
    secant = _lw_clear_secant(clr_tau, tables=tables)
    fields = dict(tau=clr_tau, secant=secant, lay_source=lay_source, lev_source=lev_source)
    return _lw_transport_optics(fields, level_major=level_major)


def _lw_allsky_transport(columns, clr_tau, c_tau, c_ssa, c_g, cloudy, lay_source, lev_source,
                         *, tables, config, level_major=False):
    fields = _lw_allsky_fields(columns, clr_tau, c_tau, c_ssa, c_g, cloudy, tables=tables, config=config)
    fields.update(lay_source=lay_source, lev_source=lev_source)
    return _lw_transport_optics(fields, level_major=level_major)


def _lw_columns(columns: dict[str, torch.Tensor], *, tables: RrtmgpTableBundle,
                config: RadiationConfig, region=None) -> dict[str, torch.Tensor]:
    """Column orchestration; only bounded tensor regions enter the compiler."""
    call = (lambda fn: fn) if region is None else region
    limits = table_metadata(tables)["lw_band_lims_gpt"]
    table = reduce_gas_optics_table(tables.files["lw_gas"])
    coordinates = _column_gas_coordinates(columns, table, region)
    preparation = _prepare_longwave_gas_v17(table, columns, coordinates=coordinates, region=region)
    gas = longwave_gas_optics_v17(table, columns, coordinates=coordinates, preparation=preparation, region=region)
    del coordinates, preparation
    clr_tau, sfc_emis, c_tau, c_ssa, c_g, cloudy = call(_lw_column_optics)(
        columns, gas.pop("tau"), tables=tables, config=config)
    # Clear-sky fluxes are reduced before all-sky optics become live.
    if config.lwhtr:
        transport = call(_lw_clear_transport)(clr_tau, gas["lay_source"], gas["lev_source"],
            tables=tables, level_major=region is not None)
        flux_clr_up, flux_clr_dn = _lw_solve(
            {}, sfc_emis=sfc_emis, sfc_src=gas["sfc_source"], region=region,
            band_lims_gpt=limits, transport_optics=transport)
        del transport
    transport = call(_lw_allsky_transport)(columns, clr_tau, c_tau, c_ssa, c_g, cloudy,
        gas.pop("lay_source"), gas.pop("lev_source"), tables=tables, config=config,
        level_major=region is not None)
    del clr_tau, c_tau, c_ssa, c_g, cloudy
    flux_all_up, flux_all_dn = _lw_solve({}, sfc_emis=sfc_emis, sfc_src=gas["sfc_source"],
        region=region, band_lims_gpt=limits, transport_optics=transport)
    del transport, gas
    if not config.lwhtr:
        # doLWclrsky false (rrtmgp_lw_main.F90:495): no clear-sky solve.
        flux_clr_up = torch.zeros_like(flux_all_up)
        flux_clr_dn = torch.zeros_like(flux_all_dn)
    return dict(zip(_LW_OUTPUTS, (flux_all_up, flux_all_dn, flux_clr_up, flux_clr_dn)))


# ---------------------------------------------------------------------------
# GFS_radiation_post
# ---------------------------------------------------------------------------
def run_radiation_post(
    state: Mapping[str, torch.Tensor],
    pre: Mapping[str, torch.Tensor],
    sw: Mapping[str, torch.Tensor],
    lw: Mapping[str, torch.Tensor],
    config: RadiationConfig,
) -> dict[str, torch.Tensor]:
    """Port ``GFS_radiation_post_run`` heating, flux copies and tendency step.

    The group applies ``GFS_radiation_diagnostics`` after this calculation.
    """

    p_lev = state["prsi"]
    day = pre["day"]
    # GFS_rrtmgp_pre sets iTOA = nLev for the bottom-up host; the post scheme
    # indexes the interface arrays with it, i.e. the interface below the model top.
    i_toa = p_lev.shape[0] - 2
    out: dict[str, torch.Tensor] = {}
    htrlwc = compute_heating_rate(lw["fluxlwUP_clrsky"], lw["fluxlwDOWN_clrsky"], p_lev) if config.lwhtr else state["lwhc_prev"]
    htrlw = compute_heating_rate(lw["fluxlwUP_allsky"], lw["fluxlwDOWN_allsky"], p_lev)
    out.update(
        lwhc=htrlwc,
        htrlw=htrlw,
        htrlwu=htrlw,
        topflw_upfxc=lw["fluxlwUP_allsky"][i_toa],
        topflw_upfx0=lw["fluxlwUP_clrsky"][i_toa],
        sfcflw_upfxc=lw["fluxlwUP_allsky"][0],
        sfcflw_upfx0=lw["fluxlwUP_clrsky"][0],
        sfcflw_dnfxc=lw["fluxlwDOWN_allsky"][0],
        sfcflw_dnfx0=lw["fluxlwDOWN_clrsky"][0],
        tsflw=pre["tsfa"],
        sfcdlw=lw["fluxlwDOWN_allsky"][0],
        sfculw=lw["fluxlwUP_allsky"][0],
    )
    day_row = day.unsqueeze(0)
    zeros = torch.zeros_like(htrlw)
    htrswc = (
        torch.where(day_row, compute_heating_rate(sw["fluxswUP_clrsky"], sw["fluxswDOWN_clrsky"], p_lev), zeros)
        if config.swhtr
        else state["swhc_prev"]
    )
    htrsw = torch.where(day_row, compute_heating_rate(sw["fluxswUP_allsky"], sw["fluxswDOWN_allsky"], p_lev), zeros)
    sfcalb = state["sfcalb"]
    topfsw_upfxc = sw["fluxswUP_allsky"][i_toa]
    topfsw_dnfxc = sw["fluxswDOWN_allsky"][i_toa]
    sfcfsw_upfxc = sw["fluxswUP_allsky"][0]
    sfcfsw_dnfxc = sw["fluxswDOWN_allsky"][0]
    out.update(
        swhc=htrswc,
        htrsw=htrsw,
        topfsw_upfxc=topfsw_upfxc,
        topfsw_upfx0=sw["fluxswUP_clrsky"][i_toa],
        topfsw_dnfxc=topfsw_dnfxc,
        sfcfsw_upfxc=sfcfsw_upfxc,
        sfcfsw_upfx0=sw["fluxswUP_clrsky"][0],
        sfcfsw_dnfxc=sfcfsw_dnfxc,
        sfcfsw_dnfx0=sw["fluxswDOWN_clrsky"][0],
        nirbmdi=sw["scmpsw_nirbm"],
        nirdfdi=sw["scmpsw_nirdf"],
        visbmdi=sw["scmpsw_visbm"],
        visdfdi=sw["scmpsw_visdf"],
        nirbmui=sw["scmpsw_nirbm"] * sfcalb[0],
        nirdfui=sw["scmpsw_nirdf"] * sfcalb[1],
        visbmui=sw["scmpsw_visbm"] * sfcalb[2],
        visdfui=sw["scmpsw_visdf"] * sfcalb[3],
        sfcnsw=sfcfsw_dnfxc - sfcfsw_upfxc,
        sfcdsw=sfcfsw_dnfxc,
    )
    out["total_albedo"] = torch.where(
        topfsw_dnfxc > 0.0, topfsw_upfxc / torch.where(topfsw_dnfxc > 0.0, topfsw_dnfxc, torch.ones_like(topfsw_dnfxc)), torch.zeros_like(topfsw_dnfxc)
    )

    # tendency application: ten_u/v/q are zero, ten_t is the heating rate
    gt0, gu0, gv0, gq0 = state["gt0"], state["gu0"], state["gv0"], state["gq0"]
    dtdt, dudt, dvdt, dqdt = state["dtdt_prev"], state["dudt_prev"], state["dvdt_prev"], state["dqdt_prev"]
    ten_zero = torch.zeros_like(htrlw)
    ten_q_zero = torch.zeros_like(gq0)
    delt = config.dtp
    for ten_t, option in ((htrlw, config.tend_opt_lwrad), (htrsw, config.tend_opt_swrad)):
        if option == 1:
            gt0 = gt0 + delt * ten_t
            gu0 = gu0 + delt * ten_zero
            gv0 = gv0 + delt * ten_zero
            gq0 = gq0 + delt * ten_q_zero
        elif option == 2:
            dtdt = dtdt + ten_t
            dudt = dudt + ten_zero
            dvdt = dvdt + ten_zero
            dqdt = dqdt + ten_q_zero
        elif option == 3:
            gt0 = gt0 + delt * (dtdt + ten_t)
            gu0 = gu0 + delt * (dudt + ten_zero)
            gv0 = gv0 + delt * (dvdt + ten_zero)
            gq0 = gq0 + delt * (dqdt + ten_q_zero)
            dtdt = torch.zeros_like(dtdt)
            dudt = torch.zeros_like(dudt)
            dvdt = torch.zeros_like(dvdt)
            dqdt = torch.zeros_like(dqdt)
        elif option != 4:
            raise RadiationConfigError("A tendency application control was outside of the acceptable range (1-4)")
    out.update(
        gt0=gt0, gu0=gu0, gv0=gv0, gq0=gq0, dtdt=dtdt, dudt=dudt, dvdt=dvdt, dqdt=dqdt,
        ten_t=htrsw, ten_u=ten_zero, ten_v=ten_zero, ten_q=ten_q_zero,
    )
    return out


# ---------------------------------------------------------------------------
# group driver
# ---------------------------------------------------------------------------
def _group_head(state: Mapping[str, torch.Tensor], config: RadiationConfig, bundle: RrtmgpTableBundle):
    """The group up to the mains: setup, pre, MERRA-2 aerosol, surface, cloud, overlap, diagnostics."""

    r1, slag, sdec, cdec = _solar_state(
        config.jdat, con_pi=CON_PI, device=state["prsl"].device
    )
    if config.isol == 2:  # sol_update: solc0 of the model year (annual table), solcon = solc0/r1**2
        year, _ = _data_year_index(config, config.solc0.shape[0], state["prsl"].device)
        solcon = config.solc0.index_select(0, year.reshape(1))[0] / (r1 * r1)
    else:
        solcon = torch.div(1366.0, r1 * r1)
    config = replace(config, solcon=solcon)
    pre = run_rrtmgp_pre(state, config, tables=bundle, solar=(slag, sdec, cdec))
    aerosol = {}
    if config.iaerflg != 0:
        # GFS_phys_time_vary aerinterpol (at fhour, on the timestep's Statein prsl) feeds aer_nm;
        # it is read only by this group, so it is formed here on radiation steps.
        aer_nm = merra2_aer_nm(config.merra2, state["prsl"], config.aer_jdat)
        preparation = _aerosol_preparation(config.merra2_optics, aer_nm, pre["p_lev"], pre["p_lay"],
                                            pre["tv_lay"], pre["relhum"])
        aerodp, ext550 = merra2_aod550(config.merra2_optics, aer_nm, pre["p_lev"], pre["p_lay"],
                                       pre["tv_lay"], pre["relhum"], preparation=preparation)
        aerosol = {"aer_nm": aer_nm, "aerodp": aerodp, "ext550": ext550}
        pre = {**pre, "aer_nm": aer_nm, **dict(zip(_AEROSOL_PREPARATION_FIELDS, preparation))}
    surface = radiation_surface(state, pre, config)
    state = {**state, **surface}
    cloud = run_rrtmgp_cloud_mp(state, pre, config, tables=bundle)
    overlap = run_rrtmgp_cloud_overlap(state, cloud, config, pre=pre)
    diagnostics = cloud_diagnostics(state, pre, cloud, config, overlap=overlap)
    setup = {k: v.to(dtype=state['prsl'].dtype, device=state['prsl'].device)
             for k, v in (('slag', slag), ('sdec', sdec), ('cdec', cdec), ('solcon', solcon))}
    return solcon, setup, pre, aerosol, surface, cloud, overlap, diagnostics


def _group_tail(state, pre, sw, lw, config, aerosol, diagnostics):
    """``GFS_radiation_post`` and the ``fluxr`` diagnostics."""

    post = run_radiation_post(state, pre, sw, lw, config)
    if aerosol:
        post["aerodp"], post["ext550"] = aerosol["aerodp"], aerosol["ext550"]
    post['fluxr'] = radiation_fluxr(state, pre, sw, lw, post, diagnostics, config)
    return post


def run_radiation_group(
    state: Mapping[str, torch.Tensor],
    config: RadiationConfig,
    *,
    tables: RrtmgpTableBundle | None = None,
    sw_day_fraction: float = 1.0,
    blocks=None,
    region=None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run the ported radiation group; returns the per-scheme output dictionaries.

    ``sw_day_fraction`` sizes the daylit column buffer of ``run_rrtmgp_sw_main``; ``blocks``
    runs the column work of both mains over column blocks (``run_rrtmgp_sw_main``). ``region(fn)``,
    if given, wraps the whole-mesh parts around the mains (``_group_head``, ``_group_tail``), e.g.
    as compiled regions when the host compiles the column blocks separately.
    """

    config.require_supported()
    bundle = tables if tables is not None else load_rrtmgp_tables(device=state["prsl"].device)
    head, tail = (_group_head, _group_tail) if region is None else (region(_group_head), region(_group_tail))
    solcon, setup, pre, aerosol, surface, cloud, overlap, diagnostics = head(state, config, bundle)
    config = replace(config, solcon=solcon)
    state = {**state, **surface}
    sw = run_rrtmgp_sw_main(state, pre, cloud, config, tables=bundle, day_fraction=sw_day_fraction, blocks=blocks,
                            overlap=overlap)
    lw = run_rrtmgp_lw_main(state, pre, cloud, config, tables=bundle, blocks=blocks, overlap=overlap)
    post = tail(state, pre, sw, lw, config, aerosol, diagnostics)
    pre = {name: value for name, value in pre.items() if name not in _AEROSOL_PREPARATION_FIELDS}
    return {"setup": setup, "pre": pre, "surface": surface, "cloud_mp": cloud, "cloud_overlap": overlap,
            "diagnostics": diagnostics, "aerosol": aerosol, "sw": sw, "lw": lw, "post": post}


__all__ = [
    "RadiationConfig",
    "RadiationConfigError",
    "cld_frac_xu_randall",
    "cmp_reff_tempo",
    "coszmn_with_daily",
    "gfs_mersenne_twister_uniform",
    "longwave_gas_optics_v17",
    "lw_solver_noscat_v17",
    "mcica_randoms_maxrand",
    "run_radiation_group",
    "run_radiation_post",
    "run_rrtmgp_cloud_mp",
    "run_rrtmgp_cloud_overlap",
    "run_rrtmgp_lw_main",
    "run_rrtmgp_pre",
    "run_rrtmgp_sw_main",
    "sampled_mask_maxrand",
    "shortwave_gas_optics_v17",
    "sw_solver_2stream_v17",
]
