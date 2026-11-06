# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Frozen Noah-MP table helpers for the Torch target."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch


PACKAGE_TABLE_VARIANT = "package_current"
GFS_V17_PROFILE_SCM_TABLE_VARIANT = "gfs_v17_profile_scm_da75531"
# The ccpp-physics Noah-MP table of the GFS v17 suite, shipped with the package.
TABLE_PATH = Path(__file__).resolve().parent / "data" / "gfs_v17" / "noahmptable.tbl"
SELECTED_GFS_V17_TABLE_PATH = Path(__file__).resolve().parent / "data" / GFS_V17_PROFILE_SCM_TABLE_VARIANT / "noahmptable.tbl"
TABLE_PATHS = {
    PACKAGE_TABLE_VARIANT: TABLE_PATH,
    GFS_V17_PROFILE_SCM_TABLE_VARIANT: SELECTED_GFS_V17_TABLE_PATH,
}


@dataclass(frozen=True)
class NoahMpTableBundle:
    table_variant: str
    isurban: int
    iswater: int
    isbarren: int
    isice: int
    z0mvt: torch.Tensor
    nroot: torch.Tensor
    nroot_int: torch.Tensor
    ch2op: torch.Tensor
    hvt: torch.Tensor
    hvb: torch.Tensor
    tmin: torch.Tensor
    laim: torch.Tensor
    saim: torch.Tensor
    smcmax: torch.Tensor
    smcref: torch.Tensor
    smcdry: torch.Tensor
    slope: torch.Tensor
    frzk: torch.Tensor


def _section(text: str, name: str) -> str:
    match = re.search(rf"&{re.escape(name)}(?P<body>.*?)\n/", text, re.S)
    if not match:
        raise ValueError(f"missing noahmptable section {name}")
    return match.group("body")


def _raw_value(body: str, name: str) -> str:
    pattern = rf"^\s*{re.escape(name)}\s*=\s*(?P<value>.*?)(?=^\s*[A-Za-z_][A-Za-z0-9_]*\s*=|^\s*/|\Z)"
    match = re.search(pattern, body, re.S | re.M)
    if not match:
        raise ValueError(f"missing noahmptable value {name}")
    value = re.sub(r"!.*", "", match.group("value"))
    return value.replace("\n", " ").strip().rstrip(",")


def _numbers(body: str, name: str) -> list[float]:
    raw = _raw_value(body, name).replace("d", "e").replace("D", "e")
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _integer(body: str, name: str) -> int:
    return int(float(_raw_value(body, name)))


def resolve_noahmp_table_path(table_variant: str) -> Path:
    try:
        path = TABLE_PATHS[table_variant]
    except KeyError as exc:
        raise ValueError(f"unsupported Noah-MP table variant {table_variant!r}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Noah-MP table variant {table_variant!r} is missing: {path}")
    return path


# Static parsed file; traced compiles treat the result as a constant.
@lru_cache(maxsize=None)
@torch.compiler.assume_constant_result
def _parsed_table(table_variant: str) -> dict[str, object]:
    text = resolve_noahmp_table_path(table_variant).read_text(encoding="utf-8")
    veg = _section(text, "noahmp_modis_parameters")
    soil = _section(text, "noahmp_soil_stas_parameters")
    general = _section(text, "noahmp_general_parameters")
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    laim_by_month = [_numbers(veg, f"lai_{month}") for month in months]
    saim_by_month = [_numbers(veg, f"sai_{month}") for month in months]
    return {
        "isurban": _integer(veg, "isurban"),
        "iswater": _integer(veg, "iswater"),
        "isbarren": _integer(veg, "isbarren"),
        "isice": _integer(veg, "isice"),
        "z0mvt": _numbers(veg, "z0mvt"),
        "nroot": _numbers(veg, "nroot"),
        "ch2op": _numbers(veg, "ch2op"),
        "hvt": _numbers(veg, "hvt"),
        "hvb": _numbers(veg, "hvb"),
        "tmin": _numbers(veg, "tmin"),
        "laim": [list(row) for row in zip(*laim_by_month, strict=True)],
        "saim": [list(row) for row in zip(*saim_by_month, strict=True)],
        "smcmax": _numbers(soil, "maxsmc"),
        "smcref": _numbers(soil, "refsmc"),
        "smcdry": _numbers(soil, "drysmc"),
        "slope": _numbers(general, "slope_data"),
        "frzk": _numbers(general, "frzk_data")[0],
    }


def _canonical_device(device: torch.device | str) -> tuple[str, int | None]:
    dev = torch.device(device)
    index = dev.index
    if dev.type == "cuda" and index is None and torch.cuda.is_available():
        index = int(torch.cuda.current_device())
    return dev.type, index


def _device_from_key(device_type: str, device_index: int | None) -> torch.device:
    if device_index is None:
        return torch.device(device_type)
    return torch.device(f"{device_type}:{device_index}")


@lru_cache(maxsize=None)
def _cached_noahmp_table_bundle(
    device_type: str,
    device_index: int | None,
    table_variant: str,
    dtype: torch.dtype = torch.float64,
) -> NoahMpTableBundle:
    device = _device_from_key(device_type, device_index)
    parsed = _parsed_table(table_variant)
    return NoahMpTableBundle(
        table_variant=table_variant,
        isurban=int(parsed["isurban"]),
        iswater=int(parsed["iswater"]),
        isbarren=int(parsed["isbarren"]),
        isice=int(parsed["isice"]),
        z0mvt=torch.as_tensor(parsed["z0mvt"], device=device, dtype=dtype),
        nroot=torch.as_tensor(parsed["nroot"], device=device, dtype=dtype),
        nroot_int=torch.as_tensor(parsed["nroot"], device=device, dtype=torch.int64),
        ch2op=torch.as_tensor(parsed["ch2op"], device=device, dtype=dtype),
        hvt=torch.as_tensor(parsed["hvt"], device=device, dtype=dtype),
        hvb=torch.as_tensor(parsed["hvb"], device=device, dtype=dtype),
        tmin=torch.as_tensor(parsed["tmin"], device=device, dtype=dtype),
        laim=torch.as_tensor(parsed["laim"], device=device, dtype=dtype),
        saim=torch.as_tensor(parsed["saim"], device=device, dtype=dtype),
        smcmax=torch.as_tensor(parsed["smcmax"], device=device, dtype=dtype),
        smcref=torch.as_tensor(parsed["smcref"], device=device, dtype=dtype),
        smcdry=torch.as_tensor(parsed["smcdry"], device=device, dtype=dtype),
        slope=torch.as_tensor(parsed["slope"], device=device, dtype=dtype),
        frzk=torch.tensor(float(parsed["frzk"]), device=device, dtype=dtype),
    )


def noahmp_table_bundle(
    *,
    device: torch.device,
    table_variant: str = PACKAGE_TABLE_VARIANT,
    dtype: torch.dtype = torch.float64,
) -> NoahMpTableBundle:
    """Return frozen Noah-MP driver tables cached per device and working dtype.

    The tables are read-only in the target path.  Caching avoids rebuilding and
    re-copying many tiny constants around each CUDA forward/profile iteration.
    """

    return _cached_noahmp_table_bundle(*_canonical_device(device), table_variant, dtype)
