# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Frozen RRTMGP coefficient-table loading for PyTorch kernels.

This module handles one-time NetCDF ingestion only.  Loaded coefficient arrays
are represented as device-native torch tensors so gas-optics, cloud-optics, and
RTE kernels can consume the frozen corpus without host materialization in the
scientific hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import torch


SCHEME = "RRTMGP_SHORTWAVE_LONGWAVE_RADIATION"
_TABLE_BUNDLE_CACHE: dict[tuple[str, str, str], "RrtmgpTableBundle"] = {}


@dataclass(frozen=True)
class RrtmgpTableFile:
    """One frozen NetCDF coefficient file loaded as tensors and strings."""

    role: str
    path: Path
    sha256: str
    dimensions: Mapping[str, int]
    tensors: Mapping[str, torch.Tensor]
    strings: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class RrtmgpTableBundle:
    """Frozen gas/cloud optics coefficients on a single torch device."""

    files: Mapping[str, RrtmgpTableFile]
    device: torch.device
    runtime_kind: str = "torch_rrtmgp_frozen_table_bundle"

    @property
    def tensor_count(self) -> int:
        return sum(len(table.tensors) for table in self.files.values())

    @property
    def byte_count(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for table in self.files.values()
            for tensor in table.tensors.values()
        )


# The RRTMGP coefficient files of the GFS v17 suite: role, file name, SHA-256.
TABLE_FILES = {
    "lw_gas": ("longwave_gas_optics_coefficients", "rrtmgp-data-lw-g128-210809.nc",
               "77c39fc74e5cddf3aef4c1cf1ccc73aba5008cdb32b22a56ad068fb517ecbebd"),
    "sw_gas": ("shortwave_gas_optics_coefficients", "rrtmgp-data-sw-g112-210809.nc",
               "74f699d147cd5074eb495793a1fafd2924442d0350c06295f8e6a00b738cf932"),
    "lw_cloud": ("longwave_cloud_optics_coefficients", "rrtmgp-cloud-optics-coeffs-lw.nc",
                 "443eae499e46e871163eb379ea975cea01f65b8b45d5524bd79ee50ea5eb3773"),
    "sw_cloud": ("shortwave_cloud_optics_coefficients_reordered_for_gfs_rrtmgp",
                 "rrtmgp-cloud-optics-coeffs-reordered-sw.nc",
                 "189ae00e97f41d3c2c8de793786aa47d7d21a6bfd87adab64965231f7002b96c"),
}
# Without ``table_dir`` the files are read from the repository's reference copy.
_REFERENCE_DIRS = {"lw_gas": "data", "sw_gas": "data",
                   "lw_cloud": "extensions/cloud_optics", "sw_cloud": "extensions/cloud_optics"}


def _reference_root() -> Path:
    return Path(__file__).resolve().parents[4] / "orchestration" / "rrtmgp_radiation" / "reference_inputs" / "rrtmgp"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_char_array(array: object) -> tuple[str, ...]:
    import numpy as np

    values = np.asarray(array)
    if values.ndim == 0:
        return (str(values.item()).strip(),)
    if values.dtype.kind == "S":
        rows = values.reshape((-1, values.shape[-1])) if values.ndim > 1 else values.reshape((1, -1))
        return tuple(b"".join(row.tolist()).decode("utf-8").rstrip("\x00 ").strip() for row in rows)
    rows = values.astype(str).reshape((-1, values.shape[-1])) if values.ndim > 1 else values.astype(str).reshape((1, -1))
    return tuple("".join(row.tolist()).rstrip("\x00 ").strip() for row in rows)


def _load_table_file(
    *,
    key: str,
    role: str,
    path: Path,
    expected_sha256: str,
    device: torch.device,
) -> RrtmgpTableFile:
    from netCDF4 import Dataset
    import numpy as np

    if not path.exists():
        raise FileNotFoundError(path)
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"frozen RRTMGP table hash mismatch for {key}: expected {expected_sha256}, got {actual_sha256}"
        )

    tensors: dict[str, torch.Tensor] = {}
    strings: dict[str, tuple[str, ...]] = {}
    with Dataset(path) as dataset:
        dimensions = {name: len(dim) for name, dim in dataset.dimensions.items()}
        for name, variable in dataset.variables.items():
            array = np.asarray(variable[...])
            if array.dtype.kind in {"S", "U", "O"}:
                strings[name] = _decode_char_array(array)
            elif array.dtype.kind == "f":
                tensors[name] = torch.as_tensor(array, dtype=torch.float64, device=device)
            elif array.dtype.kind in {"i", "u", "b"}:
                tensors[name] = torch.as_tensor(array, dtype=torch.int64, device=device)
            else:
                raise TypeError(f"unsupported NetCDF variable dtype for {path.name}:{name}: {array.dtype}")

    return RrtmgpTableFile(
        role=role,
        path=path,
        sha256=actual_sha256,
        dimensions=dimensions,
        tensors=tensors,
        strings=strings,
    )


@torch.inference_mode(False)
def load_rrtmgp_tables(
    *,
    table_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> RrtmgpTableBundle:
    """Load the frozen gas/cloud optics coefficient corpus as torch tensors.

    ``table_dir`` holds the four files of ``TABLE_FILES``; each must match its SHA-256.

    The NetCDF read itself is a one-time initialization step.  The returned
    bundle contains only torch tensors, strings, dimensions, and hash metadata;
    downstream scientific kernels must operate on these tensors directly.
    Cache construction uses ordinary tensors even during an inference call,
    because subsequent differentiable calls must save the same coefficients.
    """

    torch_device = torch.device(device)
    directory = Path(table_dir).resolve() if table_dir is not None else None
    cache_key = (str(directory), str(torch_device))
    cached = _TABLE_BUNDLE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    files = {
        key: _load_table_file(
            key=key,
            role=role,
            path=(directory / name if directory is not None
                  else _reference_root() / _REFERENCE_DIRS[key] / name).resolve(),
            expected_sha256=sha256,
            device=torch_device,
        )
        for key, (role, name, sha256) in TABLE_FILES.items()
    }
    bundle = RrtmgpTableBundle(files=files, device=torch_device)
    _TABLE_BUNDLE_CACHE[cache_key] = bundle
    return bundle


__all__ = [
    "TABLE_FILES",
    "RrtmgpTableBundle",
    "RrtmgpTableFile",
    "load_rrtmgp_tables",
]
