"""Runtime-folder inputs: the packaged tables and the checks on user files."""
import hashlib

import pytest
import torch

from noahmp_land_surface_torch.tables import TABLE_PATH
from rrtmgp_radiation_torch.tables import TABLE_FILES, load_rrtmgp_tables
from ufs_mpas_ccpp import native_case


def test_rrtmgp_table_list_names_four_files_with_hashes():
    assert set(TABLE_FILES) == {"lw_gas", "sw_gas", "lw_cloud", "sw_cloud"}
    for role, name, sha256 in TABLE_FILES.values():
        assert name.endswith(".nc") and len(sha256) == 64


def test_rrtmgp_tables_must_be_present(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rrtmgp_tables(table_dir=tmp_path)


def test_rrtmgp_tables_must_match_their_hash(tmp_path):
    for _, name, _ in TABLE_FILES.values():
        (tmp_path / name).write_bytes(b"not a coefficient file")
    with pytest.raises(ValueError):
        load_rrtmgp_tables(table_dir=tmp_path)


def test_packaged_noahmp_table_gives_the_soil_parameters():
    soil = native_case.load_soil_parameters(TABLE_PATH)
    assert soil.bexp.shape == (30,)
    assert bool(torch.isfinite(soil.bexp[:19]).all()) and bool((soil.bexp[19:] == -1.0e36).all())
    land = native_case.load_land_initialization_tables(TABLE_PATH)
    assert land.laim.shape[1] == 12


def test_tempo_tables_are_required(tmp_path, monkeypatch):
    monkeypatch.delenv("TEMPO_TABLE_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="TEMPO"):
        native_case.tempo_table_dir(tmp_path)
    for name in native_case.TEMPO_TABLE_FILES:
        (tmp_path / name).touch()
    assert native_case.tempo_table_dir(tmp_path) == tmp_path


def test_ozone_file_must_be_the_supported_one(tmp_path):
    path = tmp_path / "global_o3prdlos.f77"
    path.write_bytes(b"\x00" * 64)
    assert hashlib.sha256(path.read_bytes()).hexdigest() != native_case.OZONE_SHA256
    with pytest.raises(ValueError, match="SHA256"):
        native_case.load_ozone_table(path, expected_sha256=native_case.OZONE_SHA256)


@pytest.mark.parametrize("fhswr, dt, steps", [(3600.0, 300.0, 12), (3600.0, 1200.0, 3),
                                              (3600.0, 1440.0, 3), (3600.0, 7200.0, 1)])
def test_radiation_is_called_every_whole_number_of_steps(fhswr, dt, steps):
    assert native_case.radiation_interval_steps(fhswr, dt) == steps
