"""Forecast output: owned rows in global-ID order, named NetCDF axes, and no partial or nonfinite files."""
from types import SimpleNamespace

import pytest
import torch

from ufs_mpas_ccpp.forecast import output_fields, output_order, read_stream, write_stream


def _partition():
    cells = SimpleNamespace(global_ids=torch.tensor([7, 2, 5, 0, 9]), owned_count=3)  # rows 3-4 are halo
    edges = SimpleNamespace(global_ids=torch.tensor([4, 1, 3, 8]), owned_count=3)
    return SimpleNamespace(cells=cells, edges=edges)


def test_owned_rows_are_written_in_ascending_global_id():
    order = output_order(_partition(), labeled=True)
    theta = torch.tensor([[70.0], [20.0], [50.0], [0.0], [90.0]])
    u = torch.tensor([[40.0], [10.0], [30.0], [80.0]])
    out = output_fields({"theta_m": theta, "u": u}, order)
    assert out["theta_m"].flatten().tolist() == [20.0, 50.0, 70.0]
    assert out["u"].flatten().tolist() == [10.0, 30.0, 40.0]
    assert out["global_cell_id"].tolist() == [3, 6, 8] and out["global_edge_id"].tolist() == [2, 4, 5]
    unlabeled = output_fields({"theta_m": theta}, output_order(_partition(), labeled=False))
    assert set(unlabeled) == {"theta_m"}


def test_unpartitioned_output_drops_the_padding_row():
    out = output_fields({"t2m": torch.arange(5.0), "u": torch.arange(4.0)}, None, n_cells=4, n_edges=3)
    assert out["t2m"].tolist() == [0.0, 1.0, 2.0, 3.0] and out["u"].tolist() == [0.0, 1.0, 2.0]


def test_stream_round_trip_names_the_native_axes(tmp_path):
    import netCDF4

    fields = {"u": torch.rand(3, 2, dtype=torch.float64), "w": torch.rand(4, 3, dtype=torch.float64),
              "t2m": torch.rand(4, dtype=torch.float32), "totprcpb": torch.rand(4, 1, dtype=torch.float64)}
    path = tmp_path / "history.nc"
    write_stream(path, fields, attributes={"nCells": 4, "nEdges": 3, "nVertLevels": 2})
    with netCDF4.Dataset(path) as data:
        assert data["u"].dimensions == ("nEdges", "nVertLevels")
        assert data["w"].dimensions == ("nCells", "nVertLevelsP1")
        assert data["t2m"].dimensions == ("nCells",) and data["totprcpb"].dimensions == ("nCells",)
    back = read_stream(path)
    assert all(torch.equal(back[name], value.reshape(back[name].shape)) and back[name].numel() == value.numel()
               for name, value in fields.items())
    with pytest.raises(FileExistsError):
        write_stream(path, fields, attributes={"nCells": 4, "nEdges": 3, "nVertLevels": 2})


def test_nonfinite_or_misshapen_output_leaves_no_file(tmp_path):
    attributes = {"nCells": 2, "nEdges": 2, "nVertLevels": 1}
    with pytest.raises(RuntimeError, match="nonfinite"):
        write_stream(tmp_path / "a.nc", {"t2m": torch.tensor([1.0, float("nan")])}, attributes=attributes)
    with pytest.raises(ValueError, match="nCells has 3 rows"):
        write_stream(tmp_path / "b.nc", {"t2m": torch.ones(3)}, attributes=attributes)
    assert not list(tmp_path.iterdir())
