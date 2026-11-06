"""MPAS graph.info.part.N cell owners (METIS partitions)."""
import pytest
import torch

from mpas_pytorch.distributed.partition import cell_owners, read_cell_partition


def write_partition(path, owners):
    path.write_text("".join(f"{rank}\n" for rank in owners))
    return path


def test_partition_file_gives_the_owner_of_each_cell(tmp_path):
    path = write_partition(tmp_path / "x.graph.info.part.3", [0, 2, 1, 1, 0, 2])
    owners = cell_owners({"nCells": 6}, 3, path)
    assert owners.tolist() == [0, 2, 1, 1, 0, 2]
    assert owners.dtype == torch.int64


def test_partition_file_for_fewer_ranks_is_rejected(tmp_path):
    path = write_partition(tmp_path / "x.graph.info.part.2", [0, 1, 1, 0])
    with pytest.raises(ValueError, match="graph.info.part.4"):
        read_cell_partition(path, n_cells=4, world_size=4)


@pytest.mark.parametrize("owners", [[0, 1, 2], [0, 1]])
def test_partition_file_must_match_the_mesh_and_rank_count(tmp_path, owners):
    path = write_partition(tmp_path / "x.graph.info.part.2", owners)
    with pytest.raises(ValueError):
        read_cell_partition(path, n_cells=4, world_size=2)

