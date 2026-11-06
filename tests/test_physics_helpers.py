"""Small known answers from the physics packages."""
import torch

from gfs_surface_layer_torch.driver import fpvs, fpvs_table_like
from ufs_physics_common.ordered_sum import masked_sum


def test_saturation_vapor_pressure_at_the_triple_point():
    t = torch.tensor([273.16], dtype=torch.float64)
    torch.testing.assert_close(fpvs(t), torch.tensor([610.78], dtype=torch.float64), rtol=1e-12, atol=0.0)


def test_saturation_vapor_pressure_increases_with_temperature():
    t = torch.linspace(181.0, 329.0, 2000, dtype=torch.float64)
    es = fpvs(t, fpvs_table_like(t))
    assert bool((es > 0).all()) and bool((es[1:] > es[:-1]).all())


def test_masked_sum_adds_the_selected_levels_in_order():
    g = torch.Generator().manual_seed(0)
    terms = torch.randn(7, 5, generator=g, dtype=torch.float64)
    mask = torch.rand(7, 5, generator=g) > 0.4
    total = torch.zeros(5, dtype=torch.float64)
    expected = total.clone()
    for k in range(7):
        expected = torch.where(mask[k], expected + terms[k], expected)
    assert torch.equal(masked_sum(total, mask, terms), expected)
