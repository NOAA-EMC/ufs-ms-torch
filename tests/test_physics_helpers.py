"""Small known answers from the physics packages."""
import math

import pytest
import torch

from gfs_surface_layer_torch.driver import fpvs, fpvs_table_like
from rrtmgp_radiation_torch import aerosol
from ufs_physics_common.ordered_sum import masked_sum
from ufs_physics_common.thermo import phase_latent_heat


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


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_phase_latent_heat_is_kirchhoff_linear_in_temperature(dtype):
    g = torch.Generator().manual_seed(5)
    t = (180.0 + 150.0 * torch.rand(64, 7, generator=g, dtype=torch.float64)).to(dtype)
    cp, rd, rv, l0 = (torch.full((), v, dtype=dtype) for v in (1004.6, 287.05, 461.5, 2.5e6))
    expected = l0 - rv * 273.16 + (cp - rd) * (rv / rd) * (t - 273.16)
    assert torch.equal(phase_latent_heat(l0, t, cp, rd, rv), expected)


def test_aerosol_spectrum_needs_no_fortran_source():
    solar, infrared = aerosol._set_spectrum(math.pi)
    assert sum(aerosol.NWVNS0) == aerosol.NWVTOT and len(aerosol.S0INTV) == aerosol.NWVSOL
    assert solar.shape == (aerosol.NWVTOT + 1,) and infrared.shape == (aerosol.NWVTIR + 1,)
    assert solar[0] == 0.0 and (solar[1:] > 0.0).all() and (infrared[1:] > 0.0).all()
