"""The SAMF water reconciliation bound: rounding of the column water in float32, a 1 kg floor in float64."""
import pytest
import torch

from sa_sas_deep_convection_torch import samf_interstitials
from sa_sas_deep_convection_torch.samf_interstitials import mpas_tempo_layout, samf_water_reconciliation


def inputs():
    dtype = torch.float64
    entry_qv = torch.tensor([[.02, .008, .001], [.015, .006, .002]], dtype=dtype)
    saved = torch.tensor([
        [[.018, .001, .0002, .0001, .0003, .0001, 1.e5, 2.e4, .3, 1.e-7],
         [.007, .0001, .0001, .0002, .0001, .0003, 2.e5, 3.e4, .2, 2.e-7],
         [.001, .0001, .0001, .0001, .0001, .0001, 3.e5, 4.e4, .1, 3.e-7]],
        [[.014, .0004, .0001, .0002, .0001, .0001, 4.e5, 5.e4, .4, 4.e-7],
         [.005, .0002, .0001, .0001, .0002, .0001, 5.e5, 6.e4, .3, 5.e-7],
         [.002, .0001, .0002, .0001, .0001, .0002, 6.e5, 7.e4, .2, 6.e-7]]], dtype=dtype)
    raw = saved.clone()
    raw[:, :, 0] += torch.tensor([[1.e-5, -3.e-6, 2.e-7], [-1.e-5, 2.e-6, -1.e-7]], dtype=dtype)
    raw[:, :, 1] += 2.e-7

    def scalar(value):
        return torch.tensor(value, dtype=dtype)

    return {"gq0": raw, "gt0": torch.full((2, 3), 275., dtype=dtype), "save_q": saved,
            "rain1": torch.tensor([1.e-7, 2.e-7], dtype=dtype), "area": torch.tensor([1.e6, 2.e6], dtype=dtype),
            "prsi": torch.tensor([[100000., 70000., 30000., 1000.], [95000., 68000., 28000., 1000.]], dtype=dtype),
            "frain": scalar(1.), "con_g": scalar(9.80665), "con_cp": scalar(1004.6), "con_hvap": scalar(2.5e6),
            "layout": mpas_tempo_layout(), "delt": scalar(1200.), "entry_qv": entry_qv,
            "entry_temperature": torch.tensor([[300., 270., 235.], [290., 260., 220.]], dtype=dtype),
            "con_rd": scalar(287.05), "con_rv": scalar(461.5)}


def fp32_columns(ncol=256, km=55):
    """Float32 columns of a 240 km mesh (5e10 m^2, 1e12 kg of vapour each) with a small convective water change:
    rounding the stored water (about 7e4 kg per float32 step of the column vapour) exceeds a bound scaled to the
    net change alone."""
    g = torch.Generator().manual_seed(0)
    dtype = torch.float64
    f = inputs()
    qv = (.018 * torch.exp(-torch.arange(km, dtype=dtype) / 8.)) * (.5 + torch.rand(ncol, km, generator=g, dtype=dtype))
    saved = torch.zeros(ncol, km, 10, dtype=dtype)
    saved[:, :, 0] = qv
    saved[:, :, 1:6] = 1.e-5 * torch.rand(ncol, km, 5, generator=g, dtype=dtype)
    raw = saved.clone()
    raw[:, :, 0] += 2.e-6 * (torch.rand(ncol, km, generator=g, dtype=dtype) - .5)
    raw[:, :, 1] += 1.e-7
    f.update(gq0=raw, save_q=saved, gt0=torch.full((ncol, km), 270., dtype=dtype),
             prsi=torch.linspace(100000., 1000., km + 1, dtype=dtype).expand(ncol, km + 1).clone(),
             area=torch.full((ncol,), 5.e10, dtype=dtype), rain1=1.e-6 * torch.rand(ncol, generator=g, dtype=dtype),
             entry_qv=qv.clone(), entry_temperature=torch.full((ncol, km), 270., dtype=dtype))
    return {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in f.items()}


def as_float64(f):
    return {k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in f.items()}


def mass(f):
    return f["area"][:, None] * (f["prsi"][:, :-1] - f["prsi"][:, 1:]) / f["con_g"] * (1. - f["entry_qv"])


def column_vapour(f):
    f = as_float64(f)
    return float((mass(f) * f["gq0"][:, :, 0].clamp_min(0.) / (1. - f["gq0"][:, :, 0])).sum(dim=1)[0])


def bias_applied_change(monkeypatch, f, kilograms):
    """Add ``kilograms`` of water to column 0's applied change (the second water-change evaluation, after the
    reconciliation), so its residual exceeds the closed column's by that much."""
    layer = mass(as_float64(f))
    original, calls = samf_interstitials._native_water_change_levels, []

    def biased(*args):
        change = original(*args)
        calls.append(None)
        if len(calls) == 2:
            change = change.clone()
            change[0, 0] += kilograms / float(layer[0, 0])
        return change

    monkeypatch.setattr(samf_interstitials, "_native_water_change_levels", biased)


@pytest.mark.parametrize("steps", [None, 7., 8.5, 9.])
def test_fp32_column_residual_bound_is_the_rounding_of_its_vapour(monkeypatch, steps):
    """A closed float32 column passes; a residual added to the applied water change fails above eight float32 steps
    of the column vapour and passes below."""
    f = fp32_columns()
    if steps is not None:
        bias_applied_change(monkeypatch, f, steps * torch.finfo(torch.float32).eps * column_vapour(f))
    assert float(samf_water_reconciliation(**f)[2]) == (0. if steps in (None, 7.) else 1.)


def test_fp32_column_residual_bound_scales_with_the_requested_water(monkeypatch):
    """A column that rains 1% of its vapour: the change-scaled term admits a residual of 30 vapour steps."""
    f = fp32_columns()
    f["rain1"] = torch.full_like(f["rain1"], 1.e-2 * column_vapour(f) / (float(f["area"][0]) * 1000.))
    assert float(samf_water_reconciliation(**f)[2]) == 0.
    bias_applied_change(monkeypatch, f, 30. * torch.finfo(torch.float32).eps * column_vapour(f))
    assert float(samf_water_reconciliation(**f)[2]) == 0.


@pytest.mark.parametrize("kilograms", [.95, 1.05])
def test_fp64_column_residual_bound_floor_is_one_kilogram(monkeypatch, kilograms):
    """Float64 rounding of these columns is far below a kilogram, so the 1 kg floor is the bound."""
    f = inputs()
    bias_applied_change(monkeypatch, f, kilograms)
    assert float(samf_water_reconciliation(**f)[2]) == (1. if kilograms > 1. else 0.)
