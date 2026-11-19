"""Tuning-parameter scales: validation, the continuation digest, an adjoint gradient and the physics registry."""
import contextlib

import pytest
import torch
from torch.autograd import forward_ad

from mpas_pytorch.config_defaults import validate_config
from mpas_pytorch.exceptions import MpasConfigError
from mpas_pytorch.io_schema import continuation_configuration_digest
from ufs_mpas_ccpp import parameters

IDENTITY_RTOL = 1.0e-10


def _scale(value=1.0):
    return torch.tensor(value, dtype=torch.float64)


def test_dycore_scales_are_validated():
    with pytest.raises(MpasConfigError, match="unsupported keys"):
        validate_config({"parameter_scales": {"config_visc4": _scale()}})
    with pytest.raises(MpasConfigError, match="0-d floating tensor"):
        validate_config({"parameter_scales": {"config_smdiv": torch.ones(2, dtype=torch.float64)}})
    with pytest.raises(MpasConfigError, match="0-d floating tensor"):
        validate_config({"parameter_scales": {"config_smdiv": 1.0}})
    with pytest.raises(MpasConfigError, match="2d_smagorinsky"):
        validate_config({"config_horiz_mixing": "2d_fixed", "parameter_scales": {"config_visc4_2dsmag": _scale()}})
    validated = validate_config({"parameter_scales": {"config_smdiv": _scale(0.5)}})
    assert float(validated["parameter_scales"]["config_smdiv"]) == 0.5


def test_continuation_digest_reads_scaled_keys_not_values():
    def digest(value):
        return continuation_configuration_digest({"config_dt": 720.0, "parameter_scales": {"config_smdiv": _scale(value)}})

    assert digest(1.0) == digest(2.0)
    assert digest(1.0) != continuation_configuration_digest({"config_dt": 720.0})


def _edge_advection_case():
    """A small ordered edge-advection stencil (both upwind signs) with its output-owned transposes."""
    from mpas_pytorch.dynamics.static_payloads import build_ordered_output_transpose

    generator = torch.Generator().manual_seed(40)

    def rand(shape, offset=0.0, scale=1.0):
        return offset + scale * torch.randn(shape, generator=generator, dtype=torch.float64)

    adv_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
    adv_valid = torch.tensor([[True, True], [True, True], [True, False], [True, True], [True, True]])
    local_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
    local_valid = torch.ones_like(local_indices, dtype=torch.bool)
    local_sign = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]], dtype=torch.float64)
    adv_destinations, adv_transpose_valid = build_ordered_output_transpose(adv_indices, 4, valid=adv_valid)
    local_destinations, local_transpose_valid = build_ordered_output_transpose(local_indices, 5, valid=local_valid)
    adv_coef, adv_coef_3rd = rand((5, 2), 0.4, 0.1), rand((5, 2), 0.1, 0.02)
    # Relation tuples in StaticSlotRelation field order (indices 0, valid 1, coefficients 4 and 5, valid_all_by_slot
    # 6, transpose destinations 11 and validity 12).
    relation = (adv_indices, adv_valid, None, None, adv_coef, adv_coef_3rd, (True, False), None, None, None, None,
                adv_destinations, adv_transpose_valid)
    local = (local_indices, local_valid, local_sign, None, None, None, (True, True), None, None, None, None,
             local_destinations, local_transpose_valid)
    ru = rand((5, 4), 2.0, 0.2)
    ru[0, 1] = -ru[0, 1]
    ru[3, 2] = -ru[3, 2]
    return relation, local, rand((4, 4)), ru, rand((4, 4))


def test_coef_3rd_order_scale_gradient_through_the_ordered_adjoint():
    """The ordered edge-advection adjoint returns the gradient of the config_coef_3rd_order scale, equal to native
    autograd; a forward tangent on the scale alone takes the plain path, whose JVP equals that gradient."""
    from mpas_pytorch.autodiff import differentiable_mode
    from mpas_pytorch.dynamics import dyn_tend

    relation, local, field0, ru0, weights = _edge_advection_case()

    def tendency(field, ru, scale):
        return dyn_tend._edge_advective_cell_tendency(
            field=field, field_start=0, field_stop=4, ru_edge=ru, edge_adv_cell_relation=relation,
            local_edge_relation=local, sign_ru=dyn_tend.fct_sign_pm(ru), coef_3rd_scale=scale,
        )

    def gradients(policy):
        leaves = [field0.clone().requires_grad_(True), ru0.clone().requires_grad_(True),
                  _scale(1.3).requires_grad_(True)]
        with differentiable_mode(), policy:
            out = tendency(*leaves)
        return type(out.grad_fn).__name__, torch.autograd.grad((weights * out).sum(), leaves)

    native = dyn_tend.build_dynamics_tendency_ad_compile_plan(mode="native_autograd", device="cpu")
    ordered_fn, ordered = gradients(contextlib.nullcontext())
    native_fn, reference = gradients(dyn_tend.use_dynamics_tendency_ad_compile_plan(native))
    assert ordered_fn == "_EdgeAdvectiveCellTendencyFunctionBackward" != native_fn
    for got, want in zip(ordered, reference):
        torch.testing.assert_close(got, want, rtol=IDENTITY_RTOL, atol=0.0)

    field = field0.clone().requires_grad_(True)
    with differentiable_mode(), forward_ad.dual_level():
        out = tendency(field, ru0, forward_ad.make_dual(_scale(1.3), _scale(1.0)))
        jvp = (weights * forward_ad.unpack_dual(out).tangent).sum()
    assert out.grad_fn is not None and "EdgeAdvective" not in type(out.grad_fn).__name__
    torch.testing.assert_close(jvp, reference[2], rtol=IDENTITY_RTOL, atol=0.0)


def test_physics_scales_are_selected_by_registry_key():
    scales = {"pbl.rlmx": _scale(1.2), "samf.evef": _scale(0.9)}
    names = {"samf.evef": "evef", "pbl.elmx": "elmx", "pbl.rlmx": "rlmx"}
    picked = parameters.select(scales, names)
    assert list(picked) == ["evef", "rlmx"] and picked["rlmx"] is scales["pbl.rlmx"]
    assert parameters.select(None, names) == {}
    like = torch.zeros(3, dtype=torch.float32)
    views = parameters.select(scales, names, like=like)
    assert all(v.shape == (1,) and v.dtype == torch.float32 for v in views.values())
    with pytest.raises(KeyError, match="unregistered"):
        parameters.scale(scales, "pbl.not_a_parameter")
