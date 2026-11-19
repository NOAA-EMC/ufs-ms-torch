"""Dycore configuration: options the model does not implement are refused before any step runs."""
import pytest

from mpas_pytorch.config_defaults import validate_config
from mpas_pytorch.exceptions import MpasConfigError


@pytest.mark.parametrize("value", [True, ".true.", "true"])
def test_lateral_boundary_conditions_are_refused(value):
    for key in ("config_apply_lbcs", "apply_lbcs"):
        with pytest.raises(MpasConfigError, match="lateral boundary conditions are not implemented"):
            validate_config({key: value})
    assert validate_config({"config_apply_lbcs": False})["config_apply_lbcs"] is False
