"""The packages import together and the forecast entry point starts."""
import importlib
import pkgutil
import subprocess
import sys

import pytest

import tempo_torch.host_constants as tempo_constants

PACKAGES = ("gfs_surface_layer_torch", "mpas_pytorch", "noahmp_land_surface_torch", "rrtmgp_radiation_torch",
            "sa_sas_deep_convection_torch", "sa_sas_shallow_convection_torch", "sa_tke_edmf_torch", "tempo_torch",
            "ufs_mpas_ccpp", "ufs_physics_common")


def test_coupled_model_selects_the_ccpp_host_constants_for_tempo():
    assert tempo_constants.ACTIVE_PROFILE == "ufs_physcons"


@pytest.mark.parametrize("package", PACKAGES)
def test_every_module_imports(package):
    root = importlib.import_module(package)
    for module in pkgutil.walk_packages(root.__path__, prefix=f"{package}."):
        importlib.import_module(module.name)


def test_coupled_entry_points_exist():
    import ufs_mpas_ccpp

    assert callable(ufs_mpas_ccpp.initialize_cold_cycle) and callable(ufs_mpas_ccpp.run_coupled_interval)


def test_forecast_command_line_help():
    result = subprocess.run([sys.executable, "-m", "ufs_mpas_ccpp.forecast", "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0 and "--hours" in result.stdout
