"""The packages import together and the forecast entry point starts."""
import subprocess
import sys

import tempo_torch.host_constants as tempo_constants


def test_coupled_model_selects_the_ccpp_host_constants_for_tempo():
    assert tempo_constants.ACTIVE_PROFILE == "ufs_physcons"


def test_all_packages_import():
    import gfs_surface_layer_torch  # noqa: F401
    import mpas_pytorch  # noqa: F401
    import noahmp_land_surface_torch  # noqa: F401
    import rrtmgp_radiation_torch  # noqa: F401
    import sa_sas_deep_convection_torch  # noqa: F401
    import sa_sas_shallow_convection_torch  # noqa: F401
    import sa_tke_edmf_torch  # noqa: F401
    import ufs_mpas_ccpp
    from ufs_mpas_ccpp import native_case  # noqa: F401

    assert callable(ufs_mpas_ccpp.initialize_cold_cycle) and callable(ufs_mpas_ccpp.run_coupled_interval)


def test_forecast_command_line_help():
    result = subprocess.run([sys.executable, "-m", "ufs_mpas_ccpp.forecast", "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0 and "--hours" in result.stdout
