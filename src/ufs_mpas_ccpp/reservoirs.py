# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Selected R4 physical reservoir/flux totals, in original accumulation order."""

from typing import Mapping

import torch

from .containers import NativeAtmosphere, static_tensor
from .ledger import _observer_totals
from .surface_containers import Sfcprop
from .precision import working_like

SURFACE_INPUTS = (
    "slmsk", "smc", "weasd", "canopy", "canicexy", "canliqxy", "waxy",
    "snicexy", "snliqxy", "wtxy", "raincprv", "rainncprv", "tprcp",
)
DIAGNOSTIC_INPUTS = (
    "totprcp", "runoff", "dqsfc", "tecan", "tetran", "tedir", "sbsnoa", "dqsfci",
)
TRACERS = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "sgs_tke", "o3mr")


def physical_reservoirs(
    native: NativeAtmosphere, surface: Sfcprop, diagnostics: Mapping[str, torch.Tensor],
    *, area_cell: torch.Tensor, rdzw: torch.Tensor, dtp: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Reproduce log_physical_reservoir_ledger for the allocated R4 containers.

    Totals cover all cells: a partition sums every cell's rows in global-ID order.
    Number and TKE integrals are not water mass. Snow layers and wtxy overlap
    the land reservoir; they are crosschecks, not additional storage. Cumulative
    diagnostic fluxes can reset and do not alone establish conservation closure.
    """
    n = native.n_cells_solve
    rho, scalars = native.rho_zz, native.scalars
    if type(n) is not int or n < 1 or rho.ndim != 2 or rho.shape[0] < n:
        raise ValueError("reservoirs require a positive owned atmospheric extent")
    nz = rho.shape[1]
    if nz < 1 or scalars.shape != (*rho.shape, 10):
        raise ValueError("reservoirs require the R4 ten-tracer layout")
    if area_cell.ndim != 1 or area_cell.shape[0] < n or rdzw.shape != (nz,) or dtp.shape != ():
        raise ValueError("invalid reservoir geometry/timestep extent")
    operands = dict(rho_zz=rho[:n], scalars=scalars[:n], area_cell=area_cell[:n], rdzw=rdzw, dtp=dtp)
    for name in SURFACE_INPUTS:
        value = getattr(surface, name)
        tail = (4,) if name == "smc" else (3,) if name in ("snicexy", "snliqxy") else ()
        if value.shape != (n, *tail):
            raise ValueError(f"invalid reservoir surface extent: {name}")
        operands[name] = value
    for name in DIAGNOSTIC_INPUTS:
        value = diagnostics[name]
        if value.shape != (n,):
            raise ValueError(f"invalid reservoir diagnostic extent: {name}")
        operands[name] = value
    for name, value in operands.items():
        if not working_like(value, rho) or value.device != rho.device:
            raise ValueError(f"reservoir {name} must be device-local in the working dtype {rho.dtype}")
        torch._assert_async(torch.isfinite(value).all(), f"nonfinite reservoir {name}")
    area = operands["area_cell"]
    torch._assert_async((area > 0.).all() & (rdzw > 0.).all() & (rho[:n] > 0.).all()
        & (dtp > 0.), "nonpositive reservoir geometry/density/timestep")

    def total(terms):
        # Rows stay cell-major (global rows on a partition); the order is that of reshape(-1, 1).
        return _observer_totals(terms[..., None], rho.new_zeros(1))[0]

    land = ((surface.slmsk >= .5) & (surface.slmsk < 1.5)).to(rho.dtype)
    ice = (surface.slmsk >= 1.5).to(rho.dtype)
    ocean = (surface.slmsk < .5).to(rho.dtype)
    mass = rho[:n] / rdzw * area[:, None]
    terms = mass[..., None] * scalars[:n]
    tracer_totals = _observer_totals(terms, rho.new_zeros(10))
    out = dict(dry_air_kg=total(mass), atmospheric_water_kg=total(terms[..., :6]),
        atmospheric_ozone_kg=total(terms[..., 9]), number_species_count=total(terms[..., 6:8]),
        tke_energy_j=total(terms[..., 8]), dry_mass_weighted_tracers=tracer_totals)
    depths = static_tensor((.1, .3, .6, 1.), dtype=rho.dtype, device=rho.device)
    soil = torch.cumsum(torch.where(land[:, None] > 0., surface.smc, 0.) * depths * 1000., dim=1)[:, -1]
    snow = surface.weasd
    snow_ice = 0. + surface.snicexy[:, 0] + surface.snicexy[:, 1] + surface.snicexy[:, 2]
    snow_liquid = 0. + surface.snliqxy[:, 0] + surface.snliqxy[:, 1] + surface.snliqxy[:, 2]
    canopy = 0. + surface.canicexy + surface.canliqxy
    out.update(soil_water_kg=total(area * land * soil), snow_water_kg=total(area * snow),
        generic_canopy_water_kg=total(area * land * surface.canopy),
        noahmp_canopy_water_kg=total(area * land * canopy),
        aquifer_water_kg=total(area * land * surface.waxy),
        land_surface_water_kg=total(area * land * (soil + snow + canopy + surface.waxy)),
        snow_layer_water_crosscheck_kg=total(torch.stack((area * snow_ice,
            area * snow_liquid), dim=1)),
        wtxy_overlap_crosscheck_kg=total(area * land * surface.wtxy),
        land_area_m2=total(area * land), ocean_area_m2=total(area * ocean),
        ice_area_m2=total(area * ice), total_area_m2=total(area))
    for label, weight in (("land", land), ("ocean", ocean), ("ice", ice)):
        out[f"{label}_snow_water_kg"] = total(area * weight * snow)
    precip = area * 1000. * diagnostics["totprcp"]
    runoff = area * diagnostics["runoff"]
    latent = area * diagnostics["dqsfc"]
    staged = 0. + area * 1000. * surface.raincprv + area * 1000. * surface.rainncprv
    evap = 0. + area * diagnostics["tecan"] + area * diagnostics["tetran"] + area * diagnostics["tedir"]
    instantaneous_latent = area * diagnostics["dqsfci"] * dtp
    for name, cell in (("cumulative_precipitation_kg", precip), ("cumulative_runoff_kg", runoff),
        ("cumulative_surface_latent_energy_j", latent), ("current_step_staged_precipitation_kg", staged),
        ("current_step_surface_latent_energy_j", instantaneous_latent)):
        out[name] = total(cell)
        for label, weight in (("land", land), ("ocean", ocean), ("ice", ice)):
            out[f"{label}_{name}"] = total(weight * cell)
    out.update(land_cumulative_exact_evaporation_kg=total(land * evap),
        land_cumulative_snow_sublimation_energy_j=total(land * (area * diagnostics["sbsnoa"])),
        land_staged_previous_step_precipitation_kg=total(land * staged),
        current_step_convective_precipitation_kg=total(area * 1000. * surface.raincprv),
        current_step_diagnostic_precipitation_kg=total(area * 1000. * surface.tprcp))
    if not rho.is_cuda:  # On CUDA the observer totals are off (NaN by design).
        for name, value in out.items():
            torch._assert_async(torch.isfinite(value).all(), f"nonfinite reservoir total {name}")
    return out
