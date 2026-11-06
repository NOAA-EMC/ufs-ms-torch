# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Torch implementation of R4's deterministic UGWP-v0 solver-2 spectrum."""

import math

import torch


def latitude_wave_stress(latitude, amplitude=0.03):
    """Original slat_geos5_tamp_v0 source, with latitude in degrees."""
    lat = latitude.abs()
    tropical = 0.75 * torch.exp(-((lat - 3.0) / 8.0).square())
    tropical = torch.where((tropical < 1.2) & (lat <= 3.0), 0.75, tropical)
    middle = 0.50 * torch.exp(-((lat - 60.0) / 23.0).square())
    polar = 0.50 * torch.exp(-((lat - 60.0) / 70.0).square())
    return amplitude * torch.where(lat < 15.3, tropical,
        torch.where(lat < 31.0, 0.10, torch.where(lat < 60.0, middle, polar)))


def wave_spectrum(like, *, nwav=25, nazd=4):
    """Original initsolv_wmsdis_v0 initialization, on the input device."""
    if nwav < 2 or nazd < 1:
        raise ValueError("UGWP requires at least two waves and one azimuth")
    def scalar(value):
        return like.new_full((), value)
    pi2 = 2.0 * math.pi
    angle = torch.arange(nazd, device=like.device, dtype=like.dtype) * (scalar(pi2) / nazd)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    norm = like.new_zeros(())
    for azimuth in range(nazd):
        norm = norm + cosine[azimuth].abs()
    zaz_fct = scalar(2.0) / norm
    xmin, xmax = 1.0 / 125.0, 1.0 / 0.5
    span = xmax - xmin
    dx = scalar(span) / (nwav - 1)
    x1 = scalar(span) / (torch.exp(scalar(span / 0.25)) - 1.0)
    x2 = xmin - x1
    tx = torch.arange(nwav, device=like.device, dtype=like.dtype) * dx + xmin
    exponential = torch.exp((tx - xmin) / 0.25)
    ci = scalar(1.0) / (x1 * exponential + x2)
    dci = ci.square() * (x1 / 0.25) * exponential * dx
    mci = (pi2 / 2000.0) * ci
    ci4 = mci.square().square()
    return dict(zci=ci, zdci=dci, zci4=ci4, zcosang=cosine, zsinang=sine, zaz_fct=zaz_fct)


def nonstationary_gwd(fields, tau_ngw, *, launch_level, nwav=25, nazd=4,
                      efficiency=1.0, diagnostics=False):
    """Original fv3_ugwp_solv2_v0; launch_level uses Fortran's one-based index.

    Keep the source's launch-flux addition and final momentum-work heating.
    Spectral dissipation is an intermediate diagnostic, not the returned heat.
    """
    temperature, u, v, q = (fields[n] for n in ("t1", "u1", "v1", "q1"))
    ncol, nlev = temperature.shape
    if not 2 <= launch_level < nlev:
        raise ValueError("UGWP requires 2 <= launch_level < nlev; source level 1 is undefined")
    start = launch_level - 1
    spectrum = wave_spectrum(temperature, nwav=nwav, nazd=nazd)
    ci, dci, ci4, cosine, sine, aznorm = (spectrum[n] for n in
        ("zci", "zdci", "zci4", "zcosang", "zsinang", "zaz_fct"))
    grav, cp, rd = 9.80665, 1004.6, 287.05
    phi = fields["phil"] * (1.0 / grav)
    current, previous = slice(start, None), slice(start - 1, -1)
    tv = temperature * (1.0 + (461.50 / rd - 1.0) * q)
    inv_tv = temperature.new_full((), 2.0) / (tv[:, current] + tv[:, previous])
    uh = 0.5 * (u[:, previous] + u[:, current])
    vh = 0.5 * (v[:, previous] + v[:, current])
    pressure = fields["prsi"]
    density = pressure[:, start:nlev] * (1.0 / rd) * inv_tv
    dz = phi[:, current] - phi[:, previous]
    delpi = temperature.new_full((), grav) / (pressure[:, start-1:nlev-1] - pressure[:, start:nlev])
    viscosity = 2.e-5 * torch.exp((phi[:, current] + phi[:, previous]) * (0.5 / 7000.0)) + 0.01
    bn2 = (grav / (cp / grav)) * inv_tv * (1.0 + (cp / grav) *
        (temperature[:, current] - temperature[:, previous]) / dz)
    bn2 = torch.clamp(bn2, min=(6.28 / 60.0 / 120.0)**2, max=(6.28 / 30.0)**2)
    bv = torch.sqrt(bn2)
    kx = 6.28e-3 / 200.0
    omega2 = (2.0 * math.pi / 86400.0) + (2.0 * math.pi / 86400.0)
    rotational = ((omega2 * fields["sinlat"]) / kx).square()
    factor = density[:, :-1] / bv[:, :-1]
    bv4 = bv[:, 0].square().square()
    initial = factor[:, 0, None] * bv4[:, None] * ci / (bv4[:, None] + ci4)
    total = temperature.new_zeros(ncol)
    for wave in range(nwav):
        total = total + initial[:, wave] * dci[wave]
    normalization = tau_ngw / total
    factor = normalization[:, None] * factor
    initial = normalization[:, None] * initial
    flux_by_azimuth, heat_by_azimuth, final_spectra = [], [], []
    for azimuth in range(nazd):
        relative = cosine[azimuth] * uh + sine[azimuth] * vh
        relative = relative - relative[:, :1]
        critical = temperature.new_full((ncol,), 0.5)
        flux = initial
        flux_levels, heat_levels = [], []
        for level in range(nlev - launch_level):
            critical = torch.maximum(critical, relative[:, level])
            active = (ci >= critical[:, None]).to(temperature.dtype)
            speed = (ci - relative[:, level, None]).abs()
            doppler2 = (kx * speed).square()
            cdf2 = speed.square() - rotational[:, None]
            positive = cdf2 > 0.0
            safe_cdf2 = torch.where(positive, cdf2, 1.0)
            vertical2 = torch.where(positive, (bn2[:, level, None] - doppler2) / safe_cdf2 - kx*kx, 0.0)
            propagating = positive & (vertical2 > 0.0)
            vertical = torch.sqrt(torch.where(propagating, vertical2, 1.0))
            intrinsic = torch.sqrt(torch.where(propagating, cdf2, 1.0))
            damping = (vertical * vertical * viscosity[:, level, None] / (kx * intrinsic) * vertical).abs()
            decay = torch.where(propagating, torch.exp(-damping * (dz[:, level, None] + dz[:, level, None])), 1.0)
            dissipated = decay * flux
            intrinsic = torch.where(propagating, intrinsic, 0.0)
            inverse_ci = torch.where(ci.abs() > 1.e-12, temperature.new_full((), 1.0) /
                torch.where(ci.abs() > 1.e-12, ci, 1.0), 1.0)
            saturated = factor[:, level, None] * intrinsic * intrinsic * inverse_ci
            next_flux = torch.where(active * (dissipated - saturated) > 0.0, saturated, dissipated)
            # The source initializes launch zpu to tau, then adds the propagated integral.
            integral = tau_ngw if level == 0 else temperature.new_zeros(ncol)
            heat = temperature.new_zeros(ncol)
            for wave in range(nwav):
                vc = active[:, wave] * next_flux[:, wave]
                integral = integral + vc * dci[wave]
                if level > 0 and diagnostics:
                    vm = active[:, wave] * flux[:, wave]
                    # Source zcin retains the last propagation mode in this reduction.
                    weight = delpi[:, level] * (ci[-1] - relative[:, level]).abs() * dci[wave]
                    heat = heat + (vm - torch.minimum(vc, vm)) * weight
            flux_levels.append(integral)
            heat_levels.append(heat)
            flux = next_flux
        flux_by_azimuth.append(torch.stack(flux_levels, dim=1))
        heat_by_azimuth.append(torch.stack(heat_levels, dim=1))
        final_spectra.append(flux)
    xflux, yflux, heat = [temperature.new_zeros((ncol, nlev-launch_level)) for _ in range(3)]
    for azimuth in range(nazd):
        xflux = xflux + (aznorm * cosine[azimuth]) * flux_by_azimuth[azimuth]
        yflux = yflux + (aznorm * sine[azimuth]) * flux_by_azimuth[azimuth]
        heat = heat + (aznorm * (1.0 / cp)) * heat_by_azimuth[azimuth]
    boundary = temperature.new_zeros((ncol, 1))
    xflux, yflux = (torch.cat((boundary, flux, boundary), dim=1) for flux in (xflux, yflux))
    raw_x = (xflux[:, 1:] - xflux[:, :-1]) * delpi
    raw_y = (yflux[:, 1:] - yflux[:, :-1]) * delpi
    ax, ay = (torch.clamp(raw, min=-250.e-5, max=250.e-5) for raw in (raw_x, raw_y))
    thermal = (ax * u[:, current] + ay * v[:, current]) * (1.0 / cp)
    diffusivity = torch.clamp_min(thermal / bn2, 0.01)
    zeros = temperature.new_zeros((ncol, start))
    result = {name: torch.cat((zeros, efficiency * value), dim=1) for name, value in
              (("pdudt", -ax), ("pdvdt", -ay), ("pdtdt", thermal), ("dked", diffusivity))}
    if diagnostics:
        result.update(spectrum, zpu=torch.stack(flux_by_azimuth, dim=2), taux=xflux,
                      tauy=yflux, spectral_heat=heat, raw_x=raw_x, raw_y=raw_y, zbn2=bn2,
                      density=density, uh=uh, vh=vh, v_zmet=dz+dz, vueff=viscosity,
                      delpi=delpi, zfct=factor, initial_flux=initial, zfnorm=normalization,
                      zui=torch.stack([(cosine[a]*uh+sine[a]*vh)[:, :-1] -
                          (cosine[a]*uh[:, :1]+sine[a]*vh[:, :1]) for a in range(nazd)], dim=2),
                      final_flux=torch.stack(final_spectra, dim=2))
    return result
