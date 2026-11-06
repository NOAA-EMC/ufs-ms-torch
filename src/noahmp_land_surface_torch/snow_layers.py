# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Source-ordered snow-layer combination shared by Noah land and glacier."""

from __future__ import annotations

import torch
from .thermo import static_tensor


def combo_temperature(
    *,
    dz: torch.Tensor,
    wliq: torch.Tensor,
    wice: torch.Tensor,
    temp: torch.Tensor,
    dz2: torch.Tensor,
    wliq2: torch.Tensor,
    wice2: torch.Tensor,
    temp2: torch.Tensor,
    freezing_temperature: float = 273.16,
    latent_heat: float = 333600.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    wicec = wice + wice2
    wliqc = wliq + wliq2
    h = (2.094e6 * wice + 4.188e6 * wliq) * (
        temp - freezing_temperature
    ) + latent_heat * wliq
    h2 = (2.094e6 * wice2 + 4.188e6 * wliq2) * (
        temp2 - freezing_temperature
    ) + latent_heat * wliq2
    hc = h + h2
    cold_temp = freezing_temperature + hc / (2.094e6 * wicec + 4.188e6 * wliqc)
    warm_temp = freezing_temperature + (hc - latent_heat * wliqc) / (
        2.094e6 * wicec + 4.188e6 * wliqc
    )
    combined_temp = torch.where(
        hc < 0.0,
        cold_temp,
        torch.where(hc <= latent_heat * wliqc, freezing_temperature, warm_temp),
    )
    return dz + dz2, wliqc, wicec, combined_temp


def combine_snow_layers(
    *,
    stc: torch.Tensor,
    dzsnso: torch.Tensor,
    snice: torch.Tensor,
    snliq: torch.Tensor,
    snow_levels: torch.Tensor,
    snowh: torch.Tensor,
    sneqv: torch.Tensor,
    sh2o: torch.Tensor,
    sice: torch.Tensor,
    ponding1: torch.Tensor,
    ponding2: torch.Tensor,
    accumulate_ponding: bool = True,
    activity: torch.Tensor | None = None,
    freezing_temperature: float = 273.16,
    latent_heat: float = 333600.0,
):
    """Source-ordered three-level removal/combination; columns remain batched."""
    inputs = (
        stc,
        dzsnso,
        snice,
        snliq,
        snow_levels,
        snowh,
        sneqv,
        sh2o,
        sice,
        ponding1,
        ponding2,
    )
    live = snow_levels < 0 if activity is None else activity
    temp, dz, ice, liq = stc[:, :3], dzsnso[:, :3], snice, snliq
    original_top = snow_levels + 3
    columns = torch.arange(3, device=stc.device)[None, :]

    def put(values, index, replacement, mask):
        return torch.where(
            mask[:, None] & (columns == index[:, None]), replacement[:, None], values
        )

    zero = torch.zeros_like(snowh)
    for row in range(3):
        remove = (row >= original_top) & (ice[:, row] <= 0.1)
        if row != 2:
            destination = row + 1
            liq = torch.cat(
                (
                    liq[:, :destination],
                    torch.where(
                        remove, liq[:, destination] + liq[:, row], liq[:, destination]
                    )[:, None],
                    liq[:, destination + 1 :],
                ),
                dim=1,
            )
            ice = torch.cat(
                (
                    ice[:, :destination],
                    torch.where(
                        remove, ice[:, destination] + ice[:, row], ice[:, destination]
                    )[:, None],
                    ice[:, destination + 1 :],
                ),
                dim=1,
            )
            dz = torch.cat(
                (
                    dz[:, :destination],
                    torch.where(
                        remove, dz[:, destination] + dz[:, row], dz[:, destination]
                    )[:, None],
                    dz[:, destination + 1 :],
                ),
                dim=1,
            )
        else:
            transfer = remove & (original_top < 2)
            for name in ("liq", "ice", "dz"):
                values = {"liq": liq, "ice": ice, "dz": dz}[name]
                values = torch.cat(
                    (
                        values[:, :1],
                        torch.where(
                            transfer, values[:, 1] + values[:, 2], values[:, 1]
                        )[:, None],
                        values[:, 2:],
                    ),
                    dim=1,
                )
                if name == "liq":
                    liq = values
                elif name == "ice":
                    ice = values
                else:
                    dz = values
            singleton = remove & (original_top >= 2)
            nonnegative = ice[:, 2] >= 0.0
            ponding1 = torch.where(
                singleton,
                torch.where(nonnegative, liq[:, 2], liq[:, 2] + ice[:, 2]),
                ponding1,
            )
            deficit = singleton & ~nonnegative & (ponding1 < 0.0)
            soil_top = torch.where(
                deficit,
                torch.clamp(
                    sice[:, 0]
                    + ponding1 / torch.where(deficit, dzsnso[:, 3] * 1000.0, 1.0),
                    min=0.0,
                ),
                sice[:, 0],
            )
            sice = torch.cat((soil_top[:, None], sice[:, 1:]), dim=1)
            ponding1 = torch.where(deficit, 0.0, ponding1)
            sneqv = torch.where(
                singleton, torch.where(nonnegative, ice[:, 2], 0.0), sneqv
            )
            snowh = torch.where(
                singleton, torch.where(nonnegative, dz[:, 2], 0.0), snowh
            )
            liq = torch.cat(
                (liq[:, :2], torch.where(singleton, 0.0, liq[:, 2])[:, None]), dim=1
            )
            ice = torch.cat(
                (ice[:, :2], torch.where(singleton, 0.0, ice[:, 2])[:, None]), dim=1
            )
            dz = torch.cat(
                (dz[:, :2], torch.where(singleton, 0.0, dz[:, 2])[:, None]), dim=1
            )
        top = snow_levels + 3
        shift = remove & (row > top) & (snow_levels < -1)
        for index in range(row, 0, -1):
            move = shift & (index > top)
            destination = torch.full_like(snow_levels, index)
            temp = put(temp, destination, temp[:, index - 1], move)
            ice = put(ice, destination, ice[:, index - 1], move)
            liq = put(liq, destination, liq[:, index - 1], move)
            dz = put(dz, destination, dz[:, index - 1], move)
        snow_levels = torch.where(remove, snow_levels + 1, snow_levels)

    soil_negative = sice[:, 0] < 0.0
    sh2o = torch.cat(
        (
            torch.where(soil_negative, sh2o[:, 0] + sice[:, 0], sh2o[:, 0])[:, None],
            sh2o[:, 1:],
        ),
        dim=1,
    )
    sice = torch.cat(
        (torch.where(soil_negative, 0.0, sice[:, 0])[:, None], sice[:, 1:]), dim=1
    )
    top = snow_levels + 3
    total, height, total_ice, total_liq = zero, zero, zero, zero
    for row in range(3):
        active = row >= top
        total = (
            total
            + torch.where(active, ice[:, row], 0.0)
            + torch.where(active, liq[:, row], 0.0)
        )
        height = height + torch.where(active, dz[:, row], 0.0)
        total_ice = total_ice + torch.where(active, ice[:, row], 0.0)
        total_liq = total_liq + torch.where(active, liq[:, row], 0.0)
    has_layers = snow_levels < 0
    sneqv = torch.where(has_layers, total, sneqv)
    snowh = torch.where(has_layers, height, snowh)
    shallow = has_layers & (snowh < 0.025)
    snow_levels = torch.where(shallow, 0, snow_levels)
    sneqv = torch.where(shallow, total_ice, sneqv)
    ponding2 = torch.where(
        shallow, ponding2 + total_liq if accumulate_ponding else total_liq, ponding2
    )
    snowh = torch.where(shallow & (sneqv <= 0.0), 0.0, snowh)

    original_top = snow_levels + 3
    multi = snow_levels < -1
    rank = torch.zeros_like(snow_levels)
    dz_min = static_tensor((0.025, 0.025, 0.1), device=stc.device, dtype=stc.dtype)
    for row in range(3):
        eligible = multi & (row >= original_top) & (snow_levels < -1)
        threshold = dz_min[torch.clamp(rank, max=2)]
        merge = eligible & (dz[:, row] < threshold)
        top = snow_levels + 3
        previous, following = max(0, row - 1), min(2, row + 1)
        choose_up = (dz[:, previous] + dz[:, row]) < (dz[:, following] + dz[:, row])
        normal_neighbor = (
            torch.full_like(snow_levels, row - 1)
            if row == 2
            else torch.where(choose_up, row - 1, row + 1)
        )
        neighbor = torch.where(row == top, row + 1, normal_neighbor)
        neighbor = torch.clamp(neighbor, min=0, max=2)
        lower = torch.maximum(neighbor, torch.full_like(neighbor, row))
        upper = torch.minimum(neighbor, torch.full_like(neighbor, row))

        def take(values, index):
            return values.gather(1, index[:, None])[:, 0]

        combined = combo_temperature(
            dz=take(dz, lower),
            wliq=torch.where(merge, take(liq, lower), 0.0),
            wice=torch.where(merge, take(ice, lower), 1.0),
            temp=take(temp, lower),
            dz2=take(dz, upper),
            wliq2=torch.where(merge, take(liq, upper), 0.0),
            wice2=torch.where(merge, take(ice, upper), 0.0),
            temp2=take(temp, upper),
            freezing_temperature=freezing_temperature,
            latent_heat=latent_heat,
        )
        dz = put(dz, lower, combined[0], merge)
        liq = put(liq, lower, combined[1], merge)
        ice = put(ice, lower, combined[2], merge)
        temp = put(temp, lower, combined[3], merge)
        # At most three snow layers: the source's shift has one possible row.
        shift = merge & (lower == 2) & (top == 0)
        destination = torch.ones_like(snow_levels)
        temp = put(temp, destination, temp[:, 0], shift)
        ice = put(ice, destination, ice[:, 0], shift)
        liq = put(liq, destination, liq[:, 0], shift)
        dz = put(dz, destination, dz[:, 0], shift)
        snow_levels = torch.where(merge, snow_levels + 1, snow_levels)
        rank = torch.where(eligible & ~merge, rank + 1, rank)
    outputs = (
        torch.cat((temp, stc[:, 3:]), dim=1),
        torch.cat((dz, dzsnso[:, 3:]), dim=1),
        ice,
        liq,
        snow_levels,
        snowh,
        sneqv,
        sh2o,
        sice,
        ponding1,
        ponding2,
    )

    return tuple(
        torch.where(
            live.reshape((live.shape[0],) + (1,) * (value.ndim - 1)), value, initial
        )
        for value, initial in zip(outputs, inputs)
    )
