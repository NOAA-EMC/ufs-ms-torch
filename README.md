# ufs-ms-torch

The MPAS-Atmosphere nonhydrostatic dynamical core coupled to the GFS v17 physics
suite of the Common Community Physics Package (CCPP), written in PyTorch.

The model reads the same runtime folder as the Fortran MPAS-in-UFS model. All
numerics are PyTorch tensor operations, so a forecast runs on CPU or GPU and can
be differentiated in reverse and forward mode.

| Package | Content |
|---|---|
| `mpas_pytorch` | MPAS-Atmosphere dynamics: split-explicit RK3 with acoustic substeps, scalar transport, mesh and I/O |
| `ufs_mpas_ccpp` | Physics-dynamics coupling, the GFS host interstitials, the forecast driver |
| `ufs_physics_common` | Shared numerical helpers |
| `rrtmgp_radiation_torch` | RRTMGP shortwave and longwave radiation |
| `tempo_torch` | TEMPO microphysics |
| `sa_tke_edmf_torch` | Scale-aware TKE-EDMF boundary layer |
| `sa_sas_deep_convection_torch`, `sa_sas_shallow_convection_torch` | Scale-aware SAS convection |
| `gfs_surface_layer_torch` | GFS surface layer |
| `noahmp_land_surface_torch` | Noah-MP land surface |

## Install

Python 3.11 or 3.12 and PyTorch 2.8.0:

```bash
python -m pip install .
```

The model uses a few PyTorch internals that change between releases, so it
supports PyTorch 2.8.0 only and refuses other versions at import.

## Run a forecast

```bash
python -m ufs_mpas_ccpp.forecast RUN_DIR OUT_DIR --hours 6 --output-hours 1
```

`RUN_DIR` is an MPAS-in-UFS runtime folder for a global mesh, quasi-uniform or
variable-resolution. It holds:

- `input.nml` and `namelist.atmosphere`;
- the initial conditions named in `input.nml` and the mesh, `x<refinement>.<cells>.grid.nc`;
- `noahmptable.tbl` and `global_o3prdlos.f77` from the GFS fix files;
- the four RRTMGP coefficient files listed in
  `rrtmgp_radiation_torch.tables.TABLE_FILES`, in `rrtmgp/` or in the folder
  itself (or set `RRTMGP_TABLE_DIR`);
- the TEMPO tables `freeze_water_data_tempo_v3`, `qr_acr_qg_data_tempo_v3`,
  `qr_acr_qs_data_tempo_v3` and `ccn_activate.bin` (or set `TEMPO_TABLE_DIR`).

The forecast writes `history.<time>.nc` with the prognostic fields and
`diag.<time>.nc` with surface diagnostics at each output time, with the cells
and edges in MPAS mesh order. Options: `--device cuda`, `--dtype f32`,
`--compile` and `--threads N`.

## Several processes

Start one process per GPU or CPU rank with `torchrun` and add
`--no-step-graphs`. By default the mesh is split by spatial bisection. To use a
METIS partition, give the MPAS `graph.info.part.N` file for N ranks:

```bash
torchrun --nproc_per_node 4 -m ufs_mpas_ccpp.forecast RUN_DIR OUT_DIR --hours 6 \
    --no-step-graphs --partition MESH.graph.info.part.4
```

Each rank writes its own cells to `OUT_DIR/rankNNNNN/`, with `global_cell_id`
and `global_edge_id` to place them on the mesh.

From Python, `ufs_mpas_ccpp.forecast.run_forecast` runs the same forecast.
`ufs_mpas_ccpp.native_case.build_native_case` returns the model state and
arguments for `ufs_mpas_ccpp.initialize_cold_cycle` and
`ufs_mpas_ccpp.run_coupled_interval`, which you can call under
`torch.enable_grad()` or a `torch.autograd.forward_ad` dual level.

## Parameter scales

Selected tuning parameters of the dynamics and the physics can be scaled by a
0-d floating tensor `s`, so that `s = 1` gives the unchanged model and
derivatives with respect to `s` follow from autograd. Dynamics scales go in
`config["parameter_scales"]` of the MPAS configuration (`arguments["config"]` in
a coupled run), keyed by the names in
`mpas_pytorch.config_defaults.PARAMETER_SCALE_KEYS`. Physics scales are one
dictionary, keyed by the names in `ufs_mpas_ccpp.parameters.PHYSICS_PARAMETERS`,
passed as `parameter_scales` in both `arguments["physics_arguments"]` and
`arguments["microphysics_arguments"]`.

## Changes from the GFS v17 physics

The physics follows the GFS v17 CCPP code except in the places below. Each
change closes the water or energy budget of the coupled model, or avoids an
invalid table lookup, and each can change results relative to GFS v17. A
comment at each site in the code describes the upstream behaviour, the reason
and the effect.

Coupling

- The temperature change of each physics scheme is applied as MPAS internal
  energy on the mass the physics saw, instead of setting the MPAS temperature to
  the GFS temperature, so the dynamics keeps the heat each scheme supplies.

Convection (SAMF)

- A water reconciliation after deep and shallow convection makes each column's
  water change match its convective rain, with the matching latent heat. A large
  correction stops the run.
- Detrained condensate that freezes releases its heat of fusion.
- Rain that evaporates adds exactly its water to the vapour and cools the air by
  the latent heat at the temperature at the start of the physics step.
- At the plume launch levels, the water exchange removed from the column uses
  the humidity the plume was launched with, so the column water is conserved.

Microphysics (TEMPO)

- Phase changes heat or cool the air with the MPAS latent heats and the mass at
  the start of the physics step, including freezing and melting of falling
  hydrometeors.
- Freezing and deposition use the MPAS heats of fusion and sublimation.
- The sedimentation reconstruction resets a negative upper face value, so
  falling mass is conserved.
- Melting and freezing act only on amounts above the scheme's presence
  threshold, so round-off remnants do not change number or heat.
- Cloud water that settles to the ground counts as surface precipitation.

Land (Noah-MP)

- The moisture flux to the atmosphere is the water Noah-MP removes from the land
  (canopy evaporation, transpiration and direct evaporation), not the latent
  heat divided by one heat of vaporization.
- Water lost in the soil-water solve without a physical flux is reported as a
  separate budget term and is not added to runoff.

Radiation (RRTMGP)

- Cloud particle radii are bounded by the shortwave table limits before the
  shortwave lookup, so no column is left without cloud optics.
- The latitude index of the CO2 table is bounded at both ends, so a polar column
  reads the first table row.

## Status

CPU in float64 is the reference configuration. GPU and float32 runs work; their
accuracy against the Fortran model is not yet established. Limited-area
configurations with lateral boundary conditions are not supported.

## Tests

```bash
python -m pip install ".[test]"
python -m pytest
```

The tests need no input data. Continuous integration runs them, with a lint
check, on Python 3.11 and 3.12 for every push and pull request.

## License

PolyForm Noncommercial 1.0.0 with the additional permissions in
`PERMISSIONS.md`. Noncommercial use, and use by the noncommercial organizations
that the license names (government, educational, research and charitable
organizations, among others), is permitted. A company may do work on the
software under a contract with such an organization, for that organization only
(`PERMISSIONS.md` section 2). Any other commercial use needs a license from IC
Weather LLC. Portions modified from MPAS, CCPP physics, TEMPO and RTE+RRTMGP
keep their original terms; see `NOTICE`. By submitting a contribution, you agree
to the terms in `CLA.md`. The organization that hosts this repository does not
own the software and cannot change its license; see `PERMISSIONS.md` section 5.
