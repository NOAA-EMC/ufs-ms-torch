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

Python 3.11 or 3.12:

```bash
python -m pip install .
```

## Run a forecast

```bash
python -m ufs_mpas_ccpp.forecast RUN_DIR OUT_DIR --hours 6 --output-hours 1
```

`RUN_DIR` is an MPAS-in-UFS runtime folder for a global mesh, quasi-uniform
(`x1.10242`) or variable-resolution (`x4.2562`). It holds:

- `input.nml` and `namelist.atmosphere`;
- the initial conditions named in `input.nml` and the mesh, `x<refinement>.<cells>.grid.nc`;
- `noahmptable.tbl` and `global_o3prdlos.f77` from the GFS fix files;
- `rrtmgp/` with the four RRTMGP coefficient files listed in
  `rrtmgp_radiation_torch.tables.TABLE_FILES`;
- the TEMPO tables `freeze_water_data_tempo_v3`, `qr_acr_qg_data_tempo_v3`,
  `qr_acr_qs_data_tempo_v3` and `ccn_activate.bin` (or set `TEMPO_TABLE_DIR`).

The forecast writes `history.<time>.nc` with the prognostic fields and
`diag.<time>.nc` with surface diagnostics at each output time. Options:
`--device cuda`, `--dtype f32`, `--compile` and `--threads N`.

## Several processes

Start one process per GPU or CPU rank with `torchrun` and add
`--no-step-graphs`. By default the mesh is split by spatial bisection. To use a
METIS partition, give the MPAS `graph.info.part.N` file for N ranks:

```bash
torchrun --nproc_per_node 4 -m ufs_mpas_ccpp.forecast RUN_DIR OUT_DIR --hours 6 \
    --no-step-graphs --partition x4.2562.graph.info.part.4
```

Each rank writes its own cells to `OUT_DIR/rankNNNNN/`, with `global_cell_id`
and `global_edge_id` to place them on the mesh.

From Python, `ufs_mpas_ccpp.forecast.run_forecast` runs the same forecast.
`ufs_mpas_ccpp.native_case.build_native_case` returns the model state and
arguments for `ufs_mpas_ccpp.initialize_cold_cycle` and
`ufs_mpas_ccpp.run_coupled_interval`, which you can call under
`torch.enable_grad()` or a `torch.autograd.forward_ad` dual level.

## Status

CPU in float64 is the reference configuration. GPU and float32 runs work; their
accuracy against the Fortran model is not yet established.

## Tests

```bash
python -m pip install pytest
python -m pytest tests
```

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
