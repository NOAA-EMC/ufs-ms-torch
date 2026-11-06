# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Coupled-model inputs from an MPAS-in-UFS runtime folder.

The folder holds what the Fortran model reads: the initial conditions, the
``x<refinement>.<cells>.grid.nc`` mesh (``x1.10242``, ``x4.2562``), ``input.nml``,
``namelist.atmosphere``, the Noah-MP, ozone, RRTMGP (``rrtmgp/``) and TEMPO
tables. ``build_native_case`` returns the operands of ``initialize_cold_cycle``
and ``run_coupled_interval`` for one rank, on CPU, in float64, with every column of the mesh.

Values that the GFS host leaves unassigned on this path are set explicitly:
``doGP_smearclds`` is false and the unused TEMPO tracer indices are zero. The
microphysics carry (``hpbl``, ``ud_mf``, ``prevsq``) starts at zero.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import replace
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from struct import Struct

import f90nml
import numpy as np
import torch
from netCDF4 import Dataset

from gfs_surface_layer_torch.driver import fpvs_table_like
from mpas_pytorch import atmosphere_model
from mpas_pytorch.ic_loader import load_config_from_namelist, load_initial_state
from mpas_pytorch.state import mark_complete_step_boundary
from noahmp_land_surface_torch.tables import TABLE_PATH as NOAHMP_TABLE, _integer, _numbers, _section
from rrtmgp_radiation_torch.radiation_group import table_metadata
from rrtmgp_radiation_torch.tables import load_rrtmgp_tables
from tempo_torch.mp_tempo import mp_tempo_init

from . import (
    CommitDomain, ConvectiveCloudHistory, LandInitializationTables, PrognosticOzone, SoilParameters,
    SurfaceControl,
    advance_utc_date,
    convective_cloud_clock, initialize_bridge, initialize_noahmp_cold_surface, initialize_surface,
    prepare_physics_calendar,
    prepare_r4_diagnostic_indices, prepare_r4_gwd_controls, prepare_r4_microphysics_controls,
    prepare_r4_physics_controls, prepare_r4_radiation_config, prepare_timestep_clock,
)
from .land import prepare_noahmp_coupled_case

TRACERS = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "sgs_tke", "o3mr")
OZONE_SHA256 = "41315c2c3ae49311a6e2ca36b0e08ace6242e1d15cf3ae08f2f90644094a5cc5"  # global_o3prdlos.f77
H2O_SHA256 = "2d710cc6a58e077bcd4f0b8440dbe23397636d677aa442b311b6fec67a3e6f61"  # global_h2oprdlos.f77
UNASSIGNED_TEMPO_INDICES = {"ntclamt": 0, "nthv": 0, "ntrz": 0, "ntgz": 0, "nthz": 0, "ntccn": 0}
SMEARCLDS = False
NLEV = 55
DZS = (0.1, 0.3, 0.6, 1.0)  # GFS_typedefs.F90 four-layer soil thicknesses
ZS = (-0.1, -0.4, -1.0, -2.0)
TEMPO_TABLE_FILES = ("freeze_water_data_tempo_v3", "qr_acr_qg_data_tempo_v3", "qr_acr_qs_data_tempo_v3",
                     "ccn_activate.bin")

# GFS production diagnostics: no 3-D tendency diagnostics and no clear-sky heating rates.
# Prognostic fields are the same as with the full diagnostics.
PERFORMANCE_PHYSICS = {"ldiag3d": False, "qdiag3d": False, "swhtr": False, "lwhtr": False}


def load_active_config(runtime: Path, output: Path, *, require_dt: float | None = 300.0) -> dict:
    """The MPAS configuration of a UFS run, written to ``output`` and read by the MPAS loader.

    ``input.nml`` carries the ``mpas_*`` dynamics, damping and I/O members; they are
    merged into ``namelist.atmosphere`` as ``config_*``. ``require_dt=None`` accepts any
    time step.
    """
    mpas = f90nml.read(runtime / "namelist.atmosphere")
    ufs = f90nml.read(runtime / "input.nml")
    # This flag makes the host load the surface orography that the gravity-wave drag uses.
    gwd = mpas["assimilation"].pop("config_ufs_ccpp_gwd_static")
    if gwd is not True:
        raise ValueError("namelist.atmosphere must set config_ufs_ccpp_gwd_static = .true.")
    for group in (
        "nhyd_model",
        "damping",
        "limited_area",
        "io",
        "decomposition",
        "restart",
        "printout",
    ):
        source = ufs.get(f"mpas_{group}", {})
        for name, value in source.items():
            if not name.startswith("mpas_"):
                raise ValueError(f"Unexpected UFS MPAS namelist member: {name}")
            mpas.setdefault(group, {})["config_" + name.removeprefix("mpas_")] = value
    dt = float(ufs["mpas_nhyd_model"]["mpas_dt"])
    if require_dt is not None and dt != require_dt:
        raise ValueError(f"expected mpas_dt = {require_dt} s, found {dt} s")
    transformed = output / "dycore-active.nml"
    if transformed.exists():
        if f90nml.read(transformed) != mpas:
            raise ValueError("Existing transformed MPAS configuration differs from source inputs")
    else:
        f90nml.write(mpas, transformed)
    config = load_config_from_namelist(transformed)
    config["physical_constants_profile"] = "ufs"
    (output / "active-config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def load_soil_parameters(table: Path) -> SoilParameters:
    text = _section(table.read_text(), "noahmp_soil_stas_parameters")
    arrays = {}
    for name, key in (("bexp", "bb"), ("smcmax", "maxsmc"), ("psisat", "satpsi")):
        values = _numbers(text, key)
        if len(values) != 19:
            raise ValueError("the Noah-MP table must declare 19 STAS soil categories")
        arrays[name] = torch.tensor(
            values + [-1.0e36] * (30 - len(values)), dtype=torch.float64
        )
    return SoilParameters(**arrays)


def load_land_initialization_tables(table: Path) -> LandInitializationTables:
    text = table.read_text()
    veg = _section(text, "noahmp_modis_parameters")
    soil = _section(text, "noahmp_soil_stas_parameters")
    months = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
    arrays = {
        "sla": torch.tensor(_numbers(veg, "sla"), dtype=torch.float64),
        "laim": torch.tensor(
            [_numbers(veg, "lai_" + m) for m in months], dtype=torch.float64
        ).T.contiguous(),
    }
    for name, key in (
        ("bexp", "bb"),
        ("smcmax", "maxsmc"),
        ("dwsat", "satdw"),
        ("dksat", "satdk"),
        ("psisat", "satpsi"),
    ):
        values = _numbers(soil, key)
        if len(values) != 19:
            raise ValueError("the Noah-MP table must declare 19 STAS soil categories")
        arrays[name] = torch.tensor(values + [-1.0e36] * 11, dtype=torch.float64)
    return LandInitializationTables(
        **{
            name: _integer(veg, name)
            for name in ("isbarren", "isice", "isurban", "iswater")
        },
        **arrays,
    )


def load_ozone_table(asset, *, expected_sha256: str, device="cpu") -> PrognosticOzone:
    """Read a GFS ozone (or water vapour) production-loss file, as ``module_ozphys`` does.

    The REAL*4 records are promoted to float64. The returned tensors need no file
    access afterwards and support CPU/GPU forcing and automatic derivatives.
    """
    if (type(expected_sha256) is not str or len(expected_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("ozone preparation requires the expected SHA256")
    payload = Path(asset).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{asset}: SHA256 differs from the supported file")
    with BytesIO(payload) as stream:
        marker = Struct(">I")
        def read_record():
            leading = stream.read(marker.size)
            if len(leading) != marker.size:
                raise ValueError("truncated ozone record marker")
            size, = marker.unpack(leading)
            record = stream.read(size)
            if len(record) != size or stream.read(marker.size) != leading:
                raise ValueError("invalid ozone record framing")
            return record
        record = read_record()
        if len(record) < 16:
            raise ValueError("truncated ozone header")
        ncf,nlat,nlev,ntime = map(int,np.frombuffer(record[:16],dtype=">i4"))
        if min(ncf,nlat,nlev) < 1 or ntime < 2:
            raise ValueError("invalid ozone dimensions")
        coordinate_count = nlat+nlev+ntime+1
        expected_bytes = 24+4*coordinate_count+ntime*ncf*nlev*(4*nlat+8)
        if len(record) != 16+4*coordinate_count or len(payload) != expected_bytes:
            raise ValueError("ozone file record inventory mismatch")
        coordinates = np.frombuffer(record[16:],dtype=">f4").astype(np.float64)
        data = np.empty((nlat,nlev,ncf,ntime),dtype=np.float64)
        for month in range(ntime):
            for coefficient in range(ncf):
                for level in range(nlev):
                    values = np.frombuffer(read_record(),dtype=">f4")
                    if values.size != nlat:
                        raise ValueError("invalid ozone coefficient record")
                    data[:,level,coefficient,month] = values
    tensor = lambda value: torch.tensor(value,dtype=torch.float64,device=device)
    return PrognosticOzone(tensor(coordinates[:nlat]),
        tensor(coordinates[nlat:nlat+nlev]),tensor(coordinates[nlat+nlev:]),tensor(data))


def tempo_table_dir(run_dir: Path) -> Path:
    """The runtime folder if it holds the TEMPO tables, else ``$TEMPO_TABLE_DIR``."""
    if all((run_dir / name).is_file() for name in TEMPO_TABLE_FILES):
        return run_dir
    fallback = os.environ.get("TEMPO_TABLE_DIR")
    if fallback:
        return Path(fallback)
    raise FileNotFoundError(f"{run_dir} lacks the TEMPO tables {', '.join(TEMPO_TABLE_FILES)}; "
                            "copy them there or set TEMPO_TABLE_DIR")


@lru_cache(maxsize=2)
def cold_tempo_tables(table_dir=None):
    """TEMPO lookup tables as tensors, from ``table_dir`` or ``$TEMPO_TABLE_DIR``."""
    from tempo_torch import torch_native
    from tempo_torch.parameters import load_tables
    table_dir = table_dir or os.environ.get("TEMPO_TABLE_DIR")
    if not table_dir:
        raise FileNotFoundError("cold_tempo_tables needs a table folder or TEMPO_TABLE_DIR")
    tables = load_tables(table_dir, mmap=True)
    return torch_native.prepare_torch_tables(tables, torch.empty((), dtype=torch.float64))


def radiation_interval_steps(fhswr: float, dt: float) -> int:
    ratio = fhswr / dt
    integral = math.floor(ratio)
    return max(1, integral + int(ratio - integral >= 0.5))


def build_native_case(run_dir: Path, output: Path, *, ic_name: str | None = None,
                      physics_overrides: dict | None = None) -> dict:
    """Operands of the cold coupled cycle on CPU, float64, one rank, every column of the mesh.

    ``output`` receives the merged MPAS namelist. ``physics_overrides`` replaces
    ``gfs_physics_nml`` entries (for example ``PERFORMANCE_PHYSICS``).
    """
    run_dir, output = Path(run_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    namelist = f90nml.read(run_dir / "input.nml")
    physics_nml = namelist["gfs_physics_nml"]
    for name, value in (physics_overrides or {}).items():
        physics_nml[name] = value
    radiation_nml = physics_nml
    dt_namelist = namelist["mpas_nhyd_model"]["mpas_dt"]  # passed to setup as the source holds it
    dt = float(dt_namelist)
    ic = run_dir / (ic_name or namelist["atmos_model_nml"]["ic_filename"])
    # MPAS mesh names are x<refinement>.<cells>: x1.10242, and variable-resolution x4.2562 or x20.835586.
    match = re.search(r"\bx\d+\.\d+", ic.name)
    if match is None:
        raise ValueError(
            f"no MPAS mesh name x<refinement>.<cells> in the initial-condition name {ic.name}"
        )
    mesh_name = match.group(0)
    grid_file = run_dir / f"{mesh_name}.grid.nc"
    config = load_active_config(run_dir, output, require_dt=None)
    state, native_mesh = load_initial_state(ic, grid_file, config=config, scalar_field_names=TRACERS)
    state, _, _ = atmosphere_model._prepare_timestep_state(state, native_mesh, config=config, dt=dt,
                                                           config_validated=True)
    ids = state["indexToCellID"][:-1]
    ncol = ids.numel()
    state["nCells"] = ncol
    state["nEdges"] = int(native_mesh["nEdges"])
    with Dataset(ic) as d:
        start = datetime.strptime(d.config_start_time, "%Y-%m-%d_%H:%M:%S")
    dzs = torch.tensor(DZS, dtype=torch.float64)
    control = SurfaceControl(ncol, start.month, start.month, 1, False, dzs)
    noah_table = run_dir / "noahmptable.tbl"
    if noah_table.read_bytes() != NOAHMP_TABLE.read_bytes():
        raise ValueError(f"{noah_table} differs from the GFS v17 Noah-MP table this package supports")
    soil = load_soil_parameters(noah_table)
    bridge = initialize_bridge(state, native_mesh, config=config, surface_control=control, soil=soil,
                               n_cells_global=ncol, mpi_size=1)
    idat = torch.tensor((start.year, start.month, start.day, 0, start.hour, start.minute, start.second, 0),
                        dtype=torch.int32)
    fhswr = float(radiation_nml.get("fhswr", 3600.0))
    nsswr = radiation_interval_steps(fhswr, dt)
    clock = prepare_timestep_clock(torch.tensor(0.0, dtype=torch.float64), torch.tensor(dt, dtype=torch.float64),
                                   initial_hour=start.hour, nsswr=nsswr, nslwr=nsswr, nhfrad=0)
    current_date = advance_utc_date(idat, clock.phour)
    julian, yearlen = prepare_physics_calendar(current_date, torch.tensor(365, dtype=torch.int32))
    land = initialize_noahmp_cold_surface(bridge.sfcprop, load_land_initialization_tables(noah_table),
                                          month=start.month, zs=torch.tensor(ZS, dtype=torch.float64), dzs=dzs,
                                          n_cells_solve=ncol)
    surface = initialize_surface(bridge.native_surface, bridge.statein, land, bridge.radiation_albedo,
                                 replace(control, initialize_noahmp_liquid=True), soil).sfcprop
    tables = load_rrtmgp_tables(device="cpu", table_dir=run_dir / "rrtmgp" if (run_dir / "rrtmgp").is_dir() else None)
    table_metadata(tables)
    radiation_config = prepare_r4_radiation_config(
        radiation_nml, dt_physics=dt_namelist, dt_dycore=dt_namelist, initial_year=start.year, initial_month=start.month,
        jdat=current_date, solhr=clock.solhr, frozen_smearclds=SMEARCLDS,
        input_dir=run_dir, grid=bridge.grid.grid, aer_jdat=advance_utc_date(idat, clock.fhour))
    physics_controls = prepare_r4_physics_controls(physics_nml, dt_physics=dt_namelist, dt_dycore=dt_namelist, device="cpu")
    diagnostic_indices = prepare_r4_diagnostic_indices(physics_nml, frozen_chemistry_rows=(100,), device="cpu")
    gwd_setup = prepare_r4_gwd_controls(physics_nml, namelist["cires_ugwp_nml"],
                                        geometry=bridge.reference_geometry, input_dir=run_dir)
    surface_controls = physics_controls.surface
    precontrol, nst_control = surface_controls.composites, surface_controls.nsst
    dtidx, ndtend, nfields = diagnostic_indices.dtidx, diagnostic_indices.ndtend, physics_controls.nphy_f2d
    fhzero = physics_nml["fhzero_array"][0]
    radiation_reset = dict(fhzero=fhzero, fhswr=radiation_config.fhswr, fhlwr=radiation_config.fhlwr,
                           dtp=dt, nszero=math.floor(fhzero * 3600.0 / dt + 0.5))
    diagnostic_reset = dict(huge=precontrol.huge, **{n: bool(physics_nml[n])
                                                     for n in ("ldiag3d", "qdiag3d", "oz_phys_2015")})
    h2o_phys = bool(physics_nml.get("h2o_phys", False))
    if h2o_phys:  # GFS v17: the h2o diagnostics are reset and allocated (with ldiag3d) as well
        diagnostic_reset["h2o_phys"] = True
    # Noah-MP consumes live fields each step. Only its static grid contract belongs in setup.
    statein, grid = bridge.statein, bridge.grid.grid
    table = fpvs_table_like(statein.tgrs)
    noah_options = {**surface_controls.noah_options, "julian": float(julian), "iyrlen": int(yearlen)}
    noah_case = prepare_noahmp_coupled_case(surface, driver_options=noah_options)
    cold_arguments = dict(
        surface_control=control, land_tables=load_land_initialization_tables(noah_table),
        tempo_tables=cold_tempo_tables(str(tempo_table_dir(run_dir))),
        tempo_semi_sedi=bool(physics_nml.get("sedi_semi", False)),
        zs=torch.tensor(ZS, dtype=torch.float64), initial_date=tuple(int(v) for v in idat),
        nphy_f2d=nfields, ndtend=ndtend, n_cells_global=ncol, mpi_size=1)
    if h2o_phys:
        cold_arguments["h2o_diagnostics"] = bool(physics_nml["ldiag3d"])
    native = bridge.native_state
    mesh = native_mesh
    native["nEdges"] = int(mesh["nEdges"])
    mark_complete_step_boundary(native, mesh=mesh, config=config, dt=dt)
    # Only TEMPO's tables and initialized flag enter setup; do not allocate another coupled state.
    tempo_initialized = mp_tempo_init(spechum=statein.qgrs[:, :, 0],
        tables=cold_arguments["tempo_tables"], semi_sedi=cold_arguments["tempo_semi_sedi"])
    setup = prepare_r4_microphysics_controls(
        physics_nml, dt_physics=dt_namelist, dt_dycore=dt_namelist, ncol=ncol, device="cpu",
        frozen_unassigned_indices=dict(UNASSIGNED_TEMPO_INDICES), tempo_initialized=tempo_initialized.is_initialized)
    dtp = statein.prsl.new_tensor(dt)
    micro_clock = prepare_timestep_clock(torch.zeros_like(dtp), dtp, initial_hour=0, nsswr=nsswr, nslwr=nsswr, nhfrad=0)
    clstp = convective_cloud_clock(micro_clock.kdt, micro_clock.lsswr, micro_clock.fhour, nsswr=nsswr,
                                   fhswr=dtp.new_tensor(fhswr))
    micro = dict(
        hpbl=torch.zeros(ncol, dtype=torch.float64), ud_mf=torch.zeros_like(statein.prsl),
        prevsq=torch.zeros_like(statein.prsl),
        cloud_history=ConvectiveCloudHistory.allocate_r4(ncol=ncol, device="cpu"),
        clstp=clstp, dtp=dtp, dtf=dtp, dtidx=dtidx, coefficients=setup.convection.coefficients,
        convection_control=setup.convection.control, cloud_control=setup.cloud, tables=tempo_initialized.tables,
        min_layer_depth=float(((statein.phii[:, 1:] - statein.phii[:, :-1]) / 9.80665).min()) * 0.99,
        tempo_control=setup.tempo, frain=setup.frain, post_control=setup.post,
        kdt=torch.tensor(1, dtype=torch.int32), avg_max_length=setup.avg_max_length,
        is_photochem=setup.is_photochem, tail_control=setup.tail)
    arguments = dict(
        mesh=mesh, config=config, domain=CommitDomain(ncol, int(mesh["nEdges"]), 1), grid=grid,
        initial_date=idat, global_ids=ids, n_cells_global=ncol,
        clock_control=dict(initial_hour=start.hour, nsswr=nsswr, nslwr=nsswr, nhfrad=0),
        seed_control=dict(ipsdlim=100000000, lrseeds=True, isubc_sw=2, isubc_lw=2),
        radiation_config=radiation_config, radiation_tables=tables, radiation_reset=radiation_reset,
        diagnostic_reset=diagnostic_reset, prefix_control=physics_controls.prefix, noah_case=noah_case,
        physics_arguments=dict(
            solhr=clock.solhr, table=table, dtp=physics_controls.dtp, dtf=physics_controls.dtf,
            julian=julian, yearlen=yearlen.to(torch.float64), surface_config=surface_controls.surface_layer,
            nsst_control=nst_control, surface_post_control=surface_controls.post, huge=precontrol.huge,
            dtidx=dtidx, pbl_control=physics_controls.pbl, gwd_control=gwd_setup.control,
            cdmbgwd=gwd_setup.cdmbgwd, alpha_fd=gwd_setup.alpha_fd),
        microphysics_arguments=micro,
        ozone_table=load_ozone_table(run_dir / "global_o3prdlos.f77", expected_sha256=OZONE_SHA256),
        ozone_control=({**setup.photochemistry, "h2o_table": load_ozone_table(  # same layout as ozone
            run_dir / "global_h2oprdlos.f77", expected_sha256=H2O_SHA256)} if h2o_phys else setup.photochemistry),
        crtrh=setup.crtrh, rhcmax=setup.rhcmax,
        water_species=tuple(n in ("qv", "qc", "qr", "qi", "qs", "qg") for n in TRACERS))
    return dict(native=native, mesh=mesh, config=config, cold_arguments=cold_arguments, arguments=arguments,
                dt=dt, mesh_name=mesh_name)
