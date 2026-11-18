# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Forecast driver: the operational runtime profile, ``run_forecast`` and its NetCDF output streams.

Command line::

    python -m ufs_mpas_ccpp.forecast RUN_DIR OUT_DIR --hours 6 [--output-hours 1] [--device cuda]

``RUN_DIR`` is an MPAS-in-UFS runtime folder (see ``native_case``); a saved case file (``.pt``)
with the native state, mesh, configuration and cycle arguments also works. Every
``--output-hours`` the driver writes two files into OUT_DIR, named by valid time:
- ``history.<valid>.nc``: the prognostic state, i.e. the dycore prognostics (``u``, ``w``, ``rw``,
  ``theta_m``, ``rho_zz``, ``scalars``) and every floating surface/land field of ``Sfcprop``;
- ``diag.<valid>.nc``: accumulated precipitation (forecast total ``totprcp``, ``cnvprcp`` and the
  fhzero bucket ``totprcpb``, ``cnvprcpb``, in m), ``t2m``, ``q2m``, ``u10m``, ``v10m``, surface
  pressure (Pa), OLR (``topflw_upfxc``, W m-2, from the last longwave call) and precipitable water
  (kg m-2, vapour mass of the physics column).
Each file holds the rows a rank owns in ascending global cell and edge ID, whatever the owned order of the run
in memory: on one rank that is the MPAS mesh order, and distributed files also carry ``global_cell_id`` and
``global_edge_id``.
Values retain the state's working dtype (float64 or float32). The fields are staged to host memory
(pinned on CUDA, stream-ordered copies) and one background thread writes the files, so the next
interval runs while the previous one is written; the driver waits for the writer at the end and
re-raises its error (at the next output time, at the end, or attached to the driver's own error).
Files become visible only after a complete write. Existing output names are never overwritten;
the driver checks planned names before cold initialization, on every distributed rank.

``operational_profile`` sets the runtime switches of a forecast: observers off, the shortwave
daylit bound of the run, step graphs on (CUDA graphs on GPU, the eager emulator on CPU; one
``StepGraphRunner`` for the whole forecast, so each step variant records once), no column chunk (every
scheme full batch over the owned columns, radiation included), and the default compile policy
(whole-scheme compile off by default; on CPU float64 compiled equals eager bitwise). Overflowed compaction
buffers rerun one rung up (``on_overflow="rerun"``).
With step graphs disabled, global and CPU-partitioned forecasts retain native-five phases across output times.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt_
import queue
import threading
from pathlib import Path
from typing import Any, Mapping

import torch
from mpas_pytorch import _torch_compat as torch_compat
from mpas_pytorch.utils.atomic_output import atomic_output

from . import cycle, runtime_policy
from .precision import move_tree, prepare_case
from .dynamics import native_boundary_supported
from .cycle import (CoupledCycleState, initialize_cold_cycle, mark_cold_boundary, run_coupled_interval,
    start_native_coupled_phase, advance_native_coupled_phase,
    _native_coupled_output_fields)
from .radiation import sw_daylit_bound
from .step_graphs import StepGraphRunner, flatten_state

GRAVITY = 9.80665  # GFS con_g
# The HDF5 library under netCDF4 is not thread-safe (concurrent files crash it): the stream writer thread
# and the driver's reads take this lock. The schemes read their NetCDF tables at their first call (step 1),
# before the first output, and nothing else in the forecast loop opens NetCDF.
_NETCDF_LOCK = threading.Lock()


def retarget(obj: Any, device: torch.device, memo: dict | None = None) -> Any:
    """``obj`` (a case or any nested part of it) with every tensor on ``device``; aliases kept."""
    return move_tree(obj, device=device, memo=memo)


def _storages(obj: Any, found: set[int] | None = None, seen: set[int] | None = None) -> set[int]:
    """Storage addresses of every tensor in ``obj`` (nested mappings, sequences and dataclasses)."""
    found = set() if found is None else found
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return found
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        found.add(obj.untyped_storage().data_ptr())
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            _storages(getattr(obj, field.name), found, seen)
    elif isinstance(obj, Mapping):
        for value in obj.values():
            _storages(value, found, seen)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _storages(value, found, seen)
    return found


def operational_profile(state: CoupledCycleState, *, duration_seconds: int, arguments: Mapping[str, Any],
                        scheme_compile: bool = False) -> dict[str, Any]:
    """Set the forecast's runtime switches (module docstring); return the previous values."""
    share = sw_daylit_bound(state, duration_seconds=duration_seconds, **arguments)
    return {
        "cycle_observers": runtime_policy.set_cycle_observers(False),
        "step_graphs": runtime_policy.set_step_graphs(True),
        "sw_daylit_fraction": runtime_policy.set_sw_daylit_fraction(share),
        "scheme_compile": runtime_policy.set_scheme_compile(scheme_compile),
        "column_chunk": runtime_policy.set_column_chunk(None),  # every scheme full batch, radiation included
    }


def _initial_date(arguments: Mapping[str, Any]) -> dt_.datetime:
    # NCEP idat: year, month, day, time zone, hour, minute, second, millisecond (host output metadata)
    year, month, day, _zone, hour, minute, second, _ms = (int(v) for v in arguments["initial_date"].tolist())
    return dt_.datetime(year, month, day, hour, minute, second)


def history_fields(state: CoupledCycleState) -> dict[str, torch.Tensor]:
    native = state.native_state
    fields = {name: native[name] for name in ("u", "w", "rw", "theta_m", "rho_zz", "scalars")}
    sfcprop = state.prefix.sfcprop
    for field in dataclasses.fields(sfcprop):
        value = getattr(sfcprop, field.name)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            fields[f"sfc_{field.name}"] = value
    return fields


def diagnostic_fields(state: CoupledCycleState) -> dict[str, torch.Tensor]:
    intdiag, sfcprop, statein = state.prefix.carry.intdiag, state.prefix.sfcprop, state.statein
    ncol = statein.prsi.shape[0]
    with torch.no_grad():
        layer_mass = (statein.prsi[:, :-1] - statein.prsi[:, 1:]) / GRAVITY
        precipitable_water = (statein.qgrs[:, :, 0] * layer_mass).sum(dim=1)
    return {
        **{name: intdiag[name] for name in ("totprcp", "cnvprcp", "totprcpb", "cnvprcpb", "u10m", "v10m",
                                            "topflw_upfxc")},
        "t2m": sfcprop.t2m, "q2m": sfcprop.q2m,
        "surface_pressure": state.native_state["surface_pressure"][:ncol],
        "precipitable_water": precipitable_water,
    }


@dataclasses.dataclass(frozen=True)
class OutputOrder:
    """The owned rows a rank writes, in ascending global ID (``output_order``, built once at setup).

    ``cells`` and ``edges`` index the owned cell and edge rows; ``cell_ids`` and ``edge_ids`` are their 1-based
    global IDs in that order. ``labeled`` files (distributed runs) also carry the IDs as variables.
    """

    cells: torch.Tensor
    edges: torch.Tensor
    cell_ids: torch.Tensor
    edge_ids: torch.Tensor
    labeled: bool


def output_order(partition, *, labeled: bool = True, device: torch.device | str | None = None) -> OutputOrder | None:
    """The output row order of ``partition`` (None for an unpartitioned run, whose rows are already in mesh order).

    Host setup, once per forecast: the owned rows sorted by global ID, so every file is in global-ID order and one
    rank writes the MPAS mesh order. ``device`` places the row indices beside the fields they gather."""
    if partition is None:
        return None
    cells = partition.cells.global_ids[:partition.cells.owned_count]
    edges = partition.edges.global_ids[:partition.edges.owned_count]
    cell_rows, edge_rows = torch.argsort(cells), torch.argsort(edges)
    return OutputOrder(cell_rows.to(device), edge_rows.to(device), cells[cell_rows] + 1, edges[edge_rows] + 1,
                       labeled)


def output_fields(fields: Mapping[str, torch.Tensor], order: OutputOrder | None, *, n_cells: int | None = None,
                  n_edges: int | None = None) -> dict[str, torch.Tensor]:
    """The owned rows of ``fields`` in ascending global ID (``output_order``); without an order the first
    ``n_cells`` cell and ``n_edges`` edge rows (the padding row stays out). Labeled files also get the global IDs."""
    def rows(name, value):
        if not value.ndim or name == "pressure_level":
            return value
        if order is None:
            return value[:n_edges if name == "u" else n_cells]
        return value[order.edges if name == "u" else order.cells]

    out = {name: rows(name, value) for name, value in fields.items()}
    if order is not None and order.labeled:
        out.setdefault("global_cell_id", order.cell_ids)
        if "u" in fields:
            out.setdefault("global_edge_id", order.edge_ids)
    return out


def write_stream(path: Path, fields: Mapping[str, torch.Tensor], *, attributes: Mapping[str, Any]) -> None:
    """One NetCDF file with named native cell, edge and vertical axes; nonfinite values raise."""
    import netCDF4

    with _NETCDF_LOCK, atomic_output(path, clobber=False) as temporary, netCDF4.Dataset(temporary, "w") as out:
        for key, value in attributes.items():
            out.setncattr(key, value)
        for name in ("nCells", "nEdges", "nVertLevels"):
            if name in attributes:
                out.createDimension(name, int(attributes[name]))
        if "nVertLevels" in attributes:
            out.createDimension("nVertLevelsP1", int(attributes["nVertLevels"]) + 1)
        for name, tensor in fields.items():
            if tensor.is_floating_point():
                torch_compat.assert_async(torch.isfinite(tensor).all(), f"{name}: nonfinite forecast output")
            array = tensor.cpu().numpy()  # host output boundary (no-grad forecast state)
            first = "nEdges" if name in ("u", "global_edge_id") else "nCells"
            dims = [first] if array.ndim else []
            if name in ("u", "theta_m", "rho_zz", "scalars"):
                dims += ["nVertLevels"]
            elif name in ("w", "rw"):
                dims += ["nVertLevelsP1"]
            dims += [f"{name}_dim{axis}" for axis in range(len(dims), array.ndim)]
            for dim, size in zip(dims, array.shape):
                if dim not in out.dimensions:
                    out.createDimension(dim, size)
                if len(out.dimensions[dim]) != size:
                    raise ValueError(f"{name}: {dim} has {size} rows, expected {len(out.dimensions[dim])}")
            out.createVariable(name, array.dtype, tuple(dims))[...] = array


class StreamWriter:
    """Writes output streams from one background thread (``write_stream`` per file, in submit order).

    ``submit`` stages the fields to host memory first (pinned and non-blocking on CUDA, with an event
    the writer waits for), so the caller may advance the state at once. At most one staged file waits
    while another is written. A write error is kept and raised by the next ``submit`` or by ``close``.
    """

    def __init__(self, device: torch.device, write: Any = None):
        self._pinned = device.type == "cuda"
        self._device = (torch.device("cuda", torch.cuda.current_device())
                        if self._pinned and device.index is None else device)
        self._write = write_stream if write is None else write
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._closed = False
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="forecast-stream-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if self.error is not None:
                continue  # after an error the remaining files are dropped; close() raises it
            path, fields, attributes, event = item
            try:
                if event is not None:
                    event.synchronize()
                self._write(path, fields, attributes=attributes)
            except BaseException as error:  # handed to the driver thread
                self.error = error

    def raise_error(self) -> None:
        if self.error is not None:
            raise RuntimeError("forecast output writer failed") from self.error

    def submit(self, path: Path, fields: Mapping[str, torch.Tensor], *, attributes: Mapping[str, Any]) -> None:
        self.raise_error()
        if self._closed:
            raise RuntimeError("forecast output writer is closed")
        if self._pinned and any(value.device.type == "cuda" and value.device != self._device
                                for value in fields.values()):
            raise ValueError("forecast output fields must use the writer's CUDA device")
        staged = {name: torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=self._pinned)
                  .copy_(value, non_blocking=self._pinned) for name, value in fields.items()}
        event = None
        if self._pinned:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self._device))
        self._queue.put((path, staged, dict(attributes), event))

    def close(self, *, raise_error: bool = True) -> None:
        """Wait for the queued files; raise the writer's error unless ``raise_error`` is false."""
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._thread.join()
        if raise_error:
            self.raise_error()


def read_stream(path: Path) -> dict[str, torch.Tensor]:
    import netCDF4

    with _NETCDF_LOCK, netCDF4.Dataset(path) as data:
        return {name: torch.from_numpy(variable[...].data.copy()) for name, variable in data.variables.items()}


def _prepare_output(output_dir: Path, *, start: dt_.datetime, total: int, interval: int,
                    parallel: bool, setup_group=None) -> None:
    """Reject collisions before model work; propagate rank-local filesystem errors to every rank."""
    error = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for elapsed in range(interval, total + 1, interval):
            stamp = (start + dt_.timedelta(seconds=elapsed)).strftime("%Y-%m-%d_%H.%M.%S")
            for stream in ("history", "diag"):
                path = output_dir / f"{stream}.{stamp}.nc"
                if path.exists() or path.is_symlink():
                    raise FileExistsError(f"forecast output already exists: {path}")
    except OSError as caught:
        error = caught
    if parallel:
        import torch.distributed as dist

        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, str(error) if error else None, group=setup_group)
        if any(item is not None for item in errors):
            details = "; ".join(f"rank {rank}: {item}" for rank, item in enumerate(errors) if item is not None)
            raise OSError(f"distributed forecast output setup failed: {details}") from error
    elif error is not None:
        raise error


def run_forecast(case: Mapping[str, Any] | str | Path, output_dir: str | Path, *, hours: float,
                 output_hours: float = 1.0, device: str = "cpu", scheme_compile: bool = False,
                 dycore_compile: bool = False, step_graphs: bool = True,
                 dtype: torch.dtype | str | None = None, setup_group=None,
                 partition_path: str | Path | None = None, owned_order: str = "hilbert") -> CoupledCycleState:
    """Cold-start ``case``, run ``hours`` with the operational profile, write the two streams every
    ``output_hours``; return the final state.

    Every step runs every scheme full batch over the owned columns (``operational_profile``); no memory
    plan applies to the forward. ``step_graphs=False`` uses ordinary intervals without graph-pool memory;
    compilation of physics and dycore stays independent.
    ``partition_path`` (a ``graph.info.part.N`` file, N = world size) sets the cell owners when this call
    loads the case; one rank checks a ``part.1`` file. ``owned_order="hilbert"`` (the default) orders the owned rows
    of every entity in memory along a Hilbert curve of their centres (``mpas_pytorch.distributed.partition``), so
    neighbouring cells sit close in memory: on one rank the case is renumbered once at setup. ``"global"`` keeps the
    mesh order in memory. Either way every file holds the owned rows in ascending global ID (``output_order``): one
    rank writes the same files in both orders, and distributed files carry the global cell and edge IDs."""
    target = torch.device(device)
    runtime_policy.use_expandable_segments(target)  # before the first device allocation
    import torch.distributed as dist
    from .distributed import load_distributed_case

    owned = not isinstance(case, Mapping)
    parallel = dist.is_initialized() and dist.get_world_size() > 1
    if partition_path is not None and not owned:
        raise ValueError("partition_path applies only when run_forecast loads the case")
    renumbered = None
    if owned and parallel:
        case = load_distributed_case(case, dtype=dtype, device=target, setup_group=setup_group,
                                     partition_path=partition_path, owned_order=owned_order)
    else:
        if owned:
            from .case_io import load_cpu_case
            case = load_cpu_case(case)
        if owned_order != "global" and not parallel:  # one rank: renumber the rows once at setup
            from mpas_pytorch.distributed.partition import build_mesh_partition
            from .distributed import partition_coupled_case

            case = prepare_case(case, dtype=dtype)
            owners = torch.zeros(int(case["mesh"]["nCells"]), dtype=torch.int64)
            renumbered = build_mesh_partition(case["mesh"], owners, rank=0, world_size=1, owned_order=owned_order)
            case = partition_coupled_case(case, [renumbered], 0)
        case = prepare_case(case, dtype=dtype, device=target)
        if partition_path is not None:  # one rank: the file must give every cell to rank 0
            from mpas_pytorch.distributed.partition import cell_owners
            cell_owners(case["mesh"], 1, partition_path)
        owned = owned or target.type != "cpu"
    native, mesh, config = case["native"], case["mesh"], case["config"]
    exchanger = config.get("_halo_exchanger")
    if parallel and exchanger is None:
        raise ValueError("distributed mapping cases must already have a spatial halo exchanger")
    if parallel and dist.get_backend() != ("nccl" if target.type == "cuda" else "gloo"):
        raise ValueError("the numerical backend must match CPU/Gloo or CUDA/NCCL")
    if exchanger is not None and (not parallel or exchanger.transport.device != native["u"].device):
        raise ValueError("the halo exchanger must match the active process group and case device")
    partition = exchanger.partition if exchanger is not None else renumbered
    order = output_order(partition, labeled=parallel, device=native["u"].device)
    if parallel and step_graphs:
        raise ValueError("distributed CUDA Graph replay is not qualified; use --no-step-graphs")
    arguments = case["arguments"]
    if dycore_compile:
        config = runtime_policy.with_dycore_compile(config)
        arguments = {**arguments, "config": config}
    dt = int(config["config_dt"])
    total, interval = int(round(hours * 3600)), int(round(output_hours * 3600))
    if total <= 0 or interval <= 0 or total % interval or interval % dt:
        raise ValueError("hours and output_hours must be whole positive multiples of the timestep")
    output_dir = Path(output_dir)
    if parallel:
        output_dir /= f"rank{partition.rank:05d}"
    start = _initial_date(arguments)
    _prepare_output(output_dir, start=start, total=total, interval=interval, parallel=parallel,
                    setup_group=exchanger.setup_group if parallel else None)
    sizes = {"nCells": int(arguments["n_cells_global"]), "nEdges": int(native["u"].shape[0]) - 1,
             "nVertLevels": int(native["theta_m"].shape[1])}
    if parallel:
        sizes.update(nCells=partition.cells.owned_count, nEdges=partition.edges.owned_count,
                     nCellsGlobal=partition.cells.global_count, nEdgesGlobal=partition.edges.global_count,
                     rank=partition.rank, world_size=dist.get_world_size())
    common = {"initial_time": start.isoformat(), "working_dtype": str(native["u"].dtype), **sizes}
    previous = {}
    try:
        with torch.no_grad():
            state = initialize_cold_cycle(native, mesh, config=config, **case["cold_arguments"]).state
            mark_cold_boundary(state.native_state, mesh=mesh, config=config, dt=float(dt))
            if owned:  # the initial fields and cold-start operands are not read again
                native = case = None
            previous = operational_profile(state, duration_seconds=total, arguments=arguments,
                                           scheme_compile=scheme_compile)
            runtime_policy.set_step_graphs(step_graphs)
            # One runner: each step variant records once per forecast. The cold state's own tensors are handed
            # over; those sharing storage with what the driver still holds (the caller's case, mesh,
            # configuration and cycle arguments) are copied before their first write. The driver keeps no
            # reference to the cold state while the runner advances it (``handed``), so the tensors its first
            # step replaces are freed before any graph is recorded.
            runner = None
            if step_graphs:
                held = _storages((case, native, mesh, config, arguments))
                runner = StepGraphRunner(donated=[v for v in flatten_state(state)[0].values()
                                                  if isinstance(v, torch.Tensor)
                                                  and v.untyped_storage().data_ptr() not in held], **arguments)
            native_phases = (total > dt and runner is None and native_boundary_supported(config, device))
            handed, state = [state], None
            writer = StreamWriter(target)
            try:
                def emit(state):
                    elapsed = (state.step_index - 1) * dt
                    valid = start + dt_.timedelta(seconds=elapsed)
                    attributes = {**common, "valid_time": valid.isoformat(), "forecast_seconds": elapsed}
                    stamp = valid.strftime("%Y-%m-%d_%H.%M.%S")
                    for stream, fields in (("history", history_fields(state)), ("diag", diagnostic_fields(state))):
                        writer.submit(output_dir / f"{stream}.{stamp}.nc", output_fields(fields, order,
                                      n_cells=sizes["nCells"], n_edges=sizes["nEdges"]), attributes=attributes)

                if native_phases:
                    initial = handed.pop()
                    phase = start_native_coupled_phase(initial, **arguments)
                    passive_outputs = _native_coupled_output_fields(initial, phase, mesh=mesh)
                    initial = None
                    for elapsed in range(dt, total - dt, dt):
                        phase = advance_native_coupled_phase(phase,
                            on_completed_cycle=emit if elapsed % interval == 0 else None,
                            _output_fields=passive_outputs, **arguments)
                    state = cycle._run_coupled_operation(phase,
                        {**arguments, "_output_fields": passive_outputs,
                         "_on_completed_cycle": emit if (total - dt) % interval == 0 else None},
                        on_overflow="rerun", operator=cycle._finish_native_coupled_interval,
                        state_of=cycle._completed_cycle_state).state
                    phase = None
                    emit(state)
                else:
                    for elapsed in range(interval, total + 1, interval):
                        if runner is not None:
                            state = runner.advance(handed.pop() if handed else state, duration_seconds=interval)
                        else:
                            state = run_coupled_interval(handed.pop() if handed else state, duration_seconds=interval,
                                                         on_overflow="rerun", **arguments).state
                        emit(state)
            except BaseException as error:
                writer.close(raise_error=False)
                if writer.error is not None and writer.error is not error.__cause__:
                    error.add_note(f"the forecast output writer also failed: {writer.error!r}")
                raise
            else:
                writer.close()
    finally:
        if previous:
            runtime_policy.set_cycle_observers(previous["cycle_observers"])
            runtime_policy.set_step_graphs(previous["step_graphs"])
            runtime_policy.set_sw_daylit_fraction(previous["sw_daylit_fraction"])
            runtime_policy.set_scheme_compile(previous["scheme_compile"])
            runtime_policy.set_column_chunk(previous["column_chunk"])
    return state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case")
    parser.add_argument("output_dir")
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--output-hours", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("f64", "f32", "float64", "float32"), default="f64")
    parser.add_argument("--distributed-timeout", type=int, default=300, help="process-group timeout in seconds")
    parser.add_argument("--partition", type=Path, help="MPAS graph.info.part.N cell owners (default: spatial bisection)")
    parser.add_argument("--owned-order", choices=("global", "hilbert"), default="hilbert",
                        help="order of each rank's owned rows in memory (hilbert: along a Hilbert curve); output files "
                             "hold the owned rows in ascending global ID either way")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--scheme-compile", action="store_true")
    parser.add_argument("--dycore-compile", action="store_true")
    parser.add_argument("--compile", action="store_true", help="compile both physics and dycore regions")
    parser.add_argument("--no-step-graphs", dest="step_graphs", action="store_false",
                        help="run ordinary intervals without CUDA graph capture")
    options = parser.parse_args(argv)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if options.threads:
        torch.set_num_threads(options.threads)
    from .distributed import forecast_processes

    with forecast_processes(options.device, timeout_seconds=options.distributed_timeout) as (device, setup_group):
        run_forecast(options.case, options.output_dir, hours=options.hours, output_hours=options.output_hours,
                     device=device, dtype=options.dtype, setup_group=setup_group,
                     scheme_compile=options.scheme_compile or options.compile,
                     dycore_compile=options.dycore_compile or options.compile, step_graphs=options.step_graphs,
                     partition_path=options.partition, owned_order=options.owned_order)


if __name__ == "__main__":
    main()
