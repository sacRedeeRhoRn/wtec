from __future__ import annotations

from concurrent.futures import CancelledError
import json
import math
import os
from pathlib import Path
from typing import Any

from wtec.cluster.mpi import MPIConfig, build_command
from wtec.cluster.pbs import PBSJobConfig, generate_script
from wtec.cluster.ssh import open_ssh
from wtec.cluster.submit import JobManager
from wtec.config.cluster import ClusterConfig
from wtec.transport.nanowire_benchmark import CanonicalizedNanowireInput, NanowireBenchmarkSpec
from wtec.transport.kwant_nanowire_benchmark import (
    kwant_reference_checkpoint_payload,
    kwant_reference_is_complete,
    kwant_reference_progress,
)
from wtec.workflow.orchestrator import TopoSlabWorkflow


def _kwant_worker_layout(*, total_cores: int, task_count: int, n_nodes: int) -> tuple[int, int]:
    """Choose a conservative MPI/OMP layout for independent Kwant reference points."""
    max_ranks = max(1, min(int(total_cores), int(task_count)))
    override_raw = os.environ.get("TOPOSLAB_KWANT_BENCH_MPI_RANKS", "").strip()
    if override_raw:
        try:
            override = int(override_raw)
        except ValueError:
            override = 0
        mpi_np = max(1, min(override, max_ranks)) if override > 0 else 1
    else:
        # Use a few worker ranks to exploit task-level parallelism without
        # flooding a node with too many concurrent MUMPS factorizations.
        mpi_np = max(1, min(max_ranks, max(1, int(total_cores) // 16), 4))
    if int(n_nodes) > 1 and mpi_np % int(n_nodes) != 0:
        mpi_np = max(int(n_nodes), (mpi_np // int(n_nodes)) * int(n_nodes))
        mpi_np = min(mpi_np, max_ranks)
    omp_threads = max(1, int(total_cores) // max(1, int(mpi_np)))
    return int(mpi_np), int(omp_threads)


def _parse_walltime_seconds(value: str) -> int:
    text = str(value or "").strip()
    parts = text.split(":")
    if len(parts) != 3 or any((not part.isdigit()) for part in parts):
        raise ValueError(f"walltime must use HH:MM:SS, got {value!r}")
    hours, minutes, seconds = (int(part) for part in parts)
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"walltime must use HH:MM:SS, got {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def _format_walltime_seconds(total_seconds: int) -> str:
    seconds = max(1, int(total_seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _resolve_kwant_reference_walltime(
    *,
    base_walltime: str,
    total_cores: int,
    task_count: int,
    n_nodes: int,
) -> str:
    override_raw = os.environ.get("TOPOSLAB_KWANT_BENCH_WALLTIME", "").strip()
    if override_raw:
        return _format_walltime_seconds(_parse_walltime_seconds(override_raw))

    base_seconds = _parse_walltime_seconds(base_walltime)
    mpi_np, _ = _kwant_worker_layout(
        total_cores=int(total_cores),
        task_count=int(task_count),
        n_nodes=int(n_nodes),
    )
    worker_waves = max(1, math.ceil(int(task_count) / max(1, int(mpi_np))))
    # Native-RGF gets one PBS allocation per benchmark point, while the Kwant
    # baseline batches many points into one job. Scale the shared walltime by
    # the number of worker waves so the Kwant reference gets an equivalent
    # per-wave budget instead of timing out mid-benchmark.
    return _format_walltime_seconds(max(base_seconds, worker_waves * base_seconds))


def submit_kwant_nanowire_reference(
    *,
    canonical_input: CanonicalizedNanowireInput,
    benchmark_dir: str | Path,
    spec: NanowireBenchmarkSpec,
    model_key: str,
    model_label: str,
    fermi_ev: float,
    length_uc: int,
    queue_override: str | None = None,
    n_nodes: int = 1,
    walltime: str = "01:00:00",
    python_executable: str = "python3",
    modules: list[str] | None = None,
    bin_dirs: list[str] | None = None,
    live_log: bool = True,
    poll_interval: int = 5,
    stale_log_seconds: int = 300,
    cancel_event=None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = ClusterConfig.from_env()
    benchmark_path = Path(benchmark_dir).expanduser().resolve()
    benchmark_path.mkdir(parents=True, exist_ok=True)
    payload_path = benchmark_path / "kwant_payload.json"
    result_path = benchmark_path / "kwant_reference.json"
    script_path = benchmark_path / "kwant_reference.pbs"
    worker_zip = TopoSlabWorkflow._worker_source_zip(benchmark_path)

    def _merge_local_checkpoint() -> None:
        if not result_path.exists() and not list(benchmark_path.glob(f"{result_path.stem}.rank*.jsonl")):
            return
        merged = kwant_reference_checkpoint_payload(
            result_path,
            thicknesses=spec.thicknesses_uc,
            energies_rel_fermi_ev=spec.energies_ev,
        )
        result_path.write_text(json.dumps(merged, indent=2))

    payload = {
        "mp_id": str(spec.mp_id),
        "material": str(spec.material),
        "model_key": str(model_key),
        "model_label": str(model_label),
        "axis": str(canonical_input.axis),
        "hr_dat_path": Path(canonical_input.hr_dat_path).name,
        "fermi_ev": float(fermi_ev),
        "energies_rel_fermi_ev": [float(v) for v in spec.energies_ev],
        "thicknesses": [int(v) for v in spec.thicknesses_uc],
        "width_uc": int(spec.fixed_width_uc),
        "length_uc": int(length_uc),
        "serial_validate_thicknesses": [
            int(spec.thicknesses_uc[0]),
            int(spec.thicknesses_uc[-1]),
        ],
    }
    payload_path.write_text(json.dumps(payload, indent=2))

    if cancel_event is not None and hasattr(cancel_event, "is_set") and cancel_event.is_set():
        raise CancelledError("Kwant reference launch cancelled before submission.")

    path_tail = "_".join(benchmark_path.parts[-3:])
    remote_dir = f"{cfg.remote_workdir.rstrip('/')}/nanowire_benchmark/{spec.mp_id}/{path_tail}"
    max_attempts_raw = os.environ.get("TOPOSLAB_KWANT_BENCH_MAX_ATTEMPTS", "").strip()
    try:
        max_attempts = max(1, int(max_attempts_raw)) if max_attempts_raw else 8
    except ValueError:
        max_attempts = 8
    with open_ssh(cfg) as ssh:
        jm = JobManager(ssh)
        queue_used = jm.resolve_queue(queue_override or cfg.pbs_queue, fallback_order=cfg.pbs_queue_priority)
        cores_per_node = cfg.cores_for_queue(queue_used)
        total_cores = max(1, int(n_nodes) * int(cores_per_node))
        task_count = max(1, len(payload["thicknesses"]) * len(payload["energies_rel_fermi_ev"]))
        mpi_np, omp_threads = _kwant_worker_layout(
            total_cores=total_cores,
            task_count=task_count,
            n_nodes=int(n_nodes),
        )
        resolved_walltime = _resolve_kwant_reference_walltime(
            base_walltime=str(walltime),
            total_cores=total_cores,
            task_count=task_count,
            n_nodes=int(n_nodes),
        )
        cmd = build_command(
            f"env PYTHONPATH=$PWD/{worker_zip.name}:$PYTHONPATH {python_executable}",
            mpi=MPIConfig(n_cores=mpi_np, bind_to="none"),
            extra_args=f"-m wtec.transport.kwant_nanowire_benchmark {payload_path.name} {result_path.name}",
        )
        heartbeat_interval = os.environ.get("TOPOSLAB_KWANT_BENCH_HEARTBEAT_SECONDS", "").strip()
        env_vars = {
            "OMP_NUM_THREADS": str(omp_threads),
            "MKL_NUM_THREADS": str(omp_threads),
            "OPENBLAS_NUM_THREADS": str(omp_threads),
            "NUMEXPR_NUM_THREADS": str(omp_threads),
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "spread",
            "OMP_PLACES": "cores",
        }
        if heartbeat_interval:
            env_vars["TOPOSLAB_KWANT_BENCH_HEARTBEAT_SECONDS"] = heartbeat_interval
        script = generate_script(
            PBSJobConfig(
                job_name=f"kw{canonical_input.axis}_w{spec.fixed_width_uc}_l{length_uc}"[:15],
                n_nodes=int(n_nodes),
                n_cores_per_node=int(cores_per_node),
                mpi_procs_per_node=max(1, int(mpi_np) // max(1, int(n_nodes))),
                omp_threads=int(omp_threads),
                walltime=resolved_walltime,
                queue=queue_used,
                work_dir=remote_dir,
                modules=modules or cfg.modules,
                env_vars=env_vars,
            ),
            [cmd],
        )
        script_path.write_text(script)
        attempt = 0
        while True:
            attempt += 1
            try:
                meta = jm.submit_and_wait(
                    script,
                    remote_dir=remote_dir,
                    local_dir=benchmark_path,
                    retrieve_patterns=[
                        result_path.name,
                        f"{result_path.stem}.rank*.jsonl",
                        "*.out",
                        "wtec_job.log",
                    ],
                    script_name=script_path.name,
                    stage_files=[payload_path, worker_zip, Path(canonical_input.hr_dat_path)],
                    expected_local_outputs=[result_path.name],
                    queue_used=queue_used,
                    poll_interval=int(poll_interval),
                    verbose=True,
                    live_log=bool(live_log),
                    live_files=[result_path.name, "wtec_job.log"],
                    live_retrieve_patterns=[
                        result_path.name,
                        f"{result_path.stem}.rank*.jsonl",
                        "wtec_job.log",
                    ],
                    live_retrieve_interval_seconds=int(max(5, poll_interval)),
                    live_retrieve_hook=_merge_local_checkpoint,
                    stale_log_seconds=int(stale_log_seconds),
                    retrieve_on_failure=True,
                    stream_from_start=True,
                    cancel_event=cancel_event,
                )
            except RuntimeError as exc:
                if cancel_event is not None and hasattr(cancel_event, "is_set") and cancel_event.is_set():
                    raise CancelledError("Kwant reference job cancelled.") from exc
                if result_path.exists() or list(benchmark_path.glob(f"{result_path.stem}.rank*.jsonl")):
                    partial = kwant_reference_checkpoint_payload(
                        result_path,
                        thicknesses=spec.thicknesses_uc,
                        energies_rel_fermi_ev=spec.energies_ev,
                    )
                    if not kwant_reference_is_complete(
                        partial,
                        thicknesses=spec.thicknesses_uc,
                        energies_rel_fermi_ev=spec.energies_ev,
                    ):
                        progress = kwant_reference_progress(
                            partial,
                            thicknesses=spec.thicknesses_uc,
                            energies_rel_fermi_ev=spec.energies_ev,
                        )
                        if attempt < max_attempts:
                            print(
                                "[kwant-bench] "
                                f"checkpointed {int(progress['completed'])}/{int(progress['expected'])} "
                                f"points after failed job attempt {attempt}; resubmitting",
                                flush=True,
                            )
                            continue
                raise
            result = kwant_reference_checkpoint_payload(
                result_path,
                thicknesses=spec.thicknesses_uc,
                energies_rel_fermi_ev=spec.energies_ev,
            )
            if kwant_reference_is_complete(
                result,
                thicknesses=spec.thicknesses_uc,
                energies_rel_fermi_ev=spec.energies_ev,
            ):
                return result, meta
            progress = kwant_reference_progress(
                result,
                thicknesses=spec.thicknesses_uc,
                energies_rel_fermi_ev=spec.energies_ev,
            )
            if attempt >= max_attempts:
                raise RuntimeError(
                    "Kwant reference remains incomplete after "
                    f"{attempt} attempt(s): {int(progress['completed'])}/{int(progress['expected'])} "
                    f"points in {result_path}."
                )
            print(
                "[kwant-bench] "
                f"checkpointed {int(progress['completed'])}/{int(progress['expected'])} "
                f"points after attempt {attempt}; resubmitting remaining sweep",
                flush=True,
            )
