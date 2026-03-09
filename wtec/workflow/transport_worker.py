"""MPI worker for transport stage execution via PBS/qsub.

This worker is launched through mpirun; no fork-based launchstyle is used.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from wtec.workflow.transport_pipeline import TransportPipeline


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _mpi_context() -> tuple[Any, int, int]:
    try:
        from mpi4py import MPI
    except Exception:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, int(comm.Get_rank()), int(comm.Get_size())


def _runtime_stats() -> dict[str, Any]:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    out: dict[str, Any] = {
        "cpu_user_s": float(ru.ru_utime),
        "cpu_sys_s": float(ru.ru_stime),
        "rss_kb": int(ru.ru_maxrss),
    }
    try:
        l1, l5, l15 = os.getloadavg()
        out.update(
            {
                "loadavg_1m": float(l1),
                "loadavg_5m": float(l5),
                "loadavg_15m": float(l15),
            }
        )
    except Exception:
        pass
    return out


def _kwant_solver_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "kwant_importable": False,
        "solver": "unknown",
        "mumps_available": False,
        "reason": None,
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import kwant  # noqa: F401
            import kwant.solvers.default as _default  # noqa: F401
        status["kwant_importable"] = True
    except Exception as exc:
        status["reason"] = f"kwant_import_failed:{type(exc).__name__}:{exc}"
        return status
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import kwant.solvers.mumps as _mumps  # noqa: F401
        status["solver"] = "mumps"
        status["mumps_available"] = True
    except Exception as exc:
        status["solver"] = "scipy_fallback"
        status["mumps_available"] = False
        status["reason"] = f"mumps_unavailable:{type(exc).__name__}:{exc}"
    return status


class _ProgressLogger:
    def __init__(self, path: Path | None, *, detail: str = "minimal") -> None:
        self.path = path
        self.detail = str(detail).strip().lower() or "minimal"
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def emit(self, event: str, **payload: Any) -> None:
        rec = {
            "ts": float(time.time()),
            "event": str(event),
            **{k: _jsonable(v) for k, v in payload.items()},
        }
        line = json.dumps(rec, ensure_ascii=True)
        with self._lock:
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            if self.detail == "per_ensemble" or event in {
                "worker_start",
                "worker_config",
                "heartbeat",
                "transport_run_start",
                "transport_run_done",
                "worker_done",
                "worker_failed",
            }:
                print(f"[progress] {line}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(
            "Usage: python -m wtec.workflow.transport_worker <payload.json> <output.json>",
            file=sys.stderr,
        )
        return 2

    payload_path = Path(args[0]).expanduser()
    output_path = Path(args[1]).expanduser()
    if not payload_path.exists():
        print(f"Payload not found: {payload_path}", file=sys.stderr)
        return 2

    payload = json.loads(payload_path.read_text())
    if not isinstance(payload, dict):
        print("Payload must be a JSON object.", file=sys.stderr)
        return 2

    progress_file_raw = payload.get("progress_file")
    progress_path: Path | None = None
    if isinstance(progress_file_raw, str) and progress_file_raw.strip():
        progress_path = Path(progress_file_raw.strip())
        if not progress_path.is_absolute():
            progress_path = Path.cwd() / progress_path
    log_detail = str(payload.get("logging_detail", "minimal")).strip().lower() or "minimal"
    heartbeat_seconds = max(5, int(payload.get("heartbeat_seconds", 20)))
    logger = _ProgressLogger(progress_path, detail=log_detail)

    comm, rank, size = _mpi_context()
    hostname = platform.node()
    pid = os.getpid()
    env_threads = os.environ.get("OMP_NUM_THREADS", "").strip() or None
    env_threads_i = int(env_threads) if (env_threads and env_threads.isdigit()) else None
    expected_mpi_np = int(payload.get("expected_mpi_np", 0) or 0)
    expected_threads = int(payload.get("expected_threads", 0) or 0)
    require_mumps = bool(payload.get("require_mumps", False))
    solver_status = _kwant_solver_status()

    logger.emit(
        "worker_start",
        rank=int(rank),
        size=int(size),
        pid=int(pid),
        host=hostname,
        omp_num_threads=env_threads_i,
        expected_mpi_np=(expected_mpi_np if expected_mpi_np > 0 else None),
        expected_threads=(expected_threads if expected_threads > 0 else None),
        solver=solver_status.get("solver"),
        mumps_available=bool(solver_status.get("mumps_available")),
    )

    if expected_mpi_np > 0 and size != expected_mpi_np:
        logger.emit(
            "worker_failed",
            reason=f"mpi_size_mismatch:expected={expected_mpi_np}:actual={size}",
        )
        print(
            f"Transport worker failed: expected MPI size {expected_mpi_np}, got {size}",
            file=sys.stderr,
        )
        return 1
    if expected_threads > 0 and env_threads_i is not None and env_threads_i != expected_threads:
        logger.emit(
            "worker_failed",
            reason=f"omp_threads_mismatch:expected={expected_threads}:actual={env_threads_i}",
        )
        print(
            f"Transport worker failed: expected OMP_NUM_THREADS={expected_threads}, got {env_threads_i}",
            file=sys.stderr,
        )
        return 1
    if require_mumps and not bool(solver_status.get("mumps_available")):
        logger.emit(
            "worker_failed",
            reason=f"mumps_required_but_unavailable:{solver_status.get('reason')}",
        )
        print(
            "Transport worker failed: MUMPS-enabled Kwant solver required but unavailable. "
            f"Reason: {solver_status.get('reason')}",
            file=sys.stderr,
        )
        return 1

    hr_dat = Path(str(payload.get("hr_dat_path", "")).strip())
    if not hr_dat.exists():
        print(f"hr_dat_path not found: {hr_dat}", file=sys.stderr)
        return 2

    win_raw = payload.get("win_path")
    win_path = None
    if isinstance(win_raw, str) and win_raw.strip():
        wp = Path(win_raw.strip())
        if wp.exists():
            win_path = wp

    try:
        stop_heartbeat = threading.Event()

        def _heartbeat_loop() -> None:
            while not stop_heartbeat.wait(heartbeat_seconds):
                logger.emit("heartbeat", rank=int(rank), size=int(size), **_runtime_stats())

        hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        hb_thread.start()

        tp = TransportPipeline(
            hr_dat_path=hr_dat,
            win_path=win_path,
            thicknesses=payload.get("thicknesses"),
            disorder_strengths=payload.get("disorder_strengths"),
            n_ensemble=int(payload.get("n_ensemble", 50)),
            energy=float(payload.get("energy", 0.0)),
            n_jobs=int(payload.get("n_jobs", 1)),
            mfp_n_layers_z=int(payload.get("mfp_n_layers_z", 10)),
            mfp_lengths=payload.get("mfp_lengths"),
            lead_onsite_eV=float(payload.get("lead_onsite_eV", 0.0)),
            base_seed=int(payload.get("base_seed", 0)),
            lead_axis=str(payload.get("lead_axis", "x")),
            thickness_axis=str(payload.get("thickness_axis", "z")),
            n_layers_x=int(payload.get("n_layers_x", 4)),
            n_layers_y=int(payload.get("n_layers_y", 4)),
            carrier_density_m3=payload.get("carrier_density_m3"),
            fermi_velocity_m_per_s=payload.get("fermi_velocity_m_per_s"),
            progress_callback=logger.emit,
            log_detail=log_detail,
            heartbeat_seconds=heartbeat_seconds,
            kwant_mode=str(payload.get("kwant_mode", "auto")),
            kwant_task_workers=int(payload.get("kwant_task_workers", 0)),
        )
        results = tp.run_full()
        stop_heartbeat.set()
        hb_thread.join(timeout=1.0)
    except Exception as exc:
        try:
            stop_heartbeat.set()  # type: ignore[name-defined]
        except Exception:
            pass
        logger.emit("worker_failed", reason=f"{type(exc).__name__}:{exc}", **_runtime_stats())
        print(f"Transport worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    runtime_cert = {
        "rank": int(rank),
        "size": int(size),
        "pid": int(pid),
        "host": hostname,
        "omp_num_threads": env_threads_i,
        "expected_mpi_np": (expected_mpi_np if expected_mpi_np > 0 else None),
        "expected_threads": (expected_threads if expected_threads > 0 else None),
        "solver": solver_status.get("solver"),
        "mumps_available": bool(solver_status.get("mumps_available")),
        "kwant_mode": str(payload.get("kwant_mode", "auto")),
        "kwant_task_workers": int(payload.get("kwant_task_workers", 0)),
        **_runtime_stats(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "transport_results": _jsonable(results),
                "runtime_cert": _jsonable(runtime_cert),
            },
            indent=2,
        )
    )
    cert_file_raw = payload.get("runtime_cert_file")
    if isinstance(cert_file_raw, str) and cert_file_raw.strip():
        cert_path = Path(cert_file_raw.strip())
        if not cert_path.is_absolute():
            cert_path = Path.cwd() / cert_path
        cert_path.write_text(json.dumps(_jsonable(runtime_cert), indent=2))
    logger.emit("worker_done", **runtime_cert)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
