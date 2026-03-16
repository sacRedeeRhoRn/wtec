from __future__ import annotations

import argparse
import json
import shlex
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from wtec.cluster.pbs import PBSJobConfig, generate_script
from wtec.cluster.ssh import open_ssh
from wtec.cluster.submit import JobManager
from wtec.config.cluster import ClusterConfig
from wtec.transport.kwant_sigma_extract import extract_kwant_sigmas
from wtec.transport.rgf_postprocess import load_rgf_raw_result


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_DIR = PROJECT_ROOT / "tmp" / "model_a_single_t5_target"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "nanowire_benchmark" / "mp-1018028" / "model_a_single_t5_rgf"
DEFAULT_WORK_ROOT = PROJECT_ROOT / "tmp" / "model_a_single_t5_debug"
DEFAULT_WALLTIME = "30:00:00"
TAG_ORDER = ("m0p2", "m0p1", "p0p0", "p0p1", "p0p2")


@dataclass(frozen=True)
class TagSpec:
    tag: str
    payload_template_path: Path
    energy_rel_fermi_ev: float
    energy_abs_ev: float
    target_transmission_e2_over_h: float
    width_uc: int
    thickness_uc: int
    length_uc: int
    fermi_ev: float
    eta_ev: float
    lead_axis: str
    thickness_axis: str
    periodic_axis: str


@dataclass(frozen=True)
class PreparedHarness:
    work_root: Path
    target_dir: Path
    source_dir: Path
    payload_dir: Path
    sigma_dir: Path
    stage_dir: Path
    retrieved_dir: Path
    runs_dir: Path
    tags: tuple[TagSpec, ...]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def _tag_sort_key(tag: str) -> tuple[int, str]:
    try:
        return (TAG_ORDER.index(tag), tag)
    except ValueError:
        return (len(TAG_ORDER), tag)


def _flatten_value(value: Any) -> float:
    if isinstance(value, list):
        if not value:
            raise RuntimeError("Expected non-empty transport value.")
        return _flatten_value(value[0])
    return float(value)


def _copy_if_exists(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _find_target_row(results: list[dict[str, Any]], energy_rel: float) -> dict[str, Any]:
    for row in results:
        row_energy = float(row.get("energy_rel_fermi_ev", 0.0))
        if abs(row_energy - float(energy_rel)) <= 1.0e-12:
            return row
    raise RuntimeError(f"No Kwant target row found for energy_rel_fermi_ev={energy_rel}")


def build_tag_specs(
    *,
    target_dir: Path = DEFAULT_TARGET_DIR,
    source_dir: Path = DEFAULT_SOURCE_DIR,
) -> tuple[TagSpec, ...]:
    target_ref = _read_json(target_dir / "kwant_reference.json")
    results = target_ref.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError("Target kwant_reference.json is missing result rows.")

    specs: list[TagSpec] = []
    for template_path in sorted(source_dir.glob("payload_*.json"), key=lambda p: _tag_sort_key(p.stem.replace("payload_", "", 1))):
        payload = _read_json(template_path)
        tag = template_path.stem.replace("payload_", "", 1)
        row = _find_target_row(results, float(payload["energy"]))
        thicknesses = payload.get("thicknesses")
        if not isinstance(thicknesses, list) or len(thicknesses) != 1:
            raise RuntimeError(f"Expected one thickness in {template_path}")
        specs.append(
            TagSpec(
                tag=tag,
                payload_template_path=template_path,
                energy_rel_fermi_ev=float(payload["energy"]),
                energy_abs_ev=float(row["energy_abs_ev"]),
                target_transmission_e2_over_h=float(row["transmission_e2_over_h"]),
                width_uc=int(payload["n_layers_y"]),
                thickness_uc=int(thicknesses[0]),
                length_uc=int(payload["n_layers_x"]),
                fermi_ev=float(target_ref["fermi_ev"]),
                eta_ev=float(payload["eta"]),
                lead_axis=str(payload.get("lead_axis", "x")),
                thickness_axis=str(payload.get("thickness_axis", "z")),
                periodic_axis=str(payload.get("transport_rgf_periodic_axis", "y")),
            )
        )
    if not specs:
        raise RuntimeError(f"No payload templates found under {source_dir}")
    return tuple(sorted(specs, key=lambda item: _tag_sort_key(item.tag)))


def _sigma_stage_name(tag: str, side: str) -> str:
    return f"sigma_{side}_{tag}.bin"


def _sigma_manifest_stage_name(tag: str) -> str:
    return f"sigma_manifest_{tag}.json"


def _prepare_payload_for_stage(spec: TagSpec, *, payload: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(payload)
    prepared["sigma_left_path"] = _sigma_stage_name(spec.tag, "left")
    prepared["sigma_right_path"] = _sigma_stage_name(spec.tag, "right")
    prepared["progress_file"] = f"progress_{spec.tag}.jsonl"
    return prepared


def _write_target_spec(prepared: PreparedHarness) -> None:
    payload = {
        "target_dir": str(prepared.target_dir),
        "source_dir": str(prepared.source_dir),
        "rows": [asdict(spec) | {"payload_template_path": str(spec.payload_template_path)} for spec in prepared.tags],
    }
    _write_json(prepared.work_root / "target_spec.json", payload)


def prepare_work_root(
    *,
    work_root: Path = DEFAULT_WORK_ROOT,
    target_dir: Path = DEFAULT_TARGET_DIR,
    source_dir: Path = DEFAULT_SOURCE_DIR,
) -> PreparedHarness:
    work_root = work_root.expanduser().resolve()
    target_copy_dir = work_root / "target"
    source_copy_dir = work_root / "source"
    payload_dir = work_root / "payloads"
    sigma_dir = work_root / "sigma"
    stage_dir = work_root / "stage"
    retrieved_dir = work_root / "retrieved"
    runs_dir = work_root / "runs"
    for path in (target_copy_dir, source_copy_dir, payload_dir, sigma_dir, stage_dir, retrieved_dir, runs_dir):
        path.mkdir(parents=True, exist_ok=True)

    for name in ("kwant_payload.json", "kwant_reference.json", "target_summary.json"):
        _copy_if_exists(target_dir / name, target_copy_dir / name)
    for path in sorted(source_dir.glob("*")):
        if path.is_file():
            _copy_if_exists(path, source_copy_dir / path.name)

    specs = build_tag_specs(target_dir=target_copy_dir, source_dir=source_copy_dir)
    hr_path = source_copy_dir / "TiS_model_a_c_single_t5_c_canonical_hr.dat"
    for spec in specs:
        template_payload = _read_json(spec.payload_template_path)
        sigma_out_dir = sigma_dir / spec.tag
        sigma_out_dir.mkdir(parents=True, exist_ok=True)
        extract_kwant_sigmas(
            hr_path=hr_path,
            length_uc=spec.length_uc,
            width_uc=spec.width_uc,
            thickness_uc=spec.thickness_uc,
            energy_ev=spec.energy_abs_ev,
            eta_ev=spec.eta_ev,
            out_dir=sigma_out_dir,
            layout="full_finite_principal",
        )
        _copy_if_exists(sigma_out_dir / "sigma_left.bin", stage_dir / _sigma_stage_name(spec.tag, "left"))
        _copy_if_exists(sigma_out_dir / "sigma_right.bin", stage_dir / _sigma_stage_name(spec.tag, "right"))
        _copy_if_exists(sigma_out_dir / "sigma_manifest.json", stage_dir / _sigma_manifest_stage_name(spec.tag))
        rewritten = _prepare_payload_for_stage(spec, payload=template_payload)
        _write_json(payload_dir / f"payload_{spec.tag}.json", rewritten)

    prepared = PreparedHarness(
        work_root=work_root,
        target_dir=target_copy_dir,
        source_dir=source_copy_dir,
        payload_dir=payload_dir,
        sigma_dir=sigma_dir,
        stage_dir=stage_dir,
        retrieved_dir=retrieved_dir,
        runs_dir=runs_dir,
        tags=specs,
    )
    _write_target_spec(prepared)
    return prepared


def _load_prepared_harness(work_root: Path = DEFAULT_WORK_ROOT) -> PreparedHarness:
    work_root = work_root.expanduser().resolve()
    target_copy_dir = work_root / "target"
    source_copy_dir = work_root / "source"
    payload_dir = work_root / "payloads"
    sigma_dir = work_root / "sigma"
    stage_dir = work_root / "stage"
    retrieved_dir = work_root / "retrieved"
    runs_dir = work_root / "runs"
    specs = build_tag_specs(target_dir=target_copy_dir, source_dir=source_copy_dir)
    return PreparedHarness(
        work_root=work_root,
        target_dir=target_copy_dir,
        source_dir=source_copy_dir,
        payload_dir=payload_dir,
        sigma_dir=sigma_dir,
        stage_dir=stage_dir,
        retrieved_dir=retrieved_dir,
        runs_dir=runs_dir,
        tags=specs,
    )


def _load_rgf_router_state(state_path: Path | None = None) -> dict[str, Any]:
    state_file = state_path or (PROJECT_ROOT / ".wtec" / "init_state.json")
    payload = _read_json(state_file)
    cluster = payload.get("rgf", {}).get("cluster")
    if not isinstance(cluster, dict):
        cluster = payload.get("solver_capabilities", {}).get("cluster", {}).get("rgf", {})
    if not isinstance(cluster, dict):
        raise RuntimeError(f"RGF cluster router state missing in {state_file}")
    return cluster


def _build_remote_script(
    *,
    prepared: PreparedHarness,
    binary_path: str,
    remote_dir: str,
    queue: str,
    walltime: str,
    modules: list[str] | None = None,
    job_name: str = "rgf_t5_debug",
) -> str:
    commands: list[str] = []
    for spec in prepared.tags:
        raw_name = f"raw_{spec.tag}.json"
        commands.append(
            f'echo "[rgf-onejob] start {spec.tag} $(date -Is)" && '
            f"mpirun -np 1 --bind-to none {shlex.quote(binary_path)} "
            f"{shlex.quote(f'payload_{spec.tag}.json')} {shlex.quote(raw_name)} && "
            f'echo "[rgf-onejob] done {spec.tag} $(date -Is)"'
        )

    cfg = PBSJobConfig(
        job_name=job_name,
        n_nodes=1,
        n_cores_per_node=64,
        mpi_procs_per_node=1,
        omp_threads=64,
        walltime=walltime,
        queue=queue,
        work_dir=remote_dir,
        stdout_path=f"{remote_dir}/rgf_t5_debug.log",
        runtime_log_path=f"{remote_dir}/wtec_job.log",
        modules=list(modules or []),
        env_vars={
            "OMP_NUM_THREADS": "64",
            "MKL_NUM_THREADS": "64",
            "OPENBLAS_NUM_THREADS": "64",
            "NUMEXPR_NUM_THREADS": "64",
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "spread",
            "OMP_PLACES": "cores",
        },
    )
    return generate_script(cfg, commands)


def _stage_files_for_remote(prepared: PreparedHarness) -> list[Path]:
    files = [
        prepared.source_dir / "TiS_model_a_c_single_t5_c_canonical_hr.dat",
        prepared.source_dir / "TiS_model_a_c_single_t5_c_canonical.win",
    ]
    files.extend(sorted(prepared.payload_dir.glob("payload_*.json"), key=lambda p: _tag_sort_key(p.stem.replace("payload_", "", 1))))
    files.extend(sorted(prepared.stage_dir.glob("sigma_*.bin")))
    return files


def _organize_retrieved_outputs(prepared: PreparedHarness) -> None:
    for spec in prepared.tags:
        run_dir = prepared.runs_dir / spec.tag
        run_dir.mkdir(parents=True, exist_ok=True)
        raw_src = prepared.retrieved_dir / f"raw_{spec.tag}.json"
        if raw_src.exists():
            _copy_if_exists(raw_src, run_dir / "raw.json")
        progress_src = prepared.retrieved_dir / f"progress_{spec.tag}.jsonl"
        if progress_src.exists():
            _copy_if_exists(progress_src, run_dir / "progress.jsonl")
    for name in ("wtec_job.log", "rgf_t5_debug.log"):
        src = prepared.retrieved_dir / name
        if src.exists():
            _copy_if_exists(src, prepared.work_root / name)


def run_rgf(
    *,
    work_root: Path = DEFAULT_WORK_ROOT,
    queue: str | None = None,
    walltime: str = DEFAULT_WALLTIME,
    poll_interval: int = 30,
) -> dict[str, Any]:
    prepared = _load_prepared_harness(work_root)
    router = _load_rgf_router_state()
    binary_path = str(router.get("binary_path") or "").strip()
    if not binary_path:
        raise RuntimeError("RGF cluster router state is missing binary_path. Re-run `wtec init`.")

    cluster_cfg = ClusterConfig.from_env()
    remote_stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    remote_dir = f"{cluster_cfg.remote_workdir.rstrip('/')}/model_a_single_t5_debug_{remote_stamp}"
    prepared.retrieved_dir.mkdir(parents=True, exist_ok=True)

    with open_ssh(cluster_cfg) as ssh:
        jm = JobManager(ssh)
        queue_used = jm.resolve_queue(queue or cluster_cfg.pbs_queue, fallback_order=cluster_cfg.pbs_queue_priority)
        jm.ensure_remote_commands(["qsub", "qstat", "mpirun"], modules=cluster_cfg.modules, bin_dirs=cluster_cfg.bin_dirs)
        ssh.run(f"test -x {shlex.quote(binary_path)}")
        script = _build_remote_script(
            prepared=prepared,
            binary_path=binary_path,
            remote_dir=remote_dir,
            queue=queue_used,
            walltime=walltime,
            modules=cluster_cfg.modules,
        )
        meta = jm.submit_and_wait(
            script,
            remote_dir,
            prepared.retrieved_dir,
            ["raw_*.json", "progress_*.jsonl", "wtec_job.log", "*.log"],
            script_name="rgf_reference.pbs",
            stage_files=_stage_files_for_remote(prepared),
            expected_local_outputs=[f"raw_{spec.tag}.json" for spec in prepared.tags],
            queue_used=queue_used,
            poll_interval=poll_interval,
            verbose=True,
            live_log=True,
            live_files=["wtec_job.log"],
            live_retrieve_patterns=["raw_*.json", "progress_*.jsonl", "wtec_job.log"],
            live_retrieve_interval_seconds=max(10, int(poll_interval)),
            stream_from_start=True,
        )

    _organize_retrieved_outputs(prepared)
    run_meta = {
        "job": meta,
        "remote_dir": remote_dir,
        "binary_path": binary_path,
        "queue": meta.get("queue"),
        "binary_id": router.get("binary_id"),
        "numerical_status": router.get("numerical_status"),
    }
    _write_json(prepared.work_root / "run_meta.json", run_meta)
    return run_meta


def _candidate_kwant_log_paths(work_root: Path, explicit: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.append(work_root / "target" / "wtec_job.log")
    candidates.append(PROJECT_ROOT / "nanowire_benchmark" / "mp-1018028" / "model_a_single_t5_kwant" / "wtec_job.log")
    return candidates


def _parse_wall_seconds_from_log(path: Path) -> float | None:
    if not path.exists():
        return None
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    saw_done = False
    for line in path.read_text().splitlines():
        if "[wtec][runtime] start " in line and start_ts is None:
            stamp = line.rsplit(" ", 1)[-1].strip()
            try:
                start_ts = datetime.fromisoformat(stamp)
            except ValueError:
                continue
        if " done " in line:
            saw_done = True
        stamp_candidate = line.rsplit(" ", 1)[-1].strip()
        try:
            end_ts = datetime.fromisoformat(stamp_candidate)
        except ValueError:
            continue
    if start_ts is None or end_ts is None or not saw_done:
        return None
    return max(0.0, (end_ts - start_ts).total_seconds())


def compare_results(
    *,
    work_root: Path = DEFAULT_WORK_ROOT,
    kwant_log_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prepared = _load_prepared_harness(work_root)
    rows: list[dict[str, Any]] = []
    total_rgf_wall = 0.0
    for spec in prepared.tags:
        raw_path = prepared.runs_dir / spec.tag / "raw.json"
        if not raw_path.exists():
            raise RuntimeError(f"Missing RGF output for {spec.tag}: {raw_path}")
        raw, runtime_cert = load_rgf_raw_result(raw_path)
        transmission = _flatten_value(raw.get("thickness_G"))
        abs_delta = abs(transmission - spec.target_transmission_e2_over_h)
        rel_delta = abs_delta / max(abs(spec.target_transmission_e2_over_h), 1.0e-12)
        wall_seconds = float(runtime_cert.get("wall_seconds", 0.0) or 0.0)
        total_rgf_wall += wall_seconds
        rows.append(
            {
                "tag": spec.tag,
                "energy_rel_fermi_ev": spec.energy_rel_fermi_ev,
                "energy_abs_ev": spec.energy_abs_ev,
                "target_transmission_e2_over_h": spec.target_transmission_e2_over_h,
                "rgf_transmission_e2_over_h": transmission,
                "abs_delta": abs_delta,
                "rel_delta": rel_delta,
                "raw_path": str(raw_path),
                "progress_path": str(prepared.runs_dir / spec.tag / "progress.jsonl"),
                "sigma_manifest_path": str(prepared.sigma_dir / spec.tag / "sigma_manifest.json"),
                "sigma_left_path": str(prepared.sigma_dir / spec.tag / "sigma_left.bin"),
                "sigma_right_path": str(prepared.sigma_dir / spec.tag / "sigma_right.bin"),
                "runtime_cert": {
                    "wall_seconds": wall_seconds,
                    "effective_thread_count": float(runtime_cert.get("effective_thread_count", 0.0) or 0.0),
                    "omp_threads": int(runtime_cert.get("omp_threads", 0) or 0),
                    "binary_id": str(runtime_cert.get("binary_id", "")),
                    "numerical_status": str(runtime_cert.get("numerical_status", "")),
                    "queue": str(runtime_cert.get("queue", "")),
                },
            }
        )

    comparison = {
        "target": {
            "model_key": "model_a",
            "width_uc": 5,
            "thickness_uc": 5,
            "length_uc": 24,
            "target_reference_path": str(prepared.target_dir / "kwant_reference.json"),
        },
        "rows": rows,
        "max_abs_delta": max((float(row["abs_delta"]) for row in rows), default=0.0),
        "max_rel_delta": max((float(row["rel_delta"]) for row in rows), default=0.0),
    }
    _write_json(prepared.work_root / "comparison.json", comparison)

    kwant_wall_seconds: float | None = None
    kwant_wall_source = "unavailable"
    for candidate in _candidate_kwant_log_paths(prepared.work_root, explicit=kwant_log_path):
        parsed = _parse_wall_seconds_from_log(candidate)
        if parsed is not None:
            kwant_wall_seconds = parsed
            kwant_wall_source = str(candidate)
            break

    speed_summary = {
        "target": {
            "model_key": "model_a",
            "width_uc": 5,
            "thickness_uc": 5,
            "length_uc": 24,
        },
        "rows": [
            {
                "tag": row["tag"],
                "energy_rel_fermi_ev": row["energy_rel_fermi_ev"],
                "rgf_wall_seconds": row["runtime_cert"]["wall_seconds"],
                "effective_thread_count": row["runtime_cert"]["effective_thread_count"],
                "omp_threads": row["runtime_cert"]["omp_threads"],
                "raw_path": row["raw_path"],
            }
            for row in rows
        ],
        "rgf_total_wall_seconds": total_rgf_wall,
        "kwant_wall_seconds": kwant_wall_seconds,
        "kwant_wall_seconds_source": kwant_wall_source,
        "speedup_vs_kwant": (
            (float(kwant_wall_seconds) / total_rgf_wall)
            if kwant_wall_seconds is not None and total_rgf_wall > 0.0
            else None
        ),
    }
    _write_json(prepared.work_root / "speed_summary.json", speed_summary)
    return comparison, speed_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare and run the temporary model_a single-thickness RGF target harness.")
    parser.add_argument("--mode", choices=("prepare", "run-rgf", "compare", "all"), default="all")
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    parser.add_argument("--queue", default=None)
    parser.add_argument("--walltime", default=DEFAULT_WALLTIME)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--kwant-log-path", default=None)
    args = parser.parse_args(argv)

    work_root = Path(args.work_root).expanduser().resolve()
    kwant_log_path = Path(args.kwant_log_path).expanduser().resolve() if args.kwant_log_path else None

    if args.mode in {"prepare", "all"}:
        prepare_work_root(work_root=work_root)
    if args.mode in {"run-rgf", "all"}:
        run_rgf(work_root=work_root, queue=args.queue, walltime=args.walltime, poll_interval=args.poll_interval)
    if args.mode in {"compare", "all"}:
        compare_results(work_root=work_root, kwant_log_path=kwant_log_path)


if __name__ == "__main__":
    main()
