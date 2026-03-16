from __future__ import annotations

import json
from pathlib import Path

from wtec.transport import model_a_single_t5_target as harness


def _fake_extract_kwant_sigmas(**kwargs):
    out_dir = Path(kwargs["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sigma_left.bin").write_bytes(b"left")
    (out_dir / "sigma_right.bin").write_bytes(b"right")
    payload = {
        "layout": kwargs["layout"],
        "energy_ev": kwargs["energy_ev"],
        "sigma_left_path": str(out_dir / "sigma_left.bin"),
        "sigma_right_path": str(out_dir / "sigma_right.bin"),
    }
    (out_dir / "sigma_manifest.json").write_text(json.dumps(payload, indent=2))
    return payload


def test_build_tag_specs_matches_old_target_snapshot() -> None:
    specs = harness.build_tag_specs()
    assert [spec.tag for spec in specs] == list(harness.TAG_ORDER)
    assert [spec.energy_rel_fermi_ev for spec in specs] == [-0.2, -0.1, 0.0, 0.1, 0.2]
    assert [spec.target_transmission_e2_over_h for spec in specs] == [
        10.00000000000215,
        12.000000000000258,
        12.00000000000041,
        10.000000000000862,
        5.999999999996259,
    ]


def test_prepare_work_root_rewrites_payloads_and_stages_sigma(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(harness, "extract_kwant_sigmas", _fake_extract_kwant_sigmas)

    prepared = harness.prepare_work_root(work_root=tmp_path)

    assert (prepared.target_dir / "kwant_reference.json").exists()
    assert (prepared.source_dir / "TiS_model_a_c_single_t5_c_canonical_hr.dat").exists()
    payload = json.loads((prepared.payload_dir / "payload_m0p2.json").read_text())
    assert payload["sigma_left_path"] == "sigma_left_m0p2.bin"
    assert payload["sigma_right_path"] == "sigma_right_m0p2.bin"
    assert payload["progress_file"] == "progress_m0p2.jsonl"
    assert (prepared.sigma_dir / "m0p2" / "sigma_manifest.json").exists()
    assert (prepared.stage_dir / "sigma_left_m0p2.bin").exists()
    spec_rows = json.loads((prepared.work_root / "target_spec.json").read_text())["rows"]
    assert [row["tag"] for row in spec_rows] == list(harness.TAG_ORDER)


def test_compare_results_writes_comparison_and_speed_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(harness, "extract_kwant_sigmas", _fake_extract_kwant_sigmas)
    prepared = harness.prepare_work_root(work_root=tmp_path)

    for idx, spec in enumerate(prepared.tags, start=1):
        run_dir = prepared.runs_dir / spec.tag
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "transport_results_raw": {
                "mode": "full_finite",
                "thickness_G": [spec.target_transmission_e2_over_h - 0.5],
            },
            "runtime_cert": {
                "wall_seconds": float(idx),
                "effective_thread_count": 20.0 + idx,
                "omp_threads": 64,
                "binary_id": "wtec_rgf_runner_phase2_v6",
                "queue": "g4",
            },
        }
        (run_dir / "raw.json").write_text(json.dumps(payload, indent=2))
        (run_dir / "progress.jsonl").write_text('{"event":"worker_done"}\n')

    comparison, speed = harness.compare_results(work_root=tmp_path)

    assert comparison["max_abs_delta"] == 0.5
    assert len(comparison["rows"]) == 5
    assert comparison["rows"][0]["sigma_manifest_path"].endswith("sigma/m0p2/sigma_manifest.json")
    assert speed["rgf_total_wall_seconds"] == 15.0
    assert speed["kwant_wall_seconds"] is None
    assert speed["kwant_wall_seconds_source"] == "unavailable"
    assert json.loads((tmp_path / "comparison.json").read_text())["max_abs_delta"] == 0.5
    assert json.loads((tmp_path / "speed_summary.json").read_text())["rgf_total_wall_seconds"] == 15.0


def test_build_remote_script_uses_remote_workdir_and_sequential_tags(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(harness, "extract_kwant_sigmas", _fake_extract_kwant_sigmas)
    prepared = harness.prepare_work_root(work_root=tmp_path)

    script = harness._build_remote_script(
        prepared=prepared,
        binary_path="/remote/wtec_rgf_runner",
        remote_dir="/remote/work/model_a_single_t5_debug",
        queue="g4",
        walltime="30:00:00",
        modules=["QE/7.2"],
    )

    assert "cd /remote/work/model_a_single_t5_debug" in script
    assert "#PBS -q g4" in script
    assert "module load QE/7.2" in script
    assert "payload_m0p2.json raw_m0p2.json" in script
    assert "payload_p0p2.json raw_p0p2.json" in script


def test_run_rgf_uses_cluster_helpers_and_organizes_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(harness, "extract_kwant_sigmas", _fake_extract_kwant_sigmas)
    prepared = harness.prepare_work_root(work_root=tmp_path)

    calls: dict[str, object] = {}

    class _FakeClusterConfig:
        remote_workdir = "/remote/base"
        pbs_queue = "g4"
        pbs_queue_priority = ["g4"]
        modules: list[str] = []
        bin_dirs: list[str] = []

    class _FakeSSH:
        def run(self, command: str, check: bool = True):
            calls.setdefault("ssh_commands", []).append(command)
            return (0, "", "")

        def close(self):
            return None

    class _FakeOpenSSH:
        def __enter__(self):
            return _FakeSSH()

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeJobManager:
        def __init__(self, ssh):
            self._ssh = ssh

        def resolve_queue(self, preferred, fallback_order=None):
            calls["resolved_queue"] = preferred
            return preferred

        def ensure_remote_commands(self, commands, modules=None, bin_dirs=None):
            calls["required_commands"] = list(commands)

        def submit_and_wait(
            self,
            script_content,
            remote_dir,
            local_dir,
            retrieve_patterns,
            *,
            script_name="job.pbs",
            stage_files=None,
            expected_local_outputs=None,
            queue_used=None,
            **kwargs,
        ):
            calls["script_content"] = script_content
            calls["remote_dir"] = remote_dir
            calls["script_name"] = script_name
            calls["stage_files"] = [Path(path).name for path in stage_files or []]
            calls["retrieve_patterns"] = list(retrieve_patterns)
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            for expected in expected_local_outputs or []:
                tag = str(expected).replace("raw_", "").replace(".json", "")
                payload = {
                    "transport_results_raw": {"mode": "full_finite", "thickness_G": [1.0]},
                    "runtime_cert": {"wall_seconds": 2.0, "omp_threads": 64},
                }
                (local_dir / expected).write_text(json.dumps(payload, indent=2))
                (local_dir / f"progress_{tag}.jsonl").write_text('{"event":"worker_done"}\n')
            (local_dir / "wtec_job.log").write_text("log\n")
            return {
                "job_id": "777",
                "queue": queue_used,
                "remote_dir": remote_dir,
                "remote_script": f"{remote_dir}/{script_name}",
                "status": "COMPLETED",
            }

    monkeypatch.setattr(harness.ClusterConfig, "from_env", classmethod(lambda cls: _FakeClusterConfig()))
    monkeypatch.setattr(harness, "_load_rgf_router_state", lambda state_path=None: {
        "binary_path": "/remote/wtec_rgf_runner",
        "binary_id": "wtec_rgf_runner_phase2_v6",
        "numerical_status": "phase2_experimental",
    })
    monkeypatch.setattr(harness, "open_ssh", lambda cfg: _FakeOpenSSH())
    monkeypatch.setattr(harness, "JobManager", _FakeJobManager)

    meta = harness.run_rgf(work_root=tmp_path, queue="g4", walltime="30:00:00", poll_interval=5)

    assert meta["job"]["job_id"] == "777"
    assert "payload_m0p2.json" in calls["stage_files"]
    assert "sigma_left_m0p2.bin" in calls["stage_files"]
    assert "cd /remote/base/model_a_single_t5_debug_" in calls["script_content"]
    assert (prepared.runs_dir / "m0p2" / "raw.json").exists()
    assert (prepared.runs_dir / "p0p2" / "progress.jsonl").exists()
    assert json.loads((prepared.work_root / "run_meta.json").read_text())["job"]["job_id"] == "777"
    assert any(cmd == "test -x /remote/wtec_rgf_runner" for cmd in calls["ssh_commands"])


def test_parse_wall_seconds_from_completed_log(tmp_path: Path) -> None:
    log_path = tmp_path / "wtec_job.log"
    log_path.write_text(
        "\n".join(
            [
                "[wtec][runtime] start 2026-03-13T12:00:00+09:00",
                "[kwant-bench][rank=0] done thickness_uc=5 energy_abs_ev=13.404600 elapsed_s=10.0 2026-03-13T12:30:00+09:00",
            ]
        )
        + "\n"
    )
    assert harness._parse_wall_seconds_from_log(log_path) == 1800.0
