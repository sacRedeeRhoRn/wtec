from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from wtec.transport.nanowire_benchmark import CanonicalizedNanowireInput, NanowireBenchmarkSpec
from wtec.transport import nanowire_benchmark_cluster as nbcluster


class _DummySSH:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeClusterConfig:
    remote_workdir = "/remote/work"
    pbs_queue = "g4"
    pbs_queue_priority = ["g4"]
    modules: list[str] = []

    @staticmethod
    def cores_for_queue(queue: str) -> int:
        assert queue == "g4"
        return 64


def _complete_kwant_reference_payload(
    *, spec: NanowireBenchmarkSpec, fermi_ev: float
) -> dict[str, object]:
    results: list[dict[str, float | int]] = []
    for thickness_uc in spec.thicknesses_uc:
        for energy_rel in spec.energies_ev:
            results.append(
                {
                    "thickness_uc": int(thickness_uc),
                    "energy_rel_fermi_ev": float(energy_rel),
                    "energy_abs_ev": float(fermi_ev + float(energy_rel)),
                    "transmission_e2_over_h": 10.0,
                }
            )
    return {
        "status": "ok",
        "task_count_expected": len(results),
        "task_count_completed": len(results),
        "results": results,
        "validation": {"status": "ok"},
    }


def test_submit_kwant_nanowire_reference_uses_conservative_multi_rank_layout(
    tmp_path: Path, monkeypatch
) -> None:
    spec = NanowireBenchmarkSpec()
    fermi_ev = 1.23
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    hr_path = benchmark_dir / "toy_hr.dat"
    hr_path.write_text("dummy")
    worker_zip = benchmark_dir / "wtec_src.zip"
    worker_zip.write_text("zip")

    seen: dict[str, object] = {}

    class _FakeJobManager:
        def __init__(self, ssh: object) -> None:
            seen["ssh"] = ssh

        def resolve_queue(self, queue: str, fallback_order: list[str] | None = None) -> str:
            seen["queue"] = queue
            return queue

        def submit_and_wait(self, script: str, **kwargs):
            seen["script"] = script
            seen["kwargs"] = kwargs
            result_path = benchmark_dir / "kwant_reference.json"
            result_path.write_text(
                json.dumps(_complete_kwant_reference_payload(spec=spec, fermi_ev=fermi_ev))
            )
            return {"status": "ok"}

    monkeypatch.setattr(nbcluster, "open_ssh", lambda cfg: _DummySSH())
    monkeypatch.setattr(nbcluster, "JobManager", _FakeJobManager)
    monkeypatch.setattr(nbcluster.ClusterConfig, "from_env", staticmethod(lambda: _FakeClusterConfig()))
    monkeypatch.setattr(
        nbcluster.TopoSlabWorkflow,
        "_worker_source_zip",
        staticmethod(lambda _: worker_zip),
    )

    canonical = CanonicalizedNanowireInput(
        axis="c",
        hr_dat_path=str(hr_path),
        win_path=str(benchmark_dir / "toy.win"),
        permutation=(2, 0, 1),
        lattice_vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    result, meta = nbcluster.submit_kwant_nanowire_reference(
        canonical_input=canonical,
        benchmark_dir=benchmark_dir,
        spec=spec,
        model_key="model_a",
        model_label="Model A",
        fermi_ev=fermi_ev,
        length_uc=24,
        queue_override="g4",
        python_executable="python3",
        live_log=False,
    )

    script = str(seen["script"])
    assert "#PBS -l select=1:ncpus=64:mpiprocs=16:ompthreads=4" in script
    assert "#PBS -l walltime=03:00:00" in script
    assert "export OMP_NUM_THREADS=4" in script
    assert "export MKL_NUM_THREADS=4" in script
    assert "export OPENBLAS_NUM_THREADS=4" in script
    assert "export NUMEXPR_NUM_THREADS=4" in script
    assert "mpirun -np 16 --bind-to none" in script
    assert seen["kwargs"]["live_retrieve_patterns"] == [
        "kwant_reference.json",
        "kwant_reference.rank*.jsonl",
        "wtec_job.log",
    ]
    assert seen["kwargs"]["live_retrieve_interval_seconds"] == 5
    assert callable(seen["kwargs"]["live_retrieve_hook"])
    assert result["validation"]["status"] == "ok"
    assert meta["status"] == "ok"


def test_kwant_remote_dir_includes_local_root_identity(tmp_path: Path) -> None:
    bench_a = tmp_path / "iter_a" / "model_b" / "c" / "kwant"
    bench_b = tmp_path / "iter_b" / "model_b" / "c" / "kwant"
    remote_a = nbcluster._kwant_remote_dir(
        remote_workdir="/remote/work",
        mp_id="mp-1018028",
        benchmark_path=bench_a,
    )
    remote_b = nbcluster._kwant_remote_dir(
        remote_workdir="/remote/work",
        mp_id="mp-1018028",
        benchmark_path=bench_b,
    )

    assert remote_a != remote_b
    assert "iter_a_model_b_c_kwant_" in remote_a
    assert "iter_b_model_b_c_kwant_" in remote_b


def test_submit_kwant_nanowire_reference_forwards_heartbeat_env_override(
    tmp_path: Path, monkeypatch
) -> None:
    spec = NanowireBenchmarkSpec()
    fermi_ev = 1.23
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    hr_path = benchmark_dir / "toy_hr.dat"
    hr_path.write_text("dummy")
    worker_zip = benchmark_dir / "wtec_src.zip"
    worker_zip.write_text("zip")

    seen: dict[str, object] = {}

    class _FakeJobManager:
        def __init__(self, ssh: object) -> None:
            seen["ssh"] = ssh

        def resolve_queue(self, queue: str, fallback_order: list[str] | None = None) -> str:
            return queue

        def submit_and_wait(self, script: str, **kwargs):
            seen["script"] = script
            result_path = benchmark_dir / "kwant_reference.json"
            result_path.write_text(
                json.dumps(_complete_kwant_reference_payload(spec=spec, fermi_ev=fermi_ev))
            )
            return {"status": "ok"}

    monkeypatch.setenv("TOPOSLAB_KWANT_BENCH_HEARTBEAT_SECONDS", "20")
    monkeypatch.setattr(nbcluster, "open_ssh", lambda cfg: _DummySSH())
    monkeypatch.setattr(nbcluster, "JobManager", _FakeJobManager)
    monkeypatch.setattr(nbcluster.ClusterConfig, "from_env", staticmethod(lambda: _FakeClusterConfig()))
    monkeypatch.setattr(
        nbcluster.TopoSlabWorkflow,
        "_worker_source_zip",
        staticmethod(lambda _: worker_zip),
    )

    canonical = CanonicalizedNanowireInput(
        axis="c",
        hr_dat_path=str(hr_path),
        win_path=str(benchmark_dir / "toy.win"),
        permutation=(2, 0, 1),
        lattice_vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    result, meta = nbcluster.submit_kwant_nanowire_reference(
        canonical_input=canonical,
        benchmark_dir=benchmark_dir,
        spec=spec,
        model_key="model_a",
        model_label="Model A",
        fermi_ev=fermi_ev,
        length_uc=24,
        queue_override="g4",
        python_executable="python3",
        live_log=False,
    )

    script = str(seen["script"])
    assert "export TOPOSLAB_KWANT_BENCH_HEARTBEAT_SECONDS=20" in script
    assert result["validation"]["status"] == "ok"
    assert meta["status"] == "ok"


def test_submit_kwant_nanowire_reference_stages_local_partial_checkpoint_and_shards(
    tmp_path: Path, monkeypatch
) -> None:
    spec = NanowireBenchmarkSpec(energies_ev=(-0.2, 0.0), thicknesses_uc=(1,))
    fermi_ev = 1.23
    benchmark_dir = tmp_path / "iter_resume" / "model_b" / "c" / "kwant"
    benchmark_dir.mkdir(parents=True)
    hr_path = benchmark_dir / "toy_hr.dat"
    hr_path.write_text("dummy")
    worker_zip = benchmark_dir / "wtec_src.zip"
    worker_zip.write_text("zip")
    checkpoint_path = benchmark_dir / "kwant_reference.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "status": "partial",
                "task_count_expected": 2,
                "task_count_completed": 1,
                "results": [
                    {
                        "thickness_uc": 1,
                        "energy_rel_fermi_ev": -0.2,
                        "energy_abs_ev": 1.03,
                        "transmission_e2_over_h": 11.0,
                    }
                ],
                "validation": {"status": "partial"},
            }
        )
    )
    shard_path = benchmark_dir / "kwant_reference.rank0.jsonl"
    shard_path.write_text(
        json.dumps(
            {
                "thickness_uc": 1,
                "energy_rel_fermi_ev": -0.2,
                "energy_abs_ev": 1.03,
                "transmission_e2_over_h": 11.0,
            }
        )
        + "\n"
    )

    seen: dict[str, object] = {}

    class _FakeJobManager:
        def __init__(self, ssh: object) -> None:
            seen["ssh"] = ssh

        def resolve_queue(self, queue: str, fallback_order: list[str] | None = None) -> str:
            return queue

        def submit_and_wait(self, script: str, **kwargs):
            seen["remote_dir"] = kwargs.get("remote_dir")
            seen["stage_files"] = kwargs.get("stage_files")
            checkpoint_path.write_text(
                json.dumps(_complete_kwant_reference_payload(spec=spec, fermi_ev=fermi_ev))
            )
            return {"status": "ok"}

    monkeypatch.setattr(nbcluster, "open_ssh", lambda cfg: _DummySSH())
    monkeypatch.setattr(nbcluster, "JobManager", _FakeJobManager)
    monkeypatch.setattr(nbcluster.ClusterConfig, "from_env", staticmethod(lambda: _FakeClusterConfig()))
    monkeypatch.setattr(
        nbcluster.TopoSlabWorkflow,
        "_worker_source_zip",
        staticmethod(lambda _: worker_zip),
    )

    canonical = CanonicalizedNanowireInput(
        axis="c",
        hr_dat_path=str(hr_path),
        win_path=str(benchmark_dir / "toy.win"),
        permutation=(2, 0, 1),
        lattice_vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    result, meta = nbcluster.submit_kwant_nanowire_reference(
        canonical_input=canonical,
        benchmark_dir=benchmark_dir,
        spec=spec,
        model_key="model_b",
        model_label="Model B",
        fermi_ev=fermi_ev,
        length_uc=24,
        queue_override="g4",
        python_executable="python3",
        live_log=False,
    )

    staged = [Path(p) for p in seen["stage_files"]]
    assert checkpoint_path in staged
    assert shard_path in staged
    assert "iter_resume_model_b_c_kwant_" in str(seen["remote_dir"])
    assert result["validation"]["status"] == "ok"
    assert meta["status"] == "ok"


def test_kwant_worker_layout_honors_env_override(monkeypatch) -> None:
    monkeypatch.setenv("TOPOSLAB_KWANT_BENCH_MPI_RANKS", "2")
    mpi_np, omp_threads = nbcluster._kwant_worker_layout(total_cores=64, task_count=35, n_nodes=1)
    assert mpi_np == 2
    assert omp_threads == 32


def test_resolve_kwant_reference_walltime_scales_by_worker_waves() -> None:
    walltime = nbcluster._resolve_kwant_reference_walltime(
        base_walltime="01:00:00",
        total_cores=64,
        task_count=35,
        n_nodes=1,
    )
    assert walltime == "03:00:00"


def test_resolve_kwant_reference_walltime_honors_env_override(monkeypatch) -> None:
    monkeypatch.setenv("TOPOSLAB_KWANT_BENCH_WALLTIME", "02:30:00")
    walltime = nbcluster._resolve_kwant_reference_walltime(
        base_walltime="01:00:00",
        total_cores=64,
        task_count=35,
        n_nodes=1,
    )
    assert walltime == "02:30:00"


def test_submit_kwant_nanowire_reference_forwards_cancel_event(
    tmp_path: Path, monkeypatch
) -> None:
    spec = NanowireBenchmarkSpec()
    fermi_ev = 1.23
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    hr_path = benchmark_dir / "toy_hr.dat"
    hr_path.write_text("dummy")
    worker_zip = benchmark_dir / "wtec_src.zip"
    worker_zip.write_text("zip")
    cancel_event = Event()

    seen: dict[str, object] = {}

    class _FakeJobManager:
        def __init__(self, ssh: object) -> None:
            seen["ssh"] = ssh

        def resolve_queue(self, queue: str, fallback_order: list[str] | None = None) -> str:
            return queue

        def submit_and_wait(self, script: str, **kwargs):
            seen["cancel_event"] = kwargs.get("cancel_event")
            result_path = benchmark_dir / "kwant_reference.json"
            result_path.write_text(
                json.dumps(_complete_kwant_reference_payload(spec=spec, fermi_ev=fermi_ev))
            )
            return {"status": "ok"}

    monkeypatch.setattr(nbcluster, "open_ssh", lambda cfg: _DummySSH())
    monkeypatch.setattr(nbcluster, "JobManager", _FakeJobManager)
    monkeypatch.setattr(nbcluster.ClusterConfig, "from_env", staticmethod(lambda: _FakeClusterConfig()))
    monkeypatch.setattr(
        nbcluster.TopoSlabWorkflow,
        "_worker_source_zip",
        staticmethod(lambda _: worker_zip),
    )

    canonical = CanonicalizedNanowireInput(
        axis="c",
        hr_dat_path=str(hr_path),
        win_path=str(benchmark_dir / "toy.win"),
        permutation=(2, 0, 1),
        lattice_vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    result, meta = nbcluster.submit_kwant_nanowire_reference(
        canonical_input=canonical,
        benchmark_dir=benchmark_dir,
        spec=spec,
        model_key="model_a",
        model_label="Model A",
        fermi_ev=fermi_ev,
        length_uc=24,
        queue_override="g4",
        python_executable="python3",
        live_log=False,
        cancel_event=cancel_event,
    )

    assert seen["cancel_event"] is cancel_event
    assert result["validation"]["status"] == "ok"
    assert meta["status"] == "ok"


def test_submit_kwant_nanowire_reference_resubmits_from_partial_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    hr_path = benchmark_dir / "toy_hr.dat"
    hr_path.write_text("dummy")
    worker_zip = benchmark_dir / "wtec_src.zip"
    worker_zip.write_text("zip")

    seen: dict[str, object] = {"submit_calls": 0}

    class _FakeJobManager:
        def __init__(self, ssh: object) -> None:
            seen["ssh"] = ssh

        def resolve_queue(self, queue: str, fallback_order: list[str] | None = None) -> str:
            return queue

        def submit_and_wait(self, script: str, **kwargs):
            seen["submit_calls"] = int(seen["submit_calls"]) + 1
            result_path = benchmark_dir / "kwant_reference.json"
            if int(seen["submit_calls"]) == 1:
                result_path.write_text(
                    json.dumps(
                        {
                            "status": "partial",
                            "task_count_expected": 3,
                            "task_count_completed": 1,
                            "results": [
                                {
                                    "thickness_uc": 1,
                                    "energy_rel_fermi_ev": -0.2,
                                    "energy_abs_ev": 1.03,
                                    "transmission_e2_over_h": 11.0,
                                }
                            ],
                            "validation": {"status": "partial"},
                        }
                    )
                )
                raise RuntimeError("job failed mid-sweep")
            result_path.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "task_count_expected": 3,
                        "task_count_completed": 3,
                        "results": [
                            {
                                "thickness_uc": 1,
                                "energy_rel_fermi_ev": -0.2,
                                "energy_abs_ev": 1.03,
                                "transmission_e2_over_h": 11.0,
                            },
                            {
                                "thickness_uc": 1,
                                "energy_rel_fermi_ev": 0.0,
                                "energy_abs_ev": 1.23,
                                "transmission_e2_over_h": 12.0,
                            },
                            {
                                "thickness_uc": 1,
                                "energy_rel_fermi_ev": 0.2,
                                "energy_abs_ev": 1.43,
                                "transmission_e2_over_h": 13.0,
                            },
                        ],
                        "validation": {"status": "ok"},
                    }
                )
            )
            return {"status": "ok", "job_id": "resume-2"}

    monkeypatch.setattr(nbcluster, "open_ssh", lambda cfg: _DummySSH())
    monkeypatch.setattr(nbcluster, "JobManager", _FakeJobManager)
    monkeypatch.setattr(nbcluster.ClusterConfig, "from_env", staticmethod(lambda: _FakeClusterConfig()))
    monkeypatch.setattr(
        nbcluster.TopoSlabWorkflow,
        "_worker_source_zip",
        staticmethod(lambda _: worker_zip),
    )

    canonical = CanonicalizedNanowireInput(
        axis="c",
        hr_dat_path=str(hr_path),
        win_path=str(benchmark_dir / "toy.win"),
        permutation=(2, 0, 1),
        lattice_vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    result, meta = nbcluster.submit_kwant_nanowire_reference(
        canonical_input=canonical,
        benchmark_dir=benchmark_dir,
        spec=NanowireBenchmarkSpec(energies_ev=(-0.2, 0.0, 0.2), thicknesses_uc=(1,)),
        model_key="model_a",
        model_label="Model A",
        fermi_ev=1.23,
        length_uc=24,
        queue_override="g4",
        python_executable="python3",
        live_log=False,
    )

    assert int(seen["submit_calls"]) == 2
    assert result["status"] == "ok"
    assert result["task_count_completed"] == 3
    assert meta["job_id"] == "resume-2"


def test_submit_kwant_nanowire_reference_resubmits_from_partial_rank_shards(
    tmp_path: Path, monkeypatch
) -> None:
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    hr_path = benchmark_dir / "toy_hr.dat"
    hr_path.write_text("dummy")
    worker_zip = benchmark_dir / "wtec_src.zip"
    worker_zip.write_text("zip")

    seen: dict[str, object] = {"submit_calls": 0}

    class _FakeJobManager:
        def __init__(self, ssh: object) -> None:
            seen["ssh"] = ssh

        def resolve_queue(self, queue: str, fallback_order: list[str] | None = None) -> str:
            return queue

        def submit_and_wait(self, script: str, **kwargs):
            seen["submit_calls"] = int(seen["submit_calls"]) + 1
            shard_path = benchmark_dir / "kwant_reference.rank0.jsonl"
            result_path = benchmark_dir / "kwant_reference.json"
            if int(seen["submit_calls"]) == 1:
                shard_path.write_text(
                    json.dumps(
                        {
                            "thickness_uc": 1,
                            "energy_rel_fermi_ev": -0.2,
                            "energy_abs_ev": 1.03,
                            "transmission_e2_over_h": 11.0,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                raise RuntimeError("job cancelled mid-round")
            result_path.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "task_count_expected": 3,
                        "task_count_completed": 3,
                        "results": [
                            {
                                "thickness_uc": 1,
                                "energy_rel_fermi_ev": -0.2,
                                "energy_abs_ev": 1.03,
                                "transmission_e2_over_h": 11.0,
                            },
                            {
                                "thickness_uc": 1,
                                "energy_rel_fermi_ev": 0.0,
                                "energy_abs_ev": 1.23,
                                "transmission_e2_over_h": 12.0,
                            },
                            {
                                "thickness_uc": 1,
                                "energy_rel_fermi_ev": 0.2,
                                "energy_abs_ev": 1.43,
                                "transmission_e2_over_h": 13.0,
                            },
                        ],
                        "validation": {"status": "ok"},
                    }
                )
            )
            return {"status": "ok", "job_id": "resume-shards"}

    monkeypatch.setattr(nbcluster, "open_ssh", lambda cfg: _DummySSH())
    monkeypatch.setattr(nbcluster, "JobManager", _FakeJobManager)
    monkeypatch.setattr(nbcluster.ClusterConfig, "from_env", staticmethod(lambda: _FakeClusterConfig()))
    monkeypatch.setattr(
        nbcluster.TopoSlabWorkflow,
        "_worker_source_zip",
        staticmethod(lambda _: worker_zip),
    )

    canonical = CanonicalizedNanowireInput(
        axis="c",
        hr_dat_path=str(hr_path),
        win_path=str(benchmark_dir / "toy.win"),
        permutation=(2, 0, 1),
        lattice_vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    result, meta = nbcluster.submit_kwant_nanowire_reference(
        canonical_input=canonical,
        benchmark_dir=benchmark_dir,
        spec=NanowireBenchmarkSpec(energies_ev=(-0.2, 0.0, 0.2), thicknesses_uc=(1,)),
        model_key="model_a",
        model_label="Model A",
        fermi_ev=1.23,
        length_uc=24,
        queue_override="g4",
        python_executable="python3",
        live_log=False,
    )

    assert int(seen["submit_calls"]) == 2
    assert result["status"] == "ok"
    assert result["task_count_completed"] == 3
    assert meta["job_id"] == "resume-shards"


def test_benchmark_transport_reaches_exact_sigma_eta_branch_without_name_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import click
    from wtec.cli import benchmark_transport

    config_path = tmp_path / "run_small.json"
    config_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "bench"
    axis_dir = output_dir / "model_b" / "c"
    kwant_dir = axis_dir / "kwant"
    rgf_dir = axis_dir / "rgf" / "d01_e0p0" / "transport" / "primary"
    kwant_dir.mkdir(parents=True, exist_ok=True)
    rgf_dir.mkdir(parents=True, exist_ok=True)
    hr_path = tmp_path / "TiS_hr.dat"
    win_path = tmp_path / "TiS.win"
    hr_path.write_text("dummy", encoding="utf-8")
    win_path.write_text("dummy", encoding="utf-8")

    (kwant_dir / "kwant_payload.json").write_text(
        json.dumps({"fermi_ev": 13.6046}),
        encoding="utf-8",
    )
    (kwant_dir / "kwant_reference.json").write_text(
        json.dumps(
            {
                "status": "partial",
                "task_count_expected": 2,
                "task_count_completed": 0,
                "results": [],
                "validation": {"status": "partial"},
            }
        ),
        encoding="utf-8",
    )
    (kwant_dir / "wtec_job.log").write_text(
        "[kwant-bench][rank=0] done thickness_uc=1 energy_abs_ev=13.604600 transmission=40.000000000000\n",
        encoding="utf-8",
    )
    (rgf_dir / "transport_payload_primary_001.json").write_text(
        json.dumps(
            {
                "thicknesses": [1],
                "disorder_strengths": [0.0],
                "mfp_lengths": [],
                "energy": 13.6046,
                "eta": 1.0e-8,
                "transport_rgf_mode": "full_finite",
                "sigma_left_path": "sigma_left.bin",
                "sigma_right_path": "sigma_right.bin",
            }
        ),
        encoding="utf-8",
    )
    (rgf_dir / "transport_result.json").write_text(
        json.dumps(
            {
                "runtime_cert": {
                    "full_finite_sigma_source": "kwant_exact",
                },
                "transport_results": {
                    "meta": {
                        "rgf_full_finite_sigma_source": "kwant_exact",
                    },
                    "thickness_scan": {"0.0": {"G_mean": [29.88663916904678]}},
                },
                "transport_results_raw": {
                    "eta": 1.0e-8,
                },
            }
        ),
        encoding="utf-8",
    )

    spec = SimpleNamespace(
        mp_id="mp-1018028",
        material="TiS",
        axes=("c",),
        energies_ev=(0.0,),
        thicknesses_uc=(1, 2),
        fixed_width_uc=13,
        trim_exclude_thicknesses_uc=(),
        abs_tol=5.0e-3,
        rel_tol=5.0e-4,
        zero_tol=1.0e-12,
        fit_r2_abs_tol=1.0e-3,
    )
    model = SimpleNamespace(
        key="model_b",
        label="Model B",
        custom_projections=[],
        primary_for_rgf=True,
    )

    monkeypatch.setattr("wtec.cli._load_runtime_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("wtec.cli._load_run_config", lambda _path: {"n_nodes": 1})
    monkeypatch.setattr("wtec.cli._ensure_nanowire_benchmark_rgf_router_ready", lambda **_kwargs: {})
    monkeypatch.setattr("wtec.cli._resolve_nanowire_benchmark_source_structure", lambda **_kwargs: "")
    monkeypatch.setattr("wtec.cli._build_nanowire_benchmark_source_seed", lambda **_kwargs: {})
    monkeypatch.setattr("wtec.cli._load_benchmark_source_resume", lambda _root: (hr_path, win_path, 13.6046))
    monkeypatch.setattr("wtec.transport.nanowire_benchmark.NanowireBenchmarkSpec", lambda: spec)
    monkeypatch.setattr("wtec.transport.nanowire_benchmark.select_benchmark_models", lambda *_args, **_kwargs: [model])
    monkeypatch.setattr(
        "wtec.transport.nanowire_benchmark.prepare_canonicalized_inputs",
        lambda **_kwargs: SimpleNamespace(hr_dat_path=str(hr_path)),
    )
    monkeypatch.setattr("wtec.wannier.parser.read_hr_dat", lambda _path: object())
    monkeypatch.setattr("wtec.rgf.effective_principal_layer_width", lambda *args, **kwargs: 1)
    monkeypatch.setattr("wtec.transport.nanowire_benchmark.compute_length_uc", lambda *_args, **_kwargs: 24)
    monkeypatch.setattr(
        "wtec.transport.nanowire_benchmark_cluster.submit_kwant_nanowire_reference",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("new Kwant submission should be skipped")),
    )
    monkeypatch.setattr(
        "wtec.cli._run_rgf_benchmark_axis",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("new RGF launch should be skipped")),
    )

    with pytest.raises(click.ClickException, match="Transport benchmark failed"):
        benchmark_transport.callback(
            config_file=str(config_path),
            output_dir=str(output_dir),
            queue="g4",
            walltime="01:00:00",
            live_log=True,
            log_poll_interval=5,
            stale_log_seconds=300,
            source_nodes=2,
            all_models=False,
        )

    summary_path = output_dir / "benchmark_summary.json"
    partial_path = axis_dir / "comparison_partial.json"
    assert summary_path.exists()
    assert partial_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert "model_b:c:rgf_partial" in summary["failed_targets"]
