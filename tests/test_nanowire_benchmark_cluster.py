from __future__ import annotations

import json
from pathlib import Path
from threading import Event

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
    assert "#PBS -l select=1:ncpus=64:mpiprocs=4:ompthreads=16" in script
    assert "#PBS -l walltime=09:00:00" in script
    assert "export OMP_NUM_THREADS=16" in script
    assert "export MKL_NUM_THREADS=16" in script
    assert "export OPENBLAS_NUM_THREADS=16" in script
    assert "export NUMEXPR_NUM_THREADS=16" in script
    assert "mpirun -np 4 --bind-to none" in script
    assert seen["kwargs"]["live_retrieve_patterns"] == [
        "kwant_reference.json",
        "kwant_reference.rank*.jsonl",
        "wtec_job.log",
    ]
    assert seen["kwargs"]["live_retrieve_interval_seconds"] == 5
    assert callable(seen["kwargs"]["live_retrieve_hook"])
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
    assert walltime == "09:00:00"


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
