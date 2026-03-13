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


def test_submit_kwant_nanowire_reference_uses_conservative_multi_rank_layout(
    tmp_path: Path, monkeypatch
) -> None:
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
            result_path.write_text(json.dumps({"results": [], "validation": {"status": "ok"}}))
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
        spec=NanowireBenchmarkSpec(),
        model_key="model_a",
        model_label="Model A",
        fermi_ev=1.23,
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
            result_path.write_text(json.dumps({"results": [], "validation": {"status": "ok"}}))
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
        spec=NanowireBenchmarkSpec(),
        model_key="model_a",
        model_label="Model A",
        fermi_ev=1.23,
        length_uc=24,
        queue_override="g4",
        python_executable="python3",
        live_log=False,
        cancel_event=cancel_event,
    )

    assert seen["cancel_event"] is cancel_event
    assert result["validation"]["status"] == "ok"
    assert meta["status"] == "ok"
