import numpy as np
from pathlib import Path
import json
import sys
import types

from wtec.workflow.orchestrator import TopoSlabWorkflow, _jsonable
from wtec.wannier.parser import HoppingData, write_hr_dat


def test_orchestrator_jsonable_converts_numpy_payloads() -> None:
    payload = {
        "arr": np.array([1.0, 2.0]),
        "scalar": np.float64(3.5),
        "nested": [{"value": np.int64(7)}],
    }
    out = _jsonable(payload)
    assert out == {
        "arr": [1.0, 2.0],
        "scalar": 3.5,
        "nested": [{"value": 7}],
    }


def test_transport_stage_routes_to_kwant_qsub(tmp_path, monkeypatch) -> None:
    cfg = {
        "name": "demo",
        "run_dir": str(tmp_path / "run"),
        "transport_engine": "kwant",
        "transport_backend": "qsub",
        "transport_strict_qsub": True,
    }
    wf = TopoSlabWorkflow.from_config(cfg)
    wf._state = {"stage": "WANNIER90", "outputs": {}}

    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(wf, "_set_stage", lambda stage: calls.append(("set_stage", stage)))
    monkeypatch.setattr(wf, "_load_cached_transport_results", lambda: None)
    monkeypatch.setattr(wf, "_save_checkpoint", lambda: calls.append(("save", "checkpoint")))

    def _fake_qsub(hr_dat: Path, label: str = "primary"):
        calls.append(("qsub", str(hr_dat), label))
        return {"status": "ok"}, {"job_id": "123"}

    monkeypatch.setattr(wf, "_stage_transport_qsub", _fake_qsub)

    hr_dat = tmp_path / "TaP_hr.dat"
    hr_dat.write_text("dummy")
    out = wf._stage_transport(hr_dat)
    assert out["status"] == "ok"
    assert out["meta"]["transport_engine_requested"] == "kwant"
    assert out["meta"]["transport_engine_resolved"] == "kwant"
    assert ("qsub", str(hr_dat), "primary") in calls


def test_transport_stage_routes_auto_to_kwant_qsub(tmp_path, monkeypatch) -> None:
    cfg = {
        "name": "demo",
        "run_dir": str(tmp_path / "run"),
        "transport_engine": "auto",
        "transport_backend": "qsub",
        "transport_strict_qsub": True,
    }
    wf = TopoSlabWorkflow.from_config(cfg)
    wf._state = {"stage": "WANNIER90", "outputs": {}}

    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(wf, "_set_stage", lambda stage: calls.append(("set_stage", stage)))
    monkeypatch.setattr(wf, "_load_cached_transport_results", lambda: None)
    monkeypatch.setattr(wf, "_save_checkpoint", lambda: calls.append(("save", "checkpoint")))

    def _fake_qsub(hr_dat: Path, label: str = "primary"):
        calls.append(("qsub", str(hr_dat), label))
        return {"status": "ok"}, {"job_id": "123"}

    monkeypatch.setattr(wf, "_stage_transport_qsub", _fake_qsub)

    hr_dat = tmp_path / "TaP_hr.dat"
    hr_dat.write_text("dummy")
    out = wf._stage_transport(hr_dat)
    assert out["status"] == "ok"
    assert out["meta"]["transport_engine_requested"] == "auto"
    assert out["meta"]["transport_engine_resolved"] == "kwant"
    assert ("qsub", str(hr_dat), "primary") in calls


def test_transport_stage_routes_auto_to_rgf_qsub_when_router_ready(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / ".wtec"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "init_state.json").write_text(
        json.dumps(
            {
                "rgf": {
                    "cluster": {
                        "ready": True,
                        "binary_id": "wtec_rgf_runner_phase2_v2",
                        "binary_path": "/remote/wtec_rgf_runner",
                        "numerical_status": "phase1_ready",
                    }
                }
            }
        )
    )
    monkeypatch.setenv("WTEC_STATE_DIR", str(state_dir))
    cfg = {
        "name": "demo",
        "run_dir": str(tmp_path / "run"),
        "transport_engine": "auto",
        "transport_backend": "qsub",
        "transport_strict_qsub": True,
        "transport_rgf_mode": "periodic_transverse",
        "transport_rgf_periodic_axis": "y",
        "transport_axis": "x",
        "thickness_axis": "z",
        "disorder_strengths": [0.0],
        "n_ensemble": 1,
    }
    wf = TopoSlabWorkflow.from_config(cfg)
    wf._state = {"stage": "WANNIER90", "outputs": {}}

    calls: list[tuple[str, str, str]] = []

    monkeypatch.setattr(wf, "_set_stage", lambda stage: calls.append(("set_stage", stage, "")))
    monkeypatch.setattr(wf, "_load_cached_transport_results", lambda: None)
    monkeypatch.setattr(wf, "_save_checkpoint", lambda: calls.append(("save", "checkpoint", "")))

    def _fake_rgf_qsub(hr_dat: Path, label: str = "primary"):
        calls.append(("rgf_qsub", str(hr_dat), label))
        return {"status": "ok", "meta": {"transport_engine": "rgf"}}, {"job_id": "456"}

    monkeypatch.setattr(wf, "_stage_transport_rgf_qsub", _fake_rgf_qsub)

    hr_dat = tmp_path / "TaP_hr.dat"
    hr_dat.write_text("dummy")
    out = wf._stage_transport(hr_dat)
    assert out["status"] == "ok"
    assert out["meta"]["transport_engine_requested"] == "auto"
    assert out["meta"]["transport_engine_resolved"] == "rgf"
    assert ("rgf_qsub", str(hr_dat), "primary") in calls


def test_transport_stage_routes_rgf_to_native_qsub(tmp_path, monkeypatch) -> None:
    cfg = {
        "name": "demo",
        "run_dir": str(tmp_path / "run"),
        "transport_engine": "rgf",
        "transport_backend": "qsub",
        "transport_strict_qsub": True,
        "disorder_strengths": [0.0],
        "n_ensemble": 1,
    }
    wf = TopoSlabWorkflow.from_config(cfg)
    wf._state = {"stage": "WANNIER90", "outputs": {}}
    calls: list[tuple[str, str, str]] = []

    monkeypatch.setattr(wf, "_set_stage", lambda stage: calls.append(("set_stage", stage, "")))
    monkeypatch.setattr(wf, "_load_cached_transport_results", lambda: None)
    monkeypatch.setattr(wf, "_save_checkpoint", lambda: calls.append(("save", "checkpoint", "")))

    def _fake_rgf_qsub(hr_dat: Path, label: str = "primary"):
        calls.append(("rgf_qsub", str(hr_dat), label))
        return {"status": "ok", "meta": {"transport_engine": "rgf"}}, {"job_id": "456"}

    monkeypatch.setattr(wf, "_stage_transport_rgf_qsub", _fake_rgf_qsub)

    hr_dat = tmp_path / "TaP_hr.dat"
    hr_dat.write_text("dummy")
    out = wf._stage_transport(hr_dat)
    assert out["status"] == "ok"
    assert out["meta"]["transport_engine_requested"] == "rgf"
    assert out["meta"]["transport_engine_resolved"] == "rgf"
    assert ("rgf_qsub", str(hr_dat), "primary") in calls


def test_prepare_delta_h_artifact_uses_cfg_fermi_shift_when_reusing_hr(tmp_path, monkeypatch) -> None:
    cfg = {
        "name": "demo",
        "run_dir": str(tmp_path / "run"),
        "material": "TaP",
        "dft_mode": "dual_family",
        "dft_lcao_engine": "siesta",
        "fermi_shift_eV": 12.34,
    }
    wf = TopoSlabWorkflow.from_config(cfg)
    wf._state = {"stage": "WANNIER90", "outputs": {}}

    monkeypatch.setattr(wf, "_anchor_transfer_cfg", lambda: {
        "enabled": True,
        "mode": "delta_h",
        "reuse_existing": False,
        "max_retries": 1,
        "retry_kmesh_step": 0,
        "retry_window_step_ev": 0.0,
        "fit_window_ev": 1.5,
        "fit_kmesh": [1, 1, 1],
        "alpha_grid_min": 0.0,
        "alpha_grid_max": 1.0,
        "alpha_grid_points": 2,
        "basis_policy": "strict_same_basis",
        "scope": "onsite_plus_first_shell",
    })
    monkeypatch.setattr(wf, "_dft_mode", lambda: "dual_family")
    monkeypatch.setattr(wf, "_variant_dft_engine", lambda: "siesta")
    monkeypatch.setattr(wf, "_build_atoms_from_config", lambda: None)

    class _Atoms:
        def get_chemical_symbols(self):
            return ["Ta", "P"]

    monkeypatch.setattr(wf, "_reference_atoms", lambda atoms: _Atoms())
    monkeypatch.setattr(wf, "_dft_siesta_cfg", lambda: {"spin_orbit": False})
    monkeypatch.setattr(wf, "_dft_dispersion_cfg", lambda: {})
    monkeypatch.setattr(wf, "_save_checkpoint", lambda: None)

    import types
    import sys

    class _DummyPipe:
        def __init__(self, *args, **kwargs):
            pass
        def run_scf(self):
            return {"job_id": "1"}
        def run_nscf(self, fermi):
            return {"job_id": "2"}
        def run_wannier(self, fermi, **kwargs):
            anchor_dir = Path(cfg["run_dir"]) / "dft_anchor_lcao"
            anchor_dir.mkdir(parents=True, exist_ok=True)
            (anchor_dir / "TaP_hr.dat").write_text("dummy")
            (anchor_dir / "TaP.win").write_text("begin projections\nTa:d\nP:p\nend projections\n")
            (anchor_dir / "TaP.scf.out").write_text("Fermi = 1.23 eV\n")
            return {"job_id": "3"}

    monkeypatch.setitem(sys.modules, "wtec.cluster.ssh", types.SimpleNamespace(open_ssh=lambda cfg: _SSH()))
    monkeypatch.setitem(sys.modules, "wtec.cluster.submit", types.SimpleNamespace(JobManager=lambda ssh: object()))
    monkeypatch.setitem(sys.modules, "wtec.config.cluster", types.SimpleNamespace(ClusterConfig=types.SimpleNamespace(from_env=lambda: types.SimpleNamespace(
        remote_workdir=str(tmp_path / "remote"),
        mpi_cores=32,
        mpi_cores_by_queue={},
        pbs_queue="g2",
        pbs_queue_priority=["g2"],
        omp_threads=1,
        modules=[],
        bin_dirs=[],
        siesta_pseudo_dir="/pseudo",
        abacus_pseudo_dir="/pseudo",
        abacus_orbital_dir="/orb",
        resolved_siesta_pseudo_dir=lambda spin_orbit=True, explicit="": explicit or "/pseudo",
    ))))
    monkeypatch.setitem(sys.modules, "wtec.config.materials", types.SimpleNamespace(get_material=lambda material: types.SimpleNamespace(dis_win=(-4, 16), dis_froz_win=(-1, 1))))
    monkeypatch.setitem(sys.modules, "wtec.siesta.parser", types.SimpleNamespace(parse_fermi_energy=lambda path: 1.23))
    monkeypatch.setitem(sys.modules, "wtec.siesta.runner", types.SimpleNamespace(SiestaPipeline=_DummyPipe))

    captured = {}
    def _fake_build_delta_h_artifact(**kwargs):
        captured.update(kwargs)
        return {"anchor": {"basis": {"num_wann": 2}}, "delta_h": {"r_vectors": [], "mat_real": [], "mat_imag": []}, "fit": {}}
    monkeypatch.setitem(sys.modules, "wtec.wannier.delta_h", types.SimpleNamespace(
        build_delta_h_artifact=_fake_build_delta_h_artifact,
        write_delta_h_artifact=lambda path, artifact: Path(path).write_text("{}"),
    ))

    pes_hr = tmp_path / "TaP_hr.dat"
    pes_hr.write_text("dummy")
    (tmp_path / "TaP.win").write_text("begin projections\nTa:d\nP:p\nend projections\n")
    wf._prepare_delta_h_artifact(pes_hr)
    assert captured["fermi_pes_ev"] == 12.34


class _SSH:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, cmd, check=False):
        return 0, "", ""


def test_stage_transport_rgf_qsub_writes_standard_transport_payload(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / ".wtec"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "init_state.json").write_text(
        json.dumps(
            {
                "rgf": {
                    "cluster": {
                        "ready": True,
                        "binary_id": "wtec_rgf_runner_scaffold_v0",
                        "binary_path": "/remote/wtec_rgf_runner",
                        "numerical_status": "phase1_ready",
                    }
                }
            }
        )
    )
    monkeypatch.setenv("WTEC_STATE_DIR", str(state_dir))

    hd = HoppingData(
        num_wann=1,
        r_vectors=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int),
        deg=np.array([1, 1, 1], dtype=int),
        H_R=np.array([[[0.0 + 0.0j]], [[-1.0 + 0.0j]], [[-1.0 + 0.0j]]], dtype=complex),
    )
    hr_dat = tmp_path / "toy_hr.dat"
    write_hr_dat(hr_dat, hd, header="toy chain")
    (tmp_path / "TaP.win").write_text(
        "\n".join(
            [
                "begin unit_cell_cart",
                "ang",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "end unit_cell_cart",
            ]
        )
        + "\n"
    )

    cfg = {
        "name": "demo",
        "material": "TaP",
        "run_dir": str(tmp_path / "run"),
        "transport_engine": "rgf",
        "transport_backend": "qsub",
        "transport_strict_qsub": True,
        "transport_axis": "x",
        "thickness_axis": "z",
        "transport_n_layers_x": 3,
        "transport_n_layers_y": 1,
        "mfp_n_layers_z": 1,
        "thicknesses": [1, 2],
        "mfp_lengths": [2, 4, 6],
        "disorder_strengths": [0.0],
        "n_ensemble": 1,
        "runtime_logging_detail": "per_step",
        "runtime_logging_heartbeat_seconds": 11,
    }
    wf = TopoSlabWorkflow.from_config(cfg)
    wf._state = {"stage": "WANNIER90", "outputs": {}}
    captured: dict[str, object] = {}

    class _FakeMPIConfig:
        def __init__(self, n_cores=1, bind_to="core", **kwargs):
            self.n_cores = n_cores
            self.bind_to = bind_to

    class _FakePBSJobConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _FakeClusterConfig:
        remote_workdir = "/remote/wtec_runs"
        pbs_queue = "g2"
        pbs_queue_priority = ["g2"]
        modules = []
        bin_dirs = []

        def cores_for_queue(self, _queue):
            return 32

    class _FakeJM:
        def __init__(self, ssh):
            self.ssh = ssh

        def resolve_queue(self, preferred, fallback_order=None):
            return preferred or "g2"

        def ensure_remote_commands(self, commands, modules=None, bin_dirs=None):
            return None

        def submit_and_wait(self, script, remote_dir, local_dir, retrieve_patterns, script_name, stage_files, expected_local_outputs, queue_used, **kwargs):
            captured["retrieve_patterns"] = list(retrieve_patterns)
            captured["live_files"] = list(kwargs.get("live_files", []))
            captured["payload"] = json.loads(Path(local_dir, "transport_payload.json").read_text())
            raw = {
                "transport_results_raw": {
                    "engine": "rgf",
                    "mode": "periodic_transverse",
                    "periodic_axis": "y",
                    "lead_axis": "x",
                    "thickness_axis": "z",
                    "n_layers_x": 3,
                    "n_layers_y": 1,
                    "mfp_n_layers_z": 1,
                    "energy": 0.25,
                    "eta": 1.0e-6,
                    "thicknesses": [1, 2],
                    "thickness_G": [1.0, 2.0],
                    "thickness_p_eff": [1, 1],
                    "thickness_slice_count": [1, 2],
                    "thickness_superslice_dim": [4, 8],
                    "mfp_lengths": [2, 4, 6],
                    "length_G": [1.0, 0.5, 1.0 / 3.0],
                    "length_p_eff": [1, 1, 1],
                    "length_slice_count": [1, 1, 1],
                    "length_superslice_dim": [4, 4, 4],
                },
                "runtime_cert": {
                    "engine": "rgf",
                    "binary_id": "wtec_rgf_runner_scaffold_v0",
                    "numerical_status": "phase1_ready",
                    "mpi_size": 5,
                    "max_slice_count": 2,
                    "max_superslice_dim": 8,
                },
            }
            Path(local_dir, expected_local_outputs[0]).write_text(json.dumps(raw, indent=2))
            return {"job_id": "999", "queue": queue_used, "status": "COMPLETED"}

    monkeypatch.setitem(
        sys.modules,
        "wtec.cluster.mpi",
        types.SimpleNamespace(
            MPIConfig=_FakeMPIConfig,
            build_command=lambda executable, mpi=None, extra_args="": f"mpirun -np {mpi.n_cores} {executable} {extra_args}".strip(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "wtec.cluster.pbs",
        types.SimpleNamespace(
            PBSJobConfig=_FakePBSJobConfig,
            generate_script=lambda cfg, commands: "\n".join(commands),
        ),
    )
    monkeypatch.setitem(sys.modules, "wtec.cluster.ssh", types.SimpleNamespace(open_ssh=lambda cfg: _SSH()))
    monkeypatch.setitem(sys.modules, "wtec.cluster.submit", types.SimpleNamespace(JobManager=_FakeJM))
    monkeypatch.setitem(
        sys.modules,
        "wtec.config.cluster",
        types.SimpleNamespace(ClusterConfig=types.SimpleNamespace(from_env=lambda: _FakeClusterConfig())),
    )

    results, meta = wf._stage_transport_rgf_qsub(hr_dat, label="primary")
    assert meta["job_id"] == "999"
    assert results["meta"]["transport_engine"] == "rgf"
    assert results["meta"]["energy_eV"] == 0.25
    assert results["thickness_scan"][0.0]["G_mean"] == [1.0, 2.0]
    assert captured["payload"]["progress_file"] == "transport_progress.jsonl"
    assert captured["payload"]["logging_detail"] == "per_step"
    assert captured["payload"]["heartbeat_seconds"] == 11
    assert captured["payload"]["parallel_policy_resolved"] == "throughput"
    assert captured["payload"]["expected_mpi_np"] == 5
    assert captured["payload"]["expected_omp_threads"] == 6
    assert "transport_progress.jsonl" in captured["retrieve_patterns"]
    assert "transport_progress.jsonl" in captured["live_files"]
    written = json.loads((Path(cfg["run_dir"]) / "transport" / "primary" / "transport_result.json").read_text())
    assert written["runtime_cert"]["numerical_status"] == "phase1_ready"
    assert written["runtime_cert"]["parallel_policy"] == "throughput"
    assert written["runtime_cert"]["omp_threads"] == 6


def test_stage_transport_rgf_qsub_canonicalizes_axes_and_uses_single_point_threads(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / ".wtec"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "init_state.json").write_text(
        json.dumps(
            {
                "rgf": {
                    "cluster": {
                        "ready": True,
                        "binary_id": "wtec_rgf_runner_phase2_v4",
                        "binary_path": "/remote/wtec_rgf_runner",
                        "numerical_status": "phase2_experimental",
                        "probe": {"build_env": {"openmp_enabled": True, "blas_backend": "openblas"}},
                    }
                }
            }
        )
    )
    monkeypatch.setenv("WTEC_STATE_DIR", str(state_dir))

    hd = HoppingData(
        num_wann=1,
        r_vectors=np.array([[0, 0, 0], [0, 0, 1], [0, 0, -1]], dtype=int),
        deg=np.array([1, 1, 1], dtype=int),
        H_R=np.array([[[0.0 + 0.0j]], [[-1.0 + 0.0j]], [[-1.0 + 0.0j]]], dtype=complex),
    )
    hr_dat = tmp_path / "toy_full_finite_hr.dat"
    write_hr_dat(hr_dat, hd, header="toy chain z-lead")
    (tmp_path / "TaP.win").write_text(
        "\n".join(
            [
                "begin unit_cell_cart",
                "ang",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "end unit_cell_cart",
            ]
        )
        + "\n"
    )

    cfg = {
        "name": "demo_ff",
        "material": "TaP",
        "run_dir": str(tmp_path / "run_ff"),
        "transport_engine": "rgf",
        "transport_backend": "qsub",
        "transport_strict_qsub": True,
        "transport_axis": "z",
        "thickness_axis": "x",
        "transport_rgf_mode": "full_finite",
        "transport_rgf_parallel_policy": "auto",
        "transport_n_layers_x": 3,
        "transport_n_layers_y": 2,
        "mfp_n_layers_z": 1,
        "thicknesses": [1],
        "mfp_lengths": [],
        "disorder_strengths": [0.0],
        "n_ensemble": 1,
    }
    wf = TopoSlabWorkflow.from_config(cfg)
    wf._state = {"stage": "WANNIER90", "outputs": {}}
    captured: dict[str, object] = {}

    class _FakeMPIConfig:
        def __init__(self, n_cores=1, bind_to="core", **kwargs):
            self.n_cores = n_cores
            self.bind_to = bind_to

    class _FakePBSJobConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _FakeClusterConfig:
        remote_workdir = "/remote/wtec_runs"
        pbs_queue = "g2"
        pbs_queue_priority = ["g2"]
        modules = []
        bin_dirs = []

        def cores_for_queue(self, _queue):
            return 32

    class _FakeJM:
        def __init__(self, ssh):
            self.ssh = ssh

        def resolve_queue(self, preferred, fallback_order=None):
            return preferred or "g2"

        def ensure_remote_commands(self, commands, modules=None, bin_dirs=None):
            return None

        def submit_and_wait(self, script, remote_dir, local_dir, retrieve_patterns, script_name, stage_files, expected_local_outputs, queue_used, **kwargs):
            captured["payload"] = json.loads(Path(local_dir, "transport_payload.json").read_text())
            raw = {
                "transport_results_raw": {
                    "engine": "rgf",
                    "mode": "full_finite",
                    "periodic_axis": "y",
                    "lead_axis": "x",
                    "thickness_axis": "z",
                    "n_layers_x": 3,
                    "n_layers_y": 2,
                    "mfp_n_layers_z": 1,
                    "energy": 0.0,
                    "eta": 1.0e-6,
                    "thicknesses": [1],
                    "thickness_G": [1.0],
                    "thickness_p_eff": [1],
                    "thickness_slice_count": [1],
                    "thickness_superslice_dim": [6],
                    "mfp_lengths": [],
                    "length_G": [],
                    "length_p_eff": [],
                    "length_slice_count": [],
                    "length_superslice_dim": [],
                },
                "runtime_cert": {
                    "engine": "rgf",
                    "binary_id": "wtec_rgf_runner_phase2_v4",
                    "numerical_status": "phase2_experimental",
                    "mpi_size": 1,
                    "max_slice_count": 1,
                    "max_superslice_dim": 6,
                },
            }
            Path(local_dir, expected_local_outputs[0]).write_text(json.dumps(raw, indent=2))
            return {"job_id": "1001", "queue": queue_used, "status": "COMPLETED"}

    monkeypatch.setitem(
        sys.modules,
        "wtec.cluster.mpi",
        types.SimpleNamespace(
            MPIConfig=_FakeMPIConfig,
            build_command=lambda executable, mpi=None, extra_args="": f"mpirun -np {mpi.n_cores} {executable} {extra_args}".strip(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "wtec.cluster.pbs",
        types.SimpleNamespace(
            PBSJobConfig=_FakePBSJobConfig,
            generate_script=lambda cfg, commands: "\n".join(commands),
        ),
    )
    monkeypatch.setitem(sys.modules, "wtec.cluster.ssh", types.SimpleNamespace(open_ssh=lambda cfg: _SSH()))
    monkeypatch.setitem(sys.modules, "wtec.cluster.submit", types.SimpleNamespace(JobManager=_FakeJM))
    monkeypatch.setitem(
        sys.modules,
        "wtec.config.cluster",
        types.SimpleNamespace(ClusterConfig=types.SimpleNamespace(from_env=lambda: _FakeClusterConfig())),
    )

    results, meta = wf._stage_transport_rgf_qsub(hr_dat, label="primary")
    assert meta["job_id"] == "1001"
    assert results["meta"]["transport_engine"] == "rgf"
    assert captured["payload"]["hr_dat_path"].endswith("_canonical_hr.dat")
    assert captured["payload"]["lead_axis"] == "x"
    assert captured["payload"]["thickness_axis"] == "z"
    assert captured["payload"]["expected_mpi_np"] == 1
    assert captured["payload"]["expected_omp_threads"] == 32
    written = json.loads((Path(cfg["run_dir"]) / "transport" / "primary" / "transport_result.json").read_text())
    assert written["runtime_cert"]["parallel_policy"] == "single_point"
    assert written["runtime_cert"]["omp_threads"] == 32
