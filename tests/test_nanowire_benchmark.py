from __future__ import annotations

from concurrent.futures import CancelledError
import json
from threading import Event
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from wtec.cli import (
    _KwantOverlapError,
    _PartialOverlapFailure,
    _append_nanowire_benchmark_trace,
    _build_nanowire_benchmark_source_seed,
    _build_tis_benchmark_source_cfg,
    _ensure_nanowire_benchmark_rgf_router_ready,
    _load_complete_nanowire_kwant_reference,
    _load_nanowire_kwant_reference_checkpoint,
    _resolve_nanowire_benchmark_source_structure,
    _run_rgf_benchmark_axis,
    _run_kwant_and_rgf_overlap,
    _write_partial_nanowire_axis_artifacts,
    main,
)
from wtec.config.materials import get_material
from wtec.qe.lcao import get_projections
from wtec.transport.nanowire_benchmark import (
    CanonicalizedNanowireInput,
    NanowireBenchmarkSpec,
    axis_permutation,
    canonicalize_hopping_data,
    compare_reference_and_rgf,
    prepare_canonicalized_inputs,
    select_benchmark_models,
    select_monotonic_thickness_subsequence,
)
from wtec.rgf import RGF_BINARY_ID
from wtec.wannier.model import _parse_lattice_from_win
from wtec.wannier.parser import HoppingData, read_hr_dat, write_hr_dat


def _toy_hd() -> HoppingData:
    return HoppingData(
        num_wann=1,
        r_vectors=np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=int,
        ),
        deg=np.asarray([1, 1, 1, 1], dtype=int),
        H_R=np.asarray(
            [
                [[0.0 + 0.0j]],
                [[-1.0 + 0.0j]],
                [[-2.0 + 0.0j]],
                [[-3.0 + 0.0j]],
            ],
            dtype=np.complex128,
        ),
    )


def test_tis_material_preset_exists() -> None:
    mat = get_material("TiS")
    assert mat.formula == "TiS"
    assert mat.space_group == "P-6m2"
    assert mat.projections == ["Ti:d", "S:p"]
    assert mat.num_wann == 16


def test_tis_qe_projection_library_exists() -> None:
    assert get_projections("TiS") == ["Ti:d", "S:p"]


def test_select_benchmark_models_defaults_to_primary_rgf_model() -> None:
    spec = NanowireBenchmarkSpec()
    primary = select_benchmark_models(spec)
    assert [model.key for model in primary] == ["model_b"]
    all_models = select_benchmark_models(spec, include_supplementary=True)
    assert [model.key for model in all_models] == ["model_a", "model_b"]


def test_build_tis_benchmark_source_cfg_uses_explicit_source_nodes(tmp_path: Path) -> None:
    structure = tmp_path / "TiS.cif"
    structure.write_text("data\n", encoding="utf-8")
    cfg = _build_tis_benchmark_source_cfg(
        base_cfg={"n_nodes": 1, "kpoints_scf": [1, 1, 1], "kpoints_nscf": [1, 1, 1]},
        benchmark_root=tmp_path / "bench",
        structure_file=str(structure),
        source_name="nanowire_benchmark_source_model_b",
        custom_projections=["Ti:d", "S:p"],
        source_n_nodes=2,
        live_log=True,
        log_poll_interval=5,
        stale_log_seconds=300,
    )
    assert cfg["n_nodes"] == 2
    assert cfg["run_dir"].endswith("bench/source_run")
    assert cfg["transport_backend"] == "qsub"


def test_build_nanowire_benchmark_source_seed_preserves_local_pes_reference(tmp_path: Path) -> None:
    cfg = _build_nanowire_benchmark_source_seed(
        base_cfg={
            "material": "OverrideTiS",
            "mp_api_key_env": "ALT_MP",
            "dft_pes_reference_mp_id": "mp-local",
            "dft_pes_reference_structure_file": str(tmp_path / "TiS_local.cif"),
            "dft_pes_reference_use_primitive": False,
        },
        benchmark_root=tmp_path / "bench",
        material="TiS",
        default_mp_id="mp-1018028",
    )
    assert cfg["material"] == "OverrideTiS"
    assert cfg["mp_api_key_env"] == "ALT_MP"
    assert cfg["dft_pes_reference_mp_id"] == "mp-local"
    assert cfg["dft_pes_reference_structure_file"].endswith("TiS_local.cif")
    assert cfg["dft_pes_reference_use_primitive"] is False


def test_resolve_nanowire_benchmark_source_structure_skips_mp_when_source_artifacts_exist(
    tmp_path: Path, monkeypatch
) -> None:
    model_root = tmp_path / "bench" / "model_b"
    hr_path = tmp_path / "TiS_hr.dat"
    win_path = tmp_path / "TiS.win"
    hr_path.write_text("dummy", encoding="utf-8")
    win_path.write_text("dummy", encoding="utf-8")
    (model_root / "source_artifacts.json").parent.mkdir(parents=True, exist_ok=True)
    (model_root / "source_artifacts.json").write_text(
        json.dumps(
            {
                "hr_dat": str(hr_path),
                "win_path": str(win_path),
                "fermi_ev": 1.23,
            }
        ),
        encoding="utf-8",
    )

    def _boom(_: dict) -> str:
        raise AssertionError("MP-backed structure resolution should be skipped when source artifacts already exist")

    monkeypatch.setattr("wtec.cli._ensure_pes_reference_structure_from_mp", _boom)
    spec = NanowireBenchmarkSpec()
    selected_models = select_benchmark_models(spec)
    resolved = _resolve_nanowire_benchmark_source_structure(
        base_cfg={},
        benchmark_root=tmp_path / "bench",
        selected_models=selected_models,
        material=spec.material,
        default_mp_id=spec.mp_id,
    )
    assert resolved == ""


def test_append_nanowire_benchmark_trace_writes_jsonl(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    _append_nanowire_benchmark_trace(trace_path, "rgf_case_start", tag="d01_e0p0", ok=True)
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "rgf_case_start"
    assert rows[0]["tag"] == "d01_e0p0"
    assert rows[0]["ok"] is True
    assert isinstance(rows[0]["ts"], float)


def test_load_complete_nanowire_kwant_reference_rejects_partial_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "kwant_reference.json"
    path.write_text(
        json.dumps(
            {
                "status": "partial",
                "task_count_expected": 2,
                "task_count_completed": 1,
                "results": [
                    {
                        "thickness_uc": 1,
                        "energy_rel_fermi_ev": -0.2,
                        "energy_abs_ev": 13.4046,
                        "transmission_e2_over_h": 34.0,
                    }
                ],
                "validation": {"status": "partial"},
            }
        ),
        encoding="utf-8",
    )
    spec = NanowireBenchmarkSpec(thicknesses_uc=(1,), energies_ev=(-0.2, -0.1))
    assert _load_complete_nanowire_kwant_reference(path, spec=spec) is None


def test_load_complete_nanowire_kwant_reference_accepts_complete_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "kwant_reference.json"
    payload = {
        "status": "ok",
        "task_count_expected": 2,
        "task_count_completed": 2,
        "results": [
            {
                "thickness_uc": 1,
                "energy_rel_fermi_ev": -0.2,
                "energy_abs_ev": 13.4046,
                "transmission_e2_over_h": 34.0,
            },
            {
                "thickness_uc": 1,
                "energy_rel_fermi_ev": -0.1,
                "energy_abs_ev": 13.5046,
                "transmission_e2_over_h": 38.0,
            },
        ],
        "validation": {"status": "ok"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    spec = NanowireBenchmarkSpec(thicknesses_uc=(1,), energies_ev=(-0.2, -0.1))
    assert _load_complete_nanowire_kwant_reference(path, spec=spec) == payload


def test_load_nanowire_kwant_reference_checkpoint_accepts_partial_payload(tmp_path: Path) -> None:
    path = tmp_path / "kwant_reference.json"
    payload = {
        "status": "partial",
        "task_count_expected": 2,
        "task_count_completed": 1,
        "results": [
            {
                "thickness_uc": 1,
                "energy_rel_fermi_ev": -0.2,
                "energy_abs_ev": 13.4046,
                "transmission_e2_over_h": 34.0,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _load_nanowire_kwant_reference_checkpoint(path) == payload


def test_write_partial_nanowire_axis_artifacts_writes_json_and_markdown(tmp_path: Path) -> None:
    axis_dir = tmp_path / "model_b" / "c"
    (axis_dir / "kwant").mkdir(parents=True)
    (axis_dir / "kwant" / "kwant_payload.json").write_text(
        json.dumps({"fermi_ev": 1.5}),
        encoding="utf-8",
    )
    (axis_dir / "kwant" / "kwant_reference.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "thickness_uc": 5,
                        "energy_rel_fermi_ev": -0.1,
                        "energy_abs_ev": 1.4,
                        "transmission_e2_over_h": 0.75,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    transport_dir = axis_dir / "rgf" / "d05_em0p1" / "transport" / "primary"
    transport_dir.mkdir(parents=True)
    (transport_dir / "transport_payload_primary_001.json").write_text(
        json.dumps(
            {
                "thicknesses": [5],
                "disorder_strengths": [0.0],
                "mfp_lengths": [],
                "energy": 1.4,
            }
        ),
        encoding="utf-8",
    )
    (transport_dir / "transport_result.json").write_text(
        json.dumps({"transport_results": {"thickness_scan": {"0.0": {"G_mean": [0.5]}}}}),
        encoding="utf-8",
    )

    summary = _write_partial_nanowire_axis_artifacts(
        axis_dir=axis_dir,
        spec=NanowireBenchmarkSpec(),
    )

    assert summary["status"] == "failed"
    assert (axis_dir / "comparison_partial.json").exists()
    assert (axis_dir / "comparison_partial.md").exists()


def test_ensure_nanowire_benchmark_rgf_router_ready_reuses_ready_state(monkeypatch) -> None:
    spec = NanowireBenchmarkSpec()
    selected_models = select_benchmark_models(spec)
    ready_state = {
        "rgf": {
            "cluster": {
                "ready": True,
                "binary_id": RGF_BINARY_ID,
                "binary_path": "/remote/wtec_rgf_runner",
                "numerical_status": "phase2_experimental",
            }
        }
    }
    monkeypatch.setattr("wtec.cli._load_init_state", lambda: ready_state)
    monkeypatch.setattr(
        "wtec.cli._prepare_cluster_rgf_router_setup",
        lambda dry_run: (_ for _ in ()).throw(AssertionError("should not prepare router when ready state exists")),
    )
    out = _ensure_nanowire_benchmark_rgf_router_ready(selected_models=selected_models)
    assert out["binary_path"] == "/remote/wtec_rgf_runner"


def test_ensure_nanowire_benchmark_rgf_router_ready_prepares_missing_state(monkeypatch) -> None:
    spec = NanowireBenchmarkSpec()
    selected_models = select_benchmark_models(spec)
    updates: list[dict] = []
    prepared = {
        "ready": True,
        "binary_id": RGF_BINARY_ID,
        "binary_path": "/remote/wtec_rgf_runner",
        "numerical_status": "phase2_experimental",
    }
    monkeypatch.setattr("wtec.cli._load_init_state", lambda: {})
    monkeypatch.setattr("wtec.cli._prepare_cluster_rgf_router_setup", lambda dry_run: prepared)
    monkeypatch.setattr("wtec.cli._update_init_state", lambda patch: updates.append(patch))
    out = _ensure_nanowire_benchmark_rgf_router_ready(selected_models=selected_models)
    assert out is prepared
    assert len(updates) == 1
    assert updates[0]["rgf"]["cluster"]["binary_path"] == "/remote/wtec_rgf_runner"


def test_ensure_nanowire_benchmark_rgf_router_ready_rebuilds_stale_binary(monkeypatch) -> None:
    spec = NanowireBenchmarkSpec()
    selected_models = select_benchmark_models(spec)
    prepared = {
        "ready": True,
        "binary_id": RGF_BINARY_ID,
        "binary_path": "/remote/wtec_rgf_runner",
        "numerical_status": "phase2_experimental",
    }
    monkeypatch.setattr(
        "wtec.cli._load_init_state",
        lambda: {
            "rgf": {
                "cluster": {
                    "ready": True,
                    "binary_id": "wtec_rgf_runner_phase2_v4",
                    "binary_path": "/remote/old_runner",
                    "numerical_status": "phase2_experimental",
                }
            }
        },
    )
    calls: list[bool] = []
    monkeypatch.setattr(
        "wtec.cli._prepare_cluster_rgf_router_setup",
        lambda dry_run: (calls.append(bool(dry_run)), prepared)[1],
    )
    monkeypatch.setattr("wtec.cli._update_init_state", lambda patch: None)
    out = _ensure_nanowire_benchmark_rgf_router_ready(selected_models=selected_models)
    assert out is prepared
    assert calls == [False]


def test_run_kwant_and_rgf_overlap_runs_rgf_while_kwant_waits() -> None:
    kwant_started = Event()
    allow_kwant_finish = Event()
    call_order: list[str] = []

    def _submit_kwant_reference(cancel_event=None):
        call_order.append("kwant_started")
        kwant_started.set()
        assert cancel_event is not None
        assert allow_kwant_finish.wait(timeout=2.0)
        call_order.append("kwant_finished")
        return {"results": []}, {"status": "kwant"}

    def _run_rgf_axis():
        assert kwant_started.wait(timeout=2.0)
        call_order.append("rgf_ran")
        allow_kwant_finish.set()
        return [{"thickness_uc": 1}], [{"status": "rgf"}]

    kwant_result, kwant_job, rgf_rows, rgf_jobs = _run_kwant_and_rgf_overlap(
        submit_kwant_reference=_submit_kwant_reference,
        run_rgf_axis=_run_rgf_axis,
    )

    assert call_order == ["kwant_started", "rgf_ran", "kwant_finished"]
    assert kwant_result == {"results": []}
    assert kwant_job == {"status": "kwant"}
    assert rgf_rows == [{"thickness_uc": 1}]
    assert rgf_jobs == [{"status": "rgf"}]


def test_run_kwant_and_rgf_overlap_cancels_kwant_when_rgf_raises() -> None:
    kwant_started = Event()
    kwant_cancelled = Event()
    call_order: list[str] = []

    def _submit_kwant_reference(cancel_event=None):
        call_order.append("kwant_started")
        kwant_started.set()
        assert cancel_event is not None
        assert cancel_event.wait(timeout=2.0)
        call_order.append("kwant_cancelled")
        kwant_cancelled.set()
        raise CancelledError("kwant cancelled")

    def _run_rgf_axis():
        assert kwant_started.wait(timeout=2.0)
        call_order.append("rgf_failed")
        raise RuntimeError("rgf boom")

    with pytest.raises(RuntimeError, match="rgf boom"):
        _run_kwant_and_rgf_overlap(
            submit_kwant_reference=_submit_kwant_reference,
            run_rgf_axis=_run_rgf_axis,
        )

    assert kwant_cancelled.is_set()
    assert call_order == ["kwant_started", "rgf_failed", "kwant_cancelled"]


def test_run_kwant_and_rgf_overlap_preserves_rgf_rows_on_kwant_runtime_error() -> None:
    def _submit_kwant_reference(cancel_event=None):
        assert cancel_event is not None
        return (_ for _ in ()).throw(RuntimeError("kwant incomplete"))

    def _run_rgf_axis():
        return [{"thickness_uc": 1}], [{"job_id": "rgf"}]

    with pytest.raises(_KwantOverlapError, match="kwant incomplete") as excinfo:
        _run_kwant_and_rgf_overlap(
            submit_kwant_reference=_submit_kwant_reference,
            run_rgf_axis=_run_rgf_axis,
        )

    assert excinfo.value.rgf_rows == [{"thickness_uc": 1}]
    assert excinfo.value.rgf_jobs == [{"job_id": "rgf"}]


def test_run_kwant_and_rgf_overlap_preserves_partial_summary_on_rgf_partial_failure() -> None:
    kwant_started = Event()
    kwant_cancelled = Event()
    partial_summary = {
        "status": "failed",
        "overlap_points": 1,
        "comparison": {
            "status": "failed",
            "checked_points": 1,
            "max_abs_err": 0.1,
            "max_rel_err": 0.01,
            "failures": [],
        },
    }

    def _submit_kwant_reference(cancel_event=None):
        kwant_started.set()
        assert cancel_event is not None
        assert cancel_event.wait(timeout=2.0)
        kwant_cancelled.set()
        raise CancelledError("kwant cancelled")

    def _run_rgf_axis():
        assert kwant_started.wait(timeout=2.0)
        raise _PartialOverlapFailure(
            "overlap failed",
            partial_summary=partial_summary,
            rgf_rows=[{"thickness_uc": 1}],
            rgf_jobs=[{"job_id": "rgf"}],
        )

    with pytest.raises(_KwantOverlapError, match="overlap failed") as excinfo:
        _run_kwant_and_rgf_overlap(
            submit_kwant_reference=_submit_kwant_reference,
            run_rgf_axis=_run_rgf_axis,
        )

    assert kwant_cancelled.is_set()
    assert excinfo.value.rgf_rows == [{"thickness_uc": 1}]
    assert excinfo.value.rgf_jobs == [{"job_id": "rgf"}]
    assert excinfo.value.partial_summary == partial_summary


def test_benchmark_transport_writes_failed_summary_from_partial_overlap(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "run_small.json"
    config_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "bench"
    hr_path = tmp_path / "TiS_hr.dat"
    win_path = tmp_path / "TiS.win"
    hr_path.write_text("dummy", encoding="utf-8")
    win_path.write_text("dummy", encoding="utf-8")

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

    def _fake_submit_kwant_reference(**kwargs):
        benchmark_dir = Path(kwargs["benchmark_dir"])
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        (benchmark_dir / "kwant_payload.json").write_text(
            json.dumps({"fermi_ev": 13.6046}),
            encoding="utf-8",
        )
        (benchmark_dir / "kwant_reference.json").write_text(
            json.dumps(
                {
                    "status": "partial",
                    "task_count_expected": 2,
                    "task_count_completed": 1,
                    "results": [
                        {
                            "thickness_uc": 1,
                            "energy_rel_fermi_ev": 0.0,
                            "energy_abs_ev": 13.6046,
                            "transmission_e2_over_h": 40.0,
                        }
                    ],
                    "validation": {"status": "partial"},
                }
            ),
            encoding="utf-8",
        )
        raise RuntimeError("kwant incomplete after retry budget")

    def _fake_run_rgf_axis(**kwargs):
        axis_dir = Path(kwargs["axis_dir"])
        transport_dir = axis_dir / "rgf" / "d01_e0p0" / "transport" / "primary"
        transport_dir.mkdir(parents=True, exist_ok=True)
        (transport_dir / "transport_payload_primary_001.json").write_text(
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
        (transport_dir / "transport_result.json").write_text(
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
        return (
            [
                {
                    "thickness_uc": 1,
                    "energy_rel_fermi_ev": 0.0,
                    "energy_abs_ev": 13.6046,
                    "transmission_e2_over_h": 29.88663916904678,
                }
            ],
            [{"job_id": "60252"}],
        )

    monkeypatch.setattr(
        "wtec.transport.nanowire_benchmark_cluster.submit_kwant_nanowire_reference",
        _fake_submit_kwant_reference,
    )
    monkeypatch.setattr("wtec.cli._run_rgf_benchmark_axis", _fake_run_rgf_axis)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "benchmark-transport",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--queue",
            "g4",
            "--walltime",
            "01:00:00",
        ],
    )

    assert result.exit_code == 1
    summary_path = output_dir / "benchmark_summary.json"
    partial_path = output_dir / "model_b" / "c" / "comparison_partial.json"
    assert summary_path.exists()
    assert partial_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert "model_b:c:rgf_partial" in summary["failed_targets"]
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    assert partial["status"] == "failed"
    assert partial["comparison"]["checked_points"] == 1


def test_benchmark_transport_writes_failed_summary_from_live_partial_overlap(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "run_small.json"
    config_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "bench"
    hr_path = tmp_path / "TiS_hr.dat"
    win_path = tmp_path / "TiS.win"
    hr_path.write_text("dummy", encoding="utf-8")
    win_path.write_text("dummy", encoding="utf-8")

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
    partial_summary = {
        "status": "failed",
        "kwant": {
            "rows": [
                {
                    "thickness_uc": 1,
                    "energy_rel_fermi_ev": 0.0,
                    "energy_abs_ev": 13.6046,
                    "transmission_e2_over_h": 40.0,
                    "source_kind": "kwant_log",
                    "source_path": "wtec_job.log",
                }
            ],
            "row_count": 1,
            "validation": {"status": "partial"},
        },
        "rgf": {
            "rows": [
                {
                    "thickness_uc": 1,
                    "energy_rel_fermi_ev": 0.0,
                    "energy_abs_ev": 13.6046,
                    "transmission_e2_over_h": 29.88663916904678,
                    "source_kind": "rgf_transport_result",
                    "source_path": "transport_result.json",
                }
            ],
            "row_count": 1,
        },
        "overlap_points": 1,
        "comparison": {
            "status": "failed",
            "checked_points": 1,
            "max_abs_err": 10.113360830953447,
            "max_rel_err": 0.2528340207738347,
            "failures": [],
        },
        "missing_in_rgf": [],
        "missing_in_kwant": [],
    }

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

    def _fake_submit_kwant_reference(cancel_event=None, **_kwargs):
        assert cancel_event is not None
        assert cancel_event.wait(timeout=2.0)
        raise CancelledError("kwant cancelled")

    def _fake_run_rgf_axis(**_kwargs):
        raise _PartialOverlapFailure(
            "Current overlap already proves benchmark failure for model_b:c",
            partial_summary=partial_summary,
            rgf_rows=[partial_summary["rgf"]["rows"][0]],
            rgf_jobs=[{"job_id": "60252"}],
        )

    monkeypatch.setattr(
        "wtec.transport.nanowire_benchmark_cluster.submit_kwant_nanowire_reference",
        _fake_submit_kwant_reference,
    )
    monkeypatch.setattr("wtec.cli._run_rgf_benchmark_axis", _fake_run_rgf_axis)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "benchmark-transport",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--queue",
            "g4",
            "--walltime",
            "01:00:00",
        ],
    )

    assert result.exit_code == 1
    summary_path = output_dir / "benchmark_summary.json"
    partial_path = output_dir / "model_b" / "c" / "comparison_partial.json"
    assert summary_path.exists()
    assert partial_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert "model_b:c:rgf_partial" in summary["failed_targets"]
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    assert partial["status"] == "failed"
    assert partial["kwant"]["rows"][0]["source_kind"] == "kwant_log"


def test_benchmark_transport_uses_existing_log_overlap_before_launch(tmp_path: Path, monkeypatch) -> None:
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

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "benchmark-transport",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--queue",
            "g4",
            "--walltime",
            "01:00:00",
        ],
    )

    assert result.exit_code == 1
    summary_path = output_dir / "benchmark_summary.json"
    partial_path = axis_dir / "comparison_partial.json"
    assert summary_path.exists()
    assert partial_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert "model_b:c:rgf_partial" in summary["failed_targets"]
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    assert partial["status"] == "failed"
    assert partial["comparison"]["checked_points"] == 1
    assert partial["kwant"]["rows"][0]["source_kind"] == "kwant_log"


def test_run_rgf_benchmark_axis_requests_exact_sigma_internal_mode(
    tmp_path: Path, monkeypatch
) -> None:
    seen_cfgs: list[dict] = []

    class _FakeWorkflow:
        def __init__(self, cfg: dict) -> None:
            self.cfg = cfg

        def _stage_transport_rgf_qsub(self, hr_path: Path, label: str = "primary"):
            seen_cfgs.append(dict(self.cfg))
            return (
                {
                    "thickness_scan": {"0.0": {"G_mean": [12.5]}},
                },
                {"job_id": "12345"},
            )

    monkeypatch.setattr(
        "wtec.workflow.orchestrator.TopoSlabWorkflow.from_config",
        lambda cfg: _FakeWorkflow(cfg),
    )
    rows, jobs = _run_rgf_benchmark_axis(
        source_cfg={"material": "TiS"},
        axis_dir=tmp_path / "axis",
        canonical=SimpleNamespace(hr_dat_path=str(tmp_path / "toy_hr.dat")),
        model=SimpleNamespace(key="model_b"),
        axis="c",
        spec=SimpleNamespace(thicknesses_uc=(1,), energies_ev=(-0.2,), fixed_width_uc=13),
        fermi_ev_f=13.6046,
        length_uc=24,
        transport_nodes=1,
        live_log=False,
        log_poll_interval=5,
        stale_log_seconds=300,
        required_exact_eta=1.0e-8,
    )

    assert rows == [
        {
            "thickness_uc": 1,
            "energy_rel_fermi_ev": -0.2,
            "energy_abs_ev": 13.4046,
            "transmission_e2_over_h": 12.5,
        }
    ]
    assert jobs == [{"job_id": "12345"}]
    assert seen_cfgs and seen_cfgs[0]["_transport_rgf_internal_sigma_mode"] == "kwant_exact"
    assert seen_cfgs[0]["transport_rgf_eta"] == pytest.approx(1.0e-8)
    assert seen_cfgs[0]["reuse_transport_results"] is True


def test_run_rgf_benchmark_axis_stops_when_partial_overlap_fails(tmp_path: Path, monkeypatch) -> None:
    class _FakeWorkflow:
        def __init__(self, cfg: dict) -> None:
            self.cfg = cfg

        def _stage_transport_rgf_qsub(self, hr_path: Path, label: str = "primary"):
            return (
                {
                    "thickness_scan": {"0.0": {"G_mean": [12.5]}},
                },
                {"job_id": "12345"},
            )

    monkeypatch.setattr(
        "wtec.workflow.orchestrator.TopoSlabWorkflow.from_config",
        lambda cfg: _FakeWorkflow(cfg),
    )
    monkeypatch.setattr(
        "wtec.cli._load_existing_nanowire_partial_overlap",
        lambda **_kwargs: {
            "status": "failed",
            "overlap_points": 1,
            "comparison": {
                "status": "failed",
                "checked_points": 1,
                "max_abs_err": 0.1,
                "max_rel_err": 0.01,
                "failures": [],
            },
        },
    )

    with pytest.raises(_PartialOverlapFailure, match="Current overlap already proves benchmark failure"):
        _run_rgf_benchmark_axis(
            source_cfg={"material": "TiS"},
            axis_dir=tmp_path / "axis",
            canonical=SimpleNamespace(hr_dat_path=str(tmp_path / "toy_hr.dat")),
            model=SimpleNamespace(key="model_b"),
            axis="c",
            spec=SimpleNamespace(thicknesses_uc=(1,), energies_ev=(-0.2,), fixed_width_uc=13),
            fermi_ev_f=13.6046,
            length_uc=24,
            transport_nodes=1,
            live_log=False,
            log_poll_interval=5,
            stale_log_seconds=300,
        )


def test_run_rgf_benchmark_axis_can_ignore_partial_overlap_failure(tmp_path: Path, monkeypatch) -> None:
    class _FakeWorkflow:
        def __init__(self, cfg: dict) -> None:
            self.cfg = cfg

        def _stage_transport_rgf_qsub(self, hr_path: Path, label: str = "primary"):
            return (
                {
                    "thickness_scan": {"0.0": {"G_mean": [12.5]}},
                },
                {"job_id": "12345"},
            )

    monkeypatch.setattr(
        "wtec.workflow.orchestrator.TopoSlabWorkflow.from_config",
        lambda cfg: _FakeWorkflow(cfg),
    )
    monkeypatch.setattr(
        "wtec.cli._load_existing_nanowire_partial_overlap",
        lambda **_kwargs: {
            "status": "failed",
            "overlap_points": 1,
            "comparison": {
                "status": "failed",
                "checked_points": 1,
                "max_abs_err": 0.1,
                "max_rel_err": 0.01,
                "failures": [],
            },
        },
    )

    rows, jobs = _run_rgf_benchmark_axis(
        source_cfg={"material": "TiS"},
        axis_dir=tmp_path / "axis",
        canonical=SimpleNamespace(hr_dat_path=str(tmp_path / "toy_hr.dat")),
        model=SimpleNamespace(key="model_b"),
        axis="c",
        spec=SimpleNamespace(thicknesses_uc=(1,), energies_ev=(-0.2,), fixed_width_uc=13),
        fermi_ev_f=13.6046,
        length_uc=24,
        transport_nodes=1,
        live_log=False,
        log_poll_interval=5,
        stale_log_seconds=300,
        stop_on_partial_failure=False,
    )

    assert rows == [
        {
            "thickness_uc": 1,
            "energy_rel_fermi_ev": -0.2,
            "energy_abs_ev": 13.4046,
            "transmission_e2_over_h": 12.5,
        }
    ]
    assert jobs == [{"job_id": "12345"}]


def test_benchmark_transport_rgf_writes_sequential_summary(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "run_small.json"
    config_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "bench"
    hr_path = tmp_path / "TiS_hr.dat"
    win_path = tmp_path / "TiS.win"
    hr_path.write_text("dummy", encoding="utf-8")
    win_path.write_text("dummy", encoding="utf-8")

    seen_kwargs: list[dict[str, object]] = []

    monkeypatch.setattr("wtec.cli._load_runtime_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("wtec.cli._load_run_config", lambda _path: {"n_nodes": 1})
    monkeypatch.setattr("wtec.cli._ensure_nanowire_benchmark_rgf_router_ready", lambda **_kwargs: {})
    monkeypatch.setattr("wtec.cli._resolve_nanowire_benchmark_source_structure", lambda **_kwargs: "")
    monkeypatch.setattr("wtec.cli._build_nanowire_benchmark_source_seed", lambda **_kwargs: {})
    monkeypatch.setattr("wtec.cli._load_benchmark_source_resume", lambda _root: (hr_path, win_path, 13.6046))
    monkeypatch.setattr(
        "wtec.transport.nanowire_benchmark.prepare_canonicalized_inputs",
        lambda **_kwargs: SimpleNamespace(hr_dat_path=str(hr_path)),
    )
    monkeypatch.setattr("wtec.wannier.parser.read_hr_dat", lambda _path: object())
    monkeypatch.setattr("wtec.rgf.effective_principal_layer_width", lambda *args, **kwargs: 1)
    monkeypatch.setattr("wtec.transport.nanowire_benchmark.compute_length_uc", lambda *_args, **_kwargs: 24)

    def _fake_run_rgf_axis(**kwargs):
        seen_kwargs.append(dict(kwargs))
        return (
            [
                {
                    "thickness_uc": 1,
                    "energy_rel_fermi_ev": 0.0,
                    "energy_abs_ev": 13.6046,
                    "transmission_e2_over_h": 39.99994685099053,
                }
            ],
            [{"job_id": "60279"}],
        )

    monkeypatch.setattr("wtec.cli._run_rgf_benchmark_axis", _fake_run_rgf_axis)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "benchmark-transport-rgf",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--queue",
            "g4",
            "--walltime",
            "01:00:00",
            "--thickness",
            "1",
            "--energy",
            "0.0",
            "--no-compare-existing-kwant",
        ],
    )

    assert result.exit_code == 0
    summary_path = output_dir / "rgf_sequential_summary.json"
    axis_dir = output_dir / "model_b" / "c"
    assert summary_path.exists()
    assert (axis_dir / "rgf_raw.json").exists()
    assert (axis_dir / "rgf_fit.json").exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["execution_mode"] == "sequential_rgf_only"
    axis_summary = summary["models"]["model_b"]["axes"]["c"]
    assert axis_summary["required_exact_eta"] == pytest.approx(1.0e-8)
    assert axis_summary["rgf_points"] == 1
    assert seen_kwargs and seen_kwargs[0]["stop_on_partial_failure"] is False
    assert seen_kwargs[0]["source_cfg"]["transport_walltime"] == "01:00:00"


def test_benchmark_transport_passes_requested_walltime_to_rgf(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "run_small.json"
    config_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "bench"
    hr_path = tmp_path / "TiS_hr.dat"
    win_path = tmp_path / "TiS.win"
    hr_path.write_text("dummy", encoding="utf-8")
    win_path.write_text("dummy", encoding="utf-8")

    spec = SimpleNamespace(
        mp_id="mp-1018028",
        material="TiS",
        axes=("c",),
        energies_ev=(0.0,),
        thicknesses_uc=(1, 3),
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

    seen: dict[str, object] = {}

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

    def _fake_run_rgf_axis(**kwargs):
        seen["transport_walltime"] = kwargs["source_cfg"]["transport_walltime"]
        return (
            [
                {
                    "thickness_uc": 1,
                    "energy_rel_fermi_ev": 0.0,
                    "energy_abs_ev": 13.6046,
                    "transmission_e2_over_h": 39.99994685099053,
                },
                {
                    "thickness_uc": 3,
                    "energy_rel_fermi_ev": 0.0,
                    "energy_abs_ev": 13.6046,
                    "transmission_e2_over_h": 21.99991710570342,
                },
            ],
            [{"job_id": "60279"}],
        )

    def _fake_overlap(*, submit_kwant_reference, run_rgf_axis):
        rows, jobs = run_rgf_axis()
        return (
            {
                "status": "complete",
                "task_count_expected": 1,
                "task_count_completed": 1,
                "results": [
                    {
                        "thickness_uc": 1,
                        "energy_rel_fermi_ev": 0.0,
                        "energy_abs_ev": 13.6046,
                        "transmission_e2_over_h": 40.0,
                    },
                    {
                        "thickness_uc": 3,
                        "energy_rel_fermi_ev": 0.0,
                        "energy_abs_ev": 13.6046,
                        "transmission_e2_over_h": 22.0,
                    },
                ],
                "validation": {"status": "ok"},
            },
            {"job_id": "kwant-1"},
            rows,
            jobs,
        )

    monkeypatch.setattr("wtec.cli._run_rgf_benchmark_axis", _fake_run_rgf_axis)
    monkeypatch.setattr("wtec.cli._run_kwant_and_rgf_overlap", _fake_overlap)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "benchmark-transport",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--queue",
            "g4",
            "--walltime",
            "10:00:00",
        ],
    )

    assert result.exit_code == 0
    assert seen["transport_walltime"] == "10:00:00"


def test_axis_permutation_maps_expected_axes() -> None:
    assert axis_permutation("a") == (0, 1, 2)
    assert axis_permutation("c") == (2, 0, 1)


def test_canonicalize_hopping_data_for_c_axis() -> None:
    hd = _toy_hd()
    lv = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    hd2, lv2, perm = canonicalize_hopping_data(hd, lv, axis="c")
    assert perm == (2, 0, 1)
    assert np.array_equal(hd2.r_vectors[1], np.asarray([0, 1, 0]))
    assert np.array_equal(hd2.r_vectors[2], np.asarray([0, 0, 1]))
    assert np.array_equal(hd2.r_vectors[3], np.asarray([1, 0, 0]))
    assert np.array_equal(lv2[0], lv[2])
    assert np.array_equal(lv2[1], lv[0])
    assert np.array_equal(lv2[2], lv[1])


def test_prepare_canonicalized_inputs_writes_hr_and_win(tmp_path: Path) -> None:
    hd = _toy_hd()
    hr_path = tmp_path / "toy_hr.dat"
    win_path = tmp_path / "toy.win"
    write_hr_dat(hr_path, hd, header="toy")
    win_path.write_text(
        "begin unit_cell_cart\n"
        "ang\n"
        "1 0 0\n"
        "0 2 0\n"
        "0 0 3\n"
        "end unit_cell_cart\n",
        encoding="utf-8",
    )
    out = prepare_canonicalized_inputs(
        hr_dat_path=hr_path,
        win_path=win_path,
        axis="c",
        out_dir=tmp_path / "canon",
        seedname="toy",
    )
    assert isinstance(out, CanonicalizedNanowireInput)
    hd2 = read_hr_dat(out.hr_dat_path)
    lv2 = _parse_lattice_from_win(out.win_path)
    assert np.array_equal(hd2.r_vectors[3], np.asarray([1, 0, 0]))
    assert np.array_equal(lv2[0], np.asarray([0.0, 0.0, 3.0]))


def test_select_monotonic_thickness_subsequence() -> None:
    rows = []
    energies = (-0.2, -0.1, 0.0, 0.1, 0.2)
    values = {
        10: [1.20, 1.10, 1.00, 0.90, 0.80],
        9: [1.10, 1.00, 0.90, 0.80, 0.70],
        8: [1.15, 0.95, 0.85, 0.75, 0.65],
        7: [0.95, 0.85, 0.75, 0.65, 0.55],
        6: [0.80, 0.70, 0.60, 0.50, 0.40],
        5: [1.60, 0.60, 0.50, 0.40, 0.30],
    }
    for thickness_uc, vals in values.items():
        for e, t in zip(energies, vals):
            rows.append(
                {
                    "thickness_uc": thickness_uc,
                    "energy_rel_fermi_ev": e,
                    "transmission_e2_over_h": t,
                }
            )
    out = select_monotonic_thickness_subsequence(
        rows,
        energies_ev=energies,
        candidate_thicknesses=[10, 9, 8, 7, 6, 5],
        min_points=4,
        max_transmission_e2_over_h=1.5,
    )
    assert out["status"] == "ok"
    assert out["retained_thicknesses"] == [10, 9, 7, 6]


def test_select_monotonic_thickness_subsequence_without_cap() -> None:
    rows = []
    energies = (-0.2, -0.1, 0.0, 0.1, 0.2)
    values = {
        4: [2.0, 1.9, 1.8, 1.7, 1.6],
        3: [1.5, 1.4, 1.3, 1.2, 1.1],
        2: [1.0, 0.9, 0.8, 0.7, 0.6],
        1: [0.5, 0.4, 0.3, 0.2, 0.1],
    }
    for thickness_uc, vals in values.items():
        for e, t in zip(energies, vals):
            rows.append(
                {
                    "thickness_uc": thickness_uc,
                    "energy_rel_fermi_ev": e,
                    "transmission_e2_over_h": t,
                }
            )
    out = select_monotonic_thickness_subsequence(
        rows,
        energies_ev=energies,
        candidate_thicknesses=[4, 3, 2, 1],
        min_points=4,
        max_transmission_e2_over_h=None,
    )
    assert out["status"] == "ok"
    assert out["retained_thicknesses"] == [4, 3, 2, 1]


def test_compare_reference_and_rgf() -> None:
    ref = [
        {"thickness_uc": 6, "energy_rel_fermi_ev": -0.2, "transmission_e2_over_h": 0.5},
        {"thickness_uc": 6, "energy_rel_fermi_ev": 0.2, "transmission_e2_over_h": 0.4},
    ]
    got = [
        {"thickness_uc": 6, "energy_rel_fermi_ev": -0.2, "transmission_e2_over_h": 0.500001},
        {"thickness_uc": 6, "energy_rel_fermi_ev": 0.2, "transmission_e2_over_h": 0.399999},
    ]
    out = compare_reference_and_rgf(ref, got)
    assert out.status == "ok"
    assert out.checked_points == 2
