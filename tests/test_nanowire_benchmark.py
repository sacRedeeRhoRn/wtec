from __future__ import annotations

import json
from threading import Event
from pathlib import Path

import numpy as np

from wtec.cli import (
    _append_nanowire_benchmark_trace,
    _build_nanowire_benchmark_source_seed,
    _build_tis_benchmark_source_cfg,
    _ensure_nanowire_benchmark_rgf_router_ready,
    _resolve_nanowire_benchmark_source_structure,
    _run_kwant_and_rgf_overlap,
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

    def _submit_kwant_reference():
        call_order.append("kwant_started")
        kwant_started.set()
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
