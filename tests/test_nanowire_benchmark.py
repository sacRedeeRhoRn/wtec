from __future__ import annotations

from threading import Event
from pathlib import Path

import numpy as np

from wtec.cli import _build_tis_benchmark_source_cfg, _run_kwant_and_rgf_overlap
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
