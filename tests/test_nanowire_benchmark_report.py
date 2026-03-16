from __future__ import annotations

import csv
import json
from pathlib import Path

from wtec.transport.nanowire_benchmark_report import (
    generate_nanowire_benchmark_report,
    scan_kwant_runtime_bounds,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_minimal_benchmark_root(tmp_path: Path, *, eta: float = 1.0e-8) -> Path:
    benchmark_root = tmp_path / "bench"
    axis_root = benchmark_root / "model_b" / "c"
    kwant_dir = axis_root / "kwant"
    rgf_dir = axis_root / "rgf" / "d01_e0p0" / "transport" / "primary"

    _write_json(
        kwant_dir / "kwant_payload.json",
        {
            "fermi_ev": 13.6046,
        },
    )
    _write_json(
        kwant_dir / "kwant_reference.json",
        {
            "status": "partial",
            "task_count_expected": 35,
            "task_count_completed": 1,
            "thicknesses": [1, 3],
            "energies_rel_fermi_ev": [0.0, 0.1],
            "fermi_ev": 13.6046,
            "results": [
                {
                    "thickness_uc": 1,
                    "energy_rel_fermi_ev": 0.0,
                    "energy_abs_ev": 13.6046,
                    "transmission_e2_over_h": 40.0,
                }
            ],
            "validation": {"status": "partial"},
        },
    )
    _write_text(
        kwant_dir / "wtec_job.log",
        "\n".join(
            [
                "[kwant-bench][rank=0] start thickness_uc=1 energy_abs_ev=13.604600",
                "[kwant-bench][rank=0] heartbeat thickness_uc=1 energy_abs_ev=13.604600 elapsed_s=60.0",
                "[kwant-bench][rank=0] done thickness_uc=1 energy_abs_ev=13.604600 transmission=40.000000000000",
            ]
        )
        + "\n",
    )
    _write_text(
        kwant_dir / "kwant_reference.pbs",
        "\n".join(
            [
                "#!/bin/bash",
                "#PBS -q g4",
                "# IMPORTANT: all parallel execution uses mpirun backend.",
                "# Fork-based launchstyle is FORBIDDEN in this package.",
                "mpirun -np 16 --bind-to none python3 -m wtec.transport.kwant_nanowire_benchmark kwant_payload.json kwant_reference.json",
            ]
        )
        + "\n",
    )

    _write_json(
        rgf_dir / "transport_payload_primary_001.json",
        {
            "thicknesses": [1],
            "disorder_strengths": [0.0],
            "mfp_lengths": [],
            "energy": 13.6046,
            "eta": eta,
            "transport_rgf_mode": "full_finite",
            "sigma_left_path": "sigma_left.bin",
            "sigma_right_path": "sigma_right.bin",
        },
    )
    _write_json(
        rgf_dir / "transport_result.json",
        {
            "runtime_cert": {
                "queue": "g4",
                "mode": "full_finite",
                "mpi_size": 1,
                "omp_threads": 64,
                "wall_seconds": 1.25,
                "full_finite_sigma_source": "kwant_exact",
            },
            "transport_results": {
                "meta": {
                    "rgf_full_finite_sigma_source": "kwant_exact",
                },
                "thickness_scan": {
                    "0.0": {
                        "G_mean": [39.99995],
                    }
                },
            },
            "transport_results_raw": {
                "eta": eta,
            },
        },
    )
    _write_text(
        rgf_dir / "transport_rgf_primary_primary_20260315T000000_demo.pbs",
        "\n".join(
            [
                "#!/bin/bash",
                "# IMPORTANT: all parallel execution uses mpirun backend.",
                "# Fork-based launchstyle is FORBIDDEN in this package.",
                "mpirun -np 1 --bind-to none /tmp/wtec_rgf_runner payload.json transport_result.json",
            ]
        )
        + "\n",
    )
    return benchmark_root


def test_scan_kwant_runtime_bounds_tracks_completed_and_in_progress_points(tmp_path: Path) -> None:
    log_path = tmp_path / "wtec_job.log"
    _write_text(
        log_path,
        "\n".join(
            [
                "[kwant-bench][rank=0] start thickness_uc=1 energy_abs_ev=13.604600",
                "[kwant-bench][rank=0] heartbeat thickness_uc=1 energy_abs_ev=13.604600 elapsed_s=60.0",
                "[kwant-bench][rank=0] heartbeat thickness_uc=1 energy_abs_ev=13.604600 elapsed_s=120.0",
                "[kwant-bench][rank=0] done thickness_uc=1 energy_abs_ev=13.604600 transmission=40.000000000000",
                "[kwant-bench][rank=1] start thickness_uc=3 energy_abs_ev=13.704600",
                "[kwant-bench][rank=1] heartbeat thickness_uc=3 energy_abs_ev=13.704600 elapsed_s=60.0",
            ]
        )
        + "\n",
    )

    out = scan_kwant_runtime_bounds(log_path)
    assert out["heartbeat_interval_seconds"] == 60.0
    by_key = {
        (row["thickness_uc"], row["energy_abs_ev"]): row
        for row in out["points"]
    }
    done_row = by_key[(1, 13.6046)]
    assert done_row["completed"] is True
    assert done_row["elapsed_lower_bound_s"] == 120.0
    assert done_row["elapsed_upper_bound_s"] == 180.0

    live_row = by_key[(3, 13.7046)]
    assert live_row["completed"] is False
    assert live_row["elapsed_lower_bound_s"] == 60.0
    assert live_row["elapsed_upper_bound_s"] is None


def test_generate_nanowire_benchmark_report_writes_bundle(tmp_path: Path) -> None:
    benchmark_root = _build_minimal_benchmark_root(tmp_path, eta=1.0e-8)
    out_dir = tmp_path / "report"

    report = generate_nanowire_benchmark_report(
        benchmark_root=benchmark_root,
        out_dir=out_dir,
        required_exact_eta=1.0e-8,
    )

    assert report["current_evidence_status"] == "ok"
    assert report["coverage"]["overlap_points"] == 1
    assert report["speed"]["completed_overlap_points"] == 1
    assert report["acceptance"]["current_overlap_validation"] == "pass"
    assert report["acceptance"]["runtime_path_proof"] == "pass"

    expected_files = [
        "report.md",
        "report.json",
        "overlap_points.csv",
        "all_points.csv",
        "speed_summary.csv",
        "coverage_matrix.png",
        "transmission_all_points.png",
        "transmission_overlap_comparison.png",
        "error_overlap.png",
        "runtime_speed_bounds.png",
    ]
    for name in expected_files:
        assert (out_dir / name).exists(), name

    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "![Coverage matrix](coverage_matrix.png)" in report_md
    assert "speedup_lower_bound_geomean" in report_md

    with (out_dir / "speed_summary.csv").open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert float(rows[0]["speedup_lower_bound"]) > 40.0


def test_generate_nanowire_benchmark_report_respects_exact_eta_filter(tmp_path: Path) -> None:
    benchmark_root = _build_minimal_benchmark_root(tmp_path, eta=1.0e-6)
    out_dir = tmp_path / "report"

    report = generate_nanowire_benchmark_report(
        benchmark_root=benchmark_root,
        out_dir=out_dir,
        required_exact_eta=1.0e-8,
    )

    assert report["current_evidence_status"] == "missing_rgf_points"
    assert report["coverage"]["overlap_points"] == 0
    assert report["comparison"]["status"] is None
    assert (out_dir / "coverage_matrix.png").exists()
    assert (out_dir / "report.json").exists()
