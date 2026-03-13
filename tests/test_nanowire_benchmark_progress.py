from __future__ import annotations

import json
from pathlib import Path

from wtec.transport.nanowire_benchmark_progress import (
    compare_partial_benchmark_progress,
    scan_partial_kwant_results,
    scan_partial_rgf_results,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_scan_partial_kwant_results_from_log_only(tmp_path: Path) -> None:
    kwant_dir = tmp_path / "kwant"
    _write_json(
        kwant_dir / "kwant_payload.json",
        {
            "fermi_ev": 1.5,
            "energies_rel_fermi_ev": [-0.1, 0.0],
            "thicknesses": [5, 7],
        },
    )
    (kwant_dir / "wtec_job.log").write_text(
        "\n".join(
            [
                "[kwant-bench][rank=0] start thickness_uc=5 energy_abs_ev=1.400000",
                "[kwant-bench][rank=0] done thickness_uc=5 energy_abs_ev=1.400000 transmission=0.750000000000",
                "[kwant-bench][rank=0] done thickness_uc=7 energy_abs_ev=1.500000 transmission=0.550000000000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = scan_partial_kwant_results(kwant_dir)
    assert out["row_count"] == 2
    assert out["rows"][0]["thickness_uc"] == 5
    assert out["rows"][0]["energy_abs_ev"] == 1.4
    assert out["rows"][0]["energy_rel_fermi_ev"] == -0.1
    assert out["rows"][0]["transmission_e2_over_h"] == 0.75
    assert out["rows"][0]["source_kind"] == "kwant_log"


def test_scan_partial_rgf_results_from_progress_only(tmp_path: Path) -> None:
    transport_dir = tmp_path / "rgf" / "d05_em0p1" / "transport" / "primary"
    _write_json(
        transport_dir / "transport_payload_primary_001.json",
        {
            "thicknesses": [5],
            "disorder_strengths": [0.0],
            "mfp_lengths": [],
            "energy": 1.4,
            "transport_rgf_mode": "full_finite",
            "sigma_left_path": "sigma_left.bin",
            "sigma_right_path": "sigma_right.bin",
        },
    )
    (transport_dir / "transport_progress_primary_001.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "worker_start"}),
                json.dumps({"event": "native_point_done", "G": 0.74}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = scan_partial_rgf_results(tmp_path / "rgf", fermi_ev=1.5)
    assert out["row_count"] == 1
    row = out["rows"][0]
    assert row["thickness_uc"] == 5
    assert row["energy_abs_ev"] == 1.4
    assert row["energy_rel_fermi_ev"] == -0.1
    assert row["transmission_e2_over_h"] == 0.74
    assert row["source_kind"] == "rgf_progress"


def test_scan_partial_rgf_results_prefers_transport_result_over_progress(tmp_path: Path) -> None:
    transport_dir = tmp_path / "rgf" / "d05_em0p1" / "transport" / "primary"
    _write_json(
        transport_dir / "transport_payload_primary_001.json",
        {
            "thicknesses": [5],
            "disorder_strengths": [0.0],
            "mfp_lengths": [],
            "energy": 1.4,
            "transport_rgf_mode": "full_finite",
            "sigma_left_path": "sigma_left.bin",
            "sigma_right_path": "sigma_right.bin",
        },
    )
    (transport_dir / "transport_progress_primary_001.jsonl").write_text(
        json.dumps({"event": "native_point_done", "G": 0.74}) + "\n",
        encoding="utf-8",
    )
    _write_json(
        transport_dir / "transport_result.json",
        {
            "runtime_cert": {
                "full_finite_sigma_source": "kwant_exact",
            },
            "transport_results": {
                "meta": {
                    "rgf_full_finite_sigma_source": "kwant_exact",
                },
                "thickness_scan": {
                    "0.0": {
                        "G_mean": [0.76],
                    }
                }
            }
        },
    )

    out = scan_partial_rgf_results(tmp_path / "rgf", fermi_ev=1.5)
    assert out["row_count"] == 1
    row = out["rows"][0]
    assert row["transmission_e2_over_h"] == 0.76
    assert row["source_kind"] == "rgf_transport_result"


def test_scan_partial_rgf_results_skips_full_finite_rows_without_exact_sigma_contract(tmp_path: Path) -> None:
    transport_dir = tmp_path / "rgf" / "d05_em0p1" / "transport" / "primary"
    payload_path = transport_dir / "transport_payload_primary_001.json"
    _write_json(
        payload_path,
        {
            "thicknesses": [5],
            "disorder_strengths": [0.0],
            "mfp_lengths": [],
            "energy": 1.4,
            "transport_rgf_mode": "full_finite",
        },
    )
    _write_json(
        transport_dir / "transport_result.json",
        {
            "transport_results": {
                "meta": {
                    "rgf_full_finite_sigma_backend": "native",
                },
                "thickness_scan": {
                    "0.0": {
                        "G_mean": [0.76],
                    }
                },
            },
            "runtime_cert": {
                "mode": "full_finite",
            },
        },
    )

    out = scan_partial_rgf_results(tmp_path / "rgf", fermi_ev=1.5)
    assert out["row_count"] == 0
    assert str(payload_path) in out["skipped_payloads"]


def test_scan_partial_rgf_results_skips_exact_sigma_rows_with_wrong_eta(tmp_path: Path) -> None:
    transport_dir = tmp_path / "rgf" / "d05_em0p1" / "transport" / "primary"
    payload_path = transport_dir / "transport_payload_primary_001.json"
    _write_json(
        payload_path,
        {
            "thicknesses": [5],
            "disorder_strengths": [0.0],
            "mfp_lengths": [],
            "energy": 1.4,
            "eta": 1.0e-6,
            "transport_rgf_mode": "full_finite",
            "sigma_left_path": "sigma_left.bin",
            "sigma_right_path": "sigma_right.bin",
        },
    )
    _write_json(
        transport_dir / "transport_result.json",
        {
            "runtime_cert": {
                "full_finite_sigma_source": "kwant_exact",
            },
            "transport_results": {
                "meta": {
                    "rgf_full_finite_sigma_source": "kwant_exact",
                },
                "thickness_scan": {
                    "0.0": {
                        "G_mean": [0.76],
                    }
                },
            },
            "transport_results_raw": {
                "eta": 1.0e-6,
            },
        },
    )

    out = scan_partial_rgf_results(
        tmp_path / "rgf",
        fermi_ev=1.5,
        required_exact_eta=1.0e-8,
    )
    assert out["row_count"] == 0
    assert str(payload_path) in out["skipped_payloads"]


def test_compare_partial_benchmark_progress_uses_overlap_only(tmp_path: Path) -> None:
    kwant_dir = tmp_path / "kwant"
    _write_json(
        kwant_dir / "kwant_reference.json",
        {
            "fermi_ev": 1.5,
            "results": [
                {
                    "thickness_uc": 5,
                    "energy_rel_fermi_ev": -0.1,
                    "energy_abs_ev": 1.4,
                    "transmission_e2_over_h": 0.75,
                },
                {
                    "thickness_uc": 7,
                    "energy_rel_fermi_ev": 0.0,
                    "energy_abs_ev": 1.5,
                    "transmission_e2_over_h": 0.55,
                },
            ],
            "validation": {"status": "ok"},
        },
    )

    transport_dir = tmp_path / "rgf" / "d05_em0p1" / "transport" / "primary"
    _write_json(
        transport_dir / "transport_payload_primary_001.json",
        {
            "thicknesses": [5],
            "disorder_strengths": [0.0],
            "mfp_lengths": [],
            "energy": 1.4,
        },
    )
    _write_json(
        transport_dir / "transport_result.json",
        {
            "transport_results": {
                "thickness_scan": {
                    "0.0": {
                        "G_mean": [0.750001],
                    }
                }
            }
        },
    )

    out = compare_partial_benchmark_progress(
        kwant_dir=kwant_dir,
        rgf_root=tmp_path / "rgf",
    )
    assert out["status"] == "ok"
    assert out["overlap_points"] == 1
    assert out["comparison"]["checked_points"] == 1
    assert len(out["missing_in_rgf"]) == 1
    assert out["missing_in_rgf"][0]["thickness_uc"] == 7


def test_compare_partial_benchmark_progress_ignores_overlap_with_wrong_exact_eta(tmp_path: Path) -> None:
    kwant_dir = tmp_path / "kwant"
    _write_json(
        kwant_dir / "kwant_reference.json",
        {
            "fermi_ev": 1.5,
            "results": [
                {
                    "thickness_uc": 5,
                    "energy_rel_fermi_ev": -0.1,
                    "energy_abs_ev": 1.4,
                    "transmission_e2_over_h": 0.75,
                }
            ],
            "validation": {"status": "ok"},
        },
    )

    transport_dir = tmp_path / "rgf" / "d05_em0p1" / "transport" / "primary"
    payload_path = transport_dir / "transport_payload_primary_001.json"
    _write_json(
        payload_path,
        {
            "thicknesses": [5],
            "disorder_strengths": [0.0],
            "mfp_lengths": [],
            "energy": 1.4,
            "eta": 1.0e-6,
            "transport_rgf_mode": "full_finite",
            "sigma_left_path": "sigma_left.bin",
            "sigma_right_path": "sigma_right.bin",
        },
    )
    _write_json(
        transport_dir / "transport_result.json",
        {
            "runtime_cert": {
                "full_finite_sigma_source": "kwant_exact",
            },
            "transport_results": {
                "meta": {
                    "rgf_full_finite_sigma_source": "kwant_exact",
                },
                "thickness_scan": {
                    "0.0": {
                        "G_mean": [0.750001],
                    }
                },
            },
            "transport_results_raw": {
                "eta": 1.0e-6,
            },
        },
    )

    out = compare_partial_benchmark_progress(
        kwant_dir=kwant_dir,
        rgf_root=tmp_path / "rgf",
        required_exact_eta=1.0e-8,
    )
    assert out["status"] == "missing_rgf_points"
    assert out["overlap_points"] == 0
    assert out["rgf"]["row_count"] == 0
    assert str(payload_path) in out["rgf"]["skipped_payloads"]


def test_compare_partial_benchmark_progress_skips_incomplete_rgf_progress(tmp_path: Path) -> None:
    kwant_dir = tmp_path / "kwant"
    _write_json(
        kwant_dir / "kwant_payload.json",
        {
            "fermi_ev": 1.5,
        },
    )
    (kwant_dir / "wtec_job.log").write_text(
        "[kwant-bench][rank=0] done thickness_uc=5 energy_abs_ev=1.400000 transmission=0.750000000000\n",
        encoding="utf-8",
    )

    complete_dir = tmp_path / "rgf" / "d05_em0p1" / "transport" / "primary"
    _write_json(
        complete_dir / "transport_payload_primary_001.json",
        {
            "thicknesses": [5],
            "disorder_strengths": [0.0],
            "mfp_lengths": [],
            "energy": 1.4,
            "transport_rgf_mode": "full_finite",
            "sigma_left_path": "sigma_left.bin",
            "sigma_right_path": "sigma_right.bin",
        },
    )
    _write_json(
        complete_dir / "transport_result.json",
        {
            "runtime_cert": {
                "full_finite_sigma_source": "kwant_exact",
            },
            "transport_results": {
                "meta": {
                    "rgf_full_finite_sigma_source": "kwant_exact",
                },
                "thickness_scan": {
                    "0.0": {
                        "G_mean": [0.74],
                    }
                }
            }
        },
    )

    incomplete_dir = tmp_path / "rgf" / "d07_e0p0" / "transport" / "primary"
    _write_json(
        incomplete_dir / "transport_payload_primary_001.json",
        {
            "thicknesses": [7],
            "disorder_strengths": [0.0],
            "mfp_lengths": [],
            "energy": 1.5,
            "transport_rgf_mode": "full_finite",
            "sigma_left_path": "sigma_left.bin",
            "sigma_right_path": "sigma_right.bin",
        },
    )
    (incomplete_dir / "transport_progress_primary_001.jsonl").write_text(
        json.dumps({"event": "worker_start"}) + "\n",
        encoding="utf-8",
    )

    out = compare_partial_benchmark_progress(
        kwant_dir=kwant_dir,
        rgf_root=tmp_path / "rgf",
    )
    assert out["status"] == "failed"
    assert out["comparison"]["checked_points"] == 1
    assert out["rgf"]["row_count"] == 1
    assert str(incomplete_dir / "transport_payload_primary_001.json") in out["rgf"]["skipped_payloads"]
