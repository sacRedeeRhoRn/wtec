from typing import Any

import wtec.topology.evaluator as evaluator


class _DummyModel:
    pass


def test_evaluate_topology_point_marks_partial_on_arc_failure(monkeypatch) -> None:
    monkeypatch.setattr(evaluator, "_load_tb_model", lambda *args, **kwargs: _DummyModel())
    monkeypatch.setattr(
        evaluator,
        "validate_wannier_model",
        lambda *args, **kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(
        evaluator,
        "scan_weyl_nodes",
        lambda *args, **kwargs: {"status": "ok", "nodes": [], "n_nodes": 0},
    )
    monkeypatch.setattr(
        evaluator,
        "compute_arc_connectivity",
        lambda *args, **kwargs: {"status": "failed", "reason": "missing_ldos"},
    )

    task: dict[str, Any] = {
        "hr_dat_path": "dummy_hr.dat",
        "thickness_uc": 4,
        "variant_id": "v0",
        "point_index": 0,
        "point_name": "point_000",
    }
    out = evaluator.evaluate_topology_point(task, run_validation=True, run_node=True, run_arc=True)
    assert out["status"] == "partial"
    assert "arc_scan" in str(out.get("reason", ""))
    assert out["transport_probe"]["status"] == "skipped"
