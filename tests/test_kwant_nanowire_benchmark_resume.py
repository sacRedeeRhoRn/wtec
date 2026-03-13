from __future__ import annotations

import json
from pathlib import Path

from wtec.transport import kwant_nanowire_benchmark as knb


class _FakeSmatrix:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def transmission(self, _to_lead: int, _from_lead: int) -> float:
        return float(self._value)


def _payload() -> dict:
    return {
        "mp_id": "mp-test",
        "material": "TiS",
        "model_key": "model_b",
        "model_label": "Model B",
        "axis": "c",
        "hr_dat_path": "toy_hr.dat",
        "fermi_ev": 1.0,
        "energies_rel_fermi_ev": [-0.2, 0.0],
        "thicknesses": [1],
        "width_uc": 13,
        "length_uc": 24,
        "serial_validate_thicknesses": [],
    }


def test_run_payload_writes_partial_checkpoint_on_failure(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "kwant_reference.json"
    calls: list[float] = []

    monkeypatch.setattr(knb, "_mpi_context", lambda: (None, 0, 1))
    monkeypatch.setattr(knb, "_solver_status", lambda: {"solver": "stub", "mumps_available": True})
    monkeypatch.setattr(knb, "_hr_dict", lambda path: (1, {(0, 0, 0): 0}))
    monkeypatch.setattr(knb, "_build_system_from_hr", lambda h_r, length_uc, width_uc, thickness_uc: (object(), 0))

    def _fake_transport(_fsyst, *, energy_abs: float):
        calls.append(float(energy_abs))
        if len(calls) > 1:
            raise RuntimeError("boom")
        return _FakeSmatrix(11.0)

    monkeypatch.setattr(knb, "_transport_smatrix", _fake_transport)

    result = knb.run_payload(_payload(), checkpoint_path=checkpoint)

    written = json.loads(checkpoint.read_text())
    shard_lines = (tmp_path / "kwant_reference.rank0.jsonl").read_text().splitlines()
    assert result["status"] == "partial"
    assert "fatal_error" in result
    assert written["task_count_completed"] == 1
    assert written["results"][0]["energy_rel_fermi_ev"] == -0.2
    assert len(shard_lines) == 1
    assert json.loads(shard_lines[0])["energy_rel_fermi_ev"] == -0.2
    assert calls == [0.8, 1.0]


def test_run_payload_resumes_from_partial_checkpoint(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "kwant_reference.json"
    checkpoint.write_text(
        json.dumps(
            {
                "status": "partial",
                "task_count_expected": 2,
                "task_count_completed": 1,
                "results": [
                    {
                        "thickness_uc": 1,
                        "energy_rel_fermi_ev": -0.2,
                        "energy_abs_ev": 0.8,
                        "transmission_e2_over_h": 11.0,
                    }
                ],
                "validation": {"status": "partial"},
            }
        ),
        encoding="utf-8",
    )
    calls: list[float] = []

    monkeypatch.setattr(knb, "_mpi_context", lambda: (None, 0, 1))
    monkeypatch.setattr(knb, "_solver_status", lambda: {"solver": "stub", "mumps_available": True})
    monkeypatch.setattr(knb, "_hr_dict", lambda path: (1, {(0, 0, 0): 0}))
    monkeypatch.setattr(knb, "_build_system_from_hr", lambda h_r, length_uc, width_uc, thickness_uc: (object(), 0))
    monkeypatch.setattr(
        knb,
        "_transport_smatrix",
        lambda _fsyst, *, energy_abs: calls.append(float(energy_abs)) or _FakeSmatrix(12.0),
    )

    result = knb.run_payload(_payload(), checkpoint_path=checkpoint)

    written = json.loads(checkpoint.read_text())
    assert result["status"] == "ok"
    assert written["task_count_completed"] == 2
    assert [row["energy_rel_fermi_ev"] for row in written["results"]] == [-0.2, 0.0]
    assert calls == [1.0]


def test_run_payload_resumes_from_rank_shards_without_checkpoint(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "kwant_reference.json"
    (tmp_path / "kwant_reference.rank0.jsonl").write_text(
        json.dumps(
            {
                "thickness_uc": 1,
                "energy_rel_fermi_ev": -0.2,
                "energy_abs_ev": 0.8,
                "transmission_e2_over_h": 11.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[float] = []

    monkeypatch.setattr(knb, "_mpi_context", lambda: (None, 0, 1))
    monkeypatch.setattr(knb, "_solver_status", lambda: {"solver": "stub", "mumps_available": True})
    monkeypatch.setattr(knb, "_hr_dict", lambda path: (1, {(0, 0, 0): 0}))
    monkeypatch.setattr(knb, "_build_system_from_hr", lambda h_r, length_uc, width_uc, thickness_uc: (object(), 0))
    monkeypatch.setattr(
        knb,
        "_transport_smatrix",
        lambda _fsyst, *, energy_abs: calls.append(float(energy_abs)) or _FakeSmatrix(12.0),
    )

    result = knb.run_payload(_payload(), checkpoint_path=checkpoint)

    written = json.loads(checkpoint.read_text())
    assert result["status"] == "ok"
    assert written["task_count_completed"] == 2
    assert [row["energy_rel_fermi_ev"] for row in written["results"]] == [-0.2, 0.0]
    assert calls == [1.0]
    assert not (tmp_path / "kwant_reference.rank0.jsonl").exists()
