from __future__ import annotations

import json
from pathlib import Path
import time

from wtec.transport import kwant_nanowire_benchmark as knb


class _FakeSmatrix:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def transmission(self, _to_lead: int, _from_lead: int) -> float:
        return float(self._value)


class _NoopHeartbeat:
    def __init__(self, **_kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


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


def test_run_payload_emits_heartbeat_during_long_kwant_point(
    tmp_path: Path, monkeypatch, capfd
) -> None:
    checkpoint = tmp_path / "kwant_reference.json"

    monkeypatch.setenv("TOPOSLAB_KWANT_BENCH_HEARTBEAT_SECONDS", "0.05")
    monkeypatch.setattr(knb, "_mpi_context", lambda: (None, 0, 1))
    monkeypatch.setattr(knb, "_solver_status", lambda: {"solver": "stub", "mumps_available": True})
    monkeypatch.setattr(knb, "_hr_dict", lambda path: (1, {(0, 0, 0): 0}))
    monkeypatch.setattr(knb, "_build_system_from_hr", lambda h_r, length_uc, width_uc, thickness_uc: (object(), 0))

    def _slow_transport(_fsyst, *, energy_abs: float):
        time.sleep(0.18)
        return _FakeSmatrix(12.0)

    monkeypatch.setattr(knb, "_transport_smatrix", _slow_transport)

    result = knb.run_payload(_payload(), checkpoint_path=checkpoint)

    captured = capfd.readouterr().out
    assert result["status"] == "ok"
    assert "[kwant-bench][rank=0] heartbeat thickness_uc=1 energy_abs_ev=0.800000" in captured


def test_run_local_tasks_appends_rank_shards_per_completion(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "kwant_reference.json"
    build_calls: list[int] = []
    transport_calls: list[float] = []
    callback_rows: list[dict] = []

    def _fake_build(_h_r, *, length_uc: int, width_uc: int, thickness_uc: int):
        build_calls.append(int(thickness_uc))
        return object(), 10 + int(thickness_uc)

    def _fake_transport(_fsyst, *, energy_abs: float):
        transport_calls.append(float(energy_abs))
        return _FakeSmatrix(float(energy_abs))

    monkeypatch.setattr(knb, "_build_system_from_hr", _fake_build)
    monkeypatch.setattr(knb, "_transport_smatrix", _fake_transport)
    monkeypatch.setattr(knb, "_KwantHeartbeat", _NoopHeartbeat)

    results, fsys_cache, fatal_error = knb._run_local_tasks(
        local_tasks=[(1, -0.2, 0.8), (1, 0.0, 1.0)],
        rank=3,
        h_r={(0, 0, 0): 0},
        length_uc=24,
        width_uc=13,
        checkpoint_path=checkpoint,
        model_key="model_b",
        model_label="Model B",
        row_callback=callback_rows.append,
    )

    shard_lines = (tmp_path / "kwant_reference.rank3.jsonl").read_text().splitlines()
    assert fatal_error is None
    assert build_calls == [1]
    assert transport_calls == [0.8, 1.0]
    assert list(fsys_cache) == [1]
    assert [row["energy_rel_fermi_ev"] for row in results] == [-0.2, 0.0]
    assert [row["energy_rel_fermi_ev"] for row in callback_rows] == [-0.2, 0.0]
    assert [json.loads(line)["energy_rel_fermi_ev"] for line in shard_lines] == [-0.2, 0.0]


def test_distribute_pending_tasks_keeps_thickness_groups_together() -> None:
    pending_tasks = [
        (thickness_uc, energy_rel_fermi_ev, 13.6046 + float(energy_rel_fermi_ev))
        for thickness_uc in (3, 5, 7, 9, 11, 13)
        for energy_rel_fermi_ev in (-0.2, -0.1, 0.0, 0.1, 0.2)
    ]

    for size in (3, 16):
        buckets = knb._distribute_pending_tasks(pending_tasks, size=size)
        thickness_to_bucket: dict[int, int] = {}
        for bucket_index, bucket in enumerate(buckets):
            local_costs = [knb._task_cost_estimate(task) for task in bucket]
            assert local_costs == sorted(local_costs)
            for task in bucket:
                thickness_uc = int(task[0])
                previous = thickness_to_bucket.setdefault(thickness_uc, bucket_index)
                assert previous == bucket_index

        assert sorted(task for bucket in buckets for task in bucket) == sorted(pending_tasks)


def test_distribute_pending_tasks_defaults_to_one_rank_per_thickness() -> None:
    pending_tasks = [
        (thickness_uc, energy_rel_fermi_ev, 13.6046 + float(energy_rel_fermi_ev))
        for thickness_uc in (3, 5, 7, 9, 11, 13)
        for energy_rel_fermi_ev in (-0.2, -0.1, 0.0, 0.1, 0.2)
    ]

    buckets = knb._distribute_pending_tasks(pending_tasks, size=16)
    first_wave = [bucket[0] for bucket in buckets if bucket]
    assert [int(task[0]) for task in first_wave] == [3, 5, 7, 9, 11, 13]
    counts_by_thickness: dict[int, int] = {}
    for bucket in buckets:
        if not bucket:
            continue
        thicknesses = {int(task[0]) for task in bucket}
        assert len(thicknesses) == 1
        thickness_uc = thicknesses.pop()
        counts_by_thickness[thickness_uc] = counts_by_thickness.get(thickness_uc, 0) + 1
    assert counts_by_thickness == {3: 1, 5: 1, 7: 1, 9: 1, 11: 1, 13: 1}


def test_distribute_pending_tasks_can_split_thin_groups_when_enabled(monkeypatch) -> None:
    pending_tasks = [
        (thickness_uc, energy_rel_fermi_ev, 13.6046 + float(energy_rel_fermi_ev))
        for thickness_uc in (3, 5, 7, 9, 11, 13)
        for energy_rel_fermi_ev in (-0.2, -0.1, 0.0, 0.1, 0.2)
    ]

    monkeypatch.setenv("TOPOSLAB_KWANT_BENCH_SPLIT_THICKNESS_GROUPS", "1")

    buckets = knb._distribute_pending_tasks(pending_tasks, size=16)
    first_wave = [bucket[0] for bucket in buckets if bucket]
    assert [int(task[0]) for task in first_wave] == [3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 7, 7, 7, 9, 11, 13]
    counts_by_thickness: dict[int, int] = {}
    for bucket in buckets:
        if not bucket:
            continue
        thicknesses = {int(task[0]) for task in bucket}
        assert len(thicknesses) == 1
        thickness_uc = thicknesses.pop()
        counts_by_thickness[thickness_uc] = counts_by_thickness.get(thickness_uc, 0) + 1
    assert counts_by_thickness == {3: 5, 5: 5, 7: 3, 9: 1, 11: 1, 13: 1}
