from __future__ import annotations

from pathlib import Path

from wtec.cluster.submit import JobManager


class _DummySSH:
    def mkdir_p(self, _remote_dir: str) -> None:
        return None

    def put(self, _local: str | Path, _remote: str) -> None:
        return None


def test_submit_and_wait_retrieves_live_artifacts_before_terminal(tmp_path, monkeypatch) -> None:
    jm = JobManager(_DummySSH())
    retrieve_calls: list[tuple[str, str, tuple[str, ...]]] = []
    hook_calls: list[str] = []
    states = iter(
        [
            {
                "status": "RUNNING",
                "scheduler_state": "RUNNING",
                "exit_code": None,
                "source": "sacct",
                "terminal": False,
            },
            {
                "status": "COMPLETED",
                "scheduler_state": "COMPLETED",
                "exit_code": "0:0",
                "source": "sacct",
                "terminal": True,
            },
        ]
    )

    monkeypatch.setattr(
        jm,
        "submit",
        lambda script_content, remote_dir, script_name="job.pbs": {
            "job_id": "4242",
            "remote_script": f"{remote_dir}/{script_name}",
        },
    )
    monkeypatch.setattr(jm, "status_details", lambda _job_id: next(states))
    monkeypatch.setattr(
        jm,
        "retrieve",
        lambda remote_dir, local_dir, patterns: retrieve_calls.append(
            (str(remote_dir), str(local_dir), tuple(patterns))
        ),
    )
    monkeypatch.setattr("wtec.cluster.submit.time.sleep", lambda _seconds: None)

    meta = jm.submit_and_wait(
        "echo run",
        remote_dir="/remote/work",
        local_dir=tmp_path,
        retrieve_patterns=["result.json"],
        live_retrieve_patterns=["progress.jsonl", "partial.rank0.jsonl"],
        live_retrieve_interval_seconds=5,
        live_retrieve_hook=lambda: hook_calls.append("live"),
        script_name="job.pbs",
        poll_interval=5,
        verbose=False,
    )

    assert meta["job_id"] == "4242"
    assert retrieve_calls == [
        (
            "/remote/work",
            str(tmp_path),
            ("progress.jsonl", "partial.rank0.jsonl"),
        ),
        (
            "/remote/work",
            str(tmp_path),
            ("result.json",),
        ),
    ]
    assert hook_calls == ["live"]
