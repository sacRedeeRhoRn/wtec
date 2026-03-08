"""MPI worker for topology batch evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from wtec.topology.evaluator import evaluate_topology_point


def _mpi_context():
    try:
        from mpi4py import MPI
    except Exception:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def _mode_flags(mode: str) -> tuple[bool, bool, bool]:
    m = mode.strip().lower()
    if m == "full":
        return True, True, True
    if m == "node":
        return True, True, False
    if m == "arc":
        return False, False, True
    raise ValueError(f"Unknown worker mode: {mode!r}")


def _failed_result(task: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "point_index": task.get("point_index"),
        "point_name": task.get("point_name"),
        "variant_id": task.get("variant_id"),
        "thickness_uc": task.get("thickness_uc"),
        "reason": f"{type(exc).__name__}: {exc}",
    }


def _run_tasks(
    tasks: list[dict[str, Any]],
    *,
    mode: str,
    comm=None,
    use_comm_for_node: bool = False,
) -> list[dict[str, Any]]:
    run_validation, run_node, run_arc = _mode_flags(mode)
    cache = {}
    out: list[dict[str, Any]] = []
    for t in tasks:
        try:
            res = evaluate_topology_point(
                t,
                cache=cache,
                run_validation=run_validation,
                run_node=run_node,
                run_arc=run_arc,
                comm=comm if (run_node and use_comm_for_node) else None,
            )
            out.append(res)
        except Exception as exc:
            out.append(_failed_result(t, exc))
    return out


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in {2, 3}:
        print(
            "Usage: python -m wtec.topology.worker <payload.json> <output.json> [full|node|arc]",
            file=sys.stderr,
        )
        return 2
    payload_path = Path(args[0]).expanduser()
    output_path = Path(args[1]).expanduser()
    mode = args[2].strip().lower() if len(args) == 3 else "full"
    if mode not in {"full", "node", "arc"}:
        print(f"Unknown mode: {mode!r}", file=sys.stderr)
        return 2
    if not payload_path.exists():
        print(f"Payload not found: {payload_path}", file=sys.stderr)
        return 2

    payload = json.loads(payload_path.read_text())
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        print("payload['tasks'] must be a list", file=sys.stderr)
        return 2

    comm, rank, size = _mpi_context()

    # For single-point node/arc mode, allow all ranks to cooperate on one task.
    if mode in {"node", "arc"} and len(tasks) == 1 and comm is not None and size > 1:
        task = tasks[0]
        try:
            res = evaluate_topology_point(
                task,
                cache={},
                run_validation=(mode == "node"),
                run_node=(mode == "node"),
                run_arc=(mode == "arc"),
                comm=comm if mode == "node" else None,
            )
        except Exception as exc:
            res = _failed_result(task, exc)
        if rank == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps({"results": [res]}, indent=2))
        return 0

    local_tasks = tasks[rank::size]
    local_results = _run_tasks(local_tasks, mode=mode, comm=comm, use_comm_for_node=False)

    if comm is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"results": local_results}, indent=2))
        return 0

    gathered = comm.gather(local_results, root=0)
    if rank == 0:
        merged: list[dict[str, Any]] = []
        for chunk in gathered:
            merged.extend(chunk)
        if merged and all("point_index" in r for r in merged):
            merged.sort(key=lambda r: int(r.get("point_index", 10**9)))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"results": merged}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
