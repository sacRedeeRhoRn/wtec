from __future__ import annotations

import argparse
from contextlib import suppress
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from wtec.wannier.parser import read_hr_dat


def _solver_status() -> dict[str, Any]:
    out: dict[str, Any] = {
        "kwant_version": None,
        "solver": "unknown",
        "mumps_available": False,
    }
    try:
        import kwant

        out["kwant_version"] = str(getattr(kwant, "__version__", "unknown"))
        try:
            import kwant.solvers.mumps as _mumps  # noqa: F401

            out["solver"] = "mumps"
            out["mumps_available"] = True
        except Exception:
            out["solver"] = "scipy_fallback"
            out["mumps_available"] = False
        return out
    except Exception as exc:  # pragma: no cover - runtime dependent
        out["solver"] = f"import_failed:{type(exc).__name__}"
        return out


def _hr_dict(path: str | Path) -> tuple[int, dict[tuple[int, int, int], np.ndarray]]:
    hd = read_hr_dat(Path(path).expanduser().resolve())
    h_r: dict[tuple[int, int, int], np.ndarray] = {}
    for ri, rv in enumerate(np.asarray(hd.r_vectors, dtype=int)):
        key = tuple(int(v) for v in rv)
        denom = float(hd.deg[ri]) if int(hd.deg[ri]) != 0 else 1.0
        h_r[key] = np.asarray(hd.H_R[ri], dtype=np.complex128) / denom
    if (0, 0, 0) not in h_r:
        raise ValueError("R=(0,0,0) onsite block is missing in hr.dat.")
    return int(hd.num_wann), h_r


def _is_positive_lex(r: tuple[int, int, int]) -> bool:
    if r == (0, 0, 0):
        return False
    rx, ry, rz = r
    if rx != 0:
        return rx > 0
    if ry != 0:
        return ry > 0
    return rz > 0


def _max_hop_range_axis(h_r: dict[tuple[int, int, int], np.ndarray], axis: int = 0) -> int:
    mx = 0
    for rv in h_r:
        if rv == (0, 0, 0):
            continue
        mx = max(mx, abs(int(rv[axis])))
    return mx


def _build_system_from_hr(
    h_r: dict[tuple[int, int, int], np.ndarray],
    *,
    length_uc: int,
    width_uc: int,
    thickness_uc: int,
):
    import kwant

    onsite = np.asarray(h_r[(0, 0, 0)], dtype=np.complex128).copy()
    norb = onsite.shape[0]
    lat = kwant.lattice.general(
        [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        [(0, 0, 0)],
        norbs=norb,
    )
    a = lat.sublattices[0]

    syst = kwant.Builder()
    for x in range(int(length_uc)):
        for y in range(int(width_uc)):
            for z in range(int(thickness_uc)):
                syst[a(x, y, z)] = onsite

    for rv, mat in h_r.items():
        if rv == (0, 0, 0):
            continue
        if _is_positive_lex(rv):
            syst[kwant.builder.HoppingKind(rv, a, a)] = np.asarray(mat, dtype=np.complex128)

    sym = kwant.TranslationalSymmetry(lat.vec((1, 0, 0)))
    lead = kwant.Builder(sym)
    for y in range(int(width_uc)):
        for z in range(int(thickness_uc)):
            lead[a(0, y, z)] = onsite

    for rv, mat in h_r.items():
        if rv == (0, 0, 0):
            continue
        rx = int(rv[0])
        if rx == 0 and _is_positive_lex(rv):
            lead[kwant.builder.HoppingKind(rv, a, a)] = np.asarray(mat, dtype=np.complex128)
        if rx > 0:
            lead[kwant.builder.HoppingKind(rv, a, a)] = np.asarray(mat, dtype=np.complex128)

    max_rx = _max_hop_range_axis(h_r, axis=0)
    add_cells = max(0, max_rx - 1)
    syst.attach_lead(lead, add_cells=add_cells)
    syst.attach_lead(lead.reversed(), add_cells=add_cells)
    return syst.finalized(), add_cells


def _mpi_context() -> tuple[Any, int, int]:
    try:
        from mpi4py import MPI
    except Exception:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, int(comm.Get_rank()), int(comm.Get_size())


def _serial_reference(fsyst, energies_abs: list[float]) -> list[tuple[float, float]]:
    import kwant

    out: list[tuple[float, float]] = []
    for energy_abs in energies_abs:
        smat = kwant.smatrix(fsyst, energy=float(energy_abs))
        out.append((float(energy_abs), float(smat.transmission(1, 0))))
    return out


def _transport_smatrix(fsyst, *, energy_abs: float):
    import kwant

    # Use kwant's default solver route so MUMPS is picked when available.
    return kwant.smatrix(fsyst, energy=float(energy_abs))


def _energy_key(value: float) -> str:
    return f"{float(value):.12g}"


def _task_key(thickness_uc: int, energy_rel_fermi_ev: float) -> tuple[int, str]:
    return (int(thickness_uc), _energy_key(energy_rel_fermi_ev))


def _expected_task_count(
    *,
    thicknesses: list[int] | tuple[int, ...],
    energies_rel_fermi_ev: list[float] | tuple[float, ...],
) -> int:
    return int(len(list(thicknesses)) * len(list(energies_rel_fermi_ev)))


def _checkpoint_shard_glob(checkpoint_path: Path) -> str:
    return f"{checkpoint_path.stem}.rank*.jsonl"


def _checkpoint_shard_path(checkpoint_path: Path, *, rank: int) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.rank{int(rank)}.jsonl")


def _checkpoint_rows_by_key(checkpoint_path: Path | None) -> dict[tuple[int, str], dict[str, Any]]:
    rows_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    if checkpoint_path is None:
        return rows_by_key

    def _remember(row: Any) -> None:
        if not isinstance(row, dict):
            return
        try:
            key = _task_key(
                int(row.get("thickness_uc", 0)),
                float(row.get("energy_rel_fermi_ev", 0.0)),
            )
        except Exception:
            return
        rows_by_key[key] = row

    if checkpoint_path.exists():
        with suppress(Exception):
            existing_payload = json.loads(checkpoint_path.read_text())
            for row in list(existing_payload.get("results", [])):
                _remember(row)

    for shard_path in sorted(checkpoint_path.parent.glob(_checkpoint_shard_glob(checkpoint_path))):
        with suppress(Exception):
            for line in shard_path.read_text().splitlines():
                if line.strip():
                    _remember(json.loads(line))

    return rows_by_key


def _append_checkpoint_shard(
    checkpoint_path: Path | None,
    *,
    rank: int,
    row: dict[str, Any],
) -> None:
    if checkpoint_path is None:
        return
    shard_path = _checkpoint_shard_path(checkpoint_path, rank=rank)
    with shard_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _cleanup_checkpoint_shards(checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return
    for shard_path in checkpoint_path.parent.glob(_checkpoint_shard_glob(checkpoint_path)):
        with suppress(OSError):
            shard_path.unlink()


def kwant_reference_checkpoint_payload(
    checkpoint_path: Path,
    *,
    thicknesses: list[int] | tuple[int, ...] | None = None,
    energies_rel_fermi_ev: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    base_payload: dict[str, Any] = {}
    if checkpoint_path.exists():
        with suppress(Exception):
            maybe_payload = json.loads(checkpoint_path.read_text())
            if isinstance(maybe_payload, dict):
                base_payload = maybe_payload

    rows_by_key = _checkpoint_rows_by_key(checkpoint_path)
    results = sorted(
        rows_by_key.values(),
        key=lambda row: (int(row["thickness_uc"]), float(row["energy_rel_fermi_ev"])),
    )
    expected = int(base_payload.get("task_count_expected", 0) or 0)
    if expected <= 0 and thicknesses is not None and energies_rel_fermi_ev is not None:
        expected = _expected_task_count(
            thicknesses=thicknesses,
            energies_rel_fermi_ev=energies_rel_fermi_ev,
        )
    completed = int(len(results))
    out = dict(base_payload)
    out["results"] = results
    out["task_count_expected"] = int(expected)
    out["task_count_completed"] = int(completed)
    out["status"] = "ok" if expected > 0 and completed >= expected else str(
        base_payload.get("status", "partial")
    )
    if "validation" not in out or not isinstance(out.get("validation"), dict):
        out["validation"] = {"status": "partial"}
    return out


def kwant_reference_progress(
    result: dict[str, Any],
    *,
    thicknesses: list[int] | tuple[int, ...] | None = None,
    energies_rel_fermi_ev: list[float] | tuple[float, ...] | None = None,
) -> dict[str, int | bool]:
    rows = list(result.get("results", [])) if isinstance(result, dict) else []
    completed = len(
        {
            _task_key(
                int(row.get("thickness_uc", 0)),
                float(row.get("energy_rel_fermi_ev", 0.0)),
            )
            for row in rows
            if isinstance(row, dict)
        }
    )
    expected = int(result.get("task_count_expected", 0) or 0)
    if expected <= 0 and thicknesses is not None and energies_rel_fermi_ev is not None:
        expected = _expected_task_count(
            thicknesses=thicknesses,
            energies_rel_fermi_ev=energies_rel_fermi_ev,
        )
    if expected <= 0:
        expected = completed
    return {
        "completed": int(completed),
        "expected": int(expected),
        "complete": bool(completed >= expected),
    }


def kwant_reference_is_complete(
    result: dict[str, Any],
    *,
    thicknesses: list[int] | tuple[int, ...] | None = None,
    energies_rel_fermi_ev: list[float] | tuple[float, ...] | None = None,
) -> bool:
    return bool(
        kwant_reference_progress(
            result,
            thicknesses=thicknesses,
            energies_rel_fermi_ev=energies_rel_fermi_ev,
        )["complete"]
    )


def _build_result_payload(
    *,
    payload: dict[str, Any],
    solver: dict[str, Any],
    model_key: str,
    model_label: str,
    length_uc: int,
    width_uc: int,
    thicknesses: list[int],
    energies_rel_fermi_ev: list[float],
    fermi_ev: float,
    mpi_world_size: int,
    norb: int,
    results: list[dict[str, Any]],
    validation: dict[str, Any],
    status: str,
    fatal_error: str | None = None,
) -> dict[str, Any]:
    out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": str(status),
        "mp_id": str(payload.get("mp_id", "")),
        "material": str(payload.get("material", "")),
        "model_key": model_key,
        "model_label": model_label,
        "axis": str(payload.get("axis", "")),
        "length_uc": int(length_uc),
        "width_uc": int(width_uc),
        "thicknesses": [int(v) for v in thicknesses],
        "energies_rel_fermi_ev": [float(v) for v in energies_rel_fermi_ev],
        "fermi_ev": float(fermi_ev),
        "mpi_world_size": int(mpi_world_size),
        "solver": solver,
        "results": results,
        "validation": validation,
        "norb": int(norb),
        "task_count_expected": int(
            _expected_task_count(
                thicknesses=thicknesses,
                energies_rel_fermi_ev=energies_rel_fermi_ev,
            )
        ),
        "task_count_completed": int(len(results)),
    }
    if fatal_error:
        out["fatal_error"] = str(fatal_error)
    return out


def run_payload(payload: dict[str, Any], *, checkpoint_path: Path | None = None) -> dict[str, Any]:
    comm, rank, size = _mpi_context()
    solver = _solver_status()
    norb, h_r = _hr_dict(payload["hr_dat_path"])
    model_key = str(payload.get("model_key", "")).strip()
    model_label = str(payload.get("model_label", "")).strip()
    thicknesses = [int(v) for v in payload.get("thicknesses", [])]
    width_uc = int(payload.get("width_uc", max(thicknesses) if thicknesses else 1))
    delta_energies = [float(v) for v in payload.get("energies_rel_fermi_ev", [])]
    fermi_ev = float(payload.get("fermi_ev", 0.0))
    energies_abs = [fermi_ev + float(v) for v in delta_energies]
    length_uc = int(payload["length_uc"])
    serial_validate = {int(v) for v in payload.get("serial_validate_thicknesses", [])}

    tasks: list[tuple[int, float, float]] = []
    for thickness_uc in thicknesses:
        for de, eabs in zip(delta_energies, energies_abs):
            tasks.append((int(thickness_uc), float(de), float(eabs)))

    existing_results_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    if rank == 0:
        existing_results_by_key = _checkpoint_rows_by_key(checkpoint_path)
    completed_keys = set(existing_results_by_key)
    if comm is not None:
        completed_keys = set(comm.bcast(completed_keys, root=0))
    pending_tasks = [
        task
        for task in tasks
        if _task_key(int(task[0]), float(task[1])) not in completed_keys
    ]

    local = pending_tasks[rank::size]
    fsys_cache: dict[int, tuple[Any, int]] = {}
    max_rounds = len(local)
    if comm is not None:
        max_rounds = max(int(v) for v in comm.allgather(len(local)))

    def _partial_validation() -> dict[str, Any]:
        return {
            "enabled": bool(serial_validate),
            "checked_thicknesses": sorted(int(v) for v in serial_validate),
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "status": "partial",
            "failures": [],
        }

    def _write_checkpoint(
        merged_rows: list[dict[str, Any]],
        *,
        status: str,
        validation: dict[str, Any] | None = None,
        fatal_error: str | None = None,
    ) -> dict[str, Any]:
        result = _build_result_payload(
            payload=payload,
            solver=solver,
            model_key=model_key,
            model_label=model_label,
            length_uc=int(length_uc),
            width_uc=int(width_uc),
            thicknesses=thicknesses,
            energies_rel_fermi_ev=delta_energies,
            fermi_ev=float(fermi_ev),
            mpi_world_size=int(size),
            norb=int(norb),
            results=merged_rows,
            validation=validation or _partial_validation(),
            status=status,
            fatal_error=fatal_error,
        )
        if rank == 0 and checkpoint_path is not None:
            checkpoint_path.write_text(json.dumps(result, indent=2))
        return result

    if rank == 0:
        progress = kwant_reference_progress(
            {"results": list(existing_results_by_key.values())},
            thicknesses=thicknesses,
            energies_rel_fermi_ev=delta_energies,
        )
        env_threads = os.environ.get("OMP_NUM_THREADS", "").strip() or None
        print(
            "[kwant-bench] "
            f"solver={solver.get('solver')} "
            f"mumps={solver.get('mumps_available')} "
            f"tasks={len(tasks)} mpi={size} "
            f"threads={env_threads or 'unset'} "
            f"length_uc={length_uc}",
            flush=True,
        )
        if int(progress["completed"]) > 0:
            print(
                "[kwant-bench] "
                f"resume completed={int(progress['completed'])}/{int(progress['expected'])} "
                f"pending={len(pending_tasks)}",
                flush=True,
            )

    merged_by_key = dict(existing_results_by_key) if rank == 0 else {}
    fatal_error: str | None = None

    for round_idx in range(int(max_rounds)):
        round_result: dict[str, Any] | None = None
        round_error: str | None = None
        if round_idx < len(local):
            thickness_uc, de, eabs = local[round_idx]
            try:
                print(
                    f"[kwant-bench][rank={rank}] start thickness_uc={thickness_uc} energy_abs_ev={eabs:.6f}",
                    flush=True,
                )
                if thickness_uc not in fsys_cache:
                    fsys_cache[thickness_uc] = _build_system_from_hr(
                        h_r,
                        length_uc=length_uc,
                        width_uc=width_uc,
                        thickness_uc=thickness_uc,
                    )
                fsyst, add_cells = fsys_cache[thickness_uc]
                smat = _transport_smatrix(fsyst, energy_abs=float(eabs))
                t10 = float(smat.transmission(1, 0))
                print(
                    f"[kwant-bench][rank={rank}] done thickness_uc={thickness_uc} "
                    f"energy_abs_ev={eabs:.6f} transmission={t10:.12f}",
                    flush=True,
                )
                round_result = {
                    "thickness_uc": int(thickness_uc),
                    "energy_rel_fermi_ev": float(de),
                    "energy_abs_ev": float(eabs),
                    "transmission_e2_over_h": t10,
                    "add_cells": int(add_cells),
                    "width_uc": int(width_uc),
                    "model_key": model_key,
                    "model_label": model_label,
                    "rank": int(rank),
                }
                _append_checkpoint_shard(checkpoint_path, rank=rank, row=round_result)
            except Exception as exc:  # pragma: no cover - runtime dependent
                round_error = (
                    f"rank={rank}, thickness_uc={thickness_uc}, energy_abs_ev={eabs}, "
                    f"error={type(exc).__name__}: {exc}"
                )

        if comm is not None:
            gathered_results = comm.gather(round_result, root=0)
            gathered_errors = comm.gather(round_error, root=0)
        else:
            gathered_results = [round_result]
            gathered_errors = [round_error]

        stop = False
        if rank == 0:
            for row in gathered_results:
                if not isinstance(row, dict):
                    continue
                merged_by_key[
                    _task_key(
                        int(row["thickness_uc"]),
                        float(row["energy_rel_fermi_ev"]),
                    )
                ] = row
            merged_rows = sorted(
                merged_by_key.values(),
                key=lambda row: (int(row["thickness_uc"]), float(row["energy_rel_fermi_ev"])),
            )
            _write_checkpoint(merged_rows, status="partial")
            errors = [err for err in gathered_errors if isinstance(err, str) and err]
            if errors:
                fatal_error = "MPI worker failure(s): " + " | ".join(errors)
                stop = True
        if comm is not None:
            stop = bool(comm.bcast(bool(stop), root=0))
        if stop:
            break

    if rank != 0:
        return {}

    merged = sorted(
        merged_by_key.values(),
        key=lambda row: (int(row["thickness_uc"]), float(row["energy_rel_fermi_ev"])),
    )
    if fatal_error is not None:
        return _write_checkpoint(merged, status="partial", fatal_error=fatal_error)

    validation = {
        "enabled": bool(serial_validate),
        "checked_thicknesses": sorted(int(v) for v in serial_validate),
        "max_abs_err": 0.0,
        "max_rel_err": 0.0,
        "status": "skipped" if not serial_validate else "ok",
        "failures": [],
    }
    if serial_validate:
        merged_map = {
            (int(row["thickness_uc"]), float(row["energy_abs_ev"])): float(row["transmission_e2_over_h"])
            for row in merged
        }
        for thickness_uc in sorted(serial_validate):
            fsyst, _ = fsys_cache.get(thickness_uc) or _build_system_from_hr(
                h_r,
                length_uc=length_uc,
                width_uc=width_uc,
                thickness_uc=thickness_uc,
            )
            serial = _serial_reference(fsyst, energies_abs)
            for eabs, tref in serial:
                got = merged_map[(int(thickness_uc), float(eabs))]
                abs_err = abs(float(got) - float(tref))
                denom = max(abs(float(tref)), 1.0e-12)
                rel_err = abs_err / denom
                validation["max_abs_err"] = max(float(validation["max_abs_err"]), float(abs_err))
                validation["max_rel_err"] = max(float(validation["max_rel_err"]), float(rel_err))
                if abs_err > 1.0e-10:
                    validation["status"] = "failed"
                    validation["failures"].append(
                        {
                            "thickness_uc": int(thickness_uc),
                            "energy_abs_ev": float(eabs),
                            "parallel": float(got),
                            "serial": float(tref),
                            "abs_err": float(abs_err),
                            "rel_err": float(rel_err),
                        }
                    )

    result = _write_checkpoint(merged, status="ok", validation=validation)
    _cleanup_checkpoint_shards(checkpoint_path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else [])
    if not args:
        ap = argparse.ArgumentParser(description="Run a nanowire Kwant benchmark sweep under mpirun.")
        ap.add_argument("payload_json")
        ap.add_argument("output_json")
        ns = ap.parse_args()
        payload_path = Path(ns.payload_json).expanduser().resolve()
        output_path = Path(ns.output_json).expanduser().resolve()
    else:
        if len(args) != 2:
            raise SystemExit("Usage: python -m wtec.transport.kwant_nanowire_benchmark payload.json output.json")
        payload_path = Path(args[0]).expanduser().resolve()
        output_path = Path(args[1]).expanduser().resolve()

    payload = json.loads(payload_path.read_text())
    result = run_payload(payload, checkpoint_path=output_path)
    comm, rank, _ = _mpi_context()
    if rank == 0:
        output_path.write_text(json.dumps(result, indent=2))
    if comm is not None:
        comm.Barrier()
    if rank == 0 and str(result.get("fatal_error", "") or "").strip():
        raise RuntimeError(str(result["fatal_error"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
