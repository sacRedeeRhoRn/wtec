from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wtec.transport.nanowire_benchmark import NanowireBenchmarkSpec
from wtec.transport.nanowire_benchmark_progress import compare_partial_benchmark_progress


_KWANT_START_RE = re.compile(
    r"\[kwant-bench\]\[rank=(?P<rank>\d+)\]\s+start\s+"
    r"thickness_uc=(?P<thickness>\d+)\s+"
    r"energy_abs_ev=(?P<energy_abs>[-+0-9.eE]+)"
)
_KWANT_DONE_RE = re.compile(
    r"\[kwant-bench\]\[rank=(?P<rank>\d+)\]\s+done\s+"
    r"thickness_uc=(?P<thickness>\d+)\s+"
    r"energy_abs_ev=(?P<energy_abs>[-+0-9.eE]+)\s+"
    r"transmission=(?P<transmission>[-+0-9.eE]+)"
)
_KWANT_HEARTBEAT_RE = re.compile(
    r"\[kwant-bench\]\[rank=(?P<rank>\d+)\]\s+heartbeat\s+"
    r"thickness_uc=(?P<thickness>\d+)\s+"
    r"energy_abs_ev=(?P<energy_abs>[-+0-9.eE]+)\s+"
    r"elapsed_s=(?P<elapsed>[-+0-9.eE]+)"
)
_DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 60.0
_COVERAGE_STATE_ORDER = (
    "missing",
    "kwant_only",
    "rgf_only",
    "overlap_ok",
    "overlap_fail",
)
_COVERAGE_STATE_COLORS = {
    "missing": "#e5e7eb",
    "kwant_only": "#f59e0b",
    "rgf_only": "#2563eb",
    "overlap_ok": "#16a34a",
    "overlap_fail": "#dc2626",
}


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canon_float(value: Any, *, digits: int = 12) -> float:
    return round(float(value), digits)


def _point_key(thickness_uc: int, energy_abs_ev: float) -> tuple[int, float]:
    return int(thickness_uc), _canon_float(energy_abs_ev)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _rows_by_key(rows: list[dict[str, Any]]) -> dict[tuple[int, float], dict[str, Any]]:
    return {_point_key(int(row["thickness_uc"]), float(row["energy_abs_ev"])): dict(row) for row in rows}


def _detect_axis_root(benchmark_root: Path, *, spec: NanowireBenchmarkSpec) -> Path:
    candidates: list[Path] = []
    for kwant_dir in benchmark_root.glob("*/*/kwant"):
        axis_root = kwant_dir.parent
        if (axis_root / "rgf").is_dir():
            candidates.append(axis_root)
    if not candidates:
        raise FileNotFoundError(
            f"No nanowire benchmark axis root found under {benchmark_root} (expected */*/kwant and */*/rgf)."
        )
    preferred = [path for path in candidates if path.name in {str(axis) for axis in spec.axes}]
    if preferred:
        candidates = preferred
    candidates.sort()
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple nanowire benchmark axis roots found under {benchmark_root}: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def infer_kwant_heartbeat_interval_seconds(text: str) -> float:
    by_point: dict[tuple[int, float], list[float]] = {}
    for match in _KWANT_HEARTBEAT_RE.finditer(text):
        key = _point_key(int(match.group("thickness")), float(match.group("energy_abs")))
        by_point.setdefault(key, []).append(float(match.group("elapsed")))
    deltas: list[float] = []
    for values in by_point.values():
        values.sort()
        for prev, cur in zip(values, values[1:]):
            delta = float(cur) - float(prev)
            if delta > 0.0:
                deltas.append(delta)
    if deltas:
        return float(min(deltas))
    return _DEFAULT_HEARTBEAT_INTERVAL_SECONDS


def scan_kwant_runtime_bounds(log_path: str | Path) -> dict[str, Any]:
    path = Path(log_path).expanduser().resolve()
    if not path.exists():
        return {
            "log_path": str(path),
            "heartbeat_interval_seconds": _DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
            "points": [],
        }

    text = path.read_text(encoding="utf-8", errors="replace")
    heartbeat_interval = infer_kwant_heartbeat_interval_seconds(text)
    points: dict[tuple[int, float], dict[str, Any]] = {}

    def ensure_point(thickness_uc: int, energy_abs_ev: float) -> dict[str, Any]:
        key = _point_key(thickness_uc, energy_abs_ev)
        point = points.get(key)
        if point is None:
            point = {
                "thickness_uc": int(thickness_uc),
                "energy_abs_ev": float(energy_abs_ev),
                "start_seen": False,
                "completed": False,
                "done_seen": False,
                "heartbeat_count": 0,
                "elapsed_lower_bound_s": None,
                "elapsed_upper_bound_s": None,
            }
            points[key] = point
        return point

    for match in _KWANT_START_RE.finditer(text):
        point = ensure_point(int(match.group("thickness")), float(match.group("energy_abs")))
        point["start_seen"] = True
        point["rank"] = int(match.group("rank"))

    for match in _KWANT_HEARTBEAT_RE.finditer(text):
        point = ensure_point(int(match.group("thickness")), float(match.group("energy_abs")))
        point["start_seen"] = True
        point["rank"] = int(match.group("rank"))
        point["heartbeat_count"] = int(point.get("heartbeat_count", 0)) + 1
        elapsed = float(match.group("elapsed"))
        prev = _safe_float(point.get("elapsed_lower_bound_s"))
        point["elapsed_lower_bound_s"] = elapsed if prev is None else max(prev, elapsed)

    for match in _KWANT_DONE_RE.finditer(text):
        point = ensure_point(int(match.group("thickness")), float(match.group("energy_abs")))
        point["start_seen"] = True
        point["rank"] = int(match.group("rank"))
        point["done_seen"] = True
        point["completed"] = True
        lower = _safe_float(point.get("elapsed_lower_bound_s"))
        if lower is None:
            lower = 0.0
        point["elapsed_lower_bound_s"] = float(lower)
        point["elapsed_upper_bound_s"] = float(lower) + float(heartbeat_interval)

    ordered = sorted(points.values(), key=lambda row: (int(row["thickness_uc"]), float(row["energy_abs_ev"])))
    return {
        "log_path": str(path),
        "heartbeat_interval_seconds": float(heartbeat_interval),
        "points": ordered,
    }


def _load_kwant_result_metadata(kwant_dir: Path) -> dict[str, Any]:
    result_path = kwant_dir / "kwant_reference.json"
    if not result_path.exists():
        return {}
    payload = _json_load(result_path)
    return payload if isinstance(payload, dict) else {}


def _resolve_expected_grid(
    kwant_dir: Path,
    summary: dict[str, Any],
    spec: NanowireBenchmarkSpec,
) -> tuple[list[int], list[float]]:
    meta = _load_kwant_result_metadata(kwant_dir)
    kwant_rows = list((summary.get("kwant") or {}).get("rows", []))
    rgf_rows = list((summary.get("rgf") or {}).get("rows", []))

    thicknesses: list[int] = []
    raw_thicknesses = meta.get("thicknesses")
    if isinstance(raw_thicknesses, list) and raw_thicknesses:
        thicknesses = [int(v) for v in raw_thicknesses]

    energies: list[float] = []
    raw_energies = meta.get("energies_rel_fermi_ev")
    if isinstance(raw_energies, list) and raw_energies:
        energies = [_canon_float(v) for v in raw_energies]

    if not thicknesses:
        thicknesses = sorted({int(row["thickness_uc"]) for row in kwant_rows + rgf_rows} or {int(v) for v in spec.thicknesses_uc})
    else:
        thicknesses = sorted(set(thicknesses) | {int(row["thickness_uc"]) for row in kwant_rows + rgf_rows})

    derived_energies = {
        _canon_float(row.get("energy_rel_fermi_ev"))
        for row in kwant_rows + rgf_rows
        if row.get("energy_rel_fermi_ev") is not None
    }
    if not energies:
        energies = sorted(derived_energies or {_canon_float(v) for v in spec.energies_ev})
    else:
        energies = sorted(set(energies) | derived_energies)

    return thicknesses, energies


def _load_runtime_cert(result_path: Path) -> dict[str, Any]:
    payload = _json_load(result_path)
    if not isinstance(payload, dict):
        return {}
    runtime_cert = payload.get("runtime_cert")
    if isinstance(runtime_cert, dict):
        return dict(runtime_cert)
    results = payload.get("transport_results", {})
    if isinstance(results, dict):
        meta = results.get("meta", {})
        if isinstance(meta, dict):
            inner = meta.get("rgf_runtime_cert", {})
            if isinstance(inner, dict):
                return dict(inner)
    return {}


def _find_latest_pbs_file(transport_result_path: Path) -> Path | None:
    transport_dir = transport_result_path.parent
    candidates = [
        path
        for path in transport_dir.glob("*.pbs")
        if path.is_file() and "attempts" not in path.parts
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime)
    return candidates[-1]


def _extract_matching_line(path: Path | None, needle: str) -> str | None:
    if path is None or not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if needle in stripped:
            return stripped
    return None


def _extract_last_matching_line(path: Path | None, needle: str) -> str | None:
    if path is None or not path.exists():
        return None
    match: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if needle in stripped:
            match = stripped
    return match


def _pbs_queue(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("#PBS -q "):
            return stripped.split("#PBS -q ", 1)[1].strip()
    return None


def _pbs_forbids_fork(path: Path | None) -> bool | None:
    if path is None or not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if "--launch fork" in text:
        return False
    if "Fork-based launchstyle is FORBIDDEN" in text or "fork-based launchstyle is FORBIDDEN" in text:
        return True
    if "all parallel execution uses mpirun backend" in text:
        return True
    return None


def _compare_point(
    *,
    reference: float,
    rgf: float,
    abs_tol: float,
    rel_tol: float,
    zero_tol: float,
) -> tuple[float, float | None, bool]:
    abs_err = abs(float(rgf) - float(reference))
    ref_f = float(reference)
    if abs(ref_f) <= float(zero_tol):
        rel_err = 0.0 if abs_err <= float(zero_tol) else math.inf
        passed = abs_err <= float(zero_tol)
    else:
        rel_err = abs_err / abs(ref_f)
        passed = abs_err <= float(abs_tol) and rel_err <= float(rel_tol)
    return float(abs_err), (None if not math.isfinite(rel_err) else float(rel_err)), bool(passed)


def build_overlap_rows(
    summary: dict[str, Any],
    *,
    spec: NanowireBenchmarkSpec,
    kwant_runtime_bounds: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    kwant_rows = _rows_by_key(list((summary.get("kwant") or {}).get("rows", [])))
    rgf_rows = _rows_by_key(list((summary.get("rgf") or {}).get("rows", [])))
    kwant_runtime_by_key = {
        _point_key(int(row["thickness_uc"]), float(row["energy_abs_ev"])): row
        for row in list((kwant_runtime_bounds or {}).get("points", []))
    }

    overlap_rows: list[dict[str, Any]] = []
    for key in sorted(kwant_rows.keys() & rgf_rows.keys()):
        kwant_row = dict(kwant_rows[key])
        rgf_row = dict(rgf_rows[key])
        energy_rel = kwant_row.get("energy_rel_fermi_ev", rgf_row.get("energy_rel_fermi_ev"))
        abs_err, rel_err, passed = _compare_point(
            reference=float(kwant_row["transmission_e2_over_h"]),
            rgf=float(rgf_row["transmission_e2_over_h"]),
            abs_tol=float(spec.abs_tol),
            rel_tol=float(spec.rel_tol),
            zero_tol=float(spec.zero_tol),
        )
        runtime_cert: dict[str, Any] = {}
        rgf_result_path = Path(str(rgf_row.get("source_path", ""))).expanduser()
        if rgf_row.get("source_kind") == "rgf_transport_result" and rgf_result_path.exists():
            runtime_cert = _load_runtime_cert(rgf_result_path.resolve())
        kwant_runtime = kwant_runtime_by_key.get(key, {})
        rgf_wall_seconds = _safe_float(runtime_cert.get("wall_seconds"))
        kwant_lower = _safe_float(kwant_runtime.get("elapsed_lower_bound_s"))
        kwant_upper = _safe_float(kwant_runtime.get("elapsed_upper_bound_s"))
        speedup_lower = None
        if kwant_lower is not None and rgf_wall_seconds is not None and rgf_wall_seconds > 0.0:
            speedup_lower = kwant_lower / rgf_wall_seconds

        overlap_rows.append(
            {
                "thickness_uc": int(kwant_row["thickness_uc"]),
                "energy_abs_ev": float(kwant_row["energy_abs_ev"]),
                "energy_rel_fermi_ev": (
                    None if energy_rel is None else float(energy_rel)
                ),
                "kwant_transmission_e2_over_h": float(kwant_row["transmission_e2_over_h"]),
                "rgf_transmission_e2_over_h": float(rgf_row["transmission_e2_over_h"]),
                "abs_err": float(abs_err),
                "rel_err": rel_err,
                "status": ("ok" if passed else "failed"),
                "kwant_source_kind": kwant_row.get("source_kind"),
                "kwant_source_path": kwant_row.get("source_path"),
                "rgf_source_kind": rgf_row.get("source_kind"),
                "rgf_source_path": rgf_row.get("source_path"),
                "rgf_wall_seconds": rgf_wall_seconds,
                "kwant_runtime_lower_bound_s": kwant_lower,
                "kwant_runtime_upper_bound_s": kwant_upper,
                "kwant_completed": bool(kwant_runtime.get("completed", False)),
                "speedup_lower_bound": speedup_lower,
            }
        )
    return overlap_rows


def build_all_points_rows(
    summary: dict[str, Any],
    overlap_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    kwant_rows = _rows_by_key(list((summary.get("kwant") or {}).get("rows", [])))
    rgf_rows = _rows_by_key(list((summary.get("rgf") or {}).get("rows", [])))
    overlap_by_key = {
        _point_key(int(row["thickness_uc"]), float(row["energy_abs_ev"])): row for row in overlap_rows
    }
    all_rows: list[dict[str, Any]] = []
    for key in sorted(set(kwant_rows) | set(rgf_rows)):
        kwant_row = kwant_rows.get(key)
        rgf_row = rgf_rows.get(key)
        overlap_row = overlap_by_key.get(key)
        energy_rel = None
        if kwant_row is not None:
            energy_rel = kwant_row.get("energy_rel_fermi_ev")
        if energy_rel is None and rgf_row is not None:
            energy_rel = rgf_row.get("energy_rel_fermi_ev")
        state = "missing"
        if kwant_row is not None and rgf_row is not None:
            state = "overlap" if overlap_row is None else f"overlap_{overlap_row['status']}"
        elif kwant_row is not None:
            state = "kwant_only"
        elif rgf_row is not None:
            state = "rgf_only"
        all_rows.append(
            {
                "thickness_uc": int(key[0]),
                "energy_abs_ev": float(key[1]),
                "energy_rel_fermi_ev": (None if energy_rel is None else float(energy_rel)),
                "state": state,
                "kwant_transmission_e2_over_h": (
                    None if kwant_row is None else float(kwant_row["transmission_e2_over_h"])
                ),
                "rgf_transmission_e2_over_h": (
                    None if rgf_row is None else float(rgf_row["transmission_e2_over_h"])
                ),
                "kwant_source_kind": (None if kwant_row is None else kwant_row.get("source_kind")),
                "rgf_source_kind": (None if rgf_row is None else rgf_row.get("source_kind")),
                "abs_err": (None if overlap_row is None else float(overlap_row["abs_err"])),
                "rel_err": (None if overlap_row is None else overlap_row.get("rel_err")),
            }
        )
    return all_rows


def build_speed_rows(overlap_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in overlap_rows:
        if not bool(row.get("kwant_completed")):
            continue
        rgf_wall = _safe_float(row.get("rgf_wall_seconds"))
        kwant_lower = _safe_float(row.get("kwant_runtime_lower_bound_s"))
        kwant_upper = _safe_float(row.get("kwant_runtime_upper_bound_s"))
        if rgf_wall is None or kwant_lower is None or rgf_wall <= 0.0:
            continue
        out.append(
            {
                "thickness_uc": int(row["thickness_uc"]),
                "energy_abs_ev": float(row["energy_abs_ev"]),
                "energy_rel_fermi_ev": row.get("energy_rel_fermi_ev"),
                "rgf_wall_seconds": float(rgf_wall),
                "kwant_runtime_lower_bound_s": float(kwant_lower),
                "kwant_runtime_upper_bound_s": kwant_upper,
                "speedup_lower_bound": float(kwant_lower / rgf_wall),
            }
        )
    return out


def _geometric_mean(values: list[float]) -> float | None:
    positives = [float(value) for value in values if float(value) > 0.0]
    if not positives:
        return None
    return float(math.exp(sum(math.log(value) for value in positives) / len(positives)))


def build_speed_summary(speed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    speedups = [float(row["speedup_lower_bound"]) for row in speed_rows if row.get("speedup_lower_bound") is not None]
    if not speedups:
        return {
            "completed_overlap_points": 0,
            "speedup_lower_bound_min": None,
            "speedup_lower_bound_median": None,
            "speedup_lower_bound_geomean": None,
        }
    return {
        "completed_overlap_points": len(speedups),
        "speedup_lower_bound_min": float(min(speedups)),
        "speedup_lower_bound_median": float(statistics.median(speedups)),
        "speedup_lower_bound_geomean": _geometric_mean(speedups),
    }


def _resolve_runtime_path_evidence(axis_root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    kwant_dir = axis_root / "kwant"
    kwant_pbs = kwant_dir / "kwant_reference.pbs"
    kwant_mpirun = _extract_last_matching_line(kwant_pbs, "mpirun ")
    overlap_rows = list((summary.get("rgf") or {}).get("rows", []))
    rgf_result_path: Path | None = None
    for row in overlap_rows:
        if row.get("source_kind") == "rgf_transport_result":
            candidate = Path(str(row.get("source_path", ""))).expanduser().resolve()
            if candidate.exists():
                rgf_result_path = candidate
                break
    runtime_cert = _load_runtime_cert(rgf_result_path) if rgf_result_path is not None else {}
    rgf_pbs = _find_latest_pbs_file(rgf_result_path) if rgf_result_path is not None else None
    return {
        "kwant_pbs_path": (None if kwant_pbs is None or not kwant_pbs.exists() else str(kwant_pbs.resolve())),
        "kwant_queue": _pbs_queue(kwant_pbs),
        "kwant_mpirun_line": kwant_mpirun,
        "kwant_forbids_fork_launchstyle": _pbs_forbids_fork(kwant_pbs),
        "rgf_transport_result_path": (None if rgf_result_path is None else str(rgf_result_path)),
        "rgf_pbs_path": (None if rgf_pbs is None else str(rgf_pbs.resolve())),
        "rgf_mpirun_line": (
            _extract_last_matching_line(rgf_pbs, "wtec_rgf_runner")
            or _extract_last_matching_line(rgf_pbs, "mpirun ")
        ),
        "rgf_forbids_fork_launchstyle": _pbs_forbids_fork(rgf_pbs),
        "rgf_queue": runtime_cert.get("queue"),
        "rgf_mode": runtime_cert.get("mode"),
        "rgf_mpi_size": runtime_cert.get("mpi_size"),
        "rgf_omp_threads": runtime_cert.get("omp_threads"),
        "rgf_sigma_source": runtime_cert.get("full_finite_sigma_source"),
    }


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib import patches as mpatches

    return plt, mcolors, mpatches


def _point_label(row: dict[str, Any]) -> str:
    energy_rel = row.get("energy_rel_fermi_ev")
    if energy_rel is None:
        return f"t{int(row['thickness_uc'])}"
    return f"t{int(row['thickness_uc'])}\nE={float(energy_rel):+.1f}"


def _save_coverage_matrix(
    out_path: Path,
    *,
    thicknesses: list[int],
    energies: list[float],
    summary: dict[str, Any],
    overlap_rows: list[dict[str, Any]],
) -> None:
    plt, mcolors, mpatches = _import_pyplot()
    kwant_rows = _rows_by_key(list((summary.get("kwant") or {}).get("rows", [])))
    rgf_rows = _rows_by_key(list((summary.get("rgf") or {}).get("rows", [])))
    overlap_by_key = {
        _point_key(int(row["thickness_uc"]), float(row["energy_abs_ev"])): row for row in overlap_rows
    }
    fermi_ev = (summary.get("kwant") or {}).get("fermi_ev")
    grid: list[list[int]] = []
    for thickness_uc in thicknesses:
        row_vals: list[int] = []
        for energy_rel in energies:
            energy_abs = float(energy_rel)
            if fermi_ev is not None:
                energy_abs = float(fermi_ev) + float(energy_rel)
            key = _point_key(int(thickness_uc), energy_abs)
            state = "missing"
            if key in overlap_by_key:
                state = "overlap_ok" if overlap_by_key[key]["status"] == "ok" else "overlap_fail"
            elif key in kwant_rows:
                state = "kwant_only"
            elif key in rgf_rows:
                state = "rgf_only"
            row_vals.append(_COVERAGE_STATE_ORDER.index(state))
        grid.append(row_vals)

    fig, ax = plt.subplots(figsize=(max(6.0, len(energies) * 1.2), max(4.0, len(thicknesses) * 0.6)))
    cmap = mcolors.ListedColormap([_COVERAGE_STATE_COLORS[name] for name in _COVERAGE_STATE_ORDER])
    ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(energies)))
    ax.set_xticklabels([f"{energy:+.1f}" for energy in energies])
    ax.set_xlabel("Energy rel. Fermi (eV)")
    ax.set_yticks(range(len(thicknesses)))
    ax.set_yticklabels([str(thickness) for thickness in thicknesses])
    ax.set_ylabel("Thickness (uc)")
    ax.set_title("Benchmark Coverage Matrix")
    legend_handles = [
        mpatches.Patch(color=_COVERAGE_STATE_COLORS[name], label=name.replace("_", " "))
        for name in _COVERAGE_STATE_ORDER
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_transmission_all_points(
    out_path: Path,
    *,
    summary: dict[str, Any],
) -> None:
    plt, _, _ = _import_pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, solver_name, rows in [
        (axes[0], "Kwant", list((summary.get("kwant") or {}).get("rows", []))),
        (axes[1], "RGF", list((summary.get("rgf") or {}).get("rows", []))),
    ]:
        thicknesses = sorted({int(row["thickness_uc"]) for row in rows})
        if not rows:
            ax.text(0.5, 0.5, "No points available", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{solver_name} Current Points")
            continue
        for thickness_uc in thicknesses:
            subset = [row for row in rows if int(row["thickness_uc"]) == thickness_uc]
            subset.sort(key=lambda row: float(row.get("energy_rel_fermi_ev", 0.0)))
            xs = [float(row.get("energy_rel_fermi_ev", 0.0)) for row in subset]
            ys = [float(row["transmission_e2_over_h"]) for row in subset]
            ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"t={thickness_uc}")
        ax.set_ylabel("Transmission (e²/h)")
        ax.set_title(f"{solver_name} Current Points")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize="small")
    axes[1].set_xlabel("Energy rel. Fermi (eV)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_transmission_overlap_comparison(
    out_path: Path,
    *,
    overlap_rows: list[dict[str, Any]],
) -> None:
    plt, _, _ = _import_pyplot()
    thicknesses = sorted({int(row["thickness_uc"]) for row in overlap_rows})
    n_panels = max(1, len(thicknesses))
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, max(3.5, 3.2 * n_panels)), sharex=True)
    if n_panels == 1:
        axes = [axes]
    for ax, thickness_uc in zip(axes, thicknesses):
        subset = [row for row in overlap_rows if int(row["thickness_uc"]) == thickness_uc]
        subset.sort(key=lambda row: float(row.get("energy_rel_fermi_ev", 0.0)))
        xs = [float(row.get("energy_rel_fermi_ev", 0.0)) for row in subset]
        kwant = [float(row["kwant_transmission_e2_over_h"]) for row in subset]
        rgf = [float(row["rgf_transmission_e2_over_h"]) for row in subset]
        ax.plot(xs, kwant, marker="o", linewidth=1.8, label="Kwant")
        ax.plot(xs, rgf, marker="s", linewidth=1.8, label="RGF")
        ax.set_ylabel("Transmission (e²/h)")
        ax.set_title(f"Overlap Comparison: thickness {thickness_uc} uc")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Energy rel. Fermi (eV)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_error_overlap(
    out_path: Path,
    *,
    overlap_rows: list[dict[str, Any]],
    spec: NanowireBenchmarkSpec,
) -> None:
    plt, _, _ = _import_pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    thicknesses = sorted({int(row["thickness_uc"]) for row in overlap_rows})
    if not overlap_rows:
        for ax in axes:
            ax.text(0.5, 0.5, "No overlap points", transform=ax.transAxes, ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return
    for thickness_uc in thicknesses:
        subset = [row for row in overlap_rows if int(row["thickness_uc"]) == thickness_uc]
        subset.sort(key=lambda row: float(row.get("energy_rel_fermi_ev", 0.0)))
        xs = [float(row.get("energy_rel_fermi_ev", 0.0)) for row in subset]
        abs_err = [float(row["abs_err"]) for row in subset]
        rel_err = [0.0 if row.get("rel_err") is None else float(row["rel_err"]) for row in subset]
        axes[0].plot(xs, abs_err, marker="o", linewidth=1.5, label=f"t={thickness_uc}")
        axes[1].plot(xs, rel_err, marker="o", linewidth=1.5, label=f"t={thickness_uc}")
    axes[0].axhline(float(spec.abs_tol), linestyle="--", color="#dc2626", linewidth=1.2, label="abs_tol")
    axes[1].axhline(float(spec.rel_tol), linestyle="--", color="#dc2626", linewidth=1.2, label="rel_tol")
    axes[0].set_ylabel("Absolute error")
    axes[1].set_ylabel("Relative error")
    axes[1].set_xlabel("Energy rel. Fermi (eV)")
    axes[0].set_title("Overlap Error vs Energy")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_runtime_speed_bounds(
    out_path: Path,
    *,
    speed_rows: list[dict[str, Any]],
) -> None:
    plt, _, _ = _import_pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    if not speed_rows:
        for ax in axes:
            ax.text(0.5, 0.5, "No completed overlap speed data", transform=ax.transAxes, ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    labels = [_point_label(row) for row in speed_rows]
    x_vals = list(range(len(speed_rows)))
    rgf_wall = [float(row["rgf_wall_seconds"]) for row in speed_rows]
    kwant_lower = [float(row["kwant_runtime_lower_bound_s"]) for row in speed_rows]
    kwant_upper = [
        float(row["kwant_runtime_upper_bound_s"])
        if row.get("kwant_runtime_upper_bound_s") is not None
        else float(row["kwant_runtime_lower_bound_s"])
        for row in speed_rows
    ]
    speedup = [float(row["speedup_lower_bound"]) for row in speed_rows]

    axes[0].bar(x_vals, rgf_wall, color="#2563eb")
    axes[0].set_ylabel("RGF wall seconds")
    axes[0].set_title("RGF Exact Runtime Per Completed Overlap Point")
    axes[0].grid(axis="y", alpha=0.25)

    yerr = [max(0.0, upper - lower) for lower, upper in zip(kwant_lower, kwant_upper)]
    axes[1].errorbar(
        x_vals,
        kwant_lower,
        yerr=yerr,
        fmt="o",
        color="#f59e0b",
        ecolor="#f59e0b",
        elinewidth=1.4,
        capsize=4,
        label="Kwant runtime bounds",
    )
    axes[1].set_ylabel("Kwant runtime (s)")
    axes[1].grid(axis="y", alpha=0.25)
    ax2 = axes[1].twinx()
    ax2.plot(x_vals, speedup, color="#16a34a", marker="s", linewidth=1.6, label="Speedup lower bound")
    ax2.set_ylabel("Speedup lower bound")
    axes[1].set_title("Kwant Runtime Bounds And Derived Lower-Bound Speedup")

    handles_1, labels_1 = axes[1].get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    axes[1].legend(handles_1 + handles_2, labels_1 + labels_2, loc="best", fontsize="small")
    axes[1].set_xticks(x_vals)
    axes[1].set_xticklabels(labels)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_csv(path: Path, *, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_table(rows: list[dict[str, Any]], *, columns: list[tuple[str, str]]) -> list[str]:
    if not rows:
        return ["No rows available.", ""]

    def _fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.12g}"
        return str(value)

    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_fmt(row.get(key)) for key, _label in columns) + " |"
        for row in rows
    ]
    return [header, sep, *body, ""]


def _render_markdown(report: dict[str, Any]) -> str:
    comparison = report.get("comparison", {})
    coverage = report.get("coverage", {})
    runtime_path = report.get("runtime_path", {})
    speed = report.get("speed", {})
    acceptance = report.get("acceptance", {})
    overlap_rows = list(report.get("overlap_rows_preview", []))
    speed_rows = list(report.get("speed_rows_preview", []))

    verdict_lines = [
        "# Nanowire Benchmark Current-Evidence Report",
        "",
        f"- generated_at_utc: `{report.get('generated_at_utc')}`",
        f"- benchmark_root: `{report.get('benchmark_root')}`",
        f"- axis_root: `{report.get('axis_root')}`",
        f"- scope: `{report.get('scope')}`",
        f"- required_exact_eta: `{report.get('required_exact_eta')}`",
        "",
        "## Executive Verdict",
        "",
        f"- current_evidence_status: `{report.get('current_evidence_status')}`",
        f"- overlap_points: `{coverage.get('overlap_points')}` / `{coverage.get('expected_points')}` expected benchmark points",
        f"- kwant_completed_points: `{coverage.get('kwant_completed_points')}`",
        f"- rgf_current_points: `{coverage.get('rgf_current_points')}`",
        f"- comparison_status: `{comparison.get('status')}`",
        f"- max_abs_err: `{comparison.get('max_abs_err')}`",
        f"- max_rel_err: `{comparison.get('max_rel_err')}`",
        (
            "- interpretation: current exact-sigma overlap validates on the currently completed overlap points, "
            "but this is not yet a full-benchmark acceptance claim."
        ),
        "",
        "## Evidence Scope And Contract",
        "",
        "- This bundle is generated from the current `tmp/devise_transport_benchmark` root only.",
        "- Valid comparison points are defined by the same exact-sigma overlap filter used by the benchmark progress comparator.",
        "- Stale exact-sigma rows with the wrong eta are excluded before any comparison, speed, or coverage metric is reported.",
        "- Kwant timing is reported as lower/upper bounds derived from live `heartbeat` lines, not as exact per-point wall time.",
        "",
        "## Runtime Path Proof",
        "",
        f"- kwant_queue: `{runtime_path.get('kwant_queue')}`",
        f"- kwant_mpirun_line: `{runtime_path.get('kwant_mpirun_line')}`",
        f"- kwant_forbids_fork_launchstyle: `{runtime_path.get('kwant_forbids_fork_launchstyle')}`",
        f"- rgf_queue: `{runtime_path.get('rgf_queue')}`",
        f"- rgf_mode: `{runtime_path.get('rgf_mode')}`",
        f"- rgf_sigma_source: `{runtime_path.get('rgf_sigma_source')}`",
        f"- rgf_mpi_size: `{runtime_path.get('rgf_mpi_size')}`",
        f"- rgf_omp_threads: `{runtime_path.get('rgf_omp_threads')}`",
        f"- rgf_mpirun_line: `{runtime_path.get('rgf_mpirun_line')}`",
        f"- rgf_forbids_fork_launchstyle: `{runtime_path.get('rgf_forbids_fork_launchstyle')}`",
        "",
        "## Kwant Vs RGF Transmission Comparison",
        "",
        "![Coverage matrix](coverage_matrix.png)",
        "",
        "![All current transmissions](transmission_all_points.png)",
        "",
        "![Overlap comparison](transmission_overlap_comparison.png)",
        "",
        "![Overlap error](error_overlap.png)",
        "",
        "### Overlap Preview",
        "",
        *_markdown_table(
            overlap_rows,
            columns=[
                ("thickness_uc", "thickness_uc"),
                ("energy_rel_fermi_ev", "energy_rel_fermi_ev"),
                ("kwant_transmission_e2_over_h", "kwant"),
                ("rgf_transmission_e2_over_h", "rgf"),
                ("abs_err", "abs_err"),
                ("rel_err", "rel_err"),
                ("status", "status"),
            ],
        ),
        "## Runtime And Speed Evidence",
        "",
        "![Runtime and speed](runtime_speed_bounds.png)",
        "",
        f"- completed_overlap_points_for_speed: `{speed.get('completed_overlap_points')}`",
        f"- speedup_lower_bound_min: `{speed.get('speedup_lower_bound_min')}`",
        f"- speedup_lower_bound_median: `{speed.get('speedup_lower_bound_median')}`",
        f"- speedup_lower_bound_geomean: `{speed.get('speedup_lower_bound_geomean')}`",
        "",
        "### Speed Preview",
        "",
        *_markdown_table(
            speed_rows,
            columns=[
                ("thickness_uc", "thickness_uc"),
                ("energy_rel_fermi_ev", "energy_rel_fermi_ev"),
                ("rgf_wall_seconds", "rgf_wall_s"),
                ("kwant_runtime_lower_bound_s", "kwant_lower_s"),
                ("kwant_runtime_upper_bound_s", "kwant_upper_s"),
                ("speedup_lower_bound", "speedup_lower_bound"),
            ],
        ),
        "## Acceptance Status",
        "",
        f"- current_overlap_validation: `{acceptance.get('current_overlap_validation')}`",
        f"- runtime_path_proof: `{acceptance.get('runtime_path_proof')}`",
        f"- benchmark_complete: `{acceptance.get('benchmark_complete')}`",
        f"- benchmark_wide_speedup_ge_5: `{acceptance.get('benchmark_wide_speedup_ge_5')}`",
        f"- clean_restart_path: `{acceptance.get('clean_restart_path')}`",
        "",
        "## Remaining Gaps",
        "",
    ]
    for item in report.get("remaining_gaps", []):
        verdict_lines.append(f"- {item}")
    verdict_lines.extend(
        [
            "",
            "## Bundle Contents",
            "",
        ]
    )
    for key, rel_path in sorted((report.get("artifacts") or {}).items()):
        verdict_lines.append(f"- {key}: `{rel_path}`")
    verdict_lines.append("")
    return "\n".join(verdict_lines)


def generate_nanowire_benchmark_report(
    *,
    benchmark_root: str | Path,
    out_dir: str | Path,
    required_exact_eta: float = 1.0e-8,
    spec: NanowireBenchmarkSpec | None = None,
) -> dict[str, Any]:
    benchmark_root_path = Path(benchmark_root).expanduser().resolve()
    out_dir_path = Path(out_dir).expanduser().resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    cfg = spec or NanowireBenchmarkSpec()
    axis_root = _detect_axis_root(benchmark_root_path, spec=cfg)
    kwant_dir = axis_root / "kwant"
    rgf_root = axis_root / "rgf"
    summary = compare_partial_benchmark_progress(
        kwant_dir=kwant_dir,
        rgf_root=rgf_root,
        spec=cfg,
        required_exact_eta=float(required_exact_eta),
    )
    kwant_runtime = scan_kwant_runtime_bounds(kwant_dir / "wtec_job.log")
    overlap_rows = build_overlap_rows(summary, spec=cfg, kwant_runtime_bounds=kwant_runtime)
    all_points_rows = build_all_points_rows(summary, overlap_rows)
    speed_rows = build_speed_rows(overlap_rows)
    speed_summary = build_speed_summary(speed_rows)
    thicknesses, energies = _resolve_expected_grid(kwant_dir, summary, cfg)
    runtime_path = _resolve_runtime_path_evidence(axis_root, summary)
    kwant_meta = _load_kwant_result_metadata(kwant_dir)

    kwant_completed_points = int((summary.get("kwant") or {}).get("row_count", 0) or 0)
    rgf_current_points = int((summary.get("rgf") or {}).get("row_count", 0) or 0)
    expected_points = int(len(thicknesses) * len(energies))
    benchmark_complete = kwant_completed_points >= expected_points and len(summary.get("missing_in_rgf", [])) == 0
    comparison = dict(summary.get("comparison") or {})
    current_overlap_validation = "pass" if comparison.get("status") == "ok" and overlap_rows else "fail"
    runtime_path_proof = "pass"
    if runtime_path.get("kwant_forbids_fork_launchstyle") is False or runtime_path.get("rgf_forbids_fork_launchstyle") is False:
        runtime_path_proof = "fail"
    if not runtime_path.get("kwant_mpirun_line") or not runtime_path.get("rgf_mpirun_line"):
        runtime_path_proof = "inconclusive"
    speedup_ge_5 = "inconclusive"
    geomean = speed_summary.get("speedup_lower_bound_geomean")
    if benchmark_complete and geomean is not None:
        speedup_ge_5 = "pass" if float(geomean) >= 5.0 else "fail"

    artifacts = {
        "report_md": "report.md",
        "report_json": "report.json",
        "overlap_points_csv": "overlap_points.csv",
        "all_points_csv": "all_points.csv",
        "speed_summary_csv": "speed_summary.csv",
        "coverage_matrix_png": "coverage_matrix.png",
        "transmission_all_points_png": "transmission_all_points.png",
        "transmission_overlap_comparison_png": "transmission_overlap_comparison.png",
        "error_overlap_png": "error_overlap.png",
        "runtime_speed_bounds_png": "runtime_speed_bounds.png",
    }

    overlap_csv_rows = []
    for row in overlap_rows:
        overlap_csv_rows.append(
            {
                "thickness_uc": row["thickness_uc"],
                "energy_abs_ev": row["energy_abs_ev"],
                "energy_rel_fermi_ev": row.get("energy_rel_fermi_ev"),
                "kwant_transmission_e2_over_h": row["kwant_transmission_e2_over_h"],
                "rgf_transmission_e2_over_h": row["rgf_transmission_e2_over_h"],
                "abs_err": row["abs_err"],
                "rel_err": row.get("rel_err"),
                "status": row["status"],
                "rgf_wall_seconds": row.get("rgf_wall_seconds"),
                "kwant_runtime_lower_bound_s": row.get("kwant_runtime_lower_bound_s"),
                "kwant_runtime_upper_bound_s": row.get("kwant_runtime_upper_bound_s"),
                "speedup_lower_bound": row.get("speedup_lower_bound"),
            }
        )

    _write_csv(
        out_dir_path / artifacts["overlap_points_csv"],
        rows=overlap_csv_rows,
        fieldnames=[
            "thickness_uc",
            "energy_abs_ev",
            "energy_rel_fermi_ev",
            "kwant_transmission_e2_over_h",
            "rgf_transmission_e2_over_h",
            "abs_err",
            "rel_err",
            "status",
            "rgf_wall_seconds",
            "kwant_runtime_lower_bound_s",
            "kwant_runtime_upper_bound_s",
            "speedup_lower_bound",
        ],
    )
    _write_csv(
        out_dir_path / artifacts["all_points_csv"],
        rows=all_points_rows,
        fieldnames=[
            "thickness_uc",
            "energy_abs_ev",
            "energy_rel_fermi_ev",
            "state",
            "kwant_transmission_e2_over_h",
            "rgf_transmission_e2_over_h",
            "kwant_source_kind",
            "rgf_source_kind",
            "abs_err",
            "rel_err",
        ],
    )
    _write_csv(
        out_dir_path / artifacts["speed_summary_csv"],
        rows=speed_rows,
        fieldnames=[
            "thickness_uc",
            "energy_abs_ev",
            "energy_rel_fermi_ev",
            "rgf_wall_seconds",
            "kwant_runtime_lower_bound_s",
            "kwant_runtime_upper_bound_s",
            "speedup_lower_bound",
        ],
    )

    _save_coverage_matrix(
        out_dir_path / artifacts["coverage_matrix_png"],
        thicknesses=thicknesses,
        energies=energies,
        summary=summary,
        overlap_rows=overlap_rows,
    )
    _save_transmission_all_points(
        out_dir_path / artifacts["transmission_all_points_png"],
        summary=summary,
    )
    _save_transmission_overlap_comparison(
        out_dir_path / artifacts["transmission_overlap_comparison_png"],
        overlap_rows=overlap_rows,
    )
    _save_error_overlap(
        out_dir_path / artifacts["error_overlap_png"],
        overlap_rows=overlap_rows,
        spec=cfg,
    )
    _save_runtime_speed_bounds(
        out_dir_path / artifacts["runtime_speed_bounds_png"],
        speed_rows=speed_rows,
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope": "current_evidence",
        "benchmark_root": str(benchmark_root_path),
        "axis_root": str(axis_root),
        "required_exact_eta": float(required_exact_eta),
        "current_evidence_status": summary.get("status", "unknown"),
        "comparison": {
            "status": comparison.get("status"),
            "checked_points": comparison.get("checked_points"),
            "max_abs_err": comparison.get("max_abs_err"),
            "max_rel_err": comparison.get("max_rel_err"),
        },
        "coverage": {
            "expected_points": expected_points,
            "expected_thicknesses_uc": thicknesses,
            "expected_energies_rel_fermi_ev": energies,
            "kwant_completed_points": kwant_completed_points,
            "rgf_current_points": rgf_current_points,
            "overlap_points": int(summary.get("overlap_points", 0) or 0),
            "kwant_task_count_expected": kwant_meta.get("task_count_expected"),
            "kwant_task_count_completed": kwant_meta.get("task_count_completed"),
        },
        "speed": speed_summary,
        "runtime_path": runtime_path,
        "acceptance": {
            "current_overlap_validation": current_overlap_validation,
            "runtime_path_proof": runtime_path_proof,
            "benchmark_complete": ("pass" if benchmark_complete else "open"),
            "benchmark_wide_speedup_ge_5": speedup_ge_5,
            "clean_restart_path": "not_evaluated_by_report_generator",
        },
        "remaining_gaps": [
            "This bundle reports current overlap evidence, not a completed 35-point benchmark.",
            "Benchmark-wide raw/fit acceptance remains open until the full reference case completes.",
            "Benchmark-wide >=5x speedup remains open until the full reference case completes with accepted accuracy.",
            "The ScaLAPACK/SIESTA restart blocker is external to this report bundle and remains unresolved.",
        ],
        "artifacts": dict(artifacts),
        "overlap_rows_preview": overlap_csv_rows[:12],
        "speed_rows_preview": speed_rows[:12],
    }

    report_md = _render_markdown(report)
    (out_dir_path / artifacts["report_md"]).write_text(report_md, encoding="utf-8")
    (out_dir_path / artifacts["report_json"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a current-evidence nanowire benchmark report bundle.")
    ap.add_argument("--benchmark-root", required=True, help="Benchmark output root, typically tmp/devise_transport_benchmark")
    ap.add_argument("--out-dir", required=True, help="Destination directory for markdown, JSON, CSV, and PNG outputs")
    ap.add_argument(
        "--required-exact-eta",
        type=float,
        default=1.0e-8,
        help="Exact-sigma eta contract used to filter reusable overlap points",
    )
    ns = ap.parse_args(argv)

    report = generate_nanowire_benchmark_report(
        benchmark_root=ns.benchmark_root,
        out_dir=ns.out_dir,
        required_exact_eta=float(ns.required_exact_eta),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
