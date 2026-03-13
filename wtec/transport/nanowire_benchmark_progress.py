from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from wtec.transport.nanowire_benchmark import (
    NanowireBenchmarkSpec,
    compare_reference_and_rgf,
)
from wtec.transport.rgf_postprocess import load_rgf_raw_result


_KWANT_DONE_RE = re.compile(
    r"\[kwant-bench\]\[rank=(?P<rank>\d+)\]\s+done\s+"
    r"thickness_uc=(?P<thickness>\d+)\s+"
    r"energy_abs_ev=(?P<energy_abs>[-+0-9.eE]+)\s+"
    r"transmission=(?P<transmission>[-+0-9.eE]+)"
)

_SOURCE_PRIORITY = {
    "kwant_log": 1,
    "kwant_result": 2,
    "rgf_progress": 1,
    "rgf_raw_result": 2,
    "rgf_transport_result": 3,
}


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canon_float(value: Any, *, digits: int = 12) -> float:
    return round(float(value), digits)


def _point_key(thickness_uc: int, energy_abs_ev: float) -> tuple[int, float]:
    return int(thickness_uc), _canon_float(energy_abs_ev)


def _row_key(row: dict[str, Any]) -> tuple[int, float]:
    return _point_key(int(row["thickness_uc"]), float(row["energy_abs_ev"]))


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[tuple[int, float], dict[str, Any]] = {}
    for row in rows:
        row = dict(row)
        key = _row_key(row)
        prev = chosen.get(key)
        if prev is None:
            chosen[key] = row
            continue
        prev_rank = int(prev.get("_priority", 0))
        row_rank = int(row.get("_priority", 0))
        prev_mtime = float(prev.get("_mtime", 0.0))
        row_mtime = float(row.get("_mtime", 0.0))
        if row_rank > prev_rank or (row_rank == prev_rank and row_mtime >= prev_mtime):
            chosen[key] = row
    out: list[dict[str, Any]] = []
    for row in chosen.values():
        row.pop("_priority", None)
        row.pop("_mtime", None)
        out.append(row)
    out.sort(key=lambda item: (int(item["thickness_uc"]), float(item["energy_abs_ev"])))
    return out


def _row_missing_fields(
    thickness_uc: int,
    energy_abs_ev: float,
    *,
    energy_rel_fermi_ev: float | None,
    transmission: float,
    source_kind: str,
    source_path: Path,
) -> dict[str, Any]:
    row = {
        "thickness_uc": int(thickness_uc),
        "energy_abs_ev": float(energy_abs_ev),
        "transmission_e2_over_h": float(transmission),
        "source_kind": str(source_kind),
        "source_path": str(source_path),
        "_priority": int(_SOURCE_PRIORITY.get(source_kind, 0)),
        "_mtime": float(source_path.stat().st_mtime if source_path.exists() else 0.0),
    }
    if energy_rel_fermi_ev is not None:
        row["energy_rel_fermi_ev"] = _canon_float(energy_rel_fermi_ev)
    return row


def _parse_kwant_done_lines(
    text: str,
    *,
    fermi_ev: float | None,
    source_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in _KWANT_DONE_RE.finditer(text):
        thickness_uc = int(match.group("thickness"))
        energy_abs_ev = float(match.group("energy_abs"))
        energy_rel = None if fermi_ev is None else float(energy_abs_ev) - float(fermi_ev)
        rows.append(
            _row_missing_fields(
                thickness_uc,
                energy_abs_ev,
                energy_rel_fermi_ev=energy_rel,
                transmission=float(match.group("transmission")),
                source_kind="kwant_log",
                source_path=source_path,
            )
        )
    return rows


def scan_partial_kwant_results(kwant_dir: str | Path) -> dict[str, Any]:
    root = Path(kwant_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Kwant benchmark directory not found: {root}")

    payload_path = root / "kwant_payload.json"
    result_path = root / "kwant_reference.json"
    payload = _json_load(payload_path) if payload_path.exists() else {}
    result = _json_load(result_path) if result_path.exists() else {}

    fermi_ev: float | None = None
    if isinstance(payload, dict) and payload.get("fermi_ev") is not None:
        fermi_ev = float(payload["fermi_ev"])
    elif isinstance(result, dict) and result.get("fermi_ev") is not None:
        fermi_ev = float(result["fermi_ev"])

    rows: list[dict[str, Any]] = []
    log_paths = [root / "wtec_job.log", *sorted(root.glob("*.out"))]
    for log_path in log_paths:
        if not log_path.exists():
            continue
        rows.extend(
            _parse_kwant_done_lines(
                log_path.read_text(encoding="utf-8", errors="replace"),
                fermi_ev=fermi_ev,
                source_path=log_path,
            )
        )

    if isinstance(result, dict):
        for item in result.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            energy_abs_ev = float(item.get("energy_abs_ev", 0.0))
            energy_rel = item.get("energy_rel_fermi_ev")
            if energy_rel is None and fermi_ev is not None:
                energy_rel = float(energy_abs_ev) - float(fermi_ev)
            rows.append(
                _row_missing_fields(
                    int(item["thickness_uc"]),
                    energy_abs_ev,
                    energy_rel_fermi_ev=(None if energy_rel is None else float(energy_rel)),
                    transmission=float(item["transmission_e2_over_h"]),
                    source_kind="kwant_result",
                    source_path=result_path,
                )
            )

    rows = _dedupe_rows(rows)
    validation = result.get("validation", {}) if isinstance(result, dict) else {}
    return {
        "root": str(root),
        "fermi_ev": fermi_ev,
        "rows": rows,
        "row_count": len(rows),
        "payload_path": str(payload_path) if payload_path.exists() else None,
        "result_path": str(result_path) if result_path.exists() else None,
        "validation": validation if isinstance(validation, dict) else {},
    }


def _extract_rgf_from_transport_result(path: Path) -> float:
    payload = _json_load(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid RGF transport result: {path}")
    results = payload.get("transport_results", {})
    if isinstance(results, dict):
        thickness_scan = results.get("thickness_scan", {})
        if isinstance(thickness_scan, dict) and thickness_scan:
            first_key = next(iter(thickness_scan))
            block = thickness_scan.get(first_key)
            if isinstance(block, dict):
                g_mean = block.get("G_mean", [])
                if isinstance(g_mean, list) and g_mean:
                    return float(g_mean[0])
    raw = payload.get("transport_results_raw", {})
    if isinstance(raw, dict):
        thickness_g = raw.get("thickness_G", [])
        if isinstance(thickness_g, list):
            if thickness_g and isinstance(thickness_g[0], list):
                if thickness_g[0]:
                    return float(thickness_g[0][0])
            if thickness_g:
                return float(thickness_g[0])
    raise RuntimeError(f"Could not extract RGF transmission from {path}")


def _extract_rgf_from_raw_result(path: Path) -> float:
    raw, _ = load_rgf_raw_result(path)
    thickness_g = raw.get("thickness_G", [])
    if isinstance(thickness_g, list):
        if thickness_g and isinstance(thickness_g[0], list):
            if thickness_g[0]:
                return float(thickness_g[0][0])
        if thickness_g:
            return float(thickness_g[0])
    raise RuntimeError(f"Could not extract RGF transmission from raw result {path}")


def _extract_rgf_from_progress(path: Path) -> float:
    last_g: float | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        if str(rec.get("event", "")) != "native_point_done":
            continue
        g_val = rec.get("G")
        if g_val is None:
            continue
        last_g = float(g_val)
    if last_g is None:
        raise RuntimeError(f"No completed native_point_done event found in {path}")
    return last_g


def _iter_rgf_payloads(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("transport_payload*.json"):
        if "attempts" in path.parts:
            continue
        out.append(path)
    out.sort()
    return out


def _normalize_rgf_payload_row(
    payload_path: Path,
    *,
    fermi_ev: float | None,
) -> dict[str, Any] | None:
    payload = _json_load(payload_path)
    if not isinstance(payload, dict):
        return None
    thicknesses = payload.get("thicknesses", [])
    disorder_strengths = payload.get("disorder_strengths", [])
    mfp_lengths = payload.get("mfp_lengths", [])
    if not isinstance(thicknesses, list) or len(thicknesses) != 1:
        return None
    if not isinstance(disorder_strengths, list) or len(disorder_strengths) != 1:
        return None
    if float(disorder_strengths[0]) != 0.0:
        return None
    if isinstance(mfp_lengths, list) and mfp_lengths:
        return None
    energy_abs_ev = float(payload.get("energy", 0.0))
    energy_rel = None if fermi_ev is None else float(energy_abs_ev) - float(fermi_ev)
    return {
        "thickness_uc": int(thicknesses[0]),
        "energy_abs_ev": float(energy_abs_ev),
        "energy_rel_fermi_ev": (None if energy_rel is None else _canon_float(energy_rel)),
        "transport_dir": payload_path.parent,
    }


def scan_partial_rgf_results(rgf_root: str | Path, *, fermi_ev: float | None = None) -> dict[str, Any]:
    root = Path(rgf_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"RGF root not found: {root}")

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for payload_path in _iter_rgf_payloads(root):
        meta = _normalize_rgf_payload_row(payload_path, fermi_ev=fermi_ev)
        if meta is None:
            skipped.append(str(payload_path))
            continue
        thickness_uc = int(meta["thickness_uc"])
        energy_abs_ev = float(meta["energy_abs_ev"])
        energy_rel = meta.get("energy_rel_fermi_ev")
        transport_dir = Path(meta["transport_dir"])

        result_path = transport_dir / "transport_result.json"
        raw_paths = sorted(
            transport_dir.glob("transport_rgf_raw*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        progress_paths = sorted(
            transport_dir.glob("transport_progress*.jsonl"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )

        transmission: float | None = None
        source_kind: str | None = None
        source_path: Path | None = None

        if result_path.exists():
            transmission = _extract_rgf_from_transport_result(result_path)
            source_kind = "rgf_transport_result"
            source_path = result_path
        elif raw_paths:
            transmission = _extract_rgf_from_raw_result(raw_paths[0])
            source_kind = "rgf_raw_result"
            source_path = raw_paths[0]
        elif progress_paths:
            try:
                transmission = _extract_rgf_from_progress(progress_paths[0])
            except RuntimeError:
                skipped.append(str(payload_path))
                continue
            source_kind = "rgf_progress"
            source_path = progress_paths[0]

        if transmission is None or source_kind is None or source_path is None:
            continue

        rows.append(
            _row_missing_fields(
                thickness_uc,
                energy_abs_ev,
                energy_rel_fermi_ev=(None if energy_rel is None else float(energy_rel)),
                transmission=float(transmission),
                source_kind=source_kind,
                source_path=source_path,
            )
        )

    rows = _dedupe_rows(rows)
    return {
        "root": str(root),
        "rows": rows,
        "row_count": len(rows),
        "skipped_payloads": skipped,
    }


def compare_partial_benchmark_progress(
    *,
    kwant_dir: str | Path,
    rgf_root: str | Path,
    spec: NanowireBenchmarkSpec | None = None,
) -> dict[str, Any]:
    cfg = spec or NanowireBenchmarkSpec()
    kwant = scan_partial_kwant_results(kwant_dir)
    rgf = scan_partial_rgf_results(rgf_root, fermi_ev=kwant.get("fermi_ev"))

    kwant_rows = list(kwant["rows"])
    rgf_rows = list(rgf["rows"])
    kwant_by_key = {_row_key(row): row for row in kwant_rows}
    rgf_by_key = {_row_key(row): row for row in rgf_rows}

    kwant_keys = set(kwant_by_key)
    rgf_keys = set(rgf_by_key)
    overlap = sorted(kwant_keys & rgf_keys)
    missing_in_rgf = [kwant_by_key[key] for key in sorted(kwant_keys - rgf_keys)]
    missing_in_kwant = [rgf_by_key[key] for key in sorted(rgf_keys - kwant_keys)]

    if not kwant_rows:
        status = "missing_kwant_points"
        comparison: dict[str, Any] | None = None
    elif not rgf_rows:
        status = "missing_rgf_points"
        comparison = None
    elif not overlap:
        status = "no_overlap"
        comparison = None
    else:
        ref_subset = [dict(kwant_by_key[key]) for key in overlap]
        rgf_subset = [dict(rgf_by_key[key]) for key in overlap]
        for row in ref_subset:
            row["energy_rel_fermi_ev"] = float(row.get("energy_rel_fermi_ev", 0.0))
        for key, row in zip(overlap, rgf_subset):
            energy_rel = row.get("energy_rel_fermi_ev")
            if energy_rel is None:
                energy_rel = kwant_by_key[key].get("energy_rel_fermi_ev", 0.0)
            row["energy_rel_fermi_ev"] = float(energy_rel)
        cmp_out = compare_reference_and_rgf(
            ref_subset,
            rgf_subset,
            abs_tol=cfg.abs_tol,
            rel_tol=cfg.rel_tol,
            zero_tol=cfg.zero_tol,
        )
        comparison = {
            "status": cmp_out.status,
            "checked_points": cmp_out.checked_points,
            "max_abs_err": cmp_out.max_abs_err,
            "max_rel_err": cmp_out.max_rel_err,
            "failures": cmp_out.failures,
        }
        status = str(cmp_out.status)

    return {
        "status": status,
        "kwant": kwant,
        "rgf": rgf,
        "overlap_points": len(overlap),
        "overlap_keys": [
            {"thickness_uc": int(t), "energy_abs_ev": float(e)}
            for t, e in overlap
        ],
        "missing_in_rgf": missing_in_rgf,
        "missing_in_kwant": missing_in_kwant,
        "comparison": comparison,
    }


def render_partial_comparison_markdown(summary: dict[str, Any]) -> str:
    kwant = summary.get("kwant", {}) if isinstance(summary.get("kwant"), dict) else {}
    rgf = summary.get("rgf", {}) if isinstance(summary.get("rgf"), dict) else {}
    comparison = summary.get("comparison")
    lines = [
        "# Partial Kwant vs RGF Progress Comparison",
        "",
        f"- status: `{summary.get('status', 'unknown')}`",
        f"- kwant_dir: `{kwant.get('root', '')}`",
        f"- rgf_root: `{rgf.get('root', '')}`",
        f"- kwant_points: `{int(kwant.get('row_count', 0) or 0)}`",
        f"- rgf_points: `{int(rgf.get('row_count', 0) or 0)}`",
        f"- overlap_points: `{int(summary.get('overlap_points', 0) or 0)}`",
    ]
    if comparison:
        lines.extend(
            [
                f"- compared_points: `{int(comparison.get('checked_points', 0) or 0)}`",
                f"- max_abs_err: `{float(comparison.get('max_abs_err', 0.0)):.12g}`",
                f"- max_rel_err: `{float(comparison.get('max_rel_err', 0.0)):.12g}`",
            ]
        )
    missing_in_rgf = summary.get("missing_in_rgf", [])
    if missing_in_rgf:
        lines.extend(["", "## Missing In RGF", ""])
        for row in missing_in_rgf[:10]:
            lines.append(
                f"- thickness `{int(row['thickness_uc'])}`, energy_abs `{float(row['energy_abs_ev']):.6f}`"
            )
    missing_in_kwant = summary.get("missing_in_kwant", [])
    if missing_in_kwant:
        lines.extend(["", "## Missing In Kwant", ""])
        for row in missing_in_kwant[:10]:
            lines.append(
                f"- thickness `{int(row['thickness_uc'])}`, energy_abs `{float(row['energy_abs_ev']):.6f}`"
            )
    if comparison and comparison.get("failures"):
        lines.extend(["", "## Comparison Failures", ""])
        for row in comparison["failures"][:10]:
            lines.append(
                "- thickness "
                f"`{int(row['thickness_uc'])}`, energy_rel `{float(row['energy_rel_fermi_ev']):.6f}`, "
                f"reference `{float(row['reference']):.12f}`, rgf `{float(row['rgf']):.12f}`, "
                f"abs_err `{float(row['abs_err']):.12g}`"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Compare completed RGF points against available Kwant reference points "
            "from retrieved or manually fetched benchmark artifacts."
        )
    )
    ap.add_argument("--kwant-dir", required=True, help="Directory containing kwant_payload.json / kwant_reference.json.")
    ap.add_argument("--rgf-root", required=True, help="Root directory containing RGF transport payload/progress/result files.")
    ap.add_argument("--output-json", default=None, help="Optional path to write the JSON summary.")
    ap.add_argument("--output-md", default=None, help="Optional path to write a markdown summary.")
    ns = ap.parse_args(argv)

    try:
        summary = compare_partial_benchmark_progress(
            kwant_dir=ns.kwant_dir,
            rgf_root=ns.rgf_root,
        )
    except FileNotFoundError as exc:
        print(str(exc), flush=True)
        return 2
    if ns.output_json:
        out_json = Path(ns.output_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if ns.output_md:
        out_md = Path(ns.output_md).expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_partial_comparison_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
