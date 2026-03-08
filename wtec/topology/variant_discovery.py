"""Discover slab defect variants and compute severity from metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wtec.structure.metadata_schema import validate_slab_metadata


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _severity_from_metadata(meta: dict[str, Any]) -> float:
    interfaces = meta.get("interfaces", [])
    if not isinstance(interfaces, list):
        interfaces = []

    total_events = 0
    interface_atoms_total = 0
    for iface in interfaces:
        if not isinstance(iface, dict):
            continue
        interface_atoms_total += int(iface.get("atoms_in_window", 0) or 0)
        total_events += int(iface.get("atoms_removed", 0) or 0)
        subs = iface.get("substitutions", [])
        if isinstance(subs, list):
            for s in subs:
                if isinstance(s, dict):
                    total_events += int(s.get("applied", 0) or 0)

    # Severity saturates near 5% defect concentration within interface windows.
    denom = 0.05 * max(1, interface_atoms_total)
    return float(max(0.0, min(1.0, float(total_events) / float(denom))))


def _resolve_cif_from_meta(meta_path: Path, meta: dict[str, Any]) -> Path | None:
    export = meta.get("export", {})
    if isinstance(export, dict):
        cif_raw = export.get("cif_path")
        if isinstance(cif_raw, str) and cif_raw.strip():
            p = Path(cif_raw).expanduser()
            if not p.is_absolute():
                p = (meta_path.parent / p).resolve()
            if p.exists():
                return p
    # fallback by filename convention
    stem = meta_path.name.replace(".meta.json", "")
    cand = meta_path.with_name(f"{stem}.cif")
    if cand.exists():
        return cand
    return None


def _project_name(meta: dict[str, Any], meta_path: Path) -> str:
    proj = meta.get("project", {})
    if isinstance(proj, dict):
        name = proj.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return meta_path.stem.replace(".meta", "")


def discover_variants(
    *,
    structure_file: str | Path | None,
    run_dir: str | Path,
    glob_pattern: str = "slab_outputs/**/*.generated.meta.json",
) -> list[dict[str, Any]]:
    """Discover defect variants from slab metadata and optional structure file."""
    root = Path(run_dir).expanduser().resolve()
    candidates: list[Path] = []

    # Search relative to run_dir and current working directory.
    for base in [root, Path.cwd()]:
        try:
            candidates.extend(base.glob(glob_pattern))
        except Exception:
            pass
    # Deduplicate
    seen_paths: set[Path] = set()
    meta_paths: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp in seen_paths or not rp.exists():
            continue
        seen_paths.add(rp)
        meta_paths.append(rp)

    # Also include structure-adjacent metadata if present.
    if structure_file:
        sf = Path(structure_file).expanduser().resolve()
        if sf.exists():
            adj = sf.with_suffix(".meta.json")
            if adj.exists() and adj.resolve() not in seen_paths:
                meta_paths.append(adj.resolve())
                seen_paths.add(adj.resolve())

    variants: list[dict[str, Any]] = []
    for mp in sorted(meta_paths):
        meta = _load_json(mp)
        if not meta:
            continue
        valid, schema_errors = validate_slab_metadata(meta)
        if not valid:
            continue
        severity = _severity_from_metadata(meta)
        variant_id = _project_name(meta, mp)
        cif = _resolve_cif_from_meta(mp, meta)
        variants.append(
            {
                "variant_id": variant_id,
                "metadata_path": str(mp),
                "cif_path": str(cif) if cif else None,
                "defect_severity": severity,
                "is_pristine": bool(severity == 0.0),
                "metadata_schema_version": meta.get("wtec_slab_metadata_version"),
                "metadata_schema_errors": schema_errors,
            }
        )

    if not variants:
        sf = Path(structure_file).expanduser().resolve() if structure_file else None
        variants.append(
            {
                "variant_id": "current_structure",
                "metadata_path": None,
                "cif_path": str(sf) if sf and sf.exists() else None,
                "defect_severity": 0.0,
                "is_pristine": True,
            }
        )

    # Ensure at least one pristine baseline.
    if not any(v.get("is_pristine") for v in variants):
        variants.insert(
            0,
            {
                "variant_id": "pristine_baseline",
                "metadata_path": None,
                "cif_path": None,
                "defect_severity": 0.0,
                "is_pristine": True,
            },
        )

    variants.sort(key=lambda v: (float(v.get("defect_severity", 0.0)), str(v.get("variant_id", ""))))
    return variants
