"""Adaptive k-space helpers for node-guided surface scans."""

from __future__ import annotations

from typing import Any

import numpy as np


def _clip01_pair(values: list[float] | tuple[float, float]) -> tuple[float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return 0.06, 0.06
    out = []
    for v in values:
        fv = float(v)
        fv = max(1.0e-6, min(0.5, fv))
        out.append(fv)
    return float(out[0]), float(out[1])


def project_kfrac_to_surface(k_frac: list[float] | tuple[float, float, float], surface_axis: str) -> tuple[float, float]:
    """Project 3D fractional k to 2D surface-BZ coordinates."""
    if len(k_frac) != 3:
        raise ValueError("k_frac must have length 3.")
    k = np.mod(np.array(k_frac, dtype=float), 1.0)
    ax = str(surface_axis).strip().lower()
    if ax == "z":
        return float(k[0]), float(k[1])
    if ax == "x":
        return float(k[1]), float(k[2])
    if ax == "y":
        return float(k[0]), float(k[2])
    raise ValueError(f"surface_axis must be one of ['x','y','z'], got {surface_axis!r}")


def _periodic_dist2(u0: float, v0: float, u1: float, v1: float) -> float:
    du = abs(float(u0) - float(u1))
    dv = abs(float(v0) - float(v1))
    du = min(du, 1.0 - du)
    dv = min(dv, 1.0 - dv)
    return float(np.hypot(du, dv))


def select_node_projected_hotspots(
    node_scan: dict[str, Any] | None,
    *,
    surface_axis: str = "z",
    energy_window_ev: float = 0.12,
    hotspot_gap_max_ev: float = 0.03,
    max_hotspots: int = 8,
    dedup_radius_frac: float = 0.03,
) -> list[dict[str, Any]]:
    """Select surface-BZ hotspots from node-scan output."""
    if not isinstance(node_scan, dict):
        return []
    nodes = node_scan.get("nodes")
    if not isinstance(nodes, list):
        return []
    e_win = max(0.0, float(energy_window_ev))
    gap_max = max(0.0, float(hotspot_gap_max_ev))
    kmax = max(1, int(max_hotspots))
    dedup = max(1.0e-6, float(dedup_radius_frac))

    candidates: list[dict[str, Any]] = []
    for idx, n in enumerate(nodes):
        if not isinstance(n, dict):
            continue
        kf = n.get("k_frac")
        if not isinstance(kf, (list, tuple)) or len(kf) != 3:
            continue
        try:
            e_rel = float(n.get("energy_rel_fermi_ev", 0.0))
            gap_ev = float(n.get("gap_ev", 1.0e9))
            uv = project_kfrac_to_surface([float(kf[0]), float(kf[1]), float(kf[2])], surface_axis)
        except Exception:
            continue
        if abs(e_rel) > e_win:
            continue
        if gap_ev > gap_max:
            continue
        candidates.append(
            {
                "center_uv": [float(uv[0]), float(uv[1])],
                "energy_rel_fermi_ev": float(e_rel),
                "gap_ev": float(gap_ev),
                "chirality": n.get("chirality"),
                "source_node_index": int(idx),
            }
        )

    # Physically prioritize smallest-gap, near-EF nodes.
    candidates.sort(key=lambda x: (float(x["gap_ev"]), abs(float(x["energy_rel_fermi_ev"]))))

    selected: list[dict[str, Any]] = []
    for c in candidates:
        cu, cv = float(c["center_uv"][0]), float(c["center_uv"][1])
        duplicate = False
        for s in selected:
            su, sv = float(s["center_uv"][0]), float(s["center_uv"][1])
            if _periodic_dist2(cu, cv, su, sv) < dedup:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(c)
        if len(selected) >= kmax:
            break
    return selected


def is_node_signal_weak(
    node_scan: dict[str, Any] | None,
    hotspots: list[dict[str, Any]],
    *,
    min_hotspots: int = 4,
    median_gap_threshold_ev: float = 0.03,
) -> bool:
    """Heuristic for deciding if node-guided refinement is reliable."""
    mhot = max(1, int(min_hotspots))
    if not isinstance(node_scan, dict):
        return True
    status = str(node_scan.get("status", "")).strip().lower()
    if status not in {"ok", "partial"}:
        return True
    if len(hotspots) < mhot:
        return True
    gaps = [float(h.get("gap_ev", np.inf)) for h in hotspots]
    finite = [g for g in gaps if np.isfinite(g)]
    if not finite:
        return True
    return float(np.median(finite)) > float(median_gap_threshold_ev)


def adaptive_k_defaults() -> dict[str, Any]:
    return {
        "enabled": True,
        "surface_axis": "z",
        "global_kmesh_xy": [16, 16],
        "local_kmesh_xy": [48, 48],
        "fallback_global_refine_kmesh_xy": [40, 40],
        "window_radius_frac_xy": [0.06, 0.06],
        "energy_window_ev": 0.12,
        "hotspot_gap_max_ev": 0.03,
        "max_hotspots": 8,
        "min_hotspots": 4,
        "dedup_radius_frac": 0.03,
        "require_inplane_transport": True,
    }


def normalize_adaptive_k_cfg(raw: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(adaptive_k_defaults())
    if isinstance(raw, dict):
        out.update(raw)

    def _pair(name: str, default_pair: tuple[int, int]) -> list[int]:
        v = out.get(name, list(default_pair))
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            return [int(default_pair[0]), int(default_pair[1])]
        return [max(2, int(v[0])), max(2, int(v[1]))]

    out["enabled"] = bool(out.get("enabled", True))
    out["surface_axis"] = str(out.get("surface_axis", "z")).strip().lower() or "z"
    out["global_kmesh_xy"] = _pair("global_kmesh_xy", (16, 16))
    out["local_kmesh_xy"] = _pair("local_kmesh_xy", (48, 48))
    out["fallback_global_refine_kmesh_xy"] = _pair("fallback_global_refine_kmesh_xy", (40, 40))
    rx, ry = _clip01_pair(out.get("window_radius_frac_xy", [0.06, 0.06]))
    out["window_radius_frac_xy"] = [rx, ry]
    out["energy_window_ev"] = max(1.0e-6, float(out.get("energy_window_ev", 0.12)))
    out["hotspot_gap_max_ev"] = max(1.0e-6, float(out.get("hotspot_gap_max_ev", 0.03)))
    out["max_hotspots"] = max(1, int(out.get("max_hotspots", 8)))
    out["min_hotspots"] = max(1, int(out.get("min_hotspots", 4)))
    out["dedup_radius_frac"] = max(1.0e-6, min(0.5, float(out.get("dedup_radius_frac", 0.03))))
    out["require_inplane_transport"] = bool(out.get("require_inplane_transport", True))
    if out["min_hotspots"] > out["max_hotspots"]:
        out["min_hotspots"] = out["max_hotspots"]
    return out
