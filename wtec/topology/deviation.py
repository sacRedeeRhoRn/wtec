"""Composite topology-deviation scoring."""

from __future__ import annotations

from typing import Any


def _clip01(x: float | None) -> float | None:
    if x is None:
        return None
    val = float(x)
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def compute_s_topo(
    s_arc: float | None,
    delta_node: float | None,
) -> tuple[float | None, str, list[str]]:
    """Compute topology score with explicit union fallback.

    Rules (canonical):
    - both available: 1 - (1-S_arc)*(1-delta_node), confidence="full"
    - arc only: S_topo = S_arc, confidence="partial-node"
    - node only: S_topo = delta_node, confidence="partial-arc"
    - both missing: None, confidence="failed"
    """
    a = _clip01(s_arc)
    d = _clip01(delta_node)
    missing: list[str] = []
    if a is None:
        missing.append("S_arc")
    if d is None:
        missing.append("delta_node")

    if a is not None and d is not None:
        return 1.0 - (1.0 - a) * (1.0 - d), "full", missing
    if a is not None:
        return a, "partial-node", missing
    if d is not None:
        return d, "partial-arc", missing
    return None, "failed", missing


def compute_s_total(
    s_topo: float | None,
    s_transport: float | None,
    *,
    w_topo: float = 0.70,
    allow_missing_topology: bool = False,
) -> tuple[float | None, str, list[str]]:
    """Combine topology and transport channels.

    Rules:
    - both present: S_total = w_topo*S_topo + (1-w_topo)*S_transport
    - topology only: S_total = S_topo, confidence="partial-transport"
    - topology missing + transport present:
      - if allow_missing_topology: S_total = S_transport, confidence="partial-topology"
      - else: None, confidence="failed"
    """
    t = _clip01(s_topo)
    tr = _clip01(s_transport)
    missing: list[str] = []
    if tr is None:
        missing.append("S_transport")

    w = float(w_topo)
    if w < 0.0 or w > 1.0:
        raise ValueError(f"w_topo must be within [0,1], got {w_topo!r}")

    if t is None:
        if tr is not None and allow_missing_topology:
            return tr, "partial-topology", (missing + ["S_topo"])
        return None, "failed", (missing + ["S_topo"])
    if tr is None:
        return t, "partial-transport", missing
    return w * t + (1.0 - w) * tr, "full", missing


def build_result(
    *,
    thickness_uc: int,
    variant_id: str,
    defect_severity: float,
    s_arc: float | None,
    delta_node: float | None,
    s_transport: float | None,
    w_topo: float,
    allow_missing_topology: bool = False,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical result row."""
    s_topo, conf_topo, missing_topo = compute_s_topo(s_arc, delta_node)
    s_total, conf_total, missing_total = compute_s_total(
        s_topo,
        s_transport,
        w_topo=w_topo,
        allow_missing_topology=allow_missing_topology,
    )

    missing = sorted(set(missing_topo + missing_total))
    if s_total is None:
        status = "failed"
    elif missing:
        status = "partial"
    else:
        status = "ok"

    out: dict[str, Any] = {
        "thickness_uc": int(thickness_uc),
        "variant_id": str(variant_id),
        "defect_severity": float(defect_severity),
        "S_arc": _clip01(s_arc),
        "delta_node": _clip01(delta_node),
        "S_transport": _clip01(s_transport),
        "S_topo": _clip01(s_topo),
        "S_total": _clip01(s_total),
        "status": status,
        "confidence_topo": conf_topo,
        "confidence_total": conf_total,
        "confidence": conf_total if conf_total != "full" else conf_topo,
        "missing": missing,
    }
    if extras:
        out.update(extras)
    return out
