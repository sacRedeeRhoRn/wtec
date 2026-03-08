"""Surface-arc connectivity estimators."""

from __future__ import annotations

from typing import Any

import numpy as np


def _kwant_surface_fraction_metric(
    tb_model,
    *,
    thickness_uc: int,
    energy_ev: float,
    n_layers_x: int,
    n_layers_y: int,
    lead_axis: str,
) -> dict[str, Any]:
    try:
        import kwant
    except ImportError as exc:
        return {"status": "failed", "reason": f"kwant_missing:{exc}"}

    sys = tb_model.to_kwant_builder(
        n_layers_z=int(thickness_uc),
        n_layers_x=int(n_layers_x),
        n_layers_y=int(n_layers_y),
        lead_axis=lead_axis,
        substrate_onsite_eV=0.0,
    )
    fsys = sys.finalized()
    ldos = np.asarray(kwant.ldos(fsys, float(energy_ev)), dtype=float)
    if ldos.size == 0:
        return {"status": "failed", "reason": "empty_ldos"}

    z_tags = np.array([int(site.tag[2]) for site in fsys.sites], dtype=int)
    z_min = int(np.min(z_tags))
    z_max = int(np.max(z_tags))
    surface_idx = np.where((z_tags == z_min) | (z_tags == z_max))[0]
    bulk_idx = np.where((z_tags != z_min) & (z_tags != z_max))[0]

    surf = float(np.sum(ldos[surface_idx])) if surface_idx.size else 0.0
    bulk = float(np.sum(ldos[bulk_idx])) if bulk_idx.size else 0.0
    total = surf + bulk + 1e-30
    frac = surf / total
    return {
        "status": "ok",
        "metric": float(np.clip(frac, 0.0, 1.0)),
        "surface_fraction": float(frac),
        "engine": "kwant_ldos_surface_proxy",
    }


def _try_wannierberri_surface_metric(
    hr_dat_path: str | None,
    *,
    energy_ev: float,
) -> dict[str, Any]:
    """Best-effort WannierBerri surface metric.

    This keeps the backend hook explicit; if unavailable, caller should fallback.
    """
    if not hr_dat_path:
        return {"status": "failed", "reason": "missing_hr_dat_path"}
    try:
        import wannierberri as wb  # noqa: F401
    except Exception as exc:
        return {"status": "failed", "reason": f"wannierberri_missing:{exc}"}
    has_surface_api = False
    try:
        calc = getattr(wb, "calculators", None)
        if calc is not None:
            for scope_name in ("tabulate", "static", "dynamic"):
                scope = getattr(calc, scope_name, None)
                if scope is None:
                    continue
                for name in dir(scope):
                    if "surface" in name.lower():
                        has_surface_api = True
                        break
                if has_surface_api:
                    break
    except Exception:
        has_surface_api = False

    if not has_surface_api:
        return {"status": "failed", "reason": "wannierberri_surface_api_unavailable"}

    # The runtime environment currently lacks a stable cross-version surface-spectrum
    # workflow in the installed WannierBerri builds.
    return {"status": "failed", "reason": "wannierberri_surface_not_implemented"}


def _load_siesta_slab_ldos_metric(
    *,
    ldos_json_path: str | None,
) -> dict[str, Any]:
    if not ldos_json_path:
        return {"status": "failed", "reason": "missing_siesta_slab_ldos_json"}
    try:
        from pathlib import Path
        import json

        p = Path(str(ldos_json_path)).expanduser()
        if not p.exists():
            return {"status": "failed", "reason": f"siesta_slab_ldos_json_not_found:{p}"}
        payload = json.loads(p.read_text())
        if not isinstance(payload, dict):
            return {"status": "failed", "reason": "invalid_siesta_slab_ldos_payload"}
        if "metric" in payload:
            metric = float(payload.get("metric", 0.0))
        elif "surface_fraction" in payload:
            metric = float(payload.get("surface_fraction", 0.0))
        else:
            return {"status": "failed", "reason": "siesta_slab_ldos_metric_missing"}
        source_engine = str(payload.get("source_engine", "siesta_slab_ldos")).strip()
        source_kind = str(payload.get("source_kind", "provided")).strip()
    except Exception as exc:
        return {"status": "failed", "reason": f"siesta_slab_ldos_read_failed:{type(exc).__name__}:{exc}"}

    metric = float(np.clip(metric, 0.0, 1.0))
    return {
        "status": "ok",
        "metric": metric,
        "surface_fraction": metric,
        "engine": "siesta_slab_ldos",
        "source_engine": source_engine or "siesta_slab_ldos",
        "source_kind": source_kind or "provided",
        "source": str(ldos_json_path),
    }


def compute_arc_connectivity(
    tb_model,
    *,
    thickness_uc: int,
    energy_ev: float = 0.0,
    n_layers_x: int = 4,
    n_layers_y: int = 4,
    lead_axis: str = "x",
    prefer_engine: str = "wannierberri",
    hr_dat_path: str | None = None,
    siesta_slab_ldos_json: str | None = None,
    allow_proxy_fallback: bool = False,
) -> dict[str, Any]:
    """Compute an arc-connectivity metric in [0,1].

    Engine order:
    1) wannierberri surface hook (when available)
    2) kwant surface-LDOS proxy fallback
    """
    pref = str(prefer_engine).lower().strip()
    if pref in {"wannierberri_strict", "wb_strict"}:
        wb = _try_wannierberri_surface_metric(hr_dat_path, energy_ev=energy_ev)
        wb.setdefault("engine", "wannierberri_surface")
        return wb

    if pref == "siesta_slab_ldos":
        si = _load_siesta_slab_ldos_metric(ldos_json_path=siesta_slab_ldos_json)
        if si.get("status") == "ok":
            return si
        if not allow_proxy_fallback:
            return si
        fb = _kwant_surface_fraction_metric(
            tb_model,
            thickness_uc=thickness_uc,
            energy_ev=energy_ev,
            n_layers_x=n_layers_x,
            n_layers_y=n_layers_y,
            lead_axis=lead_axis,
        )
        fb["fallback_from"] = si.get("reason")
        return fb

    if pref == "wannierberri":
        wb = _try_wannierberri_surface_metric(hr_dat_path, energy_ev=energy_ev)
        if wb.get("status") == "ok":
            wb["metric"] = float(np.clip(float(wb.get("metric", 0.0)), 0.0, 1.0))
            return wb
        fb = _kwant_surface_fraction_metric(
            tb_model,
            thickness_uc=thickness_uc,
            energy_ev=energy_ev,
            n_layers_x=n_layers_x,
            n_layers_y=n_layers_y,
            lead_axis=lead_axis,
        )
        fb["fallback_from"] = wb.get("reason")
        return fb

    if pref in {"kwant", "kwant_proxy"}:
        return _kwant_surface_fraction_metric(
            tb_model,
            thickness_uc=thickness_uc,
            energy_ev=energy_ev,
            n_layers_x=n_layers_x,
            n_layers_y=n_layers_y,
            lead_axis=lead_axis,
        )

    return _kwant_surface_fraction_metric(
        tb_model,
        thickness_uc=thickness_uc,
        energy_ev=energy_ev,
        n_layers_x=n_layers_x,
        n_layers_y=n_layers_y,
        lead_axis=lead_axis,
    )
