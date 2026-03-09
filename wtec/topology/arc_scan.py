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

    axis = str(lead_axis).lower().strip()
    counts = {"x": int(n_layers_x), "y": int(n_layers_y), "z": int(thickness_uc)}
    req_fn = getattr(tb_model, "required_lead_axis_cells", None)
    if callable(req_fn):
        required = int(
            req_fn(
                lead_axis=axis,
                n_layers_x=counts["x"],
                n_layers_y=counts["y"],
                n_layers_z=counts["z"],
            )
        )
        counts[axis] = max(int(counts[axis]), required)
    else:
        counts[axis] = max(int(counts[axis]), 2)

    sys = tb_model.to_kwant_builder(
        n_layers_z=int(thickness_uc),
        n_layers_x=int(counts["x"]),
        n_layers_y=int(counts["y"]),
        lead_axis=axis,
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


def _connected_component_stats(mask: np.ndarray) -> dict[str, float]:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2:
        return {
            "active_count": 0.0,
            "largest_count": 0.0,
            "largest_fraction": 0.0,
            "span_x": 0.0,
            "span_y": 0.0,
        }
    nx, ny = mask_bool.shape
    active_count = int(np.count_nonzero(mask_bool))
    if active_count <= 0:
        return {
            "active_count": 0.0,
            "largest_count": 0.0,
            "largest_fraction": 0.0,
            "span_x": 0.0,
            "span_y": 0.0,
        }

    visited = np.zeros_like(mask_bool, dtype=bool)
    largest_count = 0
    best_span_x = 0.0
    best_span_y = 0.0
    for i in range(nx):
        for j in range(ny):
            if not mask_bool[i, j] or visited[i, j]:
                continue
            stack = [(i, j)]
            visited[i, j] = True
            count = 0
            min_i = max_i = i
            min_j = max_j = j
            while stack:
                x, y = stack.pop()
                count += 1
                min_i = min(min_i, x)
                max_i = max(max_i, x)
                min_j = min(min_j, y)
                max_j = max(max_j, y)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    xn = x + dx
                    yn = y + dy
                    if not (0 <= xn < nx and 0 <= yn < ny):
                        continue
                    if visited[xn, yn] or not mask_bool[xn, yn]:
                        continue
                    visited[xn, yn] = True
                    stack.append((xn, yn))
            if count > largest_count:
                largest_count = count
                best_span_x = float(max_i - min_i + 1) / float(max(1, nx))
                best_span_y = float(max_j - min_j + 1) / float(max(1, ny))
    return {
        "active_count": float(active_count),
        "largest_count": float(largest_count),
        "largest_fraction": float(largest_count) / float(max(1, active_count)),
        "span_x": float(best_span_x),
        "span_y": float(best_span_y),
    }


def _tb_kresolved_surface_metric(
    tb_model,
    *,
    thickness_uc: int,
    energy_ev: float,
    n_kx: int = 8,
    n_ky: int = 8,
    broadening_ev: float = 0.06,
) -> dict[str, Any]:
    n_layers_z = max(2, int(thickness_uc))
    n_kx = max(4, int(n_kx))
    n_ky = max(4, int(n_ky))
    eta = max(1.0e-4, float(broadening_ev))

    iter_hoppings = getattr(tb_model, "_iter_hoppings", None)
    if not callable(iter_hoppings):
        return {"status": "failed", "reason": "tb_model_missing_hopping_iterator"}
    try:
        n_orb = int(getattr(tb_model, "num_orbitals"))
    except Exception:
        try:
            n_orb = int(np.asarray(tb_model.hamiltonian_at_k(np.zeros(3, dtype=float))).shape[0])
        except Exception as exc:
            return {"status": "failed", "reason": f"tb_model_orbital_count_failed:{type(exc).__name__}:{exc}"}
    if n_orb <= 0:
        return {"status": "failed", "reason": "tb_model_zero_orbitals"}

    hoppings: list[tuple[int, int, int, np.ndarray]] = []
    for rx, ry, rz, mat in iter_hoppings():
        m = np.asarray(mat, dtype=complex)
        if m.shape != (n_orb, n_orb):
            continue
        hoppings.append((int(rx), int(ry), int(rz), m))
    if not hoppings:
        return {"status": "failed", "reason": "tb_model_empty_hoppings"}

    ndof = int(n_layers_z * n_orb)
    surf_idx = np.concatenate(
        [
            np.arange(0, n_orb, dtype=int),
            np.arange((n_layers_z - 1) * n_orb, n_layers_z * n_orb, dtype=int),
        ]
    )

    spectral_map = np.zeros((n_kx, n_ky), dtype=float)
    two_pi = 2.0 * np.pi
    eye_tol = 1.0e-12
    for ix in range(n_kx):
        kx = float(ix) / float(n_kx)
        for iy in range(n_ky):
            ky = float(iy) / float(n_ky)
            H = np.zeros((ndof, ndof), dtype=complex)
            for rx, ry, rz, mat in hoppings:
                phase = np.exp(1j * two_pi * (kx * float(rx) + ky * float(ry)))
                for iz in range(n_layers_z):
                    jz = iz + int(rz)
                    if jz < 0 or jz >= n_layers_z:
                        continue
                    i0 = int(iz * n_orb)
                    j0 = int(jz * n_orb)
                    H[i0 : i0 + n_orb, j0 : j0 + n_orb] += phase * mat
            # Numerical cleanup: enforce Hermiticity before eigendecomposition.
            H = 0.5 * (H + H.conj().T)
            try:
                evals, evecs = np.linalg.eigh(H)
            except Exception as exc:
                return {
                    "status": "failed",
                    "reason": f"tb_kresolved_eigh_failed:{type(exc).__name__}:{exc}",
                }
            weights = eta / (((float(energy_ev) - evals) ** 2) + eta * eta) / np.pi
            surf_w = np.sum(np.abs(evecs[surf_idx, :]) ** 2, axis=0)
            surf_w = np.where(np.abs(surf_w) < eye_tol, 0.0, surf_w)
            spectral_map[ix, iy] = float(np.sum(weights * surf_w))

    if not np.isfinite(spectral_map).all():
        return {"status": "failed", "reason": "tb_kresolved_non_finite_spectral_map"}
    peak = float(np.max(spectral_map))
    if peak <= 0.0:
        return {"status": "failed", "reason": "tb_kresolved_zero_spectral_weight"}

    norm_map = spectral_map / peak
    q = float(np.quantile(norm_map, 0.85))
    threshold = max(0.20, min(0.95, q))
    mask = norm_map >= threshold
    stats = _connected_component_stats(mask)
    largest_fraction = float(stats["largest_fraction"])
    span = max(float(stats["span_x"]), float(stats["span_y"]))
    metric = float(np.clip(0.5 * largest_fraction + 0.5 * span, 0.0, 1.0))
    return {
        "status": "ok",
        "metric": metric,
        "surface_fraction": metric,
        "engine": "tb_kresolved_surface_spectral",
        "kmesh_xy": [int(n_kx), int(n_ky)],
        "broadening_ev": float(eta),
        "threshold": float(threshold),
        "largest_component_fraction": largest_fraction,
        "largest_component_span": float(span),
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
    n_layers_y: int = 16,
    lead_axis: str = "x",
    prefer_engine: str = "wannierberri",
    hr_dat_path: str | None = None,
    siesta_slab_ldos_json: str | None = None,
    allow_proxy_fallback: bool = False,
    kmesh_xy: tuple[int, int] = (8, 8),
    broadening_ev: float = 0.06,
) -> dict[str, Any]:
    """Compute an arc-connectivity metric in [0,1].

    Engine order:
    1) siesta_slab_ldos (SIESTA slab surface LDOS — real physics)
    2) wannierberri surface hook (when available)
    3) tb_kresolved surface spectral map on slab TB Hamiltonian
    4) kwant surface-LDOS proxy fallback (if allow_proxy_fallback=True)

    Parameters
    ----------
    n_layers_x : int
        Must be >= 2 for Kwant lead attachment. Default 4.
    n_layers_y : int
        Transverse slab width. Default 16 (required for TaP/NbP arc resolution;
        arc k-width ≈ 0.15 Å⁻¹ demands W >> 42 Å, i.e. n_y >= 13 for a=3.30 Å).
    thickness_uc : int
        Film thickness in unit cells. Values < 10 may capture hybridized arcs.
    """
    if n_layers_x < 2:
        raise ValueError(
            f"n_layers_x={n_layers_x} is invalid for arc scan. "
            "Kwant requires n_layers_x >= 2 for lead attachment. "
            "Set topology.n_layers_x >= 4 in your config."
        )
    if int(thickness_uc) < 6:
        import warnings
        warnings.warn(
            f"thickness_uc={thickness_uc} may be too thin for well-defined arc states. "
            "Arc penetration depth ≈ 5 unit cells for TaP; recommend thickness_uc >= 10.",
            stacklevel=2,
        )
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
        fb = _tb_kresolved_surface_metric(
            tb_model,
            thickness_uc=thickness_uc,
            energy_ev=energy_ev,
            n_kx=int(kmesh_xy[0]),
            n_ky=int(kmesh_xy[1]),
            broadening_ev=float(broadening_ev),
        )
        if fb.get("status") != "ok":
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

    if pref in {"tb_kresolved", "tb_surface_kresolved", "kresolved"}:
        out = _tb_kresolved_surface_metric(
            tb_model,
            thickness_uc=thickness_uc,
            energy_ev=energy_ev,
            n_kx=int(kmesh_xy[0]),
            n_ky=int(kmesh_xy[1]),
            broadening_ev=float(broadening_ev),
        )
        if out.get("status") == "ok" or not allow_proxy_fallback:
            return out
        fb = _kwant_surface_fraction_metric(
            tb_model,
            thickness_uc=thickness_uc,
            energy_ev=energy_ev,
            n_layers_x=n_layers_x,
            n_layers_y=n_layers_y,
            lead_axis=lead_axis,
        )
        fb["fallback_from"] = out.get("reason")
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
