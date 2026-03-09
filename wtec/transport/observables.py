"""Transport observables: ρ(d), surface vs bulk decomposition."""

from __future__ import annotations

import numpy as np


def rho_from_conductance(
    G_e2h: np.ndarray,
    thickness_m: np.ndarray,
    cross_section_m2: float,
) -> np.ndarray:
    """Convert conductance to resistivity.

    ρ = d / (G·A)  with G in SI (S)

    Parameters
    ----------
    G_e2h : np.ndarray
        Conductance in units of e²/h.
    thickness_m : np.ndarray
        Film thickness in metres (same shape as G_e2h).
    cross_section_m2 : float
        Transverse cross-section area (m²).

    Returns
    -------
    np.ndarray
        Resistivity in Ω·m.
    """
    e2h = 7.748091729e-5   # S
    G_SI = G_e2h * e2h
    return np.where(G_SI > 0, thickness_m / (G_SI * cross_section_m2), np.inf)


def sheet_resistance(G_e2h: np.ndarray) -> np.ndarray:
    """Sheet resistance R_sq = 1/G in units of h/e²."""
    return np.where(G_e2h > 0, 1.0 / G_e2h, np.inf)


def surface_bulk_decomposition(
    fsys,
    energy: float,
    surface_site_indices: list[int],
    bulk_site_indices: list[int],
) -> dict:
    """Decompose conductance into surface and bulk contributions via LDOS.

    Uses kwant's local density of states to weight contributions.

    Parameters
    ----------
    fsys : kwant.system.FiniteSystem
        Finalized Kwant system.
    surface_site_indices : list[int]
        Site indices of surface layers.
    bulk_site_indices : list[int]
        Site indices of bulk layers.

    Returns
    -------
    dict with 'ldos_surface', 'ldos_bulk', 'fraction_surface'
    """
    try:
        import kwant
    except ImportError:
        raise ImportError("kwant is required")

    ldos = kwant.ldos(fsys, energy)
    ldos_surface = float(np.sum(ldos[surface_site_indices]))
    ldos_bulk    = float(np.sum(ldos[bulk_site_indices]))
    total = ldos_surface + ldos_bulk + 1e-30
    return {
        "ldos_surface": ldos_surface,
        "ldos_bulk": ldos_bulk,
        "fraction_surface": ldos_surface / total,
    }


def fuchs_sondheimer_rho(
    rho_bulk: float,
    mfp_m: float,
    thickness_m: np.ndarray,
    specularity: float = 0.0,
) -> np.ndarray:
    """Fuchs-Sondheimer resistivity for a normal (topologically trivial) metal.

    Used as a reference to contrast with topological film behaviour.

    Parameters
    ----------
    rho_bulk : float
        Bulk resistivity (Ω·m).
    mfp_m : float
        Bulk mean free path (m).
    thickness_m : np.ndarray
        Film thickness (m).
    specularity : float
        Surface specularity parameter p (0 = diffuse, 1 = specular).

    Returns
    -------
    np.ndarray
        Resistivity (Ω·m) for each thickness.
    """
    kappa = thickness_m / mfp_m
    # Approximate Fuchs-Sondheimer (Chambers 1950):
    # ρ_film/ρ_bulk ≈ 1 + (3/8)·(1-p)/κ  for κ >> 1
    # Exact integral for κ ~ 1 would require scipy.integrate
    correction = (3.0 / 8.0) * (1.0 - specularity) / np.maximum(kappa, 1e-10)
    return rho_bulk * (1.0 + correction)


def fit_two_channel_conductance(
    thickness_m: np.ndarray,
    G_e2h: np.ndarray,
    *,
    G_std: np.ndarray | None = None,
) -> dict:
    """Extract topological arc conductance via two-channel linear fit.

    Physics basis
    -------------
    In a topological semimetal film the sheet conductance factorises as:

        σ_sq(d) = G(d) × (L/W)  ≈  σ_arc_2D  +  σ_bulk_3D × d

    where σ_arc_2D is the surface-localised Fermi-arc contribution (thickness-
    independent) and σ_bulk_3D × d is the bulk Drude channel.  A linear fit of
    G(d) vs d therefore extracts σ_arc_2D as the intercept.

    A positive intercept (σ_arc > 0) is the hallmark of a topological film.
    A topologically trivial metal gives σ_arc ≤ 0 within uncertainty.

    Parameters
    ----------
    thickness_m : np.ndarray
        Film thickness in metres (1D, length N ≥ 2).
    G_e2h : np.ndarray
        Mean conductance in units of e²/h (same shape).
    G_std : np.ndarray or None
        Standard deviation of conductance (used as weights if provided).

    Returns
    -------
    dict with keys:
        sigma_arc_2D_e2h : float
            Arc sheet conductance intercept (e²/h). Positive → topological.
        sigma_bulk_3D_e2h_per_m : float
            Bulk conductivity slope (e²/h per metre).
        sigma_arc_2D_err : float
            1σ uncertainty on intercept (e²/h); nan if N < 3.
        sigma_bulk_3D_err : float
            1σ uncertainty on slope (e²/h per metre); nan if N < 3.
        r_squared : float
            Coefficient of determination (goodness of linear fit).
        topological_signal : bool
            True when σ_arc_2D > 2 × σ_arc_2D_err (> 2σ positive intercept).
        n_points : int
            Number of thickness points used.
    """
    d = np.asarray(thickness_m, dtype=float).ravel()
    G = np.asarray(G_e2h, dtype=float).ravel()
    if d.shape != G.shape:
        raise ValueError("thickness_m and G_e2h must have the same length")
    n = len(d)
    if n < 2:
        raise ValueError("At least 2 thickness points are required for fitting")

    # Weighted least-squares: G = a + b·d  (a = σ_arc, b = σ_bulk)
    if G_std is not None:
        w = 1.0 / np.maximum(np.asarray(G_std, dtype=float).ravel() ** 2, 1e-30)
    else:
        w = np.ones(n, dtype=float)

    W = np.sum(w)
    Wd = np.sum(w * d)
    Wd2 = np.sum(w * d ** 2)
    WG = np.sum(w * G)
    WdG = np.sum(w * d * G)

    denom = W * Wd2 - Wd ** 2
    if abs(denom) < 1e-60:
        # Degenerate (all thicknesses identical) — return NaN fit
        return {
            "sigma_arc_2D_e2h": float("nan"),
            "sigma_bulk_3D_e2h_per_m": float("nan"),
            "sigma_arc_2D_err": float("nan"),
            "sigma_bulk_3D_err": float("nan"),
            "r_squared": float("nan"),
            "topological_signal": False,
            "n_points": n,
        }

    a = (WG * Wd2 - WdG * Wd) / denom   # intercept = σ_arc_2D
    b = (W * WdG - Wd * WG) / denom      # slope = σ_bulk_3D

    # Residual variance and parameter uncertainties (only meaningful for n >= 3)
    G_pred = a + b * d
    residuals = G - G_pred
    if n >= 3:
        s2 = np.sum(w * residuals ** 2) / (n - 2)
        var_a = s2 * Wd2 / denom
        var_b = s2 * W / denom
        sigma_a_err = float(np.sqrt(max(var_a, 0.0)))
        sigma_b_err = float(np.sqrt(max(var_b, 0.0)))
    else:
        sigma_a_err = float("nan")
        sigma_b_err = float("nan")

    # R² from unweighted residuals for interpretability
    G_mean = float(np.mean(G))
    ss_tot = float(np.sum((G - G_mean) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    topo = (
        not np.isnan(sigma_a_err)
        and float(a) > 2.0 * sigma_a_err
        and sigma_a_err > 0
    )

    return {
        "sigma_arc_2D_e2h": float(a),
        "sigma_bulk_3D_e2h_per_m": float(b),
        "sigma_arc_2D_err": sigma_a_err,
        "sigma_bulk_3D_err": sigma_b_err,
        "r_squared": float(r2),
        "topological_signal": bool(topo),
        "n_points": n,
    }
