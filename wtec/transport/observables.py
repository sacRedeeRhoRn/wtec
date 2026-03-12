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

    Two implementations are provided:
    - Asymptotic (Chambers 1950): ρ_FS/ρ_bulk ≈ 1 + (3/8)(1−p)/κ   (κ ≫ 1)
    - Exact integral (Fuchs 1938, Sondheimer 1952): full numerical integration

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


def fuchs_sondheimer_rho_exact(
    rho_bulk: float,
    mfp_m: float,
    thickness_m: np.ndarray,
    specularity: float = 0.0,
) -> np.ndarray:
    """Exact Fuchs-Sondheimer resistivity via numerical integration.

    Uses the full Fuchs (1938) / Sondheimer (1952) integral:

        ρ_FS/ρ_bulk = [1 − (3/2κ)(1−p) ∫₁^∞ (1/u³ − 1/u⁵)
                        × (1 − exp(−κu)) / (1 − p·exp(−κu)) du]⁻¹

    where κ = d/ℓ (Knudsen number).

    Valid for all κ (thin film to bulk limit).  Requires scipy.

    Parameters
    ----------
    rho_bulk : float
    mfp_m : float
    thickness_m : np.ndarray
    specularity : float  p ∈ [0, 1]

    Returns
    -------
    np.ndarray  Resistivity (Ω·m).
    """
    try:
        from scipy.integrate import quad
    except ImportError:
        # Fallback to asymptotic if scipy unavailable
        return fuchs_sondheimer_rho(rho_bulk, mfp_m, thickness_m, specularity)

    kappa_arr = np.atleast_1d(np.asarray(thickness_m, dtype=float)) / float(mfp_m)
    p = float(np.clip(specularity, 0.0, 1.0))
    results = np.zeros_like(kappa_arr)

    for i, kappa in enumerate(kappa_arr):
        k = float(kappa)
        if k <= 0:
            results[i] = np.inf
            continue

        def integrand(u: float) -> float:
            exp_ku = np.exp(-k * u)
            denom = 1.0 - p * exp_ku
            if abs(denom) < 1e-30:
                return 0.0
            return (1.0 / u ** 3 - 1.0 / u ** 5) * (1.0 - exp_ku) / denom

        I, _ = quad(integrand, 1.0, np.inf, limit=500, epsabs=1e-10, epsrel=1e-8)
        ratio = 1.0 - 1.5 * (1.0 - p) * I / k
        results[i] = float(rho_bulk) / ratio if ratio > 1e-15 else np.inf

    return results


def conductance_finite_temperature(
    fsys,
    mu: float,
    T_kelvin: float,
    *,
    n_E: int = 80,
    E_window_kT: float = 6.0,
) -> float:
    """Landauer conductance at finite temperature via thermal smearing.

    G(T) = (e²/h) ∫ T(E) · (−∂f/∂E) dE

    where  −∂f/∂E = sech²((E−μ)/2kT) / (4kT)  is the thermal smearing kernel.

    Parameters
    ----------
    fsys : kwant.system.FiniteSystem  (finalized)
    mu : float  Chemical potential / Fermi energy (eV).
    T_kelvin : float  Temperature (K).  Use 0 to return zero-T conductance.
    n_E : int  Number of energy points for integration.
    E_window_kT : float  Integration window half-width in units of kT.

    Returns
    -------
    float  Conductance in e²/h.
    """
    try:
        import kwant
    except ImportError:
        raise ImportError("kwant is required")

    if float(T_kelvin) <= 0.0:
        smat = kwant.smatrix(fsys, float(mu))
        return float(smat.transmission(0, 1))

    kT = 8.617e-5 * float(T_kelvin)
    E_min = float(mu) - float(E_window_kT) * kT
    E_max = float(mu) + float(E_window_kT) * kT
    E_vals = np.linspace(E_min, E_max, max(10, int(n_E)))

    T_vals = np.array([
        kwant.smatrix(fsys, E).transmission(0, 1)
        for E in E_vals
    ], dtype=float)

    # Thermal smearing kernel: −∂f/∂E = sech²((E−μ)/2kT) / (4kT)
    x = (E_vals - float(mu)) / (2.0 * kT)
    kernel = 1.0 / (np.cosh(x) ** 2 * 4.0 * kT)
    # Normalise kernel to 1 (numerical)
    norm = np.trapz(kernel, E_vals)
    if abs(norm) > 1e-20:
        kernel /= norm

    G = float(np.trapz(T_vals * kernel, E_vals))
    return max(0.0, G)


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
