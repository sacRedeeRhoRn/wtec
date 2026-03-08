"""Mean free path extraction from G(L) scaling."""

from __future__ import annotations

import numpy as np


def extract_mfp_from_scaling(
    lengths_m: np.ndarray,
    G_mean: np.ndarray,
    cross_section_m2: float,
) -> dict:
    """Extract MFP from the diffusive G(L) = σ·A/L scaling.

    Uses an affine inverse-conductance model:
        1/G(L) = a + b·L
    where
      - a ≈ 1/G_ballistic (contact-limited intercept)
      - b = 1/(σ·A)

    This yields a direct transport-length estimate:
        ℓ_eff = a / b
    from the Landauer crossover model G(L)=G0·ℓ/(L+ℓ), without requiring
    external carrier-density inputs.

    Parameters
    ----------
    lengths_m : np.ndarray
        System lengths in metres.
    G_mean : np.ndarray
        Mean conductance in e²/h units.
    cross_section_m2 : float
        Transverse cross-section area (m²).

    Returns
    -------
    dict with keys:
        'mfp_m'         : effective mean free path (from affine 1/G fit)
        'mfp_nm'        : effective mean free path in nanometres
        'sigma_S_per_m' : conductivity (S/m)
        'fit_R2'        : coefficient of determination
        'regime'        : 'diffusive' | 'ballistic' | 'mixed'
    """
    e2h = 7.748091729e-5   # S

    G_SI = G_mean * e2h    # S

    lengths_m = np.asarray(lengths_m, dtype=float)
    if lengths_m.size < 2:
        return {"error": "at least two length points are required for MFP extraction"}
    if cross_section_m2 <= 0:
        return {"error": f"cross_section_m2 must be > 0, got {cross_section_m2}"}

    try:
        inv_G = 1.0 / G_SI
        if not np.all(np.isfinite(inv_G)):
            return {"error": "non-finite conductance values encountered in G(L) fit"}

        # y = a + b x
        slope, intercept = np.polyfit(lengths_m, inv_G, deg=1)
        if not np.isfinite(slope) or not np.isfinite(intercept):
            return {"error": "failed to fit affine inverse-conductance model"}
        if slope <= 0:
            return {"error": f"non-physical slope from 1/G fit: {slope}"}

        sigma = 1.0 / (slope * cross_section_m2)

        # R² for quality of fit
        fit = intercept + slope * lengths_m
        residuals = inv_G - fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((inv_G - np.mean(inv_G))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    except Exception as exc:
        return {"error": str(exc)}

    # Effective MFP from affine inverse-conductance fit.
    mfp_eff_m = None
    if intercept > 0:
        mfp_eff_m = float(intercept / slope)
        if not np.isfinite(mfp_eff_m) or mfp_eff_m <= 0:
            mfp_eff_m = None

    regime = _identify_regime(lengths_m, G_SI)

    return {
        "sigma_S_per_m": sigma,
        "mfp_m": mfp_eff_m,
        "mfp_nm": (mfp_eff_m * 1e9) if mfp_eff_m is not None else None,
        "fit_slope_inv_sigmaA": float(slope),
        "fit_intercept_inv_G0": float(intercept),
        "mfp_estimation_model": "affine_inverse_conductance",
        "fit_R2": r2,
        "regime": regime,
    }


def mfp_from_sigma(
    sigma_S_per_m: float,
    carrier_density_m3: float,
    fermi_velocity_m_per_s: float,
) -> dict:
    """Compute MFP from conductivity using Drude formula.

    Parameters
    ----------
    sigma_S_per_m : float
        Electrical conductivity (S/m).
    carrier_density_m3 : float
        3D carrier density (m⁻³).
    fermi_velocity_m_per_s : float
        Fermi velocity (m/s).

    Returns
    -------
    dict with 'mfp_m', 'mfp_nm', 'tau_s' (scattering time)
    """
    e = 1.602e-19   # C
    m_e = 9.109e-31  # kg
    # Drude: σ = n·e²·τ/m  →  τ = σ·m / (n·e²)
    # ℓ = vF·τ
    tau = sigma_S_per_m * m_e / (carrier_density_m3 * e**2)
    mfp_m = fermi_velocity_m_per_s * tau
    return {
        "mfp_m": mfp_m,
        "mfp_nm": mfp_m * 1e9,
        "tau_s": tau,
    }


def _identify_regime(lengths_m: np.ndarray, G_SI: np.ndarray) -> str:
    """Heuristic: if G varies less than 10% it's ballistic, otherwise diffusive."""
    variation = np.std(G_SI) / np.mean(G_SI) if np.mean(G_SI) > 0 else 0
    if variation < 0.10:
        return "ballistic"
    elif variation > 0.30:
        return "diffusive"
    return "mixed"
