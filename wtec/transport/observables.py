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
