"""Disorder implementations for Kwant systems.

All functions return a *new* finalized-ready Builder with disorder applied.
They do NOT finalize the system (call sys.finalized() yourself).

Fork-based parallelism is never used here.
Ensemble averaging is handled in transport.conductance via mpi4py rank split.
"""

from __future__ import annotations

import numpy as np


def add_anderson_disorder(
    sys,
    strength: float,
    *,
    rng: np.random.Generator | None = None,
    rng_seed: int | None = None,
    sites=None,
) -> None:
    """Add Anderson disorder W·ξ (ξ ∈ [-0.5, 0.5]) to onsite terms in-place.

    Parameters
    ----------
    sys : kwant.Builder
        Mutable Kwant builder (NOT finalized).
    strength : float
        Disorder amplitude W (in eV). Disorder is W·ξ.
    rng : np.random.Generator | None
        Random generator. If None, seeded by rng_seed.
    rng_seed : int | None
        Seed for reproducibility.
    sites : iterable | None
        Sites to disorder. None = all sites in the scattering region.
    """
    if rng is None:
        rng = np.random.default_rng(rng_seed)

    all_sites = list(sys.sites()) if sites is None else list(sites)
    for site in all_sites:
        current = sys[site]
        if callable(current):
            raise ValueError(
                "Cannot add disorder to a site with callable onsite. "
                "Fix the onsite to a matrix first."
            )
        h = np.asarray(current, dtype=complex)
        xi = rng.uniform(-0.5, 0.5, size=h.shape[0])
        sys[site] = h + strength * np.diag(xi)


def add_surface_anderson_disorder(
    sys,
    *,
    surface_strength: float,
    bulk_strength: float = 0.0,
    n_surface_layers: int = 2,
    rng: np.random.Generator | None = None,
    rng_seed: int | None = None,
) -> None:
    """Anderson disorder with enhanced amplitude on top/bottom surface layers.

    Physics motivation
    ------------------
    Fermi-arc states in a topological semimetal are concentrated on the top and
    bottom surfaces (penetration depth λ_arc ≈ 5 unit cells for TaP).  Interface
    defects (O vacancies, SiO₂ roughness) therefore scatter arcs far more
    efficiently than bulk dopants of comparable density.  This function applies
    stronger disorder W_surface on the outermost ``n_surface_layers`` unit-cell
    planes (identified by z-tag) and weaker W_bulk elsewhere.

    Parameters
    ----------
    sys : kwant.Builder
        Mutable Kwant builder (NOT finalized).
    surface_strength : float
        Disorder amplitude W_surface (eV) for the top/bottom surface layers.
    bulk_strength : float
        Disorder amplitude W_bulk (eV) for interior sites.  0 = clean bulk.
    n_surface_layers : int
        Number of unit-cell planes from each surface to treat as surface.
        Default 2; for TaP with λ_arc≈5 uc, values up to 5 are physical.
    rng : np.random.Generator | None
        Random generator.  If None, seeded by rng_seed.
    rng_seed : int | None
        Seed for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng(rng_seed)

    all_sites = list(sys.sites())
    if not all_sites:
        return

    # Identify z-tags to determine surface planes
    try:
        z_tags = np.array([int(site.tag[2]) for site in all_sites], dtype=int)
    except (AttributeError, IndexError):
        # Fallback: treat all sites as bulk
        add_anderson_disorder(sys, bulk_strength, rng=rng)
        return

    z_min = int(np.min(z_tags))
    z_max = int(np.max(z_tags))
    surface_z_lo = set(range(z_min, z_min + n_surface_layers))
    surface_z_hi = set(range(z_max - n_surface_layers + 1, z_max + 1))
    surface_z = surface_z_lo | surface_z_hi

    for site, zt in zip(all_sites, z_tags):
        current = sys[site]
        if callable(current):
            raise ValueError(
                f"Cannot add disorder to site {site} with callable onsite. "
                "Fix the onsite to a matrix first."
            )
        h = np.asarray(current, dtype=complex)
        norbs = h.shape[0]
        W = surface_strength if int(zt) in surface_z else bulk_strength
        if W == 0.0:
            continue
        xi = rng.uniform(-0.5, 0.5, size=norbs)
        sys[site] = h + W * np.diag(xi)


def add_vacancy_disorder(
    sys,
    concentration: float,
    *,
    rng: np.random.Generator | None = None,
    rng_seed: int | None = None,
) -> list:
    """Remove a random fraction of sites (vacancy disorder).

    Parameters
    ----------
    concentration : float
        Fraction of bulk sites to remove (0 < concentration < 1).

    Returns
    -------
    list
        List of removed site tags.
    """
    if rng is None:
        rng = np.random.default_rng(rng_seed)

    all_sites = list(sys.sites())
    n_remove = int(round(concentration * len(all_sites)))
    remove_idx = rng.choice(len(all_sites), size=n_remove, replace=False)
    removed = [all_sites[i] for i in remove_idx]
    del sys[removed]
    return removed


def add_substitutional_disorder(
    sys,
    concentration: float,
    delta_onsite: float | np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    rng_seed: int | None = None,
) -> list:
    """Replace a random fraction of sites with a substitutional impurity.

    The impurity is modelled as a shift δE to the onsite matrix:
        H_site → H_site + δE * I   (scalar δE)
    or
        H_site → H_site + diag(δE) (array δE, length = norbs)

    Parameters
    ----------
    concentration : float
        Fraction of sites to substitute.
    delta_onsite : float or np.ndarray
        Onsite energy shift for substituted sites.

    Returns
    -------
    list
        List of substituted site tags.
    """
    if rng is None:
        rng = np.random.default_rng(rng_seed)

    all_sites = list(sys.sites())
    n_subst = int(round(concentration * len(all_sites)))
    subst_idx = rng.choice(len(all_sites), size=n_subst, replace=False)
    substituted = [all_sites[i] for i in subst_idx]

    for site in substituted:
        h = np.asarray(sys[site], dtype=complex)
        norbs = h.shape[0]
        if np.isscalar(delta_onsite):
            shift = float(delta_onsite) * np.eye(norbs)
        else:
            shift = np.diag(np.asarray(delta_onsite, dtype=float)[:norbs])
        sys[site] = h + shift

    return substituted
