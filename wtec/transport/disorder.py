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
