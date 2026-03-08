"""Lead construction for the trilayer geometry."""

from __future__ import annotations

import numpy as np


def insulating_lead(lat, n_layers_z: int, a_vec, num_orbs: int, gap_eV: float = 50.0):
    """Build an insulating lead with a large onsite gap.

    The lead is semi-infinite along x and represents the substrate contact.
    A large onsite energy shifts all bands away from the Fermi window,
    making the lead effectively insulating.

    Parameters
    ----------
    lat : kwant.lattice
        Lattice shared with the scattering region.
    n_layers_z : int
        Number of z layers (same as film).
    a_vec : np.ndarray
        Translational vector (x direction).
    num_orbs : int
        Number of orbitals per site.
    gap_eV : float
        Onsite energy shift (eV). Default 50 eV >> Fermi window.
    """
    try:
        import kwant
    except ImportError:
        raise ImportError("kwant is required")

    sym = kwant.TranslationalSymmetry(a_vec)
    lead = kwant.Builder(sym)
    onsite = gap_eV * np.eye(num_orbs, dtype=complex)
    for iz in range(n_layers_z):
        lead[lat(0, 0, iz)] = onsite
    return lead


def metallic_lead(lat, n_layers_z: int, a_vec, bandwidth_eV: float = 5.0):
    """Build a flat-band metallic lead (wide bandwidth, featureless).

    Parameters
    ----------
    bandwidth_eV : float
        Half-bandwidth for the lead hopping (nearest-neighbour tight-binding).
    """
    try:
        import kwant
    except ImportError:
        raise ImportError("kwant is required")

    sym = kwant.TranslationalSymmetry(a_vec)
    lead = kwant.Builder(sym)
    for iz in range(n_layers_z):
        lead[lat(0, 0, iz)] = np.zeros((1, 1), dtype=complex)   # flat onsite
        # nearest-neighbour hopping in x for the lead (periodic direction)
        lead[lat(0, 0, iz), lat(1, 0, iz)] = -bandwidth_eV * np.ones((1, 1), dtype=complex)
    return lead
