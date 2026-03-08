"""Supercell utilities."""

from __future__ import annotations

import numpy as np


def make_supercell(atoms, nx: int, ny: int, nz: int):
    """Build an nx × ny × nz supercell.

    Returns
    -------
    ase.Atoms
    """
    from ase.build import make_supercell as _make_sc
    M = np.diag([nx, ny, nz])
    return _make_sc(atoms, M)


def slab(atoms, n_layers: int, surface_direction: int = 2, vacuum: float = 15.0):
    """Build a slab with n_layers unit cells along surface_direction.

    Parameters
    ----------
    surface_direction : int
        0=x, 1=y, 2=z
    vacuum : float
        Vacuum padding in Angstroms (added on both sides).
    """
    reps = [1, 1, 1]
    reps[surface_direction] = n_layers
    sc = make_supercell(atoms, *reps)
    sc.center(vacuum=vacuum, axis=surface_direction)
    return sc
