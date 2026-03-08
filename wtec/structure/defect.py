"""Defect engineering: vacancy, substitution, antisite."""

from __future__ import annotations

from copy import deepcopy

import numpy as np


class DefectBuilder:
    """Build defect supercells using ASE Atoms objects."""

    def __init__(self, atoms) -> None:
        """
        Parameters
        ----------
        atoms : ase.Atoms
            Primitive unit cell (will be expanded to supercell internally).
        """
        self._primitive = atoms

    # ── helpers ──────────────────────────────────────────────────────────────

    def _make_supercell(self, sc: tuple[int, int, int]):
        """Return a supercell of the primitive cell."""
        from ase.build import make_supercell
        M = np.diag(sc)
        return make_supercell(self._primitive, M)

    # ── public API ───────────────────────────────────────────────────────────

    def vacancy(
        self,
        site_index: int | list[int],
        *,
        supercell: tuple[int, int, int] = (2, 2, 2),
    ):
        """Remove atom(s) from supercell.

        Parameters
        ----------
        site_index : int or list[int]
            Atom index (or indices) in the *supercell* to remove.
        supercell : (int, int, int)
            Supercell expansion factors.

        Returns
        -------
        ase.Atoms
            Modified supercell with vacancy.
        """
        sc = self._make_supercell(supercell)
        indices = [site_index] if isinstance(site_index, int) else list(site_index)
        _check_indices(indices, len(sc))
        del sc[indices]
        sc.info["defect_type"] = "vacancy"
        sc.info["defect_sites"] = indices
        return sc

    def substitute(
        self,
        site_index: int | list[int],
        new_element: str,
        *,
        supercell: tuple[int, int, int] = (2, 2, 2),
    ):
        """Replace atom(s) with a different element.

        Parameters
        ----------
        site_index : int or list[int]
            Atom index (or indices) in the *supercell* to replace.
        new_element : str
            Chemical symbol of the substituting element.
        supercell : (int, int, int)
            Supercell expansion factors.

        Returns
        -------
        ase.Atoms
            Modified supercell.
        """
        sc = self._make_supercell(supercell)
        indices = [site_index] if isinstance(site_index, int) else list(site_index)
        _check_indices(indices, len(sc))
        syms = list(sc.get_chemical_symbols())
        for i in indices:
            syms[i] = new_element
        sc.set_chemical_symbols(syms)
        sc.info["defect_type"] = "substitution"
        sc.info["defect_sites"] = indices
        sc.info["new_element"] = new_element
        return sc

    def antisite(
        self,
        site_a: int,
        site_b: int,
        *,
        supercell: tuple[int, int, int] = (2, 2, 2),
    ):
        """Swap two atoms (antisite pair).

        Parameters
        ----------
        site_a, site_b : int
            Indices of the two atoms in the *supercell* to swap.
        supercell : (int, int, int)
            Supercell expansion factors.

        Returns
        -------
        ase.Atoms
            Modified supercell with antisite pair.
        """
        sc = self._make_supercell(supercell)
        _check_indices([site_a, site_b], len(sc))
        syms = list(sc.get_chemical_symbols())
        syms[site_a], syms[site_b] = syms[site_b], syms[site_a]
        sc.set_chemical_symbols(syms)
        sc.info["defect_type"] = "antisite"
        sc.info["defect_sites"] = [site_a, site_b]
        return sc


# ── helpers ──────────────────────────────────────────────────────────────────

def _check_indices(indices: list[int], n_atoms: int) -> None:
    for i in indices:
        if not (0 <= i < n_atoms):
            raise IndexError(
                f"Site index {i} is out of range for supercell with {n_atoms} atoms."
            )
