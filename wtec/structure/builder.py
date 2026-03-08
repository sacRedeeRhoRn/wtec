"""Build pristine structures from CIF files or material presets."""

from __future__ import annotations

from pathlib import Path


def from_cif(path: str | Path):
    """Read structure from CIF file.

    Returns
    -------
    ase.Atoms
    """
    import ase.io
    return ase.io.read(str(path))


def from_formula(formula: str):
    """Attempt to build structure from formula using ASE's built-in database.

    Only works for a limited set of common materials.
    """
    from ase.build import bulk
    return bulk(formula)


def primitive_to_conventional(atoms):
    """Convert primitive cell to conventional cell using spglib."""
    try:
        import spglib
    except ImportError:
        raise ImportError("spglib is required: pip install spglib")

    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    cell = (lattice, positions, numbers)
    conv = spglib.standardize_cell(cell, to_primitive=False)
    if conv is None:
        raise RuntimeError("spglib failed to standardize cell")
    import ase
    a = ase.Atoms(
        numbers=conv[2],
        scaled_positions=conv[1],
        cell=conv[0],
        pbc=True,
    )
    return a
