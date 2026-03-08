"""LCAO projection specifications for Wannier90 AMN generation.

In QE+Wannier90 workflow, 'LCAO projections' means specifying
atomic orbital initial projections in the Wannier90 .win file
(begin projections / end projections block) and ensuring
pw2wannier90.x computes the AMN file using these projectors.

This module provides per-material orbital specifications and
utilities to generate the Wannier90 projections block.
"""

from __future__ import annotations

from collections import Counter


# ── Projection library ────────────────────────────────────────────────────────

# Format follows Wannier90 conventions:
#   "Element:orbital" e.g. "Ta:d", "P:p", "Si:sp3", "Co:dxy;dxz;dyz"
# See Wannier90 user guide §3.1 for full syntax.

PROJECTION_LIBRARY: dict[str, list[str]] = {
    "TaP": [
        "Ta:d",          # 5 d-orbitals (SOC doubles → 10 WFs)
        "P:p",           # 3 p-orbitals (SOC doubles → 6 WFs)  → 16 WFs total
    ],
    "NbP": [
        "Nb:d",
        "P:p",
    ],
    "CoSi": [
        "Co:d",          # 5 d-orbitals
        "Si:sp3",        # 4 sp3 hybrids
    ],
    # Generic fallbacks
    "Weyl": [
        "X:s",
    ],
}

# Number of orbital channels per projection string (without spinor doubling).
_WF_COUNT_BASE: dict[str, int] = {
    "s":   1,
    "p":   3,
    "d":   5,
    "f":   7,
    "sp3": 4,
    "sp":  2,
}


def get_projections(material: str, custom: list[str] | None = None) -> list[str]:
    """Return Wannier90-format projection strings for a material.

    Parameters
    ----------
    material : str
        Key in PROJECTION_LIBRARY (e.g. 'TaP').
    custom : list[str] | None
        If provided, overrides the library entry.

    Returns
    -------
    list of str, e.g. ['Ta:d', 'P:p']
    """
    if custom is not None:
        return custom
    if material in PROJECTION_LIBRARY:
        return PROJECTION_LIBRARY[material]
    raise KeyError(
        f"No projection preset for {material!r}. "
        f"Available: {list(PROJECTION_LIBRARY.keys())}"
    )


def _projection_channel_count(proj: str) -> int:
    """Return number of orbital channels in one projection descriptor."""
    orb_spec = proj.split(":")[-1].strip()
    orbitals = [o.strip() for o in orb_spec.split(";") if o.strip()]
    if not orbitals:
        return 1
    return sum(_WF_COUNT_BASE.get(orb, 1) for orb in orbitals)


def get_num_wann(
    material: str,
    custom_projections: list[str] | None = None,
    *,
    spinors: bool = True,
    atoms=None,
) -> int:
    """Return total number of Wannier functions for a projection set.

    Parameters
    ----------
    material : str
        Material key for projection lookup.
    custom_projections : list[str] | None
        Optional projection override.
    spinors : bool
        Whether Wannier90 runs in spinor mode (`spinors=.true.`). If true,
        each orbital channel contributes two spinor components.
    """
    projs = get_projections(material, custom_projections)
    symbol_counts = None
    if atoms is not None:
        symbol_counts = Counter(atoms.get_chemical_symbols())

    total = 0
    for p in projs:
        species = p.split(":")[0].strip()
        count = _projection_channel_count(p)
        # In Wannier90, "Ta:d" applies to all Ta atoms in the cell.
        multiplicity = int(symbol_counts.get(species, 1)) if symbol_counts is not None else 1
        total += count * multiplicity
    if spinors:
        total *= 2
    return total


def projections_block(material: str, custom: list[str] | None = None) -> str:
    """Return the Wannier90 projections block as a string.

    Example output:
        begin projections
          Ta:d
          P:p
        end projections
    """
    projs = get_projections(material, custom)
    lines = ["begin projections"]
    for p in projs:
        lines.append(f"  {p}")
    lines.append("end projections")
    return "\n".join(lines)
