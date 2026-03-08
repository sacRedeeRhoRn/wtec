"""Structure I/O: read/write CIF, POSCAR, QE format."""

from __future__ import annotations

from pathlib import Path


def read(path: str | Path):
    """Read any structure file supported by ASE."""
    import ase.io
    return ase.io.read(str(path))


def write(atoms, path: str | Path, fmt: str | None = None) -> None:
    """Write structure to file. Format auto-detected from extension."""
    import ase.io
    ase.io.write(str(path), atoms, format=fmt)


def to_qe_string(
    atoms,
    ecutwfc: float = 80.0,
    ecutrho: float = 640.0,
    pseudopots: dict[str, str] | None = None,
) -> str:
    """Return a minimal QE ATOMIC_SPECIES + ATOMIC_POSITIONS block.

    For full pw.x inputs use wtec.qe.inputs.QEInputGenerator.
    """
    lines = []
    cell = atoms.get_cell()
    symbols = atoms.get_chemical_symbols()
    masses = atoms.get_masses()
    unique = list(dict.fromkeys(symbols))  # preserve order

    lines.append("ATOMIC_SPECIES")
    for s in unique:
        idx = symbols.index(s)
        pp_name = (pseudopots or {}).get(s, f"{s}.UPF")
        lines.append(f"  {s}  {masses[idx]:.4f}  {pp_name}")

    lines.append("ATOMIC_POSITIONS {crystal}")
    pos = atoms.get_scaled_positions()
    for s, p in zip(symbols, pos):
        lines.append(f"  {s}  {p[0]:.8f}  {p[1]:.8f}  {p[2]:.8f}")

    lines.append("CELL_PARAMETERS {angstrom}")
    for row in cell:
        lines.append(f"  {row[0]:.8f}  {row[1]:.8f}  {row[2]:.8f}")

    return "\n".join(lines)
