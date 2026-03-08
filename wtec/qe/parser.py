"""Parse Quantum ESPRESSO output files."""

from __future__ import annotations

import re
from pathlib import Path


def parse_fermi_energy(outfile: str | Path) -> float:
    """Extract Fermi energy (eV) from pw.x output."""
    text = Path(outfile).read_text()
    # QE prints: "the Fermi energy is    XX.XXXX ev"
    m = re.search(r"the Fermi energy is\s+([-\d.]+)\s+ev", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Also: "Fermi energy             =    XX.XXXX eV"
    m = re.search(r"Fermi energy\s*=\s*([-\d.]+)\s*eV", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    raise ValueError(f"Fermi energy not found in {outfile}")


def parse_total_energy(outfile: str | Path) -> float:
    """Extract final total energy (Ry) from pw.x output."""
    text = Path(outfile).read_text()
    # Last occurrence: "!    total energy              =     -XXX.XXXX Ry"
    matches = re.findall(r"!\s+total energy\s*=\s*([-\d.]+)\s*Ry", text)
    if not matches:
        raise ValueError(f"Total energy not found in {outfile}")
    return float(matches[-1])


def parse_nbnd(outfile: str | Path) -> int:
    """Extract number of bands from pw.x output."""
    text = Path(outfile).read_text()
    m = re.search(r"number of Kohn-Sham states\s*=\s*(\d+)", text)
    if m:
        return int(m.group(1))
    raise ValueError(f"nbnd not found in {outfile}")


def parse_convergence(outfile: str | Path) -> bool:
    """Return True if SCF converged."""
    text = Path(outfile).read_text()
    return "convergence has been achieved" in text.lower()


def parse_forces(outfile: str | Path):
    """Extract forces on each atom (eV/Å) from pw.x output.

    Returns
    -------
    list of (symbol, np.ndarray shape (3,))
    """
    import numpy as np
    text = Path(outfile).read_text()
    # "Forces acting on atoms (cartesian axes, Ry/au):"
    block = re.search(
        r"Forces acting on atoms.*?\n(.*?)\n\s*The non-local",
        text,
        re.DOTALL,
    )
    if not block:
        return []
    forces = []
    for line in block.group(1).splitlines():
        m = re.match(
            r"\s*atom\s+\d+\s+type\s+\d+\s+force\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
            line,
        )
        if m:
            forces.append(np.array([float(m.group(i)) for i in (1, 2, 3)]))
    # Convert Ry/bohr → eV/Å
    RY_BOHR_TO_EV_ANG = 25.7110
    return [f * RY_BOHR_TO_EV_ANG for f in forces]
