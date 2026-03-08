"""Parse ABACUS output files for workflow orchestration."""

from __future__ import annotations

import re
from pathlib import Path


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(errors="ignore")


def parse_fermi_energy(outfile: str | Path) -> float:
    """Extract Fermi energy in eV from ABACUS text output."""
    text = _read_text(outfile)
    patterns = [
        r"EFERMI\s*=\s*([-+]?\d+(?:\.\d+)?)\s*eV",
        r"E[_\- ]?Fermi\s*=\s*([-+]?\d+(?:\.\d+)?)\s*eV",
        r"E-fermi\s*:\s*([-+]?\d+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    raise ValueError(f"Fermi energy not found in {outfile}")


def parse_convergence(outfile: str | Path) -> bool:
    """Return True when ABACUS SCF convergence marker is present."""
    text = _read_text(outfile).lower()
    markers = [
        "charge density convergence is achieved",
        "scf convergence achieved",
        "convergence has been achieved",
        "end of scf",
    ]
    return any(m in text for m in markers)
