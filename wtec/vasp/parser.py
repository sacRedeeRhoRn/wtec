"""Parse VASP outputs for workflow orchestration."""

from __future__ import annotations

import re
from pathlib import Path


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(errors="ignore")


def parse_fermi_energy(outfile: str | Path) -> float:
    """Extract Fermi energy in eV from VASP text output."""
    text = _read_text(outfile)
    patterns = [
        r"E-fermi\s*:\s*([-+]?\d+(?:\.\d+)?)",
        r"Fermi energy\s*=\s*([-+]?\d+(?:\.\d+)?)\s*eV",
        r"the Fermi energy is\s*([-+]?\d+(?:\.\d+)?)\s*eV",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    raise ValueError(f"Fermi energy not found in {outfile}")


def parse_convergence(outfile: str | Path) -> bool:
    """Return True when VASP SCF convergence marker is present."""
    text = _read_text(outfile).lower()
    markers = [
        "reached required accuracy",
        "aborting loop because ediff is reached",
        "accuracy reached",
    ]
    return any(m in text for m in markers)
