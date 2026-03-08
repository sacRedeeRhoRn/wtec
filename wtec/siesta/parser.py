"""Parsers for SIESTA output logs."""

from __future__ import annotations

import re
from pathlib import Path


_FERMI_PATTERNS = [
    re.compile(r"Fermi\s*=\s*([\-+0-9.Ee]+)\s*eV", re.IGNORECASE),
    re.compile(r"Fermi energy\s*[:=]\s*([\-+0-9.Ee]+)\s*eV", re.IGNORECASE),
]


def parse_fermi_energy(path: str | Path) -> float:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"SIESTA output not found: {p}")
    text = p.read_text(errors="ignore")
    for pat in _FERMI_PATTERNS:
        m = pat.search(text)
        if m:
            return float(m.group(1))
    raise RuntimeError(f"Could not parse Fermi energy from SIESTA output: {p}")


def parse_convergence(path: str | Path) -> bool:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return False
    text = p.read_text(errors="ignore")
    lower = text.lower()
    return (
        ("scf converged" in lower)
        or ("siesta: exiting due to end of run" in lower)
        or ("job done" in lower)
    )

