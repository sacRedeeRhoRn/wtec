"""Parse VASP outputs for workflow orchestration."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


_FLOAT = r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?"


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


def parse_total_energy(outfile: str | Path) -> float:
    """Extract final total energy (eV) from OUTCAR."""
    text = _read_text(outfile)
    matches = re.findall(rf"free\s+energy\s+TOTEN\s*=\s*({_FLOAT})\s+eV", text)
    if matches:
        return float(matches[-1])
    raise ValueError(f"Final TOTEN not found in {outfile}")


def parse_elapsed_seconds(outfile: str | Path) -> float:
    """Extract final elapsed wall time in seconds from OUTCAR."""
    text = _read_text(outfile)
    m = re.search(rf"Elapsed\s+time\s+\(sec\)\s*:\s*({_FLOAT})", text)
    if not m:
        raise ValueError(f"Elapsed time not found in {outfile}")
    return float(m.group(1))


def parse_forces(outfile: str | Path) -> np.ndarray:
    """Extract final per-atom forces in eV/Ang from OUTCAR.

    Returns
    -------
    np.ndarray
        Shape ``(n_atoms, 3)``, ordered as (fx, fy, fz).
    """
    text = _read_text(outfile)
    pattern = re.compile(
        r"POSITION\s+TOTAL-FORCE\s+\(eV/Angst\)\s*\n"
        r"\s*-+\s*\n"
        r"(?P<body>.*?)"
        r"\n\s*-+\s*\n",
        re.DOTALL | re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"TOTAL-FORCE block not found in {outfile}")
    body = matches[-1].group("body")

    rows: list[list[float]] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            fx, fy, fz = (float(parts[3]), float(parts[4]), float(parts[5]))
        except ValueError:
            continue
        rows.append([fx, fy, fz])

    if not rows:
        raise ValueError(f"Could not parse forces from final TOTAL-FORCE block in {outfile}")
    return np.asarray(rows, dtype=float)


def parse_stress_kbar(outfile: str | Path) -> np.ndarray:
    """Extract final stress tensor components in kbar.

    Returns values in VASP order:
    ``[xx, yy, zz, xy, yz, zx]``.
    """
    text = _read_text(outfile)
    matches = re.findall(
        rf"in\s+kB\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})",
        text,
        flags=re.IGNORECASE,
    )
    if not matches:
        raise ValueError(f"Stress (in kB) line not found in {outfile}")
    vals = [float(v) for v in matches[-1]]
    return np.asarray(vals, dtype=float)
