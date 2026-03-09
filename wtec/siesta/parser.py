"""Parsers for SIESTA output logs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


_FERMI_PATTERNS = [
    re.compile(r"Fermi\s*=\s*([\-+0-9.Ee]+)(?:\s*eV)?", re.IGNORECASE),
    re.compile(r"Fermi energy\s*[:=]\s*([\-+0-9.Ee]+)\s*eV", re.IGNORECASE),
]
_RY_BOHR_TO_EV_ANG = 25.71104309541616
_RY_BOHR3_TO_KBAR = 147105.13242194743


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _read_text(path: str | Path) -> str:
    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"SIESTA output not found: {p}")
    return p.read_text(errors="ignore")


def parse_fermi_energy(path: str | Path) -> float:
    p = _resolve(path)
    text = _read_text(p)
    for pat in _FERMI_PATTERNS:
        m = pat.search(text)
        if m:
            return float(m.group(1))
    raise RuntimeError(f"Could not parse Fermi energy from SIESTA output: {p}")


def parse_convergence(path: str | Path) -> bool:
    p = _resolve(path)
    if not p.exists():
        return False
    text = p.read_text(errors="ignore")
    lower = text.lower()
    return (
        ("scf converged" in lower)
        or ("scf cycle converged" in lower)
        or ("siesta: exiting due to end of run" in lower)
        or ("job done" in lower)
    )


def parse_total_energy(path: str | Path) -> float:
    """Extract final total energy in eV from SIESTA output."""
    text = _read_text(path)
    patterns = [
        re.compile(r"siesta:\s+Total\s*=\s*([\-+0-9.Ee]+)", re.IGNORECASE),
        re.compile(r"siesta:\s+Etot\s*=\s*([\-+0-9.Ee]+)", re.IGNORECASE),
    ]
    for pat in patterns:
        matches = pat.findall(text)
        if matches:
            return float(matches[-1])
    raise RuntimeError(f"Could not parse total energy from SIESTA output: {path}")


def parse_elapsed_seconds(path: str | Path, *, times_path: str | Path | None = None) -> float:
    """Extract elapsed wall time in seconds from SIESTA .times or .out."""
    out_path = _resolve(path)
    candidates: list[Path] = []
    if times_path is not None:
        candidates.append(_resolve(times_path))
    candidates.append(out_path.with_suffix(".times"))
    candidates.append(out_path.parent / f"{out_path.stem.split('.')[0]}.times")
    candidates.append(out_path)

    pat = re.compile(r"Total elapsed wall-clock time \(sec\)\s*=\s*([\-+0-9.Ee]+)")
    fallback = re.compile(r"Elapsed time \(sec\)\s*:\s*([\-+0-9.Ee]+)")
    for p in candidates:
        if not p.exists():
            continue
        text = p.read_text(errors="ignore")
        m = pat.search(text)
        if m:
            return float(m.group(1))
        m2 = fallback.search(text)
        if m2:
            return float(m2.group(1))
    raise RuntimeError(
        f"Could not parse elapsed time from SIESTA outputs: {[str(c) for c in candidates]}"
    )


def _resolve_force_stress_file(
    out_path: str | Path,
    force_stress_path: str | Path | None,
) -> Path:
    if force_stress_path is not None:
        p = _resolve(force_stress_path)
        if not p.exists():
            raise FileNotFoundError(f"SIESTA FORCE_STRESS file not found: {p}")
        return p
    out = _resolve(out_path)
    p = out.parent / "FORCE_STRESS"
    if not p.exists():
        raise FileNotFoundError(
            f"SIESTA FORCE_STRESS not found next to output {out}. Expected: {p}"
        )
    return p


def parse_forces(
    out_path: str | Path,
    *,
    force_stress_path: str | Path | None = None,
) -> np.ndarray:
    """Extract final per-atom forces in eV/Ang from FORCE_STRESS."""
    fs = _resolve_force_stress_file(out_path, force_stress_path)
    lines = fs.read_text(errors="ignore").splitlines()
    if len(lines) < 6:
        raise RuntimeError(f"FORCE_STRESS payload is too short: {fs}")

    natoms_line = lines[4].strip()
    try:
        natoms = int(natoms_line.split()[0])
    except Exception as exc:
        raise RuntimeError(f"Failed to parse atom count from FORCE_STRESS line 5: {natoms_line!r}") from exc

    body = lines[5 : 5 + natoms]
    if len(body) < natoms:
        raise RuntimeError(f"FORCE_STRESS force block is incomplete: expected {natoms}, got {len(body)}")

    rows: list[list[float]] = []
    for raw in body:
        parts = raw.split()
        if len(parts) < 5:
            raise RuntimeError(f"Malformed FORCE_STRESS force row: {raw!r}")
        try:
            fx, fy, fz = (float(parts[2]), float(parts[3]), float(parts[4]))
        except Exception as exc:
            raise RuntimeError(f"Failed to parse FORCE_STRESS force row: {raw!r}") from exc
        rows.append([fx, fy, fz])

    arr = np.asarray(rows, dtype=float) * _RY_BOHR_TO_EV_ANG
    return arr


def parse_stress_kbar(
    out_path: str | Path,
    *,
    force_stress_path: str | Path | None = None,
) -> np.ndarray:
    """Extract final stress components in kbar, normalized to VASP order.

    Output order is ``[xx, yy, zz, xy, yz, zx]``.
    """
    text = _read_text(out_path)
    voigt_pat = re.compile(
        r"Stress\s+tensor\s+Voigt\[x,y,z,yz,xz,xy\]\s+\(kbar\)\s*:\s*"
        r"([\-+0-9.Ee]+)\s+([\-+0-9.Ee]+)\s+([\-+0-9.Ee]+)\s+"
        r"([\-+0-9.Ee]+)\s+([\-+0-9.Ee]+)\s+([\-+0-9.Ee]+)",
        re.IGNORECASE,
    )
    matches = voigt_pat.findall(text)
    if matches:
        vals = [float(v) for v in matches[-1]]
        # SIESTA voigt order = [xx, yy, zz, yz, xz, xy]
        # Normalized order here = [xx, yy, zz, xy, yz, zx]
        return np.asarray([vals[0], vals[1], vals[2], vals[5], vals[3], vals[4]], dtype=float)

    fs = _resolve_force_stress_file(out_path, force_stress_path)
    lines = fs.read_text(errors="ignore").splitlines()
    if len(lines) < 4:
        raise RuntimeError(f"FORCE_STRESS payload is too short for stress parse: {fs}")
    try:
        m = np.asarray(
            [
                [float(x) for x in lines[1].split()[:3]],
                [float(x) for x in lines[2].split()[:3]],
                [float(x) for x in lines[3].split()[:3]],
            ],
            dtype=float,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to parse stress matrix from FORCE_STRESS: {fs}") from exc

    m_kbar = m * _RY_BOHR3_TO_KBAR
    # matrix indices: xx, yy, zz, xy, yz, zx
    return np.asarray(
        [m_kbar[0, 0], m_kbar[1, 1], m_kbar[2, 2], m_kbar[0, 1], m_kbar[1, 2], m_kbar[2, 0]],
        dtype=float,
    )
