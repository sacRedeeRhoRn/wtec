"""Parse Wannier90 _hr.dat output into numpy hopping arrays."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


class HoppingData(NamedTuple):
    """Container for Wannier90 tight-binding Hamiltonian."""
    num_wann: int
    r_vectors: np.ndarray      # shape (n_R, 3) int  — R-vectors in lattice coords
    deg: np.ndarray            # shape (n_R,) int    — degeneracy weights
    H_R: np.ndarray            # shape (n_R, nw, nw) complex  — hopping matrices


def read_hr_dat(path: str | Path) -> HoppingData:
    """Parse a Wannier90 *_hr.dat file.

    Parameters
    ----------
    path : str or Path
        Path to the *_hr.dat file.

    Returns
    -------
    HoppingData
        Structured container with R-vectors, degeneracies, and H(R) matrices.
    """
    lines = Path(path).read_text().splitlines()
    idx = 0

    # Line 0: date comment
    idx += 1

    # Line 1: num_wann
    num_wann = int(lines[idx].strip())
    idx += 1

    # Line 2: n_R (number of R vectors)
    n_R = int(lines[idx].strip())
    idx += 1

    # Degeneracy weights (spread over multiple lines, 15 per line)
    deg_values: list[int] = []
    while len(deg_values) < n_R:
        row = lines[idx].strip().split()
        deg_values.extend(int(x) for x in row)
        idx += 1
    deg = np.array(deg_values[:n_R], dtype=int)

    # Hopping entries: R1 R2 R3 n m Re Im
    r_vectors = np.zeros((n_R, 3), dtype=int)
    H_R = np.zeros((n_R, num_wann, num_wann), dtype=complex)

    r_map: dict[tuple[int, int, int], int] = {}
    r_idx = 0

    for line in lines[idx:]:
        parts = line.strip().split()
        if not parts:
            continue
        r1, r2, r3, n, m = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]) - 1, int(parts[4]) - 1
        re_val, im_val = float(parts[5]), float(parts[6])
        key = (r1, r2, r3)
        if key not in r_map:
            r_map[key] = r_idx
            r_vectors[r_idx] = [r1, r2, r3]
            r_idx += 1
        ri = r_map[key]
        H_R[ri, m, n] = complex(re_val, im_val)   # Wannier90 convention: row=m, col=n

    # Trim to actual number of R vectors found
    n_found = len(r_map)
    r_vectors = r_vectors[:n_found]
    H_R = H_R[:n_found]
    deg = deg[:n_found] if len(deg) >= n_found else np.ones(n_found, dtype=int)

    return HoppingData(num_wann=num_wann, r_vectors=r_vectors, deg=deg, H_R=H_R)


def write_hr_dat(
    path: str | Path,
    hd: HoppingData,
    *,
    header: str = "Written by wtec.wannier.parser.write_hr_dat",
) -> None:
    """Write Wannier90 *_hr.dat from :class:`HoppingData`.

    The in-memory matrix convention matches :func:`read_hr_dat`:
    ``H_R[ri, m, n]`` corresponds to the file tuple ``(n, m)``.
    """
    out = Path(path)
    nw = int(hd.num_wann)
    r_vectors = np.asarray(hd.r_vectors, dtype=int)
    deg = np.asarray(hd.deg, dtype=int)
    H_R = np.asarray(hd.H_R, dtype=complex)

    if r_vectors.ndim != 2 or r_vectors.shape[1] != 3:
        raise ValueError("r_vectors must have shape (n_R, 3)")
    n_r = int(r_vectors.shape[0])
    if deg.shape[0] != n_r:
        raise ValueError("deg length must match number of r_vectors")
    if H_R.shape != (n_r, nw, nw):
        raise ValueError(f"H_R must have shape ({n_r}, {nw}, {nw})")

    lines: list[str] = []
    lines.append(str(header))
    lines.append(str(nw))
    lines.append(str(n_r))

    for start in range(0, n_r, 15):
        chunk = deg[start : start + 15]
        lines.append(" ".join(str(int(v)) for v in chunk))

    for ri in range(n_r):
        r1, r2, r3 = [int(v) for v in r_vectors[ri]]
        for n in range(nw):
            for m in range(nw):
                val = H_R[ri, m, n]
                lines.append(
                    f"{r1:4d}{r2:4d}{r3:4d}{n+1:4d}{m+1:4d}"
                    f"{float(np.real(val)):18.10f}{float(np.imag(val)):18.10f}"
                )

    out.write_text("\n".join(lines) + "\n")


def interpolate_bands(
    hd: HoppingData,
    k_path: np.ndarray,
    lattice_vectors: np.ndarray,
) -> np.ndarray:
    """Interpolate band structure along a k-path using Wannier interpolation.

    Parameters
    ----------
    hd : HoppingData
    k_path : np.ndarray shape (n_k, 3)
        k-points in fractional coordinates.
    lattice_vectors : np.ndarray shape (3, 3)
        Real-space lattice vectors (rows), in Angstroms.

    Returns
    -------
    np.ndarray shape (n_k, num_wann)
        Eigenvalues in eV at each k-point.
    """
    nw = hd.num_wann
    nk = len(k_path)
    bands = np.empty((nk, nw))

    # Reciprocal lattice vectors (2π/Å convention)
    rec = 2 * np.pi * np.linalg.inv(lattice_vectors).T

    for ik, kf in enumerate(k_path):
        # Convert fractional → Cartesian in 1/Å
        kc = kf @ rec
        # H(k) = sum_R exp(i k·R) H(R) / deg(R)
        H_k = np.zeros((nw, nw), dtype=complex)
        for ri, R in enumerate(hd.r_vectors):
            # R in Cartesian (Å)
            Rc = R @ lattice_vectors
            phase = np.exp(1j * np.dot(kc, Rc))
            H_k += phase / hd.deg[ri] * hd.H_R[ri]
        evals = np.linalg.eigvalsh(H_k)
        bands[ik] = evals

    return bands
