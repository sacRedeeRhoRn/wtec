"""Geometry helpers for axis-aware transport algebra."""

from __future__ import annotations

import numpy as np


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def axis_to_index(axis: str) -> int:
    """Map axis label to index."""
    key = axis.lower().strip()
    if key not in AXIS_INDEX:
        raise ValueError(f"axis must be one of ['x', 'y', 'z'], got {axis!r}")
    return AXIS_INDEX[key]


def axis_length_m(lattice_vectors: np.ndarray, n_cells: int, axis: str) -> float:
    """Physical length along axis for n_cells."""
    idx = axis_to_index(axis)
    vec_ang = np.asarray(lattice_vectors, dtype=float)[idx]
    return float(abs(int(n_cells)) * np.linalg.norm(vec_ang) * 1e-10)


def cross_section_m2(
    lattice_vectors: np.ndarray,
    n_layers_x: int,
    n_layers_y: int,
    n_layers_z: int,
    *,
    lead_axis: str,
) -> float:
    """Cross-sectional area perpendicular to lead_axis."""
    lv = np.asarray(lattice_vectors, dtype=float)
    counts = [int(n_layers_x), int(n_layers_y), int(n_layers_z)]
    if any(c <= 0 for c in counts):
        raise ValueError("n_layers_x, n_layers_y, n_layers_z must be > 0")

    lead_idx = axis_to_index(lead_axis)
    perp = [i for i in (0, 1, 2) if i != lead_idx]
    v1_ang = lv[perp[0]] * counts[perp[0]]
    v2_ang = lv[perp[1]] * counts[perp[1]]
    return float(np.linalg.norm(np.cross(v1_ang, v2_ang)) * 1e-20)


def region_geometry(
    lattice_vectors: np.ndarray,
    *,
    n_layers_x: int,
    n_layers_y: int,
    n_layers_z: int,
    lead_axis: str,
    thickness_axis: str = "z",
) -> dict[str, float]:
    """Return axis-aware geometry quantities in SI units."""
    counts = {
        "x": int(n_layers_x),
        "y": int(n_layers_y),
        "z": int(n_layers_z),
    }
    length_m = axis_length_m(lattice_vectors, counts[lead_axis], lead_axis)
    thickness_m = axis_length_m(lattice_vectors, counts[thickness_axis], thickness_axis)
    area_m2 = cross_section_m2(
        lattice_vectors,
        n_layers_x=counts["x"],
        n_layers_y=counts["y"],
        n_layers_z=counts["z"],
        lead_axis=lead_axis,
    )
    return {
        "length_m": float(length_m),
        "thickness_m": float(thickness_m),
        "cross_section_m2": float(area_m2),
    }
