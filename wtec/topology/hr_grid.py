"""Helpers for per-point HR grid preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from wtec.structure.io import read as read_structure
from wtec.structure.io import write as write_structure


def _load_metadata(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _middle_layer_window(metadata: dict[str, Any], *, middle_role: str) -> tuple[float, float]:
    layers = metadata.get("layers", [])
    if not isinstance(layers, list) or not layers:
        raise RuntimeError("variant metadata has no layers[] for thickness mapping")

    selected = None
    role_norm = str(middle_role).strip().lower()
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        role = str(layer.get("role", "")).strip().lower()
        if role == role_norm:
            selected = layer
            break
    if selected is None:
        raise RuntimeError(
            f"middle layer role {middle_role!r} not found in variant metadata layers"
        )

    z0 = selected.get("z_min_angstrom")
    z1 = selected.get("z_max_angstrom")
    if z0 is None or z1 is None:
        raise RuntimeError("middle layer metadata missing z_min_angstrom/z_max_angstrom")
    z0f = float(z0)
    z1f = float(z1)
    if z1f <= z0f:
        raise RuntimeError("middle layer z-range is invalid")
    return z0f, z1f


def scale_middle_layer_structure(
    *,
    structure_path: str | Path,
    metadata_path: str | Path | None,
    thickness_uc: int,
    reference_thickness_uc: int,
    middle_role: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Scale only the middle-layer z-thickness and shift upper layers."""
    if int(thickness_uc) <= 0:
        raise ValueError("thickness_uc must be > 0")
    if int(reference_thickness_uc) <= 0:
        raise ValueError("reference_thickness_uc must be > 0")

    atoms = read_structure(structure_path)
    metadata = _load_metadata(metadata_path)
    z0, z1 = _middle_layer_window(metadata, middle_role=middle_role)

    scale = float(thickness_uc) / float(reference_thickness_uc)
    if abs(scale - 1.0) < 1e-12:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        write_structure(atoms, out, fmt="cif")
        return {
            "structure_path": str(out),
            "scale_factor": 1.0,
            "delta_thickness_angstrom": 0.0,
            "middle_z_min_angstrom": z0,
            "middle_z_max_angstrom": z1,
        }

    pos = np.asarray(atoms.get_positions(), dtype=float)
    z = pos[:, 2]
    middle_mask = (z >= z0) & (z <= z1)
    top_mask = z > z1
    middle_thickness = z1 - z0
    delta = middle_thickness * (scale - 1.0)

    # Stretch only the middle region around its lower interface plane.
    pos[middle_mask, 2] = z0 + (pos[middle_mask, 2] - z0) * scale
    # Shift all upper layers rigidly.
    pos[top_mask, 2] = pos[top_mask, 2] + delta
    atoms.set_positions(pos)

    cell = np.asarray(atoms.cell.array, dtype=float)
    c_vec = np.array(cell[2], dtype=float)
    c_len = float(np.linalg.norm(c_vec))
    if c_len <= 1e-12:
        raise RuntimeError("invalid cell c-vector length")
    c_hat = c_vec / c_len
    cell[2] = c_hat * (c_len + delta)
    atoms.set_cell(cell, scale_atoms=False)
    atoms.wrap()

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_structure(atoms, out, fmt="cif")
    return {
        "structure_path": str(out),
        "scale_factor": float(scale),
        "delta_thickness_angstrom": float(delta),
        "middle_z_min_angstrom": float(z0),
        "middle_z_max_angstrom": float(z1),
    }
