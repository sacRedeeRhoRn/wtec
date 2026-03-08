"""ASCII report for generated slab structures."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _auto_metadata_path(cif_path: Path) -> Path | None:
    candidates = [
        cif_path.with_suffix(".meta.json"),
        cif_path.with_name(cif_path.stem + ".meta.json"),
        cif_path.parent / "slab_metadata.json",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _load_metadata(metadata_path: Path | None) -> dict[str, Any] | None:
    if metadata_path is None:
        return None
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text())
    except Exception:
        return None


def _format_composition(symbols: list[str]) -> str:
    counts = Counter(symbols)
    return ", ".join(f"{k}:{counts[k]}" for k in sorted(counts))


def _layer_label_for_z(z_mid: float, metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return "-"
    layers = metadata.get("layers", [])
    if not isinstance(layers, list):
        return "-"
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        z0 = layer.get("z_min_angstrom")
        z1 = layer.get("z_max_angstrom")
        if z0 is None or z1 is None:
            continue
        try:
            lo = float(z0)
            hi = float(z1)
        except Exception:
            continue
        if lo <= z_mid <= hi:
            return str(layer.get("label", "-"))
    return "-"


def _ascii_profile(
    *,
    z_positions: np.ndarray,
    symbols: list[str],
    metadata: dict[str, Any] | None,
    rows: int,
    width: int,
) -> list[str]:
    z_min = float(z_positions.min())
    z_max = float(z_positions.max())
    if z_max <= z_min:
        return [f"{z_min:9.3f} |{'*'.ljust(width)}| -"]

    bins = np.linspace(z_min, z_max, rows + 1)
    max_count = 1
    bin_counts: list[list[int]] = []
    for i in range(rows):
        if i == rows - 1:
            idxs = np.where((z_positions >= bins[i]) & (z_positions <= bins[i + 1]))[0]
        else:
            idxs = np.where((z_positions >= bins[i]) & (z_positions < bins[i + 1]))[0]
        entries = idxs.tolist()
        bin_counts.append(entries)
        if len(entries) > max_count:
            max_count = len(entries)

    lines: list[str] = []
    for i in reversed(range(rows)):
        idxs = bin_counts[i]
        z_mid = 0.5 * (bins[i] + bins[i + 1])
        label = _layer_label_for_z(float(z_mid), metadata)
        if not idxs:
            line = " " * width
        else:
            dom = Counter(symbols[j] for j in idxs).most_common(1)[0][0]
            char = dom[0].upper()
            fill = max(1, int(round(len(idxs) / max_count * width)))
            fill = min(width, fill)
            line = (char * fill).ljust(width)
        lines.append(f"{z_mid:9.3f} |{line}| {label}")
    return lines


def _format_interface_details(metadata: dict[str, Any] | None) -> list[str]:
    if not metadata:
        return []
    interfaces = metadata.get("interfaces")
    if not isinstance(interfaces, list) or not interfaces:
        return []

    out = ["Interface engineering:"]
    for idx, iface in enumerate(interfaces, start=1):
        if not isinstance(iface, dict):
            continue
        pair = iface.get("between", ["?", "?"])
        if isinstance(pair, list) and len(pair) == 2:
            pair_txt = f"{pair[0]} <-> {pair[1]}"
        else:
            pair_txt = str(pair)
        removed = int(iface.get("atoms_removed", 0))
        atoms_in_window = int(iface.get("atoms_in_window", 0))
        out.append(
            f"  {idx}. {pair_txt}: removed={removed}, atoms_in_window={atoms_in_window}"
        )

        vacs = iface.get("vacancies", [])
        if isinstance(vacs, list):
            for v in vacs:
                if not isinstance(v, dict):
                    continue
                out.append(
                    "     vacancy "
                    f"{v.get('element', '?')}: "
                    f"{v.get('applied', 0)}/{v.get('requested', 0)}"
                )
        subs = iface.get("substitutions", [])
        if isinstance(subs, list):
            for s in subs:
                if not isinstance(s, dict):
                    continue
                out.append(
                    "     substitution "
                    f"{s.get('from', '?')}->{s.get('to', '?')}: "
                    f"{s.get('applied', 0)}/{s.get('requested', 0)}"
                )
    return out


def render_slab_report(
    cif_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    rows: int = 24,
    width: int = 48,
) -> str:
    """Render text report for a slab CIF with ASCII z-profile."""
    from ase import io as ase_io

    cif = Path(cif_path).expanduser().resolve()
    if not cif.exists():
        raise FileNotFoundError(f"CIF not found: {cif}")

    md_path = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else _auto_metadata_path(cif)
    )
    metadata = _load_metadata(md_path)

    atoms = ase_io.read(str(cif))
    positions = atoms.get_positions()
    z = positions[:, 2]
    symbols = atoms.get_chemical_symbols()
    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()

    lines: list[str] = []
    lines.append(f"Slab file: {cif}")
    lines.append(f"Atoms: {len(atoms)}")
    lines.append(f"Formula: {atoms.get_chemical_formula(mode='reduce')}")
    lines.append(f"Composition: {_format_composition(symbols)}")
    lines.append(
        "Cell [A, deg]: "
        f"a={a:.4f} b={b:.4f} c={c:.4f} "
        f"alpha={alpha:.2f} beta={beta:.2f} gamma={gamma:.2f}"
    )
    lines.append(f"Z span [A]: {float(z.max() - z.min()):.4f}")
    lines.append(f"Metadata: {md_path if md_path else 'not found'}")
    lines.append("")
    lines.append("ASCII z-profile (top -> bottom):")
    lines.append(f"{'z_mid[A]':>9} +{'-' * (width + 2)}+")
    lines.extend(
        _ascii_profile(
            z_positions=z,
            symbols=symbols,
            metadata=metadata,
            rows=max(4, int(rows)),
            width=max(16, int(width)),
        )
    )
    lines.append(f"{'':>9} +{'-' * (width + 2)}+")

    iface_lines = _format_interface_details(metadata)
    if iface_lines:
        lines.append("")
        lines.extend(iface_lines)

    return "\n".join(lines)
