"""Template-driven slab generation."""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .template import InterfaceSpec, LayerSpec, SlabTemplate, load_slab_template


def _require_pymatgen() -> None:
    try:
        import pymatgen  # noqa: F401
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "pymatgen is required for slab generation. "
            "Run `wtec init` to install dependencies."
        ) from exc


def _load_source_structure(
    layer: LayerSpec,
    *,
    base_dir: Path,
    mp_api_key: str | None = None,
):
    from pymatgen.core.structure import Structure

    if layer.source == "cif":
        assert layer.cif_path is not None
        source_path = Path(layer.cif_path).expanduser()
        if not source_path.is_absolute():
            from_template = (base_dir / source_path).resolve()
            from_cwd = (Path.cwd() / source_path).resolve()
            if from_template.exists():
                source_path = from_template
            elif from_cwd.exists():
                source_path = from_cwd
            else:
                raise FileNotFoundError(
                    f"Layer {layer.label}: CIF not found: "
                    f"{from_template} (template-relative) or {from_cwd} (cwd-relative)"
                )
        if not source_path.exists():
            raise FileNotFoundError(f"Layer {layer.label}: CIF not found: {source_path}")
        return Structure.from_file(str(source_path)), str(source_path)

    if layer.source == "mp":
        assert layer.mp_id is not None
        api_key = (mp_api_key or "").strip()
        if not api_key:
            raise RuntimeError(
                f"Layer {layer.label}: source='mp' requires Materials Project API key. "
                "Set project.mp_api_key in TOML or MP_API_KEY/PMG_MAPI_KEY in environment."
            )
        try:
            from mp_api.client import MPRester
        except Exception as exc:
            raise RuntimeError(
                "mp-api is required for source='mp'. "
                "Run `wtec init` to install dependencies."
            ) from exc

        with MPRester(api_key) as mpr:
            structure = mpr.get_structure_by_material_id(layer.mp_id)
        if isinstance(structure, list):
            if not structure:
                raise RuntimeError(
                    f"Layer {layer.label}: MP returned no structure for id {layer.mp_id}. "
                    "Check that the material id is valid/current."
                )
            structure = structure[0]
        if isinstance(structure, dict):
            structure = Structure.from_dict(structure)
        if structure is None:
            raise RuntimeError(
                f"Layer {layer.label}: failed to fetch MP structure {layer.mp_id}. "
                "Check that the material id is valid/current."
            )
        return structure, f"mp:{layer.mp_id}"

    raise ValueError(f"Unsupported layer source: {layer.source}")


def _resolve_mp_api_key(tpl: SlabTemplate) -> str:
    direct = (tpl.project.mp_api_key or "").strip()
    if direct:
        return direct

    env_names: list[str] = []
    if tpl.project.mp_api_key_env:
        env_names.append(tpl.project.mp_api_key_env.strip())
    env_names.extend(["MP_API_KEY", "PMG_MAPI_KEY"])

    seen: set[str] = set()
    for name in env_names:
        key = (name or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return ""


def _build_layer_slab(layer: LayerSpec, structure, *, min_vacuum_size: float):
    from pymatgen.core.surface import SlabGenerator
    from pymatgen.io.ase import AseAtomsAdaptor

    slabgen = SlabGenerator(
        initial_structure=structure,
        miller_index=layer.miller,
        min_slab_size=float(layer.thickness_angstrom),
        min_vacuum_size=float(min_vacuum_size),
        center_slab=True,
        primitive=False,
        lll_reduce=True,
        in_unit_planes=False,
    )
    slabs = slabgen.get_slabs(symmetrize=False)
    if not slabs:
        raise RuntimeError(f"Layer {layer.label}: no slab terminations generated")

    idx = layer.termination_index
    if idx >= len(slabs):
        idx = 0
    slab = slabs[idx]
    atoms = AseAtomsAdaptor.get_atoms(slab)
    atoms.pbc = [True, True, True]
    return atoms, len(slabs), idx


def _inplane_metrics(cell: np.ndarray) -> tuple[float, float, float]:
    a = np.linalg.norm(cell[0])
    b = np.linalg.norm(cell[1])
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    return a, b, area


def _rescale_inplane(
    atoms,
    *,
    target_a: np.ndarray,
    target_b: np.ndarray,
    max_strain_percent: float,
    max_area_ratio: float,
    layer_label: str,
) -> dict[str, float]:
    cell = np.array(atoms.cell.array)
    a0, b0, area0 = _inplane_metrics(cell)
    at, bt, area_t = _inplane_metrics(np.array([target_a, target_b, cell[2]]))
    strain_a = abs(a0 - at) / max(at, 1e-12) * 100.0
    strain_b = abs(b0 - bt) / max(bt, 1e-12) * 100.0
    area_ratio = max(area0 / max(area_t, 1e-12), area_t / max(area0, 1e-12))

    if strain_a > max_strain_percent or strain_b > max_strain_percent:
        raise RuntimeError(
            f"Layer {layer_label}: in-plane mismatch too large "
            f"(strain a={strain_a:.3f}%, b={strain_b:.3f}%, "
            f"limit={max_strain_percent:.3f}%)."
        )
    if area_ratio > max_area_ratio:
        raise RuntimeError(
            f"Layer {layer_label}: area ratio too large ({area_ratio:.3f} > {max_area_ratio:.3f})."
        )

    frac = atoms.get_scaled_positions(wrap=False)
    c_len = np.linalg.norm(cell[2])
    new_cell = np.array(
        [
            target_a,
            target_b,
            np.array([0.0, 0.0, c_len]),
        ]
    )
    atoms.set_cell(new_cell, scale_atoms=False)
    atoms.set_scaled_positions(frac)
    atoms.wrap()

    return {
        "strain_a_percent": float(strain_a),
        "strain_b_percent": float(strain_b),
        "area_ratio": float(area_ratio),
    }


def _expand_inplane_supercell(atoms, *, sx: int, sy: int):
    from ase.build import make_supercell

    M = np.array(
        [
            [int(sx), 0, 0],
            [0, int(sy), 0],
            [0, 0, 1],
        ],
        dtype=int,
    )
    sc = make_supercell(atoms, M)
    sc.pbc = [True, True, True]
    return sc


def _match_inplane_supercell_and_rescale(
    atoms,
    *,
    target_a: np.ndarray,
    target_b: np.ndarray,
    max_strain_percent: float,
    max_area_ratio: float,
    max_search_supercell: int,
    layer_label: str,
) -> tuple[Any, dict[str, float]]:
    best_ok: tuple[float, int, int, Any, dict[str, float]] | None = None
    best_any: tuple[float, int, int, Any, dict[str, float]] | None = None

    max_sc = max(1, int(max_search_supercell))
    for sx in range(1, max_sc + 1):
        for sy in range(1, max_sc + 1):
            trial = _expand_inplane_supercell(atoms, sx=sx, sy=sy)
            trial_copy = trial.copy()
            metrics = _rescale_inplane(
                trial_copy,
                target_a=target_a,
                target_b=target_b,
                max_strain_percent=1e9,  # evaluate first, enforce below
                max_area_ratio=1e9,
                layer_label=layer_label,
            )
            strain_a = float(metrics["strain_a_percent"])
            strain_b = float(metrics["strain_b_percent"])
            area_ratio = float(metrics["area_ratio"])
            score = max(strain_a, strain_b) + 0.1 * area_ratio
            metrics["supercell_sx"] = int(sx)
            metrics["supercell_sy"] = int(sy)

            if best_any is None or score < best_any[0]:
                best_any = (score, sx, sy, trial_copy, metrics)

            if strain_a <= max_strain_percent and strain_b <= max_strain_percent and area_ratio <= max_area_ratio:
                if best_ok is None or score < best_ok[0]:
                    best_ok = (score, sx, sy, trial_copy, metrics)

    if best_ok is not None:
        _, _, _, matched, metrics = best_ok
        return matched, metrics

    if best_any is None:
        raise RuntimeError(f"Layer {layer_label}: failed to evaluate supercell candidates.")
    _, sx, sy, _, metrics = best_any
    raise RuntimeError(
        f"Layer {layer_label}: no valid in-plane supercell match within limits "
        f"(max_strain_percent={max_strain_percent:.3f}, max_area_ratio={max_area_ratio:.3f}, "
        f"max_search_supercell={max_sc}). "
        f"Best candidate sx={sx}, sy={sy}, "
        f"strain_a={metrics['strain_a_percent']:.3f}%, "
        f"strain_b={metrics['strain_b_percent']:.3f}%, "
        f"area_ratio={metrics['area_ratio']:.3f}."
    )


def _stack_layers(
    layer_atoms: list,
    layer_specs: list[LayerSpec],
    *,
    interface_gap_angstrom: float,
) -> tuple[Any, dict[str, tuple[float, float]], list[dict[str, Any]]]:
    if not layer_atoms:
        raise ValueError("No layer atoms to stack")

    target_cell = np.array(layer_atoms[0].cell.array)
    target_a = target_cell[0].copy()
    target_b = target_cell[1].copy()
    z_cursor = 0.0
    ranges: dict[str, tuple[float, float]] = {}
    layer_records: list[dict[str, Any]] = []
    stacked = None

    for idx, (layer, atoms) in enumerate(zip(layer_specs, layer_atoms)):
        atoms = atoms.copy()
        positions = atoms.get_positions()
        z_min = float(positions[:, 2].min())
        z_max = float(positions[:, 2].max())

        shift = z_cursor - z_min
        positions[:, 2] += shift
        atoms.set_positions(positions)
        z_min_shift = float(positions[:, 2].min())
        z_max_shift = float(positions[:, 2].max())
        ranges[layer.label] = (z_min_shift, z_max_shift)
        z_cursor = z_max_shift + float(interface_gap_angstrom)

        cell = np.array(
            [
                target_a,
                target_b,
                np.array([0.0, 0.0, np.linalg.norm(np.array(atoms.cell.array)[2])]),
            ]
        )
        atoms.set_cell(cell, scale_atoms=False)
        atoms.pbc = [True, True, True]

        if stacked is None:
            stacked = atoms.copy()
        else:
            stacked += atoms

        layer_records.append(
            {
                "index": idx,
                "label": layer.label,
                "role": layer.role,
                "source": layer.source,
                "miller": list(layer.miller),
                "thickness_angstrom": float(layer.thickness_angstrom),
                "atom_count": int(len(atoms)),
                "z_min_angstrom": z_min_shift,
                "z_max_angstrom": z_max_shift,
            }
        )

    assert stacked is not None
    return stacked, ranges, layer_records


def _choose_indices(rng: random.Random, population: list[int], k: int) -> list[int]:
    if k <= 0 or not population:
        return []
    if k >= len(population):
        return list(population)
    return rng.sample(population, k)


def _apply_interface_defects(
    atoms,
    interfaces: list[InterfaceSpec],
    *,
    layer_ranges: dict[str, tuple[float, float]],
    global_seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for idx, iface in enumerate(interfaces):
        lo, hi = iface.between
        if lo not in layer_ranges or hi not in layer_ranges:
            continue
        z_lo = layer_ranges[lo][1]
        z_hi = layer_ranges[hi][0]
        interface_z = 0.5 * (z_lo + z_hi)
        window = float(iface.vacancy_window_angstrom)
        seed = iface.vacancy_seed if iface.vacancy_seed is not None else (global_seed + idx)
        rng = random.Random(seed)

        symbols = atoms.get_chemical_symbols()
        z = atoms.get_positions()[:, 2]
        atoms_in_window = sum(
            1 for zz in z if abs(float(zz) - interface_z) <= window
        )

        removed: set[int] = set()
        vac_records: list[dict[str, Any]] = []
        if iface.vacancy_mode == "random_interface":
            for el, req in sorted(iface.vacancy_counts_by_element.items()):
                if req <= 0:
                    continue
                candidates = [
                    i
                    for i, (sym, zz) in enumerate(zip(symbols, z))
                    if sym == el and abs(float(zz) - interface_z) <= window and i not in removed
                ]
                picked = _choose_indices(rng, candidates, req)
                removed.update(picked)
                vac_records.append(
                    {
                        "element": el,
                        "requested": int(req),
                        "available": int(len(candidates)),
                        "applied": int(len(picked)),
                    }
                )

        sub_records: list[dict[str, Any]] = []
        if iface.substitutions:
            symbols_mut = list(symbols)
            for sub in iface.substitutions:
                req = int(sub.count)
                if req <= 0:
                    continue
                candidates = [
                    i
                    for i, (sym, zz) in enumerate(zip(symbols_mut, z))
                    if sym == sub.from_element
                    and abs(float(zz) - interface_z) <= window
                    and i not in removed
                ]
                picked = _choose_indices(rng, candidates, req)
                for i_pick in picked:
                    symbols_mut[i_pick] = sub.to_element
                sub_records.append(
                    {
                        "from": sub.from_element,
                        "to": sub.to_element,
                        "requested": int(req),
                        "available": int(len(candidates)),
                        "applied": int(len(picked)),
                    }
                )
            atoms.set_chemical_symbols(symbols_mut)

        if removed:
            del atoms[sorted(removed, reverse=True)]

        records.append(
            {
                "between": [lo, hi],
                "vacancy_mode": iface.vacancy_mode,
                "vacancy_window_angstrom": float(window),
                "vacancy_seed": int(seed),
                "vacancies": vac_records,
                "substitutions": sub_records,
                "atoms_removed": int(len(removed)),
                "atoms_in_window": int(atoms_in_window),
                "interface_z_angstrom": float(interface_z),
            }
        )

    return records


def _finalize_cell(atoms, *, vacuum_angstrom: float) -> None:
    pos = atoms.get_positions()
    z_min = float(pos[:, 2].min())
    z_max = float(pos[:, 2].max())
    z_span = max(1e-6, z_max - z_min)

    cell = np.array(atoms.cell.array)
    new_c = np.array([0.0, 0.0, z_span + float(vacuum_angstrom)])
    shift_z = float(vacuum_angstrom) * 0.5 - z_min
    pos[:, 2] += shift_z
    atoms.set_positions(pos)
    atoms.set_cell([cell[0], cell[1], new_c], scale_atoms=False)
    atoms.wrap()
    atoms.pbc = [True, True, True]


def _resolve_output_path(base: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def generate_slab_from_template(
    template: SlabTemplate | str | Path,
    *,
    output_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    """Generate slab CIF + metadata from template."""
    _require_pymatgen()
    tpl = template if isinstance(template, SlabTemplate) else load_slab_template(template)
    mp_api_key = _resolve_mp_api_key(tpl)

    output_dir = (
        Path(output_dir_override).expanduser().resolve()
        if output_dir_override is not None
        else _resolve_output_path(tpl.base_dir, tpl.project.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cif_path = _resolve_output_path(output_dir, tpl.export.cif_path)
    meta_path = _resolve_output_path(output_dir, tpl.export.metadata_json_path)
    cif_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    layer_atoms = []
    layer_meta: list[dict[str, Any]] = []
    target_a = None
    target_b = None

    min_layer_vacuum = max(6.0, float(tpl.stack.vacuum_angstrom) * 0.5)

    for idx, layer in enumerate(tpl.layers):
        source_structure, source_ref = _load_source_structure(
            layer,
            base_dir=tpl.base_dir,
            mp_api_key=mp_api_key,
        )
        expected_material = str(tpl.project.material or "").strip()
        if (
            expected_material
            and layer.source == "mp"
            and str(layer.role).strip().lower() == "active"
        ):
            got_formula = str(source_structure.composition.reduced_formula)
            norm = lambda s: re.sub(r"[^A-Za-z0-9]", "", str(s)).lower()
            if norm(got_formula) != norm(expected_material):
                raise RuntimeError(
                    f"Layer {layer.label}: MP structure formula mismatch for active material. "
                    f"expected={expected_material!r}, got={got_formula!r}, source={source_ref!r}"
                )
        atoms, n_terminations, selected_termination = _build_layer_slab(
            layer,
            source_structure,
            min_vacuum_size=min_layer_vacuum,
        )

        if idx == 0:
            cell0 = np.array(atoms.cell.array)
            target_a = cell0[0].copy()
            target_b = cell0[1].copy()
            metrics = {
                "strain_a_percent": 0.0,
                "strain_b_percent": 0.0,
                "area_ratio": 1.0,
                "supercell_sx": 1,
                "supercell_sy": 1,
            }
        else:
            assert target_a is not None and target_b is not None
            atoms, metrics = _match_inplane_supercell_and_rescale(
                atoms,
                target_a=target_a,
                target_b=target_b,
                max_strain_percent=float(tpl.matching.max_strain_percent),
                max_area_ratio=float(tpl.matching.max_area_ratio),
                max_search_supercell=int(tpl.matching.max_search_supercell),
                layer_label=layer.label,
            )

        layer_atoms.append(atoms)
        layer_meta.append(
            {
                "index": idx,
                "label": layer.label,
                "role": layer.role,
                "source": layer.source,
                "source_ref": source_ref,
                "miller": list(layer.miller),
                "thickness_angstrom": float(layer.thickness_angstrom),
                "termination_index_requested": int(layer.termination_index),
                "termination_index_used": int(selected_termination),
                "termination_count": int(n_terminations),
                **metrics,
            }
        )

    stacked, layer_ranges, layer_records = _stack_layers(
        layer_atoms,
        tpl.layers,
        interface_gap_angstrom=float(tpl.stack.interface_gap_angstrom),
    )
    atoms_before_defects = len(stacked)
    iface_records = _apply_interface_defects(
        stacked,
        tpl.interfaces,
        layer_ranges=layer_ranges,
        global_seed=int(tpl.project.seed),
    )
    _finalize_cell(stacked, vacuum_angstrom=float(tpl.stack.vacuum_angstrom))

    from ase import io as ase_io

    ase_io.write(str(cif_path), stacked, format="cif")

    symbols = stacked.get_chemical_symbols()
    composition = dict(sorted(Counter(symbols).items()))
    z = stacked.get_positions()[:, 2]
    z_span = float(z.max() - z.min()) if len(z) else 0.0

    metadata: dict[str, Any] = {
        "wtec_slab_metadata_version": 2,
        "generated_at_epoch": int(time.time()),
        "template_path": str(tpl.template_path),
        "project": {
            "name": tpl.project.name,
            "seed": int(tpl.project.seed),
            "output_dir": str(output_dir),
            "material": tpl.project.material,
        },
        "matching": {
            "max_strain_percent": float(tpl.matching.max_strain_percent),
            "max_area_ratio": float(tpl.matching.max_area_ratio),
            "max_search_supercell": int(tpl.matching.max_search_supercell),
        },
        "stack": {
            "align_axis": tpl.stack.align_axis,
            "vacuum_angstrom": float(tpl.stack.vacuum_angstrom),
            "interface_gap_angstrom": float(tpl.stack.interface_gap_angstrom),
        },
        "export": {
            "cif_path": str(cif_path),
            "metadata_json_path": str(meta_path),
        },
        "layers": [],
        "interfaces": iface_records,
        "summary": {
            "atoms_before_defects": int(atoms_before_defects),
            "atoms_after_defects": int(len(stacked)),
            "formula": stacked.get_chemical_formula(mode="reduce"),
            "composition": composition,
            "z_span_angstrom": z_span,
        },
    }

    layer_range_map = {item["label"]: item for item in layer_records}
    for layer in layer_meta:
        info = layer_range_map.get(layer["label"], {})
        combined = dict(layer)
        if info:
            combined.update(
                {
                    "atom_count": info.get("atom_count"),
                    "z_min_angstrom": info.get("z_min_angstrom"),
                    "z_max_angstrom": info.get("z_max_angstrom"),
                }
            )
        metadata["layers"].append(combined)

    meta_path.write_text(json.dumps(metadata, indent=2))

    return {
        "cif_path": str(cif_path),
        "metadata_path": str(meta_path),
        "atoms": int(len(stacked)),
        "formula": stacked.get_chemical_formula(mode="reduce"),
    }
