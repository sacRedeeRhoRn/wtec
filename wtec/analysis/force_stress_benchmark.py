"""Force/stress benchmark helpers for VASP↔SIESTA comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wtec.siesta.parser import (
    parse_elapsed_seconds as parse_siesta_elapsed_seconds,
    parse_forces as parse_siesta_forces,
    parse_stress_kbar as parse_siesta_stress_kbar,
    parse_total_energy as parse_siesta_total_energy,
)
from wtec.vasp.parser import (
    parse_elapsed_seconds as parse_vasp_elapsed_seconds,
    parse_forces as parse_vasp_forces,
    parse_stress_kbar as parse_vasp_stress_kbar,
    parse_total_energy as parse_vasp_total_energy,
)


@dataclass(frozen=True)
class BenchmarkThresholds:
    force_mae_eva: float = 0.03
    stress_mae_kbar: float = 0.5
    energy_mev_per_atom: float = 2.0
    min_speedup: float = 3.0


def load_vasp_reference(outcar_path: str | Path) -> dict[str, Any]:
    outcar = Path(outcar_path).expanduser().resolve()
    forces = parse_vasp_forces(outcar)
    stress = parse_vasp_stress_kbar(outcar)
    return {
        "engine": "vasp",
        "outcar_path": str(outcar),
        "natoms": int(forces.shape[0]),
        "total_energy_ev": float(parse_vasp_total_energy(outcar)),
        "elapsed_seconds": float(parse_vasp_elapsed_seconds(outcar)),
        "forces_ev_per_ang": forces,
        "stress_kbar": stress,
    }


def load_siesta_result(
    out_path: str | Path,
    *,
    force_stress_path: str | Path | None = None,
    times_path: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(out_path).expanduser().resolve()
    forces = parse_siesta_forces(out, force_stress_path=force_stress_path)
    stress = parse_siesta_stress_kbar(out, force_stress_path=force_stress_path)
    return {
        "engine": "siesta",
        "out_path": str(out),
        "natoms": int(forces.shape[0]),
        "total_energy_ev": float(parse_siesta_total_energy(out)),
        "elapsed_seconds": float(parse_siesta_elapsed_seconds(out, times_path=times_path)),
        "forces_ev_per_ang": forces,
        "stress_kbar": stress,
    }


def compare_force_stress(
    *,
    reference: dict[str, Any],
    candidate: dict[str, Any],
    allow_stress_sign_flip: bool = True,
) -> dict[str, Any]:
    ref_forces = np.asarray(reference["forces_ev_per_ang"], dtype=float)
    can_forces = np.asarray(candidate["forces_ev_per_ang"], dtype=float)
    if ref_forces.shape != can_forces.shape:
        raise ValueError(
            "Force array shape mismatch: "
            f"reference={ref_forces.shape}, candidate={can_forces.shape}"
        )

    ref_stress = np.asarray(reference["stress_kbar"], dtype=float)
    can_stress = np.asarray(candidate["stress_kbar"], dtype=float)
    if ref_stress.shape != can_stress.shape:
        raise ValueError(
            "Stress vector shape mismatch: "
            f"reference={ref_stress.shape}, candidate={can_stress.shape}"
        )

    force_abs = np.abs(can_forces - ref_forces)
    force_mae = float(np.mean(force_abs))
    force_max_abs = float(np.max(force_abs))
    force_rms = float(np.sqrt(np.mean((can_forces - ref_forces) ** 2)))

    stress_abs = np.abs(can_stress - ref_stress)
    stress_mae = float(np.mean(stress_abs))
    stress_max_abs = float(np.max(stress_abs))
    stress_sign_flipped = False
    if allow_stress_sign_flip:
        stress_abs_flip = np.abs((-can_stress) - ref_stress)
        stress_mae_flip = float(np.mean(stress_abs_flip))
        if stress_mae_flip < stress_mae:
            stress_mae = stress_mae_flip
            stress_max_abs = float(np.max(stress_abs_flip))
            stress_sign_flipped = True

    natoms = int(reference.get("natoms", ref_forces.shape[0]))
    e_ref = float(reference["total_energy_ev"])
    e_can = float(candidate["total_energy_ev"])
    energy_mev_per_atom = abs(e_can - e_ref) * 1000.0 / max(1, natoms)

    t_ref = float(reference["elapsed_seconds"])
    t_can = float(candidate["elapsed_seconds"])
    if t_can <= 0.0:
        raise ValueError(f"Candidate elapsed_seconds must be > 0, got {t_can}")
    speedup = t_ref / t_can

    return {
        "natoms": natoms,
        "force_mae_eva": force_mae,
        "force_max_abs_eva": force_max_abs,
        "force_rms_eva": force_rms,
        "stress_mae_kbar": stress_mae,
        "stress_max_abs_kbar": stress_max_abs,
        "stress_sign_flipped": stress_sign_flipped,
        "energy_mev_per_atom": float(energy_mev_per_atom),
        "speedup_vs_reference": float(speedup),
    }


def evaluate_thresholds(
    metrics: dict[str, Any],
    thresholds: BenchmarkThresholds,
) -> dict[str, Any]:
    checks = {
        "force_mae": float(metrics["force_mae_eva"]) <= float(thresholds.force_mae_eva),
        "stress_mae": float(metrics["stress_mae_kbar"]) <= float(thresholds.stress_mae_kbar),
        "energy": float(metrics["energy_mev_per_atom"]) <= float(thresholds.energy_mev_per_atom),
        "speedup": float(metrics["speedup_vs_reference"]) >= float(thresholds.min_speedup),
    }
    return {
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def to_serializable_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Convert benchmark payload containing numpy arrays to JSON-serializable dict."""
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = to_serializable_payload(v)
        elif isinstance(v, list):
            out[k] = [to_serializable_payload(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def choose_fastest_passing_case(cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [c for c in cases if bool((c.get("evaluation") or {}).get("pass", False))]
    if not passing:
        return None
    return min(passing, key=lambda c: float((c.get("candidate") or {}).get("elapsed_seconds", 1.0e30)))
