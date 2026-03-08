"""MPI worker for transport stage execution via PBS/qsub.

This worker is launched through mpirun; no fork-based launchstyle is used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from wtec.workflow.transport_pipeline import TransportPipeline


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(
            "Usage: python -m wtec.workflow.transport_worker <payload.json> <output.json>",
            file=sys.stderr,
        )
        return 2

    payload_path = Path(args[0]).expanduser()
    output_path = Path(args[1]).expanduser()
    if not payload_path.exists():
        print(f"Payload not found: {payload_path}", file=sys.stderr)
        return 2

    payload = json.loads(payload_path.read_text())
    if not isinstance(payload, dict):
        print("Payload must be a JSON object.", file=sys.stderr)
        return 2

    hr_dat = Path(str(payload.get("hr_dat_path", "")).strip())
    if not hr_dat.exists():
        print(f"hr_dat_path not found: {hr_dat}", file=sys.stderr)
        return 2

    win_raw = payload.get("win_path")
    win_path = None
    if isinstance(win_raw, str) and win_raw.strip():
        wp = Path(win_raw.strip())
        if wp.exists():
            win_path = wp

    try:
        tp = TransportPipeline(
            hr_dat_path=hr_dat,
            win_path=win_path,
            thicknesses=payload.get("thicknesses"),
            disorder_strengths=payload.get("disorder_strengths"),
            n_ensemble=int(payload.get("n_ensemble", 50)),
            energy=float(payload.get("energy", 0.0)),
            n_jobs=int(payload.get("n_jobs", 1)),
            mfp_n_layers_z=int(payload.get("mfp_n_layers_z", 10)),
            mfp_lengths=payload.get("mfp_lengths"),
            lead_onsite_eV=float(payload.get("lead_onsite_eV", 0.0)),
            base_seed=int(payload.get("base_seed", 0)),
            lead_axis=str(payload.get("lead_axis", "x")),
            thickness_axis=str(payload.get("thickness_axis", "z")),
            n_layers_x=int(payload.get("n_layers_x", 4)),
            n_layers_y=int(payload.get("n_layers_y", 4)),
            carrier_density_m3=payload.get("carrier_density_m3"),
            fermi_velocity_m_per_s=payload.get("fermi_velocity_m_per_s"),
        )
        results = tp.run_full()
    except Exception as exc:
        print(f"Transport worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"transport_results": _jsonable(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
