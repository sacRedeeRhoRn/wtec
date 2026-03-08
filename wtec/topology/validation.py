"""Pre-topology validation checks for Wannier models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_reference_bands(path: str | Path) -> dict[str, Any] | None:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def validate_wannier_model(
    tb_model,
    *,
    fermi_ev: float = 0.0,
    n_random_k: int = 32,
    hermitian_tol: float = 1e-8,
    reference_bands_path: str | Path | None = None,
    reference_tol_ev: float = 0.20,
) -> dict[str, Any]:
    """Validate model quality before topology metrics."""
    rng = np.random.default_rng(12345)

    herm_resid_max = 0.0
    finite_ok = True
    for _ in range(int(max(4, n_random_k))):
        k = rng.random(3)
        hk = np.array(tb_model.hamiltonian_at_k(k), dtype=complex)
        resid = float(np.linalg.norm(hk - hk.conj().T, ord="fro"))
        if resid > herm_resid_max:
            herm_resid_max = resid
        evals = np.linalg.eigvalsh(hk)
        if not np.all(np.isfinite(evals)):
            finite_ok = False

    out: dict[str, Any] = {
        "status": "pass",
        "hermitian_residual_max": herm_resid_max,
        "finite_eigenvalues": bool(finite_ok),
    }

    if not finite_ok or herm_resid_max > hermitian_tol:
        out["status"] = "fail"
        out["reason"] = "model_nonhermitian_or_nonfinite"

    if reference_bands_path:
        ref = _load_reference_bands(reference_bands_path)
        if ref is None:
            out["reference_status"] = "missing_or_invalid"
        else:
            try:
                k_path = np.array(ref["k_path"], dtype=float)
                dft = np.array(ref["bands_dft_ev"], dtype=float)
                tb = tb_model.bands(k_path)
                if dft.shape != tb.shape:
                    raise ValueError(f"shape mismatch dft={dft.shape} tb={tb.shape}")
                err = np.abs(tb - dft)
                max_err = float(np.max(err))
                rms_err = float(np.sqrt(np.mean(err**2)))
                out["reference_status"] = "ok"
                out["reference_max_abs_error_ev"] = max_err
                out["reference_rms_error_ev"] = rms_err
                if max_err > float(reference_tol_ev):
                    out["status"] = "fail"
                    out["reason"] = "reference_band_mismatch"
            except Exception as exc:
                out["reference_status"] = f"error:{type(exc).__name__}"
                if out["status"] == "pass":
                    out["status"] = "partial"
    else:
        out["reference_status"] = "not_provided"

    return out
