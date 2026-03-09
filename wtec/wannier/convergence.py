"""Wannier90 convergence checks used as a hard workflow gate."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np


class WannierNotConvergedError(RuntimeError):
    """Raised when Wannier90 disentanglement did not converge."""


_MAX_ITER_RE = re.compile(
    r"Warning:\s*Maximum number of disentanglement iterations reached",
    flags=re.IGNORECASE,
)
_WIN_FLOAT_RE = re.compile(
    r"^\s*(dis_num_iter|dis_win_min|dis_win_max|dis_froz_min|dis_froz_max)\s*=\s*([^\s!#]+)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_DELTA_OTOT_RE = re.compile(
    r"Delta:\s*O_D=.*?O_OD=.*?O_TOT=\s*([+-]?\d+(?:\.\d*)?E[+-]?\d+)",
    flags=re.IGNORECASE,
)


def _extract_win_params(win_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, raw_val in _WIN_FLOAT_RE.findall(win_text):
        k = key.lower()
        try:
            if k == "dis_num_iter":
                out[k] = int(float(raw_val))
            else:
                out[k] = float(raw_val)
        except Exception:
            out[k] = raw_val
    return out


def assert_wannier_converged(
    *,
    wout_path: str | Path,
    win_path: str | Path | None = None,
) -> None:
    """Raise WannierNotConvergedError when .wout indicates failed disentanglement."""
    p = Path(wout_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Wannier output not found: {p}")
    text = p.read_text(errors="ignore")
    if not _MAX_ITER_RE.search(text):
        return

    # Some Wannier90 runs print the max-iteration warning but still end at a
    # near-stationary solution and write a valid checkpoint/hr file.
    # Accept these only when the final Delta O_TOT magnitude is sufficiently
    # small and the run reached normal "All done" termination.
    if "All done: wannier90 exiting" in text:
        deltas = _DELTA_OTOT_RE.findall(text)
        if deltas:
            try:
                last_delta = abs(float(deltas[-1]))
            except Exception:
                last_delta = None
            tol_raw = os.environ.get("TOPOSLAB_WANNIER_MAXITER_DELTA_TOL", "1e-3").strip()
            try:
                tol = float(tol_raw)
            except Exception:
                tol = 1.0e-3
            if last_delta is not None and last_delta <= tol:
                return

    params: dict[str, Any] = {}
    if win_path is not None:
        wp = Path(win_path).expanduser().resolve()
        if wp.exists():
            params = _extract_win_params(wp.read_text(errors="ignore"))

    dis_iter = params.get("dis_num_iter")
    dis_win_min = params.get("dis_win_min")
    dis_win_max = params.get("dis_win_max")
    dis_froz_min = params.get("dis_froz_min")
    dis_froz_max = params.get("dis_froz_max")
    suggestion = (
        "Increase dis_num_iter, widen dis_win, or narrow dis_froz window. "
        "Recommended start: dis_num_iter=1000, dis_win +/-2 eV wider, "
        "dis_froz +/-0.5 eV tighter."
    )
    raise WannierNotConvergedError(
        "Wannier90 disentanglement did not converge: "
        f"matched pattern={_MAX_ITER_RE.pattern!r}; "
        f"dis_num_iter={dis_iter!r}, dis_win=({dis_win_min!r},{dis_win_max!r}), "
        f"dis_froz=({dis_froz_min!r},{dis_froz_max!r}). {suggestion}"
    )


class WannierTopologyError(RuntimeError):
    """Raised when Wannier bands do not capture the topological manifold."""


def assert_wannier_topology(
    tb_model,
    *,
    kz_probe: float = 0.5,
    n_kxy: int = 20,
    band_idx: int | None = None,
    min_chern: float = 0.5,
    material_class: str = "generic",
) -> None:
    """Topological sanity check after Wannierization.

    Verifies that the Wannier tight-binding model captures a non-trivial
    topological manifold by computing the Chern number at a probe k_z slice.

    Physics basis
    -------------
    For TaP/NbP Weyl semimetals: W1 nodes at k_z ≈ 0.42 (2π/c).
    A k_z probe at 0.5 lies between the W1 and W2 node pairs → C = ±2 for
    TaP (two chiral pairs on each side of probe slice).
    If C = 0 at the probe, the Wannier manifold captured a trivial set of
    bands (e.g. failed disentanglement extracted bulk bulk trivial states).

    For generic (non-Weyl) materials, this check is skipped.

    Parameters
    ----------
    tb_model : WannierTBModel
        Tight-binding model loaded from Wannier90 output.
    kz_probe : float
        k_z fractional coordinate to probe (default 0.5 = BZ midpoint).
    n_kxy : int
        k_x/k_y grid for Berry flux integration.
    band_idx : int or None
        Band index for Berry flux; None = auto (minimum-gap band).
    min_chern : float
        Minimum |C| to consider topological (default 0.5; use 1.5 for TaP).
    material_class : str
        Material class.  Only 'weyl' triggers hard assertion; others warn.

    Raises
    ------
    WannierTopologyError
        When material_class='weyl' and |C(kz_probe)| < min_chern.
    """
    mat = str(material_class).lower().strip()
    if mat not in {"weyl"}:
        return  # non-Weyl materials: skip hard check

    from wtec.topology.node_scan import _berry_plaquette_phase

    nkxy = max(4, int(n_kxy))
    dk = 1.0 / nkxy
    dk_vec1 = np.array([dk, 0.0, 0.0], dtype=float)
    dk_vec2 = np.array([0.0, dk, 0.0], dtype=float)

    total_flux = 0.0
    for ix in range(nkxy):
        for iy in range(nkxy):
            kxy = np.array([ix * dk, iy * dk, float(kz_probe)], dtype=float)
            if band_idx is None:
                import numpy as _np
                evals = _np.linalg.eigvalsh(tb_model.hamiltonian_at_k(kxy))
                bi = int(_np.argmin(_np.diff(evals)))
            else:
                bi = int(band_idx)
            total_flux += _berry_plaquette_phase(tb_model, kxy, dk_vec1, dk_vec2, bi)

    chern = total_flux / (2.0 * np.pi)
    if abs(chern) < float(min_chern):
        raise WannierTopologyError(
            f"Wannier topological sanity check FAILED: "
            f"|C(kz={kz_probe:.3f})| = {abs(chern):.3f} < {min_chern} "
            f"(n_kxy={nkxy}). "
            "The Wannier manifold likely captured a trivial band set. "
            "Remediation: (1) widen dis_win by ±1 eV to include Weyl-crossing bands; "
            "(2) ensure num_wann covers all SOC-split bands near E_F; "
            "(3) verify k-mesh z >= 6 for TaP (W1 at kz≈0.42·2π/c)."
        )


def assert_wannier_topology_from_files(
    *,
    hr_dat_path: str | Path,
    win_path: str | Path | None = None,
    kz_probe: float = 0.5,
    n_kxy: int = 20,
    band_idx: int | None = None,
    min_chern: float = 0.5,
    material_class: str = "generic",
) -> None:
    """Load a Wannier model from disk and run topological sanity checks.

    For non-Weyl materials this is a no-op by design.
    """
    mat = str(material_class).strip().lower()
    if mat != "weyl":
        return

    from wtec.wannier.model import WannierTBModel

    model = WannierTBModel.from_hr_dat(
        str(Path(hr_dat_path).expanduser().resolve()),
        win_path=str(Path(win_path).expanduser().resolve()) if win_path else None,
    )
    assert_wannier_topology(
        model,
        kz_probe=float(kz_probe),
        n_kxy=int(n_kxy),
        band_idx=band_idx,
        min_chern=float(min_chern),
        material_class=mat,
    )
