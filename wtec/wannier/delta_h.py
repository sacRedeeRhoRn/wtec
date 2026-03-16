"""Delta-H transfer utilities for PES(anchor) -> LCAO(upscaled) workflows.

The implementation is intentionally conservative:
- strict basis compatibility checks
- onsite + first-shell correction only
- scalar alpha fit by grid-search in an E_F-centered window
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from wtec.wannier.parser import HoppingData, read_hr_dat, write_hr_dat


class DeltaHError(RuntimeError):
    """Raised when Delta-H transfer cannot be constructed/applied."""


def _parse_unit_cell_from_win(win_path: str | Path) -> np.ndarray | None:
    p = Path(win_path).expanduser().resolve()
    if not p.exists():
        return None
    text = p.read_text(errors="ignore")
    import re

    m = re.search(
        r"begin\s+unit_cell_cart\s*\n\s*(?:ang|bohr)?\s*\n(.*?)end\s+unit_cell_cart",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    rows = []
    for line in m.group(1).strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if len(rows) != 3:
        return None
    return np.asarray(rows, dtype=float)


def _projection_schema_from_win(win_path: str | Path) -> str | None:
    p = Path(win_path).expanduser().resolve()
    if not p.exists():
        return None
    text = p.read_text(errors="ignore")
    import re

    m = re.search(
        r"begin\s+projections\s*\n(.*?)end\s+projections",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    rows: list[str] = []
    for line in m.group(1).splitlines():
        raw = line.strip()
        if not raw:
            continue
        # Strip inline comments while preserving deterministic schema.
        raw = raw.split("!", 1)[0].split("#", 1)[0].strip()
        if not raw:
            continue
        rows.append(raw)
    if not rows:
        return None
    return "\n".join(rows)


def _projection_hash(schema: str | None) -> str | None:
    if schema is None:
        return None
    return hashlib.sha1(schema.encode("utf-8")).hexdigest()


def _norm_hoppings(hd: HoppingData) -> dict[tuple[int, int, int], np.ndarray]:
    out: dict[tuple[int, int, int], np.ndarray] = {}
    for i, r in enumerate(hd.r_vectors):
        key = (int(r[0]), int(r[1]), int(r[2]))
        d = int(hd.deg[i])
        if d <= 0:
            raise DeltaHError(f"Invalid degeneracy for R={key}: {d}")
        out[key] = np.asarray(hd.H_R[i], dtype=complex) / float(d)
    return out


def _hamiltonian_from_hops(
    hop_map: dict[tuple[int, int, int], np.ndarray],
    k_frac: np.ndarray,
    *,
    num_wann: int,
) -> np.ndarray:
    h = np.zeros((num_wann, num_wann), dtype=complex)
    for r, mat in hop_map.items():
        phase = np.exp(2j * np.pi * np.dot(k_frac, np.asarray(r, dtype=float)))
        h += phase * mat
    # Numerical hermitization for stability.
    return 0.5 * (h + h.conj().T)


def _fit_kgrid(kmesh: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = [max(1, int(v)) for v in kmesh]
    pts: list[list[float]] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pts.append([i / nx, j / ny, k / nz])
    return np.asarray(pts, dtype=float)


def _basis_meta(*, hd: HoppingData, win_path: str | Path | None) -> dict[str, Any]:
    schema = _projection_schema_from_win(win_path) if win_path else None
    return {
        "num_wann": int(hd.num_wann),
        "projection_schema": schema,
        "projection_hash": _projection_hash(schema),
    }


def assert_basis_compatible(
    *,
    source_meta: dict[str, Any],
    target_meta: dict[str, Any],
    basis_policy: str = "strict_same_basis",
) -> None:
    pol = str(basis_policy).strip().lower() or "strict_same_basis"
    if pol != "strict_same_basis":
        raise DeltaHError(f"Unsupported basis_policy={basis_policy!r}")
    if int(source_meta.get("num_wann", -1)) != int(target_meta.get("num_wann", -2)):
        raise DeltaHError(
            "Basis mismatch: num_wann differs "
            f"({source_meta.get('num_wann')} vs {target_meta.get('num_wann')})."
        )
    src_hash = source_meta.get("projection_hash")
    tgt_hash = target_meta.get("projection_hash")
    if src_hash and tgt_hash and str(src_hash) != str(tgt_hash):
        raise DeltaHError("Basis mismatch: projection schema hash differs.")


def _first_shell_r_vectors(
    r_vectors: np.ndarray,
    *,
    lattice_vectors: np.ndarray | None,
) -> list[tuple[int, int, int]]:
    nz: list[tuple[tuple[int, int, int], float]] = []
    for r in r_vectors:
        key = (int(r[0]), int(r[1]), int(r[2]))
        if key == (0, 0, 0):
            continue
        rv = np.asarray(key, dtype=float)
        if lattice_vectors is None:
            norm = float(np.linalg.norm(rv))
        else:
            rc = rv @ lattice_vectors
            norm = float(np.linalg.norm(rc))
        if norm <= 1e-14:
            continue
        nz.append((key, norm))
    if not nz:
        return []
    min_norm = min(v for _, v in nz)
    tol = max(1e-10, min_norm * 1e-6)
    return [k for k, v in nz if abs(v - min_norm) <= tol]


def _selected_r_vectors(
    *,
    hd_reference_lcao: HoppingData,
    lattice_vectors: np.ndarray | None,
    scope: str = "onsite_plus_first_shell",
) -> list[tuple[int, int, int]]:
    sc = str(scope).strip().lower() or "onsite_plus_first_shell"
    if sc != "onsite_plus_first_shell":
        raise DeltaHError(f"Unsupported delta_h scope={scope!r}")
    out = [(0, 0, 0)]
    out.extend(
        sorted(
            _first_shell_r_vectors(
                np.asarray(hd_reference_lcao.r_vectors, dtype=int),
                lattice_vectors=lattice_vectors,
            )
        )
    )
    return out


def _error_metrics(
    *,
    pes_hop: dict[tuple[int, int, int], np.ndarray],
    lcao_hop: dict[tuple[int, int, int], np.ndarray],
    delta_hop: dict[tuple[int, int, int], np.ndarray],
    alpha: float,
    num_wann: int,
    kpts: np.ndarray,
    fermi_pes_ev: float,
    fermi_lcao_ev: float,
    fit_window_ev: float,
) -> tuple[float, float, int]:
    diffs: list[float] = []
    max_abs = 0.0
    n_samples = 0
    for kf in kpts:
        h_pes = _hamiltonian_from_hops(pes_hop, kf, num_wann=num_wann)
        h_lc = _hamiltonian_from_hops(lcao_hop, kf, num_wann=num_wann)
        h_d = _hamiltonian_from_hops(delta_hop, kf, num_wann=num_wann)
        e_pes = np.linalg.eigvalsh(h_pes) - float(fermi_pes_ev)
        e_corr = np.linalg.eigvalsh(h_lc + float(alpha) * h_d) - float(fermi_lcao_ev)
        mask = (np.abs(e_pes) <= float(fit_window_ev)) | (np.abs(e_corr) <= float(fit_window_ev))
        idx = np.where(mask)[0]
        if idx.size == 0:
            idx = np.argsort(np.abs(e_pes))[: min(6, len(e_pes))]
        local = np.asarray(e_corr[idx] - e_pes[idx], dtype=float)
        if local.size:
            n_samples += int(local.size)
            max_abs = max(max_abs, float(np.max(np.abs(local))))
            diffs.extend(float(v) for v in local.tolist())
    if not diffs:
        return float("inf"), float("inf"), 0
    arr = np.asarray(diffs, dtype=float)
    rmse = float(np.sqrt(np.mean(arr**2)))
    return rmse, float(max_abs), int(n_samples)


def build_delta_h_artifact(
    *,
    pes_hr_dat_path: str | Path,
    pes_win_path: str | Path | None,
    lcao_hr_dat_path: str | Path,
    lcao_win_path: str | Path | None,
    material: str | None = None,
    fermi_pes_ev: float = 0.0,
    fermi_lcao_ev: float = 0.0,
    basis_policy: str = "strict_same_basis",
    scope: str = "onsite_plus_first_shell",
    fit_window_ev: float = 1.5,
    fit_kmesh: tuple[int, int, int] = (8, 8, 8),
    alpha_grid: np.ndarray | None = None,
    anchor_species_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    pes_hr = Path(pes_hr_dat_path).expanduser().resolve()
    lc_hr = Path(lcao_hr_dat_path).expanduser().resolve()
    if not pes_hr.exists():
        raise FileNotFoundError(f"PES hr.dat not found: {pes_hr}")
    if not lc_hr.exists():
        raise FileNotFoundError(f"LCAO hr.dat not found: {lc_hr}")

    hd_pes = read_hr_dat(pes_hr)
    hd_lc = read_hr_dat(lc_hr)
    pes_meta = _basis_meta(hd=hd_pes, win_path=pes_win_path)
    lc_meta = _basis_meta(hd=hd_lc, win_path=lcao_win_path)
    assert_basis_compatible(
        source_meta=pes_meta,
        target_meta=lc_meta,
        basis_policy=basis_policy,
    )

    lv = _parse_unit_cell_from_win(lcao_win_path) if lcao_win_path else None
    selected = _selected_r_vectors(
        hd_reference_lcao=hd_lc,
        lattice_vectors=lv,
        scope=scope,
    )
    if not selected:
        raise DeltaHError("No R-vectors selected for Delta-H transfer.")

    hop_pes = _norm_hoppings(hd_pes)
    hop_lc = _norm_hoppings(hd_lc)
    nw = int(hd_lc.num_wann)

    delta_hop: dict[tuple[int, int, int], np.ndarray] = {}
    for r in selected:
        mp = hop_pes.get(r)
        ml = hop_lc.get(r)
        if mp is None:
            mp = np.zeros((nw, nw), dtype=complex)
        if ml is None:
            ml = np.zeros((nw, nw), dtype=complex)
        delta_hop[r] = np.asarray(mp, dtype=complex) - np.asarray(ml, dtype=complex)

    kpts = _fit_kgrid(fit_kmesh)
    if alpha_grid is None:
        alpha_grid = np.linspace(-0.5, 1.5, 81)
    alpha_candidates = [float(a) for a in np.asarray(alpha_grid, dtype=float).tolist()]
    if not alpha_candidates:
        raise DeltaHError("alpha_grid must contain at least one candidate.")

    best_alpha = alpha_candidates[0]
    best_rmse = float("inf")
    best_max_abs = float("inf")
    best_n = 0
    for alpha in alpha_candidates:
        rmse, max_abs, n_s = _error_metrics(
            pes_hop=hop_pes,
            lcao_hop=hop_lc,
            delta_hop=delta_hop,
            alpha=alpha,
            num_wann=nw,
            kpts=kpts,
            fermi_pes_ev=fermi_pes_ev,
            fermi_lcao_ev=fermi_lcao_ev,
            fit_window_ev=fit_window_ev,
        )
        if rmse < best_rmse:
            best_alpha = alpha
            best_rmse = rmse
            best_max_abs = max_abs
            best_n = n_s

    rmse_pre, max_pre, _ = _error_metrics(
        pes_hop=hop_pes,
        lcao_hop=hop_lc,
        delta_hop=delta_hop,
        alpha=0.0,
        num_wann=nw,
        kpts=kpts,
        fermi_pes_ev=fermi_pes_ev,
        fermi_lcao_ev=fermi_lcao_ev,
        fit_window_ev=fit_window_ev,
    )

    mats_real = [np.asarray(delta_hop[r]).real.tolist() for r in selected]
    mats_imag = [np.asarray(delta_hop[r]).imag.tolist() for r in selected]
    artifact = {
        "version": 1,
        "mode": "delta_h",
        "scope": str(scope),
        "basis_policy": str(basis_policy),
        "material": material,
        "created_at_epoch": int(time.time()),
        "anchor": {
            "pes_hr_dat_path": str(pes_hr),
            "pes_win_path": str(Path(pes_win_path).expanduser().resolve()) if pes_win_path else None,
            "pes_fermi_ev": float(fermi_pes_ev),
            "lcao_hr_dat_path": str(lc_hr),
            "lcao_win_path": str(Path(lcao_win_path).expanduser().resolve()) if lcao_win_path else None,
            "lcao_fermi_ev": float(fermi_lcao_ev),
            "basis": {
                "num_wann": nw,
                "projection_schema": pes_meta.get("projection_schema"),
                "projection_hash": pes_meta.get("projection_hash"),
            },
            "transport_species_counts": (
                {str(k): int(v) for k, v in anchor_species_counts.items()}
                if isinstance(anchor_species_counts, dict)
                else None
            ),
        },
        "delta_h": {
            "r_vectors": [[int(v) for v in r] for r in selected],
            "mat_real": mats_real,
            "mat_imag": mats_imag,
        },
        "fit": {
            "fit_window_ev": float(fit_window_ev),
            "fit_kmesh": [int(fit_kmesh[0]), int(fit_kmesh[1]), int(fit_kmesh[2])],
            "alpha": float(best_alpha),
            "rmse_ev_pre": float(rmse_pre),
            "max_abs_ev_pre": float(max_pre),
            "rmse_ev_post": float(best_rmse),
            "max_abs_ev_post": float(best_max_abs),
            "n_samples": int(best_n),
        },
    }
    return artifact


def write_delta_h_artifact(path: str | Path, artifact: dict[str, Any]) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(artifact, indent=2))
    return p


def load_delta_h_artifact(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Delta-H artifact not found: {p}")
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise DeltaHError("Delta-H artifact must be a JSON object.")
    return data


def apply_delta_h_to_hr_file(
    *,
    hr_dat_path: str | Path,
    output_hr_dat_path: str | Path,
    artifact: dict[str, Any],
    win_path: str | Path | None = None,
) -> dict[str, Any]:
    hd = read_hr_dat(hr_dat_path)
    target_meta = _basis_meta(hd=hd, win_path=win_path)
    anchor_basis = artifact.get("anchor", {}).get("basis", {})
    assert_basis_compatible(
        source_meta={
            "num_wann": int(anchor_basis.get("num_wann", -1)),
            "projection_schema": anchor_basis.get("projection_schema"),
            "projection_hash": anchor_basis.get("projection_hash"),
        },
        target_meta=target_meta,
        basis_policy=str(artifact.get("basis_policy", "strict_same_basis")),
    )

    delta = artifact.get("delta_h", {})
    r_vectors_raw = delta.get("r_vectors", [])
    mats_real = delta.get("mat_real", [])
    mats_imag = delta.get("mat_imag", [])
    if not isinstance(r_vectors_raw, list) or not isinstance(mats_real, list) or not isinstance(mats_imag, list):
        raise DeltaHError("Invalid delta_h payload in artifact.")
    if not (len(r_vectors_raw) == len(mats_real) == len(mats_imag)):
        raise DeltaHError("delta_h arrays must have equal length.")

    alpha = float(artifact.get("fit", {}).get("alpha", 1.0))
    hop_target = _norm_hoppings(hd)
    nw = int(hd.num_wann)

    for r, mr, mi in zip(r_vectors_raw, mats_real, mats_imag):
        if not isinstance(r, list) or len(r) != 3:
            raise DeltaHError(f"Invalid r-vector in delta_h: {r!r}")
        key = (int(r[0]), int(r[1]), int(r[2]))
        if key not in hop_target:
            raise DeltaHError(
                "Target HR missing required Delta-H shell "
                f"R={key}. Basis/pseudization is not compatible."
            )
        dmat = np.asarray(mr, dtype=float) + 1j * np.asarray(mi, dtype=float)
        if dmat.shape != (nw, nw):
            raise DeltaHError(f"Delta-H matrix shape mismatch for R={key}: {dmat.shape} vs {(nw, nw)}")
        hop_target[key] = hop_target[key] + alpha * dmat
        hop_target[key] = 0.5 * (hop_target[key] + hop_target[key].conj().T)

    # Rebuild raw H_R with original R ordering + degeneracy.
    out_H = np.asarray(hd.H_R, dtype=complex).copy()
    for i, r in enumerate(np.asarray(hd.r_vectors, dtype=int)):
        key = (int(r[0]), int(r[1]), int(r[2]))
        d = float(int(hd.deg[i]))
        out_H[i] = hop_target[key] * d
        out_H[i] = 0.5 * (out_H[i] + out_H[i].conj().T)

    out_hd = HoppingData(
        num_wann=int(hd.num_wann),
        r_vectors=np.asarray(hd.r_vectors, dtype=int),
        deg=np.asarray(hd.deg, dtype=int),
        H_R=out_H,
    )
    write_hr_dat(output_hr_dat_path, out_hd, header="Written by wtec.wannier.delta_h.apply_delta_h_to_hr_file")
    return {
        "status": "ok",
        "alpha": alpha,
        "output_hr_dat_path": str(Path(output_hr_dat_path).expanduser().resolve()),
    }
