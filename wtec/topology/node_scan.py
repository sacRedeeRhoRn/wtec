"""Weyl-node scan using coarse+refine strategy."""

from __future__ import annotations

from typing import Any

import numpy as np


_SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
_SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _wrap_k(k: np.ndarray) -> np.ndarray:
    return np.mod(k, 1.0)


def _periodic_dist(a: np.ndarray, b: np.ndarray) -> float:
    d = np.abs(a - b)
    d = np.minimum(d, 1.0 - d)
    return float(np.linalg.norm(d))


def _gap_and_index(tb_model, k: np.ndarray) -> tuple[float, int, np.ndarray]:
    evals = np.linalg.eigvalsh(tb_model.hamiltonian_at_k(k))
    gaps = np.diff(evals)
    idx = int(np.argmin(gaps))
    return float(gaps[idx]), idx, evals


def _coordinate_refine(
    tb_model,
    k0: np.ndarray,
    *,
    coarse_kmesh: tuple[int, int, int],
    max_iter: int = 50,
) -> tuple[np.ndarray, float, int, np.ndarray]:
    step = np.array([1.0 / coarse_kmesh[0], 1.0 / coarse_kmesh[1], 1.0 / coarse_kmesh[2]], dtype=float) * 0.25
    k = _wrap_k(k0.copy())
    best_gap, best_idx, best_evals = _gap_and_index(tb_model, k)

    for _ in range(max(1, int(max_iter))):
        improved = False
        for ax in (0, 1, 2):
            for sign in (-1.0, 1.0):
                trial = k.copy()
                trial[ax] += sign * step[ax]
                trial = _wrap_k(trial)
                g, gi, ge = _gap_and_index(tb_model, trial)
                if g < best_gap:
                    best_gap, best_idx, best_evals = g, gi, ge
                    k = trial
                    improved = True
        if not improved:
            step *= 0.5
            if float(np.linalg.norm(step)) < 1e-4:
                break
    return k, best_gap, best_idx, best_evals


def _dH_dk_frac(tb_model, k: np.ndarray, axis: int, delta: float = 1e-4) -> np.ndarray:
    kp = k.copy()
    km = k.copy()
    kp[axis] += delta
    km[axis] -= delta
    kp = _wrap_k(kp)
    km = _wrap_k(km)
    hp = np.array(tb_model.hamiltonian_at_k(kp), dtype=complex)
    hm = np.array(tb_model.hamiltonian_at_k(km), dtype=complex)
    return (hp - hm) / (2.0 * delta)


def _chirality_proxy(tb_model, k: np.ndarray, band_idx: int) -> tuple[int | None, float]:
    """Estimate chirality sign from 2-band projected velocity Jacobian."""
    h = np.array(tb_model.hamiltonian_at_k(k), dtype=complex)
    evals, evecs = np.linalg.eigh(h)
    n = int(band_idx)
    if n < 0 or n + 1 >= len(evals):
        return None, 0.0
    u = evecs[:, [n, n + 1]]

    rows: list[list[float]] = []
    for ax in (0, 1, 2):
        dH = _dH_dk_frac(tb_model, k, ax)
        A = u.conj().T @ dH @ u
        vx = float(0.5 * np.trace(A @ _SIGMA_X).real)
        vy = float(0.5 * np.trace(A @ _SIGMA_Y).real)
        vz = float(0.5 * np.trace(A @ _SIGMA_Z).real)
        rows.append([vx, vy, vz])
    J = np.array(rows, dtype=float)
    det = float(np.linalg.det(J))
    if abs(det) < 1e-12:
        return 0, det
    return int(np.sign(det)), det


def _normalized_overlap(u: np.ndarray, v: np.ndarray) -> complex:
    ov = np.vdot(u, v)
    mag = abs(ov)
    if mag < 1e-15:
        return 1.0 + 0.0j
    return ov / mag


def _berry_plaquette_phase(tb_model, k0: np.ndarray, dk1: np.ndarray, dk2: np.ndarray, band_idx: int) -> float:
    # Oriented loop: k00 -> k10 -> k11 -> k01 -> k00
    k00 = _wrap_k(k0)
    k10 = _wrap_k(k0 + dk1)
    k11 = _wrap_k(k0 + dk1 + dk2)
    k01 = _wrap_k(k0 + dk2)

    def _u(kp: np.ndarray) -> np.ndarray:
        _, vecs = np.linalg.eigh(np.array(tb_model.hamiltonian_at_k(kp), dtype=complex))
        return vecs[:, int(band_idx)]

    u00 = _u(k00)
    u10 = _u(k10)
    u11 = _u(k11)
    u01 = _u(k01)
    phase = (
        _normalized_overlap(u00, u10)
        * _normalized_overlap(u10, u11)
        * _normalized_overlap(u11, u01)
        * _normalized_overlap(u01, u00)
    )
    return float(np.angle(phase))


def _chirality_flux(tb_model, k: np.ndarray, band_idx: int, *, step: float) -> tuple[int | None, float]:
    """Estimate chirality from discretized Berry flux on a small cube."""
    h = float(max(1e-4, step))
    total_flux = 0.0

    for axis in (0, 1, 2):
        axes = [0, 1, 2]
        axes.remove(axis)
        a1, a2 = axes

        k_face = np.array(k, dtype=float)
        k_face[axis] += h
        k_face[a1] -= h
        k_face[a2] -= h
        dk1 = np.zeros(3, dtype=float)
        dk2 = np.zeros(3, dtype=float)
        dk1[a1] = 2.0 * h
        dk2[a2] = 2.0 * h
        total_flux += _berry_plaquette_phase(tb_model, k_face, dk1, dk2, band_idx)

        k_face = np.array(k, dtype=float)
        k_face[axis] -= h
        k_face[a1] -= h
        k_face[a2] -= h
        # Reverse orientation for the outward normal on the negative face.
        total_flux += _berry_plaquette_phase(tb_model, k_face, dk2, dk1, band_idx)

    charge_float = total_flux / (2.0 * np.pi)
    if not np.isfinite(charge_float):
        return None, float("nan")
    charge = int(np.round(charge_float))
    if abs(charge_float) < 0.25:
        charge = 0
    return charge, float(total_flux)


def scan_weyl_nodes(
    tb_model,
    *,
    coarse_kmesh: tuple[int, int, int] = (20, 20, 20),
    refine_kmesh: tuple[int, int, int] = (5, 5, 5),
    gap_threshold_ev: float = 0.05,
    max_candidates: int = 64,
    dedup_tol: float = 0.04,
    fermi_ev: float = 0.0,
    newton_max_iter: int = 50,
    node_method: str = "proxy",
    comm=None,
) -> dict[str, Any]:
    """Locate approximate Weyl nodes and estimate chirality."""
    rank = 0
    size = 1
    if comm is not None:
        rank = int(comm.Get_rank())
        size = int(comm.Get_size())

    method = str(node_method).strip().lower()
    if method not in {"proxy", "berry_flux", "wannierberri_flux"}:
        return {
            "status": "failed",
            "reason": f"unknown_node_method:{node_method!r}",
            "nodes": [],
            "n_nodes": 0,
        }

    method_requested = method
    method_effective = method
    method_warning: str | None = None

    if method == "wannierberri_flux":
        try:
            import wannierberri as _wb  # noqa: F401
        except Exception as exc:
            return {
                "status": "failed",
                "reason": f"wannierberri_missing:{exc}",
                "nodes": [],
                "n_nodes": 0,
                "node_method_requested": method_requested,
                "node_method_effective": None,
            }
        # Current production path uses the internal flux-cube integration while
        # preserving strict dependency checks for WannierBerri-enabled runs.
        method_effective = "berry_flux"
        method_warning = (
            "wannierberri_flux_requested_using_internal_berry_flux_cube:"
            "direct_wannierberri_chirality_backend_pending"
        )

    n1, n2, n3 = [max(2, int(v)) for v in coarse_kmesh]
    coarse_local: list[tuple[float, np.ndarray, int, np.ndarray]] = []

    total = n1 * n2 * n3
    for linear in range(rank, total, size):
        i = linear // (n2 * n3)
        rem = linear % (n2 * n3)
        j = rem // n3
        k = rem % n3
        kf = np.array([i / n1, j / n2, k / n3], dtype=float)
        gap, idx, evals = _gap_and_index(tb_model, kf)
        coarse_local.append((gap, kf, idx, evals))

    if comm is not None:
        gathered = comm.gather(coarse_local, root=0)
        if rank == 0:
            coarse: list[tuple[float, np.ndarray, int, np.ndarray]] = []
            for chunk in gathered:
                coarse.extend(chunk)
        else:
            coarse = []
    else:
        coarse = coarse_local

    if rank == 0:
        coarse.sort(key=lambda x: x[0])
        selected = [c for c in coarse if c[0] <= float(gap_threshold_ev)]
        if len(selected) < max(4, int(max_candidates * 0.25)):
            selected = coarse[: int(max_candidates)]
        else:
            selected = selected[: int(max_candidates)]
    else:
        selected = []

    if comm is not None:
        selected = comm.bcast(selected, root=0)

    nodes_local: list[dict[str, Any]] = []
    for sel_idx, (gap0, k0, idx0, _) in enumerate(selected):
        if sel_idx % size != rank:
            continue
        kf, gap, band_idx, evals = _coordinate_refine(
            tb_model,
            k0,
            coarse_kmesh=(n1, n2, n3),
            max_iter=newton_max_iter,
        )

        # Lightweight local subgrid polish around refined coordinate.
        rk = [max(3, int(v)) for v in refine_kmesh]
        span = np.array([0.5 / n1, 0.5 / n2, 0.5 / n3], dtype=float)
        best = (gap, kf.copy(), band_idx, evals.copy())
        for a in range(rk[0]):
            for b in range(rk[1]):
                for c in range(rk[2]):
                    off = np.array([
                        (a / (rk[0] - 1) - 0.5) * 2.0 * span[0],
                        (b / (rk[1] - 1) - 0.5) * 2.0 * span[1],
                        (c / (rk[2] - 1) - 0.5) * 2.0 * span[2],
                    ])
                    kt = _wrap_k(kf + off)
                    gt, bi, ev = _gap_and_index(tb_model, kt)
                    if gt < best[0]:
                        best = (gt, kt, bi, ev)
        gap, kf, band_idx, evals = best
        if method_effective == "berry_flux":
            flux_step = 0.5 * min(1.0 / n1, 1.0 / n2, 1.0 / n3)
            ch, det = _chirality_flux(tb_model, kf, band_idx, step=flux_step)
            if method_requested == "wannierberri_flux":
                method_tag = "coarse_refine_berry_flux_cube_internal_fallback"
            else:
                method_tag = "coarse_refine_berry_flux_cube"
        else:
            ch, det = _chirality_proxy(tb_model, kf, band_idx)
            method_tag = "coarse_refine_kdotp_proxy"
        e_mid = 0.5 * (float(evals[band_idx]) + float(evals[band_idx + 1])) - float(fermi_ev)
        nodes_local.append(
            {
                "k_frac": [float(kf[0]), float(kf[1]), float(kf[2])],
                "gap_ev": float(gap),
                "band_idx": int(band_idx),
                "energy_rel_fermi_ev": float(e_mid),
                "chirality": ch,
                "chirality_det_proxy": float(det),
                "method": method_tag,
                "coarse_gap_ev": float(gap0),
            }
        )

    if comm is not None:
        gathered_nodes = comm.gather(nodes_local, root=0)
        if rank == 0:
            nodes: list[dict[str, Any]] = []
            for chunk in gathered_nodes:
                nodes.extend(chunk)
        else:
            nodes = []
    else:
        nodes = nodes_local

    out: dict[str, Any] | None = None
    if rank == 0:
        # Deduplicate near-identical candidates.
        deduped: list[dict[str, Any]] = []
        for node in sorted(nodes, key=lambda x: x["gap_ev"]):
            kf = np.array(node["k_frac"], dtype=float)
            if any(_periodic_dist(kf, np.array(n["k_frac"], dtype=float)) < dedup_tol for n in deduped):
                continue
            deduped.append(node)

        status = "ok" if deduped else "partial"
        out = {
            "status": status,
            "coarse_kmesh": [n1, n2, n3],
            "refine_kmesh": [int(v) for v in refine_kmesh],
            "gap_threshold_ev": float(gap_threshold_ev),
            "nodes": deduped,
            "n_nodes": int(len(deduped)),
            "node_method": method_requested,
            "node_method_requested": method_requested,
            "node_method_effective": method_effective,
            "node_method_warning": method_warning,
        }
    if comm is not None:
        out = comm.bcast(out, root=0)
    assert out is not None
    return out
