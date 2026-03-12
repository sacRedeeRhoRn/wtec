"""Wilson loop / Wannier charge centre Chern number computation.

Physics basis
-------------
The Chern number of a set of occupied bands on a 2D torus (e.g. a k_z slice
of the 3D Brillouin zone) can be computed from the winding of Wannier charge
centres (WCCs) — the eigenphases of the Wilson loop operator.

Wilson loop at fixed k_z, integrating along k_x:

    W(k_z) = P exp[i ∮_{k_x=0→1} A(k_x, k_z) dk_x]

Discretized as a product of overlap matrices:

    W(k_z) = M(k₀, k₁) · M(k₁, k₂) · ... · M(k_{N-1}, k₀)

    M_{nm}(k, k') = ⟨u_n(k')| u_m(k)⟩

The WCCs are the eigenphases of W:

    W |w_n⟩ = e^{iθ_n} |w_n⟩,   θ_n ∈ (−π, π]

The Chern number is the winding number of {θ_n(k_z)} as k_z traverses the BZ:

    C = (1/2π) Δθ_total   (total winding)

This method is numerically stable and avoids the gauge-choice problems of
direct Berry curvature integration.

Reference: Soluyanov & Vanderbilt, PRB 83, 235401 (2011).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _overlap_matrix(evecs1: np.ndarray, evecs2: np.ndarray) -> np.ndarray:
    """Overlap matrix M_{nm} = ⟨u_n(k2)|u_m(k1)⟩.

    Parameters
    ----------
    evecs1 : (n_states, n_bands) — columns are |u_m(k1)⟩
    evecs2 : (n_states, n_bands) — columns are |u_n(k2)⟩

    Returns M (n_bands × n_bands).
    """
    # M_{nm} = u_n(k2)† · u_m(k1)
    return evecs2.conj().T @ evecs1


def _wilson_loop_matrix(
    tb_model,
    ky: float,
    kz: float,
    n_kx: int,
    band_indices: list[int],
) -> np.ndarray:
    """Compute the Wilson loop matrix W(k_y, k_z) along k_x.

    W = M(k_{N-1}, k_0) · ... · M(k_1, k_2) · M(k_0, k_1)

    with M_{nm}(k, k') = ⟨u_n(k')| u_m(k)⟩.

    Parameters
    ----------
    tb_model : WannierTBModel
    ky : float  fixed k_y (fractional)
    kz : float  fixed k_z (fractional)
    n_kx : int  number of k_x points (closed loop: k_N = k_0)
    band_indices : list[int]  indices of bands to include

    Returns
    -------
    W : (n_bands, n_bands) complex unitary matrix
    """
    n_bands = len(band_indices)
    kx_vals = np.linspace(0.0, 1.0, n_kx, endpoint=False)

    def _get_evecs(kx: float) -> np.ndarray:
        H = tb_model.hamiltonian_at_k(np.array([float(kx), float(ky), float(kz)]))
        _, vecs = np.linalg.eigh(H)
        return vecs[:, band_indices]  # (n_orb, n_bands)

    # Build evecs along the loop
    evec_list = [_get_evecs(kx) for kx in kx_vals]

    # W = product of overlaps M(k_i, k_{i+1}) for i = 0 ... N-1
    # where k_N = k_0 (periodic boundary)
    W = np.eye(n_bands, dtype=complex)
    for i in range(n_kx):
        u_curr = evec_list[i]
        u_next = evec_list[(i + 1) % n_kx]
        M = _overlap_matrix(u_curr, u_next)
        W = M @ W

    return W


def _wcc_from_wilson_loop(W: np.ndarray) -> np.ndarray:
    """Extract Wannier charge centres (phases θ ∈ (−π, π]) from Wilson loop."""
    eigvals = np.linalg.eigvals(W)
    phases = np.angle(eigvals)
    return np.sort(phases)


def _chern_from_wcc_winding(
    wcc_array: np.ndarray,
    param_vals: np.ndarray,
) -> float:
    """Compute total WCC winding over a periodic parameter cycle.

    wcc_array : (n_param, n_bands) phases at each parameter value
    param_vals : (n_param,)
    """
    n_param, n_bands = wcc_array.shape
    if n_bands == 0:
        return 0.0
    if n_param == 0:
        return 0.0

    del param_vals  # the winding is purely topological once the cycle is sampled
    wcc_smooth = wcc_array.copy()
    for iz in range(1, n_param):
        for ib in range(n_bands):
            diff = wcc_smooth[iz, ib] - wcc_smooth[iz - 1, ib]
            wcc_smooth[iz, ib] -= 2.0 * np.pi * np.round(diff / (2.0 * np.pi))

    total_winding = 0.0
    for ib in range(n_bands):
        closure = wcc_smooth[0, ib] - wcc_smooth[-1, ib]
        closure -= 2.0 * np.pi * np.round(closure / (2.0 * np.pi))
        total_winding += (wcc_smooth[-1, ib] - wcc_smooth[0, ib]) + closure
    return float(total_winding / (2.0 * np.pi))


def _phase_winding(phases: np.ndarray) -> float:
    """Return the winding of a periodic Wilson-loop phase over one full cycle."""
    phase_arr = np.asarray(phases, dtype=float).reshape(-1)
    if phase_arr.size == 0:
        return 0.0
    unwrapped = np.zeros_like(phase_arr)
    unwrapped[0] = phase_arr[0]
    for i in range(1, phase_arr.size):
        diff = phase_arr[i] - phase_arr[i - 1]
        unwrapped[i] = unwrapped[i - 1] + diff - 2.0 * np.pi * np.round(diff / (2.0 * np.pi))
    closure = phase_arr[0] - phase_arr[-1]
    closure -= 2.0 * np.pi * np.round(closure / (2.0 * np.pi))
    total_phase = (unwrapped[-1] - unwrapped[0]) + closure
    return float(total_phase / (2.0 * np.pi))


def compute_wilson_loop_chern(
    tb_model,
    *,
    n_kz: int = 40,
    n_kx: int = 30,
    n_ky: int | None = None,
    band_idx: int | None = None,
    n_occ_bands: int | None = None,
    fermi_ev: float = 0.0,
) -> dict[str, Any]:
    """Compute the Wilson-loop Chern profile C(k_z) over fixed-k_z slices.

    For each fixed-k_z slice, computes W(k_y, k_z) by integrating along k_x
    for a mesh of k_y values. The slice Chern number is the winding of the
    Wilson-loop phase det W as k_y traverses the full periodic cycle.

    Parameters
    ----------
    tb_model : WannierTBModel
    n_kz : int  k_z grid points
    n_kx : int  k_x integration points per Wilson loop
    n_ky : int or None  k_y points per fixed-k_z slice (defaults to n_kx)
    band_idx : int or None
        Index of the single band to include.  None = use n_occ_bands.
    n_occ_bands : int or None
        Number of occupied bands (lowest-energy) to include.
        None = auto-detect from half-filling.
    fermi_ev : float
        Fermi level for automatic band selection.

    Returns
    -------
    dict with keys:
        kz_frac : list[float]
        ky_frac : list[float]
        wcc : list[list[list[float]]] WCCs at each (k_z, k_y)
        chern_integrated : float      dominant Wilson-loop slice Chern number
        chern_profile : list[float]   Wilson-loop C(k_z)
        chern_profile_berry : list[float]  Berry-plaquette comparison profile
        jump_kz : list[float]         k_z where WCC winding changes
        topological_sanity : bool
        n_kz, n_kx, n_ky : int
        status : str
    """
    kz_vals = np.linspace(0.0, 1.0, n_kz, endpoint=False)
    ky_vals = np.linspace(0.0, 1.0, int(n_ky if n_ky is not None else n_kx), endpoint=False)

    # Determine band selection
    H0 = tb_model.hamiltonian_at_k(np.zeros(3))
    evals0 = np.linalg.eigvalsh(H0)
    n_orb = len(evals0)

    if band_idx is not None:
        band_indices = [int(band_idx)]
    elif n_occ_bands is not None:
        band_indices = list(range(max(0, int(n_occ_bands))))
    else:
        # Auto: count bands below fermi_ev at Gamma
        n_occ = int(np.sum(evals0 < float(fermi_ev)))
        if n_occ == 0:
            n_occ = max(1, n_orb // 2)
        band_indices = list(range(n_occ))

    if not band_indices:
        return {
            "status": "failed",
            "reason": "no_bands_selected",
            "kz_frac": [],
            "ky_frac": [],
            "wcc": [],
            "chern_integrated": 0.0,
            "topological_sanity": False,
        }

    wcc_list: list[list[list[float]]] = []
    chern_profile_wilson: list[float] = []
    try:
        for kz in kz_vals:
            slice_wcc: list[list[float]] = []
            slice_wilson_phase: list[float] = []
            for ky in ky_vals:
                W = _wilson_loop_matrix(
                    tb_model,
                    float(ky),
                    float(kz),
                    n_kx=n_kx,
                    band_indices=band_indices,
                )
                phases = _wcc_from_wilson_loop(W).tolist()
                slice_wcc.append(phases)
                slice_wilson_phase.append(float(np.angle(np.linalg.det(W))))
            wcc_list.append(slice_wcc)
            chern_profile_wilson.append(_phase_winding(np.asarray(slice_wilson_phase, dtype=float)))
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"wilson_loop_eigh_failed:{type(exc).__name__}:{exc}",
            "kz_frac": kz_vals.tolist(),
            "ky_frac": ky_vals.tolist(),
            "wcc": wcc_list,
            "chern_integrated": 0.0,
            "topological_sanity": False,
        }

    chern_profile_arr = np.asarray(chern_profile_wilson, dtype=float)
    if chern_profile_arr.size:
        dominant_idx = int(np.argmax(np.abs(chern_profile_arr)))
        chern_integrated = float(chern_profile_arr[dominant_idx])
    else:
        chern_integrated = 0.0

    # Detect k_z positions where the slice Chern changes (= Weyl node projections).
    jump_kz: list[float] = []
    jump_chirality: list[int] = []
    for iz in range(n_kz):
        next_iz = (iz + 1) % n_kz
        delta_c = chern_profile_arr[next_iz] - chern_profile_arr[iz]
        if abs(delta_c) < 0.5:
            continue
        kz_next = float(kz_vals[next_iz]) if next_iz != 0 else 1.0
        kz_mid = 0.5 * (float(kz_vals[iz]) + kz_next)
        if kz_mid >= 1.0:
            kz_mid -= 1.0
        jump_kz.append(float(kz_mid))
        jump_chirality.append(int(np.sign(delta_c)))

    # Also compute Berry-plaquette C(k_z) for comparison
    from wtec.topology.node_scan import compute_chern_profile
    try:
        chern_bp = compute_chern_profile(
            tb_model,
            n_kz=n_kz,
            n_kxy=max(10, n_kx // 2),
            band_idx=band_indices[-1] if band_indices else None,
        )
        chern_profile_berry = chern_bp.get("chern", [])
    except Exception:
        chern_profile_berry = []

    topological = float(np.max(np.abs(chern_profile_arr))) >= 0.5 or bool(jump_kz)

    return {
        "status": "ok" if topological else "trivial_or_unconverged",
        "kz_frac": kz_vals.tolist(),
        "ky_frac": ky_vals.tolist(),
        "wcc": wcc_list,
        "band_indices": band_indices,
        "chern_integrated": float(chern_integrated),
        "chern_profile": chern_profile_wilson,
        "chern_profile_berry": chern_profile_berry,
        "jump_kz": sorted(jump_kz),
        "jump_chirality": [jump_chirality[i] for i in np.argsort(jump_kz).tolist()],
        "topological_sanity": bool(topological),
        "n_kz": int(n_kz),
        "n_kx": int(n_kx),
        "n_ky": int(len(ky_vals)),
    }
