"""Lopez-Sancho iterative surface Green's function.

Physics basis
-------------
The retarded surface Green's function of a semi-infinite crystal is computed
via the iterative decimation scheme of Sancho, Sancho & Rubio (J. Phys. F:
Met. Phys. **15**, 851, 1985).

Starting from the bulk Bloch Hamiltonian H(k_∥) at fixed surface k-point,
the algorithm builds effective on-site and hopping matrices that converge
to the semi-infinite surface self-energy.

The surface spectral function is:
    A_surf(k_∥; E) = -(1/π) Im Tr G_00^R(E; k_∥)

where G_00^R is the (0,0) block of the surface Green's function matrix.

Advantages over finite-slab diagonalisation:
- No size-quantisation artefacts (true semi-infinite limit)
- No hybridization from the opposite surface
- Scales as O(N_orb³ × N_iter) vs O((N_orb × N_z)³) for slabs
- Converges in O(log N_z) iterations for exponentially localised states
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _build_surface_hk(
    hoppings: list[tuple[int, int, int, np.ndarray]],
    n_orb: int,
    kx: float,
    ky: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build H_00 and T_01 principal-layer blocks at (kx, ky).

    The surface normal is the z-axis.  For longer-range hopping along z we
    enlarge the principal layer so that only same-layer and nearest-layer
    couplings remain, which is the required input for Lopez-Sancho decimation.

    H(k_∥) in slab notation:

        H_IJ(k_∥) = Σ_{Rx, Ry} t(Rx, Ry, Rz=I-J) · exp(2πi(kx Rx + ky Ry))

    Returns H_00 (same principal layer) and T_01 (next principal layer).
    """
    two_pi = 2.0 * np.pi
    p_eff = 1
    for _, _, rz, _ in hoppings:
        p_eff = max(p_eff, abs(int(rz)))

    n_block = int(n_orb) * int(p_eff)
    H_00 = np.zeros((n_block, n_block), dtype=complex)
    T_01_forward = np.zeros((n_block, n_block), dtype=complex)
    T_01_backward = np.zeros((n_block, n_block), dtype=complex)

    for rx, ry, rz, mat in hoppings:
        phase = np.exp(1j * two_pi * (float(kx) * float(rx) + float(ky) * float(ry)))
        rz_int = int(rz)
        for iz in range(p_eff):
            jz = iz + rz_int
            row = slice(int(iz * n_orb), int((iz + 1) * n_orb))
            if 0 <= jz < p_eff:
                col = slice(int(jz * n_orb), int((jz + 1) * n_orb))
                H_00[row, col] += phase * mat
            elif p_eff <= jz < 2 * p_eff:
                col = slice(int((jz - p_eff) * n_orb), int((jz - p_eff + 1) * n_orb))
                T_01_forward[row, col] += phase * mat
            elif -p_eff <= jz < 0:
                col = slice(int((jz + p_eff) * n_orb), int((jz + p_eff + 1) * n_orb))
                T_01_backward[row, col] += phase * mat

    # Ensure on-site block is Hermitian
    H_00 = 0.5 * (H_00 + H_00.conj().T)
    if np.max(np.abs(T_01_forward)) > 0.0 and np.max(np.abs(T_01_backward)) > 0.0:
        T_01 = 0.5 * (T_01_forward + T_01_backward.conj().T)
    elif np.max(np.abs(T_01_forward)) > 0.0:
        T_01 = T_01_forward
    else:
        T_01 = T_01_backward.conj().T
    return H_00, T_01


def lopez_sancho_surface_gf(
    hoppings: list[tuple[int, int, int, np.ndarray]],
    n_orb: int,
    kx: float,
    ky: float,
    energy: float,
    eta: float = 0.06,
    max_iter: int = 200,
    conv_tol: float = 1e-8,
) -> tuple[np.ndarray, bool]:
    """Lopez-Sancho decimation for surface GF at (kx, ky, E+iη).

    Iterates:
        g_s     = [E + iη − ε_s]⁻¹
        t̃_s    = (I − g̃_s t̃† − g̃_s t̃)⁻¹ g̃_s t̃²
        t̃†_s   = (I − g̃_s t̃† − g̃_s t̃)⁻¹ g̃_s t̃†²
        ε_s     = ε_{s-1} + t̃†_{s-1} g̃_{s-1} t̃_{s-1} + t̃_{s-1} g̃_{s-1} t̃†_{s-1}
        ε_S     = ε_S,prev + t̃†_{s-1} g̃_{s-1} t̃_{s-1}   (surface only)

    Surface GF: G_00^R = [E + iη − ε_S]⁻¹

    Parameters
    ----------
    hoppings : list of (rx, ry, rz, mat)
    n_orb : int
    kx, ky : float  surface k-point (fractional)
    energy : float  energy (eV)
    eta : float     broadening (eV)
    max_iter : int
    conv_tol : float  convergence on |T|_∞

    Returns
    -------
    G_surf : (n_orb, n_orb) complex surface GF
    converged : bool
    """
    H_00, T_01 = _build_surface_hk(hoppings, n_orb, kx, ky)
    T_10 = T_01.conj().T

    z = complex(float(energy), float(eta))
    eye = np.eye(H_00.shape[0], dtype=complex)

    epsilon_s = H_00.copy()       # effective surface on-site
    epsilon_bulk = H_00.copy()    # effective bulk on-site
    t_forward = T_01.copy()       # t̃ (upward hopping)
    t_backward = T_10.copy()      # t̃† (downward hopping)

    converged = False
    for _ in range(max_iter):
        g_inv = z * eye - epsilon_bulk
        try:
            g = np.linalg.solve(g_inv, eye)
        except np.linalg.LinAlgError:
            break

        # Update effective hoppings
        tg = t_backward @ g @ t_forward
        gt = t_forward @ g @ t_backward

        epsilon_s = epsilon_s + tg
        epsilon_bulk = epsilon_bulk + tg + gt
        t_forward = t_forward @ g @ t_forward
        t_backward = t_backward @ g @ t_backward

        # Convergence: hopping amplitude decays
        if np.max(np.abs(t_forward)) < float(conv_tol):
            converged = True
            break

    # Surface GF: G_00^R = (z - ε_S)⁻¹
    surf_inv = z * eye - epsilon_s
    try:
        G_surf = np.linalg.solve(surf_inv, eye)
    except np.linalg.LinAlgError:
        G_surf = np.zeros((n_orb, n_orb), dtype=complex)
        converged = False

    return G_surf, converged


def surface_spectral_map_lopez_sancho(
    hoppings: list[tuple[int, int, int, np.ndarray]],
    n_orb: int,
    *,
    energy: float,
    eta: float,
    kx_coords: np.ndarray,
    ky_coords: np.ndarray,
    max_iter: int = 200,
    conv_tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute surface spectral function A(k_∥; E) over a 2D k-mesh.

    A_surf(kx, ky; E) = -(1/π) Im Tr G_00^R(kx, ky; E+iη)

    This is the semi-infinite surface limit — no size quantisation,
    no hybridization artefacts from opposite surface.

    Returns
    -------
    spectral_map : (len(kx_coords), len(ky_coords)) float array
    converged_map : (len(kx_coords), len(ky_coords)) bool array
    """
    nkx = len(kx_coords)
    nky = len(ky_coords)
    spectral_map = np.zeros((nkx, nky), dtype=float)
    converged_map = np.zeros((nkx, nky), dtype=bool)

    for ix, kx in enumerate(kx_coords):
        for iy, ky in enumerate(ky_coords):
            G, conv = lopez_sancho_surface_gf(
                hoppings, n_orb,
                float(kx), float(ky),
                energy=float(energy),
                eta=float(eta),
                max_iter=max_iter,
                conv_tol=conv_tol,
            )
            # A = -(1/π) Im Tr G
            spectral_map[ix, iy] = float(-np.trace(G[:n_orb, :n_orb]).imag / np.pi)
            converged_map[ix, iy] = bool(conv)

    # Clip to non-negative (numerical noise can give tiny negatives)
    spectral_map = np.maximum(spectral_map, 0.0)
    return spectral_map, converged_map


def compute_surface_spectral_metric_ls(
    tb_model,
    *,
    energy_ev: float = 0.0,
    n_kx: int = 16,
    n_ky: int = 16,
    broadening_ev: float = 0.06,
    max_iter: int = 200,
    conv_tol: float = 1e-8,
    kx_coords: np.ndarray | None = None,
    ky_coords: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute arc connectivity using Lopez-Sancho semi-infinite surface GF.

    This is a drop-in replacement for _tb_kresolved_surface_metric in arc_scan.py
    but avoids finite-size artefacts by using the true semi-infinite surface.

    Returns a dict compatible with compute_arc_connectivity output.
    """
    from wtec.topology.arc_scan import _collect_tb_hoppings, _metric_from_spectral_map

    parsed = _collect_tb_hoppings(tb_model)
    if parsed[0] is None:
        return {"status": "failed", "reason": parsed[1]}
    n_orb, hoppings = parsed  # type: ignore[misc]

    kx = (
        np.mod(np.asarray(kx_coords, dtype=float), 1.0)
        if kx_coords is not None
        else np.linspace(0, 1, n_kx, endpoint=False)
    )
    ky = (
        np.mod(np.asarray(ky_coords, dtype=float), 1.0)
        if ky_coords is not None
        else np.linspace(0, 1, n_ky, endpoint=False)
    )

    try:
        spectral_map, converged_map = surface_spectral_map_lopez_sancho(
            hoppings, int(n_orb),
            energy=float(energy_ev),
            eta=float(broadening_ev),
            kx_coords=kx,
            ky_coords=ky,
            max_iter=max_iter,
            conv_tol=conv_tol,
        )
    except Exception as exc:
        return {"status": "failed", "reason": f"lopez_sancho_failed:{type(exc).__name__}:{exc}"}

    if not np.any(spectral_map > 0):
        return {"status": "failed", "reason": "lopez_sancho_zero_spectral_weight"}

    try:
        stats = _metric_from_spectral_map(spectral_map)
    except Exception as exc:
        return {"status": "failed", "reason": f"metric_failed:{exc}"}

    convergence_fraction = float(np.mean(converged_map))

    return {
        "status": "ok",
        "metric": float(stats["metric"]),
        "surface_fraction": float(stats["metric"]),
        "engine": "lopez_sancho_surface_gf",
        "source_engine": "lopez_sancho_surface_gf",
        "source_kind": "semi_infinite_surface",
        "kmesh_xy": [int(len(kx)), int(len(ky))],
        "broadening_ev": float(broadening_ev),
        "threshold": float(stats["threshold"]),
        "largest_component_fraction": float(stats["largest_component_fraction"]),
        "largest_component_span": float(stats["largest_component_span"]),
        "convergence_fraction": convergence_fraction,
        "spectral_map": spectral_map,
    }
