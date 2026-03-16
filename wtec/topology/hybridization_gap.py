"""Hybridization gap Δ(d) for topological semimetal thin films.

Physics basis
-------------
When a topological semimetal film is thinned below ~10 unit cells, the Fermi-arc
surface states on the top and bottom surfaces begin to overlap through the bulk
and develop a hybridization gap:

    Δ(d) = Δ₀ · exp(−d / λ_arc)

where λ_arc = ℏ v_⊥ / Δ_bulk is the arc penetration depth and d is the film
thickness in unit cells.  This gap suppresses arc conductance:

    G_arc(d, T) ∝ sech²(Δ(d) / 2kT)

and creates a non-monotonic ρ(d) curve with a resistivity maximum near
d_c = λ_arc · ln(Δ₀ / 2kT).

This module:
- Finds the hybridization gap directly from finite-slab eigenstates (no
  phenomenological fit needed as input)
- Fits the exponential law and extracts Δ₀, λ_arc
- Predicts d_c at a given temperature
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _surface_projected_states(
    tb_model,
    thickness_uc: int,
    kpar: tuple[float, float],
    n_orb: int,
    hoppings: list[tuple[int, int, int, np.ndarray]],
    n_surface_layers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build slab H at k_∥ = kpar and return eigenvalues + surface weights.

    Surface weight of state |ψ_n⟩:
        w_surf(n) = Σ_{i ∈ surface layers} |⟨i|ψ_n⟩|²

    Parameters
    ----------
    tb_model : WannierTBModel
    thickness_uc : int
        Film thickness in unit cells (n_z).
    kpar : (kx, ky) fractional surface k-point.
    n_orb : int
    hoppings : list of (rx, ry, rz, mat)
    n_surface_layers : int
        Number of unit cells on each surface counted as surface.

    Returns
    -------
    evals : (n_states,) eigenvalues in eV
    surf_w : (n_states,) surface weights ∈ [0, 1]
    """
    nz = max(2, int(thickness_uc))
    n_surf = max(1, int(n_surface_layers))
    ndof = nz * n_orb

    # Surface site indices: first and last n_surf unit cells
    surf_idx = np.concatenate([
        np.arange(0, n_surf * n_orb, dtype=int),
        np.arange((nz - n_surf) * n_orb, nz * n_orb, dtype=int),
    ])

    kx, ky = float(kpar[0]), float(kpar[1])
    two_pi = 2.0 * np.pi

    H = np.zeros((ndof, ndof), dtype=complex)
    for rx, ry, rz, mat in hoppings:
        phase = np.exp(1j * two_pi * (kx * float(rx) + ky * float(ry)))
        for iz in range(nz):
            jz = iz + int(rz)
            if jz < 0 or jz >= nz:
                continue
            i0 = iz * n_orb
            j0 = jz * n_orb
            H[i0:i0 + n_orb, j0:j0 + n_orb] += phase * mat

    H = 0.5 * (H + H.conj().T)
    evals, evecs = np.linalg.eigh(H)

    # Surface weight = probability on surface unit cells
    surf_w = np.sum(np.abs(evecs[surf_idx, :]) ** 2, axis=0)
    return evals, surf_w


def _find_arc_pair_gap(
    evals: np.ndarray,
    surf_w: np.ndarray,
    fermi_ev: float,
    surface_weight_threshold: float = 0.30,
    n_candidates: int = 4,
) -> float | None:
    """Find the smallest gap between surface-weighted states near E_F.

    The hybridization gap is the energy splitting between the two arc
    states (one from each surface) that are most surface-localised and
    nearest to the Fermi level.

    Returns the gap in eV, or None if no arc candidates found.
    """
    # Select highly surface-localised states
    mask = surf_w > surface_weight_threshold
    if not np.any(mask):
        return None

    e_surf = evals[mask]
    # Find n_candidates states closest to E_F
    order = np.argsort(np.abs(e_surf - fermi_ev))
    e_near = e_surf[order[:min(n_candidates, len(e_surf))]]
    if len(e_near) < 2:
        return None

    # The gap is the smallest positive energy difference between the
    # highest occupied and lowest unoccupied surface state near E_F
    e_below = e_near[e_near <= fermi_ev]
    e_above = e_near[e_near > fermi_ev]
    if e_below.size == 0 or e_above.size == 0:
        # All states on same side of E_F — use min pairwise splitting
        return float(np.min(np.diff(np.sort(e_near))))

    return float(np.min(e_above) - np.max(e_below))


def compute_hybridization_gap(
    tb_model,
    thickness_range_uc: list[int],
    *,
    kpar_node: tuple[float, float] = (0.5, 0.5),
    n_kpar: int = 5,
    kpar_radius: float = 0.04,
    fermi_ev: float = 0.0,
    n_surface_layers: int = 1,
    surface_weight_threshold: float = 0.25,
) -> dict[str, Any]:
    """Compute hybridization gap Δ(d) for a range of film thicknesses.

    For each thickness d_uc, the slab Hamiltonian is diagonalised at n_kpar²
    k_∥ points centred on kpar_node.  The smallest gap between surface-weighted
    states near E_F is taken as Δ(d).

    The results are fit to:
        ln Δ(d) = ln Δ₀ − d / λ_arc   (linear in d)

    Parameters
    ----------
    tb_model : WannierTBModel
    thickness_range_uc : list[int]
        Unit-cell thicknesses to scan (e.g. [2, 4, 6, 8, 10]).
    kpar_node : (kx, ky)
        Surface-BZ centre for the arc hybridization search.
    n_kpar : int
        Linear k-mesh per dimension around kpar_node.
    kpar_radius : float
        Half-width of k-window around kpar_node (fractional).
    fermi_ev : float
    n_surface_layers : int
    surface_weight_threshold : float
        Minimum surface weight to consider a state as arc-like.

    Returns
    -------
    dict with keys:
        thickness_uc : list[int]
        gap_ev : list[float | None]   per-thickness gap
        Delta0_ev : float | None      fit intercept
        lambda_arc_uc : float | None  fit penetration depth (unit cells)
        d_crossover_uc : dict         d_c at several temperatures
        fit_quality : float | None    R² of exponential fit
        n_kpar_used : int
        status : str
    """
    from wtec.topology.arc_scan import _collect_tb_hoppings

    parsed = _collect_tb_hoppings(tb_model)
    if parsed[0] is None:
        return {"status": "failed", "reason": parsed[1], "gap_ev": [], "thickness_uc": []}
    n_orb, hoppings = parsed  # type: ignore[misc]

    kx0, ky0 = float(kpar_node[0]), float(kpar_node[1])
    r = max(1e-4, float(kpar_radius))
    n_k = max(1, int(n_kpar))
    kgrid = np.linspace(-r, r, n_k)
    kx_pts = np.mod(kx0 + kgrid, 1.0)
    ky_pts = np.mod(ky0 + kgrid, 1.0)

    gap_per_thickness: list[float | None] = []

    for d_uc in thickness_range_uc:
        gaps_at_d: list[float] = []
        for kx in kx_pts:
            for ky in ky_pts:
                evals, surf_w = _surface_projected_states(
                    tb_model,
                    thickness_uc=int(d_uc),
                    kpar=(float(kx), float(ky)),
                    n_orb=int(n_orb),
                    hoppings=hoppings,
                    n_surface_layers=n_surface_layers,
                )
                g = _find_arc_pair_gap(
                    evals,
                    surf_w,
                    fermi_ev=fermi_ev,
                    surface_weight_threshold=surface_weight_threshold,
                )
                if g is not None and g > 0:
                    gaps_at_d.append(g)

        if gaps_at_d:
            # Use minimum gap (most hybridized k-point)
            gap_per_thickness.append(float(np.min(gaps_at_d)))
        else:
            gap_per_thickness.append(None)

    # Exponential fit: ln(Δ) = ln(Δ₀) − d/λ
    d_arr = np.array([d for d, g in zip(thickness_range_uc, gap_per_thickness) if g is not None and g > 0], dtype=float)
    g_arr = np.array([g for g in gap_per_thickness if g is not None and g > 0], dtype=float)

    Delta0: float | None = None
    lambda_arc: float | None = None
    fit_r2: float | None = None
    d_crossover: dict[str, float | None] = {}

    if len(d_arr) >= 2:
        # Linear regression on log scale
        ln_g = np.log(g_arr)
        d_mean = np.mean(d_arr)
        ln_g_mean = np.mean(ln_g)
        ss_dd = np.sum((d_arr - d_mean) ** 2)
        if ss_dd > 1e-12:
            slope = np.sum((d_arr - d_mean) * (ln_g - ln_g_mean)) / ss_dd
            intercept = ln_g_mean - slope * d_mean

            Delta0 = float(np.exp(intercept))
            lambda_arc = float(-1.0 / slope) if abs(slope) > 1e-12 else None

            # R² on log scale
            ln_g_pred = intercept + slope * d_arr
            ss_res = np.sum((ln_g - ln_g_pred) ** 2)
            ss_tot = np.sum((ln_g - ln_g_mean) ** 2)
            fit_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None

            # Crossover thicknesses at various temperatures
            kB = 8.617e-5  # eV/K
            for T_K in [4.0, 10.0, 77.0, 300.0]:
                kT = kB * T_K
                if Delta0 is not None and lambda_arc is not None and Delta0 > 2 * kT:
                    dc = lambda_arc * np.log(Delta0 / (2.0 * kT))
                    d_crossover[f"d_crossover_uc_at_{int(T_K)}K"] = float(dc)
                else:
                    d_crossover[f"d_crossover_uc_at_{int(T_K)}K"] = None

    status = "ok" if lambda_arc is not None else ("partial" if gap_per_thickness else "failed")
    return {
        "status": status,
        "thickness_uc": list(thickness_range_uc),
        "gap_ev": gap_per_thickness,
        "Delta0_ev": Delta0,
        "lambda_arc_uc": lambda_arc,
        "fit_r2": fit_r2,
        "d_crossover": d_crossover,
        "n_kpar_used": n_k * n_k,
        "kpar_node": list(kpar_node),
        "fermi_ev": float(fermi_ev),
    }


def arc_transmission(
    d_uc: float | np.ndarray,
    *,
    Delta0_ev: float,
    lambda_arc_uc: float,
    T_kelvin: float = 0.0,
) -> np.ndarray:
    """Arc channel transmission factor T_arc(d, T).

    At finite temperature, the arc transmission is thermally activated
    over the hybridization gap:

        T_arc(d, T) = 1 / [1 + (Δ(d) / 2kT)²]     T > 0
        T_arc(d, 0) = Θ(d − d_c)                    T = 0 (step function)

    Parameters
    ----------
    d_uc : float or array
        Film thickness in unit cells.
    Delta0_ev : float
        Zero-thickness hybridization gap amplitude (eV).
    lambda_arc_uc : float
        Arc penetration depth (unit cells).
    T_kelvin : float
        Temperature. Use 0 for zero-temperature approximation.

    Returns
    -------
    np.ndarray
        T_arc ∈ [0, 1] for each thickness.
    """
    d = np.atleast_1d(np.asarray(d_uc, dtype=float))
    gap = float(Delta0_ev) * np.exp(-d / float(lambda_arc_uc))

    if float(T_kelvin) <= 0:
        # Zero temperature: step function at d_c
        kT_eff = 1e-6  # small but nonzero for numerical stability
    else:
        kT_eff = 8.617e-5 * float(T_kelvin)

    return 1.0 / (1.0 + (gap / (2.0 * kT_eff)) ** 2)


def crossover_thickness(
    Delta0_ev: float,
    lambda_arc_uc: float,
    T_kelvin: float,
) -> float | None:
    """Compute d_c where arc transmission T_arc = 0.5.

    d_c = λ_arc · ln(Δ₀ / 2kT)

    Returns None if T = 0 or Δ₀ ≤ 2kT.
    """
    if T_kelvin <= 0:
        return None
    kT = 8.617e-5 * float(T_kelvin)
    if Delta0_ev <= 2.0 * kT:
        return None
    return float(lambda_arc_uc) * np.log(Delta0_ev / (2.0 * kT))


def two_channel_conductance_model(
    thickness_m: np.ndarray,
    thickness_uc: np.ndarray,
    *,
    G_arc0_e2h: float,
    sigma_bulk_e2h_per_m: float,
    width_m: float,
    length_m: float,
    Delta0_ev: float,
    lambda_arc_uc: float,
    T_kelvin: float = 0.0,
) -> np.ndarray:
    """Full two-channel conductance model with hybridization gap.

    G(d, T) = G_arc0 · T_arc(d, T)  +  σ_bulk · W · d / L

    Parameters
    ----------
    thickness_m : np.ndarray
        Film thickness in metres.
    thickness_uc : np.ndarray
        Film thickness in unit cells (parallel to thickness_m).
    G_arc0_e2h : float
        Arc conductance at full transmission (e²/h).
    sigma_bulk_e2h_per_m : float
        Bulk conductivity in e²/h per metre.
    width_m, length_m : float
        Transverse width and transport length (m).
    Delta0_ev, lambda_arc_uc : float
        Hybridization gap model parameters.
    T_kelvin : float
        Temperature (K).

    Returns
    -------
    G : np.ndarray
        Total conductance in e²/h.
    """
    d_m = np.asarray(thickness_m, dtype=float)
    d_uc = np.asarray(thickness_uc, dtype=float)

    T_arc = arc_transmission(d_uc, Delta0_ev=Delta0_ev, lambda_arc_uc=lambda_arc_uc, T_kelvin=T_kelvin)
    G_arc = float(G_arc0_e2h) * T_arc
    G_bulk = float(sigma_bulk_e2h_per_m) * float(width_m) * d_m / float(length_m)
    return G_arc + G_bulk
