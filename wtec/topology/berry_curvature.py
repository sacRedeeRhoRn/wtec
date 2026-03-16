"""Berry curvature and anomalous Hall conductivity.

Physics basis
-------------
The Berry curvature of band n at k-point **k** is:

    Ω_n^z(**k**) = -2 Im Σ_{m≠n} ⟨u_n(**k**)|∂H/∂k_x|u_m(**k**)⟩ · ⟨u_m(**k**)|∂H/∂k_y|u_n(**k**)⟩
                                   ——————————————————————————————
                                              (E_m − E_n)²

The anomalous Hall conductivity (Kubo formula):

    σ_xy = -(e²/ℏ) ∫_BZ Σ_n f_n(**k**) Ω_n^z(**k**) d³k / (2π)³

For a Weyl semimetal with Weyl nodes at K_{W±}, σ_xy contains a contribution:

    σ_xy^Weyl = (e²/2πh) · (K_{W+,z} − K_{W−,z})

which is quantised in units of e²/2πh per pair of nodes.

The Berry curvature hot spots (large |Ω_n|) near Weyl nodes are also used to
guide adaptive k-sampling in arc detection, providing a more rigorous basis than
the gap-based hotspot selection currently used.

Reference: Xiao, Chang & Niu, Rev. Mod. Phys. 82, 1959 (2010).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _berry_curvature_kubo(
    tb_model,
    k: np.ndarray,
    occupied_bands: list[int],
    delta_k: float = 1e-4,
) -> np.ndarray:
    """Berry curvature Ω_n^z at a single k-point (Kubo formula).

    Returns Ω_n^z for each band n in occupied_bands.

    Ω_n^z = -2 Im Σ_{m≠n} ⟨u_n|∂H/∂k_x|u_m⟩⟨u_m|∂H/∂k_y|u_n⟩ / (E_m - E_n)²
    """
    from wtec.topology.node_scan import _dH_dk_frac, _wrap_k

    k = _wrap_k(np.asarray(k, dtype=float))
    H = tb_model.hamiltonian_at_k(k)
    evals, evecs = np.linalg.eigh(H)

    # Velocity matrix elements v_x = ∂H/∂k_x, v_y = ∂H/∂k_y
    dH_dkx = _dH_dk_frac(tb_model, k, axis=0, delta=delta_k)
    dH_dky = _dH_dk_frac(tb_model, k, axis=1, delta=delta_k)

    # v_{nm}^{x,y} = ⟨u_n|∂H/∂k_{x,y}|u_m⟩
    vx = evecs.conj().T @ dH_dkx @ evecs   # (n_bands, n_bands)
    vy = evecs.conj().T @ dH_dky @ evecs

    n_bands = len(evals)
    omega_z = np.zeros(len(occupied_bands), dtype=float)

    for idx, n in enumerate(occupied_bands):
        occ_z = 0.0 + 0.0j
        for m in range(n_bands):
            if m == n:
                continue
            dE = evals[m] - evals[n]
            if abs(dE) < 1e-6:
                continue
            occ_z += vx[n, m] * vy[m, n] / (dE ** 2)
        omega_z[idx] = float(-2.0 * occ_z.imag)

    return omega_z


def _berry_curvature_plaquette(
    tb_model,
    k: np.ndarray,
    band_idx: int,
    dk: float,
) -> float:
    """Berry curvature Ω_n^z via Berry plaquette formula (numerically stable).

    Computes Berry flux through the plaquette [k, k+dk_x, k+dk_x+dk_y, k+dk_y]
    and divides by the area dk_x·dk_y.

    This method is gauge-invariant and does not suffer from near-degeneracy
    issues that affect the Kubo formula.
    """
    from wtec.topology.node_scan import _berry_plaquette_phase, _wrap_k

    k = _wrap_k(np.asarray(k, dtype=float))
    dk1 = np.array([dk, 0, 0], dtype=float)
    dk2 = np.array([0, dk, 0], dtype=float)

    phase = _berry_plaquette_phase(tb_model, k, dk1, dk2, band_idx)
    area = dk ** 2
    return float(phase / area)


def compute_berry_curvature_map(
    tb_model,
    *,
    kz_fixed: float = 0.0,
    n_kxy: int = 30,
    occupied_bands: list[int] | None = None,
    n_occ_auto: int | None = None,
    fermi_ev: float = 0.0,
    method: str = "plaquette",
    delta_k: float = 1e-3,
) -> dict[str, Any]:
    """Compute Berry curvature map Ω_n^z(k_x, k_y) at fixed k_z.

    Parameters
    ----------
    tb_model : WannierTBModel
    kz_fixed : float  fixed k_z value (fractional)
    n_kxy : int       linear k-mesh size
    occupied_bands : list[int] | None
        Band indices to compute Ω for.  None = auto from fermi_ev.
    n_occ_auto : int | None
        If occupied_bands is None, use this many lowest bands.
    fermi_ev : float  Fermi level for auto band selection.
    method : str      'plaquette' (stable) or 'kubo' (fast)
    delta_k : float   Finite-difference step for velocity/plaquette.

    Returns
    -------
    dict with keys:
        kx_frac, ky_frac : list[float]
        omega_z_sum : (n_kxy, n_kxy) float  sum over occupied bands
        omega_z_bands : list of (n_kxy, n_kxy)  per-band
        chern_estimate : float  (1/2π) ∑ Ω dk_x dk_y
        hot_spot_kxy : list of [kx, ky, omega]  top-5 hot spots
        status : str
    """
    kz = float(kz_fixed)
    nk = max(4, int(n_kxy))
    kx_vals = np.linspace(0.0, 1.0, nk, endpoint=False)
    ky_vals = np.linspace(0.0, 1.0, nk, endpoint=False)
    dk = 1.0 / nk

    # Determine occupied bands
    H0 = tb_model.hamiltonian_at_k(np.array([0.0, 0.0, kz]))
    evals0 = np.linalg.eigvalsh(H0)
    n_orb = len(evals0)

    if occupied_bands is not None:
        occ = list(occupied_bands)
    elif n_occ_auto is not None:
        occ = list(range(min(int(n_occ_auto), n_orb)))
    else:
        n_occ = int(np.sum(evals0 < float(fermi_ev)))
        occ = list(range(max(1, n_occ)))

    if not occ:
        return {"status": "failed", "reason": "no_occupied_bands", "omega_z_sum": None}

    n_occ_bands = len(occ)
    omega_sum = np.zeros((nk, nk), dtype=float)
    omega_per_band = [np.zeros((nk, nk), dtype=float) for _ in occ]

    for ix, kx in enumerate(kx_vals):
        for iy, ky in enumerate(ky_vals):
            k = np.array([kx, ky, kz], dtype=float)
            if method == "kubo":
                omega_at_k = _berry_curvature_kubo(tb_model, k, occ, delta_k=float(delta_k))
                for band_i, omega_n in enumerate(omega_at_k):
                    omega_per_band[band_i][ix, iy] = float(omega_n)
                    omega_sum[ix, iy] += float(omega_n)
            else:
                # Plaquette method: one band at a time
                for band_i, n in enumerate(occ):
                    omega_n = _berry_curvature_plaquette(tb_model, k, n, float(delta_k))
                    omega_per_band[band_i][ix, iy] = float(omega_n)
                    omega_sum[ix, iy] += float(omega_n)

    # Chern number estimate: (1/2π) ∑_{ix,iy} Ω_z · Δk² (Δk in fractional units)
    chern_estimate = float(np.sum(omega_sum) * dk ** 2 / (2.0 * np.pi))

    # Berry curvature hot spots (Weyl node indicators)
    abs_omega = np.abs(omega_sum)
    flat = abs_omega.ravel()
    top5_idx = np.argsort(flat)[-5:][::-1]
    hot_spots = []
    for idx in top5_idx:
        ix = int(idx // nk)
        iy = int(idx % nk)
        hot_spots.append({
            "kx_frac": float(kx_vals[ix]),
            "ky_frac": float(ky_vals[iy]),
            "kz_frac": float(kz),
            "omega_z": float(omega_sum[ix, iy]),
        })

    return {
        "status": "ok",
        "kz_fixed": float(kz),
        "kx_frac": kx_vals.tolist(),
        "ky_frac": ky_vals.tolist(),
        "omega_z_sum": omega_sum,
        "omega_z_bands": omega_per_band,
        "chern_estimate": float(chern_estimate),
        "hot_spot_kxy": hot_spots,
        "occupied_bands": occ,
        "method": method,
        "n_kxy": nk,
        "dk_frac": float(dk),
    }


def compute_anomalous_hall_conductivity(
    tb_model,
    *,
    n_k3d: int = 20,
    fermi_ev: float = 0.0,
    method: str = "plaquette",
    delta_k: float = 1e-3,
) -> dict[str, Any]:
    """Compute σ_xy via Kubo formula on a 3D k-mesh.

    σ_xy = -(e²/ℏ) ∫_BZ Σ_n f_n(k) Ω_n^z(k) dk³ / (2π)³
          ≈ -(e²/ℏ) · (1/N³) Σ_{k,n} f_n(k) Ω_n^z(k)

    Units: (e²/h) per unit cell (multiply by 1/c to get S/m for 3D).

    Parameters
    ----------
    tb_model : WannierTBModel
    n_k3d : int  linear k-mesh per dimension
    fermi_ev : float
    method : str  'plaquette' or 'kubo'
    delta_k : float

    Returns
    -------
    dict with sigma_xy_e2h, sigma_xy_e2h_per_uc, status
    """
    nk = max(4, int(n_k3d))
    kgrid = np.linspace(0.0, 1.0, nk, endpoint=False)
    dk = 1.0 / nk

    H0 = tb_model.hamiltonian_at_k(np.zeros(3))
    n_orb = H0.shape[0]

    sigma_sum = 0.0
    n_pts = 0

    for ix in range(nk):
        for iy in range(nk):
            for iz in range(nk):
                k = np.array([kgrid[ix], kgrid[iy], kgrid[iz]], dtype=float)
                H = tb_model.hamiltonian_at_k(k)
                evals, _ = np.linalg.eigh(H)

                # Occupied bands at this k-point
                occ = [n for n, e in enumerate(evals) if e < float(fermi_ev)]
                if not occ:
                    n_pts += 1
                    continue

                if method == "kubo":
                    omega_z = _berry_curvature_kubo(tb_model, k, occ, delta_k=delta_k)
                    sigma_sum += float(np.sum(omega_z))
                else:
                    for n in occ:
                        sigma_sum += _berry_curvature_plaquette(tb_model, k, n, float(delta_k))
                n_pts += 1

    # σ_xy = -(e²/ℏ) ∫ f·Ω dk³/(2π)³
    # In fractional coordinates: (2π)³ / V_BZ = 1, so:
    # σ_xy = -(e²/ℏ) × (1/N³) × Σ Ω_z   [in units where BZ volume = 1]
    sigma_xy = -sigma_sum * dk ** 3 / (2.0 * np.pi)   # e²/h per unit cell

    return {
        "status": "ok",
        "sigma_xy_e2h_per_uc": float(sigma_xy),
        "n_k3d": nk,
        "fermi_ev": float(fermi_ev),
        "method": method,
    }


def berry_curvature_hotspots_for_arc_sampling(
    tb_model,
    *,
    kz_vals: list[float] | None = None,
    n_kz_scan: int = 20,
    n_kxy: int = 20,
    fermi_ev: float = 0.0,
    top_n: int = 8,
    surface_axis: str = "z",
) -> list[dict[str, Any]]:
    """Find Berry curvature hot spots for adaptive k-sampling.

    This is a more rigorous alternative to the gap-based hotspot selection
    in adaptive_k.py.  Berry curvature hot spots |Ω| are peaked at Weyl node
    projections onto the surface BZ.

    Returns a list of hotspots compatible with the format expected by
    select_node_projected_hotspots output:
        [{"center_uv": [u, v], "omega_z": float, "kz": float}, ...]
    """
    if kz_vals is None:
        kz_vals = np.linspace(0.0, 1.0, n_kz_scan, endpoint=False).tolist()

    all_spots: list[dict[str, Any]] = []

    for kz in kz_vals:
        bc_map = compute_berry_curvature_map(
            tb_model,
            kz_fixed=float(kz),
            n_kxy=n_kxy,
            fermi_ev=fermi_ev,
            method="plaquette",
        )
        if bc_map.get("status") != "ok":
            continue
        for spot in bc_map.get("hot_spot_kxy", []):
            # Project to surface BZ coordinates
            if surface_axis == "z":
                uv = [spot["kx_frac"], spot["ky_frac"]]
            elif surface_axis == "x":
                uv = [spot["ky_frac"], spot["kz_frac"]]
            else:
                uv = [spot["kx_frac"], spot["kz_frac"]]
            all_spots.append({
                "center_uv": uv,
                "omega_z": float(spot["omega_z"]),
                "kz_frac": float(kz),
                "source": "berry_curvature_hotspot",
            })

    # Sort by |Ω| and deduplicate
    all_spots.sort(key=lambda s: -abs(float(s["omega_z"])))
    deduped: list[dict[str, Any]] = []
    for spot in all_spots:
        u, v = float(spot["center_uv"][0]), float(spot["center_uv"][1])
        duplicate = any(
            abs(float(s["center_uv"][0]) - u) < 0.04
            and abs(float(s["center_uv"][1]) - v) < 0.04
            for s in deduped
        )
        if not duplicate:
            deduped.append(spot)
        if len(deduped) >= top_n:
            break

    return deduped
