"""Physical Fermi-arc metrics in reciprocal-space units (Å⁻¹).

Physics basis
-------------
The Fermi-arc length in reciprocal space is a rigorous topological observable:

    L_arc = ∮_arc dk_∥   (integral along arc contour where A_surf > threshold)

Unlike the heuristic (0.5·fraction + 0.5·span) metric in arc_scan.py, the
arc length in Å⁻¹ is material-independent and directly comparable to ARPES
measurements.  For TaP, the expected arc length is ~0.2 Å⁻¹.

The arc k-width (characteristic momentum width perpendicular to arc) is:

    Δk_arc ≈ |K_{W+} − K_{W−}|_∥   (separation of chirality-pair projections)

This sets the required n_layers_y:
    n_layers_y ≥ 2π / (a · Δk_arc)   [real-space resolution condition]

This module provides:
- Physical arc length extraction via iso-contour integration
- Arc k-width from spectral map half-maximum
- n_layers_y adequacy check with physical basis
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _fractional_to_cartesian(
    kf_pts: np.ndarray,
    recip_vecs_2d: np.ndarray,
) -> np.ndarray:
    """Convert 2D fractional k-points to Cartesian (Å⁻¹).

    Parameters
    ----------
    kf_pts : (N, 2) array of fractional coordinates
    recip_vecs_2d : (2, 2) reciprocal lattice vectors in Å⁻¹
        rows are b_1, b_2

    Returns
    -------
    (N, 2) Cartesian k-points in Å⁻¹
    """
    return kf_pts @ recip_vecs_2d


def _reciprocal_2d(lattice_vecs: np.ndarray, surface_axis: str = "z") -> np.ndarray:
    """Extract 2D reciprocal lattice vectors (Å⁻¹) for surface BZ.

    Parameters
    ----------
    lattice_vecs : (3, 3) real-space lattice vectors (Å), rows are a_1, a_2, a_3
    surface_axis : 'z', 'x', or 'y'

    Returns
    -------
    (2, 2) in-plane reciprocal lattice vectors (Å⁻¹)
    """
    lv = np.asarray(lattice_vecs, dtype=float)
    # Full reciprocal lattice: b_i = 2π (a_j × a_k) / V
    V = float(np.dot(lv[0], np.cross(lv[1], lv[2])))
    b = np.array([
        2.0 * np.pi * np.cross(lv[1], lv[2]) / V,
        2.0 * np.pi * np.cross(lv[2], lv[0]) / V,
        2.0 * np.pi * np.cross(lv[0], lv[1]) / V,
    ])  # (3, 3), rows are b_1, b_2, b_3

    ax = str(surface_axis).strip().lower()
    if ax == "z":
        # Surface BZ: b_1_xy, b_2_xy projected onto xy
        return np.array([b[0, :2], b[1, :2]], dtype=float)
    if ax == "x":
        return np.array([b[1, 1:], b[2, 1:]], dtype=float)
    if ax == "y":
        return np.array([[b[0, 0], b[0, 2]], [b[2, 0], b[2, 2]]], dtype=float)
    raise ValueError(f"surface_axis must be 'x', 'y', or 'z', got {surface_axis!r}")


def fermi_arc_length_angstrom(
    spectral_map: np.ndarray,
    recip_vecs_2d: np.ndarray,
    threshold_fraction: float = 0.20,
) -> dict[str, Any]:
    """Compute Fermi-arc length in Å⁻¹ from spectral map.

    Extracts the arc contour at A_surf = threshold × A_max using
    marching-squares (scikit-image) if available, otherwise falls back to
    a gradient-based method.

    Parameters
    ----------
    spectral_map : (N_kx, N_ky) float array
        Surface spectral function A_surf(k_x, k_y) on a uniform grid.
    recip_vecs_2d : (2, 2) float array
        In-plane reciprocal lattice vectors (Å⁻¹), rows are b_1, b_2.
    threshold_fraction : float
        Iso-contour level as fraction of maximum spectral weight.

    Returns
    -------
    dict with keys:
        arc_length_angs  : float  total arc contour length (Å⁻¹)
        arc_width_angs   : float  RMS width perpendicular to arc (Å⁻¹)
        n_contours       : int    number of disconnected arc segments
        threshold        : float  absolute threshold used
        method           : str    'marching_squares' or 'gradient_fallback'
        status           : str
    """
    sm = np.asarray(spectral_map, dtype=float)
    if sm.ndim != 2 or sm.size == 0:
        return {"status": "failed", "reason": "invalid_spectral_map", "arc_length_angs": 0.0}

    peak = float(np.max(sm))
    if peak <= 0:
        return {"status": "failed", "reason": "zero_spectral_weight", "arc_length_angs": 0.0}

    threshold = peak * float(threshold_fraction)
    nkx, nky = sm.shape
    b = np.asarray(recip_vecs_2d, dtype=float)  # (2, 2)

    # --- try scikit-image marching squares ---
    try:
        from skimage.measure import find_contours  # type: ignore[import]

        contours = find_contours(sm, threshold)
        total_length = 0.0
        for contour in contours:
            # contour[:,0] = kx index, contour[:,1] = ky index
            # convert index to fractional: kf = idx / N
            kf = contour / np.array([nkx, nky], dtype=float)  # (N_pts, 2)
            kc = kf @ b  # (N_pts, 2) Cartesian Å⁻¹
            segs = np.diff(kc, axis=0)
            total_length += float(np.sum(np.linalg.norm(segs, axis=1)))

        # Arc width: 2D moment analysis of thresholded region
        mask = sm > threshold
        arc_width = _arc_width_from_mask(mask, b, nkx, nky)

        return {
            "status": "ok",
            "arc_length_angs": float(total_length),
            "arc_width_angs": float(arc_width),
            "n_contours": int(len(contours)),
            "threshold": float(threshold),
            "threshold_fraction": float(threshold_fraction),
            "method": "marching_squares",
        }

    except ImportError:
        pass

    # --- fallback: boundary pixel method ---
    mask = sm > threshold
    total_length = _arc_length_from_mask_boundary(mask, b, nkx, nky)
    arc_width = _arc_width_from_mask(mask, b, nkx, nky)

    return {
        "status": "ok",
        "arc_length_angs": float(total_length),
        "arc_width_angs": float(arc_width),
        "n_contours": -1,  # not computed in fallback
        "threshold": float(threshold),
        "threshold_fraction": float(threshold_fraction),
        "method": "boundary_pixel_fallback",
    }


def _arc_length_from_mask_boundary(
    mask: np.ndarray,
    b: np.ndarray,
    nkx: int,
    nky: int,
) -> float:
    """Estimate arc length from the perimeter of the thresholded region."""
    mask = np.asarray(mask, dtype=bool)
    # Find boundary pixels (pixels with at least one False 4-neighbour)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    boundary = (
        mask
        & (
            ~padded[:-2, 1:-1]
            | ~padded[2:, 1:-1]
            | ~padded[1:-1, :-2]
            | ~padded[1:-1, 2:]
        )
    )
    n_boundary = int(np.sum(boundary))
    if n_boundary == 0:
        return 0.0

    # Pixel size in Å⁻¹
    dk_b1 = float(np.linalg.norm(b[0])) / float(nkx)
    dk_b2 = float(np.linalg.norm(b[1])) / float(nky)
    pixel_size = 0.5 * (dk_b1 + dk_b2)

    return float(n_boundary * pixel_size)


def _arc_width_from_mask(
    mask: np.ndarray,
    b: np.ndarray,
    nkx: int,
    nky: int,
) -> float:
    """Estimate arc width (Å⁻¹) as RMS extent of thresholded region."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return 0.0

    ix, iy = np.where(mask)
    kf = np.stack([ix / nkx, iy / nky], axis=1)  # fractional
    kc = kf @ b  # Cartesian Å⁻¹

    centre = np.mean(kc, axis=0)
    deviations = kc - centre
    rms = float(np.sqrt(np.mean(np.sum(deviations ** 2, axis=1))))
    return rms


def required_n_layers_y(
    arc_k_width_angs: float,
    lattice_a_angs: float,
    safety_factor: float = 2.0,
) -> int:
    """Minimum n_layers_y to resolve the Fermi arc.

    Condition: W = n_y · a ≫ 2π / Δk_arc
    i.e. n_y ≥ safety_factor · 2π / (a · Δk_arc)

    Parameters
    ----------
    arc_k_width_angs : float  arc k-width Δk_arc (Å⁻¹)
    lattice_a_angs : float    in-plane lattice constant (Å)
    safety_factor : float     multiplicative safety margin (default 2)

    Returns
    -------
    int  minimum n_layers_y
    """
    if float(arc_k_width_angs) <= 0:
        return 16  # fallback
    n_min = float(safety_factor) * 2.0 * np.pi / (float(lattice_a_angs) * float(arc_k_width_angs))
    return int(np.ceil(n_min))


def compute_arc_length_from_tb(
    tb_model,
    *,
    energy_ev: float = 0.0,
    n_kx: int = 40,
    n_ky: int = 40,
    broadening_ev: float = 0.06,
    thickness_uc: int = 20,
    surface_axis: str = "z",
    threshold_fraction: float = 0.20,
    engine: str = "tb_slab",
) -> dict[str, Any]:
    """Compute physical arc length (Å⁻¹) from tight-binding model.

    Parameters
    ----------
    tb_model : WannierTBModel
    energy_ev : float
    n_kx, n_ky : int  k-mesh for spectral map
    broadening_ev : float
    thickness_uc : int  slab thickness (only for tb_slab engine)
    surface_axis : str
    threshold_fraction : float
    engine : str  'tb_slab' or 'lopez_sancho'

    Returns
    -------
    dict with arc_length_angs, arc_width_angs, and full arc_metrics output
    """
    kx = np.linspace(0.0, 1.0, n_kx, endpoint=False)
    ky = np.linspace(0.0, 1.0, n_ky, endpoint=False)

    if engine == "lopez_sancho":
        from wtec.topology.surface_gf import compute_surface_spectral_metric_ls
        result = compute_surface_spectral_metric_ls(
            tb_model,
            energy_ev=energy_ev,
            n_kx=n_kx,
            n_ky=n_ky,
            broadening_ev=broadening_ev,
        )
        if result.get("status") != "ok" or "spectral_map" not in result:
            return {"status": result.get("status", "failed"), "reason": result.get("reason", "no_map")}
        spectral_map = np.asarray(result["spectral_map"], dtype=float)
    else:
        from wtec.topology.arc_scan import _collect_tb_hoppings, _surface_spectral_map_from_hoppings
        parsed = _collect_tb_hoppings(tb_model)
        if parsed[0] is None:
            return {"status": "failed", "reason": parsed[1]}
        n_orb, hoppings = parsed  # type: ignore[misc]

        try:
            spectral_map = _surface_spectral_map_from_hoppings(
                hoppings=hoppings,
                n_orb=int(n_orb),
                n_layers_z=max(2, int(thickness_uc)),
                energy_ev=float(energy_ev),
                eta=float(broadening_ev),
                kx_coords=kx,
                ky_coords=ky,
            )
        except Exception as exc:
            return {"status": "failed", "reason": f"slab_spectral_failed:{exc}"}

    # Get 2D reciprocal lattice in Å⁻¹
    try:
        recip_2d = _reciprocal_2d(tb_model.lattice_vectors, surface_axis)
    except Exception as exc:
        return {"status": "failed", "reason": f"recip_vec_failed:{exc}"}

    arc_result = fermi_arc_length_angstrom(
        spectral_map,
        recip_2d,
        threshold_fraction=threshold_fraction,
    )
    arc_result["engine"] = engine
    arc_result["n_kx"] = n_kx
    arc_result["n_ky"] = n_ky
    arc_result["energy_ev"] = float(energy_ev)
    arc_result["broadening_ev"] = float(broadening_ev)

    # Estimate required n_layers_y from arc width
    if arc_result.get("arc_width_angs", 0) > 0:
        try:
            a_lat = float(np.linalg.norm(tb_model.lattice_vectors[0]))
        except Exception:
            a_lat = 3.3  # TaP default
        arc_result["recommended_n_layers_y"] = required_n_layers_y(
            arc_result["arc_width_angs"],
            a_lat,
        )

    return arc_result
