"""Compute conductance G(d) and G(L) via Kwant Landauer formula.

Parallel ensemble averaging is MPI-rank split via mpi4py when launched
with mpirun. Fork-based parallel process launch is never used.
"""

from __future__ import annotations

import numpy as np

from wtec.transport.geometry import region_geometry


def compute_conductance_vs_thickness(
    tb_model,
    thicknesses: list[int] | np.ndarray,
    *,
    disorder_strength: float = 0.0,
    n_ensemble: int = 1,
    energy: float = 0.0,
    cross_section_m2: float | None = None,
    n_jobs: int = 1,
    base_seed: int = 0,
    lead_onsite_eV: float = 0.0,
    lead_axis: str = "x",
    thickness_axis: str = "z",
    n_layers_x: int = 4,
    n_layers_y: int = 4,
) -> dict:
    """Compute conductance G and resistivity ρ vs film thickness.

    The sweep variable is the unit-cell count along `thickness_axis`.
    Resistivity is computed using axis-consistent geometry:
        ρ = L / (G_SI * A)
    where L is length along `lead_axis` and A is cross-section normal to it.
    """
    try:
        import kwant  # noqa: F401
    except ImportError:
        raise ImportError("kwant is required")

    thicknesses = np.asarray(thicknesses, dtype=int)
    if np.any(thicknesses <= 0):
        raise ValueError("thicknesses must contain only positive integers.")

    base_counts = {
        "x": int(n_layers_x),
        "y": int(n_layers_y),
        "z": int(np.max(thicknesses)) if thickness_axis == "z" else 1,
    }
    # Keep x/y defaults unless swept.
    if thickness_axis != "x":
        base_counts["x"] = int(n_layers_x)
    if thickness_axis != "y":
        base_counts["y"] = int(n_layers_y)

    G_means: list[float] = []
    G_stds: list[float] = []
    length_m_vals: list[float] = []
    thickness_m_vals: list[float] = []
    area_vals: list[float] = []

    for d in thicknesses:
        counts = dict(base_counts)
        counts[thickness_axis] = int(d)
        if any(counts[a] <= 0 for a in ("x", "y", "z")):
            raise ValueError(f"Invalid layer counts for d={d}: {counts}")

        G_vals = _ensemble_conductance(
            tb_model=tb_model,
            n_layers_x=counts["x"],
            n_layers_y=counts["y"],
            n_layers_z=counts["z"],
            lead_axis=lead_axis,
            disorder_strength=disorder_strength,
            n_ensemble=n_ensemble,
            energy=energy,
            n_jobs=n_jobs,
            base_seed=base_seed,
            lead_onsite_eV=lead_onsite_eV,
        )
        G_means.append(float(np.mean(G_vals)))
        G_stds.append(float(np.std(G_vals)))

        geom = region_geometry(
            tb_model.lattice_vectors,
            n_layers_x=counts["x"],
            n_layers_y=counts["y"],
            n_layers_z=counts["z"],
            lead_axis=lead_axis,
            thickness_axis=thickness_axis,
        )
        length_m_vals.append(geom["length_m"])
        thickness_m_vals.append(geom["thickness_m"])
        area_vals.append(cross_section_m2 if cross_section_m2 is not None else geom["cross_section_m2"])

    G_means_arr = np.array(G_means, dtype=float)
    G_stds_arr = np.array(G_stds, dtype=float)
    length_m = np.array(length_m_vals, dtype=float)
    thickness_m = np.array(thickness_m_vals, dtype=float)
    cross_section_arr = np.array(area_vals, dtype=float)

    e2h = 7.748091729e-5   # e²/h in Siemens
    G_SI = G_means_arr * e2h
    rho_mean = np.where(
        (G_SI > 0) & (cross_section_arr > 0),
        length_m / (G_SI * cross_section_arr),
        np.inf,
    )

    G_SI_std = G_stds_arr * e2h
    rho_std = np.where(G_SI > 0, rho_mean * G_SI_std / G_SI, np.inf)

    return {
        "thickness_uc": thicknesses,
        "thickness_m": thickness_m,
        "length_m": length_m,
        "cross_section_m2": cross_section_arr,
        "lead_axis": lead_axis,
        "thickness_axis": thickness_axis,
        "n_layers_x": int(n_layers_x),
        "n_layers_y": int(n_layers_y),
        "G_mean": G_means_arr,
        "G_std": G_stds_arr,
        "rho_mean": rho_mean,
        "rho_std": rho_std,
    }


def compute_conductance_vs_length(
    tb_model,
    lengths: list[int] | np.ndarray,
    disorder_strength: float,
    *,
    n_layers_z_fixed: int = 10,
    n_layers_x_fixed: int = 4,
    n_layers_y: int = 4,
    lead_axis: str = "x",
    thickness_axis: str = "z",
    n_ensemble: int = 50,
    energy: float = 0.0,
    n_jobs: int = 1,
    base_seed: int = 0,
    lead_onsite_eV: float = 0.0,
) -> dict:
    """Compute G(L) for MFP extraction.

    `lengths` is interpreted as the unit-cell count along `lead_axis`.
    """
    lengths = np.asarray(lengths, dtype=int)
    if np.any(lengths <= 0):
        raise ValueError("lengths must contain only positive integers.")

    base_counts = {
        "x": int(n_layers_x_fixed),
        "y": int(n_layers_y),
        "z": int(n_layers_z_fixed),
    }
    if any(base_counts[a] <= 0 for a in ("x", "y", "z")):
        raise ValueError("n_layers_x_fixed, n_layers_y, n_layers_z_fixed must be > 0")

    G_means: list[float] = []
    G_stds: list[float] = []
    length_m_vals: list[float] = []
    cross_section_vals: list[float] = []

    for L in lengths:
        counts = dict(base_counts)
        counts[lead_axis] = int(L)

        G_vals = _ensemble_conductance(
            tb_model=tb_model,
            n_layers_x=counts["x"],
            n_layers_y=counts["y"],
            n_layers_z=counts["z"],
            lead_axis=lead_axis,
            disorder_strength=disorder_strength,
            n_ensemble=n_ensemble,
            energy=energy,
            n_jobs=n_jobs,
            base_seed=base_seed,
            lead_onsite_eV=lead_onsite_eV,
        )
        G_means.append(float(np.mean(G_vals)))
        G_stds.append(float(np.std(G_vals)))

        geom = region_geometry(
            tb_model.lattice_vectors,
            n_layers_x=counts["x"],
            n_layers_y=counts["y"],
            n_layers_z=counts["z"],
            lead_axis=lead_axis,
            thickness_axis=thickness_axis,
        )
        length_m_vals.append(geom["length_m"])
        cross_section_vals.append(geom["cross_section_m2"])

    return {
        "length_uc": lengths,
        "length_m": np.array(length_m_vals, dtype=float),
        "cross_section_m2": np.array(cross_section_vals, dtype=float),
        "lead_axis": lead_axis,
        "thickness_axis": thickness_axis,
        "G_mean": np.array(G_means, dtype=float),
        "G_std": np.array(G_stds, dtype=float),
    }


def _ensemble_conductance(
    tb_model,
    n_layers_x: int,
    n_layers_y: int,
    n_layers_z: int,
    lead_axis: str,
    disorder_strength: float,
    n_ensemble: int,
    energy: float,
    n_jobs: int,
    base_seed: int,
    lead_onsite_eV: float,
) -> np.ndarray:
    """Run disorder ensemble and return array of G values (e²/h)."""
    comm, rank, size = _mpi_context()

    if n_ensemble <= 1 or disorder_strength == 0.0:
        value = None
        if rank == 0:
            value = _single_conductance(
                tb_model=tb_model,
                n_layers_x=n_layers_x,
                n_layers_y=n_layers_y,
                n_layers_z=n_layers_z,
                lead_axis=lead_axis,
                disorder_strength=0.0,
                energy=energy,
                seed=0,
                lead_onsite_eV=lead_onsite_eV,
            )
        if comm is not None:
            value = comm.bcast(value, root=0)
        return np.array([value], dtype=float)

    indices = list(range(n_ensemble))
    local_indices = indices[rank::size]

    local_results: list[tuple[int, float]] = []
    local_error: str | None = None
    for idx in local_indices:
        try:
            g = _single_conductance(
                tb_model=tb_model,
                n_layers_x=n_layers_x,
                n_layers_y=n_layers_y,
                n_layers_z=n_layers_z,
                lead_axis=lead_axis,
                disorder_strength=disorder_strength,
                energy=energy,
                seed=base_seed + idx,
                lead_onsite_eV=lead_onsite_eV,
            )
            local_results.append((idx, float(g)))
        except Exception as exc:  # pragma: no cover - runtime/HPC dependent
            local_error = (
                f"rank={rank}, idx={idx}, seed={base_seed + idx}, "
                f"error={type(exc).__name__}: {exc}"
            )
            break

    if comm is None:
        if local_error:
            raise RuntimeError(local_error)
        local_results.sort(key=lambda x: x[0])
        return np.array([g for _, g in local_results], dtype=float)

    gathered_errors = comm.gather(local_error, root=0)
    gathered_results = comm.gather(local_results, root=0)

    payload: dict | None = None
    if rank == 0:
        errors = [e for e in gathered_errors if e]
        if errors:
            payload = {"error": "MPI worker failure(s): " + " | ".join(errors)}
        else:
            merged: list[tuple[int, float]] = []
            for chunk in gathered_results:
                merged.extend(chunk)
            merged.sort(key=lambda x: x[0])
            if len(merged) != n_ensemble:
                payload = {
                    "error": (
                        f"MPI result size mismatch: expected={n_ensemble}, "
                        f"got={len(merged)}"
                    )
                }
            else:
                payload = {"results": [g for _, g in merged]}

    payload = comm.bcast(payload, root=0)
    if payload is None:
        raise RuntimeError("MPI result payload is missing.")
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return np.array(payload["results"], dtype=float)


def _single_conductance(
    tb_model,
    n_layers_x: int,
    n_layers_y: int,
    n_layers_z: int,
    lead_axis: str,
    disorder_strength: float,
    energy: float,
    seed: int,
    lead_onsite_eV: float,
) -> float:
    """Build, disorder, finalize, and compute G for one configuration."""
    import kwant
    from wtec.transport.disorder import add_anderson_disorder

    try:
        sys = tb_model.to_kwant_builder(
            n_layers_z=n_layers_z,
            n_layers_x=n_layers_x,
            n_layers_y=n_layers_y,
            lead_axis=lead_axis,
            substrate_onsite_eV=lead_onsite_eV,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to build Kwant transport system. "
            f"n_layers_x={n_layers_x}, n_layers_y={n_layers_y}, n_layers_z={n_layers_z}, "
            f"lead_axis={lead_axis}. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    if disorder_strength > 0.0:
        rng = np.random.default_rng(seed)
        add_anderson_disorder(sys, disorder_strength, rng=rng)

    fsys = sys.finalized()
    try:
        smat = kwant.smatrix(fsys, energy)
    except ValueError as exc:
        raise RuntimeError(
            "Kwant scattering matrix construction failed. "
            f"energy={energy}, lead_axis={lead_axis}. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    # Transmission from right lead -> left lead
    return float(smat.transmission(0, 1))


def _mpi_context():
    """Return (comm, rank, size). Falls back to serial when mpi4py is unavailable."""
    try:
        from mpi4py import MPI
    except Exception:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()
