"""Compute conductance G(d) and G(L) via Kwant Landauer formula.

Parallel ensemble averaging is MPI-rank split via mpi4py when launched
with mpirun. Fork-based parallel process launch is never used.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable

import numpy as np

from wtec.transport.geometry import region_geometry

_CLEAN_FSYS_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_CLEAN_FSYS_CACHE_LOCK = Lock()
_CLEAN_FSYS_CACHE_MAX = 24


def _required_lead_axis_cells(tb_model, *, counts: dict[str, int], lead_axis: str) -> int:
    """Best-effort lead-axis minimum inferred from the TB model."""
    fn = getattr(tb_model, "required_lead_axis_cells", None)
    if callable(fn):
        return int(
            fn(
                lead_axis=str(lead_axis),
                n_layers_x=int(counts["x"]),
                n_layers_y=int(counts["y"]),
                n_layers_z=int(counts["z"]),
            )
        )
    return 2


def _emit_progress(progress_cb: Callable[..., None] | None, event: str, **payload: Any) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(event=event, **payload)
    except Exception:
        # Progress callbacks must never break transport numerics.
        pass


def _clean_fsys_cache_key(
    tb_model,
    *,
    n_layers_x: int,
    n_layers_y: int,
    n_layers_z: int,
    lead_axis: str,
    lead_onsite_eV: float,
) -> tuple:
    return (
        int(id(tb_model)),
        int(n_layers_x),
        int(n_layers_y),
        int(n_layers_z),
        str(lead_axis),
        float(lead_onsite_eV),
    )


def _get_or_build_clean_fsys(
    tb_model,
    *,
    n_layers_x: int,
    n_layers_y: int,
    n_layers_z: int,
    lead_axis: str,
    lead_onsite_eV: float,
):
    key = _clean_fsys_cache_key(
        tb_model,
        n_layers_x=n_layers_x,
        n_layers_y=n_layers_y,
        n_layers_z=n_layers_z,
        lead_axis=lead_axis,
        lead_onsite_eV=lead_onsite_eV,
    )
    with _CLEAN_FSYS_CACHE_LOCK:
        cached = _CLEAN_FSYS_CACHE.get(key)
        if cached is not None:
            _CLEAN_FSYS_CACHE.move_to_end(key, last=True)
            return cached

    sys = tb_model.to_kwant_builder(
        n_layers_z=n_layers_z,
        n_layers_x=n_layers_x,
        n_layers_y=n_layers_y,
        lead_axis=lead_axis,
        substrate_onsite_eV=lead_onsite_eV,
    )
    fsys = sys.finalized()

    with _CLEAN_FSYS_CACHE_LOCK:
        _CLEAN_FSYS_CACHE[key] = fsys
        _CLEAN_FSYS_CACHE.move_to_end(key, last=True)
        while len(_CLEAN_FSYS_CACHE) > _CLEAN_FSYS_CACHE_MAX:
            _CLEAN_FSYS_CACHE.popitem(last=False)
    return fsys


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
    surface_disorder_strength: float = 0.0,
    n_surface_layers: int = 2,
    progress_cb: Callable[..., None] | None = None,
    log_detail: str = "minimal",
    heartbeat_seconds: int = 20,
    kwant_mode: str = "auto",
    task_workers: int = 0,
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

    lead_axis = str(lead_axis).lower().strip()
    thickness_axis = str(thickness_axis).lower().strip()
    valid_axes = {"x", "y", "z"}
    if lead_axis not in valid_axes:
        raise ValueError(f"lead_axis must be one of {sorted(valid_axes)}, got {lead_axis!r}")
    if thickness_axis not in valid_axes:
        raise ValueError(
            f"thickness_axis must be one of {sorted(valid_axes)}, got {thickness_axis!r}"
        )

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

    requested_lead_cells = int(base_counts[lead_axis])
    required_lead_cells = requested_lead_cells
    for d in thicknesses:
        counts_probe = dict(base_counts)
        counts_probe[thickness_axis] = int(d)
        required_lead_cells = max(
            required_lead_cells,
            _required_lead_axis_cells(tb_model, counts=counts_probe, lead_axis=lead_axis),
        )

    G_means: list[float] = []
    G_stds: list[float] = []
    length_m_vals: list[float] = []
    thickness_m_vals: list[float] = []
    area_vals: list[float] = []
    thickness_uc_used: list[int] = []
    lead_axis_cells_used: list[int] = []

    if thickness_axis != lead_axis and requested_lead_cells < required_lead_cells:
        base_counts[lead_axis] = int(required_lead_cells)

    _comm, _rank, mpi_size = _mpi_context()
    requested_workers = int(task_workers if task_workers > 0 else n_jobs)
    point_workers = max(1, min(requested_workers, max(1, int(len(thicknesses)))))
    parallel_clean_points = (
        mpi_size == 1
        and point_workers > 1
        and int(n_ensemble) <= 1
        and float(disorder_strength) == 0.0
        and float(surface_disorder_strength) == 0.0
        and int(len(thicknesses)) > 1
    )

    def _eval_thickness_point(thickness_uc: int) -> tuple[dict[str, int], float, float]:
        counts = dict(base_counts)
        counts[thickness_axis] = int(thickness_uc)
        if thickness_axis == lead_axis:
            counts[lead_axis] = max(
                int(counts[lead_axis]),
                _required_lead_axis_cells(tb_model, counts=counts, lead_axis=lead_axis),
            )
        if any(counts[a] <= 0 for a in ("x", "y", "z")):
            raise ValueError(f"Invalid layer counts for d={thickness_uc}: {counts}")

        if parallel_clean_points:
            g_scalar = _single_conductance(
                tb_model=tb_model,
                n_layers_x=counts["x"],
                n_layers_y=counts["y"],
                n_layers_z=counts["z"],
                lead_axis=lead_axis,
                disorder_strength=0.0,
                energy=energy,
                seed=int(base_seed),
                lead_onsite_eV=lead_onsite_eV,
                surface_disorder_strength=0.0,
                n_surface_layers=n_surface_layers,
                use_clean_cache=True,
            )
            return counts, float(g_scalar), 0.0

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
            surface_disorder_strength=surface_disorder_strength,
            n_surface_layers=n_surface_layers,
            progress_cb=progress_cb,
            progress_context={
                "scan": "thickness",
                "disorder_strength": float(disorder_strength),
                "thickness_uc": int(counts[thickness_axis]),
            },
            log_detail=log_detail,
            heartbeat_seconds=heartbeat_seconds,
            kwant_mode=kwant_mode,
            task_workers=task_workers,
        )
        return counts, float(np.mean(G_vals)), float(np.std(G_vals))

    point_results: list[tuple[dict[str, int], float, float]] = []
    if parallel_clean_points:
        ordered: list[tuple[dict[str, int], float, float] | None] = [None] * len(thicknesses)
        with ThreadPoolExecutor(max_workers=point_workers) as ex:
            fut_to_idx = {
                ex.submit(_eval_thickness_point, int(d)): i for i, d in enumerate(thicknesses)
            }
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    ordered[idx] = fut.result()
                except Exception:
                    for pending in fut_to_idx:
                        pending.cancel()
                    raise
        for item in ordered:
            if item is None:
                raise RuntimeError("Missing thickness point result from task-parallel execution.")
            point_results.append(item)
    else:
        for d in thicknesses:
            point_results.append(_eval_thickness_point(int(d)))

    for counts, g_mean, g_std in point_results:
        G_means.append(float(g_mean))
        G_stds.append(float(g_std))

        geom = region_geometry(
            tb_model.lattice_vectors,
            n_layers_x=counts["x"],
            n_layers_y=counts["y"],
            n_layers_z=counts["z"],
            lead_axis=lead_axis,
            thickness_axis=thickness_axis,
        )
        thickness_uc_used.append(int(counts[thickness_axis]))
        lead_axis_cells_used.append(int(counts[lead_axis]))
        length_m_vals.append(geom["length_m"])
        thickness_m_vals.append(geom["thickness_m"])
        area_vals.append(cross_section_m2 if cross_section_m2 is not None else geom["cross_section_m2"])
        _emit_progress(
            progress_cb,
            "thickness_point_done",
            disorder_strength=float(disorder_strength),
            thickness_uc=int(counts[thickness_axis]),
            lead_axis_cells=int(counts[lead_axis]),
            G_mean=float(G_means[-1]),
            G_std=float(G_stds[-1]),
        )

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
        "thickness_uc": np.array(thickness_uc_used, dtype=int),
        "thickness_uc_requested": thicknesses,
        "thickness_m": thickness_m,
        "length_m": length_m,
        "cross_section_m2": cross_section_arr,
        "lead_axis": lead_axis,
        "thickness_axis": thickness_axis,
        "n_layers_x": int(base_counts["x"]),
        "n_layers_y": int(base_counts["y"]),
        "lead_axis_cells_requested": int(requested_lead_cells),
        "lead_axis_cells_used": np.array(lead_axis_cells_used, dtype=int),
        "lead_axis_min_cells_required": int(required_lead_cells),
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
    progress_cb: Callable[..., None] | None = None,
    log_detail: str = "minimal",
    heartbeat_seconds: int = 20,
    kwant_mode: str = "auto",
    task_workers: int = 0,
) -> dict:
    """Compute G(L) for MFP extraction.

    `lengths` is interpreted as the unit-cell count along `lead_axis`.
    """
    lead_axis = str(lead_axis).lower().strip()
    thickness_axis = str(thickness_axis).lower().strip()
    valid_axes = {"x", "y", "z"}
    if lead_axis not in valid_axes:
        raise ValueError(f"lead_axis must be one of {sorted(valid_axes)}, got {lead_axis!r}")
    if thickness_axis not in valid_axes:
        raise ValueError(
            f"thickness_axis must be one of {sorted(valid_axes)}, got {thickness_axis!r}"
        )

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

    required_lead_cells = _required_lead_axis_cells(
        tb_model,
        counts=base_counts,
        lead_axis=lead_axis,
    )
    requested_lengths = lengths.copy()
    lengths_used = lengths.copy()
    if int(np.min(lengths_used)) < int(required_lead_cells):
        shift = int(required_lead_cells) - int(np.min(lengths_used))
        lengths_used = lengths_used + shift

    G_means: list[float] = []
    G_stds: list[float] = []
    length_m_vals: list[float] = []
    cross_section_vals: list[float] = []

    _comm, _rank, mpi_size = _mpi_context()
    requested_workers = int(task_workers if task_workers > 0 else n_jobs)
    point_workers = max(1, min(requested_workers, max(1, int(len(lengths_used)))))
    parallel_clean_points = (
        mpi_size == 1
        and point_workers > 1
        and int(n_ensemble) <= 1
        and float(disorder_strength) == 0.0
        and int(len(lengths_used)) > 1
    )

    def _eval_length_point(length_uc: int) -> tuple[int, float, float]:
        counts = dict(base_counts)
        counts[lead_axis] = int(length_uc)

        if parallel_clean_points:
            g_scalar = _single_conductance(
                tb_model=tb_model,
                n_layers_x=counts["x"],
                n_layers_y=counts["y"],
                n_layers_z=counts["z"],
                lead_axis=lead_axis,
                disorder_strength=0.0,
                energy=energy,
                seed=int(base_seed),
                lead_onsite_eV=lead_onsite_eV,
                surface_disorder_strength=0.0,
                n_surface_layers=n_surface_layers,
                use_clean_cache=True,
            )
            return int(length_uc), float(g_scalar), 0.0

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
            surface_disorder_strength=0.0,
            n_surface_layers=2,
            progress_cb=progress_cb,
            progress_context={
                "scan": "length",
                "disorder_strength": float(disorder_strength),
                "length_uc": int(L),
            },
            log_detail=log_detail,
            heartbeat_seconds=heartbeat_seconds,
            kwant_mode=kwant_mode,
            task_workers=task_workers,
        )
        return int(length_uc), float(np.mean(G_vals)), float(np.std(G_vals))

    point_results: list[tuple[int, float, float]] = []
    if parallel_clean_points:
        ordered: list[tuple[int, float, float] | None] = [None] * len(lengths_used)
        with ThreadPoolExecutor(max_workers=point_workers) as ex:
            fut_to_idx = {
                ex.submit(_eval_length_point, int(L)): i for i, L in enumerate(lengths_used)
            }
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    ordered[idx] = fut.result()
                except Exception:
                    for pending in fut_to_idx:
                        pending.cancel()
                    raise
        for item in ordered:
            if item is None:
                raise RuntimeError("Missing length point result from task-parallel execution.")
            point_results.append(item)
    else:
        for L in lengths_used:
            point_results.append(_eval_length_point(int(L)))

    for length_uc, g_mean, g_std in point_results:
        counts = dict(base_counts)
        counts[lead_axis] = int(length_uc)
        G_means.append(float(g_mean))
        G_stds.append(float(g_std))
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
        _emit_progress(
            progress_cb,
            "length_point_done",
            disorder_strength=float(disorder_strength),
            length_uc=int(length_uc),
            G_mean=float(G_means[-1]),
            G_std=float(G_stds[-1]),
        )

    return {
        "length_uc": np.array(lengths_used, dtype=int),
        "length_uc_requested": np.array(requested_lengths, dtype=int),
        "length_m": np.array(length_m_vals, dtype=float),
        "cross_section_m2": np.array(cross_section_vals, dtype=float),
        "lead_axis": lead_axis,
        "thickness_axis": thickness_axis,
        "lead_axis_min_cells_required": int(required_lead_cells),
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
    surface_disorder_strength: float = 0.0,
    n_surface_layers: int = 2,
    progress_cb: Callable[..., None] | None = None,
    progress_context: dict[str, Any] | None = None,
    log_detail: str = "minimal",
    heartbeat_seconds: int = 20,
    kwant_mode: str = "auto",
    task_workers: int = 0,
) -> np.ndarray:
    """Run disorder ensemble and return array of G values (e²/h)."""
    comm, rank, size = _mpi_context()
    emit_samples = str(log_detail).strip().lower() == "per_ensemble"
    ctx = dict(progress_context or {})

    _no_disorder = disorder_strength == 0.0 and surface_disorder_strength == 0.0
    if n_ensemble <= 1 or _no_disorder:
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
                surface_disorder_strength=0.0,
                n_surface_layers=n_surface_layers,
                use_clean_cache=True,
            )
            if emit_samples:
                _emit_progress(
                    progress_cb,
                    "ensemble_sample_done",
                    rank=int(rank),
                    size=int(size),
                    ensemble_index=0,
                    seed=int(base_seed),
                    G=float(value),
                    **ctx,
                )
        if comm is not None:
            value = comm.bcast(value, root=0)
        return np.array([value], dtype=float)

    indices = list(range(n_ensemble))
    local_indices = indices[rank::size]

    local_results: list[tuple[int, float]] = []
    local_error: str | None = None

    def _calc_idx(idx: int) -> tuple[int, float]:
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
            surface_disorder_strength=surface_disorder_strength,
            n_surface_layers=n_surface_layers,
            use_clean_cache=False,
        )
        return (idx, float(g))

    if size == 1:
        workers = max(1, int(task_workers if task_workers > 0 else n_jobs))
        workers = min(workers, max(1, len(local_indices)))
        if workers <= 1 or len(local_indices) <= 1:
            for idx in local_indices:
                try:
                    idx_g, g = _calc_idx(idx)
                    local_results.append((idx_g, g))
                    if emit_samples:
                        _emit_progress(
                            progress_cb,
                            "ensemble_sample_done",
                            rank=int(rank),
                            size=int(size),
                            ensemble_index=int(idx),
                            seed=int(base_seed + idx),
                            G=float(g),
                            **ctx,
                        )
                except Exception as exc:
                    local_error = (
                        f"rank={rank}, idx={idx}, seed={base_seed + idx}, "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    break
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fut_to_idx = {ex.submit(_calc_idx, idx): idx for idx in local_indices}
                for fut in as_completed(fut_to_idx):
                    idx = fut_to_idx[fut]
                    try:
                        idx_g, g = fut.result()
                        local_results.append((idx_g, g))
                        if emit_samples:
                            _emit_progress(
                                progress_cb,
                                "ensemble_sample_done",
                                rank=int(rank),
                                size=int(size),
                                ensemble_index=int(idx_g),
                                seed=int(base_seed + idx_g),
                                G=float(g),
                                **ctx,
                            )
                    except Exception as exc:
                        local_error = (
                            f"rank={rank}, idx={idx}, seed={base_seed + idx}, "
                            f"error={type(exc).__name__}: {exc}"
                        )
                        for pending in fut_to_idx:
                            pending.cancel()
                        break

        if local_error:
            raise RuntimeError(local_error)
        local_results.sort(key=lambda x: x[0])
        return np.array([g for _, g in local_results], dtype=float)

    for idx in local_indices:
        try:
            idx_g, g = _calc_idx(idx)
            local_results.append((idx_g, g))
            if emit_samples:
                _emit_progress(
                    progress_cb,
                    "ensemble_sample_done",
                    rank=int(rank),
                    size=int(size),
                    ensemble_index=int(idx_g),
                    seed=int(base_seed + idx_g),
                    G=float(g),
                    **ctx,
                )
        except Exception as exc:  # pragma: no cover - runtime/HPC dependent
            local_error = (
                f"rank={rank}, idx={idx}, seed={base_seed + idx}, "
                f"error={type(exc).__name__}: {exc}"
            )
            break

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
    surface_disorder_strength: float = 0.0,
    n_surface_layers: int = 2,
    use_clean_cache: bool = False,
) -> float:
    """Build, disorder, finalize, and compute G for one configuration.

    Disorder model
    --------------
    When surface_disorder_strength > 0, surface-localized Anderson disorder is
    applied on the outermost n_surface_layers planes (z-tag based) with amplitude
    W_surface, and uniform disorder_strength is applied to the bulk interior.
    This reflects interface defects (e.g. SiO₂/TaP O-vacancies) that scatter
    Fermi-arc states far more efficiently than bulk dopants.
    """
    import kwant
    from wtec.transport.disorder import add_anderson_disorder, add_surface_anderson_disorder

    no_disorder = disorder_strength == 0.0 and surface_disorder_strength == 0.0
    if use_clean_cache and no_disorder:
        try:
            fsys = _get_or_build_clean_fsys(
                tb_model,
                n_layers_x=n_layers_x,
                n_layers_y=n_layers_y,
                n_layers_z=n_layers_z,
                lead_axis=lead_axis,
                lead_onsite_eV=lead_onsite_eV,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to build/cache clean Kwant transport system. "
                f"n_layers_x={n_layers_x}, n_layers_y={n_layers_y}, n_layers_z={n_layers_z}, "
                f"lead_axis={lead_axis}. Original error: {type(exc).__name__}: {exc}"
            ) from exc
    else:
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

        rng = np.random.default_rng(seed)
        if surface_disorder_strength > 0.0:
            add_surface_anderson_disorder(
                sys,
                surface_strength=surface_disorder_strength,
                bulk_strength=disorder_strength,
                n_surface_layers=n_surface_layers,
                rng=rng,
            )
        elif disorder_strength > 0.0:
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
