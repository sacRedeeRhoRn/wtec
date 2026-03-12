"""RGF transport helper utilities.

This module contains Python-side helpers for the native RGF transport engine:

- accepted engine and mode names
- HR-derived sizing helpers for preflight
- basic memory and work-unit estimation

The numerical transport solver itself is implemented separately in the native
RGF scaffold under ``wtec/ext/rgf``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from wtec.wannier.model import _parse_lattice_from_win
from wtec.wannier.parser import HoppingData, read_hr_dat, write_hr_dat

RGF_ENGINES = {"kwant", "rgf", "auto"}
RGF_MODES = {"periodic_transverse", "full_finite"}
RGF_DEFAULT_MODE = "periodic_transverse"
RGF_DEFAULT_PERIODIC_AXIS = "y"
RGF_WORKSPACE_MATS = 13
RGF_BINARY_ID = "wtec_rgf_runner_phase2_v4"
RGF_PARALLEL_POLICIES = {"auto", "single_point", "throughput"}
RGF_BLAS_BACKENDS = {"auto", "mkl", "openblas"}
RGF_VALIDATE_AGAINST = {"none", "kwant"}
RGF_JSON_EPS = 1.0e-12


@dataclass(frozen=True)
class RGFPreflightSummary:
    n_orb: int
    principal_layer_width: int
    superslice_dim: int
    per_rank_bytes: int
    transport_task_count: int
    thickness_task_count: int
    length_task_count: int
    periodic_k_count: int
    queue_cores: int
    safe_rank_cap: int
    mpi_np: int
    mode: str
    periodic_axis: str | None

    @property
    def n_super(self) -> int:
        return int(self.superslice_dim)

    @property
    def n_work_units(self) -> int:
        return int(self.transport_task_count)

    @property
    def task_shape(self) -> dict[str, int]:
        return {
            "periodic_k_count": int(self.periodic_k_count),
            "thickness_tasks": int(self.thickness_task_count),
            "length_tasks": int(self.length_task_count),
            "transport_task_count": int(self.transport_task_count),
        }


@dataclass(frozen=True)
class RGFExecutionPlan:
    parallel_policy: str
    mpi_np: int
    omp_threads: int
    full_node_threading: bool
    task_shape: dict[str, int]
    queue_cores: int
    safe_rank_cap: int
    transport_task_count: int


@dataclass(frozen=True)
class RGFCanonicalizedInput:
    hr_dat_path: str
    win_path: str | None
    permutation: tuple[int, int, int]
    width_axis: str
    lead_axis_original: str
    thickness_axis_original: str
    periodic_axis_original: str | None
    was_canonicalized: bool


def normalize_transport_engine(raw: Any) -> str:
    engine = str(raw or "kwant").strip().lower() or "kwant"
    if engine not in RGF_ENGINES:
        raise ValueError(f"Unsupported transport_engine={engine!r}.")
    return engine


def normalize_rgf_parallel_policy(raw: Any) -> str:
    policy = str(raw or "auto").strip().lower() or "auto"
    if policy not in RGF_PARALLEL_POLICIES:
        raise ValueError(
            "transport_rgf_parallel_policy must be one of "
            "['auto','single_point','throughput']."
        )
    return policy


def normalize_rgf_blas_backend(raw: Any) -> str:
    backend = str(raw or "auto").strip().lower() or "auto"
    if backend not in RGF_BLAS_BACKENDS:
        raise ValueError(
            "transport_rgf_blas_backend must be one of ['auto','mkl','openblas']."
        )
    return backend


def normalize_rgf_validate_against(raw: Any) -> str:
    mode = str(raw or "none").strip().lower() or "none"
    if mode not in RGF_VALIDATE_AGAINST:
        raise ValueError(
            "transport_rgf_validate_against must be one of ['none','kwant']."
        )
    return mode


def cluster_router_status(init_state: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(init_state, Mapping):
        return {}
    rgf_root = init_state.get("rgf")
    if not isinstance(rgf_root, Mapping):
        return {}
    cluster = rgf_root.get("cluster")
    return dict(cluster) if isinstance(cluster, Mapping) else {}


def _resolved_width_axis(*, lead_axis: str, thickness_axis: str) -> str:
    lead_axis_norm = normalize_axis(lead_axis, field_name="lead_axis")
    thickness_axis_norm = normalize_axis(thickness_axis, field_name="thickness_axis")
    if lead_axis_norm == thickness_axis_norm:
        raise ValueError("transport_axis and thickness_axis must differ for RGF.")
    for axis in ("x", "y", "z"):
        if axis not in {lead_axis_norm, thickness_axis_norm}:
            return axis
    raise ValueError("Failed to resolve the remaining transverse axis for RGF.")


def canonical_axis_permutation(
    *,
    lead_axis: str,
    thickness_axis: str,
    mode: str,
    periodic_axis: str | None = None,
) -> tuple[tuple[int, int, int], str]:
    lead_axis_norm = normalize_axis(lead_axis, field_name="lead_axis")
    thickness_axis_norm = normalize_axis(thickness_axis, field_name="thickness_axis")
    mode_norm = normalize_rgf_mode(mode)
    width_axis = _resolved_width_axis(
        lead_axis=lead_axis_norm,
        thickness_axis=thickness_axis_norm,
    )
    periodic_axis_norm = None
    if mode_norm == "periodic_transverse":
        periodic_axis_norm = normalize_axis(
            periodic_axis if periodic_axis is not None else width_axis,
            field_name="periodic_axis",
        )
        if periodic_axis_norm in {lead_axis_norm, thickness_axis_norm}:
            raise ValueError(
                "transport_rgf_periodic_axis must differ from transport_axis and thickness_axis."
            )
        width_axis = periodic_axis_norm
    order = (lead_axis_norm, width_axis, thickness_axis_norm)
    if len(set(order)) != 3:
        raise ValueError("RGF canonical axis order must contain three distinct axes.")
    axis_index = {"x": 0, "y": 1, "z": 2}
    return tuple(axis_index[axis] for axis in order), width_axis


def _write_minimal_win(path: str | Path, lattice_vectors: np.ndarray) -> Path:
    out = Path(path).expanduser().resolve()
    lv = np.asarray(lattice_vectors, dtype=float)
    lines = [
        "begin unit_cell_cart",
        "ang",
        *(f"  {float(row[0]):.12f}  {float(row[1]):.12f}  {float(row[2]):.12f}" for row in lv),
        "end unit_cell_cart",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def canonicalize_rgf_inputs(
    *,
    hr_dat_path: str | Path,
    win_path: str | Path | None,
    lead_axis: str,
    thickness_axis: str,
    mode: str,
    periodic_axis: str | None,
    out_dir: str | Path,
    seedname: str = "transport",
) -> RGFCanonicalizedInput:
    hr_in = Path(hr_dat_path).expanduser().resolve()
    win_in = Path(win_path).expanduser().resolve() if win_path is not None else None
    perm, width_axis = canonical_axis_permutation(
        lead_axis=lead_axis,
        thickness_axis=thickness_axis,
        mode=mode,
        periodic_axis=periodic_axis,
    )
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if perm == (0, 1, 2):
        return RGFCanonicalizedInput(
            hr_dat_path=str(hr_in),
            win_path=str(win_in) if win_in is not None and win_in.exists() else None,
            permutation=perm,
            width_axis=width_axis,
            lead_axis_original=normalize_axis(lead_axis, field_name="lead_axis"),
            thickness_axis_original=normalize_axis(
                thickness_axis,
                field_name="thickness_axis",
            ),
            periodic_axis_original=(
                normalize_axis(periodic_axis, field_name="periodic_axis")
                if periodic_axis is not None
                else None
            ),
            was_canonicalized=False,
        )
    hd = read_hr_dat(hr_in)
    rv = np.asarray(hd.r_vectors, dtype=int)
    hd2 = HoppingData(
        num_wann=int(hd.num_wann),
        r_vectors=rv[:, perm],
        deg=np.asarray(hd.deg, dtype=int).copy(),
        H_R=np.asarray(hd.H_R, dtype=np.complex128).copy(),
    )
    hr_out = out_root / f"{seedname}_canonical_hr.dat"
    write_hr_dat(
        hr_out,
        hd2,
        header=(
            "wtec canonicalized RGF input "
            f"(lead={lead_axis}, width={width_axis}, thickness={thickness_axis})"
        ),
    )
    win_out: Path | None = None
    if win_in is not None and win_in.exists():
        lv = _parse_lattice_from_win(win_in)
        lv2 = np.asarray(lv, dtype=float)[list(perm), :]
        win_out = out_root / f"{seedname}_canonical.win"
        _write_minimal_win(win_out, lv2)
    return RGFCanonicalizedInput(
        hr_dat_path=str(hr_out),
        win_path=str(win_out) if win_out is not None else None,
        permutation=perm,
        width_axis=width_axis,
        lead_axis_original=normalize_axis(lead_axis, field_name="lead_axis"),
        thickness_axis_original=normalize_axis(
            thickness_axis,
            field_name="thickness_axis",
        ),
        periodic_axis_original=(
            normalize_axis(periodic_axis, field_name="periodic_axis")
            if periodic_axis is not None
            else None
        ),
        was_canonicalized=True,
    )


def rgf_can_execute_for_config(
    cfg: Mapping[str, Any],
    *,
    init_state: Mapping[str, Any] | None = None,
    backend: str = "qsub",
    mode: str | None = None,
    periodic_axis: str | None = None,
) -> bool:
    if str(backend or "qsub").strip().lower() != "qsub":
        return False
    router = cluster_router_status(init_state)
    if not router or not bool(router.get("ready")):
        return False
    binary_id = str(router.get("binary_id") or "").strip()
    if binary_id and not (
        binary_id == RGF_BINARY_ID or binary_id.startswith("wtec_rgf_runner_phase2_v")
    ):
        return False
    numerical_status = str(router.get("numerical_status") or "scaffold_only").strip().lower()
    if numerical_status not in {"phase1_ready", "phase2_experimental", "phase2_ready"}:
        return False

    mode_norm = normalize_rgf_mode(mode if mode is not None else cfg.get("transport_rgf_mode"))
    transport_axis = normalize_axis(cfg.get("transport_axis", "x"), field_name="transport_axis")
    thickness_axis = normalize_axis(cfg.get("thickness_axis", "z"), field_name="thickness_axis")
    if transport_axis == thickness_axis:
        return False
    if mode_norm == "periodic_transverse":
        periodic_axis_norm = normalize_axis(
            periodic_axis if periodic_axis is not None else cfg.get("transport_rgf_periodic_axis", "y"),
            field_name="transport_rgf_periodic_axis",
        )
        return periodic_axis_norm not in {transport_axis, thickness_axis}
    validate_against = normalize_rgf_validate_against(
        cfg.get("transport_rgf_validate_against", "none")
    )
    if validate_against == "kwant":
        return (
            numerical_status in {"phase2_experimental", "phase2_ready"}
        )
    return numerical_status in {"phase2_experimental", "phase2_ready"}


def resolve_transport_engine(
    requested: Any,
    *,
    cfg: Mapping[str, Any] | None = None,
    init_state: Mapping[str, Any] | None = None,
    backend: str = "qsub",
) -> str:
    engine = normalize_transport_engine(requested)
    if engine != "auto":
        return engine
    if cfg is not None and rgf_can_execute_for_config(cfg, init_state=init_state, backend=backend):
        return "rgf"
    return "kwant"


def normalize_rgf_mode(raw: Any) -> str:
    mode = str(raw or RGF_DEFAULT_MODE).strip().lower() or RGF_DEFAULT_MODE
    if mode not in RGF_MODES:
        raise ValueError(f"Unsupported transport_rgf_mode={mode!r}.")
    return mode


def normalize_axis(raw: Any, *, field_name: str) -> str:
    axis = str(raw or "").strip().lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"{field_name} must be one of ['x','y','z'], got {raw!r}")
    return axis


def effective_ensemble_count(disorder_strength: float, n_ensemble: int) -> int:
    if abs(float(disorder_strength)) <= RGF_JSON_EPS:
        return 1
    return max(1, int(n_ensemble))


def mfp_disorder_index(disorder_strengths: list[float] | tuple[float, ...]) -> int:
    strengths = [float(v) for v in list(disorder_strengths or [0.0])]
    return max(0, min(len(strengths) - 1, len(strengths) // 2))


def work_unit_shape(
    *,
    thicknesses: list[int] | tuple[int, ...],
    mfp_lengths: list[int] | tuple[int, ...] | None = None,
    disorder_strengths: list[float] | tuple[float, ...],
    n_ensemble: int,
    mode: str = RGF_DEFAULT_MODE,
    periodic_k_count: int | None = None,
) -> dict[str, int]:
    strengths = [float(v) for v in list(disorder_strengths or [0.0])]
    n_thick = max(1, len(list(thicknesses)))
    n_mfp = len(list(mfp_lengths or []))
    sector_count = (
        max(1, int(periodic_k_count or 1))
        if normalize_rgf_mode(mode) == "periodic_transverse"
        else 1
    )
    thickness_tasks = n_thick * sum(
        effective_ensemble_count(strength, n_ensemble) for strength in strengths
    )
    length_tasks = 0
    if n_mfp > 0:
        mid = strengths[mfp_disorder_index(strengths)]
        length_tasks = n_mfp * effective_ensemble_count(mid, n_ensemble)
    return {
        "periodic_k_count": int(sector_count),
        "n_thickness": int(n_thick),
        "n_mfp": int(n_mfp),
        "n_disorder": int(len(strengths)),
        "thickness_tasks": int(thickness_tasks * sector_count),
        "length_tasks": int(length_tasks * sector_count),
        "transport_task_count": int((thickness_tasks + length_tasks) * sector_count),
    }


def load_hr_metadata(hr_dat_path: str | Path) -> HoppingData:
    return read_hr_dat(Path(hr_dat_path).expanduser().resolve())


def effective_principal_layer_width(
    hd: HoppingData,
    *,
    lead_axis: str,
    n_layers_x: int,
    n_layers_y: int,
    n_layers_z: int,
    mode: str = RGF_DEFAULT_MODE,
    periodic_axis: str | None = None,
) -> int:
    lead_axis_norm = normalize_axis(lead_axis, field_name="lead_axis")
    mode_norm = normalize_rgf_mode(mode)
    periodic_axis_norm = None
    if periodic_axis is not None:
        periodic_axis_norm = normalize_axis(periodic_axis, field_name="periodic_axis")

    shape = {
        "x": int(n_layers_x),
        "y": int(n_layers_y),
        "z": int(n_layers_z),
    }
    if any(v <= 0 for v in shape.values()):
        raise ValueError("n_layers_x, n_layers_y and n_layers_z must all be > 0")

    finite_axes = []
    for axis in ("x", "y", "z"):
        if axis == lead_axis_norm:
            continue
        if mode_norm == "periodic_transverse" and periodic_axis_norm == axis:
            continue
        finite_axes.append(axis)

    axis_index = {"x": 0, "y": 1, "z": 2}
    lead_idx = axis_index[lead_axis_norm]
    max_abs_span = 0
    for r_vec in hd.r_vectors:
        dx, dy, dz = (int(r_vec[0]), int(r_vec[1]), int(r_vec[2]))
        dR = {"x": dx, "y": dy, "z": dz}
        if any(abs(dR[axis]) > (shape[axis] - 1) for axis in finite_axes):
            continue
        max_abs_span = max(max_abs_span, abs((dx, dy, dz)[lead_idx]))
    return max(1, int(max_abs_span))


def superslice_dim_from_geometry(
    *,
    n_orb: int,
    principal_layer_width: int,
    lead_axis: str,
    n_layers_x: int,
    n_layers_y: int,
    n_layers_z: int,
    mode: str = RGF_DEFAULT_MODE,
    periodic_axis: str | None = None,
) -> int:
    lead_axis_norm = normalize_axis(lead_axis, field_name="lead_axis")
    mode_norm = normalize_rgf_mode(mode)
    periodic_axis_norm = None
    if periodic_axis is not None:
        periodic_axis_norm = normalize_axis(periodic_axis, field_name="periodic_axis")

    dims = {"x": int(n_layers_x), "y": int(n_layers_y), "z": int(n_layers_z)}
    cross = 1
    for axis in ("x", "y", "z"):
        if axis == lead_axis_norm:
            continue
        if mode_norm == "periodic_transverse" and periodic_axis_norm == axis:
            continue
        cross *= dims[axis]
    return int(max(1, principal_layer_width) * max(1, cross) * max(1, int(n_orb)))


def n_super_from_geometry(**kwargs: Any) -> int:
    return superslice_dim_from_geometry(**kwargs)


def memory_per_rank_bytes(
    *,
    superslice_dim: int | None = None,
    n_super: int | None = None,
    workspace_mats: int = RGF_WORKSPACE_MATS,
    overhead_bytes: int = 1 << 20,
) -> int:
    dim = superslice_dim if superslice_dim is not None else n_super
    n = int(max(1, int(dim or 1)))
    mats = int(max(1, workspace_mats))
    return int(mats * n * n * 16 + overhead_bytes)


def work_unit_count(
    *,
    thicknesses: list[int] | tuple[int, ...],
    mfp_lengths: list[int] | tuple[int, ...] | None = None,
    disorder_strengths: list[float] | tuple[float, ...],
    n_ensemble: int,
    mode: str = RGF_DEFAULT_MODE,
    periodic_k_count: int | None = None,
) -> int:
    return int(
        work_unit_shape(
            thicknesses=thicknesses,
            mfp_lengths=mfp_lengths,
            disorder_strengths=disorder_strengths,
            n_ensemble=n_ensemble,
            mode=mode,
            periodic_k_count=periodic_k_count,
        )["transport_task_count"]
    )


def resolved_mpi_np(
    *,
    queue_cores: int,
    n_work_units: int,
    safe_rank_cap: int,
) -> int:
    return max(1, min(int(queue_cores), int(n_work_units), int(safe_rank_cap)))


def plan_execution(
    *,
    mode: str,
    queue_cores: int,
    safe_rank_cap: int,
    n_work_units: int,
    requested_mpi_np: int = 0,
    requested_threads_per_rank: int | str = 0,
    parallel_policy: str = "auto",
) -> RGFExecutionPlan:
    mode_norm = normalize_rgf_mode(mode)
    policy_norm = normalize_rgf_parallel_policy(parallel_policy)
    queue_cores_i = max(1, int(queue_cores))
    safe_rank_cap_i = max(1, int(safe_rank_cap))
    n_work_units_i = max(1, int(n_work_units))
    requested_mpi_i = max(0, int(requested_mpi_np))
    requested_threads_raw = requested_threads_per_rank
    if isinstance(requested_threads_raw, str) and str(requested_threads_raw).strip().lower() == "auto":
        requested_threads_i = 0
    else:
        requested_threads_i = max(0, int(requested_threads_raw or 0))
    if policy_norm == "auto":
        policy_norm = (
            "single_point"
            if mode_norm == "full_finite" and n_work_units_i <= 1
            else "throughput"
        )
    mpi_auto = (
        1
        if policy_norm == "single_point"
        else resolved_mpi_np(
            queue_cores=queue_cores_i,
            n_work_units=n_work_units_i,
            safe_rank_cap=safe_rank_cap_i,
        )
    )
    if requested_mpi_i > 0:
        mpi_np = max(
            1,
            min(queue_cores_i, safe_rank_cap_i, n_work_units_i, requested_mpi_i),
        )
    else:
        mpi_np = mpi_auto
    max_threads_per_rank = max(1, queue_cores_i // max(1, mpi_np))
    omp_auto = queue_cores_i if policy_norm == "single_point" else max_threads_per_rank
    omp_threads = (
        min(max_threads_per_rank, requested_threads_i)
        if requested_threads_i > 0
        else omp_auto
    )
    return RGFExecutionPlan(
        parallel_policy=policy_norm,
        mpi_np=int(mpi_np),
        omp_threads=max(1, int(omp_threads)),
        full_node_threading=bool(int(mpi_np) == 1 and int(omp_threads) >= 1),
        task_shape={
            "transport_task_count": int(n_work_units_i),
        },
        queue_cores=int(queue_cores_i),
        safe_rank_cap=int(safe_rank_cap_i),
        transport_task_count=int(n_work_units_i),
    )


def preflight_summary(
    *,
    hr_dat_path: str | Path,
    lead_axis: str,
    n_layers_x: int,
    n_layers_y: int,
    n_layers_z: int,
    mode: str,
    periodic_axis: str | None,
    thicknesses: list[int] | tuple[int, ...],
    mfp_lengths: list[int] | tuple[int, ...] | None = None,
    disorder_strengths: list[float] | tuple[float, ...],
    n_ensemble: int,
    queue_cores: int,
    node_ram_bytes: int | None = None,
    safe_fraction: float = 0.85,
) -> RGFPreflightSummary:
    hd = load_hr_metadata(hr_dat_path)
    p_eff = effective_principal_layer_width(
        hd,
        lead_axis=lead_axis,
        n_layers_x=n_layers_x,
        n_layers_y=n_layers_y,
        n_layers_z=n_layers_z,
        mode=mode,
        periodic_axis=periodic_axis,
    )
    superslice_dim = superslice_dim_from_geometry(
        n_orb=int(hd.num_wann),
        principal_layer_width=p_eff,
        lead_axis=lead_axis,
        n_layers_x=n_layers_x,
        n_layers_y=n_layers_y,
        n_layers_z=n_layers_z,
        mode=mode,
        periodic_axis=periodic_axis,
    )
    per_rank = memory_per_rank_bytes(superslice_dim=superslice_dim)
    if node_ram_bytes is None or int(node_ram_bytes) <= 0:
        safe_rank_cap = int(max(1, queue_cores))
    else:
        safe_rank_cap = max(1, int((float(safe_fraction) * int(node_ram_bytes)) // per_rank))
    periodic_k_count = n_layers_y if normalize_rgf_mode(mode) == "periodic_transverse" else None
    shape = work_unit_shape(
        thicknesses=thicknesses,
        mfp_lengths=mfp_lengths,
        disorder_strengths=disorder_strengths,
        n_ensemble=n_ensemble,
        mode=mode,
        periodic_k_count=periodic_k_count,
    )
    n_work = int(shape["transport_task_count"])
    mpi_np = resolved_mpi_np(
        queue_cores=queue_cores,
        n_work_units=n_work,
        safe_rank_cap=safe_rank_cap,
    )
    return RGFPreflightSummary(
        n_orb=int(hd.num_wann),
        principal_layer_width=int(p_eff),
        superslice_dim=int(superslice_dim),
        per_rank_bytes=int(per_rank),
        transport_task_count=int(n_work),
        thickness_task_count=int(shape["thickness_tasks"]),
        length_task_count=int(shape["length_tasks"]),
        periodic_k_count=int(shape["periodic_k_count"]),
        queue_cores=int(queue_cores),
        safe_rank_cap=int(safe_rank_cap),
        mpi_np=int(mpi_np),
        mode=normalize_rgf_mode(mode),
        periodic_axis=(periodic_axis if periodic_axis else None),
    )


def phase1_alignment_issues(
    *,
    hr_dat_path: str | Path,
    lead_axis: str,
    n_layers_x: int,
    n_layers_y: int,
    thicknesses: list[int] | tuple[int, ...],
    mfp_n_layers_z: int,
    mfp_lengths: list[int] | tuple[int, ...] | None,
    mode: str,
    periodic_axis: str | None,
) -> list[str]:
    hd = load_hr_metadata(hr_dat_path)
    issues: list[str] = []

    if normalize_rgf_mode(mode) != "periodic_transverse":
        return issues

    for nz in [int(v) for v in thicknesses]:
        p_eff = effective_principal_layer_width(
            hd,
            lead_axis=lead_axis,
            n_layers_x=n_layers_x,
            n_layers_y=n_layers_y,
            n_layers_z=nz,
            mode=mode,
            periodic_axis=periodic_axis,
        )
        if int(n_layers_x) % int(p_eff) != 0:
            issues.append(
                f"transport_n_layers_x={int(n_layers_x)} is not divisible by principal_layer_width={int(p_eff)} "
                f"for thickness_uc={int(nz)} in phase-1 RGF periodic_transverse mode."
            )

    nz_mfp = int(mfp_n_layers_z)
    if mfp_lengths:
        p_eff_mfp = effective_principal_layer_width(
            hd,
            lead_axis=lead_axis,
            n_layers_x=max([int(v) for v in mfp_lengths] + [int(n_layers_x)]),
            n_layers_y=n_layers_y,
            n_layers_z=nz_mfp,
            mode=mode,
            periodic_axis=periodic_axis,
        )
        for nx in [int(v) for v in mfp_lengths]:
            if nx % int(p_eff_mfp) != 0:
                issues.append(
                    f"mfp_length_uc={int(nx)} is not divisible by principal_layer_width={int(p_eff_mfp)} "
                    f"for mfp_n_layers_z={nz_mfp} in phase-1 RGF periodic_transverse mode."
                )

    return issues
