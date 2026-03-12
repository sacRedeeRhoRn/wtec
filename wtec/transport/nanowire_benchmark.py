from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from wtec.wannier.model import _parse_lattice_from_win
from wtec.wannier.parser import HoppingData, read_hr_dat, write_hr_dat


NANOWIRE_BENCHMARK_MP_ID = "mp-1018028"
NANOWIRE_BENCHMARK_MATERIAL = "TiS"
NANOWIRE_BENCHMARK_AXES = ("c",)
NANOWIRE_BENCHMARK_ENERGIES_EV = (-0.2, -0.1, 0.0, 0.1, 0.2)
NANOWIRE_BENCHMARK_THICKNESSES_UC = (1, 3, 5, 7, 9, 11, 13)
NANOWIRE_BENCHMARK_FIXED_WIDTH_UC = 13
NANOWIRE_BENCHMARK_TRIM_EXCLUDE = (1, 3)


@dataclass(frozen=True)
class BenchmarkModelSpec:
    key: str
    label: str
    custom_projections: tuple[str, ...]
    primary_for_rgf: bool = False


TIS_MODEL_A = BenchmarkModelSpec(
    key="model_a",
    label="Model A",
    custom_projections=("Ti:s;p;d", "S:s;p"),
    primary_for_rgf=False,
)

TIS_MODEL_B = BenchmarkModelSpec(
    key="model_b",
    label="Model B",
    custom_projections=("Ti:d", "S:p"),
    primary_for_rgf=True,
)


@dataclass(frozen=True)
class NanowireBenchmarkSpec:
    mp_id: str = NANOWIRE_BENCHMARK_MP_ID
    material: str = NANOWIRE_BENCHMARK_MATERIAL
    axes: tuple[str, ...] = NANOWIRE_BENCHMARK_AXES
    energies_ev: tuple[float, ...] = NANOWIRE_BENCHMARK_ENERGIES_EV
    thicknesses_uc: tuple[int, ...] = NANOWIRE_BENCHMARK_THICKNESSES_UC
    fixed_width_uc: int = NANOWIRE_BENCHMARK_FIXED_WIDTH_UC
    trim_exclude_thicknesses_uc: tuple[int, ...] = NANOWIRE_BENCHMARK_TRIM_EXCLUDE
    models: tuple[BenchmarkModelSpec, ...] = (TIS_MODEL_A, TIS_MODEL_B)
    abs_tol: float = 5.0e-3
    rel_tol: float = 5.0e-4
    zero_tol: float = 1.0e-10
    fit_r2_abs_tol: float = 1.0e-3
    min_length_uc: int = 24
    length_multiplier: int = 3
    length_padding_uc: int = 4


def select_benchmark_models(
    spec: NanowireBenchmarkSpec,
    *,
    include_supplementary: bool = False,
) -> tuple[BenchmarkModelSpec, ...]:
    """Return the benchmark models to execute for a transport run.

    By default we run only the primary RGF-bearing model so benchmark turns
    reach the transport evidence path promptly. Supplementary article models
    can still be requested explicitly.
    """
    models = tuple(spec.models)
    if include_supplementary:
        return models
    primary = tuple(model for model in models if bool(model.primary_for_rgf))
    return primary or models


@dataclass(frozen=True)
class CanonicalizedNanowireInput:
    axis: str
    hr_dat_path: str
    win_path: str
    permutation: tuple[int, int, int]
    lattice_vectors: list[list[float]]


@dataclass(frozen=True)
class BenchmarkComparison:
    status: str
    checked_points: int
    max_abs_err: float
    max_rel_err: float
    failures: list[dict[str, Any]]


@dataclass(frozen=True)
class BenchmarkFitComparison:
    status: str
    checked_rows: int
    max_abs_err: float
    max_rel_err: float
    max_r2_abs_err: float
    failures: list[dict[str, Any]]


def axis_permutation(axis: str) -> tuple[int, int, int]:
    key = str(axis or "").strip().lower()
    if key == "a":
        return (0, 1, 2)
    if key == "c":
        return (2, 0, 1)
    raise ValueError(f"Unsupported nanowire benchmark axis={axis!r}. Use 'a' or 'c'.")


def canonicalize_hopping_data(
    hd: HoppingData,
    lattice_vectors: np.ndarray,
    *,
    axis: str,
) -> tuple[HoppingData, np.ndarray, tuple[int, int, int]]:
    perm = axis_permutation(axis)
    lv = np.asarray(lattice_vectors, dtype=float)
    if lv.shape != (3, 3):
        raise ValueError(f"lattice_vectors must have shape (3,3), got {lv.shape}")
    rv = np.asarray(hd.r_vectors, dtype=int)
    rv2 = rv[:, perm]
    lv2 = lv[list(perm), :]
    hd2 = HoppingData(
        num_wann=int(hd.num_wann),
        r_vectors=rv2,
        deg=np.asarray(hd.deg, dtype=int).copy(),
        H_R=np.asarray(hd.H_R, dtype=np.complex128).copy(),
    )
    return hd2, lv2, perm


def write_minimal_win(path: str | Path, lattice_vectors: np.ndarray) -> Path:
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


def prepare_canonicalized_inputs(
    *,
    hr_dat_path: str | Path,
    win_path: str | Path,
    axis: str,
    out_dir: str | Path,
    seedname: str = "nanowire_benchmark",
) -> CanonicalizedNanowireInput:
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    hd = read_hr_dat(Path(hr_dat_path).expanduser().resolve())
    lv = _parse_lattice_from_win(win_path)
    hd2, lv2, perm = canonicalize_hopping_data(hd, lv, axis=axis)
    hr_out = out_root / f"{seedname}_{axis}_canonical_hr.dat"
    win_out = out_root / f"{seedname}_{axis}_canonical.win"
    write_hr_dat(hr_out, hd2, header=f"wtec canonicalized nanowire benchmark ({axis}-axis)")
    write_minimal_win(win_out, lv2)
    return CanonicalizedNanowireInput(
        axis=str(axis),
        hr_dat_path=str(hr_out),
        win_path=str(win_out),
        permutation=perm,
        lattice_vectors=[[float(v) for v in row] for row in lv2],
    )


def compute_length_uc(principal_layer_width: int, *, spec: NanowireBenchmarkSpec | None = None) -> int:
    cfg = spec or NanowireBenchmarkSpec()
    p_eff = max(1, int(principal_layer_width))
    return max(int(cfg.min_length_uc), int(cfg.length_multiplier) * p_eff + int(cfg.length_padding_uc))


def _records_by_thickness_energy(records: Iterable[dict[str, Any]]) -> dict[int, dict[float, float]]:
    out: dict[int, dict[float, float]] = {}
    for row in records:
        d = int(row["thickness_uc"])
        e = float(row["energy_rel_fermi_ev"])
        t = float(row["transmission_e2_over_h"])
        out.setdefault(d, {})[e] = t
    return out


def select_monotonic_thickness_subsequence(
    records: Iterable[dict[str, Any]],
    *,
    energies_ev: Iterable[float],
    candidate_thicknesses: Iterable[int],
    min_points: int = 4,
    max_transmission_e2_over_h: float | None = None,
    tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    energies = [float(v) for v in energies_ev]
    candidates = [int(v) for v in candidate_thicknesses]
    by_te = _records_by_thickness_energy(records)

    usable: list[int] = []
    dropped: list[dict[str, Any]] = []
    for d in candidates:
        row = by_te.get(int(d), {})
        missing = [e for e in energies if e not in row]
        if missing:
            dropped.append({"thickness_uc": int(d), "reason": "missing_energy", "missing": missing})
            continue
        vals = [float(row[e]) for e in energies]
        if max_transmission_e2_over_h is not None and any(
            (v < -tolerance) or (v > float(max_transmission_e2_over_h) + tolerance) for v in vals
        ):
            dropped.append({"thickness_uc": int(d), "reason": "out_of_range", "values": vals})
            continue
        usable.append(int(d))

    if not usable:
        return {
            "status": "failed",
            "reason": "no_usable_points",
            "retained_thicknesses": [],
            "dropped": dropped,
        }

    next_idx = [-1] * len(usable)
    best_len = [1] * len(usable)
    for i in range(len(usable) - 1, -1, -1):
        di = usable[i]
        vi = by_te[di]
        for j in range(i + 1, len(usable)):
            dj = usable[j]
            vj = by_te[dj]
            if all(float(vj[e]) <= float(vi[e]) + float(tolerance) for e in energies):
                cand_len = 1 + best_len[j]
                if cand_len > best_len[i]:
                    best_len[i] = cand_len
                    next_idx[i] = j

    start = max(range(len(usable)), key=lambda idx: (best_len[idx], -idx))
    retained: list[int] = []
    cur = start
    while cur >= 0:
        retained.append(int(usable[cur]))
        cur = next_idx[cur]

    if len(retained) < int(min_points):
        return {
            "status": "failed",
            "reason": "insufficient_monotonic_points",
            "retained_thicknesses": retained,
            "dropped": dropped,
            "min_points": int(min_points),
        }

    rows = []
    for d in retained:
        for e in energies:
            rows.append(
                {
                    "thickness_uc": int(d),
                    "energy_rel_fermi_ev": float(e),
                    "transmission_e2_over_h": float(by_te[d][e]),
                }
            )
    return {
        "status": "ok",
        "retained_thicknesses": retained,
        "dropped": dropped,
        "records": rows,
    }


def _is_monotonic_non_decreasing(xs: list[float], *, tolerance: float = 1.0e-8) -> bool:
    return all(float(xs[i + 1]) >= float(xs[i]) - float(tolerance) for i in range(len(xs) - 1))


def _linear_fit(thicknesses_uc: Iterable[int], transmissions: Iterable[float]) -> dict[str, Any]:
    x = np.asarray([float(v) for v in thicknesses_uc], dtype=float)
    y = np.asarray([float(v) for v in transmissions], dtype=float)
    if x.size != y.size:
        raise ValueError("thicknesses and transmissions must have the same length")
    if x.size < 2:
        raise ValueError("At least two points are required for a linear fit")
    slope, intercept = np.polyfit(x, y, deg=1)
    y_pred = slope * x + intercept
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    ss_res = float(np.sum((y - y_pred) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "slope_e2_over_h_per_uc": float(slope),
        "intercept_e2_over_h": float(intercept),
        "r_squared": float(r_squared),
        "n_points": int(x.size),
    }


def build_article_fit_summary(
    records: Iterable[dict[str, Any]],
    *,
    energies_ev: Iterable[float],
    thicknesses_uc: Iterable[int],
    trim_exclude_thicknesses_uc: Iterable[int],
    tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    energies = [float(v) for v in energies_ev]
    thicknesses = [int(v) for v in thicknesses_uc]
    trim_exclude = {int(v) for v in trim_exclude_thicknesses_uc}
    trimmed = [int(v) for v in thicknesses if int(v) not in trim_exclude]
    by_te = _records_by_thickness_energy(records)

    missing: list[dict[str, Any]] = []
    raw_by_energy: dict[float, list[dict[str, Any]]] = {}
    for e in energies:
        rows: list[dict[str, Any]] = []
        for d in thicknesses:
            val = by_te.get(int(d), {}).get(float(e))
            if val is None:
                missing.append({"thickness_uc": int(d), "energy_rel_fermi_ev": float(e)})
                continue
            rows.append({
                "thickness_uc": int(d),
                "energy_rel_fermi_ev": float(e),
                "transmission_e2_over_h": float(val),
            })
        raw_by_energy[float(e)] = rows

    if missing:
        return {
            "status": "failed",
            "reason": "missing_points",
            "missing": missing,
            "all_points": [],
            "trimmed_points": [],
        }

    all_points_rows: list[dict[str, Any]] = []
    trimmed_rows: list[dict[str, Any]] = []
    monotonic_all: list[dict[str, Any]] = []
    monotonic_trimmed: list[dict[str, Any]] = []
    r2_improves: list[dict[str, Any]] = []

    for e in energies:
        raw_rows = raw_by_energy[float(e)]
        raw_t = [float(row["transmission_e2_over_h"]) for row in raw_rows]
        fit_all = _linear_fit(thicknesses, raw_t)
        mono_all = _is_monotonic_non_decreasing(raw_t, tolerance=tolerance)
        all_points_rows.append(
            {
                "fit_kind": "all_points",
                "energy_rel_fermi_ev": float(e),
                "thicknesses_uc": [int(v) for v in thicknesses],
                "monotonic_non_decreasing": bool(mono_all),
                **fit_all,
            }
        )
        monotonic_all.append(
            {
                "energy_rel_fermi_ev": float(e),
                "monotonic_non_decreasing": bool(mono_all),
            }
        )

        trimmed_t = [float(by_te[int(d)][float(e)]) for d in trimmed]
        fit_trim = _linear_fit(trimmed, trimmed_t)
        mono_trim = _is_monotonic_non_decreasing(trimmed_t, tolerance=tolerance)
        trimmed_rows.append(
            {
                "fit_kind": "trimmed_points",
                "energy_rel_fermi_ev": float(e),
                "thicknesses_uc": [int(v) for v in trimmed],
                "excluded_thicknesses_uc": sorted(int(v) for v in trim_exclude),
                "monotonic_non_decreasing": bool(mono_trim),
                **fit_trim,
            }
        )
        monotonic_trimmed.append(
            {
                "energy_rel_fermi_ev": float(e),
                "monotonic_non_decreasing": bool(mono_trim),
            }
        )
        r2_improves.append(
            {
                "energy_rel_fermi_ev": float(e),
                "trimmed_r2_gte_all": bool(float(fit_trim["r_squared"]) >= float(fit_all["r_squared"]) - tolerance),
            }
        )

    return {
        "status": "ok",
        "all_points": all_points_rows,
        "trimmed_points": trimmed_rows,
        "checks": {
            "monotonic_all_points": monotonic_all,
            "monotonic_trimmed": monotonic_trimmed,
            "trimmed_r2_gte_all": r2_improves,
        },
    }


def compare_reference_and_rgf(
    reference_records: Iterable[dict[str, Any]],
    rgf_records: Iterable[dict[str, Any]],
    *,
    abs_tol: float = 5.0e-3,
    rel_tol: float = 5.0e-4,
    zero_tol: float = 1.0e-10,
) -> BenchmarkComparison:
    ref = _records_by_thickness_energy(reference_records)
    got = _records_by_thickness_energy(rgf_records)
    failures: list[dict[str, Any]] = []
    max_abs = 0.0
    max_rel = 0.0
    checked = 0
    for d, row in sorted(ref.items()):
        got_row = got.get(int(d), {})
        for e, ref_val in sorted(row.items()):
            if float(e) not in got_row:
                failures.append({"thickness_uc": int(d), "energy_rel_fermi_ev": float(e), "reason": "missing_rgf"})
                continue
            got_val = float(got_row[float(e)])
            ref_f = float(ref_val)
            abs_err = abs(got_val - ref_f)
            max_abs = max(max_abs, abs_err)
            if abs(ref_f) <= float(zero_tol):
                rel_err = 0.0 if abs_err <= float(zero_tol) else float("inf")
                passed = abs_err <= float(zero_tol)
            else:
                rel_err = abs_err / abs(ref_f)
                passed = abs_err <= float(abs_tol) and rel_err <= float(rel_tol)
            max_rel = max(max_rel, 0.0 if not np.isfinite(rel_err) else rel_err)
            checked += 1
            if not passed:
                failures.append(
                    {
                        "thickness_uc": int(d),
                        "energy_rel_fermi_ev": float(e),
                        "reference": ref_f,
                        "rgf": got_val,
                        "abs_err": float(abs_err),
                        "rel_err": None if not np.isfinite(rel_err) else float(rel_err),
                    }
                )
    return BenchmarkComparison(
        status="ok" if not failures else "failed",
        checked_points=int(checked),
        max_abs_err=float(max_abs),
        max_rel_err=float(max_rel),
        failures=failures,
    )


def compare_fit_summaries(
    reference_fit: dict[str, Any],
    rgf_fit: dict[str, Any],
    *,
    abs_tol: float = 5.0e-3,
    rel_tol: float = 5.0e-4,
    zero_tol: float = 1.0e-10,
    r2_abs_tol: float = 1.0e-3,
) -> BenchmarkFitComparison:
    failures: list[dict[str, Any]] = []
    max_abs = 0.0
    max_rel = 0.0
    max_r2_abs = 0.0
    checked = 0

    def _index(rows: list[dict[str, Any]]) -> dict[tuple[str, float], dict[str, Any]]:
        out: dict[tuple[str, float], dict[str, Any]] = {}
        for row in rows:
            out[(str(row["fit_kind"]), float(row["energy_rel_fermi_ev"]))] = row
        return out

    ref_rows = _index(list(reference_fit.get("all_points", [])) + list(reference_fit.get("trimmed_points", [])))
    got_rows = _index(list(rgf_fit.get("all_points", [])) + list(rgf_fit.get("trimmed_points", [])))

    for key, ref_row in sorted(ref_rows.items()):
        got_row = got_rows.get(key)
        if got_row is None:
            failures.append(
                {
                    "fit_kind": str(key[0]),
                    "energy_rel_fermi_ev": float(key[1]),
                    "reason": "missing_rgf_fit",
                }
            )
            continue
        for metric in ("intercept_e2_over_h", "slope_e2_over_h_per_uc"):
            ref_val = float(ref_row[metric])
            got_val = float(got_row[metric])
            abs_err = abs(got_val - ref_val)
            max_abs = max(max_abs, abs_err)
            if abs(ref_val) <= float(zero_tol):
                rel_err = 0.0 if abs_err <= float(zero_tol) else float("inf")
                passed = abs_err <= float(zero_tol)
            else:
                rel_err = abs_err / abs(ref_val)
                passed = abs_err <= float(abs_tol) and rel_err <= float(rel_tol)
            max_rel = max(max_rel, 0.0 if not np.isfinite(rel_err) else rel_err)
            checked += 1
            if not passed:
                failures.append(
                    {
                        "fit_kind": str(key[0]),
                        "energy_rel_fermi_ev": float(key[1]),
                        "metric": metric,
                        "reference": ref_val,
                        "rgf": got_val,
                        "abs_err": float(abs_err),
                        "rel_err": None if not np.isfinite(rel_err) else float(rel_err),
                    }
                )
        r2_ref = float(ref_row["r_squared"])
        r2_got = float(got_row["r_squared"])
        r2_abs = abs(r2_got - r2_ref)
        max_r2_abs = max(max_r2_abs, r2_abs)
        checked += 1
        if r2_abs > float(r2_abs_tol):
            failures.append(
                {
                    "fit_kind": str(key[0]),
                    "energy_rel_fermi_ev": float(key[1]),
                    "metric": "r_squared",
                    "reference": r2_ref,
                    "rgf": r2_got,
                    "abs_err": float(r2_abs),
                }
            )
    return BenchmarkFitComparison(
        status="ok" if not failures else "failed",
        checked_rows=int(checked),
        max_abs_err=float(max_abs),
        max_rel_err=float(max_rel),
        max_r2_abs_err=float(max_r2_abs),
        failures=failures,
    )


def rows_to_csv_lines(rows: Iterable[dict[str, Any]]) -> list[str]:
    out = ["thickness_uc,energy_rel_fermi_ev,energy_abs_ev,transmission_e2_over_h"]
    for row in rows:
        out.append(
            f"{int(row['thickness_uc'])},{float(row['energy_rel_fermi_ev']):.6f},{float(row.get('energy_abs_ev', row['energy_rel_fermi_ev'])):.6f},{float(row['transmission_e2_over_h']):.12f}"
        )
    return out


def fit_rows_to_csv_lines(summary: dict[str, Any]) -> list[str]:
    out = [
        "fit_kind,energy_rel_fermi_ev,intercept_e2_over_h,slope_e2_over_h_per_uc,r_squared,n_points,thicknesses_uc"
    ]
    for key in ("all_points", "trimmed_points"):
        for row in summary.get(key, []):
            out.append(
                ",".join(
                    [
                        str(row["fit_kind"]),
                        f"{float(row['energy_rel_fermi_ev']):.6f}",
                        f"{float(row['intercept_e2_over_h']):.12f}",
                        f"{float(row['slope_e2_over_h_per_uc']):.12f}",
                        f"{float(row['r_squared']):.12f}",
                        str(int(row['n_points'])),
                        '"' + " ".join(str(int(v)) for v in row.get("thicknesses_uc", [])) + '"',
                    ]
                )
            )
    return out
