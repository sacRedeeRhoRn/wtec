"""Postprocess native RGF transport output into the standard wtec result shape."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from wtec.transport.geometry import region_geometry
from wtec.transport.mfp import extract_mfp_from_scaling, mfp_from_sigma
from wtec.wannier.model import _parse_lattice_from_win


E2_OVER_H_SI = 7.748091729e-5


def _as_float_array(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _as_int_array(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=int)
    arr = np.asarray(values, dtype=int)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _as_float_matrix(values: Any) -> np.ndarray:
    if values is None:
        return np.zeros((0, 0), dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    raise RuntimeError("RGF matrix payload must be 1D or 2D.")


def load_rgf_raw_result(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {}
    try:
        payload = __import__("json").loads(Path(path).read_text())
    except Exception as exc:  # pragma: no cover - error message path
        raise RuntimeError(f"Failed to parse RGF result file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("RGF result payload must be a JSON object.")
    raw = payload.get("transport_results_raw")
    if not isinstance(raw, dict):
        raise RuntimeError("RGF result payload is missing transport_results_raw.")
    if "thickness_slice_count" not in raw and "thickness_n_super" in raw:
        raw["thickness_slice_count"] = raw.get("thickness_n_super")
    if "length_slice_count" not in raw and "length_n_super" in raw:
        raw["length_slice_count"] = raw.get("length_n_super")
    runtime_cert = payload.get("runtime_cert")
    if not isinstance(runtime_cert, dict):
        runtime_cert = {}
    if "max_slice_count" not in runtime_cert and "n_super" in runtime_cert:
        runtime_cert["max_slice_count"] = runtime_cert.get("n_super")
    return raw, runtime_cert


def convert_rgf_raw_to_transport_results(
    raw: dict[str, Any],
    *,
    win_path: str | Path | None,
    disorder_key: float = 0.0,
    carrier_density_m3: float | None = None,
    fermi_velocity_m_per_s: float | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("raw must be a dict")

    lattice_vectors = (
        _parse_lattice_from_win(win_path)
        if win_path is not None and Path(win_path).exists()
        else np.eye(3, dtype=float)
    )

    lead_axis = str(raw.get("lead_axis", "x")).strip().lower() or "x"
    thickness_axis = str(raw.get("thickness_axis", "z")).strip().lower() or "z"
    n_layers_x = int(raw.get("n_layers_x", 1))
    n_layers_y = int(raw.get("n_layers_y", 1))
    mfp_n_layers_z = int(raw.get("mfp_n_layers_z", 1))
    disorder_strengths = _as_float_array(raw.get("disorder_strengths"))
    if disorder_strengths.size == 0:
        disorder_strengths = np.asarray([float(disorder_key)], dtype=float)

    thickness_uc = _as_int_array(raw.get("thicknesses"))
    thickness_g = _as_float_matrix(raw.get("thickness_G"))
    thickness_g_std = _as_float_matrix(raw.get("thickness_G_std"))
    thickness_p = _as_int_array(raw.get("thickness_p_eff"))

    if thickness_g.shape[1] != thickness_uc.size:
        raise RuntimeError("RGF thickness result size mismatch.")
    if thickness_g.shape[0] != disorder_strengths.size:
        raise RuntimeError("RGF thickness disorder axis mismatch.")
    if thickness_g_std.size == 0:
        thickness_g_std = np.zeros_like(thickness_g)
    if thickness_g_std.shape != thickness_g.shape:
        raise RuntimeError("RGF thickness stddev shape mismatch.")

    thickness_length_m: list[float] = []
    thickness_m: list[float] = []
    thickness_area_m2: list[float] = []
    for nz in thickness_uc.tolist():
        geom = region_geometry(
            lattice_vectors,
            n_layers_x=n_layers_x,
            n_layers_y=n_layers_y,
            n_layers_z=int(nz),
            lead_axis=lead_axis,
            thickness_axis=thickness_axis,
        )
        thickness_length_m.append(float(geom["length_m"]))
        thickness_m.append(float(geom["thickness_m"]))
        thickness_area_m2.append(float(geom["cross_section_m2"]))

    thickness_length_m_arr = np.asarray(thickness_length_m, dtype=float)
    thickness_m_arr = np.asarray(thickness_m, dtype=float)
    thickness_area_arr = np.asarray(thickness_area_m2, dtype=float)
    thickness_scan: dict[float, dict[str, Any]] = {}
    for row, disorder_strength in enumerate(disorder_strengths.tolist()):
        g_row = np.asarray(thickness_g[row], dtype=float)
        g_std_row = np.asarray(thickness_g_std[row], dtype=float)
        g_si = g_row * E2_OVER_H_SI
        g_si_std = g_std_row * E2_OVER_H_SI
        rho_mean = np.where(
            (g_si > 0) & (thickness_area_arr > 0),
            thickness_length_m_arr / (g_si * thickness_area_arr),
            np.inf,
        )
        rho_std = np.where(
            g_si > 0,
            rho_mean * g_si_std / g_si,
            np.inf,
        )
        thickness_scan[float(disorder_strength)] = {
            "thickness_uc": thickness_uc.tolist(),
            "thickness_uc_requested": thickness_uc.tolist(),
            "thickness_m": thickness_m_arr.tolist(),
            "length_m": thickness_length_m_arr.tolist(),
            "cross_section_m2": thickness_area_arr.tolist(),
            "lead_axis": lead_axis,
            "thickness_axis": thickness_axis,
            "n_layers_x": int(n_layers_x),
            "n_layers_y": int(n_layers_y),
            "lead_axis_cells_requested": int(n_layers_x),
            "lead_axis_cells_used": [int(n_layers_x)] * int(thickness_uc.size),
            "lead_axis_min_cells_required": int(np.max(thickness_p)) if thickness_p.size else 1,
            "G_mean": g_row.tolist(),
            "G_std": g_std_row.tolist(),
            "rho_mean": rho_mean.tolist(),
            "rho_std": rho_std.tolist(),
        }

    length_uc = _as_int_array(raw.get("mfp_lengths"))
    length_g = _as_float_array(raw.get("length_G"))
    length_g_std = _as_float_array(raw.get("length_G_std"))
    length_p = _as_int_array(raw.get("length_p_eff"))
    if length_uc.size != length_g.size:
        raise RuntimeError("RGF length result size mismatch.")
    if length_g_std.size == 0:
        length_g_std = np.zeros_like(length_g)
    if length_g_std.size != length_g.size:
        raise RuntimeError("RGF length stddev size mismatch.")
    length_disorder_strength = float(
        raw.get(
            "length_disorder_strength",
            disorder_strengths[min(disorder_strengths.size - 1, disorder_strengths.size // 2)]
            if disorder_strengths.size
            else disorder_key,
        )
    )

    length_m_vals: list[float] = []
    length_area_m2: list[float] = []
    for nx in length_uc.tolist():
        geom = region_geometry(
            lattice_vectors,
            n_layers_x=int(nx),
            n_layers_y=n_layers_y,
            n_layers_z=mfp_n_layers_z,
            lead_axis=lead_axis,
            thickness_axis=thickness_axis,
        )
        length_m_vals.append(float(geom["length_m"]))
        length_area_m2.append(float(geom["cross_section_m2"]))

    length_m_arr = np.asarray(length_m_vals, dtype=float)
    length_area_arr = np.asarray(length_area_m2, dtype=float)
    g_vs_l = {
        "length_uc": length_uc.tolist(),
        "length_uc_requested": length_uc.tolist(),
        "length_m": length_m_arr.tolist(),
        "cross_section_m2": length_area_arr.tolist(),
        "lead_axis": lead_axis,
        "thickness_axis": thickness_axis,
        "lead_axis_min_cells_required": int(np.max(length_p)) if length_p.size else 1,
        "G_mean": length_g.tolist(),
        "G_std": length_g_std.tolist(),
        "disorder_strength": length_disorder_strength,
    }

    cross_mean = float(np.mean(length_area_arr)) if length_area_arr.size else 0.0
    mfp_result = extract_mfp_from_scaling(
        length_m_arr,
        length_g,
        cross_section_m2=cross_mean,
    )
    if (
        isinstance(mfp_result, dict)
        and "error" not in mfp_result
        and carrier_density_m3 is not None
        and carrier_density_m3 > 0
        and fermi_velocity_m_per_s is not None
        and fermi_velocity_m_per_s > 0
    ):
        drude = mfp_from_sigma(
            float(mfp_result.get("sigma_S_per_m", 0.0)),
            carrier_density_m3=float(carrier_density_m3),
            fermi_velocity_m_per_s=float(fermi_velocity_m_per_s),
        )
        mfp_result["mfp_drude_m"] = drude.get("mfp_m")
        mfp_result["mfp_drude_nm"] = drude.get("mfp_nm")
        mfp_result["tau_s"] = drude.get("tau_s")
        mfp_result["mfp_reference_inputs"] = {
            "carrier_density_m3": float(carrier_density_m3),
            "fermi_velocity_m_per_s": float(fermi_velocity_m_per_s),
        }
        if mfp_result.get("mfp_m") is None:
            mfp_result["mfp_m"] = drude.get("mfp_m")
            mfp_result["mfp_nm"] = drude.get("mfp_nm")
    else:
        if isinstance(mfp_result, dict):
            mfp_result["mfp_reference_inputs"] = {
                "carrier_density_m3": carrier_density_m3,
                "fermi_velocity_m_per_s": fermi_velocity_m_per_s,
            }
    if isinstance(mfp_result, dict):
        mfp_result["G_vs_L"] = g_vs_l
        mfp_result["disorder_strength"] = length_disorder_strength

    return {
        "thickness_scan": thickness_scan,
        "mfp": mfp_result,
        "meta": {
            "transport_engine": "rgf",
            "rgf_mode": str(raw.get("mode", "periodic_transverse")),
            "rgf_periodic_axis": str(raw.get("periodic_axis", "y")),
            "lead_axis": lead_axis,
            "thickness_axis": thickness_axis,
            "n_layers_x": int(n_layers_x),
            "n_layers_y": int(n_layers_y),
            "mfp_n_layers_z": int(mfp_n_layers_z),
            "mfp_lengths": length_uc.tolist(),
            "disorder_strengths": disorder_strengths.tolist(),
            "length_disorder_strength": length_disorder_strength,
            "energy_eV": float(raw.get("energy", 0.0) or 0.0),
            "n_ensemble": int(raw.get("n_ensemble", 1) or 1),
        },
    }
