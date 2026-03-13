from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wtec.transport.kwant_block_extract import (
    _embed_sigma,
    _group_x_layers_keep_boundary_cells,
    _load_module,
    _sorted_slice_sites,
)
from wtec.wannier.parser import read_hr_dat


def _hr_dict(path: str | Path) -> tuple[int, dict[tuple[int, int, int], np.ndarray]]:
    hd = read_hr_dat(Path(path).expanduser().resolve())
    h_r: dict[tuple[int, int, int], np.ndarray] = {}
    for ri, rv in enumerate(np.asarray(hd.r_vectors, dtype=int)):
        key = tuple(int(v) for v in rv)
        denom = float(hd.deg[ri]) if int(hd.deg[ri]) != 0 else 1.0
        h_r[key] = np.asarray(hd.H_R[ri], dtype=np.complex128) / denom
    if (0, 0, 0) not in h_r:
        raise ValueError("R=(0,0,0) onsite block is missing in hr.dat.")
    return int(hd.num_wann), h_r


def _is_positive_lex(r: tuple[int, int, int]) -> bool:
    if r == (0, 0, 0):
        return False
    rx, ry, rz = r
    if rx != 0:
        return rx > 0
    if ry != 0:
        return ry > 0
    return rz > 0


def _max_hop_range_axis(h_r: dict[tuple[int, int, int], np.ndarray], axis: int = 0) -> int:
    mx = 0
    for rv in h_r:
        if rv == (0, 0, 0):
            continue
        mx = max(mx, abs(int(rv[axis])))
    return mx


def _effective_principal_layer_width(
    h_r: dict[tuple[int, int, int], np.ndarray],
    *,
    width_uc: int,
    thickness_uc: int,
) -> int:
    mx = 0
    for rx, ry, rz in h_r:
        if abs(int(ry)) > max(0, int(width_uc) - 1):
            continue
        if abs(int(rz)) > max(0, int(thickness_uc) - 1):
            continue
        mx = max(mx, abs(int(rx)))
    return max(1, int(mx))


def _plan_boundary_preserving_widths(nx: int, lead_width: int) -> list[int]:
    nx_i = int(nx)
    lead_width_i = int(lead_width)
    if nx_i <= 0 or lead_width_i <= 0:
        raise ValueError("nx and lead_width must both be > 0")
    if nx_i <= lead_width_i:
        return [nx_i]
    full, rem = divmod(nx_i, lead_width_i)
    if rem == 0:
        return [lead_width_i] * full
    if full < 2:
        return [nx_i]
    return [lead_width_i] * (full - 2) + [lead_width_i + rem, lead_width_i]


def _build_block_full_from_hr(
    h_r: dict[tuple[int, int, int], np.ndarray],
    *,
    norb: int,
    width_left: int,
    width_right: int,
    shift_x: int,
    width_uc: int,
    thickness_uc: int,
) -> np.ndarray:
    rows = int(width_left) * int(width_uc) * int(thickness_uc) * int(norb)
    cols = int(width_right) * int(width_uc) * int(thickness_uc) * int(norb)
    out = np.zeros((rows, cols), dtype=np.complex128)
    for (rx, ry, rz), mat in h_r.items():
        if abs(int(ry)) > max(0, int(width_uc) - 1):
            continue
        if abs(int(rz)) > max(0, int(thickness_uc) - 1):
            continue
        block = np.asarray(mat, dtype=np.complex128)
        for x_left in range(int(width_left)):
            x_right = int(x_left + int(rx) - int(shift_x))
            if x_right < 0 or x_right >= int(width_right):
                continue
            for y_left in range(int(width_uc)):
                y_right = int(y_left + int(ry))
                if y_right < 0 or y_right >= int(width_uc):
                    continue
                for z_left in range(int(thickness_uc)):
                    z_right = int(z_left + int(rz))
                    if z_right < 0 or z_right >= int(thickness_uc):
                        continue
                    row_base = ((x_left * int(width_uc) + y_left) * int(thickness_uc) + z_left) * int(norb)
                    col_base = ((x_right * int(width_uc) + y_right) * int(thickness_uc) + z_right) * int(norb)
                    out[row_base : row_base + int(norb), col_base : col_base + int(norb)] += block
    return out


def _build_system_from_hr(
    h_r: dict[tuple[int, int, int], np.ndarray],
    *,
    length_uc: int,
    width_uc: int,
    thickness_uc: int,
    ef_ev: float = 0.0,
) -> tuple[Any, int]:
    import kwant

    onsite = np.asarray(h_r[(0, 0, 0)], dtype=np.complex128).copy()
    norb = onsite.shape[0]
    onsite -= ef_ev * np.eye(norb, dtype=np.complex128)

    lat = kwant.lattice.general(
        [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        [(0, 0, 0)],
        norbs=norb,
    )
    a = lat.sublattices[0]

    syst = kwant.Builder()
    for x in range(int(length_uc)):
        for y in range(int(width_uc)):
            for z in range(int(thickness_uc)):
                syst[a(x, y, z)] = onsite

    for rv, mat in h_r.items():
        if rv == (0, 0, 0):
            continue
        if _is_positive_lex(rv):
            syst[kwant.builder.HoppingKind(rv, a, a)] = np.asarray(mat, dtype=np.complex128)

    sym = kwant.TranslationalSymmetry(lat.vec((1, 0, 0)))
    lead = kwant.Builder(sym)
    for y in range(int(width_uc)):
        for z in range(int(thickness_uc)):
            lead[a(0, y, z)] = onsite

    for rv, mat in h_r.items():
        if rv == (0, 0, 0):
            continue
        rx = int(rv[0])
        if rx == 0 and _is_positive_lex(rv):
            lead[kwant.builder.HoppingKind(rv, a, a)] = np.asarray(mat, dtype=np.complex128)
        if rx > 0:
            lead[kwant.builder.HoppingKind(rv, a, a)] = np.asarray(mat, dtype=np.complex128)

    max_rx = _max_hop_range_axis(h_r, axis=0)
    add_cells = max(0, max_rx - 1)
    syst.attach_lead(lead, add_cells=add_cells)
    syst.attach_lead(lead.reversed(), add_cells=add_cells)
    return syst.finalized(), add_cells


def extract_kwant_sigmas(
    *,
    hr_path: Path,
    length_uc: int,
    width_uc: int,
    thickness_uc: int,
    energy_ev: float,
    eta_ev: float,
    out_dir: Path,
    kwant_script: Path | None = None,
    layout: str = "raw_fsyst",
) -> dict[str, Any]:
    import kwant

    layout_norm = str(layout).strip().lower() or "raw_fsyst"
    if layout_norm not in {"raw_fsyst", "full_finite_principal"}:
        raise ValueError(
            f"Unsupported sigma extraction layout {layout!r}; expected 'raw_fsyst' or 'full_finite_principal'."
        )

    if layout_norm == "full_finite_principal":
        if kwant_script is not None:
            raise ValueError("full_finite_principal layout does not support --kwant-script.")
        norb, h_r = _hr_dict(hr_path)
        p_eff = _effective_principal_layer_width(
            h_r,
            width_uc=int(width_uc),
            thickness_uc=int(thickness_uc),
        )
        pad_x = max(0, int(p_eff) - 1)
        nx_effective = int(length_uc) + 2 * int(pad_x)
        slice_widths = _plan_boundary_preserving_widths(nx_effective, int(p_eff))
        if slice_widths[0] != int(p_eff) or slice_widths[-1] != int(p_eff):
            raise ValueError(
                "full_finite_principal layout currently requires boundary-preserving widths that start and end "
                f"with principal_layer_width={int(p_eff)}; got widths={slice_widths}."
            )

        h_lead = _build_block_full_from_hr(
            h_r,
            norb=int(norb),
            width_left=int(p_eff),
            width_right=int(p_eff),
            shift_x=0,
            width_uc=int(width_uc),
            thickness_uc=int(thickness_uc),
        )
        v_lead = _build_block_full_from_hr(
            h_r,
            norb=int(norb),
            width_left=int(p_eff),
            width_right=int(p_eff),
            shift_x=int(p_eff),
            width_uc=int(width_uc),
            thickness_uc=int(thickness_uc),
        )
        v_lead_r = np.asarray(v_lead.conj().T, dtype=np.complex128)
        z = complex(float(energy_ev), float(eta_ev))
        lead_resolvent = np.asarray(
            h_lead - z * np.eye(h_lead.shape[0], dtype=np.complex128),
            dtype=np.complex128,
        )
        sigma_left = np.asarray(
            kwant.physics.selfenergy(lead_resolvent, v_lead),
            dtype=np.complex128,
        )
        sigma_right = np.asarray(
            kwant.physics.selfenergy(lead_resolvent, v_lead_r),
            dtype=np.complex128,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        sigma_left_path = out_dir / "sigma_left.bin"
        sigma_right_path = out_dir / "sigma_right.bin"
        manifest_path = out_dir / "sigma_manifest.json"
        sigma_left.ravel(order="C").tofile(sigma_left_path)
        sigma_right.ravel(order="C").tofile(sigma_right_path)
        payload = {
            "hr_path": str(hr_path),
            "kwant_script": None,
            "layout": layout_norm,
            "length_uc": int(length_uc),
            "width_uc": int(width_uc),
            "thickness_uc": int(thickness_uc),
            "energy_ev": float(energy_ev),
            "eta_ev": float(eta_ev),
            "norb": int(norb),
            "principal_layer_width": int(p_eff),
            "pad_x": int(pad_x),
            "nx_effective": int(nx_effective),
            "slice_widths": [int(v) for v in slice_widths],
            "sigma_left_path": str(sigma_left_path),
            "sigma_right_path": str(sigma_right_path),
            "sigma_left_shape": list(sigma_left.shape),
            "sigma_right_shape": list(sigma_right.shape),
        }
        manifest_path.write_text(json.dumps(payload, indent=2))
        return payload

    if kwant_script is not None:
        mod = _load_module(Path(kwant_script).expanduser().resolve())
        norb, h_r = mod.read_wannier90_hr(str(hr_path))
        fsyst, add_cells = mod.build_system_from_HR(
            h_r,
            L=int(length_uc),
            W=int(width_uc),
            H=int(thickness_uc),
            Ef=0.0,
            transport_axis=0,
        )
    else:
        norb, h_r = _hr_dict(hr_path)
        fsyst, add_cells = _build_system_from_hr(
            h_r,
            length_uc=int(length_uc),
            width_uc=int(width_uc),
            thickness_uc=int(thickness_uc),
            ef_ev=0.0,
        )
    z = complex(float(energy_ev), float(eta_ev))
    unique_x = sorted({int(fsyst.sites[i].tag[0]) for i in range(len(fsyst.sites))})
    lead_iface_x_by_idx = [
        sorted({int(fsyst.sites[i].tag[0]) for i in fsyst.lead_interfaces[lead_idx]})
        for lead_idx in range(len(fsyst.lead_interfaces))
    ]
    left_lead_idx = min(range(len(lead_iface_x_by_idx)), key=lambda idx: min(lead_iface_x_by_idx[idx]))
    right_lead_idx = max(range(len(lead_iface_x_by_idx)), key=lambda idx: max(lead_iface_x_by_idx[idx]))
    left_iface_x = lead_iface_x_by_idx[left_lead_idx]
    right_iface_x = lead_iface_x_by_idx[right_lead_idx]

    lead_cell_x = sorted(
        {int(fsyst.leads[left_lead_idx].sites[i].tag[0]) for i in range(fsyst.leads[left_lead_idx].cell_size)}
    )
    slice_width_x = len(lead_cell_x)
    x_groups = _group_x_layers_keep_boundary_cells(unique_x, slice_width_x)
    if x_groups[0] != left_iface_x:
        raise RuntimeError(f"Left interface x-group mismatch: expected {left_iface_x}, got {x_groups[0]}")
    if x_groups[-1] != right_iface_x:
        raise RuntimeError(f"Right interface x-group mismatch: expected {right_iface_x}, got {x_groups[-1]}")

    slice_site_groups = [_sorted_slice_sites(fsyst, group) for group in x_groups]
    slice_dims = [len(group) * int(norb) for group in slice_site_groups]

    lead_left = fsyst.leads[left_lead_idx]
    lead_right = fsyst.leads[right_lead_idx]
    sigma_left_iface = np.asarray(
        kwant.physics.selfenergy(
            lead_left.cell_hamiltonian() - z * np.eye(lead_left.cell_hamiltonian().shape[0], dtype=np.complex128),
            lead_left.inter_cell_hopping(),
        ),
        dtype=np.complex128,
    )
    sigma_right_iface = np.asarray(
        kwant.physics.selfenergy(
            lead_right.cell_hamiltonian() - z * np.eye(lead_right.cell_hamiltonian().shape[0], dtype=np.complex128),
            lead_right.inter_cell_hopping(),
        ),
        dtype=np.complex128,
    )

    sigma_left = _embed_sigma(
        sigma_left_iface,
        interface_site_indices=[int(v) for v in fsyst.lead_interfaces[left_lead_idx]],
        slice_site_indices=slice_site_groups[0],
        norb=int(norb),
        dim=slice_dims[0],
    )
    sigma_right = _embed_sigma(
        sigma_right_iface,
        interface_site_indices=[int(v) for v in fsyst.lead_interfaces[right_lead_idx]],
        slice_site_indices=slice_site_groups[-1],
        norb=int(norb),
        dim=slice_dims[-1],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    sigma_left_path = out_dir / "sigma_left.bin"
    sigma_right_path = out_dir / "sigma_right.bin"
    manifest_path = out_dir / "sigma_manifest.json"
    sigma_left.astype(np.complex128).ravel(order="C").tofile(sigma_left_path)
    sigma_right.astype(np.complex128).ravel(order="C").tofile(sigma_right_path)

    payload = {
        "hr_path": str(hr_path),
        "kwant_script": str(kwant_script) if kwant_script is not None else None,
        "layout": layout_norm,
        "length_uc": int(length_uc),
        "width_uc": int(width_uc),
        "thickness_uc": int(thickness_uc),
        "energy_ev": float(energy_ev),
        "eta_ev": float(eta_ev),
        "norb": int(norb),
        "add_cells": int(add_cells),
        "slice_width_x": int(slice_width_x),
        "slice_x_groups": x_groups,
        "slice_dims": [int(v) for v in slice_dims],
        "left_lead_idx": int(left_lead_idx),
        "right_lead_idx": int(right_lead_idx),
        "left_iface_x": left_iface_x,
        "right_iface_x": right_iface_x,
        "sigma_left_path": str(sigma_left_path),
        "sigma_right_path": str(sigma_right_path),
        "sigma_left_iface_shape": list(sigma_left_iface.shape),
        "sigma_right_iface_shape": list(sigma_right_iface.shape),
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract exact Kwant lead self-energies for native RGF full-finite runs.")
    ap.add_argument("--hr-path", required=True)
    ap.add_argument("--length-uc", type=int, required=True)
    ap.add_argument("--width-uc", type=int, required=True)
    ap.add_argument("--thickness-uc", type=int, required=True)
    ap.add_argument("--energy-ev", type=float, required=True)
    ap.add_argument("--eta-ev", type=float, default=1.0e-6)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kwant-script", default=None)
    ap.add_argument(
        "--layout",
        default="raw_fsyst",
        choices=("raw_fsyst", "full_finite_principal"),
        help="Sigma extraction layout. raw_fsyst reproduces the existing Kwant-attached-system interface layout; "
        "full_finite_principal emits sigmas in the native full-finite principal-layer representation.",
    )
    args = ap.parse_args()

    payload = extract_kwant_sigmas(
        hr_path=Path(args.hr_path).expanduser().resolve(),
        length_uc=int(args.length_uc),
        width_uc=int(args.width_uc),
        thickness_uc=int(args.thickness_uc),
        energy_ev=float(args.energy_ev),
        eta_ev=float(args.eta_ev),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        kwant_script=Path(args.kwant_script).expanduser().resolve() if args.kwant_script else None,
        layout=str(args.layout),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
