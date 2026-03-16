from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("wtec_kwant_par_test_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Kwant script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _site_xyz(site: Any) -> tuple[int, int, int]:
    tag = getattr(site, "tag", None)
    if tag is None or len(tag) < 1:
        raise RuntimeError(f"Unexpected site tag: {site!r}")
    vals = [int(v) for v in tag]
    while len(vals) < 3:
        vals.append(0)
    return vals[0], vals[1], vals[2]


def _group_x_layers(unique_x: list[int], width: int) -> list[list[int]]:
    if width <= 0:
        raise ValueError("width must be > 0")
    groups: list[list[int]] = []
    for i in range(0, len(unique_x), width):
        groups.append(unique_x[i : i + width])
    return groups


def _group_x_layers_keep_boundary_cells(unique_x: list[int], width: int) -> list[list[int]]:
    if width <= 0:
        raise ValueError("width must be > 0")
    if not unique_x:
        return []
    total = len(unique_x)
    if total <= width:
        return [list(unique_x)]
    full, rem = divmod(total, width)
    if rem == 0:
        return _group_x_layers(unique_x, width)
    if full < 2:
        return [list(unique_x)]
    groups: list[list[int]] = []
    cursor = 0
    for _ in range(full - 2):
        groups.append(unique_x[cursor : cursor + width])
        cursor += width
    middle_width = width + rem
    groups.append(unique_x[cursor : cursor + middle_width])
    cursor += middle_width
    groups.append(unique_x[cursor : cursor + width])
    if cursor + width != total:
        raise RuntimeError(
            "Boundary-preserving x grouping failed to consume all x coordinates: "
            f"consumed={cursor + width}, total={total}, width={width}, rem={rem}"
        )
    return groups


def _sorted_slice_sites(fsyst: Any, x_group: list[int]) -> list[int]:
    x_set = set(int(v) for v in x_group)
    pairs: list[tuple[tuple[int, int, int], int]] = []
    for idx, site in enumerate(fsyst.sites):
        xyz = _site_xyz(site)
        if xyz[0] in x_set:
            pairs.append((xyz, idx))
    pairs.sort(key=lambda item: item[0])
    return [idx for _, idx in pairs]


def _embed_sigma(
    sigma_iface: np.ndarray,
    *,
    interface_site_indices: list[int],
    slice_site_indices: list[int],
    norb: int,
    dim: int,
) -> np.ndarray:
    pos = {site_idx: i for i, site_idx in enumerate(slice_site_indices)}
    dof_order: list[int] = []
    for site_idx in interface_site_indices:
        if site_idx not in pos:
            raise RuntimeError(
                f"Lead interface site {site_idx} is not contained in the target slice."
            )
        base = pos[site_idx] * norb
        dof_order.extend(range(base, base + norb))
    if sigma_iface.shape != (len(dof_order), len(dof_order)):
        raise RuntimeError(
            "Interface self-energy dimension mismatch: "
            f"sigma={sigma_iface.shape}, embedded_dofs={len(dof_order)}"
        )
    out = np.zeros((dim, dim), dtype=np.complex128)
    out[np.ix_(dof_order, dof_order)] = sigma_iface
    return out


def extract_kwant_blocks(
    *,
    kwant_script: Path,
    hr_path: Path,
    length_uc: int,
    width_uc: int,
    thickness_uc: int,
    energy_ev: float,
    eta_ev: float,
    out_dir: Path,
    queue: str,
    expected_mpi_np: int,
) -> dict[str, Any]:
    mod = _load_module(kwant_script)
    import kwant

    norb, h_r = mod.read_wannier90_hr(str(hr_path))
    fsyst, add_cells = mod.build_system_from_HR(
        h_r,
        L=int(length_uc),
        W=int(width_uc),
        H=int(thickness_uc),
        Ef=0.0,
        transport_axis=0,
    )
    z = complex(float(energy_ev), float(eta_ev))
    max_rx = int(mod.max_hop_range_axis(h_r, axis=0))
    unique_x = sorted({_site_xyz(site)[0] for site in fsyst.sites})
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
        raise RuntimeError(
            f"Left interface x-group mismatch: expected {left_iface_x}, got {x_groups[0]}"
        )
    if x_groups[-1] != right_iface_x:
        raise RuntimeError(
            f"Right interface x-group mismatch: expected {right_iface_x}, got {x_groups[-1]}"
        )
    slice_site_groups = [_sorted_slice_sites(fsyst, group) for group in x_groups]
    slice_dims = [len(group) * int(norb) for group in slice_site_groups]

    h_slices: list[np.ndarray] = []
    v_slices: list[np.ndarray] = []
    for i, group in enumerate(slice_site_groups):
        h_slices.append(
            np.asarray(
                fsyst.hamiltonian_submatrix(to_sites=group, from_sites=group, sparse=False),
                dtype=np.complex128,
            )
        )
        if i < len(slice_site_groups) - 1:
            v_slices.append(
                np.asarray(
                    fsyst.hamiltonian_submatrix(
                        to_sites=group,
                        from_sites=slice_site_groups[i + 1],
                        sparse=False,
                    ),
                    dtype=np.complex128,
                )
            )

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
    h_path = out_dir / "h_slices.bin"
    v_path = out_dir / "v_slices.bin"
    sigma_left_path = out_dir / "sigma_left.bin"
    sigma_right_path = out_dir / "sigma_right.bin"
    payload_path = out_dir / "payload.json"
    manifest_path = out_dir / "extract_manifest.json"

    np.concatenate([blk.ravel(order="C") for blk in h_slices]).astype(np.complex128).tofile(h_path)
    if v_slices:
        np.concatenate([blk.ravel(order="C") for blk in v_slices]).astype(np.complex128).tofile(v_path)
    else:
        np.asarray([], dtype=np.complex128).tofile(v_path)
    sigma_left.astype(np.complex128).ravel(order="C").tofile(sigma_left_path)
    sigma_right.astype(np.complex128).ravel(order="C").tofile(sigma_right_path)

    payload = {
        "hr_dat_path": str(hr_path),
        "queue": str(queue),
        "thicknesses": [int(thickness_uc)],
        "disorder_strengths": [0.0],
        "n_ensemble": 1,
        "energy": float(energy_ev),
        "eta": float(eta_ev),
        "mfp_n_layers_z": int(thickness_uc),
        "mfp_lengths": [],
        "lead_axis": "x",
        "thickness_axis": "z",
        "n_layers_x": int(length_uc),
        "n_layers_y": int(width_uc),
        "transport_engine": "rgf",
        "transport_rgf_mode": "block_validation",
        "transport_rgf_periodic_axis": "y",
        "expected_mpi_np": int(expected_mpi_np),
        "slice_dims": [int(v) for v in slice_dims],
        "h_slices_path": str(h_path),
        "v_slices_path": str(v_path),
        "sigma_left_path": str(sigma_left_path),
        "sigma_right_path": str(sigma_right_path),
    }
    payload_path.write_text(json.dumps(payload, indent=2))

    manifest = {
        "kwant_script": str(kwant_script),
        "hr_path": str(hr_path),
        "length_uc": int(length_uc),
        "width_uc": int(width_uc),
        "thickness_uc": int(thickness_uc),
        "energy_ev": float(energy_ev),
        "eta_ev": float(eta_ev),
        "norb": int(norb),
        "max_rx": int(max_rx),
        "add_cells": int(add_cells),
        "slice_width_x": int(slice_width_x),
        "unique_x_min": int(min(unique_x)),
        "unique_x_max": int(max(unique_x)),
        "unique_x_count": int(len(unique_x)),
        "slice_x_groups": x_groups,
        "slice_dims": [int(v) for v in slice_dims],
        "left_lead_idx": int(left_lead_idx),
        "right_lead_idx": int(right_lead_idx),
        "left_iface_x": left_iface_x,
        "right_iface_x": right_iface_x,
        "lead_interfaces_len": [len(fsyst.lead_interfaces[0]), len(fsyst.lead_interfaces[1])],
        "lead_paddings_len": [len(fsyst.lead_paddings[0]), len(fsyst.lead_paddings[1])],
        "sigma_left_iface_shape": list(sigma_left_iface.shape),
        "sigma_right_iface_shape": list(sigma_right_iface.shape),
        "payload_path": str(payload_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract exact Kwant full-finite blocks for native RGF parity validation.")
    ap.add_argument("--kwant-script", required=True)
    ap.add_argument("--hr-path", required=True)
    ap.add_argument("--length-uc", type=int, required=True)
    ap.add_argument("--width-uc", type=int, required=True)
    ap.add_argument("--thickness-uc", type=int, required=True)
    ap.add_argument("--energy-ev", type=float, required=True)
    ap.add_argument("--eta-ev", type=float, default=1.0e-6)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--queue", default="g4")
    ap.add_argument("--expected-mpi-np", type=int, default=64)
    args = ap.parse_args()

    manifest = extract_kwant_blocks(
        kwant_script=Path(args.kwant_script).expanduser().resolve(),
        hr_path=Path(args.hr_path).expanduser().resolve(),
        length_uc=int(args.length_uc),
        width_uc=int(args.width_uc),
        thickness_uc=int(args.thickness_uc),
        energy_ev=float(args.energy_ev),
        eta_ev=float(args.eta_ev),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        queue=str(args.queue),
        expected_mpi_np=int(args.expected_mpi_np),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
