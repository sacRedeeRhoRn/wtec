from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import kwant
import numpy as np
from mpi4py import MPI

BARRIER_WIDTH = 3
BARRIER_HEIGHT = 1.5


def read_wannier90_hr(_hr_path: str):
    h_r = {
        (0, 0, 0): np.array([[0.0 + 0.0j]], dtype=np.complex128),
        (1, 0, 0): np.array([[-1.0 + 0.0j]], dtype=np.complex128),
        (-1, 0, 0): np.array([[-1.0 + 0.0j]], dtype=np.complex128),
    }
    return 1, h_r


def write_fixture_hr_dat(path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    lines = [
        "one_channel_barrier_chain fixture",
        "1",
        "3",
        "1 1 1",
        "   0   0   0   1   1      0.0000000000      0.0000000000",
        "   1   0   0   1   1     -1.0000000000      0.0000000000",
        "  -1   0   0   1   1     -1.0000000000      0.0000000000",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def max_hop_range_axis(h_r, axis=0):
    mx = 0
    for r in h_r.keys():
        if r == (0, 0, 0):
            continue
        mx = max(mx, abs(r[axis]))
    return mx


def build_system_from_HR(h_r, L=10, W=1, H=1, Ef=0.0, transport_axis=0):
    if transport_axis != 0:
        raise NotImplementedError("Only transport_axis=0(x) is supported.")
    if W != 1 or H != 1:
        raise NotImplementedError("Benchmark fixture is one-channel only: W=1, H=1 required.")

    lat = kwant.lattice.chain(norbs=1)
    syst = kwant.Builder()
    start = (L - BARRIER_WIDTH) // 2
    stop = start + BARRIER_WIDTH
    for x in range(L):
        onsite = float(np.real(h_r[(0, 0, 0)][0, 0]) - Ef)
        if start <= x < stop:
            onsite += BARRIER_HEIGHT
        syst[lat(x)] = onsite
    syst[lat.neighbors()] = h_r[(1, 0, 0)][0, 0]

    sym = kwant.TranslationalSymmetry((-1,))
    lead = kwant.Builder(sym)
    lead[lat(0)] = float(np.real(h_r[(0, 0, 0)][0, 0]) - Ef)
    lead[lat.neighbors()] = h_r[(1, 0, 0)][0, 0]
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    return syst.finalized(), 0


def parse_deltaE_list(raw):
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def transmission_sweep_parallel(fsyst, deltaE_list, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    indexed_energies = list(enumerate(deltaE_list))
    local_jobs = indexed_energies[rank::size]
    local_results = []
    local_error = None
    for idx, dE in local_jobs:
        try:
            smat = kwant.solvers.sparse.smatrix(fsyst, energy=float(dE))
            t10 = float(smat.transmission(1, 0))
            local_results.append((idx, float(dE), t10))
        except Exception as exc:  # pragma: no cover - HPC dependent
            local_error = f"rank={rank}, idx={idx}, dE={dE}, error={type(exc).__name__}: {exc}"
            break
    gathered_results = comm.gather(local_results, root=0)
    gathered_errors = comm.gather(local_error, root=0)
    if rank != 0:
        return None
    errors = [err for err in gathered_errors if err]
    if errors:
        raise RuntimeError("MPI worker failure(s): " + " | ".join(errors))
    merged = []
    for chunk in gathered_results:
        merged.extend(chunk)
    merged.sort(key=lambda item: item[0])
    return [(dE, t10) for _, dE, t10 in merged]


def write_results_txt(txt_path: str, L: int, deltaE_T_list, mpi_size: int):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"timestamp: {ts}",
        "fixture: one_channel_barrier_chain",
        f"geometry_L(cells): {L}",
        f"barrier_width: {BARRIER_WIDTH}",
        f"barrier_height: {BARRIER_HEIGHT}",
        f"mpi_world_size: {mpi_size}",
        "",
        "deltaE_eV\tT_1<-0",
    ]
    for dE, t10 in deltaE_T_list:
        lines.append(f"{dE:+.3f}\t{t10:.12f}")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr-path", default="unused_fixture_hr.dat")
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--W", type=int, default=1)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--deltaE-list", default="-0.2,-0.1,0,0.1,0.2")
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--write-hr-dat", default="")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if args.write_hr_dat and rank == 0:
        hr_path = write_fixture_hr_dat(args.write_hr_dat)
        print(json.dumps({"fixture_hr_dat": str(hr_path)}, indent=2))
    comm.Barrier()
    norb, h_r = read_wannier90_hr(args.hr_path)
    fsyst, add_cells = build_system_from_HR(h_r, L=args.L, W=args.W, H=args.H, Ef=0.0, transport_axis=0)
    result = transmission_sweep_parallel(fsyst, parse_deltaE_list(args.deltaE_list), comm)
    if rank == 0:
        write_results_txt(args.out_txt, args.L, result, size)
        print(json.dumps({"norb": norb, "add_cells": add_cells, "results": result}, indent=2))


if __name__ == "__main__":
    main()
