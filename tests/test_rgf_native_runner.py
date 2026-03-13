import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from wtec.transport.kwant_sigma_extract import extract_kwant_sigmas
from wtec.wannier.model import WannierTBModel
from wtec.wannier.parser import HoppingData, write_hr_dat


@pytest.mark.skipif(
    shutil.which("make") is None or (shutil.which("cc") is None and shutil.which("mpicc") is None),
    reason="native RGF smoke test requires make and a C compiler",
)
def test_rgf_native_runner_spinless_chain_has_unit_transmission(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rgf_dir = repo_root / "wtec" / "ext" / "rgf"
    subprocess.run(["make", "clean", "all"], cwd=rgf_dir, check=True)

    hd = HoppingData(
        num_wann=1,
        r_vectors=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int),
        deg=np.array([1, 1, 1], dtype=int),
        H_R=np.array([[[0.0 + 0.0j]], [[-1.0 + 0.0j]], [[-1.0 + 0.0j]]], dtype=complex),
    )
    write_hr_dat(tmp_path / "toy_hr.dat", hd, header="toy chain")
    (tmp_path / "toy.win").write_text(
        "\n".join(
            [
                "begin unit_cell_cart",
                "ang",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "end unit_cell_cart",
            ]
        )
        + "\n"
    )
    payload = {
        "hr_dat_path": "toy_hr.dat",
        "win_path": "toy.win",
        "queue": "local",
        "thicknesses": [1, 2],
        "disorder_strengths": [0.0],
        "n_ensemble": 1,
        "energy": 0.0,
        "eta": 1.0e-6,
        "mfp_n_layers_z": 1,
        "mfp_lengths": [2, 4, 6],
        "lead_axis": "x",
        "thickness_axis": "z",
        "n_layers_x": 3,
        "n_layers_y": 1,
        "transport_engine": "rgf",
        "transport_rgf_mode": "periodic_transverse",
        "transport_rgf_periodic_axis": "y",
        "expected_mpi_np": 1,
    }
    (tmp_path / "payload.json").write_text(json.dumps(payload, indent=2))

    runner = rgf_dir / "build" / "wtec_rgf_runner"
    subprocess.run([str(runner), "payload.json", "raw.json"], cwd=tmp_path, check=True)

    raw = json.loads((tmp_path / "raw.json").read_text())["transport_results_raw"]
    assert raw["thickness_G"] == pytest.approx([1.0, 2.0], rel=1e-5, abs=1e-5)
    assert raw["length_G"] == pytest.approx([1.0, 1.0, 1.0], rel=1e-5, abs=1e-5)


@pytest.mark.skipif(
    shutil.which("make") is None or (shutil.which("cc") is None and shutil.which("mpicc") is None),
    reason="native RGF smoke test requires make and a C compiler",
)
def test_rgf_native_runner_matches_kwant_sector_by_sector_for_full_layers(tmp_path: Path) -> None:
    kwant = pytest.importorskip("kwant")

    repo_root = Path(__file__).resolve().parents[1]
    rgf_dir = repo_root / "wtec" / "ext" / "rgf"
    subprocess.run(["make", "clean", "all"], cwd=rgf_dir, check=True)

    hd = HoppingData(
        num_wann=1,
        r_vectors=np.array(
            [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 1, 0], [0, -1, 0]],
            dtype=int,
        ),
        deg=np.array([1, 1, 1, 1, 1, 2, 2], dtype=int),
        H_R=np.array(
            [[[0.15 + 0.0j]], [[-1.0 + 0.0j]], [[-1.0 + 0.0j]], [[-0.25 + 0.0j]], [[-0.25 + 0.0j]], [[-0.4 + 0.0j]], [[-0.4 + 0.0j]]],
            dtype=complex,
        ),
    )
    write_hr_dat(tmp_path / "toy_hr.dat", hd, header="toy full-layer compare")
    (tmp_path / "toy.win").write_text(
        "\n".join(
            [
                "begin unit_cell_cart",
                "ang",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "end unit_cell_cart",
            ]
        )
        + "\n"
    )
    payload = {
        "hr_dat_path": "toy_hr.dat",
        "win_path": "toy.win",
        "queue": "local",
        "thicknesses": [1],
        "disorder_strengths": [0.0],
        "n_ensemble": 1,
        "energy": 0.0,
        "eta": 1.0e-6,
        "mfp_n_layers_z": 1,
        "mfp_lengths": [4],
        "lead_axis": "x",
        "thickness_axis": "z",
        "n_layers_x": 4,
        "n_layers_y": 4,
        "transport_engine": "rgf",
        "transport_rgf_mode": "periodic_transverse",
        "transport_rgf_periodic_axis": "y",
        "expected_mpi_np": 1,
    }
    (tmp_path / "payload.json").write_text(json.dumps(payload, indent=2))

    runner = rgf_dir / "build" / "wtec_rgf_runner"
    subprocess.run([str(runner), "payload.json", "raw.json"], cwd=tmp_path, check=True)
    raw = json.loads((tmp_path / "raw.json").read_text())["transport_results_raw"]

    model = WannierTBModel.from_hopping_data(hd, lattice_vectors=np.eye(3))
    kwant_vals = []
    for ky in raw["ky_fractions"]:
        sysb = model.to_kwant_builder_periodic_y(
            ky_frac=float(ky),
            n_layers_x=4,
            n_layers_z=1,
            lead_axis="x",
        )
        fsys = sysb.finalized()
        kwant_vals.append(float(kwant.smatrix(fsys, 0.0).transmission(0, 1)))

    assert raw["length_sector_G"][0] == pytest.approx(kwant_vals, rel=1e-5, abs=1e-5)


@pytest.mark.skipif(
    shutil.which("make") is None or (shutil.which("cc") is None and shutil.which("mpicc") is None),
    reason="native RGF smoke test requires make and a C compiler",
)
def test_rgf_native_runner_rejects_partial_last_principal_layer(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rgf_dir = repo_root / "wtec" / "ext" / "rgf"
    subprocess.run(["make", "clean", "all"], cwd=rgf_dir, check=True)

    hd = HoppingData(
        num_wann=1,
        r_vectors=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [2, 0, 0], [-2, 0, 0]], dtype=int),
        deg=np.array([1, 1, 1, 1, 1], dtype=int),
        H_R=np.array([[[0.0 + 0.0j]], [[-1.0 + 0.0j]], [[-1.0 + 0.0j]], [[-0.25 + 0.0j]], [[-0.25 + 0.0j]]], dtype=complex),
    )
    write_hr_dat(tmp_path / "toy_hr.dat", hd, header="toy partial-layer reject")
    (tmp_path / "toy.win").write_text(
        "\n".join(
            [
                "begin unit_cell_cart",
                "ang",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "end unit_cell_cart",
            ]
        )
        + "\n"
    )
    payload = {
        "hr_dat_path": "toy_hr.dat",
        "win_path": "toy.win",
        "queue": "local",
        "thicknesses": [1],
        "disorder_strengths": [0.0],
        "n_ensemble": 1,
        "energy": 0.0,
        "eta": 1.0e-6,
        "mfp_n_layers_z": 1,
        "mfp_lengths": [3],
        "lead_axis": "x",
        "thickness_axis": "z",
        "n_layers_x": 3,
        "n_layers_y": 4,
        "transport_engine": "rgf",
        "transport_rgf_mode": "periodic_transverse",
        "transport_rgf_periodic_axis": "y",
        "expected_mpi_np": 1,
    }
    (tmp_path / "payload.json").write_text(json.dumps(payload, indent=2))

    runner = rgf_dir / "build" / "wtec_rgf_runner"
    proc = subprocess.run(
        [str(runner), "payload.json", "raw.json"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "divisible by principal_layer_width" in proc.stderr


@pytest.mark.skipif(
    shutil.which("make") is None or (shutil.which("cc") is None and shutil.which("mpicc") is None),
    reason="native RGF smoke test requires make and a C compiler",
)
def test_rgf_native_runner_emits_progress_and_energy_for_full_finite(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rgf_dir = repo_root / "wtec" / "ext" / "rgf"
    subprocess.run(["make", "clean", "all"], cwd=rgf_dir, check=True)

    hd = HoppingData(
        num_wann=1,
        r_vectors=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int),
        deg=np.array([1, 1, 1], dtype=int),
        H_R=np.array([[[0.0 + 0.0j]], [[-1.0 + 0.0j]], [[-1.0 + 0.0j]]], dtype=complex),
    )
    write_hr_dat(tmp_path / "toy_hr.dat", hd, header="toy full-finite progress")
    (tmp_path / "toy.win").write_text(
        "\n".join(
            [
                "begin unit_cell_cart",
                "ang",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "end unit_cell_cart",
            ]
        )
        + "\n"
    )
    payload = {
        "hr_dat_path": "toy_hr.dat",
        "win_path": "toy.win",
        "queue": "local",
        "thicknesses": [1],
        "disorder_strengths": [0.0],
        "n_ensemble": 1,
        "energy": 0.25,
        "eta": 1.0e-6,
        "mfp_n_layers_z": 1,
        "mfp_lengths": [],
        "lead_axis": "x",
        "thickness_axis": "z",
        "n_layers_x": 3,
        "n_layers_y": 1,
        "transport_engine": "rgf",
        "transport_rgf_mode": "full_finite",
        "transport_rgf_periodic_axis": "y",
        "expected_mpi_np": 1,
        "progress_file": "progress.jsonl",
        "logging_detail": "per_step",
        "heartbeat_seconds": 1,
    }
    (tmp_path / "payload.json").write_text(json.dumps(payload, indent=2))

    runner = rgf_dir / "build" / "wtec_rgf_runner"
    proc = subprocess.run(
        [str(runner), "payload.json", "raw.json"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((tmp_path / "raw.json").read_text())
    raw = payload["transport_results_raw"]
    runtime_cert = payload["runtime_cert"]
    progress_lines = (tmp_path / "progress.jsonl").read_text().splitlines()

    assert raw["mode"] == "full_finite"
    assert raw["energy"] == pytest.approx(0.25)
    assert raw["eta"] == pytest.approx(1.0e-6)
    assert runtime_cert["wall_seconds"] >= 0.0
    assert runtime_cert["aggregate_process_cpu_seconds"] >= 0.0
    assert runtime_cert["effective_thread_count"] >= 0.0
    assert any('"event":"worker_start"' in line for line in progress_lines)
    assert any('"event":"native_point_start"' in line for line in progress_lines)
    assert any('"event":"native_point_done"' in line for line in progress_lines)
    assert any('"event":"native_phase"' in line for line in progress_lines)
    assert "[progress]" in proc.stdout


@pytest.mark.skipif(
    shutil.which("make") is None or (shutil.which("cc") is None and shutil.which("mpicc") is None),
    reason="native RGF smoke test requires make and a C compiler",
)
def test_extract_kwant_sigmas_reports_full_finite_principal_progress(tmp_path: Path) -> None:
    pytest.importorskip("kwant")

    hd = HoppingData(
        num_wann=1,
        r_vectors=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [2, 0, 0], [-2, 0, 0]], dtype=int),
        deg=np.array([1, 1, 1, 1, 1], dtype=int),
        H_R=np.array(
            [[[0.0 + 0.0j]], [[-1.0 + 0.0j]], [[-1.0 + 0.0j]], [[-0.4 + 0.0j]], [[-0.4 + 0.0j]]],
            dtype=complex,
        ),
    )
    hr_path = tmp_path / "toy_hr.dat"
    write_hr_dat(hr_path, hd, header="toy full-finite exact sigma progress")

    progress: list[str] = []
    extract_kwant_sigmas(
        hr_path=hr_path,
        length_uc=6,
        width_uc=1,
        thickness_uc=1,
        energy_ev=0.1,
        eta_ev=1.0e-6,
        out_dir=tmp_path / "sigma",
        layout="full_finite_principal",
        progress_cb=progress.append,
    )

    events = [line.split()[1] for line in progress]
    assert events == [
        "full_finite_principal_start",
        "full_finite_principal_geometry_ready",
        "full_finite_principal_blocks_ready",
        "selfenergy_left_start",
        "selfenergy_left_done",
        "selfenergy_right_start",
        "selfenergy_right_done",
        "sigma_outputs_written",
    ]
    assert any("lead_dim=2" in line for line in progress)
    assert any("manifest_path=sigma_manifest.json" in line for line in progress)


@pytest.mark.skipif(
    shutil.which("make") is None or (shutil.which("cc") is None and shutil.which("mpicc") is None),
    reason="native RGF smoke test requires make and a C compiler",
)
def test_rgf_native_runner_matches_principal_layer_exact_sigma_for_full_finite(tmp_path: Path) -> None:
    kwant = pytest.importorskip("kwant")

    repo_root = Path(__file__).resolve().parents[1]
    rgf_dir = repo_root / "wtec" / "ext" / "rgf"
    subprocess.run(["make", "clean", "all"], cwd=rgf_dir, check=True)

    hd = HoppingData(
        num_wann=1,
        r_vectors=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [2, 0, 0], [-2, 0, 0]], dtype=int),
        deg=np.array([1, 1, 1, 1, 1], dtype=int),
        H_R=np.array(
            [[[0.0 + 0.0j]], [[-1.0 + 0.0j]], [[-1.0 + 0.0j]], [[-0.4 + 0.0j]], [[-0.4 + 0.0j]]],
            dtype=complex,
        ),
    )
    hr_path = tmp_path / "toy_hr.dat"
    write_hr_dat(hr_path, hd, header="toy full-finite exact sigma")
    (tmp_path / "toy.win").write_text(
        "\n".join(
            [
                "begin unit_cell_cart",
                "ang",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "end unit_cell_cart",
            ]
        )
        + "\n"
    )

    sigma_manifest = extract_kwant_sigmas(
        hr_path=hr_path,
        length_uc=6,
        width_uc=1,
        thickness_uc=1,
        energy_ev=0.1,
        eta_ev=1.0e-6,
        out_dir=tmp_path / "sigma",
        layout="full_finite_principal",
    )
    assert sigma_manifest["layout"] == "full_finite_principal"
    assert sigma_manifest["principal_layer_width"] == 2
    assert sigma_manifest["slice_widths"] == [2, 2, 2, 2]

    payload = {
        "hr_dat_path": "toy_hr.dat",
        "win_path": "toy.win",
        "queue": "local",
        "thicknesses": [1],
        "disorder_strengths": [0.0],
        "n_ensemble": 1,
        "energy": 0.1,
        "eta": 1.0e-6,
        "mfp_n_layers_z": 1,
        "mfp_lengths": [],
        "lead_axis": "x",
        "thickness_axis": "z",
        "n_layers_x": 6,
        "n_layers_y": 1,
        "transport_engine": "rgf",
        "transport_rgf_mode": "full_finite",
        "transport_rgf_periodic_axis": "y",
        "sigma_left_path": str(Path("sigma") / "sigma_left.bin"),
        "sigma_right_path": str(Path("sigma") / "sigma_right.bin"),
    }
    (tmp_path / "payload.json").write_text(json.dumps(payload, indent=2))

    runner = rgf_dir / "build" / "wtec_rgf_runner"
    subprocess.run([str(runner), "payload.json", "raw.json"], cwd=tmp_path, check=True)
    raw = json.loads((tmp_path / "raw.json").read_text())["transport_results_raw"]

    lat = kwant.lattice.chain(norbs=1)
    syst = kwant.Builder()
    for x in range(6):
        syst[lat(x)] = 0.0
    for x in range(5):
        syst[lat(x), lat(x + 1)] = -1.0
    for x in range(4):
        syst[lat(x), lat(x + 2)] = -0.4
    sym = kwant.TranslationalSymmetry((-1,))
    lead = kwant.Builder(sym)
    lead[lat(0)] = 0.0
    lead[lat.neighbors()] = -1.0
    lead[lat(0), lat(2)] = -0.4
    syst.attach_lead(lead, add_cells=1)
    syst.attach_lead(lead.reversed(), add_cells=1)
    kwant_t = float(kwant.smatrix(syst.finalized(), 0.1).transmission(1, 0))

    assert raw["mode"] == "full_finite"
    assert raw["thickness_G"] == pytest.approx([kwant_t], rel=1.0e-5, abs=1.0e-5)


@pytest.mark.skipif(
    shutil.which("make") is None or (shutil.which("cc") is None and shutil.which("mpicc") is None),
    reason="native RGF smoke test requires make and a C compiler",
)
def test_rgf_native_runner_supports_full_finite_disorder_ensembles(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rgf_dir = repo_root / "wtec" / "ext" / "rgf"
    subprocess.run(["make", "clean", "all"], cwd=rgf_dir, check=True)

    hd = HoppingData(
        num_wann=1,
        r_vectors=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int),
        deg=np.array([1, 1, 1], dtype=int),
        H_R=np.array([[[0.0 + 0.0j]], [[-1.0 + 0.0j]], [[-1.0 + 0.0j]]], dtype=complex),
    )
    write_hr_dat(tmp_path / "toy_hr.dat", hd, header="toy full-finite disorder")
    (tmp_path / "toy.win").write_text(
        "\n".join(
            [
                "begin unit_cell_cart",
                "ang",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "end unit_cell_cart",
            ]
        )
        + "\n"
    )
    payload = {
        "hr_dat_path": "toy_hr.dat",
        "win_path": "toy.win",
        "queue": "local",
        "thicknesses": [1],
        "disorder_strengths": [0.0, 0.4],
        "n_ensemble": 4,
        "base_seed": 11,
        "energy": 0.0,
        "eta": 1.0e-6,
        "mfp_n_layers_z": 1,
        "mfp_lengths": [2, 4],
        "lead_axis": "x",
        "thickness_axis": "z",
        "n_layers_x": 4,
        "n_layers_y": 1,
        "transport_engine": "rgf",
        "transport_rgf_mode": "full_finite",
        "transport_rgf_periodic_axis": "y",
        "expected_mpi_np": 1,
    }
    (tmp_path / "payload.json").write_text(json.dumps(payload, indent=2))

    runner = rgf_dir / "build" / "wtec_rgf_runner"
    subprocess.run([str(runner), "payload.json", "raw.json"], cwd=tmp_path, check=True)

    raw = json.loads((tmp_path / "raw.json").read_text())["transport_results_raw"]

    assert raw["disorder_strengths"] == pytest.approx([0.0, 0.4])
    assert raw["n_ensemble"] == 4
    assert np.asarray(raw["thickness_G"]).shape == (2, 1)
    assert np.asarray(raw["thickness_G_std"]).shape == (2, 1)
    assert raw["thickness_G"][0][0] == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert raw["thickness_G_std"][0][0] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= raw["thickness_G"][1][0] <= 1.0
    assert raw["thickness_G_std"][1][0] > 0.0
    assert raw["length_disorder_strength"] == pytest.approx(0.4)
    assert len(raw["length_G"]) == 2
    assert raw["length_G_std"][0] > 0.0
