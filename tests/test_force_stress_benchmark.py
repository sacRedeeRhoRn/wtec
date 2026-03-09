from __future__ import annotations

import numpy as np

from wtec.analysis.force_stress_benchmark import (
    BenchmarkThresholds,
    choose_fastest_passing_case,
    compare_force_stress,
    evaluate_thresholds,
)
from wtec.siesta.parser import (
    parse_convergence as parse_siesta_convergence,
    parse_elapsed_seconds as parse_siesta_elapsed_seconds,
    parse_fermi_energy as parse_siesta_fermi_energy,
    parse_forces as parse_siesta_forces,
    parse_stress_kbar as parse_siesta_stress_kbar,
    parse_total_energy as parse_siesta_total_energy,
)
from wtec.vasp.parser import (
    parse_convergence as parse_vasp_convergence,
    parse_elapsed_seconds as parse_vasp_elapsed_seconds,
    parse_forces as parse_vasp_forces,
    parse_stress_kbar as parse_vasp_stress_kbar,
    parse_total_energy as parse_vasp_total_energy,
)


def test_vasp_parser_force_stress_energy_elapsed(tmp_path) -> None:
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        """
 something
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.000000      0.000000      0.000000      1.000000      2.000000      3.000000
      1.000000      1.000000      1.000000     -1.000000     -2.000000     -3.000000
 -----------------------------------------------------------------------------------
 random
  in kB      10.00000   20.00000   30.00000    40.00000    50.00000    60.00000
 free  energy   TOTEN  =      -708.51000262 eV
 General timing and accounting informations for this job:
 Elapsed time (sec):    14630.665
 ------------------------ aborting loop because EDIFF is reached ----------------------------------------
 """
    )

    forces = parse_vasp_forces(outcar)
    stress = parse_vasp_stress_kbar(outcar)
    assert forces.shape == (2, 3)
    assert np.allclose(forces[0], [1.0, 2.0, 3.0])
    assert np.allclose(forces[1], [-1.0, -2.0, -3.0])
    assert np.allclose(stress, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    assert parse_vasp_total_energy(outcar) == -708.51000262
    assert parse_vasp_elapsed_seconds(outcar) == 14630.665
    assert parse_vasp_convergence(outcar)


def test_siesta_parser_force_stress_energy_elapsed(tmp_path) -> None:
    out = tmp_path / "SiBench.scf.out"
    out.write_text(
        """
 SCF cycle converged after 6 iterations
 siesta: Final energy (eV):
 siesta:         Total =  -13857.737136
 siesta:         Fermi =      -5.319059
 Stress tensor Voigt[x,y,z,yz,xz,xy] (kbar):      -17.01      -18.02      -19.03       -4.00       -5.00       -6.00
 """
    )
    times = tmp_path / "SiBench.times"
    times.write_text(
        "timer: Total elapsed wall-clock time (sec) =       84.248\n"
    )
    force_stress = tmp_path / "FORCE_STRESS"
    force_stress.write_text(
        "\n".join(
            [
                "-1018.0",
                "-0.000100 -0.000001 -0.000002",
                "-0.000003 -0.000110 -0.000004",
                "-0.000005 -0.000006 -0.000120",
                "2",
                "1 14 0.100000 0.200000 0.300000 Si",
                "1 14 -0.100000 -0.200000 -0.300000 Si",
            ]
        )
        + "\n"
    )

    forces = parse_siesta_forces(out, force_stress_path=force_stress)
    stress = parse_siesta_stress_kbar(out, force_stress_path=force_stress)
    assert forces.shape == (2, 3)
    # FORCE_STRESS stores forces in Ry/Bohr; parser returns eV/Ang.
    assert np.allclose(forces[0], [2.571104, 5.142209, 7.713313], atol=1e-6)
    # Voigt order normalized to [xx, yy, zz, xy, yz, zx].
    assert np.allclose(stress, [-17.01, -18.02, -19.03, -6.0, -4.0, -5.0], atol=1e-6)
    assert parse_siesta_total_energy(out) == -13857.737136
    assert parse_siesta_elapsed_seconds(out, times_path=times) == 84.248
    assert parse_siesta_fermi_energy(out) == -5.319059
    assert parse_siesta_convergence(out)


def test_siesta_stress_force_stress_fallback(tmp_path) -> None:
    out = tmp_path / "x.out"
    out.write_text("siesta:         Total =  -1.0\n")
    force_stress = tmp_path / "FORCE_STRESS"
    force_stress.write_text(
        "\n".join(
            [
                "-1.0",
                "0.000100 0.000010 0.000020",
                "0.000010 0.000200 0.000030",
                "0.000020 0.000030 0.000300",
                "1",
                "1 14 0.0 0.0 0.0 Si",
            ]
        )
        + "\n"
    )
    stress = parse_siesta_stress_kbar(out, force_stress_path=force_stress)
    # Converted from Ry/Bohr^3 and reordered [xx,yy,zz,xy,yz,zx]
    assert np.allclose(
        stress,
        [14.710513, 29.421026, 44.131540, 1.471051, 4.413154, 2.942103],
        atol=1e-6,
    )


def test_compare_force_stress_prefers_sign_flip_when_better() -> None:
    reference = {
        "natoms": 2,
        "total_energy_ev": -10.0,
        "elapsed_seconds": 100.0,
        "forces_ev_per_ang": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        "stress_kbar": np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0], dtype=float),
    }
    candidate = {
        "natoms": 2,
        "total_energy_ev": -9.999,
        "elapsed_seconds": 20.0,
        "forces_ev_per_ang": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        "stress_kbar": np.array([-10.0, -20.0, -30.0, -1.0, -2.0, -3.0], dtype=float),
    }
    metrics = compare_force_stress(reference=reference, candidate=candidate, allow_stress_sign_flip=True)
    assert metrics["stress_sign_flipped"] is True
    assert metrics["stress_mae_kbar"] == 0.0
    assert metrics["speedup_vs_reference"] == 5.0


def test_threshold_eval_and_winner_selection() -> None:
    thresholds = BenchmarkThresholds(
        force_mae_eva=0.03,
        stress_mae_kbar=0.5,
        energy_mev_per_atom=2.0,
        min_speedup=3.0,
    )
    case_a = {
        "name": "a",
        "candidate": {"elapsed_seconds": 10.0},
        "evaluation": evaluate_thresholds(
            {
                "force_mae_eva": 0.01,
                "stress_mae_kbar": 0.2,
                "energy_mev_per_atom": 1.0,
                "speedup_vs_reference": 4.0,
            },
            thresholds,
        ),
    }
    case_b = {
        "name": "b",
        "candidate": {"elapsed_seconds": 8.0},
        "evaluation": evaluate_thresholds(
            {
                "force_mae_eva": 0.02,
                "stress_mae_kbar": 0.3,
                "energy_mev_per_atom": 1.5,
                "speedup_vs_reference": 3.2,
            },
            thresholds,
        ),
    }
    winner = choose_fastest_passing_case([case_a, case_b])
    assert winner is not None
    assert winner["name"] == "b"
