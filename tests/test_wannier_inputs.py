from ase import Atoms

from wtec.wannier.inputs import generate_win


def _tap_like_atoms() -> Atoms:
    return Atoms(
        symbols=["Ta", "Ta", "P", "P"],
        scaled_positions=[
            (0.25, 0.75, 0.50),
            (0.00, 0.00, 0.00),
            (0.67, 0.17, 0.33),
            (0.42, 0.42, 0.83),
        ],
        cell=[
            (3.3249992, 0.0, 0.0),
            (0.0, 3.3249992, 0.0),
            (-1.66249857, -1.66249857, 5.70069652),
        ],
        pbc=True,
    )


def test_generate_win_includes_kmesh_controls() -> None:
    txt = generate_win(
        _tap_like_atoms(),
        "TaP",
        num_bands=52,
        fermi_energy=15.98,
        kpoints=(12, 12, 12),
        spinors=True,
    )
    assert "mp_grid : 12 12 12" in txt
    assert "search_shells = 300" in txt
    assert "kmesh_tol = 0.01" in txt
    assert "dis_num_iter  = 1000" in txt
    assert "num_iter      = 1000" in txt
