from pathlib import Path

import numpy as np
import pytest

from wtec.wannier.delta_h import DeltaHError, apply_delta_h_to_hr_file, build_delta_h_artifact
from wtec.wannier.parser import HoppingData, read_hr_dat, write_hr_dat


def _write_win(path: Path, *, projection: str = "Ta:d") -> None:
    path.write_text(
        "\n".join(
            [
                "begin unit_cell_cart",
                "ang",
                "1 0 0",
                "0 1 0",
                "0 0 1",
                "end unit_cell_cart",
                "begin projections",
                projection,
                "end projections",
                "",
            ]
        )
    )


def _simple_hd(scale: float) -> HoppingData:
    r_vectors = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=int)
    deg = np.asarray([1, 1], dtype=int)
    h0 = np.asarray([[0.2, 0.01], [0.01, -0.2]], dtype=complex) * scale
    h1 = np.asarray([[0.03, 0.0], [0.0, -0.03]], dtype=complex) * scale
    return HoppingData(
        num_wann=2,
        r_vectors=r_vectors,
        deg=deg,
        H_R=np.asarray([h0, h1], dtype=complex),
    )


def test_delta_h_build_and_apply(tmp_path: Path) -> None:
    pes_hr = tmp_path / "pes_hr.dat"
    lcao_hr = tmp_path / "lcao_hr.dat"
    out_hr = tmp_path / "out_hr.dat"
    pes_win = tmp_path / "pes.win"
    lcao_win = tmp_path / "lcao.win"
    _write_win(pes_win)
    _write_win(lcao_win)

    write_hr_dat(pes_hr, _simple_hd(scale=2.0))
    write_hr_dat(lcao_hr, _simple_hd(scale=1.0))

    artifact = build_delta_h_artifact(
        pes_hr_dat_path=pes_hr,
        pes_win_path=pes_win,
        lcao_hr_dat_path=lcao_hr,
        lcao_win_path=lcao_win,
        material="TaP",
        fermi_pes_ev=0.0,
        fermi_lcao_ev=0.0,
        fit_window_ev=3.0,
        fit_kmesh=(1, 1, 1),
        alpha_grid=np.asarray([1.0], dtype=float),
    )
    apply_delta_h_to_hr_file(
        hr_dat_path=lcao_hr,
        output_hr_dat_path=out_hr,
        artifact=artifact,
        win_path=lcao_win,
    )
    out = read_hr_dat(out_hr)
    pes = read_hr_dat(pes_hr)
    assert out.num_wann == pes.num_wann
    # With alpha fixed to 1.0 and identical basis, corrected HR should match PES HR.
    assert np.allclose(out.H_R, pes.H_R)


def test_delta_h_rejects_projection_mismatch(tmp_path: Path) -> None:
    pes_hr = tmp_path / "pes_hr.dat"
    lcao_hr = tmp_path / "lcao_hr.dat"
    pes_win = tmp_path / "pes.win"
    lcao_win = tmp_path / "lcao.win"
    _write_win(pes_win, projection="Ta:d")
    _write_win(lcao_win, projection="Ta:s")

    write_hr_dat(pes_hr, _simple_hd(scale=2.0))
    write_hr_dat(lcao_hr, _simple_hd(scale=1.0))

    with pytest.raises(DeltaHError):
        build_delta_h_artifact(
            pes_hr_dat_path=pes_hr,
            pes_win_path=pes_win,
            lcao_hr_dat_path=lcao_hr,
            lcao_win_path=lcao_win,
            fit_kmesh=(1, 1, 1),
            alpha_grid=np.asarray([1.0], dtype=float),
        )
