import pytest

from wtec.wannier.convergence import (
    WannierNotConvergedError,
    assert_wannier_converged,
)


def test_wannier_nonconvergence_detected(tmp_path) -> None:
    wout = tmp_path / "TaP.wout"
    wout.write_text("<<< Warning: Maximum number of disentanglement iterations reached >>>\n")
    win = tmp_path / "TaP.win"
    win.write_text(
        "dis_num_iter = 500\n"
        "dis_win_min = -4.0\n"
        "dis_win_max = 16.0\n"
        "dis_froz_min = -1.0\n"
        "dis_froz_max = 0.2\n"
    )

    with pytest.raises(WannierNotConvergedError) as ei:
        assert_wannier_converged(wout_path=wout, win_path=win)
    msg = str(ei.value)
    assert "dis_num_iter=500" in msg
    assert "dis_win=(-4.0,16.0)" in msg


def test_wannier_convergence_ok(tmp_path) -> None:
    wout = tmp_path / "ok.wout"
    wout.write_text("Time to disentangle bands 0.100 (sec)\n")
    assert_wannier_converged(wout_path=wout)

