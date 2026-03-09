import pytest
import numpy as np

from wtec.wannier.convergence import (
    WannierNotConvergedError,
    WannierTopologyError,
    assert_wannier_converged,
    assert_wannier_topology,
    assert_wannier_topology_from_files,
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


def test_wannier_max_iter_soft_accept_with_small_delta(tmp_path) -> None:
    wout = tmp_path / "soft.wout"
    wout.write_text(
        "Warning: Maximum number of disentanglement iterations reached\n"
        "Delta: O_D= -0.1508371E-03 O_OD= -0.7641326E-03 O_TOT= -0.9149697E-03\n"
        "All done: wannier90 exiting\n"
    )
    assert_wannier_converged(wout_path=wout)


class _TrivialTBModel:
    def hamiltonian_at_k(self, _k):  # noqa: ANN001
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def test_wannier_topology_gate_raises_for_trivial_weyl_model() -> None:
    with pytest.raises(WannierTopologyError):
        assert_wannier_topology(
            _TrivialTBModel(),
            material_class="weyl",
            n_kxy=4,
            min_chern=0.1,
        )


def test_wannier_topology_gate_skips_non_weyl_model() -> None:
    assert_wannier_topology(
        _TrivialTBModel(),
        material_class="generic",
        n_kxy=4,
        min_chern=0.1,
    )


def test_wannier_topology_from_files_skips_non_weyl(monkeypatch, tmp_path) -> None:
    import wtec.wannier.model as wmodel

    def _should_not_be_called(*_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("from_hr_dat should not be called for non-weyl materials")

    monkeypatch.setattr(
        wmodel.WannierTBModel,
        "from_hr_dat",
        classmethod(_should_not_be_called),
    )
    assert_wannier_topology_from_files(
        hr_dat_path=tmp_path / "dummy_hr.dat",
        material_class="generic",
    )
