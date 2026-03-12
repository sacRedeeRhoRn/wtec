import numpy as np
import pytest

from wtec.topology.node_scan import compute_weyl_velocity_tensor
from wtec.topology.surface_gf import _build_surface_hk
from wtec.topology.wilson_loop import compute_wilson_loop_chern
from wtec.transport.observables import (
    fuchs_sondheimer_rho,
    fuchs_sondheimer_rho_exact,
)


_SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


class _QWZModel:
    def __init__(self, mass: float = -1.0) -> None:
        self.mass = float(mass)
        self.lattice_vectors = np.eye(3, dtype=float)

    def hamiltonian_at_k(self, k_frac: np.ndarray) -> np.ndarray:
        kx, ky, _ = np.asarray(k_frac, dtype=float)
        x = 2.0 * np.pi * float(kx)
        y = 2.0 * np.pi * float(ky)
        dz = self.mass + np.cos(x) + np.cos(y)
        return (
            np.sin(x) * _SIGMA_X
            + np.sin(y) * _SIGMA_Y
            + dz * _SIGMA_Z
        )


class _LinearWeylModel:
    def __init__(self) -> None:
        self.lattice_vectors = np.diag([2.0, 2.0, 3.0]).astype(float)

    def hamiltonian_at_k(self, k_frac: np.ndarray) -> np.ndarray:
        q = 2.0 * np.pi * np.asarray(k_frac, dtype=float)
        return (
            np.sin(float(q[0])) * _SIGMA_X
            + np.sin(float(q[1])) * _SIGMA_Y
            + np.sin(float(q[2])) * _SIGMA_Z
        )


def test_fuchs_sondheimer_exact_keeps_thick_film_tail() -> None:
    rho_bulk = 1.0
    mfp_m = 1.0
    thickness_m = np.asarray([10.0], dtype=float)

    exact = float(fuchs_sondheimer_rho_exact(rho_bulk, mfp_m, thickness_m, specularity=0.0)[0])
    asymptotic = float(fuchs_sondheimer_rho(rho_bulk, mfp_m, thickness_m, specularity=0.0)[0])

    assert exact == pytest.approx(asymptotic, abs=1.0e-2)
    assert exact > 1.02


def test_wilson_loop_chern_uses_full_ky_slice_and_periodic_closure() -> None:
    result = compute_wilson_loop_chern(
        _QWZModel(mass=-1.0),
        n_kz=4,
        n_kx=20,
        n_ky=4,
        n_occ_bands=1,
    )

    assert result["status"] == "ok"
    assert result["jump_kz"] == []
    assert len(result["wcc"]) == 4
    assert len(result["wcc"][0]) == 4
    chern_profile = np.asarray(result["chern_profile"], dtype=float)
    assert np.all(np.abs(np.abs(chern_profile) - 1.0) < 0.2)
    assert abs(abs(float(result["chern_integrated"])) - 1.0) < 0.2


def test_weyl_velocity_tensor_converts_fractional_k_to_ev_angstrom() -> None:
    result = compute_weyl_velocity_tensor(
        _LinearWeylModel(),
        np.zeros(3, dtype=float),
        band_idx=0,
        delta=1.0e-4,
    )

    assert result["status"] == "ok"
    assert result["v_parallel_ev_ang"] == pytest.approx(2.0, rel=1.0e-3, abs=1.0e-3)
    assert result["v_perp_ev_ang"] == pytest.approx(3.0, rel=1.0e-3, abs=1.0e-3)
    assert result["v_fermi_ev_ang"] == pytest.approx(7.0 / 3.0, rel=1.0e-3, abs=1.0e-3)


def test_surface_gf_builds_principal_layer_for_longer_range_hopping() -> None:
    one = np.array([[1.0 + 0.0j]], dtype=complex)
    H_00, T_01 = _build_surface_hk(
        [
            (0, 0, 2, one),
            (0, 0, -2, one),
        ],
        n_orb=1,
        kx=0.0,
        ky=0.0,
    )

    assert H_00.shape == (2, 2)
    assert T_01.shape == (2, 2)
    assert np.allclose(H_00, np.zeros((2, 2), dtype=complex))
    assert np.allclose(T_01, np.eye(2, dtype=complex))
