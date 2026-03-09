import numpy as np

from wtec.topology.arc_scan import compute_arc_connectivity


class _ToyTBModel:
    num_orbitals = 1

    def _iter_hoppings(self):
        yield 0, 0, 0, np.array([[0.0]], dtype=complex)
        yield 1, 0, 0, np.array([[-0.8]], dtype=complex)
        yield -1, 0, 0, np.array([[-0.8]], dtype=complex)
        yield 0, 1, 0, np.array([[-0.7]], dtype=complex)
        yield 0, -1, 0, np.array([[-0.7]], dtype=complex)
        yield 0, 0, 1, np.array([[-0.5]], dtype=complex)
        yield 0, 0, -1, np.array([[-0.5]], dtype=complex)


class _NoHoppingIterator:
    num_orbitals = 1



def test_tb_kresolved_arc_metric_returns_ok() -> None:
    out = compute_arc_connectivity(
        _ToyTBModel(),
        thickness_uc=4,
        energy_ev=0.0,
        n_layers_x=4,
        n_layers_y=4,
        prefer_engine="tb_kresolved",
        allow_proxy_fallback=False,
        kmesh_xy=(4, 4),
        broadening_ev=0.08,
    )
    assert out["status"] == "ok"
    assert 0.0 <= float(out["metric"]) <= 1.0
    assert out["engine"] == "tb_kresolved_surface_spectral"



def test_tb_kresolved_arc_metric_fails_without_hopping_iterator() -> None:
    out = compute_arc_connectivity(
        _NoHoppingIterator(),
        thickness_uc=4,
        prefer_engine="tb_kresolved",
        allow_proxy_fallback=False,
    )
    assert out["status"] == "failed"
    assert "hopping_iterator" in str(out.get("reason", ""))
