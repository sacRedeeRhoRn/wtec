import json

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


def test_hybrid_adaptive_uses_provided_siesta_ldos_payload(tmp_path) -> None:
    payload = {
        "metric": 0.73,
        "source_engine": "siesta_surface_ldos",
        "source_kind": "provided",
    }
    path = tmp_path / "siesta_slab_ldos.json"
    path.write_text(json.dumps(payload))

    out = compute_arc_connectivity(
        _ToyTBModel(),
        thickness_uc=10,
        energy_ev=0.0,
        prefer_engine="hybrid_adaptive",
        siesta_slab_ldos_json=str(path),
        allow_proxy_fallback=False,
    )
    assert out["status"] == "ok"
    assert out["engine"] == "siesta_slab_ldos"
    assert out["source_engine"] == "siesta_surface_ldos"
    assert float(out["metric"]) == 0.73


def test_hybrid_adaptive_generates_node_guided_tb_metric() -> None:
    out = compute_arc_connectivity(
        _ToyTBModel(),
        thickness_uc=10,
        energy_ev=0.0,
        prefer_engine="hybrid_adaptive",
        siesta_slab_ldos_json=None,
        allow_proxy_fallback=False,
        node_scan={
            "status": "ok",
            "nodes": [
                {
                    "k_frac": [0.22, 0.31, 0.44],
                    "energy_rel_fermi_ev": 0.01,
                    "gap_ev": 0.01,
                    "chirality": 1,
                }
            ],
        },
        adaptive_k_cfg={
            "global_kmesh_xy": [4, 4],
            "local_kmesh_xy": [8, 8],
            "fallback_global_refine_kmesh_xy": [6, 6],
            "window_radius_frac_xy": [0.08, 0.08],
            "min_hotspots": 1,
            "max_hotspots": 4,
        },
    )
    assert out["status"] == "ok"
    assert out["engine"] == "tb_kresolved_adaptive_surface_spectral"
    assert out["source_engine"] == "tb_kresolved_adaptive_surface_spectral"
    assert 0.0 <= float(out["metric"]) <= 1.0
    assert "metric_global" in out
    assert int(out["hotspot_count"]) == 1
