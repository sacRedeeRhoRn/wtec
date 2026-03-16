import pytest

from wtec.topology.adaptive_k import (
    is_node_signal_weak,
    project_kfrac_to_surface,
    select_node_projected_hotspots,
)


def test_project_kfrac_to_surface_wraps_fractional_coordinates() -> None:
    uv = project_kfrac_to_surface([1.10, -0.20, 0.35], "z")
    assert uv[0] == pytest.approx(0.10)
    assert uv[1] == pytest.approx(0.80)


def test_select_node_projected_hotspots_filters_and_deduplicates() -> None:
    node_scan = {
        "status": "ok",
        "nodes": [
            {
                "k_frac": [0.02, 0.98, 0.11],
                "energy_rel_fermi_ev": 0.01,
                "gap_ev": 0.010,
                "chirality": 1,
            },
            {
                "k_frac": [0.99, 0.01, 0.12],
                "energy_rel_fermi_ev": 0.015,
                "gap_ev": 0.014,
                "chirality": -1,
            },
            {
                "k_frac": [0.50, 0.50, 0.20],
                "energy_rel_fermi_ev": 0.18,
                "gap_ev": 0.010,
            },
            {
                "k_frac": [0.40, 0.30, 0.25],
                "energy_rel_fermi_ev": -0.01,
                "gap_ev": 0.020,
                "chirality": 1,
            },
        ],
    }

    hotspots = select_node_projected_hotspots(
        node_scan,
        surface_axis="z",
        energy_window_ev=0.12,
        hotspot_gap_max_ev=0.03,
        max_hotspots=8,
        dedup_radius_frac=0.05,
    )

    assert len(hotspots) == 2
    assert hotspots[0]["source_node_index"] == 0
    assert hotspots[1]["source_node_index"] == 3


def test_is_node_signal_weak_requires_enough_low_gap_hotspots() -> None:
    weak = is_node_signal_weak(
        {"status": "ok"},
        [{"gap_ev": 0.01}, {"gap_ev": 0.02}],
        min_hotspots=4,
    )
    assert weak is True

    strong = is_node_signal_weak(
        {"status": "ok"},
        [
            {"gap_ev": 0.01},
            {"gap_ev": 0.02},
            {"gap_ev": 0.015},
            {"gap_ev": 0.012},
        ],
        min_hotspots=4,
    )
    assert strong is False
