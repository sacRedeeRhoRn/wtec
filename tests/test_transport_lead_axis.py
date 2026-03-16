from __future__ import annotations

import sys
import types

import numpy as np

from wtec.transport import conductance as c
from wtec.wannier.model import WannierTBModel


class _DummyModel:
    size = 1

    def __init__(self, hop: dict[tuple[int, int, int], np.ndarray]) -> None:
        self.hop = hop

    def hamilton(self, _k, convention=2):  # noqa: ARG002
        return np.zeros((1, 1), dtype=complex)


def test_required_lead_axis_cells_respects_cross_section_window() -> None:
    hop = {
        (0, 0, 0): np.array([[0.0]]),
        (6, 0, 0): np.array([[1.0]]),
        (8, 0, 6): np.array([[1.0]]),
        (8, 0, 7): np.array([[1.0]]),
    }
    tb = WannierTBModel(_DummyModel(hop), lattice_vectors=np.eye(3))

    req_nz3 = tb.required_lead_axis_cells(
        lead_axis="x",
        n_layers_x=4,
        n_layers_y=4,
        n_layers_z=3,
    )
    req_nz7 = tb.required_lead_axis_cells(
        lead_axis="x",
        n_layers_x=4,
        n_layers_y=4,
        n_layers_z=7,
    )

    assert req_nz3 == 6
    assert req_nz7 == 8


def test_required_lead_axis_cells_periodic_y_ignores_y_window() -> None:
    hop = {
        (0, 0, 0): np.array([[0.0]]),
        (6, 8, 0): np.array([[1.0]]),
    }
    tb = WannierTBModel(_DummyModel(hop), lattice_vectors=np.eye(3))

    req_finite_y = tb.required_lead_axis_cells(
        lead_axis="x",
        n_layers_x=7,
        n_layers_y=1,
        n_layers_z=4,
    )
    req_periodic_y = tb.required_lead_axis_cells(
        lead_axis="x",
        n_layers_x=7,
        n_layers_y=1,
        n_layers_z=4,
        periodic_axes=("y",),
    )

    assert req_finite_y == 2
    assert req_periodic_y == 6


def test_compute_conductance_vs_length_shifts_lengths_to_required(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "kwant", types.SimpleNamespace())

    class _TB:
        lattice_vectors = np.eye(3)

        def required_lead_axis_cells(self, **_kwargs):
            return 7

    def _fake_ensemble(**_kwargs):
        return np.array([1.0], dtype=float)

    def _fake_region(_lv, *, n_layers_x, n_layers_y, n_layers_z, lead_axis, thickness_axis):
        return {
            "length_m": float({"x": n_layers_x, "y": n_layers_y, "z": n_layers_z}[lead_axis]),
            "thickness_m": float({"x": n_layers_x, "y": n_layers_y, "z": n_layers_z}[thickness_axis]),
            "cross_section_m2": 1.0,
        }

    monkeypatch.setattr(c, "_ensemble_conductance", _fake_ensemble)
    monkeypatch.setattr(c, "region_geometry", _fake_region)

    out = c.compute_conductance_vs_length(
        _TB(),
        lengths=[3, 5, 7],
        disorder_strength=0.0,
        n_layers_z_fixed=5,
        n_layers_x_fixed=4,
        n_layers_y=4,
        lead_axis="x",
        thickness_axis="z",
    )
    assert out["lead_axis_min_cells_required"] == 7
    assert out["length_uc_requested"].tolist() == [3, 5, 7]
    assert out["length_uc"].tolist() == [7, 9, 11]


def test_compute_conductance_vs_thickness_raises_lead_axis_count(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "kwant", types.SimpleNamespace())

    class _TB:
        lattice_vectors = np.eye(3)

        def required_lead_axis_cells(self, *, n_layers_z, **_kwargs):
            return max(2, int(n_layers_z) + 1)

    def _fake_ensemble(**_kwargs):
        return np.array([1.0], dtype=float)

    def _fake_region(_lv, *, n_layers_x, n_layers_y, n_layers_z, lead_axis, thickness_axis):
        return {
            "length_m": float({"x": n_layers_x, "y": n_layers_y, "z": n_layers_z}[lead_axis]),
            "thickness_m": float({"x": n_layers_x, "y": n_layers_y, "z": n_layers_z}[thickness_axis]),
            "cross_section_m2": 1.0,
        }

    monkeypatch.setattr(c, "_ensemble_conductance", _fake_ensemble)
    monkeypatch.setattr(c, "region_geometry", _fake_region)

    out = c.compute_conductance_vs_thickness(
        _TB(),
        thicknesses=[3, 5, 7],
        disorder_strength=0.0,
        n_ensemble=1,
        energy=0.0,
        n_jobs=1,
        lead_axis="x",
        thickness_axis="z",
        n_layers_x=4,
        n_layers_y=4,
    )

    assert out["lead_axis_cells_requested"] == 4
    assert out["lead_axis_min_cells_required"] == 8
    assert out["lead_axis_cells_used"].tolist() == [8, 8, 8]


def test_single_conductance_requests_only_needed_lead_pair(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _SMatrix:
        def transmission(self, out_lead, in_lead):
            assert out_lead == 0
            assert in_lead == 1
            return 2.5

    def _fake_smatrix(fsys, energy, out_leads=None, in_leads=None):
        calls.append(
            {
                "fsys": fsys,
                "energy": energy,
                "out_leads": list(out_leads) if out_leads is not None else None,
                "in_leads": list(in_leads) if in_leads is not None else None,
            }
        )
        return _SMatrix()

    monkeypatch.setitem(sys.modules, "kwant", types.SimpleNamespace(smatrix=_fake_smatrix))

    class _Builder:
        def finalized(self):
            return "fake_fsys"

    class _TB:
        def to_kwant_builder(self, **_kwargs):
            return _Builder()

    g = c._single_conductance(
        _TB(),
        n_layers_x=2,
        n_layers_y=2,
        n_layers_z=2,
        lead_axis="x",
        disorder_strength=0.0,
        energy=0.125,
        seed=7,
        lead_onsite_eV=0.0,
        use_clean_cache=False,
    )

    assert g == 2.5
    assert len(calls) == 1
    assert calls[0]["fsys"] == "fake_fsys"
    assert calls[0]["energy"] == 0.125
    assert calls[0]["out_leads"] == [0]
    assert calls[0]["in_leads"] == [1]


def test_ensemble_conductance_dispatches_to_periodic_y_for_clean_mode(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_periodic(**kwargs):
        calls.append({"mode": "periodic_y", **kwargs})
        return 3.75

    def _fake_single(**kwargs):
        calls.append({"mode": "single", **kwargs})
        return 1.25

    monkeypatch.setattr(c, "_single_conductance_periodic_y", _fake_periodic)
    monkeypatch.setattr(c, "_single_conductance", _fake_single)

    out = c._ensemble_conductance(
        tb_model=object(),
        n_layers_x=7,
        n_layers_y=8,
        n_layers_z=4,
        lead_axis="x",
        disorder_strength=0.0,
        n_ensemble=1,
        energy=0.0,
        n_jobs=1,
        base_seed=11,
        lead_onsite_eV=0.0,
        kwant_mode="periodic_y",
    )

    assert out.tolist() == [3.75]
    assert len(calls) == 1
    assert calls[0]["mode"] == "periodic_y"
    assert calls[0]["n_layers_y"] == 8
