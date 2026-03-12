import pytest
import numpy as np

from wtec.rgf import (
    canonical_axis_permutation,
    effective_principal_layer_width,
    memory_per_rank_bytes,
    normalize_rgf_mode,
    plan_execution,
    normalize_transport_engine,
    phase1_alignment_issues,
    resolved_mpi_np,
    work_unit_count,
)
from wtec.wannier.parser import HoppingData, write_hr_dat


def _toy_hopping_data() -> HoppingData:
    return HoppingData(
        num_wann=4,
        r_vectors=np.array([(0, 0, 0), (2, 0, 0), (-2, 0, 0)], dtype=int),
        deg=np.array([1, 1, 1], dtype=int),
        H_R=np.zeros((3, 4, 4), dtype=complex),
    )


def test_normalize_transport_engine_accepts_auto_and_rgf() -> None:
    assert normalize_transport_engine("auto") == "auto"
    assert normalize_transport_engine("rgf") == "rgf"
    with pytest.raises(ValueError):
        normalize_transport_engine("tbtrans")


def test_normalize_rgf_mode_rejects_invalid() -> None:
    assert normalize_rgf_mode("periodic_transverse") == "periodic_transverse"
    with pytest.raises(ValueError):
        normalize_rgf_mode("bad")


def test_effective_principal_layer_width_respects_periodic_axis() -> None:
    hd = _toy_hopping_data()
    width = effective_principal_layer_width(
        hd,
        lead_axis="x",
        n_layers_x=8,
        n_layers_y=4,
        n_layers_z=3,
        mode="periodic_transverse",
        periodic_axis="y",
    )
    assert width == 2


def test_memory_per_rank_bytes_grows_quadratically() -> None:
    small = memory_per_rank_bytes(n_super=256, overhead_bytes=0)
    large = memory_per_rank_bytes(n_super=512, overhead_bytes=0)
    assert large > small * 3


def test_resolved_mpi_np_clips_to_safe_rank_cap() -> None:
    assert resolved_mpi_np(queue_cores=64, n_work_units=48, safe_rank_cap=12) == 12


def test_plan_execution_prefers_single_rank_threads_for_single_full_finite_point() -> None:
    plan = plan_execution(
        mode="full_finite",
        queue_cores=64,
        safe_rank_cap=64,
        n_work_units=1,
        parallel_policy="auto",
    )
    assert plan.parallel_policy == "single_point"
    assert plan.mpi_np == 1
    assert plan.omp_threads == 64


def test_work_unit_count_includes_mfp_points_for_transport_planning() -> None:
    assert work_unit_count(
        thicknesses=[4],
        mfp_lengths=[7],
        disorder_strengths=[0.0],
        n_ensemble=1,
        mode="periodic_transverse",
        periodic_k_count=8,
    ) == 16


def test_canonical_axis_permutation_reorders_noncanonical_axes() -> None:
    perm, width_axis = canonical_axis_permutation(
        lead_axis="z",
        thickness_axis="x",
        mode="periodic_transverse",
        periodic_axis="y",
    )
    assert perm == (2, 1, 0)
    assert width_axis == "y"


def test_phase1_alignment_issues_detect_partial_layers(tmp_path) -> None:
    hd = HoppingData(
        num_wann=1,
        r_vectors=np.array([(0, 0, 0), (2, 0, 0), (-2, 0, 0)], dtype=int),
        deg=np.array([1, 1, 1], dtype=int),
        H_R=np.zeros((3, 1, 1), dtype=complex),
    )
    hr_path = tmp_path / "toy_hr.dat"
    write_hr_dat(hr_path, hd, header="toy")
    issues = phase1_alignment_issues(
        hr_dat_path=hr_path,
        lead_axis="x",
        n_layers_x=3,
        n_layers_y=4,
        thicknesses=[1],
        mfp_n_layers_z=1,
        mfp_lengths=[3],
        mode="periodic_transverse",
        periodic_axis="y",
    )
    assert issues
    assert "transport_n_layers_x=3" in issues[0]


def test_phase1_alignment_issues_accept_full_layers(tmp_path) -> None:
    hd = HoppingData(
        num_wann=1,
        r_vectors=np.array([(0, 0, 0), (2, 0, 0), (-2, 0, 0)], dtype=int),
        deg=np.array([1, 1, 1], dtype=int),
        H_R=np.zeros((3, 1, 1), dtype=complex),
    )
    hr_path = tmp_path / "toy_hr.dat"
    write_hr_dat(hr_path, hd, header="toy")
    issues = phase1_alignment_issues(
        hr_dat_path=hr_path,
        lead_axis="x",
        n_layers_x=4,
        n_layers_y=4,
        thicknesses=[1],
        mfp_n_layers_z=1,
        mfp_lengths=[4],
        mode="periodic_transverse",
        periodic_axis="y",
    )
    assert issues == []
