from pathlib import Path

import wtec.cli as cli


def test_master_toml_dual_family_defaults_transport_policy_single_track(tmp_path: Path) -> None:
    data = {
        "project": {"name": "demo", "material": "TaP"},
        "run": {"name": "demo_run"},
        "dft": {
            "mode": "dual_family",
            "tracks": {
                "pes_reference": {
                    "family": "pes",
                    "engine": "qe",
                    "structure_file": "references/TaP_primitive.cif",
                },
                "lcao_upscaled": {
                    "family": "lcao",
                    "engine": "siesta",
                    "source": "variants",
                },
            },
        },
        "transport": {},
        "topology": {},
        "defect": {},
        "export": {},
        "layers": [],
    }
    cfg = cli._build_cfg_from_master_toml(data, source_path=tmp_path / "wtec_project.toml")
    assert cfg["dft_mode"] == "dual_family"
    assert cfg["dft_pes_engine"] == "qe"
    assert cfg["dft_lcao_engine"] == "siesta"
    assert cfg["dft_lcao_source"] == "variants"
    assert cfg["dft_pes_reference_structure_file"] == str(
        (tmp_path / "references" / "TaP_primitive.cif").resolve()
    )
    assert cfg["dft_engine"] == "qe"
    assert cfg["topology_variant_dft_engine"] == "siesta"
    assert cfg["dft_vasp"]["executable"] == "vasp_std"
    assert cfg["dft_abacus"]["executable"] == "abacus"
    assert cfg["dft_siesta"]["mpi_np_scf"] == 0
    assert cfg["dft_siesta"]["omp_threads_nscf"] == 0
    assert cfg["dft_siesta"]["variant_kpoints_scf"] == [4, 4, 4]
    assert cfg["dft_siesta"]["variant_kpoints_nscf"] == [6, 6, 6]
    assert cfg["dft_siesta"]["factorization_defaults"] == {}
    assert cfg["dft_siesta"]["dm_mixing_weight"] == 0.18
    assert cfg["dft_siesta"]["dm_number_pulay"] == 6
    assert cfg["dft_siesta"]["electronic_temperature_k"] == 300.0
    assert cfg["dft_siesta"]["max_scf_iterations"] == 120
    assert cfg["transport_policy"] == "single_track"
    assert cfg["transport_engine"] == "auto"
    assert cfg["dft_reuse_mode"] == "none"
    assert cfg["dft_anchor_transfer"]["enabled"] is True
    assert cfg["dft_anchor_transfer"]["mode"] == "delta_h"
    assert cfg["run_profile"] == "strict"


def test_master_toml_maps_reuse_and_tiering(tmp_path: Path) -> None:
    data = {
        "project": {"name": "demo", "material": "TaP"},
        "run": {"name": "demo_run"},
        "dft": {
            "mode": "hybrid_qe_ref_siesta_variants",
            "reference": {"engine": "qe", "reuse_policy": "timestamp_only"},
            "variants": {"engine": "siesta"},
        },
        "transport": {},
        "topology": {
            "tiering": {
                "mode": "two_tier",
                "refine_top_k_per_thickness": 3,
                "always_include_pristine": False,
                "selection_metric": "S_total",
                "screen": {
                    "arc_engine": "kwant",
                    "node_method": "proxy",
                    "coarse_kmesh": [8, 8, 8],
                    "refine_kmesh": [4, 4, 4],
                    "newton_max_iter": 12,
                    "max_candidates": 16,
                },
            }
        },
        "defect": {},
        "export": {},
        "layers": [],
    }
    cfg = cli._build_cfg_from_master_toml(data, source_path=tmp_path / "wtec_project.toml")
    assert cfg["dft_reference_reuse_policy"] == "timestamp_only"
    assert cfg["topology"]["tiering"]["mode"] == "two_tier"
    assert cfg["topology"]["tiering"]["refine_top_k_per_thickness"] == 3
    assert cfg["topology"]["tiering"]["always_include_pristine"] is False
    assert cfg["topology"]["tiering"]["screen"]["coarse_kmesh"] == [8, 8, 8]


def test_master_toml_maps_adaptive_k_defaults_and_overrides(tmp_path: Path) -> None:
    data = {
        "project": {"name": "demo", "material": "TaP"},
        "run": {"name": "demo_run"},
        "dft": {},
        "transport": {},
        "topology": {
            "adaptive_k": {
                "global_kmesh_xy": [18, 20],
                "local_kmesh_xy": [52, 56],
                "fallback_global_refine_kmesh_xy": [42, 44],
                "window_radius_frac_xy": [0.05, 0.07],
                "energy_window_ev": 0.10,
                "hotspot_gap_max_ev": 0.02,
                "max_hotspots": 6,
                "min_hotspots": 3,
                "dedup_radius_frac": 0.02,
                "require_inplane_transport": False,
            }
        },
        "defect": {},
        "export": {},
        "layers": [],
    }
    cfg = cli._build_cfg_from_master_toml(data, source_path=tmp_path / "wtec_project.toml")
    assert cfg["topology"]["arc_engine"] == "hybrid_adaptive"
    assert cfg["topology"]["adaptive_k"]["global_kmesh_xy"] == [18, 20]
    assert cfg["topology"]["adaptive_k"]["local_kmesh_xy"] == [52, 56]
    assert cfg["topology"]["adaptive_k"]["fallback_global_refine_kmesh_xy"] == [42, 44]
    assert cfg["topology"]["adaptive_k"]["window_radius_frac_xy"] == [0.05, 0.07]
    assert cfg["topology"]["adaptive_k"]["energy_window_ev"] == 0.10
    assert cfg["topology"]["adaptive_k"]["max_hotspots"] == 6
    assert cfg["topology"]["adaptive_k"]["require_inplane_transport"] is False


def test_default_project_template_includes_new_sections() -> None:
    text = cli._default_project_template_text()
    assert '[dft.tracks.pes_reference]' in text
    assert '[dft.tracks.lcao_upscaled]' in text
    assert '[dft.vasp]' in text
    assert '[dft.abacus]' in text
    assert '[dft.anchor_transfer]' in text
    assert 'mpi_np_scf = 0' in text
    assert 'omp_threads_wannier = 0' in text
    assert 'factorization_defaults = {}' in text
    assert 'variant_kpoints_scf = [4, 4, 4]' in text
    assert 'variant_kpoints_nscf = [6, 6, 6]' in text
    assert 'dm_mixing_weight = 0.18' in text
    assert 'mode = "dual_family"' in text
    assert 'reuse_policy = "strict_hash"' in text
    assert 'mp_id = "mp-1067587"' in text
    assert 'use_primitive = true' in text
    assert 'engine = "auto" # kwant|rgf|auto' in text
    assert 'rgf_mode = "periodic_transverse"' in text
    assert 'rgf_periodic_axis = "y"' in text
    assert 'rgf_parallel_policy = "auto"' in text
    assert 'rgf_threads_per_rank = "auto"' in text
    assert 'rgf_blas_backend = "auto"' in text
    assert 'rgf_validate_against = "none"' in text
    assert 'rgf_full_finite_sigma_backend = "native"' in text
    assert 'rgf_full_finite_kwant_script = ""' in text
    assert '[transport.tbtrans]' not in text
    assert "[topology.tiering]" in text
    assert "[topology.tiering.screen]" in text
    assert "[topology.adaptive_k]" in text
    assert 'arc_engine = "hybrid_adaptive"' in text
    assert "[topology.transport_probe]" in text


def test_master_toml_maps_transport_engine_kwant(tmp_path: Path) -> None:
    data = {
        "project": {"name": "demo", "material": "TaP"},
        "run": {"name": "demo_run"},
        "dft": {"mode": "dual_family"},
        "transport": {
            "engine": "kwant",
            "kwant_mode": "periodic_y",
            "mumps_nrhs": 8,
        },
        "topology": {},
        "defect": {},
        "export": {},
        "layers": [],
    }
    cfg = cli._build_cfg_from_master_toml(data, source_path=tmp_path / "wtec_project.toml")
    assert cfg["transport_engine"] == "kwant"
    assert cfg["transport_kwant_mode"] == "periodic_y"
    assert cfg["transport_mumps_nrhs"] == 8


def test_master_toml_maps_transport_engine_rgf(tmp_path: Path) -> None:
    data = {
        "project": {"name": "demo", "material": "TaP"},
        "run": {"name": "demo_run"},
        "dft": {"mode": "dual_family"},
        "transport": {
            "engine": "rgf",
            "rgf_mode": "full_finite",
            "rgf_periodic_axis": "z",
            "rgf_parallel_policy": "single_point",
            "rgf_threads_per_rank": 12,
            "rgf_blas_backend": "openblas",
            "rgf_validate_against": "kwant",
            "rgf_full_finite_sigma_backend": "kwant_exact",
            "rgf_full_finite_kwant_script": "benchmarks/kwant_par_test.py",
        },
        "topology": {},
        "defect": {},
        "export": {},
        "layers": [],
    }
    cfg = cli._build_cfg_from_master_toml(data, source_path=tmp_path / "wtec_project.toml")
    assert cfg["transport_engine"] == "rgf"
    assert cfg["transport_rgf_mode"] == "full_finite"
    assert cfg["transport_rgf_periodic_axis"] == "z"
    assert cfg["transport_rgf_parallel_policy"] == "single_point"
    assert cfg["transport_rgf_threads_per_rank"] == 12
    assert cfg["transport_rgf_blas_backend"] == "openblas"
    assert cfg["transport_rgf_validate_against"] == "kwant"
    assert cfg["transport_rgf_full_finite_sigma_backend"] == "kwant_exact"
    assert cfg["transport_rgf_full_finite_kwant_script"] == "benchmarks/kwant_par_test.py"


def test_master_toml_maps_pes_reference_mp_id(tmp_path: Path) -> None:
    data = {
        "project": {"name": "demo", "material": "TaP"},
        "run": {"name": "demo_run"},
        "dft": {
            "mode": "hybrid_qe_ref_siesta_variants",
            "reference": {"engine": "qe", "mp_id": "mp-1067587", "use_primitive": True},
            "variants": {"engine": "siesta"},
        },
        "transport": {},
        "topology": {},
        "defect": {},
        "export": {},
        "layers": [],
    }
    cfg = cli._build_cfg_from_master_toml(data, source_path=tmp_path / "wtec_project.toml")
    assert cfg["dft_mode"] == "hybrid_qe_ref_siesta_variants"
    assert cfg["dft_pes_engine"] == "qe"
    assert cfg["dft_lcao_engine"] == "siesta"
    assert cfg["dft_pes_reference_mp_id"] == "mp-1067587"
    assert cfg["dft_pes_reference_use_primitive"] is True


def test_master_toml_maps_transport_mumps_tuning(tmp_path: Path) -> None:
    data = {
        "project": {"name": "demo", "material": "TaP"},
        "run": {"name": "demo_run"},
        "dft": {},
        "transport": {
            "mumps_nrhs": 8,
            "mumps_ordering": "metis",
            "mumps_sparse_rhs": False,
        },
        "topology": {},
        "defect": {},
        "export": {},
        "layers": [],
    }
    cfg = cli._build_cfg_from_master_toml(data, source_path=tmp_path / "wtec_project.toml")
    assert cfg["transport_mumps_nrhs"] == 8
    assert cfg["transport_mumps_ordering"] == "metis"
    assert cfg["transport_mumps_sparse_rhs"] is False


def test_migrate_project_template_upgrades_arc_defaults(tmp_path: Path) -> None:
    legacy = tmp_path / "wtec_project.toml"
    legacy.write_text(
        "[topology]\n"
        "arc_engine = \"siesta_slab_ldos\"\n"
        "hr_scope = \"per_variant\"\n\n"
        "[topology.kmesh]\n"
        "coarse = [20, 20, 20]\n"
        "refine = [5, 5, 5]\n"
        "newton_max_iter = 50\n"
        "gap_threshold_ev = 0.05\n"
        "max_candidates = 64\n"
        "dedup_tol = 0.04\n\n"
        "[topology.tiering.screen]\n"
        "arc_engine = \"siesta_slab_ldos\"\n"
        "coarse_kmesh = [10, 10, 10]\n"
        "refine_kmesh = [3, 3, 3]\n\n"
        "[topology.score]\n"
        "w_topo = 0.70\n"
        "w_transport = 0.30\n"
    )

    changed = cli._migrate_project_template(legacy)
    migrated = legacy.read_text()

    assert changed is True
    assert 'arc_engine = "hybrid_adaptive"' in migrated
    assert 'arc_allow_proxy_fallback = false' in migrated
    assert 'node_method = "wannierberri_flux"' in migrated
    assert '[topology.adaptive_k]' in migrated
