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
    assert cfg["dft_siesta"]["factorization_defaults"] == {}
    assert cfg["dft_siesta"]["dm_mixing_weight"] == 0.10
    assert cfg["dft_siesta"]["dm_number_pulay"] == 8
    assert cfg["dft_siesta"]["electronic_temperature_k"] == 300.0
    assert cfg["dft_siesta"]["max_scf_iterations"] == 200
    assert cfg["transport_policy"] == "single_track"
    assert cfg["dft_reuse_mode"] == "none"
    assert cfg["dft_anchor_transfer"]["enabled"] is True
    assert cfg["dft_anchor_transfer"]["mode"] == "delta_h"
    assert cfg["run_profile"] == "strict"


def test_master_toml_maps_reuse_autotune_and_tiering(tmp_path: Path) -> None:
    data = {
        "project": {"name": "demo", "material": "TaP"},
        "run": {"name": "demo_run"},
        "dft": {
            "mode": "hybrid_qe_ref_siesta_variants",
            "reference": {"engine": "qe", "reuse_policy": "timestamp_only"},
            "variants": {"engine": "siesta"},
        },
        "transport": {
            "autotune": {
                "enabled": True,
                "scope": "per_queue",
                "cache_file": "~/.cache/wtec_transport_autotune.json",
                "benchmark_walltime": "00:05:00",
                "benchmark_n_ensemble": 2,
                "profiles": [
                    {"mpi_np": 1, "threads": "all"},
                    {"mpi_np": 2, "threads": "half"},
                ],
            }
        },
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
    assert cfg["transport_autotune"]["enabled"] is True
    assert cfg["transport_autotune"]["scope"] == "per_queue"
    assert cfg["transport_autotune"]["benchmark_n_ensemble"] == 2
    assert cfg["topology"]["tiering"]["mode"] == "two_tier"
    assert cfg["topology"]["tiering"]["refine_top_k_per_thickness"] == 3
    assert cfg["topology"]["tiering"]["always_include_pristine"] is False
    assert cfg["topology"]["tiering"]["screen"]["coarse_kmesh"] == [8, 8, 8]


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
    assert 'dm_mixing_weight = 0.10' in text
    assert 'mode = "dual_family"' in text
    assert 'reuse_policy = "strict_hash"' in text
    assert "[transport.autotune]" in text
    assert "[topology.tiering]" in text
    assert "[topology.tiering.screen]" in text
    assert "[topology.transport_probe]" in text
