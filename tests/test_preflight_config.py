import json
import os
from pathlib import Path
import sys
import types

import click
import pytest

import wtec.cli as cli


@pytest.fixture(autouse=True)
def _isolate_wtec_state(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WTEC_STATE_DIR", raising=False)


def _write_init_state(home: Path) -> None:
    d = home / ".wtec"
    d.mkdir(parents=True, exist_ok=True)
    (d / "init_state.json").write_text("{}")


def _write_init_state_with_solver(home: Path, *, scope: str, mumps_available: bool, python_executable: str) -> None:
    d = home / ".wtec"
    d.mkdir(parents=True, exist_ok=True)
    (d / "init_state.json").write_text(
        json.dumps(
            {
                "venv_python": "/tmp/fake-venv/bin/python",
                "solver_capabilities": {
                    scope: {
                        "kwant": {
                            "probe_completed": True,
                            "solver": "mumps" if mumps_available else "scipy_fallback",
                            "mumps_available": mumps_available,
                            "reason": None if mumps_available else "mumps_unavailable:ImportError:no _mumps",
                            "python_executable": python_executable,
                        }
                    }
                },
            }
        )
    )


def _write_init_state_with_rgf(home: Path, *, ready: bool, numerical_status: str = "scaffold_only") -> None:
    d = home / ".wtec"
    d.mkdir(parents=True, exist_ok=True)
    (d / "init_state.json").write_text(
        json.dumps(
            {
                "rgf": {
                    "cluster": {
                        "ready": ready,
                        "binary_id": cli.RGF_BINARY_ID,
                        "binary_path": "/remote/wtec_rgf_runner",
                        "numerical_status": numerical_status,
                        "note": "test-router",
                    }
                }
            }
        )
    )


def _base_cfg(hr_path: Path) -> dict:
    return {
        "material": "TaP",
        "structure_file": str(hr_path.with_name("structure.cif")),
        "run_profile": "smoke",
        "dft_reuse_mode": "all",
        "hr_dat_path": str(hr_path),
        "kpoints_scf": [8, 8, 8],
        "kpoints_nscf": [12, 12, 12],
        "transport_n_layers_x": 4,
        "transport_n_layers_y": 4,
        "topology": {
            "hr_scope": "per_variant",
            "n_layers_x": 4,
            "n_layers_y": 4,
        },
    }


def test_preflight_rejects_invalid_hr_scope(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["topology"]["hr_scope"] = "shared"
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_transport_layers_x_lt_2(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["transport_n_layers_x"] = 1
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_insufficient_weyl_kmesh(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["kpoints_nscf"] = [4, 4, 1]
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_invalid_dispersion_method(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["dft_dispersion"] = {"enabled": True, "method": "invalid"}
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_hybrid_with_non_qe_reference(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["dft_mode"] = "hybrid_qe_ref_siesta_variants"
    cfg["dft_engine"] = "siesta"
    cfg["topology"]["variant_dft_engine"] = "siesta"
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_hybrid_with_non_siesta_variant_engine(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["dft_mode"] = "hybrid_qe_ref_siesta_variants"
    cfg["dft_engine"] = "qe"
    cfg["topology"]["variant_dft_engine"] = "qe"
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_requires_sisl_interface_for_hybrid_variant_siesta(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["dft_mode"] = "hybrid_qe_ref_siesta_variants"
    cfg["dft_engine"] = "qe"
    cfg["topology"]["variant_dft_engine"] = "siesta"
    cfg["dft_siesta"] = {"wannier_interface": "builtin"}
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_invalid_dft_reference_reuse_policy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["dft_reference_reuse_policy"] = "bad_policy"
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_invalid_topology_tiering_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["topology"]["tiering"] = {"mode": "unsupported"}
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_invalid_siesta_slab_ldos_autogen_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["topology"]["siesta_slab_ldos_autogen"] = "bad_mode"
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_accepts_tb_kresolved_siesta_slab_ldos_autogen_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["topology"]["siesta_slab_ldos_autogen"] = "tb_kresolved"
    cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_invalid_adaptive_k_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["topology"]["adaptive_k"] = {"global_kmesh_xy": [16]}
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_strict_requires_dense_transport_sampling(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["run_profile"] = "strict"
    cfg["dft_reuse_mode"] = "none"
    cfg.pop("hr_dat_path", None)
    cfg["thicknesses"] = [3, 5, 7]
    cfg["mfp_lengths"] = [3, 5, 7]
    cfg["n_ensemble"] = 3
    cfg["transport_backend"] = "qsub"
    cfg["topology"].update({"backend": "qsub", "arc_engine": "siesta_slab_ldos", "node_method": "wannierberri_flux"})
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_strict_rejects_underresolved_hybrid_adaptive_kmeshes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["run_profile"] = "strict"
    cfg["dft_reuse_mode"] = "none"
    cfg.pop("hr_dat_path", None)
    cfg["thicknesses"] = [3, 5, 7, 9, 11]
    cfg["mfp_lengths"] = [3, 5, 7, 9, 11, 13, 15]
    cfg["n_ensemble"] = 30
    cfg["transport_backend"] = "qsub"
    cfg["topology"].update(
        {
            "backend": "qsub",
            "arc_engine": "hybrid_adaptive",
            "node_method": "wannierberri_flux",
            "coarse_kmesh": [20, 20, 20],
            "refine_kmesh": [7, 7, 7],
            "adaptive_k": {
                "global_kmesh_xy": [8, 8],
                "local_kmesh_xy": [16, 16],
                "fallback_global_refine_kmesh_xy": [12, 12],
                "surface_axis": "z",
                "energy_window_ev": 0.12,
                "require_inplane_transport": True,
            },
        }
    )
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_nonpositive_transport_mumps_nrhs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["transport_mumps_nrhs"] = 0
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_strict_rejects_proxy_topology_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["run_profile"] = "strict"
    cfg["dft_reuse_mode"] = "none"
    cfg.pop("hr_dat_path", None)
    cfg["thicknesses"] = [3, 5, 7, 9, 11]
    cfg["mfp_lengths"] = [3, 5, 7, 9, 11, 13, 15]
    cfg["n_ensemble"] = 30
    cfg["transport_backend"] = "qsub"
    cfg["topology"].update({"backend": "qsub", "arc_engine": "kwant", "node_method": "proxy"})
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_dual_family_requires_explicit_pes_reference_structure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    slab = tmp_path / "slab.cif"
    slab.write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg.pop("hr_dat_path", None)
    cfg["structure_file"] = str(slab)
    cfg["dft_mode"] = "dual_family"
    cfg["dft_pes_engine"] = "qe"
    cfg["dft_lcao_engine"] = "siesta"
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="dft_scf")


def test_preflight_hybrid_requires_explicit_pes_reference_structure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    slab = tmp_path / "slab.cif"
    slab.write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg.pop("hr_dat_path", None)
    cfg["structure_file"] = str(slab)
    cfg["dft_mode"] = "hybrid_qe_ref_siesta_variants"
    cfg["dft_pes_engine"] = "qe"
    cfg["dft_lcao_engine"] = "siesta"
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="dft_scf")


def test_pes_reference_can_generate_from_mp_id(tmp_path, monkeypatch) -> None:
    class _FakeStructure:
        def get_primitive_structure(self):
            return self

        def to(self, *, filename: str, fmt: str = "cif") -> None:
            Path(filename).write_text("fake_cif\n")

    class _FakeMPRester:
        def __init__(self, key: str) -> None:
            self.key = key

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def get_structure_by_material_id(self, material_id: str):
            assert self.key == "fake-key"
            assert material_id == "mp-1067587"
            return _FakeStructure()

    mp_api_mod = types.ModuleType("mp_api")
    mp_api_client_mod = types.ModuleType("mp_api.client")
    mp_api_client_mod.MPRester = _FakeMPRester
    mp_api_mod.client = mp_api_client_mod
    monkeypatch.setitem(sys.modules, "mp_api", mp_api_mod)
    monkeypatch.setitem(sys.modules, "mp_api.client", mp_api_client_mod)
    monkeypatch.setenv("MP_API_KEY", "fake-key")

    cfg = {"material": "TaP"}
    cfg["dft_pes_reference_mp_id"] = "mp-1067587"
    cfg["dft_pes_reference_use_primitive"] = True
    cfg["_runtime_config_dir"] = str(tmp_path)

    out_path = cli._ensure_pes_reference_structure_from_mp(cfg)
    out = tmp_path / "references" / "TaP_primitive_mp-1067587.cif"
    assert out.exists()
    assert out_path == str(out.resolve())
    assert cfg["dft_pes_reference_structure_file"] == str(out.resolve())


def test_load_runtime_dotenv_preserves_process_env(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir()
    cfg_dir = project_root / "configs"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "run.json"
    cfg_path.write_text("{}", encoding="utf-8")
    (project_root / ".env").write_text(
        "MP_API_KEY=\nTOPOSLAB_CLUSTER_HOST=dotenv-host\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(project_root)
    monkeypatch.setenv("MP_API_KEY", "shell-key")
    monkeypatch.delenv("TOPOSLAB_CLUSTER_HOST", raising=False)

    cli._load_runtime_dotenv(str(cfg_path))

    assert os.environ["MP_API_KEY"] == "shell-key"
    assert os.environ["TOPOSLAB_CLUSTER_HOST"] == "dotenv-host"


def test_preflight_dual_family_accepts_vasp_and_abacus_engines(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["dft_mode"] = "dual_family"
    cfg["dft_pes_engine"] = "vasp"
    cfg["dft_lcao_engine"] = "siesta"
    cfg["transport_engine"] = "kwant"
    cfg["transport_backend"] = "local"
    cfg["topology"]["backend"] = "local"
    cfg["dft_vasp"] = {"executable": "vasp_std"}
    cli._run_preflight(cfg, resume=False, stage="transport")

    cfg2 = _base_cfg(hr)
    cfg2["dft_mode"] = "dual_family"
    cfg2["dft_pes_engine"] = "qe"
    cfg2["dft_lcao_engine"] = "abacus"
    cfg2["transport_engine"] = "kwant"
    cfg2["transport_backend"] = "local"
    cfg2["topology"]["backend"] = "local"
    cfg2["dft_abacus"] = {"executable": "abacus", "basis_type": "lcao"}
    cli._run_preflight(cfg2, resume=False, stage="transport")


def test_preflight_rejects_invalid_anchor_transfer_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    pes = tmp_path / "pes_ref.cif"
    pes.write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg.pop("hr_dat_path", None)
    cfg["structure_file"] = str(pes)
    cfg["dft_mode"] = "dual_family"
    cfg["dft_pes_engine"] = "qe"
    cfg["dft_lcao_engine"] = "siesta"
    cfg["dft_pes_reference_structure_file"] = str(pes)
    cfg["dft_anchor_transfer"] = {"enabled": True, "mode": "bad"}
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_negative_siesta_mpi_np(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["dft_siesta"] = {"mpi_np_scf": -1}
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_invalid_siesta_factorization_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["dft_siesta"] = {"factorization_defaults": {"g3_32": {"mpi_np_scf": 0}}}
    with pytest.raises(click.UsageError):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_local_full_run_uses_existing_hr_without_cluster(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    structure = tmp_path / "structure.cif"
    structure.write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg["transport_backend"] = "local"
    cfg["topology"]["backend"] = "local"
    cli._run_preflight(cfg, resume=False, stage=None)
    out = capsys.readouterr().out
    assert "OK (local full run using config hr_dat_path)" in out


def test_run_requires_cluster_false_for_local_full_run_with_hr(tmp_path) -> None:
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    structure = tmp_path / "structure.cif"
    structure.write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg["transport_backend"] = "local"
    cfg["topology"]["backend"] = "local"
    assert cli._run_requires_cluster(cfg, resume=False, stage=None) is False


def test_preflight_rejects_recorded_cluster_kwant_without_mumps(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state_with_solver(
        tmp_path,
        scope="cluster",
        mumps_available=False,
        python_executable="python3",
    )
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["transport_backend"] = "qsub"
    cfg["transport_cluster_python_exe"] = "python3"
    with pytest.raises(click.UsageError, match="transport_require_mumps=true"):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_skips_recorded_cluster_solver_when_python_mismatch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state_with_solver(
        tmp_path,
        scope="cluster",
        mumps_available=False,
        python_executable="/remote/venv/bin/python",
    )
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["transport_backend"] = "qsub"
    cfg["transport_cluster_python_exe"] = "python3"
    cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_accepts_transport_periodic_y_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    structure = tmp_path / "structure.cif"
    structure.write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg["transport_kwant_mode"] = "periodic_y"
    cfg["transport_backend"] = "local"
    cfg["topology"]["backend"] = "local"
    cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_removed_tbtrans_transport_engine(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    (tmp_path / "structure.cif").write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg["transport_engine"] = "tbtrans"
    with pytest.raises(click.UsageError, match="Unsupported transport_engine"):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_accepts_auto_transport_engine(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    (tmp_path / "structure.cif").write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg["transport_engine"] = "auto"
    cfg["transport_backend"] = "local"
    cfg["topology"]["backend"] = "local"
    cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_auto_resolves_to_rgf_when_router_ready(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state_with_rgf(tmp_path, ready=True, numerical_status="phase1_ready")
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    (tmp_path / "structure.cif").write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg["transport_engine"] = "auto"
    cfg["transport_backend"] = "qsub"
    cfg["transport_rgf_mode"] = "periodic_transverse"
    cfg["transport_rgf_periodic_axis"] = "y"
    cfg["disorder_strengths"] = [0.0]
    cfg["n_ensemble"] = 1
    cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_rgf_local_backend(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    (tmp_path / "structure.cif").write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg["transport_engine"] = "rgf"
    cfg["transport_backend"] = "local"
    with pytest.raises(click.UsageError, match="requires transport.backend='qsub'"):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_rgf_scaffold_only_backend(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state_with_rgf(tmp_path, ready=True, numerical_status="scaffold_only")
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    (tmp_path / "structure.cif").write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg["transport_engine"] = "rgf"
    cfg["transport_backend"] = "qsub"
    with pytest.raises(click.UsageError, match="numerical transport core is not implemented yet"):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_accepts_phase1_ready_rgf_backend(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state_with_rgf(tmp_path, ready=True, numerical_status="phase1_ready")
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    (tmp_path / "structure.cif").write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg["transport_engine"] = "rgf"
    cfg["transport_backend"] = "qsub"
    cfg["disorder_strengths"] = [0.0]
    cfg["n_ensemble"] = 1
    cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_rgf_nonzero_disorder(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state_with_rgf(tmp_path, ready=True, numerical_status="phase1_ready")
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    (tmp_path / "structure.cif").write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg["transport_engine"] = "rgf"
    cfg["transport_backend"] = "qsub"
    cfg["disorder_strengths"] = [0.1]
    cfg["n_ensemble"] = 1
    with pytest.raises(click.UsageError, match="periodic_transverse"):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_accepts_rgf_full_finite_nonunit_ensemble(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state_with_rgf(tmp_path, ready=True, numerical_status="phase2_experimental")
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    (tmp_path / "structure.cif").write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg["transport_engine"] = "rgf"
    cfg["transport_backend"] = "qsub"
    cfg["transport_rgf_mode"] = "full_finite"
    cfg["disorder_strengths"] = [0.0]
    cfg["n_ensemble"] = 4
    cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_phase2_full_finite_kwant_exact(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state_with_rgf(tmp_path, ready=True, numerical_status="phase2_experimental")
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    script = tmp_path / "kwant_par_test.py"
    script.write_text("print('fixture')\n")
    (tmp_path / "structure.cif").write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg["transport_engine"] = "rgf"
    cfg["transport_backend"] = "qsub"
    cfg["transport_rgf_mode"] = "full_finite"
    cfg["transport_rgf_full_finite_sigma_backend"] = "kwant_exact"
    cfg["transport_rgf_full_finite_kwant_script"] = str(script)
    cfg["disorder_strengths"] = [0.0]
    cfg["n_ensemble"] = 1
    with pytest.raises(click.UsageError, match="internal-only"):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_rejects_full_finite_kwant_validation_without_script(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state_with_rgf(tmp_path, ready=True, numerical_status="phase2_experimental")
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    (tmp_path / "structure.cif").write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg["transport_engine"] = "rgf"
    cfg["transport_backend"] = "qsub"
    cfg["transport_rgf_mode"] = "full_finite"
    cfg["transport_rgf_validate_against"] = "kwant"
    cfg["disorder_strengths"] = [0.0]
    cfg["n_ensemble"] = 1
    with pytest.raises(click.UsageError, match="internal-only"):
        cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_accepts_full_finite_native_on_phase2_experimental(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state_with_rgf(tmp_path, ready=True, numerical_status="phase2_experimental")
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    (tmp_path / "structure.cif").write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg["transport_engine"] = "rgf"
    cfg["transport_backend"] = "qsub"
    cfg["transport_rgf_mode"] = "full_finite"
    cfg["transport_rgf_full_finite_sigma_backend"] = "native"
    cfg["disorder_strengths"] = [0.0]
    cfg["n_ensemble"] = 1
    cli._run_preflight(cfg, resume=False, stage="transport")


def test_preflight_accepts_full_finite_nonzero_disorder(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state_with_rgf(tmp_path, ready=True, numerical_status="phase2_experimental")
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    (tmp_path / "structure.cif").write_text("nonempty")
    cfg = _base_cfg(hr)
    cfg["transport_engine"] = "rgf"
    cfg["transport_backend"] = "qsub"
    cfg["transport_rgf_mode"] = "full_finite"
    cfg["disorder_strengths"] = [0.2]
    cfg["n_ensemble"] = 4
    cli._run_preflight(cfg, resume=False, stage="transport")
