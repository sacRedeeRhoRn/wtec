from pathlib import Path

import click
import pytest

import wtec.cli as cli


def _write_init_state(home: Path) -> None:
    d = home / ".wtec"
    d.mkdir(parents=True, exist_ok=True)
    (d / "init_state.json").write_text("{}")


def _base_cfg(hr_path: Path) -> dict:
    return {
        "material": "TaP",
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


def test_preflight_rejects_invalid_transport_autotune_scope(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["transport_autotune"] = {"enabled": True, "scope": "global"}
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


def test_preflight_dual_family_accepts_vasp_and_abacus_engines(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    _write_init_state(tmp_path)
    hr = tmp_path / "x_hr.dat"
    hr.write_text("dummy")
    cfg = _base_cfg(hr)
    cfg["dft_mode"] = "dual_family"
    cfg["dft_pes_engine"] = "vasp"
    cfg["dft_lcao_engine"] = "siesta"
    cfg["dft_vasp"] = {"executable": "vasp_std"}
    cli._run_preflight(cfg, resume=False, stage="transport")

    cfg2 = _base_cfg(hr)
    cfg2["dft_mode"] = "dual_family"
    cfg2["dft_pes_engine"] = "qe"
    cfg2["dft_lcao_engine"] = "abacus"
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
