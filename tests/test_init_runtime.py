from pathlib import Path
import json
from types import SimpleNamespace

import click
import pytest

import wtec.cli as cli


def test_critical_runtime_version_issues_detects_out_of_range_numpy(monkeypatch) -> None:
    versions = {"numpy": "2.2.6", "scipy": "1.13.1"}

    monkeypatch.setattr(
        cli,
        "_installed_distribution_version",
        lambda _python, dist: versions.get(dist),
    )

    issues = cli._critical_runtime_version_issues("/fake/python")
    assert any("numpy 2.2.6" in item for item in issues)
    assert not any(item.startswith("scipy ") for item in issues)


def test_apply_runtime_env_from_init_state_sets_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    state_dir = tmp_path / ".wtec"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "init_state.json").write_text(
        json.dumps(
            {
                "runtime_env": {
                    "MPLCONFIGDIR": "/tmp/wtec-mpl",
                    "XDG_CACHE_HOME": "/tmp/wtec-xdg",
                }
            }
        )
    )

    cli._apply_runtime_env_from_init_state()

    assert cli.os.environ["MPLCONFIGDIR"] == "/tmp/wtec-mpl"
    assert cli.os.environ["XDG_CACHE_HOME"] == "/tmp/wtec-xdg"


def test_subprocess_env_for_python_prepends_venv_bin(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    env = cli._subprocess_env_for_python("/tmp/demo-venv/bin/python")
    assert env["PATH"].split(cli.os.pathsep)[0] == "/tmp/demo-venv/bin"


def test_update_init_state_deep_merges_solver_capabilities(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    state_dir = tmp_path / ".wtec"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "init_state.json").write_text(
        json.dumps(
            {
                "runtime_env": {"MPLCONFIGDIR": "/tmp/wtec-mpl"},
                "solver_capabilities": {
                    "local": {
                        "kwant": {"solver": "mumps", "mumps_available": True},
                    }
                },
            }
        )
    )

    cli._update_init_state(
        {
            "solver_capabilities": {
                "cluster": {
                    "kwant": {"solver": "scipy_fallback", "mumps_available": False},
                }
            }
        }
    )

    payload = json.loads((state_dir / "init_state.json").read_text())
    assert payload["runtime_env"]["MPLCONFIGDIR"] == "/tmp/wtec-mpl"
    assert payload["solver_capabilities"]["local"]["kwant"]["solver"] == "mumps"
    assert payload["solver_capabilities"]["cluster"]["kwant"]["solver"] == "scipy_fallback"


def test_update_init_state_deep_merges_rgf_router_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    state_dir = tmp_path / ".wtec"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "init_state.json").write_text(
        json.dumps(
            {
                "solver_capabilities": {
                    "cluster": {
                        "kwant": {"solver": "mumps", "mumps_available": True},
                    }
                }
            }
        )
    )

    cli._update_init_state(
        {
            "rgf": {
                "cluster": {
                    "ready": True,
                    "binary_id": cli.RGF_BINARY_ID,
                    "binary_path": "/remote/wtec_rgf_runner",
                    "numerical_status": "scaffold_only",
                }
            }
        }
    )

    payload = json.loads((state_dir / "init_state.json").read_text())
    assert payload["solver_capabilities"]["cluster"]["kwant"]["solver"] == "mumps"
    assert payload["rgf"]["cluster"]["binary_id"] == cli.RGF_BINARY_ID
    assert payload["rgf"]["cluster"]["ready"] is True


def test_verify_install_raises_on_kwant_abi_mismatch(monkeypatch) -> None:
    def fake_run(cmd, capture_output=True, text=True, **kwargs):
        code = cmd[-1]
        if "import kwant" in code:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="ValueError: numpy.dtype size changed, may indicate binary incompatibility.",
            )
        version = "ok"
        if "import numpy" in code:
            version = "1.26.4"
        elif "import scipy" in code:
            version = "1.13.1"
        return SimpleNamespace(returncode=0, stdout=f"{version}\n", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    with pytest.raises(click.ClickException, match="kwant: NumPy ABI mismatch"):
        cli._verify_install(
            python_executable="/fake/python",
            dry_run=False,
            check_kwant=True,
            check_berry=False,
        )


def test_verify_install_raises_on_missing_kwant_mumps(monkeypatch) -> None:
    def fake_run(cmd, capture_output=True, text=True, **kwargs):
        code = cmd[-1]
        version = "ok"
        if "import numpy" in code:
            version = "1.26.4"
        elif "import scipy" in code:
            version = "1.13.1"
        elif "import kwant" in code:
            version = "1.5.0"
        return SimpleNamespace(returncode=0, stdout=f"{version}\n", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(
        cli,
        "_probe_local_kwant_solver",
        lambda _python: {
            "mumps_available": False,
            "solver": "scipy_fallback",
            "reason": "mumps_unavailable:ImportError:no _mumps",
        },
    )

    with pytest.raises(click.ClickException, match="kwant-mumps"):
        cli._verify_install(
            python_executable="/fake/python",
            dry_run=False,
            check_kwant=True,
            check_kwant_mumps=True,
            check_berry=False,
        )


def test_write_run_report_uses_checkpoint_topology_summary_when_json_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint_dir = tmp_path / ".wtec" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "demo.json"
    checkpoint_file.write_text(
        json.dumps(
            {
                "stage": "DONE",
                "outputs": {
                    "topology_summary": {
                        "enabled": True,
                        "status": "failed",
                        "reason": "missing_fermi_ev",
                    }
                },
            }
        )
    )

    monkeypatch.setattr(cli, "_checkpoint_file_for_cfg", lambda cfg: checkpoint_file)

    report = cli._write_run_report({"name": "demo", "run_dir": str(run_dir)})
    assert report is not None
    _, json_path = report
    payload = json.loads(json_path.read_text())
    assert payload["topology_summary"]["reason"] == "missing_fermi_ev"
