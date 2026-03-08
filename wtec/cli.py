"""wtec command-line interface."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import click

# ---------------------------------------------------------------------------
# Dependency manifest
# ---------------------------------------------------------------------------

CORE_DEPS = [
    "click>=8.1",
    # Keep kwant ABI-compatible defaults (NumPy 2.x breaks many wheels/build flows).
    "numpy>=1.24,<2",
    "scipy>=1.10,<1.14",
    "ase>=3.22",
    "pymatgen>=2024.6.10",
    "mp-api>=0.43,<0.44",
    "tbmodels>=1.4",
    "paramiko>=3.0,<4",
    "joblib>=1.3",
    "mpi4py>=3.1",
    "matplotlib>=3.7",
    "sisl>=0.14",
    # Pin to the last NumPy<2-compatible release to keep kwant ABI stable.
    "wannierberri==1.0.1",
    "ray>=2.10,<3",
    "python-dotenv>=1.0",
    "tomli>=2.0; python_version < '3.11'",
]

EXTRA_DEPS = {
    "berry": ["wannierberri==1.0.1", "ray>=2.10,<3"],
}

# Local kwant source (sibling directory of this package's project root)
_THIS_DIR = Path(__file__).resolve().parent.parent   # .../wtec/
_DEFAULT_KWANT_SRC = _THIS_DIR.parent / "kwant"


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="wtec")
@click.pass_context
def main(ctx: click.Context) -> None:
    """wtec — end-to-end topological semimetal transport workflow.

    \b
    Typical first use:
        wtec init            # install all dependencies
        wtec run --help      # run a full pipeline
    """
    if ctx.invoked_subcommand and ctx.invoked_subcommand != "init":
        _maybe_reexec_in_init_venv()


# ---------------------------------------------------------------------------
# wtec init
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--kwant-src",
    default=None,
    metavar="PATH",
    help="Path to local kwant source directory. "
         f"Auto-detected: {_DEFAULT_KWANT_SRC}",
)
@click.option(
    "--extra",
    multiple=True,
    type=click.Choice(list(EXTRA_DEPS.keys()) + ["all"]),
    help="Optional dependency groups to install (can repeat). "
         "'all' installs everything.",
)
@click.option(
    "--no-kwant",
    is_flag=True,
    default=False,
    help="Skip kwant installation (if already installed).",
)
@click.option(
    "--strict-kwant/--no-strict-kwant",
    default=True,
    help="Fail init if kwant installation is skipped/failed.",
)
@click.option(
    "--venv-path",
    default=".venv",
    metavar="PATH",
    show_default=True,
    help="Virtual environment path to create/reuse for all installs.",
)
@click.option(
    "--python-exe",
    default=None,
    metavar="PATH",
    help="Python interpreter to create the venv with (default: current python).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would be installed without doing it.",
)
@click.option("--cluster-host", default=None, help="Set TOPOSLAB_CLUSTER_HOST in .env.")
@click.option("--cluster-port", type=int, default=None, help="Set TOPOSLAB_CLUSTER_PORT in .env.")
@click.option("--cluster-user", default=None, help="Set TOPOSLAB_CLUSTER_USER in .env.")
@click.option("--cluster-pass", default=None, help="Set TOPOSLAB_CLUSTER_PASS in .env.")
@click.option("--cluster-key", default=None, help="Set TOPOSLAB_CLUSTER_KEY in .env.")
@click.option("--mp-api-key", default=None, help="Set MP_API_KEY in .env for slab generation.")
@click.option("--remote-workdir", default=None, help="Set TOPOSLAB_REMOTE_WORKDIR in .env.")
@click.option("--mpi-cores", type=int, default=None, help="Set TOPOSLAB_MPI_CORES in .env.")
@click.option(
    "--mpi-cores-by-queue",
    default=None,
    help="Set TOPOSLAB_MPI_CORES_BY_QUEUE in .env, e.g. g4:64,g3:48,g2:32,g1:16.",
)
@click.option("--qe-pseudo-dir", default=None, help="Set TOPOSLAB_QE_PSEUDO_DIR in .env.")
@click.option("--siesta-pseudo-dir", default=None, help="Set TOPOSLAB_SIESTA_PSEUDO_DIR in .env.")
@click.option("--vasp-pseudo-dir", default=None, help="Set TOPOSLAB_VASP_PSEUDO_DIR in .env.")
@click.option("--abacus-pseudo-dir", default=None, help="Set TOPOSLAB_ABACUS_PSEUDO_DIR in .env.")
@click.option("--abacus-orbital-dir", default=None, help="Set TOPOSLAB_ABACUS_ORBITAL_DIR in .env.")
@click.option(
    "--qe-pseudo-source-dir",
    default=None,
    help=(
        "Set TOPOSLAB_QE_PSEUDO_SOURCE_DIR in .env. "
        "Used by init to seed TOPOSLAB_QE_PSEUDO_DIR when empty."
    ),
)
@click.option("--omp-threads", type=int, default=None, help="Set TOPOSLAB_OMP_THREADS in .env.")
@click.option(
    "--cluster-modules",
    default=None,
    help="Set TOPOSLAB_CLUSTER_MODULES in .env (comma-separated module names).",
)
@click.option(
    "--cluster-bin-dirs",
    default=None,
    help="Set TOPOSLAB_CLUSTER_BIN_DIRS in .env (comma-separated absolute bin paths).",
)
@click.option(
    "--qe-source-dir",
    default=None,
    help="Set TOPOSLAB_QE_SOURCE_DIR in .env for auto-build of pw.x/pw2wannier90.x.",
)
@click.option(
    "--siesta-source-dir",
    default=None,
    help="Set TOPOSLAB_SIESTA_SOURCE_DIR in .env for auto-build of siesta.",
)
@click.option(
    "--abacus-source-dir",
    default=None,
    help="Set TOPOSLAB_ABACUS_SOURCE_DIR in .env for optional ABACUS source checks.",
)
@click.option(
    "--wannier90-source-dir",
    default=None,
    help="Set TOPOSLAB_WANNIER90_SOURCE_DIR in .env for auto-build of wannier90.x.",
)
@click.option(
    "--cluster-build-jobs",
    type=int,
    default=None,
    help="Set TOPOSLAB_CLUSTER_BUILD_JOBS in .env (make -j for cluster source builds).",
)
@click.option(
    "--prepare-cluster-tools/--no-prepare-cluster-tools",
    default=True,
    help="Build/install missing QE/Wannier executables on cluster during init.",
)
@click.option(
    "--prepare-cluster-pseudos/--no-prepare-cluster-pseudos",
    default=True,
    help="Create/fill cluster pseudo directory when empty during init.",
)
@click.option(
    "--prepare-cluster-python/--no-prepare-cluster-python",
    default=True,
    help="Prepare remote cluster Python packages for kwant/wannierberri during init.",
)
@click.option(
    "--cluster-python-exe",
    default="python3",
    show_default=True,
    help="Remote Python executable for cluster-side package setup.",
)
@click.option(
    "--cluster-python-berry/--no-cluster-python-berry",
    default=True,
    help="Also prepare remote wannierberri(+ray) during init.",
)
@click.option(
    "--overwrite-env",
    is_flag=True,
    default=False,
    help="Regenerate .env from template before applying options.",
)
@click.option(
    "--overwrite-slab-template",
    is_flag=True,
    default=False,
    help="Rewrite wtec_slab_template.toml if it already exists.",
)
@click.option(
    "--validate-cluster/--no-validate-cluster",
    default=True,
    show_default=True,
    help="After setup, validate SSH/qsub/mpirun/pseudo_dir against cluster config in .env.",
)
def init(
    kwant_src: str | None,
    extra: tuple,
    no_kwant: bool,
    strict_kwant: bool,
    venv_path: str,
    python_exe: str | None,
    dry_run: bool,
    cluster_host: str | None,
    cluster_port: int | None,
    cluster_user: str | None,
    cluster_pass: str | None,
    cluster_key: str | None,
    mp_api_key: str | None,
    remote_workdir: str | None,
    mpi_cores: int | None,
    mpi_cores_by_queue: str | None,
    qe_pseudo_dir: str | None,
    siesta_pseudo_dir: str | None,
    vasp_pseudo_dir: str | None,
    abacus_pseudo_dir: str | None,
    abacus_orbital_dir: str | None,
    qe_pseudo_source_dir: str | None,
    omp_threads: int | None,
    cluster_modules: str | None,
    cluster_bin_dirs: str | None,
    qe_source_dir: str | None,
    siesta_source_dir: str | None,
    abacus_source_dir: str | None,
    wannier90_source_dir: str | None,
    cluster_build_jobs: int | None,
    prepare_cluster_tools: bool,
    prepare_cluster_pseudos: bool,
    prepare_cluster_python: bool,
    cluster_python_exe: str,
    cluster_python_berry: bool,
    overwrite_env: bool,
    overwrite_slab_template: bool,
    validate_cluster: bool,
) -> None:
    """Install all wtec dependencies and set up workspace.

    \b
    This command:
      1. Creates/reuses a Python virtual environment
      2. pip-installs all core Python dependencies
      3. Installs optional extras if requested
      4. Installs this wtec package in editable mode
      5. Installs kwant (local source first, then PyPI fallback)
      6. Creates/updates .env.example and .env
      7. Verifies the installation inside the venv
      8. Optionally validates cluster connectivity/resources
      9. Prepares remote cluster executable toolchain (QE/Wannier) if missing
      10. Prepares remote pseudopotential directory if empty
      11. Prepares remote cluster Python env for kwant/wannierberri

    \b
    Examples:
        wtec init
        wtec init --venv-path .venv
        wtec init --python-exe /usr/bin/python3.12
        wtec init --extra berry
        wtec init --kwant-src /path/to/kwant
        wtec init --cluster-host 202.30.0.129 --cluster-port 54329 --cluster-user msj
        wtec init --qe-pseudo-dir /home/msj/src/QE_pseudo/pslibrary/pbe/PSEUDOPOTENTIALS
        wtec init --validate-cluster
        wtec init --dry-run
    """
    click.echo(click.style("wtec init", fg="cyan", bold=True))
    click.echo("─" * 50)
    _load_runtime_dotenv(None)

    venv_dir = Path(venv_path).expanduser()
    python_for_venv = python_exe or sys.executable
    resolved_remote_workdir = remote_workdir or str((Path.cwd() / "remote_runs").resolve())
    if remote_workdir is None:
        click.echo(
            "  TOPOSLAB_REMOTE_WORKDIR not provided; "
            f"defaulting to: {resolved_remote_workdir}"
        )
    env_updates = _collect_env_updates(
        cluster_host=cluster_host,
        cluster_port=cluster_port,
        cluster_user=cluster_user,
        cluster_pass=cluster_pass,
        cluster_key=cluster_key,
        mp_api_key=mp_api_key,
        remote_workdir=resolved_remote_workdir,
        mpi_cores=mpi_cores,
        mpi_cores_by_queue=mpi_cores_by_queue,
        pbs_queue=None,
        pbs_queue_priority=None,
        qe_pseudo_dir=qe_pseudo_dir,
        siesta_pseudo_dir=siesta_pseudo_dir,
        vasp_pseudo_dir=vasp_pseudo_dir,
        abacus_pseudo_dir=abacus_pseudo_dir,
        abacus_orbital_dir=abacus_orbital_dir,
        qe_pseudo_source_dir=qe_pseudo_source_dir,
        omp_threads=omp_threads,
        cluster_modules=cluster_modules,
        cluster_bin_dirs=cluster_bin_dirs,
        qe_source_dir=qe_source_dir,
        siesta_source_dir=siesta_source_dir,
        abacus_source_dir=abacus_source_dir,
        wannier90_source_dir=wannier90_source_dir,
        cluster_build_jobs=cluster_build_jobs,
    )
    env_updates = _merge_init_interactive_env_updates(env_updates)
    _apply_env_updates_to_process(env_updates)

    # ── 1. Virtual environment ───────────────────────────────────────────
    click.echo(click.style("\n[1/8] Virtual environment", bold=True))
    venv_python = _ensure_venv(
        venv_dir=venv_dir,
        python_executable=python_for_venv,
        dry_run=dry_run,
    )
    pip = [str(venv_python), "-m", "pip", "install", "--upgrade"]

    # ── 2. Core dependencies ─────────────────────────────────────────────
    click.echo(click.style("\n[2/8] Core dependencies", bold=True))
    _pip_install(pip, CORE_DEPS, dry_run=dry_run)

    # ── 3. Optional extras ───────────────────────────────────────────────
    berry_enabled = False
    if extra:
        groups = set(extra)
        if "all" in groups:
            groups = set(EXTRA_DEPS.keys())
        berry_enabled = "berry" in groups
        click.echo(click.style(f"\n[3/8] Optional extras: {sorted(groups)}", bold=True))
        for group in sorted(groups):
            _pip_install(pip, EXTRA_DEPS[group], dry_run=dry_run)
    else:
        click.echo(click.style("\n[3/8] Optional extras: skipped (use --extra to add)", fg="yellow"))

    # ── 4. Local wtec package ────────────────────────────────────────────
    click.echo(click.style("\n[4/8] wtec package", bold=True))
    click.echo(f"  Installing wtec from {_THIS_DIR}")
    _pip_install(
        pip,
        [str(_THIS_DIR)],
        dry_run=dry_run,
        editable=True,
        no_deps=True,
    )

    # ── 5. Kwant installation ────────────────────────────────────────────
    click.echo(click.style("\n[5/8] kwant", bold=True))
    if dry_run:
        click.echo("  (dry-run) would verify/install kwant in venv")
        kwant_ready = True
    else:
        kwant_ready = _module_importable(str(venv_python), "kwant")
    if kwant_ready:
        click.echo("  kwant already available in venv")
    if no_kwant:
        click.echo("  skipped (--no-kwant)")
        if strict_kwant:
            raise click.ClickException("--strict-kwant is incompatible with --no-kwant.")
    elif not kwant_ready:
        kwant_path = Path(kwant_src) if kwant_src else _DEFAULT_KWANT_SRC
        local_ok = False
        py_ok, why_not = _local_source_python_compatible(kwant_path, str(venv_python))
        if kwant_path.exists() and py_ok:
            click.echo(f"  Installing kwant from {kwant_path}")
            local_ok = _pip_install(
                pip,
                [str(kwant_path)],
                dry_run=dry_run,
                editable=True,
                raise_on_error=False,
            )
            if not local_ok:
                click.echo(click.style("  Local kwant install failed; trying PyPI.", fg="yellow"))
        elif kwant_path.exists() and not py_ok:
            click.echo(
                click.style(
                    f"  Skipping local kwant source ({kwant_path}): {why_not}",
                    fg="yellow",
                )
            )
        else:
            click.echo(click.style(f"  Local kwant source not found: {kwant_path}", fg="yellow"))

        if not local_ok:
            click.echo("  Installing kwant from PyPI")
            pypi_ok = _pip_install(
                pip,
                ["kwant"],
                dry_run=dry_run,
                editable=False,
                raise_on_error=False,
            )
            if not pypi_ok:
                click.echo(click.style("  PyPI kwant install failed; retrying without build isolation.", fg="yellow"))
                pypi_ok = _pip_install(
                    pip,
                    ["kwant"],
                    dry_run=dry_run,
                    editable=False,
                    raise_on_error=False,
                    extra_args=["--no-build-isolation"],
                )
            if not pypi_ok:
                click.echo(
                    click.style(
                        "  Preparing local build deps for kwant and retrying source build.",
                        fg="yellow",
                    )
                )
                _pip_install(
                    pip,
                    [
                        "numpy<2",
                        "scipy<1.14",
                        "cython",
                        "meson-python",
                        "ninja",
                        "setuptools-scm",
                    ],
                    dry_run=dry_run,
                    editable=False,
                    raise_on_error=False,
                )
                pypi_ok = _pip_install(
                    pip,
                    ["kwant<2"],
                    dry_run=dry_run,
                    editable=False,
                    raise_on_error=False,
                    extra_args=[
                        "--no-build-isolation",
                        "--no-binary=:all:",
                        "--no-cache-dir",
                        "--force-reinstall",
                    ],
                )
            if not pypi_ok and strict_kwant:
                raise click.ClickException("Kwant installation failed (local and PyPI attempts).")
            if not pypi_ok:
                click.echo(click.style("  WARNING: kwant installation failed.", fg="yellow"))

        kwant_ready = dry_run or _module_importable(str(venv_python), "kwant")
        if not kwant_ready and not dry_run:
            click.echo(
                click.style(
                    "  kwant import check failed; forcing local ABI-compatible rebuild.",
                    fg="yellow",
                )
            )
            _pip_install(
                pip,
                [
                    "numpy<2",
                    "scipy<1.14",
                    "cython",
                    "meson-python",
                    "ninja",
                    "setuptools-scm",
                ],
                dry_run=dry_run,
                editable=False,
                raise_on_error=False,
            )
            _pip_install(
                pip,
                ["kwant<2"],
                dry_run=dry_run,
                editable=False,
                raise_on_error=False,
                extra_args=[
                    "--no-build-isolation",
                    "--no-binary=:all:",
                    "--no-cache-dir",
                    "--force-reinstall",
                ],
            )
            kwant_ready = _module_importable(str(venv_python), "kwant")
        if not kwant_ready and strict_kwant:
            raise click.ClickException(
                "Kwant is not importable in the configured venv after installation attempts."
            )

    # ── 6. Workspace setup ───────────────────────────────────────────────
    click.echo(click.style("\n[6/8] Workspace + env setup", bold=True))
    _setup_workspace(
        dry_run=dry_run,
        env_updates=env_updates,
        overwrite_env=overwrite_env,
        overwrite_slab_template=overwrite_slab_template,
        venv_path=venv_dir,
        venv_python=venv_python,
    )
    _ensure_remote_workdir_exists(
        dry_run=dry_run,
    )
    click.echo(
        click.style(
            "  slab workflow ready: template=wtec_slab_template.toml, "
            "commands: `wtec slab-gen` -> `wtec slab`",
            fg="green",
        )
    )

    # ── 7. Verification ──────────────────────────────────────────────────
    click.echo(click.style("\n[7/8] Verification", bold=True))
    _verify_install(
        python_executable=str(venv_python),
        dry_run=dry_run,
        check_kwant=(not no_kwant),
        check_berry=berry_enabled,
    )

    # ── 8. Cluster validation (optional) ────────────────────────────────
    click.echo(click.style("\n[8/8] Cluster validation", bold=True))
    if validate_cluster:
        if prepare_cluster_tools:
            click.echo(
                "  deferred: will run after remote toolchain prep "
                "(use --no-prepare-cluster-tools to validate immediately)"
            )
        else:
            _validate_cluster_setup(
                dry_run=dry_run,
                python_executable=str(venv_python),
            )
    else:
        click.echo("  skipped (use --validate-cluster)")

    # Remote cluster Python prep (kwant / wannierberri)
    click.echo(click.style("\n[cluster] Remote executable/toolchain prep", bold=True))
    if prepare_cluster_tools:
        _prepare_cluster_toolchain_setup(
            dry_run=dry_run,
        )
    else:
        click.echo("  skipped (--no-prepare-cluster-tools)")

    click.echo(click.style("\n[cluster] Remote pseudopotential prep", bold=True))
    if prepare_cluster_pseudos:
        _prepare_cluster_pseudopotential_setup(
            dry_run=dry_run,
        )
    else:
        click.echo("  skipped (--no-prepare-cluster-pseudos)")

    # Remote cluster Python prep (kwant / wannierberri)
    click.echo(click.style("\n[cluster] Remote Python feature prep", bold=True))
    if prepare_cluster_python:
        _prepare_cluster_python_setup(
            dry_run=dry_run,
            remote_python_executable=cluster_python_exe,
            ensure_kwant=(not no_kwant),
            ensure_tbmodels=True,
            ensure_sisl=True,
            ensure_berry=cluster_python_berry,
        )
    else:
        click.echo("  skipped (--no-prepare-cluster-python)")

    click.echo(click.style("\n[cluster] Final backend verification", bold=True))
    if validate_cluster and prepare_cluster_tools:
        _validate_cluster_setup(
            dry_run=dry_run,
            python_executable=str(venv_python),
        )
    else:
        if not validate_cluster:
            click.echo("  skipped (--no-validate-cluster)")
        else:
            click.echo("  skipped (--no-prepare-cluster-tools)")

    click.echo(click.style("\n✓ wtec init complete", fg="green", bold=True))
    click.echo(
        "\nNext steps:\n"
        "  wtec slab-gen\n"
        "  wtec defect\n"
        "  wtec run\n"
        "  wtec slab slab_outputs/<project>.generated.cif\n"
        f"  (venv auto-activation is enabled; manual activate optional: source {venv_dir}/bin/activate)"
    )


def _ensure_venv(
    *,
    venv_dir: Path,
    python_executable: str,
    dry_run: bool,
) -> Path:
    """Create/reuse venv and return venv python executable path."""
    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():
        click.echo(f"  Creating venv: {venv_dir} (python={python_executable})")
        if dry_run:
            return venv_python

        create_cmd = [python_executable, "-m", "venv", str(venv_dir)]
        result = subprocess.run(create_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise click.ClickException(
                "Failed to create venv:\n"
                f"{result.stderr.strip()}"
            )
        if not venv_python.exists():
            raise click.ClickException(f"Failed to create venv python at {venv_python}")
    else:
        click.echo(f"  Reusing existing venv: {venv_dir}")

    click.echo(f"  venv python: {venv_python}")
    if dry_run:
        return venv_python

    # Keep packaging tools current in the venv.
    cmd = [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(
            "Failed to bootstrap pip/setuptools/wheel in venv:\n"
            f"{result.stderr.strip()}"
        )
    return venv_python


def _module_importable(python_executable: str, module_name: str) -> bool:
    result = subprocess.run(
        [python_executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _python_version_tuple(python_executable: str) -> Optional[tuple[int, int, int]]:
    result = subprocess.run(
        [
            python_executable,
            "-c",
            "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        major, minor, patch = (int(x) for x in result.stdout.strip().split("."))
        return (major, minor, patch)
    except Exception:
        return None


def _read_requires_python_from_pyproject(package_dir: Path) -> str | None:
    pyproject = package_dir / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        import tomllib

        data = tomllib.loads(pyproject.read_text())
    except Exception:
        return None
    project = data.get("project", {})
    req = project.get("requires-python")
    return str(req) if req else None


def _parse_min_python(spec: str) -> Optional[tuple[int, int, int]]:
    # Minimal parser for common forms like ">=3.12" or ">=3.12,<3.14".
    if not spec.startswith(">="):
        return None
    raw = spec[2:].split(",")[0].strip()
    parts = raw.split(".")
    try:
        nums = [int(p) for p in parts if p != ""]
    except Exception:
        return None
    if not nums:
        return None
    while len(nums) < 3:
        nums.append(0)
    return (nums[0], nums[1], nums[2])


def _local_source_python_compatible(package_dir: Path, python_executable: str) -> tuple[bool, str | None]:
    if not package_dir.exists():
        return True, None
    requires_python = _read_requires_python_from_pyproject(package_dir)
    if not requires_python:
        return True, None
    min_py = _parse_min_python(requires_python)
    cur_py = _python_version_tuple(python_executable)
    if min_py is None or cur_py is None:
        return True, None
    if cur_py < min_py:
        cur_txt = f"{cur_py[0]}.{cur_py[1]}.{cur_py[2]}"
        min_txt = f"{min_py[0]}.{min_py[1]}.{min_py[2]}"
        return False, f"requires-python {requires_python} (venv python is {cur_txt}, needs >= {min_txt})"
    return True, None


def _pip_install(
    pip_base: list[str],
    packages: list[str],
    *,
    dry_run: bool,
    editable: bool = False,
    raise_on_error: bool = True,
    extra_args: list[str] | None = None,
    no_deps: bool = False,
) -> bool:
    success = True
    pip_args = list(extra_args or [])
    for pkg in packages:
        cmd_prefix = pip_base + pip_args
        if editable:
            cmd = cmd_prefix + ["-e", pkg]
            shown = " ".join(pip_args + ["-e", pkg]).strip()
        else:
            cmd = cmd_prefix + [pkg]
            shown = " ".join(pip_args + [pkg]).strip()
        if no_deps:
            cmd.append("--no-deps")
            shown = f"{shown} --no-deps".strip()
        click.echo(f"  pip install {shown}")
        if not dry_run:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                click.echo(click.style(f"  ERROR: {result.stderr.strip()}", fg="red"))
                success = False
                if raise_on_error:
                    raise SystemExit(1)
    return success


def _collect_env_updates(
    *,
    cluster_host: str | None,
    cluster_port: int | None,
    cluster_user: str | None,
    cluster_pass: str | None,
    cluster_key: str | None,
    mp_api_key: str | None,
    remote_workdir: str | None,
    mpi_cores: int | None,
    mpi_cores_by_queue: str | None,
    pbs_queue: str | None,
    pbs_queue_priority: str | None,
    qe_pseudo_dir: str | None,
    siesta_pseudo_dir: str | None,
    vasp_pseudo_dir: str | None,
    abacus_pseudo_dir: str | None,
    abacus_orbital_dir: str | None,
    qe_pseudo_source_dir: str | None,
    omp_threads: int | None,
    cluster_modules: str | None,
    cluster_bin_dirs: str | None,
    qe_source_dir: str | None,
    siesta_source_dir: str | None,
    abacus_source_dir: str | None,
    wannier90_source_dir: str | None,
    cluster_build_jobs: int | None,
) -> dict[str, str]:
    updates: dict[str, str] = {}

    def set_if(key: str, value) -> None:
        if value is None:
            return
        updates[key] = str(value)

    set_if("TOPOSLAB_CLUSTER_HOST", cluster_host)
    set_if("TOPOSLAB_CLUSTER_PORT", cluster_port)
    set_if("TOPOSLAB_CLUSTER_USER", cluster_user)
    set_if("TOPOSLAB_CLUSTER_PASS", cluster_pass)
    set_if("TOPOSLAB_CLUSTER_KEY", cluster_key)
    set_if("MP_API_KEY", mp_api_key)
    set_if("TOPOSLAB_REMOTE_WORKDIR", remote_workdir)
    set_if("TOPOSLAB_MPI_CORES", mpi_cores)
    set_if("TOPOSLAB_MPI_CORES_BY_QUEUE", mpi_cores_by_queue)
    set_if("TOPOSLAB_PBS_QUEUE", pbs_queue)
    set_if("TOPOSLAB_PBS_QUEUE_PRIORITY", pbs_queue_priority)
    set_if("TOPOSLAB_QE_PSEUDO_DIR", qe_pseudo_dir)
    set_if("TOPOSLAB_SIESTA_PSEUDO_DIR", siesta_pseudo_dir)
    set_if("TOPOSLAB_VASP_PSEUDO_DIR", vasp_pseudo_dir)
    set_if("TOPOSLAB_ABACUS_PSEUDO_DIR", abacus_pseudo_dir)
    set_if("TOPOSLAB_ABACUS_ORBITAL_DIR", abacus_orbital_dir)
    set_if("TOPOSLAB_QE_PSEUDO_SOURCE_DIR", qe_pseudo_source_dir)
    set_if("TOPOSLAB_OMP_THREADS", omp_threads)
    set_if("TOPOSLAB_CLUSTER_MODULES", cluster_modules)
    set_if("TOPOSLAB_CLUSTER_BIN_DIRS", cluster_bin_dirs)
    set_if("TOPOSLAB_QE_SOURCE_DIR", qe_source_dir)
    set_if("TOPOSLAB_SIESTA_SOURCE_DIR", siesta_source_dir)
    set_if("TOPOSLAB_ABACUS_SOURCE_DIR", abacus_source_dir)
    set_if("TOPOSLAB_WANNIER90_SOURCE_DIR", wannier90_source_dir)
    set_if("TOPOSLAB_CLUSTER_BUILD_JOBS", cluster_build_jobs)
    return updates


def _merge_init_interactive_env_updates(
    env_updates: dict[str, str],
) -> dict[str, str]:
    """Prompt for missing cluster/env essentials during `wtec init`."""
    if not sys.stdin.isatty():
        return env_updates

    out = dict(env_updates)

    def current(key: str) -> str | None:
        if key in out:
            val = str(out[key]).strip()
            if val:
                return val
        raw = os.environ.get(key, "").strip()
        return raw or None

    click.echo(click.style("\n[init] Interactive cluster setup", bold=True))
    if not current("TOPOSLAB_CLUSTER_HOST"):
        out["TOPOSLAB_CLUSTER_HOST"] = click.prompt("Cluster host", type=str).strip()
    if not current("TOPOSLAB_CLUSTER_PORT"):
        out["TOPOSLAB_CLUSTER_PORT"] = str(
            click.prompt("Cluster port", type=int, default=22, show_default=True)
        )
    if not current("TOPOSLAB_CLUSTER_USER"):
        out["TOPOSLAB_CLUSTER_USER"] = click.prompt("Cluster user", type=str).strip()

    if not current("TOPOSLAB_CLUSTER_KEY") and not current("TOPOSLAB_CLUSTER_PASS"):
        use_key = click.confirm("Use SSH key authentication?", default=False)
        if use_key:
            out["TOPOSLAB_CLUSTER_KEY"] = click.prompt(
                "SSH key path",
                type=str,
                default="~/.ssh/id_rsa",
                show_default=True,
            ).strip()
        else:
            out["TOPOSLAB_CLUSTER_PASS"] = click.prompt(
                "Cluster password",
                hide_input=True,
                type=str,
            )

    if not current("MP_API_KEY") and not current("PMG_MAPI_KEY"):
        set_mp = click.confirm(
            "Set Materials Project API key now (needed for default mp-based slab template)?",
            default=True,
        )
        if set_mp:
            out["MP_API_KEY"] = click.prompt("MP API key", hide_input=True, type=str).strip()

    if not current("TOPOSLAB_REMOTE_WORKDIR"):
        out["TOPOSLAB_REMOTE_WORKDIR"] = click.prompt(
            "Remote workdir",
            type=str,
            default=str((Path.cwd() / "remote_runs").resolve()),
            show_default=True,
        ).strip()

    if not current("TOPOSLAB_MPI_CORES"):
        out["TOPOSLAB_MPI_CORES"] = str(
            click.prompt("MPI cores per node", type=int, default=64, show_default=True)
        )
    if not current("TOPOSLAB_DEFAULT_DFT_ENGINE"):
        out["TOPOSLAB_DEFAULT_DFT_ENGINE"] = click.prompt(
            "Default DFT engine",
            type=click.Choice(["qe", "siesta", "vasp", "abacus"], case_sensitive=False),
            default="qe",
            show_default=True,
        ).strip().lower()

    pseudo = current("TOPOSLAB_QE_PSEUDO_DIR")
    if not pseudo or pseudo == "/pseudo":
        user = current("TOPOSLAB_CLUSTER_USER") or "msj"
        out["TOPOSLAB_QE_PSEUDO_DIR"] = click.prompt(
            "QE pseudo dir",
            type=str,
            default=_guess_default_qe_pseudo_dir(user),
            show_default=True,
        ).strip()

    siesta_pseudo = current("TOPOSLAB_SIESTA_PSEUDO_DIR")
    if not siesta_pseudo or siesta_pseudo == "/pseudo":
        user = current("TOPOSLAB_CLUSTER_USER") or "msj"
        out["TOPOSLAB_SIESTA_PSEUDO_DIR"] = click.prompt(
            "SIESTA pseudo dir",
            type=str,
            default=f"/home/{user}/siesta/pseudo",
            show_default=True,
        ).strip()

    vasp_pseudo = current("TOPOSLAB_VASP_PSEUDO_DIR")
    if not vasp_pseudo or vasp_pseudo == "/pseudo":
        user = current("TOPOSLAB_CLUSTER_USER") or "msj"
        out["TOPOSLAB_VASP_PSEUDO_DIR"] = click.prompt(
            "VASP pseudo dir",
            type=str,
            default=f"/home/{user}/vasp/potpaw_PBE",
            show_default=True,
        ).strip()

    abacus_pseudo = current("TOPOSLAB_ABACUS_PSEUDO_DIR")
    if not abacus_pseudo or abacus_pseudo == "/pseudo":
        user = current("TOPOSLAB_CLUSTER_USER") or "msj"
        out["TOPOSLAB_ABACUS_PSEUDO_DIR"] = click.prompt(
            "ABACUS pseudo dir",
            type=str,
            default=f"/home/{user}/abacus/pseudo",
            show_default=True,
        ).strip()

    abacus_orb = current("TOPOSLAB_ABACUS_ORBITAL_DIR")
    if not abacus_orb or abacus_orb == "/orbital":
        user = current("TOPOSLAB_CLUSTER_USER") or "msj"
        out["TOPOSLAB_ABACUS_ORBITAL_DIR"] = click.prompt(
            "ABACUS orbital dir",
            type=str,
            default=f"/home/{user}/abacus/orbital",
            show_default=True,
        ).strip()
    if (current("TOPOSLAB_DEFAULT_DFT_ENGINE") or "qe").strip().lower() == "siesta":
        if not current("TOPOSLAB_SIESTA_SOURCE_DIR"):
            user = current("TOPOSLAB_CLUSTER_USER") or "msj"
            set_src = click.confirm(
                "Set SIESTA source dir now for auto-build during init?",
                default=False,
            )
            if set_src:
                out["TOPOSLAB_SIESTA_SOURCE_DIR"] = click.prompt(
                    "SIESTA source dir",
                    type=str,
                    default=f"/home/{user}/src/siesta",
                    show_default=True,
                ).strip()

    return out


def _apply_env_updates_to_process(updates: dict[str, str]) -> None:
    for key, value in updates.items():
        os.environ[str(key)] = str(value)


def _guess_default_qe_pseudo_dir(user: str) -> str:
    """Pick a sensible default pseudo dir, preferring SOC-capable UPFs."""
    user = (user or "msj").strip() or "msj"
    candidates = [
        f"/home/{user}/qe/pseudo/pslibrary.1.0.0",
        f"/home/{user}/src/QE_pseudo/pslibrary.1.0.0",
        f"/home/{user}/src/QE_pseudo/pslibrary/pbe/PSEUDOPOTENTIALS",
    ]

    def has_rel_upf(path: Path) -> bool:
        try:
            for entry in path.iterdir():
                if not entry.is_file():
                    continue
                name = entry.name.lower()
                if name.endswith(".upf") and ".rel-" in name:
                    return True
        except Exception:
            return False
        return False

    for raw in candidates:
        p = Path(raw).expanduser()
        if p.is_dir() and has_rel_upf(p):
            return str(p)
    for raw in candidates:
        p = Path(raw).expanduser()
        if p.is_dir():
            return str(p)
    return candidates[0]


def _extract_env_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    return keys


def _ensure_env_keys(base_text: str, template_text: str) -> str:
    present = _extract_env_keys(base_text)
    missing_lines: list[str] = []
    for line in template_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key not in present:
            missing_lines.append(line)
    if not missing_lines:
        return base_text

    out = base_text.rstrip("\n")
    out += "\n\n# Added by wtec init (missing keys)\n"
    out += "\n".join(missing_lines) + "\n"
    return out


def _apply_env_updates(text: str, updates: dict[str, str]) -> str:
    if not updates:
        return text

    lines = text.splitlines()
    seen: set[str] = set()
    out_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out_lines.append(line)
            continue

        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            out_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out_lines.append(line)

    for key, value in updates.items():
        if key not in seen:
            out_lines.append(f"{key}={value}")

    return "\n".join(out_lines) + "\n"


def _setup_workspace(
    *,
    dry_run: bool,
    env_updates: dict[str, str],
    overwrite_env: bool,
    overwrite_slab_template: bool,
    venv_path: Path,
    venv_python: Path,
) -> None:
    env_example = Path(".env.example")
    env_file = Path(".env")

    # Copy .env.example to cwd if it doesn't exist
    src_example = Path(__file__).resolve().parent.parent / ".env.example"
    if src_example.exists() and not env_example.exists():
        click.echo(f"  Creating {env_example}")
        if not dry_run:
            env_example.write_text(src_example.read_text())

    if not src_example.exists():
        raise FileNotFoundError(f"Missing template: {src_example}")

    template_text = src_example.read_text()
    if env_file.exists() and not overwrite_env:
        base_text = _ensure_env_keys(env_file.read_text(), template_text)
        rendered = _apply_env_updates(base_text, env_updates)
        action = "Updating .env"
    else:
        rendered = _apply_env_updates(template_text, env_updates)
        action = "Creating .env" if not env_file.exists() else "Rewriting .env from template"

    click.echo(f"  {action}")
    if not dry_run:
        env_file.write_text(rendered)

    # Create local workspace dir
    workspace = Path.home() / ".wtec"
    click.echo(f"  Creating workspace: {workspace}")
    if not dry_run:
        workspace.mkdir(exist_ok=True)
        (workspace / "runs").mkdir(exist_ok=True)
        (workspace / "checkpoints").mkdir(exist_ok=True)
        venv_python_abs = (venv_path / "bin" / "python").expanduser().absolute()
        init_state = {
            "initialized_at_epoch": int(time.time()),
            "python_executable": str(venv_python_abs),
            "venv_path": str(venv_path.resolve()),
            # Preserve the venv interpreter path (do not resolve symlink to base python).
            "venv_python": str(venv_python_abs),
            "cwd": str(Path.cwd()),
            "env_file": str(env_file.resolve()),
            "env_updates_applied": sorted(env_updates.keys()),
        }
        (workspace / "init_state.json").write_text(json.dumps(init_state, indent=2))

    _write_slab_template(
        cwd=Path.cwd(),
        dry_run=dry_run,
        overwrite=overwrite_slab_template,
    )
    _write_project_template(
        cwd=Path.cwd(),
        dry_run=dry_run,
        overwrite=overwrite_slab_template,
    )


def _ensure_remote_workdir_exists(*, dry_run: bool) -> None:
    """Create TOPOSLAB_REMOTE_WORKDIR on cluster if cluster config is available."""
    if dry_run:
        click.echo("  (skip remote workdir creation in dry-run mode)")
        return

    from wtec.config.cluster import ClusterConfig
    from wtec.cluster.ssh import open_ssh

    try:
        cfg = ClusterConfig.from_env()
    except Exception as exc:
        click.echo(
            click.style(
                f"  WARNING: cluster config incomplete; skip remote workdir creation: {exc}",
                fg="yellow",
            )
        )
        return

    click.echo(
        f"  Ensuring remote workdir exists on cluster: {cfg.remote_workdir}"
    )
    with open_ssh(cfg) as ssh:
        ssh.mkdir_p(cfg.remote_workdir)
    click.echo(click.style(f"  ✓ remote workdir ready: {cfg.remote_workdir}", fg="green"))


def _write_slab_template(*, cwd: Path, dry_run: bool, overwrite: bool) -> None:
    target = cwd / "wtec_slab_template.toml"
    if target.exists() and not overwrite:
        click.echo(f"  Reusing slab template: {target}")
        return
    action = "Rewriting" if target.exists() else "Creating"
    click.echo(f"  {action} slab template: {target}")
    if dry_run:
        return
    target.write_text(_default_slab_template_text())


def _write_project_template(*, cwd: Path, dry_run: bool, overwrite: bool) -> None:
    target = cwd / "wtec_project.toml"
    if target.exists() and not overwrite:
        click.echo(f"  Reusing project template: {target}")
        if not dry_run and _migrate_project_template(target):
            click.echo("  Updated legacy template defaults in existing project template")
        return
    action = "Rewriting" if target.exists() else "Creating"
    click.echo(f"  {action} project template: {target}")
    if dry_run:
        return
    target.write_text(_default_project_template_text())


def _migrate_project_template(path: Path) -> bool:
    """Patch known legacy defaults in an existing master template."""
    text = path.read_text()
    updated = text.replace('mp_id = "mp-1067687"', 'mp_id = "mp-1067587"')
    updated = updated.replace("max_strain_percent = 35.0", "max_strain_percent = 6.0")
    updated = updated.replace("max_strain_percent = 5.0", "max_strain_percent = 6.0")
    updated = updated.replace("max_search_supercell = 4", "max_search_supercell = 8")
    if "failure_policy =" not in updated:
        updated = updated.replace(
            "strict_qsub = true",
            "strict_qsub = true\nfailure_policy = \"strict\"",
        )
    if "node_method =" not in updated:
        updated = updated.replace(
            "arc_engine = \"wannierberri\"",
            "arc_engine = \"siesta_slab_ldos\"\nnode_method = \"wannierberri_flux\"",
        )
    if "hr_scope =" not in updated:
        updated = updated.replace(
            "node_method = \"wannierberri_flux\"",
            "node_method = \"wannierberri_flux\"\nhr_scope = \"per_variant\"",
        )
    updated = updated.replace('failure_policy = "soft_nan"', 'failure_policy = "strict"')
    updated = updated.replace('hr_scope = "shared"', 'hr_scope = "per_variant"')
    updated = updated.replace('arc_engine = "wannierberri"', 'arc_engine = "siesta_slab_ldos"')
    updated = updated.replace('node_method = "proxy"', 'node_method = "wannierberri_flux"')
    updated = updated.replace('policy = "dual_track_compare"', 'policy = "single_track"')
    if "[topology]\n" in updated and "arc_allow_proxy_fallback =" not in updated.split("[topology]\n", 1)[1].split("\n\n", 1)[0]:
        updated = updated.replace(
            'arc_engine = "siesta_slab_ldos"',
            'arc_engine = "siesta_slab_ldos"\narc_allow_proxy_fallback = false',
            1,
        )
    updated = updated.replace("transport_n_layers_x = 1", "transport_n_layers_x = 4")
    updated = updated.replace("scf = [4, 4, 1]", "scf = [8, 8, 8]")
    updated = updated.replace("nscf = [8, 8, 1]", "nscf = [12, 12, 12]")
    if "[dft]\n" in updated and "[dft]\nengine =" not in updated:
        updated = updated.replace("[dft]\n", "[dft]\nengine = \"qe\"\n", 1)
    if "[dft]\n" in updated and "mode =" not in updated.split("[dft]\n", 1)[1].split("\n\n", 1)[0]:
        updated = updated.replace(
            "[dft]\n",
            "[dft]\nmode = \"dual_family\"\n",
            1,
        )
    if "[dft.tracks.pes_reference]" not in updated:
        updated = updated.replace(
            "[dft.siesta]",
            "[dft.tracks.pes_reference]\n"
            "family = \"pes\"\n"
            "engine = \"qe\"\n"
            "structure_file = \"references/TaP_primitive.cif\"\n"
            "disable_symmetry = true\n"
            "reuse_policy = \"strict_hash\"\n\n"
            "[dft.tracks.lcao_upscaled]\n"
            "family = \"lcao\"\n"
            "engine = \"siesta\"\n"
            "source = \"variants\"\n\n"
            "[dft.siesta]",
            1,
        )
    if "[dft.anchor_transfer]" not in updated and "[dft.siesta]" in updated:
        updated = updated.replace(
            "[dft.siesta]",
            "[dft.anchor_transfer]\n"
            "enabled = true\n"
            "mode = \"delta_h\"\n"
            "basis_policy = \"strict_same_basis\"\n"
            "scope = \"onsite_plus_first_shell\"\n"
            "fit_window_ev = 1.5\n"
            "fit_kmesh = [8, 8, 8]\n"
            "alpha_grid_min = -0.5\n"
            "alpha_grid_max = 1.5\n"
            "alpha_grid_points = 81\n"
            "max_retries = 5\n"
            "retry_kmesh_step = 2\n"
            "retry_window_step_ev = 0.5\n"
            "reuse_existing = true\n\n"
            "[dft.siesta]",
            1,
        )
    if "[dft.reference]\n" in updated and "reuse_policy =" not in updated.split("[dft.reference]\n", 1)[1].split("\n\n", 1)[0]:
        updated = updated.replace(
            "disable_symmetry = true",
            "disable_symmetry = true\nreuse_policy = \"strict_hash\"",
            1,
        )
    if "[dft.siesta]" not in updated:
        updated = updated.replace(
            "[dft.kmesh]",
            "[dft.siesta]\n"
            "wannier_interface = \"sisl\"\n"
            "pseudo_dir = \"\"\n"
            "basis_profile = \"\"\n\n"
            "[dft.kmesh]",
            1,
        )
    if "[dft.siesta]\n" in updated:
        sec_start = updated.find("[dft.siesta]\n")
        if sec_start >= 0:
            block_start = sec_start + len("[dft.siesta]\n")
            block_end = updated.find("\n\n", block_start)
            if block_end < 0:
                block_end = len(updated)
            block = updated[block_start:block_end]
            additions: list[str] = []
            if "mpi_np_scf =" not in block:
                additions.append("mpi_np_scf = 0")
            if "mpi_np_nscf =" not in block:
                additions.append("mpi_np_nscf = 0")
            if "mpi_np_wannier =" not in block:
                additions.append("mpi_np_wannier = 0")
            if "omp_threads_scf =" not in block:
                additions.append("omp_threads_scf = 0")
            if "omp_threads_nscf =" not in block:
                additions.append("omp_threads_nscf = 0")
            if "omp_threads_wannier =" not in block:
                additions.append("omp_threads_wannier = 0")
            if "factorization_defaults =" not in block:
                additions.append("factorization_defaults = {}")
            if "dm_mixing_weight =" not in block:
                additions.append("dm_mixing_weight = 0.10")
            if "dm_number_pulay =" not in block:
                additions.append("dm_number_pulay = 8")
            if "electronic_temperature_k =" not in block:
                additions.append("electronic_temperature_k = 300.0")
            if "max_scf_iterations =" not in block:
                additions.append("max_scf_iterations = 200")
            if additions:
                block_text = block
                if block_text and not block_text.endswith("\n"):
                    block_text += "\n"
                block_text += "\n".join(additions)
                updated = updated[:block_start] + block_text + updated[block_end:]
    if "[dft.dispersion]" not in updated:
        updated = updated.replace(
            "[dft.kmesh]",
            "[dft.dispersion]\n"
            "enabled = true\n"
            "method = \"d3\"\n"
            "qe_vdw_corr = \"grimme-d3\"\n"
            "qe_dftd3_version = 4\n"
            "qe_dftd3_threebody = true\n"
            "siesta_dftd3_use_xc_defaults = true\n\n"
            "[dft.kmesh]",
            1,
        )
    if "[transport]\n" in updated and "policy =" not in updated.split("[transport]\n", 1)[1].split("\n\n", 1)[0]:
        updated = updated.replace("[transport]\n", "[transport]\npolicy = \"single_track\"\n", 1)
    if "[transport.autotune]" not in updated:
        updated = updated.replace(
            "[topology]",
            "[transport.autotune]\n"
            "enabled = true\n"
            "scope = \"per_queue\"\n"
            "cache_file = \"~/.wtec/transport_autotune.json\"\n"
            "benchmark_walltime = \"00:08:00\"\n"
            "benchmark_n_ensemble = 1\n"
            "[[transport.autotune.profiles]]\n"
            "mpi_np = 1\n"
            "threads = \"all\"\n"
            "[[transport.autotune.profiles]]\n"
            "mpi_np = 2\n"
            "threads = \"half\"\n"
            "[[transport.autotune.profiles]]\n"
            "mpi_np = 4\n"
            "threads = \"quarter\"\n\n"
            "[topology]",
            1,
        )
    if "[topology]\n" in updated and "variant_dft_engine =" not in updated.split("[topology]\n", 1)[1].split("\n\n", 1)[0]:
        updated = updated.replace("hr_scope = \"per_variant\"", "hr_scope = \"per_variant\"\nvariant_dft_engine = \"siesta\"")
    if "[topology.hr_grid]" not in updated:
        updated = updated.rstrip() + (
            "\n\n[topology.hr_grid]\n"
            "thickness_mapping = \"middle_layer_scale\"\n"
            "middle_layer_role = \"active\"\n"
            "reference_thickness_uc = 3\n"
            "reuse_successful_points = true\n"
        )
    if "[topology.tiering]" not in updated:
        updated = updated.replace(
            "[topology.score]",
            "[topology.tiering]\n"
            "mode = \"single\"\n"
            "refine_top_k_per_thickness = 2\n"
            "always_include_pristine = true\n"
            "selection_metric = \"S_total\"\n\n"
            "[topology.tiering.screen]\n"
            "arc_engine = \"siesta_slab_ldos\"\n"
            "node_method = \"wannierberri_flux\"\n"
            "coarse_kmesh = [10, 10, 10]\n"
            "refine_kmesh = [3, 3, 3]\n"
            "newton_max_iter = 20\n"
            "max_candidates = 24\n\n"
            "[topology.score]",
            1,
        )
    if "[run]\n" in updated and "profile =" not in updated.split("[run]\n", 1)[1].split("\n\n", 1)[0]:
        updated = updated.replace("n_nodes = 1", "n_nodes = 1\nprofile = \"strict\"", 1)
    if "[dft]\n" in updated and "reuse_mode =" not in updated.split("[dft]\n", 1)[1].split("\n\n", 1)[0]:
        updated = updated.replace("engine = \"qe\"", "engine = \"qe\"\nreuse_mode = \"none\"", 1)
    if "[topology.score]" not in updated:
        updated = updated.rstrip() + (
            "\n\n[topology.score]\n"
            "w_topo = 0.70\n"
            "w_transport = 0.30\n"
        )
    if "[topology.transport_probe]" not in updated:
        updated = updated.rstrip() + (
            "\n\n[topology.transport_probe]\n"
            "enabled = true\n"
            "n_ensemble = 1\n"
            "disorder_strength = 0.0\n"
            "energy_shift_ev = 0.0\n"
            "thickness_axis = \"z\"\n"
        )
    if updated == text:
        return False
    path.write_text(updated)
    return True


def _default_slab_template_text() -> str:
    return """# wtec slab generation template (TOML)
# Generated by `wtec init`.
# Edit this file and run:
#   wtec slab-gen wtec_slab_template.toml
#
# Source modes for each layer:
#   source = "cif"  -> local CIF path
#   source = "mp"   -> Materials Project id (requires API key)
#
# Layer order must be bottom -> active -> top (trilayer).
# You can expose more interfaces by adding additional layers.

[project]
name = "sio2_tap_sio2"
seed = 20260304
output_dir = "slab_outputs"
material = "TaP"
# Optional if using source="mp":
# mp_api_key = "your_materials_project_api_key"
# Optional env var name if key is not embedded:
# mp_api_key_env = "MP_API_KEY"

[matching]
# Maximum in-plane strain (%) allowed during lattice matching.
max_strain_percent = 6.0
# Maximum matched area ratio between adjacent layers.
max_area_ratio = 4.0
# Maximum search range for in-plane supercell factors.
max_search_supercell = 8

[stack]
align_axis = "z"
# Extra vacuum region (Angstrom) applied after full stack assembly.
vacuum_angstrom = 18.0
# Gap between neighboring layers before optional interface defects.
interface_gap_angstrom = 2.2

[export]
# Relative to output_dir unless absolute path.
cif_path = "sio2_tap_sio2.generated.cif"
metadata_json_path = "sio2_tap_sio2.generated.meta.json"

# --------------------------- Layers -----------------------------------------
# role can be: "substrate", "active", "cap", or custom text.
# miller controls crystallographic plane slicing.
# thickness_angstrom controls slab thickness from each source crystal.
# termination_index chooses one of generated slab terminations.

[[layers]]
label = "bottom_sio2"
role = "substrate"
source = "cif"
cif_path = "/absolute/or/relative/path/to/sio2.cif"
# source = "mp"
# mp_id = "mp-7000"
miller = [0, 0, 1]
thickness_angstrom = 8.0
termination_index = 0

[[layers]]
label = "middle_tap"
role = "active"
source = "cif"
cif_path = "/absolute/or/relative/path/to/tap.cif"
# source = "mp"
# mp_id = "mp-567501"
miller = [0, 0, 1]
thickness_angstrom = 10.0
termination_index = 0

[[layers]]
label = "top_sio2"
role = "cap"
source = "cif"
cif_path = "/absolute/or/relative/path/to/sio2.cif"
# source = "mp"
# mp_id = "mp-7000"
miller = [0, 0, 1]
thickness_angstrom = 8.0
termination_index = 0

# ------------------------ Interface Engineering -----------------------------
# `between` names two adjacent layer labels.
# vacancy_mode:
#   "none"
#   "random_interface" -> random picks in interface window
# vacancy_window_angstrom: half-window around interface plane for defect picks.
# vacancy_counts_by_element: element-specific number of vacancies in this interface.

[[interfaces]]
between = ["bottom_sio2", "middle_tap"]
vacancy_mode = "random_interface"
vacancy_window_angstrom = 2.0
vacancy_seed = 101
vacancy_counts_by_element = { O = 1, Si = 0, Ta = 0, P = 0 }

[[interfaces.substitutions]]
from = "O"
to = "S"
count = 0

[[interfaces]]
between = ["middle_tap", "top_sio2"]
vacancy_mode = "random_interface"
vacancy_window_angstrom = 2.0
vacancy_seed = 202
vacancy_counts_by_element = { O = 1, Si = 0, Ta = 0, P = 0 }

[[interfaces.substitutions]]
from = "O"
to = "S"
count = 0

# ------------------------ Optional Notes ------------------------------------
# 1) For source="mp", set either:
#      project.mp_api_key in this TOML, or
#      export MP_API_KEY=your_key (or PMG_MAPI_KEY)
# 2) To use a generated slab in workflow config:
#      "structure_file": "slab_outputs/sio2_tap_sio2.generated.cif"
# 3) Inspect generated shape and defect summary:
#      wtec slab slab_outputs/sio2_tap_sio2.generated.cif
"""


def _default_project_template_text() -> str:
    return """# wtec master project template (TOML)
# Generated by `wtec init`.
#
# No-flag workflow:
#   wtec slab-gen
#   wtec defect
#   wtec run
#
# This one file drives slab generation, defect variant generation, and run config.

[project]
# Used in slab metadata and default run naming.
name = "sio2_tap_sio2"
seed = 20260305
# Used by slab-gen exports below.
output_dir = "slab_outputs"
# Optional MP key (if omitted, MP_API_KEY/PMG_MAPI_KEY env vars are used).
# mp_api_key = ""
mp_api_key_env = "MP_API_KEY"
# Default material for run config.
material = "TaP"

[cluster]
# If empty, `wtec run` falls back to TOPOSLAB_REMOTE_WORKDIR from .env.
remote_workdir = ""
cluster_python_exe = "python3"

[matching]
# Strict physics baseline for SiO2/TaP coincidence matching.
max_strain_percent = 6.0
max_area_ratio = 4.0
# Supercell search range for slab in-plane matching.
max_search_supercell = 8

[stack]
align_axis = "z"
vacuum_angstrom = 18.0
interface_gap_angstrom = 2.2

[export]
cif_path = "sio2_tap_sio2.generated.cif"
metadata_json_path = "sio2_tap_sio2.generated.meta.json"

[[layers]]
label = "bottom_sio2"
role = "substrate"
source = "mp"
mp_id = "mp-7000"
# source = "cif"
# cif_path = "/abs/path/to/sio2.cif"
miller = [0, 0, 1]
thickness_angstrom = 8.0
termination_index = 0

[[layers]]
label = "middle_tap"
role = "active"
source = "mp"
mp_id = "mp-1067587"
# source = "cif"
# cif_path = "/abs/path/to/tap.cif"
miller = [0, 0, 1]
thickness_angstrom = 10.0
termination_index = 0

[[layers]]
label = "top_sio2"
role = "cap"
source = "mp"
mp_id = "mp-7000"
# source = "cif"
# cif_path = "/abs/path/to/sio2.cif"
miller = [0, 0, 1]
thickness_angstrom = 8.0
termination_index = 0

[[interfaces]]
between = ["bottom_sio2", "middle_tap"]
vacancy_mode = "random_interface"
vacancy_window_angstrom = 2.0
vacancy_seed = 101
vacancy_counts_by_element = { O = 1, Si = 0, Ta = 0, P = 0 }
[[interfaces.substitutions]]
from = "O"
to = "S"
count = 0

[[interfaces]]
between = ["middle_tap", "top_sio2"]
vacancy_mode = "random_interface"
vacancy_window_angstrom = 2.0
vacancy_seed = 202
vacancy_counts_by_element = { O = 1, Si = 0, Ta = 0, P = 0 }
[[interfaces.substitutions]]
from = "O"
to = "S"
count = 0

[defect]
# `wtec defect` default output (for topology variant discovery)
output_dir = "slab_variants"
generate_pristine = true
generate_defect = true
pristine_suffix = "pristine"
defect_suffix = "defect"
# If template has zero configured defects, inject this many O vacancies on first interface.
min_vacancies_total = 1

[run]
# Checkpoint/run label. Keep short.
name = "sio2_tap_sio2_balanced"
# If empty, uses <repo>/runs/<name>.
run_dir = ""
# If empty, uses defect pristine CIF if present, else slab export CIF.
structure_file = ""
n_nodes = 1
# strict = production physics guardrails, smoke = lightweight operational checks.
profile = "strict" # strict|smoke

[dft]
mode = "dual_family" # dual_family|legacy_single|hybrid_qe_ref_siesta_variants
# none = do not bypass fresh DFT with external hr_dat_path in strict runs.
reuse_mode = "none" # none|pristine-only|all
# Optional path to reuse an existing Wannier90 _hr.dat and skip DFT stages.
# hr_dat_path = "/abs/path/to/TaP_hr.dat"
qe_noncolin = true
qe_lspinorb = true

[dft.tracks.pes_reference]
# Small-structure PES reference (required in dual_family).
family = "pes"
engine = "qe" # qe|vasp
# REQUIRED for dual_family: explicit small primitive/conventional reference CIF.
structure_file = "references/TaP_primitive.cif"
disable_symmetry = true
reuse_policy = "strict_hash" # strict_hash|timestamp_only

[dft.tracks.lcao_upscaled]
# Upscaled/variant DFT track for slab and defects.
family = "lcao"
engine = "siesta" # siesta|abacus
source = "variants" # variants|run_structure

[dft.anchor_transfer]
# PES(reference) -> LCAO(anchor) transfer used to correct variant LCAO HRs.
enabled = true
mode = "delta_h" # delta_h
basis_policy = "strict_same_basis" # strict_same_basis
scope = "onsite_plus_first_shell" # onsite_plus_first_shell
fit_window_ev = 1.5
fit_kmesh = [8, 8, 8]
alpha_grid_min = -0.5
alpha_grid_max = 1.5
alpha_grid_points = 81
max_retries = 5
retry_kmesh_step = 2
retry_window_step_ev = 0.5
reuse_existing = true

[dft.siesta]
wannier_interface = "sisl" # sisl|builtin
# Optional override; falls back to TOPOSLAB_SIESTA_PSEUDO_DIR.
pseudo_dir = ""
# Optional basis profile in wtec.siesta.presets (auto from material if empty).
basis_profile = ""
# Stage-level MPI/OMP settings (0 = auto by queue/core allocation).
mpi_np_scf = 0
mpi_np_nscf = 0
mpi_np_wannier = 0
omp_threads_scf = 0
omp_threads_nscf = 0
omp_threads_wannier = 0
# Queue/core scoped defaults (key format: "<queue>_<cores_per_node>").
# Example: factorization_defaults = { "g3_32" = { mpi_np_scf = 16, omp_threads_scf = 2, mpi_np_nscf = 32, omp_threads_nscf = 1 } }
factorization_defaults = {}
# SCF convergence tuning for LCAO acceleration.
dm_mixing_weight = 0.10
dm_number_pulay = 8
electronic_temperature_k = 300.0
max_scf_iterations = 200

[dft.vasp]
# Optional override; falls back to TOPOSLAB_VASP_PSEUDO_DIR.
pseudo_dir = ""
executable = "vasp_std"
encut_ev = 520.0
ediff = 1.0e-6
ismear = 0
sigma = 0.05
disable_symmetry = true

[dft.abacus]
# Optional overrides; fall back to TOPOSLAB_ABACUS_PSEUDO_DIR / TOPOSLAB_ABACUS_ORBITAL_DIR.
pseudo_dir = ""
orbital_dir = ""
executable = "abacus"
basis_type = "lcao"
ks_solver = "genelpa"

[dft.dispersion]
enabled = true
method = "d3" # d3|none
qe_vdw_corr = "grimme-d3"
qe_dftd3_version = 4
qe_dftd3_threebody = true
siesta_dftd3_use_xc_defaults = true

[dft.kmesh]
# Balanced defaults; increase for production.
scf = [8, 8, 8]
nscf = [12, 12, 12]

[transport]
policy = "single_track" # single_track|dual_track_compare
thicknesses = [3, 5, 7, 9, 11]
disorder_strengths = [0.0, 0.2]
n_ensemble = 30
n_jobs = 1
mfp_n_layers_z = 5
mfp_lengths = [3, 5, 7, 9, 11, 13, 15]
transport_axis = "x"
thickness_axis = "z"
transport_n_layers_x = 4
transport_n_layers_y = 4
lead_onsite_eV = 0.0
base_seed = 42
# Optional Drude refinement inputs for MFP (leave unset to use fit-derived MFP only).
# carrier_density_m3 = 1.0e26
# fermi_velocity_m_per_s = 3.0e5
backend = "qsub"
strict_qsub = true
walltime = "00:30:00"
# 0 = auto (uses all allocated cores for this queue/node profile).
mpi_np = 0
threads = 0
# Optional override for transport worker python on cluster/login node.
# If empty, falls back to [cluster].cluster_python_exe.
cluster_python_exe = ""

[transport.autotune]
# Auto-pick transport (mpi_np, threads) profile for this queue/node class and cache result.
enabled = true
scope = "per_queue"
cache_file = "~/.wtec/transport_autotune.json"
benchmark_walltime = "00:08:00"
benchmark_n_ensemble = 1
[[transport.autotune.profiles]]
mpi_np = 1
threads = "all"
[[transport.autotune.profiles]]
mpi_np = 2
threads = "half"
[[transport.autotune.profiles]]
mpi_np = 4
threads = "quarter"

[topology]
enabled = true
backend = "qsub"
execution_mode = "per_point_qsub"
strict_qsub = true
failure_policy = "strict" # strict|rescale
max_concurrent_point_jobs = 1
max_concurrent_variant_dft_jobs = 1
# No-fork policy: always qsub/mpirun.
variant_discovery_glob = "slab_variants/*.generated.meta.json"
arc_engine = "siesta_slab_ldos"
arc_allow_proxy_fallback = false
siesta_slab_ldos_autogen = "kwant_proxy" # kwant_proxy|disabled
node_method = "wannierberri_flux"
hr_scope = "per_variant"
variant_dft_engine = "siesta"
walltime_per_point = "00:30:00"

[topology.kmesh]
coarse = [20, 20, 20]
refine = [5, 5, 5]
newton_max_iter = 50
gap_threshold_ev = 0.05
max_candidates = 64
dedup_tol = 0.04

[topology.resources]
node_phase_mpi_np = 64
node_phase_threads = 1
arc_phase_mpi_np = 64
arc_phase_threads = 1

[topology.hr_grid]
thickness_mapping = "middle_layer_scale"
middle_layer_role = "active"
reference_thickness_uc = 3
reuse_successful_points = true

[topology.tiering]
# single = full-quality on all points, two_tier = screen all then refine selected points.
mode = "single"
refine_top_k_per_thickness = 2
always_include_pristine = true
selection_metric = "S_total"

[topology.tiering.screen]
arc_engine = "siesta_slab_ldos"
node_method = "wannierberri_flux"
coarse_kmesh = [10, 10, 10]
refine_kmesh = [3, 3, 3]
newton_max_iter = 20
max_candidates = 24

[topology.score]
w_topo = 0.70
w_transport = 0.30

[topology.transport_probe]
enabled = true
n_ensemble = 1
disorder_strength = 0.0
energy_shift_ev = 0.0
thickness_axis = "z"

[report]
enabled = true
markdown = "report/wtec_report.md"
json = "report/wtec_report.json"

[report.validation]
min_mfp_nm = 100.0
require_rho_minimum = true
require_thinning_reduction = true
require_huge_mfp = true
"""


def _load_toml_dict(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]
    data = tomllib.loads(path.read_text())
    if not isinstance(data, dict):
        raise click.ClickException(f"TOML root must be a table/object: {path}")
    return data


def _default_project_template_path(*, required: bool = False) -> Path | None:
    cwd = Path.cwd()
    candidates = [
        cwd / "wtec_project.toml",
        cwd / "wtec_slab_template.toml",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    state = _load_init_state()
    if isinstance(state, dict):
        raw_cwd = state.get("cwd")
        if isinstance(raw_cwd, str) and raw_cwd.strip():
            init_cwd = Path(raw_cwd).expanduser()
            init_candidates = [
                init_cwd / "wtec_project.toml",
                init_cwd / "wtec_slab_template.toml",
            ]
            for p in init_candidates:
                if p.exists():
                    return p.resolve()

    if required:
        raise click.ClickException(
            "No default project template found.\n"
            "Run `wtec init` in your workspace first, or pass an explicit template path."
        )
    return None


def _verify_install(
    *,
    python_executable: str,
    dry_run: bool,
    check_kwant: bool = True,
    check_berry: bool = False,
) -> None:
    if dry_run:
        click.echo("  (skipped in dry-run mode)")
        return

    checks = [
        ("wtec",       "import wtec; print(getattr(wtec, '__version__', 'ok'))"),
        ("numpy",      "import numpy; print(numpy.__version__)"),
        ("scipy",      "import scipy; print(scipy.__version__)"),
        ("ase",        "import ase; print(ase.__version__)"),
        ("pymatgen",   "import importlib.metadata as im; print(im.version('pymatgen'))"),
        ("mp-api",     "import importlib.metadata as im; print(im.version('mp-api'))"),
        ("tbmodels",   "import tbmodels; print(tbmodels.__version__)"),
        ("paramiko",   "import paramiko; print(paramiko.__version__)"),
        ("joblib",     "import joblib; print(joblib.__version__)"),
        ("mpi4py",     "from mpi4py import __version__ as v; print(v)"),
        ("matplotlib", "import matplotlib; print(matplotlib.__version__)"),
        ("click",      "import click; print(click.__version__)"),
    ]
    if check_kwant:
        checks.append(("kwant", "import kwant; print(kwant.__version__)"))
    if check_berry:
        checks.append(
            (
                "wannierberri",
                "import wannierberri as wb; print(getattr(wb, '__version__', 'ok'))",
            )
        )

    for name, code in checks:
        result = subprocess.run(
            [python_executable, "-c", code],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            ver = result.stdout.strip()
            click.echo(click.style(f"  ✓ {name:<12} {ver}", fg="green"))
        else:
            click.echo(click.style(f"  ✗ {name:<12} NOT FOUND", fg="red"))


def _validate_cluster_setup(
    *,
    dry_run: bool,
    python_executable: str | None = None,
) -> None:
    if dry_run:
        click.echo("  (skipped in dry-run mode)")
        return

    if python_executable:
        try:
            current = Path(sys.executable).resolve()
            requested = Path(python_executable).resolve()
        except Exception:
            current = Path(sys.executable)
            requested = Path(python_executable)

        if requested != current:
            cmd = [
                str(requested),
                "-c",
                "from wtec.cli import _validate_cluster_setup_local as f; f()",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout.strip():
                for line in result.stdout.rstrip().splitlines():
                    click.echo(line)
            if result.returncode != 0:
                raise click.ClickException(
                    "Cluster validation failed in venv:\n"
                    f"{result.stderr.strip()}"
                )
            return

    _validate_cluster_setup_local()


def _validate_cluster_setup_local() -> None:

    from wtec.config.cluster import ClusterConfig
    from wtec.cluster.ssh import open_ssh
    from wtec.cluster.submit import JobManager
    from wtec.config.materials import get_material

    try:
        cfg = ClusterConfig.from_env()
    except Exception as exc:
        click.echo(click.style(f"  WARNING: cluster config incomplete: {exc}", fg="yellow"))
        return

    click.echo(f"  Validating cluster: {cfg.host}:{cfg.port} as {cfg.user}")
    siesta_min_version = os.environ.get("TOPOSLAB_SIESTA_MIN_VERSION", "4.1").strip() or "4.1"
    req_d3 = os.environ.get("TOPOSLAB_SIESTA_REQUIRE_D3", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    req_wannier = os.environ.get("TOPOSLAB_SIESTA_REQUIRE_WANNIER", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    with open_ssh(cfg) as ssh:
        jm = JobManager(ssh)
        jm.ensure_remote_commands(
            ["qsub", "qstat", "mpirun"],
            modules=cfg.modules,
            bin_dirs=cfg.bin_dirs,
        )
        qe_tools = ["pw.x", "pw2wannier90.x", "wannier90.x"]
        jm.ensure_remote_commands(
            qe_tools,
            modules=cfg.modules,
            bin_dirs=cfg.bin_dirs,
        )
        jm.ensure_remote_mpi_binaries(
            qe_tools,
            modules=cfg.modules,
            bin_dirs=cfg.bin_dirs,
        )
        queue_used = jm.resolve_queue(cfg.pbs_queue, fallback_order=cfg.pbs_queue_priority)
        cores_per_node = cfg.cores_for_queue(queue_used)
        pseudo_q = shlex.quote(cfg.qe_pseudo_dir)
        rc, _, _ = ssh.run(f"test -d {pseudo_q}", check=False)
        if rc != 0:
            raise RuntimeError(
                f"Configured TOPOSLAB_QE_PSEUDO_DIR not found on cluster: {cfg.qe_pseudo_dir}"
            )
        rc, stdout, _ = ssh.run(
            f"find {pseudo_q} -maxdepth 1 -type f -name '*.UPF' | wc -l",
            check=False,
        )
        upf_count = stdout.strip() if rc == 0 else "unknown"
        default_material = os.environ.get("TOPOSLAB_DEFAULT_MATERIAL", "TaP").strip() or "TaP"
        preset = None
        try:
            preset = get_material(default_material)
            jm.ensure_remote_files(cfg.qe_pseudo_dir, sorted(set(preset.pseudopots.values())))
            upf_check = f"required {default_material} UPFs present"
        except Exception as exc:
            upf_check = f"required UPF check skipped ({exc})"
        try:
            jm.ensure_remote_commands(
                ["siesta", "wannier90.x"],
                modules=cfg.modules,
                bin_dirs=cfg.bin_dirs,
            )
            siesta_info = _validate_remote_siesta_capability(
                ssh,
                modules=cfg.modules,
                bin_dirs=cfg.bin_dirs,
                min_version=siesta_min_version,
                require_d3=req_d3,
                require_wannier=req_wannier,
            )
            siesta_exec_check = (
                "siesta/wannier90 executable check passed "
                f"(siesta={siesta_info['version']}, mpi=yes, "
                f"d3={'yes' if siesta_info['d3'] else 'no'}, "
                f"wannier={'yes' if siesta_info['wannier'] else 'no'})"
            )
        except Exception as exc:
            siesta_exec_check = f"siesta executable check failed ({exc})"
            raise
        try:
            if preset is not None:
                jm.ensure_remote_files(
                    cfg.siesta_pseudo_dir,
                    sorted(set(preset.siesta_pseudopots.values())),
                )
                siesta_pseudo_check = f"required {default_material} SIESTA pseudos present"
            else:
                siesta_pseudo_check = "required SIESTA pseudo check skipped (no preset)"
        except Exception as exc:
            siesta_pseudo_check = f"required SIESTA pseudo check failed ({exc})"
            raise
        click.echo(
            "  ✓ qsub/qstat/mpirun + QE/Wannier MPI executables found, "
            f"queue={queue_used}, cores_per_node={cores_per_node}, "
            f"QE pseudo dir OK, UPF count={upf_count}, {upf_check}; "
            f"{siesta_exec_check}; {siesta_pseudo_check}"
        )


# ---------------------------------------------------------------------------
# Cluster-side Python feature setup
# ---------------------------------------------------------------------------

def _dedupe_nonempty(values: list[str] | tuple[str, ...]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        val = str(raw).strip()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def _cluster_bin_dirs_from_env() -> list[str]:
    raw = os.environ.get("TOPOSLAB_CLUSTER_BIN_DIRS", "").strip()
    if not raw:
        return []
    return _dedupe_nonempty(raw.split(","))


def _wrap_with_modules(command: str, modules: list[str], *, bin_dirs: list[str] | None = None) -> str:
    if not modules and not bin_dirs:
        return command
    module_prefix = (
        " && ".join(
            f"module load {shlex.quote(m)} >/dev/null 2>&1" for m in modules
        )
        if modules
        else ""
    )
    path_prefix = ""
    if bin_dirs:
        quoted = ":".join(shlex.quote(p) for p in _dedupe_nonempty(bin_dirs))
        if quoted:
            path_prefix = f"export PATH={quoted}:$PATH"
    pieces = [p for p in [module_prefix, path_prefix, command] if p]
    wrapped = " && ".join(pieces)
    return f"bash -lc {shlex.quote(wrapped)}"


def _remote_importable(
    ssh,
    *,
    python_executable: str,
    module_name: str,
    modules: list[str],
    bin_dirs: list[str] | None = None,
) -> bool:
    cmd = (
        f"{shlex.quote(python_executable)} -c "
        f"{shlex.quote(f'import {module_name}; print({module_name}.__name__)')}"
    )
    rc, _, _ = ssh.run(_wrap_with_modules(cmd, modules, bin_dirs=bin_dirs), check=False)
    return rc == 0


def _remote_python_mm(
    ssh,
    *,
    python_executable: str,
    modules: list[str],
    bin_dirs: list[str] | None = None,
) -> tuple[int, int]:
    code = "import sys; print(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))"
    cmd = f"{shlex.quote(python_executable)} -c {shlex.quote(code)}"
    rc, out, _ = ssh.run(_wrap_with_modules(cmd, modules, bin_dirs=bin_dirs), check=False)
    if rc != 0:
        return (3, 10)
    raw = out.strip()
    try:
        a, b = raw.split(".", 1)
        return int(a), int(b)
    except Exception:
        return (3, 10)


def _remote_install_or_raise(
    ssh,
    *,
    command: str,
    modules: list[str],
    bin_dirs: list[str] | None = None,
) -> None:
    rc, out, err = ssh.run(_wrap_with_modules(command, modules, bin_dirs=bin_dirs), check=False)
    if rc != 0:
        merged = (out + "\n" + err).strip()
        raise click.ClickException(f"Remote install failed:\n{command}\n{merged}")


def _prepare_remote_wannierberri(
    ssh,
    *,
    python_executable: str,
    modules: list[str],
    bin_dirs: list[str] | None = None,
) -> None:
    if _remote_importable(
        ssh,
        python_executable=python_executable,
        module_name="wannierberri",
        modules=modules,
        bin_dirs=bin_dirs,
    ):
        return

    py_mm = _remote_python_mm(
        ssh,
        python_executable=python_executable,
        modules=modules,
        bin_dirs=bin_dirs,
    )
    wb_spec = "wannierberri==1.0.1"
    pyq = shlex.quote(python_executable)
    click.echo(f"  Preparing remote wannierberri ({wb_spec})")
    _remote_install_or_raise(
        ssh,
        command=(
            f"{pyq} -m pip install --user --upgrade --force-reinstall "
            "'numpy<2' 'scipy<1.14'"
        ),
        modules=modules,
        bin_dirs=bin_dirs,
    )
    _remote_install_or_raise(
        ssh,
        command=f"{pyq} -m pip install --user --upgrade 'ray>=2.10' irrep",
        modules=modules,
        bin_dirs=bin_dirs,
    )
    _remote_install_or_raise(
        ssh,
        command=f"{pyq} -m pip install --user --upgrade --force-reinstall {shlex.quote(wb_spec)}",
        modules=modules,
        bin_dirs=bin_dirs,
    )
    if not _remote_importable(
        ssh,
        python_executable=python_executable,
        module_name="wannierberri",
        modules=modules,
        bin_dirs=bin_dirs,
    ):
        raise click.ClickException("Remote wannierberri remains unavailable after installation.")


def _prepare_remote_tbmodels(
    ssh,
    *,
    python_executable: str,
    modules: list[str],
    bin_dirs: list[str] | None = None,
) -> None:
    if _remote_importable(
        ssh,
        python_executable=python_executable,
        module_name="tbmodels",
        modules=modules,
        bin_dirs=bin_dirs,
    ):
        return
    pyq = shlex.quote(python_executable)
    click.echo("  Preparing remote tbmodels")
    _remote_install_or_raise(
        ssh,
        command=f"{pyq} -m pip install --user --upgrade 'tbmodels>=1.4'",
        modules=modules,
        bin_dirs=bin_dirs,
    )
    if not _remote_importable(
        ssh,
        python_executable=python_executable,
        module_name="tbmodels",
        modules=modules,
        bin_dirs=bin_dirs,
    ):
        raise click.ClickException("Remote tbmodels remains unavailable after installation.")


def _prepare_remote_sisl(
    ssh,
    *,
    python_executable: str,
    modules: list[str],
    bin_dirs: list[str] | None = None,
) -> None:
    if _remote_importable(
        ssh,
        python_executable=python_executable,
        module_name="sisl",
        modules=modules,
        bin_dirs=bin_dirs,
    ):
        return
    pyq = shlex.quote(python_executable)
    click.echo("  Preparing remote sisl")
    _remote_install_or_raise(
        ssh,
        command=f"{pyq} -m pip install --user --upgrade 'sisl>=0.14'",
        modules=modules,
        bin_dirs=bin_dirs,
    )
    if not _remote_importable(
        ssh,
        python_executable=python_executable,
        module_name="sisl",
        modules=modules,
        bin_dirs=bin_dirs,
    ):
        raise click.ClickException("Remote sisl remains unavailable after installation.")


def _prepare_remote_kwant(
    ssh,
    *,
    python_executable: str,
    modules: list[str],
    bin_dirs: list[str] | None = None,
) -> None:
    if _remote_importable(
        ssh,
        python_executable=python_executable,
        module_name="kwant",
        modules=modules,
        bin_dirs=bin_dirs,
    ):
        return
    pyq = shlex.quote(python_executable)
    click.echo("  Preparing remote kwant (source build with numpy<2)")
    _remote_install_or_raise(
        ssh,
        command=(
            f"{pyq} -m pip install --user --upgrade --force-reinstall "
            "'numpy<2' 'scipy<1.14'"
        ),
        modules=modules,
        bin_dirs=bin_dirs,
    )
    _remote_install_or_raise(
        ssh,
        command=f"{pyq} -m pip install --user --upgrade meson-python ninja cython setuptools-scm",
        modules=modules,
        bin_dirs=bin_dirs,
    )
    _remote_install_or_raise(
        ssh,
        command=(
            f"{pyq} -m pip install --user --force-reinstall --no-deps "
            "--no-build-isolation --no-binary=:all: --no-cache-dir kwant"
        ),
        modules=modules,
        bin_dirs=bin_dirs,
    )
    if not _remote_importable(
        ssh,
        python_executable=python_executable,
        module_name="kwant",
        modules=modules,
        bin_dirs=bin_dirs,
    ):
        raise click.ClickException("Remote kwant remains unavailable after installation.")


def _remote_module_version(
    ssh,
    *,
    python_executable: str,
    module_name: str,
    modules: list[str],
    bin_dirs: list[str] | None = None,
) -> str | None:
    code = f"import {module_name}; print(getattr({module_name}, '__version__', 'ok'))"
    cmd = f"{shlex.quote(python_executable)} -c {shlex.quote(code)}"
    rc, out, _ = ssh.run(_wrap_with_modules(cmd, modules, bin_dirs=bin_dirs), check=False)
    if rc != 0:
        return None
    return out.strip() or "ok"


def _remote_command_exists(
    ssh,
    *,
    command_name: str,
    modules: list[str],
    bin_dirs: list[str] | None = None,
) -> bool:
    cmd = f"command -v {shlex.quote(command_name)} >/dev/null 2>&1"
    rc, _, _ = ssh.run(_wrap_with_modules(cmd, modules, bin_dirs=bin_dirs), check=False)
    return rc == 0


def _remote_command_mpi_linked(
    ssh,
    *,
    command_name: str,
    modules: list[str],
    bin_dirs: list[str] | None = None,
) -> bool:
    cmd = (
        "resolved=$(command -v "
        + shlex.quote(command_name)
        + ") && [ -n \"$resolved\" ] && [ -x \"$resolved\" ] && "
        "ldd \"$resolved\" 2>/dev/null | "
        "grep -Eiq '(libmpi|openmpi|mpich|open-rte)'"
    )
    rc, _, _ = ssh.run(_wrap_with_modules(cmd, modules, bin_dirs=bin_dirs), check=False)
    return rc == 0


def _parse_version_tuple(raw: str) -> tuple[int, ...]:
    vals = [int(x) for x in re.findall(r"\d+", str(raw))]
    return tuple(vals[:3])


def _remote_command_version(
    ssh,
    *,
    command_name: str,
    modules: list[str],
    bin_dirs: list[str] | None = None,
) -> str | None:
    cmd = (
        "set -e; "
        f"if {shlex.quote(command_name)} --version >/tmp/wtec_version.$$ 2>&1; then "
        "cat /tmp/wtec_version.$$; "
        f"elif {shlex.quote(command_name)} -v >/tmp/wtec_version.$$ 2>&1; then "
        "cat /tmp/wtec_version.$$; "
        "else exit 1; fi"
    )
    rc, out, _ = ssh.run(_wrap_with_modules(cmd, modules, bin_dirs=bin_dirs), check=False)
    if rc != 0:
        return None
    text = out.strip()
    if not text:
        return None
    m = re.search(r"(\d+\.\d+(?:\.\d+)?)", text)
    return m.group(1) if m else None


def _remote_siesta_feature_flags(
    ssh,
    *,
    modules: list[str],
    bin_dirs: list[str] | None = None,
) -> dict[str, bool]:
    cmd = (
        "set -e; "
        "resolved=$(command -v siesta); "
        "[ -n \"$resolved\" ] && [ -x \"$resolved\" ]; "
        "strings \"$resolved\" 2>/dev/null | "
        "grep -Eiq '(DFTD3|dftd3|grimme)' && d3=1 || d3=0; "
        "strings \"$resolved\" 2>/dev/null | "
        "grep -Eiq '(Siesta2Wannier90|wannier90|WriteMmn|WriteAmn|WriteEig)' && wan=1 || wan=0; "
        "printf 'd3=%s\\nwan=%s\\n' \"$d3\" \"$wan\""
    )
    rc, out, _ = ssh.run(_wrap_with_modules(cmd, modules, bin_dirs=bin_dirs), check=False)
    if rc != 0:
        return {"d3": False, "wannier": False}
    d3 = False
    wan = False
    for line in out.splitlines():
        s = line.strip().lower()
        if s == "d3=1":
            d3 = True
        elif s == "wan=1":
            wan = True
    return {"d3": d3, "wannier": wan}


def _validate_remote_siesta_capability(
    ssh,
    *,
    modules: list[str],
    bin_dirs: list[str] | None = None,
    min_version: str,
    require_d3: bool = True,
    require_wannier: bool = True,
) -> dict[str, Any]:
    version = _remote_command_version(
        ssh,
        command_name="siesta",
        modules=modules,
        bin_dirs=bin_dirs,
    )
    if version is None:
        raise click.ClickException("Could not resolve SIESTA version (`siesta --version` failed).")
    current = _parse_version_tuple(version)
    minimum = _parse_version_tuple(min_version)
    if minimum and current and current < minimum:
        raise click.ClickException(
            f"SIESTA version {version} is below required minimum {min_version}."
        )
    if not _remote_command_mpi_linked(
        ssh,
        command_name="siesta",
        modules=modules,
        bin_dirs=bin_dirs,
    ):
        raise click.ClickException(
            "SIESTA binary appears non-MPI linked. MPI-enabled SIESTA is required."
        )
    flags = _remote_siesta_feature_flags(
        ssh,
        modules=modules,
        bin_dirs=bin_dirs,
    )
    if require_d3 and not flags.get("d3", False):
        raise click.ClickException(
            "SIESTA binary does not expose DFT-D3 capability markers. "
            "Rebuild with D3 support."
        )
    if require_wannier and not flags.get("wannier", False):
        raise click.ClickException(
            "SIESTA binary does not expose Wannier bridge markers. "
            "Rebuild with SIESTA↔Wannier support."
        )
    return {
        "version": version,
        "mpi": True,
        "d3": bool(flags.get("d3", False)),
        "wannier": bool(flags.get("wannier", False)),
    }


def _remote_find_first_existing_dir(
    ssh,
    *,
    candidates: list[str],
    modules: list[str],
    bin_dirs: list[str] | None = None,
) -> str | None:
    dirs = _dedupe_nonempty(candidates)
    if not dirs:
        return None
    listed = " ".join(shlex.quote(p) for p in dirs)
    cmd = f"for d in {listed}; do if [ -d \"$d\" ]; then printf '%s' \"$d\"; exit 0; fi; done; exit 1"
    rc, out, _ = ssh.run(_wrap_with_modules(cmd, modules, bin_dirs=bin_dirs), check=False)
    if rc != 0:
        return None
    val = out.strip()
    return val or None


def _remote_find_first_executable_dir(
    ssh,
    *,
    candidates: list[str],
    executable_name: str,
    modules: list[str],
    bin_dirs: list[str] | None = None,
) -> str | None:
    dirs = _dedupe_nonempty(candidates)
    if not dirs:
        return None
    listed = " ".join(shlex.quote(p) for p in dirs)
    exe_q = shlex.quote(executable_name)
    cmd = (
        f"for d in {listed}; do "
        f"if [ -x \"$d\"/{exe_q} ]; then printf '%s' \"$d\"; exit 0; fi; "
        "done; exit 1"
    )
    rc, out, _ = ssh.run(_wrap_with_modules(cmd, modules, bin_dirs=bin_dirs), check=False)
    if rc != 0:
        return None
    val = out.strip()
    return val or None


def _remote_run_or_raise(
    ssh,
    *,
    command: str,
    modules: list[str],
    bin_dirs: list[str] | None = None,
    label: str | None = None,
) -> None:
    rc, out, err = ssh.run(_wrap_with_modules(command, modules, bin_dirs=bin_dirs), check=False)
    if rc == 0:
        return
    merged = (out + "\n" + err).strip()
    subject = label or "Remote command"
    raise click.ClickException(f"{subject} failed:\n{command}\n{merged}")


def _cluster_build_jobs_from_env() -> int:
    raw = os.environ.get("TOPOSLAB_CLUSTER_BUILD_JOBS", "").strip()
    if not raw:
        return 8
    try:
        jobs = int(raw)
    except Exception:
        return 8
    return jobs if jobs > 0 else 8


def _persist_cluster_bin_dirs(bin_dirs: list[str]) -> None:
    normalized = _dedupe_nonempty(bin_dirs)
    if not normalized:
        return
    joined = ",".join(normalized)
    os.environ["TOPOSLAB_CLUSTER_BIN_DIRS"] = joined
    env_file = Path(".env")
    if env_file.exists():
        current = env_file.read_text()
        rendered = _apply_env_updates(current, {"TOPOSLAB_CLUSTER_BIN_DIRS": joined})
        if rendered != current:
            env_file.write_text(rendered)
            click.echo(f"  Updated .env: TOPOSLAB_CLUSTER_BIN_DIRS={joined}")


def _prepare_cluster_toolchain_setup(
    *,
    dry_run: bool,
) -> None:
    if dry_run:
        click.echo("  (skipped in dry-run mode)")
        return

    from wtec.config.cluster import ClusterConfig
    from wtec.cluster.ssh import open_ssh
    from wtec.cluster.submit import JobManager

    try:
        cfg = ClusterConfig.from_env()
    except Exception as exc:
        click.echo(click.style(f"  WARNING: cluster config incomplete; skip remote tool prep: {exc}", fg="yellow"))
        return

    build_jobs = _cluster_build_jobs_from_env()
    bin_dirs = _cluster_bin_dirs_from_env() or list(cfg.bin_dirs)
    bin_dirs = _dedupe_nonempty(bin_dirs)
    cluster_user = (cfg.user or os.environ.get("TOPOSLAB_CLUSTER_USER", "")).strip()

    qe_source_env = os.environ.get("TOPOSLAB_QE_SOURCE_DIR", "").strip()
    qe_candidates = _dedupe_nonempty(
        [
            qe_source_env,
            f"/home/{cluster_user}/src/q-e" if cluster_user else "",
            f"/home/{cluster_user}/src/q-e-qe" if cluster_user else "",
            f"/home/{cluster_user}/src/QE/q-e" if cluster_user else "",
            f"/home/{cluster_user}/src/espresso" if cluster_user else "",
        ]
    )
    siesta_source_env = os.environ.get("TOPOSLAB_SIESTA_SOURCE_DIR", "").strip()
    siesta_source_candidates = _dedupe_nonempty(
        [
            siesta_source_env,
            f"/home/{cluster_user}/src/siesta" if cluster_user else "",
            f"/home/{cluster_user}/src/siesta-trunk" if cluster_user else "",
            f"/home/{cluster_user}/src/Siesta" if cluster_user else "",
        ]
    )

    wannier_source_env = os.environ.get("TOPOSLAB_WANNIER90_SOURCE_DIR", "").strip()
    w90_source_candidates = _dedupe_nonempty(
        [
            wannier_source_env,
            f"/home/{cluster_user}/src/wannier90" if cluster_user else "",
            f"/home/{cluster_user}/src/wannier90-3.1.0" if cluster_user else "",
            f"/home/{cluster_user}/src/Wannier90" if cluster_user else "",
        ]
    )
    w90_bin_candidates = _dedupe_nonempty(
        [
            "/TGM/Apps/VASP/bin",
            f"/home/{cluster_user}/src/wannier90/bin" if cluster_user else "",
            f"/home/{cluster_user}/src/wannier90" if cluster_user else "",
            f"/home/{cluster_user}/src/wannier90-3.1.0/bin" if cluster_user else "",
            f"/home/{cluster_user}/src/wannier90-3.1.0" if cluster_user else "",
        ]
    )

    click.echo(
        f"  Preparing cluster executables on {cfg.host}:{cfg.port} "
        f"(make -j {build_jobs})"
    )
    siesta_min_version = os.environ.get("TOPOSLAB_SIESTA_MIN_VERSION", "4.1").strip() or "4.1"
    req_d3 = os.environ.get("TOPOSLAB_SIESTA_REQUIRE_D3", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    req_wannier = os.environ.get("TOPOSLAB_SIESTA_REQUIRE_WANNIER", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    with open_ssh(cfg) as ssh:
        jm = JobManager(ssh)
        jm.ensure_remote_commands(
            ["qsub", "qstat", "mpirun"],
            modules=cfg.modules,
            bin_dirs=bin_dirs,
        )

        needed = ["pw.x", "pw2wannier90.x", "wannier90.x"]
        qe_source_used: str | None = None
        w90_source_used: str | None = None
        resolved_bin_dirs = list(bin_dirs)
        missing = [
            exe
            for exe in needed
            if not _remote_command_exists(
                ssh,
                command_name=exe,
                modules=cfg.modules,
                bin_dirs=bin_dirs,
            )
        ]
        non_mpi = [
            exe
            for exe in needed
            if exe not in missing
            and not _remote_command_mpi_linked(
                ssh,
                command_name=exe,
                modules=cfg.modules,
                bin_dirs=bin_dirs,
            )
        ]
        if not missing and not non_mpi:
            click.echo("  ✓ pw.x, pw2wannier90.x, wannier90.x already available (MPI-linked)")
        else:
            if missing:
                click.echo(click.style(f"  Missing executables: {', '.join(missing)}", fg="yellow"))
            if non_mpi:
                click.echo(
                    click.style(
                        "  Non-MPI executables detected (will rebuild): " + ", ".join(non_mpi),
                        fg="yellow",
                    )
                )

        if "pw.x" in missing or "pw2wannier90.x" in missing or "pw.x" in non_mpi or "pw2wannier90.x" in non_mpi:
            qe_source = _remote_find_first_existing_dir(
                ssh,
                candidates=qe_candidates,
                modules=cfg.modules,
                bin_dirs=resolved_bin_dirs,
            )
            if qe_source is None:
                raise click.ClickException(
                    "QE tools are missing and source directory was not found. "
                    "Set TOPOSLAB_QE_SOURCE_DIR to your QE source checkout."
                )
            qe_source_used = qe_source
            click.echo(f"  Building QE targets from {qe_source}")
            if "pw.x" in missing:
                _remote_run_or_raise(
                    ssh,
                    command=f"cd {shlex.quote(qe_source)} && make pw -j {build_jobs}",
                    modules=cfg.modules,
                    bin_dirs=resolved_bin_dirs,
                    label="QE build (pw)",
                )
            if "pw2wannier90.x" in missing:
                _remote_run_or_raise(
                    ssh,
                    command=f"cd {shlex.quote(qe_source)} && make pp -j {build_jobs}",
                    modules=cfg.modules,
                    bin_dirs=resolved_bin_dirs,
                    label="QE build (pp/pw2wannier90)",
                )
            qe_bin = f"{qe_source.rstrip('/')}/bin"
            if qe_bin not in resolved_bin_dirs:
                resolved_bin_dirs.append(qe_bin)

        if "wannier90.x" in missing or "wannier90.x" in non_mpi:
            w90_bin = _remote_find_first_executable_dir(
                ssh,
                candidates=resolved_bin_dirs + w90_bin_candidates,
                executable_name="wannier90.x",
                modules=cfg.modules,
                bin_dirs=resolved_bin_dirs,
            )
            if w90_bin is None:
                w90_source = _remote_find_first_existing_dir(
                    ssh,
                    candidates=w90_source_candidates,
                    modules=cfg.modules,
                    bin_dirs=resolved_bin_dirs,
                )
                if w90_source is None:
                    raise click.ClickException(
                        "wannier90.x is missing and no source directory was found. "
                        "Set TOPOSLAB_WANNIER90_SOURCE_DIR or add a bin dir via "
                        "TOPOSLAB_CLUSTER_BIN_DIRS."
                    )
                w90_source_used = w90_source
                click.echo(f"  Building wannier90 from {w90_source}")
                _remote_run_or_raise(
                    ssh,
                    command=(
                        f"cd {shlex.quote(w90_source)} && "
                        f"(make -j {build_jobs} || make all -j {build_jobs})"
                    ),
                    modules=cfg.modules,
                    bin_dirs=resolved_bin_dirs,
                    label="Wannier90 build",
                )
                w90_bin = _remote_find_first_executable_dir(
                    ssh,
                    candidates=[f"{w90_source.rstrip('/')}/bin", w90_source],
                    executable_name="wannier90.x",
                    modules=cfg.modules,
                    bin_dirs=resolved_bin_dirs,
                )
            if w90_bin is None:
                raise click.ClickException("wannier90.x was not found after build/install attempts.")
            if w90_bin not in resolved_bin_dirs:
                resolved_bin_dirs.append(w90_bin)

        resolved_bin_dirs = _dedupe_nonempty(resolved_bin_dirs)
        jm.ensure_remote_commands(
            needed,
            modules=cfg.modules,
            bin_dirs=resolved_bin_dirs,
        )
        try:
            jm.ensure_remote_mpi_binaries(
                needed,
                modules=cfg.modules,
                bin_dirs=resolved_bin_dirs,
            )
        except Exception as exc:
            text = str(exc)
            qe_bad = ("pw.x" in text) or ("pw2wannier90.x" in text)
            w90_bad = "wannier90.x" in text

            if qe_bad:
                if qe_source_used is None:
                    qe_source_used = _remote_find_first_existing_dir(
                        ssh,
                        candidates=qe_candidates,
                        modules=cfg.modules,
                        bin_dirs=resolved_bin_dirs,
                    )
                if qe_source_used:
                    click.echo(
                        click.style(
                            "  Attempting QE MPI reconfigure/rebuild (mpif90/mpicc)...",
                            fg="yellow",
                        )
                    )
                    _remote_run_or_raise(
                        ssh,
                        command=(
                            f"cd {shlex.quote(qe_source_used)} && "
                            "if [ -x ./configure ]; then "
                            "./configure MPIF90=mpif90 FC=mpif90 F90=mpif90 CC=mpicc CXX=mpicxx; "
                            "fi && "
                            "make clean >/dev/null 2>&1 || true && "
                            f"make pw pp -j {build_jobs}"
                        ),
                        modules=cfg.modules,
                        bin_dirs=resolved_bin_dirs,
                        label="QE MPI reconfigure/build",
                    )
                else:
                    raise click.ClickException(
                        "Non-MPI QE binaries detected and QE source directory was not found. "
                        "Set TOPOSLAB_QE_SOURCE_DIR to an MPI-capable QE source checkout."
                    ) from exc

            if w90_bad:
                if w90_source_used is None:
                    w90_source_used = _remote_find_first_existing_dir(
                        ssh,
                        candidates=w90_source_candidates,
                        modules=cfg.modules,
                        bin_dirs=resolved_bin_dirs,
                    )
                if w90_source_used:
                    click.echo(
                        click.style(
                            "  Attempting wannier90 MPI rebuild...",
                            fg="yellow",
                        )
                    )
                    _remote_run_or_raise(
                        ssh,
                        command=(
                            f"cd {shlex.quote(w90_source_used)} && "
                            "make clean >/dev/null 2>&1 || true && "
                            f"(make -j {build_jobs} || make all -j {build_jobs})"
                        ),
                        modules=cfg.modules,
                        bin_dirs=resolved_bin_dirs,
                        label="Wannier90 MPI rebuild",
                    )
                else:
                    raise click.ClickException(
                        "Non-MPI wannier90.x detected and source directory was not found. "
                        "Set TOPOSLAB_WANNIER90_SOURCE_DIR to an MPI-capable source checkout."
                    ) from exc

            jm.ensure_remote_commands(
                needed,
                modules=cfg.modules,
                bin_dirs=resolved_bin_dirs,
            )
            jm.ensure_remote_mpi_binaries(
                needed,
                modules=cfg.modules,
                bin_dirs=resolved_bin_dirs,
            )

        # SIESTA toolchain (required when default DFT engine is siesta).
        siesta_ready = _remote_command_exists(
            ssh,
            command_name="siesta",
            modules=cfg.modules,
            bin_dirs=resolved_bin_dirs,
        )
        siesta_mpi = (
            _remote_command_mpi_linked(
                ssh,
                command_name="siesta",
                modules=cfg.modules,
                bin_dirs=resolved_bin_dirs,
            )
            if siesta_ready
            else False
        )
        if siesta_ready and siesta_mpi:
            click.echo("  ✓ siesta already available (MPI-linked)")
        else:
            reason = "missing" if not siesta_ready else "non-MPI"
            siesta_src = _remote_find_first_existing_dir(
                ssh,
                candidates=siesta_source_candidates,
                modules=cfg.modules,
                bin_dirs=resolved_bin_dirs,
            )
            if not siesta_src:
                msg = (
                    "SIESTA is "
                    + reason
                    + " and source directory was not found. "
                    "Set TOPOSLAB_SIESTA_SOURCE_DIR to a SIESTA source checkout."
                )
                raise click.ClickException(msg)
            else:
                click.echo(f"  Building siesta from {siesta_src}")
                src_q = shlex.quote(siesta_src)
                build_dir = f"{siesta_src.rstrip('/')}/build_wtec"
                install_dir = f"{siesta_src.rstrip('/')}/install_wtec"
                build_q = shlex.quote(build_dir)
                install_q = shlex.quote(install_dir)
                _remote_run_or_raise(
                    ssh,
                    command=(
                        f"mkdir -p {build_q} {install_q} && "
                        f"cd {src_q} && "
                        f"cmake -S . -B {build_q} "
                        f"-DCMAKE_BUILD_TYPE=Release "
                        f"-DCMAKE_INSTALL_PREFIX={install_q} "
                        "-DSIESTA_WITH_MPI=ON "
                        "-DSIESTA_WITH_DFTD3=ON && "
                        f"cmake --build {build_q} -j {build_jobs} && "
                        f"cmake --install {build_q}"
                    ),
                    modules=cfg.modules,
                    bin_dirs=resolved_bin_dirs,
                    label="SIESTA build/install",
                )
                siesta_bin = _remote_find_first_executable_dir(
                    ssh,
                    candidates=[
                        f"{install_dir}/bin",
                        build_dir,
                        f"{build_dir}/Obj",
                        f"{siesta_src.rstrip('/')}/Obj",
                    ],
                    executable_name="siesta",
                    modules=cfg.modules,
                    bin_dirs=resolved_bin_dirs,
                )
                if siesta_bin is None:
                    raise click.ClickException(
                        "siesta executable was not found after build/install attempts."
                    )
                if siesta_bin not in resolved_bin_dirs:
                    resolved_bin_dirs.append(siesta_bin)
                jm.ensure_remote_commands(
                    ["siesta"],
                    modules=cfg.modules,
                    bin_dirs=resolved_bin_dirs,
                )
                jm.ensure_remote_mpi_binaries(
                    ["siesta"],
                    modules=cfg.modules,
                    bin_dirs=resolved_bin_dirs,
                )

        siesta_info = _validate_remote_siesta_capability(
            ssh,
            modules=cfg.modules,
            bin_dirs=resolved_bin_dirs,
            min_version=siesta_min_version,
            require_d3=req_d3,
            require_wannier=req_wannier,
        )
        click.echo(
            click.style(
                "  ✓ siesta capability check passed: "
                f"version={siesta_info['version']}, mpi=yes, "
                f"d3={'yes' if siesta_info['d3'] else 'no'}, "
                f"wannier={'yes' if siesta_info['wannier'] else 'no'}",
                fg="green",
            )
        )

        if resolved_bin_dirs != bin_dirs:
            _persist_cluster_bin_dirs(resolved_bin_dirs)
        click.echo(
            click.style(
                "  ✓ cluster executables ready: "
                + ", ".join(needed + ["siesta"]),
                fg="green",
            )
        )


def _prepare_cluster_pseudopotential_setup(
    *,
    dry_run: bool,
) -> None:
    if dry_run:
        click.echo("  (skipped in dry-run mode)")
        return

    from wtec.config.cluster import ClusterConfig
    from wtec.cluster.ssh import open_ssh

    try:
        cfg = ClusterConfig.from_env()
    except Exception as exc:
        click.echo(
            click.style(
                f"  WARNING: cluster config incomplete; skip pseudo prep: {exc}",
                fg="yellow",
            )
        )
        return

    pseudo_dir = str(cfg.qe_pseudo_dir).strip()
    if not pseudo_dir:
        raise click.ClickException(
            "TOPOSLAB_QE_PSEUDO_DIR is empty. Set it in .env or pass --qe-pseudo-dir."
        )

    cluster_user = (cfg.user or os.environ.get("TOPOSLAB_CLUSTER_USER", "")).strip()
    build_jobs = _cluster_build_jobs_from_env()
    pseudo_source_env = os.environ.get("TOPOSLAB_QE_PSEUDO_SOURCE_DIR", "").strip()
    source_candidates = _dedupe_nonempty(
        [
            pseudo_source_env,
            f"/home/{cluster_user}/src/QE_pseudo/pslibrary" if cluster_user else "",
            f"/home/{cluster_user}/src/QE_pseudo/pslibrary.1.0.0" if cluster_user else "",
            f"/home/{cluster_user}/src/QE_pseudo" if cluster_user else "",
            f"/home/{cluster_user}/qe/pseudo" if cluster_user else "",
        ]
    )
    qe_source_env = os.environ.get("TOPOSLAB_QE_SOURCE_DIR", "").strip()
    qe_source_candidates = _dedupe_nonempty(
        [
            qe_source_env,
            f"/home/{cluster_user}/src/q-e" if cluster_user else "",
            f"/home/{cluster_user}/src/q-e-qe" if cluster_user else "",
            f"/home/{cluster_user}/src/QE/q-e" if cluster_user else "",
            f"/home/{cluster_user}/src/espresso" if cluster_user else "",
        ]
    )
    bin_dirs = _cluster_bin_dirs_from_env() or list(cfg.bin_dirs)
    pseudo_q = shlex.quote(pseudo_dir)

    def _upf_count(ssh) -> int:
        cmd = (
            f"find {pseudo_q} -type f "
            "\\( -name '*.UPF' -o -name '*.upf' \\) | wc -l"
        )
        rc, out, _ = ssh.run(_wrap_with_modules(cmd, cfg.modules, bin_dirs=bin_dirs), check=False)
        if rc != 0:
            return 0
        try:
            return int(out.strip())
        except Exception:
            return 0

    click.echo(f"  Ensuring cluster pseudo dir: {pseudo_dir}")
    with open_ssh(cfg) as ssh:
        _remote_run_or_raise(
            ssh,
            command=f"mkdir -p {pseudo_q}",
            modules=cfg.modules,
            bin_dirs=bin_dirs,
            label="Pseudo dir creation",
        )
        count = _upf_count(ssh)
        if count > 0:
            click.echo(click.style(f"  ✓ pseudo dir ready, UPF count={count}", fg="green"))
            return

        copied_from: str | None = None
        for candidate in source_candidates:
            cand_q = shlex.quote(candidate)
            probe_cmd = (
                f"test -d {cand_q} && "
                f"find {cand_q} -type f \\( -name '*.UPF' -o -name '*.upf' \\) "
                "| head -n 1"
            )
            rc, out, _ = ssh.run(_wrap_with_modules(probe_cmd, cfg.modules, bin_dirs=bin_dirs), check=False)
            if rc != 0 or not out.strip():
                continue
            click.echo(f"  Copying UPF files from {candidate}")
            _remote_run_or_raise(
                ssh,
                command=(
                    f"find {cand_q} -type f \\( -name '*.UPF' -o -name '*.upf' \\) "
                    f"-exec cp -n {{}} {pseudo_q}/ \\;"
                ),
                modules=cfg.modules,
                bin_dirs=bin_dirs,
                label=f"Pseudo copy from {candidate}",
            )
            count = _upf_count(ssh)
            if count > 0:
                copied_from = candidate
                break

        # Last resort: generate UPFs from pslibrary source when available.
        if count <= 0:
            pslib_src = _remote_find_first_existing_dir(
                ssh,
                candidates=source_candidates,
                modules=cfg.modules,
                bin_dirs=bin_dirs,
            )
            if pslib_src:
                pslib_q = shlex.quote(pslib_src)
                click.echo(
                    click.style(
                        f"  No UPFs found; attempting pslibrary generation in {pslib_src}",
                        fg="yellow",
                    )
                )
                # Ensure ld1.x exists, build it from QE source if needed.
                has_ld1 = _remote_command_exists(
                    ssh,
                    command_name="ld1.x",
                    modules=cfg.modules,
                    bin_dirs=bin_dirs,
                )
                if not has_ld1:
                    qe_src = _remote_find_first_existing_dir(
                        ssh,
                        candidates=qe_source_candidates,
                        modules=cfg.modules,
                        bin_dirs=bin_dirs,
                    )
                    if qe_src:
                        qe_q = shlex.quote(qe_src)
                        click.echo(
                            click.style(
                                f"  Building missing ld1.x from QE source: {qe_src}",
                                fg="yellow",
                            )
                        )
                        _remote_run_or_raise(
                            ssh,
                            command=f"cd {qe_q} && make ld1 -j {build_jobs}",
                            modules=cfg.modules,
                            bin_dirs=bin_dirs,
                            label="QE build (ld1)",
                        )
                    else:
                        click.echo(
                            click.style(
                                "  WARNING: ld1.x is missing and QE source was not found; "
                                "cannot auto-generate pslibrary UPFs.",
                                fg="yellow",
                            )
                        )
                # Point pslibrary QE_path to discovered QE source when available.
                qe_src_for_ps = _remote_find_first_existing_dir(
                    ssh,
                    candidates=qe_source_candidates,
                    modules=cfg.modules,
                    bin_dirs=bin_dirs,
                )
                if qe_src_for_ps:
                    qe_ps_q = shlex.quote(qe_src_for_ps.rstrip("/") + "/")
                    _remote_run_or_raise(
                        ssh,
                        command=(
                            f"if [ -f {pslib_q}/QE_path ]; then "
                            f"printf \"#!/bin/bash\\n\\nPWDIR='{qe_src_for_ps.rstrip('/')}/'\\n\" > {pslib_q}/QE_path; "
                            "fi"
                        ),
                        modules=cfg.modules,
                        bin_dirs=bin_dirs,
                        label="pslibrary QE_path update",
                    )
                # Run pslibrary generation script only when available.
                _remote_run_or_raise(
                    ssh,
                    command=(
                        f"cd {pslib_q} && "
                        "if [ -x ./make_all_ps ]; then bash ./make_all_ps; "
                        "elif [ -x ./make_ps ]; then "
                        "( cd ./pbe && . ../make_ps ); "
                        "else exit 1; fi"
                    ),
                    modules=cfg.modules,
                    bin_dirs=bin_dirs,
                    label="pslibrary UPF generation",
                )
                _remote_run_or_raise(
                    ssh,
                    command=(
                        f"find {pslib_q} -type f \\( -name '*.UPF' -o -name '*.upf' \\) "
                        f"-exec cp -n {{}} {pseudo_q}/ \\;"
                    ),
                    modules=cfg.modules,
                    bin_dirs=bin_dirs,
                    label="Pseudo copy from generated pslibrary",
                )
                count = _upf_count(ssh)
                if count > 0:
                    copied_from = pslib_src

        if count <= 0:
            raise click.ClickException(
                "No UPF files found in TOPOSLAB_QE_PSEUDO_DIR and no source candidate produced UPFs. "
                "Set --qe-pseudo-source-dir (or TOPOSLAB_QE_PSEUDO_SOURCE_DIR) to a directory that contains UPF files, "
                "or provide TOPOSLAB_QE_SOURCE_DIR so init can build ld1.x and generate pslibrary UPFs."
            )

        if copied_from:
            click.echo(
                click.style(
                    f"  ✓ pseudo dir seeded from {copied_from}, UPF count={count}",
                    fg="green",
                )
            )
        else:
            click.echo(click.style(f"  ✓ pseudo dir ready, UPF count={count}", fg="green"))


def _prepare_cluster_python_setup(
    *,
    dry_run: bool,
    remote_python_executable: str = "python3",
    ensure_kwant: bool = True,
    ensure_tbmodels: bool = True,
    ensure_sisl: bool = True,
    ensure_berry: bool = True,
) -> None:
    if dry_run:
        click.echo("  (skipped in dry-run mode)")
        return

    from wtec.config.cluster import ClusterConfig
    from wtec.cluster.ssh import open_ssh
    from wtec.cluster.submit import JobManager

    try:
        cfg = ClusterConfig.from_env()
    except Exception as exc:
        click.echo(click.style(f"  WARNING: cluster config incomplete; skip remote python prep: {exc}", fg="yellow"))
        return

    click.echo(
        f"  Preparing cluster python via {remote_python_executable} "
        f"on {cfg.host}:{cfg.port}"
    )
    bin_dirs = _cluster_bin_dirs_from_env() or list(cfg.bin_dirs)
    with open_ssh(cfg) as ssh:
        jm = JobManager(ssh)
        jm.ensure_remote_commands(
            ["qsub", "qstat", "mpirun", remote_python_executable],
            modules=cfg.modules,
            bin_dirs=bin_dirs,
        )

        # Prepare Berry stack first; kwant build runs last to preserve ABI consistency.
        if ensure_berry:
            _prepare_remote_wannierberri(
                ssh,
                python_executable=remote_python_executable,
                modules=cfg.modules,
                bin_dirs=bin_dirs,
            )
        if ensure_tbmodels:
            _prepare_remote_tbmodels(
                ssh,
                python_executable=remote_python_executable,
                modules=cfg.modules,
                bin_dirs=bin_dirs,
            )
        if ensure_sisl:
            _prepare_remote_sisl(
                ssh,
                python_executable=remote_python_executable,
                modules=cfg.modules,
                bin_dirs=bin_dirs,
            )
        if ensure_kwant:
            _prepare_remote_kwant(
                ssh,
                python_executable=remote_python_executable,
                modules=cfg.modules,
                bin_dirs=bin_dirs,
            )

        checks = [
            ("mpi4py", True),
            ("tbmodels", ensure_tbmodels),
            ("sisl", ensure_sisl),
            ("kwant", ensure_kwant),
            ("wannierberri", ensure_berry),
            ("ray", ensure_berry),
        ]
        for module_name, enabled in checks:
            if not enabled:
                continue
            ver = _remote_module_version(
                ssh,
                python_executable=remote_python_executable,
                module_name=module_name,
                modules=cfg.modules,
                bin_dirs=bin_dirs,
            )
            if ver is None:
                raise click.ClickException(
                    f"Remote module import check failed after setup: {module_name}"
                )
            click.echo(click.style(f"  ✓ remote {module_name:<12} {ver}", fg="green"))


# ---------------------------------------------------------------------------
# wtec run preflight
# ---------------------------------------------------------------------------

def _maybe_reexec_in_init_venv() -> None:
    """Re-exec command under init venv python when current interpreter differs."""
    if os.environ.get("WTEC_AUTO_VENV_REEXEC") == "1":
        return
    if os.environ.get("WTEC_AUTO_VENV", "1").strip().lower() in {"0", "false", "no", "off"}:
        return

    state = _load_init_state()
    if not state:
        return
    venv_python = state.get("venv_python")
    if not isinstance(venv_python, str) or not venv_python.strip():
        return

    target = Path(venv_python).expanduser()
    if not target.exists():
        click.echo(
            click.style(
                f"[runtime] WARNING: init venv python not found, continuing with {sys.executable}: {target}",
                fg="yellow",
            )
        )
        return

    try:
        current = Path(sys.executable).expanduser().absolute()
        resolved_target = target.absolute()
    except Exception:
        return
    if current == resolved_target:
        return

    click.echo(click.style(f"[runtime] Auto-activating venv: {resolved_target}", fg="cyan"))
    env = dict(os.environ)
    env["WTEC_AUTO_VENV_REEXEC"] = "1"
    argv = [str(resolved_target), "-m", "wtec.cli", *sys.argv[1:]]
    os.execve(str(resolved_target), argv, env)


def _load_init_state() -> dict[str, Any] | None:
    state_path = Path.home() / ".wtec" / "init_state.json"
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _default_run_dir_from_init_state() -> str | None:
    state = _load_init_state()
    if not state:
        return None
    cwd = state.get("cwd")
    if not isinstance(cwd, str) or not cwd.strip():
        return None
    return cwd.strip()


def _normalize_stage(stage: str | None) -> str | None:
    if stage is None:
        return None
    s = stage.upper().replace("-", "_")
    if s == "DFT":
        return "DFT_NSCF"
    return s


def _checkpoint_file_for_cfg(cfg: dict) -> Path:
    return Path.home() / ".wtec" / "checkpoints" / f"{cfg.get('name', 'run')}.json"


def _checkpoint_hr_dat(cfg: dict) -> Path | None:
    cp = _checkpoint_file_for_cfg(cfg)
    if not cp.exists():
        return None
    try:
        data = json.loads(cp.read_text())
    except Exception:
        return None
    hr_str = data.get("outputs", {}).get("hr_dat")
    if not hr_str:
        return None
    p = Path(hr_str)
    return p if p.exists() else None


def _validate_positive_int_list(values, name: str) -> None:
    if values is None:
        return
    if not isinstance(values, list) or not values:
        raise click.UsageError(f"{name} must be a non-empty list of positive integers.")
    for v in values:
        try:
            iv = int(v)
        except Exception:
            raise click.UsageError(f"{name} contains non-integer value: {v!r}")
        if iv <= 0:
            raise click.UsageError(f"{name} contains non-positive value: {v!r}")


def _validate_nonnegative_float_list(values, name: str) -> None:
    if values is None:
        return
    if not isinstance(values, list) or not values:
        raise click.UsageError(f"{name} must be a non-empty list of non-negative numbers.")
    for v in values:
        try:
            fv = float(v)
        except Exception:
            raise click.UsageError(f"{name} contains non-numeric value: {v!r}")
        if fv < 0:
            raise click.UsageError(f"{name} contains negative value: {v!r}")


def _validate_positive_int_triplet(values, name: str) -> None:
    if values is None:
        return
    if not isinstance(values, list) or len(values) != 3:
        raise click.UsageError(f"{name} must be a list of 3 positive integers.")
    for v in values:
        try:
            iv = int(v)
        except Exception:
            raise click.UsageError(f"{name} contains non-integer value: {v!r}")
        if iv <= 0:
            raise click.UsageError(f"{name} contains non-positive value: {v!r}")


def _validate_axis_value(value, name: str) -> None:
    if value is None:
        return
    s = str(value).strip().lower()
    if s not in {"x", "y", "z"}:
        raise click.UsageError(f"{name} must be one of ['x','y','z'], got {value!r}")


def _run_preflight(cfg: dict, *, resume: bool, stage: str | None) -> None:
    stage_norm = _normalize_stage(stage)
    workspace = Path.home() / ".wtec"
    init_state = workspace / "init_state.json"
    if not init_state.exists():
        raise click.UsageError(
            "wtec init has not completed in this environment. Run `wtec init` first."
        )

    if not isinstance(cfg, dict):
        raise click.UsageError("Run config must be a JSON object.")

    # Optional transport-only fast path
    hr_cfg = cfg.get("hr_dat_path")
    hr_path: Path | None = None
    if hr_cfg:
        hr_path = Path(str(hr_cfg))
        if not hr_path.exists():
            raise click.UsageError(f"Configured hr_dat_path does not exist: {hr_path}")
        if hr_path.stat().st_size == 0:
            raise click.UsageError(f"Configured hr_dat_path is empty: {hr_path}")
    else:
        hr_path = _checkpoint_hr_dat(cfg)

    transport_only = (
        not resume and stage_norm in {"TRANSPORT", "ANALYSIS"} and hr_path is not None
    )

    # Structural/material requirements
    require_structure = False
    require_material = False
    if not transport_only:
        if stage_norm == "INIT":
            require_structure = False
            require_material = False
        elif stage_norm == "STRUCTURE":
            require_structure = True
            require_material = False
        else:
            require_structure = True
            require_material = True

    _normalize_dft_track_config(cfg)

    material = cfg.get("material")
    structure = cfg.get("structure_file")
    run_profile = str(cfg.get("run_profile", "strict")).strip().lower() or "strict"
    if run_profile not in {"strict", "smoke"}:
        raise click.UsageError("run_profile must be one of ['strict','smoke'].")
    dft_mode_raw = cfg.get("dft_mode")
    if dft_mode_raw is None and isinstance(cfg.get("dft"), dict):
        dft_mode_raw = cfg["dft"].get("mode")
    dft_mode = str(dft_mode_raw or "legacy_single").strip().lower() or "legacy_single"
    if dft_mode not in {"legacy_single", "hybrid_qe_ref_siesta_variants", "dual_family"}:
        raise click.UsageError(
            "dft_mode must be one of ['legacy_single','hybrid_qe_ref_siesta_variants','dual_family']."
        )
    dft_engine = str(cfg.get("dft_pes_engine", cfg.get("dft_engine", "qe"))).strip().lower() or "qe"
    variant_dft_engine = str(
        cfg.get(
            "dft_lcao_engine",
            cfg.get("topology_variant_dft_engine", dft_engine),
        )
    ).strip().lower() or dft_engine

    if dft_mode == "legacy_single":
        if dft_engine not in {"siesta", "qe"}:
            raise click.UsageError(
                "legacy_single mode currently supports dft_engine in ['qe','siesta']."
            )
        if variant_dft_engine not in {"siesta", "qe"}:
            raise click.UsageError(
                "legacy_single mode currently supports topology.variant_dft_engine in ['qe','siesta']."
            )
    elif dft_mode == "hybrid_qe_ref_siesta_variants":
        if dft_engine != "qe":
            raise click.UsageError("hybrid mode requires QE as dft reference engine.")
        if variant_dft_engine != "siesta":
            raise click.UsageError("hybrid mode requires topology.variant_dft_engine='siesta'.")
    else:  # dual_family
        if dft_engine not in {"qe", "vasp"}:
            raise click.UsageError(
                "dual_family mode requires dft_pes_engine in ['qe','vasp']."
            )
        if variant_dft_engine not in {"siesta", "abacus"}:
            raise click.UsageError(
                "dual_family mode requires dft_lcao_engine in ['siesta','abacus']."
            )
        pes_reference_structure = str(
            cfg.get("dft_pes_reference_structure_file", "")
        ).strip()
        if not transport_only:
            needs_dft = (
                resume
                or stage_norm is None
                or stage_norm in {"DFT_SCF", "DFT_NSCF", "WANNIER90", "TRANSPORT", "ANALYSIS"}
            )
            if needs_dft:
                if not pes_reference_structure:
                    raise click.UsageError(
                        "dual_family mode requires dft.tracks.pes_reference.structure_file "
                        "(runtime key: dft_pes_reference_structure_file) for small PES reference."
                    )
                pes_path = Path(pes_reference_structure).expanduser()
                if not pes_path.exists():
                    raise click.UsageError(
                        f"dft_pes_reference_structure_file does not exist: {pes_path}"
                    )
                if pes_path.stat().st_size == 0:
                    raise click.UsageError(
                        f"dft_pes_reference_structure_file is empty: {pes_path}"
                    )

    cfg["dft_engine"] = dft_engine
    cfg["dft_pes_engine"] = dft_engine
    cfg["topology_variant_dft_engine"] = variant_dft_engine
    cfg["dft_lcao_engine"] = variant_dft_engine
    dft_reuse_mode_raw = cfg.get("dft_reuse_mode")
    if dft_reuse_mode_raw is None and isinstance(cfg.get("dft"), dict):
        dft_reuse_mode_raw = cfg["dft"].get("reuse_mode")
    dft_reuse_mode = str(dft_reuse_mode_raw or "none").strip().lower() or "none"
    if dft_reuse_mode not in {"none", "pristine-only", "all"}:
        raise click.UsageError("dft.reuse_mode must be one of ['none','pristine-only','all'].")
    if dft_reuse_mode == "none" and hr_cfg:
        raise click.UsageError(
            "dft.reuse_mode='none' forbids explicit hr_dat_path in strict workflow. "
            "Clear dft.hr_dat_path or set dft.reuse_mode to 'pristine-only'/'all'."
        )
    reuse_policy_raw = cfg.get("dft_reference_reuse_policy")
    if reuse_policy_raw is None and isinstance(cfg.get("dft"), dict):
        dref = cfg["dft"].get("reference")
        if isinstance(dref, dict):
            reuse_policy_raw = dref.get("reuse_policy")
    reuse_policy = str(reuse_policy_raw or "strict_hash").strip().lower() or "strict_hash"
    if reuse_policy not in {"strict_hash", "timestamp_only"}:
        raise click.UsageError(
            "dft.reference.reuse_policy must be one of ['strict_hash','timestamp_only']."
        )

    vasp_cfg = cfg.get("dft_vasp", {})
    if not isinstance(vasp_cfg, dict):
        vasp_cfg = {}
    if isinstance(cfg.get("dft"), dict) and isinstance(cfg["dft"].get("vasp"), dict):
        merged_vasp = dict(cfg["dft"]["vasp"])
        merged_vasp.update(vasp_cfg)
        vasp_cfg = merged_vasp
    if dft_engine == "vasp":
        vasp_exe = str(vasp_cfg.get("executable", "vasp_std")).strip()
        if not vasp_exe:
            raise click.UsageError("dft.vasp.executable must be a non-empty string.")
        for key in ("encut_ev", "ediff", "sigma"):
            if key in vasp_cfg:
                try:
                    float(vasp_cfg.get(key))
                except Exception as exc:
                    raise click.UsageError(f"dft.vasp.{key} must be numeric: {exc}") from exc
        if "ismear" in vasp_cfg:
            try:
                int(vasp_cfg.get("ismear"))
            except Exception as exc:
                raise click.UsageError(f"dft.vasp.ismear must be integer: {exc}") from exc
        if "disable_symmetry" in vasp_cfg and not isinstance(vasp_cfg.get("disable_symmetry"), bool):
            raise click.UsageError("dft.vasp.disable_symmetry must be a boolean.")

    abacus_cfg = cfg.get("dft_abacus", {})
    if not isinstance(abacus_cfg, dict):
        abacus_cfg = {}
    if isinstance(cfg.get("dft"), dict) and isinstance(cfg["dft"].get("abacus"), dict):
        merged_abacus = dict(cfg["dft"]["abacus"])
        merged_abacus.update(abacus_cfg)
        abacus_cfg = merged_abacus
    if variant_dft_engine == "abacus":
        abacus_exe = str(abacus_cfg.get("executable", "abacus")).strip()
        if not abacus_exe:
            raise click.UsageError("dft.abacus.executable must be a non-empty string.")
        basis_type = str(abacus_cfg.get("basis_type", "lcao")).strip().lower() or "lcao"
        if basis_type not in {"lcao"}:
            raise click.UsageError("dft.abacus.basis_type currently supports only 'lcao'.")
        ks_solver = str(abacus_cfg.get("ks_solver", "genelpa")).strip()
        if not ks_solver:
            raise click.UsageError("dft.abacus.ks_solver must be a non-empty string.")
    cfg["dft_vasp"] = vasp_cfg
    cfg["dft_abacus"] = abacus_cfg

    siesta_cfg = cfg.get("dft_siesta", {})
    if not isinstance(siesta_cfg, dict):
        siesta_cfg = {}
    if isinstance(cfg.get("dft"), dict) and isinstance(cfg["dft"].get("siesta"), dict):
        merged = dict(cfg["dft"]["siesta"])
        merged.update(siesta_cfg)
        siesta_cfg = merged
    if dft_engine == "siesta" or variant_dft_engine == "siesta":
        iface = str(siesta_cfg.get("wannier_interface", "sisl")).strip().lower() or "sisl"
        if iface != "sisl":
            raise click.UsageError(
                "dft.siesta.wannier_interface currently supports only 'sisl' in this workflow."
            )
    if siesta_cfg:
        for key in ("mpi_np_scf", "mpi_np_nscf", "mpi_np_wannier"):
            if key in siesta_cfg:
                try:
                    val = int(siesta_cfg.get(key))
                except Exception as exc:
                    raise click.UsageError(f"dft.siesta.{key} must be integer: {exc}") from exc
                if val < 0:
                    raise click.UsageError(f"dft.siesta.{key} must be >= 0 (0 means auto).")
        for key in ("omp_threads_scf", "omp_threads_nscf", "omp_threads_wannier"):
            if key in siesta_cfg:
                try:
                    val = int(siesta_cfg.get(key))
                except Exception as exc:
                    raise click.UsageError(f"dft.siesta.{key} must be integer: {exc}") from exc
                if val < 0:
                    raise click.UsageError(f"dft.siesta.{key} must be >= 0 (0 means auto).")
        if "dm_mixing_weight" in siesta_cfg:
            try:
                dmw = float(siesta_cfg.get("dm_mixing_weight"))
            except Exception as exc:
                raise click.UsageError(
                    f"dft.siesta.dm_mixing_weight must be numeric: {exc}"
                ) from exc
            if dmw <= 0.0 or dmw > 1.0:
                raise click.UsageError("dft.siesta.dm_mixing_weight must be in (0, 1].")
        if "dm_number_pulay" in siesta_cfg:
            try:
                dmp = int(siesta_cfg.get("dm_number_pulay"))
            except Exception as exc:
                raise click.UsageError(
                    f"dft.siesta.dm_number_pulay must be integer: {exc}"
                ) from exc
            if dmp <= 0:
                raise click.UsageError("dft.siesta.dm_number_pulay must be > 0.")
        if "electronic_temperature_k" in siesta_cfg:
            try:
                etk = float(siesta_cfg.get("electronic_temperature_k"))
            except Exception as exc:
                raise click.UsageError(
                    f"dft.siesta.electronic_temperature_k must be numeric: {exc}"
                ) from exc
            if etk <= 0.0:
                raise click.UsageError("dft.siesta.electronic_temperature_k must be > 0.")
        if "max_scf_iterations" in siesta_cfg:
            try:
                msi = int(siesta_cfg.get("max_scf_iterations"))
            except Exception as exc:
                raise click.UsageError(
                    f"dft.siesta.max_scf_iterations must be integer: {exc}"
                ) from exc
            if msi <= 0:
                raise click.UsageError("dft.siesta.max_scf_iterations must be > 0.")
        fdefs = siesta_cfg.get("factorization_defaults")
        if fdefs is not None:
            if not isinstance(fdefs, dict):
                raise click.UsageError("dft.siesta.factorization_defaults must be an object.")
            allowed = {
                "mpi_np_scf",
                "mpi_np_nscf",
                "mpi_np_wannier",
                "omp_threads_scf",
                "omp_threads_nscf",
                "omp_threads_wannier",
            }
            for scope, prof in fdefs.items():
                if not isinstance(prof, dict):
                    raise click.UsageError(
                        f"dft.siesta.factorization_defaults[{scope!r}] must be an object."
                    )
                for pkey, pval in prof.items():
                    if str(pkey) not in allowed:
                        raise click.UsageError(
                            f"dft.siesta.factorization_defaults[{scope!r}] has unsupported key {pkey!r}."
                        )
                    try:
                        iv = int(pval)
                    except Exception as exc:
                        raise click.UsageError(
                            f"dft.siesta.factorization_defaults[{scope!r}].{pkey} must be integer: {exc}"
                        ) from exc
                    if iv <= 0:
                        raise click.UsageError(
                            f"dft.siesta.factorization_defaults[{scope!r}].{pkey} must be > 0."
                        )
    cfg["dft_siesta"] = siesta_cfg

    disp_cfg = cfg.get("dft_dispersion", {})
    if not isinstance(disp_cfg, dict):
        disp_cfg = {}
    if isinstance(cfg.get("dft"), dict) and isinstance(cfg["dft"].get("dispersion"), dict):
        merged_disp = dict(cfg["dft"]["dispersion"])
        merged_disp.update(disp_cfg)
        disp_cfg = merged_disp
    if disp_cfg:
        method = str(disp_cfg.get("method", "d3")).strip().lower() or "d3"
        if method not in {"d3", "none"}:
            raise click.UsageError("dft.dispersion.method must be one of ['d3','none'].")
        if "enabled" in disp_cfg and not isinstance(disp_cfg.get("enabled"), bool):
            raise click.UsageError("dft.dispersion.enabled must be a boolean.")
        if method == "d3":
            if str(disp_cfg.get("qe_vdw_corr", "grimme-d3")).strip().lower() not in {"grimme-d3"}:
                raise click.UsageError("dft.dispersion.qe_vdw_corr currently supports only 'grimme-d3'.")
            try:
                int(disp_cfg.get("qe_dftd3_version", 4))
            except Exception as exc:
                raise click.UsageError(f"dft.dispersion.qe_dftd3_version must be integer: {exc}") from exc
            if "qe_dftd3_threebody" in disp_cfg and not isinstance(
                disp_cfg.get("qe_dftd3_threebody"), bool
            ):
                raise click.UsageError("dft.dispersion.qe_dftd3_threebody must be a boolean.")
            if "siesta_dftd3_use_xc_defaults" in disp_cfg and not isinstance(
                disp_cfg.get("siesta_dftd3_use_xc_defaults"), bool
            ):
                raise click.UsageError(
                    "dft.dispersion.siesta_dftd3_use_xc_defaults must be a boolean."
                )
            if "siesta_dftd3_use_xc_functional" in disp_cfg and not isinstance(
                disp_cfg.get("siesta_dftd3_use_xc_functional"), bool
            ):
                raise click.UsageError(
                    "dft.dispersion.siesta_dftd3_use_xc_functional must be a boolean."
                )

    anchor_cfg = cfg.get("dft_anchor_transfer", {})
    if not isinstance(anchor_cfg, dict):
        anchor_cfg = {}
    if isinstance(cfg.get("dft"), dict) and isinstance(cfg["dft"].get("anchor_transfer"), dict):
        merged_anchor = dict(cfg["dft"]["anchor_transfer"])
        merged_anchor.update(anchor_cfg)
        anchor_cfg = merged_anchor
    if anchor_cfg:
        enabled = bool(anchor_cfg.get("enabled", dft_mode == "dual_family"))
        mode_val = str(anchor_cfg.get("mode", "delta_h")).strip().lower() or "delta_h"
        if mode_val != "delta_h":
            raise click.UsageError("dft.anchor_transfer.mode must be 'delta_h'.")
        basis_policy = str(anchor_cfg.get("basis_policy", "strict_same_basis")).strip() or "strict_same_basis"
        if basis_policy != "strict_same_basis":
            raise click.UsageError("dft.anchor_transfer.basis_policy must be 'strict_same_basis'.")
        scope = str(anchor_cfg.get("scope", "onsite_plus_first_shell")).strip() or "onsite_plus_first_shell"
        if scope != "onsite_plus_first_shell":
            raise click.UsageError("dft.anchor_transfer.scope must be 'onsite_plus_first_shell'.")
        _validate_positive_int_triplet(anchor_cfg.get("fit_kmesh"), "dft.anchor_transfer.fit_kmesh")
        if "fit_window_ev" in anchor_cfg and float(anchor_cfg.get("fit_window_ev")) <= 0:
            raise click.UsageError("dft.anchor_transfer.fit_window_ev must be > 0.")
        if int(anchor_cfg.get("alpha_grid_points", 81)) < 2:
            raise click.UsageError("dft.anchor_transfer.alpha_grid_points must be >= 2.")
        if int(anchor_cfg.get("max_retries", 5)) <= 0:
            raise click.UsageError("dft.anchor_transfer.max_retries must be > 0.")
        if int(anchor_cfg.get("retry_kmesh_step", 2)) < 0:
            raise click.UsageError("dft.anchor_transfer.retry_kmesh_step must be >= 0.")
        if float(anchor_cfg.get("retry_window_step_ev", 0.5)) < 0:
            raise click.UsageError("dft.anchor_transfer.retry_window_step_ev must be >= 0.")
        if enabled and dft_mode != "dual_family":
            click.echo(
                click.style(
                    "[preflight] warning: dft.anchor_transfer is enabled but dft_mode is not dual_family; it will be skipped.",
                    fg="yellow",
                )
            )
    cfg["dft_anchor_transfer"] = anchor_cfg
    if require_material and not material:
        raise click.UsageError("Missing required config key: 'material'")
    if require_structure:
        if not structure:
            raise click.UsageError(
                "Missing required config key: 'structure_file'.\n"
                "Generate slab first:\n"
                "  wtec slab-gen wtec_slab_template.toml\n"
                "Inspect result:\n"
                "  wtec slab slab_outputs/<project>.generated.cif\n"
                "Then set that CIF path in your run JSON as \"structure_file\"."
            )
        structure_path = Path(str(structure))
        if not structure_path.exists():
            raise click.UsageError(
                f"structure_file does not exist: {structure_path}\n"
                "If you have not generated it yet, run:\n"
                "  wtec slab-gen wtec_slab_template.toml"
            )
        if structure_path.stat().st_size == 0:
            raise click.UsageError(f"structure_file is empty: {structure_path}")

    # General numeric/list validations
    if "n_ensemble" in cfg and int(cfg["n_ensemble"]) <= 0:
        raise click.UsageError("n_ensemble must be > 0")
    if "n_jobs" in cfg and int(cfg["n_jobs"]) <= 0:
        raise click.UsageError("n_jobs must be > 0")
    if "n_nodes" in cfg and int(cfg["n_nodes"]) <= 0:
        raise click.UsageError("n_nodes must be > 0")
    _validate_positive_int_list(cfg.get("thicknesses"), "thicknesses")
    _validate_nonnegative_float_list(cfg.get("disorder_strengths"), "disorder_strengths")
    _validate_positive_int_list(cfg.get("mfp_lengths"), "mfp_lengths")
    _validate_positive_int_triplet(cfg.get("kpoints_scf"), "kpoints_scf")
    _validate_positive_int_triplet(cfg.get("kpoints_nscf"), "kpoints_nscf")
    _validate_axis_value(cfg.get("transport_axis"), "transport_axis")
    _validate_axis_value(cfg.get("thickness_axis"), "thickness_axis")
    if "carrier_density_m3" in cfg and cfg.get("carrier_density_m3") is not None:
        if float(cfg.get("carrier_density_m3")) <= 0:
            raise click.UsageError("carrier_density_m3 must be > 0 when provided.")
    if "fermi_velocity_m_per_s" in cfg and cfg.get("fermi_velocity_m_per_s") is not None:
        if float(cfg.get("fermi_velocity_m_per_s")) <= 0:
            raise click.UsageError("fermi_velocity_m_per_s must be > 0 when provided.")
    if "transport_n_layers_x" in cfg and int(cfg["transport_n_layers_x"]) < 2:
        raise click.UsageError("transport_n_layers_x must be >= 2 for valid Kwant lead attachment.")
    if "transport_n_layers_y" in cfg and int(cfg["transport_n_layers_y"]) <= 0:
        raise click.UsageError("transport_n_layers_y must be > 0")
    if "transport_backend" in cfg:
        tb = str(cfg["transport_backend"]).strip().lower()
        if tb not in {"qsub", "local"}:
            raise click.UsageError("transport_backend must be one of ['qsub','local'].")
    if "transport_policy" in cfg:
        tp = str(cfg["transport_policy"]).strip().lower()
        if tp not in {"single_track", "dual_track_compare"}:
            raise click.UsageError(
                "transport_policy must be one of ['single_track','dual_track_compare']."
            )
    if "transport_strict_qsub" in cfg and not isinstance(cfg["transport_strict_qsub"], bool):
        raise click.UsageError("transport_strict_qsub must be a boolean.")
    if "transport_mpi_np" in cfg and int(cfg["transport_mpi_np"]) < 0:
        raise click.UsageError("transport_mpi_np must be >= 0 (0 means auto).")
    if "transport_threads" in cfg and int(cfg["transport_threads"]) < 0:
        raise click.UsageError("transport_threads must be >= 0 (0 means auto).")
    transport_autotune_cfg = cfg.get("transport_autotune")
    if transport_autotune_cfg is not None:
        if not isinstance(transport_autotune_cfg, dict):
            raise click.UsageError("transport_autotune must be a JSON object when provided.")
        scope = str(transport_autotune_cfg.get("scope", "per_queue")).strip().lower() or "per_queue"
        if scope != "per_queue":
            raise click.UsageError("transport_autotune.scope currently supports only 'per_queue'.")
        if "enabled" in transport_autotune_cfg and not isinstance(
            transport_autotune_cfg.get("enabled"), bool
        ):
            raise click.UsageError("transport_autotune.enabled must be a boolean.")
        if "benchmark_n_ensemble" in transport_autotune_cfg and int(
            transport_autotune_cfg.get("benchmark_n_ensemble")
        ) <= 0:
            raise click.UsageError("transport_autotune.benchmark_n_ensemble must be > 0.")
        profiles = transport_autotune_cfg.get("profiles")
        if profiles is not None:
            if not isinstance(profiles, list) or not profiles:
                raise click.UsageError("transport_autotune.profiles must be a non-empty list.")
            for idx, prof in enumerate(profiles):
                if not isinstance(prof, dict):
                    raise click.UsageError(f"transport_autotune.profiles[{idx}] must be an object.")
                if int(prof.get("mpi_np", 1)) <= 0:
                    raise click.UsageError(
                        f"transport_autotune.profiles[{idx}].mpi_np must be > 0."
                    )
                thr = prof.get("threads", "all")
                if isinstance(thr, int):
                    if int(thr) <= 0:
                        raise click.UsageError(
                            f"transport_autotune.profiles[{idx}].threads must be > 0."
                        )
                elif isinstance(thr, str):
                    if not str(thr).strip():
                        raise click.UsageError(
                            f"transport_autotune.profiles[{idx}].threads cannot be empty."
                        )
                else:
                    raise click.UsageError(
                        f"transport_autotune.profiles[{idx}].threads must be int or string."
                    )

    topo_cfg = cfg.get("topology")
    if topo_cfg is not None:
        if not isinstance(topo_cfg, dict):
            raise click.UsageError("topology must be a JSON object when provided.")
        backend = topo_cfg.get("backend")
        if backend is not None:
            b = str(backend).strip().lower()
            if b not in {"qsub", "local"}:
                raise click.UsageError("topology.backend must be one of ['qsub','local'].")
        exec_mode = topo_cfg.get("execution_mode")
        if exec_mode is not None:
            em = str(exec_mode).strip().lower()
            if em not in {"single_batch", "batch", "qsub_batch", "per_point", "per_point_qsub", "point"}:
                raise click.UsageError(
                    "topology.execution_mode must be one of "
                    "['single_batch','batch','qsub_batch','per_point','per_point_qsub','point']."
                )
        if "strict_qsub" in topo_cfg and not isinstance(topo_cfg["strict_qsub"], bool):
            raise click.UsageError("topology.strict_qsub must be a boolean.")
        if "failure_policy" in topo_cfg:
            fp = str(topo_cfg["failure_policy"]).strip().lower()
            if fp not in {"strict", "rescale"}:
                raise click.UsageError("topology.failure_policy must be one of ['strict','rescale'].")
        if "max_concurrent_point_jobs" in topo_cfg and int(topo_cfg["max_concurrent_point_jobs"]) <= 0:
            raise click.UsageError("topology.max_concurrent_point_jobs must be > 0.")
        if "max_concurrent_variant_dft_jobs" in topo_cfg and int(topo_cfg["max_concurrent_variant_dft_jobs"]) <= 0:
            raise click.UsageError("topology.max_concurrent_variant_dft_jobs must be > 0.")
        if "arc_engine" in topo_cfg:
            ae = str(topo_cfg["arc_engine"]).strip().lower()
            if ae not in {"wannierberri", "wannierberri_strict", "wb_strict", "kwant", "siesta_slab_ldos"}:
                raise click.UsageError(
                    "topology.arc_engine must be one of "
                    "['wannierberri','wannierberri_strict','wb_strict','kwant','siesta_slab_ldos']."
                )
        if "arc_allow_proxy_fallback" in topo_cfg and not isinstance(
            topo_cfg.get("arc_allow_proxy_fallback"), bool
        ):
            raise click.UsageError("topology.arc_allow_proxy_fallback must be a boolean.")
        if "siesta_slab_ldos_autogen" in topo_cfg:
            mode = str(topo_cfg.get("siesta_slab_ldos_autogen", "")).strip().lower()
            if mode not in {"kwant_proxy", "none", "off", "disabled", "false", "no"}:
                raise click.UsageError(
                    "topology.siesta_slab_ldos_autogen must be one of "
                    "['kwant_proxy','none','off','disabled','false','no']."
                )
        if "node_method" in topo_cfg:
            nm = str(topo_cfg["node_method"]).strip().lower()
            if nm not in {"proxy", "berry_flux", "wannierberri_flux"}:
                raise click.UsageError(
                    "topology.node_method must be one of ['proxy','berry_flux','wannierberri_flux']."
                )
        if "hr_scope" in topo_cfg:
            hs = str(topo_cfg["hr_scope"]).strip().lower()
            if hs != "per_variant":
                raise click.UsageError("topology.hr_scope must be 'per_variant'.")
        if "variant_dft_engine" in topo_cfg:
            vde = str(topo_cfg["variant_dft_engine"]).strip().lower()
            if vde not in {"qe", "siesta", "abacus"}:
                raise click.UsageError(
                    "topology.variant_dft_engine must be one of ['qe','siesta','abacus']."
                )
        if "fermi_ev" in topo_cfg and topo_cfg["fermi_ev"] is not None:
            try:
                float(topo_cfg["fermi_ev"])
            except Exception as exc:
                raise click.UsageError(f"topology.fermi_ev must be numeric: {exc}") from exc
        _validate_axis_value(topo_cfg.get("transport_axis_primary"), "topology.transport_axis_primary")
        _validate_axis_value(topo_cfg.get("transport_axis_aux"), "topology.transport_axis_aux")
        if "n_layers_y" in topo_cfg and int(topo_cfg["n_layers_y"]) <= 0:
            raise click.UsageError("topology.n_layers_y must be > 0")
        if "n_layers_x" in topo_cfg and int(topo_cfg["n_layers_x"]) < 2:
            raise click.UsageError("topology.n_layers_x must be >= 2 for valid Kwant lead attachment.")
        _validate_positive_int_triplet(topo_cfg.get("coarse_kmesh"), "topology.coarse_kmesh")
        _validate_positive_int_triplet(topo_cfg.get("refine_kmesh"), "topology.refine_kmesh")
        if "newton_max_iter" in topo_cfg and int(topo_cfg["newton_max_iter"]) <= 0:
            raise click.UsageError("topology.newton_max_iter must be > 0")
        score = topo_cfg.get("score")
        if score is not None and not isinstance(score, dict):
            raise click.UsageError("topology.score must be an object when provided.")
        if isinstance(score, dict) and "w_topo" in score:
            w = float(score["w_topo"])
            if w < 0 or w > 1:
                raise click.UsageError("topology.score.w_topo must be within [0,1].")
        if isinstance(score, dict) and "w_transport" in score:
            wt = float(score["w_transport"])
            if wt < 0 or wt > 1:
                raise click.UsageError("topology.score.w_transport must be within [0,1].")
        transport_probe_cfg = topo_cfg.get("transport_probe")
        if transport_probe_cfg is not None:
            if not isinstance(transport_probe_cfg, dict):
                raise click.UsageError("topology.transport_probe must be an object when provided.")
            if "enabled" in transport_probe_cfg and not isinstance(
                transport_probe_cfg.get("enabled"), bool
            ):
                raise click.UsageError("topology.transport_probe.enabled must be a boolean.")
            if "n_ensemble" in transport_probe_cfg and int(
                transport_probe_cfg.get("n_ensemble")
            ) <= 0:
                raise click.UsageError("topology.transport_probe.n_ensemble must be > 0.")
            if "disorder_strength" in transport_probe_cfg and float(
                transport_probe_cfg.get("disorder_strength")
            ) < 0:
                raise click.UsageError(
                    "topology.transport_probe.disorder_strength must be >= 0."
                )
            if "energy_shift_ev" in transport_probe_cfg:
                try:
                    float(transport_probe_cfg.get("energy_shift_ev"))
                except Exception as exc:
                    raise click.UsageError(
                        f"topology.transport_probe.energy_shift_ev must be numeric: {exc}"
                    ) from exc
            _validate_axis_value(
                transport_probe_cfg.get("thickness_axis"),
                "topology.transport_probe.thickness_axis",
            )
        hr_grid = topo_cfg.get("hr_grid")
        if hr_grid is not None:
            if not isinstance(hr_grid, dict):
                raise click.UsageError("topology.hr_grid must be an object when provided.")
            if "reference_thickness_uc" in hr_grid and hr_grid["reference_thickness_uc"] is not None:
                if int(hr_grid["reference_thickness_uc"]) <= 0:
                    raise click.UsageError("topology.hr_grid.reference_thickness_uc must be > 0")
            if "max_parallel_hr_points" in hr_grid and int(hr_grid["max_parallel_hr_points"]) <= 0:
                raise click.UsageError("topology.hr_grid.max_parallel_hr_points must be > 0")
            if "thickness_mapping" in hr_grid:
                tm = str(hr_grid["thickness_mapping"]).strip().lower()
                if tm not in {"middle_layer_scale"}:
                    raise click.UsageError("topology.hr_grid.thickness_mapping currently supports only 'middle_layer_scale'.")
        point_prof = topo_cfg.get("point_resource_profile")
        if point_prof is not None:
            if not isinstance(point_prof, dict):
                raise click.UsageError("topology.point_resource_profile must be an object when provided.")
            for key in ("node_phase_mpi_np", "node_phase_threads", "arc_phase_mpi_np", "arc_phase_threads"):
                val = point_prof.get(key)
                if val is None:
                    continue
                if int(val) <= 0:
                    raise click.UsageError(f"topology.point_resource_profile.{key} must be > 0.")
        tiering_cfg = topo_cfg.get("tiering")
        if tiering_cfg is not None:
            if not isinstance(tiering_cfg, dict):
                raise click.UsageError("topology.tiering must be an object when provided.")
            mode = str(tiering_cfg.get("mode", "single")).strip().lower() or "single"
            if mode not in {"single", "two_tier"}:
                raise click.UsageError("topology.tiering.mode must be one of ['single','two_tier'].")
            if "refine_top_k_per_thickness" in tiering_cfg and int(
                tiering_cfg.get("refine_top_k_per_thickness")
            ) <= 0:
                raise click.UsageError(
                    "topology.tiering.refine_top_k_per_thickness must be > 0."
                )
            if "always_include_pristine" in tiering_cfg and not isinstance(
                tiering_cfg.get("always_include_pristine"), bool
            ):
                raise click.UsageError("topology.tiering.always_include_pristine must be a boolean.")
            if "selection_metric" in tiering_cfg:
                metric = str(tiering_cfg.get("selection_metric", "S_total")).strip()
                if metric != "S_total":
                    raise click.UsageError(
                        "topology.tiering.selection_metric currently supports only 'S_total'."
                    )
            screen_cfg = tiering_cfg.get("screen")
            if screen_cfg is not None:
                if not isinstance(screen_cfg, dict):
                    raise click.UsageError("topology.tiering.screen must be an object when provided.")
                if "arc_engine" in screen_cfg:
                    ae = str(screen_cfg.get("arc_engine", "")).strip().lower()
                    if ae not in {"wannierberri", "wannierberri_strict", "wb_strict", "kwant", "siesta_slab_ldos"}:
                        raise click.UsageError(
                            "topology.tiering.screen.arc_engine must be one of "
                            "['wannierberri','wannierberri_strict','wb_strict','kwant','siesta_slab_ldos']."
                        )
                if "node_method" in screen_cfg:
                    nm = str(screen_cfg.get("node_method", "")).strip().lower()
                    if nm not in {"proxy", "berry_flux", "wannierberri_flux"}:
                        raise click.UsageError(
                            "topology.tiering.screen.node_method must be one of "
                            "['proxy','berry_flux','wannierberri_flux']."
                        )
                _validate_positive_int_triplet(
                    screen_cfg.get("coarse_kmesh"),
                    "topology.tiering.screen.coarse_kmesh",
                )
                _validate_positive_int_triplet(
                    screen_cfg.get("refine_kmesh"),
                    "topology.tiering.screen.refine_kmesh",
                )

    # Strict profile guardrails for scientific runs.
    if run_profile == "strict":
        if dft_reuse_mode != "none":
            raise click.UsageError(
                "strict run_profile requires dft.reuse_mode='none' for fresh DFT provenance."
            )
        if dft_mode == "dual_family":
            if not bool((anchor_cfg or {}).get("enabled", True)):
                raise click.UsageError(
                    "strict dual_family run_profile requires dft.anchor_transfer.enabled=true."
                )
        thicknesses = cfg.get("thicknesses")
        mfp_lengths = cfg.get("mfp_lengths")
        n_ensemble = int(cfg.get("n_ensemble", 0))
        if not isinstance(thicknesses, list) or len(thicknesses) < 5:
            raise click.UsageError(
                "strict run_profile requires at least 5 transport thickness points."
            )
        if not isinstance(mfp_lengths, list) or len(mfp_lengths) < 7:
            raise click.UsageError(
                "strict run_profile requires at least 7 mfp_lengths points."
            )
        if n_ensemble < 30:
            raise click.UsageError(
                "strict run_profile requires n_ensemble >= 30."
            )
        if str(cfg.get("transport_backend", "qsub")).strip().lower() != "qsub":
            raise click.UsageError("strict run_profile requires transport.backend='qsub'.")
        if topo_cfg is not None:
            if bool(topo_cfg.get("caveat_reuse_global_hr_dat", False)):
                raise click.UsageError(
                    "strict run_profile forbids topology.caveat_reuse_global_hr_dat=true."
                )
            if bool(topo_cfg.get("arc_allow_proxy_fallback", False)):
                raise click.UsageError(
                    "strict run_profile forbids topology.arc_allow_proxy_fallback=true."
                )
            arc_engine = str(topo_cfg.get("arc_engine", "siesta_slab_ldos")).strip().lower()
            node_method = str(topo_cfg.get("node_method", "wannierberri_flux")).strip().lower()
            if node_method == "proxy":
                raise click.UsageError("strict run_profile forbids topology.node_method='proxy'.")
            if arc_engine in {"kwant", "wannierberri"}:
                raise click.UsageError(
                    "strict run_profile requires non-proxy arc engine "
                    "('siesta_slab_ldos' or strict WannierBerri mode)."
                )
            if str(topo_cfg.get("backend", "qsub")).strip().lower() != "qsub":
                raise click.UsageError("strict run_profile requires topology.backend='qsub'.")
            tiering_cfg = topo_cfg.get("tiering")
            if isinstance(tiering_cfg, dict):
                screen_cfg = tiering_cfg.get("screen")
                if isinstance(screen_cfg, dict):
                    screen_node = str(screen_cfg.get("node_method", "")).strip().lower()
                    screen_arc = str(screen_cfg.get("arc_engine", "")).strip().lower()
                    if screen_node == "proxy":
                        raise click.UsageError(
                            "strict run_profile forbids topology.tiering.screen.node_method='proxy'."
                        )
                    if screen_arc == "kwant":
                        raise click.UsageError(
                            "strict run_profile forbids topology.tiering.screen.arc_engine='kwant'."
                        )

    # Material-class-aware k-mesh checks (independent of cluster backend).
    if material:
        from wtec.config.materials import get_material

        try:
            preset_local = get_material(material)
        except Exception as exc:
            raise click.UsageError(str(exc))
        k_scf = tuple(int(v) for v in cfg.get("kpoints_scf", [8, 8, 8]))
        k_nscf = tuple(int(v) for v in cfg.get("kpoints_nscf", [12, 12, 12]))
        if getattr(preset_local, "material_class", "generic").lower() == "weyl":
            if any(int(v) < 4 for v in k_nscf):
                raise click.UsageError(
                    f"nscf kmesh {list(k_nscf)} is insufficient for Weyl material {preset_local.name}. "
                    "All NSCF axes must be >= 4."
                )
        min_scf = tuple(int(v) for v in getattr(preset_local, "min_kmesh_scf", (1, 1, 1)))
        min_nscf = tuple(int(v) for v in getattr(preset_local, "min_kmesh_nscf", (1, 1, 1)))
        if any(k_scf[i] < min_scf[i] for i in range(3)):
            raise click.UsageError(
                f"kpoints_scf {list(k_scf)} is below material minimum {list(min_scf)} for {preset_local.name}."
            )
        if any(k_nscf[i] < min_nscf[i] for i in range(3)):
            raise click.UsageError(
                f"kpoints_nscf {list(k_nscf)} is below material minimum {list(min_nscf)} for {preset_local.name}."
            )
        if k_nscf[0] * k_nscf[1] * k_nscf[2] < 64:
            click.echo(
                click.style(
                    "[preflight] warning: nscf kmesh product < 64 may miss Weyl features.",
                    fg="yellow",
                )
            )

    # Determine if cluster checks are required before this run starts
    cluster_required = False
    if resume or stage_norm is None:
        cluster_required = True
    elif stage_norm in {"DFT_SCF", "DFT_NSCF", "WANNIER90"}:
        cluster_required = True
    elif stage_norm in {"TRANSPORT", "ANALYSIS"} and not transport_only:
        # Falls back to DFT/Wannier generation if hr_dat isn't provided.
        cluster_required = True

    if not cluster_required:
        if transport_only:
            source = "config hr_dat_path" if hr_cfg else "checkpoint hr_dat"
            msg = f"[preflight] OK (transport-only using {source})"
        else:
            msg = f"[preflight] OK (local stage={stage_norm or 'INIT'})"
        click.echo(click.style(msg, fg="green"))
        return

    from wtec.config.cluster import ClusterConfig
    from wtec.cluster.ssh import open_ssh
    from wtec.cluster.submit import JobManager
    from wtec.config.materials import get_material

    try:
        cluster_cfg = ClusterConfig.from_env()
    except Exception as exc:
        raise click.UsageError(
            "Cluster environment is not configured. Run `wtec init` with "
            "cluster options (or update .env) first.\n"
            f"Details: {exc}"
        )

    if material:
        try:
            preset = get_material(material)
        except Exception as exc:
            raise click.UsageError(str(exc))
    else:
        preset = None

    with open_ssh(cluster_cfg) as ssh:
        jm = JobManager(ssh)
        jm.ensure_remote_commands(
            ["qsub", "qstat", "mpirun"],
            modules=cluster_cfg.modules,
            bin_dirs=cluster_cfg.bin_dirs,
        )
        vasp_cfg = cfg.get("dft_vasp", {})
        if not isinstance(vasp_cfg, dict):
            vasp_cfg = {}
        if isinstance(cfg.get("dft"), dict) and isinstance(cfg["dft"].get("vasp"), dict):
            merged_vasp = dict(cfg["dft"]["vasp"])
            merged_vasp.update(vasp_cfg)
            vasp_cfg = merged_vasp
        abacus_cfg = cfg.get("dft_abacus", {})
        if not isinstance(abacus_cfg, dict):
            abacus_cfg = {}
        if isinstance(cfg.get("dft"), dict) and isinstance(cfg["dft"].get("abacus"), dict):
            merged_abacus = dict(cfg["dft"]["abacus"])
            merged_abacus.update(abacus_cfg)
            abacus_cfg = merged_abacus
        vasp_exec = str(vasp_cfg.get("executable", "vasp_std")).strip() or "vasp_std"
        abacus_exec = str(abacus_cfg.get("executable", "abacus")).strip() or "abacus"
        required_execs: list[str] = []
        if dft_mode == "hybrid_qe_ref_siesta_variants":
            required_execs = ["pw.x", "pw2wannier90.x", "siesta", "wannier90.x"]
        elif dft_mode == "dual_family":
            if resume or stage_norm is None:
                if dft_engine == "qe":
                    required_execs.extend(["pw.x", "pw2wannier90.x", "wannier90.x"])
                if dft_engine == "vasp":
                    required_execs.extend([vasp_exec, "wannier90.x"])
                if variant_dft_engine == "siesta":
                    required_execs.extend(["siesta", "wannier90.x"])
                if variant_dft_engine == "abacus":
                    required_execs.extend([abacus_exec, "wannier90.x"])
            elif stage_norm in {"DFT_SCF", "DFT_NSCF", "WANNIER90"}:
                if dft_engine == "qe":
                    if stage_norm in {"DFT_SCF", "DFT_NSCF"}:
                        required_execs.extend(["pw.x"])
                    else:
                        required_execs.extend(["pw.x", "pw2wannier90.x", "wannier90.x"])
                elif dft_engine == "siesta":
                    required_execs.extend(["siesta", "wannier90.x"])
                elif dft_engine == "vasp":
                    if stage_norm in {"DFT_SCF", "DFT_NSCF"}:
                        required_execs.extend([vasp_exec])
                    else:
                        required_execs.extend([vasp_exec, "wannier90.x"])
            elif stage_norm in {"TRANSPORT", "ANALYSIS"} and not transport_only:
                if dft_engine == "qe":
                    required_execs.extend(["pw.x", "pw2wannier90.x", "wannier90.x"])
                if dft_engine == "vasp":
                    required_execs.extend([vasp_exec, "wannier90.x"])
                if variant_dft_engine == "siesta":
                    required_execs.extend(["siesta", "wannier90.x"])
                if variant_dft_engine == "abacus":
                    required_execs.extend([abacus_exec, "wannier90.x"])
        elif dft_engine == "qe":
            if resume or stage_norm is None:
                required_execs = ["pw.x", "pw2wannier90.x", "wannier90.x"]
            elif stage_norm in {"DFT_SCF", "DFT_NSCF"}:
                required_execs = ["pw.x"]
            elif stage_norm == "WANNIER90":
                required_execs = ["pw.x", "pw2wannier90.x", "wannier90.x"]
            elif stage_norm in {"TRANSPORT", "ANALYSIS"} and not transport_only:
                required_execs = ["pw.x", "pw2wannier90.x", "wannier90.x"]
        elif dft_engine == "siesta":
            if resume or stage_norm is None:
                required_execs = ["siesta", "wannier90.x"]
            elif stage_norm in {"DFT_SCF", "DFT_NSCF"}:
                required_execs = ["siesta"]
            elif stage_norm == "WANNIER90":
                required_execs = ["siesta", "wannier90.x"]
            elif stage_norm in {"TRANSPORT", "ANALYSIS"} and not transport_only:
                required_execs = ["siesta", "wannier90.x"]
        elif dft_engine == "vasp":
            if resume or stage_norm is None:
                required_execs = [vasp_exec, "wannier90.x"]
            elif stage_norm in {"DFT_SCF", "DFT_NSCF"}:
                required_execs = [vasp_exec]
            elif stage_norm == "WANNIER90":
                required_execs = [vasp_exec, "wannier90.x"]
            elif stage_norm in {"TRANSPORT", "ANALYSIS"} and not transport_only:
                required_execs = [vasp_exec, "wannier90.x"]
        required_execs = list(dict.fromkeys(required_execs))

        if required_execs:
            jm.ensure_remote_commands(
                required_execs,
                modules=cluster_cfg.modules,
                bin_dirs=cluster_cfg.bin_dirs,
            )
            jm.ensure_remote_mpi_binaries(
                required_execs,
                modules=cluster_cfg.modules,
                bin_dirs=cluster_cfg.bin_dirs,
            )
        if preset is not None:
            if dft_mode == "hybrid_qe_ref_siesta_variants":
                qe_pp = sorted(set(preset.pseudopots.values()))
                si_pp = sorted(set(preset.siesta_pseudopots.values()))
                jm.ensure_remote_files(cluster_cfg.qe_pseudo_dir, qe_pp)
                jm.ensure_remote_files(cluster_cfg.siesta_pseudo_dir, si_pp)
                pseudo_dir_report = (
                    f"qe:{cluster_cfg.qe_pseudo_dir}, "
                    f"siesta:{cluster_cfg.siesta_pseudo_dir}"
                )
            elif dft_mode == "dual_family":
                reports: list[str] = []
                if dft_engine == "qe":
                    qe_pp = sorted(set(preset.pseudopots.values()))
                    jm.ensure_remote_files(cluster_cfg.qe_pseudo_dir, qe_pp)
                    reports.append(f"qe:{cluster_cfg.qe_pseudo_dir}")
                if dft_engine == "vasp":
                    vasp_pp = sorted(set((getattr(preset, "vasp_potcars", {}) or {}).values()))
                    vasp_pseudo_dir = str(vasp_cfg.get("pseudo_dir") or cluster_cfg.vasp_pseudo_dir)
                    if vasp_pp:
                        jm.ensure_remote_files(vasp_pseudo_dir, vasp_pp)
                    reports.append(f"vasp:{vasp_pseudo_dir}")
                if variant_dft_engine == "siesta":
                    si_pp = sorted(set(preset.siesta_pseudopots.values()))
                    jm.ensure_remote_files(cluster_cfg.siesta_pseudo_dir, si_pp)
                    reports.append(f"siesta:{cluster_cfg.siesta_pseudo_dir}")
                if variant_dft_engine == "abacus":
                    abacus_pseudo_dir = str(abacus_cfg.get("pseudo_dir") or cluster_cfg.abacus_pseudo_dir)
                    abacus_orb_dir = str(abacus_cfg.get("orbital_dir") or cluster_cfg.abacus_orbital_dir)
                    ab_pp = sorted(set((getattr(preset, "abacus_pseudopots", {}) or {}).values()))
                    ab_orb = sorted(set((getattr(preset, "abacus_orbitals", {}) or {}).values()))
                    if ab_pp:
                        jm.ensure_remote_files(abacus_pseudo_dir, ab_pp)
                    if ab_orb:
                        jm.ensure_remote_files(abacus_orb_dir, ab_orb)
                    reports.append(f"abacus_pseudo:{abacus_pseudo_dir}")
                    reports.append(f"abacus_orbital:{abacus_orb_dir}")
                pseudo_dir_report = ", ".join(reports) if reports else "(engine-specific)"
            elif dft_engine == "qe":
                required_pp = sorted(set(preset.pseudopots.values()))
                jm.ensure_remote_files(cluster_cfg.qe_pseudo_dir, required_pp)
                pseudo_dir_report = cluster_cfg.qe_pseudo_dir
            elif dft_engine == "vasp":
                vasp_pseudo_dir = str(vasp_cfg.get("pseudo_dir") or cluster_cfg.vasp_pseudo_dir)
                vasp_pp = sorted(set((getattr(preset, "vasp_potcars", {}) or {}).values()))
                if vasp_pp:
                    jm.ensure_remote_files(vasp_pseudo_dir, vasp_pp)
                pseudo_dir_report = vasp_pseudo_dir
            elif dft_engine == "abacus":
                abacus_pseudo_dir = str(abacus_cfg.get("pseudo_dir") or cluster_cfg.abacus_pseudo_dir)
                abacus_orb_dir = str(abacus_cfg.get("orbital_dir") or cluster_cfg.abacus_orbital_dir)
                ab_pp = sorted(set((getattr(preset, "abacus_pseudopots", {}) or {}).values()))
                ab_orb = sorted(set((getattr(preset, "abacus_orbitals", {}) or {}).values()))
                if ab_pp:
                    jm.ensure_remote_files(abacus_pseudo_dir, ab_pp)
                if ab_orb:
                    jm.ensure_remote_files(abacus_orb_dir, ab_orb)
                pseudo_dir_report = f"{abacus_pseudo_dir}, {abacus_orb_dir}"
            else:
                required_pp = sorted(set(preset.siesta_pseudopots.values()))
                jm.ensure_remote_files(cluster_cfg.siesta_pseudo_dir, required_pp)
                pseudo_dir_report = cluster_cfg.siesta_pseudo_dir
        else:
            if dft_mode == "hybrid_qe_ref_siesta_variants":
                pseudo_dir_report = (
                    f"qe:{cluster_cfg.qe_pseudo_dir}, "
                    f"siesta:{cluster_cfg.siesta_pseudo_dir}"
                )
            elif dft_mode == "dual_family":
                reports: list[str] = []
                if dft_engine == "qe":
                    reports.append(f"qe:{cluster_cfg.qe_pseudo_dir}")
                if dft_engine == "vasp":
                    reports.append(f"vasp:{str(vasp_cfg.get('pseudo_dir') or cluster_cfg.vasp_pseudo_dir)}")
                if variant_dft_engine == "siesta":
                    reports.append(f"siesta:{cluster_cfg.siesta_pseudo_dir}")
                if variant_dft_engine == "abacus":
                    reports.append(f"abacus_pseudo:{str(abacus_cfg.get('pseudo_dir') or cluster_cfg.abacus_pseudo_dir)}")
                    reports.append(f"abacus_orbital:{str(abacus_cfg.get('orbital_dir') or cluster_cfg.abacus_orbital_dir)}")
                pseudo_dir_report = ", ".join(reports) if reports else "(engine-specific)"
            else:
                if dft_engine == "qe":
                    pseudo_dir_report = cluster_cfg.qe_pseudo_dir
                elif dft_engine == "siesta":
                    pseudo_dir_report = cluster_cfg.siesta_pseudo_dir
                elif dft_engine == "vasp":
                    pseudo_dir_report = str(vasp_cfg.get("pseudo_dir") or cluster_cfg.vasp_pseudo_dir)
                else:
                    pseudo_dir_report = (
                        f"{str(abacus_cfg.get('pseudo_dir') or cluster_cfg.abacus_pseudo_dir)}, "
                        f"{str(abacus_cfg.get('orbital_dir') or cluster_cfg.abacus_orbital_dir)}"
                    )
        queue_used = jm.resolve_queue(
            cluster_cfg.pbs_queue,
            fallback_order=cluster_cfg.pbs_queue_priority,
        )
        cores_per_node = cluster_cfg.cores_for_queue(queue_used)
        click.echo(
            click.style(
                "[preflight] OK "
                f"(cluster queue={queue_used}, cores_per_node={cores_per_node}, "
                f"run_profile={run_profile}, "
                f"dft_mode={dft_mode}, pes_engine={dft_engine}, "
                f"lcao_engine={variant_dft_engine}, pseudo_dir={pseudo_dir_report})",
                fg="green",
            )
        )


def _run_requires_cluster(cfg: dict, *, resume: bool, stage: str | None) -> bool:
    stage_norm = _normalize_stage(stage)
    hr_cfg = cfg.get("hr_dat_path")
    hr_path: Path | None = None
    if hr_cfg:
        p = Path(str(hr_cfg))
        if p.exists() and p.stat().st_size > 0:
            hr_path = p
    else:
        hr_path = _checkpoint_hr_dat(cfg)

    transport_only = (
        not resume and stage_norm in {"TRANSPORT", "ANALYSIS"} and hr_path is not None
    )
    if resume or stage_norm is None:
        return True
    if stage_norm in {"DFT_SCF", "DFT_NSCF", "WANNIER90"}:
        return True
    if stage_norm in {"TRANSPORT", "ANALYSIS"} and not transport_only:
        return True
    return False


def _merge_runtime_cluster_interactive(
    *,
    cfg: dict[str, Any],
    resume: bool,
    stage: str | None,
    runtime_env_updates: dict[str, str],
    interactive_cluster: bool,
) -> dict[str, str]:
    out = dict(runtime_env_updates)
    if not interactive_cluster:
        return out
    if not _run_requires_cluster(cfg, resume=resume, stage=stage):
        return out
    if not sys.stdin.isatty():
        click.echo(
            click.style(
                "[runtime] non-interactive stdin; skipping cluster prompts.",
                fg="yellow",
            )
        )
        return out

    def current(key: str) -> str | None:
        if key in out:
            val = str(out[key]).strip()
            if val:
                return val
        raw = os.environ.get(key, "").strip()
        return raw or None

    if not current("TOPOSLAB_CLUSTER_HOST"):
        out["TOPOSLAB_CLUSTER_HOST"] = click.prompt("Cluster host", type=str).strip()
    if not current("TOPOSLAB_CLUSTER_PORT"):
        out["TOPOSLAB_CLUSTER_PORT"] = str(
            click.prompt("Cluster port", type=int, default=22, show_default=True)
        )
    if not current("TOPOSLAB_CLUSTER_USER"):
        out["TOPOSLAB_CLUSTER_USER"] = click.prompt("Cluster user", type=str).strip()

    if not current("TOPOSLAB_CLUSTER_KEY") and not current("TOPOSLAB_CLUSTER_PASS"):
        use_key = click.confirm("Use SSH key authentication?", default=False)
        if use_key:
            out["TOPOSLAB_CLUSTER_KEY"] = click.prompt(
                "SSH key path",
                type=str,
                default="~/.ssh/id_rsa",
                show_default=True,
            ).strip()
        else:
            out["TOPOSLAB_CLUSTER_PASS"] = click.prompt(
                "Cluster password",
                hide_input=True,
                type=str,
            )

    if not current("TOPOSLAB_REMOTE_WORKDIR"):
        out["TOPOSLAB_REMOTE_WORKDIR"] = click.prompt(
            "Remote workdir",
            type=str,
            default=str(Path.cwd()),
            show_default=True,
        ).strip()

    pseudo = current("TOPOSLAB_QE_PSEUDO_DIR")
    if not pseudo or pseudo == "/pseudo":
        user = current("TOPOSLAB_CLUSTER_USER") or "msj"
        # Prefer the generated pslibrary UPF directory layout used by QE users.
        default_pseudo = f"/home/{user}/src/QE_pseudo/pslibrary/pbe/PSEUDOPOTENTIALS"
        out["TOPOSLAB_QE_PSEUDO_DIR"] = click.prompt(
            "QE pseudo dir",
            type=str,
            default=default_pseudo,
            show_default=True,
        ).strip()

    siesta_pseudo = current("TOPOSLAB_SIESTA_PSEUDO_DIR")
    if not siesta_pseudo or siesta_pseudo == "/pseudo":
        user = current("TOPOSLAB_CLUSTER_USER") or "msj"
        out["TOPOSLAB_SIESTA_PSEUDO_DIR"] = click.prompt(
            "SIESTA pseudo dir",
            type=str,
            default=f"/home/{user}/siesta/pseudo",
            show_default=True,
        ).strip()

    vasp_pseudo = current("TOPOSLAB_VASP_PSEUDO_DIR")
    if not vasp_pseudo or vasp_pseudo == "/pseudo":
        user = current("TOPOSLAB_CLUSTER_USER") or "msj"
        out["TOPOSLAB_VASP_PSEUDO_DIR"] = click.prompt(
            "VASP pseudo dir",
            type=str,
            default=f"/home/{user}/vasp/potpaw_PBE",
            show_default=True,
        ).strip()

    abacus_pseudo = current("TOPOSLAB_ABACUS_PSEUDO_DIR")
    if not abacus_pseudo or abacus_pseudo == "/pseudo":
        user = current("TOPOSLAB_CLUSTER_USER") or "msj"
        out["TOPOSLAB_ABACUS_PSEUDO_DIR"] = click.prompt(
            "ABACUS pseudo dir",
            type=str,
            default=f"/home/{user}/abacus/pseudo",
            show_default=True,
        ).strip()

    abacus_orb = current("TOPOSLAB_ABACUS_ORBITAL_DIR")
    if not abacus_orb or abacus_orb == "/orbital":
        user = current("TOPOSLAB_CLUSTER_USER") or "msj"
        out["TOPOSLAB_ABACUS_ORBITAL_DIR"] = click.prompt(
            "ABACUS orbital dir",
            type=str,
            default=f"/home/{user}/abacus/orbital",
            show_default=True,
        ).strip()

    return out


def _load_runtime_dotenv(config_file: str | None) -> None:
    """Load .env before runtime prompts, even when run from nested directories."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except Exception:
        return

    loaded: set[str] = set()
    found_cwd = find_dotenv(usecwd=True)
    if found_cwd:
        load_dotenv(found_cwd, override=True)
        loaded.add(str(Path(found_cwd).resolve()))

    if config_file:
        cfg_path = Path(config_file).expanduser().resolve()
        for parent in (cfg_path.parent, *cfg_path.parent.parents):
            env_path = (parent / ".env").resolve()
            if not env_path.exists():
                continue
            key = str(env_path)
            if key in loaded:
                break
            load_dotenv(env_path, override=True)
            break


def _resolve_run_config_path(config_file: str | None) -> Path:
    if config_file:
        p = Path(config_file).expanduser().resolve()
        if not p.exists():
            raise click.ClickException(f"Config file not found: {p}")
        return p
    default_tpl = _default_project_template_path(required=True)
    assert default_tpl is not None
    return default_tpl


def _abs_from(path_raw: str | None, *, base: Path) -> str | None:
    if not isinstance(path_raw, str) or not path_raw.strip():
        return None
    p = Path(path_raw).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    return str(p)


def _int_list3(raw: Any, *, default: tuple[int, int, int]) -> list[int]:
    if raw is None:
        return [int(default[0]), int(default[1]), int(default[2])]
    if not isinstance(raw, list) or len(raw) != 3:
        raise click.ClickException(f"Expected list of length 3, got: {raw!r}")
    return [int(raw[0]), int(raw[1]), int(raw[2])]


def _normalize_dft_track_config(
    cfg: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Normalize dual-family/legacy DFT keys into one runtime shape.

    This keeps backward compatibility with existing configs while exposing
    dual-family keys:
      - dft_mode = dual_family|hybrid_qe_ref_siesta_variants|legacy_single
      - dft_pes_engine = qe|vasp (and legacy qe|siesta for legacy_single)
      - dft_lcao_engine = siesta|abacus (legacy qe|siesta allowed)
      - dft_pes_reference_structure_file (explicit small-structure path)
    """
    if not isinstance(cfg, dict):
        return cfg

    dft_tbl = cfg.get("dft", {})
    dft_tbl = dft_tbl if isinstance(dft_tbl, dict) else {}
    ref_tbl = dft_tbl.get("reference", {})
    ref_tbl = ref_tbl if isinstance(ref_tbl, dict) else {}
    var_tbl = dft_tbl.get("variants", {})
    var_tbl = var_tbl if isinstance(var_tbl, dict) else {}
    tracks_tbl = dft_tbl.get("tracks", {})
    tracks_tbl = tracks_tbl if isinstance(tracks_tbl, dict) else {}
    pes_track = tracks_tbl.get("pes_reference", {})
    pes_track = pes_track if isinstance(pes_track, dict) else {}
    lcao_track = tracks_tbl.get("lcao_upscaled", {})
    lcao_track = lcao_track if isinstance(lcao_track, dict) else {}
    anchor_tbl = dft_tbl.get("anchor_transfer", {})
    anchor_tbl = anchor_tbl if isinstance(anchor_tbl, dict) else {}

    mode_raw = cfg.get("dft_mode")
    if mode_raw is None:
        mode_raw = dft_tbl.get("mode")
    if mode_raw is None and (pes_track or lcao_track):
        mode_raw = "dual_family"
    mode = str(mode_raw or "legacy_single").strip().lower() or "legacy_single"

    pes_engine_raw = cfg.get("dft_pes_engine")
    if pes_engine_raw is None:
        pes_engine_raw = pes_track.get("engine")
    if pes_engine_raw is None:
        pes_engine_raw = cfg.get("dft_reference_engine")
    if pes_engine_raw is None:
        pes_engine_raw = ref_tbl.get("engine")
    if pes_engine_raw is None:
        pes_engine_raw = cfg.get("dft_engine")
    if pes_engine_raw is None:
        pes_engine_raw = dft_tbl.get("engine")
    if pes_engine_raw is None:
        if mode == "hybrid_qe_ref_siesta_variants":
            pes_engine_raw = "qe"
        else:
            pes_engine_raw = "qe"
    pes_engine = str(pes_engine_raw or "qe").strip().lower() or "qe"

    lcao_engine_raw = cfg.get("dft_lcao_engine")
    if lcao_engine_raw is None:
        lcao_engine_raw = lcao_track.get("engine")
    if lcao_engine_raw is None:
        lcao_engine_raw = cfg.get("topology_variant_dft_engine")
    topo_tbl = cfg.get("topology", {})
    topo_tbl = topo_tbl if isinstance(topo_tbl, dict) else {}
    if lcao_engine_raw is None:
        lcao_engine_raw = topo_tbl.get("variant_dft_engine")
    if lcao_engine_raw is None:
        lcao_engine_raw = var_tbl.get("engine")
    if lcao_engine_raw is None:
        if mode in {"hybrid_qe_ref_siesta_variants", "dual_family"}:
            lcao_engine_raw = "siesta"
        else:
            lcao_engine_raw = pes_engine
    lcao_engine = str(lcao_engine_raw or "siesta").strip().lower() or "siesta"

    lcao_source_raw = cfg.get("dft_lcao_source")
    if lcao_source_raw is None:
        lcao_source_raw = lcao_track.get("source")
    lcao_source = str(lcao_source_raw or "variants").strip().lower() or "variants"

    pes_ref_struct_raw = cfg.get("dft_pes_reference_structure_file")
    if pes_ref_struct_raw is None:
        pes_ref_struct_raw = pes_track.get("structure_file")
    if pes_ref_struct_raw is None:
        pes_ref_struct_raw = ref_tbl.get("structure_file")
    if pes_ref_struct_raw is None:
        pes_ref_struct_raw = ref_tbl.get("reference_structure_file")
    if pes_ref_struct_raw is None:
        pes_ref_struct_raw = ref_tbl.get("pristine_structure_file")
    pes_ref_struct: str | None = None
    if isinstance(pes_ref_struct_raw, str) and pes_ref_struct_raw.strip():
        if base_dir is not None:
            pes_ref_struct = _abs_from(pes_ref_struct_raw, base=base_dir)
        else:
            pes_ref_struct = str(Path(pes_ref_struct_raw).expanduser())

    cfg["dft_mode"] = mode
    cfg["dft_pes_engine"] = pes_engine
    cfg["dft_lcao_engine"] = lcao_engine
    cfg["dft_lcao_source"] = lcao_source
    if pes_ref_struct:
        cfg["dft_pes_reference_structure_file"] = pes_ref_struct

    anchor_flat = cfg.get("dft_anchor_transfer", {})
    anchor_flat = anchor_flat if isinstance(anchor_flat, dict) else {}
    anchor_cfg: dict[str, Any] = {
        "enabled": mode == "dual_family",
        "mode": "delta_h",
        "basis_policy": "strict_same_basis",
        "scope": "onsite_plus_first_shell",
        "fit_window_ev": 1.5,
        "fit_kmesh": [8, 8, 8],
        "alpha_grid_min": -0.5,
        "alpha_grid_max": 1.5,
        "alpha_grid_points": 81,
        "max_retries": 5,
        "retry_kmesh_step": 2,
        "retry_window_step_ev": 0.5,
        "reuse_existing": True,
    }
    anchor_cfg.update(anchor_tbl)
    anchor_cfg.update(anchor_flat)
    anchor_cfg["enabled"] = bool(anchor_cfg.get("enabled", mode == "dual_family"))
    anchor_cfg["mode"] = str(anchor_cfg.get("mode", "delta_h")).strip().lower() or "delta_h"
    anchor_cfg["basis_policy"] = (
        str(anchor_cfg.get("basis_policy", "strict_same_basis")).strip() or "strict_same_basis"
    )
    anchor_cfg["scope"] = (
        str(anchor_cfg.get("scope", "onsite_plus_first_shell")).strip() or "onsite_plus_first_shell"
    )
    anchor_cfg["fit_window_ev"] = float(anchor_cfg.get("fit_window_ev", 1.5))
    anchor_cfg["fit_kmesh"] = _int_list3(anchor_cfg.get("fit_kmesh"), default=(8, 8, 8))
    anchor_cfg["alpha_grid_min"] = float(anchor_cfg.get("alpha_grid_min", -0.5))
    anchor_cfg["alpha_grid_max"] = float(anchor_cfg.get("alpha_grid_max", 1.5))
    anchor_cfg["alpha_grid_points"] = int(anchor_cfg.get("alpha_grid_points", 81))
    anchor_cfg["max_retries"] = int(anchor_cfg.get("max_retries", 5))
    anchor_cfg["retry_kmesh_step"] = int(anchor_cfg.get("retry_kmesh_step", 2))
    anchor_cfg["retry_window_step_ev"] = float(anchor_cfg.get("retry_window_step_ev", 0.5))
    anchor_cfg["reuse_existing"] = bool(anchor_cfg.get("reuse_existing", True))
    cfg["dft_anchor_transfer"] = anchor_cfg

    # Backward-compatible runtime aliases used by existing internals.
    cfg["dft_engine"] = pes_engine
    cfg["dft_reference_engine"] = str(
        cfg.get("dft_reference_engine", pes_engine)
    ).strip().lower() or pes_engine
    cfg["topology_variant_dft_engine"] = lcao_engine
    if isinstance(cfg.get("topology"), dict):
        cfg["topology"]["variant_dft_engine"] = lcao_engine
    return cfg


def _resolve_master_structure_file(data: dict[str, Any], *, base: Path) -> str:
    run = data.get("run", {}) if isinstance(data.get("run"), dict) else {}
    explicit = _abs_from(run.get("structure_file"), base=base)
    if explicit:
        return explicit

    defect = data.get("defect", {}) if isinstance(data.get("defect"), dict) else {}
    project = data.get("project", {}) if isinstance(data.get("project"), dict) else {}
    export = data.get("export", {}) if isinstance(data.get("export"), dict) else {}
    project_name = str(project.get("name", "project")).strip() or "project"
    defect_dir = _abs_from(defect.get("output_dir"), base=base) or str((base / "slab_variants").resolve())
    pristine_suffix = str(defect.get("pristine_suffix", "pristine")).strip() or "pristine"
    pristine = Path(defect_dir) / f"{project_name}_{pristine_suffix}.generated.cif"
    if pristine.exists():
        return str(pristine.resolve())

    cif_path = _abs_from(export.get("cif_path"), base=(base / str(project.get("output_dir", "slab_outputs"))))
    if cif_path:
        return cif_path
    return str((base / "slab_outputs" / f"{project_name}.generated.cif").resolve())


def _build_cfg_from_master_toml(data: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    base = source_path.parent
    project = data.get("project", {}) if isinstance(data.get("project"), dict) else {}
    cluster = data.get("cluster", {}) if isinstance(data.get("cluster"), dict) else {}
    run = data.get("run", {}) if isinstance(data.get("run"), dict) else {}
    dft = data.get("dft", {}) if isinstance(data.get("dft"), dict) else {}
    dft_k = dft.get("kmesh", {}) if isinstance(dft.get("kmesh"), dict) else {}
    dft_siesta = dft.get("siesta", {}) if isinstance(dft.get("siesta"), dict) else {}
    dft_vasp = dft.get("vasp", {}) if isinstance(dft.get("vasp"), dict) else {}
    dft_abacus = dft.get("abacus", {}) if isinstance(dft.get("abacus"), dict) else {}
    dft_reference = dft.get("reference", {}) if isinstance(dft.get("reference"), dict) else {}
    dft_variants = dft.get("variants", {}) if isinstance(dft.get("variants"), dict) else {}
    dft_tracks = dft.get("tracks", {}) if isinstance(dft.get("tracks"), dict) else {}
    dft_anchor = dft.get("anchor_transfer", {}) if isinstance(dft.get("anchor_transfer"), dict) else {}
    pes_track = (
        dft_tracks.get("pes_reference", {})
        if isinstance(dft_tracks.get("pes_reference"), dict)
        else {}
    )
    lcao_track = (
        dft_tracks.get("lcao_upscaled", {})
        if isinstance(dft_tracks.get("lcao_upscaled"), dict)
        else {}
    )
    dft_disp = dft.get("dispersion", {}) if isinstance(dft.get("dispersion"), dict) else {}
    transport = data.get("transport", {}) if isinstance(data.get("transport"), dict) else {}
    transport_autotune = (
        transport.get("autotune", {})
        if isinstance(transport.get("autotune"), dict)
        else {}
    )
    topo = data.get("topology", {}) if isinstance(data.get("topology"), dict) else {}
    topo_k = topo.get("kmesh", {}) if isinstance(topo.get("kmesh"), dict) else {}
    topo_res = topo.get("resources", {}) if isinstance(topo.get("resources"), dict) else {}
    topo_hr = topo.get("hr_grid", {}) if isinstance(topo.get("hr_grid"), dict) else {}
    topo_score = topo.get("score", {}) if isinstance(topo.get("score"), dict) else {}
    topo_tiering = topo.get("tiering", {}) if isinstance(topo.get("tiering"), dict) else {}
    topo_transport_probe = (
        topo.get("transport_probe", {})
        if isinstance(topo.get("transport_probe"), dict)
        else {}
    )
    topo_tiering_screen = (
        topo_tiering.get("screen", {})
        if isinstance(topo_tiering.get("screen"), dict)
        else {}
    )
    defect = data.get("defect", {}) if isinstance(data.get("defect"), dict) else {}
    report = data.get("report", {}) if isinstance(data.get("report"), dict) else {}

    project_name = str(project.get("name", "run")).strip() or "run"
    run_name = str(run.get("name", f"{project_name}_run")).strip() or f"{project_name}_run"
    run_dir_raw = _abs_from(run.get("run_dir"), base=base)
    run_dir = run_dir_raw or str((base / "runs" / run_name).resolve())
    structure_file = _resolve_master_structure_file(data, base=base)

    remote_workdir = _abs_from(cluster.get("remote_workdir"), base=base)
    if remote_workdir:
        remote_workdir = remote_workdir.rstrip("/")
    else:
        remote_root = os.environ.get("TOPOSLAB_REMOTE_WORKDIR", "").strip().rstrip("/")
        remote_workdir = f"{remote_root}/{run_name}" if remote_root else ""

    variant_glob = str(topo.get("variant_discovery_glob", "")).strip()
    if not variant_glob:
        defect_dir_rel = str(defect.get("output_dir", "slab_variants")).strip() or "slab_variants"
        variant_glob = f"{defect_dir_rel}/*.generated.meta.json"

    dft_mode_raw = dft.get("mode")
    if dft_mode_raw is None and (pes_track or lcao_track):
        dft_mode_raw = "dual_family"
    dft_mode = str(dft_mode_raw or "legacy_single").strip().lower() or "legacy_single"
    dft_engine_cfg = str(dft.get("engine", "qe")).strip().lower() or "qe"
    if dft_mode in {"hybrid_qe_ref_siesta_variants", "dual_family"}:
        dft_engine_cfg = str(
            pes_track.get("engine", dft_reference.get("engine", "qe"))
        ).strip().lower() or "qe"
    transport_policy_default = "single_track"
    lcao_default = "siesta" if dft_mode in {"hybrid_qe_ref_siesta_variants", "dual_family"} else dft_engine_cfg
    variant_engine_cfg = str(
        topo.get(
            "variant_dft_engine",
            lcao_track.get("engine", dft_variants.get("engine", lcao_default)),
        )
    ).strip().lower() or lcao_default
    pes_reference_structure = _abs_from(
        pes_track.get("structure_file")
        if isinstance(pes_track.get("structure_file"), str)
        else dft_reference.get("structure_file"),
        base=base,
    )
    lcao_source = str(lcao_track.get("source", "variants")).strip().lower() or "variants"

    cfg: dict[str, Any] = {
        "name": run_name,
        "material": str(run.get("material", project.get("material", "TaP"))),
        "run_profile": str(run.get("profile", "strict")).strip().lower() or "strict",
        "dft_mode": dft_mode,
        "dft_engine": dft_engine_cfg,
        "dft_pes_engine": dft_engine_cfg,
        "dft_lcao_engine": variant_engine_cfg,
        "dft_pes_reference_structure_file": pes_reference_structure,
        "dft_lcao_source": lcao_source,
        "dft_reuse_mode": str(dft.get("reuse_mode", "none")).strip().lower() or "none",
        "dft_anchor_transfer": {
            "enabled": bool(dft_anchor.get("enabled", dft_mode == "dual_family")),
            "mode": str(dft_anchor.get("mode", "delta_h")).strip().lower() or "delta_h",
            "basis_policy": str(
                dft_anchor.get("basis_policy", "strict_same_basis")
            ).strip() or "strict_same_basis",
            "scope": str(
                dft_anchor.get("scope", "onsite_plus_first_shell")
            ).strip() or "onsite_plus_first_shell",
            "fit_window_ev": float(dft_anchor.get("fit_window_ev", 1.5)),
            "fit_kmesh": _int_list3(dft_anchor.get("fit_kmesh"), default=(8, 8, 8)),
            "alpha_grid_min": float(dft_anchor.get("alpha_grid_min", -0.5)),
            "alpha_grid_max": float(dft_anchor.get("alpha_grid_max", 1.5)),
            "alpha_grid_points": int(dft_anchor.get("alpha_grid_points", 81)),
            "max_retries": int(dft_anchor.get("max_retries", 5)),
            "retry_kmesh_step": int(dft_anchor.get("retry_kmesh_step", 2)),
            "retry_window_step_ev": float(dft_anchor.get("retry_window_step_ev", 0.5)),
            "reuse_existing": bool(dft_anchor.get("reuse_existing", True)),
        },
        "dft_reference_engine": str(
            pes_track.get("engine", dft_reference.get("engine", dft_engine_cfg))
        ).strip().lower() or dft_engine_cfg,
        "dft_reference_reuse_policy": str(
            pes_track.get("reuse_policy", dft_reference.get("reuse_policy", "strict_hash"))
        ).strip().lower() or "strict_hash",
        "topology_variant_dft_engine": variant_engine_cfg,
        "hr_dat_path": _abs_from(dft.get("hr_dat_path"), base=base),
        "dft_siesta": {
            "wannier_interface": str(dft_siesta.get("wannier_interface", "sisl")).strip().lower() or "sisl",
            "pseudo_dir": str(dft_siesta.get("pseudo_dir", "")).strip(),
            "basis_profile": str(dft_siesta.get("basis_profile", "")).strip(),
            "mpi_np_scf": int(dft_siesta.get("mpi_np_scf", 0)),
            "mpi_np_nscf": int(dft_siesta.get("mpi_np_nscf", 0)),
            "mpi_np_wannier": int(dft_siesta.get("mpi_np_wannier", 0)),
            "omp_threads_scf": int(dft_siesta.get("omp_threads_scf", 0)),
            "omp_threads_nscf": int(dft_siesta.get("omp_threads_nscf", 0)),
            "omp_threads_wannier": int(dft_siesta.get("omp_threads_wannier", 0)),
            "factorization_defaults": (
                dict(dft_siesta.get("factorization_defaults"))
                if isinstance(dft_siesta.get("factorization_defaults"), dict)
                else {}
            ),
            "dm_mixing_weight": float(dft_siesta.get("dm_mixing_weight", 0.10)),
            "dm_number_pulay": int(dft_siesta.get("dm_number_pulay", 8)),
            "electronic_temperature_k": float(dft_siesta.get("electronic_temperature_k", 300.0)),
            "max_scf_iterations": int(dft_siesta.get("max_scf_iterations", 200)),
        },
        "dft_vasp": {
            "pseudo_dir": str(dft_vasp.get("pseudo_dir", "")).strip(),
            "executable": str(dft_vasp.get("executable", "vasp_std")).strip() or "vasp_std",
            "encut_ev": float(dft_vasp.get("encut_ev", 520.0)),
            "ediff": float(dft_vasp.get("ediff", 1.0e-6)),
            "ismear": int(dft_vasp.get("ismear", 0)),
            "sigma": float(dft_vasp.get("sigma", 0.05)),
            "disable_symmetry": bool(dft_vasp.get("disable_symmetry", True)),
        },
        "dft_abacus": {
            "pseudo_dir": str(dft_abacus.get("pseudo_dir", "")).strip(),
            "orbital_dir": str(dft_abacus.get("orbital_dir", "")).strip(),
            "executable": str(dft_abacus.get("executable", "abacus")).strip() or "abacus",
            "basis_type": str(dft_abacus.get("basis_type", "lcao")).strip().lower() or "lcao",
            "ks_solver": str(dft_abacus.get("ks_solver", "genelpa")).strip().lower() or "genelpa",
        },
        "dft_dispersion": {
            "enabled": bool(dft_disp.get("enabled", True)),
            "method": str(dft_disp.get("method", "d3")).strip().lower() or "d3",
            "qe_vdw_corr": str(dft_disp.get("qe_vdw_corr", "grimme-d3")).strip() or "grimme-d3",
            "qe_dftd3_version": int(dft_disp.get("qe_dftd3_version", 4)),
            "qe_dftd3_threebody": bool(dft_disp.get("qe_dftd3_threebody", True)),
            "siesta_dftd3_use_xc_defaults": bool(
                dft_disp.get(
                    "siesta_dftd3_use_xc_defaults",
                    dft_disp.get("siesta_dftd3_use_xc_functional", True),
                )
            ),
        },
        "structure_file": structure_file,
        "run_dir": run_dir,
        "n_nodes": int(run.get("n_nodes", 1)),
        "kpoints_scf": _int_list3(dft_k.get("scf"), default=(8, 8, 8)),
        "kpoints_nscf": _int_list3(dft_k.get("nscf"), default=(12, 12, 12)),
        "qe_noncolin": bool(dft.get("qe_noncolin", True)),
        "qe_lspinorb": bool(dft.get("qe_lspinorb", True)),
        "qe_disable_symmetry": bool(
            pes_track.get(
                "disable_symmetry",
                dft_reference.get("disable_symmetry", dft.get("qe_disable_symmetry", False)),
            )
        ),
        "thicknesses": [int(x) for x in transport.get("thicknesses", [3, 5, 7, 9, 11])],
        "disorder_strengths": [float(x) for x in transport.get("disorder_strengths", [0.0, 0.2])],
        "n_ensemble": int(transport.get("n_ensemble", 30)),
        "n_jobs": int(transport.get("n_jobs", 1)),
        "mfp_n_layers_z": int(transport.get("mfp_n_layers_z", 5)),
        "mfp_lengths": [int(x) for x in transport.get("mfp_lengths", [3, 5, 7, 9, 11, 13, 15])],
        "transport_axis": str(transport.get("transport_axis", "x")),
        "thickness_axis": str(transport.get("thickness_axis", "z")),
        "transport_n_layers_x": int(transport.get("transport_n_layers_x", 4)),
        "transport_n_layers_y": int(transport.get("transport_n_layers_y", 4)),
        "carrier_density_m3": (
            float(transport.get("carrier_density_m3"))
            if transport.get("carrier_density_m3") is not None
            else None
        ),
        "fermi_velocity_m_per_s": (
            float(transport.get("fermi_velocity_m_per_s"))
            if transport.get("fermi_velocity_m_per_s") is not None
            else None
        ),
        "lead_onsite_eV": float(transport.get("lead_onsite_eV", 0.0)),
        "base_seed": int(transport.get("base_seed", int(project.get("seed", 0)))),
        "transport_backend": str(transport.get("backend", "qsub")),
        "transport_policy": str(transport.get("policy", transport_policy_default)),
        "transport_strict_qsub": bool(transport.get("strict_qsub", True)),
        "transport_walltime": str(transport.get("walltime", "00:30:00")),
        "transport_mpi_np": int(transport.get("mpi_np", 0)),
        "transport_threads": int(transport.get("threads", 0)),
        "transport_cluster_python_exe": str(
            transport.get("cluster_python_exe", cluster.get("cluster_python_exe", "python3"))
        ),
        "transport_autotune": {
            "enabled": bool(transport_autotune.get("enabled", True)),
            "scope": str(transport_autotune.get("scope", "per_queue")).strip().lower() or "per_queue",
            "cache_file": str(
                transport_autotune.get("cache_file", "~/.wtec/transport_autotune.json")
            ).strip() or "~/.wtec/transport_autotune.json",
            "profiles": (
                transport_autotune.get("profiles")
                if isinstance(transport_autotune.get("profiles"), list)
                else [
                    {"mpi_np": 1, "threads": "all"},
                    {"mpi_np": 2, "threads": "half"},
                    {"mpi_np": 4, "threads": "quarter"},
                ]
            ),
            "benchmark_walltime": str(
                transport_autotune.get("benchmark_walltime", "00:08:00")
            ).strip() or "00:08:00",
            "benchmark_n_ensemble": int(transport_autotune.get("benchmark_n_ensemble", 1)),
        },
        "topology": {
            "enabled": bool(topo.get("enabled", True)),
            "backend": str(topo.get("backend", "qsub")),
            "execution_mode": str(topo.get("execution_mode", "per_point_qsub")),
            "strict_qsub": bool(topo.get("strict_qsub", True)),
            "failure_policy": str(topo.get("failure_policy", "strict")),
            "max_concurrent_point_jobs": int(topo.get("max_concurrent_point_jobs", 1)),
            "max_concurrent_variant_dft_jobs": int(topo.get("max_concurrent_variant_dft_jobs", 1)),
            "variant_discovery_glob": variant_glob,
            "coarse_kmesh": _int_list3(topo_k.get("coarse"), default=(20, 20, 20)),
            "refine_kmesh": _int_list3(topo_k.get("refine"), default=(5, 5, 5)),
            "newton_max_iter": int(topo_k.get("newton_max_iter", 50)),
            "gap_threshold_ev": float(topo_k.get("gap_threshold_ev", 0.05)),
            "max_candidates": int(topo_k.get("max_candidates", 64)),
            "dedup_tol": float(topo_k.get("dedup_tol", 0.04)),
            "transport_axis_primary": str(topo.get("transport_axis_primary", "x")),
            "transport_axis_aux": str(topo.get("transport_axis_aux", "z")),
            "n_layers_x": int(topo.get("n_layers_x", int(transport.get("transport_n_layers_x", 4)))),
            "n_layers_y": int(topo.get("n_layers_y", int(transport.get("transport_n_layers_y", 4)))),
            "arc_engine": str(topo.get("arc_engine", "siesta_slab_ldos")),
            "arc_allow_proxy_fallback": bool(topo.get("arc_allow_proxy_fallback", False)),
            "siesta_slab_ldos_autogen": str(
                topo.get("siesta_slab_ldos_autogen", "kwant_proxy")
            ).strip().lower()
            or "kwant_proxy",
            "node_method": str(topo.get("node_method", "wannierberri_flux")),
            "hr_scope": str(topo.get("hr_scope", "per_variant")),
            "caveat_reuse_global_hr_dat": bool(topo.get("caveat_reuse_global_hr_dat", False)),
            "variant_dft_engine": variant_engine_cfg,
            "walltime_per_point": str(topo.get("walltime_per_point", "00:30:00")),
            "cluster_python_exe": str(cluster.get("cluster_python_exe", "python3")),
            "fermi_ev": (None if topo.get("fermi_ev") is None else float(topo.get("fermi_ev"))),
            "score": {
                "w_topo": float(topo_score.get("w_topo", 0.70)),
                "w_transport": float(topo_score.get("w_transport", 0.30)),
            },
            "transport_probe": {
                "enabled": bool(topo_transport_probe.get("enabled", True)),
                "n_ensemble": int(topo_transport_probe.get("n_ensemble", 1)),
                "disorder_strength": float(
                    topo_transport_probe.get("disorder_strength", 0.0)
                ),
                "energy_shift_ev": float(
                    topo_transport_probe.get("energy_shift_ev", 0.0)
                ),
                "thickness_axis": str(
                    topo_transport_probe.get(
                        "thickness_axis",
                        topo.get("transport_axis_aux", "z"),
                    )
                ),
            },
            "point_resource_profile": {
                "node_phase_mpi_np": int(topo_res.get("node_phase_mpi_np", 64)),
                "node_phase_threads": int(topo_res.get("node_phase_threads", 1)),
                "arc_phase_mpi_np": int(topo_res.get("arc_phase_mpi_np", 64)),
                "arc_phase_threads": int(topo_res.get("arc_phase_threads", 1)),
            },
            "hr_grid": {
                "thickness_mapping": str(topo_hr.get("thickness_mapping", "middle_layer_scale")),
                "middle_layer_role": str(topo_hr.get("middle_layer_role", "active")),
                "reference_thickness_uc": (
                    None
                    if topo_hr.get("reference_thickness_uc") is None
                    else int(topo_hr.get("reference_thickness_uc"))
                ),
                "reuse_successful_points": bool(topo_hr.get("reuse_successful_points", True)),
                "max_parallel_hr_points": int(topo_hr.get("max_parallel_hr_points", 1)),
            },
            "tiering": {
                "mode": str(topo_tiering.get("mode", "single")).strip().lower() or "single",
                "refine_top_k_per_thickness": int(
                    topo_tiering.get("refine_top_k_per_thickness", 2)
                ),
                "always_include_pristine": bool(
                    topo_tiering.get("always_include_pristine", True)
                ),
                "selection_metric": str(
                    topo_tiering.get("selection_metric", "S_total")
                ).strip() or "S_total",
                "screen": {
                    "arc_engine": str(topo_tiering_screen.get("arc_engine", "siesta_slab_ldos")).strip().lower() or "siesta_slab_ldos",
                    "node_method": str(topo_tiering_screen.get("node_method", "wannierberri_flux")).strip().lower() or "wannierberri_flux",
                    "coarse_kmesh": _int_list3(
                        topo_tiering_screen.get("coarse_kmesh"),
                        default=(10, 10, 10),
                    ),
                    "refine_kmesh": _int_list3(
                        topo_tiering_screen.get("refine_kmesh"),
                        default=(3, 3, 3),
                    ),
                    "newton_max_iter": int(topo_tiering_screen.get("newton_max_iter", 20)),
                    "max_candidates": int(topo_tiering_screen.get("max_candidates", 24)),
                },
            },
        },
    }
    if remote_workdir:
        cfg["remote_workdir"] = remote_workdir
    if report:
        cfg["report"] = report
    return _normalize_dft_track_config(cfg, base_dir=base)


def _load_run_config(config_path: Path) -> dict[str, Any]:
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(config_path.read_text())
        if not isinstance(data, dict):
            raise click.ClickException(f"JSON config root must be an object: {config_path}")
        return _normalize_dft_track_config(data, base_dir=config_path.parent)
    if suffix == ".toml":
        data = _load_toml_dict(config_path)
        # Master template conversion path.
        if isinstance(data.get("run"), dict) or (
            isinstance(data.get("project"), dict)
            and isinstance(data.get("export"), dict)
            and isinstance(data.get("layers"), list)
        ):
            cfg = _build_cfg_from_master_toml(data, source_path=config_path)
            return _normalize_dft_track_config(cfg, base_dir=config_path.parent)
        # Direct config in TOML (same keys as JSON).
        return _normalize_dft_track_config(data, base_dir=config_path.parent)
    raise click.ClickException(
        f"Unsupported config file type: {config_path}\n"
        "Use .json or .toml."
    )


def _infer_structure_metadata_path(structure_file: Any) -> Path | None:
    if not structure_file:
        return None
    try:
        p = Path(str(structure_file)).expanduser().resolve()
    except Exception:
        return None
    if not p.exists():
        return None
    candidates = [
        p.with_suffix(".meta.json"),
        p.with_name(p.stem + ".meta.json"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _load_transport_results_for_report(run_dir: Path) -> dict[str, Any]:
    candidates = [
        run_dir / "transport" / "primary" / "transport_result.json",
        run_dir / "transport" / "transport_result.json",
    ]
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            payload = json.loads(cand.read_text())
        except Exception:
            continue
        if isinstance(payload, dict):
            tr = payload.get("transport_results")
            if isinstance(tr, dict):
                return tr
    return {}


def _select_zero_disorder_scan(transport_results: dict[str, Any]) -> dict[str, Any] | None:
    scan = transport_results.get("thickness_scan")
    if not isinstance(scan, dict) or not scan:
        return None
    key = None
    for k in scan.keys():
        try:
            if abs(float(k)) < 1e-12:
                key = k
                break
        except Exception:
            continue
    if key is None:
        try:
            key = sorted(scan.keys(), key=lambda x: float(x))[0]
        except Exception:
            key = list(scan.keys())[0]
    payload = scan.get(key)
    return payload if isinstance(payload, dict) else None


def _compute_transport_signature(
    transport_results: dict[str, Any],
    *,
    min_mfp_nm: float,
) -> dict[str, Any]:
    curve = _select_zero_disorder_scan(transport_results)
    if not isinstance(curve, dict):
        return {
            "has_curve": False,
            "has_rho_minimum": False,
            "thinning_reduces_rho": False,
            "mfp_available": False,
            "mfp_huge": False,
            "mfp_nm": None,
            "min_mfp_nm_threshold": float(min_mfp_nm),
            "reason": "missing_zero_disorder_curve",
        }

    t_raw = curve.get("thickness_uc")
    rho_raw = curve.get("rho_mean")
    tm_raw = curve.get("thickness_m")
    if not isinstance(t_raw, list) or not isinstance(rho_raw, list):
        return {
            "has_curve": False,
            "has_rho_minimum": False,
            "thinning_reduces_rho": False,
            "mfp_available": False,
            "mfp_huge": False,
            "mfp_nm": None,
            "min_mfp_nm_threshold": float(min_mfp_nm),
            "reason": "invalid_zero_disorder_curve_shape",
        }

    points: list[tuple[int, float, float | None]] = []
    for idx, (t, rho) in enumerate(zip(t_raw, rho_raw)):
        try:
            t_uc = int(t)
            rv = float(rho)
        except Exception:
            continue
        tm = None
        if isinstance(tm_raw, list) and idx < len(tm_raw):
            try:
                tm = float(tm_raw[idx])
            except Exception:
                tm = None
        points.append((t_uc, rv, tm))

    if len(points) < 2:
        return {
            "has_curve": False,
            "has_rho_minimum": False,
            "thinning_reduces_rho": False,
            "mfp_available": False,
            "mfp_huge": False,
            "mfp_nm": None,
            "min_mfp_nm_threshold": float(min_mfp_nm),
            "reason": "insufficient_transport_points",
        }

    points.sort(key=lambda x: x[0])  # ascending thickness
    thickness_uc = [p[0] for p in points]
    rho_vals = [p[1] for p in points]
    idx_min = min(range(len(rho_vals)), key=lambda i: rho_vals[i])
    has_minimum = (0 < idx_min < (len(rho_vals) - 1))
    thinning_reduces_rho = bool(rho_vals[0] < rho_vals[-1])

    d_min_nm = None
    if points[idx_min][2] is not None:
        d_min_nm = float(points[idx_min][2] * 1e9)

    mfp_payload = transport_results.get("mfp", {})
    mfp_nm = None
    if isinstance(mfp_payload, dict):
        raw = mfp_payload.get("mfp_nm")
        if raw is None:
            raw = mfp_payload.get("mfp_drude_nm")
        if raw is not None:
            try:
                mfp_nm = float(raw)
            except Exception:
                mfp_nm = None
    mfp_available = bool(mfp_nm is not None and mfp_nm > 0.0)
    mfp_huge = bool(mfp_available and mfp_nm >= float(min_mfp_nm))

    return {
        "has_curve": True,
        "n_curve_points": int(len(points)),
        "thickness_uc": thickness_uc,
        "rho_at_thickness_uc": {str(t): float(r) for t, r in zip(thickness_uc, rho_vals)},
        "rho_minimum": {
            "has_minimum": bool(has_minimum),
            "thickness_uc": int(thickness_uc[idx_min]),
            "thickness_nm": d_min_nm,
            "rho": float(rho_vals[idx_min]),
            "index": int(idx_min),
        },
        "has_rho_minimum": bool(has_minimum),
        "thinning_reduces_rho": thinning_reduces_rho,
        "mfp_nm": mfp_nm,
        "mfp_available": mfp_available,
        "mfp_huge": mfp_huge,
        "min_mfp_nm_threshold": float(min_mfp_nm),
    }


def _compute_scientific_validity(
    *,
    cfg: dict[str, Any],
    cp: dict[str, Any],
    outputs: dict[str, Any],
    run_dir: Path,
    topo_summary: dict[str, Any],
) -> dict[str, Any]:
    run_profile = str(cfg.get("run_profile", "strict")).strip().lower() or "strict"
    dft_jobs = outputs.get("dft_jobs") if isinstance(outputs.get("dft_jobs"), dict) else {}
    scf_meta = dft_jobs.get("scf") if isinstance(dft_jobs, dict) else {}
    nscf_meta = dft_jobs.get("nscf") if isinstance(dft_jobs, dict) else {}
    topocfg = cfg.get("topology", {}) if isinstance(cfg.get("topology"), dict) else {}
    report_cfg = cfg.get("report", {}) if isinstance(cfg.get("report"), dict) else {}
    validation_cfg = (
        report_cfg.get("validation", {})
        if isinstance(report_cfg.get("validation"), dict)
        else {}
    )
    min_mfp_nm = float(validation_cfg.get("min_mfp_nm", 100.0))
    require_rho_minimum = bool(validation_cfg.get("require_rho_minimum", True))
    require_thinning_reduction = bool(validation_cfg.get("require_thinning_reduction", True))
    require_huge_mfp = bool(validation_cfg.get("require_huge_mfp", True))
    transport_results = _load_transport_results_for_report(run_dir)
    transport_signature = _compute_transport_signature(
        transport_results,
        min_mfp_nm=min_mfp_nm,
    )

    fresh_dft_per_variant = True
    if isinstance(scf_meta, dict) and str(scf_meta.get("reason", "")).startswith("configured_hr_dat_path"):
        fresh_dft_per_variant = False
    if isinstance(nscf_meta, dict) and str(nscf_meta.get("reason", "")).startswith("configured_hr_dat_path"):
        fresh_dft_per_variant = False
    if bool(topocfg.get("caveat_reuse_global_hr_dat", False)):
        fresh_dft_per_variant = False

    sampling_pass = (
        isinstance(cfg.get("thicknesses"), list)
        and len(cfg.get("thicknesses", [])) >= 5
        and isinstance(cfg.get("mfp_lengths"), list)
        and len(cfg.get("mfp_lengths", [])) >= 7
        and int(cfg.get("n_ensemble", 0)) >= 30
    )

    strain_pass = True
    max_active_abs_strain = None
    md_path = _infer_structure_metadata_path(cfg.get("structure_file"))
    if md_path is not None:
        try:
            meta = json.loads(md_path.read_text())
        except Exception:
            meta = {}
        layers = meta.get("layers", []) if isinstance(meta, dict) else []
        active_strains: list[float] = []
        if isinstance(layers, list):
            for layer in layers:
                if not isinstance(layer, dict):
                    continue
                if str(layer.get("role", "")).strip().lower() != "active":
                    continue
                for key in ("strain_a_percent", "strain_b_percent"):
                    if key in layer:
                        try:
                            active_strains.append(abs(float(layer.get(key))))
                        except Exception:
                            continue
        if active_strains:
            max_active_abs_strain = max(active_strains)
            strain_pass = max_active_abs_strain <= 6.0

    proxy_used = False
    topology_complete = (
        isinstance(topo_summary, dict)
        and str(topo_summary.get("status", "")).strip().lower() == "ok"
        and int(topo_summary.get("n_failed_points", 0) or 0) == 0
        and int(topo_summary.get("n_partial_points", 0) or 0) == 0
    )
    topo_json = run_dir / "topology" / "topology_deviation.json"
    if topo_json.exists():
        try:
            topo_payload = json.loads(topo_json.read_text())
        except Exception:
            topo_payload = {}
        results = topo_payload.get("results", []) if isinstance(topo_payload, dict) else []
        if isinstance(results, list):
            for row in results:
                if not isinstance(row, dict):
                    continue
                if str(row.get("status", "")).strip().lower() != "ok":
                    topology_complete = False
                node_req = str(
                    row.get("node_method_requested", row.get("node_method", ""))
                ).strip().lower()
                node_eff = str(row.get("node_method_effective", "")).strip().lower()
                if node_req == "proxy" or node_eff == "proxy":
                    proxy_used = True
                    break
                if node_req == "wannierberri_flux" and node_eff != "wannierberri_flux":
                    proxy_used = True
                    break
                arc_engine = str(row.get("arc_engine", "")).strip().lower()
                arc_req = str(row.get("arc_engine_requested", "")).strip().lower()
                arc_source = str(row.get("arc_source_engine", "")).strip().lower()
                arc_source_kind = str(row.get("arc_source_kind", "")).strip().lower()
                node_stat = str(row.get("raw_node_status", "")).strip().lower()
                arc_stat = str(row.get("raw_arc_status", "")).strip().lower()
                if node_stat and node_stat != "ok":
                    topology_complete = False
                if arc_stat and arc_stat != "ok":
                    topology_complete = False
                if arc_engine.endswith("_proxy") or arc_req in {"kwant"}:
                    proxy_used = True
                    break
                if "proxy" in arc_source or "kwant" in arc_source or "proxy" in arc_source_kind:
                    proxy_used = True
                    break
        point_manifest = topo_payload.get("point_manifest", []) if isinstance(topo_payload, dict) else []
        if isinstance(point_manifest, list):
            for row in point_manifest:
                if not isinstance(row, dict):
                    continue
                if str(row.get("reason", "")).strip().lower() == "caveat_reuse_global_hr_dat":
                    fresh_dft_per_variant = False
                    break

    transport_signature_pass = bool(transport_signature.get("has_curve", False))
    if require_rho_minimum:
        transport_signature_pass = transport_signature_pass and bool(
            transport_signature.get("has_rho_minimum", False)
        )
    if require_thinning_reduction:
        transport_signature_pass = transport_signature_pass and bool(
            transport_signature.get("thinning_reduces_rho", False)
        )
    if require_huge_mfp:
        transport_signature_pass = transport_signature_pass and bool(
            transport_signature.get("mfp_huge", False)
        )
    else:
        transport_signature_pass = transport_signature_pass and bool(
            transport_signature.get("mfp_available", False)
        )

    stage = str(cp.get("stage", "")).strip().upper()
    if run_profile == "smoke":
        verdict = "smoke_only"
    elif stage != "DONE":
        verdict = "invalid"
    elif (
        fresh_dft_per_variant
        and (not proxy_used)
        and sampling_pass
        and strain_pass
        and topology_complete
        and transport_signature_pass
    ):
        verdict = "valid"
    else:
        verdict = "caveat"

    return {
        "run_profile": run_profile,
        "proxy_used": bool(proxy_used),
        "fresh_dft_per_variant": bool(fresh_dft_per_variant),
        "sampling_pass": bool(sampling_pass),
        "strain_pass": bool(strain_pass),
        "topology_complete": bool(topology_complete),
        "transport_signature_pass": bool(transport_signature_pass),
        "transport_signature": transport_signature,
        "max_active_abs_strain_percent": max_active_abs_strain,
        "final_verdict": verdict,
    }


def _write_run_report(cfg: dict[str, Any]) -> tuple[Path, Path] | None:
    run_dir = Path(str(cfg.get("run_dir", ".")).strip() or ".").expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    report_cfg = cfg.get("report", {}) if isinstance(cfg.get("report"), dict) else {}
    if report_cfg and not bool(report_cfg.get("enabled", True)):
        return None

    md_rel = str(report_cfg.get("markdown", "report/wtec_report.md")) if isinstance(report_cfg, dict) else "report/wtec_report.md"
    js_rel = str(report_cfg.get("json", "report/wtec_report.json")) if isinstance(report_cfg, dict) else "report/wtec_report.json"
    md_path = (run_dir / md_rel).resolve()
    js_path = (run_dir / js_rel).resolve()
    md_path.parent.mkdir(parents=True, exist_ok=True)
    js_path.parent.mkdir(parents=True, exist_ok=True)

    cp_file = _checkpoint_file_for_cfg(cfg)
    cp = {}
    if cp_file.exists():
        try:
            cp = json.loads(cp_file.read_text())
        except Exception:
            cp = {}
    outputs = cp.get("outputs", {}) if isinstance(cp.get("outputs"), dict) else {}

    topo_summary: dict[str, Any] = {}
    topo_json = run_dir / "topology" / "topology_deviation.json"
    if topo_json.exists():
        try:
            topo_payload = json.loads(topo_json.read_text())
            topo_summary = topo_payload.get("summary", {}) if isinstance(topo_payload, dict) else {}
        except Exception:
            topo_summary = {}

    transport_compare_summary: dict[str, Any] = {}
    transport_compare_json = run_dir / "report" / "transport_compare.json"
    if transport_compare_json.exists():
        try:
            tc_payload = json.loads(transport_compare_json.read_text())
            if isinstance(tc_payload, dict):
                transport_compare_summary = tc_payload
        except Exception:
            transport_compare_summary = {}

    fermi_ev = outputs.get("fermi_ev")
    if fermi_ev is None and isinstance(topo_summary, dict):
        fermi_ev = topo_summary.get("fermi_ev")

    artifact_candidates = {
        "rho_plot": run_dir / "rho_vs_thickness.pdf",
        "topology_csv": run_dir / "topology" / "topology_deviation.csv",
        "topology_json": run_dir / "topology" / "topology_deviation.json",
        "point_manifest_csv": run_dir / "topology" / "topology_point_manifest.csv",
        "point_jobs_csv": run_dir / "topology" / "topology_point_jobs.csv",
        "transport_compare_json": transport_compare_json,
    }
    delta_h_artifact_raw = outputs.get("delta_h_artifact")
    if isinstance(delta_h_artifact_raw, str) and delta_h_artifact_raw.strip():
        artifact_candidates["delta_h_artifact_json"] = Path(delta_h_artifact_raw).expanduser().resolve()
    artifacts = {
        key: str(path.resolve())
        for key, path in artifact_candidates.items()
        if path.exists()
    }
    scientific_validity = _compute_scientific_validity(
        cfg=cfg,
        cp=cp,
        outputs=outputs,
        run_dir=run_dir,
        topo_summary=topo_summary,
    )
    transport_signature = (
        scientific_validity.get("transport_signature", {})
        if isinstance(scientific_validity, dict)
        else {}
    )

    payload = {
        "run_name": cfg.get("name"),
        "stage": cp.get("stage"),
        "material": cfg.get("material"),
        "structure_file": cfg.get("structure_file"),
        "run_dir": str(run_dir),
        "checkpoint": str(cp_file),
        "dft_jobs": outputs.get("dft_jobs"),
        "wannier_job": outputs.get("wannier_job"),
        "fermi_ev": fermi_ev,
        "hr_dat": outputs.get("hr_dat"),
        "transport_results": outputs.get("transport_results"),
        "transport_compare": outputs.get("transport_compare"),
        "dft_anchor_transfer": outputs.get("dft_anchor_transfer"),
        "topology_results": outputs.get("topology_results"),
        "topology_summary": topo_summary,
        "transport_compare_summary": transport_compare_summary,
        "scientific_validity": scientific_validity,
        "transport_signature": transport_signature,
        "artifacts": artifacts,
        "generated_at_epoch": int(time.time()),
    }
    js_path.write_text(json.dumps(payload, indent=2))

    lines = [
        "# wtec run report",
        "",
        f"- run_name: `{payload['run_name']}`",
        f"- stage: `{payload['stage']}`",
        f"- material: `{payload['material']}`",
        f"- structure_file: `{payload['structure_file']}`",
        f"- run_dir: `{payload['run_dir']}`",
        f"- fermi_ev: `{payload['fermi_ev']}`",
        "",
        "## Scientific Validity",
        f"- scientific_validity: `{payload['scientific_validity']}`",
        f"- transport_signature: `{payload['transport_signature']}`",
        "",
        "## Cluster jobs",
        f"- dft_jobs: `{payload['dft_jobs']}`",
        f"- wannier_job: `{payload['wannier_job']}`",
        "",
        "## Topology",
        f"- topology_results: `{payload['topology_results']}`",
        f"- topology_summary: `{payload['topology_summary']}`",
        f"- dft_anchor_transfer: `{payload['dft_anchor_transfer']}`",
        "",
        "## Transport Compare",
        f"- transport_compare: `{payload['transport_compare']}`",
        f"- transport_compare_summary: `{payload['transport_compare_summary']}`",
        "",
        "## Artifacts",
    ]
    for key in (
        "rho_plot",
        "topology_csv",
        "topology_json",
        "point_manifest_csv",
        "point_jobs_csv",
        "transport_compare_json",
        "delta_h_artifact_json",
    ):
        if key in payload["artifacts"]:
            lines.append(f"- {key}: `{payload['artifacts'][key]}`")
    if not payload["artifacts"]:
        lines.append("- (none)")
    md_path.write_text("\n".join(lines) + "\n")
    return md_path, js_path


# ---------------------------------------------------------------------------
# wtec run
# ---------------------------------------------------------------------------

@main.command()
@click.argument("config_file", required=False, type=click.Path(dir_okay=False))
@click.option("--resume", is_flag=True, help="Resume from checkpoint.")
@click.option("--stage", default=None, help="Run only this stage (e.g. 'dft', 'transport').")
@click.option("--cluster-host", default=None, help="Override TOPOSLAB_CLUSTER_HOST for this run only.")
@click.option("--cluster-port", type=int, default=None, help="Override TOPOSLAB_CLUSTER_PORT for this run only.")
@click.option("--cluster-user", default=None, help="Override TOPOSLAB_CLUSTER_USER for this run only.")
@click.option("--cluster-pass", default=None, help="Override TOPOSLAB_CLUSTER_PASS for this run only.")
@click.option("--cluster-key", default=None, help="Override TOPOSLAB_CLUSTER_KEY for this run only.")
@click.option("--remote-workdir", default=None, help="Override TOPOSLAB_REMOTE_WORKDIR for this run only.")
@click.option("--mpi-cores", type=int, default=None, help="Override TOPOSLAB_MPI_CORES for this run only.")
@click.option(
    "--mpi-cores-by-queue",
    default=None,
    help="Override TOPOSLAB_MPI_CORES_BY_QUEUE for this run only (e.g. g4:64,g1:16).",
)
@click.option("--qe-pseudo-dir", default=None, help="Override TOPOSLAB_QE_PSEUDO_DIR for this run only.")
@click.option("--siesta-pseudo-dir", default=None, help="Override TOPOSLAB_SIESTA_PSEUDO_DIR for this run only.")
@click.option("--vasp-pseudo-dir", default=None, help="Override TOPOSLAB_VASP_PSEUDO_DIR for this run only.")
@click.option("--abacus-pseudo-dir", default=None, help="Override TOPOSLAB_ABACUS_PSEUDO_DIR for this run only.")
@click.option("--abacus-orbital-dir", default=None, help="Override TOPOSLAB_ABACUS_ORBITAL_DIR for this run only.")
@click.option("--omp-threads", type=int, default=None, help="Override TOPOSLAB_OMP_THREADS for this run only.")
@click.option(
    "--cluster-modules",
    default=None,
    help="Override TOPOSLAB_CLUSTER_MODULES for this run only (comma-separated).",
)
@click.option(
    "--cluster-bin-dirs",
    default=None,
    help="Override TOPOSLAB_CLUSTER_BIN_DIRS for this run only (comma-separated absolute paths).",
)
@click.option(
    "--interactive-cluster/--no-interactive-cluster",
    default=True,
    show_default=True,
    help="Prompt for missing cluster connection fields at runtime when cluster stages are requested.",
)
@click.option(
    "--live-log/--no-live-log",
    default=True,
    help="Stream remote stage logs in terminal while jobs run.",
)
@click.option(
    "--log-poll-interval",
    type=int,
    default=5,
    show_default=True,
    help="Seconds between scheduler/log polling during run.",
)
@click.option(
    "--stale-log-seconds",
    type=int,
    default=300,
    show_default=True,
    help="Warn if job is RUNNING but watched logs do not grow for this long.",
)
def run(
    config_file: str | None,
    resume: bool,
    stage: str | None,
    cluster_host: str | None,
    cluster_port: int | None,
    cluster_user: str | None,
    cluster_pass: str | None,
    cluster_key: str | None,
    remote_workdir: str | None,
    mpi_cores: int | None,
    mpi_cores_by_queue: str | None,
    qe_pseudo_dir: str | None,
    siesta_pseudo_dir: str | None,
    vasp_pseudo_dir: str | None,
    abacus_pseudo_dir: str | None,
    abacus_orbital_dir: str | None,
    omp_threads: int | None,
    cluster_modules: str | None,
    cluster_bin_dirs: str | None,
    interactive_cluster: bool,
    live_log: bool,
    log_poll_interval: int,
    stale_log_seconds: int,
) -> None:
    """Run a full wtec workflow from a JSON config file.

    \b
    CONFIG_FILE: path to a JSON workflow configuration file.

    \b
    Example:
        wtec run
        wtec run myrun.json
        wtec run wtec_project.toml
        wtec run myrun.json --resume
        wtec run myrun.json --stage transport
    """
    from wtec.workflow.orchestrator import TopoSlabWorkflow

    config_path = _resolve_run_config_path(config_file)
    _load_runtime_dotenv(str(config_path))

    if log_poll_interval <= 0:
        raise click.UsageError("--log-poll-interval must be > 0")
    if stale_log_seconds <= 0:
        raise click.UsageError("--stale-log-seconds must be > 0")

    cfg = _load_run_config(config_path)

    runtime_env_updates = _collect_env_updates(
        cluster_host=cluster_host,
        cluster_port=cluster_port,
        cluster_user=cluster_user,
        cluster_pass=cluster_pass,
        cluster_key=cluster_key,
        mp_api_key=None,
        remote_workdir=remote_workdir,
        mpi_cores=mpi_cores,
        mpi_cores_by_queue=mpi_cores_by_queue,
        pbs_queue=None,
        pbs_queue_priority=None,
        qe_pseudo_dir=qe_pseudo_dir,
        siesta_pseudo_dir=siesta_pseudo_dir,
        vasp_pseudo_dir=vasp_pseudo_dir,
        abacus_pseudo_dir=abacus_pseudo_dir,
        abacus_orbital_dir=abacus_orbital_dir,
        qe_pseudo_source_dir=None,
        omp_threads=omp_threads,
        cluster_modules=cluster_modules,
        cluster_bin_dirs=cluster_bin_dirs,
        qe_source_dir=None,
        siesta_source_dir=None,
        abacus_source_dir=None,
        wannier90_source_dir=None,
        cluster_build_jobs=None,
    )
    runtime_env_updates = _merge_runtime_cluster_interactive(
        cfg=cfg,
        resume=resume,
        stage=stage,
        runtime_env_updates=runtime_env_updates,
        interactive_cluster=interactive_cluster,
    )
    if runtime_env_updates:
        _apply_env_updates_to_process(runtime_env_updates)
        shown_keys = sorted(k for k in runtime_env_updates if k != "TOPOSLAB_CLUSTER_PASS")
        if "TOPOSLAB_CLUSTER_PASS" in runtime_env_updates:
            shown_keys.append("TOPOSLAB_CLUSTER_PASS(***hidden***)")
        click.echo(
            click.style(
                "[runtime] applied cluster argument overrides: " + ", ".join(shown_keys),
                fg="cyan",
            )
        )

    if not cfg.get("run_dir"):
        init_run_dir = _default_run_dir_from_init_state()
        if init_run_dir:
            cfg["run_dir"] = init_run_dir
            click.echo(
                click.style(
                    f"[runtime] run_dir not set; using init directory: {init_run_dir}",
                    fg="cyan",
                )
            )
    cfg["_runtime_live_log"] = bool(live_log)
    cfg["_runtime_log_poll_interval"] = int(log_poll_interval)
    cfg["_runtime_stale_log_seconds"] = int(stale_log_seconds)
    click.echo(click.style("[preflight] validating run configuration...", fg="cyan"))
    _run_preflight(cfg, resume=resume, stage=stage)
    wf = TopoSlabWorkflow.from_config(cfg)

    result: Any = None
    if resume:
        result = wf.resume()
    elif stage:
        result = wf.run_stage(stage)
    else:
        result = wf.run_full()

    report_paths = _write_run_report(cfg)
    if report_paths is not None:
        md, js = report_paths
        click.echo(click.style(f"[report] markdown: {md}", fg="green"))
        click.echo(click.style(f"[report] json: {js}", fg="green"))


# ---------------------------------------------------------------------------
# wtec slab-gen
# ---------------------------------------------------------------------------

@main.command("slab-gen")
@click.argument(
    "template_file",
    required=False,
    default=None,
    type=click.Path(dir_okay=False),
)
@click.option(
    "--output-dir",
    default=None,
    help="Override [project].output_dir from template.",
)
def slab_gen(template_file: str | None, output_dir: str | None) -> None:
    """Generate a stacked slab CIF from TOML template."""
    from wtec.slab import load_slab_template, generate_slab_from_template

    if template_file:
        tpl_path = Path(template_file).expanduser().resolve()
        if not tpl_path.exists():
            raise click.ClickException(f"Template file not found: {tpl_path}")
    else:
        default_tpl = _default_project_template_path(required=True)
        assert default_tpl is not None
        tpl_path = default_tpl

    _load_runtime_dotenv(str(tpl_path))

    try:
        tpl = load_slab_template(str(tpl_path))
        out = generate_slab_from_template(tpl, output_dir_override=output_dir)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(click.style("✓ Slab generated", fg="green", bold=True))
    click.echo(f"  Project: {tpl.project.name}")
    click.echo(f"  Template: {tpl_path}")
    click.echo(f"  CIF: {out['cif_path']}")
    click.echo(f"  Metadata: {out['metadata_path']}")
    click.echo(f"  Atoms: {out['atoms']}  Formula: {out['formula']}")


# ---------------------------------------------------------------------------
# wtec slab
# ---------------------------------------------------------------------------

@main.command("slab")
@click.argument("cif_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--metadata",
    "metadata_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Optional metadata JSON; auto-detected when omitted.",
)
@click.option(
    "--rows",
    type=int,
    default=24,
    show_default=True,
    help="ASCII profile rows.",
)
@click.option(
    "--width",
    type=int,
    default=48,
    show_default=True,
    help="ASCII profile width.",
)
def slab(cif_file: str, metadata_file: str | None, rows: int, width: int) -> None:
    """Show slab geometry/defect summary and ASCII cross-section."""
    from wtec.slab import render_slab_report

    if rows <= 0:
        raise click.UsageError("--rows must be > 0")
    if width <= 0:
        raise click.UsageError("--width must be > 0")
    try:
        text = render_slab_report(
            cif_file,
            metadata_path=metadata_file,
            rows=rows,
            width=width,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(text)


# ---------------------------------------------------------------------------
# wtec status
# ---------------------------------------------------------------------------

@main.command()
@click.option("--job-id", default=None, help="Check status of a specific PBS job.")
@click.option("--all", "show_all", is_flag=True, help="Show all tracked jobs.")
def status(job_id: str | None, show_all: bool) -> None:
    """Check status of running cluster jobs."""
    from wtec.config.cluster import ClusterConfig
    from wtec.cluster.submit import JobManager
    from wtec.cluster.ssh import open_ssh

    cfg = ClusterConfig.from_env()
    with open_ssh(cfg) as ssh:
        mgr = JobManager(ssh)
        if job_id:
            details = mgr.status_details(job_id)
            click.echo(
                "Job "
                f"{job_id}: {details['status']} "
                f"(scheduler_state={details.get('scheduler_state')}, "
                f"exit_code={details.get('exit_code')}, "
                f"source={details.get('source')})"
            )
        elif show_all:
            checkpoint_dir = Path.home() / ".wtec" / "checkpoints"
            for f in sorted(checkpoint_dir.glob("*.json")):
                data = json.loads(f.read_text())
                jid = data.get("last_job_id", "—")
                stage = data.get("stage", "—")
                if jid != "—":
                    details = mgr.status_details(str(jid))
                    status = details["status"]
                    click.echo(
                        f"  {f.stem}: stage={stage}  job={jid}  "
                        f"status={status} ({details.get('scheduler_state')})"
                    )
                else:
                    click.echo(f"  {f.stem}: stage={stage}  job=—")
        else:
            click.echo("Pass --job-id JOB_ID or --all")


# ---------------------------------------------------------------------------
# wtec smoke
# ---------------------------------------------------------------------------

@main.command("smoke")
@click.option(
    "--feature",
    "features",
    multiple=True,
    type=click.Choice(["mpi", "qe", "wannier", "kwant", "berry"]),
    help="Feature smoke test to run. Repeatable. Default: run all.",
)
@click.option("--np", "mpi_ranks", type=int, default=1, show_default=True,
              help="MPI ranks per smoke job (mpirun -np).")
@click.option("--n-nodes", type=int, default=1, show_default=True,
              help="PBS nodes for smoke jobs.")
@click.option("--walltime", default="00:05:00", show_default=True,
              help="PBS walltime for smoke jobs.")
@click.option("--queue", default=None, help="Override PBS queue for smoke jobs.")
@click.option("--python-exe", default="python3", show_default=True,
              help="Remote Python executable for Python feature smoke tests.")
@click.option(
    "--prepare-python",
    is_flag=True,
    default=False,
    help="Install missing remote Python packages (kwant/wannierberri) before submission.",
)
@click.option(
    "--keep-going",
    is_flag=True,
    default=False,
    help="Continue running remaining smoke tests if one fails.",
)
def smoke(
    features: tuple[str, ...],
    mpi_ranks: int,
    n_nodes: int,
    walltime: str,
    queue: str | None,
    python_exe: str,
    prepare_python: bool,
    keep_going: bool,
) -> None:
    """Run short real qsub smoke jobs using mpirun backend only."""
    from wtec.config.cluster import ClusterConfig
    from wtec.cluster.ssh import open_ssh
    from wtec.cluster.submit import JobManager
    from wtec.cluster.pbs import PBSJobConfig, generate_script
    from wtec.cluster.mpi import MPIConfig, build_command

    if mpi_ranks <= 0:
        raise click.UsageError("--np must be > 0")
    if n_nodes <= 0:
        raise click.UsageError("--n-nodes must be > 0")

    selected = list(features) if features else ["mpi", "qe", "wannier", "kwant", "berry"]
    cluster_cfg = ClusterConfig.from_env()

    def _thread_env() -> dict[str, str]:
        env: dict[str, str] = {}
        if cluster_cfg.bin_dirs:
            env["PATH"] = ":".join(cluster_cfg.bin_dirs) + ":$PATH"
        if cluster_cfg.omp_threads is not None:
            val = str(cluster_cfg.omp_threads)
            env.update(
                {
                    "OMP_NUM_THREADS": val,
                    "MKL_NUM_THREADS": val,
                    "OPENBLAS_NUM_THREADS": val,
                    "NUMEXPR_NUM_THREADS": val,
                }
            )
        return env

    def _python_import_cmd(module_name: str, label: str) -> str:
        code = (
            "from mpi4py import MPI; "
            f"import {module_name}; "
            "c=MPI.COMM_WORLD; "
            f"print('{label}_ok', c.Get_rank(), c.Get_size(), "
            f"getattr({module_name}, '__version__', 'unknown'))"
        )
        return build_command(
            python_exe,
            extra_args=f"-c {shlex.quote(code)}",
            mpi=MPIConfig(n_cores=mpi_ranks, n_pool=1),
        )

    def _mpi_probe_cmd() -> str:
        code = (
            "from mpi4py import MPI; import socket; "
            "c=MPI.COMM_WORLD; "
            "print('mpi_ok', c.Get_rank(), c.Get_size(), socket.gethostname())"
        )
        return build_command(
            python_exe,
            extra_args=f"-c {shlex.quote(code)}",
            mpi=MPIConfig(n_cores=mpi_ranks, n_pool=1),
        )

    def _remote_importable(jm: JobManager, module_name: str) -> bool:
        check_cmd = (
            f"{shlex.quote(python_exe)} -c "
            f"{shlex.quote(f'import {module_name}; print({module_name}.__name__)')}"
        )
        rc, _, _ = jm._ssh.run(check_cmd, check=False)
        return rc == 0

    def _remote_py_mm(jm: JobManager) -> tuple[int, int]:
        code = "import sys; print(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))"
        rc, out, _ = jm._ssh.run(
            f"{shlex.quote(python_exe)} -c "
            f"{shlex.quote(code)}",
            check=False,
        )
        if rc != 0:
            return (3, 10)
        raw = out.strip()
        try:
            a, b = raw.split(".", 1)
            return int(a), int(b)
        except Exception:
            return (3, 10)

    def _remote_run_install(jm: JobManager, cmd: str) -> None:
        rc, out, err = jm._ssh.run(cmd, check=False)
        if rc != 0:
            merged = (out + "\n" + err).strip()
            raise RuntimeError(f"Remote install command failed:\n{cmd}\n{merged}")

    def _remote_prepare_kwant(jm: JobManager) -> None:
        if _remote_importable(jm, "kwant"):
            return
        click.echo("  Preparing remote Python for kwant (numpy<2 + source build)")
        pyq = shlex.quote(python_exe)
        _remote_run_install(
            jm,
            f"{pyq} -m pip install --user --upgrade --force-reinstall "
            "'numpy<2' 'scipy<1.14'",
        )
        _remote_run_install(
            jm,
            f"{pyq} -m pip install --user --upgrade meson-python ninja cython setuptools-scm",
        )
        _remote_run_install(
            jm,
            f"{pyq} -m pip install --user --force-reinstall --no-deps "
            "--no-build-isolation --no-binary=:all: --no-cache-dir kwant",
        )
        if not _remote_importable(jm, "kwant"):
            raise RuntimeError("kwant is still not importable on remote python after install.")

    def _remote_prepare_berry(jm: JobManager) -> None:
        if _remote_importable(jm, "wannierberri"):
            return
        pyq = shlex.quote(python_exe)
        py_mm = _remote_py_mm(jm)
        wb_spec = "wannierberri==1.0.1"
        click.echo(f"  Preparing remote Python for berry ({wb_spec})")
        _remote_run_install(
            jm,
            f"{pyq} -m pip install --user --upgrade --force-reinstall "
            "'numpy<2' 'scipy<1.14'",
        )
        _remote_run_install(
            jm,
            f"{pyq} -m pip install --user --upgrade 'ray>=2.10' irrep",
        )
        _remote_run_install(
            jm,
            f"{pyq} -m pip install --user --upgrade --force-reinstall {shlex.quote(wb_spec)}",
        )
        if not _remote_importable(jm, "wannierberri"):
            raise RuntimeError("wannierberri is still not importable on remote python after install.")

    # Feature command builders and command prerequisites.
    feature_specs: dict[str, dict[str, object]] = {
        "mpi": {
            "required_cmds": ["qsub", "qstat", "mpirun", python_exe],
            "command": _mpi_probe_cmd,
        },
        "qe": {
            "required_cmds": ["qsub", "qstat", "mpirun", "pw.x"],
            "command": lambda: build_command(
                "bash",
                extra_args="-lc " + shlex.quote(
                    "pw.x -h >/dev/null 2>&1 || pw.x --help >/dev/null 2>&1"
                ),
                mpi=MPIConfig(n_cores=mpi_ranks, n_pool=1),
            ),
        },
        "wannier": {
            "required_cmds": ["qsub", "qstat", "mpirun", "wannier90.x", "pw2wannier90.x"],
            "command": lambda: build_command(
                "bash",
                extra_args="-lc " + shlex.quote(
                    "(wannier90.x -h >/dev/null 2>&1 || wannier90.x --help >/dev/null 2>&1) "
                    "&& (pw2wannier90.x -h >/dev/null 2>&1 || pw2wannier90.x --help >/dev/null 2>&1)"
                ),
                mpi=MPIConfig(n_cores=mpi_ranks, n_pool=1),
            ),
        },
        "kwant": {
            "required_cmds": ["qsub", "qstat", "mpirun", python_exe],
            "command": lambda: _python_import_cmd("kwant", "kwant"),
        },
        "berry": {
            "required_cmds": ["qsub", "qstat", "mpirun", python_exe],
            "command": lambda: _python_import_cmd("wannierberri", "berry"),
        },
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    local_root = Path.home() / ".wtec" / "runs" / "smoke" / ts
    local_root.mkdir(parents=True, exist_ok=True)

    with open_ssh(cluster_cfg) as ssh:
        jm = JobManager(ssh)
        queue_used = jm.resolve_queue(queue or cluster_cfg.pbs_queue, fallback_order=cluster_cfg.pbs_queue_priority)
        cores_per_node = cluster_cfg.cores_for_queue(queue_used)
        max_ranks = n_nodes * cores_per_node
        if mpi_ranks > max_ranks:
            raise click.UsageError(
                f"--np={mpi_ranks} exceeds n_nodes*cores_per_node={max_ranks} "
                f"(queue={queue_used}, n_nodes={n_nodes}, cores_per_node={cores_per_node})"
            )

        remote_root = f"{cluster_cfg.remote_workdir.rstrip('/')}/smoke/{ts}"
        ssh.mkdir_p(remote_root)

        click.echo(
            click.style(
                f"[smoke] queue={queue_used} nodes={n_nodes} ppn={cores_per_node} np={mpi_ranks}",
                fg="cyan",
            )
        )

        failures: list[str] = []
        for feature in selected:
            spec = feature_specs[feature]
            click.echo(click.style(f"\n[smoke] {feature}", bold=True))
            required_cmds = list(spec["required_cmds"])  # type: ignore[index]
            try:
                if prepare_python and feature == "kwant":
                    _remote_prepare_kwant(jm)
                if prepare_python and feature == "berry":
                    _remote_prepare_berry(jm)
                jm.ensure_remote_commands(
                    required_cmds,
                    modules=cluster_cfg.modules,
                    bin_dirs=cluster_cfg.bin_dirs,
                )
                job_name = f"wts_{feature}_{ts[-6:]}"
                remote_dir = f"{remote_root}/{feature}"
                marker = f"smoke_{feature}.ok"
                command = spec["command"]()  # type: ignore[operator]
                script_cfg = PBSJobConfig(
                    job_name=job_name,
                    n_nodes=n_nodes,
                    n_cores_per_node=cores_per_node,
                    walltime=walltime,
                    queue=queue_used,
                    work_dir=remote_dir,
                    modules=cluster_cfg.modules,
                    env_vars=_thread_env(),
                )
                script = generate_script(script_cfg, [command, f"echo ok > {marker}"])
                feature_local = local_root / feature
                feature_local.mkdir(parents=True, exist_ok=True)
                meta = jm.submit_and_wait(
                    script,
                    remote_dir=remote_dir,
                    local_dir=feature_local,
                    retrieve_patterns=[f"{job_name}.log", marker],
                    script_name=f"{job_name}.pbs",
                    expected_local_outputs=[marker],
                    queue_used=queue_used,
                    poll_interval=10,
                    verbose=True,
                )
                click.echo(
                    click.style(
                        f"  ✓ {feature} job_id={meta['job_id']} queue={meta.get('queue')}",
                        fg="green",
                    )
                )
            except Exception as exc:
                msg = f"{feature}: {exc}"
                failures.append(msg)
                click.echo(click.style(f"  ✗ {msg}", fg="red"))
                if not keep_going:
                    break

    if failures:
        raise click.ClickException("Smoke failures:\n- " + "\n- ".join(failures))

    click.echo(click.style(f"\n✓ Smoke tests completed ({', '.join(selected)})", fg="green", bold=True))
    click.echo(f"  Local logs: {local_root}")


# ---------------------------------------------------------------------------
# wtec defect  (sub-group)
# ---------------------------------------------------------------------------

def _zeroed_interfaces(interfaces):
    from wtec.slab.template import InterfaceSpec

    out = []
    for iface in interfaces:
        out.append(
            InterfaceSpec(
                between=tuple(iface.between),
                vacancy_mode="none",
                vacancy_window_angstrom=float(iface.vacancy_window_angstrom),
                vacancy_seed=iface.vacancy_seed,
                vacancy_counts_by_element={k: 0 for k in dict(iface.vacancy_counts_by_element).keys()},
                substitutions=[
                    replace(sub, count=0) for sub in list(iface.substitutions)
                ],
            )
        )
    return out


def _defect_requested_count(interfaces) -> int:
    count = 0
    for iface in interfaces:
        counts = iface.vacancy_counts_by_element if isinstance(iface.vacancy_counts_by_element, dict) else {}
        count += sum(max(0, int(v)) for v in counts.values())
        for sub in list(iface.substitutions):
            count += max(0, int(sub.count))
    return count


def _ensure_minimum_defect_interfaces(interfaces, *, min_vacancies_total: int):
    from wtec.slab.template import InterfaceSpec

    if _defect_requested_count(interfaces) >= max(0, int(min_vacancies_total)):
        return interfaces
    if not interfaces:
        return interfaces
    first = interfaces[0]
    counts = dict(first.vacancy_counts_by_element)
    counts["O"] = max(int(counts.get("O", 0)), max(1, int(min_vacancies_total)))
    patched_first = InterfaceSpec(
        between=tuple(first.between),
        vacancy_mode="random_interface",
        vacancy_window_angstrom=float(first.vacancy_window_angstrom),
        vacancy_seed=first.vacancy_seed,
        vacancy_counts_by_element=counts,
        substitutions=list(first.substitutions),
    )
    return [patched_first, *list(interfaces[1:])]


def _generate_default_defect_variants(template_path: Path) -> list[dict[str, Any]]:
    from wtec.slab import generate_slab_from_template, load_slab_template

    _load_runtime_dotenv(str(template_path))
    tpl = load_slab_template(str(template_path))
    raw = _load_toml_dict(template_path)
    defect_cfg = raw.get("defect", {}) if isinstance(raw.get("defect"), dict) else {}

    out_dir = str(defect_cfg.get("output_dir", "slab_variants")).strip() or "slab_variants"
    output_dir_override = str((template_path.parent / out_dir).resolve())
    pristine_suffix = str(defect_cfg.get("pristine_suffix", "pristine")).strip() or "pristine"
    defect_suffix = str(defect_cfg.get("defect_suffix", "defect")).strip() or "defect"
    gen_pristine = bool(defect_cfg.get("generate_pristine", True))
    gen_defect = bool(defect_cfg.get("generate_defect", True))
    min_vac = int(defect_cfg.get("min_vacancies_total", 1))

    variants: list[dict[str, Any]] = []
    if gen_pristine:
        pristine_tpl = replace(
            tpl,
            project=replace(tpl.project, name=f"{tpl.project.name}_{pristine_suffix}"),
            export=replace(
                tpl.export,
                cif_path=f"{tpl.project.name}_{pristine_suffix}.generated.cif",
                metadata_json_path=f"{tpl.project.name}_{pristine_suffix}.generated.meta.json",
            ),
            interfaces=_zeroed_interfaces(tpl.interfaces),
        )
        variants.append(
            {
                "label": "pristine",
                "result": generate_slab_from_template(pristine_tpl, output_dir_override=output_dir_override),
            }
        )

    if gen_defect:
        defect_ifaces = _ensure_minimum_defect_interfaces(tpl.interfaces, min_vacancies_total=min_vac)
        defect_tpl = replace(
            tpl,
            project=replace(tpl.project, name=f"{tpl.project.name}_{defect_suffix}"),
            export=replace(
                tpl.export,
                cif_path=f"{tpl.project.name}_{defect_suffix}.generated.cif",
                metadata_json_path=f"{tpl.project.name}_{defect_suffix}.generated.meta.json",
            ),
            interfaces=defect_ifaces,
        )
        variants.append(
            {
                "label": "defect",
                "result": generate_slab_from_template(defect_tpl, output_dir_override=output_dir_override),
            }
        )

    return variants


@main.group(invoke_without_command=True)
@click.pass_context
def defect(ctx: click.Context) -> None:
    """Defect structure utilities."""
    if ctx.invoked_subcommand is not None:
        return
    tpl_path = _default_project_template_path(required=True)
    assert tpl_path is not None
    try:
        variants = _generate_default_defect_variants(tpl_path)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(click.style("✓ Defect variants generated", fg="green", bold=True))
    click.echo(f"  Template: {tpl_path}")
    for item in variants:
        res = item.get("result", {})
        click.echo(f"  - {item.get('label')}: {res.get('cif_path')}")
        click.echo(f"    metadata: {res.get('metadata_path')}")


@defect.command("vacancy")
@click.argument("structure_file", type=click.Path(exists=True))
@click.argument("site_index", type=int)
@click.option("--supercell", default="2,2,2", show_default=True,
              help="Supercell expansion as 'nx,ny,nz'.")
@click.option("--output", "-o", default=None, help="Output file (default: vacancy_N.cif).")
def defect_vacancy(
    structure_file: str,
    site_index: int,
    supercell: str,
    output: str | None,
) -> None:
    """Create a vacancy at SITE_INDEX in STRUCTURE_FILE."""
    from wtec.structure.defect import DefectBuilder
    import ase.io

    sc = tuple(int(x) for x in supercell.split(","))
    atoms = ase.io.read(structure_file)
    db = DefectBuilder(atoms)
    defect_atoms = db.vacancy(site_index, supercell=sc)

    out = output or f"vacancy_{site_index}.cif"
    ase.io.write(out, defect_atoms)
    click.echo(f"Written: {out}  ({len(defect_atoms)} atoms)")


@defect.command("substitute")
@click.argument("structure_file", type=click.Path(exists=True))
@click.argument("site_index", type=int)
@click.argument("new_element")
@click.option("--supercell", default="2,2,2", show_default=True)
@click.option("--output", "-o", default=None)
def defect_substitute(
    structure_file: str,
    site_index: int,
    new_element: str,
    supercell: str,
    output: str | None,
) -> None:
    """Substitute atom at SITE_INDEX with NEW_ELEMENT in STRUCTURE_FILE."""
    from wtec.structure.defect import DefectBuilder
    import ase.io

    sc = tuple(int(x) for x in supercell.split(","))
    atoms = ase.io.read(structure_file)
    db = DefectBuilder(atoms)
    defect_atoms = db.substitute(site_index, new_element, supercell=sc)

    out = output or f"subst_{site_index}_{new_element}.cif"
    ase.io.write(out, defect_atoms)
    click.echo(f"Written: {out}  ({len(defect_atoms)} atoms)")


# ---------------------------------------------------------------------------
# Entry point guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
