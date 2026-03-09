"""PBS/Torque job script generator.

All compute commands MUST use mpirun backend.
Fork-based launchstyle is NEVER used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PBSJobConfig:
    job_name: str
    n_nodes: int = 1
    n_cores_per_node: int = 32
    walltime: str = "24:00:00"      # HH:MM:SS
    memory_gb: int | None = None
    queue: str | None = None
    work_dir: str = "."
    modules: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    # NOTE: fork launchstyle is forbidden. Thread env vars are allowed only
    # with mpirun-launched workloads; no fork-based process backend is used.

    @property
    def total_cores(self) -> int:
        return self.n_nodes * self.n_cores_per_node


def generate_script(
    config: PBSJobConfig,
    commands: list[str],
    *,
    outfile: str | Path | None = None,
) -> str:
    """Generate a PBS job script.

    Parameters
    ----------
    config : PBSJobConfig
        Job resource configuration.
    commands : list[str]
        Shell commands to run (must all use mpirun; no fork).
    outfile : str | Path | None
        If provided, writes the script to this file.

    Returns
    -------
    str
        PBS script content.

    Example
    -------
    >>> cfg = PBSJobConfig("scf_TaP", n_nodes=2, n_cores_per_node=16)
    >>> script = generate_script(cfg, [
    ...     "mpirun -np 32 pw.x -npool 4 < scf.in > scf.out"
    ... ])
    """
    module_lines = "\n".join(f"module load {m}" for m in config.modules)
    env_lines = "\n".join(f"export {k}={v}" for k, v in config.env_vars.items())
    commands_str = "\n".join(commands)

    queue_line = f"#PBS -q {config.queue}" if config.queue else ""
    mem_line = f"#PBS -l mem={int(config.memory_gb)}gb" if config.memory_gb else ""
    log_path = f"{config.work_dir.rstrip('/')}/{config.job_name}.log"

    runtime_log_path = f"{config.work_dir.rstrip('/')}/wtec_job.log"

    script = f"""#!/bin/bash
#PBS -N {config.job_name}
#PBS -l nodes={config.n_nodes}:ppn={config.n_cores_per_node}
#PBS -l walltime={config.walltime}
{mem_line}
{queue_line}
#PBS -j oe
#PBS -o {log_path}

# ── environment ─────────────────────────────────────────────────────────────
{module_lines}
{env_lines}
export PYTHONUNBUFFERED=1

# ── working directory ────────────────────────────────────────────────────────
set -euo pipefail
cd {config.work_dir}
exec > >(tee -a {runtime_log_path}) 2>&1
echo "[wtec][runtime] start $(date -Is)"

# ── commands ─────────────────────────────────────────────────────────────────
# IMPORTANT: all parallel execution uses mpirun backend.
# Fork-based launchstyle is FORBIDDEN in this package.
{commands_str}
"""
    if outfile:
        Path(outfile).write_text(script)
    return script


def qe_scf_script(
    prefix: str,
    work_dir: str,
    *,
    n_nodes: int = 1,
    n_cores_per_node: int = 32,
    n_pool: int = 4,
    walltime: str = "12:00:00",
    queue: str | None = None,
    modules: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
) -> str:
    """Convenience builder for a QE SCF PBS script."""
    from wtec.cluster.mpi import MPIConfig, build_command

    mpi = MPIConfig(n_cores=n_nodes * n_cores_per_node, n_pool=n_pool)
    cmd = build_command("pw.x", input_file=f"{prefix}.scf.in",
                        output_file=f"{prefix}.scf.out", mpi=mpi)
    cfg = PBSJobConfig(
        job_name=f"scf_{prefix}",
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
        walltime=walltime,
        queue=queue,
        work_dir=work_dir,
        modules=modules or [],
        env_vars=env_vars or {},
    )
    return generate_script(cfg, [cmd])


def wannier90_script(
    seedname: str,
    work_dir: str,
    *,
    n_nodes: int = 1,
    n_cores_per_node: int = 32,
    walltime: str = "4:00:00",
    queue: str | None = None,
    modules: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
) -> str:
    """Convenience builder for a Wannier90 PBS script."""
    from wtec.cluster.mpi import MPIConfig, build_command

    mpi = MPIConfig(n_cores=n_nodes * n_cores_per_node)
    # Step 1: preprocessing (compute NNKPts)
    pre_cmd = build_command("wannier90.x", extra_args=f"-pp {seedname}", mpi=mpi)
    # Step 2: pw2wannier90
    pw2wan_cmd = build_command(
        "pw2wannier90.x",
        input_file=f"{seedname}.pw2wan.in",
        output_file=f"{seedname}.pw2wan.out",
        mpi=mpi,
    )
    # Step 3: wannier90 main run
    wan_cmd = build_command("wannier90.x", extra_args=seedname, mpi=mpi)

    cfg = PBSJobConfig(
        job_name=f"w90_{seedname}",
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
        walltime=walltime,
        queue=queue,
        work_dir=work_dir,
        modules=modules or [],
        env_vars=env_vars or {},
    )
    return generate_script(cfg, [pre_cmd, pw2wan_cmd, wan_cmd])
