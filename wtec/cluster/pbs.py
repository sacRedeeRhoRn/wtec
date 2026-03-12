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
    mpi_procs_per_node: int | None = None
    omp_threads: int | None = None
    walltime: str = "24:00:00"      # HH:MM:SS
    memory_gb: int | None = None
    queue: str | None = None
    work_dir: str = "."
    modules: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    stdout_path: str | None = None
    runtime_log_path: str | None = None
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
    mpi_procs_per_node = int(
        config.mpi_procs_per_node
        if config.mpi_procs_per_node is not None
        else config.n_cores_per_node
    )
    omp_threads = int(config.omp_threads if config.omp_threads is not None else 1)
    resource_line = (
        "#PBS -l "
        f"select={config.n_nodes}:ncpus={config.n_cores_per_node}:"
        f"mpiprocs={mpi_procs_per_node}:ompthreads={omp_threads}"
    )
    log_path = config.stdout_path or f"{config.work_dir.rstrip('/')}/{config.job_name}.log"
    runtime_log_path = config.runtime_log_path or f"{config.work_dir.rstrip('/')}/wtec_job.log"

    script = f"""#!/bin/bash
#PBS -N {config.job_name}
{resource_line}
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
    restart_only: bool = False,
) -> str:
    """Convenience builder for a Wannier90 PBS script."""
    from wtec.cluster.mpi import MPIConfig, build_command

    total_cores = int(n_nodes) * int(n_cores_per_node)
    # Step-specific layout:
    # - wannier90.x -pp is cheap and effectively serial
    # - pw2wannier90.x parallelizes over k-point pools, not OpenMP threads
    # - final wannier90.x uses threaded linear algebra well on one node, but
    #   must stay MPI-distributed when the allocation spans multiple nodes
    serial_mpi = MPIConfig(n_cores=1, bind_to="none")
    pw2wan_mpi = MPIConfig(n_cores=total_cores, bind_to="core")
    final_wannier_single_rank = int(n_nodes) == 1

    serial_env = (
        "env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1"
    )
    threaded_env = (
        f"env OMP_NUM_THREADS={total_cores} MKL_NUM_THREADS={total_cores} "
        f"OPENBLAS_NUM_THREADS={total_cores} NUMEXPR_NUM_THREADS={total_cores}"
    )
    final_wannier_env = threaded_env if final_wannier_single_rank else serial_env
    final_wannier_mpi = serial_mpi if final_wannier_single_rank else pw2wan_mpi

    # Step 3: wannier90 main run
    wan_cmd = (
        f"{final_wannier_env} "
        + build_command("wannier90.x", extra_args=seedname, mpi=final_wannier_mpi)
    )

    commands: list[str]
    if restart_only:
        commands = [wan_cmd]
    else:
        # Step 1: preprocessing (compute NNKPts)
        pre_cmd = (
            f"{serial_env} "
            + build_command("wannier90.x", extra_args=f"-pp {seedname}", mpi=serial_mpi)
        )
        # Step 2: pw2wannier90
        pw2wan_cmd = (
            f"{serial_env} "
            + build_command(
                "pw2wannier90.x",
                input_file=f"{seedname}.pw2wan.in",
                output_file=f"{seedname}.pw2wan.out",
                mpi=pw2wan_mpi,
                extra_args=f"-nk {total_cores}",
            )
        )
        commands = [pre_cmd, pw2wan_cmd, wan_cmd]

    cfg = PBSJobConfig(
        job_name=f"w90_{seedname}",
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
        mpi_procs_per_node=(1 if restart_only and final_wannier_single_rank else n_cores_per_node),
        omp_threads=(n_cores_per_node if restart_only and final_wannier_single_rank else 1),
        walltime=walltime,
        queue=queue,
        work_dir=work_dir,
        modules=modules or [],
        env_vars=env_vars or {},
    )
    return generate_script(cfg, commands)
