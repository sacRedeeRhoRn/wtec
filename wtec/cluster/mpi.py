"""MPI command builder.

All parallel compute jobs use mpirun backend.
Fork-based parallelism is explicitly forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MPIConfig:
    n_cores: int = 32
    n_pool: int = 4           # QE -npool (k-point parallelism)
    n_band: int = 1           # QE -nband
    n_diag: int = 1           # QE -ndiag
    bind_to: str = "core"     # "core" | "socket" | "none"
    mpirun_opts: str = ""     # extra flags


def build_command(
    executable: str,
    *,
    input_file: str | None = None,
    output_file: str | None = None,
    mpi: MPIConfig | None = None,
    extra_args: str = "",
) -> str:
    """Build a mpirun command string.

    Parameters
    ----------
    executable : str
        Executable name, e.g. 'pw.x', 'wannier90.x'.
    input_file : str | None
        If provided, adds '< input_file'.
    output_file : str | None
        If provided, adds '> output_file'.
    mpi : MPIConfig | None
        MPI settings. Uses defaults if None.

    Returns
    -------
    str
        Full mpirun command string.

    Examples
    --------
    >>> build_command("pw.x", input_file="scf.in", output_file="scf.out",
    ...               mpi=MPIConfig(n_cores=32, n_pool=4))
    'mpirun -np 32 --bind-to core pw.x -npool 4 < scf.in > scf.out'
    """
    if mpi is None:
        mpi = MPIConfig()

    bind = f"--bind-to {mpi.bind_to}"
    prefix = f"mpirun -np {mpi.n_cores} {bind} {mpi.mpirun_opts}".strip()

    # QE parallelism flags
    qe_flags = ""
    if "pw.x" in executable or "cp.x" in executable:
        qe_flags = f"-npool {mpi.n_pool}"
        if mpi.n_band > 1:
            qe_flags += f" -nband {mpi.n_band}"

    cmd = f"{prefix} {executable} {qe_flags} {extra_args}".strip()

    if input_file:
        cmd += f" < {input_file}"
    if output_file:
        cmd += f" > {output_file}"

    return cmd
