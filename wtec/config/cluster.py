"""Cluster connection configuration — loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional at import time


@dataclass
class ClusterConfig:
    host: str
    port: int = 22
    user: str = ""
    password: str | None = None
    key_path: str | None = None
    remote_workdir: str = "/home/$USER/wtec_runs"
    mpi_cores: int = 32
    mpi_cores_by_queue: dict[str, int] = field(default_factory=dict)
    pbs_queue: str | None = None
    pbs_queue_priority: list[str] = field(
        default_factory=lambda: ["g4", "g3", "g2", "g1"]
    )
    qe_pseudo_dir: str = "/pseudo"
    siesta_pseudo_dir: str = "/pseudo"
    vasp_pseudo_dir: str = "/pseudo"
    abacus_pseudo_dir: str = "/pseudo"
    abacus_orbital_dir: str = "/orbital"
    omp_threads: int | None = None
    modules: list[str] = field(default_factory=list)
    bin_dirs: list[str] = field(default_factory=list)

    @staticmethod
    def _parse_mpi_cores_by_queue(raw: str) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for token in raw.split(","):
            tok = token.strip()
            if not tok:
                continue
            if ":" in tok:
                q, v = tok.split(":", 1)
            elif "=" in tok:
                q, v = tok.split("=", 1)
            else:
                raise ValueError(
                    "Invalid TOPOSLAB_MPI_CORES_BY_QUEUE entry "
                    f"{tok!r}. Use queue:cores format, e.g. g4:64,g1:16."
                )
            q = q.strip()
            if not q:
                raise ValueError("Queue name cannot be empty in TOPOSLAB_MPI_CORES_BY_QUEUE.")
            cores = int(v.strip())
            if cores <= 0:
                raise ValueError(
                    f"Cores for queue {q!r} must be > 0 (got {cores})."
                )
            mapping[q] = cores
        return mapping

    def cores_for_queue(self, queue: str | None) -> int:
        """Return cores-per-node for a resolved queue, falling back to mpi_cores."""
        if not queue:
            return self.mpi_cores
        if queue in self.mpi_cores_by_queue:
            return self.mpi_cores_by_queue[queue]
        ql = queue.lower()
        if ql in self.mpi_cores_by_queue:
            return self.mpi_cores_by_queue[ql]
        return self.mpi_cores

    @classmethod
    def from_env(cls) -> "ClusterConfig":
        """Load cluster config from environment variables."""
        host = os.environ.get("TOPOSLAB_CLUSTER_HOST", "")
        if not host:
            raise EnvironmentError(
                "TOPOSLAB_CLUSTER_HOST is not set. "
                "Copy .env.example → .env and fill in values."
            )
        queue_raw = os.environ.get("TOPOSLAB_PBS_QUEUE", "").strip()
        queue_priority_raw = os.environ.get("TOPOSLAB_PBS_QUEUE_PRIORITY", "").strip()
        if queue_priority_raw:
            queue_priority = [
                q.strip() for q in queue_priority_raw.split(",") if q.strip()
            ]
        else:
            queue_priority = ["g4", "g3", "g2", "g1"]

        omp_threads_raw = os.environ.get("TOPOSLAB_OMP_THREADS", "").strip()
        omp_threads = int(omp_threads_raw) if omp_threads_raw else None
        modules_raw = os.environ.get("TOPOSLAB_CLUSTER_MODULES", "").strip()
        modules = [m.strip() for m in modules_raw.split(",") if m.strip()] if modules_raw else []
        bin_dirs_raw = os.environ.get("TOPOSLAB_CLUSTER_BIN_DIRS", "").strip()
        bin_dirs = [p.strip() for p in bin_dirs_raw.split(",") if p.strip()] if bin_dirs_raw else []
        mpi_cores_by_queue_raw = os.environ.get(
            "TOPOSLAB_MPI_CORES_BY_QUEUE", ""
        ).strip()
        mpi_cores_by_queue = (
            cls._parse_mpi_cores_by_queue(mpi_cores_by_queue_raw)
            if mpi_cores_by_queue_raw
            else {}
        )

        default_remote = (
            f"/home/{os.environ.get('TOPOSLAB_CLUSTER_USER', '')}/wtec_runs"
            if os.environ.get("TOPOSLAB_CLUSTER_USER", "").strip()
            else "/home/$USER/wtec_runs"
        )

        return cls(
            host=host,
            port=int(os.environ.get("TOPOSLAB_CLUSTER_PORT", "22")),
            user=os.environ.get("TOPOSLAB_CLUSTER_USER", ""),
            password=os.environ.get("TOPOSLAB_CLUSTER_PASS") or None,
            key_path=os.environ.get("TOPOSLAB_CLUSTER_KEY") or None,
            remote_workdir=os.environ.get(
                "TOPOSLAB_REMOTE_WORKDIR", default_remote
            ),
            mpi_cores=int(os.environ.get("TOPOSLAB_MPI_CORES", "32")),
            mpi_cores_by_queue=mpi_cores_by_queue,
            pbs_queue=queue_raw or None,
            pbs_queue_priority=queue_priority,
            qe_pseudo_dir=os.environ.get("TOPOSLAB_QE_PSEUDO_DIR", "/pseudo"),
            siesta_pseudo_dir=os.environ.get("TOPOSLAB_SIESTA_PSEUDO_DIR", "/pseudo"),
            vasp_pseudo_dir=os.environ.get("TOPOSLAB_VASP_PSEUDO_DIR", "/pseudo"),
            abacus_pseudo_dir=os.environ.get("TOPOSLAB_ABACUS_PSEUDO_DIR", "/pseudo"),
            abacus_orbital_dir=os.environ.get("TOPOSLAB_ABACUS_ORBITAL_DIR", "/orbital"),
            omp_threads=omp_threads,
            modules=modules,
            bin_dirs=bin_dirs,
        )

    def __repr__(self) -> str:  # never expose password
        return (
            f"ClusterConfig(host={self.host!r}, port={self.port}, "
            f"user={self.user!r}, auth={'key' if self.key_path else 'password'})"
        )
