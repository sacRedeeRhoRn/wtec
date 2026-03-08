"""Global settings and path management."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Local workspace ──────────────────────────────────────────────────────────
WORKSPACE_DIR   = Path.home() / ".wtec"
RUNS_DIR        = WORKSPACE_DIR / "runs"
CHECKPOINT_DIR  = WORKSPACE_DIR / "checkpoints"
PSEUDO_DIR      = Path(
    os.environ.get(
        "TOPOSLAB_QE_PSEUDO_DIR",
        os.environ.get("TOPOSLAB_PSEUDO_DIR", str(WORKSPACE_DIR / "pseudo")),
    )
)

# ── Default MPI settings (overridden by ClusterConfig) ───────────────────────
DEFAULT_MPI_CORES       = int(os.environ.get("TOPOSLAB_MPI_CORES", "32"))
DEFAULT_NPOOL           = int(os.environ.get("TOPOSLAB_NPOOL", "4"))

# ── QE executable names (must be in PATH on cluster) ─────────────────────────
QE_PW_EXE           = "pw.x"
QE_PW2WAN_EXE       = "pw2wannier90.x"
WANNIER90_EXE       = "wannier90.x"

# ── Kwant local source (sibling of this package's project root) ───────────────
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent   # .../wtec/
KWANT_SRC_DIR = Path(os.environ.get(
    "TOPOSLAB_KWANT_SRC",
    str(_PKG_ROOT.parent / "kwant"),
))

# ── Transport defaults ────────────────────────────────────────────────────────
DEFAULT_DISORDER_ENSEMBLE = int(os.environ.get("TOPOSLAB_DISORDER_ENSEMBLE", "50"))
DEFAULT_N_JOBS            = int(os.environ.get("TOPOSLAB_N_JOBS", "4"))
