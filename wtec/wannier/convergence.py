"""Wannier90 convergence checks used as a hard workflow gate."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class WannierNotConvergedError(RuntimeError):
    """Raised when Wannier90 disentanglement did not converge."""


_MAX_ITER_RE = re.compile(
    r"Warning:\s*Maximum number of disentanglement iterations reached",
    flags=re.IGNORECASE,
)
_WIN_FLOAT_RE = re.compile(
    r"^\s*(dis_num_iter|dis_win_min|dis_win_max|dis_froz_min|dis_froz_max)\s*=\s*([^\s!#]+)",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _extract_win_params(win_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, raw_val in _WIN_FLOAT_RE.findall(win_text):
        k = key.lower()
        try:
            if k == "dis_num_iter":
                out[k] = int(float(raw_val))
            else:
                out[k] = float(raw_val)
        except Exception:
            out[k] = raw_val
    return out


def assert_wannier_converged(
    *,
    wout_path: str | Path,
    win_path: str | Path | None = None,
) -> None:
    """Raise WannierNotConvergedError when .wout indicates failed disentanglement."""
    p = Path(wout_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Wannier output not found: {p}")
    text = p.read_text(errors="ignore")
    if not _MAX_ITER_RE.search(text):
        return

    params: dict[str, Any] = {}
    if win_path is not None:
        wp = Path(win_path).expanduser().resolve()
        if wp.exists():
            params = _extract_win_params(wp.read_text(errors="ignore"))

    dis_iter = params.get("dis_num_iter")
    dis_win_min = params.get("dis_win_min")
    dis_win_max = params.get("dis_win_max")
    dis_froz_min = params.get("dis_froz_min")
    dis_froz_max = params.get("dis_froz_max")
    suggestion = (
        "Increase dis_num_iter, widen dis_win, or narrow dis_froz window. "
        "Recommended start: dis_num_iter=1000, dis_win +/-2 eV wider, "
        "dis_froz +/-0.5 eV tighter."
    )
    raise WannierNotConvergedError(
        "Wannier90 disentanglement did not converge: "
        f"matched pattern={_MAX_ITER_RE.pattern!r}; "
        f"dis_num_iter={dis_iter!r}, dis_win=({dis_win_min!r},{dis_win_max!r}), "
        f"dis_froz=({dis_froz_min!r},{dis_froz_max!r}). {suggestion}"
    )

