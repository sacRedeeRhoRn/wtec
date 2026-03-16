"""SIESTA basis presets (required for topology-preserving runs)."""

from __future__ import annotations

from typing import Iterable


# NOTE: These PAO blocks are intentionally explicit instead of relying on
# BasisSize defaults. They are conservative starter profiles for robust runs.
ELEMENT_BASIS: dict[str, str] = {
    "Ta": """Ta 2
  n=6   0   2
  n=5   2   2
""",
    "Nb": """Nb 2
  n=5   0   2
  n=4   2   2
""",
    "Co": """Co 2
  n=4   0   2
  n=3   2   2
""",
    "P": """P 2
  n=3   0   2
  n=3   1   2
""",
    "Si": """Si 2
  n=3   0   2
  n=3   1   2
""",
    "O": """O 2
  n=2   0   2
  n=2   1   2
""",
}


PROFILE_ALIASES: dict[str, tuple[str, ...]] = {
    "tap": ("Ta", "P", "Si", "O"),
    "nbp": ("Nb", "P", "Si", "O"),
    "cosi": ("Co", "Si"),
    "default": ("Ta", "Nb", "Co", "P", "Si", "O"),
}


def pao_basis_block(*, profile: str, symbols: Iterable[str]) -> str:
    """Return a %block PAO.Basis text for the given species."""
    profile_key = (profile or "default").strip().lower()
    allowed = set(PROFILE_ALIASES.get(profile_key, PROFILE_ALIASES["default"]))
    present = []
    for sym in symbols:
        s = str(sym).strip()
        if s and s not in present:
            present.append(s)

    missing = [s for s in present if s not in ELEMENT_BASIS]
    if missing:
        raise ValueError(
            f"Missing required SIESTA PAO basis preset for species: {missing}. "
            "Add them in wtec.siesta.presets.ELEMENT_BASIS."
        )

    lines = ["%block PAO.Basis"]
    for sym in present:
        if sym not in allowed and profile_key != "default":
            # keep profile strict to avoid accidental cross-material basis use.
            raise ValueError(
                f"Species {sym!r} is not in SIESTA basis profile {profile_key!r}."
            )
        lines.append(ELEMENT_BASIS[sym].rstrip())
    lines.append("%endblock PAO.Basis")
    return "\n".join(lines)
