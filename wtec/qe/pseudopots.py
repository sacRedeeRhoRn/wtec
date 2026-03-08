"""Pseudopotential registry and helpers."""

from __future__ import annotations

from pathlib import Path

from wtec.config.settings import PSEUDO_DIR


# SSSP Efficiency v1.3 pseudopotentials (PBE, ONCV/USPP)
SSSP_REGISTRY: dict[str, str] = {
    "Ta": "Ta.pbe-spfn-rrkjus_psl.1.0.0.UPF",
    "Nb": "Nb.pbe-spn-rrkjus_psl.0.3.0.UPF",
    "P":  "P.pbe-n-rrkjus_psl.1.0.0.UPF",
    "Co": "Co.pbe-spn-kjpaw_psl.0.3.1.UPF",
    "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
    "W":  "W.pbe-spn-kjpaw_psl.0.1.UPF",
    "Au": "Au.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Pt": "Pt.pbe-spfn-rrkjus_psl.1.0.0.UPF",
}


def get_pseudo_filename(element: str) -> str:
    """Return pseudopotential filename for an element."""
    if element not in SSSP_REGISTRY:
        raise KeyError(
            f"No pseudopotential registered for element {element!r}. "
            "Add it to SSSP_REGISTRY or provide a custom pseudopotentials dict."
        )
    return SSSP_REGISTRY[element]


def check_pseudos_present(elements: list[str], pseudo_dir: Path | None = None) -> dict[str, bool]:
    """Check which pseudopotential files are present on disk.

    Returns
    -------
    dict mapping element → True/False
    """
    d = Path(pseudo_dir or PSEUDO_DIR)
    result = {}
    for el in elements:
        fname = SSSP_REGISTRY.get(el, "")
        result[el] = (d / fname).exists() if fname else False
    return result


def pseudopots_block(elements: list[str], pseudo_dir: Path | None = None) -> str:
    """Return QE ATOMIC_SPECIES pseudopotential lines."""
    lines = []
    for el in elements:
        fname = get_pseudo_filename(el)
        lines.append(f"  {el}  1.0  {fname}")   # mass filled by caller
    return "\n".join(lines)
