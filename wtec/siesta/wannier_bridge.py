"""Bridge SIESTA outputs into Wannier90 inputs."""

from __future__ import annotations

from pathlib import Path


def prepare_wannier_bridge(
    *,
    run_dir: str | Path,
    seedname: str,
    interface: str,
) -> dict[str, str]:
    """Validate/prepare AMN/MMN/EIG files for Wannier90."""
    rd = Path(run_dir).expanduser().resolve()
    if not rd.exists():
        raise FileNotFoundError(f"SIESTA run directory does not exist: {rd}")

    iface = str(interface).strip().lower() or "sisl"
    if iface == "builtin":
        raise RuntimeError(
            "siesta.wannier_interface='builtin' is not enabled in this workflow revision. "
            "Use 'sisl'."
        )
    if iface != "sisl":
        raise ValueError(f"Unknown siesta.wannier_interface={interface!r}. Use 'sisl'.")

    # Hard requirement for the selected interface.
    try:
        import sisl  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "siesta.wannier_interface='sisl' requires sisl>=0.14 on runtime Python. "
            f"Import failed: {type(exc).__name__}: {exc}"
        ) from exc

    amn = rd / f"{seedname}.amn"
    mmn = rd / f"{seedname}.mmn"
    eig = rd / f"{seedname}.eig"
    missing = [str(p.name) for p in (amn, mmn, eig) if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError(
            "SIESTA->Wannier files are missing after NSCF: "
            f"{missing}. Ensure Siesta2Wannier90.WriteAmn/WriteMmn/WriteEig are enabled."
        )

    return {
        "interface": iface,
        "amn": str(amn),
        "mmn": str(mmn),
        "eig": str(eig),
    }

