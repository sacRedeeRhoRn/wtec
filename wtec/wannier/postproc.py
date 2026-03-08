"""Post-processing interface to WannierBerri (Berry curvature, AHE)."""

from __future__ import annotations

from pathlib import Path
import numpy as np


def compute_berry_curvature(
    hr_dat_path: str | Path,
    win_path: str | Path,
    *,
    k_mesh: tuple[int, int, int] = (50, 50, 50),
    fermi_energy: float = 0.0,
    component: str = "z",
) -> np.ndarray:
    """Compute Berry curvature Ω_z(k) on a uniform k-mesh via WannierBerri.

    Parameters
    ----------
    component : str
        Berry curvature component: 'x', 'y', or 'z'.

    Returns
    -------
    np.ndarray
        Berry curvature on the k-mesh.
    """
    try:
        import wannierberri as wberri
    except ImportError:
        raise ImportError(
            "wannierberri is required: wtec init --extra berry"
        )

    seedname = str(Path(hr_dat_path).with_suffix("").with_suffix("")).replace("_hr", "")
    system = wberri.System_w90(
        seedname,
        berry=True,
        use_wcc_phase=True,
    )

    comp_map = {"x": 0, "y": 1, "z": 2}
    if component not in comp_map:
        raise ValueError(f"component must be 'x', 'y', or 'z'; got {component!r}")

    grid = wberri.Grid(system, NK=k_mesh)
    result = wberri.run(
        system,
        grid=grid,
        calculators={"berry": wberri.calculators.static.AHC()},
        fermi_levels=[fermi_energy],
    )
    return result


def anomalous_hall_conductivity(
    hr_dat_path: str | Path,
    win_path: str | Path,
    *,
    fermi_energies: np.ndarray | None = None,
    k_mesh: tuple[int, int, int] = (50, 50, 50),
) -> dict:
    """Compute anomalous Hall conductivity σ_xy vs Fermi energy.

    Returns
    -------
    dict with keys: 'fermi_ev', 'sigma_xy_S_per_m'
    """
    try:
        import wannierberri as wberri
    except ImportError:
        raise ImportError("wannierberri is required: wtec init --extra berry")

    seedname = str(Path(hr_dat_path)).replace("_hr.dat", "")
    system = wberri.System_w90(seedname, berry=True)

    if fermi_energies is None:
        fermi_energies = np.linspace(-2.0, 2.0, 100)

    grid = wberri.Grid(system, NK=k_mesh)
    result = wberri.run(
        system,
        grid=grid,
        calculators={"ahc": wberri.calculators.static.AHC()},
        fermi_levels=fermi_energies.tolist(),
    )
    return {"fermi_ev": fermi_energies, "result": result}
