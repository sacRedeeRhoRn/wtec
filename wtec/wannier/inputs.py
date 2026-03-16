"""Wannier90 .win file generator."""

from __future__ import annotations

from pathlib import Path

from wtec.qe.lcao import projections_block, get_num_wann


def _qe_folded_mp_axis(n: int) -> list[float]:
    """Return QE-style folded Monkhorst-Pack fractions for shift=(0,0,0).

    Example for n=4: [0.0, 0.25, -0.5, -0.25]
    """
    vals: list[float] = []
    for i in range(int(n)):
        v = i / float(n)
        if v >= 0.5:
            v -= 1.0
        vals.append(v)
    return vals


def generate_win(
    atoms,
    material: str,
    *,
    num_bands: int,
    fermi_energy: float = 0.0,
    kpoints: tuple[int, int, int] = (8, 8, 8),
    dis_win: tuple[float, float] = (-10.0, 15.0),
    dis_froz_win: tuple[float, float] = (-3.0, 3.0),
    dis_num_iter: int = 1000,
    num_iter: int = 1000,
    spinors: bool = True,
    search_shells: int = 300,
    kmesh_tol: float = 1.0e-2,
    restart: str | None = None,
    ws_distance_tol: float | None = None,
    custom_projections: list[str] | None = None,
    outfile: str | Path | None = None,
) -> str:
    """Generate a Wannier90 .win file.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure (provides cell parameters).
    material : str
        Material name (used to look up LCAO projections).
    num_bands : int
        Number of bands passed from QE nscf calculation.
    fermi_energy : float
        Fermi energy in eV (from QE scf output).
    """
    num_wann = get_num_wann(material, custom_projections, spinors=spinors, atoms=atoms)
    cell = atoms.get_cell()           # Angstroms, rows are lattice vectors
    symbols = atoms.get_chemical_symbols()
    scaled_pos = atoms.get_scaled_positions()

    # ── cell block ───────────────────────────────────────────────────────────
    cell_block = "begin unit_cell_cart\nang"
    for row in cell:
        cell_block += f"\n  {row[0]:.8f}  {row[1]:.8f}  {row[2]:.8f}"
    cell_block += "\nend unit_cell_cart"

    # ── atoms block ──────────────────────────────────────────────────────────
    atoms_block = "begin atoms_frac"
    for s, p in zip(symbols, scaled_pos):
        atoms_block += f"\n  {s}  {p[0]:.8f}  {p[1]:.8f}  {p[2]:.8f}"
    atoms_block += "\nend atoms_frac"

    # ── kpoints block ────────────────────────────────────────────────────────
    kx, ky, kz = kpoints
    kpts_block = (
        f"mp_grid : {kx} {ky} {kz}\n"
        f"search_shells = {int(search_shells)}\n"
        f"kmesh_tol = {float(kmesh_tol):.6g}\n\n"
        "begin kpoints"
    )
    xs = _qe_folded_mp_axis(kx)
    ys = _qe_folded_mp_axis(ky)
    zs = _qe_folded_mp_axis(kz)
    for x in xs:
        for y in ys:
            for z in zs:
                kpts_block += f"\n  {x:.6f}  {y:.6f}  {z:.6f}"
    kpts_block += "\nend kpoints"

    spinors_line = "spinors = .true." if spinors else "spinors = .false."

    restart_line = f"restart = {str(restart).strip()}" if restart else ""
    ws_distance_tol_line = (
        f"ws_distance_tol = {float(ws_distance_tol):.8g}" if ws_distance_tol is not None else ""
    )

    optional_lines = "\n".join(line for line in (restart_line, ws_distance_tol_line) if line)
    if optional_lines:
        optional_lines += "\n"

    text = f"""num_wann = {num_wann}
num_bands = {num_bands}
{spinors_line}
{optional_lines}

dis_win_max  = {fermi_energy + dis_win[1]:.4f}
dis_win_min  = {fermi_energy + dis_win[0]:.4f}
dis_froz_max = {fermi_energy + dis_froz_win[1]:.4f}
dis_froz_min = {fermi_energy + dis_froz_win[0]:.4f}

dis_num_iter  = {int(dis_num_iter)}
num_iter      = {int(num_iter)}
num_print_cycles = 50
write_hr = .true.

{projections_block(material, custom_projections)}

{cell_block}

{atoms_block}

{kpts_block}
"""
    if outfile:
        Path(outfile).write_text(text)
    return text
