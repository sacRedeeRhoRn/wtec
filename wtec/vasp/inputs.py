"""VASP input generation helpers."""

from __future__ import annotations

from pathlib import Path

import ase.io


class VaspInputGenerator:
    """Generate minimal VASP SCF/NSCF/Wannier input files."""

    def __init__(
        self,
        *,
        atoms,
        material_name: str,
        kpoints_scf: tuple[int, int, int] = (8, 8, 8),
        kpoints_nscf: tuple[int, int, int] = (12, 12, 12),
        encut_ev: float = 520.0,
        ediff: float = 1e-6,
        ismear: int = 0,
        sigma: float = 0.05,
        spin_orbit: bool = True,
        disable_symmetry: bool = True,
        num_bands: int | None = None,
    ) -> None:
        self.atoms = atoms
        self.material_name = str(material_name)
        self.kpoints_scf = tuple(int(v) for v in kpoints_scf)
        self.kpoints_nscf = tuple(int(v) for v in kpoints_nscf)
        self.encut_ev = float(encut_ev)
        self.ediff = float(ediff)
        self.ismear = int(ismear)
        self.sigma = float(sigma)
        self.spin_orbit = bool(spin_orbit)
        self.disable_symmetry = bool(disable_symmetry)
        self.num_bands = int(num_bands) if num_bands is not None else None

    def species_order(self) -> list[str]:
        order: list[str] = []
        for sym in self.atoms.get_chemical_symbols():
            if sym not in order:
                order.append(sym)
        return order

    def write_poscar(self, outfile: str | Path) -> None:
        ase.io.write(
            str(Path(outfile)),
            self.atoms,
            format="vasp",
            direct=True,
            vasp5=True,
            sort=False,
        )

    def write_kpoints(self, outfile: str | Path, mesh: tuple[int, int, int]) -> None:
        mesh = tuple(int(v) for v in mesh)
        text = (
            "Automatic mesh\n"
            "0\n"
            "Gamma\n"
            f"{mesh[0]} {mesh[1]} {mesh[2]} 0 0 0\n"
        )
        Path(outfile).write_text(text)

    def write_incar(self, outfile: str | Path, *, calculation: str) -> None:
        calc = str(calculation).strip().lower()
        if calc not in {"scf", "nscf", "wannier"}:
            raise ValueError(f"Unsupported VASP calculation mode: {calculation!r}")

        lines = [
            f"SYSTEM = {self.material_name}",
            "PREC = Accurate",
            f"ENCUT = {self.encut_ev:.1f}",
            f"EDIFF = {self.ediff:.2e}",
            f"ISMEAR = {self.ismear}",
            f"SIGMA = {self.sigma:.3f}",
            "LREAL = Auto",
            "ALGO = Normal",
            "NELM = 150",
            "LWAVE = .TRUE.",
            "LCHARG = .TRUE.",
            "LASPH = .TRUE.",
            "LMAXMIX = 4",
        ]
        if self.num_bands is not None and self.num_bands > 0:
            lines.append(f"NBANDS = {self.num_bands}")
        if self.disable_symmetry:
            lines.append("ISYM = 0")
        if self.spin_orbit:
            lines.extend(
                [
                    "LNONCOLLINEAR = .TRUE.",
                    "LSORBIT = .TRUE.",
                    "SAXIS = 0 0 1",
                    "GGA_COMPAT = .FALSE.",
                ]
            )

        if calc in {"nscf", "wannier"}:
            lines.extend(
                [
                    "ICHARG = 11",
                    "NELM = 1",
                    "LWAVE = .FALSE.",
                    "LCHARG = .FALSE.",
                ]
            )
        if calc == "wannier":
            lines.extend(
                [
                    "LWANNIER90 = .TRUE.",
                    "LWRITE_UNK = .FALSE.",
                ]
            )

        Path(outfile).write_text("\n".join(lines) + "\n")

    def write_scf_inputs(self, run_dir: str | Path) -> dict[str, Path]:
        run = Path(run_dir)
        poscar = run / "POSCAR"
        incar = run / "INCAR.scf"
        kpoints = run / "KPOINTS.scf"
        self.write_poscar(poscar)
        self.write_incar(incar, calculation="scf")
        self.write_kpoints(kpoints, self.kpoints_scf)
        return {"poscar": poscar, "incar": incar, "kpoints": kpoints}

    def write_nscf_inputs(self, run_dir: str | Path) -> dict[str, Path]:
        run = Path(run_dir)
        poscar = run / "POSCAR"
        incar = run / "INCAR.nscf"
        kpoints = run / "KPOINTS.nscf"
        if not poscar.exists():
            self.write_poscar(poscar)
        self.write_incar(incar, calculation="nscf")
        self.write_kpoints(kpoints, self.kpoints_nscf)
        return {"poscar": poscar, "incar": incar, "kpoints": kpoints}

    def write_wannier_inputs(self, run_dir: str | Path) -> dict[str, Path]:
        run = Path(run_dir)
        poscar = run / "POSCAR"
        incar = run / "INCAR.wannier"
        kpoints = run / "KPOINTS.nscf"
        if not poscar.exists():
            self.write_poscar(poscar)
        self.write_incar(incar, calculation="wannier")
        self.write_kpoints(kpoints, self.kpoints_nscf)
        return {"poscar": poscar, "incar": incar, "kpoints": kpoints}
