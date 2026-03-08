"""ABACUS LCAO input generation helpers."""

from __future__ import annotations

from pathlib import Path

from ase.data import atomic_masses, atomic_numbers


_BOHR_PER_ANGSTROM = 1.8897261254578281


class AbacusInputGenerator:
    """Generate minimal ABACUS STRU/KPT/INPUT files for LCAO runs."""

    def __init__(
        self,
        *,
        atoms,
        material_name: str,
        pseudo_dir: str,
        orbital_dir: str,
        pseudopots: dict[str, str],
        orbitals: dict[str, str],
        kpoints_scf: tuple[int, int, int] = (8, 8, 8),
        kpoints_nscf: tuple[int, int, int] = (12, 12, 12),
        basis_type: str = "lcao",
        ks_solver: str = "genelpa",
        spin_orbit: bool = True,
        num_bands: int = 64,
    ) -> None:
        self.atoms = atoms
        self.material_name = str(material_name)
        self.pseudo_dir = str(pseudo_dir)
        self.orbital_dir = str(orbital_dir)
        self.pseudopots = dict(pseudopots)
        self.orbitals = dict(orbitals)
        self.kpoints_scf = tuple(int(v) for v in kpoints_scf)
        self.kpoints_nscf = tuple(int(v) for v in kpoints_nscf)
        self.basis_type = str(basis_type or "lcao").strip().lower() or "lcao"
        self.ks_solver = str(ks_solver or "genelpa").strip().lower() or "genelpa"
        self.spin_orbit = bool(spin_orbit)
        self.num_bands = int(num_bands)

    def species_order(self) -> list[str]:
        order: list[str] = []
        for sym in self.atoms.get_chemical_symbols():
            if sym not in order:
                order.append(sym)
        return order

    def write_stru(self, outfile: str | Path) -> None:
        species = self.species_order()
        lines: list[str] = ["ATOMIC_SPECIES"]
        for sym in species:
            z = atomic_numbers.get(sym)
            if z is None:
                raise ValueError(f"Unknown atomic symbol {sym!r}")
            mass = float(atomic_masses[int(z)])
            pseudo = self.pseudopots.get(sym)
            if not pseudo:
                raise ValueError(
                    f"Missing ABACUS pseudopotential mapping for species {sym!r}."
                )
            lines.append(f"{sym} {mass:.8f} {pseudo}")

        lines.append("")
        lines.append("NUMERICAL_ORBITAL")
        for sym in species:
            orb = self.orbitals.get(sym)
            if not orb:
                raise ValueError(
                    f"Missing ABACUS orbital mapping for species {sym!r}."
                )
            lines.append(str(orb))

        cell = self.atoms.cell.array
        lines.extend(
            [
                "",
                "LATTICE_CONSTANT",
                f"{_BOHR_PER_ANGSTROM:.12f}",
                "",
                "LATTICE_VECTORS",
                f"{float(cell[0][0]): .10f} {float(cell[0][1]): .10f} {float(cell[0][2]): .10f}",
                f"{float(cell[1][0]): .10f} {float(cell[1][1]): .10f} {float(cell[1][2]): .10f}",
                f"{float(cell[2][0]): .10f} {float(cell[2][1]): .10f} {float(cell[2][2]): .10f}",
                "",
                "ATOMIC_POSITIONS",
                "Cartesian_angstrom",
            ]
        )

        positions = self.atoms.get_positions()
        symbols = self.atoms.get_chemical_symbols()
        for sym in species:
            lines.append(sym)
            lines.append("0.0")
            idxs = [i for i, s in enumerate(symbols) if s == sym]
            lines.append(str(len(idxs)))
            for i in idxs:
                x, y, z = positions[i]
                lines.append(
                    f"{float(x): .10f} {float(y): .10f} {float(z): .10f} 1 1 1"
                )

        Path(outfile).write_text("\n".join(lines) + "\n")

    def write_kpt(self, outfile: str | Path, mesh: tuple[int, int, int]) -> None:
        mesh = tuple(int(v) for v in mesh)
        text = (
            "K_POINTS\n"
            "0\n"
            "Gamma\n"
            f"{mesh[0]} {mesh[1]} {mesh[2]} 0 0 0\n"
        )
        Path(outfile).write_text(text)

    def write_input(self, outfile: str | Path, *, calculation: str) -> None:
        calc = str(calculation).strip().lower()
        if calc not in {"scf", "nscf", "wannier"}:
            raise ValueError(f"Unsupported ABACUS calculation mode: {calculation!r}")

        lines = [
            "INPUT_PARAMETERS",
            f"suffix {self.material_name}",
            f"calculation {'scf' if calc == 'scf' else 'nscf'}",
            "symmetry 0",
            f"pseudo_dir {self.pseudo_dir}",
            f"orbital_dir {self.orbital_dir}",
            "stru_file STRU",
            "kpoint_file KPT",
            f"basis_type {self.basis_type}",
            f"ks_solver {self.ks_solver}",
            f"nbands {self.num_bands}",
            "smearing_method gauss",
            "smearing_sigma 0.01",
            "scf_thr 1.0e-8",
            "scf_nmax 200",
            "mixing_type broyden",
            "mixing_beta 0.4",
        ]
        if self.spin_orbit:
            lines.extend(
                [
                    "noncolin 1",
                    "lspinorb 1",
                    "nspin 4",
                ]
            )
        else:
            lines.append("nspin 1")

        if calc in {"nscf", "wannier"}:
            lines.extend(
                [
                    "init_chg file",
                    "scf_nmax 1",
                ]
            )
        if calc == "wannier":
            lines.append("towannier90 1")

        Path(outfile).write_text("\n".join(lines) + "\n")

    def write_scf_inputs(self, run_dir: str | Path) -> dict[str, Path]:
        run = Path(run_dir)
        stru = run / "STRU"
        input_file = run / "INPUT.scf"
        kpt = run / "KPT.scf"
        self.write_stru(stru)
        self.write_input(input_file, calculation="scf")
        self.write_kpt(kpt, self.kpoints_scf)
        return {"stru": stru, "input": input_file, "kpt": kpt}

    def write_nscf_inputs(self, run_dir: str | Path) -> dict[str, Path]:
        run = Path(run_dir)
        stru = run / "STRU"
        input_file = run / "INPUT.nscf"
        kpt = run / "KPT.nscf"
        if not stru.exists():
            self.write_stru(stru)
        self.write_input(input_file, calculation="nscf")
        self.write_kpt(kpt, self.kpoints_nscf)
        return {"stru": stru, "input": input_file, "kpt": kpt}

    def write_wannier_inputs(self, run_dir: str | Path) -> dict[str, Path]:
        run = Path(run_dir)
        stru = run / "STRU"
        input_file = run / "INPUT.wannier"
        kpt = run / "KPT.nscf"
        if not stru.exists():
            self.write_stru(stru)
        self.write_input(input_file, calculation="wannier")
        self.write_kpt(kpt, self.kpoints_nscf)
        return {"stru": stru, "input": input_file, "kpt": kpt}
