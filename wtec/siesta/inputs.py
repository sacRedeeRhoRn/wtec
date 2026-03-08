"""SIESTA .fdf input generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ase.data import atomic_numbers

from wtec.siesta.presets import pao_basis_block


class SiestaInputGenerator:
    """Generate SCF/NSCF inputs with Wannier export switches."""

    def __init__(
        self,
        *,
        atoms,
        material_name: str,
        pseudo_dir: str,
        pseudopots: dict[str, str],
        basis_profile: str,
        kpoints_scf: tuple[int, int, int] = (8, 8, 8),
        kpoints_nscf: tuple[int, int, int] = (12, 12, 12),
        spin_orbit: bool = True,
        num_bands: int = 48,
        dispersion_enabled: bool = True,
        dispersion_method: str = "d3",
        siesta_dftd3_use_xc_defaults: bool = True,
        include_pao_basis: bool = True,
        dm_mixing_weight: float = 0.10,
        dm_number_pulay: int = 8,
        electronic_temperature_k: float = 300.0,
        max_scf_iterations: int = 200,
    ) -> None:
        self.atoms = atoms
        self.material_name = str(material_name)
        self.pseudo_dir = str(pseudo_dir)
        self.pseudopots = dict(pseudopots)
        self.basis_profile = str(basis_profile or "default")
        self.kpoints_scf = tuple(int(v) for v in kpoints_scf)
        self.kpoints_nscf = tuple(int(v) for v in kpoints_nscf)
        self.spin_orbit = bool(spin_orbit)
        self.num_bands = int(num_bands)
        self.dispersion_enabled = bool(dispersion_enabled)
        self.dispersion_method = str(dispersion_method or "d3").strip().lower() or "d3"
        self.siesta_dftd3_use_xc_defaults = bool(siesta_dftd3_use_xc_defaults)
        self.include_pao_basis = bool(include_pao_basis)
        self.dm_mixing_weight = float(dm_mixing_weight)
        self.dm_number_pulay = int(dm_number_pulay)
        self.electronic_temperature_k = float(electronic_temperature_k)
        self.max_scf_iterations = int(max_scf_iterations)

    def scf(self, outfile: str | Path) -> None:
        Path(outfile).write_text(self._render(calculation="scf"))

    def nscf(self, outfile: str | Path) -> None:
        Path(outfile).write_text(self._render(calculation="nscf"))

    def _render(self, *, calculation: str) -> str:
        if calculation not in {"scf", "nscf"}:
            raise ValueError(f"Unsupported SIESTA calculation mode: {calculation!r}")
        species = self._species()
        basis_block = ""
        if self.include_pao_basis:
            basis_block = pao_basis_block(
                profile=self.basis_profile,
                symbols=(s for s, _, _ in species),
            )
        kpts = self.kpoints_scf if calculation == "scf" else self.kpoints_nscf
        lines = [
            f"SystemName        {self.material_name}",
            f"SystemLabel       {self.material_name}",
            f"NumberOfSpecies   {len(species)}",
            f"NumberOfAtoms     {len(self.atoms)}",
            "LatticeConstant   1.0 Ang",
            "",
            "%block LatticeVectors",
            *self._lattice_vectors(),
            "%endblock LatticeVectors",
            "",
            f"PseudoPotDir      {self.pseudo_dir}",
            "%block ChemicalSpeciesLabel",
            *self._species_block(species),
            "%endblock ChemicalSpeciesLabel",
            "",
            "AtomicCoordinatesFormat Ang",
            "%block AtomicCoordinatesAndAtomicSpecies",
            *self._coords_block(species),
            "%endblock AtomicCoordinatesAndAtomicSpecies",
            "",
            basis_block,
            "",
            "XC.functional     GGA",
            "XC.authors        PBE",
            "MeshCutoff        300 Ry",
            "PAO.EnergyShift   20 meV",
            f"DM.MixingWeight   {self.dm_mixing_weight:.6f}",
            f"DM.NumberPulay    {self.dm_number_pulay}",
            f"ElectronicTemperature  {self.electronic_temperature_k:g} K",
            "Spin              spin-orbit" if self.spin_orbit else "Spin              non-polarized",
            "",
            "%block kgrid_Monkhorst_Pack",
            f"{kpts[0]}   0   0   0.0",
            f"0   {kpts[1]}   0   0.0",
            f"0   0   {kpts[2]}   0.0",
            "%endblock kgrid_Monkhorst_Pack",
            "",
            "Siesta2Wannier90.WriteMmn   true",
            "Siesta2Wannier90.WriteAmn   true",
            "Siesta2Wannier90.WriteEig   true",
            f"Siesta2Wannier90.NumberOfBands {self.num_bands}",
            f"Siesta2Wannier90.SeedName   {self.material_name}",
            "",
        ]
        if self.dispersion_enabled and self.dispersion_method == "d3":
            lines.extend(
                [
                    "DFTD3             true",
                    "DFTD3.UseXCDefaults   "
                    + ("true" if self.siesta_dftd3_use_xc_defaults else "false"),
                    "",
                ]
            )
        if calculation == "scf":
            lines.extend(
                [
                    "DM.UseSaveDM      false",
                    f"MaxSCFIterations  {self.max_scf_iterations}",
                    "SCF.MustConverge  true",
                ]
            )
        else:
            lines.extend(
                [
                    "DM.UseSaveDM      true",
                    "MaxSCFIterations  1",
                    "SCF.MustConverge  false",
                ]
            )
        return "\n".join(lines) + "\n"

    def _species(self) -> list[tuple[str, int, str]]:
        syms: list[str] = []
        for s in self.atoms.get_chemical_symbols():
            if s not in syms:
                syms.append(s)
        out: list[tuple[str, int, str]] = []
        for sym in syms:
            if sym not in self.pseudopots:
                raise ValueError(
                    f"Missing SIESTA pseudopotential mapping for species {sym!r}."
                )
            z = atomic_numbers.get(sym)
            if z is None:
                raise ValueError(f"Unknown atomic symbol {sym!r}")
            out.append((sym, int(z), self.pseudopots[sym]))
        return out

    def _species_block(self, species: Iterable[tuple[str, int, str]]) -> list[str]:
        lines: list[str] = []
        for idx, (sym, z, pseudo) in enumerate(species, start=1):
            lines.append(f"{idx}  {z}  {sym}   {pseudo}")
        return lines

    def _coords_block(self, species: list[tuple[str, int, str]]) -> list[str]:
        species_index = {sym: i for i, (sym, _, _) in enumerate(species, start=1)}
        pos = self.atoms.get_positions()
        out: list[str] = []
        for sym, (x, y, z) in zip(self.atoms.get_chemical_symbols(), pos):
            out.append(
                f"{float(x): .10f}  {float(y): .10f}  {float(z): .10f}  {species_index[sym]}   {sym}"
            )
        return out

    def _lattice_vectors(self) -> list[str]:
        cell = self.atoms.cell.array
        return [f"{float(v[0]): .10f}  {float(v[1]): .10f}  {float(v[2]): .10f}" for v in cell]
