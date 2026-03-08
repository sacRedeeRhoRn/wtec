"""Quantum ESPRESSO pw.x input file generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class QEInputGenerator:
    """Generate pw.x input files for SCF, NSCF, and RELAX calculations."""

    atoms: object                          # ase.Atoms
    material_name: str
    pseudo_dir: str = "/pseudo"
    outdir: str = "./out"
    ecutwfc: float = 80.0                  # Ry
    ecutrho: float = 640.0                 # Ry
    kpoints_scf: tuple = (8, 8, 8)
    kpoints_nscf: tuple = (12, 12, 12)
    smearing: str = "mp"
    degauss: float = 0.02                  # Ry
    noncolin: bool = True                  # SOC required for Weyl semimetals
    lspinorb: bool = True
    pseudopots: Dict[str, str] = field(default_factory=dict)
    nbnd: int | None = None
    extra_control: Dict[str, str] = field(default_factory=dict)
    dispersion_enabled: bool = True
    dispersion_method: str = "d3"
    qe_vdw_corr: str = "grimme-d3"
    qe_dftd3_version: int = 4
    qe_dftd3_threebody: bool = True
    disable_symmetry: bool = False

    # ── public interface ─────────────────────────────────────────────────────

    def scf(self, outfile: str | Path | None = None) -> str:
        text = self._header("scf") + self._atomic_section() + self._kpoints(self.kpoints_scf)
        if outfile:
            Path(outfile).write_text(text)
        return text

    def nscf(self, outfile: str | Path | None = None) -> str:
        text = self._header("nscf") + self._atomic_section() + self._kpoints(self.kpoints_nscf)
        if outfile:
            Path(outfile).write_text(text)
        return text

    def relax(self, outfile: str | Path | None = None) -> str:
        text = self._header("relax") + self._atomic_section() + self._kpoints(self.kpoints_scf)
        if outfile:
            Path(outfile).write_text(text)
        return text

    # ── private helpers ──────────────────────────────────────────────────────

    def _header(self, calculation: str) -> str:
        symbols = self.atoms.get_chemical_symbols()
        unique = list(dict.fromkeys(symbols))
        n_species = len(unique)
        n_atoms = len(symbols)

        extra_lines = "\n".join(
            f"  {k} = {v}" for k, v in self.extra_control.items()
        )

        nbnd_line = f"\n  nbnd = {self.nbnd}" if self.nbnd else ""
        soc_lines = (
            "\n  noncolin = .true.\n  lspinorb = .true."
            if self.noncolin else ""
        )
        calc = str(calculation).strip().lower()
        nscf_kmesh_lock_lines = (
            "\n  nosym = .true.\n  noinv = .true."
            if (calc == "nscf" or bool(self.disable_symmetry))
            else ""
        )
        d3_lines = ""
        if self.dispersion_enabled and str(self.dispersion_method).strip().lower() == "d3":
            threebody = ".true." if bool(self.qe_dftd3_threebody) else ".false."
            d3_lines = (
                f"\n  vdw_corr = '{self.qe_vdw_corr}'"
                f"\n  dftd3_version = {int(self.qe_dftd3_version)}"
                f"\n  dftd3_threebody = {threebody}"
            )

        return f""" &CONTROL
  calculation = '{calculation}'
  prefix = '{self.material_name}'
  outdir = '{self.outdir}'
  pseudo_dir = '{self.pseudo_dir}'
  tprnfor = .true.
  tstress = .true.
{extra_lines}
 /
 &SYSTEM
  ibrav = 0
  nat = {n_atoms}
  ntyp = {n_species}
  ecutwfc = {self.ecutwfc}
  ecutrho = {self.ecutrho}
  occupations = 'smearing'
  smearing = '{self.smearing}'
  degauss = {self.degauss}{nbnd_line}{soc_lines}{nscf_kmesh_lock_lines}{d3_lines}
 /
 &ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.4
 /
"""

    def _atomic_section(self) -> str:
        from wtec.structure.io import to_qe_string
        return to_qe_string(self.atoms, pseudopots=self.pseudopots) + "\n"

    def _kpoints(self, mesh: tuple) -> str:
        return (
            f"K_POINTS {{automatic}}\n"
            f"  {mesh[0]} {mesh[1]} {mesh[2]}  0 0 0\n"
        )
