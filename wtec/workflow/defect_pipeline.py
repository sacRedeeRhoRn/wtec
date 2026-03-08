"""Defect calculation pipeline: build defect supercell → DFT → property tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from wtec.structure.defect import DefectBuilder
from wtec.workflow.dft_pipeline import DFTPipeline


class DefectCalculation:
    """Run DFT on a defect supercell and track electronic property changes."""

    def __init__(
        self,
        atoms,
        material: str,
        defect_spec: dict,
        job_manager,
        *,
        run_dir: Path,
        remote_base: str,
        **dft_kwargs,
    ) -> None:
        """
        Parameters
        ----------
        atoms : ase.Atoms
            Pristine primitive cell.
        defect_spec : dict
            E.g. {"type": "vacancy", "site": 3, "supercell": [2,2,2]}
            or   {"type": "substitution", "site": 3, "element": "Nb", "supercell": [2,2,2]}
        """
        self.atoms = atoms
        self.material = material
        self.defect_spec = defect_spec
        self.jm = job_manager
        self.run_dir = Path(run_dir)
        self.remote_base = remote_base
        self.dft_kwargs = dft_kwargs

    def build_defect_cell(self):
        """Return the defect supercell as ASE Atoms."""
        spec = self.defect_spec
        sc = tuple(spec.get("supercell", [2, 2, 2]))
        db = DefectBuilder(self.atoms)

        dtype = spec["type"]
        site = spec["site"]

        if dtype == "vacancy":
            return db.vacancy(site, supercell=sc)
        elif dtype == "substitution":
            return db.substitute(site, spec["element"], supercell=sc)
        elif dtype == "antisite":
            return db.antisite(site, spec["site_b"], supercell=sc)
        else:
            raise ValueError(f"Unknown defect type: {dtype!r}")

    def run(self) -> dict:
        """Build defect cell, run DFT, compare electronic properties."""
        defect_atoms = self.build_defect_cell()
        defect_name = self._defect_name()
        defect_run_dir = self.run_dir / defect_name

        pipeline = DFTPipeline(
            defect_atoms,
            self.material,
            self.jm,
            run_dir=defect_run_dir,
            remote_base=f"{self.remote_base}/{defect_name}",
            **self.dft_kwargs,
        )
        hr_dat = pipeline.run_full()

        # Also run pristine for comparison
        pristine_run_dir = self.run_dir / "pristine"
        pristine_pipeline = DFTPipeline(
            self.atoms,
            self.material,
            self.jm,
            run_dir=pristine_run_dir,
            remote_base=f"{self.remote_base}/pristine",
            **self.dft_kwargs,
        )
        hr_dat_pristine = pristine_pipeline.run_full()

        return self._compare(hr_dat_pristine, hr_dat)

    def _compare(self, hr_dat_pristine: Path, hr_dat_defect: Path) -> dict:
        """Compare electronic properties of pristine vs defect."""
        from wtec.wannier.parser import read_hr_dat

        hd_p = read_hr_dat(hr_dat_pristine)
        hd_d = read_hr_dat(hr_dat_defect)

        from wtec.qe.parser import parse_fermi_energy

        fe_p = parse_fermi_energy(hr_dat_pristine.parent / f"{self.material}.scf.out")
        fe_d = parse_fermi_energy(hr_dat_defect.parent / f"{self.material}.scf.out")

        return {
            "defect_type": self.defect_spec["type"],
            "fermi_shift_eV": fe_d - fe_p,
            "num_wann_pristine": hd_p.num_wann,
            "num_wann_defect": hd_d.num_wann,
            "hr_dat_pristine": str(hr_dat_pristine),
            "hr_dat_defect": str(hr_dat_defect),
        }

    def _defect_name(self) -> str:
        spec = self.defect_spec
        dtype = spec["type"]
        site = spec["site"]
        if dtype == "vacancy":
            return f"vacancy_s{site}"
        elif dtype == "substitution":
            return f"subst_s{site}_{spec['element']}"
        elif dtype == "antisite":
            return f"antisite_s{site}_{spec['site_b']}"
        return "defect"
