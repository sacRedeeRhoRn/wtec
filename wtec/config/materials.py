"""Material presets for topology workflows (QE + SIESTA)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MaterialPreset:
    name: str
    formula: str
    space_group: str
    # Wannier90 projection strings, e.g. ["Ta:d", "P:p"]
    projections: List[str]
    # Number of Wannier functions
    num_wann: int
    # Number of bands to include (num_bands >= num_wann)
    num_bands: int
    # material class for validator/physics guardrails.
    material_class: str = "generic"
    # Disentanglement energy window [min, max] in eV relative to Fermi
    dis_win: tuple[float, float] = (-10.0, 15.0)
    dis_froz_win: tuple[float, float] = (-3.0, 3.0)
    # Pseudopotential files (filenames; actual paths set in settings.py)
    pseudopots: Dict[str, str] = field(default_factory=dict)
    # VASP POTCAR relative paths under TOPOSLAB_VASP_PSEUDO_DIR.
    vasp_potcars: Dict[str, str] = field(default_factory=dict)
    # SIESTA pseudopotentials (.psf/.psml) per species.
    siesta_pseudopots: Dict[str, str] = field(default_factory=dict)
    # ABACUS pseudopotentials/orbitals under their respective dirs.
    abacus_pseudopots: Dict[str, str] = field(default_factory=dict)
    abacus_orbitals: Dict[str, str] = field(default_factory=dict)
    # Named PAO profile in wtec.siesta.presets.
    siesta_basis_profile: str = "default"
    # Material-aware minimum meshes for physically valid Weyl workflows.
    min_kmesh_scf: tuple[int, int, int] = (4, 4, 4)
    min_kmesh_nscf: tuple[int, int, int] = (6, 6, 6)
    # Fermi energy reference (set after SCF run)
    fermi_ev: float | None = None


MATERIALS: dict[str, MaterialPreset] = {
    "TaP": MaterialPreset(
        name="TaP",
        formula="TaP",
        space_group="I4_1md",      # No. 109
        material_class="weyl",
        projections=["Ta:d", "P:p"],
        num_wann=16,               # 5 Ta-d + 3 P-p per formula unit × 2 (SOC)
        num_bands=36,
        # Keep enough states in the outer disentanglement window across all k
        # points for SOC spinor runs (num_wann=32 in primitive TaP cells).
        dis_win=(-12.0, 16.0),
        # Keep frozen window narrower to avoid over-constraining disentanglement
        # for thin-film slabs with many near-E_F bands.
        dis_froz_win=(-1.0, 0.2),
        pseudopots={"Ta": "Ta.pbe-spfn-rrkjus_psl.1.0.0.UPF",
                    "P":  "P.pbe-n-rrkjus_psl.1.0.0.UPF"},
        vasp_potcars={
            "Ta": "Ta/POTCAR",
            "P": "P/POTCAR",
        },
        siesta_pseudopots={
            "Ta": "Ta.psml",
            "P": "P.psml",
        },
        abacus_pseudopots={
            "Ta": "Ta.upf",
            "P": "P.upf",
        },
        abacus_orbitals={
            "Ta": "Ta.orb",
            "P": "P.orb",
        },
        siesta_basis_profile="tap",
        min_kmesh_scf=(8, 8, 8),
        min_kmesh_nscf=(12, 12, 12),
    ),
    "NbP": MaterialPreset(
        name="NbP",
        formula="NbP",
        space_group="I4_1md",      # isostructural to TaP
        material_class="weyl",
        projections=["Nb:d", "P:p"],
        num_wann=16,
        num_bands=36,
        # Mirror TaP safety margin for NbP SOC spinor disentanglement.
        dis_win=(-12.0, 16.0),
        dis_froz_win=(-1.0, 0.2),
        pseudopots={"Nb": "Nb.pbe-spn-rrkjus_psl.1.0.0.UPF",
                    "P":  "P.pbe-n-rrkjus_psl.1.0.0.UPF"},
        vasp_potcars={
            "Nb": "Nb/POTCAR",
            "P": "P/POTCAR",
        },
        siesta_pseudopots={
            "Nb": "Nb.psml",
            "P": "P.psml",
        },
        abacus_pseudopots={
            "Nb": "Nb.upf",
            "P": "P.upf",
        },
        abacus_orbitals={
            "Nb": "Nb.orb",
            "P": "P.orb",
        },
        siesta_basis_profile="nbp",
        min_kmesh_scf=(8, 8, 8),
        min_kmesh_nscf=(12, 12, 12),
    ),
    "CoSi": MaterialPreset(
        name="CoSi",
        formula="CoSi",
        space_group="P2_13",       # No. 198 — chiral multifold fermions
        material_class="weyl",
        projections=["Co:d", "Si:sp3"],
        num_wann=18,               # 5 Co-d + 4 Si-sp3 per f.u. × 2 (SOC)
        num_bands=28,
        dis_win=(-6.0, 10.0),
        dis_froz_win=(-1.5, 1.5),
        pseudopots={"Co": "Co.pbe-spn-kjpaw_psl.1.0.0.UPF",
                    "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF"},
        vasp_potcars={
            "Co": "Co/POTCAR",
            "Si": "Si/POTCAR",
        },
        siesta_pseudopots={
            "Co": "Co.psml",
            "Si": "Si.psml",
        },
        abacus_pseudopots={
            "Co": "Co.upf",
            "Si": "Si.upf",
        },
        abacus_orbitals={
            "Co": "Co.orb",
            "Si": "Si.orb",
        },
        siesta_basis_profile="cosi",
        min_kmesh_scf=(6, 6, 6),
        min_kmesh_nscf=(8, 8, 8),
    ),
    "SiBench": MaterialPreset(
        name="SiBench",
        formula="Si",
        space_group="Fd-3m",
        material_class="generic",
        projections=["Si:sp3"],
        num_wann=8,
        num_bands=16,
        dis_win=(-8.0, 8.0),
        dis_froz_win=(-2.0, 2.0),
        pseudopots={"Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF"},
        vasp_potcars={"Si": "Si/POTCAR"},
        siesta_pseudopots={"Si": "Si.psf"},
        abacus_pseudopots={"Si": "Si.upf"},
        abacus_orbitals={"Si": "Si.orb"},
        siesta_basis_profile="default",
        min_kmesh_scf=(2, 2, 2),
        min_kmesh_nscf=(3, 3, 3),
    ),
}


def get_material(name: str) -> MaterialPreset:
    """Return preset for a given material name. Case-insensitive."""
    key = name.strip()
    if key in MATERIALS:
        return MATERIALS[key]
    # case-insensitive fallback
    for k, v in MATERIALS.items():
        if k.lower() == key.lower():
            return v
    raise KeyError(
        f"Unknown material {name!r}. Available: {list(MATERIALS.keys())}"
    )
