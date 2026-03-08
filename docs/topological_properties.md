# Topological Semimetallic Properties in `wtec`

## Overview

`wtec` (Wannier–Tight-binding Electronic-property Calculator) is a workflow package that
computes thickness-dependent electrical resistivity and mean free path (MFP) for topological
semimetal thin films using a first-principles-derived tight-binding Hamiltonian.
This document describes every topological property the package is designed to capture,
what physics underpins each observable, and where in the codebase each property lives.

---

## 1. Target Materials

| Material | Structure    | Space Group | Topology class                | Key property                        |
|----------|--------------|-------------|-------------------------------|-------------------------------------|
| **TaP**  | Body-centred tetragonal | I4₁md (No. 109) | Weyl semimetal | Largest Fermi arcs known; extreme magnetoresistance |
| **NbP**  | Isostructural to TaP   | I4₁md (No. 109) | Weyl semimetal | Ultra-high mobility; anomalous MFP ~1–1.5 µm |
| **CoSi** | Simple cubic chiral    | P2₁3 (No. 198)  | Chiral multifold semimetal    | Spin-1 and spin-3/2 fermions; topological charge ±2 |

All three are **non-magnetic, non-centrosymmetric** crystals whose topology is driven purely
by spin-orbit coupling (SOC) and the non-symmorphic or chiral symmetry of the space group.

---

## 2. Topological Properties Targeted

### 2.1 Weyl Nodes (TaP, NbP)

Weyl nodes are isolated band-crossing points in the 3-D Brillouin zone where two
non-degenerate bands touch linearly:

```
H(k) ≈ ±vF · σ · (k − k_W)
```

Each node carries a **topological charge** (chirality) χ = ±1, defined by the Berry flux
integral over any surface enclosing the node:

```
χ = (1/2π) ∮ Ω(k) · dS,   Ω_n(k) = -2 Im ⟨∂_k u_n| × |∂_k u_n⟩
```

TaP and NbP host **24 Weyl nodes** in total (12 pairs of opposite chirality) related by the
fourfold screw axis of I4₁md. Because no inversion symmetry is present, the nodes are not
pinned to high-symmetry points and appear at generic **k** positions.

**How `wtec` captures this:**
The `_hr.dat` Hamiltonian encodes H(R) at all lattice vectors. When Fourier-transformed,
H(k) reproduces the DFT band structure including the exact Weyl node positions. Node
positions and chiralities are implicit in the hopping matrices; they emerge when the
Kwant slab is built and surface states are computed.

> **Relevant file:** `wtec/wannier/parser.py` — `read_hr_dat()`, `interpolate_bands()`

---

### 2.2 Fermi Arc Surface States

By bulk-boundary correspondence, each Weyl node of chirality χ = +1 on one surface
is connected to a node of χ = −1 by an **open arc** in the 2-D surface Brillouin zone.
This arc is a surface-bound, chiral state that:

- Does **not** exist in the bulk spectrum
- Spans the surface BZ between projected Weyl nodes
- Contributes to electrical conduction independently of bulk carriers

In TaP and NbP the arcs are unusually long (spanning nearly the full BZ) and cross the
Fermi energy, making them dominant contributors to transport in thin films.

**How `wtec` captures this:**
A hard-wall boundary at iz = 0 and iz = n_layers_z − 1 breaks translational symmetry in z,
which is the condition that forces bulk-boundary correspondence to produce arc states on
the z-faces of the slab. The Kwant system built by `WannierTBModel.to_kwant_builder()`
automatically contains these arcs — no analytic surface-state ansatz is required.

> **Relevant files:**
> - `wtec/wannier/model.py` — `to_kwant_builder()`, `_iter_hoppings()`, `_add_hoppings_region()`
> - `wtec/transport/builder.py` — `build_trilayer()`

---

### 2.3 Multifold Fermions (CoSi)

CoSi belongs to the chiral space group P2₁3. Its band structure hosts:

- A **spin-1 fermion** (threefold crossing) at Γ with topological charge χ = +2
- A **spin-3/2 fermion** (fourfold crossing) at R with topological charge χ = −2

These multifold crossings carry higher topological charges than Weyl nodes (|χ| = 2 vs 1),
which implies:
- Longer Fermi arcs connecting them on the surface
- Larger anomalous surface contribution to transport

The chiral P2₁3 structure means all multifold crossings are at high-symmetry points Γ and R,
in contrast to TaP/NbP where nodes are at generic **k**.

**How `wtec` captures this:**
The `Co:d` + `Si:sp3` projections faithfully span the relevant bands. With SOC enabled in
QE (`noncolin=.true.`, `lspinorb=.true.`), the band inversions at Γ and R are encoded into
the Wannier Hamiltonian. The Kwant slab will show surface arcs of length ~2×(Γ-to-R distance).

> **Relevant files:**
> - `wtec/config/materials.py` — `MATERIALS["CoSi"]`
> - `wtec/qe/lcao.py` — `PROJECTION_LIBRARY["CoSi"]`

---

### 2.4 Thickness-Dependent Resistivity ρ(d)

This is the **primary observable** the package computes.

#### Expected behaviour for topological films

| Thickness regime | Physics | ρ behaviour |
|------------------|---------|-------------|
| d ≫ ℓ (bulk limit) | Bulk transport dominates | ρ → ρ_bulk (constant) |
| d ~ ℓ → d_min | Fermi arc surface channels add to conductance | ρ decreases as d decreases |
| d < d_min (thin limit) | Top/bottom arcs hybridise, open a gap | ρ increases as d decreases |

The **non-monotonic minimum** at some d_min is a direct signature of topological surface
transport. It has no counterpart in a conventional Fuchs-Sondheimer film where ρ is
monotonically increasing as d → 0.

**How `wtec` computes this:**

```
ρ = d / (G × A)
```

where G is the Landauer conductance (in e²/h) from the Kwant S-matrix calculation, d is the
film thickness, and A is the transverse cross-section. The Kwant system is built for each d
from the same `_hr.dat`, so the surface vs bulk balance evolves automatically with d.

> **Relevant files:**
> - `wtec/transport/conductance.py` — `compute_conductance_vs_thickness()`
> - `wtec/transport/observables.py` — `rho_from_conductance()`
> - `wtec/analysis/thickness_scan.py` — `plot_rho_vs_thickness()`, `detect_rho_minimum()`
> - `wtec/workflow/transport_pipeline.py` — `TransportPipeline.run_thickness_scan()`

---

### 2.5 Reference: Fuchs-Sondheimer Curve

To make the topological deviation visible, the package computes the classical
Fuchs-Sondheimer (FS) prediction for a topologically trivial metal film:

```
ρ_FS(d) / ρ_bulk ≈ 1 + (3/8)(1 − p) ℓ/d     (d ≫ ℓ)
```

where p is the surface specularity (0 = fully diffuse). The FS curve is monotonically
increasing as d decreases. Any downward deviation from this curve — especially a minimum
in ρ(d) — is a signature of topological surface conduction.

Reference values for normal metals and the three target materials:

| Material | ℓ (nm) | ρ_bulk (Ω·m) |
|----------|--------|--------------|
| Cu       | 39     | 1.68 × 10⁻⁸  |
| Au       | 38     | 2.44 × 10⁻⁸  |
| Pt       | 9      | 10.6 × 10⁻⁸  |
| W        | 14     | 5.4 × 10⁻⁸   |
| TaP      | ~1000  | ~1 × 10⁻⁹   |
| NbP      | ~1500  | ~0.8 × 10⁻⁹  |
| CoSi     | ~200   | ~5 × 10⁻⁹   |

> **Relevant files:**
> - `wtec/transport/observables.py` — `fuchs_sondheimer_rho()`
> - `wtec/analysis/mfp_compare.py` — `REFERENCE_METALS`

---

### 2.6 Mean Free Path Extraction

The MFP is extracted from the diffusive G(L) ∝ 1/L scaling:

```
G(L) = σ A / L     →     1/G = L / (σ A)
```

A linear fit of 1/G vs L gives σ (Drude conductivity). The MFP then follows from:

```
ℓ = σ / (n e v_F)        (Drude formula)
```

The package extracts σ directly from the G(L) fit, reports the fit quality R², and identifies
the transport regime (ballistic, diffusive, or mixed) from the variance of G across the L sweep.

**Physical interpretation:**
The MFP extracted here is **disorder-limited** (elastic/impurity scattering from Anderson
disorder W). This differs from the experimentally reported phonon-limited MFP (~1/T in
TaP/NbP at 300 K). However:

- At low temperature where disorder dominates, the values are directly comparable to experiment.
- Relative comparisons between TaP, NbP, CoSi are valid regardless of temperature.
- The anomalously large MFP in TaP/NbP vs normal metals (×10–40 ratio at room T) is
  reproducible because it originates from the topological suppression of backscattering:
  the chirality constraint prevents momentum reversal for states on a single Fermi arc.

> **Relevant files:**
> - `wtec/transport/conductance.py` — `compute_conductance_vs_length()`
> - `wtec/transport/mfp.py` — `extract_mfp_from_scaling()`, `mfp_from_sigma()`
> - `wtec/analysis/mfp_compare.py` — `summarize_mfp()`, `plot_mfp_comparison()`
> - `wtec/workflow/transport_pipeline.py` — `TransportPipeline.run_mfp_extraction()`

---

### 2.7 Surface vs Bulk Conductance Decomposition

The package can decompose total conductance into surface and bulk contributions using
the local density of states (LDOS) from Kwant:

```
LDOS_surface = Σ_{surface sites} ldos(E)
fraction_surface = LDOS_surface / (LDOS_surface + LDOS_bulk)
```

A large `fraction_surface` at the Fermi energy confirms that transport is arc-dominated
rather than bulk-dominated, which is the microscopic origin of the anomalous MFP.

> **Relevant file:** `wtec/transport/observables.py` — `surface_bulk_decomposition()`

---

## 3. DFT → Wannier → Kwant Pipeline

```
QE SCF  →  QE NSCF  →  pw2wannier90  →  Wannier90  →  _hr.dat
                                                          │
                                               tbmodels.Model.from_wannier_folder()
                                                          │
                                               WannierTBModel (Python)
                                                          │
                                     ┌────────────────────┴───────────────────┐
                                     │                                         │
                              bands(k_path)                       to_kwant_builder(
                              (band structure check)               n_layers_z=d,
                                                                   n_layers_y=Ny,
                                                                   n_layers_x=L)
                                                                         │
                                                                  kwant.Builder
                                                                  (hard-wall slab)
                                                                         │
                                                            add_anderson_disorder(W)
                                                                         │
                                                                  sys.finalized()
                                                                         │
                                                               kwant.smatrix(E)
                                                                         │
                                                             G = Tr[t†t]  (e²/h)
                                                                         │
                                                              ρ = d / (G·A)
```

### Critical requirement: SOC must be enabled in QE

The entire topology rests on SOC-driven band inversion. The QE NSCF run must use:

```fortran
noncolin = .true.
lspinorb = .true.
```

Without SOC the `_hr.dat` is a spinless Hamiltonian. Weyl nodes do not exist in spinless
time-reversal-invariant systems (Nielsen-Ninomiya theorem: nodes cancel in pairs and gap out
unless protected by a non-symmorphic or chiral symmetry combined with SOC).

---

## 4. Orbital Projections and Wannierization Quality

### Projection strategy

The number of Wannier functions `num_wann` is determined by the orbital projections
with SOC (each spatial orbital contributes 2× spinor WFs):

| Material | Projections  | WFs per projection | Total num_wann |
|----------|--------------|--------------------|----------------|
| TaP/NbP  | Ta(Nb):d (×5) + P:p (×3) | 10 + 6 (with SOC) | **18** |
| CoSi     | Co:d (×5) + Si:sp3 (×4)  | 10 + 8 (with SOC) | **18** (but preset uses 16*) |

*CoSi preset sets `num_wann=16`; adjust if disentanglement is not converging.

### Disentanglement windows

The preset energy windows (relative to Fermi energy) are:

| Material | dis_win        | dis_froz_win  | Purpose |
|----------|----------------|---------------|---------|
| TaP/NbP  | (−4, +16) eV   | (−1, +1) eV   | Capture d-band complex + freeze Fermi-level states |
| CoSi     | (−6, +10) eV   | (−1.5, +1.5) eV | Include full d+sp3 manifold |

The frozen window should always bracket the Fermi-level Weyl/multifold crossings so that
the topologically non-trivial states are reproduced exactly by the Wannier Hamiltonian.

> **Relevant files:**
> - `wtec/config/materials.py`
> - `wtec/qe/lcao.py`

---

## 5. Anderson Disorder and Transport Regimes

Anderson disorder (on-site random energy W·ξ, ξ ∈ [−0.5, 0.5]) models impurity scattering.
The disorder strengths swept are W = 0, 0.1, 0.3, 0.5, 1.0 eV by default.

### What each disorder level represents

| W (eV) | Physical analogue | Expected MFP regime |
|--------|-------------------|---------------------|
| 0      | Clean limit (ballistic) | G independent of L |
| 0.1    | Low impurity / low T | Ballistic to mildly diffusive |
| 0.3–0.5 | Moderate disorder (RT impurities) | Diffusive; G ~ 1/L |
| 1.0    | Strong disorder (near Anderson transition) | Localisation onset |

### MFP from G(L) sweep

The lengths used for MFP extraction default to 5–100 unit cells in steps of 5,
with a fixed film thickness of `mfp_n_layers_z` = 10 unit cells.

### Ensemble averaging

For each (W, d, L) point, `n_ensemble` (default: 50) independent disorder realisations
are computed. Seeds are `base_seed + i` for i = 0..n_ensemble−1. Parallelism is by
MPI rank splitting (rank i computes indices i, i+size, i+2·size, …) — no forked processes.

---

## 6. Known Limitations and Validity Conditions

| Issue | Impact | Mitigation |
|-------|--------|------------|
| `ry ≠ 0` hopping exclusion in `_iter_hoppings` | **Critical**: collapses 3-D topology to 1-D; Fermi arcs destroyed | Include all `(rx, ry, rz)` hoppings; build slab with `n_layers_y ≥ 4` |
| `lead_onsite_eV = 0` (metallic leads) | Surface arc states mix with lead modes; contact resistance masks surface transport | Use large `lead_onsite_eV` (≫ bandwidth) for insulating substrate contacts |
| Anderson disorder = elastic scattering | MFP is impurity-limited, not phonon-limited (experiment) | Valid at low T or for relative material comparisons |
| `n_layers_y = 1` (default before fix) | No transverse degrees of freedom; unphysical 2-D model | Use `n_layers_y ≥ 4` to resolve Fermi arc dispersion in k_y |
| No Zeeman / magnetic field | Cannot probe chiral anomaly (negative MR) | Extend with `kwant.continuum` gauge field or Peierls substitution |
| Cross-section area `A = |a| × |b|` | Only one unit cell transverse; underestimates true film width | Multiply by `n_layers_y` in conductivity conversion |

---

## 7. Expected Physical Outcomes

If all validity conditions are met (SOC on, full 3-D hoppings, n_layers_y ≥ 4,
adequate Wannierization), the simulation should reproduce:

1. **ρ(d) non-monotonic minimum** at d ~ 10–30 nm for TaP/NbP (Fermi arc hybridisation
   threshold) — absent in Fuchs-Sondheimer reference.

2. **Anomalously large MFP** in TaP/NbP relative to Cu/Au at comparable disorder: ratio
   ℓ_TaP / ℓ_Cu ~ 10–40 at low disorder. Origin: chiral protection of arc states suppresses
   backscattering (no time-reversed partner on the same arc).

3. **CoSi MFP < TaP/NbP** but still > conventional metals, consistent with shorter arcs
   from the topological charge-2 fermions at Γ/R.

4. **surface_fraction → 1** as d decreases below bulk MFP, confirming arc dominance.

5. **Ballistic → diffusive crossover** as W increases from 0 to 1 eV, visible in G(L) shape.

---

## 8. Quick Reference: Module Map

```
wtec/
├── config/
│   ├── materials.py          # MATERIALS dict: TaP, NbP, CoSi presets (num_wann, dis_win, etc.)
│   └── cluster.py            # ClusterConfig: SSH / PBS settings
├── qe/
│   ├── inputs.py             # QEInputGenerator: scf.in / nscf.in generation (SOC flags)
│   └── lcao.py               # PROJECTION_LIBRARY: orbital → Wannier90 projections block
├── wannier/
│   ├── parser.py             # read_hr_dat(), HoppingData, interpolate_bands()
│   └── model.py              # WannierTBModel: _hr.dat → Kwant builder (full 3-D hoppings)
├── transport/
│   ├── builder.py            # build_trilayer(): convenience wrapper
│   ├── conductance.py        # compute_conductance_vs_thickness/length(), MPI ensemble
│   ├── disorder.py           # add_anderson_disorder(), add_vacancy_disorder()
│   ├── mfp.py                # extract_mfp_from_scaling(), mfp_from_sigma()
│   └── observables.py        # rho_from_conductance(), fuchs_sondheimer_rho(),
│                             #   surface_bulk_decomposition()
├── analysis/
│   ├── thickness_scan.py     # plot_rho_vs_thickness(), detect_rho_minimum()
│   └── mfp_compare.py        # REFERENCE_METALS, summarize_mfp(), plot_mfp_comparison()
└── workflow/
    ├── transport_pipeline.py # TransportPipeline: run_thickness_scan(), run_mfp_extraction()
    └── orchestrator.py       # TopoSlabWorkflow: full DFT→Wannier→transport state machine
```

---

## 9. Implementation Notes (2026-03)

- Transport geometry defaults are now axis-explicit:
  - primary topological transport: **in-plane x**
  - slab-thickness sweep axis: **z**
  - auxiliary through-plane channel: **z-lead** (reported separately)
- Resistivity is computed consistently as:
  - `ρ = L / (G_SI * A)` where `L` is along lead axis and `A` is normal to it.
- The topology analysis pipeline now runs under `ANALYSIS` and writes:
  - `run_dir/topology/topology_deviation.json`
  - `run_dir/topology/topology_deviation.csv`
  - `run_dir/topology/failed_points.csv`
- Composite scoring uses coupled topology logic to avoid node/arc double counting:
  - `S_topo = 1 - (1 - S_arc) * (1 - delta_node)`
  - `S_total = 0.70 * S_topo + 0.30 * S_transport`
