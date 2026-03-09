# wtec: Complete Physics Basis and Implementation Reference

**Wannier–Tight-binding Electronic-property Calculator**
Document version: 2026-03-09
Status: authoritative design reference

---

## Table of Contents

1. [Scientific Objective](#1-scientific-objective)
2. [Physics Foundation: Topological Semimetals](#2-physics-foundation-topological-semimetals)
3. [Stage 1 — Crystal Symmetry and Topological Classification](#3-stage-1--crystal-symmetry-and-topological-classification)
4. [Stage 2 — DFT Electronic Structure](#4-stage-2--dft-electronic-structure)
5. [Stage 3 — Wannier Tight-Binding Model](#5-stage-3--wannier-tight-binding-model)
6. [Stage 4 — Topology Extraction from HR](#6-stage-4--topology-extraction-from-hr)
7. [Stage 5 — Kwant Slab and Landauer Transport](#7-stage-5--kwant-slab-and-landauer-transport)
8. [Stage 6 — Defect Modeling and Severity](#8-stage-6--defect-modeling-and-severity)
9. [Stage 7 — Composite Topology Deviation Score](#9-stage-7--composite-topology-deviation-score)
10. [Stage 8 — Report and Observable Extraction](#10-stage-8--report-and-observable-extraction)
11. [Target Observable: ρ(d) with Defect Sensitivity](#11-target-observable-d-with-defect-sensitivity)
12. [Implementation Plan: All Phases](#12-implementation-plan-all-phases)

---

## 1. Scientific Objective

wtec computes how defect engineering at the substrate–film interface modifies the **topological transport signature** of a Weyl semimetal thin film.

The target observable is the **in-plane resistivity ρ(d)** as a function of film thickness d, for a substrate/TaP/substrate trilayer geometry (canonical: SiO₂/TaP/SiO₂). The expected physics is:

- **Conventional metal (Fuchs–Sondheimer):** ρ increases as d decreases, because surface scattering reduces the effective mean free path.
- **Topological Weyl semimetal:** ρ decreases as d decreases at intermediate thickness, because chirally protected Fermi arc surface states contribute a conductance channel G_arc that is roughly independent of d. As the bulk channel G_bulk ∝ d shrinks, the arc fraction grows.

The three-curve figure — pristine, low-defect, high-defect — demonstrates both the topological signature and its sensitivity to interface disorder. This is the primary scientific output of wtec.

---

## 2. Physics Foundation: Topological Semimetals

### 2.1 Weyl Nodes

A Weyl semimetal hosts band crossings (Weyl nodes) that cannot be gapped by any perturbation preserving the protecting symmetry. Each node carries a **topological charge** (chirality):

```
χ = (1/2π²) ∮_S d²k  Ω(k) · n̂
```

where S is any closed surface enclosing the node and Ω(k) is the **Berry curvature**:

```
Ω_n(k) = -2 Im Σ_{m≠n}  <u_nk|∂H/∂k_α|u_mk> × <u_mk|∂H/∂k_β|u_nk>
                           ───────────────────────────────────────────
                                      (E_m - E_n)²
```

The Berry curvature diverges at Weyl nodes (monopoles of Berry flux). χ = ±1 for Weyl semimetals; multifold fermions (CoSi) carry |χ| = 2 or 4.

**Conservation law:** The net chirality over all nodes is zero. Nodes come in pairs of opposite chirality and cannot individually be removed without pair annihilation.

### 2.2 Materials in wtec

| Material | Space group | Symmetry | Node type | N_nodes | χ |
|---|---|---|---|---|---|
| TaP | I4₁md (109) | No inversion, no TR breaking | Weyl type-I | 24 (8 W1 + 16 W2) | ±1 |
| NbP | I4₁md (109) | Isostructural to TaP | Weyl type-I | 24 | ±1 |
| CoSi | P2₁3 (198) | Chiral cubic | Multifold | — | ±2 (Γ), ∓4 (R) |

TaP W1 nodes: sit near the Σ line at generic k_z ≈ 0.42 × 2π/c, off all high-symmetry points.
TaP W2 nodes: near the Σ-S line at generic k_xy, k_z.
**Consequence:** A k-mesh with k_z = 1 (2D mesh) samples zero Weyl nodes.

### 2.3 Fermi Arc Surface States

The **bulk–boundary correspondence** of Weyl semimetals guarantees surface states on any boundary that terminates the crystal. For a (001) surface of TaP:

- Consider a 2D BZ slice at fixed k_z between a W+ and W− node.
- This 2D Hamiltonian has **Chern number C = ±1**, which by the 2D bulk–boundary correspondence implies a **chiral edge mode** on the surface.
- As k_z sweeps from 0 to 2π/c, C(k_z) changes by ±1 at each Weyl node.
- The resulting surface states form open arcs connecting W+ to W− projections in the surface BZ.

Fermi arc properties:
- **Chirality protection:** each arc is a chiral 1D mode at the surface. Elastic backscattering within the arc is forbidden by topology (analogous to quantum Hall edge states but open rather than closed).
- **Penetration depth:** λ_arc ≈ ℏv_F⊥ / Δ_gap where Δ_gap is the local bulk gap at the arc k_∥ position. For TaP: λ_arc ≈ 40–70 Å ≈ 3–6 unit cells.
- **Arc hybridization at thin d:** when d < 2λ_arc, top and bottom arc wavefunctions overlap → hybridization gap Δ(d) = Δ₀ exp(−d/λ_arc) opens → arcs become partially gapped.

---

## 3. Stage 1 — Crystal Symmetry and Topological Classification

**Physics:** Before any DFT, the space group of the target material determines:
1. Whether Weyl/multifold nodes are symmetry-protected.
2. The k-space location of the nodes (high-symmetry vs. generic k).
3. Which surface terminates to show arcs.
4. The minimum k-mesh needed to sample all nodes in DFT/Wannier.

**wtec module:** `wtec.config.materials.MATERIAL_TOPOLOGY`

**Inputs:** Material name (e.g. "TaP"), space group number.

**Outputs per material:**
- `material_class`: "weyl" | "multifold"
- `min_kgrid_nscf`: minimum (n₁, n₂, n₃) to resolve generic-k nodes
- `arc_k_width_inv_A`: arc width in k-space (Å⁻¹), sets minimum n_layers_y
- `arc_penetration_depth_uc`: λ_arc in unit cells, sets minimum n_layers_z for arc visibility
- `recommended_thickness_sweep_uc`: list of d values spanning hybridized→separated→bulk regimes

**Physics-driven parameter derivation:**

```
min_kgrid_z for TaP:
  W1 nodes at k_z ≈ 0.42 × 2π/c, k_z resolution needed: δk_z < 0.1 Å⁻¹
  n_kz ≥ 2π/(c × δk_z) = 2π/(11.89 Å × 0.1 Å⁻¹) ≈ 5 → use 12

recommended n_layers_y for TaP:
  Arc width Δk_arc ≈ 0.15 Å⁻¹ (W1–W2 separation projected on (001))
  Required slab width: W >> 2π/Δk_arc ≈ 42 Å
  With a = 3.30 Å: n_layers_y ≥ 42/3.30 ≈ 13 → use 16

thickness sweep range for TaP:
  λ_arc ≈ 5 unit cells → hybridization at d < 10 uc
  Bulk crossover at d > 20 uc
  Sweep: [2, 4, 6, 8, 10, 12, 16, 20, 25]
```

---

## 4. Stage 2 — DFT Electronic Structure

### 4.1 Role in the workflow

DFT provides the Kohn–Sham eigenvalues E_n(k) and wavefunctions |ψ_nk⟩ from which Wannier functions are constructed. The topology is entirely encoded in the DFT band structure; wtec does not compute topology from DFT directly but uses DFT as an input to the TB model.

### 4.2 Engine assignment (hybrid QE + SIESTA)

| Variant | Engine | Reason |
|---|---|---|
| Pristine bulk unit cell | QE pw.x | Mature SOC+Wannier interface, small cell (TaP: 2 atoms), well-validated |
| Defect supercell (2×2×2 bulk) | SIESTA | LCAO scales as O(N log N) for large cells; mature for SOC via noncollinear spin |
| Actual slab (substrate/film/substrate) | SIESTA | Direct surface band structure; gives arc observable without Kwant proxy |

### 4.3 QE (pristine bulk) — required settings

```
# SCF: symmetry ON for speed
calculation = 'scf'
nosym = .false.

# NSCF: symmetry OFF — required for Wannier90 (uniform k-grid, all k-points)
calculation = 'nscf'
nosym = .true.
noinv = .true.

# SOC: mandatory for Weyl semimetals
noncolin = .true.
lspinorb = .true.

# k-mesh: must resolve Weyl nodes at generic k
kpoints_nscf = [12, 12, 12]   # NOT [4,4,1] — that samples zero TaP nodes

# Energy cutoffs: conservative for Ta, Nb (semicore d-electrons)
ecutwfc = 80   # Ry
ecutrho = 640  # Ry
```

**Physics of nosym requirement:** Wannier90 needs a uniform Monkhorst–Pack grid with all k-points explicitly present (no symmetry reduction). If symmetry is used in NSCF, the Bloch states at symmetry-related k-points are not independently stored, and the Fourier transform to real space breaks down.

### 4.4 SIESTA (defect supercell) — required settings

```fortran
# Spin-orbit coupling
SpinOrbit              .true.
Spin                   SO
NonCollinearSpin       .true.

# DFT-D3 dispersion (important for layered substrate interface)
%block vdW-correction
  DFT-D3
%endblock
vdW-D3-damping         BJ

# PAO basis: must include enough angular momentum for heavy elements
%block PS.lmax
  Ta  3    # d-electrons active
  P   2
%endblock

# Wannier interface (builtin SIESTA or via sisl post-processing)
Siesta2Wannier90.WriteMmn    .true.
Siesta2Wannier90.WriteAmn    .true.
Siesta2Wannier90.NumberOfWannierFunctions  18
Wannier.SeedName             TaP_defect
```

### 4.5 Fermi level

After SCF, QE reports `Fermi energy = X eV` in the output. SIESTA reports it in `.EIG` or output log. This value is stored in the wtec checkpoint as `dft.fermi_ev` and passed to the Wannier and TB model stages. All downstream energies are referenced to this Fermi level (ε = 0 at E_F).

---

## 5. Stage 3 — Wannier Tight-Binding Model

### 5.1 Wannier functions

The maximally localized Wannier functions (MLWFs) are defined by:

```
|w_n,R⟩ = (V_cell/(2π)³) ∫_BZ dk  e^{−ik·R}  Σ_m U^(k)_{mn} |ψ_mk⟩
```

where U^(k) is a unitary gauge matrix chosen to minimize the spread functional:

```
Ω = Σ_n [<r²>_n − <r>²_n]  =  Ω_I + Ω̃
```

Ω_I is gauge-invariant (depends only on the subspace); Ω̃ is gauge-dependent and minimized to zero for smooth Wannier functions. Only smooth (rapidly decaying) Wannier functions give a valid TB model: long-range hoppings alias the Berry curvature correctly.

### 5.2 HR matrix (Hamiltonian in real space)

The TB model is stored as the matrix of hopping integrals:

```
H_{mn}(R) = <w_m,0|H_KS|w_n,R>  =  t_{mn}(R)
```

The k-space Hamiltonian is recovered by:

```
H(k) = Σ_R  t(R) exp(ik·R)   [convention 2: k in fractional, phase = exp(2πi k·R)]
```

**Topology is encoded in t(R):** The Berry curvature and Weyl node positions are determined entirely by the long-range hoppings. If Wannierization fails (non-converged disentanglement), the long-range hoppings are wrong and the topology is destroyed.

### 5.3 Required Wannier parameters for TaP

```ini
# wannier90.win

num_wann = 18          # Ta: 5d×5 + P: 3p×3 = 13, but with hybridization/entanglement: 18
num_bands = 72         # all bands from SOC NSCF (36 electrons × 2 for SOC)

# Disentanglement window — must span the full Weyl manifold
dis_win_max = 3.0      # eV above E_F
dis_win_min = -3.0     # eV below E_F
dis_froz_max = 0.5     # inner frozen window: bands here are never mixed out
dis_froz_min = -0.5

# Convergence — tighter than default for topology
dis_num_iter = 1000
dis_conv_tol = 1.0e-10
num_iter = 500
conv_tol = 1.0e-10

# Smooth gauge for Berry curvature
guiding_centres = .true.
```

**Physics of the frozen window:** Bands within the frozen window [−0.5, 0.5] eV always project fully into the Wannier subspace. This ensures the Weyl nodes (at ~0–50 meV above E_F in TaP) are never mixed out during disentanglement.

### 5.4 Convergence gate (hard requirement)

Two conditions must hold before any downstream stage runs:

**Condition 1 — Disentanglement convergence:**
The Wannier90 output (.wout) must contain "CONVERGENCE REACHED" for the disentanglement stage. The string "Maximum number of disentanglement iterations reached" without convergence indicates failure.

**Condition 2 — Topological sanity check:**
After Wannierization, compute the Chern number on a 2D k_z slice located between the expected Weyl node positions. For TaP:

```python
# k_z = 0.3 (fractional) is between W1 and W2 in the BZ
C = chern_number_on_kz_slice(hr_dat, kz_frac=0.3)
assert |C| >= 1, "Wannier functions did not capture Weyl topology"
```

If C = 0, the Wannierization captured the wrong band manifold. This gate catches wrong energy windows, wrong num_wann, or unconverged disentanglement that happens to satisfy the convergence string but is topologically wrong.

---

## 6. Stage 4 — Topology Extraction from HR

### 6.1 Weyl node detection (Chern number profile)

The Weyl node k_z positions are detected by sweeping the Chern number C(k_z) across the BZ:

```
C(k_z) = (1/2π) ∫∫ dk_x dk_y  Ω_z(k_x, k_y, k_z)
```

where Ω_z is the z-component of the Berry curvature. C(k_z) is an integer that changes by ±1 each time k_z crosses a Weyl node:

```
k_z sweep:   0 ──── W1(χ=+1) ──── W2(χ=−1) ──── π/c
C(k_z):      0 ────────── +1 ────────── 0 ─────── 0
```

This directly maps the Weyl node positions and chiralities without requiring a high-symmetry search.

**Implementation:** WannierBerri `compute_chern_numbers` on 2D k-meshes at each k_z slice.

**δ_node (defect comparison):**

```
δ_node = |k_z(W1)_defect − k_z(W1)_pristine| / (2π/c)
```

A shift of the Weyl node in k-space indicates that defects have modified the band topology. If δ_node > 0.05 (5% of BZ), the defect has substantially perturbed the topological protection. If δ_node → 0.5 (node moved to annihilation partner), topology is destroyed.

### 6.2 Arc connectivity metric (S_arc)

The Fermi arc spectral weight is measured from the Kwant slab surface LDOS:

```
ρ_surf(E) = Σ_{i ∈ surface} |<i|G^R(E)|i>|²
ρ_bulk(E) = Σ_{i ∈ interior} |<i|G^R(E)|i>|²

S_arc = ρ_surf(E_F) / [ρ_surf(E_F) + ρ_bulk(E_F)]
```

For a slab with well-defined arc surface states: ρ_surf >> ρ_bulk at E_F → S_arc → 1.
For a gapped or strongly disordered system: ρ_surf ≈ ρ_bulk → S_arc ≈ 0.5 (no surface enhancement).

**Requirements for valid S_arc:**
- n_layers_z ≥ 10 (arcs penetrate ~5 uc, need buffer)
- n_layers_y ≥ 16 (arc k-space width requires real-space resolution)
- Energy at ε = E_F = 0

### 6.3 Composite topology deviation score

```
S_arc  ∈ [0, 1]:  arc connectivity (1 = strong arcs, 0 = no arcs)
δ_node ∈ [0, 1]:  Weyl node shift (0 = no shift, 1 = node annihilated)

# Probabilistic union (avoids double-counting):
S_topo = 1 − (1 − S_arc)(1 − δ_node)
       = S_arc + δ_node − S_arc × δ_node

# Transport contribution:
S_transport = |σ_arc_2D_defect − σ_arc_2D_pristine| / σ_arc_2D_pristine

# Final score:
S_total = 0.7 × S_topo + 0.3 × S_transport
```

**Partial failure rules:**
- If S_arc missing: S_topo = δ_node (node-only estimate; confidence = "partial-arc")
- If δ_node missing: S_topo = S_arc (arc-only estimate; confidence = "partial-node")
- If both missing: S_total = NaN; status = "failed" (no imputation)
- If S_transport missing: weight reallocated to S_topo (0.7 → 1.0)

---

## 7. Stage 5 — Kwant Slab and Landauer Transport

### 7.1 Slab construction from Wannier HR

The finite slab is constructed from the TB model by tiling the unit cell:

```python
lat = kwant.lattice.general(lattice_vectors, norbs=num_wann)
# Sites: lat(ix, iy, iz) for ix ∈ [0, n_x), iy ∈ [0, n_y), iz ∈ [0, n_z)

# Onsite: H(k=0) = Σ_R t(R)  [sum of all hopping matrices at zero momentum]
h0 = model.hamilton([0,0,0], convention=2)

# Hoppings: for each R = (rx, ry, rz), add t(R) between sites (ix,iy,iz) and (ix+rx,iy+ry,iz+rz)
```

**Critical:** All three hopping directions (rx, ry, rz) must be included. Omitting any direction breaks the topology by discarding the 3D hybridization that gives rise to Weyl nodes.

### 7.2 Lead geometry

Semi-infinite leads attach along the lead_axis (default: x). Each lead spans the full transverse face of the slab:

```
Lead cross-section: all (iy, iz) with ix = 0 (left lead) or ix = n_x − 1 (right lead)
Lead period: a_x (one unit cell along x)
```

**Minimum geometry requirement:** n_layers_x ≥ 2. With n_layers_x = 1, the scattering region is a single-atom-thick slice and does not "interrupt" the lead — Kwant raises "does not interrupt the lead." This is a hard geometric constraint, not a convergence issue.

### 7.3 Landauer–Büttiker conductance

```
G = (e²/h) Tr[t†t] = (e²/h) Σ_n T_n
```

where t is the transmission matrix from right lead to left lead, and T_n ∈ [0, 1] is the transmission eigenvalue of channel n.

**Two-channel decomposition:**

In a clean slab (no disorder):
- **Arc channels:** each Fermi arc state crossing E_F contributes one transmission channel with T ≈ 1 (ballistic, chirally protected).
- **Bulk channels:** each bulk band propagating mode also contributes T ≈ 1. The number of bulk modes scales as N_bulk ∝ W × d / (λ_F)².

Total conductance:
```
G(d) = G_arc + G_bulk(d)
G_arc ≈ N_arc × e²/h       (≈ 12 × e²/h for TaP (001), ballistic)
G_bulk ≈ σ_3D × (W × d)/L  (ohmic, scales with cross-section)
```

### 7.4 Resistivity formula

In-plane 3D resistivity (units: Ω·m):

```
ρ_3D(d) = L / [G(d) × e²/h × W × d]
         = d / σ_sq(d)

σ_sq(d) = G(d) × L/W  (sheet conductance, units: e²/h per unit width)
         = σ_arc_2D + σ_bulk × d  (two-channel linear fit)
```

As d → 0: ρ_3D → 0 (arc-dominated: ρ ∝ d)
As d → ∞: ρ_3D → 1/σ_bulk = ρ_bulk (bulk-dominated: constant)

**ρ_3D decreases monotonically as d decreases** — this is the topological transport signature.

**Fuchs–Sondheimer reference (conventional metal):**

```
ρ_FS(d) ≈ ρ_bulk × [1 + 3l_mfp / (4d)]    for d << l_mfp (surface scattering limit)
```

ρ_FS increases without bound as d → 0. The difference ρ_FS(d) − ρ_topo(d) is the quantitative measure of topological surface conductance.

### 7.5 Arc hybridization at thin d

When d < 2λ_arc (hybridized regime), top and bottom arc states overlap:

```
Hybridization gap: Δ(d) = Δ₀ exp(−d/λ_arc)
Effective arc conductance: G_arc(d) = N_arc × (e²/h) × exp(−Δ(d) / k_BT)  [at finite T]
```

At T = 0 (Kwant default): the hybridization gap completely suppresses arc transmission for d < d_c → ρ(d) has a local maximum at d = d_c ≈ 2λ_arc ≈ 10 unit cells for TaP.

The full non-monotonic ρ(d) curve:
```
Large d (d > 20 uc):  ρ ≈ ρ_bulk              (bulk dominated, ρ flat)
Intermediate (10–20):  ρ decreasing as d→0     (arc channel contribution)
Small (d < 10 uc):     ρ increasing as d→0     (arcs hybridize, gap opens)
                        local minimum at d ≈ 10 uc
```

---

## 8. Stage 6 — Defect Modeling and Severity

### 8.1 Structural defects and their electronic effect

Interface defects at SiO₂/TaP boundary:

| Defect type | Mechanism | Effect on arcs | Effect on bulk |
|---|---|---|---|
| O vacancy (SiO₂ side) | Local potential fluctuation | Scatters arc at interface | Reduces surface barrier |
| Ta vacancy (TaP side) | Missing hopping terms + potential | Can shift Weyl node k_z | Reduces bulk channel |
| O substitution on Ta | Different onsite energy + SOC | Hybridizes arc with impurity level | Increases disorder scattering |
| Epitaxial strain | Uniform k-space shift of nodes | Shifts W1/W2 positions | Changes bulk band widths |

### 8.2 Defect severity formula

```python
events = n_vacancies + n_substitutions   # NOT vacancies + atoms_removed (double count)
denom = atoms_in_window                  # atoms in the interface region only, not full cell
severity = min(1.0, events / (0.05 × denom))
```

Calibration: 5% of interface atoms defective = severity 1.0 (completely disordered interface). This is physically motivated: at 5% vacancy density, the local potential landscape is strongly disordered and arc backscattering becomes significant.

### 8.3 Anderson disorder model in Kwant

The structural defect distribution is modeled as random onsite energy shifts:

```
H_disorder = Σ_i  ε_i c†_i c_i,    ε_i ~ U[-W/2, W/2]
```

For **interface defects** (O vacancies): disorder is concentrated in the top/bottom n_surface_layers of the TaP slab. Arc wavefunctions are concentrated at the surface → surface disorder disproportionately scatters arcs.

For **bulk defects** (substitutions in TaP): disorder is applied uniformly.

```python
add_anderson_disorder(sys, W_bulk, rng,
                      surface_disorder_strength=W_surface,
                      n_surface_layers=2)
```

**Disorder strength from severity:**

```
W_surface = W_max × severity    where W_max ≈ 1.0 eV (characteristic energy scale of defect potential)
W_bulk    = W_max × severity × 0.1  (bulk is less strongly perturbed by interface vacancies)
```

### 8.4 Per-variant HR requirement

The defect changes the bulk electronic structure of TaP. To capture this:
1. Run SIESTA DFT on a defect supercell (2×2×2 bulk-periodic, NOT a slab).
2. Extract _hr.dat from this defect supercell.
3. Use the defect HR to build the defect TB model in Kwant.

This captures **node position shifts** due to defects (which Anderson disorder cannot). Anderson disorder then adds the **local scattering** on top of the already-modified band structure.

**hr_scope = "shared" is physically invalid:** if pristine and defect use the same HR, the Weyl nodes are in the same place for both variants → δ_node = 0 by construction → the topology comparison is trivially zero.

---

## 9. Stage 7 — Composite Topology Deviation Score

### 9.1 Score components

```
S_arc:       surface LDOS fraction [0,1]  — measures arc spectral weight at E_F
δ_node:      Weyl node k-shift [0,1]      — measures node position change
S_transport: σ_arc_2D change [0,1]        — measures arc conductance loss
severity:    interface defect fraction [0,1] — structural input (not electronic)
```

### 9.2 Score composition

```
S_topo = 1 − (1 − S_arc)(1 − δ_node)
S_total = 0.7 × S_topo + 0.3 × S_transport
```

The probabilistic union formula for S_topo avoids double-counting: if both S_arc and δ_node are independently large, they don't simply add. Instead, S_topo saturates at 1 as either component reaches 1.

**Physical interpretation of S_total:**
- S_total ≈ 0: defect has no measurable effect on topology or transport
- S_total ≈ 0.5: partial degradation of arc connectivity or node shift
- S_total ≈ 1: topology effectively destroyed by defects

### 9.3 Confidence levels

| Condition | Confidence label |
|---|---|
| S_arc and δ_node both computed | "full" |
| S_arc missing, δ_node present | "partial-arc" |
| δ_node missing, S_arc present | "partial-node" |
| S_transport missing | "partial-transport" |
| Both S_arc and δ_node missing | "failed" — S_total = NaN |

---

## 10. Stage 8 — Report and Observable Extraction

### 10.1 Transport report fields

```json
{
  "fermi_ev": float,                    // from DFT checkpoint, never null
  "fermi_ev_source": "qe_scf",
  "transport": {
    "G_mean":                 [...],    // conductance in e²/h per thickness
    "rho_mean":               [...],    // 3D resistivity in Ω·m per thickness
    "sigma_arc_2D":           float,    // topological intercept of σ_sq fit (e²/h per Å)
    "sigma_bulk":             float,    // bulk 3D conductivity (e²/h per Å²)
    "fit_R2":                 float,    // goodness of two-channel fit
    "d_crossover_uc":         float,    // thickness where G_arc = G_bulk
    "rho_fuchs_sondheimer":   [...],    // FS reference at same thicknesses
    "arc_fraction_at_min_d":  float     // G_arc / G_total at thinnest film
  },
  "topology": {
    "S_arc":                  float,
    "delta_node":             float,
    "S_topo":                 float,
    "S_total":                float,
    "n_weyl_nodes_detected":  int,
    "node_positions_kz_frac": [...],
    "chern_profile":          [...]
  }
}
```

### 10.2 Primary figure: ρ(d) three-curve plot

Required curves:
1. **Pristine TaP:** ρ_pristine(d) — shows decreasing ρ as d decreases
2. **Low-defect:** ρ_defect_low(d) — slightly elevated, same shape
3. **High-defect:** ρ_defect_high(d) — ρ approaches ρ_bulk flat line
4. **Fuchs–Sondheimer reference:** ρ_FS(d) — diverges as d → 0

Annotations:
- Marker at d_crossover where arc dominates (ρ begins decreasing)
- Inset: σ_sq vs d linear fit with σ_arc_2D intercept labeled per variant
- Arrow indicating direction of increasing defect severity

### 10.3 Secondary figures

- Chern(k_z) profile: C vs k_z for pristine and defect variant
- S_arc vs thickness for each variant
- Disorder ensemble spread: G_mean ± G_std per thickness

---

## 11. Target Observable: ρ(d) with Defect Sensitivity

### 11.1 Why ρ decreases as d decreases (derivation)

```
σ_sq(d) = G(d) × L/W = σ_arc_2D + σ_bulk × d

ρ_3D(d) = d / σ_sq(d) = d / (σ_arc_2D + σ_bulk × d)

∂ρ_3D/∂d = σ_arc_2D / (σ_arc_2D + σ_bulk × d)² > 0
```

ρ_3D is monotonically increasing in d → monotonically decreasing in d as d decreases. This holds for any σ_arc_2D > 0, regardless of magnitude.

The arc conductance also decreases σ_sq more than the bulk as d → 0:

```
σ_sq → σ_arc_2D / d as d → 0    [arc term dominates]
ρ_3D → d / σ_arc_2D → 0
```

Compared to ρ_FS → ρ_bulk × (3l/4d) → ∞ as d → 0.

### 11.2 Defect effect on the curve

**At small d** (arc-dominated regime): G ≈ G_arc(W)

```
G_arc(W) = N_arc × (e²/h) × exp(−L/l_arc(W))
l_arc(W) ≈ ℏv_arc / W²  (Born approximation for potential disorder)
```

Higher disorder W → shorter l_arc → lower G_arc → higher ρ at small d.

**At large d** (bulk-dominated regime): G ≈ G_bulk(W)

```
G_bulk(W) = σ_bulk(W) × W × d / L
σ_bulk(W) decreases with W (conventional Drude scattering)
```

Higher disorder → lower σ_bulk → higher ρ at large d, but less sensitive than at small d.

**Result:** Defect curves lie above the pristine curve at all d. The separation is largest at small d (where arc contribution is dominant), shrinks at large d. This makes the arc-dominated regime the most sensitive probe of interface defect quality.

---

## 12. Implementation Plan: All Phases

The following implementation plan flows from the physics analysis above. Items are ordered by blocking priority: Phase 0 phases cannot start until previous phases are complete.

---

### Phase 1 — Scientific Guardrails (no physics without these)

These are targeted edits to existing files. No new modules required. Phase 1 must be complete and all acceptance tests passing before Phase 2 begins.

| ID | File | Change | Physics reason |
|---|---|---|---|
| 1.1 | `wannier/model.py` | Add `if n_layers_x < 2: raise ValueError(...)` in `to_kwant_builder` | n_layers_x=1 → lead cannot attach → G=0 → ρ=∞ always |
| 1.2 | `transport/conductance.py` | Remove both `return 0.0` silent fallbacks in `_single_conductance` → raise `RuntimeError` | Silent G=0 masks lead-attachment failure; produces infinite ρ without error |
| 1.3 | `topology/arc_scan.py` | Change default `n_layers_x=1→4`, `n_layers_y=4→16`; add `if n_layers_x < 2: raise` | n_layers_x=1 → same lead failure in arc scan; n_layers_y=4 → arc states not resolved |
| 1.4 | `topology/variant_discovery.py` | Fix double-count (remove vacancies[*].applied when atoms_removed present); denominator = `atoms_in_window`; formula = `min(1.0, events/(0.05×denom))` | Double-count + 20× scale → severity=1.0 for 2 vacancies in 44-atom cell |
| 1.5 | `workflow/topology_pipeline.py` | Hard error on `hr_scope="shared"` or `"per_variant_thickness"` | Shared HR → δ_node=0, S_arc identical for all variants by construction |
| 1.6 | `wannier/convergence.py` | Verify/complete: parse both disentanglement and main loop convergence; confirm gate is called in `dft_pipeline.py` | Non-converged HR → invalid arc states → G_arc=0 |
| 1.7 | Config templates + `config/materials.py` | `kpoints_nscf=[12,12,12]`, `n_layers_x=4`, `n_layers_y=16`, `hr_scope="per_variant"`, `failure_policy="strict"`, thickness_sweep=[2,4,6,8,10,12,16,20,25] | TaP k-mesh [4,4,1] samples zero Weyl nodes; wrong defaults propagate to all runs |

**Phase 1 acceptance gate:** `pytest tests/ -k "test_n_layers_x_guard or test_no_silent_g0 or test_hr_scope_error or test_severity_formula or test_wannier_gate"` all pass.

---

### Phase 2 — Signal Correctness (ρ(d) curve must appear and be physically meaningful)

| ID | File | Change | Physics reason |
|---|---|---|---|
| 2.1 | `transport/observables.py` | Add `fit_two_channel_conductance(thickness_uc, G_vals, geometry)` → σ_arc_2D, σ_bulk, R², d_crossover | Direct extraction of topological intercept; required for defect comparison |
| 2.2 | `transport/observables.py` | Add `fuchs_sondheimer_rho(d_m, rho_bulk, l_mfp_m)` → ρ_FS array | Needed as reference: without FS curve, cannot show topological deviation |
| 2.3 | `transport/disorder.py` | Add `surface_disorder_strength`, `n_surface_layers` params to `add_anderson_disorder` | Interface vacancies sit in surface layers; uniform disorder underestimates arc scattering |
| 2.4 | `transport/conductance.py` | Pass `surface_disorder_strength`, `n_surface_layers` through call chain | Same as 2.3 |
| 2.5 | `workflow/dft_pipeline.py` + `orchestrator.py` | Write `fermi_ev` from DFT output to checkpoint immediately after SCF; read from checkpoint in all downstream stages | fermi_ev=None in report is ambiguous and masks energy reference errors |
| 2.6 | `cli.py` | Emit only `Path.exists()` artifacts in report manifest; remove hardcoded `transport_compare.json` | Missing artifacts listed as present misleads report consumers |

**Phase 2 acceptance gate:** Run with valid Wannier HR (converged, k-mesh [12,12,12]). `transport_result.json` contains `sigma_arc_2D > 0`. ρ(d) curve decreases as d decreases from 20→4 unit cells.

---

### Phase 3 — Quantitative Topology (non-proxy, publication-grade)

| ID | File | Change | Physics reason |
|---|---|---|---|
| 3.1 | `topology/node_scan.py` | Implement `compute_chern_profile(hr_dat_path, n_kz, n_kxy)` via WannierBerri; replace proxy | Proxy is not a Berry flux calculation; cannot detect Weyl node positions |
| 3.2 | `wannier/convergence.py` | Add `assert_wannier_topology(hr_dat_path, material_class)` — compute C at k_z probe and raise `WannierTopologyError` if C=0 | Topology sanity: HR can pass convergence string but capture wrong band manifold |
| 3.3 | `config/materials.py` | Add `MATERIAL_TOPOLOGY` dict with per-material: `min_kgrid_nscf`, `arc_k_width_inv_A`, `arc_penetration_depth_uc`, `recommended_n_layers_y`, `recommended_thickness_sweep_uc` | Physics-driven parameter derivation; removes guesswork from config |
| 3.4 | `cli.py` (config validator) | Read `MATERIAL_TOPOLOGY[material_name]` and error if k-mesh below minimum, n_layers_y below recommended | Prevents misconfigured runs from reaching compute |

**Phase 3 acceptance gate:** `compute_chern_profile(valid_TaP_hr.dat)` returns `n_weyl_nodes_detected >= 8`. `assert_wannier_topology` raises for synthetic random HR. δ_node is non-zero when comparing pristine vs vacancy-defect HR.

---

### Phase 4 — Hybrid DFT Engine (QE pristine + SIESTA defect)

| ID | Files | Change | Physics reason |
|---|---|---|---|
| 4.1 | New: `wtec/siesta/` package (6 files) | `inputs.py` (FDF+SOC+D3+Wannier block), `runner.py` (PBS+qsub), `parser.py` (Fermi+convergence), `wannier_bridge.py` (sisl interface), `presets.py` (PAO basis per species) | SIESTA scales better for large defect supercells; LCAO O(N log N) vs QE plane-wave |
| 4.2 | `workflow/dft_pipeline.py` | Engine dispatch: `dft.engine="qe"` for pristine, `dft.variant_engine="siesta"` for defects | Different cells need different DFT codes for efficiency |
| 4.3 | `wannier/model.py` | Accept `fermi_ev` parameter in `from_hr_dat`; shift TB model so E=0 at E_F | QE and SIESTA have different absolute energy zeros; must align before comparison |
| 4.4 | Config templates | Add `[dft.siesta]` section: `executable`, `pseudo_dir`, `basis_size="DZP"` | SIESTA requires explicit config; no default executable on all clusters |

**Phase 4 acceptance gate:** `wtec init` generates `[dft.siesta]` section. SIESTA FDF generator produces valid input for 2×2×2 TaP supercell with SOC and D3. `WannierTBModel.from_hr_dat(path, fermi_ev=0.323)` produces bands with E=0 at Fermi.

---

### Phase 5 — Report and Figure Generation

| ID | File | Change |
|---|---|---|
| 5.1 | `analysis/plots.py` | `plot_rho_vs_thickness(result, variants, outfile)`: three-curve ρ(d) with FS reference; inset σ_sq fit; d_crossover marker |
| 5.2 | Report JSON schema | Add fields: `sigma_arc_2D`, `sigma_bulk`, `d_crossover_uc`, `rho_fuchs_sondheimer`, `arc_fraction_at_min_d`, `chern_profile`, `n_weyl_nodes_detected`, `fermi_ev_source` |
| 5.3 | Report writer | Emit `transport_compare.json` only if multi-variant transport ran; add `topology_valid: bool` flag (false if `caveat_reuse_global_hr_dat=true`) |

---

### Summary: new files and edits

**New files (Phase 4 only):**
```
wtec/siesta/__init__.py
wtec/siesta/inputs.py
wtec/siesta/runner.py
wtec/siesta/parser.py
wtec/siesta/wannier_bridge.py
wtec/siesta/presets.py
```

**Edited files:**
```
Phase 1:  wannier/model.py, transport/conductance.py, topology/arc_scan.py,
          topology/variant_discovery.py, workflow/topology_pipeline.py,
          wannier/convergence.py, config/materials.py

Phase 2:  transport/observables.py (×2 new functions), transport/disorder.py,
          transport/conductance.py, workflow/dft_pipeline.py, cli.py

Phase 3:  topology/node_scan.py, wannier/convergence.py, config/materials.py, cli.py

Phase 4:  workflow/dft_pipeline.py, wannier/model.py, config templates

Phase 5:  analysis/plots.py, report writer in cli.py
```

**Total:** 6 new files, ~14 edited files across 5 phases.

---

### Phase dependency graph

```
Phase 1 (guardrails)
    │
    ├──► Phase 2 (signal correctness)
    │         │
    │         ├──► Phase 3 (quantitative topology)
    │         │
    │         └──► Phase 5 (report figures)
    │                   ▲
    └──► Phase 4 ───────┘
         (SIESTA engine)
```

Phase 1 must complete first. Phases 2 and 4 are independent of each other and can proceed in parallel after Phase 1. Phase 3 requires Phase 2 to be complete (it adds to the topology pipeline that Phase 2 already cleaned up). Phase 5 requires both Phase 2 (transport fields) and Phase 3 (topology fields) but not Phase 4.

---

*End of document.*
