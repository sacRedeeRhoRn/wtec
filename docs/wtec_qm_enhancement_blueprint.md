# WTEC Quantum Mechanics Enhancement Blueprint
## Rigorous Analytic Foundations for Topological Semimetal Modelling

**Document scope:** This blueprint maps every current WTEC modelling step to its exact quantum-mechanical basis, identifies where approximations are made, and proposes algebraic enhancements grounded in solid-state physics. All equations are given in SI-compatible natural units (ℏ = 1 where stated, physical constants explicit otherwise).

---

## 0. Notation and Conventions

| Symbol | Meaning |
|---|---|
| ℏ | Reduced Planck constant (1.0546×10⁻³⁴ J·s) |
| e | Elementary charge (1.6022×10⁻¹⁹ C) |
| G₀ = e²/h | Conductance quantum (7.748×10⁻⁵ S) |
| **k** | Crystal momentum in fractional reciprocal lattice coords |
| **K** | Crystal momentum in Cartesian (Å⁻¹) |
| H(**k**) | Bloch Hamiltonian in Wannier basis |
| |u_n(**k**)⟩ | Periodic part of Bloch state for band n |
| E_F | Fermi energy (eV) |
| d | Film thickness in metres; d_uc in unit cells |
| λ_arc | Fermi-arc penetration depth (unit cells) |
| Ω_n(**k**) | Berry curvature of band n |
| χ | Weyl node chirality (±1) |

---

## 1. The Wannier Tight-Binding Hamiltonian

### 1.1 Current Implementation

WTEC builds H(**k**) from the Wannier90 real-space hopping file `_hr.dat`:

```
H_αβ(**k**) = Σ_R  t_αβ(R) · exp(2πi **k**·**R**)
```

where α, β ∈ {1,...,N_orb} are orbital indices and **R** are Bravais lattice vectors. This is implemented in `arc_scan.py:_surface_spectral_map_from_hoppings` (lines 214–222).

### 1.2 Algebraic Basis — Wannier Functions

The Wannier functions |**R**, α⟩ are related to the Bloch states by:

```
|**R**, α⟩ = (V_BZ / (2π)³) ∫_BZ  e^{-i**k**·**R**} Σ_n U_{nα}(**k**) |ψ_n(**k**)⟩ d**k**
```

where U(**k**) is the N_bands × N_Wan unitary gauge matrix chosen by Wannier90 to maximise real-space localisation (minimise ⟨r²⟩ − ⟨r⟩²).

The spread functional minimised is:

```
Ω = Σ_n [ ⟨r²⟩_n − |⟨**r**⟩_n|² ]
   = Ω_I + Ω̃
```

where Ω_I is gauge-invariant (disentanglement-fixed) and Ω̃ is gauge-dependent (spread minimised by MLWF rotation).

**Enhancement 1.1 — Spread Monitor:** Track Ω_I and Ω̃ per orbital after convergence. Flag if any Wannier centre is displaced > 0.5 Å from expected atomic position. Implement as `wannier/convergence.py:check_wannier_centres(wout_path, lattice_vecs, tol_ang=0.5)`.

### 1.3 Enhancement: Non-Collinear SOC Hamiltonian Block Structure

For TaP/NbP the spin-orbit coupling is essential. The Bloch Hamiltonian is 2×2 in spin space per orbital:

```
H(**k**) = H₀(**k**) ⊗ I₂ + H_SOC(**k**)

H_SOC = Σ_{i} λ_i (**L**_i · **σ**)_projected
```

where **σ** = (σ_x, σ_y, σ_z) are Pauli matrices (already defined in `node_scan.py:_SIGMA_X/Y/Z`) and λ_i is the on-site SOC strength. The current code uses these Pauli matrices for chirality estimation but does not separately extract the SOC block.

**Enhancement 1.3 — SOC Decomposition:** Extract H_SOC = H(**k**) − H(**k**)* (antisymmetric part under time-reversal) and compute the SOC gap at high-symmetry points. This validates that the Wannier model correctly inherits the DFT SOC strength.

---

## 2. Weyl Node Detection

### 2.1 Current Implementation (`node_scan.py`)

The code uses a coarse-refine strategy: scan an N₁×N₂×N₃ k-grid for minimum band gaps, refine via coordinate descent, then compute chirality by either:

**(a) Velocity Jacobian proxy** (`_chirality_proxy`):

```python
J_ij = (1/2) Tr[ u†(∂H/∂k_i)u · σ_j ]   (i,j ∈ {x,y,z})
χ = sign(det J)
```

**(b) Berry flux cube** (`_chirality_flux`):

The discretized Berry flux on a small cube around node **K**₀:

```
Φ_Berry = Σ_{faces} Φ_{face}

Φ_{face} = Im log[ ⟨u(**k**₀₀)|u(**k**₁₀)⟩ · ⟨u(**k**₁₀)|u(**k**₁₁)⟩
                  · ⟨u(**k**₁₁)|u(**k**₀₁)⟩ · ⟨u(**k**₀₁)|u(**k**₀₀)⟩ ]

χ = round(Φ_Berry / 2π)
```

This is the Wilson loop / link product formula for the Chern number on a plaquette.

### 2.2 Exact QM Basis — Weyl Points

A Weyl node occurs when two non-degenerate bands become degenerate at an isolated **k**-point. Near the node, the low-energy effective Hamiltonian is:

```
H_Weyl(**k**) = χ ℏ v_F (**k** − **K**_W) · **σ**
             = χ ℏ Σ_{i,j} v_{ij} (k_i − K_{W,i}) σ_j
```

where **v** is the 3×3 anisotropic velocity tensor and χ = ±1 is the topological charge (chirality). The chirality is a topological invariant:

```
χ = (1/2π) ∮_S² Ω(**k**) · d**S**
  = (1/2π) Σ_{faces of cube} Φ_{face}
```

where the Berry curvature of band n is:

```
Ω_n(**k**) = -2 Im Σ_{m≠n} ⟨u_n|∂H/∂k_x|u_m⟩ × ⟨u_m|∂H/∂k_y|u_n⟩
                              ————————————————————————
                                  (E_m − E_n)²
```

**The Nielson-Ninomiya theorem** requires that chiralities sum to zero: Σ_i χ_i = 0. This is a hard constraint; violation indicates a bug in node finding.

### 2.3 Enhancement: Chern Staircase Profile (already partially in `compute_chern_profile`)

The C(k_z) profile is computed as:

```
C(k_z) = (1/2π) Σ_{ix,iy} Φ_{plaquette}(k_x=ix/N_xy, k_y=iy/N_xy, k_z)
```

The enhancement needed is to **use C(k_z) jumps to locate nodes instead of gap minimisation**. Jump positions give node k_z directly, and jump signs give chirality, without requiring the coarse+refine search. This replaces the O(N₁N₂N₃) gap scan with an O(N_kz × N_kxy²) Berry flux integral, which is more robust.

```python
# Proposed: node_scan.py:scan_weyl_nodes_via_chern_staircase()
def scan_weyl_nodes_via_chern_staircase(tb_model, n_kz=40, n_kxy=30):
    profile = compute_chern_profile(tb_model, n_kz=n_kz, n_kxy=n_kxy)
    # Extract node positions from jump_kz and jump_chirality
    # For each jump at k_z*, do a 2D gap scan on the k_x-k_y plane
    # to get (k_x*, k_y*) → full 3D node position
```

### 2.4 Enhancement: Weyl Cone Velocity Tensor

Near each detected node, extract the full anisotropic velocity tensor:

```
v_{ij} = ⟨u_n(**K**_W)| ∂H/∂K_i |u_m(**K**_W)⟩   (i,j ∈ {x,y,z})
```

using the finite-difference `_dH_dk_frac` already implemented. Then:

```
v_F = (1/3) Tr[|**v**|]   (isotropic average)

Fermi arc length ≈ |**K**_{W+} − **K**_{W−}|   (separation of chirality pairs)
```

This gives a material-specific estimate of the arc k-width for use in the `n_layers_y` guard:

```
Required n_layers_y ≥ ceil( arc_k_width_Ang / (2π / (n_y · a_Ang)) )
```

---

## 3. Surface Spectral Function and Fermi Arc Detection

### 3.1 Current Implementation (`arc_scan.py`)

The surface spectral weight at energy E and surface k-point (k_x, k_y) is computed as:

```python
# For a finite slab with n_layers_z unit cells and n_orb orbitals per cell:

H_slab(k_x, k_y) = Σ_{R_x, R_y, R_z} t(R_x, R_y, R_z) · exp(2πi(k_x R_x + k_y R_y)) [block-built for each z layer]

A_surf(k_x, k_y; E) = Σ_{i ∈ surface} Σ_n |⟨i|ψ_n(k_x,k_y)⟩|² · η/π / [(E−E_n)² + η²]
```

where η = 0.06 eV is the Lorentzian broadening (line 225 of `arc_scan.py`).

### 3.2 Exact QM Basis — Surface Green's Function

The rigorous surface spectral function is obtained from the retarded Green's function:

```
G^R(E; k_∥) = [E + iη − H(**k**∥)]⁻¹

A(k_∥; E) = -(1/π) Im Tr_surf[ G^R(E; k_∥) ]
           = (η/π) Σ_n |⟨surf|n(**k**∥)⟩|² / [(E−E_n)² + η²]
```

For a semi-infinite system (true surface without hybridization), use the iterative surface Green's function (Lopez-Sancho method):

```
G_00^R(E) = [E + iη − ε₀ − Σ_s(E)]⁻¹

Σ_s(E) = t† G_{00}(E) t   (self-energy from bulk continuation)
```

where t is the inter-unit-cell hopping matrix. This is **iteration to convergence**:

```
t̃_s = (I − g̃_s t̃† − g̃_s t̃)⁻¹ g̃_s t̃²
ε̃_s = ε̃_{s-1} + t̃†_{s-1} g̃_{s-1} t̃_{s-1} + t̃_{s-1} g̃_{s-1} t̃†_{s-1}
```

(Sancho et al., J. Phys. F: Met. Phys. **15**, 851, 1985)

**Enhancement 3.2 — Lopez-Sancho Surface GF:** Replace the finite-slab diagonalisation with the iterative semi-infinite surface GF. This eliminates size-quantisation artefacts at thin d, captures the true semi-infinite surface limit, and runs in O(N_orb³ × N_iter) per k-point instead of O((N_orb × N_z)³).

```python
# Proposed: topology/surface_gf.py
def lopez_sancho_surface_gf(
    hoppings: list[tuple],   # (Rx, Ry, Rz, mat) from _collect_tb_hoppings
    n_orb: int,
    kx: float, ky: float,   # surface k-point (fractional)
    energy: float,
    eta: float = 0.06,
    max_iter: int = 200,
    conv_tol: float = 1e-8,
) -> np.ndarray:
    """Returns surface LDOS = -(1/π) Im Tr G_00^R at (kx, ky, E)."""
```

### 3.3 Arc Connectivity Metric — Rigorous Definition

The current metric combines `largest_component_fraction` and `span` via a heuristic (0.5×fraction + 0.5×span). The rigorous measure is the **arc Fermi length** in Å⁻¹:

```
L_arc = ∮_{arc} dk_∥   (integral along contour where A_surf(k_∥; E_F) > threshold)
```

This can be computed from the thresholded spectral map by:

1. Extract iso-contour at A = threshold via marching squares
2. Integrate contour length in physical k-space (not fractional)

```python
# Proposed: topology/arc_metrics.py
def fermi_arc_length_angstrom(
    spectral_map: np.ndarray,   # shape (N_kx, N_ky)
    recip_vecs_2d: np.ndarray,  # 2×2 reciprocal lattice (Å⁻¹)
    threshold_fraction: float = 0.20,
) -> float:
    """Arc length in Å⁻¹ via marching-squares contour integration."""
    from skimage.measure import find_contours
    peak = np.max(spectral_map)
    contours = find_contours(spectral_map / peak, threshold_fraction)
    total_length = 0.0
    for c in contours:
        # c in index units → convert to Å⁻¹ via recip_vecs_2d
        k_path = (c / np.array(spectral_map.shape)) @ recip_vecs_2d * 2 * np.pi
        total_length += np.sum(np.linalg.norm(np.diff(k_path, axis=0), axis=1))
    return total_length
```

---

## 4. Hybridization Gap at Sub-4nm Thickness

### 4.1 Missing Physics (Critical Gap)

This is the most significant deficiency for sub-4nm films. The current code **warns** about hybridization but does not compute Δ(d).

### 4.2 Exact QM Model

When top (T) and bottom (B) surface states hybridize through the bulk, the low-energy effective model near **K**_W is:

```
H_eff(d, **k**∥) = [ E_T(**k**∥)    Δ(d)     ]
                   [ Δ(d)*        E_B(**k**∥) ]
```

where:

```
E_{T,B}(**k**∥) = ±ℏ v_F |**k**∥ − **K**_{W,∥}|   (linearised arc dispersion)

Δ(d) = Δ₀ · exp(−d / λ_arc)   (hybridization gap)
```

The penetration depth λ_arc is determined by the bulk gap and Fermi velocity:

```
λ_arc = ℏ v_⊥ / Δ_bulk

v_⊥ = ∂E_arc/∂k_z |_{k_z=K_{W,z}}   (velocity perpendicular to surface)
```

For TaP: Δ_bulk ≈ 40 meV, v_⊥ ≈ 3 eV·Å → λ_arc ≈ 1/(Δ_bulk/v_⊥) ≈ 75 Å ≈ 6.3 uc (c = 11.89 Å).

The hybridization-induced gap is:

```
E_gap(d) = 2|Δ(d)| = 2Δ₀ · exp(−d / λ_arc)
```

**Enhancement 4.2a — Explicit Δ(d) Extraction:**

```python
# Proposed: topology/hybridization_gap.py
def compute_hybridization_gap(
    tb_model,
    thickness_range_uc: list[int],   # e.g. [2, 4, 6, 8, 10]
    kpar_node: tuple[float, float],  # surface projection of Weyl node
    n_kpar: int = 8,                 # k-points around node
) -> dict:
    """
    For each thickness d, diagonalise the finite-slab H_slab at k_∥ near K_{W,∥}.

    Find the two states with maximum surface weight (top + bottom arcs).
    Their energy splitting 2|Δ(d)| is the hybridization gap.

    Fit: log(Δ) = log(Δ₀) - d/λ_arc → extract Δ₀, λ_arc.

    Returns: {thickness_uc, gap_ev, lambda_arc_uc, Delta0_ev, fit_quality}
    """
```

**Enhancement 4.2b — Non-monotonic ρ(d) Prediction:**

The two-channel conductance model must be extended to include the gap:

```
G_arc(d) = G₀ · N_arc · T_arc(d)

T_arc(d) = 1 / [1 + (E_gap(d) / 2kT)²]    (thermal activation factor at finite T)
         = 1 / [1 + (Δ₀ e^{-d/λ_arc} / 2kT)²]
```

At T = 0: T_arc → 0 for d < d_c and T_arc → 1 for d > d_c, giving a crossover at:

```
d_c = λ_arc · ln(Δ₀ / 2kT)   (crossover thickness)
```

The full model is:

```
G(d) = G_arc(d) + G_bulk(d)

G_bulk(d) = σ_bulk · W · d / L   (bulk Drude channel, scales with d)

G_arc(d) = G_arc,0 · sech²(d_c/d − 1)   [smooth crossover approximation]
```

The resistivity:

```
ρ(d) = L / [G(d) · W]
      = L / [(G_arc,0 · f_arc(d) + σ_bulk · W · d / L) · W]
```

has a non-monotonic shape: ρ increases for d < d_c (gap suppressing arcs), passes through a maximum at d ≈ d_c, then decreases for d > d_c as arcs re-emerge, then increases again for large d as bulk dominates.

---

## 5. Transport: Landauer-Büttiker Formalism

### 5.1 Current Implementation (`conductance.py`)

```python
smat = kwant.smatrix(fsys, energy)
G = smat.transmission(0, 1)   # in units of e²/h
```

This computes:

```
G = (e²/h) · Tr[t† t]
```

where t is the N_channels × N_channels transmission matrix from lead 0 → lead 1.

### 5.2 Exact QM Basis — Landauer Formula

The Landauer-Büttiker formula at energy E is:

```
G(E) = (e²/h) Σ_{n∈L, m∈R} |t_{nm}(E)|²
      = (e²/h) Tr[t(E)† t(E)]
```

The scattering matrix is:

```
S = [ r   t' ]   with S†S = I (unitarity = current conservation)
    [ t   r' ]
```

The transmission eigenvalues τ_n ∈ [0,1] (Landauer channels) are:

```
{τ_n} = eigenvalues of t†t

G = (e²/h) Σ_n τ_n
```

**The finite-temperature conductance** (missing in current code):

```
G(T) = -(e²/h) ∫ dE · Tr[t†t](E) · ∂f(E, μ, T)/∂E

∂f/∂E = -1/(4kT) · sech²((E−μ)/(2kT))   [thermal smearing]
```

**Enhancement 5.2 — Finite-Temperature Conductance:**

```python
# Proposed extension to conductance.py
def compute_conductance_finite_T(
    fsys,
    mu: float,          # chemical potential (eV)
    T_kelvin: float,    # temperature
    E_range: np.ndarray = None,   # integration grid
    n_E: int = 100,
) -> float:
    """
    G(T) = -(e²/h) ∫ T(E) · ∂f/∂E dE

    Uses Gaussian quadrature over ±5kT around μ.
    Relevant for d < d_c where E_gap ~ kT sets the arc conductance.
    """
    kT = 8.617e-5 * T_kelvin   # eV
    if E_range is None:
        E_range = np.linspace(mu - 5*kT, mu + 5*kT, n_E)

    df_dE = -np.cosh((E_range - mu) / (2*kT))**(-2) / (4*kT)
    T_E = [kwant.smatrix(fsys, E).transmission(0, 1) for E in E_range]
    T_E = np.array(T_E)

    G = -np.trapz(T_E * df_dE, E_range)   # e²/h
    return G
```

### 5.3 Anderson Disorder Model — Current Implementation

Current model adds on-site disorder:

```
H_disorder = H_clean + Σ_i ε_i |i⟩⟨i|

ε_i ~ Uniform(-W/2, +W/2)   (Anderson disorder)
```

with surface-enhanced amplitude: W_surf = W_surface on z ∈ {z_min, z_max}±n_surface_layers.

### 5.4 Enhancement: Correlated Disorder and Interface Roughness

Real SiO₂/TaP interfaces have correlated roughness, not uncorrelated Anderson disorder. The correlation model is:

```
⟨ε_i ε_j⟩ = W² · C(|r_i − r_j|)

C(r) = exp(−r² / (2ξ²))   (Gaussian correlation, ξ = correlation length ~ 2-5 nm)
```

Implement via coloured noise: ε = F⁻¹[ F[white noise] × exp(−|K|²ξ²/2) ], where F is 2D DFT over the interface plane.

Effect on arc scattering: Fermi-arc mean free path ℓ_arc is determined by:

```
1/ℓ_arc = (2π/ℏ) ∫ |⟨**k**'|V_disorder|**k**⟩|² δ(E_arc(**k**') − E_arc(**k**)) d**k**' / (2π)²

V_q = W · C̃(q)   (disorder form factor in k-space)

C̃(q) = 2πξ² exp(−|q|²ξ²/2)   [Fourier of Gaussian correlation]
```

For ξ ≫ arc k-width: long-range disorder → strongly reduced arc scattering (smooth potential cannot backscatter). For ξ ≪ arc k-width: short-range → strong scattering, arc localization.

---

## 6. Topology Deviation Scoring (`evaluator.py`)

### 6.1 Current Formula

```
S_topo  = 1 − (1 − S_arc)(1 − δ_node)   [probabilistic union]
S_total = 0.7 × S_topo + 0.3 × S_transport
```

### 6.2 Enhanced Score — Information-Theoretic Basis

Replace the ad hoc weights with a mutual-information-based combination.

**Defect-induced topology suppression** should be modelled as:

```
P(topological | defect severity s) = Π_i P_i(topological)

P_arc(s)    = 1 − exp(−s_arc / s_arc,c)       [arc destruction at critical severity]
P_node(s)   = 1 − tanh(|δk_node| / δk_c)      [node shift normalised to BZ size]
P_transport = σ_arc(s) / σ_arc(0)              [relative arc conductance retention]
```

The composite score with proper Bayesian weighting:

```
S_topo = P_arc · P_node   [product = joint probability, assumes independence]

log S_total = w_arc · log P_arc + w_node · log P_node + w_transport · log P_transport

w_i = I_i / Σ_j I_j   [weight = mutual information between observable and topology]
```

where I_i is the Fisher information of observable i with respect to the topological phase boundary.

**Enhancement 6.2 — Topology Phase Diagram:**

The true phase boundary in (d, s) space follows:

```
Phase: topological    ↔   d > d_c(s)  AND  s < s_c(d)

d_c(s)  = λ_arc · ln(Δ₀(s) / 2kT)    [thickness crossover at defect level s]
s_c(d)  = s* · (1 − exp(−d / d_topo))  [critical defect at thickness d]
```

This phase diagram can be mapped by computing S_total on a 2D grid (d, s) and finding the contour where S_total crosses 0.5.

---

## 7. The Chern Number and Bulk-Boundary Correspondence

### 7.1 Rigorous QM Statement

For a Weyl semimetal, the bulk-boundary correspondence states:

```
N_arc(k_z) = C(k_z) = (1/2π) ∫∫_{BZ_xy} Ω_n(k_x, k_y, k_z) dk_x dk_y
```

where N_arc(k_z) is the number of Fermi arcs on the surface BZ at fixed k_z, and C(k_z) is the 2D Chern number of the 2D band structure at that k_z slice.

**Staircase function:** C(k_z) changes by ±χ each time k_z crosses a Weyl node projection:

```
C(k_z) = Σ_{i: K_{z,i} < k_z} χ_i   [sum of chiralities of crossed nodes]
```

This satisfies: C(k_z=0) = C(k_z=1) = 0 (periodicity), consistent with Σ_i χ_i = 0.

### 7.2 Wilson Loop (Wannier Charge Centre) Method

A numerically superior method to compute C(k_z) is the **Wilson loop** (already partially in codebase via Berry plaquette phases):

```
W(k_z) = P exp[ i ∮_{k_z = const} A_n(k) dk_∥ ]   [Wilson loop operator]

θ_n(k_z) = angle of eigenvalue of W(k_z)   [Wannier charge centre]
C(k_z) = winding number of {θ_n(k_z)} as k_z sweeps 0 → 1
```

**Enhancement 7.2 — Wilson Loop Implementation:**

```python
# Proposed: topology/wilson_loop.py
def compute_wilson_loop_chern(
    tb_model,
    kz_vals: np.ndarray,    # k_z grid (fractional)
    n_kxy: int = 30,        # k_x grid for Wilson loop
    band_idx: int | None = None,
) -> dict:
    """
    For each k_z, compute the Wilson loop operator by discretized product:

    W(k_z) = Π_{ix=0}^{N-1} M(k_x=ix/N, k_z)

    M_{nm}(k_x, k_z) = ⟨u_n(k_x + 1/N, k_z) | u_m(k_x, k_z)⟩

    Eigenphases θ(k_z) give Wannier charge centres.
    Chern number = winding of θ(k_z) around the cylinder.
    """
```

---

## 8. Berry Curvature and Anomalous Hall Conductivity

### 8.1 Missing Observable

The anomalous Hall conductivity σ_xy is a direct topological transport observable, related to the Berry curvature by:

```
σ_xy = (e²/ℏ) ∫_{BZ} Σ_n f_n(**k**) · Ω_n^z(**k**) · d**k** / (2π)³

Ω_n^z(**k**) = -2 Im ⟨u_n|∂H/∂k_x|u_n'⟩⟨u_n'|∂H/∂k_y|u_n⟩ / (E_n' − E_n)²
```

For a Weyl semimetal, σ_xy has a characteristic dependence on Fermi level:

```
σ_xy(E_F) = (e²/2πh) · (K_{W+,z} − K_{W−,z})   [in units of e²/hÅ]
```

when E_F is between the two Weyl node energies. This is a clean topological observable.

**Enhancement 8.1 — Berry Curvature Map:**

```python
# Proposed: topology/berry_curvature.py
def compute_berry_curvature_map(
    tb_model,
    kz_fixed: float,             # fixed k_z slice
    n_kxy: int = 40,
    occupied_bands: list[int] | None = None,
) -> dict:
    """
    Compute Ω_n^z(k_x, k_y) for bands n on the k_z = kz_fixed plane.

    Uses Kubo formula:
        Ω_n = -2 Im Σ_{m≠n} ⟨n|∂H/∂k_x|m⟩⟨m|∂H/∂k_y|n⟩ / (E_m - E_n)²

    Or equivalently, Berry plaquette formula for numerical stability.

    Returns: {kx_grid, ky_grid, Omega_z [n_kx, n_ky],
              Chern_estimate, hot_spot_kxy}
    """
```

The Berry curvature hot spots (peaked near Weyl nodes) guide the adaptive k-sampling in `adaptive_k.py` — this is more rigorous than the gap-based hotspot selection currently used.

---

## 9. Fuchs-Sondheimer Reference and Topological Contrast

### 9.1 Current Implementation

```python
def fuchs_sondheimer_rho(rho_bulk, mfp_m, thickness_m, specularity=0.0):
    kappa = thickness_m / mfp_m
    correction = (3/8) * (1 - specularity) / kappa
    return rho_bulk * (1 + correction)
```

This is the **Chambers (1950) asymptotic approximation** valid for kappa = d/ℓ ≫ 1.

### 9.2 Exact Fuchs-Sondheimer Integral

The exact result requires numerical integration:

```
ρ_FS / ρ_bulk = [ 1 − (3/2)(1−p) ∫₁^∞ (1/u³ − 1/u⁵) · exp(−κu) / (1 − p·exp(−κu)) du ]⁻¹
```

where κ = d/ℓ and p is the surface specularity (p=0: fully diffuse, p=1: specular).

**Enhancement 9.2 — Exact FS Integration:**

```python
# topology/fuchs_sondheimer.py
from scipy.integrate import quad

def fuchs_sondheimer_exact(rho_bulk, mfp_m, thickness_m, specularity=0.0):
    kappa = np.atleast_1d(thickness_m) / mfp_m
    p = float(specularity)

    def integrand(u, k):
        return (1/u**3 - 1/u**5) * np.exp(-k*u) / (1 - p * np.exp(-k*u))

    results = []
    for k in kappa:
        I, _ = quad(integrand, 1, np.inf, args=(k,), limit=500)
        rho = rho_bulk / (1 - 1.5*(1-p)*I)
        results.append(rho)
    return np.array(results)
```

### 9.3 Topological vs Normal Metal Contrast Observable

The key experimental signature is the **resistivity crossover**:

```
ρ_topo(d) < ρ_FS(d)   for d < d_crossover

Δρ(d) = ρ_FS(d) − ρ_topo(d)   [topological enhancement]

d_crossover: where dρ_topo/dd = 0   (minimum of ρ(d) curve)
```

The crossover thickness satisfies:

```
dG/dd = 0  →  σ_bulk · W/L + G_arc · ∂T_arc/∂d = 0

∂T_arc/∂d = (Δ₀/λ_arc) · e^{-d/λ_arc} · T_arc²(d) / kT
```

Solving gives d_crossover ≈ λ_arc · ln(Δ₀ · G_arc,0 / (kT · σ_bulk · W/L)).

---

## 10. Implementation Roadmap

### Priority 1 (Physics-Critical, implement first)

| Enhancement | Module | Physics Benefit |
|---|---|---|
| **4.2a** Hybridization gap Δ(d) computation | `topology/hybridization_gap.py` (new) | Quantitative arc-hybridization at d < 10 uc |
| **4.2b** Extended two-channel model with T_arc(d) | `transport/observables.py` | Correct non-monotonic ρ(d) at sub-4nm |
| **5.2** Finite-temperature Landauer | `transport/conductance.py` | Essential when E_gap ~ kT |
| **3.2** Lopez-Sancho surface GF | `topology/surface_gf.py` (new) | Eliminates finite-size artefacts in arc detection |

### Priority 2 (Topology Quality)

| Enhancement | Module | Physics Benefit |
|---|---|---|
| **7.2** Wilson loop Chern number | `topology/wilson_loop.py` (new) | More robust C(k_z) than Berry plaquette |
| **2.3** Chern staircase node finder | `topology/node_scan.py` | Replaces gap-scan with topological invariant |
| **8.1** Berry curvature map + σ_xy | `topology/berry_curvature.py` (new) | Additional topological transport observable |
| **3.3** Arc length metric (Å⁻¹) | `topology/arc_metrics.py` (new) | Physical arc length replaces heuristic score |

### Priority 3 (Refinements)

| Enhancement | Module | Physics Benefit |
|---|---|---|
| **9.2** Exact FS integration | `transport/observables.py` | Accurate normal-metal reference |
| **5.4** Correlated disorder | `transport/disorder.py` | Realistic interface roughness |
| **2.4** Velocity tensor extraction | `topology/node_scan.py` | Material-specific n_layers_y guard |
| **6.2** Phase diagram (d, s) | `topology/evaluator.py` | Topological phase boundary mapping |

---

## 11. Key Physical Parameters for TaP (Reference Values)

| Parameter | Value | Source |
|---|---|---|
| Lattice: a | 3.30 Å | DFT/experiment |
| Lattice: c | 11.89 Å | DFT/experiment |
| Weyl node W1: k_z | 0.42 × 2π/c | Liu et al. 2016 |
| Weyl node W2: k_z | 0.58 × 2π/c | Liu et al. 2016 |
| Fermi arc length | ~0.2 Å⁻¹ | ARPES |
| Arc penetration λ_arc | ~5 uc (~60 Å) | DFT slab |
| Bulk gap Δ_bulk (near nodes) | ~40 meV | DFT |
| Fermi velocity v_F | ~2–5 eV·Å | DFT |
| Arc conductance G_arc | ~12 e²/h per surface | Kwant |
| Bulk conductivity σ_bulk | ~2×10⁶ S/m | Experiment |
| Hybridization gap at d=4uc | Δ(4) ≈ Δ₀ e^{-4/5} ≈ 0.45Δ₀ | Model |
| d_crossover (T=10K) | ≈ 8–12 uc ≈ 10–14 nm | Model |

---

## 12. Algebraic Summary of the Full Model

The complete WTEC physics model, once all enhancements are implemented, reads:

**Step 1** — DFT → Wannier:
```
{ψ_n(**k**), E_n(**k**)} → U(**k**) → t_αβ(**R**) → H(**k**) = Σ_R t(**R**) e^{2πi **k**·**R**}
```

**Step 2** — Node Detection:
```
{**K**_{W,i}, χ_i, E_{W,i}} via min{|E_{n+1}(**k**) − E_n(**k**)|} + Berry flux
Verify: Σ_i χ_i = 0 (Nielson-Ninomiya)
```

**Step 3** — Hybridization Gap:
```
Δ(d) = Δ₀ e^{-d/λ_arc}   with λ_arc = ℏ v_⊥ / Δ_bulk
```

**Step 4** — Surface Spectral Function:
```
A_surf(**k**∥; E) = -(1/π) Im Tr_surf[ (E + iη − H_slab(**k**∥))⁻¹ ]   [Lopez-Sancho for d → ∞]
```

**Step 5** — Transport at finite T:
```
G(d, T) = G_arc(d, T) + G_bulk(d)

G_arc(d, T) = (e²/h) N_arc · ∫ T_arc(E) · (-∂f/∂E) dE

G_bulk(d) = σ_bulk · W · d / L

T_arc(E) = sech²((E − E_F) / E_gap(d)) / (4kT)   [arc channel transmission]
```

**Step 6** — Resistivity:
```
ρ(d, T) = L / (G(d,T) · W)
         = L / [(G_arc(d,T) + σ_bulk · W · d / L) · W]
```

**Step 7** — Topological Signature:
```
Intercept of σ_sq(d) = G(d) × (L/W) vs d:
σ_arc_2D = lim_{d→0} σ_sq(d) > 0   [topological]
                                   ≤ 0   [trivial]
```

**Step 8** — Phase Diagram:
```
T(d, s) = topological  iff  d > d_c(s)  AND  E_gap(d) < kT  AND  s < s_c(d)
```

---

*Document version 1.0 — March 2026*
*Written as enhancement blueprint for wtec v0.x → v1.0 modelling upgrade.*
*All equations reference standard condensed matter QM literature; key references: Hasan & Kane (2010), Armitage et al. (2018), Sancho et al. (1985), Landauer (1957), Fuchs (1938), Büttiker (1986).*
