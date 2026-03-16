# RGF Transport Engine - Detailed Design Specification (v3)

## Native MPI RGF Runner for PBS/qsub on g4/g3/g2/g1

**Revision log:** v3 absorbs the latest transport review and removes material-
specific assumptions from the engine design.

1. Principal-layer width is derived from the actual HR data and requested
   geometry, not from any hardcoded material expectation.
2. The MPI path is fully native C under `mpirun`; no Python, `ctypes`, or
   `mpi4py` participates in the compute path.
3. Queue selection is explicit and queue-aware, but rank count is clipped by
   measured memory fit, not blindly set to all cores.
4. `periodic_transverse` is the only production-enabled mode in phase 1.
   Full finite 3D RGF remains opt-in and preflight-gated.
5. Cluster build/deploy is owned by `wtec init`, with ISA-safe binaries and a
   manifest keyed to queue families.

---

## 0. Design Goals

The current transport implementation in `wtec.transport.conductance` uses
Kwant. That path is correct, but it scales poorly for production slab sweeps
because the hot path is a large sparse scattering solve. The RGF engine is
intended to replace Kwant only for the transport stage.

This design is intentionally **element agnostic**.

- The engine reads only the Wannier Hamiltonian (`*_hr.dat`) and lattice data.
- It does not contain per-element tables, per-material orbital counts, or
  special cases like "Ta uses P=2".
- Runtime cost and memory are derived from:
  - `n_orb` from the HR file
  - effective hopping span from the HR file
  - requested slab geometry
  - selected transport mode
  - queue capacity

The design must remain compatible with the current cluster contract:

- submission via `qsub`
- execution via `mpirun`
- no fork launchstyle
- queue families `g4`, `g3`, `g2`, `g1`

The design must also remain physically honest:

- topology stays Wannier-based elsewhere in `wtec`
- transport must not silently switch to a cheaper but different observable
- periodic reduction is only allowed when the corresponding direction is
  genuinely periodic in the modeled system

---

## 1. Scope and Non-Goals

### 1.1 In Scope

- Native C MPI executable for transport:
  - `wtec_rgf_runner`
- Direct ingestion of Wannier90 HR data and lattice vectors
- Principal-layer grouping for long-range hopping spans
- Exact clean `periodic_transverse` transport in phase 1
- Finite full-3D transport as an experimental, preflight-gated mode
- Queue-aware binary build, staging, submission, and runtime certification

### 1.2 Out of Scope

- Topology calculations
- DFT generation
- Per-element projector heuristics
- Hardcoded benchmark promises for any one material
- Silent automatic fallback from full finite transport to periodic transport
  when the physical assumptions differ

---

## 2. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ Python Orchestrator                                                │
│  ├─ resolves queue and node shape                                  │
│  ├─ parses HR metadata / geometry                                  │
│  ├─ computes exact preflight memory and safe mpi_np                │
│  ├─ stages payload + matching queue binary                         │
│  ├─ writes PBS script                                              │
│  └─ qsub -> mpirun -np N ./wtec_rgf_runner payload.json result.json│
└──────────────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────┐
│ Cluster node                                                       │
│  ├─ MPI_Init / MPI_COMM_WORLD                                      │
│  ├─ parse payload.json                                             │
│  ├─ load HR + lattice                                              │
│  ├─ compute effective principal-layer width P_eff                  │
│  ├─ validate memory fit with runtime-safe margin                   │
│  ├─ distribute work units across ranks                             │
│  ├─ run Sancho-Rubio + RGF sweep + Fisher-Lee                      │
│  ├─ MPI_Gather/Gatherv to rank 0                                   │
│  └─ rank 0 writes result.json + runtime_cert                       │
└────────────────────────────────────────────────────────────────────┘
```

The compute path contains no Python interpreter. The only Python responsibilities
are payload preparation, binary selection, PBS generation, and result retrieval.

---

## 3. Element-Agnostic Hamiltonian Ingestion

### 3.1 Core Principle

The RGF engine consumes the Wannier Hamiltonian as a pure lattice model:

- `n_orb` comes from line 2 of `*_hr.dat`
- hopping lattice vectors come from the HR entries
- lattice vectors come from the payload or the `.win` file

The engine does not know or care whether the system contains Si, Ta, P, O,
Nb, Co, or anything else. All transport sizing is inferred from the HR and
the requested geometry.

### 3.2 Required Inputs

The payload must provide:

- `hr_dat_path`
- `win_path` or `lattice_vecs`
- `lead_axis`
- slab geometry:
  - `n_layers_x`
  - one or both transverse dimensions depending on mode
- transport mode

Optional physics knobs remain generic:

- `energy`
- `eta`
- disorder amplitudes
- ensemble count

### 3.3 Effective Principal-Layer Width

The engine computes:

`P_eff = principal-layer width required by the actual HR after geometry/mode filtering`

This is not a material constant. It must be recomputed from the actual system
that will be solved.

For any mode:

1. Filter hoppings that cannot occur in the requested finite geometry.
2. Apply any exact mode reduction first.
   - Example: for `periodic_transverse`, Fourier-reduce the periodic axis
     before measuring span.
3. Compute:
   - `P_eff = max(1, max(abs(R_lead)))`

That makes the engine robust to:

- different elements
- different Wannier windows
- different localization quality
- different slab widths

### 3.4 Pseudocode

```c
int compute_principal_layer_width(
    const wtec_rgf_model_t *m,
    int lead_axis,
    const wtec_rgf_geometry_t *geom,
    int mode
) {
    int max_span = 0;
    for (int ih = 0; ih < m->n_hop; ++ih) {
        wtec_hop_t h = m->hop[ih];
        if (!wtec_hop_survives_geometry(h.R, lead_axis, geom, mode))
            continue;
        if (mode == WTEC_RGF_MODE_PERIODIC_TRANSVERSE) {
            if (!wtec_hop_survives_fourier_reduction(h.R, lead_axis, geom))
                continue;
        }
        int span = abs(h.R[lead_axis]);
        if (span > max_span)
            max_span = span;
    }
    return max_span < 1 ? 1 : max_span;
}
```

### 3.5 Consequence for the Package

The package must not ship tables like:

- "material X usually has P=2"
- "SOC model Y always fits on g4"

Instead:

- `wtec init` verifies the binary/router
- `wtec run` parses the actual HR and geometry
- preflight decides what is safe for that specific job

---

## 4. Transport Modes

### 4.1 Mode A: `periodic_transverse` (Production, Phase 1)

This is the only production-enabled RGF mode in phase 1.

Definition:

- transport axis is finite
- one transverse axis remains exactly periodic
- the remaining transverse axis is finite

In the current `wtec` slab geometry, the first concrete implementation is:

- `lead_axis = x`
- periodic transverse axis = `y`
- finite thickness axis = `z`

The internal design, however, should keep the naming generic so future
variants like periodic `z` are not blocked by the API.

This mode is physically exact only when the periodic transverse axis is not
broken by the modeled disorder or geometry.

Allowed:

- clean ordered slabs
- ordered interfaces/superlattices that remain periodic along the reduced axis

Forbidden:

- random disorder along the periodic axis
- vacancy patterns or substitutions that break the periodic axis

### 4.2 Mode B: `full_finite` (Experimental, Phase 2+)

This mode keeps all transverse directions finite. It is only allowed when:

- the exact preflight memory model says it fits
- the requested `mpi_np` is clipped to a safe value
- the user explicitly opts into the mode

It is not a universal default because long-range HR models can produce large
principal layers and very large dense superslice blocks.

### 4.3 No Silent Physics Fallback

If a requested full finite problem does not fit the node:

- the engine must fail clearly, or
- the orchestrator may suggest `periodic_transverse`

But it must not silently run `periodic_transverse` when the physics setup does
not justify that reduction.

---

## 5. Queue-Aware Execution Model

### 5.1 Queue Inputs

The current cluster interface already defines per-queue core counts:

- `g4`
- `g3`
- `g2`
- `g1`

The RGF engine must consume those through the same queue-resolution path used
elsewhere in `wtec`, not through a separate hardcoded map.

### 5.2 Rank and Thread Policy

Default RGF parallelism is:

- many MPI ranks
- one BLAS/OpenMP thread per rank

That means:

- `OMP_NUM_THREADS = 1`
- `MKL_NUM_THREADS = 1`
- `OPENBLAS_NUM_THREADS = 1`
- `MKL_THREADING_LAYER = sequential`

But `mpi_np` is **not** simply "all cores".

It must be:

`mpi_np = min(queue_cores, n_work_units, max_safe_ranks_from_memory)`

Where:

- `queue_cores` is from the resolved queue
- `n_work_units` depends on mode
- `max_safe_ranks_from_memory` comes from exact preflight

### 5.3 Work Decomposition

For `periodic_transverse`, the rank-level work unit should be:

- `(thickness, disorder_seed, k_index)` for clean periodic jobs

That matters because one clean point with many `k_perp` sectors should still
use the full node. A sequential loop over all `k` sectors within each rank
wastes the node.

For `full_finite`, the rank-level work unit should be:

- `(thickness, disorder_seed, ensemble_index)`

### 5.4 PBS Contract

The runner must always stay compatible with the package submission policy:

- `qsub`
- `mpirun`
- no fork launchstyle

Typical resource shape:

```
#PBS -l select=1:ncpus=<queue_cores>:mpiprocs=<mpi_np>:ompthreads=1
mpirun -np <mpi_np> ./wtec_rgf_runner payload.json result.json
```

The exact `mpi_np` may be smaller than `queue_cores`. The node is still
allocated as a whole if that is the queue model, but the executable must not
pretend it can safely use all cores when memory says otherwise.

---

## 6. Memory Model

### 6.1 Exact Block Size

For a given mode:

`N_super = P_eff * N_cross * n_orb`

Where:

- `P_eff` is computed from the actual HR and geometry
- `n_orb` is read from the HR file
- `N_cross` depends on mode:
  - `periodic_transverse`: finite cross-section only
  - `full_finite`: product of all finite transverse directions

Examples:

- `periodic_transverse`: `N_cross = n_finite_transverse`
- `full_finite`: `N_cross = n_b * n_c`

The document intentionally does not bind those symbols to a specific element or
orbital count.

### 6.2 Workspace Formula

The dominant workspace remains dense complex matrices:

`workspace_bytes ~= N_mats * N_super^2 * 16 + overhead`

Where:

- `N_mats` should be measured from the final workspace struct
- the preflight formula must use the same number as the implementation
- overhead includes pivots, MPI buffers, and a safety margin

### 6.3 Required Behavior

Memory sizing must happen in three places:

1. `wtec run` preflight in Python
2. runner startup before PBS submission finalization
3. C executable startup before any large allocation

All three must use the same formula.

### 6.4 Safe Rank Clipping

If:

`per_rank_bytes * requested_mpi_np > safe_node_budget_bytes`

Then the orchestrator must either:

- clip `mpi_np` to `max_safe_ranks`, or
- fail before submission

The choice is mode-dependent:

- `periodic_transverse`: clipping is acceptable
- `full_finite`: clipping is acceptable only if enough work remains to justify
  the run and the user has not requested strict queue saturation

### 6.5 No Material-Specific Memory Tables

The previous version used example tables with fixed orbital counts and fixed
principal-layer widths. That is not robust enough for an element-agnostic
engine.

This version instead requires:

- exact preflight on the actual HR
- runtime certification that records:
  - `n_orb`
  - `P_eff`
  - `N_super`
  - `per_rank_bytes`
  - `mpi_np_requested`
  - `mpi_np_used`
  - `node_ram_bytes`
  - `safe_rank_cap`

### 6.6 Python Preflight Skeleton

```python
def _rgf_preflight(self, payload, queue):
    queue_cores = self.cluster_cfg.cores_for_queue(queue)
    hr_info = self._parse_hr_metadata(payload["hr_dat_path"], payload["win_path"])
    mode = payload["rgf_mode"]
    p_eff = self._compute_effective_principal_layer_width(hr_info, payload, mode)
    n_super = self._compute_n_super(hr_info.n_orb, p_eff, payload, mode)
    per_rank = self._rgf_memory_per_rank_bytes(n_super)
    node_ram = self._node_ram_for_queue_or_probe(queue)
    safe_cap = max(1, int(0.85 * node_ram // per_rank))
    work_units = self._rgf_work_unit_count(payload, mode)
    mpi_np = min(queue_cores, work_units, safe_cap)
    if mpi_np < 1:
        raise RuntimeError("RGF job does not fit on the selected queue.")
    return {
        "p_eff": p_eff,
        "n_super": n_super,
        "per_rank_bytes": per_rank,
        "safe_rank_cap": safe_cap,
        "mpi_np": mpi_np,
    }
```

---

## 7. Core Algorithms

### 7.1 Principal-Layer Construction

Group `P_eff` consecutive unit cells along the lead axis into one superslice.
Because `P_eff` is the maximum surviving lead-axis span after mode reduction,
the resulting Hamiltonian is block tridiagonal at the superslice level.

### 7.2 Lead Self-Energies

Use Sancho-Rubio / Lopez-Sancho on superslice blocks:

- `H_00`
- `H_01`

The lead solver operates on superslice-sized blocks, not on single-cell blocks.

### 7.3 Forward/Backward Sweep

Standard RGF sweep applies to superslices once the Hamiltonian has been
reduced to block tridiagonal form.

### 7.4 Fisher-Lee Transmission

Transmission is computed from:

- `Sigma_L`
- `Sigma_R`
- `Gamma_L`
- `Gamma_R`
- end-to-end retarded Green function block

### 7.5 Periodic Transverse Integration

For `periodic_transverse`, the reduced axis is Fourier transformed and the
conductance is accumulated over the sampled `k_perp` grid:

`G_total(E) = sum_k T(E, k_perp)`

In phase 1, these `k_perp` sectors are MPI-distributed work items. They are
not looped serially inside a single rank unless the job is too small to justify
more ranks.

---

## 8. Cluster Build and Router Ownership

### 8.1 `wtec init` Owns the Router

`wtec init` must prepare the RGF router end-to-end:

1. detect cluster compiler / MPI / BLAS environment
2. build or verify the queue-safe binary
3. write a build manifest
4. record availability in `.wtec/init_state.json`

The user should not need a separate manual "router setup" step.

### 8.2 ISA Policy

The binary must be portable across the queues it targets.

That means:

- do not default to `-march=native`
- do not assume AVX-512 everywhere

Two supported policies:

1. Conservative shared binary
   - baseline ISA chosen to run on all target queues
2. Queue-family binaries
   - one binary per queue family or ISA family
   - manifest maps `g4/g3/g2/g1` to the correct binary

The default should be the safer option. Queue-family tuning is opt-in.

### 8.3 Build Manifest

`wtec init` should produce something like:

```json
{
  "rgf": {
    "enabled": true,
    "binaries": {
      "generic": {
        "path": "/home/msj/.../wtec_rgf_runner",
        "isa": "x86-64-v2",
        "queues": ["g1", "g2", "g3", "g4"]
      }
    },
    "build_env": {
      "mpicc": "mpicc",
      "blas": "mkl-sequential"
    }
  }
}
```

### 8.4 No Python in the MPI Path

The final PBS command remains:

`mpirun -np <mpi_np> ./wtec_rgf_runner payload.json result.json`

No Python wrapper should sit between `mpirun` and the executable.

---

## 9. Payload and Result Schema

### 9.1 Payload Additions

The transport payload should include:

```json
{
  "engine": "rgf",
  "rgf_mode": "periodic_transverse",
  "periodic_axis": "y",
  "lead_axis": "x",
  "thickness_axis": "z",
  "hr_dat_path": "reference_hr.dat",
  "win_path": "reference.win",
  "thicknesses": [4, 6, 8, 10],
  "disorder_strengths": [0.0],
  "n_ensemble": 1,
  "energy": 0.0,
  "eta": 1e-6,
  "expected_mpi_np": 48,
  "expected_threads": 1
}
```

### 9.2 Result and Runtime Certificate

The result must include a runtime certificate that makes the sizing explicit:

```json
{
  "runtime_cert": {
    "engine": "rgf",
    "queue": "g3",
    "mode": "periodic_transverse",
    "periodic_axis": "y",
    "mpi_size": 48,
    "threads": 1,
    "n_orb": 32,
    "principal_layer_width": 7,
    "n_super": 2240,
    "per_rank_bytes": 123456789,
    "safe_rank_cap": 48,
    "binary_isa": "x86-64-v2"
  }
}
```

The values above are examples only. The engine must report actual measured
values for the current run.

---

## 10. Validation Strategy

### 10.1 Unit and Numerical Tests

Required low-level tests:

- 1D chain with analytic conductance
- principal-layer invariance:
  - same Hamiltonian solved with minimal valid `P_eff`
  - same answer as an explicitly expanded nearest-neighbor representation
- Sancho-Rubio convergence
- MPI gather correctness
- disorder statistics reproducibility

### 10.2 Cross-Validation vs Current Kwant Path

Cross-validation must stay element agnostic.

Required reference set:

1. toy nearest-neighbor chain
2. toy multi-orbital SOC model
3. one representative non-SOC real HR
4. one representative SOC real HR

For each case:

- same geometry
- same energy / eta / disorder configuration
- compare `G_rgf` vs `G_kwant`

Acceptance:

- clean systems: relative error `< 1e-4`
- disorder-averaged systems: mean and standard deviation within agreed
  statistical tolerance

### 10.3 Physics Checks

- reciprocity
- unitarity bounds
- ballistic invariance with length in the clean limit
- diffusive trend in the disordered finite case
- invariance with respect to valid principal-layer regrouping

---

## 11. Performance Targets

This document no longer claims fixed wall times for any one material. Those
numbers were too dependent on orbital count, HR range, and geometry.

Instead, performance targets are benchmark-relative.

### 11.1 Phase-1 Target: `periodic_transverse`

For clean ordered cases that satisfy the periodic-axis assumption:

- must preserve conductance within validation tolerance vs Kwant
- should materially outperform the current Kwant baseline on the same queue
- should maintain full-node utilization by distributing `k_perp` sectors across
  MPI ranks when enough sectors exist

Suggested acceptance target:

- `>= 5x` wall-time speedup vs current Kwant baseline on the same reference case

### 11.2 Phase-2 Target: `full_finite`

For full finite transport:

- correctness is the primary goal
- performance targets are secondary until exact memory-fit behavior has been
  validated on real HR models

---

## 12. Implementation Phases

### Phase 1: Production `periodic_transverse`

1. scaffold native C project
2. HR reader and lattice parser
3. exact `P_eff` computation from actual HR and geometry
4. superslice Hamiltonian builder
5. Sancho-Rubio lead solver
6. RGF sweep + Fisher-Lee
7. MPI work distribution over `k_perp` sectors and outer work units
8. `wtec init` build/stage/manifest integration
9. queue-aware preflight and runtime certification
10. cross-validation vs Kwant

Gate:

- queue-safe build works from `wtec init`
- `periodic_transverse` passes correctness tests
- same-mode benchmark shows real speedup

### Phase 2: Experimental `full_finite`

1. enable full finite mode behind explicit config
2. verify memory model against real HR cases
3. implement safe rank clipping and strict-fit policy
4. benchmark correctness and viability on representative finite cases

Gate:

- no OOM or unsafe oversubscription
- correctness vs Kwant established on real HR cases

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Long-range HR produces large `P_eff` | Compute `P_eff` from actual HR and geometry every run; never assume a material default |
| Full finite mode exceeds node memory | Mandatory preflight, rank clipping, or hard fail before qsub |
| Mixed CPU ISA across queues | Conservative shared binary by default, queue-family binaries as an opt-in optimization |
| Small clean jobs underuse the node | Distribute `k_perp` sectors across ranks in `periodic_transverse` mode |
| Physics mismatch from invalid periodic reduction | Enforce mode validity from geometry/disorder metadata; no silent fallback |
| Divergence from current transport semantics | Keep payload/result schema aligned with existing transport worker contract |

---

## 14. Public C API Summary

```c
int  wtec_rgf_model_from_hr_dat(const char *hr_path,
                                const char *win_path,
                                wtec_rgf_model_t *out);
void wtec_rgf_model_free(wtec_rgf_model_t *m);

int  wtec_rgf_compute_principal_layer_width(const wtec_rgf_model_t *m,
                                            const wtec_rgf_geometry_t *g,
                                            int mode,
                                            int *p_out);

size_t wtec_rgf_memory_per_rank_bytes(int n_super, int workspace_mats);

int  wtec_rgf_run_periodic_transverse(const wtec_rgf_model_t *m,
                                      const wtec_rgf_payload_t *p,
                                      MPI_Comm comm,
                                      wtec_rgf_result_t *out);

int  wtec_rgf_run_full_finite(const wtec_rgf_model_t *m,
                              const wtec_rgf_payload_t *p,
                              MPI_Comm comm,
                              wtec_rgf_result_t *out);
```

---

## 15. Final Design Decision

The RGF engine is viable for this cluster environment only under the following
package rules:

1. `wtec init` owns build, staging, and queue/binary routing.
2. The engine remains element agnostic by deriving all sizing from the HR and
   geometry, not from material labels.
3. Phase 1 production transport is `periodic_transverse`, not universal
   full-3D finite transport.
4. Queue core count does not imply safe rank count; memory-fit decides the
   usable `mpi_np`.
5. The MPI path stays native C under `mpirun`, with no fork launchstyle and no
   Python participation in the compute path.
